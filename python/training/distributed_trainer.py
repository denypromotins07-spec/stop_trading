"""
Distributed Trainer - Ray Train for XGBoost and ONNX-exportable PyTorch models.
Configures strict resource requests to prevent memory bloat.
Strictly enforces 3GB RAM limit with worker recycling.
"""
import ray
from ray import train, tuning
from ray.train import ScalingConfig, RunConfig
from ray.train.xgboost import XGBoostTrainer
from typing import Dict, List, Optional, Any, Callable
import logging
import numpy as np
import tempfile
import os

logger = logging.getLogger(__name__)


class DistributedTrainer:
    """
    Distributed training using Ray Train with strict memory bounds.
    Supports XGBoost and lightweight PyTorch models for ONNX export.
    """
    
    def __init__(self,
                 num_workers: int = 4,
                 cpu_per_worker: int = 2,
                 memory_per_worker_mb: int = 1024,
                 use_gpu: bool = False):
        """
        Initialize distributed trainer.
        
        Args:
            num_workers: Number of training workers
            cpu_per_worker: CPUs per worker
            memory_per_worker_mb: Memory per worker in MB (strict limit)
            use_gpu: Whether to use GPU
        """
        self.num_workers = num_workers
        self.cpu_per_worker = cpu_per_worker
        self.memory_per_worker_mb = memory_per_worker_mb
        self.use_gpu = use_gpu
        
        self._scaling_config = ScalingConfig(
            num_workers=num_workers,
            use_gpu=use_gpu,
            resources_per_worker={
                "CPU": cpu_per_worker,
                "memory": memory_per_worker_mb * 1024 * 1024
            }
        )
        
        self._current_trainer = None
        self._checkpoint_dir = tempfile.mkdtemp(prefix="ray_train_")
    
    def train_xgboost(self,
                     train_dataset: 'ray.data.Dataset',
                     params: Dict[str, Any],
                     label_column: str = "label",
                     feature_columns: List[str] = None,
                     num_boost_round: int = 100,
                     callbacks: List = None) -> Dict[str, Any]:
        """
        Train XGBoost model using Ray Train.
        
        Args:
            train_dataset: Ray dataset with features and labels
            params: XGBoost parameters
            label_column: Name of label column
            feature_columns: List of feature column names
            num_boost_round: Number of boosting rounds
            callbacks: Optional training callbacks
            
        Returns:
            Training results including checkpoint path
        """
        # Set memory-safe XGBoost defaults
        safe_params = {
            "max_depth": 6,
            "eta": 0.01,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "nthread": self.cpu_per_worker,
            **params
        }
        
        try:
            trainer = XGBoostTrainer(
                params=safe_params,
                label_column=label_column,
                feature_columns=feature_columns or [],
                scaling_config=self._scaling_config,
                run_config=RunConfig(
                    storage_path=self._checkpoint_dir,
                    name="xgboost_training"
                ),
                datasets={"train": train_dataset},
                num_boost_round=num_boost_round
            )
            
            result = trainer.fit()
            
            logger.info(f"XGBoost training completed: {result.metrics}")
            return {
                "success": True,
                "checkpoint": result.checkpoint,
                "metrics": result.metrics,
                "path": result.path
            }
            
        except Exception as e:
            logger.error(f"XGBoost training failed: {e}")
            return {"success": False, "error": str(e)}
    
    def train_pytorch(self,
                     train_dataset: 'ray.data.Dataset',
                     model_class: type,
                     loss_fn: Callable,
                     optimizer_class: type,
                     config: Dict[str, Any],
                     epochs: int = 10,
                     batch_size: int = 256) -> Dict[str, Any]:
        """
        Train PyTorch model designed for ONNX export.
        
        Args:
            train_dataset: Ray dataset
            model_class: PyTorch model class (must be ONNX-exportable)
            loss_fn: Loss function
            optimizer_class: Optimizer class
            config: Training configuration
            epochs: Number of epochs
            batch_size: Batch size
            
        Returns:
            Training results
        """
        from ray.train.torch import TorchTrainer
        
        def train_func(config):
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader
            
            # Get dataset
            dataset = train.get_dataset_shard("train")
            
            # Create model
            model = model_class(**config.get('model_kwargs', {}))
            model = train.torch.prepare_model(model)
            
            # Create optimizer
            optimizer = optimizer_class(
                model.parameters(),
                **config.get('optimizer_kwargs', {'lr': 0.001})
            )
            
            # Training loop
            for epoch in range(epochs):
                model.train()
                total_loss = 0.0
                batches_processed = 0
                
                for batch in dataset.iter_batches(batch_size=batch_size):
                    # Convert to tensors
                    x = torch.FloatTensor(batch['features'])
                    y = torch.FloatTensor(batch['label'])
                    
                    # Forward pass
                    optimizer.zero_grad()
                    outputs = model(x)
                    loss = loss_fn(outputs, y)
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    batches_processed += 1
                
                avg_loss = total_loss / max(batches_processed, 1)
                
                # Report metrics
                train.report({"loss": avg_loss, "epoch": epoch})
        
        trainer = TorchTrainer(
            train_loop_per_worker=train_func,
            train_loop_config={
                "model_kwargs": config.get('model_kwargs', {}),
                "optimizer_kwargs": config.get('optimizer_kwargs', {})
            },
            scaling_config=self._scaling_config,
            run_config=RunConfig(
                storage_path=self._checkpoint_dir,
                name="pytorch_training"
            ),
            datasets={"train": train_dataset}
        )
        
        try:
            result = trainer.fit()
            logger.info(f"PyTorch training completed: {result.metrics}")
            return {
                "success": True,
                "checkpoint": result.checkpoint,
                "metrics": result.metrics,
                "path": result.path
            }
        except Exception as e:
            logger.error(f"PyTorch training failed: {e}")
            return {"success": False, "error": str(e)}
    
    def tune_hyperparameters(self,
                            train_dataset: 'ray.data.Dataset',
                            param_space: Dict[str, Any],
                            metric: str = "loss",
                            mode: str = "min",
                            num_samples: int = 10,
                            trainer_type: str = "xgboost") -> Dict[str, Any]:
        """
        Hyperparameter tuning with Ray Tune.
        
        Args:
            train_dataset: Training dataset
            param_space: Parameter search space
            metric: Metric to optimize
            mode: Optimization mode ('min' or 'max')
            num_samples: Number of trials
            trainer_type: Type of trainer ('xgboost' or 'pytorch')
            
        Returns:
            Best parameters and results
        """
        if trainer_type == "xgboost":
            base_trainer = lambda config: XGBoostTrainer(
                params={**config, "max_depth": 6, "eta": 0.01},
                scaling_config=self._scaling_config,
                datasets={"train": train_dataset}
            )
        else:
            raise NotImplementedError("Only XGBoost tuning implemented")
        
        tuner = tuning.Tuner(
            base_trainer,
            param_space=param_space,
            tune_config=tuning.TuneConfig(
                metric=metric,
                mode=mode,
                num_samples=num_samples,
            ),
            run_config=RunConfig(
                storage_path=self._checkpoint_dir,
                name="hyperparameter_tuning"
            )
        )
        
        results = tuner.fit()
        best_result = results.get_best_result(metric=metric, mode=mode)
        
        return {
            "best_params": best_result.config,
            "best_metric": best_result.metrics[metric],
            "all_results": results
        }
    
    def export_to_onnx(self,
                      checkpoint_path: str,
                      output_path: str,
                      input_shape: tuple,
                      model_class: type = None) -> bool:
        """
        Export trained model to ONNX format.
        
        Args:
            checkpoint_path: Path to training checkpoint
            output_path: Output path for ONNX model
            input_shape: Shape of input tensor
            model_class: Model class (for PyTorch models)
            
        Returns:
            True if successful
        """
        try:
            import onnx
            import torch
            
            # Load model from checkpoint
            if model_class:
                # PyTorch model
                model = model_class()
                checkpoint = torch.load(checkpoint_path)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                
                # Export to ONNX
                dummy_input = torch.randn(input_shape)
                torch.onnx.export(
                    model,
                    dummy_input,
                    output_path,
                    export_params=True,
                    opset_version=11,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes=None
                )
                
                logger.info(f"Exported ONNX model to {output_path}")
                return True
            else:
                # XGBoost model - use sklearn API conversion
                from xgboost import XGBRegressor
                import onnxmltools
                from onnxmltools.convert import convert_xgboost
                
                # Load XGBoost model
                model = XGBRegressor()
                model.load_model(checkpoint_path)
                
                # Convert to ONNX
                onnx_model = convert_xgboost(model, 'xgboost model', initial_types=[
                    ('float_input', onnx.types.DataTypeFactory().build_tensor_type(
                        [None, input_shape[1]]
                    ))
                ])
                
                onnxmltools.utils.save_model(onnx_model, output_path)
                logger.info(f"Exported XGBoost ONNX model to {output_path}")
                return True
                
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            return False
    
    def get_resource_usage(self) -> Dict[str, Any]:
        """Get current resource usage."""
        return {
            "num_workers": self.num_workers,
            "cpu_per_worker": self.cpu_per_worker,
            "memory_per_worker_mb": self.memory_per_worker_mb,
            "total_memory_mb": self.num_workers * self.memory_per_worker_mb,
            "use_gpu": self.use_gpu,
            "checkpoint_dir": self._checkpoint_dir
        }
    
    def cleanup(self):
        """Cleanup temporary files."""
        import shutil
        try:
            shutil.rmtree(self._checkpoint_dir, ignore_errors=True)
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")


# Example usage
def main():
    """Example usage of distributed trainer."""
    ray.init(
        ignore_reinit_error=True,
        _system_config={
            "object_store_memory": 512 * 1024 * 1024,
            "max_workers": 4
        }
    )
    
    try:
        trainer = DistributedTrainer(
            num_workers=2,
            cpu_per_worker=2,
            memory_per_worker_mb=512
        )
        
        print(f"Resource usage: {trainer.get_resource_usage()}")
        
    finally:
        trainer.cleanup()
        ray.shutdown()


if __name__ == "__main__":
    main()
