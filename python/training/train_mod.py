"""
Training Module Root - Manages training cluster resources, checkpointing, and ONNX compilation.
Coordinates Ray Data pipeline and distributed trainer.
Strictly enforces 3GB RAM limit.
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import tempfile
import shutil

from training.ray_data_pipeline import RayDataPipeline, create_pipeline
from training.distributed_trainer import DistributedTrainer


logger = logging.getLogger(__name__)


class TrainingManager:
    """
    Central manager for all training operations.
    Coordinates data loading, distributed training, and model export.
    """
    
    def __init__(self,
                 max_memory_mb: int = 1024,
                 num_workers: int = 4,
                 checkpoint_dir: str = None):
        """
        Initialize training manager.
        
        Args:
            max_memory_mb: Maximum memory budget in MB
            num_workers: Number of training workers
            checkpoint_dir: Directory for checkpoints
        """
        self.max_memory_mb = max_memory_mb
        self.num_workers = num_workers
        self.checkpoint_dir = checkpoint_dir or tempfile.mkdtemp(prefix="training_")
        
        self._pipeline: Optional[RayDataPipeline] = None
        self._trainer: Optional[DistributedTrainer] = None
        self._current_job: Optional[Dict] = None
        
        # Ensure checkpoint directory exists
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    def create_pipeline(self, batch_size: int = 512) -> RayDataPipeline:
        """Create a new data pipeline."""
        self._pipeline = create_pipeline(
            batch_size=batch_size,
            max_memory_mb=self.max_memory_mb // 2
        )
        return self._pipeline
    
    def create_trainer(self,
                      num_workers: int = None,
                      memory_per_worker_mb: int = 512) -> DistributedTrainer:
        """Create a new distributed trainer."""
        self._trainer = DistributedTrainer(
            num_workers=num_workers or self.num_workers,
            cpu_per_worker=2,
            memory_per_worker_mb=memory_per_worker_mb
        )
        return self._trainer
    
    async def run_training_job(self,
                              data_paths: List[str],
                              model_type: str = "xgboost",
                              params: Dict[str, Any] = None,
                              feature_columns: List[str] = None,
                              label_column: str = "label") -> Dict[str, Any]:
        """
        Run complete training job from data loading to model export.
        
        Args:
            data_paths: Paths to training data
            model_type: Type of model ('xgboost' or 'pytorch')
            params: Model parameters
            feature_columns: Feature column names
            label_column: Label column name
            
        Returns:
            Training results
        """
        results = {
            "success": False,
            "stages": {},
            "model_path": None
        }
        
        try:
            # Stage 1: Load data
            logger.info("Stage 1: Loading data...")
            if not self._pipeline:
                self.create_pipeline()
            
            self._pipeline.read_parquet_streaming(
                paths=data_paths,
                columns=feature_columns + [label_column] if feature_columns else None
            )
            results["stages"]["data_load"] = self._pipeline.get_stats()
            
            # Stage 2: Train model
            logger.info("Stage 2: Training model...")
            if not self._trainer:
                self.create_trainer()
            
            if model_type == "xgboost":
                train_result = self._trainer.train_xgboost(
                    train_dataset=self._pipeline._dataset,
                    params=params or {},
                    label_column=label_column,
                    feature_columns=feature_columns or []
                )
            else:
                raise NotImplementedError(f"Model type {model_type} not implemented")
            
            results["stages"]["training"] = train_result
            results["success"] = train_result.get("success", False)
            
            # Stage 3: Export to ONNX
            if results["success"]:
                logger.info("Stage 3: Exporting to ONNX...")
                onnx_path = Path(self.checkpoint_dir) / "model.onnx"
                
                exported = self._trainer.export_to_onnx(
                    checkpoint_path=train_result.get("path", ""),
                    output_path=str(onnx_path),
                    input_shape=(1, len(feature_columns)) if feature_columns else (1, 100)
                )
                
                results["stages"]["export"] = {"success": exported, "path": str(onnx_path)}
                if exported:
                    results["model_path"] = str(onnx_path)
            
        except Exception as e:
            logger.error(f"Training job failed: {e}")
            results["error"] = str(e)
        
        self._current_job = results
        return results
    
    def get_checkpoint(self, job_id: str = None) -> Optional[str]:
        """Get checkpoint path for a job."""
        if job_id:
            return str(Path(self.checkpoint_dir) / job_id / "checkpoint")
        return self.checkpoint_dir
    
    def list_checkpoints(self) -> List[str]:
        """List all available checkpoints."""
        checkpoints = []
        for path in Path(self.checkpoint_dir).glob("**/*.onnx"):
            checkpoints.append(str(path))
        return checkpoints
    
    def cleanup_old_checkpoints(self, max_age_days: int = 7):
        """Remove checkpoints older than specified age."""
        import time
        cutoff = time.time() - (max_age_days * 86400)
        
        for path in Path(self.checkpoint_dir).glob("**/*"):
            if path.is_file() and path.stat().st_mtime < cutoff:
                try:
                    path.unlink()
                    logger.info(f"Removed old checkpoint: {path}")
                except Exception as e:
                    logger.warning(f"Failed to remove {path}: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get training manager statistics."""
        stats = {
            "checkpoint_dir": self.checkpoint_dir,
            "max_memory_mb": self.max_memory_mb,
            "num_workers": self.num_workers,
            "checkpoints": self.list_checkpoints(),
            "current_job": self._current_job
        }
        
        if self._pipeline:
            stats["pipeline"] = self._pipeline.get_stats()
        
        if self._trainer:
            stats["trainer"] = self._trainer.get_resource_usage()
        
        return stats
    
    def shutdown(self):
        """Shutdown and cleanup resources."""
        if self._pipeline:
            self._pipeline.cleanup()
        if self._trainer:
            self._trainer.cleanup()
        
        # Optionally remove temp directory
        # shutil.rmtree(self.checkpoint_dir, ignore_errors=True)


# Module-level singleton
_training_manager: Optional[TrainingManager] = None


def get_manager() -> TrainingManager:
    """Get or create training manager singleton."""
    global _training_manager
    if _training_manager is None:
        _training_manager = TrainingManager()
    return _training_manager


def initialize_training(max_memory_mb: int = 1024,
                       num_workers: int = 4) -> TrainingManager:
    """Initialize training system."""
    global _training_manager
    _training_manager = TrainingManager(
        max_memory_mb=max_memory_mb,
        num_workers=num_workers
    )
    return _training_manager


async def train_model(data_paths: List[str],
                     model_type: str = "xgboost",
                     params: Dict[str, Any] = None) -> Dict[str, Any]:
    """Train model via singleton."""
    manager = get_manager()
    return await manager.run_training_job(
        data_paths=data_paths,
        model_type=model_type,
        params=params
    )


def get_training_stats() -> Dict[str, Any]:
    """Get training stats via singleton."""
    manager = get_manager()
    return manager.get_stats()


# Example usage
async def main():
    """Example usage of training module."""
    logging.basicConfig(level=logging.INFO)
    
    manager = initialize_training(max_memory_mb=512, num_workers=2)
    
    print(f"Training stats: {get_training_stats()}")
    
    # Example training job (would need actual data)
    # results = await train_model(
    #     data_paths=["data/train/*.parquet"],
    #     model_type="xgboost",
    #     params={"max_depth": 6, "eta": 0.01}
    # )
    # print(f"Training results: {results}")
    
    manager.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
