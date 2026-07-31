"""
ONNX Runtime Inference Wrapper for Ultra-Fast Deep Learning Inference
GIL-free inference using onnxruntime with CPU execution providers.
Strictly bounds session's internal thread pool to prevent CPU oversubscription.
"""

import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import threading
import time
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ONNXConfig:
    """Configuration for ONNX inference session."""
    n_threads: int = 4  # Strict thread limit
    inter_op_num_threads: int = 1
    intra_op_num_threads: int = 4
    execution_provider: str = "CPU"  # or "CUDA", "TensorRT"
    optimize_model: bool = True
    enable_memory_arena: bool = True
    max_batch_size: int = 256


class ONNXInferenceSession:
    """
    Ultra-fast ONNX runtime inference wrapper.
    Designed for GIL-free, low-latency deep learning inference.
    """
    
    def __init__(self, 
                 model_path: str,
                 config: Optional[ONNXConfig] = None):
        self.model_path = model_path
        self.config = config or ONNXConfig()
        
        self._session = None
        self._input_names: List[str] = []
        self._output_names: List[str] = []
        self._input_shapes: List[Tuple] = []
        self._lock = threading.Lock()
        
        # Set thread limits before loading
        self._set_thread_limits()
        
        # Load model
        self._load_session()
    
    def _set_thread_limits(self) -> None:
        """Set environment variables for thread control."""
        os.environ["OMP_NUM_THREADS"] = str(self.config.n_threads)
        os.environ["MKL_NUM_THREADS"] = str(self.config.n_threads)
        os.environ["NUMEXPR_NUM_THREADS"] = str(self.config.n_threads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(self.config.n_threads)
    
    def _load_session(self) -> None:
        """Load ONNX model into inference session."""
        try:
            import onnxruntime as ort
            
            # Configure session options
            sess_options = ort.SessionOptions()
            
            # Thread configuration
            sess_options.inter_op_num_threads = self.config.inter_op_num_threads
            sess_options.intra_op_num_threads = self.config.intra_op_num_threads
            
            # Optimization settings
            if self.config.optimize_model:
                sess_options.graph_optimization_level = (
                    ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                )
            
            # Memory optimization
            if not self.config.enable_memory_arena:
                sess_options.enable_mem_arena = False
            
            # Execution providers
            if self.config.execution_provider == "CUDA":
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            elif self.config.execution_provider == "TensorRT":
                providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']
            
            # Create session
            self._session = ort.InferenceSession(
                self.model_path,
                sess_options=sess_options,
                providers=providers
            )
            
            # Extract metadata
            self._input_names = [inp.name for inp in self._session.get_inputs()]
            self._output_names = [out.name for out in self._session.get_outputs()]
            self._input_shapes = [inp.shape for inp in self._session.get_inputs()]
            
            logger.info(f"Loaded ONNX model: {self.model_path}")
            logger.info(f"Inputs: {self._input_names}, Shapes: {self._input_shapes}")
            logger.info(f"Outputs: {self._output_names}")
        
        except ImportError:
            raise ImportError("onnxruntime not available. Install with: pip install onnxruntime")
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            raise
    
    def infer(self, 
              inputs: Dict[str, np.ndarray],
              timeout_ms: float = 100.0) -> Dict[str, np.ndarray]:
        """
        Run inference on input data.
        
        Args:
            inputs: Dictionary mapping input names to numpy arrays
            timeout_ms: Maximum inference time
        
        Returns:
            Dictionary mapping output names to numpy arrays
        """
        if self._session is None:
            raise RuntimeError("Session not loaded")
        
        start_time = time.time()
        
        # Validate inputs
        for name, data in inputs.items():
            if name not in self._input_names:
                raise ValueError(f"Unknown input: {name}")
            if not isinstance(data, np.ndarray):
                inputs[name] = np.array(data)
        
        # Run inference with timeout
        try:
            outputs = self._session.run(self._output_names, inputs)
            
            # Check timeout
            elapsed_ms = (time.time() - start_time) * 1000
            if elapsed_ms > timeout_ms:
                logger.warning(f"Inference exceeded timeout: {elapsed_ms:.2f}ms > {timeout_ms}ms")
            
            return {name: outputs[i] for i, name in enumerate(self._output_names)}
        
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise
    
    def infer_single(self, 
                     input_array: np.ndarray,
                     input_name: Optional[str] = None) -> np.ndarray:
        """
        Run inference on a single input array.
        Convenience method for models with single input/output.
        
        Args:
            input_array: Input numpy array
            input_name: Name of input (uses first input if None)
        
        Returns:
            Output numpy array
        """
        if input_name is None:
            input_name = self._input_names[0]
        
        inputs = {input_name: input_array}
        outputs = self.infer(inputs)
        
        # Return first output
        return list(outputs.values())[0]
    
    def infer_batch(self,
                    batch: np.ndarray,
                    input_name: Optional[str] = None) -> np.ndarray:
        """
        Run inference on a batch of inputs.
        
        Args:
            batch: Input batch (batch_size, ...)
            input_name: Name of input (uses first input if None)
        
        Returns:
            Output batch
        """
        return self.infer_single(batch, input_name)
    
    def get_input_info(self) -> List[Dict[str, Any]]:
        """Get information about model inputs."""
        if self._session is None:
            return []
        
        inputs = self._session.get_inputs()
        return [
            {
                "name": inp.name,
                "shape": inp.shape,
                "type": inp.type,
            }
            for inp in inputs
        ]
    
    def get_output_info(self) -> List[Dict[str, Any]]:
        """Get information about model outputs."""
        if self._session is None:
            return []
        
        outputs = self._session.get_outputs()
        return [
            {
                "name": out.name,
                "shape": out.shape,
                "type": out.type,
            }
            for out in outputs
        ]
    
    def benchmark(self, 
                  dummy_input: Dict[str, np.ndarray],
                  n_iterations: int = 100) -> Dict[str, float]:
        """
        Benchmark inference latency.
        
        Args:
            dummy_input: Sample input for benchmarking
            n_iterations: Number of iterations
        
        Returns:
            Benchmark statistics
        """
        latencies = []
        
        for _ in range(n_iterations):
            start = time.perf_counter()
            self.infer(dummy_input)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms
        
        latencies = np.array(latencies)
        
        return {
            "mean_latency_ms": float(np.mean(latencies)),
            "median_latency_ms": float(np.median(latencies)),
            "p95_latency_ms": float(np.percentile(latencies, 95)),
            "p99_latency_ms": float(np.percentile(latencies, 99)),
            "min_latency_ms": float(np.min(latencies)),
            "max_latency_ms": float(np.max(latencies)),
            "std_latency_ms": float(np.std(latencies)),
        }
    
    @property
    def is_loaded(self) -> bool:
        return self._session is not None
    
    @property
    def input_names(self) -> List[str]:
        return self._input_names
    
    @property
    def output_names(self) -> List[str]:
        return self._output_names


class MultiModelONNXInference:
    """
    Manage multiple ONNX models for ensemble inference.
    Thread-safe model registry with lazy loading.
    """
    
    def __init__(self, default_config: Optional[ONNXConfig] = None):
        self._models: Dict[str, ONNXInferenceSession] = {}
        self._default_config = default_config or ONNXConfig()
        self._lock = threading.RLock()
    
    def register_model(self, 
                       model_id: str,
                       model_path: str,
                       config: Optional[ONNXConfig] = None) -> None:
        """
        Register and load an ONNX model.
        
        Args:
            model_id: Unique identifier for the model
            model_path: Path to ONNX model file
            config: Model-specific configuration
        """
        with self._lock:
            cfg = config or self._default_config
            session = ONNXInferenceSession(model_path, cfg)
            self._models[model_id] = session
            logger.info(f"Registered model: {model_id}")
    
    def unregister_model(self, model_id: str) -> None:
        """Unregister a model."""
        with self._lock:
            if model_id in self._models:
                del self._models[model_id]
                logger.info(f"Unregistered model: {model_id}")
    
    def get_model(self, model_id: str) -> Optional[ONNXInferenceSession]:
        """Get a registered model by ID."""
        with self._lock:
            return self._models.get(model_id)
    
    def infer(self, 
              model_id: str,
              inputs: Dict[str, np.ndarray]) -> Optional[Dict[str, np.ndarray]]:
        """
        Run inference on a specific model.
        
        Args:
            model_id: Model identifier
            inputs: Input data
        
        Returns:
            Inference outputs or None if model not found
        """
        model = self.get_model(model_id)
        if model is None:
            logger.warning(f"Model {model_id} not found")
            return None
        
        return model.infer(inputs)
    
    def infer_ensemble(self,
                       inputs: Dict[str, np.ndarray],
                       model_ids: Optional[List[str]] = None,
                       weights: Optional[Dict[str, float]] = None) -> Optional[np.ndarray]:
        """
        Run ensemble inference across multiple models.
        
        Args:
            inputs: Input data
            model_ids: List of model IDs to use (all active if None)
            weights: Optional weights for each model
        
        Returns:
            Weighted ensemble prediction
        """
        with self._lock:
            if model_ids is None:
                model_ids = list(self._models.keys())
            
            predictions = []
            valid_weights = []
            
            for model_id in model_ids:
                model = self._models.get(model_id)
                if model is None:
                    continue
                
                try:
                    outputs = model.infer(inputs)
                    pred = list(outputs.values())[0]
                    predictions.append(pred)
                    
                    weight = weights.get(model_id, 1.0) if weights else 1.0
                    valid_weights.append(weight)
                
                except Exception as e:
                    logger.error(f"Ensemble inference failed for {model_id}: {e}")
            
            if not predictions:
                return None
            
            # Weighted average
            weights_array = np.array(valid_weights)
            weights_array /= weights_array.sum()
            
            predictions_array = np.stack(predictions)
            ensemble_pred = np.average(predictions_array, axis=0, weights=weights_array)
            
            return ensemble_pred
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about registered models."""
        with self._lock:
            stats = {
                "total_models": len(self._models),
                "model_ids": list(self._models.keys()),
            }
            
            for model_id, model in self._models.items():
                stats[f"{model_id}_inputs"] = model.get_input_info()
                stats[f"{model_id}_outputs"] = model.get_output_info()
            
            return stats


def create_onnx_session(model_path: str,
                        config: Optional[ONNXConfig] = None) -> ONNXInferenceSession:
    """
    Factory function to create ONNX inference session.
    
    Args:
        model_path: Path to ONNX model
        config: Session configuration
    
    Returns:
        ONNXInferenceSession instance
    """
    return ONNXInferenceSession(model_path, config)


if __name__ == "__main__":
    # Example usage with dummy model creation
    print("ONNX Inference Module Demo")
    print("=" * 40)
    
    # Create a simple test to verify onnxruntime availability
    try:
        import onnxruntime as ort
        print(f"ONNX Runtime version: {ort.__version__}")
        print(f"Available providers: {ort.get_available_providers()}")
    except ImportError:
        print("onnxruntime not installed. Install with: pip install onnxruntime")
    
    # Demo configuration
    config = ONNXConfig(
        n_threads=4,
        inter_op_num_threads=1,
        intra_op_num_threads=4,
        execution_provider="CPU"
    )
    
    print(f"\nConfiguration:")
    print(f"  Threads: {config.n_threads}")
    print(f"  Inter-op threads: {config.inter_op_num_threads}")
    print(f"  Intra-op threads: {config.intra_op_num_threads}")
    print(f"  Execution provider: {config.execution_provider}")
    
    # Note: Actual model loading requires a valid .onnx file
    print("\nTo use this module:")
    print("1. Export your PyTorch model to ONNX format")
    print("2. Call: session = create_onnx_session('model.onnx', config)")
    print("3. Run inference: outputs = session.infer(inputs)")
