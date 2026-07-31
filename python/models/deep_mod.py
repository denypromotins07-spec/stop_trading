"""
Deep Learning Module Root
Manages deep learning inference queue and handles batched tensor submissions.
Coordinates ONNX inference sessions for transformer models.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
import threading
import queue
import time
import logging

from .onnx_inference import (
    ONNXInferenceSession, 
    MultiModelONNXInference,
    ONNXConfig,
    create_onnx_session
)
from .timeseries_transformer import TransformerConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class InferenceRequest:
    """Request container for deep learning inference."""
    request_id: str
    input_data: np.ndarray
    model_id: str
    timestamp: float = field(default_factory=time.time)
    callback: Optional[Callable[[np.ndarray], None]] = None
    priority: int = 0  # Higher = more urgent


@dataclass
class InferenceResult:
    """Result container for deep learning inference."""
    request_id: str
    output_data: np.ndarray
    model_id: str
    latency_ms: float
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DLConfig:
    """Configuration for deep learning module."""
    max_queue_size: int = 10000
    batch_size: int = 64
    batch_timeout_ms: float = 5.0
    n_workers: int = 2
    default_n_threads: int = 4


class DeepLearningModule:
    """
    Central module for deep learning inference.
    Manages ONNX sessions and handles batched tensor submissions.
    """
    
    def __init__(self, config: Optional[DLConfig] = None):
        self.config = config or DLConfig()
        
        # Multi-model inference manager
        self._model_manager = MultiModelONNXInference(
            ONNXConfig(n_threads=self.config.default_n_threads)
        )
        
        # Request queue with priority support
        self._request_queue: queue.PriorityQueue = queue.PriorityQueue(
            maxsize=self.config.max_queue_size
        )
        
        # Result cache
        self._results: Dict[str, InferenceResult] = {}
        self._results_lock = threading.Lock()
        
        # Worker threads
        self._running = False
        self._workers: List[threading.Thread] = []
        
        # Statistics
        self._total_requests = 0
        self._total_processed = 0
        self._total_errors = 0
        self._latencies: List[float] = []
    
    def register_model(self,
                       model_id: str,
                       model_path: str,
                       config: Optional[ONNXConfig] = None) -> bool:
        """
        Register a new ONNX model.
        
        Args:
            model_id: Unique model identifier
            model_path: Path to ONNX model file
            config: Model-specific configuration
        
        Returns:
            True if successful
        """
        try:
            self._model_manager.register_model(model_id, model_path, config)
            logger.info(f"Registered DL model: {model_id} at {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to register model {model_id}: {e}")
            return False
    
    def unregister_model(self, model_id: str) -> None:
        """Unregister a model."""
        self._model_manager.unregister_model(model_id)
        logger.info(f"Unregistered DL model: {model_id}")
    
    def submit_request(self,
                       input_data: np.ndarray,
                       model_id: str,
                       callback: Optional[Callable[[np.ndarray], None]] = None,
                       priority: int = 0,
                       request_id: Optional[str] = None) -> str:
        """
        Submit an inference request.
        
        Args:
            input_data: Input tensor
            model_id: Target model
            callback: Optional callback for result
            priority: Request priority (higher = more urgent)
            request_id: Optional custom request ID
        
        Returns:
            Request ID
        """
        if request_id is None:
            request_id = f"dl_{model_id}_{time.time_ns()}"
        
        request = InferenceRequest(
            request_id=request_id,
            input_data=input_data,
            model_id=model_id,
            callback=callback,
            priority=-priority  # Negate for min-heap behavior
        )
        
        try:
            self._request_queue.put_nowait((-priority, time.time(), request))
            self._total_requests += 1
            return request_id
        except queue.Full:
            logger.warning("Request queue full, dropping request")
            return ""
    
    def submit_batch(self,
                     inputs: np.ndarray,
                     model_id: str,
                     base_priority: int = 0) -> List[str]:
        """
        Submit a batch of inference requests.
        
        Args:
            inputs: Batch of input tensors (batch_size, ...)
            model_id: Target model
            base_priority: Base priority for all requests
        
        Returns:
            List of request IDs
        """
        request_ids = []
        
        for i in range(inputs.shape[0]):
            request_id = self.submit_request(
                input_data=inputs[i],
                model_id=model_id,
                priority=base_priority
            )
            if request_id:
                request_ids.append(request_id)
        
        return request_ids
    
    def get_result(self, 
                   request_id: str,
                   timeout_ms: float = 1000.0) -> Optional[InferenceResult]:
        """
        Get result for a request.
        
        Args:
            request_id: Request identifier
            timeout_ms: Maximum wait time
        
        Returns:
            InferenceResult or None
        """
        start_time = time.time()
        
        while True:
            with self._results_lock:
                if request_id in self._results:
                    return self._results.pop(request_id)
            
            if (time.time() - start_time) * 1000 > timeout_ms:
                return None
            
            time.sleep(0.001)
    
    def _process_request(self, request: InferenceRequest) -> Optional[InferenceResult]:
        """
        Process a single inference request.
        
        Args:
            request: InferenceRequest
        
        Returns:
            InferenceResult or None
        """
        start_time = time.perf_counter()
        
        try:
            # Run inference
            outputs = self._model_manager.infer(
                request.model_id,
                {"input": request.input_data}
            )
            
            if outputs is None:
                raise RuntimeError(f"Model {request.model_id} returned None")
            
            # Extract output
            output_data = list(outputs.values())[0]
            
            # Calculate latency
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            result = InferenceResult(
                request_id=request.request_id,
                output_data=output_data,
                model_id=request.model_id,
                latency_ms=latency_ms,
                metadata={"input_shape": request.input_data.shape}
            )
            
            # Store result
            with self._results_lock:
                self._results[request.request_id] = result
                self._latencies.append(latency_ms)
            
            # Invoke callback if provided
            if request.callback is not None:
                try:
                    request.callback(output_data)
                except Exception as e:
                    logger.error(f"Callback error: {e}")
            
            self._total_processed += 1
            return result
        
        except Exception as e:
            logger.error(f"Inference error for {request.request_id}: {e}")
            self._total_errors += 1
            return None
    
    def _worker_loop(self, worker_id: int) -> None:
        """
        Worker thread loop for processing requests.
        
        Args:
            worker_id: Worker identifier
        """
        logger.info(f"DL worker {worker_id} started")
        
        while self._running:
            try:
                # Get request with timeout
                _, _, request = self._request_queue.get(timeout=0.1)
                self._process_request(request)
                self._request_queue.task_done()
            
            except queue.Empty:
                pass
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
        
        logger.info(f"DL worker {worker_id} stopped")
    
    def start_workers(self, n_workers: Optional[int] = None) -> None:
        """
        Start worker threads.
        
        Args:
            n_workers: Number of workers (uses config default if None)
        """
        if self._running:
            return
        
        n_workers = n_workers or self.config.n_workers
        self._running = True
        
        for i in range(n_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                args=(i,),
                daemon=True,
                name=f"DL_Worker_{i}"
            )
            worker.start()
            self._workers.append(worker)
        
        logger.info(f"Started {n_workers} DL workers")
    
    def stop_workers(self) -> None:
        """Stop all worker threads."""
        self._running = False
        
        for worker in self._workers:
            worker.join(timeout=2.0)
        
        self._workers.clear()
        logger.info("Stopped all DL workers")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get module statistics."""
        latencies = np.array(self._latencies[-1000:])  # Last 1000 latencies
        
        stats = {
            "total_requests": self._total_requests,
            "total_processed": self._total_processed,
            "total_errors": self._total_errors,
            "queue_size": self._request_queue.qsize(),
            "active_workers": len(self._workers),
            "registered_models": list(self._model_manager._models.keys()),
        }
        
        if len(latencies) > 0:
            stats.update({
                "mean_latency_ms": float(np.mean(latencies)),
                "median_latency_ms": float(np.median(latencies)),
                "p95_latency_ms": float(np.percentile(latencies, 95)),
                "p99_latency_ms": float(np.percentile(latencies, 99)),
            })
        
        return stats
    
    @property
    def is_running(self) -> bool:
        return self._running


# Global module instance (singleton pattern)
_deep_learning_module: Optional[DeepLearningModule] = None
_module_lock = threading.Lock()


def get_deep_learning_module(config: Optional[DLConfig] = None) -> DeepLearningModule:
    """
    Get or create the global DeepLearningModule instance.
    
    Args:
        config: Module configuration
    
    Returns:
        DeepLearningModule instance
    """
    global _deep_learning_module
    
    with _module_lock:
        if _deep_learning_module is None:
            _deep_learning_module = DeepLearningModule(config)
        
        return _deep_learning_module


def reset_deep_learning_module() -> None:
    """Reset the global module instance."""
    global _deep_learning_module
    
    with _module_lock:
        if _deep_learning_module is not None:
            _deep_learning_module.stop_workers()
            _deep_learning_module = None


if __name__ == "__main__":
    # Example usage
    print("Deep Learning Module Demo")
    print("=" * 40)
    
    # Initialize module
    config = DLConfig(
        max_queue_size=1000,
        batch_size=32,
        n_workers=2,
        default_n_threads=4
    )
    
    module = get_deep_learning_module(config)
    
    # Note: Actual model registration requires valid ONNX files
    print("\nTo use this module:")
    print("1. Export your transformer model to ONNX format")
    print("2. Call: module.register_model('model_id', 'path/to/model.onnx')")
    print("3. Start workers: module.start_workers()")
    print("4. Submit requests: request_id = module.submit_request(input_data, 'model_id')")
    print("5. Get results: result = module.get_result(request_id)")
    
    # Demo statistics
    stats = module.get_statistics()
    print(f"\nInitial statistics: {stats}")
    
    # Cleanup
    reset_deep_learning_module()
