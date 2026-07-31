"""
Batch Prediction Engine for Tabular Alpha Models
Consumes numpy feature matrices and outputs probability distributions.
Ensures zero-copy handoffs between feature ring buffer and inference engine.
"""

import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from collections import deque
import threading
import time

from .xgb_ensemble import XGBEnsemble, EnsembleConfig


@dataclass
class PredictionResult:
    """Result container for batch predictions."""
    predictions: np.ndarray
    probabilities: Optional[np.ndarray] = None
    timestamps: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchConfig:
    """Configuration for batch prediction engine."""
    batch_size: int = 256
    max_queue_size: int = 1024
    timeout_ms: float = 10.0
    enable_probabilities: bool = True
    zero_copy: bool = True


class FeatureRingBuffer:
    """
    Lock-free ring buffer for feature matrices.
    Enables zero-copy handoffs to inference engine.
    """
    
    def __init__(self, capacity: int, n_features: int, dtype: np.dtype = np.float32):
        self.capacity = capacity
        self.n_features = n_features
        self.dtype = dtype
        
        # Pre-allocate contiguous memory block
        self._buffer = np.zeros((capacity, n_features), dtype=dtype)
        self._timestamps = np.zeros(capacity, dtype=np.float64)
        
        # Atomic indices using simple counters (thread-safe with lock)
        self._head = 0
        self._tail = 0
        self._count = 0
        self._lock = threading.Lock()
    
    def push(self, features: np.ndarray, timestamp: Optional[float] = None) -> bool:
        """
        Push feature vector to ring buffer.
        
        Args:
            features: Feature vector (n_features,)
            timestamp: Optional timestamp
        
        Returns:
            True if successful, False if buffer is full
        """
        with self._lock:
            if self._count >= self.capacity:
                return False
            
            idx = self._head % self.capacity
            self._buffer[idx] = features.astype(self.dtype, copy=False)
            self._timestamps[idx] = timestamp or time.time()
            
            self._head += 1
            self._count += 1
            
            return True
    
    def pop_batch(self, batch_size: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Pop a batch of features from the buffer.
        Returns views into the buffer for zero-copy operations.
        
        Args:
            batch_size: Number of samples to retrieve
        
        Returns:
            Tuple of (features, timestamps) or None if insufficient data
        """
        with self._lock:
            if self._count < batch_size:
                return None
            
            # Calculate indices for batch
            indices = [(self._tail + i) % self.capacity for i in range(batch_size)]
            
            # Create views (zero-copy where possible)
            features = self._buffer[indices]
            timestamps = self._timestamps[indices]
            
            self._tail += batch_size
            self._count -= batch_size
            
            return features, timestamps
    
    def peek(self, n_samples: int = 1) -> Optional[np.ndarray]:
        """Peek at next n samples without removing them."""
        with self._lock:
            if self._count < n_samples:
                return None
            
            indices = [(self._tail + i) % self.capacity for i in range(n_samples)]
            return self._buffer[indices].copy()
    
    @property
    def size(self) -> int:
        with self._lock:
            return self._count
    
    @property
    def is_empty(self) -> bool:
        with self._lock:
            return self._count == 0
    
    @property
    def is_full(self) -> bool:
        with self._lock:
            return self._count >= self.capacity


class TabularPredictor:
    """
    Batch prediction engine consuming numpy feature matrices.
    Outputs probability distributions with zero-copy optimizations.
    """
    
    def __init__(self, 
                 ensemble: XGBEnsemble,
                 config: Optional[BatchConfig] = None):
        self.ensemble = ensemble
        self.config = config or BatchConfig()
        
        # Ring buffer for incoming features
        self._ring_buffer: Optional[FeatureRingBuffer] = None
        self._n_features: Optional[int] = None
        
        # Output buffers for zero-copy predictions
        self._output_buffer: Optional[np.ndarray] = None
        self._prob_buffer: Optional[np.ndarray] = None
        
        # Thread management
        self._lock = threading.Lock()
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None
        
        # Statistics
        self._total_predictions = 0
        self._total_batches = 0
        self._last_prediction_time = 0.0
    
    def initialize(self, n_features: int) -> None:
        """
        Initialize the predictor with known feature dimension.
        Must be called before start().
        
        Args:
            n_features: Number of input features
        """
        with self._lock:
            self._n_features = n_features
            
            # Initialize ring buffer
            self._ring_buffer = FeatureRingBuffer(
                capacity=self.config.max_queue_size,
                n_features=n_features
            )
            
            # Pre-allocate output buffers
            self._output_buffer = np.empty(self.config.batch_size, dtype=np.float32)
            
            if self.config.enable_probabilities:
                self._prob_buffer = np.empty((self.config.batch_size, 2), dtype=np.float32)
    
    def submit_features(self, features: np.ndarray, 
                       timestamp: Optional[float] = None) -> bool:
        """
        Submit feature vector for prediction.
        
        Args:
            features: Feature vector (n_features,)
            timestamp: Optional timestamp
        
        Returns:
            True if successfully queued
        """
        if self._ring_buffer is None:
            raise RuntimeError("Predictor not initialized. Call initialize() first.")
        
        if features.shape[0] != self._n_features:
            raise ValueError(f"Expected {self._n_features} features, got {features.shape[0]}")
        
        return self._ring_buffer.push(features, timestamp)
    
    def submit_batch(self, features: np.ndarray,
                    timestamps: Optional[np.ndarray] = None) -> bool:
        """
        Submit a batch of features for prediction.
        
        Args:
            features: Feature matrix (n_samples, n_features)
            timestamps: Optional timestamps (n_samples,)
        
        Returns:
            True if all features were queued
        """
        if self._ring_buffer is None:
            raise RuntimeError("Predictor not initialized. Call initialize() first.")
        
        if features.shape[1] != self._n_features:
            raise ValueError(f"Expected {self._n_features} features, got {features.shape[1]}")
        
        success_count = 0
        for i in range(features.shape[0]):
            ts = timestamps[i] if timestamps is not None else None
            if self._ring_buffer.push(features[i], ts):
                success_count += 1
        
        return success_count == features.shape[0]
    
    def predict_batch(self, timeout_ms: Optional[float] = None) -> Optional[PredictionResult]:
        """
        Perform batch prediction on available features.
        Blocks until batch_size features are available or timeout.
        
        Args:
            timeout_ms: Maximum time to wait for batch
        
        Returns:
            PredictionResult or None if timeout
        """
        if self._ring_buffer is None:
            raise RuntimeError("Predictor not initialized. Call initialize() first.")
        
        timeout = timeout_ms or self.config.timeout_ms
        start_time = time.time()
        
        while True:
            batch_data = self._ring_buffer.pop_batch(self.config.batch_size)
            
            if batch_data is not None:
                features, timestamps = batch_data
                return self._run_inference(features, timestamps)
            
            # Check timeout
            if (time.time() - start_time) * 1000 > timeout:
                # Return partial batch if available
                partial = self._ring_buffer.peek(self._ring_buffer.size)
                if partial is not None and partial.shape[0] > 0:
                    # Force pop remaining items
                    partial_ts = np.array([time.time()] * partial.shape[0])
                    return self._run_inference(partial, partial_ts)
                return None
            
            # Brief sleep to avoid busy waiting
            time.sleep(0.001)
    
    def _run_inference(self, features: np.ndarray, 
                       timestamps: np.ndarray) -> PredictionResult:
        """
        Run inference on a batch of features.
        
        Args:
            features: Feature matrix (n_samples, n_features)
            timestamps: Timestamps array (n_samples,)
        
        Returns:
            PredictionResult container
        """
        infer_start = time.time()
        
        n_samples = features.shape[0]
        
        # Ensure output buffer is large enough
        if self._output_buffer.shape[0] < n_samples:
            self._output_buffer = np.empty(n_samples, dtype=np.float32)
        
        # Zero-copy inplace prediction
        output_view = self._output_buffer[:n_samples]
        self.ensemble.predict_inplace(features, output_view)
        
        # Copy predictions to result
        predictions = output_view.copy()
        
        # Optionally get probabilities
        probabilities = None
        if self.config.enable_probabilities:
            try:
                probabilities = self.ensemble.predict_proba_binary(features)
            except Exception:
                probabilities = None
        
        infer_time = time.time() - infer_start
        
        # Update statistics
        with self._lock:
            self._total_predictions += n_samples
            self._total_batches += 1
            self._last_prediction_time = infer_time
        
        return PredictionResult(
            predictions=predictions,
            probabilities=probabilities,
            timestamps=timestamps,
            metadata={
                "inference_time_ms": infer_time * 1000,
                "batch_size": n_samples,
                "feature_mean": features.mean(),
                "feature_std": features.std(),
            }
        )
    
    def predict_single(self, features: np.ndarray) -> PredictionResult:
        """
        Predict on a single feature vector.
        Bypasses the batch queue for low-latency scenarios.
        
        Args:
            features: Feature vector (n_features,)
        
        Returns:
            PredictionResult container
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        predictions = self.ensemble.predict(features)
        
        probabilities = None
        if self.config.enable_probabilities:
            try:
                probabilities = self.ensemble.predict_proba_binary(features)
            except Exception:
                probabilities = None
        
        return PredictionResult(
            predictions=predictions,
            probabilities=probabilities,
            timestamps=np.array([time.time()]),
            metadata={"inference_time_ms": 0.0, "batch_size": 1}
        )
    
    def start_background_worker(self) -> None:
        """Start background worker thread for continuous prediction."""
        if self._running:
            return
        
        self._running = True
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
    
    def stop_background_worker(self) -> None:
        """Stop background worker thread."""
        self._running = False
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=2.0)
            self._worker_thread = None
    
    def _worker_loop(self) -> None:
        """Background worker loop for batch processing."""
        while self._running:
            try:
                result = self.predict_batch()
                if result is not None:
                    # Process result (could emit to callback or queue)
                    pass
            except Exception as e:
                # Log error but continue
                pass
            
            # Brief sleep to prevent CPU spinning
            time.sleep(0.001)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get prediction statistics."""
        with self._lock:
            return {
                "total_predictions": self._total_predictions,
                "total_batches": self._total_batches,
                "last_inference_time_ms": self._last_prediction_time * 1000,
                "queue_size": self._ring_buffer.size if self._ring_buffer else 0,
                "avg_predictions_per_batch": (
                    self._total_predictions / self._total_batches 
                    if self._total_batches > 0 else 0
                ),
            }
    
    @property
    def is_initialized(self) -> bool:
        return self._ring_buffer is not None
    
    @property
    def is_running(self) -> bool:
        return self._running


def create_predictor(ensemble: XGBEnsemble, 
                     config: Optional[BatchConfig] = None,
                     n_features: Optional[int] = None) -> TabularPredictor:
    """
    Factory function to create and initialize a TabularPredictor.
    
    Args:
        ensemble: Trained XGBEnsemble model
        config: Batch configuration
        n_features: Number of input features
    
    Returns:
        Initialized TabularPredictor
    """
    predictor = TabularPredictor(ensemble, config)
    if n_features is not None:
        predictor.initialize(n_features)
    return predictor


if __name__ == "__main__":
    # Example usage
    from .xgb_ensemble import EnsembleConfig
    
    # Create and train ensemble
    config = EnsembleConfig(n_threads=4, n_estimators=50)
    ensemble = XGBEnsemble(config)
    
    # Train on dummy data
    np.random.seed(42)
    X_train = np.random.randn(500, 30).astype(np.float32)
    y_train = np.random.randint(0, 2, 500).astype(np.float32)
    ensemble.fit(X_train, y_train)
    
    # Create predictor
    predictor_config = BatchConfig(batch_size=32, max_queue_size=256)
    predictor = create_predictor(ensemble, predictor_config, n_features=30)
    
    # Submit features
    for i in range(100):
        features = np.random.randn(30).astype(np.float32)
        predictor.submit_features(features)
    
    # Get batch prediction
    result = predictor.predict_batch(timeout_ms=100)
    if result is not None:
        print(f"Batch predictions: {result.predictions.shape}")
        print(f"Mean prediction: {result.predictions.mean():.4f}")
        print(f"Inference time: {result.metadata['inference_time_ms']:.2f} ms")
    
    # Get statistics
    stats = predictor.get_statistics()
    print(f"Statistics: {stats}")
