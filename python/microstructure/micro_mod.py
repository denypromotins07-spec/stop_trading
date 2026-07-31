# Microstructure Module Root
# Pushes microstructure tensors to bounded inference queue

from __future__ import annotations
import logging
import numpy as np
from typing import Optional, Dict, Any, List
from collections import deque
import threading

log = logging.getLogger(__name__)

from python.microstructure.vpin_calculator import VPINCalculator
from python.microstructure.imbalance_features import OrderBookImbalanceCalculator


class BoundedInferenceQueue:
    """
    Thread-safe bounded queue for microstructure tensors.
    Uses pre-allocated circular buffer to avoid heap allocations.
    """

    def __init__(
        self,
        max_size: int = 1000,
        tensor_shape: tuple = (100, 40),  # (sequence_length, features)
        dtype: np.dtype = np.float64,
    ) -> None:
        self.max_size = max_size
        self.tensor_shape = tensor_shape
        self.dtype = dtype
        
        # Pre-allocated circular buffer
        self._buffer = np.zeros((max_size,) + tensor_shape, dtype=dtype)
        self._head = 0
        self._tail = 0
        self._count = 0
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Metadata storage
        self._metadata: deque = deque(maxlen=max_size)

    def put(self, tensor: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add tensor to queue. Returns False if queue is full.
        Overwrites oldest entry if full (circular behavior).
        """
        with self._lock:
            if tensor.shape != self.tensor_shape:
                log.warning(
                    f"Tensor shape {tensor.shape} != expected {self.tensor_shape}"
                )
                return False
            
            # Write to buffer
            self._buffer[self._head] = tensor
            self._metadata.append(metadata)
            
            # Update pointers
            self._head = (self._head + 1) % self.max_size
            
            if self._count == self.max_size:
                # Buffer full, move tail
                self._tail = (self._tail + 1) % self.max_size
            else:
                self._count += 1
            
            return True

    def get(self, blocking: bool = False) -> Optional[tuple[np.ndarray, Optional[Dict[str, Any]]]]:
        """
        Get oldest tensor from queue.
        Returns None if empty (non-blocking) or blocks if requested.
        """
        with self._lock:
            if self._count == 0:
                return None
            
            tensor = self._buffer[self._tail].copy()
            metadata = self._metadata.popleft() if self._metadata else None
            
            self._tail = (self._tail + 1) % self.max_size
            self._count -= 1
            
            return tensor, metadata

    def get_batch(self, batch_size: int) -> tuple[np.ndarray, List[Optional[Dict[str, Any]]]]:
        """Get a batch of tensors."""
        with self._lock:
            actual_size = min(batch_size, self._count)
            
            if actual_size == 0:
                return np.empty((0,) + self.tensor_shape, dtype=self.dtype), []
            
            batch = np.empty((actual_size,) + self.tensor_shape, dtype=self.dtype)
            metadata_list = []
            
            for i in range(actual_size):
                idx = (self._tail + i) % self.max_size
                batch[i] = self._buffer[idx]
                if len(self._metadata) > i:
                    metadata_list.append(self._metadata[i])
            
            # Update pointers
            self._tail = (self._tail + actual_size) % self.max_size
            self._count -= actual_size
            
            # Remove metadata
            for _ in range(actual_size):
                if self._metadata:
                    self._metadata.popleft()
            
            return batch, metadata_list

    def size(self) -> int:
        """Return current queue size."""
        with self._lock:
            return self._count

    def is_full(self) -> bool:
        """Check if queue is full."""
        with self._lock:
            return self._count >= self.max_size

    def clear(self) -> None:
        """Clear the queue."""
        with self._lock:
            self._buffer.fill(0.0)
            self._head = 0
            self._tail = 0
            self._count = 0
            self._metadata.clear()


class MicrostructureEngine:
    """
    Central engine for microstructure analytics.
    Combines VPIN and imbalance calculations, pushes to inference queue.
    """

    def __init__(
        self,
        vpin_bucket_size: int = 1000,
        vpin_num_buckets: int = 50,
        n_orderbook_levels: int = 10,
        history_length: int = 100,
        queue_max_size: int = 1000,
    ) -> None:
        self.vpin_calculator = VPINCalculator(
            bucket_size=vpin_bucket_size,
            num_buckets=vpin_num_buckets,
        )
        self.imbalance_calculator = OrderBookImbalanceCalculator(
            n_levels=n_orderbook_levels,
            history_length=history_length,
        )
        self.inference_queue = BoundedInferenceQueue(
            max_size=queue_max_size,
            tensor_shape=(history_length, n_orderbook_levels * 4 + 8),  # Imbalance + VPIN features
        )
        
        self._running = False
        self._processed_count = 0

    def process_tick(
        self,
        price: float,
        volume: float,
        bid_prices: np.ndarray,
        bid_volumes: np.ndarray,
        ask_prices: np.ndarray,
        ask_volumes: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Process a single tick and optionally push to inference queue.
        """
        # Update VPIN
        prices = np.array([price])
        volumes = np.array([volume])
        self.vpin_calculator.update(prices, volumes)
        
        # Update order book imbalance
        self.imbalance_calculator.update_order_book(
            bid_prices, bid_volumes, ask_prices, ask_volumes
        )
        
        # Combine features
        vpin_features = self.vpin_calculator.get_toxicity_features()
        imbalance_features = self.imbalance_calculator.get_imbalance_features()
        
        # Create combined tensor
        imbalance_tensor = self.imbalance_calculator.get_transformer_ready_tensor()
        
        if len(imbalance_tensor) == 0:
            return None
        
        # Append VPIN features to each timestep
        vpin_vector = np.array([
            vpin_features.get("vpin_current", 0.0),
            vpin_features.get("vpin_mean", 0.0),
            vpin_features.get("vpin_std", 0.0),
            vpin_features.get("toxicity_regime", 0),
            vpin_features.get("volume_imbalance", 0.0),
            vpin_features.get("spread_pct", 0.0),
            vpin_features.get("weighted_imbalance", 0.0),
            vpin_features.get("imbalance_trend", 0.0),
        ], dtype=np.float64)
        
        # Broadcast VPIN features to match sequence length
        seq_len = len(imbalance_tensor)
        vpin_broadcast = np.tile(vpin_vector, (seq_len, 1))
        
        # Concatenate: (seq_len, imbalance_features + vpin_features)
        combined_tensor = np.hstack([imbalance_tensor, vpin_broadcast])
        
        # Push to inference queue
        metadata = {
            "ts": self.vpin_calculator._total_volume_processed,
            "vpin": vpin_features.get("vpin_current", 0.0),
        }
        self.inference_queue.put(combined_tensor, metadata)
        self._processed_count += 1
        
        return combined_tensor

    def get_latest_features(self) -> Dict[str, Any]:
        """Get latest feature snapshot."""
        return {
            "vpin": self.vpin_calculator.get_toxicity_features(),
            "imbalance": self.imbalance_calculator.get_imbalance_features(),
            "queue_size": self.inference_queue.size(),
            "processed_count": self._processed_count,
        }

    def reset(self) -> None:
        """Reset all components."""
        self.vpin_calculator.reset()
        self.imbalance_calculator.reset()
        self.inference_queue.clear()
        self._processed_count = 0
        log.info("MicrostructureEngine reset")


# Global instance (lazy initialization)
_engine: Optional[MicrostructureEngine] = None


def get_engine() -> MicrostructureEngine:
    """Get or create the global microstructure engine."""
    global _engine
    if _engine is None:
        _engine = MicrostructureEngine()
    return _engine


def initialize_engine(
    vpin_bucket_size: int = 1000,
    vpin_num_buckets: int = 50,
    n_orderbook_levels: int = 10,
) -> MicrostructureEngine:
    """Initialize the global microstructure engine with custom parameters."""
    global _engine
    _engine = MicrostructureEngine(
        vpin_bucket_size=vpin_bucket_size,
        vpin_num_buckets=vpin_num_buckets,
        n_orderbook_levels=n_orderbook_levels,
    )
    return _engine


__all__ = [
    "VPINCalculator",
    "OrderBookImbalanceCalculator",
    "BoundedInferenceQueue",
    "MicrostructureEngine",
    "get_engine",
    "initialize_engine",
]
