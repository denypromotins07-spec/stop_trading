# NumPy-backed Ring Buffer for Feature History
# Provides O(1) access to last N ticks without heap allocations

from __future__ import annotations
import logging
import numpy as np
from typing import Optional, Tuple, List

log = logging.getLogger(__name__)


class FeatureRingBuffer:
    """
    Strict numpy-backed ring buffer for storing recent feature history.
    Provides O(1) access to the last N ticks for online learning.
    Uses manual index wrapping on pre-allocated arrays - no np.roll overhead.
    """

    def __init__(
        self,
        capacity: int = 10000,
        feature_dim: int = 128,
        dtype: np.dtype = np.float64,
    ) -> None:
        self.capacity = capacity
        self.feature_dim = feature_dim
        self.dtype = dtype
        
        # Pre-allocated circular buffer (no dynamic allocation)
        self._buffer = np.zeros((capacity, feature_dim), dtype=dtype, order='C')
        
        # Pointers
        self._head = 0  # Next write position
        self._count = 0  # Number of valid entries
        self._total_written = 0  # Lifetime counter
        
        # Timestamp buffer for alignment
        self._timestamps = np.zeros(capacity, dtype=np.int64)
        
        # Statistics
        self._min_values: Optional[np.ndarray] = None
        self._max_values: Optional[np.ndarray] = None
        self._sum_values: Optional[np.ndarray] = None

    def append(self, features: np.ndarray, timestamp: Optional[int] = None) -> int:
        """
        Append a feature vector to the buffer.
        Returns the index where data was written.
        Uses direct assignment - no copying beyond input.
        """
        if len(features) != self.feature_dim:
            raise ValueError(
                f"Feature dimension {len(features)} != expected {self.feature_dim}"
            )
        
        # Write directly to pre-allocated buffer
        self._buffer[self._head] = features
        self._timestamps[self._head] = timestamp if timestamp is not None else self._total_written
        
        # Update statistics incrementally
        self._update_stats(features)
        
        # Advance head with manual wrap
        self._head = (self._head + 1) % self.capacity
        self._count = min(self._count + 1, self.capacity)
        self._total_written += 1
        
        return self._head

    def _update_stats(self, features: np.ndarray) -> None:
        """Update running min/max/sum statistics."""
        if self._min_values is None:
            self._min_values = features.copy()
            self._max_values = features.copy()
            self._sum_values = features.copy()
        else:
            self._min_values = np.minimum(self._min_values, features)
            self._max_values = np.maximum(self._max_values, features)
            self._sum_values += features

    def get(self, index: int) -> np.ndarray:
        """
        Get feature vector at absolute index (from start of buffer).
        Returns zeros if index is out of valid range.
        """
        if index < 0 or index >= self._count:
            return np.zeros(self.feature_dim, dtype=self.dtype)
        
        actual_idx = (self._head - self._count + index) % self.capacity
        return self._buffer[actual_idx].copy()

    def get_recent(self, n: int) -> np.ndarray:
        """
        Get the last N feature vectors as a contiguous array.
        Returns shape (n, feature_dim) or fewer if buffer has less data.
        Uses np.concatenate only when necessary (wrap-around case).
        """
        n = min(n, self._count)
        
        if n == 0:
            return np.empty((0, self.feature_dim), dtype=self.dtype)
        
        # Calculate start index
        start_idx = (self._head - n) % self.capacity
        
        if start_idx < self._head or self._count < self.capacity:
            # No wrap-around needed
            if start_idx <= self._head:
                end_idx = self._head if self._count == self.capacity else self._head
                return self._buffer[start_idx:end_idx].copy()
        
        # Wrap-around case - minimal copy
        part1 = self._buffer[start_idx:]
        part2 = self._buffer[:self._head]
        return np.vstack([part1, part2])

    def get_with_timestamps(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get last N features with their timestamps."""
        features = self.get_recent(n)
        
        n = min(n, self._count)
        start_idx = (self._head - n) % self.capacity
        
        if start_idx < self._head or self._count < self.capacity:
            timestamps = self._timestamps[start_idx:self._head].copy()
        else:
            timestamps = np.concatenate([
                self._timestamps[start_idx:],
                self._timestamps[:self._head]
            ])
        
        return features, timestamps

    def get_range(
        self,
        start_ts: int,
        end_ts: int,
    ) -> np.ndarray:
        """
        Get all features within a timestamp range.
        Linear scan but efficient for small ranges.
        """
        result = []
        
        for i in range(self._count):
            idx = (self._head - self._count + i) % self.capacity
            ts = self._timestamps[idx]
            
            if start_ts <= ts <= end_ts:
                result.append(self._buffer[idx].copy())
        
        if not result:
            return np.empty((0, self.feature_dim), dtype=self.dtype)
        
        return np.vstack(result)

    def normalize(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize features using running statistics.
        Returns z-scored features.
        """
        if self._sum_values is None or self._count == 0:
            return features.copy()
        
        mean = self._sum_values / self._count
        std = np.sqrt((self._max_values - self._min_values) ** 2 / 4 + 1e-9)
        
        return (features - mean) / std

    def get_stats(self) -> dict:
        """Get buffer statistics."""
        return {
            "capacity": self.capacity,
            "count": self._count,
            "total_written": self._total_written,
            "head": self._head,
            "feature_dim": self.feature_dim,
            "memory_bytes": self._buffer.nbytes,
            "min_values": self._min_values.tolist() if self._min_values is not None else None,
            "max_values": self._max_values.tolist() if self._max_values is not None else None,
        }

    def clear(self) -> None:
        """Clear buffer without deallocating memory."""
        self._buffer.fill(0.0)
        self._timestamps.fill(0)
        self._head = 0
        self._count = 0
        self._total_written = 0
        self._min_values = None
        self._max_values = None
        self._sum_values = None
        log.info("FeatureRingBuffer cleared")

    def __len__(self) -> int:
        """Return number of valid entries."""
        return self._count

    def __getitem__(self, key: int) -> np.ndarray:
        """Support negative indexing like Python lists."""
        if isinstance(key, int):
            if key < 0:
                key = self._count + key
            return self.get(key)
        raise TypeError(f"Key must be int, got {type(key)}")


def create_ring_buffer(
    capacity: int = 10000,
    feature_dim: int = 128,
) -> FeatureRingBuffer:
    """Factory function to create a configured ring buffer."""
    return FeatureRingBuffer(capacity=capacity, feature_dim=feature_dim)


def batch_append(
    buffer: FeatureRingBuffer,
    features: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
) -> List[int]:
    """
    Efficiently append multiple feature vectors.
    Returns list of write indices.
    """
    if features.ndim != 2:
        raise ValueError("Features must be 2D array")
    
    indices = []
    n_samples = len(features)
    
    for i in range(n_samples):
        ts = int(timestamps[i]) if timestamps is not None else None
        idx = buffer.append(features[i], ts)
        indices.append(idx)
    
    return indices
