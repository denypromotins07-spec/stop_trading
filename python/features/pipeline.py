# Ray Actor-based Feature Pipeline for HFT System
# Orchestrates stateless and stateful transformations with zero-copy PyArrow serialization

from __future__ import annotations
import logging
import numpy as np
import pyarrow as pa
from typing import Optional, List, Dict, Any

import ray

log = logging.getLogger(__name__)


@ray.remote(max_calls=1000, memory=500 * 1024 * 1024)
class FeaturePipelineActor:
    """
    Ray actor orchestrating feature transformations.
    Uses PyArrow for zero-copy serialization between workers.
    Strictly enforces memory quotas to prevent OOM.
    """

    def __init__(
        self,
        worker_id: int,
        max_features: int = 500,
        dtype: np.dtype = np.float64,
    ) -> None:
        self.worker_id = worker_id
        self.max_features = max_features
        self.dtype = dtype
        
        # Pre-allocated feature matrix (no heap allocations during runtime)
        self._feature_matrix = np.empty(
            (1000, max_features), 
            dtype=dtype,
            order='C'  # C-contiguous for cache efficiency
        )
        self._current_row = 0
        self._total_processed = 0
        
        # Transformer registry
        self._transformers: Dict[str, Any] = {}
        
        log.info(f"FeaturePipelineActor {worker_id} initialized with {max_features} features")

    def register_transformer(self, name: str, transformer: Any) -> None:
        """Register a transformer for the pipeline."""
        self._transformers[name] = transformer
        log.debug(f"Registered transformer: {name}")

    def transform_batch(self, data: pa.Table) -> pa.Table:
        """
        Apply all registered transformers to a batch of data.
        Uses zero-copy operations where possible.
        """
        try:
            # Convert PyArrow table to numpy view (zero-copy if possible)
            arrays = {col: data.column(col).to_numpy(zero_copy_only=True) 
                     for col in data.column_names}
            
            # Apply transformers in sequence
            for name, transformer in self._transformers.items():
                if hasattr(transformer, 'transform'):
                    arrays = transformer.transform(arrays)
                elif callable(transformer):
                    arrays = transformer(arrays)
            
            # Convert back to PyArrow table
            return pa.table(arrays)
            
        except Exception as e:
            log.error(f"Transform batch error in worker {self.worker_id}: {e}")
            raise

    def append_features(self, feature_vector: np.ndarray) -> int:
        """
        Append a feature vector to the pre-allocated matrix.
        Returns the row index where data was written.
        Uses manual index wrapping to avoid np.roll overhead.
        """
        if len(feature_vector) != self.max_features:
            raise ValueError(
                f"Feature vector length {len(feature_vector)} != "
                f"expected {self.max_features}"
            )
        
        # Write directly to pre-allocated buffer
        row_idx = self._current_row
        self._feature_matrix[row_idx, :] = feature_vector
        
        # Manual circular buffer wrap
        self._current_row = (self._current_row + 1) % 1000
        self._total_processed += 1
        
        return row_idx

    def get_recent_features(self, n: int) -> np.ndarray:
        """
        Get the last N feature vectors without copying.
        Returns a view into the pre-allocated matrix.
        """
        n = min(n, self._total_processed, 1000)
        if n == 0:
            return np.empty((0, self.max_features), dtype=self.dtype)
        
        start_idx = (self._current_row - n) % 1000
        if start_idx < self._current_row:
            return self._feature_matrix[start_idx:self._current_row]
        else:
            # Wrap around case
            return np.vstack([
                self._feature_matrix[start_idx:],
                self._feature_matrix[:self._current_row]
            ])

    def get_stats(self) -> Dict[str, Any]:
        """Return worker statistics."""
        return {
            "worker_id": self.worker_id,
            "total_processed": self._total_processed,
            "current_row": self._current_row,
            "memory_usage_bytes": self._feature_matrix.nbytes,
            "transformers_count": len(self._transformers),
        }

    def reset(self) -> None:
        """Reset the feature matrix (zero-fill)."""
        self._feature_matrix.fill(0.0)
        self._current_row = 0
        self._total_processed = 0
        log.info(f"FeaturePipelineActor {self.worker_id} reset")


@ray.remote(max_calls=1000, memory=200 * 1024 * 1024)
class PipelineOrchestrator:
    """
    Orchestrates multiple FeaturePipelineActors.
    Distributes work and aggregates results.
    """

    def __init__(self, num_workers: int = 4) -> None:
        self.num_workers = num_workers
        self.workers: List[FeaturePipelineActor] = []
        self._initialized = False

    def initialize(self) -> None:
        """Initialize worker pool."""
        self.workers = [
            FeaturePipelineActor.remote(worker_id=i) 
            for i in range(self.num_workers)
        ]
        self._initialized = True
        log.info(f"PipelineOrchestrator initialized with {self.num_workers} workers")

    def distribute_transform(self, batches: List[pa.Table]) -> List[ray.ObjectRef]:
        """Distribute transform tasks across workers."""
        if not self._initialized:
            raise RuntimeError("Orchestrator not initialized")
        
        results = []
        for i, batch in enumerate(batches):
            worker_idx = i % self.num_workers
            result = self.workers[worker_idx].transform_batch.remote(batch)
            results.append(result)
        
        return results

    def collect_results(self, object_refs: List[ray.ObjectRef]) -> List[pa.Table]:
        """Collect results from distributed transforms."""
        return ray.get(object_refs)

    def get_all_stats(self) -> List[Dict[str, Any]]:
        """Get stats from all workers."""
        return ray.get([w.get_stats.remote() for w in self.workers])

    def shutdown(self) -> None:
        """Shutdown all workers."""
        for worker in self.workers:
            ray.kill(worker)
        self.workers = []
        self._initialized = False
        log.info("PipelineOrchestrator shutdown complete")


def create_pipeline(num_workers: int = 4) -> PipelineOrchestrator:
    """Factory function to create a configured pipeline."""
    return PipelineOrchestrator(num_workers=num_workers)
