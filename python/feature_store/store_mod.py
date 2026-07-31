# Feature Store Module Root
# Enforces strict byte-limits on vector cache to guarantee 3GB Python RAM ceiling

from __future__ import annotations
import logging
import numpy as np
from typing import Optional, Dict, Any, List

log = logging.getLogger(__name__)

from python.feature_store.ring_buffer import (
    FeatureRingBuffer,
    create_ring_buffer,
    batch_append,
)
from python.feature_store.vector_cache import (
    VectorCache,
    create_vector_cache,
    find_similar_states,
)


class MemoryEnforcer:
    """
    Enforces strict memory limits across all feature store components.
    Monitors total Python memory usage and triggers cleanup when needed.
    """

    # 3GB Python RAM ceiling (leaving rest for Rust core and OS)
    MAX_PYTHON_MEMORY_MB = 3072
    WARNING_THRESHOLD = 0.85  # Warn at 85% usage
    CRITICAL_THRESHOLD = 0.95  # Force cleanup at 95% usage

    def __init__(self) -> None:
        self._components: List[Any] = []
        self._component_names: List[str] = []
        self._last_check_ts: int = 0

    def register_component(
        self,
        component: Any,
        name: str,
        get_memory_func: Optional[callable] = None,
    ) -> None:
        """Register a component for memory monitoring."""
        self._components.append(component)
        self._component_names.append(name)
        log.info(f"Registered memory component: {name}")

    def get_total_memory_mb(self) -> float:
        """Calculate total memory usage across all components."""
        total = 0.0
        
        for comp in self._components:
            if hasattr(comp, 'get_stats'):
                stats = comp.get_stats()
                if 'memory_bytes' in stats:
                    total += stats['memory_bytes'] / (1024 * 1024)
                elif 'memory_mb' in stats:
                    total += stats['memory_mb']
        
        # Add Python process memory estimate
        try:
            import resource
            rusage = resource.getrusage(resource.RUSAGE_SELF)
            process_mb = rusage.ru_maxrss / 1024  # Convert KB to MB (Linux)
            total = max(total, process_mb)  # Use larger of calculated or reported
        except ImportError:
            pass
        
        return total

    def check_and_cleanup(self) -> Dict[str, Any]:
        """
        Check memory usage and trigger cleanup if needed.
        Returns status report.
        """
        import time
        current_ts = int(time.time() * 1000)
        
        # Rate limit checks to once per second
        if current_ts - self._last_check_ts < 1000:
            return {"status": "skipped", "reason": "rate_limited"}
        
        self._last_check_ts = current_ts
        
        total_mb = self.get_total_memory_mb()
        utilization = total_mb / self.MAX_PYTHON_MEMORY_MB
        
        status = {
            "total_memory_mb": total_mb,
            "max_memory_mb": self.MAX_PYTHON_MEMORY_MB,
            "utilization": utilization,
            "status": "ok",
            "actions_taken": [],
        }
        
        if utilization >= self.CRITICAL_THRESHOLD:
            # Force aggressive cleanup
            log.warning(
                f"CRITICAL: Memory at {utilization*100:.1f}% - forcing cleanup"
            )
            self._force_cleanup()
            status["status"] = "critical_cleanup"
            status["actions_taken"].append("forced_cleanup")
            
        elif utilization >= self.WARNING_THRESHOLD:
            # Log warning
            log.warning(
                f"WARNING: Memory at {utilization*100:.1f}% - approaching limit"
            )
            status["status"] = "warning"
        
        return status

    def _force_cleanup(self) -> None:
        """Force cleanup of all registered components."""
        for comp in self._components:
            if hasattr(comp, 'clear'):
                comp.clear()
            elif hasattr(comp, 'reset'):
                comp.reset()
        
        # Suggest garbage collection
        import gc
        gc.collect()
        
        log.info("Memory enforcement cleanup complete")

    def get_component_breakdown(self) -> Dict[str, float]:
        """Get memory breakdown by component."""
        breakdown = {}
        
        for name, comp in zip(self._component_names, self._components):
            if hasattr(comp, 'get_stats'):
                stats = comp.get_stats()
                if 'memory_bytes' in stats:
                    breakdown[name] = stats['memory_bytes'] / (1024 * 1024)
                elif 'memory_mb' in stats:
                    breakdown[name] = stats['memory_mb']
        
        return breakdown


class FeatureStore:
    """
    Central feature store combining ring buffer and vector cache.
    Provides unified API for feature storage and retrieval.
    """

    def __init__(
        self,
        ring_capacity: int = 10000,
        feature_dim: int = 128,
        cache_max_entries: int = 50000,
        cache_max_memory_mb: int = 500,
    ) -> None:
        self.ring_buffer = create_ring_buffer(
            capacity=ring_capacity,
            feature_dim=feature_dim,
        )
        self.vector_cache = create_vector_cache(
            max_entries=cache_max_entries,
            vector_dim=feature_dim,
            max_memory_mb=cache_max_memory_mb,
        )
        
        self.memory_enforcer = MemoryEnforcer()
        self.memory_enforcer.register_component(self.ring_buffer, "ring_buffer")
        self.memory_enforcer.register_component(self.vector_cache, "vector_cache")
        
        self._insert_count = 0

    def store(
        self,
        features: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[int] = None,
        store_in_cache: bool = True,
    ) -> int:
        """
        Store features in both ring buffer and optionally in cache.
        """
        # Always store in ring buffer
        ring_idx = self.ring_buffer.append(features, timestamp)
        
        # Optionally store in vector cache for similarity search
        if store_in_cache:
            self.vector_cache.insert(features, metadata, timestamp)
        
        self._insert_count += 1
        
        # Periodic memory check
        if self._insert_count % 1000 == 0:
            self.memory_enforcer.check_and_cleanup()
        
        return ring_idx

    def query_similar(
        self,
        features: np.ndarray,
        k: int = 10,
        min_similarity: float = 0.9,
    ) -> List[Dict[str, Any]]:
        """Find similar historical states."""
        return find_similar_states(
            self.vector_cache,
            features,
            min_similarity=min_similarity,
            max_results=k,
        )

    def get_recent_features(self, n: int) -> np.ndarray:
        """Get last N feature vectors from ring buffer."""
        return self.ring_buffer.get_recent(n)

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive store statistics."""
        mem_status = self.memory_enforcer.check_and_cleanup()
        
        return {
            "ring_buffer": self.ring_buffer.get_stats(),
            "vector_cache": self.vector_cache.get_stats(),
            "memory": mem_status,
            "component_breakdown": self.memory_enforcer.get_component_breakdown(),
            "total_inserts": self._insert_count,
        }

    def reset(self) -> None:
        """Reset all components."""
        self.ring_buffer.clear()
        self.vector_cache.clear()
        self._insert_count = 0
        log.info("FeatureStore reset")


# Global instance
_store: Optional[FeatureStore] = None


def get_store() -> FeatureStore:
    """Get or create global feature store."""
    global _store
    if _store is None:
        _store = FeatureStore()
    return _store


def initialize_store(
    ring_capacity: int = 10000,
    feature_dim: int = 128,
    cache_max_entries: int = 50000,
    cache_max_memory_mb: int = 500,
) -> FeatureStore:
    """Initialize global store with custom parameters."""
    global _store
    _store = FeatureStore(
        ring_capacity=ring_capacity,
        feature_dim=feature_dim,
        cache_max_entries=cache_max_entries,
        cache_max_memory_mb=cache_max_memory_mb,
    )
    return _store


__all__ = [
    "FeatureRingBuffer",
    "VectorCache",
    "MemoryEnforcer",
    "FeatureStore",
    "create_ring_buffer",
    "create_vector_cache",
    "batch_append",
    "find_similar_states",
    "get_store",
    "initialize_store",
]
