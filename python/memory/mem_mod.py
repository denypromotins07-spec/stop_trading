"""
Memory Module Root.
Manages FAISS index lifecycle with strict 200MB memory bound.
"""

import asyncio
from typing import Optional, Dict, Any
import logging
import os

from .faiss_index import (
    FAISSIndexManager,
    VectorStorage,
    MemoryEntry,
    get_index_manager,
    get_vector_storage,
    shutdown_memory_module as shutdown_faiss,
)
from .context_retriever import (
    ContextRetriever,
    RetrievedContext,
    SOULMemory,
    MemoryType,
    get_context_retriever,
    shutdown_retriever,
)

logger = logging.getLogger(__name__)


class MemoryModule:
    """
    Central manager for the memory subsystem.
    Enforces strict memory bounds and manages lifecycle.
    """

    # Strict memory limit: 200MB for vector database
    MAX_MEMORY_MB = 200
    MAX_MEMORY_BYTES = MAX_MEMORY_MB * 1024 * 1024

    def __init__(
        self,
        index_path: Optional[str] = None,
        storage_path: Optional[str] = None,
        use_gpu: bool = False,
    ):
        self.index_path = index_path
        self.storage_path = storage_path
        self.use_gpu = use_gpu
        self._index_manager: Optional[FAISSIndexManager] = None
        self._vector_storage: Optional[VectorStorage] = None
        self._retriever: Optional[ContextRetriever] = None
        self._initialized = False

    def initialize(self) -> bool:
        """Initialize all memory components."""
        if self._initialized:
            return True

        try:
            # Initialize FAISS index manager
            self._index_manager = get_index_manager(
                index_path=self.index_path,
                use_gpu=self.use_gpu,
            )

            # Initialize vector storage if path provided
            if self.storage_path:
                self._vector_storage = get_vector_storage(
                    path=self.storage_path,
                    max_vectors=FAISSIndexManager.MAX_VECTORS,
                )

            # Initialize retriever
            self._retriever = ContextRetriever(
                index_manager=self._index_manager,
                vector_storage=self._vector_storage,
            )

            self._initialized = True
            logger.info(f"Memory module initialized (max {self.MAX_MEMORY_MB}MB)")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize memory module: {e}")
            return False

    async def store_market_state(
        self,
        feature_vector: Any,
        market_data: Dict[str, Any],
    ) -> int:
        """
        Store a market state memory.

        Args:
            feature_vector: Feature vector (array-like)
            market_data: Market data metadata

        Returns:
            Memory ID
        """
        if not self._retriever:
            raise RuntimeError("Memory module not initialized")

        import numpy as np
        if not isinstance(feature_vector, np.ndarray):
            feature_vector = np.array(feature_vector, dtype=np.float32)

        return self._retriever.store_memory(
            feature_vector=feature_vector,
            memory_type=MemoryType.MARKET_STATE,
            metadata={"market_data": market_data},
        )

    async def retrieve_context(
        self,
        ipc_data: Dict[str, Any],
        k: int = 10,
    ) -> list:
        """
        Retrieve context for current state.

        Args:
            ipc_data: IPC data from Rust
            k: Number of results

        Returns:
            List of retrieved contexts
        """
        if not self._retriever:
            raise RuntimeError("Memory module not initialized")

        return self._retriever.retrieve_similar_states(ipc_data, k=k)

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        stats = {
            "initialized": self._initialized,
            "max_memory_mb": self.MAX_MEMORY_MB,
        }

        if self._index_manager:
            stats["index"] = self._index_manager.get_stats()

        if self._retriever:
            stats["retriever"] = self._retriever.get_stats()

        # Calculate actual memory usage
        if self._index_manager:
            idx_stats = self._index_manager.get_stats()
            stats["current_memory_mb"] = idx_stats.get("memory_usage_bytes", 0) / (1024 * 1024)
            stats["memory_utilization_pct"] = idx_stats.get("utilization_pct", 0)

        return stats

    def check_memory_bounds(self) -> bool:
        """Verify we're within memory bounds."""
        if not self._index_manager:
            return True

        stats = self._index_manager.get_stats()
        current = stats.get("memory_usage_bytes", 0)
        return current <= self.MAX_MEMORY_BYTES

    async def save_checkpoint(self, path: Optional[str] = None) -> bool:
        """Save memory state to disk."""
        if not self._index_manager:
            return False

        save_path = path or self.index_path
        if save_path:
            return self._index_manager.save_index(save_path)
        return False

    async def load_checkpoint(self, path: Optional[str] = None) -> bool:
        """Load memory state from disk."""
        if not self._index_manager:
            return False

        load_path = path or self.index_path
        if load_path and os.path.exists(load_path):
            return self._index_manager.load_index(load_path)
        return False

    async def cleanup(self):
        """Cleanup and release resources."""
        await shutdown_faiss()
        await shutdown_retriever()
        self._initialized = False
        logger.info("Memory module cleaned up")


# Module singleton
_module: Optional[MemoryModule] = None


def get_memory_module(
    index_path: Optional[str] = None,
    storage_path: Optional[str] = None,
    use_gpu: bool = False,
) -> MemoryModule:
    """Get or create the memory module singleton."""
    global _module
    if _module is None:
        _module = MemoryModule(
            index_path=index_path,
            storage_path=storage_path,
            use_gpu=use_gpu,
        )
        _module.initialize()
    return _module


async def initialize_memory(
    index_path: Optional[str] = None,
    storage_path: Optional[str] = None,
    use_gpu: bool = False,
) -> MemoryModule:
    """Initialize the memory module."""
    module = get_memory_module(index_path, storage_path, use_gpu)
    if not module._initialized:
        module.initialize()
    return module


async def shutdown_memory_module():
    """Gracefully shutdown the memory module."""
    global _module
    if _module:
        await _module.cleanup()
        _module = None
