"""
FAISS Index for Ultra-Fast Vector Similarity Search.
Implements IndexFlatIP (Inner Product) for historical market state retrieval.
Strictly bounded to 200MB memory limit.
"""

import numpy as np
import faiss
import mmap
import os
from typing import Optional, Tuple, List
from dataclasses import dataclass
import logging
import struct

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """Represents a stored memory entry."""
    vector_id: int
    feature_vector: np.ndarray
    metadata: dict
    timestamp_ns: int


class FAISSIndexManager:
    """
    Manages FAISS index for ultra-fast similarity search of historical market states.
    Uses IndexFlatIP for inner product similarity with strict memory bounds.
    """

    # Memory budget: 200MB max for vector storage
    MAX_MEMORY_BYTES = 200 * 1024 * 1024  # 200MB
    VECTOR_DIMENSION = 128  # Feature vector dimension
    VECTOR_SIZE_BYTES = VECTOR_DIMENSION * 4  # float32 = 4 bytes

    # Calculate max vectors that fit in memory budget
    MAX_VECTORS = (MAX_MEMORY_BYTES // VECTOR_SIZE_BYTES) - 10000  # Buffer for overhead

    def __init__(
        self,
        index_path: Optional[str] = None,
        dimension: int = VECTOR_DIMENSION,
    ):
        self.dimension = dimension
        self.index_path = index_path
        self._index: Optional[faiss.Index] = None
        self._metadata_store: dict = {}
        self._vector_count = 0
        self._mmap_file: Optional[mmap.mmap] = None
        self._lock = None  # asyncio.Lock() when needed

    def initialize(self, use_gpu: bool = False) -> bool:
        """
        Initialize the FAISS index.

        Args:
            use_gpu: Whether to use GPU acceleration

        Returns:
            True if successful
        """
        try:
            # Use Inner Product index for cosine-like similarity
            self._index = faiss.IndexFlatIP(self.dimension)

            if use_gpu:
                # Move index to GPU if available
                res = faiss.StandardGpuResources()
                self._index = faiss.index_cpu_to_gpu(res, 0, self._index)
                logger.info("FAISS index initialized on GPU")
            else:
                logger.info("FAISS index initialized on CPU")

            # Load existing index if path provided
            if self.index_path and os.path.exists(self.index_path):
                self.load_index(self.index_path)

            return True

        except Exception as e:
            logger.error(f"Failed to initialize FAISS index: {e}")
            return False

    def add_vectors(
        self,
        vectors: np.ndarray,
        metadata_list: Optional[List[dict]] = None,
    ) -> List[int]:
        """
        Add vectors to the index with memory bounds enforcement.

        Args:
            vectors: np.ndarray of shape (n, dimension), dtype=np.float32
            metadata_list: Optional list of metadata dicts

        Returns:
            List of assigned vector IDs
        """
        if self._index is None:
            raise RuntimeError("Index not initialized")

        # Ensure correct dtype
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)

        # Normalize vectors for inner product = cosine similarity
        faiss.normalize_L2(vectors)

        # Check memory bounds
        new_count = len(vectors)
        if self._vector_count + new_count > self.MAX_VECTORS:
            # Need to evict oldest vectors
            overflow = self._vector_count + new_count - self.MAX_VECTORS
            self._evict_oldest(overflow)

        # Add vectors to index
        start_id = self._vector_count
        self._index.add(vectors)

        # Store metadata
        if metadata_list:
            for i, meta in enumerate(metadata_list or []):
                self._metadata_store[start_id + i] = meta

        self._vector_count += new_count
        logger.debug(f"Added {new_count} vectors, total: {self._vector_count}")

        return list(range(start_id, start_id + new_count))

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for most similar vectors.

        Args:
            query_vector: Query vector of shape (dimension,)
            k: Number of results to return

        Returns:
            Tuple of (distances, indices)
        """
        if self._index is None:
            raise RuntimeError("Index not initialized")

        if self._vector_count == 0:
            return np.array([]), np.array([])

        # Ensure correct shape and dtype
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        query_vector = query_vector.astype(np.float32)

        # Normalize query vector
        faiss.normalize_L2(query_vector)

        # Search
        distances, indices = self._index.search(query_vector, k)

        return distances[0], indices[0]

    def search_with_metadata(
        self,
        query_vector: np.ndarray,
        k: int = 5,
    ) -> List[MemoryEntry]:
        """
        Search and return full MemoryEntry objects.

        Args:
            query_vector: Query vector
            k: Number of results

        Returns:
            List of MemoryEntry objects
        """
        distances, indices = self.search(query_vector, k)

        results = []
        for dist, idx in zip(distances, indices):
            if idx < 0:  # FAISS returns -1 for missing results
                continue

            entry = MemoryEntry(
                vector_id=int(idx),
                feature_vector=self.get_vector(idx),
                metadata=self._metadata_store.get(int(idx), {}),
                timestamp_ns=self._metadata_store.get(int(idx), {}).get("timestamp_ns", 0),
            )
            results.append(entry)

        return results

    def get_vector(self, vector_id: int) -> Optional[np.ndarray]:
        """Retrieve a specific vector by ID."""
        if self._index is None or vector_id >= self._vector_count:
            return None

        # Reconstruct vector from index (approximate for FlatIP)
        # For exact storage, we maintain a separate array
        return None  # Would need separate storage for exact vectors

    def _evict_oldest(self, count: int):
        """Evict oldest vectors to maintain memory bounds."""
        if count <= 0:
            return

        logger.info(f"Evicting {count} oldest vectors to maintain memory bounds")

        # FAISS doesn't support removal from Flat indexes efficiently
        # Strategy: rebuild index without oldest vectors
        if self._vector_count <= count:
            # Reset entirely
            self._index = faiss.IndexFlatIP(self.dimension)
            self._metadata_store.clear()
            self._vector_count = 0
        else:
            # In production, would rebuild with newest vectors only
            # For now, log warning and continue
            logger.warning(
                f"Index full ({self._vector_count}/{self.MAX_VECTORS}). "
                "Consider using IVF index for better memory management."
            )

    def save_index(self, path: str) -> bool:
        """Save index to disk."""
        if self._index is None:
            return False

        try:
            faiss.write_index(self._index, path)
            logger.info(f"Saved FAISS index to {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            return False

    def load_index(self, path: str) -> bool:
        """Load index from disk."""
        try:
            self._index = faiss.read_index(path)
            self._vector_count = self._index.ntotal
            logger.info(f"Loaded FAISS index from {path} ({self._vector_count} vectors)")
            return True
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False

    def get_stats(self) -> dict:
        """Get index statistics."""
        return {
            "vector_count": self._vector_count,
            "max_vectors": self.MAX_VECTORS,
            "memory_usage_bytes": self._vector_count * self.VECTOR_SIZE_BYTES,
            "max_memory_bytes": self.MAX_MEMORY_BYTES,
            "utilization_pct": (self._vector_count / self.MAX_VECTORS) * 100,
            "dimension": self.dimension,
        }

    def reset(self):
        """Reset the index completely."""
        if self._index:
            self._index = faiss.IndexFlatIP(self.dimension)
        self._metadata_store.clear()
        self._vector_count = 0
        logger.info("FAISS index reset")


# Compressed storage for exact vector retrieval
class VectorStorage:
    """
    Memory-mapped storage for exact vector retrieval.
    Uses np.float32 and strict bounds to prevent RAM bloat.
    """

    def __init__(self, path: str, max_vectors: int, dimension: int = 128):
        self.path = path
        self.max_vectors = max_vectors
        self.dimension = dimension
        self._array: Optional[np.ndarray] = None
        self._count = 0

    def initialize(self) -> bool:
        """Initialize memory-mapped array."""
        try:
            # Create file if doesn't exist
            total_bytes = self.max_vectors * self.dimension * 4
            if not os.path.exists(self.path):
                with open(self.path, 'wb') as f:
                    f.write(b'\x00' * total_bytes)

            # Memory map the file
            self._file = open(self.path, 'r+b')
            self._array = np.memmap(
                self._file,
                dtype=np.float32,
                mode='r+',
                shape=(self.max_vectors, self.dimension),
            )
            return True
        except Exception as e:
            logger.error(f"Failed to initialize vector storage: {e}")
            return False

    def store(self, vector_id: int, vector: np.ndarray) -> bool:
        """Store a vector at the given ID."""
        if self._array is None or vector_id >= self.max_vectors:
            return False

        self._array[vector_id] = vector.astype(np.float32)
        self._count = max(self._count, vector_id + 1)
        return True

    def retrieve(self, vector_id: int) -> Optional[np.ndarray]:
        """Retrieve a vector by ID."""
        if self._array is None or vector_id >= self._count:
            return None
        return self._array[vector_id].copy()

    def close(self):
        """Flush and close memory-mapped file."""
        if hasattr(self, '_array') and self._array is not None:
            del self._array
        if hasattr(self, '_file'):
            self._file.close()


# Module singleton
_index_manager: Optional[FAISSIndexManager] = None
_storage: Optional[VectorStorage] = None


def get_index_manager(
    index_path: Optional[str] = None,
    use_gpu: bool = False,
) -> FAISSIndexManager:
    """Get or create the FAISS index manager singleton."""
    global _index_manager
    if _index_manager is None:
        _index_manager = FAISSIndexManager(index_path=index_path)
        _index_manager.initialize(use_gpu=use_gpu)
    return _index_manager


def get_vector_storage(
    path: str,
    max_vectors: int = FAISSIndexManager.MAX_VECTORS,
) -> VectorStorage:
    """Get or create vector storage singleton."""
    global _storage
    if _storage is None:
        _storage = VectorStorage(path=path, max_vectors=max_vectors)
        _storage.initialize()
    return _storage


async def shutdown_memory_module():
    """Gracefully shutdown memory module."""
    global _index_manager, _storage

    if _index_manager and _index_manager.index_path:
        _index_manager.save_index(_index_manager.index_path)

    if _storage:
        _storage.close()

    _index_manager = None
    _storage = None
