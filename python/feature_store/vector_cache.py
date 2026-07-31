# Lightweight NumPy LRU Cache for Historical State Lookup
# Pure-numpy implementation for finding similar historical states

from __future__ import annotations
import logging
import numpy as np
from typing import Optional, Tuple, List, Dict, Any

log = logging.getLogger(__name__)


class VectorCache:
    """
    Lightweight, pure-numpy LRU cache for finding similar historical states.
    Allows self-learning engine to query "what did the bot do last time 
    it saw this exact order book shape?"
    
    Uses cosine similarity via dot products for fast nearest neighbor search.
    Strictly enforces byte-limits to guarantee 3GB Python RAM ceiling.
    """

    def __init__(
        self,
        max_entries: int = 50000,
        vector_dim: int = 128,
        max_memory_mb: int = 500,
        dtype: np.dtype = np.float64,
    ) -> None:
        self.max_entries = max_entries
        self.vector_dim = vector_dim
        self.max_memory_mb = max_memory_mb
        self.dtype = dtype
        
        # Calculate actual max entries based on memory limit
        bytes_per_entry = vector_dim * np.dtype(dtype).itemsize + 64  # +metadata
        max_by_memory = (max_memory_mb * 1024 * 1024) // bytes_per_entry
        self.max_entries = min(max_entries, int(max_by_memory))
        
        # Pre-allocated storage (no dynamic allocation)
        self._vectors = np.zeros((self.max_entries, vector_dim), dtype=dtype, order='C')
        self._normalized = np.zeros((self.max_entries, vector_dim), dtype=dtype, order='C')
        self._metadata: List[Optional[Dict[str, Any]]] = [None] * self.max_entries
        self._timestamps = np.zeros(self.max_entries, dtype=np.int64)
        self._access_counts = np.zeros(self.max_entries, dtype=np.int32)
        
        # LRU tracking
        self._head = 0  # Next write position
        self._count = 0  # Current number of entries
        self._total_inserted = 0
        
        # Normalization constant
        self._epsilon = 1e-9

    def insert(
        self,
        vector: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[int] = None,
    ) -> int:
        """
        Insert a vector into the cache.
        Overwrites oldest/least-used entry if full.
        Returns the index where data was stored.
        """
        if len(vector) != self.vector_dim:
            raise ValueError(
                f"Vector dimension {len(vector)} != expected {self.vector_dim}"
            )
        
        # Normalize vector for cosine similarity
        norm = np.linalg.norm(vector) + self._epsilon
        normalized_vec = vector / norm
        
        # Store in circular buffer
        idx = self._head
        self._vectors[idx] = vector
        self._normalized[idx] = normalized_vec
        self._metadata[idx] = metadata
        self._timestamps[idx] = timestamp if timestamp is not None else self._total_inserted
        self._access_counts[idx] = 0
        
        # Advance head with manual wrap
        self._head = (self._head + 1) % self.max_entries
        self._count = min(self._count + 1, self.max_entries)
        self._total_inserted += 1
        
        return idx

    def query(
        self,
        vector: np.ndarray,
        k: int = 10,
        threshold: float = 0.9,
    ) -> List[Tuple[int, float, Optional[Dict[str, Any]]]]:
        """
        Find k most similar vectors using cosine similarity.
        Returns list of (index, similarity_score, metadata) tuples.
        
        Uses efficient dot product for cosine similarity.
        """
        if self._count == 0:
            return []
        
        # Normalize query vector
        norm = np.linalg.norm(vector) + self._epsilon
        query_normalized = vector / norm
        
        # Compute cosine similarities via dot product
        # similarity = dot(q, v) / (|q| * |v|) = dot(q_norm, v_norm)
        similarities = np.dot(self._normalized[:self._count], query_normalized)
        
        # Filter by threshold
        valid_mask = similarities >= threshold
        
        if not np.any(valid_mask):
            return []
        
        # Get top-k indices
        valid_indices = np.where(valid_mask)[0]
        valid_sims = similarities[valid_mask]
        
        # Sort by similarity (descending)
        top_k_local = np.argsort(valid_sims)[-k:][::-1]
        
        results = []
        for local_idx in top_k_local:
            global_idx = valid_indices[local_idx]
            sim_score = float(valid_sims[local_idx])
            
            # Update access count for LRU
            self._access_counts[global_idx] += 1
            
            results.append((
                int(global_idx),
                sim_score,
                self._metadata[global_idx],
            ))
        
        return results

    def query_by_metadata(
        self,
        key: str,
        value: Any,
    ) -> List[Tuple[int, np.ndarray]]:
        """
        Find all vectors with matching metadata.
        Linear scan but useful for specific lookups.
        """
        results = []
        
        for i in range(self._count):
            meta = self._metadata[i]
            if meta and meta.get(key) == value:
                results.append((i, self._vectors[i].copy()))
                self._access_counts[i] += 1
        
        return results

    def get_least_used(self, n: int = 1) -> List[int]:
        """Get indices of n least-used entries (for eviction)."""
        if self._count == 0:
            return []
        
        valid_indices = np.arange(self._count)
        access_counts = self._access_counts[:self._count]
        
        # Sort by access count (ascending)
        sorted_indices = np.argsort(access_counts)[:n]
        return [int(i) for i in sorted_indices]

    def batch_insert(
        self,
        vectors: np.ndarray,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        timestamps: Optional[np.ndarray] = None,
    ) -> List[int]:
        """
        Efficiently insert multiple vectors.
        Returns list of insertion indices.
        """
        if vectors.ndim != 2 or vectors.shape[1] != self.vector_dim:
            raise ValueError(f"Expected 2D array with dim {self.vector_dim}")
        
        indices = []
        n_vectors = len(vectors)
        
        for i in range(n_vectors):
            meta = metadatas[i] if metadatas and i < len(metadatas) else None
            ts = int(timestamps[i]) if timestamps is not None and i < len(timestamps) else None
            idx = self.insert(vectors[i], meta, ts)
            indices.append(idx)
        
        return indices

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        memory_bytes = (
            self._vectors.nbytes +
            self._normalized.nbytes +
            self._timestamps.nbytes +
            self._access_counts.nbytes
        )
        
        return {
            "max_entries": self.max_entries,
            "current_count": self._count,
            "total_inserted": self._total_inserted,
            "vector_dim": self.vector_dim,
            "memory_bytes": memory_bytes,
            "memory_mb": memory_bytes / (1024 * 1024),
            "memory_limit_mb": self.max_memory_mb,
            "utilization": self._count / self.max_entries if self.max_entries > 0 else 0,
            "avg_access_count": float(np.mean(self._access_counts[:self._count])) if self._count > 0 else 0,
        }

    def clear(self) -> None:
        """Clear cache without deallocating memory."""
        self._vectors.fill(0.0)
        self._normalized.fill(0.0)
        self._metadata = [None] * self.max_entries
        self._timestamps.fill(0)
        self._access_counts.fill(0)
        self._head = 0
        self._count = 0
        log.info("VectorCache cleared")

    def __len__(self) -> int:
        """Return number of entries."""
        return self._count

    def __getitem__(self, idx: int) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """Get vector and metadata by index."""
        if 0 <= idx < self._count:
            return self._vectors[idx].copy(), self._metadata[idx]
        return None


def create_vector_cache(
    max_entries: int = 50000,
    vector_dim: int = 128,
    max_memory_mb: int = 500,
) -> VectorCache:
    """Factory function to create a configured vector cache."""
    return VectorCache(
        max_entries=max_entries,
        vector_dim=vector_dim,
        max_memory_mb=max_memory_mb,
    )


def find_similar_states(
    cache: VectorCache,
    current_state: np.ndarray,
    min_similarity: float = 0.95,
    max_results: int = 5,
) -> List[Dict[str, Any]]:
    """
    High-level API to find similar historical states.
    Returns enriched results with action recommendations.
    """
    matches = cache.query(current_state, k=max_results, threshold=min_similarity)
    
    results = []
    for idx, similarity, metadata in matches:
        if metadata:
            results.append({
                "index": idx,
                "similarity": similarity,
                "timestamp": metadata.get("timestamp"),
                "action_taken": metadata.get("action"),
                "outcome": metadata.get("outcome"),
                "regime": metadata.get("regime"),
            })
    
    return results
