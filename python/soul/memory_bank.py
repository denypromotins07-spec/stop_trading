"""
Lightweight in-memory vector index for historical regime memories.
Uses numpy dot products for fast similarity search and context retrieval.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import get_logger

logger = get_logger("memory_bank")


class RegimeMemory:
    """Represents a single regime memory entry."""
    
    def __init__(
        self,
        regime_id: int,
        features: np.ndarray,
        outcome: str,
        timestamp: float,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.regime_id = regime_id
        self.features = features.astype(np.float64)
        self.outcome = outcome
        self.timestamp = timestamp
        self.metadata = metadata or {}
        
        # Normalize features for cosine similarity
        norm = np.linalg.norm(self.features)
        self.normalized_features = (
            self.features / norm if norm > 0 else self.features
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'regime_id': self.regime_id,
            'features': self.features.tolist(),
            'outcome': self.outcome,
            'timestamp': self.timestamp,
            'metadata': self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RegimeMemory":
        """Create from dictionary representation."""
        return cls(
            regime_id=data['regime_id'],
            features=np.array(data['features'], dtype=np.float64),
            outcome=data['outcome'],
            timestamp=data['timestamp'],
            metadata=data.get('metadata'),
        )


class MemoryBank:
    """
    In-memory vector index for regime memories.
    Uses numpy dot products for O(1) similarity computation.
    """
    
    def __init__(self, max_memories: int = 10000):
        self.max_memories = max_memories
        self._memories: List[RegimeMemory] = []
        self._feature_matrix: Optional[np.ndarray] = None
        self._memory_index: Dict[int, int] = {}  # regime_id -> index
        self._is_dirty = False
    
    def add_memory(self, memory: RegimeMemory) -> int:
        """
        Add a memory to the bank.
        
        Args:
            memory: RegimeMemory to add
        
        Returns:
            Index of the added memory
        """
        # Check if regime_id already exists (update case)
        if memory.regime_id in self._memory_index:
            idx = self._memory_index[memory.regime_id]
            self._memories[idx] = memory
            self._is_dirty = True
            logger.debug(f"Updated memory for regime {memory.regime_id}")
            return idx
        
        # Add new memory
        if len(self._memories) >= self.max_memories:
            # Remove oldest memory (FIFO)
            self._memories.pop(0)
            self._memory_index = {
                m.regime_id: i for i, m in enumerate(self._memories)
            }
            self._is_dirty = True
        
        self._memories.append(memory)
        idx = len(self._memories) - 1
        self._memory_index[memory.regime_id] = idx
        self._is_dirty = True
        
        logger.debug(f"Added memory for regime {memory.regime_id} at index {idx}")
        return idx
    
    def add_memories_batch(self, memories: List[RegimeMemory]) -> int:
        """
        Add multiple memories at once.
        
        Args:
            memories: List of RegimeMemory objects
        
        Returns:
            Number of memories added
        """
        count = 0
        for memory in memories:
            self.add_memory(memory)
            count += 1
        
        if count > 0:
            logger.info(f"Added {count} memories to bank")
        
        return count
    
    def _rebuild_feature_matrix(self) -> None:
        """Rebuild the feature matrix from memories."""
        if not self._memories:
            self._feature_matrix = None
            return
        
        # Stack all normalized features into a matrix
        features = np.vstack([m.normalized_features for m in self._memories])
        self._feature_matrix = features
        self._is_dirty = False
        
        logger.debug(f"Rebuilt feature matrix: shape={features.shape}")
    
    def query_similar(
        self,
        query_features: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.7,
    ) -> List[Tuple[RegimeMemory, float]]:
        """
        Find most similar memories using cosine similarity.
        
        Args:
            query_features: Feature vector to match
            top_k: Number of results to return
            threshold: Minimum similarity threshold
        
        Returns:
            List of (RegimeMemory, similarity_score) tuples
        """
        if not self._memories:
            return []
        
        # Rebuild matrix if dirty
        if self._is_dirty or self._feature_matrix is None:
            self._rebuild_feature_matrix()
        
        if self._feature_matrix is None:
            return []
        
        # Normalize query
        query_norm = np.linalg.norm(query_features)
        query_normalized = (
            query_features / query_norm if query_norm > 0 else query_features
        )
        
        # Compute cosine similarities via dot product (vectorized)
        similarities = np.dot(self._feature_matrix, query_normalized)
        
        # Get top-k indices above threshold
        valid_mask = similarities >= threshold
        if not np.any(valid_mask):
            return []
        
        # Sort by similarity (descending)
        valid_indices = np.where(valid_mask)[0]
        valid_scores = similarities[valid_mask]
        sorted_order = np.argsort(valid_scores)[::-1][:top_k]
        
        results = []
        for idx in sorted_order:
            actual_idx = valid_indices[idx]
            memory = self._memories[actual_idx]
            score = float(similarities[actual_idx])
            results.append((memory, score))
        
        logger.debug(
            f"Query returned {len(results)} similar memories "
            f"(best score: {results[0][1] if results else 0:.4f})"
        )
        
        return results
    
    def get_memory_by_regime_id(self, regime_id: int) -> Optional[RegimeMemory]:
        """Get a specific memory by regime ID."""
        idx = self._memory_index.get(regime_id)
        if idx is not None:
            return self._memories[idx]
        return None
    
    def get_all_memories(self) -> List[RegimeMemory]:
        """Get all memories in the bank."""
        return list(self._memories)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the memory bank."""
        if not self._memories:
            return {
                'total_memories': 0,
                'unique_regimes': 0,
                'feature_dimension': 0,
            }
        
        outcomes = {}
        for m in self._memories:
            outcomes[m.outcome] = outcomes.get(m.outcome, 0) + 1
        
        return {
            'total_memories': len(self._memories),
            'unique_regimes': len(self._memory_index),
            'feature_dimension': len(self._memories[0].features),
            'outcome_distribution': outcomes,
            'oldest_timestamp': min(m.timestamp for m in self._memories),
            'newest_timestamp': max(m.timestamp for m in self._memories),
        }
    
    def clear(self) -> None:
        """Clear all memories."""
        self._memories.clear()
        self._memory_index.clear()
        self._feature_matrix = None
        self._is_dirty = True
        logger.info("Memory bank cleared")


class SOULMemoryBank:
    """
    Integrates MemoryBank with SOUL.md regime memory parsing.
    Automatically populates memories from SOUL.md entries.
    """
    
    def __init__(self, max_memories: int = 10000):
        self.memory_bank = MemoryBank(max_memories=max_memories)
        self._parsed_regime_count = 0
    
    def parse_and_add_regime_memories(
        self,
        soul_data: Dict[str, Any],
        feature_extractor: Optional[callable] = None,
    ) -> int:
        """
        Parse regime memories from SOUL.md data and add to bank.
        
        Args:
            soul_data: Parsed SOUL.md data from ledger_parser
            feature_extractor: Optional function to extract features from text
        
        Returns:
            Number of memories added
        """
        regime_texts = soul_data.get('regime_memories', [])
        
        if not regime_texts:
            return 0
        
        memories_added = 0
        
        for i, text in enumerate(regime_texts):
            try:
                # Create a simple feature vector from text (can be enhanced)
                if feature_extractor:
                    features = feature_extractor(text)
                else:
                    # Default: hash-based feature extraction
                    features = self._text_to_features(text)
                
                memory = RegimeMemory(
                    regime_id=self._parsed_regime_count,
                    features=features,
                    outcome="UNKNOWN",  # Can be parsed from text
                    timestamp=time.time(),
                    metadata={'raw_text': text},
                )
                
                self.memory_bank.add_memory(memory)
                memories_added += 1
                self._parsed_regime_count += 1
                
            except Exception as e:
                logger.error(f"Failed to parse regime memory: {e}")
        
        if memories_added > 0:
            logger.info(f"Added {memories_added} regime memories from SOUL.md")
        
        return memories_added
    
    def _text_to_features(self, text: str, dim: int = 64) -> np.ndarray:
        """
        Convert text to a feature vector using simple hashing.
        This is a placeholder - can be replaced with embeddings.
        
        Args:
            text: Text to convert
            dim: Output dimension
        
        Returns:
            Feature vector
        """
        # Simple hash-based feature extraction
        features = np.zeros(dim, dtype=np.float64)
        
        words = text.lower().split()
        for word in words:
            h = hash(word) % dim
            features[h] += 1
        
        # Normalize
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        
        return features
    
    def find_similar_regime(
        self,
        current_features: np.ndarray,
        threshold: float = 0.7,
    ) -> Optional[Dict[str, Any]]:
        """
        Find similar historical regimes for context.
        
        Args:
            current_features: Current market features
            threshold: Similarity threshold
        
        Returns:
            Dictionary with similar regime info or None
        """
        results = self.memory_bank.query_similar(
            current_features, top_k=1, threshold=threshold
        )
        
        if not results:
            return None
        
        memory, score = results[0]
        
        return {
            'regime_id': memory.regime_id,
            'outcome': memory.outcome,
            'similarity': score,
            'metadata': memory.metadata,
            'timestamp': memory.timestamp,
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory bank statistics."""
        return self.memory_bank.get_statistics()


# Global memory bank instance
_memory_bank_instance: Optional[SOULMemoryBank] = None


def get_memory_bank() -> SOULMemoryBank:
    """Get or create the global memory bank instance."""
    global _memory_bank_instance
    if _memory_bank_instance is None:
        _memory_bank_instance = SOULMemoryBank()
    return _memory_bank_instance


def query_regime_context(
    features: np.ndarray,
    threshold: float = 0.7,
) -> Optional[Dict[str, Any]]:
    """
    Convenience function to query regime context.
    
    Args:
        features: Current feature vector
        threshold: Similarity threshold
    
    Returns:
        Similar regime info or None
    """
    bank = get_memory_bank()
    return bank.find_similar_regime(features, threshold)
