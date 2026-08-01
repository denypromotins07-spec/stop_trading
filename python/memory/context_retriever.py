"""
Context Retriever for RAG-style Memory Retrieval.
Non-LLM retriever that queries FAISS index using IPC feature vectors.
Fetches historical SOUL.md memories and past trade outcomes.
"""

import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
import logging
import time
from enum import Enum

from .faiss_index import (
    FAISSIndexManager,
    VectorStorage,
    MemoryEntry,
    get_index_manager,
    get_vector_storage,
)

logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """Types of stored memories."""
    MARKET_STATE = "market_state"
    TRADE_OUTCOME = "trade_outcome"
    ORDER_BOOK_SHAPE = "order_book_shape"
    EXECUTION_PATTERN = "execution_pattern"
    RISK_EVENT = "risk_event"


@dataclass
class RetrievedContext:
    """Retrieved context from memory."""
    memory_type: MemoryType
    similarity_score: float
    feature_vector: np.ndarray
    metadata: Dict[str, Any]
    timestamp_ns: int
    relevance_weight: float = 1.0


@dataclass
class SOULMemory:
    """SOUL.md compatible memory structure."""
    id: str
    content: str
    embedding: np.ndarray
    tags: List[str]
    created_at_ns: int
    trade_outcome: Optional[Dict[str, Any]] = None
    market_conditions: Optional[Dict[str, Any]] = None


class ContextRetriever:
    """
    Non-LLM, RAG-style context retriever for historical market state recall.
    Queries FAISS index using current IPC feature vectors.
    """

    # Similarity threshold for considering a memory relevant
    SIMILARITY_THRESHOLD = 0.7

    # Max results to return
    DEFAULT_K = 10

    def __init__(
        self,
        index_manager: Optional[FAISSIndexManager] = None,
        vector_storage: Optional[VectorStorage] = None,
    ):
        self.index_manager = index_manager or get_index_manager()
        self.vector_storage = vector_storage
        self._memory_cache: Dict[int, SOULMemory] = {}
        self._retrieval_count = 0

    def encode_feature_vector(
        self,
        ipc_data: Dict[str, Any],
    ) -> np.ndarray:
        """
        Encode IPC data into a feature vector for similarity search.

        Args:
            ipc_data: Raw IPC data from Rust side

        Returns:
            Normalized feature vector
        """
        # Extract relevant features from IPC data
        features = []

        # Order book features
        if "orderbook" in ipc_data:
            ob = ipc_data["orderbook"]
            features.extend(ob.get("bid_prices", [])[:10])
            features.extend(ob.get("bid_sizes", [])[:10])
            features.extend(ob.get("ask_prices", [])[:10])
            features.extend(ob.get("ask_sizes", [])[:10])

        # Market indicators
        if "indicators" in ipc_data:
            ind = ipc_data["indicators"]
            features.extend([
                ind.get("rsi", 50),
                ind.get("macd", 0),
                ind.get("volatility", 0),
                ind.get("spread_bps", 10),
            ])

        # Position features
        if "position" in ipc_data:
            pos = ipc_data["position"]
            features.extend([
                pos.get("delta", 0),
                pos.get("gamma", 0),
                pos.get("pnl", 0),
            ])

        # Pad or truncate to dimension
        target_dim = self.index_manager.dimension
        if len(features) < target_dim:
            features.extend([0.0] * (target_dim - len(features)))
        else:
            features = features[:target_dim]

        vector = np.array(features, dtype=np.float32)

        # Normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        return vector

    def retrieve_similar_states(
        self,
        ipc_data: Dict[str, Any],
        k: int = DEFAULT_K,
        memory_types: Optional[List[MemoryType]] = None,
    ) -> List[RetrievedContext]:
        """
        Retrieve similar historical market states.

        Args:
            ipc_data: Current IPC data
            k: Number of results
            memory_types: Filter by memory types

        Returns:
            List of RetrievedContext objects
        """
        # Encode current state
        query_vector = self.encode_feature_vector(ipc_data)

        # Search FAISS index
        distances, indices = self.index_manager.search(query_vector, k)

        results = []
        for dist, idx in zip(distances, indices):
            if idx < 0 or dist < self.SIMILARITY_THRESHOLD:
                continue

            # Retrieve full vector if storage available
            feature_vec = None
            if self.vector_storage:
                feature_vec = self.vector_storage.retrieve(idx)

            # Get metadata
            metadata = self._get_metadata(idx)

            # Filter by memory type if specified
            mem_type_str = metadata.get("memory_type", "")
            if memory_types:
                try:
                    mem_type = MemoryType(mem_type_str)
                    if mem_type not in memory_types:
                        continue
                except ValueError:
                    pass

            context = RetrievedContext(
                memory_type=MemoryType(mem_type_str) if mem_type_str else MemoryType.MARKET_STATE,
                similarity_score=float(dist),
                feature_vector=feature_vec if feature_vec is not None else query_vector,
                metadata=metadata,
                timestamp_ns=metadata.get("timestamp_ns", time.time_ns()),
                relevance_weight=self._calculate_relevance(dist, metadata),
            )
            results.append(context)

        self._retrieval_count += 1
        logger.debug(f"Retrieved {len(results)} similar states")
        return results

    def retrieve_trade_outcomes(
        self,
        current_state: Dict[str, Any],
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve similar past trade outcomes for RL conditioning.

        Args:
            current_state: Current market state
            k: Number of outcomes to retrieve

        Returns:
            List of trade outcome dictionaries
        """
        contexts = self.retrieve_similar_states(
            ipc_data=current_state,
            k=k * 2,  # Get more to filter
            memory_types=[MemoryType.TRADE_OUTCOME],
        )

        outcomes = []
        for ctx in contexts[:k]:
            if "trade_outcome" in ctx.metadata:
                outcome = {
                    "similarity": ctx.similarity_score,
                    "outcome": ctx.metadata["trade_outcome"],
                    "market_conditions": ctx.metadata.get("market_conditions", {}),
                    "action_taken": ctx.metadata.get("action"),
                    "pnl": ctx.metadata.get("pnl", 0),
                }
                outcomes.append(outcome)

        return outcomes

    def store_memory(
        self,
        feature_vector: np.ndarray,
        memory_type: MemoryType,
        metadata: Dict[str, Any],
    ) -> int:
        """
        Store a new memory entry.

        Args:
            feature_vector: Feature vector
            memory_type: Type of memory
            metadata: Additional metadata

        Returns:
            Vector ID
        """
        # Ensure normalized
        norm = np.linalg.norm(feature_vector)
        if norm > 0:
            feature_vector = feature_vector / norm

        # Add timestamp if not present
        if "timestamp_ns" not in metadata:
            metadata["timestamp_ns"] = time.time_ns()

        metadata["memory_type"] = memory_type.value

        # Add to index
        ids = self.index_manager.add_vectors(
            vectors=feature_vector.reshape(1, -1),
            metadata_list=[metadata],
        )

        # Store exact vector
        if self.vector_storage:
            self.vector_storage.store(ids[0], feature_vector)

        logger.debug(f"Stored {memory_type.value} memory with ID {ids[0]}")
        return ids[0]

    def store_trade_outcome(
        self,
        feature_vector: np.ndarray,
        outcome: Dict[str, Any],
        action: str,
        pnl: float,
    ) -> int:
        """
        Store a trade outcome memory.

        Args:
            feature_vector: State vector at trade time
            outcome: Trade outcome details
            action: Action taken
            pnl: PnL result

        Returns:
            Memory ID
        """
        metadata = {
            "trade_outcome": outcome,
            "action": action,
            "pnl": pnl,
            "memory_type": MemoryType.TRADE_OUTCOME.value,
        }

        return self.store_memory(
            feature_vector=feature_vector,
            memory_type=MemoryType.TRADE_OUTCOME,
            metadata=metadata,
        )

    def _get_metadata(self, vector_id: int) -> Dict[str, Any]:
        """Get metadata for a vector ID."""
        # In production, this would query persistent storage
        return {}

    def _calculate_relevance(
        self,
        similarity: float,
        metadata: Dict[str, Any],
    ) -> float:
        """
        Calculate relevance weight for a retrieved memory.

        Factors:
        - Similarity score
        - Recency
        - Historical accuracy
        """
        base_weight = similarity

        # Recency bonus (newer memories slightly preferred)
        ts = metadata.get("timestamp_ns", 0)
        age_seconds = (time.time_ns() - ts) / 1e9
        recency_factor = max(0.5, 1.0 - (age_seconds / 86400))  # Decay over 24h

        # Accuracy bonus (if historical prediction was accurate)
        accuracy = metadata.get("prediction_accuracy", 1.0)

        return base_weight * recency_factor * accuracy

    def build_context_for_rl(
        self,
        current_state: Dict[str, Any],
        max_memories: int = 20,
    ) -> Dict[str, Any]:
        """
        Build complete context dictionary for RL agent conditioning.

        Args:
            current_state: Current market state
            max_memories: Maximum memories to include

        Returns:
            Context dictionary for RL agent
        """
        # Retrieve various memory types
        similar_states = self.retrieve_similar_states(
            current_state,
            k=max_memories // 2,
        )

        trade_outcomes = self.retrieve_trade_outcomes(
            current_state,
            k=max_memories // 4,
        )

        # Aggregate statistics from retrieved memories
        if trade_outcomes:
            avg_pnl = sum(o["pnl"] for o in trade_outcomes) / len(trade_outcomes)
            win_rate = sum(1 for o in trade_outcomes if o["pnl"] > 0) / len(trade_outcomes)
        else:
            avg_pnl = 0.0
            win_rate = 0.5

        context = {
            "similar_states": [
                {
                    "similarity": ctx.similarity_score,
                    "type": ctx.memory_type.value,
                    "metadata": ctx.metadata,
                }
                for ctx in similar_states
            ],
            "historical_outcomes": trade_outcomes,
            "aggregate_stats": {
                "avg_historical_pnl": avg_pnl,
                "historical_win_rate": win_rate,
                "retrieval_confidence": similar_states[0].similarity_score if similar_states else 0.0,
            },
            "total_retrievals": self._retrieval_count,
        }

        return context

    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics."""
        index_stats = self.index_manager.get_stats()
        return {
            **index_stats,
            "retrieval_count": self._retrieval_count,
            "cached_memories": len(self._memory_cache),
        }


# Module singleton
_retriever: Optional[ContextRetriever] = None


def get_context_retriever() -> ContextRetriever:
    """Get or create the context retriever singleton."""
    global _retriever
    if _retriever is None:
        _retriever = ContextRetriever()
    return _retriever


async def shutdown_retriever():
    """Shutdown the retriever."""
    global _retriever
    _retriever = None
