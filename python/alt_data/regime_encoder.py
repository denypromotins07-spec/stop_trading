# Regime Encoder for HMM Market Regimes
# Converts discrete regime states into dense continuous vectors for RL conditioning

from __future__ import annotations
import logging
import numpy as np
from typing import Optional, Dict, Any, List

log = logging.getLogger(__name__)


class RegimeEncoder:
    """
    Embedding encoder for HMM market regimes.
    Converts discrete regime states (Trending, Mean-Reverting, High-Vol) 
    into dense continuous vectors for RL agent conditioning.
    
    Uses learned embeddings with position encoding for temporal context.
    """

    # Regime type constants
    REGIME_UNKNOWN = 0
    REGIME_TRENDING_UP = 1
    REGIME_TRENDING_DOWN = 2
    REGIME_MEAN_REVERTING = 3
    REGIME_HIGH_VOL = 4
    REGIME_LOW_VOL = 5
    REGIME_TRANSITIONING = 6
    
    REGIME_NAMES = {
        REGIME_UNKNOWN: "UNKNOWN",
        REGIME_TRENDING_UP: "TRENDING_UP",
        REGIME_TRENDING_DOWN: "TRENDING_DOWN",
        REGIME_MEAN_REVERTING: "MEAN_REVERTING",
        REGIME_HIGH_VOL: "HIGH_VOL",
        REGIME_LOW_VOL: "LOW_VOL",
        REGIME_TRANSITIONING: "TRANSITIONING",
    }

    def __init__(
        self,
        embedding_dim: int = 32,
        n_regimes: int = 7,
        context_length: int = 20,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.n_regimes = n_regimes
        self.context_length = context_length
        
        # Initialize regime embeddings (learned parameters)
        # In production, these would be trained; here we use structured initialization
        self._regime_embeddings = self._initialize_embeddings()
        
        # Position encoding for temporal ordering
        self._position_encoding = self._create_position_encoding(context_length, embedding_dim)
        
        # Regime history buffer (circular)
        self._history = np.zeros(context_length, dtype=np.int32)
        self._history_confidence = np.zeros(context_length, dtype=np.float64)
        self._head = 0
        self._count = 0
        
        # Current encoded state
        self._current_embedding: Optional[np.ndarray] = None
        self._current_context: Optional[np.ndarray] = None

    def _initialize_embeddings(self) -> np.ndarray:
        """
        Initialize regime embeddings with structured values.
        Each regime gets a distinct embedding direction.
        """
        embeddings = np.zeros((self.n_regimes, self.embedding_dim), dtype=np.float64)
        
        # Create orthogonal-ish embeddings for different regimes
        for i in range(self.n_regimes):
            angle = 2 * np.pi * i / self.n_regimes
            # Primary dimensions encode regime type
            embeddings[i, 0] = np.cos(angle)
            embeddings[i, 1] = np.sin(angle)
            # Secondary dimensions encode volatility characteristics
            if i in [self.REGIME_HIGH_VOL]:
                embeddings[i, 2:6] = 1.0
            elif i in [self.REGIME_LOW_VOL]:
                embeddings[i, 2:6] = -1.0
            # Trend direction in dimensions 6-10
            if i == self.REGIME_TRENDING_UP:
                embeddings[i, 6:11] = 1.0
            elif i == self.REGIME_TRENDING_DOWN:
                embeddings[i, 6:11] = -1.0
            # Mean-reverting signature
            if i == self.REGIME_MEAN_REVERTING:
                embeddings[i, 10:15] = 0.5
                embeddings[i, 15:20] = -0.5
        
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeddings = embeddings / norms
        
        return embeddings

    def _create_position_encoding(self, length: int, dim: int) -> np.ndarray:
        """Create sinusoidal position encodings."""
        pos_enc = np.zeros((length, dim), dtype=np.float64)
        position = np.arange(length).reshape(-1, 1)
        div_term = np.exp(np.arange(0, dim, 2) * -(np.log(10000.0) / dim))
        
        pos_enc[:, 0::2] = np.sin(position * div_term)
        pos_enc[:, 1::2] = np.cos(position * div_term)
        
        return pos_enc

    def update_regime(
        self,
        regime_type: int,
        confidence: float = 1.0,
        ts_event: Optional[int] = None,
    ) -> np.ndarray:
        """
        Update current regime and compute new embedding.
        
        Args:
            regime_type: One of REGIME_* constants
            confidence: Confidence score 0.0 to 1.0
            ts_event: Optional timestamp
            
        Returns:
            Current regime embedding vector
        """
        # Store in history buffer
        self._history[self._head] = regime_type
        self._history_confidence[self._head] = confidence
        self._head = (self._head + 1) % self.context_length
        self._count = min(self._count + 1, self.context_length)
        
        # Get base embedding for current regime
        base_embedding = self._regime_embeddings[regime_type].copy()
        
        # Scale by confidence
        base_embedding *= confidence
        
        # Compute context-aware embedding
        context_embedding = self._compute_context_embedding()
        
        # Combine base and context
        self._current_embedding = 0.7 * base_embedding + 0.3 * context_embedding
        
        # Add position encoding for temporal awareness
        position_idx = self._head % self.context_length
        self._current_embedding += 0.1 * self._position_encoding[position_idx]
        
        return self._current_embedding

    def _compute_context_embedding(self) -> np.ndarray:
        """Compute embedding from recent regime history."""
        if self._count == 0:
            return np.zeros(self.embedding_dim, dtype=np.float64)
        
        # Get valid history
        if self._head >= self._count:
            history = self._history[self._head - self._count:self._head]
            confidences = self._history_confidence[self._head - self._count:self._head]
        else:
            # Wrap around
            history = np.concatenate([
                self._history[self._head:],
                self._history[:self._head]
            ])
            confidences = np.concatenate([
                self._history_confidence[self._head:],
                self._history_confidence[:self._head]
            ])
        
        # Weighted average of historical embeddings
        context_emb = np.zeros(self.embedding_dim, dtype=np.float64)
        total_weight = 0.0
        
        for i, (regime, conf) in enumerate(zip(history, confidences)):
            weight = conf * (0.9 ** (len(history) - i - 1))  # Exponential decay
            context_emb += weight * self._regime_embeddings[regime]
            total_weight += weight
        
        if total_weight > 0:
            context_emb /= total_weight
        
        return context_emb

    def get_regime_onehot(self, regime_type: int) -> np.ndarray:
        """Get one-hot encoding for a regime type."""
        onehot = np.zeros(self.n_regimes, dtype=np.float64)
        onehot[regime_type] = 1.0
        return onehot

    def get_transition_features(self) -> Dict[str, float]:
        """
        Compute features describing regime transitions.
        Useful for RL state representation.
        """
        if self._count < 2:
            return {"transition_prob": 0.0, "regime_stability": 1.0}
        
        # Get recent history
        if self._head >= 2:
            recent = self._history[self._head - 2:self._head]
        else:
            recent = self._history[-2:]
        
        # Transition detection
        is_transition = recent[0] != recent[1]
        
        # Stability score (inverse of transition frequency)
        transitions = np.sum(np.diff(self._history[:self._count]) != 0)
        stability = 1.0 - (transitions / max(self._count - 1, 1))
        
        return {
            "is_transition": float(is_transition),
            "transition_prob": float(transitions / max(self._count - 1, 1)),
            "regime_stability": stability,
            "current_regime": float(recent[-1]),
            "previous_regime": float(recent[-2]) if len(recent) > 1 else float(recent[-1]),
        }

    def get_full_state(self) -> np.ndarray:
        """
        Get full regime state for RL agent.
        Concatenates current embedding, context, and transition features.
        """
        if self._current_embedding is None:
            self.update_regime(self.REGIME_UNKNOWN)
        
        trans_features = list(self.get_transition_features().values())
        
        return np.concatenate([
            self._current_embedding,
            trans_features,
        ])

    def get_regime_name(self, regime_type: int) -> str:
        """Get human-readable regime name."""
        return self.REGIME_NAMES.get(regime_type, "UNKNOWN")

    def decode_embedding(self, embedding: np.ndarray) -> int:
        """
        Decode an embedding back to nearest regime type.
        Useful for interpreting model outputs.
        """
        distances = np.linalg.norm(self._regime_embeddings - embedding, axis=1)
        return int(np.argmin(distances))

    def reset(self) -> None:
        """Reset all state."""
        self._history.fill(0)
        self._history_confidence.fill(0.0)
        self._head = 0
        self._count = 0
        self._current_embedding = None
        self._current_context = None
        log.info("RegimeEncoder reset")


def create_regime_encoder(
    embedding_dim: int = 32,
    context_length: int = 20,
) -> RegimeEncoder:
    """Factory function to create a configured regime encoder."""
    return RegimeEncoder(
        embedding_dim=embedding_dim,
        context_length=context_length,
    )


def regime_to_tensor(
    encoder: RegimeEncoder,
    regime_sequence: List[int],
    confidences: Optional[List[float]] = None,
) -> np.ndarray:
    """
    Convert a sequence of regimes to tensor for batch processing.
    
    Args:
        encoder: RegimeEncoder instance
        regime_sequence: List of regime type integers
        confidences: Optional list of confidence scores
        
    Returns:
        Tensor of shape (sequence_length, embedding_dim)
    """
    if confidences is None:
        confidences = [1.0] * len(regime_sequence)
    
    embeddings = []
    for regime, conf in zip(regime_sequence, confidences):
        emb = encoder.update_regime(regime, conf)
        embeddings.append(emb.copy())
    
    return np.stack(embeddings, axis=0)
