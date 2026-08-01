"""
NLP Module Root - Blends lexicon polarity scores with semantic embeddings.
Feeds the alpha ensemble with combined sentiment features.
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from dataclasses import dataclass, field
import threading
import time

from .lexicon_scorer import LexiconScorer, SentimentScore, get_scorer
from .embedding_cache import EmbeddingCache, get_embedder


@dataclass
class NLPFeatures:
    """Combined NLP features for alpha generation."""
    
    # Lexicon-based features
    polarity: float = 0.0
    subjectivity: float = 0.0
    positive_score: float = 0.0
    negative_score: float = 0.0
    uncertainty_score: float = 0.0
    litigious_score: float = 0.0
    
    # Embedding-based features
    embedding: np.ndarray = field(default_factory=lambda: np.zeros(384))
    regime_similarity: float = 0.0  # Similarity to current regime context
    sentiment_momentum: float = 0.0  # Change in sentiment over time
    
    # Metadata
    timestamp: float = 0.0
    text_length: int = 0
    source_count: int = 0
    
    def to_vector(self) -> np.ndarray:
        """Convert features to a flat feature vector."""
        return np.array([
            self.polarity,
            self.subjectivity,
            self.positive_score,
            self.negative_score,
            self.uncertainty_score,
            self.litigious_score,
            self.regime_similarity,
            self.sentiment_momentum,
            self.text_length / 1000.0,  # Normalize
            self.source_count / 10.0  # Normalize
        ], dtype=np.float32)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "polarity": self.polarity,
            "subjectivity": self.subjectivity,
            "positive_score": self.positive_score,
            "negative_score": self.negative_score,
            "uncertainty_score": self.uncertainty_score,
            "litigious_score": self.litigious_score,
            "regime_similarity": self.regime_similarity,
            "sentiment_momentum": self.sentiment_momentum,
            "timestamp": self.timestamp,
            "text_length": self.text_length,
            "source_count": self.source_count,
            "embedding_dim": len(self.embedding)
        }


class NLPFeatureExtractor:
    """
    Combines lexicon scoring and semantic embeddings for comprehensive NLP features.
    Optimized for low-latency HFT environments.
    """
    
    def __init__(
        self,
        lexicon_scorer: Optional[LexiconScorer] = None,
        embedder: Optional[EmbeddingCache] = None,
        regime_context_window: int = 100
    ):
        self.scorer = lexicon_scorer or get_scorer()
        self.embedder = embedder or get_embedder()
        
        # Regime context tracking
        self._regime_context: List[np.ndarray] = []
        self._regime_window = regime_context_window
        self._current_regime_embedding: Optional[np.ndarray] = None
        
        # Sentiment history for momentum calculation
        self._sentiment_history: List[Tuple[float, float]] = []  # (timestamp, polarity)
        self._history_window = 50
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Performance metrics
        self._total_processed = 0
        self._total_latency_ms = 0.0
    
    def update_regime_context(self, embedding: np.ndarray) -> None:
        """Update the rolling regime context embedding."""
        with self._lock:
            self._regime_context.append(embedding.copy())
            
            if len(self._regime_context) > self._regime_window:
                self._regime_context.pop(0)
            
            # Recompute average regime embedding
            if self._regime_context:
                self._current_regime_embedding = np.mean(self._regime_context, axis=0)
    
    def _compute_regime_similarity(self, embedding: np.ndarray) -> float:
        """Compute similarity to current regime context."""
        if self._current_regime_embedding is None:
            return 0.5  # Neutral if no context
        
        # Cosine similarity
        dot_product = np.dot(embedding, self._current_regime_embedding)
        norm1 = np.linalg.norm(embedding)
        norm2 = np.linalg.norm(self._current_regime_embedding)
        
        if norm1 > 0 and norm2 > 0:
            return float(dot_product / (norm1 * norm2))
        return 0.5
    
    def _compute_sentiment_momentum(self, current_polarity: float, timestamp: float) -> float:
        """Compute sentiment momentum from recent history."""
        with self._lock:
            self._sentiment_history.append((timestamp, current_polarity))
            
            # Trim history
            while len(self._sentiment_history) > self._history_window:
                self._sentiment_history.pop(0)
            
            if len(self._sentiment_history) < 2:
                return 0.0
            
            # Calculate linear regression slope (momentum)
            times = np.array([t for t, _ in self._sentiment_history])
            polarities = np.array([p for _, p in self._sentiment_history])
            
            # Normalize time
            times_norm = (times - times[0]) / max(times[-1] - times[0] + 1e-6, 1e-6)
            
            # Simple linear regression
            n = len(times_norm)
            sum_x = np.sum(times_norm)
            sum_y = np.sum(polarities)
            sum_xy = np.sum(times_norm * polarities)
            sum_xx = np.sum(times_norm ** 2)
            
            denominator = n * sum_xx - sum_x ** 2
            if abs(denominator) < 1e-10:
                return 0.0
            
            slope = (n * sum_xy - sum_x * sum_y) / denominator
            return float(slope)
    
    def extract_features(self, text: str, timestamp: Optional[float] = None) -> NLPFeatures:
        """
        Extract comprehensive NLP features from text.
        Combines lexicon scoring with semantic embeddings.
        """
        start_time = time.perf_counter()
        
        if timestamp is None:
            timestamp = time.time()
        
        # Get lexicon scores
        sentiment_score = self.scorer.score_text(text)
        
        # Get embedding
        embedding = self.embedder.generate_embedding(text)
        
        # Compute regime similarity
        regime_sim = self._compute_regime_similarity(embedding)
        
        # Compute sentiment momentum
        momentum = self._compute_sentiment_momentum(sentiment_score.polarity, timestamp)
        
        # Update regime context with new embedding
        self.update_regime_context(embedding)
        
        # Build feature object
        features = NLPFeatures(
            polarity=sentiment_score.polarity,
            subjectivity=sentiment_score.subjectivity,
            positive_score=sentiment_score.positive,
            negative_score=sentiment_score.negative,
            uncertainty_score=sentiment_score.uncertainty,
            litigious_score=sentiment_score.litigious,
            embedding=embedding,
            regime_similarity=regime_sim,
            sentiment_momentum=momentum,
            timestamp=timestamp,
            text_length=len(text),
            source_count=1
        )
        
        # Track performance
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self._total_processed += 1
        self._total_latency_ms += elapsed_ms
        
        return features
    
    def extract_batch_features(
        self,
        texts: List[str],
        timestamps: Optional[List[float]] = None
    ) -> List[NLPFeatures]:
        """Extract features for multiple texts efficiently."""
        if timestamps is None:
            timestamps = [time.time()] * len(texts)
        
        features_list = []
        for text, ts in zip(texts, timestamps):
            features = self.extract_features(text, ts)
            features_list.append(features)
        
        return features_list
    
    def aggregate_features(
        self,
        features_list: List[NLPFeatures],
        weights: Optional[np.ndarray] = None
    ) -> NLPFeatures:
        """Aggregate multiple NLP features into a single representation."""
        if not features_list:
            return NLPFeatures()
        
        n = len(features_list)
        if weights is None:
            weights = np.ones(n) / n
        else:
            weights = weights / np.sum(weights)
        
        # Aggregate scalar features
        avg_polarity = sum(f.polarity * w for f, w in zip(features_list, weights))
        avg_subjectivity = sum(f.subjectivity * w for f, w in zip(features_list, weights))
        avg_positive = sum(f.positive_score * w for f, w in zip(features_list, weights))
        avg_negative = sum(f.negative_score * w for f, w in zip(features_list, weights))
        avg_uncertainty = sum(f.uncertainty_score * w for f, w in zip(features_list, weights))
        avg_litigious = sum(f.litigious_score * w for f, w in zip(features_list, weights))
        avg_regime_sim = sum(f.regime_similarity * w for f, w in zip(features_list, weights))
        avg_momentum = sum(f.sentiment_momentum * w for f, w in zip(features_list, weights))
        
        # Aggregate embeddings (weighted average)
        aggregated_embedding = sum(w * f.embedding for f, w in zip(features_list, weights))
        
        # Normalize aggregated embedding
        norm = np.linalg.norm(aggregated_embedding)
        if norm > 0:
            aggregated_embedding = aggregated_embedding / norm
        
        # Aggregate metadata
        total_text_length = sum(f.text_length for f in features_list)
        total_sources = sum(f.source_count for f in features_list)
        latest_timestamp = max(f.timestamp for f in features_list)
        
        return NLPFeatures(
            polarity=avg_polarity,
            subjectivity=avg_subjectivity,
            positive_score=avg_positive,
            negative_score=avg_negative,
            uncertainty_score=avg_uncertainty,
            litigious_score=avg_litigious,
            embedding=aggregated_embedding,
            regime_similarity=avg_regime_sim,
            sentiment_momentum=avg_momentum,
            timestamp=latest_timestamp,
            text_length=total_text_length,
            source_count=total_sources
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Return performance statistics."""
        with self._lock:
            avg_latency = (
                self._total_latency_ms / max(self._total_processed, 1)
            )
            return {
                "total_processed": self._total_processed,
                "average_latency_ms": round(avg_latency, 3),
                "total_latency_ms": round(self._total_latency_ms, 3),
                "regime_context_size": len(self._regime_context),
                "sentiment_history_size": len(self._sentiment_history)
            }
    
    def reset(self) -> None:
        """Reset all state."""
        with self._lock:
            self._regime_context.clear()
            self._sentiment_history.clear()
            self._current_regime_embedding = None
            self._total_processed = 0
            self._total_latency_ms = 0.0


# Global singleton instance
_nlp_instance: Optional[NLPFeatureExtractor] = None
_instance_lock = threading.Lock()


def get_nlp_extractor() -> NLPFeatureExtractor:
    """Get or create the global NLP feature extractor."""
    global _nlp_instance
    if _nlp_instance is None:
        with _instance_lock:
            if _nlp_instance is None:
                _nlp_instance = NLPFeatureExtractor()
    return _nlp_instance


def extract_nlp_features(text: str) -> Dict[str, Any]:
    """Convenience function for quick feature extraction."""
    features = get_nlp_extractor().extract_features(text)
    return features.to_dict()


if __name__ == "__main__":
    # Test the NLP module
    extractor = NLPFeatureExtractor()
    
    test_texts = [
        "Bitcoin surges past resistance as institutional demand grows",
        "Market uncertainty rises amid Fed policy speculation",
        "Crypto regulations tighten globally, causing sell-off pressure"
    ]
    
    print("Testing NLP Feature Extraction:")
    features_list = []
    for i, text in enumerate(test_texts):
        features = extractor.extract_features(text, timestamp=time.time() + i)
        features_list.append(features)
        
        print(f"\nText: {text}")
        print(f"Polarity: {features.polarity:.4f}")
        print(f"Subjectivity: {features.subjectivity:.4f}")
        print(f"Regime Similarity: {features.regime_similarity:.4f}")
        print(f"Sentiment Momentum: {features.sentiment_momentum:.4f}")
    
    # Test aggregation
    print("\n\nAggregated Features:")
    aggregated = extractor.aggregate_features(features_list)
    print(f"Average Polarity: {aggregated.polarity:.4f}")
    print(f"Average Subjectivity: {aggregated.subjectivity:.4f}")
    print(f"Combined Embedding Shape: {aggregated.embedding.shape}")
    
    # Performance stats
    print(f"\nPerformance Stats: {extractor.get_performance_stats()}")
