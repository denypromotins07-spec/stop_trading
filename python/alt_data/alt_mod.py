# Alternative Data Module Root
# Injects alternative data and regime embeddings into central feature store

from __future__ import annotations
import logging
from typing import Optional, Dict, Any, List

log = logging.getLogger(__name__)

from python.alt_data.macro_ingestor import (
    MacroDataIngestor,
    MacroEventData,
    FundingRateData,
    WhaleAlertData,
)
from python.alt_data.regime_encoder import (
    RegimeEncoder,
    create_regime_encoder,
    regime_to_tensor,
)


class AlternativeDataManager:
    """
    Central manager for alternative data streams.
    Coordinates macro ingestor and regime encoder, injects into feature store.
    """

    def __init__(
        self,
        embedding_dim: int = 32,
        context_length: int = 20,
    ) -> None:
        self.regime_encoder = create_regime_encoder(
            embedding_dim=embedding_dim,
            context_length=context_length,
        )
        
        # Feature buffer for combined alt data
        self._feature_buffer: Dict[str, Any] = {}
        self._last_update_ts: Optional[int] = None
        
        # Statistics
        self._macro_count = 0
        self._funding_updates = 0
        self._whale_alerts_count = 0
        self._regime_updates = 0

    def process_macro_event(self, event: MacroEventData) -> Dict[str, float]:
        """Process macro event and extract features."""
        features = {
            "macro_impact_score": self._score_macro_impact(event),
            "macro_surprise": self._compute_surprise(event.actual, event.forecast),
            "is_high_impact": 1.0 if event.impact == "HIGH" else 0.0,
        }
        
        self._feature_buffer["macro"] = features
        self._macro_count += 1
        self._last_update_ts = event.ts_event
        
        log.debug(f"Processed macro event: {event.event_type} (impact={features['macro_impact_score']})")
        return features

    def process_funding_rate(self, data: FundingRateData) -> Dict[str, float]:
        """Process funding rate and extract features."""
        features = {
            "funding_rate": data.funding_rate,
            "funding_predicted": data.predicted_rate,
            "funding_annualized": data.annualized_rate,
            "funding_time_ratio": data.time_to_next / 28800.0,  # Normalize to 8h
            "funding_extreme": 1.0 if abs(data.funding_rate) > 0.001 else 0.0,
        }
        
        self._feature_buffer["funding"] = features
        self._funding_updates += 1
        self._last_update_ts = data.ts_event
        
        log.debug(f"Processed funding rate: {data.instrument_id} = {data.funding_rate}")
        return features

    def process_whale_alert(self, alert: WhaleAlertData) -> Dict[str, float]:
        """Process whale alert and extract features."""
        features = {
            "whale_value_usd": alert.value_usd,
            "whale_amount": alert.amount,
            "is_exchange_flow": 1.0 if alert.exchange else 0.0,
            "flow_direction": self._decode_flow_direction(alert.transaction_type),
            "whale_significance": min(alert.value_usd / 1_000_000, 10.0),  # Cap at 10M
        }
        
        self._feature_buffer["whale"] = features
        self._whale_alerts_count += 1
        self._last_update_ts = alert.ts_event
        
        log.debug(f"Processed whale alert: ${alert.value_usd:,.0f} {alert.symbol}")
        return features

    def update_regime(
        self,
        regime_type: int,
        confidence: float = 1.0,
    ) -> np.ndarray:
        """Update regime state and get embedding."""
        import numpy as np
        
        embedding = self.regime_encoder.update_regime(regime_type, confidence)
        self._feature_buffer["regime_embedding"] = embedding
        self._feature_buffer["regime_features"] = self.regime_encoder.get_transition_features()
        self._regime_updates += 1
        
        log.debug(f"Updated regime: {self.regime_encoder.get_regime_name(regime_type)}")
        return embedding

    def get_combined_features(self) -> Dict[str, Any]:
        """Get all current alternative data features."""
        return {
            **self._feature_buffer,
            "counts": {
                "macro": self._macro_count,
                "funding": self._funding_updates,
                "whale": self._whale_alerts_count,
                "regime": self._regime_updates,
            },
            "last_update_ts": self._last_update_ts,
        }

    def get_feature_vector(self) -> np.ndarray:
        """
        Flatten all features into a single vector for ML models.
        """
        import numpy as np
        
        vectors = []
        
        # Regime embedding (always present)
        if "regime_embedding" in self._feature_buffer:
            vectors.append(self._feature_buffer["regime_embedding"])
        else:
            vectors.append(np.zeros(self.regime_encoder.embedding_dim))
        
        # Regime transition features
        if "regime_features" in self._feature_buffer:
            trans = self._feature_buffer["regime_features"]
            vectors.append(np.array(list(trans.values())))
        
        # Macro features
        if "macro" in self._feature_buffer:
            macro = self._feature_buffer["macro"]
            vectors.append(np.array(list(macro.values())))
        
        # Funding features
        if "funding" in self._feature_buffer:
            funding = self._feature_buffer["funding"]
            vectors.append(np.array(list(funding.values())))
        
        # Whale features
        if "whale" in self._feature_buffer:
            whale = self._feature_buffer["whale"]
            vectors.append(np.array(list(whale.values())))
        
        return np.concatenate(vectors)

    def _score_macro_impact(self, event: MacroEventData) -> float:
        """Score macro event impact based on type and surprise."""
        base_scores = {
            "NFP": 1.0,
            "CPI": 0.9,
            "FOMC": 0.95,
            "GDP": 0.7,
            "RETAIL": 0.5,
            "PMI": 0.6,
        }
        
        base = base_scores.get(event.event_type.upper(), 0.4)
        
        # Adjust by impact level
        impact_multipliers = {"LOW": 0.3, "MEDIUM": 0.6, "HIGH": 1.0}
        base *= impact_multipliers.get(event.impact, 0.5)
        
        # Adjust by surprise magnitude
        surprise = self._compute_surprise(event.actual, event.forecast)
        base *= (1.0 + min(abs(surprise), 2.0))
        
        return min(base, 3.0)

    def _compute_surprise(
        self,
        actual: Optional[float],
        forecast: Optional[float],
    ) -> float:
        """Compute surprise as (actual - forecast) / |forecast|."""
        if actual is None or forecast is None or forecast == 0:
            return 0.0
        return (actual - forecast) / abs(forecast)

    def _decode_flow_direction(self, transaction_type: str) -> float:
        """Decode transaction type to flow direction (-1 to 1)."""
        directions = {
            "EXCHANGE_IN": -1.0,   # Selling pressure
            "EXCHANGE_OUT": 1.0,   # Buying/holding pressure
            "TRANSFER": 0.0,
            "MINING": 0.5,         # New supply
        }
        return directions.get(transaction_type.upper(), 0.0)

    def reset(self) -> None:
        """Reset all state."""
        self._feature_buffer.clear()
        self._last_update_ts = None
        self._macro_count = 0
        self._funding_updates = 0
        self._whale_alerts_count = 0
        self._regime_updates = 0
        self.regime_encoder.reset()
        log.info("AlternativeDataManager reset")


# Global instance
_manager: Optional[AlternativeDataManager] = None


def get_manager() -> AlternativeDataManager:
    """Get or create global alternative data manager."""
    global _manager
    if _manager is None:
        _manager = AlternativeDataManager()
    return _manager


def initialize_manager(
    embedding_dim: int = 32,
    context_length: int = 20,
) -> AlternativeDataManager:
    """Initialize global manager with custom parameters."""
    global _manager
    _manager = AlternativeDataManager(
        embedding_dim=embedding_dim,
        context_length=context_length,
    )
    return _manager


__all__ = [
    "MacroDataIngestor",
    "MacroEventData",
    "FundingRateData",
    "WhaleAlertData",
    "RegimeEncoder",
    "AlternativeDataManager",
    "get_manager",
    "initialize_manager",
    "create_regime_encoder",
    "regime_to_tensor",
]
