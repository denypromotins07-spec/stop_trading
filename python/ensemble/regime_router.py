"""
Deterministic Regime Router
Stage 49: Maps HMM regime states and meta-learner confidence to strategy actors.
Activates Trend strategies during high-momentum regimes, MM during mean-reverting states.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import logging
import zmq

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classifications from HMM."""
    BULL_HIGH_MOMENTUM = "bull_high_momentum"
    BEAR_HIGH_VOLATILITY = "bear_high_volatility"
    MEAN_REVERTING_LOW_VOL = "mean_reverting_low_vol"
    TRANSITION_CHOPPY = "transition_choppy"
    CRASH_PANIC = "crash_panic"


class StrategyType(Enum):
    """Strategy types that can be activated."""
    TREND = "trend"
    STATARB = "statarb"
    MARKET_MAKING = "market_making"
    MOMENTUM = "momentum"
    ARBITRAGE = "arbitrage"


@dataclass
class RegimeState:
    """Current market regime state."""
    regime: MarketRegime
    confidence: float
    volatility: float
    momentum: float
    correlation_level: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class StrategyActivation:
    """Strategy activation decision."""
    strategy_type: StrategyType
    strategy_id: str
    activated: bool
    allocation_weight: float
    reason: str


class RegimeRouter:
    """
    Deterministic routing matrix mapping HMM regime states to strategy actors.
    Ensures optimal strategy selection based on current market conditions.
    """
    
    # Routing matrix: regime -> preferred strategies with base weights
    ROUTING_MATRIX = {
        MarketRegime.BULL_HIGH_MOMENTUM: {
            StrategyType.TREND: 0.6,
            StrategyType.MOMENTUM: 0.3,
            StrategyType.STATARb: 0.1,
            StrategyType.MARKET_MAKING: 0.0,
            StrategyType.ARBITRAGE: 0.0,
        },
        MarketRegime.BEAR_HIGH_VOLATILITY: {
            StrategyType.TREND: 0.3,
            StrategyType.MOMENTUM: 0.2,
            StrategyType.STATARb: 0.2,
            StrategyType.MARKET_MAKING: 0.1,
            StrategyType.ARBITRAGE: 0.2,
        },
        MarketRegime.MEAN_REVERTING_LOW_VOL: {
            StrategyType.TREND: 0.0,
            StrategyType.MOMENTUM: 0.1,
            StrategyType.STATARb: 0.5,
            StrategyType.MARKET_MAKING: 0.4,
            StrategyType.ARBITRAGE: 0.0,
        },
        MarketRegime.TRANSITION_CHOPPY: {
            StrategyType.TREND: 0.1,
            StrategyType.MOMENTUM: 0.1,
            StrategyType.STATARb: 0.3,
            StrategyType.MARKET_MAKING: 0.3,
            StrategyType.ARBITRAGE: 0.2,
        },
        MarketRegime.CRASH_PANIC: {
            StrategyType.TREND: 0.0,
            StrategyType.MOMENTUM: 0.0,
            StrategyType.STATARb: 0.1,
            StrategyType.MARKET_MAKING: 0.0,
            StrategyType.ARBITRAGE: 0.0,
            # Rest goes to cash/hedging
        },
    }
    
    def __init__(self, 
                 min_confidence_threshold: float = 0.6,
                 max_strategy_count: int = 5):
        
        self.min_confidence_threshold = min_confidence_threshold
        self.max_strategy_count = max_strategy_count
        
        # Current regime state
        self.current_regime: Optional[RegimeState] = None
        self.regime_history: List[RegimeState] = []
        
        # Active strategies
        self.active_strategies: Dict[str, StrategyActivation] = {}
        self.registered_strategies: Dict[str, StrategyType] = {}
        
        # ZMQ socket for Rust IPC
        self._zmq_context = zmq.Context()
        self._zmq_socket = self._zmq_context.socket(zmq.PUSH)
        self._zmq_socket.connect("tcp://localhost:5561")
        
        # Pre-allocated numpy arrays for performance
        self._weights_buffer = np.zeros(10, dtype=np.float32)
        self._regime_probs = np.zeros(5, dtype=np.float32)
    
    def register_strategy(self, strategy_id: str, strategy_type: StrategyType) -> bool:
        """Register a strategy for potential activation."""
        if strategy_id in self.registered_strategies:
            logger.warning(f"Strategy {strategy_id} already registered")
            return False
        
        self.registered_strategies[strategy_id] = strategy_type
        logger.info(f"Registered strategy {strategy_id} ({strategy_type.value})")
        return True
    
    def update_regime(self, 
                     regime: MarketRegime,
                     confidence: float,
                     volatility: float,
                     momentum: float,
                     correlation_level: float) -> List[StrategyActivation]:
        """
        Update current regime and recalculate strategy activations.
        
        Returns list of activation changes.
        """
        # Create new regime state
        new_state = RegimeState(
            regime=regime,
            confidence=confidence,
            volatility=volatility,
            momentum=momentum,
            correlation_level=correlation_level,
        )
        
        # Check if regime changed significantly
        regime_changed = (
            self.current_regime is None or 
            self.current_regime.regime != regime or
            abs(self.current_regime.confidence - confidence) > 0.2
        )
        
        self.current_regime = new_state
        self.regime_history.append(new_state)
        
        # Limit history size
        if len(self.regime_history) > 100:
            self.regime_history.pop(0)
        
        if regime_changed:
            logger.info(f"Regime changed to {regime.value} (confidence: {confidence:.2f})")
            activations = self._recalculate_activations()
            self._notify_regime_change(new_state, activations)
            return activations
        
        return []
    
    def _recalculate_activations(self) -> List[StrategyActivation]:
        """Recalculate strategy activations based on current regime."""
        if self.current_regime is None:
            return []
        
        regime = self.current_regime.regime
        confidence = self.current_regime.confidence
        
        # Get base weights from routing matrix
        base_weights = self.ROUTING_MATRIX.get(regime, {})
        
        # Adjust weights by confidence
        adjusted_weights = {
            stype: weight * confidence 
            for stype, weight in base_weights.items()
        }
        
        # Normalize weights
        total = sum(adjusted_weights.values())
        if total > 0:
            adjusted_weights = {k: v / total for k, v in adjusted_weights.items()}
        
        # Generate activations for registered strategies
        activations = []
        for strategy_id, strategy_type in self.registered_strategies.items():
            base_weight = adjusted_weights.get(strategy_type, 0.0)
            
            # Apply additional filters based on regime characteristics
            weight = self._apply_regime_filters(strategy_type, base_weight)
            
            # Determine activation
            activated = weight > 0.05 and confidence >= self.min_confidence_threshold
            
            # Check if this is a change
            old_activation = self.active_strategies.get(strategy_id)
            if old_activation is None or old_activation.activated != activated:
                activation = StrategyActivation(
                    strategy_type=strategy_type,
                    strategy_id=strategy_id,
                    activated=activated,
                    allocation_weight=weight,
                    reason=f"Regime: {regime.value}, confidence: {confidence:.2f}",
                )
                activations.append(activation)
                self.active_strategies[strategy_id] = activation
            elif activated:
                # Update weight for already active strategy
                old_activation.allocation_weight = weight
        
        return activations
    
    def _apply_regime_filters(self, 
                             strategy_type: StrategyType, 
                             base_weight: float) -> float:
        """Apply additional filters based on regime characteristics."""
        if self.current_regime is None:
            return base_weight
        
        weight = base_weight
        
        # High volatility filter - reduce MM strategies
        if self.current_regime.volatility > 0.8:
            if strategy_type == StrategyType.MARKET_MAKING:
                weight *= 0.3
        
        # Low momentum filter - reduce trend strategies
        if abs(self.current_regime.momentum) < 0.2:
            if strategy_type in (StrategyType.TREND, StrategyType.MOMENTUM):
                weight *= 0.5
        
        # High correlation filter - reduce statarb
        if self.current_regime.correlation_level > 0.8:
            if strategy_type == StrategyType.STATARb:
                weight *= 0.2
        
        # Crash regime - almost everything off
        if self.current_regime.regime == MarketRegime.CRASH_PANIC:
            weight *= 0.1
        
        return weight
    
    def get_active_strategy_ids(self) -> List[str]:
        """Get list of currently active strategy IDs."""
        return [
            sid for sid, activation in self.active_strategies.items()
            if activation.activated
        ]
    
    def get_strategy_weight(self, strategy_id: str) -> float:
        """Get current allocation weight for a strategy."""
        activation = self.active_strategies.get(strategy_id)
        return activation.allocation_weight if activation else 0.0
    
    def get_regime_summary(self) -> Dict[str, Any]:
        """Get summary of current regime state."""
        if self.current_regime is None:
            return {"regime": None, "active_count": 0}
        
        return {
            "regime": self.current_regime.regime.value,
            "confidence": self.current_regime.confidence,
            "volatility": self.current_regime.volatility,
            "momentum": self.current_regime.momentum,
            "correlation_level": self.current_regime.correlation_level,
            "active_strategies": len(self.get_active_strategy_ids()),
            "timestamp": self.current_regime.timestamp.isoformat(),
        }
    
    def _notify_regime_change(self, 
                             state: RegimeState, 
                             activations: List[StrategyActivation]):
        """Send regime change notification to Rust via ZMQ."""
        try:
            self._zmq_socket.send_json({
                'type': 'REGIME_CHANGE',
                'regime': state.regime.value,
                'confidence': state.confidence,
                'volatility': state.volatility,
                'momentum': state.momentum,
                'activations': [
                    {
                        'strategy_id': a.strategy_id,
                        'strategy_type': a.strategy_type.value,
                        'activated': a.activated,
                        'weight': a.allocation_weight,
                    }
                    for a in activations
                ],
                'timestamp': state.timestamp.isoformat(),
            }, flags=zmq.NOBLOCK)
        except Exception as e:
            logger.error(f"Failed to send regime change notification: {e}")
    
    def shutdown(self):
        """Cleanup resources."""
        self._zmq_socket.close()
        self._zmq_context.term()
        logger.info("RegimeRouter shut down")


# Global instance
_router: Optional[RegimeRouter] = None


def get_router() -> RegimeRouter:
    """Get or create the global RegimeRouter instance."""
    global _router
    if _router is None:
        _router = RegimeRouter()
    return _router


def create_router(min_confidence_threshold: float = 0.6,
                  max_strategy_count: int = 5) -> RegimeRouter:
    """Create a new RegimeRouter with custom configuration."""
    global _router
    _router = RegimeRouter(
        min_confidence_threshold=min_confidence_threshold,
        max_strategy_count=max_strategy_count,
    )
    return _router
