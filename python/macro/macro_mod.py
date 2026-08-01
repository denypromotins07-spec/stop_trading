"""
Macro Module Root - Pushes macro regime probabilities to Nautilus portfolio manager.
Integrates with Rust IPC bridge for real-time regime updates.
"""

from typing import Dict, List, Optional, Any, Callable
import numpy as np
from dataclasses import dataclass, field
import threading
import time
import json

from .correlation_matrix import RollingCorrelationMatrix, CorrelationStats, get_correlation_matrix
from .regime_classifier import MacroRegimeClassifier, RegimeState, RegimeResult, get_regime_classifier


@dataclass
class MacroState:
    """Complete macro-economic state snapshot."""
    
    # Regime classification
    current_regime: RegimeState = RegimeState.TRANSITION
    regime_probabilities: np.ndarray = field(default_factory=lambda: np.zeros(3))
    regime_confidence: float = 0.0
    suggested_beta: float = 0.7
    
    # Correlation structure
    btc_dxy_corr: float = 0.0
    btc_spx_corr: float = 0.0
    btc_yields_corr: float = 0.0
    avg_abs_corr: float = 0.0
    
    # Market signals
    regime_signal: str = "NEUTRAL"
    risk_appetite: float = 0.5  # 0-1 scale
    
    # Metadata
    timestamp: float = 0.0
    update_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for IPC serialization."""
        return {
            "current_regime": self.current_regime.name,
            "regime_probabilities": self.regime_probabilities.tolist(),
            "regime_confidence": self.regime_confidence,
            "suggested_beta": self.suggested_beta,
            "btc_dxy_corr": self.btc_dxy_corr,
            "btc_spx_corr": self.btc_spx_corr,
            "btc_yields_corr": self.btc_yields_corr,
            "avg_abs_corr": self.avg_abs_corr,
            "regime_signal": self.regime_signal,
            "risk_appetite": self.risk_appetite,
            "timestamp": self.timestamp,
            "update_count": self.update_count
        }
    
    def to_json(self) -> str:
        """Serialize to JSON for Rust IPC."""
        return json.dumps(self.to_dict())


class MacroManager:
    """
    Central macro management system.
    Coordinates correlation tracking and regime classification.
    Publishes state to Nautilus portfolio manager via IPC.
    """
    
    def __init__(
        self,
        correlation_halflife: int = 50,
        regime_history_window: int = 50,
        ipc_callback: Optional[Callable[[str], None]] = None
    ):
        # Sub-modules
        self.correlation_matrix = RollingCorrelationMatrix(halflife=correlation_halflife)
        self.regime_classifier = MacroRegimeClassifier(history_window=regime_history_window)
        
        # IPC callback for Rust communication
        self._ipc_callback = ipc_callback
        
        # State tracking
        self._current_state: Optional[MacroState] = None
        self._state_history: List[MacroState] = []
        self._history_max = 100
        
        # Callbacks for regime changes
        self._regime_change_callbacks: List[Callable[[RegimeState, RegimeState], None]] = []
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Performance metrics
        self._total_updates = 0
        self._last_update_time = 0.0
        self._avg_update_latency_ms = 0.0
    
    def register_regime_callback(
        self,
        callback: Callable[[RegimeState, RegimeState], None]
    ) -> None:
        """Register callback for regime changes (old_regime, new_regime)."""
        self._regime_change_callbacks.append(callback)
    
    def update(
        self,
        asset_values: np.ndarray,
        macro_features: np.ndarray,
        timestamp: Optional[float] = None
    ) -> MacroState:
        """
        Update macro state with new data.
        
        Args:
            asset_values: [BTC, DXY, YIELDS, SPX] returns
            macro_features: [volatility, growth, inflation, credit_spread]
            timestamp: Optional timestamp (defaults to now)
        """
        start_time = time.perf_counter()
        
        if timestamp is None:
            timestamp = time.time()
        
        with self._lock:
            # Store previous regime for change detection
            prev_regime = self.regime_classifier.get_current_regime()
            
            # Update correlation matrix
            self.correlation_matrix.update_tick(asset_values)
            
            # Update regime classifier
            self.regime_classifier.partial_fit(macro_features)
            
            # Get regime classification
            regime_result = self.regime_classifier.classify(macro_features)
            
            # Get correlation stats
            corr_stats = self.correlation_matrix.get_stats()
            
            # Build new state
            risk_appetite = self._compute_risk_appetite(regime_result, corr_stats)
            
            new_state = MacroState(
                current_regime=regime_result.current_regime,
                regime_probabilities=regime_result.regime_probabilities,
                regime_confidence=regime_result.confidence,
                suggested_beta=regime_result.suggested_beta,
                btc_dxy_corr=corr_stats.btc_dxy,
                btc_spx_corr=corr_stats.btc_spx,
                btc_yields_corr=corr_stats.btc_yields,
                avg_abs_corr=corr_stats.avg_abs_corr,
                regime_signal=self.correlation_matrix.get_regime_signal(),
                risk_appetite=risk_appetite,
                timestamp=timestamp,
                update_count=self._total_updates + 1
            )
            
            # Detect regime change
            if prev_regime != regime_result.current_regime:
                self._on_regime_change(prev_regime, regime_result.current_regime)
            
            # Update state history
            self._current_state = new_state
            self._state_history.append(new_state)
            if len(self._state_history) > self._history_max:
                self._state_history.pop(0)
            
            # Update metrics
            self._total_updates += 1
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self._avg_update_latency_ms = (
                0.9 * self._avg_update_latency_ms + 0.1 * elapsed_ms
            )
            self._last_update_time = timestamp
            
            # Send to IPC if callback registered
            if self._ipc_callback:
                try:
                    self._ipc_callback(new_state.to_json())
                except Exception as e:
                    pass  # Don't let IPC errors break the pipeline
            
            return new_state
    
    def _compute_risk_appetite(
        self,
        regime_result: RegimeResult,
        corr_stats: CorrelationStats
    ) -> float:
        """Compute overall risk appetite score (0-1)."""
        # Base from regime
        regime_score = regime_result.suggested_beta / 1.2  # Normalize to ~0-1
        
        # Adjust for correlation stress
        if corr_stats.max_abs_corr > 0.8:
            regime_score *= 0.8  # High correlation = reduced diversification
        
        # Adjust for BTC-SPX relationship
        if corr_stats.btc_spx > 0.5:
            regime_score *= 1.05  # Sync with stocks = clearer signal
        elif corr_stats.btc_spx < -0.2:
            regime_score *= 0.9  # Decorrelation = uncertainty
        
        return float(np.clip(regime_score, 0.0, 1.0))
    
    def _on_regime_change(
        self,
        old_regime: RegimeState,
        new_regime: RegimeState
    ) -> None:
        """Handle regime transition."""
        # Notify callbacks
        for callback in self._regime_change_callbacks:
            try:
                callback(old_regime, new_regime)
            except Exception:
                pass  # Don't let callback errors break the pipeline
    
    def get_current_state(self) -> Optional[MacroState]:
        """Get current macro state."""
        with self._lock:
            return self._current_state
    
    def get_state_history(self) -> List[MacroState]:
        """Get recent state history."""
        with self._lock:
            return self._state_history.copy()
    
    def get_regime_exposure(self, regime: RegimeState) -> float:
        """Get fraction of time spent in a regime (recent history)."""
        with self._lock:
            if not self._state_history:
                return 0.0
            
            count = sum(1 for s in self._state_history if s.current_regime == regime)
            return count / len(self._state_history)
    
    def get_adaptive_beta(self) -> float:
        """Get dynamically adjusted beta based on macro conditions."""
        with self._lock:
            if self._current_state is None:
                return 0.7  # Default neutral
            
            base_beta = self._current_state.suggested_beta
            
            # Adjust for confidence
            confidence_factor = self._current_state.regime_confidence
            adjusted_beta = base_beta * confidence_factor + 0.7 * (1 - confidence_factor)
            
            # Adjust for correlation stability
            if self._current_state.avg_abs_corr > 0.7:
                adjusted_beta *= 0.9  # Reduce in high correlation environment
            
            return float(np.clip(adjusted_beta, 0.1, 1.5))
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        with self._lock:
            return {
                "total_updates": self._total_updates,
                "avg_update_latency_ms": round(self._avg_update_latency_ms, 3),
                "last_update_time": self._last_update_time,
                "state_history_size": len(self._state_history),
                "correlation_warmed_up": self.correlation_matrix.is_warmed_up(),
                "regime_trained": self.regime_classifier.is_trained()
            }
    
    def reset(self) -> None:
        """Reset all state."""
        with self._lock:
            self.correlation_matrix.reset()
            self.regime_classifier.reset()
            self._current_state = None
            self._state_history.clear()
            self._total_updates = 0
            self._avg_update_latency_ms = 0.0


# Global singleton instance
_macro_instance: Optional[MacroManager] = None
_instance_lock = threading.Lock()


def get_macro_manager() -> MacroManager:
    """Get or create the global macro manager."""
    global _macro_instance
    if _macro_instance is None:
        with _instance_lock:
            if _macro_instance is None:
                _macro_instance = MacroManager()
    return _macro_instance


def push_macro_update(
    asset_values: np.ndarray,
    macro_features: np.ndarray
) -> Dict[str, Any]:
    """Convenience function for quick macro update."""
    manager = get_macro_manager()
    state = manager.update(asset_values, macro_features)
    return state.to_dict()


if __name__ == "__main__":
    # Test the macro manager
    print("Testing MacroManager:")
    
    manager = MacroManager()
    
    # Simulate updates
    np.random.seed(42)
    
    for i in range(50):
        # Generate correlated asset returns
        base = np.random.randn()
        asset_values = np.array([
            base * 0.02 + np.random.randn() * 0.01,  # BTC
            -base * 0.01 + np.random.randn() * 0.005,  # DXY
            base * 0.005 + np.random.randn() * 0.002,  # Yields
            base * 0.015 + np.random.randn() * 0.008  # SPX
        ])
        
        # Generate macro features based on simulated regime
        if i < 20:
            # Risk-On
            macro_features = np.array([0.1, 0.3, 0.02, 0.01])
        elif i < 40:
            # Risk-Off
            macro_features = np.array([0.4, -0.2, 0.03, 0.04])
        else:
            # Stagflation
            macro_features = np.array([0.3, -0.1, 0.08, 0.03])
        
        macro_features = macro_features + np.random.randn(4) * 0.05
        
        state = manager.update(asset_values, macro_features)
        
        if (i + 1) % 15 == 0:
            print(f"\nUpdate {i + 1}:")
            print(f"  Regime: {state.current_regime.name}")
            print(f"  Suggested Beta: {state.suggested_beta}")
            print(f"  BTC-SPX Corr: {state.btc_spx_corr:.4f}")
            print(f"  Risk Appetite: {state.risk_appetite:.2f}")
    
    print(f"\nPerformance Metrics: {manager.get_performance_metrics()}")
    print(f"Adaptive Beta: {manager.get_adaptive_beta():.4f}")
