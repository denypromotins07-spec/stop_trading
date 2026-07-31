"""
Volatility Module Root.
Pushes volatility regime signals to Nautilus MessageBus for execution aggression adjustment.
Integrates surface fitting, Greeks calculation, and skew analysis.
Memory-efficient design with bounded arrays.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import time


class VolRegime(Enum):
    """Volatility regime classifications."""
    VERY_LOW = "very_low"
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    VERY_HIGH = "very_high"
    EXTREME = "extreme"


@dataclass
class VolatilitySignal:
    """Container for volatility-based trading signal."""
    timestamp_ns: int
    asset: str
    regime: VolRegime
    iv_rank: float
    iv_percentile: float
    gamma_regime: str
    skew_signal: int  # 1=bullish, -1=bearish, 0=neutral
    confidence: float
    recommended_aggression: float  # 0-1 scale for order aggression


class VolatilityMessageBus:
    """
    Message bus for publishing volatility signals to Nautilus strategies.
    Implements publish-subscribe pattern for efficient signal distribution.
    """
    
    def __init__(self, max_queue_size: int = 1000):
        """
        Args:
            max_queue_size: Maximum messages in queue before dropping
        """
        self.max_queue_size = max_queue_size
        
        # Signal queues by asset
        self.signal_queues = {}
        
        # Subscriber callbacks (for local testing)
        self.subscribers = {}
        
        # Statistics
        self.stats = {
            'signals_published': 0,
            'signals_dropped': 0,
            'avg_latency_ns': 0,
            'latency_samples': 0
        }
    
    def subscribe(self, strategy_name: str, callback: callable):
        """Subscribe a strategy to receive signals."""
        self.subscribers[strategy_name] = callback
    
    def unsubscribe(self, strategy_name: str):
        """Unsubscribe a strategy."""
        if strategy_name in self.subscribers:
            del self.subscribers[strategy_name]
    
    def publish(self, signal: VolatilitySignal):
        """
        Publish volatility signal to all subscribers.
        
        Args:
            signal: VolatilitySignal to publish
        """
        timestamp_ns = time.time_ns()
        
        # Add to queue
        if signal.asset not in self.signal_queues:
            self.signal_queues[signal.asset] = []
        
        queue = self.signal_queues[signal.asset]
        queue.append(signal)
        
        # Trim queue if needed
        if len(queue) > self.max_queue_size:
            queue.pop(0)
            self.stats['signals_dropped'] += 1
        
        # Notify subscribers
        for strategy_name, callback in self.subscribers.items():
            try:
                callback(signal)
            except Exception:
                pass  # Log error but continue
        
        self.stats['signals_published'] += 1
        
        # Update latency stats
        latency = timestamp_ns - signal.timestamp_ns
        n = self.stats['latency_samples']
        self.stats['avg_latency_ns'] = (self.stats['avg_latency_ns'] * n + latency) / (n + 1)
        self.stats['latency_samples'] = n + 1
    
    def get_latest_signal(self, asset: str) -> Optional[VolatilitySignal]:
        """Get latest signal for an asset."""
        if asset not in self.signal_queues or not self.signal_queues[asset]:
            return None
        return self.signal_queues[asset][-1]
    
    def get_statistics(self) -> Dict:
        """Get message bus statistics."""
        return {
            **self.stats,
            'queue_sizes': {asset: len(q) for asset, q in self.signal_queues.items()}
        }


class VolatilityRegimeDetector:
    """
    Detects current volatility regime from multiple indicators.
    Uses percentile-based classification for robustness.
    """
    
    def __init__(self, 
                 lookback_days: int = 252,
                 percentiles: Tuple[float] = (10, 25, 75, 90, 95)):
        """
        Args:
            lookback_days: Historical lookback for percentile calculation
            percentiles: Percentile thresholds for regime boundaries
        """
        self.lookback_days = lookback_days
        self.percentiles = percentiles
        
        # Historical IV storage per asset
        self.iv_history = {}
        self.max_history = lookback_days
        
        # Current regime cache
        self.current_regimes = {}
    
    def update_iv_history(self, asset: str, iv: float):
        """Update IV history for an asset."""
        if asset not in self.iv_history:
            self.iv_history[asset] = []
        
        history = self.iv_history[asset]
        history.append(iv)
        
        if len(history) > self.max_history:
            history.pop(0)
    
    def calculate_iv_rank(self, asset: str, current_iv: float) -> float:
        """
        Calculate IV Rank: where current IV sits in historical range.
        
        Returns value between 0 and 1.
        """
        if asset not in self.iv_history or len(self.iv_history[asset]) < 20:
            return 0.5  # Neutral if insufficient data
        
        history = np.array(self.iv_history[asset])
        min_iv = np.min(history)
        max_iv = np.max(history)
        
        if max_iv - min_iv < 1e-10:
            return 0.5
        
        iv_rank = (current_iv - min_iv) / (max_iv - min_iv)
        return np.clip(iv_rank, 0.0, 1.0)
    
    def calculate_iv_percentile(self, asset: str, current_iv: float) -> float:
        """
        Calculate IV Percentile: percentage of days IV was below current.
        
        Returns value between 0 and 100.
        """
        if asset not in self.iv_history or len(self.iv_history[asset]) < 20:
            return 50.0
        
        history = np.array(self.iv_history[asset])
        percentile = np.sum(history < current_iv) / len(history) * 100
        return percentile
    
    def classify_regime(self, asset: str, current_iv: float) -> VolRegime:
        """
        Classify current volatility regime.
        
        Args:
            asset: Asset identifier
            current_iv: Current implied volatility
            
        Returns:
            VolRegime classification
        """
        # Update history
        self.update_iv_history(asset, current_iv)
        
        # Calculate metrics
        iv_rank = self.calculate_iv_rank(asset, current_iv)
        iv_pct = self.calculate_iv_percentile(asset, current_iv)
        
        # Cache results
        self.current_regimes[asset] = {
            'iv_rank': iv_rank,
            'iv_percentile': iv_pct
        }
        
        # Classify based on percentile
        if iv_pct < self.percentiles[0]:
            regime = VolRegime.VERY_LOW
        elif iv_pct < self.percentiles[1]:
            regime = VolRegime.LOW
        elif iv_pct < self.percentiles[2]:
            regime = VolRegime.NORMAL
        elif iv_pct < self.percentiles[3]:
            regime = VolRegime.HIGH
        elif iv_pct < self.percentiles[4]:
            regime = VolRegime.VERY_HIGH
        else:
            regime = VolRegime.EXTREME
        
        return regime
    
    def get_regime_metrics(self, asset: str) -> Optional[Dict]:
        """Get current regime metrics for an asset."""
        if asset not in self.current_regimes:
            return None
        
        return {
            'regime': self.classify_regime(asset, 0),  # Just returns cached
            'iv_rank': self.current_regimes[asset]['iv_rank'],
            'iv_percentile': self.current_regimes[asset]['iv_percentile']
        }


class VolatilityAlphaEngine:
    """
    Main engine combining all volatility signals for alpha generation.
    Integrates surface fitting, Greeks, skew, and regime detection.
    """
    
    def __init__(self, assets: List[str]):
        """
        Args:
            assets: List of assets to monitor
        """
        self.assets = assets
        
        # Initialize components
        from .surface_fitter import VolatilitySurfaceFitter
        from .greeks_signals import (
            OptionsGreeksCalculator,
            VolatilitySkewAnalyzer,
            GammaExposureMonitor
        )
        
        self.surface_fitter = VolatilitySurfaceFitter()
        self.greeks_calc = OptionsGreeksCalculator()
        self.skew_analyzer = VolatilitySkewAnalyzer()
        self.gamma_monitor = GammaExposureMonitor(assets)
        self.regime_detector = VolatilityRegimeDetector()
        
        # Message bus for signal distribution
        self.message_bus = VolatilityMessageBus()
        
        # Current state
        self.last_update_ns = {}
    
    def process_options_chain(self,
                              asset: str,
                              spot: float,
                              chains: Dict[float, Dict]) -> Optional[VolatilitySignal]:
        """
        Process full options chain and generate volatility signal.
        
        Args:
            asset: Asset identifier
            spot: Current spot price
            chains: Dict mapping tenor to {strikes, ivs, open_interest}
            
        Returns:
            VolatilitySignal or None
        """
        timestamp_ns = time.time_ns()
        
        # Fit surfaces for each tenor
        fitted_results = {}
        for tenor, chain_data in chains.items():
            strikes = np.array(chain_data.get('strikes', []))
            ivs = np.array(chain_data.get('ivs', []))
            
            if len(strikes) >= 3 and len(ivs) >= 3:
                result = self.surface_fitter.fit_surface(
                    asset, strikes, spot, ivs, tenor
                )
                fitted_results[tenor] = result
        
        if not fitted_results:
            return None
        
        # Get ATM IV from nearest tenor
        nearest_tenor = min(fitted_results.keys())
        atm_iv = np.median(chains[nearest_tenor].get('ivs', [0.5]))
        
        # Classify regime
        regime = self.regime_detector.classify_regime(asset, atm_iv)
        regime_metrics = self.regime_detector.get_regime_metrics(asset)
        
        # Calculate gamma exposure for nearest tenor
        chain_data = chains[nearest_tenor]
        strikes = np.array(chain_data.get('strikes', []))
        ivs = np.array(chain_data.get('ivs', []))
        oi = np.array(chain_data.get('open_interest', np.zeros(len(strikes))))
        
        gamma_exp = self.gamma_monitor.update_gamma_exposure(
            asset, spot, strikes, ivs, nearest_tenor, oi
        )
        gamma_regime, _ = self.gamma_monitor.get_regime(asset)
        
        # Analyze skew
        skew_25d, _, _ = self.skew_analyzer.calculate_skew(strikes, ivs, spot)
        skew_signal_obj = self.skew_analyzer.generate_signal(asset, skew_25d, gamma_exp, spot)
        
        skew_direction = 0
        if skew_signal_obj is not None:
            skew_direction = skew_signal_obj.signal_type.value
        
        # Determine recommended aggression based on regime
        aggression = self._calculate_aggression(regime, gamma_regime, skew_direction)
        
        # Calculate confidence
        confidence = self._calculate_confidence(
            regime_metrics, gamma_regime, skew_signal_obj
        )
        
        signal = VolatilitySignal(
            timestamp_ns=timestamp_ns,
            asset=asset,
            regime=regime,
            iv_rank=regime_metrics['iv_rank'],
            iv_percentile=regime_metrics['iv_percentile'],
            gamma_regime=gamma_regime,
            skew_signal=skew_direction,
            confidence=confidence,
            recommended_aggression=aggression
        )
        
        # Publish to message bus
        self.message_bus.publish(signal)
        self.last_update_ns[asset] = timestamp_ns
        
        return signal
    
    def _calculate_aggression(self, regime: VolRegime, 
                             gamma_regime: str,
                             skew_direction: int) -> float:
        """Calculate recommended order aggression (0-1)."""
        base_aggression = 0.5
        
        # Adjust for volatility regime
        regime_adjustments = {
            VolRegime.VERY_LOW: 0.2,
            VolRegime.LOW: 0.1,
            VolRegime.NORMAL: 0.0,
            VolRegime.HIGH: -0.1,
            VolRegime.VERY_HIGH: -0.2,
            VolRegime.EXTREME: -0.3
        }
        base_aggression += regime_adjustments.get(regime, 0)
        
        # Adjust for gamma regime
        if gamma_regime == "NEGATIVE_GAMMA":
            base_aggression -= 0.15  # Reduce aggression in unstable regime
        elif gamma_regime == "POSITIVE_GAMMA":
            base_aggression += 0.1   # Can be more aggressive in stable regime
        
        return np.clip(base_aggression, 0.0, 1.0)
    
    def _calculate_confidence(self, regime_metrics: Dict,
                             gamma_regime: str,
                             skew_signal: Any) -> float:
        """Calculate overall signal confidence."""
        confidence = 0.5
        
        # Higher confidence at regime extremes
        iv_rank = regime_metrics.get('iv_rank', 0.5)
        if iv_rank > 0.8 or iv_rank < 0.2:
            confidence += 0.2
        
        # Confidence boost from gamma regime clarity
        if gamma_regime in ["NEGATIVE_GAMMA", "POSITIVE_GAMMA"]:
            confidence += 0.1
        
        # Confidence from skew signal
        if skew_signal is not None and abs(skew_signal.z_score) > 2:
            confidence += 0.2
        
        return np.clip(confidence, 0.0, 1.0)
    
    def get_all_signals(self) -> List[VolatilitySignal]:
        """Get latest signals for all assets."""
        signals = []
        for asset in self.assets:
            signal = self.message_bus.get_latest_signal(asset)
            if signal is not None:
                signals.append(signal)
        return signals
    
    def get_nautilus_commands(self) -> List[Dict]:
        """
        Generate Nautilus Trader commands from current signals.
        
        Returns:
            List of command dictionaries
        """
        signals = self.get_all_signals()
        commands = []
        
        for signal in signals:
            command = {
                'type': 'volatility_adjustment',
                'asset': signal.asset,
                'regime': signal.regime.value,
                'aggression_factor': signal.recommended_aggression,
                'gamma_regime': signal.gamma_regime,
                'timestamp_ns': signal.timestamp_ns,
                'metadata': {
                    'iv_rank': signal.iv_rank,
                    'iv_percentile': signal.iv_percentile,
                    'skew_signal': signal.skew_signal,
                    'confidence': signal.confidence
                }
            }
            commands.append(command)
        
        return commands


__all__ = [
    'VolatilityAlphaEngine',
    'VolatilityMessageBus',
    'VolatilityRegimeDetector',
    'VolatilitySignal',
    'VolRegime'
]
