"""
Kalman Filter implementation for pairs trading research and shadow validation.
Continuously updates hidden hedge ratios and mean-reversion speeds.
Validates Rust core's real-time stat-arb signals offline.
Memory-efficient design respecting 3GB RAM constraint.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class PairsState:
    """State of a pairs trading relationship."""
    pair_id: str
    hedge_ratio: float  # Beta: units of asset B per unit of asset A
    mean_reversion_speed: float  # Alpha: speed of reversion
    spread_mean: float
    spread_std: float
    kalman_gain: float
    state_covariance: float
    last_update_ns: int
    observation_count: int = 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'pair_id': self.pair_id,
            'hedge_ratio': self.hedge_ratio,
            'mean_reversion_speed': self.mean_reversion_speed,
            'spread_mean': self.spread_mean,
            'spread_std': self.spread_std,
            'kalman_gain': self.kalman_gain,
            'state_covariance': self.state_covariance,
            'last_update_ns': self.last_update_ns,
            'observation_count': self.observation_count,
        }


@dataclass
class KalmanConfig:
    """Configuration for Kalman Filter parameters."""
    # Process noise covariance (uncertainty in state evolution)
    process_noise: float = 1e-5
    
    # Observation noise covariance (measurement uncertainty)
    observation_noise: float = 1e-2
    
    # Initial state covariance
    initial_covariance: float = 1.0
    
    # Minimum observations before signal generation
    min_observations: int = 50
    
    # Maximum spread history for validation
    max_spread_history: int = 500


class KalmanPairsFilter:
    """
    Kalman Filter for estimating pairs trading parameters.
    
    State vector: [hedge_ratio, spread_mean]
    Observation: spread = price_A - hedge_ratio * price_B
    
    Implements recursive Bayesian estimation for online learning.
    """
    
    def __init__(self, config: Optional[KalmanConfig] = None):
        """
        Initialize Kalman Filter.
        
        Args:
            config: Filter configuration parameters
        """
        self.config = config or KalmanConfig()
        
        # State vector: [hedge_ratio, spread_mean]
        self._state = np.zeros(2)
        
        # State covariance matrix (2x2)
        self._covariance = np.eye(2) * self.config.initial_covariance
        
        # Kalman gain
        self._kalman_gain = np.zeros(2)
        
        # Observation history for validation
        self._spread_history: deque = deque(maxlen=self.config.max_spread_history)
        
        # Statistics
        self._observation_count = 0
        self._sum_squared_errors = 0.0
        
    def update(self, price_a: float, price_b: float, 
               timestamp_ns: int) -> Tuple[float, PairsState]:
        """
        Update filter with new price observation.
        
        Args:
            price_a: Price of asset A
            price_b: Price of asset B
            timestamp_ns: Nanosecond timestamp
            
        Returns:
            Tuple of (current_spread, updated_pairs_state)
        """
        self._observation_count += 1
        
        # Calculate observed spread
        spread = price_a - self._state[0] * price_b
        
        # Store spread for validation
        self._spread_history.append(spread)
        
        # === Kalman Filter Update Step ===
        
        # Observation matrix H: we observe spread = price_A - hedge_ratio * price_B
        # H = [-price_b, 1] (derivative of spread w.r.t. state)
        H = np.array([-price_b, 1.0])
        
        # Predicted observation
        predicted_spread = self._state[0] * (-price_b) + self._state[1]
        
        # Innovation (residual)
        innovation = spread - predicted_spread
        
        # Innovation covariance: S = H * P * H' + R
        S = H @ self._covariance @ H.T + self.config.observation_noise
        
        # Kalman gain: K = P * H' * S^-1
        self._kalman_gain = (self._covariance @ H) / S
        
        # Update state estimate: x = x + K * innovation
        self._state = self._state + self._kalman_gain * innovation
        
        # Update covariance: P = (I - K * H) * P
        I = np.eye(2)
        self._covariance = (I - np.outer(self._kalman_gain, H)) @ self._covariance
        
        # Ensure symmetry
        self._covariance = (self._covariance + self._covariance.T) / 2
        
        # Track squared error for diagnostics
        self._sum_squared_errors += innovation ** 2
        
        # Calculate mean reversion speed from recent spreads
        mr_speed = self._estimate_mean_reversion()
        
        # Calculate spread statistics
        spread_mean = np.mean(self._spread_history) if len(self._spread_history) > 0 else 0.0
        spread_std = np.std(self._spread_history) if len(self._spread_history) > 1 else 1.0
        
        # Create state object
        state = PairsState(
            pair_id="kalman_pair",
            hedge_ratio=float(self._state[0]),
            mean_reversion_speed=mr_speed,
            spread_mean=spread_mean,
            spread_std=max(spread_std, 1e-6),
            kalman_gain=float(np.mean(self._kalman_gain)),
            state_covariance=float(np.trace(self._covariance) / 2),
            last_update_ns=timestamp_ns,
            observation_count=self._observation_count,
        )
        
        return spread, state
    
    def _estimate_mean_reversion(self) -> float:
        """
        Estimate mean reversion speed using recent spread changes.
        
        Uses simple autocorrelation method:
        alpha ≈ -log(|autocorr(1)|) / dt
        
        Returns:
            Mean reversion speed (higher = faster reversion)
        """
        if len(self._spread_history) < 20:
            return 0.0
        
        spreads = np.array(self._spread_history)
        
        # Calculate lag-1 autocorrelation
        mean = np.mean(spreads)
        centered = spreads - mean
        
        numerator = np.sum(centered[:-1] * centered[1:])
        denominator = np.sum(centered[:-1] ** 2)
        
        if denominator == 0:
            return 0.0
            
        autocorr = numerator / denominator
        
        # Clamp to valid range
        autocorr = np.clip(autocorr, -0.99, 0.99)
        
        # Convert to mean reversion speed
        # Higher autocorrelation = slower mean reversion
        mr_speed = -np.log(abs(autocorr))
        
        return float(mr_speed)
    
    def get_z_score(self, current_spread: float) -> float:
        """
        Calculate Z-score of current spread relative to historical distribution.
        
        Args:
            current_spread: Current spread value
            
        Returns:
            Z-score
        """
        if len(self._spread_history) < 10:
            return 0.0
        
        mean = np.mean(self._spread_history)
        std = np.std(self._spread_history)
        
        if std < 1e-6:
            return 0.0
            
        return (current_spread - mean) / std
    
    def is_ready(self) -> bool:
        """Check if filter has enough observations for reliable signals."""
        return self._observation_count >= self.config.min_observations
    
    def get_state(self) -> np.ndarray:
        """Get current state estimate."""
        return self._state.copy()
    
    def get_covariance(self) -> np.ndarray:
        """Get current state covariance."""
        return self._covariance.copy()
    
    def reset(self):
        """Reset filter to initial state."""
        self._state = np.zeros(2)
        self._covariance = np.eye(2) * self.config.initial_covariance
        self._spread_history.clear()
        self._observation_count = 0
        self._sum_squared_errors = 0.0


class KalmanPairsTrader:
    """
    High-level pairs trader using Kalman Filter for parameter estimation.
    Manages multiple pairs and provides trading signals.
    """
    
    def __init__(self, config: Optional[KalmanConfig] = None,
                 z_score_threshold: float = 2.0):
        """
        Initialize pairs trader.
        
        Args:
            config: Kalman Filter configuration
            z_score_threshold: Z-score threshold for trade signals
        """
        self._filters: Dict[str, KalmanPairsFilter] = {}
        self._config = config or KalmanConfig()
        self._z_score_threshold = z_score_threshold
        self._max_pairs = 200  # Memory limit
        
        # Signal callbacks
        self._signal_callbacks = []
        
        # Statistics
        self._stats = {
            'pairs_tracked': 0,
            'signals_generated': 0,
            'total_updates': 0,
        }
        
    def register_pair(self, pair_id: str):
        """
        Register a new pair for tracking.
        
        Args:
            pair_id: Unique pair identifier
        """
        if len(self._filters) >= self._max_pairs:
            logger.warning(f"Max pairs ({self._max_pairs}) reached")
            # Remove oldest
            oldest = next(iter(self._filters))
            del self._filters[oldest]
        
        self._filters[pair_id] = KalmanPairsFilter(self._config)
        self._stats['pairs_tracked'] = len(self._filters)
        logger.debug(f"Registered pair {pair_id}")
    
    def update_prices(self, pair_id: str, price_a: float, price_b: float,
                     timestamp_ns: int) -> Optional[Dict]:
        """
        Update prices for a pair and check for trading signals.
        
        Args:
            pair_id: Pair identifier
            price_a: Price of asset A
            price_b: Price of asset B
            timestamp_ns: Timestamp in nanoseconds
            
        Returns:
            Trading signal dict if conditions met, None otherwise
        """
        self._stats['total_updates'] += 1
        
        # Ensure pair is registered
        if pair_id not in self._filters:
            self.register_pair(pair_id)
        
        filt = self._filters[pair_id]
        spread, state = filt.update(price_a, price_b, timestamp_ns)
        
        # Check if ready for signals
        if not filt.is_ready():
            return None
        
        # Calculate Z-score
        z_score = filt.get_z_score(spread)
        
        # Check for trading opportunity
        if abs(z_score) > self._z_score_threshold:
            signal = {
                'pair_id': pair_id,
                'type': 'pairs_trade',
                'direction': -1 if z_score > 0 else 1,  # Mean reversion
                'spread': spread,
                'z_score': z_score,
                'hedge_ratio': state.hedge_ratio,
                'mean_reversion_speed': state.mean_reversion_speed,
                'confidence': min(1.0, abs(z_score) / (self._z_score_threshold * 2)),
                'timestamp_ns': timestamp_ns,
                'state': state.to_dict(),
            }
            
            self._stats['signals_generated'] += 1
            
            # Notify callbacks
            for callback in self._signal_callbacks:
                try:
                    callback(signal)
                except Exception as e:
                    logger.error(f"Signal callback error: {e}")
            
            return signal
        
        return None
    
    def batch_update(self, updates: List[Tuple[str, float, float, int]]) -> List[Dict]:
        """
        Batch update multiple pairs efficiently.
        
        Args:
            updates: List of (pair_id, price_a, price_b, timestamp_ns)
            
        Returns:
            List of trading signals generated
        """
        signals = []
        for pair_id, price_a, price_b, ts in updates:
            signal = self.update_prices(pair_id, price_a, price_b, ts)
            if signal:
                signals.append(signal)
        return signals
    
    def register_signal_callback(self, callback):
        """Register callback for trading signals."""
        self._signal_callbacks.append(callback)
    
    def get_all_states(self) -> Dict[str, PairsState]:
        """Get current state for all tracked pairs."""
        states = {}
        for pair_id, filt in self._filters.items():
            if filt._observation_count > 0:
                spread = price_a - filt._state[0] * price_b  # Need prices
                _, state = filt.update(0, 1, 0)  # Dummy update to get state
                states[pair_id] = state
        return states
    
    def get_filter(self, pair_id: str) -> Optional[KalmanPairsFilter]:
        """Get Kalman filter for specific pair."""
        return self._filters.get(pair_id)
    
    def get_stats(self) -> Dict:
        """Get trader statistics."""
        return self._stats.copy()
    
    def validate_rust_signals(self, rust_signals: List[Dict]) -> Dict[str, float]:
        """
        Validate signals from Rust core against Kalman filter estimates.
        
        Args:
            rust_signals: List of signals from Rust core
            
        Returns:
            Dict mapping pair_id to validation score (0-1)
        """
        validation = {}
        
        for signal in rust_signals:
            pair_id = signal.get('pair_id')
            rust_z_score = signal.get('z_score', 0)
            
            if pair_id not in self._filters:
                validation[pair_id] = 0.0
                continue
            
            filt = self._filters[pair_id]
            if not filt.is_ready():
                validation[pair_id] = 0.5  # Neutral if not enough data
                continue
            
            # Get latest spread from history
            if len(filt._spread_history) == 0:
                validation[pair_id] = 0.0
                continue
                
            latest_spread = filt._spread_history[-1]
            kalman_z_score = filt.get_z_score(latest_spread)
            
            # Compare Z-scores
            # High validation if both agree on sign and magnitude
            sign_agreement = 1.0 if np.sign(rust_z_score) == np.sign(kalman_z_score) else 0.0
            magnitude_diff = abs(abs(rust_z_score) - abs(kalman_z_score))
            magnitude_score = max(0.0, 1.0 - magnitude_diff / 2.0)
            
            validation[pair_id] = (sign_agreement + magnitude_score) / 2.0
        
        return validation
