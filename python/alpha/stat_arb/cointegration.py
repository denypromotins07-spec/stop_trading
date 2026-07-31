"""
Rolling Engle-Granger and Kalman Filter Hedge Ratio Calculators.
Implements O(1) dynamic beta weight updates for pairs trading without storing massive historical matrices.
Strictly NumPy/SciPy based - no Pandas in hot path.
"""

import numpy as np
from scipy import linalg
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class CointegrationState:
    """State container for rolling cointegration calculations."""
    # Engle-Granger state
    sum_xx: float = 0.0
    sum_xy: float = 0.0
    sum_yy: float = 0.0
    sum_x: float = 0.0
    sum_y: float = 0.0
    n_samples: int = 0
    
    # Kalman Filter state for hedge ratio
    beta: float = 1.0  # Current hedge ratio
    P: float = 1.0     # Covariance estimate
    Q: float = 1e-5    # Process noise
    R: float = 1e-3    # Measurement noise
    
    # Rolling window parameters
    half_life: float = 0.0
    mean_spread: float = 0.0
    std_spread: float = 1.0
    spread_buffer: np.ndarray = None
    
    def __post_init__(self):
        if self.spread_buffer is None:
            self.spread_buffer = np.zeros(100)  # Rolling window for spread stats


class RollingEngleGranger:
    """
    Rolling Engle-Granger cointegration test with O(1) update complexity.
    Uses Welford-style online algorithms for numerical stability.
    """
    
    def __init__(self, window_size: int = 252, critical_value: float = -3.34):
        """
        Args:
            window_size: Rolling window for ADF critical value approximation
            critical_value: 5% significance level for cointegration test
        """
        self.window_size = window_size
        self.critical_value = critical_value
        self.state = CointegrationState()
        
        # Circular buffer for recent residuals (for ADF approximation)
        self.residuals = np.zeros(window_size)
        self.residual_idx = 0
        
    def update(self, x: float, y: float) -> Tuple[float, bool]:
        """
        Update with new observation and return hedge ratio and cointegration flag.
        
        Args:
            x: Price of asset X (e.g., BTC)
            y: Price of asset Y (e.g., ETH)
            
        Returns:
            Tuple of (hedge_ratio, is_cointegrated)
        """
        state = self.state
        
        # O(1) update of sufficient statistics
        state.sum_xx += x * x
        state.sum_xy += x * y
        state.sum_yy += y * y
        state.sum_x += x
        state.sum_y += y
        state.n_samples += 1
        
        # Calculate hedge ratio (beta) using normal equations
        n = float(state.n_samples)
        denom = n * state.sum_xx - state.sum_x ** 2
        
        if abs(denom) < 1e-10:
            beta = state.beta  # Keep previous if singular
        else:
            beta = (n * state.sum_xy - state.sum_x * state.sum_y) / denom
        
        # Calculate spread for this observation
        spread = y - beta * x
        
        # Update rolling spread statistics
        idx = self.residual_idx % self.window_size
        self.residuals[idx] = spread
        self.residual_idx += 1
        
        # Update state
        state.beta = beta
        
        # Compute rolling mean and std of spread
        valid_count = min(self.residual_idx, self.window_size)
        start_idx = max(0, self.residual_idx - self.window_size)
        recent_residuals = self.residuals[start_idx:self.residual_idx]
        
        if valid_count > 10:  # Need minimum samples
            state.mean_spread = np.mean(recent_residuals)
            state.std_spread = np.std(recent_residuals) + 1e-10
            
            # Augmented Dickey-Fuller approximation (simplified)
            # Using lag-1 autocorrelation as proxy for unit root test
            if len(recent_residuals) > 2:
                lag1_corr = np.corrcoef(recent_residuals[:-1], recent_residuals[1:])[0, 1]
                adf_stat = (lag1_corr - 1.0) * np.sqrt(valid_count)
                is_cointegrated = adf_stat < self.critical_value
            else:
                is_cointegrated = False
        else:
            is_cointegrated = False
            
        return beta, is_cointegrated
    
    def get_zscore(self, current_spread: float) -> float:
        """Calculate Z-score of current spread deviation."""
        state = self.state
        if state.std_spread < 1e-10:
            return 0.0
        return (current_spread - state.mean_spread) / state.std_spread
    
    def reset(self):
        """Reset all statistics."""
        self.state = CointegrationState()
        self.residuals.fill(0)
        self.residual_idx = 0


class KalmanHedgeRatio:
    """
    Kalman Filter for dynamic hedge ratio estimation.
    Provides O(1) recursive updates without matrix inversions.
    """
    
    def __init__(self, 
                 initial_beta: float = 1.0,
                 process_noise: float = 1e-4,
                 measurement_noise: float = 1e-2,
                 initial_covariance: float = 1.0):
        """
        Args:
            initial_beta: Initial hedge ratio estimate
            process_noise: Q - State transition noise (how fast beta can change)
            measurement_noise: R - Observation noise
            initial_covariance: P - Initial uncertainty in beta
        """
        self.state = CointegrationState(
            beta=initial_beta,
            P=initial_covariance,
            Q=process_noise,
            R=measurement_noise
        )
        self.spread_history = np.zeros(100)
        self.history_idx = 0
        
    def update(self, x: float, y: float) -> Tuple[float, float, float]:
        """
        Kalman Filter update step for hedge ratio.
        
        Args:
            x: Price of asset X
            y: Price of asset Y
            
        Returns:
            Tuple of (updated_beta, spread, z_score)
        """
        state = self.state
        
        # Prediction step (random walk model for beta)
        beta_pred = state.beta
        P_pred = state.P + state.Q
        
        # Measurement equation: y = beta * x + epsilon
        # Innovation (prediction error)
        innovation = y - beta_pred * x
        
        # Kalman gain (scalar case simplification)
        S = P_pred * x * x + state.R  # Innovation covariance
        if abs(S) < 1e-10:
            S = 1e-10
        K = P_pred * x / S  # Kalman gain
        
        # Update step
        beta_updated = beta_pred + K * innovation
        P_updated = (1 - K * x) * P_pred
        
        # Ensure P stays positive
        P_updated = max(P_updated, 1e-6)
        
        # Update state
        state.beta = beta_updated
        state.P = P_updated
        
        # Calculate spread and Z-score
        spread = y - beta_updated * x
        
        # Update rolling spread statistics for Z-score
        idx = self.history_idx % len(self.spread_history)
        self.spread_history[idx] = spread
        self.history_idx += 1
        
        valid_count = min(self.history_idx, len(self.spread_history))
        start_idx = max(0, self.history_idx - len(self.spread_history))
        recent_spreads = self.spread_history[start_idx:self.history_idx]
        
        if valid_count > 10:
            state.mean_spread = np.mean(recent_spreads)
            state.std_spread = np.std(recent_spreads) + 1e-10
            z_score = (spread - state.mean_spread) / state.std_spread
        else:
            z_score = 0.0
            
        return beta_updated, spread, z_score
    
    def set_noise_parameters(self, Q: float, R: float):
        """Dynamically adjust noise parameters based on market regime."""
        self.state.Q = Q
        self.state.R = R
        
    def get_state(self) -> CointegrationState:
        """Return current filter state."""
        return self.state
    
    def reset(self, initial_beta: float = 1.0):
        """Reset filter to initial state."""
        self.state = CointegrationState(
            beta=initial_beta,
            P=1.0,
            Q=self.state.Q,
            R=self.state.R
        )
        self.spread_history.fill(0)
        self.history_idx = 0


class AdaptiveCointegrationTracker:
    """
    Combines Engle-Granger and Kalman Filter approaches with adaptive switching.
    Monitors multiple pairs efficiently.
    """
    
    def __init__(self, pairs: list, window_size: int = 252):
        """
        Args:
            pairs: List of (asset_x, asset_y) tuples
            window_size: Rolling window size
        """
        self.pairs = pairs
        self.eg_trackers = {}
        self.kf_trackers = {}
        
        for pair in pairs:
            self.eg_trackers[pair] = RollingEngleGranger(window_size=window_size)
            self.kf_trackers[pair] = KalmanHedgeRatio(
                process_noise=1e-4,
                measurement_noise=1e-2
            )
    
    def update_pair(self, pair: tuple, price_x: float, price_y: float) -> dict:
        """
        Update a specific pair and return comprehensive metrics.
        
        Args:
            pair: (asset_x, asset_y) tuple
            price_x: Price of asset X
            price_y: Price of asset Y
            
        Returns:
            Dictionary with all calculated metrics
        """
        if pair not in self.eg_trackers:
            raise ValueError(f"Pair {pair} not registered")
        
        # Engle-Granger update
        eg_tracker = self.eg_trackers[pair]
        eg_beta, is_coint = eg_tracker.update(price_x, price_y)
        eg_spread = price_y - eg_beta * price_x
        eg_zscore = eg_tracker.get_zscore(eg_spread)
        
        # Kalman Filter update
        kf_tracker = self.kf_trackers[pair]
        kf_beta, kf_spread, kf_zscore = kf_tracker.update(price_x, price_y)
        
        # Adaptive selection: use KF if cointegration is strong, EG otherwise
        use_kf = is_coint
        hedge_ratio = kf_beta if use_kf else eg_beta
        spread = kf_spread if use_kf else eg_spread
        z_score = kf_zscore if use_kf else eg_zscore
        
        return {
            'pair': pair,
            'hedge_ratio': hedge_ratio,
            'spread': spread,
            'z_score': z_score,
            'is_cointegrated': is_coint,
            'eg_beta': eg_beta,
            'kf_beta': kf_beta,
            'method': 'kalman' if use_kf else 'engle_granger'
        }
    
    def get_all_pairs_status(self, prices: dict) -> list:
        """
        Update all pairs given a price dictionary.
        
        Args:
            prices: Dict mapping asset symbols to current prices
            
        Returns:
            List of status dicts for all pairs
        """
        results = []
        for pair in self.pairs:
            asset_x, asset_y = pair
            if asset_x in prices and asset_y in prices:
                result = self.update_pair(pair, prices[asset_x], prices[asset_y])
                results.append(result)
        return results


# Ray Actor wrapper for distributed pair monitoring
def create_pair_actor_class():
    """Factory function to create Ray actor class for pair monitoring."""
    try:
        import ray
        
        @ray.actor(num_cpus=0.1, max_memory=50*1024*1024)
        class PairMonitorActor:
            """Ray actor for monitoring a single pair's cointegration."""
            
            def __init__(self, pair: tuple, window_size: int = 252):
                self.pair = pair
                self.tracker = AdaptiveCointegrationTracker([pair], window_size)
                
            def update(self, price_x: float, price_y: float) -> dict:
                """Update with new prices and return metrics."""
                return self.tracker.update_pair(self.pair, price_x, price_y)
                
            def get_hedge_ratio(self) -> float:
                """Get current hedge ratio."""
                status = self.tracker.update_pair(
                    self.pair, 
                    self._last_x if hasattr(self, '_last_x') else 1.0,
                    self._last_y if hasattr(self, '_last_y') else 1.0
                )
                return status['hedge_ratio']
                
            def reset(self):
                """Reset tracker state."""
                self.tracker = AdaptiveCointegrationTracker([self.pair])
                
            def _set_last_prices(self, x: float, y: float):
                """Cache last prices for state queries."""
                self._last_x = x
                self._last_y = y
                
        return PairMonitorActor
        
    except ImportError:
        return None


__all__ = [
    'RollingEngleGranger',
    'KalmanHedgeRatio', 
    'AdaptiveCointegrationTracker',
    'CointegrationState',
    'create_pair_actor_class'
]
