"""
Ornstein-Uhlenbeck Process Modeler for Mean Reversion Trading.
Calculates half-life of mean reversion and generates precise Z-score entry/exit thresholds.
Optimized for Nautilus execution engine integration.
Strictly NumPy/SciPy based - no Pandas in hot path.
"""

import numpy as np
from scipy import optimize, stats
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum


class SignalType(Enum):
    """Trading signal types."""
    LONG = 1
    SHORT = -1
    FLAT = 0


@dataclass
class OUParameters:
    """Ornstein-Uhlenbeck process parameters."""
    theta: float      # Mean reversion speed
    mu: float         # Long-term mean
    sigma: float      # Volatility
    half_life: float  # Half-life of mean reversion
    
    @property
    def is_stationary(self) -> bool:
        """Check if process is stationary (theta > 0)."""
        return self.theta > 0


@dataclass
class TradingThresholds:
    """Entry and exit thresholds for mean reversion trading."""
    entry_zscore: float    # Z-score to enter position
    exit_zscore: float     # Z-score to exit position
    stop_loss_zscore: float  # Stop loss threshold
    
    @property
    def entry_upper(self) -> float:
        """Upper entry threshold (short signal)."""
        return self.entry_zscore
    
    @property
    def entry_lower(self) -> float:
        """Lower entry threshold (long signal)."""
        return -self.entry_zscore
    
    @property
    def exit_upper(self) -> float:
        """Upper exit threshold for long positions."""
        return self.exit_zscore
    
    @property
    def exit_lower(self) -> float:
        """Lower exit threshold for short positions."""
        return -self.exit_zscore


class OrnsteinUhlenbeckModeler:
    """
    Fits and monitors an Ornstein-Uhlenbeck process for spread modeling.
    Uses maximum likelihood estimation for parameter calibration.
    """
    
    def __init__(self, 
                 window_size: int = 252,
                 min_samples: int = 50,
                 confidence_level: float = 0.95):
        """
        Args:
            window_size: Rolling window for parameter estimation
            min_samples: Minimum samples required for fitting
            confidence_level: Confidence level for threshold calculation
        """
        self.window_size = window_size
        self.min_samples = min_samples
        self.confidence_level = confidence_level
        
        # Circular buffer for spread values
        self.spread_buffer = np.zeros(window_size)
        self.buffer_idx = 0
        self.samples_count = 0
        
        # Cached parameters
        self.params: Optional[OUParameters] = None
        self.thresholds: Optional[TradingThresholds] = None
        
        # Time step (assumes unit time between observations)
        self.dt = 1.0
        
    def update(self, spread: float) -> Optional[OUParameters]:
        """
        Update with new spread observation and refit parameters.
        
        Args:
            spread: Current spread value
            
        Returns:
            Updated OU parameters or None if insufficient data
        """
        # Add to circular buffer
        idx = self.buffer_idx % self.window_size
        self.spread_buffer[idx] = spread
        self.buffer_idx += 1
        self.samples_count += 1
        
        # Refit if enough samples
        if self.samples_count >= self.min_samples:
            self.params = self._fit_ou_mle()
            self.thresholds = self._calculate_thresholds()
            
        return self.params
    
    def _fit_ou_mle(self) -> Optional[OUParameters]:
        """
        Fit OU process using Maximum Likelihood Estimation.
        
        The OU process SDE: dX_t = theta * (mu - X_t) * dt + sigma * dW_t
        
        Discrete form: X_{t+1} = X_t + theta * (mu - X_t) * dt + sigma * sqrt(dt) * epsilon
        
        MLE solution via linear regression:
        X_{t+1} - X_t = alpha + beta * X_t + error
        where alpha = theta * mu * dt, beta = -theta * dt
        """
        valid_count = min(self.buffer_idx, self.window_size)
        
        if valid_count < self.min_samples:
            return None
        
        start_idx = max(0, self.buffer_idx - self.window_size)
        spreads = self.spread_buffer[start_idx:self.buffer_idx]
        
        if len(spreads) < 3:
            return None
        
        # Prepare regression: delta_spread ~ spread
        X = spreads[:-1]
        Y = spreads[1:] - spreads[:-1]  # First differences
        
        # OLS estimation
        n = len(X)
        sum_x = np.sum(X)
        sum_y = np.sum(Y)
        sum_xx = np.sum(X * X)
        sum_xy = np.sum(X * Y)
        
        denom = n * sum_xx - sum_x * sum_x
        if abs(denom) < 1e-10:
            return None
        
        # Regression coefficients
        beta_hat = (n * sum_xy - sum_x * sum_y) / denom
        alpha_hat = (sum_y - beta_hat * sum_x) / n
        
        # Convert to OU parameters
        theta = -beta_hat / self.dt
        
        if theta <= 0:
            # Not mean-reverting
            theta = max(theta, 1e-6)
            
        mu = alpha_hat / (theta * self.dt) if theta > 1e-10 else np.mean(spreads)
        
        # Estimate sigma from residuals
        residuals = Y - (alpha_hat + beta_hat * X)
        sigma_sq = np.var(residuals) / self.dt
        sigma = np.sqrt(sigma_sq) if sigma_sq > 0 else 1e-6
        
        # Calculate half-life
        half_life = np.log(2) / theta if theta > 0 else float('inf')
        
        return OUParameters(
            theta=theta,
            mu=mu,
            sigma=sigma,
            half_life=half_life
        )
    
    def _calculate_thresholds(self) -> Optional[TradingThresholds]:
        """Calculate trading thresholds based on fitted parameters."""
        if self.params is None:
            return None
        
        # Get critical z-value for confidence level
        z_critical = stats.norm.ppf(1 - (1 - self.confidence_level) / 2)
        
        # Entry threshold: beyond normal variation
        entry_zscore = z_critical
        
        # Exit threshold: closer to mean
        exit_zscore = z_critical * 0.5
        
        # Stop loss: extreme deviation
        stop_loss_zscore = z_critical * 2.0
        
        return TradingThresholds(
            entry_zscore=entry_zscore,
            exit_zscore=exit_zscore,
            stop_loss_zscore=stop_loss_zscore
        )
    
    def get_signal(self, current_spread: float, 
                   position: float = 0.0) -> Tuple[SignalType, float]:
        """
        Generate trading signal based on current spread and position.
        
        Args:
            current_spread: Current spread value
            position: Current position (positive = long, negative = short)
            
        Returns:
            Tuple of (signal_type, z_score)
        """
        if self.params is None or self.thresholds is None:
            return SignalType.FLAT, 0.0
        
        # Calculate Z-score
        z_score = (current_spread - self.params.mu) / (self.params.sigma + 1e-10)
        
        # Determine signal based on position and thresholds
        if position == 0:
            # No position - look for entry
            if z_score < -self.thresholds.entry_zscore:
                return SignalType.LONG, z_score
            elif z_score > self.thresholds.entry_zscore:
                return SignalType.SHORT, z_score
        elif position > 0:
            # Long position - look for exit or stop
            if z_score > -self.thresholds.exit_lower:
                return SignalType.FLAT, z_score  # Take profit
            elif z_score < -self.thresholds.stop_loss_zscore:
                return SignalType.FLAT, z_score  # Stop loss
        else:
            # Short position - look for exit or stop
            if z_score < self.thresholds.exit_upper:
                return SignalType.FLAT, z_score  # Take profit
            elif z_score > self.thresholds.stop_loss_zscore:
                return SignalType.FLAT, z_score  # Stop loss
        
        return SignalType.FLAT, z_score
    
    def get_expected_return_time(self, z_score: float) -> float:
        """
        Calculate expected time for spread to revert to mean from current Z-score.
        
        For OU process, expected first passage time to mean from x:
        E[T] = (1/theta) * integral from 0 to x of exp(y^2/2) dy (approx)
        
        Simplified approximation using half-life scaling.
        """
        if self.params is None or self.params.theta <= 0:
            return float('inf')
        
        # Approximate: time to halve the deviation
        half_lives_needed = np.log2(abs(z_score) + 1)
        return half_lives_needed * self.params.half_life
    
    def get_state(self) -> Dict:
        """Return current model state as dictionary."""
        return {
            'params': {
                'theta': self.params.theta if self.params else None,
                'mu': self.params.mu if self.params else None,
                'sigma': self.params.sigma if self.params else None,
                'half_life': self.params.half_life if self.params else None,
                'is_stationary': self.params.is_stationary if self.params else False
            },
            'thresholds': {
                'entry_zscore': self.thresholds.entry_zscore if self.thresholds else None,
                'exit_zscore': self.thresholds.exit_zscore if self.thresholds else None,
                'stop_loss_zscore': self.thresholds.stop_loss_zscore if self.thresholds else None
            } if self.thresholds else None,
            'samples_count': self.samples_count,
            'current_spread': self.spread_buffer[(self.buffer_idx - 1) % self.window_size] if self.buffer_idx > 0 else None
        }
    
    def reset(self):
        """Reset all state."""
        self.spread_buffer.fill(0)
        self.buffer_idx = 0
        self.samples_count = 0
        self.params = None
        self.thresholds = None


class SpreadMonitor:
    """
    Monitors multiple spreads simultaneously for mean reversion opportunities.
    Optimized for high-frequency updates.
    """
    
    def __init__(self, pairs: list, window_size: int = 252):
        """
        Args:
            pairs: List of (asset_x, asset_y, hedge_ratio) tuples
            window_size: Rolling window for OU estimation
        """
        self.pairs = pairs
        self.models = {}
        self.hedge_ratios = {p[0]: p[2] if len(p) > 2 else 1.0 for p in pairs}
        
        for pair in pairs:
            key = (pair[0], pair[1])
            self.models[key] = OrnsteinUhlenbeckModeler(window_size=window_size)
    
    def update_prices(self, prices: dict) -> Dict[tuple, Dict]:
        """
        Update all spreads given current prices.
        
        Args:
            prices: Dict mapping asset symbols to prices
            
        Returns:
            Dict mapping pairs to their signals and metrics
        """
        results = {}
        
        for pair in self.pairs:
            asset_x, asset_y = pair[0], pair[1]
            hedge_ratio = self.hedge_ratios.get((asset_x, asset_y), 1.0)
            
            if asset_x in prices and asset_y in prices:
                spread = prices[asset_y] - hedge_ratio * prices[asset_x]
                key = (asset_x, asset_y)
                
                model = self.models[key]
                params = model.update(spread)
                signal, z_score = model.get_signal(spread)
                
                results[key] = {
                    'spread': spread,
                    'z_score': z_score,
                    'signal': signal.name,
                    'half_life': params.half_life if params else None,
                    'theta': params.theta if params else None,
                    'mu': params.mu if params else None
                }
        
        return results
    
    def get_top_opportunities(self, n: int = 5) -> list:
        """Get top N mean reversion opportunities by absolute Z-score."""
        opportunities = []
        
        for key, model in self.models.items():
            if model.params is not None:
                current_spread = model.spread_buffer[(model.buffer_idx - 1) % model.window_size]
                _, z_score = model.get_signal(current_spread)
                
                if abs(z_score) > 0.5:  # Minimum threshold
                    opportunities.append({
                        'pair': key,
                        'z_score': z_score,
                        'half_life': model.params.half_life,
                        'expected_return_time': model.get_expected_return_time(z_score)
                    })
        
        # Sort by absolute Z-score descending
        opportunities.sort(key=lambda x: abs(x['z_score']), reverse=True)
        return opportunities[:n]


# Nautilus Trader integration helpers
def generate_nautilus_signals(spread_monitor: SpreadMonitor, 
                              prices: dict) -> list:
    """
    Generate Nautilus Trader compatible signals from spread monitor.
    
    Args:
        spread_monitor: Configured SpreadMonitor instance
        prices: Current price dictionary
        
    Returns:
        List of signal dictionaries compatible with Nautilus
    """
    results = spread_monitor.update_prices(prices)
    signals = []
    
    for (asset_x, asset_y), data in results.items():
        if data['signal'] != 'FLAT':
            signal = {
                'instrument_id': f"{asset_y}/{asset_x}",
                'signal_type': data['signal'],
                'z_score': float(data['z_score']),
                'spread': float(data['spread']),
                'half_life': float(data['half_life']) if data['half_life'] else None,
                'confidence': min(abs(data['z_score']) / 3.0, 1.0),  # Normalize to [0, 1]
                'expected_hold_time': data['expected_return_time'] if 'expected_return_time' in data else None
            }
            signals.append(signal)
    
    return signals


__all__ = [
    'OrnsteinUhlenbeckModeler',
    'SpreadMonitor',
    'OUParameters',
    'TradingThresholds',
    'SignalType',
    'generate_nautilus_signals'
]
