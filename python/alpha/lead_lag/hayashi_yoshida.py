"""
Hayashi-Yoshida Estimator for Cross-Correlation on Non-Synchronous Tick Data.
Identifies lead-lag relationships between assets without time-aggregation bias.
Optimized for high-frequency crypto tick data with irregular sampling.
Strictly NumPy based - no Pandas in hot path.
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
from collections import deque


@dataclass
class TickData:
    """Container for tick-level price data."""
    timestamp_ns: int
    price: float
    volume: float
    side: int  # 1=buy, -1=sell, 0=unknown


@dataclass 
class HYResult:
    """Hayashi-Yoshida correlation result."""
    correlation: float
    n_overlaps: int
    leader: Optional[str]
    lag_ms: float
    confidence: float


class HayashiYoshidaEstimator:
    """
    Implements the Hayashi-Yoshida estimator for computing correlation
    between two asynchronous time series without synchronization bias.
    
    The estimator uses all available data points and accounts for
    non-synchronous trading by considering overlapping return intervals.
    
    Reference: Hayashi & Yoshida (2005) "On covariance estimation of 
    non-synchronously observed diffusion processes"
    """
    
    def __init__(self, 
                 max_lag_ms: float = 1000.0,
                 min_ticks: int = 50,
                 decay_factor: float = 0.99):
        """
        Args:
            max_lag_ms: Maximum lag to consider in milliseconds
            min_ticks: Minimum ticks required for estimation
            decay_factor: Exponential decay factor for recent weighting
        """
        self.max_lag_ms = max_lag_ms
        self.min_ticks = min_ticks
        self.decay_factor = decay_factor
        
        # Circular buffers for tick data
        self.buffer_size = 10000
        self.ticks_x = deque(maxlen=self.buffer_size)
        self.ticks_y = deque(maxlen=self.buffer_size)
        
        # Precomputed returns for efficiency
        self.returns_x = deque(maxlen=self.buffer_size)
        self.returns_y = deque(maxlen=self.buffer_size)
        
        # Last prices for return calculation
        self.last_price_x: Optional[float] = None
        self.last_price_y: Optional[float] = None
        self.last_ts_x: int = 0
        self.last_ts_y: int = 0
        
        # Cached result
        self._cached_result: Optional[HYResult] = None
        self._cache_valid = False
        
    def add_tick_x(self, timestamp_ns: int, price: float, volume: float = 0.0):
        """Add tick for asset X."""
        tick = TickData(timestamp_ns=timestamp_ns, price=price, volume=volume, side=0)
        self.ticks_x.append(tick)
        
        # Calculate return
        if self.last_price_x is not None:
            ret = np.log(price / self.last_price_x)
            self.returns_x.append((timestamp_ns, ret))
        
        self.last_price_x = price
        self.last_ts_x = timestamp_ns
        self._cache_valid = False
    
    def add_tick_y(self, timestamp_ns: int, price: float, volume: float = 0.0):
        """Add tick for asset Y."""
        tick = TickData(timestamp_ns=timestamp_ns, price=price, volume=volume, side=0)
        self.ticks_y.append(tick)
        
        # Calculate return
        if self.last_price_y is not None:
            ret = np.log(price / self.last_price_y)
            self.returns_y.append((timestamp_ns, ret))
        
        self.last_price_y = price
        self.last_ts_y = timestamp_ns
        self._cache_valid = False
    
    def compute_correlation(self) -> Optional[HYResult]:
        """
        Compute Hayashi-Yoshida correlation estimate.
        
        Returns:
            HYResult with correlation and lead-lag metrics
        """
        if len(self.returns_x) < self.min_ticks or len(self.returns_y) < self.min_ticks:
            return None
        
        # Convert to numpy arrays for efficient computation
        ret_x_arr = np.array([(ts, ret) for ts, ret in self.returns_x])
        ret_y_arr = np.array([(ts, ret) for ts, ret in self.returns_y])
        
        ts_x = ret_x_arr[:, 0].astype(np.int64)
        rets_x = ret_x_arr[:, 1].astype(np.float64)
        ts_y = ret_y_arr[:, 0].astype(np.int64)
        rets_y = ret_y_arr[:, 1].astype(np.float64)
        
        # Convert max_lag to nanoseconds
        max_lag_ns = int(self.max_lag_ms * 1e6)
        
        # Hayashi-Yoshida estimator: sum of products of overlapping returns
        hy_cov = 0.0
        var_x = 0.0
        var_y = 0.0
        
        n_overlaps = 0
        
        # For each return in X, find overlapping returns in Y
        for i in range(len(ts_x)):
            ts_start_x = ts_x[i-1] if i > 0 else ts_x[0]
            ts_end_x = ts_x[i]
            
            # Find Y returns that overlap with [ts_start_x, ts_end_x]
            # A Y return at time j overlaps if ts_y[j-1] < ts_end_x AND ts_y[j] > ts_start_x
            for j in range(len(ts_y)):
                ts_start_y = ts_y[j-1] if j > 0 else ts_y[0]
                ts_end_y = ts_y[j]
                
                # Check for overlap
                if ts_start_y < ts_end_x and ts_end_y > ts_start_x:
                    # Overlapping interval exists
                    hy_cov += rets_x[i] * rets_y[j]
                    n_overlaps += 1
        
        # Calculate variances (realized variance)
        var_x = np.sum(rets_x ** 2)
        var_y = np.sum(rets_y ** 2)
        
        # Correlation
        denom = np.sqrt(var_x * var_y)
        if denom < 1e-10:
            correlation = 0.0
        else:
            correlation = hy_cov / denom
        
        # Clip to valid range
        correlation = np.clip(correlation, -1.0, 1.0)
        
        # Determine leader using cross-correlation at different lags
        leader, lag_ms, confidence = self._detect_leader(ts_x, rets_x, ts_y, rets_y, max_lag_ns)
        
        self._cached_result = HYResult(
            correlation=correlation,
            n_overlaps=n_overlaps,
            leader=leader,
            lag_ms=lag_ms,
            confidence=confidence
        )
        self._cache_valid = True
        
        return self._cached_result
    
    def _detect_leader(self, ts_x: np.ndarray, rets_x: np.ndarray,
                       ts_y: np.ndarray, rets_y: np.ndarray,
                       max_lag_ns: int) -> Tuple[Optional[str], float, float]:
        """
        Detect which asset leads the other using lagged cross-correlation.
        
        Returns:
            Tuple of (leader_name, lag_ms, confidence)
        """
        if len(rets_x) < 20 or len(rets_y) < 20:
            return None, 0.0, 0.0
        
        # Test multiple lag values
        n_lags = 10
        lag_step_ns = max_lag_ns // n_lags
        
        best_corr = 0.0
        best_lag_ns = 0
        leader = None
        
        for lag_idx in range(-n_lags, n_lags + 1):
            lag_ns = lag_idx * lag_step_ns
            
            if lag_ns > 0:
                # X leads Y: correlate rets_x[t] with rets_y[t + lag]
                corr = self._lagged_correlation(rets_x, rets_y, ts_x, ts_y, lag_ns)
                if abs(corr) > abs(best_corr):
                    best_corr = corr
                    best_lag_ns = lag_ns
                    leader = "X" if corr > 0 else "Y"
            elif lag_ns < 0:
                # Y leads X: correlate rets_y[t] with rets_x[t - lag]
                corr = self._lagged_correlation(rets_y, rets_x, ts_y, ts_x, -lag_ns)
                if abs(corr) > abs(best_corr):
                    best_corr = corr
                    best_lag_ns = lag_ns
                    leader = "Y" if corr > 0 else "X"
        
        if leader is None or abs(best_corr) < 0.1:
            return None, 0.0, 0.0
        
        lag_ms = abs(best_lag_ns) / 1e6
        confidence = min(abs(best_corr), 1.0)
        
        return leader, lag_ms, confidence
    
    def _lagged_correlation(self, rets_1: np.ndarray, rets_2: np.ndarray,
                            ts_1: np.ndarray, ts_2: np.ndarray,
                            lag_ns: int) -> float:
        """
        Compute correlation between two return series with a time lag.
        Positive lag means series 1 leads series 2.
        """
        if lag_ns <= 0:
            return 0.0
        
        # Shift series 2 forward by lag_ns
        aligned_rets_2 = []
        aligned_rets_1 = []
        
        j = 0
        for i in range(len(ts_1)):
            target_ts = ts_1[i] + lag_ns
            
            # Find closest return in series 2 after target_ts
            while j < len(ts_2) and ts_2[j] < target_ts:
                j += 1
            
            if j < len(ts_2):
                # Check if within reasonable tolerance (10% of lag)
                tolerance = max(lag_ns // 10, 1000000)  # At least 1ms
                if abs(ts_2[j] - target_ts) < tolerance:
                    aligned_rets_1.append(rets_1[i])
                    aligned_rets_2.append(rets_2[j])
        
        if len(aligned_rets_1) < 10:
            return 0.0
        
        arr_1 = np.array(aligned_rets_1)
        arr_2 = np.array(aligned_rets_2)
        
        # Pearson correlation
        if np.std(arr_1) < 1e-10 or np.std(arr_2) < 1e-10:
            return 0.0
        
        corr = np.corrcoef(arr_1, arr_2)[0, 1]
        return corr if not np.isnan(corr) else 0.0
    
    def get_realized_correlation(self, window_ms: float = 60000.0) -> Optional[float]:
        """
        Get standard realized correlation over recent window.
        Less accurate for non-synchronous data but faster.
        
        Args:
            window_ms: Window size in milliseconds
            
        Returns:
            Realized correlation or None
        """
        if len(self.returns_x) < 10 or len(self.returns_y) < 10:
            return None
        
        current_ts = max(self.last_ts_x, self.last_ts_y)
        window_ns = int(window_ms * 1e6)
        cutoff_ts = current_ts - window_ns
        
        # Filter recent returns
        recent_x = [ret for ts, ret in self.returns_x if ts > cutoff_ts]
        recent_y = [ret for ts, ret in self.returns_y if ts > cutoff_ts]
        
        if len(recent_x) < 5 or len(recent_y) < 5:
            return None
        
        arr_x = np.array(recent_x)
        arr_y = np.array(recent_y)
        
        # Pad shorter array (simple approach - not ideal)
        min_len = min(len(arr_x), len(arr_y))
        arr_x = arr_x[-min_len:]
        arr_y = arr_y[-min_len:]
        
        if np.std(arr_x) < 1e-10 or np.std(arr_y) < 1e-10:
            return 0.0
        
        corr = np.corrcoef(arr_x, arr_y)[0, 1]
        return corr if not np.isnan(corr) else 0.0
    
    def reset(self):
        """Reset all state."""
        self.ticks_x.clear()
        self.ticks_y.clear()
        self.returns_x.clear()
        self.returns_y.clear()
        self.last_price_x = None
        self.last_price_y = None
        self._cached_result = None
        self._cache_valid = False


class LeadLagMonitor:
    """
    Monitors lead-lag relationships across multiple asset pairs.
    Optimized for real-time detection of leading assets.
    """
    
    def __init__(self, pairs: List[Tuple[str, str]], **hy_kwargs):
        """
        Args:
            pairs: List of (leader_candidate, follower_candidate) tuples
            **hy_kwargs: Arguments passed to HayashiYoshidaEstimator
        """
        self.pairs = pairs
        self.estimators = {}
        
        for pair in pairs:
            self.estimators[pair] = HayashiYoshidaEstimator(**hy_kwargs)
        
        # Track detected leaders
        self.leader_cache = {}
        
    def add_tick(self, asset: str, timestamp_ns: int, price: float):
        """
        Add tick for any monitored asset.
        
        Args:
            asset: Asset symbol
            timestamp_ns: Timestamp in nanoseconds
            price: Price
        """
        for pair, estimator in self.estimators.items():
            asset_x, asset_y = pair
            if asset == asset_x:
                estimator.add_tick_x(timestamp_ns, price)
            elif asset == asset_y:
                estimator.add_tick_y(timestamp_ns, price)
    
    def update_leaders(self) -> Dict[Tuple[str, str], HYResult]:
        """
        Update lead-lag detection for all pairs.
        
        Returns:
            Dictionary mapping pairs to their HY results
        """
        results = {}
        
        for pair, estimator in self.estimators.items():
            result = estimator.compute_correlation()
            if result is not None:
                results[pair] = result
                
                # Update leader cache
                if result.leader is not None:
                    leader_asset = pair[0] if result.leader == "X" else pair[1]
                    self.leader_cache[pair] = {
                        'leader': leader_asset,
                        'lag_ms': result.lag_ms,
                        'confidence': result.confidence,
                        'correlation': result.correlation
                    }
        
        return results
    
    def get_leader_signals(self, min_confidence: float = 0.3) -> List[Dict]:
        """
        Get actionable lead-lag signals.
        
        Args:
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of signal dictionaries
        """
        signals = []
        
        for pair, info in self.leader_cache.items():
            if info['confidence'] >= min_confidence:
                signals.append({
                    'pair': f"{pair[1]}/{pair[0]}",
                    'leader': info['leader'],
                    'follower': pair[0] if info['leader'] == pair[1] else pair[1],
                    'lag_ms': info['lag_ms'],
                    'correlation': info['correlation'],
                    'confidence': info['confidence'],
                    'action': f"Watch {info['follower']} for moves following {info['leader']}"
                })
        
        return signals
    
    def reset_pair(self, pair: Tuple[str, str]):
        """Reset estimator for a specific pair."""
        if pair in self.estimators:
            self.estimators[pair].reset()
            if pair in self.leader_cache:
                del self.leader_cache[pair]


__all__ = [
    'HayashiYoshidaEstimator',
    'LeadLagMonitor',
    'TickData',
    'HYResult'
]
