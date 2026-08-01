"""
Chapter 4: Latency Arbitrage & Queue Position ML Prediction
cross_venue_lag.py - Real-time cross-venue lag tracker using Hayashi-Yoshida estimators
"""

import numpy as np
from numba import njit, prange
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass
from collections import deque


@dataclass
class VenuePairLag:
    """Lag measurement between two venues"""
    venue_a: str
    venue_b: str
    estimated_lag_ms: float
    confidence: float
    sample_count: int
    last_update_ns: int
    correlation: float
    lead_lag_direction: int  # 1 = A leads B, -1 = B leads A, 0 = unclear


@njit(cache=True, nogil=True)
def hayashi_yoshida_covariance(
    times_a: np.ndarray,
    returns_a: np.ndarray,
    times_b: np.ndarray,
    returns_b: np.ndarray
) -> float:
    """
    Calculate Hayashi-Yoshida covariance for asynchronous time series.
    
    This estimator correctly handles non-synchronous observations,
    which is critical for cross-venue analysis where trades occur
    at different times on different exchanges.
    
    Args:
        times_a: Timestamps for venue A (nanoseconds)
        returns_a: Returns for venue A
        times_b: Timestamps for venue B (nanoseconds)
        returns_b: Returns for venue B
    
    Returns:
        HY covariance estimate
    """
    n_a = len(times_a)
    n_b = len(times_b)
    
    if n_a < 2 or n_b < 2:
        return 0.0
    
    cov_sum = 0.0
    
    # For each observation in A, find overlapping observations in B
    for i in range(1, n_a):
        t_a_prev = times_a[i - 1]
        t_a_curr = times_a[i]
        
        # Find all B observations that overlap with this interval
        for j in range(1, n_b):
            t_b_prev = times_b[j - 1]
            t_b_curr = times_b[j]
            
            # Check for overlap
            overlap_start = max(t_a_prev, t_b_prev)
            overlap_end = min(t_a_curr, t_b_curr)
            
            if overlap_start < overlap_end:
                # There is overlap - contribute to covariance
                cov_sum += returns_a[i] * returns_b[j]
    
    return cov_sum


@njit(cache=True, nogil=True)
def realized_variance(
    times: np.ndarray,
    returns: np.ndarray
) -> float:
    """Calculate realized variance from returns."""
    n = len(returns)
    if n == 0:
        return 0.0
    
    var_sum = 0.0
    for r in returns:
        var_sum += r * r
    
    return var_sum


@njit(cache=True, nogil=True)
def cross_correlation_lag(
    returns_a: np.ndarray,
    returns_b: np.ndarray,
    max_lag: int = 20
) -> Tuple[int, float]:
    """
    Find optimal lag that maximizes cross-correlation.
    
    Args:
        returns_a: Returns from venue A (assumed faster)
        returns_b: Returns from venue B (assumed slower)
        max_lag: Maximum lag to search
    
    Returns:
        Tuple of (optimal_lag, correlation_at_lag)
    """
    n = min(len(returns_a), len(returns_b))
    
    if n < max_lag + 10:
        return 0, 0.0
    
    best_lag = 0
    best_corr = -2.0
    
    for lag in range(-max_lag, max_lag + 1):
        # Calculate correlation at this lag
        sum_xy = 0.0
        sum_x = 0.0
        sum_y = 0.0
        sum_xx = 0.0
        sum_yy = 0.0
        count = 0
        
        for i in range(n - abs(lag)):
            if lag >= 0:
                x = returns_a[i + lag]
                y = returns_b[i]
            else:
                x = returns_a[i]
                y = returns_b[i - lag]
            
            sum_xy += x * y
            sum_x += x
            sum_y += y
            sum_xx += x * x
            sum_yy += y * y
            count += 1
        
        if count < 10:
            continue
        
        # Pearson correlation
        numerator = count * sum_xy - sum_x * sum_y
        denom_x = count * sum_xx - sum_x * sum_x
        denom_y = count * sum_yy - sum_y * sum_y
        
        if denom_x <= 0 or denom_y <= 0:
            continue
        
        corr = numerator / np.sqrt(denom_x * denom_y)
        
        if corr > best_corr:
            best_corr = corr
            best_lag = lag
    
    return best_lag, best_corr


@njit(cache=True, nogil=True)
def estimate_latency_ms(
    times_a: np.ndarray,
    prices_a: np.ndarray,
    times_b: np.ndarray,
    prices_b: np.ndarray,
    tick_size: float
) -> float:
    """
    Estimate latency between venues by detecting price discovery.
    
    When venue A moves first and venue B follows, we can estimate
    the latency by measuring the time delay.
    
    Args:
        times_a: Timestamps for venue A
        prices_a: Prices for venue A
        times_b: Timestamps for venue B
        prices_b: Prices for venue B
        tick_size: Price tick size
    
    Returns:
        Estimated latency in milliseconds
    """
    n_a = len(times_a)
    n_b = len(times_b)
    
    if n_a < 10 or n_b < 10:
        return 0.0
    
    # Convert prices to integer ticks for comparison
    ticks_a = (prices_a / tick_size).astype(np.int64)
    ticks_b = (prices_b / tick_size).astype(np.int64)
    
    # Detect price changes
    changes_a = np.zeros(n_a, dtype=np.bool_)
    changes_b = np.zeros(n_b, dtype=np.bool_)
    
    for i in range(1, n_a):
        if ticks_a[i] != ticks_a[i - 1]:
            changes_a[i] = True
    
    for i in range(1, n_b):
        if ticks_b[i] != ticks_b[i - 1]:
            changes_b[i] = True
    
    # Match price changes and measure delays
    total_delay = 0.0
    match_count = 0
    
    for i in range(1, n_a):
        if not changes_a[i]:
            continue
        
        # Find matching change in B
        price_change = ticks_a[i] - ticks_a[i - 1]
        
        for j in range(1, n_b):
            if not changes_b[j]:
                continue
            
            b_change = ticks_b[j] - ticks_b[j - 1]
            
            if b_change == price_change:
                # Found matching change
                delay_ns = times_b[j] - times_a[i]
                
                if delay_ns > 0 and delay_ns < 1_000_000_000:  # Less than 1 second
                    total_delay += delay_ns
                    match_count += 1
                    break
    
    if match_count == 0:
        return 0.0
    
    avg_delay_ns = total_delay / match_count
    
    # Convert to milliseconds
    return avg_delay_ns / 1_000_000


class CrossVenueLagTracker:
    """
    Real-time tracker for cross-venue latency and lead-lag relationships.
    Uses Hayashi-Yoshida estimators for asynchronous data.
    """
    
    def __init__(
        self,
        venue_names: List[str],
        window_size: int = 1000,
        min_samples: int = 50
    ):
        self.venue_names = venue_names
        self.window_size = window_size
        self.min_samples = min_samples
        
        # Per-venue data buffers
        self._venue_times: Dict[str, deque] = {v: deque(maxlen=window_size) for v in venue_names}
        self._venue_prices: Dict[str, deque] = {v: deque(maxlen=window_size) for v in venue_names}
        self._venue_returns: Dict[str, deque] = {v: deque(maxlen=window_size) for v in venue_names}
        
        # Lag estimates
        self._lag_estimates: Dict[Tuple[str, str], VenuePairLag] = {}
        
        # Tick sizes per venue
        self._tick_sizes: Dict[str, float] = {}
        
        # Update counter
        self._update_count = 0
    
    def set_tick_size(self, venue: str, tick_size: float):
        """Set tick size for a venue."""
        self._tick_sizes[venue] = tick_size
    
    def add_observation(
        self,
        venue: str,
        timestamp_ns: int,
        price: float
    ):
        """Add a new price observation for a venue."""
        if venue not in self.venue_names:
            return
        
        # Add to buffers
        self._venue_times[venue].append(timestamp_ns)
        self._venue_prices[venue].append(price)
        
        # Calculate return
        if len(self._venue_prices[venue]) > 1:
            prev_price = list(self._venue_prices[venue])[-2]
            if prev_price > 0:
                ret = (price - prev_price) / prev_price
            else:
                ret = 0.0
        else:
            ret = 0.0
        
        self._venue_returns[venue].append(ret)
        
        self._update_count += 1
        
        # Periodically update lag estimates
        if self._update_count % 100 == 0:
            self.update_lag_estimates()
    
    def update_lag_estimates(self):
        """Update all pairwise lag estimates."""
        for i, venue_a in enumerate(self.venue_names):
            for venue_b in self.venue_names[i + 1:]:
                self._estimate_pairwise_lag(venue_a, venue_b)
    
    def _estimate_pairwise_lag(self, venue_a: str, venue_b: str):
        """Estimate lag between two specific venues."""
        times_a = np.array(list(self._venue_times[venue_a]), dtype=np.float64)
        prices_a = np.array(list(self._venue_prices[venue_a]))
        returns_a = np.array(list(self._venue_returns[venue_a]))
        
        times_b = np.array(list(self._venue_times[venue_b]), dtype=np.float64)
        prices_b = np.array(list(self._venue_prices[venue_b]))
        returns_b = np.array(list(self._venue_returns[venue_b]))
        
        if len(times_a) < self.min_samples or len(times_b) < self.min_samples:
            return
        
        # Get tick sizes
        tick_a = self._tick_sizes.get(venue_a, 0.01)
        tick_b = self._tick_sizes.get(venue_b, 0.01)
        avg_tick = (tick_a + tick_b) / 2
        
        # Method 1: Cross-correlation lag
        lag_idx, corr = cross_correlation_lag(returns_a, returns_b)
        
        # Method 2: Direct latency estimation
        latency_ms = estimate_latency_ms(times_a, prices_a, times_b, prices_b, avg_tick)
        
        # Method 3: Hayashi-Yoshida covariance
        hy_cov = hayashi_yoshida_covariance(times_a, returns_a, times_b, returns_b)
        
        # Combine estimates
        # Convert index lag to time using median sampling interval
        if len(times_a) > 1:
            median_interval_a = np.median(np.diff(times_a))
            lag_from_corr_ms = (lag_idx * median_interval_a) / 1_000_000  # ns to ms
        else:
            lag_from_corr_ms = 0.0
        
        # Weighted combination
        if abs(corr) > 0.3:
            estimated_lag = lag_from_corr_ms * 0.5 + latency_ms * 0.5
        else:
            estimated_lag = latency_ms
        
        # Determine direction
        if lag_idx > 5 or latency_ms > 10:
            direction = 1  # A leads B
        elif lag_idx < -5 or latency_ms < -10:
            direction = -1  # B leads A
        else:
            direction = 0
        
        # Calculate confidence based on sample size and correlation
        n_samples = min(len(times_a), len(times_b))
        confidence = min(1.0, abs(corr)) * min(1.0, n_samples / self.window_size)
        
        # Store estimate
        key = (venue_a, venue_b)
        self._lag_estimates[key] = VenuePairLag(
            venue_a=venue_a,
            venue_b=venue_b,
            estimated_lag_ms=estimated_lag,
            confidence=confidence,
            sample_count=n_samples,
            last_update_ns=int(times_a[-1]) if len(times_a) > 0 else 0,
            correlation=corr,
            lead_lag_direction=direction
        )
    
    def get_lag_estimate(
        self,
        venue_a: str,
        venue_b: str
    ) -> Optional[VenuePairLag]:
        """Get current lag estimate between two venues."""
        key = (venue_a, venue_b)
        if key in self._lag_estimates:
            return self._lag_estimates[key]
        
        # Try reverse
        key_rev = (venue_b, venue_a)
        if key_rev in self._lag_estimates:
            rev_lag = self._lag_estimates[key_rev]
            return VenuePairLag(
                venue_a=venue_a,
                venue_b=venue_b,
                estimated_lag_ms=-rev_lag.estimated_lag_ms,
                confidence=rev_lag.confidence,
                sample_count=rev_lag.sample_count,
                last_update_ns=rev_lag.last_update_ns,
                correlation=rev_lag.correlation,
                lead_lag_direction=-rev_lag.lead_lag_direction
            )
        
        return None
    
    def get_arbitrage_opportunities(
        self,
        min_lag_ms: float = 5.0,
        min_confidence: float = 0.5
    ) -> List[VenuePairLag]:
        """
        Identify potential latency arbitrage opportunities.
        
        Args:
            min_lag_ms: Minimum lag to consider
            min_confidence: Minimum confidence threshold
        
        Returns:
            List of venue pairs with significant lag
        """
        opportunities = []
        
        for key, lag in self._lag_estimates.items():
            if (abs(lag.estimated_lag_ms) >= min_lag_ms and 
                lag.confidence >= min_confidence and
                lag.lead_lag_direction != 0):
                opportunities.append(lag)
        
        return sorted(opportunities, key=lambda x: abs(x.estimated_lag_ms), reverse=True)
    
    def get_hayashi_yoshida_correlation(
        self,
        venue_a: str,
        venue_b: str
    ) -> float:
        """Get HY correlation between two venues."""
        times_a = np.array(list(self._venue_times[venue_a]), dtype=np.float64)
        returns_a = np.array(list(self._venue_returns[venue_a]))
        times_b = np.array(list(self._venue_times[venue_b]), dtype=np.float64)
        returns_b = np.array(list(self._venue_returns[venue_b]))
        
        if len(times_a) < 10 or len(times_b) < 10:
            return 0.0
        
        hy_cov = hayashi_yoshida_covariance(times_a, returns_a, times_b, returns_b)
        
        var_a = realized_variance(times_a, returns_a)
        var_b = realized_variance(times_b, returns_b)
        
        if var_a <= 0 or var_b <= 0:
            return 0.0
        
        return hy_cov / np.sqrt(var_a * var_b)


# Module convenience functions
def create_lag_tracker(
    venues: List[str],
    window_size: int = 1000
) -> CrossVenueLagTracker:
    """Factory function to create cross-venue lag tracker."""
    return CrossVenueLagTracker(venues, window_size)


def quick_cross_venue_lag(
    times_a: np.ndarray,
    prices_a: np.ndarray,
    times_b: np.ndarray,
    prices_b: np.ndarray,
    tick_size: float = 0.01
) -> float:
    """Quick lag estimation between two venues."""
    return estimate_latency_ms(times_a, prices_a, times_b, prices_b, tick_size)
