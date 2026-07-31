"""
Thermal/Optimal Lead-Lag Detector using Lagged Cross-Correlation Matrices.
Implements Hoffman-Jørgensen inequalities for statistical significance testing.
Identifies optimal lag structure between assets for predictive trading.
Strictly NumPy/SciPy based - no Pandas in hot path.
"""

import numpy as np
from scipy import stats, optimize
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
from enum import Enum


class ThermalState(Enum):
    """Thermal state of the lead-lag relationship."""
    HOT = "hot"       # Strong, stable lead-lag
    WARM = "warm"     # Moderate signal
    COLD = "cold"     # Weak or noisy relationship
    FROZEN = "frozen" # No detectable relationship


@dataclass
class ThermalOptimalResult:
    """Result from thermal optimal lag detection."""
    optimal_lag_ms: float
    correlation_at_optimal: float
    thermal_state: ThermalState
    significance_pvalue: float
    confidence_interval: Tuple[float, float]
    information_flow: float  # Directional information measure


class ThermalOptimalDetector:
    """
    Detects optimal lead-lag structure using thermal cross-correlation analysis.
    
    The "thermal" metaphor refers to how the correlation "heats up" at the 
    optimal lag value where predictive information is maximized.
    
    Uses Hoffman-Jørgensen inequality for bounding tail probabilities
    of the correlation estimator under the null hypothesis.
    """
    
    def __init__(self,
                 max_lag_ms: float = 5000.0,
                 min_lag_ms: float = 1.0,
                 n_lag_bins: int = 50,
                 confidence_level: float = 0.95,
                 min_samples: int = 100):
        """
        Args:
            max_lag_ms: Maximum lag to search in milliseconds
            min_lag_ms: Minimum lag resolution
            n_lag_bins: Number of lag bins to evaluate
            confidence_level: Confidence level for significance testing
            min_samples: Minimum samples required for estimation
        """
        self.max_lag_ms = max_lag_ms
        self.min_lag_ms = min_lag_ms
        self.n_lag_bins = n_lag_bins
        self.confidence_level = confidence_level
        self.min_samples = min_samples
        
        # Generate lag grid (logarithmic spacing for better resolution at small lags)
        self.lag_grid_ms = np.logspace(
            np.log10(min_lag_ms), 
            np.log10(max_lag_ms), 
            n_lag_bins
        )
        
        # Data buffers
        self.buffer_size = 5000
        self.returns_x = np.zeros(self.buffer_size)
        self.returns_y = np.zeros(self.buffer_size)
        self.timestamps_x = np.zeros(self.buffer_size, dtype=np.int64)
        self.timestamps_y = np.zeros(self.buffer_size, dtype=np.int64)
        self.buf_idx_x = 0
        self.buf_idx_y = 0
        
        # Last prices
        self.last_price_x: Optional[float] = None
        self.last_price_y: Optional[float] = None
        
        # Cached results
        self._cached_result: Optional[ThermalOptimalResult] = None
        
    def add_return_x(self, timestamp_ns: int, return_val: float):
        """Add return observation for asset X."""
        idx = self.buf_idx_x % self.buffer_size
        self.returns_x[idx] = return_val
        self.timestamps_x[idx] = timestamp_ns
        self.buf_idx_x += 1
    
    def add_return_y(self, timestamp_ns: int, return_val: float):
        """Add return observation for asset Y."""
        idx = self.buf_idx_y % self.buffer_size
        self.returns_y[idx] = return_val
        self.timestamps_y[idx] = timestamp_ns
        self.buf_idx_y += 1
    
    def add_price_x(self, timestamp_ns: int, price: float):
        """Add price for asset X and compute return."""
        if self.last_price_x is not None:
            ret = np.log(price / self.last_price_x)
            self.add_return_x(timestamp_ns, ret)
        self.last_price_x = price
    
    def add_price_y(self, timestamp_ns: int, price: float):
        """Add price for asset Y and compute return."""
        if self.last_price_y is not None:
            ret = np.log(price / self.last_price_y)
            self.add_return_y(timestamp_ns, ret)
        self.last_price_y = price
    
    def compute_optimal_lag(self) -> Optional[ThermalOptimalResult]:
        """
        Compute the optimal lag that maximizes cross-correlation.
        
        Returns:
            ThermalOptimalResult with optimal lag and statistics
        """
        n_x = min(self.buf_idx_x, self.buffer_size)
        n_y = min(self.buf_idx_y, self.buffer_size)
        
        if n_x < self.min_samples or n_y < self.min_samples:
            return None
        
        # Get recent returns
        start_x = max(0, self.buf_idx_x - self.buffer_size)
        start_y = max(0, self.buf_idx_y - self.buffer_size)
        
        rets_x = self.returns_x[start_x:self.buf_idx_x]
        rets_y = self.returns_y[start_y:self.buf_idx_y]
        ts_x = self.timestamps_x[start_x:self.buf_idx_x]
        ts_y = self.timestamps_y[start_y:self.buf_idx_y]
        
        # Compute cross-correlation at each lag
        correlations = np.zeros(len(self.lag_grid_ms))
        p_values = np.zeros(len(self.lag_grid_ms))
        
        for i, lag_ms in enumerate(self.lag_grid_ms):
            corr, pval = self._cross_correlation_at_lag(
                rets_x, rets_y, ts_x, ts_y, lag_ms
            )
            correlations[i] = corr
            p_values[i] = pval
        
        # Find optimal lag (maximum absolute correlation)
        abs_corr = np.abs(correlations)
        optimal_idx = np.argmax(abs_corr)
        optimal_lag_ms = self.lag_grid_ms[optimal_idx]
        optimal_corr = correlations[optimal_idx]
        
        # Compute confidence interval using Fisher transformation
        z = np.arctanh(optimal_corr)
        z_se = 1.0 / np.sqrt(n_x - 3) if n_x > 3 else 1.0
        z_crit = stats.norm.ppf((1 + self.confidence_level) / 2)
        
        ci_lower = np.tanh(z - z_crit * z_se)
        ci_upper = np.tanh(z + z_crit * z_se)
        
        # Hoffman-Jørgensen bound for tail probability
        # P(|rho_hat - rho| > t) <= 2 * exp(-n * t^2 / 2) for sub-Gaussian
        hoffman_bound = self._hoffman_jorgensen_bound(optimal_corr, n_x)
        
        # Determine thermal state
        thermal_state = self._classify_thermal_state(
            optimal_corr, p_values[optimal_idx], hoffman_bound
        )
        
        # Calculate directional information flow
        info_flow = self._calculate_information_flow(rets_x, rets_y, optimal_lag_ms)
        
        result = ThermalOptimalResult(
            optimal_lag_ms=optimal_lag_ms,
            correlation_at_optimal=optimal_corr,
            thermal_state=thermal_state,
            significance_pvalue=p_values[optimal_idx],
            confidence_interval=(ci_lower, ci_upper),
            information_flow=info_flow
        )
        
        self._cached_result = result
        return result
    
    def _cross_correlation_at_lag(self, rets_x: np.ndarray, rets_y: np.ndarray,
                                   ts_x: np.ndarray, ts_y: np.ndarray,
                                   lag_ms: float) -> Tuple[float, float]:
        """
        Compute cross-correlation at a specific lag.
        
        Args:
            rets_x, rets_y: Return series
            ts_x, ts_y: Timestamp arrays (nanoseconds)
            lag_ms: Lag in milliseconds (positive means X leads Y)
            
        Returns:
            Tuple of (correlation, p-value)
        """
        lag_ns = int(lag_ms * 1e6)
        
        # Align series with lag
        aligned_pairs = []
        
        # For each X return, find corresponding Y return at t + lag
        j = 0
        for i in range(len(ts_x)):
            target_ts = ts_x[i] + lag_ns
            
            # Binary search for closest Y timestamp
            while j < len(ts_y) and ts_y[j] < target_ts:
                j += 1
            
            if j < len(ts_y):
                # Check if within tolerance (1% of lag or 1ms minimum)
                tolerance = max(lag_ms * 0.01 * 1e6, 1e6)
                if abs(ts_y[j] - target_ts) < tolerance:
                    aligned_pairs.append((rets_x[i], rets_y[j]))
        
        if len(aligned_pairs) < 20:
            return 0.0, 1.0
        
        pairs_arr = np.array(aligned_pairs)
        if len(pairs_arr.shape) != 2 or pairs_arr.shape[1] != 2:
            return 0.0, 1.0
            
        arr_x = pairs_arr[:, 0]
        arr_y = pairs_arr[:, 1]
        
        # Pearson correlation
        if np.std(arr_x) < 1e-10 or np.std(arr_y) < 1e-10:
            return 0.0, 1.0
        
        corr, pval = stats.pearsonr(arr_x, arr_y)
        return corr if not np.isnan(corr) else 0.0, pval
    
    def _hoffman_jorgensen_bound(self, rho_hat: float, n: int) -> float:
        """
        Compute Hoffman-Jørgensen inequality bound for correlation estimator.
        
        The bound provides: P(|rho_hat - rho| > t) <= 2 * exp(-n * t^2 / C)
        where C is a constant depending on the distribution.
        
        For our purposes, we use this to assess if observed correlation
        is statistically distinguishable from zero.
        """
        # Simplified bound assuming sub-Gaussian returns
        # In practice, financial returns have heavier tails
        t = abs(rho_hat)
        
        if t < 1e-10:
            return 1.0
        
        # Conservative bound with C = 4 (accounts for heavy tails)
        C = 4.0
        bound = 2.0 * np.exp(-n * t * t / C)
        
        return min(bound, 1.0)
    
    def _classify_thermal_state(self, corr: float, pval: float, 
                                 hoffman_bound: float) -> ThermalState:
        """Classify the thermal state based on correlation strength and significance."""
        abs_corr = abs(corr)
        
        # Check statistical significance first
        if pval > 0.1 or hoffman_bound > 0.5:
            return ThermalState.FROZEN
        
        # Classify by correlation strength
        if abs_corr > 0.5:
            return ThermalState.HOT
        elif abs_corr > 0.3:
            return ThermalState.WARM
        elif abs_corr > 0.1:
            return ThermalState.COLD
        else:
            return ThermalState.FROZEN
    
    def _calculate_information_flow(self, rets_x: np.ndarray, rets_y: np.ndarray,
                                     lag_ms: float) -> float:
        """
        Calculate directional information flow metric.
        
        Based on transfer entropy approximation using correlation asymmetry.
        Positive value indicates X -> Y information flow.
        """
        lag_ns = int(lag_ms * 1e6)
        
        # Forward correlation (X leads Y)
        corr_forward, _ = self._cross_correlation_at_lag(
            rets_x, rets_y, 
            self.timestamps_x[max(0, self.buf_idx_x - self.buffer_size):self.buf_idx_x],
            self.timestamps_y[max(0, self.buf_idx_y - self.buffer_size):self.buf_idx_y],
            lag_ms
        )
        
        # Backward correlation (Y leads X)
        corr_backward, _ = self._cross_correlation_at_lag(
            rets_y, rets_x,
            self.timestamps_y[max(0, self.buf_idx_y - self.buffer_size):self.buf_idx_y],
            self.timestamps_x[max(0, self.buf_idx_x - self.buffer_size):self.buf_idx_x],
            lag_ms
        )
        
        # Information flow is the asymmetry
        info_flow = corr_forward - corr_backward
        
        return info_flow
    
    def get_cross_correlation_matrix(self, n_assets: int = 3) -> np.ndarray:
        """
        Get full cross-correlation matrix at optimal lag.
        Placeholder for multi-asset extension.
        
        Returns:
            Correlation matrix (identity for single pair)
        """
        if self._cached_result is None:
            return np.eye(2)
        
        # 2x2 correlation matrix for the pair
        rho = self._cached_result.correlation_at_optimal
        return np.array([
            [1.0, rho],
            [rho, 1.0]
        ])
    
    def reset(self):
        """Reset all state."""
        self.returns_x.fill(0)
        self.returns_y.fill(0)
        self.timestamps_x.fill(0)
        self.timestamps_y.fill(0)
        self.buf_idx_x = 0
        self.buf_idx_y = 0
        self.last_price_x = None
        self.last_price_y = None
        self._cached_result = None


class MultiAssetLeadLagTracker:
    """
    Tracks lead-lag relationships across multiple assets simultaneously.
    Builds a directed graph of information flow.
    """
    
    def __init__(self, assets: List[str], **detector_kwargs):
        """
        Args:
            assets: List of asset symbols to track
            **detector_kwargs: Arguments for ThermalOptimalDetector
        """
        self.assets = assets
        self.detectors = {}
        
        # Create detector for each pair
        for i, asset_x in enumerate(assets):
            for asset_y in assets[i+1:]:
                pair = (asset_x, asset_y)
                self.detectors[pair] = ThermalOptimalDetector(**detector_kwargs)
        
        # Leader graph
        self.leader_graph = {asset: [] for asset in assets}
        
    def update_price(self, asset: str, timestamp_ns: int, price: float):
        """Update price for an asset."""
        for pair, detector in self.detectors.items():
            if asset == pair[0]:
                detector.add_price_x(timestamp_ns, price)
            elif asset == pair[1]:
                detector.add_price_y(timestamp_ns, price)
    
    def analyze_all_pairs(self) -> Dict[Tuple[str, str], ThermalOptimalResult]:
        """Analyze all pairs and update leader graph."""
        results = {}
        
        # Clear leader graph
        self.leader_graph = {asset: [] for asset in self.assets}
        
        for pair, detector in self.detectors.items():
            result = detector.compute_optimal_lag()
            if result is not None:
                results[pair] = result
                
                # Update leader graph if significant
                if result.thermal_state in [ThermalState.HOT, ThermalState.WARM]:
                    if result.information_flow > 0:
                        # First asset leads second
                        self.leader_graph[pair[0]].append({
                            'follower': pair[1],
                            'lag_ms': result.optimal_lag_ms,
                            'strength': result.correlation_at_optimal
                        })
                    else:
                        # Second asset leads first
                        self.leader_graph[pair[1]].append({
                            'follower': pair[0],
                            'lag_ms': result.optimal_lag_ms,
                            'strength': abs(result.correlation_at_optimal)
                        })
        
        return results
    
    def get_top_leaders(self, n: int = 3) -> List[Dict]:
        """Get top N leader assets by number of followers."""
        leader_counts = []
        
        for asset, followers in self.leader_graph.items():
            if followers:
                total_strength = sum(f['strength'] for f in followers)
                leader_counts.append({
                    'asset': asset,
                    'n_followers': len(followers),
                    'total_strength': total_strength,
                    'followers': followers
                })
        
        # Sort by number of followers, then by total strength
        leader_counts.sort(key=lambda x: (x['n_followers'], x['total_strength']), reverse=True)
        
        return leader_counts[:n]
    
    def get_signal_for_pair(self, asset_x: str, asset_y: str) -> Optional[Dict]:
        """Get trading signal for a specific pair."""
        pair = (asset_x, asset_y)
        if pair not in self.detectors:
            pair = (asset_y, asset_x)
        
        if pair not in self.detectors:
            return None
        
        detector = self.detectors[pair]
        result = detector.compute_optimal_lag()
        
        if result is None or result.thermal_state == ThermalState.FROZEN:
            return None
        
        # Determine direction
        if result.information_flow > 0:
            leader, follower = asset_x, asset_y
        else:
            leader, follower = asset_y, asset_x
        
        return {
            'pair': f"{follower}/{leader}",
            'leader': leader,
            'follower': follower,
            'optimal_lag_ms': result.optimal_lag_ms,
            'correlation': result.correlation_at_optimal,
            'confidence': result.confidence_interval,
            'thermal_state': result.thermal_state.value,
            'action': f"Monitor {leader} for predictive signals on {follower}"
        }


__all__ = [
    'ThermalOptimalDetector',
    'MultiAssetLeadLagTracker',
    'ThermalOptimalResult',
    'ThermalState'
]
