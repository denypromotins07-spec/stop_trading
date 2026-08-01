"""
Rolling EWMA correlation matrix for BTC vs DXY, Yields, and SPX.
Uses pure numpy and numba JIT compilation for tick-by-tick updates without heap allocations.
"""

import numpy as np
from numba import njit, prange
from typing import Optional, Tuple, List
from dataclasses import dataclass
import threading


@njit(cache=True)
def ewma_update_1d(
    current_mean: float,
    new_value: float,
    alpha: float
) -> float:
    """Update EWMA for a single value."""
    return alpha * new_value + (1 - alpha) * current_mean


@njit(cache=True)
def ewma_update_covariance(
    current_cov: float,
    x_mean: float,
    y_mean: float,
    new_x: float,
    new_y: float,
    alpha: float
) -> float:
    """Update EWMA covariance estimate."""
    deviation_product = (new_x - x_mean) * (new_y - y_mean)
    return alpha * deviation_product + (1 - alpha) * current_cov


@njit(cache=True)
def compute_correlation_from_moments(
    cov_xy: float,
    var_x: float,
    var_y: float
) -> float:
    """Compute correlation coefficient from covariance and variances."""
    denom = np.sqrt(var_x * var_y)
    if denom < 1e-12:
        return 0.0
    return cov_xy / denom


@njit(parallel=True, cache=True)
def update_correlation_matrix_batch(
    means: np.ndarray,
    variances: np.ndarray,
    covariances: np.ndarray,
    new_values: np.ndarray,
    alpha: float,
    n_assets: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Update EWMA moments for multiple assets in parallel.
    Returns updated means, variances, and covariances.
    """
    new_means = means.copy()
    new_variances = variances.copy()
    new_covariances = covariances.copy()
    
    # Update means
    for i in range(n_assets):
        new_means[i] = ewma_update_1d(means[i], new_values[i], alpha)
    
    # Update variances and covariances
    for i in range(n_assets):
        for j in range(i, n_assets):
            if i == j:
                # Variance
                deviation_sq = (new_values[i] - means[i]) ** 2
                new_variances[i] = alpha * deviation_sq + (1 - alpha) * variances[i]
            else:
                # Covariance
                deviation_product = (new_values[i] - means[i]) * (new_values[j] - means[j])
                cov_idx = i * n_assets + j
                new_covariances[cov_idx] = alpha * deviation_product + (1 - alpha) * covariances[cov_idx]
    
    return new_means, new_variances, new_covariances


@njit(cache=True)
def correlations_from_moments(
    variances: np.ndarray,
    covariances: np.ndarray,
    n_assets: int
) -> np.ndarray:
    """Convert covariance moments to correlation matrix."""
    corr_matrix = np.zeros((n_assets, n_assets), dtype=np.float64)
    
    # Compute standard deviations
    stds = np.sqrt(variances)
    
    for i in range(n_assets):
        for j in range(n_assets):
            if i == j:
                corr_matrix[i, j] = 1.0
            else:
                # Get covariance (stored in upper triangle)
                if i < j:
                    cov_idx = i * n_assets + j
                else:
                    cov_idx = j * n_assets + i
                
                cov = covariances[cov_idx]
                
                if stds[i] > 1e-12 and stds[j] > 1e-12:
                    corr_matrix[i, j] = cov / (stds[i] * stds[j])
                else:
                    corr_matrix[i, j] = 0.0
    
    return corr_matrix


@dataclass
class CorrelationStats:
    """Statistics about the correlation matrix."""
    btc_dxy: float = 0.0
    btc_spx: float = 0.0
    btc_yields: float = 0.0
    dxy_spx: float = 0.0
    dxy_yields: float = 0.0
    spx_yields: float = 0.0
    avg_abs_corr: float = 0.0
    max_abs_corr: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "btc_dxy": self.btc_dxy,
            "btc_spx": self.btc_spx,
            "btc_yields": self.btc_yields,
            "dxy_spx": self.dxy_spx,
            "dxy_yields": self.dxy_yields,
            "spx_yields": self.spx_yields,
            "avg_abs_corr": self.avg_abs_corr,
            "max_abs_corr": self.max_abs_corr
        }


class RollingCorrelationMatrix:
    """
    High-performance rolling EWMA correlation matrix.
    Optimized for tick-by-tick updates with zero heap allocations in hot path.
    """
    
    # Asset indices
    BTC = 0
    DXY = 1
    YIELDS = 2
    SPX = 3
    
    ASSET_NAMES = ["BTC", "DXY", "YIELDS", "SPX"]
    
    def __init__(
        self,
        n_assets: int = 4,
        halflife: int = 50,
        warmup_periods: int = 100
    ):
        self.n_assets = n_assets
        self.halflife = halflife
        self.warmup_periods = warmup_periods
        
        # Calculate EWMA decay factor
        self.alpha = 1 - np.exp(-np.log(2) / halflife)
        
        # State variables (pre-allocated for zero-allocation updates)
        self._means = np.zeros(n_assets, dtype=np.float64)
        self._variances = np.ones(n_assets, dtype=np.float64) * 0.01  # Initial variance
        self._covariances = np.zeros(n_assets * n_assets, dtype=np.float64)
        
        # Current correlation matrix
        self._corr_matrix = np.eye(n_assets, dtype=np.float64)
        
        # Tracking
        self._update_count = 0
        self._is_warmed_up = False
        
        # Thread safety
        self._lock = threading.RLock()
        
        # History for diagnostics
        self._correlation_history: List[np.ndarray] = []
        self._history_max = 1000
    
    def update_tick(self, values: np.ndarray) -> np.ndarray:
        """
        Update correlation matrix with new tick values.
        Zero-allocation in hot path after warmup.
        
        Args:
            values: Array of asset values [BTC, DXY, YIELDS, SPX]
        
        Returns:
            Updated correlation matrix
        """
        if len(values) != self.n_assets:
            raise ValueError(f"Expected {self.n_assets} asset values, got {len(values)}")
        
        with self._lock:
            # Use Numba-optimized batch update
            self._means, self._variances, self._covariances = update_correlation_matrix_batch(
                self._means,
                self._variances,
                self._covariances,
                values.astype(np.float64),
                self.alpha,
                self.n_assets
            )
            
            self._update_count += 1
            
            # Check warmup status
            if self._update_count >= self.warmup_periods:
                self._is_warmed_up = True
            
            # Recompute correlation matrix
            self._corr_matrix = correlations_from_moments(
                self._variances,
                self._covariances,
                self.n_assets
            )
            
            # Store history (with limit)
            if len(self._correlation_history) >= self._history_max:
                self._correlation_history.pop(0)
            self._correlation_history.append(self._corr_matrix.copy())
            
            return self._corr_matrix.copy()
    
    def get_correlation_matrix(self) -> np.ndarray:
        """Get current correlation matrix."""
        with self._lock:
            return self._corr_matrix.copy()
    
    def get_correlation(self, asset1: int, asset2: int) -> float:
        """Get correlation between two specific assets."""
        with self._lock:
            return self._corr_matrix[asset1, asset2]
    
    def get_btc_correlations(self) -> Tuple[float, float, float]:
        """Get BTC correlations with DXY, Yields, and SPX."""
        with self._lock:
            return (
                self._corr_matrix[self.BTC, self.DXY],
                self._corr_matrix[self.BTC, self.YIELDS],
                self._corr_matrix[self.BTC, self.SPX]
            )
    
    def get_stats(self) -> CorrelationStats:
        """Get comprehensive correlation statistics."""
        with self._lock:
            stats = CorrelationStats(
                btc_dxy=self._corr_matrix[self.BTC, self.DXY],
                btc_spx=self._corr_matrix[self.BTC, self.SPX],
                btc_yields=self._corr_matrix[self.BTC, self.YIELDS],
                dxy_spx=self._corr_matrix[self.DXY, self.SPX],
                dxy_yields=self._corr_matrix[self.DXY, self.YIELDS],
                spx_yields=self._corr_matrix[self.SPX, self.YIELDS]
            )
            
            # Calculate aggregate stats
            abs_corr = np.abs(self._corr_matrix)
            # Exclude diagonal
            mask = ~np.eye(self.n_assets, dtype=bool)
            off_diag = abs_corr[mask]
            
            stats.avg_abs_corr = float(np.mean(off_diag))
            stats.max_abs_corr = float(np.max(off_diag))
            
            return stats
    
    def is_warmed_up(self) -> bool:
        """Check if the matrix has been warmed up."""
        return self._is_warmed_up
    
    def get_update_count(self) -> int:
        """Get number of updates performed."""
        return self._update_count
    
    def reset(self) -> None:
        """Reset all state."""
        with self._lock:
            self._means.fill(0)
            self._variances.fill(0.01)
            self._covariances.fill(0)
            self._corr_matrix = np.eye(self.n_assets, dtype=np.float64)
            self._update_count = 0
            self._is_warmed_up = False
            self._correlation_history.clear()
    
    def get_regime_signal(self) -> str:
        """
        Generate regime signal based on correlation structure.
        High BTC-SPX correlation suggests risk-on/off synchronization.
        Negative BTC-DXY correlation suggests dollar sensitivity.
        """
        with self._lock:
            btc_spx = self._corr_matrix[self.BTC, self.SPX]
            btc_dxy = self._corr_matrix[self.BTC, self.DXY]
            
            # Regime classification
            if btc_spx > 0.5 and btc_dxy < -0.3:
                return "RISK_ON_SYNC"  # BTC moves with stocks, inverse to dollar
            elif btc_spx > 0.5 and btc_dxy > 0.3:
                return "RISK_OFF_ANOMALY"  # Unusual positive correlation with dollar
            elif btc_spx < -0.3:
                return "DECORRELATION"  # BTC diverging from traditional assets
            else:
                return "NEUTRAL"
    
    def to_dict(self) -> dict:
        """Export state for serialization."""
        with self._lock:
            return {
                "correlation_matrix": self._corr_matrix.tolist(),
                "means": self._means.tolist(),
                "variances": self._variances.tolist(),
                "update_count": self._update_count,
                "is_warmed_up": self._is_warmed_up,
                "alpha": self.alpha,
                "halflife": self.halflife,
                "asset_names": self.ASSET_NAMES
            }


# Global singleton instance
_corr_instance: Optional[RollingCorrelationMatrix] = None
_instance_lock = threading.Lock()


def get_correlation_matrix() -> RollingCorrelationMatrix:
    """Get or create the global correlation matrix instance."""
    global _corr_instance
    if _corr_instance is None:
        with _instance_lock:
            if _corr_instance is None:
                _corr_instance = RollingCorrelationMatrix()
    return _corr_instance


if __name__ == "__main__":
    # Test the correlation matrix
    print("Testing RollingCorrelationMatrix:")
    
    corr = RollingCorrelationMatrix(halflife=50, warmup_periods=20)
    
    # Simulate some tick data
    np.random.seed(42)
    n_ticks = 100
    
    for i in range(n_ticks):
        # Generate correlated random returns
        base = np.random.randn()
        values = np.array([
            base * 0.02 + np.random.randn() * 0.01,  # BTC (volatile)
            -base * 0.01 + np.random.randn() * 0.005,  # DXY (inverse to risk)
            base * 0.005 + np.random.randn() * 0.002,  # Yields
            base * 0.015 + np.random.randn() * 0.008  # SPX
        ])
        
        corr.update_tick(values)
        
        if (i + 1) % 25 == 0:
            stats = corr.get_stats()
            print(f"\nTick {i + 1}:")
            print(f"  BTC-SPX: {stats.btc_spx:.4f}")
            print(f"  BTC-DXY: {stats.btc_dxy:.4f}")
            print(f"  Regime: {corr.get_regime_signal()}")
    
    print(f"\nFinal Stats: {corr.get_stats().to_dict()}")
    print(f"Warmed up: {corr.is_warmed_up()}")
    print(f"Total updates: {corr.get_update_count()}")
