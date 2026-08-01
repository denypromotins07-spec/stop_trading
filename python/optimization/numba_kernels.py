"""
Chapter 5: Extreme Python Optimization (Numba/Cython & GIL Bypass)
numba_kernels.py - Heavily optimized @njit Numba kernels for core mathematical loops
"""

import numpy as np
from numba import njit, prange, float64, int64
from numba.types import Tuple
from typing import Tuple as PyTuple


# ============================================================================
# Basic Statistics Kernels
# ============================================================================

@njit(cache=True, nogil=True, fastmath=True)
def rolling_mean_kernel(data: np.ndarray, window: int) -> np.ndarray:
    """
    Compute rolling mean with O(n) complexity using incremental updates.
    
    Args:
        data: Input array
        window: Rolling window size
    
    Returns:
        Array of rolling means
    """
    n = len(data)
    result = np.empty(n, dtype=np.float64)
    
    if window <= 0 or window > n:
        return result
    
    # Initialize first window
    window_sum = 0.0
    for i in range(window):
        window_sum += data[i]
    
    # First valid value
    for i in range(window - 1):
        result[i] = np.nan
    
    result[window - 1] = window_sum / window
    
    # Incremental updates
    for i in range(window, n):
        window_sum += data[i] - data[i - window]
        result[i] = window_sum / window
    
    return result


@njit(cache=True, nogil=True, fastmath=True)
def rolling_std_kernel(data: np.ndarray, window: int) -> np.ndarray:
    """
    Compute rolling standard deviation using Welford's online algorithm.
    
    Args:
        data: Input array
        window: Rolling window size
    
    Returns:
        Array of rolling stds
    """
    n = len(data)
    result = np.empty(n, dtype=np.float64)
    
    if window <= 1 or window > n:
        return result
    
    for i in range(window - 1):
        result[i] = np.nan
    
    # First window using two-pass
    mean = 0.0
    for i in range(window):
        mean += data[i]
    mean /= window
    
    m2 = 0.0
    for i in range(window):
        diff = data[i] - mean
        m2 += diff * diff
    
    result[window - 1] = np.sqrt(m2 / window)
    
    # Incremental updates using Welford
    old_mean = mean
    for i in range(window, n):
        # Add new value
        new_mean = old_mean + (data[i] - old_mean) / window
        
        # Remove old value
        prev_mean = new_mean + (old_mean - new_mean) / (window - 1)
        
        # Update M2
        m2 = m2 + (data[i] - new_mean) * (data[i] - prev_mean) - \
             (data[i - window] - old_mean) * (data[i - window] - prev_mean)
        
        result[i] = np.sqrt(max(0.0, m2 / window))
        old_mean = new_mean
    
    return result


@njit(cache=True, nogil=True, fastmath=True)
def zscore_kernel(data: np.ndarray, window: int) -> np.ndarray:
    """
    Compute rolling Z-scores.
    
    Args:
        data: Input array
        window: Lookback window
    
    Returns:
        Array of Z-scores
    """
    n = len(data)
    result = np.empty(n, dtype=np.float64)
    
    if window <= 1:
        return result
    
    means = rolling_mean_kernel(data, window)
    stds = rolling_std_kernel(data, window)
    
    for i in range(n):
        if stds[i] > 1e-10:
            result[i] = (data[i] - means[i]) / stds[i]
        else:
            result[i] = 0.0
    
    return result


# ============================================================================
# Covariance and Correlation Kernels
# ============================================================================

@njit(cache=True, nogil=True, fastmath=True)
def pairwise_covariance(
    returns_a: np.ndarray,
    returns_b: np.ndarray
) -> float:
    """
    Calculate covariance between two return series.
    
    Args:
        returns_a: First return series
        returns_b: Second return series
    
    Returns:
        Covariance value
    """
    n = min(len(returns_a), len(returns_b))
    if n < 2:
        return 0.0
    
    # Calculate means
    mean_a = 0.0
    mean_b = 0.0
    for i in range(n):
        mean_a += returns_a[i]
        mean_b += returns_b[i]
    mean_a /= n
    mean_b /= n
    
    # Calculate covariance
    cov = 0.0
    for i in range(n):
        cov += (returns_a[i] - mean_a) * (returns_b[i] - mean_b)
    
    return cov / (n - 1)


@njit(cache=True, nogil=True, fastmath=True)
def pairwise_correlation(
    returns_a: np.ndarray,
    returns_b: np.ndarray
) -> float:
    """
    Calculate Pearson correlation between two return series.
    
    Returns:
        Correlation coefficient [-1, 1]
    """
    n = min(len(returns_a), len(returns_b))
    if n < 2:
        return 0.0
    
    # Calculate means
    mean_a = 0.0
    mean_b = 0.0
    for i in range(n):
        mean_a += returns_a[i]
        mean_b += returns_b[i]
    mean_a /= n
    mean_b /= n
    
    # Calculate correlation components
    sum_xy = 0.0
    sum_xx = 0.0
    sum_yy = 0.0
    
    for i in range(n):
        dx = returns_a[i] - mean_a
        dy = returns_b[i] - mean_b
        sum_xy += dx * dy
        sum_xx += dx * dx
        sum_yy += dy * dy
    
    denom = np.sqrt(sum_xx * sum_yy)
    if denom < 1e-10:
        return 0.0
    
    return sum_xy / denom


@njit(cache=True, nogil=True, parallel=True, fastmath=True)
def covariance_matrix_kernel(
    returns: np.ndarray,  # Shape: (n_observations, n_assets)
) -> np.ndarray:
    """
    Compute full covariance matrix for multiple assets.
    
    Args:
        returns: Return matrix (time x assets)
    
    Returns:
        Covariance matrix (assets x assets)
    """
    n_obs, n_assets = returns.shape
    
    if n_obs < 2:
        return np.zeros((n_assets, n_assets), dtype=np.float64)
    
    # Calculate means for each asset
    means = np.empty(n_assets, dtype=np.float64)
    for j in range(n_assets):
        m = 0.0
        for i in range(n_obs):
            m += returns[i, j]
        means[j] = m / n_obs
    
    # Center the data
    centered = np.empty((n_obs, n_assets), dtype=np.float64)
    for i in range(n_obs):
        for j in range(n_assets):
            centered[i, j] = returns[i, j] - means[j]
    
    # Compute covariance matrix: X'X / (n-1)
    cov_matrix = np.zeros((n_assets, n_assets), dtype=np.float64)
    
    for i in prange(n_assets):
        for j in range(i, n_assets):
            cov_ij = 0.0
            for k in range(n_obs):
                cov_ij += centered[k, i] * centered[k, j]
            cov_ij /= (n_obs - 1)
            cov_matrix[i, j] = cov_ij
            cov_matrix[j, i] = cov_ij
    
    return cov_matrix


# ============================================================================
# Exponential Moving Average Kernels
# ============================================================================

@njit(cache=True, nogil=True, fastmath=True)
def ewma_kernel(
    data: np.ndarray,
    alpha: float
) -> np.ndarray:
    """
    Compute exponential weighted moving average.
    
    Args:
        data: Input array
        alpha: Smoothing factor (0 < alpha <= 1)
    
    Returns:
        EWMA values
    """
    n = len(data)
    result = np.empty(n, dtype=np.float64)
    
    if n == 0:
        return result
    
    result[0] = data[0]
    
    for i in range(1, n):
        result[i] = alpha * data[i] + (1.0 - alpha) * result[i - 1]
    
    return result


@njit(cache=True, nogil=True, fastmath=True)
def ewmvar_kernel(
    data: np.ndarray,
    alpha: float
) -> np.ndarray:
    """
    Compute exponential weighted moving variance.
    
    Uses the recurrence relation:
    Var_t = (1-alpha) * (Var_{t-1} + alpha * (x_t - mean_{t-1})^2)
    
    Args:
        data: Input array
        alpha: Smoothing factor
    
    Returns:
        EWM variance values
    """
    n = len(data)
    result = np.empty(n, dtype=np.float64)
    
    if n == 0:
        return result
    
    mean = data[0]
    var = 0.0
    result[0] = 0.0
    
    for i in range(1, n):
        diff = data[i] - mean
        mean = alpha * data[i] + (1.0 - alpha) * mean
        var = (1.0 - alpha) * (var + alpha * diff * diff)
        result[i] = var
    
    return result


# ============================================================================
# Order Book Kernels
# ============================================================================

@njit(cache=True, nogil=True, fastmath=True)
def vwap_kernel(
    prices: np.ndarray,
    volumes: np.ndarray
) -> np.ndarray:
    """
    Compute cumulative VWAP over time.
    
    Args:
        prices: Price series
        volumes: Volume series
    
    Returns:
        Cumulative VWAP at each point
    """
    n = len(prices)
    result = np.empty(n, dtype=np.float64)
    
    cum_pv = 0.0  # Cumulative price * volume
    cum_v = 0.0   # Cumulative volume
    
    for i in range(n):
        cum_pv += prices[i] * volumes[i]
        cum_v += volumes[i]
        
        if cum_v > 0:
            result[i] = cum_pv / cum_v
        else:
            result[i] = prices[i]
    
    return result


@njit(cache=True, nogil=True, fastmath=True)
def order_book_imbalance_kernel(
    bid_sizes: np.ndarray,
    ask_sizes: np.ndarray,
    n_levels: int
) -> np.ndarray:
    """
    Calculate order book imbalance at multiple levels.
    
    Imbalance = (Bid - Ask) / (Bid + Ask)
    
    Args:
        bid_sizes: Bid sizes at each level (flattened: [level1_bid, level2_bid, ...])
        ask_sizes: Ask sizes at each level
        n_levels: Number of levels per side
    
    Returns:
        Imbalance at each level
    """
    result = np.empty(n_levels, dtype=np.float64)
    
    for i in range(n_levels):
        bid = bid_sizes[i] if i < len(bid_sizes) else 0.0
        ask = ask_sizes[i] if i < len(ask_sizes) else 0.0
        
        total = bid + ask
        if total > 0:
            result[i] = (bid - ask) / total
        else:
            result[i] = 0.0
    
    return result


# ============================================================================
# Risk Metrics Kernels
# ============================================================================

@njit(cache=True, nogil=True, fastmath=True)
def var_parametric_kernel(
    returns: np.ndarray,
    confidence: float = 0.99
) -> float:
    """
    Calculate parametric Value at Risk.
    
    Assumes normal distribution.
    
    Args:
        returns: Historical returns
        confidence: Confidence level (e.g., 0.99 for 99%)
    
    Returns:
        VaR as positive number (loss)
    """
    n = len(returns)
    if n < 2:
        return 0.0
    
    # Calculate mean and std
    mean = 0.0
    for r in returns:
        mean += r
    mean /= n
    
    var_sum = 0.0
    for r in returns:
        diff = r - mean
        var_sum += diff * diff
    std = np.sqrt(var_sum / (n - 1))
    
    # Z-score for confidence level
    # Approximate inverse normal CDF
    if confidence >= 0.99:
        z = 2.326
    elif confidence >= 0.975:
        z = 1.96
    elif confidence >= 0.95:
        z = 1.645
    else:
        z = 1.282
    
    var = -(mean - z * std)
    
    return max(0.0, var)


@njit(cache=True, nogil=True, fastmath=True)
def historical_var_kernel(
    returns: np.ndarray,
    confidence: float = 0.99
) -> float:
    """
    Calculate historical Value at Risk using sorted returns.
    
    Args:
        returns: Historical returns
        confidence: Confidence level
    
    Returns:
        Historical VaR
    """
    n = len(returns)
    if n < 10:
        return 0.0
    
    # Create sorted copy
    sorted_returns = np.sort(returns)
    
    # Find percentile index
    idx = int((1.0 - confidence) * n)
    idx = max(0, min(idx, n - 1))
    
    return -sorted_returns[idx]


@njit(cache=True, nogil=True, fastmath=True)
def expected_shortfall_kernel(
    returns: np.ndarray,
    confidence: float = 0.99
) -> float:
    """
    Calculate Expected Shortfall (CVaR).
    
    Average loss beyond VaR threshold.
    
    Args:
        returns: Historical returns
        confidence: Confidence level
    
    Returns:
        Expected Shortfall
    """
    n = len(returns)
    if n < 10:
        return 0.0
    
    # Get historical VaR threshold
    var_threshold = historical_var_kernel(returns, confidence)
    
    # Average all losses beyond threshold
    tail_sum = 0.0
    tail_count = 0
    
    for r in returns:
        loss = -r
        if loss > var_threshold:
            tail_sum += loss
            tail_count += 1
    
    if tail_count == 0:
        return var_threshold
    
    return tail_sum / tail_count


# ============================================================================
# Matrix Operations (GIL-free)
# ============================================================================

@njit(cache=True, nogil=True, parallel=True, fastmath=True)
def matrix_multiply_kernel(
    A: np.ndarray,
    B: np.ndarray
) -> np.ndarray:
    """
    Matrix multiplication with parallel execution.
    
    Args:
        A: Left matrix (m x k)
        B: Right matrix (k x n)
    
    Returns:
        Result matrix (m x n)
    """
    m, k = A.shape
    k2, n = B.shape
    
    if k != k2:
        raise ValueError("Matrix dimensions don't match")
    
    C = np.zeros((m, n), dtype=np.float64)
    
    for i in prange(m):
        for j in range(n):
            s = 0.0
            for l in range(k):
                s += A[i, l] * B[l, j]
            C[i, j] = s
    
    return C


@njit(cache=True, nogil=True, fastmath=True)
def cholesky_decomposition(A: np.ndarray) -> np.ndarray:
    """
    Cholesky decomposition for positive definite matrices.
    A = L @ L.T
    
    Args:
        A: Input matrix (must be positive definite)
    
    Returns:
        Lower triangular matrix L
    """
    n = A.shape[0]
    L = np.zeros((n, n), dtype=np.float64)
    
    for i in range(n):
        for j in range(i + 1):
            s = 0.0
            for k in range(j):
                s += L[i, k] * L[j, k]
            
            if i == j:
                val = A[i, i] - s
                if val > 0:
                    L[i, j] = np.sqrt(val)
                else:
                    L[i, j] = 0.0
            else:
                if L[j, j] > 1e-10:
                    L[i, j] = (A[i, j] - s) / L[j, j]
    
    return L


# ============================================================================
# Kernel Manager Class
# ============================================================================

class NumbaKernelManager:
    """
    Manages Numba kernel compilation cache and provides unified interface.
    Enforces strict memory views to prevent numpy array copying.
    """
    
    def __init__(self):
        self._compiled_kernels = {}
        self._cache_enabled = True
    
    def get_zscore(self, data: np.ndarray, window: int) -> np.ndarray:
        """Get Z-scores using compiled kernel."""
        # Ensure contiguous memory layout
        data_contig = np.ascontiguousarray(data, dtype=np.float64)
        return zscore_kernel(data_contig, window)
    
    def get_ewma(self, data: np.ndarray, alpha: float) -> np.ndarray:
        """Get EWMA using compiled kernel."""
        data_contig = np.ascontiguousarray(data, dtype=np.float64)
        return ewma_kernel(data_contig, alpha)
    
    def get_covariance_matrix(self, returns: np.ndarray) -> np.ndarray:
        """Get covariance matrix using parallel kernel."""
        returns_contig = np.ascontiguousarray(returns, dtype=np.float64)
        return covariance_matrix_kernel(returns_contig)
    
    def clear_cache(self):
        """Clear Numba compilation cache."""
        from numba import cuda
        if cuda.is_available():
            cuda.close()
        self._compiled_kernels.clear()


# Module-level convenience functions
def create_kernel_manager() -> NumbaKernelManager:
    """Factory function to create kernel manager."""
    return NumbaKernelManager()


@njit(cache=True, nogil=True)
def quick_zscore(data: np.ndarray, window: int) -> np.ndarray:
    """Quick Z-score calculation."""
    return zscore_kernel(data, window)


@njit(cache=True, nogil=True)
def quick_ewma(data: np.ndarray, alpha: float) -> np.ndarray:
    """Quick EWMA calculation."""
    return ewma_kernel(data, alpha)
