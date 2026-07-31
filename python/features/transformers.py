# Scikit-Learn Compatible Stateless Transformers for HFT Features
# Operates on pre-allocated numpy arrays to prevent memory fragmentation

from __future__ import annotations
import logging
import numpy as np
from typing import Dict, Any, Optional, Tuple

log = logging.getLogger(__name__)


class ZScoreTransformer:
    """
    Z-score normalization transformer.
    Uses Welford's online algorithm for numerically stable mean/variance computation.
    Operates in-place on pre-allocated arrays.
    """

    def __init__(
        self,
        epsilon: float = 1e-8,
        decay: float = 0.999,  # EMA decay for online updates
    ) -> None:
        self.epsilon = epsilon
        self.decay = decay
        self._running_mean: Optional[np.ndarray] = None
        self._running_var: Optional[np.ndarray] = None
        self._n_samples = 0

    def fit(self, data: Dict[str, np.ndarray]) -> "ZScoreTransformer":
        """Initialize running statistics from initial batch."""
        first_arr = next(iter(data.values()))
        n_features = first_arr.shape[-1] if first_arr.ndim > 1 else first_arr.shape[0]
        
        self._running_mean = np.zeros(n_features, dtype=np.float64)
        self._running_var = np.ones(n_features, dtype=np.float64)
        self._n_samples = 0
        
        # Update with initial data
        self.partial_fit(data)
        log.info(f"ZScoreTransformer fitted with {self._n_samples} samples")
        return self

    def partial_fit(self, data: Dict[str, np.ndarray]) -> "ZScoreTransformer":
        """Update running statistics using Welford's online algorithm."""
        for arr in data.values():
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            
            batch_size = arr.shape[0]
            
            if self._running_mean is None:
                self._running_mean = np.mean(arr, axis=0)
                self._running_var = np.var(arr, axis=0) + self.epsilon
            else:
                # Online update with EMA
                batch_mean = np.mean(arr, axis=0)
                batch_var = np.var(arr, axis=0)
                
                self._running_mean = (
                    self.decay * self._running_mean + 
                    (1 - self.decay) * batch_mean
                )
                self._running_var = (
                    self.decay * self._running_var + 
                    (1 - self.decay) * (batch_var + self.decay * (batch_mean - self._running_mean)**2)
                )
            
            self._n_samples += batch_size
        
        return self

    def transform(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply z-score normalization in-place."""
        if self._running_mean is None:
            raise RuntimeError("Transformer not fitted. Call fit() first.")
        
        result = {}
        for key, arr in data.items():
            # Create view to avoid copy where possible
            normalized = (arr - self._running_mean) / np.sqrt(self._running_var + self.epsilon)
            result[key] = np.ascontiguousarray(normalized, dtype=np.float64)
        
        return result

    def fit_transform(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Fit and transform in one pass."""
        self.fit(data)
        return self.transform(data)


class FractionalDifferencingTransformer:
    """
    Fractional differencing for stationary feature generation.
    Uses matrix multiplication approach for efficiency.
    Pre-computes binomial weights to avoid recalculation.
    """

    def __init__(
        self,
        d: float = 0.5,  # Differencing order
        window: int = 100,  # Lookback window
    ) -> None:
        self.d = d
        self.window = window
        self._weights: Optional[np.ndarray] = None
        self._pre_allocated_output: Optional[np.ndarray] = None

    def _compute_weights(self) -> np.ndarray:
        """Pre-compute fractional differencing weights using gamma function."""
        weights = np.zeros(self.window, dtype=np.float64)
        weights[0] = 1.0
        
        for k in range(1, self.window):
            # Recursive formula for binomial coefficients
            weights[k] = weights[k-1] * (k - 1 - self.d) / k
        
        return weights

    def fit(self, data: Dict[str, np.ndarray]) -> "FractionalDifferencingTransformer":
        """Pre-compute weights and allocate output buffer."""
        self._weights = self._compute_weights()
        
        # Pre-allocate output buffer based on expected input size
        first_arr = next(iter(data.values()))
        self._pre_allocated_output = np.empty_like(first_arr, dtype=np.float64)
        
        log.info(f"FractionalDifferencingTransformer fitted with d={self.d}, window={self.window}")
        return self

    def transform(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply fractional differencing using convolution."""
        if self._weights is None:
            raise RuntimeError("Transformer not fitted. Call fit() first.")
        
        result = {}
        for key, arr in data.items():
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            
            n_samples, n_features = arr.shape
            output = np.empty((n_samples, n_features), dtype=np.float64)
            
            # Apply fractional differencing per feature
            for f in range(n_features):
                series = arr[:, f]
                diff_result = np.zeros(n_samples, dtype=np.float64)
                
                for i in range(min(self.window, n_samples), n_samples):
                    # Convolve with pre-computed weights
                    diff_result[i] = np.sum(series[i-self.window+1:i+1] * self._weights[::-1])
                
                output[:, f] = diff_result
            
            result[key] = output
        
        return result


class ClipTransformer:
    """
    Winsorization/clipping transformer for outlier handling.
    Clips values to specified percentiles or absolute bounds.
    """

    def __init__(
        self,
        lower_percentile: float = 0.5,
        upper_percentile: float = 99.5,
        lower_bound: Optional[float] = None,
        upper_bound: Optional[float] = None,
    ) -> None:
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self._computed_bounds: Dict[str, Tuple[float, float]] = {}

    def fit(self, data: Dict[str, np.ndarray]) -> "ClipTransformer":
        """Compute clipping bounds from data percentiles."""
        for key, arr in data.items():
            flat_arr = arr.ravel()
            
            if self.lower_bound is not None and self.upper_bound is not None:
                self._computed_bounds[key] = (self.lower_bound, self.upper_bound)
            else:
                lower = np.percentile(flat_arr, self.lower_percentile)
                upper = np.percentile(flat_arr, self.upper_percentile)
                self._computed_bounds[key] = (lower, upper)
        
        log.info(f"ClipTransformer fitted with bounds: {self._computed_bounds}")
        return self

    def transform(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply clipping to data."""
        if not self._computed_bounds:
            raise RuntimeError("Transformer not fitted. Call fit() first.")
        
        result = {}
        for key, arr in data.items():
            lower, upper = self._computed_bounds[key]
            clipped = np.clip(arr, lower, upper)
            result[key] = np.ascontiguousarray(clipped, dtype=np.float64)
        
        return result

    def fit_transform(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Fit and transform in one pass."""
        self.fit(data)
        return self.transform(data)


class FeatureUnionTransformer:
    """
    Combines multiple transformers into a single pipeline.
    Applies transformers sequentially and merges outputs.
    """

    def __init__(self, transformers: list[tuple[str, Any]]) -> None:
        """
        Args:
            transformers: List of (name, transformer) tuples
        """
        self.transformers = transformers

    def fit(self, data: Dict[str, np.ndarray]) -> "FeatureUnionTransformer":
        """Fit all transformers."""
        current_data = data
        for name, transformer in self.transformers:
            if hasattr(transformer, 'fit'):
                transformer.fit(current_data)
            if hasattr(transformer, 'transform'):
                current_data = transformer.transform(current_data)
        
        log.info(f"FeatureUnionTransformer fitted with {len(self.transformers)} transformers")
        return self

    def transform(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Transform through all transformers."""
        current_data = data
        for name, transformer in self.transformers:
            if hasattr(transformer, 'transform'):
                current_data = transformer.transform(current_data)
        
        return current_data

    def fit_transform(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Fit and transform in one pass."""
        self.fit(data)
        return self.transform(data)


def create_standard_pipeline() -> FeatureUnionTransformer:
    """Create a standard feature transformation pipeline."""
    return FeatureUnionTransformer([
        ("zscore", ZScoreTransformer(decay=0.999)),
        ("clip", ClipTransformer(lower_percentile=0.5, upper_percentile=99.5)),
        ("frac_diff", FractionalDifferencingTransformer(d=0.5, window=100)),
    ])
