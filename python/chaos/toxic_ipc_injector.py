"""
Fault injection engine that feeds malformed, NaN, and extreme-value feature vectors.
Validates that isolation forests and data engines gracefully quarantine toxic data.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import logging
import time
from enum import Enum
import random

logger = logging.getLogger(__name__)


class ToxicityType(Enum):
    """Types of toxic data to inject."""
    NAN_VALUES = "nan"
    INF_VALUES = "inf"
    EXTREME_VALUES = "extreme"
    NEGATIVE_VALUES = "negative"
    ZERO_DIVISION = "zero_div"
    MALFORMED_SHAPE = "malformed_shape"
    OUT_OF_RANGE = "out_of_range"
    DUPLICATE_ROWS = "duplicate_rows"


@dataclass
class ToxicityConfig:
    """Configuration for toxicity injection."""
    nan_probability: float = 0.01
    inf_probability: float = 0.005
    extreme_multiplier: float = 1e6
    negative_probability: float = 0.01
    out_of_range_min: float = -1e10
    out_of_range_max: float = 1e10
    duplicate_rate: float = 0.01
    
    # Target columns/features
    target_features: Optional[List[str]] = None
    global_toxicity_rate: float = 0.02


@dataclass
class InjectionResult:
    """Result of a toxicity injection."""
    original_shape: Tuple[int, ...]
    modified_shape: Tuple[int, ...]
    toxicity_type: str
    num_toxic_elements: int
    toxicity_locations: List[Tuple[int, int]]
    timestamp_ns: int = field(default_factory=lambda: time.time_ns())


class ToxicIPCMInjector:
    """
    Injects toxic data into IPC streams for resilience testing.
    """
    
    def __init__(self, config: Optional[ToxicityConfig] = None):
        self.config = config or ToxicityConfig()
        self._injection_count = 0
        self._detection_count = 0
    
    def inject_nan(
        self,
        data: np.ndarray,
        probability: Optional[float] = None
    ) -> Tuple[np.ndarray, InjectionResult]:
        """Inject NaN values into data."""
        prob = probability or self.config.nan_probability
        mask = np.random.random(data.shape) < prob
        
        modified = data.copy().astype(float)
        modified[mask] = np.nan
        
        locations = list(zip(*np.where(mask)))
        
        result = InjectionResult(
            original_shape=data.shape,
            modified_shape=modified.shape,
            toxicity_type="nan",
            num_toxic_elements=np.sum(mask),
            toxicity_locations=locations
        )
        
        self._injection_count += 1
        logger.debug(f"Injected {result.num_toxic_elements} NaN values")
        
        return modified, result
    
    def inject_inf(
        self,
        data: np.ndarray,
        probability: Optional[float] = None
    ) -> Tuple[np.ndarray, InjectionResult]:
        """Inject infinite values into data."""
        prob = probability or self.config.inf_probability
        mask = np.random.random(data.shape) < prob
        
        modified = data.copy().astype(float)
        
        # Randomly choose +inf or -inf
        signs = np.random.choice([-1, 1], size=np.sum(mask))
        modified[mask] = signs * np.inf
        
        locations = list(zip(*np.where(mask)))
        
        result = InjectionResult(
            original_shape=data.shape,
            modified_shape=modified.shape,
            toxicity_type="inf",
            num_toxic_elements=np.sum(mask),
            toxicity_locations=locations
        )
        
        self._injection_count += 1
        return modified, result
    
    def inject_extreme_values(
        self,
        data: np.ndarray,
        multiplier: Optional[float] = None
    ) -> Tuple[np.ndarray, InjectionResult]:
        """Inject extreme values (very large magnitude)."""
        mult = multiplier or self.config.extreme_multiplier
        prob = self.config.nan_probability
        
        mask = np.random.random(data.shape) < prob
        modified = data.copy().astype(float)
        
        # Multiply selected values by extreme factor
        signs = np.random.choice([-1, 1], size=data.shape)
        modified[mask] = data[mask] * mult * signs[mask]
        
        locations = list(zip(*np.where(mask)))
        
        result = InjectionResult(
            original_shape=data.shape,
            modified_shape=modified.shape,
            toxicity_type="extreme",
            num_toxic_elements=np.sum(mask),
            toxicity_locations=locations
        )
        
        self._injection_count += 1
        return modified, result
    
    def inject_out_of_range(
        self,
        data: np.ndarray,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None
    ) -> Tuple[np.ndarray, InjectionResult]:
        """Inject values outside expected range."""
        min_v = min_val if min_val is not None else self.config.out_of_range_min
        max_v = max_val if max_val is not None else self.config.out_of_range_max
        
        prob = self.config.nan_probability
        mask = np.random.random(data.shape) < prob
        
        modified = data.copy().astype(float)
        
        # Generate out-of-range values
        out_of_range_vals = np.where(
            np.random.random(np.sum(mask)) < 0.5,
            np.random.uniform(min_v, min_v * 0.1, size=np.sum(mask)),
            np.random.uniform(max_v * 0.1, max_v, size=np.sum(mask))
        )
        modified[mask] = out_of_range_vals
        
        locations = list(zip(*np.where(mask)))
        
        result = InjectionResult(
            original_shape=data.shape,
            modified_shape=modified.shape,
            toxicity_type="out_of_range",
            num_toxic_elements=np.sum(mask),
            toxicity_locations=locations
        )
        
        self._injection_count += 1
        return modified, result
    
    def inject_duplicates(
        self,
        data: np.ndarray,
        rate: Optional[float] = None
    ) -> Tuple[np.ndarray, InjectionResult]:
        """Inject duplicate rows."""
        rate = rate or self.config.duplicate_rate
        
        n_rows = data.shape[0]
        n_duplicates = max(1, int(n_rows * rate))
        
        # Select random rows to duplicate
        dup_indices = np.random.choice(n_rows, size=n_duplicates)
        dup_rows = data[dup_indices]
        
        modified = np.vstack([data, dup_rows])
        
        result = InjectionResult(
            original_shape=data.shape,
            modified_shape=modified.shape,
            toxicity_type="duplicate_rows",
            num_toxic_elements=n_duplicates,
            toxicity_locations=[(n_rows + i, 0) for i in range(n_duplicates)]
        )
        
        self._injection_count += 1
        return modified, result
    
    def inject_random_toxicity(
        self,
        data: np.ndarray
    ) -> Tuple[np.ndarray, InjectionResult]:
        """Apply a random toxicity type."""
        toxicity_funcs = [
            self.inject_nan,
            self.inject_inf,
            self.inject_extreme_values,
            self.inject_out_of_range,
        ]
        
        chosen_func = random.choice(toxicity_funcs)
        return chosen_func(data)
    
    def generate_toxic_batch(
        self,
        shape: Tuple[int, int],
        toxicity_rate: Optional[float] = None
    ) -> Tuple[np.ndarray, List[InjectionResult]]:
        """Generate a batch with multiple types of toxicity."""
        rate = toxicity_rate or self.config.global_toxicity_rate
        
        # Start with clean data
        data = np.random.randn(*shape).astype(np.float32)
        
        results = []
        
        # Apply multiple toxicity types
        if rate > 0:
            data, nan_result = self.inject_nan(data, rate / 4)
            results.append(nan_result)
            
            data, inf_result = self.inject_inf(data, rate / 4)
            results.append(inf_result)
            
            data, extreme_result = self.inject_extreme_values(data)
            results.append(extreme_result)
        
        return data, results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get injection statistics."""
        return {
            'total_injections': self._injection_count,
            'config': {
                'nan_prob': self.config.nan_probability,
                'inf_prob': self.config.inf_probability,
                'extreme_mult': self.config.extreme_multiplier
            }
        }


class ToxicDataQuarantine:
    """
    Detects and quarantines toxic data before it reaches ML models.
    """
    
    def __init__(self):
        self._quarantined_count = 0
        self._quarantine_buffer: List[np.ndarray] = []
    
    def detect_and_quarantine(
        self,
        data: np.ndarray,
        threshold_nan_rate: float = 0.1,
        threshold_inf_rate: float = 0.01
    ) -> Tuple[np.ndarray, bool]:
        """
        Detect toxic data and return clean version.
        
        Returns:
            Tuple of (clean_data, was_toxic)
        """
        was_toxic = False
        
        # Check for NaN
        nan_mask = np.isnan(data)
        nan_rate = np.mean(nan_mask)
        
        if nan_rate > threshold_nan_rate:
            was_toxic = True
            logger.warning(f"High NaN rate detected: {nan_rate:.2%}")
        
        # Replace NaN with zeros (or could use interpolation)
        clean_data = np.nan_to_num(data, nan=0.0, posinf=1e10, neginf=-1e10)
        
        # Check for Inf
        inf_mask = np.isinf(clean_data)
        inf_rate = np.mean(inf_mask)
        
        if inf_rate > threshold_inf_rate:
            was_toxic = True
            logger.warning(f"High Inf rate detected: {inf_rate:.2%}")
            clean_data[inf_mask] = 0.0
        
        # Check for extreme values
        std = np.std(clean_data)
        if std > 1e6:
            was_toxic = True
            logger.warning(f"Extreme variance detected: {std:.2e}")
            # Clip extreme values
            clean_data = np.clip(clean_data, -1e6, 1e6)
        
        if was_toxic:
            self._quarantined_count += 1
            self._quarantine_buffer.append(data.copy())
            
            # Keep buffer bounded
            if len(self._quarantine_buffer) > 100:
                self._quarantine_buffer.pop(0)
        
        return clean_data, was_toxic
    
    def get_quarantine_stats(self) -> Dict[str, Any]:
        """Get quarantine statistics."""
        return {
            'quarantined_batches': self._quarantined_count,
            'buffer_size': len(self._quarantine_buffer)
        }


def create_toxic_injector(config: Optional[ToxicityConfig] = None) -> ToxicIPCMInjector:
    """Factory function to create injector."""
    return ToxicIPCMInjector(config)
