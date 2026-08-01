"""
Chapter 1: Advanced Order Flow & Footprint Analytics
delta_divergence.py - Detect Delta Divergence signaling institutional exhaustion
"""

import numpy as np
from numba import njit, prange
from typing import Tuple, Optional, List
from dataclasses import dataclass
from enum import IntEnum


class DivergenceType(IntEnum):
    """Types of delta divergence patterns"""
    NONE = 0
    BULLISH = 1      # Price lower low, CVD higher low (buying absorption)
    BEARISH = 2      # Price higher high, CVD lower high (selling absorption)
    HIDDEN_BULLISH = 3
    HIDDEN_BEARISH = 4


@dataclass
class DivergenceSignal:
    """Represents a detected divergence signal"""
    timestamp: int
    divergence_type: DivergenceType
    price_level: float
    cvd_level: float
    strength: float  # 0.0 to 1.0
    lookback_bars: int
    confidence: float


@njit(cache=True, nogil=True)
def calculate_cvd(
    deltas: np.ndarray,
    cumulative: bool = True
) -> np.ndarray:
    """
    Calculate Cumulative Volume Delta (CVD).
    
    Args:
        deltas: Array of net deltas (ask_volume - bid_volume) per bar
        cumulative: If True, return cumulative sum
    
    Returns:
        CVD array
    """
    n = len(deltas)
    cvd = np.empty(n, dtype=np.float64)
    
    if cumulative:
        running_sum = 0.0
        for i in range(n):
            running_sum += deltas[i]
            cvd[i] = running_sum
    else:
        for i in range(n):
            cvd[i] = deltas[i]
    
    return cvd


@njit(cache=True, nogil=True)
def find_peaks_troughs(
    values: np.ndarray,
    order: int = 5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Find peaks (local maxima) and troughs (local minima) in a series.
    
    Args:
        values: Input array
        order: Number of points on each side to consider for local extremum
    
    Returns:
        Tuple of (peak_indices, peak_values, trough_indices, trough_values)
    """
    n = len(values)
    max_peaks = n // 2
    
    peak_indices = np.empty(max_peaks, dtype=np.int64)
    peak_values = np.empty(max_peaks, dtype=np.float64)
    trough_indices = np.empty(max_peaks, dtype=np.int64)
    trough_values = np.empty(max_peaks, dtype=np.float64)
    
    peak_count = 0
    trough_count = 0
    
    for i in range(order, n - order):
        is_peak = True
        is_trough = True
        
        for j in range(1, order + 1):
            if values[i] <= values[i - j] or values[i] <= values[i + j]:
                is_peak = False
            if values[i] >= values[i - j] or values[i] >= values[i + j]:
                is_trough = False
        
        if is_peak:
            peak_indices[peak_count] = i
            peak_values[peak_count] = values[i]
            peak_count += 1
        
        if is_trough:
            trough_indices[trough_count] = i
            trough_values[trough_count] = values[i]
            trough_count += 1
    
    return (
        peak_indices[:peak_count],
        peak_values[:peak_count],
        trough_indices[:trough_count],
        trough_values[:trough_count]
    )


@njit(cache=True, nogil=True)
def detect_divergence(
    prices: np.ndarray,
    cvd: np.ndarray,
    lookback: int = 20,
    min_strength: float = 0.5
) -> np.ndarray:
    """
    Detect delta divergence between price and CVD.
    
    Bearish Divergence: Price makes higher high, CVD makes lower high
    Bullish Divergence: Price makes lower low, CVD makes higher low
    
    Args:
        prices: Price series (typically close or vwap)
        cvd: Cumulative Volume Delta series
        lookback: Bars to look back for divergence detection
        min_strength: Minimum divergence strength threshold
    
    Returns:
        Array of divergence signals encoded as:
        [timestamp_idx, type, strength, confidence]
    """
    n = len(prices)
    max_signals = n // 5
    signals = np.zeros((max_signals, 4), dtype=np.float64)
    signal_count = 0
    
    # Find recent peaks and troughs
    price_peaks, price_peak_vals, price_troughs, price_trough_vals = \
        find_peaks_troughs(prices, order=3)
    
    cvd_peaks, cvd_peak_vals, cvd_troughs, cvd_trough_vals = \
        find_peaks_troughs(cvd, order=3)
    
    if len(price_peaks) < 2 or len(cvd_peaks) < 2:
        return signals[:signal_count]
    
    # Check for bearish divergence at recent price peaks
    for i in range(1, len(price_peaks)):
        curr_peak_idx = price_peaks[i]
        prev_peak_idx = price_peaks[i - 1]
        
        if curr_peak_idx - prev_peak_idx > lookback:
            continue
        
        # Price higher high
        price_hh = prices[curr_peak_idx] > prices[prev_peak_idx]
        
        # Find corresponding CVD peaks
        cvd_at_curr = 0.0
        cvd_at_prev = 0.0
        
        for j in range(len(cvd_peaks)):
            if abs(cvd_peaks[j] - curr_peak_idx) <= 3:
                cvd_at_curr = cvd_peak_vals[j]
            if abs(cvd_peaks[j] - prev_peak_idx) <= 3:
                cvd_at_prev = cvd_peak_vals[j]
        
        # CVD lower high
        cvd_lh = cvd_at_curr < cvd_at_prev
        
        if price_hh and cvd_lh and cvd_at_prev != 0:
            # Calculate strength
            price_change = (prices[curr_peak_idx] - prices[prev_peak_idx]) / prices[prev_peak_idx]
            cvd_change = (cvd_at_curr - cvd_at_prev) / abs(cvd_at_prev)
            
            strength = abs(cvd_change) / (abs(price_change) + abs(cvd_change) + 1e-10)
            
            if strength >= min_strength:
                signals[signal_count, 0] = curr_peak_idx
                signals[signal_count, 1] = DivergenceType.BEARISH
                signals[signal_count, 2] = strength
                signals[signal_count, 3] = min(1.0, strength * 1.5)
                signal_count += 1
    
    # Check for bullish divergence at recent price troughs
    for i in range(1, len(price_troughs)):
        curr_trough_idx = price_troughs[i]
        prev_trough_idx = price_troughs[i - 1]
        
        if curr_trough_idx - prev_trough_idx > lookback:
            continue
        
        # Price lower low
        price_ll = prices[curr_trough_idx] < prices[prev_trough_idx]
        
        # Find corresponding CVD troughs
        cvd_at_curr = 0.0
        cvd_at_prev = 0.0
        
        for j in range(len(cvd_troughs)):
            if abs(cvd_troughs[j] - curr_trough_idx) <= 3:
                cvd_at_curr = cvd_trough_vals[j]
            if abs(cvd_troughs[j] - prev_trough_idx) <= 3:
                cvd_at_prev = cvd_trough_vals[j]
        
        # CVD higher low
        cvd_hl = cvd_at_curr > cvd_at_prev
        
        if price_ll and cvd_hl and cvd_at_prev != 0:
            # Calculate strength
            price_change = abs(prices[curr_trough_idx] - prices[prev_trough_idx]) / prices[prev_trough_idx]
            cvd_change = abs(cvd_at_curr - cvd_at_prev) / abs(cvd_at_prev)
            
            strength = abs(cvd_change) / (abs(price_change) + abs(cvd_change) + 1e-10)
            
            if strength >= min_strength:
                signals[signal_count, 0] = curr_trough_idx
                signals[signal_count, 1] = DivergenceType.BULLISH
                signals[signal_count, 2] = strength
                signals[signal_count, 3] = min(1.0, strength * 1.5)
                signal_count += 1
    
    return signals[:signal_count]


@njit(cache=True, nogil=True)
def calculate_trapped_trader_index(
    prices: np.ndarray,
    volumes: np.ndarray,
    deltas: np.ndarray,
    lookback: int = 10
) -> np.ndarray:
    """
    Calculate Trapped Trader Index (TTI).
    Measures volume that entered at wrong price levels.
    
    High TTI indicates trapped traders likely to exit, causing reversals.
    """
    n = len(prices)
    tti = np.empty(n, dtype=np.float64)
    
    for i in range(lookback, n):
        trapped_volume = 0.0
        total_volume = 0.0
        
        for j in range(i - lookback, i):
            total_volume += volumes[j]
            
            # Longs trapped: positive delta but price went down
            if deltas[j] > 0 and prices[j] > prices[i]:
                trapped_volume += abs(deltas[j])
            
            # Shorts trapped: negative delta but price went up
            if deltas[j] < 0 and prices[j] < prices[i]:
                trapped_volume += abs(deltas[j])
        
        if total_volume > 0:
            tti[i] = trapped_volume / total_volume
        else:
            tti[i] = 0.0
    
    # Fill initial values
    for i in range(lookback):
        tti[i] = 0.0
    
    return tti


@njit(cache=True, nogil=True)
def detect_exhaustion_bars(
    prices: np.ndarray,
    volumes: np.ndarray,
    deltas: np.ndarray,
    vol_threshold: float = 2.0,
    delta_ratio_threshold: float = 0.8
) -> np.ndarray:
    """
    Detect exhaustion bars - high volume with minimal price progress.
    Indicates potential reversal points.
    
    Returns:
        Boolean array indicating exhaustion bars
    """
    n = len(prices)
    exhaustion = np.zeros(n, dtype=np.bool_)
    
    # Calculate rolling volume average
    vol_sum = 0.0
    vol_count = 0
    vol_avg = np.empty(n, dtype=np.float64)
    
    for i in range(n):
        vol_sum += volumes[i]
        vol_count += 1
        if vol_count > 20:
            vol_sum -= volumes[i - 20]
            vol_count = 20
        vol_avg[i] = vol_sum / vol_count
    
    for i in range(1, n):
        # High volume condition
        if volumes[i] < vol_avg[i] * vol_threshold:
            continue
        
        # Small price range relative to volume
        price_range = abs(prices[i] - prices[i - 1])
        
        # Large opposing delta
        delta_abs = abs(deltas[i])
        total_vol = volumes[i]
        
        if total_vol > 0:
            delta_ratio = delta_abs / total_vol
            
            # If high delta ratio but small price move = exhaustion
            if delta_ratio > delta_ratio_threshold and price_range < (vol_avg[i] * 0.001):
                exhaustion[i] = True
    
    return exhaustion


class DeltaDivergenceEngine:
    """
    Engine for detecting delta divergence and institutional exhaustion patterns.
    """
    
    def __init__(
        self,
        lookback: int = 20,
        min_strength: float = 0.5
    ):
        self.lookback = lookback
        self.min_strength = min_strength
        self._cvd_history = None
    
    def analyze(
        self,
        prices: np.ndarray,
        volumes: np.ndarray,
        deltas: np.ndarray
    ) -> Dict:
        """
        Perform complete delta divergence analysis.
        
        Args:
            prices: Price series
            volumes: Volume series
            deltas: Net delta series (ask_vol - bid_vol)
        
        Returns:
            Dictionary containing all divergence metrics
        """
        # Calculate CVD
        cvd = calculate_cvd(deltas, cumulative=True)
        self._cvd_history = cvd
        
        # Detect divergences
        divergence_signals = detect_divergence(
            prices, cvd, self.lookback, self.min_strength
        )
        
        # Calculate trapped trader index
        tti = calculate_trapped_trader_index(
            prices, volumes, deltas, self.lookback
        )
        
        # Detect exhaustion bars
        exhaustion = detect_exhaustion_bars(
            prices, volumes, deltas
        )
        
        # Current CVD value
        current_cvd = cvd[-1] if len(cvd) > 0 else 0.0
        
        # CVD trend (simple slope)
        cvd_trend = 0.0
        if len(cvd) >= 5:
            cvd_trend = (cvd[-1] - cvd[-5]) / 5
        
        return {
            'cvd': cvd,
            'current_cvd': current_cvd,
            'cvd_trend': cvd_trend,
            'divergence_signals': divergence_signals,
            'trapped_trader_index': tti,
            'current_tti': tti[-1] if len(tti) > 0 else 0.0,
            'exhaustion_bars': exhaustion,
            'recent_exhaustion': exhaustion[-self.lookback:].sum() if len(exhaustion) >= self.lookback else 0
        }
    
    def get_signal_summary(
        self,
        analysis_result: Dict
    ) -> List[DivergenceSignal]:
        """Convert raw signals to structured DivergenceSignal objects."""
        signals = []
        raw_signals = analysis_result['divergence_signals']
        
        for row in raw_signals:
            if row[0] >= 0:  # Valid signal
                signal = DivergenceSignal(
                    timestamp=int(row[0]),
                    divergence_type=DivergenceType(int(row[1])),
                    price_level=0.0,  # Would need price array lookup
                    cvd_level=0.0,
                    strength=row[2],
                    lookback_bars=self.lookback,
                    confidence=row[3]
                )
                signals.append(signal)
        
        return signals


# Module-level factory
def create_divergence_engine(
    lookback: int = 20,
    min_strength: float = 0.5
) -> DeltaDivergenceEngine:
    """Factory function to create optimized divergence engine."""
    return DeltaDivergenceEngine(lookback, min_strength)
