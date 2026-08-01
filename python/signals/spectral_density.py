"""
Chapter 2: Advanced Signal Processing & Spectral Analysis
spectral_density.py - Welch and Lomb-Scargle periodogram calculators for detecting algorithmic execution cycles
"""

import numpy as np
from numba import njit, prange
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass


@njit(cache=True, nogil=True)
def _hann_window(n: int) -> np.ndarray:
    """Generate Hann window coefficients."""
    window = np.empty(n, dtype=np.float64)
    for i in range(n):
        window[i] = 0.5 * (1.0 - np.cos(2.0 * np.pi * i / (n - 1)))
    return window


@njit(cache=True, nogil=True)
def _hamming_window(n: int) -> np.ndarray:
    """Generate Hamming window coefficients."""
    window = np.empty(n, dtype=np.float64)
    for i in range(n):
        window[i] = 0.54 - 0.46 * np.cos(2.0 * np.pi * i / (n - 1))
    return window


@njit(cache=True, nogil=True)
def _blackman_window(n: int) -> np.ndarray:
    """Generate Blackman window coefficients."""
    window = np.empty(n, dtype=np.float64)
    for i in range(n):
        window[i] = (0.42 - 
                    0.5 * np.cos(2.0 * np.pi * i / (n - 1)) + 
                    0.08 * np.cos(4.0 * np.pi * i / (n - 1)))
    return window


@njit(cache=True, nogil=True)
def _next_power_of_2(n: int) -> int:
    """Find next power of 2 >= n for FFT efficiency."""
    if n <= 0:
        return 1
    
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    return n + 1


@njit(cache=True, nogil=True)
def _dft_chunk(signal: np.ndarray, frequencies: np.ndarray, fs: float) -> np.ndarray:
    """
    Compute DFT at specific frequencies (for non-power-of-2 lengths).
    Slower but more flexible than FFT.
    """
    n = len(signal)
    n_freqs = len(frequencies)
    spectrum = np.zeros(n_freqs, dtype=np.complex128)
    
    for k in range(n_freqs):
        freq = frequencies[k]
        real_sum = 0.0
        imag_sum = 0.0
        
        for t in range(n):
            angle = -2.0 * np.pi * freq * t / fs
            real_sum += signal[t] * np.cos(angle)
            imag_sum += signal[t] * np.sin(angle)
        
        spectrum[k] = complex(real_sum, imag_sum)
    
    return spectrum


@njit(cache=True, nogil=True)
def welch_periodogram(
    signal: np.ndarray,
    fs: float = 1.0,
    nperseg: int = 256,
    noverlap: int = 128,
    window_type: int = 0  # 0=hann, 1=hamming, 2=blackman
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Welch's method periodogram for power spectral density estimation.
    
    Args:
        signal: Input signal
        fs: Sampling frequency
        nperseg: Length of each segment
        noverlap: Number of points to overlap between segments
        window_type: Window function type
    
    Returns:
        Tuple of (frequencies, psd)
    """
    n = len(signal)
    
    if n < nperseg:
        nperseg = n
        noverlap = 0
    
    # Generate window
    if window_type == 0:
        window = _hann_window(nperseg)
    elif window_type == 1:
        window = _hamming_window(nperseg)
    else:
        window = _blackman_window(nperseg)
    
    # Calculate window power for normalization
    window_power = 0.0
    for w in window:
        window_power += w * w
    
    # Number of segments
    step = nperseg - noverlap
    n_segments = max(1, (n - noverlap) // step)
    
    # Frequency bins (only positive frequencies up to Nyquist)
    n_freqs = nperseg // 2 + 1
    frequencies = np.empty(n_freqs, dtype=np.float64)
    for i in range(n_freqs):
        frequencies[i] = i * fs / nperseg
    
    # Accumulate PSD across segments
    psd_accum = np.zeros(n_freqs, dtype=np.float64)
    
    for seg in range(n_segments):
        start = seg * step
        end = min(start + nperseg, n)
        
        if end - start < nperseg:
            continue
        
        # Extract and window segment
        segment = np.empty(nperseg, dtype=np.float64)
        for i in range(nperseg):
            segment[i] = signal[start + i] * window[i]
        
        # Compute FFT manually (DFT for flexibility)
        for k in range(n_freqs):
            freq_idx = k
            real_sum = 0.0
            imag_sum = 0.0
            
            for t in range(nperseg):
                angle = -2.0 * np.pi * freq_idx * t / nperseg
                real_sum += segment[t] * np.cos(angle)
                imag_sum += segment[t] * np.sin(angle)
            
            # Power = |FFT|^2
            power = real_sum * real_sum + imag_sum * imag_sum
            psd_accum[k] += power
    
    # Average and normalize
    psd = psd_accum / (n_segments * fs * window_power)
    
    # Double non-DC/non-Nyquist components for single-sided spectrum
    for i in range(1, n_freqs - 1):
        psd[i] *= 2.0
    
    return frequencies, psd


@njit(cache=True, nogil=True)
def lomb_scargle_periodogram(
    times: np.ndarray,
    signal: np.ndarray,
    frequencies: np.ndarray,
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Lomb-Scargle periodogram for unevenly sampled data.
    Detects periodic signals in irregular time series (ideal for tick data).
    
    Args:
        times: Timestamps of observations
        signal: Signal values
        frequencies: Frequencies to evaluate
        normalize: Whether to normalize the periodogram
    
    Returns:
        Tuple of (frequencies, power)
    """
    n_times = len(times)
    n_freqs = len(frequencies)
    
    power = np.zeros(n_freqs, dtype=np.float64)
    
    # Pre-compute mean and variance for normalization
    signal_mean = 0.0
    signal_var = 0.0
    
    for s in signal:
        signal_mean += s
    signal_mean /= n_times
    
    for s in signal:
        diff = s - signal_mean
        signal_var += diff * diff
    signal_var /= n_times
    
    if signal_var == 0:
        signal_var = 1.0
    
    # Compute tau for each frequency (phase offset)
    for k in range(n_freqs):
        freq = frequencies[k]
        omega = 2.0 * np.pi * freq
        
        # Compute tau
        sum_sin_2wt = 0.0
        sum_cos_2wt = 0.0
        
        for i in range(n_times):
            wt = omega * times[i]
            sum_sin_2wt += np.sin(2.0 * wt)
            sum_cos_2wt += np.cos(2.0 * wt)
        
        if abs(sum_cos_2wt) > 1e-10:
            tau = np.arctan2(sum_sin_2wt, sum_cos_2wt) / (2.0 * omega)
        else:
            tau = 0.0
        
        # Compute power at this frequency
        numerator = 0.0
        denominator_cos = 0.0
        denominator_sin = 0.0
        
        for i in range(n_times):
            wt = omega * (times[i] - tau)
            cos_wt = np.cos(wt)
            sin_wt = np.sin(wt)
            
            centered_signal = signal[i] - signal_mean
            
            numerator += (centered_signal * cos_wt) ** 2
            denominator_cos += cos_wt * cos_wt
            
            numerator += (centered_signal * sin_wt) ** 2
            denominator_sin += sin_wt * sin_wt
        
        if denominator_cos > 1e-10 and denominator_sin > 1e-10:
            p = (numerator / denominator_cos + numerator / denominator_sin) / 2.0
            
            if normalize:
                p /= signal_var
            
            power[k] = p
    
    return frequencies, power


@njit(cache=True, nogil=True)
def detect_dominant_frequencies(
    frequencies: np.ndarray,
    psd: np.ndarray,
    n_peaks: int = 5,
    threshold_factor: float = 2.0
) -> np.ndarray:
    """
    Detect dominant frequency peaks in PSD.
    
    Args:
        frequencies: Frequency array
        psd: Power spectral density
        n_peaks: Maximum number of peaks to return
        threshold_factor: Minimum peak height relative to median
    
    Returns:
        Array of dominant frequencies
    """
    n = len(frequencies)
    if n < 3:
        return np.array([])
    
    # Calculate threshold
    median_psd = np.median(psd[1:])  # Exclude DC
    threshold = median_psd * threshold_factor
    
    # Find local maxima above threshold
    max_peaks = min(n_peaks, n // 3)
    peak_freqs = np.zeros(max_peaks, dtype=np.float64)
    peak_count = 0
    
    for i in range(1, n - 1):
        if psd[i] > threshold:
            if psd[i] > psd[i - 1] and psd[i] > psd[i + 1]:
                if peak_count < max_peaks:
                    peak_freqs[peak_count] = frequencies[i]
                    peak_count += 1
    
    # Sort by power (descending) and return
    return peak_freqs[:peak_count]


@njit(cache=True, nogil=True)
def estimate_cycle_period(
    frequencies: np.ndarray,
    psd: np.ndarray
) -> float:
    """
    Estimate dominant cycle period from PSD.
    
    Args:
        frequencies: Frequency array
        psd: Power spectral density
    
    Returns:
        Estimated cycle period (1/frequency)
    """
    if len(frequencies) == 0 or len(psd) == 0:
        return 0.0
    
    # Find frequency with maximum power (excluding DC)
    max_power = 0.0
    dominant_freq = frequencies[1] if len(frequencies) > 1 else 0.0
    
    for i in range(1, len(frequencies)):
        if psd[i] > max_power:
            max_power = psd[i]
            dominant_freq = frequencies[i]
    
    if dominant_freq > 0:
        return 1.0 / dominant_freq
    
    return 0.0


@dataclass
class SpectralAnalysisResult:
    """Container for spectral analysis results"""
    frequencies: np.ndarray
    psd: np.ndarray
    dominant_frequencies: np.ndarray
    dominant_periods: np.ndarray
    total_power: float
    spectral_entropy: float
    twap_signature: bool
    vwap_signature: bool


class SpectralDensityEngine:
    """
    Engine for detecting algorithmic execution patterns through spectral analysis.
    Identifies TWAP/VWAP footprints and hidden execution cycles.
    """
    
    def __init__(
        self,
        sampling_rate: float = 1.0,
        segment_size: int = 256,
        overlap_ratio: float = 0.5
    ):
        self.sampling_rate = sampling_rate
        self.segment_size = segment_size
        self.overlap_ratio = overlap_ratio
        self.noverlap = int(segment_size * overlap_ratio)
        
        # Pre-compute analysis frequencies
        self._freq_bins = segment_size // 2 + 1
    
    def analyze_welch(
        self,
        signal: np.ndarray,
        window_type: int = 0
    ) -> SpectralAnalysisResult:
        """
        Perform Welch PSD analysis on signal.
        
        Args:
            signal: Input signal
            window_type: Window function (0=hann, 1=hamming, 2=blackman)
        
        Returns:
            SpectralAnalysisResult
        """
        # Compute Welch periodogram
        frequencies, psd = welch_periodogram(
            signal,
            fs=self.sampling_rate,
            nperseg=self.segment_size,
            noverlap=self.noverlap,
            window_type=window_type
        )
        
        # Detect dominant frequencies
        dom_freqs = detect_dominant_frequencies(frequencies, psd, n_peaks=5)
        
        # Convert to periods
        dom_periods = np.zeros(len(dom_freqs), dtype=np.float64)
        for i, f in enumerate(dom_freqs):
            if f > 0:
                dom_periods[i] = 1.0 / f
        
        # Calculate total power
        total_power = np.sum(psd)
        
        # Calculate spectral entropy
        spectral_entropy = self._calculate_spectral_entropy(psd)
        
        # Detect algorithmic signatures
        twap_sig = self._detect_twap_signature(frequencies, psd)
        vwap_sig = self._detect_vwap_signature(frequencies, psd)
        
        return SpectralAnalysisResult(
            frequencies=frequencies,
            psd=psd,
            dominant_frequencies=dom_freqs,
            dominant_periods=dom_periods,
            total_power=total_power,
            spectral_entropy=spectral_entropy,
            twap_signature=twap_sig,
            vwap_signature=vwap_sig
        )
    
    def analyze_lomb_scargle(
        self,
        times: np.ndarray,
        signal: np.ndarray,
        min_freq: float = 0.001,
        max_freq: float = 0.5,
        n_freqs: int = 100
    ) -> SpectralAnalysisResult:
        """
        Perform Lomb-Scargle analysis on unevenly sampled data.
        
        Args:
            times: Observation timestamps
            signal: Signal values
            min_freq: Minimum frequency to search
            max_freq: Maximum frequency to search
            n_freqs: Number of frequency bins
        
        Returns:
            SpectralAnalysisResult
        """
        # Generate frequency grid
        frequencies = np.linspace(min_freq, max_freq, n_freqs)
        
        # Compute Lomb-Scargle periodogram
        frequencies, psd = lomb_scargle_periodogram(
            times, signal, frequencies, normalize=True
        )
        
        # Detect dominant frequencies
        dom_freqs = detect_dominant_frequencies(frequencies, psd, n_peaks=5)
        
        # Convert to periods
        dom_periods = np.zeros(len(dom_freqs), dtype=np.float64)
        for i, f in enumerate(dom_freqs):
            if f > 0:
                dom_periods[i] = 1.0 / f
        
        # Calculate metrics
        total_power = np.sum(psd)
        spectral_entropy = self._calculate_spectral_entropy(psd)
        twap_sig = self._detect_twap_signature(frequencies, psd)
        vwap_sig = self._detect_vwap_signature(frequencies, psd)
        
        return SpectralAnalysisResult(
            frequencies=frequencies,
            psd=psd,
            dominant_frequencies=dom_freqs,
            dominant_periods=dom_periods,
            total_power=total_power,
            spectral_entropy=spectral_entropy,
            twap_signature=twap_sig,
            vwap_signature=vwap_sig
        )
    
    @staticmethod
    @njit(cache=True, nogil=True)
    def _calculate_spectral_entropy(psd: np.ndarray) -> float:
        """
        Calculate spectral entropy - measure of spectral complexity.
        Low entropy = few dominant frequencies (periodic)
        High entropy = many frequencies (noise-like)
        """
        n = len(psd)
        if n == 0:
            return 0.0
        
        # Normalize PSD to probability distribution
        total = np.sum(psd)
        if total == 0:
            return 0.0
        
        entropy = 0.0
        for p in psd:
            if p > 0:
                p_norm = p / total
                entropy -= p_norm * np.log(p_norm + 1e-10)
        
        # Normalize by maximum entropy
        max_entropy = np.log(n)
        if max_entropy > 0:
            entropy /= max_entropy
        
        return entropy
    
    @staticmethod
    @njit(cache=True, nogil=True)
    def _detect_twap_signature(
        frequencies: np.ndarray,
        psd: np.ndarray
    ) -> bool:
        """
        Detect TWAP (Time-Weighted Average Price) execution signature.
        TWAP creates regular periodic patterns in order flow.
        """
        # Look for strong peaks at regular intervals
        # Typical TWAP intervals: 1min, 5min, 15min, 30min
        
        if len(frequencies) < 10:
            return False
        
        # Check for peaks at common TWAP frequencies
        twap_candidates = [1.0/60, 1.0/300, 1.0/900, 1.0/1800]  # Hz assuming 1Hz sampling
        
        median_psd = np.median(psd[1:])
        threshold = median_psd * 3.0
        
        for target_freq in twap_candidates:
            # Find closest frequency bin
            closest_idx = np.argmin(np.abs(frequencies - target_freq))
            
            if closest_idx > 0 and closest_idx < len(psd) - 1:
                if psd[closest_idx] > threshold:
                    # Check if it's a local maximum
                    if psd[closest_idx] > psd[closest_idx - 1] and \
                       psd[closest_idx] > psd[closest_idx + 1]:
                        return True
        
        return False
    
    @staticmethod
    @njit(cache=True, nogil=True)
    def _detect_vwap_signature(
        frequencies: np.ndarray,
        psd: np.ndarray
    ) -> bool:
        """
        Detect VWAP (Volume-Weighted Average Price) execution signature.
        VWAP tends to show clustering around market open/close.
        """
        # VWAP signatures often show daily/hourly patterns
        # Look for low-frequency dominance with specific harmonic structure
        
        if len(frequencies) < 10:
            return False
        
        # Check for power concentration in very low frequencies
        low_freq_mask = frequencies < 0.01  # Periods > 100 samples
        high_freq_mask = frequencies >= 0.01
        
        if np.sum(low_freq_mask) == 0 or np.sum(high_freq_mask) == 0:
            return False
        
        low_freq_power = np.sum(psd[low_freq_mask])
        high_freq_power = np.sum(psd[high_freq_mask])
        
        total_power = low_freq_power + high_freq_power
        
        if total_power == 0:
            return False
        
        # VWAP typically has >70% power in low frequencies
        low_freq_ratio = low_freq_power / total_power
        
        return low_freq_ratio > 0.7


# Module convenience functions
def create_spectral_engine(
    sampling_rate: float = 1.0,
    segment_size: int = 256
) -> SpectralDensityEngine:
    """Factory function to create spectral density engine."""
    return SpectralDensityEngine(sampling_rate, segment_size)


def quick_welch_analysis(
    signal: np.ndarray,
    fs: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Quick Welch PSD analysis with default parameters."""
    return welch_periodogram(signal, fs=fs)


def quick_lomb_scargle(
    times: np.ndarray,
    signal: np.ndarray,
    n_freqs: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """Quick Lomb-Scargle analysis with automatic frequency range."""
    if len(times) < 2:
        return np.array([]), np.array([])
    
    # Estimate frequency range from data
    dt = np.mean(np.diff(times))
    if dt <= 0:
        dt = 1.0
    
    min_freq = 1.0 / (len(times) * dt)
    max_freq = 0.5 / dt
    
    frequencies = np.linspace(min_freq, max_freq, n_freqs)
    return lomb_scargle_periodogram(times, signal, frequencies)
