"""
Chapter 2: Advanced Signal Processing & Spectral Analysis
wavelet_transform.py - Haar and Daubechies wavelet transforms for multi-resolution tick data denoising
"""

import numpy as np
from numba import njit, prange
from typing import Tuple, Optional, List
from enum import Enum

# PyWavelets for production wavelet transforms
try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False


class WaveletType(Enum):
    """Supported wavelet families"""
    HAAR = 'haar'
    DB1 = 'db1'      # Same as Haar
    DB2 = 'db2'
    DB3 = 'db3'
    DB4 = 'db4'
    DB6 = 'db6'
    DB8 = 'db8'
    SYM2 = 'sym2'
    SYM4 = 'sym4'
    COIF1 = 'coif1'
    COIF2 = 'coif2'


@njit(cache=True, nogil=True)
def haar_wavelet_decompose(signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Single-level Haar wavelet decomposition.
    Computes approximation and detail coefficients.
    
    Args:
        signal: Input signal (must be even length)
    
    Returns:
        Tuple of (approximation_coeffs, detail_coeffs)
    """
    n = len(signal)
    if n % 2 != 0:
        # Pad with last value
        n += 1
    
    half_n = n // 2
    approx = np.empty(half_n, dtype=np.float64)
    detail = np.empty(half_n, dtype=np.float64)
    
    sqrt2_inv = 0.7071067811865476  # 1/sqrt(2)
    
    for i in range(half_n):
        idx = i * 2
        if idx + 1 < len(signal):
            s0 = signal[idx]
            s1 = signal[idx + 1]
        else:
            s0 = signal[-1]
            s1 = signal[-1]
        
        # Haar transform
        approx[i] = (s0 + s1) * sqrt2_inv
        detail[i] = (s0 - s1) * sqrt2_inv
    
    return approx, detail


@njit(cache=True, nogil=True)
def haar_wavelet_reconstruct(
    approx: np.ndarray,
    detail: np.ndarray
) -> np.ndarray:
    """
    Single-level Haar wavelet reconstruction.
    
    Args:
        approx: Approximation coefficients
        detail: Detail coefficients
    
    Returns:
        Reconstructed signal
    """
    n = len(approx)
    signal = np.empty(n * 2, dtype=np.float64)
    
    sqrt2_inv = 0.7071067811865476
    
    for i in range(n):
        a = approx[i]
        d = detail[i]
        
        # Inverse Haar transform
        signal[i * 2] = (a + d) * sqrt2_inv
        signal[i * 2 + 1] = (a - d) * sqrt2_inv
    
    return signal


@njit(cache=True, nogil=True)
def multi_level_haar_decompose(
    signal: np.ndarray,
    level: int = 3
) -> List[np.ndarray]:
    """
    Multi-level Haar wavelet decomposition.
    
    Args:
        signal: Input signal
        level: Number of decomposition levels
    
    Returns:
        List of [cA_n, cD_n, cD_n-1, ..., cD_1] where cA is approximation
        and cD are detail coefficients at each level
    """
    coeffs = []
    current = signal.copy()
    
    for _ in range(level):
        if len(current) < 2:
            break
        
        approx, detail = haar_wavelet_decompose(current)
        coeffs.append(detail)
        current = approx
    
    # Add final approximation
    coeffs.append(current)
    
    # Reverse to get [cA_n, cD_n, ..., cD_1]
    coeffs.reverse()
    return coeffs


@njit(cache=True, nogil=True)
def multi_level_haar_reconstruct(coeffs: List[np.ndarray]) -> np.ndarray:
    """
    Multi-level Haar wavelet reconstruction.
    
    Args:
        coeffs: Coefficients in format [cA_n, cD_n, ..., cD_1]
    
    Returns:
        Reconstructed signal
    """
    if len(coeffs) < 2:
        return coeffs[0] if len(coeffs) > 0 else np.array([])
    
    # Start with approximation
    current = coeffs[-len(coeffs)]  # cA_n
    
    # Reconstruct from deepest level
    for i in range(len(coeffs) - 1, 0, -1):
        detail = coeffs[-i]
        min_len = min(len(current), len(detail))
        
        # Truncate to matching lengths
        current_trunc = current[:min_len]
        detail_trunc = detail[:min_len]
        
        current = haar_wavelet_reconstruct(current_trunc, detail_trunc)
    
    return current


@njit(cache=True, nogil=True)
def wavelet_denoise_threshold(
    detail_coeffs: np.ndarray,
    threshold: float,
    mode: int = 0  # 0=soft, 1=hard
) -> np.ndarray:
    """
    Apply thresholding to wavelet detail coefficients for denoising.
    
    Args:
        detail_coeffs: Detail coefficients to threshold
        threshold: Threshold value
        mode: 0 for soft thresholding, 1 for hard thresholding
    
    Returns:
        Thresholded coefficients
    """
    n = len(detail_coeffs)
    result = np.empty(n, dtype=np.float64)
    
    for i in range(n):
        coeff = detail_coeffs[i]
        abs_coeff = abs(coeff)
        
        if mode == 0:  # Soft thresholding
            if abs_coeff <= threshold:
                result[i] = 0.0
            else:
                result[i] = np.sign(coeff) * (abs_coeff - threshold)
        else:  # Hard thresholding
            if abs_coeff <= threshold:
                result[i] = 0.0
            else:
                result[i] = coeff
    
    return result


@njit(cache=True, nogil=True)
def calculate_universal_threshold(
    detail_coeffs: np.ndarray
) -> float:
    """
    Calculate universal threshold (VisuShrink) for wavelet denoising.
    threshold = sigma * sqrt(2 * log(n))
    
    Args:
        detail_coeffs: Detail coefficients (typically finest level)
    
    Returns:
        Universal threshold value
    """
    n = len(detail_coeffs)
    if n == 0:
        return 0.0
    
    # Estimate noise standard deviation using MAD (Median Absolute Deviation)
    # sigma = MAD / 0.6745
    median = np.median(np.abs(detail_coeffs))
    sigma = median / 0.6745
    
    # Universal threshold
    threshold = sigma * np.sqrt(2.0 * np.log(n))
    
    return threshold


class StreamingWaveletProcessor:
    """
    Memory-efficient streaming wavelet processor for continuous tick data.
    Processes data in chunks to prevent loading massive arrays into memory.
    """
    
    def __init__(
        self,
        wavelet: WaveletType = WaveletType.HAAR,
        level: int = 3,
        chunk_size: int = 1024,
        overlap: int = 128
    ):
        self.wavelet = wavelet
        self.level = level
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        # Overlap buffer for seamless processing
        self._overlap_buffer = np.zeros(overlap, dtype=np.float64)
        self._buffer_pos = 0
        
        # Accumulated results
        self._denoised_output = []
        self._approximation_history = []
    
    def process_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """
        Process a single chunk of data with wavelet denoising.
        
        Args:
            chunk: Input data chunk
        
        Returns:
            Denoised chunk
        """
        # Prepend overlap buffer
        if self._buffer_pos > 0:
            extended_chunk = np.concatenate([
                self._overlap_buffer[:self._buffer_pos],
                chunk
            ])
        else:
            extended_chunk = chunk
        
        # Perform wavelet decomposition and denoising
        if PYWT_AVAILABLE:
            denoised = self._pywt_denoise(extended_chunk)
        else:
            denoised = self._numba_denoise(extended_chunk)
        
        # Remove overlap portion
        if self._buffer_pos > 0:
            result = denoised[self._buffer_pos:]
        else:
            result = denoised
        
        # Update overlap buffer
        if len(chunk) >= self.overlap:
            self._overlap_buffer = chunk[-self.overlap:].copy()
            self._buffer_pos = self.overlap
        else:
            self._overlap_buffer[:len(chunk)] = chunk
            self._buffer_pos = len(chunk)
        
        return result
    
    def _pywt_denoise(self, signal: np.ndarray) -> np.ndarray:
        """Use PyWavelets for high-quality denoising."""
        wavelet_name = self.wavelet.value
        
        # Decompose
        coeffs = pywt.wavedec(signal, wavelet_name, level=self.level)
        
        # Estimate noise from finest detail coefficients
        if len(coeffs) > 1:
            finest_detail = coeffs[-1]
            threshold = calculate_universal_threshold(finest_detail)
            
            # Apply soft thresholding to detail coefficients
            for i in range(1, len(coeffs)):
                coeffs[i] = pywt.threshold(coeffs[i], threshold, mode='soft')
        
        # Reconstruct
        denoised = pywt.waverec(coeffs, wavelet_name)
        
        # Ensure same length as input
        if len(denoised) > len(signal):
            denoised = denoised[:len(signal)]
        
        return denoised
    
    def _numba_denoise(self, signal: np.ndarray) -> np.ndarray:
        """Fallback Numba-based denoising when PyWavelets unavailable."""
        # Multi-level decomposition
        coeffs = multi_level_haar_decompose(signal, self.level)
        
        if len(coeffs) > 1:
            # Calculate threshold from finest detail
            threshold = calculate_universal_threshold(coeffs[-1])
            
            # Apply soft thresholding to all detail coefficients
            for i in range(len(coeffs) - 1):
                coeffs[i] = wavelet_denoise_threshold(coeffs[i], threshold, mode=0)
        
        # Reconstruct
        denoised = multi_level_haar_reconstruct(coeffs)
        
        # Ensure same length
        if len(denoised) > len(signal):
            denoised = denoised[:len(signal)]
        
        return denoised
    
    def process_stream(self, signal_iterator) -> np.ndarray:
        """
        Process a stream of data chunks.
        
        Args:
            signal_iterator: Iterator yielding numpy arrays
        
        Returns:
            Concatenated denoised output
        """
        outputs = []
        
        for chunk in signal_iterator:
            denoised = self.process_chunk(chunk)
            outputs.append(denoised)
        
        return np.concatenate(outputs)
    
    def reset(self):
        """Reset internal state for new stream."""
        self._overlap_buffer = np.zeros(self.overlap, dtype=np.float64)
        self._buffer_pos = 0
        self._denoised_output = []
        self._approximation_history = []


def wavelet_transform(
    signal: np.ndarray,
    wavelet: WaveletType = WaveletType.DB4,
    level: int = 3
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Perform complete wavelet transform on signal.
    
    Args:
        signal: Input signal
        wavelet: Wavelet type
        level: Decomposition level
    
    Returns:
        Tuple of (detail_coefficients_list, approximation_coefficients)
    """
    if not PYWT_AVAILABLE:
        # Fallback to Numba implementation
        coeffs = multi_level_haar_decompose(signal, level)
        approx = coeffs[-1]
        details = coeffs[:-1]
        return details, approx
    
    wavelet_name = wavelet.value
    coeffs = pywt.wavedec(signal, wavelet_name, level=level)
    
    approx = coeffs[0]
    details = coeffs[1:]
    
    return details, approx


def inverse_wavelet_transform(
    details: List[np.ndarray],
    approx: np.ndarray,
    wavelet: WaveletType = WaveletType.DB4
) -> np.ndarray:
    """
    Reconstruct signal from wavelet coefficients.
    
    Args:
        details: List of detail coefficient arrays
        approx: Approximation coefficients
        wavelet: Wavelet type
    
    Returns:
        Reconstructed signal
    """
    if not PYWT_AVAILABLE:
        coeffs = [approx] + list(reversed(details))
        return multi_level_haar_reconstruct(coeffs)
    
    wavelet_name = wavelet.value
    coeffs = [approx] + details
    return pywt.waverec(coeffs, wavelet_name)


def denoise_signal(
    signal: np.ndarray,
    wavelet: WaveletType = WaveletType.DB4,
    level: int = 3,
    method: str = 'universal'
) -> np.ndarray:
    """
    Denoise a signal using wavelet thresholding.
    
    Args:
        signal: Noisy input signal
        wavelet: Wavelet type
        level: Decomposition level
        method: Thresholding method ('universal', 'sure', 'fixed')
    
    Returns:
        Denoised signal
    """
    if not PYWT_AVAILABLE:
        processor = StreamingWaveletProcessor(wavelet, level)
        return processor._numba_denoise(signal)
    
    wavelet_name = wavelet.value
    
    # Decompose
    coeffs = pywt.wavedec(signal, wavelet_name, level=level)
    
    # Estimate threshold
    if method == 'universal':
        finest_detail = coeffs[-1]
        threshold = calculate_universal_threshold(finest_detail)
        mode = 'soft'
    elif method == 'fixed':
        threshold = 0.5 * np.std(signal)
        mode = 'soft'
    else:  # SURE (Stein's Unbiased Risk Estimate)
        # Simplified SURE implementation
        finest_detail = coeffs[-1]
        sigma = np.median(np.abs(finest_detail)) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(finest_detail)))
        mode = 'soft'
    
    # Apply thresholding
    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], threshold, mode=mode)
    
    # Reconstruct
    denoised = pywt.waverec(coeffs, wavelet_name)
    
    # Trim to original length
    if len(denoised) > len(signal):
        denoised = denoised[:len(signal)]
    
    return denoised


def extract_trend_and_noise(
    signal: np.ndarray,
    wavelet: WaveletType = WaveletType.DB4,
    level: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Separate signal into low-frequency trend and high-frequency noise.
    
    Args:
        signal: Input signal
        wavelet: Wavelet type
        level: Decomposition level
    
    Returns:
        Tuple of (trend, noise)
    """
    denoised = denoise_signal(signal, wavelet, level)
    noise = signal - denoised
    
    return denoised, noise


# Module convenience functions
def create_streaming_processor(
    wavelet: WaveletType = WaveletType.HAAR,
    level: int = 3,
    chunk_size: int = 1024
) -> StreamingWaveletProcessor:
    """Factory function to create streaming wavelet processor."""
    return StreamingWaveletProcessor(wavelet, level, chunk_size)
