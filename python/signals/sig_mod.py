"""
Chapter 2: Advanced Signal Processing & Spectral Analysis
sig_mod.py - Module root feeding spectral cycle phases and wavelet-denoised prices to time-series forecasting models
"""

import numpy as np
from typing import Dict, Optional, Tuple, List, Any
from dataclasses import dataclass, field
import threading
from collections import deque

# Import local modules
from .wavelet_transform import (
    StreamingWaveletProcessor, 
    WaveletType, 
    denoise_signal,
    extract_trend_and_noise,
    create_streaming_processor
)
from .spectral_density import (
    SpectralDensityEngine,
    SpectralAnalysisResult,
    detect_dominant_frequencies,
    estimate_cycle_period,
    create_spectral_engine
)


@dataclass
class SpectralSignal:
    """Unified spectral signal for model consumption"""
    timestamp: int
    symbol: str
    
    # Wavelet metrics
    denoised_price: float
    trend_component: float
    noise_component: float
    wavelet_level: int
    
    # Spectral metrics
    dominant_frequency: float
    dominant_period: float
    spectral_entropy: float
    total_power: float
    
    # Cycle phase
    cycle_phase: float  # Radians [0, 2π]
    cycle_position: str  # 'peak', 'trough', 'rising', 'falling'
    
    # Algorithmic signatures
    twap_detected: bool
    vwap_detected: bool
    
    # Model-ready features
    feature_vector: np.ndarray = field(default_factory=lambda: np.zeros(10))
    
    # Confidence
    confidence: float = 0.0


class PhaseTracker:
    """
    Track the phase of dominant cyclical components in real-time.
    Uses Hilbert transform approximation for instantaneous phase.
    """
    
    def __init__(self, target_period: float = 100):
        self.target_period = target_period
        self._buffer = np.zeros(int(target_period * 2), dtype=np.float64)
        self._buffer_pos = 0
        self._phase_history = deque(maxlen=100)
    
    def update(self, value: float) -> float:
        """
        Update phase tracker with new value.
        Returns current phase in radians [0, 2π].
        """
        # Add to circular buffer
        self._buffer[self._buffer_pos] = value
        self._buffer_pos = (self._buffer_pos + 1) % len(self._buffer)
        
        # Estimate phase using quadrature method
        phase = self._estimate_phase()
        self._phase_history.append(phase)
        
        return phase
    
    def _estimate_phase(self) -> float:
        """Estimate instantaneous phase using delayed signal."""
        delay = int(self.target_period / 4)  # 90-degree delay
        
        if len(self._buffer) < delay + 10:
            return 0.0
        
        # Get current and delayed values
        current_idx = self._buffer_pos - 1
        delayed_idx = (self._buffer_pos - delay - 1) % len(self._buffer)
        
        if current_idx < 0:
            current_idx += len(self._buffer)
        if delayed_idx < 0:
            delayed_idx += len(self._buffer)
        
        # In-phase component (current)
        i_comp = self._buffer[current_idx]
        
        # Quadrature component (delayed)
        q_comp = self._buffer[delayed_idx]
        
        # Calculate phase
        phase = np.arctan2(q_comp, i_comp)
        
        # Normalize to [0, 2π]
        if phase < 0:
            phase += 2 * np.pi
        
        return phase
    
    def get_cycle_position(self, phase: float) -> str:
        """Convert phase to descriptive position."""
        if phase < np.pi / 4 or phase >= 7 * np.pi / 4:
            return 'peak'
        elif phase < 3 * np.pi / 4:
            return 'falling'
        elif phase < 5 * np.pi / 4:
            return 'trough'
        else:
            return 'rising'


class SignalProcessingModule:
    """
    Central module for spectral signal processing.
    Aggregates wavelet denoising and spectral analysis into model-ready features.
    """
    
    def __init__(
        self,
        wavelet_type: WaveletType = WaveletType.DB4,
        wavelet_level: int = 3,
        chunk_size: int = 512,
        sampling_rate: float = 1.0,
        segment_size: int = 256
    ):
        # Initialize wavelet processor
        self.wavelet_processor = create_streaming_processor(
            wavelet_type, wavelet_level, chunk_size
        )
        
        # Initialize spectral engine
        self.spectral_engine = create_spectral_engine(
            sampling_rate, segment_size
        )
        
        # Phase trackers for multiple timeframes
        self._phase_trackers = {
            'short': PhaseTracker(target_period=20),
            'medium': PhaseTracker(target_period=50),
            'long': PhaseTracker(target_period=100)
        }
        
        # Signal queue
        self._signal_queue: deque = deque(maxlen=500)
        self._lock = threading.Lock()
        
        # State tracking
        self._last_denoised = None
        self._last_trend = None
        self._last_noise = None
        self._dominant_period_estimate = 50.0
        
        # Feature configuration
        self.feature_names = [
            'denoised_price',
            'trend',
            'noise',
            'spectral_entropy',
            'dominant_freq',
            'cycle_phase',
            'twap_flag',
            'vwap_flag',
            'power_ratio',
            'period_stability'
        ]
    
    def process_chunk(
        self,
        prices: np.ndarray,
        timestamps: np.ndarray,
        symbol: str
    ) -> Optional[SpectralSignal]:
        """
        Process a chunk of price data through the full signal pipeline.
        
        Args:
            prices: Price series
            timestamps: Timestamps
            symbol: Trading pair symbol
        
        Returns:
            SpectralSignal or None if insufficient data
        """
        if len(prices) < 10:
            return None
        
        # Step 1: Wavelet denoising
        denoised = self.wavelet_processor.process_chunk(prices.copy())
        
        if len(denoised) == 0:
            return None
        
        # Extract trend and noise
        trend, noise = extract_trend_and_noise(
            denoised, 
            self.wavelet_processor.wavelet,
            self.wavelet_processor.level
        )
        
        # Update state
        self._last_denoised = denoised[-1] if len(denoised) > 0 else prices[-1]
        self._last_trend = trend[-1] if len(trend) > 0 else prices[-1]
        self._last_noise = noise[-1] if len(noise) > 0 else 0.0
        
        # Step 2: Spectral analysis on denoised signal
        spectral_result = self.spectral_engine.analyze_welch(denoised)
        
        # Update dominant period estimate
        if len(spectral_result.dominant_periods) > 0 and spectral_result.dominant_periods[0] > 0:
            self._dominant_period_estimate = spectral_result.dominant_periods[0]
        
        # Step 3: Update phase trackers
        primary_phase = self._phase_trackers['medium'].update(self._last_denoised)
        
        # Determine cycle position
        cycle_position = self._phase_trackers['medium'].get_cycle_position(primary_phase)
        
        # Step 4: Build feature vector
        feature_vector = self._build_feature_vector(
            denoised, trend, noise, spectral_result, primary_phase
        )
        
        # Create signal
        timestamp = int(timestamps[-1]) if len(timestamps) > 0 else 0
        
        signal = SpectralSignal(
            timestamp=timestamp,
            symbol=symbol,
            denoised_price=self._last_denoised,
            trend_component=self._last_trend,
            noise_component=self._last_noise,
            wavelet_level=self.wavelet_processor.level,
            dominant_frequency=spectral_result.dominant_frequencies[0] if len(spectral_result.dominant_frequencies) > 0 else 0.0,
            dominant_period=self._dominant_period_estimate,
            spectral_entropy=spectral_result.spectral_entropy,
            total_power=spectral_result.total_power,
            cycle_phase=primary_phase,
            cycle_position=cycle_position,
            twap_detected=spectral_result.twap_signature,
            vwap_detected=spectral_result.vwap_signature,
            feature_vector=feature_vector,
            confidence=self._calculate_confidence(spectral_result)
        )
        
        # Store in queue
        with self._lock:
            self._signal_queue.append(signal)
        
        return signal
    
    def _build_feature_vector(
        self,
        denoised: np.ndarray,
        trend: np.ndarray,
        noise: np.ndarray,
        spectral_result: SpectralAnalysisResult,
        phase: float
    ) -> np.ndarray:
        """Build normalized feature vector for ML models."""
        features = np.zeros(len(self.feature_names), dtype=np.float64)
        
        # 1. Denoised price (normalized)
        if len(denoised) > 0:
            features[0] = denoised[-1] / max(np.mean(denoised), 1e-10)
        
        # 2. Trend component
        if len(trend) > 0:
            features[1] = trend[-1] / max(np.mean(trend), 1e-10)
        
        # 3. Noise level
        features[2] = np.std(noise) / max(np.std(denoised), 1e-10) if len(denoised) > 0 else 0.0
        
        # 4. Spectral entropy
        features[3] = spectral_result.spectral_entropy
        
        # 5. Dominant frequency
        features[4] = spectral_result.dominant_frequencies[0] if len(spectral_result.dominant_frequencies) > 0 else 0.0
        
        # 6. Cycle phase (normalized to [0, 1])
        features[5] = phase / (2 * np.pi)
        
        # 7. TWAP flag
        features[6] = 1.0 if spectral_result.twap_signature else 0.0
        
        # 8. VWAP flag
        features[7] = 1.0 if spectral_result.vwap_signature else 0.0
        
        # 9. Power ratio (low freq / high freq)
        n_freqs = len(spectral_result.frequencies)
        if n_freqs > 10:
            low_freq_power = np.sum(spectral_result.psd[:n_freqs//4])
            high_freq_power = np.sum(spectral_result.psd[n_freqs//4:])
            total = low_freq_power + high_freq_power
            features[8] = low_freq_power / max(total, 1e-10)
        
        # 10. Period stability (inverse of coefficient of variation)
        if len(self._phase_history) >= 10:
            phases = list(self._phase_history)[-10:]
            phase_diffs = np.diff(phases)
            features[9] = 1.0 / (np.std(phase_diffs) + 1e-10)
        
        return features
    
    @property
    def _phase_history(self) -> deque:
        """Access medium-term phase history."""
        return self._phase_trackers['medium']._phase_history
    
    def _calculate_confidence(self, spectral_result: SpectralAnalysisResult) -> float:
        """Calculate signal confidence based on spectral quality."""
        confidence = 0.5
        
        # Higher confidence with lower entropy (clearer signal)
        if spectral_result.spectral_entropy < 0.3:
            confidence += 0.2
        elif spectral_result.spectral_entropy < 0.5:
            confidence += 0.1
        
        # Higher confidence with detected patterns
        if spectral_result.twap_signature or spectral_result.vwap_signature:
            confidence += 0.15
        
        # Higher confidence with strong dominant frequency
        if len(spectral_result.dominant_frequencies) > 0:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def get_latest_signal(self) -> Optional[SpectralSignal]:
        """Get most recent signal from queue."""
        with self._lock:
            if len(self._signal_queue) > 0:
                return self._signal_queue[-1]
        return None
    
    def get_signals(self, count: int = 10) -> List[SpectralSignal]:
        """Get last N signals from queue."""
        with self._lock:
            signals = list(self._signal_queue)
            return signals[-count:]
    
    def get_denoised_view(self) -> np.ndarray:
        """Get latest denoised price series."""
        if self._last_denoised is not None:
            return np.array([self._last_denoised])
        return np.array([])
    
    def get_trend_view(self) -> np.ndarray:
        """Get latest trend component."""
        if self._last_trend is not None:
            return np.array([self._last_trend])
        return np.array([])
    
    def reset(self):
        """Reset all internal state."""
        self.wavelet_processor.reset()
        for tracker in self._phase_trackers.values():
            tracker._buffer[:] = 0
            tracker._buffer_pos = 0
            tracker._phase_history.clear()
        self._signal_queue.clear()


# Module singleton instance
_sig_module: Optional[SignalProcessingModule] = None


def get_signal_module(
    wavelet_type: WaveletType = WaveletType.DB4,
    wavelet_level: int = 3,
    sampling_rate: float = 1.0
) -> SignalProcessingModule:
    """Get or create the global signal processing module instance."""
    global _sig_module
    if _sig_module is None:
        _sig_module = SignalProcessingModule(
            wavelet_type, wavelet_level, sampling_rate=sampling_rate
        )
    return _sig_module


def reset_signal_module():
    """Reset the global signal module (for testing)."""
    global _sig_module
    _sig_module = None


# Convenience functions for direct use
def quick_spectral_analysis(
    prices: np.ndarray,
    timestamps: np.ndarray,
    fs: float = 1.0
) -> Dict[str, Any]:
    """
    Quick spectral analysis without module initialization.
    
    Returns:
        Dictionary with key spectral metrics
    """
    # Denoise
    denoised = denoise_signal(prices, WaveletType.DB4, level=3)
    
    # Spectral analysis
    engine = create_spectral_engine(fs, segment_size=min(256, len(denoised)))
    result = engine.analyze_welch(denoised)
    
    return {
        'denoised': denoised,
        'dominant_frequency': result.dominant_frequencies[0] if len(result.dominant_frequencies) > 0 else 0.0,
        'dominant_period': result.dominant_periods[0] if len(result.dominant_periods) > 0 else 0.0,
        'spectral_entropy': result.spectral_entropy,
        'twap_detected': result.twap_signature,
        'vwap_detected': result.vwap_signature
    }
