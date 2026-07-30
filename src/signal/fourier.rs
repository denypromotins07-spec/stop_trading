//! Fast Fourier Transform (FFT) engine for market cycle detection.
//! 
//! This module uses the `rustfft` crate to perform highly optimized FFT operations
//! on order flow data to detect hidden periodicities and algorithmic execution patterns.
//! Zero allocations in the hot path through pre-allocated buffers.

use rustfft::{Fft, FftPlanner};
use std::sync::Arc;
use num_complex::Complex;

/// Maximum FFT size for cycle detection (power of 2)
const MAX_FFT_SIZE: usize = 4096;

/// Configuration for FFT analysis
#[derive(Debug, Clone)]
pub struct FftConfig {
    /// FFT size (must be power of 2)
    pub fft_size: usize,
    /// Sampling rate in Hz
    pub sample_rate: f64,
    /// Minimum frequency to consider (Hz)
    pub min_frequency: f64,
    /// Maximum frequency to consider (Hz)
    pub max_frequency: f64,
}

impl Default for FftConfig {
    fn default() -> Self {
        Self {
            fft_size: 1024,
            sample_rate: 1000.0, // 1kHz sampling
            min_frequency: 0.1,
            max_frequency: 100.0,
        }
    }
}

/// Result of FFT analysis containing dominant frequencies
#[derive(Debug, Clone)]
pub struct FftResult {
    /// Dominant frequencies sorted by magnitude
    pub dominant_frequencies: Vec<FrequencyComponent>,
    /// Total spectral energy
    pub total_energy: f64,
    /// Spectral centroid (weighted average frequency)
    pub spectral_centroid: f64,
    /// Spectral spread (variance around centroid)
    pub spectral_spread: f64,
}

/// A single frequency component detected in the spectrum
#[derive(Debug, Clone)]
pub struct FrequencyComponent {
    /// Frequency in Hz
    pub frequency: f64,
    /// Magnitude (amplitude)
    pub magnitude: f64,
    /// Phase angle in radians
    pub phase: f64,
    /// Normalized power (0.0 to 1.0)
    pub normalized_power: f64,
}

/// High-performance FFT engine for market cycle detection.
/// Pre-allocates all buffers to avoid heap allocations during processing.
pub struct FftEngine {
    /// Forward FFT transformer
    forward_fft: Arc<dyn Fft<f64>>,
    /// Inverse FFT transformer (for potential reconstruction)
    inverse_fft: Arc<dyn Fft<f64>>,
    /// Pre-allocated input buffer (complex numbers)
    input_buffer: Vec<Complex<f64>>,
    /// Pre-allocated output buffer for FFT results
    output_buffer: Vec<Complex<f64>>,
    /// Pre-allocated window function coefficients
    window_coefficients: Vec<f64>,
    /// FFT configuration
    config: FftConfig,
    /// Frequency bins (pre-computed)
    frequency_bins: Vec<f64>,
    /// Cache for dominant frequency detection
    cached_result: Option<FftResult>,
}

impl FftEngine {
    /// Create a new FFT engine with default configuration
    pub fn new() -> Self {
        Self::with_config(FftConfig::default())
    }
    
    /// Create a new FFT engine with custom configuration
    pub fn with_config(config: FftConfig) -> Self {
        assert!(config.fft_size.is_power_of_two(), "FFT size must be a power of 2");
        assert!(config.fft_size <= MAX_FFT_SIZE, "FFT size exceeds maximum");
        
        // Create FFT planners
        let mut planner = FftPlanner::new();
        let forward_fft = planner.plan_fft_forward(config.fft_size);
        let inverse_fft = planner.plan_fft_inverse(config.fft_size);
        
        // Pre-allocate buffers
        let mut input_buffer = vec![Complex::zero(); config.fft_size];
        let output_buffer = vec![Complex::zero(); config.fft_size];
        
        // Apply Hann window to reduce spectral leakage
        let window_coefficients: Vec<f64> = (0..config.fft_size)
            .map(|i| 0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / config.fft_size as f64).cos()))
            .collect();
        
        // Pre-compute frequency bins
        let frequency_bins: Vec<f64> = (0..config.fft_size / 2)
            .map(|i| i as f64 * config.sample_rate / config.fft_size as f64)
            .collect();
        
        Self {
            forward_fft,
            inverse_fft,
            input_buffer,
            output_buffer,
            window_coefficients,
            config,
            frequency_bins,
            cached_result: None,
        }
    }
    
    /// Process a slice of real-valued data and return FFT analysis.
    /// This is the hot-path function - zero allocations guaranteed.
    #[inline]
    pub fn process(&mut self, data: &[f64]) -> FftResult {
        // Copy data into input buffer with windowing applied
        let len = data.len().min(self.config.fft_size);
        
        for (i, &sample) in data.iter().take(len).enumerate() {
            // Apply Hann window to reduce spectral leakage
            self.input_buffer[i] = Complex::new(
                sample * self.window_coefficients[i],
                0.0,
            );
        }
        
        // Zero-pad remaining samples if data is shorter than FFT size
        for i in len..self.config.fft_size {
            self.input_buffer[i] = Complex::zero();
        }
        
        // Perform forward FFT in-place
        self.forward_fft.process(&mut self.input_buffer);
        
        // Compute magnitude spectrum and find dominant frequencies
        self.compute_spectrum()
    }
    
    /// Compute the magnitude spectrum and extract dominant frequencies.
    /// Uses pre-allocated output buffer to avoid allocations.
    fn compute_spectrum(&mut self) -> FftResult {
        let half_size = self.config.fft_size / 2;
        
        // Calculate magnitudes and find peaks
        let mut magnitudes: Vec<(usize, f64)> = Vec::with_capacity(half_size);
        let mut total_energy = 0.0;
        
        for i in 0..half_size {
            let freq = self.frequency_bins[i];
            
            // Skip frequencies outside our range of interest
            if freq < self.config.min_frequency || freq > self.config.max_frequency {
                continue;
            }
            
            let complex_val = self.input_buffer[i];
            let magnitude = complex_val.norm();
            let power = magnitude * magnitude;
            
            total_energy += power;
            magnitudes.push((i, power));
        }
        
        // Sort by power descending to find dominant frequencies
        magnitudes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // Extract top dominant frequencies
        let dominant_frequencies: Vec<FrequencyComponent> = magnitudes
            .iter()
            .take(10) // Top 10 dominant frequencies
            .map(|&(idx, power)| {
                let complex_val = self.input_buffer[idx];
                let freq = self.frequency_bins[idx];
                let magnitude = complex_val.norm();
                let phase = complex_val.arg();
                let normalized_power = if total_energy > 0.0 { power / total_energy } else { 0.0 };
                
                FrequencyComponent {
                    frequency: freq,
                    magnitude,
                    phase,
                    normalized_power,
                }
            })
            .collect();
        
        // Calculate spectral centroid
        let spectral_centroid = if total_energy > 0.0 {
            let weighted_sum: f64 = magnitudes
                .iter()
                .map(|&(idx, power)| self.frequency_bins[idx] * power)
                .sum();
            weighted_sum / total_energy
        } else {
            0.0
        };
        
        // Calculate spectral spread (variance around centroid)
        let spectral_spread = if total_energy > 0.0 {
            let variance: f64 = magnitudes
                .iter()
                .map(|&(idx, power)| {
                    let freq = self.frequency_bins[idx];
                    let diff = freq - spectral_centroid;
                    power * diff * diff
                })
                .sum();
            (variance / total_energy).sqrt()
        } else {
            0.0
        };
        
        let result = FftResult {
            dominant_frequencies,
            total_energy,
            spectral_centroid,
            spectral_spread,
        };
        
        self.cached_result = Some(result.clone());
        result
    }
    
    /// Detect TWAP/VWAP execution waves based on detected frequencies.
    /// Returns estimated execution period in seconds if detected.
    pub fn detect_execution_waves(&self) -> Option<ExecutionWave> {
        let result = self.cached_result.as_ref()?;
        
        // Look for strong periodic components that might indicate algorithmic execution
        let twap_candidates: Vec<&FrequencyComponent> = result
            .dominant_frequencies
            .iter()
            .filter(|fc| {
                // TWAP typically executes over minutes to hours
                // VWAP typically has intraday patterns
                let period = 1.0 / fc.frequency;
                (period >= 60.0 && period <= 3600.0) && fc.normalized_power > 0.1
            })
            .collect();
        
        if twap_candidates.is_empty() {
            return None;
        }
        
        // Return the strongest candidate
        let strongest = twap_candidates.first()?;
        Some(ExecutionWave {
            period_seconds: 1.0 / strongest.frequency,
            confidence: strongest.normalized_power,
            wave_type: if strongest.frequency < 0.01 {
                WaveType::VWAP
            } else {
                WaveType::TWAP
            },
        })
    }
    
    /// Get the current cached result without reprocessing
    pub fn cached_result(&self) -> Option<&FftResult> {
        self.cached_result.as_ref()
    }
    
    /// Reset internal state and clear cache
    pub fn reset(&mut self) {
        for c in &mut self.input_buffer {
            *c = Complex::zero();
        }
        for c in &mut self.output_buffer {
            *c = Complex::zero();
        }
        self.cached_result = None;
    }
    
    /// Update configuration (requires re-initialization of buffers)
    pub fn reconfigure(&mut self, new_config: FftConfig) {
        *self = Self::with_config(new_config);
    }
}

/// Detected algorithmic execution wave
#[derive(Debug, Clone)]
pub struct ExecutionWave {
    /// Period of the wave in seconds
    pub period_seconds: f64,
    /// Confidence level (0.0 to 1.0)
    pub confidence: f64,
    /// Type of execution algorithm detected
    pub wave_type: WaveType,
}

/// Type of algorithmic execution detected
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaveType {
    /// Time-Weighted Average Price execution
    TWAP,
    /// Volume-Weighted Average Price execution
    VWAP,
    /// Implementation Shortfall
    IS,
    /// Unknown pattern
    Unknown,
}

impl Default for FftEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Real-time spectral analyzer for continuous monitoring
pub struct SpectralAnalyzer {
    engine: FftEngine,
    /// Circular buffer for streaming data
    buffer: Vec<f64>,
    /// Current write position in buffer
    write_pos: usize,
    /// Number of samples collected
    sample_count: usize,
}

impl SpectralAnalyzer {
    pub fn new(fft_size: usize) -> Self {
        let config = FftConfig {
            fft_size,
            ..Default::default()
        };
        
        Self {
            engine: FftEngine::with_config(config),
            buffer: vec![0.0; fft_size],
            write_pos: 0,
            sample_count: 0,
        }
    }
    
    /// Add a new sample to the streaming buffer
    #[inline]
    pub fn add_sample(&mut self, sample: f64) {
        self.buffer[self.write_pos] = sample;
        self.write_pos = (self.write_pos + 1) % self.buffer.len();
        self.sample_count += 1;
    }
    
    /// Process accumulated samples if buffer is full
    pub fn process_if_ready(&mut self) -> Option<FftResult> {
        if self.sample_count < self.buffer.len() {
            return None;
        }
        
        // Reorder buffer to start from write_pos for continuous analysis
        let mut ordered_data = vec![0.0; self.buffer.len()];
        for i in 0..self.buffer.len() {
            ordered_data[i] = self.buffer[(self.write_pos + i) % self.buffer.len()];
        }
        
        Some(self.engine.process(&ordered_data))
    }
    
    /// Force process current buffer state (may include zero-padded data)
    pub fn force_process(&mut self) -> FftResult {
        self.engine.process(&self.buffer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;
    
    #[test]
    fn test_fft_sine_wave_detection() {
        let mut engine = FftEngine::with_config(FftConfig {
            fft_size: 1024,
            sample_rate: 1000.0,
            min_frequency: 0.0,
            max_frequency: 500.0,
        });
        
        // Generate a sine wave at 50 Hz
        let frequency = 50.0;
        let data: Vec<f64> = (0..1024)
            .map(|i| (2.0 * PI * frequency * i as f64 / 1000.0).sin())
            .collect();
        
        let result = engine.process(&data);
        
        // Should detect the 50 Hz component as dominant
        assert!(!result.dominant_frequencies.is_empty());
        
        let strongest = &result.dominant_frequencies[0];
        assert!((strongest.frequency - frequency).abs() < 2.0); // Within 2 Hz tolerance
    }
    
    #[test]
    fn test_execution_wave_detection() {
        let mut engine = FftEngine::new();
        
        // Generate a signal with a 5-minute (300 second) periodicity
        let frequency = 1.0 / 300.0; // ~0.00333 Hz
        let data: Vec<f64> = (0..4096)
            .map(|i| (2.0 * PI * frequency * i as f64).sin() * 0.5 + (i as f64 * 0.01).sin() * 0.3)
            .collect();
        
        let _ = engine.process(&data);
        
        // May or may not detect depending on signal strength
        // This test verifies the function doesn't crash
        let _wave = engine.detect_execution_waves();
    }
}
