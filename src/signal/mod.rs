//! Signal processing module root.
//! 
//! Provides advanced digital signal processing capabilities including:
//! - Kalman Filters for state estimation
//! - FFT for frequency analysis and cycle detection
//! - Pluggable digital filter traits (Butterworth, Chebyshev)

pub mod kalman;
pub mod fourier;

pub use kalman::{KalmanFilter, KalmanConfig, FixedPoint, MultiAssetKalman};
pub use fourier::{
    FftEngine, FftConfig, FftResult, FrequencyComponent, 
    SpectralAnalyzer, ExecutionWave, WaveType,
};

use std::fmt::Debug;

/// Trait for pluggable digital filters
/// Implement this trait to add custom filter types (Butterworth, Chebyshev, etc.)
pub trait DigitalFilter: Send + Sync + Debug {
    /// Process a single sample and return filtered output
    fn process(&mut self, sample: f64) -> f64;
    
    /// Process a batch of samples
    fn process_batch(&mut self, samples: &[f64]) -> Vec<f64> {
        let mut results = Vec::with_capacity(samples.len());
        for &sample in samples {
            results.push(self.process(sample));
        }
        results
    }
    
    /// Reset filter state
    fn reset(&mut self);
    
    /// Get filter order (number of taps/stages)
    fn order(&self) -> usize;
    
    /// Get filter type name
    fn filter_type(&self) -> &'static str;
}

/// Low-pass filter configuration
#[derive(Debug, Clone)]
pub struct LowPassConfig {
    /// Cutoff frequency in Hz
    pub cutoff_frequency: f64,
    /// Sampling rate in Hz
    pub sample_rate: f64,
    /// Filter order
    pub order: usize,
}

impl LowPassConfig {
    /// Calculate normalized cutoff frequency (0 to 1, where 1 is Nyquist)
    pub fn normalized_cutoff(&self) -> f64 {
        let nyquist = self.sample_rate / 2.0;
        (self.cutoff_frequency / nyquist).clamp(0.0, 1.0)
    }
}

/// High-pass filter configuration
#[derive(Debug, Clone)]
pub struct HighPassConfig {
    /// Cutoff frequency in Hz
    pub cutoff_frequency: f64,
    /// Sampling rate in Hz
    pub sample_rate: f64,
    /// Filter order
    pub order: usize,
}

/// Band-pass filter configuration
#[derive(Debug, Clone)]
pub struct BandPassConfig {
    /// Lower cutoff frequency in Hz
    pub low_cutoff: f64,
    /// Upper cutoff frequency in Hz
    pub high_cutoff: f64,
    /// Sampling rate in Hz
    pub sample_rate: f64,
    /// Filter order
    pub order: usize,
}

/// Simple first-order IIR low-pass filter implementation
#[derive(Debug)]
pub struct SimpleLowPassFilter {
    /// Filter coefficient (alpha)
    alpha: f64,
    /// Previous output value
    previous_output: f64,
    /// Filter order (always 1 for this implementation)
    order: usize,
}

impl SimpleLowPassFilter {
    /// Create a new simple low-pass filter
    pub fn new(cutoff_frequency: f64, sample_rate: f64) -> Self {
        let dt = 1.0 / sample_rate;
        let rc = 1.0 / (2.0 * std::f64::consts::PI * cutoff_frequency);
        let alpha = dt / (rc + dt);
        
        Self {
            alpha,
            previous_output: 0.0,
            order: 1,
        }
    }
    
    /// Create from config
    pub fn from_config(config: LowPassConfig) -> Self {
        Self::new(config.cutoff_frequency, config.sample_rate)
    }
}

impl DigitalFilter for SimpleLowPassFilter {
    fn process(&mut self, sample: f64) -> f64 {
        let output = self.alpha * sample + (1.0 - self.alpha) * self.previous_output;
        self.previous_output = output;
        output
    }
    
    fn reset(&mut self) {
        self.previous_output = 0.0;
    }
    
    fn order(&self) -> usize {
        self.order
    }
    
    fn filter_type(&self) -> &'static str {
        "SimpleLowPass"
    }
}

/// Simple first-order IIR high-pass filter implementation
#[derive(Debug)]
pub struct SimpleHighPassFilter {
    /// Filter coefficient (alpha)
    alpha: f64,
    /// Previous input value
    previous_input: f64,
    /// Previous output value
    previous_output: f64,
    /// Filter order (always 1 for this implementation)
    order: usize,
}

impl SimpleHighPassFilter {
    /// Create a new simple high-pass filter
    pub fn new(cutoff_frequency: f64, sample_rate: f64) -> Self {
        let dt = 1.0 / sample_rate;
        let rc = 1.0 / (2.0 * std::f64::consts::PI * cutoff_frequency);
        let alpha = rc / (rc + dt);
        
        Self {
            alpha,
            previous_input: 0.0,
            previous_output: 0.0,
            order: 1,
        }
    }
    
    /// Create from config
    pub fn from_config(config: HighPassConfig) -> Self {
        Self::new(config.cutoff_frequency, config.sample_rate)
    }
}

impl DigitalFilter for SimpleHighPassFilter {
    fn process(&mut self, sample: f64) -> f64 {
        let output = self.alpha * (self.previous_output + sample - self.previous_input);
        self.previous_input = sample;
        self.previous_output = output;
        output
    }
    
    fn reset(&mut self) {
        self.previous_input = 0.0;
        self.previous_output = 0.0;
    }
    
    fn order(&self) -> usize {
        self.order
    }
    
    fn filter_type(&self) -> &'static str {
        "SimpleHighPass"
    }
}

/// Moving average filter (FIR)
#[derive(Debug)]
pub struct MovingAverageFilter {
    /// Buffer for storing recent samples
    buffer: Vec<f64>,
    /// Current write position
    write_pos: usize,
    /// Sum of all values in buffer
    sum: f64,
    /// Number of samples collected so far
    count: usize,
}

impl MovingAverageFilter {
    /// Create a new moving average filter with specified window size
    pub fn new(window_size: usize) -> Self {
        Self {
            buffer: vec![0.0; window_size],
            write_pos: 0,
            sum: 0.0,
            count: 0,
        }
    }
    
    /// Get the window size
    pub fn window_size(&self) -> usize {
        self.buffer.len()
    }
}

impl DigitalFilter for MovingAverageFilter {
    fn process(&mut self, sample: f64) -> f64 {
        let old_value = self.buffer[self.write_pos];
        self.buffer[self.write_pos] = sample;
        self.sum = self.sum - old_value + sample;
        self.write_pos = (self.write_pos + 1) % self.buffer.len();
        
        if self.count < self.buffer.len() {
            self.count += 1;
        }
        
        self.sum / self.count as f64
    }
    
    fn reset(&mut self) {
        for v in &mut self.buffer {
            *v = 0.0;
        }
        self.write_pos = 0;
        self.sum = 0.0;
        self.count = 0;
    }
    
    fn order(&self) -> usize {
        self.buffer.len()
    }
    
    fn filter_type(&self) -> &'static str {
        "MovingAverage"
    }
}

/// Exponential moving average filter
#[derive(Debug)]
pub struct ExponentialMovingAverageFilter {
    /// Smoothing factor (alpha)
    alpha: f64,
    /// Current EMA value
    ema: f64,
    /// Whether EMA has been initialized
    initialized: bool,
}

impl ExponentialMovingAverageFilter {
    /// Create a new EMA filter with specified span
    /// Span is the number of periods for equivalent simple MA
    pub fn new(span: usize) -> Self {
        let alpha = 2.0 / (span as f64 + 1.0);
        Self {
            alpha,
            ema: 0.0,
            initialized: false,
        }
    }
    
    /// Create from smoothing factor directly
    pub fn with_alpha(alpha: f64) -> Self {
        Self {
            alpha: alpha.clamp(0.0, 1.0),
            ema: 0.0,
            initialized: false,
        }
    }
}

impl DigitalFilter for ExponentialMovingAverageFilter {
    fn process(&mut self, sample: f64) -> f64 {
        if !self.initialized {
            self.ema = sample;
            self.initialized = true;
            return sample;
        }
        
        self.ema = self.alpha * sample + (1.0 - self.alpha) * self.ema;
        self.ema
    }
    
    fn reset(&mut self) {
        self.ema = 0.0;
        self.initialized = false;
    }
    
    fn order(&self) -> usize {
        // EMA is technically infinite order, but we return 1 for simplicity
        1
    }
    
    fn filter_type(&self) -> &'static str {
        "ExponentialMovingAverage"
    }
}

/// Signal processor that chains multiple filters together
#[derive(Debug)]
pub struct FilterChain {
    /// Ordered list of filters to apply
    filters: Vec<Box<dyn DigitalFilter>>,
}

impl FilterChain {
    /// Create a new empty filter chain
    pub fn new() -> Self {
        Self {
            filters: Vec::new(),
        }
    }
    
    /// Add a filter to the chain
    pub fn add_filter<F: DigitalFilter + 'static>(&mut self, filter: F) {
        self.filters.push(Box::new(filter));
    }
    
    /// Process a sample through all filters in sequence
    pub fn process(&mut self, sample: f64) -> f64 {
        let mut value = sample;
        for filter in &mut self.filters {
            value = filter.process(value);
        }
        value
    }
    
    /// Process a batch of samples
    pub fn process_batch(&mut self, samples: &[f64]) -> Vec<f64> {
        let mut results = Vec::with_capacity(samples.len());
        for &sample in samples {
            results.push(self.process(sample));
        }
        results
    }
    
    /// Reset all filters in the chain
    pub fn reset(&mut self) {
        for filter in &mut self.filters {
            filter.reset();
        }
    }
    
    /// Get the number of filters in the chain
    pub fn len(&self) -> usize {
        self.filters.len()
    }
    
    /// Check if the chain is empty
    pub fn is_empty(&self) -> bool {
        self.filters.is_empty()
    }
}

impl Default for FilterChain {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility functions for signal processing
pub mod utils {
    /// Calculate the mean of a slice
    pub fn mean(data: &[f64]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        data.iter().sum::<f64>() / data.len() as f64
    }
    
    /// Calculate the variance of a slice
    pub fn variance(data: &[f64]) -> f64 {
        if data.len() < 2 {
            return 0.0;
        }
        let m = mean(data);
        data.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (data.len() - 1) as f64
    }
    
    /// Calculate the standard deviation of a slice
    pub fn std_dev(data: &[f64]) -> f64 {
        variance(data).sqrt()
    }
    
    /// Calculate the z-score of a value given mean and std dev
    pub fn z_score(value: f64, mean: f64, std_dev: f64) -> f64 {
        if std_dev == 0.0 {
            return 0.0;
        }
        (value - mean) / std_dev
    }
    
    /// Normalize data to zero mean and unit variance
    pub fn normalize(data: &[f64]) -> Vec<f64> {
        let m = mean(data);
        let s = std_dev(data);
        
        if s == 0.0 {
            return vec![0.0; data.len()];
        }
        
        data.iter().map(|x| (x - m) / s).collect()
    }
    
    /// Detect zero crossings in a signal
    pub fn zero_crossings(data: &[f64]) -> Vec<usize> {
        let mut crossings = Vec::new();
        
        for i in 1..data.len() {
            if (data[i - 1] >= 0.0 && data[i] < 0.0) || (data[i - 1] < 0.0 && data[i] >= 0.0) {
                crossings.push(i);
            }
        }
        
        crossings
    }
    
    /// Find local peaks in a signal
    pub fn find_peaks(data: &[f64], min_distance: usize) -> Vec<usize> {
        let mut peaks = Vec::new();
        
        for i in 1..data.len().saturating_sub(1) {
            if data[i] > data[i - 1] && data[i] > data[i + 1] {
                // Check minimum distance from last peak
                if peaks.last().map_or(true, |&last| i - last >= min_distance) {
                    peaks.push(i);
                }
            }
        }
        
        peaks
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_lowpass_filter() {
        let mut filter = SimpleLowPassFilter::new(10.0, 100.0);
        
        // Step response
        let step_input = vec![1.0; 50];
        let output = filter.process_batch(&step_input);
        
        // Output should rise gradually towards 1.0
        assert!(output[0] < output[10]);
        assert!(output[10] < output[30]);
        assert!(output[30] > 0.9);
    }
    
    #[test]
    fn test_moving_average() {
        let mut filter = MovingAverageFilter::new(5);
        
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let output = filter.process_batch(&input);
        
        assert_eq!(output.len(), input.len());
        
        // After 5 samples, MA should be exact
        assert!((output[4] - 3.0).abs() < 0.001); // (1+2+3+4+5)/5 = 3
        assert!((output[5] - 4.0).abs() < 0.001); // (2+3+4+5+6)/5 = 4
        assert!((output[6] - 5.0).abs() < 0.001); // (3+4+5+6+7)/5 = 5
    }
    
    #[test]
    fn test_filter_chain() {
        let mut chain = FilterChain::new();
        chain.add_filter(MovingAverageFilter::new(3));
        chain.add_filter(SimpleLowPassFilter::new(10.0, 100.0));
        
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let output = chain.process_batch(&input);
        
        assert_eq!(output.len(), input.len());
    }
    
    #[test]
    fn test_utils_normalize() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let normalized = utils::normalize(&data);
        
        // Mean should be ~0
        let mean = utils::mean(&normalized);
        assert!(mean.abs() < 1e-10);
        
        // Std dev should be ~1
        let std = utils::std_dev(&normalized);
        assert!((std - 1.0).abs() < 1e-10);
    }
}
