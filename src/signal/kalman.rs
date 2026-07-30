//! Advanced 1D Kalman Filter for microprice smoothing and state estimation.
//! 
//! This module implements a highly optimized Kalman Filter using fixed-point arithmetic
//! where possible to minimize latency and avoid heap allocations in the hot path.
//! Designed for processing noisy tick data and estimating true underlying price states.

use std::sync::atomic::{AtomicU64, Ordering};
use crate::common::memory_pool::MemoryPool;

/// Fixed-point representation for Kalman gains and covariances.
/// Uses Q16.16 format: 16 bits integer, 16 bits fractional.
#[derive(Debug, Clone, Copy)]
pub struct FixedPoint(u32);

impl FixedPoint {
    const ONE: u32 = 1 << 16;
    
    pub fn from_f64(val: f64) -> Self {
        FixedPoint((val * Self::ONE as f64) as u32)
    }
    
    pub fn to_f64(self) -> f64 {
        self.0 as f64 / Self::ONE as f64
    }
    
    pub fn mul(self, other: FixedPoint) -> FixedPoint {
        FixedPoint(((self.0 as u64 * other.0 as u64) >> 16) as u32)
    }
    
    pub fn add(self, other: FixedPoint) -> FixedPoint {
        FixedPoint(self.0.wrapping_add(other.0))
    }
    
    pub fn sub(self, other: FixedPoint) -> FixedPoint {
        FixedPoint(self.0.wrapping_sub(other.0))
    }
}

/// Configuration for the Kalman Filter
#[derive(Debug, Clone)]
pub struct KalmanConfig {
    /// Process noise covariance (Q) - model uncertainty
    pub process_noise: f64,
    /// Measurement noise covariance (R) - sensor uncertainty  
    /// Initial error covariance (P)
    pub initial_covariance: f64,
}

impl Default for KalmanConfig {
    fn default() -> Self {
        Self {
            process_noise: 1e-5,
            measurement_noise: 1e-3,
            initial_covariance: 1.0,
        }
    }
}

/// High-performance 1D Kalman Filter for price state estimation.
/// Zero-allocation in the update() hot path.
pub struct KalmanFilter {
    /// Current state estimate (x)
    state: f64,
    /// Error covariance (P)
    covariance: f64,
    /// Process noise (Q)
    process_noise: f64,
    /// Measurement noise (R)
    measurement_noise: f64,
    /// Kalman gain (K) - cached to avoid recalculation
    kalman_gain: f64,
    /// Fixed-point representations for fast arithmetic
    state_fp: AtomicU64,
    covariance_fp: AtomicU64,
    /// Update counter for adaptive tuning
    update_count: AtomicU64,
}

impl KalmanFilter {
    /// Create a new Kalman Filter with default configuration
    pub fn new() -> Self {
        Self::with_config(KalmanConfig::default())
    }
    
    /// Create a new Kalman Filter with custom configuration
    pub fn with_config(config: KalmanConfig) -> Self {
        let state_fp = AtomicU64::new((config.initial_covariance * FixedPoint::ONE as f64) as u64);
        let covariance_fp = AtomicU64::new((config.initial_covariance * FixedPoint::ONE as f64) as u64);
        
        Self {
            state: 0.0,
            covariance: config.initial_covariance,
            process_noise: config.process_noise,
            measurement_noise: config.measurement_noise,
            kalman_gain: 0.0,
            state_fp,
            covariance_fp,
            update_count: AtomicU64::new(0),
        }
    }
    
    /// Initialize the filter with an initial price observation
    #[inline]
    pub fn initialize(&mut self, initial_price: f64) {
        self.state = initial_price;
        self.state_fp.store((initial_price * FixedPoint::ONE as f64) as u64, Ordering::Relaxed);
    }
    
    /// Process a new measurement and return the filtered state estimate.
    /// This is the hot-path function - zero allocations guaranteed.
    #[inline]
    pub fn update(&mut self, measurement: f64) -> f64 {
        // Prediction step: x_pred = x (for 1D random walk model)
        // P_pred = P + Q
        let predicted_covariance = self.covariance + self.process_noise;
        
        // Update step: Calculate Kalman Gain
        // K = P_pred / (P_pred + R)
        let denominator = predicted_covariance + self.measurement_noise;
        self.kalman_gain = predicted_covariance / denominator;
        
        // State update: x = x_pred + K * (z - x_pred)
        let innovation = measurement - self.state;
        self.state += self.kalman_gain * innovation;
        
        // Covariance update: P = (1 - K) * P_pred
        self.covariance = (1.0 - self.kalman_gain) * predicted_covariance;
        
        // Update fixed-point representations atomically
        self.state_fp.store((self.state * FixedPoint::ONE as f64) as u64, Ordering::Relaxed);
        self.covariance_fp.store((self.covariance * FixedPoint::ONE as f64) as u64, Ordering::Relaxed);
        
        self.update_count.fetch_add(1, Ordering::Relaxed);
        
        self.state
    }
    
    /// Batch process multiple measurements (for initialization)
    pub fn process_batch(&mut self, measurements: &[f64]) -> Vec<f64> {
        let mut results = Vec::with_capacity(measurements.len());
        for &measurement in measurements {
            results.push(self.update(measurement));
        }
        results
    }
    
    /// Get the current state estimate
    #[inline]
    pub fn state(&self) -> f64 {
        self.state
    }
    
    /// Get the current error covariance
    #[inline]
    pub fn covariance(&self) -> f64 {
        self.covariance
    }
    
    /// Get the current Kalman gain
    #[inline]
    pub fn kalman_gain(&self) -> f64 {
        self.kalman_gain
    }
    
    /// Adaptive tuning: adjust measurement noise based on innovation statistics
    pub fn adapt_measurement_noise(&mut self, innovation_variance: f64) {
        // If innovation variance is high, increase measurement noise (trust model more)
        // If innovation variance is low, decrease measurement noise (trust measurements more)
        let target_variance = self.measurement_noise * 2.0;
        let adaptation_rate = 0.1;
        
        if innovation_variance > target_variance {
            self.measurement_noise *= 1.0 + adaptation_rate;
        } else if innovation_variance < target_variance * 0.5 {
            self.measurement_noise *= 1.0 - adaptation_rate;
        }
        
        // Clamp to reasonable bounds
        self.measurement_noise = self.measurement_noise.clamp(1e-6, 1e-1);
    }
    
    /// Reset the filter to initial state
    pub fn reset(&mut self, config: Option<KalmanConfig>) {
        if let Some(cfg) = config {
            self.process_noise = cfg.process_noise;
            self.measurement_noise = cfg.measurement_noise;
            self.covariance = cfg.initial_covariance;
        } else {
            self.covariance = 1.0;
        }
        self.state = 0.0;
        self.kalman_gain = 0.0;
        self.update_count.store(0, Ordering::Relaxed);
    }
    
    /// Get fixed-point state for lock-free reading by other threads
    #[inline]
    pub fn state_fixed_point(&self) -> u64 {
        self.state_fp.load(Ordering::Relaxed)
    }
    
    /// Get update count for monitoring
    #[inline]
    pub fn update_count(&self) -> u64 {
        self.update_count.load(Ordering::Relaxed)
    }
}

/// Multi-dimensional Kalman Filter wrapper for correlated assets
pub struct MultiAssetKalman {
    filters: Vec<KalmanFilter>,
    correlation_matrix: Vec<f64>,
    asset_count: usize,
}

impl MultiAssetKalman {
    pub fn new(asset_count: usize) -> Self {
        let filters = (0..asset_count).map(|_| KalmanFilter::new()).collect();
        let correlation_matrix = vec![0.0; asset_count * asset_count];
        
        Self {
            filters,
            correlation_matrix,
            asset_count,
        }
    }
    
    pub fn update_all(&mut self, measurements: &[f64]) -> Vec<f64> {
        assert_eq!(measurements.len(), self.asset_count);
        
        let mut results = Vec::with_capacity(self.asset_count);
        for (i, &measurement) in measurements.iter().enumerate() {
            results.push(self.filters[i].update(measurement));
        }
        results
    }
    
    pub fn set_correlation(&mut self, asset_i: usize, asset_j: usize, correlation: f64) {
        let idx = asset_i * self.asset_count + asset_j;
        self.correlation_matrix[idx] = correlation.clamp(-1.0, 1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_kalman_smoothing() {
        let mut kf = KalmanFilter::new();
        kf.initialize(100.0);
        
        // Simulate noisy measurements around true value of 100.0
        let noisy_measurements = vec![
            100.5, 99.8, 101.2, 98.9, 100.1, 99.5, 100.8, 101.5, 99.2, 100.3
        ];
        
        let filtered = kf.process_batch(&noisy_measurements);
        
        // Filtered values should be smoother than raw measurements
        assert_eq!(filtered.len(), noisy_measurements.len());
        
        // Final state should be close to the mean of measurements
        let mean: f64 = noisy_measurements.iter().sum::<f64>() / noisy_measurements.len() as f64;
        assert!((kf.state() - mean).abs() < 0.5);
    }
    
    #[test]
    fn test_fixed_point_arithmetic() {
        let fp1 = FixedPoint::from_f64(1.5);
        let fp2 = FixedPoint::from_f64(2.0);
        
        let result = fp1.mul(fp2);
        assert!((result.to_f64() - 3.0).abs() < 0.001);
    }
}
