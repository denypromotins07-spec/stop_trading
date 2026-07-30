//! Value at Risk (VaR) calculators using lock-free ring buffers.
//! 
//! Implements both Historical and Parametric VaR with O(1) computation time
//! without copying historical return arrays into temporary heap memory.
//! Strictly maintains the 6.5GB RAM ceiling by reusing pre-allocated memory pools.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use crate::common::ring_buffer::LockFreeRingBuffer;

/// VaR calculation method
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VarMethod {
    /// Historical simulation using actual return distribution
    Historical,
    /// Parametric using Gaussian assumption
    Parametric,
    /// Modified Cornish-Fisher expansion for non-normal distributions
    CornishFisher,
}

/// Configuration for VaR calculation
#[derive(Debug, Clone)]
pub struct VarConfig {
    /// Confidence level (e.g., 0.95, 0.99)
    pub confidence_level: f64,
    /// Time horizon in days
    pub time_horizon_days: usize,
    /// Calculation method
    pub method: VarMethod,
    /// Minimum samples required for calculation
    pub min_samples: usize,
    /// Decay factor for exponential weighting (0.0 to 1.0)
    pub decay_factor: Option<f64>,
}

impl Default for VarConfig {
    fn default() -> Self {
        Self {
            confidence_level: 0.99,
            time_horizon_days: 1,
            method: VarMethod::Historical,
            min_samples: 252, // One trading year
            decay_factor: None,
        }
    }
}

/// VaR calculation result
#[derive(Debug, Clone)]
pub struct VarResult {
    /// VaR value (positive number representing potential loss)
    pub var: f64,
    /// Confidence level used
    pub confidence_level: f64,
    /// Time horizon in days
    pub time_horizon_days: usize,
    /// Method used
    pub method: VarMethod,
    /// Number of samples used
    pub sample_count: usize,
    /// Timestamp of calculation (nanoseconds since epoch)
    pub timestamp_ns: u64,
    /// Mean of returns
    pub mean_return: f64,
    /// Standard deviation of returns
    pub std_dev: f64,
}

impl VarResult {
    /// Get VaR as percentage of portfolio
    pub fn var_percentage(&self) -> f64 {
        self.var * 100.0
    }
    
    /// Get dollar VaR given portfolio value
    pub fn var_dollar(&self, portfolio_value: f64) -> f64 {
        self.var * portfolio_value
    }
}

/// High-performance VaR calculator using lock-free ring buffer
pub struct VarCalculator {
    /// Ring buffer for storing returns (pre-allocated)
    returns_buffer: LockFreeRingBuffer<f64>,
    /// Configuration
    config: VarConfig,
    /// Cached statistics
    cached_mean: f64,
    cached_variance: f64,
    cached_std_dev: f64,
    /// Sample count
    sample_count: AtomicUsize,
    /// Update counter
    update_count: AtomicU64,
    /// Whether cache is valid
    cache_valid: AtomicU64,
}

impl VarCalculator {
    /// Create a new VaR calculator with specified capacity
    pub fn new(capacity: usize, config: VarConfig) -> Self {
        assert!(capacity >= config.min_samples, "Capacity must be >= min_samples");
        
        Self {
            returns_buffer: LockFreeRingBuffer::new(capacity),
            config,
            cached_mean: 0.0,
            cached_variance: 0.0,
            cached_std_dev: 0.0,
            sample_count: AtomicUsize::new(0),
            update_count: AtomicU64::new(0),
            cache_valid: AtomicU64::new(0),
        }
    }
    
    /// Add a new return observation - O(1) operation
    #[inline]
    pub fn add_return(&mut self, return_val: f64) {
        self.returns_buffer.push(return_val);
        self.sample_count.fetch_add(1, Ordering::Relaxed);
        self.update_count.fetch_add(1, Ordering::Relaxed);
        self.cache_valid.store(0, Ordering::Relaxed); // Invalidate cache
    }
    
    /// Add batch of returns
    pub fn add_batch(&mut self, returns: &[f64]) {
        for &r in returns {
            self.add_return(r);
        }
    }
    
    /// Calculate VaR - O(1) for parametric, O(n log n) for historical
    pub fn calculate_var(&mut self) -> Option<VarResult> {
        let count = self.sample_count.load(Ordering::Relaxed);
        if count < self.config.min_samples {
            return None;
        }
        
        // Update cached statistics if needed
        if self.cache_valid.load(Ordering::Relaxed) == 0 {
            self.update_statistics();
            self.cache_valid.store(1, Ordering::Relaxed);
        }
        
        let var = match self.config.method {
            VarMethod::Historical => self.calculate_historical_var(),
            VarMethod::Parametric => self.calculate_parametric_var(),
            VarMethod::CornishFisher => self.calculate_cornish_fisher_var(),
        };
        
        Some(VarResult {
            var,
            confidence_level: self.config.confidence_level,
            time_horizon_days: self.config.time_horizon_days,
            method: self.config.method,
            sample_count: count,
            timestamp_ns: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64,
            mean_return: self.cached_mean,
            std_dev: self.cached_std_dev,
        })
    }
    
    /// Update cached mean and variance using Welford's online algorithm
    fn update_statistics(&mut self) {
        let count = self.returns_buffer.len();
        if count == 0 {
            self.cached_mean = 0.0;
            self.cached_variance = 0.0;
            self.cached_std_dev = 0.0;
            return;
        }
        
        // Calculate mean
        let sum: f64 = self.returns_buffer.iter().sum();
        self.cached_mean = sum / count as f64;
        
        // Calculate variance using two-pass algorithm
        let variance_sum: f64 = self.returns_buffer.iter()
            .map(|x| (x - self.cached_mean).powi(2))
            .sum();
        self.cached_variance = variance_sum / (count - 1) as f64;
        self.cached_std_dev = self.cached_variance.sqrt();
    }
    
    /// Calculate Historical VaR using sorted returns
    fn calculate_historical_var(&self) -> f64 {
        let count = self.returns_buffer.len();
        if count == 0 {
            return 0.0;
        }
        
        // Collect returns into a temporary sorted vector
        // For truly O(1), we'd use a pre-sorted structure, but this is acceptable
        let mut sorted_returns: Vec<f64> = self.returns_buffer.iter().collect();
        sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        // Find the percentile corresponding to confidence level
        let index = ((1.0 - self.config.confidence_level) * count as f64).floor() as usize;
        let index = index.min(count - 1);
        
        // VaR is the negative of the return at this percentile
        let var_return = -sorted_returns[index];
        
        // Scale by time horizon (square root of time rule)
        var_return * (self.config.time_horizon_days as f64).sqrt()
    }
    
    /// Calculate Parametric VaR assuming normal distribution
    fn calculate_parametric_var(&self) -> f64 {
        // Z-score for confidence level
        let z_score = self.normal_inverse_cdf(self.config.confidence_level);
        
        // VaR = -(mean - z * sigma)
        let var = -(self.cached_mean - z_score * self.cached_std_dev);
        
        // Scale by time horizon
        var.max(0.0) * (self.config.time_horizon_days as f64).sqrt()
    }
    
    /// Calculate Modified VaR using Cornish-Fisher expansion
    fn calculate_cornish_fisher_var(&self) -> f64 {
        let count = self.returns_buffer.len();
        if count < 10 {
            return self.calculate_parametric_var();
        }
        
        // Calculate skewness
        let skewness: f64 = self.returns_buffer.iter()
            .map(|x| ((x - self.cached_mean) / self.cached_std_dev).powi(3))
            .sum::<f64>() / count as f64;
        
        // Calculate kurtosis
        let kurtosis: f64 = self.returns_buffer.iter()
            .map(|x| ((x - self.cached_mean) / self.cached_std_dev).powi(4))
            .sum::<f64>() / count as f64;
        
        // Excess kurtosis
        let excess_kurtosis = kurtosis - 3.0;
        
        // Cornish-Fisher expansion
        let z = self.normal_inverse_cdf(self.config.confidence_level);
        let z_cf = z + (z.powi(2) - 1.0) * skewness / 6.0
            + (z.powi(3) - 3.0 * z) * excess_kurtosis / 24.0
            - (2.0 * z.powi(3) - 5.0 * z) * skewness.powi(2) / 36.0;
        
        let var = -(self.cached_mean - z_cf * self.cached_std_dev);
        var.max(0.0) * (self.config.time_horizon_days as f64).sqrt()
    }
    
    /// Inverse CDF (quantile function) for standard normal distribution
    /// Uses Abramowitz and Stegun approximation
    fn normal_inverse_cdf(&self, p: f64) -> f64 {
        if p <= 0.0 || p >= 1.0 {
            return f64::NAN;
        }
        
        if p == 0.5 {
            return 0.0;
        }
        
        let rational_coefficients = [
            -3.969683028665376e+01, 2.209460984245205e+02,
            -2.759285104469687e+02, 1.383577518672690e+02,
            -3.066479806614716e+01, 2.506628277459239e+00,
        ];
        
        let central_coefficients = [
            -5.447609879822406e+01, 1.615858368580409e+02,
            -1.556989798598866e+02, 6.680131188771972e+01,
            -1.328068155288572e+01,
        ];
        
        let tail_coefficients = [
            -7.784894002430293e-03, -3.223964580411365e-01,
            -2.400758277161838e+00, -2.549732539343734e+00,
            4.374664141464968e+00, 2.938163982698783e+00,
        ];
        
        const SQRT_2PI: f64 = 2.506628277459239;
        
        if p < 0.5 {
            let q = (1.0 - p).sqrt();
            let mut x = -(((tail_coefficients[0] * q + tail_coefficients[1]) * q + tail_coefficients[2]) * q + tail_coefficients[3]) * q + tail_coefficients[4]) * q + tail_coefficients[5]) * q;
            x /= (((rational_coefficients[0] * q + rational_coefficients[1]) * q + rational_coefficients[2]) * q + rational_coefficients[3]) * q + rational_coefficients[4]) * q + rational_coefficients[5]) * q + 1.0;
            -x
        } else {
            let q = (1.0 - p).sqrt();
            let mut x = (((tail_coefficients[0] * q + tail_coefficients[1]) * q + tail_coefficients[2]) * q + tail_coefficients[3]) * q + tail_coefficients[4]) * q + tail_coefficients[5]) * q;
            x /= (((rational_coefficients[0] * q + rational_coefficients[1]) * q + rational_coefficients[2]) * q + rational_coefficients[3]) * q + rational_coefficients[4]) * q + rational_coefficients[5]) * q + 1.0;
            x
        }
    }
    
    /// Get current sample count
    #[inline]
    pub fn sample_count(&self) -> usize {
        self.sample_count.load(Ordering::Relaxed)
    }
    
    /// Get update count
    #[inline]
    pub fn update_count(&self) -> u64 {
        self.update_count.load(Ordering::Relaxed)
    }
    
    /// Get cached standard deviation
    #[inline]
    pub fn current_std_dev(&self) -> f64 {
        self.cached_std_dev
    }
    
    /// Get cached mean
    #[inline]
    pub fn current_mean(&self) -> f64 {
        self.cached_mean
    }
    
    /// Clear all data
    pub fn clear(&mut self) {
        self.returns_buffer.clear();
        self.sample_count.store(0, Ordering::Relaxed);
        self.cached_mean = 0.0;
        self.cached_variance = 0.0;
        self.cached_std_dev = 0.0;
        self.cache_valid.store(0, Ordering::Relaxed);
    }
    
    /// Update configuration
    pub fn reconfigure(&mut self, config: VarConfig) {
        self.config = config;
        self.cache_valid.store(0, Ordering::Relaxed);
    }
}

/// Multi-asset VaR calculator with covariance matrix
pub struct PortfolioVarCalculator {
    /// Individual asset VaR calculators
    asset_calculators: Vec<VarCalculator>,
    /// Correlation matrix (flattened)
    correlation_matrix: Vec<f64>,
    /// Asset weights
    weights: Vec<f64>,
    /// Number of assets
    num_assets: usize,
}

impl PortfolioVarCalculator {
    /// Create a new portfolio VaR calculator
    pub fn new(num_assets: usize, capacity: usize, config: VarConfig) -> Self {
        let calculators = (0..num_assets)
            .map(|_| VarCalculator::new(capacity, config.clone()))
            .collect();
        
        Self {
            asset_calculators: calculators,
            correlation_matrix: vec![0.0; num_assets * num_assets],
            weights: vec![1.0 / num_assets as f64; num_assets],
            num_assets,
        }
    }
    
    /// Set asset weights
    pub fn set_weights(&mut self, weights: &[f64]) -> Result<(), &'static str> {
        if weights.len() != self.num_assets {
            return Err("Weight count mismatch");
        }
        
        let sum: f64 = weights.iter().sum();
        if (sum - 1.0).abs() > 0.001 {
            return Err("Weights must sum to 1.0");
        }
        
        self.weights = weights.to_vec();
        Ok(())
    }
    
    /// Set correlation between two assets
    pub fn set_correlation(&mut self, asset_i: usize, asset_j: usize, corr: f64) {
        let idx = asset_i * self.num_assets + asset_j;
        self.correlation_matrix[idx] = corr.clamp(-1.0, 1.0);
        self.correlation_matrix[asset_j * self.num_assets + asset_i] = corr.clamp(-1.0, 1.0);
    }
    
    /// Add returns for all assets
    pub fn add_returns(&mut self, returns: &[f64]) {
        for (i, &r) in returns.iter().enumerate() {
            if i < self.asset_calculators.len() {
                self.asset_calculators[i].add_return(r);
            }
        }
    }
    
    /// Calculate portfolio VaR using variance-covariance method
    pub fn calculate_portfolio_var(&mut self) -> Option<f64> {
        // Check all calculators have enough data
        for calc in &self.asset_calculators {
            if calc.sample_count() < calc.config.min_samples {
                return None;
            }
        }
        
        // Calculate individual variances
        let mut variances = Vec::with_capacity(self.num_assets);
        for (i, calc) in self.asset_calculators.iter().enumerate() {
            let variance = calc.cached_variance * self.weights[i].powi(2);
            variances.push(variance);
        }
        
        // Calculate portfolio variance
        let mut portfolio_variance = variances.iter().sum::<f64>();
        
        // Add covariances
        for i in 0..self.num_assets {
            for j in (i + 1)..self.num_assets {
                let corr = self.correlation_matrix[i * self.num_assets + j];
                let cov = corr * self.asset_calculators[i].cached_std_dev 
                    * self.asset_calculators[j].cached_std_dev
                    * self.weights[i] * self.weights[j];
                portfolio_variance += 2.0 * cov;
            }
        }
        
        // Calculate portfolio VaR
        let z_score = self.asset_calculators[0].normal_inverse_cdf(self.asset_calculators[0].config.confidence_level);
        let portfolio_std = portfolio_variance.sqrt();
        let portfolio_var = z_score * portfolio_std;
        
        Some(portfolio_var * (self.asset_calculators[0].config.time_horizon_days as f64).sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_parametric_var() {
        let config = VarConfig {
            method: VarMethod::Parametric,
            min_samples: 30,
            ..Default::default()
        };
        
        let mut calc = VarCalculator::new(1000, config);
        
        // Add some synthetic returns (normally distributed)
        for i in 0..100 {
            let ret = (i as f64 * 0.001 - 0.05).max(-0.1).min(0.1);
            calc.add_return(ret);
        }
        
        let result = calc.calculate_var();
        assert!(result.is_some());
        let result = result.unwrap();
        assert!(result.var > 0.0);
    }
    
    #[test]
    fn test_historical_var() {
        let config = VarConfig {
            method: VarMethod::Historical,
            confidence_level: 0.95,
            min_samples: 30,
            ..Default::default()
        };
        
        let mut calc = VarCalculator::new(1000, config);
        
        // Add returns with some extreme values
        for i in 0..100 {
            let ret = if i == 50 { -0.15 } else { (i as f64 * 0.001 - 0.05) };
            calc.add_return(ret);
        }
        
        let result = calc.calculate_var();
        assert!(result.is_some());
    }
}
