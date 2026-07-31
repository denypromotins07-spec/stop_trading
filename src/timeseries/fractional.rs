//! Fractional Differentiation Engine for Stationary Feature Generation
//! 
//! Implements fractional differentiation preserving long-term memory
//! while achieving strict stationarity. Uses fixed-size ring buffers
//! to strictly respect the 6.5GB RAM ceiling.

use std::sync::atomic::{AtomicUsize, Ordering};

/// Maximum order of fractional differentiation
const MAX_ORDER: usize = 32;

/// Fixed buffer size for feature generation (power of 2)
const BUFFER_SIZE: usize = 2048;

/// Pre-computed binomial coefficients for fractional differentiation
/// C(d, k) = d * (d-1) * ... * (d-k+1) / k!
struct BinomialCache {
    coeffs: [[f64; MAX_ORDER]; 16], // Support d from 0.1 to 1.6 in steps
    d_values: [f64; 16],
}

impl BinomialCache {
    pub fn new() -> Self {
        let mut coeffs = [[0.0; MAX_ORDER]; 16];
        let mut d_values = [0.0; 16];
        
        // Pre-compute for d values from 0.1 to 1.6
        for (i, d) in (1..=16).map(|x| x as f64 * 0.1).enumerate() {
            d_values[i] = d;
            coeffs[i][0] = 1.0;
            
            for k in 1..MAX_ORDER {
                // C(d, k) = C(d, k-1) * (d - k + 1) / k
                coeffs[i][k] = coeffs[i][k - 1] * (d - k as f64 + 1.0) / k as f64;
            }
        }
        
        Self { coeffs, d_values }
    }
    
    #[inline]
    pub fn get_coeff(&self, d: f64, k: usize) -> f64 {
        if k >= MAX_ORDER {
            return 0.0;
        }
        
        // Find closest pre-computed d value
        let idx = ((d / 0.1).round() as usize).min(15).max(0);
        
        // Linear interpolation between adjacent d values if needed
        let d_low = self.d_values[idx];
        let d_high = if idx < 15 { self.d_values[idx + 1] } else { d_low + 0.1 };
        
        if (d - d_low).abs() < 1e-9 {
            self.coeffs[idx][k]
        } else if idx < 15 && (d_high - d).abs() < 1e-9 {
            self.coeffs[idx + 1][k]
        } else {
            // Interpolate
            let t = (d - d_low) / (d_high - d_low);
            self.coeffs[idx][k] * (1.0 - t) + self.coeffs[(idx + 1).min(15)][k] * t
        }
    }
}

/// Result of fractional differentiation
#[derive(Debug, Clone)]
pub struct FractionalDiffResult {
    /// Differentiated values
    pub values: [f64; BUFFER_SIZE],
    /// Number of valid values
    pub count: usize,
    /// Applied differentiation order
    pub order: f64,
    /// Stationarity metric (ADF-like statistic approximation)
    pub stationarity_score: f64,
}

/// Fractional differentiator with fixed ring buffer
pub struct FractionalDifferentiator {
    binomial_cache: BinomialCache,
    price_buffer: [f64; BUFFER_SIZE],
    diff_buffer: [f64; BUFFER_SIZE],
    write_index: AtomicUsize,
    last_price: f64,
}

impl FractionalDifferentiator {
    /// Create a new fractional differentiator
    pub fn new() -> Self {
        Self {
            binomial_cache: BinomialCache::new(),
            price_buffer: [0.0; BUFFER_SIZE],
            diff_buffer: [0.0; BUFFER_SIZE],
            write_index: AtomicUsize::new(0),
            last_price: 0.0,
        }
    }
    
    /// Push a new price tick to the buffer
    #[inline]
    pub fn push_price(&mut self, price: f64) {
        let idx = self.write_index.load(Ordering::Relaxed);
        self.price_buffer[idx % BUFFER_SIZE] = price;
        self.last_price = price;
        self.write_index.store(idx + 1, Ordering::Release);
    }
    
    /// Compute fractional differentiation with specified order d
    /// d typically ranges from 0.3 to 1.0 for financial time series
    pub fn differentiate(&self, d: f64) -> FractionalDiffResult {
        let idx = self.write_index.load(Ordering::Acquire);
        let valid_len = idx.min(BUFFER_SIZE);
        
        if valid_len < MAX_ORDER {
            return FractionalDiffResult {
                values: [0.0; BUFFER_SIZE],
                count: 0,
                order: d,
                stationarity_score: 0.0,
            };
        }
        
        let mut result = FractionalDiffResult {
            values: [0.0; BUFFER_SIZE],
            count: 0,
            order: d,
            stationarity_score: 0.0,
        };
        
        // Compute fractional difference for each point
        for i in (MAX_ORDER - 1)..valid_len {
            let mut diff_val = 0.0;
            
            // Apply fractional differentiation formula:
            // (1 - B)^d * x_t = sum_{k=0}^{inf} (-1)^k * C(d, k) * x_{t-k}
            for k in 0..MAX_ORDER.min(i + 1) {
                let coeff = self.binomial_cache.get_coeff(d, k);
                let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
                
                let price_idx = (i - k + BUFFER_SIZE) % BUFFER_SIZE;
                diff_val += sign * coeff * self.price_buffer[price_idx];
            }
            
            result.values[result.count] = diff_val;
            result.count += 1;
        }
        
        // Compute stationarity score (variance ratio test approximation)
        result.stationarity_score = self.compute_stationarity_score(&result.values, result.count);
        
        result
    }
    
    /// Compute stationarity score using variance ratio
    fn compute_stationarity_score(&self, values: &[f64], count: usize) -> f64 {
        if count < 10 {
            return 0.0;
        }
        
        // Compute mean
        let mean: f64 = values[..count].iter().sum::<f64>() / count as f64;
        
        // Compute variance
        let variance: f64 = values[..count]
            .iter()
            .map(|&x| (x - mean) * (x - mean))
            .sum::<f64>()
            / count as f64;
        
        if variance < 1e-12 {
            return 1.0; // Perfectly stationary
        }
        
        // Compute first-difference variance (should be similar for stationary series)
        let mut diff_variance: f64 = 0.0;
        let mut diff_count = 0;
        
        for i in 1..count {
            let diff = values[i] - values[i - 1];
            diff_variance += diff * diff;
            diff_count += 1;
        }
        
        if diff_count > 0 {
            diff_variance /= diff_count as f64;
        }
        
        // Variance ratio: close to 1.0 indicates stationarity
        let ratio = if variance > 1e-12 {
            diff_variance / (2.0 * variance)
        } else {
            1.0
        };
        
        // Score based on how close ratio is to 0.5 (ideal for random walk)
        // or 1.0 (ideal for stationary)
        (1.0 - (ratio - 0.5).abs() * 2.0).max(0.0)
    }
    
    /// Get the last differentiated value without full recomputation
    pub fn get_last_diff(&self, d: f64) -> f64 {
        let idx = self.write_index.load(Ordering::Acquire);
        
        if idx < MAX_ORDER {
            return 0.0;
        }
        
        let mut diff_val = 0.0;
        let current_idx = (idx - 1 + BUFFER_SIZE) % BUFFER_SIZE;
        
        for k in 0..MAX_ORDER {
            let coeff = self.binomial_cache.get_coeff(d, k);
            let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
            
            let price_idx = (current_idx.wrapping_sub(k) + BUFFER_SIZE) % BUFFER_SIZE;
            diff_val += sign * coeff * self.price_buffer[price_idx];
        }
        
        diff_val
    }
    
    /// Compute rolling fractional standard deviation
    pub fn rolling_frac_std(&self, d: f64, window: usize) -> f64 {
        let idx = self.write_index.load(Ordering::Acquire);
        
        if idx < window + MAX_ORDER {
            return 0.0;
        }
        
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        let mut count = 0;
        
        for i in 0..window.min(BUFFER_SIZE) {
            let start_idx = (idx.wrapping_sub(i + 1) + BUFFER_SIZE) % BUFFER_SIZE;
            
            if start_idx < MAX_ORDER {
                continue;
            }
            
            let mut diff_val = 0.0;
            for k in 0..MAX_ORDER {
                let coeff = self.binomial_cache.get_coeff(d, k);
                let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
                
                let price_idx = (start_idx.wrapping_sub(k) + BUFFER_SIZE) % BUFFER_SIZE;
                diff_val += sign * coeff * self.price_buffer[price_idx];
            }
            
            sum += diff_val;
            sum_sq += diff_val * diff_val;
            count += 1;
        }
        
        if count < 2 {
            return 0.0;
        }
        
        let mean = sum / count as f64;
        let variance = sum_sq / count as f64 - mean * mean;
        
        if variance > 0.0 {
            variance.sqrt()
        } else {
            0.0
        }
    }
    
    /// Optimal d finder using grid search for maximum stationarity
    pub fn find_optimal_d(&self) -> f64 {
        let mut best_d = 0.5;
        let mut best_score = 0.0;
        
        // Grid search over common d values
        for d in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] {
            let result = self.differentiate(d);
            
            if result.stationarity_score > best_score {
                best_score = result.stationarity_score;
                best_d = d;
            }
        }
        
        best_d
    }
    
    /// Get memory persistence estimate (Hurst exponent approximation)
    pub fn estimate_hurst(&self) -> f64 {
        // Use the relationship: d = H - 0.5 for fractional Brownian motion
        // where d is the optimal fractional differentiation order
        let optimal_d = self.find_optimal_d();
        optimal_d + 0.5
    }
}

impl Default for FractionalDifferentiator {
    fn default() -> Self {
        Self::new()
    }
}

/// Multi-order fractional differentiator for feature ensemble
pub struct MultiOrderFractionalDifferentiator {
    base: FractionalDifferentiator,
    orders: Vec<f64>,
}

impl MultiOrderFractionalDifferentiator {
    pub fn new(orders: Vec<f64>) -> Self {
        Self {
            base: FractionalDifferentiator::new(),
            orders,
        }
    }
    
    pub fn push_price(&mut self, price: f64) {
        self.base.push_price(price);
    }
    
    /// Get features at multiple differentiation orders
    pub fn get_features(&self) -> Vec<FractionalDiffResult> {
        self.orders
            .iter()
            .map(|&d| self.base.differentiate(d))
            .collect()
    }
    
    /// Get concatenated feature vector for ML model
    pub fn get_feature_vector(&self, max_len: usize) -> Vec<f64> {
        let mut features = Vec::with_capacity(self.orders.len() * max_len);
        
        for &d in &self.orders {
            let result = self.base.differentiate(d);
            let len = result.count.min(max_len);
            features.extend_from_slice(&result.values[..len]);
        }
        
        features
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_fractional_differentiation() {
        let mut diff = FractionalDifferentiator::new();
        
        // Generate synthetic price series with trend
        for i in 0..BUFFER_SIZE {
            let price = 100.0 + (i as f64 * 0.1) + (i as f64 * 0.5).sin() * 2.0;
            diff.push_price(price);
        }
        
        let result = diff.differentiate(0.5);
        assert!(result.count > 0);
        assert!(result.stationarity_score >= 0.0);
    }
    
    #[test]
    fn test_stationarity_improvement() {
        let mut diff = FractionalDifferentiator::new();
        
        // Generate trending price series
        for i in 0..BUFFER_SIZE {
            let price = 100.0 + i as f64 * 0.5;
            diff.push_price(price);
        }
        
        // Raw prices should have low stationarity
        let raw_result = diff.differentiate(0.0);
        
        // Fractionally differentiated should have higher stationarity
        let frac_result = diff.differentiate(0.5);
        
        assert!(frac_result.stationarity_score >= raw_result.stationarity_score);
    }
    
    #[test]
    fn test_optimal_d_finder() {
        let mut diff = FractionalDifferentiator::new();
        
        for i in 0..BUFFER_SIZE {
            let price = 100.0 + (i as f64 * 0.3).sin() * 5.0;
            diff.push_price(price);
        }
        
        let optimal_d = diff.find_optimal_d();
        assert!(optimal_d >= 0.3 && optimal_d <= 1.0);
    }
    
    #[test]
    fn test_multi_order_features() {
        let mut multi_diff = MultiOrderFractionalDifferentiator::new(vec![0.3, 0.5, 0.7, 1.0]);
        
        for i in 0..BUFFER_SIZE {
            let price = 100.0 + (i as f64 * 0.1).sin();
            multi_diff.push_price(price);
        }
        
        let features = multi_diff.get_features();
        assert_eq!(features.len(), 4);
        
        let feature_vec = multi_diff.get_feature_vector(100);
        assert!(feature_vec.len() >= 300); // 4 orders * ~100 values each
    }
    
    #[test]
    fn test_hurst_estimation() {
        let mut diff = FractionalDifferentiator::new();
        
        // Generate random walk-like series
        let mut price = 100.0;
        for i in 0..BUFFER_SIZE {
            price += (i as f64 * 0.1).cos() * 0.5;
            diff.push_price(price);
        }
        
        let hurst = diff.estimate_hurst();
        // Hurst should be between 0 and 1
        assert!(hurst >= 0.0 && hurst <= 1.0);
    }
}
