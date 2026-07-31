//! ARFIMA (Auto-Regressive Fractionally Integrated Moving Average) Model
//!
//! Implements fractional differencing for long-memory time series modeling
//! using fixed-size ring buffers to respect memory constraints.

use std::collections::VecDeque;

/// ARFIMA(p, d, q) model with fractional integration
/// 
/// The fractional differencing parameter d allows modeling of:
/// - d = 0: stationary process
/// - 0 < d < 0.5: stationary with long memory
/// - 0.5 <= d < 1: non-stationary but mean-reverting
/// - d >= 1: requires differencing
pub struct ARFIMA {
    /// AR order
    ar_order: usize,
    /// Fractional integration order (d)
    diff_order: f64,
    /// MA order
    ma_order: usize,
    
    /// AR coefficients
    ar_coeffs: Vec<f64>,
    /// MA coefficients
    ma_coeffs: Vec<f64>,
    
    /// Fixed-size ring buffer for past values (for fractional differencing)
    /// Size is bounded to respect memory constraints
    y_buffer: Vec<f64>,
    y_buffer_idx: usize,
    y_buffer_len: usize,
    
    /// Ring buffer for differenced series
    diff_buffer: Vec<f64>,
    diff_buffer_idx: usize,
    
    /// Residual buffer for MA component
    residual_buffer: Vec<f64>,
    residual_idx: usize,
    
    /// Pre-computed binomial coefficients for fractional differencing
    /// pi_j = (-1)^j * Gamma(d+1) / (Gamma(j+1) * Gamma(d-j+1))
    binomial_coeffs: Vec<f64>,
    
    /// Maximum lag for fractional differencing approximation
    max_diff_lag: usize,
    
    /// Number of observations processed
    n_obs: u64,
    
    /// Last value for differencing
    last_y: f64,
    
    /// Running mean for centering
    running_mean: f64,
    running_mean_count: u64,
}

impl ARFIMA {
    /// Create a new ARFIMA(p, d, q) model
    /// 
    /// # Arguments
    /// * `ar_order` - AR component order (p)
    /// * `diff_order` - Fractional differencing order (d), typically in [0, 0.5]
    /// * `ma_order` - MA component order (q)
    /// * `max_diff_lag` - Maximum lag for fractional differencing (memory bound)
    pub fn new(ar_order: usize, diff_order: f64, ma_order: usize, max_diff_lag: usize) -> Self {
        // Cap max_diff_lag to respect memory constraints (6.5GB total system limit)
        let max_diff_lag = max_diff_lag.min(2048);
        
        // Pre-compute binomial coefficients for fractional differencing
        let binomial_coeffs = Self::compute_binomial_coeffs(diff_order, max_diff_lag);
        
        ARFIMA {
            ar_order,
            diff_order,
            ma_order,
            ar_coeffs: vec![0.0; ar_order.max(1)],
            ma_coeffs: vec![0.0; ma_order.max(1)],
            y_buffer: vec![0.0; max_diff_lag + 10],
            y_buffer_idx: 0,
            y_buffer_len: 0,
            diff_buffer: vec![0.0; (ar_order.max(1) + ma_order.max(1) + 10)],
            diff_buffer_idx: 0,
            residual_buffer: vec![0.0; (ma_order.max(1) + 10)],
            residual_idx: 0,
            binomial_coeffs,
            max_diff_lag,
            n_obs: 0,
            last_y: 0.0,
            running_mean: 0.0,
            running_mean_count: 0,
        }
    }

    /// Compute binomial coefficients for fractional differencing
    /// pi_j = (-d)(1-d)(2-d)...(j-1-d) / j!
    fn compute_binomial_coeffs(d: f64, max_lag: usize) -> Vec<f64> {
        let mut coeffs = Vec::with_capacity(max_lag);
        
        // pi_0 = 1
        coeffs.push(1.0);
        
        // Recursive computation: pi_j = pi_{j-1} * (j - 1 - d) / j
        for j in 1..max_lag {
            let prev = *coeffs.last().unwrap();
            let coeff = prev * (j as f64 - 1.0 - d) / (j as f64);
            coeffs.push(coeff);
        }
        
        coeffs
    }

    /// Apply fractional differencing to get the stationary series
    /// (1 - L)^d * y_t = sum_{j=0}^{inf} pi_j * y_{t-j}
    #[inline]
    fn fractional_difference(&self, current_y: f64) -> f64 {
        let mut result = current_y;
        
        // Sum over available lags (bounded by max_diff_lag and available data)
        let available_lags = self.y_buffer_len.min(self.max_diff_lag);
        
        for j in 1..available_lags {
            let idx = (self.y_buffer_idx + self.y_buffer.len() - j) % self.y_buffer.len();
            let y_lag = self.y_buffer[idx];
            result += self.binomial_coeffs[j] * y_lag;
        }
        
        result
    }

    /// Update model with new observation
    /// Returns the one-step-ahead prediction error (innovation)
    #[inline]
    pub fn update(&mut self, y_t: f64) -> f64 {
        // Update running mean for potential centering
        self.running_mean_count += 1;
        let delta = y_t - self.running_mean;
        self.running_mean += delta / self.running_mean_count as f64;
        
        // Apply fractional differencing
        let y_diff = self.fractional_difference(y_t);
        
        // Store in buffers
        self.y_buffer[self.y_buffer_idx] = y_t;
        self.y_buffer_idx = (self.y_buffer_idx + 1) % self.y_buffer.len();
        if self.y_buffer_len < self.y_buffer.len() {
            self.y_buffer_len += 1;
        }
        
        self.diff_buffer[self.diff_buffer_idx] = y_diff;
        self.diff_buffer_idx = (self.diff_buffer_idx + 1) % self.diff_buffer.len();
        
        // One-step-ahead prediction using ARMA on differenced series
        let y_hat = self.predict_differenced();
        
        // Innovation
        let error = y_diff - y_hat;
        
        // Store residual for MA component
        self.residual_buffer[self.residual_idx] = error;
        self.residual_idx = (self.residual_idx + 1) % self.residual_buffer.len();
        
        // Update coefficients using simplified LMS (could be RLS for faster convergence)
        self.update_coefficients(y_diff, y_hat, error);
        
        self.last_y = y_t;
        self.n_obs += 1;
        
        error
    }

    /// Predict next value in differenced series
    #[inline]
    fn predict_differenced(&self) -> f64 {
        let mut y_hat = 0.0;
        
        // AR component on differenced series
        for i in 0..self.ar_order {
            let lag = i + 1;
            let idx = (self.diff_buffer_idx + self.diff_buffer.len() - lag) % self.diff_buffer.len();
            y_hat += self.ar_coeffs[i] * self.diff_buffer[idx];
        }
        
        // MA component
        for i in 0..self.ma_order {
            let lag = i + 1;
            let idx = (self.residual_idx + self.residual_buffer.len() - lag) % self.residual_buffer.len();
            y_hat += self.ma_coeffs[i] * self.residual_buffer[idx];
        }
        
        y_hat
    }

    /// Simple LMS coefficient update (can be replaced with RLS)
    #[inline]
    fn update_coefficients(&mut self, y_diff: f64, y_hat: f64, error: f64) {
        let learning_rate = 0.001 / (1.0 + self.n_obs as f64 * 0.0001);
        
        // Update AR coefficients
        for i in 0..self.ar_order {
            let lag = i + 1;
            let idx = (self.diff_buffer_idx + self.diff_buffer.len() - lag) % self.diff_buffer.len();
            let x_i = self.diff_buffer[idx];
            self.ar_coeffs[i] += learning_rate * error * x_i;
        }
        
        // Update MA coefficients
        for i in 0..self.ma_order {
            let lag = i + 1;
            let idx = (self.residual_idx + self.residual_buffer.len() - lag) % self.residual_buffer.len();
            let x_i = self.residual_buffer[idx];
            self.ma_coeffs[i] += learning_rate * error * x_i;
        }
        
        // Stability check: ensure AR coefficients are stationary
        self.ensure_stationarity();
    }

    /// Ensure AR coefficients correspond to a stationary process
    fn ensure_stationarity(&mut self) {
        if self.ar_order == 0 {
            return;
        }
        
        // Simple constraint: sum of |AR coeffs| < 1 for stationarity
        let sum_abs: f64 = self.ar_coeffs.iter().map(|c| c.abs()).sum();
        if sum_abs >= 1.0 {
            // Scale down coefficients
            let scale = 0.95 / sum_abs;
            for c in &mut self.ar_coeffs {
                *c *= scale;
            }
        }
    }

    /// Forecast h steps ahead in original (undifferenced) space
    pub fn forecast(&self, h: usize) -> f64 {
        if h == 0 {
            return self.last_y;
        }
        
        // Forecast in differenced space
        let mut diff_forecasts = Vec::new();
        for step in 1..=h {
            let mut f_hat = 0.0;
            
            // AR contribution
            for i in 0..self.ar_order {
                let lag = i + 1;
                let val = if lag <= step {
                    diff_forecasts[step - lag]
                } else {
                    let idx = (self.diff_buffer_idx + self.diff_buffer.len() - (lag - step)) % self.diff_buffer.len();
                    self.diff_buffer[idx]
                };
                f_hat += self.ar_coeffs[i] * val;
            }
            
            // MA contribution (zero beyond observed residuals)
            if step <= self.ma_order {
                for i in 0..self.ma_order.min(step) {
                    let lag = i + 1;
                    if lag >= step {
                        let idx = (self.residual_idx + self.residual_buffer.len() - (lag - step + 1)) % self.residual_buffer.len();
                        f_hat += self.ma_coeffs[i] * self.residual_buffer[idx];
                    }
                }
            }
            
            diff_forecasts.push(f_hat);
        }
        
        // Integrate back: need to apply inverse fractional difference
        // This is approximate; exact inversion requires infinite history
        let last_diff = self.diff_buffer[(self.diff_buffer_idx + self.diff_buffer.len() - 1) % self.diff_buffer.len()];
        let forecast_diff = *diff_forecasts.last().unwrap_or(&last_diff);
        
        // Approximate integration: y_{t+h} ≈ y_t + sum of differenced forecasts
        // For fractional d, use Grünwald-Letnikov approximation
        self.last_y + forecast_diff
    }

    /// Get the fractional differencing parameter
    #[inline]
    pub fn diff_order(&self) -> f64 {
        self.diff_order
    }

    /// Set the fractional differencing parameter (recomputes binomial coeffs)
    pub fn set_diff_order(&mut self, d: f64) {
        self.diff_order = d;
        self.binomial_coeffs = Self::compute_binomial_coeffs(d, self.max_diff_lag);
    }

    /// Get AR coefficients
    #[inline]
    pub fn ar_coefficients(&self) -> &[f64] {
        &self.ar_coeffs
    }

    /// Get MA coefficients
    #[inline]
    pub fn ma_coefficients(&self) -> &[f64] {
        &self.ma_coeffs
    }

    /// Get the Hurst exponent estimate from fractional d
    /// H = d + 0.5 for 0 < d < 0.5
    #[inline]
    pub fn hurst_exponent(&self) -> f64 {
        (self.diff_order + 0.5).min(1.0).max(0.0)
    }

    /// Check if series exhibits long memory (0 < d < 0.5)
    #[inline]
    pub fn has_long_memory(&self) -> bool {
        self.diff_order > 0.0 && self.diff_order < 0.5
    }

    /// Get number of observations processed
    #[inline]
    pub fn observation_count(&self) -> u64 {
        self.n_obs
    }

    /// Get current stationary (differenced) value
    #[inline]
    pub fn current_differenced(&self) -> f64 {
        self.diff_buffer[(self.diff_buffer_idx + self.diff_buffer.len() - 1) % self.diff_buffer.len()]
    }

    /// Reset model state
    pub fn reset(&mut self) {
        self.y_buffer.fill(0.0);
        self.y_buffer_idx = 0;
        self.y_buffer_len = 0;
        self.diff_buffer.fill(0.0);
        self.diff_buffer_idx = 0;
        self.residual_buffer.fill(0.0);
        self.residual_idx = 0;
        self.ar_coeffs.fill(0.0);
        self.ma_coeffs.fill(0.0);
        self.n_obs = 0;
        self.last_y = 0.0;
        self.running_mean = 0.0;
        self.running_mean_count = 0;
    }

    /// Estimate optimal d parameter using variance ratio method
    /// Returns the d that minimizes the variance of the differenced series
    pub fn estimate_diff_order(&self, candidate_ds: &[f64]) -> f64 {
        if self.y_buffer_len < self.max_diff_lag {
            return self.diff_order; // Not enough data
        }
        
        let mut best_d = self.diff_order;
        let mut min_var = f64::INFINITY;
        
        for &d in candidate_ds {
            // Compute binomial coeffs for this d
            let coeffs = Self::compute_binomial_coeffs(d, self.max_diff_lag);
            
            // Compute variance of differenced series
            let mut var_sum = 0.0;
            let mut mean = 0.0;
            let count = self.y_buffer_len.min(500); // Use recent data
            
            // First pass: compute mean
            for t in 0..count {
                let mut diff_val = 0.0;
                for j in 0..(t + 1).min(coeffs.len()) {
                    let idx = (self.y_buffer_idx + self.y_buffer.len() - j) % self.y_buffer.len();
                    diff_val += coeffs[j] * self.y_buffer[idx];
                }
                mean += diff_val;
            }
            mean /= count as f64;
            
            // Second pass: compute variance
            for t in 0..count {
                let mut diff_val = 0.0;
                for j in 0..(t + 1).min(coeffs.len()) {
                    let idx = (self.y_buffer_idx + self.y_buffer.len() - j) % self.y_buffer.len();
                    diff_val += coeffs[j] * self.y_buffer[idx];
                }
                let dev = diff_val - mean;
                var_sum += dev * dev;
            }
            
            let var = var_sum / count as f64;
            
            if var < min_var {
                min_var = var;
                best_d = d;
            }
        }
        
        best_d
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arfima_initialization() {
        let model = ARFIMA::new(2, 0.3, 1, 100);
        assert_eq!(model.ar_order, 2);
        assert_eq!(model.diff_order, 0.3);
        assert_eq!(model.ma_order, 1);
        assert!(model.has_long_memory());
    }

    #[test]
    fn test_fractional_difference() {
        let mut model = ARFIMA::new(1, 0.2, 0, 50);
        
        // Feed constant series
        for _ in 0..60 {
            model.update(100.0);
        }
        
        // Differenced series should be near zero for constant input
        let diff = model.current_differenced();
        assert!(diff.abs() < 1.0);
    }

    #[test]
    fn test_hurst_exponent() {
        let model = ARFIMA::new(1, 0.3, 0, 50);
        let h = model.hurst_exponent();
        assert!((h - 0.8).abs() < 0.01); // H = d + 0.5
    }

    #[test]
    fn test_forecast() {
        let mut model = ARFIMA::new(1, 0.1, 0, 50);
        
        // Train on trending data
        for i in 0..100 {
            model.update(100.0 + i as f64 * 0.1);
        }
        
        let forecast = model.forecast(5);
        // Forecast should be above last value due to trend
        assert!(forecast > model.last_y);
    }
}
