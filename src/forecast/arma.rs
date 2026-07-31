//! Online ARMA (Auto-Regressive Moving Average) Model
//!
//! Implements Recursive Least Squares (RLS) for O(1) coefficient updates
//! per tick without storing historical design matrices.

/// ARMA(p, q) model state for online learning
pub struct OnlineARMA {
    /// AR order (p)
    ar_order: usize,
    /// MA order (q)
    ma_order: usize,
    
    /// AR coefficients [phi_1, phi_2, ..., phi_p]
    ar_coeffs: Vec<f64>,
    /// MA coefficients [theta_1, theta_2, ..., theta_q]
    ma_coeffs: Vec<f64>,
    
    /// Pre-allocated circular buffer for past values (y_t)
    y_buffer: Vec<f64>,
    /// Pre-allocated circular buffer for past residuals (epsilon_t)
    epsilon_buffer: Vec<f64>,
    
    /// Write index for y_buffer
    y_idx: usize,
    /// Write index for epsilon_buffer
    epsilon_idx: usize,
    
    /// RLS covariance matrix P (flattened, size = (p+q)^2)
    /// Stored as row-major for cache efficiency
    p_matrix: Vec<f64>,
    
    /// Combined parameter vector [phi_1..phi_p, theta_1..theta_q]
    params: Vec<f64>,
    
    /// Forgetting factor for RLS (0.95 to 1.0)
    /// Lower values give more weight to recent data
    lambda: f64,
    
    /// Regularization constant for numerical stability
    regularization: f64,
    
    /// Last prediction error
    last_error: f64,
    
    /// Number of observations processed
    n_obs: u64,
}

impl OnlineARMA {
    /// Create a new online ARMA(p, q) model
    pub fn new(ar_order: usize, ma_order: usize, lambda: f64) -> Self {
        let total_params = ar_order + ma_order;
        let param_dim = total_params.max(1);
        
        // Initialize P matrix with large diagonal values (uncertainty)
        let mut p_matrix = vec![0.0; param_dim * param_dim];
        for i in 0..param_dim {
            p_matrix[i * param_dim + i] = 1e6; // Large initial uncertainty
        }
        
        OnlineARMA {
            ar_order,
            ma_order,
            ar_coeffs: vec![0.0; ar_order.max(1)],
            ma_coeffs: vec![0.0; ma_order.max(1)],
            y_buffer: vec![0.0; (ar_order.max(1) + 10)],
            epsilon_buffer: vec![0.0; (ma_order.max(1) + 10)],
            y_idx: 0,
            epsilon_idx: 0,
            p_matrix,
            params: vec![0.0; param_dim],
            lambda: lambda.max(0.9).min(1.0),
            regularization: 1e-8,
            last_error: 0.0,
            n_obs: 0,
        }
    }

    /// Update model with new observation using RLS
    /// Returns the one-step-ahead prediction error
    #[inline]
    pub fn update(&mut self, y_t: f64) -> f64 {
        // Get regressor vector phi_t
        let phi_t = self.build_regressor();
        
        // One-step-ahead prediction
        let y_hat = self.predict_one_step(&phi_t);
        
        // Prediction error
        let error = y_t - y_hat;
        self.last_error = error;
        
        if self.n_obs > (self.ar_order + self.ma_order) as u64 {
            // RLS update
            self.rls_update(&phi_t, error);
        }
        
        // Store observation
        self.y_buffer[self.y_idx] = y_t;
        self.y_idx = (self.y_idx + 1) % self.y_buffer.len();
        
        // Store residual
        self.epsilon_buffer[self.epsilon_idx] = error;
        self.epsilon_idx = (self.epsilon_idx + 1) % self.epsilon_buffer.len();
        
        self.n_obs += 1;
        
        error
    }

    /// Build the regressor vector for current time step
    #[inline]
    fn build_regressor(&self) -> Vec<f64> {
        let mut phi = Vec::with_capacity(self.ar_order + self.ma_order);
        
        // AR part: [-y_{t-1}, -y_{t-2}, ..., -y_{t-p}]
        for i in 1..=self.ar_order {
            let idx = (self.y_idx + self.y_buffer.len() - i) % self.y_buffer.len();
            phi.push(-self.y_buffer[idx]);
        }
        
        // MA part: [epsilon_{t-1}, epsilon_{t-2}, ..., epsilon_{t-q}]
        for i in 1..=self.ma_order {
            let idx = (self.epsilon_idx + self.epsilon_buffer.len() - i) % self.epsilon_buffer.len();
            phi.push(self.epsilon_buffer[idx]);
        }
        
        phi
    }

    /// One-step-ahead prediction given regressor
    #[inline]
    fn predict_one_step(&self, phi_t: &[f64]) -> f64 {
        let mut y_hat = 0.0;
        for (i, &phi_i) in phi_t.iter().enumerate() {
            y_hat += self.params[i] * phi_i;
        }
        y_hat
    }

    /// RLS parameter update
    #[inline]
    fn rls_update(&mut self, phi_t: &[f64], error: f64) {
        let n = self.params.len();
        if n == 0 {
            return;
        }
        
        // Compute K_t = P_{t-1} * phi_t / (lambda + phi_t' * P_{t-1} * phi_t)
        // First compute P * phi
        let mut p_phi = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                p_phi[i] += self.p_matrix[i * n + j] * phi_t[j];
            }
        }
        
        // Compute denominator: lambda + phi' * P * phi
        let mut denom = self.lambda;
        for i in 0..n {
            denom += phi_t[i] * p_phi[i];
        }
        
        if denom.abs() < self.regularization {
            return; // Avoid numerical instability
        }
        
        // K = P * phi / denom
        let gain: Vec<f64> = p_phi.iter().map(|&x| x / denom).collect();
        
        // Update parameters: theta_t = theta_{t-1} + K * error
        for i in 0..n {
            self.params[i] += gain[i] * error;
        }
        
        // Update P matrix: P_t = (P_{t-1} - K * phi' * P_{t-1}) / lambda
        // Using rank-1 update for efficiency
        for i in 0..n {
            for j in 0..n {
                self.p_matrix[i * n + j] -= gain[i] * phi_t[j] * self.p_matrix[j * n + j].max(0.0);
                self.p_matrix[i * n + j] /= self.lambda;
            }
        }
        
        // Extract updated AR and MA coefficients
        for i in 0..self.ar_order {
            self.ar_coeffs[i] = self.params[i];
        }
        for i in 0..self.ma_order {
            self.ma_coeffs[i] = self.params[self.ar_order + i];
        }
    }

    /// Forecast h steps ahead
    pub fn forecast(&self, h: usize) -> f64 {
        if h == 0 {
            return self.y_buffer[(self.y_idx + self.y_buffer.len() - 1) % self.y_buffer.len()];
        }
        
        // Iterative forecasting
        let mut forecasts = Vec::new();
        for step in 1..=h {
            let mut y_hat = 0.0;
            
            // AR contribution
            for i in 0..self.ar_order {
                let lag = i + 1;
                let y_lag = if lag <= step {
                    forecasts[step - lag]
                } else {
                    let idx = (self.y_idx + self.y_buffer.len() - (lag - step)) % self.y_buffer.len();
                    self.y_buffer[idx]
                };
                y_hat -= self.ar_coeffs[i] * y_lag;
            }
            
            // MA contribution (only for first q steps, then zero)
            if step <= self.ma_order {
                for i in 0..self.ma_order.min(step) {
                    let eps_lag = if i + 1 < step {
                        0.0 // Future residuals assumed zero
                    } else {
                        let idx = (self.epsilon_idx + self.epsilon_buffer.len() - (step - i - 1)) % self.epsilon_buffer.len();
                        self.epsilon_buffer[idx]
                    };
                    y_hat += self.ma_coeffs[i] * eps_lag;
                }
            }
            
            forecasts.push(y_hat);
        }
        
        *forecasts.last().unwrap_or(&0.0)
    }

    /// Get current AR coefficients
    #[inline]
    pub fn ar_coefficients(&self) -> &[f64] {
        &self.ar_coeffs
    }

    /// Get current MA coefficients
    #[inline]
    pub fn ma_coefficients(&self) -> &[f64] {
        &self.ma_coeffs
    }

    /// Get last prediction error (innovation)
    #[inline]
    pub fn last_prediction_error(&self) -> f64 {
        self.last_error
    }

    /// Get number of observations processed
    #[inline]
    pub fn observation_count(&self) -> u64 {
        self.n_obs
    }

    /// Calculate residual variance estimate
    pub fn residual_variance(&self) -> f64 {
        if self.n_obs < 2 {
            return 0.0;
        }
        
        // Use exponential weighted moving average of squared errors
        let mut var_sum = 0.0;
        let mut weight_sum = 0.0;
        let mut weight = 1.0;
        
        for i in 0..self.epsilon_buffer.len().min(self.n_obs as usize) {
            let idx = (self.epsilon_idx + self.epsilon_buffer.len() - i - 1) % self.epsilon_buffer.len();
            let eps = self.epsilon_buffer[idx];
            var_sum += weight * eps * eps;
            weight_sum += weight;
            weight *= self.lambda;
        }
        
        if weight_sum < 1e-10 {
            return 0.0;
        }
        
        var_sum / weight_sum
    }

    /// Check if model has converged (parameter changes below threshold)
    pub fn has_converged(&self, threshold: f64) -> bool {
        if self.n_obs < 100 {
            return false;
        }
        
        // Simple heuristic: check if last error is small relative to signal
        let last_y = self.y_buffer[(self.y_idx + self.y_buffer.len() - 1) % self.y_buffer.len()];
        if last_y.abs() < 1e-10 {
            return true;
        }
        
        (self.last_error / last_y).abs() < threshold
    }

    /// Reset model state
    pub fn reset(&mut self) {
        let param_dim = self.params.len();
        for i in 0..param_dim {
            for j in 0..param_dim {
                if i == j {
                    self.p_matrix[i * param_dim + j] = 1e6;
                } else {
                    self.p_matrix[i * param_dim + j] = 0.0;
                }
            }
        }
        self.params.fill(0.0);
        self.ar_coeffs.fill(0.0);
        self.ma_coeffs.fill(0.0);
        self.y_buffer.fill(0.0);
        self.epsilon_buffer.fill(0.0);
        self.n_obs = 0;
        self.last_error = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arma_initialization() {
        let model = OnlineARMA::new(2, 1, 0.98);
        assert_eq!(model.ar_order, 2);
        assert_eq!(model.ma_order, 1);
        assert_eq!(model.params.len(), 3);
    }

    #[test]
    fn test_arma_update_and_predict() {
        let mut model = OnlineARMA::new(1, 0, 0.99); // Simple AR(1)
        
        // Generate some AR(1) data: y_t = 0.8 * y_{t-1} + noise
        let mut y_prev = 1.0;
        for _ in 0..100 {
            let noise = (rand_f64() - 0.5) * 0.1;
            let y_t = 0.8 * y_prev + noise;
            model.update(y_t);
            y_prev = y_t;
        }
        
        // Check that AR coefficient is close to 0.8
        let ar_coef = model.ar_coefficients()[0];
        assert!((ar_coef.abs() - 0.8).abs() < 0.3, "AR coefficient should be near 0.8, got {}", ar_coef);
    }

    #[test]
    fn test_forecast() {
        let mut model = OnlineARMA::new(1, 0, 0.99);
        
        // Train on constant series
        for _ in 0..50 {
            model.update(100.0);
        }
        
        // Forecast should be close to 100
        let forecast = model.forecast(5);
        assert!((forecast - 100.0).abs() < 10.0);
    }

    // Simple pseudo-random for testing
    fn rand_f64() -> f64 {
        static mut SEED: u64 = 12345;
        unsafe {
            SEED = SEED.wrapping_mul(6364136223846793005).wrapping_add(1);
            (SEED as f64) / (u64::MAX as f64)
        }
    }
}
