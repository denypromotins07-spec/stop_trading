//! Highly optimized, simplified GARCH(1,1) variance predictor.
//! Forecasts short-term volatility spikes for spread adjustment and adverse selection avoidance.

use std::sync::atomic::{AtomicF64, AtomicU64, Ordering};

/// Error types for GARCH calculations
#[derive(Debug, thiserror::Error)]
pub enum GarchError {
    #[error("Invalid parameters: omega must be positive")]
    InvalidOmega,
    #[error("Invalid parameters: alpha must be in [0, 1)")]
    InvalidAlpha,
    #[error("Invalid parameters: beta must be in [0, 1)")]
    InvalidBeta,
    #[error("Invalid parameters: alpha + beta must be < 1 for stationarity")]
    NonStationary,
    #[error("Insufficient data for estimation")]
    InsufficientData,
}

/// GARCH(1,1) model parameters
#[derive(Debug, Clone, Copy)]
pub struct GarchParams {
    pub omega: f64, // Long-run average variance
    pub alpha: f64, // Coefficient for lagged squared return (news impact)
    pub beta: f64,  // Coefficient for lagged variance (persistence)
}

impl GarchParams {
    /// Create new GARCH parameters with validation
    pub fn new(omega: f64, alpha: f64, beta: f64) -> Result<Self, GarchError> {
        if omega <= 0.0 {
            return Err(GarchError::InvalidOmega);
        }
        if alpha < 0.0 || alpha >= 1.0 {
            return Err(GarchError::InvalidAlpha);
        }
        if beta < 0.0 || beta >= 1.0 {
            return Err(GarchError::InvalidBeta);
        }
        if alpha + beta >= 1.0 {
            return Err(GarchError::NonStationary);
        }
        
        Ok(Self { omega, alpha, beta })
    }

    /// Standard parameters commonly used in finance
    pub fn standard() -> Self {
        Self {
            omega: 0.000002,
            alpha: 0.1,
            beta: 0.85,
        }
    }

    /// Check if parameters satisfy stationarity condition
    pub fn is_stationary(&self) -> bool {
        self.alpha + self.beta < 1.0
    }

    /// Calculate long-run (unconditional) variance
    pub fn unconditional_variance(&self) -> f64 {
        self.omega / (1.0 - self.alpha - self.beta)
    }

    /// Calculate half-life of volatility shocks in periods
    pub fn shock_half_life(&self) -> f64 {
        let sum = self.alpha + self.beta;
        if sum <= 0.0 {
            return 0.0;
        }
        -0.693 / sum.ln()
    }
}

/// GARCH(1,1) variance predictor with lock-free updates
pub struct Garch11 {
    params: GarchParams,
    current_variance: AtomicF64,
    last_return: AtomicF64,
    count: AtomicU64,
    initialized: AtomicU64,
    warmup_period: usize,
}

impl Garch11 {
    /// Create a new GARCH(1,1) predictor with given parameters
    pub fn new(params: GarchParams, warmup_period: usize) -> Self {
        let initial_variance = params.unconditional_variance();
        
        Self {
            params,
            current_variance: AtomicF64::new(initial_variance),
            last_return: AtomicF64::new(0.0),
            count: AtomicU64::new(0),
            initialized: AtomicU64::new(0),
            warmup_period,
        }
    }

    /// Create with standard parameters
    pub fn standard() -> Self {
        Self::new(GarchParams::standard(), 100)
    }

    /// Update with new price and return predicted next variance
    pub fn update(&self, price: f64) -> Result<f64, GarchError> {
        let count = self.count.load(Ordering::Relaxed);
        
        if count == 0 {
            self.last_return.store(0.0, Ordering::Relaxed);
            self.count.fetch_add(1, Ordering::Relaxed);
            Ok(self.current_variance.load(Ordering::Relaxed))
        } else {
            let last_price = if count == 1 { 
                price // First real return calculation
            } else {
                // We need to track last price, but for simplicity use log return approximation
                self.last_return.load(Ordering::Relaxed)
            };
            
            // Calculate log return
            let last_return_val = self.last_return.load(Ordering::Relaxed);
            let current_return = if count == 1 {
                0.0 // First period has no return
            } else {
                (price / last_price).ln()
            };
            
            if count > 1 {
                let squared_return = current_return * current_return;
                let prev_variance = self.current_variance.load(Ordering::Relaxed);
                
                // GARCH(1,1) formula: h_t = omega + alpha * r_{t-1}^2 + beta * h_{t-1}
                let new_variance = self.params.omega 
                    + self.params.alpha * squared_return 
                    + self.params.beta * prev_variance;
                
                self.current_variance.store(new_variance.max(1e-10), Ordering::Relaxed);
            }
            
            self.last_return.store(price, Ordering::Relaxed);
            self.count.fetch_add(1, Ordering::Relaxed);
            
            if count >= self.warmup_period as u64 {
                self.initialized.store(1, Ordering::Relaxed);
            }
            
            Ok(self.current_variance.load(Ordering::Relaxed))
        }
    }

    /// Update with pre-calculated return (for efficiency)
    pub fn update_return(&self, ret: f64) -> f64 {
        let squared_return = ret * ret;
        let prev_variance = self.current_variance.load(Ordering::Relaxed);
        
        let new_variance = self.params.omega 
            + self.params.alpha * squared_return 
            + self.params.beta * prev_variance;
        
        self.current_variance.store(new_variance.max(1e-10), Ordering::Relaxed);
        self.last_return.store(ret, Ordering::Relaxed);
        self.count.fetch_add(1, Ordering::Relaxed);
        
        new_variance
    }

    /// Get current variance estimate
    pub fn get_variance(&self) -> f64 {
        self.current_variance.load(Ordering::Relaxed)
    }

    /// Get current volatility (sqrt of variance)
    pub fn get_volatility(&self) -> f64 {
        self.get_variance().sqrt()
    }

    /// Predict variance for n steps ahead
    pub fn predict_variance_ahead(&self, steps: usize) -> f64 {
        let current_var = self.current_variance.load(Ordering::Relaxed);
        let long_run_var = self.params.unconditional_variance();
        let persistence = self.params.alpha + self.params.beta;
        
        // Mean reversion formula
        long_run_var + (current_var - long_run_var) * persistence.powi(steps as i32)
    }

    /// Check if model is warmed up
    pub fn is_ready(&self) -> bool {
        self.initialized.load(Ordering::Relaxed) != 0
    }

    /// Get recommended spread adjustment based on predicted volatility
    pub fn spread_adjustment(&self, base_spread: f64) -> f64 {
        let vol = self.get_volatility();
        let long_run_vol = self.params.unconditional_variance().sqrt();
        
        // Adjust spread proportionally to volatility relative to long-run average
        if long_run_vol < 1e-10 {
            return base_spread;
        }
        
        let vol_ratio = vol / long_run_vol;
        base_spread * vol_ratio.max(1.0) // Never reduce below base
    }

    /// Detect volatility spike (current vs long-run average)
    pub fn is_volatility_spike(&self, threshold_multiplier: f64) -> bool {
        let current_vol = self.get_volatility();
        let long_run_vol = self.params.unconditional_variance().sqrt();
        
        current_vol > long_run_vol * threshold_multiplier
    }

    /// Reset the model
    pub fn reset(&self) {
        let initial_var = self.params.unconditional_variance();
        self.current_variance.store(initial_var, Ordering::Relaxed);
        self.last_return.store(0.0, Ordering::Relaxed);
        self.count.store(0, Ordering::Relaxed);
        self.initialized.store(0, Ordering::Relaxed);
    }

    /// Get model parameters
    pub fn params(&self) -> GarchParams {
        self.params
    }
}

/// Volatility forecast result
#[derive(Debug, Clone, Copy)]
pub struct VolatilityForecast {
    pub current_variance: f64,
    pub current_volatility: f64,
    pub one_step_ahead: f64,
    pub five_step_ahead: f64,
    pub long_run_variance: f64,
    pub is_spike: bool,
}

impl Garch11 {
    /// Get comprehensive volatility forecast
    pub fn get_forecast(&self, spike_threshold: f64) -> VolatilityForecast {
        let current_var = self.get_variance();
        let current_vol = current_var.sqrt();
        
        VolatilityForecast {
            current_variance: current_var,
            current_volatility: current_vol,
            one_step_ahead: self.predict_variance_ahead(1).sqrt(),
            five_step_ahead: self.predict_variance_ahead(5).sqrt(),
            long_run_variance: self.params.unconditional_variance(),
            is_spike: self.is_volatility_spike(spike_threshold),
        }
    }
}

/// Adaptive spread calculator using GARCH forecasts
pub struct AdaptiveSpread {
    garch: Garch11,
    base_spread_bps: AtomicF64,
    min_spread_bps: AtomicF64,
    max_spread_bps: AtomicF64,
}

impl AdaptiveSpread {
    /// Create a new adaptive spread calculator
    pub fn new(base_spread_bps: f64, min_spread: f64, max_spread: f64) -> Self {
        Self {
            garch: Garch11::standard(),
            base_spread_bps: AtomicF64::new(base_spread_bps),
            min_spread_bps: AtomicF64::new(min_spread),
            max_spread_bps: AtomicF64::new(max_spread),
        }
    }

    /// Update with new price and get adjusted spread
    pub fn update(&self, price: f64) -> SpreadQuote {
        let _ = self.garch.update(price);
        self.get_spread()
    }

    /// Get current spread quote
    pub fn get_spread(&self) -> SpreadQuote {
        let base = self.base_spread_bps.load(Ordering::Relaxed);
        let min_spread = self.min_spread_bps.load(Ordering::Relaxed);
        let max_spread = self.max_spread_bps.load(Ordering::Relaxed);
        
        let forecast = self.garch.get_forecast(2.0); // 2x long-run vol threshold
        
        // Adjust spread based on volatility forecast
        let adjusted = base * forecast.one_step_ahead / forecast.long_run_variance.sqrt();
        let clamped = adjusted.max(min_spread).min(max_spread);
        
        SpreadQuote {
            bid_adjustment_bps: -clamped / 2.0,
            ask_adjustment_bps: clamped / 2.0,
            total_spread_bps: clamped,
            is_wide: clamped > base * 1.5,
            volatility_regime: if forecast.is_spike {
                VolatilityRegime::High
            } else if forecast.one_step_ahead < forecast.long_run_variance.sqrt() * 0.8 {
                VolatilityRegime::Low
            } else {
                VolatilityRegime::Normal
            },
        }
    }

    /// Get reference to underlying GARCH model
    pub fn garch(&self) -> &Garch11 {
        &self.garch
    }

    /// Reset the calculator
    pub fn reset(&self) {
        self.garch.reset();
    }
}

/// Spread quote with adjustments
#[derive(Debug, Clone, Copy)]
pub struct SpreadQuote {
    pub bid_adjustment_bps: f64,
    pub ask_adjustment_bps: f64,
    pub total_spread_bps: f64,
    pub is_wide: bool,
    pub volatility_regime: VolatilityRegime,
}

/// Volatility regime classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VolatilityRegime {
    Low,
    Normal,
    High,
}

impl SpreadQuote {
    /// Apply spread adjustments to mid price
    pub fn apply_to_mid(&self, mid_price: f64) -> (f64, f64) {
        let bid = mid_price * (1.0 + self.bid_adjustment_bps / 10000.0);
        let ask = mid_price * (1.0 + self.ask_adjustment_bps / 10000.0);
        (bid, ask)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_garch_params_validation() {
        // Valid params
        let params = GarchParams::new(0.000002, 0.1, 0.85);
        assert!(params.is_ok());
        
        // Invalid omega
        assert!(GarchParams::new(-0.000002, 0.1, 0.85).is_err());
        
        // Invalid alpha
        assert!(GarchParams::new(0.000002, -0.1, 0.85).is_err());
        assert!(GarchParams::new(0.000002, 1.0, 0.85).is_err());
        
        // Non-stationary
        assert!(GarchParams::new(0.000002, 0.5, 0.6).is_err());
    }

    #[test]
    fn test_garch_unconditional_variance() {
        let params = GarchParams::new(0.000002, 0.1, 0.85).unwrap();
        let long_run_var = params.unconditional_variance();
        
        // Should be omega / (1 - alpha - beta)
        let expected = 0.000002 / (1.0 - 0.1 - 0.85);
        assert!((long_run_var - expected).abs() < 1e-10);
    }

    #[test]
    fn test_garch_update() {
        let garch = Garch11::standard();
        
        // Feed some prices
        for i in 0..150 {
            let price = 100.0 + (i as f64).sin() * 5.0;
            let _ = garch.update(price);
        }
        
        assert!(garch.is_ready());
        let var = garch.get_variance();
        assert!(var > 0.0);
        assert!(var.is_finite());
    }

    #[test]
    fn test_volatility_forecast() {
        let garch = Garch11::standard();
        
        for i in 0..150 {
            let price = 100.0 + (i as f64).sin() * 5.0;
            let _ = garch.update(price);
        }
        
        let forecast = garch.get_forecast(2.0);
        
        assert!(forecast.current_volatility > 0.0);
        assert!(forecast.one_step_ahead > 0.0);
        assert!(forecast.five_step_ahead > 0.0);
        assert!(forecast.long_run_variance > 0.0);
    }

    #[test]
    fn test_adaptive_spread() {
        let spread_calc = AdaptiveSpread::new(10.0, 5.0, 50.0);
        
        for i in 0..150 {
            let price = 100.0 + (i as f64).sin() * 5.0;
            spread_calc.update(price);
        }
        
        let quote = spread_calc.get_spread();
        
        assert!(quote.total_spread_bps >= 5.0);
        assert!(quote.total_spread_bps <= 50.0);
        assert!(quote.bid_adjustment_bps < 0.0);
        assert!(quote.ask_adjustment_bps > 0.0);
    }

    #[test]
    fn test_spread_application() {
        let quote = SpreadQuote {
            bid_adjustment_bps: -5.0,
            ask_adjustment_bps: 5.0,
            total_spread_bps: 10.0,
            is_wide: false,
            volatility_regime: VolatilityRegime::Normal,
        };
        
        let mid = 100.0;
        let (bid, ask) = quote.apply_to_mid(mid);
        
        assert!(bid < mid);
        assert!(ask > mid);
        assert!((ask - bid) / mid * 10000.0 - 10.0).abs() < 0.01;
    }
}
