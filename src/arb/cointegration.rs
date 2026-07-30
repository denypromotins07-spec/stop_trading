//! Online Cointegration Test using Rolling Window OLS
//! 
//! Implements Engle-Granger cointegration test with rolling window OLS regressions
//! for dynamic hedge ratio updates in pairs trading without storing massive matrices.

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use crossbeam_utils::CachePadded;

/// Maximum window size for rolling regression
const MAX_WINDOW_SIZE: usize = 256;

/// Rolling statistics for online OLS
#[repr(C)]
#[derive(Clone, Copy)]
struct RollingStats {
    /// Sum of x values
    sum_x: f64,
    /// Sum of y values
    sum_y: f64,
    /// Sum of x squared
    sum_xx: f64,
    /// Sum of xy products
    sum_xy: f64,
    /// Sum of y squared
    sum_yy: f64,
    /// Count of observations
    count: u32,
}

impl Default for RollingStats {
    fn default() -> Self {
        Self {
            sum_x: 0.0,
            sum_y: 0.0,
            sum_xx: 0.0,
            sum_xy: 0.0,
            sum_yy: 0.0,
            count: 0,
        }
    }
}

/// Circular buffer for rolling window
struct CircularBuffer {
    /// X values (asset 1 returns)
    x_values: [f64; MAX_WINDOW_SIZE],
    /// Y values (asset 2 returns)
    y_values: [f64; MAX_WINDOW_SIZE],
    /// Current write index
    write_idx: usize,
    /// Number of valid entries
    count: usize,
    /// Window size
    window_size: usize,
}

impl CircularBuffer {
    fn new(window_size: usize) -> Self {
        Self {
            x_values: [0.0; MAX_WINDOW_SIZE],
            y_values: [0.0; MAX_WINDOW_SIZE],
            write_idx: 0,
            count: 0,
            window_size: window_size.min(MAX_WINDOW_SIZE),
        }
    }

    #[inline]
    fn push(&mut self, x: f64, y: f64) {
        self.x_values[self.write_idx] = x;
        self.y_values[self.write_idx] = y;
        
        if self.count < self.window_size {
            self.count += 1;
        }
        
        self.write_idx = (self.write_idx + 1) % self.window_size;
    }

    #[inline]
    fn calculate_stats(&self) -> RollingStats {
        let mut stats = RollingStats::default();
        
        for i in 0..self.count {
            let x = self.x_values[i];
            let y = self.y_values[i];
            
            stats.sum_x += x;
            stats.sum_y += y;
            stats.sum_xx += x * x;
            stats.sum_xy += x * y;
            stats.sum_yy += y * y;
            stats.count += 1;
        }
        
        stats
    }
}

/// Cointegration test result
pub struct CointegrationResult {
    /// Hedge ratio (beta)
    pub hedge_ratio: f64,
    /// Alpha (intercept)
    pub alpha: f64,
    /// R-squared value
    pub r_squared: f64,
    /// Standard error of residuals
    pub residual_std: f64,
    /// ADF test statistic approximation
    pub adf_statistic: f64,
    /// Whether cointegration is significant
    pub is_cointegrated: bool,
    /// Number of observations used
    pub observation_count: usize,
}

/// Online Engle-Granger Cointegration Engine
pub struct CointegrationEngine {
    /// Rolling window buffer
    buffer: CircularBuffer,
    /// Accumulated statistics (for incremental updates)
    stats: CachePadded<RollingStats>,
    /// Current hedge ratio
    hedge_ratio: CachePadded<AtomicU64>, // Stored as scaled integer
    /// Residual mean for ADF approximation
    residual_mean: CachePadded<AtomicU64>,
    /// Residual variance accumulator
    residual_variance: CachePadded<AtomicU64>,
    /// Minimum observations required
    min_observations: usize,
    /// Critical value for ADF (scaled by 1000)
    critical_value_scaled: i32,
    /// Engine enabled
    enabled: CachePadded<AtomicBool>,
    /// Update counter
    update_count: CachePadded<AtomicU64>,
}

impl CointegrationEngine {
    /// Create a new cointegration engine
    /// 
    /// # Arguments
    /// * `window_size` - Rolling window size for regression
    /// * `min_observations` - Minimum observations before producing results
    /// * `critical_value` - ADF critical value (e.g., -3.41 for 5% significance)
    pub fn new(window_size: usize, min_observations: usize, critical_value: f64) -> Self {
        Self {
            buffer: CircularBuffer::new(window_size),
            stats: CachePadded::new(RollingStats::default()),
            hedge_ratio: CachePadded::new(AtomicU64::new(0)),
            residual_mean: CachePadded::new(AtomicU64::new(0)),
            residual_variance: CachePadded::new(AtomicU64::new(0)),
            min_observations,
            critical_value_scaled: (critical_value * 1000.0) as i32,
            enabled: CachePadded::new(AtomicBool::new(true)),
            update_count: CachePadded::new(AtomicU64::new(0)),
        }
    }

    /// Add a new observation pair
    /// 
    /// # Arguments
    /// * `price1` - Price of asset 1
    /// * `price2` - Price of asset 2
    #[inline]
    pub fn add_observation(&self, price1: f64, price2: f64) {
        if !self.enabled.load(Ordering::Relaxed) {
            return;
        }

        // Calculate log returns
        static mut PREV_PRICE1: f64 = 0.0;
        static mut PREV_PRICE2: f64 = 0.0;

        unsafe {
            let ret1 = if PREV_PRICE1 > 0.0 {
                (price1 / PREV_PRICE1).ln()
            } else {
                0.0
            };
            let ret2 = if PREV_PRICE2 > 0.0 {
                (price2 / PREV_PRICE2).ln()
            } else {
                0.0
            };

            PREV_PRICE1 = price1;
            PREV_PRICE2 = price2;

            // Push to circular buffer
            self.buffer.push(ret1, ret2);

            // Recalculate statistics
            let new_stats = self.buffer.calculate_stats();
            self.update_regression(&new_stats);
        }

        self.update_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Update regression coefficients from statistics
    #[inline]
    fn update_regression(&self, stats: &RollingStats) {
        if stats.count < 2 {
            return;
        }

        let n = stats.count as f64;
        
        // Calculate means
        let mean_x = stats.sum_x / n;
        let mean_y = stats.sum_y / n;

        // Calculate beta (hedge ratio)
        // β = Σ(xy) - n*x̄*ȳ / Σ(x²) - n*x̄²
        let numerator = stats.sum_xy - n * mean_x * mean_y;
        let denominator = stats.sum_xx - n * mean_x * mean_x;

        let beta = if denominator.abs() > 1e-10 {
            numerator / denominator
        } else {
            0.0
        };

        // Calculate alpha
        let alpha = mean_y - beta * mean_x;

        // Store hedge ratio as scaled integer (scale by 1e9)
        let beta_scaled = (beta * 1e9) as i64;
        self.hedge_ratio.store(beta_scaled as u64, Ordering::Relaxed);

        // Calculate R-squared
        let ss_tot = stats.sum_yy - n * mean_y * mean_y;
        let ss_res = if denominator.abs() > 1e-10 {
            stats.sum_yy - alpha * stats.sum_y - beta * stats.sum_xy
        } else {
            ss_tot
        };

        let r_squared = if ss_tot > 1e-10 {
            1.0 - (ss_res / ss_tot)
        } else {
            0.0
        };

        // Update residual statistics for ADF approximation
        let residual_var = if stats.count > 2 {
            ss_res / (stats.count as f64 - 2.0)
        } else {
            0.0
        };

        let residual_std = residual_var.sqrt();
        let residual_std_scaled = (residual_std * 1e9) as i64;
        self.residual_variance.store(residual_std_scaled.unsigned_abs(), Ordering::Relaxed);
    }

    /// Get current cointegration test result
    pub fn get_result(&self) -> Option<CointegrationResult> {
        let stats = self.buffer.calculate_stats();
        
        if stats.count < self.min_observations {
            return None;
        }

        let n = stats.count as f64;
        let mean_x = stats.sum_x / n;
        let mean_y = stats.sum_y / n;

        let numerator = stats.sum_xy - n * mean_x * mean_y;
        let denominator = stats.sum_xx - n * mean_x * mean_x;

        let beta = if denominator.abs() > 1e-10 {
            numerator / denominator
        } else {
            0.0
        };

        let alpha = mean_y - beta * mean_x;

        let ss_tot = stats.sum_yy - n * mean_y * mean_y;
        let ss_res = if denominator.abs() > 1e-10 {
            stats.sum_yy - alpha * stats.sum_y - beta * stats.sum_xy
        } else {
            ss_tot
        };

        let r_squared = if ss_tot > 1e-10 {
            (1.0 - (ss_res / ss_tot)).max(0.0).min(1.0)
        } else {
            0.0
        };

        let residual_std = if stats.count > 2 {
            (ss_res / (stats.count as f64 - 2.0)).sqrt()
        } else {
            0.0
        };

        // Simplified ADF statistic approximation
        // In production, this would use actual unit root testing
        let adf_approx = if residual_std > 0.0 && beta > 0.0 {
            -beta / residual_std
        } else {
            0.0
        };

        let is_cointegrated = adf_approx * 1000.0 < self.critical_value_scaled as f64;

        Some(CointegrationResult {
            hedge_ratio: beta,
            alpha,
            r_squared,
            residual_std,
            adf_statistic: adf_approx,
            is_cointegrated,
            observation_count: stats.count,
        })
    }

    /// Get current hedge ratio
    #[inline]
    pub fn get_hedge_ratio(&self) -> f64 {
        let scaled = self.hedge_ratio.load(Ordering::Relaxed) as i64;
        scaled as f64 / 1e9
    }

    /// Check if pair is currently cointegrated
    #[inline]
    pub fn is_cointegrated(&self) -> bool {
        if let Some(result) = self.get_result() {
            result.is_cointegrated
        } else {
            false
        }
    }

    /// Get spread value given current prices
    #[inline]
    pub fn calculate_spread(&self, price1: f64, price2: f64) -> f64 {
        let beta = self.get_hedge_ratio();
        price2 - beta * price1
    }

    /// Get Z-score of current spread
    pub fn calculate_zscore(&self, price1: f64, price2: f64) -> f64 {
        let spread = self.calculate_spread(price1, price2);
        
        if let Some(result) = self.get_result() {
            if result.residual_std > 0.0 {
                return spread / result.residual_std;
            }
        }
        0.0
    }

    /// Reset the engine
    pub fn reset(&self) {
        self.buffer = CircularBuffer::new(self.buffer.window_size);
        self.stats = CachePadded::new(RollingStats::default());
        self.hedge_ratio.store(0, Ordering::Relaxed);
        self.residual_mean.store(0, Ordering::Relaxed);
        self.residual_variance.store(0, Ordering::Relaxed);
        self.update_count.store(0, Ordering::Relaxed);
    }

    /// Enable the engine
    pub fn enable(&self) {
        self.enabled.store(true, Ordering::Relaxed);
    }

    /// Disable the engine
    pub fn disable(&self) {
        self.enabled.store(false, Ordering::Relaxed);
    }

    /// Get update count
    #[inline]
    pub fn get_update_count(&self) -> u64 {
        self.update_count.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cointegration_basic() {
        let engine = CointegrationEngine::new(100, 30, -3.41);
        
        // Add correlated observations
        for i in 0..50 {
            let price1 = 100.0 + (i as f64 * 0.1).sin() * 5.0;
            let price2 = 100.0 + (i as f64 * 0.1).sin() * 5.0 * 1.5; // Correlated
            engine.add_observation(price1, price2);
        }
        
        let result = engine.get_result();
        assert!(result.is_some());
        
        let r = result.unwrap();
        assert!(r.observation_count >= 30);
        assert!(r.hedge_ratio > 0.0);
    }

    #[test]
    fn test_spread_calculation() {
        let engine = CointegrationEngine::new(50, 20, -3.41);
        
        // Establish hedge ratio
        for i in 0..30 {
            engine.add_observation(100.0, 150.0);
        }
        
        let spread = engine.calculate_spread(100.0, 150.0);
        assert!(spread.is_finite());
    }
}
