//! Dual Kalman Filter for Spread Modeling
//! 
//! Implements a dual-Kalman filter system to model the hidden spread and 
//! mean-reversion speed of asset pairs for statistical arbitrage.

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use crossbeam_utils::CachePadded;

/// Kalman filter state for spread estimation
#[repr(C)]
#[derive(Clone, Copy)]
struct KalmanState {
    /// State estimate (x)
    x: f64,
    /// State covariance (P)
    p: f64,
    /// Process noise variance (Q)
    q: f64,
    /// Measurement noise variance (R)
    r: f64,
}

impl Default for KalmanState {
    fn default() -> Self {
        Self {
            x: 0.0,
            p: 1.0,
            q: 0.001,
            r: 0.1,
        }
    }
}

/// Kalman filter output
pub struct KalmanOutput {
    /// Estimated state
    pub estimate: f64,
    /// Estimation variance
    pub variance: f64,
    /// Kalman gain
    pub kalman_gain: f64,
    /// Innovation (measurement residual)
    pub innovation: f64,
}

/// Dual Kalman Filter Engine for spread modeling
pub struct KalmanSpreadEngine {
    /// Primary filter for spread level
    spread_filter: CachePadded<KalmanState>,
    /// Secondary filter for mean-reversion speed (theta)
    theta_filter: CachePadded<KalmanState>,
    /// Current Z-score
    z_score: CachePadded<AtomicU64>, // Scaled by 1000
    /// Current spread estimate
    spread_estimate: CachePadded<AtomicU64>, // Scaled by 1e9
    /// Mean reversion half-life (milliseconds)
    half_life_ms: CachePadded<AtomicU64>,
    /// Filter enabled
    enabled: CachePadded<AtomicBool>,
    /// Update count
    update_count: CachePadded<AtomicU64>,
    /// Signal threshold (scaled by 1000)
    entry_threshold_scaled: i32,
    /// Exit threshold (scaled by 1000)
    exit_threshold_scaled: i32,
}

impl KalmanSpreadEngine {
    /// Create a new Kalman spread engine
    /// 
    /// # Arguments
    /// * `process_noise` - Process noise variance for spread filter
    /// * `measurement_noise` - Measurement noise variance
    /// * `entry_threshold` - Z-score threshold for entry (e.g., 2.0)
    /// * `exit_threshold` - Z-score threshold for exit (e.g., 0.5)
    pub fn new(
        process_noise: f64,
        measurement_noise: f64,
        entry_threshold: f64,
        exit_threshold: f64,
    ) -> Self {
        Self {
            spread_filter: CachePadded::new(KalmanState {
                x: 0.0,
                p: 1.0,
                q: process_noise,
                r: measurement_noise,
            }),
            theta_filter: CachePadded::new(KalmanState {
                x: -0.1, // Initial mean-reversion speed estimate
                p: 0.1,
                q: 0.0001,
                r: 0.01,
            }),
            z_score: CachePadded::new(AtomicU64::new(0)),
            spread_estimate: CachePadded::new(AtomicU64::new(0)),
            half_life_ms: CachePadded::new(AtomicU64::new(0)),
            enabled: CachePadded::new(AtomicBool::new(true)),
            update_count: CachePadded::new(AtomicU64::new(0)),
            entry_threshold_scaled: (entry_threshold * 1000.0) as i32,
            exit_threshold_scaled: (exit_threshold * 1000.0) as i32,
        }
    }

    /// Process a new spread observation
    /// 
    /// # Arguments
    /// * `spread` - Current observed spread value
    /// * `timestamp_ns` - Observation timestamp
    #[inline]
    pub fn observe(&self, spread: f64, timestamp_ns: u64) -> KalmanOutput {
        if !self.enabled.load(Ordering::Relaxed) {
            return self.get_current_output();
        }

        // Update spread filter
        let spread_output = self.update_spread_filter(spread);
        
        // Update theta (mean-reversion speed) filter
        self.update_theta_filter(spread, spread_output.estimate);

        // Calculate Z-score
        let z = self.calculate_zscore(spread);
        let z_scaled = (z * 1000.0) as i64;
        self.z_score.store(z_scaled.unsigned_abs(), Ordering::Relaxed);

        // Store spread estimate
        let spread_scaled = (spread_output.estimate * 1e9) as i64;
        self.spread_estimate.store(spread_scaled as u64, Ordering::Relaxed);

        // Calculate half-life
        let theta = self.theta_filter.theta.x;
        let half_life = if theta < 0.0 {
            (-0.693 / theta * 1000.0).max(0.0) as u64
        } else {
            0
        };
        self.half_life_ms.store(half_life, Ordering::Relaxed);

        self.update_count.fetch_add(1, Ordering::Relaxed);

        spread_output
    }

    #[inline]
    fn update_spread_filter(&self, measurement: f64) -> KalmanOutput {
        unsafe {
            let state_ptr = &self.spread_filter.spread_filter as *const KalmanState as *mut KalmanState;
            
            // Predict step
            let x_pred = (*state_ptr).x; // State transition is identity
            let p_pred = (*state_ptr).p + (*state_ptr).q;

            // Update step
            let k = p_pred / (p_pred + (*state_ptr).r); // Kalman gain
            let innovation = measurement - x_pred;
            let x_new = x_pred + k * innovation;
            let p_new = (1.0 - k) * p_pred;

            (*state_ptr).x = x_new;
            (*state_ptr).p = p_new;

            KalmanOutput {
                estimate: x_new,
                variance: p_new,
                kalman_gain: k,
                innovation,
            }
        }
    }

    #[inline]
    fn update_theta_filter(&self, actual_spread: f64, estimated_spread: f64) {
        // Theta represents mean-reversion speed in OU process
        // dX_t = theta * (mu - X_t) * dt + sigma * dW_t
        
        // Approximate theta from spread changes
        static mut PREV_SPREAD: f64 = 0.0;
        static mut PREV_ESTIMATE: f64 = 0.0;

        unsafe {
            let spread_change = actual_spread - PREV_SPREAD;
            let estimate_change = estimated_spread - PREV_ESTIMATE;
            
            // Simple approximation of mean-reversion speed
            let theta_observation = if PREV_ESTIMATE.abs() > 1e-10 {
                -spread_change / PREV_ESTIMATE
            } else {
                0.0
            };

            PREV_SPREAD = actual_spread;
            PREV_ESTIMATE = estimated_spread;

            let state_ptr = &self.theta_filter.theta_filter as *const KalmanState as *mut KalmanState;
            
            // Predict
            let x_pred = (*state_ptr).x;
            let p_pred = (*state_ptr).p + (*state_ptr).q;

            // Update
            let k = p_pred / (p_pred + (*state_ptr).r);
            let innovation = theta_observation - x_pred;
            (*state_ptr).x = x_pred + k * innovation;
            (*state_ptr).p = (1.0 - k) * p_pred;

            // Clamp theta to reasonable range
            if (*state_ptr).x > 0.0 {
                (*state_ptr).x = 0.0; // Must be negative for mean-reversion
            } else if (*state_ptr).x < -1.0 {
                (*state_ptr).x = -1.0;
            }
        }
    }

    #[inline]
    fn calculate_zscore(&self, current_spread: f64) -> f64 {
        let state = self.spread_filter.spread_filter;
        
        if state.p > 0.0 {
            (current_spread - state.x) / state.p.sqrt()
        } else {
            0.0
        }
    }

    #[inline]
    fn get_current_output(&self) -> KalmanOutput {
        let state = self.spread_filter.spread_filter;
        KalmanOutput {
            estimate: state.x,
            variance: state.p,
            kalman_gain: 0.0,
            innovation: 0.0,
        }
    }

    /// Get current Z-score
    #[inline]
    pub fn get_zscore(&self) -> f64 {
        let scaled = self.z_score.load(Ordering::Relaxed) as i64;
        scaled as f64 / 1000.0
    }

    /// Get spread estimate
    #[inline]
    pub fn get_spread_estimate(&self) -> f64 {
        let scaled = self.spread_estimate.load(Ordering::Relaxed) as i64;
        scaled as f64 / 1e9
    }

    /// Get mean-reversion half-life in milliseconds
    #[inline]
    pub fn get_half_life_ms(&self) -> u64 {
        self.half_life_ms.load(Ordering::Relaxed)
    }

    /// Check if entry signal is triggered
    #[inline]
    pub fn is_entry_signal(&self) -> bool {
        let z_scaled = self.z_score.load(Ordering::Relaxed) as i32;
        z_scaled >= self.entry_threshold_scaled || z_scaled <= -self.entry_threshold_scaled
    }

    /// Check if exit signal is triggered
    #[inline]
    pub fn is_exit_signal(&self) -> bool {
        let z_scaled = self.z_score.load(Ordering::Relaxed) as i32;
        z_scaled.abs() <= self.exit_threshold_scaled
    }

    /// Get trading signal: 1 = long spread, -1 = short spread, 0 = no position
    pub fn get_signal(&self) -> i8 {
        let z = self.get_zscore();
        
        if z >= self.entry_threshold_scaled as f64 / 1000.0 {
            -1 // Short spread (sell high, buy low)
        } else if z <= -(self.entry_threshold_scaled as f64 / 1000.0) {
            1 // Long spread (buy low, sell high)
        } else if z.abs() <= self.exit_threshold_scaled as f64 / 1000.0 {
            0 // Exit position
        } else {
            0 // Hold
        }
    }

    /// Get mean-reversion speed (theta)
    #[inline]
    pub fn get_theta(&self) -> f64 {
        self.theta_filter.theta_filter.x
    }

    /// Reset filters
    pub fn reset(&self) {
        self.spread_filter.spread_filter = KalmanState::default();
        self.theta_filter.theta_filter = KalmanState {
            x: -0.1,
            p: 0.1,
            q: 0.0001,
            r: 0.01,
        };
        self.z_score.store(0, Ordering::Relaxed);
        self.spread_estimate.store(0, Ordering::Relaxed);
        self.half_life_ms.store(0, Ordering::Relaxed);
        self.update_count.store(0, Ordering::Relaxed);
    }

    /// Enable filters
    pub fn enable(&self) {
        self.enabled.store(true, Ordering::Relaxed);
    }

    /// Disable filters
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
    fn test_kalman_spread_basic() {
        let engine = KalmanSpreadEngine::new(0.001, 0.1, 2.0, 0.5);
        
        // Feed some observations
        for i in 0..100 {
            let spread = (i as f64 * 0.1).sin() * 2.0;
            engine.observe(spread, i as u64 * 1_000_000);
        }
        
        let z = engine.get_zscore();
        assert!(z.is_finite());
        
        let estimate = engine.get_spread_estimate();
        assert!(estimate.is_finite());
    }

    #[test]
    fn test_signal_generation() {
        let engine = KalmanSpreadEngine::new(0.001, 0.1, 2.0, 0.5);
        
        // Create large deviation
        for _ in 0..50 {
            engine.observe(5.0, 1_000_000);
        }
        
        let signal = engine.get_signal();
        assert_ne!(signal, 0); // Should have a signal
    }
}
