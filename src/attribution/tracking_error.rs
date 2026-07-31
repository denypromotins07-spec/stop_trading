//! Real-Time Tracking Error Calculator
//! 
//! Computes ex-ante and ex-post tracking error against a custom crypto benchmark index.
//! Monitors strategy drift to ensure no uncompensated, hidden beta risks.

use std::sync::atomic::{AtomicU64, AtomicI64, Ordering};

/// Maximum history size for tracking error calculation
pub const MAX_HISTORY: usize = 512;

/// Minimum samples for valid tracking error
pub const MIN_SAMPLES: usize = 20;

/// Tracking error type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TrackingErrorType {
    ExAnte = 0,   // Forward-looking (predicted)
    ExPost = 1,   // Backward-looking (realized)
}

/// Tracking error result
#[derive(Debug, Clone)]
pub struct TrackingErrorResult {
    pub tracking_error: f64,      // Annualized tracking error (bps)
    pub active_risk: f64,         // Active risk contribution
    pub beta: f64,                // Portfolio beta to benchmark
    pub alpha: f64,               // Jensen's alpha
    pub correlation: f64,         // Correlation with benchmark
    pub information_ratio: f64,   // Risk-adjusted alpha
    pub r_squared: f64,           // Coefficient of determination
}

/// Rolling statistics tracker
struct RollingStats {
    values: [f64; MAX_HISTORY],
    head: usize,
    count: usize,
    sum: f64,
    sum_sq: f64,
}

impl RollingStats {
    const fn new() -> Self {
        Self {
            values: [0.0; MAX_HISTORY],
            head: 0,
            count: 0,
            sum: 0.0,
            sum_sq: 0.0,
        }
    }
    
    fn push(&mut self, value: f64) {
        if self.count < MAX_HISTORY {
            // Buffer not full yet
            if self.count == self.head {
                self.values[self.head] = value;
                self.sum += value;
                self.sum_sq += value * value;
                self.head += 1;
                self.count += 1;
            } else {
                self.values[self.head] = value;
                self.sum += value;
                self.sum_sq += value * value;
                self.head += 1;
                self.count += 1;
            }
        } else {
            // Buffer full, overwrite oldest
            let old_value = self.values[self.head];
            self.values[self.head] = value;
            self.sum -= old_value;
            self.sum_sq -= old_value * old_value;
            self.sum += value;
            self.sum_sq += value * value;
            self.head = (self.head + 1) % MAX_HISTORY;
        }
    }
    
    fn mean(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        self.sum / self.count as f64
    }
    
    fn variance(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }
        let mean = self.mean();
        (self.sum_sq / self.count as f64) - (mean * mean)
    }
    
    fn std(&self) -> f64 {
        self.variance().max(0.0).sqrt()
    }
    
    fn count(&self) -> usize {
        self.count
    }
}

/// Tracking error calculator engine
pub struct TrackingErrorCalculator {
    /// Portfolio returns ring buffer
    portfolio_returns: RollingStats,
    /// Benchmark returns ring buffer
    benchmark_returns: RollingStats,
    /// Active returns (portfolio - benchmark)
    active_returns: RollingStats,
    /// Squared active returns for variance
    squared_active: RollingStats,
    /// Covariance accumulator
    covariance_sum: f64,
    /// Product sum for correlation
    product_sum: f64,
    /// Annualization factor (periods per year)
    annualization_factor: f64,
    /// Type of tracking error being calculated
    error_type: TrackingErrorType,
}

impl TrackingErrorCalculator {
    pub const fn new() -> Self {
        Self {
            portfolio_returns: RollingStats::new(),
            benchmark_returns: RollingStats::new(),
            active_returns: RollingStats::new(),
            squared_active: RollingStats::new(),
            covariance_sum: 0.0,
            product_sum: 0.0,
            annualization_factor: 252.0, // Daily data
            error_type: TrackingErrorType::ExPost,
        }
    }
    
    /// Set annualization factor
    #[inline]
    pub fn set_annualization_factor(&mut self, factor: f64) {
        self.annualization_factor = factor;
    }
    
    /// Set tracking error type
    #[inline]
    pub fn set_error_type(&mut self, error_type: TrackingErrorType) {
        self.error_type = error_type;
    }
    
    /// Record a new return observation
    pub fn record_return(&mut self, portfolio_return: f64, benchmark_return: f64) {
        let active_return = portfolio_return - benchmark_return;
        
        self.portfolio_returns.push(portfolio_return);
        self.benchmark_returns.push(benchmark_return);
        self.active_returns.push(active_return);
        self.squared_active.push(active_return * active_return);
        
        // Update covariance sum
        if self.portfolio_returns.count() > 1 {
            let port_mean = self.portfolio_returns.mean();
            let bench_mean = self.benchmark_returns.mean();
            self.covariance_sum += (portfolio_return - port_mean) * (benchmark_return - bench_mean);
            self.product_sum += portfolio_return * benchmark_return;
        }
    }
    
    /// Calculate ex-post tracking error (realized)
    pub fn calculate_ex_post(&self) -> Option<TrackingErrorResult> {
        if self.active_returns.count() < MIN_SAMPLES {
            return None;
        }
        
        // Tracking error = std(active returns) * sqrt(annualization)
        let active_std = self.active_returns.std();
        let tracking_error = active_std * self.annualization_factor.sqrt() * 10000.0; // Convert to bps
        
        // Calculate beta and alpha
        let (beta, alpha) = self.calculate_beta_alpha();
        
        // Calculate correlation
        let correlation = self.calculate_correlation();
        
        // Information ratio
        let ir = if tracking_error > 0.0 {
            (alpha / 10000.0) / (tracking_error / 10000.0)
        } else {
            0.0
        };
        
        // R-squared
        let r_squared = correlation * correlation;
        
        Some(TrackingErrorResult {
            tracking_error,
            active_risk: tracking_error,
            beta,
            alpha,
            correlation,
            information_ratio: ir,
            r_squared,
        })
    }
    
    /// Calculate ex-ante tracking error (predicted from factor exposures)
    pub fn calculate_ex_ante(&self, factor_volatilities: &[f64], factor_loadings: &[f64]) -> f64 {
        if factor_volatilities.len() != factor_loadings.len() {
            return 0.0;
        }
        
        // Ex-ante TE = sqrt(sum((factor_loading * factor_vol)^2))
        let mut variance = 0.0;
        for i in 0..factor_volatilities.len() {
            let contribution = factor_loadings[i] * factor_volatilities[i];
            variance += contribution * contribution;
        }
        
        variance.sqrt() * self.annualization_factor.sqrt() * 10000.0
    }
    
    /// Calculate portfolio beta and Jensen's alpha
    fn calculate_beta_alpha(&self) -> (f64, f64) {
        let n = self.portfolio_returns.count().min(self.benchmark_returns.count());
        if n < MIN_SAMPLES {
            return (1.0, 0.0);
        }
        
        let port_mean = self.portfolio_returns.mean();
        let bench_mean = self.benchmark_returns.mean();
        let port_std = self.portfolio_returns.std();
        let bench_std = self.benchmark_returns.std();
        
        if bench_std < 1e-10 {
            return (1.0, port_mean - bench_mean);
        }
        
        let correlation = self.calculate_correlation();
        let beta = correlation * (port_std / bench_std);
        
        // Jensen's alpha = Rp - [Rf + beta * (Rm - Rf)]
        // Assuming Rf = 0 for crypto
        let alpha = (port_mean - beta * bench_mean) * self.annualization_factor * 100.0; // Annualized %
        
        (beta, alpha)
    }
    
    /// Calculate correlation between portfolio and benchmark
    fn calculate_correlation(&self) -> f64 {
        let n = self.portfolio_returns.count().min(self.benchmark_returns.count());
        if n < 2 {
            return 0.0;
        }
        
        let port_std = self.portfolio_returns.std();
        let bench_std = self.benchmark_returns.std();
        
        if port_std < 1e-10 || bench_std < 1e-10 {
            return 0.0;
        }
        
        // Covariance / (std_port * std_bench)
        let cov = self.covariance_sum / n as f64;
        cov / (port_std * bench_std)
    }
    
    /// Get current active return (latest)
    pub fn current_active_return(&self) -> f64 {
        if self.active_returns.count() == 0 {
            return 0.0;
        }
        let idx = if self.active_returns.head == 0 {
            MAX_HISTORY - 1
        } else {
            self.active_returns.head - 1
        };
        self.active_returns.values[idx]
    }
    
    /// Get cumulative active return (YTD style)
    pub fn cumulative_active_return(&self) -> f64 {
        let mut cumulative = 0.0;
        for i in 0..self.active_returns.count() {
            let idx = (self.active_returns.head + i) % MAX_HISTORY;
            cumulative += self.active_returns.values[idx];
        }
        cumulative
    }
    
    /// Check if tracking error exceeds threshold
    pub fn check_threshold(&self, threshold_bps: f64) -> bool {
        if let Some(result) = self.calculate_ex_post() {
            result.tracking_error > threshold_bps
        } else {
            false
        }
    }
    
    /// Get number of samples
    #[inline]
    pub fn sample_count(&self) -> usize {
        self.active_returns.count()
    }
    
    /// Reset calculator
    pub fn reset(&mut self) {
        self.portfolio_returns = RollingStats::new();
        self.benchmark_returns = RollingStats::new();
        self.active_returns = RollingStats::new();
        self.squared_active = RollingStats::new();
        self.covariance_sum = 0.0;
        self.product_sum = 0.0;
    }
}

/// Benchmark index composition
#[derive(Debug, Clone)]
pub struct BenchmarkIndex {
    pub name: &'static str,
    pub components: [(&'static str, f64); 32],
    pub component_count: usize,
    pub rebalance_frequency_days: u32,
}

impl BenchmarkIndex {
    pub const fn new(name: &'static str) -> Self {
        Self {
            name,
            components: [("", 0.0); 32],
            component_count: 0,
            rebalance_frequency_days: 30,
        }
    }
    
    pub fn add_component(&mut self, symbol: &'static str, weight: f64) -> bool {
        if self.component_count >= 32 {
            return false;
        }
        self.components[self.component_count] = (symbol, weight);
        self.component_count += 1;
        true
    }
    
    pub fn normalize_weights(&mut self) {
        let total: f64 = self.components[..self.component_count]
            .iter()
            .map(|(_, w)| w)
            .sum();
        
        if total > 0.0 {
            for i in 0..self.component_count {
                self.components[i].1 /= total;
            }
        }
    }
}

/// Drift detection alert
#[derive(Debug, Clone)]
pub struct DriftAlert {
    pub severity: DriftSeverity,
    pub current_te: f64,
    pub target_te: f64,
    pub deviation_bps: f64,
    pub timestamp_ns: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum DriftSeverity {
    Normal = 0,
    Warning = 1,
    Critical = 2,
    Breach = 3,
}

/// Drift monitor for continuous tracking
pub struct DriftMonitor {
    calculator: TrackingErrorCalculator,
    target_tracking_error: f64,
    warning_threshold: f64,
    critical_threshold: f64,
    breach_threshold: f64,
    last_alert: Option<DriftAlert>,
}

impl DriftMonitor {
    pub fn new(target_te_bps: f64) -> Self {
        Self {
            calculator: TrackingErrorCalculator::new(),
            target_tracking_error: target_te_bps,
            warning_threshold: target_te_bps * 1.5,
            critical_threshold: target_te_bps * 2.0,
            breach_threshold: target_te_bps * 3.0,
            last_alert: None,
        }
    }
    
    /// Record return and check for drift
    pub fn record_and_check(&mut self, portfolio_return: f64, benchmark_return: f64) -> Option<DriftAlert> {
        self.calculator.record_return(portfolio_return, benchmark_return);
        
        if let Some(result) = self.calculator.calculate_ex_post() {
            let current_te = result.tracking_error;
            
            let severity = if current_te > self.breach_threshold {
                DriftSeverity::Breach
            } else if current_te > self.critical_threshold {
                DriftSeverity::Critical
            } else if current_te > self.warning_threshold {
                DriftSeverity::Warning
            } else {
                DriftSeverity::Normal
            };
            
            let alert = DriftAlert {
                severity,
                current_te,
                target_te: self.target_tracking_error,
                deviation_bps: current_te - self.target_tracking_error,
                timestamp_ns: get_timestamp_ns(),
            };
            
            // Only return alert if severity changed or increased
            if self.last_alert.is_none() || 
               severity as u8 >= self.last_alert.as_ref().unwrap().severity as u8 {
                self.last_alert = Some(alert.clone());
                return Some(alert);
            }
        }
        
        None
    }
    
    /// Get current tracking error
    pub fn current_tracking_error(&self) -> Option<f64> {
        self.calculator.calculate_ex_post().map(|r| r.tracking_error)
    }
    
    /// Update thresholds
    pub fn update_thresholds(&mut self, target_te: f64) {
        self.target_tracking_error = target_te;
        self.warning_threshold = target_te * 1.5;
        self.critical_threshold = target_te * 2.0;
        self.breach_threshold = target_te * 3.0;
    }
}

/// Get current timestamp in nanoseconds
#[inline]
fn get_timestamp_ns() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_tracking_error_calculation() {
        let mut calc = TrackingErrorCalculator::new();
        
        // Simulate correlated returns with some active risk
        for i in 0..50 {
            let bench_ret = 0.001 * ((i % 10) as f64 - 5.0) / 5.0;
            let port_ret = bench_ret + 0.0005 * ((i % 7) as f64 - 3.0) / 3.0;
            calc.record_return(port_ret, bench_ret);
        }
        
        let result = calc.calculate_ex_post().unwrap();
        
        assert!(result.tracking_error > 0.0);
        assert!(result.correlation > 0.5);
        assert!(result.r_squared > 0.25);
    }
    
    #[test]
    fn test_drift_monitor() {
        let mut monitor = DriftMonitor::new(100.0); // Target 100 bps TE
        
        // Normal period - low tracking error
        for i in 0..30 {
            let ret = 0.001 * ((i % 5) as f64 - 2.0) / 2.0;
            let _ = monitor.record_and_check(ret + 0.0001, ret);
        }
        
        // Should be normal or warning at most
        if let Some(te) = monitor.current_tracking_error() {
            assert!(te < 300.0); // Should be under critical
        }
    }
    
    #[test]
    fn test_beta_calculation() {
        let mut calc = TrackingErrorCalculator::new();
        
        // Create high-beta portfolio (moves 1.5x benchmark)
        for i in 0..50 {
            let bench_ret = 0.01 * ((i % 10) as f64 - 5.0) / 5.0;
            let port_ret = 1.5 * bench_ret + 0.001;
            calc.record_return(port_ret, bench_ret);
        }
        
        let result = calc.calculate_ex_post().unwrap();
        
        // Beta should be close to 1.5
        assert!(result.beta > 1.2);
        assert!(result.beta < 1.8);
    }
}
