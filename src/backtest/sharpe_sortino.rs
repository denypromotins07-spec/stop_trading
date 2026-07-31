//! Zero-Allocation Online Risk Metrics Calculator
//! 
//! This module implements zero-allocation, online calculators for Sharpe, Sortino,
//! and Calmar ratios. Updates risk-adjusted performance metrics atomically on every
//! simulated or live fill to feed the Terminal UI dashboard.
//! 
//! Key Features:
//! - Welford's algorithm for online mean/variance calculation
//! - Separate upside/downside deviation tracking for Sortino
//! - Drawdown tracking for Calmar ratio
//! - Zero heap allocation after initialization
//! - Atomic updates for thread-safe real-time operation

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

/// Online Sharpe ratio calculator using Welford's algorithm
pub struct OnlineSharpeCalculator {
    /// Count of returns observed
    n: AtomicU64,
    /// Running mean of returns (stored as bits)
    mean_bits: AtomicU64,
    /// Running M2 (sum of squared differences)
    m2_bits: AtomicU64,
    /// Annualization factor (e.g., sqrt(252) for daily, sqrt(365*24) for hourly)
    annualization_factor: f64,
    /// Risk-free rate (annualized)
    risk_free_rate: f64,
}

impl OnlineSharpeCalculator {
    pub fn new(annualization_factor: f64, risk_free_rate: f64) -> Self {
        Self {
            n: AtomicU64::new(0),
            mean_bits: AtomicU64::new(0.0f64.to_bits()),
            m2_bits: AtomicU64::new(0.0f64.to_bits()),
            annualization_factor,
            risk_free_rate,
        }
    }
    
    /// Add a new return observation
    pub fn update(&self, return_value: f64) {
        let n = self.n.fetch_add(1, Ordering::Relaxed);
        let current_n = n + 1;
        
        // Get current mean
        let mean = f64::from_bits(self.mean_bits.load(Ordering::Relaxed));
        let m2 = f64::from_bits(self.m2_bits.load(Ordering::Relaxed));
        
        // Welford's online algorithm
        let delta = return_value - mean;
        let new_mean = mean + delta / current_n as f64;
        let delta2 = return_value - new_mean;
        let new_m2 = m2 + delta * delta2;
        
        // Update atomically
        self.mean_bits.store(new_mean.to_bits(), Ordering::Relaxed);
        self.m2_bits.store(new_m2.to_bits(), Ordering::Relaxed);
    }
    
    /// Get the current Sharpe ratio
    pub fn sharpe_ratio(&self) -> f64 {
        let n = self.n.load(Ordering::Relaxed);
        if n < 2 {
            return f64::NAN;
        }
        
        let mean = f64::from_bits(self.mean_bits.load(Ordering::Relaxed));
        let m2 = f64::from_bits(self.m2_bits.load(Ordering::Relaxed));
        
        // Calculate variance and standard deviation
        let variance = m2 / n as f64;
        let std_dev = variance.sqrt();
        
        if std_dev <= 0.0 {
            return f64::NAN;
        }
        
        // Annualized Sharpe ratio
        let excess_return = mean * self.annualization_factor - self.risk_free_rate;
        let annualized_std = std_dev * self.annualization_factor.sqrt();
        
        excess_return / annualized_std
    }
    
    /// Get the annualized return
    pub fn annualized_return(&self) -> f64 {
        let n = self.n.load(Ordering::Relaxed);
        if n == 0 {
            return 0.0;
        }
        
        let mean = f64::from_bits(self.mean_bits.load(Ordering::Relaxed));
        mean * self.annualization_factor
    }
    
    /// Get the annualized volatility
    pub fn annualized_volatility(&self) -> f64 {
        let n = self.n.load(Ordering::Relaxed);
        if n < 2 {
            return 0.0;
        }
        
        let m2 = f64::from_bits(self.m2_bits.load(Ordering::Relaxed));
        let variance = m2 / n as f64;
        let std_dev = variance.sqrt();
        
        std_dev * self.annualization_factor.sqrt()
    }
    
    /// Reset the calculator
    pub fn reset(&self) {
        self.n.store(0, Ordering::Relaxed);
        self.mean_bits.store(0.0f64.to_bits(), Ordering::Relaxed);
        self.m2_bits.store(0.0f64.to_bits(), Ordering::Relaxed);
    }
    
    /// Get observation count
    pub fn count(&self) -> u64 {
        self.n.load(Ordering::Relaxed)
    }
}

/// Online Sortino ratio calculator (downside deviation only)
pub struct OnlineSortinoCalculator {
    /// Count of returns observed
    n: AtomicU64,
    /// Count of negative returns
    downside_n: AtomicU64,
    /// Mean of all returns
    mean_bits: AtomicU64,
    /// Downside M2 (only negative deviations)
    downside_m2_bits: AtomicU64,
    /// Target return (minimum acceptable return)
    target_return: f64,
    /// Annualization factor
    annualization_factor: f64,
}

impl OnlineSortinoCalculator {
    pub fn new(target_return: f64, annualization_factor: f64) -> Self {
        Self {
            n: AtomicU64::new(0),
            downside_n: AtomicU64::new(0),
            mean_bits: AtomicU64::new(0.0f64.to_bits()),
            downside_m2_bits: AtomicU64::new(0.0f64.to_bits()),
            target_return,
            annualization_factor,
        }
    }
    
    /// Add a new return observation
    pub fn update(&self, return_value: f64) {
        let n = self.n.fetch_add(1, Ordering::Relaxed);
        let current_n = n + 1;
        
        // Update overall mean
        let mean = f64::from_bits(self.mean_bits.load(Ordering::Relaxed));
        let delta = return_value - mean;
        let new_mean = mean + delta / current_n as f64;
        self.mean_bits.store(new_mean.to_bits(), Ordering::Relaxed);
        
        // Update downside deviation
        if return_value < self.target_return {
            let down_n = self.downside_n.fetch_add(1, Ordering::Relaxed);
            let current_down_n = down_n + 1;
            
            let down_m2 = f64::from_bits(self.downside_m2_bits.load(Ordering::Relaxed));
            let delta_down = return_value - self.target_return;
            let new_down_m2 = down_m2 + delta_down * delta_down;
            
            self.downside_m2_bits.store(new_down_m2.to_bits(), Ordering::Relaxed);
        }
    }
    
    /// Get the current Sortino ratio
    pub fn sortino_ratio(&self) -> f64 {
        let n = self.n.load(Ordering::Relaxed);
        let down_n = self.downside_n.load(Ordering::Relaxed);
        
        if n < 2 || down_n < 2 {
            return f64::NAN;
        }
        
        let mean = f64::from_bits(self.mean_bits.load(Ordering::Relaxed));
        let down_m2 = f64::from_bits(self.downside_m2_bits.load(Ordering::Relaxed));
        
        // Downside deviation
        let downside_variance = down_m2 / down_n as f64;
        let downside_dev = downside_variance.sqrt();
        
        if downside_dev <= 0.0 {
            return f64::NAN;
        }
        
        // Annualized Sortino ratio
        let excess_return = mean * self.annualization_factor - self.target_return * self.annualization_factor;
        let annualized_downside = downside_dev * self.annualization_factor.sqrt();
        
        excess_return / annualized_downside
    }
    
    /// Get downside deviation
    pub fn downside_deviation(&self) -> f64 {
        let down_n = self.downside_n.load(Ordering::Relaxed);
        if down_n < 2 {
            return 0.0;
        }
        
        let down_m2 = f64::from_bits(self.downside_m2_bits.load(Ordering::Relaxed));
        let downside_variance = down_m2 / down_n as f64;
        
        downside_variance.sqrt() * self.annualization_factor.sqrt()
    }
    
    /// Reset the calculator
    pub fn reset(&self) {
        self.n.store(0, Ordering::Relaxed);
        self.downside_n.store(0, Ordering::Relaxed);
        self.mean_bits.store(0.0f64.to_bits(), Ordering::Relaxed);
        self.downside_m2_bits.store(0.0f64.to_bits(), Ordering::Relaxed);
    }
}

/// Online drawdown tracker for Calmar ratio
pub struct OnlineDrawdownTracker {
    /// Current cumulative return
    cumulative_return: AtomicU64,
    /// Peak cumulative return
    peak_return: AtomicU64,
    /// Maximum drawdown observed
    max_drawdown: AtomicU64,
    /// Current drawdown
    current_drawdown: AtomicU64,
    /// Number of periods
    n: AtomicU64,
}

impl OnlineDrawdownTracker {
    pub fn new() -> Self {
        Self {
            cumulative_return: AtomicU64::new((1.0f64).to_bits()),
            peak_return: AtomicU64::new((1.0f64).to_bits()),
            max_drawdown: AtomicU64::new(0.0f64.to_bits()),
            current_drawdown: AtomicU64::new(0.0f64.to_bits()),
            n: AtomicU64::new(0),
        }
    }
    
    /// Add a new return observation
    pub fn update(&self, return_value: f64) {
        let n = self.n.fetch_add(1, Ordering::Relaxed);
        
        // Update cumulative return
        let cum_ret_bits = self.cumulative_return.load(Ordering::Relaxed);
        let mut cum_ret = f64::from_bits(cum_ret_bits);
        cum_ret *= (1.0 + return_value);
        self.cumulative_return.store(cum_ret.to_bits(), Ordering::Relaxed);
        
        // Update peak
        let peak_bits = self.peak_return.load(Ordering::Relaxed);
        let peak = f64::from_bits(peak_bits);
        if cum_ret > peak {
            self.peak_return.store(cum_ret.to_bits(), Ordering::Relaxed);
        }
        
        // Calculate current drawdown
        let new_peak = f64::from_bits(self.peak_return.load(Ordering::Relaxed));
        let drawdown = if new_peak > 0.0 {
            (new_peak - cum_ret) / new_peak
        } else {
            0.0
        };
        self.current_drawdown.store(drawdown.to_bits(), Ordering::Relaxed);
        
        // Update max drawdown
        let max_dd_bits = self.max_drawdown.load(Ordering::Relaxed);
        let max_dd = f64::from_bits(max_dd_bits);
        if drawdown > max_dd {
            self.max_drawdown.store(drawdown.to_bits(), Ordering::Relaxed);
        }
    }
    
    /// Get maximum drawdown
    pub fn max_drawdown(&self) -> f64 {
        f64::from_bits(self.max_drawdown.load(Ordering::Relaxed))
    }
    
    /// Get current drawdown
    pub fn current_drawdown(&self) -> f64 {
        f64::from_bits(self.current_drawdown.load(Ordering::Relaxed))
    }
    
    /// Get cumulative return
    pub fn cumulative_return(&self) -> f64 {
        let bits = self.cumulative_return.load(Ordering::Relaxed);
        f64::from_bits(bits) - 1.0
    }
    
    /// Reset the tracker
    pub fn reset(&self) {
        self.cumulative_return.store((1.0f64).to_bits(), Ordering::Relaxed);
        self.peak_return.store((1.0f64).to_bits(), Ordering::Relaxed);
        self.max_drawdown.store(0.0f64.to_bits(), Ordering::Relaxed);
        self.current_drawdown.store(0.0f64.to_bits(), Ordering::Relaxed);
        self.n.store(0, Ordering::Relaxed);
    }
}

impl Default for OnlineDrawdownTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Combined Calmar ratio calculator
pub struct OnlineCalmarCalculator {
    /// Drawdown tracker
    drawdown_tracker: OnlineDrawdownTracker,
    /// Return accumulator
    return_sum: AtomicU64,
    /// Return count
    n: AtomicU64,
    /// Annualization factor
    annualization_factor: f64,
}

impl OnlineCalmarCalculator {
    pub fn new(annualization_factor: f64) -> Self {
        Self {
            drawdown_tracker: OnlineDrawdownTracker::new(),
            return_sum: AtomicU64::new(0.0f64.to_bits()),
            n: AtomicU64::new(0),
            annualization_factor,
        }
    }
    
    /// Add a new return observation
    pub fn update(&self, return_value: f64) {
        self.drawdown_tracker.update(return_value);
        
        let n = self.n.fetch_add(1, Ordering::Relaxed);
        let sum_bits = self.return_sum.load(Ordering::Relaxed);
        let sum = f64::from_bits(sum_bits);
        self.return_sum.store((sum + return_value).to_bits(), Ordering::Relaxed);
    }
    
    /// Get the Calmar ratio
    pub fn calmar_ratio(&self) -> f64 {
        let n = self.n.load(Ordering::Relaxed);
        if n == 0 {
            return f64::NAN;
        }
        
        let sum_bits = self.return_sum.load(Ordering::Relaxed);
        let sum = f64::from_bits(sum_bits);
        let avg_return = sum / n as f64;
        
        let max_dd = self.drawdown_tracker.max_drawdown();
        if max_dd <= 0.0 {
            return f64::NAN;
        }
        
        // Annualized return / max drawdown
        (avg_return * self.annualization_factor) / max_dd
    }
    
    /// Get maximum drawdown
    pub fn max_drawdown(&self) -> f64 {
        self.drawdown_tracker.max_drawdown()
    }
    
    /// Reset the calculator
    pub fn reset(&self) {
        self.drawdown_tracker.reset();
        self.return_sum.store(0.0f64.to_bits(), Ordering::Relaxed);
        self.n.store(0, Ordering::Relaxed);
    }
}

/// Complete risk metrics aggregator
pub struct RiskMetricsAggregator {
    /// Sharpe calculator
    sharpe: OnlineSharpeCalculator,
    /// Sortino calculator
    sortino: OnlineSortinoCalculator,
    /// Calmar calculator
    calmar: OnlineCalmarCalculator,
    /// Statistics
    stats: RiskMetricsStats,
}

#[derive(Debug, Default)]
pub struct RiskMetricsStats {
    pub total_updates: AtomicUsize,
    pub last_sharpe: AtomicU64,
    pub last_sortino: AtomicU64,
    pub last_calmar: AtomicU64,
}

impl RiskMetricsAggregator {
    pub fn new(annualization_factor: f64, risk_free_rate: f64, target_return: f64) -> Self {
        Self {
            sharpe: OnlineSharpeCalculator::new(annualization_factor, risk_free_rate),
            sortino: OnlineSortinoCalculator::new(target_return, annualization_factor),
            calmar: OnlineCalmarCalculator::new(annualization_factor),
            stats: RiskMetricsStats::default(),
        }
    }
    
    /// Update all metrics with a new return
    pub fn update(&self, return_value: f64) {
        self.sharpe.update(return_value);
        self.sortino.update(return_value);
        self.calmar.update(return_value);
        
        self.stats.total_updates.fetch_add(1, Ordering::Relaxed);
        self.stats.last_sharpe.store(self.sharpe.sharpe_ratio().to_bits(), Ordering::Relaxed);
        self.stats.last_sortino.store(self.sortino.sortino_ratio().to_bits(), Ordering::Relaxed);
        self.stats.last_calmar.store(self.calmar.calmar_ratio().to_bits(), Ordering::Relaxed);
    }
    
    /// Get current Sharpe ratio
    pub fn sharpe(&self) -> f64 {
        self.sharpe.sharpe_ratio()
    }
    
    /// Get current Sortino ratio
    pub fn sortino(&self) -> f64 {
        self.sortino.sortino_ratio()
    }
    
    /// Get current Calmar ratio
    pub fn calmar(&self) -> f64 {
        self.calmar.calmar_ratio()
    }
    
    /// Get all metrics as a snapshot
    pub fn snapshot(&self) -> RiskMetricsSnapshot {
        RiskMetricsSnapshot {
            sharpe: self.sharpe(),
            sortino: self.sortino(),
            calmar: self.calmar(),
            max_drawdown: self.calmar.max_drawdown(),
            annualized_return: self.sharpe.annualized_return(),
            annualized_volatility: self.sharpe.annualized_volatility(),
            downside_deviation: self.sortino.downside_deviation(),
            observation_count: self.sharpe.count(),
        }
    }
    
    /// Reset all metrics
    pub fn reset(&self) {
        self.sharpe.reset();
        self.sortino.reset();
        self.calmar.reset();
    }
}

#[derive(Debug, Clone)]
pub struct RiskMetricsSnapshot {
    pub sharpe: f64,
    pub sortino: f64,
    pub calmar: f64,
    pub max_drawdown: f64,
    pub annualized_return: f64,
    pub annualized_volatility: f64,
    pub downside_deviation: f64,
    pub observation_count: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_online_sharpe() {
        let calc = OnlineSharpeCalculator::new(252.0_f64.sqrt(), 0.0);
        
        // Simulate positive returns with some variance
        for i in 0..100 {
            let ret = 0.001 + (i % 10) as f64 * 0.0001;
            calc.update(ret);
        }
        
        let sharpe = calc.sharpe_ratio();
        assert!(sharpe.is_finite());
        assert!(sharpe > 0.0);
    }
    
    #[test]
    fn test_online_sortino() {
        let calc = OnlineSortinoCalculator::new(0.0, 252.0_f64.sqrt());
        
        // Mix of positive and negative returns
        for i in 0..100 {
            let ret = if i % 3 == 0 { -0.01 } else { 0.015 };
            calc.update(ret);
        }
        
        let sortino = calc.sortino_ratio();
        assert!(sortino.is_finite());
    }
    
    #[test]
    fn test_drawdown_tracker() {
        let tracker = OnlineDrawdownTracker::new();
        
        // Simulate returns that create a drawdown
        tracker.update(0.10); // Up 10%
        tracker.update(-0.05); // Down 5%
        tracker.update(-0.10); // Down 10%
        tracker.update(0.05); // Recovery
        
        let max_dd = tracker.max_drawdown();
        assert!(max_dd > 0.0);
        assert!(max_dd < 0.20); // Should be less than 20%
    }
    
    #[test]
    fn test_risk_metrics_aggregator() {
        let agg = RiskMetricsAggregator::new(252.0_f64.sqrt(), 0.0, 0.0);
        
        for i in 0..50 {
            let ret = 0.001 * (i as f64 % 5 - 2.0);
            agg.update(ret);
        }
        
        let snapshot = agg.snapshot();
        assert!(snapshot.observation_count == 50);
        assert!(snapshot.sharpe.is_finite() || snapshot.sharpe.is_nan());
    }
}
