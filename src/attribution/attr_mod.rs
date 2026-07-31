//! Attribution Module Root
//! 
//! Pushes granular performance analytics to Terminal UI and self-learning hooks.

pub mod brinson;
pub mod tracking_error;

pub use brinson::{BrinsonAttributor, AttributionEffects, PeriodAttribution, AssetAttribution};
pub use tracking_error::{TrackingErrorCalculator, TrackingErrorResult, DriftMonitor, DriftAlert, DriftSeverity, BenchmarkIndex};

/// Combined performance attribution engine
pub struct PerformanceAttribution {
    pub brinson: BrinsonAttributor,
    pub tracking_error: TrackingErrorCalculator,
    pub drift_monitor: DriftMonitor,
}

impl PerformanceAttribution {
    pub fn new(target_tracking_error_bps: f64) -> Self {
        Self {
            brinson: BrinsonAttributor::new(),
            tracking_error: TrackingErrorCalculator::new(),
            drift_monitor: DriftMonitor::new(target_tracking_error_bps),
        }
    }
    
    /// Record period returns and compute all attribution metrics
    pub fn record_period(&mut self, portfolio_return: f64, benchmark_return: f64) -> AttributionReport {
        // Record for tracking error
        self.tracking_error.record_return(portfolio_return, benchmark_return);
        
        // Check drift
        let drift_alert = self.drift_monitor.record_and_check(portfolio_return, benchmark_return);
        
        // Compute Brinson attribution (weights should be set externally)
        let brinson_result = self.brinson.compute_attribution();
        
        // Get tracking error result
        let te_result = self.tracking_error.calculate_ex_post();
        
        AttributionReport {
            portfolio_return,
            benchmark_return,
            active_return: portfolio_return - benchmark_return,
            brinson_effects: brinson_result.effects,
            tracking_error: te_result.map(|r| r.tracking_error),
            beta: te_result.map(|r| r.beta),
            alpha: te_result.map(|r| r.alpha),
            information_ratio: te_result.map(|r| r.information_ratio),
            drift_alert,
        }
    }
    
    /// Set asset-level data for Brinson attribution
    pub fn set_asset_data(&self, asset_idx: usize, port_weight: f64, bench_weight: f64, port_ret: f64, bench_ret: f64) {
        self.brinson.set_portfolio_weight(asset_idx, port_weight);
        self.brinson.set_benchmark_weight(asset_idx, bench_weight);
        self.brinson.set_asset_return(asset_idx, port_ret);
        self.brinson.set_benchmark_asset_return(asset_idx, bench_ret);
    }
    
    /// Get current attribution summary
    pub fn get_summary(&self) -> AttributionSummary {
        let te = self.tracking_error.calculate_ex_post();
        let latest_brinson = self.brinson.latest_attribution();
        
        AttributionSummary {
            avg_allocation: self.brinson.avg_allocation_effect(),
            avg_selection: self.brinson.avg_selection_effect(),
            avg_interaction: self.brinson.avg_interaction_effect(),
            tracking_error: te.map(|r| r.tracking_error).unwrap_or(0.0),
            information_ratio: te.map(|r| r.information_ratio).unwrap_or(0.0),
            total_active_return: latest_brinson.map(|r| r.active_return).unwrap_or(0.0),
        }
    }
    
    /// Reset all attribution state
    pub fn reset(&mut self) {
        self.brinson.reset();
        self.tracking_error.reset();
        self.drift_monitor = DriftMonitor::new(self.drift_monitor.current_tracking_error().unwrap_or(100.0));
    }
}

/// Comprehensive attribution report
#[derive(Debug, Clone)]
pub struct AttributionReport {
    pub portfolio_return: f64,
    pub benchmark_return: f64,
    pub active_return: f64,
    pub brinson_effects: AttributionEffects,
    pub tracking_error: Option<f64>,
    pub beta: Option<f64>,
    pub alpha: Option<f64>,
    pub information_ratio: Option<f64>,
    pub drift_alert: Option<DriftAlert>,
}

/// Attribution summary for dashboard display
#[derive(Debug, Clone)]
pub struct AttributionSummary {
    pub avg_allocation: f64,
    pub avg_selection: f64,
    pub avg_interaction: f64,
    pub tracking_error: f64,
    pub information_ratio: f64,
    pub total_active_return: f64,
}

/// Attribution export format for SOUL.md integration
#[derive(Debug, Clone)]
pub struct AttributionExport {
    pub timestamp_ns: u64,
    pub period_id: u64,
    pub allocation_effect: f64,
    pub selection_effect: f64,
    pub interaction_effect: f64,
    pub tracking_error_bps: f64,
    pub information_ratio: f64,
    pub active_return_bps: f64,
}

impl AttributionExport {
    pub fn from_report(report: &AttributionReport, period_id: u64) -> Self {
        Self {
            timestamp_ns: get_timestamp_ns(),
            period_id,
            allocation_effect: report.brinson_effects.allocation_effect,
            selection_effect: report.brinson_effects.selection_effect,
            interaction_effect: report.brinson_effects.interaction_effect,
            tracking_error_bps: report.tracking_error.unwrap_or(0.0),
            information_ratio: report.information_ratio.unwrap_or(0.0),
            active_return_bps: report.active_return * 10000.0,
        }
    }
    
    /// Serialize to JSON-like format for logging
    pub fn to_log_string(&self) -> String {
        format!(
            "{{\"ts\":{},\"period\":{},\"alloc\":{:.6},\"sel\":{:.6},\"int\":{:.6},\"te\":{:.2},\"ir\":{:.4},\"ar\":{:.2}}}",
            self.timestamp_ns,
            self.period_id,
            self.allocation_effect,
            self.selection_effect,
            self.interaction_effect,
            self.tracking_error_bps,
            self.information_ratio,
            self.active_return_bps
        )
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
    fn test_performance_attribution() {
        let mut attr = PerformanceAttribution::new(100.0);
        
        // Setup asset data
        attr.brinson.set_asset_count(3);
        attr.set_asset_data(0, 0.5, 0.33, 0.02, 0.015);
        attr.set_asset_data(1, 0.3, 0.33, 0.01, 0.01);
        attr.set_asset_data(2, 0.2, 0.34, -0.01, 0.005);
        
        // Record multiple periods
        for i in 0..30 {
            let port_ret = 0.001 + (i as f64 * 0.0001);
            let bench_ret = 0.0008 + (i as f64 * 0.00005);
            let _ = attr.record_period(port_ret, bench_ret);
        }
        
        // Get summary
        let summary = attr.get_summary();
        
        assert!(summary.tracking_error > 0.0);
        assert!(summary.total_active_return > 0.0);
    }
    
    #[test]
    fn test_attribution_export() {
        let effects = AttributionEffects {
            allocation_effect: 0.001,
            selection_effect: 0.002,
            interaction_effect: 0.0005,
            total_active_return: 0.0035,
        };
        
        let report = AttributionReport {
            portfolio_return: 0.02,
            benchmark_return: 0.015,
            active_return: 0.005,
            brinson_effects: effects,
            tracking_error: Some(50.0),
            beta: Some(1.1),
            alpha: Some(2.5),
            information_ratio: Some(0.5),
            drift_alert: None,
        };
        
        let export = AttributionExport::from_report(&report, 1);
        let log_str = export.to_log_string();
        
        assert!(log_str.contains("\"alloc\":"));
        assert!(log_str.contains("\"sel\":"));
        assert!(log_str.contains("\"te\":50.00"));
    }
}
