//! Risk control module root.
//! 
//! Aggregates stress and drawdown metrics for the Terminal UI dashboard.

pub mod stress;
pub mod drawdown;

pub use stress::{
    StressTestEngine, StressMonitor, StressTestResult, StressPosition,
    CrashScenario, StressAlert,
};
pub use drawdown::{
    DrawdownTracker, PositionScaler, DrawdownState, CircuitBreakerStatus,
    CircuitBreakerTier,
};

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};

/// Aggregated risk control data for dashboard
#[derive(Debug, Clone)]
pub struct RiskControlDashboard {
    /// Current VaR
    pub var: f64,
    /// Current CVaR
    pub cvar: f64,
    /// Current drawdown
    pub drawdown: f64,
    /// Maximum drawdown
    pub max_drawdown: f64,
    /// Circuit breaker status
    pub circuit_breaker_active: bool,
    /// Size multiplier
    pub size_multiplier: f64,
    /// Worst stress scenario
    pub worst_stress_scenario: String,
    /// Worst stress drawdown
    pub worst_stress_drawdown: f64,
    /// Risk level indicator
    pub risk_level: RiskLevel,
    /// Timestamp
    pub timestamp_ns: u64,
}

/// Overall risk level classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiskLevel {
    /// Low risk - normal operations
    Low,
    /// Moderate risk - increased monitoring
    Moderate,
    /// Elevated risk - reduced sizing
    Elevated,
    /// High risk - significant restrictions
    High,
    /// Critical risk - trading halted
    Critical,
}

impl RiskLevel {
    /// Determine risk level from metrics
    pub fn from_metrics(drawdown: f64, var: f64, cvar_var_ratio: f64) -> Self {
        let drawdown_severity = if drawdown < -0.20 {
            4
        } else if drawdown < -0.15 {
            3
        } else if drawdown < -0.10 {
            2
        } else if drawdown < -0.05 {
            1
        } else {
            0
        };
        
        let tail_risk_severity = if cvar_var_ratio > 3.0 {
            4
        } else if cvar_var_ratio > 2.0 {
            3
        } else if cvar_var_ratio > 1.5 {
            2
        } else if cvar_var_ratio > 1.2 {
            1
        } else {
            0
        };
        
        let combined = (drawdown_severity + tail_risk_severity) / 2;
        
        match combined {
            0 => RiskLevel::Low,
            1 => RiskLevel::Moderate,
            2 => RiskLevel::Elevated,
            3 => RiskLevel::High,
            _ => RiskLevel::Critical,
        }
    }
    
    /// Get color code for UI
    pub fn color_code(&self) -> &'static str {
        match self {
            RiskLevel::Low => "#22c55e",      // Green
            RiskLevel::Moderate => "#84cc1e", // Lime
            RiskLevel::Elevated => "#fbbf24", // Amber
            RiskLevel::High => "#f97316",     // Orange
            RiskLevel::Critical => "#ef4444", // Red
        }
    }
    
    /// Get description
    pub fn description(&self) -> &'static str {
        match self {
            RiskLevel::Low => "Normal operations - all systems green",
            RiskLevel::Moderate => "Increased monitoring recommended",
            RiskLevel::Elevated => "Reduced position sizing active",
            RiskLevel::High => "Significant trading restrictions",
            RiskLevel::Critical => "Trading halted - manual intervention required",
        }
    }
}

/// Risk control manager coordinating all risk modules
pub struct RiskControlManager {
    /// Stress test engine
    stress_engine: StressTestEngine,
    /// Drawdown tracker
    drawdown_tracker: DrawdownTracker,
    /// Stress monitor
    stress_monitor: StressMonitor,
    /// Risk level cache
    cached_risk_level: RiskLevel,
    /// Last update timestamp
    last_update_ns: AtomicU64,
    /// Alert counter
    alert_count: AtomicU64,
    /// Dashboard enabled
    dashboard_enabled: AtomicBool,
}

impl RiskControlManager {
    /// Create a new risk control manager
    pub fn new(
        portfolio_value: f64,
        max_drawdown: f64,
        alert_threshold: f64,
    ) -> Self {
        Self {
            stress_engine: StressTestEngine::new(portfolio_value),
            drawdown_tracker: DrawdownTracker::new(portfolio_value, max_drawdown),
            stress_monitor: StressMonitor::new(portfolio_value, alert_threshold),
            cached_risk_level: RiskLevel::Low,
            last_update_ns: AtomicU64::new(0),
            alert_count: AtomicU64::new(0),
            dashboard_enabled: AtomicBool::new(true),
        }
    }
    
    /// Update with current portfolio state
    pub fn update(
        &mut self,
        current_value: f64,
        var: f64,
        cvar: f64,
        cvar_var_ratio: f64,
    ) -> RiskControlDashboard {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        
        // Update drawdown tracker
        let dd_state = self.drawdown_tracker.update(current_value);
        
        // Check stress alerts
        let stress_alert = self.stress_monitor.check();
        if stress_alert.triggered {
            self.alert_count.fetch_add(1, Ordering::Relaxed);
        }
        
        // Find worst stress scenario
        let worst_case = self.stress_engine.find_worst_case()
            .unwrap_or_else(|| (String::from("N/A"), 0.0));
        
        // Calculate overall risk level
        let risk_level = RiskLevel::from_metrics(dd_state.current_drawdown, var, cvar_var_ratio);
        self.cached_risk_level = risk_level;
        
        self.last_update_ns.store(timestamp, Ordering::Relaxed);
        
        RiskControlDashboard {
            var,
            cvar,
            drawdown: dd_state.current_drawdown,
            max_drawdown: dd_state.max_drawdown,
            circuit_breaker_active: self.drawdown_tracker.circuit_breaker_status().is_active,
            size_multiplier: self.drawdown_tracker.size_multiplier(),
            worst_stress_scenario: worst_case.0,
            worst_stress_drawdown: worst_case.1,
            risk_level,
            timestamp_ns: timestamp,
        }
    }
    
    /// Add a position for stress testing
    pub fn add_position(&mut self, position: StressPosition) {
        self.stress_engine.add_position(position);
        self.stress_monitor.engine_mut().add_position(position);
    }
    
    /// Remove a position
    pub fn remove_position(&mut self, asset_id: &str) {
        self.stress_engine.remove_position(asset_id);
    }
    
    /// Check if trading is allowed
    pub fn can_trade(&self) -> bool {
        self.drawdown_tracker.can_trade() && self.cached_risk_level != RiskLevel::Critical
    }
    
    /// Get current size multiplier
    pub fn size_multiplier(&self) -> f64 {
        self.drawdown_tracker.size_multiplier()
    }
    
    /// Get current risk level
    pub fn risk_level(&self) -> RiskLevel {
        self.cached_risk_level
    }
    
    /// Manual override of circuit breakers
    pub fn manual_override(&self, override: bool) {
        self.drawdown_tracker.manual_override(override);
    }
    
    /// Reset circuit breakers
    pub fn reset_circuit_breakers(&mut self) {
        self.drawdown_tracker.reset_breaker();
    }
    
    /// Get alert count
    pub fn alert_count(&self) -> u64 {
        self.alert_count.load(Ordering::Relaxed)
    }
    
    /// Enable/disable dashboard
    pub fn set_dashboard_enabled(&self, enabled: bool) {
        self.dashboard_enabled.store(enabled, Ordering::Relaxed);
    }
    
    /// Check if dashboard is enabled
    pub fn is_dashboard_enabled(&self) -> bool {
        self.dashboard_enabled.load(Ordering::Relaxed)
    }
}

/// Quick risk summary for logging
#[derive(Debug, Clone)]
pub struct RiskSummary {
    pub can_trade: bool,
    pub risk_level: RiskLevel,
    pub drawdown: f64,
    pub size_multiplier: f64,
    pub alerts_pending: bool,
}

impl RiskSummary {
    /// Format for display
    pub fn format(&self) -> String {
        format!(
            "RiskSummary {{ can_trade: {}, level: {:?}, dd: {:.2}%, size: {:.0}% }}",
            self.can_trade,
            self.risk_level,
            self.drawdown * 100.0,
            self.size_multiplier * 100.0
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_risk_manager() {
        let mut manager = RiskControlManager::new(1_000_000.0, -0.25, -0.10);
        
        // Add a test position
        manager.add_position(StressPosition {
            asset_id: "BTC".to_string(),
            size: 10.0,
            entry_price: 50000.0,
            current_price: 55000.0,
            volatility: 0.02,
            btc_correlation: 1.0,
        });
        
        // Update with normal conditions
        let dashboard = manager.update(1_000_000.0, 0.02, 0.03, 1.2);
        assert_eq!(dashboard.risk_level, RiskLevel::Low);
        assert!(manager.can_trade());
        
        // Simulate drawdown
        let dashboard = manager.update(900_000.0, 0.05, 0.08, 1.5);
        assert!(dashboard.drawdown < -0.05);
    }
    
    #[test]
    fn test_risk_level_classification() {
        let level = RiskLevel::from_metrics(-0.02, 0.01, 1.1);
        assert_eq!(level, RiskLevel::Low);
        
        let level = RiskLevel::from_metrics(-0.12, 0.03, 1.8);
        assert!(matches!(level, RiskLevel::Elevated | RiskLevel::High));
        
        let level = RiskLevel::from_metrics(-0.25, 0.10, 3.5);
        assert_eq!(level, RiskLevel::Critical);
    }
}
