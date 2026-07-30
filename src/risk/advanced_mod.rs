//! Advanced risk metrics module root (VaR/CVaR integration).
//! 
//! Wires VaR/CVaR limits directly into the pre-trade risk bus.

pub mod var;
pub mod cvar;

pub use var::{VarCalculator, VarConfig, VarMethod, VarResult, PortfolioVarCalculator};
pub use cvar::{CVarCalculator, CVarConfig, CVarResult, PortfolioCVarCalculator};

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use crate::risk::prelude::RiskLimits;

/// Risk limit breach event
#[derive(Debug, Clone)]
pub struct RiskBreachEvent {
    /// Type of limit breached
    pub limit_type: RiskLimitType,
    /// Current value
    pub current_value: f64,
    /// Limit threshold
    pub limit_threshold: f64,
    /// Timestamp
    pub timestamp_ns: u64,
    /// Asset/portfolio identifier
    pub asset_id: String,
}

/// Type of risk limit
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiskLimitType {
    /// Value at Risk limit
    Var,
    /// Conditional VaR limit
    CVar,
    /// Maximum position size
    PositionSize,
    /// Maximum drawdown
    Drawdown,
    /// Concentration limit
    Concentration,
    /// Leverage limit
    Leverage,
}

/// Pre-trade risk checker integrating VaR/CVaR
pub struct PreTradeRiskChecker {
    /// VaR calculator
    var_calc: VarCalculator,
    /// CVaR calculator
    cvar_calc: CVarCalculator,
    /// VaR limit (as fraction of portfolio)
    var_limit: f64,
    /// CVaR limit (as fraction of portfolio)
    cvar_limit: f64,
    /// Whether checks are enabled
    enabled: AtomicBool,
    /// Breach counter
    breach_count: AtomicU64,
    /// Last breach timestamp
    last_breach_ns: AtomicU64,
}

impl PreTradeRiskChecker {
    /// Create a new pre-trade risk checker
    pub fn new(capacity: usize, var_limit: f64, cvar_limit: f64) -> Self {
        let var_config = VarConfig::default();
        let cvar_config = CVarConfig {
            var_config: var_config.clone(),
            ..Default::default()
        };
        
        Self {
            var_calc: VarCalculator::new(capacity, var_config),
            cvar_calc: CVarCalculator::new(capacity, cvar_config),
            var_limit,
            cvar_limit,
            enabled: AtomicBool::new(true),
            breach_count: AtomicU64::new(0),
            last_breach_ns: AtomicU64::new(0),
        }
    }
    
    /// Check if a trade would breach risk limits
    pub fn check_trade(&mut self, trade_return: f64, trade_size: f64, portfolio_value: f64) -> RiskCheckResult {
        if !self.enabled.load(Ordering::Relaxed) {
            return RiskCheckResult::approved();
        }
        
        // Add return observation
        self.var_calc.add_return(trade_return);
        self.cvar_calc.add_return(trade_return);
        
        let mut result = RiskCheckResult::approved();
        
        // Check VaR limit
        if let Some(var_result) = self.var_calc.calculate_var() {
            let var_dollar = var_result.var_dollar(portfolio_value * trade_size);
            if var_dollar > self.var_limit * portfolio_value {
                result.approved = false;
                result.breach_type = Some(RiskLimitType::Var);
                result.current_value = var_result.var;
                result.limit_threshold = self.var_limit;
                
                self.record_breach();
            }
        }
        
        // Check CVaR limit
        if let Some(cvar_result) = self.cvar_calc.calculate_cvar() {
            let cvar_dollar = cvar_result.cvar_dollar(portfolio_value * trade_size);
            if cvar_dollar > self.cvar_limit * portfolio_value {
                result.approved = false;
                result.breach_type = Some(RiskLimitType::CVar);
                result.current_value = cvar_result.cvar;
                result.limit_threshold = self.cvar_limit;
                
                self.record_breach();
            }
        }
        
        result
    }
    
    /// Record a breach event
    fn record_breach(&self) {
        self.breach_count.fetch_add(1, Ordering::Relaxed);
        self.last_breach_ns.store(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64,
            Ordering::Relaxed,
        );
    }
    
    /// Get current VaR
    pub fn current_var(&mut self) -> Option<f64> {
        self.var_calc.calculate_var().map(|r| r.var)
    }
    
    /// Get current CVaR
    pub fn current_cvar(&mut self) -> Option<f64> {
        self.cvar_calc.calculate_cvar().map(|r| r.cvar)
    }
    
    /// Update limits
    pub fn update_limits(&mut self, var_limit: f64, cvar_limit: f64) {
        self.var_limit = var_limit;
        self.cvar_limit = cvar_limit;
    }
    
    /// Enable/disable checking
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Relaxed);
    }
    
    /// Get breach count
    pub fn breach_count(&self) -> u64 {
        self.breach_count.load(Ordering::Relaxed)
    }
}

/// Result of a pre-trade risk check
#[derive(Debug, Clone)]
pub struct RiskCheckResult {
    /// Whether trade is approved
    pub approved: bool,
    /// Type of limit breached (if any)
    pub breach_type: Option<RiskLimitType>,
    /// Current risk metric value
    pub current_value: f64,
    /// Limit threshold
    pub limit_threshold: f64,
    /// Recommended action
    pub recommendation: RiskAction,
}

impl RiskCheckResult {
    /// Create an approved result
    pub fn approved() -> Self {
        Self {
            approved: true,
            breach_type: None,
            current_value: 0.0,
            limit_threshold: 0.0,
            recommendation: RiskAction::Proceed,
        }
    }
    
    /// Create a rejected result
    pub fn rejected(breach_type: RiskLimitType, current: f64, limit: f64) -> Self {
        Self {
            approved: false,
            breach_type: Some(breach_type),
            current_value: current,
            limit_threshold: limit,
            recommendation: RiskAction::Reject,
        }
    }
}

/// Recommended action from risk check
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiskAction {
    /// Proceed with trade
    Proceed,
    /// Reject trade
    Reject,
    /// Reduce size
    ReduceSize,
    /// Hedge exposure
    Hedge,
    /// Wait for better conditions
    Wait,
}

/// Risk dashboard data for monitoring
#[derive(Debug, Clone)]
pub struct RiskDashboardData {
    /// Current VaR
    pub var: f64,
    /// Current CVaR
    pub cvar: f64,
    /// VaR limit utilization (0.0 to 1.0+)
    pub var_utilization: f64,
    /// CVaR limit utilization
    pub cvar_utilization: f64,
    /// Breach count in last hour
    pub recent_breaches: u64,
    /// Tail risk indicator
    pub tail_risk_level: TailRiskLevel,
}

/// Tail risk level classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TailRiskLevel {
    /// Low tail risk
    Low,
    /// Moderate tail risk
    Moderate,
    /// Elevated tail risk
    Elevated,
    /// High tail risk
    High,
    /// Extreme tail risk
    Extreme,
}

impl TailRiskLevel {
    /// Classify based on CVaR/VaR ratio
    pub fn from_ratio(ratio: f64) -> Self {
        if ratio < 1.2 {
            TailRiskLevel::Low
        } else if ratio < 1.5 {
            TailRiskLevel::Moderate
        } else if ratio < 2.0 {
            TailRiskLevel::Elevated
        } else if ratio < 3.0 {
            TailRiskLevel::High
        } else {
            TailRiskLevel::Extreme
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_pre_trade_checker() {
        let mut checker = PreTradeRiskChecker::new(1000, 0.05, 0.08);
        
        // Add some normal returns
        for i in 0..300 {
            let ret = (i as f64 * 0.0005 - 0.075);
            let _ = checker.check_trade(ret, 0.1, 1_000_000.0);
        }
        
        // Should have some valid readings now
        let var = checker.current_var();
        let cvar = checker.current_cvar();
        
        assert!(var.is_some());
        assert!(cvar.is_some());
    }
}
