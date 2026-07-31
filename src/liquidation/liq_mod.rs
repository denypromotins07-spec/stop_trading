//! Liquidation Module Root
//! 
//! Wires the liquidation simulator directly into the global kill switch and pre-trade risk bus.

pub mod sim;
pub mod tiers;

pub use sim::{
    AdlEntry, FixedPoint, GlobalLiquidationState, InsuranceFund, LiquidationResult,
    LiquidationSimulator, MarginTier, Position, PositionSide, RiskAction,
};
pub use tiers::{MarginTierState, PortfolioMarginTracker, SafetyBufferMonitor};

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::sync::Arc;

/// Pre-trade risk check result
#[derive(Debug, Clone, Copy)]
pub struct PreTradeRiskCheck {
    pub allowed: bool,
    pub reason: Option<RiskRejectReason>,
    pub max_allowed_size: FixedPoint,
    pub estimated_liq_impact: FixedPoint,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiskRejectReason {
    WouldTriggerLiquidation,
    ExceedsLeverageLimit,
    InsufficientMargin,
    KillSwitchActive,
    AdlQueueFull,
    MarginRatioTooLow,
}

/// Global liquidation manager integrating all components
pub struct LiquidationManager {
    pub state: GlobalLiquidationState,
    pub margin_tracker: PortfolioMarginTracker,
    pub safety_monitor: SafetyBufferMonitor,
    pub risk_checks_enabled: AtomicBool,
    pub pre_trade_check_counter: AtomicU64,
}

impl LiquidationManager {
    pub fn new(tiers: Vec<MarginTier>, insurance_fund: InsuranceFund, default_safety_buffer: f64) -> Self {
        let simulator = LiquidationSimulator::new(tiers.clone(), insurance_fund);
        let state = GlobalLiquidationState::new(simulator);
        
        LiquidationManager {
            state: state.clone(),
            margin_tracker: PortfolioMarginTracker::new(tiers),
            safety_monitor: SafetyBufferMonitor::new(default_safety_buffer),
            risk_checks_enabled: AtomicBool::new(true),
            pre_trade_check_counter: AtomicU64::new(0),
        }
    }

    /// Run pre-trade risk check before allowing an order
    pub fn pre_trade_check(&self, position: &Position, order_size: FixedPoint, order_side: PositionSide) -> PreTradeRiskCheck {
        self.pre_trade_check_counter.fetch_add(1, Ordering::Relaxed);

        if !self.risk_checks_enabled.load(Ordering::Relaxed) {
            return PreTradeRiskCheck {
                allowed: true,
                reason: None,
                max_allowed_size: order_size,
                estimated_liq_impact: FixedPoint(0),
            };
        }

        // Check kill switch
        if self.state.is_kill_switch_active() {
            return PreTradeRiskCheck {
                allowed: false,
                reason: Some(RiskRejectReason::KillSwitchActive),
                max_allowed_size: FixedPoint(0),
                estimated_liq_impact: FixedPoint(0),
            };
        }

        // Simulate new position state with order
        let simulated_position = self.create_simulated_position(position, order_size, order_side);
        
        // Run liquidation simulation
        let sim_result = self.state.simulator.simulate_liquidation(&simulated_position);

        // Check margin ratio
        if self.margin_tracker.is_liquidation_risk() {
            return PreTradeRiskCheck {
                allowed: false,
                reason: Some(RiskRejectReason::MarginRatioTooLow),
                max_allowed_size: FixedPoint(0),
                estimated_liq_impact: sim_result.insurance_fund_impact,
            };
        }

        // Check if order would trigger liquidation
        if sim_result.will_liquidate {
            return PreTradeRiskCheck {
                allowed: false,
                reason: Some(RiskRejectReason::WouldTriggerLiquidation),
                max_allowed_size: self.calculate_safe_size(position, order_side),
                estimated_liq_impact: sim_result.insurance_fund_impact,
            };
        }

        // Check leverage limits
        let new_notional = simulated_position.size.checked_mul(simulated_position.mark_price).unwrap_or(FixedPoint(0));
        if let Some(tier) = self.margin_tracker.get_tier_for_notional(new_notional) {
            let implied_leverage = new_notional.to_f64() / simulated_position.margin_balance.to_f64();
            if implied_leverage > tier.max_leverage as f64 {
                return PreTradeRiskCheck {
                    allowed: false,
                    reason: Some(RiskRejectReason::ExceedsLeverageLimit),
                    max_allowed_size: self.calculate_safe_size(position, order_side),
                    estimated_liq_impact: FixedPoint(0),
                };
            }
        }

        PreTradeRiskCheck {
            allowed: true,
            reason: None,
            max_allowed_size: order_size,
            estimated_liq_impact: sim_result.insurance_fund_impact,
        }
    }

    fn create_simulated_position(&self, base: &Position, add_size: FixedPoint, side: PositionSide) -> Position {
        let new_size = if side == base.side {
            base.size.checked_add(add_size).unwrap_or(base.size)
        } else {
            base.size.checked_sub(add_size).unwrap_or(FixedPoint(0))
        };

        // Calculate weighted average entry price if adding to position
        let new_entry_price = if side == base.side && add_size.0 > 0 {
            let total_value = base.entry_price.checked_mul(base.size).unwrap_or(FixedPoint(0))
                .checked_add(base.mark_price.checked_mul(add_size).unwrap_or(FixedPoint(0)))
                .unwrap_or(FixedPoint(0));
            total_value.checked_div(new_size).unwrap_or(base.entry_price)
        } else {
            base.entry_price
        };

        Position {
            id: base.id,
            side: if new_size.0 == 0 { base.side } else if side == base.side { side } else {
                if new_size.0 < base.size.0 / 2 { base.side } else { side }
            },
            size: new_size,
            entry_price: new_entry_price,
            mark_price: base.mark_price,
            leverage: base.leverage,
            margin_balance: base.margin_balance,
            isolated: base.isolated,
        }
    }

    fn calculate_safe_size(&self, position: &Position, side: PositionSide) -> FixedPoint {
        // Binary search for safe size
        let mut low = FixedPoint(0);
        let mut high = position.size.checked_mul(FixedPoint::from_f64(2.0)).unwrap_or(position.size);
        let mut safe_size = FixedPoint(0);

        for _ in 0..20 { // 20 iterations for precision
            let mid = FixedPoint((low.0 + high.0) / 2);
            if mid.0 == 0 {
                break;
            }

            let sim_pos = self.create_simulated_position(position, mid, side);
            let result = self.state.simulator.simulate_liquidation(&sim_pos);

            if !result.will_liquidate {
                safe_size = mid;
                low = mid;
            } else {
                high = mid;
            }
        }

        safe_size
    }

    /// Update all internal state from current positions
    pub fn update_state(&mut self, positions: &[Position], mark_prices: &[(u64, FixedPoint)], volatility: f64) {
        // Update margin tracker
        self.margin_tracker.update_from_positions(positions, mark_prices);

        // Update safety buffer based on volatility
        self.safety_monitor.adjust_for_volatility(volatility, 0.04);

        // Check each position against safety buffer
        for position in positions {
            if let Some(liq_price) = position.liquidation_price(
                self.margin_tracker.tier_state.maintenance_margin_required
            ) {
                self.safety_monitor.check_position_safety(position, liq_price, &self.state.simulator);
            }
        }

        // Auto-deleverage if needed
        if self.margin_tracker.should_deleverage(1.2) {
            let mut positions_mut: Vec<Position> = positions.to_vec();
            let actions = self.margin_tracker.execute_auto_deleverage(&mut positions_mut, 1.2);
            
            if !actions.is_empty() {
                log::warn!("Auto-deleverage triggered: {} actions", actions.len());
            }
        }
    }

    /// Trigger global kill switch
    pub fn trigger_kill_switch(&self) {
        self.state.trigger_kill_switch();
        log::error!("LIQUIDATION KILL SWITCH TRIGGERED");
    }

    /// Reset kill switch after manual review
    pub fn reset_kill_switch(&self) {
        self.safety_monitor.clear_risk_list();
        self.state.reset_kill_switch();
        log::info!("Kill switch reset after manual review");
    }

    /// Get current risk summary
    pub fn get_risk_summary(&self) -> RiskSummary {
        RiskSummary {
            margin_ratio: self.margin_tracker.margin_ratio.to_f64(),
            effective_leverage: self.margin_tracker.effective_leverage(),
            positions_at_risk: self.safety_monitor.positions_at_risk_count(),
            kill_switch_active: self.state.is_kill_switch_active(),
            auto_deleverage_enabled: self.margin_tracker.auto_deleverage_enabled.load(Ordering::Relaxed),
            total_notional_long: self.margin_tracker.total_notional_long.to_f64(),
            total_notional_short: self.margin_tracker.total_notional_short.to_f64(),
            margin_utilization: self.margin_tracker.margin_utilization(),
        }
    }

    /// Enable/disable risk checks (for emergency override)
    pub fn set_risk_checks_enabled(&self, enabled: bool) {
        self.risk_checks_enabled.store(enabled, Ordering::Relaxed);
    }
}

/// Risk summary for monitoring and dashboards
#[derive(Debug, Clone)]
pub struct RiskSummary {
    pub margin_ratio: f64,
    pub effective_leverage: f64,
    pub positions_at_risk: usize,
    pub kill_switch_active: bool,
    pub auto_deleverage_enabled: bool,
    pub total_notional_long: f64,
    pub total_notional_short: f64,
    pub margin_utilization: f64,
}

impl RiskSummary {
    pub fn is_safe(&self) -> bool {
        self.margin_ratio > 1.5 
            && self.positions_at_risk == 0 
            && !self.kill_switch_active
            && self.effective_leverage < 10.0
    }

    pub fn warning_level(&self) -> WarningLevel {
        if self.is_safe() {
            WarningLevel::Green
        } else if self.margin_ratio > 1.2 && self.positions_at_risk < 3 {
            WarningLevel::Yellow
        } else if self.margin_ratio > 1.0 || self.positions_at_risk < 10 {
            WarningLevel::Orange
        } else {
            WarningLevel::Red
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WarningLevel {
    Green,
    Yellow,
    Orange,
    Red,
}

// Re-export log crate for internal logging
extern crate log;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_liquidation_manager_creation() {
        let tiers = vec![MarginTier {
            lower_bound: FixedPoint::from_f64(0.0),
            upper_bound: FixedPoint::from_f64(10_000_000.0),
            initial_margin_rate: FixedPoint::from_f64(0.1),
            maintenance_margin_rate: FixedPoint::from_f64(0.05),
            max_leverage: 10,
        }];

        let insurance = InsuranceFund::new(1_000_000.0, "USDT");
        let manager = LiquidationManager::new(tiers, insurance, 0.02);

        let summary = manager.get_risk_summary();
        assert!(!summary.kill_switch_active);
        assert!(summary.auto_deleverage_enabled);
    }

    #[test]
    fn test_pre_trade_check_allows_safe_order() {
        let tiers = vec![MarginTier {
            lower_bound: FixedPoint::from_f64(0.0),
            upper_bound: FixedPoint::from_f64(10_000_000.0),
            initial_margin_rate: FixedPoint::from_f64(0.1),
            maintenance_margin_rate: FixedPoint::from_f64(0.05),
            max_leverage: 10,
        }];

        let insurance = InsuranceFund::new(1_000_000.0, "USDT");
        let manager = LiquidationManager::new(tiers, insurance, 0.02);

        let position = Position {
            id: 1,
            side: PositionSide::Long,
            size: FixedPoint::from_f64(0.1),
            entry_price: FixedPoint::from_f64(50_000.0),
            mark_price: FixedPoint::from_f64(50_000.0),
            leverage: 2,
            margin_balance: FixedPoint::from_f64(10_000.0),
            isolated: false,
        };

        let result = manager.pre_trade_check(&position, FixedPoint::from_f64(0.1), PositionSide::Long);
        assert!(result.allowed);
        assert!(result.reason.is_none());
    }

    #[test]
    fn test_risk_summary_warning_levels() {
        let summary_safe = RiskSummary {
            margin_ratio: 2.0,
            effective_leverage: 3.0,
            positions_at_risk: 0,
            kill_switch_active: false,
            auto_deleverage_enabled: true,
            total_notional_long: 50_000.0,
            total_notional_short: 0.0,
            margin_utilization: 30.0,
        };

        assert_eq!(summary_safe.warning_level(), WarningLevel::Green);
        assert!(summary_safe.is_safe());
    }
}
