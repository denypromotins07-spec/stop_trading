//! Dynamic Margin Tier Tracker
//! 
//! Implements dynamic margin tier tracking that adjusts leverage limits based on real-time notional exposure.
//! Instantly deleverages the portfolio if simulated liquidation price breaches safety buffer thresholds.

use super::sim::{FixedPoint, Position, PositionSide, LiquidationSimulator, MarginTier, RiskAction};
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::sync::Arc;

/// Real-time margin tier state
#[derive(Debug, Clone)]
pub struct MarginTierState {
    pub current_tier_idx: usize,
    pub current_leverage_limit: u32,
    pub notional_value: FixedPoint,
    pub initial_margin_required: FixedPoint,
    pub maintenance_margin_required: FixedPoint,
    pub last_updated_ns: u64,
}

impl MarginTierState {
    pub fn new() -> Self {
        MarginTierState {
            current_tier_idx: 0,
            current_leverage_limit: 125, // Default max leverage
            notional_value: FixedPoint(0),
            initial_margin_required: FixedPoint(0),
            maintenance_margin_required: FixedPoint(0),
            last_updated_ns: 0,
        }
    }

    /// Update tier state based on current notional exposure
    pub fn update(&mut self, notional: FixedPoint, tiers: &[MarginTier], timestamp_ns: u64) {
        self.notional_value = notional;
        self.last_updated_ns = timestamp_ns;

        // Find applicable tier
        for (idx, tier) in tiers.iter().enumerate() {
            if notional.0 >= tier.lower_bound.0 && notional.0 < tier.upper_bound.0 {
                self.current_tier_idx = idx;
                self.current_leverage_limit = tier.max_leverage;
                
                self.initial_margin_required = notional
                    .checked_mul(tier.initial_margin_rate)
                    .unwrap_or(FixedPoint(0));
                
                self.maintenance_margin_required = notional
                    .checked_mul(tier.maintenance_margin_rate)
                    .unwrap_or(FixedPoint(0));
                return;
            }
        }

        // If no tier matches, use the highest tier
        if let Some(last_tier) = tiers.last() {
            self.current_leverage_limit = last_tier.max_leverage;
            self.initial_margin_required = notional
                .checked_mul(last_tier.initial_margin_rate)
                .unwrap_or(FixedPoint(0));
            self.maintenance_margin_required = notional
                .checked_mul(last_tier.maintenance_margin_rate)
                .unwrap_or(FixedPoint(0));
        }
    }

    /// Check if current leverage exceeds tier limit
    pub fn is_overleveraged(&self, current_leverage: u32) -> bool {
        current_leverage > self.current_leverage_limit
    }

    /// Calculate required deleveraging amount
    pub fn calculate_deleverage_amount(&self, current_leverage: u32, position_size: FixedPoint) -> Option<FixedPoint> {
        if !self.is_overleveraged(current_leverage) {
            return Some(FixedPoint(0));
        }

        // Target size = Equity * TargetLeverage / Price
        // Simplified: reduce to tier limit
        let leverage_ratio = FixedPoint::from_f64(self.current_leverage_limit as f64 / current_leverage as f64);
        position_size.checked_mul(leverage_ratio)
    }
}

/// Portfolio-wide margin tracker
pub struct PortfolioMarginTracker {
    pub total_notional_long: FixedPoint,
    pub total_notional_short: FixedPoint,
    pub total_initial_margin: FixedPoint,
    pub total_maintenance_margin: FixedPoint,
    pub account_equity: FixedPoint,
    pub margin_ratio: FixedPoint,
    pub tiers: Vec<MarginTier>,
    pub tier_state: MarginTierState,
    pub auto_deleverage_enabled: AtomicBool,
    pub deleverage_counter: AtomicU64,
}

impl PortfolioMarginTracker {
    pub fn new(tiers: Vec<MarginTier>) -> Self {
        PortfolioMarginTracker {
            total_notional_long: FixedPoint(0),
            total_notional_short: FixedPoint(0),
            total_initial_margin: FixedPoint(0),
            total_maintenance_margin: FixedPoint(0),
            account_equity: FixedPoint(0),
            margin_ratio: FixedPoint::from_f64(1.0),
            tiers,
            tier_state: MarginTierState::new(),
            auto_deleverage_enabled: AtomicBool::new(true),
            deleverage_counter: AtomicU64::new(0),
        }
    }

    /// Update portfolio margins from position list
    pub fn update_from_positions(&mut self, positions: &[Position], mark_prices: &[(u64, FixedPoint)]) {
        self.total_notional_long = FixedPoint(0);
        self.total_notional_short = FixedPoint(0);
        self.total_initial_margin = FixedPoint(0);
        self.total_maintenance_margin = FixedPoint(0);

        let mut total_unrealized_pnl = FixedPoint(0);

        for position in positions {
            // Find mark price
            let mark_price = mark_prices
                .iter()
                .find(|(id, _)| *id == position.id)
                .map(|(_, p)| *p)
                .unwrap_or(position.mark_price);

            let notional = position.size.checked_mul(mark_price).unwrap_or(FixedPoint(0));
            
            match position.side {
                PositionSide::Long => {
                    self.total_notional_long = self.total_notional_long.checked_add(notional).unwrap_or(self.total_notional_long);
                }
                PositionSide::Short => {
                    self.total_notional_short = self.total_notional_short.checked_add(notional).unwrap_or(self.total_notional_short);
                }
            }

            // Calculate PnL
            let pnl = match position.side {
                PositionSide::Long => {
                    let diff = mark_price.checked_sub(position.entry_price).unwrap_or(FixedPoint(0));
                    diff.checked_mul(position.size).unwrap_or(FixedPoint(0))
                }
                PositionSide::Short => {
                    let diff = position.entry_price.checked_sub(mark_price).unwrap_or(FixedPoint(0));
                    diff.checked_mul(position.size).unwrap_or(FixedPoint(0))
                }
            };
            total_unrealized_pnl = total_unrealized_pnl.checked_add(pnl).unwrap_or(total_unrealized_pnl);

            // Find tier and add margins
            if let Some(tier) = self.get_tier_for_notional(notional) {
                let im = notional.checked_mul(tier.initial_margin_rate).unwrap_or(FixedPoint(0));
                let mm = notional.checked_mul(tier.maintenance_margin_rate).unwrap_or(FixedPoint(0));
                self.total_initial_margin = self.total_initial_margin.checked_add(im).unwrap_or(self.total_initial_margin);
                self.total_maintenance_margin = self.total_maintenance_margin.checked_add(mm).unwrap_or(self.total_maintenance_margin);
            }
        }

        // Update account equity
        self.account_equity = self.account_equity.checked_add(total_unrealized_pnl).unwrap_or(self.account_equity);

        // Calculate margin ratio
        let total_notional = self.total_notional_long.checked_add(self.total_notional_short).unwrap_or(FixedPoint(0));
        self.tier_state.update(total_notional, &self.tiers, get_timestamp_ns());

        if self.account_equity.0 > 0 {
            self.margin_ratio = self.account_equity
                .checked_div(self.total_maintenance_margin)
                .unwrap_or(FixedPoint::from_f64(f64::INFINITY));
        }
    }

    fn get_tier_for_notional(&self, notional: FixedPoint) -> Option<&MarginTier> {
        self.tiers.iter().find(|tier| {
            notional.0 >= tier.lower_bound.0 && notional.0 < tier.upper_bound.0
        })
    }

    /// Check if portfolio is at risk of liquidation
    pub fn is_liquidation_risk(&self) -> bool {
        // Margin ratio below 1.0 means under-margined
        self.margin_ratio.to_f64() < 1.0
    }

    /// Check if deleveraging should be triggered
    pub fn should_deleverage(&self, safety_threshold: f64) -> bool {
        if !self.auto_deleverage_enabled.load(Ordering::Relaxed) {
            return false;
        }
        
        let ratio = self.margin_ratio.to_f64();
        ratio < safety_threshold && ratio > 0.0
    }

    /// Execute automatic deleveraging
    pub fn execute_auto_deleverage(&mut self, positions: &mut [Position], safety_threshold: f64) -> Vec<RiskAction> {
        if !self.should_deleverage(safety_threshold) {
            return vec![];
        }

        let mut actions = Vec::new();
        self.deleverage_counter.fetch_add(1, Ordering::Relaxed);

        // Sort positions by leverage (highest first)
        let mut sorted_indices: Vec<usize> = (0..positions.len()).collect();
        sorted_indices.sort_by(|&a, &b| {
            positions[b].leverage.cmp(&positions[a].leverage)
        });

        for &idx in &sorted_indices {
            if !self.should_deleverage(safety_threshold) {
                break;
            }

            let position = &positions[idx];
            let notional = position.size.checked_mul(position.mark_price).unwrap_or(FixedPoint(0));
            
            if let Some(reduce_size) = self.tier_state.calculate_deleverage_amount(position.leverage, position.size) {
                if reduce_size.0 > 0 {
                    actions.push(RiskAction::ReducePosition(reduce_size));
                    
                    // Update local state
                    let reduction_notional = reduce_size.checked_mul(position.mark_price).unwrap_or(FixedPoint(0));
                    match position.side {
                        PositionSide::Long => {
                            self.total_notional_long = self.total_notional_long.checked_sub(reduction_notional)
                                .unwrap_or(self.total_notional_long);
                        }
                        PositionSide::Short => {
                            self.total_notional_short = self.total_notional_short.checked_sub(reduction_notional)
                                .unwrap_or(self.total_notional_short);
                        }
                    }
                    
                    // Recalculate margin ratio after hypothetical reduction
                    self.recalculate_margin_ratio();
                }
            }
        }

        actions
    }

    fn recalculate_margin_ratio(&mut self) {
        let total_notional = self.total_notional_long.checked_add(self.total_notional_short).unwrap_or(FixedPoint(0));
        
        // Recalculate maintenance margin based on new notional
        let mut new_mm = FixedPoint(0);
        for tier in &self.tiers {
            if total_notional.0 >= tier.lower_bound.0 && total_notional.0 < tier.upper_bound.0 {
                new_mm = total_notional.checked_mul(tier.maintenance_margin_rate).unwrap_or(FixedPoint(0));
                break;
            }
        }

        if new_mm.0 > 0 && self.account_equity.0 > 0 {
            self.margin_ratio = self.account_equity.checked_div(new_mm).unwrap_or(FixedPoint::from_f64(f64::INFINITY));
        }
    }

    /// Get current effective leverage
    pub fn effective_leverage(&self) -> f64 {
        let total_notional = self.total_notional_long.checked_add(self.total_notional_short).unwrap_or(FixedPoint(0));
        if self.account_equity.0 == 0 {
            return 0.0;
        }
        total_notional.to_f64() / self.account_equity.to_f64()
    }

    /// Enable/disable auto-deleverage
    pub fn set_auto_deleverage(&self, enabled: bool) {
        self.auto_deleverage_enabled.store(enabled, Ordering::Relaxed);
    }

    /// Get margin utilization percentage
    pub fn margin_utilization(&self) -> f64 {
        if self.account_equity.0 == 0 {
            return 0.0;
        }
        (self.total_initial_margin.to_f64() / self.account_equity.to_f64()) * 100.0
    }
}

/// Helper function to get nanosecond timestamp
fn get_timestamp_ns() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

/// Safety buffer monitor for liquidation prices
pub struct SafetyBufferMonitor {
    pub default_buffer_pct: f64,
    pub volatility_adjusted_buffer: f64,
    pub positions_at_risk: Vec<u64>,
}

impl SafetyBufferMonitor {
    pub fn new(default_buffer_pct: f64) -> Self {
        SafetyBufferMonitor {
            default_buffer_pct,
            volatility_adjusted_buffer: default_buffer_pct,
            positions_at_risk: Vec::with_capacity(256),
        }
    }

    /// Adjust buffer based on market volatility
    pub fn adjust_for_volatility(&mut self, volatility: f64, base_volatility: f64) {
        let vol_ratio = volatility / base_volatility.max(0.0001);
        self.volatility_adjusted_buffer = self.default_buffer_pct * vol_ratio.max(1.0).min(5.0);
    }

    /// Check position against safety buffer
    pub fn check_position_safety(
        &mut self,
        position: &Position,
        liq_price: FixedPoint,
        simulator: &LiquidationSimulator,
    ) -> bool {
        let distance = match position.side {
            PositionSide::Long => {
                let diff = position.mark_price.checked_sub(liq_price).unwrap_or(FixedPoint(0));
                diff.checked_div(position.mark_price).unwrap_or(FixedPoint(0))
            }
            PositionSide::Short => {
                let diff = liq_price.checked_sub(position.mark_price).unwrap_or(FixedPoint(0));
                diff.checked_div(position.mark_price).unwrap_or(FixedPoint(0))
            }
        };

        let buffer_fp = FixedPoint::from_f64(self.volatility_adjusted_buffer);
        let is_safe = distance.0 >= buffer_fp.0;

        if !is_safe {
            if !self.positions_at_risk.contains(&position.id) {
                self.positions_at_risk.push(position.id);
            }
        } else {
            self.positions_at_risk.retain(|&id| id != position.id);
        }

        is_safe
    }

    /// Clear risk list
    pub fn clear_risk_list(&mut self) {
        self.positions_at_risk.clear();
    }

    /// Get count of positions at risk
    pub fn positions_at_risk_count(&self) -> usize {
        self.positions_at_risk.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_margin_tier_state_update() {
        let tiers = vec![
            MarginTier {
                lower_bound: FixedPoint::from_f64(0.0),
                upper_bound: FixedPoint::from_f64(100_000.0),
                initial_margin_rate: FixedPoint::from_f64(0.01),
                maintenance_margin_rate: FixedPoint::from_f64(0.005),
                max_leverage: 100,
            },
            MarginTier {
                lower_bound: FixedPoint::from_f64(100_000.0),
                upper_bound: FixedPoint::from_f64(1_000_000.0),
                initial_margin_rate: FixedPoint::from_f64(0.05),
                maintenance_margin_rate: FixedPoint::from_f64(0.025),
                max_leverage: 20,
            },
        ];

        let mut state = MarginTierState::new();
        let notional = FixedPoint::from_f64(50_000.0);
        state.update(notional, &tiers, get_timestamp_ns());

        assert_eq!(state.current_leverage_limit, 100);
        assert!((state.initial_margin_required.to_f64() - 500.0).abs() < 0.01);
    }

    #[test]
    fn test_portfolio_margin_tracker() {
        let tiers = vec![MarginTier {
            lower_bound: FixedPoint::from_f64(0.0),
            upper_bound: FixedPoint::from_f64(10_000_000.0),
            initial_margin_rate: FixedPoint::from_f64(0.1),
            maintenance_margin_rate: FixedPoint::from_f64(0.05),
            max_leverage: 10,
        }];

        let mut tracker = PortfolioMarginTracker::new(tiers);
        tracker.account_equity = FixedPoint::from_f64(10_000.0);

        let positions = vec![
            Position {
                id: 1,
                side: PositionSide::Long,
                size: FixedPoint::from_f64(1.0),
                entry_price: FixedPoint::from_f64(50_000.0),
                mark_price: FixedPoint::from_f64(50_000.0),
                leverage: 5,
                margin_balance: FixedPoint::from_f64(10_000.0),
                isolated: false,
            },
        ];

        let mark_prices = vec![(1, FixedPoint::from_f64(50_000.0))];
        tracker.update_from_positions(&positions, &mark_prices);

        assert!(!tracker.is_liquidation_risk());
        assert!((tracker.effective_leverage() - 5.0).abs() < 0.1);
    }

    #[test]
    fn test_safety_buffer_monitor() {
        let mut monitor = SafetyBufferMonitor::new(0.02);
        monitor.adjust_for_volatility(0.08, 0.04);
        
        assert!((monitor.volatility_adjusted_buffer - 0.04).abs() < 0.001);
    }
}
