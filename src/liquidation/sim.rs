//! Liquidation Simulation Engine
//! 
//! Implements a local exchange-matching liquidation simulator to predict forced liquidations.
//! Models maintenance margin tiers, ADL (Auto-Deleveraging) queues, and insurance fund interventions.
//! Uses fixed-point math for deterministic behavior and zero heap allocation during simulation.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Fixed-point representation for precise financial calculations (1e-8 precision)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FixedPoint(u128);

impl FixedPoint {
    const PRECISION: u128 = 1_0000_0000; // 8 decimal places

    pub fn from_f64(val: f64) -> Self {
        FixedPoint((val * Self::PRECISION as f64).round() as u128)
    }

    pub fn to_f64(self) -> f64 {
        self.0 as f64 / Self::PRECISION as f64
    }

    pub fn checked_mul(self, other: FixedPoint) -> Option<FixedPoint> {
        let result = self.0.checked_mul(other.0)?;
        Some(FixedPoint(result / Self::PRECISION))
    }

    pub fn checked_div(self, other: FixedPoint) -> Option<FixedPoint> {
        let result = self.0.checked_mul(Self::PRECISION)?;
        Some(FixedPoint(result / other.0))
    }

    pub fn checked_add(self, other: FixedPoint) -> Option<FixedPoint> {
        Some(FixedPoint(self.0.checked_add(other.0)?))
    }

    pub fn checked_sub(self, other: FixedPoint) -> Option<FixedPoint> {
        Some(FixedPoint(self.0.checked_sub(other.0)?))
    }

    pub fn is_negative(self) -> bool {
        // Using signed interpretation for negative check
        // In practice, we store absolute value and sign separately if needed
        false // Simplified for unsigned implementation
    }
}

/// Maintenance Margin Tier structure matching Binance/Bybit specifications
#[derive(Debug, Clone, Copy)]
pub struct MarginTier {
    pub lower_bound: FixedPoint,      // Lower bound of notional value
    pub upper_bound: FixedPoint,      // Upper bound of notional value
    pub initial_margin_rate: FixedPoint,
    pub maintenance_margin_rate: FixedPoint,
    pub max_leverage: u32,
}

/// ADL Queue entry representing a position subject to auto-deleveraging
#[derive(Debug, Clone, Copy)]
pub struct AdlEntry {
    pub position_id: u64,
    pub side: PositionSide,
    pub size: FixedPoint,
    pub entry_price: FixedPoint,
    pub unrealized_pnl: FixedPoint,
    pub leverage: u32,
    pub adl_rank: u32, // Higher rank = more likely to be ADL'd first
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PositionSide {
    Long,
    Short,
}

/// Insurance Fund state
#[derive(Debug, Clone)]
pub struct InsuranceFund {
    pub balance: FixedPoint,
    pub currency: [u8; 8],
}

impl InsuranceFund {
    pub fn new(balance: f64, currency: &str) -> Self {
        let mut curr = [0u8; 8];
        curr[..currency.len().min(8)].copy_from_slice(&currency.as_bytes()[..currency.len().min(8)]);
        InsuranceFund {
            balance: FixedPoint::from_f64(balance),
            currency: curr,
        }
    }

    pub fn can_cover(&self, loss: FixedPoint) -> bool {
        self.balance.0 >= loss.0
    }

    pub fn deduct(&mut self, amount: FixedPoint) -> Option<()> {
        self.balance = self.balance.checked_sub(amount)?;
        Some(())
    }

    pub fn add(&mut self, amount: FixedPoint) -> Option<()> {
        self.balance = self.balance.checked_add(amount)?;
        Some(())
    }
}

/// Position state for liquidation simulation
#[derive(Debug, Clone, Copy)]
pub struct Position {
    pub id: u64,
    pub side: PositionSide,
    pub size: FixedPoint,
    pub entry_price: FixedPoint,
    pub mark_price: FixedPoint,
    pub leverage: u32,
    pub margin_balance: FixedPoint,
    pub isolated: bool,
}

impl Position {
    /// Calculate unrealized PnL in quote currency
    pub fn unrealized_pnl(&self) -> FixedPoint {
        match self.side {
            PositionSide::Long => {
                let price_diff = self.mark_price.checked_sub(self.entry_price).unwrap_or(FixedPoint(0));
                price_diff.checked_mul(self.size).unwrap_or(FixedPoint(0))
            }
            PositionSide::Short => {
                let price_diff = self.entry_price.checked_sub(self.mark_price).unwrap_or(FixedPoint(0));
                price_diff.checked_mul(self.size).unwrap_or(FixedPoint(0))
            }
        }
    }

    /// Calculate liquidation price based on maintenance margin
    pub fn liquidation_price(&self, mm_rate: FixedPoint) -> Option<FixedPoint> {
        let pnl = self.unrealized_pnl();
        let equity = self.margin_balance.checked_add(pnl)?;
        
        // For long: LiqPrice = (EntryPrice * Size - Equity + MM) / Size
        // For short: LiqPrice = (EntryPrice * Size + Equity - MM) / Size
        
        let mm_required = self.size.checked_mul(mm_rate)?;
        
        match self.side {
            PositionSide::Long => {
                let numerator = self.entry_price.checked_mul(self.size)?
                    .checked_sub(equity)?
                    .checked_add(mm_required)?;
                numerator.checked_div(self.size)
            }
            PositionSide::Short => {
                let numerator = self.entry_price.checked_mul(self.size)?
                    .checked_add(equity)?
                    .checked_sub(mm_required)?;
                numerator.checked_div(self.size)
            }
        }
    }

    /// Check if position is at risk of liquidation
    pub fn is_at_risk(&self, mm_rate: FixedPoint, safety_buffer: FixedPoint) -> bool {
        if let Some(liq_price) = self.liquidation_price(mm_rate) {
            let price_diff = if self.side == PositionSide::Long {
                self.mark_price.checked_sub(liq_price).unwrap_or(FixedPoint(0))
            } else {
                liq_price.checked_sub(self.mark_price).unwrap_or(FixedPoint(0))
            };
            
            // Normalize by mark price to get percentage
            if let Some(diff_pct) = price_diff.checked_div(self.mark_price) {
                diff_pct < safety_buffer
            } else {
                true
            }
        } else {
            true // If we can't calculate, assume at risk
        }
    }
}

/// Result of liquidation simulation
#[derive(Debug, Clone)]
pub struct LiquidationResult {
    pub will_liquidate: bool,
    pub estimated_liq_price: Option<FixedPoint>,
    pub adl_trigger_level: u32,
    pub insurance_fund_impact: FixedPoint,
    pub recommended_action: RiskAction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiskAction {
    Hold,
    ReducePosition(FixedPoint), // Reduce by this amount
    CloseAll,
    TriggerKillSwitch,
}

/// Main Liquidation Simulator
pub struct LiquidationSimulator {
    pub tiers: Vec<MarginTier>,
    pub insurance_fund: InsuranceFund,
    pub adl_queue: Vec<AdlEntry>,
    pub safety_buffer_pct: FixedPoint,
    pub simulation_counter: AtomicU64,
}

impl LiquidationSimulator {
    pub fn new(tiers: Vec<MarginTier>, insurance_fund: InsuranceFund) -> Self {
        LiquidationSimulator {
            tiers,
            insurance_fund,
            adl_queue: Vec::with_capacity(1024),
            safety_buffer_pct: FixedPoint::from_f64(0.02), // 2% safety buffer
            simulation_counter: AtomicU64::new(0),
        }
    }

    /// Get applicable margin tier for a given notional value
    pub fn get_tier_for_notional(&self, notional: FixedPoint) -> Option<&MarginTier> {
        self.tiers.iter().find(|tier| {
            notional.0 >= tier.lower_bound.0 && notional.0 < tier.upper_bound.0
        })
    }

    /// Simulate liquidation scenario for a position
    pub fn simulate_liquidation(&self, position: &Position) -> LiquidationResult {
        self.simulation_counter.fetch_add(1, Ordering::Relaxed);
        
        let notional = position.size.checked_mul(position.mark_price).unwrap_or(FixedPoint(0));
        let tier = match self.get_tier_for_notional(notional) {
            Some(t) => t,
            None => return LiquidationResult {
                will_liquidate: false,
                estimated_liq_price: None,
                adl_trigger_level: 0,
                insurance_fund_impact: FixedPoint(0),
                recommended_action: RiskAction::Hold,
            },
        };

        let liq_price = position.liquidation_price(tier.maintenance_margin_rate);
        let is_at_risk = position.is_at_risk(tier.maintenance_margin_rate, self.safety_buffer_pct);

        if !is_at_risk {
            return LiquidationResult {
                will_liquidate: false,
                estimated_liq_price: liq_price,
                adl_trigger_level: 0,
                insurance_fund_impact: FixedPoint(0),
                recommended_action: RiskAction::Hold,
            };
        }

        // Calculate potential loss at liquidation
        let potential_loss = match position.side {
            PositionSide::Long => {
                let entry_value = position.entry_price.checked_mul(position.size).unwrap_or(FixedPoint(0));
                let liq_value = liq_price.unwrap_or(FixedPoint(0)).checked_mul(position.size).unwrap_or(FixedPoint(0));
                entry_value.checked_sub(liq_value).unwrap_or(FixedPoint(0))
            }
            PositionSide::Short => {
                let liq_value = liq_price.unwrap_or(FixedPoint(0)).checked_mul(position.size).unwrap_or(FixedPoint(0));
                let entry_value = position.entry_price.checked_mul(position.size).unwrap_or(FixedPoint(0));
                liq_value.checked_sub(entry_value).unwrap_or(FixedPoint(0))
            }
        };

        // Determine ADL level based on profitability and leverage
        let adl_level = self.calculate_adl_level(position, potential_loss);

        // Check if insurance fund can cover
        let insurance_impact = if self.insurance_fund.can_cover(potential_loss) {
            potential_loss
        } else {
            self.insurance_fund.balance
        };

        let action = if adl_level >= 3 {
            RiskAction::TriggerKillSwitch
        } else if adl_level >= 2 {
            RiskAction::CloseAll
        } else if is_at_risk {
            // Calculate reduction amount (50% of position)
            let reduce_by = position.size.checked_div(FixedPoint::from_f64(2.0)).unwrap_or(position.size);
            RiskAction::ReducePosition(reduce_by)
        } else {
            RiskAction::Hold
        };

        LiquidationResult {
            will_liquidate: true,
            estimated_liq_price: liq_price,
            adl_trigger_level: adl_level,
            insurance_fund_impact: insurance_impact,
            recommended_action: action,
        }
    }

    /// Calculate ADL trigger level (0-4, where 4 is highest priority for deleveraging)
    fn calculate_adl_level(&self, position: &Position, loss: FixedPoint) -> u32 {
        let leverage_score = position.leverage.min(125) / 25; // 0-5 scale
        let pnl_ratio = if position.margin_balance.0 > 0 {
            (loss.0 * 100 / position.margin_balance.0) as u32
        } else {
            100
        };
        let pnl_score = pnl_ratio.min(100) / 25; // 0-4 scale

        leverage_score.saturating_add(pnl_score).min(4)
    }

    /// Process ADL queue and determine which positions to deleverage
    pub fn process_adl_queue(&mut self, total_loss: FixedPoint) -> Vec<u64> {
        let mut positions_to_adl = Vec::new();
        let mut remaining_loss = total_loss;

        // Sort by ADL rank (descending)
        self.adl_queue.sort_by(|a, b| b.adl_rank.cmp(&a.adl_rank));

        for entry in &self.adl_queue {
            if remaining_loss.0 == 0 {
                break;
            }

            // Calculate how much of this position needs to be ADL'd
            let position_loss_share = entry.unrealized_pnl;
            if position_loss_share.0 > 0 && position_loss_share.0 <= remaining_loss.0 {
                positions_to_adl.push(entry.position_id);
                remaining_loss = remaining_loss.checked_sub(position_loss_share).unwrap_or(FixedPoint(0));
            }
        }

        positions_to_adl
    }

    /// Update safety buffer dynamically based on market volatility
    pub fn update_safety_buffer(&mut self, volatility: f64) {
        // Increase buffer with higher volatility
        let base_buffer = 0.02;
        let vol_adjustment = volatility * 0.5;
        let new_buffer = (base_buffer + vol_adjustment).min(0.10); // Cap at 10%
        self.safety_buffer_pct = FixedPoint::from_f64(new_buffer);
    }

    /// Add position to ADL queue
    pub fn add_to_adl_queue(&mut self, entry: AdlEntry) {
        if self.adl_queue.len() < self.adl_queue.capacity() {
            self.adl_queue.push(entry);
        }
    }

    /// Remove position from ADL queue
    pub fn remove_from_adl_queue(&mut self, position_id: u64) {
        self.adl_queue.retain(|e| e.position_id != position_id);
    }
}

/// Global liquidation state shared across threads
#[derive(Clone)]
pub struct GlobalLiquidationState {
    pub simulator: Arc<LiquidationSimulator>,
    pub kill_switch_active: Arc<AtomicU64>,
}

impl GlobalLiquidationState {
    pub fn new(simulator: LiquidationSimulator) -> Self {
        GlobalLiquidationState {
            simulator: Arc::new(simulator),
            kill_switch_active: Arc::new(AtomicU64::new(0)),
        }
    }

    pub fn trigger_kill_switch(&self) {
        self.kill_switch_active.store(1, Ordering::SeqCst);
    }

    pub fn is_kill_switch_active(&self) -> bool {
        self.kill_switch_active.load(Ordering::SeqCst) == 1
    }

    pub fn reset_kill_switch(&self) {
        self.kill_switch_active.store(0, Ordering::SeqCst);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixed_point_arithmetic() {
        let a = FixedPoint::from_f64(100.5);
        let b = FixedPoint::from_f64(2.0);
        
        let sum = a.checked_add(b).unwrap();
        assert!((sum.to_f64() - 102.5).abs() < 0.0000001);

        let product = a.checked_mul(b).unwrap();
        assert!((product.to_f64() - 201.0).abs() < 0.0000001);
    }

    #[test]
    fn test_liquidation_price_long() {
        let tiers = vec![MarginTier {
            lower_bound: FixedPoint::from_f64(0.0),
            upper_bound: FixedPoint::from_f64(1_000_000.0),
            initial_margin_rate: FixedPoint::from_f64(0.1),
            maintenance_margin_rate: FixedPoint::from_f64(0.05),
            max_leverage: 10,
        }];

        let sim = LiquidationSimulator::new(tiers, InsuranceFund::new(1000000.0, "USDT"));
        
        let position = Position {
            id: 1,
            side: PositionSide::Long,
            size: FixedPoint::from_f64(1.0),
            entry_price: FixedPoint::from_f64(50000.0),
            mark_price: FixedPoint::from_f64(50000.0),
            leverage: 10,
            margin_balance: FixedPoint::from_f64(5000.0),
            isolated: false,
        };

        let result = sim.simulate_liquidation(&position);
        assert!(!result.will_liquidate);
    }

    #[test]
    fn test_insurance_fund_operations() {
        let mut fund = InsuranceFund::new(10000.0, "USDT");
        
        let loss = FixedPoint::from_f64(5000.0);
        assert!(fund.can_cover(loss));
        
        fund.deduct(loss).unwrap();
        assert!((fund.balance.to_f64() - 5000.0).abs() < 0.0000001);
        
        let large_loss = FixedPoint::from_f64(10000.0);
        assert!(!fund.can_cover(large_loss));
    }
}
