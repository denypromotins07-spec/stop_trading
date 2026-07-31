//! Threshold-Based Rebalancing Algorithm
//! 
//! Only triggers when drift exceeds transaction costs (No-Trade Region).
//! Executes atomic multi-leg rebalancing orders to minimize market impact.

use super::drift::{DriftMonitor, PositionDrift, TradeInstruction, Side, NoTradeRegion, DriftAnalysis, RebalanceAction};
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};

/// Rebalancing trigger condition
#[derive(Debug, Clone)]
pub struct RebalanceTrigger {
    pub triggered: bool,
    pub reason: TriggerReason,
    pub estimated_cost_bps: f64,
    pub expected_improvement_bps: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TriggerReason {
    NoDrift,
    WithinNoTradeRegion,
    ThresholdBreached,
    EmergencyCondition,
    TaxLossHarvesting,
}

/// Multi-leg order for atomic execution
#[derive(Debug, Clone)]
pub struct MultiLegOrder {
    pub legs: Vec<Leg>,
    pub total_value_usd: f64,
    pub net_cash_flow: f64,  // Positive = cash inflow, Negative = outflow
    pub execution_mode: ExecutionMode,
}

#[derive(Debug, Clone)]
pub struct Leg {
    pub symbol: [u8; 16],
    pub side: Side,
    pub quantity: f64,
    pub limit_price: Option<f64>,
    pub urgency: Urgency,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionMode {
    Simultaneous,   // All legs at once
    Sequential,     // One after another
    VWAP,           // Volume-weighted over time
    TWAP,           // Time-weighted over time
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Urgency {
    Low,
    Normal,
    High,
}

/// Main Rebalancing Algorithm
pub struct RebalancingAlgorithm {
    pub drift_monitor: DriftMonitor,
    pub no_trade_region: NoTradeRegion,
    pub min_rebalance_size_usd: f64,
    pub max_slippage_bps: f64,
    pub execution_enabled: AtomicBool,
    pub rebalance_counter: AtomicU64,
}

impl RebalancingAlgorithm {
    pub fn new(
        drift_threshold: f64,
        rebalance_threshold: f64,
        transaction_cost_rate: f64,
        min_size: f64,
    ) -> Self {
        RebalancingAlgorithm {
            drift_monitor: DriftMonitor::new(drift_threshold, rebalance_threshold),
            no_trade_region: NoTradeRegion::new(drift_threshold, transaction_cost_rate),
            min_rebalance_size_usd: min_size,
            max_slippage_bps: 10.0,
            execution_enabled: AtomicBool::new(true),
            rebalance_counter: AtomicU64::new(0),
        }
    }

    /// Check if rebalancing should be triggered
    pub fn check_trigger(&self, positions: &[super::drift::PositionData], targets: &[f64]) -> RebalanceTrigger {
        // Update drift calculations
        let mut monitor_copy = DriftMonitor::new(
            self.drift_monitor.drift_threshold_pct,
            self.drift_monitor.rebalance_threshold_pct,
        );
        monitor_copy.update_positions(positions, targets);
        
        let analysis = monitor_copy.analyze_drift();
        
        // Calculate transaction costs
        let turnover_value = analysis.rebalance_cost_estimate / 0.001; // Reverse engineer turnover
        let estimated_cost_bps = (analysis.rebalance_cost_estimate / turnover_value.max(1.0)) * 10000.0;
        
        // Expected improvement from rebalancing (reduction in tracking error)
        let expected_improvement_bps = analysis.max_single_drift * 10.0; // Simplified
        
        // Determine trigger reason
        let (triggered, reason) = if analysis.positions_out_of_band == 0 {
            (false, TriggerReason::NoDrift)
        } else if !self.check_no_trade_violation(&monitor_copy, targets) {
            (false, TriggerReason::WithinNoTradeRegion)
        } else if matches!(analysis.recommended_action, RebalanceAction::EmergencyRebalance) {
            (true, TriggerReason::EmergencyCondition)
        } else if expected_improvement_bps > estimated_cost_bps * 2.0 {
            (true, TriggerReason::ThresholdBreached)
        } else {
            (false, TriggerReason::WithinNoTradeRegion)
        };

        RebalanceTrigger {
            triggered,
            reason,
            estimated_cost_bps,
            expected_improvement_bps,
        }
    }

    fn check_no_trade_violation(&self, monitor: &DriftMonitor, targets: &[f64]) -> bool {
        for (i, pos) in monitor.positions.iter().enumerate() {
            let target = targets.get(i).copied().unwrap_or(pos.target_weight);
            if self.no_trade_region.should_rebalance(pos.current_weight, target) {
                return true;
            }
        }
        false
    }

    /// Build multi-leg rebalancing order
    pub fn build_rebalance_order(
        &self,
        positions: &[super::drift::PositionData],
        targets: &[f64],
        mode: ExecutionMode,
    ) -> Option<MultiLegOrder> {
        if !self.execution_enabled.load(Ordering::Relaxed) {
            return None;
        }

        // Get trade instructions
        let mut monitor_copy = DriftMonitor::new(
            self.drift_monitor.drift_threshold_pct,
            self.drift_monitor.rebalance_threshold_pct,
        );
        monitor_copy.update_positions(positions, targets);
        
        let instructions = monitor_copy.calculate_trade_sizes(self.min_rebalance_size_usd);
        
        if instructions.is_empty() {
            return None;
        }

        // Build legs
        let mut legs = Vec::with_capacity(instructions.len());
        let mut total_value = 0.0;
        let mut net_cash_flow = 0.0;

        for instr in &instructions {
            let quantity = if instr.current_price > 0.0 {
                instr.amount_usd / instr.current_price
            } else {
                0.0
            };

            let leg = Leg {
                symbol: instr.symbol,
                side: instr.side,
                quantity,
                limit_price: Some(self.calculate_limit_price(instr.current_price, instr.side)),
                urgency: Urgency::Normal,
            };

            total_value += instr.amount_usd;
            
            match instr.side {
                Side::Buy => net_cash_flow -= instr.amount_usd,
                Side::Sell => net_cash_flow += instr.amount_usd,
            }

            legs.push(leg);
        }

        Some(MultiLegOrder {
            legs,
            total_value_usd: total_value,
            net_cash_flow,
            execution_mode: mode,
        })
    }

    fn calculate_limit_price(&self, current_price: f64, side: Side) -> f64 {
        // Add slippage buffer to limit price
        let slippage_factor = 1.0 + self.max_slippage_bps / 10000.0;
        match side {
            Side::Buy => current_price * slippage_factor,
            Side::Sell => current_price / slippage_factor,
        }
    }

    /// Execute rebalancing (returns success status)
    pub fn execute_rebalance(&self, order: &MultiLegOrder) -> RebalanceResult {
        if !self.execution_enabled.load(Ordering::Relaxed) {
            return RebalanceResult {
                success: false,
                error: Some("Execution disabled".to_string()),
                filled_legs: 0,
                total_filled_value: 0.0,
            };
        }

        self.rebalance_counter.fetch_add(1, Ordering::Relaxed);

        // Validate order
        if order.legs.is_empty() {
            return RebalanceResult {
                success: false,
                error: Some("Empty order".to_string()),
                filled_legs: 0,
                total_filled_value: 0.0,
            };
        }

        // Check net cash flow constraint
        if order.net_cash_flow < -order.total_value_usd * 0.1 {
            // More than 10% cash outflow requires funding check
            // In production, would verify available cash
        }

        // Simulate execution (in production, would send to exchange)
        let filled_legs = order.legs.len();
        let total_filled = order.total_value_usd;

        RebalanceResult {
            success: true,
            error: None,
            filled_legs,
            total_filled_value: total_filled,
        }
    }

    /// Update volatility for no-trade region adjustment
    pub fn update_volatility(&mut self, annualized_vol: f64) {
        self.no_trade_region.update_volatility(annualized_vol);
    }

    /// Enable/disable execution
    pub fn set_execution_enabled(&self, enabled: bool) {
        self.execution_enabled.store(enabled, Ordering::Relaxed);
    }

    /// Get rebalance statistics
    pub fn get_stats(&self) -> RebalanceStats {
        RebalanceStats {
            total_rebalances: self.rebalance_counter.load(Ordering::Relaxed),
            execution_enabled: self.execution_enabled.load(Ordering::Relaxed),
            current_drift: self.drift_monitor.get_drift_metric(),
        }
    }
}

/// Rebalancing execution result
#[derive(Debug, Clone)]
pub struct RebalanceResult {
    pub success: bool,
    pub error: Option<String>,
    pub filled_legs: usize,
    pub total_filled_value: f64,
}

/// Rebalancing statistics
#[derive(Debug, Clone)]
pub struct RebalanceStats {
    pub total_rebalances: u64,
    pub execution_enabled: bool,
    pub current_drift: f64,
}

/// Tax-loss harvesting optimizer
pub struct TaxOptimizer {
    pub wash_sale_window_days: u32,
    pub harvested_losses: Vec<HarvestedLoss>,
}

impl TaxOptimizer {
    pub fn new(wash_sale_window: u32) -> Self {
        TaxOptimizer {
            wash_sale_window_days: wash_sale_window,
            harvested_losses: Vec::new(),
        }
    }

    /// Identify tax-loss harvesting opportunities
    pub fn find_harvesting_opportunities(
        &self,
        positions: &[super::drift::PositionData],
        cost_basis: &[f64],
    ) -> Vec<TaxHarvestOpportunity> {
        let mut opportunities = Vec::new();

        for (i, pos) in positions.iter().enumerate() {
            let basis = cost_basis.get(i).copied().unwrap_or(pos.value_usd);
            let unrealized_pnl = pos.value_usd - basis;

            if unrealized_pnl < -100.0 { // Minimum $100 loss
                // Check wash sale rule
                if !self.is_wash_sale_restricted(&pos.symbol) {
                    opportunities.push(TaxHarvestOpportunity {
                        symbol: pos.symbol,
                        unrealized_loss: unrealized_pnl.abs(),
                        current_value: pos.value_usd,
                        tax_benefit_estimate: unrealized_pnl.abs() * 0.25, // ~25% tax rate
                    });
                }
            }
        }

        opportunities
    }

    fn is_wash_sale_restricted(&self, symbol: &[u8; 16]) -> bool {
        use std::time::{SystemTime, UNIX_EPOCH};
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        for loss in &self.harvested_losses {
            if loss.symbol == *symbol {
                let days_since = (now - loss.harvest_timestamp) / 86400;
                if days_since < self.wash_sale_window_days as u64 {
                    return true;
                }
            }
        }
        false
    }

    /// Record a harvested loss
    pub fn record_harvest(&mut self, symbol: [u8; 16], amount: f64) {
        use std::time::{SystemTime, UNIX_EPOCH};
        self.harvested_losses.push(HarvestedLoss {
            symbol,
            amount,
            harvest_timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
        });
    }
}

#[derive(Debug, Clone)]
pub struct HarvestedLoss {
    pub symbol: [u8; 16],
    pub amount: f64,
    pub harvest_timestamp: u64,
}

#[derive(Debug, Clone)]
pub struct TaxHarvestOpportunity {
    pub symbol: [u8; 16],
    pub unrealized_loss: f64,
    pub current_value: f64,
    pub tax_benefit_estimate: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::drift::PositionData;

    #[test]
    fn test_rebalancing_algorithm_creation() {
        let algo = RebalancingAlgorithm::new(2.0, 5.0, 0.001, 100.0);
        assert!(!algo.execution_enabled.load(Ordering::Relaxed));
        algo.set_execution_enabled(true);
        assert!(algo.execution_enabled.load(Ordering::Relaxed));
    }

    #[test]
    fn test_trigger_check_no_drift() {
        let algo = RebalancingAlgorithm::new(5.0, 10.0, 0.001, 100.0);
        
        let positions = vec![
            PositionData { symbol: *b"A               ", value_usd: 500.0, quantity: 1.0 },
            PositionData { symbol: *b"B               ", value_usd: 500.0, quantity: 1.0 },
        ];
        let targets = vec![0.5, 0.5];
        
        let trigger = algo.check_trigger(&positions, &targets);
        assert!(!trigger.triggered);
        assert_eq!(trigger.reason, TriggerReason::NoDrift);
    }

    #[test]
    fn test_multi_leg_order_builder() {
        let algo = RebalancingAlgorithm::new(2.0, 5.0, 0.001, 50.0);
        algo.set_execution_enabled(true);
        
        let positions = vec![
            PositionData { symbol: *b"BTC             ", value_usd: 70_000.0, quantity: 1.0 },
            PositionData { symbol: *b"ETH             ", value_usd: 30_000.0, quantity: 10.0 },
        ];
        let targets = vec![0.5, 0.5];
        
        let order = algo.build_rebalance_order(&positions, &targets, ExecutionMode::Simultaneous);
        
        assert!(order.is_some());
        let order = order.unwrap();
        assert!(order.total_value_usd > 0.0);
    }

    #[test]
    fn test_tax_optimizer() {
        let optimizer = TaxOptimizer::new(30);
        
        let positions = vec![
            PositionData { symbol: *b"LOSS            ", value_usd: 500.0, quantity: 10.0 },
        ];
        let cost_basis = vec![1000.0];
        
        let opportunities = optimizer.find_harvesting_opportunities(&positions, &cost_basis);
        
        assert_eq!(opportunities.len(), 1);
        assert!((opportunities[0].unrealized_loss - 500.0).abs() < 0.01);
    }
}
