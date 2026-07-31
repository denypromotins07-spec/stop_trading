//! Execution Routing Module Root
//! 
//! Manages the trade-off between immediate execution certainty and rebate capture.
//! Integrates maker-taker routing with queue jumping logic.

pub mod maker_taker;
pub mod queue_jumping;

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use self::maker_taker::{MakerTakerRouter, RoutingDecision, ExecutionMode, Side as OrderSide};
use self::queue_jumping::{QueueJumper, QueueJumpDecision};

/// Combined execution decision
#[derive(Debug, Clone, Copy)]
pub struct ExecutionPlan {
    /// Final execution mode
    pub mode: ExecutionMode,
    /// Limit price (0 for market orders)
    pub limit_price: u64,
    /// Order size in base units
    pub size: u64,
    /// Side of order
    pub side: OrderSide,
    /// Expected cost/rebate in bps (negative = cost, positive = rebate)
    pub expected_cost_bps: f64,
    /// Confidence in execution plan (0-1)
    pub confidence: f64,
    /// Whether to use queue jumping
    pub use_queue_jump: bool,
    /// Timestamp
    pub timestamp_ns: u64,
}

impl Default for ExecutionPlan {
    fn default() -> Self {
        Self {
            mode: ExecutionMode::Wait,
            limit_price: 0,
            size: 0,
            side: OrderSide::Buy,
            expected_cost_bps: 0.0,
            confidence: 0.0,
            use_queue_jump: false,
            timestamp_ns: 0,
        }
    }
}

/// Cache-line aligned execution router
#[repr(align(64))]
pub struct ExecutionRouter {
    maker_taker_router: MakerTakerRouter,
    queue_jumper: QueueJumper,
    /// Maker rebate in bps
    maker_rebate_bps: f64,
    /// Taker fee in bps
    taker_fee_bps: f64,
    /// Fill probability slope for queue calculations
    fill_prob_slope: f64,
    /// Enabled flag
    enabled: AtomicBool,
    /// Sequence counter
    sequence: AtomicU64,
}

impl ExecutionRouter {
    /// Create new execution router
    pub fn new(
        tick_size: u64,
        maker_rebate_bps: f64,
        taker_fee_bps: f64,
    ) -> Self {
        let maker_taker = MakerTakerRouter::new(maker_rebate_bps, taker_fee_bps, tick_size);
        let queue_jumper = QueueJumper::new(tick_size);

        Self {
            maker_taker_router: maker_taker,
            queue_jumper,
            maker_rebate_bps,
            taker_fee_bps,
            fill_prob_slope: 0.5,
            enabled: AtomicBool::new(true),
            sequence: AtomicU64::new(0),
        }
    }

    /// Set fill probability slope
    #[inline]
    pub fn set_fill_prob_slope(&mut self, slope: f64) {
        self.fill_prob_slope = slope;
    }

    /// Enable/disable router
    #[inline]
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Relaxed);
        self.maker_taker_router.set_enabled(enabled);
        self.queue_jumper.set_enabled(enabled);
    }

    /// Check if enabled
    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }

    /// Update order book prices
    #[inline]
    pub fn update_prices(&self, best_bid: u64, best_ask: u64) {
        self.queue_jumper.update_prices(best_bid, best_ask);
    }

    /// Update queue state
    #[inline]
    pub fn update_queue_state(&self, orders_ahead: u64) {
        self.queue_jumper.update_orders_ahead(orders_ahead);
    }

    /// Generate execution plan for an order
    /// 
    /// # Arguments
    /// * `side` - Order side (buy/sell)
    /// * `size` - Order size in base units
    /// * `alpha_bps` - Predicted alpha in basis points
    /// * `drift_bps_per_ms` - Microprice drift in bps/ms
    /// * `queue_imbalance` - Queue imbalance score (-1 to 1)
    /// * `depletion_rate` - Volume depletion rate
    /// * `current_queue_position` - Estimated position in queue (0-1)
    /// * `time_urgency` - Urgency factor (0-1)
    /// * `best_bid` - Current best bid price
    /// * `best_ask` - Current best ask price
    /// * `timestamp_ns` - Current timestamp in nanoseconds
    pub fn create_execution_plan(
        &self,
        side: OrderSide,
        size: u64,
        alpha_bps: f64,
        drift_bps_per_ms: f64,
        queue_imbalance: f64,
        depletion_rate: f64,
        current_queue_position: f64,
        time_urgency: f64,
        best_bid: u64,
        best_ask: u64,
        timestamp_ns: u64,
    ) -> ExecutionPlan {
        if !self.enabled.load(Ordering::Relaxed) {
            return ExecutionPlan::default();
        }

        // Update prices
        self.update_prices(best_bid, best_ask);

        // Calculate spread in ticks
        let tick_size = self.queue_jumper.spread_ticks();
        let spread_ticks = if tick_size > 0 {
            (best_ask.saturating_sub(best_bid)) / tick_size
        } else {
            2 // Default assumption
        };

        // Estimate slippage based on size and liquidity
        let slippage_bps = self.estimate_slippage(size, side, best_bid, best_ask);

        // Get maker-taker routing decision
        let mt_decision = self.maker_taker_router.decide(
            side,
            alpha_bps,
            drift_bps_per_ms,
            queue_imbalance,
            depletion_rate,
            current_queue_position,
            spread_ticks,
            slippage_bps,
        );

        // Analyze queue jumping opportunity
        let jump_decision = self.queue_jumper.analyze_jump(
            side,
            alpha_bps,
            current_queue_position,
            self.maker_rebate_bps,
            self.fill_prob_slope,
            time_urgency,
        );

        // Combine decisions
        let (final_mode, limit_price, use_queue_jump) = self.combine_decisions(
            &mt_decision,
            &jump_decision,
            side,
            best_bid,
            best_ask,
        );

        // Calculate expected cost
        let expected_cost = match final_mode {
            ExecutionMode::Maker => -self.maker_rebate_bps, // Negative = rebate
            ExecutionMode::Taker => self.taker_fee_bps + slippage_bps,
            ExecutionMode::Wait => 0.0,
        };

        // Combine confidence scores
        let confidence = (mt_decision.confidence * 0.6 + jump_decision.confidence * 0.4).max(0.0).min(1.0);

        self.sequence.fetch_add(1, Ordering::AcqRel);

        ExecutionPlan {
            mode: final_mode,
            limit_price,
            size,
            side,
            expected_cost_bps: expected_cost,
            confidence,
            use_queue_jump,
            timestamp_ns,
        }
    }

    /// Combine maker-taker and queue jump decisions
    fn combine_decisions(
        &self,
        mt: &RoutingDecision,
        jump: &QueueJumpDecision,
        side: OrderSide,
        best_bid: u64,
        best_ask: u64,
    ) -> (ExecutionMode, u64, bool) {
        let tick_size = self.queue_jumper.spread_ticks().max(1);

        match mt.mode {
            ExecutionMode::Maker => {
                if jump.should_jump && jump.net_benefit_bps > 0.0 {
                    // Queue jumping is beneficial
                    let new_price = match side {
                        OrderSide::Buy => {
                            let spread_ticks = (best_ask - best_bid) / tick_size;
                            if spread_ticks >= 2 {
                                best_bid + tick_size
                            } else {
                                best_ask
                            }
                        }
                        OrderSide::Sell => {
                            let spread_ticks = (best_ask - best_bid) / tick_size;
                            if spread_ticks >= 2 {
                                best_ask.saturating_sub(tick_size)
                            } else {
                                best_bid
                            }
                        }
                    };
                    (ExecutionMode::Maker, new_price, true)
                } else {
                    // Stay at best price
                    let price = match side {
                        OrderSide::Buy => best_bid,
                        OrderSide::Sell => best_ask,
                    };
                    (ExecutionMode::Maker, price, false)
                }
            }
            ExecutionMode::Taker => {
                // Taker always crosses spread
                let price = match side {
                    OrderSide::Buy => best_ask,
                    OrderSide::Sell => best_bid,
                };
                (ExecutionMode::Taker, price, false)
            }
            ExecutionMode::Wait => {
                (ExecutionMode::Wait, 0, false)
            }
        }
    }

    /// Estimate slippage based on order size
    fn estimate_slippage(&self, size: u64, side: OrderSide, best_bid: u64, best_ask: u64) -> f64 {
        // Simple linear model: larger orders = more slippage
        // In production, this would use real liquidity data
        let reference_price = match side {
            OrderSide::Buy => best_ask,
            OrderSide::Sell => best_bid,
        };

        if reference_price == 0 {
            return 1.0; // Default 1 bps
        }

        // Assume 0.1 bps per 1% of average daily volume
        // This is a simplified model
        let size_factor = (size as f64 / 1000000.0).min(10.0); // Cap at 10M units
        size_factor * 0.1
    }

    /// Get sequence number
    #[inline]
    pub fn sequence(&self) -> u64 {
        self.sequence.load(Ordering::Acquire)
    }

    /// Reset router state
    pub fn reset(&self) {
        self.sequence.store(0, Ordering::Relaxed);
    }
}

/// Builder for execution router
pub struct RouterBuilder {
    tick_size: u64,
    maker_rebate_bps: f64,
    taker_fee_bps: f64,
    min_edge_bps: f64,
    time_horizon_ms: u64,
}

impl RouterBuilder {
    pub fn new(tick_size: u64) -> Self {
        Self {
            tick_size,
            maker_rebate_bps: 0.0,
            taker_fee_bps: 0.0,
            min_edge_bps: 0.5,
            time_horizon_ms: 100,
        }
    }

    pub fn maker_rebate(mut self, rebate_bps: f64) -> Self {
        self.maker_rebate_bps = rebate_bps;
        self
    }

    pub fn taker_fee(mut self, fee_bps: f64) -> Self {
        self.taker_fee_bps = fee_bps;
        self
    }

    pub fn min_edge(mut self, min_edge_bps: f64) -> Self {
        self.min_edge_bps = min_edge_bps;
        self
    }

    pub fn time_horizon(mut self, horizon_ms: u64) -> Self {
        self.time_horizon_ms = horizon_ms;
        self
    }

    pub fn build(self) -> ExecutionRouter {
        let mut router = ExecutionRouter::new(
            self.tick_size,
            self.maker_rebate_bps,
            self.taker_fee_bps,
        );

        // Configure inner routers
        // Note: Would need setter methods for full configuration

        router
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_execution_plan_maker() {
        let router = ExecutionRouter::new(100, 0.3, 0.5);

        let plan = router.create_execution_plan(
            OrderSide::Buy,
            1000,
            2.0,       // Alpha
            0.01,      // Drift
            0.3,       // Imbalance
            100.0,     // Depletion
            0.3,       // Queue position
            0.5,       // Urgency
            9900,      // Best bid
            10000,     // Best ask
            1000000,   // Timestamp
        );

        assert!(plan.confidence > 0.0);
        assert!(plan.size == 1000);
        assert!(plan.side == OrderSide::Buy);
    }

    #[test]
    fn test_execution_plan_taker_urgent() {
        let router = ExecutionRouter::new(100, 0.3, 0.5);

        let plan = router.create_execution_plan(
            OrderSide::Sell,
            5000,
            8.0,       // Strong alpha
            0.5,       // Strong drift
            -0.5,      // Against us
            200.0,     // Fast depletion
            0.8,       // Bad position
            0.9,       // Very urgent
            9900,
            10000,
            2000000,
        );

        // With high urgency and strong alpha, might be taker
        assert!(plan.confidence > 0.0);
    }

    #[test]
    fn test_router_enable_disable() {
        let router = ExecutionRouter::new(100, 0.3, 0.5);

        router.set_enabled(false);

        let plan = router.create_execution_plan(
            OrderSide::Buy,
            1000,
            5.0,
            0.1,
            0.5,
            100.0,
            0.5,
            0.5,
            9900,
            10000,
            3000000,
        );

        assert_eq!(plan.mode, ExecutionMode::Wait);
        assert_eq!(plan.confidence, 0.0);
    }
}
