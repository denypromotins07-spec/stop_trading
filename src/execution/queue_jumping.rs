//! Queue Jumping / Tick-Jumping Logic
//! 
//! Implements minimum price improvement (tick-jumping) logic to gain priority 
//! in the order book without sacrificing core statistical edge.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::marker::PhantomData;

/// Result of queue jumping analysis
#[derive(Debug, Clone, Copy)]
pub struct QueueJumpDecision {
    /// Whether to jump the queue
    pub should_jump: bool,
    /// New price level (in ticks from current best)
    pub new_price_ticks: i32,
    /// Expected edge gained from jumping (bps)
    pub edge_gained_bps: f64,
    /// Cost of jumping (rebate loss + worse price) in bps
    pub jump_cost_bps: f64,
    /// Net benefit of jumping (edge - cost) in bps
    pub net_benefit_bps: f64,
    /// Priority gain estimate (0-1, higher = more position improvement)
    pub priority_gain: f64,
    /// Confidence in decision (0-1)
    pub confidence: f64,
}

impl Default for QueueJumpDecision {
    fn default() -> Self {
        Self {
            should_jump: false,
            new_price_ticks: 0,
            edge_gained_bps: 0.0,
            jump_cost_bps: 0.0,
            net_benefit_bps: 0.0,
            priority_gain: 0.0,
            confidence: 0.0,
        }
    }
}

/// Cache-line aligned queue jumper state
#[repr(align(64))]
pub struct QueueJumper {
    /// Tick size in price units
    tick_size: AtomicU64,
    /// Minimum edge threshold to justify jumping (bps)
    min_edge_threshold_bps: f64,
    /// Maximum acceptable cost for jumping (bps)
    max_jump_cost_bps: f64,
    /// Current best bid price
    best_bid: AtomicU64,
    /// Current best ask price
    best_ask: AtomicU64,
    /// Our current queue position (estimated orders ahead)
    orders_ahead: AtomicU64,
    /// Enabled flag
    enabled: AtomicBool,
    _pad: PhantomData<[u8; 32]>,
}

impl QueueJumper {
    /// Create new queue jumper
    pub const fn new(tick_size: u64) -> Self {
        Self {
            tick_size: AtomicU64::new(tick_size),
            min_edge_threshold_bps: 0.3, // 0.3 bps minimum edge
            max_jump_cost_bps: 0.5,      // Max 0.5 bps cost acceptable
            best_bid: AtomicU64::new(0),
            best_ask: AtomicU64::new(0),
            orders_ahead: AtomicU64::new(0),
            enabled: AtomicBool::new(true),
            _pad: PhantomData,
        }
    }

    /// Set minimum edge threshold
    #[inline]
    pub fn set_min_edge_threshold(&mut self, threshold_bps: f64) {
        self.min_edge_threshold_bps = threshold_bps;
    }

    /// Set maximum jump cost
    #[inline]
    pub fn set_max_jump_cost(&mut self, max_cost_bps: f64) {
        self.max_jump_cost_bps = max_cost_bps;
    }

    /// Update best bid/ask prices
    #[inline]
    pub fn update_prices(&self, best_bid: u64, best_ask: u64) {
        self.best_bid.store(best_bid, Ordering::Relaxed);
        self.best_ask.store(best_ask, Ordering::Relaxed);
    }

    /// Update estimated orders ahead in queue
    #[inline]
    pub fn update_orders_ahead(&self, orders: u64) {
        self.orders_ahead.store(orders, Ordering::Relaxed);
    }

    /// Enable/disable queue jumping
    #[inline]
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Relaxed);
    }

    /// Check if enabled
    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }

    /// Calculate cost of jumping one tick
    /// 
    /// Cost = lost rebate + price improvement cost
    #[inline]
    fn calculate_jump_cost(&self, side: Side, maker_rebate_bps: f64) -> f64 {
        // Price improvement cost: we get worse price by one tick
        let best_price = match side {
            Side::Buy => self.best_ask.load(Ordering::Relaxed),
            Side::Sell => self.best_bid.load(Ordering::Relaxed),
        };

        if best_price == 0 {
            return f64::MAX;
        }

        let tick_size = self.tick_size.load(Ordering::Relaxed);
        let price_improvement_cost_bps = (tick_size as f64 / best_price as f64) * 10000.0;

        // When jumping, we might lose maker status and become taker
        // This depends on exchange rules, but worst case we lose full rebate
        let rebate_loss = maker_rebate_bps;

        price_improvement_cost_bps + rebate_loss
    }

    /// Estimate edge gained from improved queue position
    #[inline]
    fn estimate_edge_gained(
        &self,
        current_position: f64,
        new_position: f64,
        alpha_bps: f64,
        fill_probability_slope: f64,
    ) -> f64 {
        // Edge gained = improved fill probability * alpha
        let position_improvement = current_position - new_position;
        let fill_prob_improvement = position_improvement * fill_probability_slope;

        fill_prob_improvement * alpha_bps
    }

    /// Estimate new queue position after jumping
    #[inline]
    fn estimate_new_position(&self, current_orders_ahead: u64, jump_ticks: i32) -> f64 {
        if jump_ticks <= 0 {
            // No jump or worsening position
            return (current_orders_ahead as f64 / 1000.0).min(1.0);
        }

        // Jumping ahead puts us at front of new price level
        // Assume some fraction of orders at new level
        let estimated_orders_at_new_level = current_orders_ahead as f64 * 0.3;
        
        // We're now at front of new level
        (estimated_orders_at_new_level / 1000.0).min(0.5)
    }

    /// Determine if queue jumping is beneficial
    /// 
    /// # Arguments
    /// * `side` - Order side
    /// * `alpha_bps` - Expected alpha from trade (bps)
    /// * `current_queue_position` - Current position in queue (0=front, 1=back)
    /// * `maker_rebate_bps` - Maker rebate in bps
    /// * `fill_probability_slope` - How much fill prob improves per position unit
    /// * `time_urgency` - Urgency factor (0-1, higher = more urgent)
    pub fn analyze_jump(
        &self,
        side: Side,
        alpha_bps: f64,
        current_queue_position: f64,
        maker_rebate_bps: f64,
        fill_probability_slope: f64,
        time_urgency: f64,
    ) -> QueueJumpDecision {
        if !self.enabled.load(Ordering::Relaxed) {
            return QueueJumpDecision::default();
        }

        let current_orders = self.orders_ahead.load(Ordering::Relaxed);

        // Try jumping 1 tick
        let jump_ticks = 1;
        let new_position = self.estimate_new_position(current_orders, jump_ticks);
        let jump_cost = self.calculate_jump_cost(side, maker_rebate_bps);

        // Calculate edge gained
        let edge_gained = self.estimate_edge_gained(
            current_queue_position,
            new_position,
            alpha_bps,
            fill_probability_slope,
        );

        // Adjust for time urgency (more urgent = more willing to pay cost)
        let urgency_multiplier = 1.0 + time_urgency * 0.5;
        let adjusted_edge = edge_gained * urgency_multiplier;

        // Net benefit
        let net_benefit = adjusted_edge - jump_cost;

        // Priority gain estimate
        let priority_gain = (current_queue_position - new_position).max(0.0);

        // Decision logic
        let should_jump = net_benefit > self.min_edge_threshold_bps 
            && jump_cost < self.max_jump_cost_bps
            && priority_gain > 0.1;

        let confidence = if should_jump {
            (net_benefit / self.min_edge_threshold_bps).min(1.0)
        } else {
            ((jump_cost - net_benefit).abs() / self.max_jump_cost_bps).min(1.0)
        };

        QueueJumpDecision {
            should_jump,
            new_price_ticks: if should_jump { jump_ticks } else { 0 },
            edge_gained_bps: edge_gained,
            jump_cost_bps: jump_cost,
            net_benefit_bps: net_benefit,
            priority_gain,
            confidence,
        }
    }

    /// Calculate optimal price for order placement
    /// 
    /// Returns price that maximizes edge considering queue dynamics
    #[inline]
    pub fn optimal_price(
        &self,
        side: Side,
        alpha_bps: f64,
        queue_position: f64,
        maker_rebate_bps: f64,
    ) -> u64 {
        let best_bid = self.best_bid.load(Ordering::Relaxed);
        let best_ask = self.best_ask.load(Ordering::Relaxed);
        let tick_size = self.tick_size.load(Ordering::Relaxed);

        if best_bid == 0 || best_ask == 0 || tick_size == 0 {
            return match side {
                Side::Buy => best_ask,
                Side::Sell => best_bid,
            };
        }

        let decision = self.analyze_jump(
            side,
            alpha_bps,
            queue_position,
            maker_rebate_bps,
            0.5, // fill_probability_slope
            0.5, // time_urgency
        );

        match side {
            Side::Buy => {
                if decision.should_jump {
                    // Jump inside spread or at ask
                    let spread_ticks = (best_ask - best_bid) / tick_size;
                    if spread_ticks >= 2 {
                        best_bid + tick_size // Improve by 1 tick
                    } else {
                        best_ask // Cross spread
                    }
                } else {
                    best_bid // Stay at best bid
                }
            }
            Side::Sell => {
                if decision.should_jump {
                    let spread_ticks = (best_ask - best_bid) / tick_size;
                    if spread_ticks >= 2 {
                        best_ask - tick_size // Improve by 1 tick
                    } else {
                        best_bid // Cross spread
                    }
                } else {
                    best_ask // Stay at best ask
                }
            }
        }
    }

    /// Get current spread in ticks
    #[inline]
    pub fn spread_ticks(&self) -> u64 {
        let bid = self.best_bid.load(Ordering::Relaxed);
        let ask = self.best_ask.load(Ordering::Relaxed);
        let tick_size = self.tick_size.load(Ordering::Relaxed);

        if bid > 0 && ask > 0 && tick_size > 0 {
            (ask - bid) / tick_size
        } else {
            0
        }
    }
}

/// Order side
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Buy,
    Sell,
}

/// Builder for queue jumper configuration
pub struct QueueJumperBuilder {
    tick_size: u64,
    min_edge_bps: f64,
    max_cost_bps: f64,
}

impl QueueJumperBuilder {
    pub fn new(tick_size: u64) -> Self {
        Self {
            tick_size,
            min_edge_bps: 0.3,
            max_cost_bps: 0.5,
        }
    }

    pub fn min_edge(mut self, min_edge_bps: f64) -> Self {
        self.min_edge_bps = min_edge_bps;
        self
    }

    pub fn max_cost(mut self, max_cost_bps: f64) -> Self {
        self.max_cost_bps = max_cost_bps;
        self
    }

    pub fn build(self) -> QueueJumper {
        let mut jumper = QueueJumper::new(self.tick_size);
        jumper.set_min_edge_threshold(self.min_edge_bps);
        jumper.set_max_jump_cost(self.max_cost_bps);
        jumper
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jump_with_high_alpha() {
        let jumper = QueueJumperBuilder::new(100)
            .min_edge(0.2)
            .max_cost(0.5)
            .build();

        jumper.update_prices(9900, 10000);
        jumper.update_orders_ahead(500);

        let decision = jumper.analyze_jump(
            Side::Buy,
            5.0,   // Strong 5 bps alpha
            0.7,   // Bad current position
            0.3,   // Maker rebate
            0.5,   // Fill prob slope
            0.8,   // High urgency
        );

        // With strong alpha and bad position, should consider jumping
        assert!(decision.edge_gained_bps > 0.0);
        assert!(decision.priority_gain > 0.0);
    }

    #[test]
    fn test_no_jump_with_low_alpha() {
        let jumper = QueueJumperBuilder::new(100)
            .min_edge(0.3)
            .max_cost(0.5)
            .build();

        jumper.update_prices(9900, 10000);
        jumper.update_orders_ahead(100);

        let decision = jumper.analyze_jump(
            Side::Sell,
            0.5,   // Very weak alpha
            0.3,   // Good current position
            0.3,   // Maker rebate
            0.5,   // Fill prob slope
            0.2,   // Low urgency
        );

        // With weak alpha and good position, should not jump
        assert!(!decision.should_jump);
    }

    #[test]
    fn test_optimal_price_calculation() {
        let jumper = QueueJumperBuilder::new(50)
            .min_edge(0.2)
            .max_cost(0.4)
            .build();

        jumper.update_prices(9950, 10000);
        jumper.update_orders_ahead(1000);

        let opt_price = jumper.optimal_price(
            Side::Buy,
            3.0,   // Decent alpha
            0.8,   // Bad position
            0.3,   // Rebate
        );

        // Should suggest improved price
        assert!(opt_price >= 9950);
        assert!(opt_price <= 10000);
    }

    #[test]
    fn test_spread_calculation() {
        let jumper = QueueJumper::new(100);
        jumper.update_prices(9900, 10000);

        assert_eq!(jumper.spread_ticks(), 10);
    }
}
