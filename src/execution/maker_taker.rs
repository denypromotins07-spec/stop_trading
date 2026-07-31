//! Smart Execution - Maker vs. Taker Routing
//! 
//! Dynamic router deciding whether to cross the spread (taker) or join the queue (maker).
//! Evaluates predicted microprice drift against maker rebate to maximize expected value.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::marker::PhantomData;

/// Order side
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Buy,
    Sell,
}

/// Execution mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionMode {
    /// Join the queue as maker (passive)
    Maker,
    /// Cross the spread as taker (aggressive)
    Taker,
    /// Cancel and wait for better conditions
    Wait,
}

/// Order routing decision
#[derive(Debug, Clone, Copy)]
pub struct RoutingDecision {
    /// Chosen execution mode
    pub mode: ExecutionMode,
    /// Expected value of maker execution (in basis points)
    pub maker_ev_bps: f64,
    /// Expected value of taker execution (in basis points)
    pub taker_ev_bps: f64,
    /// Predicted microprice drift (bps per millisecond)
    pub drift_bps_per_ms: f64,
    /// Queue position estimate (0 = front, 1 = back)
    pub queue_position: f64,
    /// Probability of fill as maker within time horizon
    pub maker_fill_prob: f64,
    /// Recommended limit price offset from best (in ticks)
    pub limit_offset_ticks: i32,
    /// Confidence in decision (0.0 to 1.0)
    pub confidence: f64,
}

impl Default for RoutingDecision {
    fn default() -> Self {
        Self {
            mode: ExecutionMode::Wait,
            maker_ev_bps: 0.0,
            taker_ev_bps: 0.0,
            drift_bps_per_ms: 0.0,
            queue_position: 0.5,
            maker_fill_prob: 0.0,
            limit_offset_ticks: 0,
            confidence: 0.0,
        }
    }
}

/// Cache-line aligned maker-taker router state
#[repr(align(64))]
pub struct MakerTakerRouter {
    /// Maker rebate in basis points (positive for rebates, negative for fees)
    maker_rebate_bps: f64,
    /// Taker fee in basis points
    taker_fee_bps: f64,
    /// Tick size in price units
    tick_size: AtomicU64,
    /// Minimum edge threshold to execute (bps)
    min_edge_bps: f64,
    /// Time horizon for fill probability (milliseconds)
    time_horizon_ms: u64,
    /// Whether router is enabled
    enabled: AtomicBool,
    _pad: PhantomData<[u8; 32]>,
}

impl MakerTakerRouter {
    /// Create new router with given fee structure
    pub const fn new(maker_rebate_bps: f64, taker_fee_bps: f64, tick_size: u64) -> Self {
        Self {
            maker_rebate_bps,
            taker_fee_bps,
            tick_size: AtomicU64::new(tick_size),
            min_edge_bps: 0.5, // Minimum 0.5 bps edge required
            time_horizon_ms: 100, // 100ms default horizon
            enabled: AtomicBool::new(true),
            _pad: PhantomData,
        }
    }

    /// Set minimum edge threshold
    #[inline]
    pub fn set_min_edge(&mut self, min_edge_bps: f64) {
        self.min_edge_bps = min_edge_bps;
    }

    /// Set time horizon for probability calculations
    #[inline]
    pub fn set_time_horizon(&mut self, time_horizon_ms: u64) {
        self.time_horizon_ms = time_horizon_ms;
    }

    /// Enable/disable router
    #[inline]
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Relaxed);
    }

    /// Check if enabled
    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }

    /// Calculate expected value for maker execution
    /// 
    /// EV_maker = (fill_prob * (alpha + rebate)) - ((1 - fill_prob) * opportunity_cost)
    #[inline]
    fn calculate_maker_ev(
        &self,
        fill_prob: f64,
        alpha_bps: f64,
        queue_position: f64,
    ) -> f64 {
        if !self.enabled.load(Ordering::Relaxed) {
            return f64::NEG_INFINITY;
        }

        // Alpha gained while waiting in queue
        let alpha_capture = alpha_bps * (1.0 - queue_position);

        // Base EV from rebate and alpha
        let base_ev = fill_prob * (alpha_capture + self.maker_rebate_bps);

        // Opportunity cost: missing the move if not filled
        let opportunity_cost = (1.0 - fill_prob) * alpha_bps.abs() * 0.5;

        base_ev - opportunity_cost
    }

    /// Calculate expected value for taker execution
    /// 
    /// EV_taker = alpha - taker_fee - slippage
    #[inline]
    fn calculate_taker_ev(&self, alpha_bps: f64, slippage_bps: f64) -> f64 {
        if !self.enabled.load(Ordering::Relaxed) {
            return f64::NEG_INFINITY;
        }

        // Immediate alpha capture minus costs
        alpha_bps - self.taker_fee_bps - slippage_bps
    }

    /// Estimate fill probability based on queue dynamics
    #[inline]
    fn estimate_fill_probability(
        &self,
        queue_imbalance: f64,
        depletion_rate: f64,
        our_queue_position: f64,
        time_horizon_ms: u64,
    ) -> f64 {
        // Base probability from queue position (front has higher prob)
        let position_factor = 1.0 - our_queue_position;

        // Queue imbalance factor (imbalance toward our side increases fill prob)
        let imbalance_factor = if queue_imbalance > 0.0 {
            0.5 + queue_imbalance * 0.5
        } else {
            0.5
        };

        // Depletion rate factor (faster depletion = higher fill prob)
        let depletion_factor = (depletion_rate * time_horizon_ms as f64 / 1000.0).min(1.0);

        // Combine factors
        (position_factor * 0.4 + imbalance_factor * 0.3 + depletion_factor * 0.3).clamp(0.0, 1.0)
    }

    /// Determine optimal limit price offset
    #[inline]
    fn determine_limit_offset(
        &self,
        drift_bps: f64,
        side: Side,
        current_spread_ticks: u64,
    ) -> i32 {
        if current_spread_ticks <= 1 {
            return 0; // At top, no room to improve
        }

        // If drift is favorable, we can be more passive
        let favorable_drift = match side {
            Side::Buy => drift_bps < 0.0, // Price going down is good for buys
            Side::Sell => drift_bps > 0.0, // Price going up is good for sells
        };

        if favorable_drift && current_spread_ticks >= 3 {
            // Can afford to sit inside the spread
            1
        } else if drift_bps.abs() > 5.0 {
            // Strong drift, need to be aggressive
            0
        } else {
            // Normal conditions, stay at best
            0
        }
    }

    /// Main routing decision function
    /// 
    /// # Arguments
    /// * `side` - Order side (buy/sell)
    /// * `alpha_bps` - Predicted alpha in basis points over time horizon
    /// * `drift_bps_per_ms` - Microprice drift in bps/ms
    /// * `queue_imbalance` - Queue imbalance score (-1 to 1)
    /// * `depletion_rate` - Volume depletion rate (per ms)
    /// * `our_queue_position` - Estimated position in queue (0=front, 1=back)
    /// * `current_spread_ticks` - Current spread in ticks
    /// * `slippage_bps` - Estimated slippage for taker execution
    pub fn decide(
        &self,
        side: Side,
        alpha_bps: f64,
        drift_bps_per_ms: f64,
        queue_imbalance: f64,
        depletion_rate: f64,
        our_queue_position: f64,
        current_spread_ticks: u64,
        slippage_bps: f64,
    ) -> RoutingDecision {
        if !self.enabled.load(Ordering::Relaxed) {
            return RoutingDecision {
                mode: ExecutionMode::Wait,
                ..Default::default()
            };
        }

        // Adjust alpha for drift over time horizon
        let drift_adjustment = drift_bps_per_ms * self.time_horizon_ms as f64;
        let adjusted_alpha = match side {
            Side::Buy => alpha_bps - drift_adjustment, // Drift up hurts buy alpha
            Side::Sell => alpha_bps + drift_adjustment, // Drift up helps sell alpha
        };

        // Calculate fill probability
        let fill_prob = self.estimate_fill_probability(
            queue_imbalance,
            depletion_rate,
            our_queue_position,
            self.time_horizon_ms,
        );

        // Calculate expected values
        let maker_ev = self.calculate_maker_ev(fill_prob, adjusted_alpha, our_queue_position);
        let taker_ev = self.calculate_taker_ev(adjusted_alpha, slippage_bps);

        // Determine limit offset
        let limit_offset = self.determine_limit_offset(drift_bps_per_ms, side, current_spread_ticks);

        // Make decision based on EV comparison
        let (mode, confidence) = if maker_ev > taker_ev && maker_ev > self.min_edge_bps {
            (ExecutionMode::Maker, (maker_ev / (maker_ev - taker_ev).abs().max(1.0)).min(1.0))
        } else if taker_ev > maker_ev && taker_ev > self.min_edge_bps {
            (ExecutionMode::Taker, (taker_ev / (taker_ev - maker_ev).abs().max(1.0)).min(1.0))
        } else if maker_ev.max(taker_ev) > -self.min_edge_bps {
            // Marginal edge, prefer maker to capture rebate
            (ExecutionMode::Maker, 0.3)
        } else {
            // No edge, wait
            (ExecutionMode::Wait, 0.0)
        };

        RoutingDecision {
            mode,
            maker_ev_bps: maker_ev,
            taker_ev_bps: taker_ev,
            drift_bps_per_ms: drift_bps_per_ms,
            queue_position: our_queue_position,
            maker_fill_prob: fill_prob,
            limit_offset_ticks: limit_offset,
            confidence,
        }
    }

    /// Quick decision for high-frequency path (simplified)
    #[inline]
    pub fn quick_decide(
        &self,
        maker_ev: f64,
        taker_ev: f64,
    ) -> ExecutionMode {
        if !self.enabled.load(Ordering::Relaxed) {
            return ExecutionMode::Wait;
        }

        if maker_ev > taker_ev && maker_ev > self.min_edge_bps {
            ExecutionMode::Maker
        } else if taker_ev > maker_ev && taker_ev > self.min_edge_bps {
            ExecutionMode::Taker
        } else if maker_ev.max(taker_ev) > -self.min_edge_bps {
            ExecutionMode::Maker
        } else {
            ExecutionMode::Wait
        }
    }
}

/// Builder for configuring router
pub struct RouterBuilder {
    maker_rebate_bps: f64,
    taker_fee_bps: f64,
    tick_size: u64,
    min_edge_bps: f64,
    time_horizon_ms: u64,
}

impl RouterBuilder {
    pub fn new(tick_size: u64) -> Self {
        Self {
            maker_rebate_bps: 0.0,
            taker_fee_bps: 0.0,
            tick_size,
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

    pub fn time_horizon(mut self, time_horizon_ms: u64) -> Self {
        self.time_horizon_ms = time_horizon_ms;
        self
    }

    pub fn build(self) -> MakerTakerRouter {
        let mut router = MakerTakerRouter::new(
            self.maker_rebate_bps,
            self.taker_fee_bps,
            self.tick_size,
        );
        router.set_min_edge(self.min_edge_bps);
        router.set_time_horizon(self.time_horizon_ms);
        router
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_maker_preferred_with_rebate() {
        let router = RouterBuilder::new(100)
            .maker_rebate(0.5) // 0.5 bps rebate
            .taker_fee(0.5)    // 0.5 bps fee
            .min_edge(0.2)
            .build();

        let decision = router.decide(
            Side::Buy,
            2.0,       // 2 bps alpha
            0.01,      // Small drift
            0.3,       // Slight buy imbalance
            100.0,     // Moderate depletion
            0.3,       // Good queue position
            5,         // 5 tick spread
            0.3,       // 0.3 bps slippage
        );

        assert_eq!(decision.mode, ExecutionMode::Maker);
        assert!(decision.maker_ev_bps > decision.taker_ev_bps);
    }

    #[test]
    fn test_taker_preferred_with_strong_alpha() {
        let router = RouterBuilder::new(100)
            .maker_rebate(0.3)
            .taker_fee(0.5)
            .min_edge(0.2)
            .build();

        let decision = router.decide(
            Side::Sell,
            10.0,      // Strong 10 bps alpha
            0.5,       // Strong drift helping seller
            -0.5,      // Sell imbalance (might miss fill)
            50.0,      // Low depletion
            0.8,       // Bad queue position
            3,         // 3 tick spread
            0.5,       // 0.5 bps slippage
        );

        // With strong alpha and bad queue position, taker should be preferred
        assert_eq!(decision.mode, ExecutionMode::Taker);
    }

    #[test]
    fn test_wait_when_no_edge() {
        let router = RouterBuilder::new(100)
            .maker_rebate(0.2)
            .taker_fee(0.5)
            .min_edge(1.0) // High threshold
            .build();

        let decision = router.decide(
            Side::Buy,
            0.3,       // Very small alpha
            0.0,       // No drift
            0.0,       // Neutral imbalance
            0.0,       // No depletion
            0.5,       // Middle of queue
            2,         // Tight spread
            0.2,       // Small slippage
        );

        assert_eq!(decision.mode, ExecutionMode::Wait);
    }

    #[test]
    fn test_quick_decision() {
        let router = MakerTakerRouter::new(0.3, 0.5, 100);

        assert_eq!(
            router.quick_decide(1.0, 0.5),
            ExecutionMode::Maker
        );
        assert_eq!(
            router.quick_decide(0.5, 1.5),
            ExecutionMode::Taker
        );
        assert_eq!(
            router.quick_decide(-1.0, -0.5),
            ExecutionMode::Wait
        );
    }
}
