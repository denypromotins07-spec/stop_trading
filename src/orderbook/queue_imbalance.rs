//! Multi-Level Queue Imbalance Tracker
//! 
//! Tracks weighted bid/ask volumes across L2 levels to predict immediate 
//! next-tick direction and spread crossing probability.

use std::sync::atomic::{AtomicU64, AtomicI64, Ordering};
use std::marker::PhantomData;

/// Maximum L2 levels for queue imbalance tracking
pub const MAX_QUEUE_LEVELS: usize = 10;

/// Cache-line aligned queue state
#[repr(align(64))]
pub struct QueueState {
    /// Bid volumes at each level (in base units)
    bid_volumes: [AtomicU64; MAX_QUEUE_LEVELS],
    /// Ask volumes at each level
    ask_volumes: [AtomicU64; MAX_QUEUE_LEVELS],
    /// Number of orders at each bid level
    bid_order_counts: [AtomicU64; MAX_QUEUE_LEVELS],
    /// Number of orders at each ask level
    ask_order_counts: [AtomicU64; MAX_QUEUE_LEVELS],
    /// Last update timestamp (nanoseconds)
    last_update_ns: AtomicU64,
    _pad: PhantomData<[u8; 32]>,
}

impl QueueState {
    pub const fn new() -> Self {
        const ZERO: AtomicU64 = AtomicU64::new(0);
        Self {
            bid_volumes: [ZERO; MAX_QUEUE_LEVELS],
            ask_volumes: [ZERO; MAX_QUEUE_LEVELS],
            bid_order_counts: [ZERO; MAX_QUEUE_LEVELS],
            ask_order_counts: [ZERO; MAX_QUEUE_LEVELS],
            last_update_ns: AtomicU64::new(0),
            _pad: PhantomData,
        }
    }

    #[inline]
    pub fn update_bid(&self, level: usize, volume: u64, order_count: u64) {
        if level < MAX_QUEUE_LEVELS {
            self.bid_volumes[level].store(volume, Ordering::Relaxed);
            self.bid_order_counts[level].store(order_count, Ordering::Relaxed);
        }
    }

    #[inline]
    pub fn update_ask(&self, level: usize, volume: u64, order_count: u64) {
        if level < MAX_QUEUE_LEVELS {
            self.ask_volumes[level].store(volume, Ordering::Relaxed);
            self.ask_order_counts[level].store(order_count, Ordering::Relaxed);
        }
    }

    #[inline]
    pub fn set_timestamp(&self, ts_ns: u64) {
        self.last_update_ns.store(ts_ns, Ordering::Relaxed);
    }

    #[inline]
    pub fn get_bid_volume(&self, level: usize) -> u64 {
        if level < MAX_QUEUE_LEVELS {
            self.bid_volumes[level].load(Ordering::Relaxed)
        } else {
            0
        }
    }

    #[inline]
    pub fn get_ask_volume(&self, level: usize) -> u64 {
        if level < MAX_QUEUE_LEVELS {
            self.ask_volumes[level].load(Ordering::Relaxed)
        } else {
            0
        }
    }
}

/// Queue imbalance metrics
#[derive(Debug, Clone, Copy)]
pub struct QueueImbalanceMetrics {
    /// Weighted imbalance score (-1.0 to 1.0)
    pub imbalance_score: f64,
    /// Probability of spread crossing upward (bid depletion)
    pub cross_up_probability: f64,
    /// Probability of spread crossing downward (ask depletion)
    pub cross_down_probability: f64,
    /// Estimated time to next tick change (microseconds)
    pub estimated_tick_time_us: f64,
    /// Depletion rate (volume per microsecond)
    pub bid_depletion_rate: f64,
    pub ask_depletion_rate: f64,
    /// Next tick direction prediction (-1=sell, 0=neutral, 1=buy)
    pub predicted_direction: i8,
}

impl Default for QueueImbalanceMetrics {
    fn default() -> Self {
        Self {
            imbalance_score: 0.0,
            cross_up_probability: 0.5,
            cross_down_probability: 0.5,
            estimated_tick_time_us: 1000.0,
            bid_depletion_rate: 0.0,
            ask_depletion_rate: 0.0,
            predicted_direction: 0,
        }
    }
}

/// Exponential decay weights for queue levels
const LEVEL_WEIGHTS: [f64; MAX_QUEUE_LEVELS] = [
    1.0,       // Level 0 (best) - highest weight
    0.85,      // Level 1
    0.72,      // Level 2
    0.61,      // Level 3
    0.52,      // Level 4
    0.44,      // Level 5
    0.37,      // Level 6
    0.31,      // Level 7
    0.26,      // Level 8
    0.22,      // Level 9
];

/// Multi-level queue imbalance tracker
#[repr(align(64))]
pub struct QueueImbalanceTracker {
    state: QueueState,
    /// Rolling window for depletion rate calculation
    prev_bid_total: AtomicU64,
    prev_ask_total: AtomicU64,
    prev_timestamp_ns: AtomicU64,
    /// Tick size in price units
    tick_size: AtomicU64,
    _pad: PhantomData<[u8; 32]>,
}

impl QueueImbalanceTracker {
    pub const fn new(tick_size: u64) -> Self {
        const ZERO: AtomicU64 = AtomicU64::new(0);
        Self {
            state: QueueState::new(),
            prev_bid_total: ZERO,
            prev_ask_total: ZERO,
            prev_timestamp_ns: AtomicU64::new(0),
            tick_size: AtomicU64::new(tick_size),
            _pad: PhantomData,
        }
    }

    /// Update queue state at a specific level
    #[inline]
    pub fn update(&self, level: usize, bid_vol: u64, ask_vol: u64, 
                  bid_orders: u64, ask_orders: u64, timestamp_ns: u64) {
        self.state.update_bid(level, bid_vol, bid_orders);
        self.state.update_ask(level, ask_vol, ask_orders);
        self.state.set_timestamp(timestamp_ns);
    }

    /// Compute weighted total volume for bid side
    #[inline]
    fn weighted_bid_total(&self) -> f64 {
        let mut total = 0.0f64;
        for i in 0..MAX_QUEUE_LEVELS {
            let vol = self.state.get_bid_volume(i) as f64;
            total += vol * LEVEL_WEIGHTS[i];
        }
        total
    }

    /// Compute weighted total volume for ask side
    #[inline]
    fn weighted_ask_total(&self) -> f64 {
        let mut total = 0.0f64;
        for i in 0..MAX_QUEUE_LEVELS {
            let vol = self.state.get_ask_volume(i) as f64;
            total += vol * LEVEL_WEIGHTS[i];
        }
        total
    }

    /// Compute weighted order count
    #[inline]
    fn weighted_order_count(&self, is_bid: bool) -> f64 {
        let mut total = 0.0f64;
        for i in 0..MAX_QUEUE_LEVELS {
            let count = if is_bid {
                self.state.bid_order_counts[i].load(Ordering::Relaxed)
            } else {
                self.state.ask_order_counts[i].load(Ordering::Relaxed)
            } as f64;
            total += count * LEVEL_WEIGHTS[i];
        }
        total
    }

    /// Calculate average order size at best level
    #[inline]
    fn avg_order_size_best(&self, is_bid: bool) -> f64 {
        let vol = if is_bid {
            self.state.get_bid_volume(0)
        } else {
            self.state.get_ask_volume(0)
        } as f64;

        let count = if is_bid {
            self.state.bid_order_counts[0].load(Ordering::Relaxed)
        } else {
            self.state.ask_order_counts[0].load(Ordering::Relaxed)
        } as f64;

        if count > 0.0 {
            vol / count
        } else {
            0.0
        }
    }

    /// Estimate depletion rate based on recent changes
    #[inline]
    fn estimate_depletion_rate(&self, current_total: f64, prev_total: u64, 
                                elapsed_ns: u64) -> f64 {
        if elapsed_ns == 0 {
            return 0.0;
        }

        let delta = (current_total - prev_total as f64).abs();
        let elapsed_us = elapsed_ns as f64 / 1000.0;

        if elapsed_us > 0.0 {
            delta / elapsed_us
        } else {
            0.0
        }
    }

    /// Calculate probability of spread crossing based on queue depletion
    #[inline]
    fn calculate_cross_probability(
        &self,
        side_volume: f64,
        side_depletion_rate: f64,
        opposite_volume: f64,
    ) -> f64 {
        if side_volume <= 0.0 || side_depletion_rate <= 0.0 {
            return 0.5;
        }

        // Time to deplete current queue
        let time_to_deplete_us = side_volume / side_depletion_rate;

        // Convert to probability using exponential decay
        // Faster depletion => higher probability of crossing
        let base_prob = 1.0 - (-time_to_deplete_us / 5000.0).exp();

        // Adjust for relative queue sizes
        let size_factor = if opposite_volume > 0.0 {
            side_volume / (side_volume + opposite_volume)
        } else {
            0.5
        };

        (base_prob * 0.7 + size_factor * 0.3).clamp(0.0, 1.0)
    }

    /// Compute all queue imbalance metrics
    pub fn compute_metrics(&self, current_timestamp_ns: u64) -> QueueImbalanceMetrics {
        let weighted_bid = self.weighted_bid_total();
        let weighted_ask = self.weighted_ask_total();
        let total_weighted = weighted_bid + weighted_ask;

        // Calculate imbalance score
        let imbalance_score = if total_weighted > 0.0 {
            (weighted_bid - weighted_ask) / total_weighted
        } else {
            0.0
        };

        // Get previous totals for rate calculation
        let prev_bid = self.prev_bid_total.load(Ordering::Relaxed);
        let prev_ask = self.prev_ask_total.load(Ordering::Relaxed);
        let prev_ts = self.prev_timestamp_ns.load(Ordering::Relaxed);

        let elapsed_ns = current_timestamp_ns.saturating_sub(prev_ts);
        let elapsed_us = elapsed_ns as f64 / 1000.0;

        // Calculate depletion rates
        let bid_depletion_rate = if elapsed_us > 0.0 && prev_bid > 0 {
            let delta = (prev_bid as f64 - weighted_bid).max(0.0);
            delta / elapsed_us
        } else {
            0.0
        };

        let ask_depletion_rate = if elapsed_us > 0.0 && prev_ask > 0 {
            let delta = (prev_ask as f64 - weighted_ask).max(0.0);
            delta / elapsed_us
        } else {
            0.0
        };

        // Calculate crossing probabilities
        let cross_up_prob = self.calculate_cross_probability(
            weighted_bid,
            bid_depletion_rate,
            weighted_ask,
        );

        let cross_down_prob = self.calculate_cross_probability(
            weighted_ask,
            ask_depletion_rate,
            weighted_bid,
        );

        // Estimate time to next tick
        let total_depletion = bid_depletion_rate + ask_depletion_rate;
        let best_bid_vol = self.state.get_bid_volume(0) as f64;
        let best_ask_vol = self.state.get_ask_volume(0) as f64;
        
        let estimated_tick_time_us = if total_depletion > 0.0 {
            ((best_bid_vol + best_ask_vol) / 2.0) / total_depletion
        } else {
            10000.0 // Default 10ms
        };

        // Predict direction based on imbalance and depletion
        let predicted_direction = if imbalance_score > 0.3 && bid_depletion_rate < ask_depletion_rate {
            1  // Buy pressure
        } else if imbalance_score < -0.3 && ask_depletion_rate < bid_depletion_rate {
            -1 // Sell pressure
        } else if imbalance_score > 0.1 {
            1
        } else if imbalance_score < -0.1 {
            -1
        } else {
            0
        };

        // Update previous totals
        self.prev_bid_total.store(weighted_bid as u64, Ordering::Relaxed);
        self.prev_ask_total.store(weighted_ask as u64, Ordering::Relaxed);
        self.prev_timestamp_ns.store(current_timestamp_ns, Ordering::Relaxed);

        QueueImbalanceMetrics {
            imbalance_score,
            cross_up_probability: cross_up_prob,
            cross_down_probability: cross_down_prob,
            estimated_tick_time_us,
            bid_depletion_rate,
            ask_depletion_rate,
            predicted_direction,
        }
    }

    /// Get reference to internal state for direct updates
    #[inline]
    pub fn state(&self) -> &QueueState {
        &self.state
    }

    /// Reset tracker state
    #[inline]
    pub fn reset(&self) {
        self.prev_bid_total.store(0, Ordering::Relaxed);
        self.prev_ask_total.store(0, Ordering::Relaxed);
        self.prev_timestamp_ns.store(0, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_queue_imbalance_buy_pressure() {
        let tracker = QueueImbalanceTracker::new(100);

        // Heavy bid side, light ask side
        tracker.update(0, 5000, 1000, 50, 10, 1000000);
        tracker.update(1, 4000, 800, 40, 8, 1000000);
        tracker.update(2, 3000, 600, 30, 6, 1000000);

        // Advance time and update again to create depletion signal
        tracker.update(0, 4500, 1000, 45, 10, 2000000);

        let metrics = tracker.compute_metrics(2000000);

        assert!(metrics.imbalance_score > 0.0, "Should show buy imbalance");
        assert!(metrics.predicted_direction >= 0, "Should predict buy or neutral");
    }

    #[test]
    fn test_queue_imbalance_sell_pressure() {
        let tracker = QueueImbalanceTracker::new(100);

        // Light bid side, heavy ask side
        tracker.update(0, 1000, 5000, 10, 50, 1000000);
        tracker.update(1, 800, 4000, 8, 40, 1000000);
        tracker.update(2, 600, 3000, 6, 30, 1000000);

        let metrics = tracker.compute_metrics(1000000);

        assert!(metrics.imbalance_score < 0.0, "Should show sell imbalance");
        assert!(metrics.predicted_direction <= 0, "Should predict sell or neutral");
    }

    #[test]
    fn test_weights_decay() {
        assert!(LEVEL_WEIGHTS[0] > LEVEL_WEIGHTS[1]);
        assert!(LEVEL_WEIGHTS[1] > LEVEL_WEIGHTS[2]);
        assert!(LEVEL_WEIGHTS[MAX_QUEUE_LEVELS - 1] < 0.25);
    }
}
