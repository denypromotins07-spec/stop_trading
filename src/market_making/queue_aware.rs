//! Queue-Aware Market Making Logic
//!
//! Implements dynamic limit order placement based on real-time L3 queue position,
//! detecting institutional icebergs and optimizing queue priority.

use std::time::{Duration, Instant};

/// Order book queue state at a specific price level
#[derive(Debug, Clone, Copy)]
pub struct QueueState {
    /// Our position in the queue (number of orders ahead)
    pub position_ahead: u32,
    /// Total volume at this price level (in base units * 10^8)
    pub total_volume: u64,
    /// Volume ahead of us (in base units * 10^8)
    pub volume_ahead: u64,
    /// Estimated arrival rate of new orders (orders per second)
    pub arrival_rate: f64,
    /// Estimated cancellation rate (cancellations per second)
    pub cancellation_rate: f64,
    /// Last update timestamp
    pub last_update: Instant,
}

impl QueueState {
    pub fn new() -> Self {
        QueueState {
            position_ahead: 0,
            total_volume: 0,
            volume_ahead: 0,
            arrival_rate: 0.0,
            cancellation_rate: 0.0,
            last_update: Instant::now(),
        }
    }

    /// Update queue state with new L3 data
    #[inline]
    pub fn update(&mut self, position_ahead: u32, volume_ahead: u64, total_volume: u64) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_update).as_secs_f64();
        
        // Update rates using exponential moving average
        if elapsed > 0.001 {
            let alpha = 0.3; // Smoothing factor
            let order_arrival = if position_ahead > self.position_ahead {
                (position_ahead - self.position_ahead) as f64 / elapsed
            } else {
                0.0
            };
            
            let cancellations = if volume_ahead < self.volume_ahead {
                ((self.volume_ahead - volume_ahead) as f64) / elapsed
            } else {
                0.0
            };
            
            self.arrival_rate = (1.0 - alpha) * self.arrival_rate + alpha * order_arrival;
            self.cancellation_rate = (1.0 - alpha) * self.cancellation_rate + alpha * cancellations;
        }
        
        self.position_ahead = position_ahead;
        self.volume_ahead = volume_ahead;
        self.total_volume = total_volume;
        self.last_update = now;
    }

    /// Calculate expected wait time until execution (in milliseconds)
    #[inline]
    pub fn expected_wait_time_ms(&self) -> f64 {
        if self.arrival_rate <= 0.0 {
            return f64::INFINITY;
        }
        
        // Expected time = position / (arrival_rate - cancellation_rate)
        let net_rate = self.arrival_rate - self.cancellation_rate;
        if net_rate <= 0.0 {
            return f64::INFINITY;
        }
        
        (self.position_ahead as f64 / net_rate) * 1000.0
    }

    /// Calculate probability of execution within time horizon T
    #[inline]
    pub fn execution_probability(&self, horizon_ms: f64) -> f64 {
        let wait_time = self.expected_wait_time_ms();
        if wait_time.is_infinite() {
            return 0.0;
        }
        
        // Exponential distribution approximation
        1.0 - (-horizon_ms / wait_time).exp()
    }

    /// Detect potential iceberg orders (large hidden volume)
    #[inline]
    pub fn detect_iceberg(&self, typical_order_size: u64) -> bool {
        if self.total_volume == 0 {
            return false;
        }
        
        // Iceberg detection: visible volume much smaller than execution flow
        let visible_orders = self.total_volume / typical_order_size.max(1);
        let estimated_real_orders = self.position_ahead as u64;
        
        // If estimated real orders >> visible orders, likely an iceberg
        estimated_real_orders > visible_orders * 3
    }

    /// Calculate queue priority score (higher is better)
    #[inline]
    pub fn priority_score(&self) -> f64 {
        // Score based on position relative to total queue
        if self.total_volume == 0 {
            return 1.0;
        }
        
        let position_ratio = self.volume_ahead as f64 / self.total_volume as f64;
        
        // Penalize being behind large volume
        // Reward low arrival rate (stable queue)
        // Reward high cancellation rate ahead (queue shrinking)
        (1.0 - position_ratio) * (1.0 + self.cancellation_rate / (self.arrival_rate + 1.0))
    }
}

impl Default for QueueState {
    fn default() -> Self {
        Self::new()
    }
}

/// Quote adjustment recommendations based on queue analysis
#[derive(Debug, Clone, Copy)]
pub struct QuoteAdjustment {
    /// Recommended price offset from mid (in ticks)
    pub price_offset_ticks: i32,
    /// Whether to cancel and re-quote immediately
    pub should_requote: bool,
    /// Confidence in the recommendation (0.0 to 1.0)
    pub confidence: f64,
    /// Reason code for the adjustment
    pub reason: AdjustmentReason,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AdjustmentReason {
    Normal,
    IcebergDetected,
    QueuePositionPoor,
    HighAdverseSelection,
    SpreadTooWide,
    VolatilitySpike,
}

/// Queue-Aware Market Maker engine
pub struct QueueAwareMarketMaker {
    /// Bid queue state
    bid_queue: QueueState,
    /// Ask queue state
    ask_queue: QueueState,
    /// Minimum maker rebate (in basis points)
    min_maker_rebate_bps: f64,
    /// Maximum acceptable wait time (ms)
    max_wait_time_ms: f64,
    /// Typical order size for iceberg detection
    typical_order_size: u64,
    /// Current spread (in ticks)
    current_spread_ticks: u32,
    /// Pre-allocated quote adjustment buffer
    last_adjustment: QuoteAdjustment,
}

impl QueueAwareMarketMaker {
    pub fn new(min_maker_rebate_bps: f64, max_wait_time_ms: f64, typical_order_size: u64) -> Self {
        QueueAwareMarketMaker {
            bid_queue: QueueState::new(),
            ask_queue: QueueState::new(),
            min_maker_rebate_bps,
            max_wait_time_ms,
            typical_order_size,
            current_spread_ticks: 10, // Default 10 tick spread
            last_adjustment: QuoteAdjustment {
                price_offset_ticks: 0,
                should_requote: false,
                confidence: 1.0,
                reason: AdjustmentReason::Normal,
            },
        }
    }

    /// Update bid queue state from L3 data feed
    #[inline]
    pub fn update_bid_queue(&mut self, position_ahead: u32, volume_ahead: u64, total_volume: u64) {
        self.bid_queue.update(position_ahead, volume_ahead, total_volume);
    }

    /// Update ask queue state from L3 data feed
    #[inline]
    pub fn update_ask_queue(&mut self, position_ahead: u32, volume_ahead: u64, total_volume: u64) {
        self.ask_queue.update(position_ahead, volume_ahead, total_volume);
    }

    /// Get current bid queue state
    #[inline]
    pub fn bid_queue_state(&self) -> &QueueState {
        &self.bid_queue
    }

    /// Get current ask queue state
    #[inline]
    pub fn ask_queue_state(&self) -> &QueueState {
        &self.ask_queue
    }

    /// Analyze queue and generate quote adjustment recommendation
    pub fn analyze_and_adjust(&mut self, mid_price: u64, tick_size: u64) -> QuoteAdjustment {
        let mut adjustment = QuoteAdjustment {
            price_offset_ticks: 0,
            should_requote: false,
            confidence: 1.0,
            reason: AdjustmentReason::Normal,
        };

        // Check bid side
        let bid_wait = self.bid_queue.expected_wait_time_ms();
        let bid_iceberg = self.bid_queue.detect_iceberg(self.typical_order_size);
        let bid_priority = self.bid_queue.priority_score();

        // Check ask side
        let ask_wait = self.ask_queue.expected_wait_time_ms();
        let ask_iceberg = self.ask_queue.detect_iceberg(self.typical_order_size);
        let ask_priority = self.ask_queue.priority_score();

        // Decision logic for bid side
        if bid_iceberg {
            // Behind an iceberg: move price or cancel
            adjustment.price_offset_ticks -= 2; // Move closer to mid
            adjustment.should_requote = true;
            adjustment.reason = AdjustmentReason::IcebergDetected;
            adjustment.confidence = 0.9;
        } else if bid_wait > self.max_wait_time_ms && bid_wait.is_finite() {
            // Queue position too poor
            adjustment.price_offset_ticks -= 1;
            adjustment.should_requote = bid_priority < 0.3;
            adjustment.reason = AdjustmentReason::QueuePositionPoor;
            adjustment.confidence = 0.7;
        }

        // Decision logic for ask side (symmetric)
        if ask_iceberg {
            adjustment.price_offset_ticks += 2;
            adjustment.should_requote = true;
            adjustment.reason = AdjustmentReason::IcebergDetected;
            adjustment.confidence = 0.9;
        } else if ask_wait > self.max_wait_time_ms && ask_wait.is_finite() {
            adjustment.price_offset_ticks += 1;
            adjustment.should_requote |= ask_priority < 0.3;
            if adjustment.reason == AdjustmentReason::Normal {
                adjustment.reason = AdjustmentReason::QueuePositionPoor;
            }
            adjustment.confidence = adjustment.confidence.min(0.7);
        }

        // Store and return
        self.last_adjustment = adjustment;
        adjustment
    }

    /// Calculate optimal queue position to target
    /// Returns recommended volume ahead threshold
    #[inline]
    pub fn optimal_queue_target(&self, side: Side) -> u64 {
        let queue = match side {
            Side::Bid => &self.bid_queue,
            Side::Ask => &self.ask_queue,
        };

        // Target: be in front of at least 50% of queue but not behind icebergs
        let target_ratio = 0.5;
        (queue.total_volume as f64 * target_ratio) as u64
    }

    /// Estimate fill probability for a given queue position
    #[inline]
    pub fn estimate_fill_probability(&self, side: Side, time_horizon_ms: f64) -> f64 {
        let queue = match side {
            Side::Bid => &self.bid_queue,
            Side::Ask => &self.ask_queue,
        };
        
        queue.execution_probability(time_horizon_ms)
    }

    /// Set current spread for context-aware decisions
    #[inline]
    pub fn set_spread(&mut self, spread_ticks: u32) {
        self.current_spread_ticks = spread_ticks;
    }

    /// Get minimum spread to maintain given queue conditions
    #[inline]
    pub fn minimum_viable_spread(&self) -> u32 {
        // Base spread + adjustment for queue risk
        let bid_risk = if self.bid_queue.detect_iceberg(self.typical_order_size) { 2 } else { 0 };
        let ask_risk = if self.ask_queue.detect_iceberg(self.typical_order_size) { 2 } else { 0 };
        
        (self.current_spread_ticks / 2 + bid_risk + ask_risk).max(1) as u32
    }
}

/// Order side enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Bid,
    Ask,
}

impl Side {
    #[inline]
    pub fn opposite(self) -> Side {
        match self {
            Side::Bid => Side::Ask,
            Side::Ask => Side::Bid,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_queue_state_update() {
        let mut queue = QueueState::new();
        queue.update(10, 1000, 5000);
        
        assert_eq!(queue.position_ahead, 10);
        assert_eq!(queue.volume_ahead, 1000);
        assert_eq!(queue.total_volume, 5000);
    }

    #[test]
    fn test_iceberg_detection() {
        let mut queue = QueueState::new();
        // Simulate iceberg: many orders ahead but small visible volume
        queue.update(100, 100, 500); // 100 orders, only 100 visible volume
        
        let has_iceberg = queue.detect_iceberg(50); // Typical order = 50
        assert!(has_iceberg);
    }

    #[test]
    fn test_market_maker_adjustment() {
        let mut mm = QueueAwareMarketMaker::new(2.5, 5000.0, 100);
        
        // Set up poor queue conditions
        mm.update_bid_queue(500, 50000, 60000);
        mm.update_ask_queue(10, 1000, 50000);
        
        let adjustment = mm.analyze_and_adjust(50000, 1);
        
        // Should recommend adjustment due to poor bid queue
        assert!(adjustment.price_offset_ticks < 0 || adjustment.should_requote);
    }
}
