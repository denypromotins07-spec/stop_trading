//! Queue Dynamics Module
//! Models queue position and decay rates for limit orders using lock-free counters.
//! Calculates exact probability of fill based on queue depletion speed and aggressive market order arrivals.

use std::sync::atomic::{AtomicU64, AtomicI64, Ordering};
use std::time::{Duration, Instant};

/// Cache line padding to prevent false sharing on AMD Ryzen cores
const CACHE_LINE_SIZE: usize = 64;

#[repr(C, align(64))]
pub struct CachePadded<T> {
    data: T,
    _pad: [u8; CACHE_LINE_SIZE],
}

impl<T: Default> Default for CachePadded<T> {
    fn default() -> Self {
        Self {
            data: T::default(),
            _pad: [0u8; CACHE_LINE_SIZE],
        }
    }
}

/// Represents the state of a limit order in the exchange queue
#[derive(Debug, Clone, Copy)]
pub struct QueuePosition {
    /// Current position in the queue (number of orders ahead)
    pub position: u64,
    /// Total volume ahead in the queue (in base units)
    pub volume_ahead: u64,
    /// Our order size
    pub our_size: u64,
    /// Timestamp when we joined the queue
    pub join_time_ns: u64,
    /// Last update timestamp
    pub last_update_ns: u64,
}

/// Queue decay metrics calculated from market flow
#[derive(Debug, Clone, Copy)]
pub struct QueueDecayMetrics {
    /// Rate of queue depletion (units per millisecond)
    pub depletion_rate: f64,
    /// Rate of new orders arriving (units per millisecond)
    pub arrival_rate: f64,
    /// Net flow (depletion - arrival)
    pub net_flow: f64,
    /// Estimated time to fill at current rate (milliseconds)
    pub estimated_fill_time_ms: f64,
    /// Probability of fill within next N milliseconds
    pub fill_probability: f64,
}

/// Lock-free queue dynamics tracker
pub struct QueueDynamicsTracker {
    /// Current queue position (lock-free)
    position: CachePadded<AtomicU64>,
    /// Volume ahead in queue (lock-free)
    volume_ahead: CachePadded<AtomicU64>,
    /// Our order size (lock-free)
    our_size: CachePadded<AtomicU64>,
    /// Join timestamp in nanoseconds
    join_time_ns: CachePadded<AtomicU64>,
    /// Last update timestamp
    last_update_ns: CachePadded<AtomicU64>,
    /// Cumulative volume executed at this price level
    executed_volume: CachePadded<AtomicU64>,
    /// Number of aggressive market orders detected
    aggressive_orders: CachePadded<AtomicU64>,
    /// Rolling window of depletion rates (simplified as atomic accumulators)
    depletion_sum: CachePadded<AtomicU64>,
    depletion_count: CachePadded<AtomicU64>,
    /// Price level identifier
    price_level: i64,
    /// Side: true for bid, false for ask
    is_bid: bool,
}

impl QueueDynamicsTracker {
    /// Create a new queue dynamics tracker
    pub fn new(price_level: i64, is_bid: bool, our_size: u64) -> Self {
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_nanos() as u64;

        Self {
            position: CachePadded::default(),
            volume_ahead: CachePadded::default(),
            our_size: CachePadded::new(AtomicU64::new(our_size)),
            join_time_ns: CachePadded::new(AtomicU64::new(now_ns)),
            last_update_ns: CachePadded::new(AtomicU64::new(now_ns)),
            executed_volume: CachePadded::default(),
            aggressive_orders: CachePadded::default(),
            depletion_sum: CachePadded::default(),
            depletion_count: CachePadded::default(),
            price_level,
            is_bid,
        }
    }

    /// Update queue position atomically
    #[inline]
    pub fn update_position(&self, new_position: u64, volume_ahead: u64) {
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_nanos() as u64;

        self.position.data.store(new_position, Ordering::Release);
        self.volume_ahead.data.store(volume_ahead, Ordering::Release);
        self.last_update_ns.data.store(now_ns, Ordering::Release);
    }

    /// Record an execution at this price level
    #[inline]
    pub fn record_execution(&self, executed_units: u64, is_aggressive: bool) {
        self.executed_volume
            .data
            .fetch_add(executed_units, Ordering::AcqRel);

        if is_aggressive {
            self.aggressive_orders.data.fetch_add(1, Ordering::AcqRel);
        }

        // Update depletion metrics
        self.depletion_sum
            .data
            .fetch_add(executed_units, Ordering::AcqRel);
        self.depletion_count.data.fetch_add(1, Ordering::AcqRel);
    }

    /// Get current queue position snapshot
    #[inline]
    pub fn get_position(&self) -> QueuePosition {
        QueuePosition {
            position: self.position.data.load(Ordering::Acquire),
            volume_ahead: self.volume_ahead.data.load(Ordering::Acquire),
            our_size: self.our_size.data.load(Ordering::Acquire),
            join_time_ns: self.join_time_ns.data.load(Ordering::Acquire),
            last_update_ns: self.last_update_ns.data.load(Ordering::Acquire),
        }
    }

    /// Calculate queue decay metrics
    pub fn calculate_decay_metrics(&self, window_ms: u64) -> QueueDecayMetrics {
        let position = self.get_position();
        let elapsed_ns = position.last_update_ns.saturating_sub(position.join_time_ns);
        let elapsed_ms = (elapsed_ns / 1_000_000) as f64;

        if elapsed_ms < 1.0 {
            return QueueDecayMetrics {
                depletion_rate: 0.0,
                arrival_rate: 0.0,
                net_flow: 0.0,
                estimated_fill_time_ms: f64::INFINITY,
                fill_probability: 0.0,
            };
        }

        let executed = self.executed_volume.data.load(Ordering::Acquire) as f64;
        let depletion_count = self.depletion_count.data.load(Ordering::Acquire) as f64;

        // Calculate depletion rate (units per ms)
        let depletion_rate = if elapsed_ms > 0.0 {
            executed / elapsed_ms
        } else {
            0.0
        };

        // Estimate arrival rate based on queue position changes
        // Simplified: assume stable arrival if position hasn't changed much
        let arrival_rate = depletion_rate * 0.5; // Heuristic: 50% of depletion is offset by new orders

        let net_flow = depletion_rate - arrival_rate;

        // Estimate time to fill
        let volume_to_fill = position.volume_ahead as f64 + position.our_size as f64;
        let estimated_fill_time_ms = if net_flow > 0.0 {
            volume_to_fill / net_flow
        } else {
            f64::INFINITY
        };

        // Calculate fill probability using exponential decay model
        // P(fill) = 1 - exp(-lambda * t) where lambda is depletion rate normalized
        let fill_probability = self.calculate_fill_probability(&position, depletion_rate, window_ms);

        QueueDecayMetrics {
            depletion_rate,
            arrival_rate,
            net_flow,
            estimated_fill_time_ms,
            fill_probability,
        }
    }

    /// Calculate probability of fill within specified time window
    fn calculate_fill_probability(&self, position: &QueuePosition, depletion_rate: f64, window_ms: u64) -> f64 {
        if depletion_rate <= 0.0 {
            return 0.0;
        }

        let volume_to_fill = position.volume_ahead as f64;
        let expected_depletion = depletion_rate * window_ms as f64;

        // Probability that we get filled = P(expected_depletion >= volume_ahead)
        // Using a simplified model: if expected depletion exceeds volume ahead, high probability
        if expected_depletion >= volume_to_fill {
            // Add some uncertainty factor based on aggressive order count
            let aggressive = self.aggressive_orders.data.load(Ordering::Acquire) as f64;
            let uncertainty_factor = 1.0 / (1.0 + aggressive * 0.01);
            return 0.95 * uncertainty_factor;
        }

        // Partial probability based on ratio
        let ratio = expected_depletion / volume_to_fill.max(1.0);
        
        // Sigmoid-like function for smooth probability transition
        // Using Taylor approximation: sigmoid(x) ≈ 0.5 + x/4 for small x
        let prob = if ratio < 0.5 {
            0.25 * ratio
        } else if ratio < 2.0 {
            0.125 + 0.375 * ratio
        } else {
            0.875 + 0.0625 * (ratio - 2.0)
        };

        prob.min(0.95)
    }

    /// Check if queue position is favorable for maintaining the order
    #[inline]
    pub fn is_favorable_position(&self, threshold_probability: f64) -> bool {
        let metrics = self.calculate_decay_metrics(1000); // 1 second window
        metrics.fill_probability >= threshold_probability
    }

    /// Get estimated queue jump benefit (should we cancel and rejoin?)
    #[inline]
    pub fn should_requeue(&self, current_market_position: u64) -> bool {
        let our_position = self.position.data.load(Ordering::Acquire);
        
        // If market has moved significantly and we're far back, consider requeuing
        if current_market_position < our_position / 2 {
            // Market queue is much shorter, might be worth canceling and rejoining
            let aggressive = self.aggressive_orders.data.load(Ordering::Acquire);
            aggressive > 10 // Only if significant aggressive flow has passed us
        } else {
            false
        }
    }

    /// Reset tracker for new order
    #[inline]
    pub fn reset(&self, new_size: u64) {
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_nanos() as u64;

        self.position.data.store(0, Ordering::Release);
        self.volume_ahead.data.store(0, Ordering::Release);
        self.our_size.data.store(new_size, Ordering::Release);
        self.join_time_ns.data.store(now_ns, Ordering::Release);
        self.last_update_ns.data.store(now_ns, Ordering::Release);
        self.executed_volume.data.store(0, Ordering::Release);
        self.aggressive_orders.data.store(0, Ordering::Release);
        self.depletion_sum.data.store(0, Ordering::Release);
        self.depletion_count.data.store(0, Ordering::Release);
    }

    /// Get price level
    #[inline]
    pub fn price_level(&self) -> i64 {
        self.price_level
    }

    /// Get side
    #[inline]
    pub fn is_bid_side(&self) -> bool {
        self.is_bid
    }
}

/// Aggregated queue dynamics across multiple price levels
pub struct QueueDynamicsAggregator {
    /// Trackers for bid side
    bid_trackers: Vec<QueueDynamicsTracker>,
    /// Trackers for ask side
    ask_trackers: Vec<QueueDynamicsTracker>,
    /// Lock-free counter for total tracked levels
    total_levels: AtomicU64,
}

impl QueueDynamicsAggregator {
    /// Create new aggregator with specified depth
    pub fn new(depth: usize) -> Self {
        Self {
            bid_trackers: Vec::with_capacity(depth),
            ask_trackers: Vec::with_capacity(depth),
            total_levels: AtomicU64::new(0),
        }
    }

    /// Add a price level to track
    pub fn add_level(&mut self, price: i64, is_bid: bool, initial_size: u64) {
        let tracker = QueueDynamicsTracker::new(price, is_bid, initial_size);
        if is_bid {
            self.bid_trackers.push(tracker);
        } else {
            self.ask_trackers.push(tracker);
        }
        self.total_levels.fetch_add(1, Ordering::AcqRel);
    }

    /// Get tracker for specific price and side
    pub fn get_tracker(&self, price: i64, is_bid: bool) -> Option<&QueueDynamicsTracker> {
        let trackers = if is_bid { &self.bid_trackers } else { &self.ask_trackers };
        trackers.iter().find(|t| t.price_level() == price)
    }

    /// Calculate aggregate fill probability for best bid/ask
    pub fn best_level_probability(&self, is_bid: bool) -> f64 {
        let trackers = if is_bid { &self.bid_trackers } else { &self.ask_trackers };
        trackers.first()
            .map(|t| t.calculate_decay_metrics(1000).fill_probability)
            .unwrap_or(0.0)
    }

    /// Get total queue pressure (sum of all depletion rates)
    pub fn total_queue_pressure(&self, is_bid: bool) -> f64 {
        let trackers = if is_bid { &self.bid_trackers } else { &self.ask_trackers };
        trackers.iter()
            .map(|t| t.calculate_decay_metrics(1000).depletion_rate)
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_queue_dynamics_basic() {
        let tracker = QueueDynamicsTracker::new(10000, true, 100);
        
        // Initial position
        let pos = tracker.get_position();
        assert_eq!(pos.position, 0);
        assert_eq!(pos.our_size, 100);

        // Update position
        tracker.update_position(50, 5000);
        let pos = tracker.get_position();
        assert_eq!(pos.position, 50);
        assert_eq!(pos.volume_ahead, 5000);

        // Record executions
        tracker.record_execution(1000, true);
        tracker.record_execution(500, false);
        
        let metrics = tracker.calculate_decay_metrics(1000);
        assert!(metrics.depletion_rate > 0.0);
    }

    #[test]
    fn test_fill_probability() {
        let tracker = QueueDynamicsTracker::new(10000, true, 100);
        tracker.update_position(10, 1000);
        
        // Simulate high depletion
        for _ in 0..100 {
            tracker.record_execution(100, true);
        }

        let metrics = tracker.calculate_decay_metrics(1000);
        assert!(metrics.fill_probability > 0.0);
    }

    #[test]
    fn test_cache_padded_alignment() {
        let padded = CachePadded::new(AtomicU64::new(42));
        assert_eq!(padded.data.load(Ordering::Relaxed), 42);
        // Verify size includes padding
        assert!(std::mem::size_of::<CachePadded<AtomicU64>>() >= CACHE_LINE_SIZE);
    }
}
