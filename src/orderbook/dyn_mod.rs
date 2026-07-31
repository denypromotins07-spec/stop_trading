//! Order Book Dynamics Module Root
//! 
//! Aggregates order book shape metrics and feeds them to the alpha ensemble.

pub mod dynamics;
pub mod queue_imbalance;

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use self::dynamics::{OrderBookDynamics, ShapeMetrics};
use self::queue_imbalance::{QueueImbalanceTracker, QueueImbalanceMetrics};

/// Combined order book signal for alpha ensemble
#[derive(Debug, Clone, Copy)]
pub struct OrderBookSignal {
    /// Microprice drift from shape analysis (-1.0 to 1.0)
    pub microprice_drift: f64,
    /// Queue imbalance score (-1.0 to 1.0)
    pub queue_imbalance: f64,
    /// Probability of upward spread crossing
    pub cross_up_prob: f64,
    /// Probability of downward spread crossing
    pub cross_down_prob: f64,
    /// Convexity signal (positive = bullish)
    pub convexity_signal: f64,
    /// Combined confidence score (0.0 to 1.0)
    pub confidence: f64,
    /// Timestamp in nanoseconds
    pub timestamp_ns: u64,
}

impl Default for OrderBookSignal {
    fn default() -> Self {
        Self {
            microprice_drift: 0.0,
            queue_imbalance: 0.0,
            cross_up_prob: 0.5,
            cross_down_prob: 0.5,
            convexity_signal: 0.0,
            confidence: 0.0,
            timestamp_ns: 0,
        }
    }
}

/// Cache-line aligned aggregated signal buffer
#[repr(align(64))]
pub struct SignalBuffer {
    signal: OrderBookSignal,
    valid: AtomicBool,
    sequence: AtomicU64,
}

impl SignalBuffer {
    pub const fn new() -> Self {
        Self {
            signal: OrderBookSignal {
                microprice_drift: 0.0,
                queue_imbalance: 0.0,
                cross_up_prob: 0.5,
                cross_down_prob: 0.5,
                convexity_signal: 0.0,
                confidence: 0.0,
                timestamp_ns: 0,
            },
            valid: AtomicBool::new(false),
            sequence: AtomicU64::new(0),
        }
    }

    #[inline]
    pub fn update(&self, signal: OrderBookSignal) {
        // Safety: This is a single writer, multiple reader scenario
        unsafe {
            let ptr = &self.signal as *const OrderBookSignal as *mut OrderBookSignal;
            ptr.write(signal);
        }
        self.valid.store(true, Ordering::Release);
        self.sequence.fetch_add(1, Ordering::AcqRel);
    }

    #[inline]
    pub fn get(&self) -> Option<OrderBookSignal> {
        if self.valid.load(Ordering::Acquire) {
            Some(self.signal)
        } else {
            None
        }
    }

    #[inline]
    pub fn sequence(&self) -> u64 {
        self.sequence.load(Ordering::Acquire)
    }
}

/// Main order book dynamics aggregator
#[repr(align(64))]
pub struct OrderBookAggregator {
    dynamics: OrderBookDynamics,
    queue_tracker: QueueImbalanceTracker,
    signal_buffer: Arc<SignalBuffer>,
    tick_size: u64,
    enabled: AtomicBool,
}

impl OrderBookAggregator {
    /// Create a new aggregator with specified tick size
    pub fn new(tick_size: u64) -> Self {
        Self {
            dynamics: OrderBookDynamics::new(tick_size),
            queue_tracker: QueueImbalanceTracker::new(tick_size),
            signal_buffer: Arc::new(SignalBuffer::new()),
            tick_size,
            enabled: AtomicBool::new(true),
        }
    }

    /// Update order book state at a specific level
    #[inline]
    pub fn update_level(
        &self,
        level: usize,
        bid_price: u64,
        bid_volume: u64,
        ask_price: u64,
        ask_volume: u64,
        bid_orders: u64,
        ask_orders: u64,
        timestamp_ns: u64,
    ) {
        if !self.enabled.load(Ordering::Relaxed) {
            return;
        }

        // Update dynamics tracker
        self.dynamics.update_bid(level, bid_price, bid_volume);
        self.dynamics.update_ask(level, ask_price, ask_volume);

        // Update queue imbalance tracker
        self.queue_tracker.update(
            level,
            bid_volume,
            ask_volume,
            bid_orders,
            ask_orders,
            timestamp_ns,
        );
    }

    /// Compute and emit combined signal
    pub fn compute_signal(&self, current_timestamp_ns: u64) -> OrderBookSignal {
        if !self.enabled.load(Ordering::Relaxed) {
            return OrderBookSignal::default();
        }

        // Get shape metrics
        let shape_metrics = self.dynamics.compute_metrics();

        // Get queue metrics
        let queue_metrics = self.queue_tracker.compute_metrics(current_timestamp_ns);

        // Combine signals
        let microprice_drift = shape_metrics.microprice_drift;
        let queue_imbalance = queue_metrics.imbalance_score;

        // Weighted combination for convexity signal
        let convexity_signal = (shape_metrics.convexity_bid - shape_metrics.convexity_ask) * 0.5;

        // Calculate confidence based on signal agreement
        let signal_agreement = if (microprice_drift > 0.0 && queue_imbalance > 0.0)
            || (microprice_drift < 0.0 && queue_imbalance < 0.0)
        {
            1.0
        } else {
            0.0
        };

        // Confidence increases with signal magnitude and agreement
        let avg_magnitude = (microprice_drift.abs() + queue_imbalance.abs()) / 2.0;
        let confidence = (avg_magnitude * 0.6 + signal_agreement * 0.4).min(1.0);

        let signal = OrderBookSignal {
            microprice_drift,
            queue_imbalance,
            cross_up_prob: queue_metrics.cross_up_probability,
            cross_down_prob: queue_metrics.cross_down_probability,
            convexity_signal,
            confidence,
            timestamp_ns: current_timestamp_ns,
        };

        // Update signal buffer for consumers
        self.signal_buffer.update(signal);

        signal
    }

    /// Get direct access to dynamics analyzer
    #[inline]
    pub fn dynamics(&self) -> &OrderBookDynamics {
        &self.dynamics
    }

    /// Get direct access to queue tracker
    #[inline]
    pub fn queue_tracker(&self) -> &QueueImbalanceTracker {
        &self.queue_tracker
    }

    /// Get signal buffer for lock-free reading
    #[inline]
    pub fn signal_buffer(&self) -> Arc<SignalBuffer> {
        Arc::clone(&self.signal_buffer)
    }

    /// Enable/disable signal computation
    #[inline]
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Relaxed);
    }

    /// Check if aggregator is enabled
    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }

    /// Reset all internal state
    pub fn reset(&self) {
        self.dynamics.reset();
        self.queue_tracker.reset();
    }

    /// Get best bid price
    #[inline]
    pub fn best_bid(&self) -> u64 {
        self.dynamics.best_bid()
    }

    /// Get best ask price
    #[inline]
    pub fn best_ask(&self) -> u64 {
        self.dynamics.best_ask()
    }

    /// Get mid price
    #[inline]
    pub fn mid_price(&self) -> f64 {
        self.dynamics.mid_price()
    }

    /// Get spread in ticks
    #[inline]
    pub fn spread_ticks(&self) -> u64 {
        let bid = self.best_bid();
        let ask = self.best_ask();
        if bid > 0 && ask > 0 && self.tick_size > 0 {
            (ask - bid) / self.tick_size
        } else {
            0
        }
    }
}

/// Alpha ensemble integration helper
pub struct AlphaEnsembleFeed {
    aggregator: Arc<OrderBookAggregator>,
    last_sequence: AtomicU64,
}

impl AlphaEnsembleFeed {
    pub fn new(aggregator: Arc<OrderBookAggregator>) -> Self {
        Self {
            aggregator,
            last_sequence: AtomicU64::new(0),
        }
    }

    /// Poll for new signals (non-blocking)
    pub fn poll_signal(&self) -> Option<OrderBookSignal> {
        let buffer = self.aggregator.signal_buffer();
        let current_seq = buffer.sequence();
        let last_seq = self.last_sequence.load(Ordering::Relaxed);

        if current_seq > last_seq {
            self.last_sequence.store(current_seq, Ordering::Relaxed);
            buffer.get()
        } else {
            None
        }
    }

    /// Get latest signal regardless of sequence
    pub fn latest_signal(&self) -> Option<OrderBookSignal> {
        self.aggregator.signal_buffer().get()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aggregator_signal_generation() {
        let aggregator = OrderBookAggregator::new(100);

        // Set up bullish order book
        aggregator.update_level(0, 9900, 5000, 10000, 1000, 50, 10, 1000000);
        aggregator.update_level(1, 9800, 4000, 10100, 800, 40, 8, 1000000);
        aggregator.update_level(2, 9700, 3000, 10200, 600, 30, 6, 1000000);

        let signal = aggregator.compute_signal(2000000);

        assert!(signal.microprice_drift > 0.0, "Should have positive drift");
        assert!(signal.queue_imbalance > 0.0, "Should have buy imbalance");
        assert!(signal.confidence > 0.0, "Should have some confidence");
    }

    #[test]
    fn test_signal_buffer_lock_free() {
        let buffer = Arc::new(SignalBuffer::new());
        
        let signal = OrderBookSignal {
            microprice_drift: 0.5,
            queue_imbalance: 0.3,
            cross_up_prob: 0.7,
            cross_down_prob: 0.3,
            convexity_signal: 0.2,
            confidence: 0.8,
            timestamp_ns: 12345,
        };

        buffer.update(signal);
        
        let retrieved = buffer.get().unwrap();
        assert_eq!(retrieved.microprice_drift, 0.5);
        assert_eq!(retrieved.timestamp_ns, 12345);
    }

    #[test]
    fn test_alpha_feed_polling() {
        let aggregator = Arc::new(OrderBookAggregator::new(50));
        let feed = AlphaEnsembleFeed::new(Arc::clone(&aggregator));

        // Initial poll should return nothing
        assert!(feed.poll_signal().is_none());

        // Generate a signal
        aggregator.update_level(0, 1000, 1000, 1050, 500, 10, 5, 1000000);
        let _ = aggregator.compute_signal(2000000);

        // Now should get signal
        let signal = feed.poll_signal();
        assert!(signal.is_some());
    }
}
