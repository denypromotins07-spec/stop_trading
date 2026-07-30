//! Microstructure Module Root
//! Feed queue probabilities into the execution router.

pub mod queue_dynamics;
pub mod hidden_liquidity;

pub use queue_dynamics::{
    QueueDynamicsTracker,
    QueueDynamicsAggregator,
    QueuePosition,
    QueueDecayMetrics,
};

pub use hidden_liquidity::{
    HiddenLiquidityDetector,
    DarkPoolCorrelator,
    IcebergOrder,
    TradeTick,
    LiquidityAnomaly,
    AnomalyType,
    AnomalyMetadata,
    HiddenLiquidityStats,
    DarkPoolSignal,
};

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};

/// Cache line padding constant
const CACHE_LINE_SIZE: usize = 64;

#[repr(C, align(64))]
struct CachePadded<T> {
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

/// Aggregated microstructure signals for the execution router
#[derive(Debug, Clone, Copy)]
pub struct MicrostructureSignals {
    /// Bid side fill probability (0.0 to 1.0)
    pub bid_fill_probability: f64,
    /// Ask side fill probability (0.0 to 1.0)
    pub ask_fill_probability: f64,
    /// Bid side queue pressure (units/ms)
    pub bid_queue_pressure: f64,
    /// Ask side queue pressure (units/ms)
    pub ask_queue_pressure: f64,
    /// Iceberg detected on bid
    pub bid_iceberg_detected: bool,
    /// Iceberg detected on ask
    pub ask_iceberg_detected: bool,
    /// Hidden liquidity score (0.0 to 1.0)
    pub hidden_liquidity_score: f64,
    /// Dark pool activity detected
    pub dark_pool_activity: bool,
    /// Timestamp of signal generation (nanoseconds)
    pub timestamp_ns: u64,
}

impl MicrostructureSignals {
    /// Create empty signals
    pub fn empty() -> Self {
        Self {
            bid_fill_probability: 0.0,
            ask_fill_probability: 0.0,
            bid_queue_pressure: 0.0,
            ask_queue_pressure: 0.0,
            bid_iceberg_detected: false,
            ask_iceberg_detected: false,
            hidden_liquidity_score: 0.0,
            dark_pool_activity: false,
            timestamp_ns: 0,
        }
    }

    /// Check if conditions are favorable for aggressive orders
    #[inline]
    pub fn is_aggressive_favorable(&self, is_buy: bool) -> bool {
        if is_buy {
            // Favorable to buy aggressively if ask queue pressure is high (depleting fast)
            self.ask_queue_pressure > 100.0 && !self.ask_iceberg_detected
        } else {
            // Favorable to sell aggressively if bid queue pressure is high
            self.bid_queue_pressure > 100.0 && !self.bid_iceberg_detected
        }
    }

    /// Check if passive quoting is advisable
    #[inline]
    pub fn is_passive_favorable(&self, is_buy: bool) -> bool {
        if is_buy {
            // Good to provide liquidity on bid if fill probability is high
            self.bid_fill_probability > 0.5
        } else {
            self.ask_fill_probability > 0.5
        }
    }
}

/// Main microstructure engine combining all signals
pub struct MicrostructureEngine {
    /// Queue dynamics aggregator
    queue_aggregator: QueueDynamicsAggregator,
    /// Hidden liquidity detector
    hidden_detector: HiddenLiquidityDetector,
    /// Dark pool correlator
    dark_pool_correlator: DarkPoolCorrelator,
    /// Engine active flag
    is_active: CachePadded<AtomicBool>,
    /// Signal counter
    signal_count: CachePadded<AtomicU64>,
}

impl MicrostructureEngine {
    /// Create new microstructure engine
    pub fn new(depth: usize, buffer_size: usize) -> Self {
        Self {
            queue_aggregator: QueueDynamicsAggregator::new(depth),
            hidden_detector: HiddenLiquidityDetector::new(buffer_size, 0.5),
            dark_pool_correlator: DarkPoolCorrelator::new(100),
            is_active: CachePadded::new(AtomicBool::new(true)),
            signal_count: CachePadded::default(),
        }
    }

    /// Add price level to track
    pub fn add_price_level(&mut self, price: i64, is_bid: bool, initial_size: u64) {
        self.queue_aggregator.add_level(price, is_bid, initial_size);
    }

    /// Record a trade for analysis
    pub fn record_trade(&self, trade: TradeTick) -> Option<LiquidityAnomaly> {
        if !self.is_active.data.load(Ordering::Acquire) {
            return None;
        }
        self.hidden_detector.record_trade(trade)
    }

    /// Update queue position
    pub fn update_queue_position(&self, price: i64, is_bid: bool, position: u64, volume_ahead: u64) {
        if let Some(tracker) = self.queue_aggregator.get_tracker(price, is_bid) {
            tracker.update_position(position, volume_ahead);
        }
    }

    /// Record execution at price level
    pub fn record_execution(&self, price: i64, is_bid: bool, size: u64, is_aggressive: bool) {
        if let Some(tracker) = self.queue_aggregator.get_tracker(price, is_bid) {
            tracker.record_execution(size, is_aggressive);
        }
    }

    /// Generate current microstructure signals
    pub fn generate_signals(&self) -> MicrostructureSignals {
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or(std::time::Duration::ZERO)
            .as_nanos() as u64;

        self.signal_count.data.fetch_add(1, Ordering::AcqRel);

        let bid_fill_prob = self.queue_aggregator.best_level_probability(true);
        let ask_fill_prob = self.queue_aggregator.best_level_probability(false);
        let bid_pressure = self.queue_aggregator.total_queue_pressure(true);
        let ask_pressure = self.queue_aggregator.total_queue_pressure(false);

        let stats = self.hidden_detector.get_stats();
        let hidden_score = (stats.iceberg_count.min(10) as f64 / 10.0) 
            * (stats.mismatch_events.min(10) as f64 / 10.0);

        MicrostructureSignals {
            bid_fill_probability: bid_fill_prob,
            ask_fill_probability: ask_fill_prob,
            bid_queue_pressure: bid_pressure,
            ask_queue_pressure: ask_pressure,
            bid_iceberg_detected: stats.iceberg_count > 0,
            ask_iceberg_detected: stats.iceberg_count > 0,
            hidden_liquidity_score: hidden_score,
            dark_pool_activity: self.dark_pool_correlator.correlation_count() > 0,
            timestamp_ns: now_ns,
        }
    }

    /// Analyze book delta for hidden liquidity
    pub fn analyze_book_delta(
        &self,
        price: i64,
        is_bid: bool,
        expected: u64,
        actual: u64,
        timestamp_ns: u64,
    ) -> Option<LiquidityAnomaly> {
        self.hidden_detector.analyze_book_delta(price, is_bid, expected, actual, timestamp_ns)
    }

    /// Detect iceberg patterns
    pub fn detect_iceberg(
        &self,
        price: i64,
        is_bid: bool,
        trades: &[TradeTick],
        visible_size: u64,
    ) -> Option<IcebergOrder> {
        self.hidden_detector.detect_iceberg_pattern(price, is_bid, trades, visible_size)
    }

    /// Enable/disable engine
    #[inline]
    pub fn set_active(&self, active: bool) {
        self.is_active.data.store(active, Ordering::Release);
        self.hidden_detector.set_active(active);
    }

    /// Check if engine is active
    #[inline]
    pub fn is_active(&self) -> bool {
        self.is_active.data.load(Ordering::Acquire)
    }

    /// Get signal count
    #[inline]
    pub fn signal_count(&self) -> u64 {
        self.signal_count.data.load(Ordering::Acquire)
    }

    /// Reset all counters
    pub fn reset(&self) {
        self.signal_count.data.store(0, Ordering::Release);
        self.hidden_detector.reset_counters();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_microstructure_engine_basic() {
        let engine = MicrostructureEngine::new(10, 1024);
        
        // Add price levels
        engine.add_price_level(10000, true, 1000);
        engine.add_price_level(10100, false, 1000);

        // Update queue positions
        engine.update_queue_position(10000, true, 5, 500);
        engine.update_queue_position(10100, false, 3, 300);

        // Generate signals
        let signals = engine.generate_signals();
        assert!(signals.timestamp_ns > 0);
    }

    #[test]
    fn test_signal_favorability() {
        let mut signals = MicrostructureSignals::empty();
        signals.ask_queue_pressure = 200.0;
        
        assert!(signals.is_aggressive_favorable(true));
        assert!(!signals.is_aggressive_favorable(false));
    }

    #[test]
    fn test_passive_favorability() {
        let mut signals = MicrostructureSignals::empty();
        signals.bid_fill_probability = 0.7;
        
        assert!(signals.is_passive_favorable(true));
        assert!(!signals.is_passive_favorable(false));
    }
}
