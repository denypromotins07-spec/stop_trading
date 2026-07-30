//! Disruptor Module Root
//!
//! Advanced LMAX Disruptor implementation for ultra-low latency event routing.
//! Uses lock-free ring buffer with cache-line padding to prevent false sharing
//! across AMD Ryzen CPU cores.

pub mod ring_buffer;
pub mod sequencer;

pub use ring_buffer::{RingBuffer, RingBufferConfig, EventFactory, EventProcessor};
pub use sequencer::{Sequencer, SequenceBarrier, DependencyGraph, WaitStrategy};

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

/// Cache line size for padding (typically 64 bytes on modern CPUs)
pub const CACHE_LINE_SIZE: usize = 64;

/// Disruptor configuration
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DisruptorConfig {
    /// Buffer size (must be power of 2)
    pub buffer_size: usize,
    /// Number of producers
    pub producer_count: usize,
    /// Number of consumers
    pub consumer_count: usize,
    /// Wait strategy type
    pub wait_strategy: WaitStrategy,
}

impl Default for DisruptorConfig {
    fn default() -> Self {
        Self {
            buffer_size: 1024, // Power of 2
            producer_count: 1,
            consumer_count: 4,
            wait_strategy: WaitStrategy::AdaptiveSpin,
        }
    }
}

/// Disruptor statistics
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DisruptorStats {
    /// Events published
    pub events_published: u64,
    /// Events consumed
    pub events_consumed: u64,
    /// Consumer lag (events pending)
    pub consumer_lag: u64,
    /// Backpressure events
    pub backpressure_events: u64,
    /// Spins before yield
    pub total_spins: u64,
    /// Yields to OS
    pub total_yields: u64,
}

impl DisruptorStats {
    #[inline]
    pub fn new() -> Self {
        Self {
            events_published: 0,
            events_consumed: 0,
            consumer_lag: 0,
            backpressure_events: 0,
            total_spins: 0,
            total_yields: 0,
        }
    }
}

impl Default for DisruptorStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Main Disruptor structure coordinating ring buffer and sequencer
#[repr(C)]
pub struct Disruptor<E: EventFactory> {
    /// Ring buffer for events
    ring_buffer: Arc<RingBuffer<E>>,
    /// Sequencer for coordination
    sequencer: Arc<Sequencer>,
    /// Disruptor is running
    is_running: AtomicBool,
    /// Statistics
    stats: DisruptorStats,
    /// Shutdown flag
    shutdown_requested: AtomicBool,
}

impl<E: EventFactory> Disruptor<E> {
    /// Create a new disruptor with given configuration
    pub fn new(config: DisruptorConfig) -> Self 
    where
        E: Default,
    {
        let ring_buffer = Arc::new(RingBuffer::new(config.buffer_size));
        let sequencer = Arc::new(Sequencer::new(config.wait_strategy));

        Self {
            ring_buffer,
            sequencer,
            is_running: AtomicBool::new(false),
            stats: DisruptorStats::new(),
            shutdown_requested: AtomicBool::new(false),
        }
    }

    /// Start the disruptor
    #[inline]
    pub fn start(&self) {
        self.is_running.store(true, Ordering::Release);
        self.shutdown_requested.store(false, Ordering::Release);
        self.sequencer.start();
    }

    /// Stop the disruptor gracefully
    #[inline]
    pub fn stop(&self) {
        self.shutdown_requested.store(true, Ordering::Release);
        
        // Wait for consumers to drain
        while self.ring_buffer.has_available_events(self.sequencer.cursor()) {
            std::hint::spin_loop();
        }
        
        self.is_running.store(false, Ordering::Release);
        self.sequencer.stop();
    }

    /// Check if running
    #[inline]
    pub fn is_running(&self) -> bool {
        self.is_running.load(Ordering::Acquire)
    }

    /// Check if shutdown requested
    #[inline]
    pub fn is_shutdown_requested(&self) -> bool {
        self.shutdown_requested.load(Ordering::Acquire)
    }

    /// Publish an event using the ring buffer
    #[inline]
    pub fn publish<F>(&self, event_factory: F) -> Result<u64, ()>
    where
        F: FnOnce(&mut E),
    {
        if !self.is_running.load(Ordering::Acquire) {
            return Err(());
        }

        // Claim next sequence
        let seq = self.sequencer.next(1)?;

        // Get event slot and populate
        {
            let event = self.ring_buffer.get_mut(seq);
            event_factory(event);
        }

        // Publish the sequence
        self.sequencer.publish(seq);

        self.stats.events_published += 1;
        Ok(seq)
    }

    /// Try to publish without blocking
    #[inline]
    pub fn try_publish<F>(&self, event_factory: F) -> Result<u64, ()>
    where
        F: FnOnce(&mut E),
    {
        if !self.is_running.load(Ordering::Acquire) {
            return Err(());
        }

        match self.sequencer.try_next(1) {
            Ok(seq) => {
                {
                    let event = self.ring_buffer.get_mut(seq);
                    event_factory(event);
                }
                self.sequencer.publish(seq);
                self.stats.events_published += 1;
                Ok(seq)
            }
            Err(_) => {
                self.stats.backpressure_events += 1;
                Err(())
            }
        }
    }

    /// Create a barrier for a consumer
    #[inline]
    pub fn create_barrier(&self) -> SequenceBarrier {
        self.sequencer.create_barrier()
    }

    /// Get ring buffer reference
    #[inline]
    pub fn ring_buffer(&self) -> &Arc<RingBuffer<E>> {
        &self.ring_buffer
    }

    /// Get sequencer reference
    #[inline]
    pub fn sequencer(&self) -> &Arc<Sequencer> {
        &self.sequencer
    }

    /// Get current cursor position
    #[inline]
    pub fn cursor(&self) -> u64 {
        self.sequencer.cursor()
    }

    /// Get available sequence for consumer
    #[inline]
    pub fn get_available_sequence(&self, dependent_seq: u64) -> u64 {
        self.sequencer.get_available_sequence(dependent_seq)
    }

    /// Calculate consumer lag
    #[inline]
    pub fn calculate_lag(&self, consumer_seq: u64) -> u64 {
        let cursor = self.sequencer.cursor();
        if cursor >= consumer_seq {
            cursor - consumer_seq
        } else {
            0
        }
    }

    /// Get disruptor statistics
    #[inline]
    pub fn get_stats(&self) -> DisruptorStats {
        DisruptorStats {
            events_published: self.stats.events_published,
            events_consumed: self.stats.events_consumed,
            consumer_lag: self.calculate_lag(0),
            backpressure_events: self.stats.backpressure_events,
            total_spins: self.stats.total_spins,
            total_yields: self.stats.total_yields,
        }
    }

    /// Record consumption by a consumer
    #[inline]
    pub fn record_consumption(&self, count: u64) {
        self.stats.events_consumed += count;
    }

    /// Record wait strategy metrics
    #[inline]
    pub fn record_wait_metrics(&self, spins: u64, yields: u64) {
        self.stats.total_spins += spins;
        self.stats.total_yields += yields;
    }
}

/// Market data event for disruptor pipeline
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct MarketDataEvent {
    /// Symbol hash
    pub symbol_hash: u64,
    /// Bid price
    pub bid_price: u64,
    /// Ask price
    pub ask_price: u64,
    /// Bid size
    pub bid_size: u64,
    /// Ask size
    pub ask_size: u64,
    /// Timestamp in nanoseconds
    pub timestamp_ns: u64,
    /// Venue ID
    pub venue_id: u32,
    /// Sequence number
    pub sequence: u64,
    /// Event flags
    pub flags: u32,
}

impl EventFactory for MarketDataEvent {}

/// Order event for disruptor pipeline
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct OrderEvent {
    /// Client order ID
    pub client_order_id: u64,
    /// Symbol hash
    pub symbol_hash: u64,
    /// Side: 0 = Buy, 1 = Sell
    pub side: u8,
    /// Order type
    pub order_type: u8,
    /// Price
    pub price: u64,
    /// Quantity
    pub quantity: u64,
    /// Timestamp in nanoseconds
    pub timestamp_ns: u64,
    /// Venue ID
    pub venue_id: u32,
    /// Sequence number
    pub sequence: u64,
    /// Padding for alignment
    pub _padding: [u8; 5],
}

impl EventFactory for OrderEvent {}

/// Fill event for disruptor pipeline
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct FillEvent {
    /// Order ID
    pub order_id: u64,
    /// Fill ID
    pub fill_id: u64,
    /// Fill price
    pub fill_price: u64,
    /// Fill quantity
    pub fill_qty: u64,
    /// Timestamp in nanoseconds
    pub timestamp_ns: u64,
    /// Venue ID
    pub venue_id: u32,
    /// Sequence number
    pub sequence: u64,
    /// Commission
    pub commission: u64,
}

impl EventFactory for FillEvent {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_disruptor_creation() {
        let config = DisruptorConfig::default();
        let disruptor = Disruptor::<MarketDataEvent>::new(config);

        assert!(!disruptor.is_running());
        assert!(!disruptor.is_shutdown_requested());
    }

    #[test]
    fn test_disruptor_lifecycle() {
        let config = DisruptorConfig::default();
        let disruptor = Disruptor::<MarketDataEvent>::new(config);

        disruptor.start();
        assert!(disruptor.is_running());

        disruptor.stop();
        assert!(!disruptor.is_running());
    }

    #[test]
    fn test_market_data_event() {
        let mut event = MarketDataEvent::default();
        event.symbol_hash = 12345;
        event.bid_price = 10000;
        event.ask_price = 10010;
        event.bid_size = 100;
        event.ask_size = 200;

        assert_eq!(event.symbol_hash, 12345);
        assert_eq!(event.spread(), 10);
    }
}

impl MarketDataEvent {
    #[inline]
    pub fn spread(&self) -> u64 {
        if self.ask_price > self.bid_price {
            self.ask_price - self.bid_price
        } else {
            0
        }
    }

    #[inline]
    pub fn mid_price(&self) -> u64 {
        if self.bid_price > 0 && self.ask_price > 0 {
            (self.bid_price + self.ask_price) / 2
        } else {
            0
        }
    }
}
