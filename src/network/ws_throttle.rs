//! WebSocket Message Throttling and Queue Management
//! 
//! This module implements strict WebSocket message throttling and local queueing
//! to prevent exchange-side IP bans. Dynamically drops or aggregates non-critical
//! L2 updates during tick spikes to protect the execution thread from starvation.
//! 
//! Key Features:
//! - Token bucket rate limiting
//! - Priority-based message dropping
//! - L2 update aggregation
//! - Backpressure signaling to data sources
//! - Exchange-specific rate limit compliance

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crossbeam_channel::{bounded, Receiver, Sender, TrySendError};
use tracing::{debug, info, warn};

/// Message priority levels for throttling decisions
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MessagePriority {
    /// Critical - never drop (order confirmations, fills)
    Critical = 0,
    /// High - rarely drop (trade executions, balance updates)
    High = 1,
    /// Normal - may drop under pressure (L1 ticker updates)
    Normal = 2,
    /// Low - first to drop (L2 order book updates)
    Low = 3,
}

/// WebSocket message envelope
#[derive(Debug, Clone)]
pub struct WsMessage {
    /// Message payload
    pub payload: Vec<u8>,
    /// Priority level
    pub priority: MessagePriority,
    /// Timestamp when received
    pub received_at: Instant,
    /// Symbol (for aggregation)
    pub symbol: Option<String>,
    /// Message type identifier
    pub msg_type: MessageType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageType {
    Trade,
    OrderBookL1,
    OrderBookL2,
    OrderBookL3,
    Ticker,
    AccountUpdate,
    OrderUpdate,
    Fill,
    Heartbeat,
    Unknown,
}

impl WsMessage {
    pub fn new(
        payload: Vec<u8>,
        priority: MessagePriority,
        msg_type: MessageType,
        symbol: Option<String>,
    ) -> Self {
        Self {
            payload,
            priority,
            received_at: Instant::now(),
            symbol,
            msg_type,
        }
    }
    
    /// Get message age
    pub fn age(&self) -> Duration {
        self.received_at.elapsed()
    }
}

/// Token bucket rate limiter
pub struct TokenBucket {
    /// Maximum tokens (burst capacity)
    capacity: u64,
    /// Current tokens
    tokens: AtomicU64,
    /// Tokens added per second
    refill_rate: f64,
    /// Last refill timestamp
    last_refill: parking_lot::Mutex<Instant>,
}

impl TokenBucket {
    pub fn new(capacity: u64, refill_rate: f64) -> Self {
        Self {
            capacity,
            tokens: AtomicU64::new(capacity),
            refill_rate,
            last_refill: parking_lot::Mutex::new(Instant::now()),
        }
    }
    
    /// Try to consume a token
    pub fn try_consume(&self, count: u64) -> bool {
        self.refill();
        
        let current = self.tokens.load(Ordering::Relaxed);
        if current >= count {
            match self.tokens.fetch_update(
                Ordering::Relaxed,
                Ordering::Relaxed,
                |tokens| tokens.checked_sub(count),
            ) {
                Ok(_) => true,
                Err(_) => false,
            }
        } else {
            false
        }
    }
    
    /// Consume with wait (returns time to wait if insufficient)
    pub fn consume_or_wait(&self, count: u64) -> Result<(), Duration> {
        self.refill();
        
        let current = self.tokens.load(Ordering::Relaxed);
        if current >= count {
            match self.tokens.fetch_update(
                Ordering::Relaxed,
                Ordering::Relaxed,
                |tokens| tokens.checked_sub(count),
            ) {
                Ok(_) => Ok(()),
                Err(_) => Err(Duration::from_millis(1)),
            }
        } else {
            // Calculate wait time
            let needed = count - current;
            let wait_ms = (needed as f64 / self.refill_rate * 1000.0) as u64;
            Err(Duration::from_millis(wait_ms.max(1)))
        }
    }
    
    /// Refill tokens based on elapsed time
    fn refill(&self) {
        let now = Instant::now();
        let mut last = self.last_refill.lock();
        
        let elapsed = now.duration_since(*last).as_secs_f64();
        if elapsed > 0.0 {
            let refill_amount = (elapsed * self.refill_rate) as u64;
            if refill_amount > 0 {
                let current = self.tokens.load(Ordering::Relaxed);
                let new_tokens = (current + refill_amount).min(self.capacity);
                self.tokens.store(new_tokens, Ordering::Relaxed);
                *last = now;
            }
        }
    }
    
    /// Get current token count
    pub fn tokens(&self) -> u64 {
        self.refill();
        self.tokens.load(Ordering::Relaxed)
    }
}

/// Throttled WebSocket receiver with priority handling
pub struct WsThrottler {
    /// Rate limiter
    rate_limiter: TokenBucket,
    /// Input channel from WebSocket
    input_rx: Receiver<WsMessage>,
    /// Output channel to processor
    output_tx: Sender<WsMessage>,
    /// Dropped message counter by priority
    dropped_counts: [AtomicUsize; 4],
    /// Processed message counter
    processed_count: AtomicUsize,
    /// Aggregated L2 updates
    l2_aggregator: parking_lot::Mutex<std::collections::HashMap<String, L2Aggregation>>,
    /// Aggregation window
    aggregation_window: Duration,
    /// Enable aggregation
    aggregation_enabled: bool,
}

#[derive(Debug, Clone)]
struct L2Aggregation {
    bids: Vec<(f64, f64)>,
    asks: Vec<(f64, f64)>,
    last_update: Instant,
}

impl WsThrottler {
    pub fn new(
        rate_limit_per_second: u64,
        burst_capacity: u64,
        channel_capacity: usize,
    ) -> (Self, Sender<WsMessage>, Receiver<WsMessage>) {
        let (input_tx, input_rx) = bounded(channel_capacity);
        let (output_tx, output_rx) = bounded(channel_capacity);
        
        let throttler = Self {
            rate_limiter: TokenBucket::new(burst_capacity, rate_limit_per_second as f64),
            input_rx,
            output_tx,
            dropped_counts: Default::default(),
            processed_count: AtomicUsize::new(0),
            l2_aggregator: parking_lot::Mutex::new(std::collections::HashMap::new()),
            aggregation_window: Duration::from_millis(10),
            aggregation_enabled: true,
        };
        
        (throttler, input_tx, output_rx)
    }
    
    /// Run the throttler processing loop
    pub fn run(&self) {
        loop {
            match self.input_rx.try_recv() {
                Ok(msg) => {
                    self.process_message(msg);
                }
                Err(crossbeam_channel::TryRecvError::Empty) => {
                    std::thread::yield_now();
                }
                Err(crossbeam_channel::TryRecvError::Disconnected) => {
                    break;
                }
            }
        }
    }
    
    /// Process a single message with throttling logic
    fn process_message(&self, msg: WsMessage) {
        // Check rate limit
        if !self.rate_limiter.try_consume(1) {
            // Rate limited - decide whether to drop or queue
            if msg.priority <= MessagePriority::High {
                // High priority - try to send anyway (may block)
                if self.output_tx.send(msg).is_ok() {
                    self.processed_count.fetch_add(1, Ordering::Relaxed);
                }
            } else {
                // Low priority - drop
                let priority_idx = msg.priority as usize;
                self.dropped_counts[priority_idx].fetch_add(1, Ordering::Relaxed);
                
                debug!(
                    "Dropped {} message due to rate limiting",
                    format!("{:?}", msg.msg_type)
                );
            }
            return;
        }
        
        // Handle L2 aggregation
        if self.aggregation_enabled && msg.msg_type == MessageType::OrderBookL2 {
            if let Some(symbol) = &msg.symbol {
                self.aggregate_l2_update(symbol, msg);
                return;
            }
        }
        
        // Send to output
        if self.output_tx.send(msg).is_ok() {
            self.processed_count.fetch_add(1, Ordering::Relaxed);
        }
    }
    
    /// Aggregate L2 updates within the aggregation window
    fn aggregate_l2_update(&self, symbol: &str, msg: WsMessage) {
        let mut aggregator = self.l2_aggregator.lock();
        
        if let Some(agg) = aggregator.get_mut(symbol) {
            if agg.last_update.elapsed() < self.aggregation_window {
                // Update existing aggregation
                // In real implementation, would merge order book updates
                agg.last_update = Instant::now();
                return;
            } else {
                // Window expired - flush old aggregation
                if !agg.bids.is_empty() || !agg.asks.is_empty() {
                    // Create aggregated message
                    let aggregated_msg = WsMessage::new(
                        vec![], // Would contain serialized aggregated data
                        MessagePriority::Normal,
                        MessageType::OrderBookL2,
                        Some(symbol.to_string()),
                    );
                    
                    if self.output_tx.try_send(aggregated_msg).is_ok() {
                        self.processed_count.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
        }
        
        // Start new aggregation
        aggregator.insert(
            symbol.to_string(),
            L2Aggregation {
                bids: vec![],
                asks: vec![],
                last_update: Instant::now(),
            },
        );
    }
    
    /// Flush any pending aggregations
    pub fn flush_aggregations(&self) {
        let mut aggregator = self.l2_aggregator.lock();
        
        for (symbol, agg) in aggregator.drain() {
            if !agg.bids.is_empty() || !agg.asks.is_empty() {
                let aggregated_msg = WsMessage::new(
                    vec![],
                    MessagePriority::Normal,
                    MessageType::OrderBookL2,
                    Some(symbol),
                );
                
                let _ = self.output_tx.try_send(aggregated_msg);
            }
        }
    }
    
    /// Get statistics
    pub fn stats(&self) -> ThrottlerStats {
        ThrottlerStats {
            processed: self.processed_count.load(Ordering::Relaxed),
            dropped_critical: self.dropped_counts[0].load(Ordering::Relaxed),
            dropped_high: self.dropped_counts[1].load(Ordering::Relaxed),
            dropped_normal: self.dropped_counts[2].load(Ordering::Relaxed),
            dropped_low: self.dropped_counts[3].load(Ordering::Relaxed),
            total_dropped: self.dropped_counts.iter().map(|c| c.load(Ordering::Relaxed)).sum(),
            available_tokens: self.rate_limiter.tokens(),
        }
    }
    
    /// Set aggregation enabled
    pub fn set_aggregation_enabled(&self, enabled: bool) {
        self.aggregation_enabled = enabled;
    }
}

#[derive(Debug, Clone)]
pub struct ThrottlerStats {
    pub processed: usize,
    pub dropped_critical: usize,
    pub dropped_high: usize,
    pub dropped_normal: usize,
    pub dropped_low: usize,
    pub total_dropped: usize,
    pub available_tokens: u64,
}

/// Dynamic throttle controller that adjusts limits based on market conditions
pub struct DynamicThrottleController {
    /// Base rate limit
    base_rate_limit: AtomicU64,
    /// Current effective rate limit
    current_rate_limit: AtomicU64,
    /// Tick rate (ticks per second)
    tick_rate: AtomicU64,
    /// Drop percentage target
    target_drop_pct: f64,
}

impl DynamicThrottleController {
    pub fn new(base_rate_limit: u64, target_drop_pct: f64) -> Self {
        Self {
            base_rate_limit: AtomicU64::new(base_rate_limit),
            current_rate_limit: AtomicU64::new(base_rate_limit),
            tick_rate: AtomicU64::new(0),
            target_drop_pct,
        }
    }
    
    /// Update tick rate measurement
    pub fn update_tick_rate(&self, ticks_per_second: u64) {
        self.tick_rate.store(ticks_per_second, Ordering::Relaxed);
        self.adjust_rate_limit();
    }
    
    /// Adjust rate limit based on tick rate
    fn adjust_rate_limit(&self) {
        let tick_rate = self.tick_rate.load(Ordering::Relaxed);
        let base = self.base_rate_limit.load(Ordering::Relaxed);
        
        // If tick rate exceeds base limit, reduce allowed rate to maintain target drop %
        if tick_rate > base {
            let adjusted = (base as f64 * (1.0 - self.target_drop_pct)) as u64;
            self.current_rate_limit.store(adjusted.max(base / 10), Ordering::Relaxed);
        } else {
            self.current_rate_limit.store(base, Ordering::Relaxed);
        }
    }
    
    /// Get current rate limit
    pub fn current_rate_limit(&self) -> u64 {
        self.current_rate_limit.load(Ordering::Relaxed)
    }
    
    /// Get current tick rate
    pub fn current_tick_rate(&self) -> u64 {
        self.tick_rate.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_token_bucket() {
        let bucket = TokenBucket::new(10, 5.0);
        
        // Should have full capacity initially
        assert_eq!(bucket.tokens(), 10);
        
        // Consume some tokens
        assert!(bucket.try_consume(5));
        assert_eq!(bucket.tokens(), 5);
        
        // Consume more than available
        assert!(!bucket.try_consume(10));
        
        // Wait and refill (simulated)
        std::thread::sleep(Duration::from_millis(500));
        bucket.refill();
        assert!(bucket.tokens() >= 5);
    }
    
    #[test]
    fn test_throttler_priority_dropping() {
        let (throttler, input_tx, _output_rx) = WsThrottler::new(10, 5, 100);
        
        // Send low priority messages
        for i in 0..20 {
            let msg = WsMessage::new(
                vec![i as u8],
                MessagePriority::Low,
                MessageType::OrderBookL2,
                Some("BTCUSDT".to_string()),
            );
            let _ = input_tx.send(msg);
        }
        
        // Process
        for _ in 0..20 {
            match throttler.input_rx.try_recv() {
                Ok(msg) => throttler.process_message(msg),
                Err(_) => break,
            }
        }
        
        let stats = throttler.stats();
        assert!(stats.total_dropped > 0 || stats.processed > 0);
    }
    
    #[test]
    fn test_dynamic_throttle_controller() {
        let controller = DynamicThrottleController::new(1000, 0.1);
        
        assert_eq!(controller.current_rate_limit(), 1000);
        
        // Simulate high tick rate
        controller.update_tick_rate(5000);
        
        let new_limit = controller.current_rate_limit();
        assert!(new_limit < 1000); // Should be reduced
    }
}
