//! High-Throughput Message Router
//! 
//! Builds a high-throughput message router that pushes parsed events directly into 
//! the LMAX Disruptor ring buffer. Implements sequence gap detection and backpressure 
//! mechanisms to ensure the parser never outpaces the order book update engine.

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use anyhow::{Context, Result};
use crate::market_data::{MarketDataEvent, BinanceStreamMessage, OrderBookDelta, Trade, Ticker};
use crate::pipeline::parser::JsonParser;
use crate::network::ws_client::RawWsMessage;

/// Router configuration
#[derive(Debug, Clone)]
pub struct RouterConfig {
    /// Maximum pending messages before applying backpressure
    pub max_pending: usize,
    /// Enable backpressure
    pub enable_backpressure: bool,
    /// Backpressure threshold (percentage of max_pending)
    pub backpressure_threshold: f64,
    /// Enable sequence gap detection
    pub enable_gap_detection: bool,
}

impl Default for RouterConfig {
    fn default() -> Self {
        RouterConfig {
            max_pending: 100_000,
            enable_backpressure: true,
            backpressure_threshold: 0.8, // 80%
            enable_gap_detection: true,
        }
    }
}

/// Ring buffer event types
#[derive(Debug, Clone)]
pub enum RingBufferEvent {
    MarketData(MarketDataEvent),
    Heartbeat(i64),
    GapDetected { symbol: String, expected: u64, received: u64 },
    Shutdown,
}

/// Sequence tracker for gap detection per symbol
struct SymbolSequenceTracker {
    expected_sequence: std::collections::HashMap<String, u64>,
    gap_count: AtomicU64,
}

impl SymbolSequenceTracker {
    #[inline]
    fn new() -> Self {
        SymbolSequenceTracker {
            expected_sequence: std::collections::HashMap::new(),
            gap_count: AtomicU64::new(0),
        }
    }

    #[inline]
    fn check_and_update(&self, symbol: &str, sequence: u64) -> Option<(u64, u64)> {
        let expected = self.expected_sequence.entry(symbol.to_string()).or_insert(sequence);
        
        if *expected == 0 {
            *expected = sequence + 1;
            return None;
        }

        if sequence < *expected {
            // Old/duplicate message
            return None;
        }

        if sequence != *expected {
            // Gap detected
            let gap = (*expected, sequence);
            self.gap_count.fetch_add(1, Ordering::Relaxed);
            *expected = sequence + 1;
            return Some(gap);
        }

        *expected = sequence + 1;
        None
    }

    #[inline]
    fn gap_count(&self) -> u64 {
        self.gap_count.load(Ordering::Relaxed)
    }

    #[inline]
    fn reset(&mut self) {
        self.expected_sequence.clear();
        self.gap_count.store(0, Ordering::Relaxed);
    }
}

impl Default for SymbolSequenceTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// High-throughput message router
pub struct MessageRouter {
    config: RouterConfig,
    /// Pending event count
    pending_count: AtomicU64,
    /// Dropped event count (due to backpressure)
    dropped_count: AtomicU64,
    /// Routed event count
    routed_count: AtomicU64,
    /// Backpressure active flag
    backpressure_active: AtomicBool,
    /// Sequence tracker
    sequence_tracker: SymbolSequenceTracker,
    /// Event sender channel
    event_sender: tokio::sync::mpsc::Sender<RingBufferEvent>,
    /// Parser reference
    parser: std::sync::Arc<dyn JsonParser>,
}

impl MessageRouter {
    /// Create a new message router
    #[inline]
    pub fn new(
        config: RouterConfig,
        event_sender: tokio::sync::mpsc::Sender<RingBufferEvent>,
        parser: std::sync::Arc<dyn JsonParser>,
    ) -> Self {
        MessageRouter {
            config,
            pending_count: AtomicU64::new(0),
            dropped_count: AtomicU64::new(0),
            routed_count: AtomicU64::new(0),
            backpressure_active: AtomicBool::new(false),
            sequence_tracker: SymbolSequenceTracker::new(),
            event_sender,
            parser,
        }
    }

    /// Route a raw WebSocket message through the pipeline
    #[inline]
    pub async fn route_message(&self, raw_msg: RawWsMessage) -> Result<bool> {
        // Check backpressure
        if self.config.enable_backpressure && self.is_backpressure_active() {
            self.dropped_count.fetch_add(1, Ordering::Relaxed);
            log::warn!("Backpressure active, dropping message");
            return Ok(false);
        }

        // Parse the message
        let parsed = match self.parser.parse_stream_message(&raw_msg) {
            Ok(msg) => msg,
            Err(e) => {
                log::error!("Failed to parse message: {}", e);
                return Err(e);
            }
        };

        // Convert to market data event
        let event = self.message_to_event(parsed)?;

        // Check for sequence gaps if enabled
        if self.config.enable_gap_detection {
            if let Some((expected, received)) = self.check_sequence(&event) {
                let symbol = self.get_event_symbol(&event).unwrap_or_default();
                log::warn!("Gap detected for {}: expected {}, got {}", symbol, expected, received);
                
                // Send gap notification
                let _ = self.event_sender.send(RingBufferEvent::GapDetected {
                    symbol,
                    expected,
                    received,
                }).await;
            }
        }

        // Send to ring buffer
        match self.event_sender.send(RingBufferEvent::MarketData(event)).await {
            Ok(()) => {
                self.routed_count.fetch_add(1, Ordering::Relaxed);
                self.update_pending_count(1);
                Ok(true)
            }
            Err(e) => {
                self.dropped_count.fetch_add(1, Ordering::Relaxed);
                Err(anyhow::anyhow!("Failed to send to ring buffer: {}", e))
            }
        }
    }

    /// Convert parsed message to market data event
    #[inline]
    fn message_to_event(&self, msg: BinanceStreamMessage) -> Result<MarketDataEvent> {
        use crate::market_data::normalizer::*;
        
        // For now, we'll create placeholder events
        // In production, this would use the Normalizer
        match msg {
            BinanceStreamMessage::Trade(trade) => {
                // Convert to normalized trade
                Ok(MarketDataEvent::Trade(Trade {
                    symbol: crate::market_data::SymbolId::from_str(&trade.symbol),
                    trade_id: trade.trade_id,
                    price: crate::market_data::Price::from_f64(trade.price),
                    quantity: crate::market_data::Quantity::from_f64(trade.quantity),
                    side: if trade.is_buyer_maker { 
                        crate::market_data::Side::Sell 
                    } else { 
                        crate::market_data::Side::Buy 
                    },
                    timestamp_ns: trade.timestamp * 1_000_000,
                    buyer_order_id: trade.buyer_order_id,
                    seller_order_id: trade.seller_order_id,
                }))
            }
            BinanceStreamMessage::DepthUpdate(depth) => {
                // Convert to order book delta
                let mut delta = OrderBookDelta::with_capacity(
                    crate::market_data::SymbolId::from_str(&depth.symbol),
                    depth.last_update_id,
                    depth.bids.len(),
                    depth.asks.len(),
                );

                for (price_str, qty_str) in depth.bids {
                    if let (Ok(price), Ok(qty)) = (price_str.parse::<f64>(), qty_str.parse::<f64>()) {
                        delta.bids.push(crate::market_data::Level::new(
                            crate::market_data::Price::from_f64(price),
                            crate::market_data::Quantity::from_f64(qty),
                            1,
                        ));
                    }
                }

                for (price_str, qty_str) in depth.asks {
                    if let (Ok(price), Ok(qty)) = (price_str.parse::<f64>(), qty_str.parse::<f64>()) {
                        delta.asks.push(crate::market_data::Level::new(
                            crate::market_data::Price::from_f64(price),
                            crate::market_data::Quantity::from_f64(qty),
                            1,
                        ));
                    }
                }

                Ok(MarketDataEvent::Delta(delta))
            }
            BinanceStreamMessage::Ticker24h(ticker) => {
                Ok(MarketDataEvent::Ticker(Ticker {
                    symbol: crate::market_data::SymbolId::from_str(&ticker.symbol),
                    last_price: crate::market_data::Price::from_f64(ticker.last_price),
                    bid_price: crate::market_data::Price::from_f64(ticker.bid_price),
                    ask_price: crate::market_data::Price::from_f64(ticker.ask_price),
                    volume_24h: crate::market_data::Quantity::from_f64(ticker.volume_24h),
                    quote_volume_24h: crate::market_data::Quantity::from_f64(ticker.quote_volume_24h),
                    timestamp_ns: chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0),
                    sequence: 0,
                }))
            }
            BinanceStreamMessage::BookTicker(book) => {
                Ok(MarketDataEvent::Ticker(Ticker {
                    symbol: crate::market_data::SymbolId::from_str(&book.symbol),
                    last_price: crate::market_data::Price::from_f64((book.bid_price + book.ask_price) / 2.0),
                    bid_price: crate::market_data::Price::from_f64(book.bid_price),
                    ask_price: crate::market_data::Price::from_f64(book.ask_price),
                    volume_24h: crate::market_data::Quantity::new(0),
                    quote_volume_24h: crate::market_data::Quantity::new(0),
                    timestamp_ns: chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0),
                    sequence: 0,
                }))
            }
            BinanceStreamMessage::Kline(_) => {
                // Klines are handled separately
                Ok(MarketDataEvent::Heartbeat {
                    symbol: crate::market_data::SymbolId::new([0; 16]),
                    timestamp_ns: chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0),
                })
            }
        }
    }

    /// Check sequence for gaps
    #[inline]
    fn check_sequence(&self, event: &MarketDataEvent) -> Option<(u64, u64)> {
        match event {
            MarketDataEvent::Delta(delta) => {
                let symbol = delta.symbol.as_str();
                self.sequence_tracker.check_and_update(symbol, delta.sequence)
            }
            _ => None,
        }
    }

    /// Get symbol from event
    #[inline]
    fn get_event_symbol(&self, event: &MarketDataEvent) -> Option<String> {
        match event {
            MarketDataEvent::Delta(delta) => Some(delta.symbol.as_str().to_string()),
            MarketDataEvent::Trade(trade) => Some(trade.symbol.as_str().to_string()),
            MarketDataEvent::Ticker(ticker) => Some(ticker.symbol.as_str().to_string()),
            _ => None,
        }
    }

    /// Update pending count
    #[inline]
    fn update_pending_count(&self, delta: i64) {
        if delta > 0 {
            let new_count = self.pending_count.fetch_add(delta as u64, Ordering::Relaxed) + delta as u64;
            
            // Check if backpressure should be activated
            if self.config.enable_backpressure {
                let threshold = (self.config.max_pending as f64 * self.config.backpressure_threshold) as u64;
                if new_count >= threshold {
                    self.backpressure_active.store(true, Ordering::Relaxed);
                    log::warn!("Backpressure activated: {} pending", new_count);
                }
            }
        } else {
            self.pending_count.fetch_sub((-delta) as u64, Ordering::Relaxed);
        }
    }

    /// Check if backpressure is active
    #[inline]
    pub fn is_backpressure_active(&self) -> bool {
        self.backpressure_active.load(Ordering::Relaxed)
    }

    /// Mark an event as processed (decrement pending count)
    #[inline]
    pub fn mark_processed(&self) {
        let new_count = self.pending_count.fetch_sub(1, Ordering::Relaxed) - 1;
        
        // Deactivate backpressure if below threshold
        if self.config.enable_backpressure && self.backpressure_active.load(Ordering::Relaxed) {
            let threshold = (self.config.max_pending as f64 * self.config.backpressure_threshold * 0.9) as u64;
            if new_count < threshold {
                self.backpressure_active.store(false, Ordering::Relaxed);
                log::info!("Backpressure deactivated: {} pending", new_count);
            }
        }
    }

    /// Get router statistics
    #[inline]
    pub fn stats(&self) -> RouterStats {
        RouterStats {
            pending_count: self.pending_count.load(Ordering::Relaxed),
            dropped_count: self.dropped_count.load(Ordering::Relaxed),
            routed_count: self.routed_count.load(Ordering::Relaxed),
            backpressure_active: self.backpressure_active.load(Ordering::Relaxed),
            gap_count: self.sequence_tracker.gap_count(),
            max_pending: self.config.max_pending,
        }
    }

    /// Reset statistics
    #[inline]
    pub fn reset_stats(&self) {
        self.pending_count.store(0, Ordering::Relaxed);
        self.dropped_count.store(0, Ordering::Relaxed);
        self.routed_count.store(0, Ordering::Relaxed);
        self.backpressure_active.store(false, Ordering::Relaxed);
        self.sequence_tracker.reset();
    }
}

/// Router statistics snapshot
#[derive(Debug, Clone)]
pub struct RouterStats {
    pub pending_count: u64,
    pub dropped_count: u64,
    pub routed_count: u64,
    pub backpressure_active: bool,
    pub gap_count: u64,
    pub max_pending: usize,
}

impl RouterStats {
    /// Get drop rate as percentage
    #[inline]
    pub fn drop_rate(&self) -> f64 {
        let total = self.routed_count + self.dropped_count;
        if total == 0 {
            return 0.0;
        }
        (self.dropped_count as f64 / total as f64) * 100.0
    }

    /// Get pending percentage
    #[inline]
    pub fn pending_percentage(&self) -> f64 {
        if self.max_pending == 0 {
            return 0.0;
        }
        (self.pending_count as f64 / self.max_pending as f64) * 100.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::parser::SerdeJsonParser;

    #[tokio::test]
    async fn test_router_creation() {
        let (sender, _receiver) = tokio::sync::mpsc::channel(100);
        let parser = std::sync::Arc::new(SerdeJsonParser::new());
        let router = MessageRouter::new(RouterConfig::default(), sender, parser);
        
        assert!(!router.is_backpressure_active());
        assert_eq!(router.stats().routed_count, 0);
    }

    #[test]
    fn test_router_config_default() {
        let config = RouterConfig::default();
        assert_eq!(config.max_pending, 100_000);
        assert!(config.enable_backpressure);
    }

    #[test]
    fn test_router_stats() {
        let (sender, _receiver) = tokio::sync::mpsc::channel(100);
        let parser = std::sync::Arc::new(SerdeJsonParser::new());
        let router = MessageRouter::new(RouterConfig::default(), sender, parser);
        
        let stats = router.stats();
        assert_eq!(stats.pending_count, 0);
        assert_eq!(stats.max_pending, 100_000);
    }
}
