//! Order Book Module Root
//! 
//! Implements a global Order Book Manager that spawns a concurrent actor per trading pair.

pub mod price_level;
pub mod book;

pub use price_level::*;
pub use book::*;

use std::sync::OnceLock;
use crate::market_data::SymbolId;

/// Global order book manager singleton
static ORDER_BOOK_MANAGER: OnceLock<OrderBookManager> = OnceLock::new();

/// Initialize the global order book manager
#[inline]
pub fn init_order_book_manager() -> &'static OrderBookManager {
    ORDER_BOOK_MANAGER.get_or_init(OrderBookManager::new)
}

/// Get a reference to the global order book manager
#[inline]
pub fn get_order_book_manager() -> Option<&'static OrderBookManager> {
    ORDER_BOOK_MANAGER.get()
}

/// Order book actor message types for concurrent processing
#[derive(Debug, Clone)]
pub enum OrderBookCommand {
    /// Apply a delta update
    ApplyDelta(OrderBookDelta),
    /// Apply a full snapshot (reset)
    ApplySnapshot(OrderBookSnapshot),
    /// Request current book state
    GetBook { symbol: SymbolId },
    /// Reset a specific book
    Reset(SymbolId),
    /// Subscribe to book updates
    Subscribe { symbol: SymbolId, channel_id: u64 },
    /// Unsubscribe from book updates
    Unsubscribe { symbol: SymbolId, channel_id: u64 },
}

/// Order book actor response types
#[derive(Debug, Clone)]
pub enum OrderBookResponse {
    /// Delta applied successfully
    DeltaApplied { symbol: SymbolId, sequence: u64 },
    /// Snapshot applied
    SnapshotApplied { symbol: SymbolId },
    /// Current book state
    BookState(OrderBook),
    /// Error occurred
    Error(String),
    /// Best bid/ask update
    QuoteUpdate {
        symbol: SymbolId,
        bid_price: Option<Price>,
        ask_price: Option<Price>,
        timestamp_ns: i64,
    },
}

/// Actor-based order book processor for a single symbol
/// 
/// This can be spawned as a tokio task for isolated per-symbol processing
pub struct OrderBookActor {
    symbol: SymbolId,
    book: OrderBook,
    subscribers: Vec<u64>,
}

impl OrderBookActor {
    #[inline]
    pub fn new(symbol: SymbolId) -> Self {
        OrderBookActor {
            symbol,
            book: OrderBook::new(symbol),
            subscribers: Vec::new(),
        }
    }

    /// Process a command and return a response
    #[inline]
    pub fn process(&mut self, cmd: OrderBookCommand) -> Option<OrderBookResponse> {
        match cmd {
            OrderBookCommand::ApplyDelta(delta) => {
                match self.book.apply_delta(&delta) {
                    Ok(()) => {
                        Some(OrderBookResponse::DeltaApplied {
                            symbol: self.symbol,
                            sequence: delta.sequence,
                        })
                    }
                    Err(e) => Some(OrderBookResponse::Error(e.to_string())),
                }
            }
            OrderBookCommand::ApplySnapshot(snapshot) => {
                self.book = OrderBook::from_snapshot(snapshot);
                Some(OrderBookResponse::SnapshotApplied { symbol: self.symbol })
            }
            OrderBookCommand::GetBook { .. } => {
                Some(OrderBookResponse::BookState(self.book.clone()))
            }
            OrderBookCommand::Reset(_) => {
                self.book.reset();
                Some(OrderBookResponse::BookState(self.book.clone()))
            }
            OrderBookCommand::Subscribe { channel_id, .. } => {
                if !self.subscribers.contains(&channel_id) {
                    self.subscribers.push(channel_id);
                }
                None
            }
            OrderBookCommand::Unsubscribe { channel_id, .. } => {
                self.subscribers.retain(|&id| id != channel_id);
                None
            }
        }
    }

    /// Notify all subscribers of a quote update
    #[inline]
    pub fn notify_quote_update(&self) -> Option<OrderBookResponse> {
        if self.subscribers.is_empty() {
            return None;
        }

        Some(OrderBookResponse::QuoteUpdate {
            symbol: self.symbol,
            bid_price: self.book.best_bid(),
            ask_price: self.book.best_ask(),
            timestamp_ns: self.book.timestamp_ns,
        })
    }

    /// Get the current book reference
    #[inline]
    pub fn book(&self) -> &OrderBook {
        &self.book
    }

    /// Get subscriber count
    #[inline]
    pub fn subscriber_count(&self) -> usize {
        self.subscribers.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::market_data::{Level, Price, Quantity};

    #[test]
    fn test_actor_process_delta() {
        let symbol = SymbolId::from_str("BTC-USD");
        let mut actor = OrderBookActor::new(symbol);

        let mut delta = OrderBookDelta::new(symbol, 1);
        delta.bids.push(Level::new(Price::from_f64(50000.0), Quantity::from_f64(1.0), 1));
        delta.asks.push(Level::new(Price::from_f64(50001.0), Quantity::from_f64(0.5), 1));

        let response = actor.process(OrderBookCommand::ApplyDelta(delta)).unwrap();
        
        match response {
            OrderBookResponse::DeltaApplied { symbol: s, sequence } => {
                assert_eq!(s, symbol);
                assert_eq!(sequence, 1);
            }
            _ => panic!("Unexpected response"),
        }
    }

    #[test]
    fn test_actor_subscribe_unsubscribe() {
        let symbol = SymbolId::from_str("BTC-USD");
        let mut actor = OrderBookActor::new(symbol);

        actor.process(OrderBookCommand::Subscribe { symbol, channel_id: 1 });
        assert_eq!(actor.subscriber_count(), 1);

        actor.process(OrderBookCommand::Unsubscribe { symbol, channel_id: 1 });
        assert_eq!(actor.subscriber_count(), 0);
    }
}
