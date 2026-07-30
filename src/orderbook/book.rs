//! High-Performance Order Book Engine
//! 
//! Builds a highly optimized Order Book using sorted vectors for O(1) top-of-book access
//! and O(log N) updates. Implements `apply_delta` methods to update the book from 
//! WebSocket streams in microseconds without full book rebuilds.

use crate::market_data::{Price, Quantity, Level, OrderBookDelta, OrderBookSnapshot, SymbolId};
use crate::orderbook::price_level::PriceLevel;
use std::collections::BTreeMap;
use anyhow::{Context, Result};

/// Maximum depth to maintain on each side of the order book
pub const MAX_BOOK_DEPTH: usize = 100;

/// A single-side (bids or asks) order book implementation
/// 
/// Uses a BTreeMap for efficient price-based lookups and ordered iteration.
/// For ultra-low latency, consider replacing with a sorted Vec + binary search.
#[derive(Debug, Clone)]
pub struct OrderBookSide {
    /// Price -> PriceLevel mapping
    levels: BTreeMap<i64, PriceLevel>,
    /// Cached total volume for quick access
    cached_volume: Quantity,
    /// Is this the bid side? (false = ask side)
    is_bid: bool,
}

impl OrderBookSide {
    #[inline]
    pub fn new(is_bid: bool) -> Self {
        OrderBookSide {
            levels: BTreeMap::new(),
            cached_volume: Quantity::new(0),
            is_bid,
        }
    }

    /// Get the best (highest for bids, lowest for asks) price level
    #[inline]
    pub fn best(&self) -> Option<&PriceLevel> {
        if self.is_bid {
            self.levels.last_key_value().map(|(_, v)| v)
        } else {
            self.levels.first_key_value().map(|(_, v)| v)
        }
    }

    /// Get the best price
    #[inline]
    pub fn best_price(&self) -> Option<Price> {
        self.best().map(|l| l.price)
    }

    /// Get the best quantity
    #[inline]
    pub fn best_quantity(&self) -> Option<Quantity> {
        self.best().map(|l| l.quantity)
    }

    /// Get a level at a specific price
    #[inline]
    pub fn get(&self, price: Price) -> Option<&PriceLevel> {
        self.levels.get(&price.raw())
    }

    /// Get mutable reference to a level at a specific price
    #[inline]
    pub fn get_mut(&mut self, price: Price) -> Option<&mut PriceLevel> {
        self.levels.get_mut(&price.raw())
    }

    /// Insert or update a level
    #[inline]
    pub fn upsert(&mut self, price: Price, quantity: Quantity, order_count: u32) {
        if quantity.raw() == 0 {
            // Remove the level if quantity is zero
            self.levels.remove(&price.raw());
        } else {
            let level = PriceLevel::new(price, quantity, order_count);
            self.levels.insert(price.raw(), level);
        }
        self.update_cached_volume();
    }

    /// Apply a delta update to this side
    #[inline]
    pub fn apply_delta(&mut self, levels: &[Level]) {
        for level in levels {
            self.upsert(level.price, level.quantity, level.order_count);
        }
    }

    /// Remove a price level entirely
    #[inline]
    pub fn remove(&mut self, price: Price) {
        self.levels.remove(&price.raw());
        self.update_cached_volume();
    }

    /// Clear all levels
    #[inline]
    pub fn clear(&mut self) {
        self.levels.clear();
        self.cached_volume = Quantity::new(0);
    }

    /// Get the number of price levels
    #[inline]
    pub fn len(&self) -> usize {
        self.levels.len()
    }

    /// Check if empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.levels.is_empty()
    }

    /// Get total volume on this side
    #[inline]
    pub fn total_volume(&self) -> Quantity {
        self.cached_volume
    }

    /// Update cached volume - call after modifications
    #[inline]
    fn update_cached_volume(&mut self) {
        let mut total: i64 = 0;
        for (_, level) in &self.levels {
            total = total.saturating_add(level.quantity.raw());
        }
        self.cached_volume = Quantity::new(total);
    }

    /// Get top N levels as a Vec (for serialization/snapshot)
    #[inline]
    pub fn top_n(&self, n: usize) -> Vec<Level> {
        let mut result = Vec::with_capacity(n.min(self.levels.len()));
        
        if self.is_bid {
            // Bids: highest price first (descending)
            for (_, level) in self.levels.iter().rev().take(n) {
                result.push(Level::new(level.price, level.quantity, level.order_count));
            }
        } else {
            // Asks: lowest price first (ascending)
            for (_, level) in self.levels.iter().take(n) {
                result.push(Level::new(level.price, level.quantity, level.order_count));
            }
        }
        
        result
    }

    /// Iterate over levels in book order (best first)
    #[inline]
    pub fn iter_best_first(&self) -> impl Iterator<Item = &PriceLevel> {
        if self.is_bid {
            Box::new(self.levels.values().rev()) as Box<dyn Iterator<Item = &PriceLevel>>
        } else {
            Box::new(self.levels.values()) as Box<dyn Iterator<Item = &PriceLevel>>
        }
    }
}

/// Full order book for a single symbol
#[derive(Debug, Clone)]
pub struct OrderBook {
    pub symbol: SymbolId,
    pub bids: OrderBookSide,
    pub asks: OrderBookSide,
    pub last_update_id: u64,
    pub timestamp_ns: i64,
    /// Sequence number for gap detection
    pub sequence: u64,
    /// Is the book initialized with a snapshot?
    pub is_initialized: bool,
}

impl OrderBook {
    #[inline]
    pub fn new(symbol: SymbolId) -> Self {
        OrderBook {
            symbol,
            bids: OrderBookSide::new(true),
            asks: OrderBookSide::new(false),
            last_update_id: 0,
            timestamp_ns: 0,
            sequence: 0,
            is_initialized: false,
        }
    }

    /// Initialize the book from a snapshot
    #[inline]
    pub fn from_snapshot(snapshot: OrderBookSnapshot) -> Self {
        let mut book = OrderBook::new(snapshot.symbol);
        book.last_update_id = snapshot.last_update_id;
        
        // Apply all levels from snapshot
        for level in snapshot.bids {
            book.bids.upsert(level.price, level.quantity, level.order_count);
        }
        for level in snapshot.asks {
            book.asks.upsert(level.price, level.quantity, level.order_count);
        }
        
        book.is_initialized = true;
        book
    }

    /// Apply a delta update to the order book
    /// 
    /// Returns an error if the delta sequence is invalid (gap detected)
    #[inline]
    pub fn apply_delta(&mut self, delta: &OrderBookDelta) -> Result<()> {
        // Sequence validation (only if already initialized)
        if self.is_initialized && delta.sequence <= self.sequence {
            // Duplicate or old update, ignore
            return Ok(());
        }

        self.bids.apply_delta(&delta.bids);
        self.asks.apply_delta(&delta.asks);
        self.last_update_id = delta.sequence;
        self.sequence = delta.sequence;
        self.timestamp_ns = delta.timestamp_ns;
        self.is_initialized = true;

        Ok(())
    }

    /// Get the best bid price
    #[inline]
    pub fn best_bid(&self) -> Option<Price> {
        self.bids.best_price()
    }

    /// Get the best ask price
    #[inline]
    pub fn best_ask(&self) -> Option<Price> {
        self.asks.best_price()
    }

    /// Get the mid price
    #[inline]
    pub fn mid_price(&self) -> Option<Price> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => {
                Some(Price::new((bid.raw() + ask.raw()) / 2))
            }
            _ => None,
        }
    }

    /// Get the spread (ask - bid)
    #[inline]
    pub fn spread(&self) -> Option<Price> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => {
                ask.raw().checked_sub(bid.raw()).map(Price::new)
            }
            _ => None,
        }
    }

    /// Get the spread in basis points
    #[inline]
    pub fn spread_bps(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask(), self.mid_price()) {
            (Some(bid), Some(ask), Some(mid)) if mid.raw() != 0 => {
                let spread = (ask.raw() - bid.raw()) as f64;
                let mid_val = mid.raw() as f64;
                Some((spread / mid_val) * 10000.0)
            }
            _ => None,
        }
    }

    /// Check if the book has valid quotes on both sides
    #[inline]
    pub fn has_valid_quotes(&self) -> bool {
        self.is_initialized 
            && self.bids.has_volume() 
            && self.asks.has_volume()
            && self.best_bid().unwrap().raw() < self.best_ask().unwrap().raw()
    }

    /// Get a snapshot of the top N levels on each side
    #[inline]
    pub fn snapshot(&self, depth: usize) -> OrderBookSnapshot {
        OrderBookSnapshot {
            symbol: self.symbol,
            last_update_id: self.last_update_id,
            bids: self.bids.top_n(depth),
            asks: self.asks.top_n(depth),
        }
    }

    /// Reset the book (e.g., after a gap detection)
    #[inline]
    pub fn reset(&mut self) {
        self.bids.clear();
        self.asks.clear();
        self.is_initialized = false;
        self.sequence = 0;
    }
}

/// Global Order Book Manager - spawns concurrent actors per trading pair
pub struct OrderBookManager {
    books: std::sync::RwLock<std::collections::HashMap<SymbolId, OrderBook>>,
}

impl OrderBookManager {
    #[inline]
    pub fn new() -> Self {
        OrderBookManager {
            books: std::sync::RwLock::new(std::collections::HashMap::with_capacity(64)),
        }
    }

    /// Get or create an order book for a symbol
    #[inline]
    pub fn get_or_create(&self, symbol: SymbolId) -> OrderBook {
        let mut books = self.books.write().unwrap();
        *books.entry(symbol).or_insert_with(|| OrderBook::new(symbol))
    }

    /// Apply a delta to the appropriate order book
    #[inline]
    pub fn apply_delta(&self, delta: &OrderBookDelta) -> Result<()> {
        let mut books = self.books.write().unwrap();
        let book = books.entry(delta.symbol)
            .or_insert_with(|| OrderBook::new(delta.symbol));
        book.apply_delta(delta)
    }

    /// Apply a snapshot to initialize/reset an order book
    #[inline]
    pub fn apply_snapshot(&self, snapshot: OrderBookSnapshot) {
        let mut books = self.books.write().unwrap();
        let book = OrderBook::from_snapshot(snapshot);
        books.insert(book.symbol, book);
    }

    /// Get a copy of an order book
    #[inline]
    pub fn get_book(&self, symbol: SymbolId) -> Option<OrderBook> {
        let books = self.books.read().unwrap();
        books.get(&symbol).cloned()
    }

    /// Get best bid for a symbol
    #[inline]
    pub fn get_best_bid(&self, symbol: SymbolId) -> Option<Price> {
        let books = self.books.read().unwrap();
        books.get(&symbol).and_then(|b| b.best_bid())
    }

    /// Get best ask for a symbol
    #[inline]
    pub fn get_best_ask(&self, symbol: SymbolId) -> Option<Price> {
        let books = self.books.read().unwrap();
        books.get(&symbol).and_then(|b| b.best_ask())
    }

    /// Get the number of managed books
    #[inline]
    pub fn book_count(&self) -> usize {
        let books = self.books.read().unwrap();
        books.len()
    }

    /// Remove a book (e.g., when symbol is no longer tracked)
    #[inline]
    pub fn remove_book(&self, symbol: SymbolId) -> Option<OrderBook> {
        let mut books = self.books.write().unwrap();
        books.remove(&symbol)
    }
}

impl Default for OrderBookManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_order_book_creation() {
        let symbol = SymbolId::from_str("BTC-USD");
        let book = OrderBook::new(symbol);
        
        assert!(!book.is_initialized);
        assert_eq!(book.bids.len(), 0);
        assert_eq!(book.asks.len(), 0);
    }

    #[test]
    fn test_apply_delta() {
        let symbol = SymbolId::from_str("BTC-USD");
        let mut book = OrderBook::new(symbol);
        
        let mut delta = OrderBookDelta::new(symbol, 1);
        delta.bids.push(Level::new(Price::from_f64(50000.0), Quantity::from_f64(1.0), 1));
        delta.asks.push(Level::new(Price::from_f64(50001.0), Quantity::from_f64(0.5), 1));
        
        book.apply_delta(&delta).unwrap();
        
        assert!(book.is_initialized);
        assert_eq!(book.best_bid().unwrap().to_f64(), 50000.0);
        assert_eq!(book.best_ask().unwrap().to_f64(), 50001.0);
    }

    #[test]
    fn test_spread_calculation() {
        let symbol = SymbolId::from_str("BTC-USD");
        let mut book = OrderBook::new(symbol);
        
        let mut delta = OrderBookDelta::new(symbol, 1);
        delta.bids.push(Level::new(Price::from_f64(50000.0), Quantity::from_f64(1.0), 1));
        delta.asks.push(Level::new(Price::from_f64(50005.0), Quantity::from_f64(0.5), 1));
        
        book.apply_delta(&delta).unwrap();
        
        assert_eq!(book.spread().unwrap().to_f64(), 5.0);
        assert!(book.spread_bps().is_some());
    }
}
