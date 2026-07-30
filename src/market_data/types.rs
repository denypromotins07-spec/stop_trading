//! Core Market Data Types & Normalization
//! 
//! Defines strictly typed, zero-copy structs for normalized Tickers, Trades, and L2/L3 Order Book deltas.
//! Memory layout is contiguous and cache-friendly to minimize CPU cache misses during high-frequency data ingestion.

use std::mem;

/// Fixed-point price representation (price * 10^8) to avoid floating-point inaccuracies
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct Price(pub i64);

impl Price {
    #[inline]
    pub const fn new(raw: i64) -> Self {
        Price(raw)
    }

    #[inline]
    pub const fn from_f64(price: f64) -> Self {
        Price((price * 1e8) as i64)
    }

    #[inline]
    pub const fn to_f64(self) -> f64 {
        self.0 as f64 / 1e8
    }

    #[inline]
    pub const fn raw(self) -> i64 {
        self.0
    }
}

/// Fixed-point quantity representation (quantity * 10^8)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct Quantity(pub i64);

impl Quantity {
    #[inline]
    pub const fn new(raw: i64) -> Self {
        Quantity(raw)
    }

    #[inline]
    pub const fn from_f64(qty: f64) -> Self {
        Quantity((qty * 1e8) as i64)
    }

    #[inline]
    pub const fn to_f64(self) -> f64 {
        self.0 as f64 / 1e8
    }

    #[inline]
    pub const fn raw(self) -> i64 {
        self.0
    }

    #[inline]
    pub const fn is_zero(self) -> bool {
        self.0 == 0
    }
}

/// Unified symbol identifier using fixed-size array for cache efficiency
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct SymbolId(pub [u8; 16]);

impl SymbolId {
    #[inline]
    pub const fn new(bytes: [u8; 16]) -> Self {
        SymbolId(bytes)
    }

    #[inline]
    pub fn from_str(s: &str) -> Self {
        let mut bytes = [0u8; 16];
        let slice = s.as_bytes();
        let len = slice.len().min(16);
        bytes[..len].copy_from_slice(&slice[..len]);
        SymbolId(bytes)
    }

    #[inline]
    pub fn as_str(&self) -> &str {
        // Find first null byte or use full length
        let end = self.0.iter().position(|&b| b == 0).unwrap_or(16);
        std::str::from_utf8(&self.0[..end]).unwrap_or("")
    }
}

/// Normalized ticker data - cache-line aligned
#[derive(Debug, Clone, Copy)]
#[repr(C, align(64))]
pub struct Ticker {
    pub symbol: SymbolId,
    pub last_price: Price,
    pub bid_price: Price,
    pub ask_price: Price,
    pub volume_24h: Quantity,
    pub quote_volume_24h: Quantity,
    pub timestamp_ns: i64,
    pub sequence: u64,
}

impl Default for Ticker {
    fn default() -> Self {
        Ticker {
            symbol: SymbolId::new([0; 16]),
            last_price: Price::new(0),
            bid_price: Price::new(0),
            ask_price: Price::new(0),
            volume_24h: Quantity::new(0),
            quote_volume_24h: Quantity::new(0),
            timestamp_ns: 0,
            sequence: 0,
        }
    }
}

/// Trade side indicator
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Side {
    Buy = 0,
    Sell = 1,
}

impl Side {
    #[inline]
    pub const fn is_buy(self) -> bool {
        matches!(self, Side::Buy)
    }

    #[inline]
    pub const fn is_sell(self) -> bool {
        matches!(self, Side::Sell)
    }
}

/// Normalized trade data - compact layout
#[derive(Debug, Clone, Copy)]
#[repr(C, align(32))]
pub struct Trade {
    pub symbol: SymbolId,
    pub trade_id: u64,
    pub price: Price,
    pub quantity: Quantity,
    pub side: Side,
    pub timestamp_ns: i64,
    pub buyer_order_id: u64,
    pub seller_order_id: u64,
}

/// Order book level entry
#[derive(Debug, Clone, Copy)]
#[repr(C, align(16))]
pub struct Level {
    pub price: Price,
    pub quantity: Quantity,
    pub order_count: u32,
    _padding: u32,
}

impl Level {
    #[inline]
    pub const fn new(price: Price, quantity: Quantity, order_count: u32) -> Self {
        Level {
            price,
            quantity,
            order_count,
            _padding: 0,
        }
    }

    #[inline]
    pub const fn is_empty(self) -> bool {
        self.quantity.is_zero()
    }
}

/// Order book delta update - minimal allocation
#[derive(Debug, Clone)]
pub struct OrderBookDelta {
    pub symbol: SymbolId,
    pub sequence: u64,
    pub bids: Vec<Level>,
    pub asks: Vec<Level>,
    pub timestamp_ns: i64,
    pub is_snapshot: bool,
}

impl OrderBookDelta {
    #[inline]
    pub fn new(symbol: SymbolId, sequence: u64) -> Self {
        OrderBookDelta {
            symbol,
            sequence,
            bids: Vec::with_capacity(0),
            asks: Vec::with_capacity(0),
            timestamp_ns: 0,
            is_snapshot: false,
        }
    }

    #[inline]
    pub fn with_capacity(symbol: SymbolId, sequence: u64, bid_cap: usize, ask_cap: usize) -> Self {
        OrderBookDelta {
            symbol,
            sequence,
            bids: Vec::with_capacity(bid_cap),
            asks: Vec::with_capacity(ask_cap),
            timestamp_ns: 0,
            is_snapshot: false,
        }
    }
}

/// Full order book snapshot for initial state
#[derive(Debug, Clone)]
pub struct OrderBookSnapshot {
    pub symbol: SymbolId,
    pub last_update_id: u64,
    pub bids: Vec<Level>,
    pub asks: Vec<Level>,
}

impl OrderBookSnapshot {
    #[inline]
    pub fn new(symbol: SymbolId, last_update_id: u64) -> Self {
        OrderBookSnapshot {
            symbol,
            last_update_id,
            bids: Vec::with_capacity(100),
            asks: Vec::with_capacity(100),
        }
    }
}

/// Market data event types for the event bus
#[derive(Debug, Clone)]
pub enum MarketDataEvent {
    Ticker(Ticker),
    Trade(Trade),
    Delta(OrderBookDelta),
    Snapshot(OrderBookSnapshot),
    Heartbeat { symbol: SymbolId, timestamp_ns: i64 },
}

/// Compile-time size assertions for cache optimization
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_struct_sizes() {
        assert_eq!(mem::size_of::<Price>(), 8);
        assert_eq!(mem::size_of::<Quantity>(), 8);
        assert_eq!(mem::size_of::<SymbolId>(), 16);
        assert_eq!(mem::size_of::<Level>(), 24);
        assert_eq!(mem::size_of::<Ticker>(), 96); // Should fit in 2 cache lines
        assert_eq!(mem::size_of::<Trade>(), 64);  // Exactly one cache line
    }

    #[test]
    fn test_price_conversion() {
        let price = Price::from_f64(50000.12345678);
        assert_eq!(price.to_f64(), 50000.12345678);
    }

    #[test]
    fn test_symbol_id() {
        let sym = SymbolId::from_str("BTCUSDT");
        assert_eq!(sym.as_str(), "BTCUSDT");
    }
}
