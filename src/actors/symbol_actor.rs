//! Per-Symbol Actor State Machine
//! 
//! Core state machine for a single trading pair (e.g., BTCUSDT, SOLUSDT).
//! Encapsulates local order book, risk limits, and alpha signals into a dedicated actor.

use std::sync::atomic::{AtomicU64, AtomicI64, AtomicBool, Ordering};
use std::sync::Arc;

/// Cache line size
const CACHE_LINE_SIZE: usize = 64;

/// Padded atomic u64
#[repr(C)]
#[derive(Debug)]
pub struct PaddedAtomicU64 {
    _pad1: [u8; CACHE_LINE_SIZE - 8],
    value: AtomicU64,
    _pad2: [u8; CACHE_LINE_SIZE],
}

impl PaddedAtomicU64 {
    pub fn new(initial: u64) -> Self {
        Self {
            _pad1: [0u8; CACHE_LINE_SIZE - 8],
            value: AtomicU64::new(initial),
            _pad2: [0u8; CACHE_LINE_SIZE],
        }
    }

    #[inline]
    pub fn load(&self, ordering: Ordering) -> u64 {
        self.value.load(ordering)
    }

    #[inline]
    pub fn store(&self, val: u64, ordering: Ordering) {
        self.value.store(val, ordering);
    }

    #[inline]
    pub fn fetch_add(&self, val: u64, ordering: Ordering) -> u64 {
        self.value.fetch_add(val, ordering)
    }
}

/// Padded atomic i64
#[repr(C)]
#[derive(Debug)]
pub struct PaddedAtomicI64 {
    _pad1: [u8; CACHE_LINE_SIZE - 8],
    value: AtomicI64,
    _pad2: [u8; CACHE_LINE_SIZE],
}

impl PaddedAtomicI64 {
    pub fn new(initial: i64) -> Self {
        Self {
            _pad1: [0u8; CACHE_LINE_SIZE - 8],
            value: AtomicI64::new(initial),
            _pad2: [0u8; CACHE_LINE_SIZE],
        }
    }

    #[inline]
    pub fn load(&self, ordering: Ordering) -> i64 {
        self.value.load(ordering)
    }

    #[inline]
    pub fn store(&self, val: i64, ordering: Ordering) {
        self.value.store(val, ordering);
    }

    #[inline]
    pub fn fetch_add(&self, val: i64, ordering: Ordering) -> i64 {
        self.value.fetch_add(val, ordering)
    }
}

/// Order side
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub enum Side {
    Buy,
    Sell,
}

/// Order type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub enum OrderType {
    Limit,
    Market,
    IOC,
    FOK,
}

/// Order status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub enum OrderStatus {
    Pending,
    New,
    PartiallyFilled,
    Filled,
    Cancelled,
    Rejected,
}

/// Local order representation
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct LocalOrder {
    pub order_id: u64,
    pub symbol_hash: u64,
    pub side: Side,
    pub order_type: OrderType,
    pub price: u64,
    pub quantity: u64,
    pub filled_quantity: u64,
    pub status: OrderStatus,
    pub timestamp_ns: u64,
}

impl LocalOrder {
    pub fn new(
        order_id: u64,
        symbol_hash: u64,
        side: Side,
        order_type: OrderType,
        price: u64,
        quantity: u64,
    ) -> Self {
        Self {
            order_id,
            symbol_hash,
            side,
            order_type,
            price,
            quantity,
            filled_quantity: 0,
            status: OrderStatus::Pending,
            timestamp_ns: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64,
        }
    }

    /// Update fill quantity
    #[inline]
    pub fn update_fill(&mut self, fill_qty: u64) {
        self.filled_quantity += fill_qty;
        if self.filled_quantity >= self.quantity {
            self.status = OrderStatus::Filled;
        } else if self.filled_quantity > 0 {
            self.status = OrderStatus::PartiallyFilled;
        }
    }
}

/// Order book level
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct OrderBookLevel {
    pub price: u64,
    pub quantity: u64,
    pub order_count: u32,
}

/// Local order book snapshot (simplified for lock-free access)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct LocalOrderBook {
    pub symbol_hash: u64,
    pub best_bid: u64,
    pub best_ask: u64,
    pub bid_depth: u64,
    pub ask_depth: u64,
    pub mid_price: u64,
    pub spread_bps: u64,
    pub last_update_ns: u64,
}

impl LocalOrderBook {
    pub fn new(symbol_hash: u64) -> Self {
        Self {
            symbol_hash,
            best_bid: 0,
            best_ask: 0,
            bid_depth: 0,
            ask_depth: 0,
            mid_price: 0,
            spread_bps: 0,
            last_update_ns: 0,
        }
    }

    /// Update from market data
    #[inline]
    pub fn update(&mut self, best_bid: u64, best_ask: u64, bid_depth: u64, ask_depth: u64) {
        self.best_bid = best_bid;
        self.best_ask = best_ask;
        self.bid_depth = bid_depth;
        self.ask_depth = ask_depth;
        self.mid_price = (best_bid + best_ask) / 2;
        
        if self.mid_price > 0 {
            self.spread_bps = ((best_ask - best_bid) * 10_000) / self.mid_price;
        }
        
        self.last_update_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
    }

    /// Get mid price
    #[inline]
    pub fn get_mid_price(&self) -> u64 {
        self.mid_price
    }

    /// Check if book is valid (has quotes on both sides)
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.best_bid > 0 && self.best_ask > 0 && self.best_bid < self.best_ask
    }
}

/// Alpha signal types
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlphaSignalType {
    Momentum,
    MeanReversion,
    Arbitrage,
    MarketMaking,
    None,
}

/// Alpha signal with confidence
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AlphaSignal {
    pub signal_type: AlphaSignalType,
    pub direction: i8, // -1 = sell, 1 = buy, 0 = neutral
    pub strength: u8,  // 0-100 confidence
    pub target_price: u64,
    pub expiry_ns: u64,
    pub timestamp_ns: u64,
}

impl AlphaSignal {
    pub fn new(signal_type: AlphaSignalType, direction: i8, strength: u8, target_price: u64) -> Self {
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        
        // Default 100ms expiry for HFT signals
        let expiry_ns = now_ns + 100_000_000;
        
        Self {
            signal_type,
            direction,
            strength,
            target_price,
            expiry_ns,
            timestamp_ns: now_ns,
        }
    }

    /// Check if signal is still valid
    #[inline]
    pub fn is_valid(&self) -> bool {
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        now_ns < self.expiry_ns
    }
}

/// Symbol actor state
#[repr(C)]
pub struct SymbolActorState {
    /// Symbol hash (unique identifier)
    pub symbol_hash: u64,
    /// Symbol name hash
    pub name_hash: u64,
    /// Local order book
    pub order_book: PaddedAtomicU64, // Pointer to order book in memory pool
    /// Current position
    pub position: PaddedAtomicI64,
    /// Pending order count
    pub pending_orders: PaddedAtomicU64,
    /// Last alpha signal
    pub last_signal: PaddedAtomicU64, // Pointer to signal
    /// Risk limit for this symbol
    pub max_position: PaddedAtomicI64,
    /// Max order size
    pub max_order_size: PaddedAtomicU64,
    /// Trading enabled
    pub trading_enabled: AtomicBool,
    /// Actor active
    pub is_active: AtomicBool,
    /// Messages processed
    pub messages_processed: PaddedAtomicU64,
    /// Last message timestamp
    pub last_message_ns: PaddedAtomicU64,
}

impl SymbolActorState {
    pub fn new(symbol_hash: u64, name_hash: u64, max_position: i64, max_order_size: u64) -> Self {
        Self {
            symbol_hash,
            name_hash,
            order_book: PaddedAtomicU64::new(0),
            position: PaddedAtomicI64::new(0),
            pending_orders: PaddedAtomicU64::new(0),
            last_signal: PaddedAtomicU64::new(0),
            max_position: PaddedAtomicI64::new(max_position),
            max_order_size: PaddedAtomicU64::new(max_order_size),
            trading_enabled: AtomicBool::new(true),
            is_active: AtomicBool::new(true),
            messages_processed: PaddedAtomicU64::new(0),
            last_message_ns: PaddedAtomicU64::new(0),
        }
    }

    /// Update position
    #[inline]
    pub fn update_position(&self, delta: i64) {
        self.position.fetch_add(delta, Ordering::AcqRel);
    }

    /// Get current position
    #[inline]
    pub fn get_position(&self) -> i64 {
        self.position.load(Ordering::Acquire)
    }

    /// Check if position limit would be exceeded
    #[inline]
    pub fn would_exceed_limit(&self, additional: i64) -> bool {
        let current = self.position.load(Ordering::Acquire);
        let new_position = current + additional;
        new_position.abs() > self.max_position.load(Ordering::Acquire)
    }

    /// Update pending order count
    #[inline]
    pub fn update_pending_orders(&self, delta: i64) {
        let current = self.pending_orders.load(Ordering::Acquire);
        if delta >= 0 {
            self.pending_orders.fetch_add(delta as u64, Ordering::AcqRel);
        } else if current >= (-delta) as u64 {
            self.pending_orders.fetch_sub((-delta) as u64, Ordering::AcqRel);
        }
    }

    /// Record message processing
    #[inline]
    pub fn record_message(&self) {
        self.messages_processed.fetch_add(1, Ordering::Relaxed);
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        self.last_message_ns.store(now_ns, Ordering::Release);
    }

    /// Enable trading
    #[inline]
    pub fn enable_trading(&self) {
        self.trading_enabled.store(true, Ordering::Release);
    }

    /// Disable trading
    #[inline]
    pub fn disable_trading(&self) {
        self.trading_enabled.store(false, Ordering::Release);
    }

    /// Check if trading is enabled
    #[inline]
    pub fn is_trading_enabled(&self) -> bool {
        self.trading_enabled.load(Ordering::Acquire)
    }

    /// Deactivate actor
    #[inline]
    pub fn deactivate(&self) {
        self.is_active.store(false, Ordering::Release);
        self.trading_enabled.store(false, Ordering::Release);
    }

    /// Check if actor is active
    #[inline]
    pub fn is_active(&self) -> bool {
        self.is_active.load(Ordering::Acquire)
    }

    /// Get statistics
    #[inline]
    pub fn get_stats(&self) -> SymbolActorStats {
        SymbolActorStats {
            symbol_hash: self.symbol_hash,
            position: self.get_position(),
            pending_orders: self.pending_orders.load(Ordering::Relaxed),
            messages_processed: self.messages_processed.load(Ordering::Relaxed),
            last_message_ns: self.last_message_ns.load(Ordering::Relaxed),
            is_active: self.is_active(),
            trading_enabled: self.is_trading_enabled(),
        }
    }
}

/// Symbol actor statistics
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SymbolActorStats {
    pub symbol_hash: u64,
    pub position: i64,
    pub pending_orders: u64,
    pub messages_processed: u64,
    pub last_message_ns: u64,
    pub is_active: bool,
    pub trading_enabled: bool,
}

/// Message types for symbol actor
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum SymbolMessage {
    MarketData {
        symbol_hash: u64,
        bid: u64,
        ask: u64,
        bid_depth: u64,
        ask_depth: u64,
        timestamp_ns: u64,
    },
    ExecutionReport {
        order_id: u64,
        symbol_hash: u64,
        fill_qty: u64,
        fill_price: u64,
        remaining_qty: u64,
        status: OrderStatus,
    },
    OrderRequest {
        order_id: u64,
        symbol_hash: u64,
        side: Side,
        order_type: OrderType,
        price: u64,
        quantity: u64,
    },
    CancelRequest {
        order_id: u64,
        symbol_hash: u64,
    },
    AlphaSignal {
        signal: AlphaSignal,
    },
    RiskUpdate {
        max_position: i64,
        max_order_size: u64,
    },
    Heartbeat,
    Shutdown,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symbol_actor_state() {
        let state = SymbolActorState::new(12345, 67890, 1_000_000, 100_000);
        
        assert!(state.is_active());
        assert!(state.is_trading_enabled());
        assert_eq!(state.get_position(), 0);

        state.update_position(500_000);
        assert_eq!(state.get_position(), 500_000);
        assert!(!state.would_exceed_limit(400_000));
        assert!(state.would_exceed_limit(600_000));
    }

    #[test]
    fn test_order_book() {
        let mut book = LocalOrderBook::new(12345);
        book.update(49_990_000, 50_010_000, 1_000_000, 1_000_000);
        
        assert!(book.is_valid());
        assert_eq!(book.get_mid_price(), 50_000_000);
        assert_eq!(book.spread_bps, 4); // 4 bps spread
    }

    #[test]
    fn test_alpha_signal() {
        let signal = AlphaSignal::new(AlphaSignalType::Momentum, 1, 80, 51_000_000);
        
        assert_eq!(signal.direction, 1);
        assert_eq!(signal.strength, 80);
        assert!(signal.is_valid());
    }
}
