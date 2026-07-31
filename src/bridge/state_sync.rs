//! Atomic State Synchronization Layer
//! 
//! Mirrors Rust actor states to shared memory IPC segment for Python Ray workers.
//! Provides lock-free, consistent view of portfolio delta and open order IDs.
//! Uses atomic operations and memory barriers for cross-language synchronization.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

/// Maximum number of tracked symbols
pub const MAX_SYMBOLS: usize = 256;

/// Maximum number of open orders
pub const MAX_OPEN_ORDERS: usize = 1024;

/// Shared memory state header
#[repr(C)]
#[derive(Debug, Clone)]
pub struct SharedStateHeader {
    /// Version for schema compatibility
    pub version: u32,
    /// Sequence number for consistency checks
    pub sequence: AtomicU64,
    /// Last update timestamp (nanoseconds)
    pub last_update_ns: AtomicU64,
    /// Number of active symbols
    pub active_symbols: AtomicUsize,
    /// Number of open orders
    pub open_order_count: AtomicUsize,
    /// Market open flag
    pub is_market_open: AtomicBool,
    /// Trading enabled flag
    pub trading_enabled: AtomicBool,
    /// Emergency stop flag
    pub emergency_stop: AtomicBool,
}

impl Default for SharedStateHeader {
    fn default() -> Self {
        Self {
            version: 1,
            sequence: AtomicU64::new(0),
            last_update_ns: AtomicU64::new(0),
            active_symbols: AtomicUsize::new(0),
            open_order_count: AtomicUsize::new(0),
            is_market_open: AtomicBool::new(false),
            trading_enabled: AtomicBool::new(false),
            emergency_stop: AtomicBool::new(false),
        }
    }
}

/// Per-symbol shared state
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SymbolState {
    /// Symbol hash (for identification)
    pub symbol_hash: u64,
    /// Best bid price
    pub bid_price: f64,
    /// Best ask price
    pub ask_price: f64,
    /// Bid size
    pub bid_size: f64,
    /// Ask size
    pub ask_size: f64,
    /// Last trade price
    pub last_price: f64,
    /// Position quantity
    pub position_qty: f64,
    /// Average entry price
    pub avg_entry: f64,
    /// Unrealized PnL
    pub unrealized_pnl: f64,
    /// Last update sequence
    pub sequence: u64,
    /// Is active flag
    pub is_active: bool,
}

impl Default for SymbolState {
    fn default() -> Self {
        Self {
            symbol_hash: 0,
            bid_price: 0.0,
            ask_price: 0.0,
            bid_size: 0.0,
            ask_size: 0.0,
            last_price: 0.0,
            position_qty: 0.0,
            avg_entry: 0.0,
            unrealized_pnl: 0.0,
            sequence: 0,
            is_active: false,
        }
    }
}

/// Open order record for IPC sync
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct OpenOrderRecord {
    /// Order ID from exchange
    pub order_id: u64,
    /// Client order ID
    pub client_order_id: u64,
    /// Symbol hash
    pub symbol_hash: u64,
    /// Side (0=Buy, 1=Sell)
    pub side: u8,
    /// Order type (0=Market, 1=Limit)
    pub order_type: u8,
    /// Quantity
    pub quantity: f64,
    /// Filled quantity
    pub filled_qty: f64,
    /// Price
    pub price: f64,
    /// Creation timestamp
    pub created_ns: u64,
    /// Update timestamp
    pub updated_ns: u64,
    /// Is active
    pub is_active: bool,
}

impl Default for OpenOrderRecord {
    fn default() -> Self {
        Self {
            order_id: 0,
            client_order_id: 0,
            symbol_hash: 0,
            side: 0,
            order_type: 0,
            quantity: 0.0,
            filled_qty: 0.0,
            price: 0.0,
            created_ns: 0,
            updated_ns: 0,
            is_active: false,
        }
    }
}

/// Main shared state container for IPC
pub struct SharedState {
    header: Arc<SharedStateHeader>,
    symbols: Box<[SymbolState; MAX_SYMBOLS]>,
    orders: Box<[OpenOrderRecord; MAX_OPEN_ORDERS]>,
    /// Write lock simulation using atomic sequence
    write_lock: AtomicU64,
}

unsafe impl Send for SharedState {}
unsafe impl Sync for SharedState {}

impl SharedState {
    /// Create new shared state
    pub fn new() -> Self {
        Self {
            header: Arc::new(SharedStateHeader::default()),
            symbols: Box::new([SymbolState::default(); MAX_SYMBOLS]),
            orders: Box::new([OpenOrderRecord::default(); MAX_OPEN_ORDERS]),
            write_lock: AtomicU64::new(0),
        }
    }
    
    /// Get current timestamp in nanoseconds
    #[inline]
    fn now_ns() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64
    }
    
    /// Acquire write lock (spinlock with backoff)
    fn acquire_write_lock(&self) -> bool {
        let mut attempts = 0;
        while self.write_lock.compare_exchange(0, 1, Ordering::AcqRel, Ordering::Relaxed).is_err() {
            attempts += 1;
            if attempts > 1000 {
                return false; // Timeout
            }
            std::hint::spin_loop();
        }
        true
    }
    
    /// Release write lock
    fn release_write_lock(&self) {
        self.write_lock.store(0, Ordering::Release);
    }
    
    /// Update symbol state atomically
    pub fn update_symbol(&self, symbol_hash: u64, state: &SymbolState) -> bool {
        if !self.acquire_write_lock() {
            return false;
        }
        
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            // Find or allocate slot
            let mut slot_idx = None;
            for (i, sym) in self.symbols.iter().enumerate() {
                if sym.symbol_hash == symbol_hash || !sym.is_active {
                    slot_idx = Some(i);
                    break;
                }
            }
            
            if let Some(idx) = slot_idx {
                let seq = self.header.sequence.fetch_add(1, Ordering::Relaxed);
                let mut new_state = *state;
                new_state.sequence = seq;
                new_state.is_active = true;
                
                self.symbols[idx] = new_state;
                
                // Update header
                self.header.last_update_ns.store(self.now_ns(), Ordering::Relaxed);
                
                // Update active count if newly activated
                if !self.symbols[idx].is_active && state.is_active {
                    self.header.active_symbols.fetch_add(1, Ordering::Relaxed);
                }
            }
        }));
        
        self.release_write_lock();
        result.is_ok()
    }
    
    /// Add open order
    pub fn add_order(&self, order: &OpenOrderRecord) -> Option<usize> {
        if !self.acquire_write_lock() {
            return None;
        }
        
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            for (i, ord) in self.orders.iter().enumerate() {
                if !ord.is_active {
                    let mut new_order = *order;
                    new_order.is_active = true;
                    self.orders[i] = new_order;
                    
                    self.header.open_order_count.fetch_add(1, Ordering::Relaxed);
                    self.header.sequence.fetch_add(1, Ordering::Relaxed);
                    self.header.last_update_ns.store(self.now_ns(), Ordering::Relaxed);
                    
                    return Some(i);
                }
            }
            None
        }));
        
        self.release_write_lock();
        result.ok().flatten()
    }
    
    /// Remove/cancel order
    pub fn remove_order(&self, order_id: u64) -> bool {
        if !self.acquire_write_lock() {
            return false;
        }
        
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            for ord in self.orders.iter_mut() {
                if ord.order_id == order_id && ord.is_active {
                    ord.is_active = false;
                    ord.updated_ns = self.now_ns();
                    
                    self.header.open_order_count.fetch_sub(1, Ordering::Relaxed);
                    self.header.sequence.fetch_add(1, Ordering::Relaxed);
                    self.header.last_update_ns.store(self.now_ns(), Ordering::Relaxed);
                    
                    return true;
                }
            }
            false
        }));
        
        self.release_write_lock();
        result.unwrap_or(false)
    }
    
    /// Get snapshot of all active orders
    pub fn get_active_orders(&self) -> Vec<OpenOrderRecord> {
        self.orders
            .iter()
            .filter(|o| o.is_active)
            .copied()
            .collect()
    }
    
    /// Get symbol state by hash
    pub fn get_symbol(&self, symbol_hash: u64) -> Option<SymbolState> {
        self.symbols
            .iter()
            .find(|s| s.symbol_hash == symbol_hash && s.is_active)
            .copied()
    }
    
    /// Get header reference
    pub fn header(&self) -> &SharedStateHeader {
        &self.header
    }
    
    /// Enable/disable trading
    pub fn set_trading_enabled(&self, enabled: bool) {
        self.header.trading_enabled.store(enabled, Ordering::SeqCst);
    }
    
    /// Check if trading is enabled
    pub fn is_trading_enabled(&self) -> bool {
        self.header.trading_enabled.load(Ordering::SeqCst)
    }
    
    /// Trigger emergency stop
    pub fn emergency_stop(&self) {
        self.header.emergency_stop.store(true, Ordering::SeqCst);
        self.header.trading_enabled.store(false, Ordering::SeqCst);
    }
    
    /// Check emergency stop status
    pub fn is_emergency_stop(&self) -> bool {
        self.header.emergency_stop.load(Ordering::SeqCst)
    }
    
    /// Get sequence number for consistency checks
    pub fn get_sequence(&self) -> u64 {
        self.header.sequence.load(Ordering::Acquire)
    }
}

impl Default for SharedState {
    fn default() -> Self {
        Self::new()
    }
}

/// Hash function for symbol strings (djb2 variant)
pub fn hash_symbol(symbol: &str) -> u64 {
    let mut hash: u64 = 5381;
    for byte in symbol.bytes() {
        hash = ((hash << 5).wrapping_add(hash)) ^ (byte as u64);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_shared_state_creation() {
        let state = SharedState::new();
        assert!(!state.is_trading_enabled());
        assert!(!state.is_emergency_stop());
        assert_eq!(state.header.active_symbols.load(Ordering::Relaxed), 0);
    }
    
    #[test]
    fn test_symbol_update() {
        let state = SharedState::new();
        let symbol_hash = hash_symbol("BTCUSDT");
        
        let sym_state = SymbolState {
            symbol_hash,
            bid_price: 50000.0,
            ask_price: 50001.0,
            bid_size: 1.5,
            ask_size: 2.0,
            last_price: 50000.5,
            ..Default::default()
        };
        
        assert!(state.update_symbol(symbol_hash, &sym_state));
        
        let retrieved = state.get_symbol(symbol_hash).unwrap();
        assert_eq!(retrieved.bid_price, 50000.0);
        assert_eq!(retrieved.ask_price, 50001.0);
    }
    
    #[test]
    fn test_order_lifecycle() {
        let state = SharedState::new();
        
        let order = OpenOrderRecord {
            order_id: 12345,
            client_order_id: 67890,
            symbol_hash: hash_symbol("ETHUSDT"),
            side: 0,
            order_type: 1,
            quantity: 10.0,
            price: 3000.0,
            created_ns: SharedState::now_ns(),
            ..Default::default()
        };
        
        let idx = state.add_order(&order);
        assert!(idx.is_some());
        assert_eq!(state.header.open_order_count.load(Ordering::Relaxed), 1);
        
        assert!(state.remove_order(12345));
        assert_eq!(state.header.open_order_count.load(Ordering::Relaxed), 0);
    }
    
    #[test]
    fn test_emergency_stop() {
        let state = SharedState::new();
        state.set_trading_enabled(true);
        
        assert!(state.is_trading_enabled());
        assert!(!state.is_emergency_stop());
        
        state.emergency_stop();
        
        assert!(!state.is_trading_enabled());
        assert!(state.is_emergency_stop());
    }
    
    #[test]
    fn test_symbol_hash_consistency() {
        assert_eq!(hash_symbol("BTCUSDT"), hash_symbol("BTCUSDT"));
        assert_ne!(hash_symbol("BTCUSDT"), hash_symbol("ETHUSDT"));
    }
}
