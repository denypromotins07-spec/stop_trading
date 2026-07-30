//! Shadow Mode Engine
//! 
//! Parallel Shadow Mode engine that simulates execution against the live L2 book
//! without broadcasting to the exchange. Allows testing new ML weights from SOUL.md
//! in real-time market conditions without risking actual capital.
//! Memory-efficient design shares read-only market data ring buffers.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};
use crate::gateway::venue::VenueId;
use crate::orderbook::book::OrderBook;

/// Shadow order representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShadowOrder {
    pub order_id: u64,
    pub symbol: [u8; 12],
    pub side: ShadowSide,
    pub price_ticks: i64,
    pub quantity: u64,
    pub remaining_qty: u64,
    pub timestamp_ns: u64,
    pub strategy_id: u32,
    pub is_live: bool,  // false = shadow only
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum ShadowSide {
    Buy = 0,
    Sell = 1,
}

/// Simulated fill from shadow execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShadowFill {
    pub fill_id: u64,
    pub order_id: u64,
    pub symbol: [u8; 12],
    pub side: ShadowSide,
    pub price_ticks: i64,
    pub quantity: u64,
    pub timestamp_ns: u64,
    pub simulated_slippage_bps: f64,
    pub latency_ns: u64,
}

/// Shadow PnL calculation result
#[derive(Debug, Clone, Default)]
pub struct ShadowPnL {
    pub realized_pnl: f64,
    pub unrealized_pnl: f64,
    pub total_pnl: f64,
    pub trades_count: u64,
    pub win_rate: f64,
    pub avg_trade_pnl: f64,
}

/// Per-symbol shadow state
struct SymbolShadowState {
    symbol: [u8; 12],
    /// Local shadow order book (simulated positions)
    shadow_book: OrderBook,
    /// Pending shadow orders
    pending_orders: HashMap<u64, ShadowOrder>,
    /// Historical fills for PnL calculation
    fills: Vec<ShadowFill>,
    /// Position tracking
    position: i64,  // signed quantity
    /// Average entry price
    avg_entry_price: i64,
}

impl SymbolShadowState {
    fn new(symbol: [u8; 12]) -> Self {
        Self {
            symbol,
            shadow_book: OrderBook::new(symbol),
            pending_orders: HashMap::new(),
            fills: Vec::with_capacity(1000),
            position: 0,
            avg_entry_price: 0,
        }
    }

    fn add_order(&mut self, order: ShadowOrder) {
        self.pending_orders.insert(order.order_id, order);
    }

    fn remove_order(&mut self, order_id: u64) -> Option<ShadowOrder> {
        self.pending_orders.remove(&order_id)
    }

    fn record_fill(&mut self, fill: ShadowFill) {
        // Update position
        match fill.side {
            ShadowSide::Buy => {
                if self.position >= 0 {
                    // Adding to long or opening long
                    let total_value = (self.avg_entry_price * self.position as u64) as i64 
                        + (fill.price_ticks * fill.quantity as i64);
                    self.position += fill.quantity as i64;
                    if self.position > 0 {
                        self.avg_entry_price = total_value / self.position;
                    }
                } else {
                    // Covering short
                    self.position += fill.quantity as i64;
                    if self.position > 0 {
                        self.avg_entry_price = fill.price_ticks;
                    }
                }
            }
            ShadowSide::Sell => {
                if self.position <= 0 {
                    // Adding to short or opening short
                    let total_value = (self.avg_entry_price * (-self.position) as u64) as i64 
                        + (fill.price_ticks * fill.quantity as i64);
                    self.position -= fill.quantity as i64;
                    if self.position < 0 {
                        self.avg_entry_price = total_value / (-self.position);
                    }
                } else {
                    // Reducing long
                    self.position -= fill.quantity as i64;
                    if self.position < 0 {
                        self.avg_entry_price = fill.price_ticks;
                    }
                }
            }
        }

        // Limit fill history for memory efficiency
        if self.fills.len() >= 10000 {
            self.fills.drain(0..5000);
        }
        self.fills.push(fill);
    }

    fn calculate_pnl(&self, current_price: i64) -> ShadowPnL {
        let mut realized_pnl = 0.0f64;
        let mut winning_trades = 0u64;
        
        // Calculate realized PnL from closed positions
        // Simplified: track round-trip profits
        let mut long_opens = 0i64;
        let mut long_closes = 0i64;
        let mut short_opens = 0i64;
        let mut short_closes = 0i64;

        for fill in &self.fills {
            match fill.side {
                ShadowSide::Buy => {
                    if long_opens == 0 {
                        long_opens = fill.price_ticks * fill.quantity as i64;
                    } else {
                        // Closing short
                        short_closes += fill.price_ticks * fill.quantity as i64;
                    }
                }
                ShadowSide::Sell => {
                    if short_opens == 0 {
                        short_opens = fill.price_ticks * fill.quantity as i64;
                    } else {
                        // Closing long
                        long_closes += fill.price_ticks * fill.quantity as i64;
                    }
                }
            }
        }

        // Realized PnL from completed round trips
        if long_closes > 0 && long_opens > 0 {
            realized_pnl += (long_closes - long_opens) as f64;
        }
        if short_opens > 0 && short_closes > 0 {
            realized_pnl += (short_opens - short_closes) as f64;
        }

        // Unrealized PnL from open position
        let unrealized_pnl = if self.position != 0 {
            (current_price - self.avg_entry_price) as f64 * self.position as f64
        } else {
            0.0
        };

        let total_pnl = realized_pnl + unrealized_pnl;
        let trades_count = self.fills.len() as u64 / 2;  // Approximate round trips
        let win_rate = if trades_count > 0 {
            winning_trades as f64 / trades_count as f64
        } else {
            0.0
        };

        ShadowPnL {
            realized_pnl,
            unrealized_pnl,
            total_pnl,
            trades_count,
            win_rate,
            avg_trade_pnl: if trades_count > 0 { total_pnl / trades_count as f64 } else { 0.0 },
        }
    }
}

/// Main Shadow Engine
pub struct ShadowEngine {
    venue_id: VenueId,
    /// Per-symbol shadow states
    symbol_states: parking_lot::RwLock<HashMap<[u8; 12], SymbolShadowState>>,
    /// Reference to live market data (shared, read-only)
    live_book_ref: Arc<parking_lot::RwLock<OrderBook>>,
    /// Shadow mode enabled
    enabled: AtomicBool,
    /// Fill counter
    fill_counter: AtomicU64,
    /// Order counter
    order_counter: AtomicU64,
    /// Simulation parameters
    simulation_slippage_bps: f64,
    simulation_latency_ns: u64,
}

impl ShadowEngine {
    pub fn new(venue_id: VenueId, live_book: Arc<parking_lot::RwLock<OrderBook>>) -> Self {
        Self {
            venue_id,
            symbol_states: parking_lot::RwLock::new(HashMap::new()),
            live_book_ref: live_book,
            enabled: AtomicBool::new(true),
            fill_counter: AtomicU64::new(1),
            order_counter: AtomicU64::new(1),
            simulation_slippage_bps: 2.0,  // 2 bps default slippage
            simulation_latency_ns: 100_000,  // 100 microseconds default
        }
    }

    /// Get or create symbol state
    fn get_or_create_state(&self, symbol: &[u8; 12]) -> parking_lot::RwLockWriteGuard<SymbolShadowState> {
        let mut states = self.symbol_states.write();
        states.entry(*symbol).or_insert_with(|| SymbolShadowState::new(*symbol));
        
        parking_lot::RwLockWriteGuard::map(states, |s| {
            s.entry(*symbol).or_insert_with(|| SymbolShadowState::new(*symbol))
        })
    }

    /// Submit shadow order for simulation
    pub fn submit_shadow_order(&self, order: ShadowOrder) -> Result<u64, &'static str> {
        if !self.enabled.load(Ordering::Acquire) {
            return Err("Shadow engine disabled");
        }

        let order_id = self.order_counter.fetch_add(1, Ordering::Relaxed);
        let mut shadow_order = order;
        shadow_order.order_id = order_id;

        let mut state = self.get_or_create_state(&shadow_order.symbol);
        state.add_order(shadow_order.clone());

        // Try to match against live book (read-only reference)
        if let Some(fill) = self.try_match_order(&shadow_order) {
            state.record_fill(fill);
        }

        Ok(order_id)
    }

    /// Cancel shadow order
    pub fn cancel_shadow_order(&self, symbol: &[u8; 12], order_id: u64) -> Option<ShadowOrder> {
        let mut state = self.get_or_create_state(symbol);
        state.remove_order(order_id)
    }

    /// Try to match shadow order against live book
    fn try_match_order(&self, order: &ShadowOrder) -> Option<ShadowFill> {
        // Read from live book (shared, zero-copy)
        let live_book = self.live_book_ref.read();
        
        // Check if order would execute at current market
        let (bid, ask) = live_book.get_best_bid_ask()?;
        
        let would_execute = match order.side {
            ShadowSide::Buy => order.price_ticks >= ask,
            ShadowSide::Sell => order.price_ticks <= bid,
        };

        if !would_execute {
            return None;
        }

        // Simulate fill with slippage
        let slippage = (order.price_ticks as f64 * self.simulation_slippage_bps / 10000.0) as i64;
        let fill_price = match order.side {
            ShadowSide::Buy => order.price_ticks + slippage.max(1),
            ShadowSide::Sell => order.price_ticks - slippage.max(1),
        };

        let fill_id = self.fill_counter.fetch_add(1, Ordering::Relaxed);
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        Some(ShadowFill {
            fill_id,
            order_id: order.order_id,
            symbol: order.symbol,
            side: order.side,
            price_ticks: fill_price,
            quantity: order.quantity.min(100),  // Simulated partial fill cap
            timestamp_ns: now_ns,
            simulated_slippage_bps: self.simulation_slippage_bps,
            latency_ns: self.simulation_latency_ns,
        })
    }

    /// Get shadow PnL for symbol
    pub fn get_shadow_pnl(&self, symbol: &[u8; 12]) -> ShadowPnL {
        let state = self.get_or_create_state(symbol);
        
        // Get current price from live book
        let live_book = self.live_book_ref.read();
        let current_price = live_book.get_mid_price().unwrap_or(0);
        
        state.calculate_pnl(current_price)
    }

    /// Get all shadow positions
    pub fn get_all_positions(&self) -> Vec<([u8; 12], i64, i64)> {
        let states = self.symbol_states.read();
        states.iter().map(|(sym, state)| {
            (*sym, state.position, state.avg_entry_price)
        }).collect()
    }

    /// Enable/disable shadow mode
    #[inline]
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Release);
    }

    /// Configure simulation parameters
    pub fn configure_simulation(&mut self, slippage_bps: f64, latency_ns: u64) {
        self.simulation_slippage_bps = slippage_bps;
        self.simulation_latency_ns = latency_ns;
    }

    /// Get statistics
    pub fn get_stats(&self) -> ShadowStats {
        let states = self.symbol_states.read();
        let total_fills: usize = states.values().map(|s| s.fills.len()).sum();
        let total_orders: usize = states.values().map(|s| s.pending_orders.len()).sum();

        ShadowStats {
            symbols_tracked: states.len(),
            total_fills: total_fills as u64,
            pending_orders: total_orders as u64,
            enabled: self.enabled.load(Ordering::Acquire),
        }
    }

    /// Clear all shadow state
    pub fn clear_all(&self) {
        self.symbol_states.write().clear();
    }
}

/// Shadow engine statistics
#[derive(Debug, Clone, Default)]
pub struct ShadowStats {
    pub symbols_tracked: usize,
    pub total_fills: u64,
    pub pending_orders: u64,
    pub enabled: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shadow_engine_creation() {
        let symbol = *b"AAPL        ";
        let live_book = Arc::new(parking_lot::RwLock::new(OrderBook::new(symbol)));
        let engine = ShadowEngine::new(VenueId::Nasdaq, live_book);
        
        assert!(engine.enabled.load(Ordering::Acquire));
        assert_eq!(engine.get_stats().symbols_tracked, 0);
    }

    #[test]
    fn test_shadow_order_submission() {
        let symbol = *b"AAPL        ";
        let live_book = Arc::new(parking_lot::RwLock::new(OrderBook::new(symbol)));
        let engine = ShadowEngine::new(VenueId::Nasdaq, live_book);

        let order = ShadowOrder {
            order_id: 0,
            symbol,
            side: ShadowSide::Buy,
            price_ticks: 15000,
            quantity: 100,
            remaining_qty: 100,
            timestamp_ns: 0,
            strategy_id: 1,
            is_live: false,
        };

        let result = engine.submit_shadow_order(order);
        assert!(result.is_ok());
    }
}
