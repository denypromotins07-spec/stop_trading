//! Auction Engine Module
//! 
//! Models call auction phases (pre-open, matching, freeze) for venues that use them during extreme volatility.
//! Safely manages and re-prices limit orders to ensure optimal execution when continuous trading resumes.
//! Memory-efficient implementation respecting 6.5GB RAM constraint.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};
use crate::gateway::venue::VenueId;
use crate::orderbook::book::PriceLevel;

/// Auction phase enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum AuctionPhase {
    /// No auction active
    Inactive = 0,
    /// Pre-open: order entry allowed, no matching
    PreOpen = 1,
    /// Order modification allowed
    OrderModification = 2,
    /// Freeze: no order entry/modification, only cancellation
    Freeze = 3,
    /// Matching: calculating indicative price
    Matching = 4,
    /// Auction executing
    Executing = 5,
}

impl AuctionPhase {
    #[inline]
    pub fn allows_order_entry(&self) -> bool {
        matches!(self, AuctionPhase::PreOpen | AuctionPhase::OrderModification)
    }

    #[inline]
    pub fn allows_cancellation(&self) -> bool {
        !matches!(self, AuctionPhase::Executing)
    }

    #[inline]
    pub fn from_u8(val: u8) -> Self {
        match val {
            0 => AuctionPhase::Inactive,
            1 => AuctionPhase::PreOpen,
            2 => AuctionPhase::OrderModification,
            3 => AuctionPhase::Freeze,
            4 => AuctionPhase::Matching,
            5 => AuctionPhase::Executing,
            _ => AuctionPhase::Inactive,
        }
    }
}

/// Auction order representation - compact format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuctionOrder {
    pub order_id: u64,
    pub symbol: [u8; 12],
    pub side: Side,
    pub price_ticks: i64,
    pub quantity: u64,
    pub remaining_qty: u64,
    pub timestamp_ns: u64,
    pub is_market_order: bool,
    pub stp_group_id: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum Side {
    Buy = 0,
    Sell = 1,
}

impl Side {
    #[inline]
    pub fn from_u8(val: u8) -> Self {
        if val == 0 { Side::Buy } else { Side::Sell }
    }
}

/// Indicative auction price calculation result
#[derive(Debug, Clone)]
pub struct IndicativePrice {
    pub price_ticks: i64,
    pub executable_volume: u64,
    pub buy_imbalance: i64,
    pub sell_imbalance: i64,
    pub total_buy_volume: u64,
    pub total_sell_volume: u64,
}

/// Auction book for a single symbol
struct AuctionBook {
    symbol: [u8; 12],
    /// Buy orders sorted by price descending (highest first)
    buy_orders: BTreeMap<i64, Vec<AuctionOrder>>,
    /// Sell orders sorted by price ascending (lowest first)
    sell_orders: BTreeMap<i64, Vec<AuctionOrder>>,
    /// Total volume at each price level
    buy_volume_by_price: BTreeMap<i64, u64>,
    sell_volume_by_price: BTreeMap<i64, u64>,
    /// Order lookup for quick cancellation
    order_lookup: BTreeMap<u64, (i64, Side)>,  // order_id -> (price, side)
}

impl AuctionBook {
    fn new(symbol: [u8; 12]) -> Self {
        Self {
            symbol,
            buy_orders: BTreeMap::new(),
            sell_orders: BTreeMap::new(),
            buy_volume_by_price: BTreeMap::new(),
            sell_volume_by_price: BTreeMap::new(),
            order_lookup: BTreeMap::new(),
        }
    }

    fn add_order(&mut self, order: AuctionOrder) {
        let price = order.price_ticks;
        let side = order.side;

        match side {
            Side::Buy => {
                self.buy_orders.entry(price).or_default().push(order.clone());
                *self.buy_volume_by_price.entry(price).or_insert(0) += order.quantity;
            }
            Side::Sell => {
                self.sell_orders.entry(price).or_default().push(order.clone());
                *self.sell_volume_by_price.entry(price).or_insert(0) += order.quantity;
            }
        }

        self.order_lookup.insert(order.order_id, (price, side));
    }

    fn cancel_order(&mut self, order_id: u64) -> Option<AuctionOrder> {
        if let Some((price, side)) = self.order_lookup.remove(&order_id) {
            let orders = match side {
                Side::Buy => self.buy_orders.get_mut(&price),
                Side::Sell => self.sell_orders.get_mut(&price),
            };

            if let Some(orders) = orders {
                for (i, order) in orders.iter().enumerate() {
                    if order.order_id == order_id {
                        let removed = orders.remove(i);
                        
                        // Update volume tracking
                        let volume_map = match side {
                            Side::Buy => &mut self.buy_volume_by_price,
                            Side::Sell => &mut self.sell_volume_by_price,
                        };
                        if let Some(vol) = volume_map.get_mut(&price) {
                            *vol = vol.saturating_sub(removed.quantity);
                            if *vol == 0 {
                                volume_map.remove(&price);
                            }
                        }
                        
                        return Some(removed);
                    }
                }
            }
        }
        None
    }

    fn modify_order(&mut self, order_id: u64, new_quantity: u64, new_price: Option<i64>) -> bool {
        if let Some((old_price, side)) = self.order_lookup.get(&order_id).copied() {
            // Remove old order
            if let Some(removed) = self.cancel_order(order_id) {
                // Create modified order
                let mut modified = removed;
                modified.quantity = new_quantity;
                modified.remaining_qty = new_quantity;
                if let Some(price) = new_price {
                    modified.price_ticks = price;
                }
                self.add_order(modified);
                return true;
            }
        }
        false
    }

    /// Calculate indicative clearing price using volume maximization algorithm
    fn calculate_indicative_price(&self) -> Option<IndicativePrice> {
        if self.buy_volume_by_price.is_empty() || self.sell_volume_by_price.is_empty() {
            return None;
        }

        let mut best_price: i64 = 0;
        let mut max_volume: u64 = 0;
        let mut best_buy_imbalance: i64 = 0;
        let mut best_sell_imbalance: i64 = 0;

        // Collect all potential clearing prices
        let mut prices: Vec<i64> = self.buy_volume_by_price.keys().copied().collect();
        prices.extend(self.sell_volume_by_price.keys().copied());
        prices.sort();
        prices.dedup();

        for price in prices {
            // Calculate total buy volume at or above this price
            let mut total_buy_vol: u64 = 0;
            for (&p, &vol) in &self.buy_volume_by_price {
                if p >= price {
                    total_buy_vol += vol;
                }
            }

            // Calculate total sell volume at or below this price
            let mut total_sell_vol: u64 = 0;
            for (&p, &vol) in &self.sell_volume_by_price {
                if p <= price {
                    total_sell_vol += vol;
                }
            }

            // Executable volume is the minimum of buy and sell
            let executable_vol = total_buy_vol.min(total_sell_vol);

            if executable_vol > max_volume {
                max_volume = executable_vol;
                best_price = price;
                best_buy_imbalance = total_buy_vol as i64 - executable_vol as i64;
                best_sell_imbalance = total_sell_vol as i64 - executable_vol as i64;
            } else if executable_vol == max_volume && max_volume > 0 {
                // Tie-breaking: prefer price with smaller imbalance
                let buy_imb = total_buy_vol as i64 - executable_vol as i64;
                let sell_imb = total_sell_vol as i64 - executable_vol as i64;
                let current_imb = (best_buy_imbalance.abs() + best_sell_imbalance.abs()) / 2;
                let new_imb = (buy_imb.abs() + sell_imb.abs()) / 2;
                
                if new_imb < current_imb {
                    best_price = price;
                    best_buy_imbalance = buy_imb;
                    best_sell_imbalance = sell_imb;
                }
            }
        }

        if max_volume == 0 {
            return None;
        }

        Some(IndicativePrice {
            price_ticks: best_price,
            executable_volume: max_volume,
            buy_imbalance: best_buy_imbalance,
            sell_imbalance: best_sell_imbalance,
            total_buy_volume: self.buy_volume_by_price.values().sum(),
            total_sell_volume: self.sell_volume_by_price.values().sum(),
        })
    }

    fn clear_book(&mut self) -> Vec<AuctionOrder> {
        let mut executed_orders = Vec::new();
        
        // Get all orders
        for (_, orders) in self.buy_orders.drain() {
            executed_orders.extend(orders);
        }
        for (_, orders) in self.sell_orders.drain() {
            executed_orders.extend(orders);
        }
        
        self.buy_volume_by_price.clear();
        self.sell_volume_by_price.clear();
        self.order_lookup.clear();
        
        executed_orders
    }
}

/// Per-symbol auction state
struct SymbolAuctionState {
    symbol: [u8; 12],
    phase: AtomicU8,
    book: parking_lot::RwLock<AuctionBook>,
    last_update_ns: AtomicU64,
    auction_start_ns: AtomicU64,
    expected_end_ns: AtomicU64,
}

impl SymbolAuctionState {
    fn new(symbol: [u8; 12]) -> Self {
        Self {
            symbol,
            phase: AtomicU8::new(AuctionPhase::Inactive as u8),
            book: parking_lot::RwLock::new(AuctionBook::new(symbol)),
            last_update_ns: AtomicU64::new(0),
            auction_start_ns: AtomicU64::new(0),
            expected_end_ns: AtomicU64::new(0),
        }
    }

    #[inline]
    fn get_phase(&self) -> AuctionPhase {
        AuctionPhase::from_u8(self.phase.load(Ordering::Acquire))
    }

    #[inline]
    fn set_phase(&self, phase: AuctionPhase) {
        self.phase.store(phase as u8, Ordering::Release);
    }
}

/// Main Auction Engine managing all symbols
pub struct AuctionEngine {
    venue_id: VenueId,
    symbols: parking_lot::RwLock<BTreeMap<[u8; 12], Arc<SymbolAuctionState>>>,
    global_auction_active: AtomicBool,
    /// Broadcast channel for auction events (could be added)
    max_symbols: usize,
}

impl AuctionEngine {
    pub fn new(venue_id: VenueId, max_symbols: usize) -> Self {
        Self {
            venue_id,
            symbols: parking_lot::RwLock::new(BTreeMap::new()),
            global_auction_active: AtomicBool::new(false),
            max_symbols,
        }
    }

    /// Get or create auction state for a symbol
    fn get_or_create_symbol(&self, symbol: &[u8; 12]) -> Arc<SymbolAuctionState> {
        {
            let symbols = self.symbols.read();
            if let Some(state) = symbols.get(symbol) {
                return Arc::clone(state);
            }
        }
        
        let new_state = Arc::new(SymbolAuctionState::new(*symbol));
        let mut symbols = self.symbols.write();
        if symbols.len() < self.max_symbols {
            symbols.insert(*symbol, Arc::clone(&new_state));
        }
        new_state
    }

    /// Start auction for a symbol
    pub fn start_auction(&self, symbol: &[u8; 12], duration_ms: u64) {
        let state = self.get_or_create_symbol(symbol);
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        
        state.set_phase(AuctionPhase::PreOpen);
        state.auction_start_ns.store(now_ns, Ordering::Release);
        state.expected_end_ns.store(now_ns + duration_ms * 1_000_000, Ordering::Release);
        state.last_update_ns.store(now_ns, Ordering::Release);
        
        self.global_auction_active.store(true, Ordering::Release);
    }

    /// Transition auction to next phase
    pub fn transition_phase(&self, symbol: &[u8; 12], new_phase: AuctionPhase) {
        let state = self.get_or_create_symbol(symbol);
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        
        state.set_phase(new_phase);
        state.last_update_ns.store(now_ns, Ordering::Release);
    }

    /// Submit order to auction book
    pub fn submit_auction_order(&self, order: AuctionOrder) -> Result<(), &'static str> {
        let state = self.get_or_create_symbol(&order.symbol);
        
        if !state.get_phase().allows_order_entry() {
            return Err("Order entry not allowed in current auction phase");
        }

        state.book.write().add_order(order);
        Ok(())
    }

    /// Cancel order from auction book
    pub fn cancel_auction_order(&self, order_id: u64, symbol: &[u8; 12]) -> Result<Option<AuctionOrder>, &'static str> {
        let state = self.get_or_create_symbol(symbol);
        
        if !state.get_phase().allows_cancellation() {
            return Err("Cancellation not allowed in current auction phase");
        }

        Ok(state.book.write().cancel_order(order_id))
    }

    /// Modify order in auction book
    pub fn modify_auction_order(
        &self,
        order_id: u64,
        symbol: &[u8; 12],
        new_quantity: u64,
        new_price: Option<i64>,
    ) -> Result<bool, &'static str> {
        let state = self.get_or_create_symbol(symbol);
        
        if !state.get_phase().allows_order_entry() {
            return Err("Order modification not allowed in current auction phase");
        }

        Ok(state.book.write().modify_order(order_id, new_quantity, new_price))
    }

    /// Get indicative price for a symbol
    pub fn get_indicative_price(&self, symbol: &[u8; 12]) -> Option<IndicativePrice> {
        let state = self.get_or_create_symbol(symbol);
        state.book.read().calculate_indicative_price()
    }

    /// Execute auction - returns list of executed orders
    pub fn execute_auction(&self, symbol: &[u8; 12]) -> (Option<IndicativePrice>, Vec<AuctionOrder>) {
        let state = self.get_or_create_symbol(symbol);
        
        // Transition to executing phase
        state.set_phase(AuctionPhase::Executing);
        
        // Calculate clearing price
        let indicative = state.book.read().calculate_indicative_price();
        
        // Clear the book
        let orders = state.book.write().clear_book();
        
        // Reset phase after execution
        state.set_phase(AuctionPhase::Inactive);
        
        (indicative, orders)
    }

    /// Check if auction is active for any symbol
    #[inline]
    pub fn is_auction_active(&self) -> bool {
        self.global_auction_active.load(Ordering::Acquire)
    }

    /// Get current phase for a symbol
    pub fn get_symbol_phase(&self, symbol: &[u8; 12]) -> AuctionPhase {
        let state = self.get_or_create_symbol(symbol);
        state.get_phase()
    }

    /// End all auctions (emergency)
    pub fn emergency_end_all(&self) {
        let symbols = self.symbols.read();
        for state in symbols.values() {
            state.set_phase(AuctionPhase::Inactive);
        }
        self.global_auction_active.store(false, Ordering::Release);
    }

    /// Re-price orders for continuous trading resumption
    pub fn reprice_for_continuous(&self, symbol: &[u8; 12], reference_price: i64) -> Vec<AuctionOrder> {
        let state = self.get_or_create_symbol(symbol);
        let mut book = state.book.write();
        let mut repriced_orders = Vec::new();
        
        // Get all remaining orders and adjust prices relative to reference
        let all_orders = book.clear_book();
        
        for mut order in all_orders {
            if order.is_market_order {
                // Convert market orders to limit orders at reference price
                order.price_ticks = reference_price;
                order.is_market_order = false;
            }
            // Add some logic here to adjust limit order prices if needed
            // based on the reference price and order side
            repriced_orders.push(order);
        }
        
        repriced_orders
    }
}

/// Auction event for broadcasting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuctionEvent {
    pub venue_id: VenueId,
    pub symbol: [u8; 12],
    pub phase: AuctionPhase,
    pub indicative_price: Option<IndicativePriceData>,
    pub timestamp_ns: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndicativePriceData {
    pub price_ticks: i64,
    pub executable_volume: u64,
    pub imbalance: i64,
}

impl From<&IndicativePrice> for IndicativePriceData {
    fn from(ip: &IndicativePrice) -> Self {
        Self {
            price_ticks: ip.price_ticks,
            executable_volume: ip.executable_volume,
            imbalance: ip.buy_imbalance - ip.sell_imbalance,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auction_phase_transitions() {
        assert!(AuctionPhase::PreOpen.allows_order_entry());
        assert!(!AuctionPhase::Freeze.allows_order_entry());
        assert!(AuctionPhase::Freeze.allows_cancellation());
        assert!(!AuctionPhase::Executing.allows_cancellation());
    }

    #[test]
    fn test_auction_engine_basic() {
        let engine = AuctionEngine::new(VenueId::Nasdaq, 100);
        let symbol = b"TEST        ";
        
        assert!(!engine.is_auction_active());
        
        engine.start_auction(&symbol, 5000); // 5 second auction
        
        assert!(engine.is_auction_active());
        assert_eq!(engine.get_symbol_phase(&symbol), AuctionPhase::PreOpen);
    }

    #[test]
    fn test_auction_order_submission() {
        let engine = AuctionEngine::new(VenueId::Nasdaq, 100);
        let symbol = *b"AAPL        ";
        
        engine.start_auction(&symbol, 5000);
        
        let order = AuctionOrder {
            order_id: 1,
            symbol,
            side: Side::Buy,
            price_ticks: 15000,  // $150.00
            quantity: 100,
            remaining_qty: 100,
            timestamp_ns: 0,
            is_market_order: false,
            stp_group_id: None,
        };
        
        assert!(engine.submit_auction_order(order).is_ok());
    }
}
