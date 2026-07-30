//! Self-Trade Prevention (STP) Module
//! 
//! Implements strict Self-Trade Prevention logic at the smart order routing layer
//! using exchange-specific STP flags. Ensures the bot's aggressive maker and taker
//! orders never cross each other, avoiding toxic wash-trade penalties.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use serde::{Deserialize, Serialize};
use crate::gateway::venue::VenueId;

/// STP action enumeration - what to do when potential self-trade detected
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum StpAction {
    /// Cancel the incoming aggressive order
    CancelAggressive = 0,
    /// Cancel the resting maker order
    CancelMaker = 1,
    /// Cancel both orders
    CancelBoth = 2,
    /// Allow the trade (only for different STP groups)
    Allow = 3,
}

/// STP group identifier - orders in same group cannot trade with each other
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StpGroupId(pub u32);

impl StpGroupId {
    #[inline]
    pub fn new(id: u32) -> Self {
        StpGroupId(id)
    }

    #[inline]
    pub fn value(&self) -> u32 {
        self.0
    }

    /// Generate unique STP group ID based on strategy and symbol
    pub fn generate(strategy_id: u32, symbol_hash: u32) -> Self {
        let combined = strategy_id.wrapping_mul(31).wrapping_add(symbol_hash);
        StpGroupId(combined)
    }
}

/// Order side for STP matching
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum OrderSide {
    Buy = 0,
    Sell = 1,
}

/// Resting order record for STP checking
#[derive(Debug, Clone)]
pub struct RestingOrder {
    pub order_id: u64,
    pub symbol: [u8; 12],
    pub side: OrderSide,
    pub price_ticks: i64,
    pub quantity: u64,
    pub stp_group_id: StpGroupId,
    pub timestamp_ns: u64,
}

/// Incoming aggressive order for STP checking
#[derive(Debug, Clone)]
pub struct AggressiveOrder {
    pub symbol: [u8; 12],
    pub side: OrderSide,
    pub price_ticks: i64,
    pub quantity: u64,
    pub stp_group_id: StpGroupId,
    pub is_market: bool,
}

/// STP check result
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StpCheckResult {
    /// No self-trade conflict - order can proceed
    Pass,
    /// Self-trade detected - action required
    Conflict {
        action: StpAction,
        conflicting_order_ids: Vec<u64>,
    },
    /// Order should be rejected before checking book
    Reject {
        reason: &'static str,
    },
}

/// Per-symbol STP state tracker
struct SymbolStpState {
    symbol: [u8; 12],
    /// Resting buy orders by STP group
    buy_orders_by_group: HashMap<StpGroupId, Vec<RestingOrder>>,
    /// Resting sell orders by STP group
    sell_orders_by_group: HashMap<StpGroupId, Vec<RestingOrder>>,
    /// All resting orders by ID for quick lookup
    orders_by_id: HashMap<u64, RestingOrder>,
    /// Best bid price per STP group
    best_bid_by_group: HashMap<StpGroupId, i64>,
    /// Best offer price per STP group
    best_offer_by_group: HashMap<StpGroupId, i64>,
}

impl SymbolStpState {
    fn new(symbol: [u8; 12]) -> Self {
        Self {
            symbol,
            buy_orders_by_group: HashMap::new(),
            sell_orders_by_group: HashMap::new(),
            orders_by_id: HashMap::new(),
            best_bid_by_group: HashMap::new(),
            best_offer_by_group: HashMap::new(),
        }
    }

    fn add_order(&mut self, order: RestingOrder) {
        let orders_map = match order.side {
            OrderSide::Buy => &mut self.buy_orders_by_group,
            OrderSide::Sell => &mut self.sell_orders_by_group,
        };

        orders_map.entry(order.stp_group_id).or_default().push(order.clone());
        self.orders_by_id.insert(order.order_id, order.clone());

        // Update best prices
        match order.side {
            OrderSide::Buy => {
                let best_bid = self.best_bid_by_group.entry(order.stp_group_id).or_insert(i64::MIN);
                *best_bid = (*best_bid).max(order.price_ticks);
            }
            OrderSide::Sell => {
                let best_offer = self.best_offer_by_group.entry(order.stp_group_id).or_insert(i64::MAX);
                *best_offer = (*best_offer).min(order.price_ticks);
            }
        }
    }

    fn remove_order(&mut self, order_id: u64) -> Option<RestingOrder> {
        if let Some(order) = self.orders_by_id.remove(&order_id) {
            let orders_map = match order.side {
                OrderSide::Buy => &mut self.buy_orders_by_group,
                OrderSide::Sell => &mut self.sell_orders_by_group,
            };

            if let Some(orders) = orders_map.get_mut(&order.stp_group_id) {
                orders.retain(|o| o.order_id != order_id);
                if orders.is_empty() {
                    orders_map.remove(&order.stp_group_id);
                }
            }

            // Recalculate best prices if needed
            self.recalculate_best_prices();

            Some(order)
        } else {
            None
        }
    }

    fn recalculate_best_prices(&mut self) {
        // Recalculate best bid per group
        for (group_id, orders) in &self.buy_orders_by_group {
            if let Some(max_price) = orders.iter().map(|o| o.price_ticks).max() {
                self.best_bid_by_group.insert(*group_id, max_price);
            }
        }

        // Recalculate best offer per group
        for (group_id, orders) in &self.sell_orders_by_group {
            if let Some(min_price) = orders.iter().map(|o| o.price_ticks).min() {
                self.best_offer_by_group.insert(*group_id, min_price);
            }
        }
    }

    /// Check if aggressive order would cross any resting orders from same STP group
    fn check_self_trade(&self, aggressive: &AggressiveOrder, default_action: StpAction) -> StpCheckResult {
        let mut conflicting_ids = Vec::new();

        // Find resting orders that would match
        match aggressive.side {
            OrderSide::Buy => {
                // Aggressive buy checks against resting sells
                if let Some(resting_sells) = self.sell_orders_by_group.get(&aggressive.stp_group_id) {
                    for order in resting_sells {
                        let would_match = if aggressive.is_market {
                            true
                        } else {
                            aggressive.price_ticks >= order.price_ticks
                        };

                        if would_match {
                            conflicting_ids.push(order.order_id);
                        }
                    }
                }
            }
            OrderSide::Sell => {
                // Aggressive sell checks against resting buys
                if let Some(resting_buys) = self.buy_orders_by_group.get(&aggressive.stp_group_id) {
                    for order in resting_buys {
                        let would_match = if aggressive.is_market {
                            true
                        } else {
                            aggressive.price_ticks <= order.price_ticks
                        };

                        if would_match {
                            conflicting_ids.push(order.order_id);
                        }
                    }
                }
            }
        }

        if conflicting_ids.is_empty() {
            StpCheckResult::Pass
        } else {
            StpCheckResult::Conflict {
                action: default_action,
                conflicting_order_ids: conflicting_ids,
            }
        }
    }
}

/// Main STP Engine
pub struct StpEngine {
    venue_id: VenueId,
    /// Per-symbol STP state
    symbol_states: parking_lot::RwLock<HashMap<[u8; 12], SymbolStpState>>,
    /// Default STP action for this venue
    default_action: StpAction,
    /// Global STP enabled flag
    enabled: AtomicBool,
    /// Statistics
    checks_performed: AtomicU64,
    conflicts_detected: AtomicU64,
    orders_rejected: AtomicU64,
}

impl StpEngine {
    pub fn new(venue_id: VenueId, default_action: StpAction) -> Self {
        Self {
            venue_id,
            symbol_states: parking_lot::RwLock::new(HashMap::new()),
            default_action,
            enabled: AtomicBool::new(true),
            checks_performed: AtomicU64::new(0),
            conflicts_detected: AtomicU64::new(0),
            orders_rejected: AtomicU64::new(0),
        }
    }

    /// Get or create symbol state
    fn get_or_create_state(&self, symbol: &[u8; 12]) -> parking_lot::RwLockWriteGuard<SymbolStpState> {
        {
            let states = self.symbol_states.read();
            if states.contains_key(symbol) {
                drop(states);
                // Need to upgrade to write lock
            }
        }

        let mut states = self.symbol_states.write();
        states.entry(*symbol).or_insert_with(|| SymbolStpState::new(*symbol));
        
        // This is a bit awkward - we need a different approach
        // For now, return a guard that we'll use carefully
        parking_lot::RwLockWriteGuard::map(states, |s| {
            s.entry(*symbol).or_insert_with(|| SymbolStpState::new(*symbol))
        })
    }

    /// Register a resting order in the STP engine
    pub fn register_resting_order(&self, order: RestingOrder) {
        if !self.enabled.load(Ordering::Acquire) {
            return;
        }

        let mut states = self.symbol_states.write();
        let state = states.entry(order.symbol).or_insert_with(|| SymbolStpState::new(order.symbol));
        state.add_order(order);
    }

    /// Remove a resting order from STP tracking
    pub fn remove_resting_order(&self, symbol: &[u8; 12], order_id: u64) {
        let mut states = self.symbol_states.write();
        if let Some(state) = states.get_mut(symbol) {
            state.remove_order(order_id);
        }
    }

    /// Check incoming aggressive order for self-trade conflicts
    pub fn check_aggressive_order(&self, order: &AggressiveOrder) -> StpCheckResult {
        if !self.enabled.load(Ordering::Acquire) {
            return StpCheckResult::Pass;
        }

        self.checks_performed.fetch_add(1, Ordering::Relaxed);

        let states = self.symbol_states.read();
        if let Some(state) = states.get(&order.symbol) {
            let result = state.check_self_trade(order, self.default_action);
            
            match &result {
                StpCheckResult::Conflict { .. } => {
                    self.conflicts_detected.fetch_add(1, Ordering::Relaxed);
                }
                StpCheckResult::Reject { .. } => {
                    self.orders_rejected.fetch_add(1, Ordering::Relaxed);
                }
                _ => {}
            }
            
            result
        } else {
            StpCheckResult::Pass
        }
    }

    /// Generate STP tag for outbound order
    pub fn generate_stp_tag(&self, stp_group_id: StpGroupId) -> String {
        format!("STP{:08X}", stp_group_id.0)
    }

    /// Parse STP group ID from tag
    pub fn parse_stp_tag(tag: &str) -> Option<StpGroupId> {
        if tag.len() >= 11 && tag.starts_with("STP") {
            u32::from_str_radix(&tag[3..11], 16).ok().map(StpGroupId)
        } else {
            None
        }
    }

    /// Enable/disable STP checking
    #[inline]
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Release);
    }

    /// Check if STP is enabled
    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Acquire)
    }

    /// Get statistics
    pub fn get_stats(&self) -> StpStats {
        StpStats {
            checks_performed: self.checks_performed.load(Ordering::Relaxed),
            conflicts_detected: self.conflicts_detected.load(Ordering::Relaxed),
            orders_rejected: self.orders_rejected.load(Ordering::Relaxed),
        }
    }

    /// Clear all STP state (use with caution)
    pub fn clear_all(&self) {
        self.symbol_states.write().clear();
    }
}

/// STP statistics
#[derive(Debug, Clone, Default)]
pub struct StpStats {
    pub checks_performed: u64,
    pub conflicts_detected: u64,
    pub orders_rejected: u64,
}

/// Exchange-specific STP configuration
#[derive(Debug, Clone)]
pub struct ExchangeStpConfig {
    pub venue_id: VenueId,
    pub supported_actions: Vec<StpAction>,
    pub default_action: StpAction,
    /// Whether exchange supports STP group IDs
    pub supports_group_ids: bool,
    /// Maximum STP groups allowed
    pub max_groups: u32,
}

impl ExchangeStpConfig {
    /// NASDAQ STP configuration
    pub fn nasdaq() -> Self {
        Self {
            venue_id: VenueId::Nasdaq,
            supported_actions: vec![StpAction::CancelAggressive, StpAction::CancelMaker, StpAction::CancelBoth],
            default_action: StpAction::CancelAggressive,
            supports_group_ids: true,
            max_groups: 100,
        }
    }

    /// NYSE STP configuration
    pub fn nyse() -> Self {
        Self {
            venue_id: VenueId::NYSE,
            supported_actions: vec![StpAction::CancelAggressive, StpAction::CancelBoth],
            default_action: StpAction::CancelAggressive,
            supports_group_ids: true,
            max_groups: 50,
        }
    }

    /// Binance crypto STP configuration
    pub fn binance() -> Self {
        Self {
            venue_id: VenueId::Binance,
            supported_actions: vec![StpAction::CancelMaker],
            default_action: StpAction::CancelMaker,
            supports_group_ids: false,
            max_groups: 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stp_group_id_generation() {
        let id1 = StpGroupId::generate(1, 100);
        let id2 = StpGroupId::generate(1, 100);
        let id3 = StpGroupId::generate(2, 100);
        
        assert_eq!(id1, id2);
        assert_ne!(id1, id3);
    }

    #[test]
    fn test_stp_engine_basic() {
        let engine = StpEngine::new(VenueId::Nasdaq, StpAction::CancelAggressive);
        
        let symbol = *b"AAPL        ";
        
        // Register a resting sell order
        let resting = RestingOrder {
            order_id: 1,
            symbol,
            side: OrderSide::Sell,
            price_ticks: 15000,
            quantity: 100,
            stp_group_id: StpGroupId::new(1),
            timestamp_ns: 0,
        };
        engine.register_resting_order(resting);

        // Check aggressive buy from same STP group
        let aggressive = AggressiveOrder {
            symbol,
            side: OrderSide::Buy,
            price_ticks: 15000,
            quantity: 50,
            stp_group_id: StpGroupId::new(1),
            is_market: false,
        };

        let result = engine.check_aggressive_order(&aggressive);
        assert!(matches!(result, StpCheckResult::Conflict { .. }));

        // Check aggressive buy from different STP group
        let aggressive2 = AggressiveOrder {
            stp_group_id: StpGroupId::new(2),
            ..aggressive.clone()
        };

        let result2 = engine.check_aggressive_order(&aggressive2);
        assert_eq!(result2, StpCheckResult::Pass);
    }

    #[test]
    fn test_stp_tag_generation() {
        let engine = StpEngine::new(VenueId::Nasdaq, StpAction::CancelAggressive);
        let tag = engine.generate_stp_tag(StpGroupId::new(0x12345678));
        assert_eq!(tag, "STP12345678");

        let parsed = StpEngine::parse_stp_tag(&tag);
        assert_eq!(parsed, Some(StpGroupId::new(0x12345678)));
    }
}
