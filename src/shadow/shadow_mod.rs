//! Shadow Module Root
//! 
//! Routes market data to shadow actors while strictly blocking REST order submissions.
//! Central coordination for shadow mode execution and comparison.

pub mod engine;
pub mod comparator;

pub use engine::{ShadowEngine, ShadowOrder, ShadowFill, ShadowPnL, ShadowSide, ShadowStats};
pub use comparator::{ShadowComparator, ComparisonResult, AlertLevel, ComparatorStats, LivePnLTracker};

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use crate::gateway::venue::VenueId;
use crate::orderbook::book::OrderBook;

/// Shadow mode configuration
#[derive(Debug, Clone)]
pub struct ShadowConfig {
    pub enabled: bool,
    pub symbols: Vec<[u8; 12]>,
    pub slippage_bps: f64,
    pub latency_ns: u64,
    pub comparison_interval_ms: u64,
}

impl Default for ShadowConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            symbols: Vec::new(),
            slippage_bps: 2.0,
            latency_ns: 100_000,
            comparison_interval_ms: 1000,
        }
    }
}

/// Main Shadow Manager
/// Coordinates shadow engine and comparator across venues
pub struct ShadowManager {
    /// Per-venue shadow engines
    engines: parking_lot::RwLock<Vec<(VenueId, Arc<ShadowEngine>)>>,
    /// Per-venue comparators
    comparators: parking_lot::RwLock<Vec<(VenueId, Arc<ShadowComparator>)>>,
    /// Global shadow mode flag
    enabled: AtomicBool,
    /// Orders blocked from live submission
    orders_blocked: AtomicU64,
    /// Shadow orders processed
    shadow_orders_processed: AtomicU64,
}

impl ShadowManager {
    pub fn new(venues: &[VenueId]) -> Self {
        let mut engines = Vec::with_capacity(venues.len());
        let mut comparators = Vec::with_capacity(venues.len());

        for &venue_id in venues {
            // Create shared live book reference (zero-copy)
            let symbol = [0u8; 12];  // Will be updated per-symbol
            let live_book = Arc::new(parking_lot::RwLock::new(OrderBook::new(symbol)));
            
            let engine = Arc::new(ShadowEngine::new(venue_id, Arc::clone(&live_book)));
            let mut comparator = ShadowComparator::new(venue_id);
            comparator.set_shadow_engine(Arc::clone(&engine));
            let comparator = Arc::new(comparator);

            engines.push((venue_id, engine));
            comparators.push((venue_id, comparator));
        }

        Self {
            engines: parking_lot::RwLock::new(engines),
            comparators: parking_lot::RwLock::new(comparators),
            enabled: AtomicBool::new(false),
            orders_blocked: AtomicU64::new(0),
            shadow_orders_processed: AtomicU64::new(0),
        }
    }

    /// Get shadow engine for venue
    pub fn get_engine(&self, venue_id: VenueId) -> Option<Arc<ShadowEngine>> {
        let engines = self.engines.read();
        engines.iter().find(|(v, _)| *v == venue_id).map(|(_, e)| Arc::clone(e))
    }

    /// Get comparator for venue
    pub fn get_comparator(&self, venue_id: VenueId) -> Option<Arc<ShadowComparator>> {
        let comparators = self.comparators.read();
        comparators.iter().find(|(v, _)| *v == venue_id).map(|(_, c)| Arc::clone(c))
    }

    /// Check if an order should be routed to shadow instead of live
    pub fn should_route_to_shadow(&self, venue_id: VenueId, is_live_order: bool) -> bool {
        if !self.enabled.load(Ordering::Acquire) {
            return false;
        }

        // If shadow mode is on and this isn't explicitly a live order, route to shadow
        if !is_live_order {
            self.shadow_orders_processed.fetch_add(1, Ordering::Relaxed);
            return true;
        }

        // Check if we have an engine for this venue
        let has_engine = {
            let engines = self.engines.read();
            engines.iter().any(|(v, _)| *v == venue_id)
        };

        if !has_engine {
            return false;
        }

        // In shadow mode, block live orders
        self.orders_blocked.fetch_add(1, Ordering::Relaxed);
        false
    }

    /// Submit order - returns true if submitted to shadow, false if blocked
    pub fn submit_order(
        &self,
        venue_id: VenueId,
        order: ShadowOrder,
    ) -> Result<bool, &'static str> {
        if !self.enabled.load(Ordering::Acquire) {
            return Err("Shadow mode disabled");
        }

        if let Some(engine) = self.get_engine(venue_id) {
            engine.submit_shadow_order(order)?;
            Ok(true)
        } else {
            Err("No shadow engine for venue")
        }
    }

    /// Enable/disable shadow mode globally
    #[inline]
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Release);

        // Propagate to all engines and comparators
        let engines = self.engines.read();
        for (_, engine) in engines.iter() {
            engine.set_enabled(enabled);
        }

        let comparators = self.comparators.read();
        for (_, comparator) in comparators.iter() {
            comparator.set_enabled(enabled);
        }
    }

    /// Configure all shadow engines
    pub fn configure_all(&self, slippage_bps: f64, latency_ns: u64) {
        let engines = self.engines.read();
        for (_, engine) in engines.iter() {
            let mut mutable_engine = (*engine).clone();
            // Note: We'd need interior mutability here for proper config
            // For now, this is a placeholder
        }
    }

    /// Get aggregate statistics
    pub fn get_stats(&self) -> ShadowManagerStats {
        let engines = self.engines.read();
        let comparators = self.comparators.read();

        let mut total_symbols = 0;
        let mut total_fills = 0;
        let mut total_comparisons = 0;
        let mut total_alerts = 0;

        for (_, engine) in engines.iter() {
            let stats = engine.get_stats();
            total_symbols += stats.symbols_tracked;
            total_fills += stats.total_fills;
        }

        for (_, comparator) in comparators.iter() {
            let stats = comparator.get_stats();
            total_comparisons += stats.total_comparisons;
            total_alerts += stats.alerts_generated;
        }

        ShadowManagerStats {
            enabled: self.enabled.load(Ordering::Acquire),
            venues_count: engines.len(),
            total_symbols,
            total_fills,
            total_comparisons,
            total_alerts,
            orders_blocked: self.orders_blocked.load(Ordering::Relaxed),
            shadow_orders_processed: self.shadow_orders_processed.load(Ordering::Relaxed),
        }
    }

    /// Record live fill for comparison
    pub fn record_live_fill(&self, venue_id: VenueId, symbol: &[u8; 12], side: u8, price: i64, quantity: u64) {
        if let Some(comparator) = self.get_comparator(venue_id) {
            comparator.record_live_fill(symbol, side, price, quantity);
        }
    }

    /// Compare performance for symbol
    pub fn compare_performance(&self, venue_id: VenueId, symbol: &[u8; 12], current_price: i64) -> Option<ComparisonResult> {
        if let Some(comparator) = self.get_comparator(venue_id) {
            comparator.compare(symbol, current_price)
        } else {
            None
        }
    }
}

/// Aggregate shadow manager statistics
#[derive(Debug, Clone, Default)]
pub struct ShadowManagerStats {
    pub enabled: bool,
    pub venues_count: usize,
    pub total_symbols: usize,
    pub total_fills: u64,
    pub total_comparisons: u64,
    pub total_alerts: u64,
    pub orders_blocked: u64,
    pub shadow_orders_processed: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shadow_manager_creation() {
        let venues = vec![VenueId::Nasdaq, VenueId::NYSE];
        let manager = ShadowManager::new(&venues);

        assert!(!manager.enabled.load(Ordering::Acquire));
        assert_eq!(manager.get_stats().venues_count, 2);
    }

    #[test]
    fn test_routing_decision() {
        let venues = vec![VenueId::Nasdaq];
        let manager = ShadowManager::new(&venues);

        // When disabled, should not route to shadow
        assert!(!manager.should_route_to_shadow(VenueId::Nasdaq, false));

        // Enable shadow mode
        manager.set_enabled(true);

        // Non-live orders should route to shadow
        assert!(manager.should_route_to_shadow(VenueId::Nasdaq, false));

        // Live orders should be blocked
        assert!(!manager.should_route_to_shadow(VenueId::Nasdaq, true));
    }

    #[test]
    fn test_stats_initial() {
        let venues = vec![VenueId::Nasdaq];
        let manager = ShadowManager::new(&venues);

        let stats = manager.get_stats();
        assert!(!stats.enabled);
        assert_eq!(stats.orders_blocked, 0);
        assert_eq!(stats.shadow_orders_processed, 0);
    }
}
