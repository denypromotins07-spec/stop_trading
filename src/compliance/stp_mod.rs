//! Compliance Module Root
//! 
//! Tags all outbound orders with unique STP group IDs and expiration flags.
//! Central coordination for self-trade prevention and wash trade avoidance.

pub mod stp;
pub mod wash_trade;

pub use stp::{
    StpEngine, StpGroupId, StpAction, StpCheckResult, StpStats,
    RestingOrder, AggressiveOrder, OrderSide, ExchangeStpConfig,
};
pub use wash_trade::{
    WashTradeDetector, WashTradeAlert, WashTradeSeverity, WashPattern,
    FillRecord, FillSide, LiquidityType, WashTradeStats,
};

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use crate::gateway::venue::VenueId;
use super::stp::StpEngine;

/// Compliance tag for outbound orders
#[derive(Debug, Clone)]
pub struct ComplianceTag {
    pub stp_group_id: StpGroupId,
    pub expiration_flag: ExpirationFlag,
    pub strategy_id: u32,
    pub compliance_version: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ExpirationFlag {
    Day = 0,
    Gtc = 1,
    Ioc = 2,
    Fok = 3,
    Session = 4,
}

impl ExpirationFlag {
    #[inline]
    pub fn from_u8(val: u8) -> Self {
        match val {
            0 => ExpirationFlag::Day,
            1 => ExpirationFlag::Gtc,
            2 => ExpirationFlag::Ioc,
            3 => ExpirationFlag::Fok,
            4 => ExpirationFlag::Session,
            _ => ExpirationFlag::Day,
        }
    }

    #[inline]
    pub fn to_u8(self) -> u8 {
        self as u8
    }
}

/// Main Compliance Manager
/// Coordinates STP and wash trade detection across all venues
pub struct ComplianceManager {
    /// Per-venue STP engines
    stp_engines: parking_lot::RwLock<Vec<(VenueId, Arc<StpEngine>)>>,
    /// Global wash trade detector
    wash_detector: Arc<WashTradeDetector>,
    /// Strategy ID counter for unique STP group generation
    strategy_counter: AtomicU32,
    /// Compliance enabled flag
    enabled: AtomicBool,
    /// Total orders tagged
    orders_tagged: AtomicU64,
    /// Total compliance violations
    violations_detected: AtomicU64,
}

impl ComplianceManager {
    pub fn new(venues: &[VenueId]) -> Self {
        let mut stp_engines = Vec::with_capacity(venues.len());
        
        for &venue_id in venues {
            let config = ExchangeStpConfig::for_venue(venue_id);
            let engine = Arc::new(StpEngine::new(venue_id, config.default_action));
            stp_engines.push((venue_id, engine));
        }
        
        Self {
            stp_engines: parking_lot::RwLock::new(stp_engines),
            wash_detector: Arc::new(WashTradeDetector::new()),
            strategy_counter: AtomicU32::new(1),
            enabled: AtomicBool::new(true),
            orders_tagged: AtomicU64::new(0),
            violations_detected: AtomicU64::new(0),
        }
    }

    /// Get STP engine for venue
    pub fn get_stp_engine(&self, venue_id: VenueId) -> Option<Arc<StpEngine>> {
        let engines = self.stp_engines.read();
        engines.iter().find(|(v, _)| *v == venue_id).map(|(_, e)| Arc::clone(e))
    }

    /// Get wash trade detector
    #[inline]
    pub fn wash_detector(&self) -> &Arc<WashTradeDetector> {
        &self.wash_detector
    }

    /// Generate unique STP group ID for a strategy
    pub fn generate_stp_group(&self, strategy_name: &str, symbol_hash: u32) -> StpGroupId {
        let strategy_id = self.strategy_counter.fetch_add(1, Ordering::Relaxed);
        let name_hash = fxhash::hash_str(strategy_name) as u32;
        StpGroupId::generate(strategy_id, name_hash.wrapping_add(symbol_hash))
    }

    /// Tag outbound order with compliance information
    pub fn tag_order(
        &self,
        venue_id: VenueId,
        strategy_id: u32,
        symbol: &[u8; 12],
        expiration: ExpirationFlag,
    ) -> ComplianceTag {
        self.orders_tagged.fetch_add(1, Ordering::Relaxed);

        let symbol_hash = fxhash::hash_bytes(symbol) as u32;
        let stp_group_id = StpGroupId::generate(strategy_id, symbol_hash);

        ComplianceTag {
            stp_group_id,
            expiration_flag: expiration,
            strategy_id,
            compliance_version: 1,
        }
    }

    /// Validate order before submission
    pub fn validate_order(
        &self,
        venue_id: VenueId,
        aggressive: &AggressiveOrder,
    ) -> Result<(), ComplianceViolation> {
        if !self.enabled.load(Ordering::Acquire) {
            return Ok(());
        }

        // Check STP
        if let Some(engine) = self.get_stp_engine(venue_id) {
            let result = engine.check_aggressive_order(aggressive);
            
            match result {
                StpCheckResult::Pass => {}
                StpCheckResult::Conflict { action, conflicting_order_ids } => {
                    self.violations_detected.fetch_add(1, Ordering::Relaxed);
                    return Err(ComplianceViolation::SelfTradePrevention {
                        action,
                        conflicting_orders: conflicting_order_ids,
                    });
                }
                StpCheckResult::Reject { reason } => {
                    self.violations_detected.fetch_add(1, Ordering::Relaxed);
                    return Err(ComplianceViolation::Rejected(reason));
                }
            }
        }

        Ok(())
    }

    /// Record fill for wash trade analysis
    pub fn record_fill(&self, fill: FillRecord) -> Option<WashTradeAlert> {
        self.wash_detector.record_fill(fill)
    }

    /// Enable/disable all compliance checks
    #[inline]
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Release);
        self.wash_detector.set_enabled(enabled);
        
        let engines = self.stp_engines.read();
        for (_, engine) in engines.iter() {
            engine.set_enabled(enabled);
        }
    }

    /// Get compliance statistics
    pub fn get_stats(&self) -> ComplianceStats {
        let engines = self.stp_engines.read();
        let stp_stats: Vec<(VenueId, StpStats)> = engines
            .iter()
            .map(|(venue, engine)| (*venue, engine.get_stats()))
            .collect();

        ComplianceStats {
            orders_tagged: self.orders_tagged.load(Ordering::Relaxed),
            violations_detected: self.violations_detected.load(Ordering::Relaxed),
            stp_stats,
            wash_stats: self.wash_detector.get_stats(),
            is_enabled: self.enabled.load(Ordering::Acquire),
        }
    }

    /// Register resting order with STP engine
    pub fn register_resting_order(&self, venue_id: VenueId, order: RestingOrder) {
        if let Some(engine) = self.get_stp_engine(venue_id) {
            engine.register_resting_order(order);
        }
    }

    /// Remove resting order from STP tracking
    pub fn remove_resting_order(&self, venue_id: VenueId, symbol: &[u8; 12], order_id: u64) {
        if let Some(engine) = self.get_stp_engine(venue_id) {
            engine.remove_resting_order(symbol, order_id);
        }
    }
}

/// Compliance violation types
#[derive(Debug, Clone)]
pub enum ComplianceViolation {
    SelfTradePrevention {
        action: StpAction,
        conflicting_orders: Vec<u64>,
    },
    WashTradeDetected(WashTradeAlert),
    Rejected(&'static str),
}

/// Aggregate compliance statistics
#[derive(Debug, Clone)]
pub struct ComplianceStats {
    pub orders_tagged: u64,
    pub violations_detected: u64,
    pub stp_stats: Vec<(VenueId, StpStats)>,
    pub wash_stats: WashTradeStats,
    pub is_enabled: bool,
}

/// Helper to get exchange-specific STP config
impl ExchangeStpConfig {
    pub fn for_venue(venue_id: VenueId) -> Self {
        match venue_id {
            VenueId::Nasdaq => Self::nasdaq(),
            VenueId::NYSE => Self::nyse(),
            VenueId::Binance => Self::binance(),
            _ => Self {
                venue_id,
                supported_actions: vec![StpAction::CancelAggressive],
                default_action: StpAction::CancelAggressive,
                supports_group_ids: true,
                max_groups: 100,
            },
        }
    }
}

// Simple hash implementation to avoid external dependency issues
mod fxhash {
    pub fn hash_str(s: &str) -> u64 {
        let mut hash: u64 = 0x517cc1b727220a95;
        for byte in s.bytes() {
            hash = hash.rotate_left(5) ^ (byte as u64).wrapping_mul(0x517cc1b727220a95);
        }
        hash
    }

    pub fn hash_bytes(bytes: &[u8]) -> u64 {
        let mut hash: u64 = 0x517cc1b727220a95;
        for &byte in bytes {
            hash = hash.rotate_left(5) ^ (byte as u64).wrapping_mul(0x517cc1b727220a95);
        }
        hash
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compliance_manager_creation() {
        let venues = vec![VenueId::Nasdaq, VenueId::NYSE];
        let manager = ComplianceManager::new(&venues);
        
        assert!(manager.get_stp_engine(VenueId::Nasdaq).is_some());
        assert!(manager.get_stp_engine(VenueId::NYSE).is_some());
        assert!(manager.enabled.load(Ordering::Acquire));
    }

    #[test]
    fn test_order_tagging() {
        let venues = vec![VenueId::Nasdaq];
        let manager = ComplianceManager::new(&venues);
        
        let symbol = *b"AAPL        ";
        let tag = manager.tag_order(VenueId::Nasdaq, 1, &symbol, ExpirationFlag::Day);
        
        assert!(tag.stp_group_id.value() > 0);
        assert_eq!(tag.expiration_flag, ExpirationFlag::Day);
        assert_eq!(tag.strategy_id, 1);
    }

    #[test]
    fn test_expiration_flag_conversion() {
        assert_eq!(ExpirationFlag::Day.to_u8(), 0);
        assert_eq!(ExpirationFlag::from_u8(0), ExpirationFlag::Day);
        assert_eq!(ExpirationFlag::Gtc.to_u8(), 1);
        assert_eq!(ExpirationFlag::from_u8(1), ExpirationFlag::Gtc);
    }

    #[test]
    fn test_stats_initial() {
        let venues = vec![VenueId::Nasdaq];
        let manager = ComplianceManager::new(&venues);
        
        let stats = manager.get_stats();
        assert_eq!(stats.orders_tagged, 0);
        assert_eq!(stats.violations_detected, 0);
        assert!(stats.is_enabled);
    }
}
