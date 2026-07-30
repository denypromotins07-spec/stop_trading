//! Exchange Control Module Root
//! 
//! Wires halt signals directly to the global kill switch and UI.
//! Central coordination point for exchange state management.

pub mod halt_detector;
pub mod auction_engine;

pub use halt_detector::{HaltDetector, HaltEvent, TradingState, CircuitBreakerLevel, parse_halt_message};
pub use auction_engine::{AuctionEngine, AuctionOrder, AuctionPhase, IndicativePrice, Side};

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::broadcast::Receiver;
use crate::gateway::venue::VenueId;

/// Exchange control state - aggregates halt and auction status
#[derive(Debug, Clone)]
pub struct ExchangeControlState {
    pub venue_id: VenueId,
    pub is_halted: bool,
    pub halt_state: TradingState,
    pub auction_active: bool,
    pub auction_phase: AuctionPhase,
    pub circuit_breaker_level: CircuitBreakerLevel,
    pub last_update_ns: u64,
}

impl ExchangeControlState {
    #[inline]
    pub fn is_tradeable(&self) -> bool {
        !self.is_halted && self.halt_state.is_tradeable() && !self.auction_active
    }
}

/// Main Exchange Control Module
/// Coordinates halt detection and auction management for a venue
pub struct ExchangeControl {
    venue_id: VenueId,
    halt_detector: Arc<HaltDetector>,
    auction_engine: Arc<AuctionEngine>,
    /// Cached control state for fast reads
    cached_state: parking_lot::RwLock<ExchangeControlState>,
    /// Global kill switch reference
    global_kill_switch: Arc<AtomicBool>,
    /// Last state update timestamp
    last_update_ns: AtomicU64,
}

impl ExchangeControl {
    /// Create new ExchangeControl for a venue
    pub fn new(venue_id: VenueId, venues: &[VenueId]) -> Self {
        let (halt_detector, _halt_rx) = HaltDetector::new(venues);
        let auction_engine = AuctionEngine::new(venue_id, 1000);
        
        let halt_detector = Arc::new(halt_detector);
        let auction_engine = Arc::new(auction_engine);
        let global_kill_switch = halt_detector.get_kill_switch();
        
        let initial_state = ExchangeControlState {
            venue_id,
            is_halted: false,
            halt_state: TradingState::Open,
            auction_active: false,
            auction_phase: AuctionPhase::Inactive,
            circuit_breaker_level: CircuitBreakerLevel::None,
            last_update_ns: 0,
        };
        
        Self {
            venue_id,
            halt_detector,
            auction_engine,
            cached_state: parking_lot::RwLock::new(initial_state),
            global_kill_switch,
            last_update_ns: AtomicU64::new(0),
        }
    }

    /// Get halt detector reference
    #[inline]
    pub fn halt_detector(&self) -> &Arc<HaltDetector> {
        &self.halt_detector
    }

    /// Get auction engine reference
    #[inline]
    pub fn auction_engine(&self) -> &Arc<AuctionEngine> {
        &self.auction_engine
    }

    /// Get global kill switch for wiring to execution engine
    #[inline]
    pub fn get_global_kill_switch(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.global_kill_switch)
    }

    /// Check if trading is allowed for a symbol
    #[inline]
    pub fn is_symbol_tradeable(&self, symbol: &[u8; 12]) -> bool {
        // Fast path: check global kill switch
        if self.global_kill_switch.load(Ordering::Acquire) {
            return false;
        }

        // Check halt status
        if !self.halt_detector.is_symbol_tradeable(self.venue_id, symbol) {
            return false;
        }

        // Check auction status
        let phase = self.auction_engine.get_symbol_phase(symbol);
        if !phase.allows_order_entry() {
            return false;
        }

        true
    }

    /// Check if entire venue is tradeable
    #[inline]
    pub fn is_venue_tradeable(&self) -> bool {
        if self.global_kill_switch.load(Ordering::Acquire) {
            return false;
        }
        
        self.halt_detector.is_venue_tradeable(self.venue_id) && !self.auction_engine.is_auction_active()
    }

    /// Get current control state
    pub fn get_control_state(&self) -> ExchangeControlState {
        self.cached_state.read().clone()
    }

    /// Update cached state from underlying components
    pub fn refresh_state(&self) {
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        let is_halted = self.global_kill_switch.load(Ordering::Acquire);
        let auction_active = self.auction_engine.is_auction_active();
        
        // Get representative symbol state (could be improved with per-symbol tracking)
        let dummy_symbol = [0u8; 12];
        let halt_state = self.halt_detector.get_symbol_state(self.venue_id, &dummy_symbol);
        
        let mut state = self.cached_state.write();
        state.is_halted = is_halted;
        state.halt_state = halt_state;
        state.auction_active = auction_active;
        state.auction_phase = if auction_active {
            self.auction_engine.get_symbol_phase(&dummy_symbol)
        } else {
            AuctionPhase::Inactive
        };
        state.last_update_ns = now_ns;
        
        self.last_update_ns.store(now_ns, Ordering::Release);
    }

    /// Emergency stop - triggers global kill switch
    #[inline]
    pub fn emergency_stop(&self) {
        self.halt_detector.emergency_stop();
    }

    /// Subscribe to halt events
    pub fn subscribe_halts(&self) -> Receiver<HaltEvent> {
        self.halt_detector.subscribe()
    }

    /// Process incoming halt message
    pub fn process_halt(&self, event: HaltEvent) {
        self.halt_detector.process_halt_message(event);
        self.refresh_state();
    }

    /// Start auction for symbol
    pub fn start_auction(&self, symbol: &[u8; 12], duration_ms: u64) {
        self.auction_engine.start_auction(symbol, duration_ms);
        self.refresh_state();
    }

    /// Submit order to auction
    pub fn submit_auction_order(&self, order: AuctionOrder) -> Result<(), &'static str> {
        self.auction_engine.submit_auction_order(order)
    }

    /// Execute auction for symbol
    pub fn execute_auction(&self, symbol: &[u8; 12]) -> (Option<IndicativePrice>, Vec<AuctionOrder>) {
        let result = self.auction_engine.execute_auction(symbol);
        self.refresh_state();
        result
    }

    /// Get indicative auction price
    pub fn get_indicative_price(&self, symbol: &[u8; 12]) -> Option<IndicativePrice> {
        self.auction_engine.get_indicative_price(symbol)
    }
}

/// Multi-venue exchange control manager
pub struct ExchangeControlManager {
    controls: parking_lot::RwLock<Vec<Arc<ExchangeControl>>>,
    /// Global aggregated kill switch
    global_kill_switch: Arc<AtomicBool>,
}

impl ExchangeControlManager {
    pub fn new(venues: &[VenueId]) -> Self {
        let mut controls = Vec::with_capacity(venues.len());
        let mut global_kill: Option<Arc<AtomicBool>> = None;
        
        for &venue_id in venues {
            let control = Arc::new(ExchangeControl::new(venue_id, venues));
            if global_kill.is_none() {
                global_kill = Some(control.get_global_kill_switch());
            }
            controls.push(control);
        }
        
        Self {
            controls: parking_lot::RwLock::new(controls),
            global_kill_switch: global_kill.unwrap_or_else(|| Arc::new(AtomicBool::new(false))),
        }
    }

    /// Get control for specific venue
    pub fn get_control(&self, venue_id: VenueId) -> Option<Arc<ExchangeControl>> {
        let controls = self.controls.read();
        controls.iter().find(|c| c.venue_id == venue_id).cloned()
    }

    /// Get global kill switch
    #[inline]
    pub fn get_global_kill_switch(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.global_kill_switch)
    }

    /// Check if any venue is halted
    #[inline]
    pub fn any_venue_halted(&self) -> bool {
        self.global_kill_switch.load(Ordering::Acquire)
    }

    /// Emergency stop all venues
    pub fn emergency_stop_all(&self) {
        let controls = self.controls.read();
        for control in controls.iter() {
            control.emergency_stop();
        }
    }

    /// Refresh all control states
    pub fn refresh_all_states(&self) {
        let controls = self.controls.read();
        for control in controls.iter() {
            control.refresh_state();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exchange_control_basic() {
        let venues = vec![VenueId::Nasdaq];
        let control = ExchangeControl::new(VenueId::Nasdaq, &venues);
        
        assert!(control.is_venue_tradeable());
        assert!(!control.get_global_kill_switch().load(Ordering::Acquire));
    }

    #[test]
    fn test_manager_creation() {
        let venues = vec![VenueId::Nasdaq, VenueId::NYSE];
        let manager = ExchangeControlManager::new(&venues);
        
        assert!(manager.get_control(VenueId::Nasdaq).is_some());
        assert!(manager.get_control(VenueId::NYSE).is_some());
        assert!(!manager.any_venue_halted());
    }
}
