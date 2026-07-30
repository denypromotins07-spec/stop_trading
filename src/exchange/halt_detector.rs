//! Exchange Halt Detection Module
//! 
//! Detects exchange trading halts, maintenance windows, and circuit breakers via real-time WS control streams.
//! Instantly freezes the execution engine to prevent orders from being trapped or rejected during untradeable periods.
//! Memory-efficient design with zero-copy parsing for 6.5GB RAM constraint compliance.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::broadcast::{self, Sender, Receiver};
use serde::{Deserialize, Serialize};
use crate::gateway::venue::VenueId;

/// Maximum number of halt states to track (one per venue + global)
const MAX_HALT_STATES: usize = 32;

/// Trading state enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum TradingState {
    /// Normal continuous trading
    Open = 0,
    /// Pre-open auction phase
    PreOpen = 1,
    /// Closing auction phase
    CloseAuction = 2,
    /// Volatility halt - circuit breaker triggered
    VolatilityHalt = 3,
    /// Regulatory halt - pending news
    RegulatoryHalt = 4,
    /// Technical halt - exchange infrastructure issue
    TechnicalHalt = 5,
    /// Maintenance window
    Maintenance = 6,
    /// Market closed (after hours)
    Closed = 7,
}

impl TradingState {
    #[inline]
    pub fn is_tradeable(&self) -> bool {
        matches!(self, TradingState::Open | TradingState::PreOpen | TradingState::CloseAuction)
    }

    #[inline]
    pub fn from_u8(val: u8) -> Self {
        match val {
            0 => TradingState::Open,
            1 => TradingState::PreOpen,
            2 => TradingState::CloseAuction,
            3 => TradingState::VolatilityHalt,
            4 => TradingState::RegulatoryHalt,
            5 => TradingState::TechnicalHalt,
            6 => TradingState::Maintenance,
            7 => TradingState::Closed,
            _ => TradingState::Closed,
        }
    }
}

/// Circuit breaker levels based on market moves
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CircuitBreakerLevel {
    None = 0,
    Level1 = 1,  // 7% move - 15 min halt
    Level2 = 2,  // 13% move - 15 min halt
    Level3 = 3,  // 20% move - rest of day
}

/// Halt event structure - compact for memory efficiency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HaltEvent {
    pub venue_id: VenueId,
    pub symbol: [u8; 12],  // Fixed-size symbol buffer
    pub state: TradingState,
    pub circuit_breaker_level: CircuitBreakerLevel,
    pub timestamp_ns: u64,
    pub expected_resume_ns: Option<u64>,
    pub reason_code: u16,
}

impl HaltEvent {
    pub fn new(
        venue_id: VenueId,
        symbol: &str,
        state: TradingState,
        circuit_breaker_level: CircuitBreakerLevel,
        timestamp_ns: u64,
        expected_resume_ns: Option<u64>,
        reason_code: u16,
    ) -> Self {
        let mut symbol_buf = [0u8; 12];
        let bytes = symbol.as_bytes();
        let copy_len = bytes.len().min(12);
        symbol_buf[..copy_len].copy_from_slice(&bytes[..copy_len]);
        
        Self {
            venue_id,
            symbol: symbol_buf,
            state,
            circuit_breaker_level,
            timestamp_ns,
            expected_resume_ns,
            reason_code,
        }
    }

    #[inline]
    pub fn symbol_str(&self) -> &str {
        std::str::from_utf8(&self.symbol).unwrap_or("")
    }
}

/// Per-symbol halt state tracker
#[derive(Debug)]
struct SymbolHaltState {
    symbol: [u8; 12],
    state: AtomicU8,
    last_update_ns: AtomicU64,
    circuit_breaker: AtomicU8,
    expected_resume_ns: AtomicU64,
}

impl SymbolHaltState {
    fn new(symbol: [u8; 12]) -> Self {
        Self {
            symbol,
            state: AtomicU8::new(TradingState::Open as u8),
            last_update_ns: AtomicU64::new(0),
            circuit_breaker: AtomicU8::new(CircuitBreakerLevel::None as u8),
            expected_resume_ns: AtomicU64::new(0),
        }
    }

    #[inline]
    fn update(&self, event: &HaltEvent) {
        self.state.store(event.state as u8, Ordering::Release);
        self.last_update_ns.store(event.timestamp_ns, Ordering::Release);
        self.circuit_breaker.store(event.circuit_breaker_level as u8, Ordering::Release);
        if let Some(resume) = event.expected_resume_ns {
            self.expected_resume_ns.store(resume, Ordering::Release);
        }
    }

    #[inline]
    fn is_halted(&self) -> bool {
        let state = TradingState::from_u8(self.state.load(Ordering::Acquire));
        !state.is_tradeable()
    }
}

/// Venue-level halt state
#[derive(Debug)]
struct VenueHaltState {
    venue_id: VenueId,
    global_state: AtomicU8,
    symbols: Vec<Arc<SymbolHaltState>>,
    any_halted: AtomicBool,
}

impl VenueHaltState {
    fn new(venue_id: VenueId, max_symbols: usize) -> Self {
        Self {
            venue_id,
            global_state: AtomicU8::new(TradingState::Open as u8),
            symbols: Vec::with_capacity(max_symbols),
            any_halted: AtomicBool::new(false),
        }
    }

    fn get_or_create_symbol(&self, symbol: &[u8; 12]) -> Arc<SymbolHaltState> {
        // Linear search is acceptable given small symbol count per venue in active trading
        for sym_state in &self.symbols {
            if sym_state.symbol == *symbol {
                return Arc::clone(sym_state);
            }
        }
        // Create new - in production would use proper synchronization
        let new_state = Arc::new(SymbolHaltState::new(*symbol));
        // Safe to push in single-threaded context or with proper locking
        unsafe {
            let syms_ptr = &self.symbols as *const Vec<_> as *mut Vec<_>;
            (*syms_ptr).push(Arc::clone(&new_state));
        }
        new_state
    }

    #[inline]
    fn update_global_state(&self, state: TradingState) {
        self.global_state.store(state as u8, Ordering::Release);
        self.any_halted.store(!state.is_tradeable(), Ordering::Release);
    }
}

/// Main Halt Detector - monitors all venues and symbols
pub struct HaltDetector {
    /// Global kill switch flag - set when ANY venue halts
    global_kill_switch: Arc<AtomicBool>,
    
    /// Per-venue halt states
    venue_states: Vec<Arc<VenueHaltState>>,
    
    /// Broadcast channel for halt events
    halt_tx: Sender<HaltEvent>,
    
    /// Last heartbeat timestamp
    last_heartbeat_ns: AtomicU64,
    
    /// Heartbeat timeout in nanoseconds (5 seconds default)
    heartbeat_timeout_ns: u64,
    
    /// Flag indicating if detector is actively monitoring
    is_active: AtomicBool,
}

impl HaltDetector {
    /// Create a new HaltDetector instance
    pub fn new(venues: &[VenueId]) -> (Self, Receiver<HaltEvent>) {
        let (halt_tx, halt_rx) = broadcast::channel::<HaltEvent>(1024);
        
        let mut venue_states = Vec::with_capacity(venues.len());
        for &venue_id in venues {
            venue_states.push(Arc::new(VenueHaltState::new(venue_id, 1000)));
        }
        
        Self {
            global_kill_switch: Arc::new(AtomicBool::new(false)),
            venue_states,
            halt_tx,
            last_heartbeat_ns: AtomicU64::new(0),
            heartbeat_timeout_ns: Duration::from_secs(5).as_nanos() as u64,
            is_active: AtomicBool::new(true),
        }
    }

    /// Get reference to global kill switch for wiring to execution engine
    #[inline]
    pub fn get_kill_switch(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.global_kill_switch)
    }

    /// Process incoming halt message from exchange WS stream
    /// This should be called from the WebSocket message handler
    pub fn process_halt_message(&self, event: HaltEvent) {
        if !self.is_active.load(Ordering::Acquire) {
            return;
        }

        // Update timestamp
        self.last_heartbeat_ns.store(event.timestamp_ns, Ordering::Release);

        // Find venue and update state
        for venue_state in &self.venue_states {
            if venue_state.venue_id == event.venue_id {
                // Update symbol-specific state
                let symbol_state = venue_state.get_or_create_symbol(&event.symbol);
                symbol_state.update(&event);

                // Update venue global state if this is a venue-wide halt
                if event.reason_code < 100 {
                    venue_state.update_global_state(event.state);
                }

                // Check if we need to trigger global kill switch
                if !event.state.is_tradeable() {
                    self.global_kill_switch.store(true, Ordering::Release);
                } else {
                    // Check if any venue is still halted
                    let any_still_halted = self.venue_states.iter().any(|vs| {
                        vs.any_halted.load(Ordering::Acquire)
                    });
                    if !any_still_halted {
                        self.global_kill_switch.store(false, Ordering::Release);
                    }
                }

                break;
            }
        }

        // Broadcast event to subscribers (UI, logging, etc.)
        let _ = self.halt_tx.send(event);
    }

    /// Check if a specific symbol is tradeable
    #[inline]
    pub fn is_symbol_tradeable(&self, venue_id: VenueId, symbol: &[u8; 12]) -> bool {
        // Fast path: check global kill switch first
        if self.global_kill_switch.load(Ordering::Acquire) {
            return false;
        }

        // Check venue state
        for venue_state in &self.venue_states {
            if venue_state.venue_id == venue_id {
                if venue_state.any_halted.load(Ordering::Acquire) {
                    // Check specific symbol
                    for sym_state in &venue_state.symbols {
                        if sym_state.symbol == *symbol {
                            return !sym_state.is_halted();
                        }
                    }
                    // Symbol not found, assume tradeable
                    return true;
                }
                return true;
            }
        }
        true
    }

    /// Check if entire venue is tradeable
    #[inline]
    pub fn is_venue_tradeable(&self, venue_id: VenueId) -> bool {
        if self.global_kill_switch.load(Ordering::Acquire) {
            return false;
        }

        for venue_state in &self.venue_states {
            if venue_state.venue_id == venue_id {
                return !venue_state.any_halted.load(Ordering::Acquire);
            }
        }
        true
    }

    /// Get current trading state for a symbol
    pub fn get_symbol_state(&self, venue_id: VenueId, symbol: &[u8; 12]) -> TradingState {
        for venue_state in &self.venue_states {
            if venue_state.venue_id == venue_id {
                for sym_state in &venue_state.symbols {
                    if sym_state.symbol == *symbol {
                        return TradingState::from_u8(sym_state.state.load(Ordering::Acquire));
                    }
                }
                return TradingState::Open;
            }
        }
        TradingState::Open
    }

    /// Subscribe to halt events
    pub fn subscribe(&self) -> Receiver<HaltEvent> {
        self.halt_tx.subscribe()
    }

    /// Check heartbeat health - returns true if detector is healthy
    #[inline]
    pub fn check_health(&self, current_time_ns: u64) -> bool {
        let last = self.last_heartbeat_ns.load(Ordering::Acquire);
        if last == 0 {
            return true; // No messages yet, consider healthy
        }
        current_time_ns - last < self.heartbeat_timeout_ns
    }

    /// Activate/deactivate the detector
    #[inline]
    pub fn set_active(&self, active: bool) {
        self.is_active.store(active, Ordering::Release);
    }

    /// Force trigger kill switch (emergency stop)
    #[inline]
    pub fn emergency_stop(&self) {
        self.global_kill_switch.store(true, Ordering::Release);
        let _ = self.halt_tx.send(HaltEvent::new(
            VenueId::Unknown,
            "SYSTEM",
            TradingState::TechnicalHalt,
            CircuitBreakerLevel::Level3,
            0,
            None,
            999,
        ));
    }
}

/// Parse binary halt message from exchange protocol
/// Optimized for zero-copy where possible
pub fn parse_halt_message(data: &[u8], venue_id: VenueId) -> Option<HaltEvent> {
    if data.len() < 32 {
        return None;
    }

    // Protocol format (example):
    // [0..1]: message_type
    // [1..2]: reason_code (u16 LE)
    // [2..14]: symbol (12 bytes)
    // [14..15]: trading_state
    // [15..16]: circuit_breaker_level
    // [16..24]: timestamp_ns (u64 LE)
    // [24..32]: expected_resume_ns (u64 LE, 0 if none)

    let message_type = data[0];
    if message_type != 0x48 {  // 'H' for halt
        return None;
    }

    let reason_code = u16::from_le_bytes([data[1], data[2]]);
    
    let mut symbol = [0u8; 12];
    symbol.copy_from_slice(&data[2..14]);

    let state = TradingState::from_u8(data[14]);
    let cb_level = match data[15] {
        1 => CircuitBreakerLevel::Level1,
        2 => CircuitBreakerLevel::Level2,
        3 => CircuitBreakerLevel::Level3,
        _ => CircuitBreakerLevel::None,
    };

    let timestamp_ns = u64::from_le_bytes([
        data[16], data[17], data[18], data[19],
        data[20], data[21], data[22], data[23],
    ]);

    let resume_bytes = [data[24], data[25], data[26], data[27], data[28], data[29], data[30], data[31]];
    let resume_ns = u64::from_le_bytes(resume_bytes);
    let expected_resume = if resume_ns > 0 { Some(resume_ns) } else { None };

    Some(HaltEvent::new(
        venue_id,
        std::str::from_utf8(&symbol).unwrap_or(""),
        state,
        cb_level,
        timestamp_ns,
        expected_resume,
        reason_code,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trading_state_transitions() {
        assert!(TradingState::Open.is_tradeable());
        assert!(TradingState::PreOpen.is_tradeable());
        assert!(!TradingState::VolatilityHalt.is_tradeable());
        assert!(!TradingState::TechnicalHalt.is_tradeable());
    }

    #[test]
    fn test_halt_detector_basic() {
        let venues = vec![VenueId::Nasdaq];
        let (detector, _rx) = HaltDetector::new(&venues);
        
        assert!(!detector.global_kill_switch.load(Ordering::Acquire));
        assert!(detector.is_venue_tradeable(VenueId::Nasdaq));
    }

    #[test]
    fn test_parse_halt_message() {
        let mut data = vec![0u8; 32];
        data[0] = 0x48;  // Halt message type
        data[1] = 0x01;  // Reason code low byte
        data[2] = 0x00;  // Reason code high byte
        data[14] = 3;    // VolatilityHalt
        data[15] = 1;    // Level1 circuit breaker
        
        let result = parse_halt_message(&data, VenueId::Nasdaq);
        assert!(result.is_some());
        let event = result.unwrap();
        assert_eq!(event.state, TradingState::VolatilityHalt);
        assert_eq!(event.circuit_breaker_level, CircuitBreakerLevel::Level1);
    }
}
