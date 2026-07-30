//! Global Kill Switch Implementation
//! 
//! Master "Global Kill Switch" triggered by Terminal UI `/KILL` command or internal panics.
//! Instantly broadcasts high-priority interrupt to all actors to cancel open orders and halt new order generation.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicI64, Ordering};
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

/// Kill switch trigger source
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KillSource {
    ManualUI,           // User triggered via /KILL command
    PanicDetected,      // Internal panic caught
    RiskThreshold,      // Risk limit exceeded
    CircuitBreaker,     // Circuit breaker triggered
    NetworkFailure,     // Critical network issue
    ExchangeDisconnect, // Exchange disconnected
    HeartbeatFailure,   // Heartbeat monitor failure
    MarginCall,         // Margin call received
    PositionLimit,      // Position limit breached
    DrawdownLimit,      // Maximum drawdown reached
    SystemShutdown,     // Graceful system shutdown
}

/// Kill switch state
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KillState {
    Armed,          // Ready to be triggered
    Triggered,      // Kill signal sent
    Cancelling,     // Orders being cancelled
    Cancelled,      // All orders cancelled
    Halted,         // Trading halted
    Restarting,     // Preparing for restart
    Ready,          // Ready to resume trading
}

/// Kill switch event record
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct KillEvent {
    /// Trigger source
    pub source: KillSource,
    /// Timestamp (ns)
    pub timestamp_ns: u64,
    /// Message/error code
    pub error_code: i32,
    /// Additional data
    pub data: u64,
}

impl KillEvent {
    pub fn new(source: KillSource, error_code: i32, data: u64) -> Self {
        let timestamp_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        
        Self {
            source,
            timestamp_ns,
            error_code,
            data,
        }
    }
}

/// Global Kill Switch
#[repr(C)]
pub struct GlobalKillSwitch {
    /// Kill flag (true = killed)
    kill_flag: AtomicBool,
    /// Current state
    state: AtomicU64, // Encoded KillState
    /// Trigger source
    trigger_source: AtomicU64, // Encoded KillSource
    /// Kill event count
    kill_count: PaddedAtomicU64,
    /// Last kill timestamp
    last_kill_ns: PaddedAtomicU64,
    /// Orders pending cancellation
    orders_pending: PaddedAtomicU64,
    /// Orders cancelled
    orders_cancelled: PaddedAtomicU64,
    /// Switch is armed
    is_armed: AtomicBool,
    /// Allow restart
    allow_restart: AtomicBool,
    /// Cooldown period (ns)
    cooldown_ns: PaddedAtomicU64,
    /// Cooldown start
    cooldown_start_ns: PaddedAtomicU64,
}

impl GlobalKillSwitch {
    pub fn new(cooldown_ms: u64) -> Self {
        Self {
            kill_flag: AtomicBool::new(false),
            state: AtomicU64::new(KillState::Armed as u64),
            trigger_source: AtomicU64::new(0),
            kill_count: PaddedAtomicU64::new(0),
            last_kill_ns: PaddedAtomicU64::new(0),
            orders_pending: PaddedAtomicU64::new(0),
            orders_cancelled: PaddedAtomicU64::new(0),
            is_armed: AtomicBool::new(true),
            allow_restart: AtomicBool::new(false),
            cooldown_ns: PaddedAtomicU64::new(cooldown_ms * 1_000_000),
            cooldown_start_ns: PaddedAtomicU64::new(0),
        }
    }

    /// Trigger the kill switch
    #[inline]
    pub fn trigger(&self, source: KillSource, error_code: i32, data: u64) -> bool {
        if !self.is_armed.load(Ordering::Acquire) {
            return false;
        }

        // Already killed?
        if self.kill_flag.load(Ordering::Acquire) {
            return false;
        }

        // Set kill flag atomically
        self.kill_flag.store(true, Ordering::Release);
        
        // Record event
        let event = KillEvent::new(source, error_code, data);
        
        // Update state
        self.state.store(KillState::Triggered as u64, Ordering::Release);
        self.trigger_source.store(source as u64, Ordering::Release);
        
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        
        self.last_kill_ns.store(now_ns, Ordering::Release);
        self.kill_count.fetch_add(1, Ordering::Relaxed);
        self.cooldown_start_ns.store(now_ns, Ordering::Release);

        true
    }

    /// Trigger from UI /KILL command
    #[inline]
    pub fn trigger_manual(&self) -> bool {
        self.trigger(KillSource::ManualUI, 0, 0)
    }

    /// Trigger from panic
    #[inline]
    pub fn trigger_panic(&self, panic_code: i32) -> bool {
        self.trigger(KillSource::PanicDetected, panic_code, 0)
    }

    /// Trigger from risk threshold
    #[inline]
    pub fn trigger_risk(&self, risk_code: i32, data: u64) -> bool {
        self.trigger(KillSource::RiskThreshold, risk_code, data)
    }

    /// Begin order cancellation phase
    #[inline]
    pub fn begin_cancellation(&self, order_count: u64) {
        self.state.store(KillState::Cancelling as u64, Ordering::Release);
        self.orders_pending.store(order_count, Ordering::Release);
    }

    /// Record an order cancellation
    #[inline]
    pub fn record_cancellation(&self) -> u64 {
        let remaining = self.orders_pending.fetch_sub(1, Ordering::AcqRel).saturating_sub(1);
        self.orders_cancelled.fetch_add(1, Ordering::Relaxed);
        
        if remaining == 0 {
            self.state.store(KillState::Cancelled as u64, Ordering::Release);
        }
        
        remaining
    }

    /// Mark trading as halted
    #[inline]
    pub fn halt_trading(&self) {
        self.state.store(KillState::Halted as u64, Ordering::Release);
    }

    /// Check if killed
    #[inline]
    pub fn is_killed(&self) -> bool {
        self.kill_flag.load(Ordering::Acquire)
    }

    /// Get current state
    #[inline]
    pub fn get_state(&self) -> KillState {
        match self.state.load(Ordering::Acquire) {
            0 => KillState::Armed,
            1 => KillState::Triggered,
            2 => KillState::Cancelling,
            3 => KillState::Cancelled,
            4 => KillState::Halted,
            5 => KillState::Restarting,
            6 => KillState::Ready,
            _ => KillState::Halted,
        }
    }

    /// Get trigger source
    #[inline]
    pub fn get_trigger_source(&self) -> KillSource {
        match self.trigger_source.load(Ordering::Acquire) {
            0 => KillSource::ManualUI,
            1 => KillSource::PanicDetected,
            2 => KillSource::RiskThreshold,
            3 => KillSource::CircuitBreaker,
            4 => KillSource::NetworkFailure,
            5 => KillSource::ExchangeDisconnect,
            6 => KillSource::HeartbeatFailure,
            7 => KillSource::MarginCall,
            8 => KillSource::PositionLimit,
            9 => KillSource::DrawdownLimit,
            10 => KillSource::SystemShutdown,
            _ => KillSource::ManualUI,
        }
    }

    /// Check if cooldown has expired
    #[inline]
    pub fn is_cooldown_expired(&self) -> bool {
        let cooldown = self.cooldown_ns.load(Ordering::Acquire);
        let start = self.cooldown_start_ns.load(Ordering::Acquire);
        
        if cooldown == 0 || start == 0 {
            return true;
        }
        
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        
        now_ns.saturating_sub(start) >= cooldown
    }

    /// Prepare for restart
    #[inline]
    pub fn prepare_restart(&self) -> bool {
        if !self.is_cooldown_expired() {
            return false;
        }
        
        self.state.store(KillState::Restarting as u64, Ordering::Release);
        true
    }

    /// Reset and arm the kill switch
    #[inline]
    pub fn reset(&self) {
        self.kill_flag.store(false, Ordering::Release);
        self.state.store(KillState::Armed as u64, Ordering::Release);
        self.trigger_source.store(0, Ordering::Release);
        self.orders_pending.store(0, Ordering::Release);
        self.allow_restart.store(false, Ordering::Release);
    }

    /// Arm the kill switch
    #[inline]
    pub fn arm(&self) {
        self.is_armed.store(true, Ordering::Release);
        self.reset();
    }

    /// Disarm the kill switch (dangerous!)
    #[inline]
    pub fn disarm(&self) {
        self.is_armed.store(false, Ordering::Release);
    }

    /// Get statistics
    #[inline]
    pub fn get_stats(&self) -> KillSwitchStats {
        KillSwitchStats {
            is_killed: self.is_killed(),
            state: self.get_state(),
            trigger_source: self.get_trigger_source(),
            kill_count: self.kill_count.load(Ordering::Relaxed),
            last_kill_ns: self.last_kill_ns.load(Ordering::Relaxed),
            orders_pending: self.orders_pending.load(Ordering::Relaxed),
            orders_cancelled: self.orders_cancelled.load(Ordering::Relaxed),
            is_armed: self.is_armed.load(Ordering::Acquire),
            cooldown_remaining_ns: self.get_cooldown_remaining(),
        }
    }

    /// Get remaining cooldown time
    #[inline]
    fn get_cooldown_remaining(&self) -> u64 {
        let cooldown = self.cooldown_ns.load(Ordering::Acquire);
        let start = self.cooldown_start_ns.load(Ordering::Acquire);
        
        if cooldown == 0 || start == 0 {
            return 0;
        }
        
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        
        let elapsed = now_ns.saturating_sub(start);
        if elapsed >= cooldown {
            0
        } else {
            cooldown - elapsed
        }
    }

    /// Set cooldown period
    #[inline]
    pub fn set_cooldown_ms(&self, ms: u64) {
        self.cooldown_ns.store(ms * 1_000_000, Ordering::Release);
    }
}

/// Kill switch statistics
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct KillSwitchStats {
    pub is_killed: bool,
    pub state: KillState,
    pub trigger_source: KillSource,
    pub kill_count: u64,
    pub last_kill_ns: u64,
    pub orders_pending: u64,
    pub orders_cancelled: u64,
    pub is_armed: bool,
    pub cooldown_remaining_ns: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kill_switch_trigger() {
        let ks = GlobalKillSwitch::new(5000); // 5 second cooldown
        
        assert!(!ks.is_killed());
        assert_eq!(ks.get_state(), KillState::Armed);
        
        // Trigger kill
        assert!(ks.trigger_manual());
        assert!(ks.is_killed());
        assert_eq!(ks.get_state(), KillState::Triggered);
        assert_eq!(ks.get_trigger_source(), KillSource::ManualUI);
    }

    #[test]
    fn test_cancellation_flow() {
        let ks = GlobalKillSwitch::new(1000);
        
        ks.trigger_manual();
        ks.begin_cancellation(5);
        
        assert_eq!(ks.get_state(), KillState::Cancelling);
        assert_eq!(ks.orders_pending.load(Ordering::Relaxed), 5);
        
        // Cancel orders one by one
        for i in (0..5).rev() {
            let remaining = ks.record_cancellation();
            assert_eq!(remaining, i as u64);
        }
        
        assert_eq!(ks.get_state(), KillState::Cancelled);
        assert_eq!(ks.orders_cancelled.load(Ordering::Relaxed), 5);
    }

    #[test]
    fn test_cooldown() {
        let ks = GlobalKillSwitch::new(100); // 100ms cooldown
        
        ks.trigger_manual();
        assert!(!ks.is_cooldown_expired());
        
        std::thread::sleep(std::time::Duration::from_millis(150));
        assert!(ks.is_cooldown_expired());
        
        assert!(ks.prepare_restart());
    }

    #[test]
    fn test_reset() {
        let ks = GlobalKillSwitch::new(1000);
        
        ks.trigger_manual();
        assert!(ks.is_killed());
        
        ks.reset();
        assert!(!ks.is_killed());
        assert_eq!(ks.get_state(), KillState::Armed);
    }
}
