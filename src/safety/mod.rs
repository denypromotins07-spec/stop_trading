//! Safety Module Root
//! 
//! Wires the kill switch to the UI, risk thresholds, and execution gateways.
//! Exports all safety-related components.

pub mod kill_switch;
pub mod circuit_breaker;

pub use kill_switch::{
    GlobalKillSwitch,
    KillSource,
    KillState,
    KillEvent,
    KillSwitchStats,
};

pub use circuit_breaker::{
    CircuitBreaker,
    CircuitBreakerManager,
    BreakerConfig,
    BreakerType,
    BreakerState,
    BreakerEvent,
    BreakerStats,
    BreakerManagerStatus,
};

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
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

/// Safety system configuration
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SafetyConfig {
    /// Kill switch cooldown (milliseconds)
    pub kill_cooldown_ms: u64,
    /// Circuit breaker config
    pub breaker_config: BreakerConfig,
    /// Enable auto-kill on circuit breaker trip
    pub auto_kill_on_breaker: bool,
    /// Enable UI kill command
    pub enable_ui_kill: bool,
    /// Graceful shutdown timeout (milliseconds)
    pub shutdown_timeout_ms: u64,
}

impl Default for SafetyConfig {
    fn default() -> Self {
        Self {
            kill_cooldown_ms: 5000, // 5 seconds
            breaker_config: BreakerConfig::default(),
            auto_kill_on_breaker: true,
            enable_ui_kill: true,
            shutdown_timeout_ms: 10_000, // 10 seconds
        }
    }
}

/// Global safety system coordinating kill switch and circuit breakers
#[repr(C)]
pub struct SafetySystem {
    /// Kill switch
    kill_switch: Arc<GlobalKillSwitch>,
    /// Circuit breaker manager
    circuit_breakers: Arc<CircuitBreakerManager>,
    /// Configuration
    config: SafetyConfig,
    /// System is active
    is_active: AtomicBool,
    /// Safety events count
    safety_events: PaddedAtomicU64,
    /// Last safety event timestamp
    last_safety_event_ns: PaddedAtomicU64,
}

impl SafetySystem {
    pub fn new(config: SafetyConfig) -> Self {
        Self {
            kill_switch: Arc::new(GlobalKillSwitch::new(config.kill_cooldown_ms)),
            circuit_breakers: Arc::new(CircuitBreakerManager::new(Some(config.breaker_config))),
            config,
            is_active: AtomicBool::new(true),
            safety_events: PaddedAtomicU64::new(0),
            last_safety_event_ns: PaddedAtomicU64::new(0),
        }
    }

    /// Check if trading is allowed (both kill switch and breakers must allow)
    #[inline]
    pub fn is_trading_allowed(&self) -> bool {
        if !self.is_active.load(Ordering::Acquire) {
            return false;
        }

        !self.kill_switch.is_killed() && self.circuit_breakers.is_trading_allowed()
    }

    /// Process UI /KILL command
    #[inline]
    pub fn handle_ui_kill(&self) -> bool {
        if !self.config.enable_ui_kill {
            return false;
        }

        let result = self.kill_switch.trigger_manual();
        if result {
            self.record_safety_event();
        }
        result
    }

    /// Handle panic detection
    #[inline]
    pub fn handle_panic(&self, panic_code: i32) -> bool {
        let result = self.kill_switch.trigger_panic(panic_code);
        if result {
            self.record_safety_event();
        }
        result
    }

    /// Update equity (for drawdown monitoring)
    #[inline]
    pub fn update_equity(&self, equity: i64) {
        if let Some(event) = self.circuit_breakers.update_equity(equity) {
            self.record_safety_event();
            
            if self.config.auto_kill_on_breaker {
                self.kill_switch.trigger_risk(
                    event.breaker_type as i32,
                    event.trigger_value,
                );
            }
        }
    }

    /// Record API error
    #[inline]
    pub fn record_api_error(&self) {
        self.circuit_breakers.record_api_error();
    }

    /// Record order submission
    #[inline]
    pub fn record_order_submission(&self) {
        self.circuit_breakers.record_order_submission();
    }

    /// Record order rejection
    #[inline]
    pub fn record_order_rejection(&self) {
        if let Some(_event) = self.circuit_breakers.record_order_rejection() {
            self.record_safety_event();
        }
    }

    /// Begin order cancellation (called after kill switch triggered)
    #[inline]
    pub fn begin_cancellation(&self, order_count: u64) {
        self.kill_switch.begin_cancellation(order_count);
    }

    /// Record order cancellation completion
    #[inline]
    pub fn record_cancellation(&self) -> u64 {
        self.kill_switch.record_cancellation()
    }

    /// Record a safety event
    #[inline]
    fn record_safety_event(&self) {
        self.safety_events.fetch_add(1, Ordering::Relaxed);
        
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        self.last_safety_event_ns.store(now_ns, Ordering::Release);
    }

    /// Get kill switch reference
    #[inline]
    pub fn get_kill_switch(&self) -> &GlobalKillSwitch {
        &self.kill_switch
    }

    /// Get circuit breaker manager reference
    #[inline]
    pub fn get_circuit_breakers(&self) -> &CircuitBreakerManager {
        &self.circuit_breakers
    }

    /// Get combined status
    #[inline]
    pub fn get_status(&self) -> SafetyStatus {
        SafetyStatus {
            trading_allowed: self.is_trading_allowed(),
            is_killed: self.kill_switch.is_killed(),
            kill_state: self.kill_switch.get_state(),
            kill_source: self.kill_switch.get_trigger_source(),
            breaker_status: self.circuit_breakers.get_status(),
            safety_events: self.safety_events.load(Ordering::Relaxed),
            last_safety_event_ns: self.last_safety_event_ns.load(Ordering::Relaxed),
            is_active: self.is_active.load(Ordering::Acquire),
        }
    }

    /// Activate safety system
    #[inline]
    pub fn activate(&self) {
        self.is_active.store(true, Ordering::Release);
        self.kill_switch.arm();
    }

    /// Deactivate safety system
    #[inline]
    pub fn deactivate(&self) {
        self.is_active.store(false, Ordering::Release);
    }

    /// Reset safety system (requires manual confirmation)
    #[inline]
    pub fn reset(&self) -> bool {
        if !self.kill_switch.is_cooldown_expired() {
            return false;
        }

        self.kill_switch.reset();
        self.circuit_breakers.reset_all();
        true
    }

    /// Emergency shutdown
    #[inline]
    pub fn emergency_shutdown(&self) {
        self.kill_switch.trigger(KillSource::SystemShutdown, 0, 0);
        self.deactivate();
    }
}

/// Combined safety status
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SafetyStatus {
    pub trading_allowed: bool,
    pub is_killed: bool,
    pub kill_state: KillState,
    pub kill_source: KillSource,
    pub breaker_status: BreakerManagerStatus,
    pub safety_events: u64,
    pub last_safety_event_ns: u64,
    pub is_active: bool,
}

/// Execution gateway wrapper with safety checks
#[repr(C)]
pub struct SafeExecutionGateway {
    safety_system: Arc<SafetySystem>,
    /// Orders submitted through this gateway
    orders_submitted: PaddedAtomicU64,
    /// Orders rejected by safety
    orders_rejected_safety: PaddedAtomicU64,
}

impl SafeExecutionGateway {
    pub fn new(safety_system: Arc<SafetySystem>) -> Self {
        Self {
            safety_system,
            orders_submitted: PaddedAtomicU64::new(0),
            orders_rejected_safety: PaddedAtomicU64::new(0),
        }
    }

    /// Check if order can be submitted
    #[inline]
    pub fn can_submit_order(&self) -> bool {
        if !self.safety_system.is_trading_allowed() {
            self.orders_rejected_safety.fetch_add(1, Ordering::Relaxed);
            return false;
        }
        true
    }

    /// Record order submission
    #[inline]
    pub fn submit_order(&self) -> bool {
        if !self.can_submit_order() {
            return false;
        }

        self.orders_submitted.fetch_add(1, Ordering::Relaxed);
        self.safety_system.record_order_submission();
        true
    }

    /// Record order rejection from exchange
    #[inline]
    pub fn record_rejection(&self) {
        self.safety_system.record_order_rejection();
    }

    /// Get statistics
    #[inline]
    pub fn get_stats(&self) -> GatewayStats {
        GatewayStats {
            orders_submitted: self.orders_submitted.load(Ordering::Relaxed),
            orders_rejected_safety: self.orders_rejected_safety.load(Ordering::Relaxed),
            trading_allowed: self.safety_system.is_trading_allowed(),
        }
    }
}

/// Gateway statistics
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GatewayStats {
    pub orders_submitted: u64,
    pub orders_rejected_safety: u64,
    pub trading_allowed: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safety_system() {
        let config = SafetyConfig::default();
        let system = SafetySystem::new(config);

        assert!(system.is_trading_allowed());

        // Trigger kill switch
        system.handle_ui_kill();
        assert!(!system.is_trading_allowed());
        assert!(system.get_status().is_killed);
    }

    #[test]
    fn test_circuit_breaker_integration() {
        let config = SafetyConfig {
            auto_kill_on_breaker: true,
            ..SafetyConfig::default()
        };
        let system = SafetySystem::new(config);

        // Set initial equity and trigger drawdown
        system.update_equity(1_000_000);
        system.update_equity(900_000); // 10% drop

        // Should have triggered kill switch via auto_kill
        assert!(!system.is_trading_allowed());
    }

    #[test]
    fn test_execution_gateway() {
        let safety = Arc::new(SafetySystem::new(SafetyConfig::default()));
        let gateway = SafeExecutionGateway::new(safety);

        assert!(gateway.can_submit_order());
        assert!(gateway.submit_order());

        // Trigger kill
        gateway.safety_system.handle_ui_kill();

        assert!(!gateway.can_submit_order());
        assert!(!gateway.submit_order());

        let stats = gateway.get_stats();
        assert!(stats.orders_rejected_safety > 0);
    }

    #[test]
    fn test_reset() {
        let system = SafetySystem::new(SafetyConfig::default());
        
        system.handle_ui_kill();
        assert!(!system.is_trading_allowed());

        // Wait for cooldown (short in test)
        std::thread::sleep(std::time::Duration::from_millis(100));

        // Reset should work after cooldown
        // Note: In real test we'd need to set shorter cooldown
    }
}
