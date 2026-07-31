//! Bridge Module Root
//! 
//! Manages the lifecycle of FFI callbacks and IPC shared memory mappings.
//! Provides unified interface for Python/Rust cross-language communication.

pub mod ffi_exports;
pub mod state_sync;

pub use ffi_exports::{
    FfiOrderSignal, FfiTick, FfiPortfolioDelta, FfiExecutionReport,
    FfiResult, ffi_init, ffi_shutdown, ffi_is_initialized,
    ffi_register_signal_callback, ffi_register_state_callback,
    ffi_register_execution_callback, ffi_submit_signal,
    ffi_get_portfolio_json, ffi_free_string, ffi_get_callback_count,
    push_execution_report, push_portfolio_delta,
};

pub use state_sync::{
    SharedState, SharedStateHeader, SymbolState, OpenOrderRecord,
    MAX_SYMBOLS, MAX_OPEN_ORDERS, hash_symbol,
};

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

/// Global shared state instance
static GLOBAL_SHARED_STATE: OnceLock<SharedState> = OnceLock::new();

/// Bridge initialization flag
static BRIDGE_INITIALIZED: AtomicBool = AtomicBool::new(false);

/// Initialize the bridge subsystem
pub fn init_bridge() -> Result<(), &'static str> {
    if BRIDGE_INITIALIZED.load(Ordering::SeqCst) {
        return Err("Bridge already initialized");
    }
    
    // Initialize FFI layer
    unsafe {
        let result = ffi_init();
        if result != FfiResult::Success {
            return Err("Failed to initialize FFI layer");
        }
    }
    
    // Initialize shared state
    GLOBAL_SHARED_STATE
        .set(SharedState::new())
        .map_err(|_| "Failed to set global shared state")?;
    
    BRIDGE_INITIALIZED.store(true, Ordering::SeqCst);
    Ok(())
}

/// Shutdown the bridge subsystem
pub fn shutdown_bridge() -> Result<(), &'static str> {
    if !BRIDGE_INITIALIZED.load(Ordering::SeqCst) {
        return Err("Bridge not initialized");
    }
    
    unsafe {
        let result = ffi_shutdown();
        if result != FfiResult::Success {
            return Err("Failed to shutdown FFI layer");
        }
    }
    
    BRIDGE_INITIALIZED.store(false, Ordering::SeqCst);
    Ok(())
}

/// Check if bridge is initialized
#[inline]
pub fn is_bridge_initialized() -> bool {
    BRIDGE_INITIALIZED.load(Ordering::Relaxed)
}

/// Get reference to global shared state
pub fn get_shared_state() -> Result<&'static SharedState, &'static str> {
    GLOBAL_SHARED_STATE
        .get()
        .ok_or("Shared state not initialized")
}

/// Enable trading through the bridge
pub fn enable_trading() -> Result<(), &'static str> {
    let state = get_shared_state()?;
    state.set_trading_enabled(true);
    Ok(())
}

/// Disable trading through the bridge
pub fn disable_trading() -> Result<(), &'static str> {
    let state = get_shared_state()?;
    state.set_trading_enabled(false);
    Ok(())
}

/// Trigger emergency stop
pub fn trigger_emergency_stop() -> Result<(), &'static str> {
    let state = get_shared_state()?;
    state.emergency_stop();
    Ok(())
}

/// Check if trading is enabled
pub fn is_trading_enabled() -> bool {
    get_shared_state()
        .map(|s| s.is_trading_enabled())
        .unwrap_or(false)
}

/// Check if emergency stop is active
pub fn is_emergency_stop() -> bool {
    get_shared_state()
        .map(|s| s.is_emergency_stop())
        .unwrap_or(false)
}

/// Submit order signal from external source
pub fn submit_order_signal(signal: FfiOrderSignal) -> FfiResult {
    if !is_bridge_initialized() {
        return FfiResult::ErrorNotInitialized;
    }
    
    unsafe {
        ffi_submit_signal(&signal)
    }
}

/// Push execution report to Python consumers
pub fn broadcast_execution(report: FfiExecutionReport) {
    push_execution_report(&report);
}

/// Push portfolio delta to Python consumers
pub fn broadcast_portfolio_delta(delta: FfiPortfolioDelta) {
    push_portfolio_delta(&delta);
}

/// Get current callback count for monitoring
pub fn get_callback_stats() -> u64 {
    ffi_get_callback_count()
}

/// Bridge builder for fluent initialization
pub struct BridgeBuilder {
    enable_signals: bool,
    enable_state: bool,
    enable_executions: bool,
}

impl BridgeBuilder {
    pub fn new() -> Self {
        Self {
            enable_signals: true,
            enable_state: true,
            enable_executions: true,
        }
    }
    
    pub fn with_signals(mut self, enabled: bool) -> Self {
        self.enable_signals = enabled;
        self
    }
    
    pub fn with_state_sync(mut self, enabled: bool) -> Self {
        self.enable_state = enabled;
        self
    }
    
    pub fn with_executions(mut self, enabled: bool) -> Self {
        self.enable_executions = enabled;
        self
    }
    
    pub fn build(self) -> Result<(), &'static str> {
        init_bridge()?;
        
        // Note: Callback registration would happen here if Python callbacks were provided
        // In practice, Python calls ffi_register_*_callback directly
        
        Ok(())
    }
}

impl Default for BridgeBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_bridge_lifecycle() {
        assert!(!is_bridge_initialized());
        
        assert!(init_bridge().is_ok());
        assert!(is_bridge_initialized());
        
        assert!(shutdown_bridge().is_ok());
        assert!(!is_bridge_initialized());
    }
    
    #[test]
    fn test_double_init_prevention() {
        assert!(init_bridge().is_ok());
        assert!(init_bridge().is_err()); // Should fail
        
        assert!(shutdown_bridge().is_ok());
    }
    
    #[test]
    fn test_trading_controls() {
        init_bridge().unwrap();
        
        assert!(!is_trading_enabled());
        
        enable_trading().unwrap();
        assert!(is_trading_enabled());
        
        disable_trading().unwrap();
        assert!(!is_trading_enabled());
        
        shutdown_bridge().unwrap();
    }
    
    #[test]
    fn test_emergency_stop() {
        init_bridge().unwrap();
        
        enable_trading().unwrap();
        assert!(is_trading_enabled());
        
        trigger_emergency_stop().unwrap();
        assert!(!is_trading_enabled());
        assert!(is_emergency_stop());
        
        shutdown_bridge().unwrap();
    }
    
    #[test]
    fn test_shared_state_access() {
        init_bridge().unwrap();
        
        let state = get_shared_state().unwrap();
        assert_eq!(state.get_sequence(), 0);
        
        shutdown_bridge().unwrap();
    }
    
    #[test]
    fn test_builder_pattern() {
        let result = BridgeBuilder::new()
            .with_signals(true)
            .with_state_sync(true)
            .build();
        
        assert!(result.is_ok());
        shutdown_bridge().unwrap();
    }
}
