//! Lifecycle Manager
//! 
//! Manages strict state machine transitions: `Idle` -> `Warmup` -> `Live` -> `Shutdown`.
//! Intercepts `/KILL` or `Ctrl+C` signals, prompting "Yes/No" UI confirmation before graceful teardown.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use thiserror::Error;

/// Lifecycle states (strict state machine)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum LifecycleState {
    Idle = 0,
    Initializing = 1,
    Warmup = 2,
    HotStandby = 3,
    Live = 4,
    Paused = 5,
    ShuttingDown = 6,
    Terminated = 7,
}

impl LifecycleState {
    /// Check if state allows trading
    pub fn can_trade(&self) -> bool {
        matches!(self, LifecycleState::Live)
    }
    
    /// Check if state is terminal
    pub fn is_terminal(&self) -> bool {
        matches!(self, LifecycleState::Terminated)
    }
    
    /// Get string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            LifecycleState::Idle => "Idle",
            LifecycleState::Initializing => "Initializing",
            LifecycleState::Warmup => "Warmup",
            LifecycleState::HotStandby => "HotStandby",
            LifecycleState::Live => "Live",
            LifecycleState::Paused => "Paused",
            LifecycleState::ShuttingDown => "ShuttingDown",
            LifecycleState::Terminated => "Terminated",
        }
    }
}

#[derive(Error, Debug)]
pub enum LifecycleError {
    #[error("Invalid state transition from {from:?} to {to:?}")]
    InvalidTransition { from: LifecycleState, to: LifecycleState },
    
    #[error("System already in terminal state")]
    TerminalState,
    
    #[error("Shutdown confirmation required")]
    ConfirmationRequired,
    
    #[error("Timeout during state transition")]
    Timeout,
}

/// Shutdown confirmation result
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShutdownConfirm {
    Pending,
    Confirmed,
    Cancelled,
}

/// Lifecycle statistics
#[derive(Debug, Clone)]
pub struct LifecycleStats {
    pub state_transitions: u64,
    pub uptime_ns: u64,
    pub last_transition_ns: u64,
    pub current_state: LifecycleState,
}

/// Main lifecycle manager
pub struct LifecycleManager {
    /// Current state
    current_state: std::sync::Mutex<LifecycleState>,
    /// Start time
    start_time: std::sync::Mutex<Option<Instant>>,
    /// State transition count
    transition_count: AtomicU64,
    /// Last transition time
    last_transition_ns: AtomicU64,
    /// Shutdown confirmation status
    shutdown_confirm: std::sync::Mutex<ShutdownConfirm>,
    /// Force shutdown flag (bypasses confirmation)
    force_shutdown: AtomicBool,
    /// Pause capability enabled
    pause_enabled: AtomicBool,
}

unsafe impl Send for LifecycleManager {}
unsafe impl Sync for LifecycleManager {}

impl LifecycleManager {
    /// Create new lifecycle manager
    pub fn new() -> Self {
        Self {
            current_state: std::sync::Mutex::new(LifecycleState::Idle),
            start_time: std::sync::Mutex::new(None),
            transition_count: AtomicU64::new(0),
            last_transition_ns: AtomicU64::new(0),
            shutdown_confirm: std::sync::Mutex::new(ShutdownConfirm::Pending),
            force_shutdown: AtomicBool::new(false),
            pause_enabled: AtomicBool::new(true),
        }
    }
    
    /// Get current timestamp in nanoseconds
    fn now_ns() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64
    }
    
    /// Validate state transition
    fn validate_transition(from: LifecycleState, to: LifecycleState) -> Result<(), LifecycleError> {
        // Define valid transitions
        let valid = match (from, to) {
            (LifecycleState::Idle, LifecycleState::Initializing) => true,
            (LifecycleState::Initializing, LifecycleState::Warmup) => true,
            (LifecycleState::Warmup, LifecycleState::HotStandby) => true,
            (LifecycleState::HotStandby, LifecycleState::Live) => true,
            (LifecycleState::HotStandby, LifecycleState::ShuttingDown) => true,
            (LifecycleState::Live, LifecycleState::Paused) => true,
            (LifecycleState::Live, LifecycleState::ShuttingDown) => true,
            (LifecycleState::Paused, LifecycleState::Live) => true,
            (LifecycleState::Paused, LifecycleState::ShuttingDown) => true,
            (LifecycleState::ShuttingDown, LifecycleState::Terminated) => true,
            _ => false,
        };
        
        if valid {
            Ok(())
        } else {
            Err(LifecycleError::InvalidTransition { from, to })
        }
    }
    
    /// Transition to new state
    pub fn transition_to(&self, target: LifecycleState) -> Result<(), LifecycleError> {
        let mut state_guard = self.current_state.lock().unwrap();
        let current = *state_guard;
        
        // Check if already in terminal state
        if current.is_terminal() {
            return Err(LifecycleError::TerminalState);
        }
        
        // Validate transition
        Self::validate_transition(current, target)?;
        
        // Record transition
        self.transition_count.fetch_add(1, Ordering::Relaxed);
        self.last_transition_ns.store(Self::now_ns(), Ordering::Relaxed);
        
        // Update state
        *state_guard = target;
        
        // Record start time on first transition
        if current == LifecycleState::Idle {
            *self.start_time.lock().unwrap() = Some(Instant::now());
        }
        
        Ok(())
    }
    
    /// Get current state
    pub fn get_state(&self) -> LifecycleState {
        *self.current_state.lock().unwrap()
    }
    
    /// Request shutdown with confirmation
    pub fn request_shutdown(&self) -> Result<(), LifecycleError> {
        let current = self.get_state();
        
        if current.is_terminal() {
            return Err(LifecycleError::TerminalState);
        }
        
        if self.force_shutdown.load(Ordering::Relaxed) {
            // Force shutdown bypasses confirmation
            return self.transition_to(LifecycleState::ShuttingDown);
        }
        
        // Set pending confirmation
        *self.shutdown_confirm.lock().unwrap() = ShutdownConfirm::Pending;
        
        Err(LifecycleError::ConfirmationRequired)
    }
    
    /// Confirm shutdown
    pub fn confirm_shutdown(&self, confirmed: bool) -> Result<(), LifecycleError> {
        let mut confirm_guard = self.shutdown_confirm.lock().unwrap();
        
        *confirm_guard = if confirmed {
            ShutdownConfirm::Confirmed
        } else {
            ShutdownConfirm::Cancelled
        };
        
        if confirmed {
            drop(confirm_guard);
            self.transition_to(LifecycleState::ShuttingDown)
        } else {
            Ok(())
        }
    }
    
    /// Force immediate shutdown (bypasses confirmation)
    pub fn force_shutdown(&self) -> Result<(), LifecycleError> {
        self.force_shutdown.store(true, Ordering::SeqCst);
        self.request_shutdown()
    }
    
    /// Pause trading (only from Live state)
    pub fn pause(&self) -> Result<(), LifecycleError> {
        if !self.pause_enabled.load(Ordering::Relaxed) {
            return Err(LifecycleError::InvalidTransition {
                from: self.get_state(),
                to: LifecycleState::Paused,
            });
        }
        self.transition_to(LifecycleState::Paused)
    }
    
    /// Resume trading (from Paused state)
    pub fn resume(&self) -> Result<(), LifecycleError> {
        self.transition_to(LifecycleState::Live)
    }
    
    /// Check if system can trade
    pub fn can_trade(&self) -> bool {
        self.get_state().can_trade()
    }
    
    /// Check if system is running (not terminated)
    pub fn is_running(&self) -> bool {
        !self.get_state().is_terminal()
    }
    
    /// Get lifecycle statistics
    pub fn get_stats(&self) -> LifecycleStats {
        let start = *self.start_time.lock().unwrap();
        let uptime_ns = start
            .map(|s| s.elapsed().as_nanos() as u64)
            .unwrap_or(0);
        
        LifecycleStats {
            state_transitions: self.transition_count.load(Ordering::Relaxed),
            uptime_ns,
            last_transition_ns: self.last_transition_ns.load(Ordering::Relaxed),
            current_state: self.get_state(),
        }
    }
    
    /// Enable/disable pause capability
    pub fn set_pause_enabled(&self, enabled: bool) {
        self.pause_enabled.store(enabled, Ordering::Relaxed);
    }
    
    /// Reset to idle (for testing only)
    pub fn reset(&self) {
        *self.current_state.lock().unwrap() = LifecycleState::Idle;
        *self.start_time.lock().unwrap() = None;
        *self.shutdown_confirm.lock().unwrap() = ShutdownConfirm::Pending;
        self.force_shutdown.store(false, Ordering::Relaxed);
        self.transition_count.store(0, Ordering::Relaxed);
        self.last_transition_ns.store(0, Ordering::Relaxed);
    }
}

impl Default for LifecycleManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Global lifecycle manager instance
static GLOBAL_LIFECYCLE: std::sync::OnceLock<Arc<LifecycleManager>> = std::sync::OnceLock::new();

/// Initialize global lifecycle manager
pub fn init_lifecycle() -> Result<Arc<LifecycleManager>, &'static str> {
    let manager = Arc::new(LifecycleManager::new());
    GLOBAL_LIFECYCLE
        .set(manager.clone())
        .map_err(|_| "Lifecycle already initialized")?;
    Ok(manager)
}

/// Get reference to global lifecycle manager
pub fn get_lifecycle() -> Option<Arc<LifecycleManager>> {
    GLOBAL_LIFECYCLE.get().cloned()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_lifecycle_creation() {
        let lm = LifecycleManager::new();
        assert_eq!(lm.get_state(), LifecycleState::Idle);
        assert!(!lm.can_trade());
        assert!(lm.is_running());
    }
    
    #[test]
    fn test_valid_transitions() {
        let lm = LifecycleManager::new();
        
        assert!(lm.transition_to(LifecycleState::Initializing).is_ok());
        assert_eq!(lm.get_state(), LifecycleState::Initializing);
        
        assert!(lm.transition_to(LifecycleState::Warmup).is_ok());
        assert_eq!(lm.get_state(), LifecycleState::Warmup);
        
        assert!(lm.transition_to(LifecycleState::HotStandby).is_ok());
        assert!(lm.transition_to(LifecycleState::Live).is_ok());
        
        assert!(lm.can_trade());
    }
    
    #[test]
    fn test_invalid_transitions() {
        let lm = LifecycleManager::new();
        
        // Cannot go directly from Idle to Live
        assert!(lm.transition_to(LifecycleState::Live).is_err());
        
        // Cannot go backwards
        lm.transition_to(LifecycleState::Initializing).unwrap();
        assert!(lm.transition_to(LifecycleState::Idle).is_err());
    }
    
    #[test]
    fn test_shutdown_confirmation() {
        let lm = LifecycleManager::new();
        lm.transition_to(LifecycleState::Initializing).unwrap();
        lm.transition_to(LifecycleState::Warmup).unwrap();
        
        // Request shutdown
        let result = lm.request_shutdown();
        assert!(matches!(result, Err(LifecycleError::ConfirmationRequired)));
        
        // Confirm shutdown
        assert!(lm.confirm_shutdown(true).is_ok());
        assert_eq!(lm.get_state(), LifecycleState::ShuttingDown);
    }
    
    #[test]
    fn test_force_shutdown() {
        let lm = LifecycleManager::new();
        lm.transition_to(LifecycleState::Live).unwrap();
        
        assert!(lm.force_shutdown().is_ok());
        assert_eq!(lm.get_state(), LifecycleState::ShuttingDown);
    }
    
    #[test]
    fn test_pause_resume() {
        let lm = LifecycleManager::new();
        lm.transition_to(LifecycleState::Initializing).unwrap();
        lm.transition_to(LifecycleState::Warmup).unwrap();
        lm.transition_to(LifecycleState::HotStandby).unwrap();
        lm.transition_to(LifecycleState::Live).unwrap();
        
        assert!(lm.pause().is_ok());
        assert_eq!(lm.get_state(), LifecycleState::Paused);
        assert!(!lm.can_trade());
        
        assert!(lm.resume().is_ok());
        assert_eq!(lm.get_state(), LifecycleState::Live);
        assert!(lm.can_trade());
    }
    
    #[test]
    fn test_stats() {
        let lm = LifecycleManager::new();
        lm.transition_to(LifecycleState::Initializing).unwrap();
        lm.transition_to(LifecycleState::Warmup).unwrap();
        
        let stats = lm.get_stats();
        assert_eq!(stats.state_transitions, 2);
        assert_eq!(stats.current_state, LifecycleState::Warmup);
        assert!(stats.uptime_ns > 0);
    }
}
