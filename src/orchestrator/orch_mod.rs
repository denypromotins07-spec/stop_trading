//! Orchestrator Module Root
//! 
//! Ties the lifecycle manager to the global kill switch and hardware monitors.
//! Provides unified control interface for system state management.

pub mod lifecycle;

pub use lifecycle::{
    LifecycleManager, LifecycleState, LifecycleError, LifecycleStats,
    ShutdownConfirm, init_lifecycle, get_lifecycle,
};

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// Global kill switch (hardware-level emergency stop)
static GLOBAL_KILL_SWITCH: AtomicBool = AtomicBool::new(false);

/// Hardware monitoring enabled flag
static HW_MONITORING_ENABLED: AtomicBool = AtomicBool::new(true);

/// Memory usage tracker (bytes)
static MEMORY_USAGE_BYTES: AtomicU64 = AtomicU64::new(0);

/// Maximum allowed memory (6.5GB limit)
const MAX_MEMORY_BYTES: u64 = 6_979_321_856;

/// Orchestrator status
#[derive(Debug, Clone)]
pub struct OrchestratorStatus {
    pub lifecycle_state: LifecycleState,
    pub kill_switch_active: bool,
    pub hw_monitoring_enabled: bool,
    pub memory_usage_bytes: u64,
    pub memory_limit_bytes: u64,
    pub memory_usage_percent: f64,
}

/// Main orchestrator
pub struct Orchestrator {
    /// Lifecycle manager
    lifecycle: Arc<LifecycleManager>,
    /// Start time
    start_time: std::sync::Mutex<Option<Instant>>,
    /// Hardware monitor handle
    hw_monitor_handle: std::sync::Mutex<Option<std::thread::JoinHandle<()>>>,
}

unsafe impl Send for Orchestrator {}
unsafe impl Sync for Orchestrator {}

impl Orchestrator {
    /// Create new orchestrator
    pub fn new() -> Self {
        Self {
            lifecycle: Arc::new(LifecycleManager::new()),
            start_time: std::sync::Mutex::new(None),
            hw_monitor_handle: std::sync::Mutex::new(None),
        }
    }
    
    /// Initialize the orchestrator
    pub fn init(&self) -> Result<(), String> {
        // Record start time
        *self.start_time.lock().unwrap() = Some(Instant::now());
        
        // Initialize lifecycle
        self.lifecycle
            .transition_to(LifecycleState::Initializing)
            .map_err(|e| format!("Failed to initialize lifecycle: {}", e))?;
        
        // Start hardware monitoring
        self.start_hw_monitoring();
        
        Ok(())
    }
    
    /// Start hardware monitoring thread
    fn start_hw_monitoring(&self) {
        if !HW_MONITORING_ENABLED.load(Ordering::Relaxed) {
            return;
        }
        
        let handle = std::thread::spawn(|| {
            // In production, would monitor:
            // - CPU temperature
            // - Memory pressure
            // - Network latency
            // - Disk I/O
            
            while HW_MONITORING_ENABLED.load(Ordering::Relaxed) {
                // Sample memory usage (mock - would use actual system calls)
                let current_mem = MEMORY_USAGE_BYTES.load(Ordering::Relaxed);
                
                // Check against limit
                if current_mem > MAX_MEMORY_BYTES {
                    eprintln!("[CRITICAL] Memory limit exceeded! {} > {}", current_mem, MAX_MEMORY_BYTES);
                    GLOBAL_KILL_SWITCH.store(true, Ordering::SeqCst);
                    break;
                }
                
                std::thread::sleep(std::time::Duration::from_secs(1));
            }
        });
        
        *self.hw_monitor_handle.lock().unwrap() = Some(handle);
    }
    
    /// Stop hardware monitoring
    fn stop_hw_monitoring(&self) {
        HW_MONITORING_ENABLED.store(false, Ordering::Relaxed);
        
        let handle = self.hw_monitor_handle.lock().unwrap().take();
        if let Some(h) = handle {
            let _ = h.join();
        }
    }
    
    /// Get lifecycle manager reference
    pub fn lifecycle(&self) -> &Arc<LifecycleManager> {
        &self.lifecycle
    }
    
    /// Activate global kill switch
    pub fn activate_kill_switch(&self) {
        GLOBAL_KILL_SWITCH.store(true, Ordering::SeqCst);
        self.lifecycle.force_shutdown().ok();
    }
    
    /// Check if kill switch is active
    pub fn is_killed(&self) -> bool {
        GLOBAL_KILL_SWITCH.load(Ordering::SeqCst)
    }
    
    /// Reset kill switch (requires manual intervention)
    pub fn reset_kill_switch(&self) {
        GLOBAL_KILL_SWITCH.store(false, Ordering::SeqCst);
    }
    
    /// Update memory usage tracking
    pub fn update_memory_usage(&self, bytes: u64) {
        MEMORY_USAGE_BYTES.store(bytes, Ordering::Relaxed);
    }
    
    /// Get current memory usage
    pub fn get_memory_usage(&self) -> u64 {
        MEMORY_USAGE_BYTES.load(Ordering::Relaxed)
    }
    
    /// Check memory constraints
    pub fn check_memory_constraints(&self) -> bool {
        let usage = self.get_memory_usage();
        usage <= MAX_MEMORY_BYTES
    }
    
    /// Get orchestrator status
    pub fn get_status(&self) -> OrchestratorStatus {
        let mem_usage = self.get_memory_usage();
        
        OrchestratorStatus {
            lifecycle_state: self.lifecycle.get_state(),
            kill_switch_active: self.is_killed(),
            hw_monitoring_enabled: HW_MONITORING_ENABLED.load(Ordering::Relaxed),
            memory_usage_bytes: mem_usage,
            memory_limit_bytes: MAX_MEMORY_BYTES,
            memory_usage_percent: (mem_usage as f64 / MAX_MEMORY_BYTES as f64) * 100.0,
        }
    }
    
    /// Graceful shutdown
    pub fn shutdown(&self) -> Result<(), String> {
        println!("[ORCHESTRATOR] Initiating shutdown sequence...");
        
        // Stop hardware monitoring
        self.stop_hw_monitoring();
        
        // Transition to shutting down
        self.lifecycle
            .transition_to(LifecycleState::ShuttingDown)
            .map_err(|e| format!("Shutdown transition failed: {}", e))?;
        
        // Finalize lifecycle
        self.lifecycle
            .transition_to(LifecycleState::Terminated)
            .map_err(|e| format!("Terminate transition failed: {}", e))?;
        
        println!("[ORCHESTRATOR] Shutdown complete");
        Ok(())
    }
}

impl Default for Orchestrator {
    fn default() -> Self {
        Self::new()
    }
}

/// Global orchestrator instance
static GLOBAL_ORCHESTRATOR: std::sync::OnceLock<Arc<Orchestrator>> = std::sync::OnceLock::new();

/// Initialize global orchestrator
pub fn init_orchestrator() -> Result<Arc<Orchestrator>, &'static str> {
    let orchestrator = Arc::new(Orchestrator::new());
    orchestrator.init().map_err(|_| "Failed to initialize orchestrator")?;
    
    GLOBAL_ORCHESTRATOR
        .set(orchestrator.clone())
        .map_err(|_| "Orchestrator already initialized")?;
    
    Ok(orchestrator)
}

/// Get reference to global orchestrator
pub fn get_orchestrator() -> Option<Arc<Orchestrator>> {
    GLOBAL_ORCHESTRATOR.get().cloned()
}

/// Check if system should halt (kill switch or lifecycle terminated)
pub fn should_halt() -> bool {
    GLOBAL_KILL_SWITCH.load(Ordering::SeqCst) ||
        get_lifecycle().map(|l| l.get_state().is_terminal()).unwrap_or(false)
}

/// Emergency halt from any context
pub fn emergency_halt() {
    GLOBAL_KILL_SWITCH.store(true, Ordering::SeqCst);
    if let Some(lc) = get_lifecycle() {
        lc.force_shutdown().ok();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_orchestrator_creation() {
        let orch = Orchestrator::new();
        assert!(!orch.is_killed());
        assert!(orch.check_memory_constraints());
    }
    
    #[test]
    fn test_orchestrator_init() {
        let orch = Orchestrator::new();
        assert!(orch.init().is_ok());
        
        let status = orch.get_status();
        assert_eq!(status.lifecycle_state, LifecycleState::Initializing);
        assert!(!status.kill_switch_active);
        assert!(status.hw_monitoring_enabled);
    }
    
    #[test]
    fn test_kill_switch() {
        let orch = Orchestrator::new();
        orch.init().unwrap();
        
        assert!(!orch.is_killed());
        
        orch.activate_kill_switch();
        assert!(orch.is_killed());
        
        // Verify lifecycle was also shut down
        assert_eq!(orch.lifecycle().get_state(), LifecycleState::ShuttingDown);
    }
    
    #[test]
    fn test_memory_tracking() {
        let orch = Orchestrator::new();
        
        assert_eq!(orch.get_memory_usage(), 0);
        
        orch.update_memory_usage(1_000_000_000); // 1GB
        assert_eq!(orch.get_memory_usage(), 1_000_000_000);
        
        assert!(orch.check_memory_constraints());
        
        orch.update_memory_usage(MAX_MEMORY_BYTES + 1);
        assert!(!orch.check_memory_constraints());
    }
    
    #[test]
    fn test_graceful_shutdown() {
        let orch = Orchestrator::new();
        orch.init().unwrap();
        
        assert!(orch.shutdown().is_ok());
        
        let status = orch.get_status();
        assert_eq!(status.lifecycle_state, LifecycleState::Terminated);
    }
    
    #[test]
    fn test_global_functions() {
        // Test emergency halt
        assert!(!GLOBAL_KILL_SWITCH.load(Ordering::Relaxed));
        
        emergency_halt();
        assert!(GLOBAL_KILL_SWITCH.load(Ordering::Relaxed));
        
        // Reset for other tests
        GLOBAL_KILL_SWITCH.store(false, Ordering::Relaxed);
    }
}
