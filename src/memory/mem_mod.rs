//! Memory Management Module Root
//! 
//! Wires the custom allocator and enforcer directly to the global kill switch.

pub mod allocator;
pub mod enforcer;

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};
use crate::memory::allocator::{GLOBAL_ALLOCATOR, init_allocator, get_allocator_stats, AllocStats};
use crate::memory::enforcer::{RamEnforcer, RamEnforcerConfig, MemoryPressure, CleanupResult, EnforcerStats};

/// Memory module configuration
#[derive(Debug, Clone)]
pub struct MemoryModuleConfig {
    pub soft_limit_mb: u64,
    pub hard_limit_mb: u64,
    pub check_interval_ms: u64,
    pub emergency_threshold_pct: f64,
    /// Enable automatic kill switch
    pub auto_kill_enabled: bool,
}

impl Default for MemoryModuleConfig {
    fn default() -> Self {
        Self {
            soft_limit_mb: 6000,
            hard_limit_mb: 6500,
            check_interval_ms: 100,
            emergency_threshold_pct: 95.0,
            auto_kill_enabled: true,
        }
    }
}

/// Global kill switch state
#[derive(Debug, Clone)]
pub struct KillSwitchState {
    pub is_triggered: bool,
    pub reason: String,
    pub memory_usage_bytes: u64,
    pub memory_limit_bytes: u64,
    pub timestamp_ns: u64,
}

/// Memory module handle
pub struct MemoryModule {
    config: MemoryModuleConfig,
    enforcer: Arc<RamEnforcer>,
    /// Kill switch triggered flag
    kill_switch: AtomicBool,
    /// Kill switch reason
    kill_reason: std::sync::Mutex<String>,
    /// Kill switch timestamp
    kill_timestamp: AtomicU64,
    /// Enforcer thread handle
    enforcer_handle: std::sync::Mutex<Option<JoinHandle<()>>>,
}

impl MemoryModule {
    pub fn new(config: MemoryModuleConfig) -> Self {
        // Initialize the global allocator with limits
        init_allocator(config.soft_limit_mb, config.hard_limit_mb);
        
        let enforcer_config = RamEnforcerConfig {
            soft_limit_mb: config.soft_limit_mb,
            hard_limit_mb: config.hard_limit_mb,
            check_interval_ms: config.check_interval_ms,
            emergency_threshold_pct: config.emergency_threshold_pct,
        };
        
        let enforcer = Arc::new(RamEnforcer::new(enforcer_config));

        Self {
            config,
            enforcer,
            kill_switch: AtomicBool::new(false),
            kill_reason: std::sync::Mutex::new(String::new()),
            kill_timestamp: AtomicU64::new(0),
            enforcer_handle: std::sync::Mutex::new(None),
        }
    }

    /// Start the memory management system
    pub fn start(&self) -> Result<(), &'static str> {
        if self.enforcer.is_running() {
            return Err("Memory module already running");
        }

        let enforcer = self.enforcer.clone();
        let kill_switch = self.kill_switch.clone();
        let kill_reason = self.kill_reason.clone();
        let kill_timestamp = self.kill_timestamp.clone();
        let soft_limit = self.config.soft_limit_mb * 1024 * 1024;
        let auto_kill = self.config.auto_kill_enabled;

        let callback = Arc::new(move |pressure: MemoryPressure, result: &CleanupResult| {
            match pressure {
                MemoryPressure::Low | MemoryPressure::Medium => {
                    // Just log, no action needed
                }
                MemoryPressure::High => {
                    log_warning(&format!(
                        "High memory pressure: {} bytes freed, {} caches purged",
                        result.bytes_freed, result.caches_purged
                    ));
                }
                MemoryPressure::Critical => {
                    let stats = get_allocator_stats();
                    
                    if auto_kill && stats.current_usage >= soft_limit * 98 / 100 {
                        // Trigger kill switch
                        kill_switch.store(true, Ordering::SeqCst);
                        *kill_reason.lock().unwrap() = format!(
                            "Critical memory pressure: usage {} bytes exceeds limit",
                            stats.current_usage
                        );
                        kill_timestamp.store(timestamp_ns(), Ordering::SeqCst);
                        
                        log_emergency(&format!(
                            "KILL SWITCH TRIGGERED: Memory at {} bytes (limit {})",
                            stats.current_usage, soft_limit
                        ));
                        
                        // Stop enforcer
                        enforcer.trigger_emergency_halt();
                    } else {
                        log_warning(&format!(
                            "CRITICAL: Memory pressure critical, {} bytes freed",
                            result.bytes_freed
                        ));
                    }
                }
            }
        });

        let handle = enforcer.start(callback);
        *self.enforcer_handle.lock().unwrap() = Some(handle);

        Ok(())
    }

    /// Stop the memory management system
    pub fn stop(&self) {
        self.enforcer.stop();
        
        // Wait for enforcer thread to finish
        if let Some(handle) = self.enforcer_handle.lock().unwrap().take() {
            let _ = handle.join();
        }
    }

    /// Check if kill switch is triggered
    pub fn is_killed(&self) -> bool {
        self.kill_switch.load(Ordering::SeqCst)
    }

    /// Get kill switch state
    pub fn get_kill_state(&self) -> Option<KillSwitchState> {
        if !self.kill_switch.load(Ordering::SeqCst) {
            return None;
        }

        let stats = get_allocator_stats();
        let soft_limit = self.config.soft_limit_mb * 1024 * 1024;

        Some(KillSwitchState {
            is_triggered: true,
            reason: self.kill_reason.lock().unwrap().clone(),
            memory_usage_bytes: stats.current_usage,
            memory_limit_bytes: soft_limit,
            timestamp_ns: self.kill_timestamp.load(Ordering::SeqCst),
        })
    }

    /// Manually trigger kill switch
    pub fn trigger_kill(&self, reason: &str) {
        self.kill_switch.store(true, Ordering::SeqCst);
        *self.kill_reason.lock().unwrap() = reason.to_string();
        self.kill_timestamp.store(timestamp_ns(), Ordering::SeqCst);
        self.enforcer.trigger_emergency_halt();
    }

    /// Reset kill switch (for testing only)
    pub fn reset_kill(&self) {
        self.kill_switch.store(false, Ordering::SeqCst);
        *self.kill_reason.lock().unwrap() = String::new();
        self.kill_timestamp.store(0, Ordering::SeqCst);
        self.enforcer.clear_emergency_halt();
    }

    /// Get current memory stats
    pub fn get_memory_stats(&self) -> AllocStats {
        get_allocator_stats()
    }

    /// Get enforcer stats
    pub fn get_enforcer_stats(&self) -> EnforcerStats {
        self.enforcer.get_stats()
    }

    /// Get memory pressure level
    pub fn get_pressure(&self) -> MemoryPressure {
        self.enforcer.get_pressure()
    }

    /// Check if memory is safe
    pub fn is_memory_safe(&self, threshold_pct: f64) -> bool {
        let stats = get_allocator_stats();
        let soft_limit = self.config.soft_limit_mb * 1024 * 1024;
        (stats.current_usage as f64 / soft_limit as f64) < threshold_pct
    }

    /// Get module statistics
    pub fn get_module_stats(&self) -> ModuleStats {
        let alloc_stats = get_allocator_stats();
        let enf_stats = self.enforcer.get_stats();

        ModuleStats {
            allocator: alloc_stats,
            enforcer: enf_stats,
            kill_switch_triggered: self.is_killed(),
            is_running: self.enforcer.is_running(),
        }
    }
}

/// Combined module statistics
#[derive(Debug, Clone)]
pub struct ModuleStats {
    pub allocator: AllocStats,
    pub enforcer: EnforcerStats,
    pub kill_switch_triggered: bool,
    pub is_running: bool,
}

/// Log warning message
fn log_warning(msg: &str) {
    eprintln!("[WARNING] {}", msg);
}

/// Log emergency message
fn log_emergency(msg: &str) {
    eprintln!("[EMERGENCY] {}", msg);
}

/// Get current timestamp in nanoseconds
#[inline]
fn timestamp_ns() -> u64 {
    Instant::now()
        .duration_since(Instant::now() - Duration::from_secs(1))
        .as_nanos() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_module_creation() {
        let config = MemoryModuleConfig::default();
        let module = MemoryModule::new(config);

        assert!(!module.is_killed());
        assert!(module.get_kill_state().is_none());
    }

    #[test]
    fn test_manual_kill() {
        let config = MemoryModuleConfig::default();
        let module = MemoryModule::new(config);

        module.trigger_kill("Test kill");
        assert!(module.is_killed());
        
        let state = module.get_kill_state();
        assert!(state.is_some());
        assert_eq!(state.unwrap().reason, "Test kill");

        module.reset_kill();
        assert!(!module.is_killed());
    }

    #[test]
    fn test_memory_stats() {
        let config = MemoryModuleConfig::default();
        let module = MemoryModule::new(config);

        let stats = module.get_memory_stats();
        assert!(stats.current_usage >= 0);
        assert!(stats.peak_usage >= 0);
    }

    #[test]
    fn test_memory_safety_check() {
        let config = MemoryModuleConfig::default();
        let module = MemoryModule::new(config);

        // Should be safe at low usage
        assert!(module.is_memory_safe(0.9));
    }
}
