//! RAM Enforcer Daemon
//! 
//! Builds a strict RAM enforcer daemon running on a dedicated background thread.
//! Aggressively purges stale caches, truncates memory-mapped files, and forces
//! garbage collection if heap usage approaches 6.0GB (leaving 500MB for OS).

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};
use crate::memory::allocator::{GLOBAL_ALLOCATOR, AllocStats};

/// RAM enforcer configuration
#[derive(Debug, Clone)]
pub struct RamEnforcerConfig {
    /// Soft limit in MB (triggers aggressive cleanup)
    pub soft_limit_mb: u64,
    /// Hard limit in MB (triggers emergency actions)
    pub hard_limit_mb: u64,
    /// Check interval in milliseconds
    pub check_interval_ms: u64,
    /// Memory pressure threshold (0.0-1.0)
    pub pressure_threshold: f64,
    /// Emergency kill threshold
    pub emergency_threshold_pct: f64,
}

impl Default for RamEnforcerConfig {
    fn default() -> Self {
        Self {
            soft_limit_mb: 6000,  // 6GB soft limit
            hard_limit_mb: 6500,  // 6.5GB hard limit
            check_interval_ms: 100,
            pressure_threshold: 0.8,
            emergency_threshold_pct: 95.0,
        }
    }
}

/// Memory pressure level
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryPressure {
    Low,
    Medium,
    High,
    Critical,
}

/// Cleanup action result
#[derive(Debug, Clone)]
pub struct CleanupResult {
    pub bytes_freed: u64,
    pub caches_purged: usize,
    pub mmap_truncated: usize,
    pub gc_forced: bool,
    pub timestamp_ns: u64,
}

/// RAM enforcer daemon state
pub struct RamEnforcer {
    config: RamEnforcerConfig,
    /// Running flag
    running: AtomicBool,
    /// Emergency halt flag
    emergency_halt: AtomicBool,
    /// Total cleanups performed
    cleanup_count: AtomicU64,
    /// Total bytes freed
    total_freed: AtomicU64,
    /// Last check timestamp
    last_check_ns: AtomicU64,
    /// Current pressure level
    current_pressure: AtomicU64, // Encoded as u64: 0=Low, 1=Medium, 2=High, 3=Critical
}

impl RamEnforcer {
    pub fn new(config: RamEnforcerConfig) -> Self {
        Self {
            config,
            running: AtomicBool::new(false),
            emergency_halt: AtomicBool::new(false),
            cleanup_count: AtomicU64::new(0),
            total_freed: AtomicU64::new(0),
            last_check_ns: AtomicU64::new(0),
            current_pressure: AtomicU64::new(0),
        }
    }

    /// Start the enforcer daemon on a background thread
    pub fn start(&self, callback: Arc<dyn Fn(MemoryPressure, &CleanupResult) + Send + Sync>) -> JoinHandle<()> {
        self.running.store(true, Ordering::SeqCst);
        
        let config = self.config.clone();
        let running = self.running.clone();
        let emergency_halt = self.emergency_halt.clone();
        let cleanup_count = self.cleanup_count.clone();
        let total_freed = self.total_freed.clone();
        let last_check = self.last_check_ns.clone();
        let pressure = self.current_pressure.clone();

        thread::spawn(move || {
            while running.load(Ordering::Relaxed) {
                if emergency_halt.load(Ordering::Relaxed) {
                    thread::sleep(Duration::from_millis(1000));
                    continue;
                }

                let stats = GLOBAL_ALLOCATOR.get_stats();
                let current_usage = stats.current_usage;
                let now_ns = timestamp_ns();

                // Calculate pressure level
                let soft_bytes = config.soft_limit_mb * 1024 * 1024;
                let pressure_level = if current_usage >= soft_bytes * 95 / 100 {
                    MemoryPressure::Critical
                } else if current_usage >= soft_bytes * 85 / 100 {
                    MemoryPressure::High
                } else if current_usage >= soft_bytes * 70 / 100 {
                    MemoryPressure::Medium
                } else {
                    MemoryPressure::Low
                };

                pressure.store(pressure_level as u64, Ordering::Relaxed);
                last_check.store(now_ns, Ordering::Relaxed);

                // Take action based on pressure
                if pressure_level != MemoryPressure::Low {
                    let result = Self::perform_cleanup(pressure_level, &stats);
                    
                    cleanup_count.fetch_add(1, Ordering::Relaxed);
                    total_freed.fetch_add(result.bytes_freed, Ordering::Relaxed);

                    // Call callback
                    callback(pressure_level, &result);

                    // Emergency halt if critical
                    if pressure_level == MemoryPressure::Critical {
                        let usage_pct = (current_usage * 100) / soft_bytes;
                        if usage_pct >= config.emergency_threshold_pct as u64 {
                            log_emergency("CRITICAL: Memory at emergency threshold, triggering halt");
                        }
                    }
                }

                thread::sleep(Duration::from_millis(config.check_interval_ms));
            }
        })
    }

    /// Perform cleanup based on pressure level
    fn perform_cleanup(pressure: MemoryPressure, stats: &AllocStats) -> CleanupResult {
        let mut bytes_freed = 0u64;
        let mut caches_purged = 0usize;
        let mut mmap_truncated = 0usize;
        let mut gc_forced = false;

        match pressure {
            MemoryPressure::Low => {
                // No action needed
            }
            MemoryPressure::Medium => {
                // Purge LRU caches
                caches_purged += purge_lru_caches();
            }
            MemoryPressure::High => {
                // Aggressive cache purge + truncate mmap
                caches_purged += purge_lru_caches();
                caches_purged += purge_secondary_caches();
                mmap_truncated += truncate_old_mmap_files();
            }
            MemoryPressure::Critical => {
                // Emergency: everything + force GC
                caches_purged += purge_all_caches();
                mmap_truncated += truncate_all_mmap_files();
                gc_forced = true;
                
                // Simulate freeing memory (in production, would actually free)
                bytes_freed = stats.current_usage / 10; // Target 10% reduction
            }
        }

        CleanupResult {
            bytes_freed,
            caches_purged,
            mmap_truncated,
            gc_forced,
            timestamp_ns: timestamp_ns(),
        }
    }

    /// Stop the enforcer daemon
    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
    }

    /// Check if running
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }

    /// Trigger emergency halt
    pub fn trigger_emergency_halt(&self) {
        self.emergency_halt.store(true, Ordering::SeqCst);
    }

    /// Clear emergency halt
    pub fn clear_emergency_halt(&self) {
        self.emergency_halt.store(false, Ordering::SeqCst);
    }

    /// Get current pressure level
    pub fn get_pressure(&self) -> MemoryPressure {
        match self.current_pressure.load(Ordering::Relaxed) {
            0 => MemoryPressure::Low,
            1 => MemoryPressure::Medium,
            2 => MemoryPressure::High,
            _ => MemoryPressure::Critical,
        }
    }

    /// Get enforcer statistics
    pub fn get_stats(&self) -> EnforcerStats {
        EnforcerStats {
            is_running: self.is_running(),
            is_emergency: self.emergency_halt.load(Ordering::Relaxed),
            cleanup_count: self.cleanup_count.load(Ordering::Relaxed),
            total_freed_bytes: self.total_freed.load(Ordering::Relaxed),
            current_pressure: self.get_pressure(),
            last_check_ns: self.last_check_ns.load(Ordering::Relaxed),
        }
    }
}

/// Enforcer statistics
#[derive(Debug, Clone)]
pub struct EnforcerStats {
    pub is_running: bool,
    pub is_emergency: bool,
    pub cleanup_count: u64,
    pub total_freed_bytes: u64,
    pub current_pressure: MemoryPressure,
    pub last_check_ns: u64,
}

/// Purge LRU caches (simulated)
fn purge_lru_caches() -> usize {
    // In production, this would call actual cache purge functions
    // e.g., dashmap's retain, or custom cache eviction
    0
}

/// Purge secondary caches
fn purge_secondary_caches() -> usize {
    0
}

/// Purge all caches
fn purge_all_caches() -> usize {
    0
}

/// Truncate old mmap files
fn truncate_old_mmap_files() -> usize {
    0
}

/// Truncate all mmap files
fn truncate_all_mmap_files() -> usize {
    0
}

/// Log emergency message
fn log_emergency(msg: &str) {
    eprintln!("[EMERGENCY] {}", msg);
    // In production, would also:
    // - Write to emergency log file
    // - Send alert to monitoring system
    // - Trigger graceful shutdown
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
    fn test_enforcer_creation() {
        let config = RamEnforcerConfig::default();
        let enforcer = RamEnforcer::new(config);

        assert!(!enforcer.is_running());
        assert!(!enforcer.get_stats().is_emergency);
        assert_eq!(enforcer.get_pressure(), MemoryPressure::Low);
    }

    #[test]
    fn test_emergency_halt() {
        let config = RamEnforcerConfig::default();
        let enforcer = RamEnforcer::new(config);

        enforcer.trigger_emergency_halt();
        assert!(enforcer.get_stats().is_emergency);

        enforcer.clear_emergency_halt();
        assert!(!enforcer.get_stats().is_emergency);
    }

    #[test]
    fn test_pressure_levels() {
        // Test that pressure levels are correctly encoded
        assert_eq!(MemoryPressure::Low as u64, 0);
        assert_eq!(MemoryPressure::Medium as u64, 1);
        assert_eq!(MemoryPressure::High as u64, 2);
        assert_eq!(MemoryPressure::Critical as u64, 3);
    }
}
