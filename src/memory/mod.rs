//! Memory Module Root
//!
//! This module provides ultra-low-latency memory management primitives:
//! - Arena: Lock-free bump allocator for zero-allocation object creation
//! - Pool: Generic object pool for pre-allocated memory blocks
//! - GlobalTracker: Continuous heap usage monitoring with panic on limit breach
//!
//! All operations are designed to avoid OS garbage collection pauses
//! and maintain millisecond execution speeds.

pub mod arena;
pub mod pool;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use anyhow::Context;

pub use arena::{Arena, LocalArena};
pub use pool::{ObjectPool, PacketBuffer, PoolGuard, TickData};

/// Maximum RAM limit in bytes (default 6.5GB = 6656 MB)
static MAX_RAM_BYTES: AtomicUsize = AtomicUsize::new(6656 * 1024 * 1024);

/// Current tracked heap usage
static CURRENT_HEAP_USAGE: AtomicUsize = AtomicUsize::new(0);

/// Panic threshold percentage (panic at 95% of limit)
const PANIC_THRESHOLD: f64 = 0.95;

/// Safety margin in bytes before hard limit
const SAFETY_MARGIN: usize = 100 * 1024 * 1024; // 100MB

/// Global memory tracker for monitoring heap usage
pub struct GlobalMemoryTracker {
    /// Registered arenas
    arenas: Vec<Arc<Arena>>,
    
    /// Registered pools (approximate size tracking)
    pool_count: AtomicUsize,
    
    /// Whether the tracker is initialized
    initialized: AtomicBool,
}

impl GlobalMemoryTracker {
    /// Create a new global memory tracker
    pub fn new() -> Self {
        Self {
            arenas: Vec::new(),
            pool_count: AtomicUsize::new(0),
            initialized: AtomicBool::new(false),
        }
    }
    
    /// Initialize the tracker with the configured RAM limit from environment
    pub fn init_from_env() -> Result<Arc<Self>, anyhow::Error> {
        // Load environment variables
        dotenvy::dotenv().ok(); // Ignore error if .env doesn't exist
        
        let max_ram_mb: usize = std::env::var("MAX_RAM_LIMIT_MB")
            .unwrap_or_else(|_| "6656".to_string())
            .parse()
            .context("Failed to parse MAX_RAM_LIMIT_MB")?;
        
        let max_ram_bytes = max_ram_mb * 1024 * 1024;
        MAX_RAM_BYTES.store(max_ram_bytes, Ordering::Relaxed);
        
        tracing::info!("Global memory tracker initialized with {} MB limit", max_ram_mb);
        
        let tracker = Arc::new(Self::new());
        tracker.initialized.store(true, Ordering::Release);
        
        Ok(tracker)
    }
    
    /// Register an arena for tracking
    pub fn register_arena(&mut self, arena: Arc<Arena>) {
        self.arenas.push(arena);
    }
    
    /// Register a pool (increment count for approximate tracking)
    pub fn register_pool(&self) {
        self.pool_count.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Get current heap usage estimate
    pub fn current_usage(&self) -> usize {
        let mut total = CURRENT_HEAP_USAGE.load(Ordering::Relaxed);
        
        // Add arena usage
        for arena in &self.arenas {
            total += arena.size() - arena.remaining();
        }
        
        total
    }
    
    /// Get maximum allowed RAM in bytes
    pub fn max_ram(&self) -> usize {
        MAX_RAM_BYTES.load(Ordering::Relaxed)
    }
    
    /// Get current usage as a percentage
    pub fn usage_percentage(&self) -> f64 {
        let current = self.current_usage();
        let max = self.max_ram();
        (current as f64 / max as f64) * 100.0
    }
    
    /// Check if we're approaching the RAM limit
    pub fn is_near_limit(&self) -> bool {
        self.usage_percentage() >= (PANIC_THRESHOLD * 100.0)
    }
    
    /// Get remaining available memory
    pub fn remaining(&self) -> usize {
        self.max_ram() - self.current_usage()
    }
    
    /// Perform a safety check - panics if limit is exceeded
    ///
    /// # Panics
    /// Panics if memory usage exceeds the configured limit minus safety margin
    pub fn safety_check(&self) {
        let current = self.current_usage();
        let max = self.max_ram();
        let threshold = (max as f64 * PANIC_THRESHOLD) as usize;
        
        if current >= threshold {
            let usage_pct = (current as f64 / max as f64) * 100.0;
            
            // Log critical warning first
            tracing::error!(
                "CRITICAL: Memory usage at {:.2}% ({:.2} MB / {:.2} MB)",
                usage_pct,
                current as f64 / (1024.0 * 1024.0),
                max as f64 / (1024.0 * 1024.0)
            );
            
            // Graceful panic with detailed information
            panic!(
                "Memory limit exceeded! Usage: {:.2}%, Limit: {} MB. \
                 Trading halted to prevent system instability.",
                usage_pct,
                max / (1024 * 1024)
            );
        }
        
        // Also check against hard limit with safety margin
        if current >= max.saturating_sub(SAFETY_MARGIN) {
            tracing::warn!(
                "WARNING: Approaching hard memory limit. Current: {} MB, Limit: {} MB",
                current / (1024 * 1024),
                max / (1024 * 1024)
            );
        }
    }
    
    /// Update tracked heap usage (called by custom allocators)
    pub fn update_heap_usage(&self, delta: isize) {
        if delta > 0 {
            CURRENT_HEAP_USAGE.fetch_add(delta as usize, Ordering::Relaxed);
        } else {
            CURRENT_HEAP_USAGE.fetch_sub((-delta) as usize, Ordering::Relaxed);
        }
        
        // Perform safety check after update
        self.safety_check();
    }
    
    /// Get statistics for monitoring
    pub fn get_stats(&self) -> MemoryStats {
        MemoryStats {
            current_usage: self.current_usage(),
            max_ram: self.max_ram(),
            usage_percentage: self.usage_percentage(),
            arena_count: self.arenas.len(),
            pool_count: self.pool_count.load(Ordering::Relaxed),
            remaining: self.remaining(),
        }
    }
}

impl Default for GlobalMemoryTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Memory statistics snapshot
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub current_usage: usize,
    pub max_ram: usize,
    pub usage_percentage: f64,
    pub arena_count: usize,
    pub pool_count: usize,
    pub remaining: usize,
}

impl MemoryStats {
    /// Format stats for logging/display
    pub fn format(&self) -> String {
        format!(
            "Memory: {:.1} MB / {:.1} MB ({:.1}%) | Arenas: {} | Pools: {} | Remaining: {:.1} MB",
            self.current_usage as f64 / (1024.0 * 1024.0),
            self.max_ram as f64 / (1024.0 * 1024.0),
            self.usage_percentage,
            self.arena_count,
            self.pool_count,
            self.remaining as f64 / (1024.0 * 1024.0)
        )
    }
}

/// Global singleton instance (lazy initialized)
static mut GLOBAL_TRACKER: Option<Arc<GlobalMemoryTracker>> = None;

/// Get the global memory tracker instance
///
/// # Panics
/// Panics if the tracker hasn't been initialized yet
pub fn get_global_tracker() -> Arc<GlobalMemoryTracker> {
    unsafe {
        GLOBAL_TRACKER
            .as_ref()
            .expect("Global memory tracker not initialized. Call init_global_tracker() first.")
            .clone()
    }
}

/// Initialize the global tracker singleton
pub fn init_global_tracker() -> Result<Arc<GlobalMemoryTracker>, anyhow::Error> {
    let tracker = GlobalMemoryTracker::init_from_env()?;
    
    unsafe {
        GLOBAL_TRACKER = Some(tracker.clone());
    }
    
    Ok(tracker)
}

/// Perform a global safety check
pub fn global_safety_check() {
    unsafe {
        if let Some(ref tracker) = GLOBAL_TRACKER {
            tracker.safety_check();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_memory_tracker_basic() {
        let tracker = GlobalMemoryTracker::new();
        
        assert!(!tracker.is_near_limit());
        assert!(tracker.usage_percentage() < 1.0);
        
        let stats = tracker.get_stats();
        assert_eq!(stats.max_ram, 6656 * 1024 * 1024);
    }
    
    #[test]
    fn test_memory_stats_format() {
        let stats = MemoryStats {
            current_usage: 1024 * 1024 * 100, // 100 MB
            max_ram: 6656 * 1024 * 1024,      // 6656 MB
            usage_percentage: 1.5,
            arena_count: 2,
            pool_count: 5,
            remaining: 6556 * 1024 * 1024,
        };
        
        let formatted = stats.format();
        assert!(formatted.contains("100.0"));
        assert!(formatted.contains("6656.0"));
        assert!(formatted.contains("Arenas: 2"));
    }
}
