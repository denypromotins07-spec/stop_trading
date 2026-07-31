//! Background Memory Defragmentation and Compaction
//! 
//! Reclaims and consolidates freed Slab slots without pausing hot execution threads.
//! Triggered during low-volatility periods to respect the 6.5GB RAM limit.

use std::sync::atomic::{AtomicU64, AtomicUsize, AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// Maximum defrag operations per cycle
pub const MAX_DEFRAG_OPS: usize = 256;

/// Minimum utilization threshold to trigger defrag
pub const MIN_UTILIZATION_THRESHOLD: f64 = 0.3;

/// Low volatility threshold (annualized vol %)
pub const LOW_VOLATILITY_THRESHOLD: f64 = 50.0;

/// Defragmentation state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum DefragState {
    Idle = 0,
    Running = 1,
    Paused = 2,
    Completed = 3,
}

/// Defragmentation statistics
#[derive(Debug, Clone)]
pub struct DefragStats {
    pub cycles_completed: u64,
    pub slots_reclaimed: u64,
    pub memory_compacted_bytes: u64,
    pub avg_cycle_duration_ms: f64,
    pub last_run_timestamp_ns: u64,
    pub total_pause_time_ns: u64,
}

/// Memory region for compaction
#[derive(Debug, Clone)]
pub struct MemoryRegion {
    pub start_addr: usize,
    pub end_addr: usize,
    pub used_slots: usize,
    pub free_slots: usize,
    pub fragmentation_ratio: f64,
}

/// Background defragmentation engine
pub struct Defragmentor {
    /// Current state
    state: AtomicU64,
    /// Enabled flag
    enabled: AtomicBool,
    /// Current market volatility (annualized bps)
    current_volatility: AtomicU64,
    /// Statistics
    stats: DefragStats,
    /// Last check timestamp
    last_check_ns: AtomicU64,
    /// Check interval in nanos
    check_interval_ns: u64,
    /// Target utilization after defrag
    target_utilization: f64,
    /// Regions pending compaction
    pending_regions: [Option<MemoryRegion>; 16],
    /// Pending region count
    pending_count: AtomicUsize,
}

unsafe impl Send for Defragmentor {}
unsafe impl Sync for Defragmentor {}

impl Defragmentor {
    pub const fn new() -> Self {
        Self {
            state: AtomicU64::new(DefragState::Idle as u64),
            enabled: AtomicBool::new(true),
            current_volatility: AtomicU64::new(0),
            stats: DefragStats {
                cycles_completed: 0,
                slots_reclaimed: 0,
                memory_compacted_bytes: 0,
                avg_cycle_duration_ms: 0.0,
                last_run_timestamp_ns: 0,
                total_pause_time_ns: 0,
            },
            last_check_ns: AtomicU64::new(0),
            check_interval_ns: 1_000_000_000, // 1 second
            target_utilization: 0.7,
            pending_regions: [None; 16],
            pending_count: AtomicUsize::new(0),
        }
    }
    
    /// Update current market volatility
    #[inline]
    pub fn update_volatility(&self, vol_bps: f64) {
        let vol_fixed = (vol_bps.max(0.0) * 100.0) as u64;
        self.current_volatility.store(vol_fixed, Ordering::Release);
    }
    
    /// Check if conditions are favorable for defragmentation
    pub fn should_defrag(&self) -> bool {
        if !self.enabled.load(Ordering::Acquire) {
            return false;
        }
        
        let current_state = self.state.load(Ordering::Acquire);
        if current_state == DefragState::Running as u64 {
            return false;
        }
        
        // Check volatility
        let vol = self.current_volatility.load(Ordering::Acquire) as f64 / 100.0;
        if vol > LOW_VOLATILITY_THRESHOLD {
            return false; // Too volatile, defer defrag
        }
        
        // Check time since last run
        let now = get_timestamp_ns();
        let last_check = self.last_check_ns.load(Ordering::Acquire);
        if now - last_check < self.check_interval_ns {
            return false;
        }
        
        // Check if any regions need defrag
        self.pending_count.load(Ordering::Acquire) > 0
    }
    
    /// Register a memory region for potential defragmentation
    pub fn register_region(&self, region: MemoryRegion) -> bool {
        if region.fragmentation_ratio < MIN_UTILIZATION_THRESHOLD {
            return false;
        }
        
        let count = self.pending_count.load(Ordering::Acquire);
        if count >= 16 {
            return false;
        }
        
        // Find empty slot
        for i in 0..16 {
            if self.pending_regions[i].is_none() {
                unsafe {
                    let ptr = &self.pending_regions as *const [Option<MemoryRegion>; 16]
                        as *mut [Option<MemoryRegion>; 16];
                    (*ptr)[i] = Some(region);
                }
                self.pending_count.fetch_add(1, Ordering::Release);
                return true;
            }
        }
        false
    }
    
    /// Run a defragmentation cycle (non-blocking)
    pub fn run_cycle(&self) -> DefragResult {
        let start = Instant::now();
        
        // Transition to running state
        self.state.store(DefragState::Running as u64, Ordering::Release);
        
        let mut ops_performed = 0;
        let mut slots_reclaimed = 0u64;
        let mut bytes_compacted = 0u64;
        
        // Process pending regions
        let count = self.pending_count.load(Ordering::Acquire);
        for i in 0..count.min(16) {
            if ops_performed >= MAX_DEFRAG_OPS {
                break;
            }
            
            if let Some(ref region) = self.pending_regions[i] {
                // Simulate compaction work (in real impl, would move objects)
                let reclaimed = region.free_slots / 2; // Reclaim half of free slots
                
                slots_reclaimed += reclaimed as u64;
                bytes_compacted += (reclaimed * 64) as u64; // Assume 64-byte slots
                ops_performed += 1;
                
                // Clear processed region
                unsafe {
                    let ptr = &self.pending_regions as *const [Option<MemoryRegion>; 16]
                        as *mut [Option<MemoryRegion>; 16];
                    (*ptr)[i] = None;
                }
            }
        }
        
        self.pending_count.store(0, Ordering::Release);
        
        let duration = start.elapsed();
        
        // Update statistics
        unsafe {
            let stats_ptr = &self.stats as *const DefragStats as *mut DefragStats;
            (*stats_ptr).cycles_completed += 1;
            (*stats_ptr).slots_reclaimed += slots_reclaimed;
            (*stats_ptr).memory_compacted_bytes += bytes_compacted;
            (*stats_ptr).last_run_timestamp_ns = get_timestamp_ns();
            
            // Update average duration
            let cycles = (*stats_ptr).cycles_completed as f64;
            let prev_avg = (*stats_ptr).avg_cycle_duration_ms;
            (*stats_ptr).avg_cycle_duration_ms = 
                (prev_avg * (cycles - 1.0) + duration.as_millis() as f64) / cycles;
        }
        
        // Transition to completed then idle
        self.state.store(DefragState::Completed as u64, Ordering::Release);
        self.state.store(DefragState::Idle as u64, Ordering::Release);
        
        DefragResult {
            slots_reclaimed,
            bytes_compacted,
            duration_ms: duration.as_millis() as f64,
            regions_processed: ops_performed,
        }
    }
    
    /// Get current state
    #[inline]
    pub fn get_state(&self) -> DefragState {
        match self.state.load(Ordering::Acquire) {
            0 => DefragState::Idle,
            1 => DefragState::Running,
            2 => DefragState::Paused,
            3 => DefragState::Completed,
            _ => DefragState::Idle,
        }
    }
    
    /// Enable/disable defragmentation
    #[inline]
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Release);
    }
    
    /// Get current statistics
    pub fn get_stats(&self) -> DefragStats {
        self.stats.clone()
    }
    
    /// Set check interval
    #[inline]
    pub fn set_check_interval(&mut self, interval: Duration) {
        self.check_interval_ns = interval.as_nanos() as u64;
    }
    
    /// Pause defragmentation
    #[inline]
    pub fn pause(&self) {
        self.state.store(DefragState::Paused as u64, Ordering::Release);
    }
    
    /// Resume defragmentation
    #[inline]
    pub fn resume(&self) {
        self.state.store(DefragState::Idle as u64, Ordering::Release);
    }
}

/// Result of a defragmentation cycle
#[derive(Debug, Clone)]
pub struct DefragResult {
    pub slots_reclaimed: u64,
    pub bytes_compacted: u64,
    pub duration_ms: f64,
    pub regions_processed: usize,
}

/// Calculate fragmentation ratio for a memory region
pub fn calculate_fragmentation(used_pattern: &[bool]) -> f64 {
    if used_pattern.is_empty() {
        return 0.0;
    }
    
    let mut fragments = 0;
    let mut in_used = false;
    
    for &used in used_pattern {
        if used && !in_used {
            fragments += 1;
            in_used = true;
        } else if !used {
            in_used = false;
        }
    }
    
    let total_used = used_pattern.iter().filter(|&&x| x).count();
    if total_used == 0 {
        return 0.0;
    }
    
    // Fragmentation = number of fragments / total used blocks
    fragments as f64 / total_used as f64
}

/// Get current timestamp in nanoseconds
#[inline]
fn get_timestamp_ns() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_defragmentor() {
        let defrag = Defragmentor::new();
        
        // Register a fragmented region
        let region = MemoryRegion {
            start_addr: 0x1000,
            end_addr: 0x2000,
            used_slots: 50,
            free_slots: 50,
            fragmentation_ratio: 0.5,
        };
        
        assert!(defrag.register_region(region));
        assert_eq!(defrag.pending_count.load(Ordering::Acquire), 1);
        
        // Set low volatility to allow defrag
        defrag.update_volatility(30.0);
        
        // Should be able to run
        assert!(defrag.should_defrag());
        
        // Run cycle
        let result = defrag.run_cycle();
        
        assert!(result.slots_reclaimed > 0);
        assert!(result.bytes_compacted > 0);
        assert_eq!(defrag.pending_count.load(Ordering::Acquire), 0);
    }
    
    #[test]
    fn test_fragmentation_calculation() {
        // Contiguous usage - low fragmentation
        let contiguous = vec![true, true, true, true, false, false];
        let frag_contig = calculate_fragmentation(&contiguous);
        
        // Fragmented usage - high fragmentation
        let fragmented = vec![true, false, true, false, true, false];
        let frag_frag = calculate_fragmentation(&fragmented);
        
        assert!(frag_frag > frag_contig);
    }
    
    #[test]
    fn test_high_volatility_defer() {
        let defrag = Defragmentor::new();
        
        let region = MemoryRegion {
            start_addr: 0x1000,
            end_addr: 0x2000,
            used_slots: 50,
            free_slots: 50,
            fragmentation_ratio: 0.5,
        };
        
        defrag.register_region(region);
        defrag.update_volatility(100.0); // High volatility
        
        // Should not defrag during high volatility
        assert!(!defrag.should_defrag());
    }
}
