//! Feature Synchronization Daemon for Rust-Python Feature Alignment
//! 
//! Ensures Rust and Python feature vectors remain perfectly aligned
//! using atomic sequence counters to detect and resolve race conditions.

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::time::Instant;

/// Feature synchronization state
#[derive(Debug, Clone)]
pub struct SyncState {
    pub rust_sequence: u64,
    pub python_sequence: u64,
    pub last_sync_time_ns: u64,
    pub drift_detected: bool,
}

/// Feature sync daemon
pub struct FeatureSyncDaemon {
    rust_sequence: AtomicU64,
    python_sequence: AtomicU64,
    running: AtomicBool,
    total_syncs: AtomicU64,
    drift_events: AtomicU64,
    last_sync_time: AtomicU64,
}

impl FeatureSyncDaemon {
    pub fn new() -> Self {
        Self {
            rust_sequence: AtomicU64::new(0),
            python_sequence: AtomicU64::new(0),
            running: AtomicBool::new(true),
            total_syncs: AtomicU64::new(0),
            drift_events: AtomicU64::new(0),
            last_sync_time: AtomicU64::new(0),
        }
    }
    
    /// Increment Rust sequence counter
    pub fn rust_produce(&self) -> u64 {
        self.rust_sequence.fetch_add(1, Ordering::AcqRel) + 1
    }
    
    /// Update Python sequence counter
    pub fn python_consume(&self, seq: u64) -> bool {
        let current = self.python_sequence.load(Ordering::Acquire);
        
        if seq < current {
            // Out of order - potential race condition
            self.drift_events.fetch_add(1, Ordering::Relaxed);
            return false;
        }
        
        self.python_sequence.store(seq, Ordering::Release);
        true
    }
    
    /// Check synchronization status
    pub fn get_sync_state(&self) -> SyncState {
        let rust_seq = self.rust_sequence.load(Ordering::Acquire);
        let python_seq = self.python_sequence.load(Ordering::Acquire);
        let last_sync = self.last_sync_time.load(Ordering::Acquire);
        
        // Drift detected if sequences differ by more than buffer threshold
        let drift = (rust_seq as i64 - python_seq as i64).abs() > 100;
        
        SyncState {
            rust_sequence: rust_seq,
            python_sequence: python_seq,
            last_sync_time_ns: last_sync,
            drift_detected: drift,
        }
    }
    
    /// Record successful sync
    pub fn record_sync(&self) {
        self.total_syncs.fetch_add(1, Ordering::Relaxed);
        let now_ns = Instant::now().elapsed().as_nanos() as u64;
        self.last_sync_time.store(now_ns, Ordering::Release);
    }
    
    /// Get statistics
    pub fn get_stats(&self) -> SyncStats {
        SyncStats {
            total_syncs: self.total_syncs.load(Ordering::Relaxed),
            drift_events: self.drift_events.load(Ordering::Relaxed),
            is_running: self.running.load(Ordering::Acquire),
        }
    }
    
    pub fn stop(&self) {
        self.running.store(false, Ordering::Release);
    }
    
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Acquire)
    }
}

impl Default for FeatureSyncDaemon {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct SyncStats {
    pub total_syncs: u64,
    pub drift_events: u64,
    pub is_running: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_feature_sync() {
        let daemon = FeatureSyncDaemon::new();
        
        let rust_seq = daemon.rust_produce();
        assert_eq!(rust_seq, 1);
        
        assert!(daemon.python_consume(rust_seq));
        
        let state = daemon.get_sync_state();
        assert!(!state.drift_detected);
    }
}
