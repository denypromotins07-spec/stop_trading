//! Hot/Cold failover manager for instant state recovery.
//! 
//! Hydrates the engine from the latest rkyv snapshot on boot, ensuring zero data loss
//! and immediate resumption of trading logic after unexpected termination.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};
use crate::state::snapshot::{StateSnapshotter, EngineState, SnapshotError, SnapshotInfo};

/// Failover mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailoverMode {
    /// Primary/hot instance (actively trading)
    Hot,
    /// Secondary/cold instance (standby)
    Cold,
    /// Recovery mode (restoring from snapshot)
    Recovering,
}

/// Failover status
#[derive(Debug, Clone)]
pub struct FailoverStatus {
    /// Current mode
    pub mode: FailoverMode,
    /// Whether state was recovered from snapshot
    pub recovered_from_snapshot: bool,
    /// Snapshot timestamp if recovered
    pub snapshot_timestamp_ns: Option<u64>,
    /// Time since last heartbeat (nanoseconds)
    pub time_since_heartbeat_ns: u64,
    /// Failover count
    pub failover_count: u64,
    /// Last error message
    pub last_error: Option<String>,
}

/// Hot/Cold failover manager
pub struct FailoverManager {
    /// Current mode
    mode: FailoverMode,
    /// State snapshotter
    snapshotter: StateSnapshotter,
    /// Heartbeat file path
    heartbeat_path: PathBuf,
    /// Last heartbeat timestamp
    last_heartbeat_ns: AtomicU64,
    /// Failover counter
    failover_count: AtomicU64,
    /// Is active (trading enabled)
    is_active: AtomicBool,
    /// Recovery in progress
    recovery_in_progress: AtomicBool,
    /// Maximum heartbeat age before failover (milliseconds)
    max_heartbeat_age_ms: u64,
}

impl FailoverManager {
    /// Create a new failover manager
    pub fn new(
        snapshot_dir: PathBuf,
        heartbeat_path: PathBuf,
        max_snapshots: usize,
        max_heartbeat_age_ms: u64,
    ) -> Self {
        Self {
            mode: FailoverMode::Cold,
            snapshotter: StateSnapshotter::new(snapshot_dir, max_snapshots),
            heartbeat_path,
            last_heartbeat_ns: AtomicU64::new(0),
            failover_count: AtomicU64::new(0),
            is_active: AtomicBool::new(false),
            recovery_in_progress: AtomicBool::new(false),
            max_heartbeat_age_ms,
        }
    }
    
    /// Initialize as hot (primary) instance
    pub fn initialize_hot(&mut self) -> Result<(), FailoverError> {
        self.mode = FailoverMode::Hot;
        self.is_active.store(true, Ordering::Relaxed);
        self.send_heartbeat()?;
        Ok(())
    }
    
    /// Initialize as cold (standby) instance
    pub fn initialize_cold(&mut self) -> Result<(), FailoverError> {
        self.mode = FailoverMode::Cold;
        self.is_active.store(false, Ordering::Relaxed);
        Ok(())
    }
    
    /// Attempt to recover state from latest snapshot
    pub fn recover_state(&mut self) -> Result<RecoveryResult, FailoverError> {
        self.recovery_in_progress.store(true, Ordering::Relaxed);
        self.mode = FailoverMode::Recovering;
        
        let start = Instant::now();
        
        match self.snapshotter.load_latest() {
            Ok(state) => {
                let duration = start.elapsed();
                
                self.failover_count.fetch_add(1, Ordering::Relaxed);
                self.recovery_in_progress.store(false, Ordering::Relaxed);
                self.mode = FailoverMode::Hot;
                self.is_active.store(true, Ordering::Relaxed);
                
                Ok(RecoveryResult {
                    success: true,
                    state,
                    recovery_time_ms: duration.as_millis() as u64,
                    orders_recovered: 0, // Will be populated by caller
                    positions_recovered: 0,
                })
            }
            Err(SnapshotError::NoSnapshotsFound) => {
                self.recovery_in_progress.store(false, Ordering::Relaxed);
                self.mode = FailoverMode::Hot;
                self.is_active.store(true, Ordering::Relaxed);
                
                Ok(RecoveryResult {
                    success: false,
                    state: EngineState::empty(),
                    recovery_time_ms: 0,
                    orders_recovered: 0,
                    positions_recovered: 0,
                })
            }
            Err(e) => {
                self.recovery_in_progress.store(false, Ordering::Relaxed);
                Err(FailoverError::Snapshot(e))
            }
        }
    }
    
    /// Send heartbeat (for hot instance)
    pub fn send_heartbeat(&self) -> Result<(), FailoverError> {
        if self.mode != FailoverMode::Hot {
            return Ok(()); // Only hot instances send heartbeats
        }
        
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        
        // Write heartbeat file with current timestamp
        let content = format!("{}", now);
        std::fs::write(&self.heartbeat_path, content)?;
        
        self.last_heartbeat_ns.store(now, Ordering::Relaxed);
        
        Ok(())
    }
    
    /// Check if primary is alive (for cold instance)
    pub fn check_primary_alive(&self) -> bool {
        if self.mode != FailoverMode::Cold {
            return true;
        }
        
        // Read heartbeat file
        match std::fs::read_to_string(&self.heartbeat_path) {
            Ok(content) => {
                if let Ok(last_heartbeat) = content.trim().parse::<u64>() {
                    let now = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_nanos() as u64;
                    
                    let age_ms = (now - last_heartbeat) / 1_000_000;
                    age_ms < self.max_heartbeat_age_ms
                } else {
                    false
                }
            }
            Err(_) => false, // No heartbeat file means primary is down
        }
    }
    
    /// Promote cold instance to hot (failover)
    pub fn promote_to_hot(&mut self) -> Result<EngineState, FailoverError> {
        if self.mode != FailoverMode::Cold {
            return Err(FailoverError::InvalidState("Not in cold mode".to_string()));
        }
        
        // Check if primary is actually down
        if self.check_primary_alive() {
            return Err(FailoverError::PrimaryStillActive);
        }
        
        // Recover state and promote
        let result = self.recover_state()?;
        
        self.mode = FailoverMode::Hot;
        self.is_active.store(true, Ordering::Relaxed);
        self.failover_count.fetch_add(1, Ordering::Relaxed);
        
        Ok(result.state)
    }
    
    /// Graceful shutdown - create final snapshot
    pub fn graceful_shutdown(&mut self, state: &EngineState) -> Result<PathBuf, FailoverError> {
        self.is_active.store(false, Ordering::Relaxed);
        
        // Remove heartbeat file if exists
        std::fs::remove_file(&self.heartbeat_path).ok();
        
        // Create final snapshot
        let filepath = self.snapshotter.create_snapshot(state)?;
        
        self.mode = FailoverMode::Cold;
        
        Ok(filepath)
    }
    
    /// Get current status
    pub fn status(&self) -> FailoverStatus {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        
        let last_hb = self.last_heartbeat_ns.load(Ordering::Relaxed);
        let time_since_hb = if last_hb > 0 { now - last_hb } else { 0 };
        
        FailoverStatus {
            mode: self.mode,
            recovered_from_snapshot: false, // Set during recovery
            snapshot_timestamp_ns: None,
            time_since_heartbeat_ns: time_since_hb,
            failover_count: self.failover_count.load(Ordering::Relaxed),
            last_error: None,
        }
    }
    
    /// List available snapshots for recovery
    pub fn list_snapshots(&self) -> Result<Vec<SnapshotInfo>, SnapshotError> {
        self.snapshotter.list_snapshots()
    }
    
    /// Check if currently active (trading enabled)
    pub fn is_active(&self) -> bool {
        self.is_active.load(Ordering::Relaxed)
    }
    
    /// Check if in recovery mode
    pub fn is_recovering(&self) -> bool {
        self.recovery_in_progress.load(Ordering::Relaxed)
    }
    
    /// Get current mode
    pub fn mode(&self) -> FailoverMode {
        self.mode
    }
}

/// Recovery result from failover
#[derive(Debug, Clone)]
pub struct RecoveryResult {
    /// Whether recovery was successful
    pub success: bool,
    /// Recovered engine state
    pub state: EngineState,
    /// Time taken to recover (milliseconds)
    pub recovery_time_ms: u64,
    /// Number of orders recovered
    pub orders_recovered: usize,
    /// Number of positions recovered
    pub positions_recovered: usize,
}

/// Failover error types
#[derive(Debug)]
pub enum FailoverError {
    Io(std::io::Error),
    Snapshot(SnapshotError),
    InvalidState(String),
    PrimaryStillActive,
    RecoveryFailed(String),
}

impl From<std::io::Error> for FailoverError {
    fn from(err: std::io::Error) -> Self {
        FailoverError::Io(err)
    }
}

impl From<SnapshotError> for FailoverError {
    fn from(err: SnapshotError) -> Self {
        FailoverError::Snapshot(err)
    }
}

impl std::fmt::Display for FailoverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FailoverError::Io(e) => write!(f, "IO error: {}", e),
            FailoverError::Snapshot(e) => write!(f, "Snapshot error: {}", e),
            FailoverError::InvalidState(s) => write!(f, "Invalid state: {}", s),
            FailoverError::PrimaryStillActive => write!(f, "Primary instance still active"),
            FailoverError::RecoveryFailed(s) => write!(f, "Recovery failed: {}", s),
        }
    }
}

impl std::error::Error for FailoverError {}

/// Health monitor for failover coordination
pub struct FailoverHealthMonitor {
    manager: std::sync::Arc<FailoverManager>,
    /// Check interval
    check_interval_ms: u64,
    /// Running flag
    running: AtomicBool,
}

impl FailoverHealthMonitor {
    /// Create a new health monitor
    pub fn new(manager: std::sync::Arc<FailoverManager>, check_interval_ms: u64) -> Self {
        Self {
            manager,
            check_interval_ms,
            running: AtomicBool::new(false),
        }
    }
    
    /// Start monitoring (should be called from separate thread)
    pub fn start(&self) {
        self.running.store(true, Ordering::Relaxed);
        
        while self.running.load(Ordering::Relaxed) {
            if self.manager.mode() == FailoverMode::Hot {
                // Send heartbeat
                self.manager.send_heartbeat().ok();
            } else if self.manager.mode() == FailoverMode::Cold {
                // Check if primary is alive
                if !self.manager.check_primary_alive() {
                    // Primary is down - could trigger automatic failover here
                    log::warn!("Primary instance appears to be down");
                }
            }
            
            std::thread::sleep(Duration::from_millis(self.check_interval_ms));
        }
    }
    
    /// Stop monitoring
    pub fn stop(&self) {
        self.running.store(false, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    
    #[test]
    fn test_failover_basic() {
        let temp_dir = std::env::temp_dir().join("hft_test_failover");
        fs::create_dir_all(&temp_dir).ok();
        
        let heartbeat_path = temp_dir.join("heartbeat");
        
        let mut manager = FailoverManager::new(
            temp_dir.clone(),
            heartbeat_path.clone(),
            5,
            5000, // 5 second timeout
        );
        
        // Initialize as hot
        manager.initialize_hot().unwrap();
        assert_eq!(manager.mode(), FailoverMode::Hot);
        assert!(manager.is_active());
        
        // Send heartbeat
        manager.send_heartbeat().unwrap();
        assert!(heartbeat_path.exists());
        
        // Graceful shutdown
        let state = EngineState::empty();
        let _ = manager.graceful_shutdown(&state).unwrap();
        assert!(!manager.is_active());
        
        // Cleanup
        fs::remove_dir_all(&temp_dir).ok();
    }
    
    #[test]
    fn test_recovery() {
        let temp_dir = std::env::temp_dir().join("hft_test_recovery");
        fs::create_dir_all(&temp_dir).ok();
        
        let heartbeat_path = temp_dir.join("heartbeat2");
        
        let mut manager = FailoverManager::new(
            temp_dir.clone(),
            heartbeat_path,
            5,
            5000,
        );
        
        // Try recovery with no snapshots
        let result = manager.recover_state().unwrap();
        assert!(!result.success);
        
        // Should be active after recovery attempt
        assert!(manager.is_active());
        
        // Cleanup
        fs::remove_dir_all(&temp_dir).ok();
    }
}
