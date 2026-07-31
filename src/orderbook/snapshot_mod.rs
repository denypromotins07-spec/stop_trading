//! Order Book Module Root
//! 
//! Orchestrates the seamless transition from REST snapshot to WebSocket delta stream.

pub mod snapshot_manager;
pub mod checksum;

pub use snapshot_manager::{SnapshotManager, OrderBookSnapshot, PriceLevel, GapInfo, SnapshotStats};
pub use checksum::{ChecksumValidator, IncrementalCrc32, ValidationResult, ChecksumStats, crc32_fast};

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};

/// Order book synchronization state machine
#[repr(C, align(64))]
pub struct OrderBookSync {
    /// Snapshot manager for REST data
    pub snapshots: SnapshotManager,
    /// Checksum validator for WS data
    pub validator: ChecksumValidator,
    /// Current sync state
    state: AtomicU64,
    /// Whether currently syncing
    is_syncing: AtomicBool,
    /// Sync start timestamp
    sync_start_ns: AtomicU64,
    /// Total resyncs performed
    resync_count: AtomicU64,
}

/// Sync state enum values
pub const STATE_DISCONNECTED: u64 = 0;
pub const STATE_REQUESTING_SNAPSHOT: u64 = 1;
pub const STATE_PROCESSING_SNAPSHOT: u64 = 2;
pub const STATE_APPLYING_DELTAS: u64 = 3;
pub const STATE_SYNCHRONIZED: u64 = 4;
pub const STATE_RESYNCING: u64 = 5;

impl OrderBookSync {
    pub fn new() -> Self {
        Self {
            snapshots: SnapshotManager::new(),
            validator: ChecksumValidator::new(),
            state: AtomicU64::new(STATE_DISCONNECTED),
            is_syncing: AtomicBool::new(false),
            sync_start_ns: AtomicU64::new(0),
            resync_count: AtomicU64::new(0),
        }
    }
    
    /// Start sync process
    pub fn start_sync(&self, symbol_id: u64) {
        self.state.store(STATE_REQUESTING_SNAPSHOT, Ordering::Release);
        self.is_syncing.store(true, Ordering::Release);
        self.sync_start_ns.store(current_time_ns(), Ordering::Release);
    }
    
    /// Process received snapshot
    pub fn on_snapshot(&self, mut snapshot: OrderBookSnapshot) {
        self.state.store(STATE_PROCESSING_SNAPSHOT, Ordering::Release);
        
        // Compute and set checksum
        let bids: Vec<(i64, i64)> = snapshot.bids().iter().map(|l| (l.price, l.quantity)).collect();
        let asks: Vec<(i64, i64)> = snapshot.asks().iter().map(|l| (l.price, l.quantity)).collect();
        
        let checksum = self.validator.compute_from_levels(&bids, &asks);
        self.validator.set_expected(checksum);
        snapshot.checksum = checksum;
        
        // Store snapshot
        self.snapshots.store(snapshot);
        
        // Transition to delta application
        self.state.store(STATE_APPLYING_DELTAS, Ordering::Release);
    }
    
    /// Apply delta update
    pub fn apply_delta(&self, bids: &[(i64, i64)], asks: &[(i64, i64)], sequence: u64, expected_checksum: u32) -> DeltaResult {
        if self.state.load(Ordering::Acquire) != STATE_APPLYING_DELTAS 
            && self.state.load(Ordering::Acquire) != STATE_SYNCHRONIZED 
        {
            return DeltaResult {
                applied: false,
                should_resync: true,
                reason: "Not in synchronized state",
            };
        }
        
        self.validator.set_expected(expected_checksum);
        let computed = self.validator.compute_from_levels(bids, asks);
        
        let validation = self.validator.validate(computed, sequence);
        
        if validation.is_valid {
            // Successfully applied
            if self.state.load(Ordering::Acquire) == STATE_APPLYING_DELTAS {
                self.state.store(STATE_SYNCHRONIZED, Ordering::Release);
                self.is_syncing.store(false, Ordering::Release);
            }
            
            DeltaResult {
                applied: true,
                should_resync: false,
                reason: "OK",
            }
        } else {
            DeltaResult {
                applied: false,
                should_resync: validation.should_resync,
                reason: if validation.should_resync { "Checksum mismatch - resync required" } else { "Checksum mismatch" },
            }
        }
    }
    
    /// Trigger resync due to gap or corruption
    pub fn trigger_resync(&self) {
        self.state.store(STATE_RESYNCING, Ordering::Release);
        self.resync_count.fetch_add(1, Ordering::Relaxed);
        self.validator.clear_corruption();
    }
    
    /// Get current sync status
    pub fn get_status(&self) -> SyncStatus {
        let state = self.state.load(Ordering::Acquire);
        let sync_start = self.sync_start_ns.load(Ordering::Acquire);
        let now = current_time_ns();
        
        SyncStatus {
            state,
            is_syncing: self.is_syncing.load(Ordering::Acquire),
            sync_duration_ns: if sync_start > 0 { now.saturating_sub(sync_start) } else { 0 },
            resync_count: self.resync_count.load(Ordering::Relaxed),
            validator_stats: self.validator.get_stats(),
            snapshot_stats: self.snapshots.get_stats(),
        }
    }
    
    /// Check if fully synchronized
    pub fn is_synchronized(&self) -> bool {
        self.state.load(Ordering::Acquire) == STATE_SYNCHRONIZED
    }
}

/// Result of delta application
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct DeltaResult {
    pub applied: bool,
    pub should_resync: bool,
    pub reason: &'static str,
}

/// Sync status snapshot
#[derive(Debug, Clone)]
#[repr(C)]
pub struct SyncStatus {
    pub state: u64,
    pub is_syncing: bool,
    pub sync_duration_ns: u64,
    pub resync_count: u64,
    pub validator_stats: ChecksumStats,
    pub snapshot_stats: SnapshotStats,
}

/// Get current time in nanoseconds
#[inline]
fn current_time_ns() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

impl Default for OrderBookSync {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_sync_flow() {
        let sync = OrderBookSync::new();
        
        assert_eq!(sync.get_status().state, STATE_DISCONNECTED);
        
        // Start sync
        sync.start_sync(12345);
        assert_eq!(sync.get_status().state, STATE_REQUESTING_SNAPSHOT);
        assert!(sync.is_syncing.load(Ordering::Acquire));
        
        // Create and process snapshot
        let mut snapshot = OrderBookSnapshot::empty();
        snapshot.symbol_id = 12345;
        snapshot.sequence = 100;
        snapshot.set_bid(0, PriceLevel { price: 100_00000000, quantity: 50_00000000, order_count: 1, _padding: 0 });
        snapshot.set_ask(0, PriceLevel { price: 101_00000000, quantity: 30_00000000, order_count: 1, _padding: 0 });
        
        sync.on_snapshot(snapshot);
        assert_eq!(sync.get_status().state, STATE_APPLYING_DELTAS);
        
        // Apply valid delta
        let bids = vec![(100_00000000i64, 55_00000000i64)];
        let asks = vec![(101_00000000i64, 25_00000000i64)];
        
        // Get expected checksum
        let expected = sync.validator.compute_from_levels(&bids, &asks);
        
        let result = sync.apply_delta(&bids, &asks, 101, expected);
        assert!(result.applied);
        assert!(!result.should_resync);
        assert!(sync.is_synchronized());
    }
    
    #[test]
    fn test_resync_trigger() {
        let sync = OrderBookSync::new();
        
        sync.trigger_resync();
        assert_eq!(sync.get_status().state, STATE_RESYNCING);
        assert_eq!(sync.get_status().resync_count, 1);
    }
}
