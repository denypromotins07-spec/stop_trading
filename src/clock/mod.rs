//! Clock Module Root
//! 
//! Integrates time synchronization with telemetry and safety modules.
//! Exports all clock-related components.

pub mod global_clock;
pub mod heartbeat;

pub use global_clock::{
    GlobalClock,
    SyncState,
    SyncStats,
    TimeSource,
    TimeInForce,
};

pub use heartbeat::{
    HeartbeatMonitor,
    HeartbeatRunner,
    HeartbeatConfig,
    HeartbeatStatus,
    HeartbeatStats,
    LatencyRecord,
};

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

/// Time synchronization manager coordinating clock and heartbeat
#[repr(C)]
pub struct TimeSyncManager {
    /// Global clock
    clock: Arc<GlobalClock>,
    /// Heartbeat monitor
    heartbeat: Arc<HeartbeatMonitor>,
    /// Manager is running
    is_running: AtomicBool,
    /// Sync errors count
    sync_errors: AtomicU64,
    /// Last successful sync timestamp
    last_sync_ns: AtomicU64,
}

impl TimeSyncManager {
    pub fn new(heartbeat_config: Option<HeartbeatConfig>) -> Self {
        let config = heartbeat_config.unwrap_or_default();
        
        Self {
            clock: Arc::new(GlobalClock::new()),
            heartbeat: Arc::new(HeartbeatMonitor::new(config)),
            is_running: AtomicBool::new(true),
            sync_errors: AtomicU64::new(0),
            last_sync_ns: AtomicU64::new(0),
        }
    }

    /// Get current time in nanoseconds
    #[inline]
    pub fn now_ns(&self) -> u64 {
        self.clock.now_ns()
    }

    /// Get exchange-adjusted time
    #[inline]
    pub fn exchange_now_ns(&self) -> u64 {
        self.clock.exchange_now_ns()
    }

    /// Update exchange time offset
    #[inline]
    pub fn sync_exchange_time(&self, offset_ns: i64) {
        self.clock.update_exchange_offset(offset_ns);
        let now_ns = self.now_ns();
        self.last_sync_ns.store(now_ns, Ordering::Release);
    }

    /// Record heartbeat with latency measurement
    #[inline]
    pub fn record_heartbeat(&self, latency_ns: u64) -> HeartbeatStatus {
        self.heartbeat.beat(latency_ns)
    }

    /// Check system health
    #[inline]
    pub fn is_healthy(&self) -> bool {
        self.is_running.load(Ordering::Acquire)
            && !self.heartbeat.is_defense_triggered()
            && self.heartbeat.get_status() != HeartbeatStatus::Critical
    }

    /// Get clock reference
    #[inline]
    pub fn get_clock(&self) -> &GlobalClock {
        &self.clock
    }

    /// Get heartbeat monitor reference
    #[inline]
    pub fn get_heartbeat(&self) -> &HeartbeatMonitor {
        &self.heartbeat
    }

    /// Record sync error
    #[inline]
    pub fn record_sync_error(&self) {
        self.sync_errors.fetch_add(1, Ordering::Relaxed);
    }

    /// Get combined statistics
    #[inline]
    pub fn get_stats(&self) -> TimeSyncStats {
        TimeSyncStats {
            clock_stats: self.clock.get_sync_stats(),
            heartbeat_stats: self.heartbeat.get_stats(),
            sync_errors: self.sync_errors.load(Ordering::Relaxed),
            last_sync_ns: self.last_sync_ns.load(Ordering::Relaxed),
            is_running: self.is_running.load(Ordering::Acquire),
        }
    }

    /// Start time sync
    #[inline]
    pub fn start(&self) {
        self.is_running.store(true, Ordering::Release);
        self.clock.start();
        self.heartbeat.start();
    }

    /// Stop time sync
    #[inline]
    pub fn stop(&self) {
        self.is_running.store(false, Ordering::Release);
        self.clock.stop();
        self.heartbeat.stop();
    }

    /// Emergency stop (for kill switch integration)
    #[inline]
    pub fn emergency_stop(&self) {
        self.stop();
        self.heartbeat.reset_defense();
    }
}

/// Combined time synchronization statistics
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TimeSyncStats {
    pub clock_stats: SyncStats,
    pub heartbeat_stats: HeartbeatStats,
    pub sync_errors: u64,
    pub last_sync_ns: u64,
    pub is_running: bool,
}

/// Telemetry integration for clock metrics
#[repr(C)]
pub struct ClockTelemetry {
    /// Events emitted counter
    events_emitted: AtomicU64,
    /// Last telemetry timestamp
    last_telemetry_ns: AtomicU64,
    /// Telemetry enabled
    enabled: AtomicBool,
}

impl ClockTelemetry {
    pub fn new() -> Self {
        Self {
            events_emitted: AtomicU64::new(0),
            last_telemetry_ns: AtomicU64::new(0),
            enabled: AtomicBool::new(true),
        }
    }

    /// Emit telemetry event
    #[inline]
    pub fn emit(&self, manager: &TimeSyncManager) {
        if !self.enabled.load(Ordering::Acquire) {
            return;
        }

        let stats = manager.get_stats();
        // In production, send to telemetry backend
        // For now, just record the emission
        
        self.events_emitted.fetch_add(1, Ordering::Relaxed);
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        self.last_telemetry_ns.store(now_ns, Ordering::Release);
    }

    /// Enable telemetry
    #[inline]
    pub fn enable(&self) {
        self.enabled.store(true, Ordering::Release);
    }

    /// Disable telemetry
    #[inline]
    pub fn disable(&self) {
        self.enabled.store(false, Ordering::Release);
    }

    /// Get events emitted count
    #[inline]
    pub fn get_events_count(&self) -> u64 {
        self.events_emitted.load(Ordering::Relaxed)
    }
}

impl Default for ClockTelemetry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_sync_manager() {
        let manager = TimeSyncManager::new(None);
        
        assert!(manager.is_healthy());
        
        let now = manager.now_ns();
        assert!(now > 0);
        
        // Sync exchange time
        manager.sync_exchange_time(1_000_000);
        let exchange_time = manager.exchange_now_ns();
        assert!(exchange_time > now);
    }

    #[test]
    fn test_heartbeat_integration() {
        let manager = TimeSyncManager::new(None);
        
        // Normal heartbeat
        let status = manager.record_heartbeat(1_000_000);
        assert_eq!(status, HeartbeatStatus::Healthy);
        assert!(manager.is_healthy());
        
        // Critical latency
        let status = manager.record_heartbeat(20_000_000);
        assert_eq!(status, HeartbeatStatus::Critical);
        assert!(!manager.is_healthy());
    }

    #[test]
    fn test_telemetry() {
        let manager = TimeSyncManager::new(None);
        let telemetry = ClockTelemetry::new();
        
        telemetry.emit(&manager);
        assert_eq!(telemetry.get_events_count(), 1);
        
        telemetry.disable();
        telemetry.emit(&manager);
        assert_eq!(telemetry.get_events_count(), 1); // Should not increment
    }
}
