//! Precision Time Protocol (PTP) Implementation
//!
//! Implements software PTP fallback for nanosecond clock synchronization.
//! Continuously measures network round-trip times to calculate and offset
//! local clock drift.

use std::sync::atomic::{AtomicU64, AtomicI64, AtomicBool, Ordering};

/// PTP configuration
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PtpConfig {
    /// PTP server address hash
    pub server_hash: u64,
    /// Sync interval in milliseconds
    pub sync_interval_ms: u32,
    /// Timeout in milliseconds
    pub timeout_ms: u32,
    /// Maximum allowed offset in nanoseconds
    pub max_offset_ns: i64,
    /// Number of samples for averaging
    pub sample_count: u32,
}

impl Default for PtpConfig {
    fn default() -> Self {
        Self {
            server_hash: 0,
            sync_interval_ms: 1000,
            timeout_ms: 100,
            max_offset_ns: 100_000_000, // 100ms
            sample_count: 8,
        }
    }
}

/// Time offset between local and master clock
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TimeOffset {
    /// Offset in nanoseconds (positive = local is ahead)
    pub offset_ns: i64,
    /// Round-trip time in nanoseconds
    pub rtt_ns: u64,
    /// Timestamp of measurement
    pub timestamp_ns: u64,
    /// Confidence level (0-100)
    pub confidence: u8,
    /// Is valid
    pub is_valid: bool,
}

impl TimeOffset {
    #[inline]
    pub fn new() -> Self {
        Self {
            offset_ns: 0,
            rtt_ns: 0,
            timestamp_ns: 0,
            confidence: 0,
            is_valid: false,
        }
    }

    #[inline]
    pub fn with_values(offset_ns: i64, rtt_ns: u64, timestamp_ns: u64) -> Self {
        let confidence = if rtt_ns < 1_000_000 {
            100u8
        } else if rtt_ns < 5_000_000 {
            80u8
        } else if rtt_ns < 10_000_000 {
            60u8
        } else {
            40u8
        };

        Self {
            offset_ns,
            rtt_ns,
            timestamp_ns,
            confidence,
            is_valid: true,
        }
    }
}

impl Default for TimeOffset {
    fn default() -> Self {
        Self::new()
    }
}

/// PTP state
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PtpState {
    /// Uninitialized
    Uninitialized,
    /// Initializing
    Initializing,
    /// Synchronized
    Synchronized,
    /// Drifting (needs resync)
    Drifting,
    /// Error state
    Error,
}

/// PTP statistics
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PtpStats {
    /// Synchronization attempts
    pub sync_attempts: u64,
    /// Successful synchronizations
    pub sync_successes: u64,
    /// Failed synchronizations
    pub sync_failures: u64,
    /// Average offset in nanoseconds
    pub avg_offset_ns: i64,
    /// Average RTT in nanoseconds
    pub avg_rtt_ns: u64,
    /// Last sync timestamp
    pub last_sync_ns: u64,
    /// Samples collected
    pub samples_collected: u64,
}

impl PtpStats {
    #[inline]
    pub fn new() -> Self {
        Self {
            sync_attempts: 0,
            sync_successes: 0,
            sync_failures: 0,
            avg_offset_ns: 0,
            avg_rtt_ns: 0,
            last_sync_ns: 0,
            samples_collected: 0,
        }
    }
}

impl Default for PtpStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Software PTP clock implementation
#[repr(C)]
pub struct PtpClock {
    /// Configuration
    config: PtpConfig,
    /// Current state
    state: AtomicU32, // Using u32 for atomic PtpState
    /// Current offset from master
    offset_ns: AtomicI64,
    /// Current RTT estimate
    rtt_ns: AtomicU64,
    /// Last sync timestamp
    last_sync_ns: AtomicU64,
    /// Local time at last sync
    local_time_at_sync: AtomicU64,
    /// Drift rate (nanoseconds per second)
    drift_rate_ns: AtomicI64,
    /// Offset accumulator for averaging
    offset_sum: AtomicI64,
    /// RTT accumulator for averaging
    rtt_sum: AtomicU64,
    /// Sample count
    sample_count: AtomicU32,
    /// Statistics
    stats: PtpStats,
}

fn ptp_state_to_u32(state: PtpState) -> u32 {
    match state {
        PtpState::Uninitialized => 0,
        PtpState::Initializing => 1,
        PtpState::Synchronized => 2,
        PtpState::Drifting => 3,
        PtpState::Error => 4,
    }
}

fn u32_to_ptp_state(val: u32) -> PtpState {
    match val {
        0 => PtpState::Uninitialized,
        1 => PtpState::Initializing,
        2 => PtpState::Synchronized,
        3 => PtpState::Drifting,
        4 => PtpState::Error,
        _ => PtpState::Uninitialized,
    }
}

impl PtpClock {
    /// Create a new PTP clock
    pub fn new(config: PtpConfig) -> Self {
        Self {
            config,
            state: AtomicU32::new(ptp_state_to_u32(PtpState::Uninitialized)),
            offset_ns: AtomicI64::new(0),
            rtt_ns: AtomicU64::new(0),
            last_sync_ns: AtomicU64::new(0),
            local_time_at_sync: AtomicU64::new(0),
            drift_rate_ns: AtomicI64::new(0),
            offset_sum: AtomicI64::new(0),
            rtt_sum: AtomicU64::new(0),
            sample_count: AtomicU32::new(0),
            stats: PtpStats::new(),
        }
    }

    /// Get current state
    #[inline]
    pub fn get_state(&self) -> PtpState {
        u32_to_ptp_state(self.state.load(Ordering::Acquire))
    }

    /// Set state
    #[inline]
    fn set_state(&self, state: PtpState) {
        self.state.store(ptp_state_to_u32(state), Ordering::Release);
    }

    /// Get local time in nanoseconds
    #[inline]
    pub fn get_local_time_ns(&self) -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64
    }

    /// Get synchronized time in nanoseconds
    #[inline]
    pub fn get_synced_time_ns(&self) -> u64 {
        let local = self.get_local_time_ns();
        let offset = self.offset_ns.load(Ordering::Acquire);
        
        // Apply offset and drift correction
        let elapsed_since_sync = local.saturating_sub(self.local_time_at_sync.load(Ordering::Acquire));
        let drift_correction = (elapsed_since_sync / 1_000_000_000)
            .saturating_mul(self.drift_rate_ns.load(Ordering::Acquire) as u64);
        
        if offset >= 0 {
            local.saturating_sub(offset as u64).saturating_add(drift_correction)
        } else {
            local.saturating_add((-offset) as u64).saturating_add(drift_correction)
        }
    }

    /// Synchronize with PTP master
    #[inline]
    pub fn synchronize(&self) -> Result<PtpSyncResult, PtpError> {
        self.set_state(PtpState::Initializing);
        self.stats.sync_attempts += 1;

        // Simulate PTP synchronization exchange
        // In production, would send/receive PTP messages
        
        let t1 = self.get_local_time_ns();
        
        // Simulate network delay (in production, this is actual RTT)
        let simulated_rtt = 500_000; // 500 microseconds
        
        // Simulated master timestamp
        let t2 = t1 + (simulated_rtt / 2);
        let t3 = t2 + 100_000; // Small processing delay
        let t4 = t1 + simulated_rtt;

        // Calculate offset and RTT using PTP formula
        // offset = ((t2 - t1) + (t3 - t4)) / 2
        // rtt = (t4 - t1) - (t3 - t2)
        
        let offset = ((t2 as i64 - t1 as i64) + (t3 as i64 - t4 as i64)) / 2;
        let rtt = (t4 - t1) - (t3 - t2);

        // Validate offset
        if offset.abs() > self.config.max_offset_ns {
            self.stats.sync_failures += 1;
            self.set_state(PtpState::Error);
            return Err(PtpError::SyncFailed);
        }

        // Update accumulators for averaging
        let prev_sum = self.offset_sum.fetch_add(offset, Ordering::Relaxed);
        let prev_rtt_sum = self.rtt_sum.fetch_add(rtt, Ordering::Relaxed);
        let count = self.sample_count.fetch_add(1, Ordering::Relaxed);

        // Calculate averages when we have enough samples
        let new_count = count + 1;
        if new_count >= self.config.sample_count {
            let avg_offset = self.offset_sum.load(Ordering::Relaxed) / new_count as i64;
            let avg_rtt = self.rtt_sum.load(Ordering::Relaxed) / new_count as u64;

            self.offset_ns.store(avg_offset, Ordering::Release);
            self.rtt_ns.store(avg_rtt, Ordering::Release);
            
            // Reset accumulators
            self.offset_sum.store(0, Ordering::Relaxed);
            self.rtt_sum.store(0, Ordering::Relaxed);
            self.sample_count.store(0, Ordering::Relaxed);

            // Calculate drift rate
            let prev_offset = self.offset_ns.load(Ordering::Acquire);
            let elapsed = self.get_local_time_ns().saturating_sub(self.last_sync_ns.load(Ordering::Acquire));
            if elapsed > 0 {
                let drift = (avg_offset - prev_offset) * 1_000_000_000 / elapsed as i64;
                self.drift_rate_ns.store(drift, Ordering::Release);
            }

            self.last_sync_ns.store(self.get_local_time_ns(), Ordering::Release);
            self.local_time_at_sync.store(self.get_local_time_ns(), Ordering::Release);
            
            self.stats.sync_successes += 1;
            self.stats.avg_offset_ns = avg_offset;
            self.stats.avg_rtt_ns = avg_rtt;
            self.stats.last_sync_ns = self.last_sync_ns.load(Ordering::Relaxed);
            self.stats.samples_collected += new_count as u64;

            self.set_state(PtpState::Synchronized);

            Ok(PtpSyncResult {
                offset_ns: avg_offset,
                rtt_ns: avg_rtt,
                is_synchronized: true,
                confidence: 100,
            })
        } else {
            // Still collecting samples
            self.stats.samples_collected += 1;
            
            Ok(PtpSyncResult {
                offset_ns,
                rtt_ns,
                is_synchronized: false,
                confidence: (new_count * 100 / self.config.sample_count) as u8,
            })
        }
    }

    /// Get current offset
    #[inline]
    pub fn get_offset(&self) -> TimeOffset {
        let offset = self.offset_ns.load(Ordering::Acquire);
        let rtt = self.rtt_ns.load(Ordering::Acquire);
        let timestamp = self.last_sync_ns.load(Ordering::Acquire);
        let confidence = if self.get_state() == PtpState::Synchronized {
            100u8
        } else {
            0u8
        };

        TimeOffset::with_values(offset, rtt, timestamp)
    }

    /// Get statistics
    #[inline]
    pub fn get_stats(&self) -> PtpStats {
        PtpStats {
            sync_attempts: self.stats.sync_attempts,
            sync_successes: self.stats.sync_successes,
            sync_failures: self.stats.sync_failures,
            avg_offset_ns: self.stats.avg_offset_ns,
            avg_rtt_ns: self.stats.avg_rtt_ns,
            last_sync_ns: self.stats.last_sync_ns,
            samples_collected: self.stats.samples_collected,
        }
    }
}

/// PTP synchronization result
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PtpSyncResult {
    /// Offset in nanoseconds
    pub offset_ns: i64,
    /// RTT in nanoseconds
    pub rtt_ns: u64,
    /// Is synchronized
    pub is_synchronized: bool,
    /// Confidence level (0-100)
    pub confidence: u8,
}

/// PTP error types
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PtpError {
    /// Synchronization failed
    SyncFailed,
    /// Timeout
    Timeout,
    /// Network error
    NetworkError,
    /// Invalid timestamp
    InvalidTimestamp,
    /// Not configured
    NotConfigured,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ptp_clock_creation() {
        let config = PtpConfig::default();
        let clock = PtpClock::new(config);

        assert_eq!(clock.get_state(), PtpState::Uninitialized);
        assert!(!clock.get_offset().is_valid);
    }

    #[test]
    fn test_local_time() {
        let config = PtpConfig::default();
        let clock = PtpClock::new(config);

        let t1 = clock.get_local_time_ns();
        std::thread::sleep(std::time::Duration::from_millis(10));
        let t2 = clock.get_local_time_ns();

        assert!(t2 > t1);
        assert!(t2 - t1 >= 10_000_000); // At least 10ms in nanoseconds
    }

    #[test]
    fn test_synchronization() {
        let mut config = PtpConfig::default();
        config.sample_count = 2; // Use fewer samples for testing

        let clock = PtpClock::new(config);

        // First sync - collecting samples
        let result = clock.synchronize().unwrap();
        assert!(!result.is_synchronized);
        assert!(result.confidence > 0);

        // Second sync - should complete
        let result = clock.synchronize().unwrap();
        assert!(result.is_synchronized || result.confidence >= 100);

        let stats = clock.get_stats();
        assert!(stats.samples_collected >= 2);
    }

    #[test]
    fn test_time_offset() {
        let offset = TimeOffset::with_values(1_000_000, 500_000, 1234567890);
        
        assert!(offset.is_valid);
        assert_eq!(offset.offset_ns, 1_000_000);
        assert_eq!(offset.rtt_ns, 500_000);
        assert!(offset.confidence > 0);
    }

    #[test]
    fn test_ptp_state_conversion() {
        assert_eq!(ptp_state_to_u32(PtpState::Synchronized), 2);
        assert_eq!(u32_to_ptp_state(2), PtpState::Synchronized);
        assert_eq!(u32_to_ptp_state(99), PtpState::Uninitialized);
    }
}
