//! High-Precision Global Event Clock
//! 
//! Synchronizes system TSC, exchange server time, and NTP offsets.
//! Ensures all latency-adjusted timestamps and time-in-force expirations are aligned.

use std::sync::atomic::{AtomicI64, AtomicU64, AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// Cache line size
const CACHE_LINE_SIZE: usize = 64;

/// Nanoseconds per second
const NS_PER_SECOND: u64 = 1_000_000_000;

/// Padded atomic u64
#[repr(C)]
#[derive(Debug)]
pub struct PaddedAtomicU64 {
    _pad1: [u8; CACHE_LINE_SIZE - 8],
    value: AtomicU64,
    _pad2: [u8; CACHE_LINE_SIZE],
}

impl PaddedAtomicU64 {
    pub fn new(initial: u64) -> Self {
        Self {
            _pad1: [0u8; CACHE_LINE_SIZE - 8],
            value: AtomicU64::new(initial),
            _pad2: [0u8; CACHE_LINE_SIZE],
        }
    }

    #[inline]
    pub fn load(&self, ordering: Ordering) -> u64 {
        self.value.load(ordering)
    }

    #[inline]
    pub fn store(&self, val: u64, ordering: Ordering) {
        self.value.store(val, ordering);
    }

    #[inline]
    pub fn fetch_add(&self, val: u64, ordering: Ordering) -> u64 {
        self.value.fetch_add(val, ordering)
    }
}

/// Padded atomic i64
#[repr(C)]
#[derive(Debug)]
pub struct PaddedAtomicI64 {
    _pad1: [u8; CACHE_LINE_SIZE - 8],
    value: AtomicI64,
    _pad2: [u8; CACHE_LINE_SIZE],
}

impl PaddedAtomicI64 {
    pub fn new(initial: i64) -> Self {
        Self {
            _pad1: [0u8; CACHE_LINE_SIZE - 8],
            value: AtomicI64::new(initial),
            _pad2: [0u8; CACHE_LINE_SIZE],
        }
    }

    #[inline]
    pub fn load(&self, ordering: Ordering) -> i64 {
        self.value.load(ordering)
    }

    #[inline]
    pub fn store(&self, val: i64, ordering: Ordering) {
        self.value.store(val, ordering);
    }

    #[inline]
    pub fn fetch_add(&self, val: i64, ordering: Ordering) -> i64 {
        self.value.fetch_add(val, ordering)
    }
}

/// Time source types
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeSource {
    SystemTSC,
    ExchangeServer,
    NTP,
    PTP, // Precision Time Protocol
}

/// Clock synchronization state
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SyncState {
    /// Current offset from true time (nanoseconds)
    pub offset_ns: i64,
    /// Estimated jitter (nanoseconds)
    pub jitter_ns: u64,
    /// Last sync timestamp
    pub last_sync_ns: u64,
    /// Sync quality (0-100)
    pub quality: u8,
    /// Time source
    pub source: TimeSource,
}

/// Global event clock
#[repr(C)]
pub struct GlobalClock {
    /// System TSC base (nanoseconds at startup)
    tsc_base_ns: AtomicU64,
    /// Offset from system time to exchange time (nanoseconds)
    exchange_offset_ns: PaddedAtomicI64,
    /// Offset from system time to NTP time (nanoseconds)
    ntp_offset_ns: PaddedAtomicI64,
    /// Current sync state
    sync_state: PaddedAtomicU64, // Pointer to sync state conceptually
    /// Clock is synchronized
    is_synchronized: AtomicBool,
    /// Clock is running
    is_running: AtomicBool,
    /// Total syncs performed
    total_syncs: PaddedAtomicU64,
    /// Last tick timestamp (for latency measurement)
    last_tick_ns: PaddedAtomicU64,
    /// Tick counter
    tick_count: PaddedAtomicU64,
}

impl GlobalClock {
    pub fn new() -> Self {
        let now_ns = Self::get_system_time_ns();
        
        Self {
            tsc_base_ns: AtomicU64::new(now_ns),
            exchange_offset_ns: PaddedAtomicI64::new(0),
            ntp_offset_ns: PaddedAtomicI64::new(0),
            sync_state: PaddedAtomicU64::new(0),
            is_synchronized: AtomicBool::new(false),
            is_running: AtomicBool::new(true),
            total_syncs: PaddedAtomicU64::new(0),
            last_tick_ns: PaddedAtomicU64::new(now_ns),
            tick_count: PaddedAtomicU64::new(0),
        }
    }

    /// Get current system time in nanoseconds
    #[inline]
    fn get_system_time_ns() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64
    }

    /// Get high-resolution TSC-based time (monotonic)
    #[inline]
    pub fn now_ns(&self) -> u64 {
        if !self.is_running.load(Ordering::Acquire) {
            return self.tsc_base_ns.load(Ordering::Relaxed);
        }

        // Use instant for monotonic time
        let elapsed = Instant::now().duration_since(Instant::now());
        // In production, use actual TSC reading via rdtsc intrinsic
        
        let base = self.tsc_base_ns.load(Ordering::Relaxed);
        base + elapsed.as_nanos() as u64
    }

    /// Get exchange-adjusted time
    #[inline]
    pub fn exchange_now_ns(&self) -> u64 {
        let system_ns = self.now_ns();
        let offset = self.exchange_offset_ns.load(Ordering::Acquire);
        
        if offset >= 0 {
            system_ns.saturating_add(offset as u64)
        } else {
            system_ns.saturating_sub((-offset) as u64)
        }
    }

    /// Get NTP-adjusted time
    #[inline]
    pub fn ntp_now_ns(&self) -> u64 {
        let system_ns = self.now_ns();
        let offset = self.ntp_offset_ns.load(Ordering::Acquire);
        
        if offset >= 0 {
            system_ns.saturating_add(offset as u64)
        } else {
            system_ns.saturating_sub((-offset) as u64)
        }
    }

    /// Update exchange time offset
    #[inline]
    pub fn update_exchange_offset(&self, offset_ns: i64) {
        self.exchange_offset_ns.store(offset_ns, Ordering::Release);
        self.is_synchronized.store(true, Ordering::Release);
        self.total_syncs.fetch_add(1, Ordering::Relaxed);
    }

    /// Update NTP offset
    #[inline]
    pub fn update_ntp_offset(&self, offset_ns: i64) {
        self.ntp_offset_ns.store(offset_ns, Ordering::Release);
        self.total_syncs.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a tick (for latency measurement)
    #[inline]
    pub fn tick(&self) -> u64 {
        let now_ns = self.now_ns();
        let last_ns = self.last_tick_ns.load(Ordering::Acquire);
        self.last_tick_ns.store(now_ns, Ordering::Release);
        self.tick_count.fetch_add(1, Ordering::Relaxed);
        
        if last_ns > 0 {
            now_ns - last_ns // Return inter-tick interval
        } else {
            0
        }
    }

    /// Get tick-to-trade latency (time since last tick)
    #[inline]
    pub fn get_tick_latency_ns(&self) -> u64 {
        let now_ns = self.now_ns();
        let last_ns = self.last_tick_ns.load(Ordering::Acquire);
        now_ns.saturating_sub(last_ns)
    }

    /// Get tick count
    #[inline]
    pub fn get_tick_count(&self) -> u64 {
        self.tick_count.load(Ordering::Relaxed)
    }

    /// Convert timestamp to exchange time
    #[inline]
    pub fn to_exchange_time(&self, system_ns: u64) -> u64 {
        let offset = self.exchange_offset_ns.load(Ordering::Acquire);
        if offset >= 0 {
            system_ns.saturating_add(offset as u64)
        } else {
            system_ns.saturating_sub((-offset) as u64)
        }
    }

    /// Convert exchange timestamp to system time
    #[inline]
    pub fn from_exchange_time(&self, exchange_ns: u64) -> u64 {
        let offset = self.exchange_offset_ns.load(Ordering::Acquire);
        if offset >= 0 {
            exchange_ns.saturating_sub(offset as u64)
        } else {
            exchange_ns.saturating_add((-offset) as u64)
        }
    }

    /// Check if clock is synchronized
    #[inline]
    pub fn is_synchronized(&self) -> bool {
        self.is_synchronized.load(Ordering::Acquire)
    }

    /// Get sync statistics
    #[inline]
    pub fn get_sync_stats(&self) -> SyncStats {
        SyncStats {
            total_syncs: self.total_syncs.load(Ordering::Relaxed),
            exchange_offset_ns: self.exchange_offset_ns.load(Ordering::Relaxed),
            ntp_offset_ns: self.ntp_offset_ns.load(Ordering::Relaxed),
            is_synchronized: self.is_synchronized(),
        }
    }

    /// Start clock
    #[inline]
    pub fn start(&self) {
        self.is_running.store(true, Ordering::Release);
    }

    /// Stop clock
    #[inline]
    pub fn stop(&self) {
        self.is_running.store(false, Ordering::Release);
    }

    /// Get current time as Duration since epoch
    #[inline]
    pub fn now_duration(&self) -> Duration {
        Duration::from_nanos(self.now_ns())
    }

    /// Get exchange time as Duration since epoch
    #[inline]
    pub fn exchange_duration(&self) -> Duration {
        Duration::from_nanos(self.exchange_now_ns())
    }
}

impl Default for GlobalClock {
    fn default() -> Self {
        Self::new()
    }
}

/// Synchronization statistics
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SyncStats {
    pub total_syncs: u64,
    pub exchange_offset_ns: i64,
    pub ntp_offset_ns: i64,
    pub is_synchronized: bool,
}

/// Time-in-force expiration helper
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TimeInForce {
    /// Expiration timestamp (ns)
    pub expiry_ns: u64,
    /// Is good-till-cancelled
    pub gtc: bool,
    /// Is immediate-or-cancel
    pub ioc: bool,
    /// Is fill-or-kill
    pub fok: bool,
}

impl TimeInForce {
    /// Create GTC order
    pub fn gtc() -> Self {
        Self {
            expiry_ns: u64::MAX,
            gtc: true,
            ioc: false,
            fok: false,
        }
    }

    /// Create IOC order
    pub fn ioc() -> Self {
        Self {
            expiry_ns: 0, // Immediate
            gtc: false,
            ioc: true,
            fok: false,
        }
    }

    /// Create FOK order
    pub fn fok() -> Self {
        Self {
            expiry_ns: 0, // Immediate
            gtc: false,
            ioc: false,
            fok: true,
        }
    }

    /// Create order with specific expiry
    pub fn with_expiry(expiry_ns: u64) -> Self {
        Self {
            expiry_ns,
            gtc: false,
            ioc: false,
            fok: false,
        }
    }

    /// Create order with duration from now
    pub fn with_duration(duration_ms: u64, clock: &GlobalClock) -> Self {
        let now_ns = clock.now_ns();
        let expiry_ns = now_ns + (duration_ms * 1_000_000);
        Self {
            expiry_ns,
            gtc: false,
            ioc: false,
            fok: false,
        }
    }

    /// Check if expired
    #[inline]
    pub fn is_expired(&self, clock: &GlobalClock) -> bool {
        if self.gtc {
            return false;
        }
        clock.now_ns() >= self.expiry_ns
    }

    /// Get remaining time in nanoseconds
    #[inline]
    pub fn remaining_ns(&self, clock: &GlobalClock) -> u64 {
        if self.gtc {
            return u64::MAX;
        }
        self.expiry_ns.saturating_sub(clock.now_ns())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_global_clock() {
        let clock = GlobalClock::new();
        
        let now = clock.now_ns();
        assert!(now > 0);
        
        // Test offset
        clock.update_exchange_offset(1_000_000); // +1ms
        let exchange_time = clock.exchange_now_ns();
        assert!(exchange_time > now);
    }

    #[test]
    fn test_tick_measurement() {
        let clock = GlobalClock::new();
        
        clock.tick();
        std::thread::sleep(Duration::from_millis(10));
        let latency = clock.tick();
        
        // Should be approximately 10ms (with some variance)
        assert!(latency > 5_000_000); // > 5ms
        assert!(latency < 100_000_000); // < 100ms
        
        assert_eq!(clock.get_tick_count(), 2);
    }

    #[test]
    fn test_time_in_force() {
        let clock = GlobalClock::new();
        
        // GTC never expires
        let gtc = TimeInForce::gtc();
        assert!(!gtc.is_expired(&clock));
        
        // Short duration
        let tif = TimeInForce::with_duration(100, &clock); // 100ms
        assert!(!tif.is_expired(&clock));
        
        std::thread::sleep(Duration::from_millis(150));
        assert!(tif.is_expired(&clock));
    }
}
