//! Microsecond Heartbeat Monitor
//! 
//! Detects thread starvation, network lag, or matching engine delays.
//! Triggers automated defensive actions if tick-to-trade latency exceeds thresholds.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicI64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Cache line size
const CACHE_LINE_SIZE: usize = 64;

/// Default heartbeat interval (microseconds)
const DEFAULT_HEARTBEAT_INTERVAL_US: u64 = 100; // 100μs

/// Default latency threshold (milliseconds)
const DEFAULT_LATENCY_THRESHOLD_MS: u64 = 10; // 10ms

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

    #[inline]
    pub fn fetch_max(&self, val: u64, ordering: Ordering) -> u64 {
        let mut current = self.value.load(ordering);
        loop {
            if val <= current {
                return current;
            }
            match self.value.compare_exchange_weak(current, val, ordering, ordering) {
                Ok(_) => return val,
                Err(x) => current = x,
            }
        }
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
}

/// Heartbeat status
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeartbeatStatus {
    Healthy,
    Warning,      // Approaching threshold
    Critical,     // Exceeded threshold
    Starved,      // Thread starvation detected
    NetworkLag,   // Network latency spike
    EngineDelay,  // Matching engine delay
}

/// Latency measurement record
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct LatencyRecord {
    /// Timestamp (ns)
    pub timestamp_ns: u64,
    /// Tick-to-trade latency (ns)
    pub t2t_latency_ns: u64,
    /// Network round-trip (ns)
    pub network_rtt_ns: u64,
    /// Queue depth at measurement
    pub queue_depth: u32,
}

/// Heartbeat monitor configuration
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct HeartbeatConfig {
    /// Heartbeat interval (microseconds)
    pub interval_us: u64,
    /// Latency warning threshold (milliseconds)
    pub warning_threshold_ms: u64,
    /// Latency critical threshold (milliseconds)
    pub critical_threshold_ms: u64,
    /// Starvation detection threshold (missed heartbeats)
    pub starvation_threshold: u32,
    /// Enable auto-defensive actions
    pub auto_defense: bool,
}

impl Default for HeartbeatConfig {
    fn default() -> Self {
        Self {
            interval_us: DEFAULT_HEARTBEAT_INTERVAL_US,
            warning_threshold_ms: 5,
            critical_threshold_ms: DEFAULT_LATENCY_THRESHOLD_MS,
            starvation_threshold: 10,
            auto_defense: true,
        }
    }
}

/// Heartbeat monitor state
#[repr(C)]
pub struct HeartbeatMonitor {
    /// Configuration
    config: HeartbeatConfig,
    /// Last heartbeat timestamp (ns)
    last_heartbeat_ns: PaddedAtomicU64,
    /// Current status
    status: AtomicU64, // Encoded HeartbeatStatus
    /// Consecutive missed heartbeats
    missed_count: PaddedAtomicU64,
    /// Maximum latency observed (ns)
    max_latency_ns: PaddedAtomicU64,
    /// Minimum latency observed (ns)
    min_latency_ns: PaddedAtomicU64,
    /// Average latency (exponential moving average, scaled)
    avg_latency_ns: PaddedAtomicU64,
    /// Total heartbeats
    total_heartbeats: PaddedAtomicU64,
    /// Warning count
    warning_count: PaddedAtomicU64,
    /// Critical count
    critical_count: PaddedAtomicU64,
    /// Monitor is running
    is_running: AtomicBool,
    /// Defensive action triggered
    defense_triggered: AtomicBool,
    /// Last latency measurement
    last_latency_ns: PaddedAtomicU64,
}

impl HeartbeatMonitor {
    pub fn new(config: HeartbeatConfig) -> Self {
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        Self {
            config,
            last_heartbeat_ns: PaddedAtomicU64::new(now_ns),
            status: AtomicU64::new(HeartbeatStatus::Healthy as u64),
            missed_count: PaddedAtomicU64::new(0),
            max_latency_ns: PaddedAtomicU64::new(0),
            min_latency_ns: PaddedAtomicU64::new(u64::MAX),
            avg_latency_ns: PaddedAtomicU64::new(0),
            total_heartbeats: PaddedAtomicU64::new(0),
            warning_count: PaddedAtomicU64::new(0),
            critical_count: PaddedAtomicU64::new(0),
            is_running: AtomicBool::new(true),
            defense_triggered: AtomicBool::new(false),
            last_latency_ns: PaddedAtomicU64::new(0),
        }
    }

    /// Record a heartbeat
    #[inline]
    pub fn beat(&self, latency_ns: u64) -> HeartbeatStatus {
        if !self.is_running.load(Ordering::Acquire) {
            return HeartbeatStatus::Starved;
        }

        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        self.last_heartbeat_ns.store(now_ns, Ordering::Release);
        self.total_heartbeats.fetch_add(1, Ordering::Relaxed);
        self.last_latency_ns.store(latency_ns, Ordering::Release);

        // Update statistics
        self.max_latency_ns.fetch_max(latency_ns, Ordering::Relaxed);
        
        // Update min
        let mut min = self.min_latency_ns.load(Ordering::Relaxed);
        loop {
            if latency_ns >= min {
                break;
            }
            match self.min_latency_ns.compare_exchange_weak(min, latency_ns, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => break,
                Err(x) => min = x,
            }
        }

        // Update EMA (alpha = 0.1, scaled by 1000)
        let old_avg = self.avg_latency_ns.load(Ordering::Relaxed);
        let new_avg = (old_avg * 9 + latency_ns) / 10;
        self.avg_latency_ns.store(new_avg, Ordering::Relaxed);

        // Determine status based on latency
        let latency_ms = latency_ns / 1_000_000;
        let status = if latency_ms >= self.config.critical_threshold_ms {
            self.critical_count.fetch_add(1, Ordering::Relaxed);
            if self.config.auto_defense {
                self.defense_triggered.store(true, Ordering::Release);
            }
            HeartbeatStatus::Critical
        } else if latency_ms >= self.config.warning_threshold_ms {
            self.warning_count.fetch_add(1, Ordering::Relaxed);
            HeartbeatStatus::Warning
        } else {
            self.missed_count.store(0, Ordering::Relaxed);
            HeartbeatStatus::Healthy
        };

        self.status.store(status as u64, Ordering::Release);
        status
    }

    /// Check for thread starvation
    #[inline]
    pub fn check_starvation(&self) -> bool {
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        let last = self.last_heartbeat_ns.load(Ordering::Acquire);
        let elapsed_ns = now_ns.saturating_sub(last);
        let interval_ns = self.config.interval_us * 1000;

        let missed = elapsed_ns / interval_ns;
        if missed >= self.config.starvation_threshold as u64 {
            self.missed_count.fetch_add(missed, Ordering::Relaxed);
            self.status.store(HeartbeatStatus::Starved as u64, Ordering::Release);
            return true;
        }
        false
    }

    /// Record network RTT measurement
    #[inline]
    pub fn record_network_rtt(&self, rtt_ns: u64) -> HeartbeatStatus {
        let rtt_ms = rtt_ns / 1_000_000;
        
        if rtt_ms >= self.config.critical_threshold_ms {
            self.status.store(HeartbeatStatus::NetworkLag as u64, Ordering::Release);
            if self.config.auto_defense {
                self.defense_triggered.store(true, Ordering::Release);
            }
            HeartbeatStatus::NetworkLag
        } else {
            self.get_status()
        }
    }

    /// Record matching engine delay
    #[inline]
    pub fn record_engine_delay(&self, delay_ns: u64) -> HeartbeatStatus {
        let delay_ms = delay_ns / 1_000_000;
        
        if delay_ms >= self.config.critical_threshold_ms {
            self.status.store(HeartbeatStatus::EngineDelay as u64, Ordering::Release);
            HeartbeatStatus::EngineDelay
        } else {
            self.get_status()
        }
    }

    /// Get current status
    #[inline]
    pub fn get_status(&self) -> HeartbeatStatus {
        match self.status.load(Ordering::Acquire) {
            0 => HeartbeatStatus::Healthy,
            1 => HeartbeatStatus::Warning,
            2 => HeartbeatStatus::Critical,
            3 => HeartbeatStatus::Starved,
            4 => HeartbeatStatus::NetworkLag,
            5 => HeartbeatStatus::EngineDelay,
            _ => HeartbeatStatus::Healthy,
        }
    }

    /// Check if defense was triggered
    #[inline]
    pub fn is_defense_triggered(&self) -> bool {
        self.defense_triggered.load(Ordering::Acquire)
    }

    /// Reset defense trigger
    #[inline]
    pub fn reset_defense(&self) {
        self.defense_triggered.store(false, Ordering::Release);
    }

    /// Get statistics
    #[inline]
    pub fn get_stats(&self) -> HeartbeatStats {
        HeartbeatStats {
            status: self.get_status(),
            total_heartbeats: self.total_heartbeats.load(Ordering::Relaxed),
            missed_count: self.missed_count.load(Ordering::Relaxed),
            warning_count: self.warning_count.load(Ordering::Relaxed),
            critical_count: self.critical_count.load(Ordering::Relaxed),
            max_latency_ns: self.max_latency_ns.load(Ordering::Relaxed),
            min_latency_ns: self.min_latency_ns.load(Ordering::Relaxed),
            avg_latency_ns: self.avg_latency_ns.load(Ordering::Relaxed),
            last_latency_ns: self.last_latency_ns.load(Ordering::Relaxed),
            defense_triggered: self.is_defense_triggered(),
        }
    }

    /// Stop monitor
    #[inline]
    pub fn stop(&self) {
        self.is_running.store(false, Ordering::Release);
    }

    /// Start monitor
    #[inline]
    pub fn start(&self) {
        self.is_running.store(true, Ordering::Release);
        self.reset_defense();
    }

    /// Get expected interval in nanoseconds
    #[inline]
    pub fn get_interval_ns(&self) -> u64 {
        self.config.interval_us * 1000
    }
}

/// Heartbeat statistics
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct HeartbeatStats {
    pub status: HeartbeatStatus,
    pub total_heartbeats: u64,
    pub missed_count: u64,
    pub warning_count: u64,
    pub critical_count: u64,
    pub max_latency_ns: u64,
    pub min_latency_ns: u64,
    pub avg_latency_ns: u64,
    pub last_latency_ns: u64,
    pub defense_triggered: bool,
}

/// Background heartbeat runner
pub struct HeartbeatRunner {
    monitor: Arc<HeartbeatMonitor>,
    shutdown_flag: Arc<AtomicBool>,
}

impl HeartbeatRunner {
    pub fn new(monitor: Arc<HeartbeatMonitor>) -> Self {
        Self {
            monitor,
            shutdown_flag: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Run heartbeat loop (spawn on dedicated thread)
    pub fn run<F>(&self, mut measure_latency: F)
    where
        F: FnMut() -> u64, // Returns latency in ns
    {
        self.shutdown_flag.store(false, Ordering::Release);
        let interval = Duration::from_micros(self.monitor.config.interval_us);

        while self.monitor.is_running.load(Ordering::Acquire) 
            && !self.shutdown_flag.load(Ordering::Acquire) 
        {
            let start = Instant::now();
            
            // Measure latency
            let latency = measure_latency();
            self.monitor.beat(latency);
            
            // Check for starvation
            self.monitor.check_starvation();

            // Sleep until next interval
            let elapsed = start.elapsed();
            if elapsed < interval {
                std::thread::sleep(interval - elapsed);
            }
        }
    }

    /// Signal shutdown
    #[inline]
    pub fn shutdown(&self) {
        self.shutdown_flag.store(true, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heartbeat_monitor() {
        let config = HeartbeatConfig::default();
        let monitor = HeartbeatMonitor::new(config);

        assert_eq!(monitor.get_status(), HeartbeatStatus::Healthy);

        // Normal heartbeat
        let status = monitor.beat(1_000_000); // 1ms
        assert_eq!(status, HeartbeatStatus::Healthy);

        // High latency heartbeat
        let status = monitor.beat(15_000_000); // 15ms (exceeds 10ms threshold)
        assert_eq!(status, HeartbeatStatus::Critical);
        assert!(monitor.is_defense_triggered());
    }

    #[test]
    fn test_statistics() {
        let monitor = HeartbeatMonitor::new(HeartbeatConfig::default());

        for _ in 0..10 {
            monitor.beat(2_000_000); // 2ms
        }

        let stats = monitor.get_stats();
        assert_eq!(stats.total_heartbeats, 10);
        assert!(stats.avg_latency_ns > 0);
    }

    #[test]
    fn test_network_lag_detection() {
        let monitor = HeartbeatMonitor::new(HeartbeatConfig::default());

        let status = monitor.record_network_rtt(20_000_000); // 20ms RTT
        assert_eq!(status, HeartbeatStatus::NetworkLag);
    }
}
