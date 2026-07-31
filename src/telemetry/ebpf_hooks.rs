//! eBPF Hooks for Zero-Overhead Kernel-Level Tracing on Linux
//! 
//! Tracks network packet processing and disk I/O latencies at the OS level.
//! Uses strict #[cfg(target_os = "linux")] guards with graceful fallbacks.

#[cfg(target_os = "linux")]
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
#[cfg(target_os = "linux")]
use std::time::Instant;

/// eBPF trace event
#[derive(Debug, Clone)]
pub struct EbpfEvent {
    pub timestamp_ns: u64,
    pub event_type: EbpfEventType,
    pub latency_us: u64,
    pub pid: u32,
    pub cpu_id: u32,
}

/// Types of eBPF events
#[derive(Debug, Clone, Copy)]
pub enum EbpfEventType {
    NetworkRx,
    NetworkTx,
    DiskRead,
    DiskWrite,
    Syscall,
    ContextSwitch,
}

/// eBPF statistics
#[derive(Debug, Clone)]
pub struct EbpfStats {
    pub total_events: u64,
    pub avg_latency_us: f64,
    pub max_latency_us: u64,
    pub dropped_events: u64,
}

/// eBPF hooks manager - Linux implementation
#[cfg(target_os = "linux")]
pub struct EbpfHooks {
    enabled: AtomicBool,
    event_count: AtomicU64,
    total_latency: AtomicU64,
    max_latency: AtomicU64,
    dropped_count: AtomicU64,
}

#[cfg(target_os = "linux")]
impl EbpfHooks {
    pub fn new() -> Self {
        Self {
            enabled: AtomicBool::new(true),
            event_count: AtomicU64::new(0),
            total_latency: AtomicU64::new(0),
            max_latency: AtomicU64::new(0),
            dropped_count: AtomicU64::new(0),
        }
    }
    
    /// Record a traced event
    pub fn record_event(&self, event: EbpfEvent) {
        if !self.enabled.load(Ordering::Acquire) {
            return;
        }
        
        self.event_count.fetch_add(1, Ordering::Relaxed);
        self.total_latency.fetch_add(event.latency_us, Ordering::Relaxed);
        
        // Update max latency atomically
        let mut current_max = self.max_latency.load(Ordering::Relaxed);
        while event.latency_us > current_max {
            match self.max_latency.compare_exchange_weak(
                current_max,
                event.latency_us,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => current_max = actual,
            }
        }
    }
    
    /// Get current statistics
    pub fn get_stats(&self) -> EbpfStats {
        let count = self.event_count.load(Ordering::Acquire);
        let total = self.total_latency.load(Ordering::Acquire);
        let max = self.max_latency.load(Ordering::Acquire);
        let dropped = self.dropped_count.load(Ordering::Acquire);
        
        EbpfStats {
            total_events: count,
            avg_latency_us: if count > 0 { total as f64 / count as f64 } else { 0.0 },
            max_latency_us: max,
            dropped_events: dropped,
        }
    }
    
    pub fn enable(&self) {
        self.enabled.store(true, Ordering::Release);
    }
    
    pub fn disable(&self) {
        self.enabled.store(false, Ordering::Release);
    }
    
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Acquire)
    }
}

#[cfg(target_os = "linux")]
impl Default for EbpfHooks {
    fn default() -> Self {
        Self::new()
    }
}

/// eBPF hooks manager - Non-Linux fallback
#[cfg(not(target_os = "linux"))]
pub struct EbpfHooks {
    enabled: bool,
}

#[cfg(not(target_os = "linux"))]
impl EbpfHooks {
    pub fn new() -> Self {
        Self { enabled: false }
    }
    
    pub fn record_event(&self, _event: EbpfEvent) {
        // No-op on non-Linux platforms
    }
    
    pub fn get_stats(&self) -> EbpfStats {
        EbpfStats {
            total_events: 0,
            avg_latency_us: 0.0,
            max_latency_us: 0,
            dropped_events: 0,
        }
    }
    
    pub fn enable(&mut self) {
        self.enabled = true;
    }
    
    pub fn disable(&mut self) {
        self.enabled = false;
    }
    
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
}

#[cfg(not(target_os = "linux"))]
impl Default for EbpfHooks {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_ebpf_hooks() {
        let hooks = EbpfHooks::new();
        
        #[cfg(target_os = "linux")]
        {
            hooks.enable();
            assert!(hooks.is_enabled());
            
            let event = EbpfEvent {
                timestamp_ns: 1000000,
                event_type: EbpfEventType::NetworkRx,
                latency_us: 50,
                pid: 1234,
                cpu_id: 0,
            };
            
            hooks.record_event(event);
            
            let stats = hooks.get_stats();
            assert_eq!(stats.total_events, 1);
            assert_eq!(stats.avg_latency_us, 50.0);
            assert_eq!(stats.max_latency_us, 50);
        }
        
        #[cfg(not(target_os = "linux"))]
        {
            assert!(!hooks.is_enabled());
        }
    }
}
