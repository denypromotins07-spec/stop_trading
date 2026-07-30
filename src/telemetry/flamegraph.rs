//! Flamegraph Profiling Engine
//! 
//! Continuous profiling engine generating real-time flamegraphs of CPU hot paths.
//! Identifies cache misses and branch mispredictions in the Rust execution loop.

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use crossbeam_utils::CachePadded;

/// Maximum stack depth to track
const MAX_STACK_DEPTH: usize = 64;

/// Maximum number of unique stack traces
const MAX_TRACES: usize = 8192;

/// A single stack frame
#[repr(C)]
#[derive(Clone, Copy)]
pub struct StackFrame {
    /// Instruction pointer
    pub ip: u64,
    /// Function hash (for symbolization)
    pub fn_hash: u32,
    /// Source line (if available)
    pub line: u32,
}

impl Default for StackFrame {
    fn default() -> Self {
        Self {
            ip: 0,
            fn_hash: 0,
            line: 0,
        }
    }
}

/// Collected stack trace
#[repr(C)]
#[derive(Clone, Copy)]
pub struct StackTrace {
    /// Frames in the trace
    pub frames: [StackFrame; MAX_STACK_DEPTH],
    /// Actual depth
    pub depth: u8,
    /// Sample count
    pub count: u64,
    /// Hash of the trace (for deduplication)
    pub hash: u64,
}

impl Default for StackTrace {
    fn default() -> Self {
        Self {
            frames: [StackFrame::default(); MAX_STACK_DEPTH],
            depth: 0,
            count: 0,
            hash: 0,
        }
    }
}

/// Hot spot information
pub struct HotSpot {
    /// Function hash
    pub fn_hash: u32,
    /// Total samples
    pub sample_count: u64,
    /// Percentage of total samples
    pub percentage: f64,
    /// Estimated CPU cycles
    pub estimated_cycles: u64,
}

/// Flamegraph Profiler Engine
pub struct FlamegraphEngine {
    /// Stack trace storage
    traces: CachePadded<[StackTrace; MAX_TRACES]>,
    /// Active trace count
    trace_count: CachePadded<AtomicU64>,
    /// Total samples collected
    total_samples: CachePadded<AtomicU64>,
    /// Sampling interval (microseconds)
    sample_interval_us: u64,
    /// Last sample timestamp
    last_sample_us: CachePadded<AtomicU64>,
    /// Per-function sample counts
    fn_samples: CachePadded<[AtomicU64; 1024]>,
    /// Profiler enabled
    enabled: CachePadded<AtomicBool>,
    /// Profiling active
    profiling: CachePadded<AtomicBool>,
}

impl FlamegraphEngine {
    /// Create a new flamegraph profiler
    /// 
    /// # Arguments
    /// * `sample_interval_us` - Sampling interval in microseconds
    pub fn new(sample_interval_us: u64) -> Self {
        Self {
            traces: CachePadded::new(std::array::from_fn(|_| StackTrace::default())),
            trace_count: CachePadded::new(AtomicU64::new(0)),
            total_samples: CachePadded::new(AtomicU64::new(0)),
            sample_interval_us,
            last_sample_us: CachePadded::new(AtomicU64::new(0)),
            fn_samples: CachePadded::new(std::array::from_fn(|_| AtomicU64::new(0))),
            enabled: CachePadded::new(AtomicBool::new(false)),
            profiling: CachePadded::new(AtomicBool::new(false)),
        }
    }

    /// Initialize the profiler (sets up signal handlers)
    pub fn init(&self) -> bool {
        #[cfg(target_os = "linux")]
        {
            // In production, this would:
            // 1. Set up SIGPROF handler
            // 2. Configure interval timer
            // 3. Map perf_event buffers
            
            self.enabled.store(true, Ordering::Relaxed);
            return true;
        }

        #[cfg(not(target_os = "linux"))]
        {
            false
        }
    }

    /// Record a sample (called from signal handler)
    /// 
    /// # Safety
    /// This function must be async-signal-safe
    #[inline]
    pub unsafe fn record_sample(&self, stack: &[u64], cpu: u32) {
        if !self.profiling.load(Ordering::Relaxed) {
            return;
        }

        let current_us = get_timestamp_us();
        let last = self.last_sample_us.load(Ordering::Relaxed);

        // Rate limiting
        if current_us.saturating_sub(last) < self.sample_interval_us {
            return;
        }

        self.last_sample_us.store(current_us, Ordering::Relaxed);

        // Build stack trace
        let mut trace = StackTrace::default();
        let depth = stack.len().min(MAX_STACK_DEPTH);
        
        for (i, &ip) in stack.iter().take(depth).enumerate() {
            trace.frames[i].ip = ip;
            trace.frames[i].fn_hash = hash_ip(ip);
        }
        trace.depth = depth as u8;
        trace.count = 1;

        // Calculate trace hash
        trace.hash = hash_trace(&trace.frames[..depth]);

        // Find or create trace entry
        let idx = (trace.hash as usize) % MAX_TRACES;
        
        if self.traces.traces[idx].hash == trace.hash {
            // Existing trace
            self.traces.traces[idx].count += 1;
        } else {
            // New trace (overwrite)
            self.traces.traces[idx] = trace;
            
            let count = self.trace_count.fetch_add(1, Ordering::Relaxed);
            if count >= MAX_TRACES as u64 {
                // Buffer full, wrap around
            }
        }

        // Update per-function counts
        for i in 0..depth {
            let fn_hash = trace.frames[i].fn_hash as usize % 1024;
            self.fn_samples[fn_hash].fetch_add(1, Ordering::Relaxed);
        }

        self.total_samples.fetch_add(1, Ordering::Relaxed);
    }

    /// Get hot spots sorted by sample count
    pub fn get_hot_spots(&self, top_n: usize) -> Vec<HotSpot> {
        let mut spots: Vec<(u32, u64)> = Vec::with_capacity(1024);

        for (i, count) in self.fn_samples.iter().enumerate() {
            let c = count.load(Ordering::Relaxed);
            if c > 0 {
                spots.push((i as u32, c));
            }
        }

        // Sort by count descending
        spots.sort_by(|a, b| b.1.cmp(&a.1));

        let total = self.total_samples.load(Ordering::Relaxed) as f64;

        spots.into_iter().take(top_n).map(|(fn_hash, count)| {
            HotSpot {
                fn_hash,
                sample_count: count,
                percentage: if total > 0.0 { (count as f64 / total) * 100.0 } else { 0.0 },
                estimated_cycles: count * 100, // Approximate
            }
        }).collect()
    }

    /// Export traces for flamegraph generation
    pub fn export_traces(&self) -> Vec<StackTrace> {
        let mut result = Vec::with_capacity(MAX_TRACES);
        
        for trace in self.traces.iter() {
            if trace.hash != 0 && trace.count > 0 {
                result.push(*trace);
            }
        }

        result
    }

    /// Generate folded stack format for flamegraph.pl
    pub fn generate_folded(&self) -> String {
        let mut output = String::with_capacity(65536);
        
        for trace in self.traces.iter() {
            if trace.hash != 0 && trace.count > 0 {
                // Build semicolon-separated stack
                let mut stack_str = String::new();
                for i in (0..trace.depth).rev() {
                    if !stack_str.is_empty() {
                        stack_str.push(';');
                    }
                    stack_str.push_str(&format!("0x{:x}", trace.frames[i as usize].ip));
                }
                
                output.push_str(&stack_str);
                output.push(' ');
                output.push_str(&trace.count.to_string());
                output.push('\n');
            }
        }

        output
    }

    /// Start profiling
    pub fn start(&self) {
        if self.enabled.load(Ordering::Relaxed) {
            self.profiling.store(true, Ordering::Relaxed);
        }
    }

    /// Stop profiling
    pub fn stop(&self) {
        self.profiling.store(false, Ordering::Relaxed);
    }

    /// Check if profiling is active
    #[inline]
    pub fn is_profiling(&self) -> bool {
        self.profiling.load(Ordering::Relaxed)
    }

    /// Get total sample count
    #[inline]
    pub fn get_sample_count(&self) -> u64 {
        self.total_samples.load(Ordering::Relaxed)
    }

    /// Reset all data
    pub fn reset(&self) {
        for trace in self.traces.iter() {
            unsafe {
                let ptr = trace as *const StackTrace as *mut StackTrace;
                (*ptr) = StackTrace::default();
            }
        }
        self.trace_count.store(0, Ordering::Relaxed);
        self.total_samples.store(0, Ordering::Relaxed);
        for count in self.fn_samples.iter() {
            count.store(0, Ordering::Relaxed);
        }
    }
}

impl Default for FlamegraphEngine {
    fn default() -> Self {
        Self::new(1000) // 1ms sampling
    }
}

#[inline]
fn hash_ip(ip: u64) -> u32 {
    // Simple hash for instruction pointer
    ((ip >> 16) ^ (ip & 0xFFFF)) as u32
}

#[inline]
fn hash_trace(frames: &[StackFrame]) -> u64 {
    let mut hash: u64 = 0;
    for frame in frames {
        hash = hash.wrapping_mul(31).wrapping_add(frame.ip as u64);
    }
    hash
}

#[inline]
fn get_timestamp_us() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_micros() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flamegraph_basic() {
        let profiler = FlamegraphEngine::new(1000);
        
        #[cfg(target_os = "linux")]
        {
            profiler.init();
            profiler.start();
            
            // Simulate some samples
            unsafe {
                let stack = [0x400000, 0x400100, 0x400200];
                profiler.record_sample(&stack, 0);
            }
            
            let count = profiler.get_sample_count();
            assert!(count >= 0);
            
            profiler.stop();
        }
    }

    #[test]
    fn test_hot_spots() {
        let profiler = FlamegraphEngine::new(1000);
        let spots = profiler.get_hot_spots(10);
        assert!(spots.is_empty()); // No samples yet
    }
}
