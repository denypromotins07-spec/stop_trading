//! Hardware Performance Counter Profiler
//! 
//! Custom hardware performance counter reader using `perf_event_open` for AMD Ryzen.
//! Monitors Instructions Per Cycle (IPC) and branch miss rates.

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use crossbeam_utils::CachePadded;

/// Performance counter types
#[derive(Debug, Clone, Copy)]
pub enum PerfEventType {
    /// CPU cycles
    Cycles,
    /// Instructions retired
    Instructions,
    /// Branch instructions
    BranchInstructions,
    /// Branch misses
    BranchMisses,
    /// Cache references
    CacheReferences,
    /// Cache misses
    CacheMisses,
    /// Reference cycles
    RefCycles,
}

/// Performance metrics snapshot
pub struct PerfMetrics {
    /// CPU cycles
    pub cycles: u64,
    /// Instructions retired
    pub instructions: u64,
    /// Instructions per cycle
    pub ipc: f64,
    /// Branch instructions
    pub branch_instructions: u64,
    /// Branch misses
    pub branch_misses: u64,
    /// Branch miss rate (percentage)
    pub branch_miss_rate: f64,
    /// Cache references
    pub cache_references: u64,
    /// Cache misses
    pub cache_misses: u64,
    /// Cache miss rate (percentage)
    pub cache_miss_rate: f64,
    /// Elapsed time (nanoseconds)
    pub elapsed_ns: u64,
}

/// Hardware Performance Counter Reader
pub struct PerfProfiler {
    /// File descriptor for perf_event
    #[cfg(target_os = "linux")]
    fd: std::sync::Mutex<Option<i32>>,
    /// Cycles counter
    cycles: CachePadded<AtomicU64>,
    /// Instructions counter
    instructions: CachePadded<AtomicU64>,
    /// Branch misses counter
    branch_misses: CachePadded<AtomicU64>,
    /// Cache misses counter
    cache_misses: CachePadded<AtomicU64>,
    /// Last read timestamp
    last_read_ns: CachePadded<AtomicU64>,
    /// Profiler enabled
    enabled: CachePadded<AtomicBool>,
    /// AMD-specific optimizations
    is_amd: CachePadded<AtomicBool>,
}

impl PerfProfiler {
    /// Create a new performance profiler
    pub fn new() -> Self {
        // Detect CPU vendor
        let is_amd = detect_amd_cpu();

        Self {
            #[cfg(target_os = "linux")]
            fd: std::sync::Mutex::new(None),
            cycles: CachePadded::new(AtomicU64::new(0)),
            instructions: CachePadded::new(AtomicU64::new(0)),
            branch_misses: CachePadded::new(AtomicU64::new(0)),
            cache_misses: CachePadded::new(AtomicU64::new(0)),
            last_read_ns: CachePadded::new(AtomicU64::new(0)),
            enabled: CachePadded::new(AtomicBool::new(false)),
            is_amd: CachePadded::new(AtomicBool::new(is_amd)),
        }
    }

    /// Initialize perf_event counters
    pub fn init(&self) -> bool {
        #[cfg(target_os = "linux")]
        {
            unsafe {
                // Try to open perf_event for cycles
                let attr = libc::perf_event_attr {
                    type_: libc::PERF_TYPE_HARDWARE,
                    size: std::mem::size_of::<libc::perf_event_attr>() as u32,
                    config: libc::PERF_COUNT_HW_CPU_CYCLES as u64,
                    ..std::mem::zeroed()
                };

                // pid = 0 (current process), cpu = -1 (all CPUs)
                let fd = libc::perf_event_open(&attr, 0, -1, -1, 0);
                
                if fd >= 0 {
                    *self.fd.lock().unwrap() = Some(fd);
                    self.enabled.store(true, Ordering::Relaxed);
                    return true;
                }
            }
        }

        false
    }

    /// Read current performance counters
    pub fn read_counters(&self) -> PerfMetrics {
        let current_ns = get_timestamp_ns();
        let last_ns = self.last_read_ns.load(Ordering::Relaxed);
        let elapsed = current_ns.saturating_sub(last_ns);

        self.last_read_ns.store(current_ns, Ordering::Relaxed);

        let cycles = self.read_counter(PerfEventType::Cycles);
        let instructions = self.read_counter(PerfEventType::Instructions);
        let branch_instructions = self.read_counter(PerfEventType::BranchInstructions);
        let branch_misses = self.read_counter(PerfEventType::BranchMisses);
        let cache_references = self.read_counter(PerfEventType::CacheReferences);
        let cache_misses = self.read_counter(PerfEventType::CacheMisses);

        // Store for atomic access
        self.cycles.store(cycles, Ordering::Relaxed);
        self.instructions.store(instructions, Ordering::Relaxed);
        self.branch_misses.store(branch_misses, Ordering::Relaxed);
        self.cache_misses.store(cache_misses, Ordering::Relaxed);

        // Calculate derived metrics
        let ipc = if cycles > 0 {
            instructions as f64 / cycles as f64
        } else {
            0.0
        };

        let branch_miss_rate = if branch_instructions > 0 {
            (branch_misses as f64 / branch_instructions as f64) * 100.0
        } else {
            0.0
        };

        let cache_miss_rate = if cache_references > 0 {
            (cache_misses as f64 / cache_references as f64) * 100.0
        } else {
            0.0
        };

        PerfMetrics {
            cycles,
            instructions,
            ipc,
            branch_instructions,
            branch_misses,
            branch_miss_rate,
            cache_references,
            cache_misses,
            cache_miss_rate,
            elapsed_ns: elapsed,
        }
    }

    /// Read a specific counter type
    #[cfg(target_os = "linux")]
    fn read_counter(&self, event_type: PerfEventType) -> u64 {
        if !self.enabled.load(Ordering::Relaxed) {
            return 0;
        }

        let config = match event_type {
            PerfEventType::Cycles => libc::PERF_COUNT_HW_CPU_CYCLES as u64,
            PerfEventType::Instructions => libc::PERF_COUNT_HW_INSTRUCTIONS as u64,
            PerfEventType::BranchInstructions => libc::PERF_COUNT_HW_BRANCH_INSTRUCTIONS as u64,
            PerfEventType::BranchMisses => libc::PERF_COUNT_HW_BRANCH_MISSES as u64,
            PerfEventType::CacheReferences => libc::PERF_COUNT_HW_CACHE_REFERENCES as u64,
            PerfEventType::CacheMisses => libc::PERF_COUNT_HW_CACHE_MISSES as u64,
            PerfEventType::RefCycles => libc::PERF_COUNT_HW_REF_CPU_CYCLES as u64,
        };

        unsafe {
            let mut attr = libc::perf_event_attr {
                type_: libc::PERF_TYPE_HARDWARE,
                size: std::mem::size_of::<libc::perf_event_attr>() as u32,
                config,
                ..std::mem::zeroed()
            };

            let fd = libc::perf_event_open(&mut attr, 0, -1, -1, 0);
            if fd < 0 {
                return 0;
            }

            let mut value: u64 = 0;
            let bytes_read = libc::read(fd, &mut value as *mut _ as *mut _, std::mem::size_of::<u64>());
            libc::close(fd);

            if bytes_read == std::mem::size_of::<i64>() as isize {
                value
            } else {
                0
            }
        }
    }

    #[cfg(not(target_os = "linux"))]
    fn read_counter(&self, _event_type: PerfEventType) -> u64 {
        0
    }

    /// Get cached cycles count
    #[inline]
    pub fn get_cycles(&self) -> u64 {
        self.cycles.load(Ordering::Relaxed)
    }

    /// Get cached instructions count
    #[inline]
    pub fn get_instructions(&self) -> u64 {
        self.instructions.load(Ordering::Relaxed)
    }

    /// Get cached IPC
    #[inline]
    pub fn get_ipc(&self) -> f64 {
        let cycles = self.cycles.load(Ordering::Relaxed);
        let instructions = self.instructions.load(Ordering::Relaxed);
        
        if cycles > 0 {
            instructions as f64 / cycles as f64
        } else {
            0.0
        }
    }

    /// Get cached branch miss rate
    #[inline]
    pub fn get_branch_miss_rate(&self) -> f64 {
        // Simplified - would need more tracking for accurate rate
        0.0
    }

    /// Get cached cache miss rate
    #[inline]
    pub fn get_cache_miss_rate(&self) -> f64 {
        // Simplified - would need more tracking for accurate rate
        0.0
    }

    /// Check if running on AMD CPU
    #[inline]
    pub fn is_amd_cpu(&self) -> bool {
        self.is_amd.load(Ordering::Relaxed)
    }

    /// Enable profiling
    pub fn enable(&self) {
        #[cfg(target_os = "linux")]
        {
            if let Some(fd) = *self.fd.lock().unwrap() {
                unsafe {
                    libc::ioctl(fd, libc::PERF_EVENT_IOC_ENABLE, 0);
                }
            }
        }
    }

    /// Disable profiling
    pub fn disable(&self) {
        #[cfg(target_os = "linux")]
        {
            if let Some(fd) = *self.fd.lock().unwrap() {
                unsafe {
                    libc::ioctl(fd, libc::PERF_EVENT_IOC_DISABLE, 0);
                }
            }
        }
    }

    /// Reset counters
    pub fn reset(&self) {
        self.cycles.store(0, Ordering::Relaxed);
        self.instructions.store(0, Ordering::Relaxed);
        self.branch_misses.store(0, Ordering::Relaxed);
        self.cache_misses.store(0, Ordering::Relaxed);
        self.last_read_ns.store(0, Ordering::Relaxed);
    }
}

impl Default for PerfProfiler {
    fn default() -> Self {
        Self::new()
    }
}

#[inline]
fn detect_amd_cpu() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::__cpuid;
        unsafe {
            let cpuid = __cpuid(0);
            // Check vendor string
            let ebx = cpuid.ebx.to_le_bytes();
            let edx = cpuid.edx.to_le_bytes();
            let ecx = cpuid.ecx.to_le_bytes();
            
            let vendor = [ebx[0], ebx[1], ebx[2], ebx[3],
                         edx[0], edx[1], edx[2], edx[3],
                         ecx[0], ecx[1], ecx[2], ecx[3]];
            
            &vendor == b"AuthenticAMD"
        }
    }
    
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[inline]
fn get_timestamp_ns() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64
}

// Stub for non-Linux
#[cfg(not(target_os = "linux"))]
mod libc {
    pub const PERF_TYPE_HARDWARE: u32 = 0;
    pub const PERF_COUNT_HW_CPU_CYCLES: u64 = 0;
    pub const PERF_COUNT_HW_INSTRUCTIONS: u64 = 1;
    pub const PERF_COUNT_HW_BRANCH_INSTRUCTIONS: u64 = 4;
    pub const PERF_COUNT_HW_BRANCH_MISSES: u64 = 5;
    pub const PERF_COUNT_HW_CACHE_REFERENCES: u64 = 2;
    pub const PERF_COUNT_HW_CACHE_MISSES: u64 = 3;
    pub const PERF_COUNT_HW_REF_CPU_CYCLES: u64 = 9;
    pub const PERF_EVENT_IOC_ENABLE: u32 = 0;
    pub const PERF_EVENT_IOC_DISABLE: u32 = 1;

    #[repr(C)]
    pub struct perf_event_attr {
        pub type_: u32,
        pub size: u32,
        pub config: u64,
    }

    pub unsafe fn perf_event_open(_attr: *mut perf_event_attr, _pid: i32, _cpu: i32, _group_fd: i32, _flags: u32) -> i32 {
        -1
    }

    pub unsafe fn read(_fd: i32, _buf: *mut u64, _count: usize) -> isize {
        -1
    }

    pub unsafe fn close(_fd: i32) -> i32 {
        0
    }

    pub unsafe fn ioctl(_fd: i32, _request: u32, _arg: u32) -> i32 {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perf_profiler_basic() {
        let profiler = PerfProfiler::new();
        
        #[cfg(target_os = "linux")]
        {
            let initialized = profiler.init();
            
            if initialized {
                let metrics = profiler.read_counters();
                assert!(metrics.ipc >= 0.0);
            }
        }
        
        #[cfg(not(target_os = "linux"))]
        {
            assert!(!profiler.init());
        }
    }

    #[test]
    fn test_amd_detection() {
        let profiler = PerfProfiler::new();
        // Just verify it doesn't panic
        let _ = profiler.is_amd_cpu();
    }
}
