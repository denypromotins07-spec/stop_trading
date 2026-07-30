//! CPU Core Pinning for Critical Threads
//!
//! This module implements CPU core pinning to bind critical threads
//! to specific physical cores, preventing context-switching cache thrashing
//! for the matching engine simulator and WebSocket parsers.
//!
//! Optimized for AMD Ryzen AI 5 architecture.

use std::thread::{self, JoinHandle};
use anyhow::Context;

/// Get the number of logical CPUs available
pub fn num_cpus() -> usize {
    num_cpus::get()
}

/// Get the number of physical cores (approximate)
pub fn num_physical_cores() -> usize {
    // On modern systems with hyperthreading/SMT, this is typically num_cpus / 2
    // For AMD Ryzen AI 5, we can query this more accurately if needed
    num_cpus::get() / 2
}

/// CPU topology information for AMD Ryzen AI 5
#[derive(Debug, Clone)]
pub struct CpuTopology {
    /// Total logical CPUs
    pub logical_cpus: usize,
    /// Physical cores
    pub physical_cores: usize,
    /// NUMA nodes detected
    pub numa_nodes: usize,
    /// L3 cache groups (CCX on AMD)
    pub l3_cache_groups: usize,
}

impl CpuTopology {
    /// Detect CPU topology
    pub fn detect() -> Self {
        let logical = num_cpus::get();
        let physical = logical / 2;
        
        // For AMD Ryzen AI 5 (typically 6 cores / 12 threads)
        // We assume a basic topology - in production this could be queried from /proc/cpuinfo
        let numa_nodes = 1; // Most laptops have single NUMA node
        let l3_cache_groups = physical / 3; // AMD CCX typically has 3-4 cores
        
        Self {
            logical_cpus: logical,
            physical_cores: physical,
            numa_nodes,
            l3_cache_groups: l3_cache_groups.max(1),
        }
    }
    
    /// Get recommended core assignments for HFT workloads
    pub fn get_hft_core_assignment(&self) -> CoreAssignment {
        // Reserve cores 0-1 for system/OS
        // Use cores 2-(n-1) for trading workloads
        
        let system_reserved = 2.min(self.logical_cpus);
        let trading_cores: Vec<usize> = (system_reserved..self.logical_cpus).collect();
        
        CoreAssignment {
            market_data: trading_cores.get(0).copied().unwrap_or(system_reserved),
            order_execution: trading_cores.get(1).copied().unwrap_or(system_reserved + 1),
            risk_management: trading_cores.get(2).copied().unwrap_or(system_reserved + 2),
            telemetry: trading_cores.last().copied().unwrap_or(system_reserved),
            background: trading_cores.get(3..).unwrap_or(&[]).to_vec(),
        }
    }
}

/// Core assignment for different workload types
#[derive(Debug, Clone)]
pub struct CoreAssignment {
    /// Core for market data ingestion (WebSocket parsing)
    pub market_data: usize,
    /// Core for order execution (matching engine)
    pub order_execution: usize,
    /// Core for risk management checks
    pub risk_management: usize,
    /// Core for telemetry and logging
    pub telemetry: usize,
    /// Cores for background tasks
    pub background: Vec<usize>,
}

/// Pin current thread to a specific CPU core
///
/// # Safety
/// This uses platform-specific APIs and should only be called
/// from within the thread that needs to be pinned.
pub fn pin_current_thread_to_core(core_id: usize) -> Result<(), anyhow::Error> {
    #[cfg(target_os = "linux")]
    {
        use libc::{cpu_set_t, sched_setaffinity, pid_t};
        use std::mem;
        
        // Create CPU set
        let mut cpuset: cpu_set_t = unsafe { mem::zeroed() };
        
        // Set the target CPU
        unsafe {
            libc::CPU_SET(core_id, &mut cpuset);
        }
        
        // Get current thread ID
        let tid = unsafe { libc::syscall(libc::SYS_gettid) as pid_t };
        
        // Set affinity
        let result = unsafe {
            sched_setaffinity(tid, mem::size_of::<cpu_set_t>(), &cpuset)
        };
        
        if result == 0 {
            tracing::info!("Thread pinned to CPU core {}", core_id);
            Ok(())
        } else {
            Err(anyhow::anyhow!(
                "Failed to pin thread to core {}: errno {}",
                core_id,
                *libc::__errno_location()
            ))
        }
    }
    
    #[cfg(not(target_os = "linux"))]
    {
        // On non-Linux platforms, we can't directly pin threads
        // This is a no-op but we log a warning
        tracing::warn!("CPU pinning not supported on this platform");
        Ok(())
    }
}

/// Builder for creating pinned threads
pub struct PinnedThreadBuilder<F> {
    name: String,
    core_id: usize,
    func: F,
    stack_size: Option<usize>,
}

impl<F> PinnedThreadBuilder<F>
where
    F: FnOnce() + Send + 'static,
{
    /// Create a new pinned thread builder
    pub fn new(name: impl Into<String>, core_id: usize, func: F) -> Self {
        Self {
            name: name.into(),
            core_id,
            func,
            stack_size: None,
        }
    }
    
    /// Set custom stack size
    pub fn stack_size(mut self, size: usize) -> Self {
        self.stack_size = Some(size);
        self
    }
    
    /// Spawn the pinned thread
    pub fn spawn(self) -> Result<JoinHandle<()>, anyhow::Error> {
        let name = self.name.clone();
        let core_id = self.core_id;
        let mut builder = thread::Builder::new().name(name);
        
        if let Some(size) = self.stack_size {
            builder = builder.stack_size(size);
        }
        
        let func = self.func;
        
        builder
            .spawn(move || {
                // Pin this thread to the specified core
                if let Err(e) = pin_current_thread_to_core(core_id) {
                    tracing::error!("Failed to pin thread to core {}: {}", core_id, e);
                }
                
                // Execute the function
                func();
            })
            .context("Failed to spawn pinned thread")
    }
}

/// Spawn a new thread pinned to a specific core
pub fn spawn_pinned<F>(name: impl Into<String>, core_id: usize, func: F) -> Result<JoinHandle<()>, anyhow::Error>
where
    F: FnOnce() + Send + 'static,
{
    PinnedThreadBuilder::new(name, core_id, func).spawn()
}

/// Helper to run a closure on the current thread with temporary affinity
/// (Note: True temporary affinity requires restoring original affinity)
pub fn with_cpu_affinity<F, R>(core_id: usize, f: F) -> Result<R, anyhow::Error>
where
    F: FnOnce() -> R,
{
    pin_current_thread_to_core(core_id)?;
    let result = f();
    // Note: We don't restore original affinity here - thread will exit anyway
    Ok(result)
}

/// Validate that a core ID is valid
pub fn is_valid_core(core_id: usize) -> bool {
    core_id < num_cpus::get()
}

/// Get the current thread's CPU affinity (which core it's running on)
pub fn get_current_cpu() -> Option<usize> {
    #[cfg(target_os = "linux")]
    {
        use libc::{cpu_set_t, sched_getaffinity, pid_t};
        use std::mem;
        
        let mut cpuset: cpu_set_t = unsafe { mem::zeroed() };
        let tid = unsafe { libc::syscall(libc::SYS_gettid) as pid_t };
        
        let result = unsafe {
            sched_getaffinity(tid, mem::size_of::<cpu_set_t>(), &mut cpuset)
        };
        
        if result == 0 {
            // Find which CPU is set
            for i in 0..num_cpus::get() {
                unsafe {
                    if libc::CPU_ISSET(i, &cpuset) {
                        return Some(i);
                    }
                }
            }
        }
        
        None
    }
    
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cpu_topology_detection() {
        let topology = CpuTopology::detect();
        
        assert!(topology.logical_cpus > 0);
        assert!(topology.physical_cores > 0);
        assert!(topology.logical_cpus >= topology.physical_cores);
        
        println!("Detected CPU Topology: {:?}", topology);
    }
    
    #[test]
    fn test_core_assignment() {
        let topology = CpuTopology::detect();
        let assignment = topology.get_hft_core_assignment();
        
        assert!(is_valid_core(assignment.market_data));
        assert!(is_valid_core(assignment.order_execution));
        assert!(is_valid_core(assignment.risk_management));
        assert!(is_valid_core(assignment.telemetry));
        
        println!("HFT Core Assignment: {:?}", assignment);
    }
    
    #[test]
    fn test_is_valid_core() {
        let num = num_cpus::get();
        
        assert!(is_valid_core(0));
        assert!(!is_valid_core(num + 100));
    }
    
    #[test]
    fn test_spawn_pinned_thread() {
        let handle = spawn_pinned("test-pinned", 0, || {
            println!("Running on pinned thread");
        }).unwrap();
        
        handle.join().unwrap();
    }
}
