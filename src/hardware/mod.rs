//! Hardware Module Root
//!
//! This module provides hardware-aware concurrency primitives:
//! - CPU Pinning: Bind critical threads to specific physical cores
//! - NUMA Awareness: Ensure memory locality for minimal latency
//! - Auto-detection: Automatically apply optimal settings for AMD Ryzen AI 5
//!
//! The startup routine detects the hardware topology and applies
//! the optimal pinning and NUMA strategy automatically.

pub mod cpu_pin;
pub mod numa;

use std::sync::Arc;
use anyhow::Context;

pub use cpu_pin::{
    pin_current_thread_to_core, spawn_pinned, CpuTopology, CoreAssignment,
    num_cpus, num_physical_cores, is_valid_core, get_current_cpu,
};
pub use numa::{
    NumaTopology, NumaNode, NumaAllocator, NumaConfig,
    bind_current_thread_to_node, get_hft_numa_config,
};

/// Hardware configuration for the HFT system
#[derive(Debug, Clone)]
pub struct HardwareConfig {
    /// CPU topology information
    pub cpu_topology: CpuTopology,
    /// NUMA topology information
    pub numa_topology: NumaTopology,
    /// Core assignments for different workloads
    pub core_assignment: CoreAssignment,
    /// NUMA configuration for workloads
    pub numa_config: NumaConfig,
    /// Whether the configuration is optimal for HFT
    pub is_optimal: bool,
}

impl HardwareConfig {
    /// Detect and create optimal hardware configuration
    pub fn detect() -> Self {
        tracing::info!("Detecting hardware topology...");
        
        let cpu_topology = CpuTopology::detect();
        let numa_topology = NumaTopology::detect();
        
        tracing::info!(
            "CPU: {} logical cores, {} physical cores",
            cpu_topology.logical_cpus,
            cpu_topology.physical_cores
        );
        tracing::info!(
            "NUMA: {} node(s), {} MB total memory",
            numa_topology.nodes.len(),
            numa_topology.nodes.iter().map(|n| n.available_memory).sum::<u64>() / (1024 * 1024)
        );
        
        let core_assignment = cpu_topology.get_hft_core_assignment();
        let numa_config = get_hft_numa_config(&numa_topology);
        
        // Determine if configuration is optimal
        let is_optimal = 
            cpu_topology.logical_cpus >= 6 &&  // At least 6 cores recommended
            cpu_topology.physical_cores >= 4 && // At least 4 physical cores
            numa_topology.nodes.len() == 1;     // Single NUMA node is fine for laptops
        
        if !is_optimal {
            tracing::warn!("Hardware configuration may not be optimal for HFT workloads");
        } else {
            tracing::info!("Hardware configuration is optimal for HFT");
        }
        
        Self {
            cpu_topology,
            numa_topology,
            core_assignment,
            numa_config,
            is_optimal,
        }
    }
    
    /// Get a summary of the hardware configuration
    pub fn summary(&self) -> String {
        format!(
            "Hardware Config:\n\
             - CPU: {}L / {}P cores\n\
             - NUMA Nodes: {}\n\
             - Memory: {:.1} GB\n\
             - Market Data Core: {}\n\
             - Order Execution Core: {}\n\
             - Risk Management Core: {}\n\
             - Optimal: {}",
            self.cpu_topology.logical_cpus,
            self.cpu_topology.physical_cores,
            self.numa_topology.nodes.len(),
            self.numa_topology.nodes.iter().map(|n| n.available_memory).sum::<u64>() as f64 / (1024.0 * 1024.0 * 1024.0),
            self.core_assignment.market_data,
            self.core_assignment.order_execution,
            self.core_assignment.risk_management,
            self.is_optimal
        )
    }
}

/// Initialize hardware-aware runtime
///
/// This function should be called at application startup to:
/// 1. Detect hardware topology
/// 2. Apply CPU pinning strategy
/// 3. Configure NUMA awareness
/// 4. Log configuration for observability
pub fn init_hardware_runtime() -> Result<HardwareConfig, anyhow::Error> {
    tracing::info!("Initializing hardware-aware runtime...");
    
    let config = HardwareConfig::detect();
    
    tracing::info!("{}", config.summary());
    
    // Validate core assignments
    validate_core_assignments(&config)?;
    
    Ok(config)
}

/// Validate that core assignments are valid for the current system
fn validate_core_assignments(config: &HardwareConfig) -> Result<(), anyhow::Error> {
    let max_core = config.cpu_topology.logical_cpus;
    
    if config.core_assignment.market_data >= max_core {
        return Err(anyhow::anyhow!(
            "Market data core {} is invalid (max: {})",
            config.core_assignment.market_data,
            max_core - 1
        ));
    }
    
    if config.core_assignment.order_execution >= max_core {
        return Err(anyhow::anyhow!(
            "Order execution core {} is invalid (max: {})",
            config.core_assignment.order_execution,
            max_core - 1
        ));
    }
    
    if config.core_assignment.risk_management >= max_core {
        return Err(anyhow::anyhow!(
            "Risk management core {} is invalid (max: {})",
            config.core_assignment.risk_management,
            max_core - 1
        ));
    }
    
    // Check for conflicts (same core assigned to multiple critical tasks)
    let mut cores_used = std::collections::HashSet::new();
    
    if !cores_used.insert(config.core_assignment.market_data) {
        return Err(anyhow::anyhow!("Market data core conflict"));
    }
    if !cores_used.insert(config.core_assignment.order_execution) {
        return Err(anyhow::anyhow!("Order execution core conflict"));
    }
    if !cores_used.insert(config.core_assignment.risk_management) {
        return Err(anyhow::anyhow!("Risk management core conflict"));
    }
    
    tracing::info!("Core assignments validated successfully");
    
    Ok(())
}

/// Apply hardware optimizations for the main trading thread
pub fn apply_main_thread_optimizations(config: &HardwareConfig) -> Result<(), anyhow::Error> {
    // Pin main thread to order execution core
    pin_current_thread_to_core(config.core_assignment.order_execution)?;
    
    // Bind to appropriate NUMA node
    bind_current_thread_to_node(config.numa_config.order_execution_node)?;
    
    tracing::info!(
        "Main thread pinned to core {}, NUMA node {}",
        config.core_assignment.order_execution,
        config.numa_config.order_execution_node
    );
    
    Ok(())
}

/// Get recommended stack size for high-performance threads
pub fn recommended_stack_size() -> usize {
    // 2MB stack size for trading threads
    2 * 1024 * 1024
}

/// Get recommended thread pool sizes based on hardware
pub fn get_recommended_pool_sizes(config: &HardwareConfig) -> ThreadPoolSizes {
    ThreadPoolSizes {
        market_data_threads: config.core_assignment.background.len().max(2),
        order_execution_threads: 2, // Dedicated threads for critical path
        background_threads: config.cpu_topology.physical_cores / 2,
    }
}

/// Thread pool size recommendations
#[derive(Debug, Clone)]
pub struct ThreadPoolSizes {
    pub market_data_threads: usize,
    pub order_execution_threads: usize,
    pub background_threads: usize,
}

impl ThreadPoolSizes {
    pub fn format(&self) -> String {
        format!(
            "Thread Pools | Market Data: {} | Order Execution: {} | Background: {}",
            self.market_data_threads,
            self.order_execution_threads,
            self.background_threads
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_hardware_config_detection() {
        let config = HardwareConfig::detect();
        
        assert!(config.cpu_topology.logical_cpus > 0);
        assert!(!config.numa_topology.nodes.is_empty());
        
        println!("{}", config.summary());
    }
    
    #[test]
    fn test_validate_core_assignments() {
        let config = HardwareConfig::detect();
        
        // Should pass for valid configuration
        let result = validate_core_assignments(&config);
        
        // May fail if system has very few cores
        if config.cpu_topology.logical_cpus < 4 {
            assert!(result.is_err());
        } else {
            assert!(result.is_ok());
        }
    }
    
    #[test]
    fn test_recommended_pool_sizes() {
        let config = HardwareConfig::detect();
        let sizes = get_recommended_pool_sizes(&config);
        
        assert!(sizes.market_data_threads >= 2);
        assert!(sizes.order_execution_threads >= 1);
        assert!(sizes.background_threads >= 1);
        
        println!("{}", sizes.format());
    }
}
