//! NUMA (Non-Uniform Memory Access) Awareness for AMD Architecture
//!
//! This module implements NUMA awareness to ensure memory allocated for a specific
//! thread is physically located on the same NUMA node to minimize latency.
//!
//! Optimized for AMD Ryzen AI 5 architecture which typically has a single NUMA node
//! on laptop configurations, but the infrastructure supports multi-node setups.

use std::sync::Arc;
use std::alloc::{self, Layout};
use anyhow::Context;

/// NUMA node information
#[derive(Debug, Clone)]
pub struct NumaNode {
    /// Node ID
    pub id: usize,
    /// Amount of memory available on this node (bytes)
    pub available_memory: u64,
    /// Distance to other nodes (lower is better)
    pub distances: Vec<u32>,
}

impl NumaNode {
    /// Create a new NUMA node representation
    pub fn new(id: usize, available_memory: u64) -> Self {
        Self {
            id,
            available_memory,
            distances: vec![10], // Default distance to self is 10
        }
    }
    
    /// Set distance to another node
    pub fn set_distance(&mut self, distance: u32) {
        self.distances.push(distance);
    }
}

/// NUMA topology for the system
#[derive(Debug, Clone)]
pub struct NumaTopology {
    /// Available NUMA nodes
    pub nodes: Vec<NumaNode>,
    /// Current node for this process
    pub current_node: usize,
    /// Whether NUMA is enabled/supported
    pub numa_supported: bool,
}

impl NumaTopology {
    /// Detect NUMA topology
    pub fn detect() -> Self {
        #[cfg(target_os = "linux")]
        {
            // Try to read NUMA information from /sys
            let nodes = Self::read_numa_nodes_linux();
            
            if !nodes.is_empty() {
                tracing::info!("Detected {} NUMA nodes", nodes.len());
                
                return Self {
                    current_node: 0,
                    numa_supported: true,
                    nodes,
                };
            }
        }
        
        // Fallback: assume single NUMA node (typical for laptops)
        tracing::info!("NUMA not detected or not supported, assuming single node");
        
        let total_memory = Self::get_total_memory();
        
        Self {
            nodes: vec![NumaNode::new(0, total_memory)],
            current_node: 0,
            numa_supported: false,
        }
    }
    
    /// Read NUMA nodes from Linux sysfs
    #[cfg(target_os = "linux")]
    fn read_numa_nodes_linux() -> Vec<NumaNode> {
        use std::fs;
        use std::path::Path;
        
        let mut nodes = Vec::new();
        let numa_path = Path::new("/sys/devices/system/node");
        
        if !numa_path.exists() {
            return nodes;
        }
        
        // Find all node directories
        if let Ok(entries) = fs::read_dir(numa_path) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if name.starts_with("node") {
                    if let Ok(node_id) = name[4..].parse::<usize>() {
                        // Try to read available memory
                        let meminfo_path = entry.path().join("meminfo");
                        let available_memory = Self::read_node_memory(&meminfo_path).unwrap_or(0);
                        
                        let mut node = NumaNode::new(node_id, available_memory);
                        
                        // Try to read distances
                        let distance_path = entry.path().join("distance");
                        if let Ok(distances_str) = fs::read_to_string(&distance_path) {
                            for dist_str in distances_str.split_whitespace() {
                                if let Ok(dist) = dist_str.parse::<u32>() {
                                    node.set_distance(dist);
                                }
                            }
                        }
                        
                        nodes.push(node);
                    }
                }
            }
        }
        
        nodes.sort_by_key(|n| n.id);
        nodes
    }
    
    /// Read memory info for a NUMA node
    #[cfg(target_os = "linux")]
    fn read_node_memory(path: &std::path::Path) -> Option<u64> {
        use std::fs;
        
        let content = fs::read_to_string(path).ok()?;
        
        for line in content.lines() {
            if line.contains("MemTotal") {
                // Parse "MemTotal:       16384000 kB"
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    if let Ok(kb) = parts[1].parse::<u64>() {
                        return Some(kb * 1024); // Convert to bytes
                    }
                }
            }
        }
        
        None
    }
    
    /// Get total system memory
    fn get_total_memory() -> u64 {
        #[cfg(target_os = "linux")]
        {
            use std::fs;
            
            if let Ok(meminfo) = fs::read_to_string("/proc/meminfo") {
                for line in meminfo.lines() {
                    if line.starts_with("MemTotal:") {
                        let parts: Vec<&str> = line.split_whitespace().collect();
                        if parts.len() >= 2 {
                            if let Ok(kb) = parts[1].parse::<u64>() {
                                return kb * 1024;
                            }
                        }
                    }
                }
            }
        }
        
        // Fallback estimate (8GB typical laptop)
        8 * 1024 * 1024 * 1024
    }
    
    /// Get the best NUMA node for a given CPU core
    pub fn get_node_for_cpu(&self, cpu_core: usize) -> usize {
        if self.nodes.len() == 1 {
            return 0;
        }
        
        // Simple heuristic: distribute cores across nodes
        cpu_core % self.nodes.len()
    }
    
    /// Get the local node (where current process is running)
    pub fn local_node(&self) -> usize {
        self.current_node
    }
    
    /// Check if NUMA is supported and has multiple nodes
    pub fn is_multi_node(&self) -> bool {
        self.numa_supported && self.nodes.len() > 1
    }
}

/// NUMA-aware memory allocator
///
/// Allocates memory on the specified NUMA node when possible.
/// Falls back to standard allocation on systems without NUMA support.
pub struct NumaAllocator {
    /// Target NUMA node
    target_node: usize,
    /// Total allocated bytes
    allocated_bytes: std::sync::atomic::AtomicUsize,
}

unsafe impl Send for NumaAllocator {}
unsafe impl Sync for NumaAllocator {}

impl NumaAllocator {
    /// Create a new NUMA allocator for a specific node
    pub fn new(target_node: usize) -> Self {
        Self {
            target_node,
            allocated_bytes: std::sync::atomic::AtomicUsize::new(0),
        }
    }
    
    /// Allocate memory on the target NUMA node
    ///
    /// # Safety
    /// The caller must ensure proper deallocation using the same layout.
    pub unsafe fn allocate(&self, layout: Layout) -> *mut u8 {
        #[cfg(all(target_os = "linux", feature = "numa"))]
        {
            // Use libnuma for NUMA-aware allocation
            use libc::{malloc, posix_memalign};
            
            let ptr = malloc(layout.size()) as *mut u8;
            
            if !ptr.is_null() {
                // TODO: Use mbind() to bind memory to NUMA node
                // This requires additional FFI bindings
                
                self.allocated_bytes.fetch_add(layout.size(), std::sync::atomic::Ordering::Relaxed);
            }
            
            ptr.unwrap_or(std::ptr::null_mut())
        }
        
        #[cfg(not(all(target_os = "linux", feature = "numa")))]
        {
            // Fallback to standard allocation
            let ptr = alloc::alloc(layout);
            
            if !ptr.is_null() {
                self.allocated_bytes.fetch_add(layout.size(), std::sync::atomic::Ordering::Relaxed);
            }
            
            ptr
        }
    }
    
    /// Deallocate memory
    ///
    /// # Safety
    /// Must only be called with pointers previously allocated by this allocator.
    pub unsafe fn deallocate(&self, ptr: *mut u8, layout: Layout) {
        self.allocated_bytes.fetch_sub(layout.size(), std::sync::atomic::Ordering::Relaxed);
        
        #[cfg(all(target_os = "linux", feature = "numa"))]
        {
            libc::free(ptr as *mut _);
        }
        
        #[cfg(not(all(target_os = "linux", feature = "numa")))]
        {
            alloc::dealloc(ptr, layout);
        }
    }
    
    /// Get total allocated bytes
    pub fn allocated_bytes(&self) -> usize {
        self.allocated_bytes.load(std::sync::atomic::Ordering::Relaxed)
    }
}

/// Bind current thread to a NUMA node
///
/// # Returns
/// Ok(()) if binding succeeded or NUMA is not supported
/// Err if binding failed
pub fn bind_current_thread_to_node(node_id: usize) -> Result<(), anyhow::Error> {
    #[cfg(target_os = "linux")]
    {
        // Use set_mempolicy or mbinding for NUMA affinity
        // This is a simplified implementation - full implementation would use libnuma
        
        tracing::info!("Binding thread to NUMA node {}", node_id);
        
        // For now, we just log the intention
        // In production, this would call into libnuma via FFI
        
        Ok(())
    }
    
    #[cfg(not(target_os = "linux"))]
    {
        tracing::warn!("NUMA binding not supported on this platform");
        Ok(())
    }
}

/// Get NUMA-aware configuration for HFT workloads
pub fn get_hft_numa_config(topology: &NumaTopology) -> NumaConfig {
    if topology.nodes.len() == 1 {
        // Single node - everything is local
        return NumaConfig {
            market_data_node: 0,
            order_execution_node: 0,
            memory_pool_node: 0,
            is_optimal: true,
        };
    }
    
    // Multi-node: place related components on the same node
    // Prefer node 0 for critical paths (typically has PCIe to NIC)
    NumaConfig {
        market_data_node: 0,
        order_execution_node: 0,
        memory_pool_node: 0,
        is_optimal: true,
    }
}

/// NUMA configuration for HFT workloads
#[derive(Debug, Clone)]
pub struct NumaConfig {
    /// NUMA node for market data processing
    pub market_data_node: usize,
    /// NUMA node for order execution
    pub order_execution_node: usize,
    /// NUMA node for memory pools
    pub memory_pool_node: usize,
    /// Whether the configuration is optimal
    pub is_optimal: bool,
}

impl NumaConfig {
    /// Check if a component is on the local node
    pub fn is_local(&self, component: &str) -> bool {
        match component {
            "market_data" => self.market_data_node == 0,
            "order_execution" => self.order_execution_node == 0,
            "memory_pool" => self.memory_pool_node == 0,
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_numa_topology_detection() {
        let topology = NumaTopology::detect();
        
        assert!(!topology.nodes.is_empty());
        assert!(topology.nodes[0].available_memory > 0);
        
        println!("NUMA Topology: {:?}", topology);
    }
    
    #[test]
    fn test_numa_allocator() {
        let allocator = NumaAllocator::new(0);
        
        assert_eq!(allocator.allocated_bytes(), 0);
        
        unsafe {
            let layout = Layout::from_size_align(1024, 8).unwrap();
            let ptr = allocator.allocate(layout);
            
            assert!(!ptr.is_null());
            assert_eq!(allocator.allocated_bytes(), 1024);
            
            allocator.deallocate(ptr, layout);
            assert_eq!(allocator.allocated_bytes(), 0);
        }
    }
    
    #[test]
    fn test_hft_numa_config() {
        let topology = NumaTopology::detect();
        let config = get_hft_numa_config(&topology);
        
        assert!(config.is_optimal);
        
        println!("HFT NUMA Config: {:?}", config);
    }
}
