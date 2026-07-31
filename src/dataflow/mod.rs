//! Dataflow Module Root
//! 
//! This module provides FPGA-style dataflow pipelines for deterministic,
//! ultra-low-latency processing of market data and trading signals.

pub mod pipeline;
pub mod batching;

// Re-export main types for convenience
pub use pipeline::{
    BatchProcessor, ClosureStage, Edge, EdgeStats, LinearPipeline, Node, NodeId,
    Pipeline, PipelineBuilder, PipelineStage, PipelineStats, ProcessingFn,
    DEFAULT_CHANNEL_CAPACITY,
};

pub use batching::{
    BatchBuffer, BatchPipeline, BatchProcessor as BatchProcessorTrait, BatchSizeController,
    CollectorStats, MicroBatchCollector, Tick,
    CACHE_LINE_SIZE, MAX_BATCH_SIZE, MIN_BATCH_SIZE, TARGET_LATENCY_US,
};

/// Trait for objects that can be part of a dataflow graph
pub trait DataflowNode: Send + Sync {
    /// Get the unique identifier for this node
    fn node_id(&self) -> NodeId;
    
    /// Get the human-readable name
    fn node_name(&self) -> &'static str;
    
    /// Start processing
    fn start(&self);
    
    /// Stop processing
    fn stop(&self);
    
    /// Check if currently running
    fn is_running(&self) -> bool;
}

/// Trait for objects that provide statistics about dataflow
pub trait DataflowStats: Send + Sync {
    /// Get current statistics
    fn get_stats(&self) -> DataflowMetrics;
}

#[derive(Debug, Clone)]
pub struct DataflowMetrics {
    pub nodes_active: usize,
    pub edges_total: usize,
    pub messages_processed: u64,
    pub messages_dropped: u64,
    pub avg_latency_ns: f64,
    pub p99_latency_ns: f64,
}

impl Default for DataflowMetrics {
    fn default() -> Self {
        Self {
            nodes_active: 0,
            edges_total: 0,
            messages_processed: 0,
            messages_dropped: 0,
            avg_latency_ns: 0.0,
            p99_latency_ns: 0.0,
        }
    }
}

/// Global execution topology manager
pub struct ExecutionTopology {
    /// Maximum memory budget in bytes (6.5GB limit)
    max_memory_bytes: usize,
    /// Current estimated memory usage
    current_usage: std::sync::atomic::AtomicUsize,
}

impl ExecutionTopology {
    pub const MEMORY_LIMIT_6_5GB: usize = 6_500_000_000;
    
    pub fn new(max_memory: usize) -> Self {
        Self {
            max_memory_bytes: max_memory.min(Self::MEMORY_LIMIT_6_5GB),
            current_usage: std::sync::atomic::AtomicUsize::new(0),
        }
    }
    
    /// Try to allocate memory, returns false if would exceed limit
    pub fn try_allocate(&self, bytes: usize) -> bool {
        let current = self.current_usage.load(std::sync::atomic::Ordering::Relaxed);
        if current.saturating_add(bytes) > self.max_memory_bytes {
            return false;
        }
        self.current_usage.fetch_add(bytes, std::sync::atomic::Ordering::Relaxed);
        true
    }
    
    /// Release allocated memory
    pub fn release(&self, bytes: usize) {
        self.current_usage.fetch_sub(bytes, std::sync::atomic::Ordering::Relaxed);
    }
    
    /// Get current memory usage
    pub fn current_usage(&self) -> usize {
        self.current_usage.load(std::sync::atomic::Ordering::Relaxed)
    }
    
    /// Get available memory
    pub fn available(&self) -> usize {
        self.max_memory_bytes.saturating_sub(self.current_usage())
    }
    
    /// Get memory limit
    pub fn limit(&self) -> usize {
        self.max_memory_bytes
    }
}

impl Default for ExecutionTopology {
    fn default() -> Self {
        Self::new(Self::MEMORY_LIMIT_6_5GB)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_execution_topology_memory_limit() {
        let topology = ExecutionTopology::new(1000);
        
        assert!(topology.try_allocate(500));
        assert_eq!(topology.current_usage(), 500);
        assert_eq!(topology.available(), 500);
        
        assert!(!topology.try_allocate(600)); // Would exceed
        assert!(topology.try_allocate(500)); // Exactly at limit
        
        topology.release(250);
        assert_eq!(topology.current_usage(), 750);
        assert_eq!(topology.available(), 250);
    }
    
    #[test]
    fn test_default_memory_limit() {
        let topology = ExecutionTopology::default();
        assert_eq!(topology.limit(), ExecutionTopology::MEMORY_LIMIT_6_5GB);
    }
}
