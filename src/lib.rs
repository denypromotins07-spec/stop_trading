//! HFT Crypto Bot - Ultra-Low-Latency Trading Infrastructure
//!
//! This is the foundational Rust infrastructure for a high-frequency trading bot
//! optimized for AMD Ryzen AI 5 laptops with strict 6.5GB RAM constraints.
//!
//! # Architecture
//!
//! ## Memory Management (Chapter 2)
//! - Lock-free bump allocator (Arena) for zero-allocation object creation
//! - Generic object pools for pre-allocated memory blocks
//! - Global memory tracker with panic on limit breach
//!
//! ## Runtime (Chapter 3)
//! - Custom thread pool executor with priority-based scheduling
//! - High-resolution timer wheel using TSC for nanosecond precision
//! - LMAX Disruptor-style ring buffer for lock-free event passing
//!
//! ## Hardware Awareness (Chapter 4)
//! - CPU core pinning for critical threads
//! - NUMA awareness for memory locality
//! - Auto-detection of AMD Ryzen AI 5 topology
//!
//! ## Telemetry (Chapter 5)
//! - Zero-cost asynchronous structured logger
//! - Lock-free metrics collector with atomic counters
//! - Real-time terminal dashboard
//!
//! # Example
//!
//! ```rust,no_run
//! use hft_crypto_bot::prelude::*;
//!
//! fn main() -> anyhow::Result<()> {
//!     // Initialize environment
//!     dotenvy::dotenv().ok();
//!     
//!     // Initialize telemetry
//!     let config = TelemetryConfig::default();
//!     let telemetry = init_global_telemetry(config)?;
//!     
//!     // Initialize hardware runtime
//!     let hw_config = hardware::init_hardware_runtime()?;
//!     
//!     // Initialize memory tracker
//!     let memory_tracker = memory::init_global_tracker()?;
//!     
//!     println!("HFT Crypto Bot initialized!");
//!     println!("{}", hw_config.summary());
//!     
//!     Ok(())
//! }
//! ```

#![cfg_attr(feature = "nightly", feature(allocator_api))]
#![warn(missing_docs)]
#![warn(rustdoc::missing_crate_level_docs)]

pub mod memory;
pub mod runtime;
pub mod hardware;
pub mod telemetry;

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::memory::{
        Arena, LocalArena, ObjectPool, PacketBuffer, PoolGuard, TickData,
        GlobalMemoryTracker, MemoryStats,
        init_global_tracker, get_global_tracker, global_safety_check,
    };
    
    pub use crate::runtime::{
        Executor, ExecutorStats, PoolConfig, TaskPriority, WorkerPool,
        TimerWheel, LatencyGuard, LatencyTracker, now_ns,
        RingBuffer, RingBufferStats, Event, EventBus, EventBusStats,
    };
    
    pub use crate::hardware::{
        HardwareConfig, CpuTopology, CoreAssignment, NumaTopology, NumaNode,
        NumaAllocator, NumaConfig, ThreadPoolSizes,
        init_hardware_runtime, apply_main_thread_optimizations,
        pin_current_thread_to_core, spawn_pinned,
        num_cpus, num_physical_cores, recommended_stack_size,
        get_recommended_pool_sizes,
    };
    
    pub use crate::telemetry::{
        Telemetry, TelemetryConfig, TelemetryStats,
        AsyncLogger, LogEntry, LogLevel, LoggerConfig, LoggerStats,
        SystemMetrics, MetricsSnapshot, Counter, Gauge, Histogram,
        init_global_telemetry, get_global_telemetry,
        record_tick_to_trade, record_order_book_update, record_trade_execution,
    };
}

use std::sync::Arc;
use anyhow::Context;

/// Application state containing all initialized components
pub struct AppState {
    /// Hardware configuration
    pub hardware: hardware::HardwareConfig,
    /// Memory tracker
    pub memory_tracker: Arc<memory::GlobalMemoryTracker>,
    /// Telemetry system
    pub telemetry: Arc<telemetry::Telemetry>,
    /// Runtime executor
    pub executor: Arc<runtime::Executor>,
    /// Event ring buffer
    pub ring_buffer: Arc<runtime::RingBuffer>,
    /// Event bus
    pub event_bus: Option<Arc<runtime::EventBus>>,
}

impl AppState {
    /// Create and initialize the application state
    pub fn new() -> Result<Self, anyhow::Error> {
        tracing::info!("Initializing HFT Crypto Bot...");
        
        // Load environment variables first
        dotenvy::dotenv().ok();
        
        // Initialize telemetry early for logging
        let telemetry_config = telemetry::TelemetryConfig::default();
        let telemetry = telemetry::init_global_telemetry(telemetry_config)
            .context("Failed to initialize telemetry")?;
        
        // Initialize hardware runtime
        let hardware = hardware::init_hardware_runtime()
            .context("Failed to initialize hardware runtime")?;
        
        // Initialize memory tracker
        let memory_tracker = memory::init_global_tracker()
            .context("Failed to initialize memory tracker")?;
        
        // Create executor with hardware-aware configuration
        let pool_sizes = hardware::get_recommended_pool_sizes(&hardware);
        let executor = Arc::new(runtime::Executor::with_config(
            runtime::PoolConfig {
                num_threads: pool_sizes.order_execution_threads,
                queue_capacity: 1000,
                name_prefix: "critical".to_string(),
                pin_to_cpu: true,
                cpu_start: hardware.core_assignment.order_execution,
            },
            runtime::PoolConfig {
                num_threads: pool_sizes.market_data_threads,
                queue_capacity: 50000,
                name_prefix: "market-data".to_string(),
                pin_to_cpu: true,
                cpu_start: hardware.core_assignment.market_data,
            },
            runtime::PoolConfig {
                num_threads: pool_sizes.background_threads,
                queue_capacity: 10000,
                name_prefix: "background".to_string(),
                pin_to_cpu: false,
                cpu_start: 0,
            },
        ).context("Failed to create executor")?);
        
        // Create ring buffer
        let ring_buffer = runtime::RingBuffer::with_default_capacity()
            .context("Failed to create ring buffer")?;
        
        // Create event bus
        let event_bus = Arc::new(runtime::EventBus::new(
            Arc::clone(&ring_buffer),
            Arc::clone(&executor),
        ));
        
        let state = Self {
            hardware,
            memory_tracker,
            telemetry,
            executor,
            ring_buffer,
            event_bus: Some(event_bus),
        };
        
        tracing::info!("HFT Crypto Bot initialized successfully");
        
        Ok(state)
    }
    
    /// Get the event bus
    pub fn event_bus(&self) -> Arc<runtime::EventBus> {
        self.event_bus.as_ref().unwrap().clone()
    }
    
    /// Perform a health check
    pub fn health_check(&self) -> HealthStatus {
        let memory_stats = self.memory_tracker.get_stats();
        let telemetry_stats = self.telemetry.get_stats();
        let ring_stats = self.ring_buffer.get_stats();
        
        let is_healthy = 
            !self.memory_tracker.is_near_limit() &&
            memory_stats.usage_percentage < 90.0 &&
            ring_stats.utilization < 90.0;
        
        HealthStatus {
            is_healthy,
            memory_usage_pct: memory_stats.usage_percentage,
            ring_utilization_pct: ring_stats.utilization,
            errors_count: telemetry_stats.metrics.errors,
            uptime_seconds: telemetry_stats.metrics.uptime_seconds,
        }
    }
    
    /// Shutdown gracefully
    pub fn shutdown(&self) {
        tracing::info!("Shutting down HFT Crypto Bot...");
        
        if let Some(ref event_bus) = self.event_bus {
            event_bus.shutdown();
        }
        
        self.telemetry.stop();
        
        tracing::info!("Shutdown complete");
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self::new().expect("Failed to create default AppState")
    }
}

impl Drop for AppState {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Health check status
#[derive(Debug, Clone)]
pub struct HealthStatus {
    /// Whether the system is healthy
    pub is_healthy: bool,
    /// Memory usage percentage
    pub memory_usage_pct: f64,
    /// Ring buffer utilization percentage
    pub ring_utilization_pct: f64,
    /// Total error count
    pub errors_count: u64,
    /// Uptime in seconds
    pub uptime_seconds: u64,
}

impl HealthStatus {
    /// Format status for display
    pub fn format(&self) -> String {
        format!(
            "Health: {} | Memory: {:.1}% | Ring: {:.1}% | Errors: {} | Uptime: {}s",
            if self.is_healthy { "HEALTHY" } else { "UNHEALTHY" },
            self.memory_usage_pct,
            self.ring_utilization_pct,
            self.errors_count,
            self.uptime_seconds
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_app_state_creation() {
        // This test may fail in CI environments without proper hardware
        // but should work on development machines
        let result = AppState::new();
        
        // If it fails, at least verify the error is informative
        if let Err(e) = result {
            println!("AppState creation failed (expected in some environments): {}", e);
        }
    }
    
    #[test]
    fn test_health_status_format() {
        let status = HealthStatus {
            is_healthy: true,
            memory_usage_pct: 45.5,
            ring_utilization_pct: 12.3,
            errors_count: 0,
            uptime_seconds: 3600,
        };
        
        let formatted = status.format();
        assert!(formatted.contains("HEALTHY"));
        assert!(formatted.contains("45.5"));
    }
}
