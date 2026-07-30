//! Telemetry Module Root
//!
//! This module provides core telemetry and observability:
//! - Logger: Zero-cost asynchronous structured logger
//! - Metrics: Lock-free metrics collector using atomic counters
//!
//! The logger and metrics registry are connected to the global event bus
//! for real-time system health monitoring and pre-trade risk bus integration.

pub mod logger;
pub mod metrics;

use std::sync::Arc;
use anyhow::Context;

pub use logger::{AsyncLogger, LogEntry, LogLevel, LoggerConfig, LoggerStats};
pub use metrics::{
    Counter, Gauge, Histogram, HistogramStats, MetricsSnapshot,
    RateCalculator, SystemMetrics,
};

/// Telemetry configuration
#[derive(Debug, Clone)]
pub struct TelemetryConfig {
    /// Logger configuration
    pub logger: LoggerConfig,
    /// Metrics collection interval in milliseconds
    pub metrics_interval_ms: u64,
    /// Enable terminal dashboard
    pub enable_dashboard: bool,
    /// Dashboard refresh interval in milliseconds
    pub dashboard_interval_ms: u64,
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            logger: LoggerConfig::default(),
            metrics_interval_ms: 1000,
            enable_dashboard: true,
            dashboard_interval_ms: 500,
        }
    }
}

/// Main telemetry system combining logger and metrics
pub struct Telemetry {
    /// Async logger instance
    logger: Arc<AsyncLogger>,
    /// System metrics collector
    metrics: Arc<SystemMetrics>,
    /// Configuration
    config: TelemetryConfig,
    /// Shutdown flag
    shutdown: Arc<std::sync::atomic::AtomicBool>,
}

unsafe impl Send for Telemetry {}
unsafe impl Sync for Telemetry {}

impl Telemetry {
    /// Create a new telemetry system with the given configuration
    pub fn new(config: TelemetryConfig) -> Result<Arc<Self>, anyhow::Error> {
        let logger = AsyncLogger::new(config.logger.clone())?;
        
        let telemetry = Arc::new(Telemetry {
            logger,
            metrics: Arc::new(SystemMetrics::new()),
            config,
            shutdown: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        });
        
        Ok(telemetry)
    }
    
    /// Start all telemetry components
    pub fn start(self: &Arc<Self>) -> Result<(), anyhow::Error> {
        tracing::info!("Starting telemetry system...");
        
        // Start the async logger
        self.logger.start()?;
        
        // Start metrics collection thread if dashboard is enabled
        if self.config.enable_dashboard {
            self.start_dashboard_loop()?;
        }
        
        tracing::info!("Telemetry system started");
        
        Ok(())
    }
    
    /// Start the dashboard display loop
    fn start_dashboard_loop(self: &Arc<Self>) -> Result<(), anyhow::Error> {
        let metrics = Arc::clone(&self.metrics);
        let logger = Arc::clone(&self.logger);
        let shutdown = Arc::clone(&self.shutdown);
        let interval_ms = self.config.dashboard_interval_ms;
        
        std::thread::Builder::new()
            .name("telemetry-dashboard".to_string())
            .spawn(move || {
                use std::time::Duration;
                
                let mut last_log_time = std::time::Instant::now();
                
                while !shutdown.load(std::sync::atomic::Ordering::Relaxed) {
                    // Print dashboard to stdout
                    println!("\x1b[H\x1b[2J"); // Clear screen and move cursor to top
                    println!("{}", metrics.format_dashboard());
                    
                    // Log metrics periodically (every 10 seconds)
                    if last_log_time.elapsed() >= Duration::from_secs(10) {
                        let snapshot = metrics.get_snapshot();
                        
                        logger.info(
                            "telemetry.metrics",
                            format!(
                                "tick_to_trade_p99={:.1}µs order_book_rate={:.1}/s trade_rate={:.1}/s ram={:.1}MB",
                                snapshot.tick_to_trade.p99_ns as f64 / 1000.0,
                                snapshot.order_book_rate,
                                snapshot.trade_rate,
                                snapshot.ram_usage_mb
                            )
                        );
                        
                        last_log_time = std::time::Instant::now();
                    }
                    
                    std::thread::sleep(Duration::from_millis(interval_ms));
                }
            })
            .context("Failed to spawn dashboard thread")?;
        
        Ok(())
    }
    
    /// Get the logger instance
    pub fn logger(&self) -> &Arc<AsyncLogger> {
        &self.logger
    }
    
    /// Get the metrics collector
    pub fn metrics(&self) -> &Arc<SystemMetrics> {
        &self.metrics
    }
    
    /// Record tick-to-trade latency
    pub fn record_tick_to_trade(&self, latency_ns: u64) {
        self.metrics.record_tick_to_trade(latency_ns);
    }
    
    /// Record order book update
    pub fn record_order_book_update(&self) {
        self.metrics.record_order_book_update();
    }
    
    /// Record trade execution
    pub fn record_trade_execution(&self) {
        self.metrics.record_trade_execution();
    }
    
    /// Update RAM usage
    pub fn update_ram_usage(&self, bytes: u64) {
        self.metrics.set_ram_usage(bytes);
    }
    
    /// Stop telemetry gracefully
    pub fn stop(&self) {
        tracing::info!("Stopping telemetry system...");
        
        self.shutdown.store(true, std::sync::atomic::Ordering::Release);
        self.logger.stop();
    }
    
    /// Get combined statistics
    pub fn get_stats(&self) -> TelemetryStats {
        TelemetryStats {
            logger: self.logger.get_stats(),
            metrics: self.metrics.get_snapshot(),
        }
    }
}

impl Drop for Telemetry {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Combined telemetry statistics
#[derive(Debug, Clone)]
pub struct TelemetryStats {
    pub logger: LoggerStats,
    pub metrics: MetricsSnapshot,
}

impl TelemetryStats {
    pub fn format(&self) -> String {
        format!(
            "Telemetry Stats:\n{}\n{}",
            self.logger.format(),
            self.metrics.format_dashboard()
        )
    }
}

/// Global telemetry instance (lazy initialized)
static mut GLOBAL_TELEMETRY: Option<Arc<Telemetry>> = None;

/// Initialize the global telemetry singleton
pub fn init_global_telemetry(config: TelemetryConfig) -> Result<Arc<Telemetry>, anyhow::Error> {
    let telemetry = Telemetry::new(config)?;
    telemetry.start()?;
    
    unsafe {
        GLOBAL_TELEMETRY = Some(Arc::clone(&telemetry));
    }
    
    Ok(telemetry)
}

/// Get the global telemetry instance
///
/// # Panics
/// Panics if telemetry hasn't been initialized yet
pub fn get_global_telemetry() -> Arc<Telemetry> {
    unsafe {
        GLOBAL_TELEMETRY
            .as_ref()
            .expect("Global telemetry not initialized. Call init_global_telemetry() first.")
            .clone()
    }
}

/// Convenience function to record tick-to-trade latency globally
pub fn record_tick_to_trade(latency_ns: u64) {
    unsafe {
        if let Some(ref telemetry) = GLOBAL_TELEMETRY {
            telemetry.record_tick_to_trade(latency_ns);
        }
    }
}

/// Convenience function to record order book update globally
pub fn record_order_book_update() {
    unsafe {
        if let Some(ref telemetry) = GLOBAL_TELEMETRY {
            telemetry.record_order_book_update();
        }
    }
}

/// Convenience function to record trade execution globally
pub fn record_trade_execution() {
    unsafe {
        if let Some(ref telemetry) = GLOBAL_TELEMETRY {
            telemetry.record_trade_execution();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    
    #[test]
    fn test_telemetry_creation() {
        let config = TelemetryConfig::default();
        let telemetry = Telemetry::new(config).unwrap();
        
        assert!(Arc::strong_count(&telemetry) == 1);
    }
    
    #[test]
    fn test_telemetry_start_stop() {
        let config = TelemetryConfig {
            enable_dashboard: false, // Disable dashboard for faster test
            ..Default::default()
        };
        
        let telemetry = Telemetry::new(config).unwrap();
        telemetry.start().unwrap();
        
        // Give it time to start
        std::thread::sleep(Duration::from_millis(50));
        
        telemetry.stop();
    }
    
    #[test]
    fn test_metrics_recording() {
        let config = TelemetryConfig {
            enable_dashboard: false,
            ..Default::default()
        };
        
        let telemetry = Telemetry::new(config).unwrap();
        
        telemetry.record_tick_to_trade(100_000); // 100µs
        telemetry.record_tick_to_trade(200_000); // 200µs
        telemetry.record_order_book_update();
        telemetry.record_order_book_update();
        telemetry.record_order_book_update();
        telemetry.update_ram_usage(50 * 1024 * 1024); // 50MB
        
        let snapshot = telemetry.metrics().get_snapshot();
        
        assert_eq!(snapshot.tick_to_trade.count, 2);
        assert_eq!(snapshot.active_orders, 0);
        assert!((snapshot.ram_usage_mb - 50.0).abs() < 0.1);
    }
}
