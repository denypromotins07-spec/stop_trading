//! Lock-Free Metrics Collector using Atomic Counters
//!
//! This module builds a lock-free metrics collector using atomic counters to track system health.
//! It monitors tick-to-trade latency, order book update rates, and RAM usage, pushing data
//! to the terminal UI for real-time monitoring.
//!
//! Features:
//! - Zero-lock metric updates
//! - Histogram support for latency distributions
//! - Rate calculation (events per second)
//! - Thread-safe snapshots

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// A lock-free counter metric
pub struct Counter {
    value: AtomicU64,
}

impl Counter {
    pub fn new() -> Self {
        Self {
            value: AtomicU64::new(0),
        }
    }
    
    /// Increment the counter
    pub fn inc(&self) {
        self.value.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Increment by a specific amount
    pub fn inc_by(&self, n: u64) {
        self.value.fetch_add(n, Ordering::Relaxed);
    }
    
    /// Get current value
    pub fn get(&self) -> u64 {
        self.value.load(Ordering::Relaxed)
    }
    
    /// Reset to zero and return previous value
    pub fn reset(&self) -> u64 {
        self.value.swap(0, Ordering::Relaxed)
    }
}

impl Default for Counter {
    fn default() -> Self {
        Self::new()
    }
}

/// A gauge metric (can go up or down)
pub struct Gauge {
    value: AtomicU64,
}

impl Gauge {
    pub fn new() -> Self {
        Self {
            value: AtomicU64::new(0),
        }
    }
    
    /// Set the gauge value
    pub fn set(&self, value: u64) {
        self.value.store(value, Ordering::Relaxed);
    }
    
    /// Add to the gauge value
    pub fn add(&self, delta: i64) {
        if delta >= 0 {
            self.value.fetch_add(delta as u64, Ordering::Relaxed);
        } else {
            self.value.fetch_sub((-delta) as u64, Ordering::Relaxed);
        }
    }
    
    /// Get current value
    pub fn get(&self) -> u64 {
        self.value.load(Ordering::Relaxed)
    }
}

impl Default for Gauge {
    fn default() -> Self {
        Self::new()
    }
}

/// A simple histogram for latency tracking using fixed buckets
pub struct Histogram {
    /// Bucket counts (powers of 2 in microseconds)
    /// Bucket 0: 0-1us, Bucket 1: 1-2us, ..., Bucket 20: ~1s+
    buckets: [AtomicU64; 21],
    /// Total count
    count: AtomicU64,
    /// Sum of all values (for calculating mean)
    sum: AtomicU64,
    /// Minimum value (in nanoseconds)
    min: AtomicU64,
    /// Maximum value (in nanoseconds)
    max: AtomicU64,
}

impl Histogram {
    pub fn new() -> Self {
        const ZERO: AtomicU64 = AtomicU64::new(0);
        Self {
            buckets: [ZERO; 21],
            count: AtomicU64::new(0),
            sum: AtomicU64::new(0),
            min: AtomicU64::new(u64::MAX),
            max: AtomicU64::new(0),
        }
    }
    
    /// Record a value in nanoseconds
    pub fn record_ns(&self, value_ns: u64) {
        // Convert to microseconds for bucketing
        let value_us = value_ns / 1000;
        
        // Find bucket (log2 scale)
        let bucket = if value_us == 0 {
            0
        } else {
            (64 - value_us.leading_zeros()) as usize
        }.min(20);
        
        self.buckets[bucket].fetch_add(1, Ordering::Relaxed);
        self.count.fetch_add(1, Ordering::Relaxed);
        self.sum.fetch_add(value_ns, Ordering::Relaxed);
        
        // Update min/max
        let mut current_min = self.min.load(Ordering::Relaxed);
        while value_ns < current_min {
            match self.min.compare_exchange_weak(
                current_min,
                value_ns,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => current_min = actual,
            }
        }
        
        let mut current_max = self.max.load(Ordering::Relaxed);
        while value_ns > current_max {
            match self.max.compare_exchange_weak(
                current_max,
                value_ns,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => current_max = actual,
            }
        }
    }
    
    /// Get statistics snapshot
    pub fn get_stats(&self) -> HistogramStats {
        let count = self.count.load(Ordering::Relaxed);
        let sum = self.sum.load(Ordering::Relaxed);
        let min = self.min.load(Ordering::Relaxed);
        let max = self.max.load(Ordering::Relaxed);
        
        let mean = if count > 0 { sum / count } else { 0 };
        
        // Calculate approximate percentile (p50, p95, p99)
        let (p50, p95, p99) = self.calculate_percentiles();
        
        HistogramStats {
            count,
            mean_ns: mean,
            min_ns: min,
            max_ns: max,
            p50_ns: p50,
            p95_ns: p95,
            p99_ns: p99,
        }
    }
    
    /// Calculate approximate percentiles from buckets
    fn calculate_percentiles(&self) -> (u64, u64, u64) {
        let count = self.count.load(Ordering::Relaxed);
        if count == 0 {
            return (0, 0, 0);
        }
        
        let p50_target = count * 50 / 100;
        let p95_target = count * 95 / 100;
        let p99_target = count * 99 / 100;
        
        let mut cumulative = 0u64;
        let mut p50 = 0u64;
        let mut p95 = 0u64;
        let mut p99 = 0u64;
        
        for (i, bucket) in self.buckets.iter().enumerate() {
            cumulative += bucket.load(Ordering::Relaxed);
            
            // Each bucket represents 2^i microseconds
            let bucket_value_us = 1u64 << i;
            let bucket_value_ns = bucket_value_us * 1000;
            
            if p50 == 0 && cumulative >= p50_target {
                p50 = bucket_value_ns;
            }
            if p95 == 0 && cumulative >= p95_target {
                p95 = bucket_value_ns;
            }
            if p99 == 0 && cumulative >= p99_target {
                p99 = bucket_value_ns;
            }
        }
        
        (p50, p95, p99)
    }
    
    /// Reset the histogram
    pub fn reset(&self) {
        for bucket in &self.buckets {
            bucket.store(0, Ordering::Relaxed);
        }
        self.count.store(0, Ordering::Relaxed);
        self.sum.store(0, Ordering::Relaxed);
        self.min.store(u64::MAX, Ordering::Relaxed);
        self.max.store(0, Ordering::Relaxed);
    }
}

impl Default for Histogram {
    fn default() -> Self {
        Self::new()
    }
}

/// Histogram statistics snapshot
#[derive(Debug, Clone, Default)]
pub struct HistogramStats {
    pub count: u64,
    pub mean_ns: u64,
    pub min_ns: u64,
    pub max_ns: u64,
    pub p50_ns: u64,
    pub p95_ns: u64,
    pub p99_ns: u64,
}

impl HistogramStats {
    /// Format stats for display
    pub fn format_latency(&self) -> String {
        format!(
            "count={} mean={:.1}µs min={:.1}µs max={:.1}µs p50={:.1}µs p95={:.1}µs p99={:.1}µs",
            self.count,
            self.mean_ns as f64 / 1000.0,
            self.min_ns as f64 / 1000.0,
            self.max_ns as f64 / 1000.0,
            self.p50_ns as f64 / 1000.0,
            self.p95_ns as f64 / 1000.0,
            self.p99_ns as f64 / 1000.0,
        )
    }
}

/// Rate calculator for events per second
pub struct RateCalculator {
    counter: Counter,
    last_snapshot: AtomicU64,
    last_snapshot_time: AtomicUsize, // Store as unix timestamp seconds
}

impl RateCalculator {
    pub fn new() -> Self {
        Self {
            counter: Counter::new(),
            last_snapshot: AtomicU64::new(0),
            last_snapshot_time: AtomicUsize::new(0),
        }
    }
    
    /// Record an event
    pub fn record(&self) {
        self.counter.inc();
    }
    
    /// Calculate rate (events per second) since last call
    pub fn calculate_rate(&self) -> f64 {
        let current = self.counter.get();
        let last = self.last_snapshot.swap(current, Ordering::Relaxed);
        
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as usize;
        
        let last_time = self.last_snapshot_time.swap(now, Ordering::Relaxed);
        
        if last_time == 0 {
            // First call, initialize
            return 0.0;
        }
        
        let elapsed = (now - last_time) as f64;
        if elapsed <= 0.0 {
            return 0.0;
        }
        
        (current - last) as f64 / elapsed
    }
    
    /// Get total count
    pub fn total(&self) -> u64 {
        self.counter.get()
    }
}

impl Default for RateCalculator {
    fn default() -> Self {
        Self::new()
    }
}

/// All system metrics collected together
pub struct SystemMetrics {
    // Tick-to-trade latency
    pub tick_to_trade_latency: Histogram,
    
    // Order book update rate
    pub order_book_updates: RateCalculator,
    
    // Trade execution rate
    pub trade_executions: RateCalculator,
    
    // RAM usage (in bytes)
    pub ram_usage_bytes: Gauge,
    
    // Active orders count
    pub active_orders: Gauge,
    
    // Pending events in ring buffer
    pub pending_events: Gauge,
    
    // Errors count
    pub errors: Counter,
    
    // Warnings count
    pub warnings: Counter,
    
    // Uptime in seconds
    pub start_time: AtomicUsize,
}

impl SystemMetrics {
    pub fn new() -> Self {
        Self {
            tick_to_trade_latency: Histogram::new(),
            order_book_updates: RateCalculator::new(),
            trade_executions: RateCalculator::new(),
            ram_usage_bytes: Gauge::new(),
            active_orders: Gauge::new(),
            pending_events: Gauge::new(),
            errors: Counter::new(),
            warnings: Counter::new(),
            start_time: AtomicUsize::new(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs() as usize,
            ),
        }
    }
    
    /// Record tick-to-trade latency
    pub fn record_tick_to_trade(&self, latency_ns: u64) {
        self.tick_to_trade_latency.record_ns(latency_ns);
    }
    
    /// Record order book update
    pub fn record_order_book_update(&self) {
        self.order_book_updates.record();
    }
    
    /// Record trade execution
    pub fn record_trade_execution(&self) {
        self.trade_executions.record();
    }
    
    /// Set RAM usage
    pub fn set_ram_usage(&self, bytes: u64) {
        self.ram_usage_bytes.set(bytes);
    }
    
    /// Set active orders count
    pub fn set_active_orders(&self, count: u64) {
        self.active_orders.set(count);
    }
    
    /// Set pending events count
    pub fn set_pending_events(&self, count: u64) {
        self.pending_events.set(count);
    }
    
    /// Record an error
    pub fn record_error(&self) {
        self.errors.inc();
    }
    
    /// Record a warning
    pub fn record_warning(&self) {
        self.warnings.inc();
    }
    
    /// Get uptime in seconds
    pub fn uptime_seconds(&self) -> u64 {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as usize;
        let start = self.start_time.load(Ordering::Relaxed);
        (now - start) as u64
    }
    
    /// Get a complete snapshot of all metrics
    pub fn get_snapshot(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            tick_to_trade: self.tick_to_trade_latency.get_stats(),
            order_book_rate: self.order_book_updates.calculate_rate(),
            trade_rate: self.trade_executions.calculate_rate(),
            ram_usage_mb: self.ram_usage_bytes.get() as f64 / (1024.0 * 1024.0),
            active_orders: self.active_orders.get(),
            pending_events: self.pending_events.get(),
            errors: self.errors.get(),
            warnings: self.warnings.get(),
            uptime_seconds: self.uptime_seconds(),
        }
    }
    
    /// Format metrics for terminal display
    pub fn format_dashboard(&self) -> String {
        let snapshot = self.get_snapshot();
        snapshot.format_dashboard()
    }
}

impl Default for SystemMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Complete metrics snapshot
#[derive(Debug, Clone)]
pub struct MetricsSnapshot {
    pub tick_to_trade: HistogramStats,
    pub order_book_rate: f64,
    pub trade_rate: f64,
    pub ram_usage_mb: f64,
    pub active_orders: u64,
    pub pending_events: u64,
    pub errors: u64,
    pub warnings: u64,
    pub uptime_seconds: u64,
}

impl MetricsSnapshot {
    /// Format as dashboard string
    pub fn format_dashboard(&self) -> String {
        let uptime = Duration::from_secs(self.uptime_seconds);
        let uptime_str = format!(
            "{:02}:{:02}:{:02}",
            uptime.as_secs() / 3600,
            (uptime.as_secs() % 3600) / 60,
            uptime.as_secs() % 60,
        );
        
        format!(
            "╔══════════════════════════════════════════════════════════╗\n\
             ║  HFT CRYPTO BOT - SYSTEM METRICS                         ║\n\
             ╠══════════════════════════════════════════════════════════╣\n\
             ║ Uptime: {}                                  ║\n\
             ╠──────────────────────────────────────────────────────────╣\n\
             ║ LATENCY (tick-to-trade):                                 ║\n\
             ║   {}\n\
             ║                                        ║\n\
             ╠──────────────────────────────────────────────────────────╣\n\
             ║ RATES (events/sec):                                      ║\n\
             ║   Order Book Updates: {:>10.1}                          ║\n\
             ║   Trade Executions:   {:>10.1}                          ║\n\
             ╠──────────────────────────────────────────────────────────╣\n\
             ║ SYSTEM:                                                  ║\n\
             ║   RAM Usage:      {:>10.1} MB                           ║\n\
             ║   Active Orders:  {:>10}                                ║\n\
             ║   Pending Events: {:>10}                                ║\n\
             ║   Errors:         {:>10}                                ║\n\
             ║   Warnings:       {:>10}                                ║\n\
             ╚══════════════════════════════════════════════════════════╝",
            uptime_str,
            self.tick_to_trade.format_latency(),
            self.order_book_rate,
            self.trade_rate,
            self.ram_usage_mb,
            self.active_orders,
            self.pending_events,
            self.errors,
            self.warnings,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_counter() {
        let counter = Counter::new();
        
        counter.inc();
        counter.inc();
        counter.inc_by(5);
        
        assert_eq!(counter.get(), 7);
        
        let prev = counter.reset();
        assert_eq!(prev, 7);
        assert_eq!(counter.get(), 0);
    }
    
    #[test]
    fn test_gauge() {
        let gauge = Gauge::new();
        
        gauge.set(100);
        assert_eq!(gauge.get(), 100);
        
        gauge.add(-30);
        assert_eq!(gauge.get(), 70);
        
        gauge.add(50);
        assert_eq!(gauge.get(), 120);
    }
    
    #[test]
    fn test_histogram() {
        let histogram = Histogram::new();
        
        // Record some values
        histogram.record_ns(1000);      // 1µs
        histogram.record_ns(5000);      // 5µs
        histogram.record_ns(10000);     // 10µs
        histogram.record_ns(100000);    // 100µs
        histogram.record_ns(1000000);   // 1ms
        
        let stats = histogram.get_stats();
        
        assert_eq!(stats.count, 5);
        assert!(stats.mean_ns > 0);
        assert!(stats.min_ns <= 1000);
        assert!(stats.max_ns >= 1000000);
    }
    
    #[test]
    fn test_system_metrics() {
        let metrics = SystemMetrics::new();
        
        metrics.record_tick_to_trade(50000); // 50µs
        metrics.record_order_book_update();
        metrics.record_order_book_update();
        metrics.set_ram_usage(1024 * 1024 * 100); // 100MB
        
        let snapshot = metrics.get_snapshot();
        
        assert_eq!(snapshot.tick_to_trade.count, 1);
        assert_eq!(snapshot.active_orders, 0);
        assert!((snapshot.ram_usage_mb - 100.0).abs() < 0.1);
        
        println!("{}", metrics.format_dashboard());
    }
}
