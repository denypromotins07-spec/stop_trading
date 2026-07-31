//! High-Throughput Non-Blocking Prometheus Metrics Exporter
//! 
//! Batches and exposes internal actor states, RAM usage, and tick-to-trade
//! latencies without blocking hot execution threads.

use std::sync::atomic::{AtomicU64, AtomicI64, Ordering};
use std::collections::HashMap;
use std::time::Instant;

/// Metric type enumeration
#[derive(Debug, Clone, Copy)]
pub enum MetricType {
    Counter,
    Gauge,
    Histogram,
}

/// Prometheus metric
#[derive(Debug, Clone)]
pub struct PrometheusMetric {
    pub name: String,
    pub value: f64,
    pub metric_type: MetricType,
    pub labels: HashMap<String, String>,
    pub timestamp_ms: u64,
}

/// Metrics batch for efficient export
#[derive(Debug, Clone)]
pub struct MetricsBatch {
    pub metrics: Vec<PrometheusMetric>,
    pub batch_timestamp_ms: u64,
}

/// Prometheus exporter with atomic counters
pub struct PrometheusExporter {
    tick_to_trade_latency_sum: AtomicU64,
    tick_to_trade_count: AtomicU64,
    ram_usage_bytes: AtomicU64,
    active_actors: AtomicU64,
    messages_processed: AtomicU64,
    errors_total: AtomicU64,
    last_export_time: AtomicU64,
    enabled: AtomicBool,
}

impl PrometheusExporter {
    pub fn new() -> Self {
        Self {
            tick_to_trade_latency_sum: AtomicU64::new(0),
            tick_to_trade_count: AtomicU64::new(0),
            ram_usage_bytes: AtomicU64::new(0),
            active_actors: AtomicU64::new(0),
            messages_processed: AtomicU64::new(0),
            errors_total: AtomicU64::new(0),
            last_export_time: AtomicU64::new(0),
            enabled: AtomicBool::new(true),
        }
    }
    
    /// Record tick-to-trade latency (non-blocking)
    #[inline]
    pub fn record_tick_to_trade(&self, latency_us: u64) {
        if !self.enabled.load(Ordering::Relaxed) {
            return;
        }
        self.tick_to_trade_latency_sum.fetch_add(latency_us, Ordering::Relaxed);
        self.tick_to_trade_count.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Update RAM usage gauge
    pub fn update_ram_usage(&self, bytes: u64) {
        self.ram_usage_bytes.store(bytes, Ordering::Relaxed);
    }
    
    /// Update active actors count
    pub fn update_active_actors(&self, count: u64) {
        self.active_actors.store(count, Ordering::Relaxed);
    }
    
    /// Increment messages processed counter
    pub fn increment_messages(&self) {
        self.messages_processed.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Increment error counter
    pub fn increment_errors(&self) {
        self.errors_total.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Build metrics batch for export
    pub fn build_batch(&self) -> MetricsBatch {
        let now_ms = Instant::now().elapsed().as_millis() as u64;
        let mut metrics = Vec::with_capacity(6);
        
        // Tick-to-trade latency histogram summary
        let t2t_count = self.tick_to_trade_count.load(Ordering::Acquire);
        let t2t_sum = self.tick_to_trade_latency_sum.load(Ordering::Acquire);
        
        if t2t_count > 0 {
            metrics.push(PrometheusMetric {
                name: "hft_tick_to_trade_latency_us".to_string(),
                value: t2t_sum as f64 / t2t_count as f64,
                metric_type: MetricType::Gauge,
                labels: [("quantile".to_string(), "avg".to_string())].iter().cloned().collect(),
                timestamp_ms: now_ms,
            });
            metrics.push(PrometheusMetric {
                name: "hft_tick_to_trade_count".to_string(),
                value: t2t_count as f64,
                metric_type: MetricType::Counter,
                labels: HashMap::new(),
                timestamp_ms: now_ms,
            });
        }
        
        // RAM usage gauge
        metrics.push(PrometheusMetric {
            name: "hft_ram_usage_bytes".to_string(),
            value: self.ram_usage_bytes.load(Ordering::Acquire) as f64,
            metric_type: MetricType::Gauge,
            labels: HashMap::new(),
            timestamp_ms: now_ms,
        });
        
        // Active actors gauge
        metrics.push(PrometheusMetric {
            name: "hft_active_actors".to_string(),
            value: self.active_actors.load(Ordering::Acquire) as f64,
            metric_type: MetricType::Gauge,
            labels: HashMap::new(),
            timestamp_ms: now_ms,
        });
        
        // Messages processed counter
        metrics.push(PrometheusMetric {
            name: "hft_messages_processed_total".to_string(),
            value: self.messages_processed.load(Ordering::Acquire) as f64,
            metric_type: MetricType::Counter,
            labels: HashMap::new(),
            timestamp_ms: now_ms,
        });
        
        // Errors counter
        metrics.push(PrometheusMetric {
            name: "hft_errors_total".to_string(),
            value: self.errors_total.load(Ordering::Acquire) as f64,
            metric_type: MetricType::Counter,
            labels: HashMap::new(),
            timestamp_ms: now_ms,
        });
        
        self.last_export_time.store(now_ms, Ordering::Release);
        
        MetricsBatch {
            metrics,
            batch_timestamp_ms: now_ms,
        }
    }
    
    /// Export metrics in Prometheus text format
    pub fn export_text(&self) -> String {
        let batch = self.build_batch();
        let mut output = String::new();
        
        for metric in &batch.metrics {
            let labels_str: String = metric.labels.iter()
                .map(|(k, v)| format!("{}=\"{}\"", k, v))
                .collect::<Vec<_>>()
                .join(",");
            
            if labels_str.is_empty() {
                output.push_str(&format!("{} {}\n", metric.name, metric.value));
            } else {
                output.push_str(&format!("{}{{{}}} {}\n", metric.name, labels_str, metric.value));
            }
        }
        
        output
    }
    
    pub fn enable(&self) {
        self.enabled.store(true, Ordering::Release);
    }
    
    pub fn disable(&self) {
        self.enabled.store(false, Ordering::Release);
    }
}

impl Default for PrometheusExporter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_prometheus_exporter() {
        let exporter = PrometheusExporter::new();
        
        exporter.record_tick_to_trade(100);
        exporter.record_tick_to_trade(200);
        exporter.update_ram_usage(1_000_000_000);
        exporter.update_active_actors(10);
        exporter.increment_messages();
        exporter.increment_errors();
        
        let batch = exporter.build_batch();
        assert!(batch.metrics.len() >= 5);
        
        let text = exporter.export_text();
        assert!(text.contains("hft_tick_to_trade"));
        assert!(text.contains("hft_ram_usage_bytes"));
    }
}
