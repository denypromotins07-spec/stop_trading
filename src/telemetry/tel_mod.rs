//! Telemetry Module Root
//! 
//! Aggregates eBPF traces and internal metrics for the observability stack.

pub mod ebpf_hooks;
pub mod prometheus;

pub use ebpf_hooks::{EbpfHooks, EbpfEvent, EbpfEventType, EbpfStats};
pub use prometheus::{PrometheusExporter, PrometheusMetric, MetricsBatch, MetricType};

/// Combined telemetry state
#[derive(Debug, Clone)]
pub struct TelemetryState {
    pub ebpf_stats: EbpfStats,
    pub prometheus_text: String,
    pub timestamp_ms: u64,
}

/// Telemetry aggregator
pub struct TelemetryAggregator {
    ebpf_hooks: EbpfHooks,
    prometheus: PrometheusExporter,
}

impl TelemetryAggregator {
    pub fn new() -> Self {
        Self {
            ebpf_hooks: EbpfHooks::new(),
            prometheus: PrometheusExporter::new(),
        }
    }
    
    pub fn record_event(&self, event: EbpfEvent) {
        self.ebpf_hooks.record_event(event);
    }
    
    pub fn record_tick_to_trade(&self, latency_us: u64) {
        self.prometheus.record_tick_to_trade(latency_us);
    }
    
    pub fn update_ram_usage(&self, bytes: u64) {
        self.prometheus.update_ram_usage(bytes);
    }
    
    pub fn update_active_actors(&self, count: u64) {
        self.prometheus.update_active_actors(count);
    }
    
    pub fn get_state(&self) -> TelemetryState {
        TelemetryState {
            ebpf_stats: self.ebpf_hooks.get_stats(),
            prometheus_text: self.prometheus.export_text(),
            timestamp_ms: std::time::Instant::now().elapsed().as_millis() as u64,
        }
    }
    
    pub fn shutdown(&self) {
        self.ebpf_hooks.disable();
        self.prometheus.disable();
    }
}

impl Default for TelemetryAggregator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_telemetry_aggregator() {
        let agg = TelemetryAggregator::new();
        
        let event = EbpfEvent {
            timestamp_ns: 1000,
            event_type: EbpfEventType::NetworkRx,
            latency_us: 50,
            pid: 1,
            cpu_id: 0,
        };
        
        agg.record_event(event);
        agg.record_tick_to_trade(100);
        
        let state = agg.get_state();
        assert!(!state.prometheus_text.is_empty());
    }
}
