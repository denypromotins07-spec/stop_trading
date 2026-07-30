//! OpenTelemetry Integration for Zero-Cost Tracing
//! 
//! Integrates OpenTelemetry with zero-cost tracing spans for tick-to-trade lifecycle.
//! Measures exact nanosecond delays across network parsing, Disruptor routing, and execution.

use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, error, info, span, warn, Level};
use tracing_subscriber::{prelude::*, Registry};

/// Maximum trace buffer size (bounded for RAM limits)
const MAX_TRACE_BUFFER_SIZE: usize = 4096;

/// Trace context for propagating across async boundaries
#[derive(Debug, Clone)]
pub struct TraceContext {
    pub trace_id: u128,
    pub span_id: u64,
    pub parent_span_id: Option<u64>,
    pub start_time: Instant,
    pub attributes: Vec<(String, String)>,
}

impl TraceContext {
    pub fn new() -> Self {
        Self {
            trace_id: generate_trace_id(),
            span_id: generate_span_id(),
            parent_span_id: None,
            start_time: Instant::now(),
            attributes: Vec::new(),
        }
    }

    pub fn with_attribute(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.attributes.push((key.into(), value.into()));
        self
    }

    pub fn child(&self) -> Self {
        Self {
            trace_id: self.trace_id,
            span_id: generate_span_id(),
            parent_span_id: Some(self.span_id),
            start_time: Instant::now(),
            attributes: Vec::new(),
        }
    }
}

impl Default for TraceContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Generate unique trace ID
fn generate_trace_id() -> u128 {
    use std::sync::atomic::{AtomicU128, Ordering};
    static COUNTER: AtomicU128 = AtomicU128::new(1);
    COUNTER.fetch_add(1, Ordering::Relaxed)
}

/// Generate unique span ID
fn generate_span_id() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(1);
    COUNTER.fetch_add(1, Ordering::Relaxed)
}

/// Tick-to-trade lifecycle stages
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LifecycleStage {
    NetworkReceive,
    MessageParse,
    DisruptorRoute,
    SignalCompute,
    RiskCheck,
    OrderBuild,
    ExecutionSend,
    AckReceive,
    Complete,
}

impl LifecycleStage {
    pub fn as_str(&self) -> &'static str {
        match self {
            LifecycleStage::NetworkReceive => "network_receive",
            LifecycleStage::MessageParse => "message_parse",
            LifecycleStage::DisruptorRoute => "disruptor_route",
            LifecycleStage::SignalCompute => "signal_compute",
            LifecycleStage::RiskCheck => "risk_check",
            LifecycleStage::OrderBuild => "order_build",
            LifecycleStage::ExecutionSend => "execution_send",
            LifecycleStage::AckReceive => "ack_receive",
            LifecycleStage::Complete => "complete",
        }
    }
}

/// High-precision timer for lifecycle measurement
pub struct LifecycleTimer {
    context: TraceContext,
    stage: LifecycleStage,
    start: Instant,
}

impl LifecycleTimer {
    pub fn new(stage: LifecycleStage) -> Self {
        Self {
            context: TraceContext::new(),
            stage,
            start: Instant::now(),
        }
    }

    pub fn with_context(mut self, context: TraceContext) -> Self {
        self.context = context;
        self
    }

    /// Record completion and return duration in nanoseconds
    pub fn finish(self) -> Duration {
        let duration = self.start.elapsed();
        
        debug!(
            target: "lifecycle",
            trace_id = ?self.context.trace_id,
            stage = %self.stage.as_str(),
            duration_ns = duration.as_nanos(),
            "Lifecycle stage completed"
        );
        
        duration
    }

    /// Get elapsed time without finishing
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }
}

/// Aggregated latency statistics for a lifecycle stage
#[derive(Debug, Clone, Default)]
pub struct LatencyStats {
    pub count: u64,
    pub total_ns: u128,
    pub min_ns: u128,
    pub max_ns: u128,
    pub p50_ns: u128,
    pub p99_ns: u128,
}

impl LatencyStats {
    pub fn record(&mut self, duration_ns: u128) {
        self.count += 1;
        self.total_ns += duration_ns;
        
        if self.min_ns == 0 || duration_ns < self.min_ns {
            self.min_ns = duration_ns;
        }
        if duration_ns > self.max_ns {
            self.max_ns = duration_ns;
        }
    }

    pub fn avg_ns(&self) -> u128 {
        if self.count == 0 {
            return 0;
        }
        self.total_ns / self.count as u128
    }
}

/// OpenTelemetry tracer manager
pub struct OtelTracer {
    enabled: bool,
    service_name: String,
    sample_rate: f64,
}

impl OtelTracer {
    /// Create a new OpenTelemetry tracer
    pub fn new(service_name: impl Into<String>) -> Self {
        Self {
            enabled: true,
            service_name: service_name.into(),
            sample_rate: 1.0, // Sample all traces by default
        }
    }

    /// Initialize the tracing subscriber
    pub fn init(&self) -> Result<(), Box<dyn std::error::Error>> {
        if !self.enabled {
            return Ok(());
        }

        // Create filtering layer for sampling
        let filter = tracing_subscriber::filter::Targets::new()
            .with_target("hft", Level::DEBUG)
            .with_target("lifecycle", Level::INFO);

        // Format layer with JSON output for OTel compatibility
        let fmt_layer = tracing_subscriber::fmt::layer()
            .json()
            .with_current_span(false)
            .with_span_list(true)
            .with_thread_ids(true)
            .with_thread_names(true);

        // Build subscriber
        let subscriber = Registry::default()
            .with(filter)
            .with(fmt_layer);

        tracing::subscriber::set_global_default(subscriber)?;
        
        info!("OpenTelemetry tracer initialized for service: {}", self.service_name);
        Ok(())
    }

    /// Start a new trace span
    pub fn start_span(&self, name: &str, context: &TraceContext) -> SpanGuard {
        if !self.should_sample() {
            return SpanGuard::disabled();
        }

        let span = span!(
            Level::DEBUG,
            name,
            trace_id = ?context.trace_id,
            span_id = ?context.span_id,
        );
        let _enter = span.enter();

        SpanGuard {
            enabled: true,
            context: context.clone(),
            start: Instant::now(),
        }
    }

    /// Check if trace should be sampled
    fn should_sample(&self) -> bool {
        if self.sample_rate >= 1.0 {
            return true;
        }
        
        use std::sync::atomic::{AtomicU64, Ordering};
        static SAMPLER: AtomicU64 = AtomicU64::new(0);
        
        let val = SAMPLER.fetch_add(1, Ordering::Relaxed);
        (val as f64 / u64::MAX as f64) < self.sample_rate
    }

    /// Set sampling rate (0.0 to 1.0)
    pub fn set_sample_rate(&mut self, rate: f64) {
        self.sample_rate = rate.clamp(0.0, 1.0);
    }

    /// Export traces to collector (non-blocking)
    pub fn export_traces(&self) {
        // In production, this would batch and send to OTel collector
        debug!("Exporting traces to collector");
    }
}

/// RAII guard for span lifecycle
pub struct SpanGuard {
    enabled: bool,
    context: TraceContext,
    start: Instant,
}

impl SpanGuard {
    fn disabled() -> Self {
        Self {
            enabled: false,
            context: TraceContext::default(),
            start: Instant::now(),
        }
    }

    /// Add attribute to span
    pub fn with_attribute(&mut self, key: impl Into<String>, value: impl Into<String>) {
        if self.enabled {
            self.context.attributes.push((key.into(), value.into()));
        }
    }

    /// Record event within span
    pub fn record_event(&self, event: &str) {
        if self.enabled {
            debug!(
                target: "otel",
                trace_id = ?self.context.trace_id,
                event = %event,
                elapsed_ns = self.start.elapsed().as_nanos(),
                "Span event"
            );
        }
    }
}

impl Drop for SpanGuard {
    fn drop(&mut self) {
        if self.enabled {
            let duration = self.start.elapsed();
            debug!(
                target: "otel",
                trace_id = ?self.context.trace_id,
                span_id = ?self.context.span_id,
                duration_ns = duration.as_nanos(),
                "Span completed"
            );
        }
    }
}

/// Global latency tracker for all lifecycle stages
pub struct LatencyTracker {
    stats: parking_lot::RwLock<[LatencyStats; 9]>,
}

impl LatencyTracker {
    pub fn new() -> Self {
        Self {
            stats: parking_lot::RwLock::new(Default::default()),
        }
    }

    /// Record latency for a stage
    pub fn record(&self, stage: LifecycleStage, duration: Duration) {
        let idx = stage as usize;
        let mut stats = self.stats.write();
        stats[idx].record(duration.as_nanos() as u128);
    }

    /// Get statistics for a stage
    pub fn get_stats(&self, stage: LifecycleStage) -> LatencyStats {
        let idx = stage as usize;
        self.stats.read()[idx].clone()
    }

    /// Get all statistics
    pub fn all_stats(&self) -> [(LifecycleStage, LatencyStats); 9] {
        let stats = self.stats.read();
        [
            (LifecycleStage::NetworkReceive, stats[0].clone()),
            (LifecycleStage::MessageParse, stats[1].clone()),
            (LifecycleStage::DisruptorRoute, stats[2].clone()),
            (LifecycleStage::SignalCompute, stats[3].clone()),
            (LifecycleStage::RiskCheck, stats[4].clone()),
            (LifecycleStage::OrderBuild, stats[5].clone()),
            (LifecycleStage::ExecutionSend, stats[6].clone()),
            (LifecycleStage::AckReceive, stats[7].clone()),
            (LifecycleStage::Complete, stats[8].clone()),
        ]
    }

    /// Reset all statistics
    pub fn reset(&self) {
        let mut stats = self.stats.write();
        *stats = Default::default();
    }
}

impl Default for LatencyTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trace_context_creation() {
        let ctx = TraceContext::new();
        assert_ne!(ctx.trace_id, 0);
        assert_ne!(ctx.span_id, 0);
        assert!(ctx.parent_span_id.is_none());
    }

    #[test]
    fn test_trace_context_child() {
        let parent = TraceContext::new();
        let child = parent.child();
        
        assert_eq!(child.trace_id, parent.trace_id);
        assert_ne!(child.span_id, parent.span_id);
        assert_eq!(child.parent_span_id, Some(parent.span_id));
    }

    #[test]
    fn test_lifecycle_timer() {
        let timer = LifecycleTimer::new(LifecycleStage::SignalCompute);
        std::thread::sleep(Duration::from_millis(1));
        let duration = timer.finish();
        
        assert!(duration.as_millis() >= 1);
    }

    #[test]
    fn test_latency_stats() {
        let mut stats = LatencyStats::default();
        
        stats.record(100);
        stats.record(200);
        stats.record(300);
        
        assert_eq!(stats.count, 3);
        assert_eq!(stats.min_ns, 100);
        assert_eq!(stats.max_ns, 300);
        assert_eq!(stats.avg_ns(), 200);
    }

    #[test]
    fn test_otel_tracer_creation() {
        let tracer = OtelTracer::new("test-service");
        assert_eq!(tracer.service_name, "test-service");
        assert!(tracer.enabled);
    }
}
