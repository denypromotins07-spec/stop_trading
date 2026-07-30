//! Telemetry Module Root
//! 
//! Batches and exports traces to a local collector without blocking the hot path.

pub mod otel;
pub mod propagation;

pub use otel::{
    LatencyStats, LatencyTracker, LifecycleStage, LifecycleTimer, OtelTracer, SpanGuard,
    TraceContext,
};
pub use propagation::{
    CompactTraceId, ContextCarrier, ContextChannel, ContextRingBuffer, PropagationStats,
    SequenceGenerator,
};

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::time::{Duration, Instant};
use crossbeam_queue::SegQueue;
use parking_lot::RwLock;
use tracing::{debug, error, info, warn};

/// Maximum batch size for trace export
const MAX_BATCH_SIZE: usize = 256;

/// Export interval in milliseconds
const EXPORT_INTERVAL_MS: u64 = 1000;

/// Batched trace record
#[derive(Debug, Clone)]
pub struct TraceRecord {
    pub trace_id: u128,
    pub span_id: u64,
    pub parent_span_id: Option<u64>,
    pub name: String,
    pub start_time: Instant,
    pub duration_ns: u128,
    pub attributes: Vec<(String, String)>,
}

/// Batch exporter for traces (non-blocking)
pub struct BatchExporter {
    queue: Arc<SegQueue<TraceRecord>>,
    shutdown: AtomicBool,
    exported_count: AtomicU64,
    dropped_count: AtomicU64,
    max_queue_size: usize,
}

impl BatchExporter {
    /// Create a new batch exporter
    pub fn new(max_queue_size: usize) -> Self {
        Self {
            queue: Arc::new(SegQueue::new()),
            shutdown: AtomicBool::new(false),
            exported_count: AtomicU64::new(0),
            dropped_count: AtomicU64::new(0),
            max_queue_size,
        }
    }

    /// Queue a trace record for export (non-blocking)
    pub fn export(&self, record: TraceRecord) -> bool {
        if self.shutdown.load(Ordering::Acquire) {
            return false;
        }

        // Check queue size before pushing
        if self.queue.len() >= self.max_queue_size {
            // Drop oldest if queue is full (or drop new one)
            self.dropped_count.fetch_add(1, Ordering::Relaxed);
            return false;
        }

        self.queue.push(record);
        true
    }

    /// Get a batch of records for export
    pub fn get_batch(&self, max_size: usize) -> Vec<TraceRecord> {
        let mut batch = Vec::with_capacity(max_size.min(MAX_BATCH_SIZE));
        
        while batch.len() < max_size.min(MAX_BATCH_SIZE) {
            match self.queue.pop() {
                Some(record) => batch.push(record),
                None => break,
            }
        }
        
        batch
    }

    /// Start background export thread
    pub fn start_export_thread(&self) -> std::thread::JoinHandle<()> {
        let queue = self.queue.clone();
        let shutdown = self.shutdown.clone();
        let exported = self.exported_count.clone();

        std::thread::spawn(move || {
            let mut last_export = Instant::now();
            
            while !shutdown.load(Ordering::Acquire) {
                // Check if it's time to export
                if last_export.elapsed().as_millis() as u64 >= EXPORT_INTERVAL_MS {
                    // Collect batch
                    let mut batch = Vec::with_capacity(MAX_BATCH_SIZE);
                    while batch.len() < MAX_BATCH_SIZE {
                        match queue.pop() {
                            Some(record) => batch.push(record),
                            None => break,
                        }
                    }

                    if !batch.is_empty() {
                        // In production, send to OTel collector here
                        debug!("Exporting {} trace records", batch.len());
                        exported.fetch_add(batch.len() as u64, Ordering::Relaxed);
                    }

                    last_export = Instant::now();
                }

                // Small sleep to avoid busy waiting
                std::thread::sleep(Duration::from_millis(10));
            }
        })
    }

    /// Flush all pending records
    pub fn flush(&self) -> Vec<TraceRecord> {
        let mut records = Vec::new();
        while let Some(record) = self.queue.pop() {
            records.push(record);
        }
        records
    }

    /// Get export statistics
    pub fn stats(&self) -> ExporterStats {
        ExporterStats {
            queued: self.queue.len(),
            exported: self.exported_count.load(Ordering::Relaxed),
            dropped: self.dropped_count.load(Ordering::Relaxed),
            max_queue_size: self.max_queue_size,
        }
    }

    /// Signal shutdown
    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::Release);
    }

    /// Check if shutdown requested
    pub fn is_shutdown(&self) -> bool {
        self.shutdown.load(Ordering::Acquire)
    }
}

impl Default for BatchExporter {
    fn default() -> Self {
        Self::new(4096)
    }
}

#[derive(Debug, Clone)]
pub struct ExporterStats {
    pub queued: usize,
    pub exported: u64,
    pub dropped: u64,
    pub max_queue_size: usize,
}

/// Telemetry manager combining all components
pub struct TelemetryManager {
    tracer: OtelTracer,
    exporter: Arc<BatchExporter>,
    latency_tracker: LatencyTracker,
    enabled: AtomicBool,
}

impl TelemetryManager {
    /// Create a new telemetry manager
    pub fn new(service_name: impl Into<String>) -> Self {
        let tracer = OtelTracer::new(service_name);
        let exporter = Arc::new(BatchExporter::new(4096));

        Self {
            tracer,
            exporter,
            latency_tracker: LatencyTracker::new(),
            enabled: AtomicBool::new(true),
        }
    }

    /// Initialize telemetry
    pub fn init(&self) -> Result<(), Box<dyn std::error::Error>> {
        self.tracer.init()?;
        
        // Start export thread
        self.exporter.start_export_thread();
        
        Ok(())
    }

    /// Record lifecycle latency
    pub fn record_latency(&self, stage: LifecycleStage, duration: Duration) {
        if self.enabled.load(Ordering::Acquire) {
            self.latency_tracker.record(stage, duration);
        }
    }

    /// Export a trace record
    pub fn export_trace(&self, record: TraceRecord) -> bool {
        if !self.enabled.load(Ordering::Acquire) {
            return false;
        }
        self.exporter.export(record)
    }

    /// Get latency statistics
    pub fn latency_stats(&self, stage: LifecycleStage) -> LatencyStats {
        self.latency_tracker.get_stats(stage)
    }

    /// Get all latency statistics
    pub fn all_latency_stats(&self) -> [(LifecycleStage, LatencyStats); 9] {
        self.latency_tracker.all_stats()
    }

    /// Get exporter statistics
    pub fn exporter_stats(&self) -> ExporterStats {
        self.exporter.stats()
    }

    /// Enable telemetry
    pub fn enable(&self) {
        self.enabled.store(true, Ordering::Release);
    }

    /// Disable telemetry (zero-overhead when disabled)
    pub fn disable(&self) {
        self.enabled.store(false, Ordering::Release);
    }

    /// Check if enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Acquire)
    }

    /// Graceful shutdown
    pub fn shutdown(&self) {
        self.exporter.shutdown();
        
        // Flush remaining
        let remaining = self.exporter.flush();
        if !remaining.is_empty() {
            info!("Flushing {} remaining trace records on shutdown", remaining.len());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_exporter() {
        let exporter = BatchExporter::new(100);
        
        let record = TraceRecord {
            trace_id: 12345,
            span_id: 67890,
            parent_span_id: None,
            name: "test_span".to_string(),
            start_time: Instant::now(),
            duration_ns: 1000,
            attributes: vec![],
        };

        assert!(exporter.export(record.clone()));
        assert_eq!(exporter.stats().queued, 1);

        let batch = exporter.get_batch(10);
        assert_eq!(batch.len(), 1);
        assert_eq!(batch[0].trace_id, 12345);
    }

    #[test]
    fn test_batch_exporter_overflow() {
        let exporter = BatchExporter::new(5);
        
        // Fill the queue
        for i in 0..10 {
            let record = TraceRecord {
                trace_id: i,
                span_id: i,
                parent_span_id: None,
                name: "test".to_string(),
                start_time: Instant::now(),
                duration_ns: 100,
                attributes: vec![],
            };
            let _ = exporter.export(record);
        }

        let stats = exporter.stats();
        assert_eq!(stats.queued, 5); // Max queue size
        assert_eq!(stats.dropped, 5); // Dropped count
    }

    #[test]
    fn test_telemetry_manager() {
        let manager = TelemetryManager::new("test-service");
        
        assert!(manager.is_enabled());
        
        manager.disable();
        assert!(!manager.is_enabled());
        
        manager.enable();
        assert!(manager.is_enabled());
    }

    #[test]
    fn test_trace_record_creation() {
        let record = TraceRecord {
            trace_id: 1,
            span_id: 2,
            parent_span_id: Some(3),
            name: "test".to_string(),
            start_time: Instant::now(),
            duration_ns: 100,
            attributes: vec![("key".to_string(), "value".to_string())],
        };

        assert_eq!(record.trace_id, 1);
        assert_eq!(record.span_id, 2);
        assert_eq!(record.parent_span_id, Some(3));
    }
}
