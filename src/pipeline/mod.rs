//! Pipeline Module Root
//! 
//! Wires the parser, router, and network clients into a cohesive data ingestion engine.

pub mod parser;
pub mod router;

pub use parser::*;
pub use router::*;

use std::sync::Arc;
use anyhow::{Context, Result};
use crate::market_data::MarketDataEvent;
use crate::network::ws_client::RawWsMessage;

/// Pipeline configuration
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Parser configuration
    pub parser_config: ParserConfig,
    /// Router configuration
    pub router_config: RouterConfig,
    /// Channel buffer size
    pub channel_buffer_size: usize,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        PipelineConfig {
            parser_config: ParserConfig::default(),
            router_config: RouterConfig::default(),
            channel_buffer_size: 10_000,
        }
    }
}

/// Parser configuration placeholder
#[derive(Debug, Clone, Default)]
pub struct ParserConfig {
    /// Enable SIMD acceleration (if available)
    pub enable_simd: bool,
}

/// Complete data ingestion pipeline
pub struct DataIngestionPipeline {
    /// JSON parser
    parser: Arc<dyn JsonParser>,
    /// Message router
    router: Arc<MessageRouter>,
    /// Event receiver
    event_receiver: tokio::sync::Mutex<tokio::sync::mpsc::Receiver<RingBufferEvent>>,
    /// Shutdown flag
    shutdown: Arc<std::sync::atomic::AtomicBool>,
    /// Statistics
    stats: PipelineStats,
}

impl DataIngestionPipeline {
    /// Create a new data ingestion pipeline
    #[inline]
    pub fn new(config: PipelineConfig) -> Self {
        let (event_sender, event_receiver) = tokio::sync::mpsc::channel(config.channel_buffer_size);
        
        // Create parser
        let parser: Arc<dyn JsonParser> = if config.parser_config.enable_simd {
            #[cfg(feature = "simd")]
            {
                Arc::new(parser::SimdJsonParser::new())
            }
            #[cfg(not(feature = "simd"))]
            {
                log::warn!("SIMD requested but not available, falling back to serde_json");
                Arc::new(parser::SerdeJsonParser::new())
            }
        } else {
            Arc::new(parser::SerdeJsonParser::new())
        };

        // Create router
        let router = Arc::new(MessageRouter::new(
            config.router_config,
            event_sender,
            parser.clone(),
        ));

        DataIngestionPipeline {
            parser,
            router,
            event_receiver: tokio::sync::Mutex::new(event_receiver),
            shutdown: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            stats: PipelineStats::default(),
        }
    }

    /// Get the parser reference
    #[inline]
    pub fn parser(&self) -> Arc<dyn JsonParser> {
        self.parser.clone()
    }

    /// Get the router reference
    #[inline]
    pub fn router(&self) -> Arc<MessageRouter> {
        self.router.clone()
    }

    /// Route a raw message through the pipeline
    #[inline]
    pub async fn process_message(&self, raw_msg: RawWsMessage) -> Result<bool> {
        self.router.route_message(raw_msg).await
    }

    /// Receive the next event from the pipeline
    #[inline]
    pub async fn receive_event(&self) -> Option<RingBufferEvent> {
        let mut receiver = self.event_receiver.lock().await;
        receiver.recv().await
    }

    /// Receive events in a stream
    #[inline]
    pub fn event_stream(
        &self,
    ) -> impl futures_util::Stream<Item = RingBufferEvent> + Send + '_ {
        futures_util::stream::unfold(
            self.event_receiver.lock(),
            |receiver_fut| async move {
                let mut receiver = receiver_fut.await;
                receiver.recv().await.map(|event| (event, receiver_fut))
            },
        )
    }

    /// Signal shutdown
    #[inline]
    pub fn shutdown(&self) {
        self.shutdown.store(true, std::sync::atomic::Ordering::Relaxed);
    }

    /// Check if shutdown is requested
    #[inline]
    pub fn is_shutdown(&self) -> bool {
        self.shutdown.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get pipeline statistics
    #[inline]
    pub fn stats(&self) -> PipelineStatsSnapshot {
        PipelineStatsSnapshot {
            parser_stats: self.parser.stats(),
            router_stats: self.router.stats(),
            is_shutdown: self.is_shutdown(),
        }
    }

    /// Run the pipeline processor loop
    #[inline]
    pub async fn run_processor<F>(&self, mut handler: F) -> Result<()>
    where
        F: FnMut(RingBufferEvent) -> futures_util::future::Ready<Result<()>> + Send,
    {
        while !self.is_shutdown() {
            match self.receive_event().await {
                Some(event) => {
                    handler(event).await?;
                    self.router.mark_processed();
                }
                None => {
                    log::warn!("Event channel closed");
                    break;
                }
            }
        }
        
        log::info!("Pipeline processor stopped");
        Ok(())
    }
}

impl Default for DataIngestionPipeline {
    fn default() -> Self {
        Self::new(PipelineConfig::default())
    }
}

/// Static pipeline statistics
#[derive(Debug, Clone, Default)]
pub struct PipelineStats {
    _private: (),
}

/// Runtime pipeline statistics snapshot
#[derive(Debug, Clone)]
pub struct PipelineStatsSnapshot {
    pub parser_stats: ParserStats,
    pub router_stats: RouterStats,
    pub is_shutdown: bool,
}

impl PipelineStatsSnapshot {
    /// Get total messages processed
    #[inline]
    pub fn total_processed(&self) -> u64 {
        self.router_stats.routed_count
    }

    /// Get total messages dropped
    #[inline]
    pub fn total_dropped(&self) -> u64 {
        self.router_stats.dropped_count
    }

    /// Get overall success rate
    #[inline]
    pub fn success_rate(&self) -> f64 {
        self.parser_stats.success_rate() * (1.0 - self.router_stats.drop_rate() / 100.0)
    }
}

/// Builder for creating configured pipelines
pub struct PipelineBuilder {
    config: PipelineConfig,
}

impl PipelineBuilder {
    #[inline]
    pub fn new() -> Self {
        PipelineBuilder {
            config: PipelineConfig::default(),
        }
    }

    #[inline]
    pub fn with_channel_buffer(mut self, size: usize) -> Self {
        self.config.channel_buffer_size = size;
        self
    }

    #[inline]
    pub fn with_backpressure(mut self, enabled: bool, threshold: f64, max_pending: usize) -> Self {
        self.config.router_config.enable_backpressure = enabled;
        self.config.router_config.backpressure_threshold = threshold;
        self.config.router_config.max_pending = max_pending;
        self
    }

    #[inline]
    pub fn with_gap_detection(mut self, enabled: bool) -> Self {
        self.config.router_config.enable_gap_detection = enabled;
        self
    }

    #[inline]
    pub fn with_simd(mut self, enabled: bool) -> Self {
        self.config.parser_config.enable_simd = enabled;
        self
    }

    #[inline]
    pub fn build(self) -> DataIngestionPipeline {
        DataIngestionPipeline::new(self.config)
    }
}

impl Default for PipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_creation() {
        let pipeline = DataIngestionPipeline::new(PipelineConfig::default());
        assert!(!pipeline.is_shutdown());
    }

    #[test]
    fn test_pipeline_builder() {
        let pipeline = PipelineBuilder::new()
            .with_channel_buffer(5000)
            .with_backpressure(true, 0.75, 50_000)
            .with_gap_detection(true)
            .with_simd(false)
            .build();
        
        assert!(!pipeline.is_shutdown());
        assert_eq!(pipeline.stats().router_stats.max_pending, 50_000);
    }

    #[test]
    fn test_pipeline_shutdown() {
        let pipeline = DataIngestionPipeline::default();
        assert!(!pipeline.is_shutdown());
        
        pipeline.shutdown();
        assert!(pipeline.is_shutdown());
    }

    #[test]
    fn test_pipeline_stats_snapshot() {
        let pipeline = DataIngestionPipeline::default();
        let stats = pipeline.stats();
        
        assert_eq!(stats.total_processed(), 0);
        assert_eq!(stats.total_dropped(), 0);
        assert_eq!(stats.success_rate(), 100.0);
    }
}
