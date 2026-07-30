//! ML Feature Module Root
//! 
//! Pushes normalized feature vectors directly into zero-copy shared memory IPC buffer.

pub mod extractor;
pub mod normalizer;

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use crate::ml_features::extractor::{FeatureExtractor, FeatureVector, MarketSnapshot};
use crate::ml_features::normalizer::{FeatureNormalizer, NormalizedFeatures, SharedFeatureBuffer};

/// ML feature module configuration
#[derive(Debug, Clone)]
pub struct MlFeatureConfig {
    /// Max memory for feature extraction (MB)
    pub max_extractor_memory_mb: u64,
    /// Decay factor for normalization
    pub decay_factor: f64,
    /// Minimum samples before normalization
    pub min_samples: u64,
    /// Shared buffer capacity
    pub buffer_capacity: usize,
    /// Feature extraction interval (ns)
    pub extraction_interval_ns: u64,
}

impl Default for MlFeatureConfig {
    fn default() -> Self {
        Self {
            max_extractor_memory_mb: 100,
            decay_factor: 0.99,
            min_samples: 100,
            buffer_capacity: 10000,
            extraction_interval_ns: 100_000, // 100 microseconds
        }
    }
}

/// ML feature pipeline statistics
#[derive(Debug, Clone)]
pub struct FeaturePipelineStats {
    pub features_extracted: u64,
    pub features_normalized: u64,
    pub features_written_to_buffer: u64,
    pub extraction_errors: u64,
    pub normalization_skipped: u64,
    pub avg_extraction_latency_us: f64,
    pub buffer_utilization: f64,
    pub is_ready_for_inference: bool,
    pub timestamp_ns: u64,
}

/// ML feature module handle
pub struct MlFeatureModule {
    extractor: Arc<FeatureExtractor>,
    normalizer: Arc<FeatureNormalizer>,
    shared_buffer: Arc<SharedFeatureBuffer>,
    config: MlFeatureConfig,
    /// Statistics
    features_extracted: AtomicU64,
    features_normalized: AtomicU64,
    features_written: AtomicU64,
    extraction_errors: AtomicU64,
    normalization_skipped: AtomicU64,
    /// Latency tracking
    total_latency_ns: AtomicU64,
    latency_count: AtomicU64,
    /// Ready flag
    ready_for_inference: AtomicBool,
    /// Last extraction timestamp
    last_extraction_ns: AtomicU64,
}

impl MlFeatureModule {
    pub fn new(config: MlFeatureConfig) -> Self {
        let extractor = Arc::new(FeatureExtractor::new(config.max_extractor_memory_mb));
        let normalizer = Arc::new(FeatureNormalizer::new(config.decay_factor, config.min_samples));
        let shared_buffer = Arc::new(SharedFeatureBuffer::new(config.buffer_capacity));

        Self {
            extractor,
            normalizer,
            shared_buffer,
            config,
            features_extracted: AtomicU64::new(0),
            features_normalized: AtomicU64::new(0),
            features_written: AtomicU64::new(0),
            extraction_errors: AtomicU64::new(0),
            normalization_skipped: AtomicU64::new(0),
            total_latency_ns: AtomicU64::new(0),
            latency_count: AtomicU64::new(0),
            ready_for_inference: AtomicBool::new(false),
            last_extraction_ns: AtomicU64::new(0),
        }
    }

    /// Process a market snapshot through the full pipeline
    pub fn process_snapshot(&self, snapshot: MarketSnapshot) -> Option<u64> {
        let start = Instant::now();
        
        // Extract features
        let features = self.extractor.extract_features(snapshot);
        self.features_extracted.fetch_add(1, Ordering::Relaxed);
        
        // Normalize
        let normalized = match self.normalizer.process(&features) {
            Some(nf) => {
                self.features_normalized.fetch_add(1, Ordering::Relaxed);
                nf
            }
            None => {
                self.normalization_skipped.fetch_add(1, Ordering::Relaxed);
                
                // Check if we're now ready
                if self.normalizer.is_ready() {
                    self.ready_for_inference.store(true, Ordering::Relaxed);
                }
                
                return None;
            }
        };
        
        // Write to shared buffer
        match self.shared_buffer.write(&normalized) {
            Ok(index) => {
                self.features_written.fetch_add(1, Ordering::Relaxed);
                
                // Track latency
                let latency_ns = start.elapsed().as_nanos() as u64;
                self.total_latency_ns.fetch_add(latency_ns, Ordering::Relaxed);
                self.latency_count.fetch_add(1, Ordering::Relaxed);
                
                // Update last extraction time
                self.last_extraction_ns.store(snapshot.timestamp_ns, Ordering::Relaxed);
                
                Some(index)
            }
            Err(_) => {
                self.extraction_errors.fetch_add(1, Ordering::Relaxed);
                None
            }
        }
    }

    /// Get reference to the feature extractor
    pub fn extractor(&self) -> &Arc<FeatureExtractor> {
        &self.extractor
    }

    /// Get reference to the normalizer
    pub fn normalizer(&self) -> &Arc<FeatureNormalizer> {
        &self.normalizer
    }

    /// Get reference to the shared buffer
    pub fn shared_buffer(&self) -> &Arc<SharedFeatureBuffer> {
        &self.shared_buffer
    }

    /// Get pipeline statistics
    pub fn get_stats(&self) -> FeaturePipelineStats {
        let extracted = self.features_extracted.load(Ordering::Relaxed);
        let normalized = self.features_normalized.load(Ordering::Relaxed);
        let written = self.features_written.load(Ordering::Relaxed);
        let errors = self.extraction_errors.load(Ordering::Relaxed);
        let skipped = self.normalization_skipped.load(Ordering::Relaxed);
        
        let avg_latency = {
            let total = self.total_latency_ns.load(Ordering::Relaxed);
            let count = self.latency_count.load(Ordering::Relaxed);
            if count > 0 {
                (total / count) as f64 / 1000.0 // Convert to microseconds
            } else {
                0.0
            }
        };
        
        let buffer_util = {
            let write_idx = self.shared_buffer.write_index();
            let capacity = self.shared_buffer.capacity() as u64;
            if capacity > 0 {
                (write_idx % capacity) as f64 / capacity as f64
            } else {
                0.0
            }
        };
        
        FeaturePipelineStats {
            features_extracted: extracted,
            features_normalized: normalized,
            features_written_to_buffer: written,
            extraction_errors: errors,
            normalization_skipped: skipped,
            avg_extraction_latency_us: avg_latency,
            buffer_utilization: buffer_util,
            is_ready_for_inference: self.ready_for_inference.load(Ordering::Relaxed),
            timestamp_ns: timestamp_ns(),
        }
    }

    /// Check if ready for inference
    pub fn is_ready(&self) -> bool {
        self.ready_for_inference.load(Ordering::Relaxed)
    }

    /// Get latest N normalized features
    pub fn get_latest_features(&self, count: usize) -> Vec<NormalizedFeatures> {
        self.shared_buffer.get_latest(count)
    }

    /// Get shared buffer as bytes for zero-copy IPC
    pub fn get_buffer_bytes(&self) -> &[u8] {
        self.shared_buffer.as_bytes()
    }

    /// Get buffer write index
    pub fn get_write_index(&self) -> u64 {
        self.shared_buffer.write_index()
    }

    /// Reset the pipeline
    pub fn reset(&self) {
        self.extractor.clear();
        self.normalizer.reset();
        self.features_extracted.store(0, Ordering::Relaxed);
        self.features_normalized.store(0, Ordering::Relaxed);
        self.features_written.store(0, Ordering::Relaxed);
        self.extraction_errors.store(0, Ordering::Relaxed);
        self.normalization_skipped.store(0, Ordering::Relaxed);
        self.total_latency_ns.store(0, Ordering::Relaxed);
        self.latency_count.store(0, Ordering::Relaxed);
        self.ready_for_inference.store(false, Ordering::Relaxed);
    }

    /// Get memory usage estimate
    pub fn memory_usage_mb(&self) -> f64 {
        let extractor_mem = self.config.max_extractor_memory_mb as f64;
        let buffer_mem = (self.shared_buffer.capacity() * std::mem::size_of::<NormalizedFeatures>()) as f64 / (1024.0 * 1024.0);
        extractor_mem + buffer_mem
    }
}

/// Get current timestamp in nanoseconds
#[inline]
fn timestamp_ns() -> u64 {
    Instant::now()
        .duration_since(Instant::now() - Duration::from_secs(1))
        .as_nanos() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ml_feature_module_basic() {
        let config = MlFeatureConfig::default();
        let module = MlFeatureModule::new(config);

        assert!(!module.is_ready());

        // Process some snapshots to warm up
        for i in 0..150 {
            let snapshot = MarketSnapshot {
                symbol: "BTCUSD".to_string(),
                timestamp_ns: 1000000000 + i * 100000,
                bid_price: 49999.5 + (i as f64 * 0.1),
                ask_price: 50000.5 + (i as f64 * 0.1),
                bid_size: 100,
                ask_size: 150,
                last_trade_price: 50000.0 + (i as f64 * 0.1),
                last_trade_size: 10,
                last_trade_aggressor: i % 2 == 0,
                total_bid_depth: 5000,
                total_ask_depth: 6000,
            };

            module.process_snapshot(snapshot);
        }

        // Should be ready after warm-up
        assert!(module.is_ready());

        let stats = module.get_stats();
        assert!(stats.features_extracted >= 150);
        assert!(stats.features_written_to_buffer >= 50); // After warm-up
        assert!(stats.avg_extraction_latency_us > 0.0);
    }

    #[test]
    fn test_shared_memory_ipc() {
        let config = MlFeatureConfig::default();
        let module = MlFeatureModule::new(config);

        // Get buffer bytes for IPC
        let bytes = module.get_buffer_bytes();
        assert!(!bytes.is_empty());

        // Verify buffer size
        let expected_size = config.buffer_capacity * std::mem::size_of::<NormalizedFeatures>();
        assert_eq!(bytes.len(), expected_size);
    }

    #[test]
    fn test_memory_tracking() {
        let config = MlFeatureConfig::default();
        let module = MlFeatureModule::new(config);

        let mem = module.memory_usage_mb();
        assert!(mem > 0.0);
        assert!(mem < 500.0); // Should be well under limit
    }
}
