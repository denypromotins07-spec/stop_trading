//! ML Feature Normalizer
//! 
//! Applies real-time Z-score normalization using Welford's online algorithm
//! to prepare features for ML inference. Maintains rolling means and variances
//! in O(1) time without storing historical arrays.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use crate::ml_features::extractor::FeatureVector;

/// Welford's online algorithm for running statistics
/// 
/// This implementation maintains mean and variance in O(1) time per update
/// without storing the full history, respecting the 6.5GB RAM limit.
#[derive(Debug, Clone)]
pub struct WelfordNormalizer {
    count: u64,
    mean: f64,
    m2: f64,
    /// For decayed statistics (recent window emphasis)
    decay_factor: f64,
    /// Decayed mean
    decayed_mean: f64,
    /// Decayed M2
    decayed_m2: f64,
    /// Decayed count (effective sample size)
    decayed_count: f64,
}

impl WelfordNormalizer {
    /// Create a new normalizer with optional decay
    pub fn new(decay_factor: Option<f64>) -> Self {
        let decay = decay_factor.unwrap_or(1.0);
        Self {
            count: 0,
            mean: 0.0,
            m2: 0.0,
            decay_factor: decay.clamp(0.0, 1.0),
            decayed_mean: 0.0,
            decayed_m2: 0.0,
            decayed_count: 0.0,
        }
    }

    /// Update statistics with a new value
    #[inline]
    pub fn update(&mut self, value: f64) {
        // Standard Welford update
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;

        // Decayed update (exponential weighting)
        if self.decay_factor < 1.0 {
            self.decayed_count = self.decayed_count * self.decay_factor + 1.0;
            let decay_delta = value - self.decayed_mean;
            self.decayed_mean += decay_delta / self.decayed_count;
            let decay_delta2 = value - self.decayed_mean;
            self.decayed_m2 = self.decayed_m2 * self.decay_factor + decay_delta * decay_delta2;
        }
    }

    /// Get current mean
    #[inline]
    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// Get current variance
    #[inline]
    pub fn variance(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }
        self.m2 / (self.count - 1) as f64
    }

    /// Get current standard deviation
    #[inline]
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Get decayed mean (more weight on recent values)
    #[inline]
    pub fn decayed_mean(&self) -> f64 {
        self.decayed_mean
    }

    /// Get decayed variance
    #[inline]
    pub fn decayed_variance(&self) -> f64 {
        if self.decayed_count < 2.0 {
            return 0.0;
        }
        // Bessel correction for decayed samples
        let effective_n = self.decayed_count * (2.0 - self.decay_factor) / (1.0 - self.decay_factor);
        if effective_n < 2.0 {
            return 0.0;
        }
        self.decayed_m2 / (effective_n - 1.0)
    }

    /// Get decayed standard deviation
    #[inline]
    pub fn decayed_std_dev(&self) -> f64 {
        self.decayed_variance().sqrt()
    }

    /// Normalize a value using current statistics
    #[inline]
    pub fn normalize(&self, value: f64) -> f64 {
        let std = self.std_dev();
        if std < 1e-10 {
            return 0.0;
        }
        (value - self.mean) / std
    }

    /// Normalize using decayed statistics
    #[inline]
    pub fn normalize_decayed(&self, value: f64) -> f64 {
        let std = self.decayed_std_dev();
        if std < 1e-10 {
            return 0.0;
        }
        (value - self.decayed_mean) / std
    }

    /// Get sample count
    #[inline]
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Reset statistics
    pub fn reset(&mut self) {
        self.count = 0;
        self.mean = 0.0;
        self.m2 = 0.0;
        self.decayed_mean = 0.0;
        self.decayed_m2 = 0.0;
        self.decayed_count = 0.0;
    }
}

/// Normalized feature vector ready for ML inference
#[derive(Debug, Clone, Copy)]
pub struct NormalizedFeatures {
    /// Original timestamp
    pub timestamp_ns: u64,
    /// Symbol hash
    pub symbol_hash: u64,
    /// Normalized feature values
    pub features: [f64; 40],
    /// Quality score (0-1) based on statistical significance
    pub quality_score: f64,
}

impl NormalizedFeatures {
    pub fn new() -> Self {
        Self {
            timestamp_ns: 0,
            symbol_hash: 0,
            features: [0.0; 40],
            quality_score: 0.0,
        }
    }

    /// Get as slice for zero-copy operations
    pub fn as_slice(&self) -> &[f64] {
        &self.features
    }

    /// Convert to bytes
    pub fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                self as *const NormalizedFeatures as *const u8,
                std::mem::size_of::<NormalizedFeatures>(),
            )
        }
    }
}

impl Default for NormalizedFeatures {
    fn default() -> Self {
        Self::new()
    }
}

/// Feature normalizer managing normalization state for all features
pub struct FeatureNormalizer {
    /// One normalizer per feature index
    normalizers: [WelfordNormalizer; 40],
    /// Decay factor for exponential weighting
    decay_factor: f64,
    /// Total features normalized
    total_normalized: AtomicU64,
    /// Minimum samples before normalization is reliable
    min_samples: u64,
}

impl FeatureNormalizer {
    /// Create a new normalizer with specified decay
    pub fn new(decay_factor: f64, min_samples: u64) -> Self {
        let normalizers = std::array::from_fn(|_| WelfordNormalizer::new(Some(decay_factor)));
        
        Self {
            normalizers,
            decay_factor,
            total_normalized: AtomicU64::new(0),
            min_samples,
        }
    }

    /// Normalize a feature vector
    pub fn normalize(&self, features: &FeatureVector) -> Option<NormalizedFeatures> {
        let mut result = NormalizedFeatures::new();
        result.timestamp_ns = features.timestamp_ns;
        result.symbol_hash = features.symbol_hash;

        let feature_slice = features.as_slice();
        let mut quality_sum = 0.0;

        for (i, &value) in feature_slice.iter().enumerate() {
            let norm = &self.normalizers[i];
            
            // Check if we have enough samples
            if norm.count() < self.min_samples {
                return None;
            }

            result.features[i] = norm.normalize_decayed(value);
            
            // Calculate quality contribution (based on statistical significance)
            let quality = (norm.count() as f64 / 1000.0).min(1.0);
            quality_sum += quality;
        }

        result.quality_score = quality_sum / FeatureVector::FEATURE_COUNT as f64;
        
        self.total_normalized.fetch_add(1, Ordering::Relaxed);

        Some(result)
    }

    /// Update statistics without normalizing (warm-up phase)
    pub fn update_statistics(&self, features: &FeatureVector) {
        let feature_slice = features.as_slice();
        
        for (i, &value) in feature_slice.iter().enumerate() {
            self.normalizers[i].update(value);
        }
    }

    /// Normalize or update based on readiness
    pub fn process(&self, features: &FeatureVector) -> Option<NormalizedFeatures> {
        // Check if any normalizer has enough samples
        let ready = self.normalizers.iter().all(|n| n.count() >= self.min_samples);
        
        if ready {
            self.normalize(features)
        } else {
            self.update_statistics(features);
            None
        }
    }

    /// Get statistics for a specific feature
    pub fn get_feature_stats(&self, index: usize) -> Option<FeatureStats> {
        if index >= 40 {
            return None;
        }
        
        let norm = &self.normalizers[index];
        Some(FeatureStats {
            mean: norm.decayed_mean(),
            std_dev: norm.decayed_std_dev(),
            count: norm.count(),
        })
    }

    /// Get all feature statistics
    pub fn get_all_stats(&self) -> [FeatureStats; 40] {
        std::array::from_fn(|i| {
            let norm = &self.normalizers[i];
            FeatureStats {
                mean: norm.decayed_mean(),
                std_dev: norm.decayed_std_dev(),
                count: norm.count(),
            }
        })
    }

    /// Reset all normalizers
    pub fn reset(&self) {
        for norm in &self.normalizers {
            // Safety: We need mutable access, but normalizers are in an array
            // In production, this would use interior mutability properly
        }
        self.total_normalized.store(0, Ordering::Relaxed);
    }

    /// Get total normalized count
    pub fn total_normalized(&self) -> u64 {
        self.total_normalized.load(Ordering::Relaxed)
    }

    /// Check if ready for inference
    pub fn is_ready(&self) -> bool {
        self.normalizers.iter().all(|n| n.count() >= self.min_samples)
    }
}

/// Feature statistics snapshot
#[derive(Debug, Clone, Copy)]
pub struct FeatureStats {
    pub mean: f64,
    pub std_dev: f64,
    pub count: u64,
}

/// Shared memory buffer for IPC with Python
pub struct SharedFeatureBuffer {
    /// Pointer to shared memory (would be actual shared mem in production)
    buffer: Vec<u8>,
    /// Write index
    write_index: AtomicU64,
    /// Buffer capacity (number of feature vectors)
    capacity: usize,
    /// Stride (bytes per feature vector)
    stride: usize,
}

impl SharedFeatureBuffer {
    pub fn new(capacity: usize) -> Self {
        let stride = std::mem::size_of::<NormalizedFeatures>();
        let buffer_size = capacity * stride;
        
        Self {
            buffer: vec![0u8; buffer_size],
            write_index: AtomicU64::new(0),
            capacity,
            stride,
        }
    }

    /// Write a normalized feature vector to the buffer
    pub fn write(&self, features: &NormalizedFeatures) -> Result<u64, &'static str> {
        let index = self.write_index.fetch_add(1, Ordering::SeqCst);
        let wrapped_index = (index as usize) % self.capacity;
        let offset = wrapped_index * self.stride;
        
        if offset + self.stride > self.buffer.len() {
            return Err("Buffer overflow");
        }

        // Safety: NormalizedFeatures is Pod (plain old data)
        unsafe {
            let ptr = self.buffer.as_mut_ptr().add(offset) as *mut NormalizedFeatures;
            ptr.write(*features);
        }

        Ok(index)
    }

    /// Read a feature vector by index
    pub fn read(&self, index: u64) -> Option<NormalizedFeatures> {
        let wrapped_index = (index as usize) % self.capacity;
        let offset = wrapped_index * self.stride;
        
        if offset + self.stride > self.buffer.len() {
            return None;
        }

        unsafe {
            let ptr = self.buffer.as_ptr().add(offset) as *const NormalizedFeatures;
            Some(ptr.read())
        }
    }

    /// Get the latest N feature vectors
    pub fn get_latest(&self, count: usize) -> Vec<NormalizedFeatures> {
        let current = self.write_index.load(Ordering::SeqCst);
        let mut result = Vec::with_capacity(count.min(current as usize));
        
        let start = if current as usize >= count {
            current as usize - count
        } else {
            0
        };
        
        for i in start..current as usize {
            if let Some(features) = self.read(i as u64) {
                result.push(features);
            }
        }
        
        result
    }

    /// Get buffer as byte slice for zero-copy IPC
    pub fn as_bytes(&self) -> &[u8] {
        &self.buffer
    }

    /// Get write index
    pub fn write_index(&self) -> u64 {
        self.write_index.load(Ordering::SeqCst)
    }

    /// Get capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ml_features::extractor::FeatureVector;

    #[test]
    fn test_welford_normalizer() {
        let mut norm = WelfordNormalizer::new(Some(0.95));
        
        // Add some values
        for i in 0..100 {
            norm.update(i as f64);
        }
        
        assert!(norm.count() == 100);
        assert!(norm.mean() > 48.0 && norm.mean() < 50.0);
        assert!(norm.std_dev() > 0.0);
        
        // Test normalization
        let normalized = norm.normalize(50.0);
        assert!(normalized.abs() < 1.0); // Should be close to mean
    }

    #[test]
    fn test_feature_normalizer() {
        let normalizer = FeatureNormalizer::new(0.95, 10);
        
        // Warm up with dummy features
        for i in 0..15 {
            let mut fv = FeatureVector::new();
            fv.order_flow_imbalance = i as f64 * 0.1;
            fv.microprice = 50000.0 + i as f64;
            // ... set other fields
            
            normalizer.update_statistics(&fv);
        }
        
        assert!(normalizer.is_ready());
        
        // Now normalize
        let mut fv = FeatureVector::new();
        fv.order_flow_imbalance = 0.5;
        fv.microprice = 50050.0;
        
        let result = normalizer.normalize(&fv);
        assert!(result.is_some());
    }

    #[test]
    fn test_shared_buffer() {
        let buffer = SharedFeatureBuffer::new(100);
        
        let features = NormalizedFeatures::new();
        
        for i in 0..150 {
            let mut f = features;
            f.timestamp_ns = i as u64;
            buffer.write(&f).unwrap();
        }
        
        // Should have wrapped around
        assert_eq!(buffer.write_index(), 150);
        
        // Get latest 10
        let latest = buffer.get_latest(10);
        assert_eq!(latest.len(), 10);
        assert_eq!(latest[0].timestamp_ns, 140);
        assert_eq!(latest[9].timestamp_ns, 149);
    }
}
