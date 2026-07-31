//! Lightweight Temporal Attention Feature Extractor
//! 
//! Implements non-LLM temporal attention using decaying exponential weights.
//! Strictly bounded to 6.5GB RAM limit for Python/Nautilus IPC backend.

use std::collections::VecDeque;

/// Configuration for temporal attention
#[derive(Debug, Clone)]
pub struct TemporalAttentionConfig {
    /// Maximum sequence length (bounded for memory)
    pub max_sequence_length: usize,
    /// Decay factor for exponential weights (0.5 to 0.99)
    pub decay_factor: f64,
    /// Number of attention heads (simplified: different time scales)
    pub num_time_scales: usize,
}

impl Default for TemporalAttentionConfig {
    fn default() -> Self {
        Self {
            max_sequence_length: 1000, // Bounded for memory efficiency
            decay_factor: 0.95,
            num_time_scales: 4,
        }
    }
}

/// Single time-scale attention weight calculator
struct TimeScaleAttention {
    /// Time constant for this scale (in ticks)
    time_constant: f64,
    /// Cached weights for recent positions
    weights: Vec<f64>,
}

impl TimeScaleAttention {
    fn new(time_constant: f64, max_len: usize) -> Self {
        let mut weights = Vec::with_capacity(max_len);
        for i in 0..max_len {
            weights.push((-i as f64 / time_constant).exp());
        }
        
        Self {
            time_constant,
            weights,
        }
    }

    #[inline]
    fn get_weight(&self, position: usize) -> f64 {
        if position < self.weights.len() {
            self.weights[position]
        } else {
            (-position as f64 / self.time_constant).exp()
        }
    }

    fn update_weights(&mut self, new_len: usize) {
        if new_len > self.weights.len() {
            for i in self.weights.len()..new_len {
                self.weights.push((-i as f64 / self.time_constant).exp());
            }
        }
    }
}

/// Temporal attention feature extractor
pub struct TemporalAttentionExtractor {
    config: TemporalAttentionConfig,
    /// Price history with bounded size
    prices: VecDeque<f64>,
    /// Volume history
    volumes: VecDeque<f64>,
    /// Returns history
    returns: VecDeque<f64>,
    /// Multiple time-scale attention mechanisms
    attention_scales: Vec<TimeScaleAttention>,
    /// Current position in sequence
    current_position: usize,
    /// Cached attention output
    cached_attention_sum: f64,
    cached_attention_weighted_sum: f64,
}

impl TemporalAttentionExtractor {
    pub fn new(config: TemporalAttentionConfig) -> Self {
        let mut attention_scales = Vec::with_capacity(config.num_time_scales);
        
        // Create multiple time scales: short, medium, long, very long
        let base_tc = config.max_sequence_length as f64 / 4.0;
        for i in 0..config.num_time_scales {
            let tc = base_tc * (2.0_f64.powi(i as i32));
            attention_scales.push(TimeScaleAttention::new(tc, config.max_sequence_length));
        }

        Self {
            config,
            prices: VecDeque::with_capacity(config.max_sequence_length),
            volumes: VecDeque::with_capacity(config.max_sequence_length),
            returns: VecDeque::with_capacity(config.max_sequence_length),
            attention_scales,
            current_position: 0,
            cached_attention_sum: 0.0,
            cached_attention_weighted_sum: 0.0,
        }
    }

    /// Add a new observation
    pub fn add_observation(&mut self, price: f64, volume: f64) {
        // Calculate return
        if let Some(&prev_price) = self.prices.back() {
            if prev_price > 0.0 {
                let ret = (price / prev_price).ln();
                self.returns.push_back(ret);
                
                // Trim if needed
                while self.returns.len() > self.config.max_sequence_length {
                    self.returns.pop_front();
                }
            } else {
                self.returns.push_back(0.0);
            }
        } else {
            self.returns.push_back(0.0);
        }

        self.prices.push_back(price);
        self.volumes.push_back(volume);
        self.current_position += 1;

        // Trim if needed
        while self.prices.len() > self.config.max_sequence_length {
            self.prices.pop_front();
            self.volumes.pop_front();
        }

        // Update attention weights
        for scale in &mut self.attention_scales {
            scale.update_weights(self.prices.len());
        }

        // Recalculate cached attention
        self.recalculate_attention();
    }

    fn recalculate_attention(&mut self) {
        let n = self.prices.len();
        if n == 0 {
            self.cached_attention_sum = 0.0;
            self.cached_attention_weighted_sum = 0.0;
            return;
        }

        // Use the longest time scale for overall attention
        let primary_scale = &self.attention_scales[self.attention_scales.len() - 1];
        
        let mut total_weight = 0.0;
        let mut weighted_sum = 0.0;

        for i in 0..n {
            let weight = primary_scale.get_weight(n - 1 - i);
            total_weight += weight;
            weighted_sum += weight * self.prices[i];
        }

        self.cached_attention_sum = total_weight;
        self.cached_attention_weighted_sum = weighted_sum / total_weight.max(1e-9);
    }

    /// Get attention-weighted average price
    pub fn attention_weighted_price(&self) -> f64 {
        self.cached_attention_weighted_sum
    }

    /// Get attention-weighted volume
    pub fn attention_weighted_volume(&self) -> f64 {
        let n = self.volumes.len();
        if n == 0 {
            return 0.0;
        }

        let primary_scale = &self.attention_scales[self.attention_scales.len() - 1];
        
        let mut total_weight = 0.0;
        let mut weighted_sum = 0.0;

        for i in 0..n {
            let weight = primary_scale.get_weight(n - 1 - i);
            total_weight += weight;
            weighted_sum += weight * self.volumes[i];
        }

        weighted_sum / total_weight.max(1e-9)
    }

    /// Get multi-scale attention features
    pub fn get_multi_scale_features(&self) -> Vec<f64> {
        let n = self.prices.len();
        if n == 0 {
            return vec![0.0; self.config.num_time_scales];
        }

        let mut features = Vec::with_capacity(self.config.num_time_scales);

        for scale in &self.attention_scales {
            let mut total_weight = 0.0;
            let mut weighted_sum = 0.0;

            for i in 0..n {
                let weight = scale.get_weight(n - 1 - i);
                total_weight += weight;
                weighted_sum += weight * self.prices[i];
            }

            features.push(weighted_sum / total_weight.max(1e-9));
        }

        features
    }

    /// Get temporal attention context vector for ML
    pub fn get_context_vector(&self) -> TemporalContext {
        let n = self.prices.len();
        
        TemporalContext {
            sequence_length: n,
            attention_price: self.attention_weighted_price(),
            attention_volume: self.attention_weighted_volume(),
            multi_scale_prices: self.get_multi_scale_features(),
            price_decay_rate: self.config.decay_factor,
            current_price: self.prices.back().copied().unwrap_or(0.0),
            avg_return: self.returns.iter().sum::<f64>() / n.max(1) as f64,
        }
    }

    /// Calculate attention between two time points
    pub fn pairwise_attention(&self, t1: usize, t2: usize) -> f64 {
        if t1 >= self.prices.len() || t2 >= self.prices.len() {
            return 0.0;
        }

        let time_diff = (t2 as i64 - t1 as i64).abs() as f64;
        let scale = &self.attention_scales[0]; // Shortest scale
        
        (-time_diff / scale.time_constant).exp()
    }

    /// Get the number of observations
    pub fn sequence_length(&self) -> usize {
        self.prices.len()
    }

    /// Check if we have enough data
    pub fn is_ready(&self) -> bool {
        self.prices.len() >= 10
    }

    /// Clear all data
    pub fn clear(&mut self) {
        self.prices.clear();
        self.volumes.clear();
        self.returns.clear();
        self.current_position = 0;
        self.cached_attention_sum = 0.0;
        self.cached_attention_weighted_sum = 0.0;
    }

    /// Get memory usage estimate in bytes
    pub fn memory_usage_bytes(&self) -> usize {
        let prices_mem = self.prices.capacity() * std::mem::size_of::<f64>();
        let volumes_mem = self.volumes.capacity() * std::mem::size_of::<f64>();
        let returns_mem = self.returns.capacity() * std::mem::size_of::<f64>();
        let attention_mem = self.attention_scales.iter()
            .map(|s| s.weights.capacity() * std::mem::size_of::<f64>())
            .sum::<usize>();
        
        prices_mem + volumes_mem + returns_mem + attention_mem
    }
}

/// Context vector for IPC serialization
#[derive(Debug, Clone)]
pub struct TemporalContext {
    pub sequence_length: usize,
    pub attention_price: f64,
    pub attention_volume: f64,
    pub multi_scale_prices: Vec<f64>,
    pub price_decay_rate: f64,
    pub current_price: f64,
    pub avg_return: f64,
}

impl TemporalContext {
    /// Serialize to bytes for IPC (simple binary format)
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        
        // Sequence length
        bytes.extend_from_slice(&self.sequence_length.to_le_bytes());
        
        // Attention price
        bytes.extend_from_slice(&self.attention_price.to_le_bytes());
        
        // Attention volume
        bytes.extend_from_slice(&self.attention_volume.to_le_bytes());
        
        // Multi-scale prices
        bytes.extend_from_slice(&(self.multi_scale_prices.len() as u32).to_le_bytes());
        for price in &self.multi_scale_prices {
            bytes.extend_from_slice(&price.to_le_bytes());
        }
        
        // Decay rate
        bytes.extend_from_slice(&self.price_decay_rate.to_le_bytes());
        
        // Current price
        bytes.extend_from_slice(&self.current_price.to_le_bytes());
        
        // Avg return
        bytes.extend_from_slice(&self.avg_return.to_le_bytes());
        
        bytes
    }

    /// Deserialize from bytes
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 40 {
            return None;
        }

        let mut offset = 0;
        
        let sequence_length = usize::from_le_bytes(
            bytes[offset..offset + 8].try_into().ok()?
        );
        offset += 8;
        
        let attention_price = f64::from_le_bytes(
            bytes[offset..offset + 8].try_into().ok()?
        );
        offset += 8;
        
        let attention_volume = f64::from_le_bytes(
            bytes[offset..offset + 8].try_into().ok()?
        );
        offset += 8;
        
        let num_scales = u32::from_le_bytes(
            bytes[offset..offset + 4].try_into().ok()?
        ) as usize;
        offset += 4;
        
        let mut multi_scale_prices = Vec::with_capacity(num_scales);
        for _ in 0..num_scales {
            let price = f64::from_le_bytes(
                bytes[offset..offset + 8].try_into().ok()?
            );
            multi_scale_prices.push(price);
            offset += 8;
        }
        
        let price_decay_rate = f64::from_le_bytes(
            bytes[offset..offset + 8].try_into().ok()?
        );
        offset += 8;
        
        let current_price = f64::from_le_bytes(
            bytes[offset..offset + 8].try_into().ok()?
        );
        offset += 8;
        
        let avg_return = f64::from_le_bytes(
            bytes[offset..offset + 8].try_into().ok()?
        );
        
        Some(Self {
            sequence_length,
            attention_price,
            attention_volume,
            multi_scale_prices,
            price_decay_rate,
            current_price,
            avg_return,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temporal_attention_basic() {
        let config = TemporalAttentionConfig {
            max_sequence_length: 100,
            decay_factor: 0.95,
            num_time_scales: 3,
        };
        let mut extractor = TemporalAttentionExtractor::new(config);

        // Add some observations with uptrend
        for i in 0..50 {
            let price = 100.0 + i as f64;
            let volume = 1000.0 + (i % 10) as f64 * 100.0;
            extractor.add_observation(price, volume);
        }

        assert!(extractor.is_ready());
        
        // Attention-weighted price should be closer to recent prices
        let attn_price = extractor.attention_weighted_price();
        let current_price = 100.0 + 49.0;
        assert!(attn_price > 100.0);
        assert!(attn_price <= current_price);
    }

    #[test]
    fn test_multi_scale_features() {
        let config = TemporalAttentionConfig::default();
        let mut extractor = TemporalAttentionExtractor::new(config);

        for i in 0..100 {
            extractor.add_observation(100.0 + i as f64, 1000.0);
        }

        let features = extractor.get_multi_scale_features();
        assert_eq!(features.len(), config.num_time_scales);
        
        // Longer time scales should have lower attention prices (older data weighted more)
        for i in 1..features.len() {
            assert!(features[i] <= features[i - 1] || (features[i] - features[i - 1]).abs() < 1.0);
        }
    }

    #[test]
    fn test_context_serialization() {
        let context = TemporalContext {
            sequence_length: 100,
            attention_price: 105.5,
            attention_volume: 1500.0,
            multi_scale_prices: vec![104.0, 103.0, 102.0],
            price_decay_rate: 0.95,
            current_price: 106.0,
            avg_return: 0.001,
        };

        let bytes = context.to_bytes();
        let restored = TemporalContext::from_bytes(&bytes).unwrap();

        assert_eq!(restored.sequence_length, context.sequence_length);
        assert!((restored.attention_price - context.attention_price).abs() < 1e-9);
    }

    #[test]
    fn test_memory_bound() {
        let config = TemporalAttentionConfig {
            max_sequence_length: 1000,
            ..Default::default()
        };
        let extractor = TemporalAttentionExtractor::new(config);
        
        // Should be well under 6.5GB even at max capacity
        let mem = extractor.memory_usage_bytes();
        assert!(mem < 6_500_000_000);
    }
}
