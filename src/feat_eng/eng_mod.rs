//! Feature Engineering Module Root
//! 
//! Serializes advanced cross-sectional features into the shared memory buffer.

pub mod cross_sectional;
pub mod temporal;

use cross_sectional::{CrossSectionalRanker, CrossSectionalConfig, FeatureVector, AssetId};
use temporal::{TemporalAttentionExtractor, TemporalAttentionConfig, TemporalContext};

/// Configuration for the feature engineering module
#[derive(Debug, Clone)]
pub struct FeatureEngineConfig {
    pub cross_sectional_config: CrossSectionalConfig,
    pub temporal_config: TemporalAttentionConfig,
    /// Maximum number of assets to track
    pub max_assets: usize,
    /// Enable IPC serialization
    pub enable_ipc_serialization: bool,
}

impl Default for FeatureEngineConfig {
    fn default() -> Self {
        Self {
            cross_sectional_config: CrossSectionalConfig::default(),
            temporal_config: TemporalAttentionConfig::default(),
            max_assets: 100,
            enable_ipc_serialization: true,
        }
    }
}

/// Combined feature set for a single asset
#[derive(Debug, Clone)]
pub struct AssetFeatures {
    pub asset_id: u32,
    /// Cross-sectional momentum rank (0 to 1)
    pub momentum_rank: f64,
    /// Cross-sectional mean-reversion rank (0 to 1)
    pub mr_rank: f64,
    /// Momentum z-score
    pub momentum_z: f64,
    /// Mean-reversion z-score
    pub mr_z: f64,
    /// Volatility
    pub volatility: f64,
    /// Recent return
    pub recent_return: f64,
    /// Temporal attention price
    pub attention_price: f64,
    /// Temporal attention volume
    pub attention_volume: f64,
    /// Multi-scale attention features
    pub multi_scale_features: Vec<f64>,
}

impl AssetFeatures {
    pub fn new(asset_id: u32) -> Self {
        Self {
            asset_id,
            momentum_rank: 0.0,
            mr_rank: 0.0,
            momentum_z: 0.0,
            mr_z: 0.0,
            volatility: 0.0,
            recent_return: 0.0,
            attention_price: 0.0,
            attention_volume: 0.0,
            multi_scale_features: Vec::new(),
        }
    }

    /// Convert to flat vector for ML consumption
    pub fn to_vector(&self) -> Vec<f64> {
        let mut vec = vec![
            self.momentum_rank,
            self.mr_rank,
            self.momentum_z,
            self.mr_z,
            self.volatility,
            self.recent_return,
            self.attention_price,
            self.attention_volume,
        ];
        vec.extend(&self.multi_scale_features);
        vec
    }

    /// Get feature names
    pub fn feature_names() -> Vec<&'static str> {
        vec![
            "momentum_rank",
            "mr_rank",
            "momentum_z",
            "mr_z",
            "volatility",
            "recent_return",
            "attention_price",
            "attention_volume",
            "multi_scale_0",
            "multi_scale_1",
            "multi_scale_2",
            "multi_scale_3",
        ]
    }
}

/// Main feature engineering engine
pub struct FeatureEngine {
    config: FeatureEngineConfig,
    cross_sectional_ranker: CrossSectionalRanker,
    temporal_extractors: Vec<TemporalAttentionExtractor>, // One per asset
    asset_mapping: Vec<u32>, // Map index to asset_id
}

impl FeatureEngine {
    pub fn new(config: FeatureEngineConfig) -> Self {
        Self {
            cross_sectional_ranker: CrossSectionalRanker::new(
                config.cross_sectional_config.clone()
            ),
            temporal_extractors: Vec::with_capacity(config.max_assets),
            asset_mapping: Vec::with_capacity(config.max_assets),
            config,
        }
    }

    /// Register a new asset for tracking
    pub fn register_asset(&mut self, asset_id: u32) -> Option<usize> {
        if self.asset_mapping.len() >= self.config.max_assets {
            return None;
        }
        
        if self.asset_mapping.contains(&asset_id) {
            return None; // Already registered
        }

        let index = self.asset_mapping.len();
        self.asset_mapping.push(asset_id);
        self.temporal_extractors.push(
            TemporalAttentionExtractor::new(self.config.temporal_config.clone())
        );
        
        Some(index)
    }

    /// Update with new tick data
    pub fn update_tick(&mut self, asset_id: u32, price: f64, volume: f64) {
        // Update cross-sectional ranker
        self.cross_sectional_ranker.update_price(AssetId(asset_id), price);

        // Find and update temporal extractor
        if let Some(index) = self.asset_mapping.iter().position(|&id| id == asset_id) {
            if let Some(extractor) = self.temporal_extractors.get_mut(index) {
                extractor.add_observation(price, volume);
            }
        }
    }

    /// Get combined features for an asset
    pub fn get_features(&self, asset_id: u32) -> Option<AssetFeatures> {
        let index = self.asset_mapping.iter().position(|&id| id == asset_id)?;
        
        let mut features = AssetFeatures::new(asset_id);

        // Cross-sectional features
        if let Some(mom_rank) = self.cross_sectional_ranker.get_momentum_rank(AssetId(asset_id)) {
            features.momentum_rank = mom_rank;
        }
        if let Some(mr_rank) = self.cross_sectional_ranker.get_mr_rank(AssetId(asset_id)) {
            features.mr_rank = mr_rank;
        }

        // Get detailed cross-sectional features
        if let Some(cs_extractor) = cross_sectional::FeatureExtractor::new(&self.cross_sectional_ranker)
            .extract_features(AssetId(asset_id))
        {
            features.momentum_z = cs_extractor.momentum_z;
            features.mr_z = cs_extractor.mr_z;
            features.volatility = cs_extractor.volatility;
            features.recent_return = cs_extractor.recent_return;
        }

        // Temporal features
        if let Some(extractor) = self.temporal_extractors.get(index) {
            features.attention_price = extractor.attention_weighted_price();
            features.attention_volume = extractor.attention_weighted_volume();
            features.multi_scale_features = extractor.get_multi_scale_features();
        }

        Some(features)
    }

    /// Get features for all assets
    pub fn get_all_features(&self) -> Vec<AssetFeatures> {
        self.asset_mapping.iter()
            .filter_map(|&id| self.get_features(id))
            .collect()
    }

    /// Serialize features to shared memory buffer format
    pub fn serialize_to_buffer(&self) -> Vec<u8> {
        if !self.config.enable_ipc_serialization {
            return Vec::new();
        }

        let features = self.get_all_features();
        let mut buffer = Vec::new();

        // Header: number of assets
        buffer.extend_from_slice(&(features.len() as u32).to_le_bytes());

        // Header: number of features per asset
        let num_features = AssetFeatures::feature_names().len() as u32;
        buffer.extend_from_slice(&num_features.to_le_bytes());

        // Each asset's features
        for feature in &features {
            let vec = feature.to_vector();
            // Pad or truncate to expected size
            for i in 0..num_features as usize {
                let val = vec.get(i).copied().unwrap_or(0.0);
                buffer.extend_from_slice(&val.to_le_bytes());
            }
        }

        buffer
    }

    /// Deserialize features from shared memory buffer
    pub fn deserialize_from_buffer(buffer: &[u8]) -> Option<Vec<AssetFeatures>> {
        if buffer.len() < 8 {
            return None;
        }

        let num_assets = u32::from_le_bytes(buffer[0..4].try_into().ok()?) as usize;
        let num_features = u32::from_le_bytes(buffer[4..8].try_into().ok()?) as usize;

        let expected_len = 8 + num_assets * num_features * 8;
        if buffer.len() < expected_len {
            return None;
        }

        let mut features = Vec::with_capacity(num_assets);
        let mut offset = 8;

        for _ in 0..num_assets {
            let mut values = Vec::with_capacity(num_features);
            for _ in 0..num_features {
                let val = f64::from_le_bytes(
                    buffer[offset..offset + 8].try_into().ok()?
                );
                values.push(val);
                offset += 8;
            }

            if values.is_empty() {
                continue;
            }

            let mut feature = AssetFeatures::new(0);
            feature.momentum_rank = values.get(0).copied().unwrap_or(0.0);
            feature.mr_rank = values.get(1).copied().unwrap_or(0.0);
            feature.momentum_z = values.get(2).copied().unwrap_or(0.0);
            feature.mr_z = values.get(3).copied().unwrap_or(0.0);
            feature.volatility = values.get(4).copied().unwrap_or(0.0);
            feature.recent_return = values.get(5).copied().unwrap_or(0.0);
            feature.attention_price = values.get(6).copied().unwrap_or(0.0);
            feature.attention_volume = values.get(7).copied().unwrap_or(0.0);
            feature.multi_scale_features = values[8..].to_vec();

            features.push(feature);
        }

        Some(features)
    }

    /// Get the cross-sectional ranker for direct access
    pub fn cross_sectional_ranker(&self) -> &CrossSectionalRanker {
        &self.cross_sectional_ranker
    }

    /// Check if engine is ready
    pub fn is_ready(&self) -> bool {
        self.cross_sectional_ranker.is_ready()
            && self.temporal_extractors.iter().all(|e| e.is_ready())
    }

    /// Get number of tracked assets
    pub fn asset_count(&self) -> usize {
        self.asset_mapping.len()
    }

    /// Clear all data
    pub fn clear(&mut self) {
        self.cross_sectional_ranker.clear();
        for extractor in &mut self.temporal_extractors {
            extractor.clear();
        }
    }

    /// Estimate total memory usage
    pub fn memory_usage_bytes(&self) -> usize {
        let cs_mem = self.cross_sectional_ranker.asset_count() * std::mem::size_of::<cross_sectional::RollingStats>();
        let temporal_mem: usize = self.temporal_extractors.iter()
            .map(|e| e.memory_usage_bytes())
            .sum();
        
        cs_mem + temporal_mem
    }
}

// Helper trait implementation for FeatureExtractor reference
impl<'a> From<&'a CrossSectionalRanker> for cross_sectional::FeatureExtractor<'a> {
    fn from(ranker: &'a CrossSectionalRanker) -> Self {
        cross_sectional::FeatureExtractor::new(ranker)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_engine_basic() {
        let config = FeatureEngineConfig {
            cross_sectional_config: CrossSectionalConfig {
                momentum_lookback: 20,
                mean_reversion_lookback: 20,
                min_assets: 3,
            },
            temporal_config: TemporalAttentionConfig {
                max_sequence_length: 100,
                ..Default::default()
            },
            ..Default::default()
        };

        let mut engine = FeatureEngine::new(config);

        // Register assets
        engine.register_asset(1);
        engine.register_asset(2);
        engine.register_asset(3);

        // Add some ticks
        for i in 0..50 {
            engine.update_tick(1, 100.0 + i as f64, 1000.0);
            engine.update_tick(2, 100.0 - i as f64 * 0.5, 1500.0);
            engine.update_tick(3, 100.0 + (i % 10) as f64, 1200.0);
        }

        assert!(engine.is_ready());
        assert_eq!(engine.asset_count(), 3);

        // Get features for asset 1
        let features = engine.get_features(1);
        assert!(features.is_some());
        let f = features.unwrap();
        assert!(f.momentum_rank > 0.5); // Should have high momentum
    }

    #[test]
    fn test_serialization() {
        let config = FeatureEngineConfig::default();
        let mut engine = FeatureEngine::new(config);

        engine.register_asset(1);
        engine.register_asset(2);

        for i in 0..30 {
            engine.update_tick(1, 100.0 + i as f64, 1000.0);
            engine.update_tick(2, 95.0 + i as f64, 1500.0);
        }

        let buffer = engine.serialize_to_buffer();
        assert!(!buffer.is_empty());

        let restored = FeatureEngine::deserialize_from_buffer(&buffer);
        assert!(restored.is_some());
        let features = restored.unwrap();
        assert_eq!(features.len(), 2);
    }

    #[test]
    fn test_memory_bound() {
        let config = FeatureEngineConfig {
            max_assets: 50,
            ..Default::default()
        };
        let mut engine = FeatureEngine::new(config);

        for id in 0..50 {
            engine.register_asset(id);
            for i in 0..100 {
                engine.update_tick(id, 100.0 + i as f64, 1000.0);
            }
        }

        let mem = engine.memory_usage_bytes();
        // Should be well under 6.5GB
        assert!(mem < 6_500_000_000);
    }
}
