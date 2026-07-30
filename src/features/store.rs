//! Real-Time Feature Store
//! 
//! Lock-free concurrent hash map for storing normalized technical, order flow,
//! and on-chain metrics required for ML models. Uses dashmap for concurrent access.

use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    sync::{
        atomic::{AtomicU64, AtomicUsize, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use dashmap::DashMap;

/// Cache-line padding constant
const CACHE_LINE_SIZE: usize = 64;

/// Maximum number of features per symbol
pub const MAX_FEATURES_PER_SYMBOL: usize = 2048;

/// Feature types supported by the store
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FeatureType {
    Technical = 0,
    OrderFlow = 1,
    OnChain = 2,
    Sentiment = 3,
    Volatility = 4,
    Momentum = 5,
}

impl FeatureType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Technical => "technical",
            Self::OrderFlow => "order_flow",
            Self::OnChain => "on_chain",
            Self::Sentiment => "sentiment",
            Self::Volatility => "volatility",
            Self::Momentum => "momentum",
        }
    }
}

/// Feature entry with metadata
#[repr(C, align(64))]
#[derive(Clone)]
pub struct FeatureEntry {
    /// Feature value
    pub value: f64,
    /// Normalized value (-1 to 1)
    pub normalized: f32,
    /// Feature type
    pub feature_type: FeatureType,
    /// Timestamp in nanoseconds
    pub timestamp_ns: u64,
    /// Sequence number for ordering
    pub sequence: u64,
    /// Quality score (0-1)
    pub quality: f32,
    /// Padding to cache line
    _padding: [u8; CACHE_LINE_SIZE - 8 - 4 - 1 - 8 - 8 - 4 - 1],
}

impl Default for FeatureEntry {
    fn default() -> Self {
        Self {
            value: 0.0,
            normalized: 0.0,
            feature_type: FeatureType::Technical,
            timestamp_ns: 0,
            sequence: 0,
            quality: 1.0,
            _padding: [0u8; CACHE_LINE_SIZE - 8 - 4 - 1 - 8 - 8 - 4 - 1],
        }
    }
}

/// Key for feature lookup
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FeatureKey {
    pub symbol: String,
    pub feature_name: String,
    pub feature_type: FeatureType,
}

impl FeatureKey {
    pub fn new(symbol: &str, feature_name: &str, feature_type: FeatureType) -> Self {
        Self {
            symbol: symbol.to_string(),
            feature_name: feature_name.to_string(),
            feature_type,
        }
    }

    /// Generate hash for the key
    pub fn hash(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.symbol.hash(&mut hasher);
        self.feature_name.hash(&mut hasher);
        self.feature_type.hash(&mut hasher);
        hasher.finish()
    }
}

/// Lock-free feature store using DashMap
pub struct FeatureStore {
    /// Main storage: symbol -> feature_name -> FeatureEntry
    store: DashMap<String, DashMap<String, FeatureEntry>>,
    /// Total feature count
    feature_count: AtomicUsize,
    /// Last update timestamp
    last_update_ns: AtomicU64,
    /// Update counter for sequence generation
    update_counter: AtomicU64,
    /// Symbols registered
    symbols: DashMap<String, SymbolFeatureSet>,
}

unsafe impl Send for FeatureStore {}
unsafe impl Sync for FeatureStore {}

/// Feature set for a single symbol
#[derive(Default)]
pub struct SymbolFeatureSet {
    pub feature_names: Vec<String>,
    pub last_update_ns: u64,
    pub feature_count: usize,
}

impl FeatureStore {
    /// Create a new feature store
    pub fn new() -> Self {
        Self {
            store: DashMap::new(),
            feature_count: AtomicUsize::new(0),
            last_update_ns: AtomicU64::new(0),
            update_counter: AtomicU64::new(0),
            symbols: DashMap::new(),
        }
    }

    /// Register a symbol for feature tracking
    pub fn register_symbol(&self, symbol: &str) {
        if !self.symbols.contains_key(symbol) {
            self.symbols.insert(
                symbol.to_string(),
                SymbolFeatureSet::default(),
            );
        }
    }

    /// Set a feature value
    pub fn set_feature(
        &self,
        symbol: &str,
        feature_name: &str,
        feature_type: FeatureType,
        value: f64,
        normalized: f32,
    ) {
        let timestamp_ns = get_timestamp_ns();
        let sequence = self.update_counter.fetch_add(1, Ordering::Relaxed);

        let entry = FeatureEntry {
            value,
            normalized,
            feature_type,
            timestamp_ns,
            sequence,
            quality: 1.0,
            _padding: [0u8; CACHE_LINE_SIZE - 8 - 4 - 1 - 8 - 8 - 4 - 1],
        };

        // Get or create symbol map
        let symbol_map = self.store.entry(symbol.to_string()).or_insert_with(DashMap::new);
        symbol_map.insert(feature_name.to_string(), entry);

        // Update symbol feature set
        if let Some(mut sym_set) = self.symbols.get_mut(symbol) {
            if !sym_set.feature_names.contains(&feature_name.to_string()) {
                sym_set.feature_names.push(feature_name.to_string());
            }
            sym_set.last_update_ns = timestamp_ns;
            sym_set.feature_count = symbol_map.len();
        }

        self.feature_count.fetch_add(1, Ordering::Relaxed);
        self.last_update_ns.store(timestamp_ns, Ordering::Release);
    }

    /// Get a feature value
    pub fn get_feature(&self, symbol: &str, feature_name: &str) -> Option<FeatureEntry> {
        if let Some(symbol_map) = self.store.get(symbol) {
            symbol_map.get(feature_name).map(|e| e.clone())
        } else {
            None
        }
    }

    /// Get normalized feature value
    pub fn get_normalized(&self, symbol: &str, feature_name: &str) -> Option<f32> {
        self.get_feature(symbol, feature_name).map(|e| e.normalized)
    }

    /// Get all features for a symbol as a vector
    pub fn get_features_vector(&self, symbol: &str, feature_names: &[&str]) -> Vec<f32> {
        let mut features = Vec::with_capacity(feature_names.len());
        
        if let Some(symbol_map) = self.store.get(symbol) {
            for name in feature_names {
                if let Some(entry) = symbol_map.get(*name) {
                    features.push(entry.normalized);
                } else {
                    features.push(0.0); // Default for missing features
                }
            }
        } else {
            features.resize(feature_names.len(), 0.0);
        }

        features
    }

    /// Get feature count for a symbol
    pub fn get_symbol_feature_count(&self, symbol: &str) -> usize {
        if let Some(symbol_map) = self.store.get(symbol) {
            symbol_map.len()
        } else {
            0
        }
    }

    /// Get total feature count
    pub fn total_features(&self) -> usize {
        self.feature_count.load(Ordering::Relaxed)
    }

    /// Get last update timestamp
    pub fn last_update_ns(&self) -> u64 {
        self.last_update_ns.load(Ordering::Relaxed)
    }

    /// Get all registered symbols
    pub fn get_symbols(&self) -> Vec<String> {
        self.symbols.iter().map(|s| s.key().clone()).collect()
    }

    /// Remove stale features older than max_age_ns
    pub fn remove_stale_features(&self, max_age_ns: u64) -> usize {
        let now = get_timestamp_ns();
        let mut removed = 0;

        for symbol_entry in self.store.iter() {
            let symbol = symbol_entry.key().clone();
            let symbol_map = symbol_entry.value();
            
            let mut keys_to_remove = Vec::new();
            
            for feature_entry in symbol_map.iter() {
                if now - feature_entry.value().timestamp_ns > max_age_ns {
                    keys_to_remove.push(feature_entry.key().clone());
                }
            }

            for key in keys_to_remove {
                symbol_map.remove(&key);
                removed += 1;
            }
        }

        self.feature_count.fetch_sub(removed, Ordering::Relaxed);
        removed
    }

    /// Export features for a symbol to a flat array for ML inference
    pub fn export_features(&self, symbol: &str) -> Option<Vec<f32>> {
        if let Some(symbol_map) = self.store.get(symbol) {
            let mut features: Vec<(String, f32)> = symbol_map
                .iter()
                .map(|e| (e.key().clone(), e.value().normalized))
                .collect();
            
            // Sort by feature name for consistent ordering
            features.sort_by(|a, b| a.0.cmp(&b.0));
            
            Some(features.into_iter().map(|(_, v)| v).collect())
        } else {
            None
        }
    }
}

impl Default for FeatureStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Get current timestamp in nanoseconds
fn get_timestamp_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64
}

/// Feature normalization utilities
pub mod normalization {
    /// Min-max normalization
    pub fn min_max(value: f64, min: f64, max: f64) -> f32 {
        if max == min {
            return 0.0;
        }
        ((value - min) / (max - min)) as f32
    }

    /// Z-score normalization
    pub fn z_score(value: f64, mean: f64, std_dev: f64) -> f32 {
        if std_dev == 0.0 {
            return 0.0;
        }
        ((value - mean) / std_dev) as f32
    }

    /// Sigmoid normalization to (-1, 1)
    pub fn sigmoid(value: f64) -> f32 {
        let exp = (-value).exp();
        ((1.0 / (1.0 + exp)) * 2.0 - 1.0) as f32
    }

    /// Tanh normalization to (-1, 1)
    pub fn tanh_norm(value: f64) -> f32 {
        value.tanh() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_entry_alignment() {
        let entry = FeatureEntry::default();
        let addr = &entry as *const _ as usize;
        assert_eq!(addr % CACHE_LINE_SIZE, 0, "FeatureEntry should be cache-line aligned");
    }

    #[test]
    fn test_feature_store_basic() {
        let store = FeatureStore::new();
        
        store.register_symbol("BTCUSDT");
        store.set_feature("BTCUSDT", "rsi", FeatureType::Technical, 70.0, 0.7);
        
        let feature = store.get_feature("BTCUSDT", "rsi").unwrap();
        assert_eq!(feature.value, 70.0);
        assert_eq!(feature.normalized, 0.7);
        assert_eq!(feature.feature_type, FeatureType::Technical);
    }

    #[test]
    fn test_feature_normalization() {
        assert_eq!(normalization::min_max(50.0, 0.0, 100.0), 0.5);
        assert_eq!(normalization::z_score(100.0, 100.0, 10.0), 0.0);
        assert!(normalization::sigmoid(0.0).abs() < 0.001);
        assert!(normalization::tanh_norm(0.0).abs() < 0.001);
    }

    #[test]
    fn test_feature_key_hash() {
        let key1 = FeatureKey::new("BTCUSDT", "rsi", FeatureType::Technical);
        let key2 = FeatureKey::new("BTCUSDT", "rsi", FeatureType::Technical);
        
        assert_eq!(key1, key2);
        assert_eq!(key1.hash(), key2.hash());
    }
}
