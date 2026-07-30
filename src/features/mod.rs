//! Features Module Root
//! 
//! Defines strict feature schemas and manages the feature registry lifecycle.
//! Re-exports store and rolling window components.

pub mod store;
pub mod rolling;

use std::{
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    thread,
    time::Duration,
};

use crate::features::store::{FeatureStore, FeatureType};
use crate::features::rolling::RollingFeatures;
use crate::ipc::shared_memory::SharedMemoryManager;

/// Feature schema definition for ML models
#[derive(Debug, Clone)]
pub struct FeatureSchema {
    pub name: String,
    pub feature_type: FeatureType,
    pub normalization: NormalizationType,
    pub min_value: f64,
    pub max_value: f64,
    pub required: bool,
}

impl FeatureSchema {
    pub fn new(
        name: &str,
        feature_type: FeatureType,
        normalization: NormalizationType,
        min_value: f64,
        max_value: f64,
    ) -> Self {
        Self {
            name: name.to_string(),
            feature_type,
            normalization,
            min_value,
            max_value,
            required: true,
        }
    }

    pub fn optional(mut self) -> Self {
        self.required = false;
        self
    }
}

/// Normalization types for features
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormalizationType {
    MinMax,
    ZScore,
    Sigmoid,
    Tanh,
    None,
}

/// Feature registry managing schemas for all symbols
pub struct FeatureRegistry {
    /// Registered schemas per symbol
    schemas: dashmap::DashMap<String, Vec<FeatureSchema>>,
    /// Feature store
    store: Arc<FeatureStore>,
    /// Rolling calculators per symbol
    rolling_calculators: dashmap::DashMap<String, Arc<RollingFeatures>>,
    /// Running flag
    running: Arc<AtomicBool>,
    /// Update counter
    update_count: AtomicU64,
}

unsafe impl Send for FeatureRegistry {}
unsafe impl Sync for FeatureRegistry {}

impl FeatureRegistry {
    /// Create a new feature registry
    pub fn new() -> Self {
        Self {
            schemas: dashmap::DashMap::new(),
            store: Arc::new(FeatureStore::new()),
            rolling_calculators: dashmap::DashMap::new(),
            running: Arc::new(AtomicBool::new(false)),
            update_count: AtomicU64::new(0),
        }
    }

    /// Register a symbol with its feature schemas
    pub fn register_symbol(&self, symbol: &str, schemas: Vec<FeatureSchema>) {
        self.schemas.insert(symbol.to_string(), schemas);
        self.store.register_symbol(symbol);
        
        // Create rolling calculator for this symbol
        let calculator = Arc::new(RollingFeatures::new(100));
        self.rolling_calculators.insert(symbol.to_string(), calculator);
    }

    /// Process a new tick for a symbol
    pub fn process_tick(&self, symbol: &str, price: f64, volume: f64) {
        if !self.running.load(Ordering::Acquire) {
            return;
        }

        // Update rolling features
        if let Some(calculator) = self.rolling_calculators.get(symbol) {
            calculator.update(price, volume);
            
            // Extract and store features
            let rsi = calculator.get_rsi();
            let macd = calculator.get_macd();
            let volatility = calculator.get_volatility();
            let momentum = calculator.get_momentum();
            
            // Store in feature store with normalization
            self.store.set_feature(
                symbol,
                "rsi",
                FeatureType::Technical,
                rsi,
                (rsi / 100.0) as f32, // Normalize to 0-1
            );
            
            self.store.set_feature(
                symbol,
                "macd",
                FeatureType::Technical,
                macd,
                ((macd + 100.0) / 200.0) as f32, // Approximate normalization
            );
            
            self.store.set_feature(
                symbol,
                "volatility",
                FeatureType::Volatility,
                volatility,
                volatility.tanh() as f32,
            );
            
            self.store.set_feature(
                symbol,
                "momentum",
                FeatureType::Momentum,
                momentum,
                momentum.tanh() as f32,
            );
        }

        self.update_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Get feature vector for ML inference
    pub fn get_feature_vector(&self, symbol: &str) -> Option<Vec<f32>> {
        if let Some(calculator) = self.rolling_calculators.get(symbol) {
            Some(calculator.export_features())
        } else {
            self.store.export_features(symbol)
        }
    }

    /// Get all registered symbols
    pub fn get_symbols(&self) -> Vec<String> {
        self.schemas.iter().map(|s| s.key().clone()).collect()
    }

    /// Start the feature registry background tasks
    pub fn start(&self) {
        self.running.store(true, Ordering::Release);
    }

    /// Stop the feature registry
    pub fn stop(&self) {
        self.running.store(false, Ordering::Release);
    }

    /// Get update count
    pub fn get_update_count(&self) -> u64 {
        self.update_count.load(Ordering::Relaxed)
    }

    /// Get reference to feature store
    pub fn store(&self) -> &Arc<FeatureStore> {
        &self.store
    }
}

impl Default for FeatureRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Feature manager coordinating all feature-related operations
pub struct FeatureManager {
    registry: Arc<FeatureRegistry>,
    shared_memory: Option<Arc<SharedMemoryManager>>,
    running: Arc<AtomicBool>,
}

unsafe impl Send for FeatureManager {}
unsafe impl Sync for FeatureManager {}

impl FeatureManager {
    /// Create a new feature manager
    pub fn new() -> Self {
        Self {
            registry: Arc::new(FeatureRegistry::new()),
            shared_memory: None,
            running: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Initialize with shared memory
    pub fn with_shared_memory(mut self, shmem: SharedMemoryManager) -> Self {
        self.shared_memory = Some(Arc::new(shmem));
        self
    }

    /// Register a trading pair
    pub fn register_symbol(&self, symbol: &str) {
        let schemas = vec![
            FeatureSchema::new("rsi", FeatureType::Technical, NormalizationType::MinMax, 0.0, 100.0),
            FeatureSchema::new("macd", FeatureType::Technical, NormalizationType::ZScore, -50.0, 50.0),
            FeatureSchema::new("volatility", FeatureType::Volatility, NormalizationType::Tanh, 0.0, 1.0),
            FeatureSchema::new("momentum", FeatureType::Momentum, NormalizationType::Sigmoid, -1.0, 1.0),
            FeatureSchema::new("volume_profile", FeatureType::OrderFlow, NormalizationType::MinMax, 0.0, 1.0),
        ];
        
        self.registry.register_symbol(symbol, schemas);
    }

    /// Process market data tick
    pub fn process_tick(&self, symbol: &str, price: f64, volume: f64) {
        self.registry.process_tick(symbol, price, volume);
        
        // Optionally write to shared memory
        if let Some(ref shmem) = self.shared_memory {
            if let Some(features) = self.registry.get_feature_vector(symbol) {
                let symbol_id = symbol_to_u64(symbol);
                let timestamp_ns = get_timestamp_ns();
                
                let _ = shmem.write_features(
                    symbol_id,
                    &features,
                    timestamp_ns,
                    0, // feature flags
                );
            }
        }
    }

    /// Get feature vector for inference
    pub fn get_features(&self, symbol: &str) -> Option<Vec<f32>> {
        self.registry.get_feature_vector(symbol)
    }

    /// Start the feature manager
    pub fn start(&self) {
        self.running.store(true, Ordering::Release);
        self.registry.start();
    }

    /// Stop the feature manager
    pub fn stop(&self) {
        self.running.store(false, Ordering::Release);
        self.registry.stop();
    }

    /// Check if running
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Acquire)
    }
}

impl Default for FeatureManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert symbol string to u64 ID
fn symbol_to_u64(symbol: &str) -> u64 {
    let bytes = symbol.as_bytes();
    let mut id: u64 = 0;
    for (i, &b) in bytes.iter().take(8).enumerate() {
        id |= (b as u64) << (i * 8);
    }
    id
}

/// Get current timestamp in nanoseconds
fn get_timestamp_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_schema_creation() {
        let schema = FeatureSchema::new(
            "rsi",
            FeatureType::Technical,
            NormalizationType::MinMax,
            0.0,
            100.0,
        );
        
        assert_eq!(schema.name, "rsi");
        assert_eq!(schema.feature_type, FeatureType::Technical);
        assert!(schema.required);
        
        let optional_schema = schema.clone().optional();
        assert!(!optional_schema.required);
    }

    #[test]
    fn test_feature_registry() {
        let registry = FeatureRegistry::new();
        
        registry.register_symbol("BTCUSDT", vec![
            FeatureSchema::new("rsi", FeatureType::Technical, NormalizationType::MinMax, 0.0, 100.0),
        ]);
        
        registry.start();
        registry.process_tick("BTCUSDT", 45000.0, 1000.0);
        
        let symbols = registry.get_symbols();
        assert!(symbols.contains(&"BTCUSDT".to_string()));
        
        let features = registry.get_feature_vector("BTCUSDT");
        assert!(features.is_some());
    }

    #[test]
    fn test_feature_manager() {
        let manager = FeatureManager::new();
        
        manager.register_symbol("ETHUSDT");
        manager.start();
        
        manager.process_tick("ETHUSDT", 3000.0, 500.0);
        
        assert!(manager.is_running());
        
        let features = manager.get_features("ETHUSDT");
        assert!(features.is_some());
        
        manager.stop();
        assert!(!manager.is_running());
    }

    #[test]
    fn test_symbol_to_u64() {
        let id1 = symbol_to_u64("BTCUSDT");
        let id2 = symbol_to_u64("BTCUSDT");
        let id3 = symbol_to_u64("ETHUSDT");
        
        assert_eq!(id1, id2);
        assert_ne!(id1, id3);
    }
}
