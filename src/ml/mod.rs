//! ML Module Root
//! 
//! Manages model versioning, shadow-testing new weights, and atomic hot-swapping of live models.

pub mod registry;
pub mod inference_cache;

pub use registry::{ModelRegistry, ModelMetadata, RegistryStats, WeightArena};
pub use inference_cache::{InferenceCache, FeatureKey, CacheStats, CachedResult};

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};

/// ML system coordinator for model lifecycle management
#[repr(C, align(64))]
pub struct MlSystem {
    /// Model registry
    pub registry: ModelRegistry,
    /// Inference result cache
    pub cache: InferenceCache,
    /// Current active model version (global)
    active_version: AtomicU64,
    /// Shadow model being tested
    shadow_version: AtomicU64,
    /// Whether shadow testing is enabled
    shadow_enabled: AtomicBool,
    /// Total inferences performed
    total_inferences: AtomicU64,
    /// Shadow test inferences
    shadow_inferences: AtomicU64,
}

impl MlSystem {
    pub fn new() -> Self {
        Self {
            registry: ModelRegistry::new(),
            cache: InferenceCache::new(),
            active_version: AtomicU64::new(0),
            shadow_version: AtomicU64::new(0),
            shadow_enabled: AtomicBool::new(false),
            total_inferences: AtomicU64::new(0),
            shadow_inferences: AtomicU64::new(0),
        }
    }
    
    /// Register a new model
    pub fn register_model(&self, metadata: ModelMetadata, weights: &[f32]) -> Option<u64> {
        self.registry.register(metadata, weights)
    }
    
    /// Enable shadow testing for a model
    pub fn enable_shadow_test(&self, model_id: u64) -> bool {
        if let Some(mut meta) = self.registry.get_metadata(model_id) {
            if !meta.is_active {
                meta.is_shadow = true;
                // Need to re-register with updated metadata
                if let Some(weights) = self.registry.get_weights(model_id) {
                    let weights_vec: Vec<f32> = weights.to_vec();
                    return self.registry.register(meta, &weights_vec).is_some();
                }
            }
        }
        false
    }
    
    /// Promote shadow model to active
    pub fn promote_shadow(&self, model_id: u64) -> bool {
        if self.registry.activate(model_id) {
            let new_version = self.active_version.fetch_add(1, Ordering::AcqRel) + 1;
            self.shadow_version.store(new_version, Ordering::Release);
            self.shadow_enabled.store(false, Ordering::Release);
            true
        } else {
            false
        }
    }
    
    /// Run inference with caching
    pub fn infer(&self, model_id: u64, features: &[f64]) -> Option<(f64, f32)> {
        let key = FeatureKey::from_slice(features);
        
        // Check cache first
        if let Some(result) = self.cache.get(&key, model_id as u32) {
            self.total_inferences.fetch_add(1, Ordering::Relaxed);
            return Some(result);
        }
        
        // Cache miss - would normally call model here
        // For now, return None to indicate computation needed externally
        self.total_inferences.fetch_add(1, Ordering::Relaxed);
        None
    }
    
    /// Store inference result in cache
    pub fn cache_result(&self, features: &[f64], value: f64, confidence: f32, model_id: u64) {
        let key = FeatureKey::from_slice(features);
        self.cache.insert(key, value, confidence, model_id as u32);
    }
    
    /// Run shadow inference (parallel to main model)
    pub fn shadow_infer(&self, model_id: u64, features: &[f64]) -> Option<(f64, f32)> {
        if !self.shadow_enabled.load(Ordering::Acquire) {
            return None;
        }
        
        let key = FeatureKey::from_slice(features);
        self.shadow_inferences.fetch_add(1, Ordering::Relaxed);
        
        // Get result from shadow model
        self.cache.get(&key, model_id as u32)
    }
    
    /// Compare shadow vs active model performance
    pub fn get_shadow_comparison(&self) -> ShadowComparison {
        let total = self.total_inferences.load(Ordering::Relaxed);
        let shadow = self.shadow_inferences.load(Ordering::Relaxed);
        
        ShadowComparison {
            total_inferences: total,
            shadow_inferences: shadow,
            shadow_ratio: if total > 0 { shadow as f64 / total as f64 } else { 0.0 },
            active_version: self.active_version.load(Ordering::Acquire),
            shadow_version: self.shadow_version.load(Ordering::Acquire),
            shadow_enabled: self.shadow_enabled.load(Ordering::Acquire),
        }
    }
    
    /// Get comprehensive ML system stats
    pub fn get_stats(&self) -> MlStats {
        let reg_stats = self.registry.get_stats();
        let cache_stats = self.cache.get_stats();
        let shadow = self.get_shadow_comparison();
        
        MlStats {
            registry: reg_stats,
            cache: cache_stats,
            shadow: shadow,
        }
    }
    
    /// Clear inference cache (useful when models are updated)
    pub fn clear_cache(&self) {
        self.cache.clear();
    }
    
    /// Hot-swap to new model version atomically
    pub fn hot_swap(&self, new_model_id: u64) -> bool {
        // Verify new model exists and is active
        if let Some(meta) = self.registry.get_metadata(new_model_id) {
            if meta.is_active {
                // Clear cache to force fresh inferences with new model
                self.clear_cache();
                self.active_version.fetch_add(1, Ordering::AcqRel);
                return true;
            }
        }
        false
    }
}

/// Shadow testing comparison metrics
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct ShadowComparison {
    pub total_inferences: u64,
    pub shadow_inferences: u64,
    pub shadow_ratio: f64,
    pub active_version: u64,
    pub shadow_version: u64,
    pub shadow_enabled: bool,
}

/// Comprehensive ML system statistics
#[derive(Debug, Clone)]
#[repr(C)]
pub struct MlStats {
    pub registry: RegistryStats,
    pub cache: CacheStats,
    pub shadow: ShadowComparison,
}

impl Default for MlSystem {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_ml_system_basic() {
        let system = MlSystem::new();
        
        let metadata = ModelMetadata {
            model_id: 11111,
            version: 1,
            weight_count: 50,
            layer_count: 2,
            model_type: 0,
            is_active: true,
            is_shadow: false,
            loaded_at: 1000000,
            weight_hash: 0xDEAD,
            _padding: [0; 3],
        };
        
        let weights: Vec<f32> = (0..50).map(|i| i as f32 * 0.1).collect();
        
        assert!(system.register_model(metadata, &weights).is_some());
        
        // Test inference (cache miss initially)
        let features = [1.0, 2.0, 3.0];
        assert!(system.infer(11111, &features).is_none());
        
        // Cache a result
        system.cache_result(&features, 0.75, 0.9, 11111);
        
        // Now should hit cache
        let (value, conf) = system.infer(11111, &features).unwrap();
        assert!((value - 0.75).abs() < 1e-10);
        assert!((conf - 0.9).abs() < 1e-5);
        
        let stats = system.get_stats();
        assert_eq!(stats.cache.hits, 1);
    }
    
    #[test]
    fn test_hot_swap() {
        let system = MlSystem::new();
        
        let metadata = ModelMetadata {
            model_id: 22222,
            version: 2,
            weight_count: 30,
            layer_count: 1,
            model_type: 1,
            is_active: true,
            is_shadow: false,
            loaded_at: 2000000,
            weight_hash: 0xBEEF,
            _padding: [0; 3],
        };
        
        let weights: Vec<f32> = (0..30).map(|i| i as f32).collect();
        system.register_model(metadata, &weights);
        
        let initial_version = system.get_shadow_comparison().active_version;
        
        // Hot swap
        assert!(system.hot_swap(22222));
        
        let new_version = system.get_shadow_comparison().active_version;
        assert_eq!(new_version, initial_version + 1);
    }
}
