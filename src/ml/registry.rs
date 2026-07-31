//! Model Registry & Weight Caching
//! 
//! Lock-free model registry caching Python-trained weights (FlatBuffers) in shared memory.
//! Enables Rust core to read ML weights directly without crossing IPC boundary for every inference.

use std::sync::atomic::{AtomicU64, AtomicBool, AtomicPtr, Ordering};
use std::ptr;
use std::slice;

/// Maximum number of models in registry
pub const MAX_MODELS: usize = 256;

/// Maximum weight vector size per model
pub const MAX_WEIGHT_SIZE: usize = 1_048_576; // 1M weights

/// Model metadata with versioning
#[derive(Debug, Clone, Copy)]
#[repr(C, packed)]
pub struct ModelMetadata {
    /// Unique model identifier hash
    pub model_id: u64,
    /// Version number (incremented on each update)
    pub version: u32,
    /// Number of weights/parameters
    pub weight_count: u32,
    /// Number of layers
    pub layer_count: u16,
    /// Model type tag (0=MLP, 1=LSTM, 2=Transformer, etc.)
    pub model_type: u8,
    /// Whether model is currently active
    pub is_active: bool,
    /// Whether model is being shadow tested
    pub is_shadow: bool,
    /// Unix timestamp when model was loaded
    pub loaded_at: u64,
    /// Hash of weight data for integrity check
    pub weight_hash: u64,
    /// Padding for alignment
    _padding: [u8; 3],
}

impl ModelMetadata {
    pub const fn empty() -> Self {
        Self {
            model_id: 0,
            version: 0,
            weight_count: 0,
            layer_count: 0,
            model_type: 0,
            is_active: false,
            is_shadow: false,
            loaded_at: 0,
            weight_hash: 0,
            _padding: [0; 3],
        }
    }
}

/// Pre-allocated weight storage arena
#[repr(C, align(64))]
pub struct WeightArena {
    /// Flat weight storage buffer
    data: Box<[f32; MAX_WEIGHT_SIZE]>,
    /// Current write position
    offset: AtomicU64,
}

impl WeightArena {
    pub const fn new() -> Self {
        Self {
            data: Box::new([0.0; MAX_WEIGHT_SIZE]),
            offset: AtomicU64::new(0),
        }
    }
    
    /// Allocate space for weights and return starting index
    pub fn allocate(&self, count: usize) -> Option<u64> {
        let current = self.offset.load(Ordering::Acquire);
        if current as usize + count > MAX_WEIGHT_SIZE {
            return None;
        }
        
        match self.offset.compare_exchange(
            current,
            current + count as u64,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => Some(current),
            Err(_) => self.allocate(count), // Retry
        }
    }
    
    /// Write weights to arena at specified offset
    pub fn write(&self, offset: u64, weights: &[f32]) -> bool {
        let end = offset as usize + weights.len();
        if end > MAX_WEIGHT_SIZE {
            return false;
        }
        
        unsafe {
            let ptr = self.data.as_ptr().add(offset as usize) as *mut f32;
            ptr::copy_nonoverlapping(weights.as_ptr(), ptr, weights.len());
        }
        true
    }
    
    /// Get slice of weights from arena
    pub fn get_slice(&self, offset: u64, count: usize) -> Option<&[f32]> {
        if offset as usize + count > MAX_WEIGHT_SIZE {
            return None;
        }
        unsafe {
            Some(slice::from_raw_parts(
                self.data.as_ptr().add(offset as usize),
                count,
            ))
        }
    }
    
    /// Reset arena (only safe when no concurrent access)
    pub fn reset(&self) {
        self.offset.store(0, Ordering::Release);
    }
}

/// Lock-free model entry in registry
#[repr(C, align(64))]
pub struct ModelEntry {
    /// Model metadata
    pub metadata: AtomicModelMetadata,
    /// Offset into weight arena
    weight_offset: AtomicU64,
    /// Whether this entry is occupied
    occupied: AtomicBool,
    /// Next entry in hash chain (for collision resolution)
    next_in_chain: AtomicU64,
}

/// Atomic version of ModelMetadata for lock-free updates
#[repr(C)]
pub struct AtomicModelMetadata {
    model_id: AtomicU64,
    version: AtomicU64,
    weight_count: AtomicU64,
    layer_count: AtomicU64,
    model_type: AtomicU64,
    flags: AtomicU64, // Pack booleans
    loaded_at: AtomicU64,
    weight_hash: AtomicU64,
}

impl AtomicModelMetadata {
    pub fn store(&self, meta: &ModelMetadata) {
        self.model_id.store(meta.model_id, Ordering::Release);
        self.version.store(meta.version as u64, Ordering::Release);
        self.weight_count.store(meta.weight_count as u64, Ordering::Release);
        self.layer_count.store(meta.layer_count as u64, Ordering::Release);
        self.model_type.store(meta.model_type as u64, Ordering::Release);
        
        let mut flags = 0u64;
        if meta.is_active { flags |= 1; }
        if meta.is_shadow { flags |= 2; }
        self.flags.store(flags, Ordering::Release);
        
        self.loaded_at.store(meta.loaded_at, Ordering::Release);
        self.weight_hash.store(meta.weight_hash, Ordering::Release);
    }
    
    pub fn load(&self) -> ModelMetadata {
        let flags = self.flags.load(Ordering::Acquire);
        ModelMetadata {
            model_id: self.model_id.load(Ordering::Acquire),
            version: self.version.load(Ordering::Acquire) as u32,
            weight_count: self.weight_count.load(Ordering::Acquire) as u32,
            layer_count: self.layer_count.load(Ordering::Acquire) as u16,
            model_type: self.model_type.load(Ordering::Acquire) as u8,
            is_active: (flags & 1) != 0,
            is_shadow: (flags & 2) != 0,
            loaded_at: self.loaded_at.load(Ordering::Acquire),
            weight_hash: self.weight_hash.load(Ordering::Acquire),
            _padding: [0; 3],
        }
    }
}

impl ModelEntry {
    pub const fn new() -> Self {
        Self {
            metadata: AtomicModelMetadata {
                model_id: AtomicU64::new(0),
                version: AtomicU64::new(0),
                weight_count: AtomicU64::new(0),
                layer_count: AtomicU64::new(0),
                model_type: AtomicU64::new(0),
                flags: AtomicU64::new(0),
                loaded_at: AtomicU64::new(0),
                weight_hash: AtomicU64::new(0),
            },
            weight_offset: AtomicU64::new(u64::MAX),
            occupied: AtomicBool::new(false),
            next_in_chain: AtomicU64::new(u64::MAX),
        }
    }
}

/// Lock-free model registry
#[repr(C, align(64))]
pub struct ModelRegistry {
    /// Pre-allocated model entries
    entries: Box<[ModelEntry; MAX_MODELS]>,
    /// Weight storage arena
    weight_arena: WeightArena,
    /// Hash buckets for O(1) lookup by model_id
    buckets: Box<[AtomicU64; 1024]>,
    /// Total registered models
    model_count: AtomicU64,
    /// Active model count
    active_count: AtomicU64,
}

impl ModelRegistry {
    pub fn new() -> Self {
        let entries = Box::new([ModelEntry::new(); MAX_MODELS]);
        let buckets = Box::new([AtomicU64::new(u64::MAX); 1024]);
        
        Self {
            entries,
            weight_arena: WeightArena::new(),
            buckets,
            model_count: AtomicU64::new(0),
            active_count: AtomicU64::new(0),
        }
    }
    
    /// Hash function for model_id -> bucket index
    #[inline]
    fn hash_model_id(&self, model_id: u64) -> usize {
        // Mix bits for better distribution
        let mut h = model_id;
        h ^= h >> 33;
        h = h.wrapping_mul(0xff51afd7ed558ccd);
        h ^= h >> 33;
        h = h.wrapping_mul(0xc4ceb9fe1a85ec53);
        h ^= h >> 33;
        (h as usize) & 0x3FF
    }
    
    /// Register a new model with weights
    pub fn register(&self, metadata: ModelMetadata, weights: &[f32]) -> Option<u64> {
        if weights.len() != metadata.weight_count as usize {
            return None;
        }
        
        // Allocate space in weight arena
        let weight_offset = self.weight_arena.allocate(weights.len())?;
        
        // Write weights
        if !self.weight_arena.write(weight_offset, weights) {
            return None;
        }
        
        // Find entry slot
        let bucket = self.hash_model_id(metadata.model_id);
        let index = self.find_or_create_entry(metadata.model_id, bucket)?;
        
        let entry = &self.entries[index];
        
        // Store metadata and weight offset
        entry.metadata.store(&metadata);
        entry.weight_offset.store(weight_offset, Ordering::Release);
        entry.occupied.store(true, Ordering::Release);
        
        self.model_count.fetch_add(1, Ordering::Relaxed);
        if metadata.is_active {
            self.active_count.fetch_add(1, Ordering::Relaxed);
        }
        
        Some(index as u64)
    }
    
    /// Find existing entry or create new one
    fn find_or_create_entry(&self, model_id: u64, bucket: usize) -> Option<usize> {
        // Check existing entries in bucket chain
        let mut index = self.buckets[bucket].load(Ordering::Acquire);
        while index != u64::MAX {
            if index as usize >= MAX_MODELS {
                break;
            }
            let entry = &self.entries[index as usize];
            if entry.metadata.model_id.load(Ordering::Acquire) == model_id {
                return Some(index as usize);
            }
            index = entry.next_in_chain.load(Ordering::Acquire);
        }
        
        // Find free slot
        for i in 0..MAX_MODELS {
            if !self.entries[i].occupied.load(Ordering::Acquire) {
                // Try to claim this slot
                if self.entries[i].occupied.compare_exchange(
                    false,
                    true,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                ).is_ok() {
                    // Add to bucket chain
                    let mut head = self.buckets[bucket].load(Ordering::Acquire);
                    loop {
                        self.entries[i].next_in_chain.store(head, Ordering::Release);
                        match self.buckets[bucket].compare_exchange(
                            head,
                            i as u64,
                            Ordering::AcqRel,
                            Ordering::Acquire,
                        ) {
                            Ok(_) => return Some(i),
                            Err(current) => head = current,
                        }
                    }
                }
            }
        }
        
        None
    }
    
    /// Get model weights by model_id
    pub fn get_weights(&self, model_id: u64) -> Option<&[f32]> {
        let bucket = self.hash_model_id(model_id);
        let mut index = self.buckets[bucket].load(Ordering::Acquire);
        
        while index != u64::MAX {
            if index as usize >= MAX_MODELS {
                break;
            }
            let entry = &self.entries[index as usize];
            if entry.metadata.model_id.load(Ordering::Acquire) == model_id {
                let meta = entry.metadata.load();
                let offset = entry.weight_offset.load(Ordering::Acquire);
                return self.weight_arena.get_slice(offset, meta.weight_count as usize);
            }
            index = entry.next_in_chain.load(Ordering::Acquire);
        }
        
        None
    }
    
    /// Get model metadata by model_id
    pub fn get_metadata(&self, model_id: u64) -> Option<ModelMetadata> {
        let bucket = self.hash_model_id(model_id);
        let mut index = self.buckets[bucket].load(Ordering::Acquire);
        
        while index != u64::MAX {
            if index as usize >= MAX_MODELS {
                break;
            }
            let entry = &self.entries[index as usize];
            if entry.metadata.model_id.load(Ordering::Acquire) == model_id {
                return Some(entry.metadata.load());
            }
            index = entry.next_in_chain.load(Ordering::Acquire);
        }
        
        None
    }
    
    /// Activate a model (promote from shadow to active)
    pub fn activate(&self, model_id: u64) -> bool {
        let bucket = self.hash_model_id(model_id);
        let mut index = self.buckets[bucket].load(Ordering::Acquire);
        
        while index != u64::MAX {
            if index as usize >= MAX_MODELS {
                break;
            }
            let entry = &self.entries[index as usize];
            if entry.metadata.model_id.load(Ordering::Acquire) == model_id {
                let mut meta = entry.metadata.load();
                if !meta.is_active {
                    meta.is_active = true;
                    meta.is_shadow = false;
                    entry.metadata.store(&meta);
                    self.active_count.fetch_add(1, Ordering::Relaxed);
                }
                return true;
            }
            index = entry.next_in_chain.load(Ordering::Acquire);
        }
        
        false
    }
    
    /// Get all active models
    pub fn get_active_models(&self) -> Vec<ModelMetadata> {
        let mut models = Vec::with_capacity(32);
        for i in 0..MAX_MODELS {
            let entry = &self.entries[i];
            if entry.occupied.load(Ordering::Acquire) {
                let meta = entry.metadata.load();
                if meta.is_active {
                    models.push(meta);
                }
            }
        }
        models
    }
    
    /// Get registry statistics
    pub fn get_stats(&self) -> RegistryStats {
        RegistryStats {
            total_models: self.model_count.load(Ordering::Relaxed),
            active_models: self.active_count.load(Ordering::Relaxed),
            weight_bytes_used: self.weight_arena.offset.load(Ordering::Relaxed) * 4, // f32 = 4 bytes
            weight_capacity: MAX_WEIGHT_SIZE * 4,
        }
    }
}

/// Registry statistics snapshot
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct RegistryStats {
    pub total_models: u64,
    pub active_models: u64,
    pub weight_bytes_used: u64,
    pub weight_capacity: usize,
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_registry_basic() {
        let registry = ModelRegistry::new();
        
        let metadata = ModelMetadata {
            model_id: 12345,
            version: 1,
            weight_count: 100,
            layer_count: 3,
            model_type: 0,
            is_active: true,
            is_shadow: false,
            loaded_at: 1000000,
            weight_hash: 0xABCD,
            _padding: [0; 3],
        };
        
        let weights: Vec<f32> = (0..100).map(|i| i as f32 * 0.01).collect();
        
        assert!(registry.register(metadata, &weights).is_some());
        
        // Retrieve weights
        let retrieved = registry.get_weights(12345);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().len(), 100);
        
        // Retrieve metadata
        let meta = registry.get_metadata(12345);
        assert!(meta.is_some());
        assert_eq!(meta.unwrap().version, 1);
    }
    
    #[test]
    fn test_activate_shadow() {
        let registry = ModelRegistry::new();
        
        let metadata = ModelMetadata {
            model_id: 67890,
            version: 2,
            weight_count: 50,
            layer_count: 2,
            model_type: 1,
            is_active: false,
            is_shadow: true,
            loaded_at: 2000000,
            weight_hash: 0x1234,
            _padding: [0; 3],
        };
        
        let weights: Vec<f32> = (0..50).map(|i| i as f32).collect();
        
        registry.register(metadata, &weights);
        
        // Should be shadow initially
        let meta = registry.get_metadata(67890).unwrap();
        assert!(meta.is_shadow);
        assert!(!meta.is_active);
        
        // Activate
        registry.activate(67890);
        
        let meta = registry.get_metadata(67890).unwrap();
        assert!(!meta.is_shadow);
        assert!(meta.is_active);
    }
}
