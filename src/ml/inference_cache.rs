//! Inference Result Cache
//! 
//! Concurrent LRU cache for inference results to prevent redundant calculations
//! for identical feature vectors.

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use core::hash::{Hash, Hasher};

/// Maximum cache entries
pub const MAX_CACHE_ENTRIES: usize = 8192;

/// Number of hash buckets for O(1) lookup
const HASH_BUCKETS: usize = 2048;

/// Feature vector key (fixed size for performance)
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(C, align(64))]
pub struct FeatureKey {
    data: [u64; 8],
}

impl FeatureKey {
    pub const fn zeros() -> Self {
        Self { data: [0; 8] }
    }
    
    pub fn from_slice(slice: &[f64]) -> Self {
        let mut key = Self::zeros();
        let len = slice.len().min(64); // Max 64 f64 values
        
        // Convert f64 to u64 bits for exact comparison
        for i in 0..len {
            if i < 8 {
                key.data[i] = slice[i].to_bits();
            } else {
                // Hash additional values into the key
                key.data[i % 8] ^= slice[i].to_bits();
            }
        }
        
        key
    }
    
    pub fn hash(&self) -> u64 {
        let mut h = 0u64;
        for &word in &self.data {
            h ^= word;
            h = h.wrapping_mul(0x517cc1b727220a95);
            h ^= h >> 32;
        }
        h
    }
}

/// Cached inference result
#[derive(Clone, Copy)]
#[repr(C, align(64))]
pub struct CachedResult {
    /// Result value
    pub value: f64,
    /// Confidence score
    pub confidence: f32,
    /// Model ID that produced this result
    pub model_id: u32,
    /// Access count for LRU tracking
    pub access_count: AtomicU64,
    /// Last access timestamp
    pub last_access: AtomicU64,
    /// Whether this entry is valid
    pub valid: AtomicBool,
    /// Padding
    _padding: [u8; 52],
}

impl CachedResult {
    pub const fn empty() -> Self {
        Self {
            value: 0.0,
            confidence: 0.0,
            model_id: 0,
            access_count: AtomicU64::new(0),
            last_access: AtomicU64::new(0),
            valid: AtomicBool::new(false),
            _padding: [0; 52],
        }
    }
    
    pub fn record_access(&self, timestamp: u64) {
        self.access_count.fetch_add(1, Ordering::Relaxed);
        self.last_access.store(timestamp, Ordering::Relaxed);
    }
}

/// LRU cache entry with key and value
#[repr(C, align(64))]
pub struct CacheEntry {
    pub key: FeatureKey,
    pub result: CachedResult,
    /// Next entry in hash chain
    next_in_chain: AtomicU64,
}

impl CacheEntry {
    pub const fn new() -> Self {
        Self {
            key: FeatureKey::zeros(),
            result: CachedResult::empty(),
            next_in_chain: AtomicU64::new(u64::MAX),
        }
    }
}

/// Lock-free LRU inference cache
#[repr(C, align(64))]
pub struct InferenceCache {
    /// Pre-allocated cache entries
    entries: Box<[CacheEntry; MAX_CACHE_ENTRIES]>,
    /// Hash buckets for O(1) lookup
    buckets: Box<[AtomicU64; HASH_BUCKETS]>,
    /// Total hits
    hits: AtomicU64,
    /// Total misses
    misses: AtomicU64,
    /// Current entry count
    count: AtomicU64,
    /// Global timestamp counter for LRU
    timestamp: AtomicU64,
}

impl InferenceCache {
    pub fn new() -> Self {
        let entries = Box::new([CacheEntry::new(); MAX_CACHE_ENTRIES]);
        let buckets = Box::new([AtomicU64::new(u64::MAX); HASH_BUCKETS]);
        
        Self {
            entries,
            buckets,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            count: AtomicU64::new(0),
            timestamp: AtomicU64::new(0),
        }
    }
    
    /// Get bucket index for a key
    #[inline]
    fn get_bucket(&self, key: &FeatureKey) -> usize {
        (key.hash() as usize) & (HASH_BUCKETS - 1)
    }
    
    /// Lookup cached result by feature key
    pub fn get(&self, key: &FeatureKey, model_id: u32) -> Option<(f64, f32)> {
        let bucket = self.get_bucket(key);
        let ts = self.timestamp.fetch_add(1, Ordering::Relaxed);
        
        let mut index = self.buckets[bucket].load(Ordering::Acquire);
        while index != u64::MAX {
            if index as usize >= MAX_CACHE_ENTRIES {
                break;
            }
            
            let entry = &self.entries[index as usize];
            if entry.key == *key && entry.result.valid.load(Ordering::Acquire) {
                let meta_model_id = entry.result.model_id;
                if meta_model_id == 0 || meta_model_id == model_id {
                    entry.result.record_access(ts);
                    self.hits.fetch_add(1, Ordering::Relaxed);
                    return Some((entry.result.value, entry.result.confidence));
                }
            }
            index = entry.next_in_chain.load(Ordering::Acquire);
        }
        
        self.misses.fetch_add(1, Ordering::Relaxed);
        None
    }
    
    /// Insert result into cache
    pub fn insert(&self, key: FeatureKey, value: f64, confidence: f32, model_id: u32) -> bool {
        let bucket = self.get_bucket(&key);
        let ts = self.timestamp.load(Ordering::Relaxed);
        
        // First check if key already exists
        let mut index = self.buckets[bucket].load(Ordering::Acquire);
        while index != u64::MAX {
            if index as usize >= MAX_CACHE_ENTRIES {
                break;
            }
            
            let entry = &self.entries[index as usize];
            if entry.key == key && entry.result.valid.load(Ordering::Acquire) {
                // Update existing
                entry.result.value = value;
                entry.result.confidence = confidence;
                entry.result.model_id = model_id;
                entry.result.record_access(ts);
                return true;
            }
            index = entry.next_in_chain.load(Ordering::Acquire);
        }
        
        // Find free slot or evict LRU
        let slot = self.find_or_evict_slot(bucket, key)?;
        
        let entry = &self.entries[slot];
        entry.key = key;
        entry.result.value = value;
        entry.result.confidence = confidence;
        entry.result.model_id = model_id;
        entry.result.access_count.store(1, Ordering::Release);
        entry.result.last_access.store(ts, Ordering::Release);
        entry.result.valid.store(true, Ordering::Release);
        
        self.count.fetch_add(1, Ordering::Relaxed);
        true
    }
    
    /// Find free slot or evict least recently used
    fn find_or_evict_slot(&self, bucket: usize, new_key: FeatureKey) -> Option<usize> {
        // Try to find free slot first
        for i in 0..MAX_CACHE_ENTRIES {
            if !self.entries[i].result.valid.load(Ordering::Acquire) {
                // Claim this slot
                if self.entries[i].result.valid.compare_exchange(
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
        
        // All slots full, evict LRU
        self.evict_lru(bucket, new_key)
    }
    
    /// Evict least recently used entry
    fn evict_lru(&self, bucket: usize, new_key: FeatureKey) -> Option<usize> {
        let mut min_access = u64::MAX;
        let mut min_idx = 0usize;
        
        // Find entry with minimum access count
        for i in 0..MAX_CACHE_ENTRIES {
            let entry = &self.entries[i];
            if entry.result.valid.load(Ordering::Acquire) {
                let access = entry.result.access_count.load(Ordering::Relaxed);
                if access < min_access {
                    min_access = access;
                    min_idx = i;
                }
            }
        }
        
        // Evict this entry
        let entry = &self.entries[min_idx];
        entry.key = new_key;
        entry.result.access_count.store(1, Ordering::Release);
        entry.result.last_access.store(self.timestamp.load(Ordering::Relaxed), Ordering::Release);
        
        Some(min_idx)
    }
    
    /// Get cache statistics
    pub fn get_stats(&self) -> CacheStats {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total = hits + misses;
        
        CacheStats {
            hits,
            misses,
            hit_rate: if total > 0 { hits as f64 / total as f64 } else { 0.0 },
            count: self.count.load(Ordering::Relaxed),
            capacity: MAX_CACHE_ENTRIES,
        }
    }
    
    /// Clear all cache entries
    pub fn clear(&self) {
        for i in 0..MAX_CACHE_ENTRIES {
            self.entries[i].result.valid.store(false, Ordering::Release);
        }
        for bucket in &*self.buckets {
            bucket.store(u64::MAX, Ordering::Release);
        }
        self.count.store(0, Ordering::Release);
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
    }
}

/// Cache statistics snapshot
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub hit_rate: f64,
    pub count: u64,
    pub capacity: usize,
}

impl Default for InferenceCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cache_basic() {
        let cache = InferenceCache::new();
        
        let key = FeatureKey::from_slice(&[1.0, 2.0, 3.0, 4.0]);
        
        // Should miss initially
        assert!(cache.get(&key, 1).is_none());
        
        // Insert
        cache.insert(key, 0.95, 0.88, 1);
        
        // Should hit now
        let (value, conf) = cache.get(&key, 1).unwrap();
        assert!((value - 0.95).abs() < 1e-10);
        assert!((conf - 0.88).abs() < 1e-5);
        
        let stats = cache.get_stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
    }
    
    #[test]
    fn test_cache_different_models() {
        let cache = InferenceCache::new();
        
        let key = FeatureKey::from_slice(&[5.0, 6.0, 7.0]);
        
        // Insert for model 1
        cache.insert(key, 0.1, 0.2, 1);
        
        // Same key, different model should miss
        assert!(cache.get(&key, 2).is_none());
        
        // Same key, same model should hit
        assert!(cache.get(&key, 1).is_some());
    }
}
