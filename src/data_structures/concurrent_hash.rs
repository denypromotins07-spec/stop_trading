//! Lock-Free Concurrent Hash Map
//! Linear-probing hash map inspired by rustc-hash for O(1) feature state caching.
//! Zero heap allocations using pre-allocated buckets.

use core::sync::atomic::{AtomicI64, AtomicU8, AtomicUsize, Ordering};
use core::hash::{Hash, Hasher};

/// Maximum number of buckets (power of 2)
const MAX_BUCKETS: usize = 16384;

/// Bucket states
const BUCKET_EMPTY: u8 = 0;
const BUCKET_OCCUPIED: u8 = 1;
const BUCKET_DELETED: u8 = 2;

/// Single bucket entry
#[repr(C)]
pub struct HashBucket {
    key: AtomicI64,
    value: AtomicI64,
    state: AtomicU8,
    _padding: [u8; 5],
}

impl HashBucket {
    const fn new() -> Self {
        Self {
            key: AtomicI64::new(0),
            value: AtomicI64::new(0),
            state: AtomicU8::new(BUCKET_EMPTY),
            _padding: [0; 5],
        }
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.state.load(Ordering::Acquire) == BUCKET_EMPTY
    }

    #[inline]
    fn is_occupied(&self) -> bool {
        self.state.load(Ordering::Acquire) == BUCKET_OCCUPIED
    }

    #[inline]
    fn get_key(&self) -> i64 {
        self.key.load(Ordering::Acquire)
    }

    #[inline]
    fn get_value(&self) -> i64 {
        self.value.load(Ordering::Acquire)
    }

    fn set(&self, key: i64, value: i64) {
        self.key.store(key, Ordering::Relaxed);
        self.value.store(value, Ordering::Relaxed);
        self.state.store(BUCKET_OCCUPIED, Ordering::Release);
    }

    fn mark_deleted(&self) {
        self.state.store(BUCKET_DELETED, Ordering::Release);
    }

    fn clear(&self) {
        self.state.store(BUCKET_EMPTY, Ordering::Release);
    }
}

/// FNV-1a hash function (fast, good distribution for integers)
#[inline]
fn fnv1a_hash(key: i64) -> u64 {
    const FNV_OFFSET: u64 = 14695981039346656037;
    const FNV_PRIME: u64 = 1099511628211;

    let mut hash = FNV_OFFSET;
    let bytes = key.to_ne_bytes();
    for byte in bytes {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Lock-free concurrent hash map
pub struct ConcurrentHashMap {
    buckets: [HashBucket; MAX_BUCKETS],
    mask: usize,
    size: AtomicUsize,
}

unsafe impl Send for ConcurrentHashMap {}
unsafe impl Sync for ConcurrentHashMap {}

impl Default for ConcurrentHashMap {
    fn default() -> Self {
        Self::new()
    }
}

impl ConcurrentHashMap {
    /// Create a new hash map
    pub const fn new() -> Self {
        Self {
            buckets: unsafe { core::mem::zeroed() },
            mask: MAX_BUCKETS - 1,
            size: AtomicUsize::new(0),
        }
    }

    /// Get bucket index for a key
    #[inline]
    fn get_index(&self, key: i64) -> usize {
        (fnv1a_hash(key) as usize) & self.mask
    }

    /// Insert or update a key-value pair
    pub fn insert(&self, key: i64, value: i64) -> Option<i64> {
        let mut idx = self.get_index(key);
        let mut first_deleted: Option<usize> = None;

        loop {
            let bucket = unsafe { self.buckets.get_unchecked(idx) };
            let state = bucket.state.load(Ordering::Acquire);

            if state == BUCKET_EMPTY {
                // Found empty slot - try to claim it
                if let Some(del_idx) = first_deleted {
                    idx = del_idx;
                    bucket = unsafe { self.buckets.get_unchecked(idx) };
                }

                // Try to insert
                let expected = BUCKET_EMPTY;
                if bucket.state.compare_exchange(
                    expected,
                    BUCKET_OCCUPIED,
                    Ordering::SeqCst,
                    Ordering::Relaxed,
                ).is_ok() {
                    bucket.key.store(key, Ordering::Relaxed);
                    bucket.value.store(value, Ordering::Release);
                    self.size.fetch_add(1, Ordering::Relaxed);
                    return None;
                }
                // Another thread claimed it, continue probing
            } else if state == BUCKET_OCCUPIED {
                if bucket.get_key() == key {
                    // Key exists - update value
                    let old_value = bucket.get_value();
                    bucket.value.store(value, Ordering::Release);
                    return Some(old_value);
                }
            } else if state == BUCKET_DELETED && first_deleted.is_none() {
                first_deleted = Some(idx);
            }

            // Linear probing
            idx = (idx + 1) & self.mask;

            // Prevent infinite loop
            if idx == self.get_index(key) {
                break;
            }
        }

        // Table full
        None
    }

    /// Get value for a key
    pub fn get(&self, key: i64) -> Option<i64> {
        let mut idx = self.get_index(key);
        let start_idx = idx;

        loop {
            let bucket = unsafe { self.buckets.get_unchecked(idx) };
            let state = bucket.state.load(Ordering::Acquire);

            if state == BUCKET_EMPTY {
                return None;
            }

            if state == BUCKET_OCCUPIED && bucket.get_key() == key {
                return Some(bucket.get_value());
            }

            idx = (idx + 1) & self.mask;
            if idx == start_idx {
                return None;
            }
        }
    }

    /// Remove a key
    pub fn remove(&self, key: i64) -> Option<i64> {
        let mut idx = self.get_index(key);
        let start_idx = idx;

        loop {
            let bucket = unsafe { self.buckets.get_unchecked(idx) };
            let state = bucket.state.load(Ordering::Acquire);

            if state == BUCKET_EMPTY {
                return None;
            }

            if state == BUCKET_OCCUPIED && bucket.get_key() == key {
                let value = bucket.get_value();
                bucket.mark_deleted();
                self.size.fetch_sub(1, Ordering::Relaxed);
                return Some(value);
            }

            idx = (idx + 1) & self.mask;
            if idx == start_idx {
                return None;
            }
        }
    }

    /// Check if key exists
    #[inline]
    pub fn contains(&self, key: i64) -> bool {
        self.get(key).is_some()
    }

    /// Get number of entries
    #[inline]
    pub fn len(&self) -> usize {
        self.size.load(Ordering::Relaxed)
    }

    /// Check if empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clear all entries
    pub fn clear(&self) {
        for i in 0..MAX_BUCKETS {
            unsafe {
                self.buckets.get_unchecked(i).clear();
            }
        }
        self.size.store(0, Ordering::Relaxed);
    }

    /// Get load factor (scaled by 100)
    pub fn load_factor_percent(&self) -> usize {
        (self.len() * 100) / MAX_BUCKETS
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_get() {
        let map = ConcurrentHashMap::new();
        
        map.insert(1, 100);
        map.insert(2, 200);
        map.insert(3, 300);
        
        assert_eq!(map.get(1), Some(100));
        assert_eq!(map.get(2), Some(200));
        assert_eq!(map.get(3), Some(300));
        assert_eq!(map.get(4), None);
    }

    #[test]
    fn test_update() {
        let map = ConcurrentHashMap::new();
        
        assert_eq!(map.insert(1, 100), None);
        assert_eq!(map.insert(1, 200), Some(100));
        assert_eq!(map.get(1), Some(200));
    }

    #[test]
    fn test_remove() {
        let map = ConcurrentHashMap::new();
        
        map.insert(1, 100);
        assert_eq!(map.remove(1), Some(100));
        assert_eq!(map.get(1), None);
        assert_eq!(map.remove(1), None);
    }

    #[test]
    fn test_contains() {
        let map = ConcurrentHashMap::new();
        
        map.insert(42, 100);
        assert!(map.contains(42));
        assert!(!map.contains(43));
    }
}
