//! Advanced Memory Module Root
//! 
//! Wires the Slab allocator into global object pools and arenas.

pub mod slab;
pub mod defrag;

pub use slab::{SlabAllocator, SlabManager, SlabStats, SizeClass, NUM_SIZE_CLASSES};
pub use defrag::{Defragmentor, DefragState, DefragStats, DefragResult, MemoryRegion, calculate_fragmentation};

/// Global memory pool combining slab allocation with defragmentation
pub struct MemoryPool {
    pub slab_manager: SlabManager,
    pub defragmentor: Defragmentor,
    /// Total memory budget (bytes)
    memory_budget: usize,
    /// Current memory usage
    current_usage: std::sync::atomic::AtomicUsize,
}

unsafe impl Send for MemoryPool {}
unsafe impl Sync for MemoryPool {}

impl MemoryPool {
    /// Create a new memory pool with specified budget
    pub fn new(memory_budget_bytes: usize) -> Self {
        let mut slab_manager = SlabManager::uninit();
        
        // Calculate arena sizes per class (distribute budget)
        let mut arena_sizes = [0usize; NUM_SIZE_CLASSES];
        let per_class = memory_budget_bytes / NUM_SIZE_CLASSES;
        for i in 0..NUM_SIZE_CLASSES {
            arena_sizes[i] = per_class;
        }
        
        slab_manager.init(arena_sizes);
        
        Self {
            slab_manager,
            defragmentor: Defragmentor::new(),
            memory_budget: memory_budget_bytes,
            current_usage: std::sync::atomic::AtomicUsize::new(0),
        }
    }
    
    /// Allocate an object of given size
    pub fn allocate(&self, size: usize) -> Option<*mut u8> {
        if self.current_usage.load(std::sync::atomic::Ordering::Acquire) + size > self.memory_budget {
            return None; // Budget exceeded
        }
        
        if let Some(ptr) = self.slab_manager.allocate(size) {
            self.current_usage.fetch_add(size, std::sync::atomic::Ordering::Release);
            Some(ptr)
        } else {
            None
        }
    }
    
    /// Deallocate an object
    /// # Safety
    /// - ptr must have been allocated from this pool
    /// - size must match original allocation
    pub unsafe fn deallocate(&self, ptr: *mut u8, size: usize) {
        self.slab_manager.deallocate(ptr, size);
        self.current_usage.fetch_sub(size, std::sync::atomic::Ordering::Release);
    }
    
    /// Update market volatility for defrag scheduling
    #[inline]
    pub fn update_volatility(&self, vol_bps: f64) {
        self.defragmentor.update_volatility(vol_bps);
    }
    
    /// Run defragmentation cycle if conditions are favorable
    pub fn maybe_defrag(&self) -> Option<DefragResult> {
        if self.defragmentor.should_defrag() {
            Some(self.defragmentor.run_cycle())
        } else {
            None
        }
    }
    
    /// Get memory utilization (0.0 to 1.0)
    pub fn utilization(&self) -> f64 {
        let used = self.current_usage.load(std::sync::atomic::Ordering::Acquire);
        if self.memory_budget == 0 {
            return 0.0;
        }
        (used as f64 / self.memory_budget as f64).min(1.0)
    }
    
    /// Get remaining budget
    pub fn remaining_budget(&self) -> usize {
        self.memory_budget - self.current_usage.load(std::sync::atomic::Ordering::Acquire)
    }
    
    /// Get combined statistics
    pub fn get_stats(&self) -> PoolStats {
        PoolStats {
            memory_budget: self.memory_budget,
            current_usage: self.current_usage.load(std::sync::atomic::Ordering::Acquire),
            utilization: self.utilization(),
            slab_stats: self.slab_manager.get_stats(),
            defrag_stats: self.defragmentor.get_stats(),
        }
    }
    
    /// Check if approaching memory limit
    pub fn is_near_limit(&self, threshold: f64) -> bool {
        self.utilization() > threshold
    }
}

/// Combined pool statistics
#[derive(Debug, Clone)]
pub struct PoolStats {
    pub memory_budget: usize,
    pub current_usage: usize,
    pub utilization: f64,
    pub slab_stats: Vec<SlabStats>,
    pub defrag_stats: DefragStats,
}

/// Object pool for specific type T using slab allocation
pub struct ObjectPool<T> {
    slab: SlabAllocator,
    _marker: std::marker::PhantomData<T>,
}

impl<T> ObjectPool<T> {
    /// Create a new object pool
    pub fn new(capacity: usize) -> Self {
        let object_size = std::mem::size_of::<T>().max(32);
        let arena_size = object_size * capacity;
        
        Self {
            slab: SlabAllocator::new(object_size, arena_size),
            _marker: std::marker::PhantomData,
        }
    }
    
    /// Allocate an uninitialized object
    pub fn allocate(&self) -> Option<*mut T> {
        self.slab.allocate().map(|ptr| ptr as *mut T)
    }
    
    /// Deallocate an object
    /// # Safety
    /// - ptr must have been allocated from this pool
    pub unsafe fn deallocate(&self, ptr: *mut T) {
        self.slab.deallocate(ptr as *mut u8);
    }
    
    /// Allocate and initialize with a value
    pub fn allocate_with(&self, value: T) -> Option<*mut T> {
        self.allocate().map(|ptr| {
            unsafe {
                ptr.write(value);
            }
            ptr
        })
    }
    
    /// Get pool statistics
    pub fn stats(&self) -> SlabStats {
        self.slab.stats()
    }
    
    /// Get number of allocated objects
    pub fn len(&self) -> usize {
        self.slab.allocated()
    }
    
    /// Check if pool is empty
    pub fn is_empty(&self) -> bool {
        self.slab.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_memory_pool() {
        // Create pool with 1MB budget
        let pool = MemoryPool::new(1024 * 1024);
        
        // Allocate some memory
        let ptr1 = pool.allocate(64).unwrap();
        let ptr2 = pool.allocate(128).unwrap();
        
        assert!(pool.utilization() > 0.0);
        assert!(pool.remaining_budget() < 1024 * 1024);
        
        // Deallocate
        unsafe {
            pool.deallocate(ptr1, 64);
            pool.deallocate(ptr2, 128);
        }
        
        // Usage should be back to near zero
        assert!(pool.utilization() < 0.01);
    }
    
    #[test]
    fn test_object_pool() {
        #[derive(Debug, Clone)]
        struct TestObject {
            id: u64,
            data: [u8; 32],
        }
        
        let pool = ObjectPool::<TestObject>::new(100);
        
        // Allocate and initialize
        let obj = pool.allocate_with(TestObject { id: 1, data: [0; 32] }).unwrap();
        
        assert_eq!(pool.len(), 1);
        
        // Read back
        unsafe {
            assert_eq!((*obj).id, 1);
        }
        
        // Deallocate
        unsafe {
            pool.deallocate(obj);
        }
        
        assert_eq!(pool.len(), 0);
    }
    
    #[test]
    fn test_memory_budget_enforcement() {
        let pool = MemoryPool::new(256); // Small budget
        
        // Allocate until budget exhausted
        let mut ptrs = Vec::new();
        for _ in 0..10 {
            if let Some(ptr) = pool.allocate(32) {
                ptrs.push(ptr);
            } else {
                break;
            }
        }
        
        // Should have allocated about 8 objects (256/32)
        assert!(ptrs.len() <= 8);
        
        // Next allocation should fail
        assert!(pool.allocate(32).is_none());
    }
}
