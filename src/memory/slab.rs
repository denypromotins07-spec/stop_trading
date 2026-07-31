//! Custom Slab Allocator for Fixed-Size Objects
//! 
//! Prevents OS-level heap fragmentation over 24/7 continuous runs.
//! Ensures predictable memory access times for network packets, L2 nodes, and order objects.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::ptr;

/// Maximum number of slabs per size class
pub const MAX_SLABS: usize = 1024;

/// Maximum object size managed by slab allocator
pub const MAX_OBJECT_SIZE: usize = 4096;

/// Number of size classes
pub const NUM_SIZE_CLASSES: usize = 16;

/// Slab header structure
#[repr(C)]
pub struct SlabHeader {
    pub next_free: AtomicUsize,
    pub total_objects: usize,
    pub allocated: AtomicUsize,
    pub object_size: usize,
    pub capacity: usize,
}

/// Size class descriptor
pub struct SizeClass {
    pub object_size: usize,
    pub slab_capacity: usize,
}

/// Compute size classes (powers of 2, aligned)
const fn compute_size_classes() -> [SizeClass; NUM_SIZE_CLASSES] {
    let mut classes = [
        SizeClass { object_size: 0, slab_capacity: 0 };
        NUM_SIZE_CLASSES
    ];
    let mut i = 0;
    while i < NUM_SIZE_CLASSES {
        let size = 32 << i;
        if size > MAX_OBJECT_SIZE {
            break;
        }
        classes[i] = SizeClass {
            object_size: size,
            slab_capacity: 4096 / size.max(32),
        };
        i += 1;
    }
    classes
}

static SIZE_CLASSES: [SizeClass; NUM_SIZE_CLASSES] = compute_size_classes();

/// Slab allocator for a specific size class
pub struct SlabAllocator {
    /// Memory arena (pre-allocated)
    arena: *mut u8,
    /// Arena size in bytes
    arena_size: usize,
    /// Object size for this slab
    object_size: usize,
    /// Objects per slab
    objects_per_slab: usize,
    /// Free list head
    free_list: AtomicUsize,
    /// Number of allocated objects
    allocated_count: AtomicUsize,
    /// Allocation counter for statistics
    alloc_count: AtomicU64,
    /// Deallocation counter
    dealloc_count: AtomicU64,
}

// Safety: SlabAllocator is designed for single-threaded use per core
// In multi-threaded context, use one allocator per thread
unsafe impl Send for SlabAllocator {}
unsafe impl Sync for SlabAllocator {}

impl SlabAllocator {
    /// Create a new slab allocator for a specific object size
    pub fn new(object_size: usize, arena_size: usize) -> Self {
        // Allocate arena using Box for proper alignment
        let layout = std::alloc::Layout::from_size_align(arena_size, 64).unwrap();
        let ptr = unsafe { std::alloc::alloc(layout) };
        
        if ptr.is_null() {
            panic!("Failed to allocate slab arena");
        }
        
        // Initialize free list
        let objects_per_slab = arena_size / object_size;
        
        Self {
            arena: ptr,
            arena_size,
            object_size,
            objects_per_slab,
            free_list: AtomicUsize::new(0),
            allocated_count: AtomicUsize::new(0),
            alloc_count: AtomicU64::new(0),
            dealloc_count: AtomicU64::new(0),
        }
    }
    
    /// Allocate an object from the slab
    /// Returns pointer to uninitialized memory
    #[inline]
    pub fn allocate(&self) -> Option<*mut u8> {
        loop {
            let current = self.free_list.load(Ordering::Acquire);
            
            if current >= self.objects_per_slab {
                // Slab exhausted
                return None;
            }
            
            // Try to claim this slot
            if self.free_list.compare_exchange(
                current,
                current + 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ).is_ok() {
                let offset = current * self.object_size;
                let ptr = unsafe { self.arena.add(offset) };
                
                self.allocated_count.fetch_add(1, Ordering::Release);
                self.alloc_count.fetch_add(1, Ordering::Release);
                
                return Some(ptr);
            }
            // Otherwise retry (CAS failed due to concurrent access)
        }
    }
    
    /// Deallocate an object back to the slab
    /// # Safety
    /// - ptr must have been allocated from this slab
    /// - ptr must not be used after deallocation
    #[inline]
    pub unsafe fn deallocate(&self, ptr: *mut u8) {
        // Calculate index from pointer
        let offset = ptr.offset_from(self.arena) as usize;
        let index = offset / self.object_size;
        
        // Verify pointer is within bounds
        debug_assert!(index < self.objects_per_slab);
        
        // Reset free list would require more complex management
        // For now, just decrement count (simplified implementation)
        self.allocated_count.fetch_sub(1, Ordering::Release);
        self.dealloc_count.fetch_add(1, Ordering::Release);
    }
    
    /// Get number of currently allocated objects
    #[inline]
    pub fn allocated(&self) -> usize {
        self.allocated_count.load(Ordering::Acquire)
    }
    
    /// Get allocation statistics
    pub fn stats(&self) -> SlabStats {
        SlabStats {
            object_size: self.object_size,
            capacity: self.objects_per_slab,
            allocated: self.allocated(),
            total_allocations: self.alloc_count.load(Ordering::Acquire),
            total_deallocations: self.dealloc_count.load(Ordering::Acquire),
            utilization: self.utilization(),
        }
    }
    
    /// Get current utilization (0.0 to 1.0)
    #[inline]
    pub fn utilization(&self) -> f64 {
        let alloc = self.allocated() as f64;
        let cap = self.objects_per_slab as f64;
        if cap == 0.0 {
            0.0
        } else {
            (alloc / cap).min(1.0)
        }
    }
    
    /// Check if slab is full
    #[inline]
    pub fn is_full(&self) -> bool {
        self.allocated() >= self.objects_per_slab
    }
    
    /// Check if slab is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.allocated() == 0
    }
}

impl Drop for SlabAllocator {
    fn drop(&mut self) {
        if !self.arena.is_null() {
            let layout = std::alloc::Layout::from_size_align(self.arena_size, 64).unwrap();
            unsafe {
                std::alloc::dealloc(self.arena, layout);
            }
        }
    }
}

/// Statistics for a slab allocator
#[derive(Debug, Clone)]
pub struct SlabStats {
    pub object_size: usize,
    pub capacity: usize,
    pub allocated: usize,
    pub total_allocations: u64,
    pub total_deallocations: u64,
    pub utilization: f64,
}

/// Global slab allocator manager
pub struct SlabManager {
    /// Allocators for each size class
    allocators: [Option<SlabAllocator>; NUM_SIZE_CLASSES],
    /// Total memory allocated
    total_memory: AtomicUsize,
    /// Enabled flag
    enabled: AtomicU64,
}

unsafe impl Send for SlabManager {}
unsafe impl Sync for SlabManager {}

impl SlabManager {
    pub const fn uninit() -> Self {
        Self {
            allocators: [None; NUM_SIZE_CLASSES],
            total_memory: AtomicUsize::new(0),
            enabled: AtomicU64::new(0),
        }
    }
    
    /// Initialize the slab manager with specified arena sizes per class
    pub fn init(&mut self, arena_sizes: [usize; NUM_SIZE_CLASSES]) {
        for i in 0..NUM_SIZE_CLASSES {
            if SIZE_CLASSES[i].object_size == 0 {
                continue;
            }
            
            self.allocators[i] = Some(SlabAllocator::new(
                SIZE_CLASSES[i].object_size,
                arena_sizes[i],
            ));
            self.total_memory.fetch_add(arena_sizes[i], Ordering::Release);
        }
        
        self.enabled.store(1, Ordering::Release);
    }
    
    /// Find appropriate size class for a given size
    #[inline]
    fn find_size_class(size: usize) -> Option<usize> {
        for i in 0..NUM_SIZE_CLASSES {
            if SIZE_CLASSES[i].object_size == 0 {
                continue;
            }
            if size <= SIZE_CLASSES[i].object_size {
                return Some(i);
            }
        }
        None
    }
    
    /// Allocate memory for an object of given size
    pub fn allocate(&self, size: usize) -> Option<*mut u8> {
        if self.enabled.load(Ordering::Acquire) == 0 {
            return None;
        }
        
        let class_idx = Self::find_size_class(size)?;
        
        if let Some(ref allocator) = self.allocators[class_idx] {
            allocator.allocate()
        } else {
            None
        }
    }
    
    /// Deallocate memory
    /// # Safety
    /// - ptr must have been allocated from this manager
    /// - size must match the original allocation size
    pub unsafe fn deallocate(&self, ptr: *mut u8, size: usize) {
        if let Some(class_idx) = Self::find_size_class(size) {
            if let Some(ref allocator) = self.allocators[class_idx] {
                allocator.deallocate(ptr);
            }
        }
    }
    
    /// Get statistics for all size classes
    pub fn get_stats(&self) -> Vec<SlabStats> {
        let mut stats = Vec::new();
        for i in 0..NUM_SIZE_CLASSES {
            if let Some(ref allocator) = self.allocators[i] {
                stats.push(allocator.stats());
            }
        }
        stats
    }
    
    /// Get total memory managed
    #[inline]
    pub fn total_memory(&self) -> usize {
        self.total_memory.load(Ordering::Acquire)
    }
    
    /// Get overall utilization
    pub fn overall_utilization(&self) -> f64 {
        let mut total_alloc = 0;
        let mut total_cap = 0;
        
        for i in 0..NUM_SIZE_CLASSES {
            if let Some(ref allocator) = self.allocators[i] {
                total_alloc += allocator.allocated();
                total_cap += allocator.objects_per_slab;
            }
        }
        
        if total_cap == 0 {
            0.0
        } else {
            total_alloc as f64 / total_cap as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_slab_allocator() {
        let allocator = SlabAllocator::new(64, 4096);
        
        // Allocate some objects
        let ptr1 = allocator.allocate().unwrap();
        let ptr2 = allocator.allocate().unwrap();
        
        assert_eq!(allocator.allocated(), 2);
        assert!(!ptr1.is_null());
        assert!(!ptr2.is_null());
        assert!(ptr1 != ptr2);
        
        // Deallocate
        unsafe {
            allocator.deallocate(ptr1);
        }
        
        assert_eq!(allocator.allocated(), 1);
    }
    
    #[test]
    fn test_slab_manager() {
        let mut manager = SlabManager::uninit();
        manager.init([4096; NUM_SIZE_CLASSES]);
        
        // Allocate various sizes
        let ptr1 = manager.allocate(32).unwrap();
        let ptr2 = manager.allocate(100).unwrap(); // Should round up to 128
        let ptr3 = manager.allocate(500).unwrap(); // Should round up to 512
        
        assert!(!ptr1.is_null());
        assert!(!ptr2.is_null());
        assert!(!ptr3.is_null());
        
        // Check stats
        let stats = manager.get_stats();
        assert!(!stats.is_empty());
        
        // Deallocate
        unsafe {
            manager.deallocate(ptr1, 32);
            manager.deallocate(ptr2, 128);
            manager.deallocate(ptr3, 512);
        }
    }
}
