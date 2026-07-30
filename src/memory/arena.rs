//! Memory Arena - Lock-free bump allocator for zero-allocation object creation.
//! 
//! This module provides a high-performance memory arena that handles rapid tick data
//! and order book updates without triggering OS garbage collection pauses.
//! 
//! # Safety
//! This module uses `unsafe` blocks for performance-critical operations.
//! All unsafe code is heavily documented and audited for memory safety.

use std::alloc::{self, Layout};
use std::cell::UnsafeCell;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Maximum arena size (configurable, default 64MB per arena)
const DEFAULT_ARENA_SIZE: usize = 64 * 1024 * 1024;

/// Cache line size for padding to prevent false sharing
const CACHE_LINE_SIZE: usize = 64;

/// A lock-free bump allocator arena.
/// 
/// Uses atomic operations for thread-safe allocation without locks.
/// Multiple arenas can be created for different purposes (tick data, order books, etc.)
pub struct Arena {
    /// Pointer to the start of the allocated memory region
    base: NonNull<u8>,
    
    /// Current allocation offset (bump pointer)
    offset: AtomicUsize,
    
    /// Total size of the arena
    size: usize,
    
    /// Arena identifier for debugging
    id: usize,
    
    /// Padding to prevent false sharing with other memory locations
    _padding: [u8; CACHE_LINE_SIZE - (std::mem::size_of::<usize>() * 4 % CACHE_LINE_SIZE)],
}

unsafe impl Send for Arena {}
unsafe impl Sync for Arena {}

impl Arena {
    /// Create a new arena with the specified size
    pub fn new(id: usize, size: Option<usize>) -> Result<Arc<Self>, anyhow::Error> {
        let size = size.unwrap_or(DEFAULT_ARENA_SIZE);
        
        // Allocate memory using the system allocator
        let layout = Layout::from_size_align(size, alignof::<usize>())?;
        
        // SAFETY: We're allocating raw memory that we manage ourselves
        let ptr = unsafe { alloc::alloc(layout) };
        
        if ptr.is_null() {
            return Err(anyhow::anyhow!("Failed to allocate arena memory"));
        }
        
        let base = NonNull::new(ptr).ok_or_else(|| anyhow::anyhow!("Null pointer from allocator"))?;
        
        // Initialize memory to zero (optional, but helps with debugging)
        unsafe {
            std::ptr::write_bytes(ptr, 0, size);
        }
        
        Ok(Arc::new(Arena {
            base,
            offset: AtomicUsize::new(0),
            size,
            id,
            _padding: [0; CACHE_LINE_SIZE - (std::mem::size_of::<usize>() * 4 % CACHE_LINE_SIZE)],
        }))
    }
    
    /// Allocate memory from the arena with specified alignment
    /// 
    /// Returns None if the arena is exhausted
    pub fn allocate(&self, size: usize, align: usize) -> Option<NonNull<u8>> {
        // Ensure alignment is a power of 2
        debug_assert!(align.is_power_of_two());
        
        // Calculate aligned offset
        let mut current_offset = self.offset.load(Ordering::Relaxed);
        
        loop {
            // Align the current offset
            let aligned_offset = (current_offset + align - 1) & !(align - 1);
            let new_offset = aligned_offset + size;
            
            // Check if we have enough space
            if new_offset > self.size {
                return None; // Arena exhausted
            }
            
            // Try to atomically update the offset
            match self.offset.compare_exchange_weak(
                current_offset,
                new_offset,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    // Success! Return the pointer
                    let ptr = unsafe { self.base.as_ptr().add(aligned_offset) };
                    return NonNull::new(ptr);
                }
                Err(actual) => {
                    // Another thread modified the offset, retry
                    current_offset = actual;
                }
            }
        }
    }
    
    /// Allocate a specific type T from the arena
    pub fn allocate_type<T>(&self) -> Option<NonNull<T>> {
        let ptr = self.allocate(std::mem::size_of::<T>(), std::mem::align_of::<T>())?;
        Some(NonNull::new(ptr.as_ptr() as *mut T).unwrap())
    }
    
    /// Reset the arena (clear all allocations)
    /// 
    /// # Safety
    /// This invalidates all previously allocated pointers.
    /// Ensure no references to arena memory exist before calling.
    pub fn reset(&self) {
        self.offset.store(0, Ordering::Release);
    }
    
    /// Get current utilization percentage
    pub fn utilization(&self) -> f64 {
        let offset = self.offset.load(Ordering::Relaxed);
        (offset as f64) / (self.size as f64) * 100.0
    }
    
    /// Get remaining bytes available
    pub fn remaining(&self) -> usize {
        self.size - self.offset.load(Ordering::Relaxed)
    }
    
    /// Get arena ID
    pub fn id(&self) -> usize {
        self.id
    }
    
    /// Get total size
    pub fn size(&self) -> usize {
        self.size
    }
}

impl Drop for Arena {
    fn drop(&mut self) {
        // SAFETY: We deallocate the memory we allocated in new()
        let layout = Layout::from_size_align(self.size, alignof::<usize>()).unwrap();
        unsafe {
            alloc::dealloc(self.base.as_ptr(), layout);
        }
    }
}

/// Thread-local arena handle for zero-contention allocations
pub struct LocalArena {
    arena: Arc<Arena>,
}

impl LocalArena {
    pub fn new(arena: Arc<Arena>) -> Self {
        Self { arena }
    }
    
    pub fn allocate<T>(&self) -> Option<NonNull<T>> {
        self.arena.allocate_type::<T>()
    }
    
    pub fn arena(&self) -> &Arc<Arena> {
        &self.arena
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_arena_allocation() {
        let arena = Arena::new(0, Some(1024)).unwrap();
        
        // Allocate some u64 values
        let ptr1 = arena.allocate_type::<u64>().unwrap();
        let ptr2 = arena.allocate_type::<u64>().unwrap();
        
        // Verify pointers are different
        assert_ne!(ptr1.as_ptr(), ptr2.as_ptr());
        
        // Verify alignment
        assert_eq!(ptr1.as_ptr() as usize % std::mem::align_of::<u64>(), 0);
    }
    
    #[test]
    fn test_arena_exhaustion() {
        let arena = Arena::new(0, Some(64)).unwrap();
        
        // Keep allocating until exhausted
        let mut count = 0;
        while arena.allocate_type::<u64>().is_some() {
            count += 1;
        }
        
        // Should have allocated some values before exhaustion
        assert!(count > 0);
        assert!(count <= 8); // 64 bytes / 8 bytes per u64
    }
    
    #[test]
    fn test_arena_reset() {
        let arena = Arena::new(0, Some(1024)).unwrap();
        
        // Allocate until near exhaustion
        let mut ptrs = Vec::new();
        while let Some(ptr) = arena.allocate_type::<u64>() {
            ptrs.push(ptr);
            if ptrs.len() >= 10 {
                break;
            }
        }
        
        let utilization_before = arena.utilization();
        arena.reset();
        let utilization_after = arena.utilization();
        
        assert!(utilization_before > 0.0);
        assert_eq!(utilization_after, 0.0);
    }
}
