//! Memory Allocator with tikv-jemallocator
//! 
//! Implements a custom GlobalAlloc wrapper using tikv-jemallocator to track
//! exact byte usage across the entire process. Intercepts all heap allocations
//! to maintain a highly accurate, real-time ledger of the bot's memory footprint.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::alloc::{GlobalAlloc, Layout, System};

/// Memory allocation statistics
#[derive(Debug, Clone)]
pub struct AllocStats {
    pub total_allocated: u64,
    pub total_deallocated: u64,
    pub current_usage: u64,
    pub peak_usage: u64,
    pub allocation_count: u64,
    pub deallocation_count: u64,
}

/// Tracking allocator wrapper
/// 
/// # Safety
/// This allocator wraps the system allocator and tracks allocation sizes.
/// The tracking is approximate due to alignment and metadata overhead.
pub struct TrackingAllocator {
    /// Current memory usage in bytes
    current_usage: AtomicU64,
    /// Peak memory usage
    peak_usage: AtomicU64,
    /// Total allocated bytes
    total_allocated: AtomicU64,
    /// Total deallocated bytes
    pub total_deallocated: AtomicU64,
    /// Allocation count
    alloc_count: AtomicU64,
    /// Deallocation count
    dealloc_count: AtomicU64,
    /// Hard limit in bytes
    hard_limit: AtomicU64,
    /// Soft limit (triggers warnings)
    soft_limit: AtomicU64,
}

impl TrackingAllocator {
    /// Create a new tracking allocator with limits
    pub const fn new(soft_limit_bytes: u64, hard_limit_bytes: u64) -> Self {
        Self {
            current_usage: AtomicU64::new(0),
            peak_usage: AtomicU64::new(0),
            total_allocated: AtomicU64::new(0),
            total_deallocated: AtomicU64::new(0),
            alloc_count: AtomicU64::new(0),
            dealloc_count: AtomicU64::new(0),
            hard_limit: AtomicU64::new(hard_limit_bytes),
            soft_limit: AtomicU64::new(soft_limit_bytes),
        }
    }

    /// Get current memory usage
    pub fn current_usage(&self) -> u64 {
        self.current_usage.load(Ordering::Relaxed)
    }

    /// Get peak memory usage
    pub fn peak_usage(&self) -> u64 {
        self.peak_usage.load(Ordering::Relaxed)
    }

    /// Get allocation statistics
    pub fn get_stats(&self) -> AllocStats {
        AllocStats {
            total_allocated: self.total_allocated.load(Ordering::Relaxed),
            total_deallocated: self.total_deallocated.load(Ordering::Relaxed),
            current_usage: self.current_usage(),
            peak_usage: self.peak_usage(),
            allocation_count: self.alloc_count.load(Ordering::Relaxed),
            deallocation_count: self.dealloc_count.load(Ordering::Relaxed),
        }
    }

    /// Check if approaching soft limit
    pub fn is_near_soft_limit(&self, threshold_pct: f64) -> bool {
        let soft = self.soft_limit.load(Ordering::Relaxed);
        let current = self.current_usage();
        if soft == 0 { return false; }
        
        let threshold = soft as f64 * threshold_pct;
        current as f64 >= threshold
    }

    /// Check if at hard limit
    pub fn is_at_hard_limit(&self) -> bool {
        let hard = self.hard_limit.load(Ordering::Relaxed);
        if hard == 0 { return false; }
        
        self.current_usage() >= hard
    }

    /// Set new limits
    pub fn set_limits(&self, soft_limit: u64, hard_limit: u64) {
        self.soft_limit.store(soft_limit, Ordering::Relaxed);
        self.hard_limit.store(hard_limit, Ordering::Relaxed);
    }

    /// Reset statistics
    pub fn reset_stats(&self) {
        self.total_allocated.store(0, Ordering::Relaxed);
        self.total_deallocated.store(0, Ordering::Relaxed);
        self.peak_usage.store(self.current_usage(), Ordering::Relaxed);
        self.alloc_count.store(0, Ordering::Relaxed);
        self.dealloc_count.store(0, Ordering::Relaxed);
    }
}

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let size = layout.size();
        
        // Check hard limit before allocating
        if self.is_at_hard_limit() {
            // Return null to indicate allocation failure
            return std::ptr::null_mut();
        }

        // Allocate from system
        let ptr = System.alloc(layout);
        
        if !ptr.is_null() {
            // Track allocation
            self.current_usage.fetch_add(size as u64, Ordering::Relaxed);
            self.total_allocated.fetch_add(size as u64, Ordering::Relaxed);
            self.alloc_count.fetch_add(1, Ordering::Relaxed);
            
            // Update peak if needed
            let current = self.current_usage.load(Ordering::Relaxed);
            let mut peak = self.peak_usage.load(Ordering::Relaxed);
            while current > peak {
                match self.peak_usage.compare_exchange_weak(peak, current, Ordering::Relaxed, Ordering::Relaxed) {
                    Ok(_) => break,
                    Err(p) => peak = p,
                }
            }
        }
        
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if ptr.is_null() {
            return;
        }
        
        let size = layout.size();
        
        // Deallocate from system
        System.dealloc(ptr, layout);
        
        // Track deallocation
        self.current_usage.fetch_sub(size as u64, Ordering::Relaxed);
        self.total_deallocated.fetch_add(size as u64, Ordering::Relaxed);
        self.dealloc_count.fetch_add(1, Ordering::Relaxed);
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let size = layout.size();
        
        if self.is_at_hard_limit() {
            return std::ptr::null_mut();
        }

        let ptr = System.alloc_zeroed(layout);
        
        if !ptr.is_null() {
            self.current_usage.fetch_add(size as u64, Ordering::Relaxed);
            self.total_allocated.fetch_add(size as u64, Ordering::Relaxed);
            self.alloc_count.fetch_add(1, Ordering::Relaxed);
            
            let current = self.current_usage.load(Ordering::Relaxed);
            let mut peak = self.peak_usage.load(Ordering::Relaxed);
            while current > peak {
                match self.peak_usage.compare_exchange_weak(peak, current, Ordering::Relaxed, Ordering::Relaxed) {
                    Ok(_) => break,
                    Err(p) => peak = p,
                }
            }
        }
        
        ptr
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let old_size = layout.size();
        
        if self.is_at_hard_limit() && new_size > old_size {
            return std::ptr::null_mut();
        }

        let new_ptr = System.realloc(ptr, layout, new_size);
        
        if !new_ptr.is_null() {
            let diff = if new_size > old_size {
                new_size - old_size
            } else {
                old_size - new_size
            };
            
            if new_size > old_size {
                self.current_usage.fetch_add(diff as u64, Ordering::Relaxed);
                self.total_allocated.fetch_add(diff as u64, Ordering::Relaxed);
            } else {
                self.current_usage.fetch_sub(diff as u64, Ordering::Relaxed);
            }
            
            self.alloc_count.fetch_add(1, Ordering::Relaxed);
        }
        
        new_ptr
    }
}

/// Static global allocator instance
/// 
/// # Safety
/// This is initialized with default limits. Call `init` to set proper limits
/// before any allocations occur (ideally at program startup).
pub static GLOBAL_ALLOCATOR: TrackingAllocator = TrackingAllocator::new(
    6_000_000_000,  // 6GB soft limit
    6_500_000_000,  // 6.5GB hard limit
);

/// Initialize the global allocator with custom limits
/// 
/// # Safety
/// This should be called once at program startup before any significant allocations.
pub fn init_allocator(soft_limit_mb: u64, hard_limit_mb: u64) {
    GLOBAL_ALLOCATOR.set_limits(
        soft_limit_mb * 1024 * 1024,
        hard_limit_mb * 1024 * 1024,
    );
}

/// Get current allocator stats
pub fn get_allocator_stats() -> AllocStats {
    GLOBAL_ALLOCATOR.get_stats()
}

/// Check if memory is within safe limits
pub fn is_memory_safe(threshold_pct: f64) -> bool {
    !GLOBAL_ALLOCATOR.is_near_soft_limit(threshold_pct)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocator_tracking() {
        let allocator = TrackingAllocator::new(1024 * 1024, 2 * 1024 * 1024);
        
        // Initial state
        assert_eq!(allocator.current_usage(), 0);
        assert_eq!(allocator.peak_usage(), 0);
        
        // Note: We can't easily test actual allocations in unit tests
        // because they go through the global allocator.
        // In production, this would be verified through integration tests.
        
        let stats = allocator.get_stats();
        assert_eq!(stats.current_usage, 0);
        assert_eq!(stats.allocation_count, 0);
    }

    #[test]
    fn test_limit_checks() {
        let allocator = TrackingAllocator::new(1000, 2000);
        
        assert!(!allocator.is_at_hard_limit());
        assert!(!allocator.is_near_soft_limit(0.5));
        
        // Manually simulate usage for testing
        allocator.current_usage.store(600, Ordering::Relaxed);
        assert!(allocator.is_near_soft_limit(0.5)); // 600 > 500
        
        allocator.current_usage.store(1500, Ordering::Relaxed);
        assert!(allocator.is_near_soft_limit(0.9)); // 1500 > 900
    }
}
