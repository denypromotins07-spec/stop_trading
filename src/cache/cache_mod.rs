//! Cache Optimization Module Root
//! 
//! Enforces strict memory layout rules across the entire Rust codebase.

pub mod line;
pub mod prefetch;

use line::{CACHE_LINE_SIZE, Aligned, CacheAligned};
use prefetch::PriceLevel as PrefetchPriceLevel;

/// Re-export cache utilities for convenience
pub use line::{
    cache_line_padding,
    align_to_cache_line,
    AtomicCounter,
    CacheSeparated,
    QueueSlot,
};

pub use prefetch::{
    prefetch,
    prefetch_t0,
    prefetch_t1,
    prefetch_t2,
    PrefetchHint,
    PrefetchIterator,
    OrderBookTraversal,
};

/// Unified price level structure that is both cache-aligned and prefetch-optimized
#[repr(align(64))]
#[derive(Debug, Clone, Copy, Default)]
pub struct CacheOptimizedPriceLevel {
    pub price: u64,
    pub quantity: u64,
    pub order_count: u32,
    pub flags: u32,
    /// Explicit padding to ensure full cache line
    _padding: [u8; 32], // Already 24 bytes used, need 40 more for 64, but repr(align) handles it
}

impl CacheOptimizedPriceLevel {
    pub fn new(price: u64, quantity: u64, order_count: u32) -> Self {
        Self {
            price,
            quantity,
            order_count,
            flags: 0,
            _padding: [0; 32],
        }
    }

    /// Convert from prefetch PriceLevel
    pub fn from_prefetch(other: &PrefetchPriceLevel) -> Self {
        Self {
            price: other.price,
            quantity: other.quantity,
            order_count: other.order_count,
            flags: other.flags,
            _padding: [0; 32],
        }
    }

    /// Convert to prefetch PriceLevel
    pub fn to_prefetch(&self) -> PrefetchPriceLevel {
        PrefetchPriceLevel {
            price: self.price,
            quantity: self.quantity,
            order_count: self.order_count,
            flags: self.flags,
        }
    }
}

unsafe impl CacheAligned for CacheOptimizedPriceLevel {}

/// Cache-optimized order book snapshot
#[repr(align(64))]
pub struct CacheOptimizedOrderBook {
    /// Number of bid levels
    pub num_bids: usize,
    /// Number of ask levels  
    pub num_asks: usize,
    /// Timestamp in nanoseconds
    pub timestamp_ns: u64,
    /// Sequence number
    pub sequence: u64,
    /// Bid levels (cache-line aligned)
    pub bids: Vec<Aligned<CacheOptimizedPriceLevel>>,
    /// Ask levels (cache-line aligned)
    pub asks: Vec<Aligned<CacheOptimizedPriceLevel>>,
}

impl CacheOptimizedOrderBook {
    pub fn new(num_levels: usize) -> Self {
        Self {
            num_bids: 0,
            num_asks: 0,
            timestamp_ns: 0,
            sequence: 0,
            bids: Vec::with_capacity(num_levels),
            asks: Vec::with_capacity(num_levels),
        }
    }

    /// Add a bid level
    pub fn add_bid(&mut self, price: u64, quantity: u64, order_count: u32) {
        let level = CacheOptimizedPriceLevel::new(price, quantity, order_count);
        self.bids.push(Aligned::new(level));
        self.num_bids = self.bids.len();
    }

    /// Add an ask level
    pub fn add_ask(&mut self, price: u64, quantity: u64, order_count: u32) {
        let level = CacheOptimizedPriceLevel::new(price, quantity, order_count);
        self.asks.push(Aligned::new(level));
        self.num_asks = self.asks.len();
    }

    /// Clear the book
    pub fn clear(&mut self) {
        self.bids.clear();
        self.asks.clear();
        self.num_bids = 0;
        self.num_asks = 0;
    }

    /// Get best bid price
    pub fn best_bid(&self) -> Option<u64> {
        self.bids.first().map(|l| l.data.price)
    }

    /// Get best ask price
    pub fn best_ask(&self) -> Option<u64> {
        self.asks.first().map(|l| l.data.price)
    }

    /// Get mid price
    pub fn mid_price(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some((bid as f64 + ask as f64) / 2.0),
            _ => None,
        }
    }

    /// Get spread in ticks
    pub fn spread(&self) -> Option<u64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) if ask > bid => Some(ask - bid),
            _ => None,
        }
    }
}

/// Memory pool for cache-aligned allocations
pub struct CacheAlignedPool<T> {
    items: Vec<Aligned<T>>,
    free_list: Vec<usize>,
    capacity: usize,
}

impl<T: Default> CacheAlignedPool<T> {
    pub fn new(capacity: usize) -> Self {
        let mut items = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            items.push(Aligned::new(T::default()));
        }

        let mut free_list = Vec::with_capacity(capacity);
        for i in 0..capacity {
            free_list.push(i);
        }

        Self {
            items,
            free_list,
            capacity,
        }
    }

    /// Allocate an item from the pool
    pub fn allocate(&mut self) -> Option<usize> {
        self.free_list.pop()
    }

    /// Return an item to the pool
    pub fn deallocate(&mut self, index: usize) {
        if index < self.capacity {
            self.items[index] = Aligned::new(T::default());
            self.free_list.push(index);
        }
    }

    /// Get mutable reference to item
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        self.items.get_mut(index).map(|a| &mut a.data)
    }

    /// Get immutable reference to item
    pub fn get(&self, index: usize) -> Option<&T> {
        self.items.get(index).map(|a| &a.data)
    }

    /// Check if pool has available items
    pub fn has_available(&self) -> bool {
        !self.free_list.is_empty()
    }

    /// Get number of available items
    pub fn available_count(&self) -> usize {
        self.free_list.len()
    }
}

/// Validate that a type meets cache optimization requirements
pub trait CacheOptimized: CacheAligned + Sized {
    /// Verify the type size doesn't cause excessive padding waste
    fn verify_efficiency() -> bool {
        let size = core::mem::size_of::<Self>();
        let padding = CACHE_LINE_SIZE - (size % CACHE_LINE_SIZE);
        // Warn if more than half a cache line is wasted
        padding <= CACHE_LINE_SIZE / 2
    }
}

impl<T: CacheAligned + Sized> CacheOptimized for T {}

/// Runtime check for cache line alignment of a pointer
pub fn verify_alignment<T>(ptr: *const T) -> bool {
    (ptr as usize) % CACHE_LINE_SIZE == 0
}

/// Calculate total memory footprint including padding
pub fn calculate_footprint<T>() -> usize {
    align_to_cache_line(core::mem::size_of::<T>())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_optimized_price_level() {
        let level = CacheOptimizedPriceLevel::new(50000, 1000, 5);
        
        // Verify alignment
        let ptr = &level as *const _ as usize;
        assert_eq!(ptr % CACHE_LINE_SIZE, 0);
        
        // Verify conversion
        let prefetch_level = level.to_prefetch();
        assert_eq!(prefetch_level.price, 50000);
    }

    #[test]
    fn test_order_book() {
        let mut book = CacheOptimizedOrderBook::new(10);
        
        book.add_bid(99000, 1000, 5);
        book.add_bid(98000, 500, 3);
        book.add_ask(100000, 800, 4);
        
        assert_eq!(book.best_bid(), Some(99000));
        assert_eq!(book.best_ask(), Some(100000));
        assert_eq!(book.spread(), Some(1000));
    }

    #[test]
    fn test_memory_pool() {
        #[derive(Default, Clone)]
        struct TestItem {
            value: u64,
        }

        let mut pool = CacheAlignedPool::<TestItem>::new(5);
        
        assert_eq!(pool.available_count(), 5);
        
        let idx = pool.allocate();
        assert!(idx.is_some());
        assert_eq!(pool.available_count(), 4);
        
        if let Some(i) = idx {
            if let Some(item) = pool.get_mut(i) {
                item.value = 42;
            }
            assert_eq!(pool.get(i).unwrap().value, 42);
            
            pool.deallocate(i);
            assert_eq!(pool.available_count(), 5);
        }
    }

    #[test]
    fn test_alignment_verification() {
        let item = Aligned::new(42u64);
        assert!(verify_alignment(&item));
    }
}
