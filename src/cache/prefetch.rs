//! Software Prefetching Hints for Order Book Traversals
//! 
//! Uses core::arch::x86_64::_mm_prefetch to load L2 price levels into L1 cache.
//! Includes #[cfg(target_arch = "x86_64")] guards and safe fallbacks.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64;

/// Prefetch hint types corresponding to _MM_HINT values
#[derive(Debug, Clone, Copy)]
pub enum PrefetchHint {
    /// Prefetch to all cache levels (T0)
    T0,
    /// Prefetch to L2 and higher (T1)
    T1,
    /// Prefetch to L3 and higher or cached in DRAM (T2)
    T2,
    /// Prepare to store (write prefetch)
    Write,
    /// Non-temporal prefetch (NTA)
    NTA,
}

impl PrefetchHint {
    #[inline]
    #[cfg(target_arch = "x86_64")]
    fn as_hint(self) -> i32 {
        match self {
            PrefetchHint::T0 => x86_64::_MM_HINT_T0,
            PrefetchHint::T1 => x86_64::_MM_HINT_T1,
            PrefetchHint::T2 => x86_64::_MM_HINT_T2,
            PrefetchHint::Write => x86_64::_MM_HINT_ET0, // Exclusive T0 for writes
            PrefetchHint::NTA => x86_64::_MM_HINT_NTA,
        }
    }
}

/// Prefetch data at the given address with specified hint
/// 
/// # Safety
/// The pointer must be valid for reading (or writing for Write hint)
#[inline]
pub unsafe fn prefetch<T>(ptr: *const T, hint: PrefetchHint) {
    #[cfg(target_arch = "x86_64")]
    {
        let hint_val = hint.as_hint();
        x86_64::_mm_prefetch(ptr as *const _, hint_val);
    }
    
    // On non-x86_64 architectures, this is a no-op
    // The compiler may still optimize based on access patterns
    #[allow(clippy::let_unit_value)]
    let _ = ptr;
    let _ = hint;
}

/// Prefetch with T0 hint (most aggressive - all cache levels)
#[inline]
pub unsafe fn prefetch_t0<T>(ptr: *const T) {
    prefetch(ptr, PrefetchHint::T0);
}

/// Prefetch with T1 hint (L2 and higher)
#[inline]
pub unsafe fn prefetch_t1<T>(ptr: *const T) {
    prefetch(ptr, PrefetchHint::T1);
}

/// Prefetch with T2 hint (L3 and higher, more conservative)
#[inline]
pub unsafe fn prefetch_t2<T>(ptr: *const T) {
    prefetch(ptr, PrefetchHint::T2);
}

/// Prefetch for writing (exclusive cache line)
#[inline]
pub unsafe fn prefetch_write<T>(ptr: *mut T) {
    #[cfg(target_arch = "x86_64")]
    {
        x86_64::_mm_prefetch(ptr as *const _, x86_64::_MM_HINT_ET0);
    }
    #[allow(clippy::let_unit_value)]
    let _ = ptr;
}

/// Prefetch multiple elements ahead in an array
/// 
/// # Arguments
/// * `base_ptr` - Base pointer to array
/// * `index` - Current index being processed
/// * `ahead` - Number of elements to prefetch ahead
/// * `hint` - Prefetch hint type
#[inline]
pub unsafe fn prefetch_ahead<T>(base_ptr: *const T, index: usize, ahead: usize, hint: PrefetchHint) {
    let prefetch_ptr = base_ptr.add(index + ahead);
    prefetch(prefetch_ptr, hint);
}

/// Order book level structure optimized for prefetching
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PriceLevel {
    pub price: u64,      // Fixed point price
    pub quantity: u64,
    pub order_count: u32,
    pub flags: u32,      // Padding + flags
}

impl Default for PriceLevel {
    fn default() -> Self {
        Self {
            price: 0,
            quantity: 0,
            order_count: 0,
            flags: 0,
        }
    }
}

/// Prefetch iterator for order book levels
pub struct PrefetchIterator<'a> {
    levels: &'a [PriceLevel],
    current: usize,
    prefetch_ahead: usize,
}

impl<'a> PrefetchIterator<'a> {
    pub fn new(levels: &'a [PriceLevel], prefetch_ahead: usize) -> Self {
        Self {
            levels,
            current: 0,
            prefetch_ahead,
        }
    }

    /// Get next level with prefetching
    pub fn next(&mut self) -> Option<&PriceLevel> {
        if self.current >= self.levels.len() {
            return None;
        }

        // Prefetch ahead
        if self.current + self.prefetch_ahead < self.levels.len() {
            unsafe {
                prefetch_ahead(
                    self.levels.as_ptr(),
                    self.current,
                    self.prefetch_ahead,
                    PrefetchHint::T1,
                );
            }
        }

        let level = &self.levels[self.current];
        self.current += 1;
        Some(level)
    }

    /// Reset iterator
    pub fn reset(&mut self) {
        self.current = 0;
    }
}

/// Bulk prefetch for order book arrays
/// 
/// Prefetches all elements in chunks suitable for cache utilization
pub fn bulk_prefetch_levels(levels: &[PriceLevel], chunk_size: usize) {
    if levels.is_empty() {
        return;
    }

    let len = levels.len();
    let mut i = 0;

    while i < len {
        // Prefetch the next chunk
        let prefetch_idx = i + chunk_size;
        if prefetch_idx < len {
            unsafe {
                prefetch(levels.as_ptr().add(prefetch_idx), PrefetchHint::T2);
            }
        }
        
        // Process current chunk (caller would do actual work here)
        i += 1;
    }
}

/// Cache-aware traversal for matching engine
pub struct OrderBookTraversal<'a> {
    bids: &'a [PriceLevel],
    asks: &'a [PriceLevel],
}

impl<'a> OrderBookTraversal<'a> {
    pub fn new(bids: &'a [PriceLevel], asks: &'a [PriceLevel]) -> Self {
        Self { bids, asks }
    }

    /// Traverse bids with prefetching
    /// Returns when predicate returns true or end reached
    pub fn traverse_bids<F>(&self, mut predicate: F) -> Option<usize>
    where
        F: FnMut(&PriceLevel) -> bool,
    {
        if self.bids.is_empty() {
            return None;
        }

        // Prefetch first few levels
        for i in 0..3.min(self.bids.len()) {
            unsafe {
                prefetch(self.bids.as_ptr().add(i), PrefetchHint::T0);
            }
        }

        for (i, level) in self.bids.iter().enumerate() {
            // Prefetch next levels
            if i + 3 < self.bids.len() {
                unsafe {
                    prefetch(self.bids.as_ptr().add(i + 3), PrefetchHint::T1);
                }
            }

            if predicate(level) {
                return Some(i);
            }
        }

        None
    }

    /// Traverse asks with prefetching
    pub fn traverse_asks<F>(&self, mut predicate: F) -> Option<usize>
    where
        F: FnMut(&PriceLevel) -> bool,
    {
        if self.asks.is_empty() {
            return None;
        }

        // Prefetch first few levels
        for i in 0..3.min(self.asks.len()) {
            unsafe {
                prefetch(self.asks.as_ptr().add(i), PrefetchHint::T0);
            }
        }

        for (i, level) in self.asks.iter().enumerate() {
            // Prefetch next levels
            if i + 3 < self.asks.len() {
                unsafe {
                    prefetch(self.asks.as_ptr().add(i + 3), PrefetchHint::T1);
                }
            }

            if predicate(level) {
                return Some(i);
            }
        }

        None
    }

    /// Find best bid that matches criteria
    pub fn find_best_bid(&self, min_quantity: u64) -> Option<&PriceLevel> {
        self.traverse_bids(|level| level.quantity >= min_quantity)
            .and_then(|i| self.bids.get(i))
    }

    /// Find best ask that matches criteria
    pub fn find_best_ask(&self, min_quantity: u64) -> Option<&PriceLevel> {
        self.traverse_asks(|level| level.quantity >= min_quantity)
            .and_then(|i| self.asks.get(i))
    }
}

/// Architecture-specific optimizations
pub mod arch_optimized {
    use super::*;

    /// Check if running on x86_64 with prefetch support
    #[inline]
    pub fn has_prefetch_support() -> bool {
        cfg!(target_arch = "x86_64")
    }

    /// Get optimal prefetch distance based on architecture
    #[inline]
    pub fn optimal_prefetch_distance() -> usize {
        #[cfg(target_arch = "x86_64")]
        {
            // AMD Ryzen typically benefits from 3-5 element prefetch
            4
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            // Conservative default for other architectures
            2
        }
    }

    /// Memory fence for ensuring prefetch completion
    #[inline]
    pub fn memory_fence() {
        #[cfg(target_arch = "x86_64")]
        {
            core::sync::atomic::fence(core::sync::atomic::Ordering::SeqCst);
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            core::sync::atomic::fence(core::sync::atomic::Ordering::SeqCst);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prefetch_iterator() {
        let levels = vec![
            PriceLevel { price: 100, quantity: 1000, order_count: 5, flags: 0 },
            PriceLevel { price: 99, quantity: 500, order_count: 3, flags: 0 },
            PriceLevel { price: 98, quantity: 750, order_count: 4, flags: 0 },
        ];

        let mut iter = PrefetchIterator::new(&levels, 2);
        
        let mut count = 0;
        while let Some(_level) = iter.next() {
            count += 1;
        }
        
        assert_eq!(count, 3);
    }

    #[test]
    fn test_order_book_traversal() {
        let bids = vec![
            PriceLevel { price: 100, quantity: 1000, order_count: 5, flags: 0 },
            PriceLevel { price: 99, quantity: 500, order_count: 3, flags: 0 },
        ];
        let asks = vec![
            PriceLevel { price: 101, quantity: 800, order_count: 4, flags: 0 },
        ];

        let traversal = OrderBookTraversal::new(&bids, &asks);
        
        let best_bid = traversal.find_best_bid(500);
        assert!(best_bid.is_some());
        assert_eq!(best_bid.unwrap().price, 100);
    }

    #[test]
    fn test_arch_detection() {
        // Just verify the functions compile and run
        let _has_support = arch_optimized::has_prefetch_support();
        let _distance = arch_optimized::optimal_prefetch_distance();
    }
}
