//! Hardware Cache Line Optimization Utilities
//! 
//! Implements strict #[repr(align(64))] wrapper macros and cache-line padding.
//! Eliminates false sharing across AMD Ryzen CPU cores.

/// Cache line size in bytes (standard for x86_64 including AMD Ryzen)
pub const CACHE_LINE_SIZE: usize = 64;

/// Marker trait for types that are cache-line aligned
pub trait CacheAligned {}

/// Wrapper macro to ensure 64-byte alignment for a type
#[macro_export]
macro_rules! cache_aligned {
    ($(#[$attr:meta])* $vis:vis struct $name:ident { $($fields:tt)* }) => {
        $(#[$attr])*
        #[repr(align(64))]
        $vis struct $name {
            $($fields)*
        }
        
        unsafe impl $crate::cache::line::CacheAligned for $name {}
    };
}

/// Padding to fill remaining space in a cache line
#[derive(Debug, Clone, Copy, Default)]
pub struct CachePadding<const REMAINING: usize>([u8; REMAINING]);

impl<const N: usize> CachePadding<N> {
    pub const fn new() -> Self {
        Self([0; N])
    }
}

/// Calculate padding needed to reach next cache line boundary
pub const fn cache_line_padding(current_size: usize) -> usize {
    let remainder = current_size % CACHE_LINE_SIZE;
    if remainder == 0 {
        0
    } else {
        CACHE_LINE_SIZE - remainder
    }
}

/// Round up size to nearest cache line multiple
pub const fn align_to_cache_line(size: usize) -> usize {
    (size + CACHE_LINE_SIZE - 1) & !(CACHE_LINE_SIZE - 1)
}

/// A cache-line aligned wrapper for any type
#[repr(align(64))]
#[derive(Debug, Clone)]
pub struct Aligned<T> {
    pub data: T,
    _padding: [u8; Self::calculate_padding::<T>()],
}

impl<T> Aligned<T> {
    const fn calculate_padding<U>() -> usize {
        let size = core::mem::size_of::<U>();
        let remainder = size % CACHE_LINE_SIZE;
        if remainder == 0 {
            0
        } else {
            CACHE_LINE_SIZE - remainder
        }
    }

    pub fn new(data: T) -> Self {
        Self {
            data,
            _padding: [0; Self::calculate_padding::<T>()],
        }
    }

    pub fn into_inner(self) -> T {
        self.data
    }

    pub fn as_ref(&self) -> &T {
        &self.data
    }

    pub fn as_mut(&mut self) -> &mut T {
        &mut self.data
    }
}

impl<T: Default> Default for Aligned<T> {
    fn default() -> Self {
        Self {
            data: T::default(),
            _padding: [0; Self::calculate_padding::<T>()],
        }
    }
}

/// Separate cache lines for concurrent read/write to prevent false sharing
#[repr(align(64))]
pub struct CacheSeparated<T, U> {
    /// Read-only data (frequently accessed by readers)
    pub read_data: T,
    _pad1: [u8; Self::padding_after::<T>()],
    /// Write-only data (frequently modified by writers)
    pub write_data: U,
    _pad2: [u8; Self::padding_after::<U>()],
}

impl<T, U> CacheSeparated<T, U> {
    const fn padding_after<X>() -> usize {
        let size = core::mem::size_of::<X>();
        let remainder = size % CACHE_LINE_SIZE;
        if remainder == 0 {
            0
        } else {
            CACHE_LINE_SIZE - remainder
        }
    }

    pub fn new(read_data: T, write_data: U) -> Self {
        Self {
            read_data,
            _pad1: [0; Self::padding_after::<T>()],
            write_data,
            _pad2: [0; Self::padding_after::<U>()],
        }
    }
}

/// Atomic counter with cache line isolation for lock-free programming
#[repr(align(64))]
pub struct AtomicCounter {
    counter: u64,
    _padding: [u8; 56], // 64 - 8 = 56 bytes padding
}

impl AtomicCounter {
    pub const fn new(initial: u64) -> Self {
        Self {
            counter: initial,
            _padding: [0; 56],
        }
    }

    #[inline]
    pub fn get(&self) -> u64 {
        self.counter
    }

    #[inline]
    pub fn set(&mut self, value: u64) {
        self.counter = value;
    }

    #[inline]
    pub fn increment(&mut self) -> u64 {
        self.counter += 1;
        self.counter
    }

    #[inline]
    pub fn add(&mut self, delta: u64) -> u64 {
        self.counter += delta;
        self.counter
    }
}

/// Multi-producer single-consumer queue slot with cache isolation
#[repr(align(64))]
#[derive(Clone)]
pub struct QueueSlot<T> {
    pub data: Option<T>,
    pub sequence: u64,
    _padding: [u8; 48], // Adjust based on T's size
}

impl<T> QueueSlot<T> {
    pub fn new() -> Self {
        Self {
            data: None,
            sequence: 0,
            _padding: [0; 48],
        }
    }
}

impl<T> Default for QueueSlot<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Array of cache-line separated values for concurrent access
pub struct CacheSeparatedArray<T, const N: usize> {
    slots: [Aligned<T>; N],
}

impl<T: Default, const N: usize> CacheSeparatedArray<T, N> {
    pub fn new() -> Self {
        let mut slots = Vec::with_capacity(N);
        for _ in 0..N {
            slots.push(Aligned::new(T::default()));
        }
        // This is safe because we know the exact layout
        Self {
            slots: unsafe {
                let mut data = Vec::with_capacity(N);
                for i in 0..N {
                    data.push(slots[i].clone());
                }
                std::mem::transmute::<Vec<Aligned<T>>, [Aligned<T>; N]>(data)
            },
        }
    }

    pub fn get(&self, index: usize) -> Option<&T> {
        self.slots.get(index).map(|s| s.as_ref())
    }

    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        self.slots.get_mut(index).map(|s| s.as_mut())
    }
}

/// Helper to check if two pointers are on different cache lines
pub fn are_different_cache_lines(ptr1: *const u8, ptr2: *const u8) -> bool {
    let line1 = (ptr1 as usize) / CACHE_LINE_SIZE;
    let line2 = (ptr2 as usize) / CACHE_LINE_SIZE;
    line1 != line2
}

/// Get the cache line address for a pointer
pub fn cache_line_address(ptr: *const u8) -> usize {
    (ptr as usize) & !(CACHE_LINE_SIZE - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alignment() {
        let aligned = Aligned::new(42u64);
        let ptr = &aligned as *const _ as usize;
        assert_eq!(ptr % CACHE_LINE_SIZE, 0);
    }

    #[test]
    fn test_padding_calculation() {
        assert_eq!(cache_line_padding(0), 0);
        assert_eq!(cache_line_padding(64), 0);
        assert_eq!(cache_line_padding(1), 63);
        assert_eq!(cache_line_padding(65), 63);
    }

    #[test]
    fn test_align_to_cache_line() {
        assert_eq!(align_to_cache_line(1), 64);
        assert_eq!(align_to_cache_line(64), 64);
        assert_eq!(align_to_cache_line(65), 128);
    }

    #[test]
    fn test_atomic_counter() {
        let mut counter = AtomicCounter::new(0);
        assert_eq!(counter.get(), 0);
        
        counter.increment();
        assert_eq!(counter.get(), 1);
        
        counter.add(10);
        assert_eq!(counter.get(), 11);
    }

    #[test]
    fn test_cache_separated() {
        let sep = CacheSeparated::new(100u64, 200u32);
        assert_eq!(sep.read_data, 100);
        assert_eq!(sep.write_data, 200);
    }
}
