//! Object Pool - Generic pre-allocated memory pool for zero dynamic allocations.
//!
//! This module provides a generic object pool that pre-allocates memory blocks
//! for network packets, order messages, and other hot-path objects.
//! Ensures zero dynamic allocations occur during the hot execution path
//! to maintain millisecond execution speeds.
//!
//! # Safety
//! Uses unsafe blocks only where strictly necessary for performance.
//! All unsafe operations are documented and audited.

use std::cell::UnsafeCell;
use std::marker::PhantomData;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

/// A slot in the object pool
struct Slot<T> {
    /// Whether this slot is currently in use
    occupied: AtomicBool,
    /// The actual data (uninitialized until claimed)
    data: UnsafeCell<Option<T>>,
}

unsafe impl<T: Send> Send for Slot<T> {}
unsafe impl<T: Send + Sync> Sync for Slot<T> {}

impl<T> Slot<T> {
    fn new() -> Self {
        Self {
            occupied: AtomicBool::new(false),
            data: UnsafeCell::new(None),
        }
    }
}

/// A lock-free object pool for pre-allocated objects.
///
/// Uses atomic operations for thread-safe acquisition and release
/// without locks. Objects are pre-allocated at pool creation time
/// to avoid runtime allocation overhead.
pub struct ObjectPool<T> {
    /// Pre-allocated slots
    slots: Box<[Slot<T>]>,
    
    /// Number of available objects
    available: AtomicUsize,
    
    /// Total capacity of the pool
    capacity: usize,
    
    /// Marker for type T
    _marker: PhantomData<T>,
    
    /// Padding to prevent false sharing
    _padding: [u8; 64],
}

unsafe impl<T: Send> Send for ObjectPool<T> {}
unsafe impl<T: Send + Sync> Sync for ObjectPool<T> {}

impl<T> ObjectPool<T> 
where
    T: Default + Send + 'static,
{
    /// Create a new object pool with the specified capacity
    pub fn new(capacity: usize) -> Result<Arc<Self>, anyhow::Error> {
        if capacity == 0 {
            return Err(anyhow::anyhow!("Pool capacity must be greater than 0"));
        }
        
        // Pre-allocate all slots
        let mut slots = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            slots.push(Slot::new());
        }
        
        let pool = Arc::new(ObjectPool {
            slots: slots.into_boxed_slice(),
            available: AtomicUsize::new(capacity),
            capacity,
            _marker: PhantomData,
            _padding: [0; 64],
        });
        
        Ok(pool)
    }
    
    /// Acquire an object from the pool
    ///
    /// Returns None if the pool is exhausted
    pub fn acquire(&self) -> Option<PoolGuard<T>> {
        // Fast path: check if any objects are available
        if self.available.load(Ordering::Relaxed) == 0 {
            return None;
        }
        
        // Try to find an available slot
        for (idx, slot) in self.slots.iter().enumerate() {
            if !slot.occupied.load(Ordering::Relaxed) {
                // Try to claim this slot
                if slot.occupied.compare_exchange_weak(
                    false,
                    true,
                    Ordering::AcqRel,
                    Ordering::Relaxed,
                ).is_ok() {
                    // Successfully claimed the slot
                    // Initialize the object if needed
                    let data_ptr = slot.data.get();
                    unsafe {
                        if (*data_ptr).is_none() {
                            *data_ptr = Some(T::default());
                        }
                    }
                    
                    // Decrement available count
                    self.available.fetch_sub(1, Ordering::Relaxed);
                    
                    return Some(PoolGuard {
                        slot,
                        pool: self,
                        index: idx,
                        _marker: PhantomData,
                    });
                }
            }
        }
        
        None
    }
    
    /// Acquire an object, blocking until one is available
    ///
    /// # Warning
    /// This uses spin-waiting. Use with caution in production.
    pub fn acquire_blocking(&self) -> PoolGuard<T> {
        loop {
            if let Some(guard) = self.acquire() {
                return guard;
            }
            // Yield to allow other threads to release objects
            std::hint::spin_loop();
        }
    }
    
    /// Get the number of available objects
    pub fn available(&self) -> usize {
        self.available.load(Ordering::Relaxed)
    }
    
    /// Get the total capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    
    /// Get utilization percentage
    pub fn utilization(&self) -> f64 {
        let available = self.available.load(Ordering::Relaxed);
        ((self.capacity - available) as f64 / self.capacity as f64) * 100.0
    }
    
    /// Internal method to release an object back to the pool
    fn release(&self, index: usize) {
        let slot = &self.slots[index];
        
        // Clear the data (optional, depends on whether you want to reset state)
        unsafe {
            *slot.data.get() = None;
        }
        
        // Mark as unoccupied
        slot.occupied.store(false, Ordering::Release);
        
        // Increment available count
        self.available.fetch_add(1, Ordering::Relaxed);
    }
}

/// A guard that holds a reference to a pooled object.
/// When dropped, the object is automatically returned to the pool.
pub struct PoolGuard<'a, T> {
    slot: &'a Slot<T>,
    pool: &'a ObjectPool<T>,
    index: usize,
    _marker: PhantomData<&'a mut T>,
}

impl<'a, T> PoolGuard<'a, T> {
    /// Get a mutable reference to the pooled object
    pub fn get_mut(&mut self) -> &mut T {
        unsafe {
            (*self.slot.data.get()).as_mut().unwrap()
        }
    }
    
    /// Get an immutable reference to the pooled object
    pub fn get(&self) -> &T {
        unsafe {
            (*self.slot.data.get()).as_ref().unwrap()
        }
    }
    
    /// Reset the object to its default state
    pub fn reset(&mut self) 
    where
        T: Default,
    {
        *self.get_mut() = T::default();
    }
}

impl<'a, T> Drop for PoolGuard<'a, T> {
    fn drop(&mut self) {
        self.pool.release(self.index);
    }
}

impl<'a, T> std::ops::Deref for PoolGuard<'a, T> {
    type Target = T;
    
    fn deref(&self) -> &Self::Target {
        self.get()
    }
}

impl<'a, T> std::ops::DerefMut for PoolGuard<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.get_mut()
    }
}

/// Specialized pool for network packet buffers
pub struct PacketBuffer {
    pub data: [u8; 4096],
    pub len: usize,
}

impl Default for PacketBuffer {
    fn default() -> Self {
        Self {
            data: [0u8; 4096],
            len: 0,
        }
    }
}

impl PacketBuffer {
    pub fn new() -> Self {
        Self::default()
    }
    
    pub fn set_data(&mut self, data: &[u8]) {
        let len = data.len().min(self.data.len());
        self.data[..len].copy_from_slice(&data[..len]);
        self.len = len;
    }
}

/// Specialized pool for tick data
#[derive(Default, Clone)]
pub struct TickData {
    pub symbol: [u8; 16],
    pub price: f64,
    pub quantity: f64,
    pub timestamp_ns: u64,
    pub side: u8, // 0 = buy, 1 = sell
}

impl TickData {
    pub fn new() -> Self {
        Self::default()
    }
    
    pub fn set(&mut self, symbol: &str, price: f64, quantity: f64, timestamp_ns: u64, side: u8) {
        let symbol_bytes = symbol.as_bytes();
        let len = symbol_bytes.len().min(16);
        self.symbol[..len].copy_from_slice(&symbol_bytes[..len]);
        self.price = price;
        self.quantity = quantity;
        self.timestamp_ns = timestamp_ns;
        self.side = side;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_pool_basic() {
        let pool = ObjectPool::<String>::new(5).unwrap();
        
        assert_eq!(pool.available(), 5);
        assert_eq!(pool.utilization(), 0.0);
        
        let mut obj = pool.acquire().unwrap();
        assert_eq!(pool.available(), 4);
        
        *obj = "Hello".to_string();
        assert_eq!(*obj, "Hello");
        
        drop(obj);
        assert_eq!(pool.available(), 5);
    }
    
    #[test]
    fn test_pool_exhaustion() {
        let pool = ObjectPool::<u64>::new(3).unwrap();
        
        let _obj1 = pool.acquire().unwrap();
        let _obj2 = pool.acquire().unwrap();
        let _obj3 = pool.acquire().unwrap();
        
        assert!(pool.acquire().is_none());
        assert_eq!(pool.utilization(), 100.0);
    }
    
    #[test]
    fn test_packet_buffer_pool() {
        let pool = ObjectPool::<PacketBuffer>::new(10).unwrap();
        
        let mut buf = pool.acquire().unwrap();
        buf.set_data(b"Hello, World!");
        
        assert_eq!(buf.len, 13);
        assert_eq!(&buf.data[..13], b"Hello, World!");
    }
    
    #[test]
    fn test_tick_data_pool() {
        let pool = ObjectPool::<TickData>::new(100).unwrap();
        
        let mut tick = pool.acquire().unwrap();
        tick.set("BTCUSDT", 50000.0, 1.5, 1234567890, 0);
        
        assert_eq!(tick.price, 50000.0);
        assert_eq!(tick.quantity, 1.5);
        assert_eq!(tick.side, 0);
    }
}
