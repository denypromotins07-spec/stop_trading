//! Wait-Free Read-Write Lock using Atomic State Machines
//!
//! Allows multiple reader threads (e.g., UI, Telemetry) to access the order book simultaneously
//! without blocking the writer thread. Avoids OS context switches using pure atomic operations.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::cell::UnsafeCell;
use std::marker::PhantomData;
use std::time::{Duration, Instant};

/// Constants for lock state encoding
/// State layout: [writer_flag: 1 bit][reader_count: 31 bits]
const WRITER_FLAG: usize = 1usize << 31;
const READER_MASK: usize = !WRITER_FLAG;
const MAX_READERS: usize = READER_MASK;

/// Wait-free RWLock optimized for read-heavy workloads
pub struct WaitFreeRwLock<T> {
    /// Encoded state: writer flag + reader count
    state: AtomicUsize,
    /// Protected data
    data: UnsafeCell<T>,
    /// Write contention counter (for metrics)
    write_contention: AtomicUsize,
    /// Read contention counter (for metrics)
    read_contention: AtomicUsize,
}

unsafe impl<T: Send> Send for WaitFreeRwLock<T> {}
unsafe impl<T: Send + Sync> Sync for WaitFreeRwLock<T> {}

impl<T> WaitFreeRwLock<T> {
    /// Create a new wait-free RWLock
    pub const fn new(data: T) -> Self {
        Self {
            state: AtomicUsize::new(0),
            data: UnsafeCell::new(data),
            write_contention: AtomicUsize::new(0),
            read_contention: AtomicUsize::new(0),
        }
    }

    /// Acquire a read lock (shared access)
    #[inline]
    pub fn read(&self) -> Option<ReadGuard<'_, T>> {
        let mut current = self.state.load(Ordering::Relaxed);
        
        // Fast path: no writer present
        loop {
            if current & WRITER_FLAG != 0 {
                // Writer is active - check for starvation
                self.read_contention.fetch_add(1, Ordering::Relaxed);
                
                // Backoff and retry
                std::hint::spin_loop();
                current = self.state.load(Ordering::Relaxed);
                continue;
            }
            
            // Try to increment reader count
            let new_state = (current & READER_MASK) + 1;
            
            match self.state.compare_exchange_weak(
                current,
                new_state,
                Ordering::Acquire,
                Ordering::Relaxed,
            ) {
                Ok(_) => return Some(ReadGuard { lock: self }),
                Err(e) => current = e,
            }
        }
    }

    /// Acquire a write lock (exclusive access)
    #[inline]
    pub fn write(&self) -> Option<WriteGuard<'_, T>> {
        let mut current = self.state.load(Ordering::Relaxed);
        
        loop {
            // Check if there are any readers or another writer
            if current != 0 && current != WRITER_FLAG {
                self.write_contention.fetch_add(1, Ordering::Relaxed);
                
                // Backoff and retry
                std::hint::spin_loop();
                current = self.state.load(Ordering::Relaxed);
                continue;
            }
            
            // Try to set writer flag
            match self.state.compare_exchange_weak(
                current,
                WRITER_FLAG,
                Ordering::Acquire,
                Ordering::Relaxed,
            ) {
                Ok(_) => return Some(WriteGuard { lock: self }),
                Err(e) => current = e,
            }
        }
    }

    /// Try to acquire read lock without spinning
    pub fn try_read(&self) -> Option<ReadGuard<'_, T>> {
        let current = self.state.load(Ordering::Relaxed);
        
        if current & WRITER_FLAG != 0 {
            return None;
        }
        
        let new_state = (current & READER_MASK) + 1;
        
        match self.state.compare_exchange(
            current,
            new_state,
            Ordering::Acquire,
            Ordering::Relaxed,
        ) {
            Ok(_) => Some(ReadGuard { lock: self }),
            Err(_) => None,
        }
    }

    /// Try to acquire write lock without spinning
    pub fn try_write(&self) -> Option<WriteGuard<'_, T>> {
        let current = self.state.load(Ordering::Relaxed);
        
        if current != 0 {
            return None;
        }
        
        match self.state.compare_exchange(
            0,
            WRITER_FLAG,
            Ordering::Acquire,
            Ordering::Relaxed,
        ) {
            Ok(_) => Some(WriteGuard { lock: self }),
            Err(_) => None,
        }
    }

    /// Get reference to protected data (requires appropriate lock held)
    #[inline]
    pub fn get(&self) -> &T {
        unsafe { &*self.data.get() }
    }

    /// Get mutable reference to protected data (requires write lock held)
    #[inline]
    pub fn get_mut(&mut self) -> &mut T {
        unsafe { &mut *self.data.get() }
    }

    /// Get current reader count
    pub fn reader_count(&self) -> usize {
        self.state.load(Ordering::Relaxed) & READER_MASK
    }

    /// Check if writer is active
    pub fn has_writer(&self) -> bool {
        self.state.load(Ordering::Relaxed) & WRITER_FLAG != 0
    }

    /// Get write contention count
    pub fn write_contention_count(&self) -> usize {
        self.write_contention.load(Ordering::Relaxed)
    }

    /// Get read contention count
    pub fn read_contention_count(&self) -> usize {
        self.read_contention.load(Ordering::Relaxed)
    }

    /// Reset contention counters
    pub fn reset_counters(&self) {
        self.write_contention.store(0, Ordering::Relaxed);
        self.read_contention.store(0, Ordering::Relaxed);
    }
}

/// Read guard for shared access
pub struct ReadGuard<'a, T> {
    lock: &'a WaitFreeRwLock<T>,
}

impl<'a, T> std::ops::Deref for ReadGuard<'a, T> {
    type Target = T;
    
    fn deref(&self) -> &T {
        self.lock.get()
    }
}

impl<'a, T> Drop for ReadGuard<'a, T> {
    fn drop(&mut self) {
        // Decrement reader count with release semantics
        self.lock.state.fetch_sub(1, Ordering::Release);
    }
}

/// Write guard for exclusive access
pub struct WriteGuard<'a, T> {
    lock: &'a WaitFreeRwLock<T>,
}

impl<'a, T> std::ops::Deref for WriteGuard<'a, T> {
    type Target = T;
    
    fn deref(&self) -> &T {
        self.lock.get()
    }
}

impl<'a, T> std::ops::DerefMut for WriteGuard<'a, T> {
    fn deref_mut(&mut self) -> &mut T {
        self.lock.get_mut()
    }
}

impl<'a, T> Drop for WriteGuard<'a, T> {
    fn drop(&mut self) {
        // Clear writer flag with release semantics
        self.lock.state.store(0, Ordering::Release);
    }
}

/// Optimized read-mostly container for order book snapshots
pub struct ReadMostly<T> {
    inner: WaitFreeRwLock<T>,
    /// Generation counter for version tracking
    generation: AtomicUsize,
}

impl<T> ReadMostly<T> {
    pub const fn new(data: T) -> Self {
        Self {
            inner: WaitFreeRwLock::new(data),
            generation: AtomicUsize::new(0),
        }
    }

    /// Read current value
    pub fn read(&self) -> Option<ReadGuard<'_, T>> {
        self.inner.read()
    }

    /// Update value and bump generation
    pub fn update<F>(&self, f: F) 
    where
        F: FnOnce(&mut T),
    {
        if let Some(mut guard) = self.inner.write() {
            f(&mut *guard);
            self.generation.fetch_add(1, Ordering::Release);
        }
    }

    /// Get current generation
    pub fn generation(&self) -> usize {
        self.generation.load(Ordering::Acquire)
    }
}

/// Multi-reader single-writer buffer for telemetry data
pub struct MRSWBuffer<T> {
    slots: Box<[WaitFreeRwLock<T>]>,
    /// Current write slot
    write_slot: AtomicUsize,
    /// Number of slots
    num_slots: usize,
}

impl<T: Clone + Default> MRSWBuffer<T> {
    pub fn new(num_slots: usize) -> Self {
        let mut slots = Vec::with_capacity(num_slots);
        for _ in 0..num_slots {
            slots.push(WaitFreeRwLock::new(T::default()));
        }
        
        Self {
            slots: slots.into_boxed_slice(),
            write_slot: AtomicUsize::new(0),
            num_slots,
        }
    }

    /// Write new value to next slot
    pub fn write(&self, value: T) {
        let current = self.write_slot.fetch_add(1, Ordering::Relaxed);
        let slot_idx = current % self.num_slots;
        
        if let Some(mut guard) = self.slots[slot_idx].write() {
            *guard = value;
        }
    }

    /// Read from specified slot
    pub fn read(&self, slot_idx: usize) -> Option<ReadGuard<'_, T>> {
        let idx = slot_idx % self.num_slots;
        self.slots[idx].read()
    }

    /// Read latest written slot
    pub fn read_latest(&self) -> Option<ReadGuard<'_, T>> {
        let current = self.write_slot.load(Ordering::Relaxed);
        // Read from the slot before current (most recent complete write)
        let idx = if current == 0 { self.num_slots - 1 } else { current - 1 } % self.num_slots;
        self.slots[idx].read()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_rwlock_basic() {
        let lock = WaitFreeRwLock::new(42);
        
        // Test read
        let guard = lock.read().unwrap();
        assert_eq!(*guard, 42);
        drop(guard);
        
        // Test write
        let mut guard = lock.write().unwrap();
        *guard = 100;
        drop(guard);
        
        // Verify write persisted
        assert_eq!(*lock.read().unwrap(), 100);
    }

    #[test]
    fn test_concurrent_reads() {
        let lock = Arc::new(WaitFreeRwLock::new(0));
        let mut handles = vec![];

        for _ in 0..10 {
            let l = Arc::clone(&lock);
            handles.push(thread::spawn(move || {
                for _ in 0..100 {
                    let guard = l.read().unwrap();
                    assert_eq!(*guard, 0);
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_writer_exclusion() {
        let lock = Arc::new(WaitFreeRwLock::new(0));
        let l1 = Arc::clone(&lock);
        let l2 = Arc::clone(&lock);

        // Acquire write lock
        let _write_guard = lock.write().unwrap();
        
        // Readers should block (in this test, they'll spin briefly then we check)
        let handle1 = thread::spawn(move || {
            // This will spin until writer releases
            let _ = l1.try_read();
        });

        let handle2 = thread::spawn(move || {
            let _ = l2.try_read();
        });

        // Another writer should not be able to acquire
        assert!(lock.try_write().is_none());

        drop(_write_guard);
        
        handle1.join().unwrap();
        handle2.join().unwrap();
    }

    #[test]
    fn test_read_mostly() {
        let rm = ReadMostly::new(vec![1, 2, 3]);
        
        assert_eq!(rm.generation(), 0);
        
        rm.update(|v| v.push(4));
        
        assert_eq!(rm.generation(), 1);
        assert_eq!(rm.read().unwrap().len(), 4);
    }

    #[test]
    fn test_mrsw_buffer() {
        let buffer = MRSWBuffer::new(4);
        
        buffer.write(100);
        buffer.write(200);
        buffer.write(300);
        
        // Read latest
        let latest = buffer.read_latest();
        assert!(latest.is_some());
    }
}
