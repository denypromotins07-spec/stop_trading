//! Concurrency Module Root
//!
//! Replaces standard library mutexes in the hot path with custom lock-free primitives.

pub mod futex;
pub mod rwlock;

pub use futex::{
    LowLatencyMutex,
    AtomicFlag,
};

#[cfg(target_os = "linux")]
pub use futex::FutexMutex;

#[cfg(not(target_os = "linux"))]
pub use futex::SpinMutex;

pub use rwlock::{
    WaitFreeRwLock,
    ReadMostly,
    MRSWBuffer,
    ReadGuard,
    WriteGuard,
};

/// High-performance concurrent data structures for trading systems
pub mod primitives {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// Lock-free ring buffer for tick data
    pub struct TickRingBuffer<T: Clone + Default> {
        buffer: Box<[WaitFreeRwLock<T>]>,
        head: AtomicU64,
        tail: AtomicU64,
        capacity: usize,
    }

    impl<T: Clone + Default> TickRingBuffer<T> {
        pub fn new(capacity: usize) -> Self {
            let mut buffer = Vec::with_capacity(capacity);
            for _ in 0..capacity {
                buffer.push(WaitFreeRwLock::new(T::default()));
            }

            Self {
                buffer: buffer.into_boxed_slice(),
                head: AtomicU64::new(0),
                tail: AtomicU64::new(0),
                capacity,
            }
        }

        /// Push new tick (overwrites oldest if full)
        pub fn push(&self, tick: T) {
            let head = self.head.fetch_add(1, Ordering::Relaxed);
            let idx = head % self.capacity as u64;

            if let Some(mut guard) = self.buffer[idx as usize].write() {
                *guard = tick;
            }

            // Update tail if we've wrapped around
            let tail = self.tail.load(Ordering::Relaxed);
            if head >= tail + self.capacity as u64 {
                self.tail.store(head - self.capacity as u64 + 1, Ordering::Relaxed);
            }
        }

        /// Read tick at index (relative to head)
        pub fn read_at(&self, offset: u64) -> Option<ReadGuard<'_, T>> {
            let head = self.head.load(Ordering::Relaxed);
            if offset >= head {
                return None;
            }

            let idx = (head - 1 - offset) % self.capacity as u64;
            self.buffer[idx as usize].read()
        }

        /// Get number of ticks available
        pub fn len(&self) -> u64 {
            let head = self.head.load(Ordering::Relaxed);
            let tail = self.tail.load(Ordering::Relaxed);
            head.saturating_sub(tail)
        }

        pub fn is_empty(&self) -> bool {
            self.len() == 0
        }
    }

    /// Sequence lock for ordered updates
    pub struct SeqLock<T> {
        sequence: AtomicU64,
        data: UnsafeCell<T>,
    }

    use std::cell::UnsafeCell;

    unsafe impl<T: Send> Send for SeqLock<T> {}
    unsafe impl<T: Send + Sync> Sync for SeqLock<T> {}

    impl<T: Clone> SeqLock<T> {
        pub const fn new(data: T) -> Self {
            Self {
                sequence: AtomicU64::new(0),
                data: UnsafeCell::new(data),
            }
        }

        /// Read with retry on concurrent write
        pub fn read<F, R>(&self, f: F) -> Option<R>
        where
            F: Fn(&T) -> R,
        {
            for _ in 0..10 {
                let seq_start = self.sequence.load(Ordering::Acquire);

                // Even sequence means no write in progress
                if seq_start % 2 != 0 {
                    std::hint::spin_loop();
                    continue;
                }

                let result = f(unsafe { &*self.data.get() });

                let seq_end = self.sequence.load(Ordering::Acquire);

                // Check if sequence changed during read
                if seq_start == seq_end {
                    return Some(result);
                }

                std::hint::spin_loop();
            }

            None
        }

        /// Write with sequence bump
        pub fn write<F>(&self, f: F)
        where
            F: FnOnce(&mut T),
        {
            // Increment to odd (write in progress)
            self.sequence.fetch_add(1, Ordering::Release);
            std::sync::atomic::fence(Ordering::SeqCst);

            unsafe {
                f(&mut *self.data.get());
            }

            std::sync::atomic::fence(Ordering::SeqCst);
            // Increment to even (write complete)
            self.sequence.fetch_add(1, Ordering::Release);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use primitives::TickRingBuffer;

    #[test]
    fn test_tick_ring_buffer() {
        let buffer = TickRingBuffer::new(10);

        assert!(buffer.is_empty());

        buffer.push(100);
        buffer.push(200);
        buffer.push(300);

        assert_eq!(buffer.len(), 3);

        let val = buffer.read_at(0);
        assert!(val.is_some());
    }

    #[test]
    fn test_seq_lock() {
        use primitives::SeqLock;

        let lock = SeqLock::new(42);

        let val = lock.read(|v| *v);
        assert_eq!(val, Some(42));

        lock.write(|v| *v = 100);

        let val = lock.read(|v| *v);
        assert_eq!(val, Some(100));
    }
}
