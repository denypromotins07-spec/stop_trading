//! Lock-free, fixed-size ring buffer for rolling statistics.
//! Provides O(1) time complexity for updating and querying time-series data.

use std::sync::atomic::{AtomicUsize, AtomicF64, Ordering};
use std::marker::PhantomData;

/// Error types for ring buffer operations
#[derive(Debug, thiserror::Error)]
pub enum RingBufferError {
    #[error("Buffer is empty")]
    Empty,
    #[error("Buffer capacity exceeded")]
    CapacityExceeded,
    #[error("Invalid index")]
    InvalidIndex,
}

/// A lock-free ring buffer for storing fixed-point or floating point values.
/// Uses atomic operations for thread-safe access without mutexes.
pub struct RingBuffer<const N: usize> {
    buffer: [AtomicF64; N],
    head: AtomicUsize,
    tail: AtomicUsize,
    count: AtomicUsize,
}

impl<const N: usize> RingBuffer<N> {
    /// Create a new empty ring buffer
    pub fn new() -> Self {
        // Initialize all slots to 0.0
        let mut buffer = Vec::with_capacity(N);
        for _ in 0..N {
            buffer.push(AtomicF64::new(0.0));
        }
        
        Self {
            buffer: buffer.try_into().unwrap(),
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
            count: AtomicUsize::new(0),
        }
    }

    /// Push a value into the buffer (O(1))
    /// If buffer is full, overwrites the oldest value
    pub fn push(&self, value: f64) {
        let current_count = self.count.load(Ordering::Relaxed);
        
        if current_count < N {
            // Buffer not full yet
            let tail = self.tail.load(Ordering::Relaxed);
            self.buffer[tail].store(value, Ordering::Relaxed);
            self.tail.store((tail + 1) % N, Ordering::Relaxed);
            self.count.fetch_add(1, Ordering::Relaxed);
        } else {
            // Buffer full, overwrite oldest (head position)
            let head = self.head.load(Ordering::Relaxed);
            self.buffer[head].store(value, Ordering::Relaxed);
            self.head.store((head + 1) % N, Ordering::Relaxed);
            self.tail.store((self.tail.load(Ordering::Relaxed) + 1) % N, Ordering::Relaxed);
        }
    }

    /// Get the value at a specific index (0 = newest, count-1 = oldest)
    pub fn get(&self, index: usize) -> Result<f64, RingBufferError> {
        let count = self.count.load(Ordering::Relaxed);
        if index >= count {
            return Err(RingBufferError::InvalidIndex);
        }
        
        let actual_index = (self.tail.load(Ordering::Relaxed) + N - 1 - index) % N;
        Ok(self.buffer[actual_index].load(Ordering::Relaxed))
    }

    /// Get the number of elements currently in the buffer
    pub fn len(&self) -> usize {
        self.count.load(Ordering::Relaxed)
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.count.load(Ordering::Relaxed) == 0
    }

    /// Check if buffer is full
    pub fn is_full(&self) -> bool {
        self.count.load(Ordering::Relaxed) >= N
    }

    /// Clear the buffer
    pub fn clear(&self) {
        self.head.store(0, Ordering::Relaxed);
        self.tail.store(0, Ordering::Relaxed);
        self.count.store(0, Ordering::Relaxed);
    }
}

impl<const N: usize> Default for RingBuffer<N> {
    fn default() -> Self {
        Self::new()
    }
}

/// Rolling statistics calculator using a lock-free ring buffer
pub struct RollingStats<const N: usize> {
    buffer: RingBuffer<N>,
    sum: AtomicF64,
    sum_sq: AtomicF64,
    count: AtomicUsize,
}

impl<const N: usize> RollingStats<N> {
    /// Create a new rolling statistics calculator
    pub fn new() -> Self {
        Self {
            buffer: RingBuffer::new(),
            sum: AtomicF64::new(0.0),
            sum_sq: AtomicF64::new(0.0),
            count: AtomicUsize::new(0),
        }
    }

    /// Add a new value and update rolling statistics (O(1))
    pub fn update(&self, value: f64) {
        let current_count = self.count.load(Ordering::Relaxed);
        
        if current_count >= N {
            // Buffer is full, remove oldest value from sums
            if let Ok(oldest) = self.buffer.get(current_count - 1) {
                let current_sum = self.sum.load(Ordering::Relaxed);
                let current_sum_sq = self.sum_sq.load(Ordering::Relaxed);
                
                self.sum.store(current_sum - oldest + value, Ordering::Relaxed);
                self.sum_sq.store(current_sum_sq - oldest * oldest + value * value, Ordering::Relaxed);
            }
        } else {
            // Buffer not full yet
            let current_sum = self.sum.load(Ordering::Relaxed);
            let current_sum_sq = self.sum_sq.load(Ordering::Relaxed);
            
            self.sum.store(current_sum + value, Ordering::Relaxed);
            self.sum_sq.store(current_sum_sq + value * value, Ordering::Relaxed);
            self.count.fetch_add(1, Ordering::Relaxed);
        }
        
        self.buffer.push(value);
    }

    /// Calculate the rolling mean (O(1))
    pub fn mean(&self) -> Option<f64> {
        let count = self.count.load(Ordering::Relaxed);
        if count == 0 {
            return None;
        }
        Some(self.sum.load(Ordering::Relaxed) / count as f64)
    }

    /// Calculate the rolling variance (O(1))
    pub fn variance(&self) -> Option<f64> {
        let count = self.count.load(Ordering::Relaxed);
        if count < 2 {
            return None;
        }
        
        let sum = self.sum.load(Ordering::Relaxed);
        let sum_sq = self.sum_sq.load(Ordering::Relaxed);
        let n = count as f64;
        
        let mean = sum / n;
        Some((sum_sq / n) - (mean * mean))
    }

    /// Calculate the rolling standard deviation (O(1))
    pub fn std_dev(&self) -> Option<f64> {
        self.variance().map(|v| v.sqrt())
    }

    /// Calculate the Z-score for a given value (O(1))
    pub fn z_score(&self, value: f64) -> Option<f64> {
        let mean = self.mean()?;
        let std_dev = self.std_dev()?;
        
        if std_dev < 1e-10 {
            return Some(0.0);
        }
        
        Some((value - mean) / std_dev)
    }

    /// Get the current count of values
    pub fn count(&self) -> usize {
        self.count.load(Ordering::Relaxed)
    }

    /// Reset all statistics
    pub fn reset(&self) {
        self.buffer.clear();
        self.sum.store(0.0, Ordering::Relaxed);
        self.sum_sq.store(0.0, Ordering::Relaxed);
        self.count.store(0, Ordering::Relaxed);
    }
}

impl<const N: usize> Default for RollingStats<N> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ring_buffer_basic() {
        let buffer: RingBuffer<5> = RingBuffer::new();
        
        assert!(buffer.is_empty());
        assert_eq!(buffer.len(), 0);
        
        buffer.push(1.0);
        buffer.push(2.0);
        buffer.push(3.0);
        
        assert_eq!(buffer.len(), 3);
        assert_eq!(buffer.get(0).unwrap(), 3.0); // newest
        assert_eq!(buffer.get(2).unwrap(), 1.0); // oldest
    }

    #[test]
    fn test_rolling_stats() {
        let stats: RollingStats<5> = RollingStats::new();
        
        stats.update(1.0);
        stats.update(2.0);
        stats.update(3.0);
        stats.update(4.0);
        stats.update(5.0);
        
        assert_eq!(stats.mean(), Some(3.0));
        assert!(stats.variance().is_some());
        assert!(stats.std_dev().is_some());
        
        // Test Z-score
        let z = stats.z_score(3.0).unwrap();
        assert!((z - 0.0).abs() < 1e-10);
        
        let z_high = stats.z_score(5.0).unwrap();
        assert!(z_high > 0.0);
    }

    #[test]
    fn test_ring_buffer_overflow() {
        let buffer: RingBuffer<3> = RingBuffer::new();
        
        buffer.push(1.0);
        buffer.push(2.0);
        buffer.push(3.0);
        buffer.push(4.0); // Should overwrite 1.0
        
        assert_eq!(buffer.len(), 3);
        assert_eq!(buffer.get(0).unwrap(), 4.0); // newest
        assert_eq!(buffer.get(1).unwrap(), 3.0);
        assert_eq!(buffer.get(2).unwrap(), 2.0); // oldest now
    }
}
