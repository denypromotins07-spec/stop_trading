//! Lock-free Ring Buffer Implementation
//!
//! Custom single-producer/multi-consumer lock-free ring buffer array.
//! Uses cache-line padding and atomic sequence tracking to prevent
//! false sharing across AMD Ryzen CPU cores.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::cell::UnsafeCell;

/// Cache line size for padding
const CACHE_LINE_SIZE: usize = 64;

/// Ring buffer configuration
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RingBufferConfig {
    /// Buffer size (must be power of 2)
    pub size: usize,
    /// Enable bounds checking
    pub bounds_checking: bool,
}

impl Default for RingBufferConfig {
    fn default() -> Self {
        Self {
            size: 1024,
            bounds_checking: true,
        }
    }
}

/// Event factory trait for creating events
pub trait EventFactory: Default + Send + Sync {
    /// Create a new event instance
    fn create() -> Self {
        Self::default()
    }
}

/// Event processor trait for consuming events
pub trait EventProcessor<E>: Send + Sync {
    /// Process an event at the given sequence
    fn on_event(&mut self, event: &E, sequence: u64, end_of_batch: bool);
    
    /// Called when batch starts
    fn on_start(&mut self) {}
    
    /// Called when batch ends
    fn on_shutdown(&mut self) {}
}

/// Slot in the ring buffer with cache-line padding
#[repr(C)]
struct RingSlot<E> {
    /// The event data
    event: UnsafeCell<E>,
    /// Sequence number for this slot
    sequence: AtomicU64,
    /// Padding to prevent false sharing (before)
    _pad_before: [u8; CACHE_LINE_SIZE - std::mem::size_of::<UnsafeCell<E>>() - 8],
    /// Padding to prevent false sharing (after)  
    _pad_after: [u8; CACHE_LINE_SIZE],
}

impl<E: Default> RingSlot<E> {
    #[inline]
    fn new(sequence: u64) -> Self {
        Self {
            event: UnsafeCell::new(E::default()),
            sequence: AtomicU64::new(sequence),
            _pad_before: [0u8; CACHE_LINE_SIZE - std::mem::size_of::<UnsafeCell<E>>() - 8],
            _pad_after: [0u8; CACHE_LINE_SIZE],
        }
    }

    #[inline]
    fn get(&self) -> &E {
        unsafe { &*self.event.get() }
    }

    #[inline]
    fn get_mut(&self) -> &mut E {
        unsafe { &mut *self.event.get() }
    }

    #[inline]
    fn set_sequence(&self, seq: u64) {
        self.sequence.store(seq, Ordering::Release);
    }

    #[inline]
    fn get_sequence(&self) -> u64 {
        self.sequence.load(Ordering::Acquire)
    }
}

/// Lock-free ring buffer for single-producer/multi-consumer
#[repr(C)]
pub struct RingBuffer<E> {
    /// Buffer slots
    slots: Box<[RingSlot<E>]>,
    /// Buffer size (power of 2)
    size: usize,
    /// Mask for fast modulo (size - 1)
    mask: usize,
    /// Gate sequence for consumers
    gate_sequence: AtomicU64,
    /// Padding before cursor
    _pad_before_cursor: [u8; CACHE_LINE_SIZE],
    /// Current cursor position
    cursor: AtomicU64,
    /// Padding after cursor
    _pad_after_cursor: [u8; CACHE_LINE_SIZE],
    /// Bounds checking enabled
    bounds_checking: bool,
}

// Safety: RingBuffer is safe to share between threads when E is Send + Sync
unsafe impl<E: Send + Sync> Send for RingBuffer<E> {}
unsafe impl<E: Send + Sync> Sync for RingBuffer<E> {}

impl<E: Default> RingBuffer<E> {
    /// Create a new ring buffer with given size
    /// Size must be a power of 2
    pub fn new(size: usize) -> Self {
        // Verify size is power of 2
        assert!(size > 0 && (size & (size - 1)) == 0, "Size must be a power of 2");

        let mut slots = Vec::with_capacity(size);
        for i in 0..size {
            slots.push(RingSlot::new(i as i64 - 1)); // Initialize as available
        }

        Self {
            slots: slots.into_boxed_slice(),
            size,
            mask: size - 1,
            gate_sequence: AtomicU64::new(0),
            _pad_before_cursor: [0u8; CACHE_LINE_SIZE],
            cursor: AtomicU64::new(0),
            _pad_after_cursor: [0u8; CACHE_LINE_SIZE],
            bounds_checking: true,
        }
    }

    /// Create ring buffer with custom config
    pub fn with_config(config: RingBufferConfig) -> Self {
        let mut rb = Self::new(config.size);
        rb.bounds_checking = config.bounds_checking;
        rb
    }

    /// Get buffer size
    #[inline]
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get mask for fast indexing
    #[inline]
    pub fn mask(&self) -> usize {
        self.mask
    }

    /// Calculate index from sequence using bitwise AND (fast modulo)
    #[inline]
    fn calculate_index(&self, sequence: u64) -> usize {
        (sequence as usize) & self.mask
    }

    /// Get event at sequence (immutable)
    #[inline]
    pub fn get(&self, sequence: u64) -> &E {
        let index = self.calculate_index(sequence);
        
        if self.bounds_checking {
            debug_assert!(index < self.size, "Index out of bounds");
        }
        
        self.slots[index].get()
    }

    /// Get mutable reference to event at sequence
    /// Only safe to call when you own the sequence slot
    #[inline]
    pub fn get_mut(&self, sequence: u64) -> &mut E {
        let index = self.calculate_index(sequence);
        
        if self.bounds_checking {
            debug_assert!(index < self.size, "Index out of bounds");
        }
        
        self.slots[index].get_mut()
    }

    /// Get multiple events for a batch
    #[inline]
    pub fn get_range(&self, start_seq: u64, count: usize) -> Vec<&E> {
        let mut events = Vec::with_capacity(count);
        for i in 0..count {
            let seq = start_seq + i as u64;
            events.push(self.get(seq));
        }
        events
    }

    /// Check if sequence is available for consumption
    #[inline]
    pub fn is_available(&self, sequence: u64) -> bool {
        let index = self.calculate_index(sequence);
        self.slots[index].get_sequence() >= sequence as i64
    }

    /// Get highest available sequence for consumer
    #[inline]
    pub fn get_highest_available(&self, dependent_seq: u64) -> u64 {
        let cursor = self.cursor.load(Ordering::Acquire);
        
        if dependent_seq >= cursor {
            return dependent_seq;
        }

        // Check how far we can read
        let mut available = dependent_seq;
        while available < cursor {
            if !self.is_available(available + 1) {
                break;
            }
            available += 1;
        }

        available
    }

    /// Check if there are available events to consume
    #[inline]
    pub fn has_available_events(&self, consumer_seq: u64) -> bool {
        let cursor = self.cursor.load(Ordering::Acquire);
        cursor > consumer_seq
    }

    /// Get current cursor position
    #[inline]
    pub fn cursor(&self) -> u64 {
        self.cursor.load(Ordering::Acquire)
    }

    /// Set cursor position (for initialization)
    #[inline]
    pub fn set_cursor(&self, cursor: u64) {
        self.cursor.store(cursor, Ordering::Release);
    }

    /// Get gate sequence
    #[inline]
    pub fn gate_sequence(&self) -> u64 {
        self.gate_sequence.load(Ordering::Acquire)
    }

    /// Update gate sequence based on consumer positions
    #[inline]
    pub fn update_gate_sequence(&self, consumer_sequences: &[u64]) {
        if consumer_sequences.is_empty() {
            return;
        }

        let min_seq = *consumer_sequences.iter().min().unwrap_or(&0);
        self.gate_sequence.store(min_seq, Ordering::Release);
    }

    /// Calculate remaining capacity
    #[inline]
    pub fn remaining_capacity(&self) -> usize {
        let cursor = self.cursor.load(Ordering::Acquire);
        let gate = self.gate_sequence.load(Ordering::Acquire);
        
        // Available slots = buffer_size - (cursor - gate)
        let used = if cursor >= gate {
            (cursor - gate) as usize
        } else {
            0
        };

        self.size.saturating_sub(used)
    }

    /// Check if buffer is full
    #[inline]
    pub fn is_full(&self) -> bool {
        self.remaining_capacity() == 0
    }

    /// Check if buffer is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        let cursor = self.cursor.load(Ordering::Acquire);
        let gate = self.gate_sequence.load(Ordering::Acquire);
        cursor <= gate
    }

    /// Get utilization percentage (0-100)
    #[inline]
    pub fn utilization(&self) -> u8 {
        let used = self.size - self.remaining_capacity();
        ((used * 100) / self.size) as u8
    }

    /// Clear all slots (reset to initial state)
    #[inline]
    pub fn clear(&self) {
        for slot in self.slots.iter() {
            *slot.get_mut() = E::default();
            slot.set_sequence(slot.get_sequence() & !((1 << 63) as u64));
        }
        self.cursor.store(0, Ordering::Release);
        self.gate_sequence.store(0, Ordering::Release);
    }

    /// Get statistics about the buffer
    #[inline]
    pub fn get_stats(&self) -> RingBufferStats {
        RingBufferStats {
            size: self.size,
            cursor: self.cursor(),
            gate_sequence: self.gate_sequence(),
            remaining_capacity: self.remaining_capacity(),
            utilization: self.utilization(),
        }
    }
}

/// Ring buffer statistics
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RingBufferStats {
    pub size: usize,
    pub cursor: u64,
    pub gate_sequence: u64,
    pub remaining_capacity: usize,
    pub utilization: u8,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ring_buffer_creation() {
        let rb: RingBuffer<[u64; 8]> = RingBuffer::new(1024);
        
        assert_eq!(rb.size(), 1024);
        assert_eq!(rb.mask(), 1023);
        assert_eq!(rb.cursor(), 0);
    }

    #[test]
    fn test_power_of_two_validation() {
        // Should panic for non-power-of-2 sizes
        let result = std::panic::catch_unwind(|| {
            let _rb: RingBuffer<u64> = RingBuffer::new(100);
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_index_calculation() {
        let rb: RingBuffer<u64> = RingBuffer::new(1024);
        
        assert_eq!(rb.calculate_index(0), 0);
        assert_eq!(rb.calculate_index(1023), 1023);
        assert_eq!(rb.calculate_index(1024), 0); // Wraps around
        assert_eq!(rb.calculate_index(2048), 0); // Wraps around again
    }

    #[test]
    fn test_get_set_event() {
        let rb: RingBuffer<u64> = RingBuffer::new(1024);
        
        // Get mutable reference and set value
        let event = rb.get_mut(0);
        *event = 42;
        
        // Read back
        assert_eq!(*rb.get(0), 42);
    }

    #[test]
    fn test_remaining_capacity() {
        let rb: RingBuffer<u64> = RingBuffer::new(1024);
        
        // Initially full capacity
        assert_eq!(rb.remaining_capacity(), 1024);
        
        // Simulate some usage
        rb.set_cursor(100);
        assert_eq!(rb.remaining_capacity(), 1024 - 100);
    }

    #[test]
    fn test_utilization() {
        let rb: RingBuffer<u64> = RingBuffer::new(1024);
        
        assert_eq!(rb.utilization(), 0);
        
        rb.set_cursor(512);
        assert_eq!(rb.utilization(), 50);
        
        rb.set_cursor(1024);
        assert_eq!(rb.utilization(), 100);
    }

    #[test]
    fn test_is_available() {
        let rb: RingBuffer<u64> = RingBuffer::new(1024);
        
        // Initially nothing available
        assert!(!rb.is_available(0));
    }

    #[test]
    fn test_clear() {
        let rb: RingBuffer<u64> = RingBuffer::new(1024);
        
        rb.set_cursor(100);
        rb.clear();
        
        assert_eq!(rb.cursor(), 0);
        assert_eq!(rb.gate_sequence(), 0);
    }

    #[test]
    fn test_buffer_stats() {
        let rb: RingBuffer<u64> = RingBuffer::new(1024);
        rb.set_cursor(256);
        
        let stats = rb.get_stats();
        assert_eq!(stats.size, 1024);
        assert_eq!(stats.cursor, 256);
        assert_eq!(stats.remaining_capacity, 1024 - 256);
    }
}
