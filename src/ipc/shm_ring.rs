//! Lock-Free Shared Memory Ring Buffer for Zero-Copy IPC
//! 
//! Implements a single-producer/single-consumer ring buffer using
//! crossbeam atomics for zero-copy feature transfer to Python/Ray backend.

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::mem;

/// Maximum ring buffer capacity (power of 2)
const RING_CAPACITY: usize = 16384;

/// Feature vector element
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct FeatureElement {
    pub value: f64,
    pub timestamp_ns: u64,
    pub feature_id: u32,
    pub valid: bool,
}

/// Ring buffer slot with atomic state
#[repr(C)]
pub struct RingSlot {
    pub data: FeatureElement,
    pub sequence: AtomicU64,
}

impl RingSlot {
    fn new() -> Self {
        Self {
            data: FeatureElement::default(),
            sequence: AtomicU64::new(0),
        }
    }
}

/// Lock-free SPSC ring buffer
pub struct ShmRingBuffer {
    buffer: [RingSlot; RING_CAPACITY],
    mask: usize,
    head: AtomicU64,
    tail: AtomicU64,
    closed: AtomicBool,
}

unsafe impl Send for ShmRingBuffer {}
unsafe impl Sync for ShmRingBuffer {}

impl ShmRingBuffer {
    /// Create a new ring buffer
    pub fn new() -> Self {
        Self {
            buffer: std::array::from_fn(|_| RingSlot::new()),
            mask: RING_CAPACITY - 1,
            head: AtomicU64::new(0),
            tail: AtomicU64::new(0),
            closed: AtomicBool::new(false),
        }
    }
    
    /// Push a feature element (producer side)
    #[inline]
    pub fn push(&self, element: FeatureElement) -> bool {
        if self.closed.load(Ordering::Relaxed) {
            return false;
        }
        
        let mut head = self.head.load(Ordering::Relaxed);
        
        loop {
            let tail = self.tail.load(Ordering::Acquire);
            
            // Check if buffer is full
            if head.wrapping_sub(tail) >= RING_CAPACITY as u64 {
                return false;
            }
            
            let next_head = head + 1;
            
            match self.head.compare_exchange_weak(
                head,
                next_head,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => head = actual,
            }
        }
        
        // Write data
        let idx = (head as usize) & self.mask;
        unsafe {
            let slot = &self.buffer[idx];
            slot.data = element;
            slot.sequence.store(head + 1, Ordering::Release);
        }
        
        true
    }
    
    /// Pop a feature element (consumer side)
    #[inline]
    pub fn pop(&self) -> Option<FeatureElement> {
        let mut tail = self.tail.load(Ordering::Relaxed);
        
        loop {
            let head = self.head.load(Ordering::Acquire);
            
            // Check if buffer is empty
            if tail >= head {
                return None;
            }
            
            let idx = (tail as usize) & self.mask;
            let slot = unsafe { &self.buffer[idx] };
            let seq = slot.sequence.load(Ordering::Acquire);
            
            // Check if data is ready
            if seq != tail + 1 {
                return None;
            }
            
            let next_tail = tail + 1;
            
            match self.tail.compare_exchange_weak(
                tail,
                next_tail,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    let data = slot.data;
                    return Some(data);
                }
                Err(actual) => tail = actual,
            }
        }
    }
    
    /// Try to pop without blocking
    #[inline]
    pub fn try_pop(&self) -> Option<FeatureElement> {
        self.pop()
    }
    
    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Acquire);
        tail >= head
    }
    
    /// Get current size
    pub fn len(&self) -> usize {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Acquire);
        (head.wrapping_sub(tail)) as usize
    }
    
    /// Close the ring buffer
    pub fn close(&self) {
        self.closed.store(true, Ordering::Release);
    }
    
    /// Check if closed
    pub fn is_closed(&self) -> bool {
        self.closed.load(Ordering::Acquire)
    }
    
    /// Get capacity
    pub const fn capacity(&self) -> usize {
        RING_CAPACITY
    }
}

impl Default for ShmRingBuffer {
    fn default() -> Self {
        Self::new()
    }
}

/// Feature batch for bulk transfer
#[derive(Debug, Clone)]
pub struct FeatureBatch {
    pub features: Vec<FeatureElement>,
    pub batch_id: u64,
    pub timestamp_ns: u64,
}

impl FeatureBatch {
    pub fn new(batch_id: u64) -> Self {
        Self {
            features: Vec::with_capacity(256),
            batch_id,
            timestamp_ns: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64,
        }
    }
    
    pub fn add_feature(&mut self, feature_id: u32, value: f64) {
        self.features.push(FeatureElement {
            value,
            timestamp_ns: self.timestamp_ns,
            feature_id,
            valid: true,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    
    #[test]
    fn test_ring_buffer_push_pop() {
        let buffer = ShmRingBuffer::new();
        
        let element = FeatureElement {
            value: 42.0,
            timestamp_ns: 12345,
            feature_id: 1,
            valid: true,
        };
        
        assert!(buffer.push(element));
        assert_eq!(buffer.len(), 1);
        
        let popped = buffer.pop();
        assert!(popped.is_some());
        assert_eq!(popped.unwrap().value, 42.0);
        assert!(buffer.is_empty());
    }
    
    #[test]
    fn test_ring_buffer_full() {
        let buffer = ShmRingBuffer::new();
        
        // Fill the buffer
        for i in 0..RING_CAPACITY {
            let element = FeatureElement {
                value: i as f64,
                timestamp_ns: i as u64,
                feature_id: i as u32,
                valid: true,
            };
            assert!(buffer.push(element));
        }
        
        // Should fail when full
        let extra = FeatureElement::default();
        assert!(!buffer.push(extra));
    }
    
    #[test]
    fn test_ring_buffer_concurrent() {
        let buffer = std::sync::Arc::new(ShmRingBuffer::new());
        let buffer_clone = std::sync::Arc::clone(&buffer);
        
        let producer = thread::spawn(move || {
            for i in 0..1000 {
                let element = FeatureElement {
                    value: i as f64,
                    timestamp_ns: i as u64,
                    feature_id: 1,
                    valid: true,
                };
                while !buffer_clone.push(element) {
                    thread::yield_now();
                }
            }
        });
        
        let consumer = thread::spawn(move || {
            let mut count = 0;
            while count < 1000 {
                if buffer.pop().is_some() {
                    count += 1;
                }
            }
        });
        
        producer.join().unwrap();
        consumer.join().unwrap();
    }
}
