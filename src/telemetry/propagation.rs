//! Context Propagation Across Lock-Free Channels and Disruptor Ring Buffer
//! 
//! Ensures trace IDs survive asynchronous boundaries without allocating heap memory
//! for context structs on every tick.

use std::sync::atomic::{AtomicU64, AtomicU128, Ordering};
use std::cell::UnsafeCell;

/// Compact trace context that fits in a single u128 (16 bytes)
/// Optimized for zero-allegation propagation through ring buffers
#[derive(Debug, Clone, Copy)]
#[repr(transparent)]
pub struct CompactTraceId {
    bits: u128,
}

impl CompactTraceId {
    /// Trace ID bits layout:
    /// - Bits 0-63: Span ID
    /// - Bits 64-95: Thread ID (32 bits)
    /// - Bits 96-127: Sequence counter (32 bits)
    
    const SPAN_MASK: u128 = 0xFFFF_FFFF_FFFF_FFFF;
    const THREAD_MASK: u128 = 0xFFFF_FFFF << 64;
    const SEQ_MASK: u128 = 0xFFFF_FFFF << 96;

    pub fn new(span_id: u64, thread_id: u32, seq: u32) -> Self {
        let bits = (span_id as u128)
            | ((thread_id as u128) << 64)
            | ((seq as u128) << 96);
        Self { bits }
    }

    pub fn span_id(&self) -> u64 {
        (self.bits & Self::SPAN_MASK) as u64
    }

    pub fn thread_id(&self) -> u32 {
        ((self.bits & Self::THREAD_MASK) >> 64) as u32
    }

    pub fn sequence(&self) -> u32 {
        ((self.bits & Self::SEQ_MASK) >> 96) as u32
    }

    /// Create child span with same trace context
    pub fn child(&self) -> Self {
        let new_span_id = generate_span_id();
        Self::new(new_span_id, self.thread_id(), self.sequence().wrapping_add(1))
    }

    /// Check if this is a valid (non-zero) trace ID
    pub fn is_valid(&self) -> bool {
        self.bits != 0
    }

    /// Get raw bits for efficient storage
    pub fn as_u128(&self) -> u128 {
        self.bits
    }

    /// Create from raw bits
    pub fn from_u128(bits: u128) -> Self {
        Self { bits }
    }

    /// None value for optional contexts
    pub const NONE: Self = Self { bits: 0 };
}

/// Generate unique span ID
fn generate_span_id() -> u64 {
    static COUNTER: AtomicU64 = AtomicU64::new(1);
    COUNTER.fetch_add(1, Ordering::Relaxed)
}

/// Thread-local sequence generator
pub struct SequenceGenerator {
    thread_id: u32,
    counter: UnsafeCell<u32>,
}

unsafe impl Sync for SequenceGenerator {}

impl SequenceGenerator {
    pub fn new(thread_id: u32) -> Self {
        Self {
            thread_id,
            counter: UnsafeCell::new(0),
        }
    }

    #[inline]
    pub fn next(&self) -> CompactTraceId {
        unsafe {
            let seq = *self.counter.get();
            *self.counter.get() = seq.wrapping_add(1);
            CompactTraceId::new(generate_span_id(), self.thread_id, seq)
        }
    }

    pub fn thread_id(&self) -> u32 {
        self.thread_id
    }
}

/// Context carrier for propagating through channels
/// Uses slot optimization to avoid allocations
pub struct ContextCarrier {
    slots: [CompactTraceId; 8],
    active_mask: u8,
}

impl ContextCarrier {
    pub fn new() -> Self {
        Self {
            slots: [CompactTraceId::NONE; 8],
            active_mask: 0,
        }
    }

    /// Set context at slot
    pub fn set(&mut self, slot: u8, ctx: CompactTraceId) {
        if slot < 8 {
            self.slots[slot as usize] = ctx;
            self.active_mask |= 1 << slot;
        }
    }

    /// Get context from slot
    pub fn get(&self, slot: u8) -> Option<CompactTraceId> {
        if slot < 8 && (self.active_mask & (1 << slot)) != 0 {
            Some(self.slots[slot as usize])
        } else {
            None
        }
    }

    /// Clear all contexts
    pub fn clear(&mut self) {
        self.active_mask = 0;
    }

    /// Check if any context is set
    pub fn is_empty(&self) -> bool {
        self.active_mask == 0
    }

    /// Get count of active contexts
    pub fn len(&self) -> usize {
        self.active_mask.count_ones() as usize
    }
}

impl Default for ContextCarrier {
    fn default() -> Self {
        Self::new()
    }
}

/// Lock-free channel adapter with context propagation
pub struct ContextChannel<T> {
    sender: crossbeam_channel::Sender<(T, CompactTraceId)>,
    receiver: crossbeam_channel::Receiver<(T, CompactTraceId)>,
}

impl<T> ContextChannel<T> {
    pub fn bounded(capacity: usize) -> Self {
        let (sender, receiver) = crossbeam_channel::bounded(capacity);
        Self { sender, receiver }
    }

    pub fn unbounded() -> Self {
        let (sender, receiver) = crossbeam_channel::unbounded();
        Self { sender, receiver }
    }

    /// Send with trace context
    pub fn send(&self, value: T, ctx: CompactTraceId) -> Result<(), crossbeam_channel::SendError<(T, CompactTraceId)>> {
        self.sender.send((value, ctx))
    }

    /// Receive with trace context
    pub fn recv(&self) -> Result<(T, CompactTraceId), crossbeam_channel::RecvError> {
        self.receiver.recv()
    }

    /// Try receive with trace context
    pub fn try_recv(&self) -> Result<(T, CompactTraceId), crossbeam_channel::TryRecvError> {
        self.receiver.try_recv()
    }
}

/// Disruptor-style ring buffer with embedded context
pub struct ContextRingBuffer<T, const SIZE: usize> {
    buffer: UnsafeCell<[MaybeUninitSlot<T>; SIZE]>,
    head: AtomicU64,
    tail: AtomicU64,
    mask: u64,
}

struct MaybeUninitSlot<T> {
    value: core::mem::MaybeUninit<T>,
    context: CompactTraceId,
    initialized: AtomicBool,
}

use std::sync::atomic::AtomicBool;
use core::mem::MaybeUninit;

unsafe impl<T: Send, const SIZE: usize> Send for ContextRingBuffer<T, SIZE> {}
unsafe impl<T: Send, const SIZE: usize> Sync for ContextRingBuffer<T, SIZE> {}

impl<T, const SIZE: usize> ContextRingBuffer<T, SIZE> {
    pub fn new() -> Self {
        assert!(SIZE.is_power_of_two(), "Size must be power of 2");
        
        Self {
            buffer: UnsafeCell::new(unsafe {
                core::mem::transmute::<[u8; SIZE * core::mem::size_of::<MaybeUninitSlot<T>>()], [MaybeUninitSlot<T>; SIZE]>(
                    [0u8; SIZE * core::mem::size_of::<MaybeUninitSlot<T>>()]
                )
            }),
            head: AtomicU64::new(0),
            tail: AtomicU64::new(0),
            mask: (SIZE - 1) as u64,
        }
    }

    /// Try to publish an item with context
    pub fn try_publish(&self, value: T, ctx: CompactTraceId) -> bool {
        let tail = self.tail.load(Ordering::Acquire);
        let head = self.head.load(Ordering::Relaxed);
        
        // Check if buffer is full
        if tail.wrapping_sub(head) >= SIZE as u64 {
            return false;
        }

        let idx = (tail as usize) & self.mask;
        let slot = unsafe { &mut *(*self.buffer.get()).add(idx) };

        // Write value and context
        unsafe {
            slot.value.as_mut_ptr().write(value);
        }
        slot.context = ctx;
        slot.initialized.store(true, Ordering::Release);

        self.tail.store(tail.wrapping_add(1), Ordering::Release);
        true
    }

    /// Try to consume an item
    pub fn try_consume(&self) -> Option<(T, CompactTraceId)> {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Relaxed);

        if head >= tail {
            return None;
        }

        let idx = (head as usize) & self.mask;
        let slot = unsafe { &*(*self.buffer.get()).add(idx) };

        if !slot.initialized.load(Ordering::Acquire) {
            return None;
        }

        let value = unsafe { slot.value.as_ptr().read() };
        let ctx = slot.context;
        slot.initialized.store(false, Ordering::Release);

        self.head.store(head.wrapping_add(1), Ordering::Release);
        Some((value, ctx))
    }

    /// Get current size
    pub fn len(&self) -> usize {
        let tail = self.tail.load(Ordering::Relaxed);
        let head = self.head.load(Ordering::Relaxed);
        (tail.wrapping_sub(head)) as usize
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get capacity
    pub const fn capacity(&self) -> usize {
        SIZE
    }
}

impl<T, const SIZE: usize> Default for ContextRingBuffer<T, SIZE> {
    fn default() -> Self {
        Self::new()
    }
}

/// Propagation statistics
#[derive(Debug, Default, Clone)]
pub struct PropagationStats {
    pub contexts_propagated: u64,
    pub contexts_dropped: u64,
    pub orphan_traces: u64,
    pub avg_latency_ns: u128,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compact_trace_id() {
        let ctx = CompactTraceId::new(12345, 42, 100);
        
        assert_eq!(ctx.span_id(), 12345);
        assert_eq!(ctx.thread_id(), 42);
        assert_eq!(ctx.sequence(), 100);
        assert!(ctx.is_valid());
    }

    #[test]
    fn test_compact_trace_child() {
        let parent = CompactTraceId::new(100, 1, 0);
        let child = parent.child();
        
        assert_eq!(child.thread_id(), parent.thread_id());
        assert_ne!(child.span_id(), parent.span_id());
        assert_eq!(child.sequence(), parent.sequence() + 1);
    }

    #[test]
    fn test_context_carrier() {
        let mut carrier = ContextCarrier::new();
        let ctx = CompactTraceId::new(1, 1, 1);
        
        assert!(carrier.is_empty());
        
        carrier.set(0, ctx);
        assert!(!carrier.is_empty());
        assert_eq!(carrier.len(), 1);
        
        let retrieved = carrier.get(0).unwrap();
        assert_eq!(retrieved.span_id(), 1);
        
        carrier.clear();
        assert!(carrier.is_empty());
    }

    #[test]
    fn test_context_channel() {
        let channel: ContextChannel<i32> = ContextChannel::bounded(10);
        let ctx = CompactTraceId::new(1, 1, 1);
        
        channel.send(42, ctx).unwrap();
        
        let (value, received_ctx) = channel.recv().unwrap();
        assert_eq!(value, 42);
        assert_eq!(received_ctx.span_id(), 1);
    }

    #[test]
    fn test_ring_buffer() {
        let buffer: ContextRingBuffer<i32, 16> = ContextRingBuffer::new();
        
        assert!(buffer.is_empty());
        assert_eq!(buffer.capacity(), 16);
        
        let ctx = CompactTraceId::new(1, 1, 1);
        assert!(buffer.try_publish(42, ctx));
        assert_eq!(buffer.len(), 1);
        
        let (value, received_ctx) = buffer.try_consume().unwrap();
        assert_eq!(value, 42);
        assert_eq!(received_ctx.span_id(), 1);
        assert!(buffer.is_empty());
    }
}
