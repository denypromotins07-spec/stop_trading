//! Runtime Module Root
//!
//! This module provides the asynchronous runtime and event bus infrastructure:
//! - Executor: Custom thread pool with priority-based task scheduling
//! - Timer: High-resolution timer wheel with TSC-based nanosecond precision
//! - RingBuffer: Lock-free LMAX Disruptor-style event passing mechanism
//!
//! The ring buffer serves as the primary inter-thread event passing mechanism
//! for concurrent per-symbol actors.

pub mod executor;
pub mod timer;

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::cell::UnsafeCell;

pub use executor::{Executor, ExecutorStats, PoolConfig, TaskPriority, WorkerPool};
pub use timer::{LatencyGuard, LatencyTracker, TimerWheel, now_ns};

/// Cache line size for padding to prevent false sharing
const CACHE_LINE_SIZE: usize = 64;

/// Default ring buffer capacity (must be power of 2)
const DEFAULT_RING_CAPACITY: usize = 1 << 18; // 256K events

/// Event types for the trading system
#[derive(Debug, Clone)]
pub enum Event {
    /// Market tick data received
    Tick {
        symbol: String,
        price: f64,
        quantity: f64,
        timestamp_ns: u64,
        side: u8,
    },
    /// Order book update
    OrderBookUpdate {
        symbol: String,
        bids: Vec<(f64, f64)>,
        asks: Vec<(f64, f64)>,
        timestamp_ns: u64,
    },
    /// New order request
    NewOrder {
        symbol: String,
        side: u8,
        order_type: u8,
        price: Option<f64>,
        quantity: f64,
        client_order_id: String,
    },
    /// Order execution report
    ExecutionReport {
        order_id: String,
        status: u8,
        filled_quantity: f64,
        average_price: f64,
        timestamp_ns: u64,
    },
    /// Risk check result
    RiskCheckResult {
        order_id: String,
        passed: bool,
        reason: Option<String>,
    },
    /// System health check
    HealthCheck {
        component: String,
        status: u8,
        timestamp_ns: u64,
    },
    /// Timer event
    TimerEvent {
        event_id: u64,
        timestamp_ns: u64,
    },
}

impl Event {
    /// Get event type as string for logging
    pub fn event_type(&self) -> &'static str {
        match self {
            Event::Tick { .. } => "Tick",
            Event::OrderBookUpdate { .. } => "OrderBookUpdate",
            Event::NewOrder { .. } => "NewOrder",
            Event::ExecutionReport { .. } => "ExecutionReport",
            Event::RiskCheckResult { .. } => "RiskCheckResult",
            Event::HealthCheck { .. } => "HealthCheck",
            Event::TimerEvent { .. } => "TimerEvent",
        }
    }
}

/// A slot in the ring buffer
struct RingSlot {
    /// Sequence number for this slot
    sequence: AtomicU64,
    /// The event data (wrapped in UnsafeCell for interior mutability)
    event: UnsafeCell<Option<Event>>,
    /// Padding to prevent false sharing
    _padding: [u8; CACHE_LINE_SIZE - std::mem::size_of::<AtomicU64>() - std::mem::size_of::<Option<Event>>()],
}

unsafe impl Send for RingSlot {}
unsafe impl Sync for RingSlot {}

impl RingSlot {
    fn new() -> Self {
        Self {
            sequence: AtomicU64::new(0),
            event: UnsafeCell::new(None),
            _padding: [0; CACHE_LINE_SIZE - std::mem::size_of::<AtomicU64>() - std::mem::size_of::<Option<Event>>()],
        }
    }
}

/// Lock-free ring buffer implementing LMAX Disruptor pattern
pub struct RingBuffer {
    /// Buffer slots
    buffer: Box<[RingSlot]>,
    /// Buffer mask for fast modulo operation (capacity - 1)
    mask: usize,
    /// Next sequence number to publish
    next_sequence: AtomicU64,
    /// Last consumed sequence number (for monitoring)
    last_consumed: AtomicU64,
    /// Events published counter
    events_published: AtomicUsize,
    /// Events dropped counter (when buffer is full)
    events_dropped: AtomicUsize,
}

unsafe impl Send for RingBuffer {}
unsafe impl Sync for RingBuffer {}

impl RingBuffer {
    /// Create a new ring buffer with the specified capacity
    /// Capacity must be a power of 2
    pub fn new(capacity: usize) -> Result<Arc<Self>, anyhow::Error> {
        if !capacity.is_power_of_two() {
            return Err(anyhow::anyhow!("Ring buffer capacity must be a power of 2"));
        }
        
        let mut buffer = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            buffer.push(RingSlot::new());
        }
        
        Ok(Arc::new(RingBuffer {
            buffer: buffer.into_boxed_slice(),
            mask: capacity - 1,
            next_sequence: AtomicU64::new(0),
            last_consumed: AtomicU64::new(0),
            events_published: AtomicUsize::new(0),
            events_dropped: AtomicUsize::new(0),
        }))
    }
    
    /// Create a ring buffer with default capacity
    pub fn with_default_capacity() -> Result<Arc<Self>, anyhow::Error> {
        Self::new(DEFAULT_RING_CAPACITY)
    }
    
    /// Try to publish an event to the ring buffer
    /// Returns true if successful, false if buffer is full
    pub fn try_publish(&self, event: Event) -> bool {
        let sequence = self.next_sequence.fetch_add(1, Ordering::Relaxed);
        let index = (sequence as usize) & self.mask;
        let slot = &self.buffer[index];
        
        // Wait until this slot is ready for us
        let mut spin_count = 0;
        while slot.sequence.load(Ordering::Acquire) != sequence {
            spin_count += 1;
            if spin_count > 1000 {
                // Buffer is full, decrement sequence and return failure
                self.next_sequence.fetch_sub(1, Ordering::Relaxed);
                self.events_dropped.fetch_add(1, Ordering::Relaxed);
                return false;
            }
            std::hint::spin_loop();
        }
        
        // Write the event
        unsafe {
            *slot.event.get() = Some(event);
        }
        
        // Release the slot for consumers
        slot.sequence.store(sequence + 1, Ordering::Release);
        
        self.events_published.fetch_add(1, Ordering::Relaxed);
        true
    }
    
    /// Publish an event, blocking if necessary until space is available
    pub fn publish(&self, event: Event) {
        loop {
            if self.try_publish(event.clone()) {
                return;
            }
            std::hint::spin_loop();
        }
    }
    
    /// Try to consume an event from the ring buffer
    /// Returns None if no event is available at the given sequence
    pub fn try_consume(&self, sequence: u64) -> Option<Event> {
        let index = (sequence as usize) & self.mask;
        let slot = &self.buffer[index];
        
        if slot.sequence.load(Ordering::Acquire) <= sequence {
            return None;
        }
        
        unsafe {
            let event = (*slot.event.get()).take();
            if event.is_some() {
                self.last_consumed.store(sequence, Ordering::Relaxed);
            }
            event
        }
    }
    
    /// Get the next sequence number to consume
    pub fn next_to_consume(&self) -> u64 {
        self.last_consumed.load(Ordering::Relaxed) + 1
    }
    
    /// Check if there are events available to consume
    pub fn has_events(&self) -> bool {
        let next_produce = self.next_sequence.load(Ordering::Relaxed);
        let next_consume = self.next_to_consume();
        next_produce > next_consume
    }
    
    /// Get the number of pending events
    pub fn pending_events(&self) -> u64 {
        let next_produce = self.next_sequence.load(Ordering::Relaxed);
        let next_consume = self.next_to_consume();
        next_produce.saturating_sub(next_consume)
    }
    
    /// Get buffer statistics
    pub fn get_stats(&self) -> RingBufferStats {
        RingBufferStats {
            capacity: self.mask + 1,
            events_published: self.events_published.load(Ordering::Relaxed),
            events_dropped: self.events_dropped.load(Ordering::Relaxed),
            pending_events: self.pending_events(),
            utilization: self.pending_events() as f64 / (self.mask + 1) as f64 * 100.0,
        }
    }
    
    /// Reset the ring buffer (only safe when no other threads are accessing)
    pub fn reset(&self) {
        for slot in self.buffer.iter() {
            unsafe {
                *slot.event.get() = None;
            }
            slot.sequence.store(0, Ordering::Release);
        }
        self.next_sequence.store(0, Ordering::Release);
        self.last_consumed.store(0, Ordering::Release);
    }
}

/// Ring buffer statistics
#[derive(Debug, Clone)]
pub struct RingBufferStats {
    pub capacity: usize,
    pub events_published: usize,
    pub events_dropped: usize,
    pub pending_events: u64,
    pub utilization: f64,
}

impl RingBufferStats {
    pub fn format(&self) -> String {
        format!(
            "RingBuffer | Capacity: {} | Published: {} | Dropped: {} | Pending: {} | Utilization: {:.1}%",
            self.capacity,
            self.events_published,
            self.events_dropped,
            self.pending_events,
            self.utilization
        )
    }
}

/// Event bus that combines ring buffer with executor for event processing
pub struct EventBus {
    ring_buffer: Arc<RingBuffer>,
    executor: Arc<Executor>,
    shutdown: Arc<std::sync::atomic::AtomicBool>,
}

unsafe impl Send for EventBus {}
unsafe impl Sync for EventBus {}

impl EventBus {
    /// Create a new event bus
    pub fn new(ring_buffer: Arc<RingBuffer>, executor: Arc<Executor>) -> Self {
        Self {
            ring_buffer,
            executor,
            shutdown: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }
    
    /// Publish an event to the bus
    pub fn publish(&self, event: Event) -> bool {
        self.ring_buffer.try_publish(event)
    }
    
    /// Publish with high priority (critical path)
    pub fn publish_critical(&self, event: Event) -> bool {
        // For critical events, we might want to use blocking publish
        if !self.ring_buffer.try_publish(event.clone()) {
            // If ring buffer is full, execute directly on critical pool
            let _ = self.executor.submit_critical(move || {
                // Handle the event directly
                tracing::warn!("Critical event handled directly due to ring buffer full: {:?}", event);
            });
            return true;
        }
        true
    }
    
    /// Start consuming events with the given handler
    pub fn start_consumer<F>(&self, handler: F) -> Result<(), anyhow::Error>
    where
        F: Fn(Event) + Send + Sync + 'static,
    {
        let ring = Arc::clone(&self.ring_buffer);
        let shutdown = Arc::clone(&self.shutdown);
        let handler = Arc::new(handler);
        
        self.executor.submit_background(move || {
            let mut sequence = ring.next_to_consume();
            
            while !shutdown.load(Ordering::Relaxed) {
                if let Some(event) = ring.try_consume(sequence) {
                    let handler_clone = Arc::clone(&handler);
                    handler_clone(event);
                    sequence += 1;
                } else {
                    std::hint::spin_loop();
                }
            }
        })?;
        
        Ok(())
    }
    
    /// Signal shutdown
    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::Release);
    }
    
    /// Get event bus statistics
    pub fn get_stats(&self) -> EventBusStats {
        EventBusStats {
            ring_buffer: self.ring_buffer.get_stats(),
            executor: self.executor.get_stats(),
        }
    }
}

/// Combined event bus statistics
#[derive(Debug, Clone)]
pub struct EventBusStats {
    pub ring_buffer: RingBufferStats,
    pub executor: ExecutorStats,
}

impl EventBusStats {
    pub fn format(&self) -> String {
        format!(
            "{}\n{}",
            self.ring_buffer.format(),
            self.executor.format()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    
    #[test]
    fn test_ring_buffer_basic() {
        let ring = RingBuffer::new(16).unwrap();
        
        assert!(ring.try_publish(Event::HealthCheck {
            component: "test".to_string(),
            status: 0,
            timestamp_ns: now_ns(),
        }));
        
        assert!(ring.has_events());
        assert_eq!(ring.pending_events(), 1);
        
        let event = ring.try_consume(0);
        assert!(event.is_some());
        assert_eq!(ring.pending_events(), 0);
    }
    
    #[test]
    fn test_ring_buffer_wraparound() {
        let ring = RingBuffer::new(4).unwrap();
        
        // Fill the buffer
        for i in 0..4 {
            assert!(ring.try_publish(Event::HealthCheck {
                component: format!("test-{}", i),
                status: 0,
                timestamp_ns: now_ns(),
            }));
        }
        
        // Consume all
        for i in 0..4 {
            let event = ring.try_consume(i);
            assert!(event.is_some());
        }
        
        // Fill again (should wrap around)
        for i in 4..8 {
            assert!(ring.try_publish(Event::HealthCheck {
                component: format!("test-{}", i),
                status: 0,
                timestamp_ns: now_ns(),
            }));
        }
        
        assert_eq!(ring.pending_events(), 4);
    }
    
    #[test]
    fn test_event_bus() {
        let ring = RingBuffer::new(16).unwrap();
        let executor = Arc::new(Executor::new().unwrap());
        let bus = EventBus::new(ring, executor);
        
        let counter = Arc::new(Mutex::new(0));
        let counter_clone = Arc::clone(&counter);
        
        bus.start_consumer(move |_event| {
            *counter_clone.lock().unwrap() += 1;
        }).unwrap();
        
        bus.publish(Event::HealthCheck {
            component: "test".to_string(),
            status: 0,
            timestamp_ns: now_ns(),
        });
        
        std::thread::sleep(std::time::Duration::from_millis(100));
        
        assert_eq!(*counter.lock().unwrap(), 1);
        
        bus.shutdown();
    }
}
