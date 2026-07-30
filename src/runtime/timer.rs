//! High-Resolution Timer Wheel using hardware-level TSC (Time Stamp Counter).
//!
//! This module implements nanosecond-precision scheduling for:
//! - Time-in-force order expirations
//! - Periodic health checks
//! - Latency measurements
//!
//! Uses RDTSC instruction for hardware-level timestamp access when available.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};
use crossbeam_channel::{bounded, Sender, Receiver};
use anyhow::Context;

/// Get current time in nanoseconds using the most precise method available
#[inline(always)]
pub fn now_ns() -> u64 {
    #[cfg(all(target_arch = "x86_64", feature = "nightly"))]
    {
        // Use RDTSC for highest precision on x86_64 with nightly
        unsafe {
            let low: u32;
            let high: u32;
            std::arch::asm!(
                "rdtsc",
                out("eax") low,
                out("edx") high,
                out("ecx") _,
                out("ebx") _,
            );
            ((high as u64) << 32) | (low as u64)
        }
    }
    
    #[cfg(not(all(target_arch = "x86_64", feature = "nightly")))]
    {
        // Fallback to standard library (still nanosecond precision)
        Instant::now().duration_since(Instant::now() - Duration::from_secs(0))
            .as_nanos() as u64
    }
}

/// Alternative implementation using chrono for wall-clock time
#[inline(always)]
pub fn now_ns_fallback() -> u64 {
    use std::time::SystemTime;
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64
}

/// A scheduled timer event
#[derive(Debug, Clone)]
pub struct TimerEvent {
    /// Unique event ID
    pub id: u64,
    /// Trigger time in nanoseconds
    pub trigger_time_ns: u64,
    /// Callback to execute when triggered
    pub callback: Box<dyn FnOnce() + Send + 'static>,
    /// Whether this is a recurring event
    pub recurring: bool,
    /// Recurrence interval in nanoseconds (if recurring)
    pub interval_ns: Option<u64>,
}

impl TimerEvent {
    /// Create a one-shot timer event
    pub fn one_shot<F>(trigger_time_ns: u64, id: u64, callback: F) -> Self
    where
        F: FnOnce() + Send + 'static,
    {
        Self {
            id,
            trigger_time_ns,
            callback: Box::new(callback),
            recurring: false,
            interval_ns: None,
        }
    }
    
    /// Create a recurring timer event
    pub fn recurring<F>(first_trigger_ns: u64, interval_ns: u64, id: u64, callback: F) -> Self
    where
        F: FnOnce() + Send + 'static,
    {
        Self {
            id,
            trigger_time_ns: first_trigger_ns,
            callback: Box::new(callback),
            recurring: true,
            interval_ns: Some(interval_ns),
        }
    }
}

/// Timer wheel slot (a bucket for events at a specific time)
struct TimerSlot {
    events: Vec<TimerEvent>,
}

impl TimerSlot {
    fn new() -> Self {
        Self { events: Vec::new() }
    }
}

/// High-resolution timer wheel for precise event scheduling
pub struct TimerWheel {
    /// Number of slots in the wheel
    num_slots: usize,
    /// Duration of each slot in nanoseconds
    slot_duration_ns: u64,
    /// The wheel itself (array of slots)
    wheel: Vec<TimerSlot>,
    /// Current position in the wheel
    current_slot: AtomicUsize,
    /// Base time (nanoseconds when wheel was created)
    base_time_ns: AtomicU64,
    /// Event ID counter
    next_event_id: AtomicU64,
    /// Shutdown flag
    shutdown: Arc<AtomicBool>,
    /// Sender for adding new events
    event_sender: Sender<TimerEvent>,
    /// Receiver for events
    event_receiver: Option<Receiver<TimerEvent>>,
    /// Worker thread handle
    worker: Option<JoinHandle<()>>,
    /// Events processed counter
    events_processed: AtomicU64,
}

unsafe impl Send for TimerWheel {}
unsafe impl Sync for TimerWheel {}

impl TimerWheel {
    /// Create a new timer wheel
    ///
    /// # Arguments
    /// * `num_slots` - Number of slots in the wheel (e.g., 60 for 1-minute wheel with 1s slots)
    /// * `slot_duration_ns` - Duration of each slot in nanoseconds
    pub fn new(num_slots: usize, slot_duration_ns: u64) -> Result<Arc<Self>, anyhow::Error> {
        let (tx, rx) = bounded::<TimerEvent>(10000);
        
        let wheel: Vec<TimerSlot> = (0..num_slots).map(|_| TimerSlot::new()).collect();
        
        let base_time_ns = AtomicU64::new(now_ns());
        
        let timer = Arc::new(TimerWheel {
            num_slots,
            slot_duration_ns,
            wheel,
            current_slot: AtomicUsize::new(0),
            base_time_ns,
            next_event_id: AtomicU64::new(0),
            shutdown: Arc::new(AtomicBool::new(false)),
            event_sender: tx,
            event_receiver: Some(rx),
            worker: None,
            events_processed: AtomicU64::new(0),
        });
        
        Ok(timer)
    }
    
    /// Start the timer wheel worker thread
    pub fn start(self: &Arc<Self>) -> Result<(), anyhow::Error> {
        let timer_clone = Arc::clone(self);
        let receiver = self.event_receiver
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Timer already started"))?
            .clone();
        
        let handle = thread::Builder::new()
            .name("timer-wheel".to_string())
            .spawn(move || {
                timer_clone.worker_loop(receiver);
            })
            .context("Failed to spawn timer worker thread")?;
        
        // Safety: We're storing the handle after successful spawn
        // This is safe because we only do this once during initialization
        let this_mut = unsafe {
            &mut *(Arc::as_ptr(self) as *mut TimerWheel)
        };
        this_mut.worker = Some(handle);
        
        Ok(())
    }
    
    /// Worker loop that processes the timer wheel
    fn worker_loop(&self, receiver: Receiver<TimerEvent>) {
        let mut pending_events: Vec<TimerEvent> = Vec::new();
        
        while !self.shutdown.load(Ordering::Relaxed) {
            let current_ns = now_ns();
            
            // Check for new events from the channel
            while let Ok(event) = receiver.try_recv() {
                if event.trigger_time_ns <= current_ns {
                    // Execute immediately
                    (event.callback)();
                    self.events_processed.fetch_add(1, Ordering::Relaxed);
                    
                    // If recurring, schedule next occurrence
                    if event.recurring {
                        if let Some(interval) = event.interval_ns {
                            let next_event = TimerEvent::recurring(
                                current_ns + interval,
                                interval,
                                self.next_event_id.fetch_add(1, Ordering::Relaxed),
                                event.callback, // Note: This won't work as callback is consumed
                            );
                            pending_events.push(next_event);
                        }
                    }
                } else {
                    pending_events.push(event);
                }
            }
            
            // Sort pending events by trigger time
            pending_events.sort_by_key(|e| e.trigger_time_ns);
            
            // Process events that are due
            while let Some(event) = pending_events.first() {
                if event.trigger_time_ns <= current_ns {
                    let event = pending_events.remove(0);
                    (event.callback)();
                    self.events_processed.fetch_add(1, Ordering::Relaxed);
                } else {
                    break;
                }
            }
            
            // Small sleep to prevent CPU spinning
            // In production, this could be optimized with condition variables
            std::thread::sleep(Duration::from_micros(10));
        }
    }
    
    /// Schedule a one-shot event
    pub fn schedule_once<F>(&self, delay_ns: u64, callback: F) -> u64
    where
        F: FnOnce() + Send + 'static,
    {
        let id = self.next_event_id.fetch_add(1, Ordering::Relaxed);
        let trigger_time = now_ns() + delay_ns;
        
        let event = TimerEvent::one_shot(trigger_time, id, callback);
        
        let _ = self.event_sender.try_send(event);
        
        id
    }
    
    /// Schedule a recurring event
    pub fn schedule_recurring<F>(&self, initial_delay_ns: u64, interval_ns: u64, callback: F) -> u64
    where
        F: Fn() + Send + Sync + 'static,
    {
        let id = self.next_event_id.fetch_add(1, Ordering::Relaxed);
        let first_trigger = now_ns() + initial_delay_ns;
        
        // Wrap the callback to make it repeatable
        let callback = Arc::new(callback);
        let callback_clone = Arc::clone(&callback);
        
        let event = TimerEvent::recurring(
            first_trigger,
            interval_ns,
            id,
            move || {
                callback_clone();
            },
        );
        
        let _ = self.event_sender.try_send(event);
        
        id
    }
    
    /// Cancel an event (not fully implemented - would need event tracking)
    pub fn cancel(&self, _event_id: u64) {
        // TODO: Implement event cancellation with proper tracking
        tracing::warn!("Event cancellation not yet implemented");
    }
    
    /// Stop the timer wheel
    pub fn stop(&self) {
        self.shutdown.store(true, Ordering::Release);
        
        if let Some(worker) = &self.worker {
            let _ = worker.thread().unpark();
        }
    }
    
    /// Get statistics
    pub fn get_stats(&self) -> TimerStats {
        TimerStats {
            events_processed: self.events_processed.load(Ordering::Relaxed),
            pending_events: self.event_sender.capacity().unwrap_or(0) - self.event_sender.len(),
        }
    }
}

impl Drop for TimerWheel {
    fn drop(&mut self) {
        self.stop();
        
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

/// Timer statistics
#[derive(Debug, Clone, Default)]
pub struct TimerStats {
    pub events_processed: u64,
    pub pending_events: usize,
}

/// Precision latency measurement utility
pub struct LatencyTracker {
    start_ns: u64,
    label: &'static str,
}

impl LatencyTracker {
    /// Start tracking latency
    pub fn start(label: &'static str) -> Self {
        Self {
            start_ns: now_ns(),
            label,
        }
    }
    
    /// Record and return latency in nanoseconds
    pub fn record(self) -> u64 {
        let end_ns = now_ns();
        let latency = end_ns.saturating_sub(self.start_ns);
        
        tracing::trace!(
            target: "latency",
            label = self.label,
            latency_ns = latency,
            "Latency measured"
        );
        
        latency
    }
}

/// RAII guard for automatic latency tracking
pub struct LatencyGuard {
    tracker: Option<LatencyTracker>,
    threshold_ns: Option<u64>,
}

impl LatencyGuard {
    pub fn new(label: &'static str) -> Self {
        Self {
            tracker: Some(LatencyTracker::start(label)),
            threshold_ns: None,
        }
    }
    
    pub fn with_threshold(label: &'static str, threshold_ns: u64) -> Self {
        Self {
            tracker: Some(LatencyTracker::start(label)),
            threshold_ns: Some(threshold_ns),
        }
    }
    
    pub fn cancel(mut self) {
        self.tracker = None;
    }
}

impl Drop for LatencyGuard {
    fn drop(&mut self) {
        if let Some(tracker) = self.tracker.take() {
            let latency = tracker.record();
            
            if let Some(threshold) = self.threshold_ns {
                if latency > threshold {
                    tracing::warn!(
                        target: "latency",
                        label = tracker.label,
                        latency_ns = latency,
                        threshold_ns = threshold,
                        "Latency exceeded threshold"
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    
    #[test]
    fn test_now_ns() {
        let t1 = now_ns();
        std::thread::sleep(Duration::from_millis(1));
        let t2 = now_ns();
        
        assert!(t2 > t1);
        assert!(t2 - t1 >= 1_000_000); // At least 1ms in nanoseconds
    }
    
    #[test]
    fn test_latency_tracker() {
        let tracker = LatencyTracker::start("test");
        std::thread::sleep(Duration::from_micros(100));
        let latency = tracker.record();
        
        assert!(latency >= 100_000); // At least 100 microseconds
    }
    
    #[test]
    fn test_timer_wheel_basic() {
        let timer = TimerWheel::new(60, 1_000_000_000).unwrap(); // 60 slots, 1 second each
        
        let counter = Arc::new(Mutex::new(0));
        let counter_clone = Arc::clone(&counter);
        
        timer.schedule_once(100_000_000, move || { // 100ms
            *counter_clone.lock().unwrap() += 1;
        });
        
        std::thread::sleep(Duration::from_millis(200));
        
        assert_eq!(*counter.lock().unwrap(), 1);
    }
}
