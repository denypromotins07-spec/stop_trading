//! Event Dispatcher for Symbol Actors
//! 
//! Routes market data and execution reports to the correct symbol actor.
//! Uses lock-free MPSC channels to prevent cross-thread contention.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::collections::HashMap;
use std::hash::BuildHasherDefault;

// Using crossbeam for lock-free MPSC channels
use crossbeam_channel::{bounded, Sender, Receiver, TrySendError};

/// Cache line size
const CACHE_LINE_SIZE: usize = 64;

/// Maximum channel capacity per actor
const CHANNEL_CAPACITY: usize = 4096;

/// Padded atomic u64
#[repr(C)]
#[derive(Debug)]
pub struct PaddedAtomicU64 {
    _pad1: [u8; CACHE_LINE_SIZE - 8],
    value: AtomicU64,
    _pad2: [u8; CACHE_LINE_SIZE],
}

impl PaddedAtomicU64 {
    pub fn new(initial: u64) -> Self {
        Self {
            _pad1: [0u8; CACHE_LINE_SIZE - 8],
            value: AtomicU64::new(initial),
            _pad2: [0u8; CACHE_LINE_SIZE],
        }
    }

    #[inline]
    pub fn load(&self, ordering: Ordering) -> u64 {
        self.value.load(ordering)
    }

    #[inline]
    pub fn fetch_add(&self, val: u64, ordering: Ordering) -> u64 {
        self.value.fetch_add(val, ordering)
    }
}

/// Message envelope for routing
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Envelope {
    /// Target symbol hash
    pub symbol_hash: u64,
    /// Message priority (higher = more urgent)
    pub priority: u8,
    /// Timestamp (ns)
    pub timestamp_ns: u64,
    /// Payload pointer (type-erased)
    pub payload_ptr: u64,
    /// Payload size
    pub payload_size: u16,
    /// Message type tag
    pub message_type: u8,
}

impl Envelope {
    pub fn new(symbol_hash: u64, priority: u8, message_type: u8) -> Self {
        Self {
            symbol_hash,
            priority,
            timestamp_ns: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64,
            payload_ptr: 0,
            payload_size: 0,
            message_type,
        }
    }
}

/// Actor channel pair
#[repr(C)]
pub struct ActorChannel {
    /// Sender for this actor
    pub sender: Sender<Envelope>,
    /// Receiver for this actor (kept for monitoring)
    pub receiver: Option<Receiver<Envelope>>,
    /// Messages sent counter
    pub messages_sent: PaddedAtomicU64,
    /// Messages dropped counter (channel full)
    pub messages_dropped: PaddedAtomicU64,
    /// Channel is active
    pub is_active: AtomicBool,
}

impl ActorChannel {
    pub fn new(capacity: usize) -> Self {
        let (tx, rx) = bounded(capacity);
        Self {
            sender: tx,
            receiver: Some(rx),
            messages_sent: PaddedAtomicU64::new(0),
            messages_dropped: PaddedAtomicU64::new(0),
            is_active: AtomicBool::new(true),
        }
    }

    /// Send message to actor (non-blocking)
    #[inline]
    pub fn try_send(&self, envelope: Envelope) -> Result<(), TrySendError<Envelope>> {
        if !self.is_active.load(Ordering::Acquire) {
            return Err(TrySendError::Disconnected(envelope));
        }

        match self.sender.try_send(envelope) {
            Ok(_) => {
                self.messages_sent.fetch_add(1, Ordering::Relaxed);
                Ok(())
            }
            Err(TrySendError::Full(e)) => {
                self.messages_dropped.fetch_add(1, Ordering::Relaxed);
                Err(TrySendError::Full(e))
            }
            Err(e @ TrySendError::Disconnected(_)) => {
                self.is_active.store(false, Ordering::Release);
                Err(e)
            }
        }
    }

    /// Send with backpressure (blocks until space available or timeout)
    #[inline]
    pub fn send_timeout(
        &self,
        envelope: Envelope,
        timeout: std::time::Duration,
    ) -> Result<(), crossbeam_channel::SendTimeoutError<Envelope>> {
        if !self.is_active.load(Ordering::Acquire) {
            return Err(crossbeam_channel::SendTimeoutError::Disconnected(envelope));
        }

        match self.sender.send_timeout(envelope, timeout) {
            Ok(_) => {
                self.messages_sent.fetch_add(1, Ordering::Relaxed);
                Ok(())
            }
            Err(e) => {
                if matches!(e, crossbeam_channel::SendTimeoutError::Disconnected(_)) {
                    self.is_active.store(false, Ordering::Release);
                }
                Err(e)
            }
        }
    }

    /// Get receiver (consumes it)
    pub fn take_receiver(&mut self) -> Option<Receiver<Envelope>> {
        self.receiver.take()
    }

    /// Deactivate channel
    #[inline]
    pub fn deactivate(&self) {
        self.is_active.store(false, Ordering::Release);
    }

    /// Check if active
    #[inline]
    pub fn is_active(&self) -> bool {
        self.is_active.load(Ordering::Acquire)
    }

    /// Get statistics
    #[inline]
    pub fn get_stats(&self) -> ChannelStats {
        ChannelStats {
            messages_sent: self.messages_sent.load(Ordering::Relaxed),
            messages_dropped: self.messages_dropped.load(Ordering::Relaxed),
            is_active: self.is_active(),
        }
    }
}

/// Channel statistics
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ChannelStats {
    pub messages_sent: u64,
    pub messages_dropped: u64,
    pub is_active: bool,
}

/// Event dispatcher routing messages to symbol actors
#[repr(C)]
pub struct EventDispatcher {
    /// Map of symbol_hash -> actor channel
    /// Note: In production, use a proper concurrent map like dashmap
    /// For this implementation, we use a simplified approach
    channels: HashMap<u64, Arc<ActorChannel>, BuildHasherDefault<IdentityHasher>>,
    /// Total messages dispatched
    total_dispatched: PaddedAtomicU64,
    /// Total messages failed
    total_failed: PaddedAtomicU64,
    /// Dispatcher active
    is_active: AtomicBool,
    /// Default channel for unknown symbols
    default_channel: Option<Arc<ActorChannel>>,
}

/// Identity hasher for u64 keys (zero-copy)
#[derive(Default)]
pub struct IdentityHasher(u64);

impl std::hash::Hasher for IdentityHasher {
    fn write(&mut self, _: &[u8]) {
        unreachable!("u64 is written in one go");
    }

    fn write_u64(&mut self, n: u64) {
        self.0 = n;
    }

    fn finish(&self) -> u64 {
        self.0
    }
}

impl EventDispatcher {
    pub fn new() -> Self {
        Self {
            channels: HashMap::with_hasher(BuildHasherDefault::default()),
            total_dispatched: PaddedAtomicU64::new(0),
            total_failed: PaddedAtomicU64::new(0),
            is_active: AtomicBool::new(true),
            default_channel: None,
        }
    }

    /// Register a new symbol actor channel
    #[inline]
    pub fn register_actor(&mut self, symbol_hash: u64, channel: Arc<ActorChannel>) {
        self.channels.insert(symbol_hash, channel);
    }

    /// Unregister a symbol actor
    #[inline]
    pub fn unregister_actor(&mut self, symbol_hash: u64) -> Option<Arc<ActorChannel>> {
        self.channels.remove(&symbol_hash)
    }

    /// Set default channel for unknown symbols
    #[inline]
    pub fn set_default_channel(&mut self, channel: Arc<ActorChannel>) {
        self.default_channel = Some(channel);
    }

    /// Dispatch message to specific symbol (lock-free read)
    #[inline]
    pub fn dispatch(&self, envelope: Envelope) -> DispatchResult {
        if !self.is_active.load(Ordering::Acquire) {
            return DispatchResult::DispatcherInactive;
        }

        // Fast path: look up by symbol hash
        if let Some(channel) = self.channels.get(&envelope.symbol_hash) {
            match channel.try_send(envelope) {
                Ok(_) => {
                    self.total_dispatched.fetch_add(1, Ordering::Relaxed);
                    DispatchResult::Sent
                }
                Err(TrySendError::Full(_)) => {
                    self.total_failed.fetch_add(1, Ordering::Relaxed);
                    DispatchResult::ChannelFull
                }
                Err(TrySendError::Disconnected(_)) => {
                    self.total_failed.fetch_add(1, Ordering::Relaxed);
                    DispatchResult::ChannelDisconnected
                }
            }
        } else if let Some(ref default) = self.default_channel {
            // Fallback to default channel
            match default.try_send(envelope) {
                Ok(_) => {
                    self.total_dispatched.fetch_add(1, Ordering::Relaxed);
                    DispatchResult::SentToDefault
                }
                Err(_) => {
                    self.total_failed.fetch_add(1, Ordering::Relaxed);
                    DispatchResult::DefaultFailed
                }
            }
        } else {
            DispatchResult::NoRoute
        }
    }

    /// Dispatch market data to symbol
    #[inline]
    pub fn dispatch_market_data(
        &self,
        symbol_hash: u64,
        bid: u64,
        ask: u64,
        bid_depth: u64,
        ask_depth: u64,
    ) -> DispatchResult {
        // Create envelope for market data (type 1)
        let mut envelope = Envelope::new(symbol_hash, 100, 1); // High priority
        
        // In production, payload would point to serialized market data
        // For now, we encode directly in the envelope conceptually
        
        self.dispatch(envelope)
    }

    /// Dispatch execution report to symbol
    #[inline]
    pub fn dispatch_execution_report(
        &self,
        symbol_hash: u64,
        order_id: u64,
        fill_qty: u64,
        fill_price: u64,
    ) -> DispatchResult {
        // Execution reports are highest priority
        let mut envelope = Envelope::new(symbol_hash, 255, 2);
        
        self.dispatch(envelope)
    }

    /// Broadcast message to all actors
    #[inline]
    pub fn broadcast(&self, message_type: u8, priority: u8) -> BroadcastStats {
        let mut sent = 0u64;
        let mut failed = 0u64;
        
        for (_symbol_hash, channel) in &self.channels {
            let envelope = Envelope::new(*_symbol_hash, priority, message_type);
            match channel.try_send(envelope) {
                Ok(_) => sent += 1,
                Err(_) => failed += 1,
            }
        }
        
        BroadcastStats { sent, failed }
    }

    /// Get number of registered actors
    #[inline]
    pub fn actor_count(&self) -> usize {
        self.channels.len()
    }

    /// Get channel stats for a symbol
    #[inline]
    pub fn get_channel_stats(&self, symbol_hash: u64) -> Option<ChannelStats> {
        self.channels.get(&symbol_hash).map(|c| c.get_stats())
    }

    /// Get dispatcher statistics
    #[inline]
    pub fn get_stats(&self) -> DispatcherStats {
        DispatcherStats {
            total_dispatched: self.total_dispatched.load(Ordering::Relaxed),
            total_failed: self.total_failed.load(Ordering::Relaxed),
            actor_count: self.channels.len(),
            is_active: self.is_active.load(Ordering::Acquire),
        }
    }

    /// Activate dispatcher
    #[inline]
    pub fn activate(&self) {
        self.is_active.store(true, Ordering::Release);
        for (_, channel) in &self.channels {
            channel.is_active.store(true, Ordering::Release);
        }
    }

    /// Deactivate dispatcher (shutdown)
    #[inline]
    pub fn deactivate(&self) {
        self.is_active.store(false, Ordering::Release);
        for (_, channel) in &self.channels {
            channel.deactivate();
        }
    }
}

impl Default for EventDispatcher {
    fn default() -> Self {
        Self::new()
    }
}

/// Dispatch result
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DispatchResult {
    Sent,
    SentToDefault,
    ChannelFull,
    ChannelDisconnected,
    NoRoute,
    DefaultFailed,
    DispatcherInactive,
}

/// Broadcast statistics
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BroadcastStats {
    pub sent: u64,
    pub failed: u64,
}

/// Dispatcher statistics
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DispatcherStats {
    pub total_dispatched: u64,
    pub total_failed: u64,
    pub actor_count: usize,
    pub is_active: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_dispatcher() {
        let mut dispatcher = EventDispatcher::new();
        
        // Create and register an actor channel
        let channel = Arc::new(ActorChannel::new(CHANNEL_CAPACITY));
        dispatcher.register_actor(12345, Arc::clone(&channel));
        
        assert_eq!(dispatcher.actor_count(), 1);
        
        // Dispatch a message
        let envelope = Envelope::new(12345, 100, 1);
        let result = dispatcher.dispatch(envelope);
        assert_eq!(result, DispatchResult::Sent);
        
        // Check stats
        let stats = dispatcher.get_stats();
        assert_eq!(stats.total_dispatched, 1);
        assert_eq!(stats.total_failed, 0);
    }

    #[test]
    fn test_channel_full() {
        let small_channel = Arc::new(ActorChannel::new(1));
        
        // Fill the channel
        let e1 = Envelope::new(12345, 100, 1);
        let e2 = Envelope::new(12345, 100, 1);
        
        assert!(small_channel.try_send(e1).is_ok());
        assert!(matches!(small_channel.try_send(e2), Err(TrySendError::Full(_))));
        
        let stats = small_channel.get_stats();
        assert_eq!(stats.messages_sent, 1);
        assert_eq!(stats.messages_dropped, 1);
    }

    #[test]
    fn test_broadcast() {
        let mut dispatcher = EventDispatcher::new();
        
        for i in 0..5 {
            let channel = Arc::new(ActorChannel::new(CHANNEL_CAPACITY));
            dispatcher.register_actor(i, channel);
        }
        
        let stats = dispatcher.broadcast(1, 50);
        assert_eq!(stats.sent, 5);
        assert_eq!(stats.failed, 0);
    }
}
