//! Actors Module Root
//! 
//! Manages the lifecycle, memory pooling, and thread affinity of all active symbol actors.
//! Exports all actor-related components.

pub mod symbol_actor;
pub mod dispatcher;

pub use symbol_actor::{
    SymbolActorState,
    SymbolActorStats,
    SymbolMessage,
    LocalOrder,
    LocalOrderBook,
    OrderBookLevel,
    AlphaSignal,
    AlphaSignalType,
    Side,
    OrderType,
    OrderStatus,
};

pub use dispatcher::{
    EventDispatcher,
    ActorChannel,
    Envelope,
    DispatchResult,
    ChannelStats,
    DispatcherStats,
    BroadcastStats,
    IdentityHasher,
};

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicIsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::collections::HashMap;

/// Cache line size
const CACHE_LINE_SIZE: usize = 64;

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

/// Actor lifecycle state
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActorLifecycle {
    Created,
    Starting,
    Running,
    Stopping,
    Stopped,
    Failed,
}

/// Actor configuration
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ActorConfig {
    /// CPU core to pin actor thread (-1 = no pinning)
    pub cpu_affinity: isize,
    /// Channel capacity
    pub channel_capacity: usize,
    /// Stack size for actor thread
    pub stack_size: usize,
    /// Priority (higher = more important)
    pub priority: u8,
}

impl Default for ActorConfig {
    fn default() -> Self {
        Self {
            cpu_affinity: -1, // No pinning by default
            channel_capacity: 4096,
            stack_size: 2 * 1024 * 1024, // 2MB
            priority: 50,
        }
    }
}

/// Actor handle for management
#[repr(C)]
pub struct ActorHandle {
    /// Symbol hash
    pub symbol_hash: u64,
    /// Actor state
    pub state: Arc<SymbolActorState>,
    /// Actor channel
    pub channel: Arc<ActorChannel>,
    /// Lifecycle state
    pub lifecycle: AtomicU64, // Encoded ActorLifecycle
    /// Thread handle (optional, stored as raw pointer conceptually)
    pub thread_id: PaddedAtomicU64,
    /// Messages processed
    pub messages_processed: PaddedAtomicU64,
    /// Last heartbeat timestamp
    pub last_heartbeat_ns: PaddedAtomicU64,
}

impl ActorHandle {
    pub fn new(
        symbol_hash: u64,
        state: Arc<SymbolActorState>,
        channel: Arc<ActorChannel>,
    ) -> Self {
        Self {
            symbol_hash,
            state,
            channel,
            lifecycle: AtomicU64::new(ActorLifecycle::Created as u64),
            thread_id: PaddedAtomicU64::new(0),
            messages_processed: PaddedAtomicU64::new(0),
            last_heartbeat_ns: PaddedAtomicU64::new(0),
        }
    }

    #[inline]
    pub fn set_lifecycle(&self, lifecycle: ActorLifecycle) {
        self.lifecycle.store(lifecycle as u64, Ordering::Release);
    }

    #[inline]
    pub fn get_lifecycle(&self) -> ActorLifecycle {
        match self.lifecycle.load(Ordering::Acquire) {
            0 => ActorLifecycle::Created,
            1 => ActorLifecycle::Starting,
            2 => ActorLifecycle::Running,
            3 => ActorLifecycle::Stopping,
            4 => ActorLifecycle::Stopped,
            5 => ActorLifecycle::Failed,
            _ => ActorLifecycle::Failed,
        }
    }

    #[inline]
    pub fn record_heartbeat(&self) {
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        self.last_heartbeat_ns.store(now_ns, Ordering::Release);
    }

    #[inline]
    pub fn get_last_heartbeat_ns(&self) -> u64 {
        self.last_heartbeat_ns.load(Ordering::Acquire)
    }

    #[inline]
    pub fn is_alive(&self) -> bool {
        let lifecycle = self.get_lifecycle();
        lifecycle == ActorLifecycle::Running || lifecycle == ActorLifecycle::Starting
    }
}

/// Actor manager handling lifecycle and pooling
#[repr(C)]
pub struct ActorManager {
    /// Map of symbol_hash -> actor handle
    handles: HashMap<u64, Arc<ActorHandle>>,
    /// Total actors created
    total_actors: PaddedAtomicU64,
    /// Active actors count
    active_actors: PaddedAtomicU64,
    /// Manager active
    is_active: AtomicBool,
    /// Default config for new actors
    default_config: ActorConfig,
    /// Memory pool size (conceptual)
    pool_size: AtomicU64,
}

impl ActorManager {
    pub fn new() -> Self {
        Self {
            handles: HashMap::new(),
            total_actors: PaddedAtomicU64::new(0),
            active_actors: PaddedAtomicU64::new(0),
            is_active: AtomicBool::new(true),
            default_config: ActorConfig::default(),
            pool_size: AtomicU64::new(100), // Default pool size
        }
    }

    /// Create a new actor
    #[inline]
    pub fn create_actor(
        &mut self,
        symbol_hash: u64,
        name_hash: u64,
        max_position: i64,
        max_order_size: u64,
        config: Option<ActorConfig>,
    ) -> Arc<ActorHandle> {
        let cfg = config.unwrap_or(self.default_config);
        
        // Create actor state
        let state = Arc::new(SymbolActorState::new(
            symbol_hash,
            name_hash,
            max_position,
            max_order_size,
        ));

        // Create channel
        let channel = Arc::new(ActorChannel::new(cfg.channel_capacity));

        // Create handle
        let handle = Arc::new(ActorHandle::new(symbol_hash, state, channel));

        // Register
        self.handles.insert(symbol_hash, Arc::clone(&handle));
        self.total_actors.fetch_add(1, Ordering::Relaxed);
        self.active_actors.fetch_add(1, Ordering::Relaxed);

        handle
    }

    /// Remove an actor
    #[inline]
    pub fn remove_actor(&mut self, symbol_hash: u64) -> Option<Arc<ActorHandle>> {
        if let Some(handle) = self.handles.remove(&symbol_hash) {
            handle.set_lifecycle(ActorLifecycle::Stopped);
            handle.channel.deactivate();
            handle.state.deactivate();
            self.active_actors.fetch_sub(1, Ordering::Relaxed);
            Some(handle)
        } else {
            None
        }
    }

    /// Get actor handle
    #[inline]
    pub fn get_actor(&self, symbol_hash: u64) -> Option<Arc<ActorHandle>> {
        self.handles.get(&symbol_hash).cloned()
    }

    /// Start an actor (spawn thread)
    #[inline]
    pub fn start_actor(&self, symbol_hash: u64) -> Result<(), ActorError> {
        if let Some(handle) = self.handles.get(&symbol_hash) {
            if !handle.is_alive() {
                handle.set_lifecycle(ActorLifecycle::Starting);
                
                // In production, spawn actual thread with CPU affinity
                // For now, just mark as running
                handle.set_lifecycle(ActorLifecycle::Running);
                Ok(())
            } else {
                Err(ActorError::AlreadyRunning)
            }
        } else {
            Err(ActorError::NotFound)
        }
    }

    /// Stop an actor
    #[inline]
    pub fn stop_actor(&self, symbol_hash: u64) -> Result<(), ActorError> {
        if let Some(handle) = self.handles.get(&symbol_hash) {
            handle.set_lifecycle(ActorLifecycle::Stopping);
            handle.state.disable_trading();
            
            // Send shutdown message
            let envelope = Envelope::new(symbol_hash, 255, 255); // Shutdown type
            let _ = handle.channel.try_send(envelope);
            
            handle.set_lifecycle(ActorLifecycle::Stopped);
            Ok(())
        } else {
            Err(ActorError::NotFound)
        }
    }

    /// Stop all actors
    #[inline]
    pub fn stop_all(&self) {
        for (_symbol, handle) in &self.handles {
            let _ = self.stop_actor(*_symbol);
        }
    }

    /// Get active actor count
    #[inline]
    pub fn active_count(&self) -> u64 {
        self.active_actors.load(Ordering::Relaxed)
    }

    /// Get total actor count
    #[inline]
    pub fn total_count(&self) -> u64 {
        self.total_actors.load(Ordering::Relaxed)
    }

    /// Check if any actor is unhealthy (missed heartbeats)
    #[inline]
    pub fn check_health(&self, timeout_ns: u64) -> Vec<u64> {
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        
        let mut unhealthy = Vec::new();
        for (symbol, handle) in &self.handles {
            if handle.is_alive() {
                let last_hb = handle.get_last_heartbeat_ns();
                if now_ns.saturating_sub(last_hb) > timeout_ns {
                    unhealthy.push(*symbol);
                }
            }
        }
        unhealthy
    }

    /// Set default config
    #[inline]
    pub fn set_default_config(&mut self, config: ActorConfig) {
        self.default_config = config;
    }

    /// Get manager statistics
    #[inline]
    pub fn get_stats(&self) -> ActorManagerStats {
        ActorManagerStats {
            total_actors: self.total_actors.load(Ordering::Relaxed),
            active_actors: self.active_actors.load(Ordering::Relaxed),
            is_active: self.is_active.load(Ordering::Acquire),
        }
    }

    /// Activate manager
    #[inline]
    pub fn activate(&self) {
        self.is_active.store(true, Ordering::Release);
    }

    /// Deactivate manager
    #[inline]
    pub fn deactivate(&self) {
        self.is_active.store(false, Ordering::Release);
        self.stop_all();
    }
}

impl Default for ActorManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Actor error types
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActorError {
    NotFound,
    AlreadyRunning,
    AlreadyStopped,
    ChannelFull,
    ChannelDisconnected,
    CpuAffinityFailed,
}

/// Actor manager statistics
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ActorManagerStats {
    pub total_actors: u64,
    pub active_actors: u64,
    pub is_active: bool,
}

/// Simple memory pool for actor allocations (conceptual)
#[repr(C)]
pub struct MemoryPool {
    /// Pool size
    size: AtomicU64,
    /// Allocated count
    allocated: PaddedAtomicU64,
    /// Freed count
    freed: PaddedAtomicU64,
}

impl MemoryPool {
    pub fn new(size: u64) -> Self {
        Self {
            size: AtomicU64::new(size),
            allocated: PaddedAtomicU64::new(0),
            freed: PaddedAtomicU64::new(0),
        }
    }

    #[inline]
    pub fn allocate(&self) -> bool {
        let current = self.allocated.load(Ordering::Relaxed);
        let freed = self.freed.load(Ordering::Relaxed);
        let net = current - freed;
        
        if net < self.size.load(Ordering::Relaxed) {
            self.allocated.fetch_add(1, Ordering::Relaxed);
            true
        } else {
            false
        }
    }

    #[inline]
    pub fn free(&self) {
        self.freed.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn get_usage(&self) -> (u64, u64, u64) {
        (
            self.size.load(Ordering::Relaxed),
            self.allocated.load(Ordering::Relaxed),
            self.freed.load(Ordering::Relaxed),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_actor_manager() {
        let mut manager = ActorManager::new();
        
        // Create actor
        let handle = manager.create_actor(12345, 67890, 1_000_000, 100_000, None);
        assert_eq!(manager.active_count(), 1);
        assert_eq!(manager.total_count(), 1);
        
        // Start actor
        assert!(manager.start_actor(12345).is_ok());
        assert!(handle.is_alive());
        
        // Stop actor
        assert!(manager.stop_actor(12345).is_ok());
        assert!(!handle.is_alive());
    }

    #[test]
    fn test_actor_lifecycle() {
        let state = Arc::new(SymbolActorState::new(12345, 67890, 1_000_000, 100_000));
        let channel = Arc::new(ActorChannel::new(4096));
        let handle = ActorHandle::new(12345, state, channel);
        
        assert_eq!(handle.get_lifecycle(), ActorLifecycle::Created);
        
        handle.set_lifecycle(ActorLifecycle::Running);
        assert_eq!(handle.get_lifecycle(), ActorLifecycle::Running);
        assert!(handle.is_alive());
        
        handle.set_lifecycle(ActorLifecycle::Stopped);
        assert!(!handle.is_alive());
    }

    #[test]
    fn test_memory_pool() {
        let pool = MemoryPool::new(10);
        
        for _ in 0..10 {
            assert!(pool.allocate());
        }
        
        // Should fail on 11th
        assert!(!pool.allocate());
        
        // Free one and try again
        pool.free();
        assert!(pool.allocate());
    }
}
