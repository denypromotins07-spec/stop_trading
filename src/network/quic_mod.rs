//! QUIC Module Root
//! 
//! Manages connection migration, congestion control, and strict RAM-bounded stream buffers.

pub mod quic_client;
pub mod webtransport;

pub use quic_client::{QuicClient, QuicClientBuilder, QuicClientConfig, QuicStats, QuicStatsSnapshot};
pub use webtransport::{
    MessagePriority, PrioritizedMessage, WebTransportClient, WebTransportBuilder,
    WebTransportConfig, WebTransportStats,
};

use std::sync::Arc;
use std::time::Duration;
use tracing::{debug, info, warn};

/// Maximum total RAM for all QUIC stream buffers (bytes)
const MAX_STREAM_BUFFER_RAM: usize = 64 * 1024 * 1024; // 64MB bounded limit

/// Connection migration state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MigrationState {
    Stable,
    Migrating,
    Migrated,
    Failed,
}

/// Congestion control algorithm selection
#[derive(Debug, Clone, Copy, Default)]
pub enum CongestionAlgorithm {
    /// BBR (Bottleneck Bandwidth and RTT) - default for low latency
    #[default]
    Bbr,
    /// Cubic - standard TCP congestion control
    Cubic,
    /// Reno - classic congestion control
    Reno,
}

/// QUIC manager for handling multiple connections and migrations
pub struct QuicManager {
    connections: Vec<Arc<QuicConnectionHandle>>,
    migration_state: MigrationState,
    congestion_algo: CongestionAlgorithm,
    total_buffer_usage: usize,
    max_connections: usize,
}

/// Handle to a managed QUIC connection
pub struct QuicConnectionHandle {
    pub id: u64,
    pub primary: bool,
    pub backup: Option<u64>,
    pub last_activity: std::time::Instant,
}

impl QuicManager {
    /// Create a new QUIC manager
    pub fn new(max_connections: usize) -> Self {
        Self {
            connections: Vec::with_capacity(max_connections),
            migration_state: MigrationState::Stable,
            congestion_algo: CongestionAlgorithm::Bbr,
            total_buffer_usage: 0,
            max_connections,
        }
    }

    /// Set congestion control algorithm
    pub fn set_congestion_algorithm(&mut self, algo: CongestionAlgorithm) {
        self.congestion_algo = algo;
        info!("Congestion algorithm set to {:?}", algo);
    }

    /// Add a new connection to the manager
    pub fn add_connection(&mut self, primary: bool) -> Option<u64> {
        if self.connections.len() >= self.max_connections {
            warn!("Maximum connections reached");
            return None;
        }

        let id = generate_connection_id();
        let handle = Arc::new(QuicConnectionHandle {
            id,
            primary,
            backup: None,
            last_activity: std::time::Instant::now(),
        });

        self.connections.push(handle);
        debug!("Added connection {} (primary={})", id, primary);
        Some(id)
    }

    /// Initiate connection migration to backup
    pub fn initiate_migration(&mut self, from_id: u64, to_id: u64) -> bool {
        if self.migration_state != MigrationState::Stable {
            warn!("Migration already in progress");
            return false;
        }

        let from_idx = self.connections.iter().position(|c| c.id == from_id);
        let to_idx = self.connections.iter().position(|c| c.id == to_id);

        if from_idx.is_none() || to_idx.is_none() {
            warn!("Invalid connection IDs for migration");
            return false;
        }

        self.migration_state = MigrationState::Migrating;
        info!("Initiating migration from {} to {}", from_id, to_id);
        true
    }

    /// Complete connection migration
    pub fn complete_migration(&mut self, new_primary_id: u64) -> bool {
        if self.migration_state != MigrationState::Migrating {
            return false;
        }

        // Update primary status
        for conn in &mut self.connections {
            let conn_mut = Arc::get_mut(conn).unwrap();
            conn_mut.primary = conn_mut.id == new_primary_id;
        }

        self.migration_state = MigrationState::Migrated;
        info!("Migration completed, new primary: {}", new_primary_id);
        
        // Reset to stable after brief period
        self.migration_state = MigrationState::Stable;
        true
    }

    /// Get current migration state
    pub fn migration_state(&self) -> MigrationState {
        self.migration_state
    }

    /// Allocate stream buffer with RAM limits
    pub fn allocate_stream_buffer(&mut self, size: usize) -> Option<Vec<u8>> {
        if self.total_buffer_usage + size > MAX_STREAM_BUFFER_RAM {
            warn!("Stream buffer allocation would exceed RAM limit");
            return None;
        }

        self.total_buffer_usage += size;
        Some(vec![0u8; size])
    }

    /// Free stream buffer
    pub fn free_stream_buffer(&mut self, size: usize) {
        self.total_buffer_usage = self.total_buffer_usage.saturating_sub(size);
    }

    /// Get current buffer usage
    pub fn buffer_usage(&self) -> (usize, usize) {
        (self.total_buffer_usage, MAX_STREAM_BUFFER_RAM)
    }

    /// Get active connection count
    pub fn connection_count(&self) -> usize {
        self.connections.len()
    }

    /// Find primary connection
    pub fn get_primary(&self) -> Option<&Arc<QuicConnectionHandle>> {
        self.connections.iter().find(|c| c.primary)
    }

    /// Cleanup inactive connections
    pub fn cleanup_inactive(&mut self, timeout: Duration) -> usize {
        let now = std::time::Instant::now();
        let initial_len = self.connections.len();

        self.connections.retain(|conn| {
            now.duration_since(conn.last_activity) < timeout
        });

        let removed = initial_len - self.connections.len();
        if removed > 0 {
            debug!("Cleaned up {} inactive connections", removed);
        }
        removed
    }

    /// Gracefully shutdown all connections
    pub async fn shutdown(&mut self) {
        info!("Shutting down QUIC manager with {} connections", self.connections.len());
        self.connections.clear();
        self.total_buffer_usage = 0;
        self.migration_state = MigrationState::Failed;
    }
}

/// Generate unique connection ID
fn generate_connection_id() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(1);
    COUNTER.fetch_add(1, Ordering::Relaxed)
}

/// Stream buffer manager with fixed-size pools
pub struct StreamBufferPool {
    pools: [Vec<Vec<u8>>; 4], // 4 size classes
    max_per_pool: usize,
    total_allocated: usize,
}

impl StreamBufferPool {
    /// Size classes: 256B, 1KB, 4KB, 64KB
    const SIZE_CLASSES: [usize; 4] = [256, 1024, 4096, 65536];

    pub fn new(max_per_pool: usize) -> Self {
        Self {
            pools: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
            max_per_pool,
            total_allocated: 0,
        }
    }

    /// Acquire buffer from pool or allocate new
    pub fn acquire(&mut self, size_class: usize) -> Option<Vec<u8>> {
        if size_class >= 4 {
            return None;
        }

        // Try to get from pool first
        if let Some(buffer) = self.pools[size_class].pop() {
            return Some(buffer);
        }

        // Check limits before allocating new
        if self.total_allocated >= self.max_per_pool * 4 {
            return None;
        }

        let size = Self::SIZE_CLASSES[size_class];
        self.total_allocated += 1;
        Some(vec![0u8; size])
    }

    /// Return buffer to pool
    pub fn release(&mut self, mut buffer: Vec<u8>, size_class: usize) {
        if size_class >= 4 {
            return;
        }

        // Clear buffer before returning to pool
        buffer.clear();
        
        if self.pools[size_class].len() < self.max_per_pool {
            self.pools[size_class].push(buffer);
        }
        // If pool is full, buffer is dropped
    }

    /// Get pool statistics
    pub fn stats(&self) -> StreamPoolStats {
        StreamPoolStats {
            available: [
                self.pools[0].len(),
                self.pools[1].len(),
                self.pools[2].len(),
                self.pools[3].len(),
            ],
            total_allocated: self.total_allocated,
        }
    }
}

#[derive(Debug, Clone)]
pub struct StreamPoolStats {
    pub available: [usize; 4],
    pub total_allocated: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quic_manager_creation() {
        let manager = QuicManager::new(10);
        assert_eq!(manager.connection_count(), 0);
        assert_eq!(manager.migration_state(), MigrationState::Stable);
    }

    #[test]
    fn test_connection_management() {
        let mut manager = QuicManager::new(10);
        
        let id1 = manager.add_connection(true);
        let id2 = manager.add_connection(false);
        
        assert!(id1.is_some());
        assert!(id2.is_some());
        assert_eq!(manager.connection_count(), 2);
        assert!(manager.get_primary().is_some());
    }

    #[test]
    fn test_migration_flow() {
        let mut manager = QuicManager::new(10);
        
        let id1 = manager.add_connection(true).unwrap();
        let id2 = manager.add_connection(false).unwrap();
        
        assert!(manager.initiate_migration(id1, id2));
        assert_eq!(manager.migration_state(), MigrationState::Migrating);
        
        assert!(manager.complete_migration(id2));
        assert_eq!(manager.migration_state(), MigrationState::Stable);
    }

    #[test]
    fn test_buffer_allocation_limits() {
        let mut manager = QuicManager::new(10);
        
        // Allocate up to limit
        let small_alloc = manager.allocate_stream_buffer(1024);
        assert!(small_alloc.is_some());
        
        // Try to exceed limit
        let huge_alloc = manager.allocate_stream_buffer(MAX_STREAM_BUFFER_RAM + 1);
        assert!(huge_alloc.is_none());
    }

    #[test]
    fn test_stream_buffer_pool() {
        let mut pool = StreamBufferPool::new(10);
        
        // Acquire buffers
        let buf1 = pool.acquire(0); // 256B
        let buf2 = pool.acquire(2); // 4KB
        
        assert!(buf1.is_some());
        assert!(buf2.is_some());
        assert_eq!(buf1.unwrap().len(), 256);
        assert_eq!(buf2.unwrap().len(), 4096);
        
        // Release back to pool
        let buf1 = vec![0u8; 256];
        pool.release(buf1, 0);
        
        let stats = pool.stats();
        assert_eq!(stats.available[0], 1);
    }
}
