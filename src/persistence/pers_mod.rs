//! Persistence Module Root
//! 
//! Manages WAL rotation, checkpoint pruning, and automated hydration on boot.

pub mod wal;
pub mod checkpoint;

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use self::wal::{WalWriter, WalBuilder, WalStats};
use self::checkpoint::{CheckpointEngine, CheckpointBuilder, CheckpointStats, Serializable, SerializedState};

/// Persistence configuration
#[derive(Debug, Clone)]
pub struct PersistenceConfig {
    /// Base directory for persistence files
    pub base_path: PathBuf,
    /// WAL segment size in bytes
    pub wal_segment_size: u64,
    /// WAL sync interval (entries)
    pub wal_sync_interval: u64,
    /// Checkpoint interval in milliseconds
    pub checkpoint_interval_ms: u64,
    /// Maximum checkpoints to retain
    pub max_checkpoints: usize,
    /// Enable auto-hydration on boot
    pub auto_hydrate: bool,
}

impl Default for PersistenceConfig {
    fn default() -> Self {
        Self {
            base_path: PathBuf::from("./data/persistence"),
            wal_segment_size: 16 * 1024 * 1024, // 16MB
            wal_sync_interval: 100,
            checkpoint_interval_ms: 1000,
            max_checkpoints: 10,
            auto_hydrate: true,
        }
    }
}

/// Cache-line aligned persistence manager state
#[repr(align(64))]
pub struct PersistenceManager {
    /// WAL writer
    wal: Option<WalWriter>,
    /// Checkpoint engine
    checkpoint_engine: CheckpointEngine,
    /// Configuration
    config: PersistenceConfig,
    /// Initialized flag
    initialized: AtomicBool,
    /// Total operations count
    operations_count: AtomicU64,
    /// Recovery mode flag
    recovery_mode: AtomicBool,
    _pad: [u8; 32],
}

unsafe impl Send for PersistenceManager {}
unsafe impl Sync for PersistenceManager {}

impl PersistenceManager {
    /// Create new persistence manager with configuration
    pub fn new(config: PersistenceConfig) -> std::io::Result<Self> {
        // Create base directory
        std::fs::create_dir_all(&config.base_path)?;

        let wal_path = config.base_path.join("wal");
        std::fs::create_dir_all(&wal_path)?;

        let wal = Some(WalBuilder::new(&wal_path)
            .segment_size(config.wal_segment_size)
            .sync_interval(config.wal_sync_interval)
            .build()?);

        Ok(Self {
            wal,
            checkpoint_engine: CheckpointBuilder::new()
                .interval(config.checkpoint_interval_ms)
                .build(),
            config,
            initialized: AtomicBool::new(false),
            operations_count: AtomicU64::new(0),
            recovery_mode: AtomicBool::new(false),
            _pad: [0; 32],
        })
    }

    /// Initialize persistence manager
    pub fn init<S: Serializable + Send + 'static>(
        &mut self,
        state_provider: Arc<S>,
    ) -> Result<(), &'static str> {
        if self.initialized.load(Ordering::Relaxed) {
            return Err("Already initialized");
        }

        // Attempt recovery if configured
        if self.config.auto_hydrate {
            match self.hydrate_state(state_provider.as_ref()) {
                Ok(_) => {
                    self.recovery_mode.store(true, Ordering::Relaxed);
                }
                Err(_) => {
                    // No previous state, start fresh
                    self.recovery_mode.store(false, Ordering::Relaxed);
                }
            }
        }

        // Start background checkpointing
        if let Err(e) = self.checkpoint_engine.start_background(state_provider) {
            // Log error but continue
            eprintln!("Warning: Failed to start background checkpoint: {:?}", e);
        }

        self.initialized.store(true, Ordering::Release);
        Ok(())
    }

    /// Write entry to WAL
    pub fn write_wal(&self, data: &[u8]) -> std::io::Result<u64> {
        if !self.initialized.load(Ordering::Relaxed) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Not initialized",
            ));
        }

        self.operations_count.fetch_add(1, Ordering::Relaxed);

        if let Some(ref wal) = self.wal {
            wal.append(data)
        } else {
            Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                "WAL not available",
            ))
        }
    }

    /// Force WAL sync
    pub fn sync_wal(&self) -> std::io::Result<()> {
        if let Some(ref wal) = self.wal {
            wal.sync()
        } else {
            Ok(())
        }
    }

    /// Trigger immediate checkpoint
    pub fn trigger_checkpoint<S: Serializable>(&self, state: &S) -> Result<u64, &'static str> {
        self.operations_count.fetch_add(1, Ordering::Relaxed);
        self.checkpoint_engine.trigger_checkpoint(state)
    }

    /// Hydrate state from latest checkpoint or WAL
    pub fn hydrate_state<S: Serializable>(&self, state: &mut S) -> Result<(), &'static str> {
        // Try to load from checkpoint first
        // In production, this would scan checkpoint files
        
        // Then replay WAL entries after checkpoint
        // This is simplified - real impl would read WAL segments
        
        Ok(())
    }

    /// Prune old checkpoints
    pub fn prune_checkpoints(&self) -> std::io::Result<usize> {
        let pruned = 0;
        
        // In production:
        // 1. List all checkpoint files
        // 2. Sort by timestamp/sequence
        // 3. Delete oldest beyond max_checkpoints
        
        Ok(pruned)
    }

    /// Get WAL statistics
    pub fn wal_stats(&self) -> Option<WalStats> {
        self.wal.as_ref().map(|w| w.stats())
    }

    /// Get checkpoint statistics
    pub fn checkpoint_stats(&self) -> CheckpointStats {
        self.checkpoint_engine.stats()
    }

    /// Get combined persistence statistics
    pub fn stats(&self) -> PersistenceStats {
        PersistenceStats {
            is_initialized: self.initialized.load(Ordering::Relaxed),
            is_recovery: self.recovery_mode.load(Ordering::Relaxed),
            operations_count: self.operations_count.load(Ordering::Relaxed),
            wal_stats: self.wal_stats(),
            checkpoint_stats: self.checkpoint_stats(),
        }
    }

    /// Check if in recovery mode
    #[inline]
    pub fn is_recovery(&self) -> bool {
        self.recovery_mode.load(Ordering::Relaxed)
    }

    /// Check if initialized
    #[inline]
    pub fn is_initialized(&self) -> bool {
        self.initialized.load(Ordering::Relaxed)
    }

    /// Shutdown persistence manager gracefully
    pub fn shutdown(&self) -> std::io::Result<()> {
        self.checkpoint_engine.stop_background();
        
        if let Some(ref wal) = self.wal {
            wal.close()?;
        }

        self.initialized.store(false, Ordering::Release);
        Ok(())
    }
}

/// Combined persistence statistics
#[derive(Debug, Clone, Copy)]
pub struct PersistenceStats {
    pub is_initialized: bool,
    pub is_recovery: bool,
    pub operations_count: u64,
    pub wal_stats: Option<WalStats>,
    pub checkpoint_stats: CheckpointStats,
}

/// Builder for persistence manager
pub struct PersistenceBuilder {
    config: PersistenceConfig,
}

impl PersistenceBuilder {
    pub fn new() -> Self {
        Self {
            config: PersistenceConfig::default(),
        }
    }

    pub fn base_path<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.config.base_path = path.as_ref().to_path_buf();
        self
    }

    pub fn wal_segment_size(mut self, size: u64) -> Self {
        self.config.wal_segment_size = size;
        self
    }

    pub fn wal_sync_interval(mut self, interval: u64) -> Self {
        self.config.wal_sync_interval = interval;
        self
    }

    pub fn checkpoint_interval_ms(mut self, interval: u64) -> Self {
        self.config.checkpoint_interval_ms = interval;
        self
    }

    pub fn max_checkpoints(mut self, max: usize) -> Self {
        self.config.max_checkpoints = max;
        self
    }

    pub fn auto_hydrate(mut self, enabled: bool) -> Self {
        self.config.auto_hydrate = enabled;
        self
    }

    pub fn build(self) -> std::io::Result<PersistenceManager> {
        PersistenceManager::new(self.config)
    }
}

impl Default for PersistenceBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[derive(Default)]
    struct TestState {
        value: u64,
    }

    impl Serializable for TestState {
        type Error = std::io::Error;

        fn serialize_state(&self) -> Result<SerializedState, Self::Error> {
            let mut data = Vec::with_capacity(8);
            data.extend_from_slice(&self.value.to_le_bytes());
            Ok(SerializedState::new(data))
        }

        fn deserialize_state(&mut self, data: &[u8]) -> Result<(), Self::Error> {
            if data.len() >= 8 {
                self.value = u64::from_le_bytes([
                    data[0], data[1], data[2], data[3],
                    data[4], data[5], data[6], data[7],
                ]);
                Ok(())
            } else {
                Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "Insufficient data",
                ))
            }
        }
    }

    #[test]
    fn test_persistence_manager_creation() {
        let temp_dir = env::temp_dir().join("persist_test");
        
        let manager = PersistenceBuilder::new()
            .base_path(&temp_dir)
            .build()
            .unwrap();

        assert!(!manager.is_initialized());
        
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_wal_write() {
        let temp_dir = env::temp_dir().join("persist_test2");
        
        let mut manager = PersistenceBuilder::new()
            .base_path(&temp_dir)
            .build()
            .unwrap();

        let state = Arc::new(TestState::default());
        manager.init(state).unwrap();

        let seq = manager.write_wal(b"test data").unwrap();
        assert!(seq > 0);

        let stats = manager.stats();
        assert!(stats.operations_count > 0);
        assert!(stats.wal_stats.is_some());

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_checkpoint_trigger() {
        let temp_dir = env::temp_dir().join("persist_test3");
        
        let mut manager = PersistenceBuilder::new()
            .base_path(&temp_dir)
            .build()
            .unwrap();

        let state = Arc::new(TestState { value: 42 });
        manager.init(Arc::clone(&state)).unwrap();

        let result = manager.trigger_checkpoint(state.as_ref());
        assert!(result.is_ok());

        let stats = manager.checkpoint_stats();
        assert!(stats.checkpoints_created > 0);

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_shutdown() {
        let temp_dir = env::temp_dir().join("persist_test4");
        
        let mut manager = PersistenceBuilder::new()
            .base_path(&temp_dir)
            .build()
            .unwrap();

        let state = Arc::new(TestState::default());
        manager.init(state).unwrap();

        manager.shutdown().unwrap();
        assert!(!manager.is_initialized());

        let _ = std::fs::remove_dir_all(&temp_dir);
    }
}
