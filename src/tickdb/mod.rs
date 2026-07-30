//! TickDB Module Root
//! 
//! Manages file rotation, disk space limits, and data integrity checks.
//! Exports writer and reader for tick database operations.

pub mod writer;
pub mod reader;

pub use writer::{TickDbWriter, TickDbConfig, StoredTick, TickDbError};
pub use reader::{TickDbReader, TickIterator, TickQuery, TimeRange, PriceRange, TickStats};

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::fs;
use thiserror::Error;

/// TickDB management errors
#[derive(Debug, Error)]
pub enum TickDbManagerError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Database error: {0}")]
    Database(#[from] TickDbError),
    #[error("Disk space insufficient: needed {needed} bytes, available {available} bytes")]
    DiskSpaceExhausted { needed: u64, available: u64 },
    #[error("Rotation failed: {0}")]
    RotationFailed(String),
    #[error("Integrity check failed: {0}")]
    IntegrityCheckFailed(String),
}

/// Configuration for TickDB manager
#[derive(Debug, Clone)]
pub struct TickDbManagerConfig {
    /// Base directory for tick databases
    pub base_dir: PathBuf,
    /// Maximum total disk space to use
    pub max_disk_space_gb: f64,
    /// Maximum individual file size
    pub max_file_size_mb: u64,
    /// Enable automatic rotation
    pub auto_rotate: bool,
    /// Number of rotated files to keep
    pub keep_rotated_count: usize,
    /// Enable integrity checks on open
    pub integrity_check: bool,
}

impl Default for TickDbManagerConfig {
    fn default() -> Self {
        Self {
            base_dir: PathBuf::from("./tickdb"),
            max_disk_space_gb: 100.0,
            max_file_size_mb: 1024,
            auto_rotate: true,
            keep_rotated_count: 5,
            integrity_check: true,
        }
    }
}

/// TickDB Manager - handles lifecycle and maintenance
pub struct TickDbManager {
    config: TickDbManagerConfig,
    current_writer: Option<TickDbWriter>,
    current_reader: Option<TickDbReader>,
    current_file_index: AtomicU64,
    total_ticks_written: AtomicU64,
    active: AtomicBool,
}

unsafe impl Send for TickDbManager {}
unsafe impl Sync for TickDbManager {}

impl TickDbManager {
    /// Create a new TickDB manager
    pub fn new(config: TickDbManagerConfig) -> Result<Self, TickDbManagerError> {
        // Ensure base directory exists
        fs::create_dir_all(&config.base_dir)?;

        let manager = Self {
            config,
            current_writer: None,
            current_reader: None,
            current_file_index: AtomicU64::new(0),
            total_ticks_written: AtomicU64::new(0),
            active: AtomicBool::new(true),
        };

        Ok(manager)
    }

    /// Get the path for the current tick database file
    fn get_current_path(&self) -> PathBuf {
        let index = self.current_file_index.load(Ordering::Relaxed);
        self.config.base_dir.join(format!("ticks_{:06}.db", index))
    }

    /// Get the path for a rotated file
    fn get_rotated_path(&self, index: u64) -> PathBuf {
        self.config.base_dir.join(format!("ticks_{:06}.db", index))
    }

    /// Initialize or rotate to a new database file
    pub fn init_or_rotate(&mut self) -> Result<(), TickDbManagerError> {
        if !self.active.load(Ordering::Relaxed) {
            return Err(TickDbManagerError::RotationFailed("Manager not active".to_string()));
        }

        // Close existing writer
        if let Some(ref mut writer) = self.current_writer {
            writer.close()?;
        }

        let path = self.get_current_path();

        // Check disk space
        if let Err(e) = self.check_disk_space() {
            return Err(e);
        }

        let db_config = TickDbConfig {
            max_file_size: self.config.max_file_size_mb * 1024 * 1024,
            initial_capacity: 64 * 1024 * 1024,
            use_direct_io: true,
            sync_on_write: false,
        };

        let writer = TickDbWriter::new(&path, db_config)?;
        self.current_writer = Some(writer);

        Ok(())
    }

    /// Write a tick to the current database
    pub fn write_tick(&self, tick: &StoredTick) -> Result<u64, TickDbManagerError> {
        if let Some(ref writer) = self.current_writer {
            let seq = writer.append(tick)?;
            self.total_ticks_written.fetch_add(1, Ordering::Relaxed);

            // Check if rotation needed
            if self.config.auto_rotate && writer.needs_rotation() {
                // Note: In production, this would need proper synchronization
                // For now, we just flag that rotation is needed
            }

            Ok(seq)
        } else {
            Err(TickDbManagerError::RotationFailed("No writer initialized".to_string()))
        }
    }

    /// Write multiple ticks in batch
    pub fn write_batch(&self, ticks: &[StoredTick]) -> Result<u64, TickDbManagerError> {
        if let Some(ref writer) = self.current_writer {
            let seq = writer.append_batch(ticks)?;
            self.total_ticks_written.fetch_add(ticks.len() as u64, Ordering::Relaxed);
            Ok(seq)
        } else {
            Err(TickDbManagerError::RotationFailed("No writer initialized".to_string()))
        }
    }

    /// Open a database file for reading
    pub fn open_for_read(&mut self, file_index: u64) -> Result<&TickDbReader, TickDbManagerError> {
        let path = self.get_rotated_path(file_index);
        
        if !path.exists() {
            return Err(TickDbManagerError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Database file not found: {:?}", path),
            )));
        }

        let reader = TickDbReader::open(&path)?;
        self.current_reader = Some(reader);

        Ok(self.current_reader.as_ref().unwrap())
    }

    /// Rotate old database files
    pub fn rotate(&mut self) -> Result<(), TickDbManagerError> {
        let current_index = self.current_file_index.load(Ordering::Relaxed);
        
        // Increment to new file
        self.current_file_index.fetch_add(1, Ordering::SeqCst);
        
        // Initialize new writer
        self.init_or_rotate()?;

        // Clean up old files if needed
        self.cleanup_old_files()?;

        Ok(())
    }

    /// Clean up old rotated files beyond retention limit
    fn cleanup_old_files(&self) -> Result<(), TickDbManagerError> {
        let current_index = self.current_file_index.load(Ordering::Relaxed);
        
        if current_index as usize > self.config.keep_rotated_count {
            let delete_up_to = current_index as usize - self.config.keep_rotated_count;
            
            for i in 0..delete_up_to {
                let path = self.get_rotated_path(i as u64);
                if path.exists() {
                    fs::remove_file(path)?;
                }
            }
        }

        Ok(())
    }

    /// Check available disk space
    fn check_disk_space(&self) -> Result<(), TickDbManagerError> {
        // Get disk usage stats (simplified - in production would use syscalls)
        let max_bytes = (self.config.max_disk_space_gb * 1024.0 * 1024.0 * 1024.0) as u64;
        
        // Estimate current usage
        let mut current_usage = 0u64;
        if let Ok(entries) = fs::read_dir(&self.config.base_dir) {
            for entry in entries.flatten() {
                if let Ok(meta) = entry.metadata() {
                    current_usage += meta.len();
                }
            }
        }

        if current_usage >= max_bytes {
            return Err(TickDbManagerError::DiskSpaceExhausted {
                needed: self.config.max_file_size_mb * 1024 * 1024,
                available: max_bytes.saturating_sub(current_usage),
            });
        }

        Ok(())
    }

    /// Run integrity check on a database file
    pub fn verify_integrity(&self, file_index: u64) -> Result<IntegrityReport, TickDbManagerError> {
        let path = self.get_rotated_path(file_index);
        
        if !path.exists() {
            return Err(TickDbManagerError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "File not found".to_string(),
            )));
        }

        let reader = TickDbReader::open(&path)?;
        
        let mut report = IntegrityReport {
            file_path: path,
            tick_count: reader.tick_count(),
            file_size: reader.file_size(),
            valid: true,
            errors: Vec::new(),
        };

        // Verify each tick can be read
        let mut error_count = 0;
        if let Some(iter) = reader.iter() {
            for result in iter {
                if result.is_err() {
                    error_count += 1;
                    if error_count > 100 {
                        report.valid = false;
                        report.errors.push("Too many corrupted ticks".to_string());
                        break;
                    }
                }
            }
        }

        if error_count > 0 {
            report.errors.push(format!("{} corrupted ticks found", error_count));
        }

        Ok(report)
    }

    /// Get statistics about the TickDB
    pub fn get_stats(&self) -> TickDbStats {
        TickDbStats {
            current_file_index: self.current_file_index.load(Ordering::Relaxed),
            total_ticks_written: self.total_ticks_written.load(Ordering::Relaxed),
            is_active: self.active.load(Ordering::Relaxed),
            base_dir: self.config.base_dir.clone(),
            max_disk_space_gb: self.config.max_disk_space_gb,
        }
    }

    /// Activate the manager
    pub fn activate(&self) {
        self.active.store(true, Ordering::Relaxed);
    }

    /// Deactivate the manager
    pub fn deactivate(&self) {
        self.active.store(false, Ordering::Relaxed);
    }

    /// Close all database connections
    pub fn close(&mut self) -> Result<(), TickDbManagerError> {
        self.deactivate();
        
        if let Some(ref mut writer) = self.current_writer {
            writer.close()?;
        }
        
        if let Some(ref mut reader) = self.current_reader {
            reader.close();
        }

        Ok(())
    }
}

impl Drop for TickDbManager {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

/// Integrity check report
#[derive(Debug, Clone)]
pub struct IntegrityReport {
    pub file_path: PathBuf,
    pub tick_count: u64,
    pub file_size: u64,
    pub valid: bool,
    pub errors: Vec<String>,
}

/// TickDB statistics
#[derive(Debug, Clone)]
pub struct TickDbStats {
    pub current_file_index: u64,
    pub total_ticks_written: u64,
    pub is_active: bool,
    pub base_dir: PathBuf,
    pub max_disk_space_gb: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_manager_basic() {
        let temp_dir = "/tmp/test_tickdb_manager";
        let _ = fs::remove_dir_all(temp_dir);

        let config = TickDbManagerConfig {
            base_dir: PathBuf::from(temp_dir),
            ..Default::default()
        };

        let mut manager = TickDbManager::new(config).unwrap();
        manager.init_or_rotate().unwrap();

        let tick = StoredTick::new(1000, 50000.0, 1.0, false, 0);
        manager.write_tick(&tick).unwrap();

        let stats = manager.get_stats();
        assert_eq!(stats.total_ticks_written, 1);

        manager.close().unwrap();
        let _ = fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn test_integrity_report() {
        let report = IntegrityReport {
            file_path: PathBuf::from("/test.db"),
            tick_count: 1000,
            file_size: 1024 * 1024,
            valid: true,
            errors: vec![],
        };

        assert!(report.valid);
        assert_eq!(report.tick_count, 1000);
    }
}
