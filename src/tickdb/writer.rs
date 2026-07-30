//! Trade Tick Database Writer
//! 
//! High-throughput, append-only disk writer for trade ticks using memory-mapped files.
//! Bypasses OS page cache where possible to ensure tick persistence does not introduce latency spikes.

use std::fs::{File, OpenOptions};
use std::io::{Write, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use memmap2::{MmapMut, MmapOptions};
use thiserror::Error;
use serde::{Serialize, Deserialize};

/// Errors that can occur in tick database operations
#[derive(Debug, Error)]
pub enum TickDbError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("File mapping error: {0}")]
    MmapError(String),
    #[error("File size limit exceeded")]
    SizeLimitExceeded,
    #[error("Invalid tick data: {0}")]
    InvalidTickData(String),
    #[error("Database corrupted: {0}")]
    Corruption(String),
    #[error("Database closed")]
    DatabaseClosed,
}

/// Trade tick structure optimized for storage
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct StoredTick {
    pub timestamp_ns: u64,
    pub price: f64,
    pub volume: f64,
    pub is_buyer_maker: bool,
    pub sequence: u64,
}

impl StoredTick {
    pub fn new(timestamp_ns: u64, price: f64, volume: f64, is_buyer_maker: bool, sequence: u64) -> Self {
        Self {
            timestamp_ns,
            price,
            volume,
            is_buyer_maker,
            sequence,
        }
    }

    /// Serialized size in bytes (fixed for performance)
    pub const SERIALIZE_SIZE: usize = 8 + 8 + 8 + 1 + 8; // 33 bytes
}

/// File header for tick database
#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct FileHeader {
    magic: u32,
    version: u32,
    tick_count: u64,
    file_size: u64,
    created_timestamp: u64,
    checksum: u32,
}

impl FileHeader {
    const MAGIC: u32 = 0x5449434B; // "TICK"
    const VERSION: u32 = 1;
    const SIZE: usize = std::mem::size_of::<Self>();

    fn new() -> Self {
        Self {
            magic: Self::MAGIC,
            version: Self::VERSION,
            tick_count: 0,
            file_size: 0,
            created_timestamp: 0,
            checksum: 0,
        }
    }

    fn calculate_checksum(&self) -> u32 {
        // Simple XOR-based checksum
        self.magic as u32 
            ^ self.version 
            ^ (self.tick_count as u32) 
            ^ (self.tick_count >> 32) as u32
            ^ (self.file_size as u32)
            ^ (self.file_size >> 32) as u32
            ^ (self.created_timestamp as u32)
            ^ (self.created_timestamp >> 32) as u32
    }

    fn validate(&self) -> Result<(), TickDbError> {
        if self.magic != Self::MAGIC {
            return Err(TickDbError::Corruption("Invalid magic number".to_string()));
        }
        if self.version != Self::VERSION {
            return Err(TickDbError::Corruption("Unsupported version".to_string()));
        }
        let expected = self.calculate_checksum();
        if self.checksum != expected {
            return Err(TickDbError::Corruption("Checksum mismatch".to_string()));
        }
        Ok(())
    }
}

/// Configuration for tick database writer
#[derive(Debug, Clone)]
pub struct TickDbConfig {
    pub max_file_size: u64,
    pub initial_capacity: u64,
    pub use_direct_io: bool,
    pub sync_on_write: bool,
}

impl Default for TickDbConfig {
    fn default() -> Self {
        Self {
            max_file_size: 1024 * 1024 * 1024, // 1GB
            initial_capacity: 64 * 1024 * 1024, // 64MB
            use_direct_io: true,
            sync_on_write: false,
        }
    }
}

/// High-performance tick database writer
pub struct TickDbWriter {
    /// Path to the database file
    path: PathBuf,
    /// Memory-mapped file
    mmap: Option<MmapMut>,
    /// Current write position
    write_pos: AtomicU64,
    /// Number of ticks written
    tick_count: AtomicU64,
    /// Sequence counter
    sequence: AtomicU64,
    /// File handle
    file: Option<File>,
    /// Configuration
    config: TickDbConfig,
    /// Closed flag
    closed: AtomicBool,
    /// Current file size
    file_size: AtomicU64,
}

unsafe impl Send for TickDbWriter {}
unsafe impl Sync for TickDbWriter {}

impl TickDbWriter {
    /// Create a new tick database writer
    pub fn new<P: AsRef<Path>>(path: P, config: TickDbConfig) -> Result<Self, TickDbError> {
        let path = path.as_ref().to_path_buf();

        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let mut writer = Self {
            path,
            mmap: None,
            write_pos: AtomicU64::new(0),
            tick_count: AtomicU64::new(0),
            sequence: AtomicU64::new(0),
            file: None,
            config,
            closed: AtomicBool::new(false),
            file_size: AtomicU64::new(0),
        };

        writer.initialize()?;
        Ok(writer)
    }

    /// Initialize the database file
    fn initialize(&mut self) -> Result<(), TickDbError> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&self.path)?;

        let metadata = file.metadata()?;
        let file_len = metadata.len();

        if file_len == 0 {
            // New file - initialize with header and pre-allocate space
            file.set_len(self.config.initial_capacity)?;

            let mut mmap = unsafe {
                MmapOptions::new()
                    .map_mut(&file)?
            };

            // Write header
            let mut header = FileHeader::new();
            header.created_timestamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64;
            header.checksum = header.calculate_checksum();

            mmap[0..FileHeader::SIZE].copy_from_slice(unsafe {
                std::slice::from_raw_parts(&header as *const FileHeader as *const u8, FileHeader::SIZE)
            });

            mmap.flush()?;

            self.mmap = Some(mmap);
            self.write_pos.store(FileHeader::SIZE as u64, Ordering::Relaxed);
        } else {
            // Existing file - validate and map
            let mut mmap = unsafe {
                MmapOptions::new()
                    .map_mut(&file)?
            };

            // Validate header
            let header = unsafe {
                &*(mmap[0..FileHeader::SIZE].as_ptr() as *const FileHeader)
            };
            header.validate()?;

            self.mmap = Some(mmap);
            self.tick_count.store(header.tick_count, Ordering::Relaxed);
            self.sequence.store(header.tick_count, Ordering::Relaxed);
            self.write_pos.store(file_len, Ordering::Relaxed);
        }

        self.file = Some(file);
        self.file_size.store(self.write_pos.load(Ordering::Relaxed), Ordering::Relaxed);

        Ok(())
    }

    /// Append a tick to the database (lock-free hot path)
    #[inline]
    pub fn append(&self, tick: &StoredTick) -> Result<u64, TickDbError> {
        if self.closed.load(Ordering::Relaxed) {
            return Err(TickDbError::DatabaseClosed);
        }

        // Validate tick
        if tick.price <= 0.0 || tick.volume <= 0.0 {
            return Err(TickDbError::InvalidTickData(
                "Price and volume must be positive".to_string(),
            ));
        }

        let current_pos = self.write_pos.load(Ordering::Relaxed);
        let tick_data = bincode::serialize(tick)
            .map_err(|e| TickDbError::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?;

        let new_pos = current_pos.checked_add(tick_data.len() as u64)
            .ok_or(TickDbError::SizeLimitExceeded)?;

        // Check size limit
        if new_pos > self.config.max_file_size {
            return Err(TickDbError::SizeLimitExceeded);
        }

        // Write to mmap (lock-free for single writer)
        if let Some(mmap) = &self.mmap {
            let mmap_slice = &mut (**mmap);
            if new_pos as usize <= mmap_slice.len() {
                mmap_slice[current_pos as usize..new_pos as usize].copy_from_slice(&tick_data);
            } else {
                // Need to expand - this requires synchronization
                return self.expand_and_write(current_pos, &tick_data);
            }
        } else {
            return Err(TickDbError::DatabaseClosed);
        }

        // Update counters
        self.write_pos.store(new_pos, Ordering::Release);
        self.tick_count.fetch_add(1, Ordering::Relaxed);
        self.file_size.store(new_pos, Ordering::Relaxed);

        // Optional sync
        if self.config.sync_on_write {
            if let Some(mmap) = &self.mmap {
                mmap.flush_async_range(current_pos as usize, tick_data.len())?;
            }
        }

        Ok(self.sequence.fetch_add(1, Ordering::Relaxed))
    }

    /// Expand file and write (requires synchronization)
    fn expand_and_write(&self, pos: u64, data: &[u8]) -> Result<u64, TickDbError> {
        // This would need proper locking in production
        // For now, we'll just return an error to trigger rotation
        Err(TickDbError::SizeLimitExceeded)
    }

    /// Append multiple ticks in batch
    pub fn append_batch(&self, ticks: &[StoredTick]) -> Result<u64, TickDbError> {
        if self.closed.load(Ordering::Relaxed) {
            return Err(TickDbError::DatabaseClosed);
        }

        let mut serialized = Vec::with_capacity(ticks.len() * StoredTick::SERIALIZE_SIZE);
        for tick in ticks {
            if tick.price <= 0.0 || tick.volume <= 0.0 {
                return Err(TickDbError::InvalidTickData(
                    "All ticks must have positive price and volume".to_string(),
                ));
            }
            bincode::serialize_into(&mut serialized, tick)
                .map_err(|e| TickDbError::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?;
        }

        let current_pos = self.write_pos.load(Ordering::Relaxed);
        let new_pos = current_pos.checked_add(serialized.len() as u64)
            .ok_or(TickDbError::SizeLimitExceeded)?;

        if new_pos > self.config.max_file_size {
            return Err(TickDbError::SizeLimitExceeded);
        }

        if let Some(mmap) = &self.mmap {
            let mmap_slice = &mut (**mmap);
            if new_pos as usize <= mmap_slice.len() {
                mmap_slice[current_pos as usize..new_pos as usize].copy_from_slice(&serialized);
            } else {
                return Err(TickDbError::SizeLimitExceeded);
            }
        }

        self.write_pos.store(new_pos, Ordering::Release);
        self.tick_count.fetch_add(ticks.len() as u64, Ordering::Relaxed);
        self.file_size.store(new_pos, Ordering::Relaxed);

        Ok(self.sequence.load(Ordering::Relaxed))
    }

    /// Flush pending writes to disk
    pub fn flush(&self) -> Result<(), TickDbError> {
        if let Some(mmap) = &self.mmap {
            mmap.flush()?;
        }
        
        // Update header
        self.update_header()?;
        
        // Sync file
        if let Some(file) = &self.file {
            file.sync_all()?;
        }

        Ok(())
    }

    /// Update file header with current state
    fn update_header(&self) -> Result<(), TickDbError> {
        if let Some(mmap) = &self.mmap {
            let mut header = FileHeader::new();
            header.tick_count = self.tick_count.load(Ordering::Relaxed);
            header.file_size = self.file_size.load(Ordering::Relaxed);
            header.created_timestamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64;
            header.checksum = header.calculate_checksum();

            let mmap_slice = &mut (**mmap);
            mmap_slice[0..FileHeader::SIZE].copy_from_slice(unsafe {
                std::slice::from_raw_parts(&header as *const FileHeader as *const u8, FileHeader::SIZE)
            });
        }
        Ok(())
    }

    /// Get current tick count
    pub fn tick_count(&self) -> u64 {
        self.tick_count.load(Ordering::Relaxed)
    }

    /// Get current file size
    pub fn file_size(&self) -> u64 {
        self.file_size.load(Ordering::Relaxed)
    }

    /// Check if rotation is needed
    pub fn needs_rotation(&self) -> bool {
        self.file_size.load(Ordering::Relaxed) >= self.config.max_file_size
    }

    /// Close the database
    pub fn close(&mut self) -> Result<(), TickDbError> {
        self.closed.store(true, Ordering::SeqCst);
        self.flush()?;
        
        // Drop mmap first
        self.mmap = None;
        self.file = None;

        Ok(())
    }
}

impl Drop for TickDbWriter {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_writer_basic() {
        let temp_path = "/tmp/test_tickdb_writer.db";
        let _ = fs::remove_file(temp_path);

        let config = TickDbConfig::default();
        let writer = TickDbWriter::new(temp_path, config).unwrap();

        let tick = StoredTick::new(1000, 50000.0, 1.5, false, 0);
        let seq = writer.append(&tick).unwrap();

        assert_eq!(seq, 0);
        assert_eq!(writer.tick_count(), 1);

        let _ = fs::remove_file(temp_path);
    }

    #[test]
    fn test_writer_batch() {
        let temp_path = "/tmp/test_tickdb_batch.db";
        let _ = fs::remove_file(temp_path);

        let config = TickDbConfig::default();
        let writer = TickDbWriter::new(temp_path, config).unwrap();

        let ticks = vec![
            StoredTick::new(1000, 50000.0, 1.0, false, 0),
            StoredTick::new(2000, 50001.0, 2.0, true, 1),
            StoredTick::new(3000, 50002.0, 1.5, false, 2),
        ];

        writer.append_batch(&ticks).unwrap();
        assert_eq!(writer.tick_count(), 3);

        let _ = fs::remove_file(temp_path);
    }
}
