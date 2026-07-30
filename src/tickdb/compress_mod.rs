//! Compression Module Root
//! 
//! Integrates Gorilla compression with memory-mapped TickDB writer and WAL.

pub mod compression;
pub mod xor_encoding;

pub use compression::{
    FloatCompressionStats, GorillaFloatCompressor, GorillaTimestampCompressor,
    TickCompressedData, TickCompressor, TimestampCompressionStats, TimestampDecompressor,
};
pub use xor_encoding::{
    CompressedBlock, SimdXorCompressor, StreamingXorCompressor, XorBlockCompressor,
    XorBlockDecompressor, XorCompressionStats,
};

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use tracing::{debug, error, info};

/// Maximum tick file size before rotation (1GB)
const MAX_TICK_FILE_SIZE: u64 = 1024 * 1024 * 1024;

/// Configuration for compressed tick storage
#[derive(Clone, Debug)]
pub struct CompressionConfig {
    /// Base directory for tick data
    pub base_dir: PathBuf,
    /// Block size for XOR compression
    pub block_size: usize,
    /// Enable WAL for durability
    pub enable_wal: bool,
    /// WAL flush interval (ms)
    pub wal_flush_interval_ms: u64,
    /// Maximum RAM for compression buffers
    pub max_buffer_ram_mb: usize,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            base_dir: PathBuf::from("/tmp/tickdb"),
            block_size: 256,
            enable_wal: true,
            wal_flush_interval_ms: 100,
            max_buffer_ram_mb: 512,
        }
    }
}

/// Compressed tick database writer
pub struct CompressedTickWriter {
    config: CompressionConfig,
    current_file: Option<BufWriter<File>>,
    current_file_size: u64,
    current_file_path: Option<PathBuf>,
    compressor: TickCompressor,
    ticks_written: u64,
    wal_writer: Option<WalWriter>,
}

impl CompressedTickWriter {
    /// Create new compressed tick writer
    pub fn new(config: CompressionConfig) -> std::io::Result<Self> {
        // Ensure base directory exists
        std::fs::create_dir_all(&config.base_dir)?;

        let wal_writer = if config.enable_wal {
            Some(WalWriter::new(&config.base_dir.join("tick.wal"))?)
        } else {
            None
        };

        Ok(Self {
            config,
            current_file: None,
            current_file_size: 0,
            current_file_path: None,
            compressor: TickCompressor::new(),
            ticks_written: 0,
            wal_writer,
        })
    }

    /// Write a tick to the database
    pub fn write_tick(&mut self, timestamp: u64, price: f64, size: f64) -> std::io::Result<()> {
        // Add to compressor
        self.compressor.add_tick(timestamp, price, size);
        self.ticks_written += 1;

        // Write to WAL if enabled
        if let Some(ref mut wal) = self.wal_writer {
            wal.write_entry(timestamp, price, size)?;
        }

        // Check if we should flush
        if self.should_flush() {
            self.flush()?;
        }

        Ok(())
    }

    /// Check if compression buffer should be flushed
    fn should_flush(&self) -> bool {
        // Flush when estimated compressed size exceeds threshold
        self.compressor.estimated_ratio() > 0.0
            && self.ticks_written % 1000 == 0
    }

    /// Flush compressed data to disk
    pub fn flush(&mut self) -> std::io::Result<()> {
        if self.ticks_written == 0 {
            return Ok(());
        }

        // Finalize compression
        let compressed = self.compressor.finalize();

        // Get or create output file
        if self.current_file.is_none() {
            self.rotate_file()?;
        }

        if let Some(ref mut writer) = self.current_file {
            // Write header: magic + timestamp count
            let header = [0x5449434B, compressed.ticks_count as u32]; // "TICK" magic
            writer.write_all(&header[0].to_le_bytes())?;
            writer.write_all(&header[1].to_le_bytes())?;

            // Write compressed sections
            writer.write_all(&(compressed.timestamps.len() as u32).to_le_bytes())?;
            writer.write_all(&compressed.timestamps)?;

            writer.write_all(&(compressed.prices.len() as u32).to_le_bytes())?;
            writer.write_all(&compressed.prices)?;

            writer.write_all(&(compressed.sizes.len() as u32).to_le_bytes())?;
            writer.write_all(&compressed.sizes)?;

            writer.flush()?;

            self.current_file_size += (16 + compressed.timestamps.len() + compressed.prices.len() + compressed.sizes.len()) as u64;
        }

        // Reset compressor
        self.compressor.reset();

        // Check if file rotation needed
        if self.current_file_size >= MAX_TICK_FILE_SIZE {
            self.rotate_file()?;
        }

        debug!("Flushed {} ticks to disk", self.ticks_written);
        Ok(())
    }

    /// Rotate to new file
    fn rotate_file(&mut self) -> std::io::Result<()> {
        // Close current file
        if let Some(ref mut writer) = self.current_file {
            writer.flush()?;
        }

        // Create new file with timestamp-based name
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap();
        let filename = format!("ticks_{}.bin", now.as_secs());
        let path = self.config.base_dir.join(&filename);

        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&path)?;

        self.current_file = Some(BufWriter::with_capacity(1024 * 1024, file));
        self.current_file_path = Some(path);
        self.current_file_size = 0;

        info!("Rotated tick file to {}", filename);
        Ok(())
    }

    /// Get statistics
    pub fn stats(&self) -> WriterStats {
        WriterStats {
            ticks_written: self.ticks_written,
            current_file_size: self.current_file_size,
            estimated_ratio: self.compressor.estimated_ratio(),
        }
    }

    /// Close writer gracefully
    pub fn close(mut self) -> std::io::Result<()> {
        self.flush()?;
        
        if let Some(ref mut wal) = self.wal_writer {
            wal.close()?;
        }

        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct WriterStats {
    pub ticks_written: u64,
    pub current_file_size: u64,
    pub estimated_ratio: f64,
}

/// Write-Ahead Log for durability
pub struct WalWriter {
    file: BufWriter<File>,
    sequence: u64,
    pending_bytes: usize,
}

impl WalWriter {
    /// Create new WAL writer
    pub fn new(path: &Path) -> std::io::Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;

        Ok(Self {
            file: BufWriter::with_capacity(8192, file),
            sequence: 0,
            pending_bytes: 0,
        })
    }

    /// Write WAL entry
    pub fn write_entry(&mut self, timestamp: u64, price: f64, size: f64) -> std::io::Result<()> {
        // Entry format: seq(8) + ts(8) + price(8) + size(8) + checksum(4)
        let mut buf = [0u8; 36];
        
        buf[0..8].copy_from_slice(&self.sequence.to_le_bytes());
        buf[8..16].copy_from_slice(&timestamp.to_le_bytes());
        buf[16..24].copy_from_slice(&price.to_bits().to_le_bytes());
        buf[24..32].copy_from_slice(&size.to_bits().to_le_bytes());
        
        // Simple CRC32 checksum
        let checksum = crc32(&buf[0..32]);
        buf[32..36].copy_from_slice(&checksum.to_le_bytes());

        self.file.write_all(&buf)?;
        self.pending_bytes += buf.len();
        self.sequence += 1;

        // Auto-flush every 4KB
        if self.pending_bytes >= 4096 {
            self.flush()?;
        }

        Ok(())
    }

    /// Flush WAL to disk
    pub fn flush(&mut self) -> std::io::Result<()> {
        self.file.flush()?;
        self.pending_bytes = 0;
        Ok(())
    }

    /// Close WAL
    pub fn close(mut self) -> std::io::Result<()> {
        self.flush()?;
        Ok(())
    }

    /// Get current sequence number
    pub fn sequence(&self) -> u64 {
        self.sequence
    }
}

/// Simple CRC32 implementation
fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFFFFFFu32;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            crc = (crc >> 1) ^ ((crc & 1) * 0xEDB88320);
        }
    }
    !crc
}

/// Reader for compressed tick files
pub struct CompressedTickReader {
    // Implementation would mirror writer for decompression
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_compressed_writer() {
        let temp_dir = std::env::temp_dir().join("tickdb_test");
        fs::create_dir_all(&temp_dir).ok();

        let config = CompressionConfig {
            base_dir: temp_dir.clone(),
            ..Default::default()
        };

        let mut writer = CompressedTickWriter::new(config).unwrap();

        // Write some ticks
        for i in 0..100 {
            writer
                .write_tick(
                    1000000 + i * 5,
                    100.0 + (i % 10) as f64 * 0.01,
                    1.0 + (i % 5) as f64 * 0.1,
                )
                .unwrap();
        }

        let stats = writer.stats();
        assert_eq!(stats.ticks_written, 100);
        assert!(stats.estimated_ratio > 1.0);

        writer.close().unwrap();
        fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn test_crc32() {
        let data = b"hello world";
        let checksum = crc32(data);
        assert_ne!(checksum, 0);
    }
}
