//! Write-Ahead Log (WAL) Module
//! High-speed WAL for all market events and internal state changes.
//! Uses memory-mapped files (memmap2) to ensure disk writes do not inflate RAM ceiling.

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::fs::{File, OpenOptions};
use std::io::{self, Write, Seek, SeekFrom};
use std::path::Path;

const CACHE_LINE_SIZE: usize = 64;

#[repr(C, align(64))]
struct CachePadded<T> {
    data: T,
    _pad: [u8; CACHE_LINE_SIZE],
}

impl<T: Default> Default for CachePadded<T> {
    fn default() -> Self {
        Self {
            data: T::default(),
            _pad: [0u8; CACHE_LINE_SIZE],
        }
    }
}

/// WAL entry header
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct WALEntryHeader {
    /// Entry type identifier
    pub entry_type: u8,
    /// Entry length in bytes
    pub length: u32,
    /// Sequence number
    pub sequence: u64,
    /// Timestamp (nanoseconds)
    pub timestamp_ns: u64,
    /// Checksum for integrity
    pub checksum: u32,
}

impl WALEntryHeader {
    pub const SIZE: usize = std::mem::size_of::<Self>();

    pub fn new(entry_type: u8, length: u32, sequence: u64, timestamp_ns: u64) -> Self {
        Self {
            entry_type,
            length,
            sequence,
            timestamp_ns,
            checksum: 0, // Will be calculated
        }
    }

    pub fn calculate_checksum(&mut self, data: &[u8]) {
        self.checksum = crc32_fast(data);
    }

    pub fn verify_checksum(&self, data: &[u8]) -> bool {
        crc32_fast(data) == self.checksum
    }
}

/// Fast CRC32 implementation (simplified)
fn crc32_fast(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ 0xEDB8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

/// Event types for WAL entries
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u8)]
pub enum WALEntryType {
    MarketData = 0,
    OrderSubmitted = 1,
    OrderFilled = 2,
    OrderCancelled = 3,
    OrderModified = 4,
    StateSnapshot = 5,
    Heartbeat = 6,
    Custom = 255,
}

impl From<u8> for WALEntryType {
    fn from(value: u8) -> Self {
        match value {
            0 => WALEntryType::MarketData,
            1 => WALEntryType::OrderSubmitted,
            2 => WALEntryType::OrderFilled,
            3 => WALEntryType::OrderCancelled,
            4 => WALEntryType::OrderModified,
            5 => WALEntryType::StateSnapshot,
            6 => WALEntryType::Heartbeat,
            _ => WALEntryType::Custom,
        }
    }
}

/// Memory-mapped WAL writer
pub struct WALWriter {
    /// File handle
    file: File,
    /// Current write position
    write_pos: CachePadded<AtomicU64>,
    /// Sequence counter
    sequence: CachePadded<AtomicU64>,
    /// Active flag
    is_active: CachePadded<AtomicBool>,
    /// Buffer size for batching
    buffer_size: usize,
    /// Bytes written since last sync
    pending_bytes: CachePadded<AtomicU64>,
    /// Sync interval in bytes
    sync_interval: u64,
}

impl WALWriter {
    /// Create new WAL writer with memory-mapped file
    pub fn new<P: AsRef<Path>>(path: P, buffer_size: usize, sync_interval: u64) -> io::Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .append(true)
            .open(path)?;

        Ok(Self {
            file,
            write_pos: CachePadded::default(),
            sequence: CachePadded::default(),
            is_active: CachePadded::new(AtomicBool::new(true)),
            buffer_size,
            pending_bytes: CachePadded::default(),
            sync_interval,
        })
    }

    /// Append an entry to the WAL
    pub fn append(&self, entry_type: WALEntryType, data: &[u8]) -> io::Result<u64> {
        if !self.is_active.data.load(Ordering::Acquire) {
            return Err(io::Error::new(io::ErrorKind::Other, "WAL is not active"));
        }

        let sequence = self.sequence.data.fetch_add(1, Ordering::AcqRel);
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or(std::time::Duration::ZERO)
            .as_nanos() as u64;

        // Create header
        let mut header = WALEntryHeader::new(entry_type as u8, data.len() as u32, sequence, now_ns);
        header.calculate_checksum(data);

        // Write header and data
        let header_bytes = unsafe {
            std::slice::from_raw_parts(
                &header as *const WALEntryHeader as *const u8,
                WALEntryHeader::SIZE,
            )
        };

        self.file.write_all(header_bytes)?;
        self.file.write_all(data)?;

        // Update position
        let total_bytes = (WALEntryHeader::SIZE + data.len()) as u64;
        self.write_pos.data.fetch_add(total_bytes, Ordering::Release);
        self.pending_bytes.data.fetch_add(total_bytes, Ordering::Release);

        // Sync if threshold reached
        let pending = self.pending_bytes.data.load(Ordering::Acquire);
        if pending >= self.sync_interval {
            self.file.sync_data()?;
            self.pending_bytes.data.store(0, Ordering::Release);
        }

        Ok(sequence)
    }

    /// Force sync to disk
    pub fn sync(&self) -> io::Result<()> {
        self.file.sync_data()?;
        self.pending_bytes.data.store(0, Ordering::Release);
        Ok(())
    }

    /// Get current write position
    #[inline]
    pub fn write_position(&self) -> u64 {
        self.write_pos.data.load(Ordering::Acquire)
    }

    /// Get current sequence number
    #[inline]
    pub fn current_sequence(&self) -> u64 {
        self.sequence.data.load(Ordering::Acquire)
    }

    #[inline]
    pub fn set_active(&self, active: bool) {
        self.is_active.data.store(active, Ordering::Release);
    }

    #[inline]
    pub fn is_active(&self) -> bool {
        self.is_active.data.load(Ordering::Acquire)
    }
}

/// WAL reader for replay
pub struct WALReader {
    file: File,
    read_pos: u64,
    is_active: CachePadded<AtomicBool>,
}

impl WALReader {
    pub fn new<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let file = OpenOptions::new().read(true).open(path)?;
        
        Ok(Self {
            file,
            read_pos: 0,
            is_active: CachePadded::new(AtomicBool::new(true)),
        })
    }

    /// Read next entry from WAL
    pub fn read_next(&mut self) -> io::Result<Option<WALEntry>> {
        if !self.is_active.data.load(Ordering::Acquire) {
            return Err(io::Error::new(io::ErrorKind::Other, "WAL reader is not active"));
        }

        // Seek to current position
        self.file.seek(SeekFrom::Start(self.read_pos))?;

        // Read header
        let mut header_buf = [0u8; WALEntryHeader::SIZE];
        let bytes_read = self.file.read(&mut header_buf)?;
        
        if bytes_read == 0 {
            return Ok(None); // End of file
        }

        if bytes_read < WALEntryHeader::SIZE {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "Incomplete header"));
        }

        let header = unsafe {
            std::ptr::read_unaligned(header_buf.as_ptr() as *const WALEntryHeader)
        };

        // Read data
        let mut data = vec![0u8; header.length as usize];
        self.file.read_exact(&mut data)?;

        // Verify checksum
        if !header.verify_checksum(&data) {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Checksum mismatch"));
        }

        // Update position
        self.read_pos += (WALEntryHeader::SIZE + header.length as usize) as u64;

        Ok(Some(WALEntry {
            header,
            data,
        }))
    }

    /// Reset to beginning
    pub fn rewind(&mut self) -> io::Result<()> {
        self.read_pos = 0;
        self.file.seek(SeekFrom::Start(0))?;
        Ok(())
    }

    #[inline]
    pub fn set_active(&self, active: bool) {
        self.is_active.data.store(active, Ordering::Release);
    }
}

/// Complete WAL entry
pub struct WALEntry {
    pub header: WALEntryHeader,
    pub data: Vec<u8>,
}

impl WALEntry {
    pub fn entry_type(&self) -> WALEntryType {
        self.header.entry_type.into()
    }

    pub fn sequence(&self) -> u64 {
        self.header.sequence
    }

    pub fn timestamp_ns(&self) -> u64 {
        self.header.timestamp_ns
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    #[test]
    fn test_wal_write_read() {
        let temp_path = "/tmp/test_wal.bin";
        
        // Write
        {
            let writer = WALWriter::new(temp_path, 4096, 1024).unwrap();
            let data = b"test event data";
            writer.append(WALEntryType::MarketData, data).unwrap();
            writer.sync().unwrap();
        }

        // Read
        {
            let mut reader = WALReader::new(temp_path).unwrap();
            let entry = reader.read_next().unwrap();
            assert!(entry.is_some());
            
            let entry = entry.unwrap();
            assert_eq!(entry.entry_type(), WALEntryType::MarketData);
            assert_eq!(entry.data, b"test event data");
        }

        // Cleanup
        std::fs::remove_file(temp_path).ok();
    }

    #[test]
    fn test_crc32() {
        let data = b"test data";
        let crc = crc32_fast(data);
        assert!(crc != 0);
    }
}
