//! IPC Shared Memory Module
//! 
//! Implements zero-copy POSIX/Windows shared memory for passing real-time
//! feature vectors from Rust to Python/Ray ML backend without serialization overhead.
//! Allocates a fixed 512MB buffer using the `shared_memory` crate.

use std::{
    fs,
    io,
    marker::PhantomData,
    ptr,
    sync::atomic::{AtomicBool, AtomicU64, Ordering},
};

use shared_memory::{Shmem, ShmemConf};

/// Cache-line padding constant
const CACHE_LINE_SIZE: usize = 64;

/// Shared memory segment size: 512MB for feature vectors
const SHARED_MEMORY_SIZE: usize = 512 * 1024 * 1024;

/// Magic number for shared memory validation
const SHMEM_MAGIC: u64 = 0x48465453484D454D; // "HFTSHMEM" in hex

/// Header structure at the beginning of shared memory
#[repr(C, align(64))]
pub struct SharedMemoryHeader {
    /// Magic number for validation
    pub magic: AtomicU64,
    /// Version of the shared memory format
    pub version: AtomicU64,
    /// Total size of the shared memory region
    pub total_size: AtomicU64,
    /// Offset to the first data block
    pub data_offset: AtomicU64,
    /// Number of active data blocks
    pub block_count: AtomicU64,
    /// Write position (for circular buffer)
    pub write_pos: AtomicU64,
    /// Read position (for circular buffer)
    pub read_pos: AtomicU64,
    /// Flag indicating new data available
    pub data_ready: AtomicBool,
    /// Flag indicating consumer has acknowledged
    pub data_acknowledged: AtomicBool,
    /// Timestamp of last write (nanoseconds since epoch)
    pub last_write_ts: AtomicU64,
    /// Timestamp of last read (nanoseconds since epoch)
    pub last_read_ts: AtomicU64,
    /// Padding to reach cache line boundary
    _padding: [u8; CACHE_LINE_SIZE - 10 * 8 - 2],
}

impl Default for SharedMemoryHeader {
    fn default() -> Self {
        Self {
            magic: AtomicU64::new(SHMEM_MAGIC),
            version: AtomicU64::new(1),
            total_size: AtomicU64::new(SHARED_MEMORY_SIZE as u64),
            data_offset: AtomicU64::new(CACHE_LINE_SIZE as u64),
            block_count: AtomicU64::new(0),
            write_pos: AtomicU64::new(0),
            read_pos: AtomicU64::new(0),
            data_ready: AtomicBool::new(false),
            data_acknowledged: AtomicBool::new(true),
            last_write_ts: AtomicU64::new(0),
            last_read_ts: AtomicU64::new(0),
            _padding: [0u8; CACHE_LINE_SIZE - 10 * 8 - 2],
        }
    }
}

/// Feature vector data block header
#[repr(C, align(64))]
pub struct FeatureBlockHeader {
    /// Block ID
    pub id: AtomicU64,
    /// Feature count in this block
    pub feature_count: AtomicU64,
    /// Data size in bytes
    pub data_size: AtomicU64,
    /// Timestamp (nanoseconds)
    pub timestamp_ns: AtomicU64,
    /// Symbol identifier (first 8 chars as u64)
    pub symbol_id: AtomicU64,
    /// Feature type flags
    pub feature_flags: AtomicU64,
    /// Valid flag
    pub valid: AtomicBool,
    /// Padding
    _padding: [u8; CACHE_LINE_SIZE - 7 * 8 - 1],
}

impl Default for FeatureBlockHeader {
    fn default() -> Self {
        Self {
            id: AtomicU64::new(0),
            feature_count: AtomicU64::new(0),
            data_size: AtomicU64::new(0),
            timestamp_ns: AtomicU64::new(0),
            symbol_id: AtomicU64::new(0),
            feature_flags: AtomicU64::new(0),
            valid: AtomicBool::new(false),
            _padding: [0u8; CACHE_LINE_SIZE - 7 * 8 - 1],
        }
    }
}

/// Maximum features per block
pub const MAX_FEATURES_PER_BLOCK: usize = 1024;

/// Shared memory manager for zero-copy IPC
pub struct SharedMemoryManager {
    shmem: Shmem,
    header_ptr: *mut SharedMemoryHeader,
    data_ptr: *mut u8,
    is_owner: bool,
    name: String,
}

unsafe impl Send for SharedMemoryManager {}
unsafe impl Sync for SharedMemoryManager {}

impl SharedMemoryManager {
    /// Create a new shared memory segment (owner mode)
    pub fn create(name: &str) -> io::Result<Self> {
        let shmem = ShmemConf::new()
            .size(SHARED_MEMORY_SIZE)
            .os_create(name)
            .create()?;

        let addr = shmem.as_ptr();
        
        // Initialize header
        let header_ptr = addr as *mut SharedMemoryHeader;
        unsafe {
            ptr::write(header_ptr, SharedMemoryHeader::default());
        }

        let data_ptr = unsafe { addr.add(CACHE_LINE_SIZE) };

        Ok(Self {
            shmem,
            header_ptr,
            data_ptr,
            is_owner: true,
            name: name.to_string(),
        })
    }

    /// Open an existing shared memory segment (consumer mode)
    pub fn open(name: &str) -> io::Result<Self> {
        let shmem = ShmemConf::new()
            .os_open(name)
            .open()?;

        let addr = shmem.as_ptr();
        let header_ptr = addr as *mut SharedMemoryHeader;
        let data_ptr = unsafe { addr.add(CACHE_LINE_SIZE) };

        // Validate magic number
        let header = unsafe { &*header_ptr };
        if header.magic.load(Ordering::Relaxed) != SHMEM_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid shared memory magic number",
            ));
        }

        Ok(Self {
            shmem,
            header_ptr,
            data_ptr,
            is_owner: false,
            name: name.to_string(),
        })
    }

    /// Get reference to header
    #[inline]
    pub fn header(&self) -> &SharedMemoryHeader {
        unsafe { &*self.header_ptr }
    }

    /// Get mutable reference to header
    #[inline]
    pub fn header_mut(&mut self) -> &mut SharedMemoryHeader {
        unsafe { &mut *self.header_ptr }
    }

    /// Write feature data to shared memory
    pub fn write_features(
        &self,
        symbol_id: u64,
        features: &[f32],
        timestamp_ns: u64,
        feature_flags: u64,
    ) -> io::Result<u64> {
        if features.is_empty() || features.len() > MAX_FEATURES_PER_BLOCK {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Feature count out of range",
            ));
        }

        let header = unsafe { &*self.header_ptr };
        
        // Wait for acknowledgment if previous data not consumed
        while !header.data_acknowledged.load(Ordering::Acquire) {
            std::hint::spin_loop();
        }

        // Calculate write position
        let write_pos = header.write_pos.load(Ordering::Relaxed);
        let block_size = CACHE_LINE_SIZE + features.len() * std::mem::size_of::<f32>();
        let data_start = (write_pos * block_size as u64) as usize % (SHARED_MEMORY_SIZE - CACHE_LINE_SIZE);

        // Write block header
        let block_header_ptr = unsafe { self.data_ptr.add(data_start) } as *mut FeatureBlockHeader;
        unsafe {
            let block_header = FeatureBlockHeader {
                id: AtomicU64::new(write_pos),
                feature_count: AtomicU64::new(features.len() as u64),
                data_size: AtomicU64::new((features.len() * std::mem::size_of::<f32>()) as u64),
                timestamp_ns: AtomicU64::new(timestamp_ns),
                symbol_id: AtomicU64::new(symbol_id),
                feature_flags: AtomicU64::new(feature_flags),
                valid: AtomicBool::new(true),
                _padding: [0u8; CACHE_LINE_SIZE - 7 * 8 - 1],
            };
            ptr::write(block_header_ptr, block_header);
        }

        // Write feature data
        let feature_data_ptr = unsafe { self.data_ptr.add(data_start + CACHE_LINE_SIZE) };
        unsafe {
            ptr::copy_nonoverlapping(
                features.as_ptr() as *const u8,
                feature_data_ptr,
                features.len() * std::mem::size_of::<f32>(),
            );
        }

        // Update header
        header.block_count.fetch_add(1, Ordering::Release);
        header.write_pos.fetch_add(1, Ordering::Release);
        header.last_write_ts.store(timestamp_ns, Ordering::Release);
        header.data_acknowledged.store(false, Ordering::Release);
        header.data_ready.store(true, Ordering::Release);

        Ok(write_pos)
    }

    /// Read feature data from shared memory
    pub fn read_features(&self, buffer: &mut [f32]) -> io::Result<(u64, u64, u64)> {
        let header = unsafe { &*self.header_ptr };

        if !header.data_ready.load(Ordering::Acquire) {
            return Err(io::Error::new(
                io::ErrorKind::WouldBlock,
                "No data available",
            ));
        }

        let read_pos = header.read_pos.load(Ordering::Relaxed);
        let block_size = CACHE_LINE_SIZE + buffer.len() * std::mem::size_of::<f32>();
        let data_start = (read_pos * block_size as u64) as usize % (SHARED_MEMORY_SIZE - CACHE_LINE_SIZE);

        // Read block header
        let block_header_ptr = unsafe { self.data_ptr.add(data_start) } as *const FeatureBlockHeader;
        let block_header = unsafe { &*block_header_ptr };

        if !block_header.valid.load(Ordering::Acquire) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid block header",
            ));
        }

        let feature_count = block_header.feature_count.load(Ordering::Acquire) as usize;
        if feature_count > buffer.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Buffer too small",
            ));
        }

        // Read feature data
        let feature_data_ptr = unsafe { self.data_ptr.add(data_start + CACHE_LINE_SIZE) };
        unsafe {
            ptr::copy_nonoverlapping(
                feature_data_ptr,
                buffer.as_mut_ptr() as *mut u8,
                feature_count * std::mem::size_of::<f32>(),
            );
        }

        // Update header
        header.read_pos.fetch_add(1, Ordering::Release);
        header.last_read_ts.store(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
            Ordering::Release,
        );
        header.data_ready.store(false, Ordering::Release);
        header.data_acknowledged.store(true, Ordering::Release);

        Ok((
            block_header.symbol_id.load(Ordering::Relaxed),
            block_header.timestamp_ns.load(Ordering::Relaxed),
            feature_count as u64,
        ))
    }

    /// Check if new data is available
    #[inline]
    pub fn has_data(&self) -> bool {
        self.header().data_ready.load(Ordering::Acquire)
    }

    /// Get the shared memory name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Check if this instance owns the shared memory
    pub fn is_owner(&self) -> bool {
        self.is_owner
    }

    /// Get current latency in nanoseconds (write to read)
    pub fn get_latency_ns(&self) -> u64 {
        let header = self.header();
        let write_ts = header.last_write_ts.load(Ordering::Relaxed);
        let read_ts = header.last_read_ts.load(Ordering::Relaxed);
        
        if write_ts > 0 && read_ts > 0 {
            write_ts.saturating_sub(read_ts)
        } else {
            0
        }
    }
}

impl Drop for SharedMemoryManager {
    fn drop(&mut self) {
        if self.is_owner {
            // Clean up shared memory on Unix systems
            #[cfg(unix)]
            {
                let _ = fs::remove_file(format!("/dev/shm/{}", self.name));
            }
        }
    }
}

/// Feature flags for different feature types
pub mod feature_flags {
    pub const TECHNICAL: u64 = 1 << 0;
    pub const ORDER_FLOW: u64 = 1 << 1;
    pub const ON_CHAIN: u64 = 1 << 2;
    pub const SENTIMENT: u64 = 1 << 3;
    pub const VOLATILITY: u64 = 1 << 4;
    pub const MOMENTUM: u64 = 1 << 5;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_cache_alignment() {
        let header = SharedMemoryHeader::default();
        let addr = &header as *const _ as usize;
        assert_eq!(addr % CACHE_LINE_SIZE, 0, "Header should be cache-line aligned");
    }

    #[test]
    fn test_feature_block_header_alignment() {
        let block = FeatureBlockHeader::default();
        let addr = &block as *const _ as usize;
        assert_eq!(addr % CACHE_LINE_SIZE, 0, "Block header should be cache-line aligned");
    }

    #[test]
    fn test_shared_memory_creation() {
        let name = "hft_test_shmem";
        let result = SharedMemoryManager::create(name);
        
        // May fail on systems without shared memory support
        if result.is_ok() {
            let sm = result.unwrap();
            assert!(sm.is_owner());
            assert_eq!(sm.name(), name);
            assert_eq!(sm.header().magic.load(Ordering::Relaxed), SHMEM_MAGIC);
        }
    }
}
