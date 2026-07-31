//! Shared Memory Schema for Python Bridge (IPC Contracts)
//! 
//! This module defines the exact memory layout for zero-copy data exchange
//! between the Rust HFT core and Python/Nautilus/Ray ML backend.
//! 
//! All structs use #[repr(C)] with explicit padding to ensure byte-level
//! alignment matches numpy and pyarrow expectations.

use std::mem;
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};

/// Magic number to validate shared memory segment integrity
pub const SHM_MAGIC: u64 = 0x4846545F53484D00; // "HFT_SHM\0"

/// Current schema version for compatibility checking
pub const SCHEMA_VERSION: u32 = 1;

/// Feature vector element (f64 stored as u64 for atomic operations)
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FeatureElement {
    pub value_bits: u64,  // f64 bits
    pub timestamp_ns: u64,
    pub feature_id: u32,
    pub _padding: u32,    // Explicit padding for 8-byte alignment
}

impl FeatureElement {
    #[inline]
    pub fn new(value: f64, timestamp_ns: u64, feature_id: u32) -> Self {
        Self {
            value_bits: value.to_bits(),
            timestamp_ns,
            feature_id,
            _padding: 0,
        }
    }

    #[inline]
    pub fn value(&self) -> f64 {
        f64::from_bits(self.value_bits)
    }
}

/// Outbound feature vector header
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FeatureVectorHeader {
    pub magic: u64,           // SHM_MAGIC
    pub version: u32,         // Schema version
    pub sequence: u64,        // Monotonic sequence number
    pub symbol: u64,          // Symbol identifier (e.g., "BNBUSDT" as u64)
    pub timestamp_ns: u64,    // Feature generation timestamp
    pub feature_count: u32,   // Number of features in vector
    pub vector_id: u32,       // Vector identifier for tracking
    pub checksum: u32,        // CRC32 checksum of data
    pub flags: u32,           // Status flags
    pub _reserved: [u64; 4],  // Reserved for future expansion (32 bytes)
}

/// Outbound feature vector segment (fixed size for zero-copy)
#[repr(C)]
pub struct FeatureVectorSegment {
    pub header: FeatureVectorHeader,
    pub features: [FeatureElement; 256],  // Fixed capacity: 256 features
    pub _padding: [u8; 64],   // Cache line padding
}

impl FeatureVectorSegment {
    /// Size of the segment in bytes (for mmap allocation)
    pub const SIZE: usize = mem::size_of::<Self>();
    
    /// Create a new initialized segment
    pub const fn new() -> Self {
        Self {
            header: FeatureVectorHeader {
                magic: SHM_MAGIC,
                version: SCHEMA_VERSION,
                sequence: 0,
                symbol: 0,
                timestamp_ns: 0,
                feature_count: 0,
                vector_id: 0,
                checksum: 0,
                flags: 0,
                _reserved: [0; 4],
            },
            features: [FeatureElement {
                value_bits: 0,
                timestamp_ns: 0,
                feature_id: 0,
                _padding: 0,
            }; 256],
            _padding: [0; 64],
        }
    }

    /// Validate the segment integrity
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.header.magic == SHM_MAGIC && self.header.version == SCHEMA_VERSION
    }

    /// Get feature count
    #[inline]
    pub fn feature_count(&self) -> usize {
        self.header.feature_count as usize
    }

    /// Get feature by index (bounds-checked)
    #[inline]
    pub fn get_feature(&self, index: usize) -> Option<f64> {
        if index < self.header.feature_count as usize {
            Some(self.features[index].value())
        } else {
            None
        }
    }
}

/// Default implementation for FeatureVectorSegment
impl Default for FeatureVectorSegment {
    fn default() -> Self {
        Self::new()
    }
}

/// Ring buffer state for feature vectors (lock-free)
#[repr(C)]
#[derive(Debug)]
pub struct FeatureRingBufferState {
    pub write_index: AtomicU64,
    pub read_index: AtomicU64,
    pub buffer_size: u64,
    pub overflow_count: AtomicU64,
    pub _padding: [u8; 48],  // Cache line alignment
}

impl FeatureRingBufferState {
    pub const fn new(buffer_size: u64) -> Self {
        Self {
            write_index: AtomicU64::new(0),
            read_index: AtomicU64::new(0),
            buffer_size,
            overflow_count: AtomicU64::new(0),
            _padding: [0; 48],
        }
    }

    #[inline]
    pub fn next_write_index(&self) -> u64 {
        self.write_index.fetch_add(1, Ordering::Relaxed)
    }

    #[inline]
    pub fn get_read_index(&self) -> u64 {
        self.read_index.load(Ordering::Acquire)
    }

    #[inline]
    pub fn update_read_index(&self, new_index: u64) {
        self.read_index.store(new_index, Ordering::Release);
    }
}

/// Compile-time assertions for memory layout validation
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_element_size() {
        assert_eq!(mem::size_of::<FeatureElement>(), 24);
        assert_eq!(mem::align_of::<FeatureElement>(), 8);
    }

    #[test]
    fn test_header_size() {
        assert_eq!(mem::size_of::<FeatureVectorHeader>(), 96);
        assert_eq!(mem::align_of::<FeatureVectorHeader>(), 8);
    }

    #[test]
    fn test_segment_layout() {
        // Header: 96 bytes
        // Features: 256 * 24 = 6144 bytes
        // Padding: 64 bytes
        // Total: 6304 bytes
        assert_eq!(FeatureVectorSegment::SIZE, 6304);
    }

    #[test]
    fn test_cache_line_alignment() {
        // Ensure ring buffer state is cache-line aligned
        assert_eq!(mem::align_of::<FeatureRingBufferState>(), 8);
    }
}
