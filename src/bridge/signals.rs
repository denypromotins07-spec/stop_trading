//! Inbound Alpha Signals and Weight Updates from Python/Nautilus Backend
//!
//! This module defines the memory structures for receiving ML-generated
//! trading signals and model weight updates from the Python backend.
//!
//! All structs use #[repr(C)] with explicit padding for cross-language FFI.

use std::mem;
use std::sync::atomic::{AtomicU64, AtomicU8, AtomicI32, Ordering};
use std::time::{Duration, Instant};

/// Signal type enumeration
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SignalType {
    Alpha = 0,          // Directional alpha signal (-1.0 to 1.0)
    Volatility = 1,     // Volatility estimate
    Spread = 2,         // Optimal spread suggestion
    Inventory = 3,      // Target inventory adjustment
    Risk = 4,           // Risk regime indicator
    Custom = 255,       // Custom signal type
}

impl From<u8> for SignalType {
    fn from(value: u8) -> Self {
        match value {
            0 => SignalType::Alpha,
            1 => SignalType::Volatility,
            2 => SignalType::Spread,
            3 => SignalType::Inventory,
            4 => SignalType::Risk,
            _ => SignalType::Custom,
        }
    }
}

/// Individual alpha signal element
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct AlphaSignal {
    pub signal_type: u8,      // SignalType as u8
    pub confidence: u8,       // Confidence level (0-100)
    pub _padding: [u8; 2],    // Explicit padding for 4-byte alignment
    pub symbol: u64,          // Symbol identifier
    pub value: i64,           // Fixed-point signal value (scaled by 1e9)
    pub timestamp_ns: u64,    // Signal generation timestamp
    pub expiry_ns: u64,       // Signal expiry timestamp
    pub model_id: u32,        // Originating model identifier
    pub sequence: u64,        // Sequence number for ordering
}

impl AlphaSignal {
    /// Create a new alpha signal
    #[inline]
    pub fn new(
        signal_type: SignalType,
        symbol: u64,
        value: f64,
        confidence: u8,
        model_id: u32,
        timestamp_ns: u64,
        ttl_ms: u64,
    ) -> Self {
        Self {
            signal_type: signal_type as u8,
            confidence,
            _padding: [0; 2],
            symbol,
            value: (value * 1_000_000_000.0) as i64,  // Scale to fixed-point
            timestamp_ns,
            expiry_ns: timestamp_ns + (ttl_ms * 1_000_000),
            model_id,
            sequence: 0,
        }
    }

    /// Get signal value as f64
    #[inline]
    pub fn value_f64(&self) -> f64 {
        self.value as f64 / 1_000_000_000.0
    }

    /// Check if signal has expired
    #[inline]
    pub fn is_expired(&self, current_ns: u64) -> bool {
        current_ns > self.expiry_ns
    }

    /// Get signal type enum
    #[inline]
    pub fn get_type(&self) -> SignalType {
        SignalType::from(self.signal_type)
    }
}

/// Signal batch header for bulk transmission
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SignalBatchHeader {
    pub magic: u64,           // SHM_MAGIC
    pub version: u32,         // Schema version
    pub batch_id: u64,        // Unique batch identifier
    pub timestamp_ns: u64,    // Batch creation timestamp
    pub signal_count: u32,    // Number of signals in batch
    pub model_version: u32,   // Model version that generated signals
    pub checksum: u32,        // CRC32 checksum
    pub flags: u32,           // Status flags (bit 0: stale, bit 1: urgent)
    pub latency_us: u32,      // Python->Rust latency in microseconds
    pub _reserved: [u64; 3],  // Reserved for future expansion
}

/// Signal batch segment (fixed capacity)
#[repr(C)]
pub struct SignalBatchSegment {
    pub header: SignalBatchHeader,
    pub signals: [AlphaSignal; 64],  // Fixed capacity: 64 signals per batch
    pub _padding: [u8; 64],          // Cache line padding
}

impl SignalBatchSegment {
    pub const SIZE: usize = mem::size_of::<Self>();
    pub const MAX_SIGNALS: usize = 64;

    pub const fn new() -> Self {
        use super::schema::SHM_MAGIC;
        use super::schema::SCHEMA_VERSION;
        
        Self {
            header: SignalBatchHeader {
                magic: SHM_MAGIC,
                version: SCHEMA_VERSION,
                batch_id: 0,
                timestamp_ns: 0,
                signal_count: 0,
                model_version: 0,
                checksum: 0,
                flags: 0,
                latency_us: 0,
                _reserved: [0; 3],
            },
            signals: [AlphaSignal {
                signal_type: 0,
                confidence: 0,
                _padding: [0; 2],
                symbol: 0,
                value: 0,
                timestamp_ns: 0,
                expiry_ns: 0,
                model_id: 0,
                sequence: 0,
            }; 64],
            _padding: [0; 64],
        }
    }

    #[inline]
    pub fn is_valid(&self) -> bool {
        use super::schema::SHM_MAGIC;
        use super::schema::SCHEMA_VERSION;
        
        self.header.magic == SHM_MAGIC 
            && self.header.version == SCHEMA_VERSION
            && self.header.signal_count <= Self::MAX_SIGNALS as u32
    }

    #[inline]
    pub fn add_signal(&mut self, signal: AlphaSignal) -> bool {
        if self.header.signal_count < Self::MAX_SIGNALS as u32 {
            let idx = self.header.signal_count as usize;
            self.signals[idx] = signal;
            self.header.signal_count += 1;
            true
        } else {
            false
        }
    }

    #[inline]
    pub fn get_signals(&self) -> &[AlphaSignal] {
        &self.signals[..self.header.signal_count as usize]
    }
}

impl Default for SignalBatchSegment {
    fn default() -> Self {
        Self::new()
    }
}

/// Model weight update structure for online learning
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct WeightUpdate {
    pub model_id: u32,
    pub layer_id: u32,
    param_index: u32,
    data_type: u8,      // 0=f32, 1=f64
    _padding: [u8; 3],
    timestamp_ns: u64,
    value_f64_bits: u64,  // Weight value as f64 bits
    gradient_f64_bits: u64,  // Gradient value as f64 bits
    learning_rate_bits: u64, // Current learning rate as f64 bits
}

impl WeightUpdate {
    #[inline]
    pub fn value(&self) -> f64 {
        f64::from_bits(self.value_f64_bits)
    }

    #[inline]
    pub fn gradient(&self) -> f64 {
        f64::from_bits(self.gradient_f64_bits)
    }

    #[inline]
    pub fn learning_rate(&self) -> f64 {
        f64::from_bits(self.learning_rate_bits)
    }
}

/// Weight update batch for model synchronization
#[repr(C)]
pub struct WeightUpdateBatch {
    pub header: SignalBatchHeader,
    pub update_count: u32,
    _padding: [u8; 28],  // Align to cache line
    pub updates: [WeightUpdate; 32],  // Fixed capacity
}

impl WeightUpdateBatch {
    pub const SIZE: usize = mem::size_of::<Self>();
    
    pub const fn new() -> Self {
        Self {
            header: SignalBatchHeader {
                magic: super::schema::SHM_MAGIC,
                version: super::schema::SCHEMA_VERSION,
                batch_id: 0,
                timestamp_ns: 0,
                signal_count: 0,
                model_version: 0,
                checksum: 0,
                flags: 0,
                latency_us: 0,
                _reserved: [0; 3],
            },
            update_count: 0,
            _padding: [0; 28],
            updates: [WeightUpdate {
                model_id: 0,
                layer_id: 0,
                param_index: 0,
                data_type: 0,
                _padding: [0; 3],
                timestamp_ns: 0,
                value_f64_bits: 0,
                gradient_f64_bits: 0,
                learning_rate_bits: 0,
            }; 32],
        }
    }
}

impl Default for WeightUpdateBatch {
    fn default() -> Self {
        Self::new()
    }
}

/// Lock-free signal queue state for IPC coordination
#[repr(C)]
#[derive(Debug)]
pub struct SignalQueueState {
    pub write_seq: AtomicU64,
    pub read_seq: AtomicU64,
    pub processed_seq: AtomicU64,
    pub pending_count: AtomicU32,
    pub dropped_count: AtomicU64,
    pub last_signal_ns: AtomicU64,
    pub _padding: [u8; 32],  // Cache line alignment
}

impl SignalQueueState {
    pub const fn new() -> Self {
        Self {
            write_seq: AtomicU64::new(0),
            read_seq: AtomicU64::new(0),
            processed_seq: AtomicU64::new(0),
            pending_count: AtomicU32::new(0),
            dropped_count: AtomicU64::new(0),
            last_signal_ns: AtomicU64::new(0),
            _padding: [0; 32],
        }
    }

    #[inline]
    pub fn enqueue(&self) -> u64 {
        let seq = self.write_seq.fetch_add(1, Ordering::Relaxed);
        self.pending_count.fetch_add(1, Ordering::Relaxed);
        seq
    }

    #[inline]
    pub fn dequeue(&self) -> Option<u64> {
        let read = self.read_seq.load(Ordering::Relaxed);
        let write = self.write_seq.load(Ordering::Relaxed);
        
        if read < write {
            self.read_seq.fetch_add(1, Ordering::Relaxed);
            self.pending_count.fetch_sub(1, Ordering::Relaxed);
            Some(read)
        } else {
            None
        }
    }

    #[inline]
    pub fn mark_processed(&self, seq: u64) {
        self.processed_seq.store(seq, Ordering::Release);
    }
}

impl Default for SignalQueueState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alpha_signal_size() {
        assert_eq!(mem::size_of::<AlphaSignal>(), 48);
        assert_eq!(mem::align_of::<AlphaSignal>(), 8);
    }

    #[test]
    fn test_signal_batch_segment_size() {
        // Header: 96 bytes
        // Signals: 64 * 48 = 3072 bytes
        // Padding: 64 bytes
        // Total: 3232 bytes
        assert_eq!(SignalBatchSegment::SIZE, 3232);
    }

    #[test]
    fn test_signal_value_conversion() {
        let signal = AlphaSignal::new(
            SignalType::Alpha,
            0x424E42555344,
            0.75,
            85,
            1,
            1234567890,
            1000,
        );
        
        assert!((signal.value_f64() - 0.75).abs() < 1e-9);
    }
}
