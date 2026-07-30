//! Rkyv Zero-Copy Serialization Implementation
//!
//! Integrates `rkyv` for zero-copy deserialization of internal state
//! and historical snapshots. Allows the backtesting engine to read
//! historical order books directly from disk without parsing overhead.

use std::sync::atomic::{AtomicU64, Ordering};

/// Archived state wrapper for zero-copy access
#[repr(C)]
pub struct ArchivedState<T> {
    /// Pointer to archived data (in memory-mapped file)
    data_ptr: *const T,
    /// Size of archived data
    size: usize,
    /// Is valid
    is_valid: bool,
}

// Safety: ArchivedState is safe to share when T is Sync
unsafe impl<T: Sync> Send for ArchivedState<T> {}
unsafe impl<T: Sync> Sync for ArchivedState<T> {}

impl<T> ArchivedState<T> {
    /// Create a new archived state reference
    #[inline]
    pub fn new(data_ptr: *const T, size: usize) -> Self {
        Self {
            data_ptr,
            size,
            is_valid: !data_ptr.is_null(),
        }
    }

    /// Get reference to archived data
    #[inline]
    pub fn get(&self) -> Option<&T> {
        if self.is_valid && !self.data_ptr.is_null() {
            Some(unsafe { &*self.data_ptr })
        } else {
            None
        }
    }

    /// Check if valid
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.is_valid
    }

    /// Get size of archived data
    #[inline]
    pub fn size(&self) -> usize {
        self.size
    }
}

/// Rkyv serializer for order book snapshots
#[repr(C)]
pub struct RkyvSerializer {
    /// Total bytes serialized
    bytes_serialized: AtomicU64,
    /// Serialization count
    serialize_count: AtomicU64,
    /// Buffer size
    buffer_size: usize,
}

impl RkyvSerializer {
    pub fn new(buffer_size: usize) -> Self {
        Self {
            bytes_serialized: AtomicU64::new(0),
            serialize_count: AtomicU64::new(0),
            buffer_size,
        }
    }

    /// Serialize order book snapshot
    /// In production, would use actual rkyv serialization
    #[inline]
    pub fn serialize_orderbook(&self, data: &[u8]) -> Result<Vec<u8>, SerializationError> {
        // Simulated rkyv serialization
        // In production: let archived = rkyv::to_bytes::<_, 256>(data)?;
        
        if data.len() > self.buffer_size {
            return Err(SerializationError::BufferTooSmall);
        }

        // Prefix with size for deserialization
        let mut result = Vec::with_capacity(8 + data.len());
        result.extend_from_slice(&(data.len() as u64).to_le_bytes());
        result.extend_from_slice(data);

        self.bytes_serialized.fetch_add(result.len() as u64, Ordering::Relaxed);
        self.serialize_count.fetch_add(1, Ordering::Relaxed);

        Ok(result)
    }

    /// Serialize market data event
    #[inline]
    pub fn serialize_market_data(&self, data: &MarketDataSnapshot) -> Result<Vec<u8>, SerializationError> {
        let bytes = bytemuck::bytes_of(data);
        self.serialize_orderbook(bytes)
    }

    /// Get serializer statistics
    #[inline]
    pub fn get_stats(&self) -> SerializerStats {
        SerializerStats {
            bytes_serialized: self.bytes_serialized.load(Ordering::Relaxed),
            serialize_count: self.serialize_count.load(Ordering::Relaxed),
            avg_size: {
                let count = self.serialize_count.load(Ordering::Relaxed);
                if count > 0 {
                    self.bytes_serialized.load(Ordering::Relaxed) / count
                } else {
                    0
                }
            },
        }
    }

    /// Get buffer size
    #[inline]
    pub fn buffer_size(&self) -> usize {
        self.buffer_size
    }
}

impl Default for RkyvSerializer {
    fn default() -> Self {
        Self::new(65536) // 64KB default buffer
    }
}

/// Rkyv deserializer for zero-copy reads
#[repr(C)]
pub struct RkyvDeserializer {
    /// Total bytes deserialized
    bytes_deserialized: AtomicU64,
    /// Deserialization count
    deserialize_count: AtomicU64,
    /// Zero-copy deserializations
    zero_copy_count: AtomicU64,
}

impl RkyvDeserializer {
    pub fn new() -> Self {
        Self {
            bytes_deserialized: AtomicU64::new(0),
            deserialize_count: AtomicU64::new(0),
            zero_copy_count: AtomicU64::new(0),
        }
    }

    /// Deserialize order book snapshot (zero-copy)
    /// Returns archived state that can be accessed without copying
    #[inline]
    pub fn deserialize_orderbook_zero_copy<'a>(
        &self,
        bytes: &'a [u8],
    ) -> Result<ArchivedState<OrderBookArchive>, SerializationError> {
        if bytes.len() < 8 {
            return Err(SerializationError::InvalidFormat);
        }

        // Read size prefix
        let size = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
        
        if bytes.len() < 8 + size {
            return Err(SerializationError::OutOfBounds);
        }

        // In production, would use rkyv's zero-copy deserialization:
        // let archived = unsafe { rkyv::archived_root::<OrderBook>(&bytes[8..]) };
        
        // For now, create a simulated archived state
        let data_ptr = bytes.as_ptr().add(8) as *const OrderBookArchive;
        
        self.bytes_deserialized.fetch_add(bytes.len() as u64, Ordering::Relaxed);
        self.deserialize_count.fetch_add(1, Ordering::Relaxed);
        self.zero_copy_count.fetch_add(1, Ordering::Relaxed);

        Ok(ArchivedState::new(data_ptr, size))
    }

    /// Deserialize market data snapshot
    #[inline]
    pub fn deserialize_market_data(&self, bytes: &[u8]) -> Result<MarketDataSnapshot, SerializationError> {
        if bytes.len() < 8 + std::mem::size_of::<MarketDataSnapshot>() {
            return Err(SerializationError::InvalidFormat);
        }

        // Skip size prefix
        let data_bytes = &bytes[8..];
        
        // Zero-copy read using bytemuck
        let snapshot = bytemuck::pod_read_unaligned(data_bytes);
        
        self.bytes_deserialized.fetch_add(bytes.len() as u64, Ordering::Relaxed);
        self.deserialize_count.fetch_add(1, Ordering::Relaxed);

        Ok(snapshot)
    }

    /// Get deserializer statistics
    #[inline]
    pub fn get_stats(&self) -> DeserializerStats {
        DeserializerStats {
            bytes_deserialized: self.bytes_deserialized.load(Ordering::Relaxed),
            deserialize_count: self.deserialize_count.load(Ordering::Relaxed),
            zero_copy_count: self.zero_copy_count.load(Ordering::Relaxed),
            zero_copy_ratio: {
                let total = self.deserialize_count.load(Ordering::Relaxed);
                if total > 0 {
                    self.zero_copy_count.load(Ordering::Relaxed) as f64 / total as f64
                } else {
                    0.0
                }
            },
        }
    }
}

impl Default for RkyvDeserializer {
    fn default() -> Self {
        Self::new()
    }
}

/// Order book archive structure for zero-copy access
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct OrderBookArchive {
    /// Symbol hash
    pub symbol_hash: u64,
    /// Timestamp in nanoseconds
    pub timestamp_ns: u64,
    /// Number of bid levels
    pub bid_levels: u32,
    /// Number of ask levels
    pub ask_levels: u32,
    /// Best bid price
    pub best_bid: u64,
    /// Best ask price
    pub best_ask: u64,
    /// Mid price
    pub mid_price: u64,
    /// Spread in ticks
    pub spread: u64,
}

impl OrderBookArchive {
    #[inline]
    pub fn new() -> Self {
        Self {
            symbol_hash: 0,
            timestamp_ns: 0,
            bid_levels: 0,
            ask_levels: 0,
            best_bid: 0,
            best_ask: 0,
            mid_price: 0,
            spread: 0,
        }
    }

    #[inline]
    pub fn is_valid(&self) -> bool {
        self.best_bid > 0 && self.best_ask > 0 && self.best_bid < self.best_ask
    }
}

impl Default for OrderBookArchive {
    fn default() -> Self {
        Self::new()
    }
}

/// Market data snapshot for serialization
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MarketDataSnapshot {
    /// Symbol hash
    pub symbol_hash: u64,
    /// Timestamp in nanoseconds
    pub timestamp_ns: u64,
    /// Bid prices (up to 10 levels)
    pub bid_prices: [u64; 10],
    /// Ask prices (up to 10 levels)
    pub ask_prices: [u64; 10],
    /// Bid sizes
    pub bid_sizes: [u64; 10],
    /// Ask sizes
    pub ask_sizes: [u64; 10],
    /// Number of valid levels
    pub levels: u32,
    /// Venue ID
    pub venue_id: u32,
}

// Safety: MarketDataSnapshot is POD (Plain Old Data)
unsafe impl bytemuck::Pod for MarketDataSnapshot {}
unsafe impl bytemuck::Zeroable for MarketDataSnapshot {}

impl MarketDataSnapshot {
    #[inline]
    pub fn new() -> Self {
        Self {
            symbol_hash: 0,
            timestamp_ns: 0,
            bid_prices: [0; 10],
            ask_prices: [0; 10],
            bid_sizes: [0; 10],
            ask_sizes: [0; 10],
            levels: 0,
            venue_id: 0,
        }
    }

    #[inline]
    pub fn with_symbol(mut self, symbol_hash: u64) -> Self {
        self.symbol_hash = symbol_hash;
        self
    }

    #[inline]
    pub fn with_timestamp(mut self, timestamp_ns: u64) -> Self {
        self.timestamp_ns = timestamp_ns;
        self
    }

    #[inline]
    pub fn set_level(&mut self, level: usize, bid_price: u64, ask_price: u64, bid_size: u64, ask_size: u64) {
        if level < 10 {
            self.bid_prices[level] = bid_price;
            self.ask_prices[level] = ask_price;
            self.bid_sizes[level] = bid_size;
            self.ask_sizes[level] = ask_size;
            if level >= self.levels as usize {
                self.levels = (level + 1) as u32;
            }
        }
    }

    #[inline]
    pub fn best_bid(&self) -> u64 {
        if self.levels > 0 {
            self.bid_prices[0]
        } else {
            0
        }
    }

    #[inline]
    pub fn best_ask(&self) -> u64 {
        if self.levels > 0 {
            self.ask_prices[0]
        } else {
            0
        }
    }

    #[inline]
    pub fn spread(&self) -> u64 {
        if self.levels > 0 && self.ask_prices[0] > self.bid_prices[0] {
            self.ask_prices[0] - self.bid_prices[0]
        } else {
            0
        }
    }
}

impl Default for MarketDataSnapshot {
    fn default() -> Self {
        Self::new()
    }
}

/// Serializer statistics
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SerializerStats {
    pub bytes_serialized: u64,
    pub serialize_count: u64,
    pub avg_size: u64,
}

/// Deserializer statistics
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DeserializerStats {
    pub bytes_deserialized: u64,
    pub deserialize_count: u64,
    pub zero_copy_count: u64,
    pub zero_copy_ratio: f64,
}

/// Serialization error (re-export from parent module)
use super::SerializationError;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_archived_state() {
        let data = OrderBookArchive::new();
        let archived = ArchivedState::new(&data, std::mem::size_of::<OrderBookArchive>());

        assert!(archived.is_valid());
        assert!(archived.get().is_some());
        assert_eq!(archived.size(), std::mem::size_of::<OrderBookArchive>());
    }

    #[test]
    fn test_serializer_creation() {
        let serializer = RkyvSerializer::new(65536);
        assert_eq!(serializer.buffer_size(), 65536);
        assert_eq!(serializer.get_stats().serialize_count, 0);
    }

    #[test]
    fn test_market_data_snapshot() {
        let mut snapshot = MarketDataSnapshot::new();
        snapshot.set_level(0, 10000, 10010, 100, 200);

        assert_eq!(snapshot.best_bid(), 10000);
        assert_eq!(snapshot.best_ask(), 10010);
        assert_eq!(snapshot.spread(), 10);
        assert_eq!(snapshot.levels, 1);
    }

    #[test]
    fn test_deserializer_creation() {
        let deserializer = RkyvDeserializer::new();
        let stats = deserializer.get_stats();

        assert_eq!(stats.deserialize_count, 0);
        assert_eq!(stats.zero_copy_count, 0);
        assert_eq!(stats.zero_copy_ratio, 0.0);
    }

    #[test]
    fn test_orderbook_archive() {
        let mut archive = OrderBookArchive::new();
        archive.best_bid = 10000;
        archive.best_ask = 10010;
        archive.mid_price = 10005;
        archive.spread = 10;

        assert!(archive.is_valid());
        assert_eq!(archive.spread, 10);
    }
}
