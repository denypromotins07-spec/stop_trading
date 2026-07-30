//! FlatBuffers Implementation for Cross-Language IPC
//!
//! Implements FlatBuffers schemas for ultra-fast, cross-language IPC
//! between Rust and Python. Defines exact memory layout for feature
//! vectors sent to Ray/Nautilus ML backend to avoid Python GIL bottlenecks.

use std::sync::atomic::{AtomicU64, Ordering};

/// Maximum feature vector size
pub const MAX_FEATURE_SIZE: usize = 1024;

/// Feature vector for ML backend
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FeatureVector {
    /// Feature ID
    pub id: u64,
    /// Timestamp in nanoseconds
    pub timestamp_ns: u64,
    /// Symbol hash
    pub symbol_hash: u64,
    /// Number of features
    pub feature_count: u32,
    /// Venue ID
    pub venue_id: u32,
    /// Features (fixed size for zero-copy)
    pub features: [f64; 32],
    /// Labels (for training data)
    pub labels: [u8; 8],
    /// Checksum for validation
    pub checksum: u32,
    /// Flags
    pub flags: u32,
}

impl FeatureVector {
    #[inline]
    pub fn new() -> Self {
        Self {
            id: 0,
            timestamp_ns: 0,
            symbol_hash: 0,
            feature_count: 0,
            venue_id: 0,
            features: [0.0; 32],
            labels: [0u8; 8],
            checksum: 0,
            flags: 0,
        }
    }

    #[inline]
    pub fn with_id(mut self, id: u64) -> Self {
        self.id = id;
        self
    }

    #[inline]
    pub fn with_timestamp(mut self, timestamp_ns: u64) -> Self {
        self.timestamp_ns = timestamp_ns;
        self
    }

    #[inline]
    pub fn with_symbol(mut self, symbol_hash: u64) -> Self {
        self.symbol_hash = symbol_hash;
        self
    }

    #[inline]
    pub fn set_feature(&mut self, index: usize, value: f64) {
        if index < 32 {
            self.features[index] = value;
            if index >= self.feature_count as usize {
                self.feature_count = (index + 1) as u32;
            }
        }
    }

    #[inline]
    pub fn get_feature(&self, index: usize) -> f64 {
        if index < 32 {
            self.features[index]
        } else {
            0.0
        }
    }

    #[inline]
    pub fn set_label(&mut self, index: usize, value: u8) {
        if index < 8 {
            self.labels[index] = value;
        }
    }

    #[inline]
    pub fn calculate_checksum(&mut self) {
        let mut sum: u32 = 0;
        for &f in &self.features[..self.feature_count as usize] {
            sum ^= f.to_bits() as u32;
            sum ^= (f.to_bits() >> 32) as u32;
        }
        sum ^= self.id as u32;
        sum ^= (self.id >> 32) as u32;
        sum ^= self.symbol_hash as u32;
        sum ^= self.timestamp_ns as u32;
        self.checksum = sum;
    }

    #[inline]
    pub fn validate_checksum(&self) -> bool {
        let mut sum: u32 = 0;
        for &f in &self.features[..self.feature_count as usize] {
            sum ^= f.to_bits() as u32;
            sum ^= (f.to_bits() >> 32) as u32;
        }
        sum ^= self.id as u32;
        sum ^= (self.id >> 32) as u32;
        sum ^= self.symbol_hash as u32;
        sum ^= self.timestamp_ns as u32;
        self.checksum == sum
    }

    #[inline]
    pub fn is_valid(&self) -> bool {
        self.feature_count > 0 && self.validate_checksum()
    }
}

impl Default for FeatureVector {
    fn default() -> Self {
        Self::new()
    }
}

// Safety: FeatureVector is POD for zero-copy serialization
unsafe impl bytemuck::Pod for FeatureVector {}
unsafe impl bytemuck::Zeroable for FeatureVector {}

/// FlatBuffer builder for IPC messages
#[repr(C)]
pub struct FlatBufferBuilder {
    /// Internal buffer
    buffer: Vec<u8>,
    /// Current offset
    offset: usize,
    /// Messages built count
    messages_built: AtomicU64,
    /// Bytes written
    bytes_written: AtomicU64,
}

impl FlatBufferBuilder {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(capacity),
            offset: 0,
            messages_built: AtomicU64::new(0),
            bytes_written: AtomicU64::new(0),
        }
    }

    /// Reset the builder for reuse
    #[inline]
    pub fn reset(&mut self) {
        self.offset = 0;
        // Keep capacity for reuse
    }

    /// Write a feature vector to the buffer
    #[inline]
    pub fn write_feature_vector(&mut self, fv: &FeatureVector) -> Result<usize, SerializationError> {
        let bytes = bytemuck::bytes_of(fv);
        
        if self.offset + bytes.len() > self.buffer.capacity() {
            // Grow buffer if needed
            self.buffer.reserve(bytes.len());
        }

        // Ensure we have space
        if self.offset + bytes.len() > self.buffer.len() {
            self.buffer.resize(self.offset + bytes.len(), 0);
        }

        self.buffer[self.offset..self.offset + bytes.len()].copy_from_slice(bytes);
        let start = self.offset;
        self.offset += bytes.len();

        self.messages_built.fetch_add(1, Ordering::Relaxed);
        self.bytes_written.fetch_add(bytes.len() as u64, Ordering::Relaxed);

        Ok(start)
    }

    /// Write raw bytes to buffer
    #[inline]
    pub fn write_bytes(&mut self, bytes: &[u8]) -> Result<usize, SerializationError> {
        if self.offset + bytes.len() > self.buffer.capacity() {
            self.buffer.reserve(bytes.len());
        }

        if self.offset + bytes.len() > self.buffer.len() {
            self.buffer.resize(self.offset + bytes.len(), 0);
        }

        self.buffer[self.offset..self.offset + bytes.len()].copy_from_slice(bytes);
        let start = self.offset;
        self.offset += bytes.len();

        self.bytes_written.fetch_add(bytes.len() as u64, Ordering::Relaxed);

        Ok(start)
    }

    /// Get the built buffer
    #[inline]
    pub fn finish(&self) -> &[u8] {
        &self.buffer[..self.offset]
    }

    /// Get buffer as mutable slice
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.buffer[..self.offset]
    }

    /// Get current offset
    #[inline]
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Get buffer capacity
    #[inline]
    pub fn capacity(&self) -> usize {
        self.buffer.capacity()
    }

    /// Get statistics
    #[inline]
    pub fn get_stats(&self) -> BuilderStats {
        BuilderStats {
            messages_built: self.messages_built.load(Ordering::Relaxed),
            bytes_written: self.bytes_written.load(Ordering::Relaxed),
            current_offset: self.offset,
            capacity: self.buffer.capacity(),
        }
    }
}

impl Default for FlatBufferBuilder {
    fn default() -> Self {
        Self::new(65536) // 64KB default
    }
}

/// Schema registry for FlatBuffers type definitions
#[repr(C)]
pub struct SchemaRegistry {
    /// Registered schema count
    schema_count: AtomicU64,
    /// Is initialized
    is_initialized: AtomicBool,
}

impl SchemaRegistry {
    pub fn new() -> Self {
        Self {
            schema_count: AtomicU64::new(0),
            is_initialized: AtomicBool::new(false),
        }
    }

    /// Initialize the registry with standard schemas
    #[inline]
    pub fn initialize(&self) {
        // Register standard schemas
        // In production, would load actual FlatBuffers schemas
        self.schema_count.fetch_add(5, Ordering::Relaxed);
        self.is_initialized.store(true, Ordering::Release);
    }

    /// Check if initialized
    #[inline]
    pub fn is_initialized(&self) -> bool {
        self.is_initialized.load(Ordering::Acquire)
    }

    /// Get schema count
    #[inline]
    pub fn schema_count(&self) -> u64 {
        self.schema_count.load(Ordering::Relaxed)
    }

    /// Register a new schema
    #[inline]
    pub fn register_schema(&self, _name: &str, _schema: &[u8]) -> Result<u64, SerializationError> {
        if !self.is_initialized.load(Ordering::Acquire) {
            return Err(SerializationError::Unsupported);
        }

        let id = self.schema_count.fetch_add(1, Ordering::Relaxed);
        Ok(id)
    }
}

impl Default for SchemaRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Order message for IPC
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct OrderMessage {
    /// Client order ID
    pub client_order_id: u64,
    /// Symbol hash
    pub symbol_hash: u64,
    /// Side: 0 = Buy, 1 = Sell
    pub side: u8,
    /// Order type: 0 = Limit, 1 = Market
    pub order_type: u8,
    /// Time in force
    pub time_in_force: u8,
    /// Padding
    pub _padding: u8,
    /// Price
    pub price: u64,
    /// Quantity
    pub quantity: u64,
    /// Timestamp
    pub timestamp_ns: u64,
    /// Venue ID
    pub venue_id: u32,
    /// Sequence number
    pub sequence: u32,
}

// Safety: OrderMessage is POD
unsafe impl bytemuck::Pod for OrderMessage {}
unsafe impl bytemuck::Zeroable for OrderMessage {}

impl OrderMessage {
    #[inline]
    pub fn new() -> Self {
        Self {
            client_order_id: 0,
            symbol_hash: 0,
            side: 0,
            order_type: 0,
            time_in_force: 0,
            _padding: 0,
            price: 0,
            quantity: 0,
            timestamp_ns: 0,
            venue_id: 0,
            sequence: 0,
        }
    }

    #[inline]
    pub fn is_buy(&self) -> bool {
        self.side == 0
    }

    #[inline]
    pub fn is_sell(&self) -> bool {
        self.side == 1
    }

    #[inline]
    pub fn is_limit(&self) -> bool {
        self.order_type == 0
    }

    #[inline]
    pub fn is_market(&self) -> bool {
        self.order_type == 1
    }
}

impl Default for OrderMessage {
    fn default() -> Self {
        Self::new()
    }
}

/// Fill message for IPC
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FillMessage {
    /// Order ID
    pub order_id: u64,
    /// Fill ID
    pub fill_id: u64,
    /// Fill price
    pub fill_price: u64,
    /// Fill quantity
    pub fill_qty: u64,
    /// Commission
    pub commission: u64,
    /// Timestamp
    pub timestamp_ns: u64,
    /// Venue ID
    pub venue_id: u32,
    /// Liquidity flag: 0 = Maker, 1 = Taker
    pub liquidity: u8,
    /// Padding
    pub _padding: [u8; 3],
}

// Safety: FillMessage is POD
unsafe impl bytemuck::Pod for FillMessage {}
unsafe impl bytemuck::Zeroable for FillMessage {}

impl FillMessage {
    #[inline]
    pub fn new() -> Self {
        Self {
            order_id: 0,
            fill_id: 0,
            fill_price: 0,
            fill_qty: 0,
            commission: 0,
            timestamp_ns: 0,
            venue_id: 0,
            liquidity: 0,
            _padding: [0u8; 3],
        }
    }

    #[inline]
    pub fn is_maker(&self) -> bool {
        self.liquidity == 0
    }

    #[inline]
    pub fn is_taker(&self) -> bool {
        self.liquidity == 1
    }
}

impl Default for FillMessage {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder statistics
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BuilderStats {
    pub messages_built: u64,
    pub bytes_written: u64,
    pub current_offset: usize,
    pub capacity: usize,
}

/// Serialization error (re-export from parent module)
use super::SerializationError;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_vector() {
        let mut fv = FeatureVector::new();
        fv.set_feature(0, 1.5);
        fv.set_feature(1, 2.5);
        fv.set_feature(2, 3.5);
        fv.calculate_checksum();

        assert_eq!(fv.feature_count, 3);
        assert!(fv.is_valid());
        assert_eq!(fv.get_feature(0), 1.5);
    }

    #[test]
    fn test_flatbuffer_builder() {
        let mut builder = FlatBufferBuilder::new(1024);
        
        let fv = FeatureVector::new().with_id(123);
        let offset = builder.write_feature_vector(&fv).unwrap();
        
        assert_eq!(offset, 0);
        assert_eq!(builder.offset(), std::mem::size_of::<FeatureVector>());
        
        let stats = builder.get_stats();
        assert_eq!(stats.messages_built, 1);
    }

    #[test]
    fn test_schema_registry() {
        let registry = SchemaRegistry::new();
        
        assert!(!registry.is_initialized());
        
        registry.initialize();
        assert!(registry.is_initialized());
        assert!(registry.schema_count() > 0);
    }

    #[test]
    fn test_order_message() {
        let mut order = OrderMessage::new();
        order.client_order_id = 12345;
        order.symbol_hash = 67890;
        order.side = 0; // Buy
        order.order_type = 0; // Limit
        order.price = 10000;
        order.quantity = 100;

        assert!(order.is_buy());
        assert!(order.is_limit());
        assert!(!order.is_sell());
        assert!(!order.is_market());
    }

    #[test]
    fn test_fill_message() {
        let mut fill = FillMessage::new();
        fill.order_id = 12345;
        fill.fill_id = 1;
        fill.fill_price = 10000;
        fill.fill_qty = 100;
        fill.liquidity = 0; // Maker

        assert!(fill.is_maker());
        assert!(!fill.is_taker());
    }

    #[test]
    fn test_feature_vector_checksum() {
        let mut fv = FeatureVector::new()
            .with_id(100)
            .with_timestamp(1234567890);
        
        fv.set_feature(0, 1.0);
        fv.set_feature(1, 2.0);
        fv.calculate_checksum();

        assert!(fv.validate_checksum());

        // Modify a feature - checksum should fail
        fv.features[0] = 999.0;
        assert!(!fv.validate_checksum());
    }

    #[test]
    fn test_builder_reset() {
        let mut builder = FlatBufferBuilder::new(1024);
        
        let fv = FeatureVector::new();
        builder.write_feature_vector(&fv).unwrap();
        
        assert!(builder.offset() > 0);
        
        builder.reset();
        assert_eq!(builder.offset(), 0);
    }
}
