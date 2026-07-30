//! Gorilla Time-Series Compression for High-Frequency Ticks
//! 
//! Implements Facebook's Gorilla compression algorithm for trade and quote ticks.
//! Uses delta-of-delta encoding for timestamps to achieve extreme compression ratios.

use std::mem;

/// Maximum number of leading zeros that can be encoded in 5 bits
const MAX_LEADING_ZEROS: u64 = 31;

/// Gorilla compressor for timestamp series using delta-of-delta encoding
pub struct GorillaTimestampCompressor {
    /// Previous timestamp (nanoseconds)
    prev_timestamp: u64,
    /// Previous delta
    prev_delta: i64,
    /// Whether this is the first value
    first: bool,
    /// Compressed bitstream buffer
    buffer: Vec<u8>,
    /// Current bit position in buffer
    bit_position: usize,
}

impl GorillaTimestampCompressor {
    pub fn new() -> Self {
        Self {
            prev_timestamp: 0,
            prev_delta: 0,
            first: true,
            buffer: Vec::with_capacity(4096),
            bit_position: 0,
        }
    }

    /// Write a single bit to the buffer
    #[inline]
    fn write_bit(&mut self, bit: u8) {
        if self.bit_position % 8 == 0 {
            self.buffer.push(0);
        }
        let byte_idx = self.bit_position / 8;
        let bit_idx = 7 - (self.bit_position % 8);
        self.buffer[byte_idx] |= bit << bit_idx;
        self.bit_position += 1;
    }

    /// Write multiple bits to the buffer
    #[inline]
    fn write_bits(&mut self, value: u64, num_bits: usize) {
        for i in (0..num_bits).rev() {
            let bit = ((value >> i) & 1) as u8;
            self.write_bit(bit);
        }
    }

    /// Count leading zeros using LZCNT instruction when available
    #[inline]
    fn leading_zeros(value: u64) -> u32 {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("bmi1") {
                unsafe {
                    return core::arch::x86_64::_lzcnt_u64(value);
                }
            }
        }
        value.leading_zeros()
    }

    /// Add a timestamp to the compressor
    pub fn add_timestamp(&mut self, timestamp: u64) {
        if self.first {
            // Store raw timestamp for first value (64 bits)
            self.prev_timestamp = timestamp;
            self.write_bits(timestamp, 64);
            self.first = false;
            return;
        }

        // Calculate delta
        let delta = timestamp as i64 - self.prev_timestamp as i64;
        
        // Calculate delta-of-delta
        let dod = delta - self.prev_delta;

        if dod == 0 {
            // Same delta as before: write '0'
            self.write_bit(0);
        } else {
            // Different delta: write '1' + encoding
            self.write_bit(1);

            if dod.abs() < (1i64 << 5) {
                // Small delta fits in 5 bits with sign: '00' + 5 bits
                self.write_bits(0b00, 2);
                self.write_bits((dod & 0x1F) as u64, 5);
            } else if dod.abs() < (1i64 << 10) {
                // Medium delta: '01' + 5 bits leading zeros + 10 bits value
                let abs_dod = dod.unsigned_abs();
                let leading = Self::leading_zeros(abs_dod as u64 | 1).min(MAX_LEADING_ZEROS) as u64;
                let significant_bits = 64 - leading;
                
                self.write_bits(0b01, 2);
                self.write_bits(leading, 5);
                self.write_bits(abs_dod, significant_bits as usize);
            } else {
                // Large delta: '1' + 5 bits leading zeros + 10 bits length + full value
                let abs_dod = dod.unsigned_abs();
                let leading = Self::leading_zeros(abs_dod as u64 | 1).min(MAX_LEADING_ZEROS) as u64;
                let significant_bits = 64 - leading;
                
                self.write_bit(1);
                self.write_bits(leading, 5);
                self.write_bits(significant_bits - 1, 10);
                self.write_bits(abs_dod, significant_bits as usize);
            }
        }

        self.prev_delta = delta;
        self.prev_timestamp = timestamp;
    }

    /// Get compressed data
    pub fn get_compressed(&self) -> &[u8] {
        &self.buffer
    }

    /// Finalize and get compressed data (aligns to byte boundary)
    pub fn finalize(mut self) -> Vec<u8> {
        // Pad to byte boundary
        while self.bit_position % 8 != 0 {
            self.write_bit(0);
        }
        self.buffer
    }

    /// Reset compressor for new series
    pub fn reset(&mut self) {
        self.prev_timestamp = 0;
        self.prev_delta = 0;
        self.first = true;
        self.buffer.clear();
        self.bit_position = 0;
    }

    /// Get compression statistics
    pub fn stats(&self) -> TimestampCompressionStats {
        TimestampCompressionStats {
            values_encoded: if self.first { 0 } else { 1 },
            compressed_bytes: self.buffer.len(),
            bits_used: self.bit_position,
        }
    }
}

impl Default for GorillaTimestampCompressor {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct TimestampCompressionStats {
    pub values_encoded: usize,
    pub compressed_bytes: usize,
    pub bits_used: usize,
}

/// Gorilla compressor for floating-point values using XOR encoding
pub struct GorillaFloatCompressor {
    /// Previous value
    prev_value: f64,
    /// Whether this is the first value
    first: bool,
    /// Compressed bitstream buffer
    buffer: Vec<u8>,
    /// Current bit position
    bit_position: usize,
}

impl GorillaFloatCompressor {
    pub fn new() -> Self {
        Self {
            prev_value: 0.0,
            first: true,
            buffer: Vec::with_capacity(4096),
            bit_position: 0,
        }
    }

    /// Write a single bit
    #[inline]
    fn write_bit(&mut self, bit: u8) {
        if self.bit_position % 8 == 0 {
            self.buffer.push(0);
        }
        let byte_idx = self.bit_position / 8;
        let bit_idx = 7 - (self.bit_position % 8);
        self.buffer[byte_idx] |= bit << bit_idx;
        self.bit_position += 1;
    }

    /// Write multiple bits
    #[inline]
    fn write_bits(&mut self, value: u64, num_bits: usize) {
        for i in (0..num_bits).rev() {
            let bit = ((value >> i) & 1) as u8;
            self.write_bit(bit);
        }
    }

    /// Count leading zeros
    #[inline]
    fn leading_zeros(value: u64) -> u32 {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("bmi1") {
                unsafe {
                    return core::arch::x86_64::_lzcnt_u64(value);
                }
            }
        }
        value.leading_zeros()
    }

    /// Count trailing zeros
    #[inline]
    fn trailing_zeros(value: u64) -> u32 {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("bmi1") {
                unsafe {
                    return core::arch::x86_64::_tzcnt_u64(value);
                }
            }
        }
        value.trailing_zeros()
    }

    /// Add a floating-point value (price or size)
    pub fn add_value(&mut self, value: f64) {
        let value_bits = value.to_bits();

        if self.first {
            // Store raw value for first entry (64 bits)
            self.prev_value = value;
            self.write_bits(value_bits, 64);
            self.first = false;
            return;
        }

        // XOR with previous value
        let xor = value_bits ^ self.prev_value.to_bits();

        if xor == 0 {
            // Same value: write '0'
            self.write_bit(0);
        } else {
            // Different value: write '1' + encoding
            self.write_bit(1);

            // Count leading and trailing zeros in XOR result
            let leading = Self::leading_zeros(xor);
            let trailing = Self::trailing_zeros(xor);
            
            // Calculate significant bits
            let significant_bits = 64 - leading - trailing;

            // Check if we can use same block as previous
            // For simplicity, always encode block info (can be optimized)
            self.write_bits(leading, 5);
            self.write_bits(significant_bits, 6);
            self.write_bits(xor >> trailing, significant_bits as usize);
        }

        self.prev_value = value;
    }

    /// Get compressed data
    pub fn get_compressed(&self) -> &[u8] {
        &self.buffer
    }

    /// Finalize and get compressed data
    pub fn finalize(mut self) -> Vec<u8> {
        while self.bit_position % 8 != 0 {
            self.write_bit(0);
        }
        self.buffer
    }

    /// Reset compressor
    pub fn reset(&mut self) {
        self.prev_value = 0.0;
        self.first = true;
        self.buffer.clear();
        self.bit_position = 0;
    }

    /// Get compression statistics
    pub fn stats(&self) -> FloatCompressionStats {
        FloatCompressionStats {
            values_encoded: if self.first { 0 } else { 1 },
            compressed_bytes: self.buffer.len(),
            bits_used: self.bit_position,
        }
    }
}

impl Default for GorillaFloatCompressor {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct FloatCompressionStats {
    pub values_encoded: usize,
    pub compressed_bytes: usize,
    pub bits_used: usize,
}

/// Combined tick compressor for full OHLCV/tick records
pub struct TickCompressor {
    timestamp_compressor: GorillaTimestampCompressor,
    price_compressor: GorillaFloatCompressor,
    size_compressor: GorillaFloatCompressor,
    ticks_compressed: u64,
}

impl TickCompressor {
    pub fn new() -> Self {
        Self {
            timestamp_compressor: GorillaTimestampCompressor::new(),
            price_compressor: GorillaFloatCompressor::new(),
            size_compressor: GorillaFloatCompressor::new(),
            ticks_compressed: 0,
        }
    }

    /// Add a complete tick record
    pub fn add_tick(&mut self, timestamp: u64, price: f64, size: f64) {
        self.timestamp_compressor.add_timestamp(timestamp);
        self.price_compressor.add_value(price);
        self.size_compressor.add_value(size);
        self.ticks_compressed += 1;
    }

    /// Finalize and get all compressed data
    pub fn finalize(self) -> TickCompressedData {
        TickCompressedData {
            timestamps: self.timestamp_compressor.finalize(),
            prices: self.price_compressor.finalize(),
            sizes: self.size_compressor.finalize(),
            ticks_count: self.ticks_compressed,
        }
    }

    /// Reset all compressors
    pub fn reset(&mut self) {
        self.timestamp_compressor.reset();
        self.price_compressor.reset();
        self.size_compressor.reset();
        self.ticks_compressed = 0;
    }

    /// Get overall compression ratio estimate
    pub fn estimated_ratio(&self) -> f64 {
        let uncompressed_size = self.ticks_compressed as usize * (8 + 8 + 8); // ts + price + size
        let compressed_size = self.timestamp_compressor.stats().compressed_bytes
            + self.price_compressor.stats().compressed_bytes
            + self.size_compressor.stats().compressed_bytes;
        
        if compressed_size == 0 {
            return 0.0;
        }
        uncompressed_size as f64 / compressed_size as f64
    }
}

impl Default for TickCompressor {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct TickCompressedData {
    pub timestamps: Vec<u8>,
    pub prices: Vec<u8>,
    pub sizes: Vec<u8>,
    pub ticks_count: u64,
}

/// Decompressor for Gorilla-compressed timestamps
pub struct TimestampDecompressor {
    data: Vec<u8>,
    bit_position: usize,
    prev_timestamp: u64,
    prev_delta: i64,
    first: bool,
}

impl TimestampDecompressor {
    pub fn new(data: Vec<u8>) -> Self {
        Self {
            data,
            bit_position: 0,
            prev_timestamp: 0,
            prev_delta: 0,
            first: true,
        }
    }

    #[inline]
    fn read_bit(&mut self) -> Option<u8> {
        if self.bit_position >= self.data.len() * 8 {
            return None;
        }
        let byte_idx = self.bit_position / 8;
        let bit_idx = 7 - (self.bit_position % 8);
        let bit = (self.data[byte_idx] >> bit_idx) & 1;
        self.bit_position += 1;
        Some(bit)
    }

    #[inline]
    fn read_bits(&mut self, num_bits: usize) -> Option<u64> {
        let mut value = 0u64;
        for _ in 0..num_bits {
            value = (value << 1) | self.read_bit()? as u64;
        }
        Some(value)
    }

    /// Decode next timestamp
    pub fn next(&mut self) -> Option<u64> {
        if self.first {
            self.prev_timestamp = self.read_bits(64)?;
            self.first = false;
            return Some(self.prev_timestamp);
        }

        match self.read_bit()? {
            0 => {
                // Same delta
                self.prev_timestamp = (self.prev_timestamp as i64 + self.prev_delta) as u64;
            }
            1 => {
                // Different delta
                let mode = self.read_bits(2)?;
                let dod = match mode {
                    0b00 => {
                        // 5-bit signed delta
                        let val = self.read_bits(5)? as i64;
                        // Sign extend
                        if val & 0x10 != 0 {
                            val | !0x1F
                        } else {
                            val
                        }
                    }
                    0b01 => {
                        // Variable length with leading zeros
                        let leading = self.read_bits(5)?;
                        let remaining = 64 - leading;
                        let val = self.read_bits(remaining as usize)? as i64;
                        val
                    }
                    _ => {
                        // Full 64-bit value
                        let leading = self.read_bits(5)?;
                        let sig_bits = self.read_bits(10)? + 1;
                        let val = self.read_bits(sig_bits as usize)? as i64;
                        val
                    }
                };
                self.prev_delta = dod;
                self.prev_timestamp = (self.prev_timestamp as i64 + self.prev_delta) as u64;
            }
            _ => unreachable!(),
        }

        Some(self.prev_timestamp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timestamp_compression_monotonic() {
        let mut compressor = GorillaTimestampCompressor::new();
        
        // Monotonically increasing timestamps (typical tick data)
        let timestamps = vec![1000u64, 1005, 1010, 1015, 1020, 1025, 1030];
        
        for ts in &timestamps {
            compressor.add_timestamp(*ts);
        }

        let compressed = compressor.finalize();
        let original_size = timestamps.len() * 8;
        let compressed_size = compressed.len();
        
        println!("Original: {} bytes, Compressed: {} bytes", original_size, compressed_size);
        assert!(compressed_size < original_size);
    }

    #[test]
    fn test_float_compression_similar_values() {
        let mut compressor = GorillaFloatCompressor::new();
        
        // Similar prices (typical in HFT)
        let prices = vec![100.00, 100.01, 100.00, 100.02, 100.01, 100.00];
        
        for price in &prices {
            compressor.add_value(*price);
        }

        let compressed = compressor.finalize();
        let original_size = prices.len() * 8;
        let compressed_size = compressed.len();
        
        println!("Original: {} bytes, Compressed: {} bytes", original_size, compressed_size);
        assert!(compressed_size < original_size);
    }

    #[test]
    fn test_tick_compressor() {
        let mut compressor = TickCompressor::new();
        
        for i in 0..100 {
            compressor.add_tick(
                1000000 + i * 5,
                100.0 + (i % 10) as f64 * 0.01,
                1.0 + (i % 5) as f64 * 0.1,
            );
        }

        let ratio = compressor.estimated_ratio();
        println!("Estimated compression ratio: {:.2}x", ratio);
        assert!(ratio > 1.0);
    }
}
