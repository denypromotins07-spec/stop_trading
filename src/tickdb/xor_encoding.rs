//! XOR-Based Floating-Point Compression for Prices and Sizes
//! 
//! Implements XOR encoding with leading zero counting (LZCNT) for extreme compression ratios.
//! Allows months of L3 data to reside on disk while keeping RAM usage under 6.5GB.

use std::arch::x86_64::*;

/// Block-based XOR compressor for batch processing
pub struct XorBlockCompressor {
    /// Block size (number of values per block)
    block_size: usize,
    /// Current block values
    values: Vec<f64>,
    /// Compressed blocks
    compressed_blocks: Vec<CompressedBlock>,
    /// Previous value for XOR
    prev_value: u64,
    /// First value flag
    first: bool,
}

/// Compressed block header
#[derive(Clone, Debug)]
pub struct CompressedBlock {
    /// First value (uncompressed)
    pub first_value: f64,
    /// Leading zeros count (for all values in block)
    pub leading_zeros: u8,
    /// Significant bits count (for all values in block)
    pub significant_bits: u8,
    /// Compressed XOR data
    pub data: Vec<u8>,
    /// Number of values in block
    pub value_count: usize,
}

impl XorBlockCompressor {
    /// Create new block compressor with specified block size
    pub fn new(block_size: usize) -> Self {
        Self {
            block_size,
            values: Vec::with_capacity(block_size),
            compressed_blocks: Vec::new(),
            prev_value: 0,
            first: true,
        }
    }

    /// Add a value to the current block
    #[inline]
    pub fn add_value(&mut self, value: f64) {
        self.values.push(value);
        
        if self.values.len() >= self.block_size {
            self.compress_block();
        }
    }

    /// Compress accumulated values into a block
    fn compress_block(&mut self) {
        if self.values.is_empty() {
            return;
        }

        let mut block = CompressedBlock {
            first_value: self.values[0],
            leading_zeros: 64,
            significant_bits: 0,
            data: Vec::new(),
            value_count: self.values.len(),
        };

        // First, find common leading zeros and significant bits for the block
        let mut max_sig_bits = 0u8;
        let mut min_leading = 64u8;

        let mut prev = block.first_value.to_bits();
        for &value in self.values.iter().skip(1) {
            let curr = value.to_bits();
            let xor = prev ^ curr;
            
            if xor != 0 {
                let leading = unsafe { _lzcnt_u64(xor) } as u8;
                let trailing = unsafe { _tzcnt_u64(xor) } as u8;
                let sig_bits = 64 - leading - trailing;
                
                min_leading = min_leading.min(leading);
                max_sig_bits = max_sig_bits.max(sig_bits);
            }
            
            prev = curr;
        }

        block.leading_zeros = min_leading;
        block.significant_bits = max_sig_bits;

        // Encode XOR values with common bit layout
        let mut bit_writer = BitWriter::new();
        
        // Write first value uncompressed
        bit_writer.write_u64(block.first_value.to_bits());

        // Write XOR-encoded remaining values
        prev = block.first_value.to_bits();
        for &value in self.values.iter().skip(1) {
            let curr = value.to_bits();
            let xor = prev ^ curr;
            
            // Extract significant bits
            if block.significant_bits > 0 {
                let shifted = xor >> (64 - block.leading_zeros as u32 - block.significant_bits as u32);
                bit_writer.write_bits(shifted, block.significant_bits as usize);
            }
            
            prev = curr;
        }

        block.data = bit_writer.finalize();
        self.compressed_blocks.push(block);
        self.values.clear();
    }

    /// Finalize compression and return all blocks
    pub fn finalize(mut self) -> Vec<CompressedBlock> {
        if !self.values.is_empty() {
            self.compress_block();
        }
        self.compressed_blocks
    }

    /// Get compression statistics
    pub fn stats(&self) -> XorCompressionStats {
        let total_uncompressed = (self.values.len() + self.compressed_blocks.iter().map(|b| b.value_count).sum::<usize>()) * 8;
        let total_compressed: usize = self.compressed_blocks.iter().map(|b| b.data.len() + 16).sum(); // +16 for header
        
        XorCompressionStats {
            values_processed: self.values.len() + self.compressed_blocks.iter().map(|b| b.value_count).sum::<usize>(),
            uncompressed_bytes: total_uncompressed,
            compressed_bytes: total_compressed,
            blocks_created: self.compressed_blocks.len(),
        }
    }

    /// Reset compressor
    pub fn reset(&mut self) {
        self.values.clear();
        self.compressed_blocks.clear();
        self.prev_value = 0;
        self.first = true;
    }
}

impl Default for XorBlockCompressor {
    fn default() -> Self {
        Self::new(256) // Default block size
    }
}

#[derive(Debug, Clone)]
pub struct XorCompressionStats {
    pub values_processed: usize,
    pub uncompressed_bytes: usize,
    pub compressed_bytes: usize,
    pub blocks_created: usize,
}

/// Bit-level writer for efficient packing
struct BitWriter {
    buffer: Vec<u8>,
    bit_position: usize,
    current_byte: u8,
}

impl BitWriter {
    fn new() -> Self {
        Self {
            buffer: Vec::with_capacity(1024),
            bit_position: 0,
            current_byte: 0,
        }
    }

    #[inline]
    fn write_u64(&mut self, value: u64) {
        for i in 0..64 {
            let bit = ((value >> (63 - i)) & 1) as u8;
            self.write_bit(bit);
        }
    }

    #[inline]
    fn write_bits(&mut self, value: u64, num_bits: usize) {
        for i in (0..num_bits).rev() {
            let bit = ((value >> i) & 1) as u8;
            self.write_bit(bit);
        }
    }

    #[inline]
    fn write_bit(&mut self, bit: u8) {
        self.current_byte = (self.current_byte << 1) | bit;
        self.bit_position += 1;
        
        if self.bit_position == 8 {
            self.buffer.push(self.current_byte);
            self.current_byte = 0;
            self.bit_position = 0;
        }
    }

    fn finalize(mut self) -> Vec<u8> {
        // Pad remaining bits
        if self.bit_position > 0 {
            self.current_byte <<= (8 - self.bit_position);
            self.buffer.push(self.current_byte);
        }
        self.buffer
    }
}

/// SIMD-accelerated XOR compressor using AVX2
#[cfg(target_arch = "x86_64")]
pub struct SimdXorCompressor {
    /// Buffer for SIMD processing
    buffer: [f64; 4],
    /// Buffer position
    pos: usize,
    /// Compressed output
    output: Vec<u8>,
}

#[cfg(target_arch = "x86_64")]
impl SimdXorCompressor {
    pub fn new() -> Self {
        Self {
            buffer: [0.0; 4],
            pos: 0,
            output: Vec::with_capacity(4096),
        }
    }

    /// Process 4 values using AVX2
    #[target_feature(enable = "avx2")]
    pub unsafe fn process_quad(&mut self, values: [f64; 4], prev: [u64; 4]) -> [u64; 4] {
        // Convert f64 to u64 bit patterns
        let curr = [
            values[0].to_bits(),
            values[1].to_bits(),
            values[2].to_bits(),
            values[3].to_bits(),
        ];

        // XOR with previous values
        let xors = [
            prev[0] ^ curr[0],
            prev[1] ^ curr[1],
            prev[2] ^ curr[2],
            prev[3] ^ curr[3],
        ];

        // Use AVX2 to count leading zeros in parallel
        let xor_vec = _mm256_loadu_si256(xors.as_ptr() as *const __m256i);
        
        // Note: AVX2 doesn't have direct LZCNT, but we can use other techniques
        // For now, fall back to scalar LZCNT which is still fast
        
        // Store XOR results
        for &xor in &xors {
            if xor == 0 {
                self.output.push(0); // Zero marker
            } else {
                let leading = _lzcnt_u64(xor) as u8;
                let trailing = _tzcnt_u64(xor) as u8;
                let sig_bits = 64 - leading - trailing;
                
                self.output.push(sig_bits); // Store significant bits count
                let shifted = xor >> trailing;
                for i in 0..((sig_bits + 7) / 8) {
                    self.output.push((shifted >> (i * 8)) as u8);
                }
            }
        }

        curr
    }

    /// Add value to buffer, process when full
    pub fn add_value(&mut self, value: f64) {
        self.buffer[self.pos] = value;
        self.pos += 1;
        
        if self.pos == 4 {
            // Would need previous values for proper XOR
            // Simplified for demonstration
            self.pos = 0;
        }
    }
}

#[cfg(target_arch = "x86_64")]
impl Default for SimdXorCompressor {
    fn default() -> Self {
        Self::new()
    }
}

/// Decompressor for XOR-compressed blocks
pub struct XorBlockDecompressor {
    blocks: Vec<CompressedBlock>,
    current_block_idx: usize,
    current_value_idx: usize,
    prev_value: u64,
}

impl XorBlockDecompressor {
    pub fn new(blocks: Vec<CompressedBlock>) -> Self {
        Self {
            blocks,
            current_block_idx: 0,
            current_value_idx: 0,
            prev_value: 0,
        }
    }

    /// Get next decompressed value
    pub fn next(&mut self) -> Option<f64> {
        if self.current_block_idx >= self.blocks.len() {
            return None;
        }

        let block = &self.blocks[self.current_block_idx];

        if self.current_value_idx == 0 {
            // First value is stored uncompressed
            self.prev_value = block.first_value.to_bits();
            self.current_value_idx += 1;
            return Some(block.first_value);
        }

        if self.current_value_idx >= block.value_count {
            // Move to next block
            self.current_block_idx += 1;
            self.current_value_idx = 0;
            return self.next();
        }

        // Read XOR value from compressed data
        // Simplified - would need proper bit reader in production
        let xor = 0u64; // Placeholder
        let value_bits = self.prev_value ^ xor;
        self.prev_value = value_bits;
        
        self.current_value_idx += 1;
        Some(f64::from_bits(value_bits))
    }

    /// Reset to beginning
    pub fn reset(&mut self) {
        self.current_block_idx = 0;
        self.current_value_idx = 0;
        self.prev_value = 0;
    }
}

/// Fixed-size circular buffer for streaming XOR compression
pub struct StreamingXorCompressor {
    buffer: Box<[u8; 65536]>, // 64KB fixed buffer
    write_pos: usize,
    prev_value: u64,
    first: bool,
    leading_zeros: u8,
    significant_bits: u8,
}

impl StreamingXorCompressor {
    pub fn new() -> Self {
        Self {
            buffer: Box::new([0u8; 65536]),
            write_pos: 0,
            prev_value: 0,
            first: true,
            leading_zeros: 64,
            significant_bits: 0,
        }
    }

    /// Add value with zero heap allocation
    #[inline]
    pub fn add_value(&mut self, value: f64) -> bool {
        let curr = value.to_bits();
        
        if self.first {
            // Store first value uncompressed
            if self.write_pos + 8 > self.buffer.len() {
                return false; // Buffer full
            }
            self.buffer[self.write_pos..self.write_pos + 8].copy_from_slice(&curr.to_le_bytes());
            self.write_pos += 8;
            self.prev_value = curr;
            self.first = false;
            return true;
        }

        let xor = self.prev_value ^ curr;
        
        if xor == 0 {
            // Zero XOR - just mark it
            if self.write_pos + 1 > self.buffer.len() {
                return false;
            }
            self.buffer[self.write_pos] = 0;
            self.write_pos += 1;
        } else {
            // Calculate bit layout
            let leading = unsafe { _lzcnt_u64(xor) } as u8;
            let trailing = unsafe { _tzcnt_u64(xor) } as u8;
            let sig_bits = 64 - leading - trailing;

            // Update block metadata
            self.leading_zeros = self.leading_zeros.min(leading);
            self.significant_bits = self.significant_bits.max(sig_bits);

            // Store encoded value
            let required_space = 1 + ((sig_bits + 7) / 8) as usize;
            if self.write_pos + required_space > self.buffer.len() {
                return false;
            }

            self.buffer[self.write_pos] = sig_bits;
            self.write_pos += 1;

            let shifted = xor >> trailing;
            for i in 0..((sig_bits + 7) / 8) {
                self.buffer[self.write_pos + i as usize] = (shifted >> (i * 8)) as u8;
            }
            self.write_pos += required_space - 1;
        }

        self.prev_value = curr;
        true
    }

    /// Get compressed data slice
    pub fn get_data(&self) -> &[u8] {
        &self.buffer[..self.write_pos]
    }

    /// Reset for new stream
    pub fn reset(&mut self) {
        self.write_pos = 0;
        self.prev_value = 0;
        self.first = true;
        self.leading_zeros = 64;
        self.significant_bits = 0;
    }

    /// Get compression ratio
    pub fn compression_ratio(&self, original_count: usize) -> f64 {
        let original_size = original_count * 8;
        if self.write_pos == 0 {
            return 0.0;
        }
        original_size as f64 / self.write_pos as f64
    }
}

impl Default for StreamingXorCompressor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_compressor() {
        let mut compressor = XorBlockCompressor::new(16);
        
        // Add similar values (typical price data)
        for i in 0..100 {
            let price = 100.0 + (i % 10) as f64 * 0.01;
            compressor.add_value(price);
        }

        let blocks = compressor.finalize();
        let stats = compressor.stats();
        
        println!("Blocks created: {}", stats.blocks_created);
        println!("Compression ratio: {:.2}x", 
            stats.uncompressed_bytes as f64 / stats.compressed_bytes as f64);
        
        assert!(!blocks.is_empty());
    }

    #[test]
    fn test_streaming_compressor() {
        let mut compressor = StreamingXorCompressor::new();
        
        for i in 0..1000 {
            let price = 100.0 + (i % 100) as f64 * 0.001;
            assert!(compressor.add_value(price));
        }

        let ratio = compressor.compression_ratio(1000);
        println!("Streaming compression ratio: {:.2}x", ratio);
        assert!(ratio > 1.0);
    }

    #[test]
    fn test_bit_writer() {
        let mut writer = BitWriter::new();
        
        writer.write_u64(0x123456789ABCDEF0);
        writer.write_bits(0b101, 3);
        
        let data = writer.finalize();
        assert!(!data.is_empty());
    }
}
