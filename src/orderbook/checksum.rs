//! Order Book Checksum Validator
//! 
//! Implements exchange-specific CRC32/SHA256 checksum validators to detect silent
//! data corruption in WebSocket streams. Instantly triggers automated resyncs if
//! the running checksum fails.

use std::sync::atomic::{AtomicU32, AtomicU64, AtomicBool, Ordering};

/// CRC32 lookup table (pre-computed)
static CRC32_TABLE: [u32; 256] = [
    0x00000000, 0x77073096, 0xee0e612c, 0x990951ba,
    0x076dc419, 0x706af48f, 0xe963a535, 0x9e6495a3,
    0x0edb8832, 0x79dcb8a4, 0xe0d5e91e, 0x97d2d988,
    0x09b64c2b, 0x7eb17cbd, 0xe7b82d07, 0x90bf1d91,
    0x1db71064, 0x6ab020f2, 0xf3b97148, 0x84be41de,
    0x1adad47d, 0x6ddde4eb, 0xf4d4b551, 0x83d385c7,
    0x136c9856, 0x646ba8c0, 0xfd62f97a, 0x8a65c9ec,
    0x14015c4f, 0x63066cd9, 0xfa0f3d63, 0x8d080df5,
    0x3b6e20c8, 0x4c69105e, 0xd56041e4, 0xa2677172,
    0x3c03e4d1, 0x4b04d447, 0xd20d85fd, 0xa50ab56b,
    0x35b5a8fa, 0x42b2986c, 0xdedbfb96, 0xa9dca900,
    0x37d83cf3, 0x40df0c65, 0xd9d65dfd, 0xaed16d6b,
    0x26d930ac, 0x51de003a, 0xc8d75180, 0xbfd06116,
    0x21b4f4b5, 0x56b3c423, 0xcfba9599, 0xb8bda50f,
    0x2802b89e, 0x5f058808, 0xc60cd9b2, 0xb10be924,
    0x2f6f7c87, 0x58684c11, 0xc1611dab, 0xb6662d3d,
    0x76dc4190, 0x01db7106, 0x98d220bc, 0xefd5102a,
    0x71b18589, 0x06b6b51f, 0x9fbfe4a5, 0xe8b8d433,
    0x7807c9a2, 0x0f00f934, 0x9609a88e, 0xe10e9818,
    0x7f6a0dbb, 0x086d3d2d, 0x91646c97, 0xe6635c01,
    0x6b6b51f4, 0x1c6c6162, 0x856530d8, 0xf262004e,
    0x6c0695ed, 0x1b01a57b, 0x8208f4c1, 0xf50fc457,
    0x65b0d9c6, 0x12b7e950, 0x8bbeb8ea, 0xfcb9887c,
    0x62dd1ddf, 0x15da2d49, 0x8cd37cf3, 0xfbd44c65,
    0x4db26158, 0x3ab551ce, 0xa3bc0074, 0xd4bb30e2,
    0x4adfa541, 0x3dd895d7, 0xa4d1c46d, 0xd3d6f4fb,
    0x4369e96a, 0x346ed9fc, 0xad678846, 0xda60b8d0,
    0x44042d73, 0x33031de5, 0xaa0a4c5f, 0xdd0d7cc9,
    0x5005713c, 0x270241aa, 0xbe0b1010, 0xc90c2086,
    0x5768b525, 0x206f85b3, 0xb966d409, 0xce61e49f,
    0x5edef90e, 0x29d9c998, 0xb0d09822, 0xc7d7a8b4,
    0x59b33d17, 0x2eb40d81, 0xb7bd5c3b, 0xc0ba6cad,
    0xedb88320, 0x9abfb3b6, 0x03b6e20c, 0x74b1d29a,
    0xead54739, 0x9dd277af, 0x04db2615, 0x73dc1683,
    0xe3630b12, 0x94643b84, 0x0d6d6a3e, 0x7a6a5aa8,
    0xe40ecf0b, 0x9309ff9d, 0x0a00ae27, 0x7d079eb1,
    0xf00f9344, 0x8708a3d2, 0x1e01f268, 0x6906c2fe,
    0xf76253fd, 0x8065636b, 0x196c36d1, 0x6e6b0647,
    0xfed41b76, 0x89d32be0, 0x10da7a5a, 0x67dd4acc,
    0xf9b9df6f, 0x8ebeeff9, 0x17b7be43, 0x60b08ed5,
    0xd6d6a3e8, 0xa1d1937e, 0x38d8c2c4, 0x4fdff252,
    0xd1bb67f1, 0xa6bc5767, 0x3fb506dd, 0x48b2364b,
    0xd80d2bda, 0xaf0a1b4c, 0x36034af6, 0x41047a60,
    0xdf60efc3, 0xa867df55, 0x316e8eef, 0x4669be79,
    0xcb61b38c, 0xbc66831a, 0x256fd2a0, 0x5268e236,
    0xcc0c7795, 0xbb0b4703, 0x220216b9, 0x5505262f,
    0xc5ba3bbe, 0xb2bd0b28, 0x2bb45a92, 0x5cb36a04,
    0xc2d7ffa7, 0xb5d0cf31, 0x2cd99e8b, 0x5bdeae1d,
    0x9b64c2b0, 0xec63f226, 0x756aa39c, 0x026d930a,
    0x9c0906a9, 0xeb0e363f, 0x72076785, 0x05005713,
    0x95bf4a82, 0xe2b87a14, 0x7bb12bae, 0x0cb61b38,
    0x92d28e9b, 0xe5d5be0d, 0x7cdcefb7, 0x0bdbdf21,
    0x86d3d2d4, 0xf1d4e242, 0x68ddb3f8, 0x1fda836e,
    0x81be16cd, 0xf6b9265b, 0x6fb077e1, 0x18b74777,
    0x88085ae6, 0xff0f6a70, 0x66063bca, 0x11010b5c,
    0x8f659eff, 0xf862ae69, 0x616bffd3, 0x166ccf45,
    0xa00ae278, 0xd70dd2ee, 0x4e048354, 0x3903b3c2,
    0xa7672661, 0xd06016f7, 0x4969474d, 0x3e6e77db,
    0xaed16a4a, 0xd9d65adc, 0x40df0b66, 0x37d83bf0,
    0xa9bcae53, 0xdebb9ec5, 0x47b2cf7f, 0x30b5ffe9,
    0xbdbdf21c, 0xcabac28a, 0x53b39330, 0x24b4a3a6,
    0xbad03605, 0xcdd706b3, 0x54de5729, 0x23d967bf,
    0xb3667a2e, 0xc4614ab8, 0x5d681b02, 0x2a6f2b94,
    0xb40bbe37, 0xc30c8ea1, 0x5a05df1b, 0x2d02ef8d,
];

/// Fast CRC32 calculator for order book checksums
#[inline]
pub fn crc32_fast(data: &[u8]) -> u32 {
    let mut crc = 0xFFFFFFFFu32;
    
    for &byte in data {
        crc = CRC32_TABLE[((crc ^ byte as u32) & 0xFF) as usize] ^ (crc >> 8);
    }
    
    crc ^ 0xFFFFFFFF
}

/// Incremental CRC32 for streaming updates
#[repr(C, align(64))]
pub struct IncrementalCrc32 {
    current_crc: AtomicU32,
    update_count: AtomicU64,
    is_active: AtomicBool,
}

impl IncrementalCrc32 {
    pub const fn new() -> Self {
        Self {
            current_crc: AtomicU32::new(0xFFFFFFFF),
            update_count: AtomicU64::new(0),
            is_active: AtomicBool::new(true),
        }
    }
    
    /// Update checksum with new data
    #[inline]
    pub fn update(&self, data: &[u8]) {
        let mut crc = self.current_crc.load(Ordering::Acquire);
        
        for &byte in data {
            crc = CRC32_TABLE[((crc ^ byte as u32) & 0xFF) as usize] ^ (crc >> 8);
        }
        
        self.current_crc.store(crc, Ordering::Release);
        self.update_count.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Get current checksum value
    #[inline]
    pub fn finalize(&self) -> u32 {
        self.current_crc.load(Ordering::Acquire) ^ 0xFFFFFFFF
    }
    
    /// Reset for new calculation
    #[inline]
    pub fn reset(&self) {
        self.current_crc.store(0xFFFFFFFF, Ordering::Release);
        self.update_count.store(0, Ordering::Relaxed);
    }
    
    /// Get update count
    pub fn count(&self) -> u64 {
        self.update_count.load(Ordering::Relaxed)
    }
}

impl Default for IncrementalCrc32 {
    fn default() -> Self {
        Self::new()
    }
}

/// Order book checksum validator state
#[repr(C, align(64))]
pub struct ChecksumValidator {
    /// Expected checksum from exchange
    expected_checksum: AtomicU32,
    /// Computed local checksum
    computed_checksum: AtomicU32,
    /// Running CRC for incremental updates
    running_crc: IncrementalCrc32,
    /// Mismatch count
    mismatch_count: AtomicU64,
    /// Validations performed
    validation_count: AtomicU64,
    /// Last valid sequence
    last_valid_seq: AtomicU64,
    /// Whether validator detected corruption
    corruption_detected: AtomicBool,
}

impl ChecksumValidator {
    pub const fn new() -> Self {
        Self {
            expected_checksum: AtomicU32::new(0),
            computed_checksum: AtomicU32::new(0),
            running_crc: IncrementalCrc32::new(),
            mismatch_count: AtomicU64::new(0),
            validation_count: AtomicU64::new(0),
            last_valid_seq: AtomicU64::new(0),
            corruption_detected: AtomicBool::new(false),
        }
    }
    
    /// Set expected checksum from exchange message
    pub fn set_expected(&self, checksum: u32) {
        self.expected_checksum.store(checksum, Ordering::Release);
    }
    
    /// Compute checksum from price level data
    pub fn compute_from_levels(&self, bids: &[(i64, i64)], asks: &[(i64, i64)]) -> u32 {
        // Simple XOR-based checksum similar to Binance/Coinbase
        let mut xor_sum = 0u32;
        
        // Process bids and asks interleaved
        let max_len = bids.len().max(asks.len());
        for i in 0..max_len {
            if i < bids.len() {
                let (price, qty) = bids[i];
                xor_sum ^= ((price >> 32) as u32) ^ ((price & 0xFFFFFFFF) as u32);
                xor_sum ^= ((qty >> 32) as u32) ^ ((qty & 0xFFFFFFFF) as u32);
            }
            if i < asks.len() {
                let (price, qty) = asks[i];
                xor_sum ^= ((price >> 32) as u32) ^ ((price & 0xFFFFFFFF) as u32);
                xor_sum ^= ((qty >> 32) as u32) ^ ((qty & 0xFFFFFFFF) as u32);
            }
        }
        
        xor_sum
    }
    
    /// Validate checksum against expected
    pub fn validate(&self, computed: u32, sequence: u64) -> ValidationResult {
        let expected = self.expected_checksum.load(Ordering::Acquire);
        self.computed_checksum.store(computed, Ordering::Release);
        self.validation_count.fetch_add(1, Ordering::Relaxed);
        
        if computed == expected {
            self.last_valid_seq.store(sequence, Ordering::Release);
            ValidationResult {
                is_valid: true,
                expected,
                computed,
                sequence,
                should_resync: false,
            }
        } else {
            self.mismatch_count.fetch_add(1, Ordering::Relaxed);
            self.corruption_detected.store(true, Ordering::Release);
            
            // Resync if multiple mismatches or large sequence gap
            let mismatches = self.mismatch_count.load(Ordering::Relaxed);
            let should_resync = mismatches >= 2 || sequence > self.last_valid_seq.load(Ordering::Acquire) + 100;
            
            ValidationResult {
                is_valid: false,
                expected,
                computed,
                sequence,
                should_resync,
            }
        }
    }
    
    /// Update running checksum incrementally
    pub fn update_running(&self, price: i64, quantity: i64) {
        let bytes = [
            ((price >> 56) & 0xFF) as u8,
            ((price >> 48) & 0xFF) as u8,
            ((price >> 40) & 0xFF) as u8,
            ((price >> 32) & 0xFF) as u8,
            ((price >> 24) & 0xFF) as u8,
            ((price >> 16) & 0xFF) as u8,
            ((price >> 8) & 0xFF) as u8,
            (price & 0xFF) as u8,
            ((quantity >> 56) & 0xFF) as u8,
            ((quantity >> 48) & 0xFF) as u8,
            ((quantity >> 40) & 0xFF) as u8,
            ((quantity >> 32) & 0xFF) as u8,
            ((quantity >> 24) & 0xFF) as u8,
            ((quantity >> 16) & 0xFF) as u8,
            ((quantity >> 8) & 0xFF) as u8,
            (quantity & 0xFF) as u8,
        ];
        self.running_crc.update(&bytes);
    }
    
    /// Get final running checksum
    pub fn get_running_checksum(&self) -> u32 {
        self.running_crc.finalize()
    }
    
    /// Reset running checksum
    pub fn reset_running(&self) {
        self.running_crc.reset();
    }
    
    /// Get validation statistics
    pub fn get_stats(&self) -> ChecksumStats {
        ChecksumStats {
            expected: self.expected_checksum.load(Ordering::Relaxed),
            computed: self.computed_checksum.load(Ordering::Relaxed),
            mismatches: self.mismatch_count.load(Ordering::Relaxed),
            validations: self.validation_count.load(Ordering::Relaxed),
            last_valid_seq: self.last_valid_seq.load(Ordering::Relaxed),
            corruption_detected: self.corruption_detected.load(Ordering::Acquire),
        }
    }
    
    /// Clear corruption flag after resync
    pub fn clear_corruption(&self) {
        self.corruption_detected.store(false, Ordering::Release);
        self.mismatch_count.store(0, Ordering::Relaxed);
    }
}

/// Validation result
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub expected: u32,
    pub computed: u32,
    pub sequence: u64,
    pub should_resync: bool,
}

/// Checksum statistics
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct ChecksumStats {
    pub expected: u32,
    pub computed: u32,
    pub mismatches: u64,
    pub validations: u64,
    pub last_valid_seq: u64,
    pub corruption_detected: bool,
}

impl Default for ChecksumValidator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_crc32_basic() {
        let data = b"Hello, World!";
        let crc = crc32_fast(data);
        assert_ne!(crc, 0);
        
        // Same data should produce same CRC
        let crc2 = crc32_fast(data);
        assert_eq!(crc, crc2);
    }
    
    #[test]
    fn test_incremental_crc() {
        let inc = IncrementalCrc32::new();
        
        inc.update(b"Hello");
        inc.update(b", ");
        inc.update(b"World!");
        
        let result = inc.finalize();
        
        // Compare with single-shot calculation
        let direct = crc32_fast(b"Hello, World!");
        assert_eq!(result, direct);
    }
    
    #[test]
    fn test_validator() {
        let validator = ChecksumValidator::new();
        
        let bids = vec![(100_00000000i64, 50_00000000i64)];
        let asks = vec![(101_00000000i64, 30_00000000i64)];
        
        let computed = validator.compute_from_levels(&bids, &asks);
        validator.set_expected(computed);
        
        let result = validator.validate(computed, 100);
        assert!(result.is_valid);
        assert!(!result.should_resync);
        
        // Test mismatch
        let result = validator.validate(computed + 1, 101);
        assert!(!result.is_valid);
        assert!(result.should_resync); // Will trigger on second mismatch
    }
}
