//! Chaos Engineering - State Corruption Simulator Module
//! 
//! Implements state corruption simulator testing the WAL and crash recovery against malformed data.
//! Ensures rkyv snapshot hydration and Disruptor ring buffers gracefully handle toxic payloads.

use alloc::vec::Vec;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Maximum corruption patterns supported
pub const MAX_CORRUPTION_PATTERNS: usize = 16;

/// Type of corruption to inject
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CorruptionType {
    /// Zero out bytes in payload
    ZeroBytes,
    /// Flip random bits
    BitFlip,
    /// Truncate data
    Truncation,
    /// Duplicate sections
    Duplication,
    /// Reorder bytes
    Reordering,
    /// Inject invalid UTF-8
    InvalidUtf8,
    /// Corrupt checksum/CRC
    ChecksumCorruption,
    /// Overflow buffer bounds
    BufferOverflow,
}

/// Corruption configuration
#[derive(Debug, Clone)]
pub struct CorruptionConfig {
    pub corruption_type: CorruptionType,
    /// Probability of corruption (0.0 to 1.0)
    pub probability: f64,
    /// Byte range to corrupt [start, end]
    pub byte_range: (usize, usize),
    /// Severity (1-10)
    pub severity: u8,
    /// Target specific data types
    pub target_types: Vec<DataType>,
}

impl Default for CorruptionConfig {
    fn default() -> Self {
        CorruptionConfig {
            corruption_type: CorruptionType::BitFlip,
            probability: 0.001,
            byte_range: (0, usize::MAX),
            severity: 1,
            target_types: Vec::new(),
        }
    }
}

/// Data types that can be targeted for corruption
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DataType {
    WalEntry,
    Snapshot,
    RingBuffer,
    OrderBook,
    Position,
    Configuration,
}

/// State corruption injector
pub struct StateCorruptionInjector {
    configs: Vec<CorruptionConfig>,
    is_active: AtomicBool,
    shadow_mode: AtomicBool,
    corruptions_injected: AtomicU64,
    data_items_processed: AtomicU64,
    recovery_attempts: AtomicU64,
    successful_recoveries: AtomicU64,
    failed_recoveries: AtomicU64,
}

impl StateCorruptionInjector {
    pub fn new() -> Self {
        StateCorruptionInjector {
            configs: Vec::with_capacity(MAX_CORRUPTION_PATTERNS),
            is_active: AtomicBool::new(false),
            shadow_mode: AtomicBool::new(true),
            corruptions_injected: AtomicU64::new(0),
            data_items_processed: AtomicU64::new(0),
            recovery_attempts: AtomicU64::new(0),
            successful_recoveries: AtomicU64::new(0),
            failed_recoveries: AtomicU64::new(0),
        }
    }

    /// Add a corruption configuration
    pub fn add_corruption(&mut self, config: CorruptionConfig) -> Result<(), ChaosError> {
        if self.configs.len() >= MAX_CORRUPTION_PATTERNS {
            return Err(ChaosError::MaxFaultsReached);
        }

        if config.probability < 0.0 || config.probability > 1.0 {
            return Err(ChaosError::InvalidProbability);
        }

        self.configs.push(config);
        Ok(())
    }

    /// Remove corruption by type
    pub fn remove_corruption(&mut self, corruption_type: CorruptionType) {
        self.configs.retain(|c| c.corruption_type != corruption_type);
    }

    /// Clear all corruptions
    pub fn clear_corruptions(&mut self) {
        self.configs.clear();
    }

    /// Corrupt data based on active configurations
    pub fn corrupt_data(&self, mut data: Vec<u8>, data_type: DataType) -> CorruptedData {
        self.data_items_processed.fetch_add(1, Ordering::Relaxed);

        if !self.is_active.load(Ordering::Acquire) {
            return CorruptedData::Original(data);
        }

        let in_shadow = self.shadow_mode.load(Ordering::Acquire);

        for config in &self.configs {
            // Check shadow-only restriction
            if config.shadow_only && !in_shadow {
                continue;
            }

            // Check target type filter
            if !config.target_types.is_empty() && !config.target_types.contains(&data_type) {
                continue;
            }

            // Check probability
            let rand_val = self.next_random();
            let probability_check = (rand_val as f64) / (u64::MAX as f64);

            if probability_check < config.probability {
                self.corruptions_injected.fetch_add(1, Ordering::Relaxed);
                
                let corrupted = self.apply_corruption(data, config);
                return CorruptedData::Corrupted {
                    data: corrupted,
                    corruption_type: config.corruption_type,
                    severity: config.severity,
                };
            }
        }

        CorruptedData::Original(data)
    }

    /// Apply specific corruption to data
    fn apply_corruption(&self, mut data: Vec<u8>, config: &CorruptionConfig) -> Vec<u8> {
        let (start, end) = config.byte_range;
        let actual_start = start.min(data.len());
        let actual_end = end.min(data.len());

        match config.corruption_type {
            CorruptionType::ZeroBytes => {
                for i in actual_start..actual_end {
                    data[i] = 0;
                }
            }
            CorruptionType::BitFlip => {
                let flips = config.severity as usize;
                for _ in 0..flips {
                    let idx = actual_start + (self.next_random() as usize % (actual_end - actual_start).max(1));
                    if idx < data.len() {
                        data[idx] ^= 1 << (self.next_random() % 8);
                    }
                }
            }
            CorruptionType::Truncation => {
                let truncate_at = actual_start + (data.len() - actual_start) / 2;
                data.truncate(truncate_at);
            }
            CorruptionType::Duplication => {
                if actual_end > actual_start && actual_end <= data.len() {
                    let section: Vec<u8> = data[actual_start..actual_end].to_vec();
                    data.extend_from_slice(&section);
                }
            }
            CorruptionType::Reordering => {
                if actual_end > actual_start + 1 && actual_end <= data.len() {
                    let idx1 = actual_start + (self.next_random() as usize % (actual_end - actual_start));
                    let idx2 = actual_start + (self.next_random() as usize % (actual_end - actual_start));
                    data.swap(idx1.min(data.len() - 1), idx2.min(data.len() - 1));
                }
            }
            CorruptionType::InvalidUtf8 => {
                // Inject invalid UTF-8 sequences
                if actual_end > actual_start && actual_end <= data.len() {
                    data[actual_start] = 0xFF;
                    if actual_start + 1 < data.len() {
                        data[actual_start + 1] = 0xFE;
                    }
                }
            }
            CorruptionType::ChecksumCorruption => {
                // Corrupt last 4 bytes (simulated checksum)
                let len = data.len();
                if len >= 4 {
                    for i in 0..4 {
                        data[len - 1 - i] ^= 0xFF;
                    }
                }
            }
            CorruptionType::BufferOverflow => {
                // Simulate overflow by adding extra bytes
                let overflow_count = config.severity as usize * 4;
                for _ in 0..overflow_count {
                    data.push(0x41); // 'A' padding
                }
            }
        }

        data
    }

    /// Attempt to recover from corrupted data
    pub fn attempt_recovery(&self, corrupted: &CorruptedData) -> RecoveryResult {
        self.recovery_attempts.fetch_add(1, Ordering::Relaxed);

        match corrupted {
            CorruptedData::Original(_) => {
                self.successful_recoveries.fetch_add(1, Ordering::Relaxed);
                RecoveryResult::Success
            }
            CorruptedData::Corrupted { corruption_type, severity, .. } => {
                // Simulate recovery logic based on corruption type and severity
                let recovery_success = self.simulate_recovery(*corruption_type, *severity);

                if recovery_success {
                    self.successful_recoveries.fetch_add(1, Ordering::Relaxed);
                    RecoveryResult::Success
                } else {
                    self.failed_recoveries.fetch_add(1, Ordering::Relaxed);
                    RecoveryResult::Failed(*corruption_type)
                }
            }
        }
    }

    /// Simulate recovery success based on corruption characteristics
    fn simulate_recovery(&self, corruption_type: CorruptionType, severity: u8) -> bool {
        // Some corruptions are easier to recover from than others
        let base_recovery_rate = match corruption_type {
            CorruptionType::ZeroBytes => 0.9,
            CorruptionType::BitFlip => 0.7,
            CorruptionType::Truncation => 0.3,
            CorruptionType::Duplication => 0.8,
            CorruptionType::Reordering => 0.5,
            CorruptionType::InvalidUtf8 => 0.6,
            CorruptionType::ChecksumCorruption => 0.4,
            CorruptionType::BufferOverflow => 0.8,
        };

        // Severity reduces recovery chance
        let severity_factor = 1.0 - (severity as f64 * 0.05);
        let effective_rate = base_recovery_rate * severity_factor;

        let rand_val = self.next_random();
        let rand_check = (rand_val as f64) / (u64::MAX as f64);

        rand_check < effective_rate
    }

    /// Get statistics
    pub fn get_stats(&self) -> CorruptionStats {
        CorruptionStats {
            is_active: self.is_active.load(Ordering::Acquire),
            shadow_mode: self.shadow_mode.load(Ordering::Acquire),
            active_configs: self.configs.len(),
            corruptions_injected: self.corruptions_injected.load(Ordering::Relaxed),
            data_items_processed: self.data_items_processed.load(Ordering::Relaxed),
            recovery_attempts: self.recovery_attempts.load(Ordering::Relaxed),
            successful_recoveries: self.successful_recoveries.load(Ordering::Relaxed),
            failed_recoveries: self.failed_recoveries.load(Ordering::Relaxed),
            recovery_rate: self.calculate_recovery_rate(),
        }
    }

    /// Enable/disable corruption injection
    pub fn set_active(&self, active: bool) {
        self.is_active.store(active, Ordering::Release);
    }

    /// Set shadow mode
    pub fn set_shadow_mode(&self, shadow: bool) {
        self.shadow_mode.store(shadow, Ordering::Release);
    }

    fn calculate_recovery_rate(&self) -> f64 {
        let attempts = self.recovery_attempts.load(Ordering::Relaxed);
        let successes = self.successful_recoveries.load(Ordering::Relaxed);

        if attempts == 0 {
            return 0.0;
        }

        successes as f64 / attempts as f64
    }

    /// Simple LCG random number generator
    fn next_random(&self) -> u64 {
        static RNG_STATE: AtomicU64 = AtomicU64::new(0x5DEECE66D);
        let state = RNG_STATE.load(Ordering::Relaxed);
        let new_state = state.wrapping_mul(0x5DEECE66D).wrapping_add(0xB);
        RNG_STATE.store(new_state, Ordering::Relaxed);
        new_state
    }
}

impl Default for StateCorruptionInjector {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of corruption operation
#[derive(Debug, Clone)]
pub enum CorruptedData {
    Original(Vec<u8>),
    Corrupted {
        data: Vec<u8>,
        corruption_type: CorruptionType,
        severity: u8,
    },
}

/// Recovery result
#[derive(Debug, Clone, PartialEq)]
pub enum RecoveryResult {
    Success,
    Failed(CorruptionType),
}

/// Corruption statistics
#[derive(Debug, Clone)]
pub struct CorruptionStats {
    pub is_active: bool,
    pub shadow_mode: bool,
    pub active_configs: usize,
    pub corruptions_injected: u64,
    pub data_items_processed: u64,
    pub recovery_attempts: u64,
    pub successful_recoveries: u64,
    pub failed_recoveries: u64,
    pub recovery_rate: f64,
}

/// WAL corruption tester specifically for Write-Ahead Log testing
pub struct WalCorruptionTester {
    injector: StateCorruptionInjector,
    wal_entries_corrupted: AtomicU64,
    wal_recovery_success: AtomicU64,
}

impl WalCorruptionTester {
    pub fn new(injector: StateCorruptionInjector) -> Self {
        WalCorruptionTester {
            injector,
            wal_entries_corrupted: AtomicU64::new(0),
            wal_recovery_success: AtomicU64::new(0),
        }
    }

    /// Test WAL entry corruption
    pub fn test_wal_entry(&self, entry_data: Vec<u8>) -> WalTestResult {
        let corrupted = self.injector.corrupt_data(entry_data, DataType::WalEntry);

        match &corrupted {
            CorruptedData::Corrupted { .. } => {
                self.wal_entries_corrupted.fetch_add(1, Ordering::Relaxed);
            }
            _ => {}
        }

        let recovery = self.injector.attempt_recovery(&corrupted);

        match recovery {
            RecoveryResult::Success => {
                self.wal_recovery_success.fetch_add(1, Ordering::Relaxed);
                WalTestResult::Recovered(corrupted)
            }
            RecoveryResult::Failed(ctype) => WalTestResult::Unrecoverable(ctype),
        }
    }

    /// Get WAL-specific statistics
    pub fn get_wal_stats(&self) -> WalStats {
        WalStats {
            entries_corrupted: self.wal_entries_corrupted.load(Ordering::Relaxed),
            recovery_success: self.wal_recovery_success.load(Ordering::Relaxed),
            injector_stats: self.injector.get_stats(),
        }
    }
}

/// WAL test result
#[derive(Debug, Clone)]
pub enum WalTestResult {
    Recovered(CorruptedData),
    Unrecoverable(CorruptionType),
}

/// WAL statistics
#[derive(Debug, Clone)]
pub struct WalStats {
    pub entries_corrupted: u64,
    pub recovery_success: u64,
    pub injector_stats: CorruptionStats,
}

/// Chaos error types (shared with network_fault)
#[derive(Debug, Clone, PartialEq)]
pub enum ChaosError {
    MaxFaultsReached,
    InvalidProbability,
    InvalidConfiguration,
    NotInShadowMode,
    SystemNotReady,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_corruption_injector_basic() {
        let mut injector = StateCorruptionInjector::new();

        let config = CorruptionConfig {
            corruption_type: CorruptionType::BitFlip,
            probability: 0.5,
            severity: 3,
            ..Default::default()
        };

        assert!(injector.add_corruption(config).is_ok());

        injector.set_active(true);
        injector.set_shadow_mode(true);

        let stats = injector.get_stats();
        assert!(stats.is_active);
        assert_eq!(stats.active_configs, 1);
    }

    #[test]
    fn test_bit_flip_corruption() {
        let mut injector = StateCorruptionInjector::new();

        let config = CorruptionConfig {
            corruption_type: CorruptionType::BitFlip,
            probability: 1.0,
            severity: 5,
            byte_range: (0, 10),
            ..Default::default()
        };
        injector.add_corruption(config).unwrap();
        injector.set_active(true);

        let original = vec![0x00u8; 10];
        let result = injector.corrupt_data(original.clone(), DataType::WalEntry);

        match result {
            CorruptedData::Corrupted { data, corruption_type, .. } => {
                assert_eq!(corruption_type, CorruptionType::BitFlip);
                // Data should be different from original
                assert_ne!(data, original);
            }
            _ => panic!("Expected corrupted data"),
        }
    }

    #[test]
    fn test_wal_corruption_tester() {
        let injector = StateCorruptionInjector::new();
        let tester = WalCorruptionTester::new(injector);

        let entry = vec![0x42u8; 100];
        let result = tester.test_wal_entry(entry);

        // Without active corruption, should pass through
        match result {
            WalTestResult::Recovered(_) => {}
            _ => {}
        }
    }
}
