//! Pre-Trade Risk Validation Engine
//! 
//! Implements lock-free fat-finger checks, position limits, and idempotent order ID generation.
//! Uses cache-line padding to prevent false sharing on AMD Ryzen CPUs.

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// Cache line size for x86_64 (AMD Ryzen)
const CACHE_LINE_SIZE: usize = 64;

/// Padded atomic for cache-line alignment
#[repr(C)]
#[derive(Debug)]
pub struct PaddedAtomicU64 {
    _pad1: [u8; CACHE_LINE_SIZE - 8],
    value: AtomicU64,
    _pad2: [u8; CACHE_LINE_SIZE],
}

impl PaddedAtomicU64 {
    pub fn new(initial: u64) -> Self {
        Self {
            _pad1: [0u8; CACHE_LINE_SIZE - 8],
            value: AtomicU64::new(initial),
            _pad2: [0u8; CACHE_LINE_SIZE],
        }
    }

    #[inline]
    pub fn load(&self, ordering: Ordering) -> u64 {
        self.value.load(ordering)
    }

    #[inline]
    pub fn fetch_add(&self, val: u64, ordering: Ordering) -> u64 {
        self.value.fetch_add(val, ordering)
    }

    #[inline]
    pub fn compare_exchange_weak(
        &self,
        current: u64,
        new: u64,
        success: Ordering,
        failure: Ordering,
    ) -> Result<u64, u64> {
        self.value.compare_exchange_weak(current, new, success, failure)
    }
}

/// Order ID generator with idempotency tracking
#[repr(C)]
#[derive(Debug)]
pub struct IdempotentOrderIdGenerator {
    /// Monotonically increasing counter
    counter: PaddedAtomicU64,
    /// Last generated ID for deduplication
    last_id: PaddedAtomicU64,
    /// Epoch timestamp for uniqueness across restarts
    epoch_base: u64,
}

impl IdempotentOrderIdGenerator {
    pub fn new(epoch_base: u64) -> Self {
        Self {
            counter: PaddedAtomicU64::new(0),
            last_id: PaddedAtomicU64::new(0),
            epoch_base,
        }
    }

    /// Generate a unique, idempotent order ID
    /// Format: [epoch_bits(32)][counter_bits(32)]
    #[inline]
    pub fn generate_order_id(&self) -> u64 {
        let counter = self.counter.fetch_add(1, Ordering::AcqRel);
        let order_id = (self.epoch_base << 32) | (counter & 0xFFFFFFFF);
        
        // Idempotency check: ensure monotonicity
        let mut last = self.last_id.load(Ordering::Acquire);
        loop {
            if order_id > last {
                match self.last_id.compare_exchange_weak(
                    last,
                    order_id,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                ) {
                    Ok(_) => return order_id,
                    Err(new_last) => last = new_last,
                }
            } else {
                // Counter wrapped or duplicate detected, increment
                let new_counter = self.counter.fetch_add(1, Ordering::AcqRel);
                let new_order_id = (self.epoch_base << 32) | (new_counter & 0xFFFFFFFF);
                last = self.last_id.load(Ordering::Acquire);
                if new_order_id > last {
                    match self.last_id.compare_exchange_weak(
                        last,
                        new_order_id,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    ) {
                        Ok(_) => return new_order_id,
                        Err(new_last) => last = new_last,
                    }
                }
            }
        }
    }

    /// Check if an order ID has already been issued
    #[inline]
    pub fn is_duplicate(&self, order_id: u64) -> bool {
        let last = self.last_id.load(Ordering::Acquire);
        order_id <= last
    }
}

/// Fat-finger validation result
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub enum FatFingerResult {
    Ok,
    PriceTooHigh(u64),
    PriceTooLow(u64),
    SizeTooLarge(u64),
    SizeTooSmall(u64),
    NotionalExceeded(u128),
}

/// Pre-trade risk parameters
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PreTradeRiskParams {
    /// Maximum order size in base units
    pub max_order_size: u64,
    /// Minimum order size in base units
    pub min_order_size: u64,
    /// Maximum price deviation from mid (basis points)
    pub max_price_deviation_bps: u64,
    /// Maximum notional value per order (in quote units * 10^8)
    pub max_notional: u128,
    /// Maximum position size (absolute value)
    pub max_position: i64,
}

impl Default for PreTradeRiskParams {
    fn default() -> Self {
        Self {
            max_order_size: 1_000_000_000, // 10 BTC in satoshis equivalent
            min_order_size: 1_000,         // Minimum dust
            max_price_deviation_bps: 500,  // 5% max deviation
            max_notional: 10_000_000_000,  // $10k max per order (scaled)
            max_position: 10_000_000_000,  // 100 BTC max position
        }
    }
}

/// Lock-free pre-trade risk validator
#[repr(C)]
pub struct PreTradeRiskValidator {
    params: PreTradeRiskParams,
    order_id_gen: IdempotentOrderIdGenerator,
    /// Current position (atomic for lock-free access)
    current_position: PaddedAtomicI64,
    /// Active flag
    is_active: AtomicBool,
    /// Rejection counter for monitoring
    rejection_count: PaddedAtomicU64,
}

/// Padded atomic i64
#[repr(C)]
#[derive(Debug)]
struct PaddedAtomicI64 {
    _pad1: [u8; CACHE_LINE_SIZE - 8],
    value: AtomicI64,
    _pad2: [u8; CACHE_LINE_SIZE],
}

impl PaddedAtomicI64 {
    fn new(initial: i64) -> Self {
        Self {
            _pad1: [0u8; CACHE_LINE_SIZE - 8],
            value: AtomicI64::new(initial),
            _pad2: [0u8; CACHE_LINE_SIZE],
        }
    }

    #[inline]
    fn load(&self, ordering: Ordering) -> i64 {
        self.value.load(ordering)
    }

    #[inline]
    fn fetch_add(&self, val: i64, ordering: Ordering) -> i64 {
        self.value.fetch_add(val, ordering)
    }
}

impl PreTradeRiskValidator {
    pub fn new(epoch_base: u64, params: PreTradeRiskParams) -> Self {
        Self {
            params,
            order_id_gen: IdempotentOrderIdGenerator::new(epoch_base),
            current_position: PaddedAtomicI64::new(0),
            is_active: AtomicBool::new(true),
            rejection_count: PaddedAtomicU64::new(0),
        }
    }

    /// Validate an order before submission
    #[inline]
    pub fn validate_order(
        &self,
        side: bool, // true = buy, false = sell
        price: u64,
        size: u64,
        mid_price: u64,
    ) -> FatFingerResult {
        if !self.is_active.load(Ordering::Acquire) {
            self.rejection_count.fetch_add(1, Ordering::Relaxed);
            return FatFingerResult::PriceTooHigh(0); // System halted
        }

        // Size checks
        if size < self.params.min_order_size {
            self.rejection_count.fetch_add(1, Ordering::Relaxed);
            return FatFingerResult::SizeTooSmall(size);
        }
        if size > self.params.max_order_size {
            self.rejection_count.fetch_add(1, Ordering::Relaxed);
            return FatFingerResult::SizeTooLarge(size);
        }

        // Price deviation check
        let max_deviation = (mid_price as u128 * self.params.max_price_deviation_bps as u128 / 10_000) as u64;
        let price_floor = mid_price.saturating_sub(max_deviation);
        let price_ceiling = mid_price.saturating_add(max_deviation);

        if price < price_floor {
            self.rejection_count.fetch_add(1, Ordering::Relaxed);
            return FatFingerResult::PriceTooLow(price);
        }
        if price > price_ceiling {
            self.rejection_count.fetch_add(1, Ordering::Relaxed);
            return FatFingerResult::PriceTooHigh(price);
        }

        // Notional check (price * size, scaled)
        let notional = (price as u128).checked_mul(size as u128).unwrap_or(u128::MAX);
        if notional > self.params.max_notional {
            self.rejection_count.fetch_add(1, Ordering::Relaxed);
            return FatFingerResult::NotionalExceeded(notional);
        }

        // Position limit check
        let current_pos = self.current_position.load(Ordering::Acquire);
        let new_position = if side {
            current_pos.saturating_add(size as i64)
        } else {
            current_pos.saturating_sub(size as i64)
        };

        if new_position.abs() > self.params.max_position {
            self.rejection_count.fetch_add(1, Ordering::Relaxed);
            return FatFingerResult::NotionalExceeded(new_position.unsigned_abs() as u128);
        }

        FatFingerResult::Ok
    }

    /// Update position after execution (call only after confirmed fill)
    #[inline]
    pub fn update_position(&self, side: bool, size: u64) {
        let delta = if side { size as i64 } else { -(size as i64) };
        self.current_position.fetch_add(delta, Ordering::AcqRel);
    }

    /// Get current position
    #[inline]
    pub fn get_position(&self) -> i64 {
        self.current_position.load(Ordering::Acquire)
    }

    /// Generate idempotent order ID
    #[inline]
    pub fn generate_order_id(&self) -> u64 {
        self.order_id_gen.generate_order_id()
    }

    /// Check for duplicate order ID
    #[inline]
    pub fn is_duplicate_order_id(&self, order_id: u64) -> bool {
        self.order_id_gen.is_duplicate(order_id)
    }

    /// Activate the validator
    #[inline]
    pub fn activate(&self) {
        self.is_active.store(true, Ordering::Release);
    }

    /// Deactivate the validator (kill switch integration)
    #[inline]
    pub fn deactivate(&self) {
        self.is_active.store(false, Ordering::Release);
    }

    /// Get rejection count for monitoring
    #[inline]
    pub fn get_rejection_count(&self) -> u64 {
        self.rejection_count.load(Ordering::Relaxed)
    }

    /// Get risk parameters
    #[inline]
    pub fn get_params(&self) -> PreTradeRiskParams {
        self.params
    }

    /// Update risk parameters (careful: not atomic, use during initialization only)
    pub fn update_params(&mut self, params: PreTradeRiskParams) {
        self.params = params;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_order_id_generation() {
        let gen = IdempotentOrderIdGenerator::new(12345);
        let id1 = gen.generate_order_id();
        let id2 = gen.generate_order_id();
        assert!(id2 > id1);
        assert!(!gen.is_duplicate(id2));
        assert!(gen.is_duplicate(id1));
    }

    #[test]
    fn test_fat_finger_validation() {
        let params = PreTradeRiskParams::default();
        let validator = PreTradeRiskValidator::new(12345, params);
        
        let mid_price = 50_000_000; // $50k in scaled units
        let result = validator.validate_order(true, 50_000_000, 1_000_000, mid_price);
        assert_eq!(result, FatFingerResult::Ok);

        // Test size too small
        let result = validator.validate_order(true, 50_000_000, 100, mid_price);
        assert!(matches!(result, FatFingerResult::SizeTooSmall(_)));

        // Test price too high
        let result = validator.validate_order(true, 100_000_000, 1_000_000, mid_price);
        assert!(matches!(result, FatFingerResult::PriceTooHigh(_)));
    }
}
