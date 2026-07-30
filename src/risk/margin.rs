//! Real-Time Atomic Margin Calculator
//! 
//! Tracks cross/isolated futures and spot balances with nanosecond updates.
//! Uses lock-free atomics to avoid blocking the execution thread.

use std::sync::atomic::{AtomicU64, AtomicI64, AtomicBool, Ordering};
use std::collections::HashMap;

/// Cache line size for x86_64
const CACHE_LINE_SIZE: usize = 64;

/// Padded atomic u64 for cache-line alignment
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
    pub fn store(&self, val: u64, ordering: Ordering) {
        self.value.store(val, ordering);
    }

    #[inline]
    pub fn fetch_add(&self, val: u64, ordering: Ordering) -> u64 {
        self.value.fetch_add(val, ordering)
    }

    #[inline]
    pub fn fetch_sub(&self, val: u64, ordering: Ordering) -> u64 {
        self.value.fetch_sub(val, ordering)
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

/// Padded atomic i64
#[repr(C)]
#[derive(Debug)]
pub struct PaddedAtomicI64 {
    _pad1: [u8; CACHE_LINE_SIZE - 8],
    value: AtomicI64,
    _pad2: [u8; CACHE_LINE_SIZE],
}

impl PaddedAtomicI64 {
    pub fn new(initial: i64) -> Self {
        Self {
            _pad1: [0u8; CACHE_LINE_SIZE - 8],
            value: AtomicI64::new(initial),
            _pad2: [0u8; CACHE_LINE_SIZE],
        }
    }

    #[inline]
    pub fn load(&self, ordering: Ordering) -> i64 {
        self.value.load(ordering)
    }

    #[inline]
    pub fn store(&self, val: i64, ordering: Ordering) {
        self.value.store(val, ordering);
    }

    #[inline]
    pub fn fetch_add(&self, val: i64, ordering: Ordering) -> i64 {
        self.value.fetch_add(val, ordering)
    }

    #[inline]
    pub fn fetch_sub(&self, val: i64, ordering: Ordering) -> i64 {
        self.value.fetch_sub(val, ordering)
    }
}

/// Margin mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub enum MarginMode {
    Cross,
    Isolated,
}

/// Balance type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub enum BalanceType {
    Spot,
    FuturesCross,
    FuturesIsolated,
}

/// Per-symbol margin state
#[repr(C)]
#[derive(Debug)]
pub struct SymbolMarginState {
    /// Symbol identifier (hashed)
    pub symbol_hash: u64,
    /// Margin mode
    pub mode: MarginMode,
    /// Available balance (scaled by 1e8)
    pub available: PaddedAtomicU64,
    /// Used margin (scaled by 1e8)
    pub used_margin: PaddedAtomicU64,
    /// Unrealized PnL (scaled by 1e8, can be negative)
    pub unrealized_pnl: PaddedAtomicI64,
    /// Position size (signed)
    pub position: PaddedAtomicI64,
    /// Entry price (scaled by 1e8)
    pub entry_price: PaddedAtomicU64,
    /// Liquidation price (scaled by 1e8)
    pub liquidation_price: PaddedAtomicU64,
    /// Leverage (1x = 1, 100x = 100)
    pub leverage: AtomicU64,
}

impl SymbolMarginState {
    pub fn new(symbol_hash: u64, mode: MarginMode, initial_balance: u64) -> Self {
        Self {
            symbol_hash,
            mode,
            available: PaddedAtomicU64::new(initial_balance),
            used_margin: PaddedAtomicU64::new(0),
            unrealized_pnl: PaddedAtomicI64::new(0),
            position: PaddedAtomicI64::new(0),
            entry_price: PaddedAtomicU64::new(0),
            liquidation_price: PaddedAtomicU64::new(0),
            leverage: AtomicU64::new(1),
        }
    }

    /// Update available balance atomically
    #[inline]
    pub fn update_available(&self, delta: i64) {
        if delta >= 0 {
            self.available.fetch_add(delta as u64, Ordering::AcqRel);
        } else {
            self.available.fetch_sub((-delta) as u64, Ordering::AcqRel);
        }
    }

    /// Update used margin atomically
    #[inline]
    pub fn update_used_margin(&self, delta: i64) {
        if delta >= 0 {
            self.used_margin.fetch_add(delta as u64, Ordering::AcqRel);
        } else {
            self.used_margin.fetch_sub((-delta) as u64, Ordering::AcqRel);
        }
    }

    /// Update unrealized PnL
    #[inline]
    pub fn update_unrealized_pnl(&self, pnl: i64) {
        self.unrealized_pnl.store(pnl, Ordering::Release);
    }

    /// Get total equity (available + unrealized PnL)
    #[inline]
    pub fn get_equity(&self) -> i64 {
        let available = self.available.load(Ordering::Acquire) as i64;
        let unrealized = self.unrealized_pnl.load(Ordering::Acquire);
        available + unrealized
    }

    /// Get margin ratio (used / equity * 10000 for basis points)
    #[inline]
    pub fn get_margin_ratio_bps(&self) -> u64 {
        let equity = self.get_equity();
        if equity <= 0 {
            return u64::MAX;
        }
        let used = self.used_margin.load(Ordering::Acquire);
        ((used as u128 * 10_000) / equity as u128) as u64
    }

    /// Check if margin is sufficient for additional order
    #[inline]
    pub fn can_open_position(&self, required_margin: u64) -> bool {
        let available = self.available.load(Ordering::Acquire);
        available >= required_margin
    }

    /// Set leverage
    #[inline]
    pub fn set_leverage(&self, leverage: u64) {
        self.leverage.store(leverage, Ordering::Release);
    }

    /// Get leverage
    #[inline]
    pub fn get_leverage(&self) -> u64 {
        self.leverage.load(Ordering::Acquire)
    }
}

/// Global margin calculator state
#[repr(C)]
pub struct MarginCalculator {
    /// Total account equity across all modes
    total_equity: PaddedAtomicI64,
    /// Total used margin
    total_used_margin: PaddedAtomicU64,
    /// Account health flag (false if margin call)
    is_healthy: AtomicBool,
    /// Margin call threshold (basis points, e.g., 8000 = 80%)
    margin_call_threshold_bps: AtomicU64,
    /// Number of active symbols
    active_symbol_count: AtomicU64,
    /// Last update timestamp (nanoseconds since epoch)
    last_update_ns: PaddedAtomicU64,
}

impl MarginCalculator {
    pub fn new(initial_equity: i64, margin_call_threshold_bps: u64) -> Self {
        Self {
            total_equity: PaddedAtomicI64::new(initial_equity),
            total_used_margin: PaddedAtomicU64::new(0),
            is_healthy: AtomicBool::new(true),
            margin_call_threshold_bps: AtomicU64::new(margin_call_threshold_bps),
            active_symbol_count: AtomicU64::new(0),
            last_update_ns: PaddedAtomicU64::new(0),
        }
    }

    /// Update total equity atomically
    #[inline]
    pub fn update_total_equity(&self, delta: i64) {
        let new_equity = if delta >= 0 {
            self.total_equity.fetch_add(delta, Ordering::AcqRel) + delta
        } else {
            self.total_equity.fetch_sub(-delta, Ordering::AcqRel) - delta
        };
        
        // Check health
        self.check_health(new_equity);
        self.update_timestamp();
    }

    /// Update total used margin
    #[inline]
    pub fn update_total_used_margin(&self, delta: i64) {
        if delta >= 0 {
            self.total_used_margin.fetch_add(delta as u64, Ordering::AcqRel);
        } else {
            self.total_used_margin.fetch_sub((-delta) as u64, Ordering::AcqRel);
        }
        self.check_health(self.total_equity.load(Ordering::Acquire));
        self.update_timestamp();
    }

    /// Check account health based on margin ratio
    #[inline]
    fn check_health(&self, equity: i64) {
        if equity <= 0 {
            self.is_healthy.store(false, Ordering::Release);
            return;
        }

        let used = self.total_used_margin.load(Ordering::Acquire);
        let threshold = self.margin_call_threshold_bps.load(Ordering::Acquire);
        let ratio_bps = ((used as u128 * 10_000) / equity as u128) as u64;

        if ratio_bps >= threshold {
            self.is_healthy.store(false, Ordering::Release);
        } else {
            self.is_healthy.store(true, Ordering::Release);
        }
    }

    /// Update timestamp
    #[inline]
    fn update_timestamp(&self) {
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        self.last_update_ns.store(now_ns, Ordering::Release);
    }

    /// Get current timestamp
    #[inline]
    pub fn get_last_update_ns(&self) -> u64 {
        self.last_update_ns.load(Ordering::Acquire)
    }

    /// Check if account is healthy
    #[inline]
    pub fn is_healthy(&self) -> bool {
        self.is_healthy.load(Ordering::Acquire)
    }

    /// Get total equity
    #[inline]
    pub fn get_total_equity(&self) -> i64 {
        self.total_equity.load(Ordering::Acquire)
    }

    /// Get total used margin
    #[inline]
    pub fn get_total_used_margin(&self) -> u64 {
        self.total_used_margin.load(Ordering::Acquire)
    }

    /// Get available margin (equity - used)
    #[inline]
    pub fn get_available_margin(&self) -> i64 {
        let equity = self.total_equity.load(Ordering::Acquire);
        let used = self.total_used_margin.load(Ordering::Acquire) as i64;
        equity - used
    }

    /// Get margin ratio in basis points
    #[inline]
    pub fn get_margin_ratio_bps(&self) -> u64 {
        let equity = self.total_equity.load(Ordering::Acquire);
        if equity <= 0 {
            return u64::MAX;
        }
        let used = self.total_used_margin.load(Ordering::Acquire);
        ((used as u128 * 10_000) / equity as u128) as u64
    }

    /// Set margin call threshold
    #[inline]
    pub fn set_margin_call_threshold(&self, threshold_bps: u64) {
        self.margin_call_threshold_bps.store(threshold_bps, Ordering::Release);
    }

    /// Force health status (for kill switch integration)
    #[inline]
    pub fn force_unhealthy(&self) {
        self.is_healthy.store(false, Ordering::Release);
    }

    /// Reset health status
    #[inline]
    pub fn reset_health(&self) {
        let equity = self.total_equity.load(Ordering::Acquire);
        self.check_health(equity);
    }
}

/// Margin update event for cross-thread communication
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MarginUpdateEvent {
    pub symbol_hash: u64,
    pub balance_type: BalanceType,
    pub delta: i64,
    pub timestamp_ns: u64,
}

impl MarginUpdateEvent {
    pub fn new(symbol_hash: u64, balance_type: BalanceType, delta: i64) -> Self {
        let timestamp_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        Self {
            symbol_hash,
            balance_type,
            delta,
            timestamp_ns,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_margin_calculator() {
        let calc = MarginCalculator::new(1_000_000_000, 8000); // $10k initial, 80% threshold
        
        assert!(calc.is_healthy());
        assert_eq!(calc.get_total_equity(), 1_000_000_000);
        assert_eq!(calc.get_available_margin(), 1_000_000_000);

        // Use some margin
        calc.update_total_used_margin(500_000_000);
        assert!(calc.is_healthy());
        assert_eq!(calc.get_available_margin(), 500_000_000);

        // Use more margin (should trigger margin call at 80%)
        calc.update_total_used_margin(300_000_000);
        assert!(!calc.is_healthy()); // 80% used
    }

    #[test]
    fn test_symbol_margin_state() {
        let state = SymbolMarginState::new(12345, MarginMode::Cross, 100_000_000);
        
        assert!(state.can_open_position(50_000_000));
        assert!(!state.can_open_position(150_000_000));

        state.update_available(-10_000_000);
        state.update_used_margin(10_000_000);
        
        assert_eq!(state.get_equity(), 100_000_000); // Available + unrealized (0)
    }
}
