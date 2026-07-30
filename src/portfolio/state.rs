//! Lock-Free Portfolio State Tracker
//! 
//! Monitors global exposure, net delta, and gamma using concurrent per-symbol actors.
//! Aggregates local states into a global view without global mutex locks.

use std::sync::atomic::{AtomicI64, AtomicU64, AtomicBool, Ordering};
use std::sync::Arc;

/// Cache line size
const CACHE_LINE_SIZE: usize = 64;

/// Padded atomic u64
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

    #[inline]
    pub fn compare_exchange_weak(
        &self,
        current: i64,
        new: i64,
        success: Ordering,
        failure: Ordering,
    ) -> Result<i64, i64> {
        self.value.compare_exchange_weak(current, new, success, failure)
    }
}

/// Per-symbol risk state (managed by symbol actor)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SymbolRiskState {
    /// Symbol hash
    pub symbol_hash: u64,
    /// Net position (signed, in base units scaled by 1e8)
    pub net_position: i64,
    /// Notional value (in quote units scaled by 1e8)
    pub notional: u64,
    /// Delta exposure (scaled by 1e8)
    pub delta: i64,
    /// Gamma exposure (scaled by 1e12)
    pub gamma: i64,
    /// Unrealized PnL (scaled by 1e8)
    pub unrealized_pnl: i64,
    /// Last update timestamp (ns)
    pub last_update_ns: u64,
}

impl SymbolRiskState {
    pub fn new(symbol_hash: u64) -> Self {
        Self {
            symbol_hash,
            net_position: 0,
            notional: 0,
            delta: 0,
            gamma: 0,
            unrealized_pnl: 0,
            last_update_ns: 0,
        }
    }
}

/// Global portfolio state aggregator
#[repr(C)]
pub struct PortfolioState {
    /// Global net delta (sum of all symbol deltas)
    global_delta: PaddedAtomicI64,
    /// Global net gamma (sum of all symbol gammas)
    global_gamma: PaddedAtomicI64,
    /// Total notional exposure
    total_notional: PaddedAtomicU64,
    /// Total unrealized PnL
    total_unrealized_pnl: PaddedAtomicI64,
    /// Number of active symbols
    active_symbols: AtomicU64,
    /// Maximum delta limit (absolute value)
    max_delta_limit: PaddedAtomicI64,
    /// Maximum gamma limit (absolute value)
    max_gamma_limit: PaddedAtomicI64,
    /// Maximum notional limit
    max_notional_limit: PaddedAtomicU64,
    /// Limits exceeded flag
    limits_exceeded: AtomicBool,
    /// Last aggregation timestamp
    last_aggregation_ns: PaddedAtomicU64,
}

impl PortfolioState {
    pub fn new(
        max_delta: i64,
        max_gamma: i64,
        max_notional: u64,
    ) -> Self {
        Self {
            global_delta: PaddedAtomicI64::new(0),
            global_gamma: PaddedAtomicI64::new(0),
            total_notional: PaddedAtomicU64::new(0),
            total_unrealized_pnl: PaddedAtomicI64::new(0),
            active_symbols: AtomicU64::new(0),
            max_delta_limit: PaddedAtomicI64::new(max_delta),
            max_gamma_limit: PaddedAtomicI64::new(max_gamma),
            max_notional_limit: PaddedAtomicU64::new(max_notional),
            limits_exceeded: AtomicBool::new(false),
            last_aggregation_ns: PaddedAtomicU64::new(0),
        }
    }

    /// Aggregate a symbol's state into the global view
    #[inline]
    pub fn aggregate_symbol(&self, state: SymbolRiskState) {
        // Update global delta
        let old_delta = self.global_delta.fetch_add(state.delta, Ordering::AcqRel);
        
        // Update global gamma
        let old_gamma = self.global_gamma.fetch_add(state.gamma, Ordering::AcqRel);
        
        // Update total notional
        let old_notional = self.total_notional.fetch_add(state.notional, Ordering::AcqRel);
        
        // Update total unrealized PnL
        let old_pnl = self.total_unrealized_pnl.fetch_add(state.unrealized_pnl, Ordering::AcqRel);

        // Check limits after aggregation
        self.check_limits(
            old_delta + state.delta,
            old_gamma + state.gamma,
            old_notional + state.notional,
        );

        self.update_timestamp();
    }

    /// Remove a symbol's contribution (for cleanup)
    #[inline]
    pub fn remove_symbol(&self, state: SymbolRiskState) {
        self.global_delta.fetch_sub(state.delta, Ordering::AcqRel);
        self.global_gamma.fetch_sub(state.gamma, Ordering::AcqRel);
        self.total_notional.fetch_sub(state.notional, Ordering::AcqRel);
        self.total_unrealized_pnl.fetch_sub(state.unrealized_pnl, Ordering::AcqRel);
        self.active_symbols.fetch_sub(1, Ordering::AcqRel);
        
        self.check_limits(
            self.global_delta.load(Ordering::Acquire),
            self.global_gamma.load(Ordering::Acquire),
            self.total_notional.load(Ordering::Acquire),
        );
    }

    /// Check if limits are exceeded
    #[inline]
    fn check_limits(&self, delta: i64, gamma: i64, notional: u64) {
        let max_delta = self.max_delta_limit.load(Ordering::Acquire);
        let max_gamma = self.max_gamma_limit.load(Ordering::Acquire);
        let max_notional = self.max_notional_limit.load(Ordering::Acquire);

        let exceeded = delta.abs() > max_delta
            || gamma.abs() > max_gamma
            || notional > max_notional;

        self.limits_exceeded.store(exceeded, Ordering::Release);
    }

    /// Update timestamp
    #[inline]
    fn update_timestamp(&self) {
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        self.last_aggregation_ns.store(now_ns, Ordering::Release);
    }

    /// Increment active symbol count
    #[inline]
    pub fn add_symbol(&self) {
        self.active_symbols.fetch_add(1, Ordering::AcqRel);
    }

    /// Get global delta
    #[inline]
    pub fn get_global_delta(&self) -> i64 {
        self.global_delta.load(Ordering::Acquire)
    }

    /// Get global gamma
    #[inline]
    pub fn get_global_gamma(&self) -> i64 {
        self.global_gamma.load(Ordering::Acquire)
    }

    /// Get total notional
    #[inline]
    pub fn get_total_notional(&self) -> u64 {
        self.total_notional.load(Ordering::Acquire)
    }

    /// Get total unrealized PnL
    #[inline]
    pub fn get_total_unrealized_pnl(&self) -> i64 {
        self.total_unrealized_pnl.load(Ordering::Acquire)
    }

    /// Get active symbol count
    #[inline]
    pub fn get_active_symbols(&self) -> u64 {
        self.active_symbols.load(Ordering::Acquire)
    }

    /// Check if limits are exceeded
    #[inline]
    pub fn are_limits_exceeded(&self) -> bool {
        self.limits_exceeded.load(Ordering::Acquire)
    }

    /// Get last aggregation timestamp
    #[inline]
    pub fn get_last_aggregation_ns(&self) -> u64 {
        self.last_aggregation_ns.load(Ordering::Acquire)
    }

    /// Update delta limit
    #[inline]
    pub fn set_max_delta_limit(&self, limit: i64) {
        self.max_delta_limit.store(limit, Ordering::Release);
        self.check_limits(
            self.global_delta.load(Ordering::Acquire),
            self.global_gamma.load(Ordering::Acquire),
            self.total_notional.load(Ordering::Acquire),
        );
    }

    /// Update gamma limit
    #[inline]
    pub fn set_max_gamma_limit(&self, limit: i64) {
        self.max_gamma_limit.store(limit, Ordering::Release);
        self.check_limits(
            self.global_delta.load(Ordering::Acquire),
            self.global_gamma.load(Ordering::Acquire),
            self.total_notional.load(Ordering::Acquire),
        );
    }

    /// Update notional limit
    #[inline]
    pub fn set_max_notional_limit(&self, limit: u64) {
        self.max_notional_limit.store(limit, Ordering::Release);
        self.check_limits(
            self.global_delta.load(Ordering::Acquire),
            self.global_gamma.load(Ordering::Acquire),
            self.total_notional.load(Ordering::Acquire),
        );
    }

    /// Get current exposure summary
    #[inline]
    pub fn get_exposure_summary(&self) -> ExposureSummary {
        ExposureSummary {
            global_delta: self.get_global_delta(),
            global_gamma: self.get_global_gamma(),
            total_notional: self.get_total_notional(),
            total_unrealized_pnl: self.get_total_unrealized_pnl(),
            active_symbols: self.get_active_symbols(),
            limits_exceeded: self.are_limits_exceeded(),
            timestamp_ns: self.get_last_aggregation_ns(),
        }
    }
}

/// Exposure summary snapshot
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ExposureSummary {
    pub global_delta: i64,
    pub global_gamma: i64,
    pub total_notional: u64,
    pub total_unrealized_pnl: i64,
    pub active_symbols: u64,
    pub limits_exceeded: bool,
    pub timestamp_ns: u64,
}

/// Delta update event for cross-thread communication
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DeltaUpdateEvent {
    pub symbol_hash: u64,
    pub delta_change: i64,
    pub gamma_change: i64,
    pub notional_change: u64,
    pub pnl_change: i64,
    pub timestamp_ns: u64,
}

impl DeltaUpdateEvent {
    pub fn new(
        symbol_hash: u64,
        delta_change: i64,
        gamma_change: i64,
        notional_change: u64,
        pnl_change: i64,
    ) -> Self {
        let timestamp_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        Self {
            symbol_hash,
            delta_change,
            gamma_change,
            notional_change,
            pnl_change,
            timestamp_ns,
        }
    }

    pub fn from_symbol_state(old: SymbolRiskState, new: SymbolRiskState) -> Self {
        Self::new(
            new.symbol_hash,
            new.delta - old.delta,
            new.gamma - old.gamma,
            new.notional - old.notional,
            new.unrealized_pnl - old.unrealized_pnl,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_portfolio_state() {
        let portfolio = PortfolioState::new(
            1_000_000_000,  // max delta
            100_000_000,    // max gamma
            10_000_000_000, // max notional
        );

        assert!(!portfolio.are_limits_exceeded());
        assert_eq!(portfolio.get_global_delta(), 0);

        // Add a symbol
        let state = SymbolRiskState {
            symbol_hash: 12345,
            net_position: 100_000_000,
            notional: 5_000_000_000,
            delta: 500_000_000,
            gamma: 50_000_000,
            unrealized_pnl: 10_000_000,
            last_update_ns: 0,
        };

        portfolio.aggregate_symbol(state);
        
        assert_eq!(portfolio.get_global_delta(), 500_000_000);
        assert_eq!(portfolio.get_global_gamma(), 50_000_000);
        assert_eq!(portfolio.get_total_notional(), 5_000_000_000);
        assert!(!portfolio.are_limits_exceeded());
    }

    #[test]
    fn test_limits_exceeded() {
        let portfolio = PortfolioState::new(
            100_000_000,    // max delta (small)
            10_000_000,     // max gamma
            1_000_000_000,  // max notional
        );

        let state = SymbolRiskState {
            symbol_hash: 12345,
            net_position: 100_000_000,
            notional: 500_000_000,
            delta: 200_000_000, // Exceeds limit
            gamma: 5_000_000,
            unrealized_pnl: 0,
            last_update_ns: 0,
        };

        portfolio.aggregate_symbol(state);
        assert!(portfolio.are_limits_exceeded());
    }
}
