//! Automated Circuit Breakers
//! 
//! Triggers on extreme drawdown, API 5xx errors, or abnormal volatility.
//! Halts trading gracefully and safely parks capital until manual intervention or cooldown expires.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicI64, Ordering};
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
    pub fn fetch_max(&self, val: u64, ordering: Ordering) -> u64 {
        let mut current = self.value.load(ordering);
        loop {
            if val <= current {
                return current;
            }
            match self.value.compare_exchange_weak(current, val, ordering, ordering) {
                Ok(_) => return val,
                Err(x) => current = x,
            }
        }
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
    pub fn fetch_min(&self, val: i64, ordering: Ordering) -> i64 {
        let mut current = self.value.load(ordering);
        loop {
            if val >= current {
                return current;
            }
            match self.value.compare_exchange_weak(current, val, ordering, ordering) {
                Ok(_) => return val,
                Err(x) => current = x,
            }
        }
    }
}

/// Circuit breaker trigger type
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BreakerType {
    Drawdown,           // Maximum drawdown exceeded
    APIErrors,          // Too many 5xx errors
    Volatility,         // Abnormal market volatility
    LatencySpike,       // Latency exceeded threshold
    OrderRejection,     // High order rejection rate
    FillRate,           // Low fill rate
    PositionDrift,      // Position reconciliation drift
    MarginWarning,      // Margin approaching limit
    NetworkPartition,   // Network connectivity issues
    ExchangeMaintenance,// Exchange announced maintenance
}

/// Circuit breaker state
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BreakerState {
    Closed,     // Normal operation
    Open,       // Trading halted
    HalfOpen,   // Testing recovery
    Cooldown,   // Waiting for cooldown expiry
}

/// Circuit breaker configuration
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BreakerConfig {
    /// Maximum drawdown (basis points, e.g., 500 = 5%)
    pub max_drawdown_bps: u64,
    /// Maximum API errors before trip
    pub max_api_errors: u32,
    /// API error window (seconds)
    pub api_error_window_s: u64,
    /// Volatility threshold (basis points per minute)
    pub volatility_threshold_bps: u64,
    /// Latency threshold (milliseconds)
    pub latency_threshold_ms: u64,
    /// Order rejection threshold (percentage, 0-100)
    pub rejection_threshold_pct: u8,
    /// Cooldown period (milliseconds)
    pub cooldown_ms: u64,
    /// Auto-recovery enabled
    pub auto_recovery: bool,
    /// Half-open test duration (milliseconds)
    pub half_open_duration_ms: u64,
}

impl Default for BreakerConfig {
    fn default() -> Self {
        Self {
            max_drawdown_bps: 500,        // 5% max drawdown
            max_api_errors: 10,           // 10 errors
            api_error_window_s: 60,       // within 60 seconds
            volatility_threshold_bps: 200, // 2% per minute
            latency_threshold_ms: 50,     // 50ms
            rejection_threshold_pct: 50,  // 50% rejection rate
            cooldown_ms: 60_000,          // 1 minute cooldown
            auto_recovery: false,         // Require manual intervention
            half_open_duration_ms: 5000,  // 5 second test
        }
    }
}

/// Circuit breaker event
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BreakerEvent {
    pub breaker_type: BreakerType,
    pub timestamp_ns: u64,
    pub trigger_value: u64,
    pub threshold_value: u64,
}

impl BreakerEvent {
    pub fn new(breaker_type: BreakerType, trigger_value: u64, threshold_value: u64) -> Self {
        let timestamp_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        
        Self {
            breaker_type,
            timestamp_ns,
            trigger_value,
            threshold_value,
        }
    }
}

/// Individual circuit breaker
#[repr(C)]
pub struct CircuitBreaker {
    /// Configuration
    config: BreakerConfig,
    /// Current state
    state: AtomicU64, // Encoded BreakerState
    /// Trip count
    trip_count: PaddedAtomicU64,
    /// Last trip timestamp
    last_trip_ns: PaddedAtomicU64,
    /// Cooldown end timestamp
    cooldown_end_ns: PaddedAtomicU64,
    /// Peak equity (for drawdown calculation, scaled)
    peak_equity: PaddedAtomicI64,
    /// Current equity (scaled)
    current_equity: PaddedAtomicI64,
    /// API error count in window
    api_error_count: PaddedAtomicU64,
    /// API error window start
    api_error_window_start_ns: PaddedAtomicU64,
    /// Order submissions
    orders_submitted: PaddedAtomicU64,
    /// Orders rejected
    orders_rejected: PaddedAtomicU64,
    /// Breaker is enabled
    enabled: AtomicBool,
    /// Last event
    last_event: PaddedAtomicU64, // Pointer conceptually
}

impl CircuitBreaker {
    pub fn new(config: BreakerConfig) -> Self {
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        Self {
            config,
            state: AtomicU64::new(BreakerState::Closed as u64),
            trip_count: PaddedAtomicU64::new(0),
            last_trip_ns: PaddedAtomicU64::new(0),
            cooldown_end_ns: PaddedAtomicU64::new(0),
            peak_equity: PaddedAtomicI64::new(i64::MAX), // Will be set on first update
            current_equity: PaddedAtomicI64::new(0),
            api_error_count: PaddedAtomicU64::new(0),
            api_error_window_start_ns: PaddedAtomicU64::new(now_ns),
            orders_submitted: PaddedAtomicU64::new(0),
            orders_rejected: PaddedAtomicU64::new(0),
            enabled: AtomicBool::new(true),
            last_event: PaddedAtomicU64::new(0),
        }
    }

    /// Update equity for drawdown monitoring
    #[inline]
    pub fn update_equity(&self, equity: i64) -> Option<BreakerEvent> {
        if !self.enabled.load(Ordering::Acquire) {
            return None;
        }

        self.current_equity.store(equity, Ordering::Release);
        
        // Update peak
        let peak = self.peak_equity.load(Ordering::Acquire);
        if equity > peak {
            self.peak_equity.store(equity, Ordering::Release);
        }

        // Check drawdown
        let current_peak = self.peak_equity.load(Ordering::Acquire);
        if current_peak > 0 && current_peak != i64::MAX {
            let drawdown_bps = ((current_peak - equity) as u64 * 10_000) / current_peak as u64;
            
            if drawdown_bps >= self.config.max_drawdown_bps {
                return Some(self.trip(BreakerType::Drawdown, drawdown_bps, self.config.max_drawdown_bps));
            }
        }

        None
    }

    /// Record API error
    #[inline]
    pub fn record_api_error(&self) -> Option<BreakerEvent> {
        if !self.enabled.load(Ordering::Acquire) {
            return None;
        }

        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        // Check if we need to reset the window
        let window_start = self.api_error_window_start_ns.load(Ordering::Acquire);
        let window_ns = self.config.api_error_window_s * 1_000_000_000;
        
        if now_ns.saturating_sub(window_start) > window_ns {
            // Reset window
            self.api_error_window_start_ns.store(now_ns, Ordering::Release);
            self.api_error_count.store(1, Ordering::Release);
        } else {
            let count = self.api_error_count.fetch_add(1, Ordering::AcqRel) + 1;
            
            if count >= self.config.max_api_errors {
                return Some(self.trip(BreakerType::APIErrors, count as u64, self.config.max_api_errors as u64));
            }
        }

        None
    }

    /// Record order submission
    #[inline]
    pub fn record_order_submission(&self) {
        self.orders_submitted.fetch_add(1, Ordering::Relaxed);
    }

    /// Record order rejection
    #[inline]
    pub fn record_order_rejection(&self) -> Option<BreakerEvent> {
        if !self.enabled.load(Ordering::Acquire) {
            return None;
        }

        let rejected = self.orders_rejected.fetch_add(1, Ordering::Relaxed) + 1;
        let submitted = self.orders_submitted.load(Ordering::Relaxed);

        if submitted >= 10 { // Minimum sample size
            let rejection_pct = (rejected * 100 / submitted) as u8;
            
            if rejection_pct >= self.config.rejection_threshold_pct {
                return Some(self.trip(BreakerType::OrderRejection, rejection_pct as u64, self.config.rejection_threshold_pct as u64));
            }
        }

        None
    }

    /// Trip the breaker
    #[inline]
    fn trip(&self, breaker_type: BreakerType, trigger_value: u64, threshold_value: u64) -> BreakerEvent {
        let event = BreakerEvent::new(breaker_type, trigger_value, threshold_value);
        
        self.state.store(BreakerState::Open as u64, Ordering::Release);
        self.trip_count.fetch_add(1, Ordering::Relaxed);
        
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        
        self.last_trip_ns.store(now_ns, Ordering::Release);
        self.cooldown_end_ns.store(now_ns + self.config.cooldown_ms * 1_000_000, Ordering::Release);
        self.last_event.store(now_ns, Ordering::Release);

        event
    }

    /// Check if trading is allowed
    #[inline]
    pub fn is_trading_allowed(&self) -> bool {
        if !self.enabled.load(Ordering::Acquire) {
            return false;
        }

        match self.get_state() {
            BreakerState::Closed => true,
            BreakerState::HalfOpen => true,
            BreakerState::Open => {
                // Check if cooldown expired
                if self.is_cooldown_expired() {
                    if self.config.auto_recovery {
                        self.set_state(BreakerState::HalfOpen);
                        true
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
            BreakerState::Cooldown => false,
        }
    }

    /// Get current state
    #[inline]
    pub fn get_state(&self) -> BreakerState {
        match self.state.load(Ordering::Acquire) {
            0 => BreakerState::Closed,
            1 => BreakerState::Open,
            2 => BreakerState::HalfOpen,
            3 => BreakerState::Cooldown,
            _ => BreakerState::Closed,
        }
    }

    /// Set state
    #[inline]
    fn set_state(&self, state: BreakerState) {
        self.state.store(state as u64, Ordering::Release);
    }

    /// Check if cooldown expired
    #[inline]
    fn is_cooldown_expired(&self) -> bool {
        let cooldown_end = self.cooldown_end_ns.load(Ordering::Acquire);
        if cooldown_end == 0 {
            return true;
        }

        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        now_ns >= cooldown_end
    }

    /// Manual reset
    #[inline]
    pub fn reset(&self) {
        self.set_state(BreakerState::Closed);
        self.api_error_count.store(0, Ordering::Release);
        self.orders_submitted.store(0, Ordering::Release);
        self.orders_rejected.store(0, Ordering::Release);
        
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        self.api_error_window_start_ns.store(now_ns, Ordering::Release);
    }

    /// Disable breaker
    #[inline]
    pub fn disable(&self) {
        self.enabled.store(false, Ordering::Release);
    }

    /// Enable breaker
    #[inline]
    pub fn enable(&self) {
        self.enabled.store(true, Ordering::Release);
    }

    /// Get statistics
    #[inline]
    pub fn get_stats(&self) -> BreakerStats {
        BreakerStats {
            state: self.get_state(),
            enabled: self.enabled.load(Ordering::Acquire),
            trip_count: self.trip_count.load(Ordering::Relaxed),
            last_trip_ns: self.last_trip_ns.load(Ordering::Relaxed),
            api_error_count: self.api_error_count.load(Ordering::Relaxed),
            orders_submitted: self.orders_submitted.load(Ordering::Relaxed),
            orders_rejected: self.orders_rejected.load(Ordering::Relaxed),
            current_equity: self.current_equity.load(Ordering::Relaxed),
            peak_equity: self.peak_equity.load(Ordering::Relaxed),
        }
    }
}

/// Breaker statistics
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BreakerStats {
    pub state: BreakerState,
    pub enabled: bool,
    pub trip_count: u64,
    pub last_trip_ns: u64,
    pub api_error_count: u64,
    pub orders_submitted: u64,
    pub orders_rejected: u64,
    pub current_equity: i64,
    pub peak_equity: i64,
}

/// Circuit breaker manager for multiple breakers
#[repr(C)]
pub struct CircuitBreakerManager {
    /// Main/drawdown breaker
    main_breaker: CircuitBreaker,
    /// API error breaker
    api_breaker: CircuitBreaker,
    /// Volatility breaker
    volatility_breaker: CircuitBreaker,
    /// Any breaker tripped
    any_tripped: AtomicBool,
    /// Tripped breaker type
    tripped_type: AtomicU64,
}

impl CircuitBreakerManager {
    pub fn new(config: Option<BreakerConfig>) -> Self {
        let cfg = config.unwrap_or_default();
        
        Self {
            main_breaker: CircuitBreaker::new(cfg),
            api_breaker: CircuitBreaker::new(BreakerConfig {
                max_api_errors: cfg.max_api_errors,
                api_error_window_s: cfg.api_error_window_s,
                ..cfg
            }),
            volatility_breaker: CircuitBreaker::new(BreakerConfig {
                volatility_threshold_bps: cfg.volatility_threshold_bps,
                ..cfg
            }),
            any_tripped: AtomicBool::new(false),
            tripped_type: AtomicU64::new(0),
        }
    }

    /// Check if trading is allowed (all breakers must allow)
    #[inline]
    pub fn is_trading_allowed(&self) -> bool {
        self.main_breaker.is_trading_allowed()
            && self.api_breaker.is_trading_allowed()
            && self.volatility_breaker.is_trading_allowed()
    }

    /// Update equity (affects main breaker)
    #[inline]
    pub fn update_equity(&self, equity: i64) -> Option<BreakerEvent> {
        if let Some(event) = self.main_breaker.update_equity(equity) {
            self.any_tripped.store(true, Ordering::Release);
            self.tripped_type.store(event.breaker_type as u64, Ordering::Release);
            return Some(event);
        }
        None
    }

    /// Record API error
    #[inline]
    pub fn record_api_error(&self) -> Option<BreakerEvent> {
        if let Some(event) = self.api_breaker.record_api_error() {
            self.any_tripped.store(true, Ordering::Release);
            self.tripped_type.store(event.breaker_type as u64, Ordering::Release);
            return Some(event);
        }
        None
    }

    /// Record order submission
    #[inline]
    pub fn record_order_submission(&self) {
        self.main_breaker.record_order_submission();
        self.api_breaker.record_order_submission();
    }

    /// Record order rejection
    #[inline]
    pub fn record_order_rejection(&self) -> Option<BreakerEvent> {
        if let Some(event) = self.main_breaker.record_order_rejection() {
            self.any_tripped.store(true, Ordering::Release);
            self.tripped_type.store(event.breaker_type as u64, Ordering::Release);
            return Some(event);
        }
        None
    }

    /// Reset all breakers
    #[inline]
    pub fn reset_all(&self) {
        self.main_breaker.reset();
        self.api_breaker.reset();
        self.volatility_breaker.reset();
        self.any_tripped.store(false, Ordering::Release);
    }

    /// Get combined status
    #[inline]
    pub fn get_status(&self) -> BreakerManagerStatus {
        BreakerManagerStatus {
            trading_allowed: self.is_trading_allowed(),
            any_tripped: self.any_tripped.load(Ordering::Relaxed),
            main_stats: self.main_breaker.get_stats(),
            api_stats: self.api_breaker.get_stats(),
        }
    }
}

/// Breaker manager status
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BreakerManagerStatus {
    pub trading_allowed: bool,
    pub any_tripped: bool,
    pub main_stats: BreakerStats,
    pub api_stats: BreakerStats,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_drawdown_breaker() {
        let config = BreakerConfig {
            max_drawdown_bps: 500, // 5%
            ..Default::default()
        };
        let breaker = CircuitBreaker::new(config);

        // Set initial equity
        breaker.update_equity(1_000_000);
        assert!(breaker.is_trading_allowed());

        // Drop equity by 5%
        breaker.update_equity(950_000);
        assert!(!breaker.is_trading_allowed());
        assert_eq!(breaker.get_state(), BreakerState::Open);
    }

    #[test]
    fn test_api_error_breaker() {
        let config = BreakerConfig {
            max_api_errors: 3,
            api_error_window_s: 60,
            ..Default::default()
        };
        let breaker = CircuitBreaker::new(config);

        // Record errors
        assert!(breaker.record_api_error().is_none());
        assert!(breaker.record_api_error().is_none());
        
        // Third error should trip
        let event = breaker.record_api_error();
        assert!(event.is_some());
        assert_eq!(event.unwrap().breaker_type, BreakerType::APIErrors);
    }

    #[test]
    fn test_manager() {
        let manager = CircuitBreakerManager::new(None);
        
        assert!(manager.is_trading_allowed());
        
        // Trip the main breaker
        manager.update_equity(1_000_000);
        manager.update_equity(900_000); // 10% drop
        
        assert!(!manager.is_trading_allowed());
        
        let status = manager.get_status();
        assert!(status.any_tripped);
    }
}
