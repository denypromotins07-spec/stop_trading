//! Systemic Risk Module Root
//! 
//! Wires contagion and evaporation metrics to automated circuit breakers.

pub mod contagion;
pub mod liquidity_evap;

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

use self::contagion::{ContagionDetector, ContagionSignal, ContagionLevel, ContagionDetectorBuilder};
use self::liquidity_evap::{LiquidityEvaporationDetector, LiquiditySignal, LiquidityLevel, LiquidityDetectorBuilder};

/// Circuit breaker state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitBreakerState {
    Closed = 0,      // Normal operation
    Open = 1,        // Trading halted
    HalfOpen = 2,    // Testing resumption
}

/// Combined systemic risk signal
#[derive(Debug, Clone, Copy)]
pub struct SystemicRiskSignal {
    /// Overall risk level (0-3)
    pub risk_level: u8,
    /// Contagion component
    pub contagion_signal: ContagionSignal,
    /// Liquidity component
    pub liquidity_signal: LiquiditySignal,
    /// Whether circuit breaker is triggered
    pub circuit_breaker_triggered: bool,
    /// Recommended action: 0=none, 1=reduce, 2=halt
    pub recommended_action: u8,
    /// Timestamp in nanoseconds
    pub timestamp_ns: u64,
}

impl Default for SystemicRiskSignal {
    fn default() -> Self {
        Self {
            risk_level: 0,
            contagion_signal: ContagionSignal::default(),
            liquidity_signal: LiquiditySignal::default(),
            circuit_breaker_triggered: false,
            recommended_action: 0,
            timestamp_ns: 0,
        }
    }
}

/// Cache-line aligned systemic risk monitor
#[repr(align(64))]
pub struct SystemicRiskMonitor {
    /// Contagion detector
    contagion_detector: ContagionDetector,
    /// Liquidity evaporation detector
    liquidity_detector: LiquidityEvaporationDetector,
    /// Circuit breaker state
    circuit_breaker_state: CircuitBreakerState,
    /// Auto-halt enabled
    auto_halt_enabled: AtomicBool,
    /// Total signals processed
    signals_processed: AtomicU64,
    /// Circuit breaker trips count
    breaker_trips: AtomicU64,
    _pad: [u8; 32],
}

unsafe impl Send for SystemicRiskMonitor {}
unsafe impl Sync for SystemicRiskMonitor {}

impl SystemicRiskMonitor {
    /// Create new systemic risk monitor
    pub fn new() -> Self {
        Self {
            contagion_detector: ContagionDetectorBuilder::new().build(),
            liquidity_detector: LiquidityDetectorBuilder::new().build(),
            circuit_breaker_state: CircuitBreakerState::Closed,
            auto_halt_enabled: AtomicBool::new(true),
            signals_processed: AtomicU64::new(0),
            breaker_trips: AtomicU64::new(0),
            _pad: [0; 32],
        }
    }

    /// Create with custom thresholds
    pub fn with_thresholds(
        contagion_warning: f64,
        contagion_critical: f64,
        liquidity_reduced: f64,
        liquidity_stressed: f64,
        liquidity_evaporated: f64,
    ) -> Self {
        Self {
            contagion_detector: ContagionDetectorBuilder::new()
                .warning_threshold(contagion_warning)
                .critical_threshold(contagion_critical)
                .build(),
            liquidity_detector: LiquidityDetectorBuilder::new()
                .thresholds(liquidity_reduced, liquidity_stressed, liquidity_evaporated)
                .build(),
            circuit_breaker_state: CircuitBreakerState::Closed,
            auto_halt_enabled: AtomicBool::new(true),
            signals_processed: AtomicU64::new(0),
            breaker_trips: AtomicU64::new(0),
            _pad: [0; 32],
        }
    }

    /// Register asset for contagion tracking
    pub fn register_asset(&mut self, asset_id: u64, base_correlation: f64) -> bool {
        self.contagion_detector.register_asset(asset_id, base_correlation)
    }

    /// Update asset correlation
    pub fn update_correlation(&mut self, asset_id: u64, correlation: f64) {
        let timestamp_ns = Instant::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos() as u64;
        self.contagion_detector.update_correlation(asset_id, correlation, timestamp_ns);
    }

    /// Register venue for liquidity tracking
    pub fn register_venue(&mut self, venue_id: u64, normal_depth: u64) -> bool {
        self.liquidity_detector.register_venue(venue_id, normal_depth)
    }

    /// Update venue liquidity
    pub fn update_venue_liquidity(&mut self, venue_id: u64, depth: u64) {
        self.liquidity_detector.update_venue_liquidity(venue_id, depth);
    }

    /// Enable/disable auto-halt
    #[inline]
    pub fn set_auto_halt(&self, enabled: bool) {
        self.auto_halt_enabled.store(enabled, Ordering::Relaxed);
    }

    /// Check if auto-halt is enabled
    #[inline]
    pub fn is_auto_halt_enabled(&self) -> bool {
        self.auto_halt_enabled.load(Ordering::Relaxed)
    }

    /// Evaluate systemic risk and generate signal
    pub fn evaluate(&mut self) -> SystemicRiskSignal {
        let timestamp_ns = Instant::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos() as u64;

        self.signals_processed.fetch_add(1, Ordering::Relaxed);

        // Get contagion signal
        let contagion = self.contagion_detector.detect(timestamp_ns);

        // Get liquidity signal
        let liquidity = self.liquidity_detector.detect();

        // Calculate combined risk level (0-3)
        let contagion_risk = match contagion.level {
            ContagionLevel::None => 0,
            ContagionLevel::Watch => 1,
            ContagionLevel::Warning => 2,
            ContagionLevel::Critical => 3,
        };

        let liquidity_risk = match liquidity.level {
            LiquidityLevel::Normal => 0,
            LiquidityLevel::Reduced => 1,
            LiquidityLevel::Stressed => 2,
            LiquidityLevel::Evaporated => 3,
        };

        // Take maximum of both risks
        let risk_level = contagion_risk.max(liquidity_risk) as u8;

        // Determine recommended action
        let recommended_action = if risk_level >= 3 {
            2 // Halt
        } else if risk_level >= 2 {
            1 // Reduce exposure
        } else {
            0 // No action
        };

        // Check circuit breaker conditions
        let circuit_breaker_triggered = self.check_circuit_breaker(&contagion, &liquidity);

        let signal = SystemicRiskSignal {
            risk_level,
            contagion_signal: contagion,
            liquidity_signal: liquidity,
            circuit_breaker_triggered,
            recommended_action,
            timestamp_ns,
        };

        // Update circuit breaker state
        if circuit_breaker_triggered {
            self.circuit_breaker_state = CircuitBreakerState::Open;
            self.breaker_trips.fetch_add(1, Ordering::Relaxed);
        } else if risk_level == 0 && self.circuit_breaker_state == CircuitBreakerState::Open {
            self.circuit_breaker_state = CircuitBreakerState::HalfOpen;
        }

        signal
    }

    /// Check if circuit breaker should trip
    fn check_circuit_breaker(
        &self,
        contagion: &ContagionSignal,
        liquidity: &LiquiditySignal,
    ) -> bool {
        if !self.auto_halt_enabled.load(Ordering::Relaxed) {
            return false;
        }

        // Trip on critical contagion
        if contagion.level == ContagionLevel::Critical {
            return true;
        }

        // Trip on evaporated liquidity
        if liquidity.should_halt {
            return true;
        }

        // Trip on combined warning levels
        if contagion.level >= ContagionLevel::Warning 
            && liquidity.level >= LiquidityLevel::Stressed {
            return true;
        }

        false
    }

    /// Get current circuit breaker state
    #[inline]
    pub fn circuit_breaker_state(&self) -> CircuitBreakerState {
        self.circuit_breaker_state
    }

    /// Reset circuit breaker (manual override)
    #[inline]
    pub fn reset_circuit_breaker(&mut self) {
        self.circuit_breaker_state = CircuitBreakerState::Closed;
        self.contagion_detector.clear_systemic_event();
        self.liquidity_detector.clear_halt();
    }

    /// Get statistics
    pub fn stats(&self) -> SystemicRiskStats {
        SystemicRiskStats {
            signals_processed: self.signals_processed.load(Ordering::Relaxed),
            breaker_trips: self.breaker_trips.load(Ordering::Relaxed),
            circuit_breaker_state: self.circuit_breaker_state,
            is_auto_halt_enabled: self.auto_halt_enabled.load(Ordering::Relaxed),
            contagion_alerts: self.contagion_detector.alerts_triggered(),
            liquidity_updates: self.liquidity_detector.updates_count(),
        }
    }

    /// Check if trading should be halted
    #[inline]
    pub fn should_halt(&self) -> bool {
        self.circuit_breaker_state == CircuitBreakerState::Open
            || self.contagion_detector.is_systemic_event()
            || self.liquidity_detector.is_halted()
    }

    /// Reset all detectors
    pub fn reset(&mut self) {
        self.contagion_detector.reset();
        self.liquidity_detector.reset();
        self.reset_circuit_breaker();
    }
}

/// Systemic risk statistics
#[derive(Debug, Clone, Copy)]
pub struct SystemicRiskStats {
    pub signals_processed: u64,
    pub breaker_trips: u64,
    pub circuit_breaker_state: CircuitBreakerState,
    pub is_auto_halt_enabled: bool,
    pub contagion_alerts: u64,
    pub liquidity_updates: u64,
}

impl Default for SystemicRiskMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for systemic risk monitor
pub struct SystemicRiskBuilder {
    contagion_warning: f64,
    contagion_critical: f64,
    liquidity_reduced: f64,
    liquidity_stressed: f64,
    liquidity_evaporated: f64,
}

impl SystemicRiskBuilder {
    pub fn new() -> Self {
        Self {
            contagion_warning: 0.3,
            contagion_critical: 0.5,
            liquidity_reduced: 0.7,
            liquidity_stressed: 0.4,
            liquidity_evaporated: 0.2,
        }
    }

    pub fn contagion_thresholds(mut self, warning: f64, critical: f64) -> Self {
        self.contagion_warning = warning;
        self.contagion_critical = critical;
        self
    }

    pub fn liquidity_thresholds(mut self, reduced: f64, stressed: f64, evaporated: f64) -> Self {
        self.liquidity_reduced = reduced;
        self.liquidity_stressed = stressed;
        self.liquidity_evaporated = evaporated;
        self
    }

    pub fn build(self) -> SystemicRiskMonitor {
        SystemicRiskMonitor::with_thresholds(
            self.contagion_warning,
            self.contagion_critical,
            self.liquidity_reduced,
            self.liquidity_stressed,
            self.liquidity_evaporated,
        )
    }
}

impl Default for SystemicRiskBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normal_operation() {
        let mut monitor = SystemicRiskBuilder::new().build();

        monitor.register_asset(1, 0.1);
        monitor.register_venue(1, 1000000);

        monitor.update_correlation(1, 0.15);
        monitor.update_venue_liquidity(1, 950000);

        let signal = monitor.evaluate();

        assert_eq!(signal.risk_level, 0);
        assert!(!signal.circuit_breaker_triggered);
        assert_eq!(monitor.circuit_breaker_state(), CircuitBreakerState::Closed);
    }

    #[test]
    fn test_contagion_trigger() {
        let mut monitor = SystemicRiskBuilder::new().build();

        monitor.register_asset(1, 0.1);
        monitor.register_asset(2, 0.15);

        // Trigger contagion
        monitor.update_correlation(1, 0.9);
        monitor.update_correlation(2, 0.85);

        let signal = monitor.evaluate();

        assert!(signal.risk_level >= 2);
        assert!(signal.circuit_breaker_triggered);
        assert_eq!(monitor.circuit_breaker_state(), CircuitBreakerState::Open);
    }

    #[test]
    fn test_liquidity_trigger() {
        let mut monitor = SystemicRiskBuilder::new().build();

        monitor.register_venue(1, 1000000);
        monitor.register_venue(2, 1000000);

        // Trigger liquidity evaporation
        monitor.update_venue_liquidity(1, 100000);
        monitor.update_venue_liquidity(2, 80000);

        let signal = monitor.evaluate();

        assert!(signal.risk_level >= 2);
        assert!(signal.circuit_breaker_triggered);
    }

    #[test]
    fn test_manual_reset() {
        let mut monitor = SystemicRiskBuilder::new().build();

        monitor.register_asset(1, 0.1);
        monitor.update_correlation(1, 0.9);
        monitor.evaluate();

        assert_eq!(monitor.circuit_breaker_state(), CircuitBreakerState::Open);

        monitor.reset_circuit_breaker();
        assert_eq!(monitor.circuit_breaker_state(), CircuitBreakerState::Closed);
    }

    #[test]
    fn test_statistics() {
        let mut monitor = SystemicRiskBuilder::new().build();

        monitor.register_asset(1, 0.1);
        monitor.register_venue(1, 1000000);

        monitor.update_correlation(1, 0.15);
        monitor.update_venue_liquidity(1, 950000);
        monitor.evaluate();

        let stats = monitor.stats();
        assert!(stats.signals_processed > 0);
        assert_eq!(stats.breaker_trips, 0);
    }
}
