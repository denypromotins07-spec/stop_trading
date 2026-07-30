//! Microprice Module Root
//! 
//! Feeds fair value calculations into the pre-trade risk bus.
//! Exports microprice calculator and spread analyzer.

pub mod calculator;
pub mod spread;

pub use calculator::{
    MicropriceCalculator, MicropriceResult, MicropriceSignal, MicropriceAction,
    OrderBookSnapshot, Level, MicropriceError, RollingMicropriceTracker, MicropriceTrend,
};
pub use spread::{
    SpreadAnalyzer, SpreadMetrics, SpreadRegime, TickSizeConfig, SpreadError,
    TickAlignDirection, OrderSide, SpreadStatsCollector,
};

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use thiserror::Error;

/// Microprice module errors
#[derive(Debug, Error)]
pub enum MicropriceModuleError {
    #[error("Calculation error: {0}")]
    Calculation(#[from] MicropriceError),
    #[error("Spread error: {0}")]
    Spread(#[from] SpreadError),
    #[error("Risk check failed: {0}")]
    RiskCheckFailed(String),
}

/// Fair value state for pre-trade risk checks
#[derive(Debug, Clone, Copy)]
pub struct FairValueState {
    pub microprice: f64,
    pub mid_price: f64,
    pub deviation_bps: f64,
    pub spread_bps: f64,
    pub spread_regime: SpreadRegime,
    pub bid_pressure: f64,
    pub ask_pressure: f64,
    pub timestamp_ns: u64,
}

impl FairValueState {
    pub fn new(
        microprice_result: &MicropriceResult,
        spread_metrics: &SpreadMetrics,
        spread_regime: SpreadRegime,
    ) -> Self {
        Self {
            microprice: microprice_result.microprice,
            mid_price: microprice_result.mid_price,
            deviation_bps: microprice_result.deviation_bps,
            spread_bps: spread_metrics.relative_spread_bps,
            spread_regime,
            bid_pressure: microprice_result.bid_pressure,
            ask_pressure: microprice_result.ask_pressure,
            timestamp_ns: microprice_result.timestamp_ns.max(spread_metrics.timestamp_ns),
        }
    }

    /// Check if conditions are favorable for buying
    pub fn is_favorable_for_buy(&self, max_deviation_bps: f64, max_spread_bps: f64) -> bool {
        // Microprice above mid (bullish signal)
        let microprice_signal = self.deviation_bps > 0.0;
        
        // Spread not too wide
        let spread_ok = self.spread_bps < max_spread_bps;
        
        // Not in stress regime
        let regime_ok = self.spread_regime != SpreadRegime::Extreme;
        
        // Bid pressure dominant
        let pressure_ok = self.bid_pressure > 0.5;

        microprice_signal && spread_ok && regime_ok && pressure_ok
    }

    /// Check if conditions are favorable for selling
    pub fn is_favorable_for_sell(&self, max_deviation_bps: f64, max_spread_bps: f64) -> bool {
        // Microprice below mid (bearish signal)
        let microprice_signal = self.deviation_bps < 0.0;
        
        // Spread not too wide
        let spread_ok = self.spread_bps < max_spread_bps;
        
        // Not in stress regime
        let regime_ok = self.spread_regime != SpreadRegime::Extreme;
        
        // Ask pressure dominant
        let pressure_ok = self.ask_pressure > 0.5;

        microprice_signal && spread_ok && regime_ok && pressure_ok
    }

    /// Get combined signal strength (-1.0 to 1.0)
    pub fn signal_strength(&self) -> f64 {
        // Combine deviation and pressure signals
        let deviation_signal = (self.deviation_bps / 100.0).clamp(-1.0, 1.0);
        let pressure_signal = self.bid_pressure - self.ask_pressure;
        
        (deviation_signal * 0.6 + pressure_signal * 0.4).clamp(-1.0, 1.0)
    }
}

/// Pre-trade risk check result
#[derive(Debug, Clone, Copy)]
pub struct PreTradeRiskCheck {
    pub passed: bool,
    pub fair_value_ok: bool,
    pub spread_ok: bool,
    pub regime_ok: bool,
    pub reason: Option<RiskCheckReason>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiskCheckReason {
    SpreadTooWide,
    ExtremeRegime,
    AdverseFairValue,
    StaleData,
}

/// Pre-trade risk checker using microprice and spread data
pub struct PreTradeRiskChecker {
    /// Maximum allowed spread in bps
    max_spread_bps: AtomicU64,
    /// Maximum allowed deviation in bps
    max_deviation_bps: AtomicU64,
    /// Data staleness threshold in ms
    staleness_threshold_ms: AtomicU64,
    /// Active flag
    active: AtomicBool,
    /// Price scale factor
    price_scale: i64,
}

unsafe impl Send for PreTradeRiskChecker {}
unsafe impl Sync for PreTradeRiskChecker {}

impl PreTradeRiskChecker {
    /// Create a new risk checker
    pub fn new(max_spread_bps: f64, max_deviation_bps: f64, staleness_threshold_ms: u64) -> Self {
        Self {
            max_spread_bps: AtomicU64::new((max_spread_bps * 1e6) as u64),
            max_deviation_bps: AtomicU64::new((max_deviation_bps * 1e6) as u64),
            staleness_threshold_ms: AtomicU64::new(staleness_threshold_ms),
            active: AtomicBool::new(true),
            price_scale: 1_000_000,
        }
    }

    /// Run pre-trade risk check
    pub fn check(&self, state: &FairValueState) -> PreTradeRiskCheck {
        if !self.active.load(Ordering::Relaxed) {
            return PreTradeRiskCheck {
                passed: true,
                fair_value_ok: true,
                spread_ok: true,
                regime_ok: true,
                reason: None,
            };
        }

        let current_time_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        let staleness_ms = (current_time_ns - state.timestamp_ns) / 1_000_000;
        let stale_threshold = self.staleness_threshold_ms.load(Ordering::Relaxed);

        // Check data staleness
        if staleness_ms > stale_threshold {
            return PreTradeRiskCheck {
                passed: false,
                fair_value_ok: false,
                spread_ok: true,
                regime_ok: true,
                reason: Some(RiskCheckReason::StaleData),
            };
        }

        // Check spread
        let max_spread = self.max_spread_bps.load(Ordering::Relaxed) as f64 / 1e6;
        let spread_ok = state.spread_bps < max_spread;

        // Check regime
        let regime_ok = state.spread_regime != SpreadRegime::Extreme;

        // Check fair value deviation
        let max_deviation = self.max_deviation_bps.load(Ordering::Relaxed) as f64 / 1e6;
        let fair_value_ok = state.deviation_bps.abs() < max_deviation;

        let passed = spread_ok && regime_ok && fair_value_ok;

        let reason = if !spread_ok {
            Some(RiskCheckReason::SpreadTooWide)
        } else if !regime_ok {
            Some(RiskCheckReason::ExtremeRegime)
        } else if !fair_value_ok {
            Some(RiskCheckReason::AdverseFairValue)
        } else {
            None
        };

        PreTradeRiskCheck {
            passed,
            fair_value_ok,
            spread_ok,
            regime_ok,
            reason,
        }
    }

    /// Update maximum spread threshold
    pub fn set_max_spread_bps(&self, max_spread_bps: f64) {
        self.max_spread_bps.store((max_spread_bps * 1e6) as u64, Ordering::Relaxed);
    }

    /// Update maximum deviation threshold
    pub fn set_max_deviation_bps(&self, max_deviation_bps: f64) {
        self.max_deviation_bps.store((max_deviation_bps * 1e6) as u64, Ordering::Relaxed);
    }

    /// Activate/deactivate checker
    pub fn set_active(&self, active: bool) {
        self.active.store(active, Ordering::Relaxed);
    }

    pub fn is_active(&self) -> bool {
        self.active.load(Ordering::Relaxed)
    }
}

impl Default for PreTradeRiskChecker {
    fn default() -> Self {
        Self::new(100.0, 50.0, 1000) // 100 bps max spread, 50 bps max deviation, 1s staleness
    }
}

/// Combined microprice pipeline
pub struct MicropricePipeline {
    pub calculator: MicropriceCalculator,
    pub spread_analyzer: SpreadAnalyzer,
    pub risk_checker: PreTradeRiskChecker,
}

impl MicropricePipeline {
    pub fn new(depth_levels: usize, baseline_spread_bps: f64) -> Self {
        Self {
            calculator: MicropriceCalculator::new(depth_levels),
            spread_analyzer: SpreadAnalyzer::new(baseline_spread_bps, baseline_spread_bps * 5.0),
            risk_checker: PreTradeRiskChecker::default(),
        }
    }

    /// Process order book update through the pipeline
    pub fn process_book(&self, book: &OrderBookSnapshot) -> Result<FairValueState, MicropriceModuleError> {
        // Calculate microprice
        let microprice_result = self.calculator.calculate(book)?;

        // Update spread analyzer
        let spread_metrics = self.spread_analyzer.update(
            book.best_bid().map(|l| l.price).unwrap_or(0.0),
            book.best_ask().map(|l| l.price).unwrap_or(0.0),
            book.timestamp_ns,
        )?;

        let spread_regime = self.spread_analyzer.current_regime();

        Ok(FairValueState::new(&microprice_result, &spread_metrics, spread_regime))
    }

    /// Run full pre-trade check
    pub fn pre_trade_check(&self, state: &FairValueState) -> PreTradeRiskCheck {
        self.risk_checker.check(state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fair_value_state() {
        let microprice_result = MicropriceResult::new(1000, 100.0, 100.1, 0.55, 0.45);
        let spread_metrics = SpreadMetrics::new(1000, 99.9, 100.1, 0.01).unwrap();
        
        let state = FairValueState::new(&microprice_result, &spread_metrics, SpreadRegime::Normal);
        
        assert!((state.microprice - 100.1).abs() < 0.001);
        assert!(state.is_favorable_for_buy(100.0, 50.0));
    }

    #[test]
    fn test_risk_checker() {
        let checker = PreTradeRiskChecker::new(100.0, 50.0, 1000);
        
        let state = FairValueState {
            microprice: 100.0,
            mid_price: 100.0,
            deviation_bps: 5.0,
            spread_bps: 20.0,
            spread_regime: SpreadRegime::Normal,
            bid_pressure: 0.5,
            ask_pressure: 0.5,
            timestamp_ns: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
        };

        let result = checker.check(&state);
        assert!(result.passed);
    }

    #[test]
    fn test_pipeline() {
        let pipeline = MicropricePipeline::new(3, 10.0);
        
        let book = OrderBookSnapshot::new(
            vec![Level::new(99.9, 100.0, 5)],
            vec![Level::new(100.1, 50.0, 3)],
            1000,
        );

        let state = pipeline.process_book(&book).unwrap();
        assert!((state.microprice - state.mid_price).abs() > 0.0);

        let risk_result = pipeline.pre_trade_check(&state);
        assert!(risk_result.passed);
    }
}
