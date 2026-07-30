//! Derivatives Module Root
//! 
//! Exports liquidation and OI metrics to the global event clock.
//! Aggregates derivatives data for the alpha engine.

pub mod liquidations;
pub mod open_interest;

pub use liquidations::{
    LiquidationEvent, LiquidationSide, CascadeSignal, CascadeDirection, CascadeAction,
    LiquidationDetector, LiquidationStats, LiquidationWindow, LiquidationError,
};
pub use open_interest::{
    OiUpdate, FundingUpdate, MarketRegime, DerivativesState, OpenInterestTracker,
    FundingRateTracker, DerivativesTracker, OIMetrics, FundingMetrics, OiError,
};

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use thiserror::Error;

/// Derivatives module errors
#[derive(Debug, Error)]
pub enum DerivativesModuleError {
    #[error("Liquidation error: {0}")]
    Liquidation(#[from] LiquidationError),
    #[error("OI/Funding error: {0}")]
    OiFunding(#[from] OiError),
    #[error("Data synchronization error")]
    SyncError,
}

/// Combined derivatives event for the global clock
#[derive(Debug, Clone, Copy)]
pub struct DerivativesEvent {
    pub timestamp_ns: u64,
    pub event_type: DerivativesEventType,
    pub symbol: [u8; 16],
    pub liquidation_data: Option<LiquidationEvent>,
    pub oi_data: Option<OIMetrics>,
    pub funding_data: Option<FundingMetrics>,
    pub cascade_signal: Option<CascadeSignal>,
    pub regime: MarketRegime,
}

impl DerivativesEvent {
    pub fn new_liquidation(event: LiquidationEvent) -> Self {
        let mut symbol = [0u8; 16];
        symbol.copy_from_slice(&event.symbol);

        Self {
            timestamp_ns: event.timestamp_ns,
            event_type: DerivativesEventType::Liquidation,
            symbol,
            liquidation_data: Some(event),
            oi_data: None,
            funding_data: None,
            cascade_signal: None,
            regime: MarketRegime::Normal,
        }
    }

    pub fn new_oi_update(update: &OiUpdate, metrics: OIMetrics) -> Self {
        Self {
            timestamp_ns: update.timestamp_ns,
            event_type: DerivativesEventType::OpenInterest,
            symbol: update.symbol,
            liquidation_data: None,
            oi_data: Some(metrics),
            funding_data: None,
            cascade_signal: None,
            regime: MarketRegime::Normal,
        }
    }

    pub fn new_funding_update(update: &FundingUpdate, metrics: FundingMetrics) -> Self {
        Self {
            timestamp_ns: update.timestamp_ns,
            event_type: DerivativesEventType::FundingRate,
            symbol: update.symbol,
            liquidation_data: None,
            oi_data: None,
            funding_data: Some(metrics),
            cascade_signal: None,
            regime: MarketRegime::Normal,
        }
    }

    pub fn with_cascade_signal(mut self, signal: CascadeSignal) -> Self {
        self.cascade_signal = Some(signal);
        self
    }

    pub fn with_regime(mut self, regime: MarketRegime) -> Self {
        self.regime = regime;
        self
        }
}

/// Type of derivatives event
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DerivativesEventType {
    Liquidation,
    OpenInterest,
    FundingRate,
    CascadeDetected,
    RegimeChange,
}

/// Derivatives data aggregator
pub struct DerivativesAggregator {
    /// Liquidation detector
    liquidation_detector: LiquidationDetector,
    /// Derivatives tracker (OI + Funding)
    derivatives_tracker: DerivativesTracker,
    /// Event counter
    event_count: AtomicU64,
    /// Last event timestamp
    last_event_ns: AtomicU64,
    /// Active flag
    active: AtomicBool,
}

unsafe impl Send for DerivativesAggregator {}
unsafe impl Sync for DerivativesAggregator {}

impl DerivativesAggregator {
    /// Create a new derivatives aggregator
    pub fn new(cascade_threshold: u64, window_size_ms: u64) -> Self {
        Self {
            liquidation_detector: LiquidationDetector::new(cascade_threshold, window_size_ms),
            derivatives_tracker: DerivativesTracker::default(),
            event_count: AtomicU64::new(0),
            last_event_ns: AtomicU64::new(0),
            active: AtomicBool::new(true),
        }
    }

    /// Process a liquidation event
    pub fn process_liquidation(
        &self,
        event: LiquidationEvent,
    ) -> Result<Option<DerivativesEvent>, DerivativesModuleError> {
        if !self.active.load(Ordering::Relaxed) {
            return Ok(None);
        }

        let mut output_event = DerivativesEvent::new_liquidation(event.clone());

        // Check for cascade
        if let Some(signal) = self.liquidation_detector.process_event(&event)? {
            output_event = output_event.with_cascade_signal(signal);
            output_event.event_type = DerivativesEventType::CascadeDetected;
        }

        self.event_count.fetch_add(1, Ordering::Relaxed);
        self.last_event_ns.store(event.timestamp_ns, Ordering::Relaxed);

        Ok(Some(output_event))
    }

    /// Process OI and funding updates together
    pub fn process_oi_funding(
        &self,
        oi_update: OiUpdate,
        funding_update: FundingUpdate,
    ) -> Result<DerivativesEvent, DerivativesModuleError> {
        if !self.active.load(Ordering::Relaxed) {
            return Err(DerivativesModuleError::SyncError);
        }

        let state = self.derivatives_tracker.process_update(&oi_update, &funding_update)?;

        let oi_metrics = OIMetrics {
            timestamp_ns: oi_update.timestamp_ns,
            open_interest: oi_update.open_interest,
            open_interest_usd: oi_update.open_interest_usd,
            oi_change: 0.0, // Would be calculated by tracker
            oi_change_pct: 0.0,
            vs_baseline_pct: 0.0,
        };

        let funding_metrics = FundingMetrics {
            timestamp_ns: funding_update.timestamp_ns,
            funding_rate: funding_update.funding_rate,
            annualized_rate: funding_update.annualized_rate,
            rate_change: 0.0,
            next_funding_ns: funding_update.next_funding_ns,
            time_to_funding_ms: 0,
        };

        let mut event = DerivativesEvent::new_oi_update(&oi_update, oi_metrics);
        event.funding_data = Some(funding_metrics);
        event.regime = state.regime;

        // Check for regime change
        if state.regime != MarketRegime::Normal {
            event.event_type = DerivativesEventType::RegimeChange;
        }

        self.event_count.fetch_add(1, Ordering::Relaxed);
        self.last_event_ns.store(
            oi_update.timestamp_ns.max(funding_update.timestamp_ns),
            Ordering::Relaxed,
        );

        Ok(event)
    }

    /// Get current derivatives state
    pub fn get_state(&self) -> DerivativesStateSummary {
        let liq_stats = self.liquidation_detector.stats();
        let deriv_state = self.derivatives_tracker.get_state();

        DerivativesStateSummary {
            long_liquidation_count: liq_stats.long_count,
            short_liquidation_count: liq_stats.short_count,
            net_liquidation_pressure: liq_stats.net_pressure,
            current_oi: deriv_state.as_ref().map(|s| s.open_interest).unwrap_or(0.0),
            oi_change_pct: deriv_state.as_ref().map(|s| s.oi_change_pct).unwrap_or(0.0),
            funding_rate: deriv_state.as_ref().map(|s| s.funding_rate).unwrap_or(0.0),
            annualized_funding: deriv_state.as_ref().map(|s| s.annualized_funding).unwrap_or(0.0),
            regime: deriv_state.as_ref().map(|s| s.regime).unwrap_or(MarketRegime::Normal),
        }
    }

    /// Activate/deactivate aggregator
    pub fn set_active(&self, active: bool) {
        self.active.store(active, Ordering::Relaxed);
        self.liquidation_detector.set_active(active);
        self.derivatives_tracker.oi_tracker.set_active(active);
        self.derivatives_tracker.funding_tracker.set_active(active);
    }

    pub fn is_active(&self) -> bool {
        self.active.load(Ordering::Relaxed)
    }

    /// Get event count
    pub fn event_count(&self) -> u64 {
        self.event_count.load(Ordering::Relaxed)
    }
}

/// Summary of current derivatives state
#[derive(Debug, Clone, Copy)]
pub struct DerivativesStateSummary {
    pub long_liquidation_count: u64,
    pub short_liquidation_count: u64,
    pub net_liquidation_pressure: f64,
    pub current_oi: f64,
    pub oi_change_pct: f64,
    pub funding_rate: f64,
    pub annualized_funding: f64,
    pub regime: MarketRegime,
}

impl DerivativesStateSummary {
    /// Check if conditions suggest high volatility ahead
    pub fn is_high_volatility_expected(&self) -> bool {
        // High leverage + extreme funding often precedes volatility
        let high_leverage = matches!(
            self.regime,
            MarketRegime::OverleveragedLong | MarketRegime::OverleveragedShort
        );
        let deleveraging = self.regime == MarketRegime::Deleveraging;
        let extreme_funding = self.annualized_funding.abs() > 100.0;

        (high_leverage || deleveraging) && extreme_funding
    }

    /// Get directional bias from derivatives data
    pub fn directional_bias(&self) -> f64 {
        // Combine signals: positive = bullish, negative = bearish
        let liq_bias = self.net_liquidation_pressure; // Positive = shorts squeezed (bullish)
        let funding_bias = -self.funding_rate * 1000.0; // Negative funding = bullish
        let oi_bias = if self.oi_change_pct > 10.0 { 0.5 } else if self.oi_change_pct < -10.0 { -0.5 } else { 0.0 };

        (liq_bias * 0.4 + funding_bias.clamp(-1.0, 1.0) * 0.4 + oi_bias * 0.2).clamp(-1.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aggregator_liquidation() {
        let aggregator = DerivativesAggregator::new(3, 60000);

        let event = LiquidationEvent::new(
            1000,
            "BTCUSDT",
            LiquidationSide::Long,
            50000.0,
            10.0,
        ).unwrap();

        let result = aggregator.process_liquidation(event).unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap().event_type, DerivativesEventType::Liquidation);
    }

    #[test]
    fn test_state_summary() {
        let summary = DerivativesStateSummary {
            long_liquidation_count: 100,
            short_liquidation_count: 50,
            net_liquidation_pressure: 0.3,
            current_oi: 1000000.0,
            oi_change_pct: 5.0,
            funding_rate: 0.0001,
            annualized_funding: 10.95,
            regime: MarketRegime::Normal,
        };

        assert!(!summary.is_high_volatility_expected());
        assert!(summary.directional_bias() > 0.0);
    }
}
