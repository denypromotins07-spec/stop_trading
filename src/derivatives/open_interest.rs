//! Open Interest and Funding Rate Tracker
//! 
//! Tracks Open Interest (OI) and Funding Rates for perpetual swaps.
//! Correlates OI build-ups with price action to detect regime shifts.

use std::sync::atomic::{AtomicU64, AtomicI64, AtomicBool, Ordering};
use thiserror::Error;

/// Errors that can occur in OI/funding tracking
#[derive(Debug, Error)]
pub enum OiError {
    #[error("Invalid data: {0}")]
    InvalidData(String),
    #[error("Overflow detected")]
    Overflow,
    #[error("Rate of change calculation error")]
    RocError,
}

/// Open Interest update
#[derive(Debug, Clone, Copy)]
pub struct OiUpdate {
    pub timestamp_ns: u64,
    pub symbol: [u8; 16],
    pub open_interest: f64,
    pub open_interest_usd: f64,
    pub price: f64,
}

impl OiUpdate {
    pub fn new(
        timestamp_ns: u64,
        symbol: &str,
        open_interest: f64,
        open_interest_usd: f64,
        price: f64,
    ) -> Result<Self, OiError> {
        if open_interest < 0.0 || open_interest_usd < 0.0 || price <= 0.0 {
            return Err(OiError::InvalidData(
                "Values must be non-negative and price must be positive".to_string(),
            ));
        }

        let mut bytes = [0u8; 16];
        let slice = symbol.as_bytes();
        let copy_len = slice.len().min(16);
        bytes[..copy_len].copy_from_slice(&slice[..copy_len]);

        Ok(Self {
            timestamp_ns,
            symbol: bytes,
            open_interest,
            open_interest_usd,
            price,
        })
    }

    pub fn symbol_str(&self) -> String {
        String::from_utf8_lossy(&self.symbol)
            .trim_end_matches('\0')
            .to_string()
    }
}

/// Funding rate update
#[derive(Debug, Clone, Copy)]
pub struct FundingUpdate {
    pub timestamp_ns: u64,
    pub symbol: [u8; 16],
    pub funding_rate: f64,      // Per interval (usually 8 hours)
    pub annualized_rate: f64,   // Annualized percentage
    pub next_funding_ns: u64,
}

impl FundingUpdate {
    pub fn new(
        timestamp_ns: u64,
        symbol: &str,
        funding_rate: f64,
        next_funding_ns: u64,
    ) -> Result<Self, OiError> {
        let mut bytes = [0u8; 16];
        let slice = symbol.as_bytes();
        let copy_len = slice.len().min(16);
        bytes[..copy_len].copy_from_slice(&slice[..copy_len]);

        // Annualized: funding_rate * 3 * 365 (3 intervals per day)
        let annualized_rate = funding_rate * 3.0 * 365.0 * 100.0;

        Ok(Self {
            timestamp_ns,
            symbol: bytes,
            funding_rate,
            annualized_rate,
            next_funding_ns,
        })
    }

    pub fn symbol_str(&self) -> String {
        String::from_utf8_lossy(&self.symbol)
            .trim_end_matches('\0')
            .to_string()
    }

    /// Check if funding is extremely high (potential top signal)
    pub fn is_extreme_positive(&self, threshold: f64) -> bool {
        self.annualized_rate > threshold
    }

    /// Check if funding is extremely negative (potential bottom signal)
    pub fn is_extreme_negative(&self, threshold: f64) -> bool {
        self.annualized_rate < -threshold
    }
}

/// Market regime based on OI and funding
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarketRegime {
    Normal,           // Balanced conditions
    OverleveragedLong,  // High OI + positive funding
    OverleveragedShort, // High OI + negative funding
    Deleveraging,     // Rapid OI decrease
    Accumulation,     // OI building with stable price
}

/// Combined OI/Funding state
#[derive(Debug, Clone, Copy)]
pub struct DerivativesState {
    pub open_interest: f64,
    pub open_interest_usd: f64,
    pub oi_change_pct: f64,
    pub funding_rate: f64,
    pub annualized_funding: f64,
    pub regime: MarketRegime,
    pub timestamp_ns: u64,
}

impl DerivativesState {
    pub fn new(
        open_interest: f64,
        open_interest_usd: f64,
        oi_change_pct: f64,
        funding_rate: f64,
        annualized_funding: f64,
        regime: MarketRegime,
        timestamp_ns: u64,
    ) -> Self {
        Self {
            open_interest,
            open_interest_usd,
            oi_change_pct,
            funding_rate,
            annualized_funding,
            regime,
            timestamp_ns,
        }
    }

    /// Check if conditions suggest potential long squeeze
    pub fn is_long_squeeze_risk(&self) -> bool {
        self.regime == MarketRegime::OverleveragedLong && self.oi_change_pct < -5.0
    }

    /// Check if conditions suggest potential short squeeze
    pub fn is_short_squeeze_risk(&self) -> bool {
        self.regime == MarketRegime::OverleveragedShort && self.oi_change_pct > 5.0
    }
}

/// Lock-free Open Interest Tracker
pub struct OpenInterestTracker {
    /// Current OI (scaled by 1e6)
    current_oi: AtomicU64,
    /// Previous OI (scaled by 1e6)
    previous_oi: AtomicU64,
    /// Current OI USD (scaled by 1e6)
    current_oi_usd: AtomicU64,
    /// Last timestamp
    last_timestamp_ns: AtomicU64,
    /// Baseline OI for comparison (scaled by 1e6)
    baseline_oi: AtomicU64,
    /// Active flag
    active: AtomicBool,
    /// Scale factor
    scale: i64,
}

unsafe impl Send for OpenInterestTracker {}
unsafe impl Sync for OpenInterestTracker {}

impl OpenInterestTracker {
    /// Create a new OI tracker
    pub fn new() -> Self {
        Self {
            current_oi: AtomicU64::new(0),
            previous_oi: AtomicU64::new(0),
            current_oi_usd: AtomicU64::new(0),
            last_timestamp_ns: AtomicU64::new(0),
            baseline_oi: AtomicU64::new(0),
            active: AtomicBool::new(true),
            scale: 1_000_000,
        }
    }

    /// Process an OI update (lock-free)
    pub fn process_update(&self, update: &OiUpdate) -> Result<OIMetrics, OiError> {
        if !self.active.load(Ordering::Relaxed) {
            return Ok(OIMetrics::default());
        }

        let scaled_oi = (update.open_interest * self.scale as f64) as u64;
        let scaled_oi_usd = (update.open_interest_usd * self.scale as f64) as u64;

        // Store previous OI before updating
        let prev_oi = self.current_oi.load(Ordering::Relaxed);
        
        // Update atomically
        self.previous_oi.store(prev_oi, Ordering::Relaxed);
        self.current_oi.store(scaled_oi, Ordering::Relaxed);
        self.current_oi_usd.store(scaled_oi_usd, Ordering::Relaxed);
        self.last_timestamp_ns.store(update.timestamp_ns, Ordering::Relaxed);

        // Set baseline if not set
        if self.baseline_oi.load(Ordering::Relaxed) == 0 {
            self.baseline_oi.store(scaled_oi, Ordering::Relaxed);
        }

        // Calculate metrics
        let prev_oi_val = prev_oi as f64 / self.scale as f64;
        let oi_change = update.open_interest - prev_oi_val;
        let oi_change_pct = if prev_oi_val > 0.0 {
            (oi_change / prev_oi_val) * 100.0
        } else {
            0.0
        };

        let baseline = self.baseline_oi.load(Ordering::Relaxed) as f64 / self.scale as f64;
        let vs_baseline_pct = if baseline > 0.0 {
            ((update.open_interest - baseline) / baseline) * 100.0
        } else {
            0.0
        };

        Ok(OIMetrics {
            timestamp_ns: update.timestamp_ns,
            open_interest: update.open_interest,
            open_interest_usd: update.open_interest_usd,
            oi_change,
            oi_change_pct,
            vs_baseline_pct,
        })
    }

    /// Get OI change percentage
    pub fn oi_change_pct(&self) -> f64 {
        let current = self.current_oi.load(Ordering::Relaxed) as f64 / self.scale as f64;
        let previous = self.previous_oi.load(Ordering::Relaxed) as f64 / self.scale as f64;

        if previous > 0.0 {
            ((current - previous) / previous) * 100.0
        } else {
            0.0
        }
    }

    /// Get current OI
    pub fn current_oi(&self) -> f64 {
        self.current_oi.load(Ordering::Relaxed) as f64 / self.scale as f64
    }

    /// Update baseline for comparison
    pub fn set_baseline(&self, baseline_oi: f64) {
        let scaled = (baseline_oi * self.scale as f64) as u64;
        self.baseline_oi.store(scaled, Ordering::Relaxed);
    }

    /// Activate/deactivate tracker
    pub fn set_active(&self, active: bool) {
        self.active.store(active, Ordering::Relaxed);
    }

    pub fn is_active(&self) -> bool {
        self.active.load(Ordering::Relaxed)
    }
}

impl Default for OpenInterestTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// OI metrics
#[derive(Debug, Clone, Copy, Default)]
pub struct OIMetrics {
    pub timestamp_ns: u64,
    pub open_interest: f64,
    pub open_interest_usd: f64,
    pub oi_change: f64,
    pub oi_change_pct: f64,
    pub vs_baseline_pct: f64,
}

/// Lock-free Funding Rate Tracker
pub struct FundingRateTracker {
    /// Current funding rate (scaled by 1e9)
    current_rate: AtomicI64,
    /// Previous funding rate (scaled by 1e9)
    previous_rate: AtomicI64,
    /// Annualized rate (scaled by 1e6)
    annualized_rate: AtomicI64,
    /// Next funding timestamp
    next_funding_ns: AtomicU64,
    /// Last timestamp
    last_timestamp_ns: AtomicU64,
    /// Active flag
    active: AtomicBool,
    /// Scale factors
    rate_scale: i64,
    annual_scale: i64,
}

unsafe impl Send for FundingRateTracker {}
unsafe impl Sync for FundingRateTracker {}

impl FundingRateTracker {
    /// Create a new funding rate tracker
    pub fn new() -> Self {
        Self {
            current_rate: AtomicI64::new(0),
            previous_rate: AtomicI64::new(0),
            annualized_rate: AtomicI64::new(0),
            next_funding_ns: AtomicU64::new(0),
            last_timestamp_ns: AtomicU64::new(0),
            active: AtomicBool::new(true),
            rate_scale: 1_000_000_000,
            annual_scale: 1_000_000,
        }
    }

    /// Process a funding rate update (lock-free)
    pub fn process_update(&self, update: &FundingUpdate) -> Result<FundingMetrics, OiError> {
        if !self.active.load(Ordering::Relaxed) {
            return Ok(FundingMetrics::default());
        }

        // Store previous rate
        let prev_rate = self.current_rate.load(Ordering::Relaxed);

        // Update atomically
        let scaled_rate = (update.funding_rate * self.rate_scale as f64) as i64;
        let scaled_annual = (update.annualized_rate * self.annual_scale as f64) as i64;

        self.previous_rate.store(prev_rate, Ordering::Relaxed);
        self.current_rate.store(scaled_rate, Ordering::Relaxed);
        self.annualized_rate.store(scaled_annual, Ordering::Relaxed);
        self.next_funding_ns.store(update.next_funding_ns, Ordering::Relaxed);
        self.last_timestamp_ns.store(update.timestamp_ns, Ordering::Relaxed);

        // Calculate metrics
        let prev_rate_val = prev_rate as f64 / self.rate_scale as f64;
        let rate_change = update.funding_rate - prev_rate_val;

        Ok(FundingMetrics {
            timestamp_ns: update.timestamp_ns,
            funding_rate: update.funding_rate,
            annualized_rate: update.annualized_rate,
            rate_change,
            next_funding_ns: update.next_funding_ns,
            time_to_funding_ms: update.next_funding_ns.saturating_sub(update.timestamp_ns) / 1_000_000,
        })
    }

    /// Get current funding rate
    pub fn current_rate(&self) -> f64 {
        self.current_rate.load(Ordering::Relaxed) as f64 / self.rate_scale as f64
    }

    /// Get annualized funding rate
    pub fn annualized_rate(&self) -> f64 {
        self.annualized_rate.load(Ordering::Relaxed) as f64 / self.annual_scale as f64
    }

    /// Get time until next funding (ms)
    pub fn time_to_next_funding_ms(&self, current_time_ns: u64) -> u64 {
        let next = self.next_funding_ns.load(Ordering::Relaxed);
        next.saturating_sub(current_time_ns) / 1_000_000
    }

    /// Activate/deactivate tracker
    pub fn set_active(&self, active: bool) {
        self.active.store(active, Ordering::Relaxed);
    }

    pub fn is_active(&self) -> bool {
        self.active.load(Ordering::Relaxed)
    }
}

impl Default for FundingRateTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Funding metrics
#[derive(Debug, Clone, Copy, Default)]
pub struct FundingMetrics {
    pub timestamp_ns: u64,
    pub funding_rate: f64,
    pub annualized_rate: f64,
    pub rate_change: f64,
    pub next_funding_ns: u64,
    pub time_to_funding_ms: u64,
}

/// Combined Derivatives Tracker
pub struct DerivativesTracker {
    pub oi_tracker: OpenInterestTracker,
    pub funding_tracker: FundingRateTracker,
    /// OI threshold for overleveraged detection
    oi_threshold_pct: AtomicU64,
    /// Funding threshold for extreme detection (scaled by 1e6)
    funding_threshold: AtomicU64,
}

unsafe impl Send for DerivativesTracker {}
unsafe impl Sync for DerivativesTracker {}

impl DerivativesTracker {
    /// Create a new derivatives tracker
    pub fn new(oi_threshold_pct: f64, funding_threshold_pct: f64) -> Self {
        Self {
            oi_tracker: OpenInterestTracker::new(),
            funding_tracker: FundingRateTracker::new(),
            oi_threshold_pct: AtomicU64::new((oi_threshold_pct * 1e6) as u64),
            funding_threshold: AtomicU64::new((funding_threshold_pct * 1e6) as u64),
        }
    }

    /// Process combined update
    pub fn process_update(
        &self,
        oi_update: &OiUpdate,
        funding_update: &FundingUpdate,
    ) -> Result<DerivativesState, OiError> {
        let oi_metrics = self.oi_tracker.process_update(oi_update)?;
        let funding_metrics = self.funding_tracker.process_update(funding_update)?;

        // Determine regime
        let regime = self.determine_regime(&oi_metrics, &funding_metrics);

        Ok(DerivativesState::new(
            oi_metrics.open_interest,
            oi_metrics.open_interest_usd,
            oi_metrics.oi_change_pct,
            funding_metrics.funding_rate,
            funding_metrics.annualized_rate,
            regime,
            oi_metrics.timestamp_ns.max(funding_metrics.timestamp_ns),
        ))
    }

    /// Determine market regime
    fn determine_regime(&self, oi: &OIMetrics, funding: &FundingMetrics) -> MarketRegime {
        let oi_threshold = self.oi_threshold_pct.load(Ordering::Relaxed) as f64 / 1e6;
        let funding_threshold = self.funding_threshold.load(Ordering::Relaxed) as f64 / 1e6;

        let oi_high = oi.vs_baseline_pct > oi_threshold;
        let funding_extreme_pos = funding.annualized_rate > funding_threshold;
        let funding_extreme_neg = funding.annualized_rate < -funding_threshold;
        let oi_decreasing = oi.oi_change_pct < -oi_threshold;

        if oi_decreasing {
            MarketRegime::Deleveraging
        } else if oi_high && funding_extreme_pos {
            MarketRegime::OverleveragedLong
        } else if oi_high && funding_extreme_neg {
            MarketRegime::OverleveragedShort
        } else if oi.oi_change_pct > oi_threshold && funding.annualized_rate.abs() < 10.0 {
            MarketRegime::Accumulation
        } else {
            MarketRegime::Normal
        }
    }

    /// Get combined state
    pub fn get_state(&self) -> Option<DerivativesState> {
        let oi = self.oi_tracker.current_oi();
        let funding = self.funding_tracker.current_rate();
        let annualized = self.funding_tracker.annualized_rate();
        let oi_change = self.oi_tracker.oi_change_pct();

        // Simplified regime detection without full updates
        let regime = if oi_change < -10.0 {
            MarketRegime::Deleveraging
        } else if annualized > 50.0 {
            MarketRegime::OverleveragedLong
        } else if annualized < -50.0 {
            MarketRegime::OverleveragedShort
        } else {
            MarketRegime::Normal
        };

        Some(DerivativesState::new(
            oi,
            oi, // Simplified
            oi_change,
            funding,
            annualized,
            regime,
            self.oi_tracker.last_timestamp_ns.load(Ordering::Relaxed),
        ))
    }
}

impl Default for DerivativesTracker {
    fn default() -> Self {
        Self::new(20.0, 50.0) // 20% OI threshold, 50% annualized funding threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_oi_update() {
        let tracker = OpenInterestTracker::new();
        
        let update = OiUpdate::new(1000, "BTCUSDT", 1000000.0, 50000000000.0, 50000.0).unwrap();
        let metrics = tracker.process_update(&update).unwrap();

        assert!((metrics.open_interest - 1000000.0).abs() < 0.001);
        assert_eq!(tracker.current_oi(), 1000000.0);
    }

    #[test]
    fn test_funding_update() {
        let tracker = FundingRateTracker::new();
        
        let update = FundingUpdate::new(1000, "BTCUSDT", 0.0001, 2000000000).unwrap();
        let metrics = tracker.process_update(&update).unwrap();

        assert!((metrics.funding_rate - 0.0001).abs() < 0.00001);
        assert!((metrics.annualized_rate - 10.95).abs() < 0.1);
    }

    #[test]
    fn test_regime_detection() {
        let tracker = DerivativesTracker::new(20.0, 50.0);
        
        let oi_update = OiUpdate::new(1000, "BTCUSDT", 1500000.0, 75000000000.0, 50000.0).unwrap();
        let funding_update = FundingUpdate::new(1000, "BTCUSDT", 0.0002, 2000000000).unwrap();

        // Set baseline first
        tracker.oi_tracker.set_baseline(1000000.0);
        
        let state = tracker.process_update(&oi_update, &funding_update).unwrap();
        
        // Should detect overleveraged long (high OI + high funding)
        assert_eq!(state.regime, MarketRegime::OverleveragedLong);
    }
}
