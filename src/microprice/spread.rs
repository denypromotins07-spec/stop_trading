//! Bid-Ask Spread Analyzer
//! 
//! Analyzes bid-ask spread and tick-size alignment for various crypto assets.
//! Monitors spread widening as a proxy for liquidity evaporation during market stress.

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use thiserror::Error;

/// Errors that can occur in spread analysis
#[derive(Debug, Error)]
pub enum SpreadError {
    #[error("Invalid spread data: {0}")]
    InvalidSpreadData(String),
    #[error("Tick size not configured for asset")]
    TickSizeNotConfigured,
    #[error("Overflow detected")]
    Overflow,
}

/// Asset tick size configuration
#[derive(Debug, Clone, Copy)]
pub struct TickSizeConfig {
    pub symbol: [u8; 16],
    pub tick_size: f64,
    pub lot_size: f64,
    pub min_notional: f64,
}

impl TickSizeConfig {
    pub fn new(symbol: &str, tick_size: f64, lot_size: f64, min_notional: f64) -> Self {
        let mut bytes = [0u8; 16];
        let slice = symbol.as_bytes();
        let copy_len = slice.len().min(16);
        bytes[..copy_len].copy_from_slice(&slice[..copy_len]);

        Self {
            symbol: bytes,
            tick_size,
            lot_size,
            min_notional,
        }
    }

    pub fn symbol_str(&self) -> String {
        String::from_utf8_lossy(&self.symbol)
            .trim_end_matches('\0')
            .to_string()
    }
}

/// Spread metrics snapshot
#[derive(Debug, Clone, Copy)]
pub struct SpreadMetrics {
    pub timestamp_ns: u64,
    pub absolute_spread: f64,
    pub relative_spread_bps: f64, // Basis points
    pub mid_price: f64,
    pub best_bid: f64,
    pub best_ask: f64,
    /// Number of ticks in the spread
    pub spread_ticks: u32,
    /// Effective spread (considering depth)
    pub effective_spread_bps: f64,
}

impl SpreadMetrics {
    pub fn new(
        timestamp_ns: u64,
        best_bid: f64,
        best_ask: f64,
        tick_size: f64,
    ) -> Result<Self, SpreadError> {
        if best_bid <= 0.0 || best_ask <= 0.0 {
            return Err(SpreadError::InvalidSpreadData(
                "Prices must be positive".to_string(),
            ));
        }

        if best_bid >= best_ask {
            return Err(SpreadError::InvalidSpreadData(
                "Bid must be less than ask".to_string(),
            ));
        }

        let absolute_spread = best_ask - best_bid;
        let mid_price = (best_bid + best_ask) / 2.0;
        let relative_spread_bps = if mid_price > 0.0 {
            (absolute_spread / mid_price) * 10000.0
        } else {
            0.0
        };

        let spread_ticks = if tick_size > 0.0 {
            (absolute_spread / tick_size).round() as u32
        } else {
            0
        };

        Ok(Self {
            timestamp_ns,
            absolute_spread,
            relative_spread_bps,
            mid_price,
            best_bid,
            best_ask,
            spread_ticks,
            effective_spread_bps: relative_spread_bps, // Simplified
        })
    }

    /// Check if spread is abnormally wide
    pub fn is_wide(&self, threshold_bps: f64) -> bool {
        self.relative_spread_bps > threshold_bps
    }

    /// Check if spread is tight (good liquidity)
    pub fn is_tight(&self, threshold_bps: f64) -> bool {
        self.relative_spread_bps < threshold_bps
    }
}

/// Spread regime classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpreadRegime {
    Normal,     // Typical spread conditions
    Wide,       // Spread wider than normal
    Extreme,    // Extremely wide spread (market stress)
    Tight,      // Unusually tight spread
}

/// Lock-free Spread Analyzer
pub struct SpreadAnalyzer {
    /// Last absolute spread (scaled by 1e9)
    last_spread: AtomicU64,
    /// Last relative spread in bps (scaled by 1e6)
    last_spread_bps: AtomicU64,
    /// Baseline spread in bps (scaled by 1e6)
    baseline_spread_bps: AtomicU64,
    /// Current spread regime
    current_regime: AtomicU64, // Encoded as u8
    /// Tick size (scaled by 1e9)
    tick_size_scaled: AtomicU64,
    /// Alert threshold in bps (scaled by 1e6)
    alert_threshold_bps: AtomicU64,
    /// Active flag
    active: AtomicBool,
    /// Price scale factor
    price_scale: i64,
}

unsafe impl Send for SpreadAnalyzer {}
unsafe impl Sync for SpreadAnalyzer {}

impl SpreadAnalyzer {
    /// Create a new spread analyzer
    pub fn new(baseline_spread_bps: f64, alert_threshold_bps: f64) -> Self {
        Self {
            last_spread: AtomicU64::new(0),
            last_spread_bps: AtomicU64::new(0),
            baseline_spread_bps: AtomicU64::new((baseline_spread_bps * 1e6) as u64),
            current_regime: AtomicU64::new(SpreadRegime::Normal as u64),
            tick_size_scaled: AtomicU64::new(0),
            alert_threshold_bps: AtomicU64::new((alert_threshold_bps * 1e6) as u64),
            active: AtomicBool::new(true),
            price_scale: 1_000_000_000,
        }
    }

    /// Set tick size for an asset
    pub fn set_tick_size(&self, tick_size: f64) {
        let scaled = (tick_size * self.price_scale as f64) as u64;
        self.tick_size_scaled.store(scaled, Ordering::Relaxed);
    }

    /// Get tick size
    pub fn get_tick_size(&self) -> f64 {
        self.tick_size_scaled.load(Ordering::Relaxed) as f64 / self.price_scale as f64
    }

    /// Update spread metrics
    pub fn update(&self, best_bid: f64, best_ask: f64, timestamp_ns: u64) -> Result<SpreadMetrics, SpreadError> {
        if !self.active.load(Ordering::Relaxed) {
            return Err(SpreadError::InvalidSpreadData("Analyzer not active".to_string()));
        }

        let tick_size = self.get_tick_size();
        let metrics = SpreadMetrics::new(timestamp_ns, best_bid, best_ask, tick_size)?;

        // Store atomically
        self.last_spread.store(
            (metrics.absolute_spread * self.price_scale as f64) as u64,
            Ordering::Relaxed,
        );
        self.last_spread_bps.store(
            (metrics.relative_spread_bps * 1e6) as u64,
            Ordering::Relaxed,
        );

        // Update regime
        let regime = self.classify_regime(metrics.relative_spread_bps);
        self.current_regime.store(regime as u64, Ordering::Relaxed);

        Ok(metrics)
    }

    /// Classify spread regime
    fn classify_regime(&self, spread_bps: f64) -> SpreadRegime {
        let baseline = self.baseline_spread_bps.load(Ordering::Relaxed) as f64 / 1e6;
        let threshold = self.alert_threshold_bps.load(Ordering::Relaxed) as f64 / 1e6;

        if spread_bps > threshold * 2.0 {
            SpreadRegime::Extreme
        } else if spread_bps > threshold {
            SpreadRegime::Wide
        } else if spread_bps < baseline * 0.5 {
            SpreadRegime::Tight
        } else {
            SpreadRegime::Normal
        }
    }

    /// Get current spread regime
    pub fn current_regime(&self) -> SpreadRegime {
        match self.current_regime.load(Ordering::Relaxed) {
            1 => SpreadRegime::Wide,
            2 => SpreadRegime::Extreme,
            3 => SpreadRegime::Tight,
            _ => SpreadRegime::Normal,
        }
    }

    /// Check if spread indicates market stress
    pub fn is_stress_condition(&self) -> bool {
        let regime = self.current_regime();
        regime == SpreadRegime::Wide || regime == SpreadRegime::Extreme
    }

    /// Get last spread in basis points
    pub fn last_spread_bps(&self) -> f64 {
        self.last_spread_bps.load(Ordering::Relaxed) as f64 / 1e6
    }

    /// Align price to tick size
    pub fn align_to_tick(&self, price: f64, direction: TickAlignDirection) -> f64 {
        let tick_size = self.get_tick_size();
        if tick_size <= 0.0 {
            return price;
        }

        match direction {
            TickAlignDirection::Up => (price / tick_size).ceil() * tick_size,
            TickAlignDirection::Down => (price / tick_size).floor() * tick_size,
            TickAlignDirection::Nearest => (price / tick_size).round() * tick_size,
        }
    }

    /// Calculate optimal limit order price
    pub fn calculate_limit_price(
        &self,
        side: OrderSide,
        reference_price: f64,
        aggressiveness: f64,
    ) -> f64 {
        let tick_size = self.get_tick_size();
        let spread = self.last_spread.load(Ordering::Relaxed) as f64 / self.price_scale as f64;

        let offset = spread * aggressiveness.clamp(0.0, 1.0);

        let raw_price = match side {
            OrderSide::Buy => reference_price - offset / 2.0,
            OrderSide::Sell => reference_price + offset / 2.0,
        };

        self.align_to_tick(raw_price, TickAlignDirection::Nearest)
    }

    /// Activate/deactivate analyzer
    pub fn set_active(&self, active: bool) {
        self.active.store(active, Ordering::Relaxed);
    }

    pub fn is_active(&self) -> bool {
        self.active.load(Ordering::Relaxed)
    }
}

impl Default for SpreadAnalyzer {
    fn default() -> Self {
        Self::new(10.0, 50.0) // 10 bps baseline, 50 bps alert threshold
    }
}

/// Direction for tick alignment
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TickAlignDirection {
    Up,
    Down,
    Nearest,
}

/// Order side for price calculation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderSide {
    Buy,
    Sell,
}

/// Spread statistics collector
pub struct SpreadStatsCollector {
    count: AtomicU64,
    sum_spread_bps: AtomicU64,
    min_spread_bps: AtomicU64,
    max_spread_bps: AtomicU64,
    stress_events: AtomicU64,
    price_scale: i64,
}

unsafe impl Send for SpreadStatsCollector {}
unsafe impl Sync for SpreadStatsCollector {}

impl SpreadStatsCollector {
    pub fn new() -> Self {
        Self {
            count: AtomicU64::new(0),
            sum_spread_bps: AtomicU64::new(0),
            min_spread_bps: AtomicU64::new(u64::MAX),
            max_spread_bps: AtomicU64::new(0),
            stress_events: AtomicU64::new(0),
            price_scale: 1_000_000, // For bps with 2 decimal places
        }
    }

    /// Record a spread observation
    pub fn record(&self, spread_bps: f64, is_stress: bool) {
        let scaled = (spread_bps * self.price_scale as f64) as u64;

        self.count.fetch_add(1, Ordering::Relaxed);
        self.sum_spread_bps.fetch_add(scaled, Ordering::Relaxed);

        // Update min/max atomically (may have races but acceptable for stats)
        let mut current_min = self.min_spread_bps.load(Ordering::Relaxed);
        while scaled < current_min {
            match self.min_spread_bps.compare_exchange_weak(
                current_min,
                scaled,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(x) => current_min = x,
            }
        }

        let mut current_max = self.max_spread_bps.load(Ordering::Relaxed);
        while scaled > current_max {
            match self.max_spread_bps.compare_exchange_weak(
                current_max,
                scaled,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(x) => current_max = x,
            }
        }

        if is_stress {
            self.stress_events.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Get average spread
    pub fn avg_spread_bps(&self) -> f64 {
        let count = self.count.load(Ordering::Relaxed);
        if count == 0 {
            return 0.0;
        }
        self.sum_spread_bps.load(Ordering::Relaxed) as f64 / count as f64 / self.price_scale as f64
    }

    /// Get minimum spread observed
    pub fn min_spread_bps(&self) -> f64 {
        let min = self.min_spread_bps.load(Ordering::Relaxed);
        if min == u64::MAX {
            0.0
        } else {
            min as f64 / self.price_scale as f64
        }
    }

    /// Get maximum spread observed
    pub fn max_spread_bps(&self) -> f64 {
        self.max_spread_bps.load(Ordering::Relaxed) as f64 / self.price_scale as f64
    }

    /// Get stress event count
    pub fn stress_event_count(&self) -> u64 {
        self.stress_events.load(Ordering::Relaxed)
    }

    /// Get total observations
    pub fn observation_count(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }

    /// Reset statistics
    pub fn reset(&self) {
        self.count.store(0, Ordering::Relaxed);
        self.sum_spread_bps.store(0, Ordering::Relaxed);
        self.min_spread_bps.store(u64::MAX, Ordering::Relaxed);
        self.max_spread_bps.store(0, Ordering::Relaxed);
        self.stress_events.store(0, Ordering::Relaxed);
    }
}

impl Default for SpreadStatsCollector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spread_metrics() {
        let metrics = SpreadMetrics::new(1000, 99.9, 100.1, 0.01).unwrap();

        assert!((metrics.absolute_spread - 0.2).abs() < 0.001);
        assert!((metrics.mid_price - 100.0).abs() < 0.001);
        assert!((metrics.relative_spread_bps - 20.0).abs() < 0.1);
    }

    #[test]
    fn test_spread_analyzer() {
        let analyzer = SpreadAnalyzer::new(10.0, 50.0);
        analyzer.set_tick_size(0.01);

        let metrics = analyzer.update(99.9, 100.1, 1000).unwrap();
        
        assert!((metrics.absolute_spread - 0.2).abs() < 0.001);
        assert_eq!(analyzer.current_regime(), SpreadRegime::Normal);
    }

    #[test]
    fn test_tick_alignment() {
        let analyzer = SpreadAnalyzer::default();
        analyzer.set_tick_size(0.01);

        // Round up
        let aligned = analyzer.align_to_tick(99.913, TickAlignDirection::Up);
        assert!((aligned - 99.92).abs() < 0.001);

        // Round down
        let aligned = analyzer.align_to_tick(99.917, TickAlignDirection::Down);
        assert!((aligned - 99.91).abs() < 0.001);

        // Round nearest
        let aligned = analyzer.align_to_tick(99.914, TickAlignDirection::Nearest);
        assert!((aligned - 99.91).abs() < 0.001);
    }

    #[test]
    fn test_stats_collector() {
        let collector = SpreadStatsCollector::new();

        collector.record(10.0, false);
        collector.record(20.0, false);
        collector.record(100.0, true);

        assert!((collector.avg_spread_bps() - 43.33).abs() < 0.1);
        assert!((collector.min_spread_bps() - 10.0).abs() < 0.01);
        assert!((collector.max_spread_bps() - 100.0).abs() < 0.01);
        assert_eq!(collector.stress_event_count(), 1);
    }
}
