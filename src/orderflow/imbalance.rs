//! Order Flow Imbalance Calculator
//! 
//! Analyzes bid/ask volume ratios at the top of the book to detect
//! absorption and exhaustion patterns signaling potential reversals.

use std::sync::atomic::{AtomicU64, AtomicI64, Ordering};
use thiserror::Error;

/// Errors that can occur in imbalance calculation
#[derive(Debug, Error)]
pub enum ImbalanceError {
    #[error("Invalid book state: {0}")]
    InvalidBookState(String),
    #[error("Division by zero in ratio calculation")]
    DivisionByZero,
    #[error("Overflow detected")]
    Overflow,
}

/// Represents a price level in the order book
#[derive(Debug, Clone, Copy)]
pub struct PriceLevel {
    pub price: f64,
    pub volume: f64,
    pub order_count: u32,
}

impl PriceLevel {
    pub fn new(price: f64, volume: f64, order_count: u32) -> Self {
        Self {
            price,
            volume,
            order_count,
        }
    }
}

/// Top of book state for imbalance calculation
#[derive(Debug, Clone, Copy)]
pub struct TopOfBook {
    pub best_bid: PriceLevel,
    pub best_ask: PriceLevel,
    pub timestamp_ns: u64,
}

impl TopOfBook {
    pub fn new(best_bid: PriceLevel, best_ask: PriceLevel, timestamp_ns: u64) -> Self {
        Self {
            best_bid,
            best_ask,
            timestamp_ns,
        }
    }

    /// Calculate the spread in absolute terms
    pub fn spread(&self) -> f64 {
        self.best_ask.price - self.best_bid.price
    }

    /// Calculate the spread in percentage terms (relative to mid-price)
    pub fn spread_pct(&self) -> f64 {
        let mid = (self.best_bid.price + self.best_ask.price) / 2.0;
        if mid == 0.0 {
            return 0.0;
        }
        (self.spread() / mid) * 100.0
    }

    /// Calculate the mid-price
    pub fn mid_price(&self) -> f64 {
        (self.best_bid.price + self.best_ask.price) / 2.0
    }
}

/// Order flow imbalance metrics
#[derive(Debug, Clone, Copy)]
pub struct ImbalanceMetrics {
    pub timestamp_ns: u64,
    /// Bid-Ask Volume Ratio (BAVR): bid_volume / (bid_volume + ask_volume)
    pub bavr: f64,
    /// Order Book Imbalance (OBI): (bid_volume - ask_volume) / (bid_volume + ask_volume)
    pub obi: f64,
    /// Weighted imbalance considering depth
    pub weighted_imbalance: f64,
    /// Pressure indicator: positive = buy pressure, negative = sell pressure
    pub pressure: f64,
    /// Total volume at top of book
    pub total_volume: f64,
}

impl ImbalanceMetrics {
    pub fn new(
        timestamp_ns: u64,
        bavr: f64,
        obi: f64,
        weighted_imbalance: f64,
        pressure: f64,
        total_volume: f64,
    ) -> Self {
        Self {
            timestamp_ns,
            bavr,
            obi,
            weighted_imbalance,
            pressure,
            total_volume,
        }
    }

    /// Check if imbalance indicates extreme buy pressure
    pub fn is_extreme_buy(&self, threshold: f64) -> bool {
        self.obi > threshold || self.bavr > (0.5 + threshold / 2.0)
    }

    /// Check if imbalance indicates extreme sell pressure
    pub fn is_extreme_sell(&self, threshold: f64) -> bool {
        self.obi < -threshold || self.bavr < (0.5 - threshold / 2.0)
    }
}

/// Signal type detected by the imbalance calculator
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImbalanceSignal {
    None,
    Absorption,      // Large orders being absorbed without price movement
    Exhaustion,      // Volume drying up on one side
    BuyPressure,     // Strong buying imbalance
    SellPressure,    // Strong selling imbalance
    ReversalLong,    // Potential reversal to upside
    ReversalShort,   // Potential reversal to downside
}

/// Lock-free Order Flow Imbalance Calculator
/// 
/// Analyzes bid/ask volume ratios at the top of the book to detect
/// absorption and exhaustion patterns that signal potential reversals.
pub struct OrderFlowImbalance {
    /// Running sum of bid volume (scaled)
    bid_volume_sum: AtomicI64,
    /// Running sum of ask volume (scaled)
    ask_volume_sum: AtomicI64,
    /// Count of updates
    update_count: AtomicU64,
    /// Last calculated OBI
    last_obi: AtomicI64, // Scaled by 1e9
    /// Last timestamp
    last_timestamp_ns: AtomicU64,
    /// Volume scale factor
    volume_scale: i64,
    /// Lookback window for rolling calculations (nanoseconds)
    lookback_window_ns: AtomicU64,
}

impl OrderFlowImbalance {
    /// Create a new imbalance calculator
    pub fn new() -> Self {
        Self {
            bid_volume_sum: AtomicI64::new(0),
            ask_volume_sum: AtomicI64::new(0),
            update_count: AtomicU64::new(0),
            last_obi: AtomicI64::new(0),
            last_timestamp_ns: AtomicU64::new(0),
            volume_scale: 1_000_000_000,
            lookback_window_ns: AtomicU64::new(60_000_000_000), // 60 seconds default
        }
    }

    /// Create with custom volume scaling
    pub fn with_scale(volume_scale: i64) -> Self {
        Self {
            bid_volume_sum: AtomicI64::new(0),
            ask_volume_sum: AtomicI64::new(0),
            update_count: AtomicU64::new(0),
            last_obi: AtomicI64::new(0),
            last_timestamp_ns: AtomicU64::new(0),
            volume_scale,
            lookback_window_ns: AtomicU64::new(60_000_000_000),
        }
    }

    /// Set the lookback window for rolling calculations
    pub fn set_lookback_window_ms(&self, window_ms: u64) {
        self.lookback_window_ns.store(window_ms * 1_000_000, Ordering::Relaxed);
    }

    /// Process a top-of-book update (lock-free)
    pub fn process_update(&self, tob: &TopOfBook) -> Result<ImbalanceMetrics, ImbalanceError> {
        if tob.best_bid.volume < 0.0 || tob.best_ask.volume < 0.0 {
            return Err(ImbalanceError::InvalidBookState(
                "Negative volume detected".to_string(),
            ));
        }

        let bid_vol_scaled = (tob.best_bid.volume * self.volume_scale as f64) as i64;
        let ask_vol_scaled = (tob.best_ask.volume * self.volume_scale as f64) as i64;

        // Calculate metrics
        let total_vol_scaled = bid_vol_scaled.checked_add(ask_vol_scaled)
            .ok_or(ImbalanceError::Overflow)?;

        let bavr = if total_vol_scaled == 0 {
            0.5 // Neutral when no volume
        } else {
            bid_vol_scaled as f64 / total_vol_scaled as f64
        };

        let obi = if total_vol_scaled == 0 {
            0.0
        } else {
            (bid_vol_scaled - ask_vol_scaled) as f64 / total_vol_scaled as f64
        };

        // Calculate weighted imbalance (considering relative size)
        let volume_ratio = if tob.best_ask.volume > 0.0 {
            tob.best_bid.volume / tob.best_ask.volume
        } else if tob.best_bid.volume > 0.0 {
            f64::MAX
        } else {
            1.0
        };

        let weighted_imbalance = if volume_ratio.is_finite() {
            (volume_ratio - 1.0) / (volume_ratio + 1.0)
        } else {
            1.0
        };

        // Pressure calculation: combines OBI with spread dynamics
        let spread = tob.spread();
        let mid = tob.mid_price();
        let spread_factor = if mid > 0.0 { spread / mid } else { 0.0 };
        let pressure = obi * (1.0 - spread_factor.min(0.1) * 10.0);

        // Store atomically
        self.bid_volume_sum.fetch_add(bid_vol_scaled, Ordering::Relaxed);
        self.ask_volume_sum.fetch_add(ask_vol_scaled, Ordering::Relaxed);
        self.update_count.fetch_add(1, Ordering::Relaxed);
        self.last_obi.store((obi * self.volume_scale as f64) as i64, Ordering::Relaxed);
        self.last_timestamp_ns.store(tob.timestamp_ns, Ordering::Relaxed);

        Ok(ImbalanceMetrics::new(
            tob.timestamp_ns,
            bavr,
            obi,
            weighted_imbalance,
            pressure,
            tob.best_bid.volume + tob.best_ask.volume,
        ))
    }

    /// Detect absorption pattern
    /// 
    /// Absorption occurs when large volume is present but price doesn't move,
    /// indicating hidden liquidity or iceberg orders.
    pub fn detect_absorption(&self, metrics: &ImbalanceMetrics, price_change: f64) -> bool {
        let obi = self.last_obi.load(Ordering::Relaxed) as f64 / self.volume_scale as f64;
        
        // High imbalance but minimal price movement suggests absorption
        let high_imbalance = obi.abs() > 0.7;
        let low_price_movement = price_change.abs() < 0.001; // Less than 0.1%
        let high_volume = metrics.total_volume > 100.0; // Significant volume

        high_imbalance && low_price_movement && high_volume
    }

    /// Detect exhaustion pattern
    /// 
    /// Exhaustion occurs when volume on one side dries up significantly,
    /// often preceding a reversal.
    pub fn detect_exhaustion(&self, metrics: &ImbalanceMetrics, avg_volume: f64) -> bool {
        let current_ratio = metrics.total_volume / avg_volume.max(1.0);
        
        // Volume significantly below average with extreme imbalance
        let low_volume = current_ratio < 0.3; // Below 30% of average
        let extreme_imbalance = metrics.obi.abs() > 0.8;

        low_volume && extreme_imbalance
    }

    /// Generate trading signal based on imbalance analysis
    pub fn generate_signal(
        &self,
        metrics: &ImbalanceMetrics,
        price_change: f64,
        avg_volume: f64,
    ) -> ImbalanceSignal {
        if self.detect_absorption(metrics, price_change) {
            // Absorption often leads to reversal against the pressure
            if metrics.obi > 0.5 {
                return ImbalanceSignal::ReversalShort; // Buying absorbed, expect down
            } else {
                return ImbalanceSignal::ReversalLong; // Selling absorbed, expect up
            }
        }

        if self.detect_exhaustion(metrics, avg_volume) {
            // Exhaustion suggests trend ending
            if metrics.obi > 0.5 {
                return ImbalanceSignal::Exhaustion; // Buy side exhausted
            } else {
                return ImbalanceSignal::Exhaustion; // Sell side exhausted
            }
        }

        // Standard pressure signals
        if metrics.obi > 0.6 {
            ImbalanceSignal::BuyPressure
        } else if metrics.obi < -0.6 {
            ImbalanceSignal::SellPressure
        } else {
            ImbalanceSignal::None
        }
    }

    /// Get current OBI value
    pub fn current_obi(&self) -> f64 {
        self.last_obi.load(Ordering::Relaxed) as f64 / self.volume_scale as f64
    }

    /// Reset all counters
    pub fn reset(&self) {
        self.bid_volume_sum.store(0, Ordering::Relaxed);
        self.ask_volume_sum.store(0, Ordering::Relaxed);
        self.update_count.store(0, Ordering::Relaxed);
        self.last_obi.store(0, Ordering::Relaxed);
        self.last_timestamp_ns.store(0, Ordering::Relaxed);
    }

    /// Get statistics about processed updates
    pub fn stats(&self) -> ImbalanceStats {
        ImbalanceStats {
            total_updates: self.update_count.load(Ordering::Relaxed),
            total_bid_volume: self.bid_volume_sum.load(Ordering::Relaxed) as f64 / self.volume_scale as f64,
            total_ask_volume: self.ask_volume_sum.load(Ordering::Relaxed) as f64 / self.volume_scale as f64,
            current_obi: self.current_obi(),
        }
    }
}

impl Default for OrderFlowImbalance {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics from the imbalance calculator
#[derive(Debug, Clone, Copy)]
pub struct ImbalanceStats {
    pub total_updates: u64,
    pub total_bid_volume: f64,
    pub total_ask_volume: f64,
    pub current_obi: f64,
}

/// Rolling window calculator for imbalance metrics
pub struct RollingImbalanceCalculator {
    window_size: usize,
    buffer: crossbeam::queue::SegQueue<ImbalanceMetrics>,
    max_entries: usize,
}

impl RollingImbalanceCalculator {
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            buffer: crossbeam::queue::SegQueue::new(),
            max_entries: window_size * 2, // Allow some overflow
        }
    }

    /// Add a new metrics sample to the rolling window
    pub fn add_sample(&self, metrics: ImbalanceMetrics) {
        self.buffer.push(metrics);

        // Prune old entries if too many
        while self.buffer.len() > self.max_entries {
            let _ = self.buffer.pop();
        }
    }

    /// Calculate rolling average OBI
    pub fn rolling_avg_obi(&self) -> Option<f64> {
        let samples: Vec<ImbalanceMetrics> = self.buffer.iter().cloned().collect();
        if samples.is_empty() {
            return None;
        }

        let sum: f64 = samples.iter().map(|m| m.obi).sum();
        Some(sum / samples.len() as f64)
    }

    /// Calculate rolling standard deviation of OBI
    pub fn rolling_std_obi(&self) -> Option<f64> {
        let samples: Vec<ImbalanceMetrics> = self.buffer.iter().cloned().collect();
        if samples.len() < 2 {
            return None;
        }

        let mean = samples.iter().map(|m| m.obi).sum::<f64>() / samples.len() as f64;
        let variance = samples.iter()
            .map(|m| (m.obi - mean).powi(2))
            .sum::<f64>() / (samples.len() - 1) as f64;

        Some(variance.sqrt())
    }

    /// Detect if current OBI is statistically significant (Z-score > 2)
    pub fn is_statistically_significant(&self, current_obi: f64) -> Option<bool> {
        let mean = self.rolling_avg_obi()?;
        let std = self.rolling_std_obi()?;

        if std == 0.0 {
            return Some(false);
        }

        let z_score = (current_obi - mean).abs() / std;
        Some(z_score > 2.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_imbalance_calculation() {
        let calc = OrderFlowImbalance::new();
        
        let tob = TopOfBook::new(
            PriceLevel::new(49999.0, 10.0, 5),
            PriceLevel::new(50001.0, 5.0, 3),
            1000,
        );

        let metrics = calc.process_update(&tob).unwrap();
        
        // Bid volume is double ask volume, so BAVR should be ~0.67
        assert!(metrics.bavr > 0.65);
        assert!(metrics.obi > 0.3);
        assert_eq!(metrics.signal_type(), ImbalanceSignal::BuyPressure);
    }

    #[test]
    fn test_top_of_book_spread() {
        let tob = TopOfBook::new(
            PriceLevel::new(49990.0, 10.0, 5),
            PriceLevel::new(50010.0, 5.0, 3),
            1000,
        );

        assert_eq!(tob.spread(), 20.0);
        assert!((tob.spread_pct() - 0.04).abs() < 0.01);
        assert_eq!(tob.mid_price(), 50000.0);
    }

    #[test]
    fn test_absorption_detection() {
        let calc = OrderFlowImbalance::new();
        
        // Simulate high imbalance with no price movement
        let tob = TopOfBook::new(
            PriceLevel::new(49999.0, 100.0, 50),
            PriceLevel::new(50001.0, 10.0, 5),
            1000,
        );

        let metrics = calc.process_update(&tob).unwrap();
        let is_absorption = calc.detect_absorption(&metrics, 0.0001);
        
        // Should detect absorption due to high imbalance, high volume, low price movement
        assert!(is_absorption);
    }
}

impl ImbalanceMetrics {
    fn signal_type(&self) -> ImbalanceSignal {
        if self.obi > 0.6 {
            ImbalanceSignal::BuyPressure
        } else if self.obi < -0.6 {
            ImbalanceSignal::SellPressure
        } else {
            ImbalanceSignal::None
        }
    }
}
