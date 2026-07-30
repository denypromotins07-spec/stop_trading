//! Microprice Calculator
//! 
//! Calculates the volume-weighted midprice using L2 order book depth.
//! Adjusts the theoretical mid-price based on relative queue sizes at best bid/ask.

use std::sync::atomic::{AtomicU64, AtomicI64, Ordering};
use thiserror::Error;

/// Errors that can occur in microprice calculation
#[derive(Debug, Error)]
pub enum MicropriceError {
    #[error("Invalid book state: {0}")]
    InvalidBookState(String),
    #[error("Division by zero")]
    DivisionByZero,
    #[error("Overflow detected")]
    Overflow,
}

/// Price level with volume information
#[derive(Debug, Clone, Copy)]
pub struct Level {
    pub price: f64,
    pub volume: f64,
    pub order_count: u32,
}

impl Level {
    pub fn new(price: f64, volume: f64, order_count: u32) -> Self {
        Self {
            price,
            volume,
            order_count,
        }
    }
}

/// Order book snapshot for microprice calculation
#[derive(Debug, Clone)]
pub struct OrderBookSnapshot {
    pub bids: Vec<Level>,
    pub asks: Vec<Level>,
    pub timestamp_ns: u64,
}

impl OrderBookSnapshot {
    pub fn new(bids: Vec<Level>, asks: Vec<Level>, timestamp_ns: u64) -> Self {
        Self {
            bids,
            asks,
            timestamp_ns,
        }
    }

    /// Get best bid
    pub fn best_bid(&self) -> Option<&Level> {
        self.bids.first()
    }

    /// Get best ask
    pub fn best_ask(&self) -> Option<&Level> {
        self.asks.first()
    }

    /// Get mid price
    pub fn mid_price(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some((bid.price + ask.price) / 2.0),
            _ => None,
        }
    }

    /// Get spread
    pub fn spread(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some(ask.price - bid.price),
            _ => None,
        }
    }
}

/// Microprice calculation result
#[derive(Debug, Clone, Copy)]
pub struct MicropriceResult {
    pub timestamp_ns: u64,
    /// Standard mid-price
    pub mid_price: f64,
    /// Volume-weighted microprice
    pub microprice: f64,
    /// Microprice deviation from mid (in basis points)
    pub deviation_bps: f64,
    /// Bid pressure (0.0 to 1.0)
    pub bid_pressure: f64,
    /// Ask pressure (0.0 to 1.0)
    pub ask_pressure: f64,
    /// Predicted short-term direction (-1.0 to 1.0)
    pub predicted_direction: f64,
}

impl MicropriceResult {
    pub fn new(
        timestamp_ns: u64,
        mid_price: f64,
        microprice: f64,
        bid_pressure: f64,
        ask_pressure: f64,
    ) -> Self {
        let deviation_bps = if mid_price > 0.0 {
            ((microprice - mid_price) / mid_price) * 10000.0
        } else {
            0.0
        };

        // Predicted direction based on microprice vs mid
        let predicted_direction = if mid_price > 0.0 {
            (microprice - mid_price) / mid_price * 1000.0 // Scale for readability
        } else {
            0.0
        }.clamp(-1.0, 1.0);

        Self {
            timestamp_ns,
            mid_price,
            microprice,
            deviation_bps,
            bid_pressure,
            ask_pressure,
            predicted_direction,
        }
    }

    /// Check if microprice suggests upward movement
    pub fn is_bullish(&self, threshold_bps: f64) -> bool {
        self.deviation_bps > threshold_bps
    }

    /// Check if microprice suggests downward movement
    pub fn is_bearish(&self, threshold_bps: f64) -> bool {
        self.deviation_bps < -threshold_bps
    }
}

/// Lock-free Microprice Calculator
pub struct MicropriceCalculator {
    /// Last calculated microprice (scaled by 1e9)
    last_microprice: AtomicU64,
    /// Last mid price (scaled by 1e9)
    last_mid_price: AtomicU64,
    /// Last bid pressure (scaled by 1e9)
    last_bid_pressure: AtomicU64,
    /// Last ask pressure (scaled by 1e9)
    last_ask_pressure: AtomicU64,
    /// Last timestamp
    last_timestamp_ns: AtomicU64,
    /// Depth levels to consider
    depth_levels: usize,
    /// Price scale factor
    price_scale: i64,
}

unsafe impl Send for MicropriceCalculator {}
unsafe impl Sync for MicropriceCalculator {}

impl MicropriceCalculator {
    /// Create a new microprice calculator
    pub fn new(depth_levels: usize) -> Self {
        Self {
            last_microprice: AtomicU64::new(0),
            last_mid_price: AtomicU64::new(0),
            last_bid_pressure: AtomicU64::new(0),
            last_ask_pressure: AtomicU64::new(0),
            last_timestamp_ns: AtomicU64::new(0),
            depth_levels,
            price_scale: 1_000_000_000,
        }
    }

    /// Calculate microprice from order book snapshot
    pub fn calculate(&self, book: &OrderBookSnapshot) -> Result<MicropriceResult, MicropriceError> {
        if book.bids.is_empty() || book.asks.is_empty() {
            return Err(MicropriceError::InvalidBookState(
                "Order book must have both bids and asks".to_string(),
            ));
        }

        let best_bid = book.best_bid().unwrap();
        let best_ask = book.best_ask().unwrap();

        // Validate prices
        if best_bid.price <= 0.0 || best_ask.price <= 0.0 {
            return Err(MicropriceError::InvalidBookState(
                "Prices must be positive".to_string(),
            ));
        }

        if best_bid.price >= best_ask.price {
            return Err(MicropriceError::InvalidBookState(
                "Bid price must be less than ask price".to_string(),
            ));
        }

        // Calculate standard mid-price
        let mid_price = (best_bid.price + best_ask.price) / 2.0;

        // Calculate volume-weighted microprice using top N levels
        let (bid_volume_total, bid_weighted_sum) = self.calculate_weighted_sum(&book.bids, true);
        let (ask_volume_total, ask_weighted_sum) = self.calculate_weighted_sum(&book.asks, false);

        if bid_volume_total == 0.0 || ask_volume_total == 0.0 {
            return Err(MicropriceError::DivisionByZero);
        }

        // Calculate bid and ask pressure
        let total_volume = bid_volume_total + ask_volume_total;
        let bid_pressure = bid_volume_total / total_volume;
        let ask_pressure = ask_volume_total / total_volume;

        // Calculate microprice as volume-weighted average of best bid/ask
        // Formula: microprice = (bid_vol * ask_price + ask_vol * bid_price) / (bid_vol + ask_vol)
        // This gives more weight to the side with less liquidity (price moves toward illiquid side)
        let microprice = if total_volume > 0.0 {
            (bid_volume_total * best_ask.price + ask_volume_total * best_bid.price) / total_volume
        } else {
            mid_price
        };

        // Store results atomically
        self.last_microprice.store(
            (microprice * self.price_scale as f64) as u64,
            Ordering::Relaxed,
        );
        self.last_mid_price.store(
            (mid_price * self.price_scale as f64) as u64,
            Ordering::Relaxed,
        );
        self.last_bid_pressure.store(
            (bid_pressure * self.price_scale as f64) as u64,
            Ordering::Relaxed,
        );
        self.last_ask_pressure.store(
            (ask_pressure * self.price_scale as f64) as u64,
            Ordering::Relaxed,
        );
        self.last_timestamp_ns.store(book.timestamp_ns, Ordering::Relaxed);

        Ok(MicropriceResult::new(
            book.timestamp_ns,
            mid_price,
            microprice,
            bid_pressure,
            ask_pressure,
        ))
    }

    /// Calculate weighted sum of volumes and prices
    fn calculate_weighted_sum(&self, levels: &[Level], is_bid: bool) -> (f64, f64) {
        let mut total_volume = 0.0;
        let mut weighted_sum = 0.0;

        let depth = levels.len().min(self.depth_levels);

        for (i, level) in levels.iter().take(depth).enumerate() {
            // Apply exponential decay to further levels
            let weight = 1.0 / (i + 1) as f64;
            let weighted_volume = level.volume * weight;

            total_volume += weighted_volume;
            weighted_sum += weighted_volume * level.price;
        }

        (total_volume, weighted_sum)
    }

    /// Get last calculated microprice
    pub fn last_microprice(&self) -> f64 {
        self.last_microprice.load(Ordering::Relaxed) as f64 / self.price_scale as f64
    }

    /// Get last mid price
    pub fn last_mid_price(&self) -> f64 {
        self.last_mid_price.load(Ordering::Relaxed) as f64 / self.price_scale as f64
    }

    /// Get last bid pressure
    pub fn last_bid_pressure(&self) -> f64 {
        self.last_bid_pressure.load(Ordering::Relaxed) as f64 / self.price_scale as f64
    }

    /// Get last ask pressure
    pub fn last_ask_pressure(&self) -> f64 {
        self.last_ask_pressure.load(Ordering::Relaxed) as f64 / self.price_scale as f64
    }

    /// Get predicted direction
    pub fn predicted_direction(&self) -> f64 {
        let microprice = self.last_microprice();
        let mid_price = self.last_mid_price();

        if mid_price > 0.0 {
            ((microprice - mid_price) / mid_price * 1000.0).clamp(-1.0, 1.0)
        } else {
            0.0
        }
    }
}

impl Default for MicropriceCalculator {
    fn default() -> Self {
        Self::new(5) // Use top 5 levels by default
    }
}

/// Microprice signal for trading decisions
#[derive(Debug, Clone, Copy)]
pub struct MicropriceSignal {
    pub microprice: f64,
    pub mid_price: f64,
    pub deviation_bps: f64,
    pub action: MicropriceAction,
    pub confidence: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MicropriceAction {
    Buy,
    Sell,
    Hold,
}

/// Rolling microprice tracker for trend detection
pub struct RollingMicropriceTracker {
    window_size: usize,
    buffer: crossbeam::queue::SegQueue<MicropriceResult>,
    max_entries: usize,
}

impl RollingMicropriceTracker {
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            buffer: crossbeam::queue::SegQueue::new(),
            max_entries: window_size * 2,
        }
    }

    /// Add a new microprice sample
    pub fn add_sample(&self, result: MicropriceResult) {
        self.buffer.push(result);

        // Prune old entries
        while self.buffer.len() > self.max_entries {
            let _ = self.buffer.pop();
        }
    }

    /// Calculate rolling average microprice
    pub fn rolling_avg_microprice(&self) -> Option<f64> {
        let samples: Vec<MicropriceResult> = self.buffer.iter().cloned().collect();
        if samples.is_empty() {
            return None;
        }

        let sum: f64 = samples.iter().map(|r| r.microprice).sum();
        Some(sum / samples.len() as f64)
    }

    /// Detect microprice trend
    pub fn detect_trend(&self) -> Option<MicropriceTrend> {
        let samples: Vec<MicropriceResult> = self.buffer.iter().cloned().collect();
        if samples.len() < 3 {
            return None;
        }

        let recent: Vec<f64> = samples.iter().take(samples.len() / 2).map(|r| r.microprice).collect();
        let older: Vec<f64> = samples.iter().skip(samples.len() / 2).map(|r| r.microprice).collect();

        let recent_avg = recent.iter().sum::<f64>() / recent.len() as f64;
        let older_avg = older.iter().sum::<f64>() / older.len() as f64;

        let change_pct = if older_avg > 0.0 {
            (recent_avg - older_avg) / older_avg * 100.0
        } else {
            0.0
        };

        if change_pct > 0.01 {
            Some(MicropriceTrend::Rising)
        } else if change_pct < -0.01 {
            Some(MicropriceTrend::Falling)
        } else {
            Some(MicropriceTrend::Flat)
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MicropriceTrend {
    Rising,
    Falling,
    Flat,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_microprice_calculation() {
        let calc = MicropriceCalculator::new(3);

        let book = OrderBookSnapshot::new(
            vec![
                Level::new(99.9, 100.0, 5),
                Level::new(99.8, 150.0, 8),
            ],
            vec![
                Level::new(100.1, 50.0, 3),
                Level::new(100.2, 75.0, 4),
            ],
            1000,
        );

        let result = calc.calculate(&book).unwrap();

        // Mid price should be (99.9 + 100.1) / 2 = 100.0
        assert!((result.mid_price - 100.0).abs() < 0.001);

        // Since ask volume is less than bid volume, microprice should be above mid
        // (price tends to move toward illiquid side)
        assert!(result.microprice > result.mid_price);
    }

    #[test]
    fn test_pressure_calculation() {
        let calc = MicropriceCalculator::new(1);

        let book = OrderBookSnapshot::new(
            vec![Level::new(99.0, 200.0, 10)],
            vec![Level::new(101.0, 100.0, 5)],
            1000,
        );

        let result = calc.calculate(&book).unwrap();

        // Bid volume is double ask volume
        assert!(result.bid_pressure > 0.6);
        assert!(result.ask_pressure < 0.4);
    }

    #[test]
    fn test_invalid_book() {
        let calc = MicropriceCalculator::new(3);

        // Empty book
        let book = OrderBookSnapshot::new(vec![], vec![], 1000);
        assert!(calc.calculate(&book).is_err());

        // Crossed book
        let book = OrderBookSnapshot::new(
            vec![Level::new(101.0, 100.0, 5)],
            vec![Level::new(99.0, 100.0, 5)],
            1000,
        );
        assert!(calc.calculate(&book).is_err());
    }
}
