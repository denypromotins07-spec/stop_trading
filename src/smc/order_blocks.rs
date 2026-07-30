//! Smart Money Concepts - Order Blocks Detector
//! 
//! Algorithmic detector for institutional Order Blocks and Mitigation blocks.
//! Tracks the origin of strong impulsive moves to identify high-probability
//! premium/discount entry zones.

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use thiserror::Error;

/// Errors that can occur in order block detection
#[derive(Debug, Error)]
pub enum OrderBlockError {
    #[error("Invalid price data: {0}")]
    InvalidPriceData(String),
    #[error("Insufficient data points")]
    InsufficientData,
    #[error("Overflow detected")]
    Overflow,
}

/// Candle representation for order block analysis
#[derive(Debug, Clone, Copy)]
pub struct Candle {
    pub timestamp_ns: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

impl Candle {
    pub fn new(
        timestamp_ns: u64,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
    ) -> Result<Self, OrderBlockError> {
        if open <= 0.0 || high <= 0.0 || low <= 0.0 || close <= 0.0 {
            return Err(OrderBlockError::InvalidPriceData(
                "Prices must be positive".to_string(),
            ));
        }
        if high < low || high < open || high < close || low > open || low > close {
            return Err(OrderBlockError::InvalidPriceData(
                "Invalid OHLC relationship".to_string(),
            ));
        }

        Ok(Self {
            timestamp_ns,
            open,
            high,
            low,
            close,
            volume,
        })
    }

    /// Check if candle is bullish
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }

    /// Check if candle is bearish
    pub fn is_bearish(&self) -> bool {
        self.close < self.open
    }

    /// Get candle body size
    pub fn body_size(&self) -> f64 {
        (self.close - self.open).abs()
    }

    /// Get candle range (high - low)
    pub fn range(&self) -> f64 {
        self.high - self.low
    }

    /// Check if candle is impulsive (large body relative to range)
    pub fn is_impulsive(&self, threshold: f64) -> bool {
        let range = self.range();
        if range == 0.0 {
            return false;
        }
        self.body_size() / range > threshold
    }
}

/// Type of order block
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderBlockType {
    Bullish,
    Bearish,
    MitigationBullish,
    MitigationBearish,
}

/// Detected order block with metadata
#[derive(Debug, Clone, Copy)]
pub struct OrderBlock {
    pub block_type: OrderBlockType,
    pub high: f64,
    pub low: f64,
    pub open: f64,
    pub close: f64,
    pub timestamp_ns: u64,
    pub strength: f64,      // 0.0 to 1.0 based on impulsiveness
    pub tested: bool,       // Whether price has returned to test the block
    pub mitigated: bool,    // Whether the block has been fully mitigated
    pub consequent_encroachment: f64, // 50% level of the block
}

impl OrderBlock {
    pub fn new(
        block_type: OrderBlockType,
        high: f64,
        low: f64,
        open: f64,
        close: f64,
        timestamp_ns: u64,
        strength: f64,
    ) -> Self {
        Self {
            block_type,
            high,
            low,
            open,
            close,
            timestamp_ns,
            strength,
            tested: false,
            mitigated: false,
            consequent_encroachment: (high + low) / 2.0,
        }
    }

    /// Check if current price is within the order block zone
    pub fn contains_price(&self, price: f64) -> bool {
        price >= self.low && price <= self.high
    }

    /// Calculate distance from current price to the block
    pub fn distance_from(&self, price: f64) -> f64 {
        if price > self.high {
            price - self.high
        } else if price < self.low {
            self.low - price
        } else {
            0.0
        }
    }

    /// Mark the block as tested
    pub fn mark_tested(&mut self) {
        self.tested = true;
    }

    /// Mark the block as mitigated
    pub fn mark_mitigated(&mut self) {
        self.mitigated = true;
    }
}

/// Premium/Discount zone classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZoneType {
    Premium,    // Upper 50% of range (sell zones)
    Discount,   // Lower 50% of range (buy zones)
    Equilibrium, // Middle 50% (neutral)
}

/// Order Block detection engine
pub struct OrderBlockDetector {
    /// Maximum number of order blocks to track
    max_blocks: usize,
    /// Minimum impulsiveness threshold (0.0 to 1.0)
    impulse_threshold: f64,
    /// Current price for testing blocks
    current_price: AtomicU64, // Scaled by 1e9
    /// Price scale factor
    price_scale: i64,
    /// Active flag
    active: AtomicBool,
}

impl OrderBlockDetector {
    /// Create a new order block detector
    pub fn new(max_blocks: usize, impulse_threshold: f64) -> Self {
        Self {
            max_blocks,
            impulse_threshold,
            current_price: AtomicU64::new(0),
            price_scale: 1_000_000_000,
            active: AtomicBool::new(true),
        }
    }

    /// Set the price scale factor
    pub fn set_price_scale(&self, scale: i64) {
        self.price_scale = scale;
    }

    /// Update current price
    pub fn update_price(&self, price: f64) {
        let scaled = (price * self.price_scale as f64) as u64;
        self.current_price.store(scaled, Ordering::Relaxed);
    }

    /// Get current price
    pub fn get_current_price(&self) -> f64 {
        self.current_price.load(Ordering::Relaxed) as f64 / self.price_scale as f64
    }

    /// Detect order blocks from a series of candles
    pub fn detect_blocks(&self, candles: &[Candle]) -> Result<Vec<OrderBlock>, OrderBlockError> {
        if candles.len() < 3 {
            return Err(OrderBlockError::InsufficientData);
        }

        let mut blocks = Vec::new();
        let current_price = self.get_current_price();

        // Look for 3-candle patterns indicating order blocks
        for i in 1..candles.len() - 1 {
            let prev = &candles[i - 1];
            let curr = &candles[i];
            let next = &candles[i + 1];

            // Bullish order block: strong bear candle followed by strong bullish move
            if self.is_bullish_order_block(prev, curr, next) {
                let strength = self.calculate_strength(curr, next);
                if strength >= self.impulse_threshold {
                    let mut block = OrderBlock::new(
                        OrderBlockType::Bullish,
                        prev.high,
                        prev.low,
                        prev.open,
                        prev.close,
                        prev.timestamp_ns,
                        strength,
                    );

                    // Check if already tested or mitigated
                    if current_price <= block.high && current_price >= block.low {
                        block.mark_tested();
                    }
                    if current_price > prev.high && next.close > prev.high {
                        block.mark_mitigated();
                    }

                    blocks.push(block);
                }
            }

            // Bearish order block: strong bullish candle followed by strong bearish move
            if self.is_bearish_order_block(prev, curr, next) {
                let strength = self.calculate_strength(curr, next);
                if strength >= self.impulse_threshold {
                    let mut block = OrderBlock::new(
                        OrderBlockType::Bearish,
                        prev.high,
                        prev.low,
                        prev.open,
                        prev.close,
                        prev.timestamp_ns,
                        strength,
                    );

                    if current_price <= block.high && current_price >= block.low {
                        block.mark_tested();
                    }
                    if current_price < prev.low && next.close < prev.low {
                        block.mark_mitigated();
                    }

                    blocks.push(block);
                }
            }
        }

        // Limit number of blocks returned (keep strongest/most recent)
        blocks.truncate(self.max_blocks);

        Ok(blocks)
    }

    /// Detect mitigation blocks (order blocks that have been partially filled)
    pub fn detect_mitigation_blocks(
        &self,
        candles: &[Candle],
        existing_blocks: &[OrderBlock],
    ) -> Vec<OrderBlock> {
        let mut mitigation_blocks = Vec::new();
        let current_price = self.get_current_price();

        for block in existing_blocks {
            // Check if price has returned to the block after moving away
            let was_tested = block.distance_from(current_price) == 0.0;

            if was_tested && !block.mitigated {
                // Price is currently in the block - potential mitigation
                let mit_type = match block.block_type {
                    OrderBlockType::Bullish => OrderBlockType::MitigationBullish,
                    OrderBlockType::Bearish => OrderBlockType::MitigationBearish,
                    _ => continue,
                };

                let mut mit_block = OrderBlock::new(
                    mit_type,
                    block.high,
                    block.low,
                    block.open,
                    block.close,
                    block.timestamp_ns,
                    block.strength,
                );
                mit_block.tested = true;
                mitigation_blocks.push(mit_block);
            }
        }

        mitigation_blocks
    }

    /// Calculate premium/discount zone for a given range
    pub fn calculate_premium_discount(
        &self,
        swing_high: f64,
        swing_low: f64,
        price: f64,
    ) -> ZoneType {
        if swing_high <= swing_low {
            return ZoneType::Equilibrium;
        }

        let range = swing_high - swing_low;
        let midpoint = swing_low + range / 2.0;

        if price > midpoint + range * 0.1 {
            ZoneType::Premium
        } else if price < midpoint - range * 0.1 {
            ZoneType::Discount
        } else {
            ZoneType::Equilibrium
        }
    }

    /// Check if price is in a premium zone (good for selling)
    pub fn is_in_premium(&self, swing_high: f64, swing_low: f64, price: f64) -> bool {
        self.calculate_premium_discount(swing_high, swing_low, price) == ZoneType::Premium
    }

    /// Check if price is in a discount zone (good for buying)
    pub fn is_in_discount(&self, swing_high: f64, swing_low: f64, price: f64) -> bool {
        self.calculate_premium_discount(swing_high, swing_low, price) == ZoneType::Discount
    }

    /// Activate/deactivate the detector
    pub fn set_active(&self, active: bool) {
        self.active.store(active, Ordering::Relaxed);
    }

    /// Check if detector is active
    pub fn is_active(&self) -> bool {
        self.active.load(Ordering::Relaxed)
    }

    // Internal helper methods

    fn is_bullish_order_block(&self, prev: &Candle, curr: &Candle, next: &Candle) -> bool {
        // Previous candle is bearish
        prev.is_bearish()
            // Current candle gaps down or opens near prev low
            && (curr.open <= prev.low || (curr.open - prev.low).abs() < prev.range() * 0.1)
            // Strong bullish move follows
            && next.is_bullish()
            && next.close > prev.high
    }

    fn is_bearish_order_block(&self, prev: &Candle, curr: &Candle, next: &Candle) -> bool {
        // Previous candle is bullish
        prev.is_bullish()
            // Current candle gaps up or opens near prev high
            && (curr.open >= prev.high || (curr.open - prev.high).abs() < prev.range() * 0.1)
            // Strong bearish move follows
            && next.is_bearish()
            && next.close < prev.low
    }

    fn calculate_strength(&self, impulse_candle: &Candle, follow_candle: &Candle) -> f64 {
        let impulse_strength = impulse_candle.body_size() / impulse_candle.range().max(0.0001);
        let follow_through = follow_candle.body_size() / follow_candle.range().max(0.0001);
        
        // Weight impulse more than follow-through
        (impulse_strength * 0.7 + follow_through * 0.3).min(1.0)
    }
}

impl Default for OrderBlockDetector {
    fn default() -> Self {
        Self::new(10, 0.6) // Track 10 blocks, 60% impulsiveness threshold
    }
}

/// Order block signal for trading decisions
#[derive(Debug, Clone, Copy)]
pub struct OrderBlockSignal {
    pub block: OrderBlock,
    pub action: OrderBlockAction,
    pub confidence: f64,
    pub target_price: f64,
    pub stop_loss: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderBlockAction {
    EnterLong,
    EnterShort,
    ExitLong,
    ExitShort,
    Wait,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_candle_properties() {
        let candle = Candle::new(1000, 100.0, 105.0, 98.0, 104.0, 1000.0).unwrap();
        
        assert!(candle.is_bullish());
        assert!(!candle.is_bearish());
        assert_eq!(candle.body_size(), 4.0);
        assert_eq!(candle.range(), 7.0);
    }

    #[test]
    fn test_order_block_detection() {
        let detector = OrderBlockDetector::new(10, 0.5);
        
        // Create a bullish order block pattern
        let candles = vec![
            Candle::new(1000, 100.0, 102.0, 98.0, 99.0, 100.0).unwrap(), // Bearish
            Candle::new(2000, 99.0, 99.5, 97.0, 98.0, 150.0).unwrap(),   // Setup
            Candle::new(3000, 98.0, 105.0, 97.5, 104.0, 200.0).unwrap(), // Impulse up
        ];

        detector.update_price(103.0);
        let blocks = detector.detect_blocks(&candles).unwrap();
        
        // Should detect at least one bullish order block
        assert!(!blocks.is_empty());
    }

    #[test]
    fn test_premium_discount_zones() {
        let detector = OrderBlockDetector::default();
        
        let swing_high = 100.0;
        let swing_low = 50.0;
        
        assert_eq!(detector.calculate_premium_discount(swing_high, swing_low, 80.0), ZoneType::Premium);
        assert_eq!(detector.calculate_premium_discount(swing_high, swing_low, 30.0), ZoneType::Discount);
        assert_eq!(detector.calculate_premium_discount(swing_high, swing_low, 55.0), ZoneType::Equilibrium);
    }
}
