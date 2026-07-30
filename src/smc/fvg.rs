//! Fair Value Gap (FVG) and Liquidity Void Identifier
//! 
//! Identifies imbalanced price deliveries where the market is likely
//! to reprice and fill the liquidity gap.

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use thiserror::Error;

/// Errors that can occur in FVG detection
#[derive(Debug, Error)]
pub enum FvgError {
    #[error("Invalid price data: {0}")]
    InvalidPriceData(String),
    #[error("Insufficient data points")]
    InsufficientData,
    #[error("Overflow detected")]
    Overflow,
}

/// Candle representation for FVG analysis
#[derive(Debug, Clone, Copy)]
pub struct FvgCandle {
    pub timestamp_ns: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
}

impl FvgCandle {
    pub fn new(
        timestamp_ns: u64,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
    ) -> Result<Self, FvgError> {
        if open <= 0.0 || high <= 0.0 || low <= 0.0 || close <= 0.0 {
            return Err(FvgError::InvalidPriceData(
                "Prices must be positive".to_string(),
            ));
        }
        if high < low {
            return Err(FvgError::InvalidPriceData(
                "High must be >= Low".to_string(),
            ));
        }

        Ok(Self {
            timestamp_ns,
            open,
            high,
            low,
            close,
        })
    }

    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }

    pub fn is_bearish(&self) -> bool {
        self.close < self.open
    }

    pub fn wick_top(&self) -> f64 {
        if self.is_bullish() {
            self.high - self.close
        } else {
            self.high - self.open
        }
    }

    pub fn wick_bottom(&self) -> f64 {
        if self.is_bullish() {
            self.open - self.low
        } else {
            self.close - self.low
        }
    }

    pub fn body(&self) -> f64 {
        (self.close - self.open).abs()
    }
}

/// Fair Value Gap type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FvgType {
    Bullish, // Price gapped up, expect pullback to fill
    Bearish, // Price gapped down, expect rally to fill
}

/// Detected Fair Value Gap
#[derive(Debug, Clone, Copy)]
pub struct FairValueGap {
    pub fvg_type: FvgType,
    /// Upper bound of the gap
    pub high: f64,
    /// Lower bound of the gap
    pub low: f64,
    /// Midpoint (consequent encroachment)
    pub midpoint: f64,
    /// Timestamp when gap was formed
    pub timestamp_ns: u64,
    /// Size of the gap
    pub size: f64,
    /// Whether the gap has been filled
    pub filled: bool,
    /// Fill percentage (0.0 to 1.0)
    pub fill_pct: f64,
    /// Number of times price has tested the gap
    pub test_count: u32,
}

impl FairValueGap {
    pub fn new(
        fvg_type: FvgType,
        high: f64,
        low: f64,
        timestamp_ns: u64,
    ) -> Self {
        let midpoint = (high + low) / 2.0;
        let size = high - low;

        Self {
            fvg_type,
            high,
            low,
            midpoint,
            timestamp_ns,
            size,
            filled: false,
            fill_pct: 0.0,
            test_count: 0,
        }
    }

    /// Check if current price is within the FVG
    pub fn contains_price(&self, price: f64) -> bool {
        price >= self.low && price <= self.high
    }

    /// Update fill status based on current price
    pub fn update_fill_status(&mut self, current_price: f64) {
        match self.fvg_type {
            FvgType::Bullish => {
                // Bullish FVG fills when price drops to or below the low
                if current_price <= self.low {
                    self.filled = true;
                    self.fill_pct = 1.0;
                } else if current_price <= self.high {
                    self.fill_pct = (self.high - current_price) / self.size.max(0.0001);
                }
            }
            FvgType::Bearish => {
                // Bearish FVG fills when price rises to or above the high
                if current_price >= self.high {
                    self.filled = true;
                    self.fill_pct = 1.0;
                } else if current_price >= self.low {
                    self.fill_pct = (current_price - self.low) / self.size.max(0.0001);
                }
            }
        }
    }

    /// Increment test count
    pub fn mark_tested(&mut self) {
        self.test_count += 1;
    }

    /// Calculate distance from current price to the FVG
    pub fn distance_from(&self, price: f64) -> f64 {
        if price > self.high {
            price - self.high
        } else if price < self.low {
            self.low - price
        } else {
            0.0
        }
    }

    /// Get the optimal entry point (usually the midpoint/CE)
    pub fn optimal_entry(&self) -> f64 {
        self.midpoint
    }
}

/// Liquidity Void - a more extreme form of FVG
#[derive(Debug, Clone, Copy)]
pub struct LiquidityVoid {
    pub high: f64,
    pub low: f64,
    pub timestamp_ns: u64,
    pub volume_imbalance: f64,
    pub filled: bool,
}

impl LiquidityVoid {
    pub fn new(high: f64, low: f64, timestamp_ns: u64, volume_imbalance: f64) -> Self {
        Self {
            high,
            low,
            timestamp_ns,
            volume_imbalance,
            filled: false,
        }
    }
}

/// Fair Value Gap Detector
pub struct FvgDetector {
    /// Minimum FVG size threshold (as percentage of price)
    min_size_pct: f64,
    /// Maximum number of FVGs to track
    max_fvgs: usize,
    /// Current price
    current_price: AtomicU64,
    /// Price scale factor
    price_scale: i64,
    /// Active flag
    active: AtomicBool,
}

impl FvgDetector {
    /// Create a new FVG detector
    pub fn new(min_size_pct: f64, max_fvgs: usize) -> Self {
        Self {
            min_size_pct,
            max_fvgs,
            current_price: AtomicU64::new(0),
            price_scale: 1_000_000_000,
            active: AtomicBool::new(true),
        }
    }

    /// Set price scale factor
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

    /// Detect FVGs from candle data
    pub fn detect_fvgs(&self, candles: &[FvgCandle]) -> Result<Vec<FairValueGap>, FvgError> {
        if candles.len() < 3 {
            return Err(FvgError::InsufficientData);
        }

        let mut fvgs = Vec::new();
        let current_price = self.get_current_price();

        for i in 1..candles.len() - 1 {
            let prev = &candles[i - 1];
            let curr = &candles[i];
            let next = &candles[i + 1];

            // Bullish FVG: gap between prev high and next low with current candle being bullish
            if curr.is_bullish() && next.low > prev.high {
                let gap_high = next.low;
                let gap_low = prev.high;
                let gap_size = gap_high - gap_low;

                // Check if gap meets minimum size threshold
                let mid_price = (gap_high + gap_low) / 2.0;
                if mid_price > 0.0 && (gap_size / mid_price) >= self.min_size_pct / 100.0 {
                    let mut fvg = FairValueGap::new(FvgType::Bullish, gap_high, gap_low, curr.timestamp_ns);
                    fvg.update_fill_status(current_price);
                    fvgs.push(fvg);
                }
            }

            // Bearish FVG: gap between prev low and next high with current candle being bearish
            if curr.is_bearish() && next.high < prev.low {
                let gap_high = prev.low;
                let gap_low = next.high;
                let gap_size = gap_high - gap_low;

                let mid_price = (gap_high + gap_low) / 2.0;
                if mid_price > 0.0 && (gap_size / mid_price) >= self.min_size_pct / 100.0 {
                    let mut fvg = FairValueGap::new(FvgType::Bearish, gap_high, gap_low, curr.timestamp_ns);
                    fvg.update_fill_status(current_price);
                    fvgs.push(fvg);
                }
            }
        }

        // Limit number of FVGs (keep most recent/unfilled)
        fvgs.sort_by(|a, b| {
            // Prioritize unfilled gaps, then by recency
            let a_priority = if a.filled { 1 } else { 0 };
            let b_priority = if b.filled { 1 } else { 0 };
            a_priority.cmp(&b_priority).then(b.timestamp_ns.cmp(&a.timestamp_ns))
        });
        fvgs.truncate(self.max_fvgs);

        Ok(fvgs)
    }

    /// Detect liquidity voids (extreme FVGs with volume imbalance)
    pub fn detect_liquidity_voids(
        &self,
        candles: &[FvgCandle],
        volumes: &[f64],
    ) -> Result<Vec<LiquidityVoid>, FvgError> {
        if candles.len() < 3 || volumes.len() != candles.len() {
            return Err(FvgError::InsufficientData);
        }

        let mut voids = Vec::new();

        for i in 1..candles.len() - 1 {
            let prev = &candles[i - 1];
            let curr = &candles[i];
            let next = &candles[i + 1];

            // Calculate volume imbalance
            let avg_vol = (volumes[i - 1] + volumes[i + 1]) / 2.0;
            let vol_imbalance = if avg_vol > 0.0 {
                (volumes[i] - avg_vol) / avg_vol
            } else {
                0.0
            };

            // Liquidity void requires significant gap AND volume imbalance
            if curr.is_bullish() && next.low > prev.high && vol_imbalance > 0.5 {
                let void = LiquidityVoid::new(next.low, prev.high, curr.timestamp_ns, vol_imbalance);
                voids.push(void);
            } else if curr.is_bearish() && next.high < prev.low && vol_imbalance < -0.5 {
                let void = LiquidityVoid::new(prev.low, next.high, curr.timestamp_ns, vol_imbalance.abs());
                voids.push(void);
            }
        }

        Ok(voids)
    }

    /// Find the nearest unfilled FVG
    pub fn find_nearest_unfilled(&self, fvgs: &[FairValueGap]) -> Option<&FairValueGap> {
        let current_price = self.get_current_price();

        fvgs.iter()
            .filter(|fvg| !fvg.filled)
            .min_by(|a, b| {
                let dist_a = a.distance_from(current_price);
                let dist_b = b.distance_from(current_price);
                dist_a.partial_cmp(&dist_b).unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Calculate probability of FVG being filled based on historical patterns
    pub fn calculate_fill_probability(&self, fvg: &FairValueGap, time_elapsed_ms: u64) -> f64 {
        // Base probability increases with time
        let time_factor = (time_elapsed_ms as f64 / 3600000.0).min(1.0); // Max out at 1 hour

        // Larger gaps have higher fill probability
        let size_factor = fvg.size.min(0.01); // Normalize to 1% max

        // Partially filled gaps more likely to complete
        let fill_momentum = fvg.fill_pct * 0.3;

        (time_factor * 0.4 + size_factor * 100.0 * 0.3 + fill_momentum).min(1.0)
    }

    /// Activate/deactivate detector
    pub fn set_active(&self, active: bool) {
        self.active.store(active, Ordering::Relaxed);
    }

    pub fn is_active(&self) -> bool {
        self.active.load(Ordering::Relaxed)
    }
}

impl Default for FvgDetector {
    fn default() -> Self {
        Self::new(0.05, 20) // 0.05% minimum size, track 20 FVGs
    }
}

/// FVG signal for trading
#[derive(Debug, Clone, Copy)]
pub struct FvgSignal {
    pub fvg: FairValueGap,
    pub action: FvgAction,
    pub entry_price: f64,
    pub target_price: f64,
    pub stop_loss: f64,
    pub confidence: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FvgAction {
    EnterLong,
    EnterShort,
    Exit,
    Wait,
}

/// Normalized tick data for FVG calculation
#[derive(Debug, Clone, Copy)]
pub struct NormalizedTick {
    pub timestamp_ns: u64,
    pub price: f64,
    pub volume: f64,
    pub normalized_volume: f64, // Volume normalized to recent average
}

impl NormalizedTick {
    pub fn new(timestamp_ns: u64, price: f64, volume: f64, avg_volume: f64) -> Self {
        let normalized_volume = if avg_volume > 0.0 {
            volume / avg_volume
        } else {
            1.0
        };

        Self {
            timestamp_ns,
            price,
            volume,
            normalized_volume,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fvg_detection_bullish() {
        let detector = FvgDetector::new(0.01, 10);
        
        // Create bullish FVG pattern
        let candles = vec![
            FvgCandle::new(1000, 100.0, 101.0, 99.0, 100.5).unwrap(),
            FvgCandle::new(2000, 100.5, 102.0, 100.0, 101.5).unwrap(),
            FvgCandle::new(3000, 101.5, 103.0, 101.5, 102.5).unwrap(),
        ];

        detector.update_price(102.0);
        let fvgs = detector.detect_fvgs(&candles).unwrap();

        // Should detect bullish FVG between prev high (101.0) and next low (101.5)
        assert!(!fvgs.is_empty());
    }

    #[test]
    fn test_fvg_fill_status() {
        let mut fvg = FairValueGap::new(FvgType::Bullish, 102.0, 100.0, 1000);
        
        // Price above FVG - not filled
        fvg.update_fill_status(103.0);
        assert!(!fvg.filled);
        assert!(fvg.fill_pct < 0.5);

        // Price in FVG - partially filled
        fvg.update_fill_status(101.0);
        assert!(!fvg.filled);
        assert!((fvg.fill_pct - 0.5).abs() < 0.01);

        // Price below FVG - fully filled
        fvg.update_fill_status(99.0);
        assert!(fvg.filled);
        assert!((fvg.fill_pct - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_fvg_distance_calculation() {
        let fvg = FairValueGap::new(FvgType::Bullish, 102.0, 100.0, 1000);
        
        assert_eq!(fvg.distance_from(103.0), 1.0);
        assert_eq!(fvg.distance_from(99.0), 1.0);
        assert_eq!(fvg.distance_from(101.0), 0.0);
    }
}
