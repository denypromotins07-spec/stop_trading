//! Fibonacci Retracement and Extension Calculator
//! Auto-detects significant swing highs/lows to draw dynamic Fibonacci levels.
//! Stores only minimal required high/low primitives - no full candle history.

use core::sync::atomic::{AtomicUsize, Ordering};

/// Maximum number of swing points to track
const MAX_SWINGS: usize = 128;

/// Standard Fibonacci ratios (scaled by 1e8)
const FIB_0: i64 = 0;
const FIB_236: i64 = 23_600_000;   // 0.236
const FIB_382: i64 = 38_200_000;   // 0.382
const FIB_500: i64 = 50_000_000;   // 0.500
const FIB_618: i64 = 61_800_000;   // 0.618
const FIB_786: i64 = 78_600_000;   // 0.786
const FIB_1000: i64 = 100_000_000; // 1.000
const FIB_1272: i64 = 127_200_000; // 1.272
const FIB_1414: i64 = 141_400_000; // 1.414
const FIB_1618: i64 = 161_800_000; // 1.618
const FIB_2000: i64 = 200_000_000; // 2.000
const FIB_2618: i64 = 261_800_000; // 2.618

/// Swing point type
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u8)]
pub enum SwingType {
    High = 0,
    Low = 1,
}

/// Minimal swing point storage
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SwingPoint {
    pub price: i64,
    pub timestamp: u64,
    pub swing_type: SwingType,
    pub significance: u8, // Number of times tested
    _padding: [u8; 3],
}

/// Fibonacci level with price and ratio
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FibLevel {
    pub price: i64,
    pub ratio: i64,      // Scaled by 1e8
    pub level_type: FibType,
    pub strength: u8,    // Based on confluence
    _padding: [u8; 3],
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u8)]
pub enum FibType {
    Retracement = 0,
    Extension = 1,
}

/// Fibonacci calculator state machine
pub struct FibonacciCalculator {
    /// Swing highs
    highs: [SwingPoint; MAX_SWINGS],
    high_count: AtomicUsize,
    /// Swing lows
    lows: [SwingPoint; MAX_SWINGS],
    low_count: AtomicUsize,
    /// Lookback for swing detection
    lookback: usize,
    /// Minimum swing size in basis points
    min_swing_bps: i64,
}

impl Default for FibonacciCalculator {
    fn default() -> Self {
        Self::new(20, 100)
    }
}

impl FibonacciCalculator {
    /// Create new calculator with lookback and minimum swing size (in bps)
    pub const fn new(lookback: usize, min_swing_bps: i64) -> Self {
        Self {
            highs: unsafe { core::mem::zeroed() },
            high_count: AtomicUsize::new(0),
            lows: unsafe { core::mem::zeroed() },
            low_count: AtomicUsize::new(0),
            lookback,
            min_swing_bps,
        }
    }

    /// Process a new candle and detect swings
    pub fn process_candle(
        &mut self,
        high: i64,
        low: i64,
        timestamp: u64,
    ) -> Option<(SwingPoint, Option<[FibLevel; 11]>)> {
        let mut detected_swing: Option<SwingPoint> = None;
        let mut fib_levels: Option<[FibLevel; 11]> = None;

        // Check for swing high
        if self.is_swing_high(high) {
            let swing = SwingPoint {
                price: high,
                timestamp,
                swing_type: SwingType::High,
                significance: 1,
                _padding: [0; 3],
            };
            self.add_high(swing);
            detected_swing = Some(swing);
        }

        // Check for swing low
        if self.is_swing_low(low) {
            let swing = SwingPoint {
                price: low,
                timestamp,
                swing_type: SwingType::Low,
                significance: 1,
                _padding: [0; 3],
            };
            self.add_low(swing);
            detected_swing = Some(swing);
        }

        // If we have both high and low, calculate Fibonacci levels
        if let Some(ref swing) = detected_swing {
            if let Some(levels) = self.calculate_fib_levels(swing) {
                fib_levels = Some(levels);
            }
        }

        detected_swing.map(|s| (s, fib_levels))
    }

    #[inline]
    fn is_swing_high(&self, current_high: i64) -> bool {
        let count = self.high_count.load(Ordering::Relaxed);
        if count < self.lookback {
            return false;
        }

        // Check previous candles
        for i in 1..=self.lookback {
            let idx = count.wrapping_sub(i);
            if idx >= MAX_SWINGS {
                continue;
            }
            unsafe {
                if *self.highs.get_unchecked(idx).price >= current_high {
                    return false;
                }
            }
        }
        true
    }

    #[inline]
    fn is_swing_low(&self, current_low: i64) -> bool {
        let count = self.low_count.load(Ordering::Relaxed);
        if count < self.lookback {
            return false;
        }

        for i in 1..=self.lookback {
            let idx = count.wrapping_sub(i);
            if idx >= MAX_SWINGS {
                continue;
            }
            unsafe {
                if *self.lows.get_unchecked(idx).price <= current_low {
                    return false;
                }
            }
        }
        true
    }

    fn add_high(&mut self, swing: SwingPoint) {
        let count = self.high_count.load(Ordering::Relaxed);
        if count < MAX_SWINGS {
            self.highs[count] = swing;
            self.high_count.store(count + 1, Ordering::Relaxed);
        } else {
            // Circular buffer
            unsafe {
                core::ptr::copy(
                    self.highs.as_ptr().add(1),
                    self.highs.as_mut_ptr(),
                    MAX_SWINGS - 1,
                );
            }
            self.highs[MAX_SWINGS - 1] = swing;
        }
    }

    fn add_low(&mut self, swing: SwingPoint) {
        let count = self.low_count.load(Ordering::Relaxed);
        if count < MAX_SWINGS {
            self.lows[count] = swing;
            self.low_count.store(count + 1, Ordering::Relaxed);
        } else {
            unsafe {
                core::ptr::copy(
                    self.lows.as_ptr().add(1),
                    self.lows.as_mut_ptr(),
                    MAX_SWINGS - 1,
                );
            }
            self.lows[MAX_SWINGS - 1] = swing;
        }
    }

    /// Calculate Fibonacci retracement and extension levels
    fn calculate_fib_levels(&self, latest_swing: &SwingPoint) -> Option<[FibLevel; 11]> {
        // Find the opposite swing to calculate from
        let (start_price, end_price) = match latest_swing.swing_type {
            SwingType::High => {
                // Find most recent low before this high
                self.find_opposite_low(latest_swing.timestamp)
                    .map(|low| (low.price, latest_swing.price))
            }
            SwingType::Low => {
                // Find most recent high before this low
                self.find_opposite_high(latest_swing.timestamp)
                    .map(|high| (high.price, latest_swing.price))
            }
        }?;

        let range = (end_price - start_price).abs();
        if range == 0 {
            return None;
        }

        // Validate minimum swing size
        let swing_bps = (range * 10000) / start_price.abs().max(1);
        if swing_bps < self.min_swing_bps {
            return None;
        }

        let mut levels: [FibLevel; 11] = [FibLevel {
            price: 0,
            ratio: 0,
            level_type: FibType::Retracement,
            strength: 0,
            _padding: [0; 3],
        }; 11];

        // Retracement levels (for uptrend: pullback levels; for downtrend: bounce levels)
        let is_uptrend = end_price > start_price;
        
        // Key Fibonacci levels
        let fib_ratios = [
            (FIB_0, FibType::Retracement),
            (FIB_236, FibType::Retracement),
            (FIB_382, FibType::Retracement),
            (FIB_500, FibType::Retracement),
            (FIB_618, FibType::Retracement),
            (FIB_786, FibType::Retracement),
            (FIB_1000, FibType::Retracement),
            (FIB_1272, FibType::Extension),
            (FIB_1414, FibType::Extension),
            (FIB_1618, FibType::Extension),
            (FIB_2618, FibType::Extension),
        ];

        for (i, &(ratio, fib_type)) in fib_ratios.iter().enumerate() {
            let retracement_amount = range * ratio / FIB_1000;
            
            let price = if is_uptrend {
                // For uptrend, retracements are below the high
                end_price - retracement_amount
            } else {
                // For downtrend, retracements are above the low
                end_price + retracement_amount
            };

            // Calculate strength based on confluence with round numbers
            let strength = self.calculate_level_strength(price);

            levels[i] = FibLevel {
                price,
                ratio,
                level_type: fib_type,
                strength,
                _padding: [0; 3],
            };
        }

        Some(levels)
    }

    fn find_opposite_low(&self, before_timestamp: u64) -> Option<SwingPoint> {
        let count = self.low_count.load(Ordering::Relaxed);
        for i in (0..count).rev() {
            unsafe {
                let low = self.lows.get_unchecked(i);
                if low.timestamp < before_timestamp {
                    return Some(*low);
                }
            }
        }
        None
    }

    fn find_opposite_high(&self, before_timestamp: u64) -> Option<SwingPoint> {
        let count = self.high_count.load(Ordering::Relaxed);
        for i in (0..count).rev() {
            unsafe {
                let high = self.highs.get_unchecked(i);
                if high.timestamp < before_timestamp {
                    return Some(*high);
                }
            }
        }
        None
    }

    /// Calculate strength based on confluence with psychological levels
    fn calculate_level_strength(&self, price: i64) -> u8 {
        let mut strength = 1u8;

        // Check for round number confluence
        if price % 100_0000_0000i64 == 0 {
            strength += 2; // Major round number
        } else if price % 10_0000_0000i64 == 0 {
            strength += 1; // Minor round number
        }

        // Check for 0.5 midpoint (often significant)
        if price % 50_0000_0000i64 == 0 {
            strength += 1;
        }

        strength.min(10)
    }

    /// Get the most significant swing high
    pub fn get_major_high(&self) -> Option<SwingPoint> {
        let count = self.high_count.load(Ordering::Relaxed);
        if count == 0 {
            return None;
        }

        let mut major = unsafe { *self.highs.get_unchecked(0) };
        for i in 1..count {
            unsafe {
                let swing = *self.highs.get_unchecked(i);
                if swing.price > major.price {
                    major = swing;
                }
            }
        }
        Some(major)
    }

    /// Get the most significant swing low
    pub fn get_major_low(&self) -> Option<SwingPoint> {
        let count = self.low_count.load(Ordering::Relaxed);
        if count == 0 {
            return None;
        }

        let mut major = unsafe { *self.lows.get_unchecked(0) };
        for i in 1..count {
            unsafe {
                let swing = *self.lows.get_unchecked(i);
                if swing.price < major.price {
                    major = swing;
                }
            }
        }
        Some(major)
    }

    /// Get all Fibonacci levels for the current trend
    pub fn get_current_fib_range(&self) -> Option<(i64, i64)> {
        let major_high = self.get_major_high()?;
        let major_low = self.get_major_low()?;
        
        if major_high.timestamp > major_low.timestamp {
            Some((major_low.price, major_high.price))
        } else {
            Some((major_high.price, major_low.price))
        }
    }

    /// Detect price approaching a Fibonacci level
    pub fn is_near_fib_level(&self, price: i64, tolerance_bps: i64) -> Option<FibLevel> {
        if let Some((low, high)) = self.get_current_fib_range() {
            let range = (high - low).abs();
            if range == 0 {
                return None;
            }

            let fib_ratios = [FIB_0, FIB_236, FIB_382, FIB_500, FIB_618, FIB_786, FIB_1000];
            
            for &ratio in &fib_ratios {
                let fib_price = low + range * ratio / FIB_1000;
                let diff = (price - fib_price).abs();
                let threshold = (fib_price * tolerance_bps) / 10000;
                
                if diff <= threshold.max(1) {
                    return Some(FibLevel {
                        price: fib_price,
                        ratio,
                        level_type: FibType::Retracement,
                        strength: 5,
                        _padding: [0; 3],
                    });
                }
            }
        }
        None
    }

    /// Reset the calculator
    pub fn reset(&mut self) {
        self.highs = unsafe { core::mem::zeroed() };
        self.high_count.store(0, Ordering::Relaxed);
        self.lows = unsafe { core::mem::zeroed() };
        self.low_count.store(0, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fib_calculation() {
        let mut calc = FibonacciCalculator::new(5, 50);

        // Create an uptrend
        for i in 0..20 {
            let base = 100_0000_0000i64;
            let high = base + (i as i64) * 1000_0000i64 + 500_0000i64;
            let low = base + (i as i64) * 1000_0000i64 - 500_0000i64;
            calc.process_candle(high, low, i as u64 * 60_000_000);
        }

        // Should have detected swings
        assert!(calc.get_major_high().is_some());
        assert!(calc.get_major_low().is_some());
    }

    #[test]
    fn test_fib_ratios() {
        let start = 100_0000_0000i64;
        let end = 110_0000_0000i64;
        let range = end - start;

        // Test 0.618 retracement
        let fib_618 = start + range * FIB_618 / FIB_1000;
        assert!(fib_618 > start && fib_618 < end);

        // Test 1.618 extension
        let fib_ext = start + range * FIB_1618 / FIB_1000;
        assert!(fib_ext > end);
    }
}
