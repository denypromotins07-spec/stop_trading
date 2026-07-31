//! Inducement Detection
//! Tracks minor pullbacks that induce early retail entries before the real institutional move.
//! Filters out fake breakouts using state machine logic.

use super::liquidity_pools::{LiquidityPool, PoolType};
use super::breaker_blocks::{OrderBlock, OrderBlockType};

/// Maximum inducement levels to track
const MAX_INDUCEMENT_LEVELS: usize = 32;

/// Type of inducement pattern
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u8)]
pub enum InducementType {
    /// Bullish inducement: price dips to trigger stops before rallying
    LongTrap = 0,
    /// Bearish inducement: price spikes to trigger FOMO before dropping
    ShortTrap = 1,
    /// Double trap: complex pattern with both long and short traps
    DoubleTrap = 2,
}

/// Inducement level detected in the market
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct InducementLevel {
    pub price: i64,              // Fixed-point price level
    pub inducement_type: InducementType,
    pub trigger_timestamp: u64,  // When the inducement was triggered
    pub sweep_high: i64,         // High of the sweep (for long traps)
    pub sweep_low: i64,          // Low of the sweep (for short traps)
    pub volume_spike: bool,      // Whether volume spiked during inducement
    pub confidence: u8,          // Confidence score 0-100
    _padding: [u8; 3],           // Cache-line alignment
}

/// State for tracking inducement patterns
#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct InducementState {
    last_swing_high: i64,
    last_swing_low: i64,
    higher_high_count: u8,
    lower_low_count: u8,
    fakeout_detected: bool,
    _padding: [u8; 7],
}

impl Default for InducementState {
    fn default() -> Self {
        Self {
            last_swing_high: 0,
            last_swing_low: 0,
            higher_high_count: 0,
            lower_low_count: 0,
            fakeout_detected: false,
            _padding: [0; 7],
        }
    }
}

/// Inducement detector state machine
pub struct InducementDetector {
    state: InducementState,
    inducement_levels: [InducementLevel; MAX_INDUCEMENT_LEVELS],
    level_count: usize,
    lookback_bars: usize,
    recent_highs: [i64; 20],
    recent_lows: [i64; 20],
    recent_index: usize,
    recent_count: usize,
}

impl Default for InducementDetector {
    fn default() -> Self {
        Self::new(14)
    }
}

impl InducementDetector {
    /// Create a new inducement detector
    pub const fn new(lookback_bars: usize) -> Self {
        Self {
            state: InducementState {
                last_swing_high: 0,
                last_swing_low: 0,
                higher_high_count: 0,
                lower_low_count: 0,
                fakeout_detected: false,
                _padding: [0; 7],
            },
            inducement_levels: unsafe { core::mem::zeroed() },
            level_count: 0,
            lookback_bars,
            recent_highs: [0; 20],
            recent_lows: [i64::MAX; 20],
            recent_index: 0,
            recent_count: 0,
        }
    }

    /// Process a new candle and detect inducement patterns
    pub fn process_candle(
        &mut self,
        open: i64,
        high: i64,
        low: i64,
        close: i64,
        volume: u64,
        timestamp: u64,
    ) -> Option<InducementLevel> {
        // Update recent highs/lows ring buffer
        self.update_recent(high, low);

        let mut detected_level: Option<InducementLevel> = None;

        // Check for long trap (bullish inducement)
        if let Some(level) = self.check_long_trap(high, low, close, volume, timestamp) {
            detected_level = Some(level);
        }

        // Check for short trap (bearish inducement)
        if detected_level.is_none() {
            if let Some(level) = self.check_short_trap(high, low, close, volume, timestamp) {
                detected_level = Some(level);
            }
        }

        // Update state based on price action
        self.update_state(high, low);

        detected_level
    }

    /// Update the ring buffer of recent highs/lows
    #[inline]
    fn update_recent(&mut self, high: i64, low: i64) {
        let idx = self.recent_index;
        self.recent_highs[idx] = high;
        self.recent_lows[idx] = low;
        
        self.recent_index = (idx + 1) % 20;
        if self.recent_count < 20 {
            self.recent_count += 1;
        }
    }

    /// Check for long trap pattern
    /// Pattern: Price makes lower low, sweeps liquidity, then closes above previous low
    fn check_long_trap(
        &self,
        high: i64,
        low: i64,
        close: i64,
        volume: u64,
        timestamp: u64,
    ) -> Option<InducementLevel> {
        if self.recent_count < 5 {
            return None;
        }

        // Find recent swing low
        let mut min_low = i64::MAX;
        let mut min_idx = 0;
        
        for i in 0..self.recent_count.min(self.lookback_bars) {
            let idx = (self.recent_index.wrapping_sub(i + 1)) % 20;
            if self.recent_lows[idx] < min_low {
                min_low = self.recent_lows[idx];
                min_idx = i;
            }
        }

        // Check if current candle swept below the swing low
        if low < min_low {
            // Check if we closed back above (trap confirmed)
            if close > min_low {
                // Calculate confidence based on how far we swept and recovered
                let sweep_depth = min_low - low;
                let recovery = close - min_low;
                let total_range = high - low;
                
                if total_range == 0 {
                    return None;
                }

                let confidence = ((sweep_depth + recovery) * 100 / total_range).min(100) as u8;
                
                // Volume spike adds confidence
                let volume_spike = volume > self.get_average_volume() * 2;
                let final_confidence = if volume_spike {
                    confidence.min(100).saturating_add(10).min(100)
                } else {
                    confidence
                };

                return Some(InducementLevel {
                    price: min_low,
                    inducement_type: InducementType::LongTrap,
                    trigger_timestamp: timestamp,
                    sweep_high: high,
                    sweep_low: low,
                    volume_spike,
                    confidence: final_confidence,
                    _padding: [0; 3],
                });
            }
        }

        None
    }

    /// Check for short trap pattern
    /// Pattern: Price makes higher high, sweeps liquidity, then closes below previous high
    fn check_short_trap(
        &self,
        high: i64,
        low: i64,
        close: i64,
        volume: u64,
        timestamp: u64,
    ) -> Option<InducementLevel> {
        if self.recent_count < 5 {
            return None;
        }

        // Find recent swing high
        let mut max_high = i64::MIN;
        
        for i in 0..self.recent_count.min(self.lookback_bars) {
            let idx = (self.recent_index.wrapping_sub(i + 1)) % 20;
            if self.recent_highs[idx] > max_high {
                max_high = self.recent_highs[idx];
            }
        }

        // Check if current candle swept above the swing high
        if high > max_high {
            // Check if we closed back below (trap confirmed)
            if close < max_high {
                // Calculate confidence
                let sweep_depth = high - max_high;
                let rejection = max_high - close;
                let total_range = high - low;
                
                if total_range == 0 {
                    return None;
                }

                let confidence = ((sweep_depth + rejection) * 100 / total_range).min(100) as u8;
                
                // Volume spike adds confidence
                let volume_spike = volume > self.get_average_volume() * 2;
                let final_confidence = if volume_spike {
                    confidence.min(100).saturating_add(10).min(100)
                } else {
                    confidence
                };

                return Some(InducementLevel {
                    price: max_high,
                    inducement_type: InducementType::ShortTrap,
                    trigger_timestamp: timestamp,
                    sweep_high: high,
                    sweep_low: low,
                    volume_spike,
                    confidence: final_confidence,
                    _padding: [0; 3],
                });
            }
        }

        None
    }

    /// Get simple average volume from recent candles (placeholder - would integrate with volume data)
    #[inline]
    fn get_average_volume(&self) -> u64 {
        1_000_000 // Placeholder - in production would track actual volume history
    }

    /// Update internal state based on price action
    fn update_state(&mut self, high: i64, low: i64) {
        if self.state.last_swing_high != 0 {
            if high > self.state.last_swing_high {
                self.state.higher_high_count = self.state.higher_high_count.saturating_add(1);
            } else {
                self.state.higher_high_count = 0;
            }
        }
        
        if self.state.last_swing_low != 0 {
            if low < self.state.last_swing_low {
                self.state.lower_low_count = self.state.lower_low_count.saturating_add(1);
            } else {
                self.state.lower_low_count = 0;
            }
        }

        // Detect potential fakeout when we have multiple higher highs followed by reversal
        if self.state.higher_high_count >= 3 {
            self.state.fakeout_detected = true;
        }

        self.state.last_swing_high = high;
        self.state.last_swing_low = low;
    }

    /// Check if current price action is likely a fake breakout
    #[inline]
    pub fn is_fakeout_likely(&self) -> bool {
        self.state.fakeout_detected
    }

    /// Get the count of consecutive higher highs
    #[inline]
    pub fn get_higher_high_count(&self) -> u8 {
        self.state.higher_high_count
    }

    /// Get the count of consecutive lower lows
    #[inline]
    pub fn get_lower_low_count(&self) -> u8 {
        self.state.lower_low_count
    }

    /// Record an inducement level for external tracking
    pub fn record_inducement(&mut self, level: InducementLevel) {
        if self.level_count < MAX_INDUCEMENT_LEVELS {
            self.inducement_levels[self.level_count] = level;
            self.level_count += 1;
        } else {
            // Shift left
            unsafe {
                core::ptr::copy(
                    self.inducement_levels.as_ptr().add(1),
                    self.inducement_levels.as_mut_ptr(),
                    MAX_INDUCEMENT_LEVELS - 1,
                );
            }
            self.inducement_levels[MAX_INDUCEMENT_LEVELS - 1] = level;
        }
    }

    /// Get all recorded inducement levels
    pub fn get_inducement_levels(&self) -> &[InducementLevel] {
        unsafe {
            core::slice::from_raw_parts(
                self.inducement_levels.as_ptr(),
                self.level_count,
            )
        }
    }

    /// Filter fake breakouts: returns true if breakout is genuine
    pub fn validate_breakout(
        &self,
        breakout_price: i64,
        resistance_level: i64,
        is_upside: bool,
    ) -> bool {
        // If fakeout is likely, require stronger confirmation
        if self.state.fakeout_detected {
            let threshold = (resistance_level.abs() * 30) / 10000; // 0.3% beyond level
            if is_upside {
                breakout_price > resistance_level + threshold
            } else {
                breakout_price < resistance_level - threshold
            }
        } else {
            // Normal validation
            if is_upside {
                breakout_price > resistance_level
            } else {
                breakout_price < resistance_level
            }
        }
    }

    /// Check if price is near an inducement level (potential reversal zone)
    pub fn is_near_inducement(&self, price: i64, tolerance_bps: i64) -> Option<&InducementLevel> {
        for i in 0..self.level_count {
            let level = unsafe { self.inducement_levels.get_unchecked(i) };
            let diff = (price - level.price).abs();
            let threshold = (level.price.abs() * tolerance_bps) / 10000;
            
            if diff <= threshold.max(1) {
                return Some(level);
            }
        }
        None
    }

    /// Reset the detector state
    pub fn reset(&mut self) {
        self.state = InducementState::default();
        self.level_count = 0;
        self.recent_count = 0;
        self.recent_index = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_long_trap_detection() {
        let mut detector = InducementDetector::new(5);
        
        // Set up a swing low
        for i in 0..10 {
            let high = 100_0000_0000i64 + (i as i64) * 1000_0000i64;
            let low = 99_0000_0000i64 + (i as i64) * 1000_0000i64;
            detector.process_candle(high, low, high, 1_000_000, i * 60_000_000);
        }
        
        // Now create a long trap: sweep below then recover
        let swing_low = 108_0000_0000i64;
        let sweep_low = 107_5000_0000i64;
        let close = 108_5000_0000i64;
        
        let result = detector.process_candle(
            swing_low,
            109_0000_0000i64,
            sweep_low,
            close,
            5_000_000, // Volume spike
            600_000_000,
        );
        
        assert!(result.is_some());
        assert_eq!(result.unwrap().inducement_type, InducementType::LongTrap);
    }

    #[test]
    fn test_fakeout_detection() {
        let mut detector = InducementDetector::new(3);
        
        // Create multiple higher highs
        for i in 0..5 {
            let high = 100_0000_0000i64 + (i as i64) * 1000_0000i64;
            let low = 99_0000_0000i64 + (i as i64) * 1000_0000i64;
            detector.process_candle(high, low, high, 1_000_000, i * 60_000_000);
        }
        
        assert!(detector.is_fakeout_likely());
        assert!(detector.get_higher_high_count() >= 3);
    }
}
