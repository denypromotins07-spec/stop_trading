//! Stop-Hunt Detection Module
//! Identifies equal highs/lows and liquidity sweeps.
//! Calculates probability of retail stop clusters being targeted by institutional smart money.

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::time::Duration;

const CACHE_LINE_SIZE: usize = 64;

#[repr(C, align(64))]
struct CachePadded<T> {
    data: T,
    _pad: [u8; CACHE_LINE_SIZE],
}

impl<T: Default> Default for CachePadded<T> {
    fn default() -> Self {
        Self {
            data: T::default(),
            _pad: [0u8; CACHE_LINE_SIZE],
        }
    }
}

/// Detected stop-hunt pattern
#[derive(Debug, Clone, Copy)]
pub struct StopHuntSignal {
    /// Pattern type
    pub pattern_type: StopHuntPattern,
    /// Price level where hunt occurred
    pub hunt_price: i64,
    /// Timestamp (nanoseconds)
    pub timestamp_ns: u64,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
    /// Estimated stop cluster size
    pub estimated_stops: u64,
    /// Sweep depth (how far price went beyond level)
    pub sweep_depth_ticks: i64,
    /// Side targeted (true = longs stopped, false = shorts stopped)
    pub longs_targeted: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StopHuntPattern {
    /// Equal highs swept
    EqualHighsSweep,
    /// Equal lows swept
    EqualLowsSweep,
    /// Liquidity grab above resistance
    ResistanceGrab,
    /// Liquidity grab below support
    SupportGrab,
    /// Wick reversal after sweep
    WickReversal,
}

/// Price level with touch count
#[derive(Debug, Clone, Copy)]
struct PriceLevel {
    price: i64,
    touch_count: u32,
    last_touch_ns: u64,
    high: i64,
    low: i64,
}

/// Lock-free stop-hunt detector
pub struct StopHuntDetector {
    /// Detected signals count
    signals_detected: CachePadded<AtomicU64>,
    /// False positives (reversed before fill)
    false_positives: CachePadded<AtomicU64>,
    /// Active flag
    is_active: CachePadded<AtomicBool>,
    /// Minimum touches to establish level
    min_touches: u32,
    /// Sweep threshold in ticks
    sweep_threshold_ticks: i64,
    /// Recent highs (simplified as atomic storage)
    recent_highs: CachePadded<AtomicI64>,
    /// Recent lows
    recent_lows: CachePadded<AtomicI64>,
}

impl StopHuntDetector {
    pub fn new(min_touches: u32, sweep_threshold_ticks: i64) -> Self {
        Self {
            signals_detected: CachePadded::default(),
            false_positives: CachePadded::default(),
            is_active: CachePadded::new(AtomicBool::new(true)),
            min_touches,
            sweep_threshold_ticks,
            recent_highs: CachePadded::default(),
            recent_lows: CachePadded::default(),
        }
    }

    /// Analyze price action for stop-hunt patterns
    pub fn analyze(&self, price_history: &[PriceBar]) -> Option<StopHuntSignal> {
        if !self.is_active.data.load(Ordering::Acquire) || price_history.len() < 5 {
            return None;
        }

        // Find equal highs/lows
        let equal_levels = self.find_equal_levels(price_history);
        
        // Check for sweeps
        for level in equal_levels {
            if let Some(signal) = self.check_for_sweep(&level, price_history) {
                self.signals_detected.data.fetch_add(1, Ordering::AcqRel);
                return Some(signal);
            }
        }

        None
    }

    /// Find price levels with multiple touches
    fn find_equal_levels(&self, bars: &[PriceBar]) -> Vec<PriceLevel> {
        let mut levels = Vec::new();
        let tolerance = self.sweep_threshold_ticks / 2;

        // Scan for highs with multiple touches
        for i in 0..bars.len() {
            let high = bars[i].high;
            
            // Count touches within tolerance
            let mut touch_count = 0u32;
            let mut last_touch = 0u64;
            
            for j in 0..bars.len() {
                if (bars[j].high - high).abs() <= tolerance {
                    touch_count += 1;
                    last_touch = bars[j].timestamp_ns;
                }
            }

            if touch_count >= self.min_touches {
                // Check if this level is already recorded
                if !levels.iter().any(|l| (l.price - high).abs() <= tolerance) {
                    levels.push(PriceLevel {
                        price: high,
                        touch_count,
                        last_touch_ns: last_touch,
                        high,
                        low: bars[i].low,
                    });
                }
            }
        }

        // Scan for lows with multiple touches
        for i in 0..bars.len() {
            let low = bars[i].low;
            
            let mut touch_count = 0u32;
            let mut last_touch = 0u64;
            
            for j in 0..bars.len() {
                if (bars[j].low - low).abs() <= tolerance {
                    touch_count += 1;
                    last_touch = bars[j].timestamp_ns;
                }
            }

            if touch_count >= self.min_touches {
                if !levels.iter().any(|l| (l.price - low).abs() <= tolerance) {
                    levels.push(PriceLevel {
                        price: low,
                        touch_count,
                        last_touch_ns: last_touch,
                        high: bars[i].high,
                        low,
                    });
                }
            }
        }

        levels
    }

    /// Check if a level was swept
    fn check_for_sweep(&self, level: &PriceLevel, bars: &[PriceBar]) -> Option<StopHuntSignal> {
        if bars.is_empty() {
            return None;
        }

        let latest_bar = bars[bars.len() - 1];
        
        // Determine if this is a high or low level
        let is_high_level = level.touch_count > 0 && level.price >= level.high;
        
        if is_high_level {
            // Check for sweep above equal highs
            let sweep_above = latest_bar.high - level.price;
            
            if sweep_above > 0 && sweep_above <= self.sweep_threshold_ticks * 2 {
                // Check for reversal (wick)
                let wick = latest_bar.high - latest_bar.close;
                let body = (latest_bar.close - latest_bar.open).abs();
                
                // Strong reversal if wick is larger than body
                let is_reversal = wick > body * 2;
                
                let confidence = if is_reversal { 0.8 } else { 0.5 }
                    + (sweep_above as f64 / self.sweep_threshold_ticks as f64).min(0.2);

                return Some(StopHuntSignal {
                    pattern_type: if is_reversal { 
                        StopHuntPattern::WickReversal 
                    } else { 
                        StopHuntPattern::EqualHighsSweep 
                    },
                    hunt_price: level.price,
                    timestamp_ns: latest_bar.timestamp_ns,
                    confidence: confidence.min(1.0),
                    estimated_stops: self.estimate_stop_volume(sweep_above, level.touch_count),
                    sweep_depth_ticks: sweep_above,
                    longs_targeted: true, // Stops above highs are long stops
                });
            }
        } else {
            // Check for sweep below equal lows
            let sweep_below = level.price - latest_bar.low;
            
            if sweep_below > 0 && sweep_below <= self.sweep_threshold_ticks * 2 {
                let wick = latest_bar.close - latest_bar.low;
                let body = (latest_bar.close - latest_bar.open).abs();
                let is_reversal = wick > body * 2;
                
                let confidence = if is_reversal { 0.8 } else { 0.5 }
                    + (sweep_below as f64 / self.sweep_threshold_ticks as f64).min(0.2);

                return Some(StopHuntSignal {
                    pattern_type: if is_reversal { 
                        StopHuntPattern::WickReversal 
                    } else { 
                        StopHuntPattern::EqualLowsSweep 
                    },
                    hunt_price: level.price,
                    timestamp_ns: latest_bar.timestamp_ns,
                    confidence: confidence.min(1.0),
                    estimated_stops: self.estimate_stop_volume(sweep_below, level.touch_count),
                    sweep_depth_ticks: sweep_below,
                    longs_targeted: false, // Stops below lows are short stops
                });
            }
        }

        None
    }

    /// Estimate stop cluster volume based on sweep characteristics
    fn estimate_stop_volume(&self, sweep_depth: i64, touch_count: u32) -> u64 {
        // Simplified model: more touches = more stops accumulated
        // Deeper sweep = more stops triggered
        let base_volume = touch_count as u64 * 100;
        let depth_multiplier = (sweep_depth as f64 / 10.0).min(5.0);
        (base_volume as f64 * depth_multiplier) as u64
    }

    /// Get statistics
    pub fn get_stats(&self) -> StopHuntStats {
        StopHuntStats {
            signals_detected: self.signals_detected.data.load(Ordering::Acquire),
            false_positives: self.false_positives.data.load(Ordering::Acquire),
            accuracy: {
                let total = self.signals_detected.data.load(Ordering::Acquire);
                let fp = self.false_positives.data.load(Ordering::Acquire);
                if total > 0 {
                    1.0 - (fp as f64 / total as f64)
                } else {
                    1.0
                }
            },
            is_active: self.is_active.data.load(Ordering::Acquire),
        }
    }

    #[inline]
    pub fn set_active(&self, active: bool) {
        self.is_active.data.store(active, Ordering::Release);
    }

    #[inline]
    pub fn reset(&self) {
        self.signals_detected.data.store(0, Ordering::Release);
        self.false_positives.data.store(0, Ordering::Release);
    }
}

/// OHLCV price bar
#[derive(Debug, Clone, Copy)]
pub struct PriceBar {
    pub timestamp_ns: u64,
    pub open: i64,
    pub high: i64,
    pub low: i64,
    pub close: i64,
    pub volume: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct StopHuntStats {
    pub signals_detected: u64,
    pub false_positives: u64,
    pub accuracy: f64,
    pub is_active: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_equal_highs_sweep() {
        let detector = StopHuntDetector::new(2, 10);

        // Create price bars with equal highs then sweep
        let bars = vec![
            PriceBar { timestamp_ns: 1000, open: 10000, high: 10050, low: 9990, close: 10020, volume: 100 },
            PriceBar { timestamp_ns: 2000, open: 10020, high: 10050, low: 10000, close: 10030, volume: 100 },
            PriceBar { timestamp_ns: 3000, open: 10030, high: 10055, low: 10020, close: 10025, volume: 150 }, // Sweep!
        ];

        let signal = detector.analyze(&bars);
        assert!(signal.is_some());
        
        let signal = signal.unwrap();
        assert_eq!(signal.pattern_type, StopHuntPattern::EqualHighsSweep);
        assert!(signal.confidence > 0.0);
    }

    #[test]
    fn test_wick_reversal() {
        let detector = StopHuntDetector::new(2, 10);

        // Strong wick reversal after sweep
        let bars = vec![
            PriceBar { timestamp_ns: 1000, open: 10000, high: 10050, low: 9990, close: 10020, volume: 100 },
            PriceBar { timestamp_ns: 2000, open: 10020, high: 10050, low: 10000, close: 10010, volume: 100 },
            PriceBar { timestamp_ns: 3000, open: 10010, high: 10065, low: 10005, close: 10010, volume: 200 }, // Big wick!
        ];

        let signal = detector.analyze(&bars);
        assert!(signal.is_some());
        assert_eq!(signal.unwrap().pattern_type, StopHuntPattern::WickReversal);
    }
}
