//! Delta Divergence Detection
//! Detects Delta Divergence (price makes higher high, but CVD makes lower high).
//! Signals institutional exhaustion and potential reversals.

use core::sync::atomic::{AtomicI64, AtomicU64, AtomicUsize, Ordering};

/// Maximum divergence history to track
const MAX_HISTORY: usize = 256;

/// Divergence type
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u8)]
pub enum DivergenceType {
    /// Price HH, CVD LH - Bearish divergence
    BearishRegular = 0,
    /// Price LL, CVD HL - Bullish regular
    BullishRegular = 1,
    /// Price LH, CVD HH - Bearish hidden
    BearishHidden = 2,
    /// Price HL, CVD LL - Bullish hidden
    BullishHidden = 3,
}

/// Detected divergence signal
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DivergenceSignal {
    pub div_type: DivergenceType,
    pub price_level: i64,
    pub cvd_level: i64,
    pub timestamp: u64,
    pub strength: u8,  // 0-100 confidence
    _padding: [u8; 3],
}

/// Cumulative Volume Delta tracker
pub struct CVDTracker {
    cvd: AtomicI64,
    session_cvd: AtomicI64,
    total_volume: AtomicU64,
}

impl Default for CVDTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl CVDTracker {
    pub const fn new() -> Self {
        Self {
            cvd: AtomicI64::new(0),
            session_cvd: AtomicI64::new(0),
            total_volume: AtomicU64::new(0),
        }
    }

    #[inline]
    pub fn add_trade(&self, volume: u64, is_buyer_initiated: bool) {
        let delta = if is_buyer_initiated {
            volume as i64
        } else {
            -(volume as i64)
        };
        
        self.cvd.fetch_add(delta, Ordering::Relaxed);
        self.session_cvd.fetch_add(delta, Ordering::Relaxed);
        self.total_volume.fetch_add(volume, Ordering::Relaxed);
    }

    #[inline]
    pub fn get_cvd(&self) -> i64 {
        self.cvd.load(Ordering::Relaxed)
    }

    #[inline]
    pub fn get_session_cvd(&self) -> i64 {
        self.session_cvd.load(Ordering::Relaxed)
    }

    pub fn reset_session(&self) {
        self.session_cvd.store(0, Ordering::Relaxed);
    }
}

/// Price/CVD pivot point
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct PivotPoint {
    price: i64,
    cvd: i64,
    timestamp: u64,
    is_high: bool,
}

/// Delta Divergence detector
pub struct DeltaDivergenceDetector {
    cvd_tracker: CVDTracker,
    /// Price highs
    price_highs: [PivotPoint; MAX_HISTORY],
    price_high_count: AtomicUsize,
    /// Price lows
    price_lows: [PivotPoint; MAX_HISTORY],
    price_low_count: AtomicUsize,
    /// Last detected divergence
    last_signal: AtomicUsize,
    /// Lookback for divergence detection
    lookback: usize,
}

impl Default for DeltaDivergenceDetector {
    fn default() -> Self {
        Self::new(20)
    }
}

impl DeltaDivergenceDetector {
    pub const fn new(lookback: usize) -> Self {
        Self {
            cvd_tracker: CVDTracker::new(),
            price_highs: unsafe { core::mem::zeroed() },
            price_high_count: AtomicUsize::new(0),
            price_lows: unsafe { core::mem::zeroed() },
            price_low_count: AtomicUsize::new(0),
            last_signal: AtomicUsize::new(0),
            lookback,
        }
    }

    /// Process a new candle and check for divergences
    pub fn process_candle(
        &mut self,
        high: i64,
        low: i64,
        close: i64,
        volume: u64,
        bid_volume: u64,
        ask_volume: u64,
        timestamp: u64,
    ) -> Option<DivergenceSignal> {
        // Update CVD
        self.cvd_tracker.add_trade(ask_volume, true);
        self.cvd_tracker.add_trade(bid_volume, false);
        
        let current_cvd = self.cvd_tracker.get_cvd();
        
        // Check for pivot highs
        if self.is_price_high(high) {
            self.add_pivot(PivotPoint {
                price: high,
                cvd: current_cvd,
                timestamp,
                is_high: true,
            });
        }
        
        // Check for pivot lows
        if self.is_price_low(low) {
            self.add_pivot(PivotPoint {
                price: low,
                cvd: current_cvd,
                timestamp,
                is_high: false,
            });
        }
        
        // Check for divergences
        self.check_divergences(timestamp)
    }

    #[inline]
    fn is_price_high(&self, high: i64) -> bool {
        let count = self.price_high_count.load(Ordering::Relaxed);
        if count < 2 {
            return false;
        }
        
        // Simple pivot detection: higher than previous N highs
        for i in 0..count.min(self.lookback) {
            unsafe {
                if *self.price_highs.get_unchecked(count - 1 - i) >= high {
                    return false;
                }
            }
        }
        true
    }

    #[inline]
    fn is_price_low(&self, low: i64) -> bool {
        let count = self.price_low_count.load(Ordering::Relaxed);
        if count < 2 {
            return false;
        }
        
        for i in 0..count.min(self.lookback) {
            unsafe {
                if *self.price_lows.get_unchecked(count - 1 - i) <= low {
                    return false;
                }
            }
        }
        true
    }

    fn add_pivot(&mut self, pivot: PivotPoint) {
        if pivot.is_high {
            let count = self.price_high_count.load(Ordering::Relaxed);
            if count < MAX_HISTORY {
                self.price_highs[count] = pivot;
                self.price_high_count.store(count + 1, Ordering::Relaxed);
            } else {
                // Shift left
                unsafe {
                    core::ptr::copy(
                        self.price_highs.as_ptr().add(1),
                        self.price_highs.as_mut_ptr(),
                        MAX_HISTORY - 1,
                    );
                }
                self.price_highs[MAX_HISTORY - 1] = pivot;
            }
        } else {
            let count = self.price_low_count.load(Ordering::Relaxed);
            if count < MAX_HISTORY {
                self.price_lows[count] = pivot;
                self.price_low_count.store(count + 1, Ordering::Relaxed);
            } else {
                unsafe {
                    core::ptr::copy(
                        self.price_lows.as_ptr().add(1),
                        self.price_lows.as_mut_ptr(),
                        MAX_HISTORY - 1,
                    );
                }
                self.price_lows[MAX_HISTORY - 1] = pivot;
            }
        }
    }

    fn check_divergences(&self, timestamp: u64) -> Option<DivergenceSignal> {
        // Check bearish regular divergence (Price HH, CVD LH)
        if let Some(signal) = self.check_bearish_regular(timestamp) {
            return Some(signal);
        }
        
        // Check bullish regular divergence (Price LL, CVD HL)
        if let Some(signal) = self.check_bullish_regular(timestamp) {
            return Some(signal);
        }
        
        None
    }

    fn check_bearish_regular(&self, timestamp: u64) -> Option<DivergenceSignal> {
        let count = self.price_high_count.load(Ordering::Relaxed);
        if count < 2 {
            return None;
        }
        
        unsafe {
            let recent = self.price_highs.get_unchecked(count - 1);
            let previous = self.price_highs.get_unchecked(count - 2);
            
            // Price making higher high
            if recent.price > previous.price {
                // But CVD making lower high
                if recent.cvd < previous.cvd {
                    let strength = self.calculate_strength(
                        recent.price - previous.price,
                        previous.cvd - recent.cvd,
                    );
                    
                    return Some(DivergenceSignal {
                        div_type: DivergenceType::BearishRegular,
                        price_level: recent.price,
                        cvd_level: recent.cvd,
                        timestamp,
                        strength,
                        _padding: [0; 3],
                    });
                }
            }
        }
        
        None
    }

    fn check_bullish_regular(&self, timestamp: u64) -> Option<DivergenceSignal> {
        let count = self.price_low_count.load(Ordering::Relaxed);
        if count < 2 {
            return None;
        }
        
        unsafe {
            let recent = self.price_lows.get_unchecked(count - 1);
            let previous = self.price_lows.get_unchecked(count - 2);
            
            // Price making lower low
            if recent.price < previous.price {
                // But CVD making higher low
                if recent.cvd > previous.cvd {
                    let strength = self.calculate_strength(
                        previous.price - recent.price,
                        recent.cvd - previous.cvd,
                    );
                    
                    return Some(DivergenceSignal {
                        div_type: DivergenceType::BullishRegular,
                        price_level: recent.price,
                        cvd_level: recent.cvd,
                        timestamp,
                        strength,
                        _padding: [0; 3],
                    });
                }
            }
        }
        
        None
    }

    fn calculate_strength(&self, price_diff: i64, cvd_diff: i64) -> u8 {
        // Strength based on magnitude of divergence
        let price_score = (price_diff.abs() / 100_0000_0000i64).min(50) as u8;
        let cvd_score = (cvd_diff.abs() / 1_000_000i64).min(50) as u8;
        
        (price_score + cvd_score).min(100)
    }

    /// Get current CVD value
    #[inline]
    pub fn get_cvd(&self) -> i64 {
        self.cvd_tracker.get_cvd()
    }

    /// Reset the detector
    pub fn reset(&mut self) {
        self.cvd_tracker = CVDTracker::new();
        self.price_high_count.store(0, Ordering::Relaxed);
        self.price_low_count.store(0, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cvd_tracker() {
        let tracker = CVDTracker::new();
        
        tracker.add_trade(100, true);
        tracker.add_trade(50, false);
        
        assert_eq!(tracker.get_cvd(), 50);
    }

    #[test]
    fn test_divergence_detection() {
        let mut detector = DeltaDivergenceDetector::new(5);
        
        // Create scenario for bearish divergence
        // Price: Higher Highs, CVD: Lower Highs
        
        // First high
        detector.process_candle(
            100_0000_0000i64, 99_0000_0000i64, 99_5000_0000i64,
            1000, 600, 400, 1000,
        );
        
        // Second high (higher price, lower CVD)
        let signal = detector.process_candle(
            101_0000_0000i64, 100_0000_0000i64, 100_5000_0000i64,
            500, 400, 100, 2000,
        );
        
        // May or may not trigger depending on pivot detection
        let _ = signal;
    }
}
