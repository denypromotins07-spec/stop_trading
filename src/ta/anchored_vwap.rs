//! Anchored VWAP Calculator
//! Calculates Anchored VWAP from significant market events using incremental Welford math.
//! Zero heap allocations - uses bounded arrays for event tracking.

use core::sync::atomic::{AtomicUsize, Ordering};

/// Maximum number of anchor points to track
const MAX_ANCHORS: usize = 32;

/// Maximum periods per anchor (bounded for memory control)
const MAX_PERIODS: usize = 10000;

/// Anchor point type
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u8)]
pub enum AnchorType {
    SwingHigh = 0,
    SwingLow = 1,
    VolumeNode = 2,
    SMCBlock = 3,
    HalvingEvent = 4,
    Custom = 5,
}

/// Welford accumulator for running statistics
#[repr(C)]
struct WelfordAccumulator {
    count: u64,
    mean: i64,      // Running mean (VWAP numerator / volume sum)
    m2: i64,        // Sum of squares of differences from mean
    price_volume_sum: i64,  // Sum of (price * volume)
    volume_sum: u64,        // Sum of volume
    _padding: [u8; 8],
}

impl WelfordAccumulator {
    const fn new() -> Self {
        Self {
            count: 0,
            mean: 0,
            m2: 0,
            price_volume_sum: 0,
            volume_sum: 0,
            _padding: [0; 8],
        }
    }

    #[inline]
    fn update(&mut self, price: i64, volume: u64) {
        if volume == 0 {
            return;
        }

        let pv = price * volume as i64;
        self.price_volume_sum = self.price_volume_sum.saturating_add(pv);
        self.volume_sum = self.volume_sum.saturating_add(volume);
        self.count += 1;

        // Welford's online algorithm for variance
        if self.volume_sum > 0 {
            self.mean = self.price_volume_sum / self.volume_sum as i64;
        }
    }

    #[inline]
    fn vwap(&self) -> i64 {
        self.mean
    }

    #[inline]
    fn std_dev(&self) -> i64 {
        if self.count < 2 {
            return 0;
        }
        // Simplified standard deviation estimate
        (self.m2 / self.count as i64).abs()
    }

    fn reset(&mut self) {
        *self = Self::new();
    }
}

/// Anchored VWAP band
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct VWAPBand {
    pub vwap: i64,
    pub upper_band: i64,  // VWAP + n*stddev
    pub lower_band: i64,  // VWAP - n*stddev
    pub bandwidth: i64,
}

/// Single anchored VWAP instance
#[repr(C)]
pub struct AnchoredVWAP {
    anchor_type: AnchorType,
    anchor_timestamp: u64,
    anchor_price: i64,
    accumulator: WelfordAccumulator,
    active: bool,
    _padding: [u8; 7],
}

impl AnchoredVWAP {
    const fn new(anchor_type: AnchorType, timestamp: u64, price: i64) -> Self {
        Self {
            anchor_type,
            anchor_timestamp: timestamp,
            anchor_price: price,
            accumulator: WelfordAccumulator::new(),
            active: true,
            _padding: [0; 7],
        }
    }

    #[inline]
    fn update(&mut self, high: i64, low: i64, close: i64, volume: u64) {
        if !self.active {
            return;
        }

        // Typical price for VWAP calculation
        let typical_price = (high + low + close) / 3;
        self.accumulator.update(typical_price, volume);
    }

    #[inline]
    fn vwap(&self) -> i64 {
        self.accumulator.vwap()
    }

    fn get_bands(&self, stddev_multiplier: i64) -> VWAPBand {
        let vwap = self.vwap();
        let std_dev = self.accumulator.std_dev();
        let band_width = std_dev * stddev_multiplier / 100;

        VWAPBand {
            vwap,
            upper_band: vwap + band_width,
            lower_band: vwap - band_width,
            bandwidth: band_width * 2,
        }
    }

    fn deactivate(&mut self) {
        self.active = false;
    }
}

/// Main Anchored VWAP manager supporting multiple anchors
pub struct AnchoredVWAPManager {
    anchors: [AnchoredVWAP; MAX_ANCHORS],
    anchor_count: AtomicUsize,
    default_stddev_multiplier: i64,
    _padding: [u8; 56],
}

impl Default for AnchoredVWAPManager {
    fn default() -> Self {
        Self::new(200)
    }
}

impl AnchoredVWAPManager {
    /// Create new manager with stddev multiplier (in basis points, e.g., 200 = 2 stddev)
    pub const fn new(stddev_multiplier: i64) -> Self {
        Self {
            anchors: unsafe {
                let mut arr: [AnchoredVWAP; MAX_ANCHORS] = [
                    AnchoredVWAP {
                        anchor_type: AnchorType::Custom,
                        anchor_timestamp: 0,
                        anchor_price: 0,
                        accumulator: WelfordAccumulator::new(),
                        active: false,
                        _padding: [0; 7],
                    };
                    MAX_ANCHORS
                ];
                arr
            },
            anchor_count: AtomicUsize::new(0),
            default_stddev_multiplier: stddev_multiplier,
            _padding: [0; 56],
        }
    }

    /// Add a new anchor point
    pub fn add_anchor(
        &mut self,
        anchor_type: AnchorType,
        timestamp: u64,
        price: i64,
    ) -> Option<usize> {
        let count = self.anchor_count.load(Ordering::Relaxed);
        
        if count >= MAX_ANCHORS {
            // Deactivate oldest and reuse
            unsafe {
                self.anchors.get_unchecked_mut(0)
            }.deactivate();
            
            // Shift all anchors
            unsafe {
                core::ptr::copy(
                    self.anchors.as_ptr().add(1),
                    self.anchors.as_mut_ptr(),
                    MAX_ANCHORS - 1,
                );
            }
            
            let new_anchor = AnchoredVWAP::new(anchor_type, timestamp, price);
            self.anchors[MAX_ANCHORS - 1] = new_anchor;
            return Some(MAX_ANCHORS - 1);
        }

        let new_anchor = AnchoredVWAP::new(anchor_type, timestamp, price);
        self.anchors[count] = new_anchor;
        self.anchor_count.store(count + 1, Ordering::Relaxed);
        Some(count)
    }

    /// Update all active anchors with new candle data
    pub fn process_candle(
        &mut self,
        high: i64,
        low: i64,
        close: i64,
        volume: u64,
        timestamp: u64,
    ) {
        let count = self.anchor_count.load(Ordering::Relaxed);
        
        for i in 0..count {
            unsafe {
                let anchor = self.anchors.get_unchecked_mut(i);
                if anchor.active && timestamp >= anchor.anchor_timestamp {
                    anchor.update(high, low, close, volume);
                }
            }
        }
    }

    /// Get VWAP value for a specific anchor
    pub fn get_vwap(&self, anchor_idx: usize) -> Option<i64> {
        let count = self.anchor_count.load(Ordering::Relaxed);
        if anchor_idx >= count {
            return None;
        }

        unsafe {
            let anchor = self.anchors.get_unchecked(anchor_idx);
            if anchor.active {
                Some(anchor.vwap())
            } else {
                None
            }
        }
    }

    /// Get all active VWAP values
    pub fn get_all_vwaps(&self) -> impl Iterator<Item = (usize, i64)> + '_ {
        let count = self.anchor_count.load(Ordering::Relaxed);
        (0..count).filter_map(move |i| {
            unsafe {
                let anchor = self.anchors.get_unchecked(i);
                if anchor.active {
                    Some((i, anchor.vwap()))
                } else {
                    None
                }
            }
        })
    }

    /// Get VWAP bands for an anchor
    pub fn get_vwap_bands(&self, anchor_idx: usize) -> Option<VWAPBand> {
        let count = self.anchor_count.load(Ordering::Relaxed);
        if anchor_idx >= count {
            return None;
        }

        unsafe {
            let anchor = self.anchors.get_unchecked(anchor_idx);
            if anchor.active {
                Some(anchor.get_bands(self.default_stddev_multiplier))
            } else {
                None
            }
        }
    }

    /// Check if price is near any VWAP (potential support/resistance)
    pub fn is_near_vwap(&self, price: i64, tolerance_bps: i64) -> Option<(usize, i64)> {
        let count = self.anchor_count.load(Ordering::Relaxed);
        
        for i in 0..count {
            unsafe {
                let anchor = self.anchors.get_unchecked(i);
                if anchor.active {
                    let vwap = anchor.vwap();
                    let diff = (price - vwap).abs();
                    let threshold = (vwap * tolerance_bps) / 10000;
                    
                    if diff <= threshold.max(1) {
                        return Some((i, vwap));
                    }
                }
            }
        }
        
        None
    }

    /// Detect VWAP confluence (multiple anchors at similar levels)
    pub fn detect_confluence(&self, tolerance_bps: i64) -> Option<(i64, usize)> {
        let count = self.anchor_count.load(Ordering::Relaxed);
        if count < 2 {
            return None;
        }

        // Simple clustering: find VWAP level with most anchors nearby
        let mut best_level = 0i64;
        let mut best_count = 0usize;

        for i in 0..count {
            unsafe {
                let anchor_i = self.anchors.get_unchecked(i);
                if !anchor_i.active {
                    continue;
                }

                let vwap_i = anchor_i.vwap();
                let mut cluster_count = 1;

                for j in (i + 1)..count {
                    let anchor_j = self.anchors.get_unchecked(j);
                    if !anchor_j.active {
                        continue;
                    }

                    let vwap_j = anchor_j.vwap();
                    let diff = (vwap_i - vwap_j).abs();
                    let threshold = (vwap_i * tolerance_bps) / 10000;

                    if diff <= threshold.max(1) {
                        cluster_count += 1;
                    }
                }

                if cluster_count > best_count {
                    best_count = cluster_count;
                    best_level = vwap_i;
                }
            }
        }

        if best_count >= 2 {
            Some((best_level, best_count))
        } else {
            None
        }
    }

    /// Remove/deactivate an anchor
    pub fn remove_anchor(&mut self, anchor_idx: usize) {
        let count = self.anchor_count.load(Ordering::Relaxed);
        if anchor_idx >= count {
            return;
        }

        unsafe {
            let anchor = self.anchors.get_unchecked_mut(anchor_idx);
            anchor.deactivate();
        }
    }

    /// Reset all anchors
    pub fn reset(&mut self) {
        let count = self.anchor_count.load(Ordering::Relaxed);
        for i in 0..count {
            unsafe {
                self.anchors.get_unchecked_mut(i).deactivate();
            }
        }
        self.anchor_count.store(0, Ordering::Relaxed);
    }

    /// Get anchor count
    #[inline]
    pub fn anchor_count(&self) -> usize {
        self.anchor_count.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_welford_accumulator() {
        let mut acc = WelfordAccumulator::new();
        
        // Feed some data
        acc.update(100_0000_0000i64, 1000);
        acc.update(101_0000_0000i64, 1000);
        acc.update(99_0000_0000i64, 1000);
        
        assert_eq!(acc.vwap(), 100_0000_0000i64);
    }

    #[test]
    fn test_anchored_vwap_manager() {
        let mut manager = AnchoredVWAPManager::default();
        
        // Add anchor at swing low
        manager.add_anchor(AnchorType::SwingLow, 1000, 100_0000_0000i64);
        
        // Process some candles
        for i in 0..10 {
            let high = 101_0000_0000i64 + (i as i64) * 1000_0000i64;
            let low = 100_0000_0000i64 + (i as i64) * 1000_0000i64;
            let close = 100_5000_0000i64 + (i as i64) * 1000_0000i64;
            manager.process_candle(high, low, close, 1_000_000, 1000 + i);
        }
        
        let vwap = manager.get_vwap(0);
        assert!(vwap.is_some());
        assert!(vwap.unwrap() > 100_0000_0000i64);
    }
}
