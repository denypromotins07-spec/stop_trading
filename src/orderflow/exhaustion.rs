//! Exhaustion Detection
//! Identifies zero-delta or trapped-volume nodes where aggressive market orders
//! fail to move the price, triggering mean-reversion signals.

use core::sync::atomic::{AtomicI64, AtomicU64, AtomicUsize, Ordering};

/// Maximum exhaustion zones to track
const MAX_ZONES: usize = 64;

/// Exhaustion type
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u8)]
pub enum ExhaustionType {
    /// High volume but no price progress (absorption)
    Absorption = 0,
    /// Zero delta at extreme (no more sellers/buyers)
    ZeroDelta = 1,
    /// Trapped buyers (longs stuck at top)
    TrappedLongs = 2,
    /// Trapped sellers (shorts stuck at bottom)
    TrappedShorts = 3,
}

/// Detected exhaustion zone
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ExhaustionZone {
    pub zone_type: ExhaustionType,
    pub price_level: i64,
    pub trapped_volume: u64,
    pub delta: i64,
    pub timestamp: u64,
    pub confidence: u8,
    _padding: [u8; 3],
}

/// Volume node for tracking absorption
#[repr(C)]
struct VolumeNode {
    price: i64,
    total_volume: u64,
    delta: i64,
    price_change: i64,
    timestamp: u64,
}

impl Default for VolumeNode {
    fn default() -> Self {
        Self {
            price: 0,
            total_volume: 0,
            delta: 0,
            price_change: 0,
            timestamp: 0,
        }
    }
}

/// Exhaustion detector state machine
pub struct ExhaustionDetector {
    /// Recent volume nodes for absorption detection
    volume_nodes: [VolumeNode; MAX_ZONES],
    node_count: AtomicUsize,
    /// Detected exhaustion zones
    zones: [ExhaustionZone; MAX_ZONES],
    zone_count: AtomicUsize,
    /// Thresholds
    min_volume_threshold: u64,
    max_delta_ratio: i64,  // Delta/Volume ratio threshold (scaled by 1e8)
    /// Rolling window for recent activity
    recent_high: AtomicI64,
    recent_low: AtomicI64,
}

impl Default for ExhaustionDetector {
    fn default() -> Self {
        Self::new(1_000_000, 10_000_000)
    }
}

impl ExhaustionDetector {
    /// Create new detector with volume and delta thresholds
    pub const fn new(min_volume: u64, max_delta_ratio: i64) -> Self {
        Self {
            volume_nodes: unsafe { core::mem::zeroed() },
            node_count: AtomicUsize::new(0),
            zones: unsafe { core::mem::zeroed() },
            zone_count: AtomicUsize::new(0),
            min_volume_threshold: min_volume,
            max_delta_ratio,
            recent_high: AtomicI64::new(0),
            recent_low: AtomicI64::new(i64::MAX),
        }
    }

    /// Process a candle and detect exhaustion patterns
    pub fn process_candle(
        &mut self,
        open: i64,
        high: i64,
        low: i64,
        close: i64,
        volume: u64,
        delta: i64,
        timestamp: u64,
    ) -> Option<ExhaustionZone> {
        // Update recent range
        self.update_range(high, low);

        // Add volume node
        self.add_volume_node(VolumeNode {
            price: close,
            total_volume: volume,
            delta,
            price_change: close - open,
            timestamp,
        });

        let mut detected_zone: Option<ExhaustionZone> = None;

        // Check for absorption (high volume, small price change)
        if let Some(zone) = self.check_absorption(volume, close - open, close, timestamp) {
            detected_zone = Some(zone);
        }

        // Check for zero delta exhaustion
        if detected_zone.is_none() {
            if let Some(zone) = self.check_zero_delta(delta, volume, high, low, timestamp) {
                detected_zone = Some(zone);
            }
        }

        // Check for trapped positions
        if detected_zone.is_none() {
            if let Some(zone) = self.check_trapped_positions(close, high, low, delta, volume, timestamp) {
                detected_zone = Some(zone);
            }
        }

        // Record zone if detected
        if let Some(zone) = detected_zone {
            self.record_zone(zone);
            return Some(zone);
        }

        None
    }

    #[inline]
    fn update_range(&self, high: i64, low: i64) {
        let prev_high = self.recent_high.load(Ordering::Relaxed);
        let prev_low = self.recent_low.load(Ordering::Relaxed);

        if high > prev_high {
            self.recent_high.store(high, Ordering::Relaxed);
        }
        if low < prev_low {
            self.recent_low.store(low, Ordering::Relaxed);
        }
    }

    fn add_volume_node(&mut self, node: VolumeNode) {
        let count = self.node_count.load(Ordering::Relaxed);
        
        if count < MAX_ZONES {
            self.volume_nodes[count] = node;
            self.node_count.store(count + 1, Ordering::Relaxed);
        } else {
            // Shift left
            unsafe {
                core::ptr::copy(
                    self.volume_nodes.as_ptr().add(1),
                    self.volume_nodes.as_mut_ptr(),
                    MAX_ZONES - 1,
                );
            }
            self.volume_nodes[MAX_ZONES - 1] = node;
        }
    }

    /// Check for absorption pattern
    fn check_absorption(
        &self,
        volume: u64,
        price_change: i64,
        price: i64,
        timestamp: u64,
    ) -> Option<ExhaustionZone> {
        if volume < self.min_volume_threshold {
            return None;
        }

        // Calculate volume-to-progress ratio
        let price_change_abs = price_change.abs();
        if price_change_abs == 0 {
            // Perfect absorption: huge volume, zero progress
            return Some(ExhaustionZone {
                zone_type: ExhaustionType::Absorption,
                price_level: price,
                trapped_volume: volume,
                delta: 0,
                timestamp,
                confidence: 90,
                _padding: [0; 3],
            });
        }

        // Check if volume is disproportionately large vs price movement
        let volume_price_ratio = volume as i64 / price_change_abs.max(1);
        let threshold = self.min_volume_threshold as i64 / 100_0000i64;

        if volume_price_ratio > threshold {
            let confidence = ((volume_price_ratio * 100) / threshold).min(100) as u8;
            
            Some(ExhaustionZone {
                zone_type: ExhaustionType::Absorption,
                price_level: price,
                trapped_volume: volume,
                delta: 0,
                timestamp,
                confidence,
                _padding: [0; 3],
            })
        } else {
            None
        }
    }

    /// Check for zero-delta exhaustion at extremes
    fn check_zero_delta(
        &self,
        delta: i64,
        volume: u64,
        high: i64,
        low: i64,
        timestamp: u64,
    ) -> Option<ExhaustionZone> {
        if volume < self.min_volume_threshold / 2 {
            return None;
        }

        // Calculate delta ratio (delta / volume)
        let delta_ratio = (delta.abs() as i64 * 1_000_000_000) / volume as i64;

        // Near-zero delta with significant volume
        if delta_ratio < self.max_delta_ratio {
            let recent_high = self.recent_high.load(Ordering::Relaxed);
            let recent_low = self.recent_low.load(Ordering::Relaxed);

            // Check if at extreme
            let near_high = (recent_high - high).abs() < recent_high / 1000; // Within 0.1%
            let near_low = (low - recent_low).abs() < recent_low / 1000;

            if near_high || near_low {
                let zone_type = if near_high {
                    ExhaustionType::ZeroDelta
                } else {
                    ExhaustionType::ZeroDelta
                };

                return Some(ExhaustionZone {
                    zone_type,
                    price_level: if near_high { high } else { low },
                    trapped_volume: volume,
                    delta,
                    timestamp,
                    confidence: 70,
                    _padding: [0; 3],
                });
            }
        }

        None
    }

    /// Check for trapped positions
    fn check_trapped_positions(
        &self,
        close: i64,
        high: i64,
        low: i64,
        delta: i64,
        volume: u64,
        timestamp: u64,
    ) -> Option<ExhaustionZone> {
        let range = high - low;
        if range == 0 {
            return None;
        }

        // Upper wick ratio (trapped longs)
        let upper_wick = high - close;
        let upper_wick_ratio = (upper_wick * 100) / range;

        // Lower wick ratio (trapped shorts)
        let lower_wick = close - low;
        let lower_wick_ratio = (lower_wick * 100) / range;

        // Trapped longs: long upper wick, positive delta (buyers got trapped)
        if upper_wick_ratio > 50 && delta > 0 && volume >= self.min_volume_threshold / 2 {
            let confidence = (upper_wick_ratio / 2).min(100) as u8;
            
            return Some(ExhaustionZone {
                zone_type: ExhaustionType::TrappedLongs,
                price_level: high,
                trapped_volume: volume,
                delta,
                timestamp,
                confidence,
                _padding: [0; 3],
            });
        }

        // Trapped shorts: long lower wick, negative delta (sellers got trapped)
        if lower_wick_ratio > 50 && delta < 0 && volume >= self.min_volume_threshold / 2 {
            let confidence = (lower_wick_ratio / 2).min(100) as u8;
            
            return Some(ExhaustionZone {
                zone_type: ExhaustionType::TrappedShorts,
                price_level: low,
                trapped_volume: volume,
                delta,
                timestamp,
                confidence,
                _padding: [0; 3],
            });
        }

        None
    }

    fn record_zone(&mut self, zone: ExhaustionZone) {
        let count = self.zone_count.load(Ordering::Relaxed);
        
        if count < MAX_ZONES {
            self.zones[count] = zone;
            self.zone_count.store(count + 1, Ordering::Relaxed);
        } else {
            unsafe {
                core::ptr::copy(
                    self.zones.as_ptr().add(1),
                    self.zones.as_mut_ptr(),
                    MAX_ZONES - 1,
                );
            }
            self.zones[MAX_ZONES - 1] = zone;
        }
    }

    /// Get all detected exhaustion zones
    pub fn get_zones(&self) -> &[ExhaustionZone] {
        let count = self.zone_count.load(Ordering::Relaxed);
        unsafe { core::slice::from_raw_parts(self.zones.as_ptr(), count) }
    }

    /// Check if price is near an exhaustion zone
    pub fn is_near_exhaustion(&self, price: i64, tolerance_bps: i64) -> Option<&ExhaustionZone> {
        let count = self.zone_count.load(Ordering::Relaxed);
        
        for i in 0..count {
            unsafe {
                let zone = self.zones.get_unchecked(i);
                let diff = (price - zone.price_level).abs();
                let threshold = (zone.price_level.abs() * tolerance_bps) / 10000;
                
                if diff <= threshold.max(1) {
                    return Some(zone);
                }
            }
        }
        
        None
    }

    /// Reset the detector
    pub fn reset(&mut self) {
        self.node_count.store(0, Ordering::Relaxed);
        self.zone_count.store(0, Ordering::Relaxed);
        self.recent_high.store(0, Ordering::Relaxed);
        self.recent_low.store(i64::MAX, Ordering::Relaxed);
    }

    /// Generate mean-reversion signal based on exhaustion
    pub fn get_mean_reversion_signal(&self, current_price: i64) -> Option<i8> {
        // Look for recent exhaustion zones
        if let Some(zone) = self.is_near_exhaustion(current_price, 50) {
            match zone.zone_type {
                ExhaustionType::TrappedLongs | ExhaustionType::Absorption => {
                    // Expect downward mean reversion
                    Some(-1)
                }
                ExhaustionType::TrappedShorts => {
                    // Expect upward mean reversion
                    Some(1)
                }
                ExhaustionType::ZeroDelta => {
                    // Neutral - wait for confirmation
                    None
                }
            }
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_absorption_detection() {
        let mut detector = ExhaustionDetector::new(100_000, 1_000_000);
        
        // High volume, no price change = absorption
        let result = detector.process_candle(
            100_0000_0000i64,  // open
            100_0000_0000i64,  // high
            100_0000_0000i64,  // low
            100_0000_0000i64,  // close
            1_000_000,         // volume (high)
            0,                 // delta
            1000,              // timestamp
        );
        
        assert!(result.is_some());
        assert_eq!(result.unwrap().zone_type, ExhaustionType::Absorption);
    }

    #[test]
    fn test_trapped_longs() {
        let mut detector = ExhaustionDetector::new(100_000, 1_000_000);
        
        // Long upper wick, positive delta
        let result = detector.process_candle(
            99_0000_0000i64,   // open
            101_0000_0000i64,  // high
            99_0000_0000i64,   // low
            99_2000_0000i64,   // close (near low)
            500_000,           // volume
            100_000,           // positive delta
            1000,
        );
        
        assert!(result.is_some());
        assert_eq!(result.unwrap().zone_type, ExhaustionType::TrappedLongs);
    }
}
