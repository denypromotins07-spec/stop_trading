//! Liquidity Evaporation Index
//! 
//! Tracks sudden withdrawal of top-of-book depth across all venues.
//! Instantly widens execution slippage tolerances or halts trading if global liquidity drops.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

/// Maximum number of venues to track
pub const MAX_VENUES: usize = 20;

/// Liquidity state levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LiquidityLevel {
    Normal = 0,
    Reduced = 1,
    Stressed = 2,
    Evaporated = 3,
}

/// Liquidity evaporation signal
#[derive(Debug, Clone, Copy)]
pub struct LiquiditySignal {
    /// Current liquidity level
    pub level: LiquidityLevel,
    /// Global liquidity score (0-1, 1=full)
    pub liquidity_score: f64,
    /// Rate of change (negative = deteriorating)
    pub rate_of_change: f64,
    /// Recommended slippage multiplier
    pub slippage_multiplier: f64,
    /// Whether to halt trading
    pub should_halt: bool,
    /// Number of stressed venues
    pub stressed_venues: u8,
    /// Timestamp in nanoseconds
    pub timestamp_ns: u64,
}

impl Default for LiquiditySignal {
    fn default() -> Self {
        Self {
            level: LiquidityLevel::Normal,
            liquidity_score: 1.0,
            rate_of_change: 0.0,
            slippage_multiplier: 1.0,
            should_halt: false,
            stressed_venues: 0,
            timestamp_ns: 0,
        }
    }
}

/// Venue liquidity state
#[derive(Debug, Clone)]
struct VenueState {
    /// Venue identifier hash
    venue_id: u64,
    /// Normal/expected depth (base units)
    normal_depth: u64,
    /// Current depth
    current_depth: u64,
    /// Depth ratio (current/normal)
    depth_ratio: f64,
    /// Last update timestamp
    last_update_ns: u64,
}

impl VenueState {
    fn new(venue_id: u64, normal_depth: u64) -> Self {
        Self {
            venue_id,
            normal_depth,
            current_depth: normal_depth,
            depth_ratio: 1.0,
            last_update_ns: 0,
        }
    }

    fn update(&mut self, current_depth: u64, timestamp_ns: u64) {
        self.current_depth = current_depth;
        self.depth_ratio = if self.normal_depth > 0 {
            current_depth as f64 / self.normal_depth as f64
        } else {
            0.0
        };
        self.last_update_ns = timestamp_ns;
    }
}

/// Cache-line aligned liquidity evaporation detector
#[repr(align(64))]
pub struct LiquidityEvaporationDetector {
    /// Venue states
    venues: [Option<VenueState>; MAX_VENUES],
    /// Number of tracked venues
    venue_count: usize,
    /// Previous global score for rate calculation
    prev_global_score: f64,
    /// Score threshold for reduced liquidity
    reduced_threshold: f64,
    /// Score threshold for stressed liquidity
    stressed_threshold: f64,
    /// Score threshold for evaporated (halt)
    evaporated_threshold: f64,
    /// Halting enabled flag
    halt_enabled: AtomicBool,
    /// Currently halted
    halted: AtomicBool,
    /// Total updates count
    updates_count: AtomicU64,
    _pad: [u8; 32],
}

unsafe impl Send for LiquidityEvaporationDetector {}
unsafe impl Sync for LiquidityEvaporationDetector {}

impl LiquidityEvaporationDetector {
    /// Create new liquidity evaporation detector
    pub fn new(
        reduced_threshold: f64,
        stressed_threshold: f64,
        evaporated_threshold: f64,
    ) -> Self {
        Self {
            venues: std::array::from_fn(|_| None),
            venue_count: 0,
            prev_global_score: 1.0,
            reduced_threshold,
            stressed_threshold,
            evaporated_threshold,
            halt_enabled: AtomicBool::new(true),
            halted: AtomicBool::new(false),
            updates_count: AtomicU64::new(0),
            _pad: [0; 32],
        }
    }

    /// Register venue for liquidity tracking
    pub fn register_venue(&mut self, venue_id: u64, normal_depth: u64) -> bool {
        if self.venue_count >= MAX_VENUES {
            return false;
        }

        self.venues[self.venue_count] = Some(VenueState::new(venue_id, normal_depth));
        self.venue_count += 1;

        true
    }

    /// Update liquidity depth for a venue
    pub fn update_venue_liquidity(&mut self, venue_id: u64, depth: u64) {
        let timestamp_ns = Instant::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos() as u64;

        for i in 0..self.venue_count {
            if let Some(ref mut venue) = self.venues[i] {
                if venue.venue_id == venue_id {
                    venue.update(depth, timestamp_ns);
                    self.updates_count.fetch_add(1, Ordering::Relaxed);
                    return;
                }
            }
        }
    }

    /// Calculate global liquidity score
    fn calculate_global_score(&self) -> f64 {
        if self.venue_count == 0 {
            return 1.0;
        }

        let mut total_ratio = 0.0f64;
        let mut valid_venues = 0usize;

        for i in 0..self.venue_count {
            if let Some(ref venue) = self.venues[i] {
                total_ratio += venue.depth_ratio;
                valid_venues += 1;
            }
        }

        if valid_venues == 0 {
            return 1.0;
        }

        total_ratio / valid_venues as f64
    }

    /// Count stressed venues
    fn count_stressed_venues(&self) -> u8 {
        let mut count = 0u8;

        for i in 0..self.venue_count {
            if let Some(ref venue) = self.venues[i] {
                if venue.depth_ratio < self.stressed_threshold {
                    count += 1;
                }
            }
        }

        count
    }

    /// Detect liquidity evaporation
    pub fn detect(&mut self) -> LiquiditySignal {
        let timestamp_ns = Instant::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos() as u64;

        let mut signal = LiquiditySignal::default();
        signal.timestamp_ns = timestamp_ns;

        // Calculate global score
        let current_score = self.calculate_global_score();
        signal.liquidity_score = current_score;

        // Calculate rate of change
        signal.rate_of_change = current_score - self.prev_global_score;
        self.prev_global_score = current_score;

        // Count stressed venues
        signal.stressed_venues = self.count_stressed_venues();

        // Determine liquidity level
        signal.level = if current_score <= self.evaporated_threshold {
            LiquidityLevel::Evaporated
        } else if current_score <= self.stressed_threshold {
            LiquidityLevel::Stressed
        } else if current_score <= self.reduced_threshold {
            LiquidityLevel::Reduced
        } else {
            LiquidityLevel::Normal
        };

        // Calculate slippage multiplier based on level
        signal.slippage_multiplier = match signal.level {
            LiquidityLevel::Normal => 1.0,
            LiquidityLevel::Reduced => 1.5,
            LiquidityLevel::Stressed => 2.5,
            LiquidityLevel::Evaporated => 5.0,
        };

        // Adjust multiplier based on rate of change (rapid deterioration = higher multiplier)
        if signal.rate_of_change < -0.1 {
            signal.slippage_multiplier *= 1.5;
        }

        // Determine if should halt
        signal.should_halt = signal.level == LiquidityLevel::Evaporated 
            && self.halt_enabled.load(Ordering::Relaxed);

        // Update halt status
        if signal.should_halt {
            self.halted.store(true, Ordering::Release);
        } else if signal.level == LiquidityLevel::Normal {
            self.halted.store(false, Ordering::Relaxed);
        }

        signal
    }

    /// Check if currently halted
    #[inline]
    pub fn is_halted(&self) -> bool {
        self.halted.load(Ordering::Acquire)
    }

    /// Enable/disable automatic halting
    #[inline]
    pub fn set_halt_enabled(&self, enabled: bool) {
        self.halt_enabled.store(enabled, Ordering::Relaxed);
    }

    /// Clear halt status manually
    #[inline]
    pub fn clear_halt(&self) {
        self.halted.store(false, Ordering::Release);
    }

    /// Get updates count
    #[inline]
    pub fn updates_count(&self) -> u64 {
        self.updates_count.load(Ordering::Relaxed)
    }

    /// Set thresholds
    pub fn set_thresholds(
        &mut self,
        reduced: f64,
        stressed: f64,
        evaporated: f64,
    ) {
        self.reduced_threshold = reduced;
        self.stressed_threshold = stressed;
        self.evaporated_threshold = evaporated;
    }

    /// Reset detector state
    pub fn reset(&mut self) {
        for i in 0..self.venue_count {
            if let Some(ref mut venue) = self.venues[i] {
                venue.current_depth = venue.normal_depth;
                venue.depth_ratio = 1.0;
            }
        }
        self.prev_global_score = 1.0;
        self.clear_halt();
    }
}

/// Builder for liquidity evaporation detector
pub struct LiquidityDetectorBuilder {
    reduced_threshold: f64,
    stressed_threshold: f64,
    evaporated_threshold: f64,
}

impl LiquidityDetectorBuilder {
    pub fn new() -> Self {
        Self {
            reduced_threshold: 0.7,
            stressed_threshold: 0.4,
            evaporated_threshold: 0.2,
        }
    }

    pub fn thresholds(mut self, reduced: f64, stressed: f64, evaporated: f64) -> Self {
        self.reduced_threshold = reduced;
        self.stressed_threshold = stressed;
        self.evaporated_threshold = evaporated;
        self
    }

    pub fn build(self) -> LiquidityEvaporationDetector {
        LiquidityEvaporationDetector::new(
            self.reduced_threshold,
            self.stressed_threshold,
            self.evaporated_threshold,
        )
    }
}

impl Default for LiquidityDetectorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normal_liquidity() {
        let mut detector = LiquidityDetectorBuilder::new().build();
        
        detector.register_venue(1, 1000000);
        detector.register_venue(2, 1000000);

        // Update with normal depths
        detector.update_venue_liquidity(1, 950000);
        detector.update_venue_liquidity(2, 980000);

        let signal = detector.detect();

        assert_eq!(signal.level, LiquidityLevel::Normal);
        assert!(signal.liquidity_score > 0.9);
        assert!(!signal.should_halt);
    }

    #[test]
    fn test_evaporated_liquidity() {
        let mut detector = LiquidityDetectorBuilder::new().build();
        
        detector.register_venue(1, 1000000);
        detector.register_venue(2, 1000000);
        detector.register_venue(3, 1000000);

        // Update with very low depths (evaporation)
        detector.update_venue_liquidity(1, 100000);
        detector.update_venue_liquidity(2, 150000);
        detector.update_venue_liquidity(3, 80000);

        let signal = detector.detect();

        assert_eq!(signal.level, LiquidityLevel::Evaporated);
        assert!(signal.liquidity_score < 0.2);
        assert!(signal.should_halt);
        assert!(detector.is_halted());
    }

    #[test]
    fn test_slippage_multiplier() {
        let mut detector = LiquidityDetectorBuilder::new().build();
        
        detector.register_venue(1, 1000000);

        // Stressed level
        detector.update_venue_liquidity(1, 300000);
        let signal = detector.detect();

        assert!(signal.slippage_multiplier > 2.0);
    }

    #[test]
    fn test_halt_override() {
        let mut detector = LiquidityDetectorBuilder::new().build();
        detector.set_halt_enabled(false);
        
        detector.register_venue(1, 1000000);
        detector.update_venue_liquidity(1, 50000);

        let signal = detector.detect();

        // Should detect evaporated but not halt due to disabled setting
        assert_eq!(signal.level, LiquidityLevel::Evaporated);
        assert!(!signal.should_halt);
    }
}
