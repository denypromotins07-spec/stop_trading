//! Trailing Stop Engine with Microsecond Precision
//! 
//! Implements lock-free atomic price trackers for dynamic stop price adjustment
//! based on real-time ATR and tick movements without excessive REST amendment calls.

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::time::{Duration, Instant};
use crossbeam_utils::CachePadded;

/// Lock-free trailing stop tracker using atomic operations
pub struct TrailingStopEngine {
    /// Current best price (in micro-units for precision)
    best_price: CachePadded<AtomicU64>,
    /// Current stop price (in micro-units)
    stop_price: CachePadded<AtomicU64>,
    /// Trailing distance in basis points (e.g., 50 = 0.5%)
    trailing_distance_bps: CachePadded<AtomicU64>,
    /// ATR-based multiplier (scaled by 1000)
    atr_multiplier: CachePadded<AtomicU64>,
    /// Current ATR value (in micro-units)
    current_atr: CachePadded<AtomicU64>,
    /// Direction: true = long, false = short
    is_long: CachePadded<AtomicBool>,
    /// Whether the stop has been triggered
    triggered: CachePadded<AtomicBool>,
    /// Last update timestamp (nanoseconds since epoch)
    last_update_ns: CachePadded<AtomicU64>,
    /// Minimum update interval to avoid excessive amendments (nanoseconds)
    min_update_interval_ns: u64,
    /// Triggered stop price (for post-trigger analysis)
    triggered_price: CachePadded<AtomicU64>,
}

impl TrailingStopEngine {
    /// Create a new trailing stop engine
    /// 
    /// # Arguments
    /// * `initial_price` - Starting price in micro-units (e.g., 50000000000 for $50,000)
    /// * `trailing_distance_bps` - Trailing distance in basis points
    /// * `atr_multiplier` - ATR multiplier scaled by 1000 (e.g., 2000 = 2.0x)
    /// * `is_long` - Direction of the position
    /// * `min_update_interval_ms` - Minimum interval between stop amendments
    pub fn new(
        initial_price: u64,
        trailing_distance_bps: u64,
        atr_multiplier: u64,
        is_long: bool,
        min_update_interval_ms: u64,
    ) -> Self {
        let initial_stop = Self::calculate_stop_price(initial_price, trailing_distance_bps, 0, is_long);
        
        Self {
            best_price: CachePadded::new(AtomicU64::new(initial_price)),
            stop_price: CachePadded::new(AtomicU64::new(initial_stop)),
            trailing_distance_bps: CachePadded::new(AtomicU64::new(trailing_distance_bps)),
            atr_multiplier: CachePadded::new(AtomicU64::new(atr_multiplier)),
            current_atr: CachePadded::new(AtomicU64::new(0)),
            is_long: CachePadded::new(AtomicBool::new(is_long)),
            triggered: CachePadded::new(AtomicBool::new(false)),
            last_update_ns: CachePadded::new(AtomicU64::new(0)),
            min_update_interval_ns: min_update_interval_ms * 1_000_000,
            triggered_price: CachePadded::new(AtomicU64::new(0)),
        }
    }

    #[inline]
    fn calculate_stop_price(price: u64, trailing_bps: u64, atr_value: u64, is_long: bool) -> u64 {
        // Combine fixed trailing distance with ATR-based component
        // stop_distance = max(fixed_bps, atr_multiplier * atr)
        let atr_component = (atr_value / 1000) * (atr_value / 1000); // Simplified ATR contribution
        
        let total_distance = if atr_value > 0 {
            // Use the larger of fixed trailing or ATR-based distance
            let fixed_distance = (price as u128 * trailing_bps as u128) / 10000;
            let atr_distance = (atr_value as u128 * 2) / 1000; // ATR * multiplier simplified
            std::cmp::max(fixed_distance, atr_distance) as u64
        } else {
            (price as u128 * trailing_bps as u128) / 10000 as u128
        };

        if is_long {
            price.saturating_sub(total_distance)
        } else {
            price.saturating_add(total_distance)
        }
    }

    /// Update the current market price atomically
    /// Returns true if the stop price was adjusted
    pub fn update_price(&self, new_price: u64) -> bool {
        let now_ns = Instant::now().duration_since(Instant::now()).as_nanos() as u64;
        let last_update = self.last_update_ns.load(Ordering::Relaxed);
        
        // Check minimum update interval (avoid excessive amendments)
        if now_ns.saturating_sub(last_update) < self.min_update_interval_ns {
            // Still check for trigger even if not updating stop
            self.check_trigger(new_price);
            return false;
        }

        let current_best = self.best_price.load(Ordering::Relaxed);
        let is_long = self.is_long.load(Ordering::Relaxed);
        
        let should_update = if is_long {
            new_price > current_best
        } else {
            new_price < current_best && new_price > 0
        };

        if should_update {
            self.best_price.store(new_price, Ordering::Relaxed);
            
            let atr = self.current_atr.load(Ordering::Relaxed);
            let trailing_bps = self.trailing_distance_bps.load(Ordering::Relaxed);
            
            let new_stop = Self::calculate_stop_price(new_price, trailing_bps, atr, is_long);
            
            // Only move stop in favorable direction
            let current_stop = self.stop_price.load(Ordering::Relaxed);
            let should_move_stop = if is_long {
                new_stop > current_stop
            } else {
                new_stop < current_stop && new_stop > 0
            };

            if should_move_stop {
                self.stop_price.store(new_stop, Ordering::Relaxed);
                self.last_update_ns.store(now_ns, Ordering::Relaxed);
                return true;
            }
        }

        self.check_trigger(new_price);
        false
    }

    /// Update the ATR value dynamically
    pub fn update_atr(&self, new_atr: u64) {
        self.current_atr.store(new_atr, Ordering::Relaxed);
        
        // Recalculate stop price with new ATR
        let current_price = self.best_price.load(Ordering::Relaxed);
        let trailing_bps = self.trailing_distance_bps.load(Ordering::Relaxed);
        let is_long = self.is_long.load(Ordering::Relaxed);
        
        let new_stop = Self::calculate_stop_price(current_price, trailing_bps, new_atr, is_long);
        
        let current_stop = self.stop_price.load(Ordering::Relaxed);
        let should_move_stop = if is_long {
            new_stop > current_stop
        } else {
            new_stop < current_stop && new_stop > 0
        };

        if should_move_stop {
            self.stop_price.store(new_stop, Ordering::Relaxed);
        }
    }

    /// Check if the stop has been triggered
    #[inline]
    pub fn check_trigger(&self, current_price: u64) -> bool {
        if self.triggered.load(Ordering::Relaxed) {
            return true;
        }

        let stop_price = self.stop_price.load(Ordering::Relaxed);
        let is_long = self.is_long.load(Ordering::Relaxed);

        let triggered = if is_long {
            current_price <= stop_price
        } else {
            current_price >= stop_price && stop_price > 0
        };

        if triggered {
            self.triggered.store(true, Ordering::Relaxed);
            self.triggered_price.store(current_price, Ordering::Relaxed);
        }

        triggered
    }

    /// Get the current stop price
    #[inline]
    pub fn get_stop_price(&self) -> u64 {
        self.stop_price.load(Ordering::Relaxed)
    }

    /// Get the current best price
    #[inline]
    pub fn get_best_price(&self) -> u64 {
        self.best_price.load(Ordering::Relaxed)
    }

    /// Check if stop has been triggered
    #[inline]
    pub fn is_triggered(&self) -> bool {
        self.triggered.load(Ordering::Relaxed)
    }

    /// Get the triggered price (only valid after trigger)
    #[inline]
    pub fn get_triggered_price(&self) -> u64 {
        self.triggered_price.load(Ordering::Relaxed)
    }

    /// Update trailing distance dynamically (in basis points)
    pub fn update_trailing_distance(&self, new_distance_bps: u64) {
        self.trailing_distance_bps.store(new_distance_bps, Ordering::Relaxed);
        
        // Recalculate stop with new distance
        let current_price = self.best_price.load(Ordering::Relaxed);
        let atr = self.current_atr.load(Ordering::Relaxed);
        let is_long = self.is_long.load(Ordering::Relaxed);
        
        let new_stop = Self::calculate_stop_price(current_price, new_distance_bps, atr, is_long);
        self.stop_price.store(new_stop, Ordering::Relaxed);
    }

    /// Reset the trailing stop (for re-entry scenarios)
    pub fn reset(&self, new_initial_price: u64) {
        let trailing_bps = self.trailing_distance_bps.load(Ordering::Relaxed);
        let atr = self.current_atr.load(Ordering::Relaxed);
        let is_long = self.is_long.load(Ordering::Relaxed);
        
        let new_stop = Self::calculate_stop_price(new_initial_price, trailing_bps, atr, is_long);
        
        self.best_price.store(new_initial_price, Ordering::Relaxed);
        self.stop_price.store(new_stop, Ordering::Relaxed);
        self.triggered.store(false, Ordering::Relaxed);
        self.triggered_price.store(0, Ordering::Relaxed);
        self.last_update_ns.store(0, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trailing_stop_long() {
        let engine = TrailingStopEngine::new(50000000, 50, 2000, true, 100);
        
        assert_eq!(engine.get_best_price(), 50000000);
        assert!(!engine.is_triggered());
        
        // Price moves up - stop should trail
        engine.update_price(51000000);
        assert_eq!(engine.get_best_price(), 51000000);
        
        // Price drops below stop - should trigger
        engine.update_price(49000000);
        assert!(engine.is_triggered());
    }

    #[test]
    fn test_trailing_stop_short() {
        let engine = TrailingStopEngine::new(50000000, 50, 2000, false, 100);
        
        // Price moves down - stop should trail
        engine.update_price(49000000);
        assert_eq!(engine.get_best_price(), 49000000);
    }
}
