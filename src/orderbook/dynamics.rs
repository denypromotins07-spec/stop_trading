//! Advanced Order Book Dynamics & Shape Analytics
//! 
//! Implements order book shape analysis (convexity/concavity) to predict short-term 
//! directional microprice drift using lock-free atomic arrays.

use std::sync::atomic::{AtomicU64, AtomicI64, Ordering};
use std::marker::PhantomData;

/// Maximum number of L2 levels to track (top 20)
pub const MAX_L2_LEVELS: usize = 20;

/// Cache-line aligned atomic array for lock-free depth tracking
#[repr(align(64))]
pub struct DepthArray {
    data: [AtomicU64; MAX_L2_LEVELS],
}

impl DepthArray {
    pub const fn new() -> Self {
        const ZERO: AtomicU64 = AtomicU64::new(0);
        Self {
            data: [ZERO; MAX_L2_LEVELS],
        }
    }

    #[inline]
    pub fn set(&self, idx: usize, value: u64) {
        if idx < MAX_L2_LEVELS {
            self.data[idx].store(value, Ordering::Relaxed);
        }
    }

    #[inline]
    pub fn get(&self, idx: usize) -> u64 {
        if idx < MAX_L2_LEVELS {
            self.data[idx].load(Ordering::Relaxed)
        } else {
            0
        }
    }

    #[inline]
    pub fn add(&self, idx: usize, delta: u64) {
        if idx < MAX_L2_LEVELS {
            let _ = self.data[idx].fetch_add(delta, Ordering::Relaxed);
        }
    }

    #[inline]
    pub fn sub(&self, idx: usize, delta: u64) {
        if idx < MAX_L2_LEVELS {
            let _ = self.data[idx].fetch_update(
                Ordering::Relaxed,
                Ordering::Relaxed,
                |current| current.checked_sub(delta),
            );
        }
    }
}

/// Order book shape metrics computed from depth distribution
#[derive(Debug, Clone, Copy)]
pub struct ShapeMetrics {
    /// Convexity measure: positive = convex (bulging), negative = concave (thin)
    pub convexity_bid: f64,
    pub convexity_ask: f64,
    /// Weighted average depth
    pub avg_depth_bid: f64,
    pub avg_depth_ask: f64,
    /// Microprice drift prediction (-1.0 to 1.0)
    pub microprice_drift: f64,
    /// Imbalance ratio
    pub depth_ratio: f64,
}

impl Default for ShapeMetrics {
    fn default() -> Self {
        Self {
            convexity_bid: 0.0,
            convexity_ask: 0.0,
            avg_depth_bid: 0.0,
            avg_depth_ask: 0.0,
            microprice_drift: 0.0,
            depth_ratio: 1.0,
        }
    }
}

/// Lock-free order book dynamics analyzer
#[repr(align(64))]
pub struct OrderBookDynamics {
    bid_depths: DepthArray,
    ask_depths: DepthArray,
    bid_prices: [AtomicU64; MAX_L2_LEVELS],
    ask_prices: [AtomicU64; MAX_L2_LEVELS],
    tick_size: AtomicU64,
    _pad: PhantomData<[u8; 64]>,
}

impl OrderBookDynamics {
    pub const fn new(tick_size: u64) -> Self {
        const ZERO_U64: AtomicU64 = AtomicU64::new(0);
        Self {
            bid_depths: DepthArray::new(),
            ask_depths: DepthArray::new(),
            bid_prices: [ZERO_U64; MAX_L2_LEVELS],
            ask_prices: [ZERO_U64; MAX_L2_LEVELS],
            tick_size: AtomicU64::new(tick_size),
            _pad: PhantomData,
        }
    }

    /// Update bid depth at level without allocation
    #[inline]
    pub fn update_bid(&self, level: usize, price: u64, depth: u64) {
        self.bid_depths.set(level, depth);
        if level < MAX_L2_LEVELS {
            self.bid_prices[level].store(price, Ordering::Relaxed);
        }
    }

    /// Update ask depth at level without allocation
    #[inline]
    pub fn update_ask(&self, level: usize, price: u64, depth: u64) {
        self.ask_depths.set(level, depth);
        if level < MAX_L2_LEVELS {
            self.ask_prices[level].store(price, Ordering::Relaxed);
        }
    }

    /// Compute convexity using second-order finite differences
    #[inline]
    fn compute_convexity(depths: &DepthArray) -> f64 {
        let mut sum = 0.0f64;
        let mut count = 0usize;

        for i in 1..MAX_L2_LEVELS - 1 {
            let d_prev = depths.get(i - 1) as f64;
            let d_curr = depths.get(i) as f64;
            let d_next = depths.get(i + 1) as f64;

            if d_prev > 0.0 && d_next > 0.0 {
                // Second derivative approximation
                let convexity = d_next - 2.0 * d_curr + d_prev;
                sum += convexity;
                count += 1;
            }
        }

        if count > 0 {
            sum / count as f64
        } else {
            0.0
        }
    }

    /// Compute weighted average depth with exponential decay
    #[inline]
    fn compute_weighted_depth(depths: &DepthArray, decay: f64) -> f64 {
        let mut weighted_sum = 0.0f64;
        let mut weight_total = 0.0f64;

        for i in 0..MAX_L2_LEVELS {
            let depth = depths.get(i) as f64;
            let weight = (-decay * i as f64).exp();
            weighted_sum += depth * weight;
            weight_total += weight;
        }

        if weight_total > 0.0 {
            weighted_sum / weight_total
        } else {
            0.0
        }
    }

    /// Predict microprice drift from shape analysis
    #[inline]
    fn predict_microprice_drift(
        bid_convexity: f64,
        ask_convexity: f64,
        bid_depth: f64,
        ask_depth: f64,
    ) -> f64 {
        // Convex bid side suggests support => upward pressure
        // Concave ask side suggests resistance => upward pressure
        let shape_signal = (bid_convexity - ask_convexity).clamp(-1.0, 1.0);

        // Depth imbalance signal
        let total_depth = bid_depth + ask_depth;
        let depth_signal = if total_depth > 0.0 {
            (bid_depth - ask_depth) / total_depth
        } else {
            0.0
        };

        // Combine signals with weights
        0.6 * shape_signal + 0.4 * depth_signal
    }

    /// Compute all shape metrics in a single pass
    pub fn compute_metrics(&self) -> ShapeMetrics {
        let bid_convexity = Self::compute_convexity(&self.bid_depths);
        let ask_convexity = Self::compute_convexity(&self.ask_depths);
        let bid_depth = Self::compute_weighted_depth(&self.bid_depths, 0.15);
        let ask_depth = Self::compute_weighted_depth(&self.ask_depths, 0.15);

        let depth_ratio = if ask_depth > 0.0 {
            bid_depth / ask_depth
        } else {
            1.0
        };

        let microprice_drift = Self::predict_microprice_drift(
            bid_convexity,
            ask_convexity,
            bid_depth,
            ask_depth,
        );

        ShapeMetrics {
            convexity_bid: bid_convexity,
            convexity_ask: ask_convexity,
            avg_depth_bid: bid_depth,
            avg_depth_ask: ask_depth,
            microprice_drift,
            depth_ratio,
        }
    }

    /// Get best bid price
    #[inline]
    pub fn best_bid(&self) -> u64 {
        self.bid_prices[0].load(Ordering::Relaxed)
    }

    /// Get best ask price
    #[inline]
    pub fn best_ask(&self) -> u64 {
        self.ask_prices[0].load(Ordering::Relaxed)
    }

    /// Get mid price
    #[inline]
    pub fn mid_price(&self) -> f64 {
        let bid = self.best_bid() as f64;
        let ask = self.best_ask() as f64;
        if bid > 0.0 && ask > 0.0 {
            (bid + ask) / 2.0
        } else {
            0.0
        }
    }

    /// Reset all depths (for reinitialization)
    #[inline]
    pub fn reset(&self) {
        for i in 0..MAX_L2_LEVELS {
            self.bid_depths.set(i, 0);
            self.ask_depths.set(i, 0);
            self.bid_prices[i].store(0, Ordering::Relaxed);
            self.ask_prices[i].store(0, Ordering::Relaxed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dynamics_computation() {
        let dynamics = OrderBookDynamics::new(100);

        // Set up a convex bid side (bulging)
        dynamics.update_bid(0, 9900, 1000);
        dynamics.update_bid(1, 9800, 2000);
        dynamics.update_bid(2, 9700, 3000);
        dynamics.update_bid(3, 9600, 2000);
        dynamics.update_bid(4, 9500, 1000);

        // Set up a concave ask side (thin)
        dynamics.update_ask(0, 10000, 1000);
        dynamics.update_ask(1, 10100, 500);
        dynamics.update_ask(2, 10200, 300);
        dynamics.update_ask(3, 10300, 500);
        dynamics.update_ask(4, 10400, 1000);

        let metrics = dynamics.compute_metrics();

        assert!(metrics.convexity_bid > 0.0, "Bid side should be convex");
        assert!(metrics.convexity_ask < 0.0, "Ask side should be concave");
        assert!(metrics.microprice_drift > 0.0, "Drift should be upward");
    }

    #[test]
    fn test_lock_free_updates() {
        let dynamics = OrderBookDynamics::new(50);
        
        for i in 0..MAX_L2_LEVELS {
            dynamics.update_bid(i, 1000 + i as u64, 100 * (i + 1) as u64);
            dynamics.update_ask(i, 2000 + i as u64, 100 * (i + 1) as u64);
        }

        let metrics = dynamics.compute_metrics();
        assert!(metrics.avg_depth_bid > 0.0);
        assert!(metrics.avg_depth_ask > 0.0);
    }
}
