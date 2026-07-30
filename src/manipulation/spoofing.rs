//! Spoofing and Layering Detection Module
//! Analyzes rapid order cancellations to detect fake liquidity.
//! Filters out spoofed orders from the order book to prevent adverse trades.

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

/// Detected spoofing pattern
#[derive(Debug, Clone, Copy)]
pub struct SpoofingSignal {
    /// Pattern type
    pub pattern_type: SpoofingPattern,
    /// Price level where spoofing detected
    pub price_level: i64,
    /// Side (true = bid spoof, false = ask spoof)
    pub is_bid: bool,
    /// Timestamp (nanoseconds)
    pub timestamp_ns: u64,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
    /// Estimated fake volume
    pub fake_volume: u64,
    /// Number of rapid cancellations
    pub cancel_count: u32,
    /// Average order lifetime in microseconds
    pub avg_lifetime_us: u64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SpoofingPattern {
    /// Large order placed then cancelled quickly
    FlashOrder,
    /// Multiple orders at consecutive levels (layering)
    Layering,
    /// Order moved progressively away from touch
    WalkingBook,
    /// Mirror orders on both sides (wash trading setup)
    MirrorOrders,
}

/// Order lifecycle record
#[derive(Debug, Clone, Copy)]
pub struct OrderLifecycle {
    pub order_id: u64,
    pub price: i64,
    pub quantity: u64,
    pub is_bid: bool,
    pub placement_time_ns: u64,
    pub cancellation_time_ns: u64,
    pub was_modified: bool,
}

/// Lock-free spoofing detector
pub struct SpoofingDetector {
    /// Signals detected count
    signals_detected: CachePadded<AtomicU64>,
    /// Orders analyzed count
    orders_analyzed: CachePadded<AtomicU64>,
    /// Active flag
    is_active: CachePadded<AtomicBool>,
    /// Lifetime threshold (microseconds) - orders shorter than this are suspicious
    lifetime_threshold_us: u64,
    /// Size threshold - orders larger than this get more scrutiny
    size_threshold: u64,
    /// Recent order lifetimes (simplified storage)
    recent_lifetimes: CachePadded<AtomicU64>,
}

impl SpoofingDetector {
    pub fn new(lifetime_threshold_us: u64, size_threshold: u64) -> Self {
        Self {
            signals_detected: CachePadded::default(),
            orders_analyzed: CachePadded::default(),
            is_active: CachePadded::new(AtomicBool::new(true)),
            lifetime_threshold_us,
            size_threshold,
            recent_lifetimes: CachePadded::default(),
        }
    }

    /// Analyze an order lifecycle for spoofing patterns
    pub fn analyze_order(&self, lifecycle: OrderLifecycle) -> Option<SpoofingSignal> {
        if !self.is_active.data.load(Ordering::Acquire) {
            return None;
        }

        self.orders_analyzed.data.fetch_add(1, Ordering::AcqRel);

        // Calculate lifetime in microseconds
        let lifetime_ns = lifecycle.cancellation_time_ns.saturating_sub(lifecycle.placement_time_ns);
        let lifetime_us = lifetime_ns / 1000;

        // Store for rolling analysis
        self.recent_lifetimes.data.store(lifetime_us, Ordering::Release);

        // Check for flash order (very short lifetime)
        if lifetime_us < self.lifetime_threshold_us {
            let size_factor = if lifecycle.quantity >= self.size_threshold {
                1.5 // Larger orders are more suspicious
            } else {
                1.0
            };

            let confidence = (1.0 - (lifetime_us as f64 / self.lifetime_threshold_us as f64))
                .min(1.0) * size_factor;

            if confidence > 0.5 {
                self.signals_detected.data.fetch_add(1, Ordering::AcqRel);

                return Some(SpoofingSignal {
                    pattern_type: SpoofingPattern::FlashOrder,
                    price_level: lifecycle.price,
                    is_bid: lifecycle.is_bid,
                    timestamp_ns: lifecycle.cancellation_time_ns,
                    confidence: confidence.min(1.0),
                    fake_volume: lifecycle.quantity,
                    cancel_count: 1,
                    avg_lifetime_us: lifetime_us,
                });
            }
        }

        None
    }

    /// Analyze a batch of order lifecycles for layering patterns
    pub fn analyze_layering(&self, lifecycles: &[OrderLifecycle]) -> Option<SpoofingSignal> {
        if !self.is_active.data.load(Ordering::Acquire) || lifecycles.len() < 3 {
            return None;
        }

        // Group by side
        let bid_orders: Vec<_> = lifecycles.iter().filter(|o| o.is_bid).collect();
        let ask_orders: Vec<_> = lifecycles.iter().filter(|o| !o.is_bid).collect();

        // Check for layering on each side
        for (orders, is_bid) in [(bid_orders, true), (ask_orders, false)].iter() {
            if orders.len() < 3 {
                continue;
            }

            // Check for consecutive price levels
            let mut consecutive = 0;
            let mut total_lifetime = 0u64;
            let mut total_volume = 0u64;

            for i in 0..orders.len() - 1 {
                let price_diff = (orders[i].price - orders[i + 1].price).abs();
                
                if price_diff <= 2 { // Consecutive or near-consecutive levels
                    consecutive += 1;
                    
                    let lifetime = orders[i].cancellation_time_ns.saturating_sub(orders[i].placement_time_ns) / 1000;
                    total_lifetime += lifetime;
                    total_volume += orders[i].quantity;
                }
            }

            // Layering detected if multiple consecutive levels with short lifetimes
            if consecutive >= 2 {
                let avg_lifetime = total_lifetime / consecutive as u64;
                
                if avg_lifetime < self.lifetime_threshold_us {
                    let confidence = (consecutive as f64 / 5.0).min(1.0) 
                        * (1.0 - (avg_lifetime as f64 / self.lifetime_threshold_us as f64));

                    if confidence > 0.5 {
                        self.signals_detected.data.fetch_add(1, Ordering::AcqRel);

                        return Some(SpoofingSignal {
                            pattern_type: SpoofingPattern::Layering,
                            price_level: orders[0].price,
                            is_bid,
                            timestamp_ns: lifecycles.last().unwrap().cancellation_time_ns,
                            confidence: confidence.min(1.0),
                            fake_volume: total_volume,
                            cancel_count: (consecutive + 1) as u32,
                            avg_lifetime_us: avg_lifetime,
                        });
                    }
                }
            }
        }

        None
    }

    /// Check if a price level has suspicious order activity
    pub fn is_level_suspicious(&self, level_volume: u64, recent_cancels: u32) -> bool {
        if !self.is_active.data.load(Ordering::Acquire) {
            return false;
        }

        // High volume with many recent cancels is suspicious
        if level_volume >= self.size_threshold && recent_cancels >= 3 {
            return true;
        }

        false
    }

    /// Get filtered "real" volume at a level (excluding likely spoofed orders)
    pub fn get_real_volume_estimate(&self, total_volume: u64, suspicious_ratio: f64) -> u64 {
        // Reduce volume based on suspicion ratio
        let real_ratio = 1.0 - suspicious_ratio.min(0.8); // Cap at 80% reduction
        (total_volume as f64 * real_ratio) as u64
    }

    /// Get statistics
    pub fn get_stats(&self) -> SpoofingStats {
        SpoofingStats {
            signals_detected: self.signals_detected.data.load(Ordering::Acquire),
            orders_analyzed: self.orders_analyzed.data.load(Ordering::Acquire),
            detection_rate: {
                let analyzed = self.orders_analyzed.data.load(Ordering::Acquire);
                let detected = self.signals_detected.data.load(Ordering::Acquire);
                if analyzed > 0 {
                    detected as f64 / analyzed as f64
                } else {
                    0.0
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
        self.orders_analyzed.data.store(0, Ordering::Release);
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SpoofingStats {
    pub signals_detected: u64,
    pub orders_analyzed: u64,
    pub detection_rate: f64,
    pub is_active: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flash_order_detection() {
        let detector = SpoofingDetector::new(1000, 1000); // 1ms threshold, 1000 size

        let lifecycle = OrderLifecycle {
            order_id: 1,
            price: 10000,
            quantity: 5000, // Large order
            is_bid: true,
            placement_time_ns: 1000000,
            cancellation_time_ns: 1000500, // 500us lifetime - very fast!
            was_modified: false,
        };

        let signal = detector.analyze_order(lifecycle);
        assert!(signal.is_some());
        
        let signal = signal.unwrap();
        assert_eq!(signal.pattern_type, SpoofingPattern::FlashOrder);
        assert!(signal.confidence > 0.5);
    }

    #[test]
    fn test_layering_detection() {
        let detector = SpoofingDetector::new(1000, 500);

        // Create layered orders at consecutive prices
        let lifecycles = vec![
            OrderLifecycle {
                order_id: 1,
                price: 10000,
                quantity: 1000,
                is_bid: true,
                placement_time_ns: 1000000,
                cancellation_time_ns: 1000500,
                was_modified: false,
            },
            OrderLifecycle {
                order_id: 2,
                price: 9999,
                quantity: 1000,
                is_bid: true,
                placement_time_ns: 1000100,
                cancellation_time_ns: 1000600,
                was_modified: false,
            },
            OrderLifecycle {
                order_id: 3,
                price: 9998,
                quantity: 1000,
                is_bid: true,
                placement_time_ns: 1000200,
                cancellation_time_ns: 1000700,
                was_modified: false,
            },
        ];

        let signal = detector.analyze_layering(&lifecycles);
        assert!(signal.is_some());
        assert_eq!(signal.unwrap().pattern_type, SpoofingPattern::Layering);
    }

    #[test]
    fn test_legitimate_order() {
        let detector = SpoofingDetector::new(1000, 1000);

        let lifecycle = OrderLifecycle {
            order_id: 1,
            price: 10000,
            quantity: 100, // Small order
            is_bid: true,
            placement_time_ns: 1000000,
            cancellation_time_ns: 2000000, // 1 second lifetime
            was_modified: false,
        };

        let signal = detector.analyze_order(lifecycle);
        assert!(signal.is_none()); // Should not trigger
    }
}
