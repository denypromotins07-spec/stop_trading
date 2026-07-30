//! Latency Arbitrage Engine
//! 
//! Predictive latency-arbitrage using PTP-adjusted timestamps.
//! Detects when slower venues haven't repriced to leader venue moves.

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::time::Instant;
use dashmap::DashMap;

/// Maximum symbols tracked
const MAX_SYMBOLS: usize = 256;

/// Venue latency profile
#[derive(Debug, Clone)]
pub struct VenueLatency {
    /// Average latency in microseconds
    pub avg_latency_us: u32,
    /// Standard deviation of latency
    pub stddev_us: u32,
    /// Last observed latency
    pub last_latency_us: u32,
    /// Is venue considered "slow" relative to leader
    pub is_slow: bool,
}

/// Price update with PTP timestamp
#[derive(Debug, Clone)]
pub struct TimestampedPrice {
    /// Symbol
    pub symbol: String,
    /// Price
    pub price: f64,
    /// Size
    pub size: f64,
    /// PTP-adjusted timestamp (nanoseconds)
    pub ptp_timestamp_ns: u64,
    /// Received timestamp (nanoseconds)
    pub received_timestamp_ns: u64,
    /// Venue identifier
    pub venue_id: u8,
}

/// Latency arbitrage opportunity
#[derive(Debug, Clone)]
pub struct LatencyArbOpportunity {
    /// Symbol
    pub symbol: String,
    /// Leader venue
    pub leader_venue: u8,
    /// Laggard venue
    pub laggard_venue: u8,
    /// Leader price
    pub leader_price: f64,
    /// Laggard price (stale)
    pub laggard_price: f64,
    /// Expected convergence direction
    pub direction: TradeDirection,
    /// Estimated latency advantage in microseconds
    pub latency_advantage_us: u32,
    /// Confidence score
    pub confidence: f64,
    /// Timestamp
    pub timestamp_ns: u64,
}

/// Trade direction
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TradeDirection {
    Buy,
    Sell,
}

/// Lock-free latency arb engine
pub struct LatencyArbEngine {
    /// Latest prices per venue per symbol
    prices: DashMap<(u8, String), TimestampedPrice>,
    /// Venue latency profiles
    venue_latencies: DashMap<u8, VenueLatency>,
    /// Leader venue (fastest)
    leader_venue: AtomicU64,
    /// Latency threshold for arb detection (microseconds)
    latency_threshold_us: u32,
    /// Price change threshold (basis points)
    price_change_threshold_bps: u16,
    /// Opportunities detected
    opportunities_detected: AtomicU64,
    /// Is engine active
    is_active: AtomicBool,
}

impl LatencyArbEngine {
    pub fn new(latency_threshold_us: u32, price_threshold_bps: u16) -> Self {
        Self {
            prices: DashMap::new(),
            venue_latencies: DashMap::new(),
            leader_venue: AtomicU64::new(0),
            latency_threshold_us,
            price_change_threshold_bps: price_threshold_bps,
            opportunities_detected: AtomicU64::new(0),
            is_active: AtomicBool::new(true),
        }
    }

    /// Register a venue with its expected latency
    pub fn register_venue(&self, venue_id: u8, expected_latency_us: u32) {
        let latency = VenueLatency {
            avg_latency_us: expected_latency_us,
            stddev_us: (expected_latency_us / 10).max(1),
            last_latency_us: expected_latency_us,
            is_slow: false,
        };
        self.venue_latencies.insert(venue_id, latency);

        // Update leader if this is faster
        let current_leader = self.leader_venue.load(Ordering::Relaxed);
        if let Some(current_latency) = self.venue_latencies.get(&(current_leader as u8)) {
            if expected_latency_us < current_latency.avg_latency_us {
                self.leader_venue.store(venue_id as u64, Ordering::Relaxed);
            }
        } else {
            self.leader_venue.store(venue_id as u64, Ordering::Relaxed);
        }
    }

    /// Process a price update
    pub fn process_price(&self, update: TimestampedPrice) -> Vec<LatencyArbOpportunity> {
        if !self.is_active.load(Ordering::Relaxed) {
            return Vec::new();
        }

        let symbol = update.symbol.clone();
        let venue_id = update.venue_id;
        let now_ns = Instant::now().duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default().as_nanos() as u64;

        // Calculate actual latency
        let latency_us = ((now_ns - update.received_timestamp_ns) / 1000) as u32;

        // Update venue latency profile
        if let Some(mut profile) = self.venue_latencies.get_mut(&venue_id) {
            // Exponential moving average for latency
            let alpha = 0.1;
            profile.avg_latency_us = ((profile.avg_latency_us as f64 * (1.0 - alpha)) 
                + (latency_us as f64 * alpha)) as u32;
            profile.last_latency_us = latency_us;
            
            // Mark as slow if significantly behind leader
            let leader = self.leader_venue.load(Ordering::Relaxed) as u8;
            if let Some(leader_profile) = self.venue_latencies.get(&leader) {
                profile.is_slow = latency_us > leader_profile.avg_latency_us + self.latency_threshold_us;
            }
        }

        // Check for previous price to detect move
        let key = (venue_id, symbol.clone());
        let mut opportunities = Vec::new();

        if let Some(prev) = self.prices.get(&key) {
            let price_change_bps = ((update.price - prev.price) / prev.price * 10000.0).abs() as u16;
            
            if price_change_bps >= self.price_change_threshold_bps {
                // Significant price move detected
                // Check if other venues are stale
                opportunities = self.find_stale_venues(&symbol, &update);
            }
        }

        // Store latest price
        self.prices.insert(key, update);

        opportunities
    }

    fn find_stale_venues(
        &self,
        symbol: &str,
        leader_update: &TimestampedPrice,
    ) -> Vec<LatencyArbOpportunity> {
        let mut opportunities = Vec::new();
        let timestamp_ns = Instant::now().duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default().as_nanos() as u64;

        let leader_venue = self.leader_venue.load(Ordering::Relaxed) as u8;
        let leader_latency = self.venue_latencies
            .get(&leader_venue)
            .map(|l| l.avg_latency_us)
            .unwrap_or(0);

        for entry in self.prices.iter() {
            let ((venue_id, sym), prev_price) = entry.pair();
            
            if sym != symbol || *venue_id == leader_venue {
                continue;
            }

            // Get venue latency
            let venue_latency = self.venue_latencies
                .get(venue_id)
                .map(|l| l.avg_latency_us)
                .unwrap_or(u32::MAX);

            // Check if venue is significantly slower
            let latency_diff = venue_latency.saturating_sub(leader_latency);
            
            if latency_diff >= self.latency_threshold_us {
                // This venue is likely stale
                let price_diff_bps = ((leader_update.price - prev_price.price) / prev_price.price * 10000.0).abs();
                
                if price_diff_bps >= self.price_change_threshold_bps as f64 {
                    let direction = if leader_update.price > prev_price.price {
                        TradeDirection::Buy
                    } else {
                        TradeDirection::Sell
                    };

                    opportunities.push(LatencyArbOpportunity {
                        symbol: symbol.to_string(),
                        leader_venue,
                        laggard_venue: *venue_id,
                        leader_price: leader_update.price,
                        laggard_price: prev_price.price,
                        direction,
                        latency_advantage_us: latency_diff,
                        confidence: self.calculate_confidence(latency_diff, price_diff_bps),
                        timestamp_ns,
                    });

                    self.opportunities_detected.fetch_add(1, Ordering::Relaxed);
                }
            }
        }

        opportunities
    }

    fn calculate_confidence(&self, latency_diff_us: u32, price_diff_bps: f64) -> f64 {
        // Higher confidence with larger latency advantage and price discrepancy
        let latency_factor = (latency_diff_us as f64 / self.latency_threshold_us as f64).min(2.0) / 2.0;
        let price_factor = (price_diff_bps / self.price_change_threshold_bps as f64).min(2.0) / 2.0;
        
        (latency_factor * 0.6 + price_factor * 0.4).min(1.0)
    }

    /// Get opportunities count
    pub fn get_opportunity_count(&self) -> u64 {
        self.opportunities_detected.load(Ordering::Relaxed)
    }

    /// Set latency threshold
    pub fn set_latency_threshold(&mut self, threshold_us: u32) {
        self.latency_threshold_us = threshold_us;
    }

    /// Set price change threshold
    pub fn set_price_threshold(&mut self, bps: u16) {
        self.price_change_threshold_bps = bps;
    }

    /// Deactivate engine
    pub fn deactivate(&self) {
        self.is_active.store(false, Ordering::Relaxed);
    }

    /// Activate engine
    pub fn activate(&self) {
        self.is_active.store(true, Ordering::Relaxed);
    }

    /// Get current leader venue
    pub fn get_leader_venue(&self) -> u8 {
        self.leader_venue.load(Ordering::Relaxed) as u8
    }

    /// Get venue latency profile
    pub fn get_venue_latency(&self, venue_id: u8) -> Option<VenueLatency> {
        self.venue_latencies.get(&venue_id).map(|l| l.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_latency_arb_detection() {
        let engine = LatencyArbEngine::new(1000, 10); // 1ms threshold, 10bps price change

        // Register venues with different latencies
        engine.register_venue(0, 100); // Fast leader
        engine.register_venue(1, 5000); // Slow laggard

        // Initial prices (same on both venues)
        let initial = TimestampedPrice {
            symbol: "BTCUSDT".to_string(),
            price: 50000.0,
            size: 1.0,
            ptp_timestamp_ns: 1000000000,
            received_timestamp_ns: 1000000000,
            venue_id: 0,
        };
        engine.process_price(initial.clone());

        let initial_laggard = TimestampedPrice {
            symbol: "BTCUSDT".to_string(),
            price: 50000.0,
            size: 1.0,
            ptp_timestamp_ns: 1000000000,
            received_timestamp_ns: 1000000000,
            venue_id: 1,
        };
        engine.process_price(initial_laggard);

        // Leader price moves significantly (+1%)
        let leader_move = TimestampedPrice {
            symbol: "BTCUSDT".to_string(),
            price: 50500.0, // +1% move
            size: 1.0,
            ptp_timestamp_ns: 2000000000,
            received_timestamp_ns: 2000000000,
            venue_id: 0,
        };

        let opps = engine.process_price(leader_move);
        
        assert!(!opps.is_empty(), "Should detect latency arb opportunity");
        
        if let Some(opp) = opps.first() {
            println!("Latency arb: Leader={}, Laggard={}", opp.leader_venue, opp.laggard_venue);
            println!("Confidence: {:.2}", opp.confidence);
            assert_eq!(opp.direction, TradeDirection::Buy);
        }
    }

    #[test]
    fn test_no_arb_small_move() {
        let engine = LatencyArbEngine::new(1000, 50); // 50bps threshold

        engine.register_venue(0, 100);
        engine.register_venue(1, 5000);

        // Small price move (below threshold)
        let initial = TimestampedPrice {
            symbol: "ETHUSDT".to_string(),
            price: 3000.0,
            size: 10.0,
            ptp_timestamp_ns: 1000000000,
            received_timestamp_ns: 1000000000,
            venue_id: 0,
        };
        engine.process_price(initial);

        let small_move = TimestampedPrice {
            symbol: "ETHUSDT".to_string(),
            price: 3010.0, // Only ~0.33% move
            size: 10.0,
            ptp_timestamp_ns: 2000000000,
            received_timestamp_ns: 2000000000,
            venue_id: 0,
        };

        let opps = engine.process_price(small_move);
        assert!(opps.is_empty(), "Should not detect arb for small moves");
    }
}
