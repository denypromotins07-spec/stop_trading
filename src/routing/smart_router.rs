//! Smart Order Routing (SOR) evaluating fee tiers, queue position, and latency.
//! Routes child orders to optimal price levels or venues.

use std::sync::atomic::{AtomicF64, AtomicU64, AtomicBool, Ordering};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SorError {
    #[error("No viable route found")]
    NoRoute,
    #[error("Invalid venue configuration")]
    InvalidVenue,
    #[error("Insufficient liquidity")]
    InsufficientLiquidity,
}

/// Venue representation
#[derive(Debug, Clone)]
pub struct Venue {
    pub id: u64,
    pub name: String,
    pub maker_fee_bps: f64,
    pub taker_fee_bps: f64,
    pub latency_ms: f64,
    pub fill_probability: f64,
}

impl Venue {
    pub fn new(
        id: u64,
        name: String,
        maker_fee_bps: f64,
        taker_fee_bps: f64,
        latency_ms: f64,
        fill_prob: f64,
    ) -> Self {
        Self {
            id,
            name,
            maker_fee_bps,
            taker_fee_bps,
            latency_ms,
            fill_probability: fill_prob.min(1.0),
        }
    }

    /// Calculate effective cost for a trade
    pub fn effective_cost(&self, quantity: f64, price: f64, is_maker: bool) -> f64 {
        let notional = quantity * price;
        let fee_rate = if is_maker { self.maker_fee_bps } else { self.taker_fee_bps };
        notional * fee_rate / 10000.0
    }

    /// Score venue for routing (higher is better)
    pub fn score(&self, is_maker: bool) -> f64 {
        let fee = if is_maker { self.maker_fee_bps } else { self.taker_fee_bps };
        
        // Lower fees are better
        let fee_score = 100.0 - fee;
        
        // Lower latency is better
        let latency_score = 100.0 - self.latency_ms.min(100.0);
        
        // Higher fill probability is better
        let fill_score = self.fill_probability * 100.0;
        
        // Weighted combination
        fee_score * 0.4 + latency_score * 0.3 + fill_score * 0.3
    }
}

/// Price level in order book
#[derive(Debug, Clone)]
pub struct PriceLevel {
    pub price: f64,
    pub quantity: f64,
    pub venue_id: u64,
    pub queue_position: usize,
}

/// Route decision
#[derive(Debug, Clone)]
pub struct RouteDecision {
    pub venue_id: u64,
    pub price: f64,
    pub quantity: f64,
    pub expected_cost: f64,
    pub confidence: f64,
}

/// Smart Order Router
pub struct SmartOrderRouter {
    venues: Vec<Venue>,
    preferred_venue: AtomicU64,
    route_count: AtomicU64,
    successful_routes: AtomicU64,
}

impl SmartOrderRouter {
    pub fn new(venues: Vec<Venue>) -> Result<Self, SorError> {
        if venues.is_empty() {
            return Err(SorError::InvalidVenue);
        }
        
        Ok(Self {
            venues,
            preferred_venue: AtomicU64::new(0),
            route_count: AtomicU64::new(0),
            successful_routes: AtomicU64::new(0),
        })
    }

    /// Find best route for an order
    pub fn find_best_route(
        &self,
        quantity: f64,
        price: f64,
        side: OrderSide,
        is_maker: bool,
    ) -> Result<RouteDecision, SorError> {
        if self.venues.is_empty() {
            return Err(SorError::NoRoute);
        }

        let mut best_score = -1.0;
        let mut best_venue: Option<&Venue> = None;

        for venue in &self.venues {
            let score = venue.score(is_maker);
            if score > best_score {
                best_score = score;
                best_venue = Some(venue);
            }
        }

        let venue = best_venue.ok_or(SorError::NoRoute)?;

        let expected_cost = venue.effective_cost(quantity, price, is_maker);
        let confidence = venue.fill_probability * (best_score / 100.0);

        Ok(RouteDecision {
            venue_id: venue.id,
            price,
            quantity,
            expected_cost,
            confidence,
        })
    }

    /// Split order across multiple venues
    pub fn split_order(
        &self,
        total_quantity: f64,
        price: f64,
        side: OrderSide,
        is_maker: bool,
    ) -> Result<Vec<RouteDecision>, SorError> {
        if self.venues.is_empty() {
            return Err(SorError::NoRoute);
        }

        // Sort venues by score
        let mut scored_venues: Vec<(&Venue, f64)> = self.venues
            .iter()
            .map(|v| (v, v.score(is_maker)))
            .collect();
        
        scored_venues.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut routes = Vec::new();
        let mut remaining = total_quantity;

        for (venue, score) in scored_venues {
            if remaining <= 0.0 {
                break;
            }

            // Allocate proportional to score
            let total_score: f64 = scored_venues.iter().map(|(_, s)| s).sum();
            let allocation = (score / total_score) * total_quantity;
            let qty = allocation.min(remaining);

            let cost = venue.effective_cost(qty, price, is_maker);
            let confidence = venue.fill_probability * (score / 100.0);

            routes.push(RouteDecision {
                venue_id: venue.id,
                price,
                quantity: qty,
                expected_cost: cost,
                confidence,
            });

            remaining -= qty;
        }

        if routes.is_empty() {
            return Err(SorError::NoRoute);
        }

        Ok(routes)
    }

    /// Record successful route execution
    pub fn record_success(&self) {
        self.route_count.fetch_add(1, Ordering::Relaxed);
        self.successful_routes.fetch_add(1, Ordering::Relaxed);
    }

    /// Record failed route
    pub fn record_failure(&self) {
        self.route_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Get success rate
    pub fn get_success_rate(&self) -> f64 {
        let total = self.route_count.load(Ordering::Relaxed);
        let success = self.successful_routes.load(Ordering::Relaxed);
        
        if total == 0 {
            return 1.0;
        }
        
        success as f64 / total as f64
    }

    /// Set preferred venue
    pub fn set_preferred_venue(&self, venue_id: u64) {
        self.preferred_venue.store(venue_id, Ordering::Relaxed);
    }

    /// Get venues
    pub fn venues(&self) -> &[Venue] {
        &self.venues
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderSide {
    Buy,
    Sell,
}

/// Fee tier structure
#[derive(Debug, Clone)]
pub struct FeeTier {
    pub min_volume: f64,
    pub maker_rebate_bps: f64,
    pub taker_fee_bps: f64,
}

impl FeeTier {
    pub fn applicable_tier(tiers: &[FeeTier], volume: f64) -> &FeeTier {
        tiers
            .iter()
            .rev()
            .find(|t| volume >= t.min_volume)
            .unwrap_or(&tiers[0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_venue_scoring() {
        let venue = Venue::new(1, "Test".to_string(), 2.0, 5.0, 10.0, 0.9);
        
        let maker_score = venue.score(true);
        let taker_score = venue.score(false);
        
        assert!(maker_score > 0.0);
        assert!(maker_score > taker_score); // Maker should score higher due to lower fees
    }

    #[test]
    fn test_smart_routing() {
        let venues = vec![
            Venue::new(1, "A".to_string(), 1.0, 3.0, 5.0, 0.95),
            Venue::new(2, "B".to_string(), 2.0, 4.0, 10.0, 0.90),
            Venue::new(3, "C".to_string(), 3.0, 5.0, 15.0, 0.85),
        ];

        let sor = SmartOrderRouter::new(venues).unwrap();
        
        let route = sor.find_best_route(100.0, 50000.0, OrderSide::Buy, true).unwrap();
        
        assert_eq!(route.venue_id, 1); // Should pick venue A (best score)
        assert!(route.confidence > 0.0);
    }

    #[test]
    fn test_order_splitting() {
        let venues = vec![
            Venue::new(1, "A".to_string(), 1.0, 3.0, 5.0, 0.95),
            Venue::new(2, "B".to_string(), 2.0, 4.0, 10.0, 0.90),
        ];

        let sor = SmartOrderRouter::new(venues).unwrap();
        
        let routes = sor.split_order(1000.0, 50000.0, OrderSide::Buy, true).unwrap();
        
        let total_qty: f64 = routes.iter().map(|r| r.quantity).sum();
        assert!((total_qty - 1000.0).abs() < 0.01);
        assert_eq!(routes.len(), 2);
    }

    #[test]
    fn test_success_tracking() {
        let venues = vec![Venue::new(1, "A".to_string(), 1.0, 3.0, 5.0, 0.95)];
        let sor = SmartOrderRouter::new(venues).unwrap();
        
        assert_eq!(sor.get_success_rate(), 1.0);
        
        sor.record_success();
        sor.record_success();
        sor.record_failure();
        
        assert!((sor.get_success_rate() - 0.667).abs() < 0.01);
    }
}
