//! Smart Order Routing Module Root
//! 
//! Integrates venue scoring and sweep logic into the global execution gateway.

pub mod venue_scoring;
pub mod sweep_logic;

pub use venue_scoring::{
    VenueMetrics,
    VenueScore,
    VenueScoringEngine,
    VenueWeights,
};

pub use sweep_logic::{
    Side,
    OrderBookLevel,
    LiquiditySnapshot,
    SweepResult,
    SweepConfig,
    SweepEngine,
};

/// Combined SOR state for order routing decisions
#[derive(Debug, Clone)]
pub struct SorDecision {
    pub selected_venue_id: u32,
    pub order_size: f64,
    pub expected_slippage_bps: f64,
    pub child_orders: Vec<f64>,
    pub triggers_cascade: bool,
    pub confidence: f64,
}

/// Smart Order Router combining all components
pub struct SmartOrderRouter {
    venue_engine: VenueScoringEngine,
    sweep_engine: SweepEngine,
}

impl SmartOrderRouter {
    pub fn new(venue_weights: VenueWeights, sweep_config: SweepConfig) -> Self {
        Self {
            venue_engine: VenueScoringEngine::new(venue_weights),
            sweep_engine: SweepEngine::new(sweep_config),
        }
    }
    
    /// Update venue metrics
    pub fn update_venue(&mut self, metrics: VenueMetrics) {
        self.venue_engine.update_venue(metrics);
    }
    
    /// Update order book snapshot for a venue
    pub fn update_order_book(&mut self, snapshot: LiquiditySnapshot) {
        self.sweep_engine.update_snapshot(snapshot);
    }
    
    /// Get optimal routing decision for an order
    pub fn route_order(&mut self, side: Side, total_volume: f64) -> Option<SorDecision> {
        // Recalculate venue scores
        let is_maker = false; // Default to taker for routing
        self.venue_engine.recalculate_scores(is_maker);
        
        // Select best venue
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        
        let venue_id = self.venue_engine.select_venue(seed)?;
        
        // Calculate sweep for the order
        let sweep_result = self.sweep_engine.calculate_sweep(side, total_volume)?;
        
        // Calculate child orders
        let num_children = (total_volume / 100.0).ceil() as usize;
        let child_orders = self.sweep_engine.calculate_child_orders(side, total_volume, num_children.max(1));
        
        let confidence = sweep_result.slippage_bps.min(100.0) / 100.0;
        
        Some(SorDecision {
            selected_venue_id: venue_id,
            order_size: sweep_result.total_volume,
            expected_slippage_bps: sweep_result.slippage_bps,
            child_orders,
            triggers_cascade: sweep_result.triggers_cascade,
            confidence: 1.0 - confidence,
        })
    }
    
    /// Get venue scoring engine reference
    pub fn venue_engine(&self) -> &VenueScoringEngine {
        &self.venue_engine
    }
    
    /// Get sweep engine reference  
    pub fn sweep_engine(&self) -> &SweepEngine {
        &self.sweep_engine
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;
    
    #[test]
    fn test_smart_order_router() {
        let mut router = SmartOrderRouter::new(
            VenueWeights::default(),
            SweepConfig::default(),
        );
        
        // Add a venue
        let mut venue = VenueMetrics::new(0, "TestVenue");
        venue.avg_latency_us = 100;
        venue.bid_depth = 100000.0;
        venue.ask_depth = 100000.0;
        venue.maker_fee_bps = -5.0;
        venue.fill_rate = 0.95;
        router.update_venue(venue);
        
        // Add order book snapshot
        let mut snapshot = LiquiditySnapshot::default();
        snapshot.ask_count = 5;
        snapshot.bid_count = 5;
        
        for i in 0..5 {
            snapshot.asks[i] = OrderBookLevel {
                price: 100.0 + i as f64 * 0.1,
                quantity: 1000.0,
                order_count: 5,
            };
            snapshot.bids[i] = OrderBookLevel {
                price: 99.9 - i as f64 * 0.1,
                quantity: 1000.0,
                order_count: 5,
            };
        }
        
        router.update_order_book(snapshot);
        
        // Route an order
        let decision = router.route_order(Side::Buy, 500.0);
        assert!(decision.is_some());
        
        let decision = decision.unwrap();
        assert_eq!(decision.selected_venue_id, 0);
        assert!(decision.confidence > 0.0);
    }
}
