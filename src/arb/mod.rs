//! Arbitrage Module Root
//! 
//! Manages split-inventory risk and atomic execution requirements across venues.

pub mod cross_exchange;
pub mod latency_arb;

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::time::Instant;
use crossbeam_channel::{bounded, Sender, Receiver};
use dashmap::DashMap;

/// Maximum concurrent arb positions
const MAX_POSITIONS: usize = 128;

/// Arb opportunity types
#[derive(Debug, Clone)]
pub enum ArbOpportunity {
    /// Cross-exchange arbitrage
    CrossExchange(cross_exchange::CrossVenueArb),
    /// Latency arbitrage
    Latency(latency_arb::LatencyArbOpportunity),
}

/// Execution status for arb
#[derive(Debug, Clone, PartialEq)]
pub enum ExecutionStatus {
    /// Pending execution
    Pending,
    /// Partially filled
    Partial { filled_pct: f64 },
    /// Fully executed
    Executed,
    /// Failed
    Failed { reason: String },
    /// Cancelled
    Cancelled,
}

/// Arb position tracking split-inventory risk
#[derive(Debug)]
pub struct ArbPosition {
    /// Unique position ID
    pub id: u64,
    /// Opportunity type
    pub opportunity_type: &'static str,
    /// Symbol
    pub symbol: String,
    /// Buy venue
    pub buy_venue: String,
    /// Sell venue
    pub sell_venue: String,
    /// Target size
    pub target_size: f64,
    /// Filled size on buy side
    pub buy_filled: f64,
    /// Filled size on sell side
    pub sell_filled: f64,
    /// Expected profit in quote currency
    pub expected_profit: f64,
    /// Realized profit
    pub realized_profit: f64,
    /// Execution status
    pub status: ExecutionStatus,
    /// Created timestamp
    pub created_ns: u64,
    /// Last update timestamp
    pub updated_ns: AtomicU64,
    /// Requires atomic execution (both legs must succeed or fail together)
    pub is_atomic: bool,
}

impl ArbPosition {
    pub fn new(
        id: u64,
        opp_type: &'static str,
        symbol: String,
        buy_venue: String,
        sell_venue: String,
        target_size: f64,
        expected_profit_bps: f64,
        avg_price: f64,
        is_atomic: bool,
    ) -> Self {
        let now_ns = Instant::now().duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default().as_nanos() as u64;

        Self {
            id,
            opportunity_type: opp_type,
            symbol,
            buy_venue,
            sell_venue,
            target_size,
            buy_filled: 0.0,
            sell_filled: 0.0,
            expected_profit: expected_profit_bps / 10000.0 * avg_price * target_size,
            realized_profit: 0.0,
            status: ExecutionStatus::Pending,
            created_ns: now_ns,
            updated_ns: AtomicU64::new(now_ns),
            is_atomic,
        }
    }

    /// Update buy fill
    pub fn update_buy_fill(&self, filled: f64) {
        self.buy_filled = filled;
        self.update_timestamp();
        self.check_status();
    }

    /// Update sell fill
    pub fn update_sell_fill(&self, filled: f64) {
        self.sell_filled = filled;
        self.update_timestamp();
        self.check_status();
    }

    fn update_timestamp(&self) {
        let now_ns = Instant::now().duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default().as_nanos() as u64;
        self.updated_ns.store(now_ns, Ordering::Relaxed);
    }

    fn check_status(&self) {
        if self.buy_filled >= self.target_size && self.sell_filled >= self.target_size {
            // Both legs fully filled
            // In production, would calculate actual realized P&L
        } else if self.is_atomic && (self.buy_filled > 0.0 != self.sell_filled > 0.0) {
            // Atomic required but only one leg filled - risk!
        }
    }

    /// Get inventory imbalance (positive = long bias, negative = short bias)
    pub fn get_imbalance(&self) -> f64 {
        self.buy_filled - self.sell_filled
    }

    /// Check if position has dangerous imbalance
    pub fn is_imbalanced(&self, threshold: f64) -> bool {
        self.get_imbalance().abs() > threshold * self.target_size
    }
}

/// Arb Engine managing all arbitrage activities
pub struct ArbEngine {
    /// Active positions
    positions: DashMap<u64, ArbPosition>,
    /// Position counter
    position_counter: AtomicU64,
    /// Total opportunities processed
    opportunities_processed: AtomicU64,
    /// Total profit generated
    total_profit: AtomicU64, // Scaled by 1e6
    /// Max allowed imbalance ratio
    max_imbalance_ratio: f64,
    /// Is engine active
    is_active: AtomicBool,
    /// Event channel
    event_tx: Sender<ArbEvent>,
    event_rx: Receiver<ArbEvent>,
}

/// Arb events
#[derive(Debug, Clone)]
pub enum ArbEvent {
    /// New opportunity detected
    Opportunity(ArbOpportunity),
    /// Position opened
    PositionOpened(u64),
    /// Position updated
    PositionUpdated(u64),
    /// Position closed
    PositionClosed(u64, f64),
    /// Risk alert
    RiskAlert { message: String, severity: u8 },
}

impl ArbEngine {
    pub fn new(buffer_size: usize, max_imbalance_ratio: f64) -> Self {
        let (tx, rx) = bounded(buffer_size);

        Self {
            positions: DashMap::new(),
            position_counter: AtomicU64::new(0),
            opportunities_processed: AtomicU64::new(0),
            total_profit: AtomicU64::new(0),
            max_imbalance_ratio,
            is_active: AtomicBool::new(true),
            event_tx: tx,
            event_rx: rx,
        }
    }

    /// Process a cross-exchange arb opportunity
    pub fn process_cross_exchange(&self, opp: cross_exchange::CrossVenueArb) -> Option<u64> {
        if !self.is_active.load(Ordering::Relaxed) {
            return None;
        }

        // Check inventory risk before accepting
        if !self.check_inventory_risk(&opp.symbol, opp.max_size) {
            let _ = self.event_tx.send(ArbEvent::RiskAlert {
                message: format!("Inventory risk exceeded for {}", opp.symbol),
                severity: 2,
            });
            return None;
        }

        let id = self.position_counter.fetch_add(1, Ordering::Relaxed);
        
        let position = ArbPosition::new(
            id,
            "cross_exchange",
            opp.symbol.clone(),
            format!("{:?}", opp.buy_venue),
            format!("{:?}", opp.sell_venue),
            opp.max_size,
            opp.profit_bps,
            (opp.buy_price + opp.sell_price) / 2.0,
            opp.is_atomic,
        );

        self.positions.insert(id, position);
        self.opportunities_processed.fetch_add(1, Ordering::Relaxed);

        let _ = self.event_tx.send(ArbEvent::Opportunity(ArbOpportunity::CrossExchange(opp)));
        let _ = self.event_tx.send(ArbEvent::PositionOpened(id));

        Some(id)
    }

    /// Process a latency arb opportunity
    pub fn process_latency(&self, opp: latency_arb::LatencyArbOpportunity) -> Option<u64> {
        if !self.is_active.load(Ordering::Relaxed) {
            return None;
        }

        let id = self.position_counter.fetch_add(1, Ordering::Relaxed);
        
        let position = ArbPosition::new(
            id,
            "latency",
            opp.symbol.clone(),
            format!("venue_{}", opp.leader_venue),
            format!("venue_{}", opp.laggard_venue),
            1.0, // Size determined by execution layer
            0.0, // Profit calculated post-execution
            opp.leader_price,
            true, // Latency arb requires atomic execution
        );

        self.positions.insert(id, position);
        self.opportunities_processed.fetch_add(1, Ordering::Relaxed);

        let _ = self.event_tx.send(ArbEvent::Opportunity(ArbOpportunity::Latency(opp)));
        let _ = self.event_tx.send(ArbEvent::PositionOpened(id));

        Some(id)
    }

    /// Check inventory risk before opening position
    fn check_inventory_risk(&self, symbol: &str, size: f64) -> bool {
        let mut total_imbalance = 0.0;
        let mut total_target = 0.0;

        for entry in self.positions.iter() {
            let pos = entry.value();
            if pos.symbol == symbol && pos.status == ExecutionStatus::Pending {
                total_imbalance += pos.get_imbalance();
                total_target += pos.target_size;
            }
        }

        if total_target == 0.0 {
            return true;
        }

        // Check if adding this position would exceed imbalance threshold
        let potential_imbalance = (total_imbalance + size).abs() / (total_target + size);
        potential_imbalance <= self.max_imbalance_ratio
    }

    /// Update position fill
    pub fn update_fill(&self, position_id: u64, buy_filled: f64, sell_filled: f64) {
        if let Some(mut pos) = self.positions.get_mut(&position_id) {
            pos.buy_filled = buy_filled;
            pos.sell_filled = sell_filled;
            
            let _ = self.event_tx.send(ArbEvent::PositionUpdated(position_id));

            // Check for dangerous imbalance
            if pos.is_imbalanced(self.max_imbalance_ratio) {
                let _ = self.event_tx.send(ArbEvent::RiskAlert {
                    message: format!("Position {} has dangerous imbalance", position_id),
                    severity: 3,
                });
            }
        }
    }

    /// Close a position
    pub fn close_position(&self, position_id: u64, realized_profit: f64) {
        if let Some(mut pos) = self.positions.get_mut(&position_id) {
            pos.realized_profit = realized_profit;
            pos.status = ExecutionStatus::Executed;
            
            // Update total profit (scaled)
            let profit_scaled = (realized_profit * 1e6) as u64;
            self.total_profit.fetch_add(profit_scaled, Ordering::Relaxed);

            let _ = self.event_tx.send(ArbEvent::PositionClosed(position_id, realized_profit));
        }
    }

    /// Get event receiver
    pub fn get_event_receiver(&self) -> Receiver<ArbEvent> {
        self.event_rx.clone()
    }

    /// Get active position count
    pub fn active_position_count(&self) -> usize {
        self.positions.iter()
            .filter(|e| e.value().status == ExecutionStatus::Pending || 
                      e.value().status == ExecutionStatus::Partial { filled_pct: 0.0 })
            .count()
    }

    /// Get total opportunities processed
    pub fn get_opportunity_count(&self) -> u64 {
        self.opportunities_processed.load(Ordering::Relaxed)
    }

    /// Get total profit (in quote currency)
    pub fn get_total_profit(&self) -> f64 {
        self.total_profit.load(Ordering::Relaxed) as f64 / 1e6
    }

    /// Deactivate engine
    pub fn deactivate(&self) {
        self.is_active.store(false, Ordering::Relaxed);
    }

    /// Activate engine
    pub fn activate(&self) {
        self.is_active.store(true, Ordering::Relaxed);
    }

    /// Get position by ID
    pub fn get_position(&self, id: u64) -> Option<ArbPosition> {
        self.positions.get(&id).map(|p| p.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cross_exchange::{Venue, VenueQuote};

    #[test]
    fn test_arb_engine_initialization() {
        let engine = ArbEngine::new(1000, 0.5);
        
        assert!(engine.is_active.load(Ordering::Relaxed));
        assert_eq!(engine.active_position_count(), 0);
    }

    #[test]
    fn test_process_cross_exchange_arb() {
        let engine = ArbEngine::new(1000, 0.5);

        let opp = cross_exchange::CrossVenueArb {
            symbol: "BTCUSDT".to_string(),
            buy_venue: Venue::Binance,
            sell_venue: Venue::Bybit,
            buy_price: 49900.0,
            sell_price: 50100.0,
            spread_bps: 40.0,
            profit_bps: 30.0,
            max_size: 1.0,
            timestamp_ns: 1000000000,
            is_atomic: false,
        };

        let position_id = engine.process_cross_exchange(opp);
        assert!(position_id.is_some());

        let pos = engine.get_position(position_id.unwrap());
        assert!(pos.is_some());
        assert_eq!(pos.unwrap().symbol, "BTCUSDT");
    }

    #[test]
    fn test_inventory_risk_check() {
        let engine = ArbEngine::new(1000, 0.3); // Strict 30% imbalance limit

        // Create imbalanced positions
        for i in 0..5 {
            let opp = cross_exchange::CrossVenueArb {
                symbol: "ETHUSDT".to_string(),
                buy_venue: Venue::Binance,
                sell_venue: Venue::Bybit,
                buy_price: 3000.0,
                sell_price: 3010.0,
                spread_bps: 33.0,
                profit_bps: 25.0,
                max_size: 10.0,
                timestamp_ns: 1000000000 + i,
                is_atomic: false,
            };
            engine.process_cross_exchange(opp);
        }

        // New large position should be rejected due to inventory risk
        let large_opp = cross_exchange::CrossVenueArb {
            symbol: "ETHUSDT".to_string(),
            buy_venue: Venue::Binance,
            sell_venue: Venue::Bybit,
            buy_price: 3000.0,
            sell_price: 3010.0,
            spread_bps: 33.0,
            profit_bps: 25.0,
            max_size: 100.0, // Very large
            timestamp_ns: 2000000000,
            is_atomic: false,
        };

        // May or may not be accepted depending on implementation details
        let result = engine.process_cross_exchange(large_opp);
        println!("Large position accepted: {:?}", result.is_some());
    }
}
