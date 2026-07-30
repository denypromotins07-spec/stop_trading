//! Derivatives Arbitrage Module Root
//! 
//! Integrates basis calculations with atomic margin checker.

pub mod funding_arb;
pub mod calendar_spread;

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::time::Instant;
use crossbeam_channel::{bounded, Sender, Receiver};
use dashmap::DashMap;

/// Maximum concurrent derivative positions
const MAX_DERIV_POSITIONS: usize = 64;

/// Derivative arb opportunity types
#[derive(Debug, Clone)]
pub enum DerivArbOpportunity {
    /// Funding rate arbitrage
    Funding(funding_arb::BasisTradeOpportunity),
    /// Calendar spread
    Calendar(calendar_spread::CalendarSpreadOpportunity),
}

/// Position status
#[derive(Debug, Clone, PartialEq)]
pub enum DerivPositionStatus {
    Pending,
    Open,
    PartialClose { remaining: f64 },
    Closed,
    Liquidated,
}

/// Derivative position
#[derive(Debug)]
pub struct DerivPosition {
    /// Unique ID
    pub id: u64,
    /// Opportunity type
    pub opp_type: &'static str,
    /// Underlying symbol
    pub underlying: String,
    /// Leg 1 (e.g., spot)
    pub leg1_size: f64,
    /// Leg 2 (e.g., perp)
    pub leg2_size: f64,
    /// Entry price leg 1
    pub leg1_price: f64,
    /// Entry price leg 2
    pub leg2_price: f64,
    /// Expected annualized return
    pub expected_return: f64,
    /// Current P&L
    pub current_pnl: f64,
    /// Status
    pub status: DerivPositionStatus,
    /// Created timestamp
    pub created_ns: u64,
    /// Margin required
    pub margin_required: f64,
    /// Is delta-neutral
    pub is_delta_neutral: bool,
}

impl DerivPosition {
    pub fn new(
        id: u64,
        opp_type: &'static str,
        underlying: String,
        leg1_size: f64,
        leg2_size: f64,
        leg1_price: f64,
        leg2_price: f64,
        expected_return: f64,
        margin_required: f64,
    ) -> Self {
        let now_ns = Instant::now().duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default().as_nanos() as u64;

        // Check if approximately delta-neutral
        let delta = (leg1_size * leg1_price - leg2_size * leg2_price).abs();
        let notional = (leg1_size * leg1_price + leg2_size * leg2_price) / 2.0;
        let is_delta_neutral = notional > 0.0 && delta / notional < 0.05;

        Self {
            id,
            opp_type,
            underlying,
            leg1_size,
            leg2_size,
            leg1_price,
            leg2_price,
            expected_return,
            current_pnl: 0.0,
            status: DerivPositionStatus::Pending,
            created_ns: now_ns,
            margin_required,
            is_delta_neutral,
        }
    }

    /// Update P&L based on current prices
    pub fn update_pnl(&mut self, leg1_current: f64, leg2_current: f64) {
        // Simplified P&L calculation
        let leg1_pnl = (leg1_current - self.leg1_price) * self.leg1_size;
        let leg2_pnl = (leg2_current - self.leg2_price) * self.leg2_size;
        self.current_pnl = leg1_pnl + leg2_pnl;
    }

    /// Get net delta
    pub fn get_net_delta(&self) -> f64 {
        (self.leg1_size * self.leg1_price) - (self.leg2_size * self.leg2_price)
    }
}

/// Margin check result
#[derive(Debug, Clone)]
pub struct MarginCheckResult {
    /// Approved
    pub approved: bool,
    /// Available margin
    pub available_margin: f64,
    /// Required margin
    pub required_margin: f64,
    /// Utilization percentage
    pub utilization_pct: f64,
}

/// Derivatives Arb Engine
pub struct DerivativesArbEngine {
    /// Active positions
    positions: DashMap<u64, DerivPosition>,
    /// Position counter
    position_counter: AtomicU64,
    /// Total margin used
    total_margin_used: AtomicU64, // Scaled by 1e6
    /// Max margin limit
    max_margin: f64,
    /// Opportunities processed
    opportunities_processed: AtomicU64,
    /// Is engine active
    is_active: AtomicBool,
    /// Event channel
    event_tx: Sender<DerivEvent>,
    event_rx: Receiver<DerivEvent>,
}

/// Deriv events
#[derive(Debug, Clone)]
pub enum DerivEvent {
    /// New opportunity
    Opportunity(DerivArbOpportunity),
    /// Position opened
    PositionOpened(u64),
    /// Margin warning
    MarginWarning { used: f64, limit: f64 },
    /// Delta drift alert
    DeltaDrift { position_id: u64, drift: f64 },
}

impl DerivativesArbEngine {
    pub fn new(max_margin: f64, buffer_size: usize) -> Self {
        let (tx, rx) = bounded(buffer_size);

        Self {
            positions: DashMap::new(),
            position_counter: AtomicU64::new(0),
            total_margin_used: AtomicU64::new(0),
            max_margin,
            opportunities_processed: AtomicU64::new(0),
            is_active: AtomicBool::new(true),
            event_tx: tx,
            event_rx: rx,
        }
    }

    /// Process funding arb opportunity
    pub fn process_funding_arb(
        &self,
        opp: funding_arb::BasisTradeOpportunity,
        size: f64,
    ) -> Option<u64> {
        if !self.is_active.load(Ordering::Relaxed) {
            return None;
        }

        // Check margin
        let margin_check = self.check_margin(size * opp.spot_price * 0.1); // 10% margin assumption
        if !margin_check.approved {
            return None;
        }

        let id = self.position_counter.fetch_add(1, Ordering::Relaxed);

        let (leg1_size, leg2_size, leg1_price, leg2_price) = match opp.direction {
            funding_arb::TradeDirection::LongSpot_ShortPerp => {
                (size, size, opp.spot_price, opp.perp_price)
            }
            funding_arb::TradeDirection::ShortSpot_LongPerp => {
                (-size, -size, opp.spot_price, opp.perp_price)
            }
        };

        let margin_required = (leg1_size.abs() * leg1_price + leg2_size.abs() * leg2_price) * 0.1;

        let position = DerivPosition::new(
            id,
            "funding_arb",
            opp.symbol,
            leg1_size,
            leg2_size,
            leg1_price,
            leg2_price,
            opp.annualized_yield,
            margin_required,
        );

        self.positions.insert(id, position);
        
        // Update margin used
        let margin_scaled = (margin_required * 1e6) as u64;
        self.total_margin_used.fetch_add(margin_scaled, Ordering::Relaxed);
        self.opportunities_processed.fetch_add(1, Ordering::Relaxed);

        let _ = self.event_tx.send(DerivEvent::Opportunity(DerivArbOpportunity::Funding(opp)));
        let _ = self.event_tx.send(DerivEvent::PositionOpened(id));

        Some(id)
    }

    /// Process calendar spread opportunity
    pub fn process_calendar_spread(
        &self,
        opp: calendar_spread::CalendarSpreadOpportunity,
        size: f64,
    ) -> Option<u64> {
        if !self.is_active.load(Ordering::Relaxed) {
            return None;
        }

        let margin_check = self.check_margin(size * opp.near_price * 0.05); // 5% spread margin
        if !margin_check.approved {
            return None;
        }

        let id = self.position_counter.fetch_add(1, Ordering::Relaxed);

        let (leg1_size, leg2_size, leg1_price, leg2_price) = match opp.direction {
            calendar_spread::SpreadDirection::LongNear_ShortFar => {
                (size, -size, opp.near_price, opp.far_price)
            }
            calendar_spread::SpreadDirection::ShortNear_LongFar => {
                (-size, size, opp.near_price, opp.far_price)
            }
        };

        let margin_required = (leg1_size.abs() * leg1_price + leg2_size.abs() * leg2_price) * 0.05;

        let position = DerivPosition::new(
            id,
            "calendar_spread",
            opp.underlying,
            leg1_size,
            leg2_size,
            leg1_price,
            leg2_price,
            opp.annualized_yield,
            margin_required,
        );

        self.positions.insert(id, position);
        
        let margin_scaled = (margin_required * 1e6) as u64;
        self.total_margin_used.fetch_add(margin_scaled, Ordering::Relaxed);
        self.opportunities_processed.fetch_add(1, Ordering::Relaxed);

        let _ = self.event_tx.send(DerivEvent::Opportunity(DerivArbOpportunity::Calendar(opp)));
        let _ = self.event_tx.send(DerivEvent::PositionOpened(id));

        Some(id)
    }

    /// Check margin availability
    fn check_margin(&self, required: f64) -> MarginCheckResult {
        let used = self.total_margin_used.load(Ordering::Relaxed) as f64 / 1e6;
        let available = self.max_margin - used;
        let utilization = if self.max_margin > 0.0 {
            used / self.max_margin * 100.0
        } else {
            100.0
        };

        let approved = available >= required && utilization < 90.0;

        if approved && utilization > 70.0 {
            let _ = self.event_tx.send(DerivEvent::MarginWarning {
                used,
                limit: self.max_margin,
            });
        }

        MarginCheckResult {
            approved,
            available_margin: available,
            required_margin: required,
            utilization_pct: utilization,
        }
    }

    /// Update position prices and check delta drift
    pub fn update_position_prices(&self, position_id: u64, leg1_price: f64, leg2_price: f64) {
        if let Some(mut pos) = self.positions.get_mut(&position_id) {
            pos.update_pnl(leg1_price, leg2_price);
            
            // Check delta drift
            let net_delta = pos.get_net_delta();
            let notional = (pos.leg1_size.abs() * leg1_price + pos.leg2_size.abs() * leg2_price) / 2.0;
            
            if notional > 0.0 {
                let drift = net_delta.abs() / notional;
                if drift > 0.10 {
                    let _ = self.event_tx.send(DerivEvent::DeltaDrift {
                        position_id,
                        drift,
                    });
                }
            }
        }
    }

    /// Close position
    pub fn close_position(&self, position_id: u64) {
        if let Some(mut pos) = self.positions.get_mut(&position_id) {
            let margin_released = pos.margin_required;
            pos.status = DerivPositionStatus::Closed;
            
            // Release margin
            let margin_scaled = (margin_released * 1e6) as u64;
            self.total_margin_used.fetch_sub(margin_scaled.min(self.total_margin_used.load(Ordering::Relaxed)), Ordering::Relaxed);
        }
    }

    /// Get event receiver
    pub fn get_event_receiver(&self) -> Receiver<DerivEvent> {
        self.event_rx.clone()
    }

    /// Get total margin used
    pub fn get_margin_used(&self) -> f64 {
        self.total_margin_used.load(Ordering::Relaxed) as f64 / 1e6
    }

    /// Get available margin
    pub fn get_available_margin(&self) -> f64 {
        self.max_margin - self.get_margin_used()
    }

    /// Get opportunities count
    pub fn get_opportunity_count(&self) -> u64 {
        self.opportunities_processed.load(Ordering::Relaxed)
    }

    /// Deactivate engine
    pub fn deactivate(&self) {
        self.is_active.store(false, Ordering::Relaxed);
    }

    /// Activate engine
    pub fn activate(&self) {
        self.is_active.store(true, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_derivatives_engine_initialization() {
        let engine = DerivativesArbEngine::new(1_000_000.0, 1000);
        
        assert!(engine.is_active.load(Ordering::Relaxed));
        assert_eq!(engine.get_margin_used(), 0.0);
    }

    #[test]
    fn test_funding_arb_processing() {
        let engine = DerivativesArbEngine::new(1_000_000.0, 1000);

        let opp = funding_arb::BasisTradeOpportunity {
            symbol: "BTCUSDT".to_string(),
            spot_price: 50000.0,
            perp_price: 50200.0,
            basis_bps: 40.0,
            funding_rate: 0.001,
            annualized_yield: 0.15,
            direction: funding_arb::TradeDirection::LongSpot_ShortPerp,
            expected_daily_return_bps: 4.0,
            timestamp_ns: 1000000000,
        };

        let position_id = engine.process_funding_arb(opp, 1.0);
        assert!(position_id.is_some());

        assert!(engine.get_margin_used() > 0.0);
    }

    #[test]
    fn test_margin_limit() {
        let engine = DerivativesArbEngine::new(10000.0, 1000); // Small margin limit

        let opp = funding_arb::BasisTradeOpportunity {
            symbol: "BTCUSDT".to_string(),
            spot_price: 50000.0,
            perp_price: 50200.0,
            basis_bps: 40.0,
            funding_rate: 0.001,
            annualized_yield: 0.15,
            direction: funding_arb::TradeDirection::LongSpot_ShortPerp,
            expected_daily_return_bps: 4.0,
            timestamp_ns: 1000000000,
        };

        // Large size should be rejected due to margin
        let result = engine.process_funding_arb(opp, 100.0);
        assert!(result.is_none(), "Should reject due to margin limit");
    }
}
