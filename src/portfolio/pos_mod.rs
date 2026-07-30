//! Advanced Position Module Root
//! 
//! Manages complex multi-leg portfolio states and atomic margin requirements.

pub mod delta_hedger;
pub mod gamma_scalper;

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use crate::portfolio::delta_hedger::{DeltaHedger, DeltaHedgerConfig, PositionDelta};
use crate::portfolio::gamma_scalper::{GammaScalper, GammaScalperConfig, PositionGamma};

/// Portfolio configuration
#[derive(Debug, Clone)]
pub struct PortfolioConfig {
    pub delta_hedger_config: DeltaHedgerConfig,
    pub gamma_scalper_config: GammaScalperConfig,
    pub max_total_margin: f64,
    pub margin_buffer_pct: f64,
}

impl Default for PortfolioConfig {
    fn default() -> Self {
        Self {
            delta_hedger_config: DeltaHedgerConfig::default(),
            gamma_scalper_config: GammaScalperConfig::default(),
            max_total_margin: 10_000_000.0, // $10M
            margin_buffer_pct: 0.2, // 20% buffer
        }
    }
}

/// Multi-leg position
#[derive(Debug, Clone)]
pub struct MultiLegPosition {
    pub id: String,
    pub symbol: String,
    pub legs: Vec<PositionLeg>,
    pub net_delta: f64,
    pub net_gamma: f64,
    pub net_theta: f64,
    pub net_vega: f64,
    pub total_margin_required: f64,
    pub pnl_unrealized: f64,
    pub created_ns: u64,
}

#[derive(Debug, Clone)]
pub struct PositionLeg {
    pub leg_type: LegType,
    pub side: LegSide,
    pub quantity: f64,
    pub strike: Option<f64>,
    pub expiry: Option<u64>,
    pub entry_price: f64,
    pub current_price: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LegType {
    Spot,
    Perp,
    Call,
    Put,
    Future,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LegSide {
    Long,
    Short,
}

/// Margin state
#[derive(Debug, Clone)]
pub struct MarginState {
    pub total_margin_used: f64,
    pub total_margin_available: f64,
    pub margin_by_symbol: dashmap::DashMap<String, f64>,
    pub maintenance_margin: f64,
    pub margin_call_threshold: f64,
    pub is_margin_call: bool,
    pub timestamp_ns: u64,
}

/// Portfolio module handle
pub struct PortfolioModule {
    config: PortfolioConfig,
    delta_hedger: Arc<DeltaHedger>,
    gamma_scalper: Arc<GammaScalper>,
    positions: dashmap::DashMap<String, MultiLegPosition>,
    margin_used: AtomicU64, // Fixed point * 1000
    total_pnl: AtomicU64,   // Fixed point * 1000
    halted: AtomicBool,
}

impl PortfolioModule {
    pub fn new(config: PortfolioConfig) -> Self {
        let delta_hedger = Arc::new(DeltaHedger::new(config.delta_hedger_config.clone()));
        let gamma_scalper = Arc::new(GammaScalper::new(config.gamma_scalper_config.clone()));

        Self {
            config,
            delta_hedger,
            gamma_scalper,
            positions: dashmap::DashMap::new(),
            margin_used: AtomicU64::new(0),
            total_pnl: AtomicU64::new(0),
            halted: AtomicBool::new(false),
        }
    }

    /// Get reference to delta hedger
    pub fn delta_hedger(&self) -> &Arc<DeltaHedger> {
        &self.delta_hedger
    }

    /// Get reference to gamma scalper
    pub fn gamma_scalper(&self) -> &Arc<GammaScalper> {
        &self.gamma_scalper
    }

    /// Add or update a multi-leg position
    pub fn update_position(&self, position: MultiLegPosition) -> Result<(), &'static str> {
        if self.halted.load(Ordering::Relaxed) {
            return Err("Portfolio module is halted");
        }

        // Check margin
        let current_margin = self.margin_used.load(Ordering::Relaxed) as f64 / 1000.0;
        if current_margin + position.total_margin_required > self.config.max_total_margin {
            return Err("Margin limit exceeded");
        }

        // Update delta hedger
        let delta = PositionDelta {
            symbol: position.symbol.clone(),
            spot_position: position.legs.iter()
                .filter(|l| l.leg_type == LegType::Spot)
                .map(|l| l.quantity * if l.side == LegSide::Long { 1.0 } else { -1.0 })
                .sum(),
            perp_position: position.legs.iter()
                .filter(|l| l.leg_type == LegType::Perp)
                .map(|l| l.quantity * if l.side == LegSide::Long { 1.0 } else { -1.0 })
                .sum(),
            options_delta: position.net_delta - position.legs.iter()
                .filter(|l| l.leg_type == LegType::Spot || l.leg_type == LegType::Perp)
                .map(|l| l.quantity * if l.side == LegSide::Long { 1.0 } else { -1.0 })
                .sum(),
            total_delta: position.net_delta,
            notional_value: position.total_margin_required,
            last_update_ns: timestamp_ns(),
        };
        self.delta_hedger.update_position(delta);

        // Update gamma scalper
        let gamma = PositionGamma {
            symbol: position.symbol.clone(),
            options_gamma: position.net_gamma,
            spot_position: 0.0,
            perp_position: 0.0,
            net_gamma: position.net_gamma,
            gamma_pnl_sensitivity: position.pnl_unrealized,
            last_update_ns: timestamp_ns(),
        };
        self.gamma_scalper.update_position(gamma);

        self.positions.insert(position.id.clone(), position);

        Ok(())
    }

    /// Remove a position
    pub fn remove_position(&self, position_id: &str) -> Option<MultiLegPosition> {
        if let Some((_, position)) = self.positions.remove(position_id) {
            // Note: In production, would also clean up hedger/scalper state
            Some(position)
        } else {
            None
        }
    }

    /// Get margin state
    pub fn get_margin_state(&self) -> MarginState {
        let margin_used = self.margin_used.load(Ordering::Relaxed) as f64 / 1000.0;
        let margin_available = self.config.max_total_margin - margin_used;
        let maintenance = margin_used * 0.5; // Simplified: 50% of used
        let margin_call = margin_used * (1.0 + self.config.margin_buffer_pct);

        let mut by_symbol = dashmap::DashMap::new();
        for entry in self.positions.iter() {
            let pos = entry.value();
            let current = by_symbol.entry(pos.symbol.clone()).or_insert(0.0);
            *current += pos.total_margin_required;
        }

        let is_call = margin_used > margin_call;

        MarginState {
            total_margin_used: margin_used,
            total_margin_available: margin_available,
            margin_by_symbol: by_symbol,
            maintenance_margin: maintenance,
            margin_call_threshold: margin_call,
            is_margin_call: is_call,
            timestamp_ns: timestamp_ns(),
        }
    }

    /// Get portfolio summary
    pub fn get_portfolio_summary(&self) -> PortfolioSummary {
        let mut total_delta = 0.0;
        let mut total_gamma = 0.0;
        let mut total_theta = 0.0;
        let mut total_vega = 0.0;
        let mut total_pnl = 0.0;
        let mut total_margin = 0.0;

        for entry in self.positions.iter() {
            let pos = entry.value();
            total_delta += pos.net_delta;
            total_gamma += pos.net_gamma;
            total_theta += pos.net_theta;
            total_vega += pos.net_vega;
            total_pnl += pos.pnl_unrealized;
            total_margin += pos.total_margin_required;
        }

        PortfolioSummary {
            total_delta,
            total_gamma,
            total_theta,
            total_vega,
            total_pnl,
            total_margin_used: total_margin,
            position_count: self.positions.len(),
            timestamp_ns: timestamp_ns(),
        }
    }

    /// Run portfolio rebalancing
    pub fn rebalance(&self) -> RebalanceResult {
        let mut hedge_executions = self.delta_hedger.check_and_rebalance();
        
        let opportunities = self.gamma_scalper.get_gamma_summary();
        
        RebalanceResult {
            hedges_executed: hedge_executions.len(),
            scalp_opportunities: opportunities.scalp_count,
            new_delta: self.delta_hedger.get_portfolio_summary().total_delta,
            new_gamma: opportunities.total_gamma,
            timestamp_ns: timestamp_ns(),
        }
    }

    /// Halt all operations
    pub fn halt(&self) {
        self.halted.store(true, Ordering::SeqCst);
        self.delta_hedger.halt();
        self.gamma_scalper.halt();
    }

    /// Resume operations
    pub fn resume(&self) {
        self.halted.store(false, Ordering::SeqCst);
        self.delta_hedger.resume();
        self.gamma_scalper.resume();
    }

    /// Check if halted
    pub fn is_halted(&self) -> bool {
        self.halted.load(Ordering::Relaxed)
    }

    /// Get position count
    pub fn position_count(&self) -> usize {
        self.positions.len()
    }
}

/// Portfolio summary
#[derive(Debug, Clone)]
pub struct PortfolioSummary {
    pub total_delta: f64,
    pub total_gamma: f64,
    pub total_theta: f64,
    pub total_vega: f64,
    pub total_pnl: f64,
    pub total_margin_used: f64,
    pub position_count: usize,
    pub timestamp_ns: u64,
}

/// Rebalance result
#[derive(Debug, Clone)]
pub struct RebalanceResult {
    pub hedges_executed: usize,
    pub scalp_opportunities: u64,
    pub new_delta: f64,
    pub new_gamma: f64,
    pub timestamp_ns: u64,
}

/// Get current timestamp in nanoseconds
#[inline]
fn timestamp_ns() -> u64 {
    Instant::now()
        .duration_since(Instant::now() - Duration::from_secs(1))
        .as_nanos() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_portfolio_module_basic() {
        let config = PortfolioConfig::default();
        let module = PortfolioModule::new(config);

        assert!(!module.is_halted());
        assert_eq!(module.position_count(), 0);

        // Add a position
        let position = MultiLegPosition {
            id: "pos_1".to_string(),
            symbol: "BTCUSD".to_string(),
            legs: vec![
                PositionLeg {
                    leg_type: LegType::Spot,
                    side: LegSide::Long,
                    quantity: 1.0,
                    strike: None,
                    expiry: None,
                    entry_price: 50000.0,
                    current_price: 50000.0,
                },
            ],
            net_delta: 1.0,
            net_gamma: 0.0,
            net_theta: 0.0,
            net_vega: 0.0,
            total_margin_required: 50000.0,
            pnl_unrealized: 0.0,
            created_ns: timestamp_ns(),
        };

        assert!(module.update_position(position).is_ok());
        assert_eq!(module.position_count(), 1);

        let summary = module.get_portfolio_summary();
        assert_eq!(summary.position_count, 1);
        assert_eq!(summary.total_delta, 1.0);
    }

    #[test]
    fn test_margin_check() {
        let config = PortfolioConfig::default();
        let module = PortfolioModule::new(config);

        let margin = module.get_margin_state();
        assert!(margin.total_margin_available > 0.0);
        assert!(!margin.is_margin_call);
    }
}
