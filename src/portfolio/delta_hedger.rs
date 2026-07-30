//! Portfolio Delta Hedger
//! 
//! Implements automated delta-neutral hedging using perpetual swaps to isolate
//! pure alpha from directional beta. Continuously calculates portfolio delta
//! and executes micro-hedges when exposure drifts beyond tolerance band.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Position delta information
#[derive(Debug, Clone)]
pub struct PositionDelta {
    pub symbol: String,
    pub spot_position: f64,
    pub perp_position: f64,
    pub options_delta: f64,
    pub total_delta: f64,
    pub notional_value: f64,
    pub last_update_ns: u64,
}

/// Hedge recommendation
#[derive(Debug, Clone)]
pub struct HedgeRecommendation {
    pub symbol: String,
    pub action: HedgeAction,
    pub size: f64,
    pub expected_delta_change: f64,
    pub urgency: HedgeUrgency,
    pub reason: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HedgeAction {
    BuyPerp,
    SellPerp,
    ReduceSpot,
    AdjustOptions,
    Hold,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HedgeUrgency {
    Low,
    Medium,
    High,
    Critical,
}

/// Portfolio-wide delta summary
#[derive(Debug, Clone)]
pub struct PortfolioDeltaSummary {
    pub total_delta: f64,
    pub total_notional: f64,
    pub net_exposure_usd: f64,
    pub delta_by_symbol: Vec<PositionDelta>,
    pub hedge_ratio: f64,
    pub target_delta: f64,
    pub current_drift: f64,
    pub timestamp_ns: u64,
}

/// Delta hedger configuration
#[derive(Debug, Clone)]
pub struct DeltaHedgerConfig {
    /// Target delta (0 for delta-neutral)
    pub target_delta: f64,
    /// Tolerance band (absolute delta units)
    pub tolerance_band: f64,
    /// Maximum hedge size per operation
    pub max_hedge_size: f64,
    /// Minimum hedge size to avoid dust trades
    pub min_hedge_size: f64,
    /// Rebalance interval (ms)
    pub rebalance_interval_ms: u64,
    /// Use dynamic sizing based on volatility
    pub dynamic_sizing: bool,
}

impl Default for DeltaHedgerConfig {
    fn default() -> Self {
        Self {
            target_delta: 0.0,
            tolerance_band: 100.0, // $100k notional
            max_hedge_size: 1000.0,
            min_hedge_size: 10.0,
            rebalance_interval_ms: 100,
            dynamic_sizing: true,
        }
    }
}

/// Delta hedger engine
pub struct DeltaHedger {
    config: DeltaHedgerConfig,
    /// Current positions by symbol
    positions: dashmap::DashMap<String, PositionDelta>,
    /// Last hedge timestamp per symbol
    last_hedge_ns: dashmap::DashMap<String, AtomicU64>,
    /// Total hedges executed
    hedge_count: AtomicU64,
    /// Total delta hedged
    total_delta_hedged: AtomicU64, // Fixed point * 1000
    /// Global halt flag
    halted: AtomicBool,
    /// Last rebalance timestamp
    last_rebalance_ns: AtomicU64,
}

impl DeltaHedger {
    pub fn new(config: DeltaHedgerConfig) -> Self {
        Self {
            config,
            positions: dashmap::DashMap::new(),
            last_hedge_ns: dashmap::DashMap::new(),
            hedge_count: AtomicU64::new(0),
            total_delta_hedged: AtomicU64::new(0),
            halted: AtomicBool::new(false),
            last_rebalance_ns: AtomicU64::new(0),
        }
    }

    /// Update position delta for a symbol
    pub fn update_position(&self, delta: PositionDelta) {
        let now_ns = timestamp_ns();
        
        self.positions.insert(delta.symbol.clone(), delta);
        
        if !self.last_hedge_ns.contains_key(&delta.symbol) {
            self.last_hedge_ns.insert(delta.symbol.clone(), AtomicU64::new(0));
        }
        
        // Update last position time
        if let Some(entry) = self.positions.get(&delta.symbol) {
            entry.value().last_update_ns = now_ns;
        }
    }

    /// Calculate portfolio-wide delta summary
    pub fn get_portfolio_summary(&self) -> PortfolioDeltaSummary {
        let mut total_delta = 0.0;
        let mut total_notional = 0.0;
        let mut deltas = Vec::new();

        for entry in self.positions.iter() {
            let pos = entry.value();
            total_delta += pos.total_delta;
            total_notional += pos.notional_value.abs();
            deltas.push(pos.clone());
        }

        let net_exposure = total_delta; // Simplified: delta = USD exposure
        let hedge_ratio = if total_notional > 0.0 {
            (total_notional - total_delta.abs()) / total_notional
        } else {
            1.0
        };

        let current_drift = (total_delta - self.config.target_delta).abs();

        PortfolioDeltaSummary {
            total_delta,
            total_notional,
            net_exposure_usd: net_exposure,
            delta_by_symbol: deltas,
            hedge_ratio,
            target_delta: self.config.target_delta,
            current_drift,
            timestamp_ns: timestamp_ns(),
        }
    }

    /// Generate hedge recommendations
    pub fn generate_hedge_recommendations(&self) -> Vec<HedgeRecommendation> {
        if self.halted.load(Ordering::Relaxed) {
            return Vec::new();
        }

        let summary = self.get_portfolio_summary();
        let mut recommendations = Vec::new();

        // Check if we're outside tolerance band
        if summary.current_drift > self.config.tolerance_band {
            let drift = summary.total_delta - self.config.target_delta;
            
            // Determine action
            let (action, sign) = if drift > 0.0 {
                (HedgeAction::SellPerp, -1.0)
            } else {
                (HedgeAction::BuyPerp, 1.0)
            };

            // Calculate hedge size
            let mut hedge_size = drift.abs();
            
            if self.config.dynamic_sizing {
                // Scale by volatility (simplified - would use actual vol in production)
                let vol_adjustment = 1.0; // Placeholder
                hedge_size *= vol_adjustment;
            }

            // Apply limits
            hedge_size = hedge_size.clamp(self.config.min_hedge_size, self.config.max_hedge_size);

            // Determine urgency
            let urgency = if summary.current_drift > self.config.tolerance_band * 3.0 {
                HedgeUrgency::Critical
            } else if summary.current_drift > self.config.tolerance_band * 2.0 {
                HedgeUrgency::High
            } else if summary.current_drift > self.config.tolerance_band * 1.5 {
                HedgeUrgency::Medium
            } else {
                HedgeUrgency::Low
            };

            recommendations.push(HedgeRecommendation {
                symbol: "PORTFOLIO".to_string(),
                action,
                size: hedge_size * sign,
                expected_delta_change: -drift,
                urgency,
                reason: format!(
                    "Delta drift ${:.2} exceeds tolerance ${:.2}",
                    summary.current_drift, self.config.tolerance_band
                ),
            });
        }

        // Per-symbol recommendations
        for pos in &summary.delta_by_symbol {
            let symbol_drift = pos.total_delta.abs();
            let symbol_tolerance = self.config.tolerance_band / 5.0; // Per-symbol is tighter

            if symbol_drift > symbol_tolerance {
                let (action, sign) = if pos.total_delta > 0.0 {
                    (HedgeAction::SellPerp, -1.0)
                } else {
                    (HedgeAction::BuyPerp, 1.0)
                };

                let hedge_size = symbol_drift.clamp(self.config.min_hedge_size, self.config.max_hedge_size / 2.0);

                recommendations.push(HedgeRecommendation {
                    symbol: pos.symbol.clone(),
                    action,
                    size: hedge_size * sign,
                    expected_delta_change: -pos.total_delta,
                    urgency: HedgeUrgency::Medium,
                    reason: format!("Symbol delta ${:.2} exceeds per-symbol tolerance", symbol_drift),
                });
            }
        }

        recommendations
    }

    /// Execute a hedge (simulation - would call actual exchange in production)
    pub fn execute_hedge(&self, recommendation: &HedgeRecommendation) -> Result<HedgeExecution, &'static str> {
        if self.halted.load(Ordering::Relaxed) {
            return Err("Hedger is halted");
        }

        // Check rate limiting
        let now_ns = timestamp_ns();
        if let Some(last_hedge) = self.last_hedge_ns.get(&recommendation.symbol) {
            let last = last_hedge.load(Ordering::Relaxed);
            let min_interval_ns = 10_000_000; // 10ms minimum between hedges
            
            if now_ns - last < min_interval_ns {
                return Err("Rate limited - too frequent hedging");
            }
        }

        // Simulate execution
        let execution = HedgeExecution {
            symbol: recommendation.symbol.clone(),
            action: recommendation.action,
            requested_size: recommendation.size,
            executed_size: recommendation.size * 0.999, // Simulate slight slippage
            execution_price: 0.0, // Would be filled by exchange
            delta_changed: recommendation.expected_delta_change * 0.999,
            timestamp_ns: now_ns,
            latency_us: 50, // Simulated latency
        };

        // Update tracking
        self.hedge_count.fetch_add(1, Ordering::Relaxed);
        let delta_fixed = (execution.delta_changed.abs() * 1000.0) as u64;
        self.total_delta_hedged.fetch_add(delta_fixed, Ordering::Relaxed);

        if let Some(last_hedge) = self.last_hedge_ns.get(&recommendation.symbol) {
            last_hedge.store(now_ns, Ordering::Relaxed);
        }

        Ok(execution)
    }

    /// Automatic rebalance check
    pub fn check_and_rebalance(&self) -> Vec<HedgeExecution> {
        let now_ns = timestamp_ns();
        let last = self.last_rebalance_ns.load(Ordering::Relaxed);
        
        if now_ns - last < (self.config.rebalance_interval_ms as u64) * 1_000_000 {
            return Vec::new();
        }

        let recommendations = self.generate_hedge_recommendations();
        let mut executions = Vec::new();

        for rec in recommendations {
            if let Ok(exec) = self.execute_hedge(&rec) {
                executions.push(exec);
            }
        }

        self.last_rebalance_ns.store(now_ns, Ordering::Relaxed);
        executions
    }

    /// Halt hedging
    pub fn halt(&self) {
        self.halted.store(true, Ordering::SeqCst);
    }

    /// Resume hedging
    pub fn resume(&self) {
        self.halted.store(false, Ordering::SeqCst);
    }

    /// Check if halted
    pub fn is_halted(&self) -> bool {
        self.halted.load(Ordering::Relaxed)
    }

    /// Get statistics
    pub fn get_stats(&self) -> DeltaHedgerStats {
        DeltaHedgerStats {
            total_hedges: self.hedge_count.load(Ordering::Relaxed),
            total_delta_hedged: self.total_delta_hedged.load(Ordering::Relaxed) as f64 / 1000.0,
            is_halted: self.halted.load(Ordering::Relaxed),
            position_count: self.positions.len(),
        }
    }
}

/// Hedge execution result
#[derive(Debug, Clone)]
pub struct HedgeExecution {
    pub symbol: String,
    pub action: HedgeAction,
    pub requested_size: f64,
    pub executed_size: f64,
    pub execution_price: f64,
    pub delta_changed: f64,
    pub timestamp_ns: u64,
    pub latency_us: u64,
}

/// Hedger statistics
#[derive(Debug, Clone)]
pub struct DeltaHedgerStats {
    pub total_hedges: u64,
    pub total_delta_hedged: f64,
    pub is_halted: bool,
    pub position_count: usize,
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
    fn test_delta_hedger_basic() {
        let config = DeltaHedgerConfig::default();
        let hedger = DeltaHedger::new(config);

        // Add a long position
        let pos = PositionDelta {
            symbol: "BTCUSD".to_string(),
            spot_position: 10.0,
            perp_position: 0.0,
            options_delta: 0.0,
            total_delta: 500000.0, // $500k long
            notional_value: 500000.0,
            last_update_ns: timestamp_ns(),
        };

        hedger.update_position(pos);

        let summary = hedger.get_portfolio_summary();
        assert_eq!(summary.total_delta, 500000.0);
        assert!(summary.current_drift > 100.0); // Exceeds tolerance

        let recommendations = hedger.generate_hedge_recommendations();
        assert!(!recommendations.is_empty());
        assert_eq!(recommendations[0].action, HedgeAction::SellPerp);
    }

    #[test]
    fn test_delta_neutral_portfolio() {
        let config = DeltaHedgerConfig::default();
        let hedger = DeltaHedger::new(config);

        // Add offsetting positions
        hedger.update_position(PositionDelta {
            symbol: "BTCUSD".to_string(),
            spot_position: 10.0,
            perp_position: -10.0,
            options_delta: 0.0,
            total_delta: 0.0,
            notional_value: 500000.0,
            last_update_ns: timestamp_ns(),
        });

        let summary = hedger.get_portfolio_summary();
        assert_eq!(summary.total_delta, 0.0);
        assert!(summary.current_drift <= config.tolerance_band);

        let recommendations = hedger.generate_hedge_recommendations();
        assert!(recommendations.is_empty()); // No hedge needed
    }
}
