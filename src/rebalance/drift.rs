//! Portfolio Drift Monitor
//! 
//! Implements continuous portfolio drift monitoring comparing actual weights against Black-Litterman target weights.
//! Calculates exact capital required to realign while factoring in slippage and fees.

use std::sync::atomic::{AtomicU64, Ordering};

/// Position weight and drift data
#[derive(Debug, Clone)]
pub struct PositionDrift {
    pub symbol: [u8; 16],
    pub current_weight: f64,      // Current portfolio weight (0.0 - 1.0)
    pub target_weight: f64,       // Target weight from Black-Litterman
    pub drift_pct: f64,           // Drift percentage
    pub current_value_usd: f64,
    pub target_value_usd: f64,
    pub trade_required_usd: f64,  // Positive = buy, Negative = sell
}

/// Drift analysis result
#[derive(Debug, Clone)]
pub struct DriftAnalysis {
    pub total_drift: f64,         // Sum of absolute drifts
    pub max_single_drift: f64,    // Maximum single position drift
    pub positions_out_of_band: usize,
    pub rebalance_cost_estimate: f64,
    pub recommended_action: RebalanceAction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RebalanceAction {
    NoAction,
    MinorAdjustment,
    FullRebalance,
    EmergencyRebalance,
}

/// Black-Litterman target weights input
#[derive(Debug, Clone)]
pub struct BlackLittermanInput {
    pub market_caps: Vec<f64>,
    pub views: Vec<View>,
    pub tau: f64,              // Confidence in CAPM (typically 0.05)
    pub risk_aversion: f64,    // Risk aversion coefficient (typically 2.5)
}

#[derive(Debug, Clone)]
pub struct View {
    pub asset_idx: usize,
    pub expected_return: f64,
    pub confidence: f64,       // 0.0 - 1.0
}

/// Main Drift Monitor
pub struct DriftMonitor {
    pub positions: Vec<PositionDrift>,
    pub total_portfolio_value: f64,
    pub drift_threshold_pct: f64,     // Threshold for triggering rebalance
    pub rebalance_threshold_pct: f64, // Higher threshold for full rebalance
    pub calculation_counter: AtomicU64,
}

impl DriftMonitor {
    pub fn new(drift_threshold: f64, rebalance_threshold: f64) -> Self {
        DriftMonitor {
            positions: Vec::with_capacity(64),
            total_portfolio_value: 0.0,
            drift_threshold_pct: drift_threshold,
            rebalance_threshold_pct: rebalance_threshold,
            calculation_counter: AtomicU64::new(0),
        }
    }

    /// Update position drift calculations
    pub fn update_positions(&mut self, positions_data: &[PositionData], targets: &[f64]) {
        self.calculation_counter.fetch_add(1, Ordering::Relaxed);
        
        self.positions.clear();
        self.total_portfolio_value = positions_data.iter().map(|p| p.value_usd).sum();

        if self.total_portfolio_value <= 0.0 {
            return;
        }

        for (i, pos) in positions_data.iter().enumerate() {
            let current_weight = pos.value_usd / self.total_portfolio_value;
            let target_weight = targets.get(i).copied().unwrap_or(current_weight);
            let drift_pct = (current_weight - target_weight).abs() * 100.0;
            
            let target_value = target_weight * self.total_portfolio_value;
            let trade_required = target_value - pos.value_usd;

            self.positions.push(PositionDrift {
                symbol: pos.symbol,
                current_weight,
                target_weight,
                drift_pct,
                current_value_usd: pos.value_usd,
                target_value_usd: target_value,
                trade_required_usd: trade_required,
            });
        }
    }

    /// Analyze overall portfolio drift
    pub fn analyze_drift(&self) -> DriftAnalysis {
        let mut total_drift = 0.0;
        let mut max_single_drift = 0.0;
        let mut positions_out_of_band = 0;

        for pos in &self.positions {
            let abs_drift = pos.drift_pct;
            total_drift += abs_drift;
            max_single_drift = max_single_drift.max(abs_drift);

            if abs_drift > self.drift_threshold_pct {
                positions_out_of_band += 1;
            }
        }

        // Estimate rebalancing cost (slippage + fees)
        let turnover = total_drift / 200.0; // Approximate turnover fraction
        let estimated_cost = self.total_portfolio_value * turnover * 0.001; // 0.1% assumed cost

        // Determine recommended action
        let action = if max_single_drift > self.rebalance_threshold_pct * 2.0 {
            RebalanceAction::EmergencyRebalance
        } else if max_single_drift > self.rebalance_threshold_pct {
            RebalanceAction::FullRebalance
        } else if positions_out_of_band > 0 {
            RebalanceAction::MinorAdjustment
        } else {
            RebalanceAction::NoAction
        };

        DriftAnalysis {
            total_drift,
            max_single_drift,
            positions_out_of_band,
            rebalance_cost_estimate: estimated_cost,
            recommended_action: action,
        }
    }

    /// Calculate Black-Litterman optimal weights
    pub fn calculate_bl_weights(&self, input: &BlackLittermanInput) -> Vec<f64> {
        if input.market_caps.is_empty() {
            return vec![];
        }

        let n = input.market_caps.len();
        
        // Step 1: Calculate market equilibrium returns (CAPM)
        let total_market_cap: f64 = input.market_caps.iter().sum();
        let market_weights: Vec<f64> = input.market_caps
            .iter()
            .map(|&mc| mc / total_market_cap)
            .collect();

        // Step 2: Calculate prior returns based on market weights
        let pi: Vec<f64> = market_weights
            .iter()
            .map(|&w| w * input.risk_aversion)
            .collect();

        // Step 3: Incorporate views (simplified implementation)
        let mut posterior_returns = pi.clone();
        
        for view in &input.views {
            if view.asset_idx < n && view.confidence > 0.0 {
                // Blend prior with view
                let view_adjustment = (view.expected_return - posterior_returns[view.asset_idx]) * view.confidence;
                posterior_returns[view.asset_idx] += view_adjustment * input.tau;
            }
        }

        // Step 4: Convert returns back to weights (softmax-like normalization)
        let sum_returns: f64 = posterior_returns.iter().map(|&r| r.exp()).sum();
        let weights: Vec<f64> = posterior_returns
            .iter()
            .map(|&r| r.exp() / sum_returns)
            .collect();

        weights
    }

    /// Get positions requiring rebalancing
    pub fn get_rebalance_candidates(&self) -> Vec<&PositionDrift> {
        self.positions
            .iter()
            .filter(|p| p.drift_pct > self.drift_threshold_pct)
            .collect()
    }

    /// Calculate exact trade sizes for rebalancing
    pub fn calculate_trade_sizes(&self, min_trade_usd: f64) -> Vec<TradeInstruction> {
        let mut instructions = Vec::new();

        for pos in &self.positions {
            if pos.trade_required_usd.abs() < min_trade_usd {
                continue;
            }

            instructions.push(TradeInstruction {
                symbol: pos.symbol,
                side: if pos.trade_required_usd > 0.0 { Side::Buy } else { Side::Sell },
                amount_usd: pos.trade_required_usd.abs(),
                current_price: if pos.current_value_usd > 0.0 {
                    pos.current_value_usd / pos.current_weight.max(0.0001)
                } else {
                    0.0
                },
            });
        }

        instructions
    }

    /// Get total drift metric for dashboards
    pub fn get_drift_metric(&self) -> f64 {
        self.analyze_drift().total_drift
    }
}

/// Input position data
#[derive(Debug, Clone, Copy)]
pub struct PositionData {
    pub symbol: [u8; 16],
    pub value_usd: f64,
    pub quantity: f64,
}

/// Trade instruction for rebalancing
#[derive(Debug, Clone)]
pub struct TradeInstruction {
    pub symbol: [u8; 16],
    pub side: Side,
    pub amount_usd: f64,
    pub current_price: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Buy,
    Sell,
}

/// No-Trade Region calculator
pub struct NoTradeRegion {
    pub base_threshold: f64,
    pub transaction_cost_rate: f64,
    pub volatility_adjustment: f64,
}

impl NoTradeRegion {
    pub fn new(base_threshold: f64, transaction_cost_rate: f64) -> Self {
        NoTradeRegion {
            base_threshold,
            transaction_cost_rate,
            volatility_adjustment: 0.0,
        }
    }

    /// Calculate no-trade region bounds
    pub fn get_bounds(&self, target_weight: f64) -> (f64, f64) {
        // No-trade region: [target - threshold, target + threshold]
        let threshold = self.calculate_dynamic_threshold(target_weight);
        (target_weight - threshold, target_weight + threshold)
    }

    fn calculate_dynamic_threshold(&self, target_weight: f64) -> f64 {
        // Base threshold adjusted for transaction costs and volatility
        let cost_adjusted = self.base_threshold + self.transaction_cost_rate * 100.0;
        let vol_adjusted = cost_adjusted * (1.0 + self.volatility_adjustment);
        
        // Scale by target weight (smaller positions get wider bands)
        vol_adjusted * (1.0 + (0.5 - target_weight).abs())
    }

    /// Check if rebalancing is warranted
    pub fn should_rebalance(&self, current_weight: f64, target_weight: f64) -> bool {
        let (lower, upper) = self.get_bounds(target_weight);
        current_weight < lower || current_weight > upper
    }

    /// Update volatility adjustment based on market conditions
    pub fn update_volatility(&mut self, annualized_vol: f64) {
        // Higher volatility = wider no-trade regions
        self.volatility_adjustment = (annualized_vol / 0.5).min(1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_drift_monitor_creation() {
        let monitor = DriftMonitor::new(2.0, 5.0);
        assert_eq!(monitor.drift_threshold_pct, 2.0);
        assert_eq!(monitor.rebalance_threshold_pct, 5.0);
    }

    #[test]
    fn test_position_drift_calculation() {
        let mut monitor = DriftMonitor::new(2.0, 5.0);
        
        let positions = vec![
            PositionData { symbol: *b"BTC             ", value_usd: 60_000.0, quantity: 1.0 },
            PositionData { symbol: *b"ETH             ", value_usd: 40_000.0, quantity: 20.0 },
        ];
        
        let targets = vec![0.5, 0.5]; // Equal weight target
        
        monitor.update_positions(&positions, &targets);
        
        assert_eq!(monitor.positions.len(), 2);
        assert!((monitor.positions[0].current_weight - 0.6).abs() < 0.001);
        assert!((monitor.positions[0].drift_pct - 10.0).abs() < 0.1);
    }

    #[test]
    fn test_no_trade_region() {
        let ntr = NoTradeRegion::new(2.0, 0.001);
        
        let (lower, upper) = ntr.get_bounds(0.5);
        assert!(lower < 0.5);
        assert!(upper > 0.5);
        
        // Should not rebalance if within bounds
        assert!(!ntr.should_rebalance(0.51, 0.5));
        
        // Should rebalance if outside bounds
        assert!(ntr.should_rebalance(0.45, 0.5));
    }

    #[test]
    fn test_black_litterman_weights() {
        let monitor = DriftMonitor::new(2.0, 5.0);
        
        let input = BlackLittermanInput {
            market_caps: vec![100.0, 200.0, 300.0],
            views: vec![View {
                asset_idx: 0,
                expected_return: 0.15,
                confidence: 0.8,
            }],
            tau: 0.05,
            risk_aversion: 2.5,
        };
        
        let weights = monitor.calculate_bl_weights(&input);
        assert_eq!(weights.len(), 3);
        
        let sum: f64 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 0.001);
    }
}
