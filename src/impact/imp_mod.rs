//! Market Impact Module Root
//! 
//! Feeds dynamic impact multipliers into TWAP/VWAP execution algos.

pub mod almgren;
pub mod square_root;

use almgren::{AlmgrenChrissSolver, AlmgrenChrissParams, FixedPoint, ExecutionTrajectory};
use square_root::{SquareRootImpactModel, SquareRootConfig, TradeObservation};

/// Configuration for the impact module
#[derive(Debug, Clone)]
pub struct ImpactConfig {
    pub almgren_params: AlmgrenChrissParams,
    pub square_root_config: SquareRootConfig,
    /// Blend weight between Almgren-Chriss and square-root model (0 to 1)
    pub model_blend_weight: f64,
}

impl Default for ImpactConfig {
    fn default() -> Self {
        Self {
            almgren_params: AlmgrenChrissParams {
                total_quantity: FixedPoint::from_i64(10000),
                time_horizon: FixedPoint::from_f64(3600.0),
                num_intervals: 10,
                eta: FixedPoint::from_f64(1e-6),
                sigma: FixedPoint::from_f64(0.02),
                lambda: FixedPoint::from_f64(1e-5),
                gamma: FixedPoint::from_f64(1e-7),
            },
            square_root_config: SquareRootConfig::default(),
            model_blend_weight: 0.5,
        }
    }
}

/// Combined market impact signal
#[derive(Debug, Clone, Copy)]
pub struct ImpactSignal {
    /// Predicted impact from Almgren-Chriss model (decimal)
    pub ac_impact: f64,
    /// Predicted impact from square-root model (decimal)
    pub sr_impact: f64,
    /// Blended impact prediction
    pub blended_impact: f64,
    /// Recommended execution urgency (0 to 1)
    pub urgency: f64,
    /// Dynamic multiplier for TWAP/VWAP
    pub twap_multiplier: f64,
}

impl Default for ImpactSignal {
    fn default() -> Self {
        Self {
            ac_impact: 0.0,
            sr_impact: 0.0,
            blended_impact: 0.0,
            urgency: 0.5,
            twap_multiplier: 1.0,
        }
    }
}

/// Main market impact engine combining multiple models
pub struct ImpactEngine {
    config: ImpactConfig,
    ac_solver: AlmgrenChrissSolver,
    sr_model: SquareRootImpactModel,
    current_signal: ImpactSignal,
    last_trajectory: Option<ExecutionTrajectory>,
}

impl ImpactEngine {
    pub fn new(config: ImpactConfig) -> Self {
        Self {
            ac_solver: AlmgrenChrissSolver::new(config.almgren_params.clone()),
            sr_model: SquareRootImpactModel::new(config.square_root_config.clone()),
            config,
            current_signal: ImpactSignal::default(),
            last_trajectory: None,
        }
    }

    /// Add a trade observation to the square-root model
    pub fn add_trade_observation(&mut self, obs: TradeObservation) {
        self.sr_model.add_observation(obs);
        self.update_signal();
    }

    /// Calculate optimal trajectory using Almgren-Chriss
    pub fn calculate_optimal_trajectory(&mut self, params: AlmgrenChrissParams) -> ExecutionTrajectory {
        let solver = AlmgrenChrissSolver::new(params);
        let trajectory = solver.calculate_trajectory();
        
        // Update AC impact estimate
        self.current_signal.ac_impact = trajectory.expected_impact_cost.to_f64();
        self.last_trajectory = Some(trajectory.clone());
        
        self.update_signal();
        trajectory
    }

    /// Get predicted impact for a specific order
    pub fn predict_order_impact(&self, volume: u64, is_buy: bool) -> f64 {
        let sr_impact = self.sr_model.predict_impact(volume, is_buy).abs();
        
        // AC model gives us baseline from trajectory
        let ac_impact = if let Some(ref traj) = self.last_trajectory {
            traj.expected_impact_cost.to_f64()
        } else {
            0.0
        };
        
        // Blend the two models
        let blended = ac_impact * self.config.model_blend_weight 
                    + sr_impact * (1.0 - self.config.model_blend_weight);
        
        blended
    }

    fn update_signal(&mut self) {
        let sr_alpha = self.sr_model.alpha();
        let sr_confidence = self.sr_model.confidence();
        
        // Use SR model's current alpha as base impact estimate
        self.current_signal.sr_impact = sr_alpha * 0.1; // Scale to reasonable range
        
        // Recalculate blended
        self.current_signal.blended_impact = 
            self.current_signal.ac_impact * self.config.model_blend_weight
            + self.current_signal.sr_impact * (1.0 - self.config.model_blend_weight);
        
        // Urgency based on impact level
        // Higher impact = more urgent to execute carefully
        self.current_signal.urgency = (self.current_signal.blended_impact * 10.0).clamp(0.1, 0.9);
        
        // TWAP multiplier: reduce speed when impact is high
        self.current_signal.twap_multiplier = 1.0 / (1.0 + self.current_signal.blended_impact * 5.0);
    }

    /// Get current impact signal
    pub fn get_signal(&self) -> ImpactSignal {
        self.current_signal
    }

    /// Get recommended TWAP interval adjustment
    /// Returns factor to multiply base interval by
    pub fn twap_interval_multiplier(&self) -> f64 {
        self.current_signal.twap_multiplier
    }

    /// Get recommended VWAP participation rate
    /// Returns fraction of market volume to participate
    pub fn vwap_participation_rate(&self, market_volume: u64) -> f64 {
        let base_rate = 0.1; // 10% default
        base_rate * self.current_signal.twap_multiplier
    }

    /// Get the square-root model for direct access
    pub fn square_root_model(&self) -> &SquareRootImpactModel {
        &self.sr_model
    }

    /// Get the Almgren-Chriss solver
    pub fn almgren_chriss_solver(&self) -> &AlmgrenChrissSolver {
        &self.ac_solver
    }

    /// Update model parameters dynamically
    pub fn update_almgren_params(&mut self, params: AlmgrenChrissParams) {
        self.ac_solver = AlmgrenChrissSolver::new(params);
        self.update_signal();
    }

    /// Reset all models
    pub fn reset(&mut self) {
        self.sr_model.reset();
        self.current_signal = ImpactSignal::default();
        self.last_trajectory = None;
    }
}

/// Execution quality monitor
pub struct ExecutionMonitor {
    expected_impact: f64,
    actual_slippage: Vec<f64>,
    window_size: usize,
}

impl ExecutionMonitor {
    pub fn new(window_size: usize) -> Self {
        Self {
            expected_impact: 0.0,
            actual_slippage: Vec::with_capacity(window_size),
            window_size,
        }
    }

    /// Record an executed trade's slippage
    pub fn record_execution(&mut self, expected_price: f64, actual_price: f64, is_buy: bool) {
        let slippage = if is_buy {
            (actual_price - expected_price) / expected_price
        } else {
            (expected_price - actual_price) / expected_price
        };
        
        if self.actual_slippage.len() >= self.window_size {
            self.actual_slippage.remove(0);
        }
        self.actual_slippage.push(slippage.abs());
    }

    /// Set expected impact for comparison
    pub fn set_expected_impact(&mut self, impact: f64) {
        self.expected_impact = impact;
    }

    /// Get average realized slippage
    pub fn avg_slippage(&self) -> f64 {
        if self.actual_slippage.is_empty() {
            return 0.0;
        }
        self.actual_slippage.iter().sum::<f64>() / self.actual_slippage.len() as f64
    }

    /// Check if actual slippage exceeds expectations
    pub fn is_underperforming(&self, threshold_factor: f64) -> bool {
        let avg = self.avg_slippage();
        avg > self.expected_impact * threshold_factor
    }

    /// Get slippage variance
    pub fn slippage_variance(&self) -> f64 {
        if self.actual_slippage.len() < 2 {
            return 0.0;
        }
        let avg = self.avg_slippage();
        let variance: f64 = self.actual_slippage.iter()
            .map(|s| (s - avg).powi(2))
            .sum::<f64>() / (self.actual_slippage.len() - 1) as f64;
        variance
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_impact_engine_basic() {
        let config = ImpactConfig::default();
        let mut engine = ImpactEngine::new(config);

        // Initial signal should have default values
        let signal = engine.get_signal();
        assert_eq!(signal.twap_multiplier, 1.0);

        // Add some observations
        for i in 0..15 {
            let obs = TradeObservation::new(
                10000,
                50000.0,
                50000.0 * 1.001,
                true,
                i * 1000000,
            );
            engine.add_trade_observation(obs);
        }

        let signal = engine.get_signal();
        assert!(signal.twap_multiplier <= 1.0);
    }

    #[test]
    fn test_execution_monitor() {
        let mut monitor = ExecutionMonitor::new(10);
        monitor.set_expected_impact(0.001);

        for _ in 0..5 {
            monitor.record_execution(100.0, 100.05, true);
        }

        assert!(monitor.avg_slippage() > 0.0);
    }
}
