//! Portfolio Construction Module Root
//! 
//! Integrates HRP and Risk Parity weights into the global execution router.
//! Provides unified interface for portfolio construction strategies.

use alloc::vec::Vec;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::portfolio::hrp::{HierarchicalRiskParity, CovarianceMatrix, HRPArena};
use crate::portfolio::risk_parity::{RiskParityOptimizer, RiskParityResult, AdaptiveRiskParity};

/// Maximum number of assets in construction pipeline
pub const MAX_CONSTRUCTION_ASSETS: usize = 512;

/// Portfolio allocation strategy selector
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AllocationStrategy {
    /// Hierarchical Risk Parity
    HRP,
    /// Classic Risk Parity
    RiskParity,
    /// Equal weight
    EqualWeight,
    /// Inverse volatility
    InverseVolatility,
    /// Hybrid HRP + Risk Parity
    Hybrid,
}

/// Portfolio weights result
#[derive(Debug, Clone)]
pub struct PortfolioWeights {
    pub asset_ids: Vec<u64>,
    pub weights: Vec<f64>,
    pub strategy: AllocationStrategy,
    pub timestamp_ns: u64,
    pub total_risk: f64,
    pub expected_return: f64,
}

impl PortfolioWeights {
    pub fn new(asset_ids: Vec<u64>, weights: Vec<f64>, strategy: AllocationStrategy) -> Self {
        let timestamp_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        
        PortfolioWeights {
            asset_ids,
            weights,
            strategy,
            timestamp_ns,
            total_risk: 0.0,
            expected_return: 0.0,
        }
    }

    pub fn validate(&self) -> Result<(), ConstructionError> {
        if self.asset_ids.len() != self.weights.len() {
            return Err(ConstructionError::MismatchedLengths);
        }

        let sum: f64 = self.weights.iter().sum();
        if (sum - 1.0).abs() > 1e-6 {
            return Err(ConstructionError::WeightsNotNormalized(sum));
        }

        for (i, &w) in self.weights.iter().enumerate() {
            if w < 0.0 || w > 1.0 {
                return Err(ConstructionError::InvalidWeight(i, w));
            }
        }

        Ok(())
    }

    pub fn with_risk(mut self, risk: f64) -> Self {
        self.total_risk = risk;
        self
    }

    pub fn with_expected_return(mut self, ret: f64) -> Self {
        self.expected_return = ret;
        self
    }
}

/// Error types for portfolio construction
#[derive(Debug, Clone, PartialEq)]
pub enum ConstructionError {
    MismatchedLengths,
    WeightsNotNormalized(f64),
    InvalidWeight(usize, f64),
    InsufficientData,
    OptimizationFailed,
    MemoryLimitExceeded,
    CovarianceInvalid,
}

/// Main portfolio constructor integrating multiple strategies
pub struct PortfolioConstructor {
    strategy: AllocationStrategy,
    arena: HRPArena,
    last_rebalance_ts: AtomicU64,
    rebalance_interval_ns: AtomicU64,
    is_active: AtomicBool,
    max_assets: usize,
}

impl PortfolioConstructor {
    /// Create new portfolio constructor with specified strategy
    pub fn new(strategy: AllocationStrategy, arena_size_mb: usize) -> Self {
        PortfolioConstructor {
            strategy,
            arena: HRPArena::new(arena_size_mb),
            last_rebalance_ts: AtomicU64::new(0),
            rebalance_interval_ns: AtomicU64::new(60_000_000_000), // 1 minute default
            is_active: AtomicBool::new(true),
            max_assets: MAX_CONSTRUCTION_ASSETS,
        }
    }

    /// Set rebalance interval in nanoseconds
    pub fn set_rebalance_interval_ns(&mut self, interval_ns: u64) {
        self.rebalance_interval_ns.store(interval_ns, Ordering::Release);
    }

    /// Check if rebalance is due
    pub fn should_rebalance(&self, current_ts: u64) -> bool {
        if !self.is_active.load(Ordering::Acquire) {
            return false;
        }

        let last_ts = self.last_rebalance_ts.load(Ordering::Acquire);
        let interval = self.rebalance_interval_ns.load(Ordering::Acquire);

        if last_ts == 0 {
            return true;
        }

        current_ts.saturating_sub(last_ts) >= interval
    }

    /// Construct portfolio weights based on selected strategy
    pub fn construct(&mut self, covariance: &[[f64]], asset_ids: &[u64]) -> Result<PortfolioWeights, ConstructionError> {
        let n = covariance.len();
        if n == 0 {
            return Err(ConstructionError::InsufficientData);
        }

        if n > self.max_assets {
            return Err(ConstructionError::MemoryLimitExceeded);
        }

        let weights = match self.strategy {
            AllocationStrategy::HRP => self.construct_hrp(covariance)?,
            AllocationStrategy::RiskParity => self.construct_risk_parity(covariance)?,
            AllocationStrategy::EqualWeight => self.construct_equal_weight(n),
            AllocationStrategy::InverseVolatility => self.construct_inverse_vol(covariance)?,
            AllocationStrategy::Hybrid => self.construct_hybrid(covariance)?,
        };

        let result = PortfolioWeights::new(asset_ids.to_vec(), weights, self.strategy);
        
        self.last_rebalance_ts.store(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64,
            Ordering::Release,
        );

        Ok(result)
    }

    /// Construct using HRP
    fn construct_hrp(&mut self, covariance: &[[f64]]) -> Result<Vec<f64>, ConstructionError> {
        self.arena.reset();
        
        let cov_matrix = CovarianceMatrix::from_slice(covariance);
        let mut hrp = HierarchicalRiskParity::new(cov_matrix);
        
        hrp.allocate_validated()
            .map_err(|_| ConstructionError::OptimizationFailed)
    }

    /// Construct using Risk Parity
    fn construct_risk_parity(&mut self, covariance: &[[f64]]) -> Result<Vec<f64>, ConstructionError> {
        let optimizer = RiskParityOptimizer::new(covariance, None);
        let result = optimizer.optimize();
        
        if !result.converged {
            return Err(ConstructionError::OptimizationFailed);
        }

        Ok(result.weights)
    }

    /// Equal weight construction
    fn construct_equal_weight(&self, n: usize) -> Vec<f64> {
        let weight = 1.0 / n as f64;
        vec![weight; n]
    }

    /// Inverse volatility construction
    fn construct_inverse_vol(&self, covariance: &[[f64]]) -> Result<Vec<f64>, ConstructionError> {
        let n = covariance.len();
        let mut inv_vols = Vec::with_capacity(n);

        for i in 0..n {
            let var = covariance[i][i];
            if var <= 0.0 {
                return Err(ConstructionError::CovarianceInvalid);
            }
            inv_vols.push(1.0 / var.sqrt());
        }

        let sum: f64 = inv_vols.iter().sum();
        Ok(inv_vols.iter().map(|&v| v / sum).collect())
    }

    /// Hybrid HRP + Risk Parity construction
    fn construct_hybrid(&mut self, covariance: &[[f64]]) -> Result<Vec<f64>, ConstructionError> {
        // Get HRP weights
        let hrp_weights = self.construct_hrp(covariance)?;
        
        // Get Risk Parity weights
        let rp_weights = self.construct_risk_parity(covariance)?;

        // Blend weights (60% HRP, 40% RP by default)
        let blend_factor = 0.6;
        let mut blended = Vec::with_capacity(hrp_weights.len());

        for (h, r) in hrp_weights.iter().zip(rp_weights.iter()) {
            blended.push(blend_factor * h + (1.0 - blend_factor) * r);
        }

        // Normalize
        let sum: f64 = blended.iter().sum();
        Ok(blended.iter().map(|&w| w / sum).collect())
    }

    /// Activate/deactivate constructor
    pub fn set_active(&self, active: bool) {
        self.is_active.store(active, Ordering::Release);
    }

    pub fn is_active(&self) -> bool {
        self.is_active.load(Ordering::Acquire)
    }

    /// Get current strategy
    pub fn strategy(&self) -> AllocationStrategy {
        self.strategy
    }

    /// Update strategy
    pub fn update_strategy(&mut self, strategy: AllocationStrategy) {
        self.strategy = strategy;
    }

    /// Get arena memory usage
    pub fn arena_usage_bytes(&self) -> usize {
        self.arena.used_bytes()
    }
}

/// Execution router integration for portfolio weights
pub struct ExecutionRouterIntegration {
    constructor: PortfolioConstructor,
    routing_enabled: AtomicBool,
    pending_weights: core::cell::RefCell<Option<PortfolioWeights>>,
}

impl ExecutionRouterIntegration {
    pub fn new(strategy: AllocationStrategy, arena_size_mb: usize) -> Self {
        ExecutionRouterIntegration {
            constructor: PortfolioConstructor::new(strategy, arena_size_mb),
            routing_enabled: AtomicBool::new(true),
            pending_weights: core::cell::RefCell::new(None),
        }
    }

    /// Process new weights and route to execution engine
    pub fn process_and_route(&self, covariance: &[[f64]], asset_ids: &[u64]) -> Result<(), ConstructionError> {
        if !self.routing_enabled.load(Ordering::Acquire) {
            return Ok(());
        }

        let mut guard = self.constructor.clone();
        let weights = guard.construct(covariance, asset_ids)?;
        weights.validate()?;

        *self.pending_weights.borrow_mut() = Some(weights);
        
        // In production, this would trigger IPC to execution router
        // self.route_to_execution(&weights)?;

        Ok(())
    }

    /// Get pending weights for execution
    pub fn get_pending_weights(&self) -> Option<PortfolioWeights> {
        self.pending_weights.borrow().clone()
    }

    /// Clear pending weights after execution
    pub fn clear_pending(&self) {
        *self.pending_weights.borrow_mut() = None;
    }

    /// Enable/disable routing
    pub fn set_routing_enabled(&self, enabled: bool) {
        self.routing_enabled.store(enabled, Ordering::Release);
    }
}

/// Builder pattern for PortfolioConstructor
pub struct PortfolioConstructorBuilder {
    strategy: AllocationStrategy,
    arena_size_mb: usize,
    rebalance_interval_ns: u64,
    max_assets: usize,
}

impl Default for PortfolioConstructorBuilder {
    fn default() -> Self {
        PortfolioConstructorBuilder {
            strategy: AllocationStrategy::HRP,
            arena_size_mb: 64,
            rebalance_interval_ns: 60_000_000_000,
            max_assets: MAX_CONSTRUCTION_ASSETS,
        }
    }
}

impl PortfolioConstructorBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn strategy(mut self, strategy: AllocationStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    pub fn arena_size_mb(mut self, size: usize) -> Self {
        self.arena_size_mb = size;
        self
    }

    pub fn rebalance_interval_ns(mut self, interval: u64) -> Self {
        self.rebalance_interval_ns = interval;
        self
    }

    pub fn max_assets(mut self, max: usize) -> Self {
        self.max_assets = max;
        self
    }

    pub fn build(self) -> PortfolioConstructor {
        let mut constructor = PortfolioConstructor::new(self.strategy, self.arena_size_mb);
        constructor.set_rebalance_interval_ns(self.rebalance_interval_ns);
        constructor.max_assets = self.max_assets;
        constructor
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_equal_weight_construction() {
        let mut constructor = PortfolioConstructor::new(AllocationStrategy::EqualWeight, 32);
        let covariance = vec![
            vec![0.04, 0.01],
            vec![0.01, 0.09],
        ];
        let asset_ids = vec![1, 2];

        let result = constructor.construct(&covariance, &asset_ids).unwrap();
        
        assert_eq!(result.weights.len(), 2);
        assert!((result.weights[0] - 0.5).abs() < 1e-6);
        assert!((result.weights[1] - 0.5).abs() < 1e-6);
        assert!(result.validate().is_ok());
    }

    #[test]
    fn test_hrp_construction() {
        let mut constructor = PortfolioConstructor::new(AllocationStrategy::HRP, 64);
        let covariance = vec![
            vec![0.04, 0.01, 0.02],
            vec![0.01, 0.09, 0.03],
            vec![0.02, 0.03, 0.16],
        ];
        let asset_ids = vec![1, 2, 3];

        let result = constructor.construct(&covariance, &asset_ids).unwrap();
        
        assert_eq!(result.weights.len(), 3);
        let sum: f64 = result.weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(result.validate().is_ok());
    }

    #[test]
    fn test_risk_parity_construction() {
        let mut constructor = PortfolioConstructor::new(AllocationStrategy::RiskParity, 64);
        let covariance = vec![
            vec![0.04, 0.01],
            vec![0.01, 0.09],
        ];
        let asset_ids = vec![1, 2];

        let result = constructor.construct(&covariance, &asset_ids).unwrap();
        
        assert_eq!(result.weights.len(), 2);
        assert!(result.validate().is_ok());
    }

    #[test]
    fn test_builder_pattern() {
        let constructor = PortfolioConstructorBuilder::new()
            .strategy(AllocationStrategy::Hybrid)
            .arena_size_mb(128)
            .rebalance_interval_ns(30_000_000_000)
            .max_assets(100)
            .build();

        assert_eq!(constructor.strategy(), AllocationStrategy::Hybrid);
        assert!(constructor.arena_usage_bytes() >= 0);
    }

    #[test]
    fn test_rebalance_trigger() {
        let mut constructor = PortfolioConstructor::new(AllocationStrategy::EqualWeight, 32);
        constructor.set_rebalance_interval_ns(1_000_000_000); // 1 second

        assert!(constructor.should_rebalance(0)); // First call should trigger
        
        // Simulate last rebalance at time 0
        constructor.last_rebalance_ts.store(0, Ordering::Release);
        
        assert!(!constructor.should_rebalance(500_000_000)); // 0.5s later - no
        assert!(constructor.should_rebalance(1_000_000_000)); // 1s later - yes
    }
}
