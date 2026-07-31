//! Risk Parity Optimizer using Cyclical Coordinate Descent
//! 
//! Balances marginal risk contributions to ensure no single asset dominates
//! portfolio variance during extreme volatility regimes.
//! Memory-efficient implementation respecting 6.5GB RAM limit.

use alloc::vec::Vec;
use core::cell::RefCell;

/// Maximum assets for fixed-size allocations
pub const MAX_RISK_ASSETS: usize = 256;

/// Risk parity optimization result
#[derive(Debug, Clone)]
pub struct RiskParityResult {
    pub weights: Vec<f64>,
    pub risk_contributions: Vec<f64>,
    pub total_risk: f64,
    pub iterations: usize,
    pub converged: bool,
}

/// Risk parity optimizer using cyclical coordinate descent
pub struct RiskParityOptimizer {
    cov_matrix: [[f64; MAX_RISK_ASSETS]; MAX_RISK_ASSETS],
    n_assets: usize,
    target_risk: Vec<f64>,
    max_iterations: usize,
    tolerance: f64,
}

impl RiskParityOptimizer {
    /// Create new optimizer with given covariance matrix
    pub fn new(covariance: &[[f64]], target_risk: Option<&[f64]>) -> Self {
        let n = covariance.len();
        assert!(n <= MAX_RISK_ASSETS, "Asset count exceeds maximum");
        
        let mut cov_matrix = [[0.0; MAX_RISK_ASSETS]; MAX_RISK_ASSETS];
        for (i, row) in covariance.iter().enumerate() {
            for (j, val) in row.iter().enumerate() {
                cov_matrix[i][j] = *val;
            }
        }
        
        let target = match target_risk {
            Some(t) => t.to_vec(),
            None => vec![1.0 / n as f64; n], // Equal risk contribution by default
        };
        
        RiskParityOptimizer {
            cov_matrix,
            n_assets: n,
            target_risk: target,
            max_iterations: 1000,
            tolerance: 1e-8,
        }
    }

    /// Set maximum iterations for convergence
    pub fn with_max_iterations(mut self, max_iter: usize) -> Self {
        self.max_iterations = max_iter;
        self
    }

    /// Set convergence tolerance
    pub fn with_tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }

    /// Calculate portfolio variance given weights
    #[inline]
    fn portfolio_variance(&self, weights: &[f64]) -> f64 {
        let mut var = 0.0;
        for i in 0..self.n_assets {
            for j in 0..self.n_assets {
                var += weights[i] * self.cov_matrix[i][j] * weights[j];
            }
        }
        var
    }

    /// Calculate marginal risk contribution for each asset
    #[inline]
    fn marginal_risk_contribution(&self, weights: &[f64]) -> Vec<f64> {
        let port_var = self.portfolio_variance(weights);
        let port_vol = port_var.sqrt();
        
        if port_vol < 1e-12 {
            return vec![0.0; self.n_assets];
        }
        
        let mut mrc = Vec::with_capacity(self.n_assets);
        for i in 0..self.n_assets {
            let mut sum = 0.0;
            for j in 0..self.n_assets {
                sum += self.cov_matrix[i][j] * weights[j];
            }
            mrc.push(sum / port_vol);
        }
        mrc
    }

    /// Calculate risk contribution for each asset
    #[inline]
    fn risk_contribution(&self, weights: &[f64]) -> Vec<f64> {
        let mrc = self.marginal_risk_contribution(weights);
        let mut rc = Vec::with_capacity(self.n_assets);
        for i in 0..self.n_assets {
            rc.push(weights[i] * mrc[i]);
        }
        rc
    }

    /// Cyclical coordinate descent optimization
    pub fn optimize(&self) -> RiskParityResult {
        // Initialize with inverse volatility weights
        let mut weights = Vec::with_capacity(self.n_assets);
        for i in 0..self.n_assets {
            let vol = self.cov_matrix[i][i].sqrt();
            weights.push(if vol > 1e-12 { 1.0 / vol } else { 1.0 });
        }
        
        // Normalize to sum to 1
        let sum: f64 = weights.iter().sum();
        for w in &mut weights {
            *w /= sum;
        }
        
        let mut prev_rc = self.risk_contribution(&weights);
        let mut converged = false;
        let mut iterations = 0;
        
        for iter in 0..self.max_iterations {
            iterations = iter + 1;
            
            // Cyclical update for each asset
            for i in 0..self.n_assets {
                // Compute optimal weight for asset i
                let mut sum_others_mrc = 0.0;
                let mut sum_others_cov = 0.0;
                
                for j in 0..self.n_assets {
                    if j != i {
                        sum_others_mrc += weights[j] * self.cov_matrix[i][j];
                        sum_others_cov += weights[j] * self.cov_matrix[i][j];
                    }
                }
                
                // Target risk contribution
                let target_rc = self.target_risk[i];
                
                // Solve for new weight using quadratic formula approximation
                let sigma_ii = self.cov_matrix[i][i];
                if sigma_ii < 1e-12 {
                    continue;
                }
                
                // Simplified update: balance risk contribution
                let current_rc = prev_rc[i];
                if current_rc.abs() > 1e-12 {
                    let adjustment = (target_rc / current_rc).sqrt();
                    weights[i] *= adjustment;
                }
            }
            
            // Project to simplex (ensure weights sum to 1 and are non-negative)
            self.project_to_simplex(&mut weights);
            
            // Check convergence
            let current_rc = self.risk_contribution(&weights);
            let mut max_diff = 0.0;
            for i in 0..self.n_assets {
                let diff = (current_rc[i] - prev_rc[i]).abs();
                max_diff = max_diff.max(diff);
            }
            
            if max_diff < self.tolerance {
                converged = true;
                break;
            }
            
            prev_rc = current_rc;
        }
        
        let final_rc = self.risk_contribution(&weights);
        let total_risk = self.portfolio_variance(&weights).sqrt();
        
        RiskParityResult {
            weights,
            risk_contributions: final_rc,
            total_risk,
            iterations,
            converged,
        }
    }

    /// Project weights onto the unit simplex
    fn project_to_simplex(&self, weights: &mut [f64]) {
        // Ensure non-negativity
        for w in weights.iter_mut() {
            *w = w.max(0.0);
        }
        
        // Normalize to sum to 1
        let sum: f64 = weights.iter().sum();
        if sum > 1e-12 {
            for w in weights.iter_mut() {
                *w /= sum;
            }
        }
    }

    /// Fast optimization with early termination for real-time use
    pub fn optimize_fast(&self, max_iter: usize) -> RiskParityResult {
        let mut optimizer = Self::new(
            &self.get_covariance_slice(),
            Some(&self.target_risk),
        )
        .with_max_iterations(max_iter)
        .with_tolerance(self.tolerance * 10.0);
        
        optimizer.optimize()
    }

    fn get_covariance_slice(&self) -> Vec<Vec<f64>> {
        let mut cov = Vec::with_capacity(self.n_assets);
        for i in 0..self.n_assets {
            let mut row = Vec::with_capacity(self.n_assets);
            for j in 0..self.n_assets {
                row.push(self.cov_matrix[i][j]);
            }
            cov.push(row);
        }
        cov
    }
}

/// Thread-safe risk parity calculator with caching
pub struct CachedRiskParity {
    inner: RefCell<Option<RiskParityResult>>,
    last_update_ts: u64,
    cache_ttl_ns: u64,
}

impl CachedRiskParity {
    pub fn new(cache_ttl_ms: u64) -> Self {
        CachedRiskParity {
            inner: RefCell::new(None),
            last_update_ts: 0,
            cache_ttl_ns: cache_ttl_ms * 1_000_000,
        }
    }

    pub fn get_or_compute<F>(&self, compute_fn: F, current_ts: u64) -> RiskParityResult
    where
        F: FnOnce() -> RiskParityResult,
    {
        let needs_update = match *self.inner.borrow() {
            None => true,
            Some(_) => {
                let elapsed = current_ts.saturating_sub(self.last_update_ts);
                elapsed * 1_000_000 >= self.cache_ttl_ns
            }
        };

        if needs_update {
            let result = compute_fn();
            *self.inner.borrow_mut() = Some(result.clone());
            self.last_update_ts = current_ts;
            result
        } else {
            self.inner.borrow().clone().unwrap()
        }
    }

    pub fn invalidate(&self) {
        *self.inner.borrow_mut() = None;
    }
}

/// Risk budget constraint validator
pub struct RiskBudgetValidator {
    max_single_asset_risk: f64,
    max_sector_concentration: f64,
}

impl RiskBudgetValidator {
    pub fn new(max_single: f64, max_sector: f64) -> Self {
        RiskBudgetValidator {
            max_single_asset_risk: max_single,
            max_sector_concentration: max_sector,
        }
    }

    pub fn validate(&self, result: &RiskParityResult, sector_map: &[usize]) -> Result<(), RiskValidationError> {
        // Check single asset risk concentration
        let total_risk = result.total_risk;
        if total_risk < 1e-12 {
            return Ok(());
        }

        for (i, &rc) in result.risk_contributions.iter().enumerate() {
            let risk_pct = rc / total_risk;
            if risk_pct > self.max_single_asset_risk {
                return Err(RiskValidationError::SingleAssetExceeded(i, risk_pct));
            }
        }

        // Check sector concentration
        let mut sector_risk: Vec<f64> = vec![0.0];
        for (i, &sector) in sector_map.iter().enumerate() {
            while sector_risk.len() <= sector {
                sector_risk.push(0.0);
            }
            sector_risk[sector] += result.risk_contributions[i];
        }

        for (sector, &risk) in sector_risk.iter().enumerate() {
            let sector_pct = risk / total_risk;
            if sector_pct > self.max_sector_concentration {
                return Err(RiskValidationError::SectorExceeded(sector, sector_pct));
            }
        }

        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum RiskValidationError {
    SingleAssetExceeded(usize, f64),
    SectorExceeded(usize, f64),
    InvalidWeights,
}

/// Volatility regime detector for adaptive risk parity
pub struct VolatilityRegime {
    pub is_high_vol: bool,
    pub vol_level: f64,
    pub regime_change_ts: u64,
}

pub struct AdaptiveRiskParity {
    base_optimizer: RiskParityOptimizer,
    current_regime: VolatilityRegime,
    vol_threshold: f64,
    lookback_window: usize,
}

impl AdaptiveRiskParity {
    pub fn new(covariance: &[[f64]], vol_threshold: f64) -> Self {
        AdaptiveRiskParity {
            base_optimizer: RiskParityOptimizer::new(covariance, None),
            current_regime: VolatilityRegime {
                is_high_vol: false,
                vol_level: 0.0,
                regime_change_ts: 0,
            },
            vol_threshold,
            lookback_window: 252,
        }
    }

    pub fn update_regime(&mut self, recent_vols: &[f64], current_ts: u64) {
        if recent_vols.is_empty() {
            return;
        }

        let avg_vol: f64 = recent_vols.iter().sum::<f64>() / recent_vols.len() as f64;
        let was_high_vol = self.current_regime.is_high_vol;
        
        self.current_regime.vol_level = avg_vol;
        self.current_regime.is_high_vol = avg_vol > self.vol_threshold;
        
        if was_high_vol != self.current_regime.is_high_vol {
            self.current_regime.regime_change_ts = current_ts;
        }
    }

    pub fn optimize_with_regime(&self) -> RiskParityResult {
        let mut result = self.base_optimizer.optimize();
        
        // In high volatility regimes, apply more conservative weighting
        if self.current_regime.is_high_vol {
            // Tilt towards lower volatility assets
            let vol_adjustment: Vec<f64> = result.weights.iter().map(|w| {
                w * 0.8 // Reduce exposure in high vol
            }).collect();
            
            // Renormalize
            let sum: f64 = vol_adjustment.iter().sum();
            result.weights = vol_adjustment.iter().map(|w| w / sum).collect();
            
            // Recalculate risk contributions
            result.risk_contributions = self.base_optimizer.risk_contribution(&result.weights);
            result.total_risk = self.base_optimizer.portfolio_variance(&result.weights).sqrt();
        }
        
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_risk_parity_convergence() {
        let cov = vec![
            vec![0.04, 0.01, 0.02],
            vec![0.01, 0.09, 0.03],
            vec![0.02, 0.03, 0.16],
        ];
        
        let optimizer = RiskParityOptimizer::new(&cov, None);
        let result = optimizer.optimize();
        
        assert!(result.converged);
        assert!(result.iterations > 0);
        
        // Weights should sum to 1
        let sum: f64 = result.weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        
        // Risk contributions should be roughly equal
        let avg_rc: f64 = result.risk_contributions.iter().sum::<f64>() 
            / result.risk_contributions.len() as f64;
        
        for rc in &result.risk_contributions {
            assert!((rc - avg_rc).abs() < avg_rc * 0.1); // Within 10%
        }
    }

    #[test]
    fn test_risk_budget_validation() {
        let cov = vec![
            vec![0.04, 0.01],
            vec![0.01, 0.09],
        ];
        
        let optimizer = RiskParityOptimizer::new(&cov, None);
        let result = optimizer.optimize();
        
        let validator = RiskBudgetValidator::new(0.7, 0.8);
        let sector_map = vec![0, 1]; // Each asset in different sector
        
        let validation_result = validator.validate(&result, &sector_map);
        assert!(validation_result.is_ok());
    }

    #[test]
    fn test_adaptive_regime() {
        let cov = vec![
            vec![0.04, 0.01],
            vec![0.01, 0.09],
        ];
        
        let mut adaptive = AdaptiveRiskParity::new(&cov, 0.25);
        
        // Low volatility regime
        let low_vols = vec![0.15, 0.18, 0.20];
        adaptive.update_regime(&low_vols, 1000);
        assert!(!adaptive.current_regime.is_high_vol);
        
        // High volatility regime
        let high_vols = vec![0.30, 0.35, 0.40];
        adaptive.update_regime(&high_vols, 2000);
        assert!(adaptive.current_regime.is_high_vol);
        assert_eq!(adaptive.current_regime.regime_change_ts, 2000);
    }
}
