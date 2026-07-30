//! Portfolio Optimization - Black-Litterman Model
//! 
//! Implements a simplified Black-Litterman model combining market equilibrium
//! with proprietary alpha views. Calculates optimal multi-asset weight allocations
//! (BTC, ETH, SOL) balancing expected returns against the covariance matrix.

use std::collections::HashMap;
use nalgebra::{Matrix2, Matrix3, MatrixXx, Vector2, Vector3, VectorX};

use tracing::{debug, info, warn};

/// Black-Litterman configuration
#[derive(Debug, Clone)]
pub struct BlackLittermanConfig {
    /// Risk aversion coefficient (lambda)
    pub risk_aversion: f64,
    /// Scaling factor for uncertainty in views (tau)
    pub tau: f64,
    /// Market capitalization weights
    pub market_caps: HashMap<String, f64>,
    /// Risk-free rate (annualized)
    pub risk_free_rate: f64,
}

impl Default for BlackLittermanConfig {
    fn default() -> Self {
        let mut market_caps = HashMap::new();
        // Approximate crypto market caps (would be updated dynamically)
        market_caps.insert("BTC".to_string(), 0.50);
        market_caps.insert("ETH".to_string(), 0.30);
        market_caps.insert("SOL".to_string(), 0.20);

        Self {
            risk_aversion: 2.5,
            tau: 0.05,
            market_caps,
            risk_free_rate: 0.05, // 5% annual
        }
    }
}

/// Alpha view from proprietary models
#[derive(Debug, Clone)]
pub struct AlphaView {
    pub asset: String,
    pub expected_return: f64,
    pub confidence: f64, // 0.0 to 1.0
    pub view_type: ViewType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViewType {
    Absolute,   // Expected return for single asset
    Relative,   // Asset A will outperform Asset B by X%
}

/// Black-Litterman result
#[derive(Debug, Clone)]
pub struct BlackLittermanResult {
    pub optimal_weights: HashMap<String, f64>,
    pub expected_returns: HashMap<String, f64>,
    pub portfolio_expected_return: f64,
    pub portfolio_volatility: f64,
    pub sharpe_ratio: f64,
}

/// Black-Litterman optimizer
pub struct BlackLittermanOptimizer {
    config: BlackLittermanConfig,
    covariance_matrix: HashMap<(String, String), f64>,
    assets: Vec<String>,
}

impl BlackLittermanOptimizer {
    /// Create a new Black-Litterman optimizer
    pub fn new(config: BlackLittermanConfig) -> Self {
        let assets: Vec<String> = config.market_caps.keys().cloned().collect();
        
        Self {
            config,
            covariance_matrix: HashMap::new(),
            assets,
        }
    }

    /// Set covariance matrix from historical data
    pub fn set_covariance(&mut self, covariances: HashMap<(String, String), f64>) {
        self.covariance_matrix = covariances;
    }

    /// Calculate implied equilibrium returns from market cap weights
    pub fn calculate_equilibrium_returns(&self) -> HashMap<String, f64> {
        let lambda = self.config.risk_aversion;
        let mut returns = HashMap::new();

        // Σ * w gives the marginal contribution to variance
        // Π = λ * Σ * w (implied excess returns)
        
        for asset in &self.assets {
            let weight = self.config.market_caps.get(asset).copied().unwrap_or(0.0);
            
            // Calculate marginal variance contribution
            let mut marginal_var = 0.0;
            for other_asset in &self.assets {
                let other_weight = self.config.market_caps.get(other_asset).copied().unwrap_or(0.0);
                let cov = self.covariance_matrix.get(&(asset.clone(), other_asset.clone()))
                    .copied()
                    .or_else(|| self.covariance_matrix.get(&(other_asset.clone(), asset.clone())).copied())
                    .unwrap_or(0.0);
                
                marginal_var += cov * other_weight;
            }

            // Implied return = λ * marginal variance contribution
            let implied_return = lambda * marginal_var;
            returns.insert(asset.clone(), implied_return + self.config.risk_free_rate);
        }

        returns
    }

    /// Combine equilibrium with views using Black-Litterman formula
    pub fn calculate_optimal_weights(
        &self,
        views: &[AlphaView],
    ) -> BlackLittermanResult {
        let n = self.assets.len();
        
        if n == 0 {
            return self.empty_result();
        }

        // Build covariance matrix
        let sigma = self.build_covariance_matrix(n);
        
        // Get equilibrium returns
        let pi = self.calculate_equilibrium_returns();
        
        // If no views, return equilibrium weights
        if views.is_empty() {
            return self.result_from_weights(self.config.market_caps.clone(), &pi);
        }

        // Build view matrix P and view vector Q
        let k = views.len();
        let mut p_matrix = vec![vec![0.0; n]; k];
        let mut q_vector = vec![0.0; k];
        let mut omega_diag = vec![0.0; k]; // Uncertainty in views

        for (i, view) in views.iter().enumerate() {
            if let Some(asset_idx) = self.assets.iter().position(|a| a == &view.asset) {
                match view.view_type {
                    ViewType::Absolute => {
                        p_matrix[i][asset_idx] = 1.0;
                        q_vector[i] = view.expected_return;
                        // Omega = (confidence * τ * σ²)^-1
                        let var = sigma[asset_idx][asset_idx];
                        omega_diag[i] = (view.confidence * self.config.tau * var).max(1e-10);
                    }
                    ViewType::Relative => {
                        // Would need second asset for relative views
                        p_matrix[i][asset_idx] = 1.0;
                        q_vector[i] = view.expected_return;
                        omega_diag[i] = self.config.tau * 0.1; // Simplified
                    }
                }
            }
        }

        // Black-Litterman formula:
        // E[R] = [(τΣ)^-1 + P'Ω^-1P]^-1 * [(τΣ)^-1Π + P'Ω^-1Q]
        
        // For simplicity, use scalar approximation
        let tau_sigma_inv = self.matrix_scalar_multiply(&sigma, 1.0 / self.config.tau);
        
        // Combine views with equilibrium
        let blended_returns = self.blend_views_with_equilibrium(&pi, views, n);

        // Calculate optimal weights using mean-variance optimization
        // w* = (λΣ)^-1 * E[R]
        let lambda = self.config.risk_aversion;
        let optimal_weights = self.calculate_mean_variance_weights(&blended_returns, lambda);

        // Normalize weights to sum to 1
        let weight_sum: f64 = optimal_weights.values().sum();
        let mut normalized_weights = optimal_weights.clone();
        if weight_sum > 0.0 {
            for w in normalized_weights.values_mut() {
                *w /= weight_sum;
            }
        }

        self.result_from_weights(normalized_weights, &blended_returns)
    }

    /// Blend views with equilibrium returns
    fn blend_views_with_equilibrium(
        &self,
        equilibrium: &HashMap<String, f64>,
        views: &[AlphaView],
        n: usize,
    ) -> HashMap<String, f64> {
        let mut blended = equilibrium.clone();
        
        // Tau controls how much we deviate from equilibrium
        let tau = self.config.tau;

        for view in views {
            if let Some(current) = blended.get_mut(&view.asset) {
                // Weighted average of equilibrium and view
                let view_weight = view.confidence * (1.0 - tau);
                let equilibrium_weight = 1.0 - view_weight;
                
                *current = equilibrium_weight * (*current) + view_weight * view.expected_return;
            } else {
                blended.insert(view.asset.clone(), view.expected_return);
            }
        }

        blended
    }

    /// Calculate mean-variance optimal weights
    fn calculate_mean_variance_weights(
        &self,
        expected_returns: &HashMap<String, f64>,
        lambda: f64,
    ) -> HashMap<String, f64> {
        let mut weights = HashMap::new();

        for asset in &self.assets {
            let ret = expected_returns.get(asset).copied().unwrap_or(0.0);
            let var = self.covariance_matrix.get(&(asset.clone(), asset.clone()))
                .copied()
                .unwrap_or(0.04); // Default 20% vol squared

            // Simple approximation: w_i = E[R_i] / (λ * σ²_i)
            let weight = if var > 1e-10 {
                ret / (lambda * var)
            } else {
                0.0
            };

            weights.insert(asset.clone(), weight.max(0.0)); // Long only
        }

        weights
    }

    /// Build NxN covariance matrix
    fn build_covariance_matrix(&self, n: usize) -> Vec<Vec<f64>> {
        let mut matrix = vec![vec![0.0; n]; n];

        for (i, asset_i) in self.assets.iter().enumerate() {
            for (j, asset_j) in self.assets.iter().enumerate() {
                let cov = self.covariance_matrix.get(&(asset_i.clone(), asset_j.clone()))
                    .copied()
                    .or_else(|| self.covariance_matrix.get(&(asset_j.clone(), asset_i.clone())).copied())
                    .unwrap_or_else(|| {
                        // Default: use variance on diagonal, 0 otherwise
                        if i == j {
                            0.04 // 20% annualized volatility
                        } else {
                            0.0
                        }
                    });
                
                matrix[i][j] = cov;
            }
        }

        matrix
    }

    /// Multiply matrix by scalar
    fn matrix_scalar_multiply(&self, matrix: &[Vec<f64>], scalar: f64) -> Vec<Vec<f64>> {
        matrix.iter()
            .map(|row| row.iter().map(|&v| v * scalar).collect())
            .collect()
    }

    /// Create result from weights and returns
    fn result_from_weights(
        &self,
        weights: HashMap<String, f64>,
        returns: &HashMap<String, f64>,
    ) -> BlackLittermanResult {
        // Calculate portfolio expected return
        let portfolio_return: f64 = weights.iter()
            .map(|(asset, weight)| {
                let ret = returns.get(asset).copied().unwrap_or(0.0);
                weight * ret
            })
            .sum();

        // Calculate portfolio volatility
        let mut portfolio_var = 0.0;
        for (asset_i, weight_i) in &weights {
            for (asset_j, weight_j) in &weights {
                let cov = self.covariance_matrix.get(&(asset_i.clone(), asset_j.clone()))
                    .copied()
                    .or_else(|| self.covariance_matrix.get(&(asset_j.clone(), asset_i.clone())).copied())
                    .unwrap_or(0.0);
                
                portfolio_var += weight_i * weight_j * cov;
            }
        }
        let portfolio_vol = portfolio_var.sqrt();

        // Calculate Sharpe ratio
        let excess_return = portfolio_return - self.config.risk_free_rate;
        let sharpe = if portfolio_vol > 1e-10 {
            excess_return / portfolio_vol
        } else {
            0.0
        };

        BlackLittermanResult {
            optimal_weights: weights,
            expected_returns: returns.clone(),
            portfolio_expected_return: portfolio_return,
            portfolio_volatility: portfolio_vol,
            sharpe_ratio: sharpe,
        }
    }

    fn empty_result(&self) -> BlackLittermanResult {
        BlackLittermanResult {
            optimal_weights: HashMap::new(),
            expected_returns: HashMap::new(),
            portfolio_expected_return: 0.0,
            portfolio_volatility: 0.0,
            sharpe_ratio: 0.0,
        }
    }

    /// Update market cap weights dynamically
    pub fn update_market_caps(&mut self, market_caps: HashMap<String, f64>) {
        // Normalize to sum to 1
        let total: f64 = market_caps.values().sum();
        if total > 0.0 {
            self.config.market_caps = market_caps.into_iter()
                .map(|(k, v)| (k, v / total))
                .collect();
            
            // Update assets list
            self.assets = self.config.market_caps.keys().cloned().collect();
        }
    }
}

/// Risk parity allocator as alternative
pub struct RiskParityAllocator {
    assets: Vec<String>,
    target_risk: f64,
}

impl RiskParityAllocator {
    pub fn new(assets: Vec<String>, target_risk: f64) -> Self {
        Self { assets, target_risk }
    }

    /// Calculate risk parity weights
    pub fn calculate_weights(&self, volatilities: &HashMap<String, f64>) -> HashMap<String, f64> {
        let mut weights = HashMap::new();
        let mut total_inverse_vol = 0.0;

        // Calculate inverse volatility weights
        for asset in &self.assets {
            let vol = volatilities.get(asset).copied().unwrap_or(0.2);
            let inverse_vol = 1.0 / vol.max(0.01);
            total_inverse_vol += inverse_vol;
            weights.insert(asset.clone(), inverse_vol);
        }

        // Normalize to sum to 1
        for weight in weights.values_mut() {
            *weight /= total_inverse_vol;
        }

        weights
    }
}

/// Print allocation summary
pub fn print_allocation_summary(result: &BlackLittermanResult) {
    println!("=== Black-Litterman Allocation ===");
    println!();
    println!("Optimal Weights:");
    for (asset, weight) in &result.optimal_weights {
        println!("  {}: {:.1}%", asset, weight * 100.0);
    }
    println!();
    println!("Expected Returns:");
    for (asset, ret) in &result.expected_returns {
        println!("  {}: {:.2}% annual", asset, ret * 100.0);
    }
    println!();
    println!("Portfolio Metrics:");
    println!("  Expected Return: {:.2}%", result.portfolio_expected_return * 100.0);
    println!("  Volatility: {:.2}%", result.portfolio_volatility * 100.0);
    println!("  Sharpe Ratio: {:.3}", result.sharpe_ratio);
}
