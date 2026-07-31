//! Surface Module Root
//! 
//! Integrates volatility and correlation metrics into the Black-Litterman portfolio optimizer.

pub mod cross_asset;
pub mod correlation;

use cross_asset::{VolatilitySurface, VolPoint};
use correlation::{CorrelationMatrix, EwmaConfig, Asset};
use std::collections::HashMap;

/// Configuration for the surface module
#[derive(Debug, Clone)]
pub struct SurfaceConfig {
    pub ewma_config: EwmaConfig,
    /// Min strikes per expiry for vol surface
    pub min_vol_points: usize,
}

impl Default for SurfaceConfig {
    fn default() -> Self {
        Self {
            ewma_config: EwmaConfig::default(),
            min_vol_points: 3,
        }
    }
}

/// Combined market state from volatility and correlation
#[derive(Debug, Clone)]
pub struct MarketState {
    /// Current ATM vol by asset and expiry
    pub atm_vols: HashMap<Asset, HashMap<u32, f64>>,
    /// Correlation matrix
    pub correlations: [[f64; 3]; 3],
    /// Volatility by asset
    pub vols: HashMap<Asset, f64>,
    /// Skew by asset (10-delta put skew)
    pub skews: HashMap<Asset, f64>,
}

impl Default for MarketState {
    fn default() -> Self {
        Self {
            atm_vols: HashMap::new(),
            correlations: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            vols: HashMap::new(),
            skews: HashMap::new(),
        }
    }
}

/// Black-Litterman expected return estimator
pub struct BlackLittermanOptimizer {
    /// Risk-free rate
    risk_free_rate: f64,
    /// Risk aversion coefficient
    risk_aversion: f64,
    /// Market cap weights (prior)
    market_weights: HashMap<Asset, f64>,
    /// Current view adjustments
    views: Vec<View>,
}

/// A view on expected returns
#[derive(Debug, Clone)]
pub struct View {
    /// Assets involved in the view
    pub assets: Vec<Asset>,
    /// Weights (positive = long, negative = short)
    pub weights: Vec<f64>,
    /// Expected return of the view
    pub expected_return: f64,
    /// Confidence in the view (0 to 1)
    pub confidence: f64,
}

impl BlackLittermanOptimizer {
    pub fn new(risk_free_rate: f64, risk_aversion: f64) -> Self {
        let mut market_weights = HashMap::new();
        // Default market caps (normalized)
        market_weights.insert(Asset::BTC, 0.6);
        market_weights.insert(Asset::ETH, 0.3);
        market_weights.insert(Asset::SOL, 0.1);

        Self {
            risk_free_rate,
            risk_aversion,
            market_weights,
            views: Vec::new(),
        }
    }

    /// Calculate implied equilibrium returns using Black-Litterman
    pub fn calculate_equilibrium_returns(&self, cov_matrix: &[[f64; 3]; 3]) -> HashMap<Asset, f64> {
        let assets = Asset::all();
        let n = assets.len();
        
        // Build weight vector
        let weights: Vec<f64> = assets.iter().map(|&a| self.market_weights.get(&a).copied().unwrap_or(0.0)).collect();
        
        // Pi = δ * Σ * w (implied returns)
        let mut returns = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                returns[i] += cov_matrix[i][j] * weights[j];
            }
            returns[i] *= self.risk_aversion;
        }

        // Add risk-free rate
        let mut result = HashMap::new();
        for (i, &asset) in assets.iter().enumerate() {
            result.insert(asset, returns[i] + self.risk_free_rate);
        }
        result
    }

    /// Add a view on expected returns
    pub fn add_view(&mut self, view: View) {
        self.views.push(view);
    }

    /// Clear all views
    pub fn clear_views(&mut self) {
        self.views.clear();
    }

    /// Calculate posterior expected returns with views
    pub fn calculate_posterior_returns(&self, cov_matrix: &[[f64; 3]; 3]) -> HashMap<Asset, f64> {
        if self.views.is_empty() {
            return self.calculate_equilibrium_returns(cov_matrix);
        }

        let assets = Asset::all();
        let n = assets.len();
        let tau = 0.05; // Scaling factor for uncertainty in prior
        
        // Get prior returns
        let pi = self.calculate_equilibrium_returns(cov_matrix);
        let pi_vec: Vec<f64> = assets.iter().map(|&a| pi.get(&a).copied().unwrap_or(0.0)).collect();

        // For each view, blend with prior based on confidence
        let mut posterior = pi_vec.clone();
        
        for view in &self.views {
            if view.assets.is_empty() || view.confidence <= 0.0 {
                continue;
            }

            // Simple blending: posterior = (1-c)*prior + c*view
            let conf = view.confidence.min(1.0);
            
            for (i, &asset) in assets.iter().enumerate() {
                if let Some(idx) = view.assets.iter().position(|&a| a == asset) {
                    let view_contrib = view.weights[idx] * view.expected_return;
                    posterior[i] = (1.0 - conf) * posterior[i] + conf * view_contrib;
                }
            }
        }

        let mut result = HashMap::new();
        for (i, &asset) in assets.iter().enumerate() {
            result.insert(asset, posterior[i]);
        }
        result
    }

    /// Optimize portfolio weights given expected returns
    pub fn optimize_weights(&self, expected_returns: &HashMap<Asset, f64>, cov_matrix: &[[f64; 3]; 3]) -> HashMap<Asset, f64> {
        let assets = Asset::all();
        let n = assets.len();
        
        // Simple mean-variance optimization: w = (1/δ) * Σ^(-1) * μ
        // Using simplified inverse for 3x3
        
        let returns: Vec<f64> = assets.iter().map(|&a| expected_returns.get(&a).copied().unwrap_or(0.0)).collect();
        
        // For simplicity, use diagonal approximation
        let mut weights = Vec::with_capacity(n);
        for i in 0..n {
            let var = cov_matrix[i][i].max(1e-9);
            let excess_return = returns[i] - self.risk_free_rate;
            weights.push(excess_return / (self.risk_aversion * var));
        }

        // Normalize to sum to 1
        let sum: f64 = weights.iter().sum();
        if sum.abs() > 1e-9 {
            for w in &mut weights {
                *w /= sum;
            }
        }

        // Clamp to reasonable bounds
        for w in &mut weights {
            *w = w.clamp(-0.5, 1.0);
        }

        let mut result = HashMap::new();
        for (i, &asset) in assets.iter().enumerate() {
            result.insert(asset, weights[i]);
        }
        result
    }

    /// Update market weights
    pub fn set_market_weights(&mut self, weights: HashMap<Asset, f64>) {
        self.market_weights = weights;
    }
}

/// Main surface analytics engine
pub struct SurfaceEngine {
    config: SurfaceConfig,
    vol_surface: VolatilitySurface,
    correlation_matrix: CorrelationMatrix,
    bl_optimizer: BlackLittermanOptimizer,
    current_state: MarketState,
}

impl SurfaceEngine {
    pub fn new(config: SurfaceConfig) -> Self {
        Self {
            config,
            vol_surface: VolatilitySurface::new(),
            correlation_matrix: CorrelationMatrix::new(config.ewma_config.clone()),
            bl_optimizer: BlackLittermanOptimizer::new(0.05, 2.5),
            current_state: MarketState::default(),
        }
    }

    /// Update with new option data
    pub fn update_option_data(&mut self, asset: Asset, strike: f64, expiry_days: u32, 
                               is_call: bool, price: f64, spot: f64) {
        if spot > 0.0 && price > 0.0 {
            let moneyness = strike / spot;
            let time_years = expiry_days as f64 / 365.0;
            
            // Calculate IV using Black-Scholes
            if let Some(iv) = cross_asset::black_scholes::implied_vol(
                price, spot, strike, time_years, 0.05, is_call
            ) {
                self.vol_surface.add_point(VolPoint {
                    moneyness,
                    time_to_expiry: time_years,
                    implied_vol: iv,
                });
            }
        }
    }

    /// Update with new spot prices
    pub fn update_prices(&mut self, prices: &HashMap<Asset, f64>) {
        self.correlation_matrix.update_prices(prices);
        self.update_state();
    }

    fn update_state(&mut self) {
        // Build vol surface
        self.vol_surface.build_surface();

        // Extract ATM vols by asset (simplified - in practice would have per-asset surfaces)
        let mut atm_vols = HashMap::new();
        let mut vols = HashMap::new();
        let mut skews = HashMap::new();

        for &asset in Asset::all() {
            let mut asset_vols = HashMap::new();
            
            // Sample common expiries
            for days in [7, 14, 30, 60, 90] {
                if let Some(vol) = self.vol_surface.get_atm_vol(days) {
                    asset_vols.insert(days, vol);
                }
            }
            
            if !asset_vols.is_empty() {
                atm_vols.insert(asset, asset_vols);
                
                // Use 30-day vol as representative
                if let Some(v) = self.vol_surface.get_atm_vol(30) {
                    vols.insert(asset, v);
                }
                
                // Calculate skew
                if let Some(skew) = self.vol_surface.get_skew(30, 0.1) {
                    skews.insert(asset, skew);
                }
            }
        }

        self.current_state = MarketState {
            atm_vols,
            correlations: self.correlation_matrix.as_matrix(),
            vols,
            skews,
        };
    }

    /// Get current market state
    pub fn get_state(&self) -> &MarketState {
        &self.current_state
    }

    /// Get optimal portfolio weights
    pub fn get_optimal_weights(&mut self) -> HashMap<Asset, f64> {
        let cov = self.current_state.correlations;
        let returns = self.bl_optimizer.calculate_posterior_returns(&cov);
        self.bl_optimizer.optimize_weights(&returns, &cov)
    }

    /// Add a view to the Black-Litterman model
    pub fn add_view(&mut self, view: View) {
        self.bl_optimizer.add_view(view);
    }

    /// Get the volatility surface for direct access
    pub fn vol_surface(&self) -> &VolatilitySurface {
        &self.vol_surface
    }

    /// Get the correlation matrix for direct access
    pub fn correlation_matrix(&self) -> &CorrelationMatrix {
        &self.correlation_matrix
    }

    /// Reset all models
    pub fn reset(&mut self) {
        self.vol_surface.clear();
        self.correlation_matrix.reset();
        self.bl_optimizer.clear_views();
        self.current_state = MarketState::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_black_litterman_basic() {
        let mut optimizer = BlackLittermanOptimizer::new(0.05, 2.5);
        
        let cov = [[0.04, 0.02, 0.01],
                   [0.02, 0.06, 0.02],
                   [0.01, 0.02, 0.08]];
        
        let equilibrium = optimizer.calculate_equilibrium_returns(&cov);
        assert!(equilibrium.contains_key(&Asset::BTC));
        
        // Add a bullish view on BTC
        optimizer.add_view(View {
            assets: vec![Asset::BTC],
            weights: vec![1.0],
            expected_return: 0.20,
            confidence: 0.7,
        });
        
        let posterior = optimizer.calculate_posterior_returns(&cov);
        assert!(posterior.get(&Asset::BTC).unwrap() > equilibrium.get(&Asset::BTC).unwrap());
    }

    #[test]
    fn test_surface_engine() {
        let config = SurfaceConfig::default();
        let mut engine = SurfaceEngine::new(config);

        // Add some option data
        engine.update_option_data(Asset::BTC, 50000.0, 30, true, 2000.0, 50000.0);
        engine.update_option_data(Asset::BTC, 50000.0, 30, false, 1500.0, 50000.0);
        
        // Add prices for correlation
        let mut prices = HashMap::new();
        prices.insert(Asset::BTC, 50000.0);
        prices.insert(Asset::ETH, 3000.0);
        prices.insert(Asset::SOL, 100.0);
        
        for _ in 0..35 {
            engine.update_prices(&prices);
        }

        let state = engine.get_state();
        assert!(!state.vols.is_empty() || !state.correlations[0][1].is_nan());
    }
}
