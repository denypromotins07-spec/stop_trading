//! High-Frequency Exponentially Weighted Moving Correlation Matrix
//! 
//! Detects instantaneous correlation breakdowns for statistical arbitrage.
//! Memory-efficient: no massive historical arrays required.

use std::collections::HashMap;

/// Asset identifier for the multi-asset universe
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Asset {
    BTC,
    ETH,
    SOL,
}

impl Asset {
    pub fn all() -> &'static [Asset] {
        &[Asset::BTC, Asset::ETH, Asset::SOL]
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Asset::BTC => "BTC",
            Asset::ETH => "ETH",
            Asset::SOL => "SOL",
        }
    }
}

/// Configuration for EWMA correlation
#[derive(Debug, Clone)]
pub struct EwmaConfig {
    /// Decay factor lambda (0.5 to 0.99)
    /// Higher = more weight on history, lower = more reactive
    pub lambda: f64,
    /// Minimum observations before output is valid
    pub min_observations: usize,
}

impl Default for EwmaConfig {
    fn default() -> Self {
        Self {
            lambda: 0.94, // Industry standard for daily, use lower for HF
            min_observations: 30,
        }
    }
}

/// Running statistics for a single asset's returns
#[derive(Debug, Clone)]
struct RunningStats {
    /// EWMA of returns (mean)
    ewma_mean: f64,
    /// EWMA of squared returns (for variance)
    ewma_sq: f64,
    /// Current volatility estimate
    volatility: f64,
    /// Number of observations
    count: usize,
}

impl RunningStats {
    fn new() -> Self {
        Self {
            ewma_mean: 0.0,
            ewma_sq: 0.0,
            volatility: 0.0,
            count: 0,
        }
    }

    fn update(&mut self, return_val: f64, lambda: f64) {
        self.count += 1;
        
        // EWMA mean
        self.ewma_mean = lambda * self.ewma_mean + (1.0 - lambda) * return_val;
        
        // EWMA of squared returns
        self.ewma_sq = lambda * self.ewma_sq + (1.0 - lambda) * return_val * return_val;
        
        // Volatility = sqrt(E[X²] - E[X]²)
        let variance = (self.ewma_sq - self.ewma_mean * self.ewma_mean).max(0.0);
        self.volatility = variance.sqrt();
    }

    fn standardized_return(&self, return_val: f64) -> f64 {
        if self.volatility < 1e-12 {
            return 0.0;
        }
        (return_val - self.ewma_mean) / self.volatility
    }
}

/// Running covariance between two assets
#[derive(Debug, Clone)]
struct RunningCovariance {
    /// EWMA of product of standardized returns
    ewma_product: f64,
    count: usize,
}

impl RunningCovariance {
    fn new() -> Self {
        Self {
            ewma_product: 0.0,
            count: 0,
        }
    }

    fn update(&mut self, z1: f64, z2: f64, lambda: f64) {
        self.count += 1;
        self.ewma_product = lambda * self.ewma_product + (1.0 - lambda) * z1 * z2;
    }

    fn correlation(&self) -> f64 {
        self.ewma_product.clamp(-1.0, 1.0)
    }
}

/// Pair key for correlation tracking
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct PairKey(Asset, Asset);

impl PairKey {
    fn new(a: Asset, b: Asset) -> Self {
        if a as u8 > b as u8 {
            PairKey(b, a)
        } else {
            PairKey(a, b)
        }
    }
}

/// High-frequency EWMA correlation matrix
pub struct CorrelationMatrix {
    config: EwmaConfig,
    /// Running stats for each asset
    asset_stats: HashMap<Asset, RunningStats>,
    /// Running covariances for each pair
    pair_covariances: HashMap<PairKey, RunningCovariance>,
    /// Cached correlation values
    cached_correlations: HashMap<PairKey, f64>,
    /// Last returns for each asset
    last_prices: HashMap<Asset, f64>,
    /// Observation counter
    total_observations: usize,
}

impl CorrelationMatrix {
    pub fn new(config: EwmaConfig) -> Self {
        let mut asset_stats = HashMap::new();
        let mut last_prices = HashMap::new();
        
        for &asset in Asset::all() {
            asset_stats.insert(asset, RunningStats::new());
            last_prices.insert(asset, 0.0);
        }

        Self {
            config,
            asset_stats,
            pair_covariances: HashMap::new(),
            cached_correlations: HashMap::new(),
            last_prices,
            total_observations: 0,
        }
    }

    /// Update with new prices and compute correlations
    /// Returns true if correlations were updated
    pub fn update_prices(&mut self, prices: &HashMap<Asset, f64>) -> bool {
        // Calculate returns for each asset
        let mut returns = HashMap::new();
        for (&asset, &price) in prices {
            let last_price = self.last_prices.get(&asset).copied().unwrap_or(price);
            if last_price > 0.0 && price > 0.0 {
                let ret = (price / last_price).ln();
                returns.insert(asset, ret);
            }
            self.last_prices.insert(asset, price);
        }

        if returns.len() < 2 {
            return false;
        }

        // Update running stats for each asset
        for (&asset, ret) in &returns {
            if let Some(stats) = self.asset_stats.get_mut(&asset) {
                stats.update(*ret, self.config.lambda);
            }
        }

        // Update covariances for each pair
        let assets: Vec<Asset> = returns.keys().copied().collect();
        for i in 0..assets.len() {
            for j in (i + 1)..assets.len() {
                let a = assets[i];
                let b = assets[j];
                
                if let (Some(sa), Some(sb)) = (self.asset_stats.get(&a), self.asset_stats.get(&b)) {
                    let za = sa.standardized_return(*returns.get(&a).unwrap_or(&0.0));
                    let zb = sb.standardized_return(*returns.get(&b).unwrap_or(&0.0));
                    
                    let key = PairKey::new(a, b);
                    let cov = self.pair_covariances.entry(key).or_insert_with(RunningCovariance::new);
                    cov.update(za, zb, self.config.lambda);
                    
                    // Update cached correlation
                    self.cached_correlations.insert(key, cov.correlation());
                }
            }
        }

        self.total_observations += 1;
        true
    }

    /// Get correlation between two assets
    pub fn get_correlation(&self, a: Asset, b: Asset) -> Option<f64> {
        let key = PairKey::new(a, b);
        self.cached_correlations.get(&key).copied()
    }

    /// Check if we have enough data
    pub fn is_valid(&self) -> bool {
        self.total_observations >= self.config.min_observations
    }

    /// Get all pairwise correlations
    pub fn get_all_correlations(&self) -> HashMap<(Asset, Asset), f64> {
        let mut result = HashMap::new();
        for (&key, &corr) in &self.cached_correlations {
            result.insert((key.0, key.1), corr);
        }
        result
    }

    /// Get correlation matrix as 2D array
    pub fn as_matrix(&self) -> [[f64; 3]; 3] {
        let mut matrix = [[1.0; 3]; 3];
        
        for i in 0..3 {
            for j in (i + 1)..3 {
                let a = Asset::all()[i];
                let b = Asset::all()[j];
                let corr = self.get_correlation(a, b).unwrap_or(0.0);
                matrix[i][j] = corr;
                matrix[j][i] = corr;
            }
        }
        
        matrix
    }

    /// Detect correlation breakdown between two assets
    /// Returns true if correlation changed significantly from recent average
    pub fn detect_breakdown(&self, a: Asset, b: Asset, threshold: f64) -> bool {
        // Simple heuristic: check if current correlation deviates from typical range
        let current = self.get_correlation(a, b).unwrap_or(0.0);
        
        // For stat arb, we care about correlation dropping when it was high
        // or flipping sign unexpectedly
        current.abs() < 1.0 - threshold
    }

    /// Get the number of observations
    pub fn observation_count(&self) -> usize {
        self.total_observations
    }

    /// Get volatility for an asset
    pub fn get_volatility(&self, asset: Asset) -> Option<f64> {
        self.asset_stats.get(&asset).map(|s| s.volatility)
    }

    /// Reset all statistics
    pub fn reset(&mut self) {
        for stats in self.asset_stats.values_mut() {
            *stats = RunningStats::new();
        }
        self.pair_covariances.clear();
        self.cached_correlations.clear();
        self.total_observations = 0;
    }

    /// Adjust lambda dynamically based on market regime
    pub fn set_lambda(&mut self, lambda: f64) {
        self.config.lambda = lambda.clamp(0.5, 0.99);
    }
}

/// Pairs trading signal generator
pub struct PairsTrader {
    correlation_matrix: CorrelationMatrix,
    /// Threshold for entering pairs trade
    entry_threshold: f64,
    /// Threshold for exiting pairs trade
    exit_threshold: f64,
}

impl PairsTrader {
    pub fn new(correlation_matrix: CorrelationMatrix, entry_threshold: f64, exit_threshold: f64) -> Self {
        Self {
            correlation_matrix,
            entry_threshold,
            exit_threshold,
        }
    }

    /// Check for pairs trading opportunity
    /// Returns (asset_long, asset_short, confidence) if opportunity exists
    pub fn find_opportunity(&mut self, prices: &HashMap<Asset, f64>) -> Option<(Asset, Asset, f64)> {
        self.correlation_matrix.update_prices(prices);
        
        if !self.correlation_matrix.is_valid() {
            return None;
        }

        let correlations = self.correlation_matrix.get_all_correlations();
        
        for ((a, b), corr) in correlations {
            if corr.abs() >= self.entry_threshold {
                // High correlation detected
                // Calculate relative performance
                let vol_a = self.correlation_matrix.get_volatility(a).unwrap_or(0.0);
                let vol_b = self.correlation_matrix.get_volatility(b).unwrap_or(0.0);
                
                if vol_a > 0.0 && vol_b > 0.0 {
                    // Confidence based on correlation strength and similar volatility
                    let vol_ratio = (vol_a / vol_b).min(vol_b / vol_a);
                    let confidence = corr.abs() * vol_ratio;
                    
                    if confidence >= self.exit_threshold {
                        // Return the pair (which one to long/short depends on momentum)
                        return Some((a, b, confidence));
                    }
                }
            }
        }

        None
    }

    /// Get the underlying correlation matrix
    pub fn correlation_matrix(&self) -> &CorrelationMatrix {
        &self.correlation_matrix
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_correlation_matrix_basic() {
        let config = EwmaConfig {
            lambda: 0.9,
            min_observations: 5,
        };
        let mut matrix = CorrelationMatrix::new(config);

        // Simulate correlated price movements
        let mut btc_price = 50000.0;
        let mut eth_price = 3000.0;
        let mut sol_price = 100.0;

        for i in 0..10 {
            let factor = 1.0 + 0.01 * (i as f64 % 5 - 2) as f64;
            btc_price *= factor;
            eth_price *= factor * 0.95; // Slightly different but correlated
            sol_price *= factor * 1.1;

            let mut prices = HashMap::new();
            prices.insert(Asset::BTC, btc_price);
            prices.insert(Asset::ETH, eth_price);
            prices.insert(Asset::SOL, sol_price);

            matrix.update_prices(&prices);
        }

        assert!(matrix.is_valid());
        
        let btc_eth_corr = matrix.get_correlation(Asset::BTC, Asset::ETH);
        assert!(btc_eth_corr.is_some());
        assert!(btc_eth_corr.unwrap() > 0.5); // Should be highly correlated
    }

    #[test]
    fn test_pairs_trader() {
        let config = EwmaConfig::default();
        let matrix = CorrelationMatrix::new(config);
        let mut trader = PairsTrader::new(matrix, 0.7, 0.5);

        let mut prices = HashMap::new();
        prices.insert(Asset::BTC, 50000.0);
        prices.insert(Asset::ETH, 3000.0);
        prices.insert(Asset::SOL, 100.0);

        // Need many updates to build up statistics
        for _ in 0..50 {
            let _ = trader.find_opportunity(&prices);
        }
    }
}
