//! Cross-Sectional Momentum and Mean-Reversion Z-Score Ranker
//! 
//! Builds cross-sectional momentum and mean-reversion features across all trading pairs.

use std::collections::HashMap;

/// Asset identifier for cross-sectional analysis
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AssetId(pub u32);

/// Configuration for cross-sectional ranking
#[derive(Debug, Clone)]
pub struct CrossSectionalConfig {
    /// Lookback period for momentum calculation (in ticks)
    pub momentum_lookback: usize,
    /// Lookback period for mean reversion
    pub mean_reversion_lookback: usize,
    /// Minimum number of assets required for ranking
    pub min_assets: usize,
}

impl Default for CrossSectionalConfig {
    fn default() -> Self {
        Self {
            momentum_lookback: 100,
            mean_reversion_lookback: 50,
            min_assets: 3,
        }
    }
}

/// Rolling statistics for a single asset
#[derive(Debug, Clone)]
struct RollingStats {
    prices: Vec<f64>,
    returns: Vec<f64>,
    sum_prices: f64,
    sum_sq_prices: f64,
    sum_returns: f64,
    sum_sq_returns: f64,
}

impl RollingStats {
    fn new(capacity: usize) -> Self {
        Self {
            prices: Vec::with_capacity(capacity),
            returns: Vec::with_capacity(capacity),
            sum_prices: 0.0,
            sum_sq_prices: 0.0,
            sum_returns: 0.0,
            sum_sq_returns: 0.0,
        }
    }

    fn update(&mut self, price: f64, max_len: usize) {
        // Calculate return if we have previous price
        if let Some(&prev_price) = self.prices.last() {
            if prev_price > 0.0 {
                let ret = (price / prev_price).ln();
                self.returns.push(ret);
                self.sum_returns += ret;
                self.sum_sq_returns += ret * ret;
                
                // Trim returns if needed
                if self.returns.len() > max_len {
                    if let Some(old_ret) = self.returns.remove(0) {
                        self.sum_returns -= old_ret;
                        self.sum_sq_returns -= old_ret * old_ret;
                    }
                }
            }
        }

        // Update prices
        self.prices.push(price);
        self.sum_prices += price;
        self.sum_sq_prices += price * price;

        // Trim prices if needed
        if self.prices.len() > max_len {
            if let Some(old_price) = self.prices.remove(0) {
                self.sum_prices -= old_price;
                self.sum_sq_prices -= old_price * old_price;
            }
        }
    }

    fn mean_return(&self) -> f64 {
        if self.returns.is_empty() {
            return 0.0;
        }
        self.sum_returns / self.returns.len() as f64
    }

    fn volatility(&self) -> f64 {
        let n = self.returns.len() as f64;
        if n < 2.0 {
            return 0.0;
        }
        let variance = (self.sum_sq_returns - self.sum_returns * self.sum_returns / n) / (n - 1.0);
        variance.max(0.0).sqrt()
    }

    fn momentum(&self, lookback: usize) -> f64 {
        if self.prices.len() < 2 {
            return 0.0;
        }
        
        let start_idx = self.prices.len().saturating_sub(lookback);
        if start_idx >= self.prices.len() {
            return 0.0;
        }
        
        let start_price = self.prices[start_idx];
        let end_price = *self.prices.last().unwrap();
        
        if start_price <= 0.0 {
            return 0.0;
        }
        
        (end_price / start_price) - 1.0
    }

    fn z_score(&self, value: f64, lookback: usize) -> f64 {
        let n = self.returns.len() as f64;
        if n < 2.0 {
            return 0.0;
        }
        
        let mean = self.mean_return();
        let vol = self.volatility();
        
        if vol < 1e-9 {
            return 0.0;
        }
        
        (value - mean) / vol
    }
}

/// Cross-sectional ranker for momentum and mean-reversion
pub struct CrossSectionalRanker {
    config: CrossSectionalConfig,
    asset_stats: HashMap<AssetId, RollingStats>,
    /// Current momentum z-scores
    momentum_zscores: HashMap<AssetId, f64>,
    /// Current mean-reversion z-scores
    mr_zscores: HashMap<AssetId, f64>,
    /// Momentum ranks (higher = stronger momentum)
    momentum_ranks: HashMap<AssetId, f64>,
    /// Mean-reversion ranks (higher = more oversold)
    mr_ranks: HashMap<AssetId, f64>,
}

impl CrossSectionalRanker {
    pub fn new(config: CrossSectionalConfig) -> Self {
        Self {
            config,
            asset_stats: HashMap::new(),
            momentum_zscores: HashMap::new(),
            mr_zscores: HashMap::new(),
            momentum_ranks: HashMap::new(),
            mr_ranks: HashMap::new(),
        }
    }

    /// Update with new price for an asset
    pub fn update_price(&mut self, asset: AssetId, price: f64) {
        let max_lookback = self.config.momentum_lookback.max(self.config.mean_reversion_lookback);
        
        let stats = self.asset_stats.entry(asset)
            .or_insert_with(|| RollingStats::new(max_lookback));
        stats.update(price, max_lookback);
        
        // Recalculate all scores
        self.recalculate_scores();
    }

    fn recalculate_scores(&mut self) {
        if self.asset_stats.len() < self.config.min_assets {
            return;
        }

        // Calculate momentum for each asset
        let mut momentums: Vec<(AssetId, f64)> = self.asset_stats.iter()
            .map(|(&asset, stats)| (asset, stats.momentum(self.config.momentum_lookback)))
            .collect();
        
        // Calculate mean reversion signal (negative of recent return = mean reversion opportunity)
        let mut mr_signals: Vec<(AssetId, f64)> = self.asset_stats.iter()
            .map(|(&asset, stats)| (asset, -stats.mean_return()))
            .collect();

        // Sort and rank momentums
        momentums.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        for (rank, &(asset, mom)) in momentums.iter().enumerate() {
            let normalized_rank = rank as f64 / (momentums.len() - 1) as f64;
            self.momentum_ranks.insert(asset, normalized_rank);
            self.momentum_zscores.insert(asset, mom);
        }

        // Sort and rank mean-reversion signals
        mr_signals.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        for (rank, &(asset, mr)) in mr_signals.iter().enumerate() {
            let normalized_rank = rank as f64 / (mr_signals.len() - 1) as f64;
            self.mr_ranks.insert(asset, normalized_rank);
            self.mr_zscores.insert(asset, mr);
        }
    }

    /// Get momentum rank for an asset (0 to 1)
    pub fn get_momentum_rank(&self, asset: AssetId) -> Option<f64> {
        self.momentum_ranks.get(&asset).copied()
    }

    /// Get mean-reversion rank for an asset (0 to 1)
    pub fn get_mr_rank(&self, asset: AssetId) -> Option<f64> {
        self.mr_ranks.get(&asset).copied()
    }

    /// Get combined score (positive = long momentum, negative = short momentum/long MR)
    pub fn get_combined_signal(&self, asset: AssetId, momentum_weight: f64) -> Option<f64> {
        let mom_rank = self.get_momentum_rank(asset)?;
        let mr_rank = self.get_mr_rank(asset)?;
        
        let mr_weight = 1.0 - momentum_weight;
        Some(mom_rank * momentum_weight + mr_rank * mr_weight)
    }

    /// Get top N assets by momentum
    pub fn top_momentum(&self, n: usize) -> Vec<(AssetId, f64)> {
        let mut assets: Vec<_> = self.momentum_ranks.iter()
            .map(|(&a, &r)| (a, r))
            .collect();
        assets.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        assets.truncate(n);
        assets
    }

    /// Get bottom N assets by momentum (best for mean-reversion longs)
    pub fn bottom_momentum(&self, n: usize) -> Vec<(AssetId, f64)> {
        let mut assets: Vec<_> = self.momentum_ranks.iter()
            .map(|(&a, &r)| (a, r))
            .collect();
        assets.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        assets.truncate(n);
        assets
    }

    /// Get the number of tracked assets
    pub fn asset_count(&self) -> usize {
        self.asset_stats.len()
    }

    /// Check if we have enough data for reliable rankings
    pub fn is_ready(&self) -> bool {
        self.asset_stats.len() >= self.config.min_assets
            && self.asset_stats.values().all(|s| s.prices.len() >= self.config.momentum_lookback / 2)
    }

    /// Clear all data
    pub fn clear(&mut self) {
        self.asset_stats.clear();
        self.momentum_zscores.clear();
        self.mr_zscores.clear();
        self.momentum_ranks.clear();
        self.mr_ranks.clear();
    }
}

/// Feature vector for ML model consumption
#[derive(Debug, Clone)]
pub struct FeatureVector {
    pub asset: AssetId,
    pub momentum_z: f64,
    pub mr_z: f64,
    pub momentum_rank: f64,
    pub mr_rank: f64,
    pub volatility: f64,
    pub recent_return: f64,
}

impl FeatureVector {
    pub fn new(asset: AssetId) -> Self {
        Self {
            asset,
            momentum_z: 0.0,
            mr_z: 0.0,
            momentum_rank: 0.0,
            mr_rank: 0.0,
            volatility: 0.0,
            recent_return: 0.0,
        }
    }
}

/// Feature extractor that generates vectors for IPC serialization
pub struct FeatureExtractor<'a> {
    ranker: &'a CrossSectionalRanker,
}

impl<'a> FeatureExtractor<'a> {
    pub fn new(ranker: &'a CrossSectionalRanker) -> Self {
        Self { ranker }
    }

    /// Generate feature vector for an asset
    pub fn extract_features(&self, asset: AssetId) -> Option<FeatureVector> {
        let stats = self.ranker.asset_stats.get(&asset)?;
        
        let mut features = FeatureVector::new(asset);
        features.momentum_z = *self.ranker.momentum_zscores.get(&asset)?;
        features.mr_z = *self.ranker.mr_zscores.get(&asset)?;
        features.momentum_rank = *self.ranker.momentum_ranks.get(&asset)?;
        features.mr_rank = *self.ranker.mr_ranks.get(&asset)?;
        features.volatility = stats.volatility();
        features.recent_return = stats.mean_return();
        
        Some(features)
    }

    /// Generate feature vectors for all assets
    pub fn extract_all(&self) -> Vec<FeatureVector> {
        self.ranker.asset_stats.keys()
            .filter_map(|&asset| self.extract_features(asset))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cross_sectional_ranker() {
        let config = CrossSectionalConfig {
            momentum_lookback: 10,
            mean_reversion_lookback: 10,
            min_assets: 3,
        };
        let mut ranker = CrossSectionalRanker::new(config);

        // Simulate price movements for 3 assets
        // Asset 0: strong uptrend
        // Asset 1: sideways
        // Asset 2: downtrend
        
        for i in 0..20 {
            ranker.update_price(AssetId(0), 100.0 * (1.0 + 0.01 * i as f64));
            ranker.update_price(AssetId(1), 100.0 + (i % 3) as f64);
            ranker.update_price(AssetId(2), 100.0 * (1.0 - 0.01 * i as f64));
        }

        assert!(ranker.is_ready());
        
        // Asset 0 should have highest momentum rank
        let rank0 = ranker.get_momentum_rank(AssetId(0)).unwrap();
        let rank2 = ranker.get_momentum_rank(AssetId(2)).unwrap();
        assert!(rank0 > rank2);
    }

    #[test]
    fn test_feature_extractor() {
        let config = CrossSectionalConfig::default();
        let mut ranker = CrossSectionalRanker::new(config);

        for i in 0..100 {
            ranker.update_price(AssetId(1), 100.0 + i as f64);
            ranker.update_price(AssetId(2), 100.0 - i as f64 * 0.5);
            ranker.update_price(AssetId(3), 100.0 + (i % 10) as f64);
        }

        let extractor = FeatureExtractor::new(&ranker);
        let features = extractor.extract_features(AssetId(1));
        
        assert!(features.is_some());
        let f = features.unwrap();
        assert!(f.momentum_z > 0.0); // Uptrend
    }
}
