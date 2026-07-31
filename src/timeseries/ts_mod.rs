//! Time-Series Module Root
//! 
//! Exports wavelet and fractional differentiation features
//! directly to the lock-free feature store.

pub mod wavelet;
pub mod fractional;

pub use wavelet::{
    Haar,
    WaveletDecomposition,
    WaveletTransformer,
    WaveletType,
    MultiResolutionAnalyzer,
};

pub use fractional::{
    FractionalDiffResult,
    FractionalDifferentiator,
    MultiOrderFractionalDifferentiator,
};

/// Combined time-series feature extractor
pub struct TimeSeriesFeatureExtractor {
    wavelet_analyzer: MultiResolutionAnalyzer,
    fractional_diff: FractionalDifferentiator,
}

impl TimeSeriesFeatureExtractor {
    pub fn new() -> Self {
        Self {
            wavelet_analyzer: MultiResolutionAnalyzer::new(),
            fractional_diff: FractionalDifferentiator::new(),
        }
    }
    
    /// Push a new tick to all time-series analyzers
    #[inline]
    pub fn push_tick(&mut self, price: f64) {
        self.wavelet_analyzer.push_tick(price);
        self.fractional_diff.push_price(price);
    }
    
    /// Get combined features for ML model
    pub fn get_all_features(&self) -> TimeSeriesFeatures {
        TimeSeriesFeatures {
            trend: self.wavelet_analyzer.get_consensus_trend(4),
            noise: self.wavelet_analyzer.get_noise_estimate(),
            hurst: self.fractional_diff.estimate_hurst(),
            optimal_d: self.fractional_diff.find_optimal_d(),
            last_frac_diff: self.fractional_diff.get_last_diff(0.5),
        }
    }
}

impl Default for TimeSeriesFeatureExtractor {
    fn default() -> Self {
        Self::new()
    }
}

/// Aggregated time-series features
#[derive(Debug, Clone)]
pub struct TimeSeriesFeatures {
    pub trend: f64,
    pub noise: f64,
    pub hurst: f64,
    pub optimal_d: f64,
    pub last_frac_diff: f64,
}

impl TimeSeriesFeatures {
    /// Convert to feature vector for ML
    pub fn to_vector(&self) -> [f64; 5] {
        [self.trend, self.noise, self.hurst, self.optimal_d, self.last_frac_diff]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_feature_extractor() {
        let mut extractor = TimeSeriesFeatureExtractor::new();
        
        for i in 0..2048 {
            let price = 100.0 + (i as f64 * 0.05).sin() * 2.0;
            extractor.push_tick(price);
        }
        
        let features = extractor.get_all_features();
        assert!(features.hurst >= 0.0 && features.hurst <= 1.0);
    }
}
