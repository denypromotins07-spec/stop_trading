//! Quantitative module root.
//! Exports traits for feature generation feeding ML models.

pub mod series;
pub mod technicals;

pub use series::{RingBuffer, RollingStats, RingBufferError};
pub use technicals::{
    EMA, SMA, RSI, MACD, MacdResult, BollingerBands, BollingerBandsResult, FixedPoint,
};

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Trait for quantitative feature generators
pub trait FeatureGenerator {
    /// Update the feature with a new data point
    fn update(&self, value: f64);
    
    /// Get the current feature value(s)
    fn get_features(&self) -> Vec<f64>;
    
    /// Reset the feature generator
    fn reset(&self);
    
    /// Get the number of features produced
    fn feature_count(&self) -> usize;
}

/// Trait for technical indicator calculators
pub trait TechnicalIndicator {
    /// Update indicator with new price data
    fn update_price(&self, price: f64) -> Option<f64>;
    
    /// Get the current indicator value
    fn get_value(&self) -> Option<f64>;
    
    /// Check if indicator is ready (warmed up)
    fn is_ready(&self) -> bool;
    
    /// Reset the indicator
    fn reset(&self);
}

/// Implementation of TechnicalIndicator for EMA
impl TechnicalIndicator for EMA {
    fn update_price(&self, price: f64) -> Option<f64> {
        Some(self.update(price))
    }
    
    fn get_value(&self) -> Option<f64> {
        self.get()
    }
    
    fn is_ready(&self) -> bool {
        self.initialized.load(Ordering::Relaxed) != 0
    }
    
    fn reset(&self) {
        EMA::reset(self);
    }
}

/// Implementation of TechnicalIndicator for RSI
impl TechnicalIndicator for RSI {
    fn update_price(&self, price: f64) -> Option<f64> {
        self.update(price)
    }
    
    fn get_value(&self) -> Option<f64> {
        self.get()
    }
    
    fn is_ready(&self) -> bool {
        self.initialized.load(Ordering::Relaxed) != 0
    }
    
    fn reset(&self) {
        RSI::reset(self);
    }
}

/// Composite feature generator combining multiple indicators
pub struct CompositeFeatures {
    ema_fast: EMA,
    ema_slow: EMA,
    rsi: RSI,
    macd: MACD,
    last_update: AtomicU64,
}

impl CompositeFeatures {
    /// Create a new composite feature generator
    pub fn new(
        fast_period: usize,
        slow_period: usize,
        rsi_period: usize,
        macd_fast: usize,
        macd_slow: usize,
        macd_signal: usize,
    ) -> Self {
        Self {
            ema_fast: EMA::new(fast_period),
            ema_slow: EMA::new(slow_period),
            rsi: RSI::new(rsi_period),
            macd: MACD::new(macd_fast, macd_slow, macd_signal),
            last_update: AtomicU64::new(0),
        }
    }

    /// Standard configuration (12/26 EMA, 14 RSI, standard MACD)
    pub fn standard() -> Self {
        Self::new(12, 26, 14, 12, 26, 9)
    }

    /// Get timestamp of last update
    pub fn last_update_timestamp(&self) -> u64 {
        self.last_update.load(Ordering::Relaxed)
    }
}

impl FeatureGenerator for CompositeFeatures {
    fn update(&self, value: f64) {
        self.ema_fast.update(value);
        self.ema_slow.update(value);
        self.rsi.update(value);
        self.macd.update(value);
        
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;
        self.last_update.store(timestamp, Ordering::Relaxed);
    }

    fn get_features(&self) -> Vec<f64> {
        let mut features = Vec::with_capacity(7);
        
        // EMA values
        if let Some(fast) = self.ema_fast.get() {
            features.push(fast);
        } else {
            features.push(0.0);
        }
        
        if let Some(slow) = self.ema_slow.get() {
            features.push(slow);
        } else {
            features.push(0.0);
        }
        
        // EMA crossover signal (fast - slow)
        let fast = self.ema_fast.get().unwrap_or(0.0);
        let slow = self.ema_slow.get().unwrap_or(0.0);
        features.push(fast - slow);
        
        // RSI
        if let Some(rsi) = self.rsi.get() {
            features.push(rsi);
        } else {
            features.push(50.0); // Neutral
        }
        
        // MACD components
        if let Some(macd_result) = self.macd.get() {
            features.push(macd_result.macd_line);
            features.push(macd_result.signal_line);
            features.push(macd_result.histogram);
        } else {
            features.extend_from_slice(&[0.0, 0.0, 0.0]);
        }
        
        features
    }

    fn reset(&self) {
        self.ema_fast.reset();
        self.ema_slow.reset();
        self.rsi.reset();
        self.macd.reset();
        self.last_update.store(0, Ordering::Relaxed);
    }

    fn feature_count(&self) -> usize {
        7 // fast_ema, slow_ema, ema_diff, rsi, macd_line, signal_line, histogram
    }
}

/// Volatility-adjusted feature generator
pub struct VolatilityAdjustedFeatures<const WINDOW: usize> {
    rolling_stats: RollingStats<WINDOW>,
    base_features: CompositeFeatures,
}

impl<const WINDOW: usize> VolatilityAdjustedFeatures<WINDOW> {
    pub fn new() -> Self {
        Self {
            rolling_stats: RollingStats::new(),
            base_features: CompositeFeatures::standard(),
        }
    }

    /// Get volatility-adjusted features including Z-scores
    pub fn get_adjusted_features(&self, current_price: f64) -> Vec<f64> {
        let mut features = self.base_features.get_features();
        
        // Add Z-score of current price
        if let Some(z) = self.rolling_stats.z_score(current_price) {
            features.push(z);
        } else {
            features.push(0.0);
        }
        
        // Add rolling standard deviation as volatility proxy
        if let Some(std_dev) = self.rolling_stats.std_dev() {
            features.push(std_dev);
        } else {
            features.push(0.0);
        }
        
        features
    }
}

impl<const WINDOW: usize> Default for VolatilityAdjustedFeatures<WINDOW> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const WINDOW: usize> FeatureGenerator for VolatilityAdjustedFeatures<WINDOW> {
    fn update(&self, value: f64) {
        self.rolling_stats.update(value);
        self.base_features.update(value);
    }

    fn get_features(&self) -> Vec<f64> {
        self.base_features.get_features()
    }

    fn reset(&self) {
        self.rolling_stats.reset();
        self.base_features.reset();
    }

    fn feature_count(&self) -> usize {
        self.base_features.feature_count() + 2 // +2 for z_score and std_dev
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_composite_features() {
        let features = CompositeFeatures::standard();
        
        // Feed some prices
        for i in 0..50 {
            let price = 100.0 + (i as f64).sin() * 5.0;
            features.update(price);
        }
        
        let feature_vec = features.get_features();
        assert_eq!(feature_vec.len(), 7);
        
        // All features should be finite
        for f in &feature_vec {
            assert!(f.is_finite());
        }
    }

    #[test]
    fn test_volatility_adjusted_features() {
        let features: VolatilityAdjustedFeatures<20> = VolatilityAdjustedFeatures::new();
        
        for i in 0..30 {
            let price = 100.0 + (i as f64).sin() * 10.0;
            features.update(price);
        }
        
        let adjusted = features.get_adjusted_features(105.0);
        assert_eq!(adjusted.len(), 9); // 7 base + 2 volatility
        
        // Z-score and std_dev should be present
        let z_score = adjusted[7];
        let std_dev = adjusted[8];
        
        assert!(std_dev >= 0.0);
    }

    #[test]
    fn test_technical_indicator_trait() {
        let ema = EMA::new(10);
        
        assert!(!ema.is_ready());
        
        ema.update_price(100.0);
        assert!(ema.is_ready());
        assert_eq!(ema.get_value(), Some(100.0));
        
        ema.reset();
        assert!(!ema.is_ready());
    }
}
