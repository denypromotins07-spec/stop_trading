//! Volatility module root.
//! Integrates volatility metrics directly into the pre-trade risk bus.

pub mod atr;
pub mod garch;

pub use atr::{
    ATR, BollingerBandsWidth, BollingerBandsResult, VolatilityRegime, VolatilityRegimeType,
};
pub use garch::{
    Garch11, GarchParams, AdaptiveSpread, SpreadQuote, VolatilityForecast, VolatilityRegime as GarchVolRegime,
};

use std::sync::atomic::{AtomicF64, AtomicBool, Ordering};
use thiserror::Error;

/// Error types for volatility module
#[derive(Debug, Error)]
pub enum VolatilityError {
    #[error("ATR error: {0}")]
    Atr(#[from] atr::AtrError),
    #[error("GARCH error: {0}")]
    Garch(#[from] garch::GarchError),
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
}

/// Trait for volatility providers
pub trait VolatilityProvider {
    /// Get current volatility estimate
    fn get_volatility(&self) -> f64;
    
    /// Get volatility forecast
    fn get_forecast(&self) -> Option<f64>;
    
    /// Check if volatility is elevated
    fn is_elevated(&self, threshold: f64) -> bool;
    
    /// Reset the provider
    fn reset(&self);
}

impl VolatilityProvider for ATR {
    fn get_volatility(&self) -> f64 {
        self.get().unwrap_or(0.0)
    }
    
    fn get_forecast(&self) -> Option<f64> {
        // ATR doesn't forecast, return current value
        self.get()
    }
    
    fn is_elevated(&self, threshold: f64) -> bool {
        self.get().map(|atr| atr > threshold).unwrap_or(false)
    }
    
    fn reset(&self) {
        ATR::reset(self);
    }
}

impl VolatilityProvider for Garch11 {
    fn get_volatility(&self) -> f64 {
        self.get_volatility()
    }
    
    fn get_forecast(&self) -> Option<f64> {
        Some(self.predict_variance_ahead(1).sqrt())
    }
    
    fn is_elevated(&self, threshold: f64) -> bool {
        self.is_volatility_spike(threshold)
    }
    
    fn reset(&self) {
        Garch11::reset(self);
    }
}

/// Composite volatility estimator combining multiple methods
pub struct CompositeVolatility {
    atr: ATR,
    garch: Garch11,
    bb_width: BollingerBandsWidth<20>,
    weight_atr: AtomicF64,
    weight_garch: AtomicF64,
    initialized: AtomicBool,
}

impl CompositeVolatility {
    /// Create a new composite volatility estimator
    pub fn new(atr_period: usize, atr_weight: f64, garch_weight: f64) -> Result<Self, VolatilityError> {
        if (atr_weight + garch_weight - 1.0).abs() > 1e-6 {
            return Err(VolatilityError::InvalidConfig(
                "Weights must sum to 1.0".to_string()
            ));
        }
        
        Ok(Self {
            atr: ATR::new(atr_period)?,
            garch: Garch11::standard(),
            bb_width: BollingerBandsWidth::new(2.0),
            weight_atr: AtomicF64::new(atr_weight),
            weight_garch: AtomicF64::new(garch_weight),
            initialized: AtomicBool::new(false),
        })
    }

    /// Standard configuration (50% ATR, 50% GARCH)
    pub fn standard() -> Result<Self, VolatilityError> {
        Self::new(14, 0.5, 0.5)
    }

    /// Update with OHLC data
    pub fn update(&self, high: f64, low: f64, close: f64) -> CompositeVolatilityResult {
        let _ = self.atr.update(high, low, close);
        let _ = self.garch.update(close);
        let bb_result = self.bb_width.update(close);
        
        let atr_vol = self.atr.get().unwrap_or(0.0);
        let garch_vol = self.garch.get_volatility();
        
        let w_atr = self.weight_atr.load(Ordering::Relaxed);
        let w_garch = self.weight_garch.load(Ordering::Relaxed);
        
        let composite_vol = atr_vol * w_atr + garch_vol * w_garch;
        
        self.initialized.store(true, Ordering::Relaxed);
        
        CompositeVolatilityResult {
            composite_volatility: composite_vol,
            atr_volatility: atr_vol,
            garch_volatility: garch_vol,
            bb_bandwidth: bb_result.bandwidth,
            regime: self.detect_regime(composite_vol, bb_result.bandwidth),
        }
    }

    /// Detect combined volatility regime
    fn detect_regime(&self, vol: f64, bb_bandwidth: f64) -> CombinedRegime {
        let long_run_vol = self.garch.params().unconditional_variance().sqrt();
        
        let vol_ratio = if long_run_vol > 1e-10 { vol / long_run_vol } else { 1.0 };
        
        if vol_ratio > 2.0 || bb_bandwidth > 0.15 {
            CombinedRegime::High
        } else if vol_ratio < 0.7 && bb_bandwidth < 0.05 {
            CombinedRegime::Low
        } else {
            CombinedRegime::Normal
        }
    }

    /// Get composite result
    pub fn get(&self) -> Option<CompositeVolatilityResult> {
        if !self.initialized.load(Ordering::Relaxed) {
            return None;
        }
        
        let atr_vol = self.atr.get().unwrap_or(0.0);
        let garch_vol = self.garch.get_volatility();
        let bb_result = self.bb_width.get()?;
        
        let w_atr = self.weight_atr.load(Ordering::Relaxed);
        let w_garch = self.weight_garch.load(Ordering::Relaxed);
        
        let composite_vol = atr_vol * w_atr + garch_vol * w_garch;
        
        Some(CompositeVolatilityResult {
            composite_volatility: composite_vol,
            atr_volatility: atr_vol,
            garch_volatility: garch_vol,
            bb_bandwidth: bb_result.bandwidth,
            regime: self.detect_regime(composite_vol, bb_result.bandwidth),
        })
    }

    /// Get recommended position size multiplier
    pub fn position_multiplier(&self) -> f64 {
        self.get()
            .map(|r| r.regime.position_multiplier())
            .unwrap_or(1.0)
    }

    /// Get recommended stop loss multiplier
    pub fn stop_multiplier(&self) -> f64 {
        self.get()
            .map(|r| r.regime.stop_multiplier())
            .unwrap_or(2.0)
    }

    /// Reset the estimator
    pub fn reset(&self) {
        self.atr.reset();
        self.garch.reset();
        self.bb_width.reset();
        self.initialized.store(false, Ordering::Relaxed);
    }
}

/// Result from composite volatility estimation
#[derive(Debug, Clone, Copy)]
pub struct CompositeVolatilityResult {
    pub composite_volatility: f64,
    pub atr_volatility: f64,
    pub garch_volatility: f64,
    pub bb_bandwidth: f64,
    pub regime: CombinedRegime,
}

/// Combined volatility regime
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CombinedRegime {
    Low,
    Normal,
    High,
}

impl CombinedRegime {
    /// Position size multiplier based on regime
    pub fn position_multiplier(&self) -> f64 {
        match self {
            CombinedRegime::Low => 1.5,
            CombinedRegime::Normal => 1.0,
            CombinedRegime::High => 0.5,
        }
    }

    /// Stop loss multiplier based on regime
    pub fn stop_multiplier(&self) -> f64 {
        match self {
            CombinedRegime::Low => 1.5,
            CombinedRegime::Normal => 2.0,
            CombinedRegime::High => 3.0,
        }
    }

    /// Spread widening factor
    pub fn spread_factor(&self) -> f64 {
        match self {
            CombinedRegime::Low => 1.0,
            CombinedRegime::Normal => 1.5,
            CombinedRegime::High => 2.5,
        }
    }
}

/// Pre-trade risk integration for volatility
pub struct VolatilityRiskBus {
    composite: CompositeVolatility,
    max_volatility_threshold: AtomicF64,
    halted: AtomicBool,
}

impl VolatilityRiskBus {
    /// Create a new volatility risk bus
    pub fn new(max_vol_threshold: f64) -> Result<Self, VolatilityError> {
        Ok(Self {
            composite: CompositeVolatility::standard()?,
            max_volatility_threshold: AtomicF64::new(max_vol_threshold),
            halted: AtomicBool::new(false),
        })
    }

    /// Update with market data and check risk limits
    pub fn update(&self, high: f64, low: f64, close: f64) -> RiskAssessment {
        let vol_result = self.composite.update(high, low, close);
        
        let max_thresh = self.max_volatility_threshold.load(Ordering::Relaxed);
        let is_halted = vol_result.composite_volatility > max_thresh;
        
        self.halted.store(is_halted, Ordering::Relaxed);
        
        RiskAssessment {
            volatility: vol_result.composite_volatility,
            regime: vol_result.regime,
            trading_allowed: !is_halted,
            position_multiplier: vol_result.regime.position_multiplier(),
            stop_multiplier: vol_result.regime.stop_multiplier(),
            spread_factor: vol_result.regime.spread_factor(),
        }
    }

    /// Check if trading is allowed
    pub fn is_trading_allowed(&self) -> bool {
        !self.halted.load(Ordering::Relaxed)
    }

    /// Get current risk assessment
    pub fn get_assessment(&self) -> Option<RiskAssessment> {
        let vol_result = self.composite.get()?;
        let max_thresh = self.max_volatility_threshold.load(Ordering::Relaxed);
        let is_halted = vol_result.composite_volatility > max_thresh;
        
        Some(RiskAssessment {
            volatility: vol_result.composite_volatility,
            regime: vol_result.regime,
            trading_allowed: !is_halted,
            position_multiplier: vol_result.regime.position_multiplier(),
            stop_multiplier: vol_result.regime.stop_multiplier(),
            spread_factor: vol_result.regime.spread_factor(),
        })
    }

    /// Manually halt trading
    pub fn halt(&self) {
        self.halted.store(true, Ordering::Relaxed);
    }

    /// Resume trading
    pub fn resume(&self) {
        self.halted.store(false, Ordering::Relaxed);
    }

    /// Reset the risk bus
    pub fn reset(&self) {
        self.composite.reset();
        self.halted.store(false, Ordering::Relaxed);
    }
}

/// Risk assessment result
#[derive(Debug, Clone, Copy)]
pub struct RiskAssessment {
    pub volatility: f64,
    pub regime: CombinedRegime,
    pub trading_allowed: bool,
    pub position_multiplier: f64,
    pub stop_multiplier: f64,
    pub spread_factor: f64,
}

impl RiskAssessment {
    /// Calculate adjusted position size
    pub fn adjusted_position_size(&self, base_size: f64) -> f64 {
        base_size * self.position_multiplier
    }

    /// Calculate adjusted stop distance
    pub fn adjusted_stop_distance(&self, base_distance: f64) -> f64 {
        base_distance * self.stop_multiplier
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_composite_volatility() {
        let composite = CompositeVolatility::standard().unwrap();
        
        for i in 0..150 {
            let high = 105.0 + (i as f64).sin() * 3.0;
            let low = 95.0 + (i as f64).sin() * 2.0;
            let close = 100.0 + (i as f64).sin();
            composite.update(high, low, close);
        }
        
        let result = composite.get().unwrap();
        assert!(result.composite_volatility > 0.0);
        assert!(result.atr_volatility > 0.0);
        assert!(result.garch_volatility > 0.0);
    }

    #[test]
    fn test_volatility_risk_bus() {
        let bus = VolatilityRiskBus::new(0.5).unwrap();
        
        for i in 0..150 {
            let high = 105.0 + (i as f64).sin() * 3.0;
            let low = 95.0 + (i as f64).sin() * 2.0;
            let close = 100.0 + (i as f64).sin();
            bus.update(high, low, close);
        }
        
        let assessment = bus.get_assessment().unwrap();
        assert!(assessment.volatility > 0.0);
        assert!(assessment.position_multiplier > 0.0);
        assert!(assessment.stop_multiplier > 0.0);
    }

    #[test]
    fn test_regime_multipliers() {
        assert_eq!(CombinedRegime::Low.position_multiplier(), 1.5);
        assert_eq!(CombinedRegime::Normal.position_multiplier(), 1.0);
        assert_eq!(CombinedRegime::High.position_multiplier(), 0.5);
        
        assert_eq!(CombinedRegime::Low.spread_factor(), 1.0);
        assert_eq!(CombinedRegime::Normal.spread_factor(), 1.5);
        assert_eq!(CombinedRegime::High.spread_factor(), 2.5);
    }

    #[test]
    fn test_risk_assessment_adjustments() {
        let assessment = RiskAssessment {
            volatility: 0.02,
            regime: CombinedRegime::High,
            trading_allowed: true,
            position_multiplier: 0.5,
            stop_multiplier: 3.0,
            spread_factor: 2.5,
        };
        
        assert_eq!(assessment.adjusted_position_size(100.0), 50.0);
        assert_eq!(assessment.adjusted_stop_distance(1.0), 3.0);
    }
}
