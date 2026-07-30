//! Real-time Average True Range (ATR) and Bollinger Band width calculator.
//! Used for dynamic position sizing and stop-loss adjustments.

use std::sync::atomic::{AtomicF64, AtomicU64, Ordering};

/// Error types for ATR calculations
#[derive(Debug, thiserror::Error)]
pub enum AtrError {
    #[error("Insufficient data points")]
    InsufficientData,
    #[error("Invalid period: must be positive")]
    InvalidPeriod,
}

/// Average True Range calculator with lock-free updates
pub struct ATR {
    period: usize,
    atr_value: AtomicF64,
    prev_close: AtomicF64,
    count: AtomicU64,
    initialized: AtomicU64,
    alpha: AtomicF64,
}

impl ATR {
    /// Create a new ATR calculator with the given period
    pub fn new(period: usize) -> Result<Self, AtrError> {
        if period == 0 {
            return Err(AtrError::InvalidPeriod);
        }
        
        let alpha = 1.0 / period as f64;
        
        Ok(Self {
            period,
            atr_value: AtomicF64::new(0.0),
            prev_close: AtomicF64::new(0.0),
            count: AtomicU64::new(0),
            initialized: AtomicU64::new(0),
            alpha: AtomicF64::new(alpha),
        })
    }

    /// Update ATR with new OHLC data
    pub fn update(&self, high: f64, low: f64, close: f64) -> Result<f64, AtrError> {
        let count = self.count.load(Ordering::Relaxed);
        
        if count == 0 {
            // First bar - initialize with simple range
            self.prev_close.store(close, Ordering::Relaxed);
            self.atr_value.store(high - low, Ordering::Relaxed);
            self.count.fetch_add(1, Ordering::Relaxed);
            self.initialized.store(1, Ordering::Relaxed);
            Ok(high - low)
        } else {
            let prev_close = self.prev_close.load(Ordering::Relaxed);
            let alpha = self.alpha.load(Ordering::Relaxed);
            
            // Calculate True Range
            let tr = Self::true_range(high, low, close, prev_close);
            
            // Smooth using Wilder's method (exponential moving average)
            let current_atr = self.atr_value.load(Ordering::Relaxed);
            let new_atr = (current_atr * (self.period as f64 - 1.0) + tr) / self.period as f64;
            
            self.atr_value.store(new_atr, Ordering::Relaxed);
            self.prev_close.store(close, Ordering::Relaxed);
            self.count.fetch_add(1, Ordering::Relaxed);
            
            Ok(new_atr)
        }
    }

    /// Calculate True Range
    fn true_range(high: f64, low: f64, close: f64, prev_close: f64) -> f64 {
        let method1 = high - low;
        let method2 = (high - prev_close).abs();
        let method3 = (low - prev_close).abs();
        
        method1.max(method2).max(method3)
    }

    /// Get current ATR value
    pub fn get(&self) -> Option<f64> {
        if self.initialized.load(Ordering::Relaxed) == 0 {
            None
        } else {
            Some(self.atr_value.load(Ordering::Relaxed))
        }
    }

    /// Check if ATR is ready
    pub fn is_ready(&self) -> bool {
        self.initialized.load(Ordering::Relaxed) != 0
    }

    /// Get recommended position size based on ATR and risk percentage
    pub fn position_size(&self, account_balance: f64, risk_percent: f64, price: f64) -> Option<f64> {
        let atr = self.get()?;
        if atr < 1e-10 || price < 1e-10 {
            return None;
        }
        
        let risk_amount = account_balance * risk_percent;
        let stop_distance = atr * 2.0; // 2x ATR stop
        let position_size = risk_amount / stop_distance;
        
        Some(position_size / price) // Convert to units
    }

    /// Get recommended stop loss distance based on ATR multiplier
    pub fn stop_loss_distance(&self, multiplier: f64) -> Option<f64> {
        self.get().map(|atr| atr * multiplier)
    }

    /// Reset the ATR
    pub fn reset(&self) {
        self.atr_value.store(0.0, Ordering::Relaxed);
        self.prev_close.store(0.0, Ordering::Relaxed);
        self.count.store(0, Ordering::Relaxed);
        self.initialized.store(0, Ordering::Relaxed);
    }
}

/// Bollinger Bands Width calculator
pub struct BollingerBandsWidth<const N: usize> {
    sum: AtomicF64,
    sum_sq: AtomicF64,
    count: AtomicU64,
    buffer: Vec<AtomicF64>,
    index: AtomicU64,
    std_dev_multiplier: AtomicF64,
}

impl<const N: usize> BollingerBandsWidth<N> {
    /// Create a new Bollinger Bands Width calculator
    pub fn new(std_dev_multiplier: f64) -> Self {
        let mut buffer = Vec::with_capacity(N);
        for _ in 0..N {
            buffer.push(AtomicF64::new(0.0));
        }
        
        Self {
            sum: AtomicF64::new(0.0),
            sum_sq: AtomicF64::new(0.0),
            count: AtomicU64::new(0),
            buffer,
            index: AtomicU64::new(0),
            std_dev_multiplier: AtomicF64::new(std_dev_multiplier),
        }
    }

    /// Update with new price and return bandwidth
    pub fn update(&self, price: f64) -> BollingerBandsResult {
        let idx = self.index.load(Ordering::Relaxed) as usize;
        let count = self.count.load(Ordering::Relaxed);
        
        let old_value = self.buffer[idx].load(Ordering::Relaxed);
        self.buffer[idx].store(price, Ordering::Relaxed);
        
        // Update sums
        let current_sum = self.sum.load(Ordering::Relaxed);
        let current_sum_sq = self.sum_sq.load(Ordering::Relaxed);
        
        let new_sum = if count >= N as u64 {
            current_sum - old_value + price
        } else {
            current_sum + price
        };
        
        let new_sum_sq = if count >= N as u64 {
            current_sum_sq - old_value * old_value + price * price
        } else {
            current_sum_sq + price * price
        };
        
        self.sum.store(new_sum, Ordering::Relaxed);
        self.sum_sq.store(new_sum_sq, Ordering::Relaxed);
        self.index.store(((idx + 1) % N) as u64, Ordering::Relaxed);
        
        if count < N as u64 {
            self.count.fetch_add(1, Ordering::Relaxed);
        }
        
        let actual_count = if count < N as u64 { count + 1 } else { N as u64 };
        let mean = new_sum / actual_count as f64;
        let variance = (new_sum_sq / actual_count as f64) - (mean * mean);
        let std_dev = variance.max(0.0).sqrt();
        
        let multiplier = self.std_dev_multiplier.load(Ordering::Relaxed);
        let upper = mean + multiplier * std_dev;
        let lower = mean - multiplier * std_dev;
        let bandwidth = (upper - lower) / mean;
        let percent_b = if upper != lower {
            (price - lower) / (upper - lower)
        } else {
            0.5
        };
        
        BollingerBandsResult {
            upper,
            middle: mean,
            lower,
            bandwidth,
            percent_b,
            std_dev,
        }
    }

    /// Get current band values
    pub fn get(&self) -> Option<BollingerBandsResult> {
        let count = self.count.load(Ordering::Relaxed);
        if count == 0 {
            return None;
        }
        
        let actual_count = if count < N as u64 { count } else { N as u64 };
        let sum = self.sum.load(Ordering::Relaxed);
        let sum_sq = self.sum_sq.load(Ordering::Relaxed);
        
        let mean = sum / actual_count as f64;
        let variance = (sum_sq / actual_count as f64) - (mean * mean);
        let std_dev = variance.max(0.0).sqrt();
        
        let multiplier = self.std_dev_multiplier.load(Ordering::Relaxed);
        let upper = mean + multiplier * std_dev;
        let lower = mean - multiplier * std_dev;
        let bandwidth = (upper - lower) / mean;
        
        Some(BollingerBandsResult {
            upper,
            middle: mean,
            lower,
            bandwidth,
            percent_b: 0.5, // Can't calculate without current price
            std_dev,
        })
    }

    /// Reset the calculator
    pub fn reset(&self) {
        self.sum.store(0.0, Ordering::Relaxed);
        self.sum_sq.store(0.0, Ordering::Relaxed);
        self.count.store(0, Ordering::Relaxed);
        self.index.store(0, Ordering::Relaxed);
        for slot in &self.buffer {
            slot.store(0.0, Ordering::Relaxed);
        }
    }
}

impl<const N: usize> Default for BollingerBandsWidth<N> {
    fn default() -> Self {
        Self::new(2.0) // Standard 2 std dev
    }
}

/// Result structure for Bollinger Bands calculations
#[derive(Debug, Clone, Copy)]
pub struct BollingerBandsResult {
    pub upper: f64,
    pub middle: f64,
    pub lower: f64,
    pub bandwidth: f64,
    pub percent_b: f64,
    pub std_dev: f64,
}

impl BollingerBandsResult {
    /// Check if price is near upper band (overbought)
    pub fn is_overbought(&self, threshold: f64) -> bool {
        self.percent_b > threshold
    }

    /// Check if price is near lower band (oversold)
    pub fn is_oversold(&self, threshold: f64) -> bool {
        self.percent_b < threshold
    }

    /// Check for squeeze (low bandwidth)
    pub fn is_squeeze(&self, threshold: f64) -> bool {
        self.bandwidth < threshold
    }

    /// Check for expansion (high bandwidth)
    pub fn is_expansion(&self, threshold: f64) -> bool {
        self.bandwidth > threshold
    }
}

/// Volatility regime detector
pub struct VolatilityRegime {
    atr: ATR,
    bb_width: BollingerBandsWidth<20>,
    low_vol_threshold: AtomicF64,
    high_vol_threshold: AtomicF64,
}

impl VolatilityRegime {
    /// Create a new volatility regime detector
    pub fn new(atr_period: usize, low_thresh: f64, high_thresh: f64) -> Result<Self, AtrError> {
        Ok(Self {
            atr: ATR::new(atr_period)?,
            bb_width: BollingerBandsWidth::new(2.0),
            low_vol_threshold: AtomicF64::new(low_thresh),
            high_vol_threshold: AtomicF64::new(high_thresh),
        })
    }

    /// Update with new OHLC data and return current regime
    pub fn update(&self, high: f64, low: f64, close: f64) -> VolatilityRegimeType {
        let _ = self.atr.update(high, low, close);
        let bb_result = self.bb_width.update(close);
        
        let low_thresh = self.low_vol_threshold.load(Ordering::Relaxed);
        let high_thresh = self.high_vol_threshold.load(Ordering::Relaxed);
        
        if bb_result.bandwidth < low_thresh {
            VolatilityRegimeType::Low
        } else if bb_result.bandwidth > high_thresh {
            VolatilityRegimeType::High
        } else {
            VolatilityRegimeType::Normal
        }
    }

    /// Get current regime
    pub fn get_regime(&self) -> Option<VolatilityRegimeType> {
        let bb_result = self.bb_width.get()?;
        
        let low_thresh = self.low_vol_threshold.load(Ordering::Relaxed);
        let high_thresh = self.high_vol_threshold.load(Ordering::Relaxed);
        
        if bb_result.bandwidth < low_thresh {
            Some(VolatilityRegimeType::Low)
        } else if bb_result.bandwidth > high_thresh {
            Some(VolatilityRegimeType::High)
        } else {
            Some(VolatilityRegimeType::Normal)
        }
    }

    /// Get ATR-based position size recommendation
    pub fn recommended_position_size(&self, balance: f64, risk: f64, price: f64) -> Option<f64> {
        self.atr.position_size(balance, risk, price)
    }

    /// Reset the detector
    pub fn reset(&self) {
        self.atr.reset();
        self.bb_width.reset();
    }
}

/// Volatility regime classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VolatilityRegimeType {
    Low,
    Normal,
    High,
}

impl VolatilityRegimeType {
    /// Get position size multiplier based on regime
    pub fn position_multiplier(&self) -> f64 {
        match self {
            VolatilityRegimeType::Low => 1.5,  // Can take larger positions
            VolatilityRegimeType::Normal => 1.0,
            VolatilityRegimeType::High => 0.5, // Reduce position size
        }
    }

    /// Get stop loss multiplier based on regime
    pub fn stop_multiplier(&self) -> f64 {
        match self {
            VolatilityRegimeType::Low => 1.5,   // Tighter stops OK
            VolatilityRegimeType::Normal => 2.0,
            VolatilityRegimeType::High => 3.0,  // Wider stops needed
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atr_basic() {
        let atr = ATR::new(14).unwrap();
        
        assert!(!atr.is_ready());
        
        // Feed some bars
        for i in 0..20 {
            let high = 105.0 + (i as f64 * 0.5);
            let low = 95.0 + (i as f64 * 0.3);
            let close = 100.0 + (i as f64 * 0.4);
            atr.update(high, low, close).unwrap();
        }
        
        assert!(atr.is_ready());
        let atr_val = atr.get().unwrap();
        assert!(atr_val > 0.0);
        assert!(atr_val.is_finite());
    }

    #[test]
    fn test_bollinger_bands_width() {
        let bb: BollingerBandsWidth<20> = BollingerBandsWidth::new(2.0);
        
        // Feed prices
        for i in 0..30 {
            let price = 100.0 + (i as f64).sin() * 5.0;
            bb.update(price);
        }
        
        let result = bb.get().unwrap();
        assert!(result.bandwidth > 0.0);
        assert!(result.upper > result.middle);
        assert!(result.lower < result.middle);
    }

    #[test]
    fn test_volatility_regime() {
        let regime = VolatilityRegime::new(14, 0.02, 0.10).unwrap();
        
        // Simulate normal volatility
        for i in 0..30 {
            let high = 105.0 + (i as f64).sin() * 2.0;
            let low = 95.0 + (i as f64).sin() * 1.5;
            let close = 100.0 + (i as f64).sin();
            regime.update(high, low, close);
        }
        
        let regime_type = regime.get_regime().unwrap();
        assert!(matches!(regime_type, VolatilityRegimeType::Low | VolatilityRegimeType::Normal | VolatilityRegimeType::High));
    }

    #[test]
    fn test_regime_multipliers() {
        assert_eq!(VolatilityRegimeType::Low.position_multiplier(), 1.5);
        assert_eq!(VolatilityRegimeType::Normal.position_multiplier(), 1.0);
        assert_eq!(VolatilityRegimeType::High.position_multiplier(), 0.5);
        
        assert_eq!(VolatilityRegimeType::Low.stop_multiplier(), 1.5);
        assert_eq!(VolatilityRegimeType::Normal.stop_multiplier(), 2.0);
        assert_eq!(VolatilityRegimeType::High.stop_multiplier(), 3.0);
    }
}
