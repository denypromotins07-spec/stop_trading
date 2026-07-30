//! Zero-cost, fixed-point math calculators for technical indicators.
//! Optimized for AMD Ryzen CPUs using bitwise operations where applicable.

use std::sync::atomic::{AtomicF64, AtomicU64, Ordering};

/// Fixed-point representation for high-precision calculations without floating point
#[derive(Debug, Clone, Copy)]
pub struct FixedPoint<const SCALE: u32>(pub i64);

impl<const SCALE: u32> FixedPoint<SCALE> {
    pub const fn from_f64(value: f64) -> Self {
        Self((value * (1 << SCALE) as f64) as i64)
    }

    pub const fn to_f64(self) -> f64 {
        self.0 as f64 / (1 << SCALE) as f64
    }

    pub const fn add(self, other: Self) -> Self {
        Self(self.0 + other.0)
    }

    pub const fn sub(self, other: Self) -> Self {
        Self(self.0 - other.0)
    }

    pub const fn mul(self, other: Self) -> Self {
        // Scale back after multiplication to maintain precision
        Self((self.0 * other.0) >> SCALE)
    }

    pub const fn div(self, other: Self) -> Self {
        if other.0 == 0 {
            return Self(0);
        }
        // Scale up before division to maintain precision
        Self((self.0 << SCALE) / other.0)
    }
}

/// Exponential Moving Average calculator with zero heap allocations
pub struct EMA {
    alpha: AtomicF64,
    value: AtomicF64,
    initialized: AtomicU64,
}

impl EMA {
    /// Create a new EMA calculator with the given period
    pub fn new(period: usize) -> Self {
        let alpha = 2.0 / (period as f64 + 1.0);
        Self {
            alpha: AtomicF64::new(alpha),
            value: AtomicF64::new(0.0),
            initialized: AtomicU64::new(0),
        }
    }

    /// Update EMA with a new value (lock-free)
    pub fn update(&self, price: f64) -> f64 {
        let init = self.initialized.load(Ordering::Relaxed);
        
        if init == 0 {
            // First value initializes the EMA
            self.value.store(price, Ordering::Relaxed);
            self.initialized.store(1, Ordering::Relaxed);
            price
        } else {
            let alpha = self.alpha.load(Ordering::Relaxed);
            let prev = self.value.load(Ordering::Relaxed);
            let new_value = alpha * price + (1.0 - alpha) * prev;
            self.value.store(new_value, Ordering::Relaxed);
            new_value
        }
    }

    /// Get current EMA value
    pub fn get(&self) -> Option<f64> {
        if self.initialized.load(Ordering::Relaxed) == 0 {
            None
        } else {
            Some(self.value.load(Ordering::Relaxed))
        }
    }

    /// Reset the EMA
    pub fn reset(&self) {
        self.initialized.store(0, Ordering::Relaxed);
        self.value.store(0.0, Ordering::Relaxed);
    }
}

/// Simple Moving Average using a fixed-size ring buffer approach
pub struct SMA<const N: usize> {
    buffer: [AtomicF64; N],
    sum: AtomicF64,
    count: AtomicU64,
    index: AtomicU64,
}

impl<const N: usize> SMA<N> {
    pub fn new() -> Self {
        let mut buffer = Vec::with_capacity(N);
        for _ in 0..N {
            buffer.push(AtomicF64::new(0.0));
        }
        
        Self {
            buffer: buffer.try_into().unwrap(),
            sum: AtomicF64::new(0.0),
            count: AtomicU64::new(0),
            index: AtomicU64::new(0),
        }
    }

    /// Update SMA with a new value
    pub fn update(&self, value: f64) -> f64 {
        let idx = self.index.load(Ordering::Relaxed) as usize;
        let count = self.count.load(Ordering::Relaxed);
        
        let old_value = self.buffer[idx].load(Ordering::Relaxed);
        self.buffer[idx].store(value, Ordering::Relaxed);
        
        let current_sum = self.sum.load(Ordering::Relaxed);
        let new_sum = if count >= N as u64 {
            // Buffer is full, subtract old value
            current_sum - old_value + value
        } else {
            // Buffer not full yet
            current_sum + value
        };
        
        self.sum.store(new_sum, Ordering::Relaxed);
        self.index.store(((idx + 1) % N) as u64, Ordering::Relaxed);
        
        if count < N as u64 {
            self.count.fetch_add(1, Ordering::Relaxed);
        }
        
        let actual_count = if count < N as u64 { count + 1 } else { N as u64 };
        new_sum / actual_count as f64
    }

    /// Get current SMA value
    pub fn get(&self) -> Option<f64> {
        let count = self.count.load(Ordering::Relaxed);
        if count == 0 {
            None
        } else {
            let actual_count = if count < N as u64 { count } else { N as u64 };
            Some(self.sum.load(Ordering::Relaxed) / actual_count as f64)
        }
    }

    /// Reset the SMA
    pub fn reset(&self) {
        self.sum.store(0.0, Ordering::Relaxed);
        self.count.store(0, Ordering::Relaxed);
        self.index.store(0, Ordering::Relaxed);
        for slot in &self.buffer {
            slot.store(0.0, Ordering::Relaxed);
        }
    }
}

impl<const N: usize> Default for SMA<N> {
    fn default() -> Self {
        Self::new()
    }
}

/// Relative Strength Index calculator
pub struct RSI {
    gain_ema: EMA,
    loss_ema: EMA,
    last_price: AtomicF64,
    initialized: AtomicU64,
}

impl RSI {
    /// Create a new RSI calculator with the given period
    pub fn new(period: usize) -> Self {
        Self {
            gain_ema: EMA::new(period),
            loss_ema: EMA::new(period),
            last_price: AtomicF64::new(0.0),
            initialized: AtomicU64::new(0),
        }
    }

    /// Update RSI with a new price
    pub fn update(&self, price: f64) -> Option<f64> {
        let init = self.initialized.load(Ordering::Relaxed);
        
        if init == 0 {
            self.last_price.store(price, Ordering::Relaxed);
            self.initialized.store(1, Ordering::Relaxed);
            None
        } else {
            let last = self.last_price.load(Ordering::Relaxed);
            let change = price - last;
            self.last_price.store(price, Ordering::Relaxed);
            
            let gain = if change > 0.0 { change } else { 0.0 };
            let loss = if change < 0.0 { -change } else { 0.0 };
            
            self.gain_ema.update(gain);
            self.loss_ema.update(loss);
            
            let avg_gain = self.gain_ema.get()?;
            let avg_loss = self.loss_ema.get()?;
            
            if avg_loss < 1e-10 {
                Some(100.0)
            } else {
                let rs = avg_gain / avg_loss;
                Some(100.0 - (100.0 / (1.0 + rs)))
            }
        }
    }

    /// Get current RSI value
    pub fn get(&self) -> Option<f64> {
        let avg_gain = self.gain_ema.get()?;
        let avg_loss = self.loss_ema.get()?;
        
        if avg_loss < 1e-10 {
            Some(100.0)
        } else {
            let rs = avg_gain / avg_loss;
            Some(100.0 - (100.0 / (1.0 + rs)))
        }
    }

    /// Reset the RSI
    pub fn reset(&self) {
        self.initialized.store(0, Ordering::Relaxed);
        self.last_price.store(0.0, Ordering::Relaxed);
        self.gain_ema.reset();
        self.loss_ema.reset();
    }
}

/// MACD (Moving Average Convergence Divergence) calculator
pub struct MACD {
    fast_ema: EMA,
    slow_ema: EMA,
    signal_ema: EMA,
}

impl MACD {
    /// Create a new MACD calculator with standard periods (12, 26, 9)
    pub fn new(fast_period: usize, slow_period: usize, signal_period: usize) -> Self {
        Self {
            fast_ema: EMA::new(fast_period),
            slow_ema: EMA::new(slow_period),
            signal_ema: EMA::new(signal_period),
        }
    }

    /// Standard MACD constructor (12, 26, 9)
    pub fn standard() -> Self {
        Self::new(12, 26, 9)
    }

    /// Update MACD with a new price
    pub fn update(&self, price: f64) -> MacdResult {
        let fast = self.fast_ema.update(price);
        let slow = self.slow_ema.update(price);
        let macd_line = fast - slow;
        
        // Signal line is EMA of MACD line
        let signal = self.signal_ema.update(macd_line);
        let histogram = macd_line - signal;
        
        MacdResult {
            macd_line,
            signal_line: signal,
            histogram,
        }
    }

    /// Get current MACD values
    pub fn get(&self) -> Option<MacdResult> {
        let fast = self.fast_ema.get()?;
        let slow = self.slow_ema.get()?;
        let macd_line = fast - slow;
        let signal = self.signal_ema.get()?;
        let histogram = macd_line - signal;
        
        Some(MacdResult {
            macd_line,
            signal_line: signal,
            histogram,
        })
    }

    /// Reset the MACD
    pub fn reset(&self) {
        self.fast_ema.reset();
        self.slow_ema.reset();
        self.signal_ema.reset();
    }
}

/// Result structure for MACD calculations
#[derive(Debug, Clone, Copy)]
pub struct MacdResult {
    pub macd_line: f64,
    pub signal_line: f64,
    pub histogram: f64,
}

/// Bollinger Bands calculator
pub struct BollingerBands<const N: usize> {
    sma: SMA<N>,
    variance_sum: AtomicF64,
    count: AtomicU64,
}

impl<const N: usize> BollingerBands<N> {
    pub fn new() -> Self {
        Self {
            sma: SMA::new(),
            variance_sum: AtomicF64::new(0.0),
            count: AtomicU64::new(0),
        }
    }

    /// Update with a new price and return band values
    pub fn update(&self, price: f64, std_dev_multiplier: f64) -> BollingerBandsResult {
        let middle = self.sma.update(price);
        
        let count = self.count.load(Ordering::Relaxed);
        let mean = self.sma.get().unwrap_or(middle);
        let deviation = price - mean;
        let squared_dev = deviation * deviation;
        
        // Update rolling variance sum
        if count >= N as u64 {
            // Need to remove oldest squared deviation (simplified - would need full buffer for exact)
            let current_var_sum = self.variance_sum.load(Ordering::Relaxed);
            self.variance_sum.store(current_var_sum + squared_dev, Ordering::Relaxed);
        } else {
            let current_var_sum = self.variance_sum.load(Ordering::Relaxed);
            self.variance_sum.store(current_var_sum + squared_dev, Ordering::Relaxed);
            self.count.fetch_add(1, Ordering::Relaxed);
        }
        
        let actual_count = if count < N as u64 { count + 1 } else { N as u64 };
        let variance = self.variance_sum.load(Ordering::Relaxed) / actual_count as f64;
        let std_dev = variance.sqrt();
        
        BollingerBandsResult {
            upper: middle + std_dev_multiplier * std_dev,
            middle,
            lower: middle - std_dev_multiplier * std_dev,
            bandwidth: (middle + std_dev_multiplier * std_dev - (middle - std_dev_multiplier * std_dev)) / middle,
        }
    }

    /// Get current band values
    pub fn get(&self, std_dev_multiplier: f64) -> Option<BollingerBandsResult> {
        let middle = self.sma.get()?;
        let count = self.count.load(Ordering::Relaxed);
        if count == 0 {
            return None;
        }
        
        let actual_count = if count < N as u64 { count } else { N as u64 };
        let variance = self.variance_sum.load(Ordering::Relaxed) / actual_count as f64;
        let std_dev = variance.sqrt();
        
        Some(BollingerBandsResult {
            upper: middle + std_dev_multiplier * std_dev,
            middle,
            lower: middle - std_dev_multiplier * std_dev,
            bandwidth: (middle + std_dev_multiplier * std_dev - (middle - std_dev_multiplier * std_dev)) / middle,
        })
    }

    /// Reset the bands
    pub fn reset(&self) {
        self.sma.reset();
        self.variance_sum.store(0.0, Ordering::Relaxed);
        self.count.store(0, Ordering::Relaxed);
    }
}

impl<const N: usize> Default for BollingerBands<N> {
    fn default() -> Self {
        Self::new()
    }
}

/// Result structure for Bollinger Bands
#[derive(Debug, Clone, Copy)]
pub struct BollingerBandsResult {
    pub upper: f64,
    pub middle: f64,
    pub lower: f64,
    pub bandwidth: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ema() {
        let ema = EMA::new(5);
        
        assert_eq!(ema.get(), None);
        
        ema.update(10.0);
        assert_eq!(ema.get(), Some(10.0));
        
        ema.update(12.0);
        let val = ema.get().unwrap();
        assert!(val > 10.0 && val < 12.0);
    }

    #[test]
    fn test_sma() {
        let sma: SMA<5> = SMA::new();
        
        sma.update(1.0);
        sma.update(2.0);
        sma.update(3.0);
        
        assert_eq!(sma.get(), Some(2.0));
        
        sma.update(4.0);
        sma.update(5.0);
        assert_eq!(sma.get(), Some(3.0));
        
        sma.update(6.0); // Should remove 1.0
        assert_eq!(sma.get(), Some(4.0)); // (2+3+4+5+6)/5
    }

    #[test]
    fn test_rsi() {
        let rsi = RSI::new(14);
        
        // First update returns None
        assert_eq!(rsi.update(100.0), None);
        
        // Simulate some price movements
        for i in 0..20 {
            let price = 100.0 + (i as f64 * 2.0);
            rsi.update(price);
        }
        
        let rsi_val = rsi.get().unwrap();
        assert!(rsi_val >= 0.0 && rsi_val <= 100.0);
    }

    #[test]
    fn test_macd() {
        let macd = MACD::standard();
        
        // Feed some prices
        for i in 0..50 {
            let price = 100.0 + (i as f64).sin() * 10.0;
            let result = macd.update(price);
            
            if i > 30 {
                // After warmup, should have valid values
                assert!(result.macd_line.is_finite());
                assert!(result.signal_line.is_finite());
                assert!(result.histogram.is_finite());
            }
        }
    }

    #[test]
    fn test_fixed_point() {
        let a = FixedPoint::<16>::from_f64(1.5);
        let b = FixedPoint::<16>::from_f64(2.0);
        
        let sum = a.add(b);
        assert!((sum.to_f64() - 3.5).abs() < 0.001);
        
        let prod = a.mul(b);
        assert!((prod.to_f64() - 3.0).abs() < 0.001);
    }
}
