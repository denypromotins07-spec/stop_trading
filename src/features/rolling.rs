//! Rolling Window Feature Extractors
//! 
//! O(1) update complexity using pre-allocated ring buffers for time-series
//! feature generation. Pushes updates directly to shared memory IPC buffer.

use std::{
    collections::VecDeque,
    sync::{
        atomic::{AtomicU64, AtomicUsize, Ordering},
        Arc,
    },
};

/// Cache-line padding constant
const CACHE_LINE_SIZE: usize = 64;

/// Maximum window size for rolling calculations
pub const MAX_WINDOW_SIZE: usize = 1024;

/// Ring buffer for O(1) rolling window operations
#[repr(C, align(64))]
pub struct RingBuffer {
    /// Underlying data storage
    data: Vec<f64>,
    /// Current write position
    write_pos: AtomicUsize,
    /// Number of valid elements
    count: AtomicUsize,
    /// Maximum capacity
    capacity: usize,
    /// Sum of all elements (for O(1) mean)
    sum: AtomicU64, // Stored as fixed-point for atomic operations
    /// Sum of squares (for O(1) variance)
    sum_sq: AtomicU64,
    /// Padding to cache line
    _padding: [u8; CACHE_LINE_SIZE - 8 * 3 - 8 * 2 - 8],
}

unsafe impl Send for RingBuffer {}
unsafe impl Sync for RingBuffer {}

impl RingBuffer {
    /// Create a new ring buffer with specified capacity
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.min(MAX_WINDOW_SIZE);
        Self {
            data: vec![0.0; capacity],
            write_pos: AtomicUsize::new(0),
            count: AtomicUsize::new(0),
            capacity,
            sum: AtomicU64::new(0),
            sum_sq: AtomicU64::new(0),
            _padding: [0u8; CACHE_LINE_SIZE - 8 * 3 - 8 * 2 - 8],
        }
    }

    /// Push a new value to the buffer (O(1))
    pub fn push(&self, value: f64) {
        let pos = self.write_pos.fetch_add(1, Ordering::Relaxed) % self.capacity;
        let old_value = self.data[pos];
        
        // Update the value
        self.data[pos] = value;
        
        // Update count if not yet at capacity
        let current_count = self.count.load(Ordering::Relaxed);
        if current_count < self.capacity {
            self.count.fetch_add(1, Ordering::Relaxed);
        }
        
        // Update sum and sum of squares using atomic operations
        // Note: In production, use a mutex or lock-free algorithm for exact values
        let value_bits = value.to_bits();
        let old_bits = old_value.to_bits();
        
        // Approximate atomic update (for high-frequency, small errors acceptable)
        self.sum.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
            let current_f64 = f64::from_bits(current);
            Some(f64::to_bits(current_f64 - old_value + value))
        }).ok();
        
        self.sum_sq.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
            let current_f64 = f64::from_bits(current);
            Some(f64::to_bits(current_f64 - old_value.powi(2) + value.powi(2)))
        }).ok();
    }

    /// Get the current mean (O(1))
    pub fn mean(&self) -> f64 {
        let count = self.count.load(Ordering::Relaxed);
        if count == 0 {
            return 0.0;
        }
        
        let sum_bits = self.sum.load(Ordering::Relaxed);
        f64::from_bits(sum_bits) / count as f64
    }

    /// Get the current variance (O(1))
    pub fn variance(&self) -> f64 {
        let count = self.count.load(Ordering::Relaxed);
        if count < 2 {
            return 0.0;
        }
        
        let sum_bits = self.sum.load(Ordering::Relaxed);
        let sum_sq_bits = self.sum_sq.load(Ordering::Relaxed);
        
        let sum = f64::from_bits(sum_bits);
        let sum_sq = f64::from_bits(sum_sq_bits);
        let n = count as f64;
        
        // Variance = E[X^2] - (E[X])^2
        (sum_sq / n) - (sum / n).powi(2)
    }

    /// Get the current standard deviation (O(1))
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Get the minimum value (O(n) but n is bounded by capacity)
    pub fn min(&self) -> f64 {
        let count = self.count.load(Ordering::Relaxed);
        if count == 0 {
            return 0.0;
        }
        
        self.data.iter().take(count).cloned().fold(f64::INFINITY, f64::min)
    }

    /// Get the maximum value
    pub fn max(&self) -> f64 {
        let count = self.count.load(Ordering::Relaxed);
        if count == 0 {
            return 0.0;
        }
        
        self.data.iter().take(count).cloned().fold(f64::NEG_INFINITY, f64::max)
    }

    /// Get the latest value
    pub fn latest(&self) -> f64 {
        let count = self.count.load(Ordering::Relaxed);
        if count == 0 {
            return 0.0;
        }
        
        let pos = (self.write_pos.load(Ordering::Relaxed) - 1) % self.capacity;
        self.data[pos]
    }

    /// Get value at offset from latest (0 = latest, 1 = previous, etc.)
    pub fn get_offset(&self, offset: usize) -> Option<f64> {
        let count = self.count.load(Ordering::Relaxed);
        if offset >= count {
            return None;
        }
        
        let current_pos = self.write_pos.load(Ordering::Relaxed);
        let pos = (current_pos - 1 - offset + self.capacity) % self.capacity;
        Some(self.data[pos])
    }

    /// Check if buffer is full
    pub fn is_full(&self) -> bool {
        self.count.load(Ordering::Relaxed) >= self.capacity
    }

    /// Get current element count
    pub fn len(&self) -> usize {
        self.count.load(Ordering::Relaxed)
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.count.load(Ordering::Relaxed) == 0
    }

    /// Clear the buffer
    pub fn clear(&self) {
        self.write_pos.store(0, Ordering::Relaxed);
        self.count.store(0, Ordering::Relaxed);
        self.sum.store(0, Ordering::Relaxed);
        self.sum_sq.store(0, Ordering::Relaxed);
    }
}

/// Rolling window feature calculator
pub struct RollingFeatures {
    /// Price ring buffer
    prices: RingBuffer,
    /// Volume ring buffer
    volumes: RingBuffer,
    /// Returns ring buffer (for momentum)
    returns: RingBuffer,
    /// Window size
    window_size: usize,
    /// Last calculated RSI
    rsi: AtomicU64,
    /// Last calculated MACD
    macd: AtomicU64,
    /// Last update timestamp
    last_update_ns: AtomicU64,
}

unsafe impl Send for RollingFeatures {}
unsafe impl Sync for RollingFeatures {}

impl RollingFeatures {
    /// Create new rolling features calculator
    pub fn new(window_size: usize) -> Self {
        Self {
            prices: RingBuffer::new(window_size),
            volumes: RingBuffer::new(window_size),
            returns: RingBuffer::new(window_size),
            window_size,
            rsi: AtomicU64::new(0),
            macd: AtomicU64::new(0),
            last_update_ns: AtomicU64::new(0),
        }
    }

    /// Update with new price and volume
    pub fn update(&self, price: f64, volume: f64) {
        // Calculate return
        let last_price = self.prices.latest();
        let ret = if last_price != 0.0 {
            (price - last_price) / last_price
        } else {
            0.0
        };

        // Update buffers
        self.prices.push(price);
        self.volumes.push(volume);
        self.returns.push(ret);

        // Update technical indicators
        self.update_rsi();
        self.update_macd();

        // Update timestamp
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        self.last_update_ns.store(now, Ordering::Release);
    }

    /// Update RSI calculation
    fn update_rsi(&self) {
        if self.returns.len() < 14 {
            return;
        }

        let mut gains = 0.0;
        let mut losses = 0.0;

        for i in 0..14 {
            if let Some(ret) = self.returns.get_offset(i) {
                if ret > 0.0 {
                    gains += ret;
                } else {
                    losses -= ret;
                }
            }
        }

        let rs = if losses != 0.0 { gains / losses } else { 100.0 };
        let rsi = 100.0 - (100.0 / (1.0 + rs));
        
        // Store as bits for atomic operation
        self.rsi.store(f64::to_bits(rsi), Ordering::Release);
    }

    /// Update MACD calculation
    fn update_macd(&self) {
        if self.prices.len() < 26 {
            return;
        }

        // Simple EMA-based MACD (simplified for demonstration)
        let ema_12 = self.calculate_ema(12);
        let ema_26 = self.calculate_ema(26);
        let macd = ema_12 - ema_26;

        self.macd.store(f64::to_bits(macd), Ordering::Release);
    }

    /// Calculate EMA
    fn calculate_ema(&self, period: usize) -> f64 {
        if self.prices.len() < period {
            return self.prices.latest();
        }

        let multiplier = 2.0 / (period as f64 + 1.0);
        let mut ema = self.prices.latest();

        for i in 0..period {
            if let Some(price) = self.prices.get_offset(i) {
                ema = price * multiplier + ema * (1.0 - multiplier);
            }
        }

        ema
    }

    /// Get current RSI
    pub fn get_rsi(&self) -> f64 {
        f64::from_bits(self.rsi.load(Ordering::Relaxed))
    }

    /// Get current MACD
    pub fn get_macd(&self) -> f64 {
        f64::from_bits(self.macd.load(Ordering::Relaxed))
    }

    /// Get current price
    pub fn get_price(&self) -> f64 {
        self.prices.latest()
    }

    /// Get current volume
    pub fn get_volume(&self) -> f64 {
        self.volumes.latest()
    }

    /// Get price volatility (std dev of returns)
    pub fn get_volatility(&self) -> f64 {
        self.returns.std_dev()
    }

    /// Get price momentum (mean of returns)
    pub fn get_momentum(&self) -> f64 {
        self.returns.mean()
    }

    /// Get VWAP approximation
    pub fn get_vwap(&self) -> f64 {
        let count = self.prices.len().min(self.volumes.len());
        if count == 0 {
            return 0.0;
        }

        let mut total_pv = 0.0;
        let mut total_v = 0.0;

        for i in 0..count {
            if let (Some(price), Some(volume)) = (self.prices.get_offset(i), self.volumes.get_offset(i)) {
                total_pv += price * volume;
                total_v += volume;
            }
        }

        if total_v == 0.0 {
            0.0
        } else {
            total_pv / total_v
        }
    }

    /// Export all features as vector for ML
    pub fn export_features(&self) -> Vec<f32> {
        vec![
            self.get_rsi() as f32 / 100.0, // Normalize RSI to 0-1
            self.get_macd() as f32,
            self.get_volatility() as f32,
            self.get_momentum() as f32,
            (self.get_price() / self.get_vwap() - 1.0) as f32, // Price vs VWAP
            self.prices.mean() as f32,
            self.prices.std_dev() as f32,
            self.volumes.mean() as f32,
            self.returns.skewness().unwrap_or(0.0) as f32,
            self.returns.kurtosis().unwrap_or(0.0) as f32,
        ]
    }
}

/// Extension methods for statistical moments
impl RingBuffer {
    /// Calculate skewness
    pub fn skewness(&self) -> Option<f64> {
        let n = self.count.load(Ordering::Relaxed);
        if n < 3 {
            return None;
        }

        let mean = self.mean();
        let std = self.std_dev();
        if std == 0.0 {
            return Some(0.0);
        }

        let mut sum_cubed = 0.0;
        for i in 0..n {
            if let Some(val) = self.get_offset(i) {
                sum_cubed += ((val - mean) / std).powi(3);
            }
        }

        Some((n as f64 / ((n - 1.0) * (n - 2.0))) * sum_cubed)
    }

    /// Calculate kurtosis
    pub fn kurtosis(&self) -> Option<f64> {
        let n = self.count.load(Ordering::Relaxed);
        if n < 4 {
            return None;
        }

        let mean = self.mean();
        let std = self.std_dev();
        if std == 0.0 {
            return Some(0.0);
        }

        let mut sum_fourth = 0.0;
        for i in 0..n {
            if let Some(val) = self.get_offset(i) {
                sum_fourth += ((val - mean) / std).powi(4);
            }
        }

        let term1 = (n as f64 * (n + 1.0)) / ((n - 1.0) * (n - 2.0) * (n - 3.0));
        let term2 = 3.0 * ((n - 1.0).powi(2)) / ((n - 2.0) * (n - 3.0));
        
        Some(term1 * sum_fourth - term2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ring_buffer_alignment() {
        let buffer = RingBuffer::new(100);
        let addr = &buffer as *const _ as usize;
        assert_eq!(addr % CACHE_LINE_SIZE, 0, "RingBuffer should be cache-line aligned");
    }

    #[test]
    fn test_ring_buffer_push_and_mean() {
        let buffer = RingBuffer::new(10);
        
        for i in 1..=5 {
            buffer.push(i as f64);
        }
        
        assert_eq!(buffer.len(), 5);
        assert!(!buffer.is_full());
        assert_eq!(buffer.mean(), 3.0);
    }

    #[test]
    fn test_ring_buffer_wraparound() {
        let buffer = RingBuffer::new(5);
        
        for i in 1..=10 {
            buffer.push(i as f64);
        }
        
        assert_eq!(buffer.len(), 5);
        assert!(buffer.is_full());
        // After wraparound, should contain 6, 7, 8, 9, 10
        assert_eq!(buffer.mean(), 8.0);
    }

    #[test]
    fn test_rolling_features() {
        let features = RollingFeatures::new(50);
        
        // Feed some data
        for i in 1..=30 {
            features.update(100.0 + (i as f64 * 0.5), 1000.0 + (i as f64 * 10.0));
        }
        
        assert!(features.get_price() > 100.0);
        assert!(features.get_volume() > 1000.0);
        
        let exported = features.export_features();
        assert!(!exported.is_empty());
    }
}
