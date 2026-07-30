//! ML Feature Extractor
//! 
//! Extracts high-frequency alpha features into contiguous memory blocks
//! with strict alignment for zero-copy transfer to Python/Nautilus/Ray backend.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// Feature vector aligned for zero-copy IPC
/// 
/// # Safety
/// This struct uses #[repr(C)] to ensure memory layout matches Python's numpy expectations.
/// All fields are explicitly sized and padded for 8-byte alignment.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FeatureVector {
    /// Timestamp in nanoseconds
    pub timestamp_ns: u64,
    /// Symbol identifier (hashed)
    pub symbol_hash: u64,
    
    // Order Flow Features (8 features)
    pub order_flow_imbalance: f64,
    pub buy_volume_pressure: f64,
    pub sell_volume_pressure: f64,
    pub trade_sign_autocorr: f64,
    pub order_arrival_rate: f64,
    pub cancel_ratio: f64,
    pub modify_ratio: f64,
    pub aggressor_ratio: f64,
    
    // Price Features (8 features)
    pub microprice: f64,
    pub microprice_drift_1ms: f64,
    pub microprice_drift_10ms: f64,
    pub mid_price: f64,
    pub spread_bps: f64,
    pub price_momentum_100ms: f64,
    pub price_momentum_1s: f64,
    pub vwap_deviation: f64,
    
    // Book Features (8 features)
    pub bid_depth_imbalance: f64,
    pub ask_depth_imbalance: f64,
    pub book_slope_bid: f64,
    pub book_slope_ask: f64,
    pub liquidity_concentration: f64,
    pub queue_position_estimate: f64,
    pub hidden_liquidity_ratio: f64,
    pub book_resilience: f64,
    
    // SMC (Smart Money Concept) Features (8 features)
    pub smc_block_signal: f64,
    pub smc_order_block_hit: f64,
    pub smc_fair_value_gap: f64,
    pub smc_liquidity_sweep: f64,
    pub smc_breaker_level: f64,
    pub smc_premium_discount: f64,
    pub smc_market_structure_shift: f64,
    pub smc_change_of_character: f64,
    
    // Volatility Features (8 features)
    pub realized_vol_10ms: f64,
    pub realized_vol_100ms: f64,
    pub implied_vol_estimate: f64,
    pub vol_of_vol: f64,
    pub jump_intensity: f64,
    pub tail_risk_indicator: f64,
    pub kurtosis_estimate: f64,
    pub skew_estimate: f64,
    
    // Padding for future expansion (ensures fixed size)
    _padding: [u64; 16],
}

impl FeatureVector {
    /// Size of feature vector in bytes
    pub const SIZE_BYTES: usize = std::mem::size_of::<FeatureVector>();
    
    /// Number of features (excluding timestamp, symbol_hash, and padding)
    pub const FEATURE_COUNT: usize = 40;
    
    /// Create a new zero-initialized feature vector
    pub fn new() -> Self {
        Self {
            timestamp_ns: 0,
            symbol_hash: 0,
            order_flow_imbalance: 0.0,
            buy_volume_pressure: 0.0,
            sell_volume_pressure: 0.0,
            trade_sign_autocorr: 0.0,
            order_arrival_rate: 0.0,
            cancel_ratio: 0.0,
            modify_ratio: 0.0,
            aggressor_ratio: 0.0,
            microprice: 0.0,
            microprice_drift_1ms: 0.0,
            microprice_drift_10ms: 0.0,
            mid_price: 0.0,
            spread_bps: 0.0,
            price_momentum_100ms: 0.0,
            price_momentum_1s: 0.0,
            vwap_deviation: 0.0,
            bid_depth_imbalance: 0.0,
            ask_depth_imbalance: 0.0,
            book_slope_bid: 0.0,
            book_slope_ask: 0.0,
            liquidity_concentration: 0.0,
            queue_position_estimate: 0.0,
            hidden_liquidity_ratio: 0.0,
            book_resilience: 0.0,
            smc_block_signal: 0.0,
            smc_order_block_hit: 0.0,
            smc_fair_value_gap: 0.0,
            smc_liquidity_sweep: 0.0,
            smc_breaker_level: 0.0,
            smc_premium_discount: 0.0,
            smc_market_structure_shift: 0.0,
            smc_change_of_character: 0.0,
            realized_vol_10ms: 0.0,
            realized_vol_100ms: 0.0,
            implied_vol_estimate: 0.0,
            vol_of_vol: 0.0,
            jump_intensity: 0.0,
            tail_risk_indicator: 0.0,
            kurtosis_estimate: 0.0,
            skew_estimate: 0.0,
            _padding: [0u64; 16],
        }
    }
    
    /// Get feature as slice for zero-copy operations
    pub fn as_slice(&self) -> &[f64] {
        // Safety: FeatureVector is repr(C) and all f64 fields are contiguous
        unsafe {
            std::slice::from_raw_parts(
                &self.order_flow_imbalance as *const f64,
                Self::FEATURE_COUNT,
            )
        }
    }
    
    /// Get mutable feature as slice
    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        unsafe {
            std::slice::from_raw_parts_mut(
                &mut self.order_flow_imbalance as *mut f64,
                Self::FEATURE_COUNT,
            )
        }
    }
    
    /// Convert to bytes for IPC
    pub fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                self as *const FeatureVector as *const u8,
                Self::SIZE_BYTES,
            )
        }
    }
}

impl Default for FeatureVector {
    fn default() -> Self {
        Self::new()
    }
}

/// Market data snapshot for feature extraction
#[derive(Debug, Clone)]
pub struct MarketSnapshot {
    pub symbol: String,
    pub timestamp_ns: u64,
    pub bid_price: f64,
    pub ask_price: f64,
    pub bid_size: u64,
    pub ask_size: u64,
    pub last_trade_price: f64,
    pub last_trade_size: u64,
    pub last_trade_aggressor: bool, // true = buy, false = sell
    pub total_bid_depth: u64,
    pub total_ask_depth: u64,
}

/// Circular buffer for rolling calculations
struct RollingBuffer {
    values: Vec<f64>,
    head: usize,
    count: usize,
    capacity: usize,
    sum: f64,
}

impl RollingBuffer {
    fn new(capacity: usize) -> Self {
        Self {
            values: vec![0.0; capacity],
            head: 0,
            count: 0,
            capacity,
            sum: 0.0,
        }
    }

    fn push(&mut self, value: f64) {
        if self.count == self.capacity {
            // Remove oldest
            let old_idx = self.head;
            self.sum -= self.values[old_idx];
        }
        
        self.values[self.head] = value;
        self.sum += value;
        self.head = (self.head + 1) % self.capacity;
        
        if self.count < self.capacity {
            self.count += 1;
        }
    }

    fn mean(&self) -> f64 {
        if self.count == 0 { return 0.0; }
        self.sum / self.count as f64
    }

    fn variance(&self) -> f64 {
        if self.count < 2 { return 0.0; }
        let mean = self.mean();
        let mut sum_sq = 0.0;
        for i in 0..self.count {
            let idx = (self.head + self.capacity - self.count + i) % self.capacity;
            let diff = self.values[idx] - mean;
            sum_sq += diff * diff;
        }
        sum_sq / (self.count - 1) as f64
    }

    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }
}

/// High-frequency feature extractor
pub struct FeatureExtractor {
    /// Rolling buffers for different time windows
    trade_signs: RollingBuffer,
    trade_sizes: RollingBuffer,
    price_changes_1ms: RollingBuffer,
    price_changes_10ms: RollingBuffer,
    price_changes_100ms: RollingBuffer,
    /// Last snapshot per symbol
    last_snapshot: dashmap::DashMap<String, MarketSnapshot>,
    /// Feature count
    features_extracted: AtomicU64,
    /// Memory usage
    memory_bytes: AtomicUsize,
    /// Max memory limit
    max_memory_mb: u64,
}

impl FeatureExtractor {
    pub fn new(max_memory_mb: u64) -> Self {
        Self {
            trade_signs: RollingBuffer::new(1000),
            trade_sizes: RollingBuffer::new(1000),
            price_changes_1ms: RollingBuffer::new(100),
            price_changes_10ms: RollingBuffer::new(1000),
            price_changes_100ms: RollingBuffer::new(10000),
            last_snapshot: dashmap::DashMap::new(),
            features_extracted: AtomicU64::new(0),
            memory_bytes: AtomicUsize::new(0),
            max_memory_mb,
        }
    }

    /// Process market snapshot and extract features
    pub fn extract_features(&self, snapshot: MarketSnapshot) -> FeatureVector {
        let mut features = FeatureVector::new();
        features.timestamp_ns = snapshot.timestamp_ns;
        features.symbol_hash = hash_symbol(&snapshot.symbol);
        
        // Store/update last snapshot
        self.last_snapshot.insert(snapshot.symbol.clone(), snapshot.clone());
        
        // Calculate microprice
        let spread = snapshot.ask_price - snapshot.bid_price;
        let midpoint = (snapshot.bid_price + snapshot.ask_price) / 2.0;
        
        if spread > 0.0 && snapshot.bid_size + snapshot.ask_size > 0 {
            let ask_weight = snapshot.bid_size as f64 / (snapshot.bid_size + snapshot.ask_size) as f64;
            features.microprice = snapshot.bid_price + spread * ask_weight;
        } else {
            features.microprice = midpoint;
        }
        
        features.mid_price = midpoint;
        
        // Spread in basis points
        if midpoint > 0.0 {
            features.spread_bps = (spread / midpoint) * 10000.0;
        }
        
        // Depth imbalance
        let total_depth = snapshot.total_bid_depth + snapshot.total_ask_depth;
        if total_depth > 0 {
            features.bid_depth_imbalance = 
                (snapshot.total_bid_depth as f64 - snapshot.total_ask_depth as f64) / total_depth as f64;
        }
        
        // Order flow imbalance from recent trades
        if self.trade_signs.count > 0 {
            features.order_flow_imbalance = self.calculate_order_flow_imbalance();
            features.buy_volume_pressure = self.calculate_buy_pressure();
            features.sell_volume_pressure = self.calculate_sell_pressure();
            features.trade_sign_autocorr = self.calculate_autocorrelation();
        }
        
        // Update rolling buffers with new trade
        let trade_sign = if snapshot.last_trade_aggressor { 1.0 } else { -1.0 };
        self.trade_signs.push(trade_sign);
        self.trade_sizes.push(snapshot.last_trade_size as f64);
        
        // Price changes (would need historical prices - simplified here)
        self.price_changes_1ms.push(0.0);
        self.price_changes_10ms.push(0.0);
        self.price_changes_100ms.push(0.0);
        
        // Volatility estimates
        features.realized_vol_10ms = self.price_changes_1ms.std_dev();
        features.realized_vol_100ms = self.price_changes_100ms.std_dev();
        
        // SMC signals (simplified - would use more complex logic in production)
        features.smc_block_signal = self.calculate_smc_block_signal(&snapshot);
        features.smc_fair_value_gap = self.calculate_fvg(&snapshot);
        features.smc_liquidity_sweep = self.calculate_liquidity_sweep(&snapshot);
        
        self.features_extracted.fetch_add(1, Ordering::Relaxed);
        
        features
    }
    
    fn calculate_order_flow_imbalance(&self) -> f64 {
        if self.trade_signs.count == 0 { return 0.0; }
        self.trade_signs.mean()
    }
    
    fn calculate_buy_pressure(&self) -> f64 {
        let mut buy_volume = 0.0;
        let mut total_volume = 0.0;
        
        for i in 0..self.trade_signs.count {
            let idx = (self.trade_signs.head + self.trade_signs.capacity - self.trade_signs.count + i) % self.trade_signs.capacity;
            if self.trade_signs.values[idx] > 0.0 {
                buy_volume += self.trade_sizes.values[idx];
            }
            total_volume += self.trade_sizes.values[idx];
        }
        
        if total_volume > 0.0 { buy_volume / total_volume } else { 0.0 }
    }
    
    fn calculate_sell_pressure(&self) -> f64 {
        1.0 - self.calculate_buy_pressure()
    }
    
    fn calculate_autocorrelation(&self) -> f64 {
        if self.trade_signs.count < 2 { return 0.0; }
        
        let mean = self.trade_signs.mean();
        let mut numerator = 0.0;
        let mut denominator = 0.0;
        
        for i in 1..self.trade_signs.count {
            let idx_curr = (self.trade_signs.head + self.trade_signs.capacity - self.trade_signs.count + i) % self.trade_signs.capacity;
            let idx_prev = (self.trade_signs.head + self.trade_signs.capacity - self.trade_signs.count + i - 1) % self.trade_signs.capacity;
            
            let curr = self.trade_signs.values[idx_curr] - mean;
            let prev = self.trade_signs.values[idx_prev] - mean;
            
            numerator += curr * prev;
            denominator += curr * curr;
        }
        
        if denominator > 0.0 { numerator / denominator } else { 0.0 }
    }
    
    fn calculate_smc_block_signal(&self, snapshot: &MarketSnapshot) -> f64 {
        // Simplified SMC block detection
        // In production, this would analyze multi-timeframe structure
        0.0
    }
    
    fn calculate_fvg(&self, snapshot: &MarketSnapshot) -> f64 {
        // Fair Value Gap calculation
        0.0
    }
    
    fn calculate_liquidity_sweep(&self, snapshot: &MarketSnapshot) -> f64 {
        // Liquidity sweep detection
        0.0
    }
    
    /// Get number of features extracted
    pub fn features_extracted(&self) -> u64 {
        self.features_extracted.load(Ordering::Relaxed)
    }
    
    /// Clear all data
    pub fn clear(&self) {
        self.last_snapshot.clear();
        self.features_extracted.store(0, Ordering::Relaxed);
    }
}

/// Hash symbol string to u64
fn hash_symbol(symbol: &str) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    let mut hasher = DefaultHasher::new();
    symbol.hash(&mut hasher);
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_vector_alignment() {
        let fv = FeatureVector::new();
        
        // Verify size
        assert_eq!(FeatureVector::SIZE_BYTES, std::mem::size_of::<FeatureVector>());
        
        // Verify feature count
        assert_eq!(FeatureVector::FEATURE_COUNT, 40);
        
        // Verify slice access
        let slice = fv.as_slice();
        assert_eq!(slice.len(), 40);
    }

    #[test]
    fn test_feature_extractor_basic() {
        let extractor = FeatureExtractor::new(100);
        
        let snapshot = MarketSnapshot {
            symbol: "BTCUSD".to_string(),
            timestamp_ns: 1000000000,
            bid_price: 49999.5,
            ask_price: 50000.5,
            bid_size: 100,
            ask_size: 150,
            last_trade_price: 50000.0,
            last_trade_size: 10,
            last_trade_aggressor: true,
            total_bid_depth: 5000,
            total_ask_depth: 6000,
        };
        
        let features = extractor.extract_features(snapshot);
        
        assert_eq!(features.symbol_hash, hash_symbol("BTCUSD"));
        assert!(features.microprice > 0.0);
        assert!(features.spread_bps > 0.0);
        
        assert!(extractor.features_extracted() >= 1);
    }
}
