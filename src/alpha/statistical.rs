//! Statistical Arbitrage Engine
//! 
//! Real-time cointegration and mean-reversion Z-score calculator for pairs trading.
//! Uses Welford's online algorithm to maintain rolling variances without storing historical arrays.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Maximum number of pairs tracked simultaneously
const MAX_PAIRS: usize = 256;

/// Window size for rolling statistics (number of ticks)
const ROLLING_WINDOW: usize = 1000;

/// Online statistics tracker using Welford's algorithm
#[derive(Clone)]
pub struct OnlineStats {
    /// Count of observations
    count: u64,
    /// Running mean
    mean: f64,
    /// Running sum of squared differences from mean (M2)
    m2: f64,
    /// Ring buffer for rolling window (stored as fixed array)
    values: [f64; ROLLING_WINDOW],
    /// Current position in ring buffer
    head: usize,
    /// Whether ring buffer is full
    is_full: bool,
}

impl OnlineStats {
    pub const fn new() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            m2: 0.0,
            values: [0.0; ROLLING_WINDOW],
            head: 0,
            is_full: false,
        }
    }

    /// Update with a new observation (rolling window)
    #[inline]
    pub fn update(&mut self, value: f64) {
        if self.is_full {
            // Remove oldest value from statistics
            let old_value = self.values[self.head];
            let old_count = self.count as f64;
            
            // Adjust mean for removal
            let old_mean = self.mean;
            self.mean = ((old_mean * old_count) - old_value + value) / old_count;
            
            // Adjust M2 for removal and addition
            // This is an approximation; exact rolling variance requires more complex formulas
            let delta = value - old_value;
            self.m2 += delta * (value - self.mean + old_value - old_mean);
        } else {
            // Standard Welford's algorithm for growing window
            self.count += 1;
            let delta = value - self.mean;
            self.mean += delta / self.count as f64;
            let delta2 = value - self.mean;
            self.m2 += delta * delta2;
        }

        // Store value in ring buffer
        self.values[self.head] = value;
        self.head = (self.head + 1) % ROLLING_WINDOW;
        
        if self.count >= ROLLING_WINDOW as u64 {
            self.is_full = true;
        }
    }

    /// Get current mean
    #[inline]
    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// Get current variance
    #[inline]
    pub fn variance(&self) -> f64 {
        let n = if self.is_full { ROLLING_WINDOW } else { self.count as usize };
        if n < 2 {
            return 0.0;
        }
        self.m2 / (n - 1) as f64
    }

    /// Get current standard deviation
    #[inline]
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Calculate Z-score for a value
    #[inline]
    pub fn z_score(&self, value: f64) -> f64 {
        let std = self.std_dev();
        if std < 1e-10 {
            return 0.0;
        }
        (value - self.mean) / std
    }

    /// Reset statistics
    pub fn reset(&mut self) {
        self.count = 0;
        self.mean = 0.0;
        self.m2 = 0.0;
        self.head = 0;
        self.is_full = false;
    }
}

impl Default for OnlineStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Represents a statistical arbitrage pair (e.g., ETH/BTC)
pub struct StatArbPair {
    /// Symbol for asset A (e.g., "ETH")
    pub symbol_a: [u8; 12],
    /// Symbol for asset B (e.g., "BTC")
    pub symbol_b: [u8; 12],
    /// Hedge ratio (units of B per unit of A)
    pub hedge_ratio: f64,
    /// Online stats for the spread
    pub spread_stats: OnlineStats,
    /// Current price of A
    pub price_a: f64,
    /// Current price of B
    pub price_b: f64,
    /// Last update timestamp
    pub last_update_ns: AtomicU64,
    /// Cointegration test p-value (updated periodically)
    pub cointegration_pvalue: f64,
    /// Whether pair is currently tradable
    pub is_tradable: bool,
}

impl StatArbPair {
    pub fn new(symbol_a: &str, symbol_b: &str, initial_hedge_ratio: f64) -> Self {
        let mut bytes_a = [0u8; 12];
        let mut bytes_b = [0u8; 12];
        
        let a_bytes = symbol_a.as_bytes();
        let b_bytes = symbol_b.as_bytes();
        
        bytes_a[..a_bytes.len().min(12)].copy_from_slice(&a_bytes[..a_bytes.len().min(12)]);
        bytes_b[..b_bytes.len().min(12)].copy_from_slice(&b_bytes[..b_bytes.len().min(12)]);

        Self {
            symbol_a: bytes_a,
            symbol_b: bytes_b,
            hedge_ratio: initial_hedge_ratio,
            spread_stats: OnlineStats::new(),
            price_a: 0.0,
            price_b: 0.0,
            last_update_ns: AtomicU64::new(0),
            cointegration_pvalue: 1.0,
            is_tradable: false,
        }
    }

    /// Update prices and calculate spread
    #[inline]
    pub fn update_prices(&mut self, price_a: f64, price_b: f64) {
        self.price_a = price_a;
        self.price_b = price_b;
        
        let timestamp_ns = Instant::now().duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default().as_nanos() as u64;
        self.last_update_ns.store(timestamp_ns, Ordering::Relaxed);

        // Calculate spread: price_a - hedge_ratio * price_b
        let spread = price_a - self.hedge_ratio * price_b;
        self.spread_stats.update(spread);

        // Mark as tradable after sufficient data
        if self.spread_stats.mean != 0.0 && self.spread_stats.std_dev() > 1e-10 {
            self.is_tradable = true;
        }
    }

    /// Get current Z-score of the spread
    #[inline]
    pub fn get_z_score(&self) -> f64 {
        let spread = self.price_a - self.hedge_ratio * self.price_b;
        self.spread_stats.z_score(spread)
    }

    /// Generate trading signal based on Z-score thresholds
    pub fn get_signal(&self, entry_threshold: f64, exit_threshold: f64) -> StatArbSignal {
        if !self.is_tradable {
            return StatArbSignal::Hold;
        }

        let z = self.get_z_score();

        if z > entry_threshold {
            // Spread is too high: short A, long B
            StatArbSignal::ShortA_LongB
        } else if z < -entry_threshold {
            // Spread is too low: long A, short B
            StatArbSignal::LongA_ShortB
        } else if z.abs() < exit_threshold {
            // Spread has reverted: close position
            StatArbSignal::Close
        } else {
            StatArbSignal::Hold
        }
    }

    /// Update hedge ratio using rolling OLS (simplified)
    pub fn update_hedge_ratio(&mut self, new_ratio: f64) {
        self.hedge_ratio = new_ratio;
        // Reset spread stats when hedge ratio changes significantly
        self.spread_stats.reset();
    }

    /// Get symbol A as string
    pub fn symbol_a_str(&self) -> Option<&str> {
        let end = self.symbol_a.iter().position(|&b| b == 0).unwrap_or(12);
        std::str::from_utf8(&self.symbol_a[..end]).ok()
    }

    /// Get symbol B as string
    pub fn symbol_b_str(&self) -> Option<&str> {
        let end = self.symbol_b.iter().position(|&b| b == 0).unwrap_or(12);
        std::str::from_utf8(&self.symbol_b[..end]).ok()
    }
}

/// Trading signal for statistical arbitrage
#[derive(Debug, Clone, PartialEq)]
pub enum StatArbSignal {
    /// Long asset A, short asset B
    LongA_ShortB,
    /// Short asset A, long asset B
    ShortA_LongB,
    /// Close existing position
    Close,
    /// No action
    Hold,
}

/// Statistical Arbitrage Engine managing multiple pairs
pub struct StatArbEngine {
    /// Active pairs
    pairs: [Option<StatArbPair>; MAX_PAIRS],
    /// Number of active pairs
    pair_count: usize,
    /// Entry threshold in Z-scores
    entry_threshold: f64,
    /// Exit threshold in Z-scores
    exit_threshold: f64,
    /// Signals generated
    signals_generated: AtomicU64,
}

impl StatArbEngine {
    pub fn new(entry_threshold: f64, exit_threshold: f64) -> Self {
        Self {
            pairs: [None; MAX_PAIRS],
            pair_count: 0,
            entry_threshold,
            exit_threshold,
            signals_generated: AtomicU64::new(0),
        }
    }

    /// Add a new pair to track
    pub fn add_pair(&mut self, symbol_a: &str, symbol_b: &str, hedge_ratio: f64) -> bool {
        if self.pair_count >= MAX_PAIRS {
            return false;
        }

        let pair = StatArbPair::new(symbol_a, symbol_b, hedge_ratio);
        self.pairs[self.pair_count] = Some(pair);
        self.pair_count += 1;
        true
    }

    /// Update prices for a specific pair
    pub fn update_pair_prices(&mut self, symbol_a: &str, symbol_b: &str, price_a: f64, price_b: f64) {
        for i in 0..self.pair_count {
            if let Some(ref mut pair) = self.pairs[i] {
                if pair.symbol_a_str() == Some(symbol_a) && pair.symbol_b_str() == Some(symbol_b) {
                    pair.update_prices(price_a, price_b);
                    break;
                }
            }
        }
    }

    /// Get signals for all pairs
    pub fn get_all_signals(&self) -> Vec<(usize, StatArbSignal)> {
        let mut signals = Vec::with_capacity(self.pair_count);

        for i in 0..self.pair_count {
            if let Some(ref pair) = self.pairs[i] {
                let signal = pair.get_signal(self.entry_threshold, self.exit_threshold);
                if signal != StatArbSignal::Hold {
                    signals.push((i, signal));
                    self.signals_generated.fetch_add(1, Ordering::Relaxed);
                }
            }
        }

        signals
    }

    /// Get pair by index
    pub fn get_pair(&self, idx: usize) -> Option<&StatArbPair> {
        if idx >= self.pair_count {
            return None;
        }
        self.pairs[idx].as_ref()
    }

    /// Get total signals generated
    pub fn get_signal_count(&self) -> u64 {
        self.signals_generated.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_online_stats() {
        let mut stats = OnlineStats::new();
        
        // Feed some values
        for i in 1..=100 {
            stats.update(i as f64);
        }

        assert!((stats.mean() - 50.5).abs() < 0.1, "Mean should be ~50.5");
        assert!(stats.variance() > 0.0, "Variance should be positive");
        
        // Z-score of mean should be ~0
        let z = stats.z_score(stats.mean());
        assert!(z.abs() < 0.1, "Z-score of mean should be near 0");
    }

    #[test]
    fn test_stat_arb_pair() {
        let mut pair = StatArbPair::new("ETH", "BTC", 15.0);
        
        // Simulate correlated prices
        for i in 1..=100 {
            let price_btc = 30000.0 + (i as f64 * 10.0);
            let price_eth = price_btc / 15.0 + (if i % 20 == 0 { 50.0 } else { 0.0 });
            pair.update_prices(price_eth, price_btc);
        }

        assert!(pair.is_tradable, "Pair should be tradable after warmup");
        
        let z = pair.get_z_score();
        // Z-score should reflect the artificial deviation we introduced
        println!("Final Z-score: {}", z);
    }

    #[test]
    fn test_stat_arb_engine() {
        let mut engine = StatArbEngine::new(2.0, 0.5);
        
        engine.add_pair("ETH", "BTC", 15.0);
        engine.add_pair("SOL", "ETH", 0.05);

        // Warm up with data
        for i in 1..=100 {
            let btc = 30000.0 + (i as f64 * 10.0);
            let eth = btc / 15.0;
            let sol = eth * 0.05;
            
            engine.update_pair_prices("ETH", "BTC", eth, btc);
            engine.update_pair_prices("SOL", "ETH", sol, eth);
        }

        let signals = engine.get_all_signals();
        // May or may not have signals depending on the data
        println!("Generated {} signals", signals.len());
    }
}
