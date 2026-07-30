//! Cross-Chain Depeg Guard
//! 
//! Implements strict, multi-venue depegging safeguards for stablecoins across
//! different chains. Instantly halts trading and liquidates toxic inventory
//! if a stablecoin deviates beyond dynamically calculated statistical threshold.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::collections::HashMap;

/// Stablecoin price observation from a venue
#[derive(Debug, Clone)]
pub struct PriceObservation {
    pub symbol: String,
    pub chain: String,
    pub venue: String,
    pub price: f64,
    pub volume_24h: f64,
    pub timestamp_ns: u64,
}

/// Statistical bounds for depeg detection
#[derive(Debug, Clone)]
pub struct StatisticalBounds {
    pub mean: f64,
    pub std_dev: f64,
    pub upper_bound: f64,
    pub lower_bound: f64,
    pub z_score_threshold: f64,
    pub sample_count: usize,
}

/// Depeg alert severity
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DepegSeverity {
    Normal,
    Warning,
    Critical,
    Emergency,
}

/// Depeg alert
#[derive(Debug, Clone)]
pub struct DepegAlert {
    pub symbol: String,
    pub chain: String,
    pub current_price: f64,
    pub expected_price: f64,
    pub deviation_pct: f64,
    pub severity: DepegSeverity,
    pub recommended_action: DepegAction,
    pub timestamp_ns: u64,
}

/// Recommended action for depeg situation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DepegAction {
    Hold,
    ReduceExposure,
    FullLiquidation,
    HaltTrading,
}

/// Welford's online algorithm for running statistics
struct WelfordStats {
    count: usize,
    mean: f64,
    m2: f64,
    min_value: f64,
    max_value: f64,
}

impl WelfordStats {
    fn new() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            m2: 0.0,
            min_value: f64::MAX,
            max_value: f64::MIN,
        }
    }

    fn update(&mut self, value: f64) {
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
        self.min_value = self.min_value.min(value);
        self.max_value = self.max_value.max(value);
    }

    fn variance(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }
        self.m2 / (self.count - 1) as f64
    }

    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    fn get_bounds(&self, z_score: f64) -> StatisticalBounds {
        let std_dev = self.std_dev();
        StatisticalBounds {
            mean: self.mean,
            std_dev,
            upper_bound: self.mean + z_score * std_dev,
            lower_bound: self.mean - z_score * std_dev,
            z_score_threshold: z_score,
            sample_count: self.count,
        }
    }

    fn reset(&mut self) {
        self.count = 0;
        self.mean = 0.0;
        self.m2 = 0.0;
        self.min_value = f64::MAX;
        self.max_value = f64::MIN;
    }
}

/// Circular buffer for recent prices
struct PriceBuffer {
    prices: Vec<f64>,
    head: usize,
    count: usize,
    capacity: usize,
}

impl PriceBuffer {
    fn new(capacity: usize) -> Self {
        Self {
            prices: vec![0.0; capacity],
            head: 0,
            count: 0,
            capacity,
        }
    }

    fn push(&mut self, price: f64) {
        self.prices[self.head] = price;
        self.head = (self.head + 1) % self.capacity;
        if self.count < self.capacity {
            self.count += 1;
        }
    }

    fn iter(&self) -> impl Iterator<Item = &f64> {
        if self.count == 0 {
            return self.prices[0..0].iter();
        }
        
        let start = if self.count < self.capacity { 0 } else { self.head };
        let end = start + self.count;
        
        // Create a wrapped iterator
        (0..self.count).map(move |i| {
            let idx = (start + i) % self.capacity;
            &self.prices[idx]
        })
    }

    fn clear(&mut self) {
        self.head = 0;
        self.count = 0;
    }
}

/// Depeg guard for monitoring stablecoin pegs across venues
pub struct DepegGuard {
    /// Welford stats per (symbol, chain)
    stats: dashmap::DashMap<(String, String), WelfordStats>,
    /// Recent price buffer per (symbol, chain)
    price_buffers: dashmap::DashMap<(String, String), PriceBuffer>,
    /// Current prices per venue
    current_prices: dashmap::DashMap<(String, String, String), f64>,
    /// Z-score threshold for alerts
    z_score_threshold: f64,
    /// Deviation percentage threshold for emergency
    emergency_deviation_pct: f64,
    /// Global halt flag
    global_halt: AtomicBool,
    /// Alert counter
    alert_count: AtomicU64,
    /// Last check timestamp
    last_check_ns: AtomicU64,
}

impl DepegGuard {
    pub fn new(z_score_threshold: f64, emergency_deviation_pct: f64) -> Self {
        Self {
            stats: dashmap::DashMap::new(),
            price_buffers: dashmap::DashMap::new(),
            current_prices: dashmap::DashMap::new(),
            z_score_threshold,
            emergency_deviation_pct,
            global_halt: AtomicBool::new(false),
            alert_count: AtomicU64::new(0),
            last_check_ns: AtomicU64::new(0),
        }
    }

    /// Add a price observation
    pub fn add_observation(&self, obs: PriceObservation) -> Option<DepegAlert> {
        let key = (obs.symbol.clone(), obs.chain.clone());
        let venue_key = (obs.symbol.clone(), obs.chain.clone(), obs.venue.clone());
        
        // Update current price
        self.current_prices.insert(venue_key, obs.price);
        
        // Get or create stats
        let mut stats_entry = self.stats.entry(key.clone()).or_insert(WelfordStats::new());
        stats_entry.update(obs.price);
        
        // Get or create price buffer
        let mut buffer_entry = self.price_buffers.entry(key.clone()).or_insert(PriceBuffer::new(100));
        buffer_entry.push(obs.price);
        
        // Check for depeg
        self.check_depeg(&obs.symbol, &obs.chain, obs.price)
    }

    /// Check if current price indicates depeg
    fn check_depeg(&self, symbol: &str, chain: &str, current_price: f64) -> Option<DepegAlert> {
        let key = (symbol.to_string(), chain.to_string());
        let stats_entry = self.stats.get(&key)?;
        
        let bounds = stats_entry.get_bounds(self.z_score_threshold);
        
        if bounds.sample_count < 10 {
            return None; // Not enough data
        }
        
        let deviation = current_price - bounds.mean;
        let deviation_pct = (deviation / bounds.mean * 100.0).abs();
        let z_score = if bounds.std_dev > 0.0 {
            deviation.abs() / bounds.std_dev
        } else {
            0.0
        };
        
        // Determine severity and action
        let (severity, action) = if deviation_pct > self.emergency_deviation_pct || z_score > 5.0 {
            (DepegSeverity::Emergency, DepegAction::FullLiquidation)
        } else if deviation_pct > self.emergency_deviation_pct / 2.0 || z_score > 3.0 {
            (DepegSeverity::Critical, DepegAction::HaltTrading)
        } else if z_score > self.z_score_threshold {
            (DepegSeverity::Warning, DepegAction::ReduceExposure)
        } else {
            (DepegSeverity::Normal, DepegAction::Hold)
        };
        
        self.last_check_ns.store(timestamp_ns(), Ordering::Relaxed);
        
        if severity != DepegSeverity::Normal {
            self.alert_count.fetch_add(1, Ordering::Relaxed);
            
            if severity == DepegSeverity::Emergency || severity == DepegSeverity::Critical {
                self.global_halt.store(true, Ordering::SeqCst);
            }
            
            Some(DepegAlert {
                symbol: symbol.to_string(),
                chain: chain.to_string(),
                current_price,
                expected_price: bounds.mean,
                deviation_pct,
                severity,
                recommended_action: action,
                timestamp_ns: timestamp_ns(),
            })
        } else {
            None
        }
    }

    /// Get statistical bounds for a symbol/chain pair
    pub fn get_bounds(&self, symbol: &str, chain: &str) -> Option<StatisticalBounds> {
        let key = (symbol.to_string(), chain.to_string());
        let stats_entry = self.stats.get(&key)?;
        Some(stats_entry.get_bounds(self.z_score_threshold))
    }

    /// Get current price for a symbol/chain/venue
    pub fn get_price(&self, symbol: &str, chain: &str, venue: &str) -> Option<f64> {
        let key = (symbol.to_string(), chain.to_string(), venue.to_string());
        self.current_prices.get(&key).map(|e| *e.value())
    }

    /// Get average price across venues for a symbol/chain
    pub fn get_average_price(&self, symbol: &str, chain: &str) -> Option<f64> {
        let prefix = (symbol.to_string(), chain.to_string());
        let mut sum = 0.0;
        let mut count = 0;
        
        for entry in self.current_prices.iter() {
            let (sym, chn, _) = entry.key();
            if sym == &prefix.0 && chn == &prefix.1 {
                sum += *entry.value();
                count += 1;
            }
        }
        
        if count > 0 {
            Some(sum / count as f64)
        } else {
            None
        }
    }

    /// Check price divergence between chains for same symbol
    pub fn check_cross_chain_divergence(&self, symbol: &str) -> Vec<DepegAlert> {
        let mut alerts = Vec::new();
        
        // Get all chains for this symbol
        let chains: Vec<String> = self.stats
            .iter()
            .filter(|entry| entry.key().0 == symbol)
            .map(|entry| entry.key().1.clone())
            .collect();
        
        if chains.len() < 2 {
            return alerts;
        }
        
        // Get average prices per chain
        let mut prices: HashMap<String, f64> = HashMap::new();
        for chain in &chains {
            if let Some(price) = self.get_average_price(symbol, chain) {
                prices.insert(chain.clone(), price);
            }
        }
        
        if prices.len() < 2 {
            return alerts;
        }
        
        // Find reference price (median)
        let mut sorted_prices: Vec<f64> = prices.values().cloned().collect();
        sorted_prices.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let reference = if sorted_prices.len() % 2 == 0 {
            (sorted_prices[sorted_prices.len() / 2 - 1] + sorted_prices[sorted_prices.len() / 2]) / 2.0
        } else {
            sorted_prices[sorted_prices.len() / 2]
        };
        
        // Check each chain against reference
        for (chain, price) in &prices {
            let deviation_pct = ((price - reference) / reference * 100.0).abs();
            
            if deviation_pct > self.emergency_deviation_pct / 2.0 {
                let severity = if deviation_pct > self.emergency_deviation_pct {
                    DepegSeverity::Critical
                } else {
                    DepegSeverity::Warning
                };
                
                alerts.push(DepegAlert {
                    symbol: symbol.to_string(),
                    chain: chain.clone(),
                    current_price: *price,
                    expected_price: reference,
                    deviation_pct,
                    severity,
                    recommended_action: if severity == DepegSeverity::Critical {
                        DepegAction::HaltTrading
                    } else {
                        DepegAction::ReduceExposure
                    },
                    timestamp_ns: timestamp_ns(),
                });
            }
        }
        
        alerts
    }

    /// Trigger global halt
    pub fn trigger_halt(&self) {
        self.global_halt.store(true, Ordering::SeqCst);
    }

    /// Clear global halt
    pub fn clear_halt(&self) {
        self.global_halt.store(false, Ordering::SeqCst);
    }

    /// Check if halted
    pub fn is_halted(&self) -> bool {
        self.global_halt.load(Ordering::Relaxed)
    }

    /// Get alert count
    pub fn alert_count(&self) -> u64 {
        self.alert_count.load(Ordering::Relaxed)
    }

    /// Reset statistics for a symbol/chain
    pub fn reset_stats(&self, symbol: &str, chain: &str) {
        let key = (symbol.to_string(), chain.to_string());
        if let Some(mut stats) = self.stats.get_mut(&key) {
            stats.value().reset();
        }
        if let Some(mut buffer) = self.price_buffers.get_mut(&key) {
            buffer.value().clear();
        }
    }

    /// Clear all data
    pub fn clear(&self) {
        self.stats.clear();
        self.price_buffers.clear();
        self.current_prices.clear();
        self.alert_count.store(0, Ordering::Relaxed);
        self.global_halt.store(false, Ordering::SeqCst);
    }
}

/// Get current timestamp in nanoseconds
#[inline]
fn timestamp_ns() -> u64 {
    Instant::now()
        .duration_since(Instant::now() - Duration::from_secs(1))
        .as_nanos() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_depeg_guard_basic() {
        let guard = DepegGuard::new(3.0, 5.0);
        
        // Add normal observations to build statistics
        for i in 0..20 {
            let obs = PriceObservation {
                symbol: "USDT".to_string(),
                chain: "Ethereum".to_string(),
                venue: "Binance".to_string(),
                price: 1.0 + (i as f64 * 0.001 - 0.01), // Prices around 1.0
                volume_24h: 1000000.0,
                timestamp_ns: timestamp_ns(),
            };
            guard.add_observation(obs);
        }
        
        // Check bounds
        let bounds = guard.get_bounds("USDT", "Ethereum");
        assert!(bounds.is_some());
        let bounds = bounds.unwrap();
        assert!(bounds.mean > 0.98 && bounds.mean < 1.02);
        
        // Add extreme price (depeg)
        let extreme_obs = PriceObservation {
            symbol: "USDT".to_string(),
            chain: "Ethereum".to_string(),
            venue: "Binance".to_string(),
            price: 0.85, // 15% depeg
            volume_24h: 1000000.0,
            timestamp_ns: timestamp_ns(),
        };
        
        let alert = guard.add_observation(extreme_obs);
        assert!(alert.is_some());
        let alert = alert.unwrap();
        assert!(alert.severity == DepegSeverity::Emergency || alert.severity == DepegSeverity::Critical);
        assert!(guard.is_halted());
    }

    #[test]
    fn test_cross_chain_divergence() {
        let guard = DepegGuard::new(3.0, 5.0);
        
        // Add observations for USDT on Ethereum
        for _ in 0..20 {
            guard.add_observation(PriceObservation {
                symbol: "USDT".to_string(),
                chain: "Ethereum".to_string(),
                venue: "Uniswap".to_string(),
                price: 1.0,
                volume_24h: 1000000.0,
                timestamp_ns: timestamp_ns(),
            });
        }
        
        // Add observations for USDT on Solana (slightly different)
        for _ in 0..20 {
            guard.add_observation(PriceObservation {
                symbol: "USDT".to_string(),
                chain: "Solana".to_string(),
                venue: "Raydium".to_string(),
                price: 1.005,
                volume_24h: 500000.0,
                timestamp_ns: timestamp_ns(),
            });
        }
        
        // Add observations for USDT on Tron (significantly different - depeg)
        for _ in 0..20 {
            guard.add_observation(PriceObservation {
                symbol: "USDT".to_string(),
                chain: "Tron".to_string(),
                venue: "SunSwap".to_string(),
                price: 0.94,
                volume_24h: 200000.0,
                timestamp_ns: timestamp_ns(),
            });
        }
        
        let alerts = guard.check_cross_chain_divergence("USDT");
        assert!(!alerts.is_empty());
    }
}
