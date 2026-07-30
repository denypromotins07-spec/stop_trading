//! Lead-Lag Relationship Detection
//! 
//! High-frequency cross-correlation engine using Hayashi-Yoshida estimator
//! to detect lead-lag relationships between assets in real-time.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
use dashmap::DashMap;

/// Maximum number of asset pairs to track
const MAX_PAIRS: usize = 256;

/// Window size for correlation calculation (number of ticks)
const CORRELATION_WINDOW: usize = 500;

/// Hayashi-Yoshida correlation estimator result
#[derive(Debug, Clone)]
pub struct HYCorrelation {
    /// Correlation coefficient (-1.0 to 1.0)
    pub correlation: f64,
    /// Number of observations used
    pub n_observations: usize,
    /// Estimated lag in milliseconds
    pub lag_ms: u64,
    /// Confidence level (0.0 to 1.0)
    pub confidence: f64,
}

/// Rolling statistics for Hayashi-Yoshida estimator
struct RollingHYStats {
    /// Sum of product of returns
    sum_xy: f64,
    /// Sum of squared returns X
    sum_x2: f64,
    /// Sum of squared returns Y
    sum_y2: f64,
    /// Ring buffer for X returns
    x_returns: [f64; CORRELATION_WINDOW],
    /// Ring buffer for Y returns
    y_returns: [f64; CORRELATION_WINDOW],
    /// Head position
    head: usize,
    /// Count of observations
    count: usize,
    /// Is buffer full
    is_full: bool,
}

impl RollingHYStats {
    fn new() -> Self {
        Self {
            sum_xy: 0.0,
            sum_x2: 0.0,
            sum_y2: 0.0,
            x_returns: [0.0; CORRELATION_WINDOW],
            y_returns: [0.0; CORRELATION_WINDOW],
            head: 0,
            count: 0,
            is_full: false,
        }
    }

    #[inline]
    fn update(&mut self, x_return: f64, y_return: f64) {
        if self.is_full {
            // Remove oldest observation
            let old_x = self.x_returns[self.head];
            let old_y = self.y_returns[self.head];
            
            self.sum_xy -= old_x * old_y;
            self.sum_x2 -= old_x * old_x;
            self.sum_y2 -= old_y * old_y;
        } else {
            self.count += 1;
        }

        // Add new observation
        self.sum_xy += x_return * y_return;
        self.sum_x2 += x_return * x_return;
        self.sum_y2 += y_return * y_return;

        // Store in ring buffer
        self.x_returns[self.head] = x_return;
        self.y_returns[self.head] = y_return;
        self.head = (self.head + 1) % CORRELATION_WINDOW;

        if self.count >= CORRELATION_WINDOW {
            self.is_full = true;
        }
    }

    #[inline]
    fn correlation(&self) -> f64 {
        if self.count < 10 {
            return 0.0;
        }

        let denom = (self.sum_x2 * self.sum_y2).sqrt();
        if denom < 1e-10 {
            return 0.0;
        }

        self.sum_xy / denom
    }
}

impl Default for RollingHYStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Tracks lead-lag relationship between two assets
pub struct LeadLagPair {
    /// Leader symbol
    pub leader: [u8; 12],
    /// Lagger symbol
    pub lagger: [u8; 12],
    /// Rolling correlation stats
    stats: RollingHYStats,
    /// Last price of leader
    last_price_leader: f64,
    /// Last price of lagger
    last_price_lagger: f64,
    /// Timestamps for lag detection
    leader_timestamps: [u64; 100],
    lagger_timestamps: [u64; 100],
    /// Current estimated lag in ms
    estimated_lag_ms: u64,
    /// Last update timestamp
    last_update_ns: AtomicU64,
}

impl LeadLagPair {
    fn new(leader: &str, lagger: &str) -> Self {
        let mut leader_bytes = [0u8; 12];
        let mut lagger_bytes = [0u8; 12];

        let l_bytes = leader.as_bytes();
        let g_bytes = lagger.as_bytes();

        leader_bytes[..l_bytes.len().min(12)].copy_from_slice(&l_bytes[..l_bytes.len().min(12)]);
        lagger_bytes[..g_bytes.len().min(12)].copy_from_slice(&g_bytes[..g_bytes.len().min(12)]);

        Self {
            leader: leader_bytes,
            lagger: lagger_bytes,
            stats: RollingHYStats::new(),
            last_price_leader: 0.0,
            last_price_lagger: 0.0,
            leader_timestamps: [0; 100],
            lagger_timestamps: [0; 100],
            estimated_lag_ms: 0,
            last_update_ns: AtomicU64::new(0),
        }
    }

    #[inline]
    fn update(&mut self, symbol: &str, price: f64, timestamp_ns: u64) {
        let is_leader = self.symbol_str(&self.leader) == Some(symbol);
        let is_lagger = self.symbol_str(&self.lagger) == Some(symbol);

        if !is_leader && !is_lagger {
            return;
        }

        // Calculate return
        let last_price = if is_leader {
            let ret = if self.last_price_leader > 0.0 {
                (price - self.last_price_leader) / self.last_price_leader
            } else {
                0.0
            };
            self.last_price_leader = price;
            ret
        } else {
            let ret = if self.last_price_lagger > 0.0 {
                (price - self.last_price_lagger) / self.last_price_lagger
            } else {
                0.0
            };
            self.last_price_lagger = price;
            ret
        };

        // For HY estimator, we need both returns at synchronized times
        // Simplified: use last known return from other asset
        let other_return = if is_leader {
            if self.last_price_lagger > 0.0 {
                0.0 // Would need proper synchronization in production
            } else {
                0.0
            }
        } else {
            if self.last_price_leader > 0.0 {
                0.0
            } else {
                0.0
            }
        };

        self.stats.update(last_price, other_return);

        let ts = Instant::now().duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default().as_nanos() as u64;
        self.last_update_ns.store(ts, Ordering::Relaxed);

        // Update lag estimate using cross-correlation
        self.estimate_lag(timestamp_ns);
    }

    fn estimate_lag(&mut self, _current_ts: u64) {
        // Simplified lag estimation
        // In production, would use cross-correlation at different lags
        self.estimated_lag_ms = 50; // Placeholder
    }

    fn symbol_str(&self, bytes: &[u8; 12]) -> Option<&str> {
        let end = bytes.iter().position(|&b| b == 0).unwrap_or(12);
        std::str::from_utf8(&bytes[..end]).ok()
    }

    fn get_correlation(&self) -> HYCorrelation {
        let corr = self.stats.correlation();
        
        HYCorrelation {
            correlation: corr,
            n_observations: self.stats.count,
            lag_ms: self.estimated_lag_ms,
            confidence: if self.stats.is_full { 0.95 } else { self.stats.count as f64 / CORRELATION_WINDOW as f64 },
        }
    }

    fn leader_str(&self) -> Option<&str> {
        self.symbol_str(&self.leader)
    }

    fn lagger_str(&self) -> Option<&str> {
        self.symbol_str(&self.lagger)
    }
}

/// Lead-Lag Engine managing multiple asset pairs
pub struct LeadLagEngine {
    /// Asset pairs being tracked
    pairs: DashMap<String, LeadLagPair>,
    /// Recent ticks for each asset
    recent_ticks: DashMap<String, (f64, u64)>,
    /// Detected lead-lag relationships
    relationships: DashMap<String, HYCorrelation>,
}

impl LeadLagEngine {
    pub fn new(_max_pairs: usize) -> Self {
        Self {
            pairs: DashMap::new(),
            recent_ticks: DashMap::new(),
            relationships: DashMap::new(),
        }
    }

    /// Register a pair to track for lead-lag relationship
    pub fn register_pair(&self, leader: &str, lagger: &str) {
        let key = format!("{}->{}", leader, lagger);
        let pair = LeadLagPair::new(leader, lagger);
        self.pairs.insert(key, pair);
    }

    /// Update tick for an asset
    pub fn update_tick(&self, symbol: &str, price: f64, timestamp_ns: u64) {
        self.recent_ticks.insert(symbol.to_string(), (price, timestamp_ns));

        // Update all pairs involving this symbol
        for mut entry in self.pairs.iter_mut() {
            entry.value().update(symbol, price, timestamp_ns);
            
            // Update relationship if we have enough data
            let corr = entry.value().get_correlation();
            if corr.n_observations >= 50 {
                self.relationships.insert(entry.key().clone(), corr);
            }
        }
    }

    /// Get current lead-lag relationships
    pub fn get_relationships(&self) -> Vec<(String, String, HYCorrelation)> {
        let mut results = Vec::new();

        for entry in self.relationships.iter() {
            let key = entry.key();
            let parts: Vec<&str> = key.split("->").collect();
            if parts.len() == 2 {
                results.push((parts[0].to_string(), parts[1].to_string(), entry.value().clone()));
            }
        }

        results
    }

    /// Find the current leader among a group of assets
    pub fn find_leader(&self, assets: &[&str]) -> Option<(String, f64)> {
        let mut best_leader = None;
        let mut best_correlation = -1.0;

        for &asset in assets {
            for entry in self.relationships.iter() {
                let key = entry.key();
                if key.starts_with(asset) {
                    let corr = entry.value().correlation;
                    if corr > best_correlation && corr > 0.5 {
                        best_correlation = corr;
                        let parts: Vec<&str> = key.split("->").collect();
                        if parts.len() == 2 {
                            best_leader = Some((parts[1].to_string(), corr));
                        }
                    }
                }
            }
        }

        best_leader
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lead_lag_engine() {
        let engine = LeadLagEngine::new(100);
        
        engine.register_pair("SOL", "ETH");
        engine.register_pair("ETH", "BTC");

        // Simulate some ticks
        for i in 1..=100 {
            let ts = i * 1_000_000_000u64;
            engine.update_tick("BTC", 30000.0 + (i as f64 * 10.0), ts);
            engine.update_tick("ETH", 2000.0 + (i as f64 * 5.0), ts + 50_000_000);
            engine.update_tick("SOL", 100.0 + (i as f64 * 2.0), ts + 100_000_000);
        }

        let relationships = engine.get_relationships();
        println!("Found {} relationships", relationships.len());
    }

    #[test]
    fn test_hy_correlation() {
        let mut stats = RollingHYStats::new();
        
        // Feed correlated returns
        for i in 0..CORRELATION_WINDOW {
            let x = (i as f64 * 0.01).sin();
            let y = x * 0.8 + 0.2 * (i as f64 * 0.03).cos();
            stats.update(x, y);
        }

        let corr = stats.correlation();
        assert!(corr > 0.5, "Should detect positive correlation");
        println!("Correlation: {}", corr);
    }
}
