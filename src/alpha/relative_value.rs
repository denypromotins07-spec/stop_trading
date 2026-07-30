//! Relative Value Matrix
//! 
//! Multi-asset relative value matrix calculating instantaneous pairwise mispricings.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
use dashmap::DashMap;

/// Maximum number of assets in the matrix
const MAX_ASSETS: usize = 64;

/// Pairwise mispricing signal
#[derive(Debug, Clone)]
pub struct PairMispricing {
    /// First asset
    pub asset_a: String,
    /// Second asset
    pub asset_b: String,
    /// Z-score of mispricing
    pub z_score: f64,
    /// Fair value ratio
    pub fair_value_ratio: f64,
    /// Current ratio
    pub current_ratio: f64,
    /// Recommended trade direction
    pub direction: TradeDirection,
    /// Confidence score
    pub confidence: f64,
    /// Timestamp
    pub timestamp_ns: u64,
}

/// Trade direction for relative value
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TradeDirection {
    /// Long A, Short B
    LongA_ShortB,
    /// Short A, Long B
    ShortA_LongB,
    /// No trade
    Neutral,
}

/// Rolling statistics for a pair
struct PairStats {
    /// Sum of ratios
    sum_ratio: f64,
    /// Sum of squared ratios
    sum_ratio2: f64,
    /// Count
    count: usize,
    /// Ring buffer for rolling window
    ratios: [f64; 200],
    /// Head position
    head: usize,
    /// Is full
    is_full: bool,
}

impl PairStats {
    fn new() -> Self {
        Self {
            sum_ratio: 0.0,
            sum_ratio2: 0.0,
            count: 0,
            ratios: [0.0; 200],
            head: 0,
            is_full: false,
        }
    }

    #[inline]
    fn update(&mut self, ratio: f64) {
        if self.is_full {
            let old = self.ratios[self.head];
            self.sum_ratio -= old;
            self.sum_ratio2 -= old * old;
        } else {
            self.count += 1;
        }

        self.sum_ratio += ratio;
        self.sum_ratio2 += ratio * ratio;
        self.ratios[self.head] = ratio;
        self.head = (self.head + 1) % 200;

        if self.count >= 200 {
            self.is_full = true;
        }
    }

    #[inline]
    fn mean(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        self.sum_ratio / self.count as f64
    }

    #[inline]
    fn variance(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }
        let mean = self.mean();
        (self.sum_ratio2 / self.count as f64) - (mean * mean)
    }

    #[inline]
    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    #[inline]
    fn z_score(&self, ratio: f64) -> f64 {
        let std = self.std_dev();
        if std < 1e-10 {
            return 0.0;
        }
        (ratio - self.mean()) / std
    }
}

impl Default for PairStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Asset entry in the matrix
struct AssetEntry {
    /// Symbol
    symbol: [u8; 12],
    /// Current price
    price: f64,
    /// Last update timestamp
    last_update_ns: AtomicU64,
}

/// Relative Value Matrix
pub struct RelativeValueMatrix {
    /// Assets
    assets: DashMap<u8, AssetEntry>,
    /// Asset count
    asset_count: u8,
    /// Pair statistics (keyed by pair index)
    pair_stats: DashMap<u64, PairStats>,
    /// Mispricing threshold (Z-score)
    threshold: f64,
    /// Signals generated
    signals_generated: AtomicU64,
}

impl RelativeValueMatrix {
    pub fn new(max_assets: usize) -> Self {
        let _ = max_assets; // Use parameter for validation if needed
        Self {
            assets: DashMap::new(),
            asset_count: 0,
            pair_stats: DashMap::new(),
            threshold: 2.0,
            signals_generated: AtomicU64::new(0),
        }
    }

    /// Register an asset
    pub fn register_asset(&mut self, symbol: &str) -> Option<u8> {
        if self.asset_count >= MAX_ASSETS as u8 {
            return None;
        }

        let idx = self.asset_count;
        
        let mut bytes = [0u8; 12];
        let name_bytes = symbol.as_bytes();
        bytes[..name_bytes.len().min(12)].copy_from_slice(&name_bytes[..name_bytes.len().min(12)]);

        let entry = AssetEntry {
            symbol: bytes,
            price: 0.0,
            last_update_ns: AtomicU64::new(0),
        };

        self.assets.insert(idx, entry);
        self.asset_count += 1;
        Some(idx)
    }

    /// Update price for an asset
    pub fn update_price(&self, symbol: &str, price: f64, timestamp_ns: u64) {
        // Find asset index
        let asset_idx = self.find_asset_index(symbol);
        if let Some(idx) = asset_idx {
            if let Some(mut entry) = self.assets.get_mut(&idx) {
                entry.price = price;
                entry.last_update_ns.store(timestamp_ns, Ordering::Relaxed);
            }

            // Update all pairs involving this asset
            self.update_pairs(idx, price, timestamp_ns);
        }
    }

    fn find_asset_index(&self, symbol: &str) -> Option<u8> {
        for entry in self.assets.iter() {
            let sym = self.symbol_str(&entry.value().symbol);
            if sym == Some(symbol) {
                return Some(*entry.key());
            }
        }
        None
    }

    fn symbol_str(&self, bytes: &[u8; 12]) -> Option<&str> {
        let end = bytes.iter().position(|&b| b == 0).unwrap_or(12);
        std::str::from_utf8(&bytes[..end]).ok()
    }

    fn update_pairs(&self, updated_idx: u8, updated_price: f64, timestamp_ns: u64) {
        // Update all pairs involving this asset
        for other_entry in self.assets.iter() {
            let other_idx = *other_entry.key();
            if other_idx == updated_idx {
                continue;
            }

            let other_price = other_entry.value().price;
            if other_price <= 0.0 {
                continue;
            }

            // Create pair key (sorted to ensure consistency)
            let (a_idx, b_idx) = if updated_idx < other_idx {
                (updated_idx, other_idx)
            } else {
                (other_idx, updated_idx)
            };

            let pair_key = ((a_idx as u64) << 32) | (b_idx as u64);
            let ratio = updated_price / other_price;

            // Update pair stats
            if let Some(mut stats) = self.pair_stats.get_mut(&pair_key) {
                stats.update(ratio);
                
                // Check for mispricing
                let z = stats.z_score(ratio);
                if z.abs() > self.threshold && stats.is_full {
                    self.signals_generated.fetch_add(1, Ordering::Relaxed);
                }
            } else {
                let mut stats = PairStats::new();
                stats.update(ratio);
                self.pair_stats.insert(pair_key, stats);
            }
        }
    }

    /// Get all current mispricings above threshold
    pub fn get_mispricings(&self) -> Vec<PairMispricing> {
        let mut mispricings = Vec::new();
        let timestamp_ns = Instant::now().duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default().as_nanos() as u64;

        for entry in self.pair_stats.iter() {
            let pair_key = *entry.key();
            let stats = entry.value();
            
            if !stats.is_full || stats.count < 50 {
                continue;
            }

            // Decode pair indices
            let a_idx = (pair_key >> 32) as u8;
            let b_idx = (pair_key & 0xFFFFFFFF) as u8;

            // Get current prices
            let price_a = self.assets.get(&a_idx).map(|e| e.price).unwrap_or(0.0);
            let price_b = self.assets.get(&b_idx).map(|e| e.price).unwrap_or(0.0);

            if price_a <= 0.0 || price_b <= 0.0 {
                continue;
            }

            let current_ratio = price_a / price_b;
            let z = stats.z_score(current_ratio);

            if z.abs() > self.threshold {
                let direction = if z > 0.0 {
                    TradeDirection::ShortA_LongB
                } else {
                    TradeDirection::LongA_ShortB
                };

                let symbol_a = self.symbol_str(&self.assets.get(&a_idx).unwrap().symbol)
                    .unwrap_or("Unknown").to_string();
                let symbol_b = self.symbol_str(&self.assets.get(&b_idx).unwrap().symbol)
                    .unwrap_or("Unknown").to_string();

                mispricings.push(PairMispricing {
                    asset_a: symbol_a,
                    asset_b: symbol_b,
                    z_score: z,
                    fair_value_ratio: stats.mean(),
                    current_ratio,
                    direction,
                    confidence: (z.abs() / self.threshold).min(1.0),
                    timestamp_ns,
                });
            }
        }

        mispricings
    }

    /// Set mispricing threshold
    pub fn set_threshold(&mut self, threshold: f64) {
        self.threshold = threshold;
    }

    /// Get signals generated count
    pub fn get_signal_count(&self) -> u64 {
        self.signals_generated.load(Ordering::Relaxed)
    }

    /// Get number of tracked pairs
    pub fn pair_count(&self) -> usize {
        self.pair_stats.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relative_value_matrix() {
        let mut matrix = RelativeValueMatrix::new(32);

        let btc = matrix.register_asset("BTC").unwrap();
        let eth = matrix.register_asset("ETH").unwrap();
        let sol = matrix.register_asset("SOL").unwrap();

        println!("Registered assets: BTC={}, ETH={}, SOL={}", btc, eth, sol);

        // Simulate price updates
        for i in 1..=250 {
            let ts = i * 1_000_000_000u64;
            let btc_price = 50000.0 + (i as f64 * 10.0);
            let eth_price = 3000.0 + (i as f64 * 5.0);
            let sol_price = 100.0 + (i as f64 * 2.0);

            matrix.update_price("BTC", btc_price, ts);
            matrix.update_price("ETH", eth_price, ts);
            matrix.update_price("SOL", sol_price, ts);
        }

        println!("Tracked pairs: {}", matrix.pair_count());
        
        let mispricings = matrix.get_mispricings();
        println!("Found {} mispricings", mispricings.len());
        
        for mp in mispricings {
            println!("{} vs {}: z={:.2}, direction={:?}", 
                mp.asset_a, mp.asset_b, mp.z_score, mp.direction);
        }
    }

    #[test]
    fn test_pair_stats() {
        let mut stats = PairStats::new();
        
        // Feed consistent ratio
        for _ in 0..250 {
            stats.update(2.0);
        }

        assert!((stats.mean() - 2.0).abs() < 0.01);
        assert!(stats.std_dev() < 0.01);
        
        // Add outlier
        let z = stats.z_score(3.0);
        assert!(z > 5.0, "Should detect outlier");
    }
}
