//! Volume-Synchronized Probability of Informed Trading (VPIN)
//! 
//! Implements VPIN to measure order flow toxicity in real-time.
//! Uses volume buckets instead of time buckets to detect institutional informed traders.

use std::collections::VecDeque;

/// Configuration for VPIN calculation
#[derive(Debug, Clone)]
pub struct VpinConfig {
    /// Number of volume buckets to maintain
    pub num_buckets: usize,
    /// Target volume per bucket (in base asset units)
    pub bucket_volume: u64,
    /// Threshold for toxicity alert
    pub toxicity_threshold: f64,
}

impl Default for VpinConfig {
    fn default() -> Self {
        Self {
            num_buckets: 50,
            bucket_volume: 1_000_000, // 1M base units
            toxicity_threshold: 0.85,
        }
    }
}

/// Represents a single volume bucket
#[derive(Debug, Clone)]
struct VolumeBucket {
    buy_volume: u64,
    sell_volume: u64,
    total_volume: u64,
}

impl VolumeBucket {
    fn new() -> Self {
        Self {
            buy_volume: 0,
            sell_volume: 0,
            total_volume: 0,
        }
    }

    fn add_trade(&mut self, volume: u64, is_buy: bool) {
        if is_buy {
            self.buy_volume += volume;
        } else {
            self.sell_volume += volume;
        }
        self.total_volume += volume;
    }

    fn is_full(&self, target: u64) -> bool {
        self.total_volume >= target
    }

    /// Calculate |V_buy - V_sell| / (V_buy + V_sell) for this bucket
    fn imbalance(&self) -> f64 {
        let total = self.buy_volume as f64 + self.sell_volume as f64;
        if total < 1e-9 {
            return 0.0;
        }
        ((self.buy_volume as f64) - (self.sell_volume as f64)).abs() / total
    }
}

/// Real-time VPIN calculator
#[derive(Debug)]
pub struct VpinCalculator {
    config: VpinConfig,
    buckets: VecDeque<VolumeBucket>,
    current_bucket: VolumeBucket,
    sum_imbalances: f64,
    is_toxic: bool,
    last_vpin: f64,
}

impl VpinCalculator {
    pub fn new(config: VpinConfig) -> Self {
        Self {
            config,
            buckets: VecDeque::with_capacity(config.num_buckets),
            current_bucket: VolumeBucket::new(),
            sum_imbalances: 0.0,
            is_toxic: false,
            last_vpin: 0.0,
        }
    }

    /// Process a trade and update VPIN
    /// Returns (new_vpin, is_toxic) if a bucket was completed
    pub fn process_trade(&mut self, volume: u64, price: f64, is_buyer_maker: bool) -> Option<(f64, bool)> {
        // is_buyer_maker = true means the buyer was the maker, so this was a sell
        let is_buy = !is_buyer_maker;
        
        self.current_bucket.add_trade(volume, is_buy);
        
        // Check if current bucket is full
        if self.current_bucket.is_full(self.config.bucket_volume) {
            self.complete_bucket();
            Some((self.last_vpin, self.is_toxic))
        } else {
            None
        }
    }

    fn complete_bucket(&mut self) {
        let bucket_imbalance = self.current_bucket.imbalance();
        
        // Add new bucket
        let old_bucket = if self.buckets.len() >= self.config.num_buckets {
            // Remove oldest bucket and subtract its contribution
            self.buckets.pop_front().map(|b| b.imbalance())
        } else {
            None
        };

        if let Some(old_imb) = old_bucket {
            self.sum_imbalances -= old_imb;
        }

        self.sum_imbalances += bucket_imbalance;
        self.buckets.push_back(std::mem::replace(&mut self.current_bucket, VolumeBucket::new()));

        // Calculate VPIN
        let num_buckets = self.buckets.len() as f64;
        if num_buckets > 0.0 {
            self.last_vpin = self.sum_imbalances / num_buckets;
        }

        // Update toxicity flag
        self.is_toxic = self.last_vpin >= self.config.toxicity_threshold;
    }

    /// Get current VPIN value
    pub fn vpin(&self) -> f64 {
        self.last_vpin
    }

    /// Check if market is currently toxic
    pub fn is_toxic(&self) -> bool {
        self.is_toxic
    }

    /// Get number of filled buckets
    pub fn bucket_count(&self) -> usize {
        self.buckets.len()
    }

    /// Reset the calculator
    pub fn reset(&mut self) {
        self.buckets.clear();
        self.current_bucket = VolumeBucket::new();
        self.sum_imbalances = 0.0;
        self.is_toxic = false;
        self.last_vpin = 0.0;
    }

    /// Get recommended spread widening factor based on toxicity
    pub fn spread_multiplier(&self) -> f64 {
        if self.is_toxic {
            // Exponential widening based on VPIN level
            1.0 + (self.last_vpin * 2.0).exp() - 1.0
        } else {
            1.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vpin_basic() {
        let config = VpinConfig {
            num_buckets: 10,
            bucket_volume: 1000,
            toxicity_threshold: 0.7,
        };
        let mut calc = VpinCalculator::new(config);

        // Add only buy trades
        for i in 0..10 {
            let result = calc.process_trade(100, 50000.0, false);
            if i == 9 {
                assert!(result.is_some());
                let (vpin, toxic) = result.unwrap();
                assert!(vpin > 0.9); // Should be highly toxic
                assert!(toxic);
            } else {
                assert!(result.is_none());
            }
        }
    }

    #[test]
    fn test_spread_multiplier() {
        let mut calc = VpinCalculator::new(VpinConfig::default());
        assert_eq!(calc.spread_multiplier(), 1.0);
    }
}
