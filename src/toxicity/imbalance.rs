//! Multi-level Weighted Order Imbalance (OIB) Calculator
//! 
//! Analyzes bid/ask depth across the top 10 L2 levels with exponential decay weighting.

/// Configuration for OIB calculation
#[derive(Debug, Clone)]
pub struct OibConfig {
    /// Number of L2 levels to analyze (max 10)
    pub num_levels: usize,
    /// Decay factor for level weighting (0.5 = each level worth half of previous)
    pub decay_factor: f64,
}

impl Default for OibConfig {
    fn default() -> Self {
        Self {
            num_levels: 10,
            decay_factor: 0.7,
        }
    }
}

/// Represents a single price level in the order book
#[derive(Debug, Clone, Copy)]
pub struct PriceLevel {
    pub price: f64,
    pub quantity: u64,
    pub order_count: u32,
}

impl PriceLevel {
    pub fn new(price: f64, quantity: u64, order_count: u32) -> Self {
        Self { price, quantity, order_count }
    }
}

/// Real-time Order Imbalance calculator
#[derive(Debug)]
pub struct OrderImbalanceCalculator {
    config: OibConfig,
    weights: Vec<f64>,
    bid_levels: Vec<PriceLevel>,
    ask_levels: Vec<PriceLevel>,
    last_oib: f64,
    last_weighted_oib: f64,
    /// Cached sum of weights for normalization
    weight_sum: f64,
}

impl OrderImbalanceCalculator {
    pub fn new(config: OibConfig) -> Self {
        let num_levels = config.num_levels.min(10);
        let mut weights = Vec::with_capacity(num_levels);
        let mut weight_sum = 0.0;

        // Calculate exponential decay weights
        for i in 0..num_levels {
            let w = config.decay_factor.powi(i as i32);
            weight_sum += w;
            weights.push(w);
        }

        Self {
            config,
            weights,
            bid_levels: Vec::with_capacity(num_levels),
            ask_levels: Vec::with_capacity(num_levels),
            last_oib: 0.0,
            last_weighted_oib: 0.0,
            weight_sum,
        }
    }

    /// Update the order book snapshot and recalculate OIB
    /// Expects sorted vectors: bids descending, asks ascending
    pub fn update(&mut self, bids: &[PriceLevel], asks: &[PriceLevel]) -> (f64, f64) {
        let num_levels = self.config.num_levels.min(10);
        
        // Truncate or pad to expected size
        self.bid_levels.clear();
        self.ask_levels.clear();
        
        for i in 0..num_levels {
            self.bid_levels.push(bids.get(i).copied().unwrap_or(PriceLevel { price: 0.0, quantity: 0, order_count: 0 }));
            self.ask_levels.push(asks.get(i).copied().unwrap_or(PriceLevel { price: 0.0, quantity: 0, order_count: 0 }));
        }

        self.calculate_oib();
        (self.last_oib, self.last_weighted_oib)
    }

    fn calculate_oib(&mut self) {
        let mut total_bid_vol = 0u64;
        let mut total_ask_vol = 0u64;
        let mut weighted_bid_vol = 0.0;
        let mut weighted_ask_vol = 0.0;

        for (i, (bid, ask)) in self.bid_levels.iter().zip(self.ask_levels.iter()).enumerate() {
            let weight = self.weights[i];
            
            total_bid_vol += bid.quantity;
            total_ask_vol += ask.quantity;
            
            weighted_bid_vol += (bid.quantity as f64) * weight;
            weighted_ask_vol += (ask.quantity as f64) * weight;
        }

        // Raw OIB: (BidVol - AskVol) / (BidVol + AskVol)
        let total_vol = total_bid_vol as f64 + total_ask_vol as f64;
        self.last_oib = if total_vol < 1e-9 {
            0.0
        } else {
            ((total_bid_vol as f64) - (total_ask_vol as f64)) / total_vol
        };

        // Weighted OIB with normalization
        let weighted_total = weighted_bid_vol + weighted_ask_vol;
        self.last_weighted_oib = if weighted_total < 1e-9 {
            0.0
        } else {
            (weighted_bid_vol - weighted_ask_vol) / weighted_total
        };
    }

    /// Get the raw order imbalance (-1 to 1)
    pub fn oib(&self) -> f64 {
        self.last_oib
    }

    /// Get the weighted order imbalance (-1 to 1)
    pub fn weighted_oib(&self) -> f64 {
        self.last_weighted_oib
    }

    /// Check if there's significant bid pressure (> 0.3)
    pub fn is_bid_pressure(&self) -> bool {
        self.last_weighted_oib > 0.3
    }

    /// Check if there's significant ask pressure (< -0.3)
    pub fn is_ask_pressure(&self) -> bool {
        self.last_weighted_oib < -0.3
    }

    /// Get the imbalance score scaled to [0, 1] where 0.5 is neutral
    pub fn normalized_score(&self) -> f64 {
        0.5 + (self.last_weighted_oib / 2.0)
    }

    /// Calculate the volume-weighted mid price considering imbalance
    pub fn imbalanced_mid_price(&self, best_bid: f64, best_ask: f64) -> f64 {
        let standard_mid = (best_bid + best_ask) / 2.0;
        let spread = best_ask - best_bid;
        // Adjust mid price based on imbalance
        standard_mid + (spread * self.last_weighted_oib * 0.1)
    }

    /// Get recommended position sizing adjustment based on OIB
    /// Returns multiplier: >1.0 for favorable direction, <1.0 for adverse
    pub fn position_multiplier(&self, is_long: bool) -> f64 {
        if is_long {
            // Favorable if OIB is positive (more bid pressure)
            1.0 + self.last_weighted_oib.clamp(-0.5, 0.5)
        } else {
            // Favorable if OIB is negative (more ask pressure)
            1.0 - self.last_weighted_oib.clamp(-0.5, 0.5)
        }
    }
}

/// Streaming OIB delta calculator for detecting changes
#[derive(Debug)]
pub struct OibDeltaTracker {
    prev_oib: f64,
    prev_weighted_oib: f64,
    delta_threshold: f64,
}

impl OibDeltaTracker {
    pub fn new(delta_threshold: f64) -> Self {
        Self {
            prev_oib: 0.0,
            prev_weighted_oib: 0.0,
            delta_threshold,
        }
    }

    /// Check if OIB changed significantly
    pub fn check_delta(&mut self, current_oib: f64, current_weighted: f64) -> bool {
        let oib_delta = (current_oib - self.prev_oib).abs();
        let weighted_delta = (current_weighted - self.prev_weighted_oib).abs();
        
        let changed = oib_delta > self.delta_threshold || weighted_delta > self.delta_threshold;
        
        if changed {
            self.prev_oib = current_oib;
            self.prev_weighted_oib = current_weighted;
        }
        
        changed
    }

    /// Get the magnitude of the last change
    pub fn last_change_magnitude(&self, current_oib: f64, current_weighted: f64) -> f64 {
        ((current_oib - self.prev_oib).powi(2) + (current_weighted - self.prev_weighted_oib).powi(2)).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_oib_basic() {
        let config = OibConfig::default();
        let mut calc = OrderImbalanceCalculator::new(config);

        let bids = vec![
            PriceLevel::new(99.0, 1000, 5),
            PriceLevel::new(98.0, 500, 3),
        ];
        let asks = vec![
            PriceLevel::new(100.0, 200, 2),
            PriceLevel::new(101.0, 300, 4),
        ];

        let (oib, weighted) = calc.update(&bids, &asks);
        
        // More bid volume than ask, should be positive
        assert!(oib > 0.0);
        assert!(weighted > 0.0);
    }

    #[test]
    fn test_imbalanced_mid() {
        let mut calc = OrderImbalanceCalculator::new(OibConfig::default());
        
        // Set up strong bid pressure
        let bids = vec![PriceLevel::new(99.0, 10000, 10)];
        let asks = vec![PriceLevel::new(100.0, 100, 1)];
        calc.update(&bids, &asks);

        let imbalanced_mid = calc.imbalanced_mid_price(99.0, 100.0);
        let standard_mid = 99.5;
        
        // With strong bid pressure, imbalanced mid should be higher
        assert!(imbalanced_mid > standard_mid);
    }

    #[test]
    fn test_delta_tracker() {
        let mut tracker = OibDeltaTracker::new(0.1);
        
        assert!(!tracker.check_delta(0.0, 0.0)); // Initial
        assert!(tracker.check_delta(0.5, 0.5));  // Large change
        assert!(!tracker.check_delta(0.51, 0.51)); // Small change
    }
}
