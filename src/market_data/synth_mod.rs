//! Synthetic Data Module Root
//! 
//! Feeds global liquidity metrics to alpha and execution engines.

pub mod synthetic;
pub mod aggregated_l2;

pub use synthetic::{SyntheticIndex, IndexComponent, IndexTick, WeightingMethod};
pub use aggregated_l2::{L2Aggregator, Exchange, Level, AggregatedLevel, ArbOpportunity};

/// Combined market data engine
pub struct MarketDataEngine {
    pub synthetic_indices: [Option<SyntheticIndex>; 8],
    pub l2_aggregator: L2Aggregator,
}

impl MarketDataEngine {
    pub const fn new() -> Self {
        Self {
            synthetic_indices: [None; 8],
            l2_aggregator: L2Aggregator::new(),
        }
    }
    
    /// Register a synthetic index
    pub fn register_index(&mut self, name: &'static str, base_value: f64) -> Option<usize> {
        for i in 0..8 {
            if self.synthetic_indices[i].is_none() {
                self.synthetic_indices[i] = Some(SyntheticIndex::new(name, base_value));
                return Some(i);
            }
        }
        None
    }
    
    /// Get reference to a synthetic index
    pub fn get_index(&self, idx: usize) -> Option<&SyntheticIndex> {
        self.synthetic_indices.get(idx).and_then(|i| i.as_ref())
    }
    
    /// Update price across all relevant indices
    pub fn broadcast_price(&self, symbol: &str, price: i64) {
        for i in 0..8 {
            if let Some(ref index) = self.synthetic_indices[i] {
                let _ = index.update_price(symbol, price);
            }
        }
    }
    
    /// Get cross-exchange spread for a symbol
    pub fn get_cross_exchange_spread(&self) -> f64 {
        self.l2_aggregator.get_spread_bps()
    }
    
    /// Check for arbitrage opportunities
    pub fn check_arbitrage(&self) -> bool {
        self.get_cross_exchange_spread() > 50.0 // 50 bps threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_market_data_engine() {
        let mut engine = MarketDataEngine::new();
        
        // Register synthetic index
        let idx = engine.register_index("Crypto10", 1000.0);
        assert!(idx.is_some());
        
        // Add component
        if let Some(ref mut index) = engine.synthetic_indices[idx.unwrap()].as_mut() {
            index.add_component("BTC", 0.5);
            index.add_component("ETH", 0.5);
        }
        
        // Broadcast price
        engine.broadcast_price("BTC", 50000 << 48);
        
        // Verify
        let index = engine.get_index(idx.unwrap()).unwrap();
        assert_eq!(index.component_count(), 2);
    }
}
