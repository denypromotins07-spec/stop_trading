//! Factor Model Module Root
//! 
//! Feeds composite alpha signals directly to the strategy router and ensemble logic.

pub mod cross_sectional;
pub mod multi_factor;

pub use cross_sectional::{CrossSectionalRanker, AssetUniverse};
pub use multi_factor::{MultiFactorEngine, FactorType, FactorMetrics, FactorTracker};

/// Combined alpha signal pipeline
pub struct AlphaPipeline {
    pub ranker: CrossSectionalRanker,
    pub factor_engine: MultiFactorEngine,
}

impl AlphaPipeline {
    pub const fn new() -> Self {
        Self {
            ranker: CrossSectionalRanker::new(),
            factor_engine: MultiFactorEngine::new(),
        }
    }
    
    /// Process all signals and generate composite alpha
    pub fn process(&self) {
        // Compute cross-sectional ranks
        self.ranker.compute_ranks();
        
        // Compute multi-factor composites
        self.factor_engine.compute_composite_scores();
    }
    
    /// Get combined signal for an asset
    pub fn get_combined_signal(&self, asset_idx: usize) -> f64 {
        let cs_signal = self.ranker.get_composite_signal(asset_idx, 0.5);
        let mf_signal = self.factor_engine.get_composite_score(asset_idx);
        
        // Blend signals (equal weight by default)
        (cs_signal + mf_signal) / 2.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_alpha_pipeline() {
        let pipeline = AlphaPipeline::new();
        
        // Setup universe
        let mut universe = AssetUniverse::new();
        let btc = universe.register_asset("BTC").unwrap();
        let eth = universe.register_asset("ETH").unwrap();
        
        pipeline.ranker.set_count(universe.active_count());
        pipeline.factor_engine.set_num_assets(universe.active_count());
        
        // Update signals
        pipeline.ranker.update_return(btc, 100 << 48);
        pipeline.ranker.update_return(eth, -50 << 48);
        
        pipeline.factor_engine.register_factor(FactorType::Momentum, 0.5);
        pipeline.factor_engine.register_factor(FactorType::Value, 0.5);
        
        pipeline.factor_engine.update_factor_score(FactorType::Momentum, btc, 0.8);
        pipeline.factor_engine.update_factor_score(FactorType::Value, btc, 0.6);
        pipeline.factor_engine.update_factor_score(FactorType::Momentum, eth, -0.3);
        pipeline.factor_engine.update_factor_score(FactorType::Value, eth, 0.2);
        
        // Process
        pipeline.process();
        
        // Verify BTC has higher signal than ETH
        let btc_signal = pipeline.get_combined_signal(btc);
        let eth_signal = pipeline.get_combined_signal(eth);
        
        assert!(btc_signal > eth_signal);
    }
}
