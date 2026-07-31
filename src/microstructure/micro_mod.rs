//! Microstructure Module Root
//!
//! Feeds tick signatures and algorithmic footprints to the alpha ensemble.

pub mod tick_analytics;
pub mod trade_signature;

pub use tick_analytics::{
    TickAnalytics,
    TradeTick,
    InterArrivalStats,
    PoissonTestResult,
    AlgoType,
    AlgoFootprint,
    TickEvent,
};

pub use trade_signature::{
    LeeReadyClassifier,
    ClassifiedTrade,
    QuoteSnapshot,
    ClassificationMethod,
    ClassificationStats,
    ClassificationEvent,
};

/// Combined microstructure analysis engine
pub struct MicrostructureEngine {
    pub tick_analytics: tick_analytics::TickAnalytics,
    pub classifier: trade_signature::LeeReadyClassifier,
}

impl MicrostructureEngine {
    pub fn new(buffer_size: usize) -> Self {
        Self {
            tick_analytics: TickAnalytics::new(1000, 60_000_000_000, buffer_size),
            classifier: LeeReadyClassifier::new(buffer_size),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_creation() {
        let engine = MicrostructureEngine::new(1000);
        
        assert!(engine.tick_analytics.get_iat_stats("BTCUSDT").is_none());
        assert_eq!(engine.classifier.get_stats().total_classified, 0);
    }
}
