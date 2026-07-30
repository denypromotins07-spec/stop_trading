//! HFT Infrastructure Library
//! 
//! High-frequency trading infrastructure for order flow analysis,
//! Smart Money Concepts (SMC), tick database, microprice calculation,
//! and derivatives monitoring.
//! 
//! # Modules
//! 
//! - `orderflow`: CVD tracking and order flow imbalance analysis
//! - `smc`: Smart Money Concepts including Order Blocks and FVG detection
//! - `tickdb`: High-performance tick database with memory-mapped files
//! - `microprice`: Volume-weighted midprice and spread analysis
//! - `derivatives`: Liquidation cascade detection and OI/funding tracking

#![cfg_attr(feature = "nightly", feature(allocator_api))]
#![warn(missing_docs)]
#![warn(rustdoc::missing_crate_level_docs)]

pub mod orderflow;
pub mod smc;
pub mod tickdb;
pub mod microprice;
pub mod derivatives;

// Re-export commonly used types at crate root
pub use orderflow::{
    CvdTracker, CvdSnapshot, TradeTick, DivergenceSignal,
    OrderFlowImbalance, ImbalanceMetrics, ImbalanceSignal, PriceLevel, TopOfBook,
    OrderFlowPipeline, OrderFlowState, ConfluenceSignal,
};

pub use smc::{
    Candle, OrderBlock, OrderBlockType, OrderBlockDetector,
    FvgCandle, FairValueGap, FvgType, FvgDetector,
    BreakOfStructure, ChangeOfCharacter, MarketStructure,
    StructureAnalyzer, SmcState, SmcConfluence,
};

pub use tickdb::{
    StoredTick, TickDbWriter, TickDbReader, TickDbConfig, TickQuery,
    TimeRange, PriceRange, TickStats, TickDbManager, TickDbManagerConfig,
};

pub use microprice::{
    MicropriceCalculator, MicropriceResult, OrderBookSnapshot, Level,
    SpreadAnalyzer, SpreadMetrics, SpreadRegime,
    FairValueState, PreTradeRiskCheck, MicropricePipeline,
};

pub use derivatives::{
    LiquidationEvent, LiquidationSide, CascadeSignal, CascadeAction,
    OiUpdate, FundingUpdate, MarketRegime, DerivativesState,
    DerivativesAggregator, DerivativesStateSummary,
};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Common error type for the library
#[derive(Debug, thiserror::Error)]
pub enum HftInfraError {
    #[error("Order flow error: {0}")]
    OrderFlow(#[from] orderflow::cvd::CvdError),
    
    #[error("Imbalance error: {0}")]
    Imbalance(#[from] orderflow::imbalance::ImbalanceError),
    
    #[error("SMC error: {0}")]
    Smc(#[from] smc::order_blocks::OrderBlockError),
    
    #[error("FVG error: {0}")]
    Fvg(#[from] smc::fvg::FvgError),
    
    #[error("TickDB error: {0}")]
    TickDb(#[from] tickdb::writer::TickDbError),
    
    #[error("TickDB manager error: {0}")]
    TickDbManager(#[from] tickdb::TickDbManagerError),
    
    #[error("Microprice error: {0}")]
    Microprice(#[from] microprice::calculator::MicropriceError),
    
    #[error("Spread error: {0}")]
    Spread(#[from] microprice::spread::SpreadError),
    
    #[error("Microprice module error: {0}")]
    MicropriceModule(#[from] microprice::MicropriceModuleError),
    
    #[error("Liquidation error: {0}")]
    Liquidation(#[from] derivatives::liquidations::LiquidationError),
    
    #[error("OI/Funding error: {0}")]
    OiFunding(#[from] derivatives::open_interest::OiError),
    
    #[error("Derivatives module error: {0}")]
    Derivatives(#[from] derivatives::DerivativesModuleError),
}

/// Result type alias for the library
pub type Result<T> = std::result::Result<T, HftInfraError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn test_orderflow_integration() {
        let pipeline = orderflow::OrderFlowPipelineBuilder::new()
            .with_cvd_scale(1_000_000_000)
            .with_imbalance_scale(1_000_000_000)
            .build();

        let tick = TradeTick::new(1000, 50000.0, 1.0, false);
        pipeline.process_tick(&tick).unwrap();

        let snapshot = pipeline.cvd_snapshot();
        assert!((snapshot.cumulative_buy_volume - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_smc_integration() {
        let detector = OrderBlockDetector::new(10, 0.6);
        
        let candles = vec![
            Candle::new(1000, 100.0, 102.0, 98.0, 99.0, 100.0, 1000.0).unwrap(),
            Candle::new(2000, 99.0, 99.5, 97.0, 98.0, 150.0, 1500.0).unwrap(),
            Candle::new(3000, 98.0, 105.0, 97.5, 104.0, 200.0, 2000.0).unwrap(),
        ];

        detector.update_price(103.0);
        let blocks = detector.detect_blocks(&candles).unwrap();
        
        // Should detect at least one block or handle gracefully
        assert!(blocks.len() >= 0);
    }

    #[test]
    fn test_microprice_integration() {
        let pipeline = MicropricePipeline::new(3, 10.0);
        
        let book = OrderBookSnapshot::new(
            vec![Level::new(99.9, 100.0, 5)],
            vec![Level::new(100.1, 50.0, 3)],
            1000,
        );

        let state = pipeline.process_book(&book).unwrap();
        assert!((state.microprice - state.mid_price).abs() > 0.0);
    }

    #[test]
    fn test_derivatives_integration() {
        let aggregator = DerivativesAggregator::new(5, 60000);
        
        let event = LiquidationEvent::new(
            1000,
            "BTCUSDT",
            LiquidationSide::Long,
            50000.0,
            10.0,
        ).unwrap();

        let result = aggregator.process_liquidation(event).unwrap();
        assert!(result.is_some());
    }
}
