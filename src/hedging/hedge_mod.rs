//! Hedging Module Root
//! 
//! This module integrates beta tracking and cross-margin metrics directly into
//! the global risk bus for real-time statistical hedging operations.

pub mod beta_tracker;
pub mod cross_margin;

// Re-export main types for convenience
pub use beta_tracker::{
    BetaAdjustedReturn, BetaTrackerStats, MultiAssetBetaTracker, OnlineCovariance,
    OnlineStats, RollingBetaCalculator,
};

pub use cross_margin::{
    AssetClass, CapitalEfficiencyCalculator, CrossMarginEngine, CrossMarginStats,
    MarginResult, MarginWarningLevel, Position, PositionSide, RiskParameters,
};

/// Global risk bus integration point
pub struct RiskBusIntegration {
    /// Beta tracker reference
    beta_tracker: MultiAssetBetaTracker,
    /// Cross-margin engine reference
    margin_engine: CrossMarginEngine,
    /// Last update timestamp
    last_update_ns: std::sync::atomic::AtomicU64,
    /// Integration enabled flag
    enabled: std::sync::atomic::AtomicBool,
}

impl RiskBusIntegration {
    pub fn new(
        beta_window_size: usize,
        initial_equity: f64,
    ) -> Self {
        Self {
            beta_tracker: MultiAssetBetaTracker::new(beta_window_size),
            margin_engine: CrossMarginEngine::new(initial_equity),
            last_update_ns: std::sync::atomic::AtomicU64::new(0),
            enabled: std::sync::atomic::AtomicBool::new(true),
        }
    }
    
    /// Update beta for a benchmark
    pub fn update_beta(&self, benchmark: &str, asset_return: f64, benchmark_return: f64) -> f64 {
        if !self.enabled.load(std::sync::atomic::Ordering::Relaxed) {
            return f64::NAN;
        }
        
        let beta = self.beta_tracker.update_beta(benchmark, asset_return, benchmark_return);
        self.last_update_ns.store(current_timestamp_ns(), std::sync::atomic::Ordering::Relaxed);
        beta
    }
    
    /// Update position in margin engine
    pub fn update_position(&self, position: Position) {
        if !self.enabled.load(std::sync::atomic::Ordering::Relaxed) {
            return;
        }
        
        self.margin_engine.update_position(position);
        self.last_update_ns.store(current_timestamp_ns(), std::sync::atomic::Ordering::Relaxed);
    }
    
    /// Get current portfolio margin status
    pub fn get_margin_status(&self) -> Option<MarginResult> {
        if !self.enabled.load(std::sync::atomic::Ordering::Relaxed) {
            return None;
        }
        
        Some(self.margin_engine.calculate_portfolio_margin())
    }
    
    /// Get beta against a benchmark
    pub fn get_beta(&self, benchmark: &str) -> Option<f64> {
        self.beta_tracker.get_beta(benchmark)
    }
    
    /// Calculate beta-adjusted PnL
    pub fn calculate_beta_adjusted_pnl(
        &self,
        raw_pnl: f64,
        benchmark: &str,
        benchmark_return: f64,
    ) -> BetaAdjustedPnl {
        let beta = self.get_beta(benchmark).unwrap_or(1.0);
        let beta_contribution = beta * benchmark_return;
        let alpha = raw_pnl - beta_contribution;
        
        BetaAdjustedPnl {
            raw_pnl,
            beta,
            benchmark_return,
            beta_contribution,
            alpha,
        }
    }
    
    /// Check if we can add more positions
    pub fn can_add_exposure(&self, notional: f64) -> bool {
        // Simplified check - would need symbol info for full implementation
        self.margin_engine.equity() > notional * 0.1 // 10x max leverage
    }
    
    /// Get hedging recommendations
    pub fn get_hedge_recommendation(
        &self,
        target_beta: f64,
        current_exposure: f64,
    ) -> HedgeRecommendation {
        let betas = self.beta_tracker.get_all_betas();
        let btc_beta = betas.get("BTC").copied().unwrap_or(1.0);
        
        // Calculate required hedge to achieve target beta
        let current_beta_exposure = current_exposure * btc_beta;
        let target_beta_exposure = current_exposure * target_beta;
        let hedge_required = target_beta_exposure - current_beta_exposure;
        
        HedgeRecommendation {
            action: if hedge_required.abs() < current_exposure * 0.01 {
                HedgeAction::Hold
            } else if hedge_required > 0.0 {
                HedgeAction::IncreaseLong(hedge_required)
            } else {
                HedgeAction::IncreaseShort(hedge_required.abs())
            },
            target_beta,
            current_beta: btc_beta,
            confidence: self.calculate_hedge_confidence(),
        }
    }
    
    /// Enable/disable risk bus integration
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, std::sync::atomic::Ordering::SeqCst);
    }
    
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(std::sync::atomic::Ordering::Relaxed)
    }
    
    /// Calculate confidence score for hedge recommendations
    fn calculate_hedge_confidence(&self) -> f64 {
        // Based on data quality, recency, and observation count
        let stats = self.beta_tracker.stats();
        let updates = stats.total_updates.load(std::sync::atomic::Ordering::Relaxed);
        
        if updates < 10 {
            return 0.3;
        } else if updates < 50 {
            return 0.6;
        } else if updates < 200 {
            return 0.8;
        }
        
        0.95
    }
}

/// Beta-adjusted PnL breakdown
#[derive(Debug, Clone)]
pub struct BetaAdjustedPnl {
    pub raw_pnl: f64,
    pub beta: f64,
    pub benchmark_return: f64,
    pub beta_contribution: f64,
    pub alpha: f64,
}

/// Hedge recommendation
#[derive(Debug, Clone)]
pub struct HedgeRecommendation {
    pub action: HedgeAction,
    pub target_beta: f64,
    pub current_beta: f64,
    pub confidence: f64,
}

#[derive(Debug, Clone)]
pub enum HedgeAction {
    Hold,
    IncreaseLong(f64),
    IncreaseShort(f64),
    ClosePosition,
}

fn current_timestamp_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64
}

/// Hedging strategy executor
pub struct HedgingStrategy {
    /// Target beta (0 = market neutral)
    target_beta: f64,
    /// Rebalance threshold (beta drift)
    rebalance_threshold: f64,
    /// Minimum trade size
    min_trade_size: f64,
}

impl HedgingStrategy {
    pub fn new(target_beta: f64, rebalance_threshold: f64, min_trade_size: f64) -> Self {
        Self {
            target_beta,
            rebalance_threshold,
            min_trade_size,
        }
    }
    
    /// Check if rebalancing is needed
    pub fn needs_rebalance(&self, current_beta: f64) -> bool {
        (current_beta - self.target_beta).abs() > self.rebalance_threshold
    }
    
    /// Calculate rebalance trade size
    pub fn calculate_rebalance_size(
        &self,
        current_beta: f64,
        portfolio_value: f64,
        hedge_instrument_price: f64,
    ) -> f64 {
        let beta_drift = current_beta - self.target_beta;
        let hedge_notional_needed = beta_drift * portfolio_value;
        let hedge_units = hedge_notional_needed / hedge_instrument_price;
        
        if hedge_units.abs() < self.min_trade_size {
            0.0
        } else {
            hedge_units
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_risk_bus_integration() {
        let integration = RiskBusIntegration::new(20, 100_000.0);
        
        assert!(integration.is_enabled());
        
        // Update some beta data
        for i in 0..20 {
            integration.update_beta("BTC", 0.01 * i as f64, 0.005 * i as f64);
        }
        
        let beta = integration.get_beta("BTC");
        assert!(beta.is_some());
        
        // Get margin status
        let margin = integration.get_margin_status();
        assert!(margin.is_some());
    }
    
    #[test]
    fn test_beta_adjusted_pnl() {
        let integration = RiskBusIntegration::new(20, 100_000.0);
        
        // Pre-populate beta
        for i in 0..20 {
            integration.update_beta("BTC", 0.02 * i as f64, 0.01 * i as f64);
        }
        
        let adjusted = integration.calculate_beta_adjusted_pnl(1000.0, "BTC", 0.05);
        
        assert!((adjusted.raw_pnl - 1000.0).abs() < 1e-10);
        assert!(adjusted.beta.is_finite());
    }
    
    #[test]
    fn test_hedging_strategy() {
        let strategy = HedgingStrategy::new(0.0, 0.1, 0.01);
        
        // Should need rebalance when beta drifts
        assert!(strategy.needs_rebalance(0.15));
        assert!(!strategy.needs_rebalance(0.05));
        
        // Calculate rebalance size
        let size = strategy.calculate_rebalance_size(0.2, 100_000.0, 50_000.0);
        assert!((size - (-0.4)).abs() < 0.01); // Need to short 0.4 units
    }
    
    #[test]
    fn test_hedge_recommendation() {
        let integration = RiskBusIntegration::new(20, 100_000.0);
        
        // Pre-populate with some data
        for i in 0..30 {
            integration.update_beta("BTC", 0.015 * i as f64, 0.01 * i as f64);
        }
        
        let rec = integration.get_hedge_recommendation(0.0, 100_000.0);
        
        assert!(rec.confidence > 0.0);
        assert!(rec.current_beta.is_finite());
    }
}
