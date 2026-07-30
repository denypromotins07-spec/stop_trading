//! Macro Data Module Root
//! 
//! Coordinates macroeconomic data ingestion, correlation analysis,
//! and regime shift signal generation for the trading system.

pub mod ingestion;
pub mod correlation;

use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn};

/// Macro data manager coordinating all macro-related components
pub struct MacroDataManager {
    ingestor: Arc<RwLock<ingestion::MacroDataIngestor>>,
    correlation_engine: Arc<correlation::CorrelationEngine>,
}

impl MacroDataManager {
    /// Create a new macro data manager
    pub fn new() -> Self {
        Self {
            ingestor: Arc::new(RwLock::new(ingestion::MacroDataIngestor::new())),
            correlation_engine: Arc::new(correlation::CorrelationEngine::new()),
        }
    }
    
    /// Process an incoming macro event
    pub async fn process_event(
        &self,
        event: ingestion::MacroEvent,
    ) -> Option<ingestion::RegimeShiftSignal> {
        let ingestor = self.ingestor.read().await;
        ingestor.process_event(event)
    }
    
    /// Update market data snapshot
    pub async fn update_market_data(&self, snapshot: ingestion::MarketDataSnapshot) {
        let ingestor = self.ingestor.read().await;
        ingestor.update_market_data(snapshot);
    }
    
    /// Record correlation observation between two assets
    pub fn record_correlation(&self, asset1_price: f64, asset2_price: f64) -> Option<correlation::RegimeChange> {
        self.correlation_engine.record(asset1_price, asset2_price)
    }
    
    /// Get current BTC-Gold correlation
    pub fn get_btc_gold_correlation(&self) -> correlation::CorrelationResult {
        self.correlation_engine.get_correlation()
    }
    
    /// Get fast correlation estimate
    pub fn get_fast_correlation(&self) -> f64 {
        self.correlation_engine.get_fast_correlation()
    }
    
    /// Get slow correlation baseline
    pub fn get_slow_correlation(&self) -> f64 {
        self.correlation_engine.get_slow_correlation()
    }
    
    /// Get last CPI reading
    pub async fn get_last_cpi(&self) -> f64 {
        let ingestor = self.ingestor.read().await;
        ingestor.get_last_cpi()
    }
    
    /// Get last Fed rate decision
    pub async fn get_last_fed_rate(&self) -> i32 {
        let ingestor = self.ingestor.read().await;
        ingestor.get_last_fed_rate()
    }
    
    /// Get current market snapshot
    pub async fn get_market_snapshot(&self) -> Option<ingestion::MarketDataSnapshot> {
        let ingestor = self.ingestor.read().await;
        ingestor.get_market_snapshot()
    }
    
    /// Start background tasks for macro data processing
    pub async fn start(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        info!("Starting macro data manager");
        
        // Spawn task to monitor correlation breakdowns
        let corr_engine = self.correlation_engine.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(std::time::Duration::from_secs(60)).await;
                
                let fast = corr_engine.get_fast_correlation();
                let slow = corr_engine.get_slow_correlation();
                
                if (fast - slow).abs() > 0.3 {
                    warn!(
                        "Correlation breakdown detected: fast={:.3}, slow={:.3}",
                        fast, slow
                    );
                }
            }
        });
        
        Ok(())
    }
    
    /// Convert regime shift signal to normalized action
    pub fn normalize_regime_signal(signal: &ingestion::RegimeShiftSignal) -> RegimeAction {
        match (&signal.direction, &signal.recommended_action) {
            (ingestion::RegimeDirection::RiskOn, ingestion::RecommendedAction::IncreaseExposure) => {
                RegimeAction::AggressiveLong
            }
            (ingestion::RegimeDirection::RiskOn, _) => RegimeAction::ModerateLong,
            (ingestion::RegimeDirection::RiskOff, ingestion::RecommendedAction::DecreaseExposure) => {
                RegimeAction::AggressiveShort
            }
            (ingestion::RegimeDirection::RiskOff, ingestion::RecommendedAction::Hedge) => {
                RegimeAction::DefensiveHedge
            }
            (ingestion::RegimeDirection::RiskOff, _) => RegimeAction::ModerateShort,
            (ingestion::RegimeDirection::Neutral, _) => RegimeAction::Neutral,
        }
    }
}

impl Default for MacroDataManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Normalized regime action for the trading system
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RegimeAction {
    AggressiveLong,
    ModerateLong,
    Neutral,
    ModerateShort,
    AggressiveShort,
    DefensiveHedge,
}

impl RegimeAction {
    /// Get position size multiplier based on regime action
    pub fn position_multiplier(&self) -> f64 {
        match self {
            RegimeAction::AggressiveLong => 1.5,
            RegimeAction::ModerateLong => 1.2,
            RegimeAction::Neutral => 1.0,
            RegimeAction::ModerateShort => 0.8,
            RegimeAction::AggressiveShort => 0.5,
            RegimeAction::DefensiveHedge => 0.3,
        }
    }
    
    /// Get risk reduction factor
    pub fn risk_factor(&self) -> f64 {
        match self {
            RegimeAction::AggressiveLong | RegimeAction::AggressiveShort => 1.0,
            RegimeAction::ModerateLong | RegimeAction::ModerateShort => 0.7,
            RegimeAction::Neutral => 0.5,
            RegimeAction::DefensiveHedge => 0.3,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_manager_creation() {
        let _manager = MacroDataManager::new();
    }
    
    #[test]
    fn test_position_multipliers() {
        assert_eq!(RegimeAction::AggressiveLong.position_multiplier(), 1.5);
        assert_eq!(RegimeAction::DefensiveHedge.position_multiplier(), 0.3);
    }
    
    #[test]
    fn test_risk_factors() {
        assert_eq!(RegimeAction::Neutral.risk_factor(), 0.5);
        assert_eq!(RegimeAction::DefensiveHedge.risk_factor(), 0.3);
    }
}
