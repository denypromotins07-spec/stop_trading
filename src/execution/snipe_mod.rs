//! Aggressive Execution Module Root
//! 
//! Integrates sniping tactics with pre-trade risk bus and L3 tracker.

pub mod sniper;
pub mod sweeper;

pub use sniper::{IcebergSniper, IcebergCandidate, IcebergState, SnipeOrder, Urgency, L3TradeTick};
pub use sweeper::{LiquiditySweeper, OrderBookSnapshot, OrderBookLevel, SweepResult, SweepUrgency, CascadeDetection};

/// Combined aggressive execution engine
pub struct AggressiveExecution {
    pub sniper: IcebergSniper,
    pub sweeper: LiquiditySweeper,
}

impl AggressiveExecution {
    pub const fn new() -> Self {
        Self {
            sniper: IcebergSniper::new(),
            sweeper: LiquiditySweeper::new(),
        }
    }
    
    /// Process trade tick and check for opportunities
    pub fn process_tick(&mut self, tick: L3TradeTick) -> Option<ExecutionAction> {
        // Record tick for iceberg detection
        self.sniper.record_trade(tick);
        
        // Check for snipe opportunity
        if let Some(snipe_order) = self.sniper.generate_snipe_order() {
            return Some(ExecutionAction::Snipe(snipe_order));
        }
        
        None
    }
    
    /// Execute sweep based on current book state
    pub fn execute_sweep(&self, side: u8, target_size: i64) -> Option<SweepResult> {
        let result = self.sweeper.calculate_sweep(side, target_size);
        
        if result.total_size > 0 && result.slippage_bps < 100.0 {
            Some(result)
        } else {
            None
        }
    }
    
    /// Get best aggressive opportunity
    pub fn get_best_opportunity(&self) -> Option<AggressiveOpportunity> {
        // Check iceberg snipe first
        if let Some(cand) = self.sniper.get_best_opportunity() {
            return Some(AggressiveOpportunity::Iceberg(cand.clone()));
        }
        
        // Check cascade opportunity
        if let Some(cascade) = self.sweeper.detect_cascade(1) {
            return Some(AggressiveOpportunity::Cascade(cascade));
        }
        
        None
    }
}

/// Execution action types
#[derive(Debug, Clone)]
pub enum ExecutionAction {
    Snipe(SnipeOrder),
    Sweep(SweepResult),
    Wait,
}

/// Aggressive opportunity types
#[derive(Debug, Clone)]
pub enum AggressiveOpportunity {
    Iceberg(IcebergCandidate),
    Cascade(CascadeDetection),
}

/// Pre-trade risk checks for aggressive execution
pub struct PreTradeRisk {
    max_single_order_size: i64,
    max_daily_volume: i64,
    executed_today: i64,
    enabled: bool,
}

impl PreTradeRisk {
    pub const fn new() -> Self {
        Self {
            max_single_order_size: 10000,
            max_daily_volume: 1000000,
            executed_today: 0,
            enabled: true,
        }
    }
    
    pub fn check_order(&self, size: i64) -> bool {
        if !self.enabled {
            return true;
        }
        
        if size > self.max_single_order_size {
            return false;
        }
        
        if self.executed_today + size > self.max_daily_volume {
            return false;
        }
        
        true
    }
    
    pub fn record_execution(&mut self, size: i64) {
        self.executed_today += size;
    }
    
    pub fn reset_daily(&mut self) {
        self.executed_today = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_aggressive_execution() {
        let mut exec = AggressiveExecution::new();
        
        // Simulate iceberg pattern
        let base_time = 1000000000u64;
        for i in 0..10 {
            let _ = exec.process_tick(L3TradeTick {
                price: 50000 << 48,
                size: 100,
                aggressor_side: 1,
                timestamp_ns: base_time + i * 1000000,
                trade_id: i as u64,
            });
        }
        
        // Should detect opportunity
        let opp = exec.get_best_opportunity();
        assert!(opp.is_some());
        
        match opp.unwrap() {
            AggressiveOpportunity::Iceberg(_) => {},
            _ => panic!("Expected iceberg opportunity"),
        }
    }
    
    #[test]
    fn test_pretrade_risk() {
        let mut risk = PreTradeRisk::new();
        
        assert!(risk.check_order(5000));
        assert!(!risk.check_order(15000)); // Exceeds max single order
        
        risk.record_execution(500000);
        assert!(!risk.check_order(600000)); // Would exceed daily limit
    }
}
