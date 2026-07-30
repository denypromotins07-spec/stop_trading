//! Real-time stress testing engine applying historical crash profiles.
//! 
//! Simulates instantaneous portfolio liquidation under extreme spread-widening
//! and liquidity-evaporation scenarios (LUNA, FTX, March 2020, etc.)

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};

/// Historical crash scenario definitions
#[derive(Debug, Clone)]
pub struct CrashScenario {
    /// Scenario name
    pub name: &'static str,
    /// Date of the crash
    pub date: &'static str,
    /// Maximum drawdown during crash
    pub max_drawdown: f64,
    /// Duration in days
    pub duration_days: usize,
    /// Volatility multiplier
    pub vol_multiplier: f64,
    /// Liquidity reduction factor (0.0 to 1.0)
    pub liquidity_factor: f64,
    /// Spread widening factor
    pub spread_multiplier: f64,
    /// Correlation breakdown (correlations go to 1.0)
    pub correlation_shift: f64,
}

/// Predefined historical crash scenarios
pub mod scenarios {
    use super::CrashScenario;
    
    /// LUNA/UST collapse (May 2022)
    pub const LUNA: CrashScenario = CrashScenario {
        name: "LUNA Collapse",
        date: "2022-05-09",
        max_drawdown: -0.95,
        duration_days: 7,
        vol_multiplier: 10.0,
        liquidity_factor: 0.1,
        spread_multiplier: 50.0,
        correlation_shift: 0.8,
    };
    
    /// FTX collapse (November 2022)
    pub const FTX: CrashScenario = CrashScenario {
        name: "FTX Collapse",
        date: "2022-11-08",
        max_drawdown: -0.25,
        duration_days: 14,
        vol_multiplier: 5.0,
        liquidity_factor: 0.3,
        spread_multiplier: 10.0,
        correlation_shift: 0.5,
    };
    
    /// COVID March 2020
    pub const COVID_MARCH_2020: CrashScenario = CrashScenario {
        name: "COVID March 2020",
        date: "2020-03-12",
        max_drawdown: -0.35,
        duration_days: 23,
        vol_multiplier: 4.0,
        liquidity_factor: 0.4,
        spread_multiplier: 8.0,
        correlation_shift: 0.6,
    };
    
    /// Black Monday 1987
    pub const BLACK_MONDAY: CrashScenario = CrashScenario {
        name: "Black Monday",
        date: "1987-10-19",
        max_drawdown: -0.22,
        duration_days: 3,
        vol_multiplier: 8.0,
        liquidity_factor: 0.2,
        spread_multiplier: 15.0,
        correlation_shift: 0.7,
    };
    
    /// Flash Crash 2010
    pub const FLASH_CRASH: CrashScenario = CrashScenario {
        name: "Flash Crash",
        date: "2010-05-06",
        max_drawdown: -0.10,
        duration_days: 1,
        vol_multiplier: 20.0,
        liquidity_factor: 0.05,
        spread_multiplier: 30.0,
        correlation_shift: 0.4,
    };
    
    /// All predefined scenarios
    pub const ALL: &[&CrashScenario] = &[&LUNA, &FTX, &COVID_MARCH_2020, &BLACK_MONDAY, &FLASH_CRASH];
}

/// Position data for stress testing
#[derive(Debug, Clone)]
pub struct StressPosition {
    /// Asset identifier
    pub asset_id: String,
    /// Position size (positive for long, negative for short)
    pub size: f64,
    /// Entry price
    pub entry_price: f64,
    /// Current price
    pub current_price: f64,
    /// Asset volatility (annualized)
    pub volatility: f64,
    /// Correlation with BTC
    pub btc_correlation: f64,
}

/// Stress test result for a single position
#[derive(Debug, Clone)]
pub struct PositionStressResult {
    /// Asset ID
    pub asset_id: String,
    /// Stressed P&L
    pub stressed_pnl: f64,
    /// Stressed value
    pub stressed_value: f64,
    /// Liquidation loss estimate
    pub liquidation_loss: f64,
    /// Margin call threshold breached
    pub margin_call: bool,
}

/// Portfolio stress test result
#[derive(Debug, Clone)]
pub struct StressTestResult {
    /// Scenario used
    pub scenario_name: String,
    /// Total portfolio value before stress
    pub initial_value: f64,
    /// Total portfolio value after stress
    pub stressed_value: f64,
    /// Total P&L impact
    pub total_pnl: f64,
    /// Percentage drawdown
    pub drawdown_pct: f64,
    /// Individual position results
    pub position_results: Vec<PositionStressResult>,
    /// Estimated liquidation cost
    pub liquidation_cost: f64,
    /// Timestamp
    pub timestamp_ns: u64,
}

impl StressTestResult {
    /// Check if result exceeds risk tolerance
    pub fn exceeds_tolerance(&self, tolerance: f64) -> bool {
        self.drawdown_pct.abs() > tolerance
    }
}

/// Real-time stress testing engine
pub struct StressTestEngine {
    /// Current positions
    positions: Vec<StressPosition>,
    /// Portfolio value
    portfolio_value: f64,
    /// Whether stress testing is enabled
    enabled: AtomicBool,
    /// Last stress test timestamp
    last_test_ns: AtomicU64,
    /// Test counter
    test_count: AtomicU64,
}

impl StressTestEngine {
    /// Create a new stress test engine
    pub fn new(portfolio_value: f64) -> Self {
        Self {
            positions: Vec::new(),
            portfolio_value,
            enabled: AtomicBool::new(true),
            last_test_ns: AtomicU64::new(0),
            test_count: AtomicU64::new(0),
        }
    }
    
    /// Add a position to stress test
    pub fn add_position(&mut self, position: StressPosition) {
        self.positions.push(position);
    }
    
    /// Remove a position
    pub fn remove_position(&mut self, asset_id: &str) {
        self.positions.retain(|p| p.asset_id != asset_id);
    }
    
    /// Clear all positions
    pub fn clear_positions(&mut self) {
        self.positions.clear();
    }
    
    /// Run stress test with specified scenario
    pub fn run_stress_test(&self, scenario: &CrashScenario) -> StressTestResult {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        
        let mut position_results = Vec::with_capacity(self.positions.len());
        let mut total_stressed_value = 0.0;
        let mut total_liquidation_cost = 0.0;
        
        for position in &self.positions {
            // Apply stress factors
            let price_shock = scenario.max_drawdown * scenario.vol_multiplier * position.volatility;
            let stressed_price = position.current_price * (1.0 + price_shock);
            
            // Calculate P&L
            let pnl = if position.size > 0.0 {
                (stressed_price - position.entry_price) * position.size
            } else {
                (position.entry_price - stressed_price) * position.size.abs()
            };
            
            // Calculate stressed position value
            let stressed_value = stressed_price * position.size.abs();
            total_stressed_value += stressed_value;
            
            // Estimate liquidation cost based on liquidity factor and spread
            let base_liquidation_cost = position.current_price * position.size.abs() * 0.001; // 0.1% base
            let adjusted_liquidation_cost = base_liquidation_cost 
                / scenario.liquidity_factor 
                * scenario.spread_multiplier;
            total_liquidation_cost += adjusted_liquidation_cost;
            
            // Check for margin call (simplified)
            let margin_call = pnl < -(position.current_price * position.size.abs() * 0.2); // 20% loss triggers
            
            position_results.push(PositionStressResult {
                asset_id: position.asset_id.clone(),
                stressed_pnl: pnl,
                stressed_value,
                liquidation_loss: adjusted_liquidation_cost,
                margin_call,
            });
        }
        
        let total_pnl: f64 = position_results.iter().map(|r| r.stressed_pnl).sum();
        let drawdown_pct = total_pnl / self.portfolio_value;
        
        StressTestResult {
            scenario_name: scenario.name.to_string(),
            initial_value: self.portfolio_value,
            stressed_value: self.portfolio_value + total_pnl - total_liquidation_cost,
            total_pnl,
            drawdown_pct,
            position_results,
            liquidation_cost: total_liquidation_cost,
            timestamp_ns: timestamp,
        }
    }
    
    /// Run all predefined stress scenarios
    pub fn run_all_scenarios(&self) -> Vec<StressTestResult> {
        scenarios::ALL.iter()
            .map(|&scenario| self.run_stress_test(scenario))
            .collect()
    }
    
    /// Find worst-case scenario
    pub fn find_worst_case(&self) -> Option<(String, f64)> {
        let results = self.run_all_scenarios();
        
        results.into_iter()
            .max_by(|a, b| a.drawdown_pct.partial_cmp(&b.drawdown_pct).unwrap_or(std::cmp::Ordering::Equal))
            .map(|r| (r.scenario_name, r.drawdown_pct))
    }
    
    /// Update portfolio value
    pub fn update_portfolio_value(&mut self, value: f64) {
        self.portfolio_value = value;
    }
    
    /// Get number of positions
    pub fn position_count(&self) -> usize {
        self.positions.len()
    }
    
    /// Enable/disable stress testing
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Relaxed);
    }
    
    /// Check if enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }
}

/// Continuous stress monitor
pub struct StressMonitor {
    engine: StressTestEngine,
    /// Alert threshold
    alert_threshold: f64,
    /// Alert counter
    alert_count: AtomicU64,
}

impl StressMonitor {
    /// Create a new stress monitor
    pub fn new(portfolio_value: f64, alert_threshold: f64) -> Self {
        Self {
            engine: StressTestEngine::new(portfolio_value),
            alert_threshold,
            alert_count: AtomicU64::new(0),
        }
    }
    
    /// Check current positions against stress scenarios
    pub fn check(&self) -> StressAlert {
        let results = self.engine.run_all_scenarios();
        
        let worst = results.iter()
            .max_by(|a, b| a.drawdown_pct.partial_cmp(&b.drawdown_pct).unwrap_or(std::cmp::Ordering::Equal));
        
        if let Some(result) = worst {
            if result.exceeds_tolerance(self.alert_threshold) {
                self.alert_count.fetch_add(1, Ordering::Relaxed);
                return StressAlert {
                    triggered: true,
                    scenario: result.scenario_name.clone(),
                    drawdown: result.drawdown_pct,
                    recommended_action: self.get_recommended_action(result),
                };
            }
        }
        
        StressAlert::none()
    }
    
    /// Get recommended action based on stress level
    fn get_recommended_action(&self, result: &StressTestResult) -> &'static str {
        if result.drawdown_pct < -0.50 {
            "IMMEDIATE LIQUIDATION"
        } else if result.drawdown_pct < -0.30 {
            "REDUCE EXPOSURE BY 50%"
        } else if result.drawdown_pct < -0.15 {
            "HEDGE WITH OPTIONS/FUTURES"
        } else {
            "MONITOR CLOSELY"
        }
    }
    
    /// Get the underlying engine
    pub fn engine(&self) -> &StressTestEngine {
        &self.engine
    }
    
    /// Get mutable engine
    pub fn engine_mut(&mut self) -> &mut StressTestEngine {
        &mut self.engine
    }
    
    /// Get alert count
    pub fn alert_count(&self) -> u64 {
        self.alert_count.load(Ordering::Relaxed)
    }
}

/// Stress alert notification
#[derive(Debug, Clone)]
pub struct StressAlert {
    /// Whether alert was triggered
    pub triggered: bool,
    /// Triggering scenario
    pub scenario: String,
    /// Expected drawdown
    pub drawdown: f64,
    /// Recommended action
    pub recommended_action: &'static str,
}

impl StressAlert {
    /// Create empty alert
    pub fn none() -> Self {
        Self {
            triggered: false,
            scenario: String::new(),
            drawdown: 0.0,
            recommended_action: "NONE",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_stress_engine() {
        let mut engine = StressTestEngine::new(1_000_000.0);
        
        engine.add_position(StressPosition {
            asset_id: "BTC".to_string(),
            size: 10.0,
            entry_price: 50000.0,
            current_price: 55000.0,
            volatility: 0.02,
            btc_correlation: 1.0,
        });
        
        let result = engine.run_stress_test(&scenarios::LUNA);
        assert!(result.drawdown_pct < 0.0);
        assert!(!result.scenario_name.is_empty());
    }
    
    #[test]
    fn test_worst_case() {
        let mut engine = StressTestEngine::new(1_000_000.0);
        
        engine.add_position(StressPosition {
            asset_id: "ETH".to_string(),
            size: 100.0,
            entry_price: 3000.0,
            current_price: 3200.0,
            volatility: 0.03,
            btc_correlation: 0.8,
        });
        
        let worst = engine.find_worst_case();
        assert!(worst.is_some());
    }
}
