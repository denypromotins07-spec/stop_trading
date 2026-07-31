//! Inventory Penalty Calculator for Market Making
//!
//! Builds a dynamic inventory risk penalty calculator adjusting quotes based on real-time portfolio VaR.
//! Aggressively skews bid/ask quotes to flatten toxic inventory accumulations during high-volatility regime shifts.

use std::sync::atomic::{AtomicU64, AtomicBool, AtomicI64, Ordering};
use std::time::{Instant, Duration};
use crossbeam_channel::{bounded, Sender, Receiver};
use dashmap::DashMap;

/// Inventory state for a single asset
#[derive(Debug, Clone)]
pub struct InventoryState {
    /// Symbol/asset identifier
    pub symbol: String,
    /// Current position (positive = long, negative = short)
    pub position: f64,
    /// Average entry price
    pub avg_entry_price: f64,
    /// Current mark price
    pub current_price: f64,
    /// Position notional value
    pub notional_value: f64,
    /// Unrealized PnL
    pub unrealized_pnl: f64,
    /// Time-weighted inventory (for toxicity calculation)
    pub time_weighted_inventory: f64,
    /// Last update timestamp
    pub last_update_ns: u64,
}

impl InventoryState {
    pub fn new(symbol: String) -> Self {
        let now_ns = Instant::now().duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default().as_nanos() as u64;
        
        Self {
            symbol,
            position: 0.0,
            avg_entry_price: 0.0,
            current_price: 0.0,
            notional_value: 0.0,
            unrealized_pnl: 0.0,
            time_weighted_inventory: 0.0,
            last_update_ns: now_ns,
        }
    }

    /// Update position after trade
    pub fn update_position(&mut self, quantity: f64, price: f64, is_buy: bool) {
        let signed_qty = if is_buy { quantity } else { -quantity };
        let old_position = self.position;
        let new_position = old_position + signed_qty;
        
        // Update average entry price
        if (old_position > 0.0 && signed_qty > 0.0) || 
           (old_position < 0.0 && signed_qty < 0.0) {
            // Adding to position: weighted average
            let old_value = old_position.abs() * self.avg_entry_price;
            let new_value = signed_qty.abs() * price;
            self.avg_entry_price = (old_value + new_value) / new_position.abs();
        } else if new_position.abs() < old_position.abs() {
            // Reducing position: keep existing avg entry for remaining
            // If fully flattened and reversed, use trade price
            if old_position * new_position < 0.0 {
                self.avg_entry_price = price;
            }
        } else {
            self.avg_entry_price = price;
        }
        
        self.position = new_position;
        self.current_price = price;
        self.notional_value = new_position.abs() * price;
        self.unrealized_pnl = self.calculate_unrealized_pnl();
        
        let now_ns = Instant::now().duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default().as_nanos() as u64;
        self.last_update_ns = now_ns;
    }

    /// Update mark price
    pub fn update_mark_price(&mut self, price: f64) {
        self.current_price = price;
        self.notional_value = self.position.abs() * price;
        self.unrealized_pnl = self.calculate_unrealized_pnl();
    }

    /// Calculate unrealized PnL
    fn calculate_unrealized_pnl(&self) -> f64 {
        if self.position == 0.0 || self.avg_entry_price == 0.0 {
            return 0.0;
        }
        
        if self.position > 0.0 {
            (self.current_price - self.avg_entry_price) * self.position
        } else {
            (self.avg_entry_price - self.current_price) * self.position.abs()
        }
    }

    /// Get inventory skew direction (-1 to 1)
    pub fn get_skew_direction(&self) -> f64 {
        (self.position / 100.0).clamp(-1.0, 1.0)
    }
}

/// VaR calculation method
#[derive(Debug, Clone, Copy)]
pub enum VarMethod {
    /// Historical simulation
    Historical,
    /// Parametric (variance-covariance)
    Parametric,
    /// Monte Carlo simulation
    MonteCarlo,
    /// EWMA (Exponentially Weighted Moving Average)
    EWMA,
}

/// VaR configuration
#[derive(Debug, Clone)]
pub struct VarConfig {
    /// Confidence level (e.g., 0.95, 0.99)
    pub confidence_level: f64,
    /// Time horizon in days
    pub time_horizon_days: u32,
    /// Calculation method
    pub method: VarMethod,
    /// Lookback period for historical (days)
    pub lookback_days: u32,
    /// Decay factor for EWMA
    pub ewma_lambda: f64,
}

impl Default for VarConfig {
    fn default() -> Self {
        Self {
            confidence_level: 0.99,
            time_horizon_days: 1,
            method: VarMethod::EWMA,
            lookback_days: 252,
            ewma_lambda: 0.94,
        }
    }
}

/// Portfolio VaR state
#[derive(Debug, Clone)]
pub struct PortfolioVar {
    /// Total portfolio VaR
    pub total_var: f64,
    /// Marginal VaR per asset
    pub marginal_vars: DashMap<String, f64>,
    /// Component VaR per asset
    pub component_vars: DashMap<String, f64>,
    /// Diversification benefit
    pub diversification_benefit: f64,
    /// Last calculation timestamp
    pub last_calculation_ns: u64,
}

/// Inventory penalty parameters
#[derive(Debug, Clone)]
pub struct InventoryPenaltyParams {
    /// Base penalty coefficient
    pub base_penalty: f64,
    /// Volatility scaling factor
    pub vol_scaling: f64,
    /// VaR scaling factor
    pub var_scaling: f64,
    /// Toxicity threshold
    pub toxicity_threshold: f64,
    /// Maximum penalty multiplier
    pub max_penalty_multiplier: f64,
    /// Skew aggressiveness
    pub skew_aggressiveness: f64,
}

impl Default for InventoryPenaltyParams {
    fn default() -> Self {
        Self {
            base_penalty: 0.001,
            vol_scaling: 2.0,
            var_scaling: 1.5,
            toxicity_threshold: 0.7,
            max_penalty_multiplier: 10.0,
            skew_aggressiveness: 0.5,
        }
    }
}

/// Calculated inventory penalty
#[derive(Debug, Clone)]
pub struct InventoryPenalty {
    /// Base penalty in bps
    pub base_penalty_bps: f64,
    /// Volatility adjustment
    pub vol_adjustment: f64,
    /// VaR adjustment
    pub var_adjustment: f64,
    /// Toxicity adjustment
    pub toxicity_adjustment: f64,
    /// Total penalty multiplier
    pub total_multiplier: f64,
    /// Bid skew (bps to subtract from bid)
    pub bid_skew_bps: f64,
    /// Ask skew (bps to add to ask)
    pub ask_skew_bps: f64,
    /// Recommended action
    pub recommended_action: InventoryAction,
}

/// Recommended inventory action
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InventoryAction {
    /// Hold current position
    Hold,
    /// Reduce long exposure
    ReduceLong,
    /// Reduce short exposure
    ReduceShort,
    /// Flatten immediately
    FlattenImmediately,
    /// Reverse position
    Reverse,
}

/// Main inventory penalty calculator
pub struct InventoryPenaltyCalculator {
    /// Inventory states by symbol
    inventories: DashMap<String, InventoryState>,
    /// VaR configuration
    var_config: VarConfig,
    /// Penalty parameters
    params: InventoryPenaltyParams,
    /// Current portfolio VaR
    portfolio_var: PortfolioVar,
    /// Recent returns for VaR calculation (scaled by 1e6)
    recent_returns: DashMap<String, Box<[i64]>>,
    /// Current volatility estimates
    volatilities: DashMap<String, f64>,
    /// Correlation matrix (simplified)
    correlations: DashMap<(String, String), f64>,
    /// Is calculator active
    is_active: AtomicBool,
    /// Event channel
    event_tx: Sender<PenaltyEvent>,
    event_rx: Receiver<PenaltyEvent>,
}

/// Penalty events
#[derive(Debug, Clone)]
pub enum PenaltyEvent {
    /// Penalty calculated for symbol
    PenaltyCalculated {
        symbol: String,
        penalty: InventoryPenalty,
    },
    /// VaR limit breach
    VarLimitBreach {
        symbol: String,
        current_var: f64,
        limit: f64,
    },
    /// Toxic inventory warning
    ToxicInventoryWarning {
        symbol: String,
        toxicity_score: f64,
    },
    /// Action recommendation
    ActionRecommended {
        symbol: String,
        action: InventoryAction,
        urgency: u8,
    },
}

impl InventoryPenaltyCalculator {
    pub fn new(
        var_config: VarConfig,
        params: InventoryPenaltyParams,
        buffer_size: usize,
    ) -> Self {
        let (tx, rx) = bounded(buffer_size);

        Self {
            inventories: DashMap::new(),
            var_config,
            params,
            portfolio_var: PortfolioVar {
                total_var: 0.0,
                marginal_vars: DashMap::new(),
                component_vars: DashMap::new(),
                diversification_benefit: 0.0,
                last_calculation_ns: 0,
            },
            recent_returns: DashMap::new(),
            volatilities: DashMap::new(),
            correlations: DashMap::new(),
            is_active: AtomicBool::new(true),
            event_tx: tx,
            event_rx: rx,
        }
    }

    /// Initialize inventory tracking for a symbol
    pub fn initialize_symbol(&self, symbol: &str, initial_returns: &[f64]) {
        self.inventories.insert(symbol.to_string(), InventoryState::new(symbol.to_string()));
        
        // Store recent returns
        let returns_scaled: Box<[i64]> = initial_returns.iter()
            .map(|r| (r * 1e6) as i64)
            .collect::<Vec<_>>()
            .into_boxed_slice();
        self.recent_returns.insert(symbol.to_string(), returns_scaled);
        
        // Initial volatility estimate
        let vol = Self::calculate_ewma_volatility(initial_returns, self.var_config.ewma_lambda);
        self.volatilities.insert(symbol.to_string(), vol);
    }

    /// Calculate penalty for a symbol given market conditions
    pub fn calculate_penalty(&self, symbol: &str, current_price: f64) -> Option<InventoryPenalty> {
        if !self.is_active.load(Ordering::Relaxed) {
            return None;
        }

        let inventory = match self.inventories.get(symbol) {
            Some(inv) => inv.clone(),
            None => return None,
        };

        // Update mark price
        // Note: In production, would need mutable access
        // For now, we'll work with the cloned state

        let volatility = self.volatilities.get(symbol).copied().unwrap_or(0.6);
        
        // 1. Base penalty
        let base_penalty = self.params.base_penalty * 10000.0; // Convert to bps

        // 2. Volatility adjustment
        let vol_adjustment = (volatility / 0.6) * self.params.vol_scaling;

        // 3. VaR-based adjustment
        let position_var = self.calculate_position_var(&inventory, volatility);
        let var_adjustment = (position_var / 1000.0) * self.params.var_scaling;

        // 4. Toxicity adjustment
        let toxicity = self.calculate_toxicity(&inventory);
        let toxicity_adjustment = if toxicity > self.params.toxicity_threshold {
            ((toxicity - self.params.toxicity_threshold) / (1.0 - self.params.toxicity_threshold)) 
                * self.params.max_penalty_multiplier
        } else {
            1.0
        };

        // Total multiplier
        let total_multiplier = (1.0 + vol_adjustment + var_adjustment) * toxicity_adjustment;
        let total_multiplier = total_multiplier.min(self.params.max_penalty_multiplier);

        // Calculate skews based on inventory direction
        let skew_direction = inventory.get_skew_direction();
        let base_skew = base_penalty * total_multiplier * self.params.skew_aggressiveness;
        
        let (bid_skew, ask_skew, action) = if skew_direction > 0.3 {
            // Long inventory: skew bids down more aggressively
            let bid_skew = base_skew * (1.0 + skew_direction);
            let ask_skew = base_skew * (1.0 - skew_direction * 0.5);
            let action = if skew_direction > 0.8 {
                InventoryAction::FlattenImmediately
            } else if skew_direction > 0.5 {
                InventoryAction::ReduceLong
            } else {
                InventoryAction::Hold
            };
            (bid_skew, ask_skew, action)
        } else if skew_direction < -0.3 {
            // Short inventory: skew asks up more aggressively
            let bid_skew = base_skew * (1.0 + skew_direction.abs() * 0.5);
            let ask_skew = base_skew * (1.0 + skew_direction.abs());
            let action = if skew_direction < -0.8 {
                InventoryAction::FlattenImmediately
            } else if skew_direction < -0.5 {
                InventoryAction::ReduceShort
            } else {
                InventoryAction::Hold
            };
            (bid_skew, ask_skew, action)
        } else {
            // Neutral: symmetric skew
            (base_skew, base_skew, InventoryAction::Hold)
        };

        let penalty = InventoryPenalty {
            base_penalty_bps: base_penalty,
            vol_adjustment,
            var_adjustment,
            toxicity_adjustment,
            total_multiplier,
            bid_skew_bps: bid_skew,
            ask_skew_bps: ask_skew,
            recommended_action: action,
        };

        // Emit events
        let _ = self.event_tx.send(PenaltyEvent::PenaltyCalculated {
            symbol: symbol.to_string(),
            penalty: penalty.clone(),
        });

        if action != InventoryAction::Hold {
            let urgency = match action {
                InventoryAction::FlattenImmediately => 5,
                InventoryAction::ReduceLong | InventoryAction::ReduceShort => 3,
                InventoryAction::Reverse => 4,
                InventoryAction::Hold => 0,
            };
            let _ = self.event_tx.send(PenaltyEvent::ActionRecommended {
                symbol: symbol.to_string(),
                action,
                urgency,
            });
        }

        // Check VaR limits
        let var_limit = self.calculate_var_limit(&inventory);
        if position_var > var_limit {
            let _ = self.event_tx.send(PenaltyEvent::VarLimitBreach {
                symbol: symbol.to_string(),
                current_var: position_var,
                limit: var_limit,
            });
        }

        // Check toxicity
        if toxicity > self.params.toxicity_threshold {
            let _ = self.event_tx.send(PenaltyEvent::ToxicInventoryWarning {
                symbol: symbol.to_string(),
                toxicity_score: toxicity,
            });
        }

        Some(penalty)
    }

    /// Calculate position VaR using parametric method
    fn calculate_position_var(&self, inventory: &InventoryState, volatility: f64) -> f64 {
        // Simple parametric VaR: position_value * volatility * z_score * sqrt(time)
        let z_score = match self.var_config.confidence_level {
            0.99 => 2.33,
            0.95 => 1.65,
            0.90 => 1.28,
            _ => 2.33,
        };
        
        let time_sqrt = (self.var_config.time_horizon_days as f64).sqrt();
        let position_value = inventory.notional_value;
        
        position_value * volatility * z_score * time_sqrt
    }

    /// Calculate VaR limit for position
    fn calculate_var_limit(&self, inventory: &InventoryState) -> f64 {
        // Limit based on notional value (e.g., 5% of position)
        inventory.notional_value * 0.05
    }

    /// Calculate inventory toxicity score (0 to 1)
    fn calculate_toxicity(&self, inventory: &InventoryState) -> f64 {
        // Toxicity factors:
        // 1. Large absolute position
        // 2. Adverse price movement since entry
        // 3. Time held (longer = more toxic if underwater)
        
        let position_factor = (inventory.position.abs() / 100.0).min(1.0);
        
        let pnl_factor = if inventory.unrealized_pnl < 0.0 {
            ((-inventory.unrealized_pnl) / inventory.notional_value).min(0.5) * 2.0
        } else {
            0.0
        };
        
        let now_ns = Instant::now().duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default().as_nanos() as u64;
        let age_seconds = (now_ns - inventory.last_update_ns) as f64 / 1e9;
        let time_factor = (age_seconds / 3600.0).min(1.0); // Cap at 1 hour
        
        // Weighted combination
        let toxicity = 0.4 * position_factor + 0.4 * pnl_factor + 0.2 * time_factor;
        toxicity.min(1.0)
    }

    /// Calculate EWMA volatility
    fn calculate_ewma_volatility(returns: &[f64], lambda: f64) -> f64 {
        if returns.len() < 2 {
            return 0.6; // Default
        }

        let mut variance = 0.0;
        let mut weight = 1.0 - lambda;

        for i in (1..returns.len()).rev() {
            let ret = returns[i];
            variance += weight * ret * ret;
            weight *= lambda;
        }

        variance.sqrt()
    }

    /// Update volatility estimate with new return
    pub fn update_volatility(&self, symbol: &str, new_return: f64) {
        let current_vol = self.volatilities.get(symbol).copied().unwrap_or(0.6);
        let lambda = self.var_config.ewma_lambda;
        
        // EWMA update: sigma^2_t = lambda * sigma^2_{t-1} + (1-lambda) * r^2_{t-1}
        let new_variance = lambda * current_vol.powi(2) + (1.0 - lambda) * new_return.powi(2);
        let new_vol = new_variance.sqrt();
        
        self.volatilities.insert(symbol.to_string(), new_vol);
    }

    /// Update inventory after fill
    pub fn update_inventory(&self, symbol: &str, quantity: f64, price: f64, is_buy: bool) {
        if let Some(mut inv) = self.inventories.get_mut(symbol) {
            inv.update_position(quantity, price, is_buy);
        }
    }

    /// Get inventory state
    pub fn get_inventory(&self, symbol: &str) -> Option<InventoryState> {
        self.inventories.get(symbol).map(|i| i.clone())
    }

    /// Get current volatility
    pub fn get_volatility(&self, symbol: &str) -> f64 {
        self.volatilities.get(symbol).copied().unwrap_or(0.6)
    }

    /// Get event receiver
    pub fn get_event_receiver(&self) -> Receiver<PenaltyEvent> {
        self.event_rx.clone()
    }

    /// Deactivate calculator
    pub fn deactivate(&self) {
        self.is_active.store(false, Ordering::Relaxed);
    }

    /// Activate calculator
    pub fn activate(&self) {
        self.is_active.store(true, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inventory_state_updates() {
        let mut inv = InventoryState::new("BTCUSDT".to_string());
        
        assert_eq!(inv.position, 0.0);
        
        inv.update_position(10.0, 50000.0, true); // Buy 10 @ 50k
        assert_eq!(inv.position, 10.0);
        assert_eq!(inv.avg_entry_price, 50000.0);
        
        inv.update_position(5.0, 51000.0, true); // Buy 5 more @ 51k
        assert_eq!(inv.position, 15.0);
        assert!((inv.avg_entry_price - 50333.33).abs() < 1.0);
        
        inv.update_position(10.0, 50500.0, false); // Sell 10 @ 50.5k
        assert_eq!(inv.position, 5.0);
    }

    #[test]
    fn test_penalty_calculator_initialization() {
        let config = VarConfig::default();
        let params = InventoryPenaltyParams::default();
        
        let calc = InventoryPenaltyCalculator::new(config, params, 1000);
        
        assert!(calc.is_active.load(Ordering::Relaxed));
        
        calc.initialize_symbol("BTCUSDT", &[0.01, -0.02, 0.015, -0.01]);
        
        assert!(calc.get_inventory("BTCUSDT").is_some());
        assert!(calc.get_volatility("BTCUSDT") > 0.0);
    }

    #[test]
    fn test_penalty_calculation_long_position() {
        let config = VarConfig::default();
        let params = InventoryPenaltyParams::default();
        let calc = InventoryPenaltyCalculator::new(config, params, 1000);
        
        calc.initialize_symbol("BTCUSDT", &[0.01, -0.02, 0.015]);
        calc.update_inventory("BTCUSDT", 50.0, 50000.0, true); // Long 50 BTC
        
        let penalty = calc.calculate_penalty("BTCUSDT", 50000.0);
        assert!(penalty.is_some());
        
        let penalty = penalty.unwrap();
        assert!(penalty.bid_skew_bps > penalty.ask_skew_bps); // Should skew bids down
        assert_eq!(penalty.recommended_action, InventoryAction::Hold); // Not extreme enough
    }

    #[test]
    fn test_toxicity_calculation() {
        let config = VarConfig::default();
        let params = InventoryPenaltyParams::default();
        let calc = InventoryPenaltyCalculator::new(config, params, 1000);
        
        calc.initialize_symbol("BTCUSDT", &[0.01, -0.02, 0.015]);
        
        // Create losing position
        calc.update_inventory("BTCUSDT", 100.0, 50000.0, true); // Buy @ 50k
        
        // Price drops significantly
        let penalty = calc.calculate_penalty("BTCUSDT", 45000.0);
        assert!(penalty.is_some());
        
        let penalty = penalty.unwrap();
        assert!(penalty.toxicity_adjustment > 1.0); // Should have toxicity penalty
    }

    #[test]
    fn test_ewma_volatility() {
        let returns = vec![0.02, -0.03, 0.01, -0.015, 0.025];
        let vol = InventoryPenaltyCalculator::calculate_ewma_volatility(&returns, 0.94);
        
        assert!(vol > 0.0);
        assert!(vol < 0.1); // Should be reasonable
    }
}
