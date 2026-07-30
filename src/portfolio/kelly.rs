//! Portfolio Optimization - Fractional Kelly Criterion
//! 
//! Implements Fractional Kelly Criterion sizing adjusted dynamically by real-time
//! volatility and win-rate. Reads historical accuracy metrics from SOUL.md to scale
//! position sizes aggressively during high-confidence regimes.

use std::collections::HashMap;

use tracing::{debug, info, warn};

/// Kelly criterion configuration
#[derive(Debug, Clone)]
pub struct KellyConfig {
    /// Fraction of full Kelly (0.5 = half Kelly)
    pub kelly_fraction: f64,
    /// Maximum position size as fraction of portfolio
    pub max_position_size: f64,
    /// Minimum position size as fraction of portfolio
    pub min_position_size: f64,
    /// Volatility scaling enabled
    pub enable_volatility_scaling: bool,
    /// Target portfolio volatility (annualized)
    pub target_volatility: f64,
    /// Lookback period for win rate calculation (number of trades)
    pub win_rate_lookback: usize,
    /// Confidence threshold for aggressive sizing
    pub high_confidence_threshold: f64,
}

impl Default for KellyConfig {
    fn default() -> Self {
        Self {
            kelly_fraction: 0.25, // Quarter Kelly for safety
            max_position_size: 0.20, // Max 20% per position
            min_position_size: 0.01, // Min 1% per position
            enable_volatility_scaling: true,
            target_volatility: 0.15, // 15% annualized
            win_rate_lookback: 100,
            high_confidence_threshold: 0.65,
        }
    }
}

/// Historical trade result for Kelly calculation
#[derive(Debug, Clone)]
pub struct TradeResult {
    pub pnl: f64,
    pub is_win: bool,
    pub confidence: f64,
    pub timestamp_ns: u64,
}

/// Kelly sizing result
#[derive(Debug, Clone)]
pub struct KellySizing {
    pub recommended_size: f64,
    pub kelly_criterion: f64,
    pub win_rate: f64,
    pub avg_win: f64,
    pub avg_loss: f64,
    pub volatility_adjustment: f64,
    pub confidence_adjustment: f64,
    pub final_size: f64,
}

/// Fractional Kelly calculator
pub struct KellyCalculator {
    config: KellyConfig,
    trade_history: Vec<TradeResult>,
    current_volatility: f64,
    soul_metrics: SoulMetrics,
}

/// Metrics loaded from SOUL.md
#[derive(Debug, Clone, Default)]
pub struct SoulMetrics {
    pub historical_win_rate: f64,
    pub avg_profit_factor: f64,
    pub max_consecutive_wins: usize,
    pub max_consecutive_losses: usize,
    pub high_confidence_accuracy: f64,
    pub low_confidence_accuracy: f64,
    pub regime_performance: HashMap<String, RegimePerformance>,
}

#[derive(Debug, Clone)]
pub struct RegimePerformance {
    pub win_rate: f64,
    pub sharpe: f64,
    pub num_trades: usize,
}

impl KellyCalculator {
    /// Create a new Kelly calculator
    pub fn new(config: KellyConfig) -> Self {
        Self {
            config,
            trade_history: Vec::new(),
            current_volatility: 0.15,
            soul_metrics: SoulMetrics::default(),
        }
    }

    /// Load metrics from SOUL.md file
    pub fn load_soul_metrics(&mut self, soul_content: &str) {
        // Parse SOUL.md for historical metrics
        // Expected format in SOUL.md:
        // ## Performance Metrics
        // - Win Rate: XX%
        // - Profit Factor: X.XX
        // etc.
        
        for line in soul_content.lines() {
            if line.contains("Win Rate") {
                if let Some(rate) = extract_percentage(line) {
                    self.soul_metrics.historical_win_rate = rate / 100.0;
                }
            } else if line.contains("Profit Factor") {
                if let Some(pf) = extract_float(line) {
                    self.soul_metrics.avg_profit_factor = pf;
                }
            } else if line.contains("High Confidence") {
                if let Some(acc) = extract_percentage(line) {
                    self.soul_metrics.high_confidence_accuracy = acc / 100.0;
                }
            }
        }

        info!(
            "Loaded SOUL metrics: win_rate={:.2}, profit_factor={:.2}",
            self.soul_metrics.historical_win_rate,
            self.soul_metrics.avg_profit_factor
        );
    }

    /// Add a trade result to history
    pub fn add_trade(&mut self, result: TradeResult) {
        self.trade_history.push(result);
        
        // Trim history to lookback window
        if self.trade_history.len() > self.config.win_rate_lookback + 100 {
            self.trade_history.drain(0..self.trade_history.len() - self.config.win_rate_lookback);
        }
    }

    /// Calculate optimal position size using Fractional Kelly
    pub fn calculate_position_size(
        &self,
        portfolio_value: f64,
        signal_confidence: f64,
        asset_volatility: f64,
    ) -> KellySizing {
        // Calculate recent win rate and payoff ratio
        let (win_rate, avg_win, avg_loss, win_count, loss_count) = self.calculate_statistics();

        // Standard Kelly formula: f* = W - (1-W)/R
        // Where W = win probability, R = win/loss ratio
        let kelly_criterion = if loss_count > 0 && avg_loss > 0.0 {
            let payoff_ratio = avg_win / avg_loss;
            win_rate - ((1.0 - win_rate) / payoff_ratio)
        } else if win_rate > 0.0 {
            // No losses yet, use conservative estimate
            win_rate * 0.5
        } else {
            0.0
        };

        // Apply fractional Kelly
        let fractional_kelly = kelly_criterion * self.config.kelly_fraction;

        // Volatility adjustment
        let vol_adjustment = if self.config.enable_volatility_scaling {
            if asset_volatility > 0.0 {
                self.config.target_volatility / asset_volatility.max(0.01)
            } else {
                1.0
            }
        } else {
            1.0
        };

        // Confidence adjustment based on SOUL metrics
        let conf_adjustment = self.calculate_confidence_adjustment(signal_confidence);

        // Calculate final size
        let mut recommended_size = fractional_kelly * vol_adjustment * conf_adjustment;

        // Apply bounds
        recommended_size = recommended_size.clamp(
            self.config.min_position_size,
            self.config.max_position_size,
        );

        // Scale by portfolio value
        let final_size = recommended_size * portfolio_value;

        KellySizing {
            recommended_size,
            kelly_criterion,
            win_rate,
            avg_win,
            avg_loss,
            volatility_adjustment: vol_adjustment,
            confidence_adjustment: conf_adjustment,
            final_size,
        }
    }

    /// Calculate statistics from recent trade history
    fn calculate_statistics(&self) -> (f64, f64, f64, usize, usize) {
        if self.trade_history.is_empty() {
            return (0.5, 0.0, 0.0, 0, 0);
        }

        let wins: Vec<&TradeResult> = self.trade_history.iter().filter(|t| t.is_win).collect();
        let losses: Vec<&TradeResult> = self.trade_history.iter().filter(|t| !t.is_win).collect();

        let win_count = wins.len();
        let loss_count = losses.len();
        let total = win_count + loss_count;

        let win_rate = if total > 0 {
            win_count as f64 / total as f64
        } else {
            0.5
        };

        let avg_win = if !wins.is_empty() {
            wins.iter().map(|t| t.pnl.abs()).sum::<f64>() / wins.len() as f64
        } else {
            0.0
        };

        let avg_loss = if !losses.is_empty() {
            losses.iter().map(|t| t.pnl.abs()).sum::<f64>() / losses.len() as f64
        } else {
            0.0
        };

        (win_rate, avg_win, avg_loss, win_count, loss_count)
    }

    /// Calculate confidence adjustment based on SOUL metrics
    fn calculate_confidence_adjustment(&self, signal_confidence: f64) -> f64 {
        // If signal confidence exceeds threshold, scale up
        if signal_confidence >= self.config.high_confidence_threshold {
            // Use high confidence accuracy from SOUL
            let soul_multiplier = if self.soul_metrics.high_confidence_accuracy > 0.0 {
                self.soul_metrics.high_confidence_accuracy / 0.5 // Normalize around 50%
            } else {
                1.0
            };
            
            1.0 + (signal_confidence - self.config.high_confidence_threshold) * soul_multiplier
        } else {
            // Reduce size for low confidence
            let low_conf_acc = if self.soul_metrics.low_confidence_accuracy > 0.0 {
                self.soul_metrics.low_confidence_accuracy
            } else {
                0.5
            };
            
            (signal_confidence / 0.5) * low_conf_acc
        }.clamp(0.5, 2.0)
    }

    /// Update current volatility estimate
    pub fn update_volatility(&mut self, volatility: f64) {
        self.current_volatility = volatility;
    }

    /// Get current regime recommendation
    pub fn get_regime_recommendation(&self) -> RegimeRecommendation {
        let (win_rate, _, _, _, _) = self.calculate_statistics();
        
        // Determine regime based on recent performance
        let regime = if win_rate > 0.6 {
            "hot"
        } else if win_rate < 0.4 {
            "cold"
        } else {
            "normal"
        };

        // Check SOUL metrics for regime-specific advice
        let soul_regime = self.soul_metrics.regime_performance.get(regime);
        
        RegimeRecommendation {
            regime: regime.to_string(),
            suggested_kelly_fraction: match regime {
                "hot" => self.config.kelly_fraction * 1.5,
                "cold" => self.config.kelly_fraction * 0.5,
                _ => self.config.kelly_fraction,
            },
            reasoning: format!(
                "Current win rate {:.1}% suggests {} regime",
                win_rate * 100.0,
                regime
            ),
        }
    }
}

/// Recommendation for current market regime
#[derive(Debug, Clone)]
pub struct RegimeRecommendation {
    pub regime: String,
    pub suggested_kelly_fraction: f64,
    pub reasoning: String,
}

/// Helper functions for parsing SOUL.md
fn extract_percentage(line: &str) -> Option<f64> {
    for part in line.split_whitespace() {
        if part.ends_with('%') {
            return part.trim_end_matches('%').parse().ok();
        }
    }
    None
}

fn extract_float(line: &str) -> Option<f64> {
    for part in line.split_whitespace() {
        if let Ok(val) = part.parse::<f64>() {
            return Some(val);
        }
    }
    None
}

/// Multi-asset Kelly optimizer
pub struct MultiAssetKelly {
    calculators: HashMap<String, KellyCalculator>,
    correlation_matrix: HashMap<(String, String), f64>,
}

impl MultiAssetKelly {
    pub fn new() -> Self {
        Self {
            calculators: HashMap::new(),
            correlation_matrix: HashMap::new(),
        }
    }

    /// Add an asset to track
    pub fn add_asset(&mut self, symbol: &str, config: KellyConfig) {
        self.calculators.insert(symbol.to_string(), KellyCalculator::new(config));
    }

    /// Calculate optimal multi-asset allocation
    pub fn calculate_allocation(
        &self,
        portfolio_value: f64,
        signals: &HashMap<String, Signal>,
    ) -> AllocationResult {
        let mut allocations = HashMap::new();
        let mut total_allocated = 0.0;

        for (symbol, signal) in signals {
            if let Some(calculator) = self.calculators.get(symbol) {
                let sizing = calculator.calculate_position_size(
                    portfolio_value,
                    signal.confidence,
                    signal.volatility,
                );

                allocations.insert(symbol.clone(), AssetAllocation {
                    symbol: symbol.clone(),
                    size: sizing.final_size,
                    side: signal.side,
                    confidence: signal.confidence,
                });

                total_allocated += sizing.final_size;
            }
        }

        // Normalize if over-allocated
        if total_allocated > portfolio_value * 0.95 {
            let scale = (portfolio_value * 0.95) / total_allocated;
            for alloc in allocations.values_mut() {
                alloc.size *= scale;
            }
        }

        AllocationResult {
            allocations,
            total_allocated,
            remaining_cash: portfolio_value - total_allocated,
        }
    }

    /// Set correlation between assets
    pub fn set_correlation(&mut self, asset1: &str, asset2: &str, correlation: f64) {
        self.correlation_matrix.insert(
            (asset1.to_string(), asset2.to_string()),
            correlation.clamp(-1.0, 1.0),
        );
    }
}

#[derive(Debug, Clone)]
pub struct Signal {
    pub side: PositionSide,
    pub confidence: f64,
    pub volatility: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PositionSide {
    Long,
    Short,
}

#[derive(Debug, Clone)]
pub struct AllocationResult {
    pub allocations: HashMap<String, AssetAllocation>,
    pub total_allocated: f64,
    pub remaining_cash: f64,
}

#[derive(Debug, Clone)]
pub struct AssetAllocation {
    pub symbol: String,
    pub size: f64,
    pub side: PositionSide,
    pub confidence: f64,
}

impl Default for MultiAssetKelly {
    fn default() -> Self {
        Self::new()
    }
}
