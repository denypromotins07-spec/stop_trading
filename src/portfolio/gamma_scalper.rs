//! Portfolio Gamma Scalper
//! 
//! Builds a gamma scalping logic engine that dynamically adjusts limit orders
//! around the current position to exploit mean-reverting micro-movements.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Gamma scalper configuration
#[derive(Debug, Clone)]
pub struct GammaScalperConfig {
    /// Target gamma exposure
    pub target_gamma: f64,
    /// Rebalance threshold (gamma units)
    pub rebalance_threshold: f64,
    /// Scalp profit target (bps)
    pub profit_target_bps: f64,
    /// Maximum scalp size
    pub max_scalp_size: f64,
    /// Minimum spread for scalp (bps)
    pub min_spread_bps: f64,
    /// Mean reversion lookback (ms)
    pub lookback_ms: u64,
}

impl Default for GammaScalperConfig {
    fn default() -> Self {
        Self {
            target_gamma: 0.0,
            rebalance_threshold: 0.1,
            profit_target_bps: 2.0,
            max_scalp_size: 100.0,
            min_spread_bps: 1.0,
            lookback_ms: 1000,
        }
    }
}

/// Position gamma information
#[derive(Debug, Clone)]
pub struct PositionGamma {
    pub symbol: String,
    pub options_gamma: f64,
    pub spot_position: f64,
    pub perp_position: f64,
    pub net_gamma: f64,
    pub gamma_pnl_sensitivity: f64,
    pub last_update_ns: u64,
}

/// Scalp opportunity
#[derive(Debug, Clone)]
pub struct ScalpOpportunity {
    pub symbol: String,
    pub direction: ScalpDirection,
    pub size: f64,
    pub entry_price: f64,
    pub target_price: f64,
    pub stop_price: f64,
    pub expected_profit_bps: f64,
    pub confidence: f64,
    pub expiry_ns: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalpDirection {
    Long,
    Short,
}

/// Gamma scalper engine
pub struct GammaScalper {
    config: GammaScalperConfig,
    positions: dashmap::DashMap<String, PositionGamma>,
    scalp_count: AtomicU64,
    total_scalp_pnl: AtomicU64, // Fixed point * 1000
    halted: AtomicBool,
}

impl GammaScalper {
    pub fn new(config: GammaScalperConfig) -> Self {
        Self {
            config,
            positions: dashmap::DashMap::new(),
            scalp_count: AtomicU64::new(0),
            total_scalp_pnl: AtomicU64::new(0),
            halted: AtomicBool::new(false),
        }
    }

    /// Update position gamma
    pub fn update_position(&self, gamma: PositionGamma) {
        self.positions.insert(gamma.symbol.clone(), gamma);
    }

    /// Detect scalp opportunities based on gamma positioning
    pub fn detect_opportunities(&self, current_price: f64, price_history: &[f64]) -> Vec<ScalpOpportunity> {
        if self.halted.load(Ordering::Relaxed) {
            return Vec::new();
        }

        let mut opportunities = Vec::new();

        for entry in self.positions.iter() {
            let pos = entry.value();
            
            // Only scalp if we have significant gamma
            if pos.net_gamma.abs() < self.config.rebalance_threshold {
                continue;
            }

            // Calculate mean reversion signal
            if let Some(signal) = self.calculate_mean_reversion(price_history) {
                let direction = if signal > 0.0 {
                    ScalpDirection::Long
                } else {
                    ScalpDirection::Short
                };

                let size = (pos.net_gamma.abs() * 100.0).clamp(1.0, self.config.max_scalp_size);
                let profit_bps = self.config.profit_target_bps;
                
                let target_price = if direction == ScalpDirection::Long {
                    current_price * (1.0 + profit_bps / 10000.0)
                } else {
                    current_price * (1.0 - profit_bps / 10000.0)
                };

                let stop_price = if direction == ScalpDirection::Long {
                    current_price * (1.0 - profit_bps / 5000.0)
                } else {
                    current_price * (1.0 + profit_bps / 5000.0)
                };

                opportunities.push(ScalpOpportunity {
                    symbol: pos.symbol.clone(),
                    direction,
                    size,
                    entry_price: current_price,
                    target_price,
                    stop_price,
                    expected_profit_bps: profit_bps,
                    confidence: signal.abs().min(1.0),
                    expiry_ns: timestamp_ns() + 1_000_000_000, // 1 second expiry
                });
            }
        }

        opportunities
    }

    /// Calculate mean reversion signal from price history
    fn calculate_mean_reversion(&self, prices: &[f64]) -> Option<f64> {
        if prices.len() < 10 {
            return None;
        }

        let mean: f64 = prices.iter().sum::<f64>() / prices.len() as f64;
        let current = prices[prices.len() - 1];
        
        let variance: f64 = prices.iter()
            .map(|p| (p - mean).powi(2))
            .sum::<f64>() / prices.len() as f64;
        
        let std_dev = variance.sqrt();
        if std_dev < 1e-10 {
            return Some(0.0);
        }

        // Z-score: negative means below mean (buy signal), positive means above (sell)
        let z_score = (current - mean) / std_dev;
        
        // Invert: if price is below mean, expect reversion up (positive signal)
        Some(-z_score / 3.0) // Normalize to roughly [-1, 1]
    }

    /// Execute a scalp trade
    pub fn execute_scalp(&self, opportunity: &ScalpOpportunity) -> Result<ScalpExecution, &'static str> {
        if self.halted.load(Ordering::Relaxed) {
            return Err("Scalper is halted");
        }

        if timestamp_ns() > opportunity.expiry_ns {
            return Err("Opportunity expired");
        }

        let execution = ScalpExecution {
            symbol: opportunity.symbol.clone(),
            direction: opportunity.direction,
            size: opportunity.size,
            entry_price: opportunity.entry_price,
            target_price: opportunity.target_price,
            stop_price: opportunity.stop_price,
            status: ScalpStatus::Active,
            timestamp_ns: timestamp_ns(),
        };

        self.scalp_count.fetch_add(1, Ordering::Relaxed);

        Ok(execution)
    }

    /// Get gamma summary
    pub fn get_gamma_summary(&self) -> GammaSummary {
        let mut total_gamma = 0.0;
        let mut positions = Vec::new();

        for entry in self.positions.iter() {
            total_gamma += entry.value().net_gamma;
            positions.push(entry.value().clone());
        }

        GammaSummary {
            total_gamma,
            positions,
            scalp_count: self.scalp_count.load(Ordering::Relaxed),
            total_pnl: self.total_scalp_pnl.load(Ordering::Relaxed) as f64 / 1000.0,
        }
    }

    /// Halt scalping
    pub fn halt(&self) {
        self.halted.store(true, Ordering::SeqCst);
    }

    /// Resume scalping
    pub fn resume(&self) {
        self.halted.store(false, Ordering::SeqCst);
    }
}

/// Scalp execution result
#[derive(Debug, Clone)]
pub struct ScalpExecution {
    pub symbol: String,
    pub direction: ScalpDirection,
    pub size: f64,
    pub entry_price: f64,
    pub target_price: f64,
    pub stop_price: f64,
    pub status: ScalpStatus,
    pub timestamp_ns: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalpStatus {
    Active,
    Filled,
    Cancelled,
    Stopped,
}

/// Gamma summary
#[derive(Debug, Clone)]
pub struct GammaSummary {
    pub total_gamma: f64,
    pub positions: Vec<PositionGamma>,
    pub scalp_count: u64,
    pub total_pnl: f64,
}

/// Get current timestamp in nanoseconds
#[inline]
fn timestamp_ns() -> u64 {
    Instant::now()
        .duration_since(Instant::now() - Duration::from_secs(1))
        .as_nanos() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gamma_scalper_basic() {
        let config = GammaScalperConfig::default();
        let scalper = GammaScalper::new(config);

        // Add a gamma position
        let pos = PositionGamma {
            symbol: "BTCUSD".to_string(),
            options_gamma: 0.5,
            spot_position: 0.0,
            perp_position: 0.0,
            net_gamma: 0.5,
            gamma_pnl_sensitivity: 50000.0,
            last_update_ns: timestamp_ns(),
        };

        scalper.update_position(pos);

        // Create price history showing mean reversion setup
        let prices: Vec<f64> = vec![50000.0, 50010.0, 50020.0, 50015.0, 50010.0, 50005.0, 50000.0, 49995.0, 49990.0, 49985.0];
        
        let opportunities = scalper.detect_opportunities(49985.0, &prices);
        
        // Should detect opportunity (price below mean, expect reversion up)
        assert!(!opportunities.is_empty());
        assert_eq!(opportunities[0].direction, ScalpDirection::Long);
    }

    #[test]
    fn test_mean_reversion_calculation() {
        let config = GammaScalperConfig::default();
        let scalper = GammaScalper::new(config);

        // Prices trending down, currently below mean
        let prices: Vec<f64> = vec![100.0, 100.0, 100.0, 100.0, 100.0, 95.0, 90.0, 85.0, 80.0, 75.0];
        
        let signal = scalper.calculate_mean_reversion(&prices);
        assert!(signal.is_some());
        assert!(signal.unwrap() > 0.0); // Positive signal = buy expectation
    }
}
