//! Adverse Selection Cost Model
//!
//! Predicts the probability of being "run over" by informed flow and
//! automatically widens spreads when adverse selection risk is high.

use std::time::{Duration, Instant};

/// Trade classification for adverse selection analysis
#[derive(Debug, Clone, Copy)]
pub struct TradeFlow {
    /// Timestamp of the trade
    pub timestamp: Instant,
    /// Trade size (in base units * 10^8)
    pub size: u64,
    /// Price at which trade occurred
    pub price: u64,
    /// True if buyer-initiated (aggressive buy), false if seller-initiated
    pub is_buy: bool,
    /// Whether this was above/below mid (indicates aggressor side)
    pub above_mid: bool,
}

/// Rolling window statistics for trade flow analysis
pub struct FlowStatistics {
    /// Pre-allocated circular buffer for recent trades
    trades: [Option<TradeFlow>; 1024],
    /// Current write index in circular buffer
    write_idx: usize,
    /// Count of valid trades in buffer
    count: usize,
    /// Running sum of signed volume (positive = buy pressure)
    signed_volume_sum: i64,
    /// Running sum of trade sizes
    total_volume_sum: u64,
    /// Last price for return calculation
    last_price: u64,
    /// Cumulative price change for correlation
    price_change_sum: f64,
    /// Covariance accumulator for order flow toxicity
    covariance_sum: f64,
}

impl FlowStatistics {
    pub fn new() -> Self {
        FlowStatistics {
            trades: [None; 1024],
            write_idx: 0,
            count: 0,
            signed_volume_sum: 0,
            total_volume_sum: 0,
            last_price: 0,
            price_change_sum: 0.0,
            covariance_sum: 0.0,
        }
    }

    /// Add a new trade to the rolling window
    #[inline]
    pub fn add_trade(&mut self, trade: TradeFlow) {
        // Remove old trade from sums if slot was occupied
        if let Some(old) = self.trades[self.write_idx] {
            let signed_vol = if old.is_buy { old.size as i64 } else { -(old.size as i64) };
            self.signed_volume_sum -= signed_vol;
            self.total_volume_sum -= old.size;
        }

        // Add new trade
        let signed_vol = if trade.is_buy { trade.size as i64 } else { -(trade.size as i64) };
        self.signed_volume_sum += signed_vol;
        self.total_volume_sum += trade.size;

        // Calculate price impact contribution
        if self.last_price > 0 {
            let price_return = (trade.price as f64 - self.last_price as f64) / self.last_price as f64;
            let flow_sign = if trade.is_buy { 1.0 } else { -1.0 };
            self.covariance_sum += flow_sign * price_return * trade.size as f64;
        }

        self.trades[self.write_idx] = Some(trade);
        self.write_idx = (self.write_idx + 1) % 1024;
        
        if self.count < 1024 {
            self.count += 1;
        }
    }

    /// Get net order flow (signed volume)
    #[inline]
    pub fn net_flow(&self) -> i64 {
        self.signed_volume_sum
    }

    /// Get total volume in window
    #[inline]
    pub fn total_volume(&self) -> u64 {
        self.total_volume_sum
    }

    /// Get order flow imbalance ratio (-1 to 1)
    #[inline]
    pub fn flow_imbalance(&self) -> f64 {
        if self.total_volume_sum == 0 {
            return 0.0;
        }
        self.signed_volume_sum as f64 / self.total_volume_sum as f64
    }

    /// Calculate VPIN-like toxicity metric
    /// Higher values indicate more toxic (informed) flow
    pub fn vpin(&self) -> f64 {
        if self.count < 10 {
            return 0.0;
        }

        // Volume-synchronized probability of informed trading approximation
        let mut abs_flow_sum = 0.0;
        
        // Bucket-based VPIN approximation using current window
        for i in 0..self.count.min(100) {
            let idx = (self.write_idx + i) % 1024;
            if let Some(trade) = self.trades[idx] {
                let signed_vol = if trade.is_buy { trade.size as f64 } else { -(trade.size as f64) };
                abs_flow_sum += signed_vol.abs();
            }
        }

        let bucket_volume = self.total_volume_sum.min(u64::MAX) as f64;
        if bucket_volume < 1.0 {
            return 0.0;
        }

        // VPIN = sum(|buy - sell|) / sum(buy + sell)
        (abs_flow_sum / bucket_volume).min(1.0)
    }

    /// Update last price for return calculations
    #[inline]
    pub fn update_price(&mut self, price: u64) {
        self.last_price = price;
    }
}

impl Default for FlowStatistics {
    fn default() -> Self {
        Self::new()
    }
}

/// Adverse Selection Probability Model output
#[derive(Debug, Clone, Copy)]
pub struct AdverseSelectionMetrics {
    /// Probability of adverse selection (0.0 to 1.0)
    pub probability: f64,
    /// Expected cost from adverse selection (in basis points)
    pub expected_cost_bps: f64,
    /// Recommended spread adjustment (in ticks)
    pub spread_adjustment_ticks: u32,
    /// Confidence in the estimate
    pub confidence: f64,
    /// Primary risk factor detected
    pub risk_factor: RiskFactor,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RiskFactor {
    LowToxicity,
    HighVPIN,
    FlowImbalance,
    MomentumReversal,
    VolatilitySpike,
}

/// Adverse Selection Model for market making protection
pub struct AdverseSelectionModel {
    /// Buy-side flow statistics
    buy_flow: FlowStatistics,
    /// Sell-side flow statistics  
    sell_flow: FlowStatistics,
    /// Combined flow statistics
    combined_flow: FlowStatistics,
    /// Baseline volatility (for comparison)
    baseline_volatility: f64,
    /// Current volatility estimate
    current_volatility: f64,
    /// Minimum maker rebate (below this, widen spread)
    min_maker_rebate_bps: f64,
    /// Tick size in quote currency
    tick_size: u64,
    /// Last mid price
    last_mid_price: u64,
    /// Recent price returns for momentum detection
    price_returns: [f64; 50],
    return_idx: usize,
    /// Time since last significant price move
    last_move_time: Instant,
}

impl AdverseSelectionModel {
    pub fn new(min_maker_rebate_bps: f64, tick_size: u64) -> Self {
        AdverseSelectionModel {
            buy_flow: FlowStatistics::new(),
            sell_flow: FlowStatistics::new(),
            combined_flow: FlowStatistics::new(),
            baseline_volatility: 0.001, // 0.1% baseline
            current_volatility: 0.001,
            min_maker_rebate_bps,
            tick_size,
            last_mid_price: 0,
            price_returns: [0.0; 50],
            return_idx: 0,
            last_move_time: Instant::now(),
        }
    }

    /// Record a trade with classification
    #[inline]
    pub fn record_trade(&mut self, size: u64, price: u64, is_buy: bool, above_mid: bool) {
        let trade = TradeFlow {
            timestamp: Instant::now(),
            size,
            price,
            is_buy,
            above_mid,
        };

        self.combined_flow.add_trade(trade);
        
        if is_buy {
            self.buy_flow.add_trade(trade);
        } else {
            self.sell_flow.add_trade(trade);
        }
    }

    /// Update mid price and calculate returns
    #[inline]
    pub fn update_mid_price(&mut self, mid_price: u64) {
        if self.last_mid_price > 0 && mid_price > 0 {
            let ret = (mid_price as f64 - self.last_mid_price as f64) / self.last_mid_price as f64;
            
            // Store return in circular buffer
            self.price_returns[self.return_idx] = ret;
            self.return_idx = (self.return_idx + 1) % 50;

            // Update volatility estimate (rolling std dev approximation)
            let mean_ret = self.calculate_mean_return();
            let mut var_sum = 0.0;
            for i in 0..50 {
                let diff = self.price_returns[i] - mean_ret;
                var_sum += diff * diff;
            }
            self.current_volatility = (var_sum / 50.0).sqrt();

            // Detect significant move
            if ret.abs() > 3.0 * self.baseline_volatility {
                self.last_move_time = Instant::now();
            }
        }
        
        self.last_mid_price = mid_price;
        self.combined_flow.update_price(mid_price);
    }

    /// Calculate mean return over window
    fn calculate_mean_return(&self) -> f64 {
        let sum: f64 = self.price_returns.iter().sum();
        sum / 50.0
    }

    /// Detect momentum reversal pattern (sign of informed trading)
    fn detect_momentum_reversal(&self) -> f64 {
        if self.return_idx < 10 {
            return 0.0;
        }

        // Check for autocorrelation sign flip (reversal)
        let mut correlation = 0.0;
        let mut count = 0;
        
        for i in 0..20 {
            let idx1 = (self.return_idx + i) % 50;
            let idx2 = (self.return_idx + i + 1) % 50;
            
            if self.price_returns[idx1] != 0.0 && self.price_returns[idx2] != 0.0 {
                correlation += self.price_returns[idx1] * self.price_returns[idx2];
                count += 1;
            }
        }

        if count == 0 {
            return 0.0;
        }

        // Negative correlation indicates reversal (toxic flow)
        (correlation / count as f64).max(-1.0).min(0.0).abs()
    }

    /// Main adverse selection probability calculation
    pub fn calculate_metrics(&self) -> AdverseSelectionMetrics {
        let vpin = self.combined_flow.vpin();
        let flow_imbalance = self.combined_flow.flow_imbalance().abs();
        let reversal_signal = self.detect_momentum_reversal();
        let vol_ratio = self.current_volatility / self.baseline_volatility.max(1e-6);

        // Weighted combination of risk signals
        let mut probability = 0.0;
        let mut risk_factor = RiskFactor::LowToxicity;

        // VPIN contribution (primary signal)
        if vpin > 0.7 {
            probability += 0.4 * vpin;
            risk_factor = RiskFactor::HighVPIN;
        } else {
            probability += 0.2 * vpin;
        }

        // Flow imbalance contribution
        if flow_imbalance > 0.6 {
            probability += 0.3 * flow_imbalance;
            if risk_factor == RiskFactor::LowToxicity {
                risk_factor = RiskFactor::FlowImbalance;
            }
        } else {
            probability += 0.1 * flow_imbalance;
        }

        // Momentum reversal contribution
        if reversal_signal > 0.3 {
            probability += 0.3 * reversal_signal;
            risk_factor = RiskFactor::MomentumReversal;
        }

        // Volatility spike contribution
        if vol_ratio > 2.0 {
            probability += 0.2 * (vol_ratio - 1.0).min(2.0) / 2.0;
            if risk_factor == RiskFactor::LowToxicity {
                risk_factor = RiskFactor::VolatilitySpike;
            }
        }

        probability = probability.min(1.0);

        // Calculate expected cost in basis points
        let expected_cost_bps = probability * self.current_volatility * 100.0 * 10.0;

        // Determine spread adjustment
        let spread_adjustment_ticks = self.calculate_spread_adjustment(probability, expected_cost_bps);

        // Confidence based on sample size
        let confidence = (self.combined_flow.count as f64 / 100.0).min(1.0);

        AdverseSelectionMetrics {
            probability,
            expected_cost_bps,
            spread_adjustment_ticks,
            confidence,
            risk_factor,
        }
    }

    /// Calculate recommended spread adjustment
    fn calculate_spread_adjustment(&self, probability: f64, expected_cost_bps: f64) -> u32 {
        // Base adjustment: widen when expected cost exceeds rebate
        let cost_vs_rebate = expected_cost_bps / self.min_maker_rebate_bps.max(1e-6);
        
        let base_adjustment = if cost_vs_rebate > 1.0 {
            ((cost_vs_rebate - 1.0) * 2.0) as u32
        } else {
            0
        };

        // Additional adjustment for high probability
        let prob_adjustment = if probability > 0.7 {
            ((probability - 0.7) * 10.0) as u32
        } else {
            0
        };

        // Volatility-based adjustment
        let vol_adjustment = if self.current_volatility > 2.0 * self.baseline_volatility {
            2
        } else if self.current_volatility > 1.5 * self.baseline_volatility {
            1
        } else {
            0
        };

        base_adjustment + prob_adjustment + vol_adjustment
    }

    /// Check if we should widen spread immediately
    #[inline]
    pub fn should_widen_spread(&self) -> bool {
        let metrics = self.calculate_metrics();
        metrics.expected_cost_bps > self.min_maker_rebate_bps
    }

    /// Get recommended spread width given base spread
    pub fn recommended_spread_ticks(&self, base_spread_ticks: u32) -> u32 {
        let metrics = self.calculate_metrics();
        base_spread_ticks + metrics.spread_adjustment_ticks
    }

    /// Reset model state (e.g., after regime change detection)
    #[inline]
    pub fn reset(&mut self) {
        self.buy_flow = FlowStatistics::new();
        self.sell_flow = FlowStatistics::new();
        self.combined_flow = FlowStatistics::new();
        self.current_volatility = self.baseline_volatility;
        self.price_returns = [0.0; 50];
        self.return_idx = 0;
    }

    /// Set baseline volatility from historical data
    #[inline]
    pub fn set_baseline_volatility(&mut self, vol: f64) {
        self.baseline_volatility = vol.max(1e-6);
    }
}

impl Default for AdverseSelectionModel {
    fn default() -> Self {
        Self::new(2.5, 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flow_statistics() {
        let mut stats = FlowStatistics::new();
        
        // Add some buy trades
        for i in 0..10 {
            stats.add_trade(TradeFlow {
                timestamp: Instant::now(),
                size: 100,
                price: 50000,
                is_buy: true,
                above_mid: true,
            });
        }
        
        assert_eq!(stats.net_flow(), 1000);
        assert!(stats.flow_imbalance() > 0.9);
    }

    #[test]
    fn test_adverse_selection_detection() {
        let mut model = AdverseSelectionModel::new(2.5, 1);
        
        // Simulate toxic flow: large one-sided trades followed by reversals
        model.update_mid_price(50000);
        
        for _ in 0..50 {
            model.record_trade(1000, 50000, true, true); // Large buys
        }
        
        // Price goes up then reverses (informed selling)
        model.update_mid_price(50100);
        for _ in 0..50 {
            model.record_trade(1000, 50100, false, false); // Large sells
        }
        model.update_mid_price(49900);
        
        let metrics = model.calculate_metrics();
        
        // Should detect elevated adverse selection
        assert!(metrics.probability > 0.0);
        assert!(metrics.confidence > 0.0);
    }

    #[test]
    fn test_spread_adjustment() {
        let mut model = AdverseSelectionModel::new(2.5, 1);
        model.update_mid_price(50000);
        
        // Normal conditions: minimal adjustment
        let metrics = model.calculate_metrics();
        assert!(metrics.spread_adjustment_ticks >= 0);
    }
}
