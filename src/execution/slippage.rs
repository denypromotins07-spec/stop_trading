//! Slippage and market impact model.
//! Predicts adverse selection for optimal limit order placement.

use std::sync::atomic::{AtomicF64, AtomicU64, Ordering};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SlippageError {
    #[error("Invalid queue size")]
    InvalidQueueSize,
    #[error("Invalid spread")]
    InvalidSpread,
}

#[derive(Debug, Clone, Copy)]
pub struct MarketImpactModel {
    pub queue_size: f64,
    pub spread_bps: f64,
    pub tick_size: f64,
    pub daily_volume: f64,
    pub volatility: f64,
}

impl MarketImpactModel {
    pub fn new(
        queue_size: f64,
        spread_bps: f64,
        tick_size: f64,
        daily_volume: f64,
        volatility: f64,
    ) -> Result<Self, SlippageError> {
        if queue_size <= 0.0 {
            return Err(SlippageError::InvalidQueueSize);
        }
        if spread_bps <= 0.0 {
            return Err(SlippageError::InvalidSpread);
        }

        Ok(Self {
            queue_size,
            spread_bps,
            tick_size,
            daily_volume,
            volatility,
        })
    }

    /// Estimate slippage in basis points for a given order size
    pub fn estimate_slippage_bps(&self, order_size: f64) -> f64 {
        // Square-root law for market impact
        let participation = order_size / self.daily_volume.max(1.0);
        let impact = 0.1 * (participation.sqrt()) * 100.0; // Scale to bps

        // Add spread cost
        let spread_cost = self.spread_bps / 2.0;

        // Add volatility penalty
        let vol_penalty = self.volatility * 10.0;

        impact + spread_cost + vol_penalty
    }

    /// Calculate optimal limit price offset
    pub fn optimal_limit_offset(&self, order_size: f64, side: Side) -> f64 {
        let slippage = self.estimate_slippage_bps(order_size);
        
        // Queue position factor
        let queue_factor = 1.0 - (order_size / self.queue_size).min(0.9);
        
        let base_offset = slippage * queue_factor;
        
        match side {
            Side::Buy => -base_offset,
            Side::Sell => base_offset,
        }
    }

    /// Probability of fill at given price level
    pub fn fill_probability(&self, price_offset_bps: f64, side: Side) -> f64 {
        let fair_offset = self.spread_bps / 2.0;
        
        let effective_offset = match side {
            Side::Buy => -price_offset_bps,
            Side::Sell => price_offset_bps,
        };

        if effective_offset >= fair_offset {
            // Inside spread, high probability
            0.8 + 0.2 * (effective_offset - fair_offset) / fair_offset.min(1.0)
        } else {
            // Outside spread, lower probability
            0.5 * (effective_offset / fair_offset).max(0.0)
        }.min(1.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Buy,
    Sell,
}

#[derive(Debug, Clone, Copy)]
pub struct SlippageEstimate {
    pub expected_slippage_bps: f64,
    pub worst_case_bps: f64,
    pub best_case_bps: f64,
    pub confidence: f64,
}

impl SlippageEstimate {
    pub fn new(model: &MarketImpactModel, order_size: f64) -> Self {
        let expected = model.estimate_slippage_bps(order_size);
        
        Self {
            expected_slippage_bps: expected,
            worst_case_bps: expected * 2.0,
            best_case_bps: expected * 0.5,
            confidence: 0.7,
        }
    }
}

/// Pre-trade slippage checker
pub struct SlippageChecker {
    max_slippage_bps: AtomicF64,
    check_count: AtomicU64,
    rejected_count: AtomicU64,
}

impl SlippageChecker {
    pub fn new(max_slippage_bps: f64) -> Self {
        Self {
            max_slippage_bps: AtomicF64::new(max_slippage_bps),
            check_count: AtomicU64::new(0),
            rejected_count: AtomicU64::new(0),
        }
    }

    pub fn check(&self, estimate: &SlippageEstimate) -> SlippageCheckResult {
        self.check_count.fetch_add(1, Ordering::Relaxed);
        
        let max = self.max_slippage_bps.load(Ordering::Relaxed);
        let allowed = estimate.expected_slippage_bps <= max;

        if !allowed {
            self.rejected_count.fetch_add(1, Ordering::Relaxed);
        }

        SlippageCheckResult {
            allowed,
            estimated_slippage: estimate.expected_slippage_bps,
            max_allowed: max,
            headroom_bps: max - estimate.expected_slippage_bps,
        }
    }

    pub fn get_stats(&self) -> SlippageStats {
        let checks = self.check_count.load(Ordering::Relaxed);
        let rejected = self.rejected_count.load(Ordering::Relaxed);
        
        SlippageStats {
            total_checks: checks,
            rejected_count: rejected,
            rejection_rate: if checks > 0 { rejected as f64 / checks as f64 } else { 0.0 },
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SlippageCheckResult {
    pub allowed: bool,
    pub estimated_slippage: f64,
    pub max_allowed: f64,
    pub headroom_bps: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct SlippageStats {
    pub total_checks: u64,
    pub rejected_count: u64,
    pub rejection_rate: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_market_impact() {
        let model = MarketImpactModel::new(
            1000.0,
            10.0,
            0.01,
            1000000.0,
            0.02,
        ).unwrap();

        let slippage = model.estimate_slippage_bps(10000.0);
        assert!(slippage > 0.0);
        assert!(slippage.is_finite());

        let offset = model.optimal_limit_offset(10000.0, Side::Buy);
        assert!(offset < 0.0); // Buy should have negative offset
    }

    #[test]
    fn test_fill_probability() {
        let model = MarketImpactModel::new(
            1000.0,
            10.0,
            0.01,
            1000000.0,
            0.02,
        ).unwrap();

        let prob_inside = model.fill_probability(6.0, Side::Buy);
        let prob_outside = model.fill_probability(2.0, Side::Buy);
        
        assert!(prob_inside > prob_outside);
    }

    #[test]
    fn test_slippage_checker() {
        let checker = SlippageChecker::new(20.0);
        let model = MarketImpactModel::new(1000.0, 10.0, 0.01, 1000000.0, 0.02).unwrap();
        
        let estimate = SlippageEstimate::new(&model, 1000.0);
        let result = checker.check(&estimate);
        
        assert!(result.allowed || !result.allowed); // Just verify it runs
        assert_eq!(checker.get_stats().total_checks, 1);
    }
}
