//! Execution module root.
//! Defines the `ExecutionAlgo` trait for dynamic dispatch and state tracking.

pub mod twap;
pub mod vwap;
pub mod pov;
pub mod iceberg;
pub mod slippage;

pub use twap::{TwapEngine, TwapParams, TwapState, TwapProgress, Side as TwapSide};
pub use vwap::{VwapEngine, VwapParams, VwapProgress};
pub use pov::PovEngine;
pub use iceberg::{IcebergOrder, IcebergState, IcebergProgress};
pub use slippage::{MarketImpactModel, SlippageEstimate, SlippageChecker};

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use thiserror::Error;

/// Error types for execution module
#[derive(Debug, Error)]
pub enum ExecutionError {
    #[error("Invalid order: {0}")]
    InvalidOrder(String),
    #[error("Risk check failed: {0}")]
    RiskCheckFailed(String),
    #[error("Execution failed: {0}")]
    ExecutionFailed(String),
    #[error("Order not found")]
    OrderNotFound,
}

/// Common side type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Buy,
    Sell,
}

impl From<TwapSide> for Side {
    fn from(side: TwapSide) -> Self {
        match side {
            TwapSide::Buy => Side::Buy,
            TwapSide::Sell => Side::Sell,
        }
    }
}

/// Trait for all execution algorithms
pub trait ExecutionAlgo {
    /// Start the algorithm
    fn start(&self) -> Result<(), ExecutionError>;
    
    /// Get current fill quantity
    fn filled_quantity(&self) -> f64;
    
    /// Get remaining quantity
    fn remaining_quantity(&self) -> f64;
    
    /// Check if execution is complete
    fn is_complete(&self) -> bool;
    
    /// Cancel execution
    fn cancel(&self);
    
    /// Get average fill price
    fn average_price(&self) -> Option<f64>;
}

impl ExecutionAlgo for TwapEngine {
    fn start(&self) -> Result<(), ExecutionError> {
        self.start().map_err(|e| ExecutionError::ExecutionFailed(e.to_string()))
    }
    
    fn filled_quantity(&self) -> f64 {
        self.get_progress().executed_quantity
    }
    
    fn remaining_quantity(&self) -> f64 {
        self.get_progress().remaining_quantity
    }
    
    fn is_complete(&self) -> bool {
        self.is_complete()
    }
    
    fn cancel(&self) {
        self.cancel();
    }
    
    fn average_price(&self) -> Option<f64> {
        let progress = self.get_progress();
        if progress.executed_quantity > 0.0 {
            Some(progress.average_price)
        } else {
            None
        }
    }
}

impl ExecutionAlgo for VwapEngine {
    fn start(&self) -> Result<(), ExecutionError> {
        self.start().map_err(|e| ExecutionError::ExecutionFailed(e.to_string()))
    }
    
    fn filled_quantity(&self) -> f64 {
        self.get_progress().executed_quantity
    }
    
    fn remaining_quantity(&self) -> f64 {
        self.get_progress().remaining_quantity
    }
    
    fn is_complete(&self) -> bool {
        self.is_complete()
    }
    
    fn cancel(&self) {
        self.cancel();
    }
    
    fn average_price(&self) -> Option<f64> {
        let progress = self.get_progress();
        if progress.executed_quantity > 0.0 {
            Some(progress.average_price)
        } else {
            None
        }
    }
}

/// Pre-trade risk limits
#[derive(Debug, Clone)]
pub struct RiskLimits {
    pub max_order_size: f64,
    pub max_open_orders: usize,
    pub max_notional: f64,
    pub max_daily_volume: f64,
}

impl RiskLimits {
    pub fn new(
        max_order_size: f64,
        max_open_orders: usize,
        max_notional: f64,
        max_daily_volume: f64,
    ) -> Self {
        Self {
            max_order_size,
            max_open_orders,
            max_notional,
            max_daily_volume,
        }
    }
}

/// Pre-trade risk checker
pub struct PreTradeRisk {
    limits: RiskLimits,
    current_exposure: AtomicU64, // In cents to avoid float issues
    open_orders: AtomicU64,
    daily_volume: AtomicU64,
    halted: AtomicBool,
}

impl PreTradeRisk {
    pub fn new(limits: RiskLimits) -> Self {
        Self {
            limits,
            current_exposure: AtomicU64::new(0),
            open_orders: AtomicU64::new(0),
            daily_volume: AtomicU64::new(0),
            halted: AtomicBool::new(false),
        }
    }

    pub fn check_order(&self, size: f64, price: f64) -> RiskCheckResult {
        if self.halted.load(Ordering::Relaxed) {
            return RiskCheckResult {
                allowed: false,
                reason: Some("Trading halted".to_string()),
            };
        }

        let notional = size * price;

        if size > self.limits.max_order_size {
            return RiskCheckResult {
                allowed: false,
                reason: Some(format!("Order size {} exceeds max {}", size, self.limits.max_order_size)),
            };
        }

        if notional > self.limits.max_notional {
            return RiskCheckResult {
                allowed: false,
                reason: Some(format!("Notional {} exceeds max {}", notional, self.limits.max_notional)),
            };
        }

        let open = self.open_orders.load(Ordering::Relaxed);
        if open >= self.limits.max_open_orders as u64 {
            return RiskCheckResult {
                allowed: false,
                reason: Some(format!("Max open orders {} reached", self.limits.max_open_orders)),
            };
        }

        RiskCheckResult {
            allowed: true,
            reason: None,
        }
    }

    pub fn record_order(&self, size: f64, price: f64) {
        let notional_cents = (size * price * 100.0) as u64;
        self.current_exposure.fetch_add(notional_cents, Ordering::Relaxed);
        self.open_orders.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_fill(&self, size: f64, price: f64) {
        let volume_cents = (size * price * 100.0) as u64;
        self.daily_volume.fetch_add(volume_cents, Ordering::Relaxed);
    }

    pub fn release_order(&self, size: f64, price: f64) {
        let notional_cents = (size * price * 100.0) as u64;
        self.current_exposure.fetch_sub(notional_cents.min(self.current_exposure.load(Ordering::Relaxed)), Ordering::Relaxed);
        self.open_orders.fetch_sub(1, Ordering::Relaxed);
    }

    pub fn halt(&self) {
        self.halted.store(true, Ordering::Relaxed);
    }

    pub fn resume(&self) {
        self.halted.store(false, Ordering::Relaxed);
    }

    pub fn is_halted(&self) -> bool {
        self.halted.load(Ordering::Relaxed)
    }

    pub fn reset_daily(&self) {
        self.daily_volume.store(0, Ordering::Relaxed);
    }
}

#[derive(Debug, Clone)]
pub struct RiskCheckResult {
    pub allowed: bool,
    pub reason: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pre_trade_risk() {
        let limits = RiskLimits::new(1000.0, 10, 100000.0, 1000000.0);
        let risk = PreTradeRisk::new(limits);

        let result = risk.check_order(500.0, 100.0);
        assert!(result.allowed);

        let result = risk.check_order(2000.0, 100.0);
        assert!(!result.allowed);
    }

    #[test]
    fn test_risk_halt() {
        let limits = RiskLimits::new(1000.0, 10, 100000.0, 1000000.0);
        let risk = PreTradeRisk::new(limits);

        risk.halt();
        let result = risk.check_order(100.0, 100.0);
        assert!(!result.allowed);

        risk.resume();
        let result = risk.check_order(100.0, 100.0);
        assert!(result.allowed);
    }
}
