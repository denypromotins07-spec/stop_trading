//! Volume-Weighted Average Price (VWAP) execution algorithm.
//! Tracks historical volume profiles to hide execution footprint.

use std::sync::atomic::{AtomicF64, AtomicU64, AtomicBool, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};
use thiserror::Error;

/// Error types for VWAP execution
#[derive(Debug, Error)]
pub enum VwapError {
    #[error("Invalid quantity: must be positive")]
    InvalidQuantity,
    #[error("Invalid volume profile")]
    InvalidProfile,
    #[error("Execution not started")]
    NotStarted,
    #[error("Order completed")]
    OrderCompleted,
}

/// Historical volume bucket
#[derive(Debug, Clone)]
pub struct VolumeBucket {
    pub time_offset_secs: u64,
    pub avg_volume: f64,
    pub std_dev: f64,
}

/// VWAP execution parameters
#[derive(Debug, Clone)]
pub struct VwapParams {
    pub total_quantity: f64,
    pub volume_profile: Vec<VolumeBucket>,
    pub participation_rate: f64,
    pub max_participation: f64,
    pub min_order_size: f64,
}

impl VwapParams {
    pub fn new(total_quantity: f64, participation_rate: f64) -> Result<Self, VwapError> {
        if total_quantity <= 0.0 {
            return Err(VwapError::InvalidQuantity);
        }
        if participation_rate <= 0.0 || participation_rate > 1.0 {
            return Err(VwapError::InvalidProfile);
        }

        Ok(Self {
            total_quantity,
            volume_profile: Vec::new(),
            participation_rate,
            max_participation: 0.25,
            min_order_size: 0.001,
        })
    }

    pub fn with_profile(total_quantity: f64, profile: Vec<VolumeBucket>, participation: f64) -> Result<Self, VwapError> {
        let mut params = Self::new(total_quantity, participation)?;
        params.volume_profile = profile;
        Ok(params)
    }
}

/// VWAP child order
#[derive(Debug, Clone)]
pub struct VwapChildOrder {
    pub target_quantity: f64,
    pub market_volume: f64,
    pub executed_quantity: f64,
    pub average_price: f64,
    pub timestamp: u64,
}

/// VWAP execution engine
pub struct VwapEngine {
    params: VwapParams,
    side: crate::execution::twap::Side,
    remaining_quantity: AtomicF64,
    executed_quantity: AtomicF64,
    executed_value: AtomicF64,
    total_market_volume: AtomicF64,
    started: AtomicBool,
    completed: AtomicBool,
    current_bucket: AtomicU64,
}

impl VwapEngine {
    pub fn new(params: VwapParams, side: crate::execution::twap::Side) -> Self {
        Self {
            params,
            side,
            remaining_quantity: AtomicF64::new(params.total_quantity),
            executed_quantity: AtomicF64::new(0.0),
            executed_value: AtomicF64::new(0.0),
            total_market_volume: AtomicF64::new(0.0),
            started: AtomicBool::new(false),
            completed: AtomicBool::new(false),
            current_bucket: AtomicU64::new(0),
        }
    }

    pub fn start(&self) -> Result<(), VwapError> {
        if self.started.load(Ordering::Relaxed) {
            return Err(VwapError::NotStarted);
        }
        self.started.store(true, Ordering::Relaxed);
        Ok(())
    }

    /// Calculate target quantity based on market volume
    pub fn calculate_target(&self, market_volume: f64) -> f64 {
        let participation = self.params.participation_rate;
        let max_part = self.params.max_participation;
        
        let target = market_volume * participation;
        target.min(market_volume * max_part).max(self.params.min_order_size)
    }

    /// Update with market volume and get target order size
    pub fn update_market_volume(&self, market_volume: f64) -> Option<VwapChildOrder> {
        if !self.started.load(Ordering::Relaxed) || self.completed.load(Ordering::Relaxed) {
            return None;
        }

        let remaining = self.remaining_quantity.load(Ordering::Relaxed);
        if remaining <= 0.0 {
            self.completed.store(true, Ordering::Relaxed);
            return None;
        }

        let target_qty = self.calculate_target(market_volume).min(remaining);
        
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        self.total_market_volume.fetch_add(market_volume, Ordering::Relaxed);
        self.current_bucket.fetch_add(1, Ordering::Relaxed);

        Some(VwapChildOrder {
            target_quantity: target_qty,
            market_volume,
            executed_quantity: 0.0,
            average_price: 0.0,
            timestamp: now,
        })
    }

    /// Report fill
    pub fn report_fill(&self, quantity: f64, price: f64) -> Result<(), VwapError> {
        if quantity <= 0.0 {
            return Err(VwapError::InvalidQuantity);
        }

        let exec_qty = self.executed_quantity.load(Ordering::Relaxed);
        let exec_val = self.executed_value.load(Ordering::Relaxed);
        let remaining = self.remaining_quantity.load(Ordering::Relaxed);

        let actual_qty = quantity.min(remaining);
        
        self.executed_quantity.store(exec_qty + actual_qty, Ordering::Relaxed);
        self.executed_value.store(exec_val + actual_qty * price, Ordering::Relaxed);
        self.remaining_quantity.store(remaining - actual_qty, Ordering::Relaxed);

        if self.remaining_quantity.load(Ordering::Relaxed) <= 0.0 {
            self.completed.store(true, Ordering::Relaxed);
        }

        Ok(())
    }

    /// Get VWAP progress
    pub fn get_progress(&self) -> VwapProgress {
        let executed = self.executed_quantity.load(Ordering::Relaxed);
        let value = self.executed_value.load(Ordering::Relaxed);
        let market_vol = self.total_market_volume.load(Ordering::Relaxed);

        VwapProgress {
            executed_quantity: executed,
            remaining_quantity: self.remaining_quantity.load(Ordering::Relaxed),
            average_price: if executed > 0.0 { value / executed } else { 0.0 },
            market_vwap: if market_vol > 0.0 { value / executed.max(1.0) } else { 0.0 },
            participation_rate: if market_vol > 0.0 { executed / market_vol } else { 0.0 },
            is_complete: self.completed.load(Ordering::Relaxed),
        }
    }

    pub fn is_complete(&self) -> bool {
        self.completed.load(Ordering::Relaxed)
    }

    pub fn cancel(&self) {
        self.completed.store(true, Ordering::Relaxed);
    }
}

#[derive(Debug, Clone, Copy)]
pub struct VwapProgress {
    pub executed_quantity: f64,
    pub remaining_quantity: f64,
    pub average_price: f64,
    pub market_vwap: f64,
    pub participation_rate: f64,
    pub is_complete: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::twap::Side;

    #[test]
    fn test_vwap_basic() {
        let params = VwapParams::new(1000.0, 0.1).unwrap();
        let engine = VwapEngine::new(params, Side::Buy);
        
        engine.start().unwrap();
        
        let order = engine.update_market_volume(100.0);
        assert!(order.is_some());
        
        let o = order.unwrap();
        engine.report_fill(o.target_quantity, 100.0).unwrap();
        
        let progress = engine.get_progress();
        assert!(progress.executed_quantity > 0.0);
    }
}
