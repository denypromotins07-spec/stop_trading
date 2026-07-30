//! Percentage of Volume (POV) execution algorithm.
//! Participates at a fixed percentage of real-time market volume.

use std::sync::atomic::{AtomicF64, AtomicU64, AtomicBool, Ordering};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum PovError {
    #[error("Invalid participation rate: must be between 0 and 1")]
    InvalidRate,
    #[error("Invalid quantity")]
    InvalidQuantity,
}

pub struct PovEngine {
    target_participation: AtomicF64,
    remaining_quantity: AtomicF64,
    executed_quantity: AtomicF64,
    market_volume_tracked: AtomicF64,
    started: AtomicBool,
    completed: AtomicBool,
}

impl PovEngine {
    pub fn new(total_quantity: f64, participation_rate: f64) -> Result<Self, PovError> {
        if participation_rate <= 0.0 || participation_rate > 1.0 {
            return Err(PovError::InvalidRate);
        }
        if total_quantity <= 0.0 {
            return Err(PovError::InvalidQuantity);
        }

        Ok(Self {
            target_participation: AtomicF64::new(participation_rate),
            remaining_quantity: AtomicF64::new(total_quantity),
            executed_quantity: AtomicF64::new(0.0),
            market_volume_tracked: AtomicF64::new(0.0),
            started: AtomicBool::new(false),
            completed: AtomicBool::new(false),
        })
    }

    pub fn start(&self) {
        self.started.store(true, Ordering::Relaxed);
    }

    pub fn calculate_order_size(&self, market_volume: f64) -> f64 {
        let rate = self.target_participation.load(Ordering::Relaxed);
        let remaining = self.remaining_quantity.load(Ordering::Relaxed);
        
        let target = market_volume * rate;
        target.min(remaining)
    }

    pub fn update(&self, market_volume: f64) -> Option<f64> {
        if !self.started.load(Ordering::Relaxed) || self.completed.load(Ordering::Relaxed) {
            return None;
        }

        let order_size = self.calculate_order_size(market_volume);
        if order_size <= 0.0 {
            self.completed.store(true, Ordering::Relaxed);
            return None;
        }

        self.market_volume_tracked.fetch_add(market_volume, Ordering::Relaxed);
        Some(order_size)
    }

    pub fn report_fill(&self, quantity: f64) {
        let exec = self.executed_quantity.load(Ordering::Relaxed);
        let remaining = self.remaining_quantity.load(Ordering::Relaxed);
        
        let actual = quantity.min(remaining);
        self.executed_quantity.store(exec + actual, Ordering::Relaxed);
        self.remaining_quantity.store(remaining - actual, Ordering::Relaxed);

        if self.remaining_quantity.load(Ordering::Relaxed) <= 0.0 {
            self.completed.store(true, Ordering::Relaxed);
        }
    }

    pub fn is_complete(&self) -> bool {
        self.completed.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pov_basic() {
        let engine = PovEngine::new(1000.0, 0.1).unwrap();
        engine.start();

        let order = engine.update(100.0);
        assert_eq!(order, Some(10.0));

        if let Some(qty) = order {
            engine.report_fill(qty);
        }

        assert!(!engine.is_complete());
    }
}
