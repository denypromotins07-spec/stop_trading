//! Iceberg order logic to hide large order sizes.
//! Only exposes a fraction on the L2 book with automatic refresh.

use std::sync::atomic::{AtomicF64, AtomicU64, AtomicBool, Ordering};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum IcebergError {
    #[error("Invalid total quantity")]
    InvalidTotalQuantity,
    #[error("Invalid display quantity")]
    InvalidDisplayQuantity,
    #[error("Order completed")]
    OrderCompleted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IcebergState {
    Pending,
    Active,
    Refreshing,
    Completed,
    Cancelled,
}

pub struct IcebergOrder {
    total_quantity: AtomicF64,
    display_quantity: AtomicF64,
    remaining_total: AtomicF64,
    remaining_display: AtomicF64,
    executed_quantity: AtomicF64,
    fill_count: AtomicU64,
    state: AtomicU64,
    auto_refresh: AtomicBool,
}

impl IcebergOrder {
    pub fn new(total_qty: f64, display_qty: f64) -> Result<Self, IcebergError> {
        if total_qty <= 0.0 {
            return Err(IcebergError::InvalidTotalQuantity);
        }
        if display_qty <= 0.0 || display_qty > total_qty {
            return Err(IcebergError::InvalidDisplayQuantity);
        }

        Ok(Self {
            total_quantity: AtomicF64::new(total_qty),
            display_quantity: AtomicF64::new(display_qty),
            remaining_total: AtomicF64::new(total_qty),
            remaining_display: AtomicF64::new(display_qty),
            executed_quantity: AtomicF64::new(0.0),
            fill_count: AtomicU64::new(0),
            state: AtomicU64::new(IcebergState::Pending as u64),
            auto_refresh: AtomicBool::new(true),
        })
    }

    pub fn start(&self) {
        self.state.store(IcebergState::Active as u64, Ordering::Relaxed);
    }

    pub fn get_visible_quantity(&self) -> f64 {
        self.remaining_display.load(Ordering::Relaxed)
    }

    pub fn report_fill(&self, quantity: f64) -> Result<IcebergFillResult, IcebergError> {
        let state = self.get_state();
        if state == IcebergState::Completed || state == IcebergState::Cancelled {
            return Err(IcebergError::OrderCompleted);
        }

        let remaining_disp = self.remaining_display.load(Ordering::Relaxed);
        let remaining_total = self.remaining_total.load(Ordering::Relaxed);
        let executed = self.executed_quantity.load(Ordering::Relaxed);

        let actual_fill = quantity.min(remaining_disp).min(remaining_total);

        let new_remaining_disp = remaining_disp - actual_fill;
        let new_remaining_total = remaining_total - actual_fill;
        let new_executed = executed + actual_fill;

        self.remaining_display.store(new_remaining_disp, Ordering::Relaxed);
        self.remaining_total.store(new_remaining_total, Ordering::Relaxed);
        self.executed_quantity.store(new_executed, Ordering::Relaxed);
        self.fill_count.fetch_add(1, Ordering::Relaxed);

        let needs_refresh = new_remaining_disp <= 0.0 && new_remaining_total > 0.0;
        let is_complete = new_remaining_total <= 0.0;

        if is_complete {
            self.state.store(IcebergState::Completed as u64, Ordering::Relaxed);
        } else if needs_refresh && self.auto_refresh.load(Ordering::Relaxed) {
            self.refresh();
        }

        Ok(IcebergFillResult {
            filled_quantity: actual_fill,
            remaining_total: new_remaining_total,
            remaining_display: new_remaining_disp,
            needs_refresh,
            is_complete,
        })
    }

    fn refresh(&self) {
        let display_qty = self.display_quantity.load(Ordering::Relaxed);
        let remaining_total = self.remaining_total.load(Ordering::Relaxed);
        
        let new_display = display_qty.min(remaining_total);
        self.remaining_display.store(new_display, Ordering::Relaxed);
        self.state.store(IcebergState::Active as u64, Ordering::Relaxed);
    }

    pub fn get_state(&self) -> IcebergState {
        match self.state.load(Ordering::Relaxed) {
            0 => IcebergState::Pending,
            1 => IcebergState::Active,
            2 => IcebergState::Refreshing,
            3 => IcebergState::Completed,
            4 => IcebergState::Cancelled,
            _ => IcebergState::Pending,
        }
    }

    pub fn cancel(&self) {
        self.state.store(IcebergState::Cancelled as u64, Ordering::Relaxed);
    }

    pub fn get_progress(&self) -> IcebergProgress {
        let total = self.total_quantity.load(Ordering::Relaxed);
        let executed = self.executed_quantity.load(Ordering::Relaxed);
        let remaining_total = self.remaining_total.load(Ordering::Relaxed);
        let remaining_display = self.remaining_display.load(Ordering::Relaxed);

        IcebergProgress {
            total_quantity: total,
            executed_quantity: executed,
            remaining_total,
            remaining_display,
            progress_percent: executed / total * 100.0,
            fill_count: self.fill_count.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct IcebergFillResult {
    pub filled_quantity: f64,
    pub remaining_total: f64,
    pub remaining_display: f64,
    pub needs_refresh: bool,
    pub is_complete: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct IcebergProgress {
    pub total_quantity: f64,
    pub executed_quantity: f64,
    pub remaining_total: f64,
    pub remaining_display: f64,
    pub progress_percent: f64,
    pub fill_count: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iceberg_basic() {
        let iceberg = IcebergOrder::new(1000.0, 100.0).unwrap();
        iceberg.start();

        assert_eq!(iceberg.get_visible_quantity(), 100.0);

        let result = iceberg.report_fill(50.0).unwrap();
        assert!(!result.is_complete);
        assert_eq!(result.remaining_display, 50.0);

        let result2 = iceberg.report_fill(50.0).unwrap();
        assert!(result2.needs_refresh);
        assert_eq!(iceberg.get_visible_quantity(), 100.0); // Auto-refreshed
    }

    #[test]
    fn test_iceberg_completion() {
        let iceberg = IcebergOrder::new(100.0, 50.0).unwrap();
        iceberg.start();

        iceberg.report_fill(50.0).unwrap();
        iceberg.report_fill(50.0).unwrap();

        assert_eq!(iceberg.get_state(), IcebergState::Completed);
    }
}
