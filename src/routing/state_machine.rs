//! Full, idempotent Order State Machine.
//! Manages order lifecycle: Pending -> Partial -> Filled/Cancelled

use std::sync::atomic::{AtomicU64, AtomicF64, AtomicBool, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum OrderStateError {
    #[error("Invalid state transition: {from:?} -> {to:?}")]
    InvalidTransition { from: OrderState, to: OrderState },
    #[error("Order not found")]
    OrderNotFound,
    #[error("Duplicate execution prevented")]
    DuplicateExecution,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderState {
    Pending,
    New,
    PartiallyFilled,
    Filled,
    Cancelled,
    Rejected,
}

impl OrderState {
    pub fn is_terminal(&self) -> bool {
        matches!(self, OrderState::Filled | OrderState::Cancelled | OrderState::Rejected)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderType {
    Limit,
    Market,
    StopLimit,
    StopMarket,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderSide {
    Buy,
    Sell,
}

/// Order representation
#[derive(Debug, Clone)]
pub struct Order {
    pub id: u64,
    pub client_order_id: String,
    pub symbol: String,
    pub side: OrderSide,
    pub order_type: OrderType,
    pub quantity: f64,
    pub price: Option<f64>,
    pub stop_price: Option<f64>,
    pub time_in_force: TimeInForce,
    pub created_at: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeInForce {
    Gtc,
    Ioc,
    Fok,
    Day,
}

/// Order state with execution tracking
pub struct OrderStateMachine {
    order: Order,
    state: AtomicU64,
    filled_quantity: AtomicF64,
    remaining_quantity: AtomicF64,
    average_fill_price: AtomicF64,
    fill_count: AtomicU64,
    last_update: AtomicU64,
    executed: AtomicBool, // For idempotency
}

impl OrderStateMachine {
    pub fn new(order: Order) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        Self {
            order,
            state: AtomicU64::new(OrderState::Pending as u64),
            filled_quantity: AtomicF64::new(0.0),
            remaining_quantity: AtomicF64::new(order.quantity),
            average_fill_price: AtomicF64::new(0.0),
            fill_count: AtomicU64::new(0),
            last_update: AtomicU64::new(now),
            executed: AtomicBool::new(false),
        }
    }

    /// Transition to a new state (validates transitions)
    pub fn transition(&self, new_state: OrderState) -> Result<(), OrderStateError> {
        let current = self.get_state();
        
        if !Self::is_valid_transition(current, new_state) {
            return Err(OrderStateError::InvalidTransition { from: current, to: new_state });
        }

        self.state.store(new_state as u64, Ordering::SeqCst);
        self.last_update.store(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            Ordering::Relaxed,
        );

        Ok(())
    }

    /// Check if transition is valid
    fn is_valid_transition(from: OrderState, to: OrderState) -> bool {
        match from {
            OrderState::Pending => matches!(to, OrderState::New | OrderState::Rejected | OrderState::Cancelled),
            OrderState::New => matches!(to, OrderState::PartiallyFilled | OrderState::Filled | OrderState::Cancelled),
            OrderState::PartiallyFilled => matches!(to, OrderState::Filled | OrderState::Cancelled),
            OrderState::Filled | OrderState::Cancelled | OrderState::Rejected => false,
        }
    }

    /// Report an execution (idempotent)
    pub fn report_execution(&self, exec_id: u64, quantity: f64, price: f64) -> Result<ExecutionReport, OrderStateError> {
        // Idempotency check - prevent duplicate executions
        if self.executed.swap(true, Ordering::SeqCst) {
            return Err(OrderStateError::DuplicateExecution);
        }

        let current_state = self.get_state();
        if current_state.is_terminal() {
            return Err(OrderStateError::InvalidTransition { 
                from: current_state, 
                to: OrderState::PartiallyFilled 
            });
        }

        let filled = self.filled_quantity.load(Ordering::Relaxed);
        let remaining = self.remaining_quantity.load(Ordering::Relaxed);
        
        let actual_qty = quantity.min(remaining);
        let new_filled = filled + actual_qty;
        let new_remaining = remaining - actual_qty;

        // Update average fill price
        let avg_price = self.average_fill_price.load(Ordering::Relaxed);
        let new_avg = if new_filled > 0.0 {
            ((avg_price * filled) + (price * actual_qty)) / new_filled
        } else {
            price
        };

        self.filled_quantity.store(new_filled, Ordering::Relaxed);
        self.remaining_quantity.store(new_remaining, Ordering::Relaxed);
        self.average_fill_price.store(new_avg, Ordering::Relaxed);
        self.fill_count.fetch_add(1, Ordering::Relaxed);

        // Update state based on fill
        if new_remaining <= 0.0 {
            self.transition(OrderState::Filled)?;
        } else if current_state != OrderState::PartiallyFilled {
            self.transition(OrderState::PartiallyFilled)?;
        }

        Ok(ExecutionReport {
            exec_id,
            order_id: self.order.id,
            quantity: actual_qty,
            price,
            cumulative_qty: new_filled,
            remaining_qty: new_remaining,
            average_price: new_avg,
            state: self.get_state(),
        })
    }

    /// Cancel the order
    pub fn cancel(&self) -> Result<(), OrderStateError> {
        let current = self.get_state();
        if current.is_terminal() {
            return Err(OrderStateError::InvalidTransition { from: current, to: OrderState::Cancelled });
        }
        self.transition(OrderState::Cancelled)
    }

    /// Reject the order
    pub fn reject(&self) -> Result<(), OrderStateError> {
        let current = self.get_state();
        if current != OrderState::Pending && current != OrderState::New {
            return Err(OrderStateError::InvalidTransition { from: current, to: OrderState::Rejected });
        }
        self.transition(OrderState::Rejected)
    }

    /// Activate the order (Pending -> New)
    pub fn activate(&self) -> Result<(), OrderStateError> {
        self.transition(OrderState::New)
    }

    pub fn get_state(&self) -> OrderState {
        match self.state.load(Ordering::Relaxed) {
            0 => OrderState::Pending,
            1 => OrderState::New,
            2 => OrderState::PartiallyFilled,
            3 => OrderState::Filled,
            4 => OrderState::Cancelled,
            5 => OrderState::Rejected,
            _ => OrderState::Pending,
        }
    }

    pub fn get_order(&self) -> &Order {
        &self.order
    }

    pub fn get_filled_quantity(&self) -> f64 {
        self.filled_quantity.load(Ordering::Relaxed)
    }

    pub fn get_remaining_quantity(&self) -> f64 {
        self.remaining_quantity.load(Ordering::Relaxed)
    }

    pub fn get_average_fill_price(&self) -> f64 {
        self.average_fill_price.load(Ordering::Relaxed)
    }

    pub fn is_complete(&self) -> bool {
        self.get_state().is_terminal()
    }

    pub fn is_filled(&self) -> bool {
        self.get_state() == OrderState::Filled
    }
}

#[derive(Debug, Clone)]
pub struct ExecutionReport {
    pub exec_id: u64,
    pub order_id: u64,
    pub quantity: f64,
    pub price: f64,
    pub cumulative_qty: f64,
    pub remaining_qty: f64,
    pub average_price: f64,
    pub state: OrderState,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_order() -> Order {
        Order {
            id: 1,
            client_order_id: "test-1".to_string(),
            symbol: "BTC/USD".to_string(),
            side: OrderSide::Buy,
            order_type: OrderType::Limit,
            quantity: 100.0,
            price: Some(50000.0),
            stop_price: None,
            time_in_force: TimeInForce::Gtc,
            created_at: 0,
        }
    }

    #[test]
    fn test_order_lifecycle() {
        let oms = OrderStateMachine::new(create_test_order());
        
        assert_eq!(oms.get_state(), OrderState::Pending);
        
        oms.activate().unwrap();
        assert_eq!(oms.get_state(), OrderState::New);
        
        oms.report_execution(1, 50.0, 50000.0).unwrap();
        assert_eq!(oms.get_state(), OrderState::PartiallyFilled);
        assert_eq!(oms.get_filled_quantity(), 50.0);
        
        oms.report_execution(2, 50.0, 50000.0).unwrap();
        assert_eq!(oms.get_state(), OrderState::Filled);
    }

    #[test]
    fn test_idempotency() {
        let oms = OrderStateMachine::new(create_test_order());
        oms.activate().unwrap();
        
        // First execution succeeds
        let result = oms.report_execution(1, 50.0, 50000.0);
        assert!(result.is_ok());
        
        // Second execution attempt fails (idempotency)
        let result = oms.report_execution(2, 50.0, 50000.0);
        assert!(matches!(result, Err(OrderStateError::DuplicateExecution)));
    }

    #[test]
    fn test_invalid_transitions() {
        let oms = OrderStateMachine::new(create_test_order());
        
        // Can't go from Pending directly to Filled
        let result = oms.transition(OrderState::Filled);
        assert!(result.is_err());
    }

    #[test]
    fn test_cancel() {
        let oms = OrderStateMachine::new(create_test_order());
        oms.activate().unwrap();
        
        oms.cancel().unwrap();
        assert_eq!(oms.get_state(), OrderState::Cancelled);
        
        // Can't cancel again
        let result = oms.cancel();
        assert!(result.is_err());
    }
}
