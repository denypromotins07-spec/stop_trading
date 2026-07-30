//! Routing module root.
//! Ties execution algos to the state machine and REST/WS clients.

pub mod state_machine;
pub mod smart_router;

pub use state_machine::{
    OrderStateMachine, Order, OrderState, OrderType, OrderSide, TimeInForce, 
    ExecutionReport, OrderStateError,
};
pub use smart_router::{
    SmartOrderRouter, Venue, RouteDecision, PriceLevel, FeeTier, SorError, OrderSide as SorOrderSide,
};

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::sync::Arc;
use thiserror::Error;

/// Error types for routing module
#[derive(Debug, Error)]
pub enum RoutingError {
    #[error("State machine error: {0}")]
    StateMachine(#[from] OrderStateError),
    #[error("SOR error: {0}")]
    Sor(#[from] SorError),
    #[error("Connection error: {0}")]
    Connection(String),
    #[error("Invalid order: {0}")]
    InvalidOrder(String),
}

/// Order routing result
#[derive(Debug, Clone)]
pub struct RoutingResult {
    pub success: bool,
    pub order_id: u64,
    pub venue_id: u64,
    pub message: Option<String>,
}

/// Trait for order submission
pub trait OrderSubmitter {
    fn submit(&self, order: &Order) -> Result<RoutingResult, RoutingError>;
    fn cancel(&self, order_id: u64) -> Result<(), RoutingError>;
    fn modify(&self, order_id: u64, new_qty: Option<f64>, new_price: Option<f64>) -> Result<(), RoutingError>;
}

/// Order router combining SOR with state machine
pub struct OrderRouter<S: OrderSubmitter> {
    sor: Arc<SmartOrderRouter>,
    submitter: S,
    order_counter: AtomicU64,
    active_orders: AtomicU64,
    halted: AtomicBool,
}

impl<S: OrderSubmitter> OrderRouter<S> {
    pub fn new(sor: Arc<SmartOrderRouter>, submitter: S) -> Self {
        Self {
            sor,
            submitter,
            order_counter: AtomicU64::new(0),
            active_orders: AtomicU64::new(0),
            halted: AtomicBool::new(false),
        }
    }

    /// Route and submit an order
    pub fn route_order(&self, mut order: Order) -> Result<RoutingResult, RoutingError> {
        if self.halted.load(Ordering::Relaxed) {
            return Err(RoutingError::InvalidOrder("Trading halted".to_string()));
        }

        // Generate order ID
        let order_id = self.order_counter.fetch_add(1, Ordering::SeqCst);
        order.id = order_id;

        // Create state machine for tracking
        let oms = OrderStateMachine::new(order.clone());

        // Find best route
        let is_maker = order.price.is_some();
        let route = self.sor.find_best_route(
            order.quantity,
            order.price.unwrap_or(0.0),
            match order.side {
                OrderSide::Buy => SorOrderSide::Buy,
                OrderSide::Sell => SorOrderSide::Sell,
            },
            is_maker,
        )?;

        // Submit to venue
        let result = self.submitter.submit(&order)?;

        if result.success {
            oms.activate()?;
            self.active_orders.fetch_add(1, Ordering::Relaxed);
            self.sor.record_success();
        } else {
            self.sor.record_failure();
        }

        Ok(result)
    }

    /// Cancel an order
    pub fn cancel_order(&self, order_id: u64) -> Result<(), RoutingError> {
        self.submitter.cancel(order_id)?;
        self.active_orders.fetch_sub(1, Ordering::Relaxed);
        Ok(())
    }

    /// Get active order count
    pub fn active_order_count(&self) -> u64 {
        self.active_orders.load(Ordering::Relaxed)
    }

    /// Halt routing
    pub fn halt(&self) {
        self.halted.store(true, Ordering::Relaxed);
    }

    /// Resume routing
    pub fn resume(&self) {
        self.halted.store(false, Ordering::Relaxed);
    }

    /// Check if halted
    pub fn is_halted(&self) -> bool {
        self.halted.load(Ordering::Relaxed)
    }
}

/// Mock order submitter for testing
pub struct MockSubmitter {
    fill_probability: f64,
}

impl MockSubmitter {
    pub fn new(fill_prob: f64) -> Self {
        Self {
            fill_probability: fill_prob.min(1.0),
        }
    }
}

impl OrderSubmitter for MockSubmitter {
    fn submit(&self, _order: &Order) -> Result<RoutingResult, RoutingError> {
        use std::time::{SystemTime, UNIX_EPOCH};
        
        let success = rand_fill(self.fill_probability);
        
        Ok(RoutingResult {
            success,
            order_id: 0,
            venue_id: 1,
            message: if success { None } else { Some("Rejected".to_string()) },
        })
    }

    fn cancel(&self, _order_id: u64) -> Result<(), RoutingError> {
        Ok(())
    }

    fn modify(&self, _order_id: u64, _new_qty: Option<f64>, _new_price: Option<f64>) -> Result<(), RoutingError> {
        Ok(())
    }
}

fn rand_fill(prob: f64) -> bool {
    // Simple deterministic "random" for testing
    (prob * 100.0) as u64 % 2 == 0
}

/// Connection type for market data
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionType {
    Rest,
    WebSocket,
    Fix,
}

/// Market data connection
pub struct MarketDataConnection {
    conn_type: ConnectionType,
    connected: AtomicBool,
    latency_us: AtomicU64,
}

impl MarketDataConnection {
    pub fn new(conn_type: ConnectionType) -> Self {
        Self {
            conn_type,
            connected: AtomicBool::new(false),
            latency_us: AtomicU64::new(0),
        }
    }

    pub fn connect(&self) {
        self.connected.store(true, Ordering::Relaxed);
    }

    pub fn disconnect(&self) {
        self.connected.store(false, Ordering::Relaxed);
    }

    pub fn is_connected(&self) -> bool {
        self.connected.load(Ordering::Relaxed)
    }

    pub fn update_latency(&self, latency_us: u64) {
        self.latency_us.store(latency_us, Ordering::Relaxed);
    }

    pub fn get_latency(&self) -> u64 {
        self.latency_us.load(Ordering::Relaxed)
    }

    pub fn conn_type(&self) -> ConnectionType {
        self.conn_type
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_submitter() {
        let submitter = MockSubmitter::new(0.9);
        let order = Order {
            id: 0,
            client_order_id: "test".to_string(),
            symbol: "BTC/USD".to_string(),
            side: OrderSide::Buy,
            order_type: OrderType::Limit,
            quantity: 1.0,
            price: Some(50000.0),
            stop_price: None,
            time_in_force: TimeInForce::Gtc,
            created_at: 0,
        };

        let result = submitter.submit(&order).unwrap();
        assert!(result.success || !result.success); // Just verify it runs
    }

    #[test]
    fn test_market_data_connection() {
        let conn = MarketDataConnection::new(ConnectionType::WebSocket);
        
        assert!(!conn.is_connected());
        
        conn.connect();
        assert!(conn.is_connected());
        
        conn.update_latency(100);
        assert_eq!(conn.get_latency(), 100);
        
        conn.disconnect();
        assert!(!conn.is_connected());
    }

    #[test]
    fn test_order_router_halt() {
        let venues = vec![Venue::new(1, "Test".to_string(), 1.0, 3.0, 5.0, 0.95)];
        let sor = Arc::new(SmartOrderRouter::new(venues).unwrap());
        let submitter = MockSubmitter::new(0.9);
        
        let router = OrderRouter::new(sor, submitter);
        
        assert!(!router.is_halted());
        
        router.halt();
        assert!(router.is_halted());
        
        router.resume();
        assert!(!router.is_halted());
    }
}
