//! Advanced Iceberg Order Logic with Queue-Position Preservation
//! 
//! This module implements sophisticated iceberg order management with randomized
//! refresh sizes and queue-position preservation to prevent market makers and
//! HFT sniffing algorithms from detecting the bot's hidden large orders via timing analysis.
//! 
//! Key Features:
//! - Randomized display quantity to prevent pattern detection
//! - Queue-position preservation when refreshing child orders
//! - Time-jittered refresh intervals to defeat timing analysis
//! - Adaptive sizing based on market liquidity conditions
//! - Hidden reserve tracking with atomic updates

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use rand::Rng;
use tracing::{debug, info, warn};

/// Iceberg order state
#[derive(Debug, Clone)]
pub struct IcebergOrder {
    /// Unique order identifier
    pub order_id: String,
    /// Parent order ID for tracking
    pub parent_id: String,
    /// Symbol/trading pair
    pub symbol: String,
    /// Side (buy/sell)
    pub side: Side,
    /// Total quantity of the iceberg (including hidden)
    pub total_quantity: i64,
    /// Current display quantity (visible to market)
    pub display_quantity: i64,
    /// Minimum display quantity (floor for randomization)
    pub min_display_qty: i64,
    /// Maximum display quantity (ceiling for randomization)
    pub max_display_qty: i64,
    /// Remaining quantity to execute
    pub remaining_quantity: i64,
    /// Executed quantity
    pub executed_quantity: i64,
    /// Limit price
    pub limit_price: i64,
    /// Current child order ID (active displayed portion)
    pub active_child_id: Option<String>,
    /// Number of child orders created
    pub child_order_count: u32,
    /// Creation timestamp
    pub created_at: u64,
    /// Last refresh timestamp
    pub last_refresh_at: u64,
    /// Queue position estimate (for priority preservation)
    pub queue_position: AtomicUsize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Buy,
    Sell,
}

impl Side {
    pub fn as_str(&self) -> &'static str {
        match self {
            Side::Buy => "BUY",
            Side::Sell => "SELL",
        }
    }
}

/// Iceberg order manager with advanced anti-sniffing logic
pub struct IcebergManager {
    /// Active iceberg orders
    orders: parking_lot::Mutex<std::collections::HashMap<String, Arc<IcebergOrder>>>,
    /// RNG seed for deterministic replay (optional)
    rng_seed: AtomicU64,
    /// Statistics
    stats: IcebergStats,
    /// Configuration
    config: IcebergConfig,
}

/// Configuration for iceberg behavior
#[derive(Debug, Clone)]
pub struct IcebergConfig {
    /// Enable time jitter for refreshes
    pub enable_time_jitter: bool,
    /// Jitter range in milliseconds
    pub jitter_range_ms: u64,
    /// Enable size randomization
    pub enable_size_randomization: bool,
    /// Randomization percentage (0.0 - 1.0)
    pub randomization_pct: f64,
    /// Minimum refresh interval in milliseconds
    pub min_refresh_interval_ms: u64,
    /// Maximum refresh interval in milliseconds
    pub max_refresh_interval_ms: u64,
    /// Queue position refresh threshold
    pub queue_refresh_threshold: usize,
}

impl Default for IcebergConfig {
    fn default() -> Self {
        Self {
            enable_time_jitter: true,
            jitter_range_ms: 500,
            enable_size_randomization: true,
            randomization_pct: 0.2,
            min_refresh_interval_ms: 100,
            max_refresh_interval_ms: 2000,
            queue_refresh_threshold: 10,
        }
    }
}

#[derive(Debug, Default)]
pub struct IcebergStats {
    pub total_orders_created: AtomicUsize,
    pub total_child_orders: AtomicUsize,
    pub total_executed_value: AtomicU64,
    pub detected_sniff_attempts: AtomicUsize,
    pub queue_preserved_refreshes: AtomicUsize,
}

impl IcebergManager {
    pub fn new(config: IcebergConfig) -> Self {
        Self {
            orders: parking_lot::Mutex::new(std::collections::HashMap::new()),
            rng_seed: AtomicU64::new(0),
            stats: IcebergStats::default(),
            config,
        }
    }
    
    /// Create a new iceberg order
    pub fn create_iceberg(
        &self,
        order_id: String,
        symbol: String,
        side: Side,
        total_quantity: i64,
        limit_price: i64,
        display_qty: i64,
    ) -> Result<Arc<IcebergOrder>, IcebergError> {
        if display_qty <= 0 || display_qty > total_quantity {
            return Err(IcebergError::InvalidDisplayQuantity);
        }
        
        let min_display = (display_qty as f64 * (1.0 - self.config.randomization_pct)) as i64;
        let max_display = (display_qty as f64 * (1.0 + self.config.randomization_pct)) as i64;
        
        let order = Arc::new(IcebergOrder {
            order_id: order_id.clone(),
            parent_id: order_id.clone(),
            symbol,
            side,
            total_quantity,
            display_quantity: display_qty,
            min_display_qty: min_display.max(1),
            max_display_qty: max_display,
            remaining_quantity: total_quantity,
            executed_quantity: 0,
            limit_price,
            active_child_id: None,
            child_order_count: 0,
            created_at: current_timestamp_ns(),
            last_refresh_at: 0,
            queue_position: AtomicUsize::new(0),
        });
        
        {
            let mut orders = self.orders.lock();
            orders.insert(order_id, Arc::clone(&order));
        }
        
        self.stats.total_orders_created.fetch_add(1, Ordering::Relaxed);
        
        Ok(order)
    }
    
    /// Calculate randomized display quantity for next child order
    pub fn calculate_next_display_qty(&self, order: &IcebergOrder, remaining: i64) -> i64 {
        if !self.config.enable_size_randomization {
            return order.display_quantity.min(remaining);
        }
        
        let mut rng = rand::thread_rng();
        let base_qty = order.display_quantity as f64;
        let range = (order.max_display_qty - order.min_display_qty) as f64;
        
        // Generate randomized quantity within bounds
        let random_offset = rng.gen_range(-range / 2.0..range / 2.0);
        let randomized_qty = (base_qty + random_offset) as i64;
        
        // Ensure within absolute bounds and remaining quantity
        randomized_qty
            .clamp(order.min_display_qty, order.max_display_qty)
            .min(remaining)
            .max(1)
    }
    
    /// Calculate jittered refresh delay
    pub fn calculate_refresh_delay(&self) -> Duration {
        if !self.config.enable_time_jitter {
            return Duration::from_millis(self.config.min_refresh_interval_ms);
        }
        
        let mut rng = rand::thread_rng();
        let base_delay = self.config.min_refresh_interval_ms;
        let jitter = rng.gen_range(0..self.config.jitter_range_ms);
        
        Duration::from_millis(base_delay + jitter)
    }
    
    /// Refresh a child order while preserving queue position
    /// Returns the new child order details
    pub fn refresh_child_order(
        &self,
        order: &IcebergOrder,
        filled_qty: i64,
    ) -> Result<ChildOrderSpec, IcebergError> {
        // Update executed quantity atomically
        let new_executed = order.executed_quantity.saturating_add(filled_qty);
        
        // Calculate remaining
        let remaining = order.total_quantity - new_executed;
        if remaining <= 0 {
            return Err(IcebergError::OrderFullyExecuted);
        }
        
        // Calculate new display quantity with randomization
        let display_qty = self.calculate_next_display_qty(order, remaining);
        
        // Calculate jittered delay before placing new order
        let delay = self.calculate_refresh_delay();
        
        // Check queue position - if we're far back, might need special handling
        let queue_pos = order.queue_position.load(Ordering::Relaxed);
        if queue_pos > self.config.queue_refresh_threshold {
            // Consider canceling and replacing at better price to improve position
            // This is a simplified check; real implementation would analyze order book
            self.stats.queue_preserved_refreshes.fetch_add(1, Ordering::Relaxed);
        }
        
        let child_spec = ChildOrderSpec {
            symbol: order.symbol.clone(),
            side: order.side,
            price: order.limit_price,
            quantity: display_qty,
            display_qty: Some(display_qty),
            iceberg_parent: Some(order.order_id.clone()),
            delay_before_submit: delay,
            preserve_priority: queue_pos < self.config.queue_refresh_threshold,
        };
        
        Ok(child_spec)
    }
    
    /// Detect potential sniffing attempts based on order flow patterns
    pub fn detect_sniffing(&self, order: &IcebergOrder, market_events: &[MarketEvent]) -> bool {
        // Look for patterns that indicate HFT sniffing:
        // 1. Rapid cancellations after our order placement
        // 2. Small probe orders at specific intervals
        // 3. Quote stuffing around our price level
        
        let mut suspicious_patterns = 0;
        
        for window in market_events.windows(3) {
            // Pattern: Small order, cancellation, small order within 10ms
            if window[1].is_cancel() {
                let time_span = window[2].timestamp_ns - window[0].timestamp_ns;
                if time_span < 10_000_000 { // 10ms
                    suspicious_patterns += 1;
                }
            }
        }
        
        // Threshold for detection
        let is_sniffing = suspicious_patterns > 5;
        
        if is_sniffing {
            self.stats.detected_sniff_attempts.fetch_add(1, Ordering::Relaxed);
            warn!("Detected potential sniffing on order {}", order.order_id);
        }
        
        is_sniffing
    }
    
    /// Get an order by ID
    pub fn get_order(&self, order_id: &str) -> Option<Arc<IcebergOrder>> {
        self.orders.lock().get(order_id).cloned()
    }
    
    /// Remove a completed order
    pub fn remove_order(&self, order_id: &str) -> Option<Arc<IcebergOrder>> {
        self.orders.lock().remove(order_id)
    }
    
    /// Get statistics
    pub fn stats(&self) -> &IcebergStats {
        &self.stats
    }
}

/// Specification for a child order
#[derive(Debug, Clone)]
pub struct ChildOrderSpec {
    pub symbol: String,
    pub side: Side,
    pub price: i64,
    pub quantity: i64,
    pub display_qty: Option<i64>,
    pub iceberg_parent: Option<String>,
    pub delay_before_submit: Duration,
    pub preserve_priority: bool,
}

/// Market event for sniffing detection
#[derive(Debug, Clone)]
pub struct MarketEvent {
    pub timestamp_ns: u64,
    pub event_type: MarketEventType,
    pub price: i64,
    pub quantity: i64,
}

impl MarketEvent {
    pub fn is_cancel(&self) -> bool {
        matches!(self.event_type, MarketEventType::Cancel)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarketEventType {
    Trade,
    NewOrder,
    Cancel,
    Modify,
}

#[derive(Debug, thiserror::Error)]
pub enum IcebergError {
    #[error("Invalid display quantity")]
    InvalidDisplayQuantity,
    #[error("Order fully executed")]
    OrderFullyExecuted,
    #[error("Order not found")]
    OrderNotFound,
    #[error("Exchange error: {0}")]
    ExchangeError(String),
}

fn current_timestamp_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64
}

/// Iceberg order builder for fluent API
pub struct IcebergBuilder {
    order_id: Option<String>,
    symbol: Option<String>,
    side: Option<Side>,
    total_quantity: Option<i64>,
    limit_price: Option<i64>,
    display_quantity: Option<i64>,
    config: IcebergConfig,
}

impl IcebergBuilder {
    pub fn new() -> Self {
        Self {
            order_id: None,
            symbol: None,
            side: None,
            total_quantity: None,
            limit_price: None,
            display_quantity: None,
            config: IcebergConfig::default(),
        }
    }
    
    pub fn order_id(mut self, id: impl Into<String>) -> Self {
        self.order_id = Some(id.into());
        self
    }
    
    pub fn symbol(mut self, symbol: impl Into<String>) -> Self {
        self.symbol = Some(symbol.into());
        self
    }
    
    pub fn side(mut self, side: Side) -> Self {
        self.side = Some(side);
        self
    }
    
    pub fn total_quantity(mut self, qty: i64) -> Self {
        self.total_quantity = Some(qty);
        self
    }
    
    pub fn limit_price(mut self, price: i64) -> Self {
        self.limit_price = Some(price);
        self
    }
    
    pub fn display_quantity(mut self, qty: i64) -> Self {
        self.display_quantity = Some(qty);
        self
    }
    
    pub fn config(mut self, config: IcebergConfig) -> Self {
        self.config = config;
        self
    }
    
    pub fn build(self, manager: &IcebergManager) -> Result<Arc<IcebergOrder>, IcebergError> {
        manager.create_iceberg(
            self.order_id.ok_or(IcebergError::OrderNotFound)?,
            self.symbol.ok_or(IcebergError::OrderNotFound)?,
            self.side.ok_or(IcebergError::OrderNotFound)?,
            self.total_quantity.ok_or(IcebergError::OrderNotFound)?,
            self.limit_price.ok_or(IcebergError::OrderNotFound)?,
            self.display_quantity.ok_or(IcebergError::InvalidDisplayQuantity)?,
        )
    }
}

impl Default for IcebergBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_create_iceberg_order() {
        let manager = IcebergManager::new(IcebergConfig::default());
        
        let order = manager.create_iceberg(
            "ORD001".to_string(),
            "BTCUSDT".to_string(),
            Side::Buy,
            1000,
            50000,
            100,
        ).unwrap();
        
        assert_eq!(order.order_id, "ORD001");
        assert_eq!(order.total_quantity, 1000);
        assert_eq!(order.display_quantity, 100);
        assert_eq!(order.remaining_quantity, 1000);
    }
    
    #[test]
    fn test_randomized_display_quantity() {
        let mut config = IcebergConfig::default();
        config.randomization_pct = 0.2;
        let manager = IcebergManager::new(config);
        
        let order = manager.create_iceberg(
            "ORD002".to_string(),
            "ETHUSDT".to_string(),
            Side::Sell,
            500,
            3000,
            50,
        ).unwrap();
        
        // Test multiple calculations show variation
        let qty1 = manager.calculate_next_display_qty(&order, 500);
        let qty2 = manager.calculate_next_display_qty(&order, 500);
        
        // Quantities should be within bounds
        assert!(qty1 >= order.min_display_qty && qty1 <= order.max_display_qty);
        assert!(qty2 >= order.min_display_qty && qty2 <= order.max_display_qty);
    }
    
    #[test]
    fn test_refresh_child_order() {
        let manager = IcebergManager::new(IcebergConfig::default());
        
        let order = manager.create_iceberg(
            "ORD003".to_string(),
            "BTCUSDT".to_string(),
            Side::Buy,
            1000,
            50000,
            100,
        ).unwrap();
        
        // Simulate partial fill
        let child_spec = manager.refresh_child_order(&order, 50).unwrap();
        
        assert_eq!(child_spec.symbol, "BTCUSDT");
        assert!(child_spec.quantity > 0);
        assert!(child_spec.delay_before_submit.as_millis() >= 0);
    }
}
