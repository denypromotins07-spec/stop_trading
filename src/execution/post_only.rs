//! Post-Only Order Enforcement with Immediate Cancel/Amend Logic
//! 
//! This module implements strict Post-Only enforcement with immediate cancel/amend logic
//! if the matching engine threatens to cross the spread. Ensures the bot strictly captures
//! maker rebates and never accidentally executes as a taker during extreme volatility spikes.
//! 
//! Key Features:
//! - Real-time spread monitoring before order submission
//! - Immediate cancel if order would execute as taker
//! - Price adjustment logic to maintain post-only status
//! - Volatility spike detection and circuit breaker integration
//! - Maker rebate tracking and optimization

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use tracing::{debug, error, info, warn};

/// Post-only order state
#[derive(Debug, Clone)]
pub struct PostOnlyOrder {
    /// Unique order identifier
    pub order_id: String,
    /// Symbol/trading pair
    pub symbol: String,
    /// Side (buy/sell)
    pub side: Side,
    /// Original intended price
    pub intended_price: i64,
    /// Actual submitted price (may be adjusted for post-only)
    pub submitted_price: i64,
    /// Quantity
    pub quantity: i64,
    /// Current best bid (snapshot at submission time)
    pub best_bid: i64,
    /// Current best ask (snapshot at submission time)
    pub best_ask: i64,
    /// Whether order was modified from intended price
    pub price_modified: bool,
    /// Modification reason
    pub modification_reason: Option<ModificationReason>,
    /// Submission timestamp
    pub submitted_at: u64,
    /// Status
    pub status: OrderStatus,
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
    
    pub fn is_buy(&self) -> bool {
        matches!(self, Side::Buy)
    }
    
    pub fn is_sell(&self) -> bool {
        matches!(self, Side::Sell)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderStatus {
    Pending,
    Submitted,
    LiveOnBook,
    Cancelled,
    Rejected,
    Executed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModificationReason {
    /// Price adjusted to avoid crossing spread
    SpreadAvoidance,
    /// Price adjusted due to volatility spike
    VolatilityAdjustment,
    /// Price adjusted to maintain queue priority
    QueuePriority,
    /// Minimum tick size adjustment
    TickSizeAdjustment,
}

/// Post-only enforcement manager
pub struct PostOnlyManager {
    /// Active post-only orders
    orders: parking_lot::Mutex<std::collections::HashMap<String, Arc<PostOnlyOrder>>>,
    /// Current best bid/ask cache
    best_bid: AtomicI64,
    best_ask: AtomicI64,
    /// Last update time for spread data
    spread_last_updated: AtomicU64,
    /// Spread staleness threshold in nanoseconds
    spread_staleness_ns: u64,
    /// Statistics
    stats: PostOnlyStats,
    /// Configuration
    config: PostOnlyConfig,
    /// Circuit breaker for extreme volatility
    circuit_breaker: AtomicBool,
}

/// Configuration for post-only behavior
#[derive(Debug, Clone)]
pub struct PostOnlyConfig {
    /// Minimum spread required to submit order (in basis points)
    pub min_spread_bps: u16,
    /// Maximum price adjustment percentage (0.0 - 1.0)
    pub max_price_adjustment_pct: f64,
    /// Tick size for the symbol
    pub tick_size: i64,
    /// Enable volatility circuit breaker
    pub enable_circuit_breaker: bool,
    /// Volatility threshold (percentage move in window)
    pub volatility_threshold_pct: f64,
    /// Volatility lookback window in milliseconds
    pub volatility_window_ms: u64,
    /// Spread staleness threshold in milliseconds
    pub spread_staleness_ms: u64,
    /// Auto-cancel on fill risk detection
    pub auto_cancel_on_fill_risk: bool,
}

impl Default for PostOnlyConfig {
    fn default() -> Self {
        Self {
            min_spread_bps: 5, // 0.05% minimum spread
            max_price_adjustment_pct: 0.001, // 0.1% max adjustment
            tick_size: 1,
            enable_circuit_breaker: true,
            volatility_threshold_pct: 2.0, // 2% move triggers breaker
            volatility_window_ms: 1000, // 1 second window
            spread_staleness_ms: 100, // 100ms staleness threshold
            auto_cancel_on_fill_risk: true,
        }
    }
}

#[derive(Debug, Default)]
pub struct PostOnlyStats {
    pub total_orders_submitted: AtomicUsize,
    pub orders_live_on_book: AtomicUsize,
    pub orders_price_adjusted: AtomicUsize,
    pub orders_cancelled_pre_fill: AtomicUsize,
    pub maker_rebates_earned: AtomicU64,
    pub taker_fees_avoided: AtomicU64,
    pub circuit_breaker_trips: AtomicUsize,
    pub spread_violations_prevented: AtomicUsize,
}

use std::sync::atomic::AtomicI64;

impl PostOnlyManager {
    pub fn new(config: PostOnlyConfig) -> Self {
        Self {
            orders: parking_lot::Mutex::new(std::collections::HashMap::new()),
            best_bid: AtomicI64::new(0),
            best_ask: AtomicI64::new(0),
            spread_last_updated: AtomicU64::new(0),
            spread_staleness_ns: config.spread_staleness_ms * 1_000_000,
            stats: PostOnlyStats::default(),
            config,
            circuit_breaker: AtomicBool::new(false),
        }
    }
    
    /// Update best bid/ask from market data feed
    pub fn update_market_data(&self, best_bid: i64, best_ask: i64) {
        if best_bid > 0 && best_ask > 0 && best_bid < best_ask {
            self.best_bid.store(best_bid, Ordering::Relaxed);
            self.best_ask.store(best_ask, Ordering::Relaxed);
            self.spread_last_updated.store(current_timestamp_ns(), Ordering::Relaxed);
        }
    }
    
    /// Check if spread data is fresh enough for safe order submission
    pub fn is_spread_data_fresh(&self) -> bool {
        let last_update = self.spread_last_updated.load(Ordering::Relaxed);
        let now = current_timestamp_ns();
        now.saturating_sub(last_update) < self.spread_staleness_ns
    }
    
    /// Calculate current spread in basis points
    pub fn current_spread_bps(&self) -> Option<u16> {
        let bid = self.best_bid.load(Ordering::Relaxed);
        let ask = self.best_ask.load(Ordering::Relaxed);
        
        if bid <= 0 || ask <= 0 || bid >= ask {
            return None;
        }
        
        let mid = (bid + ask) / 2;
        let spread = ask - bid;
        Some(((spread as u64 * 10000) / mid as u64) as u16)
    }
    
    /// Validate and potentially adjust order for post-only compliance
    pub fn validate_post_only(
        &self,
        symbol: &str,
        side: Side,
        price: i64,
        quantity: i64,
    ) -> PostOnlyValidationResult {
        // Check circuit breaker
        if self.circuit_breaker.load(Ordering::Relaxed) {
            return PostOnlyValidationResult::CircuitBreakerActive;
        }
        
        // Check spread data freshness
        if !self.is_spread_data_fresh() {
            return PostOnlyValidationResult::StaleSpreadData;
        }
        
        // Check minimum spread requirement
        if let Some(spread_bps) = self.current_spread_bps() {
            if spread_bps < self.config.min_spread_bps {
                self.stats.spread_violations_prevented.fetch_add(1, Ordering::Relaxed);
                return PostOnlyValidationResult::SpreadTooTight(spread_bps);
            }
        }
        
        let best_bid = self.best_bid.load(Ordering::Relaxed);
        let best_ask = self.best_ask.load(Ordering::Relaxed);
        
        // Check if order would cross the spread
        let would_cross = match side {
            Side::Buy => price >= best_ask && best_ask > 0,
            Side::Sell => price <= best_bid && best_bid > 0,
        };
        
        if would_cross {
            // Calculate adjusted price to stay post-only
            let adjusted_price = match side {
                Side::Buy => {
                    // For buys, must be below best ask
                    (best_ask - self.config.tick_size).max(1)
                }
                Side::Sell => {
                    // For sells, must be above best bid
                    best_bid + self.config.tick_size
                }
            };
            
            // Check if adjustment is within tolerance
            let price_diff = (price - adjusted_price).abs() as f64;
            let adjustment_pct = price_diff / price as f64;
            
            if adjustment_pct > self.config.max_price_adjustment_pct {
                return PostOnlyValidationResult::AdjustmentExceedsLimit(adjustment_pct);
            }
            
            return PostOnlyValidationResult::RequiresAdjustment {
                original_price: price,
                adjusted_price,
                reason: ModificationReason::SpreadAvoidance,
            };
        }
        
        // Order is valid as post-only
        PostOnlyValidationResult::Valid
    }
    
    /// Submit a post-only order with automatic price adjustment
    pub fn submit_post_only(
        &self,
        order_id: String,
        symbol: String,
        side: Side,
        intended_price: i64,
        quantity: i64,
    ) -> Result<Arc<PostOnlyOrder>, PostOnlyError> {
        let validation = self.validate_post_only(&symbol, side, intended_price, quantity);
        
        let (submitted_price, price_modified, modification_reason) = match validation {
            PostOnlyValidationResult::Valid => (intended_price, false, None),
            PostOnlyValidationResult::RequiresAdjustment { adjusted_price, reason, .. } => {
                (adjusted_price, true, Some(reason))
            }
            PostOnlyValidationResult::CircuitBreakerActive => {
                return Err(PostOnlyError::CircuitBreakerActive);
            }
            PostOnlyValidationResult::StaleSpreadData => {
                return Err(PostOnlyError::StaleMarketData);
            }
            PostOnlyValidationResult::SpreadTooTight(spread) => {
                return Err(PostOnlyError::SpreadTooTight(spread));
            }
            PostOnlyValidationResult::AdjustmentExceedsLimit(pct) => {
                return Err(PostOnlyError::PriceAdjustmentTooLarge(pct));
            }
        };
        
        let best_bid = self.best_bid.load(Ordering::Relaxed);
        let best_ask = self.best_ask.load(Ordering::Relaxed);
        
        let order = Arc::new(PostOnlyOrder {
            order_id: order_id.clone(),
            symbol,
            side,
            intended_price,
            submitted_price,
            quantity,
            best_bid,
            best_ask,
            price_modified,
            modification_reason,
            submitted_at: current_timestamp_ns(),
            status: OrderStatus::Submitted,
        });
        
        {
            let mut orders = self.orders.lock();
            orders.insert(order_id, Arc::clone(&order));
        }
        
        self.stats.total_orders_submitted.fetch_add(1, Ordering::Relaxed);
        if price_modified {
            self.stats.orders_price_adjusted.fetch_add(1, Ordering::Relaxed);
        }
        
        Ok(order)
    }
    
    /// Immediate cancel if fill risk detected (e.g., spread crossed after submission)
    pub fn check_fill_risk_and_cancel(&self, order_id: &str) -> Option<CancelDecision> {
        let order = self.orders.lock().get(order_id).cloned()?;
        
        if order.status != OrderStatus::LiveOnBook && order.status != OrderStatus::Submitted {
            return None;
        }
        
        // Refresh current market data
        let best_bid = self.best_bid.load(Ordering::Relaxed);
        let best_ask = self.best_ask.load(Ordering::Relaxed);
        
        // Check if our order would now execute immediately
        let fill_risk = match order.side {
            Side::Buy => {
                // Our buy order at submitted_price would hit if best_ask <= our price
                best_ask > 0 && best_ask <= order.submitted_price
            }
            Side::Sell => {
                // Our sell order at submitted_price would hit if best_bid >= our price
                best_bid > 0 && best_bid >= order.submitted_price
            }
        };
        
        if fill_risk && self.config.auto_cancel_on_fill_risk {
            self.stats.orders_cancelled_pre_fill.fetch_add(1, Ordering::Relaxed);
            Some(CancelDecision::CancelToAvoidTakerExecution)
        } else if fill_risk {
            Some(CancelDecision::FillRiskDetectedButHold)
        } else {
            Some(CancelDecision::NoRisk)
        }
    }
    
    /// Trip circuit breaker on extreme volatility
    pub fn trip_circuit_breaker(&self) {
        if self.config.enable_circuit_breaker {
            self.circuit_breaker.store(true, Ordering::SeqCst);
            self.stats.circuit_breaker_trips.fetch_add(1, Ordering::Relaxed);
            warn!("Post-only circuit breaker tripped due to extreme volatility");
        }
    }
    
    /// Reset circuit breaker
    pub fn reset_circuit_breaker(&self) {
        self.circuit_breaker.store(false, Ordering::SeqCst);
    }
    
    /// Check if circuit breaker is active
    pub fn is_circuit_breaker_active(&self) -> bool {
        self.circuit_breaker.load(Ordering::Relaxed)
    }
    
    /// Record maker rebate earned
    pub fn record_maker_rebate(&self, rebate_amount: u64) {
        self.stats.maker_rebates_earned.fetch_add(rebate_amount, Ordering::Relaxed);
    }
    
    /// Record taker fee avoided
    pub fn record_taker_fee_avoided(&self, fee_amount: u64) {
        self.stats.taker_fees_avoided.fetch_add(fee_amount, Ordering::Relaxed);
    }
    
    /// Update order status
    pub fn update_order_status(&self, order_id: &str, status: OrderStatus) {
        let mut orders = self.orders.lock();
        if let Some(order) = orders.get_mut(order_id) {
            let mut mutable_order = (*order).clone();
            mutable_order.status = status;
            *order = Arc::new(mutable_order);
            
            if status == OrderStatus::LiveOnBook {
                self.stats.orders_live_on_book.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
    
    /// Get statistics
    pub fn stats(&self) -> &PostOnlyStats {
        &self.stats
    }
}

/// Result of post-only validation
#[derive(Debug, Clone)]
pub enum PostOnlyValidationResult {
    Valid,
    RequiresAdjustment {
        original_price: i64,
        adjusted_price: i64,
        reason: ModificationReason,
    },
    CircuitBreakerActive,
    StaleSpreadData,
    SpreadTooTight(u16),
    AdjustmentExceedsLimit(f64),
}

/// Cancel decision based on fill risk analysis
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CancelDecision {
    NoRisk,
    FillRiskDetectedButHold,
    CancelToAvoidTakerExecution,
}

#[derive(Debug, thiserror::Error)]
pub enum PostOnlyError {
    #[error("Circuit breaker is active")]
    CircuitBreakerActive,
    #[error("Market data is stale")]
    StaleMarketData,
    #[error("Spread too tight: {0} bps")]
    SpreadTooTight(u16),
    #[error("Price adjustment exceeds limit: {0}%")]
    PriceAdjustmentTooLarge(f64),
    #[error("Order not found")]
    OrderNotFound,
    #[error("Invalid parameters: {0}")]
    InvalidParameters(String),
}

fn current_timestamp_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64
}

/// Builder for post-only orders
pub struct PostOnlyOrderBuilder {
    order_id: Option<String>,
    symbol: Option<String>,
    side: Option<Side>,
    price: Option<i64>,
    quantity: Option<i64>,
}

impl PostOnlyOrderBuilder {
    pub fn new() -> Self {
        Self {
            order_id: None,
            symbol: None,
            side: None,
            price: None,
            quantity: None,
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
    
    pub fn price(mut self, price: i64) -> Self {
        self.price = Some(price);
        self
    }
    
    pub fn quantity(mut self, qty: i64) -> Self {
        self.quantity = Some(qty);
        self
    }
    
    pub fn build(self, manager: &PostOnlyManager) -> Result<Arc<PostOnlyOrder>, PostOnlyError> {
        manager.submit_post_only(
            self.order_id.ok_or(PostOnlyError::InvalidParameters("Missing order_id".into()))?,
            self.symbol.ok_or(PostOnlyError::InvalidParameters("Missing symbol".into()))?,
            self.side.ok_or(PostOnlyError::InvalidParameters("Missing side".into()))?,
            self.price.ok_or(PostOnlyError::InvalidParameters("Missing price".into()))?,
            self.quantity.ok_or(PostOnlyError::InvalidParameters("Missing quantity".into()))?,
        )
    }
}

impl Default for PostOnlyOrderBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_validate_post_only_valid() {
        let manager = PostOnlyManager::new(PostOnlyConfig::default());
        manager.update_market_data(9900, 10100); // Spread of 200
        
        let result = manager.validate_post_only("BTCUSDT", Side::Buy, 9800, 100);
        assert_eq!(result, PostOnlyValidationResult::Valid);
        
        let result = manager.validate_post_only("BTCUSDT", Side::Sell, 10200, 100);
        assert_eq!(result, PostOnlyValidationResult::Valid);
    }
    
    #[test]
    fn test_validate_post_only_requires_adjustment() {
        let manager = PostOnlyManager::new(PostOnlyConfig::default());
        manager.update_market_data(9900, 10100);
        
        // Buy order at or above ask would cross
        let result = manager.validate_post_only("BTCUSDT", Side::Buy, 10100, 100);
        assert!(matches!(result, PostOnlyValidationResult::RequiresAdjustment { .. }));
        
        // Sell order at or below bid would cross
        let result = manager.validate_post_only("BTCUSDT", Side::Sell, 9900, 100);
        assert!(matches!(result, PostOnlyValidationResult::RequiresAdjustment { .. }));
    }
    
    #[test]
    fn test_spread_too_tight() {
        let mut config = PostOnlyConfig::default();
        config.min_spread_bps = 50; // Require 0.5% spread
        let manager = PostOnlyManager::new(config);
        manager.update_market_data(9990, 10010); // Very tight spread (~0.2%)
        
        let result = manager.validate_post_only("BTCUSDT", Side::Buy, 9900, 100);
        assert!(matches!(result, PostOnlyValidationResult::SpreadTooTight(_)));
    }
    
    #[test]
    fn test_fill_risk_detection() {
        let manager = PostOnlyManager::new(PostOnlyConfig::default());
        manager.update_market_data(9900, 10100);
        
        let order = manager.submit_post_only(
            "ORD001".to_string(),
            "BTCUSDT".to_string(),
            Side::Buy,
            9800,
            100,
        ).unwrap();
        
        // Initially no risk
        let decision = manager.check_fill_risk_and_cancel("ORD001");
        assert_eq!(decision, Some(CancelDecision::NoRisk));
        
        // Simulate market moving against us
        manager.update_market_data(9900, 9750); // Ask dropped below our bid
        
        let decision = manager.check_fill_risk_and_cancel("ORD001");
        assert_eq!(decision, Some(CancelDecision::CancelToAvoidTakerExecution));
    }
}
