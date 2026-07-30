//! Venue Adapter Trait Interfaces
//!
//! Defines strict trait interfaces for venue-specific adapters,
//! abstracting REST and WebSocket differences. All venue implementations
//! must adhere to these unified normalized data types.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// Venue type identifier
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VenueType {
    /// Centralized Exchange (CEX)
    CEX,
    /// Decentralized Exchange (DEX)
    DEX,
    /// Market Maker
    MarketMaker,
    /// Dark Pool
    DarkPool,
}

impl VenueType {
    #[inline]
    pub fn as_str(&self) -> &'static str {
        match self {
            VenueType::CEX => "CEX",
            VenueType::DEX => "DEX",
            VenueType::MarketMaker => "MM",
            VenueType::DarkPool => "DARK",
        }
    }
}

/// Connection configuration for a venue
#[repr(C)]
#[derive(Debug, Clone)]
pub struct ConnectionConfig {
    /// Venue ID
    pub venue_id: u32,
    /// Venue type
    pub venue_type: VenueType,
    /// WebSocket endpoint URL
    pub ws_endpoint: [u8; 256],
    /// REST API endpoint URL
    pub rest_endpoint: [u8; 256],
    /// API key (encrypted in production)
    pub api_key: [u8; 128],
    /// API secret (encrypted in production)
    pub api_secret: [u8; 256],
    /// Connection timeout in milliseconds
    pub connect_timeout_ms: u32,
    /// Heartbeat interval in milliseconds
    pub heartbeat_interval_ms: u32,
    /// Reconnect delay in milliseconds
    pub reconnect_delay_ms: u32,
    /// Maximum reconnection attempts
    pub max_reconnect_attempts: u32,
    /// Enable rate limiting
    pub rate_limit_enabled: bool,
    /// Maximum orders per second
    pub max_orders_per_second: u32,
}

impl Default for ConnectionConfig {
    fn default() -> Self {
        Self {
            venue_id: 0,
            venue_type: VenueType::CEX,
            ws_endpoint: [0u8; 256],
            rest_endpoint: [0u8; 256],
            api_key: [0u8; 128],
            api_secret: [0u8; 256],
            connect_timeout_ms: 5000,
            heartbeat_interval_ms: 30000,
            reconnect_delay_ms: 1000,
            max_reconnect_attempts: 10,
            rate_limit_enabled: true,
            max_orders_per_second: 100,
        }
    }
}

/// Order routing decision from the load balancer
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct OrderRoutingDecision {
    /// Target venue ID
    pub venue_id: u32,
    /// Expected latency in nanoseconds
    pub expected_latency_ns: u64,
    /// Available liquidity (base currency)
    pub available_liquidity: u64,
    /// Routing score (higher is better)
    pub score: u32,
    /// Is this a failover route
    pub is_failover: bool,
}

impl OrderRoutingDecision {
    #[inline]
    pub fn new(venue_id: u32, expected_latency_ns: u64, available_liquidity: u64) -> Self {
        // Calculate score based on latency and liquidity
        // Lower latency and higher liquidity = higher score
        let latency_score = if expected_latency_ns < 1_000_000 {
            100u32
        } else if expected_latency_ns < 5_000_000 {
            80u32
        } else if expected_latency_ns < 10_000_000 {
            60u32
        } else {
            40u32
        };

        let liquidity_score = if available_liquidity > 1_000_000_000 {
            100u32
        } else if available_liquidity > 100_000_000 {
            80u32
        } else if available_liquidity > 10_000_000 {
            60u32
        } else {
            40u32
        };

        Self {
            venue_id,
            expected_latency_ns,
            available_liquidity,
            score: (latency_score + liquidity_score) / 2,
            is_failover: false,
        }
    }

    #[inline]
    pub fn as_failover(mut self) -> Self {
        self.is_failover = true;
        self.score = self.score.saturating_sub(20);
        self
    }
}

/// Venue adapter trait - all venues must implement this
pub trait VenueAdapter: Send + Sync {
    /// Get venue ID
    fn venue_id(&self) -> u32;

    /// Get venue type
    fn venue_type(&self) -> VenueType;

    /// Connect to the venue
    fn connect(&self) -> Result<(), VenueError>;

    /// Disconnect from the venue
    fn disconnect(&self) -> Result<(), VenueError>;

    /// Check if connected
    fn is_connected(&self) -> bool;

    /// Submit an order
    fn submit_order(&self, order: &OrderRequest) -> Result<OrderResponse, VenueError>;

    /// Cancel an order
    fn cancel_order(&self, order_id: u64) -> Result<CancelResponse, VenueError>;

    /// Subscribe to market data
    fn subscribe_market_data(&self, symbols: &[&str]) -> Result<(), VenueError>;

    /// Unsubscribe from market data
    fn unsubscribe_market_data(&self, symbols: &[&str]) -> Result<(), VenueError>;

    /// Get current latency measurement
    fn get_latency_ns(&self) -> u64;

    /// Get available liquidity for a symbol
    fn get_liquidity(&self, symbol_hash: u64) -> LiquidityInfo;

    /// Send heartbeat
    fn send_heartbeat(&self) -> Result<(), VenueError>;

    /// Get connection statistics
    fn get_stats(&self) -> VenueStats;
}

/// Venue error types
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum VenueError {
    /// Not connected
    NotConnected,
    /// Connection failed
    ConnectionFailed,
    /// Authentication failed
    AuthenticationFailed,
    /// Rate limited
    RateLimited,
    /// Invalid request
    InvalidRequest,
    /// Timeout
    Timeout,
    /// Protocol error
    ProtocolError,
    /// Insufficient balance
    InsufficientBalance,
    /// Order rejected
    OrderRejected,
    /// Symbol not found
    SymbolNotFound,
    /// Internal error
    InternalError,
}

impl VenueError {
    #[inline]
    pub fn error_code(&self) -> u32 {
        match self {
            VenueError::NotConnected => 1,
            VenueError::ConnectionFailed => 2,
            VenueError::AuthenticationFailed => 3,
            VenueError::RateLimited => 4,
            VenueError::InvalidRequest => 5,
            VenueError::Timeout => 6,
            VenueError::ProtocolError => 7,
            VenueError::InsufficientBalance => 8,
            VenueError::OrderRejected => 9,
            VenueError::SymbolNotFound => 10,
            VenueError::InternalError => 99,
        }
    }
}

/// Order request structure
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct OrderRequest {
    /// Client order ID
    pub client_order_id: u64,
    /// Symbol hash
    pub symbol_hash: u64,
    /// Side: 0 = Buy, 1 = Sell
    pub side: u8,
    /// Order type: 0 = Limit, 1 = Market, 2 = IOC, 3 = FOK
    pub order_type: u8,
    /// Price (in quote ticks)
    pub price: u64,
    /// Quantity (in base ticks)
    pub quantity: u64,
    /// Time in force: 0 = GTC, 1 = IOC, 2 = FOK, 3 = GTD
    pub time_in_force: u8,
    /// Timestamp in nanoseconds
    pub timestamp_ns: u64,
}

impl OrderRequest {
    #[inline]
    pub fn new(
        client_order_id: u64,
        symbol_hash: u64,
        side: u8,
        order_type: u8,
        price: u64,
        quantity: u64,
        time_in_force: u8,
    ) -> Self {
        Self {
            client_order_id,
            symbol_hash,
            side,
            order_type,
            price,
            quantity,
            time_in_force,
            timestamp_ns: 0, // Will be set by clock module
        }
    }
}

/// Order response structure
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct OrderResponse {
    /// Order ID assigned by venue
    pub order_id: u64,
    /// Client order ID
    pub client_order_id: u64,
    /// Status: 0 = New, 1 = PartiallyFilled, 2 = Filled, 3 = Cancelled, 4 = Rejected
    pub status: u8,
    /// Average fill price
    pub avg_fill_price: u64,
    /// Total filled quantity
    pub filled_qty: u64,
    /// Remaining quantity
    pub remaining_qty: u64,
    /// Timestamp in nanoseconds
    pub timestamp_ns: u64,
    /// Error code if rejected
    pub error_code: u32,
}

impl OrderResponse {
    #[inline]
    pub fn is_filled(&self) -> bool {
        self.status == 2
    }

    #[inline]
    pub fn is_rejected(&self) -> bool {
        self.status == 4 || self.error_code != 0
    }
}

/// Cancel request/response structure
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CancelResponse {
    /// Order ID
    pub order_id: u64,
    /// Client order ID
    pub client_order_id: u64,
    /// Cancel successful
    pub success: bool,
    /// Remaining quantity at cancel time
    pub canceled_qty: u64,
    /// Timestamp in nanoseconds
    pub timestamp_ns: u64,
    /// Error code if failed
    pub error_code: u32,
}

/// Liquidity information
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct LiquidityInfo {
    /// Best bid price
    pub best_bid: u64,
    /// Best ask price
    pub best_ask: u64,
    /// Bid size (base currency)
    pub bid_size: u64,
    /// Ask size (base currency)
    pub ask_size: u64,
    /// Spread in ticks
    pub spread: u64,
    /// Mid price
    pub mid_price: u64,
    /// Timestamp in nanoseconds
    pub timestamp_ns: u64,
}

impl LiquidityInfo {
    #[inline]
    pub fn empty() -> Self {
        Self {
            best_bid: 0,
            best_ask: 0,
            bid_size: 0,
            ask_size: 0,
            spread: 0,
            mid_price: 0,
            timestamp_ns: 0,
        }
    }

    #[inline]
    pub fn is_valid(&self) -> bool {
        self.best_bid > 0 && self.best_ask > 0 && self.best_bid < self.best_ask
    }
}

/// Venue statistics
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct VenueStats {
    /// Orders submitted
    pub orders_submitted: u64,
    /// Orders filled
    pub orders_filled: u64,
    /// Orders rejected
    pub orders_rejected: u64,
    /// Orders canceled
    pub orders_canceled: u64,
    /// Average latency in nanoseconds
    pub avg_latency_ns: u64,
    /// Last heartbeat timestamp
    pub last_heartbeat_ns: u64,
    /// Connection uptime in seconds
    pub uptime_seconds: u64,
    /// Error count
    pub error_count: u64,
}

impl VenueStats {
    #[inline]
    pub fn new() -> Self {
        Self {
            orders_submitted: 0,
            orders_filled: 0,
            orders_rejected: 0,
            orders_canceled: 0,
            avg_latency_ns: 0,
            last_heartbeat_ns: 0,
            uptime_seconds: 0,
            error_count: 0,
        }
    }
}

impl Default for VenueStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Base venue adapter with common functionality
#[repr(C)]
pub struct BaseVenueAdapter {
    /// Venue ID
    venue_id: u32,
    /// Venue type
    venue_type: VenueType,
    /// Connected state
    connected: AtomicBool,
    /// Last latency measurement
    last_latency_ns: AtomicU64,
    /// Orders submitted counter
    orders_submitted: AtomicU64,
    /// Connection start time
    connection_start_ns: AtomicU64,
}

impl BaseVenueAdapter {
    #[inline]
    pub fn new(venue_id: u32, venue_type: VenueType) -> Self {
        Self {
            venue_id,
            venue_type,
            connected: AtomicBool::new(false),
            last_latency_ns: AtomicU64::new(0),
            orders_submitted: AtomicU64::new(0),
            connection_start_ns: AtomicU64::new(0),
        }
    }

    #[inline]
    pub fn venue_id(&self) -> u32 {
        self.venue_id
    }

    #[inline]
    pub fn venue_type(&self) -> VenueType {
        self.venue_type
    }

    #[inline]
    pub fn is_connected(&self) -> bool {
        self.connected.load(Ordering::Acquire)
    }

    #[inline]
    pub fn set_connected(&self, connected: bool) {
        self.connected.store(connected, Ordering::Release);
        if connected {
            // Set connection start time (would use actual clock in production)
            self.connection_start_ns.store(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos() as u64,
                Ordering::Release,
            );
        }
    }

    #[inline]
    pub fn update_latency(&self, latency_ns: u64) {
        self.last_latency_ns.store(latency_ns, Ordering::Release);
    }

    #[inline]
    pub fn get_latency_ns(&self) -> u64 {
        self.last_latency_ns.load(Ordering::Acquire)
    }

    #[inline]
    pub fn increment_orders_submitted(&self) {
        self.orders_submitted.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn get_orders_submitted(&self) -> u64 {
        self.orders_submitted.load(Ordering::Relaxed)
    }

    #[inline]
    pub fn get_uptime_seconds(&self) -> u64 {
        let start = self.connection_start_ns.load(Ordering::Acquire);
        if start == 0 {
            return 0;
        }
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        now.saturating_sub(start) / 1_000_000_000
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_connection_config() {
        let config = ConnectionConfig::default();
        assert_eq!(config.venue_type, VenueType::CEX);
        assert_eq!(config.connect_timeout_ms, 5000);
        assert!(config.rate_limit_enabled);
    }

    #[test]
    fn test_order_routing_decision() {
        let decision = OrderRoutingDecision::new(1, 500_000, 500_000_000);
        assert_eq!(decision.venue_id, 1);
        assert!(decision.score > 0);
        assert!(!decision.is_failover);

        let failover = decision.as_failover();
        assert!(failover.is_failover);
        assert!(failover.score < decision.score);
    }

    #[test]
    fn test_order_request() {
        let order = OrderRequest::new(
            12345,
            67890,
            0, // Buy
            0, // Limit
            50000,
            1000,
            0, // GTC
        );
        assert_eq!(order.client_order_id, 12345);
        assert_eq!(order.side, 0);
    }

    #[test]
    fn test_liquidity_info() {
        let mut liq = LiquidityInfo::empty();
        assert!(!liq.is_valid());

        liq.best_bid = 49990;
        liq.best_ask = 50010;
        liq.bid_size = 1000;
        liq.ask_size = 1500;
        liq.spread = 20;
        liq.mid_price = 50000;

        assert!(liq.is_valid());
        assert_eq!(liq.spread, 20);
    }

    #[test]
    fn test_base_venue_adapter() {
        let adapter = BaseVenueAdapter::new(1, VenueType::CEX);
        
        assert!(!adapter.is_connected());
        assert_eq!(adapter.get_uptime_seconds(), 0);

        adapter.set_connected(true);
        assert!(adapter.is_connected());
        
        adapter.update_latency(1_000_000);
        assert_eq!(adapter.get_latency_ns(), 1_000_000);

        adapter.increment_orders_submitted();
        assert_eq!(adapter.get_orders_submitted(), 1);
    }
}
