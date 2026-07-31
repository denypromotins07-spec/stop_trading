//! Advanced Execution Module Root
//! 
//! This module handles exchange-specific matching engine quirks and advanced order flags.
//! It provides unified interfaces for complex execution strategies including iceberg orders,
//! post-only enforcement, and other advanced execution patterns.

pub mod iceberg_refresh;
pub mod post_only;

// Re-export main types for convenience
pub use iceberg_refresh::{
    ChildOrderSpec, IcebergBuilder, IcebergConfig, IcebergError, IcebergManager,
    IcebergOrder, IcebergStats, MarketEvent, MarketEventType, Side as IcebergSide,
};

pub use post_only::{
    CancelDecision, ModificationReason, OrderStatus, PostOnlyConfig, PostOnlyError,
    PostOnlyManager, PostOnlyOrder, PostOnlyOrderBuilder, PostOnlyStats,
    PostOnlyValidationResult, Side as PostOnlySide,
};

/// Unified side type that can be converted between modules
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

impl From<IcebergSide> for Side {
    fn from(side: IcebergSide) -> Self {
        match side {
            IcebergSide::Buy => Side::Buy,
            IcebergSide::Sell => Side::Sell,
        }
    }
}

impl From<PostOnlySide> for Side {
    fn from(side: PostOnlySide) -> Self {
        match side {
            PostOnlySide::Buy => Side::Buy,
            PostOnlySide::Sell => Side::Sell,
        }
    }
}

impl From<Side> for IcebergSide {
    fn from(side: Side) -> Self {
        match side {
            Side::Buy => IcebergSide::Buy,
            Side::Sell => IcebergSide::Sell,
        }
    }
}

impl From<Side> for PostOnlySide {
    fn from(side: Side) -> Self {
        match side {
            Side::Buy => PostOnlySide::Buy,
            Side::Sell => PostOnlySide::Sell,
        }
    }
}

/// Exchange-specific matching engine quirks
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExchangeQuirk {
    /// Binance: Minimum price/quantity precision varies by symbol
    BinancePrecision,
    /// Binance: Self-trade prevention behavior
    BinanceStp,
    /// Coinbase: Order size limits change with volatility
    CoinbaseSizeLimits,
    /// FTX-style: Liquidation engine behavior
    FtxLiquidation,
    /// OKX: Partial fill handling quirks
    OkxPartialFills,
    /// Bybit: Position mode (one-way vs hedge)
    BybitPositionMode,
    /// Kraken: Batch order rate limits
    KrakenBatchLimits,
    /// Generic: Unknown quirk, handle conservatively
    Unknown,
}

/// Time-in-force policies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeInForce {
    /// Good till cancelled
    GTC,
    /// Immediate or cancel
    IOC,
    /// Fill or kill
    FOK,
    /// Good till date
    GTD(u64), // timestamp_ns
    /// Post only (will reject if would execute immediately)
    PostOnly,
}

impl TimeInForce {
    pub fn as_binance_str(&self) -> &'static str {
        match self {
            TimeInForce::GTC => "GTC",
            TimeInForce::IOC => "IOC",
            TimeInForce::FOK => "FOK",
            TimeInForce::GTD(_) => "GTD",
            TimeInForce::PostOnly => "GTC", // Post-only is a separate flag on Binance
        }
    }
}

/// Order execution instruction
#[derive(Debug, Clone)]
pub struct ExecutionInstruction {
    /// Unique client order ID
    pub client_order_id: String,
    /// Symbol
    pub symbol: String,
    /// Side
    pub side: Side,
    /// Order type
    pub order_type: OrderType,
    /// Quantity
    pub quantity: i64,
    /// Price (for limit orders)
    pub price: Option<i64>,
    /// Time in force
    pub time_in_force: TimeInForce,
    /// Post-only flag
    pub post_only: bool,
    /// Reduce-only flag
    pub reduce_only: bool,
    /// Iceberg display quantity (if iceberg order)
    pub iceberg_qty: Option<i64>,
    /// Self-trade prevention mode
    pub stp_mode: StpMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderType {
    Market,
    Limit,
    StopLimit,
    StopMarket,
    TrailingStop,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StpMode {
    /// None - allow self-trades
    None,
    /// Expire maker order
    ExpireMaker,
    /// Expire taker order
    ExpireTaker,
    /// Cancel both orders
    CancelBoth,
    /// Decrement and cancel (reduce quantity)
    DecrementCancel,
}

impl ExecutionInstruction {
    /// Create a new market order instruction
    pub fn market(symbol: impl Into<String>, side: Side, quantity: i64) -> Self {
        Self {
            client_order_id: generate_client_order_id(),
            symbol: symbol.into(),
            side,
            order_type: OrderType::Market,
            quantity,
            price: None,
            time_in_force: TimeInForce::IOC,
            post_only: false,
            reduce_only: false,
            iceberg_qty: None,
            stp_mode: StpMode::None,
        }
    }
    
    /// Create a new limit order instruction
    pub fn limit(
        symbol: impl Into<String>,
        side: Side,
        quantity: i64,
        price: i64,
    ) -> Self {
        Self {
            client_order_id: generate_client_order_id(),
            symbol: symbol.into(),
            side,
            order_type: OrderType::Limit,
            quantity,
            price: Some(price),
            time_in_force: TimeInForce::GTC,
            post_only: false,
            reduce_only: false,
            iceberg_qty: None,
            stp_mode: StpMode::None,
        }
    }
    
    /// Set post-only flag
    pub fn with_post_only(mut self, post_only: bool) -> Self {
        self.post_only = post_only;
        self.time_in_force = if post_only {
            TimeInForce::PostOnly
        } else {
            self.time_in_force
        };
        self
    }
    
    /// Set iceberg quantity
    pub fn with_iceberg(mut self, display_qty: i64) -> Self {
        self.iceberg_qty = Some(display_qty);
        self
    }
    
    /// Set reduce-only flag
    pub fn with_reduce_only(mut self, reduce_only: bool) -> Self {
        self.reduce_only = reduce_only;
        self
    }
    
    /// Set self-trade prevention mode
    pub fn with_stp_mode(mut self, stp_mode: StpMode) -> Self {
        self.stp_mode = stp_mode;
        self
    }
}

/// Generate unique client order ID
pub fn generate_client_order_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_micros();
    
    // Add random suffix to prevent collisions
    let random_suffix = rand::random::<u16>();
    format!("HFT{}{:04X}", timestamp, random_suffix)
}

/// Execution result after order submission
#[derive(Debug, Clone)]
pub struct ExecutionResult {
    /// Client order ID
    pub client_order_id: String,
    /// Exchange order ID
    pub exchange_order_id: Option<String>,
    /// Status
    pub status: ExecutionStatus,
    /// Filled quantity
    pub filled_qty: i64,
    /// Average fill price
    pub avg_fill_price: Option<i64>,
    /// Fees paid
    pub fees_paid: i64,
    /// Whether we were maker or taker
    pub maker_taker: Option<MakerTaker>,
    /// Error message if rejected
    pub error_message: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionStatus {
    Pending,
    Submitted,
    PartiallyFilled,
    Filled,
    Cancelled,
    Rejected,
    Expired,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MakerTaker {
    Maker,
    Taker,
}

/// Execution manager coordinating all execution strategies
pub struct ExecutionManager {
    /// Iceberg manager
    iceberg_manager: IcebergManager,
    /// Post-only manager
    post_only_manager: PostOnlyManager,
    /// Active instructions
    pending_instructions: parking_lot::Mutex<std::collections::HashMap<String, ExecutionInstruction>>,
    /// Execution results
    results: parking_lot::Mutex<std::collections::HashMap<String, ExecutionResult>>,
}

impl ExecutionManager {
    pub fn new(iceberg_config: IcebergConfig, post_only_config: PostOnlyConfig) -> Self {
        Self {
            iceberg_manager: IcebergManager::new(iceberg_config),
            post_only_manager: PostOnlyManager::new(post_only_config),
            pending_instructions: parking_lot::Mutex::new(std::collections::HashMap::new()),
            results: parking_lot::Mutex::new(std::collections::HashMap::new()),
        }
    }
    
    /// Get reference to iceberg manager
    pub fn iceberg(&self) -> &IcebergManager {
        &self.iceberg_manager
    }
    
    /// Get reference to post-only manager
    pub fn post_only(&self) -> &PostOnlyManager {
        &self.post_only_manager
    }
    
    /// Record an execution result
    pub fn record_result(&self, result: ExecutionResult) {
        let mut results = self.results.lock();
        results.insert(result.client_order_id.clone(), result);
        
        // Remove from pending
        let mut pending = self.pending_instructions.lock();
        pending.remove(&result.client_order_id);
    }
    
    /// Get result for an order
    pub fn get_result(&self, client_order_id: &str) -> Option<ExecutionResult> {
        self.results.lock().get(client_order_id).cloned()
    }
    
    /// Check if circuit breakers are active
    pub fn any_circuit_breaker_active(&self) -> bool {
        self.post_only_manager.is_circuit_breaker_active()
    }
    
    /// Emergency stop - trip all circuit breakers
    pub fn emergency_stop(&self) {
        self.post_only_manager.trip_circuit_breaker();
    }
    
    /// Reset all circuit breakers
    pub fn reset_circuit_breakers(&self) {
        self.post_only_manager.reset_circuit_breaker();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_execution_instruction_builder() {
        let instr = ExecutionInstruction::limit("BTCUSDT", Side::Buy, 100, 50000)
            .with_post_only(true)
            .with_iceberg(10)
            .with_reduce_only(false);
        
        assert_eq!(instr.symbol, "BTCUSDT");
        assert!(instr.post_only);
        assert_eq!(instr.iceberg_qty, Some(10));
        assert_eq!(instr.price, Some(50000));
    }
    
    #[test]
    fn test_side_conversions() {
        let iceberg_side = IcebergSide::Buy;
        let unified: Side = iceberg_side.into();
        assert_eq!(unified, Side::Buy);
        
        let post_only_side: PostOnlySide = Side::Sell.into();
        assert_eq!(post_only_side, PostOnlySide::Sell);
    }
    
    #[test]
    fn test_generate_client_order_id() {
        let id1 = generate_client_order_id();
        let id2 = generate_client_order_id();
        
        assert!(id1.starts_with("HFT"));
        assert_ne!(id1, id2); // Should be unique
    }
    
    #[test]
    fn test_execution_manager() {
        let manager = ExecutionManager::new(
            IcebergConfig::default(),
            PostOnlyConfig::default(),
        );
        
        assert!(!manager.any_circuit_breaker_active());
        
        manager.emergency_stop();
        assert!(manager.any_circuit_breaker_active());
        
        manager.reset_circuit_breakers();
        assert!(!manager.any_circuit_breaker_active());
    }
}
