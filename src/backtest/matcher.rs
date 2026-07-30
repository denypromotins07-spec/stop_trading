//! Simulated Matching Engine for Backtesting
//! 
//! Implements a realistic L2 order book traversal for limit order fill simulation,
//! modeling adverse selection, partial fills, and toxic execution rejection.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use crate::market_data::types::{Side, Tick, Level};
use crate::backtest::engine::BacktestTrade;

/// Order in the simulated book
#[derive(Debug, Clone)]
pub struct SimulatedOrder {
    pub order_id: u64,
    pub symbol: String,
    pub side: Side,
    pub price: f64,
    pub quantity: f64,
    pub filled_quantity: f64,
    pub timestamp_ns: u64,
    pub priority: u64, // Queue position priority
}

/// L2 Order Book level for simulation
#[derive(Debug, Clone, Default)]
pub struct BookLevel {
    pub bids: Vec<SimulatedOrder>,
    pub asks: Vec<SimulatedOrder>,
    pub total_bid_volume: f64,
    pub total_ask_volume: f64,
}

/// Simulated L2 Order Book
pub struct SimulatedOrderBook {
    pub symbol: String,
    pub levels: BTreeMap<f64, BookLevel>,
    pub best_bid: Option<f64>,
    pub best_ask: Option<f64>,
    pub last_update_ns: u64,
    /// Adverse selection indicator (higher = more toxic flow)
    pub toxic_flow_score: f64,
    /// Recent fill history for adverse selection modeling
    pub recent_fills: Vec<FillEvent>,
}

/// Fill event for tracking
#[derive(Debug, Clone)]
pub struct FillEvent {
    pub timestamp_ns: u64,
    pub side: Side,
    pub price: f64,
    pub quantity: f64,
    pub was_aggressive: bool,
}

impl SimulatedOrderBook {
    /// Create a new simulated order book
    pub fn new(symbol: String) -> Self {
        Self {
            symbol,
            levels: BTreeMap::new(),
            best_bid: None,
            best_ask: None,
            last_update_ns: 0,
            toxic_flow_score: 0.0,
            recent_fills: Vec::with_capacity(100),
        }
    }

    /// Update book from L2 snapshot
    pub fn update_from_snapshot(&mut self, bids: &[Level], asks: &[Level], timestamp_ns: u64) {
        self.levels.clear();
        
        // Process bids (descending price)
        for level in bids {
            let entry = self.levels.entry(level.price).or_insert_with(BookLevel::default);
            entry.total_bid_volume = level.volume;
        }
        
        // Process asks (ascending price)
        for level in asks {
            let entry = self.levels.entry(level.price).or_insert_with(BookLevel::default);
            entry.total_ask_volume = level.volume;
        }
        
        // Update best bid/ask
        self.best_bid = bids.first().map(|l| l.price);
        self.best_ask = asks.first().map(|l| l.price);
        self.last_update_ns = timestamp_ns;
    }

    /// Update book from tick
    pub fn update_from_tick(&mut self, tick: &Tick) {
        self.last_update_ns = tick.timestamp_ns;
        
        // Track aggressive fills for adverse selection
        if tick.last_price > self.best_ask.unwrap_or(f64::MAX) * 0.9999 {
            // Aggressive buy
            self.recent_fills.push(FillEvent {
                timestamp_ns: tick.timestamp_ns,
                side: Side::Buy,
                price: tick.last_price,
                quantity: tick.volume,
                was_aggressive: true,
            });
            self.toxic_flow_score += 0.1;
        } else if tick.last_price < self.best_bid.unwrap_or(0.0) * 1.0001 {
            // Aggressive sell
            self.recent_fills.push(FillEvent {
                timestamp_ns: tick.timestamp_ns,
                side: Side::Sell,
                price: tick.last_price,
                quantity: tick.volume,
                was_aggressive: true,
            });
            self.toxic_flow_score += 0.1;
        }
        
        // Decay toxic flow score
        self.toxic_flow_score *= 0.99;
        
        // Keep only recent fills
        if self.recent_fills.len() > 100 {
            self.recent_fills.remove(0);
        }
    }

    /// Simulate limit order placement and potential fill
    pub fn simulate_limit_order(
        &mut self,
        order: &mut SimulatedOrder,
        config: &MatcherConfig,
    ) -> OrderSimulationResult {
        match order.side {
            Side::Buy => self.simulate_buy_order(order, config),
            Side::Sell => self.simulate_sell_order(order, config),
        }
    }

    /// Simulate buy order execution
    fn simulate_buy_order(
        &mut self,
        order: &mut SimulatedOrder,
        config: &MatcherConfig,
    ) -> OrderSimulationResult {
        let mut result = OrderSimulationResult::new(order.order_id);
        
        // Check if order would cross the spread
        if let Some(best_ask) = self.best_ask {
            if order.price >= best_ask {
                // Marketable order - execute immediately
                return self.execute_market_order(order, config);
            }
        }
        
        // Add to book and wait for execution
        self.add_to_book(order);
        
        // Calculate probability of fill based on queue position and market conditions
        let fill_probability = self.calculate_fill_probability(order, config);
        
        // Check for adverse selection
        if config.model_adverse_selection && self.is_toxic_flow(order) {
            result.rejected_reason = Some("Adverse selection detected".to_string());
            result.would_be_toxic = true;
            return result;
        }
        
        // Simulate partial fills based on incoming market orders
        let remaining = self.simulate_incoming_flow(order, fill_probability, config);
        order.filled_quantity = order.quantity - remaining;
        result.filled_quantity = order.filled_quantity;
        result.remaining_quantity = remaining;
        result.avg_fill_price = order.price;
        
        if remaining < order.quantity {
            result.partially_filled = true;
        }
        
        if remaining <= 0.0 {
            result.fully_filled = true;
        }
        
        result
    }

    /// Simulate sell order execution
    fn simulate_sell_order(
        &mut self,
        order: &mut SimulatedOrder,
        config: &MatcherConfig,
    ) -> OrderSimulationResult {
        let mut result = OrderSimulationResult::new(order.order_id);
        
        // Check if order would cross the spread
        if let Some(best_bid) = self.best_bid {
            if order.price <= best_bid {
                // Marketable order - execute immediately
                return self.execute_market_order(order, config);
            }
        }
        
        // Add to book and wait for execution
        self.add_to_book(order);
        
        // Calculate probability of fill
        let fill_probability = self.calculate_fill_probability(order, config);
        
        // Check for adverse selection
        if config.model_adverse_selection && self.is_toxic_flow(order) {
            result.rejected_reason = Some("Adverse selection detected".to_string());
            result.would_be_toxic = true;
            return result;
        }
        
        // Simulate partial fills
        let remaining = self.simulate_incoming_flow(order, fill_probability, config);
        order.filled_quantity = order.quantity - remaining;
        result.filled_quantity = order.filled_quantity;
        result.remaining_quantity = remaining;
        result.avg_fill_price = order.price;
        
        if remaining < order.quantity {
            result.partially_filled = true;
        }
        
        if remaining <= 0.0 {
            result.fully_filled = true;
        }
        
        result
    }

    /// Execute a marketable order immediately
    fn execute_market_order(
        &mut self,
        order: &SimulatedOrder,
        config: &MatcherConfig,
    ) -> OrderSimulationResult {
        let mut result = OrderSimulationResult::new(order.order_id);
        
        let mut remaining = order.quantity;
        let mut total_cost = 0.0;
        let mut fill_count = 0;
        
        // Traverse the book
        let prices_to_check: Vec<f64> = match order.side {
            Side::Buy => {
                self.levels.keys().copied().filter(|p| {
                    self.levels.get(p).map_or(false, |l| l.total_ask_volume > 0.0)
                }).collect()
            }
            Side::Sell => {
                self.levels.keys().copied().rev().filter(|p| {
                    self.levels.get(p).map_or(false, |l| l.total_bid_volume > 0.0)
                }).collect()
            }
        };
        
        for price in prices_to_check {
            if remaining <= 0.0 {
                break;
            }
            
            if order.side == Side::Buy && price > order.price {
                continue; // Price too high
            }
            if order.side == Side::Sell && price < order.price {
                continue; // Price too low
            }
            
            if let Some(level) = self.levels.get_mut(&price) {
                let available = match order.side {
                    Side::Buy => level.total_ask_volume,
                    Side::Sell => level.total_bid_volume,
                };
                
                let fill_qty = remaining.min(available);
                if fill_qty > 0.0 {
                    total_cost += fill_qty * price;
                    remaining -= fill_qty;
                    fill_count += 1;
                    
                    // Update level volume
                    match order.side {
                        Side::Buy => level.total_ask_volume -= fill_qty,
                        Side::Sell => level.total_bid_volume -= fill_qty,
                    }
                }
            }
        }
        
        result.filled_quantity = order.quantity - remaining;
        result.remaining_quantity = remaining;
        result.avg_fill_price = if result.filled_quantity > 0.0 {
            total_cost / result.filled_quantity
        } else {
            0.0
        };
        
        // Apply slippage model
        if config.apply_slippage {
            let slippage_factor = 1.0 + (config.slippage_bps / 10000.0);
            result.avg_fill_price *= slippage_factor;
        }
        
        if remaining <= 0.0 {
            result.fully_filled = true;
        } else if result.filled_quantity > 0.0 {
            result.partially_filled = true;
        }
        
        result
    }

    /// Add order to the book
    fn add_to_book(&mut self, order: &SimulatedOrder) {
        let level = self.levels.entry(order.price).or_insert_with(BookLevel::default);
        match order.side {
            Side::Buy => {
                level.bids.push(order.clone());
                level.total_bid_volume += order.quantity;
            }
            Side::Sell => {
                level.asks.push(order.clone());
                level.total_ask_volume += order.quantity;
            }
        }
    }

    /// Calculate probability of fill for a limit order
    fn calculate_fill_probability(
        &self,
        order: &SimulatedOrder,
        config: &MatcherConfig,
    ) -> f64 {
        let mut probability = 0.5; // Base probability
        
        // Distance from mid-price affects probability
        let mid_price = (self.best_bid.unwrap_or(0.0) + self.best_ask.unwrap_or(f64::MAX)) / 2.0;
        let distance_bps = ((order.price - mid_price).abs() / mid_price) * 10000.0;
        
        // Closer to mid = higher probability
        probability *= 1.0 - (distance_bps / 100.0).min(0.9);
        
        // Queue position affects probability
        probability *= config.queue_position_factor;
        
        // Volatility adjustment
        probability *= 1.0 + (self.toxic_flow_score * 0.5);
        
        probability.clamp(0.0, 1.0)
    }

    /// Detect toxic flow / adverse selection
    fn is_toxic_flow(&self, order: &SimulatedOrder) -> bool {
        if self.recent_fills.is_empty() {
            return false;
        }
        
        // Count aggressive fills in same direction
        let same_direction_fills = self.recent_fills.iter()
            .filter(|f| f.side == order.side && f.was_aggressive)
            .count();
        
        // If >70% of recent fills are aggressive in same direction, likely toxic
        let ratio = same_direction_fills as f64 / self.recent_fills.len() as f64;
        ratio > 0.7
    }

    /// Simulate incoming market flow against resting orders
    fn simulate_incoming_flow(
        &self,
        order: &SimulatedOrder,
        base_probability: f64,
        config: &MatcherConfig,
    ) -> f64 {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        let mut remaining = order.quantity;
        
        // Simulate N time steps
        for _ in 0..config.simulation_steps {
            if remaining <= 0.0 {
                break;
            }
            
            // Probability of incoming market order hitting this level
            let hit_probability = base_probability * config.incoming_flow_rate;
            
            if rng.gen::<f64>() < hit_probability {
                // Incoming order hits this level
                let fill_amount = remaining.min(rng.gen_range(0.1..=1.0) * order.quantity * 0.1);
                remaining -= fill_amount;
            }
        }
        
        remaining
    }
}

/// Configuration for the matching engine
#[derive(Debug, Clone)]
pub struct MatcherConfig {
    /// Model adverse selection
    pub model_adverse_selection: bool,
    /// Apply slippage to fills
    pub apply_slippage: bool,
    /// Slippage in basis points
    pub slippage_bps: f64,
    /// Queue position factor (0.0-1.0)
    pub queue_position_factor: f64,
    /// Rate of incoming market flow (0.0-1.0)
    pub incoming_flow_rate: f64,
    /// Number of simulation steps
    pub simulation_steps: usize,
}

impl Default for MatcherConfig {
    fn default() -> Self {
        Self {
            model_adverse_selection: true,
            apply_slippage: true,
            slippage_bps: 2.0,
            queue_position_factor: 0.5,
            incoming_flow_rate: 0.3,
            simulation_steps: 100,
        }
    }
}

/// Result of order simulation
#[derive(Debug, Clone, Default)]
pub struct OrderSimulationResult {
    pub order_id: u64,
    pub filled_quantity: f64,
    pub remaining_quantity: f64,
    pub avg_fill_price: f64,
    pub fully_filled: bool,
    pub partially_filled: bool,
    pub rejected_reason: Option<String>,
    pub would_be_toxic: bool,
    pub fill_events: Vec<FillEvent>,
}

impl OrderSimulationResult {
    pub fn new(order_id: u64) -> Self {
        Self {
            order_id,
            ..Default::default()
        }
    }
    
    /// Convert to backtest trade
    pub fn to_backtest_trade(&self, order: &SimulatedOrder) -> Option<BacktestTrade> {
        if self.filled_quantity <= 0.0 {
            return None;
        }
        
        Some(BacktestTrade {
            timestamp_ns: order.timestamp_ns,
            symbol: order.symbol.clone(),
            side: order.side,
            quantity: self.filled_quantity,
            price: order.price,
            fill_price: self.avg_fill_price,
            slippage_bps: (self.avg_fill_price - order.price).abs() / order.price * 10000.0,
            latency_us: 0,
            queue_position: 0,
            was_partial_fill: self.partially_filled,
            remaining_quantity: self.remaining_quantity,
            pnl: 0.0,
            fees: 0.0,
        })
    }
}

/// Multi-symbol matching engine manager
pub struct MatchingEngine {
    books: HashMap<String, SimulatedOrderBook>,
    config: MatcherConfig,
    next_order_id: u64,
}

impl MatchingEngine {
    pub fn new(config: MatcherConfig) -> Self {
        Self {
            books: HashMap::new(),
            config,
            next_order_id: 1,
        }
    }

    /// Get or create order book for symbol
    pub fn get_book(&mut self, symbol: &str) -> &mut SimulatedOrderBook {
        self.books.entry(symbol.to_string())
            .or_insert_with(|| SimulatedOrderBook::new(symbol.to_string()))
    }

    /// Submit order for simulation
    pub fn submit_order(
        &mut self,
        symbol: &str,
        side: Side,
        price: f64,
        quantity: f64,
        timestamp_ns: u64,
    ) -> OrderSimulationResult {
        let order_id = self.next_order_id;
        self.next_order_id += 1;
        
        let mut order = SimulatedOrder {
            order_id,
            symbol: symbol.to_string(),
            side,
            price,
            quantity,
            filled_quantity: 0.0,
            timestamp_ns,
            priority: timestamp_ns,
        };
        
        let book = self.get_book(symbol);
        book.simulate_limit_order(&mut order, &self.config)
    }

    /// Update book from L2 data
    pub fn update_book(&mut self, symbol: &str, bids: &[Level], asks: &[Level], timestamp_ns: u64) {
        let book = self.get_book(symbol);
        book.update_from_snapshot(bids, asks, timestamp_ns);
    }

    /// Update book from tick
    pub fn update_from_tick(&mut self, symbol: &str, tick: &Tick) {
        if let Some(book) = self.books.get_mut(symbol) {
            book.update_from_tick(tick);
        }
    }
}

// Conditional rand dependency
#[cfg(feature = "rand")]
mod rand_impl {
    pub use rand::Rng;
}
