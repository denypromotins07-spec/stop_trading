//! DEX Aggregator Module
//! 
//! Normalizes quotes from multiple DEXs (Uniswap V3, Raydium, Jupiter) into a unified
//! L2 order book format. Streams on-chain liquidity pool state changes via WebSocket
//! subscriptions to maintain a real-time decentralized order book.

use std::collections::{HashMap, BTreeMap};
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::watch;
use tracing::{debug, info, warn, error};

use crate::market_data::types::{Side, Level, Tick};

/// Supported DEX venues
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DexVenue {
    UniswapV3,
    UniswapV2,
    Raydium,
    Orca,
    Jupiter,
    Curve,
    Balancer,
}

impl DexVenue {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::UniswapV3 => "uniswap_v3",
            Self::UniswapV2 => "uniswap_v2",
            Self::Raydium => "raydium",
            Self::Orca => "orca",
            Self::Jupiter => "jupiter",
            Self::Curve => "curve",
            Self::Balancer => "balancer",
        }
    }
}

/// Normalized liquidity pool state
#[derive(Debug, Clone)]
pub struct LiquidityPool {
    pub venue: DexVenue,
    pub pool_id: String,
    pub token_a: String,
    pub token_b: String,
    pub fee_tier_bps: u32,
    pub reserve_a: f64,
    pub reserve_b: f64,
    pub price: f64, // token_a per token_b
    pub liquidity: f64,
    pub last_update_ns: u64,
    pub tick_spacing: i32, // For concentrated liquidity (Uniswap V3)
    pub current_tick: i32,
}

impl LiquidityPool {
    /// Calculate output amount for given input (constant product AMM)
    pub fn get_amount_out(&self, amount_in: f64, token_in_is_a: bool) -> f64 {
        let fee_multiplier = 1.0 - (self.fee_tier_bps as f64 / 10000.0);
        
        if token_in_is_a {
            // Input token A, output token B
            let numerator = amount_in * self.reserve_b * fee_multiplier;
            let denominator = self.reserve_a + amount_in * fee_multiplier;
            numerator / denominator
        } else {
            // Input token B, output token A
            let numerator = amount_in * self.reserve_a * fee_multiplier;
            let denominator = self.reserve_b + amount_in * fee_multiplier;
            numerator / denominator
        }
    }

    /// Calculate price impact for a trade
    pub fn calculate_price_impact(&self, amount_in: f64, token_in_is_a: bool) -> f64 {
        let output = self.get_amount_out(amount_in, token_in_is_a);
        let expected_output = amount_in * if token_in_is_a { self.reserve_b / self.reserve_a } else { self.reserve_a / self.reserve_b };
        
        if expected_output > 0.0 {
            1.0 - (output / expected_output)
        } else {
            0.0
        }
    }
}

/// Normalized quote from a DEX
#[derive(Debug, Clone)]
pub struct DexQuote {
    pub venue: DexVenue,
    pub pool_id: String,
    pub token_in: String,
    pub token_out: String,
    pub amount_in: f64,
    pub amount_out: f64,
    pub price: f64,
    pub price_impact_bps: f64,
    pub gas_estimate: u64,
    pub slippage_tolerance_bps: u32,
    pub timestamp_ns: u64,
    pub route: Vec<String>, // Pool IDs in route
}

/// Unified L2 order book built from DEX liquidity
#[derive(Debug, Clone)]
pub struct DexOrderBook {
    pub symbol: String,
    pub token_base: String,
    pub token_quote: String,
    pub bids: Vec<Level>,
    pub asks: Vec<Level>,
    pub best_bid: Option<f64>,
    pub best_ask: Option<f64>,
    pub spread_bps: f64,
    pub total_bid_liquidity: f64,
    pub total_ask_liquidity: f64,
    pub last_update_ns: u64,
    pub venue_breakdown: HashMap<DexVenue, VenueLiquidity>,
}

#[derive(Debug, Clone)]
pub struct VenueLiquidity {
    pub bid_liquidity: f64,
    pub ask_liquidity: f64,
    pub num_pools: usize,
}

impl DexOrderBook {
    pub fn new(symbol: String, token_base: String, token_quote: String) -> Self {
        Self {
            symbol,
            token_base,
            token_quote,
            bids: Vec::new(),
            asks: Vec::new(),
            best_bid: None,
            best_ask: None,
            spread_bps: 0.0,
            total_bid_liquidity: 0.0,
            total_ask_liquidity: 0.0,
            last_update_ns: 0,
            venue_breakdown: HashMap::new(),
        }
    }

    /// Build order book from liquidity pools
    pub fn build_from_pools(&mut self, pools: &[LiquidityPool], depth_levels: usize) {
        let mut bid_map: BTreeMap<f64, f64> = BTreeMap::new();
        let mut ask_map: BTreeMap<f64, f64> = BTreeMap::new();
        let mut venue_liquidity: HashMap<DexVenue, VenueLiquidity> = HashMap::new();

        for pool in pools {
            // Only include pools with sufficient liquidity
            if pool.liquidity < 1000.0 {
                continue;
            }

            // Generate price levels around current price
            let base_price = pool.price;
            
            // Create bid levels (below mid price)
            for i in 0..depth_levels {
                let price_offset = (i + 1) as f64 * 0.001; // 10 bps per level
                let price = base_price * (1.0 - price_offset);
                let volume = pool.liquidity * (1.0 - i as f64 / depth_levels as f64) * 0.1;
                
                *bid_map.entry(price).or_insert(0.0) += volume;
                
                // Track venue liquidity
                let entry = venue_liquidity.entry(pool.venue).or_insert(VenueLiquidity {
                    bid_liquidity: 0.0,
                    ask_liquidity: 0.0,
                    num_pools: 0,
                });
                entry.bid_liquidity += volume;
            }

            // Create ask levels (above mid price)
            for i in 0..depth_levels {
                let price_offset = (i + 1) as f64 * 0.001;
                let price = base_price * (1.0 + price_offset);
                let volume = pool.liquidity * (1.0 - i as f64 / depth_levels as f64) * 0.1;
                
                *ask_map.entry(price).or_insert(0.0) += volume;
                
                let entry = venue_liquidity.entry(pool.venue).or_insert(VenueLiquidity {
                    bid_liquidity: 0.0,
                    ask_liquidity: 0.0,
                    num_pools: 0,
                });
                entry.ask_liquidity += volume;
            }

            // Count pools per venue
            if let Some(entry) = venue_liquidity.get_mut(&pool.venue) {
                entry.num_pools += 1;
            }
        }

        // Convert to sorted vectors
        self.bids = bid_map.iter().rev().take(depth_levels)
            .map(|(&price, &volume)| Level { price, volume })
            .collect();

        self.asks = ask_map.iter().take(depth_levels)
            .map(|(&price, &volume)| Level { price, volume })
            .collect();

        // Update best prices
        self.best_bid = self.bids.first().map(|l| l.price);
        self.best_ask = self.asks.first().map(|l| l.price);

        // Calculate spread
        if let (Some(bid), Some(ask)) = (self.best_bid, self.best_ask) {
            self.spread_bps = ((ask - bid) / ((bid + ask) / 2.0)) * 10000.0;
        }

        // Calculate total liquidity
        self.total_bid_liquidity = self.bids.iter().map(|l| l.volume).sum();
        self.total_ask_liquidity = self.asks.iter().map(|l| l.volume).sum();

        self.last_update_ns = chrono::Utc::now().timestamp_nanos() as u64;
        self.venue_breakdown = venue_liquidity;
    }
}

/// Aggregator state and subscription manager
pub struct DexAggregator {
    pools: HashMap<String, LiquidityPool>, // pool_id -> pool
    books: HashMap<String, DexOrderBook>,  // symbol -> book
    subscriptions: Vec<DexSubscription>,
    config: AggregatorConfig,
    last_update_ns: u64,
}

#[derive(Debug, Clone)]
pub struct AggregatorConfig {
    pub refresh_interval_ms: u64,
    pub max_pools_per_symbol: usize,
    pub min_liquidity_usd: f64,
    pub supported_venues: Vec<DexVenue>,
}

impl Default for AggregatorConfig {
    fn default() -> Self {
        Self {
            refresh_interval_ms: 100,
            max_pools_per_symbol: 50,
            min_liquidity_usd: 10000.0,
            supported_venues: vec![
                DexVenue::UniswapV3,
                DexVenue::Raydium,
                DexVenue::Jupiter,
                DexVenue::Orca,
            ],
        }
    }
}

#[derive(Debug, Clone)]
pub struct DexSubscription {
    pub venue: DexVenue,
    pub topic: String,
    pub active: bool,
}

impl DexAggregator {
    /// Create a new DEX aggregator
    pub fn new(config: AggregatorConfig) -> Self {
        Self {
            pools: HashMap::new(),
            books: HashMap::new(),
            subscriptions: Vec::new(),
            config,
            last_update_ns: 0,
        }
    }

    /// Subscribe to a DEX venue's liquidity updates
    pub fn subscribe(&mut self, venue: DexVenue, topic: String) {
        if !self.config.supported_venues.contains(&venue) {
            warn!("Venue {:?} not supported, skipping subscription", venue);
            return;
        }

        self.subscriptions.push(DexSubscription {
            venue,
            topic,
            active: true,
        });

        info!("Subscribed to {}:{} ", venue.as_str(), topic);
    }

    /// Update pool state from WebSocket message
    pub fn update_pool(&mut self, pool: LiquidityPool) {
        if pool.liquidity < self.config.min_liquidity_usd {
            return; // Skip low liquidity pools
        }

        self.pools.insert(pool.pool_id.clone(), pool);
        self.last_update_ns = chrono::Utc::now().timestamp_nanos() as u64;
    }

    /// Get or create order book for a symbol pair
    pub fn get_or_create_book(&mut self, symbol: &str, token_base: &str, token_quote: &str) -> &mut DexOrderBook {
        use std::collections::hash_map::Entry;
        
        match self.books.entry(symbol.to_string()) {
            Entry::Vacant(e) => e.insert(DexOrderBook::new(
                symbol.to_string(),
                token_base.to_string(),
                token_quote.to_string(),
            )),
            Entry::Occupied(e) => e.into_mut(),
        }
    }

    /// Rebuild order books from current pool states
    pub fn rebuild_books(&mut self) {
        // Group pools by trading pair
        let mut pools_by_pair: HashMap<(String, String), Vec<LiquidityPool>> = HashMap::new();

        for pool in self.pools.values() {
            let key = (pool.token_a.clone(), pool.token_b.clone());
            pools_by_pair.entry(key).or_default().push(pool.clone());
        }

        // Build/update books for each pair
        for ((token_a, token_b), mut pools) in pools_by_pair {
            // Sort by liquidity and take top N
            pools.sort_by(|a, b| b.liquidity.partial_cmp(&a.liquidity).unwrap_or(std::cmp::Ordering::Equal));
            pools.truncate(self.config.max_pools_per_symbol);

            let symbol = format!("{}/{}", token_a, token_b);
            let book = self.get_or_create_book(&symbol, &token_a, &token_b);
            book.build_from_pools(&pools, 10); // 10 depth levels
        }
    }

    /// Get best quote for a swap
    pub fn get_best_quote(&self, token_in: &str, token_out: &str, amount_in: f64) -> Option<DexQuote> {
        let mut best_quote: Option<DexQuote> = None;

        for pool in self.pools.values() {
            // Check if pool matches the pair
            let matches = (pool.token_a == token_in && pool.token_b == token_out)
                || (pool.token_a == token_out && pool.token_b == token_in);

            if !matches {
                continue;
            }

            let token_in_is_a = pool.token_a == token_in;
            let amount_out = pool.get_amount_out(amount_in, token_in_is_a);
            let price = amount_out / amount_in;
            let impact = pool.calculate_price_impact(amount_in, token_in_is_a);

            let quote = DexQuote {
                venue: pool.venue,
                pool_id: pool.pool_id.clone(),
                token_in: token_in.to_string(),
                token_out: token_out.to_string(),
                amount_in,
                amount_out,
                price,
                price_impact_bps: impact * 10000.0,
                gas_estimate: 150000, // Base estimate
                slippage_tolerance_bps: 50,
                timestamp_ns: self.last_update_ns,
                route: vec![pool.pool_id.clone()],
            };

            // Keep best quote (highest output)
            if best_quote.as_ref().map_or(true, |q| amount_out > q.amount_out) {
                best_quote = Some(quote);
            }
        }

        best_quote
    }

    /// Get all available quotes for comparison
    pub fn get_all_quotes(&self, token_in: &str, token_out: &str, amount_in: f64) -> Vec<DexQuote> {
        let mut quotes = Vec::new();

        for pool in self.pools.values() {
            let matches = (pool.token_a == token_in && pool.token_b == token_out)
                || (pool.token_a == token_out && pool.token_b == token_in);

            if !matches {
                continue;
            }

            let token_in_is_a = pool.token_a == token_in;
            let amount_out = pool.get_amount_out(amount_in, token_in_is_a);
            let price = amount_out / amount_in;
            let impact = pool.calculate_price_impact(amount_in, token_in_is_a);

            quotes.push(DexQuote {
                venue: pool.venue,
                pool_id: pool.pool_id.clone(),
                token_in: token_in.to_string(),
                token_out: token_out.to_string(),
                amount_in,
                amount_out,
                price,
                price_impact_bps: impact * 10000.0,
                gas_estimate: 150000,
                slippage_tolerance_bps: 50,
                timestamp_ns: self.last_update_ns,
                route: vec![pool.pool_id.clone()],
            });
        }

        // Sort by amount_out descending
        quotes.sort_by(|a, b| b.amount_out.partial_cmp(&a.amount_out).unwrap_or(std::cmp::Ordering::Equal));
        quotes
    }

    /// Stream order book updates
    pub async fn stream_book_updates(
        &self,
        symbol: String,
    ) -> anyhow::Result<watch::Receiver<DexOrderBook>> {
        let (tx, rx) = watch::channel(
            self.books.get(&symbol).cloned().unwrap_or_else(|| {
                DexOrderBook::new(symbol.clone(), String::new(), String::new())
            })
        );

        // In production, this would spawn a task that periodically updates the channel
        Ok(rx)
    }
}

/// WebSocket connection manager for DEX subscriptions
pub struct WebSocketManager {
    connections: HashMap<DexVenue, WebSocketConnection>,
}

#[derive(Debug)]
pub struct WebSocketConnection {
    pub venue: DexVenue,
    pub url: String,
    pub connected: bool,
    pub last_message_ns: u64,
}

impl WebSocketManager {
    pub fn new() -> Self {
        Self {
            connections: HashMap::new(),
        }
    }

    pub async fn connect(&mut self, venue: DexVenue, url: String) -> anyhow::Result<()> {
        // In production, this would establish actual WebSocket connection
        let conn = WebSocketConnection {
            venue,
            url,
            connected: true,
            last_message_ns: chrono::Utc::now().timestamp_nanos() as u64,
        };

        self.connections.insert(venue, conn);
        info!("Connected to {} WebSocket", venue.as_str());
        Ok(())
    }

    pub fn is_connected(&self, venue: DexVenue) -> bool {
        self.connections.get(&venue).map_or(false, |c| c.connected)
    }
}
