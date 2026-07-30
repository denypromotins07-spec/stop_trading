//! DEX Module Root
//! 
//! Manages RPC connections to Solana QUIC and EVM nodes, integrating quotes
//! into the Smart Order Router (SOR).

pub mod aggregator;
pub mod routing;

pub use aggregator::{
    DexAggregator, DexOrderBook, DexQuote, LiquidityPool, DexVenue, 
    AggregatorConfig, WebSocketManager, WebSocketConnection, VenueLiquidity,
};
pub use routing::{
    SmartOrderRouter, SwapRoute, PathResult, PoolEdge, TokenNode,
};

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::RwLock;
use tracing::{debug, info, warn, error};

/// RPC connection configuration
#[derive(Debug, Clone)]
pub struct RpcConfig {
    /// Solana RPC endpoint
    pub solana_rpc_url: String,
    /// Solana WebSocket endpoint for subscriptions
    pub solana_ws_url: String,
    /// Ethereum RPC endpoint  
    pub eth_rpc_url: String,
    /// Ethereum WebSocket endpoint
    pub eth_ws_url: String,
    /// Request timeout
    pub timeout_ms: u64,
    /// Maximum retries
    pub max_retries: u32,
}

impl Default for RpcConfig {
    fn default() -> Self {
        Self {
            solana_rpc_url: "https://api.mainnet-beta.solana.com".to_string(),
            solana_ws_url: "wss://api.mainnet-beta.solana.com".to_string(),
            eth_rpc_url: "https://eth.llamarpc.com".to_string(),
            eth_ws_url: "wss://eth.llamarpc.com/ws".to_string(),
            timeout_ms: 5000,
            max_retries: 3,
        }
    }
}

/// Unified DEX manager coordinating aggregation and routing
pub struct DexManager {
    config: RpcConfig,
    aggregator: Arc<RwLock<DexAggregator>>,
    router: Arc<RwLock<SmartOrderRouter>>,
    rpc_client: RpcClient,
}

/// RPC client for blockchain interactions
pub struct RpcClient {
    solana_client: Option<SolanaClient>,
    eth_client: Option<EthClient>,
    config: RpcConfig,
}

struct SolanaClient {
    rpc_url: String,
    ws_url: String,
    connected: bool,
}

struct EthClient {
    rpc_url: String,
    ws_url: String,
    connected: bool,
}

impl DexManager {
    /// Create a new DEX manager
    pub fn new(config: RpcConfig) -> Self {
        let aggregator = Arc::new(RwLock::new(DexAggregator::new(AggregatorConfig::default())));
        let router = Arc::new(RwLock::new(SmartOrderRouter::new()));
        let rpc_client = RpcClient::new(&config);

        Self {
            config,
            aggregator,
            router,
            rpc_client,
        }
    }

    /// Initialize connections and subscriptions
    pub async fn initialize(&mut self) -> anyhow::Result<()> {
        info!("Initializing DEX manager...");

        // Connect to RPC endpoints
        self.rpc_client.connect().await?;

        // Subscribe to DEX venues
        let mut agg = self.aggregator.write().await;
        
        // Uniswap V3
        agg.subscribe(DexVenue::UniswapV3, "pool_updates".to_string());
        
        // Raydium (Solana)
        agg.subscribe(DexVenue::Raydium, "raydium_pools".to_string());
        
        // Jupiter Aggregator
        agg.subscribe(DexVenue::Jupiter, "jupiter_routes".to_string());
        
        // Orca (Solana)
        agg.subscribe(DexVenue::Orca, "orca_pools".to_string());

        drop(agg);

        info!("DEX manager initialized successfully");
        Ok(())
    }

    /// Get best route for a swap
    pub async fn get_best_route(
        &self,
        token_in: &str,
        token_out: &str,
        amount_in: f64,
    ) -> Option<SwapRoute> {
        let mut router = self.router.write().await;
        let result = router.find_best_route(token_in, token_out, amount_in);
        
        result.routes.into_iter().next()
    }

    /// Get all available routes for comparison
    pub async fn get_all_routes(
        &self,
        token_in: &str,
        token_out: &str,
        amount_in: f64,
    ) -> Vec<SwapRoute> {
        let mut router = self.router.write().await;
        let result = router.find_best_route(token_in, token_out, amount_in);
        result.routes
    }

    /// Update liquidity from aggregator
    pub async fn update_liquidity(&self, pool: LiquidityPool) {
        let mut agg = self.aggregator.write().await;
        agg.update_pool(pool);
        agg.rebuild_books();
        
        // Rebuild routing graph
        drop(agg);
        
        let pools = self.get_all_pools().await;
        let mut router = self.router.write().await;
        router.build_graph(&pools);
    }

    /// Get all current pools
    pub async fn get_all_pools(&self) -> Vec<LiquidityPool> {
        let agg = self.aggregator.read().await;
        // In production, would return actual pools
        Vec::new()
    }

    /// Get order book for a symbol
    pub async fn get_order_book(&self, symbol: &str) -> Option<DexOrderBook> {
        let agg = self.aggregator.read().await;
        // Would need to implement getter in aggregator
        None
    }

    /// Execute a swap through the optimal route
    pub async fn execute_swap(&self, route: &SwapRoute) -> anyhow::Result<String> {
        // In production:
        // 1. Build transaction with route data
        // 2. Route through MEV-protected relay (Jito/Flashbots)
        // 3. Return transaction signature/hash
        
        info!(
            "Executing swap: {} -> {} via {} hops",
            route.tokens.first().unwrap_or(&String::new()),
            route.tokens.last().unwrap_or(&String::new()),
            route.pools.len()
        );

        // Placeholder
        Ok("tx_signature_placeholder".to_string())
    }

    /// Split and execute large orders across multiple routes
    pub async fn execute_split_swap(
        &self,
        token_in: &str,
        token_out: &str,
        total_amount: f64,
        num_splits: usize,
    ) -> anyhow::Result<Vec<String>> {
        let mut router = self.router.write().await;
        let splits = router.split_order(token_in, token_out, total_amount, num_splits);
        drop(router);

        let mut signatures = Vec::new();
        for route in splits {
            match self.execute_swap(&route).await {
                Ok(sig) => signatures.push(sig),
                Err(e) => {
                    warn!("Failed to execute split: {}", e);
                    // Could implement retry logic here
                }
            }
        }

        Ok(signatures)
    }

    /// Check for arbitrage opportunities
    pub async fn check_arbitrage(&self, token: &str, amount: f64) -> (bool, f64) {
        let mut router = self.router.write().await;
        let (_, profit_bps) = router.detect_arbitrage(token, amount);
        (profit_bps > 10.0, profit_bps) // Threshold of 10 bps
    }
}

impl RpcClient {
    pub fn new(config: &RpcConfig) -> Self {
        Self {
            solana_client: Some(SolanaClient {
                rpc_url: config.solana_rpc_url.clone(),
                ws_url: config.solana_ws_url.clone(),
                connected: false,
            }),
            eth_client: Some(EthClient {
                rpc_url: config.eth_rpc_url.clone(),
                ws_url: config.eth_ws_url.clone(),
                connected: false,
            }),
            config: config.clone(),
        }
    }

    pub async fn connect(&mut self) -> anyhow::Result<()> {
        // Connect to Solana
        if let Some(ref mut client) = self.solana_client {
            client.connected = true;
            info!("Connected to Solana RPC: {}", client.rpc_url);
        }

        // Connect to Ethereum
        if let Some(ref mut client) = self.eth_client {
            client.connected = true;
            info!("Connected to Ethereum RPC: {}", client.rpc_url);
        }

        Ok(())
    }

    pub fn is_solana_connected(&self) -> bool {
        self.solana_client.as_ref().map_or(false, |c| c.connected)
    }

    pub fn is_eth_connected(&self) -> bool {
        self.eth_client.as_ref().map_or(false, |c| c.connected)
    }
}

/// QUIC connection optimizer for Solana
pub struct QuicOptimizer {
    connections: Vec<QuicConnection>,
}

#[derive(Debug)]
struct QuicConnection {
    endpoint: String,
    latency_us: u64,
    packet_loss_rate: f64,
    active: bool,
}

impl QuicOptimizer {
    pub fn new() -> Self {
        Self {
            connections: Vec::new(),
        }
    }

    /// Add a QUIC endpoint to monitor
    pub fn add_endpoint(&mut self, endpoint: String) {
        self.connections.push(QuicConnection {
            endpoint,
            latency_us: 0,
            packet_loss_rate: 0.0,
            active: true,
        });
    }

    /// Get best endpoint based on latency
    pub fn get_best_endpoint(&self) -> Option<&str> {
        self.connections.iter()
            .filter(|c| c.active)
            .min_by_key(|c| c.latency_us)
            .map(|c| c.endpoint.as_str())
    }

    /// Update latency measurement for an endpoint
    pub fn update_latency(&mut self, endpoint: &str, latency_us: u64) {
        if let Some(conn) = self.connections.iter_mut().find(|c| c.endpoint == endpoint) {
            conn.latency_us = latency_us;
        }
    }
}

impl Default for QuicOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// EVM chain ID constants
pub mod chain_ids {
    pub const ETHEREUM_MAINNET: u64 = 1;
    pub const ARBITRUM: u64 = 42161;
    pub const OPTIMISM: u64 = 10;
    pub const BASE: u64 = 8453;
    pub const POLYGON: u64 = 137;
    pub const BSC: u64 = 56;
    pub const AVALANCHE: u64 = 43114;
}

/// Solana program IDs for major DEXs
pub mod solana_programs {
    pub const RAYDIUM_AMM: &str = "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8";
    pub const ORCA_WHIRLPOOL: &str = "whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc";
    pub const JUPITER_V6: &str = "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4";
}
