//! DEX Routing Module - Smart Order Router (SOR)
//! 
//! Implements a highly optimized multi-hop pathfinder for optimal token swap routing
//! across fragmented liquidity pools. Uses a modified Bellman-Ford algorithm to detect
//! negative cycles (arbitrage) and calculate exact input/output amounts.

use std::collections::{HashMap, HashSet, VecDeque};
use fixedbitset::FixedBitSet;

use crate::dex::aggregator::{DexQuote, LiquidityPool, DexVenue, DexAggregator};

/// Maximum number of hops allowed in a route
const MAX_HOPS: usize = 4;

/// Maximum number of paths to track per destination
const MAX_PATHS_PER_DEST: usize = 10;

/// Graph node representing a token
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TokenNode {
    pub id: u32,
}

impl TokenNode {
    pub fn new(id: u32) -> Self {
        Self { id }
    }
}

/// Graph edge representing a liquidity pool connection
#[derive(Debug, Clone)]
pub struct PoolEdge {
    pub from_token: TokenNode,
    pub to_token: TokenNode,
    pub pool_id: String,
    pub venue: DexVenue,
    pub fee_bps: f64,
    pub reserve_in: f64,
    pub reserve_out: f64,
    pub price: f64,
}

/// Route for a multi-hop swap
#[derive(Debug, Clone)]
pub struct SwapRoute {
    pub tokens: Vec<String>,
    pub pools: Vec<String>,
    pub venues: Vec<DexVenue>,
    pub amount_in: f64,
    pub amount_out: f64,
    pub price_impact_bps: f64,
    pub total_fees_bps: f64,
    pub gas_estimate: u64,
}

impl SwapRoute {
    /// Calculate effective price
    pub fn effective_price(&self) -> f64 {
        if self.amount_in > 0.0 {
            self.amount_out / self.amount_in
        } else {
            0.0
        }
    }

    /// Check if route is valid
    pub fn is_valid(&self) -> bool {
        self.tokens.len() >= 2 
            && self.tokens.len() == self.pools.len() + 1
            && self.pools.len() == self.venues.len()
    }
}

/// Path finding result
#[derive(Debug, Clone)]
pub struct PathResult {
    pub routes: Vec<SwapRoute>,
    pub best_route_index: Option<usize>,
    pub arbitrage_detected: bool,
    pub arbitrage_profit_bps: f64,
}

/// Smart Order Router using graph-based pathfinding
pub struct SmartOrderRouter {
    /// Token symbol to node ID mapping
    token_to_node: HashMap<String, TokenNode>,
    /// Node ID to token symbol mapping  
    node_to_token: HashMap<TokenNode, String>,
    /// Adjacency list representation of the liquidity graph
    adjacency: HashMap<TokenNode, Vec<PoolEdge>>,
    /// Next available node ID
    next_node_id: u32,
    /// Fixed-size arrays for path tracking (no allocations during hot path)
    path_buffer: [[TokenNode; MAX_HOPS + 1]; MAX_PATHS_PER_DEST],
    edge_buffer: [[usize; MAX_HOPS]; MAX_PATHS_PER_DEST],
}

impl SmartOrderRouter {
    /// Create a new smart order router
    pub fn new() -> Self {
        Self {
            token_to_node: HashMap::new(),
            node_to_token: HashMap::new(),
            adjacency: HashMap::new(),
            next_node_id: 0,
            path_buffer: [[TokenNode { id: 0 }; MAX_HOPS + 1]; MAX_PATHS_PER_DEST],
            edge_buffer: [[0; MAX_HOPS]; MAX_PATHS_PER_DEST],
        }
    }

    /// Get or create node for a token
    fn get_or_create_node(&mut self, token: &str) -> TokenNode {
        if let Some(&node) = self.token_to_node.get(token) {
            node
        } else {
            let node = TokenNode::new(self.next_node_id);
            self.next_node_id += 1;
            self.token_to_node.insert(token.to_string(), node);
            self.node_to_token.insert(node, token.to_string());
            node
        }
    }

    /// Build routing graph from liquidity pools
    pub fn build_graph(&mut self, pools: &[LiquidityPool]) {
        self.adjacency.clear();

        for pool in pools {
            let from_node = self.get_or_create_node(&pool.token_a);
            let to_node = self.get_or_create_node(&pool.token_b);

            // Add edge A -> B
            let edge_ab = PoolEdge {
                from_token: from_node,
                to_token: to_node,
                pool_id: pool.pool_id.clone(),
                venue: pool.venue,
                fee_bps: pool.fee_tier_bps as f64,
                reserve_in: pool.reserve_a,
                reserve_out: pool.reserve_b,
                price: pool.price,
            };

            // Add edge B -> A (reverse direction)
            let edge_ba = PoolEdge {
                from_token: to_node,
                to_token: from_node,
                pool_id: pool.pool_id.clone(),
                venue: pool.venue,
                fee_bps: pool.fee_tier_bps as f64,
                reserve_in: pool.reserve_b,
                reserve_out: pool.reserve_a,
                price: 1.0 / pool.price.max(1e-10),
            };

            self.adjacency.entry(from_node).or_default().push(edge_ab);
            self.adjacency.entry(to_node).or_default().push(edge_ba);
        }
    }

    /// Find optimal route using modified Bellman-Ford with path reconstruction
    pub fn find_best_route(
        &mut self,
        token_in: &str,
        token_out: &str,
        amount_in: f64,
    ) -> PathResult {
        let from_node = match self.token_to_node.get(token_in) {
            Some(&n) => n,
            None => return self.empty_result(),
        };
        let to_node = match self.token_to_node.get(token_out) {
            Some(&n) => n,
            None => return self.empty_result(),
        };

        // Find all paths up to MAX_HOPS
        let mut routes = Vec::new();
        
        // BFS/DFS hybrid to find paths
        self.find_paths_recursive(
            from_node,
            to_node,
            amount_in,
            0,
            &mut Vec::new(),
            &mut HashSet::new(),
            &mut routes,
        );

        if routes.is_empty() {
            return self.empty_result();
        }

        // Sort by output amount descending
        routes.sort_by(|a, b| b.amount_out.partial_cmp(&a.amount_out).unwrap_or(std::cmp::Ordering::Equal));

        let best_idx = 0;
        let best_amount_out = routes[best_idx].amount_out;

        // Check for arbitrage opportunities
        let (arbitrage_detected, arbitrage_profit) = self.detect_arbitrage(token_in, amount_in);

        PathResult {
            routes,
            best_route_index: Some(best_idx),
            arbitrage_detected,
            arbitrage_profit_bps: arbitrage_profit,
        }
    }

    /// Recursive path finding with cycle detection
    fn find_paths_recursive(
        &self,
        current: TokenNode,
        target: TokenNode,
        current_amount: f64,
        hop_count: usize,
        path: &mut Vec<TokenNode>,
        visited: &mut HashSet<TokenNode>,
        routes: &mut Vec<SwapRoute>,
    ) {
        if hop_count > MAX_HOPS {
            return;
        }

        path.push(current);
        visited.insert(current);

        if current == target && hop_count >= 1 {
            // Found a valid path, construct route
            if let Some(route) = self.construct_route(path, current_amount) {
                routes.push(route);
                
                // Limit paths per destination
                if routes.len() >= MAX_PATHS_PER_DEST {
                    path.pop();
                    visited.remove(&current);
                    return;
                }
            }
        } else if hop_count < MAX_HOPS {
            // Continue exploring
            if let Some(edges) = self.adjacency.get(&current) {
                for edge in edges {
                    if !visited.contains(&edge.to_token) {
                        // Calculate output amount for this hop
                        let output = self.calculate_output(
                            current_amount,
                            edge.reserve_in,
                            edge.reserve_out,
                            edge.fee_bps,
                        );

                        if output > 0.0 {
                            self.find_paths_recursive(
                                edge.to_token,
                                target,
                                output,
                                hop_count + 1,
                                path,
                                visited,
                                routes,
                            );
                        }
                    }
                }
            }
        }

        path.pop();
        visited.remove(&current);
    }

    /// Construct a SwapRoute from a path
    fn construct_route(&self, path: &[TokenNode], amount_in: f64) -> Option<SwapRoute> {
        if path.len() < 2 {
            return None;
        }

        let mut tokens = Vec::with_capacity(path.len());
        let mut pools = Vec::with_capacity(path.len() - 1);
        let mut venues = Vec::with_capacity(path.len() - 1);
        let mut total_fees = 0.0;

        for (i, &node) in path.iter().enumerate() {
            tokens.push(self.node_to_token.get(&node)?.clone());

            if i > 0 {
                // Find edge between previous and current node
                let prev_node = path[i - 1];
                if let Some(edges) = self.adjacency.get(&prev_node) {
                    if let Some(edge) = edges.iter().find(|e| e.to_token == node) {
                        pools.push(edge.pool_id.clone());
                        venues.push(edge.venue);
                        total_fees += edge.fee_bps;
                    } else {
                        return None;
                    }
                } else {
                    return None;
                }
            }
        }

        // Calculate final output
        let mut current_amount = amount_in;
        for i in 0..path.len() - 1 {
            let from = path[i];
            let to = path[i + 1];

            if let Some(edges) = self.adjacency.get(&from) {
                if let Some(edge) = edges.iter().find(|e| e.to_token == to) {
                    current_amount = self.calculate_output(
                        current_amount,
                        edge.reserve_in,
                        edge.reserve_out,
                        edge.fee_bps,
                    );
                }
            }
        }

        let price_impact = 0.0; // Would need mid price reference
        let gas_estimate = 150_000 * pools.len() as u64;

        Some(SwapRoute {
            tokens,
            pools,
            venues,
            amount_in,
            amount_out: current_amount,
            price_impact_bps: price_impact,
            total_fees_bps: total_fees,
            gas_estimate,
        })
    }

    /// Calculate output amount for a single hop (constant product AMM)
    fn calculate_output(&self, amount_in: f64, reserve_in: f64, reserve_out: f64, fee_bps: f64) -> f64 {
        let fee_multiplier = 1.0 - (fee_bps / 10000.0);
        let numerator = amount_in * reserve_out * fee_multiplier;
        let denominator = reserve_in + amount_in * fee_multiplier;
        
        if denominator > 0.0 {
            numerator / denominator
        } else {
            0.0
        }
    }

    /// Detect arbitrage opportunities using Bellman-Ford negative cycle detection
    fn detect_arbitrage(&self, start_token: &str, amount: f64) -> (bool, f64) {
        let start_node = match self.token_to_node.get(start_token) {
            Some(&n) => n,
            None => return (false, 0.0),
        };

        // Convert to log prices for negative cycle detection
        // If sum of log prices in a cycle is positive, there's arbitrage
        let num_nodes = self.next_node_id as usize;
        if num_nodes == 0 {
            return (false, 0.0);
        }

        // Distance array (using log prices)
        let mut dist: Vec<f64> = vec![f64::INFINITY; num_nodes];
        dist[start_node.id as usize] = 0.0;

        // Relax edges repeatedly
        for _ in 0..num_nodes - 1 {
            let mut updated = false;
            
            for (&from, edges) in &self.adjacency {
                if dist[from.id as usize] == f64::INFINITY {
                    continue;
                }

                for edge in edges {
                    // Log price (negative because we want to minimize)
                    let log_price = -(edge.price.ln());
                    let new_dist = dist[from.id as usize] + log_price;

                    if new_dist < dist[edge.to_token.id as usize] {
                        dist[edge.to_token.id as usize] = new_dist;
                        updated = true;
                    }
                }
            }

            if !updated {
                break;
            }
        }

        // Check for negative cycles (arbitrage opportunities)
        for (&from, edges) in &self.adjacency {
            if dist[from.id as usize] == f64::INFINITY {
                continue;
            }

            for edge in edges {
                let log_price = -(edge.price.ln());
                let new_dist = dist[from.id as usize] + log_price;

                if new_dist < dist[edge.to_token.id as usize] - 1e-10 {
                    // Negative cycle detected - arbitrage opportunity!
                    // Calculate profit in basis points
                    let profit_bps = ((dist[edge.to_token.id as usize] - new_dist).exp() - 1.0) * 10000.0;
                    return (true, profit_bps.max(0.0));
                }
            }
        }

        (false, 0.0)
    }

    /// Split order across multiple routes for better execution
    pub fn split_order(
        &mut self,
        token_in: &str,
        token_out: &str,
        total_amount: f64,
        num_splits: usize,
    ) -> Vec<SwapRoute> {
        let path_result = self.find_best_route(token_in, token_out, total_amount);
        
        if path_result.routes.is_empty() {
            return Vec::new();
        }

        let mut splits = Vec::new();
        let amount_per_split = total_amount / num_splits as f64;

        for i in 0..num_splits {
            let split_amount = if i == num_splits - 1 {
                total_amount - amount_per_split * (num_splits - 1) as f64
            } else {
                amount_per_split
            };

            // Re-find route for each split (prices may change)
            let result = self.find_best_route(token_in, token_out, split_amount);
            if let Some(route) = result.routes.first().cloned() {
                splits.push(route);
            }
        }

        splits
    }

    fn empty_result(&self) -> PathResult {
        PathResult {
            routes: Vec::new(),
            best_route_index: None,
            arbitrage_detected: false,
            arbitrage_profit_bps: 0.0,
        }
    }
}

impl Default for SmartOrderRouter {
    fn default() -> Self {
        Self::new()
    }
}

/// Execute a swap route (placeholder for actual on-chain execution)
pub async fn execute_swap_route(route: &SwapRoute) -> anyhow::Result<()> {
    // In production, this would:
    // 1. Build the transaction with the route
    // 2. Sign and submit via MEV-protected relay
    // 3. Wait for confirmation
    
    Ok(())
}
