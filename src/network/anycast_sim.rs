//! Anycast Simulation and Routing Logic
//!
//! Selects geographically closest exchange edge nodes by continuously
//! measuring RTT to exchange REST endpoints.

use std::time::{Duration, Instant};

/// Exchange edge node information
#[derive(Debug, Clone)]
pub struct EdgeNode {
    /// Node identifier
    pub node_id: u32,
    /// Exchange name
    pub exchange: String,
    /// Geographic region
    pub region: String,
    /// IP address or hostname
    pub address: String,
    /// REST endpoint path
    pub rest_endpoint: String,
    /// WebSocket endpoint
    pub ws_endpoint: String,
    /// Last measured RTT in microseconds
    pub last_rtt_us: u64,
    /// RTT history for smoothing
    rtt_history: [u64; 8],
    /// RTT history index
    rtt_idx: usize,
    /// Whether node is reachable
    pub is_reachable: bool,
    /// Consecutive probe failures
    pub failures: u32,
    /// Last successful probe time
    pub last_probe: Option<Instant>,
}

impl EdgeNode {
    pub fn new(node_id: u32, exchange: &str, region: &str, address: &str) -> Self {
        EdgeNode {
            node_id,
            exchange: exchange.to_string(),
            region: region.to_string(),
            address: address.to_string(),
            rest_endpoint: format!("https://{}/api/v1/time", address),
            ws_endpoint: format!("wss://{}/ws", address),
            last_rtt_us: u64::MAX,
            rtt_history: [u64::MAX; 8],
            rtt_idx: 0,
            is_reachable: false,
            failures: 0,
            last_probe: None,
        }
    }

    /// Update RTT measurement with exponential smoothing
    #[inline]
    pub fn update_rtt(&mut self, rtt_us: u64) {
        // Store in history
        self.rtt_history[self.rtt_idx] = rtt_us;
        self.rtt_idx = (self.rtt_idx + 1) % 8;

        // Calculate smoothed average
        let mut sum = 0u64;
        let mut count = 0;
        for &rtt in &self.rtt_history {
            if rtt != u64::MAX {
                sum += rtt;
                count += 1;
            }
        }

        if count > 0 {
            self.last_rtt_us = sum / count as u64;
            self.is_reachable = true;
            self.failures = 0;
            self.last_probe = Some(Instant::now());
        }
    }

    /// Record probe failure
    #[inline]
    pub fn record_failure(&mut self) {
        self.failures += 1;
        if self.failures >= 3 {
            self.is_reachable = false;
        }
    }

    /// Get smoothed RTT
    #[inline]
    pub fn smoothed_rtt(&self) -> u64 {
        self.last_rtt_us
    }

    /// Check if node needs probing
    #[inline]
    pub fn needs_probe(&self) -> bool {
        match self.last_probe {
            Some(last) => last.elapsed() > Duration::from_millis(500),
            None => true,
        }
    }

    /// Calculate node score (higher is better)
    #[inline]
    pub fn score(&self) -> f64 {
        if !self.is_reachable {
            return 0.0;
        }

        // Score inversely proportional to RTT
        // Nodes with RTT < 1ms get score > 0.9
        1.0 / (1.0 + self.last_rtt_us as f64 / 1000.0)
    }
}

/// Anycast routing table
pub struct AnycastRouter {
    /// Available edge nodes
    nodes: Vec<EdgeNode>,
    /// Currently selected best node per exchange
    best_nodes: Vec<usize>,
    /// Probe interval in milliseconds
    probe_interval_ms: u64,
    /// Pre-allocated RTT results buffer
    rtt_results: [Option<u64>; 16],
}

impl AnycastRouter {
    pub fn new() -> Self {
        // Initialize with major crypto exchange endpoints
        let mut nodes = Vec::new();

        // Binance endpoints
        nodes.push(EdgeNode::new(0, "binance", "global", "api.binance.com"));
        nodes.push(EdgeNode::new(1, "binance", "us", "api.binance.us"));

        // Coinbase endpoints
        nodes.push(EdgeNode::new(2, "coinbase", "us-east", "api.coinbase.com"));
        nodes.push(EdgeNode::new(3, "coinbase", "eu", "api.eu.coinbase.com"));

        // FTX-style endpoints (for reference)
        nodes.push(EdgeNode::new(4, "ftx", "us-west", "ftx.com"));

        // Bybit endpoints
        nodes.push(EdgeNode::new(5, "bybit", "asia", "api.bybit.com"));
        nodes.push(EdgeNode::new(6, "bybit", "global", "api.bytick.com"));

        // OKX endpoints
        nodes.push(EdgeNode::new(7, "okx", "global", "www.okx.com"));

        // Kraken endpoints
        nodes.push(EdgeNode::new(8, "kraken", "us", "api.kraken.com"));
        nodes.push(EdgeNode::new(9, "kraken", "eu", "api.kraken.com"));

        let num_nodes = nodes.len();
        
        AnycastRouter {
            nodes,
            best_nodes: (0..num_nodes).collect(),
            probe_interval_ms: 500,
            rtt_results: [None; 16],
        }
    }

    /// Simulate RTT probe to a node (in real implementation, this would be actual HTTP ping)
    /// Returns simulated RTT in microseconds
    pub fn probe_node(&mut self, node_idx: usize) -> Option<u64> {
        if node_idx >= self.nodes.len() {
            return None;
        }

        // In production, this would make actual HTTP request
        // For simulation, we generate realistic RTT based on region
        let node = &self.nodes[node_idx];
        
        let simulated_rtt = self.simulate_rtt(&node.region);
        
        self.nodes[node_idx].update_rtt(simulated_rtt);
        Some(simulated_rtt)
    }

    /// Simulate RTT based on geographic region
    fn simulate_rtt(&self, region: &str) -> u64 {
        // Realistic RTT ranges by region (in microseconds)
        match region {
            "us-east" => 1_000 + (rand_u32() % 500) as u64,      // 1-1.5ms
            "us-west" => 5_000 + (rand_u32() % 2000) as u64,     // 5-7ms
            "us" => 3_000 + (rand_u32() % 3000) as u64,          // 3-6ms
            "eu" => 20_000 + (rand_u32() % 10000) as u64,        // 20-30ms
            "asia" => 50_000 + (rand_u32() % 20000) as u64,      // 50-70ms
            "global" => 30_000 + (rand_u32() % 20000) as u64,    // 30-50ms
            _ => 50_000,
        }
    }

    /// Probe all nodes and update routing table
    pub fn probe_all(&mut self) {
        for i in 0..self.nodes.len() {
            if self.nodes[i].needs_probe() {
                self.probe_node(i);
            }
        }
        self.update_best_nodes();
    }

    /// Update best node selection for each exchange
    fn update_best_nodes(&mut self) {
        // Group nodes by exchange and find best for each
        let exchanges: Vec<&str> = self.nodes.iter()
            .map(|n| n.exchange.as_str())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        self.best_nodes.clear();

        for exchange in &exchanges {
            let mut best_idx = None;
            let mut best_score = 0.0;

            for (i, node) in self.nodes.iter().enumerate() {
                if node.exchange == *exchange {
                    let score = node.score();
                    if score > best_score {
                        best_score = score;
                        best_idx = Some(i);
                    }
                }
            }

            if let Some(idx) = best_idx {
                self.best_nodes.push(idx);
            }
        }
    }

    /// Get best node for an exchange
    pub fn get_best_node(&self, exchange: &str) -> Option<&EdgeNode> {
        let mut best_idx = None;
        let mut best_score = 0.0;

        for (i, node) in self.nodes.iter().enumerate() {
            if node.exchange == exchange {
                let score = node.score();
                if score > best_score {
                    best_score = score;
                    best_idx = Some(i);
                }
            }
        }

        best_idx.and_then(|i| self.nodes.get(i))
    }

    /// Get best node across all exchanges (lowest RTT)
    pub fn get_global_best(&self) -> Option<&EdgeNode> {
        let mut best_idx = None;
        let mut best_rtt = u64::MAX;

        for (i, node) in self.nodes.iter().enumerate() {
            if node.is_reachable && node.last_rtt_us < best_rtt {
                best_rtt = node.last_rtt_us;
                best_idx = Some(i);
            }
        }

        best_idx.and_then(|i| self.nodes.get(i))
    }

    /// Get all nodes for an exchange
    pub fn get_exchange_nodes(&self, exchange: &str) -> Vec<&EdgeNode> {
        self.nodes.iter()
            .filter(|n| n.exchange == exchange)
            .collect()
    }

    /// Get node by ID
    pub fn get_node(&self, node_id: u32) -> Option<&EdgeNode> {
        self.nodes.iter().find(|n| n.node_id == node_id)
    }

    /// Get mutable node by ID
    pub fn get_node_mut(&mut self, node_id: u32) -> Option<&mut EdgeNode> {
        self.nodes.iter_mut().find(|n| n.node_id == node_id)
    }

    /// Add a custom edge node
    pub fn add_node(&mut self, node: EdgeNode) {
        self.nodes.push(node);
        self.update_best_nodes();
    }

    /// Remove unreachable nodes from consideration
    pub fn prune_unreachable(&mut self) {
        self.nodes.retain(|n| n.failures < 5);
        self.update_best_nodes();
    }

    /// Get routing statistics
    pub fn get_stats(&self) -> RouterStats {
        let total = self.nodes.len();
        let reachable = self.nodes.iter().filter(|n| n.is_reachable).count();
        
        let min_rtt = self.nodes.iter()
            .filter(|n| n.is_reachable)
            .map(|n| n.last_rtt_us)
            .min()
            .unwrap_or(u64::MAX);
        
        let max_rtt = self.nodes.iter()
            .filter(|n| n.is_reachable)
            .map(|n| n.last_rtt_us)
            .max()
            .unwrap_or(0);
        
        let avg_rtt = if reachable > 0 {
            self.nodes.iter()
                .filter(|n| n.is_reachable)
                .map(|n| n.last_rtt_us)
                .sum::<u64>() / reachable as u64
        } else {
            0
        };

        RouterStats {
            total_nodes: total,
            reachable_nodes: reachable,
            min_rtt_us: min_rtt,
            max_rtt_us: max_rtt,
            avg_rtt_us: avg_rtt,
        }
    }

    /// Set probe interval
    #[inline]
    pub fn set_probe_interval(&mut self, interval_ms: u64) {
        self.probe_interval_ms = interval_ms;
    }
}

impl Default for AnycastRouter {
    fn default() -> Self {
        Self::new()
    }
}

/// Router statistics
#[derive(Debug, Clone, Copy)]
pub struct RouterStats {
    pub total_nodes: usize,
    pub reachable_nodes: usize,
    pub min_rtt_us: u64,
    pub max_rtt_us: u64,
    pub avg_rtt_us: u64,
}

/// Simple pseudo-random for simulation
fn rand_u32() -> u32 {
    static mut SEED: u32 = 98765;
    unsafe {
        SEED = SEED.wrapping_mul(1664525).wrapping_add(1013904223);
        SEED
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_router_initialization() {
        let router = AnycastRouter::new();
        assert!(router.nodes.len() > 0);
    }

    #[test]
    fn test_probe_and_select() {
        let mut router = AnycastRouter::new();
        
        // Probe all nodes
        router.probe_all();
        
        // Should have selected best nodes
        let stats = router.get_stats();
        assert!(stats.reachable_nodes > 0);
    }

    #[test]
    fn test_get_best_node() {
        let mut router = AnycastRouter::new();
        router.probe_all();
        
        let best = router.get_best_node("binance");
        assert!(best.is_some());
        assert!(best.unwrap().is_reachable);
    }

    #[test]
    fn test_node_scoring() {
        let mut node = EdgeNode::new(0, "test", "us-east", "test.example.com");
        
        // Low RTT should give high score
        node.update_rtt(1000); // 1ms
        let score_low = node.score();
        
        node.update_rtt(50000); // 50ms
        let score_high = node.score();
        
        assert!(score_low > score_high);
    }
}
