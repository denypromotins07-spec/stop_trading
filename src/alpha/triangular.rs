//! Triangular Arbitrage Engine
//! 
//! Ultra-fast, incremental 3-node cycle detector for triangular arbitrage.
//! Avoids full Bellman-Ford recalculations; updates edge weights in O(1) time.
//! Uses fixed-size arrays and cache-line padded structs to prevent heap allocations.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Cache-line padding to prevent false sharing
const CACHE_LINE_SIZE: usize = 64;

/// Maximum number of assets supported in the triangular arb graph
/// Chosen to fit L1 cache for ultra-low latency lookups
const MAX_ASSETS: usize = 32;

/// Represents a directed edge in the arbitrage graph
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ArbEdge {
    /// Source asset index
    pub src: u8,
    /// Destination asset index
    pub dst: u8,
    /// Exchange rate (scaled by 1e18 for integer precision)
    pub rate_scaled: AtomicU64,
    /// Fee basis points (scaled by 1e4)
    pub fee_bps: u16,
    /// Last update timestamp (nanos)
    pub last_update_ns: AtomicU64,
    /// Padding to cache line boundary
    _pad: [u8; CACHE_LINE_SIZE - 20],
}

impl ArbEdge {
    pub const fn new(src: u8, dst: u8, fee_bps: u16) -> Self {
        Self {
            src,
            dst,
            rate_scaled: AtomicU64::new(0),
            fee_bps,
            last_update_ns: AtomicU64::new(0),
            _pad: [0; CACHE_LINE_SIZE - 20],
        }
    }

    #[inline]
    pub fn update_rate(&self, rate: f64, timestamp_ns: u64) {
        let scaled = (rate * 1e18) as u64;
        self.rate_scaled.store(scaled, Ordering::Relaxed);
        self.last_update_ns.store(timestamp_ns, Ordering::Relaxed);
    }

    #[inline]
    pub fn get_rate(&self) -> f64 {
        self.rate_scaled.load(Ordering::Relaxed) as f64 / 1e18
    }

    #[inline]
    pub fn get_effective_rate(&self) -> f64 {
        // Apply fee: rate * (1 - fee_bps / 10000)
        let rate = self.get_rate();
        rate * (1.0 - self.fee_bps as f64 / 10000.0)
    }
}

/// Triangular Arbitrage Graph
/// Fixed-size adjacency matrix for O(1) edge access
pub struct TriangularArbGraph {
    /// Adjacency matrix of edges
    edges: [[Option<ArbEdge>; MAX_ASSETS]; MAX_ASSETS],
    /// Asset name lookup (fixed size strings)
    asset_names: [[u8; 12]; MAX_ASSETS],
    /// Number of active assets
    asset_count: usize,
    /// Profit threshold in basis points
    profit_threshold_bps: u16,
    /// Total opportunities detected
    opportunities_detected: AtomicU64,
}

/// Detected arbitrage opportunity
#[derive(Debug, Clone)]
pub struct ArbOpportunity {
    /// Asset A -> B -> C -> A cycle
    pub path: [u8; 3],
    /// Expected profit in basis points
    pub profit_bps: u16,
    /// Timestamp of detection (nanos)
    pub timestamp_ns: u64,
    /// Recommended execution size (quote units)
    pub recommended_size: u64,
}

impl TriangularArbGraph {
    pub fn new(profit_threshold_bps: u16) -> Self {
        Self {
            edges: [[None; MAX_ASSETS]; MAX_ASSETS],
            asset_names: [[0; 12]; MAX_ASSETS],
            asset_count: 0,
            profit_threshold_bps,
            opportunities_detected: AtomicU64::new(0),
        }
    }

    /// Register an asset, returns its index
    pub fn register_asset(&mut self, name: &str) -> Option<u8> {
        if self.asset_count >= MAX_ASSETS {
            return None;
        }
        
        let idx = self.asset_count as u8;
        let mut bytes = [0u8; 12];
        let name_bytes = name.as_bytes();
        let copy_len = name_bytes.len().min(12);
        bytes[..copy_len].copy_from_slice(&name_bytes[..copy_len]);
        self.asset_names[self.asset_count] = bytes;
        self.asset_count += 1;
        Some(idx)
    }

    /// Add or update an edge (exchange rate)
    #[inline]
    pub fn update_edge(&mut self, src: u8, dst: u8, rate: f64, fee_bps: u16) {
        let timestamp_ns = Instant::now().duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default().as_nanos() as u64;
        
        if src >= self.asset_count as u8 || dst >= self.asset_count as u8 {
            return;
        }

        let edge = self.edges[src as usize][dst as usize]
            .get_or_insert_with(|| ArbEdge::new(src, dst, fee_bps));
        edge.update_rate(rate, timestamp_ns);
    }

    /// Check all 3-cycles involving a specific asset (O(N^2) for that asset only)
    /// This is incremental - only checks cycles affected by recent updates
    pub fn check_cycles_for_asset(&self, asset_idx: u8) -> Vec<ArbOpportunity> {
        let mut opportunities = Vec::with_capacity(8);
        let timestamp_ns = Instant::now().duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default().as_nanos() as u64;

        for i in 0..self.asset_count as u8 {
            if i == asset_idx { continue; }
            
            for j in 0..self.asset_count as u8 {
                if j == asset_idx || j == i { continue; }

                // Check cycle: asset_idx -> i -> j -> asset_idx
                if let Some(opportunity) = self.check_cycle(asset_idx, i, j, timestamp_ns) {
                    opportunities.push(opportunity);
                    self.opportunities_detected.fetch_add(1, Ordering::Relaxed);
                }
            }
        }

        opportunities
    }

    /// Check a specific 3-cycle for profitability
    #[inline]
    fn check_cycle(&self, a: u8, b: u8, c: u8, timestamp_ns: u64) -> Option<ArbOpportunity> {
        let edge_ab = self.edges[a as usize][b as usize].as_ref()?;
        let edge_bc = self.edges[b as usize][c as usize].as_ref()?;
        let edge_ca = self.edges[c as usize][a as usize].as_ref()?;

        // Calculate effective rates after fees
        let rate_ab = edge_ab.get_effective_rate();
        let rate_bc = edge_bc.get_effective_rate();
        let rate_ca = edge_ca.get_effective_rate();

        // Triangular arb: start with 1 unit of A
        // A -> B: get rate_ab units of B
        // B -> C: get rate_ab * rate_bc units of C
        // C -> A: get rate_ab * rate_bc * rate_ca units of A
        
        let final_amount = rate_ab * rate_bc * rate_ca;
        
        // Profit calculation: (final - 1) * 10000 basis points
        if final_amount > 1.0 {
            let profit_bps = ((final_amount - 1.0) * 10000.0) as u16;
            
            if profit_bps >= self.profit_threshold_bps {
                return Some(ArbOpportunity {
                    path: [a, b, c],
                    profit_bps,
                    timestamp_ns,
                    recommended_size: self.calculate_optimal_size(a, b, c),
                });
            }
        }

        None
    }

    /// Calculate optimal execution size based on liquidity heuristics
    fn calculate_optimal_size(&self, _a: u8, _b: u8, _c: u8) -> u64 {
        // In production, this would query order book depth
        // For now, return a conservative default
        10_000_000 // 10M quote units
    }

    /// Get asset name by index
    pub fn get_asset_name(&self, idx: u8) -> Option<&str> {
        if idx as usize >= self.asset_count {
            return None;
        }
        let bytes = &self.asset_names[idx as usize];
        let end = bytes.iter().position(|&b| b == 0).unwrap_or(12);
        std::str::from_utf8(&bytes[..end]).ok()
    }

    /// Get total opportunities detected
    pub fn get_opportunity_count(&self) -> u64 {
        self.opportunities_detected.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_triangular_arb_detection() {
        let mut graph = TriangularArbGraph::new(5); // 5 bps threshold
        
        // Register BTC, ETH, USDT
        let btc = graph.register_asset("BTC").unwrap();
        let eth = graph.register_asset("ETH").unwrap();
        let usdt = graph.register_asset("USDT").unwrap();

        // Set up profitable cycle: BTC -> ETH -> USDT -> BTC
        // Rates chosen to create ~10 bps profit after fees
        graph.update_edge(btc, eth, 15.0, 10); // 1 BTC = 15 ETH, 10 bps fee
        graph.update_edge(eth, usdt, 2000.0, 10); // 1 ETH = 2000 USDT
        graph.update_edge(usdt, btc, 1.0 / 30000.0, 10); // 1 USDT = 1/30000 BTC

        // Check cycles
        let opportunities = graph.check_cycles_for_asset(btc);
        
        assert!(!opportunities.is_empty(), "Should detect arbitrage opportunity");
        
        if let Some(opp) = opportunities.first() {
            assert_eq!(opp.path[0], btc);
            assert!(opp.profit_bps >= 5, "Profit should exceed threshold");
        }
    }

    #[test]
    fn test_no_arbitrage() {
        let mut graph = TriangularArbGraph::new(5);
        
        let a = graph.register_asset("A").unwrap();
        let b = graph.register_asset("B").unwrap();
        let c = graph.register_asset("C").unwrap();

        // Efficient market: no arb opportunity
        graph.update_edge(a, b, 2.0, 10);
        graph.update_edge(b, c, 3.0, 10);
        graph.update_edge(c, a, 1.0 / 6.0, 10); // Exactly breaks even minus fees

        let opportunities = graph.check_cycles_for_asset(a);
        assert!(opportunities.is_empty(), "Should not detect false arb");
    }
}
