//! Footprint Charts & Delta Analytics
//! Aggregates trade ticks into price-node footprint clusters using bounded arrays.
//! Uses #[repr(C, packed)] for exact memory alignment for SIMD processing.

use core::sync::atomic::{AtomicI64, AtomicU64, AtomicUsize, Ordering};

/// Maximum price levels to track
const MAX_PRICE_LEVELS: usize = 1024;

/// Bid/Ask volume at a single price level
#[repr(C, packed)]
#[derive(Debug, Clone, Copy, Default)]
pub struct PriceNode {
    pub price: i64,           // Fixed-point price
    pub bid_volume: u64,      // Volume traded at bid
    pub ask_volume: u64,      // Volume traded at ask
    pub total_volume: u64,    // Total volume
    pub delta: i64,           // Ask - Bid (signed)
    pub trade_count: u32,     // Number of trades
    pub high: i64,            // Highest trade price at this level
    pub low: i64,             // Lowest trade price at this level
}

/// Footprint cluster for a single candle/bar
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct FootprintCluster {
    pub nodes: [PriceNode; MAX_PRICE_LEVELS],
    pub node_count: AtomicUsize,
    pub total_bid_volume: AtomicU64,
    pub total_ask_volume: AtomicU64,
    pub net_delta: AtomicI64,
    pub poc_price: AtomicI64,   // Point of Control (highest volume price)
    pub poc_volume: AtomicU64,
}

impl Default for FootprintCluster {
    fn default() -> Self {
        Self::new()
    }
}

impl FootprintCluster {
    pub const fn new() -> Self {
        Self {
            nodes: unsafe { core::mem::zeroed() },
            node_count: AtomicUsize::new(0),
            total_bid_volume: AtomicU64::new(0),
            total_ask_volume: AtomicU64::new(0),
            net_delta: AtomicI64::new(0),
            poc_price: AtomicI64::new(0),
            poc_volume: AtomicU64::new(0),
        }
    }

    /// Add a trade to the footprint
    pub fn add_trade(&self, price: i64, volume: u64, is_buyer_initiated: bool) {
        let node_idx = self.find_or_create_node(price);
        
        if let Some(node) = unsafe { self.nodes.get_unchecked_mut(node_idx) } {
            if is_buyer_initiated {
                node.ask_volume = node.ask_volume.saturating_add(volume);
            } else {
                node.bid_volume = node.bid_volume.saturating_add(volume);
            }
            
            node.total_volume = node.total_volume.saturating_add(volume);
            node.delta = node.ask_volume as i64 - node.bid_volume as i64;
            node.trade_count += 1;
            
            if node.high == 0 || price > node.high {
                node.high = price;
            }
            if node.low == 0 || price < node.low {
                node.low = price;
            }

            // Update totals
            if is_buyer_initiated {
                self.total_ask_volume.fetch_add(volume, Ordering::Relaxed);
            } else {
                self.total_bid_volume.fetch_add(volume, Ordering::Relaxed);
            }
            
            self.net_delta.fetch_add(
                if is_buyer_initiated { volume as i64 } else { -(volume as i64) },
                Ordering::Relaxed,
            );

            // Update POC
            let current_poc_vol = self.poc_volume.load(Ordering::Relaxed);
            if node.total_volume > current_poc_vol {
                self.poc_volume.store(node.total_volume, Ordering::Relaxed);
                self.poc_price.store(price, Ordering::Relaxed);
            }
        }
    }

    fn find_or_create_node(&self, price: i64) -> usize {
        let count = self.node_count.load(Ordering::Relaxed);
        
        // Search existing nodes
        for i in 0..count {
            unsafe {
                let node = self.nodes.get_unchecked(i);
                if node.price == price {
                    return i;
                }
            }
        }
        
        // Create new node
        if count < MAX_PRICE_LEVELS {
            let idx = count;
            unsafe {
                let node = self.nodes.get_unchecked_mut(idx);
                node.price = price;
                node.low = price;
                node.high = price;
            }
            self.node_count.store(count + 1, Ordering::Relaxed);
            idx
        } else {
            0 // Fallback to first node if full
        }
    }

    /// Get cumulative volume delta (CVD) up to a price level
    pub fn get_cvd_up_to(&self, price: i64) -> i64 {
        let count = self.node_count.load(Ordering::Relaxed);
        let mut cvd = 0i64;
        
        for i in 0..count {
            unsafe {
                let node = self.nodes.get_unchecked(i);
                if node.price <= price {
                    cvd += node.delta;
                }
            }
        }
        
        cvd
    }

    /// Get imbalance ratio at a price level
    pub fn get_imbalance(&self, price: i64) -> Option<f64> {
        let count = self.node_count.load(Ordering::Relaxed);
        
        for i in 0..count {
            unsafe {
                let node = self.nodes.get_unchecked(i);
                if node.price == price {
                    let total = node.bid_volume + node.ask_volume;
                    if total == 0 {
                        return None;
                    }
                    return Some(node.ask_volume as f64 / total as f64);
                }
            }
        }
        
        None
    }

    /// Get all nodes as slice
    pub fn get_nodes(&self) -> &[PriceNode] {
        let count = self.node_count.load(Ordering::Relaxed);
        unsafe { core::slice::from_raw_parts(self.nodes.as_ptr(), count) }
    }

    /// Reset the cluster
    pub fn reset(&mut self) {
        self.node_count.store(0, Ordering::Relaxed);
        self.total_bid_volume.store(0, Ordering::Relaxed);
        self.total_ask_volume.store(0, Ordering::Relaxed);
        self.net_delta.store(0, Ordering::Relaxed);
        self.poc_price.store(0, Ordering::Relaxed);
        self.poc_volume.store(0, Ordering::Relaxed);
        
        for i in 0..MAX_PRICE_LEVELS {
            unsafe {
                *self.nodes.get_unchecked_mut(i) = PriceNode::default();
            }
        }
    }
}

/// Footprint chart manager for multiple bars
pub struct FootprintChart {
    clusters: [FootprintCluster; 100],
    current_idx: AtomicUsize,
}

impl Default for FootprintChart {
    fn default() -> Self {
        Self::new()
    }
}

impl FootprintChart {
    pub const fn new() -> Self {
        Self {
            clusters: unsafe { core::mem::zeroed() },
            current_idx: AtomicUsize::new(0),
        }
    }

    pub fn get_current_cluster(&self) -> &FootprintCluster {
        let idx = self.current_idx.load(Ordering::Relaxed);
        unsafe { self.clusters.get_unchecked(idx.min(99)) }
    }

    pub fn next_bar(&self) {
        let idx = self.current_idx.load(Ordering::Relaxed);
        self.current_idx.store((idx + 1).min(99), Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_footprint_cluster() {
        let cluster = FootprintCluster::new();
        
        // Add some trades
        cluster.add_trade(100_0000_0000i64, 100, true);   // Buyer initiated
        cluster.add_trade(100_0000_0000i64, 50, false);   // Seller initiated
        cluster.add_trade(101_0000_0000i64, 200, true);
        
        assert_eq!(cluster.node_count.load(Ordering::Relaxed), 2);
        assert!(cluster.net_delta.load(Ordering::Relaxed) > 0);
    }
}
