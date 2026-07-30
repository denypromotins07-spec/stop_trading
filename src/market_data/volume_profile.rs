//! Volume Profile Engine
//! 
//! High-speed Volume Profile aggregation identifying Point of Control (POC),
//! Value Area High/Low (VAH/VAL), and high-volume nodes using fixed-size hash maps.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use crossbeam_utils::CachePadded;

/// Maximum number of price nodes in the volume profile
const MAX_PRICE_NODES: usize = 8192;

/// Fixed-size bucket for volume at price
#[repr(C)]
#[derive(Clone, Copy)]
pub struct VolumeNode {
    /// Price level (in micro-units)
    pub price_micros: u64,
    /// Total volume at this price (base units, scaled by 1000)
    pub volume_scaled: u64,
    /// Buy volume (scaled by 1000)
    pub buy_volume_scaled: u64,
    /// Sell volume (scaled by 1000)
    pub sell_volume_scaled: u64,
    /// Trade count at this level
    pub trade_count: u32,
    /// Hash for quick lookup
    pub hash: u32,
}

impl Default for VolumeNode {
    fn default() -> Self {
        Self {
            price_micros: 0,
            volume_scaled: 0,
            buy_volume_scaled: 0,
            sell_volume_scaled: 0,
            trade_count: 0,
            hash: 0,
        }
    }
}

impl VolumeNode {
    #[inline]
    fn new(price_micros: u64) -> Self {
        Self {
            price_micros,
            ..Default::default()
        }
    }

    #[inline]
    fn add_trade(&mut self, volume: u64, is_buy: bool) {
        let vol_scaled = volume * 1000;
        self.volume_scaled = self.volume_scaled.saturating_add(vol_scaled);
        if is_buy {
            self.buy_volume_scaled = self.buy_volume_scaled.saturating_add(vol_scaled);
        } else {
            self.sell_volume_scaled = self.sell_volume_scaled.saturating_add(vol_scaled);
        }
        self.trade_count = self.trade_count.saturating_add(1);
    }

    #[inline]
    fn clear(&mut self) {
        self.volume_scaled = 0;
        self.buy_volume_scaled = 0;
        self.sell_volume_scaled = 0;
        self.trade_count = 0;
    }
}

/// Volume Profile statistics
pub struct VolumeProfileStats {
    /// Point of Control - price with highest volume
    pub poc_price: u64,
    /// Point of Control volume
    pub poc_volume: u64,
    /// Value Area High
    pub vah_price: u64,
    /// Value Area Low
    pub val_price: u64,
    /// Total profile volume
    pub total_volume: u64,
    /// Number of active price nodes
    pub node_count: usize,
    /// Value area percentage (typically 70%)
    pub value_area_pct: u32,
}

/// Lock-free Volume Profile engine with pre-allocated storage
pub struct VolumeProfileEngine {
    /// Fixed-size node array (pre-allocated)
    nodes: CachePadded<[VolumeNode; MAX_PRICE_NODES]>,
    /// Number of active nodes
    active_count: CachePadded<AtomicUsize>,
    /// Price bucket size (for grouping prices into levels)
    bucket_size_micros: u64,
    /// Total volume across all nodes
    total_volume: CachePadded<AtomicU64>,
    /// Session high price
    session_high: CachePadded<AtomicU64>,
    /// Session low price
    session_low: CachePadded<AtomicU64>,
    /// Profile version (for snapshot consistency)
    version: CachePadded<AtomicU64>,
    /// Value area percentage (scaled by 100, e.g., 70 = 70%)
    value_area_pct: u32,
}

impl VolumeProfileEngine {
    /// Create a new volume profile engine
    /// 
    /// # Arguments
    /// * `bucket_size_micros` - Price bucket size in micro-units for grouping
    /// * `value_area_pct` - Value area percentage (e.g., 70 for 70%)
    pub fn new(bucket_size_micros: u64, value_area_pct: u32) -> Self {
        Self {
            nodes: CachePadded::new(std::array::from_fn(|_| VolumeNode::default())),
            active_count: CachePadded::new(AtomicUsize::new(0)),
            bucket_size_micros,
            total_volume: CachePadded::new(AtomicU64::new(0)),
            session_high: CachePadded::new(AtomicU64::new(0)),
            session_low: CachePadded::new(AtomicU64::new(u64::MAX)),
            version: CachePadded::new(AtomicU64::new(0)),
            value_area_pct,
        }
    }

    /// Add a trade to the volume profile
    /// 
    /// # Arguments
    /// * `price_micros` - Trade price in micro-units
    /// * `volume` - Trade volume in base units
    /// * `is_buy` - True if buyer-initiated, false if seller-initiated
    #[inline]
    pub fn add_trade(&self, price_micros: u64, volume: u64, is_buy: bool) {
        // Round price to bucket
        let bucketed_price = (price_micros / self.bucket_size_micros) * self.bucket_size_micros;
        
        // Update session extremes
        let current_high = self.session_high.load(Ordering::Relaxed);
        let current_low = self.session_low.load(Ordering::Relaxed);
        
        if price_micros > current_high {
            self.session_high.store(price_micros, Ordering::Relaxed);
        }
        if price_micros < current_low {
            self.session_low.store(price_micros, Ordering::Relaxed);
        }

        // Find or create node using linear probing
        let hash = (bucketed_price / self.bucket_size_micros) as u32;
        let mut idx = (hash as usize) % MAX_PRICE_NODES;
        let mut first_empty = None;
        
        for _ in 0..MAX_PRICE_NODES {
            let node_price = self.nodes[idx].price_micros;
            
            if node_price == 0 {
                // Empty slot
                if first_empty.is_none() {
                    first_empty = Some(idx);
                }
                idx = (idx + 1) % MAX_PRICE_NODES;
                continue;
            }
            
            if node_price == bucketed_price {
                // Found existing node
                unsafe {
                    let node_ptr = &self.nodes[idx] as *const VolumeNode as *mut VolumeNode;
                    (*node_ptr).add_trade(volume, is_buy);
                }
                self.total_volume.fetch_add(volume * 1000, Ordering::Relaxed);
                self.version.fetch_add(1, Ordering::Relaxed);
                return;
            }
            
            idx = (idx + 1) % MAX_PRICE_NODES;
        }

        // Need to insert into empty slot
        if let Some(insert_idx) = first_empty {
            unsafe {
                let node_ptr = &self.nodes[insert_idx] as *const VolumeNode as *mut VolumeNode;
                (*node_ptr) = VolumeNode::new(bucketed_price);
                (*node_ptr).add_trade(volume, is_buy);
                (*node_ptr).hash = hash;
            }
            self.active_count.fetch_add(1, Ordering::Relaxed);
            self.total_volume.fetch_add(volume * 1000, Ordering::Relaxed);
            self.version.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Calculate Point of Control
    #[inline]
    pub fn calculate_poc(&self) -> (u64, u64) {
        let mut poc_price = 0u64;
        let mut poc_volume = 0u64;
        let count = self.active_count.load(Ordering::Relaxed);

        for i in 0..MAX_PRICE_NODES {
            let node = &self.nodes[i];
            if node.price_micros > 0 && node.volume_scaled > poc_volume {
                poc_volume = node.volume_scaled;
                poc_price = node.price_micros;
            }
        }

        (poc_price, poc_volume)
    }

    /// Calculate Value Area High and Low
    /// Returns (VAH, VAL) encompassing the specified percentage of volume around POC
    pub fn calculate_value_area(&self) -> (u64, u64) {
        let (poc_price, poc_volume) = self.calculate_poc();
        if poc_price == 0 || poc_volume == 0 {
            return (0, 0);
        }

        let total_vol = self.total_volume.load(Ordering::Relaxed);
        if total_vol == 0 {
            return (0, 0);
        }

        let target_volume = (total_vol * self.value_area_pct as u64) / 100;
        
        // Collect and sort nodes by price
        let mut price_volumes: [(u64, u64); MAX_PRICE_NODES] = [(0, 0); MAX_PRICE_NODES];
        let mut valid_count = 0;

        for i in 0..MAX_PRICE_NODES {
            let node = &self.nodes[i];
            if node.price_micros > 0 {
                price_volumes[valid_count] = (node.price_micros, node.volume_scaled);
                valid_count += 1;
            }
        }

        if valid_count == 0 {
            return (0, 0);
        }

        // Simple bubble sort for small arrays (more efficient than complex sorting for our use case)
        for i in 0..valid_count.saturating_sub(1) {
            for j in 0..valid_count.saturating_sub(i + 1) {
                if price_volumes[j].0 > price_volumes[j + 1].0 {
                    price_volumes.swap(j, j + 1);
                }
            }
        }

        // Find POC index
        let mut poc_idx = 0;
        for i in 0..valid_count {
            if price_volumes[i].0 == poc_price {
                poc_idx = i;
                break;
            }
        }

        // Expand from POC to capture target volume
        let mut accumulated = price_volumes[poc_idx].1;
        let mut left = poc_idx;
        let mut right = poc_idx;

        while accumulated < target_volume {
            let left_vol = if left > 0 { price_volumes[left - 1].1 } else { 0 };
            let right_vol = if right + 1 < valid_count { price_volumes[right + 1].1 } else { 0 };

            if left_vol >= right_vol {
                if left > 0 {
                    left -= 1;
                    accumulated = accumulated.saturating_add(left_vol);
                } else if right + 1 < valid_count {
                    right += 1;
                    accumulated = accumulated.saturating_add(right_vol);
                } else {
                    break;
                }
            } else {
                if right + 1 < valid_count {
                    right += 1;
                    accumulated = accumulated.saturating_add(right_vol);
                } else if left > 0 {
                    left -= 1;
                    accumulated = accumulated.saturating_add(left_vol);
                } else {
                    break;
                }
            }
        }

        (price_volumes[right].0, price_volumes[left].0)
    }

    /// Get complete volume profile statistics
    pub fn get_stats(&self) -> VolumeProfileStats {
        let (poc_price, poc_volume) = self.calculate_poc();
        let (vah, val) = self.calculate_value_area();

        VolumeProfileStats {
            poc_price,
            poc_volume,
            vah_price: vah,
            val_price: val,
            total_volume: self.total_volume.load(Ordering::Relaxed),
            node_count: self.active_count.load(Ordering::Relaxed),
            value_area_pct: self.value_area_pct,
        }
    }

    /// Get volume at a specific price level
    #[inline]
    pub fn get_volume_at_price(&self, price_micros: u64) -> u64 {
        let bucketed_price = (price_micros / self.bucket_size_micros) * self.bucket_size_micros;
        
        for i in 0..MAX_PRICE_NODES {
            let node = &self.nodes[i];
            if node.price_micros == bucketed_price {
                return node.volume_scaled;
            }
            if node.price_micros == 0 {
                break;
            }
        }
        0
    }

    /// Get session high
    #[inline]
    pub fn get_session_high(&self) -> u64 {
        self.session_high.load(Ordering::Relaxed)
    }

    /// Get session low
    #[inline]
    pub fn get_session_low(&self) -> u64 {
        let low = self.session_low.load(Ordering::Relaxed);
        if low == u64::MAX { 0 } else { low }
    }

    /// Reset the profile (for new session)
    pub fn reset(&self) {
        for i in 0..MAX_PRICE_NODES {
            unsafe {
                let node_ptr = &self.nodes[i] as *const VolumeNode as *mut VolumeNode;
                (*node_ptr).clear();
            }
        }
        self.active_count.store(0, Ordering::Relaxed);
        self.total_volume.store(0, Ordering::Relaxed);
        self.session_high.store(0, Ordering::Relaxed);
        self.session_low.store(u64::MAX, Ordering::Relaxed);
        self.version.fetch_add(1, Ordering::Relaxed);
    }

    /// Get profile version
    #[inline]
    pub fn get_version(&self) -> u64 {
        self.version.load(Ordering::Relaxed)
    }

    /// Identify high-volume nodes (nodes with volume > average * threshold_multiplier)
    pub fn get_high_volume_nodes(&self, threshold_multiplier: u32) -> Vec<(u64, u64)> {
        let mut result = Vec::with_capacity(64);
        let count = self.active_count.load(Ordering::Relaxed);
        if count == 0 {
            return result;
        }

        let total = self.total_volume.load(Ordering::Relaxed);
        let avg_volume = total / count as u64;
        let threshold = avg_volume * threshold_multiplier as u64;

        for i in 0..MAX_PRICE_NODES {
            let node = &self.nodes[i];
            if node.price_micros > 0 && node.volume_scaled > threshold {
                result.push((node.price_micros, node.volume_scaled));
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_volume_profile_basic() {
        let profile = VolumeProfileEngine::new(100, 70);
        
        // Add trades at different price levels
        profile.add_trade(50000000, 100, true);
        profile.add_trade(50000100, 50, false);
        profile.add_trade(50000000, 200, true); // More volume at this level
        
        let stats = profile.get_stats();
        
        assert!(stats.total_volume > 0);
        assert_eq!(stats.poc_price, 50000000); // Highest volume
        assert!(stats.node_count > 0);
    }

    #[test]
    fn test_value_area_calculation() {
        let profile = VolumeProfileEngine::new(100, 70);
        
        // Create a distribution with clear POC
        for i in 0..100 {
            let price = 50000000 + (i * 100);
            let volume = if i == 50 { 1000 } else { 10 }; // Middle is POC
            profile.add_trade(price, volume, i % 2 == 0);
        }
        
        let stats = profile.get_stats();
        assert_eq!(stats.poc_price, 50005000);
        assert!(stats.vah_price >= stats.poc_price);
        assert!(stats.val_price <= stats.poc_price);
    }
}
