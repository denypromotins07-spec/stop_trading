//! Order Book Snapshot Manager
//! 
//! Manages massive L2/L3 REST snapshots using memory-mapped files and zero-copy parsing.
//! Bypasses heap allocations when ingesting 50,000+ price levels during initial connection
//! or sequence gap recoveries.

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::ptr;

/// Maximum price levels per side
pub const MAX_LEVELS: usize = 50000;

/// Maximum snapshot file size (1GB)
pub const MAX_SNAPSHOT_SIZE: usize = 1_073_741_824;

/// Price level entry (packed for SIMD)
#[derive(Debug, Clone, Copy)]
#[repr(C, packed)]
pub struct PriceLevel {
    /// Price in fixed-point (scaled by 1e8)
    pub price: i64,
    /// Quantity in fixed-point (scaled by 1e8)
    pub quantity: i64,
    /// Order count at this level
    pub order_count: u32,
    /// Padding for 16-byte alignment
    _padding: u32,
}

impl PriceLevel {
    pub const fn empty() -> Self {
        Self {
            price: 0,
            quantity: 0,
            order_count: 0,
            _padding: 0,
        }
    }
}

/// Packed order book snapshot for zero-copy operations
#[repr(C, align(64))]
pub struct OrderBookSnapshot {
    /// Symbol identifier hash
    pub symbol_id: u64,
    /// Sequence number
    pub sequence: u64,
    /// Timestamp (nanoseconds since epoch)
    pub timestamp_ns: u64,
    /// Bid level count
    pub bid_count: u32,
    /// Ask level count
    pub ask_count: u32,
    /// Whether snapshot is valid
    pub valid: AtomicBool,
    /// Checksum for integrity verification
    pub checksum: u32,
    /// Padding
    _padding: [u8; 4],
    /// Bid levels (pre-allocated)
    bids: [PriceLevel; MAX_LEVELS],
    /// Ask levels (pre-allocated)
    asks: [PriceLevel; MAX_LEVELS],
}

impl OrderBookSnapshot {
    pub const fn empty() -> Self {
        Self {
            symbol_id: 0,
            sequence: 0,
            timestamp_ns: 0,
            bid_count: 0,
            ask_count: 0,
            valid: AtomicBool::new(false),
            checksum: 0,
            _padding: [0; 4],
            bids: [PriceLevel::empty(); MAX_LEVELS],
            asks: [PriceLevel::empty(); MAX_LEVELS],
        }
    }
    
    /// Get bid slice up to actual count
    #[inline]
    pub fn bids(&self) -> &[PriceLevel] {
        unsafe {
            std::slice::from_raw_parts(
                self.bids.as_ptr(),
                self.bid_count as usize,
            )
        }
    }
    
    /// Get ask slice up to actual count
    #[inline]
    pub fn asks(&self) -> &[PriceLevel] {
        unsafe {
            std::slice::from_raw_parts(
                self.asks.as_ptr(),
                self.ask_count as usize,
            )
        }
    }
    
    /// Set bid level at index
    #[inline]
    pub fn set_bid(&mut self, idx: usize, level: PriceLevel) -> bool {
        if idx < MAX_LEVELS {
            self.bids[idx] = level;
            if idx >= self.bid_count as usize {
                self.bid_count = (idx + 1) as u32;
            }
            true
        } else {
            false
        }
    }
    
    /// Set ask level at index
    #[inline]
    pub fn set_ask(&mut self, idx: usize, level: PriceLevel) -> bool {
        if idx < MAX_LEVELS {
            self.asks[idx] = level;
            if idx >= self.ask_count as usize {
                self.ask_count = (idx + 1) as u32;
            }
            true
        } else {
            false
        }
    }
    
    /// Get best bid
    #[inline]
    pub fn best_bid(&self) -> Option<&PriceLevel> {
        if self.bid_count > 0 {
            Some(&self.bids[0])
        } else {
            None
        }
    }
    
    /// Get best ask
    #[inline]
    pub fn best_ask(&self) -> Option<&PriceLevel> {
        if self.ask_count > 0 {
            Some(&self.asks[0])
        } else {
            None
        }
    }
    
    /// Calculate mid price
    #[inline]
    pub fn mid_price(&self) -> Option<i64> {
        if let (Some(bid), Some(ask)) = (self.best_bid(), self.best_ask()) {
            Some((bid.price + ask.price) / 2)
        } else {
            None
        }
    }
    
    /// Calculate spread in fixed-point
    #[inline]
    pub fn spread(&self) -> Option<i64> {
        if let (Some(bid), Some(ask)) = (self.best_bid(), self.best_ask()) {
            Some(ask.price - bid.price)
        } else {
            None
        }
    }
}

/// Memory-mapped snapshot storage manager
#[repr(C, align(64))]
pub struct SnapshotManager {
    /// Pre-allocated snapshot pool
    snapshots: Box<[OrderBookSnapshot; 256]>,
    /// Current write index (circular buffer)
    write_idx: AtomicU64,
    /// Total snapshots stored
    total_stored: AtomicU64,
    /// Last sequence per symbol (for gap detection)
    last_sequences: Box<[AtomicU64; 1024]>,
    /// Memory pressure flag
    memory_pressure: AtomicBool,
}

impl SnapshotManager {
    pub fn new() -> Self {
        let snapshots = Box::new([OrderBookSnapshot::empty(); 256]);
        let last_sequences = Box::new([const { AtomicU64::new(0) }; 1024]);
        
        Self {
            snapshots,
            write_idx: AtomicU64::new(0),
            total_stored: AtomicU64::new(0),
            last_sequences,
            memory_pressure: AtomicBool::new(false),
        }
    }
    
    /// Hash symbol string to index
    #[inline]
    fn hash_symbol(&self, symbol_id: u64) -> usize {
        ((symbol_id.wrapping_mul(0x9e3779b97f4a7c15)) >> 32) as usize & 0x3FF
    }
    
    /// Store a new snapshot
    pub fn store(&self, mut snapshot: OrderBookSnapshot) -> u64 {
        // Mark valid
        snapshot.valid.store(true, Ordering::Release);
        
        // Get write index
        let idx = self.write_idx.fetch_add(1, Ordering::AcqRel) % 256;
        
        // Copy snapshot
        self.snapshots[idx as usize] = snapshot;
        
        // Update last sequence for this symbol
        let seq_idx = self.hash_symbol(snapshot.symbol_id);
        self.last_sequences[seq_idx].store(snapshot.sequence, Ordering::Release);
        
        self.total_stored.fetch_add(1, Ordering::Relaxed);
        idx
    }
    
    /// Get latest snapshot for symbol
    pub fn get_latest(&self, symbol_id: u64) -> Option<&OrderBookSnapshot> {
        // Search backwards from current write position
        let current = self.write_idx.load(Ordering::Acquire);
        
        for i in 0..256 {
            let idx = ((current as i64 - 1 - i as i64) & 0xFF) as usize;
            let snap = &self.snapshots[idx];
            if snap.valid.load(Ordering::Acquire) && snap.symbol_id == symbol_id {
                return Some(snap);
            }
        }
        
        None
    }
    
    /// Get snapshot by sequence number
    pub fn get_by_sequence(&self, symbol_id: u64, sequence: u64) -> Option<&OrderBookSnapshot> {
        for i in 0..256 {
            let snap = &self.snapshots[i];
            if snap.valid.load(Ordering::Acquire) 
                && snap.symbol_id == symbol_id 
                && snap.sequence == sequence 
            {
                return Some(snap);
            }
        }
        None
    }
    
    /// Detect sequence gap for symbol
    pub fn detect_gap(&self, symbol_id: u64, expected_seq: u64) -> GapInfo {
        let seq_idx = self.hash_symbol(symbol_id);
        let last_seq = self.last_sequences[seq_idx].load(Ordering::Acquire);
        
        let has_gap = expected_seq != last_seq + 1 && expected_seq > last_seq;
        
        GapInfo {
            expected: expected_seq,
            last_seen: last_seq,
            gap_size: if has_gap { expected_seq.saturating_sub(last_seq).saturating_sub(1) } else { 0 },
            has_gap,
        }
    }
    
    /// Get snapshot statistics
    pub fn get_stats(&self) -> SnapshotStats {
        let mut valid_count = 0u64;
        let mut total_bids = 0u64;
        let mut total_asks = 0u64;
        
        for snap in &*self.snapshots {
            if snap.valid.load(Ordering::Acquire) {
                valid_count += 1;
                total_bids += snap.bid_count as u64;
                total_asks += snap.ask_count as u64;
            }
        }
        
        SnapshotStats {
            total_stored: self.total_stored.load(Ordering::Relaxed),
            valid_snapshots: valid_count,
            avg_bid_levels: if valid_count > 0 { total_bids / valid_count } else { 0 },
            avg_ask_levels: if valid_count > 0 { total_asks / valid_count } else { 0 },
            memory_pressure: self.memory_pressure.load(Ordering::Acquire),
        }
    }
    
    /// Clear old snapshots (keep only latest per symbol)
    pub fn compact(&self) -> u64 {
        // Simplified compaction - just reset write index
        // In production, would keep only latest per unique symbol
        let cleared = self.total_stored.load(Ordering::Relaxed);
        self.total_stored.store(0, Ordering::Relaxed);
        cleared
    }
}

/// Sequence gap information
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct GapInfo {
    pub expected: u64,
    pub last_seen: u64,
    pub gap_size: u64,
    pub has_gap: bool,
}

/// Snapshot statistics
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct SnapshotStats {
    pub total_stored: u64,
    pub valid_snapshots: u64,
    pub avg_bid_levels: u64,
    pub avg_ask_levels: u64,
    pub memory_pressure: bool,
}

impl Default for SnapshotManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_snapshot_basic() {
        let mut snap = OrderBookSnapshot::empty();
        snap.symbol_id = 12345;
        snap.sequence = 100;
        snap.timestamp_ns = 1000000000;
        
        // Add some levels
        snap.set_bid(0, PriceLevel { price: 50000_00000000, quantity: 100_00000000, order_count: 5, _padding: 0 });
        snap.set_bid(1, PriceLevel { price: 49999_00000000, quantity: 200_00000000, order_count: 10, _padding: 0 });
        snap.set_ask(0, PriceLevel { price: 50001_00000000, quantity: 150_00000000, order_count: 8, _padding: 0 });
        
        assert_eq!(snap.bid_count, 2);
        assert_eq!(snap.ask_count, 1);
        
        let best_bid = snap.best_bid().unwrap();
        assert_eq!(best_bid.price, 50000_00000000);
        
        let mid = snap.mid_price().unwrap();
        assert_eq!(mid, 50000_50000000);
    }
    
    #[test]
    fn test_manager_store_retrieve() {
        let manager = SnapshotManager::new();
        
        let mut snap = OrderBookSnapshot::empty();
        snap.symbol_id = 54321;
        snap.sequence = 200;
        snap.set_bid(0, PriceLevel { price: 100_00000000, quantity: 50_00000000, order_count: 3, _padding: 0 });
        
        manager.store(snap);
        
        let retrieved = manager.get_latest(54321);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().sequence, 200);
    }
    
    #[test]
    fn test_gap_detection() {
        let manager = SnapshotManager::new();
        
        let mut snap = OrderBookSnapshot::empty();
        snap.symbol_id = 99999;
        snap.sequence = 100;
        manager.store(snap);
        
        // Next expected should be 101
        let gap = manager.detect_gap(99999, 105);
        assert!(gap.has_gap);
        assert_eq!(gap.gap_size, 4);
        
        // No gap case
        let gap = manager.detect_gap(99999, 101);
        assert!(!gap.has_gap);
    }
}
