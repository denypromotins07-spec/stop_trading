//! Mempool Transaction Tracker
//! 
//! Tracks unconfirmed transactions, RBF (Replace-By-Fee), and CPFP (Child-Pays-For-Parent)
//! in a lock-free DAG. Monitors mempool density and transaction conflicts to predict
//! block inclusion probabilities for on-chain settlements.

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::sync::Arc;
use core::sync::atomic::AtomicPtr;

/// Fixed-point fee rate in sat/vByte scaled by 1e6 for precision
pub type FeeRate = u64;

/// Transaction ID (32 bytes compressed to u128 pair for cache efficiency)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct TxId {
    pub lo: u128,
    pub hi: u128,
}

impl TxId {
    pub const fn new(lo: u128, hi: u128) -> Self {
        Self { lo, hi }
    }
    
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.lo == 0 && self.hi == 0
    }
}

/// Mempool transaction entry with parent/child relationships
#[derive(Clone, Copy, Debug)]
#[repr(C, packed)]
pub struct MempoolTx {
    /// Transaction ID
    pub txid: TxId,
    /// Fee rate in sat/vByte (fixed-point, scaled by 1e6)
    pub fee_rate: FeeRate,
    /// Total fee in sats
    pub total_fee: u64,
    /// Virtual size in vBytes
    pub vsize: u32,
    /// Number of ancestors
    pub ancestor_count: u32,
    /// Ancestor fee sum in sats
    pub ancestor_fee: u64,
    /// Number of descendants
    pub descendant_count: u32,
    /// Descendant fee sum in sats
    pub descendant_fee: u64,
    /// Block height when first seen
    pub first_seen_height: u32,
    /// Unix timestamp when first seen
    pub first_seen_time: u64,
    /// Whether this tx signals RBF
    pub signals_rbf: bool,
    /// Whether this is a CPFP candidate
    pub is_cpfp_candidate: bool,
    /// Padding for 64-byte cache line alignment
    _padding: [u8; 5],
}

impl MempoolTx {
    pub const fn empty() -> Self {
        Self {
            txid: TxId::new(0, 0),
            fee_rate: 0,
            total_fee: 0,
            vsize: 0,
            ancestor_count: 0,
            ancestor_fee: 0,
            descendant_count: 0,
            descendant_fee: 0,
            first_seen_height: 0,
            first_seen_time: 0,
            signals_rbf: false,
            is_cpfp_candidate: false,
            _padding: [0; 5],
        }
    }
    
    /// Calculate effective fee rate including ancestors (for CPFP)
    #[inline]
    pub fn effective_fee_rate(&self) -> FeeRate {
        if self.vsize == 0 {
            return 0;
        }
        let total_vsize = self.vsize.saturating_add(
            (self.ancestor_fee * 1_000_000 / self.fee_rate.max(1)) as u32
        );
        let total_fee = self.total_fee.saturating_add(self.ancestor_fee);
        (total_fee * 1_000_000) / total_vsize.max(1) as u64
    }
}

/// Lock-free DAG node for mempool transaction relationships
#[repr(C)]
pub struct DagNode {
    /// Transaction data
    pub tx: MempoolTx,
    /// Parent pointers (compressed indices into arena)
    pub parent_indices: [AtomicU64; 4],
    /// Child pointers
    pub child_indices: [AtomicU64; 4],
    /// Number of parents
    pub parent_count: AtomicU64,
    /// Number of children
    pub child_count: AtomicU64,
    /// Next sibling in hash bucket chain
    pub next_in_bucket: AtomicU64,
    /// Whether this node is occupied
    pub occupied: AtomicBool,
}

impl DagNode {
    pub const fn new() -> Self {
        Self {
            tx: MempoolTx::empty(),
            parent_indices: [AtomicU64::new(u64::MAX); 4],
            child_indices: [AtomicU64::new(u64::MAX); 4],
            parent_count: AtomicU64::new(0),
            child_count: AtomicU64::new(0),
            next_in_bucket: AtomicU64::new(u64::MAX),
            occupied: AtomicBool::new(false),
        }
    }
}

/// Maximum number of transactions in the mempool DAG
pub const MAX_MEMPOOL_TXS: usize = 65536;

/// Lock-free Mempool DAG tracker
pub struct MempoolDag {
    /// Pre-allocated node arena (avoids malloc during operation)
    nodes: Box<[DagNode; MAX_MEMPOOL_TXS]>,
    /// Hash buckets for O(1) lookup by txid
    buckets: Box<[AtomicU64; 4096]>,
    /// Total transaction count
    tx_count: AtomicU64,
    /// Total mempool size in vBytes
    total_vsize: AtomicU64,
    /// Total fees in sats
    total_fees: AtomicU64,
    /// Current block height
    current_height: AtomicU64,
}

impl MempoolDag {
    pub fn new() -> Self {
        // Initialize with pre-allocated arena
        let nodes = Box::new([DagNode::new(); MAX_MEMPOOL_TXS]);
        let buckets = Box::new([AtomicU64::new(u64::MAX); 4096]);
        
        Self {
            nodes,
            buckets,
            tx_count: AtomicU64::new(0),
            total_vsize: AtomicU64::new(0),
            total_fees: AtomicU64::new(0),
            current_height: AtomicU64::new(0),
        }
    }
    
    /// Hash function for txid -> bucket index
    #[inline]
    fn hash_txid(&self, txid: &TxId) -> usize {
        // Simple xor-fold hash for speed
        ((txid.lo ^ txid.hi) as usize) & 0xFFF
    }
    
    /// Insert a transaction into the DAG
    pub fn insert(&self, tx: MempoolTx, parent_txids: &[TxId]) -> Option<u64> {
        let bucket = self.hash_txid(&tx.txid);
        let mut index = self.buckets[bucket].load(Ordering::Acquire);
        
        // Check if already exists
        while index != u64::MAX {
            if index as usize >= MAX_MEMPOOL_TXS {
                break;
            }
            let node = &self.nodes[index as usize];
            if node.tx.txid == tx.txid {
                // Update existing
                node.tx = tx;
                return Some(index);
            }
            index = node.next_in_bucket.load(Ordering::Acquire);
        }
        
        // Find free slot
        let new_index = self.find_free_slot()?;
        let node = &self.nodes[new_index];
        
        // Initialize node
        node.tx = tx;
        node.parent_count.store(parent_txids.len() as u64, Ordering::Release);
        node.occupied.store(true, Ordering::Release);
        
        // Link parents
        for (i, parent_txid) in parent_txids.iter().take(4).enumerate() {
            if let Some(parent_idx) = self.lookup_by_txid(parent_txid) {
                node.parent_indices[i].store(parent_idx, Ordering::Release);
                
                // Add reverse link (child pointer in parent)
                let parent_node = &self.nodes[parent_idx as usize];
                let child_count = parent_node.child_count.fetch_add(1, Ordering::AcqRel);
                if child_count < 4 {
                    parent_node.child_indices[child_count as usize].store(new_index as u64, Ordering::Release);
                }
            }
        }
        
        // Insert into bucket chain
        let mut head = self.buckets[bucket].load(Ordering::Acquire);
        loop {
            node.next_in_bucket.store(head, Ordering::Release);
            match self.buckets[bucket].compare_exchange(
                head,
                new_index as u64,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => break,
                Err(current) => head = current,
            }
        }
        
        // Update global stats
        self.tx_count.fetch_add(1, Ordering::Relaxed);
        self.total_vsize.fetch_add(tx.vsize as u64, Ordering::Relaxed);
        self.total_fees.fetch_add(tx.total_fee, Ordering::Relaxed);
        
        Some(new_index as u64)
    }
    
    /// Find a free slot in the arena
    fn find_free_slot(&self) -> Option<usize> {
        let count = self.tx_count.load(Ordering::Relaxed) as usize;
        if count >= MAX_MEMPOOL_TXS {
            return None;
        }
        
        // Linear scan from last known position (could be optimized with free list)
        for i in 0..MAX_MEMPOOL_TXS {
            if !self.nodes[i].occupied.load(Ordering::Acquire) {
                return Some(i);
            }
        }
        None
    }
    
    /// Lookup transaction by txid
    pub fn lookup_by_txid(&self, txid: &TxId) -> Option<u64> {
        let bucket = self.hash_txid(txid);
        let mut index = self.buckets[bucket].load(Ordering::Acquire);
        
        while index != u64::MAX {
            if index as usize >= MAX_MEMPOOL_TXS {
                break;
            }
            let node = &self.nodes[index as usize];
            if node.tx.txid == *txid {
                return Some(index);
            }
            index = node.next_in_bucket.load(Ordering::Acquire);
        }
        None
    }
    
    /// Remove confirmed transactions
    pub fn remove_confirmed(&self, confirmed_txids: &[TxId]) -> u64 {
        let mut removed = 0u64;
        for txid in confirmed_txids {
            if let Some(index) = self.lookup_by_txid(txid) {
                let node = &self.nodes[index as usize];
                self.total_vsize.fetch_sub(node.tx.vsize as u64, Ordering::Relaxed);
                self.total_fees.fetch_sub(node.tx.total_fee, Ordering::Relaxed);
                node.occupied.store(false, Ordering::Release);
                node.tx = MempoolTx::empty();
                removed += 1;
            }
        }
        self.tx_count.fetch_sub(removed, Ordering::Relaxed);
        removed
    }
    
    /// Predict block inclusion probability for a fee rate target
    pub fn inclusion_probability(&self, target_fee_rate: FeeRate) -> f64 {
        let total = self.tx_count.load(Ordering::Relaxed);
        if total == 0 {
            return 0.0;
        }
        
        let mut qualifying = 0u64;
        for i in 0..MAX_MEMPOOL_TXS {
            let node = &self.nodes[i];
            if node.occupied.load(Ordering::Acquire) {
                if node.tx.effective_fee_rate() >= target_fee_rate {
                    qualifying += 1;
                }
            }
        }
        
        qualifying as f64 / total as f64
    }
    
    /// Get mempool density metrics
    pub fn density_metrics(&self) -> MempoolDensity {
        let tx_count = self.tx_count.load(Ordering::Relaxed);
        let total_vsize = self.total_vsize.load(Ordering::Relaxed);
        let total_fees = self.total_fees.load(Ordering::Relaxed);
        
        MempoolDensity {
            tx_count,
            total_vsize,
            total_fees,
            avg_fee_rate: if total_vsize > 0 {
                (total_fees * 1_000_000) / total_vsize
            } else {
                0
            },
            utilization_pct: (total_vsize * 100) / (4_000_000), // Assuming 4MB block target
        }
    }
    
    /// Detect RBF candidates
    pub fn get_rbf_candidates(&self) -> Vec<&MempoolTx> {
        let mut candidates = Vec::with_capacity(256);
        for i in 0..MAX_MEMPOOL_TXS {
            let node = &self.nodes[i];
            if node.occupied.load(Ordering::Acquire) && node.tx.signals_rbf {
                candidates.push(&node.tx);
            }
        }
        candidates
    }
    
    /// Detect CPFP opportunities
    pub fn get_cpfp_opportunities(&self, min_effective_rate: FeeRate) -> Vec<&MempoolTx> {
        let mut opportunities = Vec::with_capacity(128);
        for i in 0..MAX_MEMPOOL_TXS {
            let node = &self.nodes[i];
            if node.occupied.load(Ordering::Acquire) {
                if node.tx.effective_fee_rate() >= min_effective_rate {
                    opportunities.push(&node.tx);
                }
            }
        }
        opportunities
    }
    
    /// Set current block height
    pub fn set_block_height(&self, height: u64) {
        self.current_height.store(height, Ordering::Release);
    }
}

/// Mempool density metrics
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct MempoolDensity {
    pub tx_count: u64,
    pub total_vsize: u64,
    pub total_fees: u64,
    pub avg_fee_rate: FeeRate,
    pub utilization_pct: u64,
}

impl Default for MempoolDag {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_mempool_insert_lookup() {
        let dag = MempoolDag::new();
        let tx = MempoolTx {
            txid: TxId::new(1, 2),
            fee_rate: 50_000_000, // 50 sat/vByte
            total_fee: 10_000,
            vsize: 200,
            ..MempoolTx::empty()
        };
        
        assert!(dag.insert(tx, &[]).is_some());
        assert!(dag.lookup_by_txid(&TxId::new(1, 2)).is_some());
        assert!(dag.lookup_by_txid(&TxId::new(999, 999)).is_none());
    }
    
    #[test]
    fn test_inclusion_probability() {
        let dag = MempoolDag::new();
        
        // Insert transactions with varying fee rates
        for i in 0..100 {
            let tx = MempoolTx {
                txid: TxId::new(i, 0),
                fee_rate: (i * 1_000_000) as u64, // 0 to 99 sat/vByte
                total_fee: i as u64 * 100,
                vsize: 100,
                ..MempoolTx::empty()
            };
            dag.insert(tx, &[]);
        }
        
        // 50% should qualify at 50 sat/vByte
        let prob = dag.inclusion_probability(50_000_000);
        assert!(prob > 0.4 && prob < 0.6);
    }
}
