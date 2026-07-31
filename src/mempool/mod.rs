//! Mempool Module Root
//! 
//! Integrates mempool data into the cross-chain settlement and MEV protection engines.

pub mod tracker;
pub mod fee_estimator;

pub use tracker::{MempoolDag, MempoolTx, TxId, MempoolDensity, FeeRate};
pub use fee_estimator::{FeeEstimator, BlockTemplate, FeeTarget, Confidence, FeeHistogramStats};

use std::sync::atomic::{AtomicU64, Ordering};

/// Mempool analytics aggregator for cross-chain settlement optimization
#[repr(C, align(64))]
pub struct MempoolAnalytics {
    /// Core mempool DAG tracker
    pub dag: MempoolDag,
    /// Fee estimator
    pub fee_estimator: FeeEstimator,
    /// Last block height processed
    last_block_height: AtomicU64,
    /// Pending settlement queue size
    pending_settlements: AtomicU64,
    /// Total value pending settlement (in sats)
    pending_value: AtomicU64,
}

impl MempoolAnalytics {
    pub fn new() -> Self {
        Self {
            dag: MempoolDag::new(),
            fee_estimator: FeeEstimator::new(),
            last_block_height: AtomicU64::new(0),
            pending_settlements: AtomicU64::new(0),
            pending_value: AtomicU64::new(0),
        }
    }
    
    /// Process new block confirmation
    pub fn on_block(&self, height: u64, txids: &[TxId], template: BlockTemplate) {
        // Remove confirmed transactions from mempool
        self.dag.remove_confirmed(txids);
        
        // Update block height
        self.dag.set_block_height(height);
        self.last_block_height.store(height, Ordering::Release);
        
        // Update fee estimator with new block template
        self.fee_estimator.add_block(template);
    }
    
    /// Add unconfirmed transaction to mempool
    pub fn add_transaction(&self, tx: MempoolTx, parents: &[TxId]) {
        self.dag.insert(tx, parents);
        self.fee_estimator.add_sample(tx.fee_rate, tx.vsize as u64);
    }
    
    /// Queue a settlement for processing
    pub fn queue_settlement(&self, value_sats: u64, urgency: u8) -> SettlementAdvice {
        self.pending_settlements.fetch_add(1, Ordering::Relaxed);
        self.pending_value.fetch_add(value_sats, Ordering::Relaxed);
        
        // Get optimal fee rate based on urgency and current mempool state
        let recommended_fee = self.fee_estimator.optimal_settlement_fee(urgency);
        let inclusion_prob = self.dag.inclusion_probability(recommended_fee);
        let congestion = self.fee_estimator.congestion_factor();
        
        SettlementAdvice {
            recommended_fee_rate: recommended_fee,
            estimated_inclusion_blocks: self.estimate_inclusion_time(recommended_fee, congestion),
            inclusion_probability,
            current_congestion: congestion,
            suggested_rbf: inclusion_prob < 0.5 && urgency >= 50,
        }
    }
    
    /// Estimate blocks until inclusion based on fee rate and congestion
    fn estimate_inclusion_time(&self, fee_rate: FeeRate, congestion: u64) -> u32 {
        let base_blocks = if fee_rate >= 100_000_000 { // 100 sat/vByte
            1
        } else if fee_rate >= 50_000_000 { // 50 sat/vByte
            2
        } else if fee_rate >= 20_000_000 { // 20 sat/vByte
            4
        } else if fee_rate >= 10_000_000 { // 10 sat/vByte
            6
        } else {
            12
        };
        
        // Adjust for congestion
        let congestion_multiplier = 1 + (congestion / 50);
        (base_blocks * congestion_multiplier) as u32
    }
    
    /// Get comprehensive mempool status
    pub fn get_status(&self) -> MempoolStatus {
        let density = self.dag.density_metrics();
        let fee_stats = self.fee_estimator.get_histogram_stats();
        
        MempoolStatus {
            tx_count: density.tx_count,
            total_vsize: density.total_vsize,
            total_fees: density.total_fees,
            avg_fee_rate: density.avg_fee_rate,
            median_fee_rate: fee_stats.median_rate,
            min_fee_rate: fee_stats.min_rate,
            max_fee_rate: fee_stats.max_rate,
            congestion_factor: self.fee_estimator.congestion_factor(),
            pending_settlements: self.pending_settlements.load(Ordering::Relaxed),
            pending_value: self.pending_value.load(Ordering::Relaxed),
            last_block_height: self.last_block_height.load(Ordering::Acquire),
        }
    }
    
    /// Detect RBF opportunities for pending settlements
    pub fn detect_rbf_opportunities(&self) -> Vec<&MempoolTx> {
        self.dag.get_rbf_candidates()
    }
    
    /// Detect CPFP opportunities for accelerating stuck transactions
    pub fn detect_cpfp_opportunities(&self, min_effective_rate: FeeRate) -> Vec<&MempoolTx> {
        self.dag.get_cpfp_opportunities(min_effective_rate)
    }
    
    /// Clear settled transactions and reset counters
    pub fn clear_settled(&self, count: u64, value: u64) {
        self.pending_settlements.fetch_sub(count.min(self.pending_settlements.load(Ordering::Relaxed)), Ordering::Relaxed);
        self.pending_value.fetch_sub(value.min(self.pending_value.load(Ordering::Relaxed)), Ordering::Relaxed);
    }
}

/// Settlement advice returned when queuing a transaction
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct SettlementAdvice {
    /// Recommended fee rate in sat/vByte (scaled by 1e6)
    pub recommended_fee_rate: FeeRate,
    /// Estimated blocks until inclusion
    pub estimated_inclusion_blocks: u32,
    /// Probability of inclusion at recommended fee
    pub inclusion_probability: f64,
    /// Current network congestion factor (0-100)
    pub current_congestion: u64,
    /// Whether RBF should be enabled
    pub suggested_rbf: bool,
}

/// Comprehensive mempool status snapshot
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct MempoolStatus {
    pub tx_count: u64,
    pub total_vsize: u64,
    pub total_fees: u64,
    pub avg_fee_rate: FeeRate,
    pub median_fee_rate: FeeRate,
    pub min_fee_rate: FeeRate,
    pub max_fee_rate: FeeRate,
    pub congestion_factor: u64,
    pub pending_settlements: u64,
    pub pending_value: u64,
    pub last_block_height: u64,
}

impl Default for MempoolAnalytics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_mempool_analytics_basic() {
        let analytics = MempoolAnalytics::new();
        
        // Add some transactions
        for i in 0..10 {
            let tx = MempoolTx {
                txid: TxId::new(i, 0),
                fee_rate: 20_000_000 + (i * 1_000_000),
                total_fee: 5000 + (i * 100),
                vsize: 250,
                ..MempoolTx::empty()
            };
            analytics.add_transaction(tx, &[]);
        }
        
        // Get status
        let status = analytics.get_status();
        assert_eq!(status.tx_count, 10);
        assert!(status.avg_fee_rate > 0);
        
        // Queue a settlement
        let advice = analytics.queue_settlement(100_000, 75);
        assert!(advice.recommended_fee_rate > 0);
        assert!(advice.inclusion_probability >= 0.0 && advice.inclusion_probability <= 1.0);
    }
    
    #[test]
    fn test_block_processing() {
        let analytics = MempoolAnalytics::new();
        
        // Add transactions
        for i in 0..20 {
            let tx = MempoolTx {
                txid: TxId::new(i, 0),
                fee_rate: 30_000_000,
                total_fee: 6000,
                vsize: 200,
                ..MempoolTx::empty()
            };
            analytics.add_transaction(tx, &[]);
        }
        
        // Process block confirming half
        let confirmed: Vec<TxId> = (0..10).map(|i| TxId::new(i, 0)).collect();
        let template = BlockTemplate {
            height: 800000,
            total_weight: 4_000_000,
            total_fees: 50_000_000,
            tx_count: 10,
            min_fee_rate: 20_000_000,
            max_fee_rate: 50_000_000,
            timestamp: 1000000,
        };
        
        analytics.on_block(800000, &confirmed, template);
        
        let status = analytics.get_status();
        assert_eq!(status.tx_count, 10); // Half should remain
        assert_eq!(status.last_block_height, 800000);
    }
}
