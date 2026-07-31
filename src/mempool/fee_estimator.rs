//! Real-time Fee Estimator
//! 
//! Builds a real-time fee estimator analyzing block templates and mempool queues
//! to optimize settlement costs. Dynamically adjusts sat/vByte targets to ensure
//! fast inclusion without overpaying during network congestion spikes.

use crate::mempool::tracker::{MempoolDag, MempoolDensity, FeeRate};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// Maximum number of fee buckets for histogram
const FEE_BUCKETS: usize = 256;

/// Fee bucket boundaries (in sat/vByte, scaled by 1e6)
const FEE_BUCKET_MIN: FeeRate = 1_000;      // 0.001 sat/vByte
const FEE_BUCKET_MAX: FeeRate = 1_000_000_000; // 1000 sat/vByte

/// Block template information
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct BlockTemplate {
    /// Block height
    pub height: u32,
    /// Total weight in weight units
    pub total_weight: u32,
    /// Total fees in sats
    pub total_fees: u64,
    /// Number of transactions
    pub tx_count: u32,
    /// Minimum fee rate in template (scaled by 1e6)
    pub min_fee_rate: FeeRate,
    /// Maximum fee rate in template (scaled by 1e6)
    pub max_fee_rate: FeeRate,
    /// Timestamp
    pub timestamp: u64,
}

impl BlockTemplate {
    pub const fn empty() -> Self {
        Self {
            height: 0,
            total_weight: 0,
            total_fees: 0,
            tx_count: 0,
            min_fee_rate: 0,
            max_fee_rate: 0,
            timestamp: 0,
        }
    }
    
    /// Calculate average fee rate (scaled by 1e6)
    #[inline]
    pub fn avg_fee_rate(&self) -> FeeRate {
        if self.total_weight == 0 {
            return 0;
        }
        // Convert weight to vBytes (weight / 4)
        let vbytes = self.total_weight / 4;
        (self.total_fees * 1_000_000) / vbytes as u64
    }
}

/// Fee estimate confidence levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Confidence {
    High = 0,    // ~95% probability
    Medium = 1,  // ~75% probability  
    Low = 2,     // ~50% probability
}

/// Fee target timeframes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum FeeTarget {
    NextBlock = 0,
    Within2Blocks = 1,
    Within4Blocks = 2,
    Within6Blocks = 3,
    Within10Blocks = 4,
    Within20Blocks = 5,
}

impl FeeTarget {
    pub fn to_blocks(self) -> u32 {
        match self {
            FeeTarget::NextBlock => 1,
            FeeTarget::Within2Blocks => 2,
            FeeTarget::Within4Blocks => 4,
            FeeTarget::Within6Blocks => 6,
            FeeTarget::Within10Blocks => 10,
            FeeTarget::Within20Blocks => 20,
        }
    }
}

/// Fee histogram bucket
#[repr(C, align(64))]
pub struct FeeBucket {
    /// Count of transactions in this bucket
    pub count: AtomicU64,
    /// Total value (for weighted average)
    pub total_value: AtomicU64,
    /// Lower bound of bucket (sat/vByte * 1e6)
    pub lower_bound: FeeRate,
    /// Upper bound of bucket (sat/vByte * 1e6)
    pub upper_bound: FeeRate,
}

impl FeeBucket {
    pub const fn new(lower: FeeRate, upper: FeeRate) -> Self {
        Self {
            count: AtomicU64::new(0),
            total_value: AtomicU64::new(0),
            lower_bound: lower,
            upper_bound: upper,
        }
    }
    
    pub fn reset(&self) {
        self.count.store(0, Ordering::Relaxed);
        self.total_value.store(0, Ordering::Relaxed);
    }
    
    pub fn add(&self, fee_rate: FeeRate, vsize: u64) {
        self.count.fetch_add(1, Ordering::Relaxed);
        self.total_value.fetch_add(vsize, Ordering::Relaxed);
    }
}

/// Real-time fee estimator state
#[repr(C, align(64))]
pub struct FeeEstimator {
    /// Fee histogram buckets
    buckets: [FeeBucket; FEE_BUCKETS],
    /// Recent block templates (circular buffer)
    recent_blocks: [BlockTemplate; 12],
    /// Index into recent_blocks circular buffer
    block_index: AtomicUsize,
    /// Total blocks tracked
    blocks_tracked: AtomicUsize,
    /// Current estimated feerate for next block
    next_block_estimate: AtomicU64,
    /// Network congestion factor (0-100)
    congestion_factor: AtomicU64,
    /// Last update timestamp
    last_update: AtomicU64,
}

impl FeeEstimator {
    pub fn new() -> Self {
        // Initialize buckets with logarithmic spacing
        let mut buckets = [FeeBucket::new(0, 0); FEE_BUCKETS];
        for i in 0..FEE_BUCKETS {
            let lower = Self::bucket_lower(i);
            let upper = Self::bucket_upper(i);
            buckets[i] = FeeBucket::new(lower, upper);
        }
        
        Self {
            buckets,
            recent_blocks: [BlockTemplate::empty(); 12],
            block_index: AtomicUsize::new(0),
            blocks_tracked: AtomicUsize::new(0),
            next_block_estimate: AtomicU64::new(10_000_000), // Default 10 sat/vByte
            congestion_factor: AtomicU64::new(0),
            last_update: AtomicU64::new(0),
        }
    }
    
    /// Calculate bucket lower bound using logarithmic spacing
    #[inline]
    fn bucket_lower(index: usize) -> FeeRate {
        if index == 0 {
            return FEE_BUCKET_MIN;
        }
        // Logarithmic spacing: lower = min * (max/min)^(index/buckets)
        let ratio = (FEE_BUCKET_MAX as f64 / FEE_BUCKET_MIN as f64).ln();
        let exponent = ratio * (index as f64 / FEE_BUCKETS as f64);
        (FEE_BUCKET_MIN as f64 * exponent.exp()) as FeeRate
    }
    
    /// Calculate bucket upper bound
    #[inline]
    fn bucket_upper(index: usize) -> FeeRate {
        if index >= FEE_BUCKETS - 1 {
            return FEE_BUCKET_MAX;
        }
        Self::bucket_lower(index + 1)
    }
    
    /// Find bucket index for a given fee rate
    #[inline]
    fn find_bucket(&self, fee_rate: FeeRate) -> Option<usize> {
        if fee_rate < FEE_BUCKET_MIN || fee_rate > FEE_BUCKET_MAX {
            return None;
        }
        
        // Binary search for bucket
        let mut lo = 0;
        let mut hi = FEE_BUCKETS - 1;
        
        while lo <= hi {
            let mid = (lo + hi) / 2;
            let bucket_lower = self.buckets[mid].lower_bound;
            let bucket_upper = self.buckets[mid].upper_bound;
            
            if fee_rate >= bucket_lower && fee_rate < bucket_upper {
                return Some(mid);
            } else if fee_rate < bucket_lower {
                hi = mid - 1;
            } else {
                lo = mid + 1;
            }
        }
        
        Some(FEE_BUCKETS - 1)
    }
    
    /// Add transaction sample to histogram
    pub fn add_sample(&self, fee_rate: FeeRate, vsize: u64) {
        if let Some(bucket_idx) = self.find_bucket(fee_rate) {
            self.buckets[bucket_idx].add(fee_rate, vsize);
        }
    }
    
    /// Add block template to history
    pub fn add_block(&self, template: BlockTemplate) {
        let idx = self.block_index.fetch_add(1, Ordering::AcqRel) % 12;
        self.recent_blocks[idx] = template;
        self.blocks_tracked.fetch_add(1, Ordering::Relaxed);
        self.last_update.store(template.timestamp, Ordering::Release);
        
        // Update estimates after adding block
        self.update_estimates();
    }
    
    /// Update all fee estimates based on current state
    fn update_estimates(&self) {
        let blocks = self.blocks_tracked.load(Ordering::Relaxed).min(12);
        if blocks == 0 {
            return;
        }
        
        // Calculate weighted average from recent blocks
        let mut total_weight = 0u64;
        let mut total_feerate_weighted = 0u64;
        
        let start_idx = if blocks < 12 {
            0
        } else {
            (self.block_index.load(Ordering::Relaxed) + 12 - blocks) % 12
        };
        
        for i in 0..blocks {
            let idx = (start_idx + i) % 12;
            let block = self.recent_blocks[idx];
            if block.total_weight > 0 {
                let weight = block.total_weight as u64;
                let feerate = block.avg_fee_rate();
                total_weight += weight;
                total_feerate_weighted += weight * feerate;
            }
        }
        
        if total_weight > 0 {
            let avg = total_feerate_weighted / total_weight;
            self.next_block_estimate.store(avg, Ordering::Release);
        }
        
        // Calculate congestion factor
        self.update_congestion();
    }
    
    /// Update congestion factor based on mempool pressure
    fn update_congestion(&self) {
        // Analyze fee distribution skew
        let mut high_fee_count = 0u64;
        let mut total_count = 0u64;
        
        for (i, bucket) in self.buckets.iter().enumerate() {
            let count = bucket.count.load(Ordering::Relaxed);
            total_count += count;
            
            // Count transactions above median fee rate
            if i > FEE_BUCKETS / 2 {
                high_fee_count += count;
            }
        }
        
        if total_count > 0 {
            let factor = (high_fee_count * 100) / total_count;
            self.congestion_factor.store(factor.min(100), Ordering::Release);
        }
    }
    
    /// Estimate fee rate for target confirmation within N blocks
    pub fn estimate_fee(&self, target: FeeTarget, confidence: Confidence) -> FeeRate {
        let base_estimate = self.next_block_estimate.load(Ordering::Acquire);
        let blocks = target.to_blocks();
        
        // Adjust based on target timeframe
        let timeframe_factor = match blocks {
            1 => 1.0,
            2 => 0.9,
            4 => 0.8,
            6 => 0.7,
            10 => 0.6,
            _ => 0.5,
        };
        
        // Adjust based on confidence level
        let confidence_factor = match confidence {
            Confidence::High => 1.2,
            Confidence::Medium => 1.0,
            Confidence::Low => 0.8,
        };
        
        // Adjust based on congestion
        let congestion = self.congestion_factor.load(Ordering::Acquire) as f64;
        let congestion_factor = 1.0 + (congestion / 100.0) * 0.5;
        
        let adjusted = base_estimate as f64 
            * timeframe_factor 
            * confidence_factor 
            * congestion_factor;
        
        adjusted as FeeRate
    }
    
    /// Get optimal fee rate for cost-efficient settlement
    pub fn optimal_settlement_fee(&self, urgency: u8) -> FeeRate {
        // Urgency: 0 = minimize cost, 100 = maximize speed
        if urgency >= 80 {
            self.estimate_fee(FeeTarget::NextBlock, Confidence::High)
        } else if urgency >= 50 {
            self.estimate_fee(FeeTarget::Within4Blocks, Confidence::Medium)
        } else if urgency >= 20 {
            self.estimate_fee(FeeTarget::Within10Blocks, Confidence::Medium)
        } else {
            self.estimate_fee(FeeTarget::Within20Blocks, Confidence::Low)
        }
    }
    
    /// Get fee histogram statistics
    pub fn get_histogram_stats(&self) -> FeeHistogramStats {
        let mut min_rate = FeeRate::MAX;
        let mut max_rate = 0u64;
        let mut total_txs = 0u64;
        let mut weighted_sum = 0u128;
        
        for bucket in &self.buckets {
            let count = bucket.count.load(Ordering::Relaxed);
            if count > 0 {
                min_rate = min_rate.min(bucket.lower_bound);
                max_rate = max_rate.max(bucket.upper_bound);
                total_txs += count;
                weighted_sum += (bucket.lower_bound as u128 + bucket.upper_bound as u128) / 2 * count as u128;
            }
        }
        
        let median_rate = if total_txs > 0 {
            let mut cumulative = 0u64;
            let median_pos = total_txs / 2;
            
            for bucket in &self.buckets {
                cumulative += bucket.count.load(Ordering::Relaxed);
                if cumulative >= median_pos {
                    break;
                }
            }
            (weighted_sum / total_txs as u128) as FeeRate
        } else {
            0
        };
        
        FeeHistogramStats {
            min_rate,
            max_rate,
            median_rate,
            total_transactions: total_txs,
        }
    }
    
    /// Reset all histogram buckets
    pub fn reset_histogram(&self) {
        for bucket in &self.buckets {
            bucket.reset();
        }
    }
    
    /// Get current congestion factor (0-100)
    pub fn congestion_factor(&self) -> u64 {
        self.congestion_factor.load(Ordering::Acquire)
    }
}

/// Fee histogram statistics
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct FeeHistogramStats {
    pub min_rate: FeeRate,
    pub max_rate: FeeRate,
    pub median_rate: FeeRate,
    pub total_transactions: u64,
}

impl Default for FeeEstimator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_fee_estimator_basic() {
        let estimator = FeeEstimator::new();
        
        // Add some samples
        for i in 0..100 {
            let fee_rate = 10_000_000 + (i * 100_000); // 10-20 sat/vByte
            estimator.add_sample(fee_rate, 200);
        }
        
        // Add block templates
        for i in 0..5 {
            let template = BlockTemplate {
                height: 800000 + i as u32,
                total_weight: 4_000_000,
                total_fees: 100_000_000,
                tx_count: 2000,
                min_fee_rate: 5_000_000,
                max_fee_rate: 50_000_000,
                timestamp: 1000000 + i * 600,
            };
            estimator.add_block(template);
        }
        
        // Check estimates
        let estimate = estimator.estimate_fee(FeeTarget::NextBlock, Confidence::Medium);
        assert!(estimate > 0);
        
        let stats = estimator.get_histogram_stats();
        assert!(stats.total_transactions > 0);
    }
    
    #[test]
    fn test_optimal_settlement() {
        let estimator = FeeEstimator::new();
        
        // High urgency should give higher fee
        let high_fee = estimator.optimal_settlement_fee(90);
        let low_fee = estimator.optimal_settlement_fee(10);
        
        // With no data, both might be default, but logic should differ
        assert!(high_fee >= low_fee);
    }
}
