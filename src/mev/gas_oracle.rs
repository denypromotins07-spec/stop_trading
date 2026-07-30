//! MEV Gas Oracle Module
//! 
//! Implements a predictive EIP-1559 and Solana priority fee oracle to optimize
//! transaction inclusion time. Dynamically adjusts base fees and tips based on
//! real-time mempool congestion without overpaying for block space.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Gas oracle configuration
#[derive(Debug, Clone)]
pub struct GasOracleConfig {
    /// Target inclusion time in milliseconds
    pub target_inclusion_ms: u64,
    /// Maximum base fee multiplier (cap overpayment)
    pub max_base_fee_multiplier: f64,
    /// Minimum tip in gwei
    pub min_tip_gwei: f64,
    /// Maximum tip in gwei
    pub max_tip_gwei: f64,
    /// Sample window for mempool analysis (number of blocks)
    pub sample_window_blocks: usize,
    /// Enable Solana priority fee estimation
    pub enable_solana_fees: bool,
}

impl Default for GasOracleConfig {
    fn default() -> Self {
        Self {
            target_inclusion_ms: 2000, // 2 seconds
            max_base_fee_multiplier: 2.0,
            min_tip_gwei: 0.5,
            max_tip_gwei: 500.0,
            sample_window_blocks: 20,
            enable_solana_fees: true,
        }
    }
}

/// EIP-1559 gas fee recommendation
#[derive(Debug, Clone)]
pub struct Eip1559Fee {
    pub base_fee_gwei: f64,
    pub priority_fee_gwei: f64,
    pub max_fee_gwei: f64,
    pub estimated_inclusion_ms: u64,
    pub confidence: f64,
    pub block_number: u64,
}

/// Solana priority fee recommendation
#[derive(Debug, Clone)]
pub struct SolanaPriorityFee {
    pub compute_unit_price_lamports: u64,
    pub compute_unit_limit: u32,
    pub total_fee_lamports: u64,
    pub estimated_inclusion_slots: u64,
    pub confidence: f64,
}

/// Mempool statistics for gas estimation
#[derive(Debug, Clone, Default)]
pub struct MempoolStats {
    pub pending_tx_count: usize,
    pub avg_gas_price_gwei: f64,
    pub median_gas_price_gwei: f64,
    pub p90_gas_price_gwei: f64,
    pub p99_gas_price_gwei: f64,
    pub recent_base_fees: VecDeque<f64>,
    pub recent_priority_fees: VecDeque<f64>,
}

/// Historical block data for prediction
#[derive(Debug, Clone)]
pub struct BlockHistory {
    pub block_number: u64,
    pub base_fee_gwei: f64,
    pub gas_used: u64,
    pub gas_limit: u64,
    pub timestamp: u64,
    pub tx_count: usize,
}

/// Main gas oracle service
pub struct GasOracle {
    config: GasOracleConfig,
    mempool_stats: Arc<RwLock<MempoolStats>>,
    block_history: Arc<RwLock<VecDeque<BlockHistory>>>,
    evm_fees: Arc<RwLock<Option<Eip1559Fee>>>,
    solana_fees: Arc<RwLock<Option<SolanaPriorityFee>>>,
    last_update: Instant,
}

impl GasOracle {
    /// Create a new gas oracle
    pub fn new(config: GasOracleConfig) -> Self {
        Self {
            config,
            mempool_stats: Arc::new(RwLock::new(MempoolStats {
                recent_base_fees: VecDeque::with_capacity(config.sample_window_blocks),
                recent_priority_fees: VecDeque::with_capacity(config.sample_window_blocks),
            })),
            block_history: Arc::new(RwLock::new(VecDeque::with_capacity(config.sample_window_blocks))),
            evm_fees: Arc::new(RwLock::new(None)),
            solana_fees: Arc::new(RwLock::new(None)),
            last_update: Instant::now(),
        }
    }

    /// Update mempool statistics from live data
    pub async fn update_mempool_stats(&self, stats: MempoolStats) {
        let mut current = self.mempool_stats.write().await;
        *current = stats;
        self.last_update = Instant::now();
    }

    /// Add a new block to history
    pub async fn add_block(&self, block: BlockHistory) {
        let mut history = self.block_history.write().await;
        
        if history.len() >= self.config.sample_window_blocks {
            history.pop_front();
        }
        history.push_back(block);

        // Update fee estimates after adding block
        drop(history);
        self.update_fee_estimates().await;
    }

    /// Get recommended EIP-1559 fees
    pub async fn get_evm_fees(&self) -> Option<Eip1559Fee> {
        let fees = self.evm_fees.read().await;
        fees.clone()
    }

    /// Get recommended Solana priority fees
    pub async fn get_solana_fees(&self) -> Option<SolanaPriorityFee> {
        let fees = self.solana_fees.read().await;
        fees.clone()
    }

    /// Update internal fee estimates based on current data
    async fn update_fee_estimates(&self) {
        // Update EVM fees
        let evm_fee = self.calculate_evm_fees().await;
        {
            let mut fees = self.evm_fees.write().await;
            *fees = Some(evm_fee.clone());
        }

        // Update Solana fees if enabled
        if self.config.enable_solana_fees {
            let solana_fee = self.calculate_solana_fees().await;
            let mut fees = self.solana_fees.write().await;
            *fees = Some(solana_fee);
        }
    }

    /// Calculate optimal EIP-1559 fees
    async fn calculate_evm_fees(&self) -> Eip1559Fee {
        let mempool = self.mempool_stats.read().await;
        let history = self.block_history.read().await;

        // Estimate next block's base fee using linear regression on recent blocks
        let base_fee = self.predict_base_fee(&history);

        // Calculate priority fee based on mempool congestion
        let priority_fee = self.calculate_priority_fee(&mempool);

        // Cap the max fee
        let max_fee = (base_fee * self.config.max_base_fee_multiplier)
            .min(base_fee + self.config.max_tip_gwei);

        // Estimate inclusion time based on fee competitiveness
        let inclusion_estimate = self.estimate_inclusion_time(
            base_fee + priority_fee,
            &mempool,
        );

        // Calculate confidence based on data quality
        let confidence = self.calculate_confidence(&history, &mempool);

        Eip1559Fee {
            base_fee_gwei: base_fee,
            priority_fee_gwei: priority_fee,
            max_fee_gwei: max_fee,
            estimated_inclusion_ms: inclusion_estimate,
            confidence,
            block_number: history.back().map(|b| b.block_number).unwrap_or(0) + 1,
        }
    }

    /// Predict next block's base fee using historical trend
    fn predict_base_fee(&self, history: &VecDeque<BlockHistory>) -> f64 {
        if history.is_empty() {
            return 20.0; // Default base fee
        }

        // Simple linear extrapolation
        let recent: Vec<f64> = history.iter().take(5).map(|b| b.base_fee_gwei).collect();
        
        if recent.len() < 2 {
            return recent.first().copied().unwrap_or(20.0);
        }

        // Calculate trend
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_xx = 0.0;
        let n = recent.len() as f64;

        for (i, &fee) in recent.iter().enumerate() {
            let x = i as f64;
            sum_x += x;
            sum_y += fee;
            sum_xy += x * fee;
            sum_xx += x * x;
        }

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x).max(1e-10);
        let intercept = (sum_y - slope * sum_x) / n;

        // Predict next value
        let predicted = slope * n + intercept;
        
        // Clamp to reasonable range
        predicted.clamp(1.0, 500.0)
    }

    /// Calculate priority fee based on mempool congestion
    fn calculate_priority_fee(&self, mempool: &MempoolStats) -> f64 {
        // Use percentile-based approach
        let mut priority_fee = mempool.p90_gas_price_gwei - mempool.median_gas_price_gwei;
        
        // Adjust based on pending tx count
        let congestion_factor = match mempool.pending_tx_count {
            0..=50000 => 1.0,
            50001..=100000 => 1.2,
            100001..=200000 => 1.5,
            _ => 2.0,
        };

        priority_fee *= congestion_factor;

        // Apply bounds
        priority_fee.clamp(self.config.min_tip_gwei, self.config.max_tip_gwei)
    }

    /// Estimate inclusion time for given fee
    fn estimate_inclusion_time(&self, offered_fee: f64, mempool: &MempoolStats) -> u64 {
        // Higher fee relative to mempool = faster inclusion
        let median_fee = mempool.median_gas_price_gwei;
        
        if offered_fee >= mempool.p99_gas_price_gwei {
            // Very high fee - next block
            12000 // ~1 Ethereum block time
        } else if offered_fee >= mempool.p90_gas_price_gwei {
            // High fee - within 1-2 blocks
            24000
        } else if offered_fee >= median_fee {
            // Average fee - 2-5 blocks
            60000
        } else {
            // Low fee - may take longer
            120000
        }
    }

    /// Calculate confidence in the fee estimate
    fn calculate_confidence(&self, history: &VecDeque<BlockHistory>, mempool: &MempoolStats) -> f64 {
        let mut confidence = 1.0;

        // Reduce confidence if not enough history
        if history.len() < self.config.sample_window_blocks / 2 {
            confidence *= 0.7;
        }

        // Reduce confidence if mempool is highly volatile
        if mempool.recent_priority_fees.len() >= 2 {
            let fees: Vec<f64> = mempool.recent_priority_fees.iter().copied().collect();
            let variance = self.calculate_variance(&fees);
            if variance > 100.0 {
                confidence *= 0.8;
            }
        }

        confidence
    }

    fn calculate_variance(&self, values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64
    }

    /// Calculate Solana priority fees
    async fn calculate_solana_fees(&self) -> SolanaPriorityFee {
        // Solana uses compute unit pricing
        // Typical values based on network conditions
        
        let mempool = self.mempool_stats.read().await;
        
        // Base calculation - would use actual Solana fee data in production
        let base_cu_price = 1000; // lamports per CU
        let cu_limit = 200_000; // Standard compute limit
        
        // Adjust based on congestion
        let congestion_multiplier = if mempool.pending_tx_count > 1000 {
            2.0
        } else if mempool.pending_tx_count > 500 {
            1.5
        } else {
            1.0
        };

        let cu_price = (base_cu_price as f64 * congestion_multiplier) as u64;
        let total_fee = cu_price * cu_limit as u64;

        // Estimate slots to inclusion
        let inclusion_slots = if congestion_multiplier > 1.5 {
            4
        } else if congestion_multiplier > 1.0 {
            2
        } else {
            1
        };

        SolanaPriorityFee {
            compute_unit_price_lamports: cu_price,
            compute_unit_limit: cu_limit,
            total_fee_lamports: total_fee,
            estimated_inclusion_slots: inclusion_slots,
            confidence: 0.9,
        }
    }

    /// Get fee statistics summary
    pub async fn get_fee_summary(&self) -> FeeSummary {
        let evm = self.evm_fees.read().await.clone();
        let solana = self.solana_fees.read().await.clone();
        let mempool = self.mempool_stats.read().await;

        FeeSummary {
            evm_base_fee: evm.as_ref().map(|f| f.base_fee_gwei),
            evm_priority_fee: evm.as_ref().map(|f| f.priority_fee_gwei),
            evm_max_fee: evm.as_ref().map(|f| f.max_fee_gwei),
            solana_cu_price: solana.as_ref().map(|f| f.compute_unit_price_lamports),
            solana_total_fee: solana.as_ref().map(|f| f.total_fee_lamports),
            mempool_pending: mempool.pending_tx_count,
            last_update_age_ms: self.last_update.elapsed().as_millis() as u64,
        }
    }
}

/// Summary of current fee recommendations
#[derive(Debug, Clone, Default)]
pub struct FeeSummary {
    pub evm_base_fee: Option<f64>,
    pub evm_priority_fee: Option<f64>,
    pub evm_max_fee: Option<f64>,
    pub solana_cu_price: Option<u64>,
    pub solana_total_fee: Option<u64>,
    pub mempool_pending: usize,
    pub last_update_age_ms: u64,
}

impl FeeSummary {
    pub fn print(&self) {
        println!("=== Gas Fee Summary ===");
        println!("Last Update: {}ms ago", self.last_update_age_ms);
        println!();
        println!("EVM Fees:");
        if let Some(base) = self.evm_base_fee {
            println!("  Base Fee: {:.2} gwei", base);
        }
        if let Some(priority) = self.evm_priority_fee {
            println!("  Priority Fee: {:.2} gwei", priority);
        }
        if let Some(max) = self.evm_max_fee {
            println!("  Max Fee: {:.2} gwei", max);
        }
        println!();
        println!("Solana Fees:");
        if let Some(cu) = self.solana_cu_price {
            println!("  CU Price: {} lamports", cu);
        }
        if let Some(total) = self.solana_total_fee {
            println!("  Total Fee: {} lamports", total);
        }
        println!();
        println!("Mempool: {} pending transactions", self.mempool_pending);
    }
}

/// Fast gas tracker for urgent transactions
pub struct FastGasTracker {
    oracle: Arc<GasOracle>,
    urgency_multiplier: f64,
}

impl FastGasTracker {
    pub fn new(oracle: Arc<GasOracle>, urgency_multiplier: f64) -> Self {
        Self {
            oracle,
            urgency_multiplier,
        }
    }

    /// Get expedited fee recommendation
    pub async fn get_fast_fees(&self) -> Option<Eip1559Fee> {
        let base_fees = self.oracle.get_evm_fees().await?;
        
        Some(Eip1559Fee {
            base_fee_gwei: base_fees.base_fee_gwei,
            priority_fee_gwei: base_fees.priority_fee_gwei * self.urgency_multiplier,
            max_fee_gwei: base_fees.max_fee_gwei * self.urgency_multiplier.min(3.0),
            estimated_inclusion_ms: base_fees.estimated_inclusion_ms / 2,
            confidence: base_fees.confidence * 0.9, // Slightly lower confidence for fast track
            block_number: base_fees.block_number,
        })
    }
}
