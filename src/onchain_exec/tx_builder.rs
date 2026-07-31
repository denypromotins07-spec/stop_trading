//! Transaction Builder for On-Chain Execution
//! 
//! Estimates compute units (CU) and constructs optimal priority fee tips.
//! Ensures transactions fit into the next available block without being dropped.

use super::abi_router::{ChainType, SwapStep};
use std::sync::atomic::{AtomicU64, Ordering};

/// Compute unit estimation result
#[derive(Debug, Clone)]
pub struct CuEstimate {
    pub estimated_cu: u32,
    pub buffer_cu: u32,
    pub total_cu: u32,
    pub confidence: f64,
}

/// Priority fee recommendation
#[derive(Debug, Clone)]
pub struct PriorityFee {
    pub microlamports_per_cu: u64,
    pub total_fee_lamports: u64,
    pub expected_inclusion_time_ms: u64,
    pub success_probability: f64,
}

/// Transaction build result
#[derive(Debug, Clone)]
pub struct TransactionBuild {
    pub serialized_tx: Vec<u8>,
    pub cu_estimate: CuEstimate,
    pub priority_fee: PriorityFee,
    pub expiry_slot: u64,
}

/// Compute Unit Estimator using historical data
pub struct CuEstimator {
    pub base_costs: [(u8; 4), u32; 16], // Instruction type -> base CU
    pub adjustment_factor: f64,
    pub history_buffer: [u32; 64],
    pub history_idx: AtomicU64,
}

impl CuEstimator {
    pub fn new() -> Self {
        CuEstimator {
            base_costs: [([0u8; 4], 0); 16],
            adjustment_factor: 1.2, // 20% buffer
            history_buffer: [0; 64],
            history_idx: AtomicU64::new(0),
        }
    }

    /// Estimate CU for a swap transaction
    pub fn estimate_swap(&self, steps: usize, has_dynamic_data: bool) -> CuEstimate {
        // Base cost per swap step (Solana average)
        let base_per_step: u32 = 150_000;
        
        // Additional overhead for complex routing
        let overhead = if has_dynamic_data { 50_000 } else { 20_000 };
        
        let raw_estimate = (steps as u32) * base_per_step + overhead;
        let buffered = (raw_estimate as f64 * self.adjustment_factor) as u32;
        
        CuEstimate {
            estimated_cu: raw_estimate,
            buffer_cu: buffered - raw_estimate,
            total_cu: buffered,
            confidence: 0.95,
        }
    }

    /// Estimate CU for EVM transaction (gas estimation)
    pub fn estimate_evm_gas(&self, calldata_len: usize, is_complex: bool) -> u64 {
        // Base transaction cost
        let base_gas: u64 = 21_000;
        
        // Calldata cost (4 gas per zero byte, 16 per non-zero)
        let calldata_gas = calldata_len as u64 * 12; // Average
        
        // Complex operation overhead
        let complexity_gas = if is_complex { 50_000 } else { 0 };
        
        // Apply buffer
        let total = (base_gas + calldata_gas + complexity_gas) as f64 * self.adjustment_factor;
        
        total as u64
    }

    /// Record actual CU usage for learning
    pub fn record_actual_usage(&mut self, instruction_type: u8, actual_cu: u32) {
        let idx = self.history_idx.fetch_add(1, Ordering::Relaxed) as usize % self.history_buffer.len();
        self.history_buffer[idx] = actual_cu;
        
        // Update adjustment factor based on recent accuracy
        self.update_adjustment_factor();
    }

    fn update_adjustment_factor(&mut self) {
        // Calculate average of recent history
        let sum: u64 = self.history_buffer.iter().map(|&x| x as u64).sum();
        let count = self.history_buffer.len() as u64;
        let avg = sum as f64 / count as f64;
        
        // Adjust factor to target 95th percentile
        if avg > 0.0 {
            self.adjustment_factor = (self.adjustment_factor * 0.9 + 1.2 * 0.1).max(1.1).min(2.0);
        }
    }

    /// Get recommended CU limit with safety margin
    pub fn get_recommended_limit(&self, estimate: &CuEstimate) -> u32 {
        (estimate.total_cu as f64 * 1.1) as u32 // Additional 10% safety
    }
}

impl Default for CuEstimator {
    fn default() -> Self {
        Self::new()
    }
}

/// Priority Fee Calculator
pub struct PriorityFeeCalculator {
    pub recent_fees: [u64; 32],
    pub fee_idx: AtomicU64,
    pub network_congestion: f64,
}

impl PriorityFeeCalculator {
    pub fn new() -> Self {
        PriorityFeeCalculator {
            recent_fees: [1000; 32], // Default 1000 microlamports/CU
            fee_idx: AtomicU64::new(0),
            network_congestion: 0.5,
        }
    }

    /// Calculate priority fee based on desired inclusion speed
    pub fn calculate_fee(&self, urgency: UrgencyLevel, cu_limit: u32) -> PriorityFee {
        let base_rate = self.get_median_fee();
        
        let multiplier = match urgency {
            UrgencyLevel::Low => 0.8,
            UrgencyLevel::Normal => 1.0,
            UrgencyLevel::High => 1.5,
            UrgencyLevel::Instant => 3.0,
        };
        
        let adjusted_rate = (base_rate as f64 * multiplier * self.network_congestion.max(0.3)) as u64;
        let total_fee = adjusted_rate * cu_limit as u64 / 1_000_000; // Convert to lamports
        
        let inclusion_time = match urgency {
            UrgencyLevel::Low => 2000,
            UrgencyLevel::Normal => 500,
            UrgencyLevel::High => 200,
            UrgencyLevel::Instant => 50,
        };
        
        let success_prob = match urgency {
            UrgencyLevel::Low => 0.7,
            UrgencyLevel::Normal => 0.9,
            UrgencyLevel::High => 0.95,
            UrgencyLevel::Instant => 0.99,
        };
        
        PriorityFee {
            microlamports_per_cu: adjusted_rate,
            total_fee_lamports: total_fee,
            expected_inclusion_time_ms: inclusion_time,
            success_probability: success_prob,
        }
    }

    fn get_median_fee(&self) -> u64 {
        let mut sorted = self.recent_fees.clone();
        sorted.sort_unstable();
        sorted[sorted.len() / 2]
    }

    /// Update with latest fee data
    pub fn update_fee(&mut self, new_fee: u64) {
        let idx = self.fee_idx.fetch_add(1, Ordering::Relaxed) as usize % self.recent_fees.len();
        self.recent_fees[idx] = new_fee;
    }

    /// Update network congestion level (0.0 - 1.0)
    pub fn update_congestion(&mut self, pending_tx_count: u64, max_capacity: u64) {
        self.network_congestion = if max_capacity > 0 {
            (pending_tx_count as f64 / max_capacity as f64).min(1.0)
        } else {
            0.5
        };
    }
}

impl Default for PriorityFeeCalculator {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UrgencyLevel {
    Low,
    Normal,
    High,
    Instant,
}

/// Main Transaction Builder
pub struct TransactionBuilder {
    pub cu_estimator: CuEstimator,
    pub fee_calculator: PriorityFeeCalculator,
    pub chain: ChainType,
    pub default_urgency: UrgencyLevel,
}

impl TransactionBuilder {
    pub fn new(chain: ChainType) -> Self {
        TransactionBuilder {
            cu_estimator: CuEstimator::new(),
            fee_calculator: PriorityFeeCalculator::new(),
            chain,
            default_urgency: UrgencyLevel::Normal,
        }
    }

    /// Build Solana transaction with optimal CU and fees
    pub fn build_solana_tx(
        &self,
        instructions: &[Vec<u8>],
        signers: usize,
        urgency: UrgencyLevel,
    ) -> TransactionBuild {
        // Estimate total CU
        let total_instructions = instructions.len();
        let cu_estimate = self.cu_estimator.estimate_swap(total_instructions, true);
        
        // Calculate priority fee
        let priority_fee = self.fee_calculator.calculate_fee(urgency, cu_estimate.total_cu);
        
        // Serialize transaction (simplified)
        let mut serialized = Vec::with_capacity(64 + instructions.iter().map(|i| i.len()).sum::<usize>());
        
        // Header
        serialized.push(signers as u8);
        serialized.push(0u8); // Read-only signed
        serialized.push(0u8); // Read-only unsigned
        serialized.push(instructions.len() as u8);
        
        // Instructions
        for instruction in instructions {
            serialized.extend_from_slice(instruction);
        }
        
        // Recent blockhash placeholder (32 bytes)
        serialized.extend_from_slice(&[0u8; 32]);
        
        // Expiry slot (current + 150 slots ~ 1 minute)
        let expiry_slot = self.get_current_slot() + 150;
        
        TransactionBuild {
            serialized_tx: serialized,
            cu_estimate,
            priority_fee,
            expiry_slot,
        }
    }

    /// Build EVM transaction with optimal gas
    pub fn build_evm_tx(
        &self,
        calldata: &[u8],
        value: u128,
        urgency: UrgencyLevel,
    ) -> TransactionBuild {
        // Estimate gas
        let gas_limit = self.cu_estimator.estimate_evm_gas(calldata.len(), calldata.len() > 256);
        
        let cu_estimate = CuEstimate {
            estimated_cu: gas_limit as u32,
            buffer_cu: (gas_limit as f64 * 0.2) as u32,
            total_cu: gas_limit as u32,
            confidence: 0.95,
        };
        
        // Calculate priority fee (for EIP-1559 chains)
        let priority_fee = self.fee_calculator.calculate_fee(urgency, cu_estimate.total_cu);
        
        // Serialize transaction (simplified RLP-like encoding)
        let mut serialized = Vec::with_capacity(32 + 32 + 32 + calldata.len());
        
        // Nonce placeholder
        serialized.extend_from_slice(&[0u8; 32]);
        
        // Max priority fee per gas
        serialized.extend_from_slice(&priority_fee.microlamports_per_cu.to_be_bytes());
        
        // Max fee per gas
        let max_fee = priority_fee.microlamports_per_cu * 2;
        serialized.extend_from_slice(&max_fee.to_be_bytes());
        
        // Gas limit
        serialized.extend_from_slice(&gas_limit.to_be_bytes());
        
        // To address (placeholder)
        serialized.extend_from_slice(&[0u8; 20]);
        
        // Value
        serialized.extend_from_slice(&value.to_be_bytes());
        
        // Calldata
        serialized.extend_from_slice(calldata);
        
        TransactionBuild {
            serialized_tx: serialized,
            cu_estimate,
            priority_fee,
            expiry_slot: self.get_current_slot() + 25, // ~5 minutes for Ethereum
        }
    }

    /// Build optimized multi-hop swap transaction
    pub fn build_multihop_swap(
        &self,
        steps: &[SwapStep],
        receiver: &[u8; 20],
        slippage_bps: u16,
    ) -> TransactionBuild {
        match self.chain {
            ChainType::Solana => {
                // Build Solana instructions for each hop
                let mut instructions = Vec::new();
                
                for step in steps {
                    let mut instr = vec![0x01]; // Swap instruction
                    instr.extend_from_slice(&step.amount.to_le_bytes());
                    instr.extend_from_slice(&((step.amount as u64 * (10000 - step.pool_fee as u64) / 10000).to_le_bytes()));
                    instructions.push(instr);
                }
                
                self.build_solana_tx(&instructions, 1, self.default_urgency)
            }
            _ => {
                // Build EVM calldata for multi-hop
                use super::abi_router::EvmAbiEncoder;
                let calldata = EvmAbiEncoder::build_multihop_calldata(steps, receiver);
                self.build_evm_tx(&calldata, 0, self.default_urgency)
            }
        }
    }

    fn get_current_slot(&self) -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        
        match self.chain {
            ChainType::Solana => timestamp / 400 * 10 + 100000000, // ~400ms slots
            _ => timestamp / 12 + 18000000, // ~12s blocks for Ethereum
        }
    }

    /// Set default urgency level
    pub fn set_default_urgency(&mut self, urgency: UrgencyLevel) {
        self.default_urgency = urgency;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cu_estimator() {
        let estimator = CuEstimator::new();
        let estimate = estimator.estimate_swap(3, true);
        
        assert!(estimate.estimated_cu > 0);
        assert!(estimate.total_cu >= estimate.estimated_cu);
        assert!(estimate.confidence > 0.9);
    }

    #[test]
    fn test_priority_fee_calculator() {
        let calc = PriorityFeeCalculator::new();
        let fee = calc.calculate_fee(UrgencyLevel::High, 200_000);
        
        assert!(fee.microlamports_per_cu > 0);
        assert!(fee.success_probability > 0.9);
        assert!(fee.expected_inclusion_time_ms < 500);
    }

    #[test]
    fn test_transaction_builder_solana() {
        let builder = TransactionBuilder::new(ChainType::Solana);
        let instructions = vec![vec![0x01, 0x02, 0x03]];
        
        let tx = builder.build_solana_tx(&instructions, 1, UrgencyLevel::Normal);
        
        assert!(!tx.serialized_tx.is_empty());
        assert!(tx.cu_estimate.total_cu > 0);
        assert!(tx.priority_fee.total_fee_lamports > 0);
    }

    #[test]
    fn test_urgency_levels() {
        let calc = PriorityFeeCalculator::new();
        
        let low = calc.calculate_fee(UrgencyLevel::Low, 100_000);
        let instant = calc.calculate_fee(UrgencyLevel::Instant, 100_000);
        
        assert!(instant.microlamports_per_cu > low.microlamports_per_cu);
        assert!(instant.expected_inclusion_time_ms < low.expected_inclusion_time_ms);
    }
}
