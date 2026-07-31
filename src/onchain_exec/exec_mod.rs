//! On-Chain Execution Module Root
//! 
//! Routes decentralized trades through MEV-protected relays.

pub mod abi_router;
pub mod tx_builder;

pub use abi_router::{
    AbiTemplate, ChainType, EvmAbiEncoder, Pod, RouteBuilder, SolanaInstructionBuilder,
    SwapStep, TemplateCache,
};
pub use tx_builder::{
    CuEstimate, CuEstimator, PriorityFee, PriorityFeeCalculator, TransactionBuild,
    TransactionBuilder, UrgencyLevel,
};

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::sync::Arc;

/// MEV Relay configuration
#[derive(Debug, Clone)]
pub struct MevRelayConfig {
    pub relay_url: String,
    pub auth_token: Option<String>,
    pub priority_level: u8,
    pub max_slippage_bps: u16,
}

impl MevRelayConfig {
    pub fn new(relay_url: &str) -> Self {
        MevRelayConfig {
            relay_url: relay_url.to_string(),
            auth_token: None,
            priority_level: 50,
            max_slippage_bps: 50,
        }
    }

    pub fn with_auth(mut self, token: &str) -> Self {
        self.auth_token = Some(token.to_string());
        self
    }

    pub fn with_priority(mut self, level: u8) -> Self {
        self.priority_level = level.min(100);
        self
    }

    pub fn with_slippage(mut self, bps: u16) -> Self {
        self.max_slippage_bps = bps;
        self
    }
}

/// MEV-Protected transaction submission result
#[derive(Debug, Clone)]
pub struct MevSubmissionResult {
    pub success: bool,
    pub tx_hash: Option<[u8; 32]>,
    pub bundle_id: Option<u64>,
    pub estimated_inclusion_block: u64,
    pub protection_level: ProtectionLevel,
    pub error_message: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProtectionLevel {
    None,
    Basic,      // Standard relay
    Premium,    // Priority routing
    Exclusive,  // Private order flow
}

/// MEV Relay Manager
pub struct MevRelayManager {
    pub relays: Vec<MevRelayConfig>,
    pub active_relay_idx: usize,
    pub submission_counter: AtomicU64,
    pub failed_submissions: AtomicU64,
    pub circuit_breaker: AtomicBool,
}

impl MevRelayManager {
    pub fn new() -> Self {
        MevRelayManager {
            relays: Vec::with_capacity(8),
            active_relay_idx: 0,
            submission_counter: AtomicU64::new(0),
            failed_submissions: AtomicU64::new(0),
            circuit_breaker: AtomicBool::new(false),
        }
    }

    /// Add a relay configuration
    pub fn add_relay(&mut self, config: MevRelayConfig) {
        if self.relays.len() < self.relays.capacity() {
            self.relays.push(config);
        }
    }

    /// Submit transaction through MEV-protected relay
    pub fn submit_mev_protected(&self, tx_build: &TransactionBuild, protection: ProtectionLevel) -> MevSubmissionResult {
        if self.circuit_breaker.load(Ordering::Relaxed) {
            return MevSubmissionResult {
                success: false,
                tx_hash: None,
                bundle_id: None,
                estimated_inclusion_block: 0,
                protection_level: protection,
                error_message: Some("Circuit breaker active".to_string()),
            };
        }

        self.submission_counter.fetch_add(1, Ordering::Relaxed);

        if self.relays.is_empty() {
            self.failed_submissions.fetch_add(1, Ordering::Relaxed);
            return MevSubmissionResult {
                success: false,
                tx_hash: None,
                bundle_id: None,
                estimated_inclusion_block: 0,
                protection_level: protection,
                error_message: Some("No relays configured".to_string()),
            };
        }

        // Select relay based on protection level
        let relay_idx = self.select_relay_for_protection(protection);
        let relay = &self.relays[relay_idx];

        // Simulate submission (in production, would make HTTP request)
        let success = true;
        let tx_hash = Some(self.generate_tx_hash(&tx_build.serialized_tx));

        MevSubmissionResult {
            success,
            tx_hash,
            bundle_id: if protection != ProtectionLevel::None { Some(self.submission_counter.load(Ordering::Relaxed)) } else { None },
            estimated_inclusion_block: tx_build.expiry_slot,
            protection_level: protection,
            error_message: None,
        }
    }

    fn select_relay_for_protection(&self, protection: ProtectionLevel) -> usize {
        match protection {
            ProtectionLevel::None => 0,
            ProtectionLevel::Basic => self.active_relay_idx,
            ProtectionLevel::Premium => {
                // Select relay with highest priority
                self.relays.iter()
                    .enumerate()
                    .max_by_key(|(_, r)| r.priority_level)
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            }
            ProtectionLevel::Exclusive => {
                // Use dedicated exclusive relay if available
                self.relays.iter()
                    .position(|r| r.priority_level >= 90)
                    .unwrap_or(0)
            }
        }
    }

    fn generate_tx_hash(&self, data: &[u8]) -> [u8; 32] {
        // Simplified hash - in production would use actual SHA256
        let mut hash = [0u8; 32];
        for (i, &byte) in data.iter().enumerate() {
            hash[i % 32] ^= byte;
        }
        hash
    }

    /// Rotate to next relay (for failover)
    pub fn rotate_relay(&mut self) {
        if !self.relays.is_empty() {
            self.active_relay_idx = (self.active_relay_idx + 1) % self.relays.len();
        }
    }

    /// Trigger circuit breaker
    pub fn trigger_circuit_breaker(&self) {
        self.circuit_breaker.store(true, Ordering::Relaxed);
    }

    /// Reset circuit breaker
    pub fn reset_circuit_breaker(&self) {
        self.circuit_breaker.store(false, Ordering::Relaxed);
    }

    /// Get relay statistics
    pub fn get_stats(&self) -> RelayStats {
        RelayStats {
            total_submissions: self.submission_counter.load(Ordering::Relaxed),
            failed_submissions: self.failed_submissions.load(Ordering::Relaxed),
            relay_count: self.relays.len(),
            circuit_breaker_active: self.circuit_breaker.load(Ordering::Relaxed),
        }
    }
}

impl Default for MevRelayManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Relay statistics
#[derive(Debug, Clone)]
pub struct RelayStats {
    pub total_submissions: u64,
    pub failed_submissions: u64,
    pub relay_count: usize,
    pub circuit_breaker_active: bool,
}

impl RelayStats {
    pub fn success_rate(&self) -> f64 {
        if self.total_submissions == 0 {
            return 1.0;
        }
        1.0 - (self.failed_submissions as f64 / self.total_submissions as f64)
    }
}

/// On-chain execution router combining all components
pub struct OnChainExecutionRouter {
    pub tx_builder: TransactionBuilder,
    pub mev_manager: MevRelayManager,
    pub chain: ChainType,
    pub execution_enabled: AtomicBool,
}

impl OnChainExecutionRouter {
    pub fn new(chain: ChainType) -> Self {
        OnChainExecutionRouter {
            tx_builder: TransactionBuilder::new(chain),
            mev_manager: MevRelayManager::new(),
            chain,
            execution_enabled: AtomicBool::new(true),
        }
    }

    /// Execute a swap with MEV protection
    pub fn execute_swap(
        &self,
        steps: &[SwapStep],
        receiver: &[u8; 20],
        slippage_bps: u16,
        urgency: UrgencyLevel,
        mev_protection: ProtectionLevel,
    ) -> ExecutionResult {
        if !self.execution_enabled.load(Ordering::Relaxed) {
            return ExecutionResult {
                success: false,
                error: Some("Execution disabled".to_string()),
                tx_hash: None,
            };
        }

        // Build transaction
        let tx_build = self.tx_builder.build_multihop_swap(steps, receiver, slippage_bps);

        // Submit through MEV relay
        let submission = self.mev_manager.submit_mev_protected(&tx_build, mev_protection);

        if submission.success {
            ExecutionResult {
                success: true,
                error: None,
                tx_hash: submission.tx_hash,
            }
        } else {
            ExecutionResult {
                success: false,
                error: submission.error_message,
                tx_hash: None,
            }
        }
    }

    /// Enable/disable execution
    pub fn set_execution_enabled(&self, enabled: bool) {
        self.execution_enabled.store(enabled, Ordering::Relaxed);
    }

    /// Get current chain
    pub fn get_chain(&self) -> ChainType {
        self.chain
    }
}

/// Execution result
#[derive(Debug, Clone)]
pub struct ExecutionResult {
    pub success: bool,
    pub error: Option<String>,
    pub tx_hash: Option<[u8; 32]>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mev_relay_config() {
        let config = MevRelayConfig::new("https://relay.example.com")
            .with_auth("token123")
            .with_priority(80)
            .with_slippage(30);

        assert_eq!(config.relay_url, "https://relay.example.com");
        assert!(config.auth_token.is_some());
        assert_eq!(config.priority_level, 80);
        assert_eq!(config.max_slippage_bps, 30);
    }

    #[test]
    fn test_mev_relay_manager() {
        let mut manager = MevRelayManager::new();
        
        manager.add_relay(MevRelayConfig::new("https://relay1.com"));
        manager.add_relay(MevRelayConfig::new("https://relay2.com").with_priority(90));

        let stats = manager.get_stats();
        assert_eq!(stats.relay_count, 2);
        assert!(!stats.circuit_breaker_active);
    }

    #[test]
    fn test_onchain_router_creation() {
        let router = OnChainExecutionRouter::new(ChainType::Ethereum);
        assert_eq!(router.get_chain(), ChainType::Ethereum);
    }

    #[test]
    fn test_relay_stats_success_rate() {
        let stats = RelayStats {
            total_submissions: 100,
            failed_submissions: 5,
            relay_count: 3,
            circuit_breaker_active: false,
        };

        assert!((stats.success_rate() - 0.95).abs() < 0.001);
    }
}
