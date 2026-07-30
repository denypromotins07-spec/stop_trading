//! MEV Module Root
//! 
//! Intercepts on-chain settlement orders and wraps them in protected payloads,
//! integrating Flashbots/Jito relays with gas optimization.

pub mod flashbots;
pub mod gas_oracle;

pub use flashbots::{
    MevProtector, FlashbotsBundle, JitoBundle, SignedTransaction, SolanaTransaction,
    BundleResult, BundleStatus, SubmissionStatus, ArbitrageOpportunity, RelayConfig,
    ProtectedTxBuilder, ChainType, TipInstruction,
};
pub use gas_oracle::{
    GasOracle, GasOracleConfig, Eip1559Fee, SolanaPriorityFee, MempoolStats,
    BlockHistory, FeeSummary, FastGasTracker,
};

use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// MEV protection configuration
#[derive(Debug, Clone)]
pub struct MevConfig {
    /// Enable MEV protection
    pub enabled: bool,
    /// Minimum profit threshold for arbitrage (in bps)
    pub min_profit_bps: f64,
    /// Maximum slippage tolerance (in bps)
    pub max_slippage_bps: u32,
    /// Enable backrunning protection
    pub enable_backrun_protection: bool,
    /// Enable sandwich attack detection
    pub enable_sandwich_detection: bool,
}

impl Default for MevConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_profit_bps: 5.0,
            max_slippage_bps: 50,
            enable_backrun_protection: true,
            enable_sandwich_detection: true,
        }
    }
}

/// Main MEV manager coordinating all protection services
pub struct MevManager {
    config: MevConfig,
    protector: Arc<RwLock<MevProtector>>,
    gas_oracle: Arc<RwLock<GasOracle>>,
}

impl MevManager {
    /// Create a new MEV manager
    pub fn new(config: MevConfig, relay_config: RelayConfig, gas_config: GasOracleConfig) -> Self {
        let protector = MevProtector::new(relay_config);
        let oracle = GasOracle::new(gas_config);

        Self {
            config,
            protector: Arc::new(RwLock::new(protector)),
            gas_oracle: Arc::new(RwLock::new(oracle)),
        }
    }

    /// Submit a protected transaction
    pub async fn submit_protected_transaction(
        &self,
        tx: SignedTransaction,
        chain: ChainType,
    ) -> anyhow::Result<BundleResult> {
        if !self.config.enabled {
            // Fall through to public mempool
            return self.submit_public_transaction(tx).await;
        }

        match chain {
            ChainType::Evm => {
                let bundle = FlashbotsBundle {
                    transactions: vec![tx],
                    block_number: 0,
                    timestamp_ns: chrono::Utc::now().timestamp_nanos() as u64,
                    min_timestamp: None,
                    max_timestamp: None,
                    reverting_tx_hashes: Vec::new(),
                    total_tip: 0,
                };

                let protector = self.protector.read().await;
                protector.submit_flashbots_bundle(bundle).await
            }
            ChainType::Solana => {
                let solana_tx = SolanaTransaction {
                    serialized: tx.raw_bytes,
                    signature: tx.hash[..32].try_into().unwrap_or([0u8; 64]),
                };

                let bundle = JitoBundle {
                    transactions: vec![solana_tx],
                    slot: 0,
                    tips: Vec::new(),
                    total_tip_lamports: 0,
                };

                let protector = self.protector.read().await;
                protector.submit_jito_bundle(bundle).await
            }
        }
    }

    /// Submit transaction to public mempool (fallback)
    async fn submit_public_transaction(&self, tx: SignedTransaction) -> anyhow::Result<BundleResult> {
        warn!("Submitting to public mempool - vulnerable to MEV");
        
        Ok(BundleResult {
            bundle_id: format!("public_{}", chrono::Utc::now().timestamp()),
            submitted_at_ns: chrono::Utc::now().timestamp_nanos() as u64,
            included: false,
            inclusion_slot: None,
            profit_eth: 0.0,
            tip_paid_eth: 0.0,
            reverted: false,
            error_message: None,
        })
    }

    /// Check if a pending transaction is being sandwiched
    pub async fn detect_sandwich_attack(
        &self,
        pending_tx_hash: &[u8; 32],
    ) -> Option<SandwichAttack> {
        if !self.config.enable_sandwich_detection {
            return None;
        }

        // In production, would analyze mempool for front-run and back-run patterns
        // This is a placeholder implementation
        
        debug!("Checking for sandwich attack on {:?}", pending_tx_hash);
        
        // Would return Some(SandwichAttack) if detected
        None
    }

    /// Find arbitrage opportunities from DEX price differences
    pub async fn find_arbitrage_opportunities(
        &self,
        quotes: &[crate::dex::aggregator::DexQuote],
    ) -> Vec<ArbitrageOpportunity> {
        let mut opportunities = Vec::new();

        // Group quotes by token pair
        let mut pairs: std::collections::HashMap<(String, String), Vec<&crate::dex::aggregator::DexQuote>> = 
            std::collections::HashMap::new();

        for quote in quotes {
            let key = (quote.token_in.clone(), quote.token_out.clone());
            pairs.entry(key).or_default().push(quote);
        }

        // Look for price discrepancies
        for (key, pair_quotes) in pairs.iter() {
            if pair_quotes.len() < 2 {
                continue;
            }

            // Sort by output amount
            let mut sorted: Vec<_> = pair_quotes.iter().collect();
            sorted.sort_by(|a, b| b.amount_out.partial_cmp(&a.amount_out).unwrap_or(std::cmp::Ordering::Equal));

            // Check for profitable arbitrage
            for i in 0..sorted.len() {
                for j in (i + 1)..sorted.len() {
                    let buy_quote = sorted[j]; // Lower output = worse price = buy here
                    let sell_quote = sorted[i]; // Higher output = better price = sell here

                    // Calculate potential profit
                    let input_amount = buy_quote.amount_in;
                    let output_on_buy = buy_quote.amount_out;
                    
                    // Simulate selling on other venue
                    let profit_ratio = (sell_quote.amount_out / sell_quote.amount_in) 
                                     - (buy_quote.amount_out / buy_quote.amount_in);
                    
                    let profit_bps = profit_ratio * 10000.0;

                    if profit_bps > self.config.min_profit_bps {
                        opportunities.push(ArbitrageOpportunity {
                            route: vec![buy_quote.pool_id.clone(), sell_quote.pool_id.clone()],
                            input_token: key.0.clone(),
                            output_token: key.1.clone(),
                            input_amount_wei: (input_amount * 1e18) as u128,
                            expected_output_wei: (output_on_buy * 1e18) as u128,
                            expected_profit_wei: ((profit_ratio * input_amount) * 1e18) as u128,
                            profit_bps,
                            router_address: [0u8; 20], // Would be actual router
                            confidence: 0.8,
                        });
                    }
                }
            }
        }

        opportunities
    }

    /// Execute arbitrage bundle if profitable
    pub async fn execute_arbitrage(
        &self,
        opportunity: &ArbitrageOpportunity,
    ) -> anyhow::Result<Option<BundleResult>> {
        if opportunity.profit_bps < self.config.min_profit_bps {
            return Ok(None);
        }

        // Get current gas fees
        let gas_oracle = self.gas_oracle.read().await;
        let evm_fees = gas_oracle.get_evm_fees().await;
        drop(gas_oracle);

        // Build arbitrage bundle
        let protector = self.protector.read().await;
        
        let base_fee = evm_fees.map(|f| f.base_fee_gwei as u64).unwrap_or(20_000_000_000);
        
        if let Some(bundle) = protector.build_arbitrage_bundle(&[opportunity.clone()], base_fee) {
            drop(protector);
            
            let protector = self.protector.read().await;
            let result = protector.submit_flashbots_bundle(bundle).await?;
            
            if result.profit_eth > 0.0 {
                info!(
                    "Arbitrage executed: profit {:.4} ETH, tip {:.4} ETH",
                    result.profit_eth, result.tip_paid_eth
                );
                return Ok(Some(result));
            }
        }

        Ok(None)
    }

    /// Get current gas fee recommendations
    pub async fn get_gas_recommendation(&self) -> Option<Eip1559Fee> {
        let oracle = self.gas_oracle.read().await;
        oracle.get_evm_fees().await
    }

    /// Update mempool statistics
    pub async fn update_mempool_stats(&self, stats: MempoolStats) {
        let oracle = self.gas_oracle.read().await;
        oracle.update_mempool_stats(stats).await;
    }
}

/// Detected sandwich attack pattern
#[derive(Debug, Clone)]
pub struct SandwichAttack {
    pub victim_tx_hash: [u8; 32],
    pub frontrun_tx_hash: Option<[u8; 32]>,
    pub backrun_tx_hash: Option<[u8; 32]>,
    pub estimated_victim_loss_eth: f64,
    pub attacker_profit_eth: f64,
    pub confidence: f64,
}

/// MEV statistics and metrics
#[derive(Debug, Clone, Default)]
pub struct MevStats {
    pub bundles_submitted: usize,
    pub bundles_included: usize,
    pub total_profit_eth: f64,
    pub total_tips_paid_eth: f64,
    pub sandwich_attacks_prevented: usize,
    pub arbitrage_opportunities_found: usize,
    pub arbitrage_executed: usize,
}

impl MevStats {
    pub fn print(&self) {
        println!("=== MEV Statistics ===");
        println!("Bundles Submitted: {}", self.bundles_submitted);
        println!("Bundles Included: {}", self.bundles_included);
        println!("Success Rate: {:.1}%", 
            if self.bundles_submitted > 0 {
                (self.bundles_included as f64 / self.bundles_submitted as f64) * 100.0
            } else { 0.0 }
        );
        println!("Total Profit: {:.4} ETH", self.total_profit_eth);
        println!("Total Tips Paid: {:.4} ETH", self.total_tips_paid_eth);
        println!("Net Profit: {:.4} ETH", self.total_profit_eth - self.total_tips_paid_eth);
        println!();
        println!("Sandwich Attacks Prevented: {}", self.sandwich_attacks_prevented);
        println!("Arbitrage Opportunities: {}", self.arbitrage_opportunities_found);
        println!("Arbitrage Executed: {}", self.arbitrage_executed);
    }
}
