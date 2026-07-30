//! On-Chain Data Ingestion & Normalization Module
//! 
//! Provides high-performance RPC clients for EVM and Solana networks,
//! zero-allocation event decoders, whale tracking, and network metrics.
//! Optimized for strict 6.5GB RAM constraints.

pub mod provider;
pub mod decoder;
pub mod whales;
pub mod stablecoins;
pub mod metrics;

use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, error};

/// On-chain data manager coordinating multiple RPC connections
pub struct OnChainManager {
    evm_provider: Arc<RwLock<provider::EvmRpcClient>>,
    solana_provider: Arc<RwLock<provider::SolanaRpcClient>>,
    whale_tracker: Arc<whales::WhaleClusterEngine>,
    stablecoin_monitor: Arc<stablecoins::StablecoinFlowMonitor>,
    metrics_collector: Arc<metrics::NetworkMetricsCollector>,
}

impl OnChainManager {
    /// Create a new on-chain manager with connection pooling and failover
    pub fn new(
        evm_endpoints: Vec<String>,
        solana_endpoints: Vec<String>,
        exchange_wallets: Vec<[u8; 20]>,
    ) -> Self {
        let evm_provider = Arc::new(RwLock::new(
            provider::EvmRpcClient::new(evm_endpoints)
        ));
        
        let solana_provider = Arc::new(RwLock::new(
            provider::SolanaRpcClient::new(solana_endpoints)
        ));
        
        let whale_tracker = Arc::new(whales::WhaleClusterEngine::new(exchange_wallets));
        let stablecoin_monitor = Arc::new(stablecoins::StablecoinFlowMonitor::new());
        let metrics_collector = Arc::new(metrics::NetworkMetricsCollector::new());
        
        Self {
            evm_provider,
            solana_provider,
            whale_tracker,
            stablecoin_monitor,
            metrics_collector,
        }
    }
    
    /// Start all ingestion pipelines
    pub async fn start(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        info!("Starting on-chain data ingestion pipelines");
        
        // Spawn block subscription tasks
        let evm_prov = self.evm_provider.clone();
        let whale_tracker = self.whale_tracker.clone();
        let stablecoin_monitor = self.stablecoin_monitor.clone();
        let metrics_collector = self.metrics_collector.clone();
        
        tokio::spawn(async move {
            if let Err(e) = provider::subscribe_evm_blocks(
                evm_prov,
                whale_tracker,
                stablecoin_monitor,
                metrics_collector,
            ).await {
                error!("EVM block subscription failed: {}", e);
            }
        });
        
        let sol_prov = self.solana_provider.clone();
        let whale_tracker_sol = self.whale_tracker.clone();
        let metrics_sol = self.metrics_collector.clone();
        
        tokio::spawn(async move {
            if let Err(e) = provider::subscribe_solana_blocks(
                sol_prov,
                whale_tracker_sol,
                metrics_sol,
            ).await {
                error!("Solana block subscription failed: {}", e);
            }
        });
        
        Ok(())
    }
    
    /// Get reference to whale tracker
    pub fn whale_tracker(&self) -> Arc<whales::WhaleClusterEngine> {
        self.whale_tracker.clone()
    }
    
    /// Get reference to stablecoin monitor
    pub fn stablecoin_monitor(&self) -> Arc<stablecoins::StablecoinFlowMonitor> {
        self.stablecoin_monitor.clone()
    }
    
    /// Get current network metrics
    pub fn get_metrics(&self) -> metrics::NetworkMetrics {
        self.metrics_collector.get_current_metrics()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_manager_creation() {
        let endpoints = vec!["wss://eth-mainnet.example.com".to_string()];
        let solana_endpoints = vec!["wss://solana-mainnet.example.com".to_string()];
        let exchange_wallets = vec![[0u8; 20]];
        
        let _manager = OnChainManager::new(endpoints, solana_endpoints, exchange_wallets);
    }
}
