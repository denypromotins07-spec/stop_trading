//! MEV Module Root
//! 
//! Routes on-chain settlement orders through MEV-protected relays based on chain ID.

pub mod jito_bundles;
pub mod flashbots_rs;

pub use jito_bundles::{
    JitoBundle,
    JitoBundleBuilder,
    JitoConfig,
    BundleTransaction,
    BundleResult,
    BundleStats,
};

pub use flashbots_rs::{
    FlashbotsBundle,
    FlashbotsClient,
    FlashbotsConfig,
    PrivateTransaction,
    BundleStatus,
    FlashbotsResult,
};

/// Chain type enumeration
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ChainType {
    Solana,
    Ethereum,
    Arbitrum,
    Optimism,
    Polygon,
}

impl ChainType {
    pub fn from_chain_id(chain_id: u64) -> Self {
        match chain_id {
            1 => ChainType::Ethereum,
            42161 => ChainType::Arbitrum,
            10 => ChainType::Optimism,
            137 => ChainType::Polygon,
            _ => ChainType::Ethereum,
        }
    }
    
    pub fn is_evm(&self) -> bool {
        matches!(self, ChainType::Ethereum | ChainType::Arbitrum | ChainType::Optimism | ChainType::Polygon)
    }
}

/// MEV router selecting appropriate relay based on chain
pub struct MevRouter {
    jito_builder: Option<JitoBundleBuilder>,
    flashbots_client: Option<FlashbotsClient>,
}

impl MevRouter {
    pub fn new() -> Self {
        Self {
            jito_builder: None,
            flashbots_client: None,
        }
    }
    
    pub fn with_solana(mut self, config: JitoConfig) -> Self {
        self.jito_builder = Some(JitoBundleBuilder::new(config));
        self
    }
    
    pub fn with_evm(mut self, config: FlashbotsConfig) -> Self {
        self.flashbots_client = Some(FlashbotsClient::new(config));
        self
    }
    
    /// Route transaction through appropriate MEV protection
    pub fn route_by_chain(&self, chain: ChainType) -> Option<MevRoute> {
        match chain {
            ChainType::Solana => {
                self.jito_builder.as_ref().map(|_| MevRoute::SolanaJito)
            }
            ChainType::Ethereum | ChainType::Arbitrum | ChainType::Optimism | ChainType::Polygon => {
                self.flashbots_client.as_ref().map(|_| MevRoute::EvmFlashbots)
            }
        }
    }
}

impl Default for MevRouter {
    fn default() -> Self {
        Self::new()
    }
}

/// MEV routing decision
#[derive(Debug, Clone)]
pub enum MevRoute {
    SolanaJito,
    EvmFlashbots,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_chain_type_from_id() {
        assert_eq!(ChainType::from_chain_id(1), ChainType::Ethereum);
        assert_eq!(ChainType::from_chain_id(42161), ChainType::Arbitrum);
    }
    
    #[test]
    fn test_mev_router() {
        let router = MevRouter::new()
            .with_solana(JitoConfig::default())
            .with_evm(FlashbotsConfig::default());
        
        let solana_route = router.route_by_chain(ChainType::Solana);
        assert!(solana_route.is_some());
        
        let eth_route = router.route_by_chain(ChainType::Ethereum);
        assert!(eth_route.is_some());
    }
}
