//! DeFi Module Root
//! 
//! Integrates on-chain LP state tracking with the centralized portfolio state manager.

pub mod impermanent;
pub mod yield_optimizer;

pub use impermanent::{
    ConcentratedPosition, FixedPointU64, ImpermanentLossCalculator, ImpermanentLossResult,
    PortfolioIlTracker, tick_math,
};
pub use yield_optimizer::{
    EmissionRewards, PoolState, RebalanceRecommendation, RebalanceReason, TvlDecayAnalysis,
    YieldMetrics, YieldOptimizer,
};

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::sync::Arc;

/// Unified DeFi position state combining CEX and DEX positions
#[derive(Debug, Clone)]
pub struct DefiPosition {
    pub position_id: u64,
    pub pool_address: [u8; 32],
    pub chain: ChainType,
    pub token0_amount: FixedPointU64,
    pub token1_amount: FixedPointU64,
    pub entry_tick_lower: i32,
    pub entry_tick_upper: i32,
    pub current_price_tick: i32,
    pub accrued_fees_token0: FixedPointU64,
    pub accrued_fees_token1: FixedPointU64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChainType {
    Ethereum,
    Solana,
    Arbitrum,
    Optimism,
    Polygon,
}

impl ChainType {
    pub fn as_str(&self) -> &'static str {
        match self {
            ChainType::Ethereum => "ethereum",
            ChainType::Solana => "solana",
            ChainType::Arbitrum => "arbitrum",
            ChainType::Optimism => "optimism",
            ChainType::Polygon => "polygon",
        }
    }
}

/// DeFi portfolio aggregator
pub struct DefiPortfolio {
    pub positions: Vec<DefiPosition>,
    pub il_tracker: PortfolioIlTracker,
    pub yield_optimizer: YieldOptimizer,
    pub total_value_usd: FixedPointU64,
    pub total_accrued_fees_usd: FixedPointU64,
    pub net_apy: FixedPointU64,
    pub update_counter: AtomicU64,
}

impl DefiPortfolio {
    pub fn new() -> Self {
        DefiPortfolio {
            positions: Vec::with_capacity(64),
            il_tracker: PortfolioIlTracker::new(),
            yield_optimizer: YieldOptimizer::new(),
            total_value_usd: FixedPointU64(0),
            total_accrued_fees_usd: FixedPointU64(0),
            net_apy: FixedPointU64(0),
            update_counter: AtomicU64::new(0),
        }
    }

    /// Add a new DeFi position
    pub fn add_position(&mut self, position: DefiPosition) {
        if self.positions.len() < self.positions.capacity() {
            // Also add to IL tracker
            let lp_position = ConcentratedPosition {
                lower_tick: position.entry_tick_lower,
                upper_tick: position.entry_tick_upper,
                liquidity: ((position.token0_amount.to_f64() * position.token1_amount.to_f64()).sqrt() * 1_000_000.0) as u128,
                token0_amount: position.token0_amount,
                token1_amount: position.token1_amount,
                entry_price_ratio: FixedPointU64::from_f64(
                    tick_math::tick_to_price((position.entry_tick_lower + position.entry_tick_upper) / 2)
                ),
            };
            self.il_tracker.add_position(lp_position);
            self.positions.push(position);
        }
    }

    /// Update all positions with current prices and calculate metrics
    pub fn update_metrics(&mut self, token_prices: &[( [u8; 32], FixedPointU64 )]) {
        self.update_counter.fetch_add(1, Ordering::Relaxed);
        
        self.total_value_usd = FixedPointU64(0);
        self.total_accrued_fees_usd = FixedPointU64(0);

        let mut total_net_return = 0.0;
        let mut position_count = 0;

        for position in &mut self.positions {
            // Get current prices
            let price0 = token_prices.iter()
                .find(|(addr, _)| *addr == position.pool_address)
                .map(|(_, p)| *p)
                .unwrap_or(FixedPointU64::from_f64(1.0));
            
            let price1 = price0; // Simplified - would be different token

            // Calculate position value
            let value0 = position.token0_amount.to_f64() * price0.to_f64();
            let value1 = position.token1_amount.to_f64() * price1.to_f64();
            let position_value = FixedPointU64::from_f64(value0 + value1);
            
            self.total_value_usd = self.total_value_usd.checked_add(position_value).unwrap_or(self.total_value_usd);

            // Calculate accrued fees value
            let fees_value = position.accrued_fees_token0.to_f64() * price0.to_f64()
                + position.accrued_fees_token1.to_f64() * price1.to_f64();
            let fees_fp = FixedPointU64::from_f64(fees_value);
            self.total_accrued_fees_usd = self.total_accrued_fees_usd.checked_add(fees_fp).unwrap_or(self.total_accrued_fees_usd);

            // Update current price tick
            position.current_price_tick = tick_math::price_to_tick(price0.to_f64());

            position_count += 1;
        }

        // Calculate net APY (simplified)
        if self.total_value_usd.0 > 0 && position_count > 0 {
            let fee_yield = self.total_accrued_fees_usd.to_f64() / self.total_value_usd.to_f64();
            // Annualize (assuming daily update)
            let annualized = fee_yield * 365.0;
            self.net_apy = FixedPointU64::from_f64(annualized);
        }
    }

    /// Get rebalance recommendations for all positions
    pub fn get_rebalance_recommendations(&self) -> Vec<(usize, RebalanceRecommendation)> {
        let mut recommendations = Vec::new();

        for (idx, position) in self.positions.iter().enumerate() {
            // Create dummy pool state for recommendation
            let pool = PoolState {
                pool_id: position.pool_address,
                token0: [0; 8],
                token1: [0; 8],
                tvl_usd: FixedPointU64::from_f64(1_000_000.0),
                volume_24h_usd: FixedPointU64::from_f64(100_000.0),
                fee_tier: FixedPointU64::from_f64(0.003),
                current_tick: position.current_price_tick,
                liquidity: 10_000_000,
            };

            let lp_position = ConcentratedPosition {
                lower_tick: position.entry_tick_lower,
                upper_tick: position.entry_tick_upper,
                liquidity: 100_000,
                token0_amount: position.token0_amount,
                token1_amount: position.token1_amount,
                entry_price_ratio: FixedPointU64::from_f64(
                    tick_math::tick_to_price((position.entry_tick_lower + position.entry_tick_upper) / 2)
                ),
            };

            let metrics = YieldMetrics {
                base_apr: FixedPointU64::from_f64(0.1),
                emission_apr: FixedPointU64(0),
                total_apr: FixedPointU64::from_f64(0.1),
                impermanent_loss_estimate: FixedPointU64(0),
                net_apr: FixedPointU64::from_f64(0.1),
                sharpe_ratio: FixedPointU64(0),
            };

            let rec = self.yield_optimizer.recommend_rebalance(&pool, &lp_position, &metrics);
            
            if rec.should_rebalance {
                recommendations.push((idx, rec));
            }
        }

        recommendations
    }

    /// Execute rebalance for a specific position
    pub fn execute_rebalance(&mut self, position_idx: usize, new_lower: i32, new_upper: i32) -> bool {
        if position_idx >= self.positions.len() {
            return false;
        }

        let position = &mut self.positions[position_idx];
        position.entry_tick_lower = new_lower;
        position.entry_tick_upper = new_upper;

        true
    }

    /// Get portfolio summary
    pub fn get_summary(&self) -> DefiPortfolioSummary {
        DefiPortfolioSummary {
            total_value_usd: self.total_value_usd.to_f64(),
            total_fees_usd: self.total_accrued_fees_usd.to_f64(),
            net_apy_pct: self.net_apy.to_f64() * 100.0,
            position_count: self.positions.len(),
            chains: self.get_chain_breakdown(),
        }
    }

    fn get_chain_breakdown(&self) -> Vec<(ChainType, f64)> {
        let mut breakdown = vec![
            (ChainType::Ethereum, 0.0),
            (ChainType::Solana, 0.0),
            (ChainType::Arbitrum, 0.0),
            (ChainType::Optimism, 0.0),
            (ChainType::Polygon, 0.0),
        ];

        for position in &self.positions {
            let value = position.token0_amount.to_f64() + position.token1_amount.to_f64();
            match position.chain {
                ChainType::Ethereum => breakdown[0].1 += value,
                ChainType::Solana => breakdown[1].1 += value,
                ChainType::Arbitrum => breakdown[2].1 += value,
                ChainType::Optimism => breakdown[3].1 += value,
                ChainType::Polygon => breakdown[4].1 += value,
            }
        }

        breakdown
    }
}

impl Default for DefiPortfolio {
    fn default() -> Self {
        Self::new()
    }
}

/// Portfolio summary for dashboards
#[derive(Debug, Clone)]
pub struct DefiPortfolioSummary {
    pub total_value_usd: f64,
    pub total_fees_usd: f64,
    pub net_apy_pct: f64,
    pub position_count: usize,
    pub chains: Vec<(ChainType, f64)>,
}

/// Integration with centralized portfolio manager
pub struct CentralizedDefiBridge {
    pub defi_portfolio: DefiPortfolio,
    pub sync_enabled: AtomicBool,
    pub last_sync_timestamp: AtomicU64,
}

impl CentralizedDefiBridge {
    pub fn new() -> Self {
        CentralizedDefiBridge {
            defi_portfolio: DefiPortfolio::new(),
            sync_enabled: AtomicBool::new(true),
            last_sync_timestamp: AtomicU64::new(0),
        }
    }

    /// Sync DeFi positions with centralized portfolio state
    pub fn sync_with_centralized(&self, cex_positions: &[u64]) -> SyncResult {
        if !self.sync_enabled.load(Ordering::Relaxed) {
            return SyncResult {
                success: false,
                reason: "Sync disabled",
                positions_synced: 0,
            };
        }

        // In production, this would fetch actual on-chain data
        let positions_synced = self.defi_portfolio.positions.len();
        
        use std::time::{SystemTime, UNIX_EPOCH};
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        
        self.last_sync_timestamp.store(timestamp, Ordering::Relaxed);

        SyncResult {
            success: true,
            reason: "",
            positions_synced,
        }
    }

    /// Enable/disable syncing
    pub fn set_sync_enabled(&self, enabled: bool) {
        self.sync_enabled.store(enabled, Ordering::Relaxed);
    }
}

#[derive(Debug, Clone)]
pub struct SyncResult {
    pub success: bool,
    pub reason: &'static str,
    pub positions_synced: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_defi_portfolio_creation() {
        let portfolio = DefiPortfolio::new();
        assert_eq!(portfolio.positions.len(), 0);
        assert_eq!(portfolio.total_value_usd.0, 0);
    }

    #[test]
    fn test_add_position() {
        let mut portfolio = DefiPortfolio::new();
        
        let position = DefiPosition {
            position_id: 1,
            pool_address: [1; 32],
            chain: ChainType::Ethereum,
            token0_amount: FixedPointU64::from_f64(1000.0),
            token1_amount: FixedPointU64::from_f64(1000.0),
            entry_tick_lower: -1000,
            entry_tick_upper: 1000,
            current_price_tick: 0,
            accrued_fees_token0: FixedPointU64(0),
            accrued_fees_token1: FixedPointU64(0),
        };

        portfolio.add_position(position);
        assert_eq!(portfolio.positions.len(), 1);
    }

    #[test]
    fn test_chain_type_conversion() {
        assert_eq!(ChainType::Ethereum.as_str(), "ethereum");
        assert_eq!(ChainType::Solana.as_str(), "solana");
    }

    #[test]
    fn test_portfolio_summary() {
        let mut portfolio = DefiPortfolio::new();
        
        let position = DefiPosition {
            position_id: 1,
            pool_address: [1; 32],
            chain: ChainType::Ethereum,
            token0_amount: FixedPointU64::from_f64(1000.0),
            token1_amount: FixedPointU64::from_f64(1000.0),
            entry_tick_lower: -1000,
            entry_tick_upper: 1000,
            current_price_tick: 0,
            accrued_fees_token0: FixedPointU64(0),
            accrued_fees_token1: FixedPointU64(0),
        };

        portfolio.add_position(position);
        let summary = portfolio.get_summary();
        
        assert_eq!(summary.position_count, 1);
        assert!(summary.chains.iter().any(|(c, _)| *c == ChainType::Ethereum));
    }
}
