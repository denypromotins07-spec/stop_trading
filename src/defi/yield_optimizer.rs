//! DeFi Yield Optimizer
//! 
//! Implements yield optimization engine analyzing APR, token emissions, and TVL decay.
//! Dynamically rebalances LP ranges to maintain optimal delta exposure and maximize risk-adjusted yields.

use super::impermanent::{ConcentratedPosition, FixedPointU64, ImpermanentLossCalculator, tick_math};
use std::sync::atomic::{AtomicU64, Ordering};

/// Pool state for yield analysis
#[derive(Debug, Clone)]
pub struct PoolState {
    pub pool_id: [u8; 32],
    pub token0: [u8; 8],
    pub token1: [u8; 8],
    pub tvl_usd: FixedPointU64,
    pub volume_24h_usd: FixedPointU64,
    pub fee_tier: FixedPointU64, // 0.0001, 0.0005, 0.003, 0.01
    pub current_tick: i32,
    pub liquidity: u128,
}

/// Emission rewards for a pool
#[derive(Debug, Clone)]
pub struct EmissionRewards {
    pub token_address: [u8; 32],
    pub tokens_per_day: FixedPointU64,
    pub token_price_usd: FixedPointU64,
}

/// Yield metrics for a pool/position
#[derive(Debug, Clone)]
pub struct YieldMetrics {
    pub base_apr: FixedPointU64,      // From trading fees
    pub emission_apr: FixedPointU64,  // From token rewards
    pub total_apr: FixedPointU64,     // Combined APR
    pub impermanent_loss_estimate: FixedPointU64,
    pub net_apr: FixedPointU64,       // Total APR - IL estimate
    pub sharpe_ratio: FixedPointU64,
}

/// Rebalance recommendation
#[derive(Debug, Clone)]
pub struct RebalanceRecommendation {
    pub should_rebalance: bool,
    pub reason: Option<RebalanceReason>,
    pub new_lower_tick: Option<i32>,
    pub new_upper_tick: Option<i32>,
    pub expected_improvement_bps: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RebalanceReason {
    PriceOutOfRange,
    SuboptimalRange,
    LowCapitalEfficiency,
    HighImpermanentLoss,
    BetterOpportunity,
}

/// Main Yield Optimizer Engine
pub struct YieldOptimizer {
    pub calculator: ImpermanentLossCalculator,
    pub optimization_counter: AtomicU64,
    pub min_rebalance_improvement_bps: u32, // Minimum improvement to trigger rebalance
}

impl YieldOptimizer {
    pub fn new() -> Self {
        YieldOptimizer {
            calculator: ImpermanentLossCalculator::new(),
            optimization_counter: AtomicU64::new(0),
            min_rebalance_improvement_bps: 50, // 0.5% minimum improvement
        }
    }

    /// Calculate comprehensive yield metrics for a pool
    pub fn calculate_yield_metrics(
        &self,
        pool: &PoolState,
        position: &ConcentratedPosition,
        emissions: &[EmissionRewards],
    ) -> YieldMetrics {
        self.optimization_counter.fetch_add(1, Ordering::Relaxed);

        // Calculate base APR from fees
        let base_apr = self.calculate_base_apr(pool, position);

        // Calculate emission APR
        let emission_apr = self.calculate_emission_apr(pool, emissions, position);

        // Total APR
        let total_apr = base_apr.checked_add(emission_apr).unwrap_or(FixedPointU64(0));

        // Estimate IL based on historical volatility
        let il_estimate = self.estimate_il_for_pool(pool, position);

        // Net APR = Total APR - IL
        let net_apr = if total_apr.0 > il_estimate.0 {
            total_apr.checked_sub(il_estimate).unwrap_or(FixedPointU64(0))
        } else {
            FixedPointU64(0)
        };

        // Simplified Sharpe ratio (would need more data in production)
        let sharpe_ratio = self.calculate_sharpe_approximation(&net_apr, &il_estimate);

        YieldMetrics {
            base_apr,
            emission_apr,
            total_apr,
            impermanent_loss_estimate: il_estimate,
            net_apr,
            sharpe_ratio,
        }
    }

    /// Calculate base APR from trading fees
    fn calculate_base_apr(&self, pool: &PoolState, position: &ConcentratedPosition) -> FixedPointU64 {
        // Annualized fee revenue / Position value
        let daily_fees = pool.volume_24h_usd.checked_mul(pool.fee_tier).unwrap_or(FixedPointU64(0));
        let annual_fees = FixedPointU64::from_f64(daily_fees.to_f64() * 365.0);

        // Position value estimate
        let position_value = position.token0_amount.checked_mul(FixedPointU64::from_f64(1.0))
            .unwrap_or(FixedPointU64(0))
            .checked_add(position.token1_amount)
            .unwrap_or(FixedPointU64(0));

        if position_value.0 == 0 {
            return FixedPointU64(0);
        }

        // Capital efficiency multiplier for concentrated positions
        let capital_efficiency = self.calculate_capital_efficiency(position, pool.current_tick);
        
        let base_apr_raw = annual_fees.to_f64() / position_value.to_f64();
        let enhanced_apr = base_apr_raw * capital_efficiency;

        FixedPointU64::from_f64(enhanced_apr)
    }

    /// Calculate APR from token emissions
    fn calculate_emission_apr(
        &self,
        pool: &PoolState,
        emissions: &[EmissionRewards],
        position: &ConcentratedPosition,
    ) -> FixedPointU64 {
        if emissions.is_empty() {
            return FixedPointU64(0);
        }

        // Calculate position's share of pool liquidity
        let position_liquidity = position.liquidity as f64;
        let pool_liquidity = pool.liquidity as f64;
        
        if pool_liquidity <= 0.0 {
            return FixedPointU64(0);
        }

        let liquidity_share = position_liquidity / pool_liquidity;

        // Sum up all emission rewards
        let mut total_emission_value = 0.0;
        for emission in emissions {
            let daily_value = emission.tokens_per_day.to_f64() * emission.token_price_usd.to_f64();
            let annual_value = daily_value * 365.0;
            let position_share = annual_value * liquidity_share;
            total_emission_value += position_share;
        }

        // Position value
        let position_value = position.token0_amount.to_f64() + position.token1_amount.to_f64();
        
        if position_value <= 0.0 {
            return FixedPointU64(0);
        }

        FixedPointU64::from_f64(total_emission_value / position_value)
    }

    /// Calculate capital efficiency of concentrated position
    fn calculate_capital_efficiency(&self, position: &ConcentratedPosition, current_tick: i32) -> f64 {
        let p_current = tick_math::tick_to_price(current_tick);
        let p_lower = tick_math::tick_to_price(position.lower_tick);
        let p_upper = tick_math::tick_to_price(position.upper_tick);

        if p_current <= 0.0 || p_lower >= p_upper {
            return 1.0;
        }

        // Capital efficiency = Full-range liquidity / Concentrated liquidity needed
        // Higher efficiency when range is tighter around current price
        let range_ratio = (p_upper / p_lower.max(0.0001)).ln();
        
        if range_ratio <= 0.0 {
            return 1.0;
        }

        // Check if price is in range
        if p_current < p_lower || p_current >= p_upper {
            return 1.0; // No efficiency boost when out of range
        }

        // Efficiency multiplier (simplified)
        // Real implementation would use exact Uniswap V3 formulas
        let max_range_ln = 10.0; // Approximate full range
        max_range_ln / range_ratio.max(0.01)
    }

    /// Estimate IL for a pool based on historical volatility
    fn estimate_il_for_pool(&self, pool: &PoolState, position: &ConcentratedPosition) -> FixedPointU64 {
        // Use quick IL estimate with assumed volatility
        // In production, this would use actual historical data
        let assumed_annual_vol = 0.80; // 80% annual vol assumption
        let daily_vol = assumed_annual_vol / 365.0_f64.sqrt();
        
        // IL estimate based on position range and volatility
        let range_width = tick_math::tick_to_price(position.upper_tick) / 
            tick_math::tick_to_price(position.lower_tick).max(0.0001);
        
        let range_ln = range_width.ln().abs();
        
        // Narrower ranges have higher IL
        let il_factor = 1.0 / range_ln.max(0.01);
        let il_estimate = self.calculator.quick_il_estimate(daily_vol * il_factor);
        
        FixedPointU64::from_f64(il_estimate.abs() * 365.0) // Annualize
    }

    /// Calculate Sharpe ratio approximation
    fn calculate_sharpe_approximation(&self, net_apr: &FixedPointU64, il_estimate: &FixedPointU64) -> FixedPointU64 {
        // Simplified Sharpe: (Return - RiskFree) / Volatility
        // Assuming risk-free = 0, volatility approximated by IL
        let net_return = net_apr.to_f64();
        let risk = il_estimate.to_f64().max(0.01);
        
        FixedPointU64::from_f64(net_return / risk)
    }

    /// Generate rebalance recommendation for a position
    pub fn recommend_rebalance(
        &self,
        pool: &PoolState,
        position: &ConcentratedPosition,
        current_metrics: &YieldMetrics,
    ) -> RebalanceRecommendation {
        // Check if price is out of range
        let p_current = tick_math::tick_to_price(pool.current_tick);
        let p_lower = tick_math::tick_to_price(position.lower_tick);
        let p_upper = tick_math::tick_to_price(position.upper_tick);

        if p_current < p_lower || p_current >= p_upper {
            // Price out of range - no fees being earned
            let (new_lower, new_upper) = self.calculate_optimal_range_around_price(pool.current_tick, 0.05);
            
            return RebalanceRecommendation {
                should_rebalance: true,
                reason: Some(RebalanceReason::PriceOutOfRange),
                new_lower_tick: Some(new_lower),
                new_upper_tick: Some(new_upper),
                expected_improvement_bps: 500, // Significant improvement expected
            };
        }

        // Check capital efficiency
        let efficiency = self.calculate_capital_efficiency(position, pool.current_tick);
        if efficiency < 2.0 {
            // Low efficiency - could tighten range
            let (new_lower, new_upper) = self.calculate_optimal_range_around_price(pool.current_tick, 0.03);
            
            return RebalanceRecommendation {
                should_rebalance: true,
                reason: Some(RebalanceReason::LowCapitalEfficiency),
                new_lower_tick: Some(new_lower),
                new_upper_tick: Some(new_upper),
                expected_improvement_bps: 100,
            };
        }

        // Check if better opportunities exist (simplified check)
        if current_metrics.net_apr.to_f64() < 0.10 {
            // Less than 10% net APR - might be worth considering other pools
            return RebalanceRecommendation {
                should_rebalance: false,
                reason: Some(RebalanceReason::BetterOpportunity),
                new_lower_tick: None,
                new_upper_tick: None,
                expected_improvement_bps: 0,
            };
        }

        // No rebalance needed
        RebalanceRecommendation {
            should_rebalance: false,
            reason: None,
            new_lower_tick: None,
            new_upper_tick: None,
            expected_improvement_bps: 0,
        }
    }

    /// Calculate optimal tick range centered around current price
    fn calculate_optimal_range_around_price(&self, current_tick: i32, half_range_pct: f64) -> (i32, i32) {
        let p_current = tick_math::tick_to_price(current_tick);
        
        let lower_price = p_current * (1.0 - half_range_pct);
        let upper_price = p_current * (1.0 + half_range_pct);
        
        let lower_tick = tick_math::price_to_tick(lower_price);
        let upper_tick = tick_math::price_to_tick(upper_price);
        
        (lower_tick, upper_tick)
    }

    /// Compare multiple pools and rank by risk-adjusted yield
    pub fn rank_pools(&self, pools: &[(PoolState, Vec<EmissionRewards>)]) -> Vec<(usize, YieldMetrics)> {
        let mut rankings: Vec<(usize, YieldMetrics)> = Vec::with_capacity(pools.len());
        
        for (idx, (pool, emissions)) in pools.iter().enumerate() {
            // Create a dummy position for comparison
            let dummy_position = ConcentratedPosition {
                lower_tick: pool.current_tick - 1000,
                upper_tick: pool.current_tick + 1000,
                liquidity: pool.liquidity / 100,
                token0_amount: FixedPointU64::from_f64(1000.0),
                token1_amount: FixedPointU64::from_f64(1000.0),
                entry_price_ratio: FixedPointU64::from_f64(tick_math::tick_to_price(pool.current_tick)),
            };
            
            let metrics = self.calculate_yield_metrics(pool, &dummy_position, emissions);
            rankings.push((idx, metrics));
        }
        
        // Sort by net APR descending
        rankings.sort_by(|a, b| b.1.net_apr.0.cmp(&a.1.net_apr.0));
        
        rankings
    }

    /// Calculate TVL decay impact on yields
    pub fn analyze_tvl_decay(&self, initial_tvl: FixedPointU64, current_tvl: FixedPointU64, days: u32) -> TvlDecayAnalysis {
        let decay_rate = if initial_tvl.0 > 0 && days > 0 {
            ((initial_tvl.to_f64() - current_tvl.to_f64()) / initial_tvl.to_f64()) / days as f64
        } else {
            0.0
        };

        let projected_tvl_30d = FixedPointU64::from_f64(
            current_tvl.to_f64() * (1.0 - decay_rate * 30.0).max(0.0)
        );

        TvlDecayAnalysis {
            daily_decay_rate: FixedPointU64::from_f64(decay_rate.abs()),
            projected_tvl_30d,
            yield_impact_bps: FixedPointU64::from_f64(decay_rate.abs() * 10000.0),
        }
    }
}

/// TVL Decay Analysis result
#[derive(Debug, Clone)]
pub struct TvlDecayAnalysis {
    pub daily_decay_rate: FixedPointU64,
    pub projected_tvl_30d: FixedPointU64,
    pub yield_impact_bps: FixedPointU64,
}

impl Default for YieldOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_yield_optimizer_creation() {
        let optimizer = YieldOptimizer::new();
        assert_eq!(optimizer.min_rebalance_improvement_bps, 50);
    }

    #[test]
    fn test_capital_efficiency_calculation() {
        let optimizer = YieldOptimizer::new();
        
        let position = ConcentratedPosition {
            lower_tick: -1000,
            upper_tick: 1000,
            liquidity: 1_000_000,
            token0_amount: FixedPointU64::from_f64(1000.0),
            token1_amount: FixedPointU64::from_f64(1000.0),
            entry_price_ratio: FixedPointU64::from_f64(1.0),
        };

        let efficiency = optimizer.calculate_capital_efficiency(&position, 0);
        assert!(efficiency > 1.0); // Should be > 1 for concentrated position
    }

    #[test]
    fn test_rebalance_recommendation_out_of_range() {
        let optimizer = YieldOptimizer::new();
        
        let pool = PoolState {
            pool_id: [0; 32],
            token0: [0; 8],
            token1: [0; 8],
            tvl_usd: FixedPointU64::from_f64(1_000_000.0),
            volume_24h_usd: FixedPointU64::from_f64(100_000.0),
            fee_tier: FixedPointU64::from_f64(0.003),
            current_tick: 1000,
            liquidity: 10_000_000,
        };

        let position = ConcentratedPosition {
            lower_tick: -500,
            upper_tick: 500,
            liquidity: 100_000,
            token0_amount: FixedPointU64::from_f64(100.0),
            token1_amount: FixedPointU64::from_f64(100.0),
            entry_price_ratio: FixedPointU64::from_f64(1.0),
        };

        let metrics = YieldMetrics {
            base_apr: FixedPointU64::from_f64(0.1),
            emission_apr: FixedPointU64(0),
            total_apr: FixedPointU64::from_f64(0.1),
            impermanent_loss_estimate: FixedPointU64(0),
            net_apr: FixedPointU64::from_f64(0.1),
            sharpe_ratio: FixedPointU64(0),
        };

        let rec = optimizer.recommend_rebalance(&pool, &position, &metrics);
        assert!(rec.should_rebalance);
        assert_eq!(rec.reason, Some(RebalanceReason::PriceOutOfRange));
    }

    #[test]
    fn test_tvl_decay_analysis() {
        let optimizer = YieldOptimizer::new();
        
        let initial = FixedPointU64::from_f64(1_000_000.0);
        let current = FixedPointU64::from_f64(900_000.0);
        
        let analysis = optimizer.analyze_tvl_decay(initial, current, 10);
        
        assert!(analysis.daily_decay_rate.to_f64() > 0.0);
        assert!(analysis.projected_tvl_30d.to_f64() < current.to_f64());
    }
}
