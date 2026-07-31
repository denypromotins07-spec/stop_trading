//! Impermanent Loss Calculator for Uniswap V3 and Raydium
//! 
//! Implements high-speed Impermanent Loss (IL) calculations for concentrated liquidity positions.
//! Uses analytical approximations to evaluate IL vs accrued fees in O(1) time.

use std::sync::atomic::{AtomicU64, Ordering};

/// Fixed-point representation for DeFi calculations (6 decimal places for gas efficiency)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FixedPointU64(u64);

impl FixedPointU64 {
    const PRECISION: u64 = 1_000_000; // 6 decimal places

    pub fn from_f64(val: f64) -> Self {
        FixedPointU64((val * Self::PRECISION as f64).round() as u64)
    }

    pub fn to_f64(self) -> f64 {
        self.0 as f64 / Self::PRECISION as f64
    }

    pub fn checked_mul(self, other: FixedPointU64) -> Option<FixedPointU64> {
        let result = (self.0 as u128).checked_mul(other.0 as u128)?;
        Some(FixedPointU64((result / Self::PRECISION as u128) as u64))
    }

    pub fn checked_div(self, other: FixedPointU64) -> Option<FixedPointU64> {
        let result = (self.0 as u128).checked_mul(Self::PRECISION as u128)?;
        Some(FixedPointU64((result / other.0 as u128) as u64))
    }

    pub fn checked_add(self, other: FixedPointU64) -> Option<FixedPointU64> {
        Some(FixedPointU64(self.0.checked_add(other.0)?))
    }

    pub fn checked_sub(self, other: FixedPointU64) -> Option<FixedPointU64> {
        Some(FixedPointU64(self.0.checked_sub(other.0)?))
    }

    pub fn sqrt(&self) -> FixedPointU64 {
        let val = self.0 as f64 / Self::PRECISION as f64;
        FixedPointU64((val.sqrt() * Self::PRECISION as f64).round() as u64)
    }
}

/// Concentrated liquidity position parameters (Uniswap V3 style)
#[derive(Debug, Clone, Copy)]
pub struct ConcentratedPosition {
    pub lower_tick: i32,
    pub upper_tick: i32,
    pub liquidity: u128,
    pub token0_amount: FixedPointU64,
    pub token1_amount: FixedPointU64,
    pub entry_price_ratio: FixedPointU64, // P = token1/token0
}

/// Impermanent Loss calculation result
#[derive(Debug, Clone, Copy)]
pub struct ImpermanentLossResult {
    pub il_percentage: FixedPointU64,
    pub value_hodl: FixedPointU64,      // Value if held outside pool
    pub value_lp: FixedPointU64,        // Current LP value
    pub fee_income: FixedPointU64,      // Accrued fees
    pub net_pnl: FixedPointU64,         // IL + fees
    pub break_even_fee_rate: FixedPointU64,
}

/// Tick-to-price conversion utilities
pub mod tick_math {
    use super::FixedPointU64;

    const MIN_TICK: i32 = -887272;
    const MAX_TICK: i32 = 887272;
    const Q96: u128 = 1u128 << 96;

    /// Convert tick to price (P = 1.0001^tick)
    pub fn tick_to_price(tick: i32) -> f64 {
        if tick < MIN_TICK || tick > MAX_TICK {
            return 0.0;
        }
        1.0001_f64.powi(tick)
    }

    /// Convert price to tick
    pub fn price_to_tick(price: f64) -> i32 {
        if price <= 0.0 {
            return MIN_TICK;
        }
        (price.ln() / 1.0001_f64.ln()).floor() as i32
    }

    /// Get sqrt price from tick (for internal calculations)
    pub fn get_sqrt_price_x96(tick: i32) -> u128 {
        if tick < MIN_TICK || tick > MAX_TICK {
            return 0;
        }
        let sqrt_price = 1.0001_f64.powi(tick / 2);
        (sqrt_price * Q96 as f64) as u128
    }
}

/// Main Impermanent Loss Calculator
pub struct ImpermanentLossCalculator {
    pub calculation_counter: AtomicU64,
}

impl ImpermanentLossCalculator {
    pub fn new() -> Self {
        ImpermanentLossCalculator {
            calculation_counter: AtomicU64::new(0),
        }
    }

    /// Calculate IL for a concentrated liquidity position in O(1)
    /// 
    /// Formula adapted from Uniswap V3 whitepaper:
    /// IL = 2 * sqrt(P_new) / (1 + P_new) - 1 (for full-range)
    /// For concentrated: more complex formula involving liquidity bounds
    pub fn calculate_il(&self, position: &ConcentratedPosition, current_price: FixedPointU64) -> ImpermanentLossResult {
        self.calculation_counter.fetch_add(1, Ordering::Relaxed);

        let p_entry = position.entry_price_ratio.to_f64();
        let p_current = current_price.to_f64();

        if p_entry <= 0.0 || p_current <= 0.0 {
            return self.zero_result();
        }

        // Calculate value if held (HODL)
        let total_value_entry = position.token0_amount.to_f64() * p_entry + position.token1_amount.to_f64();
        
        // HODL value at current price
        let value_hodl = FixedPointU64::from_f64(
            position.token0_amount.to_f64() * p_current + position.token1_amount.to_f64()
        );

        // Calculate LP value using concentrated liquidity formulas
        let lp_value = self.calculate_concentrated_value(position, p_current);
        let value_lp = FixedPointU64::from_f64(lp_value);

        // IL = (LP Value - HODL Value) / HODL Value
        let il_raw = if value_hodl.0 > 0 {
            (lp_value - total_value_entry) / total_value_entry
        } else {
            0.0
        };

        let il_percentage = FixedPointU64::from_f64(il_raw.abs());

        // Estimate fee income based on liquidity share and volume
        // This is a simplified model; real implementation would track actual fees
        let fee_income = self.estimate_fee_income(position, p_current);

        // Net PnL = IL + Fees
        let net_pnl = FixedPointU64::from_f64(il_raw + fee_income);

        // Break-even fee rate: what fee rate would offset IL
        let break_even_fee_rate = if il_raw < 0.0 && total_value_entry > 0.0 {
            FixedPointU64::from_f64(il_raw.abs())
        } else {
            FixedPointU64(0)
        };

        ImpermanentLossResult {
            il_percentage,
            value_hodl,
            value_lp,
            fee_income: FixedPointU64::from_f64(fee_income),
            net_pnl,
            break_even_fee_rate,
        }
    }

    /// Calculate concentrated liquidity position value at given price
    fn calculate_concentrated_value(&self, position: &ConcentratedPosition, p_current: f64) -> f64 {
        let p_lower = tick_math::tick_to_price(position.lower_tick);
        let p_upper = tick_math::tick_to_price(position.upper_tick);
        let L = position.liquidity as f64;

        if p_current <= 0.0 {
            return 0.0;
        }

        // Three cases based on current price relative to position range
        if p_current < p_lower {
            // Price below range: all token0
            let amount0 = L * (1.0 / p_lower.sqrt() - 1.0 / p_upper.sqrt());
            amount0 * p_current
        } else if p_current >= p_upper {
            // Price above range: all token1
            let amount1 = L * (p_upper.sqrt() - p_lower.sqrt());
            amount1
        } else {
            // Price within range: split
            let sqrt_p = p_current.sqrt();
            let sqrt_lower = p_lower.sqrt();
            let sqrt_upper = p_upper.sqrt();

            let amount0 = L * (1.0 / sqrt_p - 1.0 / sqrt_upper);
            let amount1 = L * (sqrt_p - sqrt_lower);

            amount0 * p_current + amount1
        }
    }

    /// Estimate fee income based on position liquidity and assumed volume
    fn estimate_fee_income(&self, position: &ConcentratedPosition, p_current: f64) -> f64 {
        // Simplified fee estimation
        // Real implementation would track actual accumulated fees
        let p_lower = tick_math::tick_to_price(position.lower_tick);
        let p_upper = tick_math::tick_to_price(position.upper_tick);

        // Fee tier assumption (0.05%, 0.3%, or 1%)
        let fee_tier = 0.003;

        // Liquidity concentration factor
        let range_width = (p_upper / p_lower.max(0.0001)).ln().abs();
        let concentration_factor = 1.0 / range_width.max(0.01);

        // Estimated daily fees (simplified)
        let assumed_daily_volume = 1_000_000.0; // $1M daily volume assumption
        let liquidity_share = concentration_factor.min(1.0);
        
        assumed_daily_volume * fee_tier * liquidity_share * 0.01 // 1% of daily fees
    }

    fn zero_result(&self) -> ImpermanentLossResult {
        ImpermanentLossResult {
            il_percentage: FixedPointU64(0),
            value_hodl: FixedPointU64(0),
            value_lp: FixedPointU64(0),
            fee_income: FixedPointU64(0),
            net_pnl: FixedPointU64(0),
            break_even_fee_rate: FixedPointU64(0),
        }
    }

    /// Quick IL approximation without full position details
    /// Useful for rapid screening of potential positions
    pub fn quick_il_estimate(&self, price_change_pct: f64) -> f64 {
        // Standard AMM IL approximation: IL ≈ 2*sqrt(r)/(1+r) - 1
        // where r = price ratio change
        let r = 1.0 + price_change_pct;
        if r <= 0.0 {
            return -1.0; // Total loss
        }
        
        let il = 2.0 * r.sqrt() / (1.0 + r) - 1.0;
        il
    }

    /// Calculate optimal tick range for given volatility target
    pub fn calculate_optimal_range(&self, volatility: f64, confidence: f64) -> (i32, i32) {
        // Using normal distribution approximation
        // Range should cover ±Z*σ where Z is confidence level z-score
        let z_score = match confidence {
            c if c >= 0.99 => 2.576,
            c if c >= 0.95 => 1.96,
            c if c >= 0.90 => 1.645,
            _ => 1.0,
        };

        let price_range = z_score * volatility;
        let center_tick = 0; // Would be calculated from current price

        let lower_price = (-price_range).exp();
        let upper_price = price_range.exp();

        let lower_tick = tick_math::price_to_tick(lower_price);
        let upper_tick = tick_math::price_to_tick(upper_price);

        (center_tick + lower_tick, center_tick + upper_tick)
    }
}

/// Multi-pool IL tracker for portfolio-wide analysis
pub struct PortfolioIlTracker {
    pub calculator: ImpermanentLossCalculator,
    pub positions: Vec<ConcentratedPosition>,
    pub total_il: FixedPointU64,
    pub total_fees: FixedPointU64,
}

impl PortfolioIlTracker {
    pub fn new() -> Self {
        PortfolioIlTracker {
            calculator: ImpermanentLossCalculator::new(),
            positions: Vec::with_capacity(64),
            total_il: FixedPointU64(0),
            total_fees: FixedPointU64(0),
        }
    }

    pub fn add_position(&mut self, position: ConcentratedPosition) {
        if self.positions.len() < self.positions.capacity() {
            self.positions.push(position);
        }
    }

    pub fn update_portfolio_metrics(&mut self, prices: &[FixedPointU64]) {
        self.total_il = FixedPointU64(0);
        self.total_fees = FixedPointU64(0);

        for (i, position) in self.positions.iter().enumerate() {
            let price = prices.get(i).copied().unwrap_or(position.entry_price_ratio);
            let result = self.calculator.calculate_il(position, price);
            
            self.total_il = self.total_il.checked_add(result.il_percentage).unwrap_or(self.total_il);
            self.total_fees = self.total_fees.checked_add(result.fee_income).unwrap_or(self.total_fees);
        }
    }

    pub fn get_net_portfolio_pnl(&self) -> FixedPointU64 {
        // Net PnL = Total Fees - Total IL
        self.total_fees.checked_sub(self.total_il).unwrap_or(FixedPointU64(0))
    }
}

impl Default for PortfolioIlTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tick_to_price_conversion() {
        let price = tick_math::tick_to_price(0);
        assert!((price - 1.0).abs() < 0.0001);

        let price = tick_math::tick_to_price(1000);
        assert!(price > 1.0);

        let tick = tick_math::price_to_tick(1.0);
        assert!(tick >= -1 && tick <= 1);
    }

    #[test]
    fn test_quick_il_estimate() {
        let calc = ImpermanentLossCalculator::new();
        
        // No price change = no IL
        let il = calc.quick_il_estimate(0.0);
        assert!(il.abs() < 0.0001);

        // 100% price increase
        let il = calc.quick_il_estimate(1.0);
        assert!(il < 0.0); // IL is always negative for LP
        
        // Should be approximately -5.7% for 2x price change
        assert!(il > -0.1 && il < -0.05);
    }

    #[test]
    fn test_fixed_point_arithmetic() {
        let a = FixedPointU64::from_f64(100.5);
        let b = FixedPointU64::from_f64(2.0);
        
        let sum = a.checked_add(b).unwrap();
        assert!((sum.to_f64() - 102.5).abs() < 0.000001);

        let product = a.checked_mul(b).unwrap();
        assert!((product.to_f64() - 201.0).abs() < 0.000001);
    }

    #[test]
    fn test_optimal_range_calculation() {
        let calc = ImpermanentLossCalculator::new();
        let (lower, upper) = calc.calculate_optimal_range(0.05, 0.95);
        
        assert!(lower < 0);
        assert!(upper > 0);
        assert!(upper - lower > 0);
    }
}
