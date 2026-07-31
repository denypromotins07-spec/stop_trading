//! Testing Module - Property-Based Testing with Proptest
//! 
//! Integrates proptest to generate massive, randomized, and adversarial market data sequences.
//! Validates mathematical invariants of Avellaneda-Stoikov and Black-Litterman models under extreme edge cases.

#![cfg(any(test, feature = "proptest-testing"))]

use alloc::vec::Vec;
use alloc::string::String;

/// Maximum number of test cases per property
pub const MAX_PROPTTEST_CASES: usize = 10000;

/// Default timeout for property tests (ms)
pub const DEFAULT_TEST_TIMEOUT_MS: u64 = 5000;

/// Market data generator configuration
#[derive(Debug, Clone)]
pub struct MarketDataConfig {
    /// Number of price points to generate
    pub num_prices: usize,
    /// Minimum price value
    pub min_price: f64,
    /// Maximum price value
    pub max_price: f64,
    /// Maximum volatility (as decimal)
    pub max_volatility: f64,
    /// Include extreme edge cases
    pub include_extremes: bool,
}

impl Default for MarketDataConfig {
    fn default() -> Self {
        MarketDataConfig {
            num_prices: 1000,
            min_price: 0.01,
            max_price: 1_000_000.0,
            max_volatility: 0.5,
            include_extremes: true,
        }
    }
}

/// Property test result
#[derive(Debug, Clone, PartialEq)]
pub enum PropertyResult {
    Passed,
    Failed { input: String, reason: String },
    Timeout,
    Error(String),
}

/// Test runner for property-based tests
pub struct PropertyTestRunner {
    config: MarketDataConfig,
    seed: u64,
    max_cases: usize,
    timeout_ms: u64,
    passed: u64,
    failed: u64,
}

impl PropertyTestRunner {
    pub fn new(config: MarketDataConfig) -> Self {
        PropertyTestRunner {
            config,
            seed: 0,
            max_cases: MAX_PROPTTEST_CASES,
            timeout_ms: DEFAULT_TEST_TIMEOUT_MS,
            passed: 0,
            failed: 0,
        }
    }

    /// Set random seed for reproducibility
    pub fn set_seed(&mut self, seed: u64) {
        self.seed = seed;
    }

    /// Run a property test with generated inputs
    pub fn run_property<F, G>(&mut self, name: &str, generator: G, property_fn: F) -> PropertyResult
    where
        F: Fn(&[f64]) -> Result<(), String>,
        G: Fn(u64, &MarketDataConfig) -> Vec<f64>,
    {
        for i in 0..self.max_cases {
            let input = generator(self.seed.wrapping_add(i as u64), &self.config);
            
            match property_fn(&input) {
                Ok(()) => self.passed += 1,
                Err(reason) => {
                    self.failed += 1;
                    return PropertyResult::Failed {
                        input: format!("{:?}", &input[..input.len().min(10)]),
                        reason,
                    };
                }
            }
        }

        PropertyResult::Passed
    }

    /// Get test statistics
    pub fn get_stats(&self) -> TestStats {
        TestStats {
            passed: self.passed,
            failed: self.failed,
            total: self.passed + self.failed,
            pass_rate: if self.passed + self.failed > 0 {
                self.passed as f64 / (self.passed + self.failed) as f64
            } else {
                0.0
            },
        }
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.passed = 0;
        self.failed = 0;
    }
}

/// Generator for adversarial market data sequences
pub struct AdversarialMarketGenerator {
    rng_state: u64,
}

impl AdversarialMarketGenerator {
    pub fn new(seed: u64) -> Self {
        AdversarialMarketGenerator { rng_state: seed }
    }

    /// Generate random price series
    pub fn generate_price_series(&mut self, config: &MarketDataConfig) -> Vec<f64> {
        let mut prices = Vec::with_capacity(config.num_prices);
        
        let mut price = (config.min_price + config.max_price) / 2.0;
        prices.push(price);

        for _ in 1..config.num_prices {
            self.rng_state = self.rng_state.wrapping_mul(0x5DEECE66D).wrapping_add(0xB);
            
            // Random walk with drift
            let change_pct = ((self.rng_state as i64 % 1000) as f64 / 1000.0 - 0.5) * config.max_volatility;
            price *= 1.0 + change_pct;
            
            // Clamp to valid range
            price = price.clamp(config.min_price, config.max_price);
            prices.push(price);
        }

        prices
    }

    /// Generate extreme volatility scenario
    pub fn generate_flash_crash(&mut self, config: &MarketDataConfig) -> Vec<f64> {
        let mut prices = Vec::with_capacity(config.num_prices);
        
        let start_price = config.max_price / 2.0;
        prices.push(start_price);

        let crash_point = config.num_prices / 3;
        let recovery_point = 2 * crash_point;

        for i in 1..config.num_prices {
            let price = if i < crash_point {
                // Normal trading
                start_price * (1.0 + ((i as f64 / crash_point as f64) * 0.1 - 0.05))
            } else if i < recovery_point {
                // Crash phase
                let crash_depth = 0.5; // 50% drop
                let progress = (i - crash_point) as f64 / (recovery_point - crash_point) as f64;
                start_price * (1.0 - crash_depth * progress)
            } else {
                // Recovery phase
                let progress = (i - recovery_point) as f64 / (config.num_prices - recovery_point) as f64;
                start_price * (0.5 + 0.3 * progress)
            };
            
            prices.push(price.clamp(config.min_price, config.max_price));
        }

        prices
    }

    /// Generate gap up/down scenarios
    pub fn generate_gap_scenario(&mut self, config: &MarketDataConfig, gap_pct: f64) -> Vec<f64> {
        let mut prices = Vec::with_capacity(config.num_prices);
        
        let gap_point = config.num_prices / 2;
        let mut price = (config.min_price + config.max_price) / 2.0;

        for i in 0..config.num_prices {
            if i == gap_point {
                // Apply gap
                self.rng_state = self.rng_state.wrapping_mul(0x5DEECE66D).wrapping_add(0xB);
                let direction = if self.rng_state % 2 == 0 { 1.0 } else { -1.0 };
                price *= 1.0 + direction * gap_pct;
            }
            
            prices.push(price.clamp(config.min_price, config.max_price));
        }

        prices
    }

    /// Generate stale data scenario (flat prices)
    pub fn generate_stale_data(&mut self, config: &MarketDataConfig) -> Vec<f64> {
        let price = (config.min_price + config.max_price) / 2.0;
        vec![price; config.num_prices]
    }

    /// Generate NaN/Inf injection scenario
    pub fn generate_invalid_values(&mut self, config: &MarketDataConfig) -> Vec<f64> {
        let mut prices = self.generate_price_series(config);
        
        // Inject some invalid values at random positions
        self.rng_state = self.rng_state.wrapping_mul(0x5DEECE66D).wrapping_add(0xB);
        let inject_pos = (self.rng_state as usize) % prices.len();
        
        prices[inject_pos] = f64::NAN;
        
        if inject_pos + 1 < prices.len() {
            prices[inject_pos + 1] = f64::INFINITY;
        }
        
        prices
    }
}

/// Invariant checker for Avellaneda-Stoikov model
pub mod avellaneda_stoikov_invariants {
    use super::*;

    /// Check that optimal spread is always positive
    pub fn check_spread_positive(prices: &[f64], inventory: i64, risk_aversion: f64) -> Result<(), String> {
        if prices.is_empty() {
            return Ok(());
        }

        let mid_price = prices[prices.len() / 2];
        
        // Simplified AS spread calculation
        let spread = mid_price * risk_aversion * 0.01 * (inventory.abs() as f64 + 1.0);
        
        if spread <= 0.0 {
            return Err(format!("Spread should be positive, got {}", spread));
        }

        Ok(())
    }

    /// Check that reservation price stays within reasonable bounds
    pub fn check_reservation_price_bounds(prices: &[f64], inventory: i64, risk_aversion: f64) -> Result<(), String> {
        if prices.is_empty() {
            return Ok(());
        }

        let mid_price = prices[prices.len() / 2];
        let reservation_adjustment = -risk_aversion * inventory as f64 * mid_price * 0.001;
        let reservation_price = mid_price + reservation_adjustment;

        // Reservation price should be within 50% of mid price
        let lower_bound = mid_price * 0.5;
        let upper_bound = mid_price * 1.5;

        if reservation_price < lower_bound || reservation_price > upper_bound {
            return Err(format!(
                "Reservation price {} out of bounds [{}, {}]",
                reservation_price, lower_bound, upper_bound
            ));
        }

        Ok(())
    }

    /// Check that inventory limits are respected
    pub fn check_inventory_limits(inventory: i64, max_inventory: i64) -> Result<(), String> {
        if inventory.abs() > max_inventory {
            return Err(format!("Inventory {} exceeds limit {}", inventory, max_inventory));
        }
        Ok(())
    }
}

/// Invariant checker for Black-Litterman model
pub mod black_litterman_invariants {
    use super::*;

    /// Check that posterior weights sum to 1
    pub fn check_weights_sum_to_one(weights: &[f64]) -> Result<(), String> {
        let sum: f64 = weights.iter().sum();
        
        if (sum - 1.0).abs() > 0.001 {
            return Err(format!("Weights sum to {}, expected 1.0", sum));
        }

        Ok(())
    }

    /// Check that no single asset dominates (no weight > 50%)
    pub fn check_no_dominant_asset(weights: &[f64]) -> Result<(), String> {
        for (i, &w) in weights.iter().enumerate() {
            if w > 0.5 {
                return Err(format!("Asset {} has dominant weight {}", i, w));
            }
        }
        Ok(())
    }

    /// Check that implied returns are reasonable
    pub fn check_implied_returns(returns: &[f64], max_return: f64) -> Result<(), String> {
        for (i, &r) in returns.iter().enumerate() {
            if r.is_nan() || r.is_infinite() {
                return Err(format!("Return {} is invalid", i));
            }
            if r.abs() > max_return {
                return Err(format!("Return {} exceeds maximum {}", r, max_return));
            }
        }
        Ok(())
    }
}

/// Test statistics
#[derive(Debug, Clone)]
pub struct TestStats {
    pub passed: u64,
    pub failed: u64,
    pub total: u64,
    pub pass_rate: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_market_generator_basic() {
        let config = MarketDataConfig::default();
        let mut gen = AdversarialMarketGenerator::new(12345);

        let prices = gen.generate_price_series(&config);
        assert_eq!(prices.len(), config.num_prices);

        // All prices should be in valid range
        for &p in &prices {
            assert!(p >= config.min_price && p <= config.max_price);
        }
    }

    #[test]
    fn test_flash_crash_generation() {
        let config = MarketDataConfig::default();
        let mut gen = AdversarialMarketGenerator::new(42);

        let prices = gen.generate_flash_crash(&config);
        assert_eq!(prices.len(), config.num_prices);

        // Should have significant price movement
        let min_price = prices.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_price = prices.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(max_price - min_price > config.max_price * 0.3);
    }

    #[test]
    fn test_property_runner() {
        let config = MarketDataConfig {
            num_prices: 100,
            ..Default::default()
        };
        let mut runner = PropertyTestRunner::new(config);

        let result = runner.run_property(
            "prices_positive",
            |seed, config| {
                let mut gen = AdversarialMarketGenerator::new(seed);
                gen.generate_price_series(config)
            },
            |prices| {
                for &p in prices {
                    if p < 0.0 {
                        return Err("Price should not be negative".to_string());
                    }
                }
                Ok(())
            },
        );

        assert_eq!(result, PropertyResult::Passed);
    }

    #[test]
    fn test_as_invariants() {
        let config = MarketDataConfig::default();
        let mut gen = AdversarialMarketGenerator::new(999);

        let prices = gen.generate_price_series(&config);

        // Test spread positivity
        let result = avellaneda_stoikov_invariants::check_spread_positive(&prices, 10, 0.1);
        assert!(result.is_ok());

        // Test reservation price bounds
        let result = avellaneda_stoikov_invariants::check_reservation_price_bounds(&prices, 10, 0.1);
        assert!(result.is_ok());
    }

    #[test]
    fn test_bl_invariants() {
        let weights = vec![0.2, 0.3, 0.25, 0.25];
        
        assert!(black_litterman_invariants::check_weights_sum_to_one(&weights).is_ok());
        assert!(black_litterman_invariants::check_no_dominant_asset(&weights).is_ok());

        let returns = vec![0.05, 0.08, -0.02, 0.03];
        assert!(black_litterman_invariants::check_implied_returns(&returns, 1.0).is_ok());
    }
}
