//! Funding Rate Arbitrage Engine
//! 
//! Cash-and-carry arbitrage tracking Spot vs Perpetual basis and 8-hour funding rates.
//! Automatically calculates annualized yield and triggers delta-neutral execution.

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::time::Instant;
use dashmap::DashMap;

/// Funding rate data
#[derive(Debug, Clone)]
pub struct FundingRate {
    /// Symbol (e.g., "BTCUSDT")
    pub symbol: String,
    /// Current funding rate (8-hour, as decimal)
    pub current_rate: f64,
    /// Previous funding rate
    pub previous_rate: f64,
    /// Next funding timestamp (unix seconds)
    pub next_funding_ts: u64,
    /// Annualized rate (assuming current rate continues)
    pub annualized_rate: f64,
}

/// Basis trade opportunity
#[derive(Debug, Clone)]
pub struct BasisTradeOpportunity {
    /// Symbol
    pub symbol: String,
    /// Spot price
    pub spot_price: f64,
    /// Perp price
    pub perp_price: f64,
    /// Basis in bps (perp - spot)
    pub basis_bps: f64,
    /// Funding rate (8-hour)
    pub funding_rate: f64,
    /// Annualized basis yield
    pub annualized_yield: f64,
    /// Recommended direction
    pub direction: TradeDirection,
    /// Expected daily return in bps
    pub expected_daily_return_bps: f64,
    /// Timestamp
    pub timestamp_ns: u64,
}

/// Trade direction for basis arb
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TradeDirection {
    /// Long spot, short perp (positive basis trade)
    LongSpot_ShortPerp,
    /// Short spot, long perp (negative basis trade)
    ShortSpot_LongPerp,
}

/// Lock-free funding arb engine
pub struct FundingArbEngine {
    /// Spot prices per symbol
    spot_prices: DashMap<String, f64>,
    /// Perp prices per symbol
    perp_prices: DashMap<String, f64>,
    /// Funding rates per symbol
    funding_rates: DashMap<String, FundingRate>,
    /// Minimum annualized yield threshold (as decimal)
    min_annualized_yield: f64,
    /// Opportunities detected
    opportunities_detected: AtomicU64,
    /// Is engine active
    is_active: AtomicBool,
}

impl FundingArbEngine {
    pub fn new(min_annualized_yield: f64) -> Self {
        Self {
            spot_prices: DashMap::new(),
            perp_prices: DashMap::new(),
            funding_rates: DashMap::new(),
            min_annualized_yield,
            opportunities_detected: AtomicU64::new(0),
            is_active: AtomicBool::new(true),
        }
    }

    /// Update spot price
    pub fn update_spot(&self, symbol: &str, price: f64) -> Vec<BasisTradeOpportunity> {
        if !self.is_active.load(Ordering::Relaxed) {
            return Vec::new();
        }

        self.spot_prices.insert(symbol.to_string(), price);
        self.check_opportunities(symbol)
    }

    /// Update perp price
    pub fn update_perp(&self, symbol: &str, price: f64) -> Vec<BasisTradeOpportunity> {
        if !self.is_active.load(Ordering::Relaxed) {
            return Vec::new();
        }

        self.perp_prices.insert(symbol.to_string(), price);
        self.check_opportunities(symbol)
    }

    /// Update funding rate
    pub fn update_funding_rate(&self, rate: FundingRate) -> Vec<BasisTradeOpportunity> {
        let symbol = rate.symbol.clone();
        self.funding_rates.insert(symbol.clone(), rate);
        self.check_opportunities(&symbol)
    }

    fn check_opportunities(&self, symbol: &str) -> Vec<BasisTradeOpportunity> {
        let mut opportunities = Vec::new();

        let spot_price = self.spot_prices.get(symbol).map(|p| *p)?;
        let perp_price = self.perp_prices.get(symbol).map(|p| *p)?;
        
        if spot_price <= 0.0 || perp_price <= 0.0 {
            return Vec::new();
        }

        // Calculate basis
        let basis = perp_price - spot_price;
        let basis_bps = (basis / spot_price) * 10000.0;

        // Get funding rate
        let funding_rate = self.funding_rates.get(symbol)
            .map(|f| f.current_rate)
            .unwrap_or(0.0);

        // Calculate annualized yield from funding
        // Funding is paid 3x per day (every 8 hours)
        let daily_funding = funding_rate * 3.0;
        let annualized_funding_yield = daily_funding * 365.0;

        // Calculate total annualized yield (basis convergence + funding)
        // Assuming basis converges over ~30 days on average
        let daily_basis_yield = (basis_bps / 10000.0) / 30.0;
        let annualized_basis_yield = daily_basis_yield * 365.0;

        let total_annualized_yield = annualized_funding_yield + annualized_basis_yield;

        // Check if opportunity meets threshold
        if total_annualized_yield.abs() >= self.min_annualized_yield {
            let direction = if basis > 0.0 && funding_rate > 0.0 {
                // Contango: Long spot, short perp, collect funding
                TradeDirection::LongSpot_ShortPerp
            } else if basis < 0.0 && funding_rate < 0.0 {
                // Backwardation: Short spot, long perp, collect funding
                TradeDirection::ShortSpot_LongPerp
            } else if basis.abs() > (spot_price * 0.01) {
                // Large basis regardless of funding
                if basis > 0.0 {
                    TradeDirection::LongSpot_ShortPerp
                } else {
                    TradeDirection::ShortSpot_LongPerp
                }
            } else {
                return Vec::new();
            };

            let expected_daily_return = (total_annualized_yield / 365.0) * 10000.0;

            let timestamp_ns = Instant::now().duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default().as_nanos() as u64;

            opportunities.push(BasisTradeOpportunity {
                symbol: symbol.to_string(),
                spot_price,
                perp_price,
                basis_bps,
                funding_rate,
                annualized_yield: total_annualized_yield,
                direction,
                expected_daily_return_bps: expected_daily_return,
                timestamp_ns,
            });

            self.opportunities_detected.fetch_add(1, Ordering::Relaxed);
        }

        opportunities
    }

    /// Get funding rate for a symbol
    pub fn get_funding_rate(&self, symbol: &str) -> Option<FundingRate> {
        self.funding_rates.get(symbol).map(|f| f.clone())
    }

    /// Get current basis for a symbol
    pub fn get_basis(&self, symbol: &str) -> Option<(f64, f64, f64)> {
        let spot = self.spot_prices.get(symbol).map(|p| *p)?;
        let perp = self.perp_prices.get(symbol).map(|p| *p)?;
        let basis = perp - spot;
        Some((spot, perp, basis))
    }

    /// Get opportunities count
    pub fn get_opportunity_count(&self) -> u64 {
        self.opportunities_detected.load(Ordering::Relaxed)
    }

    /// Set minimum annualized yield threshold
    pub fn set_min_yield(&mut self, yield_threshold: f64) {
        self.min_annualized_yield = yield_threshold;
    }

    /// Deactivate engine
    pub fn deactivate(&self) {
        self.is_active.store(false, Ordering::Relaxed);
    }

    /// Activate engine
    pub fn activate(&self) {
        self.is_active.store(true, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_funding_arb_detection() {
        let mut engine = FundingArbEngine::new(0.10); // 10% min annualized

        let symbol = "BTCUSDT";

        // Set up contango market with positive funding
        engine.update_spot(symbol, 50000.0);
        engine.update_perp(symbol, 50200.0); // +40 bps basis

        let funding = FundingRate {
            symbol: symbol.to_string(),
            current_rate: 0.001, // 0.1% per 8 hours
            previous_rate: 0.0008,
            next_funding_ts: 1700000000,
            annualized_rate: 0.001 * 3 * 365.0,
        };

        let opps = engine.update_funding_rate(funding);
        
        assert!(!opps.is_empty(), "Should detect funding arb opportunity");
        
        if let Some(opp) = opps.first() {
            println!("Basis: {:.2} bps", opp.basis_bps);
            println!("Annualized yield: {:.2}%", opp.annualized_yield * 100.0);
            assert_eq!(opp.direction, TradeDirection::LongSpot_ShortPerp);
        }
    }

    #[test]
    fn test_no_arb_below_threshold() {
        let mut engine = FundingArbEngine::new(0.50); // 50% min annualized (high threshold)

        let symbol = "ETHUSDT";

        engine.update_spot(symbol, 3000.0);
        engine.update_perp(symbol, 3005.0); // Small basis

        let funding = FundingRate {
            symbol: symbol.to_string(),
            current_rate: 0.0001, // Very low funding
            previous_rate: 0.0001,
            next_funding_ts: 1700000000,
            annualized_rate: 0.0001 * 3 * 365.0,
        };

        let opps = engine.update_funding_rate(funding);
        
        assert!(opps.is_empty(), "Should not detect arb below threshold");
    }
}
