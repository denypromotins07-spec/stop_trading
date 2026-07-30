//! Calendar Spread Analyzer
//! 
//! Quarterly futures calendar spread analyzer for term-structure trading.
//! Tracks contango/backwardation shifts between different expiration months.

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::time::Instant;
use dashmap::DashMap;

/// Maximum contracts tracked per underlying
const MAX_CONTRACTS: usize = 12;

/// Futures contract
#[derive(Debug, Clone)]
pub struct FuturesContract {
    /// Underlying symbol (e.g., "BTC")
    pub underlying: String,
    /// Contract month code (e.g., "Z4" for Dec 2024)
    pub contract_month: String,
    /// Expiration timestamp (unix seconds)
    pub expiration_ts: u64,
    /// Current price
    pub price: f64,
    /// Volume
    pub volume: u64,
    /// Open interest
    pub open_interest: u64,
}

/// Calendar spread opportunity
#[derive(Debug, Clone)]
pub struct CalendarSpreadOpportunity {
    /// Underlying
    pub underlying: String,
    /// Near contract month
    pub near_month: String,
    /// Far contract month
    pub far_month: String,
    /// Near price
    pub near_price: f64,
    /// Far price
    pub far_price: f64,
    /// Spread in bps (far - near)
    pub spread_bps: f64,
    /// Annualized spread yield
    pub annualized_yield: f64,
    /// Market structure
    pub structure: MarketStructure,
    /// Recommended trade
    pub direction: SpreadDirection,
    /// Days to near expiry
    pub days_to_expiry: u32,
    /// Timestamp
    pub timestamp_ns: u64,
}

/// Market structure
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MarketStructure {
    /// Contango: far > near
    Contango,
    /// Backwardation: near > far
    Backwardation,
    /// Flat
    Flat,
}

/// Spread direction
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SpreadDirection {
    /// Long near, short far (bull spread)
    LongNear_ShortFar,
    /// Short near, long far (bear spread)
    ShortNear_LongFar,
}

/// Lock-free calendar spread engine
pub struct CalendarSpreadEngine {
    /// Contracts by underlying
    contracts: DashMap<String, Vec<FuturesContract>>,
    /// Minimum yield threshold
    min_annualized_yield: f64,
    /// Opportunities detected
    opportunities_detected: AtomicU64,
    /// Is engine active
    is_active: AtomicBool,
}

impl CalendarSpreadEngine {
    pub fn new(min_annualized_yield: f64) -> Self {
        Self {
            contracts: DashMap::new(),
            min_annualized_yield,
            opportunities_detected: AtomicU64::new(0),
            is_active: AtomicBool::new(true),
        }
    }

    /// Add or update a contract
    pub fn update_contract(&self, contract: FuturesContract) -> Vec<CalendarSpreadOpportunity> {
        if !self.is_active.load(Ordering::Relaxed) {
            return Vec::new();
        }

        let underlying = contract.underlying.clone();
        
        // Get or create contract list
        let mut contracts = self.contracts.entry(underlying.clone()).or_insert_with(Vec::new);
        
        // Update existing or add new
        let exists = contracts.iter_mut().find(|c| c.contract_month == contract.contract_month);
        if let Some(existing) = exists {
            *existing = contract;
        } else {
            if contracts.len() < MAX_CONTRACTS {
                contracts.push(contract);
            }
        }

        // Check all calendar spreads for this underlying
        self.check_spreads(&underlying)
    }

    fn check_spreads(&self, underlying: &str) -> Vec<CalendarSpreadOpportunity> {
        let mut opportunities = Vec::new();

        let contracts = match self.contracts.get(underlying) {
            Some(c) => c.clone(),
            None => return Vec::new(),
        };

        if contracts.len() < 2 {
            return Vec::new();
        }

        // Sort by expiration
        let mut sorted: Vec<_> = contracts.into_iter().collect();
        sorted.sort_by_key(|c| c.expiration_ts);

        let now_ts = Instant::now().duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default().as_secs();

        // Check all pairs
        for i in 0..sorted.len() {
            for j in (i + 1)..sorted.len() {
                let near = &sorted[i];
                let far = &sorted[j];

                if near.price <= 0.0 || far.price <= 0.0 {
                    continue;
                }

                // Calculate spread
                let spread = far.price - near.price;
                let spread_bps = (spread / near.price) * 10000.0;

                // Calculate time difference in years
                let days_between = ((far.expiration_ts - near.expiration_ts) / 86400) as f64;
                let years = days_between / 365.0;

                if years <= 0.0 {
                    continue;
                }

                // Annualized yield
                let annualized_yield = (spread_bps / 10000.0) / years;

                // Determine market structure
                let structure = if spread_bps > 5.0 {
                    MarketStructure::Contango
                } else if spread_bps < -5.0 {
                    MarketStructure::Backwardation
                } else {
                    MarketStructure::Flat
                };

                // Check threshold
                if annualized_yield.abs() >= self.min_annualized_yield {
                    let direction = if spread > 0.0 {
                        SpreadDirection::ShortNear_LongFar
                    } else {
                        SpreadDirection::LongNear_ShortFar
                    };

                    let days_to_expiry = ((near.expiration_ts - now_ts) / 86400) as u32;
                    let timestamp_ns = Instant::now().duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default().as_nanos() as u64;

                    opportunities.push(CalendarSpreadOpportunity {
                        underlying: underlying.to_string(),
                        near_month: near.contract_month.clone(),
                        far_month: far.contract_month.clone(),
                        near_price: near.price,
                        far_price: far.price,
                        spread_bps,
                        annualized_yield,
                        structure,
                        direction,
                        days_to_expiry,
                        timestamp_ns,
                    });

                    self.opportunities_detected.fetch_add(1, Ordering::Relaxed);
                }
            }
        }

        opportunities
    }

    /// Get contracts for an underlying
    pub fn get_contracts(&self, underlying: &str) -> Vec<FuturesContract> {
        self.contracts.get(underlying).map(|c| c.clone()).unwrap_or_default()
    }

    /// Get current term structure
    pub fn get_term_structure(&self, underlying: &str) -> Vec<(String, f64)> {
        let mut contracts = self.get_contracts(underlying);
        contracts.sort_by_key(|c| c.expiration_ts);
        contracts.into_iter().map(|c| (c.contract_month, c.price)).collect()
    }

    /// Get opportunities count
    pub fn get_opportunity_count(&self) -> u64 {
        self.opportunities_detected.load(Ordering::Relaxed)
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
    fn test_calendar_spread_detection() {
        let engine = CalendarSpreadEngine::new(0.05); // 5% min annualized

        let now = Instant::now().duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default().as_secs();

        // Create Dec 2024 and Mar 2025 contracts
        let dec_contract = FuturesContract {
            underlying: "BTC".to_string(),
            contract_month: "Z4".to_string(),
            expiration_ts: now + (90 * 86400),
            price: 50000.0,
            volume: 10000,
            open_interest: 5000,
        };

        let mar_contract = FuturesContract {
            underlying: "BTC".to_string(),
            contract_month: "H5".to_string(),
            expiration_ts: now + (180 * 86400),
            price: 50500.0, // Contango
            volume: 8000,
            open_interest: 4000,
        };

        let opps1 = engine.update_contract(dec_contract);
        assert!(opps1.is_empty()); // Need both contracts

        let opps2 = engine.update_contract(mar_contract);
        assert!(!opps2.is_empty(), "Should detect calendar spread");

        if let Some(opp) = opps2.first() {
            println!("Spread: {:.2} bps", opp.spread_bps);
            println!("Annualized: {:.2}%", opp.annualized_yield * 100.0);
            println!("Structure: {:?}", opp.structure);
        }
    }
}
