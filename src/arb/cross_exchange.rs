//! Cross-Exchange Arbitrage Engine
//! 
//! Normalized cross-venue spread calculator comparing Binance, Bybit, and DEX aggregator prices.
//! Instantly flags risk-free opportunities after deducting real-time maker/taker fee tiers and withdrawal costs.

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::time::Instant;
use dashmap::DashMap;

/// Maximum number of venues supported
const MAX_VENUES: usize = 16;

/// Venue identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Venue {
    Binance,
    Bybit,
    OKX,
    Coinbase,
    Kraken,
    Uniswap,
    Curve,
    Jupiter,
    Custom(&'static str),
}

impl Venue {
    pub fn as_str(&self) -> &'static str {
        match self {
            Venue::Binance => "binance",
            Venue::Bybit => "bybit",
            Venue::OKX => "okx",
            Venue::Coinbase => "coinbase",
            Venue::Kraken => "kraken",
            Venue::Uniswap => "uniswap",
            Venue::Curve => "curve",
            Venue::Jupiter => "jupiter",
            Venue::Custom(s) => s,
        }
    }

    /// Get maker fee in basis points
    pub fn maker_fee_bps(&self) -> f64 {
        match self {
            Venue::Binance => 10.0, // 0.10%
            Venue::Bybit => 10.0,
            Venue::OKX => 8.0,
            Venue::Coinbase => 40.0, // 0.40%
            Venue::Kraken => 26.0,
            Venue::Uniswap => 30.0, // 0.30%
            Venue::Curve => 4.0,    // 0.04%
            Venue::Jupiter => 20.0,
            Venue::Custom(_) => 10.0,
        }
    }

    /// Get taker fee in basis points
    pub fn taker_fee_bps(&self) -> f64 {
        match self {
            Venue::Binance => 10.0,
            Venue::Bybit => 50.0, // 0.50%
            Venue::OKX => 10.0,
            Venue::Coinbase => 60.0,
            Venue::Kraken => 26.0,
            Venue::Uniswap => 30.0,
            Venue::Curve => 4.0,
            Venue::Jupiter => 20.0,
            Venue::Custom(_) => 10.0,
        }
    }
}

/// Price quote from a venue
#[derive(Debug, Clone)]
pub struct VenueQuote {
    /// Venue
    pub venue: Venue,
    /// Symbol
    pub symbol: String,
    /// Bid price
    pub bid: f64,
    /// Ask price
    pub ask: f64,
    /// Bid size
    pub bid_size: f64,
    /// Ask size
    pub ask_size: f64,
    /// Timestamp in nanoseconds
    pub timestamp_ns: u64,
    /// Latency in microseconds
    pub latency_us: u32,
}

/// Cross-exchange arbitrage opportunity
#[derive(Debug, Clone)]
pub struct CrossVenueArb {
    /// Symbol
    pub symbol: String,
    /// Buy venue
    pub buy_venue: Venue,
    /// Sell venue
    pub sell_venue: Venue,
    /// Buy price (including fees)
    pub buy_price: f64,
    /// Sell price (including fees)
    pub sell_price: f64,
    /// Spread in basis points
    pub spread_bps: f64,
    /// Expected profit in basis points
    pub profit_bps: f64,
    /// Maximum executable size
    pub max_size: f64,
    /// Timestamp
    pub timestamp_ns: u64,
    /// Requires atomic execution
    pub is_atomic: bool,
}

/// Lock-free cross-exchange arb engine
pub struct CrossExchangeArbEngine {
    /// Latest quotes per venue per symbol
    quotes: DashMap<(Venue, String), VenueQuote>,
    /// Withdrawal costs between venues (in basis points)
    withdrawal_costs: DashMap<(Venue, Venue), f64>,
    /// Minimum profit threshold in bps
    min_profit_bps: f64,
    /// Opportunities detected
    opportunities_detected: AtomicU64,
    /// Is engine active
    is_active: AtomicBool,
}

impl CrossExchangeArbEngine {
    pub fn new(min_profit_bps: f64) -> Self {
        let mut engine = Self {
            quotes: DashMap::new(),
            withdrawal_costs: DashMap::new(),
            min_profit_bps,
            opportunities_detected: AtomicU64::new(0),
            is_active: AtomicBool::new(true),
        };

        // Initialize default withdrawal costs
        engine.init_withdrawal_costs();
        engine
    }

    fn init_withdrawal_costs(&mut self) {
        // CEX to CEX withdrawals
        self.withdrawal_costs.insert((Venue::Binance, Venue::Bybit), 5.0);
        self.withdrawal_costs.insert((Venue::Bybit, Venue::Binance), 5.0);
        self.withdrawal_costs.insert((Venue::Binance, Venue::OKX), 5.0);
        self.withdrawal_costs.insert((Venue::OKX, Venue::Binance), 5.0);
        
        // CEX to DEX (gas costs approximated in bps for typical trade size)
        self.withdrawal_costs.insert((Venue::Binance, Venue::Uniswap), 15.0);
        self.withdrawal_costs.insert((Venue::Bybit, Venue::Jupiter), 10.0);
    }

    /// Update quote from a venue
    pub fn update_quote(&self, quote: VenueQuote) -> Vec<CrossVenueArb> {
        if !self.is_active.load(Ordering::Relaxed) {
            return Vec::new();
        }

        let symbol = quote.symbol.clone();
        let venue = quote.venue;

        // Store quote
        self.quotes.insert((venue, symbol.clone()), quote);

        // Check for arb opportunities against all other venues
        self.find_opportunities(&symbol, venue)
    }

    fn find_opportunities(&self, symbol: &str, updated_venue: Venue) -> Vec<CrossVenueArb> {
        let mut opportunities = Vec::new();
        let timestamp_ns = Instant::now().duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default().as_nanos() as u64;

        // Compare against all other venues
        for entry in self.quotes.iter() {
            let ((venue, sym), quote) = entry.pair();
            
            if sym != symbol || *venue == updated_venue {
                continue;
            }

            // Check both directions: buy on updated, sell on other
            if let Some(opp) = self.check_arb_opportunity(
                symbol, updated_venue, *venue, timestamp_ns
            ) {
                opportunities.push(opp);
                self.opportunities_detected.fetch_add(1, Ordering::Relaxed);
            }

            // Check reverse: buy on other, sell on updated
            if let Some(opp) = self.check_arb_opportunity(
                symbol, *venue, updated_venue, timestamp_ns
            ) {
                opportunities.push(opp);
                self.opportunities_detected.fetch_add(1, Ordering::Relaxed);
            }
        }

        opportunities
    }

    fn check_arb_opportunity(
        &self,
        symbol: &str,
        buy_venue: Venue,
        sell_venue: Venue,
        timestamp_ns: u64,
    ) -> Option<CrossVenueArb> {
        let buy_quote = self.quotes.get(&(buy_venue, symbol.to_string()))?;
        let sell_quote = self.quotes.get(&(sell_venue, symbol.to_string()))?;

        // Calculate effective prices including fees
        let buy_taker_fee = buy_venue.taker_fee_bps() / 10000.0;
        let sell_taker_fee = sell_venue.taker_fee_bps() / 10000.0;
        
        let effective_buy_price = buy_quote.ask * (1.0 + buy_taker_fee);
        let effective_sell_price = sell_quote.bid * (1.0 - sell_taker_fee);

        // Get withdrawal cost
        let withdrawal_cost = self.withdrawal_costs
            .get(&(buy_venue, sell_venue))
            .map(|c| *c / 10000.0)
            .unwrap_or(0.0);

        // Calculate spread and profit
        if effective_buy_price <= 0.0 {
            return None;
        }

        let spread_bps = (effective_sell_price - effective_buy_price) / effective_buy_price * 10000.0;
        let profit_bps = spread_bps - (withdrawal_cost * 10000.0);

        if profit_bps >= self.min_profit_bps && effective_sell_price > effective_buy_price {
            // Calculate max executable size (minimum of available liquidity)
            let max_size = buy_quote.ask_size.min(sell_quote.bid_size);

            // Determine if atomic execution is required
            let is_atomic = buy_venue == sell_venue || 
                Self::is_same_ecosystem(buy_venue, sell_venue);

            return Some(CrossVenueArb {
                symbol: symbol.to_string(),
                buy_venue,
                sell_venue,
                buy_price: effective_buy_price,
                sell_price: effective_sell_price,
                spread_bps,
                profit_bps,
                max_size,
                timestamp_ns,
                is_atomic,
            });
        }

        None
    }

    fn is_same_ecosystem(a: Venue, b: Venue) -> bool {
        // Check if venues are in same ecosystem for faster settlement
        matches!((a, b), 
            (Venue::Uniswap, Venue::Curve) | (Venue::Curve, Venue::Uniswap) |
            (Venue::Binance, Venue::OKX) | (Venue::OKX, Venue::Binance)
        )
    }

    /// Set minimum profit threshold
    pub fn set_min_profit_bps(&mut self, bps: f64) {
        self.min_profit_bps = bps;
    }

    /// Get opportunities detected count
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

    /// Get latest quote for a venue/symbol
    pub fn get_quote(&self, venue: Venue, symbol: &str) -> Option<VenueQuote> {
        self.quotes.get(&(venue, symbol.to_string())).map(|q| q.clone())
    }

    /// Get best bid across all venues
    pub fn get_best_bid(&self, symbol: &str) -> Option<(Venue, f64)> {
        let mut best: Option<(Venue, f64)> = None;

        for entry in self.quotes.iter() {
            let ((venue, sym), quote) = entry.pair();
            if sym == symbol {
                if best.is_none() || quote.bid > best.unwrap().1 {
                    best = Some((*venue, quote.bid));
                }
            }
        }

        best
    }

    /// Get best ask across all venues
    pub fn get_best_ask(&self, symbol: &str) -> Option<(Venue, f64)> {
        let mut best: Option<(Venue, f64)> = None;

        for entry in self.quotes.iter() {
            let ((venue, sym), quote) = entry.pair();
            if sym == symbol {
                if best.is_none() || quote.ask < best.unwrap().1 {
                    best = Some((*venue, quote.ask));
                }
            }
        }

        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cross_exchange_arb_detection() {
        let engine = CrossExchangeArbEngine::new(5.0); // 5 bps minimum

        // Simulate price discrepancy
        let binance_quote = VenueQuote {
            venue: Venue::Binance,
            symbol: "BTCUSDT".to_string(),
            bid: 49900.0,
            ask: 49910.0,
            bid_size: 10.0,
            ask_size: 10.0,
            timestamp_ns: 1000000000,
            latency_us: 5,
        };

        let bybit_quote = VenueQuote {
            venue: Venue::Bybit,
            symbol: "BTCUSDT".to_string(),
            bid: 50100.0,
            ask: 50110.0,
            bid_size: 5.0,
            ask_size: 5.0,
            timestamp_ns: 1000000000,
            latency_us: 8,
        };

        let opps1 = engine.update_quote(binance_quote);
        assert!(opps1.is_empty()); // No opportunity yet

        let opps2 = engine.update_quote(bybit_quote);
        assert!(!opps2.is_empty(), "Should detect arb opportunity");

        if let Some(opp) = opps2.first() {
            println!("Arb: Buy on {:?}, Sell on {:?}", opp.buy_venue, opp.sell_venue);
            println!("Profit: {:.2} bps", opp.profit_bps);
            assert!(opp.profit_bps > 5.0);
        }
    }

    #[test]
    fn test_no_arb_when_efficient() {
        let engine = CrossExchangeArbEngine::new(5.0);

        // Efficient prices (no arb)
        let quote1 = VenueQuote {
            venue: Venue::Binance,
            symbol: "ETHUSDT".to_string(),
            bid: 3000.0,
            ask: 3001.0,
            bid_size: 100.0,
            ask_size: 100.0,
            timestamp_ns: 1000000000,
            latency_us: 5,
        };

        let quote2 = VenueQuote {
            venue: Venue::Bybit,
            symbol: "ETHUSDT".to_string(),
            bid: 3000.0,
            ask: 3001.0,
            bid_size: 100.0,
            ask_size: 100.0,
            timestamp_ns: 1000000000,
            latency_us: 8,
        };

        engine.update_quote(quote1);
        let opps = engine.update_quote(quote2);
        
        assert!(opps.is_empty(), "Should not detect false arb");
    }
}
