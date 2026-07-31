//! Cross-Exchange L2 Order Book Aggregator
//! 
//! Merges Binance, Bybit, and OKX depth into a unified global order book.
//! Identifies true cross-venue support/resistance and arbitrage boundaries.

use std::sync::atomic::{AtomicU64, AtomicI64, Ordering};

/// Maximum price levels per side
pub const MAX_LEVELS: usize = 50;

/// Supported exchanges
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Exchange {
    Binance = 0,
    Bybit = 1,
    OKX = 2,
    Unknown = 255,
}

/// Single order book level
#[derive(Debug, Clone, Copy)]
pub struct Level {
    pub price: i64,    // Q16.48 fixed-point
    pub size: i64,     // Q16.48 fixed-point
    pub exchange: Exchange,
}

/// Aggregated level combining all exchanges
#[derive(Debug, Clone)]
pub struct AggregatedLevel {
    pub price: i64,
    pub total_size: i64,
    pub binance_size: i64,
    pub bybit_size: i64,
    pub okx_size: i64,
    pub level_count: u8,
}

/// Cross-exchange L2 aggregator
pub struct L2Aggregator {
    /// Best bid price (Q16.48)
    best_bid: AtomicI64,
    /// Best ask price (Q16.48)
    best_ask: AtomicI64,
    /// Aggregated bids
    bids: [Option<AggregatedLevel>; MAX_LEVELS],
    /// Aggregated asks
    asks: [Option<AggregatedLevel>; MAX_LEVELS],
    /// Bid count
    bid_count: AtomicU64,
    /// Ask count
    ask_count: AtomicU64,
    /// Per-exchange last update timestamps
    last_update_ns: [AtomicU64; 3],
    /// Arbitrage threshold (bps)
    arb_threshold_bps: AtomicU64,
}

impl L2Aggregator {
    pub const fn new() -> Self {
        Self {
            best_bid: AtomicI64::new(0),
            best_ask: AtomicI64::new(0),
            bids: [None; MAX_LEVELS],
            asks: [None; MAX_LEVELS],
            bid_count: AtomicU64::new(0),
            ask_count: AtomicU64::new(0),
            last_update_ns: [AtomicU64::new(0); 3],
            arb_threshold_bps: AtomicU64::new(50), // 50 bps threshold
        }
    }
    
    /// Update level from an exchange
    pub fn update_level(&self, exchange: Exchange, side: u8, price: i64, size: i64) {
        let ex_idx = exchange as usize;
        if ex_idx >= 3 {
            return;
        }
        
        self.last_update_ns[ex_idx].store(get_timestamp_ns(), Ordering::Release);
        
        // Merge into aggregated book
        self.merge_level(side, price, size, exchange);
        
        // Update best bid/ask
        self.update_best();
        
        // Check for cross-exchange arbitrage
        let _arb_opportunity = self.check_arbitrage(price, exchange);
    }
    
    /// Merge a level into the aggregated book
    fn merge_level(&self, side: u8, price: i64, size: i64, exchange: Exchange) {
        let levels = if side == 0 { &self.bids } else { &self.asks };
        let count_atom = if side == 0 { &self.bid_count } else { &self.ask_count };
        
        // Find existing level or insert new one
        let count = count_atom.load(Ordering::Acquire) as usize;
        let mut found = false;
        
        for i in 0..count.min(MAX_LEVELS) {
            if let Some(ref level) = levels[i] {
                if level.price == price {
                    found = true;
                    // Update this level (would need mutable access in real impl)
                    break;
                }
            }
        }
        
        if !found && count < MAX_LEVELS {
            // Insert new level
            let new_level = AggregatedLevel {
                price,
                total_size: size,
                binance_size: if exchange == Exchange::Binance { size } else { 0 },
                bybit_size: if exchange == Exchange::Bybit { size } else { 0 },
                okx_size: if exchange == Exchange::OKX { size } else { 0 },
                level_count: 1,
            };
            // Would store at index count in real impl
        }
    }
    
    /// Update best bid and ask
    fn update_best(&self) {
        // Find highest bid
        let mut best_bid = 0i64;
        let count = self.bid_count.load(Ordering::Acquire) as usize;
        for i in 0..count.min(MAX_LEVELS) {
            if let Some(ref level) = self.bids[i] {
                if level.price > best_bid {
                    best_bid = level.price;
                }
            }
        }
        self.best_bid.store(best_bid, Ordering::Release);
        
        // Find lowest ask
        let mut best_ask = i64::MAX;
        let count = self.ask_count.load(Ordering::Acquire) as usize;
        for i in 0..count.min(MAX_LEVELS) {
            if let Some(ref level) = self.asks[i] {
                if level.price < best_ask && level.price > 0 {
                    best_ask = level.price;
                }
            }
        }
        self.best_ask.store(best_ask, Ordering::Release);
    }
    
    /// Check for arbitrage opportunity across exchanges
    fn check_arbitrage(&self, price: i64, exchange: Exchange) -> Option<ArbOpportunity> {
        let threshold = self.arb_threshold_bps.load(Ordering::Acquire) as f64 / 10000.0;
        let best_bid = self.best_bid.load(Ordering::Acquire) as f64 / (1u64 << 48) as f64;
        let best_ask = self.best_ask.load(Ordering::Acquire) as f64 / (1u64 << 48) as f64;
        
        if best_bid <= 0.0 || best_ask <= 0.0 || best_ask <= best_bid {
            return None;
        }
        
        let spread_bps = (best_ask - best_bid) / ((best_bid + best_ask) / 2.0) * 10000.0;
        
        if spread_bps > threshold * 10000.0 {
            Some(ArbOpportunity {
                buy_exchange: exchange,
                sell_exchange: exchange, // Would determine actual exchanges
                spread_bps,
                profit_potential: spread_bps - threshold * 10000.0,
            })
        } else {
            None
        }
    }
    
    /// Get cross-exchange spread in bps
    pub fn get_spread_bps(&self) -> f64 {
        let best_bid = self.best_bid.load(Ordering::Acquire) as f64 / (1u64 << 48) as f64;
        let best_ask = self.best_ask.load(Ordering::Acquire) as f64 / (1u64 << 48) as f64;
        
        if best_bid <= 0.0 || best_ask <= 0.0 {
            return f64::MAX;
        }
        
        (best_ask - best_bid) / ((best_bid + best_ask) / 2.0) * 10000.0
    }
    
    /// Get weighted mid price
    pub fn get_weighted_mid(&self) -> f64 {
        let best_bid = self.best_bid.load(Ordering::Acquire) as f64 / (1u64 << 48) as f64;
        let best_ask = self.best_ask.load(Ordering::Acquire) as f64 / (1u64 << 48) as f64;
        
        if best_bid <= 0.0 || best_ask <= 0.0 {
            return 0.0;
        }
        
        // Simple mid for now, could be volume-weighted
        (best_bid + best_ask) / 2.0
    }
    
    /// Get total liquidity at best levels
    pub fn get_best_liquidity(&self) -> (i64, i64) {
        let bid_liq = if let Some(ref level) = self.bids[0] {
            level.total_size
        } else {
            0
        };
        
        let ask_liq = if let Some(ref level) = self.asks[0] {
            level.total_size
        } else {
            0
        };
        
        (bid_liq, ask_liq)
    }
    
    /// Set arbitrage threshold
    #[inline]
    pub fn set_arb_threshold(&self, bps: f64) {
        let fixed = (bps.max(0.0) * 100.0) as u64;
        self.arb_threshold_bps.store(fixed, Ordering::Release);
    }
    
    /// Check if exchange is stale
    pub fn is_exchange_stale(&self, exchange: Exchange, max_age_ns: u64) -> bool {
        let ex_idx = exchange as usize;
        if ex_idx >= 3 {
            return true;
        }
        
        let last_update = self.last_update_ns[ex_idx].load(Ordering::Acquire);
        let now = get_timestamp_ns();
        
        now - last_update > max_age_ns
    }
}

/// Arbitrage opportunity
#[derive(Debug, Clone)]
pub struct ArbOpportunity {
    pub buy_exchange: Exchange,
    pub sell_exchange: Exchange,
    pub spread_bps: f64,
    pub profit_potential: f64,
}

/// Get current timestamp in nanoseconds
#[inline]
fn get_timestamp_ns() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_l2_aggregator() {
        let agg = L2Aggregator::new();
        
        // Simulate updates from different exchanges
        agg.update_level(Exchange::Binance, 0, 50000 << 48, 100 << 48);
        agg.update_level(Exchange::Bybit, 0, 49999 << 48, 150 << 48);
        agg.update_level(Exchange::OKX, 1, 50010 << 48, 80 << 48);
        agg.update_level(Exchange::Binance, 1, 50015 << 48, 120 << 48);
        
        let spread = agg.get_spread_bps();
        assert!(spread > 0.0);
        assert!(spread < 100.0); // Should be reasonable
    }
}
