//! Market Making Module Root
//! Integrates the quoter with the pre-trade risk bus.
pub mod avellaneda;
pub mod inventory_penalty;
pub mod adv_mm_mod;

pub mod skew;
pub mod quoter;

pub use skew::{
    InventorySkewCalculator,
    ASParameters,
    SkewedQuote,
    InventoryStats,
    init_exp_lookup_table,
    fast_exp_neg,
};

pub use quoter::{
    ActiveQuoter,
    QuoteOrder,
    QuoteResult,
    QuoterStats,
    TimeInForce,
    Side,
};

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};

const CACHE_LINE_SIZE: usize = 64;

#[repr(C, align(64))]
struct CachePadded<T> {
    data: T,
    _pad: [u8; CACHE_LINE_SIZE],
}

impl<T: Default> Default for CachePadded<T> {
    fn default() -> Self {
        Self {
            data: T::default(),
            _pad: [0u8; CACHE_LINE_SIZE],
        }
    }
}

/// Pre-trade risk check result
#[derive(Debug, Clone, Copy)]
pub struct RiskCheckResult {
    pub passed: bool,
    pub reason: Option<RiskViolation>,
}

#[derive(Debug, Clone, Copy)]
pub enum RiskViolation {
    InventoryLimitExceeded,
    MaxOrderSizeExceeded,
    PriceBandsViolated,
    RateLimitExceeded,
    DailyLossLimitExceeded,
}

/// Market making engine combining skew and quoting
pub struct MarketMakingEngine {
    /// Inventory skew calculator
    skew_calc: InventorySkewCalculator,
    /// Active quoter
    quoter: ActiveQuoter,
    /// Engine active flag
    is_active: CachePadded<AtomicBool>,
    /// Order rate counter (orders per second window)
    order_rate: CachePadded<AtomicU64>,
    /// Maximum order size
    max_order_size: u64,
    /// Price band percentage (scaled by 1e6)
    price_band_bps: i64,
    /// Reference price for bands
    reference_price: CachePadded<AtomicI64>,
}

impl MarketMakingEngine {
    pub fn new(
        skew_params: ASParameters,
        tick_size: i64,
        lot_size: u64,
        maker_rebate_bps: i64,
        taker_fee_bps: i64,
        max_order_size: u64,
        price_band_bps: i64,
    ) -> Self {
        Self {
            skew_calc: InventorySkewCalculator::new(skew_params, tick_size, lot_size),
            quoter: ActiveQuoter::new(maker_rebate_bps, taker_fee_bps),
            is_active: CachePadded::new(AtomicBool::new(true)),
            order_rate: CachePadded::default(),
            max_order_size,
            price_band_bps,
            reference_price: CachePadded::default(),
        }
    }

    /// Update reference price for band calculations
    #[inline]
    pub fn update_reference_price(&self, price: i64) {
        self.reference_price.data.store(price, Ordering::Release);
    }

    /// Pre-trade risk check
    pub fn pre_trade_check(&self, price: i64, quantity: u64, side: Side) -> RiskCheckResult {
        if !self.is_active.data.load(Ordering::Acquire) {
            return RiskCheckResult {
                passed: false,
                reason: Some(RiskViolation::RateLimitExceeded),
            };
        }

        // Check order size
        if quantity > self.max_order_size {
            return RiskCheckResult {
                passed: false,
                reason: Some(RiskViolation::MaxOrderSizeExceeded),
            };
        }

        // Check price bands
        let ref_price = self.reference_price.data.load(Ordering::Acquire);
        if ref_price > 0 {
            let price_diff_bps = ((price - ref_price).abs() as i64 * 10_000) / ref_price;
            if price_diff_bps > self.price_band_bps {
                return RiskCheckResult {
                    passed: false,
                    reason: Some(RiskViolation::PriceBandsViolated),
                };
            }
        }

        // Check inventory limits
        let qty_signed = if side == Side::Buy { quantity as i64 } else { -(quantity as i64) };
        if self.skew_calc.would_breach_limit(qty_signed, side == Side::Buy) {
            return RiskCheckResult {
                passed: false,
                reason: Some(RiskViolation::InventoryLimitExceeded),
            };
        }

        RiskCheckResult { passed: true, reason: None }
    }

    /// Generate risk-compliant skewed quotes
    pub fn generate_quotes(&self, mid_price: i64, base_spread: i64) -> Option<(QuoteOrder, QuoteOrder)> {
        if !self.is_active.data.load(Ordering::Acquire) {
            return None;
        }

        // Update reference price
        self.update_reference_price(mid_price);

        // Get skewed quote prices
        let skewed = self.skew_calc.generate_skewed_quote(mid_price, base_spread);

        // Check risk for both sides
        let bid_qty = (100.0 * skewed.bid_size_factor) as u64;
        let ask_qty = (100.0 * skewed.ask_size_factor) as u64;

        let bid_risk = self.pre_trade_check(skewed.bid_price, bid_qty, Side::Buy);
        let ask_risk = self.pre_trade_check(skewed.ask_price, ask_qty, Side::Sell);

        let bid_order = if bid_risk.passed {
            Some(self.quoter.generate_post_only_quote(skewed.bid_price, bid_qty, Side::Buy))
        } else {
            None
        };

        let ask_order = if ask_risk.passed {
            Some(self.quoter.generate_post_only_quote(skewed.ask_price, ask_qty, Side::Sell))
        } else {
            None
        };

        match (bid_order, ask_order) {
            (Some(b), Some(a)) => Some((b, a)),
            (Some(b), None) => Some((b, self.quoter.generate_post_only_quote(0, 0, Side::Sell))),
            (None, Some(a)) => Some((self.quoter.generate_post_only_quote(0, 0, Side::Buy), a)),
            (None, None) => None,
        }
    }

    /// Record a fill
    pub fn record_fill(&self, quantity: u64, price: i64, is_buy: bool, is_maker: bool) -> QuoteResult {
        // Update inventory
        self.skew_calc.record_fill(quantity, is_buy);
        
        // Record for rebate/fee tracking
        self.quoter.record_fill(quantity, price, is_maker)
    }

    /// Get combined statistics
    pub fn get_stats(&self) -> MarketMakingStats {
        let skew_stats = self.skew_calc.get_stats();
        let quoter_stats = self.quoter.get_stats();

        MarketMakingStats {
            inventory: skew_stats.current_inventory,
            total_buys: skew_stats.total_buys,
            total_sells: skew_stats.total_sells,
            total_orders: quoter_stats.total_orders,
            maker_orders: quoter_stats.maker_orders,
            maker_ratio: quoter_stats.maker_ratio,
            net_rebates: quoter_stats.net_rebates,
            is_active: self.is_active.data.load(Ordering::Acquire),
        }
    }

    #[inline]
    pub fn set_active(&self, active: bool) {
        self.is_active.data.store(active, Ordering::Release);
        self.skew_calc.set_active(active);
        self.quoter.set_active(active);
    }

    #[inline]
    pub fn is_active(&self) -> bool {
        self.is_active.data.load(Ordering::Acquire)
    }

    pub fn reset(&self) {
        self.skew_calc.reset();
        self.quoter.reset();
        self.order_rate.data.store(0, Ordering::Release);
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MarketMakingStats {
    pub inventory: i64,
    pub total_buys: u64,
    pub total_sells: u64,
    pub total_orders: u64,
    pub maker_orders: u64,
    pub maker_ratio: f64,
    pub net_rebates: i64,
    pub is_active: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_market_making_engine_basic() {
        let params = ASParameters::default();
        let engine = MarketMakingEngine::new(
            params,
            1,      // tick_size
            1,      // lot_size
            100,    // maker_rebate_bps
            300,    // taker_fee_bps
            1000,   // max_order_size
            1000,   // price_band_bps (10%)
        );

        let quotes = engine.generate_quotes(10000, 20);
        assert!(quotes.is_some());

        let (bid, ask) = quotes.unwrap();
        assert!(bid.price < ask.price);
    }

    #[test]
    fn test_pre_trade_risk_check() {
        let params = ASParameters::default();
        let engine = MarketMakingEngine::new(
            params,
            1, 1, 100, 300,
            100,    // max_order_size = 100
            1000,
        );

        // Too large order
        let result = engine.pre_trade_check(10000, 150, Side::Buy);
        assert!(!result.passed);

        // Valid order
        let result = engine.pre_trade_check(10000, 50, Side::Buy);
        assert!(result.passed);
    }

    #[test]
    fn test_price_band_violation() {
        let params = ASParameters::default();
        let engine = MarketMakingEngine::new(
            params,
            1, 1, 100, 300,
            1000,
            100,    // 1% price band
        );

        engine.update_reference_price(10000);

        // Price within band (10050 is 0.5% away)
        let result = engine.pre_trade_check(10050, 50, Side::Buy);
        assert!(result.passed);

        // Price outside band (10200 is 2% away)
        let result = engine.pre_trade_check(10200, 50, Side::Buy);
        assert!(!result.passed);
    }
}
