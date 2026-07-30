//! Quoter Module - Active Quote Management
//! Manages Post-Only, IOC, and FOK order flags.
//! Ensures strict maker rebate capture while avoiding taker fees.

use std::sync::atomic::{AtomicU64, AtomicBool, AtomicI64, Ordering};
use std::time::Duration;

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

/// Order time-in-force flags
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TimeInForce {
    /// Good till cancelled
    GTC,
    /// Immediate or cancel
    IOC,
    /// Fill or kill
    FOK,
    /// Post only (maker only)
    PostOnly,
}

/// Order side
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Side {
    Buy,
    Sell,
}

/// Quote order representation
#[derive(Debug, Clone, Copy)]
pub struct QuoteOrder {
    pub order_id: u64,
    pub price: i64,
    pub quantity: u64,
    pub side: Side,
    pub tif: TimeInForce,
    pub timestamp_ns: u64,
}

/// Quote management result
#[derive(Debug, Clone, Copy)]
pub struct QuoteResult {
    pub success: bool,
    pub order_id: u64,
    pub filled_quantity: u64,
    pub remaining_quantity: u64,
    pub is_maker: bool,
    pub fee_paid: i64,
    pub rebate_earned: i64,
}

/// Lock-free active quoter
pub struct ActiveQuoter {
    /// Current bid quote
    bid_quote: CachePadded<AtomicI64>,
    /// Current ask quote  
    ask_quote: CachePadded<AtomicI64>,
    /// Bid size
    bid_size: CachePadded<AtomicU64>,
    /// Ask size
    ask_size: CachePadded<AtomicU64>,
    /// Order counter
    order_counter: CachePadded<AtomicU64>,
    /// Total maker orders
    maker_orders: CachePadded<AtomicU64>,
    /// Total taker orders (should be minimal)
    taker_orders: CachePadded<AtomicU64>,
    /// Total rebates earned (in base currency units * 1e6)
    total_rebates: CachePadded<AtomicI64>,
    /// Total fees paid
    total_fees: CachePadded<AtomicI64>,
    /// Active flag
    is_active: CachePadded<AtomicBool>,
    /// Maker rebate rate (scaled by 1e6)
    maker_rebate_bps: i64,
    /// Taker fee rate (scaled by 1e6)
    taker_fee_bps: i64,
}

impl ActiveQuoter {
    pub fn new(maker_rebate_bps: i64, taker_fee_bps: i64) -> Self {
        Self {
            bid_quote: CachePadded::default(),
            ask_quote: CachePadded::default(),
            bid_size: CachePadded::default(),
            ask_size: CachePadded::default(),
            order_counter: CachePadded::default(),
            maker_orders: CachePadded::default(),
            taker_orders: CachePadded::default(),
            total_rebates: CachePadded::default(),
            total_fees: CachePadded::default(),
            is_active: CachePadded::new(AtomicBool::new(true)),
            maker_rebate_bps,
            taker_fee_bps,
        }
    }

    /// Generate a new quote with post-only flag
    #[inline]
    pub fn generate_post_only_quote(&self, price: i64, quantity: u64, side: Side) -> QuoteOrder {
        let order_id = self.order_counter.data.fetch_add(1, Ordering::AcqRel);
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_nanos() as u64;

        let quote = QuoteOrder {
            order_id,
            price,
            quantity,
            side,
            tif: TimeInForce::PostOnly,
            timestamp_ns: now_ns,
        };

        // Update stored quotes
        match side {
            Side::Buy => {
                self.bid_quote.data.store(price, Ordering::Release);
                self.bid_size.data.store(quantity, Ordering::Release);
            }
            Side::Sell => {
                self.ask_quote.data.store(price, Ordering::Release);
                self.ask_size.data.store(quantity, Ordering::Release);
            }
        }

        quote
    }

    /// Generate IOC quote for aggressive execution
    #[inline]
    pub fn generate_ioc_quote(&self, price: i64, quantity: u64, side: Side) -> QuoteOrder {
        let order_id = self.order_counter.data.fetch_add(1, Ordering::AcqRel);
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_nanos() as u64;

        QuoteOrder {
            order_id,
            price,
            quantity,
            side,
            tif: TimeInForce::IOC,
            timestamp_ns: now_ns,
        }
    }

    /// Generate FOK quote
    #[inline]
    pub fn generate_fok_quote(&self, price: i64, quantity: u64, side: Side) -> QuoteOrder {
        let order_id = self.order_counter.data.fetch_add(1, Ordering::AcqRel);
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_nanos() as u64;

        QuoteOrder {
            order_id,
            price,
            quantity,
            side,
            tif: TimeInForce::FOK,
            timestamp_ns: now_ns,
        }
    }

    /// Record a fill and calculate rebate/fee
    #[inline]
    pub fn record_fill(&self, quantity: u64, price: i64, is_maker: bool) -> QuoteResult {
        let notional = (quantity as i64) * price;
        
        let (fee, rebate) = if is_maker {
            self.maker_orders.data.fetch_add(1, Ordering::AcqRel);
            let rebate = (notional * self.maker_rebate_bps) / 1_000_000;
            self.total_rebates.data.fetch_add(rebate, Ordering::AcqRel);
            (0, rebate)
        } else {
            self.taker_orders.data.fetch_add(1, Ordering::AcqRel);
            let fee = (notional * self.taker_fee_bps) / 1_000_000;
            self.total_fees.data.fetch_add(fee, Ordering::AcqRel);
            (fee, 0)
        };

        QuoteResult {
            success: true,
            order_id: 0,
            filled_quantity: quantity,
            remaining_quantity: 0,
            is_maker,
            fee_paid: fee,
            rebate_earned: rebate,
        }
    }

    /// Check if quote would cross the spread (and thus not be maker)
    #[inline]
    pub fn would_cross_spread(&self, price: i64, side: Side, opposite_price: i64) -> bool {
        match side {
            Side::Buy => price >= opposite_price,
            Side::Sell => price <= opposite_price,
        }
    }

    /// Get current best bid
    #[inline]
    pub fn get_bid(&self) -> (i64, u64) {
        (
            self.bid_quote.data.load(Ordering::Acquire),
            self.bid_size.data.load(Ordering::Acquire),
        )
    }

    /// Get current best ask
    #[inline]
    pub fn get_ask(&self) -> (i64, u64) {
        (
            self.ask_quote.data.load(Ordering::Acquire),
            self.ask_size.data.load(Ordering::Acquire),
        )
    }

    /// Get quoter statistics
    pub fn get_stats(&self) -> QuoterStats {
        QuoterStats {
            total_orders: self.order_counter.data.load(Ordering::Acquire),
            maker_orders: self.maker_orders.data.load(Ordering::Acquire),
            taker_orders: self.taker_orders.data.load(Ordering::Acquire),
            maker_ratio: {
                let total = self.order_counter.data.load(Ordering::Acquire);
                if total > 0 {
                    self.maker_orders.data.load(Ordering::Acquire) as f64 / total as f64
                } else {
                    0.0
                }
            },
            total_rebates: self.total_rebates.data.load(Ordering::Acquire),
            total_fees: self.total_fees.data.load(Ordering::Acquire),
            net_rebates: self.total_rebates.data.load(Ordering::Acquire) 
                - self.total_fees.data.load(Ordering::Acquire),
            is_active: self.is_active.data.load(Ordering::Acquire),
        }
    }

    #[inline]
    pub fn set_active(&self, active: bool) {
        self.is_active.data.store(active, Ordering::Release);
    }

    #[inline]
    pub fn reset(&self) {
        self.order_counter.data.store(0, Ordering::Release);
        self.maker_orders.data.store(0, Ordering::Release);
        self.taker_orders.data.store(0, Ordering::Release);
        self.total_rebates.data.store(0, Ordering::Release);
        self.total_fees.data.store(0, Ordering::Release);
    }
}

#[derive(Debug, Clone, Copy)]
pub struct QuoterStats {
    pub total_orders: u64,
    pub maker_orders: u64,
    pub taker_orders: u64,
    pub maker_ratio: f64,
    pub total_rebates: i64,
    pub total_fees: i64,
    pub net_rebates: i64,
    pub is_active: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_post_only_quote() {
        let quoter = ActiveQuoter::new(10, 30);
        let quote = quoter.generate_post_only_quote(10000, 100, Side::Buy);
        
        assert_eq!(quote.tif, TimeInForce::PostOnly);
        assert_eq!(quote.price, 10000);
        assert_eq!(quote.quantity, 100);
    }

    #[test]
    fn test_maker_rebate_calculation() {
        let quoter = ActiveQuoter::new(100, 300); // 0.01% rebate, 0.03% fee
        
        let result = quoter.record_fill(100, 10000, true);
        assert!(result.is_maker);
        assert!(result.rebate_earned > 0);
        assert_eq!(result.fee_paid, 0);
    }

    #[test]
    fn test_taker_fee_calculation() {
        let quoter = ActiveQuoter::new(100, 300);
        
        let result = quoter.record_fill(100, 10000, false);
        assert!(!result.is_maker);
        assert!(result.fee_paid > 0);
        assert_eq!(result.rebate_earned, 0);
    }

    #[test]
    fn test_spread_crossing() {
        let quoter = ActiveQuoter::new(100, 300);
        
        // Bid at 9990, asking at 10000
        // Buying at 10000 would cross
        assert!(quoter.would_cross_spread(10000, Side::Buy, 10000));
        assert!(!quoter.would_cross_spread(9990, Side::Buy, 10000));
        
        // Selling at 9990 would cross
        assert!(quoter.would_cross_spread(9990, Side::Sell, 10000));
        assert!(!quoter.would_cross_spread(10010, Side::Sell, 10000));
    }
}
