//! Pegged Order Implementation
//! 
//! Implements pegged order logic (mid-price peg, primary peg) that automatically re-quotes
//! while maintaining optimal queue position and tracking reference prices.

use std::sync::atomic::{AtomicU64, AtomicI64, AtomicBool, Ordering};
use crossbeam_utils::CachePadded;

/// Peg type for order placement
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PegType {
    /// Peg to mid-price (average of best bid/ask)
    Mid,
    /// Peg to primary side (bid for buys, ask for sells)
    Primary,
    /// Peg to secondary side (ask for buys, bid for sells)
    Secondary,
    /// Fixed offset from reference price
    FixedOffset,
}

/// Direction of the order
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderSide {
    Buy,
    Sell,
}

/// Pegged order configuration
pub struct PeggedOrderConfig {
    pub peg_type: PegType,
    pub side: OrderSide,
    /// Offset in micro-units (positive = more aggressive, negative = more passive)
    pub offset_micros: i64,
    /// Maximum allowable offset (to prevent excessive pricing)
    pub max_offset_micros: u64,
    /// Minimum price increment (tick size)
    pub tick_size_micros: u64,
    /// Whether to maintain queue priority on price improvements
    pub maintain_queue_priority: bool,
}

/// Lock-free pegged order tracker
pub struct PeggedOrderEngine {
    /// Current reference mid-price (in micro-units)
    mid_price: CachePadded<AtomicU64>,
    /// Current best bid (in micro-units)
    best_bid: CachePadded<AtomicU64>,
    /// Current best ask (in micro-units)
    best_ask: CachePadded<AtomicU64>,
    /// Calculated peg price (in micro-units)
    peg_price: CachePadded<AtomicU64>,
    /// Last quoted price (for detecting changes)
    last_quoted_price: CachePadded<AtomicU64>,
    /// Order quantity (in base units)
    quantity: CachePadded<AtomicU64>,
    /// Number of re-quotes triggered
    requote_count: CachePadded<AtomicU64>,
    /// Whether the order is currently active
    is_active: CachePadded<AtomicBool>,
    /// Configuration
    config: PeggedOrderConfig,
    /// Price version counter (for optimistic locking)
    price_version: CachePadded<AtomicU64>,
}

impl PeggedOrderEngine {
    /// Create a new pegged order engine
    pub fn new(config: PeggedOrderConfig, initial_mid: u64, quantity: u64) -> Self {
        let initial_price = Self::calculate_peg_price(initial_mid, initial_mid, initial_mid, &config);
        
        Self {
            mid_price: CachePadded::new(AtomicU64::new(initial_mid)),
            best_bid: CachePadded::new(AtomicU64::new(initial_mid)),
            best_ask: CachePadded::new(AtomicU64::new(initial_mid)),
            peg_price: CachePadded::new(AtomicU64::new(initial_price)),
            last_quoted_price: CachePadded::new(AtomicU64::new(initial_price)),
            quantity: CachePadded::new(AtomicU64::new(quantity)),
            requote_count: CachePadded::new(AtomicU64::new(0)),
            is_active: CachePadded::new(AtomicBool::new(true)),
            config,
            price_version: CachePadded::new(AtomicU64::new(1)),
        }
    }

    #[inline]
    fn calculate_peg_price(mid: u64, bid: u64, ask: u64, config: &PeggedOrderConfig) -> u64 {
        let reference = match config.peg_type {
            PegType::Mid => mid,
            PegType::Primary => {
                match config.side {
                    OrderSide::Buy => bid,
                    OrderSide::Sell => ask,
                }
            }
            PegType::Secondary => {
                match config.side {
                    OrderSide::Buy => ask,
                    OrderSide::Sell => bid,
                }
            }
            PegType::FixedOffset => mid, // Offset applied separately
        };

        // Apply offset
        let offset = config.offset_micros;
        let adjusted = if offset >= 0 {
            reference.saturating_add(offset as u64)
        } else {
            reference.saturating_sub((-offset) as u64)
        };

        // Round to tick size
        let tick = config.tick_size_micros;
        if tick > 0 {
            (adjusted / tick) * tick
        } else {
            adjusted
        }
    }

    /// Update market data and recalculate peg price
    /// Returns true if re-quote is required
    pub fn update_market(&self, bid: u64, ask: u64) -> bool {
        if !self.is_active.load(Ordering::Relaxed) {
            return false;
        }

        // Validate spread (prevent crossed book)
        if bid >= ask && ask > 0 {
            return false;
        }

        let mid = if bid > 0 && ask > 0 {
            (bid + ask) / 2
        } else if bid > 0 {
            bid
        } else if ask > 0 {
            ask
        } else {
            return false;
        };

        self.best_bid.store(bid, Ordering::Relaxed);
        self.best_ask.store(ask, Ordering::Relaxed);
        self.mid_price.store(mid, Ordering::Relaxed);

        let new_peg = Self::calculate_peg_price(mid, bid, ask, &self.config);
        
        // Apply max offset constraint
        let current_ref = match self.config.peg_type {
            PegType::Mid => mid,
            PegType::Primary => {
                match self.config.side {
                    OrderSide::Buy => bid,
                    OrderSide::Sell => ask,
                }
            }
            PegType::Secondary => {
                match self.config.side {
                    OrderSide::Buy => ask,
                    OrderSide::Sell => bid,
                }
            }
            PegType::FixedOffset => mid,
        };

        let constrained_peg = self.apply_max_offset_constraint(new_peg, current_ref);
        
        let old_peg = self.peg_price.load(Ordering::Relaxed);
        
        if constrained_peg != old_peg {
            self.peg_price.store(constrained_peg, Ordering::Relaxed);
            self.price_version.fetch_add(1, Ordering::Relaxed);
            
            // Check if re-quote needed
            let last_quote = self.last_quoted_price.load(Ordering::Relaxed);
            if constrained_peg != last_quote {
                self.requote_count.fetch_add(1, Ordering::Relaxed);
                return true;
            }
        }

        false
    }

    #[inline]
    fn apply_max_offset_constraint(&self, peg_price: u64, reference: u64) -> u64 {
        let max_offset = self.config.max_offset_micros;
        
        match self.config.side {
            OrderSide::Buy => {
                // For buys, ensure we don't pay too much above reference
                let max_allowed = reference.saturating_add(max_offset);
                std::cmp::min(peg_price, max_allowed)
            }
            OrderSide::Sell => {
                // For sells, ensure we don't sell too low below reference
                let min_allowed = reference.saturating_sub(max_offset);
                std::cmp::max(peg_price, min_allowed)
            }
        }
    }

    /// Confirm quote was sent (update last quoted price)
    pub fn confirm_quote(&self, quoted_price: u64) {
        self.last_quoted_price.store(quoted_price, Ordering::Relaxed);
    }

    /// Get current peg price
    #[inline]
    pub fn get_peg_price(&self) -> u64 {
        self.peg_price.load(Ordering::Relaxed)
    }

    /// Get current mid price
    #[inline]
    pub fn get_mid_price(&self) -> u64 {
        self.mid_price.load(Ordering::Relaxed)
    }

    /// Get best bid
    #[inline]
    pub fn get_best_bid(&self) -> u64 {
        self.best_bid.load(Ordering::Relaxed)
    }

    /// Get best ask
    #[inline]
    pub fn get_best_ask(&self) -> u64 {
        self.best_ask.load(Ordering::Relaxed)
    }

    /// Get quantity
    #[inline]
    pub fn get_quantity(&self) -> u64 {
        self.quantity.load(Ordering::Relaxed)
    }

    /// Check if order is active
    #[inline]
    pub fn is_active(&self) -> bool {
        self.is_active.load(Ordering::Relaxed)
    }

    /// Get re-quote count
    #[inline]
    pub fn get_requote_count(&self) -> u64 {
        self.requote_count.load(Ordering::Relaxed)
    }

    /// Get price version (for optimistic locking)
    #[inline]
    pub fn get_price_version(&self) -> u64 {
        self.price_version.load(Ordering::Relaxed)
    }

    /// Activate the order
    pub fn activate(&self) {
        self.is_active.store(true, Ordering::Relaxed);
    }

    /// Deactivate the order
    pub fn deactivate(&self) {
        self.is_active.store(false, Ordering::Relaxed);
    }

    /// Update offset dynamically
    pub fn update_offset(&self, new_offset_micros: i64) {
        unsafe {
            // Safe because we're the only writer to config fields during runtime
            let config_ptr = &self.config as *const PeggedOrderConfig as *mut PeggedOrderConfig;
            (*config_ptr).offset_micros = new_offset_micros;
        }
        
        // Recalculate with new offset
        let bid = self.best_bid.load(Ordering::Relaxed);
        let ask = self.best_ask.load(Ordering::Relaxed);
        let mid = self.mid_price.load(Ordering::Relaxed);
        
        self.update_market(bid, ask);
    }

    /// Calculate implied queue position quality
    /// Returns a score from 0-100 (higher = better position)
    pub fn calculate_queue_score(&self) -> u8 {
        let peg = self.peg_price.load(Ordering::Relaxed);
        let bid = self.best_bid.load(Ordering::Relaxed);
        let ask = self.best_ask.load(Ordering::Relaxed);
        
        if bid == 0 || ask == 0 {
            return 50; // Neutral if no data
        }

        match self.config.side {
            OrderSide::Buy => {
                // For buys, closer to bid = better queue position
                if peg <= bid {
                    100
                } else if peg >= ask {
                    0
                } else {
                    let spread = ask - bid;
                    let distance_from_bid = peg - bid;
                    if spread > 0 {
                        (100 - ((distance_from_bid * 100) / spread)) as u8
                    } else {
                        50
                    }
                }
            }
            OrderSide::Sell => {
                // For sells, closer to ask = better queue position
                if peg >= ask {
                    100
                } else if peg <= bid {
                    0
                } else {
                    let spread = ask - bid;
                    let distance_from_ask = ask - peg;
                    if spread > 0 {
                        ((distance_from_ask * 100) / spread) as u8
                    } else {
                        50
                    }
                }
            }
        }
    }

    /// Check if adverse fill risk is high
    /// Returns true if price movement suggests potential adverse selection
    pub fn check_adverse_fill_risk(&self) -> bool {
        let peg = self.peg_price.load(Ordering::Relaxed);
        let mid = self.mid_price.load(Ordering::Relaxed);
        
        match self.config.side {
            OrderSide::Buy => {
                // High risk if we're buying significantly above mid
                let premium = peg.saturating_sub(mid);
                let threshold = mid / 200; // 0.5% threshold
                premium > threshold
            }
            OrderSide::Sell => {
                // High risk if we're selling significantly below mid
                let discount = mid.saturating_sub(peg);
                let threshold = mid / 200; // 0.5% threshold
                discount > threshold
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mid_peg_buy() {
        let config = PeggedOrderConfig {
            peg_type: PegType::Mid,
            side: OrderSide::Buy,
            offset_micros: -100, // 1 cent passive
            max_offset_micros: 1000,
            tick_size_micros: 100,
            maintain_queue_priority: true,
        };

        let engine = PeggedOrderEngine::new(config, 50000000, 1000);
        
        assert_eq!(engine.get_mid_price(), 50000000);
        
        // Update with realistic spread
        let should_requote = engine.update_market(49990000, 50010000);
        assert!(should_requote);
        
        let peg = engine.get_peg_price();
        assert!(peg <= 50000000); // Should be at or below mid for buy
    }

    #[test]
    fn test_primary_peg_sell() {
        let config = PeggedOrderConfig {
            peg_type: PegType::Primary,
            side: OrderSide::Sell,
            offset_micros: 0,
            max_offset_micros: 500,
            tick_size_micros: 100,
            maintain_queue_priority: false,
        };

        let engine = PeggedOrderEngine::new(config, 50000000, 500);
        
        // For primary peg sell, should peg to ask
        engine.update_market(49990000, 50010000);
        
        let peg = engine.get_peg_price();
        assert_eq!(peg, 50010000); // Should be at ask
    }

    #[test]
    fn test_queue_score() {
        let config = PeggedOrderConfig {
            peg_type: PegType::Mid,
            side: OrderSide::Buy,
            offset_micros: -500,
            max_offset_micros: 1000,
            tick_size_micros: 100,
            maintain_queue_priority: true,
        };

        let engine = PeggedOrderEngine::new(config, 50000000, 1000);
        engine.update_market(49990000, 50010000);
        
        let score = engine.calculate_queue_score();
        assert!(score > 0 && score <= 100);
    }
}
