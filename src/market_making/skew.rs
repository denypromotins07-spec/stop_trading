//! Inventory Skew Module - Avellaneda-Stoikov Market Making
//! Implements inventory risk penalization using optimized fixed-point math.
//! Adjusts bid/ask quotes dynamically based on current portfolio delta to prevent toxic accumulation.

use std::sync::atomic::{AtomicI64, AtomicU64, AtomicBool, Ordering};

/// Cache line padding for AMD Ryzen cores
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

/// Fixed-point representation for fast math (scaled by 1e6)
const FIXED_POINT_SCALE: i64 = 1_000_000;

/// Pre-computed lookup table for exponential approximation
/// Contains exp(-x) values for x in [0, 10] with step 0.01
const EXP_TABLE_SIZE: usize = 1001;
static EXP_LOOKUP: [i64; EXP_TABLE_SIZE] = [0; EXP_TABLE_SIZE]; // Placeholder - initialized at runtime

/// Avellaneda-Stoikov model parameters
#[derive(Debug, Clone, Copy)]
pub struct ASParameters {
    /// Risk aversion coefficient (gamma)
    pub gamma: f64,
    /// Volatility estimate (sigma)
    pub sigma: f64,
    /// Time horizon in seconds
    pub time_horizon: f64,
    /// Order book liquidity parameter (kappa)
    pub kappa: f64,
    /// Maker rebate (as fraction)
    pub maker_rebate: f64,
    /// Maximum inventory position
    pub max_inventory: i64,
    /// Skew multiplier for aggressive adjustment
    pub skew_multiplier: f64,
}

impl Default for ASParameters {
    fn default() -> Self {
        Self {
            gamma: 0.1,
            sigma: 0.02,
            time_horizon: 60.0,
            kappa: 1.0,
            maker_rebate: 0.0001,
            max_inventory: 1000,
            skew_multiplier: 1.0,
        }
    }
}

/// Quote adjustment result
#[derive(Debug, Clone, Copy)]
pub struct SkewedQuote {
    /// Original mid price
    pub mid_price: i64,
    /// Adjusted bid price
    pub bid_price: i64,
    /// Adjusted ask price
    pub ask_price: i64,
    /// Bid size adjustment factor
    pub bid_size_factor: f64,
    /// Ask size adjustment factor
    pub ask_size_factor: f64,
    /// Current inventory
    pub inventory: i64,
    /// Reservation price (fair value given inventory)
    pub reservation_price: i64,
    /// Spread in ticks
    pub spread_ticks: i64,
    /// Timestamp (nanoseconds)
    pub timestamp_ns: u64,
}

/// Lock-free inventory skew calculator
pub struct InventorySkewCalculator {
    /// Current inventory position
    inventory: CachePadded<AtomicI64>,
    /// Total buy volume executed
    total_buys: CachePadded<AtomicU64>,
    /// Total sell volume executed
    total_sells: CachePadded<AtomicU64>,
    /// Parameters
    params: ASParameters,
    /// Active flag
    is_active: CachePadded<AtomicBool>,
    /// Tick size for the instrument
    tick_size: i64,
    /// Lot size for the instrument
    lot_size: u64,
}

impl InventorySkewCalculator {
    /// Create new skew calculator with given parameters
    pub fn new(params: ASParameters, tick_size: i64, lot_size: u64) -> Self {
        Self {
            inventory: CachePadded::default(),
            total_buys: CachePadded::default(),
            total_sells: CachePadded::default(),
            params,
            is_active: CachePadded::new(AtomicBool::new(true)),
            tick_size,
            lot_size,
        }
    }

    /// Update inventory position atomically
    #[inline]
    pub fn update_inventory(&self, new_inventory: i64) {
        self.inventory.data.store(new_inventory, Ordering::Release);
    }

    /// Record a fill and update inventory
    #[inline]
    pub fn record_fill(&self, quantity: u64, is_buy: bool) {
        let current = self.inventory.data.load(Ordering::Acquire);
        let new_inventory = if is_buy {
            current + quantity as i64
        } else {
            current - quantity as i64
        };
        
        self.inventory.data.store(new_inventory, Ordering::Release);
        
        if is_buy {
            self.total_buys.data.fetch_add(quantity, Ordering::AcqRel);
        } else {
            self.total_sells.data.fetch_add(quantity, Ordering::AcqRel);
        }
    }

    /// Calculate reservation price using Avellaneda-Stoikov formula
    /// r = s - q * gamma * sigma^2 * T
    /// where s = mid price, q = inventory, gamma = risk aversion, sigma = vol, T = time horizon
    #[inline]
    pub fn calculate_reservation_price(&self, mid_price: i64) -> i64 {
        let inventory = self.inventory.data.load(Ordering::Acquire);
        
        // Using fixed-point arithmetic for speed
        // reservation_adjustment = inventory * gamma * sigma^2 * T
        let gamma_fp = (self.params.gamma * FIXED_POINT_SCALE as f64) as i64;
        let sigma_sq_fp = ((self.params.sigma * self.params.sigma) * FIXED_POINT_SCALE as f64) as i64;
        let time_fp = (self.params.time_horizon * FIXED_POINT_SCALE as f64) as i64;
        
        // Multiply in fixed point, then scale back
        let adjustment_fp = (inventory as i64)
            .wrapping_mul(gamma_fp)
            .wrapping_mul(sigma_sq_fp)
            .wrapping_mul(time_fp);
        
        // Scale back: divide by FIXED_POINT_SCALE^3
        let adjustment = adjustment_fp / (FIXED_POINT_SCALE * FIXED_POINT_SCALE * FIXED_POINT_SCALE);
        
        mid_price - adjustment
    }

    /// Calculate optimal spread using AS formula
    /// delta = 1/gamma * ln(1 + gamma/kappa)
    #[inline]
    pub fn calculate_optimal_spread(&self) -> f64 {
        let gamma = self.params.gamma;
        let kappa = self.params.kappa;
        
        if gamma <= 0.0 || kappa <= 0.0 {
            return self.tick_size as f64 * 2.0; // Fallback to minimum spread
        }
        
        // Using Taylor series approximation for ln(1 + x) where x = gamma/kappa
        let x = gamma / kappa;
        let ln_approx = if x < 0.5 {
            // Taylor: ln(1+x) ≈ x - x^2/2 + x^3/3 - x^4/4
            x - x*x/2.0 + x*x*x/3.0 - x*x*x*x/4.0
        } else if x < 2.0 {
            // For larger x, use a different approximation
            // ln(1+x) ≈ 2 * artanh(x/(2+x)) for better convergence
            let y = x / (2.0 + x);
            2.0 * (y + y*y*y/3.0 + y*y*y*y*y/5.0)
        } else {
            // For very large x, ln(1+x) ≈ ln(x) + 1/x
            x.ln() + 1.0/x
        };
        
        (1.0 / gamma) * ln_approx
    }

    /// Generate skewed quotes based on current inventory
    pub fn generate_skewed_quote(&self, mid_price: i64, base_spread: i64) -> SkewedQuote {
        if !self.is_active.data.load(Ordering::Acquire) {
            return self.generate_neutral_quote(mid_price, base_spread);
        }

        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or(std::time::Duration::ZERO)
            .as_nanos() as u64;

        let inventory = self.inventory.data.load(Ordering::Acquire);
        let reservation_price = self.calculate_reservation_price(mid_price);
        
        // Calculate spread adjustment based on inventory
        let inventory_ratio = inventory as f64 / self.params.max_inventory as f64;
        let skew_factor = inventory_ratio * self.params.skew_multiplier;
        
        // Base spread from AS model
        let optimal_spread = self.calculate_optimal_spread();
        let spread_ticks = ((optimal_spread / self.tick_size as f64).ceil() as i64).max(self.tick_size);
        
        // Adjust bid/ask around reservation price
        let half_spread = (spread_ticks * self.tick_size) / 2;
        
        // Apply inventory skew
        // Long inventory: lower bid, raise ask (want to sell)
        // Short inventory: raise bid, lower ask (want to buy)
        let skew_adjustment = (skew_factor * half_spread as f64) as i64;
        
        let mut bid_price = reservation_price - half_spread - skew_adjustment;
        let mut ask_price = reservation_price + half_spread - skew_adjustment;
        
        // Round to tick size
        bid_price = (bid_price / self.tick_size) * self.tick_size;
        ask_price = ((ask_price + self.tick_size - 1) / self.tick_size) * self.tick_size;
        
        // Ensure bid < ask
        if bid_price >= ask_price {
            ask_price = bid_price + self.tick_size;
        }
        
        // Calculate size adjustment factors
        // Reduce size on side we want to discourage
        let bid_size_factor = if inventory > 0 {
            // Long inventory: reduce bid size
            (1.0 - skew_factor.abs()).max(0.1)
        } else {
            1.0
        };
        
        let ask_size_factor = if inventory < 0 {
            // Short inventory: reduce ask size
            (1.0 - skew_factor.abs()).max(0.1)
        } else {
            1.0
        };

        SkewedQuote {
            mid_price,
            bid_price,
            ask_price,
            bid_size_factor,
            ask_size_factor,
            inventory,
            reservation_price,
            spread_ticks: (ask_price - bid_price) / self.tick_size,
            timestamp_ns: now_ns,
        }
    }

    /// Generate neutral quote (no inventory skew)
    fn generate_neutral_quote(&self, mid_price: i64, base_spread: i64) -> SkewedQuote {
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or(std::time::Duration::ZERO)
            .as_nanos() as u64;

        let half_spread = base_spread / 2;
        let bid_price = ((mid_price - half_spread) / self.tick_size) * self.tick_size;
        let ask_price = (((mid_price + half_spread + self.tick_size - 1) / self.tick_size) * self.tick_size)
            .max(bid_price + self.tick_size);

        SkewedQuote {
            mid_price,
            bid_price,
            ask_price,
            bid_size_factor: 1.0,
            ask_size_factor: 1.0,
            inventory: self.inventory.data.load(Ordering::Acquire),
            reservation_price: mid_price,
            spread_ticks: (ask_price - bid_price) / self.tick_size,
            timestamp_ns: now_ns,
        }
    }

    /// Check if inventory limit would be breached by a trade
    #[inline]
    pub fn would_breach_limit(&self, quantity: i64, is_buy: bool) -> bool {
        let current = self.inventory.data.load(Ordering::Acquire);
        let new_inventory = if is_buy {
            current + quantity
        } else {
            current - quantity
        };
        
        new_inventory.abs() > self.params.max_inventory
    }

    /// Get current inventory
    #[inline]
    pub fn get_inventory(&self) -> i64 {
        self.inventory.data.load(Ordering::Acquire)
    }

    /// Get inventory statistics
    pub fn get_stats(&self) -> InventoryStats {
        InventoryStats {
            current_inventory: self.inventory.data.load(Ordering::Acquire),
            total_buys: self.total_buys.data.load(Ordering::Acquire),
            total_sells: self.total_sells.data.load(Ordering::Acquire),
            net_trades: self.total_buys.data.load(Ordering::Acquire) as i64 
                - self.total_sells.data.load(Ordering::Acquire) as i64,
            is_active: self.is_active.data.load(Ordering::Acquire),
        }
    }

    /// Set active state
    #[inline]
    pub fn set_active(&self, active: bool) {
        self.is_active.data.store(active, Ordering::Release);
    }

    /// Update parameters
    pub fn update_parameters(&mut self, new_params: ASParameters) {
        self.params = new_params;
    }

    /// Reset counters
    #[inline]
    pub fn reset(&self) {
        self.inventory.data.store(0, Ordering::Release);
        self.total_buys.data.store(0, Ordering::Release);
        self.total_sells.data.store(0, Ordering::Release);
    }
}

/// Inventory statistics
#[derive(Debug, Clone, Copy)]
pub struct InventoryStats {
    pub current_inventory: i64,
    pub total_buys: u64,
    pub total_sells: u64,
    pub net_trades: i64,
    pub is_active: bool,
}

/// Quick lookup table initializer for exp function
pub fn init_exp_lookup_table() -> [f64; EXP_TABLE_SIZE] {
    let mut table = [0.0; EXP_TABLE_SIZE];
    for i in 0..EXP_TABLE_SIZE {
        let x = i as f64 / 100.0;
        table[i] = (-x).exp();
    }
    table
}

/// Fast exponential approximation using lookup table and interpolation
#[inline]
pub fn fast_exp_neg(x: f64, lookup: &[f64; EXP_TABLE_SIZE]) -> f64 {
    if x < 0.0 {
        return x.exp(); // Fallback for negative input
    }
    
    if x >= 10.0 {
        return 0.0; // exp(-10) is essentially 0
    }
    
    let idx = (x * 100.0) as usize;
    let idx = idx.min(EXP_TABLE_SIZE - 2);
    
    let frac = x * 100.0 - idx as f64;
    let v0 = lookup[idx];
    let v1 = lookup[idx + 1];
    
    // Linear interpolation
    v0 + frac * (v1 - v0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inventory_skew_basic() {
        let params = ASParameters::default();
        let calc = InventorySkewCalculator::new(params, 1, 1);
        
        // Neutral inventory
        let quote = calc.generate_skewed_quote(10000, 10);
        assert!(quote.bid_price < quote.ask_price);
        assert_eq!(quote.inventory, 0);
    }

    #[test]
    fn test_inventory_skew_long() {
        let params = ASParameters::default();
        let calc = InventorySkewCalculator::new(params, 1, 1);
        
        // Set long inventory
        calc.update_inventory(500);
        
        let quote = calc.generate_skewed_quote(10000, 10);
        
        // With long inventory, should skew to encourage selling
        // Lower bid, higher ask relative to neutral
        assert!(quote.bid_price < quote.mid_price);
        assert!(quote.ask_size_factor < 1.0); // Reduce ask size when short? No, we're long
    }

    #[test]
    fn test_reservation_price() {
        let mut params = ASParameters::default();
        params.gamma = 0.1;
        params.sigma = 0.02;
        params.time_horizon = 60.0;
        
        let calc = InventorySkewCalculator::new(params, 1, 1);
        
        // Long inventory should have lower reservation price
        calc.update_inventory(100);
        let res_long = calc.calculate_reservation_price(10000);
        assert!(res_long < 10000);
        
        // Short inventory should have higher reservation price
        calc.update_inventory(-100);
        let res_short = calc.calculate_reservation_price(10000);
        assert!(res_short > 10000);
    }

    #[test]
    fn test_breach_detection() {
        let mut params = ASParameters::default();
        params.max_inventory = 100;
        
        let calc = InventorySkewCalculator::new(params, 1, 1);
        calc.update_inventory(90);
        
        // Small buy should not breach
        assert!(!calc.would_breach_limit(5, true));
        
        // Large buy should breach
        assert!(calc.would_breach_limit(20, true));
    }

    #[test]
    fn test_exp_lookup() {
        let lookup = init_exp_lookup_table();
        
        // Test some values
        let x = 0.5;
        let approx = fast_exp_neg(x, &lookup);
        let exact = (-x).exp();
        
        // Should be within 1% error
        let error = (approx - exact).abs() / exact;
        assert!(error < 0.01);
    }
}
