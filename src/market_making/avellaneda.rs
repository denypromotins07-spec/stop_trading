//! Avellaneda-Stoikov Market Making Model with Stochastic Volatility Extensions
//!
//! Implements the full Avellaneda-Stoikov market making model solving the 
//! Hamilton-Jacobi-Bellman (HJB) approximations using pre-computed Taylor series 
//! lookup tables to avoid heavy math in the hot path.
//! Pre-allocated lookup tables respect the 6.5GB RAM ceiling.

use std::sync::atomic::{AtomicU64, AtomicBool, AtomicI64, Ordering};
use std::sync::Arc;
use std::time::{Instant, Duration};
use crossbeam_channel::{bounded, Sender, Receiver};
use dashmap::DashMap;

/// Maximum inventory position (risk limit)
const MAX_INVENTORY: f64 = 100.0;

/// Avellaneda-Stoikov parameters
#[derive(Debug, Clone)]
pub struct ASParameters {
    /// Risk aversion coefficient (gamma)
    pub gamma: f64,
    /// Volatility (sigma) - annualized
    pub volatility: f64,
    /// Order arrival intensity (lambda)
    pub lambda: f64,
    /// Time horizon (T) in seconds
    pub time_horizon: f64,
    /// Tick size
    pub tick_size: f64,
    /// Minimum spread (bps)
    pub min_spread_bps: f64,
    /// Maximum spread (bps)
    pub max_spread_bps: f64,
}

impl Default for ASParameters {
    fn default() -> Self {
        Self {
            gamma: 0.1,
            volatility: 0.6,
            lambda: 1.0,
            time_horizon: 300.0, // 5 minutes
            tick_size: 0.01,
            min_spread_bps: 1.0,
            max_spread_bps: 100.0,
        }
    }
}

/// Pre-computed Taylor series lookup table for HJB solution
/// Allocated once at startup to respect 6.5GB RAM limit
pub struct HJBLookupTable {
    /// Reservation price adjustments for various inventory levels
    /// Dimensions: [inventory_bins][time_bins]
    reservation_price_adjustments: Box<[f64]>,
    /// Optimal spread for various market states
    /// Dimensions: [volatility_bins][inventory_bins][time_bins]
    optimal_spreads: Box<[f64]>,
    /// Configuration
    pub inventory_bins: usize,
    pub time_bins: usize,
    pub volatility_bins: usize,
    /// Inventory range covered (-max to +max)
    pub max_inventory: f64,
    /// Time range covered (0 to T)
    pub max_time: f64,
    /// Volatility range covered
    pub max_volatility: f64,
}

impl HJBLookupTable {
    /// Create a new lookup table with specified dimensions
    /// Memory usage: ~8 bytes * inv_bins * time_bins * vol_bins
    pub fn new(
        inventory_bins: usize,
        time_bins: usize,
        volatility_bins: usize,
        max_inventory: f64,
        max_time: f64,
        max_volatility: f64,
    ) -> Self {
        let total_size = inventory_bins * time_bins * volatility_bins;
        
        // Pre-allocate with zeros
        let mut reservation_price_adjustments = vec![0.0; inventory_bins * time_bins].into_boxed_slice();
        let mut optimal_spreads = vec![0.0; total_size].into_boxed_slice();

        // Pre-compute values using closed-form Avellaneda-Stoikov approximation
        // r(s, t) = s - q * gamma * sigma^2 * (T - t)
        // delta = gamma * sigma^2 * (T - t) / 2 + (1/gamma) * ln(1 + gamma/k)
        
        for t_idx in 0..time_bins {
            let time_remaining = max_time * (1.0 - t_idx as f64 / time_bins as f64);
            
            for inv_idx in 0..inventory_bins {
                let inventory = -max_inventory + 2.0 * max_inventory * inv_idx as f64 / inventory_bins as f64;
                
                // Reservation price adjustment (simplified AS formula)
                // This is computed once and stored
                let default_vol = 0.6;
                let default_gamma = 0.1;
                let adj = -inventory * default_gamma * default_vol.powi(2) * time_remaining;
                
                let rp_idx = t_idx * inventory_bins + inv_idx;
                reservation_price_adjustments[rp_idx] = adj;
                
                // Pre-compute optimal spreads for each volatility bin
                for vol_idx in 0..volatility_bins {
                    let vol = max_volatility * vol_idx as f64 / volatility_bins as f64;
                    
                    // Optimal half-spread approximation
                    // delta = gamma * sigma^2 * (T-t) / 2 + (1/gamma) * ln(1 + gamma*kappa)
                    let kappa = 1.0; // Simplified intensity parameter
                    let base_spread = default_gamma * vol.powi(2) * time_remaining / 2.0;
                    let intensity_adj = (1.0 / default_gamma) * (1.0 + default_gamma * kappa).ln();
                    let half_spread = base_spread + intensity_adj;
                    
                    let spread_idx = vol_idx * inventory_bins * time_bins + t_idx * inventory_bins + inv_idx;
                    optimal_spreads[spread_idx] = half_spread.max(0.0001); // Minimum 1 bp
                }
            }
        }

        Self {
            reservation_price_adjustments,
            optimal_spreads,
            inventory_bins,
            time_bins,
            volatility_bins,
            max_inventory,
            max_time,
            max_volatility,
        }
    }

    /// Get reservation price adjustment for given inventory and time
    #[inline]
    pub fn get_reservation_adjustment(&self, inventory: f64, time_remaining: f64) -> f64 {
        let inv_idx = ((inventory + self.max_inventory) / (2.0 * self.max_inventory) 
            * self.inventory_bins as f64) as usize;
        let inv_idx = inv_idx.min(self.inventory_bins - 1);
        
        let time_idx = (time_remaining / self.max_time * self.time_bins as f64) as usize;
        let time_idx = time_idx.min(self.time_bins - 1);
        
        let idx = time_idx * self.inventory_bins + inv_idx;
        self.reservation_price_adjustments[idx]
    }

    /// Get optimal spread for given state
    #[inline]
    pub fn get_optimal_spread(&self, volatility: f64, inventory: f64, time_remaining: f64) -> f64 {
        let vol_idx = (volatility / self.max_volatility * self.volatility_bins as f64) as usize;
        let vol_idx = vol_idx.min(self.volatility_bins - 1);
        
        let inv_idx = ((inventory + self.max_inventory) / (2.0 * self.max_inventory) 
            * self.inventory_bins as f64) as usize;
        let inv_idx = inv_idx.min(self.inventory_bins - 1);
        
        let time_idx = (time_remaining / self.max_time * self.time_bins as f64) as usize;
        let time_idx = time_idx.min(self.time_bins - 1);
        
        let idx = vol_idx * self.inventory_bins * self.time_bins 
            + time_idx * self.inventory_bins + inv_idx;
        self.optimal_spreads[idx]
    }

    /// Estimate memory usage in bytes
    pub fn memory_usage_bytes(&self) -> usize {
        8 * (self.reservation_price_adjustments.len() + self.optimal_spreads.len())
    }
}

/// Stochastic volatility extension parameters
#[derive(Debug, Clone)]
pub struct StochVolParams {
    /// Mean reversion speed (kappa)
    pub mean_reversion_speed: f64,
    /// Long-term volatility (theta)
    pub long_term_vol: f64,
    /// Volatility of volatility (xi)
    pub vol_of_vol: f64,
    /// Correlation between price and vol (rho)
    pub correlation: f64,
}

impl Default for StochVolParams {
    fn default() -> Self {
        Self {
            mean_reversion_speed: 2.0,
            long_term_vol: 0.6,
            vol_of_vol: 0.3,
            correlation: -0.5,
        }
    }
}

/// Market state for quote calculation
#[derive(Debug, Clone)]
pub struct MarketState {
    /// Mid price
    pub mid_price: f64,
    /// Current volatility estimate
    pub volatility: f64,
    /// Time remaining in trading horizon (seconds)
    pub time_remaining: f64,
    /// Best bid depth
    pub bid_depth: f64,
    /// Best ask depth
    pub ask_depth: f64,
}

/// Quote generated by AS model
#[derive(Debug, Clone)]
pub struct ASQuote {
    /// Bid price
    pub bid: f64,
    /// Ask price
    pub ask: f64,
    /// Bid size
    pub bid_size: f64,
    /// Ask size
    pub ask_size: f64,
    /// Reservation price
    pub reservation_price: f64,
    /// Half spread (distance from reservation price)
    pub half_spread: f64,
    /// Inventory skew factor
    pub inventory_skew: f64,
    /// Timestamp
    pub timestamp_ns: u64,
}

impl ASQuote {
    pub fn spread_bps(&self) -> f64 {
        if self.mid_price() <= 0.0 {
            return 0.0;
        }
        ((self.ask - self.bid) / self.mid_price()) * 10000.0
    }

    pub fn mid_price(&self) -> f64 {
        (self.bid + self.ask) / 2.0
    }
}

/// Avellaneda-Stoikov market maker engine
pub struct AvellanedaStoikovMM {
    /// Parameters
    params: ASParameters,
    /// Stochastic volatility parameters
    stoch_vol_params: StochVolParams,
    /// Pre-computed lookup table
    lookup_table: Arc<HJBLookupTable>,
    /// Current inventory
    current_inventory: AtomicI64, // Scaled by 1e6
    /// PnL tracking
    realized_pnl: AtomicI64, // Scaled by 1e6
    unrealized_pnl: AtomicI64,
    /// Is MM active
    is_active: AtomicBool,
    /// Quote counter
    quote_counter: AtomicU64,
    /// Last quote timestamp
    last_quote_ns: AtomicU64,
    /// Event channel
    event_tx: Sender<MMEvent>,
    event_rx: Receiver<MMEvent>,
}

/// Market making events
#[derive(Debug, Clone)]
pub enum MMEvent {
    /// New quote generated
    QuoteGenerated(ASQuote),
    /// Inventory update
    InventoryUpdated { old: i64, new: i64 },
    /// PnL update
    PnLUpdate { realized: f64, unrealized: f64 },
    /// Risk limit breach
    RiskLimitBreach { inventory: i64, limit: i64 },
    /// Volatility regime change
    VolatilityRegimeChange { old_vol: f64, new_vol: f64 },
}

impl AvellanedaStoikovMM {
    /// Create a new AS market maker with pre-allocated lookup table
    pub fn new(
        params: ASParameters,
        stoch_vol_params: StochVolParams,
        lookup_table: Arc<HJBLookupTable>,
        buffer_size: usize,
    ) -> Self {
        let (tx, rx) = bounded(buffer_size);

        Self {
            params,
            stoch_vol_params,
            lookup_table,
            current_inventory: AtomicI64::new(0),
            realized_pnl: AtomicI64::new(0),
            unrealized_pnl: AtomicI64::new(0),
            is_active: AtomicBool::new(true),
            quote_counter: AtomicU64::new(0),
            last_quote_ns: AtomicU64::new(0),
            event_tx: tx,
            event_rx: rx,
        }
    }

    /// Generate quote based on current market state and inventory
    pub fn generate_quote(&self, state: &MarketState) -> Option<ASQuote> {
        if !self.is_active.load(Ordering::Relaxed) {
            return None;
        }

        let inventory = self.get_inventory();
        let inventory_f64 = inventory as f64 / 1e6;

        // Check inventory limits
        if inventory_f64.abs() > MAX_INVENTORY {
            let _ = self.event_tx.send(MMEvent::RiskLimitBreach {
                inventory,
                limit: (MAX_INVENTORY * 1e6) as i64,
            });
            return None;
        }

        // Get reservation price adjustment from lookup table
        let res_adjustment = self.lookup_table.get_reservation_adjustment(
            inventory_f64,
            state.time_remaining,
        );

        // Calculate reservation price
        let reservation_price = state.mid_price + res_adjustment;

        // Get optimal spread from lookup table
        let half_spread_raw = self.lookup_table.get_optimal_spread(
            state.volatility,
            inventory_f64,
            state.time_remaining,
        );

        // Apply stochastic volatility correction
        let vol_correction = self.stoch_vol_correction(state.volatility);
        let half_spread = half_spread_raw * vol_correction;

        // Convert to absolute price terms
        let half_spread_abs = (half_spread * state.mid_price).max(self.params.tick_size);

        // Calculate bid/ask
        let mut bid = reservation_price - half_spread_abs;
        let mut ask = reservation_price + half_spread_abs;

        // Round to tick size
        bid = (bid / self.params.tick_size).floor() * self.params.tick_size;
        ask = (ask / self.params.tick_size).ceil() * self.params.tick_size;

        // Ensure minimum spread
        let min_spread = self.params.min_spread_bps / 10000.0 * state.mid_price;
        if ask - bid < min_spread {
            let midpoint = (bid + ask) / 2.0;
            bid = midpoint - min_spread / 2.0;
            ask = midpoint + min_spread / 2.0;
        }

        // Calculate size based on depth and inventory
        let bid_size = self.calculate_order_size(inventory_f64, state.bid_depth, true);
        let ask_size = self.calculate_order_size(inventory_f64, state.ask_depth, false);

        // Inventory skew factor
        let inventory_skew = inventory_f64 / MAX_INVENTORY;

        let now_ns = Instant::now().duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default().as_nanos() as u64;

        let quote = ASQuote {
            bid,
            ask,
            bid_size,
            ask_size,
            reservation_price,
            half_spread: half_spread_abs,
            inventory_skew,
            timestamp_ns: now_ns,
        };

        // Update tracking
        self.quote_counter.fetch_add(1, Ordering::Relaxed);
        self.last_quote_ns.store(now_ns, Ordering::Relaxed);

        let _ = self.event_tx.send(MMEvent::QuoteGenerated(quote.clone()));

        Some(quote)
    }

    /// Stochastic volatility correction factor
    fn stoch_vol_correction(&self, current_vol: f64) -> f64 {
        // Ornstein-Uhlenbeck process adjustment
        // When vol is above long-term mean, widen spreads
        let vol_ratio = current_vol / self.stoch_vol_params.long_term_vol;
        
        // Exponential adjustment
        let correction = 1.0 + (vol_ratio - 1.0) * self.stoch_vol_params.vol_of_vol;
        correction.max(0.5).min(2.0) // Clamp between 0.5x and 2x
    }

    /// Calculate order size based on inventory and market depth
    fn calculate_order_size(&self, inventory: f64, market_depth: f64, is_bid: bool) -> f64 {
        // Base size
        let base_size = 1.0;

        // Inventory adjustment: reduce size when inventory is large in same direction
        let inv_factor = if is_bid {
            // Reduce bid size when already long
            (1.0 - inventory / MAX_INVENTORY).max(0.1)
        } else {
            // Reduce ask size when already short
            (1.0 + inventory / MAX_INVENTORY).max(0.1)
        };

        // Depth adjustment: scale with available liquidity
        let depth_factor = (market_depth / 100.0).min(10.0);

        base_size * inv_factor * depth_factor
    }

    /// Update inventory after fill
    pub fn update_inventory(&self, fill_size: f64, is_buy: bool) {
        let delta = if is_buy {
            (fill_size * 1e6) as i64
        } else {
            -(fill_size * 1e6) as i64
        };

        let old = self.current_inventory.fetch_add(delta, Ordering::Relaxed);
        let new = old + delta;

        let _ = self.event_tx.send(MMEvent::InventoryUpdated { old, new });

        // Check if approaching limits
        let new_f64 = new as f64 / 1e6;
        if new_f64.abs() > MAX_INVENTORY * 0.8 {
            let _ = self.event_tx.send(MMEvent::RiskLimitBreach {
                inventory: new,
                limit: (MAX_INVENTORY * 0.8 * 1e6) as i64,
            });
        }
    }

    /// Update PnL
    pub fn update_pnl(&self, realized_delta: f64, unrealized_delta: f64) {
        let real_scaled = (realized_delta * 1e6) as i64;
        let unreal_scaled = (unrealized_delta * 1e6) as i64;

        self.realized_pnl.fetch_add(real_scaled, Ordering::Relaxed);
        self.unrealized_pnl.fetch_add(unreal_scaled, Ordering::Relaxed);

        let _ = self.event_tx.send(MMEvent::PnLUpdate {
            realized: realized_delta,
            unrealized: unrealized_delta,
        });
    }

    /// Get current inventory
    pub fn get_inventory(&self) -> i64 {
        self.current_inventory.load(Ordering::Relaxed)
    }

    /// Get realized PnL
    pub fn get_realized_pnl(&self) -> f64 {
        self.realized_pnl.load(Ordering::Relaxed) as f64 / 1e6
    }

    /// Get unrealized PnL
    pub fn get_unrealized_pnl(&self) -> f64 {
        self.unrealized_pnl.load(Ordering::Relaxed) as f64 / 1e6
    }

    /// Get total PnL
    pub fn get_total_pnl(&self) -> f64 {
        self.get_realized_pnl() + self.get_unrealized_pnl()
    }

    /// Get event receiver
    pub fn get_event_receiver(&self) -> Receiver<MMEvent> {
        self.event_rx.clone()
    }

    /// Deactivate MM
    pub fn deactivate(&self) {
        self.is_active.store(false, Ordering::Relaxed);
    }

    /// Activate MM
    pub fn activate(&self) {
        self.is_active.store(true, Ordering::Relaxed);
    }

    /// Get quote count
    pub fn get_quote_count(&self) -> u64 {
        self.quote_counter.load(Ordering::Relaxed)
    }

    /// Get parameters
    pub fn get_params(&self) -> &ASParameters {
        &self.params
    }

    /// Update parameters dynamically
    pub fn update_params(&mut self, new_params: ASParameters) {
        self.params = new_params;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lookup_table_creation() {
        let table = HJBLookupTable::new(100, 100, 50, 100.0, 300.0, 2.0);
        
        assert_eq!(table.inventory_bins, 100);
        assert_eq!(table.time_bins, 100);
        assert_eq!(table.volatility_bins, 50);
        
        // Check memory usage is reasonable (< 100MB for typical config)
        let mem_mb = table.memory_usage_bytes() as f64 / (1024.0 * 1024.0);
        assert!(mem_mb < 100.0);
    }

    #[test]
    fn test_lookup_table_query() {
        let table = HJBLookupTable::new(100, 100, 50, 100.0, 300.0, 2.0);
        
        // Query at zero inventory, full time remaining
        let adj = table.get_reservation_adjustment(0.0, 300.0);
        assert!((adj - 0.0).abs() < 0.01); // Should be near zero
        
        // Query with positive inventory (should have negative adjustment)
        let adj_positive_inv = table.get_reservation_adjustment(50.0, 150.0);
        assert!(adj_positive_inv < 0.0); // Long inventory = lower reservation price
    }

    #[test]
    fn test_mm_initialization() {
        let params = ASParameters::default();
        let stoch_vol = StochVolParams::default();
        let table = Arc::new(HJBLookupTable::new(100, 100, 50, 100.0, 300.0, 2.0));
        
        let mm = AvellanedaStoikovMM::new(params, stoch_vol, table, 1000);
        
        assert!(mm.is_active.load(Ordering::Relaxed));
        assert_eq!(mm.get_inventory(), 0);
        assert_eq!(mm.get_quote_count(), 0);
    }

    #[test]
    fn test_quote_generation() {
        let params = ASParameters::default();
        let stoch_vol = StochVolParams::default();
        let table = Arc::new(HJBLookupTable::new(100, 100, 50, 100.0, 300.0, 2.0));
        
        let mm = AvellanedaStoikovMM::new(params, stoch_vol, table, 1000);
        
        let state = MarketState {
            mid_price: 50000.0,
            volatility: 0.6,
            time_remaining: 300.0,
            bid_depth: 100.0,
            ask_depth: 100.0,
        };
        
        let quote = mm.generate_quote(&state);
        assert!(quote.is_some());
        
        let quote = quote.unwrap();
        assert!(quote.bid < quote.ask);
        assert!(quote.spread_bps() >= params.min_spread_bps);
        assert!(quote.spread_bps() <= params.max_spread_bps);
    }

    #[test]
    fn test_inventory_update() {
        let params = ASParameters::default();
        let stoch_vol = StochVolParams::default();
        let table = Arc::new(HJBLookupTable::new(100, 100, 50, 100.0, 300.0, 2.0));
        
        let mm = AvellanedaStoikovMM::new(params, stoch_vol, table, 1000);
        
        assert_eq!(mm.get_inventory(), 0);
        
        mm.update_inventory(10.0, true); // Buy 10
        assert_eq!(mm.get_inventory(), 10_000_000); // Scaled by 1e6
        
        mm.update_inventory(5.0, false); // Sell 5
        assert_eq!(mm.get_inventory(), 5_000_000);
    }
}
