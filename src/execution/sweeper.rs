//! Liquidity Sweeping Engine
//! 
//! Dynamically sizes child orders to clear L2 levels without excessive slippage.
//! Calculates exact volume required to trigger stop-loss cascades hidden behind thin order book walls.

use std::sync::atomic::{AtomicU64, AtomicI64, Ordering};

/// Maximum number of price levels to sweep
pub const MAX_SWEEP_LEVELS: usize = 32;

/// Order book level structure
#[derive(Debug, Clone, Copy)]
pub struct OrderBookLevel {
    pub price: i64,   // Q16.48 fixed-point
    pub size: i64,    // Base currency amount (Q16.48)
}

/// L2 order book snapshot
#[derive(Debug, Clone)]
pub struct OrderBookSnapshot {
    pub bids: [OrderBookLevel; MAX_SWEEP_LEVELS],
    pub asks: [OrderBookLevel; MAX_SWEEP_LEVELS],
    pub bid_count: usize,
    pub ask_count: usize,
    pub timestamp_ns: u64,
}

impl Default for OrderBookSnapshot {
    fn default() -> Self {
        Self {
            bids: [OrderBookLevel { price: 0, size: 0 }; MAX_SWEEP_LEVELS],
            asks: [OrderBookLevel { price: 0, size: 0 }; MAX_SWEEP_LEVELS],
            bid_count: 0,
            ask_count: 0,
            timestamp_ns: 0,
        }
    }
}

/// Sweep calculation result
#[derive(Debug, Clone)]
pub struct SweepResult {
    pub total_size: i64,       // Total size to sweep
    pub avg_price: i64,        // Volume-weighted average price
    pub slippage_bps: f64,     // Expected slippage in basis points
    pub levels_to_sweep: u32,  // Number of levels to clear
    pub estimated_cost: i64,   // Total cost in quote currency (Q16.48)
    pub urgency: SweepUrgency,
}

/// Sweep urgency based on market conditions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SweepUrgency {
    Normal = 0,
    Elevated = 1,
    High = 2,
    Emergency = 3,
}

/// Stop-loss cascade detection
#[derive(Debug, Clone)]
pub struct CascadeDetection {
    pub trigger_price: i64,
    pub estimated_volume: i64,
    pub cascade_probability: f64,
    pub levels_affected: u32,
}

/// Liquidity sweeper engine
pub struct LiquiditySweeper {
    /// Current order book snapshot
    book: OrderBookSnapshot,
    /// Slippage tolerance (Q16.48)
    slippage_tolerance: AtomicU64,
    /// Max sweep size (Q16.48)
    max_sweep_size: AtomicU64,
    /// Minimum liquidity threshold
    min_liquidity: AtomicU64,
    /// Cascade detection enabled
    cascade_enabled: AtomicU64,
}

impl LiquiditySweeper {
    pub const fn new() -> Self {
        Self {
            book: OrderBookSnapshot {
                bids: [OrderBookLevel { price: 0, size: 0 }; MAX_SWEEP_LEVELS],
                asks: [OrderBookLevel { price: 0, size: 0 }; MAX_SWEEP_LEVELS],
                bid_count: 0,
                ask_count: 0,
                timestamp_ns: 0,
            },
            slippage_tolerance: AtomicU64::new((50 * (1u64 << 48) as f64 / 10000.0) as i64 as u64), // 50 bps
            max_sweep_size: AtomicU64::new((1000 << 48) as i64 as u64), // 1000 units
            min_liquidity: AtomicU64::new((100 << 48) as i64 as u64),   // 100 units
            cascade_enabled: AtomicU64::new(1),
        }
    }
    
    /// Update order book snapshot
    pub fn update_book(&mut self, snapshot: OrderBookSnapshot) {
        self.book = snapshot;
    }
    
    /// Calculate optimal sweep size for a given side
    pub fn calculate_sweep(&self, side: u8, target_size: i64) -> SweepResult {
        let levels = if side == 0 { &self.book.bids } else { &self.book.asks };
        let count = if side == 0 { self.book.bid_count } else { self.book.ask_count };
        
        let max_size = self.max_sweep_size.load(Ordering::Acquire) as i64;
        let slippage_tol = self.slippage_tolerance.load(Ordering::Acquire) as i64;
        let slippage_tol_f64 = slippage_tol as f64 / (1u64 << 48) as f64;
        
        let effective_target = target_size.min(max_size);
        
        let mut remaining = effective_target;
        let mut total_cost: i128 = 0;
        let mut levels_swept = 0u32;
        let mut cumulative_slippage = 0.0f64;
        let first_price = levels[0].price as f64 / (1u64 << 48) as f64;
        
        for i in 0..count.min(MAX_SWEEP_LEVELS) {
            if remaining <= 0 {
                break;
            }
            
            let level_size = levels[i].size;
            let level_price = levels[i].price as f64 / (1u64 << 48) as f64;
            
            let fill_size = remaining.min(level_size);
            if fill_size <= 0 {
                continue;
            }
            
            // Calculate cost for this level
            let level_cost = (fill_size as f64 * level_price) as i128;
            total_cost += level_cost;
            remaining -= fill_size;
            levels_swept += 1;
            
            // Calculate slippage from mid-price
            let level_slippage = ((level_price - first_price).abs() / first_price) * 10000.0;
            cumulative_slippage = cumulative_slippage.max(level_slippage);
            
            // Check if we've exceeded slippage tolerance
            if cumulative_slippage > slippage_tol_f64 * 10000.0 {
                break;
            }
        }
        
        let swept_size = effective_target - remaining;
        let avg_price = if swept_size > 0 {
            ((total_cost / swept_size as i128) as f64 * (1u64 << 48) as f64) as i64
        } else {
            0
        };
        
        // Determine urgency based on slippage and liquidity
        let urgency = if cumulative_slippage < slippage_tol_f64 * 5000.0 {
            SweepUrgency::Normal
        } else if cumulative_slippage < slippage_tol_f64 * 7500.0 {
            SweepUrgency::Elevated
        } else if cumulative_slippage < slippage_tol_f64 * 10000.0 {
            SweepUrgency::High
        } else {
            SweepUrgency::Emergency
        };
        
        SweepResult {
            total_size: swept_size,
            avg_price,
            slippage_bps: cumulative_slippage,
            levels_to_sweep: levels_swept,
            estimated_cost: total_cost as i64,
            urgency,
        }
    }
    
    /// Detect potential stop-loss cascade levels
    pub fn detect_cascade(&self, side: u8) -> Option<CascadeDetection> {
        if self.cascade_enabled.load(Ordering::Acquire) == 0 {
            return None;
        }
        
        let levels = if side == 0 { &self.book.bids } else { &self.book.asks };
        let count = if side == 0 { self.book.bid_count } else { self.book.ask_count };
        
        if count < 3 {
            return None;
        }
        
        // Look for thin liquidity followed by thick walls (stop cluster signature)
        let mut thin_levels = 0u32;
        let mut thin_volume: i64 = 0;
        let mut wall_idx = None;
        
        let min_liq = self.min_liquidity.load(Ordering::Acquire) as i64;
        
        for i in 0..count.min(MAX_SWEEP_LEVELS) {
            if levels[i].size < min_liq {
                thin_levels += 1;
                thin_volume += levels[i].size;
            } else if thin_levels >= 2 && wall_idx.is_none() {
                // Found a wall after thin levels
                if levels[i].size > min_liq * 5 {
                    wall_idx = Some(i);
                    break;
                }
            }
        }
        
        if let Some(idx) = wall_idx {
            let trigger_price = levels[idx].price;
            let est_volume = thin_volume + levels[idx].size;
            let cascade_prob = (thin_levels as f64 / 10.0).min(0.9);
            
            Some(CascadeDetection {
                trigger_price,
                estimated_volume: est_volume,
                cascade_probability: cascade_prob,
                levels_affected: thin_levels + 1,
            })
        } else {
            None
        }
    }
    
    /// Generate aggressive sweep order to trigger cascade
    pub fn generate_cascade_order(&self, side: u8, cascade: &CascadeDetection) -> Option<SweepResult> {
        // Size order to just clear the thin levels and hit the stop wall
        let target_size = cascade.estimated_volume;
        let result = self.calculate_sweep(side, target_size);
        
        // Only proceed if cascade probability is high enough
        if cascade.cascade_probability > 0.5 && result.levels_to_sweep >= cascade.levels_affected {
            Some(result)
        } else {
            None
        }
    }
    
    /// Calculate market impact coefficient
    pub fn calculate_market_impact(&self, size: i64) -> f64 {
        let total_liquidity: i64 = self.book.bids.iter()
            .take(self.book.bid_count)
            .map(|l| l.size)
            .sum::<i64>() + self.book.asks.iter()
            .take(self.book.ask_count)
            .map(|l| l.size)
            .sum::<i64>();
        
        if total_liquidity == 0 {
            return 1.0;
        }
        
        // Square-root market impact model
        let size_f64 = size.abs() as f64;
        let liq_f64 = total_liquidity as f64;
        
        0.1 * (size_f64 / liq_f64).sqrt()
    }
    
    /// Set slippage tolerance (in basis points)
    #[inline]
    pub fn set_slippage_tolerance(&self, bps: f64) {
        let fixed = ((bps / 10000.0).clamp(0.0, 1.0) * (1u64 << 48) as f64) as i64 as u64;
        self.slippage_tolerance.store(fixed, Ordering::Release);
    }
    
    /// Set maximum sweep size
    #[inline]
    pub fn set_max_sweep_size(&self, size: i64) {
        let fixed = (size.max(0) as u64).min((100000 << 48) as i64 as u64);
        self.max_sweep_size.store(fixed, Ordering::Release);
    }
    
    /// Enable/disable cascade detection
    #[inline]
    pub fn set_cascade_detection(&self, enabled: bool) {
        self.cascade_enabled.store(if enabled { 1 } else { 0 }, Ordering::Release);
    }
    
    /// Get current spread in basis points
    pub fn get_spread_bps(&self) -> f64 {
        if self.book.bid_count == 0 || self.book.ask_count == 0 {
            return f64::MAX;
        }
        
        let best_bid = self.book.bids[0].price as f64 / (1u64 << 48) as f64;
        let best_ask = self.book.asks[0].price as f64 / (1u64 << 48) as f64;
        
        if best_bid <= 0.0 || best_ask <= 0.0 {
            return f64::MAX;
        }
        
        ((best_ask - best_bid) / ((best_bid + best_ask) / 2.0)) * 10000.0
    }
    
    /// Get order book imbalance
    pub fn get_imbalance(&self) -> f64 {
        let bid_liquidity: i64 = self.book.bids.iter()
            .take(self.book.bid_count.min(5))
            .map(|l| l.size)
            .sum();
        
        let ask_liquidity: i64 = self.book.asks.iter()
            .take(self.book.ask_count.min(5))
            .map(|l| l.size)
            .sum();
        
        let total = bid_liquidity + ask_liquidity;
        if total == 0 {
            return 0.0;
        }
        
        (bid_liquidity - ask_liquidity) as f64 / total as f64
    }
}

/// Child order for execution slicing
#[derive(Debug, Clone)]
pub struct ChildOrder {
    pub price: i64,
    pub size: i64,
    pub sequence: u32,
    pub wait_ms: u32,
}

/// Execution slicer for large sweeps
pub struct ExecutionSlicer {
    pub parent_size: i64,
    pub executed_size: i64,
    pub child_orders: Vec<ChildOrder>,
    pub next_child_idx: usize,
}

impl ExecutionSlicer {
    pub fn new(parent_size: i64, num_slices: u32, urgency: SweepUrgency) -> Self {
        let slice_size = parent_size / num_slices as i64;
        let base_wait = match urgency {
            SweepUrgency::Normal => 100,
            SweepUrgency::Elevated => 50,
            SweepUrgency::High => 10,
            SweepUrgency::Emergency => 0,
        };
        
        let mut child_orders = Vec::with_capacity(num_slices as usize);
        for i in 0..num_slices {
            child_orders.push(ChildOrder {
                price: 0,  // Will be set at execution time
                size: slice_size,
                sequence: i,
                wait_ms: base_wait + (i * 10),
            });
        }
        
        Self {
            parent_size,
            executed_size: 0,
            child_orders,
            next_child_idx: 0,
        }
    }
    
    pub fn get_next_order(&mut self) -> Option<&ChildOrder> {
        if self.next_child_idx < self.child_orders.len() {
            Some(&self.child_orders[self.next_child_idx])
        } else {
            None
        }
    }
    
    pub fn advance(&mut self) {
        self.next_child_idx += 1;
    }
    
    pub fn is_complete(&self) -> bool {
        self.next_child_idx >= self.child_orders.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_sweep_calculation() {
        let mut sweeper = LiquiditySweeper::new();
        
        // Create a simple order book
        let mut snapshot = OrderBookSnapshot::default();
        snapshot.asks[0] = OrderBookLevel { price: 50000 << 48, size: 100 };
        snapshot.asks[1] = OrderBookLevel { price: 50001 << 48, size: 200 };
        snapshot.asks[2] = OrderBookLevel { price: 50002 << 48, size: 150 };
        snapshot.ask_count = 3;
        
        sweeper.update_book(snapshot);
        
        // Calculate sweep for 250 units
        let result = sweeper.calculate_sweep(1, 250);
        
        assert!(result.total_size >= 250);
        assert!(result.levels_to_sweep >= 2);
        assert!(result.slippage_bps > 0.0);
    }
    
    #[test]
    fn test_cascade_detection() {
        let mut sweeper = LiquiditySweeper::new();
        
        // Create order book with thin levels followed by wall
        let mut snapshot = OrderBookSnapshot::default();
        snapshot.asks[0] = OrderBookLevel { price: 50000 << 48, size: 10 };  // Thin
        snapshot.asks[1] = OrderBookLevel { price: 50001 << 48, size: 15 };  // Thin
        snapshot.asks[2] = OrderBookLevel { price: 50002 << 48, size: 500 }; // Wall
        snapshot.ask_count = 3;
        
        sweeper.update_book(snapshot);
        
        let cascade = sweeper.detect_cascade(1);
        assert!(cascade.is_some());
        
        let c = cascade.unwrap();
        assert!(c.cascade_probability > 0.2);
        assert_eq!(c.levels_affected, 3);
    }
    
    #[test]
    fn test_market_impact() {
        let mut sweeper = LiquiditySweeper::new();
        
        let mut snapshot = OrderBookSnapshot::default();
        for i in 0..10 {
            snapshot.bids[i] = OrderBookLevel { 
                price: (50000 - i) << 48, 
                size: 100 
            };
            snapshot.asks[i] = OrderBookLevel { 
                price: (50000 + i) << 48, 
                size: 100 
            };
        }
        snapshot.bid_count = 10;
        snapshot.ask_count = 10;
        
        sweeper.update_book(snapshot);
        
        let impact_small = sweeper.calculate_market_impact(10);
        let impact_large = sweeper.calculate_market_impact(500);
        
        assert!(impact_large > impact_small);
    }
}
