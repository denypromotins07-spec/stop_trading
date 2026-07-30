//! Arrival Price Module - Implementation Shortfall Calculation
//! Calculates Implementation Shortfall (Arrival Price vs Execution Price) for all child orders.
//! Tracks market impact costs for TWAP/VWAP algos to dynamically tune aggression parameters.

use std::sync::atomic::{AtomicU64, AtomicI64, AtomicBool, Ordering};
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

/// Side indicator
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Side {
    Buy,
    Sell,
}

/// Parent order arrival price record
#[derive(Debug, Clone, Copy)]
pub struct ArrivalPriceRecord {
    pub parent_order_id: u64,
    pub side: Side,
    /// Price at time of order arrival (decision price)
    pub arrival_price: i64,
    /// Timestamp of arrival (nanoseconds)
    pub arrival_time_ns: u64,
    /// Total quantity to execute
    pub total_quantity: u64,
    /// Executed quantity so far
    pub executed_quantity: u64,
    /// Total execution value (price * qty sum)
    pub total_execution_value: i64,
    /// Is complete
    pub is_complete: bool,
}

/// Child order execution record
#[derive(Debug, Clone, Copy)]
pub struct ChildExecution {
    pub parent_order_id: u64,
    pub child_order_id: u64,
    pub execution_price: i64,
    pub execution_quantity: u64,
    pub execution_time_ns: u64,
    /// Market price at execution time (for impact calc)
    pub market_price_at_exec: i64,
}

/// Implementation shortfall result
#[derive(Debug, Clone, Copy)]
pub struct ImplementationShortfall {
    pub parent_order_id: u64,
    /// Shortfall in ticks (positive = underperformed)
    pub shortfall_ticks: i64,
    /// Shortfall in basis points
    pub shortfall_bps: f64,
    /// Market impact component (bps)
    pub market_impact_bps: f64,
    /// Timing cost component (bps)
    pub timing_cost_bps: f64,
    /// Total cost in currency units
    pub total_cost: i64,
    /// Execution quality score (0-100, higher is better)
    pub quality_score: f64,
}

/// Lock-free arrival price tracker
pub struct ArrivalPriceTracker {
    /// Active parent orders count
    active_orders: CachePadded<AtomicU64>,
    /// Total shortfall tracked (ticks * quantity)
    total_shortfall: CachePadded<AtomicI64>,
    /// Total executed quantity
    total_executed: CachePadded<AtomicU64>,
    /// Completed orders count
    completed_orders: CachePadded<AtomicU64>,
    /// Tick size
    tick_size: i64,
    /// Active flag
    is_active: CachePadded<AtomicBool>,
}

impl ArrivalPriceTracker {
    pub fn new(tick_size: i64) -> Self {
        Self {
            active_orders: CachePadded::default(),
            total_shortfall: CachePadded::default(),
            total_executed: CachePadded::default(),
            completed_orders: CachePadded::default(),
            tick_size,
            is_active: CachePadded::new(AtomicBool::new(true)),
        }
    }

    /// Record arrival of a new parent order
    #[inline]
    pub fn record_arrival(&self, _arrival: &ArrivalPriceRecord) {
        if !self.is_active.data.load(Ordering::Acquire) {
            return;
        }
        self.active_orders.data.fetch_add(1, Ordering::AcqRel);
    }

    /// Record a child execution and calculate incremental shortfall
    pub fn record_child_execution(&self, child: &ChildExecution, arrival: &ArrivalPriceRecord) -> ImplementationShortfall {
        if !self.is_active.data.load(Ordering::Acquire) {
            return self.create_empty_result(child.parent_order_id);
        }

        // Calculate shortfall for this child execution
        let shortfall_ticks = match arrival.side {
            Side::Buy => {
                // For buys: shortfall = exec_price - arrival_price (positive = paid more)
                child.execution_price - arrival.arrival_price
            }
            Side::Sell => {
                // For sells: shortfall = arrival_price - exec_price (positive = received less)
                arrival.arrival_price - child.execution_price
            }
        };

        // Calculate components
        let market_impact_ticks = child.execution_price - child.market_price_at_exec;
        let timing_cost_ticks = child.market_price_at_exec - arrival.arrival_price;

        // Convert to basis points
        let shortfall_bps = if arrival.arrival_price > 0 {
            (shortfall_ticks as f64 * self.tick_size as f64 / arrival.arrival_price as f64) * 10_000.0
        } else {
            0.0
        };

        let market_impact_bps = if arrival.arrival_price > 0 {
            (market_impact_ticks as f64 * self.tick_size as f64 / arrival.arrival_price as f64) * 10_000.0
        } else {
            0.0
        };

        let timing_cost_bps = if arrival.arrival_price > 0 {
            (timing_cost_ticks as f64 * self.tick_size as f64 / arrival.arrival_price as f64) * 10_000.0
        } else {
            0.0
        };

        // Total cost
        let total_cost = shortfall_ticks * self.tick_size * child.execution_quantity as i64;

        // Quality score (inverse of shortfall, scaled to 0-100)
        let quality_score = (100.0 - shortfall_bps.abs()).max(0.0).min(100.0);

        // Update counters
        self.total_shortfall
            .data
            .fetch_add(shortfall_ticks * child.execution_quantity as i64, Ordering::AcqRel);
        self.total_executed
            .data
            .fetch_add(child.execution_quantity, Ordering::AcqRel);

        // Check if order is complete
        let new_executed = arrival.executed_quantity + child.execution_quantity;
        if new_executed >= arrival.total_quantity && !arrival.is_complete {
            self.completed_orders.data.fetch_add(1, Ordering::AcqRel);
            self.active_orders.data.fetch_sub(1, Ordering::AcqRel);
        }

        ImplementationShortfall {
            parent_order_id: child.parent_order_id,
            shortfall_ticks,
            shortfall_bps,
            market_impact_bps,
            timing_cost_bps,
            total_cost,
            quality_score,
        }
    }

    /// Calculate aggregate statistics for a parent order
    pub fn calculate_order_shortfall(
        &self,
        arrival: &ArrivalPriceRecord,
        child_executions: &[ChildExecution],
    ) -> ImplementationShortfall {
        if child_executions.is_empty() || arrival.total_quantity == 0 {
            return self.create_empty_result(arrival.parent_order_id);
        }

        // Calculate VWAP of executions
        let mut total_value: i64 = 0;
        let mut total_qty: u64 = 0;

        for child in child_executions {
            total_value += child.execution_price * child.execution_quantity as i64;
            total_qty += child.execution_quantity;
        }

        let vwap = if total_qty > 0 {
            total_value / total_qty as i64
        } else {
            arrival.arrival_price
        };

        // Overall shortfall
        let shortfall_ticks = match arrival.side {
            Side::Buy => vwap - arrival.arrival_price,
            Side::Sell => arrival.arrival_price - vwap,
        };

        let shortfall_bps = if arrival.arrival_price > 0 {
            (shortfall_ticks as f64 * self.tick_size as f64 / arrival.arrival_price as f64) * 10_000.0
        } else {
            0.0
        };

        // Average market impact and timing cost
        let mut total_impact: i64 = 0;
        let mut total_timing: i64 = 0;

        for child in child_executions {
            total_impact += child.execution_price - child.market_price_at_exec;
            total_timing += child.market_price_at_exec - arrival.arrival_price;
        }

        let avg_impact = total_impact / child_executions.len() as i64;
        let avg_timing = total_timing / child_executions.len() as i64;

        let market_impact_bps = if arrival.arrival_price > 0 {
            (avg_impact as f64 * self.tick_size as f64 / arrival.arrival_price as f64) * 10_000.0
        } else {
            0.0
        };

        let timing_cost_bps = if arrival.arrival_price > 0 {
            (avg_timing as f64 * self.tick_size as f64 / arrival.arrival_price as f64) * 10_000.0
        } else {
            0.0
        };

        let total_cost = shortfall_ticks * self.tick_size * total_qty as i64;
        let quality_score = (100.0 - shortfall_bps.abs()).max(0.0).min(100.0);

        ImplementationShortfall {
            parent_order_id: arrival.parent_order_id,
            shortfall_ticks,
            shortfall_bps,
            market_impact_bps,
            timing_cost_bps,
            total_cost,
            quality_score,
        }
    }

    /// Get aggregate statistics
    pub fn get_stats(&self) -> ArrivalPriceStats {
        let active = self.active_orders.data.load(Ordering::Acquire);
        let completed = self.completed_orders.data.load(Ordering::Acquire);
        let total_slip = self.total_shortfall.data.load(Ordering::Acquire);
        let total_exec = self.total_executed.data.load(Ordering::Acquire);

        ArrivalPriceStats {
            active_orders: active,
            completed_orders: completed,
            total_shortfall_ticks: total_slip,
            average_shortfall_ticks: if total_exec > 0 {
                total_slip as f64 / total_exec as f64
            } else {
                0.0
            },
            total_executed_quantity: total_exec,
            is_active: self.is_active.data.load(Ordering::Acquire),
        }
    }

    fn create_empty_result(&self, parent_order_id: u64) -> ImplementationShortfall {
        ImplementationShortfall {
            parent_order_id,
            shortfall_ticks: 0,
            shortfall_bps: 0.0,
            market_impact_bps: 0.0,
            timing_cost_bps: 0.0,
            total_cost: 0,
            quality_score: 100.0,
        }
    }

    #[inline]
    pub fn set_active(&self, active: bool) {
        self.is_active.data.store(active, Ordering::Release);
    }

    #[inline]
    pub fn reset(&self) {
        self.active_orders.data.store(0, Ordering::Release);
        self.total_shortfall.data.store(0, Ordering::Release);
        self.total_executed.data.store(0, Ordering::Release);
        self.completed_orders.data.store(0, Ordering::Release);
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ArrivalPriceStats {
    pub active_orders: u64,
    pub completed_orders: u64,
    pub total_shortfall_ticks: i64,
    pub average_shortfall_ticks: f64,
    pub total_executed_quantity: u64,
    pub is_active: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buy_order_shortfall() {
        let tracker = ArrivalPriceTracker::new(1);

        let arrival = ArrivalPriceRecord {
            parent_order_id: 1,
            side: Side::Buy,
            arrival_price: 10000,
            arrival_time_ns: 1000,
            total_quantity: 1000,
            executed_quantity: 0,
            total_execution_value: 0,
            is_complete: false,
        };

        let children = vec![
            ChildExecution {
                parent_order_id: 1,
                child_order_id: 1,
                execution_price: 10005,
                execution_quantity: 500,
                execution_time_ns: 2000,
                market_price_at_exec: 10003,
            },
            ChildExecution {
                parent_order_id: 1,
                child_order_id: 2,
                execution_price: 10008,
                execution_quantity: 500,
                execution_time_ns: 3000,
                market_price_at_exec: 10006,
            },
        ];

        let result = tracker.calculate_order_shortfall(&arrival, &children);
        assert!(result.shortfall_ticks > 0); // Paid more than arrival
        assert!(result.shortfall_bps > 0.0);
    }

    #[test]
    fn test_sell_order_shortfall() {
        let tracker = ArrivalPriceTracker::new(1);

        let arrival = ArrivalPriceRecord {
            parent_order_id: 2,
            side: Side::Sell,
            arrival_price: 10000,
            arrival_time_ns: 1000,
            total_quantity: 1000,
            executed_quantity: 0,
            total_execution_value: 0,
            is_complete: false,
        };

        let children = vec![
            ChildExecution {
                parent_order_id: 2,
                child_order_id: 1,
                execution_price: 9995,
                execution_quantity: 1000,
                execution_time_ns: 2000,
                market_price_at_exec: 9997,
            },
        ];

        let result = tracker.calculate_order_shortfall(&arrival, &children);
        assert!(result.shortfall_ticks > 0); // Received less than arrival
    }

    #[test]
    fn test_quality_score() {
        let tracker = ArrivalPriceTracker::new(1);

        let arrival = ArrivalPriceRecord {
            parent_order_id: 3,
            side: Side::Buy,
            arrival_price: 10000,
            arrival_time_ns: 1000,
            total_quantity: 100,
            executed_quantity: 0,
            total_execution_value: 0,
            is_complete: false,
        };

        // Perfect execution at arrival price
        let children = vec![
            ChildExecution {
                parent_order_id: 3,
                child_order_id: 1,
                execution_price: 10000,
                execution_quantity: 100,
                execution_time_ns: 2000,
                market_price_at_exec: 10000,
            },
        ];

        let result = tracker.calculate_order_shortfall(&arrival, &children);
        assert_eq!(result.shortfall_ticks, 0);
        assert_eq!(result.quality_score, 100.0);
    }
}
