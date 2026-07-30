//! Slippage Model Module
//! Real-time slippage and adverse selection model using high-res PTP clock.
//! Compares theoretical microprice at decision nanosecond vs actual exchange execution price.

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

/// Single trade execution record
#[derive(Debug, Clone, Copy)]
pub struct ExecutionRecord {
    pub order_id: u64,
    pub side: Side,
    pub decision_price: i64,
    pub decision_time_ns: u64,
    pub execution_price: i64,
    pub execution_time_ns: u64,
    pub quantity: u64,
    pub venue_latency_ns: u64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Side {
    Buy,
    Sell,
}

/// Slippage calculation result
#[derive(Debug, Clone, Copy)]
pub struct SlippageResult {
    pub order_id: u64,
    /// Slippage in ticks (positive = unfavorable)
    pub slippage_ticks: i64,
    /// Slippage in basis points
    pub slippage_bps: f64,
    /// Adverse selection cost
    pub adverse_selection: f64,
    /// Market impact estimate
    pub market_impact: f64,
    /// Total cost in base currency units
    pub total_cost: i64,
}

/// Lock-free slippage tracker
pub struct SlippageTracker {
    /// Total executions tracked
    execution_count: CachePadded<AtomicU64>,
    /// Total slippage (in ticks * quantity)
    total_slippage: CachePadded<AtomicI64>,
    /// Total adverse selection
    total_adverse_selection: CachePadded<AtomicI64>,
    /// Tick size
    tick_size: i64,
    /// Active flag
    is_active: CachePadded<AtomicBool>,
    /// Rolling window sum for average calculation
    rolling_slippage_sum: CachePadded<AtomicI64>,
    rolling_count: CachePadded<AtomicU64>,
    rolling_window_size: u64,
}

impl SlippageTracker {
    pub fn new(tick_size: i64, window_size: u64) -> Self {
        Self {
            execution_count: CachePadded::default(),
            total_slippage: CachePadded::default(),
            total_adverse_selection: CachePadded::default(),
            tick_size,
            is_active: CachePadded::new(AtomicBool::new(true)),
            rolling_slippage_sum: CachePadded::default(),
            rolling_count: CachePadded::default(),
            rolling_window_size: window_size,
        }
    }

    /// Record an execution and calculate slippage
    pub fn record_execution(&self, exec: ExecutionRecord) -> SlippageResult {
        if !self.is_active.data.load(Ordering::Acquire) {
            return self.create_empty_result(exec.order_id);
        }

        // Calculate slippage
        let slippage_ticks = match exec.side {
            Side::Buy => exec.execution_price - exec.decision_price,
            Side::Sell => exec.decision_price - exec.execution_price,
        };

        // Convert to basis points
        let slippage_bps = if exec.decision_price > 0 {
            (slippage_ticks as f64 * self.tick_size as f64 / exec.decision_price as f64) * 10_000.0
        } else {
            0.0
        };

        // Estimate adverse selection (price movement after execution)
        // This would typically use post-execution price data
        let adverse_selection = self.estimate_adverse_selection(&exec);

        // Estimate market impact based on quantity and latency
        let market_impact = self.estimate_market_impact(exec.quantity, exec.venue_latency_ns);

        // Total cost
        let total_cost = slippage_ticks * self.tick_size * exec.quantity as i64;

        // Update counters
        self.execution_count.data.fetch_add(1, Ordering::AcqRel);
        self.total_slippage.data.fetch_add(slippage_ticks * exec.quantity as i64, Ordering::AcqRel);
        self.total_adverse_selection
            .data
            .fetch_add((adverse_selection * 1_000_000.0) as i64, Ordering::AcqRel);

        // Update rolling window
        self.update_rolling_window(slippage_ticks);

        SlippageResult {
            order_id: exec.order_id,
            slippage_ticks,
            slippage_bps,
            adverse_selection,
            market_impact,
            total_cost,
        }
    }

    /// Estimate adverse selection cost
    fn estimate_adverse_selection(&self, exec: &ExecutionRecord) -> f64 {
        // Simplified model: adverse selection increases with latency
        // In production, this would compare to mid-price movement after execution
        let latency_ms = exec.venue_latency_ns as f64 / 1_000_000.0;
        
        // Assume 0.1 bps adverse selection per ms of latency
        let base_adverse = latency_ms * 0.1;
        
        // Adjust for volatility proxy (latency variance could indicate volatile periods)
        let volatility_adjustment = 1.0 + (latency_ms / 100.0).min(1.0);
        
        base_adverse * volatility_adjustment
    }

    /// Estimate market impact
    fn estimate_market_impact(&self, quantity: u64, latency_ns: u64) -> f64 {
        // Simple square-root impact model: impact = a * sqrt(quantity / avg_volume)
        // Simplified here to just scale with quantity
        
        let qty_factor = (quantity as f64 / 1000.0).sqrt();
        let latency_factor = 1.0 + (latency_ns as f64 / 1_000_000_000.0); // 1s baseline
        
        qty_factor * latency_factor * 0.5 // Scale factor
    }

    /// Update rolling window statistics
    fn update_rolling_window(&self, slippage_ticks: i64) {
        let count = self.rolling_count.data.load(Ordering::Acquire);
        
        if count >= self.rolling_window_size {
            // Remove oldest entry (simplified - would need queue in production)
            let old_sum = self.rolling_slippage_sum.data.load(Ordering::Acquire);
            let adjusted_sum = old_sum - (old_sum / self.rolling_window_size as i64);
            self.rolling_slippage_sum.data.store(
                adjusted_sum + slippage_ticks,
                Ordering::Release,
            );
        } else {
            self.rolling_slippage_sum
                .data
                .fetch_add(slippage_ticks, Ordering::AcqRel);
            self.rolling_count.data.fetch_add(1, Ordering::AcqRel);
        }
    }

    /// Get average slippage over rolling window
    pub fn get_average_slippage(&self) -> f64 {
        let count = self.rolling_count.data.load(Ordering::Acquire);
        if count == 0 {
            return 0.0;
        }
        
        let sum = self.rolling_slippage_sum.data.load(Ordering::Acquire);
        sum as f64 / count as f64
    }

    /// Get overall statistics
    pub fn get_stats(&self) -> SlippageStats {
        let count = self.execution_count.data.load(Ordering::Acquire);
        let total_slip = self.total_slippage.data.load(Ordering::Acquire);
        
        SlippageStats {
            execution_count: count,
            total_slippage_ticks: total_slip,
            average_slippage_ticks: if count > 0 { total_slip as f64 / count as f64 } else { 0.0 },
            average_slippage_bps: self.get_average_slippage() as f64 * self.tick_size as f64 * 10_000.0 / 10_000.0,
            total_adverse_selection: self.total_adverse_selection.data.load(Ordering::Acquire),
            is_active: self.is_active.data.load(Ordering::Acquire),
        }
    }

    fn create_empty_result(&self, order_id: u64) -> SlippageResult {
        SlippageResult {
            order_id,
            slippage_ticks: 0,
            slippage_bps: 0.0,
            adverse_selection: 0.0,
            market_impact: 0.0,
            total_cost: 0,
        }
    }

    #[inline]
    pub fn set_active(&self, active: bool) {
        self.is_active.data.store(active, Ordering::Release);
    }

    #[inline]
    pub fn reset(&self) {
        self.execution_count.data.store(0, Ordering::Release);
        self.total_slippage.data.store(0, Ordering::Release);
        self.total_adverse_selection.data.store(0, Ordering::Release);
        self.rolling_slippage_sum.data.store(0, Ordering::Release);
        self.rolling_count.data.store(0, Ordering::Release);
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SlippageStats {
    pub execution_count: u64,
    pub total_slippage_ticks: i64,
    pub average_slippage_ticks: f64,
    pub average_slippage_bps: f64,
    pub total_adverse_selection: i64,
    pub is_active: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slippage_tracker_buy() {
        let tracker = SlippageTracker::new(1, 100);
        
        let exec = ExecutionRecord {
            order_id: 1,
            side: Side::Buy,
            decision_price: 10000,
            decision_time_ns: 1000,
            execution_price: 10005, // Worse by 5 ticks
            execution_time_ns: 2000,
            quantity: 100,
            venue_latency_ns: 1_000_000, // 1ms
        };

        let result = tracker.record_execution(exec);
        assert_eq!(result.slippage_ticks, 5);
        assert!(result.slippage_bps > 0.0);
        assert!(result.adverse_selection > 0.0);
    }

    #[test]
    fn test_slippage_tracker_sell() {
        let tracker = SlippageTracker::new(1, 100);
        
        let exec = ExecutionRecord {
            order_id: 1,
            side: Side::Sell,
            decision_price: 10000,
            decision_time_ns: 1000,
            execution_price: 9995, // Worse by 5 ticks
            execution_time_ns: 2000,
            quantity: 100,
            venue_latency_ns: 1_000_000,
        };

        let result = tracker.record_execution(exec);
        assert_eq!(result.slippage_ticks, 5);
    }

    #[test]
    fn test_favorable_execution() {
        let tracker = SlippageTracker::new(1, 100);
        
        let exec = ExecutionRecord {
            order_id: 1,
            side: Side::Buy,
            decision_price: 10000,
            decision_time_ns: 1000,
            execution_price: 9998, // Better by 2 ticks
            execution_time_ns: 2000,
            quantity: 100,
            venue_latency_ns: 500_000,
        };

        let result = tracker.record_execution(exec);
        assert_eq!(result.slippage_ticks, -2); // Negative = favorable
    }
}
