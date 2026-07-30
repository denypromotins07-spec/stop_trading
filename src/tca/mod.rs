//! TCA Module Root
//! Stream execution analytics directly to the SOUL.md feedback loop.

pub mod slippage_model;
pub mod arrival_price;

pub use slippage_model::{
    SlippageTracker,
    ExecutionRecord,
    SlippageResult,
    SlippageStats,
};

pub use arrival_price::{
    ArrivalPriceTracker,
    ArrivalPriceRecord,
    ChildExecution,
    ImplementationShortfall,
    ArrivalPriceStats,
};

use std::sync::atomic::{AtomicU64, AtomicI64, AtomicBool, Ordering};

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

/// Aggregated TCA metrics for feedback loop
#[derive(Debug, Clone, Copy)]
pub struct TCAMetrics {
    /// Total executions analyzed
    pub total_executions: u64,
    /// Average slippage in bps
    pub avg_slippage_bps: f64,
    /// Average implementation shortfall in bps
    pub avg_shortfall_bps: f64,
    /// Market impact component (bps)
    pub avg_market_impact_bps: f64,
    /// Timing cost component (bps)
    pub avg_timing_cost_bps: f64,
    /// Adverse selection cost (bps)
    pub avg_adverse_selection_bps: f64,
    /// Overall execution quality score (0-100)
    pub quality_score: f64,
    /// Timestamp (nanoseconds)
    pub timestamp_ns: u64,
}

/// TCA Engine combining all analytics
pub struct TCAEngine {
    /// Slippage tracker
    slippage_tracker: SlippageTracker,
    /// Arrival price tracker
    arrival_tracker: ArrivalPriceTracker,
    /// Total analysis count
    analysis_count: CachePadded<AtomicU64>,
    /// Cumulative quality score
    cumulative_quality: CachePadded<AtomicI64>,
    /// Active flag
    is_active: CachePadded<AtomicBool>,
}

impl TCAEngine {
    pub fn new(tick_size: i64, slippage_window: u64) -> Self {
        Self {
            slippage_tracker: SlippageTracker::new(tick_size, slippage_window),
            arrival_tracker: ArrivalPriceTracker::new(tick_size),
            analysis_count: CachePadded::default(),
            cumulative_quality: CachePadded::default(),
            is_active: CachePadded::new(AtomicBool::new(true)),
        }
    }

    /// Record an execution for slippage analysis
    pub fn record_execution(&self, exec: ExecutionRecord) -> SlippageResult {
        if !self.is_active.data.load(Ordering::Acquire) {
            return self.create_empty_slippage_result(exec.order_id);
        }

        let result = self.slippage_tracker.record_execution(exec);
        self.analysis_count.data.fetch_add(1, Ordering::AcqRel);
        result
    }

    /// Record arrival of a parent order
    pub fn record_arrival(&self, arrival: &ArrivalPriceRecord) {
        if !self.is_active.data.load(Ordering::Acquire) {
            return;
        }
        self.arrival_tracker.record_arrival(arrival);
    }

    /// Record child execution and calculate implementation shortfall
    pub fn record_child_execution(
        &self,
        child: &ChildExecution,
        arrival: &ArrivalPriceRecord,
    ) -> ImplementationShortfall {
        if !self.is_active.data.load(Ordering::Acquire) {
            return self.create_empty_shortfall(child.parent_order_id);
        }

        let result = self.arrival_tracker.record_child_execution(child, arrival);
        
        // Update cumulative quality
        let quality_scaled = (result.quality_score * 1000.0) as i64;
        self.cumulative_quality
            .data
            .fetch_add(quality_scaled, Ordering::AcqRel);
        
        result
    }

    /// Calculate aggregate metrics for feedback loop
    pub fn generate_metrics(&self) -> TCAMetrics {
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or(std::time::Duration::ZERO)
            .as_nanos() as u64;

        let slip_stats = self.slippage_tracker.get_stats();
        let arrival_stats = self.arrival_tracker.get_stats();
        let analysis_count = self.analysis_count.data.load(Ordering::Acquire);

        // Calculate average quality score
        let cumulative_quality = self.cumulative_quality.data.load(Ordering::Acquire);
        let avg_quality = if analysis_count > 0 {
            cumulative_quality as f64 / analysis_count as f64 / 1000.0
        } else {
            100.0
        };

        TCAMetrics {
            total_executions: slip_stats.execution_count,
            avg_slippage_bps: slip_stats.average_slippage_bps,
            avg_shortfall_bps: arrival_stats.average_shortfall_ticks, // Simplified
            avg_market_impact_bps: 0.0, // Would need more detailed tracking
            avg_timing_cost_bps: 0.0,
            avg_adverse_selection_bps: slip_stats.total_adverse_selection as f64 
                / slip_stats.execution_count.max(1) as f64 / 1_000_000.0,
            quality_score: avg_quality,
            timestamp_ns: now_ns,
        }
    }

    /// Get recommendation for algorithm aggression adjustment
    pub fn get_aggression_recommendation(&self, metrics: &TCAMetrics) -> AggressionAdjustment {
        // Simple heuristic-based recommendation
        let slippage_threshold = 5.0; // 5 bps
        let quality_threshold = 70.0;

        if metrics.avg_slippage_bps > slippage_threshold * 2.0 || metrics.quality_score < quality_threshold - 20.0 {
            AggressionAdjustment::Decrease
        } else if metrics.avg_slippage_bps < slippage_threshold / 2.0 && metrics.quality_score > quality_threshold + 20.0 {
            AggressionAdjustment::Increase
        } else {
            AggressionAdjustment::Maintain
        }
    }

    fn create_empty_slippage_result(&self, order_id: u64) -> SlippageResult {
        SlippageResult {
            order_id,
            slippage_ticks: 0,
            slippage_bps: 0.0,
            adverse_selection: 0.0,
            market_impact: 0.0,
            total_cost: 0,
        }
    }

    fn create_empty_shortfall(&self, parent_order_id: u64) -> ImplementationShortfall {
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
        self.slippage_tracker.set_active(active);
        self.arrival_tracker.set_active(active);
    }

    #[inline]
    pub fn is_active(&self) -> bool {
        self.is_active.data.load(Ordering::Acquire)
    }

    pub fn reset(&self) {
        self.slippage_tracker.reset();
        self.arrival_tracker.reset();
        self.analysis_count.data.store(0, Ordering::Release);
        self.cumulative_quality.data.store(0, Ordering::Release);
    }
}

/// Recommendation for algorithm aggression
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AggressionAdjustment {
    Increase,
    Decrease,
    Maintain,
}

#[cfg(test)]
mod tests {
    use super::*;
    use slippage_model::Side as SlipSide;

    #[test]
    fn test_tca_engine_basic() {
        let engine = TCAEngine::new(1, 100);

        // Record an execution
        let exec = ExecutionRecord {
            order_id: 1,
            side: SlipSide::Buy,
            decision_price: 10000,
            decision_time_ns: 1000,
            execution_price: 10005,
            execution_time_ns: 2000,
            quantity: 100,
            venue_latency_ns: 1_000_000,
        };

        let result = engine.record_execution(exec);
        assert!(result.slippage_ticks > 0);

        // Generate metrics
        let metrics = engine.generate_metrics();
        assert!(metrics.total_executions >= 1);
    }

    #[test]
    fn test_aggression_recommendation() {
        let engine = TCAEngine::new(1, 100);

        // Good metrics should suggest increase
        let good_metrics = TCAMetrics {
            total_executions: 100,
            avg_slippage_bps: 1.0,
            avg_shortfall_bps: 1.0,
            avg_market_impact_bps: 0.5,
            avg_timing_cost_bps: 0.5,
            avg_adverse_selection_bps: 0.2,
            quality_score: 95.0,
            timestamp_ns: 1000,
        };

        let rec = engine.get_aggression_recommendation(&good_metrics);
        assert_eq!(rec, AggressionAdjustment::Increase);

        // Bad metrics should suggest decrease
        let bad_metrics = TCAMetrics {
            total_executions: 100,
            avg_slippage_bps: 15.0,
            avg_shortfall_bps: 15.0,
            avg_market_impact_bps: 5.0,
            avg_timing_cost_bps: 10.0,
            avg_adverse_selection_bps: 5.0,
            quality_score: 40.0,
            timestamp_ns: 1000,
        };

        let rec = engine.get_aggression_recommendation(&bad_metrics);
        assert_eq!(rec, AggressionAdjustment::Decrease);
    }
}
