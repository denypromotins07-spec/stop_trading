//! L3 Order Book Module Root
//! 
//! Integrates queue estimations with the smart order router to optimize
//! maker rebate capture.

pub mod l3_tracker;
pub mod queue_estimator;

use std::sync::Arc;
use std::time::{Duration, Instant};
use crate::orderbook::l3_tracker::{L3Tracker, Side, QueuePosition};
use crate::orderbook::queue_estimator::{QueueEstimator, QueueDepletionStats};

/// L3 module configuration
#[derive(Debug, Clone)]
pub struct L3Config {
    /// Max memory for L3 tracking (MB)
    pub max_l3_memory_mb: u64,
    /// Max memory for queue estimation (MB)
    pub max_estimator_memory_mb: u64,
    /// Trade buffer capacity per price level
    pub buffer_capacity: usize,
    /// Stale order purge interval
    pub purge_interval_ms: u64,
}

impl Default for L3Config {
    fn default() -> Self {
        Self {
            max_l3_memory_mb: 200,
            max_estimator_memory_mb: 100,
            buffer_capacity: 100,
            purge_interval_ms: 1000,
        }
    }
}

/// L3 module handle providing integrated access to tracking and estimation
pub struct L3Module {
    tracker: Arc<L3Tracker>,
    estimator: Arc<QueueEstimator>,
    config: L3Config,
    last_purge: std::sync::Mutex<Instant>,
}

impl L3Module {
    /// Create a new L3 module
    pub fn new(config: L3Config) -> Self {
        let tracker = Arc::new(L3Tracker::new(config.max_l3_memory_mb));
        let estimator = Arc::new(QueueEstimator::new(
            tracker.clone(),
            config.max_estimator_memory_mb,
            config.buffer_capacity,
        ));

        Self {
            tracker,
            estimator,
            config,
            last_purge: std::sync::Mutex::new(Instant::now()),
        }
    }

    /// Get reference to the L3 tracker
    pub fn tracker(&self) -> &Arc<L3Tracker> {
        &self.tracker
    }

    /// Get reference to the queue estimator
    pub fn estimator(&self) -> &Arc<QueueEstimator> {
        &self.estimator
    }

    /// Insert an order and get estimated time to top
    pub fn track_order(&self, position: QueuePosition) -> Result<Option<QueueDepletionStats>, &'static str> {
        self.tracker.insert_order(position.clone())?;
        
        // Give it a moment for the order to be registered
        let stats = self.estimator.estimate_time_to_top(position.order_id);
        
        Ok(stats)
    }

    /// Update order size and recalculate estimates
    pub fn update_order(&self, order_id: u64, new_remaining: u64) -> Option<QueueDepletionStats> {
        self.tracker.update_order_size(order_id, new_remaining)?;
        self.estimator.estimate_time_to_top(order_id)
    }

    /// Remove an order from tracking
    pub fn remove_order(&self, order_id: u64) -> Option<QueuePosition> {
        self.tracker.remove_order(order_id)
    }

    /// Add a trade tick for depletion analysis
    pub fn add_trade(&self, symbol: String, price: i64, size: u64, aggressor_side: crate::orderbook::queue_estimator::AggressorSide) -> Result<(), &'static str> {
        use crate::orderbook::queue_estimator::TradeTick;
        
        let trade = TradeTick {
            symbol,
            price,
            size,
            aggressor_side,
            timestamp_ns: Instant::now().duration_since(Instant::now() - Duration::from_secs(1)).as_nanos() as u64,
        };
        
        self.estimator.add_trade(trade)
    }

    /// Periodic maintenance: purge stale data
    pub fn maintenance(&self) -> usize {
        let now = Instant::now();
        let mut last_purge = self.last_purge.lock().unwrap();
        
        if now.duration_since(*last_purge).as_millis() >= self.config.purge_interval_ms as u128 {
            let purged = self.tracker.purge_stale(Duration::from_secs(60).as_nanos() as u64);
            *last_purge = now;
            purged
        } else {
            0
        }
    }

    /// Get total memory usage
    pub fn memory_usage(&self) -> u64 {
        self.tracker.memory_usage() + self.estimator.memory_usage()
    }

    /// Check if memory usage is within limits
    pub fn is_memory_ok(&self, threshold_mb: u64) -> bool {
        self.memory_usage() < threshold_mb * 1024 * 1024
    }

    /// Clear all data
    pub fn clear(&self) {
        self.tracker.clear();
        self.estimator.clear();
    }

    /// Generate maker rebate optimization recommendations
    pub fn get_rebate_recommendations(&self, symbol: &str) -> Vec<RebateRecommendation> {
        let mut recommendations = Vec::new();
        
        // Get all queue positions for this symbol
        // In production, this would iterate through actual tracked orders
        // For now, we return empty vec as placeholder
        
        recommendations
    }
}

/// Recommendation for optimizing maker rebates
#[derive(Debug, Clone)]
pub struct RebateRecommendation {
    pub order_id: u64,
    pub action: RebateAction,
    pub confidence: f64,
    pub expected_improvement_bps: f64,
    pub reason: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RebateAction {
    Hold,      // Keep current position
    AmendUp,   // Move price up in queue
    AmendDown, // Move price down in queue
    Cancel,    // Cancel and re-evaluate
    Wait,      // Wait for natural queue movement
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::orderbook::queue_estimator::AggressorSide;

    #[test]
    fn test_l3_module_basic() {
        let config = L3Config::default();
        let module = L3Module::new(config);

        let position = QueuePosition {
            order_id: 1,
            symbol: "BTCUSD".to_string(),
            side: Side::Bid,
            price: 50000,
            original_size: 100,
            remaining_size: 100,
            queue_position: 0,
            estimated_ahead_size: 50,
            timestamp_ns: 0,
            last_update_ns: 0,
        };

        assert!(module.track_order(position).is_ok());
        assert_eq!(module.tracker().total_orders(), 1);

        // Add a trade
        assert!(module.add_trade(
            "BTCUSD".to_string(),
            50000,
            10,
            AggressorSide::Sell
        ).is_ok());

        // Run maintenance
        let purged = module.maintenance();
        assert!(purged >= 0);
    }

    #[test]
    fn test_memory_tracking() {
        let config = L3Config::default();
        let module = L3Module::new(config);

        assert!(module.is_memory_ok(1000)); // Should be OK with 1GB threshold
        assert!(!module.is_memory_ok(0));   // Should fail with 0 threshold
    }
}
