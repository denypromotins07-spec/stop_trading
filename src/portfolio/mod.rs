//! Portfolio Module Root
//! 
//! Manages global netting logic and exposure limits.
//! Exports all portfolio management components including HRP and Risk Parity.

pub mod state;
pub mod reconciliation;
pub mod hrp;
pub mod risk_parity;
pub mod const_mod;

pub use state::{
    PortfolioState,
    SymbolRiskState,
    ExposureSummary,
    DeltaUpdateEvent,
};

pub use reconciliation::{
    ReconciliationEngine,
    ReconciliationRunner,
    ReconciliationConfig,
    ReconciliationStatus,
    ReconciliationEvent,
    ReconciliationStats,
    ExchangeSnapshot,
    InternalSnapshot,
};

pub use hrp::{
    HierarchicalRiskParity,
    CovarianceMatrix,
    HRPArena,
    HRPError,
    MAX_ASSETS,
};

pub use risk_parity::{
    RiskParityOptimizer,
    RiskParityResult,
    CachedRiskParity,
    RiskBudgetValidator,
    RiskValidationError,
    AdaptiveRiskParity,
    VolatilityRegime,
    MAX_RISK_ASSETS,
};

pub use const_mod::{
    PortfolioConstructor,
    AllocationStrategy,
    PortfolioWeights,
    ConstructionError,
};

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

/// Global portfolio manager coordinating state and reconciliation
#[repr(C)]
pub struct PortfolioManager {
    /// Portfolio state tracker
    state: Arc<PortfolioState>,
    /// Reconciliation engine
    reconciliation: Arc<ReconciliationEngine>,
    /// Netting enabled flag
    netting_enabled: AtomicBool,
    /// Auto-hedge enabled flag
    auto_hedge_enabled: AtomicBool,
    /// Hedge threshold (basis points of delta)
    hedge_threshold_bps: AtomicU64,
}

impl PortfolioManager {
    pub fn new(
        max_delta: i64,
        max_gamma: i64,
        max_notional: u64,
        reconciliation_config: ReconciliationConfig,
    ) -> Self {
        Self {
            state: Arc::new(PortfolioState::new(max_delta, max_gamma, max_notional)),
            reconciliation: Arc::new(ReconciliationEngine::new(reconciliation_config)),
            netting_enabled: AtomicBool::new(true),
            auto_hedge_enabled: AtomicBool::new(false),
            hedge_threshold_bps: AtomicU64::new(5000), // 50% delta threshold
        }
    }

    /// Aggregate symbol state into portfolio
    #[inline]
    pub fn aggregate_symbol(&self, symbol_state: SymbolRiskState) {
        self.state.aggregate_symbol(symbol_state);
    }

    /// Check if portfolio is within limits
    #[inline]
    pub fn is_within_limits(&self) -> bool {
        !self.state.are_limits_exceeded()
    }

    /// Get current exposure summary
    #[inline]
    pub fn get_exposure(&self) -> ExposureSummary {
        self.state.get_exposure_summary()
    }

    /// Check if auto-hedge should trigger
    #[inline]
    pub fn should_hedge(&self) -> bool {
        if !self.auto_hedge_enabled.load(Ordering::Acquire) {
            return false;
        }

        let delta = self.state.get_global_delta().abs();
        let threshold = self.hedge_threshold_bps.load(Ordering::Acquire);
        
        // If delta exceeds threshold (scaled), trigger hedge
        delta > (threshold as i64 * 1_000_000 / 10_000)
    }

    /// Run reconciliation check
    #[inline]
    pub fn reconcile(
        &self,
        internal: InternalSnapshot,
        exchange: ExchangeSnapshot,
    ) -> ReconciliationStatus {
        self.reconciliation.reconcile(internal, exchange)
    }

    /// Get portfolio state reference
    #[inline]
    pub fn get_state(&self) -> &PortfolioState {
        &self.state
    }

    /// Get reconciliation engine reference
    #[inline]
    pub fn get_reconciliation(&self) -> &ReconciliationEngine {
        &self.reconciliation
    }

    /// Enable netting
    #[inline]
    pub fn enable_netting(&self) {
        self.netting_enabled.store(true, Ordering::Release);
    }

    /// Disable netting
    #[inline]
    pub fn disable_netting(&self) {
        self.netting_enabled.store(false, Ordering::Release);
    }

    /// Check if netting is enabled
    #[inline]
    pub fn is_netting_enabled(&self) -> bool {
        self.netting_enabled.load(Ordering::Acquire)
    }

    /// Enable auto-hedging
    #[inline]
    pub fn enable_auto_hedge(&self) {
        self.auto_hedge_enabled.store(true, Ordering::Release);
    }

    /// Disable auto-hedging
    #[inline]
    pub fn disable_auto_hedge(&self) {
        self.auto_hedge_enabled.store(false, Ordering::Release);
    }

    /// Set hedge threshold
    #[inline]
    pub fn set_hedge_threshold_bps(&self, threshold: u64) {
        self.hedge_threshold_bps.store(threshold, Ordering::Release);
    }

    /// Start reconciliation loop
    #[inline]
    pub fn start_reconciliation(&self) {
        self.reconciliation.start();
    }

    /// Stop reconciliation loop
    #[inline]
    pub fn stop_reconciliation(&self) {
        self.reconciliation.stop();
    }

    /// Get combined health status
    #[inline]
    pub fn is_healthy(&self) -> bool {
        self.is_within_limits() && self.reconciliation.is_running()
    }
}

/// Netting result for order aggregation
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct NettingResult {
    /// Symbol hash
    pub symbol_hash: u64,
    /// Net quantity (positive = buy, negative = sell)
    pub net_quantity: i64,
    /// Number of orders netted
    pub order_count: u64,
    /// Average price (scaled)
    pub avg_price: u64,
}

/// Order for netting
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PendingOrder {
    pub symbol_hash: u64,
    pub side: bool, // true = buy, false = sell
    pub quantity: u64,
    pub price: u64,
    pub order_id: u64,
    pub timestamp_ns: u64,
}

impl PendingOrder {
    pub fn new(symbol_hash: u64, side: bool, quantity: u64, price: u64, order_id: u64) -> Self {
        let timestamp_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        Self {
            symbol_hash,
            side,
            quantity,
            price,
            order_id,
            timestamp_ns,
        }
    }
}

/// Simple netting function (can be expanded for more complex strategies)
#[inline]
pub fn net_orders(orders: &[PendingOrder]) -> Vec<NettingResult> {
    use std::collections::HashMap;
    
    let mut net_map: HashMap<u64, (i64, u64, u64)> = HashMap::new(); // symbol -> (net_qty, total_value, count)
    
    for order in orders {
        let entry = net_map.entry(order.symbol_hash).or_insert((0, 0, 0));
        let qty_signed = if order.side { order.quantity as i64 } else { -(order.quantity as i64) };
        entry.0 += qty_signed;
        entry.1 += order.quantity * order.price;
        entry.2 += 1;
    }
    
    net_map
        .into_iter()
        .map(|(symbol_hash, (net_qty, total_value, count))| {
            let avg_price = if net_qty != 0 {
                total_value / net_qty.unsigned_abs()
            } else {
                0
            };
            NettingResult {
                symbol_hash,
                net_quantity: net_qty,
                order_count: count,
                avg_price,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_portfolio_manager() {
        let config = ReconciliationConfig::default();
        let manager = PortfolioManager::new(
            1_000_000_000,  // max delta
            100_000_000,    // max gamma
            10_000_000_000, // max notional
            config,
        );

        assert!(manager.is_within_limits());
        assert!(!manager.should_hedge()); // Auto-hedge disabled by default

        let exposure = manager.get_exposure();
        assert_eq!(exposure.global_delta, 0);
    }

    #[test]
    fn test_order_netting() {
        let orders = vec![
            PendingOrder::new(12345, true, 100, 50_000, 1),
            PendingOrder::new(12345, true, 50, 50_000, 2),
            PendingOrder::new(12345, false, 75, 50_000, 3),
        ];

        let results = net_orders(&orders);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].net_quantity, 75); // 100 + 50 - 75
        assert_eq!(results[0].order_count, 3);
    }
}
