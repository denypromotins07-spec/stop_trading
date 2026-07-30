//! Risk Module Root
//! 
//! Wires the pre-trade risk bus into the smart order router.
//! Exports all risk management components.

pub mod pre_trade;
pub mod margin;

pub use pre_trade::{
    PreTradeRiskValidator,
    PreTradeRiskParams,
    FatFingerResult,
    IdempotentOrderIdGenerator,
};

pub use margin::{
    MarginCalculator,
    SymbolMarginState,
    MarginMode,
    BalanceType,
    MarginUpdateEvent,
};

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Smart Order Router integration with risk management
#[repr(C)]
pub struct RiskAwareRouter {
    /// Pre-trade validator
    validator: Arc<PreTradeRiskValidator>,
    /// Margin calculator
    margin_calc: Arc<MarginCalculator>,
    /// Router active flag
    is_active: AtomicBool,
    /// Orders routed counter
    orders_routed: margin::PaddedAtomicU64,
    /// Orders rejected counter
    orders_rejected: margin::PaddedAtomicU64,
}

impl RiskAwareRouter {
    pub fn new(validator: Arc<PreTradeRiskValidator>, margin_calc: Arc<MarginCalculator>) -> Self {
        Self {
            validator,
            margin_calc,
            is_active: AtomicBool::new(true),
            orders_routed: margin::PaddedAtomicU64::new(0),
            orders_rejected: margin::PaddedAtomicU64::new(0),
        }
    }

    /// Validate and route an order
    /// Returns Some(order_id) if approved, None if rejected
    #[inline]
    pub fn validate_and_route(
        &self,
        side: bool,
        price: u64,
        size: u64,
        mid_price: u64,
    ) -> Option<u64> {
        if !self.is_active.load(Ordering::Acquire) {
            self.orders_rejected.fetch_add(1, Ordering::Relaxed);
            return None;
        }

        // Check margin health first
        if !self.margin_calc.is_healthy() {
            self.orders_rejected.fetch_add(1, Ordering::Relaxed);
            return None;
        }

        // Run pre-trade validation
        match self.validator.validate_order(side, price, size, mid_price) {
            FatFingerResult::Ok => {
                // Generate idempotent order ID
                let order_id = self.validator.generate_order_id();
                self.orders_routed.fetch_add(1, Ordering::Relaxed);
                Some(order_id)
            }
            _ => {
                self.orders_rejected.fetch_add(1, Ordering::Relaxed);
                None
            }
        }
    }

    /// Confirm execution and update position
    #[inline]
    pub fn confirm_execution(&self, side: bool, size: u64, pnl_delta: i64) {
        self.validator.update_position(side, size);
        self.margin_calc.update_total_equity(pnl_delta);
    }

    /// Get validator reference
    #[inline]
    pub fn get_validator(&self) -> &PreTradeRiskValidator {
        &self.validator
    }

    /// Get margin calculator reference
    #[inline]
    pub fn get_margin_calc(&self) -> &MarginCalculator {
        &self.margin_calc
    }

    /// Activate router
    #[inline]
    pub fn activate(&self) {
        self.is_active.store(true, Ordering::Release);
        self.validator.activate();
    }

    /// Deactivate router (kill switch)
    #[inline]
    pub fn deactivate(&self) {
        self.is_active.store(false, Ordering::Release);
        self.validator.deactivate();
    }

    /// Get statistics
    #[inline]
    pub fn get_stats(&self) -> (u64, u64) {
        (
            self.orders_routed.load(Ordering::Relaxed),
            self.orders_rejected.load(Ordering::Relaxed),
        )
    }
}

/// Risk bus for broadcasting risk events
#[repr(C)]
pub struct RiskBus {
    /// Kill signal flag
    kill_signal: AtomicBool,
    /// Margin warning flag
    margin_warning: AtomicBool,
    /// Position limit warning flag
    position_warning: AtomicBool,
}

impl RiskBus {
    pub fn new() -> Self {
        Self {
            kill_signal: AtomicBool::new(false),
            margin_warning: AtomicBool::new(false),
            position_warning: AtomicBool::new(false),
        }
    }

    #[inline]
    pub fn trigger_kill(&self) {
        self.kill_signal.store(true, Ordering::Release);
    }

    #[inline]
    pub fn clear_kill(&self) {
        self.kill_signal.store(false, Ordering::Release);
    }

    #[inline]
    pub fn is_killed(&self) -> bool {
        self.kill_signal.load(Ordering::Acquire)
    }

    #[inline]
    pub fn trigger_margin_warning(&self) {
        self.margin_warning.store(true, Ordering::Release);
    }

    #[inline]
    pub fn clear_margin_warning(&self) {
        self.margin_warning.store(false, Ordering::Release);
    }

    #[inline]
    pub fn has_margin_warning(&self) -> bool {
        self.margin_warning.load(Ordering::Acquire)
    }

    #[inline]
    pub fn trigger_position_warning(&self) {
        self.position_warning.store(true, Ordering::Release);
    }

    #[inline]
    pub fn clear_position_warning(&self) {
        self.position_warning.store(false, Ordering::Release);
    }

    #[inline]
    pub fn has_position_warning(&self) -> bool {
        self.position_warning.load(Ordering::Acquire)
    }
}

impl Default for RiskBus {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_risk_aware_router() {
        let validator = Arc::new(PreTradeRiskValidator::new(
            12345,
            PreTradeRiskParams::default(),
        ));
        let margin_calc = Arc::new(MarginCalculator::new(1_000_000_000, 8000));
        
        let router = RiskAwareRouter::new(validator, margin_calc);
        
        let mid_price = 50_000_000;
        let result = router.validate_and_route(true, 50_000_000, 1_000_000, mid_price);
        assert!(result.is_some());

        let (routed, rejected) = router.get_stats();
        assert_eq!(routed, 1);
        assert_eq!(rejected, 0);
    }

    #[test]
    fn test_risk_bus() {
        let bus = RiskBus::new();
        assert!(!bus.is_killed());
        
        bus.trigger_kill();
        assert!(bus.is_killed());
        
        bus.clear_kill();
        assert!(!bus.is_killed());
    }
}
