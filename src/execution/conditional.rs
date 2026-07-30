//! Conditional Order Router
//! 
//! Implements complex boolean logic evaluation for conditional order execution
//! (e.g., IF CVD > X AND Spread < Y) in the hot path.

use std::sync::atomic::{AtomicU64, AtomicI64, AtomicBool, Ordering};
use crossbeam_utils::CachePadded;

/// Market condition types for conditional orders
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConditionType {
    /// Cumulative Volume Delta comparison
    CvdGreaterThan(i64),
    CvdLessThan(i64),
    /// Spread comparison (in micro-units)
    SpreadLessThan(u64),
    SpreadGreaterThan(u64),
    /// Price comparison
    PriceAbove(u64),
    PriceBelow(u64),
    /// Order book imbalance ratio (scaled by 1000)
    ImbalanceAbove(u32),
    ImbalanceBelow(u32),
    /// Volatility threshold (scaled by 1000)
    VolatilityBelow(u32),
    VolatilityAbove(u32),
    /// Microprice drift (scaled by 1000)
    MicropriceDriftAbove(i32),
    MicropriceDriftBelow(i32),
    /// Custom flag (for external signals)
    FlagSet(u8),
}

/// Logical operator for combining conditions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogicalOp {
    And,
    Or,
    Not,
}

/// A single condition node in the evaluation tree
#[derive(Clone)]
pub struct ConditionNode {
    pub condition: ConditionType,
    pub operator: LogicalOp,
    /// Index of left child (if any)
    pub left_child: Option<usize>,
    /// Index of right child (if any)
    pub right_child: Option<usize>,
}

/// Pre-compiled conditional expression for hot-path evaluation
pub struct ConditionalExpression {
    /// Fixed-size condition pool (max 16 conditions per expression)
    conditions: [Option<ConditionNode>; 16],
    /// Number of active conditions
    condition_count: usize,
    /// Root node index
    root_index: usize,
}

impl Default for ConditionalExpression {
    fn default() -> Self {
        Self {
            conditions: std::array::from_fn(|_| None),
            condition_count: 0,
            root_index: 0,
        }
    }
}

impl ConditionalExpression {
    /// Create a new empty conditional expression
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a simple condition as the root
    pub fn with_condition(mut self, condition: ConditionType) -> Self {
        if self.condition_count < 16 {
            self.conditions[self.condition_count] = Some(ConditionNode {
                condition,
                operator: LogicalOp::And,
                left_child: None,
                right_child: None,
            });
            self.root_index = self.condition_count;
            self.condition_count += 1;
        }
        self
    }

    /// Build an AND condition chain
    pub fn and(mut self, condition: ConditionType) -> Self {
        if self.condition_count < 16 && self.condition_count > 0 {
            let parent_idx = self.root_index;
            let new_idx = self.condition_count;
            
            // Update parent to be AND operation
            if let Some(ref mut parent) = self.conditions[parent_idx] {
                parent.operator = LogicalOp::And;
                parent.right_child = Some(new_idx);
            }
            
            self.conditions[new_idx] = Some(ConditionNode {
                condition,
                operator: LogicalOp::And,
                left_child: None,
                right_child: None,
            });
            self.condition_count += 1;
        }
        self
    }

    /// Build an OR condition chain
    pub fn or(mut self, condition: ConditionType) -> Self {
        if self.condition_count < 16 && self.condition_count > 0 {
            let parent_idx = self.root_index;
            let new_idx = self.condition_count;
            
            if let Some(ref mut parent) = self.conditions[parent_idx] {
                parent.operator = LogicalOp::Or;
                parent.right_child = Some(new_idx);
            }
            
            self.conditions[new_idx] = Some(ConditionNode {
                condition,
                operator: LogicalOp::Or,
                left_child: None,
                right_child: None,
            });
            self.condition_count += 1;
        }
        self
    }

    /// Evaluate the expression against current market state
    #[inline]
    pub fn evaluate(&self, ctx: &MarketContext) -> bool {
        if self.condition_count == 0 {
            return true;
        }
        self.evaluate_node(self.root_index, ctx)
    }

    #[inline]
    fn evaluate_node(&self, node_idx: usize, ctx: &MarketContext) -> bool {
        let node = match &self.conditions[node_idx] {
            Some(n) => n,
            None => return false,
        };

        let result = Self::eval_condition(node.condition, ctx);

        match node.operator {
            LogicalOp::And => {
                if !result {
                    return false;
                }
                if let Some(right) = node.right_child {
                    result && self.evaluate_node(right, ctx)
                } else {
                    result
                }
            }
            LogicalOp::Or => {
                if result {
                    return true;
                }
                if let Some(right) = node.right_child {
                    result || self.evaluate_node(right, ctx)
                } else {
                    result
                }
            }
            LogicalOp::Not => !result,
        }
    }

    #[inline]
    fn eval_condition(condition: ConditionType, ctx: &MarketContext) -> bool {
        match condition {
            ConditionType::CvdGreaterThan(threshold) => ctx.cvd > threshold,
            ConditionType::CvdLessThan(threshold) => ctx.cvd < threshold,
            ConditionType::SpreadLessThan(threshold) => ctx.spread_micros < threshold,
            ConditionType::SpreadGreaterThan(threshold) => ctx.spread_micros > threshold,
            ConditionType::PriceAbove(threshold) => ctx.current_price > threshold,
            ConditionType::PriceBelow(threshold) => ctx.current_price < threshold,
            ConditionType::ImbalanceAbove(threshold) => ctx.imbalance_ratio > threshold,
            ConditionType::ImbalanceBelow(threshold) => ctx.imbalance_ratio < threshold,
            ConditionType::VolatilityBelow(threshold) => ctx.volatility_scaled < threshold,
            ConditionType::VolatilityAbove(threshold) => ctx.volatility_scaled > threshold,
            ConditionType::MicropriceDriftAbove(threshold) => ctx.microprice_drift_scaled > threshold,
            ConditionType::MicropriceDriftBelow(threshold) => ctx.microprice_drift_scaled < threshold,
            ConditionType::FlagSet(flag_id) => ctx.flags & (1u64 << flag_id) != 0,
        }
    }
}

/// Market context for condition evaluation (cache-line optimized)
#[repr(C)]
pub struct MarketContext {
    /// Current price (micro-units)
    pub current_price: u64,
    /// Current spread (micro-units)
    pub spread_micros: u64,
    /// Cumulative Volume Delta
    pub cvd: i64,
    /// Order book imbalance ratio (scaled by 1000)
    pub imbalance_ratio: u32,
    /// Volatility measure (scaled by 1000)
    pub volatility_scaled: u32,
    /// Microprice drift (scaled by 1000)
    pub microprice_drift_scaled: i32,
    /// Custom flags bitmask
    pub flags: u64,
    /// Timestamp (nanoseconds)
    pub timestamp_ns: u64,
}

impl Default for MarketContext {
    fn default() -> Self {
        Self {
            current_price: 0,
            spread_micros: 0,
            cvd: 0,
            imbalance_ratio: 0,
            volatility_scaled: 0,
            microprice_drift_scaled: 0,
            flags: 0,
            timestamp_ns: 0,
        }
    }
}

/// Conditional order router managing multiple conditional orders
pub struct ConditionalOrderRouter {
    /// Active conditional orders (fixed size pool)
    orders: CachePadded<[Option<ConditionalOrder>; 64]>,
    /// Order count
    order_count: CachePadded<AtomicU64>,
    /// Execution counter
    executions: CachePadded<AtomicU64>,
    /// Rejections counter
    rejections: CachePadded<AtomicU64>,
    /// Router enabled flag
    enabled: CachePadded<AtomicBool>,
}

/// A conditional order ready for execution
pub struct ConditionalOrder {
    /// Unique order ID
    pub id: u64,
    /// The condition expression
    pub expression: ConditionalExpression,
    /// Child order to execute when condition is met
    pub child_order: ChildOrder,
    /// Whether order has been executed
    pub executed: AtomicBool,
    /// Whether order is cancelled
    pub cancelled: AtomicBool,
    /// Expiration timestamp (nanoseconds)
    pub expiration_ns: u64,
}

/// Child order specification
pub struct ChildOrder {
    /// Side: true = buy, false = sell
    pub is_buy: bool,
    /// Quantity in base units
    pub quantity: u64,
    /// Limit price in micro-units (0 = market)
    pub limit_price_micros: u64,
    /// Time in force
    pub tif: TimeInForce,
    /// Target venue
    pub venue_id: u8,
}

/// Time in force options
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeInForce {
    ImmediateOrCancel,
    FillOrKill,
    GoodTillCancelled,
    GoodTillDate(u64),
}

impl ConditionalOrderRouter {
    /// Create a new conditional order router
    pub fn new() -> Self {
        Self {
            orders: CachePadded::new(std::array::from_fn(|_| None)),
            order_count: CachePadded::new(AtomicU64::new(0)),
            executions: CachePadded::new(AtomicU64::new(0)),
            rejections: CachePadded::new(AtomicU64::new(0)),
            enabled: CachePadded::new(AtomicBool::new(true)),
        }
    }

    /// Add a new conditional order
    /// Returns order ID on success, None if pool is full
    pub fn add_order(
        &self,
        expression: ConditionalExpression,
        child_order: ChildOrder,
        expiration_ns: u64,
    ) -> Option<u64> {
        if !self.enabled.load(Ordering::Relaxed) {
            return None;
        }

        let count = self.order_count.load(Ordering::Relaxed);
        if count >= 64 {
            return None;
        }

        let id = count + 1;
        let order = ConditionalOrder {
            id,
            expression,
            child_order,
            executed: AtomicBool::new(false),
            cancelled: AtomicBool::new(false),
            expiration_ns,
        };

        self.orders.orders[count as usize] = Some(order);
        self.order_count.store(id, Ordering::Relaxed);
        
        Some(id)
    }

    /// Evaluate all conditions and execute triggered orders
    /// This is the hot path - must be extremely fast
    #[inline]
    pub fn evaluate_and_execute(&self, ctx: &MarketContext, executor: &mut dyn FnMut(&ChildOrder)) -> usize {
        if !self.enabled.load(Ordering::Relaxed) {
            return 0;
        }

        let count = self.order_count.load(Ordering::Relaxed);
        let mut triggered = 0;

        for i in 0..count as usize {
            if let Some(ref order) = self.orders.orders[i] {
                // Skip already executed or cancelled orders
                if order.executed.load(Ordering::Relaxed) || order.cancelled.load(Ordering::Relaxed) {
                    continue;
                }

                // Check expiration
                if ctx.timestamp_ns > order.expiration_ns && order.expiration_ns > 0 {
                    order.cancelled.store(true, Ordering::Relaxed);
                    self.rejections.fetch_add(1, Ordering::Relaxed);
                    continue;
                }

                // Evaluate condition (hot path)
                if order.expression.evaluate(ctx) {
                    executor(&order.child_order);
                    order.executed.store(true, Ordering::Relaxed);
                    self.executions.fetch_add(1, Ordering::Relaxed);
                    triggered += 1;
                }
            }
        }

        triggered
    }

    /// Cancel an order by ID
    pub fn cancel_order(&self, id: u64) -> bool {
        if id == 0 || id > 64 {
            return false;
        }

        if let Some(ref order) = self.orders.orders[id as usize - 1] {
            if !order.executed.load(Ordering::Relaxed) {
                order.cancelled.store(true, Ordering::Relaxed);
                return true;
            }
        }
        false
    }

    /// Get execution count
    #[inline]
    pub fn get_execution_count(&self) -> u64 {
        self.executions.load(Ordering::Relaxed)
    }

    /// Get rejection count
    #[inline]
    pub fn get_rejection_count(&self) -> u64 {
        self.rejections.load(Ordering::Relaxed)
    }

    /// Get active order count
    #[inline]
    pub fn get_active_count(&self) -> u64 {
        let total = self.order_count.load(Ordering::Relaxed);
        let mut active = 0;

        for i in 0..total as usize {
            if let Some(ref order) = self.orders.orders[i] {
                if !order.executed.load(Ordering::Relaxed) && !order.cancelled.load(Ordering::Relaxed) {
                    active += 1;
                }
            }
        }

        active
    }

    /// Enable the router
    pub fn enable(&self) {
        self.enabled.store(true, Ordering::Relaxed);
    }

    /// Disable the router
    pub fn disable(&self) {
        self.enabled.store(false, Ordering::Relaxed);
    }

    /// Clear all orders
    pub fn clear(&self) {
        for i in 0..64 {
            self.orders.orders[i] = None;
        }
        self.order_count.store(0, Ordering::Relaxed);
    }
}

impl Default for ConditionalOrderRouter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_condition() {
        let expr = ConditionalExpression::new()
            .with_condition(ConditionType::CvdGreaterThan(1000));

        let ctx_positive = MarketContext {
            cvd: 2000,
            ..Default::default()
        };

        let ctx_negative = MarketContext {
            cvd: 500,
            ..Default::default()
        };

        assert!(expr.evaluate(&ctx_positive));
        assert!(!expr.evaluate(&ctx_negative));
    }

    #[test]
    fn test_and_conditions() {
        let expr = ConditionalExpression::new()
            .with_condition(ConditionType::CvdGreaterThan(1000))
            .and(ConditionType::SpreadLessThan(500));

        let ctx_pass = MarketContext {
            cvd: 2000,
            spread_micros: 300,
            ..Default::default()
        };

        let ctx_fail_cvd = MarketContext {
            cvd: 500,
            spread_micros: 300,
            ..Default::default()
        };

        let ctx_fail_spread = MarketContext {
            cvd: 2000,
            spread_micros: 600,
            ..Default::default()
        };

        assert!(expr.evaluate(&ctx_pass));
        assert!(!expr.evaluate(&ctx_fail_cvd));
        assert!(!expr.evaluate(&ctx_fail_spread));
    }

    #[test]
    fn test_router_execution() {
        let router = ConditionalOrderRouter::new();
        
        let expr = ConditionalExpression::new()
            .with_condition(ConditionType::PriceAbove(50000000));

        let child = ChildOrder {
            is_buy: true,
            quantity: 100,
            limit_price_micros: 0,
            tif: TimeInForce::ImmediateOrCancel,
            venue_id: 1,
        };

        let order_id = router.add_order(expr, child, 0).unwrap();
        assert_eq!(order_id, 1);

        let ctx = MarketContext {
            current_price: 51000000,
            timestamp_ns: 1000000,
            ..Default::default()
        };

        let mut executed_orders: Vec<ChildOrder> = Vec::new();
        let trigger_count = router.evaluate_and_execute(&ctx, &mut |order| {
            executed_orders.push(ChildOrder {
                is_buy: order.is_buy,
                quantity: order.quantity,
                limit_price_micros: order.limit_price_micros,
                tif: order.tif,
                venue_id: order.venue_id,
            });
        });

        assert_eq!(trigger_count, 1);
        assert_eq!(router.get_execution_count(), 1);
    }
}
