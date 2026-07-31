//! Rebalancing Module Root
//! 
//! Integrates drift control with concurrent per-symbol actors.

pub mod drift;
pub mod algo;

pub use drift::{
    BlackLittermanInput, DriftAnalysis, DriftMonitor, PositionData, PositionDrift,
    RebalanceAction, Side, TradeInstruction, View, NoTradeRegion,
};
pub use algo::{
    ExecutionMode, Leg, MultiLegOrder, RebalancingAlgorithm, RebalanceResult,
    RebalanceStats, RebalanceTrigger, TriggerReason, TaxOptimizer, TaxHarvestOpportunity,
};

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;

/// Per-symbol rebalancing actor message
#[derive(Debug, Clone)]
pub enum ActorMessage {
    UpdateTarget(f64),
    ExecuteRebalance(MultiLegOrder),
    Halt,
    Resume,
}

/// Per-symbol rebalancing actor state
pub struct SymbolActor {
    pub symbol: [u8; 16],
    pub current_weight: f64,
    pub target_weight: f64,
    pub halted: AtomicBool,
    pub message_count: AtomicU64,
}

impl SymbolActor {
    pub fn new(symbol: [u8; 16], initial_target: f64) -> Self {
        SymbolActor {
            symbol,
            current_weight: 0.0,
            target_weight: initial_target,
            halted: AtomicBool::new(false),
            message_count: AtomicU64::new(0),
        }
    }

    pub fn handle_message(&self, msg: ActorMessage) -> ActorResponse {
        self.message_count.fetch_add(1, Ordering::Relaxed);

        match msg {
            ActorMessage::UpdateTarget(new_target) => {
                if !self.halted.load(Ordering::Relaxed) {
                    // In production, would update via channel
                    ActorResponse::TargetUpdated(new_target)
                } else {
                    ActorResponse::Halted
                }
            }
            ActorMessage::ExecuteRebalance(order) => {
                if !self.halted.load(Ordering::Relaxed) {
                    ActorResponse::RebalanceExecuted(order.total_value_usd)
                } else {
                    ActorResponse::Halted
                }
            }
            ActorMessage::Halt => {
                self.halted.store(true, Ordering::Relaxed);
                ActorResponse::Halted
            }
            ActorMessage::Resume => {
                self.halted.store(false, Ordering::Relaxed);
                ActorResponse::Resumed
            }
        }
    }

    pub fn is_halted(&self) -> bool {
        self.halted.load(Ordering::Relaxed)
    }
}

#[derive(Debug, Clone)]
pub enum ActorResponse {
    TargetUpdated(f64),
    RebalanceExecuted(f64),
    Halted,
    Resumed,
}

/// Concurrent rebalancing coordinator managing all symbol actors
pub struct RebalancingCoordinator {
    pub actors: Vec<Arc<SymbolActor>>,
    pub algorithm: RebalancingAlgorithm,
    pub coordination_enabled: AtomicBool,
    pub last_coordination_ts: AtomicU64,
}

impl RebalancingCoordinator {
    pub fn new(algorithm: RebalancingAlgorithm) -> Self {
        RebalancingCoordinator {
            actors: Vec::with_capacity(64),
            algorithm,
            coordination_enabled: AtomicBool::new(true),
            last_coordination_ts: AtomicU64::new(0),
        }
    }

    /// Register a symbol actor
    pub fn register_symbol(&mut self, symbol: [u8; 16], target_weight: f64) {
        if self.actors.len() < self.actors.capacity() {
            let actor = Arc::new(SymbolActor::new(symbol, target_weight));
            self.actors.push(actor);
        }
    }

    /// Coordinate rebalancing across all symbols
    pub fn coordinate_rebalance(&self, positions: &[PositionData], targets: &[f64]) -> CoordinationResult {
        if !self.coordination_enabled.load(Ordering::Relaxed) {
            return CoordinationResult {
                success: false,
                reason: "Coordination disabled",
                symbols_rebalanced: 0,
            };
        }

        // Check trigger
        let trigger = self.algorithm.check_trigger(positions, targets);
        
        if !trigger.triggered {
            return CoordinationResult {
                success: true,
                reason: "No rebalance needed",
                symbols_rebalanced: 0,
            };
        }

        // Build multi-leg order
        let order = self.algorithm.build_rebalance_order(positions, targets, ExecutionMode::Simultaneous);
        
        if let Some(order) = order {
            // Execute through algorithm
            let result = self.algorithm.execute_rebalance(&order);
            
            use std::time::{SystemTime, UNIX_EPOCH};
            self.last_coordination_ts.store(
                SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0),
                Ordering::Relaxed,
            );

            if result.success {
                // Notify relevant actors
                for leg in &order.legs {
                    if let Some(actor) = self.find_actor_for_symbol(&leg.symbol) {
                        actor.handle_message(ActorMessage::UpdateTarget(
                            targets.iter().position(|&t| t > 0.0).map(|i| targets[i]).unwrap_or(0.0)
                        ));
                    }
                }

                CoordinationResult {
                    success: true,
                    reason: "Rebalance executed",
                    symbols_rebalanced: order.legs.len(),
                }
            } else {
                CoordinationResult {
                    success: false,
                    reason: result.error.unwrap_or_else(|| "Unknown error".to_string()),
                    symbols_rebalanced: 0,
                }
            }
        } else {
            CoordinationResult {
                success: true,
                reason: "No valid order generated",
                symbols_rebalanced: 0,
            }
        }
    }

    fn find_actor_for_symbol(&self, symbol: &[u8; 16]) -> Option<Arc<SymbolActor>> {
        self.actors.iter().find(|a| a.symbol == *symbol).cloned()
    }

    /// Halt all actors (emergency stop)
    pub fn halt_all(&self) {
        for actor in &self.actors {
            actor.handle_message(ActorMessage::Halt);
        }
    }

    /// Resume all actors
    pub fn resume_all(&self) {
        for actor in &self.actors {
            actor.handle_message(ActorMessage::Resume);
        }
    }

    /// Get actor statistics
    pub fn get_actor_stats(&self) -> Vec<ActorStats> {
        self.actors.iter().map(|a| ActorStats {
            symbol: a.symbol,
            message_count: a.message_count.load(Ordering::Relaxed),
            is_halted: a.is_halted(),
        }).collect()
    }

    /// Enable/disable coordination
    pub fn set_coordination_enabled(&self, enabled: bool) {
        self.coordination_enabled.store(enabled, Ordering::Relaxed);
    }
}

/// Coordination result
#[derive(Debug, Clone)]
pub struct CoordinationResult {
    pub success: bool,
    pub reason: String,
    pub symbols_rebalanced: usize,
}

/// Actor statistics
#[derive(Debug, Clone)]
pub struct ActorStats {
    pub symbol: [u8; 16],
    pub message_count: u64,
    pub is_halted: bool,
}

/// Global rebalancing manager
pub struct GlobalRebalancingManager {
    pub coordinator: RebalancingCoordinator,
    pub tax_optimizer: TaxOptimizer,
    pub auto_rebalance_enabled: AtomicBool,
}

impl GlobalRebalancingManager {
    pub fn new(algorithm: RebalancingAlgorithm) -> Self {
        GlobalRebalancingManager {
            coordinator: RebalancingCoordinator::new(algorithm),
            tax_optimizer: TaxOptimizer::new(30),
            auto_rebalance_enabled: AtomicBool::new(true),
        }
    }

    /// Run automatic rebalancing check
    pub fn run_auto_rebalance(&self, positions: &[PositionData], cost_basis: &[f64]) -> AutoRebalanceReport {
        if !self.auto_rebalance_enabled.load(Ordering::Relaxed) {
            return AutoRebalanceReport {
                rebalance_triggered: false,
                tax_opportunities: vec![],
                message: "Auto-rebalance disabled".to_string(),
            };
        }

        // Calculate targets (equal weight for simplicity)
        let n = positions.len();
        let targets: Vec<f64> = if n > 0 {
            vec![1.0 / n as f64; n]
        } else {
            vec![]
        };

        // Check for tax harvesting opportunities
        let tax_opps = self.tax_optimizer.find_harvesting_opportunities(positions, cost_basis);

        // Run coordination
        let coord_result = self.coordinator.coordinate_rebalance(positions, &targets);

        AutoRebalanceReport {
            rebalance_triggered: coord_result.success && coord_result.symbols_rebalanced > 0,
            tax_opportunities: tax_opps,
            message: coord_result.reason,
        }
    }

    /// Set auto-rebalance enabled
    pub fn set_auto_rebalance(&self, enabled: bool) {
        self.auto_rebalance_enabled.store(enabled, Ordering::Relaxed);
    }

    /// Get comprehensive status
    pub fn get_status(&self) -> RebalancingStatus {
        let algo_stats = self.algorithm.get_stats();
        let actor_stats = self.coordinator.get_actor_stats();

        RebalancingStatus {
            auto_rebalance_enabled: self.auto_rebalance_enabled.load(Ordering::Relaxed),
            total_rebalances: algo_stats.total_rebalances,
            current_drift: algo_stats.current_drift,
            active_actors: actor_stats.iter().filter(|a| !a.is_halted).count(),
            halted_actors: actor_stats.iter().filter(|a| a.is_halted).count(),
        }
    }
}

impl GlobalRebalancingManager {
    fn algorithm(&self) -> &RebalancingAlgorithm {
        &self.coordinator.algorithm
    }
}

/// Auto-rebalance report
#[derive(Debug, Clone)]
pub struct AutoRebalanceReport {
    pub rebalance_triggered: bool,
    pub tax_opportunities: Vec<TaxHarvestOpportunity>,
    pub message: String,
}

/// Comprehensive rebalancing status
#[derive(Debug, Clone)]
pub struct RebalancingStatus {
    pub auto_rebalance_enabled: bool,
    pub total_rebalances: u64,
    pub current_drift: f64,
    pub active_actors: usize,
    pub halted_actors: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symbol_actor_creation() {
        let actor = SymbolActor::new(*b"BTC             ", 0.5);
        assert!(!actor.is_halted());
        assert_eq!(actor.message_count.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_actor_message_handling() {
        let actor = SymbolActor::new(*b"ETH             ", 0.3);
        
        let response = actor.handle_message(ActorMessage::UpdateTarget(0.4));
        assert!(matches!(response, ActorResponse::TargetUpdated(0.4)));
        
        actor.handle_message(ActorMessage::Halt);
        assert!(actor.is_halted());
        
        let response = actor.handle_message(ActorMessage::UpdateTarget(0.5));
        assert!(matches!(response, ActorResponse::Halted));
    }

    #[test]
    fn test_coordinator_creation() {
        let algo = RebalancingAlgorithm::new(2.0, 5.0, 0.001, 100.0);
        let coordinator = RebalancingCoordinator::new(algo);
        
        assert!(!coordinator.coordination_enabled.load(Ordering::Relaxed));
        coordinator.set_coordination_enabled(true);
        assert!(coordinator.coordination_enabled.load(Ordering::Relaxed));
    }

    #[test]
    fn test_global_manager_status() {
        let algo = RebalancingAlgorithm::new(2.0, 5.0, 0.001, 100.0);
        let manager = GlobalRebalancingManager::new(algo);
        
        let status = manager.get_status();
        assert!(status.auto_rebalance_enabled);
        assert_eq!(status.active_actors, 0);
    }
}
