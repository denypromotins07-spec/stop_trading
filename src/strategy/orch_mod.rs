//! Strategy Orchestration Module Root
//! 
//! Manages strategy state machines and dynamic DAG reconfiguration.

pub mod dag;
pub mod orchestrator;

pub use dag::{
    AggregateType, ComputeResult, DagNode, DagStats, Direction, ExecutionContext, MathOperation,
    NodeData, NodeId, NodeType, SignalType, StatOperation, StrategyDag,
};
pub use orchestrator::{
    Orchestrator, OrchestratorBuilder, OrchestratorStats, Task,
};

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::time::{Duration, Instant};
use parking_lot::RwLock;
use tracing::{debug, error, info, warn};

/// Maximum number of concurrent strategies
const MAX_STRATEGIES: usize = 64;

/// Unique identifier for strategies
pub type StrategyId = u64;

/// Strategy execution state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StrategyState {
    /// Strategy is being initialized
    Initializing,
    /// Strategy is running normally
    Running,
    /// Strategy is paused (e.g., waiting for conditions)
    Paused,
    /// Strategy is in cooldown after a trade
    Cooldown,
    /// Strategy encountered an error
    Error,
    /// Strategy is being shut down
    Stopping,
    /// Strategy has stopped
    Stopped,
}

/// Configuration for a strategy instance
#[derive(Clone, Debug)]
pub struct StrategyConfig {
    pub name: String,
    pub enabled: bool,
    pub max_position_size: f64,
    pub risk_limit: f64,
    pub cooldown_ms: u64,
    pub parameters: Vec<(String, f64)>,
}

impl Default for StrategyConfig {
    fn default() -> Self {
        Self {
            name: String::new(),
            enabled: true,
            max_position_size: 1.0,
            risk_limit: 0.02,
            cooldown_ms: 1000,
            parameters: Vec::new(),
        }
    }
}

/// Runtime state for a strategy
pub struct StrategyInstance {
    pub id: StrategyId,
    pub config: StrategyConfig,
    pub state: StrategyState,
    pub dag: Arc<StrategyDag>,
    pub orchestrator: Option<Arc<Orchestrator>>,
    pub last_signal_time: Option<Instant>,
    pub signals_generated: u64,
    pub trades_executed: u64,
    pub pnl: f64,
    pub created_at: Instant,
    pub updated_at: Instant,
}

impl StrategyInstance {
    pub fn new(id: StrategyId, config: StrategyConfig) -> Self {
        let dag = Arc::new(StrategyDag::new());
        
        Self {
            id,
            config,
            state: StrategyState::Initializing,
            dag,
            orchestrator: None,
            last_signal_time: None,
            signals_generated: 0,
            trades_executed: 0,
            pnl: 0.0,
            created_at: Instant::now(),
            updated_at: Instant::now(),
        }
    }

    /// Initialize the orchestrator for this strategy
    pub fn initialize(&mut self, num_workers: usize) {
        let orchestrator = OrchestratorBuilder::new()
            .workers(num_workers)
            .build(self.dag.clone());
        
        self.orchestrator = Some(Arc::new(orchestrator));
        self.state = StrategyState::Running;
        self.updated_at = Instant::now();
        
        info!("Strategy {} initialized with {} workers", self.config.name, num_workers);
    }

    /// Start strategy execution
    pub fn start(&self) -> bool {
        if let Some(ref orch) = self.orchestrator {
            orch.start();
            true
        } else {
            false
        }
    }

    /// Pause strategy execution
    pub fn pause(&mut self) {
        self.state = StrategyState::Paused;
        self.updated_at = Instant::now();
        info!("Strategy {} paused", self.config.name);
    }

    /// Resume strategy execution
    pub fn resume(&mut self) {
        if self.state == StrategyState::Paused {
            self.state = StrategyState::Running;
            self.updated_at = Instant::now();
            info!("Strategy {} resumed", self.config.name);
        }
    }

    /// Stop strategy execution
    pub fn stop(&mut self) {
        self.state = StrategyState::Stopping;
        
        if let Some(ref orch) = self.orchestrator {
            orch.shutdown();
        }
        
        self.state = StrategyState::Stopped;
        self.updated_at = Instant::now();
        info!("Strategy {} stopped", self.config.name);
    }

    /// Record a signal generation
    pub fn record_signal(&mut self) {
        self.signals_generated += 1;
        self.last_signal_time = Some(Instant::now());
        self.updated_at = Instant::now();
    }

    /// Record a trade execution
    pub fn record_trade(&mut self, pnl: f64) {
        self.trades_executed += 1;
        self.pnl += pnl;
        self.updated_at = Instant::now();
        
        // Enter cooldown
        if self.config.cooldown_ms > 0 {
            self.state = StrategyState::Cooldown;
        }
    }

    /// Check if cooldown has expired
    pub fn check_cooldown(&mut self) {
        if self.state == StrategyState::Cooldown {
            if let Some(last) = self.last_signal_time {
                if last.elapsed().as_millis() as u64 >= self.config.cooldown_ms {
                    self.state = StrategyState::Running;
                }
            }
        }
    }

    /// Get strategy statistics
    pub fn stats(&self) -> StrategyStats {
        StrategyStats {
            id: self.id,
            name: self.config.name.clone(),
            state: self.state,
            signals_generated: self.signals_generated,
            trades_executed: self.trades_executed,
            pnl: self.pnl,
            uptime: self.created_at.elapsed(),
            dag_nodes: self.dag.node_count(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct StrategyStats {
    pub id: StrategyId,
    pub name: String,
    pub state: StrategyState,
    pub signals_generated: u64,
    pub trades_executed: u64,
    pub pnl: f64,
    pub uptime: Duration,
    pub dag_nodes: usize,
}

/// Manager for multiple strategy instances
pub struct StrategyManager {
    strategies: RwLock<Vec<Arc<RwLock<StrategyInstance>>>>,
    next_id: AtomicU64,
    max_strategies: usize,
    enabled_count: AtomicU64,
}

impl StrategyManager {
    /// Create a new strategy manager
    pub fn new(max_strategies: usize) -> Self {
        Self {
            strategies: RwLock::new(Vec::with_capacity(max_strategies.min(MAX_STRATEGIES))),
            next_id: AtomicU64::new(1),
            max_strategies: max_strategies.min(MAX_STRATEGIES),
            enabled_count: AtomicU64::new(0),
        }
    }

    /// Create and register a new strategy
    pub fn create_strategy(&self, config: StrategyConfig) -> Option<StrategyId> {
        let mut strategies = self.strategies.write();
        
        if strategies.len() >= self.max_strategies {
            warn!("Maximum strategy count reached");
            return None;
        }

        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let instance = Arc::new(RwLock::new(StrategyInstance::new(id, config)));
        
        strategies.push(instance);
        self.enabled_count.fetch_add(1, Ordering::Relaxed);
        
        debug!("Created strategy with ID {}", id);
        Some(id)
    }

    /// Get a strategy by ID
    pub fn get_strategy(&self, id: StrategyId) -> Option<Arc<RwLock<StrategyInstance>>> {
        let strategies = self.strategies.read();
        strategies.iter()
            .find(|s| s.read().id == id)
            .cloned()
    }

    /// Remove a strategy by ID
    pub fn remove_strategy(&self, id: StrategyId) -> bool {
        let mut strategies = self.strategies.write();
        let initial_len = strategies.len();
        
        strategies.retain(|s| s.read().id != id);
        
        if strategies.len() < initial_len {
            self.enabled_count.fetch_sub(1, Ordering::Relaxed);
            debug!("Removed strategy {}", id);
            true
        } else {
            false
        }
    }

    /// Enable a strategy
    pub fn enable_strategy(&self, id: StrategyId) -> bool {
        if let Some(strategy) = self.get_strategy(id) {
            let mut s = strategy.write();
            s.config.enabled = true;
            s.updated_at = Instant::now();
            true
        } else {
            false
        }
    }

    /// Disable a strategy
    pub fn disable_strategy(&self, id: StrategyId) -> bool {
        if let Some(strategy) = self.get_strategy(id) {
            let mut s = strategy.write();
            s.config.enabled = false;
            s.pause();
            true
        } else {
            false
        }
    }

    /// Start all enabled strategies
    pub fn start_all(&self) {
        let strategies = self.strategies.read();
        for strategy in strategies.iter() {
            let s = strategy.read();
            if s.config.enabled && s.state == StrategyState::Initializing {
                drop(s);
                let mut s = strategy.write();
                s.initialize(4); // Default worker count
                s.start();
            }
        }
    }

    /// Stop all strategies
    pub fn stop_all(&self) {
        let strategies = self.strategies.read();
        for strategy in strategies.iter() {
            strategy.write().stop();
        }
    }

    /// Get all strategy statistics
    pub fn all_stats(&self) -> Vec<StrategyStats> {
        let strategies = self.strategies.read();
        strategies.iter()
            .map(|s| s.read().stats())
            .collect()
    }

    /// Get count of active strategies
    pub fn active_count(&self) -> usize {
        let strategies = self.strategies.read();
        strategies.iter()
            .filter(|s| s.read().state == StrategyState::Running)
            .count()
    }

    /// Get total enabled count
    pub fn enabled_count(&self) -> u64 {
        self.enabled_count.load(Ordering::Relaxed)
    }

    /// Update cooldown states for all strategies
    pub fn update_cooldowns(&self) {
        let strategies = self.strategies.read();
        for strategy in strategies.iter() {
            strategy.write().check_cooldown();
        }
    }
}

impl Default for StrategyManager {
    fn default() -> Self {
        Self::new(MAX_STRATEGIES)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strategy_manager_creation() {
        let manager = StrategyManager::new(10);
        assert_eq!(manager.active_count(), 0);
        assert_eq!(manager.enabled_count(), 0);
    }

    #[test]
    fn test_create_strategy() {
        let manager = StrategyManager::new(10);
        
        let config = StrategyConfig {
            name: "TestStrategy".to_string(),
            ..Default::default()
        };
        
        let id = manager.create_strategy(config);
        assert!(id.is_some());
        assert_eq!(manager.enabled_count(), 1);
        
        let strategy = manager.get_strategy(id.unwrap());
        assert!(strategy.is_some());
    }

    #[test]
    fn test_strategy_lifecycle() {
        let manager = StrategyManager::new(10);
        
        let config = StrategyConfig {
            name: "LifecycleTest".to_string(),
            ..Default::default()
        };
        
        let id = manager.create_strategy(config).unwrap();
        let strategy = manager.get_strategy(id).unwrap();
        
        // Initial state
        assert_eq!(strategy.read().state, StrategyState::Initializing);
        
        // Initialize and start
        {
            let mut s = strategy.write();
            s.initialize(2);
        }
        strategy.read().start();
        
        assert_eq!(strategy.read().state, StrategyState::Running);
        
        // Pause
        strategy.write().pause();
        assert_eq!(strategy.read().state, StrategyState::Paused);
        
        // Resume
        strategy.write().resume();
        assert_eq!(strategy.read().state, StrategyState::Running);
        
        // Stop
        strategy.write().stop();
        assert_eq!(strategy.read().state, StrategyState::Stopped);
    }

    #[test]
    fn test_remove_strategy() {
        let manager = StrategyManager::new(10);
        
        let config = StrategyConfig::default();
        let id = manager.create_strategy(config).unwrap();
        
        assert!(manager.remove_strategy(id));
        assert!(manager.get_strategy(id).is_none());
        assert_eq!(manager.enabled_count(), 0);
    }
}
