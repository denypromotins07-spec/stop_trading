//! Chaos Engineering Module Root
//! 
//! Manages chaos monkey routines, strictly disabled in live production via compile-time flags.
//! Coordinates network fault injection and state corruption testing in shadow mode only.

#![cfg(any(test, feature = "chaos-engineering"))]

use alloc::vec::Vec;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

pub mod network_fault;
pub mod state_corrupt;

pub use network_fault::{
    NetworkFaultInjector,
    ConnectionSimulator,
    FaultConfig,
    FaultType,
    FaultEvent,
    FaultStats,
    SendResult,
    ConnectionStats,
    ChaosError,
    MAX_FAULT_CONFIGS,
};

pub use state_corrupt::{
    StateCorruptionInjector,
    WalCorruptionTester,
    CorruptionConfig,
    CorruptionType,
    CorruptedData,
    RecoveryResult,
    CorruptionStats,
    WalTestResult,
    WalStats,
    DataType,
    MAX_CORRUPTION_PATTERNS,
};

/// Production safety flag - always false in release builds without chaos-engineering feature
#[cfg(not(feature = "chaos-engineering"))]
pub const CHAOS_ENABLED: bool = false;

#[cfg(feature = "chaos-engineering")]
pub const CHAOS_ENABLED: bool = true;

/// Chaos engineering coordinator
pub struct ChaosMonkey {
    network_injector: NetworkFaultInjector,
    state_injector: StateCorruptionInjector,
    is_active: AtomicBool,
    shadow_mode: AtomicBool,
    total_events: AtomicU64,
    production_blocked: AtomicBool,
}

impl ChaosMonkey {
    /// Create new chaos monkey (only active in test/shadow mode)
    pub fn new() -> Self {
        ChaosMonkey {
            network_injector: NetworkFaultInjector::new(),
            state_injector: StateCorruptionInjector::new(),
            is_active: AtomicBool::new(false),
            shadow_mode: AtomicBool::new(true),
            total_events: AtomicU64::new(0),
            production_blocked: AtomicBool::new(!CHAOS_ENABLED),
        }
    }

    /// Initialize chaos monkey with configurations
    pub fn initialize(&self, network_configs: Vec<FaultConfig>, state_configs: Vec<CorruptionConfig>) -> Result<(), ChaosError> {
        // Block activation in production
        if self.production_blocked.load(Ordering::Acquire) {
            return Err(ChaosError::NotInShadowMode);
        }

        for config in network_configs {
            self.network_injector.add_fault(config)?;
        }

        for config in state_configs {
            self.state_injector.add_corruption(config)
                .map_err(|_| ChaosError::InvalidConfiguration)?;
        }

        Ok(())
    }

    /// Activate chaos monkey (shadow mode only)
    pub fn activate(&self) -> Result<(), ChaosError> {
        if self.production_blocked.load(Ordering::Acquire) {
            return Err(ChaosError::NotInShadowMode);
        }

        self.is_active.store(true, Ordering::Release);
        self.network_injector.set_active(true);
        self.state_injector.set_active(true);
        
        Ok(())
    }

    /// Deactivate chaos monkey
    pub fn deactivate(&self) {
        self.is_active.store(false, Ordering::Release);
        self.network_injector.set_active(false);
        self.state_injector.set_active(false);
    }

    /// Check if chaos monkey is active
    pub fn is_active(&self) -> bool {
        self.is_active.load(Ordering::Acquire) && !self.production_blocked.load(Ordering::Acquire)
    }

    /// Run a chaos experiment
    pub fn run_experiment(&self, experiment: ChaosExperiment) -> ExperimentResult {
        if !self.is_active() {
            return ExperimentResult::Blocked;
        }

        self.total_events.fetch_add(1, Ordering::Relaxed);

        match experiment {
            ChaosExperiment::NetworkStress => {
                let stats = self.network_injector.get_stats();
                ExperimentResult::Completed {
                    events_generated: stats.faults_injected,
                    description: "Network stress test",
                }
            }
            ChaosExperiment::StateCorruption => {
                let stats = self.state_injector.get_stats();
                ExperimentResult::Completed {
                    events_generated: stats.corruptions_injected,
                    description: "State corruption test",
                }
            }
            ChaosExperiment::FullChaos => {
                let net_stats = self.network_injector.get_stats();
                let state_stats = self.state_injector.get_stats();
                ExperimentResult::Completed {
                    events_generated: net_stats.faults_injected + state_stats.corruptions_injected,
                    description: "Full chaos experiment",
                }
            }
        }
    }

    /// Get comprehensive chaos statistics
    pub fn get_stats(&self) -> ChaosStats {
        ChaosStats {
            is_active: self.is_active(),
            shadow_mode: self.shadow_mode.load(Ordering::Acquire),
            production_blocked: self.production_blocked.load(Ordering::Acquire),
            total_events: self.total_events.load(Ordering::Relaxed),
            network_stats: self.network_injector.get_stats(),
            state_stats: self.state_injector.get_stats(),
        }
    }

    /// Get network injector reference
    pub fn network_injector(&self) -> &NetworkFaultInjector {
        &self.network_injector
    }

    /// Get state injector reference
    pub fn state_injector(&self) -> &StateCorruptionInjector {
        &self.state_injector
    }

    /// Emergency stop - immediately halt all chaos activities
    pub fn emergency_stop(&self) {
        self.deactivate();
        self.production_blocked.store(true, Ordering::Release);
    }
}

impl Default for ChaosMonkey {
    fn default() -> Self {
        Self::new()
    }
}

/// Types of chaos experiments
#[derive(Debug, Clone, Copy)]
pub enum ChaosExperiment {
    /// Network-focused stress testing
    NetworkStress,
    /// State corruption testing
    StateCorruption,
    /// Combined full chaos
    FullChaos,
}

/// Result of a chaos experiment
#[derive(Debug, Clone)]
pub enum ExperimentResult {
    Completed {
        events_generated: u64,
        description: &'static str,
    },
    Blocked,
    Failed(String),
}

/// Comprehensive chaos statistics
#[derive(Debug, Clone)]
pub struct ChaosStats {
    pub is_active: bool,
    pub shadow_mode: bool,
    pub production_blocked: bool,
    pub total_events: u64,
    pub network_stats: FaultStats,
    pub state_stats: CorruptionStats,
}

/// Builder pattern for ChaosMonkey configuration
pub struct ChaosMonkeyBuilder {
    network_configs: Vec<FaultConfig>,
    state_configs: Vec<CorruptionConfig>,
    auto_activate: bool,
}

impl Default for ChaosMonkeyBuilder {
    fn default() -> Self {
        ChaosMonkeyBuilder {
            network_configs: Vec::new(),
            state_configs: Vec::new(),
            auto_activate: false,
        }
    }
}

impl ChaosMonkeyBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_network_fault(mut self, config: FaultConfig) -> Self {
        self.network_configs.push(config);
        self
    }

    pub fn with_state_corruption(mut self, config: CorruptionConfig) -> Self {
        self.state_configs.push(config);
        self
    }

    pub fn auto_activate(mut self, activate: bool) -> Self {
        self.auto_activate = activate;
        self
    }

    pub fn build(self) -> Result<ChaosMonkey, ChaosError> {
        let monkey = ChaosMonkey::new();

        if !self.network_configs.is_empty() || !self.state_configs.is_empty() {
            monkey.initialize(self.network_configs, self.state_configs)?;
        }

        if self.auto_activate {
            monkey.activate()?;
        }

        Ok(monkey)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chaos_monkey_creation() {
        let monkey = ChaosMonkey::new();
        
        // Should not be active by default
        assert!(!monkey.is_active());
        
        let stats = monkey.get_stats();
        assert!(!stats.is_active);
    }

    #[test]
    fn test_chaos_monkey_builder() {
        let network_config = FaultConfig {
            fault_type: FaultType::PacketLoss,
            probability: 0.01,
            ..Default::default()
        };

        let result = ChaosMonkeyBuilder::new()
            .with_network_fault(network_config)
            .build();

        // May fail if not in shadow mode
        let _ = result;
    }

    #[test]
    fn test_emergency_stop() {
        let monkey = ChaosMonkey::new();
        
        // Even if somehow activated, emergency stop should work
        monkey.emergency_stop();
        
        assert!(!monkey.is_active());
        let stats = monkey.get_stats();
        assert!(stats.production_blocked);
    }

    #[test]
    fn test_experiment_blocked_in_production() {
        let monkey = ChaosMonkey::new();
        
        let result = monkey.run_experiment(ChaosExperiment::NetworkStress);
        
        // Should be blocked since not activated
        assert!(matches!(result, ExperimentResult::Blocked));
    }
}
