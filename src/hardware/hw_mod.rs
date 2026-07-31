//! Hardware Module Root
//!
//! Integrates thermal and topology data into the global observability stack.

pub mod topology;
pub mod diagnostic;

pub use topology::{
    SystemTopology,
    CoreInfo,
    NumaNode,
    ThreadAffinityManager,
    ThreadPriority,
};

pub use diagnostic::{
    HardwareDiagnostics,
    ThermalState,
    PowerState,
    HardwareHealth,
    ThermalSeverity,
    MitigationAction,
    DiagnosticEvent,
};

/// Combined hardware monitoring engine
pub struct HardwareMonitor {
    pub topology: SystemTopology,
    pub diagnostics: diagnostic::HardwareDiagnostics,
}

impl HardwareMonitor {
    pub fn new(buffer_size: usize) -> Self {
        let topology = SystemTopology::detect();
        let diagnostics = diagnostic::HardwareDiagnostics::new(buffer_size);

        Self {
            topology,
            diagnostics,
        }
    }

    /// Get optimal core assignment for a thread type
    pub fn assign_optimal_core(&self, thread_name: &str, priority: ThreadPriority) -> Option<u32> {
        let manager = ThreadAffinityManager::new(self.topology.clone());
        manager.assign_thread(thread_name, priority)
    }

    /// Check if system is healthy for trading
    pub fn is_trading_safe(&self) -> bool {
        // Check topology suitability
        if !self.topology.is_hft_suitable() {
            return false;
        }

        // Check current thermal state
        if let Some(thermal) = self.diagnostics.get_last_thermal() {
            if !thermal.is_safe() {
                return false;
            }
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_monitor_creation() {
        let monitor = HardwareMonitor::new(1000);

        assert!(!monitor.topology.cpu_model.is_empty());
    }
}
