//! Final Orchestrator Module Root
//!
//! This module ties together the Rust HFT core and Python ML backend into a
//! single unified trading organism. It coordinates:
//! - Start sequence execution
//! - Python handoff management
//! - Global kill switch integration
//! - TUI event loop coordination

pub mod handoff;
pub mod start_sequence;

use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::Arc;
use std::time::Duration;

use crossbeam_channel::{bounded, Receiver, Sender};
use log::{debug, error, info, warn};
use parking_lot::RwLock;

pub use handoff::{KillSwitch, PythonBackendHandle, PythonBackendStatus, ShmPaths};
pub use start_sequence::{PhaseResult, StartPhase, StartSequence, StartSequenceReport};

/// System-wide operational mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum OperationalMode {
    /// System idle, not trading
    Idle = 0,
    /// Warming up systems
    WarmingUp = 1,
    /// Paper trading / simulation
    Shadow = 2,
    /// Live trading enabled
    Live = 3,
    /// Emergency shutdown
    Stopping = 4,
}

impl From<u8> for OperationalMode {
    fn from(value: u8) -> Self {
        match value {
            0 => Self::Idle,
            1 => Self::WarmingUp,
            2 => Self::Shadow,
            3 => Self::Live,
            4 => Self::Stopping,
            _ => Self::Idle,
        }
    }
}

/// Commands that can be sent to the orchestrator
#[derive(Debug, Clone)]
pub enum OrchestratorCommand {
    /// Start the system (triggers start sequence)
    Start,
    /// Stop gracefully
    Stop,
    /// Switch to shadow mode
    ShadowMode,
    /// Force kill all subsystems
    ForceKill,
    /// Dump current state
    DumpState,
    /// Set risk limit
    SetRiskLimit { symbol: String, max_position: f64 },
    /// Inject fault for testing
    InjectFault { fault_type: String },
}

/// Events emitted by the orchestrator
#[derive(Debug, Clone)]
pub enum OrchestratorEvent {
    /// Start sequence phase completed
    PhaseCompleted(PhaseResult),
    /// Mode changed
    ModeChanged(OperationalMode),
    /// Python backend status update
    PythonStatus(PythonBackendStatus),
    /// Kill switch triggered
    KillSwitchTriggered(String),
    /// System ready for trading
    ReadyForTrading,
    /// Error occurred
    Error(String),
}

/// Main orchestrator state
pub struct Orchestrator {
    mode: AtomicU8,
    is_running: AtomicBool,
    start_sequence: Arc<StartSequence>,
    python_backend: Arc<PythonBackendHandle>,
    kill_switch: Arc<KillSwitch>,
    command_tx: Sender<OrchestratorCommand>,
    command_rx: Receiver<OrchestratorCommand>,
    event_tx: Sender<OrchestratorEvent>,
    event_rx: Receiver<OrchestratorEvent>,
    config: RwLock<OrchestratorConfig>,
}

/// Orchestrator configuration
#[derive(Debug, Clone)]
pub struct OrchestratorConfig {
    pub enable_python_ml: bool,
    pub max_python_restarts: u32,
    pub python_ready_timeout_secs: u64,
    pub health_check_interval_ms: u64,
    pub graceful_shutdown_timeout_secs: u64,
}

impl Default for OrchestratorConfig {
    fn default() -> Self {
        Self {
            enable_python_ml: true,
            max_python_restarts: 3,
            python_ready_timeout_secs: 30,
            health_check_interval_ms: 1000,
            graceful_shutdown_timeout_secs: 10,
        }
    }
}

impl Orchestrator {
    /// Create a new orchestrator instance
    pub fn new(shm_paths: ShmPaths, config: OrchestratorConfig) -> Self {
        let (command_tx, command_rx) = bounded(100);
        let (event_tx, event_rx) = bounded(1000);
        
        let python_backend = Arc::new(PythonBackendHandle::new(shm_paths));
        let kill_switch = Arc::new(KillSwitch::new());
        
        Self {
            mode: AtomicU8::new(OperationalMode::Idle as u8),
            is_running: AtomicBool::new(false),
            start_sequence: Arc::new(StartSequence::new()),
            python_backend,
            kill_switch,
            command_tx,
            command_rx,
            event_tx,
            event_rx,
            config: RwLock::new(config),
        }
    }

    /// Get command sender for external use (TUI, CLI)
    pub fn command_sender(&self) -> Sender<OrchestratorCommand> {
        self.command_tx.clone()
    }

    /// Get event receiver for external use (TUI, logging)
    pub fn event_receiver(&self) -> Receiver<OrchestratorEvent> {
        self.event_rx.clone()
    }

    /// Get current operational mode
    pub fn current_mode(&self) -> OperationalMode {
        OperationalMode::from(self.mode.load(Ordering::Acquire))
    }

    /// Check if system is live trading
    pub fn is_live(&self) -> bool {
        self.current_mode() == OperationalMode::Live
    }

    /// Check if kill switch is triggered
    pub fn is_killed(&self) -> bool {
        self.kill_switch.is_triggered()
    }

    /// Get start sequence reference
    pub fn start_sequence(&self) -> Arc<StartSequence> {
        self.start_sequence.clone()
    }

    /// Get Python backend reference
    pub fn python_backend(&self) -> Arc<PythonBackendHandle> {
        self.python_backend.clone()
    }

    /// Get kill switch reference
    pub fn kill_switch(&self) -> Arc<KillSwitch> {
        self.kill_switch.clone()
    }

    /// Run the main orchestrator loop
    pub async fn run(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        info!("Orchestrator starting...");
        self.is_running.store(true, Ordering::Release);
        
        // Spawn crash monitor for Python backend
        let config = self.config.read().clone();
        let _monitor_handle = handoff::spawn_crash_monitor(
            self.python_backend.clone(),
            self.kill_switch.clone(),
            config.max_python_restarts,
        );
        
        // Spawn health check loop
        let health_handle = self.spawn_health_checks();
        
        // Main command processing loop
        while self.is_running.load(Ordering::Acquire) && !self.kill_switch.is_triggered() {
            match self.command_rx.recv_timeout(Duration::from_millis(100)) {
                Ok(cmd) => self.process_command(cmd).await?,
                Err(_) => {
                    // Timeout, continue loop
                }
            }
            
            // Check for kill switch
            if let Some(reason) = self.kill_switch.reason() {
                let _ = self.event_tx.send(OrchestratorEvent::KillSwitchTriggered(reason));
                break;
            }
        }
        
        // Graceful shutdown
        info!("Orchestrator shutting down...");
        self.graceful_shutdown().await;
        
        // Wait for background tasks
        drop(health_handle);
        
        Ok(())
    }

    /// Process incoming command
    async fn process_command(&self, cmd: OrchestratorCommand) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        debug!("Processing command: {:?}", cmd);
        
        match cmd {
            OrchestratorCommand::Start => {
                self.execute_start_sequence().await?;
            }
            OrchestratorCommand::Stop => {
                self.initiate_stop().await?;
            }
            OrchestratorCommand::ShadowMode => {
                self.mode.store(OperationalMode::Shadow as u8, Ordering::Release);
                let _ = self.event_tx.send(OrchestratorEvent::ModeChanged(OperationalMode::Shadow));
                info!("Switched to shadow mode");
            }
            OrchestratorCommand::ForceKill => {
                warn!("FORCE KILL commanded!");
                self.kill_switch.trigger("Force kill command received".to_string());
            }
            OrchestratorCommand::DumpState => {
                self.dump_state();
            }
            OrchestratorCommand::SetRiskLimit { symbol, max_position } => {
                info!("Setting risk limit for {}: {}", symbol, max_position);
                // Would update risk manager
            }
            OrchestratorCommand::InjectFault { fault_type } => {
                warn!("Injecting fault: {}", fault_type);
                // Would trigger fault injection framework
            }
        }
        
        Ok(())
    }

    /// Execute the full start sequence
    async fn execute_start_sequence(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        info!("Executing start sequence...");
        self.mode.store(OperationalMode::WarmingUp as u8, Ordering::Release);
        
        // Phase 1: Warmup
        let warmup_result = self.start_sequence.execute_warmup().await;
        let _ = self.event_tx.send(OrchestratorEvent::PhaseCompleted(warmup_result.clone()));
        if !warmup_result.success {
            self.mode.store(OperationalMode::Idle as u8, Ordering::Release);
            return Err(format!("Warmup failed: {}", warmup_result.message).into());
        }
        
        // Phase 2: IPC Init
        let ipc_result = self.start_sequence.execute_ipc_init().await;
        let _ = self.event_tx.send(OrchestratorEvent::PhaseCompleted(ipc_result.clone()));
        if !ipc_result.success {
            self.mode.store(OperationalMode::Idle as u8, Ordering::Release);
            return Err(format!("IPC init failed: {}", ipc_result.message).into());
        }
        
        // Phase 3: Python Handoff (if enabled)
        let config = self.config.read().clone();
        if config.enable_python_ml {
            let handoff_result = self.start_sequence
                .execute_python_handoff(self.python_backend.clone())
                .await;
            let _ = self.event_tx.send(OrchestratorEvent::PhaseCompleted(handoff_result.clone()));
            if !handoff_result.success {
                self.mode.store(OperationalMode::Idle as u8, Ordering::Release);
                return Err(format!("Python handoff failed: {}", handoff_result.message).into());
            }
        }
        
        // Phase 4: Pre-live checks
        let prelive_result = self.start_sequence.execute_pre_live_check().await;
        let _ = self.event_tx.send(OrchestratorEvent::PhaseCompleted(prelive_result.clone()));
        if !prelive_result.success {
            self.mode.store(OperationalMode::Idle as u8, Ordering::Release);
            return Err(format!("Pre-live checks failed: {}", prelive_result.message).into());
        }
        
        // Success!
        self.mode.store(OperationalMode::Live as u8, Ordering::SeqCst);
        let _ = self.event_tx.send(OrchestratorEvent::ModeChanged(OperationalMode::Live));
        let _ = self.event_tx.send(OrchestratorEvent::ReadyForTrading);
        
        info!("Start sequence completed successfully - LIVE TRADING ENABLED");
        Ok(())
    }

    /// Initiate graceful stop
    async fn initiate_stop(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        info!("Initiating graceful stop...");
        self.mode.store(OperationalMode::Stopping as u8, Ordering::Release);
        let _ = self.event_tx.send(OrchestratorEvent::ModeChanged(OperationalMode::Stopping));
        Ok(())
    }

    /// Perform graceful shutdown
    async fn graceful_shutdown(&self) {
        let config = self.config.read().clone();
        
        // Stop Python backend
        if config.enable_python_ml {
            info!("Stopping Python backend...");
            self.python_backend.stop(config.graceful_shutdown_timeout_secs);
        }
        
        // Cancel outstanding orders (would be implemented)
        // Close connections (would be implemented)
        // Flush logs (would be implemented)
        
        self.is_running.store(false, Ordering::Release);
        info!("Graceful shutdown complete");
    }

    /// Dump current system state
    fn dump_state(&self) {
        let report = self.start_sequence.get_report();
        info!("=== SYSTEM STATE DUMP ===");
        info!("Mode: {:?}", self.current_mode());
        info!("Running: {}", self.is_running.load(Ordering::Acquire));
        info!("Kill Switch: {}", if self.is_killed() { "TRIGGERED" } else { "OK" });
        info!("Start Sequence Phase: {:?}", report.final_phase);
        info!("Python Backend Status: {:?}", self.python_backend.status());
        info!("Python Crash Count: {}", self.python_backend.crash_count());
        
        if let Some(reason) = self.kill_switch.reason() {
            info!("Kill Switch Reason: {}", reason);
        }
        
        info!("=== END STATE DUMP ===");
    }

    /// Spawn health check background task
    fn spawn_health_checks(&self) -> std::thread::JoinHandle<()> {
        let python_backend = self.python_backend.clone();
        let event_tx = self.event_tx.clone();
        let is_running = self.is_running.clone();
        let config = self.config.read().clone();
        
        std::thread::spawn(move || {
            while is_running.load(Ordering::Acquire) {
                std::thread::sleep(Duration::from_millis(config.health_check_interval_ms));
                
                // Check Python backend health
                if !python_backend.check_health(5000) {
                    let status = python_backend.status();
                    let _ = event_tx.send(OrchestratorEvent::PythonStatus(status));
                    
                    if matches!(status, PythonBackendStatus::Crashed) {
                        warn!("Python backend health check failed!");
                    }
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orchestrator_creation() {
        let shm_paths = ShmPaths {
            feature_vector_path: "/test/feature".to_string(),
            signal_batch_path: "/test/signal".to_string(),
            state_path: "/test/state".to_string(),
        };
        
        let orchestrator = Orchestrator::new(shm_paths, OrchestratorConfig::default());
        assert_eq!(orchestrator.current_mode(), OperationalMode::Idle);
        assert!(!orchestrator.is_live());
        assert!(!orchestrator.is_killed());
    }

    #[test]
    fn test_operational_mode_transitions() {
        let mode = OperationalMode::Idle;
        assert_eq!(OperationalMode::from(mode as u8), mode);
        
        let mode = OperationalMode::Live;
        assert_eq!(OperationalMode::from(mode as u8), mode);
    }
}
