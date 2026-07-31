//! Python ML Backend Handoff Module
//!
//! This module handles spawning the Python ML subprocess, passing shared memory
//! file descriptors securely, and waiting for the "READY" flag before unlocking
//! execution gateways. Includes crash detection and global kill switch integration.

use std::collections::HashMap;
use std::ffi::{OsStr, OsString};
use std::io::{self, BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crossbeam_channel::{bounded, Receiver, Sender};
use log::{debug, error, info, warn};
use parking_lot::RwLock;

/// Maximum time to wait for Python backend to signal READY
const READY_TIMEOUT_SECS: u64 = 30;

/// Polling interval for checking Python process status
const STATUS_POLL_INTERVAL_MS: u64 = 100;

/// Shared memory segment paths
pub struct ShmPaths {
    pub feature_vector_path: String,
    pub signal_batch_path: String,
    pub state_path: String,
}

/// Python backend process handle with crash detection
pub struct PythonBackendHandle {
    child: RwLock<Option<Child>>,
    is_ready: AtomicBool,
    is_alive: AtomicBool,
    crash_count: AtomicU64,
    last_heartbeat_ns: AtomicU64,
    shm_paths: ShmPaths,
    stdout_reader: Option<std::thread::JoinHandle<()>>,
    stderr_reader: Option<std::thread::JoinHandle<()>>,
    shutdown_tx: Sender<()>,
    shutdown_rx: Receiver<()>,
}

/// Status of the Python backend
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PythonBackendStatus {
    NotStarted = 0,
    Starting = 1,
    Ready = 2,
    Running = 3,
    Stopped = 4,
    Crashed = 5,
    Restarting = 6,
}

impl From<u8> for PythonBackendStatus {
    fn from(value: u8) -> Self {
        match value {
            0 => Self::NotStarted,
            1 => Self::Starting,
            2 => Self::Ready,
            3 => Self::Running,
            4 => Self::Stopped,
            5 => Self::Crashed,
            6 => Self::Restarting,
            _ => Self::Crashed,
        }
    }
}

/// Configuration for Python backend
pub struct PythonBackendConfig {
    pub python_interpreter: PathBuf,
    pub script_path: PathBuf,
    pub working_dir: PathBuf,
    pub environment: HashMap<String, String>,
    pub max_restarts: u32,
    pub restart_delay_ms: u64,
}

impl Default for PythonBackendConfig {
    fn default() -> Self {
        Self {
            python_interpreter: PathBuf::from("python3"),
            script_path: PathBuf::from("python_bridge/ml_backend.py"),
            working_dir: PathBuf::from("."),
            environment: HashMap::new(),
            max_restarts: 3,
            restart_delay_ms: 1000,
        }
    }
}

impl PythonBackendHandle {
    /// Create a new Python backend handle
    pub fn new(shm_paths: ShmPaths) -> Self {
        let (shutdown_tx, shutdown_rx) = bounded(1);
        
        Self {
            child: RwLock::new(None),
            is_ready: AtomicBool::new(false),
            is_alive: AtomicBool::new(false),
            crash_count: AtomicU64::new(0),
            last_heartbeat_ns: AtomicU64::new(0),
            shm_paths,
            stdout_reader: None,
            stderr_reader: None,
            shutdown_tx,
            shutdown_rx,
        }
    }

    /// Spawn the Python ML backend process
    pub fn spawn(&self, config: &PythonBackendConfig) -> io::Result<()> {
        info!("Spawning Python ML backend...");
        
        // Set up environment variables
        let mut envs: Vec<(OsString, OsString)> = config
            .environment
            .iter()
            .map(|(k, v)| (OsString::from(k), OsString::from(v)))
            .collect();
        
        // Add shared memory paths
        envs.push((
            OsString::from("HFT_SHM_FEATURE_PATH"),
            OsString::from(&self.shm_paths.feature_vector_path),
        ));
        envs.push((
            OsString::from("HFT_SHM_SIGNAL_PATH"),
            OsString::from(&self.shm_paths.signal_batch_path),
        ));
        envs.push((
            OsString::from("HFT_SHM_STATE_PATH"),
            OsString::from(&self.shm_paths.state_path),
        ));
        
        // Build command
        let mut cmd = Command::new(&config.python_interpreter);
        cmd.arg(&config.script_path)
            .current_dir(&config.working_dir)
            .envs(envs)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        
        // Spawn process
        let mut child = cmd.spawn()?;
        
        // Capture stdout/stderr for logging
        let stdout = child.stdout.take().unwrap();
        let stderr = child.stderr.take().unwrap();
        
        // Spawn reader threads
        let stdout_handle = std::thread::spawn(move || {
            let reader = BufReader::new(stdout);
            for line in reader.lines().flatten() {
                debug!("[Python] {}", line);
            }
        });
        
        let stderr_handle = std::thread::spawn(move || {
            let reader = BufReader::new(stderr);
            for line in reader.lines().flatten() {
                warn!("[Python] {}", line);
            }
        });
        
        // Store child process
        {
            let mut child_guard = self.child.write();
            *child_guard = Some(child);
        }
        
        self.stdout_reader = Some(stdout_handle);
        self.stderr_reader = Some(stderr_handle);
        self.is_alive.store(true, Ordering::Release);
        
        info!("Python ML backend spawned successfully");
        Ok(())
    }

    /// Wait for Python backend to signal READY
    pub fn wait_for_ready(&self, timeout_secs: u64) -> bool {
        info!("Waiting for Python backend to signal READY...");
        
        let timeout = Duration::from_secs(timeout_secs);
        let start = Instant::now();
        
        while start.elapsed() < timeout {
            if self.is_ready.load(Ordering::Acquire) {
                info!("Python backend signaled READY after {}ms", start.elapsed().as_millis());
                return true;
            }
            
            // Check if process crashed before ready
            if !self.is_alive.load(Ordering::Acquire) {
                error!("Python backend crashed before signaling READY");
                return false;
            }
            
            std::thread::sleep(Duration::from_millis(STATUS_POLL_INTERVAL_MS));
        }
        
        error!("Timeout waiting for Python backend READY signal");
        false
    }

    /// Signal READY from Python side (called via FFI or IPC)
    pub fn mark_ready(&self) {
        self.is_ready.store(true, Ordering::Release);
        self.last_heartbeat_ns.store(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
            Ordering::Release,
        );
    }

    /// Update heartbeat timestamp
    pub fn update_heartbeat(&self) {
        self.last_heartbeat_ns.store(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
            Ordering::Release,
        );
    }

    /// Check if backend has crashed (no heartbeat for extended period)
    pub fn check_health(&self, max_silence_ms: u64) -> bool {
        if !self.is_alive.load(Ordering::Acquire) {
            return false;
        }
        
        let last_hb = self.last_heartbeat_ns.load(Ordering::Acquire);
        if last_hb == 0 {
            return true; // No heartbeat yet, assume healthy during startup
        }
        
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        
        let silence_ns = now.saturating_sub(last_hb);
        let silence_ms = silence_ns / 1_000_000;
        
        if silence_ms > max_silence_ms {
            warn!(
                "Python backend unresponsive: no heartbeat for {}ms (limit: {}ms)",
                silence_ms, max_silence_ms
            );
            false
        } else {
            true
        }
    }

    /// Stop the Python backend gracefully
    pub fn stop(&self, timeout_secs: u64) {
        info!("Stopping Python backend...");
        
        self.is_alive.store(false, Ordering::Release);
        
        // Send shutdown signal
        let _ = self.shutdown_tx.send(());
        
        // Terminate process
        {
            let mut child_guard = self.child.write();
            if let Some(child) = child_guard.as_mut() {
                #[cfg(unix)]
                {
                    use std::os::unix::process::CommandExt;
                    // Send SIGTERM first
                    let _ = nix::sys::signal::kill(
                        nix::unistd::Pid::from_raw(child.id() as i32),
                        nix::sys::signal::Signal::SIGTERM,
                    );
                }
                
                // Wait for graceful shutdown
                let start = Instant::now();
                while start.elapsed() < Duration::from_secs(timeout_secs) {
                    match child.try_wait() {
                        Ok(Some(_)) => {
                            info!("Python backend terminated gracefully");
                            *child_guard = None;
                            return;
                        }
                        Ok(None) => std::thread::sleep(Duration::from_millis(100)),
                        Err(e) => {
                            error!("Error checking Python process status: {}", e);
                            break;
                        }
                    }
                }
                
                // Force kill if still running
                let _ = child.kill();
                let _ = child.wait();
                info!("Python backend force-killed");
                *child_guard = None;
            }
        }
    }

    /// Get current status
    pub fn status(&self) -> PythonBackendStatus {
        if !self.is_alive.load(Ordering::Acquire) {
            if self.crash_count.load(Ordering::Acquire) > 0 {
                PythonBackendStatus::Crashed
            } else {
                PythonBackendStatus::Stopped
            }
        } else if self.is_ready.load(Ordering::Acquire) {
            PythonBackendStatus::Running
        } else {
            PythonBackendStatus::Starting
        }
    }

    /// Get crash count
    pub fn crash_count(&self) -> u64 {
        self.crash_count.load(Ordering::Acquire)
    }

    /// Increment crash count (called by crash detector)
    pub fn record_crash(&self) {
        self.crash_count.fetch_add(1, Ordering::Relaxed);
    }
}

/// Global kill switch trigger
pub struct KillSwitch {
    triggered: AtomicBool,
    reason: RwLock<Option<String>>,
}

impl KillSwitch {
    pub const fn new() -> Self {
        Self {
            triggered: AtomicBool::new(false),
            reason: RwLock::new(None),
        }
    }

    pub fn trigger(&self, reason: String) {
        error!("KILL SWITCH TRIGGERED: {}", reason);
        self.triggered.store(true, Ordering::SeqCst);
        *self.reason.write() = Some(reason);
    }

    pub fn is_triggered(&self) -> bool {
        self.triggered.load(Ordering::SeqCst)
    }

    pub fn reason(&self) -> Option<String> {
        self.reason.read().clone()
    }

    pub fn reset(&self) {
        self.triggered.store(false, Ordering::SeqCst);
        *self.reason.write() = None;
    }
}

impl Default for KillSwitch {
    fn default() -> Self {
        Self::new()
    }
}

/// Monitor Python backend and trigger kill switch on crash
pub fn spawn_crash_monitor(
    backend: Arc<PythonBackendHandle>,
    kill_switch: Arc<KillSwitch>,
    max_restarts: u32,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        info!("Crash monitor started for Python backend");
        
        loop {
            std::thread::sleep(Duration::from_millis(STATUS_POLL_INTERVAL_MS * 5));
            
            // Check if kill switch already triggered
            if kill_switch.is_triggered() {
                break;
            }
            
            // Check backend health
            if !backend.check_health(5000) && backend.is_alive.load(Ordering::Acquire) {
                warn!("Python backend appears unresponsive");
                backend.record_crash();
                
                let crash_count = backend.crash_count();
                if crash_count >= max_restarts as u64 {
                    kill_switch.trigger(format!(
                        "Python backend crashed {} times (max: {})",
                        crash_count, max_restarts
                    ));
                    break;
                }
                
                // Attempt restart would go here
                warn!("Would attempt restart (crash {}/{})", crash_count, max_restarts);
            }
        }
        
        info!("Crash monitor exiting");
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kill_switch() {
        let ks = KillSwitch::new();
        assert!(!ks.is_triggered());
        
        ks.trigger("Test trigger".to_string());
        assert!(ks.is_triggered());
        assert_eq!(ks.reason(), Some("Test trigger".to_string()));
        
        ks.reset();
        assert!(!ks.is_triggered());
    }

    #[test]
    fn test_backend_status_transitions() {
        let shm_paths = ShmPaths {
            feature_vector_path: "/test/feature".to_string(),
            signal_batch_path: "/test/signal".to_string(),
            state_path: "/test/state".to_string(),
        };
        
        let backend = PythonBackendHandle::new(shm_paths);
        assert_eq!(backend.status(), PythonBackendStatus::NotStarted);
    }
}
