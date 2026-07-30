//! Daemon Process Management
//! 
//! Implements robust daemonization logic handling PID file management
//! and standard stream redirection for 24/7 background operation.

use std::fs::{self, File, OpenOptions};
use std::io::{self, Write, Read};
use std::path::{Path, PathBuf};
use std::process;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use crossbeam_utils::CachePadded;

/// Daemon state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DaemonState {
    /// Not yet daemonized
    Starting,
    /// Running as daemon
    Running,
    /// Shutting down
    Stopping,
    /// Stopped
    Stopped,
}

/// Daemon configuration
pub struct DaemonConfig {
    /// PID file path
    pub pid_file: PathBuf,
    /// Log file path
    pub log_file: Option<PathBuf>,
    /// Working directory
    pub work_dir: PathBuf,
    /// User to run as (if different)
    pub run_as_user: Option<String>,
    /// Group to run as (if different)
    pub run_as_group: Option<String>,
    /// Umask
    pub umask: u32,
}

impl Default for DaemonConfig {
    fn default() -> Self {
        Self {
            pid_file: PathBuf::from("/var/run/hft_bot.pid"),
            log_file: Some(PathBuf::from("/var/log/hft_bot.log")),
            work_dir: PathBuf::from("/"),
            run_as_user: None,
            run_as_group: None,
            umask: 0o022,
        }
    }
}

/// Daemon manager
pub struct DaemonManager {
    /// Current state
    state: CachePadded<AtomicU64>,
    /// PID file handle
    pid_file: CachePadded<std::sync::Mutex<Option<File>>>,
    /// Log file handle
    log_file: CachePadded<std::sync::Mutex<Option<File>>>,
    /// Process start time
    start_time_ns: CachePadded<AtomicU64>,
    /// Configuration
    config: DaemonConfig,
}

impl DaemonManager {
    /// Create a new daemon manager
    pub fn new(config: DaemonConfig) -> Self {
        Self {
            state: CachePadded::new(AtomicU64::new(DaemonState::Starting as u64)),
            pid_file: CachePadded::new(std::sync::Mutex::new(None)),
            log_file: CachePadded::new(std::sync::Mutex::new(None)),
            start_time_ns: CachePadded::new(AtomicU64::new(0)),
            config,
        }
    }

    /// Check if another instance is running
    pub fn check_existing(&self) -> Result<bool, String> {
        if self.config.pid_file.exists() {
            if let Ok(content) = fs::read_to_string(&self.config.pid_file) {
                if let Ok(pid) = content.trim().parse::<u32>() {
                    // Check if process exists
                    #[cfg(unix)]
                    {
                        use libc;
                        unsafe {
                            if libc::kill(pid as i32, 0) == 0 {
                                return Ok(true); // Process exists
                            }
                        }
                    }
                    
                    #[cfg(not(unix))]
                    {
                        // On non-Unix, assume it exists if PID file exists
                        return Ok(true);
                    }
                }
            }
        }
        Ok(false)
    }

    /// Write PID file
    fn write_pid_file(&self) -> Result<(), String> {
        // Ensure parent directory exists
        if let Some(parent) = self.config.pid_file.parent() {
            if !parent.exists() {
                fs::create_dir_all(parent).map_err(|e| format!("Failed to create PID dir: {}", e))?;
            }
        }

        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&self.config.pid_file)
            .map_err(|e| format!("Failed to open PID file: {}", e))?;

        writeln!(file, "{}", process::id())
            .map_err(|e| format!("Failed to write PID: {}", e))?;

        *self.pid_file.lock().unwrap() = Some(file);
        Ok(())
    }

    /// Remove PID file
    fn remove_pid_file(&self) {
        drop(self.pid_file.lock().unwrap().take());
        let _ = fs::remove_file(&self.config.pid_file);
    }

    /// Setup log file
    fn setup_log_file(&self) -> Result<(), String> {
        if let Some(ref log_path) = self.config.log_file {
            if let Some(parent) = log_path.parent() {
                if !parent.exists() {
                    fs::create_dir_all(parent).map_err(|e| format!("Failed to create log dir: {}", e))?;
                }
            }

            let file = OpenOptions::new()
                .write(true)
                .create(true)
                .append(true)
                .open(log_path)
                .map_err(|e| format!("Failed to open log file: {}", e))?;

            *self.log_file.lock().unwrap() = Some(file);
        }
        Ok(())
    }

    /// Redirect standard streams
    fn redirect_streams(&self) -> Result<(), String> {
        #[cfg(unix)]
        {
            use std::os::unix::io::AsRawFd;
            
            // Open /dev/null for stdin
            let null = OpenOptions::new()
                .read(true)
                .open("/dev/null")
                .map_err(|e| format!("Failed to open /dev/null: {}", e))?;

            unsafe {
                libc::dup2(null.as_raw_fd(), 0);
            }

            // Redirect stdout/stderr to log file if configured
            if let Some(ref mut log_file) = *self.log_file.lock().unwrap() {
                unsafe {
                    libc::dup2(log_file.as_raw_fd(), 1);
                    libc::dup2(log_file.as_raw_fd(), 2);
                }
            }
        }

        Ok(())
    }

    /// Daemonize the process
    pub fn daemonize(&self) -> Result<(), String> {
        // Check for existing instance
        if self.check_existing()? {
            return Err("Another instance is already running".to_string());
        }

        #[cfg(unix)]
        {
            // First fork
            unsafe {
                match libc::fork() {
                    -1 => return Err("First fork failed".to_string()),
                    0 => {
                        // Child continues
                    }
                    _ => {
                        // Parent exits
                        process::exit(0);
                    }
                }
            }

            // Create new session
            unsafe {
                if libc::setsid() == -1 {
                    return Err("setsid failed".to_string());
                }
            }

            // Ignore SIGCHLD
            unsafe {
                libc::signal(libc::SIGCHLD, libc::SIG_IGN);
            }

            // Second fork (optional, prevents zombie processes)
            unsafe {
                match libc::fork() {
                    -1 => return Err("Second fork failed".to_string()),
                    0 => {
                        // Grandchild continues
                    }
                    _ => {
                        // First child exits
                        process::exit(0);
                    }
                }
            }

            // Change working directory
            let _ = std::env::set_current_dir(&self.config.work_dir);

            // Set umask
            unsafe {
                libc::umask(self.config.umask as libc::mode_t);
            }
        }

        // Setup files
        self.setup_log_file()?;
        self.redirect_streams()?;
        self.write_pid_file()?;

        // Record start time
        self.start_time_ns.store(get_timestamp_ns(), Ordering::Relaxed);

        // Update state
        self.state.store(DaemonState::Running as u64, Ordering::Relaxed);

        Ok(())
    }

    /// Run in foreground (no daemonization)
    pub fn run_foreground(&self) -> Result<(), String> {
        self.setup_log_file()?;
        self.write_pid_file()?;
        self.start_time_ns.store(get_timestamp_ns(), Ordering::Relaxed);
        self.state.store(DaemonState::Running as u64, Ordering::Relaxed);
        Ok(())
    }

    /// Get current state
    pub fn get_state(&self) -> DaemonState {
        match self.state.load(Ordering::Relaxed) {
            0 => DaemonState::Starting,
            1 => DaemonState::Running,
            2 => DaemonState::Stopping,
            _ => DaemonState::Stopped,
        }
    }

    /// Set state to stopping
    pub fn set_stopping(&self) {
        self.state.store(DaemonState::Stopping as u64, Ordering::Relaxed);
    }

    /// Cleanup on shutdown
    pub fn cleanup(&self) {
        self.remove_pid_file();
        self.state.store(DaemonState::Stopped as u64, Ordering::Relaxed);
    }

    /// Get uptime in seconds
    pub fn get_uptime_secs(&self) -> u64 {
        let start = self.start_time_ns.load(Ordering::Relaxed);
        if start == 0 {
            return 0;
        }
        (get_timestamp_ns() - start) / 1_000_000_000
    }

    /// Get PID
    pub fn get_pid(&self) -> u32 {
        process::id()
    }

    /// Write to log file
    pub fn log(&self, message: &str) {
        if let Some(ref mut file) = *self.log_file.lock().unwrap() {
            let timestamp = get_timestamp_ns();
            let _ = writeln!(file, "[{}] {}", timestamp, message);
            let _ = file.flush();
        }
    }
}

impl Drop for DaemonManager {
    fn drop(&mut self) {
        self.remove_pid_file();
    }
}

#[inline]
fn get_timestamp_ns() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64
}

// Stub libc for non-Unix
#[cfg(not(unix))]
mod libc {
    pub type mode_t = u32;
    
    pub const SIGCHLD: i32 = 17;
    pub const SIG_IGN: usize = 1;
    
    pub unsafe fn fork() -> i32 { -1 }
    pub unsafe fn setsid() -> i32 { -1 }
    pub unsafe fn signal(_sig: i32, _handler: usize) -> usize { 0 }
    pub unsafe fn umask(_mask: mode_t) -> mode_t { 0 }
    pub unsafe fn kill(_pid: i32, _sig: i32) -> i32 { -1 }
    pub unsafe fn dup2(_old: i32, _new: i32) -> i32 { -1 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_daemon_config_default() {
        let config = DaemonConfig::default();
        assert_eq!(config.pid_file, PathBuf::from("/var/run/hft_bot.pid"));
    }

    #[test]
    fn test_daemon_state() {
        let config = DaemonConfig::default();
        let daemon = DaemonManager::new(config);
        
        assert_eq!(daemon.get_state(), DaemonState::Starting);
    }
}
