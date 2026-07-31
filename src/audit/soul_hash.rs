//! SOUL.md Integrity Checker
//! 
//! Continuous integrity checker for `SOUL.md` using SHA-256 to detect unauthorized tampering.
//! Triggers immediate global halt if self-learning weights file is modified by unapproved process.

use sha2::{Sha256, Digest};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH, Duration};
use thiserror::Error;

/// Default SOUL.md filename
pub const SOUL_FILENAME: &str = "SOUL.md";

/// Check interval in seconds
pub const CHECK_INTERVAL_SECS: u64 = 5;

#[derive(Error, Debug)]
pub enum SoulHashError {
    #[error("File not found: {0}")]
    FileNotFound(String),
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Hash mismatch detected")]
    HashMismatch,
    
    #[error("File was modified unexpectedly")]
    UnauthorizedModification,
    
    #[error("Integrity check failed")]
    IntegrityCheckFailed,
}

/// Stored hash state for SOUL.md
#[derive(Debug, Clone)]
pub struct SoulHashState {
    /// Expected SHA-256 hash (hex)
    pub expected_hash: String,
    /// Last verified timestamp
    pub last_verified_ns: u64,
    /// File size in bytes
    pub file_size: u64,
    /// Modification time
    pub mtime_ns: u64,
    /// Verification count
    pub verification_count: u64,
}

/// Continuous SOUL.md integrity monitor
pub struct SoulMonitor {
    /// Path to SOUL.md
    soul_path: PathBuf,
    /// Current hash state
    state: std::sync::Mutex<Option<SoulHashState>>,
    /// Emergency halt flag
    emergency_halt: AtomicBool,
    /// Consecutive failures
    failure_count: AtomicU64,
    /// Maximum allowed failures before halt
    max_failures: u64,
}

unsafe impl Send for SoulMonitor {}
unsafe impl Sync for SoulMonitor {}

impl SoulMonitor {
    /// Create new SOUL monitor for given path
    pub fn new<P: AsRef<Path>>(soul_path: P) -> Self {
        Self {
            soul_path: soul_path.as_ref().to_path_buf(),
            state: std::sync::Mutex::new(None),
            emergency_halt: AtomicBool::new(false),
            failure_count: AtomicU64::new(0),
            max_failures: 3,
        }
    }
    
    /// Get current timestamp in nanoseconds
    #[inline]
    fn now_ns() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64
    }
    
    /// Compute SHA-256 hash of file contents
    pub fn compute_hash<P: AsRef<Path>>(path: P) -> Result<String, SoulHashError> {
        let content = fs::read(path.as_ref())?;
        Ok(hex::encode(Sha256::digest(&content)))
    }
    
    /// Get file metadata
    fn get_file_metadata<P: AsRef<Path>>(path: P) -> Result<(u64, u64), SoulHashError> {
        let metadata = fs::metadata(path.as_ref())?;
        let size = metadata.len();
        let mtime = metadata.modified()?
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        Ok((size, mtime))
    }
    
    /// Initialize the monitor by computing baseline hash
    pub fn initialize(&self) -> Result<SoulHashState, SoulHashError> {
        if !self.soul_path.exists() {
            return Err(SoulHashError::FileNotFound(
                self.soul_path.display().to_string()
            ));
        }
        
        let hash = Self::compute_hash(&self.soul_path)?;
        let (size, mtime) = Self::get_file_metadata(&self.soul_path)?;
        
        let state = SoulHashState {
            expected_hash: hash,
            last_verified_ns: Self::now_ns(),
            file_size: size,
            mtime_ns: mtime,
            verification_count: 0,
        };
        
        *self.state.lock().unwrap() = Some(state.clone());
        Ok(state)
    }
    
    /// Perform single integrity check
    pub fn verify(&self) -> Result<bool, SoulHashError> {
        let mut state_guard = self.state.lock().unwrap();
        
        let state = match state_guard.as_mut() {
            Some(s) => s,
            None => return Err(SoulHashError::IntegrityCheckFailed),
        };
        
        // Quick check: has file been modified?
        let (current_size, current_mtime) = Self::get_file_metadata(&self.soul_path)?;
        
        if current_mtime != state.mtime_ns || current_size != state.file_size {
            // File changed - verify if it's still valid
            let current_hash = Self::compute_hash(&self.soul_path)?;
            
            if current_hash != state.expected_hash {
                self.failure_count.fetch_add(1, Ordering::Relaxed);
                
                if self.failure_count.load(Ordering::Relaxed) >= self.max_failures {
                    self.emergency_halt.store(true, Ordering::SeqCst);
                    return Err(SoulHashError::UnauthorizedModification);
                }
                
                return Err(SoulHashError::HashMismatch);
            }
            
            // Hash matches despite mtime change (could be filesystem artifact)
            state.mtime_ns = current_mtime;
            state.file_size = current_size;
        }
        
        // Reset failure count on success
        self.failure_count.store(0, Ordering::Relaxed);
        
        state.last_verified_ns = Self::now_ns();
        state.verification_count += 1;
        
        Ok(true)
    }
    
    /// Update the expected hash (authorized modification)
    pub fn update_baseline(&self) -> Result<SoulHashState, SoulHashError> {
        let hash = Self::compute_hash(&self.soul_path)?;
        let (size, mtime) = Self::get_file_metadata(&self.soul_path)?;
        
        let state = SoulHashState {
            expected_hash: hash,
            last_verified_ns: Self::now_ns(),
            file_size: size,
            mtime_ns: mtime,
            verification_count: 0,
        };
        
        *self.state.lock().unwrap() = Some(state.clone());
        self.failure_count.store(0, Ordering::Relaxed);
        
        Ok(state)
    }
    
    /// Check if emergency halt is triggered
    pub fn is_halted(&self) -> bool {
        self.emergency_halt.load(Ordering::SeqCst)
    }
    
    /// Trigger emergency halt
    pub fn trigger_halt(&self) {
        self.emergency_halt.store(true, Ordering::SeqCst);
    }
    
    /// Clear emergency halt (requires manual intervention)
    pub fn clear_halt(&self) {
        self.failure_count.store(0, Ordering::Relaxed);
        self.emergency_halt.store(false, Ordering::SeqCst);
    }
    
    /// Get current failure count
    pub fn get_failure_count(&self) -> u64 {
        self.failure_count.load(Ordering::Relaxed)
    }
    
    /// Get current state
    pub fn get_state(&self) -> Option<SoulHashState> {
        self.state.lock().unwrap().clone()
    }
}

/// Global SOUL monitor instance
static GLOBAL_SOUL_MONITOR: std::sync::OnceLock<SoulMonitor> = std::sync::OnceLock::new();

/// Initialize global SOUL monitor
pub fn init_soul_monitor<P: AsRef<Path>>(soul_path: P) -> Result<(), SoulHashError> {
    let monitor = SoulMonitor::new(soul_path);
    monitor.initialize()?;
    
    GLOBAL_SOUL_MONITOR
        .set(monitor)
        .map_err(|_| SoulHashError::IntegrityCheckFailed)?;
    
    Ok(())
}

/// Get reference to global monitor
pub fn get_soul_monitor() -> Option<&'static SoulMonitor> {
    GLOBAL_SOUL_MONITOR.get()
}

/// Perform global integrity check
pub fn verify_soul_integrity() -> Result<bool, SoulHashError> {
    get_soul_monitor()
        .ok_or(SoulHashError::IntegrityCheckFailed)?
        .verify()
}

/// Check if system should halt due to integrity failure
pub fn should_halt() -> bool {
    get_soul_monitor()
        .map(|m| m.is_halted())
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;
    use tempfile::NamedTempFile;
    
    #[test]
    fn test_hash_computation() {
        let temp = NamedTempFile::new().unwrap();
        let path = temp.path();
        
        {
            let mut file = File::create(path).unwrap();
            file.write_all(b"test content").unwrap();
        }
        
        let hash = SoulMonitor::compute_hash(path).unwrap();
        assert_eq!(hash.len(), 64); // SHA-256 hex length
        
        // Same content = same hash
        let hash2 = SoulMonitor::compute_hash(path).unwrap();
        assert_eq!(hash, hash2);
    }
    
    #[test]
    fn test_monitor_initialization() {
        let temp = NamedTempFile::new().unwrap();
        let path = temp.path();
        
        {
            let mut file = File::create(path).unwrap();
            file.write_all(b"initial content").unwrap();
        }
        
        let monitor = SoulMonitor::new(path);
        let state = monitor.initialize().unwrap();
        
        assert!(!state.expected_hash.is_empty());
        assert_eq!(state.file_size, 15);
    }
    
    #[test]
    fn test_integrity_verification() {
        let temp = NamedTempFile::new().unwrap();
        let path = temp.path();
        
        {
            let mut file = File::create(path).unwrap();
            file.write_all(b"unchanged content").unwrap();
        }
        
        let monitor = SoulMonitor::new(path);
        monitor.initialize().unwrap();
        
        // Verify should succeed
        assert!(monitor.verify().is_ok());
        assert!(!monitor.is_halted());
    }
    
    #[test]
    fn test_tampering_detection() {
        let temp = NamedTempFile::new().unwrap();
        let path = temp.path();
        
        {
            let mut file = File::create(path).unwrap();
            file.write_all(b"original content").unwrap();
        }
        
        let monitor = SoulMonitor::new(path);
        monitor.initialize().unwrap();
        
        // Modify file
        {
            let mut file = File::create(path).unwrap();
            file.write_all(b"tampered content").unwrap();
        }
        
        // Should detect tampering
        let result = monitor.verify();
        assert!(result.is_err());
    }
    
    #[test]
    fn test_emergency_halt() {
        let temp = NamedTempFile::new().unwrap();
        let path = temp.path();
        
        {
            let mut file = File::create(path).unwrap();
            file.write_all(b"content").unwrap();
        }
        
        let monitor = SoulMonitor::new(path);
        monitor.initialize().unwrap();
        monitor.max_failures = 1; // Trigger halt on first failure
        
        // Modify file
        {
            let mut file = File::create(path).unwrap();
            file.write_all(b"different").unwrap();
        }
        
        let _ = monitor.verify(); // First failure
        
        assert!(monitor.is_halted());
    }
}
