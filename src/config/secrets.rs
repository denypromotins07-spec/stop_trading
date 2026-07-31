//! Secure Memory Enclave for API Keys
//! 
//! Uses `mlock` to prevent secrets from being swapped to disk.
//! Strictly caps locked memory footprint to avoid OOM-killer termination.
//! Implements volatile_write for secure memory wiping on drop.

use libc::{mlock, munlock};
use std::alloc::{alloc, dealloc, Layout};
use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::cell::UnsafeCell;
use thiserror::Error;

/// Maximum locked memory per secret (4KB pages)
const MAX_LOCKED_SECRET_SIZE: usize = 4096;

/// Global counter for locked memory bytes
static LOCKED_MEMORY_BYTES: AtomicUsize = AtomicUsize::new(0);

/// Maximum total locked memory (1MB limit to avoid OOM issues)
const MAX_TOTAL_LOCKED_BYTES: usize = 1_048_576;

#[derive(Error, Debug)]
pub enum SecretError {
    #[error("Memory lock failed: {0}")]
    LockFailed(String),
    
    #[error("Memory unlock failed: {0}")]
    UnlockFailed(String),
    
    #[error("Exceeded maximum locked memory limit")]
    MemoryLimitExceeded,
    
    #[error("Invalid secret size")]
    InvalidSize,
    
    #[error("Allocation failed")]
    AllocationFailed,
}

/// Secure buffer that locks memory and wipes on drop
pub struct SecureBuffer {
    ptr: *mut u8,
    layout: Layout,
    len: usize,
    is_locked: AtomicBool,
}

unsafe impl Send for SecureBuffer {}
unsafe impl Sync for SecureBuffer {}

impl SecureBuffer {
    /// Create a new secure buffer with memory locking
    pub fn new(data: &[u8]) -> Result<Self, SecretError> {
        let len = data.len();
        
        if len == 0 || len > MAX_LOCKED_SECRET_SIZE {
            return Err(SecretError::InvalidSize);
        }
        
        // Check global locked memory limit
        let current = LOCKED_MEMORY_BYTES.load(Ordering::Relaxed);
        if current + len > MAX_TOTAL_LOCKED_BYTES {
            return Err(SecretError::MemoryLimitExceeded);
        }
        
        // Allocate memory with proper alignment
        let layout = Layout::from_size_align(len, 64)
            .map_err(|_| SecretError::InvalidSize)?;
        
        let ptr = unsafe { alloc(layout) };
        if ptr.is_null() {
            return Err(SecretError::AllocationFailed);
        }
        
        // Copy data to allocated memory
        unsafe {
            ptr::copy_nonoverlapping(data.as_ptr(), ptr, len);
        }
        
        // Lock memory to prevent swapping
        let lock_result = unsafe { mlock(ptr as *const libc::c_void, len) };
        if lock_result != 0 {
            // Clean up on failure
            unsafe {
                dealloc(ptr, layout);
            }
            return Err(SecretError::LockFailed(format!(
                "mlock failed with errno: {}",
                lock_result
            )));
        }
        
        // Update global counter
        LOCKED_MEMORY_BYTES.fetch_add(len, Ordering::Relaxed);
        
        Ok(SecureBuffer {
            ptr,
            layout,
            len,
            is_locked: AtomicBool::new(true),
        })
    }
    
    /// Get immutable reference to the secret data
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(self.ptr, self.len)
        }
    }
    
    /// Get mutable reference (careful with this!)
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe {
            std::slice::from_raw_parts_mut(self.ptr, self.len)
        }
    }
    
    /// Securely wipe the buffer contents using volatile writes
    pub fn wipe(&mut self) {
        if !self.is_locked.load(Ordering::Relaxed) {
            return;
        }
        
        // Volatile write zeros to prevent compiler optimization
        unsafe {
            for i in 0..self.len {
                ptr::write_volatile(self.ptr.add(i), 0u8);
            }
        }
    }
    
    /// Unlock memory manually (called automatically on drop)
    pub fn unlock(&mut self) -> Result<(), SecretError> {
        if !self.is_locked.load(Ordering::Relaxed) {
            return Ok(());
        }
        
        let result = unsafe { munlock(self.ptr as *const libc::c_void, self.len) };
        if result != 0 {
            return Err(SecretError::UnlockFailed(format!(
                "munlock failed with errno: {}",
                result
            )));
        }
        
        // Update global counter
        LOCKED_MEMORY_BYTES.fetch_sub(self.len, Ordering::Relaxed);
        self.is_locked.store(false, Ordering::Relaxed);
        
        Ok(())
    }
    
    /// Get the length of the secret
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }
    
    /// Check if buffer is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Drop for SecureBuffer {
    fn drop(&mut self) {
        // First wipe the data
        self.wipe();
        
        // Then unlock if still locked
        if self.is_locked.load(Ordering::Relaxed) {
            let _ = self.unlock();
        }
        
        // Finally deallocate
        unsafe {
            dealloc(self.ptr, self.layout);
        }
    }
}

/// Secure enclave holding multiple API credentials
pub struct SecretEnclave {
    api_key: UnsafeCell<Option<SecureBuffer>>,
    api_secret: UnsafeCell<Option<SecureBuffer>>,
    is_initialized: AtomicBool,
}

unsafe impl Send for SecretEnclave {}
unsafe impl Sync for SecretEnclave {}

impl SecretEnclave {
    /// Create a new empty enclave
    pub const fn new() -> Self {
        SecretEnclave {
            api_key: UnsafeCell::new(None),
            api_secret: UnsafeCell::new(None),
            is_initialized: AtomicBool::new(false),
        }
    }
    
    /// Initialize the enclave with API credentials
    pub fn init(&self, key: &[u8], secret: &[u8]) -> Result<(), SecretError> {
        if self.is_initialized.load(Ordering::Relaxed) {
            // Already initialized, wipe old values first
            self.wipe_all();
        }
        
        unsafe {
            *self.api_key.get() = Some(SecureBuffer::new(key)?);
            *self.api_secret.get() = Some(SecureBuffer::new(secret)?);
        }
        
        self.is_initialized.store(true, Ordering::Relaxed);
        Ok(())
    }
    
    /// Get API key slice (valid only while enclave exists)
    pub fn get_api_key(&self) -> Option<&[u8]> {
        if !self.is_initialized.load(Ordering::Relaxed) {
            return None;
        }
        unsafe {
            (*self.api_key.get()).as_ref().map(|b| b.as_slice())
        }
    }
    
    /// Get API secret slice
    pub fn get_api_secret(&self) -> Option<&[u8]> {
        if !self.is_initialized.load(Ordering::Relaxed) {
            return None;
        }
        unsafe {
            (*self.api_secret.get()).as_ref().map(|b| b.as_slice())
        }
    }
    
    /// Wipe all secrets in the enclave
    pub fn wipe_all(&self) {
        unsafe {
            if let Some(ref mut buf) = *self.api_key.get() {
                buf.wipe();
            }
            if let Some(ref mut buf) = *self.api_secret.get() {
                buf.wipe();
            }
        }
    }
    
    /// Check if enclave is initialized
    #[inline]
    pub fn is_initialized(&self) -> bool {
        self.is_initialized.load(Ordering::Relaxed)
    }
    
    /// Get total locked memory usage
    #[inline]
    pub fn locked_memory_bytes() -> usize {
        LOCKED_MEMORY_BYTES.load(Ordering::Relaxed)
    }
}

impl Default for SecretEnclave {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for SecretEnclave {
    fn drop(&mut self) {
        self.wipe_all();
    }
}

/// Helper function to load secrets from environment into enclave
pub fn load_secrets_from_env(
    enclave: &SecretEnclave,
    key_env: &str,
    secret_env: &str,
) -> Result<(), SecretError> {
    let key = std::env::var(key_env)
        .map_err(|_| SecretError::LockFailed(format!("Env var {} not found", key_env)))?;
    let secret = std::env::var(secret_env)
        .map_err(|_| SecretError::LockFailed(format!("Env var {} not found", secret_env)))?;
    
    enclave.init(key.as_bytes(), secret.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_secure_buffer_creation() {
        let data = b"test_secret_key_12345";
        let buffer = SecureBuffer::new(data).unwrap();
        assert_eq!(buffer.as_slice(), data);
        assert_eq!(buffer.len(), data.len());
    }
    
    #[test]
    fn test_secure_buffer_wipe() {
        let data = b"test_secret_key_12345";
        let mut buffer = SecureBuffer::new(data).unwrap();
        
        // Verify initial data
        assert_eq!(buffer.as_slice(), data);
        
        // Wipe and verify zeros
        buffer.wipe();
        let wiped = buffer.as_slice();
        assert!(wiped.iter().all(|&b| b == 0));
    }
    
    #[test]
    fn test_enclave_initialization() {
        let enclave = SecretEnclave::new();
        assert!(!enclave.is_initialized());
        
        enclave.init(b"api_key", b"api_secret").unwrap();
        assert!(enclave.is_initialized());
        
        assert_eq!(enclave.get_api_key(), Some(&b"api_key"[..]));
        assert_eq!(enclave.get_api_secret(), Some(&b"api_secret"[..]));
    }
    
    #[test]
    fn test_locked_memory_tracking() {
        let initial = SecretEnclave::locked_memory_bytes();
        
        let buffer = SecureBuffer::new(b"test").unwrap();
        let after_alloc = SecretEnclave::locked_memory_bytes();
        
        assert_eq!(after_alloc, initial + 4);
        
        drop(buffer);
        let after_drop = SecretEnclave::locked_memory_bytes();
        assert_eq!(after_drop, initial);
    }
}
