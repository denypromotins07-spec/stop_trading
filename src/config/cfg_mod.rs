//! Configuration Module Root
//! 
//! Exports validated, immutable config structs to the global orchestrator.
//! Provides unified access to system configuration, secrets, and risk limits.

pub mod parser;
pub mod secrets;

pub use parser::{Config, ConfigError, SystemConfig, RiskConfig, NetworkConfig, ExchangeConfig, MLConfig};
pub use secrets::{SecretEnclave, SecureBuffer, SecretError, load_secrets_from_env};

use std::sync::OnceLock;
use crate::config::parser::MAX_MEMORY_BYTES;

/// Global configuration instance (initialized once at startup)
static GLOBAL_CONFIG: OnceLock<Config> = OnceLock::new();

/// Global secret enclave (initialized once at startup)
static GLOBAL_SECRETS: OnceLock<SecretEnclave> = OnceLock::new();

/// Initialize the global configuration
pub fn init_config(config: Config) -> Result<(), ConfigError> {
    GLOBAL_CONFIG
        .set(config)
        .map_err(|_| ConfigError::SchemaValidation("Config already initialized".to_string()))
}

/// Initialize the global secret enclave
pub fn init_secrets(enclave: SecretEnclave) -> Result<(), SecretError> {
    GLOBAL_SECRETS
        .set(enclave)
        .map_err(|_| SecretError::LockFailed("Secrets already initialized".to_string()))
}

/// Get reference to global configuration
/// Panics if config not initialized - call init_config first
#[inline]
pub fn get_config() -> &'static Config {
    GLOBAL_CONFIG
        .get()
        .expect("Config not initialized. Call init_config first.")
}

/// Get reference to global secrets enclave
/// Returns None if secrets not initialized
#[inline]
pub fn get_secrets() -> Option<&'static SecretEnclave> {
    GLOBAL_SECRETS.get()
}

/// Check if configuration is initialized
#[inline]
pub fn is_config_initialized() -> bool {
    GLOBAL_CONFIG.get().is_some()
}

/// Check if secrets are initialized
#[inline]
pub fn is_secrets_initialized() -> bool {
    GLOBAL_SECRETS.get().is_some()
}

/// Get current locked memory usage
#[inline]
pub fn locked_memory_usage() -> usize {
    SecretEnclave::locked_memory_bytes()
}

/// Get remaining locked memory budget
#[inline]
pub fn remaining_locked_memory() -> usize {
    const MAX_LOCKED: usize = 1_048_576; // 1MB
    MAX_LOCKED.saturating_sub(locked_memory_usage())
}

/// Validate total memory constraints
pub fn validate_memory_constraints() -> bool {
    let heap_usage = get_config().system.max_heap_mb as usize * 1024 * 1024;
    let locked_usage = locked_memory_usage();
    
    // Total must stay under 6.5GB
    heap_usage + locked_usage <= MAX_MEMORY_BYTES
}

/// Configuration builder for fluent initialization
pub struct ConfigBuilder {
    config: Config,
}

impl ConfigBuilder {
    pub fn new() -> Self {
        Self {
            config: Config::default(),
        }
    }
    
    pub fn with_max_heap(mut self, mb: u64) -> Self {
        self.config.system.max_heap_mb = mb;
        self
    }
    
    pub fn with_worker_threads(mut self, threads: usize) -> Self {
        self.config.system.worker_threads = threads;
        self
    }
    
    pub fn with_ring_buffer_size(mut self, size: usize) -> Self {
        self.config.system.ring_buffer_size = size;
        self
    }
    
    pub fn with_max_leverage(mut self, leverage: f64) -> Self {
        self.config.risk.max_leverage = leverage;
        self
    }
    
    pub fn with_max_daily_loss(mut self, loss: f64) -> Self {
        self.config.risk.max_daily_loss = loss;
        self
    }
    
    pub fn with_exchange(mut self, name: &str, rest: &str, ws: &str) -> Self {
        self.config.exchange.name = name.to_string();
        self.config.exchange.rest_endpoint = rest.to_string();
        self.config.exchange.ws_endpoint = ws.to_string();
        self
    }
    
    pub fn build(mut self) -> Result<Config, ConfigError> {
        self.config.validate()?;
        Ok(self.config)
    }
    
    pub fn build_and_init(self) -> Result<(), ConfigError> {
        let config = self.build()?;
        init_config(config)
    }
}

impl Default for ConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_config_builder() {
        let config = ConfigBuilder::new()
            .with_max_heap(2048)
            .with_worker_threads(4)
            .with_max_leverage(2.0)
            .build()
            .unwrap();
        
        assert_eq!(config.system.max_heap_mb, 2048);
        assert_eq!(config.system.worker_threads, 4);
        assert_eq!(config.risk.max_leverage, 2.0);
    }
    
    #[test]
    fn test_global_config_initialization() {
        // Clear any existing config (for test isolation)
        // Note: In real code, we'd need a way to reset OnceLock
        
        let config = Config::default();
        assert!(init_config(config).is_ok());
        
        assert!(is_config_initialized());
        assert!(validate_memory_constraints());
    }
    
    #[test]
    fn test_secret_enclave_integration() {
        let enclave = SecretEnclave::new();
        assert!(init_secrets(enclave).is_ok());
        
        assert!(is_secrets_initialized());
        assert!(remaining_locked_memory() <= 1_048_576);
    }
    
    #[test]
    fn test_memory_budget_tracking() {
        let initial = locked_memory_usage();
        
        let buffer = SecureBuffer::new(b"test_data").unwrap();
        let after = locked_memory_usage();
        
        assert_eq!(after, initial + 9);
        
        drop(buffer);
        let final_usage = locked_memory_usage();
        assert_eq!(final_usage, initial);
    }
}
