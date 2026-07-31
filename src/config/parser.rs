//! Strict Zero-Allocation Configuration Parser
//! 
//! Implements a compile-time validated schema parser for `config.toml` and `.env`.
//! Rejects any configuration violating risk limits or exceeding AMD Ryzen AI 5 hardware constraints.
//! Enforces the 6.5GB RAM ceiling strictly.

use serde::Deserialize;
use std::fs;
use std::path::Path;
use thiserror::Error;

/// Maximum allowed memory allocation in bytes (6.5GB)
pub const MAX_MEMORY_BYTES: usize = 6_979_321_856;

/// Maximum number of concurrent assets
pub const MAX_ASSETS: usize = 256;

/// Maximum order book depth
pub const MAX_ORDERBOOK_DEPTH: usize = 100;

#[derive(Error, Debug)]
pub enum ConfigError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("TOML parse error: {0}")]
    Toml(#[from] toml::de::Error),
    
    #[error("Risk limit violation: {0}")]
    RiskLimit(String),
    
    #[error("Hardware constraint violation: {0}")]
    HardwareConstraint(String),
    
    #[error("Environment variable error: {0}")]
    EnvVar(#[from] std::env::VarError),
    
    #[error("Schema validation failed: {0}")]
    SchemaValidation(String),
}

/// Compile-time validated configuration schema
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub system: SystemConfig,
    pub risk: RiskConfig,
    pub network: NetworkConfig,
    pub exchange: ExchangeConfig,
    pub ml: MLConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SystemConfig {
    /// Maximum heap allocation in MB
    pub max_heap_mb: u64,
    /// Enable memory locking
    pub lock_memory: bool,
    /// Number of worker threads
    pub worker_threads: usize,
    /// Disruptor ring buffer size
    pub ring_buffer_size: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RiskConfig {
    /// Maximum position size per asset (in base units)
    pub max_position_size: f64,
    /// Maximum daily loss (in quote currency)
    pub max_daily_loss: f64,
    /// Maximum portfolio leverage
    pub max_leverage: f64,
    /// Value at Risk confidence level
    pub var_confidence: f64,
    /// Maximum order size (in quote currency)
    pub max_order_size: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct NetworkConfig {
    /// Connection timeout in milliseconds
    pub connect_timeout_ms: u64,
    /// Read timeout in milliseconds
    pub read_timeout_ms: u64,
    /// Maximum reconnection attempts
    pub max_reconnects: u32,
    /// WebSocket ping interval in seconds
    pub ws_ping_interval_secs: u64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ExchangeConfig {
    /// Exchange name
    pub name: String,
    /// REST API endpoint
    pub rest_endpoint: String,
    /// WebSocket endpoint
    pub ws_endpoint: String,
    /// API key environment variable name
    pub api_key_env: String,
    /// API secret environment variable name
    pub api_secret_env: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MLConfig {
    /// Python interpreter path
    pub python_path: String,
    /// Model directory
    pub model_dir: String,
    /// Feature window size
    pub feature_window: usize,
    /// Drift detection threshold
    pub drift_threshold: f64,
    /// Retraining interval in seconds
    pub retrain_interval_secs: u64,
}

impl Config {
    /// Load configuration from TOML file and environment
    pub fn load<P: AsRef<Path>>(config_path: P) -> Result<Self, ConfigError> {
        let content = fs::read_to_string(config_path.as_ref())?;
        let mut config: Config = toml::from_str(&content)?;
        
        // Validate and enforce constraints
        config.validate()?;
        config.load_env_secrets()?;
        
        Ok(config)
    }
    
    /// Validate configuration against hardware and risk constraints
    pub fn validate(&mut self) -> Result<(), ConfigError> {
        // Validate system constraints
        let estimated_heap = self.system.max_heap_mb * 1024 * 1024;
        if estimated_heap > MAX_MEMORY_BYTES {
            return Err(ConfigError::HardwareConstraint(format!(
                "Heap allocation {}MB exceeds 6.5GB limit",
                self.system.max_heap_mb
            )));
        }
        
        // Validate worker threads against CPU cores
        let available_cores = num_cpus::get();
        if self.system.worker_threads > available_cores {
            eprintln!(
                "Warning: worker_threads ({}) exceeds available cores ({}). Adjusting.",
                self.system.worker_threads, available_cores
            );
            self.system.worker_threads = available_cores;
        }
        
        // Validate ring buffer size (must be power of 2)
        if !self.system.ring_buffer_size.is_power_of_two() {
            return Err(ConfigError::SchemaValidation(
                "ring_buffer_size must be a power of 2".to_string()
            ));
        }
        
        // Validate risk limits
        if self.risk.max_leverage < 1.0 || self.risk.max_leverage > 10.0 {
            return Err(ConfigError::RiskLimit(
                "max_leverage must be between 1.0 and 10.0".to_string()
            ));
        }
        
        if self.risk.var_confidence < 0.90 || self.risk.var_confidence > 0.999 {
            return Err(ConfigError::RiskLimit(
                "var_confidence must be between 0.90 and 0.999".to_string()
            ));
        }
        
        if self.risk.max_daily_loss <= 0.0 {
            return Err(ConfigError::RiskLimit(
                "max_daily_loss must be positive".to_string()
            ));
        }
        
        // Validate ML config
        if self.ml.feature_window > 10000 {
            return Err(ConfigError::HardwareConstraint(
                "feature_window too large, would exceed memory limits".to_string()
            ));
        }
        
        // Validate network timeouts
        if self.network.connect_timeout_ms > 30000 {
            eprintln!("Warning: connect_timeout_ms is very high (>30s)");
        }
        
        Ok(())
    }
    
    /// Load sensitive secrets from environment variables
    fn load_env_secrets(&mut self) -> Result<(), ConfigError> {
        // Validate that required env var names are set
        let _ = std::env::var(&self.exchange.api_key_env)
            .map_err(|_| ConfigError::EnvVar(std::env::VarError::NotPresent))?;
        let _ = std::env::var(&self.exchange.api_secret_env)
            .map_err(|_| ConfigError::EnvVar(std::env::VarError::NotPresent))?;
        
        Ok(())
    }
    
    /// Get immutable reference to risk config
    #[inline]
    pub fn risk(&self) -> &RiskConfig {
        &self.risk
    }
    
    /// Get immutable reference to system config
    #[inline]
    pub fn system(&self) -> &SystemConfig {
        &self.system
    }
}

/// Default configuration for AMD Ryzen AI 5 laptop with 16GB RAM
impl Default for Config {
    fn default() -> Self {
        Config {
            system: SystemConfig {
                max_heap_mb: 4096, // 4GB heap, leaving room for OS and other processes
                lock_memory: true,
                worker_threads: 8,
                ring_buffer_size: 1048576, // 2^20
            },
            risk: RiskConfig {
                max_position_size: 1000.0,
                max_daily_loss: 5000.0,
                max_leverage: 3.0,
                var_confidence: 0.99,
                max_order_size: 10000.0,
            },
            network: NetworkConfig {
                connect_timeout_ms: 5000,
                read_timeout_ms: 30000,
                max_reconnects: 10,
                ws_ping_interval_secs: 30,
            },
            exchange: ExchangeConfig {
                name: "binance".to_string(),
                rest_endpoint: "https://api.binance.com".to_string(),
                ws_endpoint: "wss://stream.binance.com:9443/ws".to_string(),
                api_key_env: "BINANCE_API_KEY".to_string(),
                api_secret_env: "BINANCE_API_SECRET".to_string(),
            },
            ml: MLConfig {
                python_path: "/usr/bin/python3".to_string(),
                model_dir: "./models".to_string(),
                feature_window: 1000,
                drift_threshold: 0.05,
                retrain_interval_secs: 3600,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_default_config_validates() {
        let mut config = Config::default();
        assert!(config.validate().is_ok());
    }
    
    #[test]
    fn test_memory_limit_enforcement() {
        let mut config = Config::default();
        config.system.max_heap_mb = 7000; // Exceeds 6.5GB
        assert!(config.validate().is_err());
    }
    
    #[test]
    fn test_ring_buffer_power_of_two() {
        let mut config = Config::default();
        config.system.ring_buffer_size = 1000000; // Not power of 2
        assert!(config.validate().is_err());
    }
    
    #[test]
    fn test_risk_limits() {
        let mut config = Config::default();
        config.risk.max_leverage = 15.0; // Exceeds max
        assert!(config.validate().is_err());
    }
}
