//! Network Resilience Module Root
//! 
//! This module manages connection health, circuit breakers, and automated
//! API key rotation for robust network operations.

pub mod ws_throttle;
pub mod rest_retry;

// Re-export main types for convenience
pub use ws_throttle::{
    DynamicThrottleController, MessagePriority, MessageType, ThrottlerStats,
    TokenBucket, WsMessage, WsThrottler,
};

pub use rest_retry::{
    BackoffCalculator, BinanceRetryHelper, IdempotencyKeyGenerator, RetryConfig,
    RetryFailure, RetryResult, RetryStats, RestRetryClient,
};

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use tracing::{debug, error, info, warn};

/// Connection health status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionHealth {
    Healthy,
    Degraded,
    Unhealthy,
    Disconnected,
}

/// Circuit breaker state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    Closed,   // Normal operation
    Open,     // Failing fast
    HalfOpen, // Testing recovery
}

/// Circuit breaker configuration
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    /// Failure threshold to trip the breaker
    pub failure_threshold: u32,
    /// Success threshold to close from half-open
    pub success_threshold: u32,
    /// Time in open state before half-open
    pub open_timeout: Duration,
    /// Sliding window size for failure counting
    pub sliding_window_size: usize,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            success_threshold: 2,
            open_timeout: Duration::from_secs(30),
            sliding_window_size: 10,
        }
    }
}

/// Circuit breaker implementation
pub struct CircuitBreaker {
    /// Current state
    state: parking_lot::Mutex<CircuitState>,
    /// State changed at timestamp
    state_changed_at: AtomicU64,
    /// Failure count in current window
    failure_count: AtomicUsize,
    /// Success count (for half-open)
    success_count: AtomicUsize,
    /// Configuration
    config: CircuitBreakerConfig,
    /// Statistics
    stats: CircuitBreakerStats,
}

#[derive(Debug, Default)]
pub struct CircuitBreakerStats {
    pub times_opened: AtomicUsize,
    pub times_closed: AtomicUsize,
    pub total_failures: AtomicUsize,
    pub total_successes: AtomicUsize,
    pub requests_rejected: AtomicUsize,
}

impl CircuitBreaker {
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            state: parking_lot::Mutex::new(CircuitState::Closed),
            state_changed_at: AtomicU64::new(current_timestamp_ns()),
            failure_count: AtomicUsize::new(0),
            success_count: AtomicUsize::new(0),
            config,
            stats: CircuitBreakerStats::default(),
        }
    }
    
    /// Check if request should be allowed
    pub fn allow_request(&self) -> bool {
        let state = *self.state.lock();
        
        match state {
            CircuitState::Closed => true,
            CircuitState::Open => {
                // Check if timeout has elapsed
                let elapsed = current_timestamp_ns().saturating_sub(
                    self.state_changed_at.load(Ordering::Relaxed)
                );
                
                if elapsed >= self.config.open_timeout.as_nanos() as u64 {
                    // Transition to half-open
                    *self.state.lock() = CircuitState::HalfOpen;
                    self.success_count.store(0, Ordering::Relaxed);
                    true
                } else {
                    self.stats.requests_rejected.fetch_add(1, Ordering::Relaxed);
                    false
                }
            }
            CircuitState::HalfOpen => true,
        }
    }
    
    /// Record a successful request
    pub fn record_success(&self) {
        self.stats.total_successes.fetch_add(1, Ordering::Relaxed);
        
        let mut state = self.state.lock();
        
        match *state {
            CircuitState::HalfOpen => {
                let successes = self.success_count.fetch_add(1, Ordering::Relaxed) + 1;
                if successes >= self.config.success_threshold {
                    *state = CircuitState::Closed;
                    self.failure_count.store(0, Ordering::Relaxed);
                    self.state_changed_at.store(current_timestamp_ns(), Ordering::Relaxed);
                    self.stats.times_closed.fetch_add(1, Ordering::Relaxed);
                    info!("Circuit breaker closed after successful recovery");
                }
            }
            CircuitState::Closed => {
                // Reset failure count on success
                self.failure_count.store(0, Ordering::Relaxed);
            }
            CircuitState::Open => {}
        }
    }
    
    /// Record a failed request
    pub fn record_failure(&self) {
        self.stats.total_failures.fetch_add(1, Ordering::Relaxed);
        
        let mut state = self.state.lock();
        
        match *state {
            CircuitState::Closed => {
                let failures = self.failure_count.fetch_add(1, Ordering::Relaxed) + 1;
                if failures >= self.config.failure_threshold as usize {
                    *state = CircuitState::Open;
                    self.state_changed_at.store(current_timestamp_ns(), Ordering::Relaxed);
                    self.stats.times_opened.fetch_add(1, Ordering::Relaxed);
                    warn!("Circuit breaker opened due to repeated failures");
                }
            }
            CircuitState::HalfOpen => {
                // Immediately reopen on any failure
                *state = CircuitState::Open;
                self.state_changed_at.store(current_timestamp_ns(), Ordering::Relaxed);
                self.stats.times_opened.fetch_add(1, Ordering::Relaxed);
            }
            CircuitState::Open => {}
        }
    }
    
    /// Get current state
    pub fn state(&self) -> CircuitState {
        *self.state.lock()
    }
    
    /// Force open the circuit breaker
    pub fn force_open(&self) {
        *self.state.lock() = CircuitState::Open;
        self.state_changed_at.store(current_timestamp_ns(), Ordering::Relaxed);
        self.stats.times_opened.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Force close the circuit breaker
    pub fn force_close(&self) {
        *self.state.lock() = CircuitState::Closed;
        self.failure_count.store(0, Ordering::Relaxed);
        self.state_changed_at.store(current_timestamp_ns(), Ordering::Relaxed);
        self.stats.times_closed.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Get statistics
    pub fn stats(&self) -> &CircuitBreakerStats {
        &self.stats
    }
}

fn current_timestamp_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64
}

/// API key rotation manager
pub struct ApiKeyRotator {
    /// Current active key index
    active_key_index: AtomicUsize,
    /// Available keys
    keys: parking_lot::Mutex<Vec<ApiKey>>,
    /// Key health tracking
    key_health: parking_lot::Mutex<std::collections::HashMap<String, KeyHealth>>,
    /// Rotation enabled
    rotation_enabled: AtomicBool,
    /// Statistics
    stats: RotatorStats,
}

#[derive(Debug, Clone)]
struct ApiKey {
    id: String,
    api_key: String,
    secret_key: String,
    created_at: u64,
    last_rotated: u64,
}

#[derive(Debug, Clone)]
struct KeyHealth {
    success_count: u32,
    failure_count: u32,
    last_used: u64,
    is_blocked: bool,
    block_until: u64,
}

#[derive(Debug, Default)]
pub struct RotatorStats {
    pub rotations_performed: AtomicUsize,
    pub keys_blocked: AtomicUsize,
    pub fallback_activations: AtomicUsize,
}

impl ApiKeyRotator {
    pub fn new() -> Self {
        Self {
            active_key_index: AtomicUsize::new(0),
            keys: parking_lot::Mutex::new(Vec::new()),
            key_health: parking_lot::Mutex::new(std::collections::HashMap::new()),
            rotation_enabled: AtomicBool::new(true),
            stats: RotatorStats::default(),
        }
    }
    
    /// Add an API key
    pub fn add_key(&self, id: impl Into<String>, api_key: impl Into<String>, secret_key: impl Into<String>) {
        let key = ApiKey {
            id: id.into(),
            api_key: api_key.into(),
            secret_key: secret_key.into(),
            created_at: current_timestamp_ns(),
            last_rotated: current_timestamp_ns(),
        };
        
        let mut keys = self.keys.lock();
        keys.push(key);
        
        let mut health = self.key_health.lock();
        health.insert(key.id.clone(), KeyHealth {
            success_count: 0,
            failure_count: 0,
            last_used: 0,
            is_blocked: false,
            block_until: 0,
        });
    }
    
    /// Get current active key
    pub fn get_active_key(&self) -> Option<(String, String, String)> {
        let keys = self.keys.lock();
        let index = self.active_key_index.load(Ordering::Relaxed);
        
        if index >= keys.len() {
            return None;
        }
        
        let key = &keys[index];
        Some((key.id.clone(), key.api_key.clone(), key.secret_key.clone()))
    }
    
    /// Record success for current key
    pub fn record_success(&self) {
        if let Some((id, _, _)) = self.get_active_key() {
            let mut health = self.key_health.lock();
            if let Some(h) = health.get_mut(&id) {
                h.success_count += 1;
                h.last_used = current_timestamp_ns();
            }
        }
    }
    
    /// Record failure for current key
    pub fn record_failure(&self) {
        if let Some((id, _, _)) = self.get_active_key() {
            let mut health = self.key_health.lock();
            if let Some(h) = health.get_mut(&id) {
                h.failure_count += 1;
                h.last_used = current_timestamp_ns();
                
                // Block key if too many failures
                if h.failure_count >= 10 {
                    h.is_blocked = true;
                    h.block_until = current_timestamp_ns() + Duration::from_secs(300).as_nanos() as u64;
                    self.stats.keys_blocked.fetch_add(1, Ordering::Relaxed);
                    
                    // Rotate to next key
                    self.rotate_key();
                }
            }
        }
    }
    
    /// Rotate to next available key
    pub fn rotate_key(&self) -> bool {
        if !self.rotation_enabled.load(Ordering::Relaxed) {
            return false;
        }
        
        let keys = self.keys.lock();
        if keys.len() <= 1 {
            return false;
        }
        
        let current = self.active_key_index.load(Ordering::Relaxed);
        let mut next = (current + 1) % keys.len();
        
        // Find next unblocked key
        let health = self.key_health.lock();
        let start = next;
        loop {
            if let Some(h) = health.get(&keys[next].id) {
                if !h.is_blocked || current_timestamp_ns() > h.block_until {
                    break;
                }
            }
            
            next = (next + 1) % keys.len();
            if next == start {
                // All keys blocked, use current anyway
                break;
            }
        }
        
        self.active_key_index.store(next, Ordering::Relaxed);
        self.stats.rotations_performed.fetch_add(1, Ordering::Relaxed);
        
        info!("Rotated API key to index {}", next);
        true
    }
    
    /// Enable/disable rotation
    pub fn set_rotation_enabled(&self, enabled: bool) {
        self.rotation_enabled.store(enabled, Ordering::Relaxed);
    }
    
    /// Get number of available keys
    pub fn key_count(&self) -> usize {
        self.keys.lock().len()
    }
    
    /// Get statistics
    pub fn stats(&self) -> &RotatorStats {
        &self.stats
    }
}

impl Default for ApiKeyRotator {
    fn default() -> Self {
        Self::new()
    }
}

/// Connection health monitor
pub struct ConnectionHealthMonitor {
    /// Last successful heartbeat
    last_heartbeat: AtomicU64,
    /// Consecutive failures
    consecutive_failures: AtomicUsize,
    /// Health check interval
    check_interval: Duration,
    /// Failure threshold
    failure_threshold: usize,
}

impl ConnectionHealthMonitor {
    pub fn new(check_interval: Duration, failure_threshold: usize) -> Self {
        Self {
            last_heartbeat: AtomicU64::new(current_timestamp_ns()),
            consecutive_failures: AtomicUsize::new(0),
            check_interval,
            failure_threshold,
        }
    }
    
    /// Record a successful heartbeat
    pub fn record_heartbeat(&self) {
        self.last_heartbeat.store(current_timestamp_ns(), Ordering::Relaxed);
        self.consecutive_failures.store(0, Ordering::Relaxed);
    }
    
    /// Record a failed health check
    pub fn record_failure(&self) {
        self.consecutive_failures.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Get current health status
    pub fn get_health(&self) -> ConnectionHealth {
        let failures = self.consecutive_failures.load(Ordering::Relaxed);
        
        if failures >= self.failure_threshold {
            return ConnectionHealth::Disconnected;
        }
        
        let elapsed = current_timestamp_ns().saturating_sub(
            self.last_heartbeat.load(Ordering::Relaxed)
        );
        let elapsed_duration = Duration::from_nanos(elapsed);
        
        if elapsed_duration > self.check_interval * 3 {
            ConnectionHealth::Unhealthy
        } else if elapsed_duration > self.check_interval * 2 {
            ConnectionHealth::Degraded
        } else {
            ConnectionHealth::Healthy
        }
    }
    
    /// Check if connection needs reconnection
    pub fn needs_reconnect(&self) -> bool {
        matches!(self.get_health(), ConnectionHealth::Unhealthy | ConnectionHealth::Disconnected)
    }
}

/// Global resilience manager coordinating all resilience components
pub struct ResilienceManager {
    /// Circuit breaker for REST API
    rest_circuit_breaker: CircuitBreaker,
    /// Circuit breaker for WebSocket
    ws_circuit_breaker: CircuitBreaker,
    /// API key rotator
    key_rotator: ApiKeyRotator,
    /// Connection health monitor
    health_monitor: ConnectionHealthMonitor,
}

impl ResilienceManager {
    pub fn new() -> Self {
        Self {
            rest_circuit_breaker: CircuitBreaker::new(CircuitBreakerConfig::default()),
            ws_circuit_breaker: CircuitBreaker::new(CircuitBreakerConfig {
                failure_threshold: 10,
                ..Default::default()
            }),
            key_rotator: ApiKeyRotator::new(),
            health_monitor: ConnectionHealthMonitor::new(
                Duration::from_secs(30),
                5,
            ),
        }
    }
    
    /// Get REST circuit breaker
    pub fn rest_cb(&self) -> &CircuitBreaker {
        &self.rest_circuit_breaker
    }
    
    /// Get WebSocket circuit breaker
    pub fn ws_cb(&self) -> &CircuitBreaker {
        &self.ws_circuit_breaker
    }
    
    /// Get API key rotator
    pub fn key_rotator(&self) -> &ApiKeyRotator {
        &self.key_rotator
    }
    
    /// Get health monitor
    pub fn health_monitor(&self) -> &ConnectionHealthMonitor {
        &self.health_monitor
    }
    
    /// Record overall system health
    pub fn record_heartbeat(&self) {
        self.health_monitor.record_heartbeat();
    }
    
    /// Check if system is healthy for trading
    pub fn is_trading_allowed(&self) -> bool {
        self.rest_circuit_breaker.allow_request()
            && self.health_monitor.get_health() != ConnectionHealth::Disconnected
    }
}

impl Default for ResilienceManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_circuit_breaker() {
        let cb = CircuitBreaker::new(CircuitBreakerConfig {
            failure_threshold: 3,
            ..Default::default()
        });
        
        assert_eq!(cb.state(), CircuitState::Closed);
        assert!(cb.allow_request());
        
        // Trip the breaker
        for _ in 0..3 {
            cb.record_failure();
        }
        
        assert_eq!(cb.state(), CircuitState::Open);
        assert!(!cb.allow_request());
        
        // Recovery
        cb.force_close();
        assert_eq!(cb.state(), CircuitState::Closed);
    }
    
    #[test]
    fn test_api_key_rotator() {
        let rotator = ApiKeyRotator::new();
        
        rotator.add_key("key1", "api1", "secret1");
        rotator.add_key("key2", "api2", "secret2");
        
        assert_eq!(rotator.key_count(), 2);
        
        let (id, _, _) = rotator.get_active_key().unwrap();
        assert_eq!(id, "key1");
        
        // Rotate
        rotator.rotate_key();
        let (id, _, _) = rotator.get_active_key().unwrap();
        assert_eq!(id, "key2");
    }
    
    #[test]
    fn test_connection_health_monitor() {
        let monitor = ConnectionHealthMonitor::new(Duration::from_millis(100), 3);
        
        assert_eq!(monitor.get_health(), ConnectionHealth::Healthy);
        
        // Simulate failures
        for _ in 0..3 {
            monitor.record_failure();
        }
        
        assert_eq!(monitor.get_health(), ConnectionHealth::Disconnected);
        assert!(monitor.needs_reconnect());
        
        // Recover
        monitor.record_heartbeat();
        assert_eq!(monitor.get_health(), ConnectionHealth::Healthy);
    }
    
    #[test]
    fn test_resilience_manager() {
        let manager = ResilienceManager::new();
        
        assert!(manager.is_trading_allowed());
        
        manager.record_heartbeat();
        assert_eq!(manager.health_monitor().get_health(), ConnectionHealth::Healthy);
    }
}
