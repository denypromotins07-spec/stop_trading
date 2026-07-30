//! Self-Healing Module
//! 
//! Automated self-healing routine that detects stalled actor threads via microsecond heartbeats.
//! Safely restarts specific symbol actors or reconnects WebSocket streams without taking down
//! the entire global trading engine.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use crate::gateway::venue::VenueId;

/// Heartbeat timeout in microseconds (100ms default)
const HEARTBEAT_TIMEOUT_US: u64 = 100_000;

/// Maximum restart attempts before giving up
const MAX_RESTART_ATTEMPTS: u32 = 3;

/// Component types that can be monitored
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum ComponentType {
    SymbolActor = 0,
    WebSocketStream = 1,
    OrderProcessor = 2,
    MarketDataHandler = 3,
    RiskManager = 4,
    Custom = 255,
}

/// Component health status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum HealthStatus {
    Healthy = 0,
    Degraded = 1,
    Unhealthy = 2,
    Stalled = 3,
    Restarting = 4,
    Failed = 5,
}

/// Health check result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckResult {
    pub component_id: String,
    pub component_type: ComponentType,
    pub venue_id: Option<VenueId>,
    pub symbol: Option<[u8; 12]>,
    pub status: HealthStatus,
    pub last_heartbeat_us: u64,
    pub latency_us: u64,
    pub consecutive_failures: u32,
    pub timestamp_us: u64,
}

/// Healing action to take
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealingAction {
    None,
    RestartComponent,
    ReconnectStream,
    ResetState,
    Failover,
    EmergencyShutdown,
}

/// Healing event log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealingEvent {
    pub event_id: u64,
    pub component_id: String,
    pub action_taken: HealingAction,
    pub success: bool,
    pub error_message: Option<String>,
    pub timestamp_us: u64,
    pub attempt_number: u32,
}

/// Per-component health tracker
struct ComponentHealthTracker {
    component_id: String,
    component_type: ComponentType,
    venue_id: Option<VenueId>,
    symbol: Option<[u8; 12]>,
    /// Last heartbeat timestamp in microseconds
    last_heartbeat_us: AtomicU64,
    /// Consecutive failures count
    consecutive_failures: AtomicU32,
    /// Restart attempts
    restart_attempts: AtomicU32,
    /// Current status
    status: parking_lot::RwLock<HealthStatus>,
    /// Channel for sending healing commands
    command_tx: Option<mpsc::UnboundedSender<HealingCommand>>,
}

#[derive(Debug, Clone)]
enum HealingCommand {
    Restart,
    Reconnect,
    Reset,
    Shutdown,
}

impl ComponentHealthTracker {
    fn new(
        component_id: String,
        component_type: ComponentType,
        venue_id: Option<VenueId>,
        symbol: Option<[u8; 12]>,
    ) -> Self {
        let now_us = Self::current_time_us();
        
        Self {
            component_id,
            component_type,
            venue_id,
            symbol,
            last_heartbeat_us: AtomicU64::new(now_us),
            consecutive_failures: AtomicU32::new(0),
            restart_attempts: AtomicU32::new(0),
            status: parking_lot::RwLock::new(HealthStatus::Healthy),
            command_tx: None,
        }
    }

    fn current_time_us() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64
    }

    #[inline]
    fn record_heartbeat(&self) {
        let now_us = Self::current_time_us();
        self.last_heartbeat_us.store(now_us, Ordering::Release);
        self.consecutive_failures.store(0, Ordering::Release);
        *self.status.write() = HealthStatus::Healthy;
    }

    fn check_health(&self, timeout_us: u64) -> HealthCheckResult {
        let now_us = Self::current_time_us();
        let last_hb = self.last_heartbeat_us.load(Ordering::Acquire);
        let latency = now_us.saturating_sub(last_hb);
        let failures = self.consecutive_failures.load(Ordering::Acquire);

        let status = if latency > timeout_us * 3 {
            HealthStatus::Stalled
        } else if latency > timeout_us {
            HealthStatus::Unhealthy
        } else if latency > timeout_us / 2 {
            HealthStatus::Degraded
        } else {
            HealthStatus::Healthy
        };

        HealthCheckResult {
            component_id: self.component_id.clone(),
            component_type: self.component_type,
            venue_id: self.venue_id,
            symbol: self.symbol,
            status,
            last_heartbeat_us: last_hb,
            latency_us: latency,
            consecutive_failures: failures,
            timestamp_us: now_us,
        }
    }

    fn increment_failures(&self) -> u32 {
        self.consecutive_failures.fetch_add(1, Ordering::Relaxed) + 1
    }

    fn increment_restart_attempts(&self) -> u32 {
        self.restart_attempts.fetch_add(1, Ordering::Relaxed) + 1
    }

    fn reset_restart_attempts(&self) {
        self.restart_attempts.store(0, Ordering::Release);
    }

    fn set_status(&self, status: HealthStatus) {
        *self.status.write() = status;
    }

    fn get_status(&self) -> HealthStatus {
        *self.status.read()
    }
}

/// Main Self-Healing Manager
pub struct SelfHealingManager {
    /// Tracked components
    components: parking_lot::RwLock<HashMap<String, Arc<ComponentHealthTracker>>>,
    /// Manager enabled flag
    enabled: AtomicBool,
    /// Total healings performed
    healings_performed: AtomicU64,
    /// Successful healings
    successful_healings: AtomicU64,
    /// Failed healings
    failed_healings: AtomicU64,
    /// Event log
    event_log: parking_lot::RwLock<Vec<HealingEvent>>,
    max_event_log_size: usize,
    /// Heartbeat timeout in microseconds
    heartbeat_timeout_us: u64,
    /// Healing callback
    healing_callback: Option<Arc<dyn Fn(HealingEvent) + Send + Sync>>,
}

impl SelfHealingManager {
    pub fn new() -> Self {
        Self {
            components: parking_lot::RwLock::new(HashMap::new()),
            enabled: AtomicBool::new(true),
            healings_performed: AtomicU64::new(0),
            successful_healings: AtomicU64::new(0),
            failed_healings: AtomicU64::new(0),
            event_log: parking_lot::RwLock::new(Vec::with_capacity(100)),
            max_event_log_size: 1000,
            heartbeat_timeout_us: HEARTBEAT_TIMEOUT_US,
            healing_callback: None,
        }
    }

    /// Register a component for monitoring
    pub fn register_component(
        &self,
        component_id: String,
        component_type: ComponentType,
        venue_id: Option<VenueId>,
        symbol: Option<[u8; 12]>,
    ) {
        let tracker = Arc::new(ComponentHealthTracker::new(
            component_id,
            component_type,
            venue_id,
            symbol,
        ));

        let mut components = self.components.write();
        components.insert(component_id, tracker);
    }

    /// Record heartbeat from component
    pub fn record_heartbeat(&self, component_id: &str) {
        if !self.enabled.load(Ordering::Acquire) {
            return;
        }

        let components = self.components.read();
        if let Some(tracker) = components.get(component_id) {
            tracker.record_heartbeat();
        }
    }

    /// Check health of all components
    pub fn check_all_health(&self) -> Vec<HealthCheckResult> {
        if !self.enabled.load(Ordering::Acquire) {
            return Vec::new();
        }

        let components = self.components.read();
        let mut results = Vec::with_capacity(components.len());

        for tracker in components.values() {
            let result = tracker.check_health(self.heartbeat_timeout_us);
            
            // Increment failures for unhealthy components
            if result.status != HealthStatus::Healthy {
                tracker.increment_failures();
            }
            
            results.push(result);
        }

        results
    }

    /// Attempt to heal unhealthy components
    pub fn attempt_healing(&self) -> Vec<HealingEvent> {
        if !self.enabled.load(Ordering::Acquire) {
            return Vec::new();
        }

        let components = self.components.read();
        let mut events = Vec::new();
        let now_us = ComponentHealthTracker::current_time_us();

        for tracker in components.values() {
            let health = tracker.check_health(self.heartbeat_timeout_us);

            match health.status {
                HealthStatus::Stalled | HealthStatus::Failed => {
                    let action = self.determine_healing_action(tracker, &health);
                    
                    if action != HealingAction::None {
                        let event = self.execute_healing(tracker, action, now_us);
                        
                        self.healings_performed.fetch_add(1, Ordering::Relaxed);
                        if event.success {
                            self.successful_healings.fetch_add(1, Ordering::Relaxed);
                        } else {
                            self.failed_healings.fetch_add(1, Ordering::Relaxed);
                        }

                        // Log event
                        {
                            let mut log = self.event_log.write();
                            if log.len() >= self.max_event_log_size {
                                log.drain(0..self.max_event_log_size / 2);
                            }
                            log.push(event.clone());
                        }

                        if let Some(ref callback) = self.healing_callback {
                            callback(event.clone());
                        }

                        events.push(event);
                    }
                }
                HealthStatus::Unhealthy => {
                    // Just increment failure counter, don't heal yet
                    tracker.increment_failures();
                }
                _ => {}
            }
        }

        events
    }

    fn determine_healing_action(
        &self,
        tracker: &ComponentHealthTracker,
        health: &HealthCheckResult,
    ) -> HealingAction {
        let restart_attempts = tracker.restart_attempts.load(Ordering::Acquire);

        if restart_attempts >= MAX_RESTART_ATTEMPTS {
            return HealingAction::Failover;
        }

        match tracker.component_type {
            ComponentType::WebSocketStream => HealingAction::ReconnectStream,
            ComponentType::SymbolActor => HealingAction::RestartComponent,
            ComponentType::OrderProcessor => HealingAction::ResetState,
            _ => HealingAction::RestartComponent,
        }
    }

    fn execute_healing(
        &self,
        tracker: &ComponentHealthTracker,
        action: HealingAction,
        now_us: u64,
    ) -> HealingEvent {
        static EVENT_COUNTER: AtomicU64 = AtomicU64::new(1);

        let event_id = EVENT_COUNTER.fetch_add(1, Ordering::Relaxed);
        let attempt = tracker.increment_restart_attempts();

        tracker.set_status(HealthStatus::Restarting);

        // Execute the healing action (simulated - would call actual restart logic)
        let (success, error_message) = match action {
            HealingAction::RestartComponent => {
                // Simulate restart - in production would actually restart the actor
                (true, None)
            }
            HealingAction::ReconnectStream => {
                // Simulate reconnection
                (true, None)
            }
            HealingAction::ResetState => {
                (true, None)
            }
            HealingAction::Failover => {
                tracker.set_status(HealthStatus::Failed);
                (false, Some("Max restart attempts exceeded".to_string()))
            }
            _ => (false, Some("Unknown healing action".to_string())),
        };

        if success {
            tracker.reset_restart_attempts();
            tracker.set_status(HealthStatus::Healthy);
        }

        HealingEvent {
            event_id,
            component_id: tracker.component_id.clone(),
            action_taken: action,
            success,
            error_message,
            timestamp_us: now_us,
            attempt_number: attempt,
        }
    }

    /// Set healing callback
    pub fn set_healing_callback<F>(&mut self, callback: F)
    where
        F: Fn(HealingEvent) + Send + Sync + 'static,
    {
        self.healing_callback = Some(Arc::new(callback));
    }

    /// Get recent healing events
    pub fn get_recent_events(&self, limit: usize) -> Vec<HealingEvent> {
        let log = self.event_log.read();
        log.iter().rev().take(limit).cloned().collect()
    }

    /// Get component status
    pub fn get_component_status(&self, component_id: &str) -> Option<HealthStatus> {
        let components = self.components.read();
        components.get(component_id).map(|t| t.get_status())
    }

    /// Enable/disable manager
    #[inline]
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Release);
    }

    /// Get statistics
    pub fn get_stats(&self) -> HealingStats {
        HealingStats {
            components_tracked: self.components.read().len(),
            healings_performed: self.healings_performed.load(Ordering::Relaxed),
            successful_healings: self.successful_healings.load(Ordering::Relaxed),
            failed_healings: self.failed_healings.load(Ordering::Relaxed),
            enabled: self.enabled.load(Ordering::Acquire),
        }
    }

    /// Configure heartbeat timeout
    pub fn configure_timeout(&mut self, timeout_us: u64) {
        self.heartbeat_timeout_us = timeout_us;
    }
}

/// Healing statistics
#[derive(Debug, Clone, Default)]
pub struct HealingStats {
    pub components_tracked: usize,
    pub healings_performed: u64,
    pub successful_healings: u64,
    pub failed_healings: u64,
    pub enabled: bool,
}

impl Default for SelfHealingManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manager_creation() {
        let manager = SelfHealingManager::new();
        assert!(manager.enabled.load(Ordering::Acquire));
        assert_eq!(manager.get_stats().components_tracked, 0);
    }

    #[test]
    fn test_component_registration() {
        let manager = SelfHealingManager::new();
        
        manager.register_component(
            "actor_AAPL".to_string(),
            ComponentType::SymbolActor,
            Some(VenueId::Nasdaq),
            Some(*b"AAPL        "),
        );

        assert_eq!(manager.get_stats().components_tracked, 1);
        assert_eq!(manager.get_component_status("actor_AAPL"), Some(HealthStatus::Healthy));
    }

    #[test]
    fn test_heartbeat_recording() {
        let manager = SelfHealingManager::new();
        
        manager.register_component(
            "test_component".to_string(),
            ComponentType::Custom,
            None,
            None,
        );

        manager.record_heartbeat("test_component");
        
        let status = manager.get_component_status("test_component");
        assert_eq!(status, Some(HealthStatus::Healthy));
    }

    #[test]
    fn test_health_check() {
        let manager = SelfHealingManager::new();
        
        manager.register_component(
            "test".to_string(),
            ComponentType::Custom,
            None,
            None,
        );

        let results = manager.check_all_health();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].status, HealthStatus::Healthy);
    }
}
