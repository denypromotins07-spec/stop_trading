//! Chaos Engineering - Network Fault Injection Module
//! 
//! Builds a fault injection engine simulating packet loss, latency spikes, and WebSocket disconnects.
//! Stress-tests reconnection and sequence gap recovery logic in shadow mode without risking capital.

use alloc::vec::Vec;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Maximum fault configurations that can be active
pub const MAX_FAULT_CONFIGS: usize = 32;

/// Type of network fault to inject
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FaultType {
    /// Drop packets with specified probability
    PacketLoss,
    /// Add latency to packets
    LatencySpike,
    /// Disconnect WebSocket connection
    WebSocketDisconnect,
    /// Corrupt packet payload
    PayloadCorruption,
    /// Reorder packets
    PacketReordering,
    /// Duplicate packets
    PacketDuplication,
    /// Throttle bandwidth
    BandwidthThrottle,
}

/// Network fault configuration
#[derive(Debug, Clone)]
pub struct FaultConfig {
    pub fault_type: FaultType,
    /// Probability of fault occurrence (0.0 to 1.0)
    pub probability: f64,
    /// Duration of fault in milliseconds (for continuous faults)
    pub duration_ms: u64,
    /// Severity level (1-10)
    pub severity: u8,
    /// Target connection IDs (empty = all connections)
    pub target_connections: Vec<u64>,
    /// Enable in shadow mode only
    pub shadow_only: bool,
}

impl Default for FaultConfig {
    fn default() -> Self {
        FaultConfig {
            fault_type: FaultType::PacketLoss,
            probability: 0.01,
            duration_ms: 100,
            severity: 1,
            target_connections: Vec::new(),
            shadow_only: true,
        }
    }
}

/// Network fault injector engine
pub struct NetworkFaultInjector {
    configs: Vec<FaultConfig>,
    is_active: AtomicBool,
    shadow_mode: AtomicBool,
    faults_injected: AtomicU64,
    packets_processed: AtomicU64,
    last_fault_ts: AtomicU64,
    rng_state: AtomicU64,
}

impl NetworkFaultInjector {
    pub fn new() -> Self {
        NetworkFaultInjector {
            configs: Vec::with_capacity(MAX_FAULT_CONFIGS),
            is_active: AtomicBool::new(false),
            shadow_mode: AtomicBool::new(true),
            faults_injected: AtomicU64::new(0),
            packets_processed: AtomicU64::new(0),
            last_fault_ts: AtomicU64::new(0),
            rng_state: AtomicU64::new(0x5DEECE66D),
        }
    }

    /// Add a fault configuration
    pub fn add_fault(&mut self, config: FaultConfig) -> Result<(), ChaosError> {
        if self.configs.len() >= MAX_FAULT_CONFIGS {
            return Err(ChaosError::MaxFaultsReached);
        }

        // Validate probability
        if config.probability < 0.0 || config.probability > 1.0 {
            return Err(ChaosError::InvalidProbability);
        }

        self.configs.push(config);
        Ok(())
    }

    /// Remove fault by type
    pub fn remove_fault(&mut self, fault_type: FaultType) {
        self.configs.retain(|c| c.fault_type != fault_type);
    }

    /// Clear all faults
    pub fn clear_faults(&mut self) {
        self.configs.clear();
    }

    /// Check if a packet should be affected by any fault
    pub fn should_fault_packet(&self, connection_id: u64, timestamp_ns: u64) -> Option<FaultEvent> {
        if !self.is_active.load(Ordering::Acquire) {
            return None;
        }

        // Skip if not in shadow mode and fault is shadow-only
        let in_shadow = self.shadow_mode.load(Ordering::Acquire);
        
        self.packets_processed.fetch_add(1, Ordering::Relaxed);

        for config in &self.configs {
            // Check shadow-only restriction
            if config.shadow_only && !in_shadow {
                continue;
            }

            // Check target connection filter
            if !config.target_connections.is_empty() 
                && !config.target_connections.contains(&connection_id) 
            {
                continue;
            }

            // Check probability using simple LCG
            let rand_val = self.next_random();
            let probability_check = (rand_val as f64) / (u64::MAX as f64);

            if probability_check < config.probability {
                self.faults_injected.fetch_add(1, Ordering::Relaxed);
                self.last_fault_ts.store(timestamp_ns, Ordering::Release);

                return Some(FaultEvent {
                    fault_type: config.fault_type,
                    severity: config.severity,
                    timestamp_ns,
                    connection_id,
                });
            }
        }

        None
    }

    /// Simulate latency spike
    pub fn get_latency_spike(&self) -> Option<u64> {
        if !self.is_active.load(Ordering::Acquire) {
            return None;
        }

        for config in &self.configs {
            if config.fault_type == FaultType::LatencySpike {
                let rand_val = self.next_random();
                let probability_check = (rand_val as f64) / (u64::MAX as f64);

                if probability_check < config.probability {
                    // Calculate latency based on severity
                    let base_latency = 10; // 10ms base
                    let multiplier = config.severity as u64 * 10;
                    let random_component = rand_val % (multiplier * 10);
                    return Some(base_latency + random_component / 10);
                }
            }
        }

        None
    }

    /// Simulate WebSocket disconnect decision
    pub fn should_disconnect_ws(&self, connection_id: u64) -> bool {
        if !self.is_active.load(Ordering::Acquire) {
            return false;
        }

        for config in &self.configs {
            if config.fault_type == FaultType::WebSocketDisconnect {
                if !config.target_connections.is_empty() 
                    && !config.target_connections.contains(&connection_id) 
                {
                    continue;
                }

                let rand_val = self.next_random();
                let probability_check = (rand_val as f64) / (u64::MAX as f64);

                if probability_check < config.probability {
                    self.faults_injected.fetch_add(1, Ordering::Relaxed);
                    return true;
                }
            }
        }

        false
    }

    /// Get statistics about fault injection
    pub fn get_stats(&self) -> FaultStats {
        FaultStats {
            is_active: self.is_active.load(Ordering::Acquire),
            shadow_mode: self.shadow_mode.load(Ordering::Acquire),
            active_configs: self.configs.len(),
            faults_injected: self.faults_injected.load(Ordering::Relaxed),
            packets_processed: self.packets_processed.load(Ordering::Relaxed),
            fault_rate: self.calculate_fault_rate(),
            last_fault_ts: self.last_fault_ts.load(Ordering::Relaxed),
        }
    }

    /// Enable/disable fault injection
    pub fn set_active(&self, active: bool) {
        self.is_active.store(active, Ordering::Release);
    }

    /// Set shadow mode
    pub fn set_shadow_mode(&self, shadow: bool) {
        self.shadow_mode.store(shadow, Ordering::Release);
    }

    /// Check if in shadow mode
    pub fn is_shadow_mode(&self) -> bool {
        self.shadow_mode.load(Ordering::Acquire)
    }

    /// Simple LCG random number generator
    fn next_random(&self) -> u64 {
        let state = self.rng_state.load(Ordering::Relaxed);
        let new_state = state.wrapping_mul(0x5DEECE66D).wrapping_add(0xB);
        self.rng_state.store(new_state, Ordering::Relaxed);
        new_state
    }

    fn calculate_fault_rate(&self) -> f64 {
        let packets = self.packets_processed.load(Ordering::Relaxed);
        let faults = self.faults_injected.load(Ordering::Relaxed);
        
        if packets == 0 {
            return 0.0;
        }
        
        faults as f64 / packets as f64
    }
}

impl Default for NetworkFaultInjector {
    fn default() -> Self {
        Self::new()
    }
}

/// Fault event generated by the injector
#[derive(Debug, Clone)]
pub struct FaultEvent {
    pub fault_type: FaultType,
    pub severity: u8,
    pub timestamp_ns: u64,
    pub connection_id: u64,
}

/// Fault injection statistics
#[derive(Debug, Clone)]
pub struct FaultStats {
    pub is_active: bool,
    pub shadow_mode: bool,
    pub active_configs: usize,
    pub faults_injected: u64,
    pub packets_processed: u64,
    pub fault_rate: f64,
    pub last_fault_ts: u64,
}

/// Chaos error types
#[derive(Debug, Clone, PartialEq)]
pub enum ChaosError {
    MaxFaultsReached,
    InvalidProbability,
    InvalidConfiguration,
    NotInShadowMode,
    SystemNotReady,
}

/// Connection simulator for testing reconnection logic
pub struct ConnectionSimulator {
    injector: NetworkFaultInjector,
    connected: AtomicBool,
    reconnect_attempts: AtomicU64,
    successful_reconnects: AtomicU64,
    failed_reconnects: AtomicU64,
}

impl ConnectionSimulator {
    pub fn new(injector: NetworkFaultInjector) -> Self {
        ConnectionSimulator {
            injector,
            connected: AtomicBool::new(true),
            reconnect_attempts: AtomicU64::new(0),
            successful_reconnects: AtomicU64::new(0),
            failed_reconnects: AtomicU64::new(0),
        }
    }

    /// Simulate sending a message with potential fault injection
    pub fn send_message(&self, connection_id: u64, _data: &[u8], timestamp_ns: u64) -> SendResult {
        if !self.connected.load(Ordering::Acquire) {
            return SendResult::NotConnected;
        }

        // Check for WebSocket disconnect
        if self.injector.should_disconnect_ws(connection_id) {
            self.connected.store(false, Ordering::Release);
            return SendResult::Disconnected;
        }

        // Check for packet-level faults
        if let Some(event) = self.injector.should_fault_packet(connection_id, timestamp_ns) {
            match event.fault_type {
                FaultType::PacketLoss => SendResult::Dropped,
                FaultType::PayloadCorruption => SendResult::Corrupted,
                FaultType::LatencySpike => {
                    if let Some(latency) = self.injector.get_latency_spike() {
                        SendResult::Delayed(latency)
                    } else {
                        SendResult::Sent
                    }
                }
                _ => SendResult::Sent,
            }
        } else {
            SendResult::Sent
        }
    }

    /// Attempt reconnection
    pub fn attempt_reconnect(&self) -> bool {
        self.reconnect_attempts.fetch_add(1, Ordering::Relaxed);

        // Simulate reconnection success (in real impl, would test actual reconnect logic)
        let success = true; // Simplified
        
        if success {
            self.connected.store(true, Ordering::Release);
            self.successful_reconnects.fetch_add(1, Ordering::Relaxed);
        } else {
            self.failed_reconnects.fetch_add(1, Ordering::Relaxed);
        }

        success
    }

    /// Get connection statistics
    pub fn get_stats(&self) -> ConnectionStats {
        ConnectionStats {
            is_connected: self.connected.load(Ordering::Acquire),
            reconnect_attempts: self.reconnect_attempts.load(Ordering::Relaxed),
            successful_reconnects: self.successful_reconnects.load(Ordering::Relaxed),
            failed_reconnects: self.failed_reconnects.load(Ordering::Relaxed),
            injector_stats: self.injector.get_stats(),
        }
    }

    /// Force disconnect for testing
    pub fn force_disconnect(&self) {
        self.connected.store(false, Ordering::Release);
    }

    /// Force connect for testing
    pub fn force_connect(&self) {
        self.connected.store(true, Ordering::Release);
    }
}

/// Send result after fault injection
#[derive(Debug, Clone, PartialEq)]
pub enum SendResult {
    Sent,
    Dropped,
    Corrupted,
    Delayed(u64), // latency in ms
    Disconnected,
    NotConnected,
}

/// Connection statistics
#[derive(Debug, Clone)]
pub struct ConnectionStats {
    pub is_connected: bool,
    pub reconnect_attempts: u64,
    pub successful_reconnects: u64,
    pub failed_reconnects: u64,
    pub injector_stats: FaultStats,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fault_injector_basic() {
        let mut injector = NetworkFaultInjector::new();
        
        let config = FaultConfig {
            fault_type: FaultType::PacketLoss,
            probability: 0.5, // 50% loss rate for testing
            ..Default::default()
        };
        
        assert!(injector.add_fault(config).is_ok());
        
        injector.set_active(true);
        injector.set_shadow_mode(true);
        
        let stats = injector.get_stats();
        assert!(stats.is_active);
        assert_eq!(stats.active_configs, 1);
    }

    #[test]
    fn test_connection_simulator() {
        let injector = NetworkFaultInjector::new();
        let simulator = ConnectionSimulator::new(injector);
        
        assert!(simulator.get_stats().is_connected);
        
        simulator.force_disconnect();
        assert!(!simulator.get_stats().is_connected);
        
        simulator.attempt_reconnect();
        assert!(simulator.get_stats().is_connected);
    }

    #[test]
    fn test_send_with_faults() {
        let mut injector = NetworkFaultInjector::new();
        
        // Add high probability packet loss
        let config = FaultConfig {
            fault_type: FaultType::PacketLoss,
            probability: 1.0, // Always drop
            ..Default::default()
        };
        injector.add_fault(config).unwrap();
        injector.set_active(true);
        
        let simulator = ConnectionSimulator::new(injector);
        
        let result = simulator.send_message(1, b"test", 1_000_000_000);
        assert_eq!(result, SendResult::Dropped);
    }
}
