//! WebSocket Reconnection and Failover Mechanism
//! 
//! Builds a robust failover and reconnection mechanism using exponential backoff and jitter.
//! Implements sequence gap detection to automatically trigger REST API snapshots when 
//! WebSocket stream sequences break.

use std::time::Duration;
use rand::Rng;
use crate::market_data::{OrderBookDelta, OrderBookSnapshot};

/// Reconnection state machine
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReconnectState {
    /// Not currently reconnecting
    Idle,
    /// Waiting before next attempt
    Waiting,
    /// Currently connecting
    Connecting,
    /// Failed permanently
    Failed,
}

/// Exponential backoff configuration
#[derive(Debug, Clone)]
pub struct BackoffConfig {
    /// Initial delay in milliseconds
    pub initial_delay_ms: u64,
    /// Maximum delay in milliseconds
    pub max_delay_ms: u64,
    /// Multiplier for each retry
    pub multiplier: f64,
    /// Jitter factor (0.0 - 1.0)
    pub jitter: f64,
    /// Maximum number of retries (0 = infinite)
    pub max_retries: u32,
}

impl Default for BackoffConfig {
    fn default() -> Self {
        BackoffConfig {
            initial_delay_ms: 1_000,      // 1 second
            max_delay_ms: 60_000,         // 60 seconds
            multiplier: 2.0,              // Double each time
            jitter: 0.1,                  // 10% jitter
            max_retries: 0,               // Infinite
        }
    }
}

/// Backoff calculator for reconnection delays
pub struct BackoffCalculator {
    config: BackoffConfig,
    current_attempt: u32,
    current_delay_ms: u64,
}

impl BackoffCalculator {
    #[inline]
    pub fn new(config: BackoffConfig) -> Self {
        BackoffCalculator {
            config,
            current_attempt: 0,
            current_delay_ms: config.initial_delay_ms,
        }
    }

    /// Get the next delay with jitter
    #[inline]
    pub fn next_delay(&mut self) -> Duration {
        if self.config.max_retries > 0 && self.current_attempt >= self.config.max_retries {
            return Duration::from_secs(3600); // 1 hour cooldown after max retries
        }

        let mut rng = rand::thread_rng();
        
        // Calculate base delay
        let base_delay = self.current_delay_ms as f64;
        
        // Apply jitter
        let jitter_range = base_delay * self.config.jitter;
        let jitter_offset = rng.gen_range(-jitter_range..=jitter_range);
        let delayed_ms = (base_delay + jitter_offset) as u64;
        
        // Clamp to max delay
        let final_delay_ms = delayed_ms.min(self.config.max_delay_ms);
        
        // Update for next iteration
        self.current_attempt += 1;
        self.current_delay_ms = (base_delay * self.config.multiplier) as u64;
        
        Duration::from_millis(final_delay_ms)
    }

    /// Reset the backoff calculator
    #[inline]
    pub fn reset(&mut self) {
        self.current_attempt = 0;
        self.current_delay_ms = self.config.initial_delay_ms;
    }

    /// Get current attempt count
    #[inline]
    pub fn attempt_count(&self) -> u32 {
        self.current_attempt
    }

    /// Check if max retries exceeded
    #[inline]
    pub fn is_exhausted(&self) -> bool {
        self.config.max_retries > 0 && self.current_attempt >= self.config.max_retries
    }
}

/// Sequence tracker for gap detection
#[derive(Debug, Clone)]
pub struct SequenceTracker {
    /// Last seen sequence number
    last_sequence: u64,
    /// Expected next sequence
    expected_next: u64,
    /// Number of gaps detected
    gap_count: u32,
    /// Whether we need a snapshot refresh
    needs_snapshot: bool,
    /// First update ID from last snapshot
    snapshot_first_id: u64,
    /// Last update ID from last snapshot
    snapshot_last_id: u64,
}

impl SequenceTracker {
    #[inline]
    pub fn new() -> Self {
        SequenceTracker {
            last_sequence: 0,
            expected_next: 0,
            gap_count: 0,
            needs_snapshot: false,
            snapshot_first_id: 0,
            snapshot_last_id: 0,
        }
    }

    /// Initialize with a snapshot
    #[inline]
    pub fn init_from_snapshot(&mut self, first_id: u64, last_id: u64) {
        self.snapshot_first_id = first_id;
        self.snapshot_last_id = last_id;
        self.expected_next = last_id + 1;
        self.last_sequence = last_id;
        self.needs_snapshot = false;
    }

    /// Process an update and check for gaps
    /// 
    /// Returns true if a gap was detected and snapshot refresh is needed
    #[inline]
    pub fn process_update(&mut self, first_id: u64, last_id: u64) -> bool {
        // First update after initialization
        if self.expected_next == 0 {
            self.last_sequence = last_id;
            self.expected_next = last_id + 1;
            return false;
        }

        // Check for gap
        if first_id != self.expected_next {
            self.gap_count += 1;
            self.needs_snapshot = true;
            log::warn!(
                "Sequence gap detected: expected {}, got {}-{}",
                self.expected_next,
                first_id,
                last_id
            );
            return true;
        }

        // Check for overlap (duplicate or old update)
        if last_id < self.expected_next {
            log::debug!("Received overlapping update: {}-{} (expected {})", 
                       first_id, last_id, self.expected_next);
            return false;
        }

        // Normal case - update sequence
        self.last_sequence = last_id;
        self.expected_next = last_id + 1;
        false
    }

    /// Mark that a snapshot has been received
    #[inline]
    pub fn snapshot_received(&mut self, first_id: u64, last_id: u64) {
        self.init_from_snapshot(first_id, last_id);
    }

    /// Check if snapshot refresh is needed
    #[inline]
    pub fn needs_snapshot_refresh(&self) -> bool {
        self.needs_snapshot
    }

    /// Clear the needs_snapshot flag
    #[inline]
    pub fn clear_snapshot_flag(&mut self) {
        self.needs_snapshot = false;
    }

    /// Get gap count
    #[inline]
    pub fn gap_count(&self) -> u32 {
        self.gap_count
    }

    /// Get last sequence number
    #[inline]
    pub fn last_sequence(&self) -> u64 {
        self.last_sequence
    }

    /// Reset the tracker
    #[inline]
    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

impl Default for SequenceTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Reconnection manager combining backoff and sequence tracking
pub struct ReconnectionManager {
    state: ReconnectState,
    backoff: BackoffCalculator,
    sequence_tracker: SequenceTracker,
    /// Consecutive successful updates
    consecutive_successes: u32,
    /// Threshold to consider connection stable
    stability_threshold: u32,
}

impl ReconnectionManager {
    #[inline]
    pub fn new() -> Self {
        ReconnectionManager {
            state: ReconnectState::Idle,
            backoff: BackoffCalculator::new(BackoffConfig::default()),
            sequence_tracker: SequenceTracker::new(),
            consecutive_successes: 0,
            stability_threshold: 100,
        }
    }

    /// Called when connection is lost
    #[inline]
    pub fn on_disconnect(&mut self) -> Duration {
        self.state = ReconnectState::Waiting;
        let delay = self.backoff.next_delay();
        log::info!("Scheduling reconnection in {:?}", delay);
        delay
    }

    /// Called when connection attempt starts
    #[inline]
    pub fn on_connecting(&mut self) {
        self.state = ReconnectState::Connecting;
    }

    /// Called when connection is established
    #[inline]
    pub fn on_connected(&mut self) {
        self.state = ReconnectState::Idle;
        self.consecutive_successes = 0;
        log::info!("Connection established");
    }

    /// Called on successful message processing
    #[inline]
    pub fn on_success(&mut self) {
        self.consecutive_successes += 1;
        
        // If we've had enough successes, reset backoff
        if self.consecutive_successes >= self.stability_threshold {
            self.backoff.reset();
            log::debug!("Connection stable, backoff reset");
        }
    }

    /// Called when a gap is detected
    #[inline]
    pub fn on_gap_detected(&mut self) -> bool {
        self.sequence_tracker.process_update(0, 0); // Just increment gap count
        self.sequence_tracker.needs_snapshot_refresh()
    }

    /// Process a delta update
    #[inline]
    pub fn process_delta(&mut self, delta: &OrderBookDelta) -> bool {
        // For delta updates, we'd typically extract first/last IDs
        // This is a simplified version
        let needs_snapshot = self.sequence_tracker.process_update(delta.sequence, delta.sequence);
        
        if !needs_snapshot {
            self.on_success();
        }
        
        needs_snapshot
    }

    /// Initialize from a snapshot
    #[inline]
    pub fn init_from_snapshot(&mut self, snapshot: &OrderBookSnapshot) {
        self.sequence_tracker.snapshot_received(
            snapshot.last_update_id,
            snapshot.last_update_id,
        );
    }

    /// Get current state
    #[inline]
    pub fn state(&self) -> ReconnectState {
        self.state
    }

    /// Check if reconnection is needed
    #[inline]
    pub fn needs_reconnect(&self) -> bool {
        matches!(self.state, ReconnectState::Waiting | ReconnectState::Connecting)
    }

    /// Check if snapshot refresh is needed
    #[inline]
    pub fn needs_snapshot(&self) -> bool {
        self.sequence_tracker.needs_snapshot_refresh()
    }

    /// Get gap statistics
    #[inline]
    pub fn gap_stats(&self) -> (u32, u64) {
        (
            self.sequence_tracker.gap_count(),
            self.sequence_tracker.last_sequence(),
        )
    }

    /// Reset everything
    #[inline]
    pub fn reset(&mut self) {
        self.state = ReconnectState::Idle;
        self.backoff.reset();
        self.sequence_tracker.reset();
        self.consecutive_successes = 0;
    }
}

impl Default for ReconnectionManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backoff_calculator() {
        let config = BackoffConfig {
            initial_delay_ms: 1000,
            max_delay_ms: 10000,
            multiplier: 2.0,
            jitter: 0.0,
            max_retries: 0,
        };
        
        let mut calc = BackoffCalculator::new(config);
        
        let delay1 = calc.next_delay();
        assert_eq!(delay1.as_millis(), 1000);
        
        let delay2 = calc.next_delay();
        assert_eq!(delay2.as_millis(), 2000);
        
        let delay3 = calc.next_delay();
        assert_eq!(delay3.as_millis(), 4000);
    }

    #[test]
    fn test_sequence_tracker_gap_detection() {
        let mut tracker = SequenceTracker::new();
        
        // Initialize
        tracker.init_from_snapshot(1, 100);
        
        // Normal update
        assert!(!tracker.process_update(101, 102));
        
        // Gap detected
        assert!(tracker.process_update(105, 106));
        assert!(tracker.needs_snapshot_refresh());
        assert_eq!(tracker.gap_count(), 1);
    }

    #[test]
    fn test_reconnection_manager() {
        let mut manager = ReconnectionManager::new();
        
        assert_eq!(manager.state(), ReconnectState::Idle);
        
        let delay = manager.on_disconnect();
        assert_eq!(manager.state(), ReconnectState::Waiting);
        assert!(delay.as_millis() >= 1000);
        
        manager.on_connected();
        assert_eq!(manager.state(), ReconnectState::Idle);
    }
}
