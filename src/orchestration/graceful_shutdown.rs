//! Graceful Shutdown Handler
//! 
//! Multi-phase graceful shutdown sequence triggered by SIGTERM or /KILL.
//! Safely cancels orders, flattens inventory, flushes WAL, and persists state.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::time::Duration;
use crossbeam_utils::CachePadded;

/// Shutdown phase
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ShutdownPhase {
    /// Running normally
    Running = 0,
    /// Phase 1: Stop accepting new orders
    StopNewOrders = 1,
    /// Phase 2: Cancel open orders
    CancelOrders = 2,
    /// Phase 3: Flatten positions
    FlattenPositions = 3,
    /// Phase 4: Flush write-ahead log
    FlushWal = 4,
    /// Phase 5: Persist final state
    PersistState = 5,
    /// Complete shutdown
    Complete = 6,
}

/// Shutdown reason
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShutdownReason {
    /// Normal shutdown (SIGTERM)
    SigTerm,
    /// Immediate shutdown (SIGINT)
    SigInt,
    /// Kill signal (SIGKILL via /KILL endpoint)
    Kill,
    /// Internal error
    Error,
    /// System shutdown
    System,
}

/// Shutdown coordinator
pub struct GracefulShutdown {
    /// Current phase
    phase: CachePadded<AtomicU8>,
    /// Shutdown initiated flag
    initiated: CachePadded<AtomicBool>,
    /// Shutdown complete flag
    complete: CachePadded<AtomicBool>,
    /// Shutdown reason
    reason: CachePadded<AtomicU8>,
    /// Start timestamp (nanoseconds)
    start_time_ns: CachePadded<AtomicU64>,
    /// Timeout for each phase (milliseconds)
    phase_timeout_ms: u64,
    /// Total timeout (milliseconds)
    total_timeout_ms: u64,
    /// Orders cancelled count
    orders_cancelled: CachePadded<AtomicU64>,
    /// Positions flattened count
    positions_flattened: CachePadded<AtomicU64>,
    /// WAL records flushed
    wal_records_flushed: CachePadded<AtomicU64>,
}

impl GracefulShutdown {
    /// Create a new graceful shutdown handler
    /// 
    /// # Arguments
    /// * `phase_timeout_ms` - Timeout per phase in milliseconds
    /// * `total_timeout_ms` - Total shutdown timeout in milliseconds
    pub fn new(phase_timeout_ms: u64, total_timeout_ms: u64) -> Self {
        Self {
            phase: CachePadded::new(AtomicU8::new(ShutdownPhase::Running as u8)),
            initiated: CachePadded::new(AtomicBool::new(false)),
            complete: CachePadded::new(AtomicBool::new(false)),
            reason: CachePadded::new(AtomicU8::new(ShutdownReason::SigTerm as u8)),
            start_time_ns: CachePadded::new(AtomicU64::new(0)),
            phase_timeout_ms,
            total_timeout_ms,
            orders_cancelled: CachePadded::new(AtomicU64::new(0)),
            positions_flattened: CachePadded::new(AtomicU64::new(0)),
            wal_records_flushed: CachePadded::new(AtomicU64::new(0)),
        }
    }

    /// Initiate graceful shutdown
    /// 
    /// # Arguments
    /// * `reason` - Reason for shutdown
    pub fn initiate(&self, reason: ShutdownReason) -> bool {
        if self.initiated.swap(true, Ordering::SeqCst) {
            return false; // Already initiated
        }

        self.reason.store(reason as u8, Ordering::Relaxed);
        self.start_time_ns.store(get_timestamp_ns(), Ordering::Relaxed);
        
        // Enter first phase
        self.phase.store(ShutdownPhase::StopNewOrders as u8, Ordering::SeqCst);

        true
    }

    /// Check if shutdown has been initiated
    #[inline]
    pub fn is_initiated(&self) -> bool {
        self.initiated.load(Ordering::Relaxed)
    }

    /// Check if shutdown is complete
    #[inline]
    pub fn is_complete(&self) -> bool {
        self.complete.load(Ordering::Relaxed)
    }

    /// Get current phase
    #[inline]
    pub fn get_phase(&self) -> ShutdownPhase {
        match self.phase.load(Ordering::Relaxed) {
            0 => ShutdownPhase::Running,
            1 => ShutdownPhase::StopNewOrders,
            2 => ShutdownPhase::CancelOrders,
            3 => ShutdownPhase::FlattenPositions,
            4 => ShutdownPhase::FlushWal,
            5 => ShutdownPhase::PersistState,
            _ => ShutdownPhase::Complete,
        }
    }

    /// Get shutdown reason
    #[inline]
    pub fn get_reason(&self) -> ShutdownReason {
        match self.reason.load(Ordering::Relaxed) {
            0 => ShutdownReason::SigTerm,
            1 => ShutdownReason::SigInt,
            2 => ShutdownReason::Kill,
            3 => ShutdownReason::Error,
            _ => ShutdownReason::System,
        }
    }

    /// Advance to next phase
    /// Returns true if should continue, false if timed out or complete
    pub fn advance_phase(&self) -> bool {
        let current = self.get_phase();
        let elapsed_ms = self.get_elapsed_ms();

        // Check total timeout
        if elapsed_ms > self.total_timeout_ms {
            self.force_complete();
            return false;
        }

        let next = match current {
            ShutdownPhase::Running => return true,
            ShutdownPhase::StopNewOrders => ShutdownPhase::CancelOrders,
            ShutdownPhase::CancelOrders => ShutdownPhase::FlattenPositions,
            ShutdownPhase::FlattenPositions => ShutdownPhase::FlushWal,
            ShutdownPhase::FlushWal => ShutdownPhase::PersistState,
            ShutdownPhase::PersistState => {
                self.complete.store(true, Ordering::SeqCst);
                self.phase.store(ShutdownPhase::Complete as u8, Ordering::SeqCst);
                return false;
            }
            ShutdownPhase::Complete => return false,
        };

        self.phase.store(next as u8, Ordering::SeqCst);
        true
    }

    /// Execute shutdown sequence with callbacks
    /// 
    /// # Arguments
    /// * `cancel_orders_fn` - Callback to cancel all open orders
    /// * `flatten_positions_fn` - Callback to flatten all positions
    /// * `flush_wal_fn` - Callback to flush write-ahead log
    /// * `persist_state_fn` - Callback to persist final state
    pub fn execute_shutdown<F1, F2, F3, F4>(
        &self,
        mut cancel_orders_fn: F1,
        mut flatten_positions_fn: F2,
        mut flush_wal_fn: F3,
        mut persist_state_fn: F4,
    ) -> bool
    where
        F1: FnMut() -> usize,
        F2: FnMut() -> usize,
        F3: FnMut() -> usize,
        F4: FnMut() -> bool,
    {
        if !self.initiated.load(Ordering::Relaxed) {
            return true;
        }

        loop {
            let phase = self.get_phase();
            
            match phase {
                ShutdownPhase::StopNewOrders => {
                    // Just advance - stopping new orders is handled elsewhere
                    if !self.advance_phase() {
                        return self.complete.load(Ordering::Relaxed);
                    }
                }
                ShutdownPhase::CancelOrders => {
                    let cancelled = cancel_orders_fn();
                    self.orders_cancelled.store(cancelled as u64, Ordering::Relaxed);
                    if !self.advance_phase() {
                        return self.complete.load(Ordering::Relaxed);
                    }
                }
                ShutdownPhase::FlattenPositions => {
                    let flattened = flatten_positions_fn();
                    self.positions_flattened.store(flattened as u64, Ordering::Relaxed);
                    if !self.advance_phase() {
                        return self.complete.load(Ordering::Relaxed);
                    }
                }
                ShutdownPhase::FlushWal => {
                    let flushed = flush_wal_fn();
                    self.wal_records_flushed.store(flushed as u64, Ordering::Relaxed);
                    if !self.advance_phase() {
                        return self.complete.load(Ordering::Relaxed);
                    }
                }
                ShutdownPhase::PersistState => {
                    let persisted = persist_state_fn();
                    if !persisted {
                        // State persistence failed, but continue anyway
                    }
                    self.complete.store(true, Ordering::SeqCst);
                    self.phase.store(ShutdownPhase::Complete as u8, Ordering::SeqCst);
                    return true;
                }
                ShutdownPhase::Running | ShutdownPhase::Complete => {
                    return self.complete.load(Ordering::Relaxed);
                }
            }
        }
    }

    /// Force immediate completion (for SIGKILL scenarios)
    pub fn force_complete(&self) {
        self.complete.store(true, Ordering::SeqCst);
        self.phase.store(ShutdownPhase::Complete as u8, Ordering::SeqCst);
    }

    /// Get elapsed time since shutdown initiation (milliseconds)
    #[inline]
    pub fn get_elapsed_ms(&self) -> u64 {
        let start = self.start_time_ns.load(Ordering::Relaxed);
        if start == 0 {
            return 0;
        }
        (get_timestamp_ns() - start) / 1_000_000
    }

    /// Get remaining time before timeout (milliseconds)
    #[inline]
    pub fn get_remaining_ms(&self) -> u64 {
        let elapsed = self.get_elapsed_ms();
        self.total_timeout_ms.saturating_sub(elapsed)
    }

    /// Get orders cancelled count
    #[inline]
    pub fn get_orders_cancelled(&self) -> u64 {
        self.orders_cancelled.load(Ordering::Relaxed)
    }

    /// Get positions flattened count
    #[inline]
    pub fn get_positions_flattened(&self) -> u64 {
        self.positions_flattened.load(Ordering::Relaxed)
    }

    /// Get WAL records flushed count
    #[inline]
    pub fn get_wal_records_flushed(&self) -> u64 {
        self.wal_records_flushed.load(Ordering::Relaxed)
    }

    /// Reset the shutdown handler (for testing)
    pub fn reset(&self) {
        self.phase.store(ShutdownPhase::Running as u8, Ordering::Relaxed);
        self.initiated.store(false, Ordering::Relaxed);
        self.complete.store(false, Ordering::Relaxed);
        self.reason.store(ShutdownReason::SigTerm as u8, Ordering::Relaxed);
        self.start_time_ns.store(0, Ordering::Relaxed);
        self.orders_cancelled.store(0, Ordering::Relaxed);
        self.positions_flattened.store(0, Ordering::Relaxed);
        self.wal_records_flushed.store(0, Ordering::Relaxed);
    }
}

#[inline]
fn get_timestamp_ns() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shutdown_phases() {
        let shutdown = GracefulShutdown::new(1000, 5000);
        
        assert_eq!(shutdown.get_phase(), ShutdownPhase::Running);
        assert!(!shutdown.is_initiated());
        
        shutdown.initiate(ShutdownReason::SigTerm);
        
        assert!(shutdown.is_initiated());
        assert_eq!(shutdown.get_phase(), ShutdownPhase::StopNewOrders);
        assert_eq!(shutdown.get_reason(), ShutdownReason::SigTerm);
    }

    #[test]
    fn test_shutdown_execution() {
        let shutdown = GracefulShutdown::new(1000, 5000);
        
        let mut cancel_called = false;
        let mut flatten_called = false;
        let mut flush_called = false;
        let mut persist_called = false;
        
        shutdown.initiate(ShutdownReason::SigTerm);
        
        shutdown.execute_shutdown(
            || { cancel_called = true; 0 },
            || { flatten_called = true; 0 },
            || { flush_called = true; 0 },
            || { persist_called = true; true },
        );
        
        assert!(cancel_called);
        assert!(flatten_called);
        assert!(flush_called);
        assert!(persist_called);
        assert!(shutdown.is_complete());
    }

    #[test]
    fn test_force_complete() {
        let shutdown = GracefulShutdown::new(1000, 5000);
        shutdown.initiate(ShutdownReason::Kill);
        shutdown.force_complete();
        
        assert!(shutdown.is_complete());
        assert_eq!(shutdown.get_phase(), ShutdownPhase::Complete);
    }
}
