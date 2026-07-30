//! FIX Module Root
//!
//! High-Performance FIX Protocol Adapter for institutional execution.
//! Integrates with the smart order router for seamless order routing.

pub mod codec;
pub mod session;

pub use codec::{FixCodec, FixMessage, FixField, FixError, FixTag};
pub use session::{FixSession, SessionState, SessionConfig, ResendRequest};

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

/// FIX Engine integrating codec and session management
#[repr(C)]
pub struct FixEngine {
    /// Codec for message encoding/decoding
    codec: Arc<FixCodec>,
    /// Active session
    session: Option<Arc<FixSession>>,
    /// Engine is running
    is_running: AtomicBool,
    /// Messages processed count
    messages_processed: AtomicU64,
    /// Encoding errors count
    encode_errors: AtomicU64,
    /// Decoding errors count
    decode_errors: AtomicU64,
}

impl FixEngine {
    pub fn new() -> Self {
        Self {
            codec: Arc::new(FixCodec::new()),
            session: None,
            is_running: AtomicBool::new(false),
            messages_processed: AtomicU64::new(0),
            encode_errors: AtomicU64::new(0),
            decode_errors: AtomicU64::new(0),
        }
    }

    /// Create a new FIX session
    #[inline]
    pub fn create_session(&mut self, config: SessionConfig) -> Result<Arc<FixSession>, FixError> {
        let session = Arc::new(FixSession::new(config, self.codec.clone())?);
        self.session = Some(session.clone());
        Ok(session)
    }

    /// Get current session
    #[inline]
    pub fn get_session(&self) -> Option<Arc<FixSession>> {
        self.session.clone()
    }

    /// Encode a message
    #[inline]
    pub fn encode(&self, msg: &FixMessage, buffer: &mut [u8]) -> Result<usize, FixError> {
        if !self.is_running.load(Ordering::Acquire) {
            return Err(FixError::SessionNotActive);
        }

        match self.codec.encode(msg, buffer) {
            Ok(len) => {
                self.messages_processed.fetch_add(1, Ordering::Relaxed);
                Ok(len)
            }
            Err(e) => {
                self.encode_errors.fetch_add(1, Ordering::Relaxed);
                Err(e)
            }
        }
    }

    /// Decode a message from buffer
    #[inline]
    pub fn decode<'a>(&self, buffer: &'a [u8]) -> Result<FixMessage<'a>, FixError> {
        if !self.is_running.load(Ordering::Acquire) {
            return Err(FixError::SessionNotActive);
        }

        match self.codec.decode(buffer) {
            Ok(msg) => {
                self.messages_processed.fetch_add(1, Ordering::Relaxed);
                Ok(msg)
            }
            Err(e) => {
                self.decode_errors.fetch_add(1, Ordering::Relaxed);
                Err(e)
            }
        }
    }

    /// Start the engine
    #[inline]
    pub fn start(&self) {
        self.is_running.store(true, Ordering::Release);
        if let Some(ref session) = self.session {
            session.start();
        }
    }

    /// Stop the engine
    #[inline]
    pub fn stop(&self) {
        self.is_running.store(false, Ordering::Release);
        if let Some(ref session) = self.session {
            session.stop();
        }
    }

    /// Check if running
    #[inline]
    pub fn is_running(&self) -> bool {
        self.is_running.load(Ordering::Acquire)
    }

    /// Get engine statistics
    #[inline]
    pub fn get_stats(&self) -> FixEngineStats {
        FixEngineStats {
            messages_processed: self.messages_processed.load(Ordering::Relaxed),
            encode_errors: self.encode_errors.load(Ordering::Relaxed),
            decode_errors: self.decode_errors.load(Ordering::Relaxed),
            is_running: self.is_running(),
            session_active: self.session.is_some() && self.session.as_ref().map_or(false, |s| s.is_logged_on()),
        }
    }
}

impl Default for FixEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// FIX engine statistics
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FixEngineStats {
    pub messages_processed: u64,
    pub encode_errors: u64,
    pub decode_errors: u64,
    pub is_running: bool,
    pub session_active: bool,
}

/// Smart Order Router integration for FIX execution
#[repr(C)]
pub struct FixRouter {
    /// FIX engine reference
    engine: Arc<FixEngine>,
    /// Orders routed via FIX
    orders_routed: AtomicU64,
    /// Orders rejected by venue
    orders_rejected: AtomicU64,
    /// Average fill latency in nanoseconds
    avg_fill_latency_ns: AtomicU64,
    /// Total fill latency for averaging
    total_fill_latency_ns: AtomicU64,
}

impl FixRouter {
    pub fn new(engine: Arc<FixEngine>) -> Self {
        Self {
            engine,
            orders_routed: AtomicU64::new(0),
            orders_rejected: AtomicU64::new(0),
            avg_fill_latency_ns: AtomicU64::new(0),
            total_fill_latency_ns: AtomicU64::new(0),
        }
    }

    /// Route order through FIX session
    #[inline]
    pub fn route_order(&self, order_msg: &FixMessage) -> Result<(), FixError> {
        if !self.engine.is_running() {
            return Err(FixError::SessionNotActive);
        }

        // Encode and send order
        let mut buffer = [0u8; 4096];
        let _len = self.engine.encode(order_msg, &mut buffer)?;

        self.orders_routed.fetch_add(1, Ordering::Relaxed);

        // In production, would send to network layer here
        Ok(())
    }

    /// Record fill with latency measurement
    #[inline]
    pub fn record_fill(&self, latency_ns: u64) {
        let total = self.total_fill_latency_ns.fetch_add(latency_ns, Ordering::Relaxed);
        let count = self.orders_routed.load(Ordering::Relaxed);
        
        if count > 0 {
            let avg = (total + latency_ns) / count;
            self.avg_fill_latency_ns.store(avg, Ordering::Release);
        }
    }

    /// Record order rejection
    #[inline]
    pub fn record_rejection(&self) {
        self.orders_rejected.fetch_add(1, Ordering::Relaxed);
    }

    /// Get router statistics
    #[inline]
    pub fn get_stats(&self) -> FixRouterStats {
        FixRouterStats {
            orders_routed: self.orders_routed.load(Ordering::Relaxed),
            orders_rejected: self.orders_rejected.load(Ordering::Relaxed),
            avg_fill_latency_ns: self.avg_fill_latency_ns.load(Ordering::Acquire),
            fill_rate: if self.orders_routed.load(Ordering::Relaxed) > 0 {
                (self.orders_routed.load(Ordering::Relaxed) - self.orders_rejected.load(Ordering::Relaxed)) as f64 
                    / self.orders_routed.load(Ordering::Relaxed) as f64
            } else {
                0.0
            },
        }
    }
}

/// FIX router statistics
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FixRouterStats {
    pub orders_routed: u64,
    pub orders_rejected: u64,
    pub avg_fill_latency_ns: u64,
    pub fill_rate: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fix_engine_creation() {
        let engine = FixEngine::new();
        
        assert!(!engine.is_running());
        assert_eq!(engine.get_stats().messages_processed, 0);

        engine.start();
        assert!(engine.is_running());

        engine.stop();
        assert!(!engine.is_running());
    }

    #[test]
    fn test_fix_router() {
        let engine = Arc::new(FixEngine::new());
        let router = FixRouter::new(engine.clone());

        assert_eq!(router.get_stats().orders_routed, 0);

        router.record_fill(1_000_000);
        router.record_fill(2_000_000);
        
        let stats = router.get_stats();
        assert!(stats.avg_fill_latency_ns > 0);
    }

    #[test]
    fn test_engine_stats() {
        let engine = FixEngine::new();
        let stats = engine.get_stats();

        assert_eq!(stats.messages_processed, 0);
        assert_eq!(stats.encode_errors, 0);
        assert_eq!(stats.decode_errors, 0);
        assert!(!stats.is_running);
    }
}
