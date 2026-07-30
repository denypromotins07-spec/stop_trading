//! Gateway Module Root
//!
//! Multi-Exchange Gateway Architecture for institutional-grade connectivity.
//! Handles concurrent connections to multiple CEX/DEX venues with automatic
//! load balancing and failover logic.

pub mod manager;
pub mod venue;

pub use manager::{GatewayManager, VenueStatus, LoadBalancer, FailoverPolicy};
pub use venue::{VenueAdapter, VenueType, ConnectionConfig, OrderRoutingDecision};

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use crossbeam_channel::{bounded, Sender, Receiver};

/// Connection pool managing multi-venue connections
#[repr(C)]
pub struct ConnectionPool {
    /// Maximum connections per venue
    max_connections_per_venue: usize,
    /// Total active connections
    active_connections: AtomicUsize,
    /// Pool is running
    is_running: AtomicBool,
    /// Connection errors count
    connection_errors: AtomicU64,
    /// Reconnection attempts
    reconnection_attempts: AtomicU64,
}

impl ConnectionPool {
    pub fn new(max_connections_per_venue: usize) -> Self {
        Self {
            max_connections_per_venue,
            active_connections: AtomicUsize::new(0),
            is_running: AtomicBool::new(true),
            connection_errors: AtomicU64::new(0),
            reconnection_attempts: AtomicU64::new(0),
        }
    }

    /// Try to acquire a connection slot
    #[inline]
    pub fn try_acquire(&self) -> bool {
        if !self.is_running.load(Ordering::Acquire) {
            return false;
        }

        let current = self.active_connections.load(Ordering::Acquire);
        if current >= self.max_connections_per_venue * 10 {
            return false;
        }

        self.active_connections.fetch_add(1, Ordering::AcqRel);
        true
    }

    /// Release a connection slot
    #[inline]
    pub fn release(&self) {
        self.active_connections.fetch_sub(1, Ordering::AcqRel);
    }

    /// Get active connection count
    #[inline]
    pub fn active_count(&self) -> usize {
        self.active_connections.load(Ordering::Acquire)
    }

    /// Record connection error
    #[inline]
    pub fn record_error(&self) {
        self.connection_errors.fetch_add(1, Ordering::Relaxed);
    }

    /// Record reconnection attempt
    #[inline]
    pub fn record_reconnect(&self) {
        self.reconnection_attempts.fetch_add(1, Ordering::Relaxed);
    }

    /// Stop the pool
    #[inline]
    pub fn stop(&self) {
        self.is_running.store(false, Ordering::Release);
    }

    /// Check if pool is running
    #[inline]
    pub fn is_running(&self) -> bool {
        self.is_running.load(Ordering::Acquire)
    }

    /// Get pool statistics
    #[inline]
    pub fn get_stats(&self) -> ConnectionPoolStats {
        ConnectionPoolStats {
            active_connections: self.active_count(),
            max_connections: self.max_connections_per_venue * 10,
            connection_errors: self.connection_errors.load(Ordering::Relaxed),
            reconnection_attempts: self.reconnection_attempts.load(Ordering::Relaxed),
            is_running: self.is_running(),
        }
    }
}

/// Connection pool statistics
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ConnectionPoolStats {
    pub active_connections: usize,
    pub max_connections: usize,
    pub connection_errors: u64,
    pub reconnection_attempts: u64,
    pub is_running: bool,
}

/// Event channel for gateway events
#[repr(C)]
pub struct GatewayEventChannel {
    sender: Sender<GatewayEvent>,
    receiver: Receiver<GatewayEvent>,
}

impl GatewayEventChannel {
    pub fn new(buffer_size: usize) -> Self {
        let (sender, receiver) = bounded(buffer_size);
        Self { sender, receiver }
    }

    #[inline]
    pub fn send(&self, event: GatewayEvent) -> Result<(), GatewayEvent> {
        self.sender.send(event).map_err(|e| e.0)
    }

    #[inline]
    pub fn try_send(&self, event: GatewayEvent) -> Result<(), GatewayEvent> {
        self.sender.try_send(event).map_err(|e| e.0)
    }

    #[inline]
    pub fn recv(&self) -> Result<GatewayEvent, crossbeam_channel::RecvError> {
        self.receiver.recv()
    }

    #[inline]
    pub fn try_recv(&self) -> Result<GatewayEvent, crossbeam_channel::TryRecvError> {
        self.receiver.try_recv()
    }

    pub fn sender(&self) -> &Sender<GatewayEvent> {
        &self.sender
    }

    pub fn receiver(&self) -> &Receiver<GatewayEvent> {
        &self.receiver
    }
}

/// Gateway events for internal communication
#[repr(C)]
#[derive(Debug, Clone)]
pub enum GatewayEvent {
    /// Venue connected
    VenueConnected { venue_id: u32, timestamp_ns: u64 },
    /// Venue disconnected
    VenueDisconnected { venue_id: u32, reason: DisconnectReason },
    /// Order routed
    OrderRouted { order_id: u64, venue_id: u32 },
    /// Fill received
    FillReceived { order_id: u64, fill_price: u64, fill_qty: u64 },
    /// Market data update
    MarketDataUpdate { venue_id: u32, symbol_hash: u64 },
    /// Latency measurement
    LatencyMeasurement { venue_id: u32, latency_ns: u64 },
    /// Error occurred
    Error { venue_id: u32, error_code: u32 },
}

/// Reason for disconnection
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum DisconnectReason {
    /// Network error
    NetworkError,
    /// Authentication failed
    AuthFailed,
    /// Rate limited
    RateLimited,
    /// Graceful shutdown
    GracefulShutdown,
    /// Timeout
    Timeout,
    /// Protocol error
    ProtocolError,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_connection_pool() {
        let pool = ConnectionPool::new(5);
        
        assert!(pool.is_running());
        assert_eq!(pool.active_count(), 0);

        // Acquire connections
        assert!(pool.try_acquire());
        assert_eq!(pool.active_count(), 1);

        // Release connection
        pool.release();
        assert_eq!(pool.active_count(), 0);

        // Stop pool
        pool.stop();
        assert!(!pool.is_running());
        assert!(!pool.try_acquire());
    }

    #[test]
    fn test_event_channel() {
        let channel = GatewayEventChannel::new(100);
        
        let event = GatewayEvent::VenueConnected { 
            venue_id: 1, 
            timestamp_ns: 1234567890 
        };
        
        channel.send(event.clone()).unwrap();
        
        let received = channel.try_recv().unwrap();
        match received {
            GatewayEvent::VenueConnected { venue_id, .. } => {
                assert_eq!(venue_id, 1);
            }
            _ => panic!("Wrong event type"),
        }
    }
}
