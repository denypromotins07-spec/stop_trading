//! Gateway Lifecycle Manager
//!
//! Handles concurrent connections to multiple CEX/DEX venues with
//! automatic load balancing and failover logic. Routes orders to
//! the venue with the best current liquidity.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::collections::HashMap;
use crossbeam_channel::{bounded, Sender, Receiver};

use super::venue::{
    VenueAdapter, VenueType, ConnectionConfig, OrderRoutingDecision,
    OrderRequest, OrderResponse, CancelResponse, VenueError, LiquidityInfo, VenueStats,
};
use super::{GatewayEvent, GatewayEventChannel, ConnectionPool};

/// Maximum number of venues supported
pub const MAX_VENUES: usize = 32;

/// Failover policy for routing decisions
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailoverPolicy {
    /// Route to next best venue immediately
    Immediate,
    /// Wait for retry delay before failover
    Delayed { delay_ms: u32 },
    /// Only failover if primary is completely down
    StrictPrimary,
    /// Round-robin among healthy venues
    RoundRobin,
}

impl Default for FailoverPolicy {
    fn default() -> Self {
        FailoverPolicy::Immediate
    }
}

/// Venue status information
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct VenueStatus {
    /// Venue ID
    pub venue_id: u32,
    /// Is connected
    pub is_connected: bool,
    /// Current latency in nanoseconds
    pub latency_ns: u64,
    /// Available liquidity score
    pub liquidity_score: u32,
    /// Health score (0-100)
    pub health_score: u8,
    /// Is primary venue
    pub is_primary: bool,
    /// Is in failover state
    pub is_failover: bool,
    /// Last update timestamp
    pub last_update_ns: u64,
}

impl VenueStatus {
    #[inline]
    pub fn new(venue_id: u32) -> Self {
        Self {
            venue_id,
            is_connected: false,
            latency_ns: u64::MAX,
            liquidity_score: 0,
            health_score: 0,
            is_primary: false,
            is_failover: false,
            last_update_ns: 0,
        }
    }

    #[inline]
    pub fn calculate_routing_score(&self) -> u32 {
        if !self.is_connected || self.health_score < 50 {
            return 0;
        }

        // Higher score = better route
        // Factors: low latency, high liquidity, good health
        let latency_factor = if self.latency_ns < 1_000_000 {
            100u32
        } else if self.latency_ns < 5_000_000 {
            80u32
        } else if self.latency_ns < 10_000_000 {
            60u32
        } else {
            40u32
        };

        let health_factor = self.health_score as u32;
        let liquidity_factor = self.liquidity_score;

        (latency_factor * 3 + health_factor * 2 + liquidity_factor) / 6
    }
}

/// Load balancer for venue selection
#[repr(C)]
pub struct LoadBalancer {
    /// Current venue statuses
    venue_statuses: [VenueStatus; MAX_VENUES],
    /// Number of active venues
    active_count: AtomicUsize,
    /// Primary venue ID
    primary_venue_id: AtomicU64,
    /// Current round-robin index
    rr_index: AtomicUsize,
    /// Failover policy
    failover_policy: FailoverPolicy,
    /// Total routing decisions made
    routing_decisions: AtomicU64,
    /// Failover events count
    failover_count: AtomicU64,
}

impl LoadBalancer {
    pub fn new(failover_policy: FailoverPolicy) -> Self {
        Self {
            venue_statuses: std::array::from_fn(|_| VenueStatus::new(0)),
            active_count: AtomicUsize::new(0),
            primary_venue_id: AtomicU64::new(u64::MAX),
            rr_index: AtomicUsize::new(0),
            failover_policy,
            routing_decisions: AtomicU64::new(0),
            failover_count: AtomicU64::new(0),
        }
    }

    /// Update venue status
    #[inline]
    pub fn update_venue_status(&self, status: VenueStatus) {
        let idx = status.venue_id as usize;
        if idx < MAX_VENUES {
            self.venue_statuses[idx] = status;
            
            if status.is_connected && status.health_score >= 50 {
                // Increment active count if this venue just became healthy
                // (simplified - in production would track previous state)
            }
        }
    }

    /// Set primary venue
    #[inline]
    pub fn set_primary_venue(&self, venue_id: u32) {
        self.primary_venue_id.store(venue_id as u64, Ordering::Release);
        
        // Mark as primary in status array
        let idx = venue_id as usize;
        if idx < MAX_VENUES {
            self.venue_statuses[idx].is_primary = true;
        }
    }

    /// Get best venue for routing based on current conditions
    #[inline]
    pub fn select_best_venue(&self, symbol_hash: u64) -> Option<OrderRoutingDecision> {
        let mut best_score = 0u32;
        let mut best_venue: Option<VenueStatus> = None;

        let count = self.active_count.load(Ordering::Acquire);
        if count == 0 {
            return None;
        }

        for i in 0..MAX_VENUES {
            let status = &self.venue_statuses[i];
            if !status.is_connected || status.health_score < 50 {
                continue;
            }

            let score = status.calculate_routing_score();
            if score > best_score {
                best_score = score;
                best_venue = Some(*status);
            }
        }

        best_venue.map(|v| {
            self.routing_decisions.fetch_add(1, Ordering::Relaxed);
            OrderRoutingDecision::new(v.venue_id, v.latency_ns, v.liquidity_score as u64)
        })
    }

    /// Select venue using round-robin among healthy venues
    #[inline]
    pub fn select_round_robin(&self) -> Option<OrderRoutingDecision> {
        let count = self.active_count.load(Ordering::Acquire);
        if count == 0 {
            return None;
        }

        let start = self.rr_index.load(Ordering::Acquire);
        let mut checked = 0;

        while checked < MAX_VENUES {
            let idx = (start + checked) % MAX_VENUES;
            let status = &self.venue_statuses[idx];

            if status.is_connected && status.health_score >= 50 {
                self.rr_index.store((idx + 1) % MAX_VENUES, Ordering::Release);
                self.routing_decisions.fetch_add(1, Ordering::Relaxed);
                
                return Some(OrderRoutingDecision::new(
                    status.venue_id,
                    status.latency_ns,
                    status.liquidity_score as u64,
                ));
            }

            checked += 1;
        }

        None
    }

    /// Get failover venue when primary fails
    #[inline]
    pub fn get_failover_venue(&self, failed_venue_id: u32) -> Option<OrderRoutingDecision> {
        self.failover_count.fetch_add(1, Ordering::Relaxed);

        match self.failover_policy {
            FailoverPolicy::StrictPrimary => {
                // Only use primary, no failover
                None
            }
            FailoverPolicy::RoundRobin => {
                self.select_round_robin().map(|mut d| d.as_failover())
            }
            _ => {
                // Find next best venue that's not the failed one
                let mut best_score = 0u32;
                let mut best_venue: Option<VenueStatus> = None;

                for i in 0..MAX_VENUES {
                    let status = &self.venue_statuses[i];
                    if !status.is_connected 
                        || status.health_score < 50 
                        || status.venue_id == failed_venue_id 
                    {
                        continue;
                    }

                    let score = status.calculate_routing_score();
                    if score > best_score {
                        best_score = score;
                        best_venue = Some(*status);
                    }
                }

                best_venue.map(|v| {
                    OrderRoutingDecision::new(v.venue_id, v.latency_ns, v.liquidity_score as u64)
                        .as_failover()
                })
            }
        }
    }

    /// Get all healthy venues
    #[inline]
    pub fn get_healthy_venues(&self) -> Vec<VenueStatus> {
        self.venue_statuses
            .iter()
            .filter(|s| s.is_connected && s.health_score >= 50)
            .copied()
            .collect()
    }

    /// Get routing statistics
    #[inline]
    pub fn get_stats(&self) -> LoadBalancerStats {
        LoadBalancerStats {
            active_venues: self.active_count.load(Ordering::Acquire),
            routing_decisions: self.routing_decisions.load(Ordering::Relaxed),
            failover_count: self.failover_count.load(Ordering::Relaxed),
            primary_venue_id: self.primary_venue_id.load(Ordering::Acquire) as u32,
        }
    }

    /// Update active venue count
    #[inline]
    fn update_active_count(&self) {
        let count = self.venue_statuses
            .iter()
            .filter(|s| s.is_connected && s.health_score >= 50)
            .count();
        self.active_count.store(count, Ordering::Release);
    }
}

/// Load balancer statistics
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct LoadBalancerStats {
    pub active_venues: usize,
    pub routing_decisions: u64,
    pub failover_count: u64,
    pub primary_venue_id: u32,
}

/// Gateway lifecycle manager
#[repr(C)]
pub struct GatewayManager {
    /// Connection pool
    connection_pool: Arc<ConnectionPool>,
    /// Load balancer
    load_balancer: Arc<LoadBalancer>,
    /// Event channel
    event_channel: Arc<GatewayEventChannel>,
    /// Registered venues (stored as trait objects via Arc)
    venues: Vec<Arc<dyn VenueAdapter>>,
    /// Manager is running
    is_running: AtomicBool,
    /// Total orders routed
    orders_routed: AtomicU64,
    /// Total fills received
    fills_received: AtomicU64,
    /// Error count
    error_count: AtomicU64,
}

impl GatewayManager {
    pub fn new(buffer_size: usize, max_connections_per_venue: usize, failover_policy: FailoverPolicy) -> Self {
        Self {
            connection_pool: Arc::new(ConnectionPool::new(max_connections_per_venue)),
            load_balancer: Arc::new(LoadBalancer::new(failover_policy)),
            event_channel: Arc::new(GatewayEventChannel::new(buffer_size)),
            venues: Vec::with_capacity(MAX_VENUES),
            is_running: AtomicBool::new(false),
            orders_routed: AtomicU64::new(0),
            fills_received: AtomicU64::new(0),
            error_count: AtomicU64::new(0),
        }
    }

    /// Register a venue adapter
    #[inline]
    pub fn register_venue(&mut self, venue: Arc<dyn VenueAdapter>) -> Result<(), VenueError> {
        if self.venues.len() >= MAX_VENUES {
            return Err(VenueError::InternalError);
        }

        let venue_id = venue.venue_id();
        
        // Initialize venue status
        let status = VenueStatus::new(venue_id);
        self.load_balancer.update_venue_status(status);

        self.venues.push(venue);
        Ok(())
    }

    /// Connect to all registered venues
    #[inline]
    pub fn connect_all(&self) -> Result<(), VenueError> {
        if !self.connection_pool.is_running() {
            return Err(VenueError::NotConnected);
        }

        for venue in &self.venues {
            match venue.connect() {
                Ok(_) => {
                    let mut status = VenueStatus::new(venue.venue_id());
                    status.is_connected = true;
                    status.health_score = 100;
                    status.last_update_ns = self.get_timestamp_ns();
                    self.load_balancer.update_venue_status(status);

                    let _ = self.event_channel.send(GatewayEvent::VenueConnected {
                        venue_id: venue.venue_id(),
                        timestamp_ns: status.last_update_ns,
                    });
                }
                Err(e) => {
                    self.error_count.fetch_add(1, Ordering::Relaxed);
                    let _ = self.event_channel.send(GatewayEvent::Error {
                        venue_id: venue.venue_id(),
                        error_code: e.error_code(),
                    });
                }
            }
        }

        self.load_balancer.update_active_count();
        Ok(())
    }

    /// Disconnect from all venues
    #[inline]
    pub fn disconnect_all(&self) {
        for venue in &self.venues {
            let _ = venue.disconnect();
            
            let mut status = VenueStatus::new(venue.venue_id());
            status.last_update_ns = self.get_timestamp_ns();
            self.load_balancer.update_venue_status(status);

            let _ = self.event_channel.send(GatewayEvent::VenueDisconnected {
                venue_id: venue.venue_id(),
                reason: super::DisconnectReason::GracefulShutdown,
            });
        }

        self.load_balancer.update_active_count();
    }

    /// Route order to best venue
    #[inline]
    pub fn route_order(&self, order: &OrderRequest) -> Result<OrderResponse, VenueError> {
        let decision = self.load_balancer
            .select_best_venue(order.symbol_hash)
            .ok_or(VenueError::NotConnected)?;

        let venue = self.find_venue(decision.venue_id)
            .ok_or(VenueError::NotConnected)?;

        self.orders_routed.fetch_add(1, Ordering::Relaxed);

        let response = venue.submit_order(order)?;

        let _ = self.event_channel.send(GatewayEvent::OrderRouted {
            order_id: order.client_order_id,
            venue_id: decision.venue_id,
        });

        Ok(response)
    }

    /// Route order with failover
    #[inline]
    pub fn route_order_with_failover(&self, order: &OrderRequest) -> Result<OrderResponse, VenueError> {
        let mut attempt_count = 0;
        let mut last_error: Option<VenueError> = None;

        loop {
            let decision = if attempt_count == 0 {
                self.load_balancer.select_best_venue(order.symbol_hash)
            } else {
                // Use failover after first attempt fails
                let failed_venue = if let Some(ref err) = last_error {
                    // In production, would track which venue failed
                    0
                } else {
                    0
                };
                self.load_balancer.get_failover_venue(failed_venue)
            };

            let decision = decision.ok_or_else|| {
                last_error.unwrap_or(VenueError::NotConnected)
            }?;

            let venue = self.find_venue(decision.venue_id)
                .ok_or(VenueError::NotConnected)?;

            match venue.submit_order(order) {
                Ok(response) => {
                    if response.is_rejected() {
                        last_error = Some(VenueError::OrderRejected);
                    } else {
                        self.orders_routed.fetch_add(1, Ordering::Relaxed);
                        
                        let _ = self.event_channel.send(GatewayEvent::OrderRouted {
                            order_id: order.client_order_id,
                            venue_id: decision.venue_id,
                        });

                        return Ok(response);
                    }
                }
                Err(e) => {
                    last_error = Some(e);
                }
            }

            attempt_count += 1;
            if attempt_count >= 3 {
                self.error_count.fetch_add(1, Ordering::Relaxed);
                return Err(last_error.unwrap_or(VenueError::InternalError));
            }
        }
    }

    /// Cancel order on specific venue
    #[inline]
    pub fn cancel_order(&self, venue_id: u32, order_id: u64) -> Result<CancelResponse, VenueError> {
        let venue = self.find_venue(venue_id)
            .ok_or(VenueError::NotConnected)?;

        venue.cancel_order(order_id)
    }

    /// Get liquidity info across all venues
    #[inline]
    pub fn get_aggregated_liquidity(&self, symbol_hash: u64) -> AggregatedLiquidity {
        let mut total_bid_size = 0u64;
        let mut total_ask_size = 0u64;
        let mut best_bid = 0u64;
        let mut best_ask = u64::MAX;
        let mut venue_count = 0u32;

        for venue in &self.venues {
            if !venue.is_connected() {
                continue;
            }

            let liq = venue.get_liquidity(symbol_hash);
            if liq.is_valid() {
                total_bid_size += liq.bid_size;
                total_ask_size += liq.ask_size;

                if liq.best_bid > best_bid {
                    best_bid = liq.best_bid;
                }
                if liq.best_ask < best_ask {
                    best_ask = liq.best_ask;
                }

                venue_count += 1;
            }
        }

        AggregatedLiquidity {
            best_bid,
            best_ask,
            total_bid_size,
            total_ask_size,
            spread: if best_ask > best_bid { best_ask - best_bid } else { 0 },
            mid_price: if best_bid > 0 && best_ask < u64::MAX {
                (best_bid + best_ask) / 2
            } else {
                0
            },
            venue_count,
            timestamp_ns: self.get_timestamp_ns(),
        }
    }

    /// Find venue by ID
    #[inline]
    fn find_venue(&self, venue_id: u32) -> Option<Arc<dyn VenueAdapter>> {
        self.venues
            .iter()
            .find(|v| v.venue_id() == venue_id)
            .cloned()
    }

    /// Get current timestamp in nanoseconds
    #[inline]
    fn get_timestamp_ns(&self) -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64
    }

    /// Start the gateway manager
    #[inline]
    pub fn start(&self) {
        self.is_running.store(true, Ordering::Release);
        self.connection_pool.stop(); // Reset and restart
    }

    /// Stop the gateway manager
    #[inline]
    pub fn stop(&self) {
        self.is_running.store(false, Ordering::Release);
        self.disconnect_all();
        self.connection_pool.stop();
    }

    /// Check if running
    #[inline]
    pub fn is_running(&self) -> bool {
        self.is_running.load(Ordering::Acquire)
    }

    /// Get manager statistics
    #[inline]
    pub fn get_stats(&self) -> GatewayManagerStats {
        GatewayManagerStats {
            venues_registered: self.venues.len(),
            orders_routed: self.orders_routed.load(Ordering::Relaxed),
            fills_received: self.fills_received.load(Ordering::Relaxed),
            error_count: self.error_count.load(Ordering::Relaxed),
            is_running: self.is_running(),
            lb_stats: self.load_balancer.get_stats(),
            pool_stats: self.connection_pool.get_stats(),
        }
    }

    /// Get event channel receiver
    #[inline]
    pub fn event_receiver(&self) -> &Receiver<GatewayEvent> {
        self.event_channel.receiver()
    }

    /// Get event channel sender
    #[inline]
    pub fn event_sender(&self) -> &Sender<GatewayEvent> {
        self.event_channel.sender()
    }
}

/// Aggregated liquidity across venues
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AggregatedLiquidity {
    pub best_bid: u64,
    pub best_ask: u64,
    pub total_bid_size: u64,
    pub total_ask_size: u64,
    pub spread: u64,
    pub mid_price: u64,
    pub venue_count: u32,
    pub timestamp_ns: u64,
}

/// Gateway manager statistics
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GatewayManagerStats {
    pub venues_registered: usize,
    pub orders_routed: u64,
    pub fills_received: u64,
    pub error_count: u64,
    pub is_running: bool,
    pub lb_stats: LoadBalancerStats,
    pub pool_stats: super::ConnectionPoolStats,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_venue_status_scoring() {
        let mut status = VenueStatus::new(1);
        status.is_connected = true;
        status.health_score = 100;
        status.latency_ns = 500_000;
        status.liquidity_score = 100;

        let score = status.calculate_routing_score();
        assert!(score > 0);
        assert!(score <= 100);
    }

    #[test]
    fn test_load_balancer() {
        let lb = LoadBalancer::new(FailoverPolicy::default());

        // No venues yet
        assert!(lb.select_best_venue(0).is_none());

        // Add a venue status
        let mut status = VenueStatus::new(1);
        status.is_connected = true;
        status.health_score = 100;
        status.latency_ns = 1_000_000;
        status.liquidity_score = 500;
        lb.update_venue_status(status);

        // Should now have a route
        let decision = lb.select_best_venue(0);
        assert!(decision.is_some());
    }

    #[test]
    fn test_gateway_manager_creation() {
        let manager = GatewayManager::new(1000, 5, FailoverPolicy::default());
        
        assert!(!manager.is_running());
        assert_eq!(manager.get_stats().venues_registered, 0);

        manager.start();
        assert!(manager.is_running());

        manager.stop();
        assert!(!manager.is_running());
    }

    #[test]
    fn test_failover_policy() {
        let lb = LoadBalancer::new(FailoverPolicy::StrictPrimary);
        
        // Strict primary should not provide failover
        let failover = lb.get_failover_venue(1);
        assert!(failover.is_none());

        let lb_immediate = LoadBalancer::new(FailoverPolicy::Immediate);
        // Would need venues registered to test actual failover
    }
}
