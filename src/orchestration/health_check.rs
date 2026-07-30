//! Health Check Endpoint
//! 
//! Comprehensive health check endpoint exposing internal actor states and memory metrics.
//! Allows external monitoring tools to verify the bot's structural integrity.

use std::sync::atomic::{AtomicU64, AtomicBool, AtomicUsize, Ordering};
use crossbeam_utils::CachePadded;

/// Maximum number of actors to track
const MAX_ACTORS: usize = 64;

/// Actor state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ActorState {
    /// Not initialized
    Uninitialized = 0,
    /// Starting up
    Starting = 1,
    /// Running normally
    Running = 2,
    /// Processing messages
    Busy = 3,
    /// Waiting for resources
    Blocked = 4,
    /// Error state
    Error = 5,
    /// Shutting down
    Stopping = 6,
    /// Stopped
    Stopped = 7,
}

/// Actor health information
#[derive(Clone)]
pub struct ActorHealth {
    /// Actor name
    pub name: &'static str,
    /// Current state
    pub state: ActorState,
    /// Messages processed
    pub messages_processed: u64,
    /// Messages pending
    pub messages_pending: u64,
    /// Last activity timestamp (nanoseconds)
    pub last_activity_ns: u64,
    /// Error count
    pub error_count: u64,
}

/// Memory metrics
pub struct MemoryMetrics {
    /// Total heap allocated (bytes)
    pub heap_allocated: u64,
    /// Total heap used (bytes)
    pub heap_used: u64,
    /// Stack size (bytes)
    pub stack_size: u64,
    /// RSS (resident set size) in bytes
    pub rss_bytes: u64,
    /// Virtual memory size in bytes
    pub vms_bytes: u64,
}

/// System health metrics
pub struct SystemHealth {
    /// CPU usage percentage (scaled by 100)
    pub cpu_usage_pct: u32,
    /// Memory usage percentage (scaled by 100)
    pub memory_usage_pct: u32,
    /// Network latency average (microseconds)
    pub network_latency_us: u32,
    /// Disk I/O latency average (microseconds)
    pub disk_latency_us: u32,
}

/// Overall system status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SystemStatus {
    /// All systems operational
    Healthy,
    /// Some degradation but functional
    Degraded,
    /// Critical issues
    Critical,
    /// System unavailable
    Unavailable,
}

/// Health check response
pub struct HealthResponse {
    /// Overall status
    pub status: SystemStatus,
    /// Uptime in seconds
    pub uptime_secs: u64,
    /// Timestamp of this check
    pub timestamp_ns: u64,
    /// Memory metrics
    pub memory: Option<MemoryMetrics>,
    /// System metrics
    pub system: Option<SystemHealth>,
    /// Actor states
    pub actors: Vec<ActorHealth>,
    /// Version string
    pub version: &'static str,
}

/// Health check engine
pub struct HealthCheckEngine {
    /// Actors registry
    actors: CachePadded<[ActorRegistryEntry; MAX_ACTORS]>,
    /// Actor count
    actor_count: CachePadded<AtomicUsize>,
    /// Start timestamp
    start_time_ns: CachePadded<AtomicU64>,
    /// Last health check timestamp
    last_check_ns: CachePadded<AtomicU64>,
    /// Health check counter
    check_count: CachePadded<AtomicU64>,
    /// Engine enabled
    enabled: CachePadded<AtomicBool>,
    /// Version string
    version: &'static str,
}

/// Internal actor registry entry
struct ActorRegistryEntry {
    name: &'static str,
    state: CachePadded<AtomicU8>,
    messages_processed: CachePadded<AtomicU64>,
    messages_pending: CachePadded<AtomicU64>,
    last_activity_ns: CachePadded<AtomicU64>,
    error_count: CachePadded<AtomicU64>,
    active: CachePadded<AtomicBool>,
}

impl Default for ActorRegistryEntry {
    fn default() -> Self {
        Self {
            name: "",
            state: CachePadded::new(AtomicU8::new(ActorState::Uninitialized as u8)),
            messages_processed: CachePadded::new(AtomicU64::new(0)),
            messages_pending: CachePadded::new(AtomicU64::new(0)),
            last_activity_ns: CachePadded::new(AtomicU64::new(0)),
            error_count: CachePadded::new(AtomicU64::new(0)),
            active: CachePadded::new(AtomicBool::new(false)),
        }
    }
}

impl HealthCheckEngine {
    /// Create a new health check engine
    pub fn new(version: &'static str) -> Self {
        let now_ns = get_timestamp_ns();
        
        Self {
            actors: CachePadded::new(std::array::from_fn(|_| ActorRegistryEntry::default())),
            actor_count: CachePadded::new(AtomicUsize::new(0)),
            start_time_ns: CachePadded::new(AtomicU64::new(now_ns)),
            last_check_ns: CachePadded::new(AtomicU64::new(0)),
            check_count: CachePadded::new(AtomicU64::new(0)),
            enabled: CachePadded::new(AtomicBool::new(true)),
            version,
        }
    }

    /// Register an actor for health monitoring
    /// Returns actor ID on success, None if registry is full
    pub fn register_actor(&self, name: &'static str) -> Option<usize> {
        let count = self.actor_count.load(Ordering::Relaxed);
        if count >= MAX_ACTORS {
            return None;
        }

        // Find empty slot or append
        for i in 0..MAX_ACTORS {
            if !self.actors.actors[i].active.load(Ordering::Relaxed) {
                self.actors.actors[i].name = name;
                self.actors.actors[i].state.store(ActorState::Starting as u8, Ordering::Relaxed);
                self.actors.actors[i].active.store(true, Ordering::Relaxed);
                
                if i >= count {
                    self.actor_count.store(i + 1, Ordering::Relaxed);
                }
                
                return Some(i);
            }
        }
        
        None
    }

    /// Update actor state
    pub fn update_actor_state(&self, actor_id: usize, state: ActorState) {
        if actor_id < MAX_ACTORS {
            self.actors.actors[actor_id].state.store(state as u8, Ordering::Relaxed);
            self.actors.actors[actor_id].last_activity_ns.store(get_timestamp_ns(), Ordering::Relaxed);
        }
    }

    /// Increment actor message count
    pub fn increment_actor_messages(&self, actor_id: usize) {
        if actor_id < MAX_ACTORS {
            self.actors.actors[actor_id].messages_processed.fetch_add(1, Ordering::Relaxed);
            self.actors.actors[actor_id].last_activity_ns.store(get_timestamp_ns(), Ordering::Relaxed);
        }
    }

    /// Update actor pending messages
    pub fn set_actor_pending(&self, actor_id: usize, pending: u64) {
        if actor_id < MAX_ACTORS {
            self.actors.actors[actor_id].messages_pending.store(pending, Ordering::Relaxed);
        }
    }

    /// Increment actor error count
    pub fn increment_actor_errors(&self, actor_id: usize) {
        if actor_id < MAX_ACTORS {
            self.actors.actors[actor_id].error_count.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Perform health check
    pub fn check_health(&self) -> HealthResponse {
        let now_ns = get_timestamp_ns();
        self.last_check_ns.store(now_ns, Ordering::Relaxed);
        self.check_count.fetch_add(1, Ordering::Relaxed);

        let uptime_secs = (now_ns - self.start_time_ns.load(Ordering::Relaxed)) / 1_000_000_000;

        // Collect actor states
        let mut actors = Vec::with_capacity(self.actor_count.load(Ordering::Relaxed));
        let mut has_error = false;
        let mut has_blocked = false;

        for i in 0..MAX_ACTORS {
            if self.actors.actors[i].active.load(Ordering::Relaxed) {
                let state_val = self.actors.actors[i].state.load(Ordering::Relaxed);
                let state = match state_val {
                    0 => ActorState::Uninitialized,
                    1 => ActorState::Starting,
                    2 => ActorState::Running,
                    3 => ActorState::Busy,
                    4 => ActorState::Blocked,
                    5 => ActorState::Error,
                    6 => ActorState::Stopping,
                    _ => ActorState::Stopped,
                };

                if state == ActorState::Error {
                    has_error = true;
                }
                if state == ActorState::Blocked {
                    has_blocked = true;
                }

                actors.push(ActorHealth {
                    name: self.actors.actors[i].name,
                    state,
                    messages_processed: self.actors.actors[i].messages_processed.load(Ordering::Relaxed),
                    messages_pending: self.actors.actors[i].messages_pending.load(Ordering::Relaxed),
                    last_activity_ns: self.actors.actors[i].last_activity_ns.load(Ordering::Relaxed),
                    error_count: self.actors.actors[i].error_count.load(Ordering::Relaxed),
                });
            }
        }

        // Determine overall status
        let status = if has_error {
            SystemStatus::Critical
        } else if has_blocked {
            SystemStatus::Degraded
        } else {
            SystemStatus::Healthy
        };

        // Get memory metrics
        let memory = get_memory_metrics();

        // Get system metrics
        let system = get_system_metrics();

        HealthResponse {
            status,
            uptime_secs,
            timestamp_ns: now_ns,
            memory,
            system,
            actors,
            version: self.version,
        }
    }

    /// Get current status without full check
    #[inline]
    pub fn get_status(&self) -> SystemStatus {
        let count = self.actor_count.load(Ordering::Relaxed);
        
        for i in 0..count {
            let state = self.actors.actors[i].state.load(Ordering::Relaxed);
            if state == ActorState::Error as u8 {
                return SystemStatus::Critical;
            }
            if state == ActorState::Blocked as u8 {
                return SystemStatus::Degraded;
            }
        }
        
        SystemStatus::Healthy
    }

    /// Get uptime in seconds
    #[inline]
    pub fn get_uptime_secs(&self) -> u64 {
        let now_ns = get_timestamp_ns();
        (now_ns - self.start_time_ns.load(Ordering::Relaxed)) / 1_000_000_000
    }

    /// Get check count
    #[inline]
    pub fn get_check_count(&self) -> u64 {
        self.check_count.load(Ordering::Relaxed)
    }

    /// Enable health checks
    pub fn enable(&self) {
        self.enabled.store(true, Ordering::Relaxed);
    }

    /// Disable health checks
    pub fn disable(&self) {
        self.enabled.store(false, Ordering::Relaxed);
    }

    /// Check if enabled
    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }
}

#[inline]
fn get_timestamp_ns() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64
}

#[cfg(unix)]
fn get_memory_metrics() -> Option<MemoryMetrics> {
    // Try to read from /proc/self/status on Linux
    #[cfg(target_os = "linux")]
    {
        use std::fs;
        
        if let Ok(status) = fs::read_to_string("/proc/self/status") {
            let mut vm_rss = 0u64;
            let mut vm_size = 0u64;
            
            for line in status.lines() {
                if line.starts_with("VmRSS:") {
                    vm_rss = parse_proc_value(line).unwrap_or(0) * 1024; // Convert KB to bytes
                }
                if line.starts_with("VmSize:") {
                    vm_size = parse_proc_value(line).unwrap_or(0) * 1024;
                }
            }
            
            return Some(MemoryMetrics {
                heap_allocated: 0, // Would need allocator stats
                heap_used: 0,
                stack_size: 8 * 1024 * 1024, // Default assumption
                rss_bytes: vm_rss,
                vms_bytes: vm_size,
            });
        }
    }
    
    None
}

#[cfg(not(unix))]
fn get_memory_metrics() -> Option<MemoryMetrics> {
    None
}

#[cfg(target_os = "linux")]
fn parse_proc_value(line: &str) -> Option<u64> {
    line.split_whitespace().nth(1)?.parse().ok()
}

#[cfg(unix)]
fn get_system_metrics() -> Option<SystemHealth> {
    // Simplified - would need proper system stats collection
    Some(SystemHealth {
        cpu_usage_pct: 0,
        memory_usage_pct: 0,
        network_latency_us: 0,
        disk_latency_us: 0,
    })
}

#[cfg(not(unix))]
fn get_system_metrics() -> Option<SystemHealth> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_check_basic() {
        let engine = HealthCheckEngine::new("1.0.0");
        
        let response = engine.check_health();
        assert_eq!(response.status, SystemStatus::Healthy);
        assert_eq!(response.version, "1.0.0");
    }

    #[test]
    fn test_actor_registration() {
        let engine = HealthCheckEngine::new("1.0.0");
        
        let actor_id = engine.register_actor("TestActor");
        assert!(actor_id.is_some());
        
        engine.update_actor_state(actor_id.unwrap(), ActorState::Running);
        
        let response = engine.check_health();
        assert_eq!(response.actors.len(), 1);
        assert_eq!(response.actors[0].state, ActorState::Running);
    }

    #[test]
    fn test_error_status() {
        let engine = HealthCheckEngine::new("1.0.0");
        
        let actor_id = engine.register_actor("FailingActor").unwrap();
        engine.update_actor_state(actor_id, ActorState::Error);
        
        assert_eq!(engine.get_status(), SystemStatus::Critical);
    }
}
