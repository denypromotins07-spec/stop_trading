//! IPC Module Root
//! 
//! Manages connection lifecycles, heartbeat checks, and serialization of feature arrays.
//! Re-exports shared memory and ZMQ bridge components.

pub mod shared_memory;
pub mod zmq_bridge;

use std::{
    io,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    thread,
    time::{Duration, Instant},
};

use crossbeam_channel::{bounded, Receiver, Sender};

use crate::ipc::shared_memory::SharedMemoryManager;
use crate::ipc::zmq_bridge::{ZmqBridge, WeightUpdate};

/// IPC Manager coordinating all inter-process communication
pub struct IpcManager {
    shared_memory: Option<SharedMemoryManager>,
    zmq_bridge: Option<ZmqBridge>,
    running: Arc<AtomicBool>,
    heartbeat_interval_ms: u64,
    last_heartbeat: Arc<AtomicU64>,
    connection_errors: Arc<AtomicU64>,
}

unsafe impl Send for IpcManager {}
unsafe impl Sync for IpcManager {}

impl IpcManager {
    /// Create a new IPC manager
    pub fn new() -> Self {
        Self {
            shared_memory: None,
            zmq_bridge: None,
            running: Arc::new(AtomicBool::new(false)),
            heartbeat_interval_ms: 100, // 100ms default
            last_heartbeat: Arc::new(AtomicU64::new(0)),
            connection_errors: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Initialize shared memory segment
    pub fn init_shared_memory(&mut self, name: &str) -> io::Result<()> {
        let shmem = SharedMemoryManager::create(name)?;
        self.shared_memory = Some(shmem);
        Ok(())
    }

    /// Open existing shared memory segment
    pub fn open_shared_memory(&mut self, name: &str) -> io::Result<()> {
        let shmem = SharedMemoryManager::open(name)?;
        self.shared_memory = Some(shmem);
        Ok(())
    }

    /// Initialize ZMQ bridge with default endpoints
    pub fn init_zmq_bridge(&mut self) -> io::Result<()> {
        let mut bridge = ZmqBridge::new()?;
        
        // Start publisher for broadcasting to Python
        bridge.start_publisher(zmq_bridge::DEFAULT_PUB_ENDPOINT)?;
        
        // Start requester for sending inference requests
        bridge.start_requester(zmq_bridge::DEFAULT_REQ_ENDPOINT)?;
        
        self.zmq_bridge = Some(bridge);
        Ok(())
    }

    /// Start the IPC manager and background tasks
    pub fn start(&mut self) -> io::Result<()> {
        self.running.store(true, Ordering::Release);
        
        // Start heartbeat thread
        let running = self.running.clone();
        let last_hb = self.last_heartbeat.clone();
        let interval = self.heartbeat_interval_ms;
        
        thread::spawn(move || {
            while running.load(Ordering::Acquire) {
                let now = get_timestamp_ms();
                last_hb.store(now, Ordering::Release);
                thread::sleep(Duration::from_millis(interval));
            }
        });

        Ok(())
    }

    /// Write features to shared memory
    pub fn write_features(
        &self,
        symbol_id: u64,
        features: &[f32],
        timestamp_ns: u64,
        feature_flags: u64,
    ) -> io::Result<u64> {
        if let Some(ref shmem) = self.shared_memory {
            shmem.write_features(symbol_id, features, timestamp_ns, feature_flags)
        } else {
            Err(io::Error::new(
                io::ErrorKind::NotConnected,
                "Shared memory not initialized",
            ))
        }
    }

    /// Broadcast features via ZMQ
    pub fn broadcast_features(
        &self,
        symbol: &str,
        features: &[f32],
        feature_flags: u64,
    ) -> io::Result<()> {
        if let Some(ref bridge) = self.zmq_bridge {
            bridge.broadcast_features(symbol, features, feature_flags)
        } else {
            Err(io::Error::new(
                io::ErrorKind::NotConnected,
                "ZMQ bridge not initialized",
            ))
        }
    }

    /// Send inference request to Python
    pub fn request_inference(&self, symbol: &str, features: &[f32]) -> io::Result<u64> {
        if let Some(ref bridge) = self.zmq_bridge {
            bridge.send_inference_request(symbol, features)
        } else {
            Err(io::Error::new(
                io::ErrorKind::NotConnected,
                "ZMQ bridge not initialized",
            ))
        }
    }

    /// Receive inference response from Python
    pub fn receive_inference(&self) -> io::Result<Option<zmq_bridge::InferenceResult>> {
        if let Some(ref bridge) = self.zmq_bridge {
            bridge.receive_inference_response()
        } else {
            Err(io::Error::new(
                io::ErrorKind::NotConnected,
                "ZMQ bridge not initialized",
            ))
        }
    }

    /// Send weight update to Rust execution engine
    pub fn send_weight_update(&self, update: &WeightUpdate) -> io::Result<()> {
        if let Some(ref bridge) = self.zmq_bridge {
            bridge.send_weight_update(update)
        } else {
            Err(io::Error::new(
                io::ErrorKind::NotConnected,
                "ZMQ bridge not initialized",
            ))
        }
    }

    /// Check if IPC is healthy
    pub fn is_healthy(&self) -> bool {
        let now = get_timestamp_ms();
        let last_hb = self.last_heartbeat.load(Ordering::Relaxed);
        
        // Consider unhealthy if no heartbeat in 500ms
        now - last_hb < 500 && self.running.load(Ordering::Acquire)
    }

    /// Get connection error count
    pub fn get_error_count(&self) -> u64 {
        self.connection_errors.load(Ordering::Relaxed)
    }

    /// Record a connection error
    pub fn record_error(&self) {
        self.connection_errors.fetch_add(1, Ordering::Relaxed);
    }

    /// Shutdown IPC manager
    pub fn shutdown(&mut self) {
        self.running.store(false, Ordering::Release);
        
        if let Some(ref mut bridge) = self.zmq_bridge {
            bridge.shutdown();
        }
        
        self.zmq_bridge = None;
        self.shared_memory = None;
    }

    /// Set heartbeat interval
    pub fn set_heartbeat_interval(&mut self, interval_ms: u64) {
        self.heartbeat_interval_ms = interval_ms;
    }
}

impl Default for IpcManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Get current timestamp in milliseconds
fn get_timestamp_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

/// Feature serialization helper using bincode for ultra-fast serialization
pub mod serialization {
    use bincode::Options;

    /// Serialize feature vector with minimal overhead
    pub fn serialize_features(features: &[f32]) -> Result<Vec<u8>, bincode::Error> {
        bincode::options()
            .with_fixint_encoding()
            .allow_trailing_bytes()
            .serialize(features)
    }

    /// Deserialize feature vector
    pub fn deserialize_features(data: &[u8]) -> Result<Vec<f32>, bincode::Error> {
        bincode::options()
            .with_fixint_encoding()
            .allow_trailing_bytes()
            .deserialize(data)
    }

    /// Zero-copy serialization using rkyv (when available)
    #[cfg(feature = "rkyv")]
    pub mod rkyv_serialization {
        use rkyv::{archive_bytes, deserialize_bytes, ser::Serializer};

        pub fn serialize_zero_copy(features: &[f32]) -> Result<Vec<u8>, String> {
            let mut serializer = Serializer::default();
            serializer.serialize_value(features)
                .map_err(|e| e.to_string())?;
            
            let pos = serializer.pos();
            Ok(serializer.into_serializer().into_inner()[..pos].to_vec())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ipc_manager_creation() {
        let manager = IpcManager::new();
        assert!(!manager.is_healthy());
        assert_eq!(manager.get_error_count(), 0);
    }

    #[test]
    fn test_ipc_manager_error_recording() {
        let manager = IpcManager::new();
        
        manager.record_error();
        manager.record_error();
        
        assert_eq!(manager.get_error_count(), 2);
    }

    #[test]
    fn test_feature_serialization() {
        let features = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        
        let serialized = serialization::serialize_features(&features).unwrap();
        let deserialized = serialization::deserialize_features(&serialized).unwrap();
        
        assert_eq!(features, deserialized);
    }

    #[test]
    fn test_timestamp_functions() {
        let ts1 = get_timestamp_ms();
        thread::sleep(Duration::from_millis(10));
        let ts2 = get_timestamp_ms();
        
        assert!(ts2 > ts1);
        assert!(ts2 - ts1 >= 10);
    }
}
