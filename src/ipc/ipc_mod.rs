//! IPC Module Root
//! 
//! Manages shared memory lifecycle and ZeroMQ fallback connections.

pub mod shm_ring;
pub mod feature_sync;

pub use shm_ring::{ShmRingBuffer, FeatureElement, FeatureBatch, RingSlot};
pub use feature_sync::{FeatureSyncDaemon, SyncState, SyncStats};

/// IPC configuration
#[derive(Debug, Clone)]
pub struct IpcConfig {
    pub shm_enabled: bool,
    pub zeromq_fallback: bool,
    pub zeromq_endpoint: String,
    pub buffer_size: usize,
}

impl Default for IpcConfig {
    fn default() -> Self {
        Self {
            shm_enabled: true,
            zeromq_fallback: true,
            zeromq_endpoint: "tcp://127.0.0.1:5555".to_string(),
            buffer_size: 16384,
        }
    }
}

/// IPC manager coordinating all communication channels
pub struct IpcManager {
    config: IpcConfig,
    ring_buffer: ShmRingBuffer,
    sync_daemon: FeatureSyncDaemon,
}

impl IpcManager {
    pub fn new(config: IpcConfig) -> Self {
        Self {
            config,
            ring_buffer: ShmRingBuffer::new(),
            sync_daemon: FeatureSyncDaemon::new(),
        }
    }
    
    pub fn send_feature(&self, element: FeatureElement) -> bool {
        let seq = self.sync_daemon.rust_produce();
        let mut elem = element;
        // Embed sequence in timestamp for tracking
        elem.timestamp_ns = elem.timestamp_ns | (seq << 40);
        self.ring_buffer.push(elem)
    }
    
    pub fn receive_feature(&self) -> Option<FeatureElement> {
        if let Some(elem) = self.ring_buffer.pop() {
            self.sync_daemon.record_sync();
            Some(elem)
        } else {
            None
        }
    }
    
    pub fn get_sync_state(&self) -> SyncState {
        self.sync_daemon.get_sync_state()
    }
    
    pub fn shutdown(&self) {
        self.ring_buffer.close();
        self.sync_daemon.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_ipc_manager() {
        let manager = IpcManager::new(IpcConfig::default());
        
        let elem = FeatureElement {
            value: 1.0,
            timestamp_ns: 1000,
            feature_id: 1,
            valid: true,
        };
        
        assert!(manager.send_feature(elem));
        
        let received = manager.receive_feature();
        assert!(received.is_some());
        assert_eq!(received.unwrap().value, 1.0);
    }
}
