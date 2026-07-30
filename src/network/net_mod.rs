//! Network Module Root
//! 
//! Abstracts the receive layer to allow future drop-in hardware DPDK integration.
//! Provides unified interface for io_uring (Linux) and DPDK-style fallback (Windows).

#[cfg(target_os = "linux")]
pub mod io_uring_impl;

#[cfg(not(target_os = "linux"))]
pub mod dpdk_fallback;

// Re-export based on platform
#[cfg(target_os = "linux")]
pub use io_uring_impl::{
    IoUringReceiver, IoUringBuffer, IoUringStats, IoUringBatchProcessor,
    register_memory_for_io_uring, configure_socket_for_io_uring,
};

#[cfg(not(target_os = "linux"))]
pub use dpdk_fallback::{
    BatchReceiver, NetworkPacket, BatchStats, PacketRingBuffer,
    configure_socket_low_latency,
};

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::io;

/// Unified network receiver trait for abstraction
pub trait NetworkReceiver: Send + Sync {
    /// Initialize the receiver
    fn initialize(&mut self) -> io::Result<()>;
    
    /// Receive a batch of packets
    fn receive_batch(&mut self) -> io::Result<usize>;
    
    /// Get statistics
    fn get_stats(&self) -> NetworkStats;
    
    /// Enable/disable receiver
    fn set_enabled(&self, enabled: bool);
    
    /// Shutdown receiver
    fn shutdown(&mut self);
}

/// Unified network statistics
#[derive(Debug, Clone, Default)]
pub struct NetworkStats {
    pub packets_received: u64,
    pub bytes_received: u64,
    pub batches_processed: u64,
    pub errors: u64,
    pub enabled: bool,
}

/// Platform-specific receiver wrapper
pub enum PlatformReceiver {
    #[cfg(target_os = "linux")]
    IoUring(IoUringReceiver),
    
    #[cfg(not(target_os = "linux"))]
    Batch(BatchReceiver),
}

impl PlatformReceiver {
    /// Create new platform-appropriate receiver
    pub fn new(batch_size: usize) -> io::Result<Self> {
        #[cfg(target_os = "linux")]
        {
            let receiver = IoUringReceiver::new()?;
            Ok(PlatformReceiver::IoUring(receiver))
        }
        
        #[cfg(not(target_os = "linux"))]
        {
            let receiver = BatchReceiver::new(batch_size)?;
            Ok(PlatformReceiver::Batch(receiver))
        }
    }
    
    /// Initialize the receiver
    pub fn initialize(&mut self) -> io::Result<()> {
        match self {
            #[cfg(target_os = "linux")]
            PlatformReceiver::IoUring(r) => r.initialize(),
            
            #[cfg(not(target_os = "linux"))]
            PlatformReceiver::Batch(r) => r.initialize(),
        }
    }
    
    /// Get statistics
    pub fn get_stats(&self) -> NetworkStats {
        match self {
            #[cfg(target_os = "linux")]
            PlatformReceiver::IoUring(r) => {
                let stats = r.get_stats();
                NetworkStats {
                    packets_received: stats.packets_received,
                    bytes_received: stats.bytes_received,
                    batches_processed: 0,
                    errors: 0,
                    enabled: stats.enabled,
                }
            }
            
            #[cfg(not(target_os = "linux"))]
            PlatformReceiver::Batch(r) => {
                let stats = r.get_stats();
                NetworkStats {
                    packets_received: stats.packets_received,
                    bytes_received: stats.bytes_received,
                    batches_processed: stats.batches_processed,
                    errors: 0,
                    enabled: stats.enabled,
                }
            }
        }
    }
    
    /// Enable/disable receiver
    pub fn set_enabled(&self, enabled: bool) {
        match self {
            #[cfg(target_os = "linux")]
            PlatformReceiver::IoUring(r) => r.enabled.store(enabled, Ordering::Release),
            
            #[cfg(not(target_os = "linux"))]
            PlatformReceiver::Batch(r) => r.set_enabled(enabled),
        }
    }
    
    /// Shutdown receiver
    pub fn shutdown(&mut self) {
        match self {
            #[cfg(target_os = "linux")]
            PlatformReceiver::IoUring(r) => r.shutdown(),
            
            #[cfg(not(target_os = "linux"))]
            PlatformReceiver::Batch(r) => r.shutdown(),
        }
    }
}

/// Main Network Manager
/// Coordinates network reception across platforms
pub struct NetworkManager {
    /// Platform-specific receiver
    receiver: Option<PlatformReceiver>,
    /// Manager enabled flag
    enabled: AtomicBool,
    /// Total packets processed
    total_packets: AtomicU64,
    /// Total bytes processed
    total_bytes: AtomicU64,
}

impl NetworkManager {
    pub fn new() -> io::Result<Self> {
        let receiver = PlatformReceiver::new(64)?;
        
        Ok(Self {
            receiver: Some(receiver),
            enabled: AtomicBool::new(false),
            total_packets: AtomicU64::new(0),
            total_bytes: AtomicU64::new(0),
        })
    }
    
    /// Initialize the network manager
    pub fn initialize(&mut self) -> io::Result<()> {
        if let Some(ref mut receiver) = self.receiver {
            receiver.initialize()?;
            self.enabled.store(true, Ordering::Release);
        }
        Ok(())
    }
    
    /// Get receiver reference
    pub fn receiver(&self) -> Option<&PlatformReceiver> {
        self.receiver.as_ref()
    }
    
    /// Get mutable receiver reference
    pub fn receiver_mut(&mut self) -> Option<&mut PlatformReceiver> {
        self.receiver.as_mut()
    }
    
    /// Update statistics from receiver
    pub fn update_stats(&self) -> NetworkStats {
        if let Some(ref receiver) = self.receiver {
            let mut stats = receiver.get_stats();
            stats.packets_received = self.total_packets.load(Ordering::Relaxed);
            stats.bytes_received = self.total_bytes.load(Ordering::Relaxed);
            stats.enabled = self.enabled.load(Ordering::Acquire);
            stats
        } else {
            NetworkStats::default()
        }
    }
    
    /// Enable/disable network processing
    #[inline]
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Release);
        if let Some(ref receiver) = self.receiver {
            receiver.set_enabled(enabled);
        }
    }
    
    /// Shutdown network manager
    pub fn shutdown(&mut self) {
        self.enabled.store(false, Ordering::Release);
        if let Some(ref mut receiver) = self.receiver {
            receiver.shutdown();
        }
    }
}

impl Default for NetworkManager {
    fn default() -> Self {
        Self::new().expect("Failed to create network manager")
    }
}

/// Socket options for low-latency networking
#[derive(Debug, Clone, Copy)]
pub struct SocketOptions {
    pub tcp_nodelay: bool,
    pub recv_buffer_size: usize,
    pub send_buffer_size: usize,
    pub busy_poll: bool,
    pub reuse_port: bool,
}

impl Default for SocketOptions {
    fn default() -> Self {
        Self {
            tcp_nodelay: true,
            recv_buffer_size: 256 * 1024,
            send_buffer_size: 256 * 1024,
            busy_poll: true,
            reuse_port: true,
        }
    }
}

/// Configure socket with optimal settings for HFT
pub fn configure_socket(socket_fd: i32, options: SocketOptions) -> io::Result<()> {
    #[cfg(target_os = "linux")]
    {
        unsafe {
            use libc::{setsockopt, IPPROTO_TCP, TCP_NODELAY, SOL_SOCKET, SO_RCVBUF, SO_SNDBUF};
            
            if options.tcp_nodelay {
                let nodelay: libc::c_int = 1;
                let _ = setsockopt(socket_fd, IPPROTO_TCP, TCP_NODELAY,
                                  &nodelay as *const _ as *const _, 4);
            }
            
            let recv_buf: libc::c_int = options.recv_buffer_size as libc::c_int;
            let _ = setsockopt(socket_fd, SOL_SOCKET, SO_RCVBUF,
                              &recv_buf as *const _ as *const _, 4);
            
            let send_buf: libc::c_int = options.send_buffer_size as libc::c_int;
            let _ = setsockopt(socket_fd, SOL_SOCKET, SO_SNDBUF,
                              &send_buf as *const _ as *const _, 4);
            
            if options.busy_poll {
                use libc::{SO_BUSY_POLL, SO_PREFER_BUSY_POLL};
                let poll: libc::c_int = 1;
                let _ = setsockopt(socket_fd, SOL_SOCKET, SO_BUSY_POLL,
                                  &poll as *const _ as *const _, 4);
                let _ = setsockopt(socket_fd, SOL_SOCKET, SO_PREFER_BUSY_POLL,
                                  &poll as *const _ as *const _, 4);
            }
            
            if options.reuse_port {
                use libc::SO_REUSEPORT;
                let reuse: libc::c_int = 1;
                let _ = setsockopt(socket_fd, SOL_SOCKET, SO_REUSEPORT,
                                  &reuse as *const _ as *const _, 4);
            }
        }
    }
    
    #[cfg(target_os = "windows")]
    {
        // Windows socket configuration
        configure_socket_low_latency(socket_fd)?;
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_manager_creation() {
        let manager = NetworkManager::new();
        assert!(manager.is_ok());
    }

    #[test]
    fn test_platform_receiver_creation() {
        let receiver = PlatformReceiver::new(64);
        assert!(receiver.is_ok());
    }

    #[test]
    fn test_socket_options_default() {
        let opts = SocketOptions::default();
        assert!(opts.tcp_nodelay);
        assert_eq!(opts.recv_buffer_size, 256 * 1024);
    }
}
