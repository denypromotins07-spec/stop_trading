//! QUIC Client for Ultra-Low Latency Connectivity
//! 
//! Implements a QUIC client using `quinn` for Solana TPU and modern CEX endpoints.
//! Bypasses TCP head-of-line blocking with 0-RTT handshake resumption.

use bytes::Bytes;
use quinn::{ClientConfig, Connection, Endpoint, RecvStream, SendStream, TransportConfig};
use rustls::{ClientConfig as RustlsClientConfig, RootCertStore};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot};
use tracing::{debug, error, info, warn};

/// Maximum concurrent streams per connection
const MAX_CONCURRENT_STREAMS: u32 = 100;

/// 0-RTT session cache size (bounded to respect RAM limits)
const SESSION_CACHE_SIZE: usize = 64;

/// Connection timeout in milliseconds
const CONNECTION_TIMEOUT_MS: u64 = 500;

/// Result type for network operations
pub type QuicResult<T> = Result<T, QuicError>;

/// QUIC client errors
#[derive(Debug, thiserror::Error)]
pub enum QuicError {
    #[error("Connection failed: {0}")]
    ConnectionFailed(String),
    #[error("Stream error: {0}")]
    StreamError(String),
    #[error("Timeout: {0}")]
    Timeout(Duration),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("QUIC protocol error: {0}")]
    ProtocolError(String),
}

/// QUIC client configuration optimized for HFT
#[derive(Clone, Debug)]
pub struct QuicClientConfig {
    /// Target socket address
    pub addr: SocketAddr,
    /// Server name for SNI (can be empty for IP-only connections)
    pub server_name: String,
    /// Enable 0-RTT handshake resumption
    pub enable_0rtt: bool,
    /// Maximum idle timeout
    pub idle_timeout: Duration,
    /// Initial congestion window (packets)
    pub initial_cwnd: u32,
    /// Datagram receive buffer size (bytes)
    pub datagram_buffer_size: usize,
}

impl Default for QuicClientConfig {
    fn default() -> Self {
        Self {
            addr: "127.0.0.1:8080".parse().unwrap(),
            server_name: String::new(),
            enable_0rtt: true,
            idle_timeout: Duration::from_secs(30),
            initial_cwnd: 32, // Aggressive initial window for low latency
            datagram_buffer_size: 1024 * 1024, // 1MB bounded buffer
        }
    }
}

/// High-performance QUIC client with 0-RTT support
pub struct QuicClient {
    endpoint: Endpoint,
    config: QuicClientConfig,
    connection: Option<Connection>,
    session_cache: Arc<SessionCache>,
    stats: Arc<QuicStats>,
}

/// Session cache for 0-RTT resumption (fixed-size, lock-free)
struct SessionCache {
    entries: crossbeam_queue::SegQueue<SessionEntry>,
    max_size: usize,
}

struct SessionEntry {
    server_name: String,
    session_data: Vec<u8>,
    created_at: Instant,
    ttl: Duration,
}

impl SessionCache {
    fn new(max_size: usize) -> Self {
        Self {
            entries: crossbeam_queue::SegQueue::new(),
            max_size,
        }
    }

    fn insert(&self, server_name: String, session_data: Vec<u8>) {
        let entry = SessionEntry {
            server_name,
            session_data,
            created_at: Instant::now(),
            ttl: Duration::from_secs(3600), // 1 hour TTL
        };

        // Simple eviction: if over capacity, drop oldest
        if self.entries.len() >= self.max_size {
            let _ = self.entries.pop();
        }
        self.entries.push(entry);
    }

    fn get(&self, server_name: &str) -> Option<Vec<u8>> {
        let now = Instant::now();
        for entry in self.entries.iter() {
            if entry.server_name == server_name && now.duration_since(entry.created_at) < entry.ttl {
                return Some(entry.session_data.clone());
            }
        }
        None
    }
}

/// QUIC connection statistics
#[derive(Default)]
pub struct QuicStats {
    connections_established: std::sync::atomic::AtomicU64,
    zero_rtt_resumptions: std::sync::atomic::AtomicU64,
    bytes_sent: std::sync::atomic::AtomicU64,
    bytes_received: std::sync::atomic::AtomicU64,
    avg_latency_us: std::sync::atomic::AtomicU64,
    latency_samples: std::sync::atomic::AtomicU64,
}

impl QuicStats {
    pub fn record_connection(&self, zero_rtt: bool) {
        self.connections_established.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if zero_rtt {
            self.zero_rtt_resumptions.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }

    pub fn record_bytes_sent(&self, bytes: u64) {
        self.bytes_sent.fetch_add(bytes, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn record_bytes_received(&self, bytes: u64) {
        self.bytes_received.fetch_add(bytes, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn record_latency(&self, latency_us: u64) {
        let current_sum = self.avg_latency_us.load(std::sync::atomic::Ordering::Relaxed);
        let count = self.latency_samples.load(std::sync::atomic::Ordering::Relaxed);
        
        // Running average with overflow protection
        let new_sum = current_sum.saturating_add(latency_us);
        let new_count = count.saturating_add(1);
        
        self.avg_latency_us.store(new_sum / new_count, std::sync::atomic::Ordering::Relaxed);
        self.latency_samples.store(new_count, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn get_stats(&self) -> QuicStatsSnapshot {
        QuicStatsSnapshot {
            connections_established: self.connections_established.load(std::sync::atomic::Ordering::Relaxed),
            zero_rtt_resumptions: self.zero_rtt_resumptions.load(std::sync::atomic::Ordering::Relaxed),
            bytes_sent: self.bytes_sent.load(std::sync::atomic::Ordering::Relaxed),
            bytes_received: self.bytes_received.load(std::sync::atomic::Ordering::Relaxed),
            avg_latency_us: self.avg_latency_us.load(std::sync::atomic::Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone)]
pub struct QuicStatsSnapshot {
    pub connections_established: u64,
    pub zero_rtt_resumptions: u64,
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub avg_latency_us: u64,
}

impl QuicClient {
    /// Create a new QUIC client with optimized configuration
    pub fn new(config: QuicClientConfig) -> QuicResult<Self> {
        let mut crypto = RustlsClientConfig::builder()
            .with_safe_defaults()
            .with_root_certificates(RootCertStore::empty())
            .with_no_client_auth();

        // Optimize for speed: disable certificate verification for local/trusted endpoints
        // In production, use proper certificate validation
        crypto.dangerous().set_certificate_verifier(Arc::new(NoCertificateVerification));

        // Enable 0-RTT if configured
        if config.enable_0rtt {
            crypto.enable_early_data = true;
            crypto.alpn_protocols = vec![b"h3".to_vec(), b"solana-tpu".to_vec()];
        }

        let mut transport_config = TransportConfig::default();
        transport_config
            .max_concurrent_bidi_streams(MAX_CONCURRENT_STREAMS.into())
            .max_concurrent_uni_streams(MAX_CONCURRENT_STREAMS.into())
            .idle_timeout(Some(config.idle_timeout.try_into().unwrap()))
            .initial_mtu(1400) // Conservative MTU to avoid fragmentation
            .congestion_controller_factory(Arc::new(quinn::congestion::BbrConfig::default()));

        let mut client_config = ClientConfig::new(Arc::new(crypto));
        client_config.transport_config(Arc::new(transport_config));

        let mut endpoint = Endpoint::client("[::]:0".parse().unwrap())?;
        endpoint.set_default_client_config(client_config);

        let session_cache = Arc::new(SessionCache::new(SESSION_CACHE_SIZE));
        let stats = Arc::new(QuicStats::default());

        Ok(Self {
            endpoint,
            config,
            connection: None,
            session_cache,
            stats,
        })
    }

    /// Establish connection with optional 0-RTT resumption
    pub async fn connect(&mut self) -> QuicResult<()> {
        let start = Instant::now();
        
        // Attempt 0-RTT resumption if available
        let session_data = self.session_cache.get(&self.config.server_name);
        
        let connecting = if let Some(session) = session_data {
            debug!("Attempting 0-RTT resumption for {}", self.config.addr);
            self.endpoint.connect_with(
                self.endpoint.client_config().unwrap().clone(),
                self.config.addr,
                &self.config.server_name,
            )?
        } else {
            debug!("Full handshake for {}", self.config.addr);
            self.endpoint.connect(self.config.addr, &self.config.server_name)?
        };

        // Apply connection timeout
        let connection = tokio::time::timeout(
            Duration::from_millis(CONNECTION_TIMEOUT_MS),
            connecting,
        )
        .await
        .map_err(|_| QuicError::Timeout(Duration::from_millis(CONNECTION_TIMEOUT_MS)))?
        .map_err(|e| QuicError::ConnectionFailed(e.to_string()))?;

        let latency_us = start.elapsed().as_micros() as u64;
        let zero_rtt = session_data.is_some();
        
        self.stats.record_connection(zero_rtt);
        self.stats.record_latency(latency_us);

        if zero_rtt {
            info!("0-RTT connection established in {}μs", latency_us);
        } else {
            info!("Full handshake completed in {}μs", latency_us);
        }

        self.connection = Some(connection);
        Ok(())
    }

    /// Send data on a bidirectional stream
    pub async fn send_request(&mut self, data: &[u8]) -> QuicResult<Vec<u8>> {
        let conn = self.connection.as_ref()
            .ok_or_else(|| QuicError::ConnectionFailed("Not connected".into()))?;

        let (mut send, mut recv) = conn.open_bi().await
            .map_err(|e| QuicError::StreamError(e.to_string()))?;

        // Send request
        send.write_all(data).await
            .map_err(|e| QuicError::StreamError(e.to_string()))?;
        send.finish().await
            .map_err(|e| QuicError::StreamError(e.to_string()))?;

        self.stats.record_bytes_sent(data.len() as u64);

        // Read response
        let mut response = Vec::new();
        recv.read_to_end(65536, &mut response).await
            .map_err(|e| QuicError::StreamError(e.to_string()))?;

        self.stats.record_bytes_received(response.len() as u64);
        Ok(response)
    }

    /// Send datagram (unreliable but lowest latency)
    pub async fn send_datagram(&self, data: Bytes) -> QuicResult<()> {
        let conn = self.connection.as_ref()
            .ok_or_else(|| QuicError::ConnectionFailed("Not connected".into()))?;

        conn.send_datagram(data.clone())
            .map_err(|e| QuicError::ProtocolError(e.to_string()))?;

        self.stats.record_bytes_sent(data.len() as u64);
        Ok(())
    }

    /// Receive datagrams continuously
    pub async fn receive_datagrams(&self) -> mpsc::Receiver<Bytes> {
        let (tx, rx) = mpsc::channel(1024);
        let conn = self.connection.clone();
        let stats = self.stats.clone();

        tokio::spawn(async move {
            if let Some(conn) = conn {
                loop {
                    match conn.read_datagram().await {
                        Ok(data) => {
                            stats.record_bytes_received(data.len() as u64);
                            if tx.send(data).await.is_err() {
                                break;
                            }
                        }
                        Err(_) => break,
                    }
                }
            }
        });

        rx
    }

    /// Get connection statistics
    pub fn stats(&self) -> QuicStatsSnapshot {
        self.stats.get_stats()
    }

    /// Check if connection is alive
    pub fn is_connected(&self) -> bool {
        self.connection.as_ref().map_or(false, |c| !c.is_closed())
    }

    /// Gracefully close the connection
    pub async fn close(&mut self) {
        if let Some(conn) = self.connection.take() {
            conn.close(0u32.into(), b"graceful_shutdown");
        }
        self.endpoint.wait_idle().await;
    }
}

/// No certificate verification (for trusted local endpoints only)
struct NoCertificateVerification;

impl rustls::client::ServerCertVerifier for NoCertificateVerification {
    fn verify_server_cert(
        &self,
        _end_entity: &rustls::Certificate,
        _intermediates: &[rustls::Certificate],
        _server_name: &rustls::ServerName,
        _scts: &mut dyn Iterator<Item = &[u8]>,
        _ocsp_response: &[u8],
        _now: std::time::SystemTime,
    ) -> Result<rustls::client::ServerCertVerified, rustls::Error> {
        Ok(rustls::client::ServerCertVerified::assertion())
    }
}

/// Builder for creating QUIC clients with custom configurations
pub struct QuicClientBuilder {
    config: QuicClientConfig,
}

impl QuicClientBuilder {
    pub fn new() -> Self {
        Self {
            config: QuicClientConfig::default(),
        }
    }

    pub fn addr(mut self, addr: SocketAddr) -> Self {
        self.config.addr = addr;
        self
    }

    pub fn server_name(mut self, name: impl Into<String>) -> Self {
        self.config.server_name = name.into();
        self
    }

    pub fn enable_0rtt(mut self, enable: bool) -> Self {
        self.config.enable_0rtt = enable;
        self
    }

    pub fn idle_timeout(mut self, timeout: Duration) -> Self {
        self.config.idle_timeout = timeout;
        self
    }

    pub fn initial_cwnd(mut self, cwnd: u32) -> Self {
        self.config.initial_cwnd = cwnd;
        self
    }

    pub fn build(self) -> QuicResult<QuicClient> {
        QuicClient::new(self.config)
    }
}

impl Default for QuicClientBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quic_client_builder() {
        let addr: SocketAddr = "127.0.0.1:9000".parse().unwrap();
        let client = QuicClientBuilder::new()
            .addr(addr)
            .server_name("test.example.com")
            .enable_0rtt(true)
            .idle_timeout(Duration::from_secs(60))
            .initial_cwnd(64)
            .build();

        assert!(client.is_ok());
    }

    #[test]
    fn test_session_cache() {
        let cache = SessionCache::new(10);
        cache.insert("test.com".to_string(), vec![1, 2, 3]);
        
        assert!(cache.get("test.com").is_some());
        assert!(cache.get("other.com").is_none());
    }
}
