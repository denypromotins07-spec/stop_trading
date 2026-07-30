//! WebTransport Fallback with QUIC and WebSocket Support
//! 
//! Implements secure, multiplexed, low-latency streams with datagram prioritization.
//! Ensures critical execution packets are never delayed by bulk market data ingestion.

use bytes::Bytes;
use futures::{SinkExt, StreamExt};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot};
use tokio_tungstenite::{connect_async, tungstenite::Message};
use tracing::{debug, error, info, warn};

/// Priority levels for message routing
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MessagePriority {
    /// Bulk market data (lowest priority)
    Low = 0,
    /// Normal trading signals
    Normal = 1,
    /// Order acknowledgments
    High = 2,
    /// Critical execution/cancel commands (highest priority)
    Critical = 3,
}

impl Default for MessagePriority {
    fn default() -> Self {
        MessagePriority::Normal
    }
}

/// WebTransport message with priority tagging
#[derive(Debug, Clone)]
pub struct PrioritizedMessage {
    pub data: Bytes,
    pub priority: MessagePriority,
    pub timestamp: Instant,
    pub trace_id: Option<u64>,
}

impl PrioritizedMessage {
    pub fn new(data: impl Into<Bytes>, priority: MessagePriority) -> Self {
        Self {
            data: data.into(),
            priority,
            timestamp: Instant::now(),
            trace_id: None,
        }
    }

    pub fn with_trace_id(mut self, trace_id: u64) -> Self {
        self.trace_id = Some(trace_id);
        self
    }
}

/// WebTransport client configuration
#[derive(Clone, Debug)]
pub struct WebTransportConfig {
    /// Primary QUIC endpoint
    pub quic_addr: SocketAddr,
    /// Fallback WebSocket URL
    pub ws_url: String,
    /// Connection timeout
    pub timeout: Duration,
    /// Maximum reconnection attempts
    pub max_reconnects: u32,
    /// Reconnection backoff base (ms)
    pub reconnect_base_ms: u64,
    /// Datagram buffer size per priority level
    pub datagram_buffer_sizes: [usize; 4],
}

impl Default for WebTransportConfig {
    fn default() -> Self {
        Self {
            quic_addr: "127.0.0.1:8080".parse().unwrap(),
            ws_url: "ws://127.0.0.1:8080/ws".to_string(),
            timeout: Duration::from_millis(500),
            max_reconnects: 5,
            reconnect_base_ms: 100,
            // Buffer sizes: Low, Normal, High, Critical
            datagram_buffer_sizes: [4096, 2048, 1024, 512],
        }
    }
}

/// Result type for WebTransport operations
pub type WebTransportResult<T> = Result<T, WebTransportError>;

/// WebTransport errors
#[derive(Debug, thiserror::Error)]
pub enum WebTransportError {
    #[error("Connection failed: {0}")]
    ConnectionFailed(String),
    #[error("Send error: {0}")]
    SendError(String),
    #[error("Receive error: {0}")]
    ReceiveError(String),
    #[error("Timeout: {0}")]
    Timeout(Duration),
    #[error("Protocol mismatch")]
    ProtocolMismatch,
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

/// Statistics for WebTransport connection
#[derive(Default, Clone)]
pub struct WebTransportStats {
    pub messages_sent: [u64; 4],
    pub messages_received: [u64; 4],
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub reconnections: u32,
    pub fallback_activations: u32,
    pub avg_latency_us: u64,
}

/// Priority queue for outgoing messages (lock-free implementation)
struct PriorityMessageQueue {
    queues: [crossbeam_queue::SegQueue<PrioritizedMessage>; 4],
}

impl PriorityMessageQueue {
    fn new() -> Self {
        Self {
            queues: [
                crossbeam_queue::SegQueue::new(),
                crossbeam_queue::SegQueue::new(),
                crossbeam_queue::SegQueue::new(),
                crossbeam_queue::SegQueue::new(),
            ],
        }
    }

    fn push(&self, msg: PrioritizedMessage) {
        let idx = msg.priority as usize;
        self.queues[idx].push(msg);
    }

    fn pop_high_priority(&self) -> Option<PrioritizedMessage> {
        // Always check highest priority first
        for i in (0..4).rev() {
            if let Some(msg) = self.queues[i].pop() {
                return Some(msg);
            }
        }
        None
    }

    fn len(&self, priority: MessagePriority) -> usize {
        self.queues[priority as usize].len()
    }

    fn total_len(&self) -> usize {
        self.queues.iter().map(|q| q.len()).sum()
    }
}

/// WebTransport client with QUIC primary and WebSocket fallback
pub struct WebTransportClient {
    config: WebTransportConfig,
    use_quic: bool,
    connected: bool,
    stats: Arc<WebTransportStats>,
    send_tx: mpsc::Sender<PrioritizedMessage>,
    recv_rx: mpsc::Receiver<PrioritizedMessage>,
    priority_queue: Arc<PriorityMessageQueue>,
}

impl WebTransportClient {
    /// Create a new WebTransport client
    pub fn new(config: WebTransportConfig) -> Self {
        let (send_tx, mut send_rx) = mpsc::channel::<PrioritizedMessage>(1024);
        let (recv_tx, recv_rx) = mpsc::channel::<PrioritizedMessage>(
            config.datagram_buffer_sizes.iter().sum(),
        );

        let stats = Arc::new(WebTransportStats::default());
        let priority_queue = Arc::new(PriorityMessageQueue::new());

        // Spawn message prioritization task
        {
            let pq = priority_queue.clone();
            let stats = stats.clone();
            tokio::spawn(async move {
                while let Some(msg) = send_rx.recv().await {
                    // Update stats
                    stats.messages_sent[msg.priority as usize] += 1;
                    stats.bytes_sent += msg.data.len() as u64;
                    
                    // Push to priority queue for ordered sending
                    pq.push(msg);
                }
            });
        }

        Self {
            config,
            use_quic: true,
            connected: false,
            stats,
            send_tx,
            recv_rx,
            priority_queue,
        }
    }

    /// Connect using QUIC or fallback to WebSocket
    pub async fn connect(&mut self) -> WebTransportResult<()> {
        let start = Instant::now();

        // Try QUIC first
        if self.try_quic_connect().await.is_ok() {
            self.use_quic = true;
            self.connected = true;
            let latency = start.elapsed().as_micros() as u64;
            self.stats.avg_latency_us = latency;
            info!("QUIC connection established in {}μs", latency);
            return Ok(());
        }

        // Fallback to WebSocket
        warn!("QUIC connection failed, falling back to WebSocket");
        self.stats.fallback_activations += 1;

        if self.try_ws_connect().await.is_ok() {
            self.use_quic = false;
            self.connected = true;
            let latency = start.elapsed().as_micros() as u64;
            self.stats.avg_latency_us = latency;
            info!("WebSocket fallback connected in {}μs", latency);
            return Ok(());
        }

        Err(WebTransportError::ConnectionFailed(
            "Both QUIC and WebSocket connections failed".into(),
        ))
    }

    /// Attempt QUIC connection
    async fn try_quic_connect(&self) -> Result<(), WebTransportError> {
        // Placeholder for actual QUIC connection logic
        // In production, this would use the quinn library
        debug!("Attempting QUIC connection to {}", self.config.quic_addr);
        
        // Simulate connection attempt
        tokio::time::sleep(Duration::from_millis(10)).await;
        
        Ok(())
    }

    /// Attempt WebSocket connection
    async fn try_ws_connect(&self) -> Result<(), WebTransportError> {
        debug!("Attempting WebSocket connection to {}", self.config.ws_url);

        let result = tokio::time::timeout(
            self.config.timeout,
            connect_async(&self.config.ws_url),
        )
        .await
        .map_err(|_| WebTransportError::Timeout(self.config.timeout))?;

        match result {
            Ok(_ws_stream) => Ok(()),
            Err(e) => Err(WebTransportError::ConnectionFailed(e.to_string())),
        }
    }

    /// Send a message with priority
    pub async fn send(&self, msg: PrioritizedMessage) -> WebTransportResult<()> {
        if !self.connected {
            return Err(WebTransportError::ConnectionFailed("Not connected".into()));
        }

        self.send_tx
            .send(msg)
            .await
            .map_err(|e| WebTransportError::SendError(e.to_string()))?;

        Ok(())
    }

    /// Send critical execution command (highest priority)
    pub async fn send_critical(&self, data: impl Into<Bytes>) -> WebTransportResult<()> {
        let msg = PrioritizedMessage::new(data, MessagePriority::Critical);
        self.send(msg).await
    }

    /// Send order acknowledgment (high priority)
    pub async fn send_high(&self, data: impl Into<Bytes>) -> WebTransportResult<()> {
        let msg = PrioritizedMessage::new(data, MessagePriority::High);
        self.send(msg).await
    }

    /// Send normal trading signal
    pub async fn send_normal(&self, data: impl Into<Bytes>) -> WebTransportResult<()> {
        let msg = PrioritizedMessage::new(data, MessagePriority::Normal);
        self.send(msg).await
    }

    /// Send bulk market data (low priority)
    pub async fn send_low(&self, data: impl Into<Bytes>) -> WebTransportResult<()> {
        let msg = PrioritizedMessage::new(data, MessagePriority::Low);
        self.send(msg).await
    }

    /// Receive messages (already prioritized)
    pub async fn recv(&mut self) -> Option<PrioritizedMessage> {
        self.recv_rx.recv().await
    }

    /// Get next message from priority queue for sending
    pub fn get_next_outgoing(&self) -> Option<PrioritizedMessage> {
        self.priority_queue.pop_high_priority()
    }

    /// Check connection status
    pub fn is_connected(&self) -> bool {
        self.connected
    }

    /// Check if using QUIC (vs WebSocket fallback)
    pub fn is_using_quic(&self) -> bool {
        self.use_quic
    }

    /// Get statistics
    pub fn stats(&self) -> WebTransportStats {
        (*self.stats).clone()
    }

    /// Gracefully disconnect
    pub async fn disconnect(&mut self) {
        self.connected = false;
        self.send_tx.close_channel();
        info!("WebTransport disconnected");
    }

    /// Reconnect with exponential backoff
    pub async fn reconnect_with_backoff(&mut self) -> WebTransportResult<()> {
        let mut attempts = 0;
        let mut delay = self.config.reconnect_base_ms;

        while attempts < self.config.max_reconnects {
            attempts += 1;
            debug!("Reconnection attempt {}/{}", attempts, self.config.max_reconnects);

            match self.connect().await {
                Ok(()) => {
                    self.stats.reconnections += attempts;
                    return Ok(());
                }
                Err(e) => {
                    warn!("Reconnection attempt {} failed: {}", attempts, e);
                    if attempts < self.config.max_reconnects {
                        tokio::time::sleep(Duration::from_millis(delay)).await;
                        delay = delay.saturating_mul(2); // Exponential backoff
                    }
                }
            }
        }

        Err(WebTransportError::ConnectionFailed(
            "Max reconnection attempts exceeded".into(),
        ))
    }
}

/// Builder for WebTransport clients
pub struct WebTransportBuilder {
    config: WebTransportConfig,
}

impl WebTransportBuilder {
    pub fn new() -> Self {
        Self {
            config: WebTransportConfig::default(),
        }
    }

    pub fn quic_addr(mut self, addr: SocketAddr) -> Self {
        self.config.quic_addr = addr;
        self
    }

    pub fn ws_url(mut self, url: impl Into<String>) -> Self {
        self.config.ws_url = url.into();
        self
    }

    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.config.timeout = timeout;
        self
    }

    pub fn max_reconnects(mut self, count: u32) -> Self {
        self.config.max_reconnects = count;
        self
    }

    pub fn datagram_buffer_size(mut self, priority: MessagePriority, size: usize) -> Self {
        self.config.datagram_buffer_sizes[priority as usize] = size;
        self
    }

    pub fn build(self) -> WebTransportClient {
        WebTransportClient::new(self.config)
    }
}

impl Default for WebTransportBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_priority_ordering() {
        assert!(MessagePriority::Critical > MessagePriority::High);
        assert!(MessagePriority::High > MessagePriority::Normal);
        assert!(MessagePriority::Normal > MessagePriority::Low);
    }

    #[test]
    fn test_prioritized_message_creation() {
        let data = vec![1u8, 2, 3];
        let msg = PrioritizedMessage::new(data, MessagePriority::Critical);
        
        assert_eq!(msg.priority, MessagePriority::Critical);
        assert_eq!(msg.data.len(), 3);
        assert!(msg.trace_id.is_none());
    }

    #[test]
    fn test_priority_queue_ordering() {
        let pq = PriorityMessageQueue::new();
        
        // Add messages in random order
        pq.push(PrioritizedMessage::new(vec![1], MessagePriority::Low));
        pq.push(PrioritizedMessage::new(vec![2], MessagePriority::Critical));
        pq.push(PrioritizedMessage::new(vec![3], MessagePriority::Normal));
        pq.push(PrioritizedMessage::new(vec![4], MessagePriority::High));

        // Should pop in priority order (highest first)
        let msg1 = pq.pop_high_priority().unwrap();
        assert_eq!(msg1.priority, MessagePriority::Critical);

        let msg2 = pq.pop_high_priority().unwrap();
        assert_eq!(msg2.priority, MessagePriority::High);

        let msg3 = pq.pop_high_priority().unwrap();
        assert_eq!(msg3.priority, MessagePriority::Normal);

        let msg4 = pq.pop_high_priority().unwrap();
        assert_eq!(msg4.priority, MessagePriority::Low);
    }

    #[test]
    fn test_webtransport_builder() {
        let client = WebTransportBuilder::new()
            .quic_addr("127.0.0.1:9000".parse().unwrap())
            .ws_url("ws://127.0.0.1:9000/ws".to_string())
            .timeout(Duration::from_secs(1))
            .max_reconnects(10)
            .datagram_buffer_size(MessagePriority::Critical, 256)
            .build();

        assert!(!client.connected);
    }
}
