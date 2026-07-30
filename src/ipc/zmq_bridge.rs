//! ZeroMQ Bridge Module
//! 
//! High-throughput ZeroMQ PUB/SUB and REQ/REP bridge for communication
//! with Python ensemble models. Sub-millisecond latency for inference signals
//! and dynamic weight updates.

use std::{
    io,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use crossbeam_channel::{bounded, Receiver, Sender, TrySendError};
use zmq::{Context, Message, Socket, SocketType};

/// Cache-line padding constant
const CACHE_LINE_SIZE: usize = 64;

/// Default ZMQ endpoints
pub const DEFAULT_PUB_ENDPOINT: &str = "tcp://127.0.0.1:5555";
pub const DEFAULT_SUB_ENDPOINT: &str = "tcp://127.0.0.1:5556";
pub const DEFAULT_REQ_ENDPOINT: &str = "tcp://127.0.0.1:5557";
pub const DEFAULT_REP_ENDPOINT: &str = "tcp://127.0.0.1:5558";

/// Message types for IPC communication
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageType {
    InferenceRequest = 0,
    InferenceResponse = 1,
    WeightUpdate = 2,
    Heartbeat = 3,
    Shutdown = 4,
    FeatureVector = 5,
    SignalResult = 6,
}

impl From<u8> for MessageType {
    fn from(value: u8) -> Self {
        match value {
            0 => Self::InferenceRequest,
            1 => Self::InferenceResponse,
            2 => Self::WeightUpdate,
            3 => Self::Heartbeat,
            4 => Self::Shutdown,
            5 => Self::FeatureVector,
            _ => Self::InferenceResponse,
        }
    }
}

/// Message header for ZMQ communication
#[repr(C, align(64))]
pub struct ZmqMessageHeader {
    pub message_type: u8,
    pub version: u8,
    pub flags: u16,
    pub payload_size: u32,
    pub timestamp_ns: u64,
    pub sequence_id: u64,
    pub symbol_id: u64,
    pub confidence: f32,
    _padding: [u8; CACHE_LINE_SIZE - 8 - 4 - 8 - 8 - 4],
}

impl Default for ZmqMessageHeader {
    fn default() -> Self {
        Self {
            message_type: 0,
            version: 1,
            flags: 0,
            payload_size: 0,
            timestamp_ns: 0,
            sequence_id: 0,
            symbol_id: 0,
            confidence: 0.0,
            _padding: [0u8; CACHE_LINE_SIZE - 8 - 4 - 8 - 8 - 4],
        }
    }
}

/// Inference result from Python ML backend
#[derive(Debug, Clone)]
pub struct InferenceResult {
    pub symbol: String,
    pub signal: f32,       // -1.0 (sell) to 1.0 (buy)
    pub confidence: f32,   // 0.0 to 1.0
    pub timestamp_ns: u64,
    pub model_version: u32,
    pub features_used: u32,
}

/// Weight update from Python training loop
#[derive(Debug, Clone)]
pub struct WeightUpdate {
    pub strategy_id: String,
    pub weights: Vec<f32>,
    pub timestamp_ns: u64,
    pub validation_score: f32,
}

/// ZMQ Bridge manager for Python communication
pub struct ZmqBridge {
    context: Context,
    pub_socket: Option<Socket>,
    sub_socket: Option<Socket>,
    req_socket: Option<Socket>,
    rep_socket: Option<Socket>,
    sender: Sender<ZmqEnvelope>,
    receiver: Receiver<ZmqEnvelope>,
    running: Arc<AtomicBool>,
    sequence_counter: Arc<AtomicU64>,
    messages_sent: Arc<AtomicU64>,
    messages_received: Arc<AtomicU64>,
    last_heartbeat: Arc<AtomicU64>,
}

unsafe impl Send for ZmqBridge {}
unsafe impl Sync for ZmqBridge {}

/// Envelope for internal message passing
#[derive(Debug, Clone)]
pub struct ZmqEnvelope {
    pub header: ZmqMessageHeader,
    pub payload: Vec<u8>,
}

impl ZmqBridge {
    /// Create a new ZMQ bridge
    pub fn new() -> io::Result<Self> {
        let context = Context::new();
        let (sender, receiver) = bounded(10000); // 10K message buffer

        Ok(Self {
            context,
            pub_socket: None,
            sub_socket: None,
            req_socket: None,
            rep_socket: None,
            sender,
            receiver,
            running: Arc::new(AtomicBool::new(false)),
            sequence_counter: Arc::new(AtomicU64::new(0)),
            messages_sent: Arc::new(AtomicU64::new(0)),
            messages_received: Arc::new(AtomicU64::new(0)),
            last_heartbeat: Arc::new(AtomicU64::new(0)),
        })
    }

    /// Start the publisher socket for broadcasting to Python
    pub fn start_publisher(&mut self, endpoint: &str) -> io::Result<()> {
        let socket = self.context.socket(SocketType::PUB)?;
        socket.set_sndhwm(10000)?;
        socket.set_sndtimeo(100)?; // 100ms timeout
        socket.bind(endpoint)?;
        self.pub_socket = Some(socket);
        Ok(())
    }

    /// Start the subscriber socket for receiving from Python
    pub fn start_subscriber(&mut self, endpoint: &str, topic: &str) -> io::Result<()> {
        let socket = self.context.socket(SocketType::SUB)?;
        socket.set_rcvhwm(10000)?;
        socket.set_rcvtimeo(100)?;
        socket.connect(endpoint)?;
        socket.subscribe(topic)?;
        self.sub_socket = Some(socket);
        Ok(())
    }

    /// Start the REQ socket for sending requests to Python
    pub fn start_requester(&mut self, endpoint: &str) -> io::Result<()> {
        let socket = self.context.socket(SocketType::REQ)?;
        socket.set_sndtimeo(1000)?; // 1s timeout
        socket.set_rcvtimeo(5000)?; // 5s timeout for inference
        socket.connect(endpoint)?;
        self.req_socket = Some(socket);
        Ok(())
    }

    /// Start the REP socket for receiving requests from Python
    pub fn start_responder(&mut self, endpoint: &str) -> io::Result<()> {
        let socket = self.context.socket(SocketType::REP)?;
        socket.set_sndtimeo(1000)?;
        socket.set_rcvtimeo(1000)?;
        socket.bind(endpoint)?;
        self.rep_socket = Some(socket);
        Ok(())
    }

    /// Send inference request to Python backend
    pub fn send_inference_request(
        &self,
        symbol: &str,
        features: &[f32],
    ) -> io::Result<u64> {
        if self.req_socket.is_none() {
            return Err(io::Error::new(
                io::ErrorKind::NotConnected,
                "REQ socket not initialized",
            ));
        }

        let sequence_id = self.sequence_counter.fetch_add(1, Ordering::Relaxed);
        let timestamp_ns = get_timestamp_ns();

        // Create header
        let mut header = ZmqMessageHeader::default();
        header.message_type = MessageType::InferenceRequest as u8;
        header.timestamp_ns = timestamp_ns;
        header.sequence_id = sequence_id;
        header.symbol_id = symbol_to_id(symbol);
        header.payload_size = (features.len() * std::mem::size_of::<f32>()) as u32;

        // Serialize header
        let header_bytes = unsafe {
            std::slice::from_raw_parts(
                &header as *const ZmqMessageHeader as *const u8,
                CACHE_LINE_SIZE,
            )
        };

        // Send header + features
        let socket = self.req_socket.as_ref().unwrap();
        
        socket.send(header_bytes, zmq::SNDMORE)?;
        socket.send(
            bytemuck::cast_slice(features),
            0,
        )?;

        self.messages_sent.fetch_add(1, Ordering::Relaxed);
        Ok(sequence_id)
    }

    /// Receive inference response from Python
    pub fn receive_inference_response(&self) -> io::Result<Option<InferenceResult>> {
        if self.req_socket.is_none() {
            return Err(io::Error::new(
                io::ErrorKind::NotConnected,
                "REQ socket not initialized",
            ));
        }

        let socket = self.req_socket.as_ref().unwrap();
        
        // Receive header
        let mut header_msg = Message::new();
        match socket.recv(&mut header_msg, zmq::DONTWAIT) {
            Ok(_) => {}
            Err(zmq::Error::EAGAIN) => return Ok(None),
            Err(e) => return Err(io::Error::new(io::ErrorKind::Other, e)),
        }

        let header_bytes = header_msg.as_str().unwrap_or("");
        if header_bytes.len() < CACHE_LINE_SIZE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid header size",
            ));
        }

        let header = unsafe {
            &*(header_bytes.as_bytes().as_ptr() as *const ZmqMessageHeader)
        };

        // Receive payload
        let mut payload_msg = Message::new();
        socket.recv(&mut payload_msg, 0)?;

        let feature_count = header.payload_size as usize / std::mem::size_of::<f32>();
        let features: &[f32] = bytemuck::cast_slice(payload_msg.as_slice());

        self.messages_received.fetch_add(1, Ordering::Relaxed);

        // Parse inference result (first feature is signal, second is confidence)
        let signal = features.first().copied().unwrap_or(0.0);
        let confidence = features.get(1).copied().unwrap_or(0.5);

        Ok(Some(InferenceResult {
            symbol: id_to_symbol(header.symbol_id),
            signal,
            confidence,
            timestamp_ns: header.timestamp_ns,
            model_version: header.sequence_id as u32,
            features_used: feature_count as u32,
        }))
    }

    /// Broadcast feature vector to Python subscribers
    pub fn broadcast_features(
        &self,
        symbol: &str,
        features: &[f32],
        feature_flags: u64,
    ) -> io::Result<()> {
        if self.pub_socket.is_none() {
            return Err(io::Error::new(
                io::ErrorKind::NotConnected,
                "PUB socket not initialized",
            ));
        }

        let sequence_id = self.sequence_counter.fetch_add(1, Ordering::Relaxed);
        let timestamp_ns = get_timestamp_ns();

        let mut header = ZmqMessageHeader::default();
        header.message_type = MessageType::FeatureVector as u8;
        header.timestamp_ns = timestamp_ns;
        header.sequence_id = sequence_id;
        header.symbol_id = symbol_to_id(symbol);
        header.payload_size = (features.len() * std::mem::size_of::<f32>()) as u32;
        header.flags = feature_flags as u16;

        let header_bytes = unsafe {
            std::slice::from_raw_parts(
                &header as *const ZmqMessageHeader as *const u8,
                CACHE_LINE_SIZE,
            )
        };

        let socket = self.pub_socket.as_ref().unwrap();
        socket.send(header_bytes, zmq::SNDMORE)?;
        socket.send(bytemuck::cast_slice(features), 0)?;

        self.messages_sent.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Send weight update to Rust execution engine
    pub fn send_weight_update(&self, update: &WeightUpdate) -> io::Result<()> {
        if self.pub_socket.is_none() {
            return Err(io::Error::new(
                io::ErrorKind::NotConnected,
                "PUB socket not initialized",
            ));
        }

        let sequence_id = self.sequence_counter.fetch_add(1, Ordering::Relaxed);
        let timestamp_ns = get_timestamp_ns();

        let mut header = ZmqMessageHeader::default();
        header.message_type = MessageType::WeightUpdate as u8;
        header.timestamp_ns = timestamp_ns;
        header.sequence_id = sequence_id;
        header.confidence = update.validation_score;
        header.payload_size = (update.weights.len() * std::mem::size_of::<f32>()) as u32;

        let header_bytes = unsafe {
            std::slice::from_raw_parts(
                &header as *const ZmqMessageHeader as *const u8,
                CACHE_LINE_SIZE,
            )
        };

        let socket = self.pub_socket.as_ref().unwrap();
        socket.send(header_bytes, zmq::SNDMORE)?;
        socket.send(bytemuck::cast_slice(&update.weights), 0)?;

        self.messages_sent.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Send heartbeat
    pub fn send_heartbeat(&self) -> io::Result<()> {
        if self.pub_socket.is_none() {
            return Err(io::Error::new(
                io::ErrorKind::NotConnected,
                "PUB socket not initialized",
            ));
        }

        let timestamp_ns = get_timestamp_ns();
        self.last_heartbeat.store(timestamp_ns, Ordering::Release);

        let mut header = ZmqMessageHeader::default();
        header.message_type = MessageType::Heartbeat as u8;
        header.timestamp_ns = timestamp_ns;

        let header_bytes = unsafe {
            std::slice::from_raw_parts(
                &header as *const ZmqMessageHeader as *const u8,
                CACHE_LINE_SIZE,
            )
        };

        let socket = self.pub_socket.as_ref().unwrap();
        socket.send(header_bytes, 0)?;

        Ok(())
    }

    /// Check if connection is alive
    pub fn is_alive(&self) -> bool {
        self.running.load(Ordering::Acquire)
    }

    /// Get message statistics
    pub fn get_stats(&self) -> (u64, u64, u64) {
        (
            self.messages_sent.load(Ordering::Relaxed),
            self.messages_received.load(Ordering::Relaxed),
            self.last_heartbeat.load(Ordering::Relaxed),
        )
    }

    /// Shutdown the bridge
    pub fn shutdown(&mut self) {
        self.running.store(false, Ordering::Release);
        
        // Close sockets
        self.pub_socket = None;
        self.sub_socket = None;
        self.req_socket = None;
        self.rep_socket = None;
    }
}

impl Default for ZmqBridge {
    fn default() -> Self {
        Self::new().expect("Failed to create ZMQ bridge")
    }
}

/// Convert symbol string to numeric ID
fn symbol_to_id(symbol: &str) -> u64 {
    let bytes = symbol.as_bytes();
    let mut id: u64 = 0;
    for (i, &b) in bytes.iter().take(8).enumerate() {
        id |= (b as u64) << (i * 8);
    }
    id
}

/// Convert numeric ID back to symbol string
fn id_to_symbol(id: u64) -> String {
    let bytes = id.to_le_bytes();
    let mut symbol = String::new();
    for &b in &bytes {
        if b == 0 {
            break;
        }
        symbol.push(b as char);
    }
    symbol
}

/// Get current timestamp in nanoseconds
fn get_timestamp_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_cache_alignment() {
        let header = ZmqMessageHeader::default();
        let addr = &header as *const _ as usize;
        assert_eq!(addr % CACHE_LINE_SIZE, 0, "Header should be cache-line aligned");
    }

    #[test]
    fn test_symbol_id_conversion() {
        let symbol = "BTCUSDT";
        let id = symbol_to_id(symbol);
        let recovered = id_to_symbol(id);
        assert_eq!(symbol, recovered);
    }

    #[test]
    fn test_bridge_creation() {
        let bridge = ZmqBridge::new().unwrap();
        assert!(!bridge.is_alive());
        
        let (sent, recv, hb) = bridge.get_stats();
        assert_eq!(sent, 0);
        assert_eq!(recv, 0);
        assert_eq!(hb, 0);
    }

    #[test]
    fn test_message_type_conversion() {
        assert_eq!(MessageType::from(0), MessageType::InferenceRequest);
        assert_eq!(MessageType::from(1), MessageType::InferenceResponse);
        assert_eq!(MessageType::from(2), MessageType::WeightUpdate);
        assert_eq!(MessageType::from(3), MessageType::Heartbeat);
        assert_eq!(MessageType::from(4), MessageType::Shutdown);
    }
}
