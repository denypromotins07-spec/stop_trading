//! FIX Session Layer
//!
//! Manages FIX session lifecycle including logon, heartbeats, test requests,
//! and sequence number gap fills. Handles automated resend requests gracefully
//! without blocking the main execution thread.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Duration;

use super::codec::{FixCodec, FixMessage, FixError, FixTag};

/// Session state enumeration
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionState {
    /// Session disconnected
    Disconnected,
    /// Connecting to counterparty
    Connecting,
    /// Awaiting logon response
    AwaitingLogon,
    /// Logged on and active
    LoggedOn,
    /// Logging off
    LoggingOff,
    /// Session terminated
    Terminated,
    /// In resend request mode
    ResendRequested,
}

impl SessionState {
    #[inline]
    pub fn is_active(self) -> bool {
        matches!(self, SessionState::LoggedOn | SessionState::ResendRequested)
    }
}

/// Session configuration
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SessionConfig {
    /// Sender CompID
    pub sender_comp_id: [u8; 32],
    /// Target CompID
    pub target_comp_id: [u8; 32],
    /// Heartbeat interval in seconds
    pub heartbeat_interval_sec: u32,
    /// Session start time (HHMMSS format)
    pub start_time: u32,
    /// Session end time (HHMMSS format)
    pub end_time: u32,
    /// Enable encryption
    pub enable_encryption: bool,
    /// Encrypt method (0 = none, 1 = PKCS, etc.)
    pub encrypt_method: u8,
    /// Maximum message size
    pub max_message_size: u32,
    /// Reset sequence on logon
    pub reset_on_logon: bool,
    /// Reset sequence on disconnect
    pub reset_on_disconnect: bool,
    /// Refresh on logon
    pub refresh_on_logon: bool,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            sender_comp_id: [0u8; 32],
            target_comp_id: [0u8; 32],
            heartbeat_interval_sec: 30,
            start_time: 0,
            end_time: 235959,
            enable_encryption: false,
            encrypt_method: 0,
            max_message_size: 4096,
            reset_on_logon: false,
            reset_on_disconnect: false,
            refresh_on_logon: false,
        }
    }
}

/// Resend request parameters
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ResendRequest {
    /// Begin sequence number
    pub begin_seq_no: u64,
    /// End sequence number (0 = infinity)
    pub end_seq_no: u64,
    /// Request timestamp
    pub request_time_ns: u64,
}

impl ResendRequest {
    #[inline]
    pub fn new(begin_seq_no: u64, end_seq_no: u64) -> Self {
        Self {
            begin_seq_no,
            end_seq_no,
            request_time_ns: 0,
        }
    }

    #[inline]
    pub fn with_timestamp(mut self, timestamp_ns: u64) -> Self {
        self.request_time_ns = timestamp_ns;
        self
    }
}

/// Session statistics
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SessionStats {
    /// Messages sent
    pub messages_sent: u64,
    /// Messages received
    pub messages_received: u64,
    /// Logons sent
    pub logons_sent: u64,
    /// Logons received
    pub logons_received: u64,
    /// Heartbeats sent
    pub heartbeats_sent: u64,
    /// Heartbeats received
    pub heartbeats_received: u64,
    /// Resend requests sent
    pub resend_requests_sent: u64,
    /// Sequence gaps detected
    pub sequence_gaps: u64,
    /// Test requests sent
    pub test_requests_sent: u64,
    /// Last message timestamp
    pub last_message_ns: u64,
    /// Last heartbeat timestamp
    pub last_heartbeat_ns: u64,
}

impl SessionStats {
    #[inline]
    pub fn new() -> Self {
        Self {
            messages_sent: 0,
            messages_received: 0,
            logons_sent: 0,
            logons_received: 0,
            heartbeats_sent: 0,
            heartbeats_received: 0,
            resend_requests_sent: 0,
            sequence_gaps: 0,
            test_requests_sent: 0,
            last_message_ns: 0,
            last_heartbeat_ns: 0,
        }
    }
}

impl Default for SessionStats {
    fn default() -> Self {
        Self::new()
    }
}

/// FIX Session manager
#[repr(C)]
pub struct FixSession {
    /// Session configuration
    config: SessionConfig,
    /// Current session state
    state: AtomicU32, // Using u32 for atomic SessionState
    /// Outgoing sequence number
    outgoing_seq_num: AtomicU64,
    /// Incoming (expected) sequence number
    incoming_seq_num: AtomicU64,
    /// Last received sequence number
    last_received_seq: AtomicU64,
    /// Codec reference
    codec: Arc<FixCodec>,
    /// Session is running
    is_running: AtomicBool,
    /// Last activity timestamp (nanoseconds)
    last_activity_ns: AtomicU64,
    /// Last heartbeat timestamp (nanoseconds)
    last_heartbeat_ns: AtomicU64,
    /// Heartbeat timeout threshold (nanoseconds)
    heartbeat_timeout_ns: AtomicU64,
    /// Statistics
    stats: SessionStats,
    /// Pending resend request
    pending_resend: AtomicBool,
    /// Resend begin sequence
    resend_begin_seq: AtomicU64,
    /// Resend end sequence
    resend_end_seq: AtomicU64,
}

// Map SessionState to u32 for atomic operations
fn state_to_u32(state: SessionState) -> u32 {
    match state {
        SessionState::Disconnected => 0,
        SessionState::Connecting => 1,
        SessionState::AwaitingLogon => 2,
        SessionState::LoggedOn => 3,
        SessionState::LoggingOff => 4,
        SessionState::Terminated => 5,
        SessionState::ResendRequested => 6,
    }
}

fn u32_to_state(val: u32) -> SessionState {
    match val {
        0 => SessionState::Disconnected,
        1 => SessionState::Connecting,
        2 => SessionState::AwaitingLogon,
        3 => SessionState::LoggedOn,
        4 => SessionState::LoggingOff,
        5 => SessionState::Terminated,
        6 => SessionState::ResendRequested,
        _ => SessionState::Disconnected,
    }
}

impl FixSession {
    /// Create a new FIX session
    pub fn new(config: SessionConfig, codec: Arc<FixCodec>) -> Result<Self, FixError> {
        let heartbeat_timeout_ns = (config.heartbeat_interval_sec as u64) 
            .saturating_mul(1_000_000_000)
            .saturating_mul(2); // 2x heartbeat interval for timeout

        Ok(Self {
            config,
            state: AtomicU32::new(state_to_u32(SessionState::Disconnected)),
            outgoing_seq_num: AtomicU64::new(1),
            incoming_seq_num: AtomicU64::new(1),
            last_received_seq: AtomicU64::new(0),
            codec,
            is_running: AtomicBool::new(false),
            last_activity_ns: AtomicU64::new(0),
            last_heartbeat_ns: AtomicU64::new(0),
            heartbeat_timeout_ns: AtomicU64::new(heartbeat_timeout_ns),
            stats: SessionStats::new(),
            pending_resend: AtomicBool::new(false),
            resend_begin_seq: AtomicU64::new(0),
            resend_end_seq: AtomicU64::new(0),
        })
    }

    /// Get current session state
    #[inline]
    pub fn get_state(&self) -> SessionState {
        u32_to_state(self.state.load(Ordering::Acquire))
    }

    /// Set session state
    #[inline]
    pub fn set_state(&self, state: SessionState) {
        self.state.store(state_to_u32(state), Ordering::Release);
    }

    /// Check if logged on
    #[inline]
    pub fn is_logged_on(&self) -> bool {
        self.get_state() == SessionState::LoggedOn
    }

    /// Start the session
    #[inline]
    pub fn start(&self) {
        self.set_state(SessionState::Connecting);
        self.is_running.store(true, Ordering::Release);
        
        let now = self.get_timestamp_ns();
        self.last_activity_ns.store(now, Ordering::Release);
    }

    /// Stop the session
    #[inline]
    pub fn stop(&self) {
        self.is_running.store(false, Ordering::Release);
        
        if self.config.reset_on_disconnect {
            self.reset_sequence_numbers();
        }
        
        self.set_state(SessionState::Terminated);
    }

    /// Process incoming message
    #[inline]
    pub fn process_message(&self, msg: &FixMessage) -> Result<(), FixError> {
        if !self.is_running.load(Ordering::Acquire) {
            return Err(FixError::SessionNotActive);
        }

        let now = self.get_timestamp_ns();
        self.last_activity_ns.store(now, Ordering::Release);
        self.stats.messages_received += 1;
        self.stats.last_message_ns = now;

        // Validate sequence number
        if let Some(seq_field) = msg.get_field(FixTag::MsgSeqNum) {
            let seq_num = seq_field.as_uint()
                .map_err(|_| FixError::ParseError)?;
            
            if !self.validate_sequence(seq_num)? {
                // Sequence gap detected - request resend
                self.request_resend(seq_num)?;
            }
        }

        // Handle message based on type
        if let Some(msg_type) = msg.msg_type() {
            match msg_type {
                "0" => self.handle_heartbeat(msg),
                "A" => self.handle_logon(msg),
                "5" => self.handle_logout(msg),
                "1" => self.handle_test_request(msg),
                "2" => self.handle_resend_request(msg),
                "3" => self.handle_reject(msg),
                _ => self.handle_application_message(msg),
            }
        }

        Ok(())
    }

    /// Send logon message
    #[inline]
    pub fn send_logon(&self, buffer: &mut [u8]) -> Result<usize, FixError> {
        // Build logon message fields
        let mut fields = Vec::new();
        
        // Standard header
        fields.push((FixTag::BeginString.as_u32(), b"FIX.4.4" as &[u8]));
        fields.push((FixTag::BodyLength.as_u32(), b"000" as &[u8])); // Placeholder
        fields.push((FixTag::MsgType.as_u32(), b"A"));
        fields.push((FixTag::SenderCompID.as_u32(), &self.config.sender_comp_id));
        fields.push((FixTag::TargetCompID.as_u32(), &self.config.target_comp_id));
        fields.push((FixTag::MsgSeqNum.as_u32(), self.get_and_increment_seq().to_string().as_bytes()));
        fields.push((FixTag::SendingTime.as_u32(), self.get_sending_time().as_bytes()));
        
        // Logon body
        fields.push((FixTag::EncryptMethod.as_u32(), &[self.config.encrypt_method]));
        fields.push((FixTag::HeartBtInt.as_u32(), self.config.heartbeat_interval_sec.to_string().as_bytes()));
        
        if self.config.reset_on_logon {
            fields.push((FixTag::ResetSeqNumFlag.as_u32(), b"Y"));
        }

        self.stats.logons_sent += 1;
        self.set_state(SessionState::AwaitingLogon);

        // Encode message (simplified - in production would use proper builder)
        self.encode_message(&fields, buffer)
    }

    /// Send logout message
    #[inline]
    pub fn send_logout(&self, buffer: &mut [u8], reason: Option<&str>) -> Result<usize, FixError> {
        self.set_state(SessionState::LoggingOff);
        
        let mut fields = Vec::new();
        fields.push((FixTag::BeginString.as_u32(), b"FIX.4.4" as &[u8]));
        fields.push((FixTag::BodyLength.as_u32(), b"000" as &[u8]));
        fields.push((FixTag::MsgType.as_u32(), b"5"));
        fields.push((FixTag::SenderCompID.as_u32(), &self.config.sender_comp_id));
        fields.push((FixTag::TargetCompID.as_u32(), &self.config.target_comp_id));
        fields.push((FixTag::MsgSeqNum.as_u32(), self.get_and_increment_seq().to_string().as_bytes()));
        fields.push((FixTag::SendingTime.as_u32(), self.get_sending_time().as_bytes()));

        if let Some(r) = reason {
            fields.push((FixTag::Text.as_u32(), r.as_bytes()));
        }

        self.encode_message(&fields, buffer)
    }

    /// Send heartbeat
    #[inline]
    pub fn send_heartbeat(&self, buffer: &mut [u8], test_req_id: Option<&str>) -> Result<usize, FixError> {
        let mut fields = Vec::new();
        fields.push((FixTag::BeginString.as_u32(), b"FIX.4.4" as &[u8]));
        fields.push((FixTag::BodyLength.as_u32(), b"000" as &[u8]));
        fields.push((FixTag::MsgType.as_u32(), b"0"));
        fields.push((FixTag::SenderCompID.as_u32(), &self.config.sender_comp_id));
        fields.push((FixTag::TargetCompID.as_u32(), &self.config.target_comp_id));
        fields.push((FixTag::MsgSeqNum.as_u32(), self.get_and_increment_seq().to_string().as_bytes()));
        fields.push((FixTag::SendingTime.as_u32(), self.get_sending_time().as_bytes()));

        if let Some(id) = test_req_id {
            fields.push((FixTag::TestReqID.as_u32(), id.as_bytes()));
        }

        self.stats.heartbeats_sent += 1;
        let now = self.get_timestamp_ns();
        self.last_heartbeat_ns.store(now, Ordering::Release);

        self.encode_message(&fields, buffer)
    }

    /// Send test request
    #[inline]
    pub fn send_test_request(&self, buffer: &mut [u8], test_req_id: &str) -> Result<usize, FixError> {
        let mut fields = Vec::new();
        fields.push((FixTag::BeginString.as_u32(), b"FIX.4.4" as &[u8]));
        fields.push((FixTag::BodyLength.as_u32(), b"000" as &[u8]));
        fields.push((FixTag::MsgType.as_u32(), b"1"));
        fields.push((FixTag::SenderCompID.as_u32(), &self.config.sender_comp_id));
        fields.push((FixTag::TargetCompID.as_u32(), &self.config.target_comp_id));
        fields.push((FixTag::MsgSeqNum.as_u32(), self.get_and_increment_seq().to_string().as_bytes()));
        fields.push((FixTag::SendingTime.as_u32(), self.get_sending_time().as_bytes()));
        fields.push((FixTag::TestReqID.as_u32(), test_req_id.as_bytes()));

        self.stats.test_requests_sent += 1;
        self.encode_message(&fields, buffer)
    }

    /// Request resend for missing messages
    #[inline]
    pub fn request_resend(&self, expected_seq: u64) -> Result<(), FixError> {
        let last_received = self.last_received_seq.load(Ordering::Acquire);
        
        if expected_seq > last_received + 1 {
            self.stats.sequence_gaps += 1;
            
            self.pending_resend.store(true, Ordering::Release);
            self.resend_begin_seq.store(last_received + 1, Ordering::Release);
            self.resend_end_seq.store(expected_seq - 1, Ordering::Release);
            
            self.set_state(SessionState::ResendRequested);
            self.stats.resend_requests_sent += 1;
        }

        Ok(())
    }

    /// Validate sequence number
    #[inline]
    fn validate_sequence(&self, received_seq: u64) -> Result<bool, FixError> {
        let expected = self.incoming_seq_num.load(Ordering::Acquire);

        if received_seq < expected {
            // Duplicate or old message - might need special handling
            return Ok(false);
        }

        if received_seq > expected {
            // Gap detected
            return Ok(false);
        }

        // Sequence is correct
        self.incoming_seq_num.fetch_add(1, Ordering::AcqRel);
        self.last_received_seq.store(received_seq, Ordering::Release);
        Ok(true)
    }

    /// Get and increment outgoing sequence number
    #[inline]
    fn get_and_increment_seq(&self) -> u64 {
        self.outgoing_seq_num.fetch_add(1, Ordering::AcqRel)
    }

    /// Reset sequence numbers
    #[inline]
    pub fn reset_sequence_numbers(&self) {
        self.outgoing_seq_num.store(1, Ordering::Release);
        self.incoming_seq_num.store(1, Ordering::Release);
        self.last_received_seq.store(0, Ordering::Release);
    }

    /// Check heartbeat timeout
    #[inline]
    pub fn check_heartbeat_timeout(&self) -> bool {
        let last_activity = self.last_activity_ns.load(Ordering::Acquire);
        let timeout = self.heartbeat_timeout_ns.load(Ordering::Acquire);
        let now = self.get_timestamp_ns();

        if last_activity > 0 && now.saturating_sub(last_activity) > timeout {
            return true; // Timeout occurred
        }
        false
    }

    /// Get session statistics
    #[inline]
    pub fn get_stats(&self) -> SessionStats {
        SessionStats {
            messages_sent: self.stats.messages_sent,
            messages_received: self.stats.messages_received,
            logons_sent: self.stats.logons_sent,
            logons_received: self.stats.logons_received,
            heartbeats_sent: self.stats.heartbeats_sent,
            heartbeats_received: self.stats.heartbeats_received,
            resend_requests_sent: self.stats.resend_requests_sent,
            sequence_gaps: self.stats.sequence_gaps,
            test_requests_sent: self.stats.test_requests_sent,
            last_message_ns: self.stats.last_message_ns,
            last_heartbeat_ns: self.last_heartbeat_ns.load(Ordering::Acquire),
        }
    }

    /// Get current timestamp in nanoseconds
    #[inline]
    fn get_timestamp_ns(&self) -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64
    }

    /// Get sending time in FIX format (YYYYMMDD-HH:MM:SS)
    #[inline]
    fn get_sending_time(&self) -> [u8; 21] {
        // Simplified - in production would use proper chrono formatting
        *b"20240101-12:00:00.000"
    }

    /// Encode message from fields
    #[inline]
    fn encode_message(&self, fields: &[(u32, &[u8])], buffer: &mut [u8]) -> Result<usize, FixError> {
        // Simplified encoding - in production would use proper FIX encoder
        let mut pos = 0;
        
        for &(tag, value) in fields {
            if tag == FixTag::BodyLength.as_u32() {
                continue; // Skip placeholder
            }

            // Write tag=value<SOH>
            let tag_str = tag.to_string();
            for b in tag_str.as_bytes() {
                if pos >= buffer.len() {
                    return Err(FixError::BufferTooSmall);
                }
                buffer[pos] = *b;
                pos += 1;
            }

            if pos >= buffer.len() {
                return Err(FixError::BufferTooSmall);
            }
            buffer[pos] = b'=';
            pos += 1;

            for b in value {
                if pos >= buffer.len() {
                    return Err(FixError::BufferTooSmall);
                }
                buffer[pos] = *b;
                pos += 1;
            }

            if pos >= buffer.len() {
                return Err(FixError::BufferTooSmall);
            }
            buffer[pos] = 1; // SOH
            pos += 1;
        }

        self.stats.messages_sent += 1;
        Ok(pos)
    }

    /// Handle heartbeat message
    #[inline]
    fn handle_heartbeat(&self, _msg: &FixMessage) {
        self.stats.heartbeats_received += 1;
        let now = self.get_timestamp_ns();
        self.last_heartbeat_ns.store(now, Ordering::Release);
    }

    /// Handle logon message
    #[inline]
    fn handle_logon(&self, _msg: &FixMessage) {
        self.stats.logons_received += 1;
        self.set_state(SessionState::LoggedOn);
    }

    /// Handle logout message
    #[inline]
    fn handle_logout(&self, _msg: &FixMessage) {
        self.set_state(SessionState::Disconnected);
    }

    /// Handle test request
    #[inline]
    fn handle_test_request(&self, _msg: &FixMessage) {
        // Would send heartbeat response in production
    }

    /// Handle resend request
    #[inline]
    fn handle_resend_request(&self, _msg: &FixMessage) {
        // Would replay messages in production
        self.set_state(SessionState::LoggedOn);
        self.pending_resend.store(false, Ordering::Release);
    }

    /// Handle reject message
    #[inline]
    fn handle_reject(&self, _msg: &FixMessage) {
        // Log rejection - in production would track and alert
    }

    /// Handle application message
    #[inline]
    fn handle_application_message(&self, _msg: &FixMessage) {
        // Route to application layer - in production would dispatch to handlers
    }
}

// Additional FIX tags needed for session
impl FixTag {
    pub const ResetSeqNumFlag: FixTag = unsafe { std::mem::transmute(141u32) };
    pub const Text: FixTag = unsafe { std::mem::transmute(58u32) };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_creation() {
        let config = SessionConfig::default();
        let codec = Arc::new(FixCodec::new());
        
        let session = FixSession::new(config, codec).unwrap();
        
        assert_eq!(session.get_state(), SessionState::Disconnected);
        assert!(!session.is_logged_on());
    }

    #[test]
    fn test_session_lifecycle() {
        let config = SessionConfig::default();
        let codec = Arc::new(FixCodec::new());
        
        let session = FixSession::new(config, codec).unwrap();
        
        session.start();
        assert_eq!(session.get_state(), SessionState::Connecting);
        assert!(session.is_running.load(Ordering::Acquire));

        session.stop();
        assert_eq!(session.get_state(), SessionState::Terminated);
    }

    #[test]
    fn test_sequence_validation() {
        let config = SessionConfig::default();
        let codec = Arc::new(FixCodec::new());
        
        let session = FixSession::new(config, codec).unwrap();
        
        // First message should be seq 1
        assert!(session.validate_sequence(1).unwrap());
        
        // Next should be seq 2
        assert!(session.validate_sequence(2).unwrap());
        
        // Seq 2 again would be duplicate
        assert!(!session.validate_sequence(2).unwrap());
        
        // Seq 4 would be gap
        assert!(!session.validate_sequence(4).unwrap());
    }

    #[test]
    fn test_heartbeat_timeout() {
        let mut config = SessionConfig::default();
        config.heartbeat_interval_sec = 1; // 1 second for testing
        
        let codec = Arc::new(FixCodec::new());
        let session = FixSession::new(config, codec).unwrap();
        
        // Initially no timeout
        assert!(!session.check_heartbeat_timeout());
        
        // Simulate activity
        session.last_activity_ns.store(session.get_timestamp_ns(), Ordering::Release);
        
        // Still no timeout immediately
        assert!(!session.check_heartbeat_timeout());
    }

    #[test]
    fn test_session_stats() {
        let config = SessionConfig::default();
        let codec = Arc::new(FixCodec::new());
        
        let session = FixSession::new(config, codec).unwrap();
        let stats = session.get_stats();
        
        assert_eq!(stats.messages_sent, 0);
        assert_eq!(stats.messages_received, 0);
        assert_eq!(stats.sequence_gaps, 0);
    }
}
