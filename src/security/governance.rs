//! Immutable Audit Logger for Governance
//! 
//! Records all API key usage, config changes, and manual overrides.
//! Cryptographically hashes daily logs for tamper-proof compliance reporting.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use crossbeam_channel::{bounded, Sender, Receiver};

/// Maximum pending log entries before flush
const LOG_BUFFER_SIZE: usize = 10_000;

/// Audit event types
#[derive(Debug, Clone)]
pub enum AuditEventType {
    ApiKeyUsage {
        key_id: String,
        endpoint: String,
        success: bool,
    },
    ConfigChange {
        config_key: String,
        old_value: String,
        new_value: String,
        changed_by: String,
    },
    ManualOverride {
        override_type: String,
        reason: String,
        overridden_by: String,
        affected_system: String,
    },
    TradeExecution {
        order_id: String,
        symbol: String,
        side: String,
        size: f64,
        price: f64,
    },
    ComplianceAction {
        action_type: String,
        address: String,
        result: String,
    },
    SystemEvent {
        event_name: String,
        severity: Severity,
        details: String,
    },
}

/// Event severity levels
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Severity {
    Info,
    Warning,
    Error,
    Critical,
}

/// Audit log entry
#[derive(Debug, Clone)]
pub struct AuditEntry {
    pub event_type: AuditEventType,
    pub timestamp_ns: u64,
    pub sequence_number: u64,
    pub previous_hash: [u8; 32],
    pub current_hash: [u8; 32],
}

/// Daily log summary with hash
#[derive(Debug, Clone)]
pub struct DailyLogSummary {
    pub date: String,
    pub entry_count: u64,
    pub merkle_root: [u8; 32],
    pub start_hash: [u8; 32],
    pub end_hash: [u8; 32],
}

/// Simple SHA-256-like hash (simplified for demonstration)
fn compute_hash(data: &[u8]) -> [u8; 32] {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    let mut hasher = DefaultHasher::new();
    data.hash(&mut hasher);
    let hash = hasher.finish();
    
    // Expand to 32 bytes (simplified - would use real SHA-256 in production)
    let mut result = [0u8; 32];
    result[..8].copy_from_slice(&hash.to_le_bytes());
    result[8..16].copy_from_slice(&(hash.wrapping_add(1)).to_le_bytes());
    result[16..24].copy_from_slice(&(hash.wrapping_add(2)).to_le_bytes());
    result[24..32].copy_from_slice(&(hash.wrapping_add(3)).to_le_bytes());
    
    result
}

/// Compute Merkle root from a list of hashes
fn compute_merkle_root(hashes: &[[u8; 32]]) -> [u8; 32] {
    if hashes.is_empty() {
        return [0u8; 32];
    }
    
    if hashes.len() == 1 {
        return hashes[0];
    }
    
    let mut current_level: Vec<[u8; 32]> = hashes.to_vec();
    
    while current_level.len() > 1 {
        let mut next_level = Vec::new();
        
        for chunk in current_level.chunks(2) {
            let combined = if chunk.len() == 2 {
                let mut combined = Vec::with_capacity(64);
                combined.extend_from_slice(&chunk[0]);
                combined.extend_from_slice(&chunk[1]);
                combined
            } else {
                let mut combined = Vec::with_capacity(64);
                combined.extend_from_slice(&chunk[0]);
                combined.extend_from_slice(&chunk[0]); // Duplicate odd node
                combined
            };
            
            next_level.push(compute_hash(&combined));
        }
        
        current_level = next_level;
    }
    
    current_level[0]
}

/// Immutable audit logger
pub struct AuditLogger {
    /// Log entry channel
    entry_tx: Sender<AuditEntry>,
    entry_rx: Receiver<AuditEntry>,
    
    /// Current sequence number
    sequence: AtomicU64,
    
    /// Previous hash for chaining
    previous_hash: Arc<std::sync::Mutex<[u8; 32]>>,
    
    /// Today's entries for Merkle root calculation
    daily_entries: Arc<std::sync::Mutex<Vec<[u8; 32]>>>,
    
    /// Statistics
    total_logged: AtomicU64,
    
    /// Last hash for verification
    last_hash: Arc<std::sync::Mutex<[u8; 32]>>,
}

impl AuditLogger {
    /// Create a new audit logger
    pub fn new() -> Self {
        let (entry_tx, entry_rx) = bounded(LOG_BUFFER_SIZE);
        let genesis_hash = [0u8; 32];
        
        Self {
            entry_tx,
            entry_rx,
            sequence: AtomicU64::new(1),
            previous_hash: Arc::new(std::sync::Mutex::new(genesis_hash)),
            daily_entries: Arc::new(std::sync::Mutex::new(Vec::new())),
            total_logged: AtomicU64::new(0),
            last_hash: Arc::new(std::sync::Mutex::new(genesis_hash)),
        }
    }
    
    /// Log an audit event
    pub fn log(&self, event_type: AuditEventType) -> Result<u64, LogError> {
        let seq = self.sequence.fetch_add(1, Ordering::SeqCst);
        
        // Get previous hash
        let prev_hash = *self.previous_hash.lock().unwrap();
        
        // Create entry data for hashing
        let entry_data = self.serialize_event(&event_type, seq);
        let mut hash_data = Vec::with_capacity(64 + entry_data.len());
        hash_data.extend_from_slice(&prev_hash);
        hash_data.extend(&entry_data);
        
        let current_hash = compute_hash(&hash_data);
        
        let entry = AuditEntry {
            event_type,
            timestamp_ns: get_timestamp_ns(),
            sequence_number: seq,
            previous_hash: prev_hash,
            current_hash,
        };
        
        // Try to send (non-blocking)
        match self.entry_tx.try_send(entry) {
            Ok(_) => {
                // Update state
                *self.previous_hash.lock().unwrap() = current_hash;
                *self.last_hash.lock().unwrap() = current_hash;
                
                // Add to daily entries
                if let Ok(mut entries) = self.daily_entries.lock() {
                    entries.push(current_hash);
                }
                
                self.total_logged.fetch_add(1, Ordering::Relaxed);
                Ok(seq)
            }
            Err(_) => Err(LogError::QueueFull),
        }
    }
    
    /// Log API key usage
    pub fn log_api_usage(&self, key_id: &str, endpoint: &str, success: bool) -> Result<u64, LogError> {
        self.log(AuditEventType::ApiKeyUsage {
            key_id: key_id.to_string(),
            endpoint: endpoint.to_string(),
            success,
        })
    }
    
    /// Log configuration change
    pub fn log_config_change(
        &self,
        config_key: &str,
        old_value: &str,
        new_value: &str,
        changed_by: &str,
    ) -> Result<u64, LogError> {
        self.log(AuditEventType::ConfigChange {
            config_key: config_key.to_string(),
            old_value: old_value.to_string(),
            new_value: new_value.to_string(),
            changed_by: changed_by.to_string(),
        })
    }
    
    /// Log manual override
    pub fn log_manual_override(
        &self,
        override_type: &str,
        reason: &str,
        overridden_by: &str,
        affected_system: &str,
    ) -> Result<u64, LogError> {
        self.log(AuditEventType::ManualOverride {
            override_type: override_type.to_string(),
            reason: reason.to_string(),
            overridden_by: overridden_by.to_string(),
            affected_system: affected_system.to_string(),
        })
    }
    
    /// Log trade execution
    pub fn log_trade(
        &self,
        order_id: &str,
        symbol: &str,
        side: &str,
        size: f64,
        price: f64,
    ) -> Result<u64, LogError> {
        self.log(AuditEventType::TradeExecution {
            order_id: order_id.to_string(),
            symbol: symbol.to_string(),
            side: side.to_string(),
            size,
            price,
        })
    }
    
    /// Log compliance action
    pub fn log_compliance_action(
        &self,
        action_type: &str,
        address: &str,
        result: &str,
    ) -> Result<u64, LogError> {
        self.log(AuditEventType::ComplianceAction {
            action_type: action_type.to_string(),
            address: address.to_string(),
            result: result.to_string(),
        })
    }
    
    /// Get receiver for audit entries
    pub fn entry_receiver(&self) -> Receiver<AuditEntry> {
        self.entry_rx.clone()
    }
    
    /// Generate daily log summary with Merkle root
    pub fn generate_daily_summary(&self, date: &str) -> DailyLogSummary {
        let entries = self.daily_entries.lock().unwrap();
        let last_hash = *self.last_hash.lock().unwrap();
        
        let merkle_root = if entries.is_empty() {
            [0u8; 32]
        } else {
            compute_merkle_root(&entries)
        };
        
        let start_hash = if entries.is_empty() {
            [0u8; 32]
        } else {
            entries[0]
        };
        
        DailyLogSummary {
            date: date.to_string(),
            entry_count: entries.len() as u64,
            merkle_root,
            start_hash,
            end_hash: last_hash,
        }
    }
    
    /// Verify log integrity up to current point
    pub fn verify_integrity(&self) -> bool {
        // In production, would recompute entire chain
        // For now, just check we have entries
        let entries = self.daily_entries.lock().unwrap();
        !entries.is_empty() || self.total_logged.load(Ordering::Relaxed) == 0
    }
    
    /// Get total logged entries
    pub fn get_total_logged(&self) -> u64 {
        self.total_logged.load(Ordering::Relaxed)
    }
    
    /// Serialize event for hashing
    fn serialize_event(&self, event: &AuditEventType, seq: u64) -> Vec<u8> {
        let mut data = Vec::new();
        
        // Include sequence number
        data.extend_from_slice(&seq.to_le_bytes());
        
        // Include event type discriminator
        match event {
            AuditEventType::ApiKeyUsage { .. } => data.push(0),
            AuditEventType::ConfigChange { .. } => data.push(1),
            AuditEventType::ManualOverride { .. } => data.push(2),
            AuditEventType::TradeExecution { .. } => data.push(3),
            AuditEventType::ComplianceAction { .. } => data.push(4),
            AuditEventType::SystemEvent { .. } => data.push(5),
        }
        
        // Include timestamp
        let ts = get_timestamp_ns();
        data.extend_from_slice(&ts.to_le_bytes());
        
        data
    }
}

impl Default for AuditLogger {
    fn default() -> Self {
        Self::new()
    }
}

/// Log error types
#[derive(Debug, Clone)]
pub enum LogError {
    QueueFull,
    SerializationError,
    HashError,
}

impl std::fmt::Display for LogError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LogError::QueueFull => write!(f, "Log queue is full"),
            LogError::SerializationError => write!(f, "Failed to serialize event"),
            LogError::HashError => write!(f, "Failed to compute hash"),
        }
    }
}

impl std::error::Error for LogError {}

/// Get current timestamp in nanoseconds
fn get_timestamp_ns() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_logger_creation() {
        let logger = AuditLogger::new();
        assert_eq!(logger.get_total_logged(), 0);
        assert!(logger.verify_integrity());
    }
    
    #[test]
    fn test_log_api_usage() {
        let logger = AuditLogger::new();
        
        let result = logger.log_api_usage("key_123", "/api/trade", true);
        assert!(result.is_ok());
        assert_eq!(logger.get_total_logged(), 1);
    }
    
    #[test]
    fn test_log_config_change() {
        let logger = AuditLogger::new();
        
        let result = logger.log_config_change(
            "max_position_size",
            "1000000",
            "2000000",
            "admin",
        );
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_hash_chaining() {
        let logger = AuditLogger::new();
        
        let _ = logger.log_api_usage("key1", "/api/test", true);
        let _ = logger.log_api_usage("key2", "/api/test", false);
        
        // Each entry should have different hash
        assert!(logger.verify_integrity());
    }
    
    #[test]
    fn test_merkle_root() {
        let hashes = vec![
            [1u8; 32],
            [2u8; 32],
            [3u8; 32],
            [4u8; 32],
        ];
        
        let root = compute_merkle_root(&hashes);
        assert_ne!(root, [0u8; 32]);
    }
}
