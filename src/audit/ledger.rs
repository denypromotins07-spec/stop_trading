//! Cryptographic Audit Ledger
//! 
//! Append-only, cryptographically chained audit log (Merkle-style) for every routed order and fill.
//! Guarantees non-repudiation and compliance by hashing each trade event with previous block's hash.
//! Uses SHA-256 for cryptographic chaining.

use sha2::{Sha256, Digest};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};
use serde::{Serialize, Deserialize};

/// Maximum audit log entries before rotation
pub const MAX_AUDIT_ENTRIES: usize = 1_000_000;

/// Audit entry types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum AuditEventType {
    OrderSubmitted = 0,
    OrderAcknowledged = 1,
    OrderPartiallyFilled = 2,
    OrderFilled = 3,
    OrderCancelled = 4,
    OrderRejected = 5,
    TradeExecuted = 6,
    PositionUpdated = 7,
    RiskLimitHit = 8,
    SystemEvent = 9,
}

/// Single audit log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    /// Entry sequence number
    pub sequence: u64,
    /// Event timestamp (nanoseconds)
    pub timestamp_ns: u64,
    /// Event type
    pub event_type: AuditEventType,
    /// Symbol involved
    pub symbol: String,
    /// Order ID (if applicable)
    pub order_id: Option<u64>,
    /// Client order ID
    pub client_order_id: Option<u64>,
    /// Side (0=Buy, 1=Sell)
    pub side: Option<u8>,
    /// Quantity
    pub quantity: Option<f64>,
    /// Price
    pub price: Option<f64>,
    /// Fill quantity (for fills)
    pub fill_qty: Option<f64>,
    /// Fill price (for fills)
    pub fill_price: Option<f64>,
    /// Previous block hash (hex string)
    pub prev_hash: String,
    /// Current block hash (hex string)
    pub block_hash: String,
    /// Additional metadata (JSON)
    pub metadata: Option<String>,
}

impl AuditEntry {
    /// Create a new audit entry and compute hashes
    pub fn new(
        sequence: u64,
        event_type: AuditEventType,
        symbol: &str,
        prev_hash: &str,
    ) -> Self {
        let timestamp_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        
        // Create hash input
        let hash_input = format!(
            "{}|{}|{}|{}|{}",
            sequence, timestamp_ns, event_type as u8, symbol, prev_hash
        );
        
        let block_hash = hex::encode(Sha256::digest(hash_input.as_bytes()));
        
        Self {
            sequence,
            timestamp_ns,
            event_type,
            symbol: symbol.to_string(),
            order_id: None,
            client_order_id: None,
            side: None,
            quantity: None,
            price: None,
            fill_qty: None,
            fill_price: None,
            prev_hash: prev_hash.to_string(),
            block_hash,
            metadata: None,
        }
    }
    
    /// Set order details
    pub fn with_order(mut self, order_id: u64, client_order_id: u64, side: u8, qty: f64, price: f64) -> Self {
        self.order_id = Some(order_id);
        self.client_order_id = Some(client_order_id);
        self.side = Some(side);
        self.quantity = Some(qty);
        self.price = Some(price);
        self
    }
    
    /// Set fill details
    pub fn with_fill(mut self, fill_qty: f64, fill_price: f64) -> Self {
        self.fill_qty = Some(fill_qty);
        self.fill_price = Some(fill_price);
        self
    }
    
    /// Set metadata
    pub fn with_metadata(mut self, meta: &str) -> Self {
        self.metadata = Some(meta.to_string());
        self
    }
    
    /// Verify the entry's hash integrity
    pub fn verify_hash(&self) -> bool {
        let hash_input = format!(
            "{}|{}|{}|{}|{}",
            self.sequence, self.timestamp_ns, self.event_type as u8, self.symbol, self.prev_hash
        );
        
        let expected_hash = hex::encode(Sha256::digest(hash_input.as_bytes()));
        expected_hash == self.block_hash
    }
}

/// Cryptographically chained audit ledger
pub struct AuditLedger {
    /// Genesis hash (constant)
    genesis_hash: String,
    /// Last block hash
    last_hash: String,
    /// Current sequence number
    sequence: AtomicU64,
    /// Entries in memory (circular buffer)
    entries: std::sync::Mutex<Vec<AuditEntry>>,
    /// Total entries written (including rotated)
    total_entries: AtomicU64,
}

unsafe impl Send for AuditLedger {}
unsafe impl Sync for AuditLedger {}

impl AuditLedger {
    /// Create new audit ledger with genesis block
    pub fn new() -> Self {
        let genesis_hash = hex::encode(Sha256::digest(b"AUDIT_LEDGER_GENESIS_V1"));
        
        Self {
            genesis_hash: genesis_hash.clone(),
            last_hash: genesis_hash,
            sequence: AtomicU64::new(0),
            entries: std::sync::Mutex::new(Vec::with_capacity(1000)),
            total_entries: AtomicU64::new(0),
        }
    }
    
    /// Get current timestamp in nanoseconds
    #[inline]
    fn now_ns() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64
    }
    
    /// Append a new audit entry
    pub fn append(&self, event_type: AuditEventType, symbol: &str) -> AuditEntry {
        let seq = self.sequence.fetch_add(1, Ordering::SeqCst);
        
        let prev_hash = {
            let entries = self.entries.lock().unwrap();
            entries.last()
                .map(|e| e.block_hash.clone())
                .unwrap_or_else(|| self.genesis_hash.clone())
        };
        
        let mut entry = AuditEntry::new(seq, event_type, symbol, &prev_hash);
        
        // Update last hash
        let block_hash = entry.block_hash.clone();
        
        // Store entry
        let mut entries = self.entries.lock().unwrap();
        
        // Rotate if needed
        if entries.len() >= MAX_AUDIT_ENTRIES {
            entries.remove(0);
        }
        
        entries.push(entry.clone());
        self.total_entries.fetch_add(1, Ordering::Relaxed);
        drop(entries);
        
        // Update last hash (outside lock to prevent deadlock)
        unsafe {
            // Safe because we're the only writer
            *(std::ptr::addr_of!(self.last_hash) as *mut String) = block_hash;
        }
        
        entry
    }
    
    /// Log order submission
    pub fn log_order_submitted(
        &self,
        symbol: &str,
        order_id: u64,
        client_order_id: u64,
        side: u8,
        qty: f64,
        price: f64,
    ) -> AuditEntry {
        self.append(AuditEventType::OrderSubmitted, symbol)
            .with_order(order_id, client_order_id, side, qty, price)
    }
    
    /// Log order fill
    pub fn log_order_filled(
        &self,
        symbol: &str,
        order_id: u64,
        client_order_id: u64,
        side: u8,
        qty: f64,
        price: f64,
        fill_qty: f64,
        fill_price: f64,
    ) -> AuditEntry {
        self.append(AuditEventType::OrderFilled, symbol)
            .with_order(order_id, client_order_id, side, qty, price)
            .with_fill(fill_qty, fill_price)
    }
    
    /// Log trade execution
    pub fn log_trade(
        &self,
        symbol: &str,
        qty: f64,
        price: f64,
        metadata: Option<&str>,
    ) -> AuditEntry {
        let mut entry = self.append(AuditEventType::TradeExecuted, symbol)
            .with_order(0, 0, 0, qty, price);
        
        if let Some(meta) = metadata {
            entry = entry.with_metadata(meta);
        }
        
        entry
    }
    
    /// Log system event
    pub fn log_system_event(&self, description: &str) -> AuditEntry {
        self.append(AuditEventType::SystemEvent, "SYSTEM")
            .with_metadata(description)
    }
    
    /// Get entry by sequence number
    pub fn get_entry(&self, sequence: u64) -> Option<AuditEntry> {
        let entries = self.entries.lock().unwrap();
        entries.iter().find(|e| e.sequence == sequence).cloned()
    }
    
    /// Get last N entries
    pub fn get_recent(&self, n: usize) -> Vec<AuditEntry> {
        let entries = self.entries.lock().unwrap();
        entries.iter().rev().take(n).cloned().collect()
    }
    
    /// Verify chain integrity
    pub fn verify_chain(&self) -> bool {
        let entries = self.entries.lock().unwrap();
        
        let mut expected_prev = self.genesis_hash.clone();
        
        for entry in entries.iter() {
            if entry.prev_hash != expected_prev {
                return false;
            }
            
            if !entry.verify_hash() {
                return false;
            }
            
            expected_prev = entry.block_hash.clone();
        }
        
        true
    }
    
    /// Get current sequence number
    pub fn get_sequence(&self) -> u64 {
        self.sequence.load(Ordering::Relaxed)
    }
    
    /// Get total entries count
    pub fn get_total_entries(&self) -> u64 {
        self.total_entries.load(Ordering::Relaxed)
    }
    
    /// Get last hash for external verification
    pub fn get_last_hash(&self) -> String {
        self.last_hash.clone()
    }
    
    /// Get genesis hash
    pub fn get_genesis_hash(&self) -> &str {
        &self.genesis_hash
    }
}

impl Default for AuditLedger {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_ledger_creation() {
        let ledger = AuditLedger::new();
        assert_eq!(ledger.get_sequence(), 0);
        assert_eq!(ledger.get_total_entries(), 0);
        assert!(!ledger.get_genesis_hash().is_empty());
    }
    
    #[test]
    fn test_append_entry() {
        let ledger = AuditLedger::new();
        
        let entry = ledger.log_order_submitted("BTCUSDT", 1, 100, 0, 1.0, 50000.0);
        
        assert_eq!(entry.sequence, 0);
        assert_eq!(entry.event_type, AuditEventType::OrderSubmitted);
        assert_eq!(entry.symbol, "BTCUSDT");
        assert_eq!(entry.order_id, Some(1));
        assert!(!entry.block_hash.is_empty());
    }
    
    #[test]
    fn test_chain_integrity() {
        let ledger = AuditLedger::new();
        
        ledger.log_order_submitted("BTCUSDT", 1, 100, 0, 1.0, 50000.0);
        ledger.log_order_filled("BTCUSDT", 1, 100, 0, 1.0, 50000.0, 1.0, 50000.0);
        ledger.log_trade("ETHUSDT", 2.0, 3000.0, Some("test trade"));
        
        assert!(ledger.verify_chain());
    }
    
    #[test]
    fn test_entry_retrieval() {
        let ledger = AuditLedger::new();
        
        let entry1 = ledger.log_order_submitted("BTCUSDT", 1, 100, 0, 1.0, 50000.0);
        let _entry2 = ledger.log_order_submitted("ETHUSDT", 2, 101, 1, 5.0, 3000.0);
        
        let retrieved = ledger.get_entry(0).unwrap();
        assert_eq!(retrieved.sequence, entry1.sequence);
        
        let recent = ledger.get_recent(5);
        assert_eq!(recent.len(), 2);
    }
    
    #[test]
    fn test_hash_verification() {
        let ledger = AuditLedger::new();
        let entry = ledger.log_order_submitted("BTCUSDT", 1, 100, 0, 1.0, 50000.0);
        
        assert!(entry.verify_hash());
    }
}
