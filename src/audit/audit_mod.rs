//! Audit Module Root
//! 
//! Wires the cryptographic ledger directly to the disk writer and execution gateway.
//! Provides unified interface for audit logging and SOUL.md integrity monitoring.

pub mod ledger;
pub mod soul_hash;

pub use ledger::{AuditLedger, AuditEntry, AuditEventType, MAX_AUDIT_ENTRIES};
pub use soul_hash::{
    SoulMonitor, SoulHashState, SoulHashError,
    SOUL_FILENAME, CHECK_INTERVAL_SECS,
    init_soul_monitor, get_soul_monitor, verify_soul_integrity, should_halt,
};

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

/// Global audit ledger instance
static GLOBAL_AUDIT_LEDGER: OnceLock<AuditLedger> = OnceLock::new();

/// Audit system initialization flag
static AUDIT_INITIALIZED: AtomicBool = AtomicBool::new(false);

/// Initialize the audit subsystem
pub fn init_audit<P: AsRef<std::path::Path>>(soul_path: P) -> Result<(), &'static str> {
    if AUDIT_INITIALIZED.load(Ordering::SeqCst) {
        return Err("Audit system already initialized");
    }
    
    // Initialize SOUL monitor
    init_soul_monitor(soul_path)
        .map_err(|_| "Failed to initialize SOUL monitor")?;
    
    // Initialize audit ledger
    GLOBAL_AUDIT_LEDGER
        .set(AuditLedger::new())
        .map_err(|_| "Failed to set global audit ledger")?;
    
    AUDIT_INITIALIZED.store(true, Ordering::SeqCst);
    Ok(())
}

/// Shutdown the audit subsystem
pub fn shutdown_audit() -> Result<(), &'static str> {
    if !AUDIT_INITIALIZED.load(Ordering::SeqCst) {
        return Err("Audit system not initialized");
    }
    
    AUDIT_INITIALIZED.store(false, Ordering::SeqCst);
    Ok(())
}

/// Check if audit system is initialized
#[inline]
pub fn is_audit_initialized() -> bool {
    AUDIT_INITIALIZED.load(Ordering::Relaxed)
}

/// Get reference to global audit ledger
pub fn get_audit_ledger() -> Result<&'static AuditLedger, &'static str> {
    GLOBAL_AUDIT_LEDGER
        .get()
        .ok_or("Audit ledger not initialized")
}

/// Log order submission to audit ledger
pub fn audit_order_submitted(
    symbol: &str,
    order_id: u64,
    client_order_id: u64,
    side: u8,
    qty: f64,
    price: f64,
) -> Result<AuditEntry, &'static str> {
    let ledger = get_audit_ledger()?;
    Ok(ledger.log_order_submitted(symbol, order_id, client_order_id, side, qty, price))
}

/// Log order fill to audit ledger
pub fn audit_order_filled(
    symbol: &str,
    order_id: u64,
    client_order_id: u64,
    side: u8,
    qty: f64,
    price: f64,
    fill_qty: f64,
    fill_price: f64,
) -> Result<AuditEntry, &'static str> {
    let ledger = get_audit_ledger()?;
    Ok(ledger.log_order_filled(symbol, order_id, client_order_id, side, qty, price, fill_qty, fill_price))
}

/// Log trade execution to audit ledger
pub fn audit_trade(
    symbol: &str,
    qty: f64,
    price: f64,
    metadata: Option<&str>,
) -> Result<AuditEntry, &'static str> {
    let ledger = get_audit_ledger()?;
    Ok(ledger.log_trade(symbol, qty, price, metadata))
}

/// Log system event to audit ledger
pub fn audit_system_event(description: &str) -> Result<AuditEntry, &'static str> {
    let ledger = get_audit_ledger()?;
    Ok(ledger.log_system_event(description))
}

/// Verify SOUL.md integrity
pub fn check_soul_integrity() -> Result<bool, SoulHashError> {
    verify_soul_integrity()
}

/// Check if system should halt due to integrity failure
pub fn check_halt_required() -> bool {
    should_halt() || get_soul_monitor().map(|m| m.is_halted()).unwrap_or(false)
}

/// Get current audit sequence number
pub fn get_audit_sequence() -> u64 {
    get_audit_ledger()
        .map(|l| l.get_sequence())
        .unwrap_or(0)
}

/// Get total audit entries count
pub fn get_audit_count() -> u64 {
    get_audit_ledger()
        .map(|l| l.get_total_entries())
        .unwrap_or(0)
}

/// Verify full audit chain integrity
pub fn verify_audit_chain() -> bool {
    get_audit_ledger()
        .map(|l| l.verify_chain())
        .unwrap_or(false)
}

/// Audit builder for fluent initialization
pub struct AuditBuilder {
    soul_path: Option<std::path::PathBuf>,
    enable_chain_verification: bool,
}

impl AuditBuilder {
    pub fn new() -> Self {
        Self {
            soul_path: Some(std::path::PathBuf::from(SOUL_FILENAME)),
            enable_chain_verification: true,
        }
    }
    
    pub fn with_soul_path<P: Into<std::path::PathBuf>>(mut self, path: P) -> Self {
        self.soul_path = Some(path.into());
        self
    }
    
    pub fn with_chain_verification(mut self, enabled: bool) -> Self {
        self.enable_chain_verification = enabled;
        self
    }
    
    pub fn build(self) -> Result<(), &'static str> {
        let soul_path = self.soul_path.ok_or("SOUL path required")?;
        init_audit(soul_path)?;
        
        if self.enable_chain_verification {
            // Perform initial chain verification (will be empty at startup)
            if let Ok(ledger) = get_audit_ledger() {
                assert!(ledger.verify_chain(), "Initial chain verification failed");
            }
        }
        
        Ok(())
    }
}

impl Default for AuditBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    use std::io::Write;
    
    #[test]
    fn test_audit_initialization() {
        let temp = NamedTempFile::new().unwrap();
        temp.as_file().write_all(b"initial soul content").unwrap();
        
        assert!(!is_audit_initialized());
        assert!(init_audit(temp.path()).is_ok());
        assert!(is_audit_initialized());
        
        shutdown_audit().unwrap();
    }
    
    #[test]
    fn test_audit_logging() {
        let temp = NamedTempFile::new().unwrap();
        temp.as_file().write_all(b"soul content").unwrap();
        
        init_audit(temp.path()).unwrap();
        
        let entry = audit_order_submitted("BTCUSDT", 1, 100, 0, 1.0, 50000.0).unwrap();
        assert_eq!(entry.event_type, AuditEventType::OrderSubmitted);
        
        let entry2 = audit_trade("ETHUSDT", 2.0, 3000.0, Some("test")).unwrap();
        assert_eq!(entry2.event_type, AuditEventType::TradeExecuted);
        
        assert_eq!(get_audit_sequence(), 2);
        
        shutdown_audit().unwrap();
    }
    
    #[test]
    fn test_chain_verification() {
        let temp = NamedTempFile::new().unwrap();
        temp.as_file().write_all(b"soul").unwrap();
        
        init_audit(temp.path()).unwrap();
        
        audit_order_submitted("BTCUSDT", 1, 100, 0, 1.0, 50000.0).unwrap();
        audit_order_filled("BTCUSDT", 1, 100, 0, 1.0, 50000.0, 1.0, 50000.0).unwrap();
        
        assert!(verify_audit_chain());
        
        shutdown_audit().unwrap();
    }
    
    #[test]
    fn test_soul_integrity_check() {
        let temp = NamedTempFile::new().unwrap();
        temp.as_file().write_all(b"immutable soul").unwrap();
        
        init_audit(temp.path()).unwrap();
        
        assert!(check_soul_integrity().is_ok());
        assert!(!check_halt_required());
        
        shutdown_audit().unwrap();
    }
    
    #[test]
    fn test_builder_pattern() {
        let temp = NamedTempFile::new().unwrap();
        temp.as_file().write_all(b"soul").unwrap();
        
        let result = AuditBuilder::new()
            .with_soul_path(temp.path())
            .with_chain_verification(true)
            .build();
        
        assert!(result.is_ok());
        shutdown_audit().unwrap();
    }
}
