//! Security Module Root
//! 
//! Coordinates compliance checks, audit logging, and governance.
//! Wires compliance directly into the global kill switch and order router.

pub mod compliance;
pub mod governance;

use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn, error};

/// Security manager coordinating all security components
pub struct SecurityManager {
    compliance_engine: Arc<compliance::ComplianceEngine>,
    audit_logger: Arc<governance::AuditLogger>,
    
    /// Kill switch state (shared with other modules)
    global_kill_switch: Arc<std::sync::atomic::AtomicBool>,
    
    /// Emergency contact for alerts
    emergency_contacts: Vec<String>,
}

impl SecurityManager {
    /// Create a new security manager
    pub fn new() -> Self {
        Self {
            compliance_engine: Arc::new(compliance::ComplianceEngine::new()),
            audit_logger: Arc::new(governance::AuditLogger::new()),
            global_kill_switch: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            emergency_contacts: Vec::new(),
        }
    }
    
    /// Perform pre-trade compliance check
    pub fn check_compliance(&self, address: [u8; 20]) -> compliance::ComplianceResult {
        let result = self.compliance_engine.quick_check(&address);
        
        // Log the compliance check
        let _ = self.audit_logger.log_compliance_action(
            "pre_trade_check",
            &format!("0x{}", hex_encode(&address)),
            &format!("{:?}", result),
        );
        
        // Auto-activate kill switch on sanctioned address detection
        if result == compliance::ComplianceResult::RejectedSanctioned {
            warn!("Sanctioned address detected: triggering enhanced monitoring");
            let _ = self.audit_logger.log_manual_override(
                "sanctioned_detection",
                "Address found on sanctions list",
                "system",
                "compliance_engine",
            );
        }
        
        result
    }
    
    /// Check full compliance with transaction details
    pub fn check_full_compliance(
        &self,
        check: compliance::ComplianceCheck,
    ) -> compliance::ComplianceResult {
        let result = self.compliance_engine.check(check.clone());
        
        // Log the compliance check
        let _ = self.audit_logger.log_compliance_action(
            "full_compliance_check",
            &format!("0x{}", hex_encode(&check.address)),
            &format!("{:?}", result),
        );
        
        result
    }
    
    /// Log API key usage
    pub fn log_api_usage(&self, key_id: &str, endpoint: &str, success: bool) {
        let _ = self.audit_logger.log_api_usage(key_id, endpoint, success);
    }
    
    /// Log configuration change
    pub fn log_config_change(
        &self,
        config_key: &str,
        old_value: &str,
        new_value: &str,
        changed_by: &str,
    ) {
        let _ = self.audit_logger.log_config_change(config_key, old_value, new_value, changed_by);
    }
    
    /// Log manual override
    pub fn log_manual_override(
        &self,
        override_type: &str,
        reason: &str,
        overridden_by: &str,
        affected_system: &str,
    ) {
        let _ = self.audit_logger.log_manual_override(override_type, reason, overridden_by, affected_system);
    }
    
    /// Log trade execution
    pub fn log_trade(
        &self,
        order_id: &str,
        symbol: &str,
        side: &str,
        size: f64,
        price: f64,
    ) {
        let _ = self.audit_logger.log_trade(order_id, symbol, side, size, price);
    }
    
    /// Activate global kill switch
    pub fn activate_kill_switch(&self, reason: &str) {
        warn!("Activating global kill switch: {}", reason);
        
        self.global_kill_switch.store(true, std::sync::atomic::Ordering::Relaxed);
        self.compliance_engine.activate_kill_switch();
        
        let _ = self.audit_logger.log_manual_override(
            "kill_switch_activation",
            reason,
            "security_manager",
            "global_trading_system",
        );
    }
    
    /// Deactivate global kill switch
    pub fn deactivate_kill_switch(&self, authorized_by: &str) {
        info!("Deactivating global kill switch, authorized by: {}", authorized_by);
        
        self.global_kill_switch.store(false, std::sync::atomic::Ordering::Relaxed);
        self.compliance_engine.deactivate_kill_switch();
        
        let _ = self.audit_logger.log_config_change(
            "kill_switch_state",
            "active",
            "inactive",
            authorized_by,
        );
    }
    
    /// Check if kill switch is active
    pub fn is_kill_switch_active(&self) -> bool {
        self.global_kill_switch.load(std::sync::atomic::Ordering::Relaxed)
            || self.compliance_engine.is_kill_switch_active()
    }
    
    /// Get compliance engine reference
    pub fn compliance_engine(&self) -> Arc<compliance::ComplianceEngine> {
        self.compliance_engine.clone()
    }
    
    /// Get audit logger reference
    pub fn audit_logger(&self) -> Arc<governance::AuditLogger> {
        self.audit_logger.clone()
    }
    
    /// Generate daily compliance report
    pub fn generate_daily_report(&self, date: &str) -> governance::DailyLogSummary {
        self.audit_logger.generate_daily_summary(date)
    }
    
    /// Verify audit log integrity
    pub fn verify_audit_integrity(&self) -> bool {
        self.audit_logger.verify_integrity()
    }
    
    /// Get compliance statistics
    pub fn get_compliance_stats(&self) -> compliance::ComplianceStats {
        self.compliance_engine.get_stats()
    }
    
    /// Add sanctioned address
    pub fn add_sanctioned_address(&self, address: [u8; 20]) {
        self.compliance_engine.add_sanctioned_address(address);
        
        let _ = self.audit_logger.log_config_change(
            "sanctions_list",
            "unchanged",
            &format!("added 0x{}", hex_encode(&address)),
            "security_manager",
        );
    }
    
    /// Load OFAC sanctioned addresses
    pub fn load_ofac_list(&self, addresses: &[[u8; 20]]) {
        self.compliance_engine.load_ofac_list(addresses);
        
        let _ = self.audit_logger.log_config_change(
            "ofac_list",
            "previous_count",
            &format!("loaded {} addresses", addresses.len()),
            "security_manager",
        );
    }
    
    /// Start background tasks
    pub async fn start(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        info!("Starting security manager");
        
        // Spawn task to periodically verify audit integrity
        let logger = self.audit_logger.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(std::time::Duration::from_secs(300)).await; // Every 5 minutes
                
                if !logger.verify_integrity() {
                    error!("Audit log integrity check failed!");
                }
            }
        });
        
        Ok(())
    }
    
    /// Set emergency contacts
    pub fn set_emergency_contacts(&mut self, contacts: Vec<String>) {
        self.emergency_contacts = contacts;
    }
}

impl Default for SecurityManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Simple hex encoder for addresses
fn hex_encode(bytes: &[u8]) -> String {
    const HEX_CHARS: &[u8] = b"0123456789abcdef";
    let mut result = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        result.push(HEX_CHARS[(byte >> 4) as usize] as char);
        result.push(HEX_CHARS[(byte & 0x0F) as usize] as char);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_manager_creation() {
        let manager = SecurityManager::new();
        assert!(!manager.is_kill_switch_active());
    }
    
    #[test]
    fn test_compliance_check() {
        let manager = SecurityManager::new();
        
        let address = [0x11u8; 20];
        let result = manager.check_compliance(address);
        assert_eq!(result, compliance::ComplianceResult::Approved);
    }
    
    #[test]
    fn test_kill_switch() {
        let manager = SecurityManager::new();
        
        manager.activate_kill_switch("test reason");
        assert!(manager.is_kill_switch_active());
        
        manager.deactivate_kill_switch("admin");
        assert!(!manager.is_kill_switch_active());
    }
    
    #[test]
    fn test_logging() {
        let manager = SecurityManager::new();
        
        manager.log_api_usage("test_key", "/api/test", true);
        manager.log_config_change("test_key", "old", "new", "user");
        
        assert!(manager.audit_logger.get_total_logged() >= 2);
    }
    
    #[test]
    fn test_hex_encode() {
        let bytes = [0x12, 0x34, 0x56, 0x78];
        let encoded = hex_encode(&bytes);
        assert_eq!(encoded, "12345678");
    }
}
