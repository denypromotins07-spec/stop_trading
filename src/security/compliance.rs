//! Pre-Trade Compliance Engine
//! 
//! Checks counterparty addresses against OFAC sanctioned lists.
//! Uses lock-free Bloom filters for instant rejection of toxic addresses.
//! Integrated with global kill switch and order router.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// Bloom filter configuration for address filtering
const COMPLIANCE_BLOOM_SIZE: usize = 1 << 22; // 4MB
const COMPLIANCE_HASH_COUNT: usize = 5;

/// Compliance check result
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ComplianceResult {
    Approved,
    RejectedSanctioned,
    RejectedHighRisk,
    PendingReview,
    Unknown,
}

/// Risk level classification
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

/// Compliance check request
#[derive(Debug, Clone)]
pub struct ComplianceCheck {
    pub address: [u8; 20],
    pub chain_id: u64,
    pub transaction_type: TransactionType,
    pub amount_usd: Option<u64>,
    pub timestamp_ns: u64,
}

/// Transaction type enumeration
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TransactionType {
    Transfer,
    Swap,
    Bridge,
    Withdrawal,
    Deposit,
}

/// Compliance statistics
#[derive(Debug, Clone)]
pub struct ComplianceStats {
    pub total_checks: u64,
    pub approved: u64,
    pub rejected_sanctioned: u64,
    pub rejected_high_risk: u64,
    pub pending_review: u64,
    pub average_check_time_ns: u64,
}

/// Lock-free Bloom filter for sanctioned addresses
struct SanctionsFilter {
    bits: Vec<std::sync::atomic::AtomicU8>,
    size: usize,
    hash_count: usize,
}

impl SanctionsFilter {
    fn new(size: usize, hash_count: usize) -> Self {
        let bits = (0..size).map(|_| std::sync::atomic::AtomicU8::new(0)).collect();
        
        Self {
            bits,
            size,
            hash_count,
        }
    }
    
    fn insert(&self, address: &[u8]) {
        let hashes = self.compute_hashes(address);
        for hash in hashes {
            let idx = hash % self.size;
            self.bits[idx].store(1, Ordering::Relaxed);
        }
    }
    
    fn contains(&self, address: &[u8]) -> bool {
        let hashes = self.compute_hashes(address);
        for hash in hashes {
            let idx = hash % self.size;
            if self.bits[idx].load(Ordering::Relaxed) == 0 {
                return false;
            }
        }
        true
    }
    
    fn compute_hashes(&self, address: &[u8]) -> Vec<usize> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hashes = Vec::with_capacity(self.hash_count);
        
        let mut h1_hasher = DefaultHasher::new();
        address.hash(&mut h1_hasher);
        let h1 = h1_hasher.finish() as usize;
        
        let mut h2_hasher = DefaultHasher::new();
        address.iter().rev().copied().collect::<Vec<_>>().hash(&mut h2_hasher);
        let h2 = h2_hasher.finish() as usize;
        
        for i in 0..self.hash_count {
            let hash = h1.wrapping_add(i.wrapping_mul(h2));
            hashes.push(hash);
        }
        
        hashes
    }
}

/// High-risk address patterns (simplified)
struct HighRiskPatterns {
    /// Known mixer addresses
    mixer_prefixes: Vec<[u8; 4]>,
    /// Gambling-related prefixes
    gambling_prefixes: Vec<[u8; 4]>,
}

impl HighRiskPatterns {
    fn new() -> Self {
        Self {
            // Simplified patterns - would be populated from threat intelligence
            mixer_prefixes: vec![
                [0x00, 0x00, 0x00, 0x00], // Tornado Cash-like
            ],
            gambling_prefixes: vec![],
        }
    }
    
    fn is_mixer(&self, address: &[u8; 20]) -> bool {
        self.mixer_prefixes.iter().any(|prefix| {
            address.starts_with(prefix)
        })
    }
    
    fn is_gambling(&self, address: &[u8; 20]) -> bool {
        self.gambling_prefixes.iter().any(|prefix| {
            address.starts_with(prefix)
        })
    }
}

/// Pre-trade compliance engine
pub struct ComplianceEngine {
    /// Sanctions Bloom filter
    sanctions_filter: Arc<SanctionsFilter>,
    
    /// High-risk pattern matcher
    high_risk_patterns: HighRiskPatterns,
    
    /// Statistics counters
    total_checks: AtomicU64,
    approved_count: AtomicU64,
    rejected_sanctioned: AtomicU64,
    rejected_high_risk: AtomicU64,
    pending_count: AtomicU64,
    
    /// Total check time for averaging
    total_check_time_ns: AtomicU64,
    
    /// Kill switch state
    kill_switch_active: std::sync::atomic::AtomicBool,
}

impl ComplianceEngine {
    /// Create a new compliance engine
    pub fn new() -> Self {
        Self {
            sanctions_filter: Arc::new(SanctionsFilter::new(COMPLIANCE_BLOOM_SIZE, COMPLIANCE_HASH_COUNT)),
            high_risk_patterns: HighRiskPatterns::new(),
            total_checks: AtomicU64::new(0),
            approved_count: AtomicU64::new(0),
            rejected_sanctioned: AtomicU64::new(0),
            rejected_high_risk: AtomicU64::new(0),
            pending_count: AtomicU64::new(0),
            total_check_time_ns: AtomicU64::new(0),
            kill_switch_active: std::sync::atomic::AtomicBool::new(false),
        }
    }
    
    /// Add an address to the sanctions list
    pub fn add_sanctioned_address(&self, address: [u8; 20]) {
        self.sanctions_filter.insert(&address);
    }
    
    /// Load OFAC sanctioned addresses (would load from file/API in production)
    pub fn load_ofac_list(&self, addresses: &[[u8; 20]]) {
        for addr in addresses {
            self.add_sanctioned_address(*addr);
        }
    }
    
    /// Perform pre-trade compliance check
    pub fn check(&self, check: ComplianceCheck) -> ComplianceResult {
        let start = get_timestamp_ns();
        
        // Check kill switch first
        if self.kill_switch_active.load(Ordering::Relaxed) {
            return ComplianceResult::RejectedHighRisk;
        }
        
        self.total_checks.fetch_add(1, Ordering::Relaxed);
        
        // Check against sanctions list (fastest check)
        if self.sanctions_filter.contains(&check.address) {
            self.rejected_sanctioned.fetch_add(1, Ordering::Relaxed);
            self.record_check_time(start);
            return ComplianceResult::RejectedSanctioned;
        }
        
        // Check high-risk patterns
        if self.high_risk_patterns.is_mixer(&check.address) {
            self.rejected_high_risk.fetch_add(1, Ordering::Relaxed);
            self.record_check_time(start);
            return ComplianceResult::RejectedHighRisk;
        }
        
        // Check transaction amount thresholds
        if let Some(amount) = check.amount_usd {
            if amount > 10_000_000 { // $10M threshold
                self.pending_count.fetch_add(1, Ordering::Relaxed);
                self.record_check_time(start);
                return ComplianceResult::PendingReview;
            }
        }
        
        // Default approve
        self.approved_count.fetch_add(1, Ordering::Relaxed);
        self.record_check_time(start);
        ComplianceResult::Approved
    }
    
    /// Quick check without full ComplianceCheck struct
    pub fn quick_check(&self, address: &[u8; 20]) -> ComplianceResult {
        if self.kill_switch_active.load(Ordering::Relaxed) {
            return ComplianceResult::RejectedHighRisk;
        }
        
        self.total_checks.fetch_add(1, Ordering::Relaxed);
        
        if self.sanctions_filter.contains(address) {
            self.rejected_sanctioned.fetch_add(1, Ordering::Relaxed);
            return ComplianceResult::RejectedSanctioned;
        }
        
        if self.high_risk_patterns.is_mixer(address) {
            self.rejected_high_risk.fetch_add(1, Ordering::Relaxed);
            return ComplianceResult::RejectedHighRisk;
        }
        
        self.approved_count.fetch_add(1, Ordering::Relaxed);
        ComplianceResult::Approved
    }
    
    /// Activate kill switch (blocks all transactions)
    pub fn activate_kill_switch(&self) {
        self.kill_switch_active.store(true, Ordering::Relaxed);
    }
    
    /// Deactivate kill switch
    pub fn deactivate_kill_switch(&self) {
        self.kill_switch_active.store(false, Ordering::Relaxed);
    }
    
    /// Check if kill switch is active
    pub fn is_kill_switch_active(&self) -> bool {
        self.kill_switch_active.load(Ordering::Relaxed)
    }
    
    /// Get compliance statistics
    pub fn get_stats(&self) -> ComplianceStats {
        let total = self.total_checks.load(Ordering::Relaxed);
        ComplianceStats {
            total_checks: total,
            approved: self.approved_count.load(Ordering::Relaxed),
            rejected_sanctioned: self.rejected_sanctioned.load(Ordering::Relaxed),
            rejected_high_risk: self.rejected_high_risk.load(Ordering::Relaxed),
            pending_review: self.pending_count.load(Ordering::Relaxed),
            average_check_time_ns: if total > 0 {
                self.total_check_time_ns.load(Ordering::Relaxed) / total
            } else {
                0
            },
        }
    }
    
    /// Get risk level for an address
    pub fn get_risk_level(&self, address: &[u8; 20]) -> RiskLevel {
        if self.sanctions_filter.contains(address) {
            return RiskLevel::Critical;
        }
        
        if self.high_risk_patterns.is_mixer(address) {
            return RiskLevel::High;
        }
        
        if self.high_risk_patterns.is_gambling(address) {
            return RiskLevel::Medium;
        }
        
        RiskLevel::Low
    }
    
    /// Record check time for statistics
    fn record_check_time(&self, start_ns: u64) {
        let elapsed = get_timestamp_ns().saturating_sub(start_ns);
        self.total_check_time_ns.fetch_add(elapsed, Ordering::Relaxed);
    }
}

impl Default for ComplianceEngine {
    fn default() -> Self {
        Self::new()
    }
}

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
    fn test_engine_creation() {
        let engine = ComplianceEngine::new();
        assert!(!engine.is_kill_switch_active());
        
        let stats = engine.get_stats();
        assert_eq!(stats.total_checks, 0);
    }
    
    #[test]
    fn test_compliance_check_approved() {
        let engine = ComplianceEngine::new();
        
        let check = ComplianceCheck {
            address: [0x11u8; 20],
            chain_id: 1,
            transaction_type: TransactionType::Transfer,
            amount_usd: Some(1000),
            timestamp_ns: get_timestamp_ns(),
        };
        
        let result = engine.check(check);
        assert_eq!(result, ComplianceResult::Approved);
    }
    
    #[test]
    fn test_kill_switch() {
        let engine = ComplianceEngine::new();
        
        engine.activate_kill_switch();
        assert!(engine.is_kill_switch_active());
        
        let check = ComplianceCheck {
            address: [0x11u8; 20],
            chain_id: 1,
            transaction_type: TransactionType::Transfer,
            amount_usd: None,
            timestamp_ns: get_timestamp_ns(),
        };
        
        let result = engine.check(check);
        assert_eq!(result, ComplianceResult::RejectedHighRisk);
        
        engine.deactivate_kill_switch();
        assert!(!engine.is_kill_switch_active());
    }
    
    #[test]
    fn test_risk_levels() {
        let engine = ComplianceEngine::new();
        
        // Normal address should be low risk
        assert_eq!(engine.get_risk_level(&[0x11u8; 20]), RiskLevel::Low);
    }
}
