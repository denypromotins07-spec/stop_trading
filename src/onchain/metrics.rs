//! Network Metrics Collector
//! 
//! Calculates TVL, active addresses, and gas/priority fee oracles using rolling ring buffers.
//! Provides real-time network congestion metrics for execution urgency adjustment.
//! Memory-efficient design with bounded data structures.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// Ring buffer size for metrics history
const METRICS_BUFFER_SIZE: usize = 256;

/// Gas price percentiles for oracle calculation
const GAS_P50_IDX: usize = 128;
const GAS_P90_IDX: usize = 230;

/// Network type enumeration
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NetworkType {
    Ethereum,
    Solana,
    Arbitrum,
    Optimism,
    Polygon,
}

/// Current network metrics snapshot
#[derive(Debug, Clone)]
pub struct NetworkMetrics {
    pub network: NetworkType,
    pub block_number: u64,
    pub gas_price_gwei: u64,
    pub gas_price_p50_gwei: u64,
    pub gas_price_p90_gwei: u64,
    pub priority_fee_gwei: u64,
    pub base_fee_gwei: u64,
    pub tvl_usd: u64,
    pub active_addresses_24h: u64,
    pub transactions_per_block: usize,
    pub block_utilization_pct: f32,
    pub congestion_level: CongestionLevel,
    pub recommended_slippage_bps: u16,
    pub timestamp_ns: u64,
}

/// Network congestion levels
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CongestionLevel {
    Low,       // < 30% utilization
    Medium,    // 30-70% utilization
    High,      // 70-90% utilization
    Extreme,   // > 90% utilization
}

impl CongestionLevel {
    /// Get recommended slippage in basis points based on congestion
    pub fn recommended_slippage(&self) -> u16 {
        match self {
            CongestionLevel::Low => 10,      // 0.10%
            CongestionLevel::Medium => 30,   // 0.30%
            CongestionLevel::High => 50,     // 0.50%
            CongestionLevel::Extreme => 100, // 1.00%
        }
    }
    
    /// Get urgency multiplier for order execution
    pub fn urgency_multiplier(&self) -> f64 {
        match self {
            CongestionLevel::Low => 1.0,
            CongestionLevel::Medium => 1.2,
            CongestionLevel::High => 1.5,
            CongestionLevel::Extreme => 2.0,
        }
    }
}

/// Rolling ring buffer for gas prices (sorted for percentile calculation)
struct GasPriceBuffer {
    data: [u64; METRICS_BUFFER_SIZE],
    head: usize,
    count: usize,
    sorted_cache: Option<[u64; METRICS_BUFFER_SIZE]>,
    cache_valid: bool,
}

impl GasPriceBuffer {
    fn new() -> Self {
        Self {
            data: [0; METRICS_BUFFER_SIZE],
            head: 0,
            count: 0,
            sorted_cache: None,
            cache_valid: false,
        }
    }
    
    fn push(&mut self, value: u64) {
        self.data[self.head] = value;
        self.head = (self.head + 1) % METRICS_BUFFER_SIZE;
        if self.count < METRICS_BUFFER_SIZE {
            self.count += 1;
        }
        self.cache_valid = false;
    }
    
    fn get_percentile(&mut self, p: usize) -> u64 {
        if self.count == 0 {
            return 0;
        }
        
        if !self.cache_valid {
            self.rebuild_sorted_cache();
        }
        
        if let Some(ref sorted) = self.sorted_cache {
            let idx = (p * self.count / 100).min(self.count - 1);
            sorted[idx]
        } else {
            0
        }
    }
    
    fn average(&self) -> u64 {
        if self.count == 0 {
            return 0;
        }
        
        let sum: u64 = self.data.iter().take(self.count).sum();
        sum / self.count as u64
    }
    
    fn rebuild_sorted_cache(&mut self) {
        let mut sorted = [0u64; METRICS_BUFFER_SIZE];
        sorted[..self.count].copy_from_slice(&self.data[..self.count]);
        sorted[..self.count].sort_unstable();
        self.sorted_cache = Some(sorted);
        self.cache_valid = true;
    }
}

/// Active address tracker using hyperloglog-style approximation
struct ActiveAddressTracker {
    /// Simplified counter-based approximation for memory efficiency
    unique_addresses_1h: AtomicUsize,
    unique_addresses_24h: AtomicUsize,
    last_reset_1h: AtomicU64,
    last_reset_24h: AtomicU64,
}

impl ActiveAddressTracker {
    fn new() -> Self {
        Self {
            unique_addresses_1h: AtomicUsize::new(0),
            unique_addresses_24h: AtomicUsize::new(0),
            last_reset_1h: AtomicU64::new(get_timestamp_ns()),
            last_reset_24h: AtomicU64::new(get_timestamp_ns()),
        }
    }
    
    fn record_address(&self, _address: &[u8]) {
        // In production, would use HyperLogLog for cardinality estimation
        // For now, use simple atomic increment
        self.unique_addresses_1h.fetch_add(1, Ordering::Relaxed);
        self.unique_addresses_24h.fetch_add(1, Ordering::Relaxed);
    }
    
    fn get_24h_count(&self) -> u64 {
        self.unique_addresses_24h.load(Ordering::Relaxed) as u64
    }
    
    fn reset_if_needed(&self) {
        let now = get_timestamp_ns();
        let one_hour_ns = 3_600_000_000_000u64;
        let twenty_four_hours_ns = 86_400_000_000_000u64;
        
        let last_1h = self.last_reset_1h.load(Ordering::Relaxed);
        if now - last_1h > one_hour_ns {
            self.unique_addresses_1h.store(0, Ordering::Relaxed);
            self.last_reset_1h.store(now, Ordering::Relaxed);
        }
        
        let last_24h = self.last_reset_24h.load(Ordering::Relaxed);
        if now - last_24h > twenty_four_hours_ns {
            self.unique_addresses_24h.store(0, Ordering::Relaxed);
            self.last_reset_24h.store(now, Ordering::Relaxed);
        }
    }
}

/// Network metrics collector
pub struct NetworkMetricsCollector {
    network: NetworkType,
    
    /// Gas price rolling buffer
    gas_buffer: Arc<std::sync::Mutex<GasPriceBuffer>>,
    
    /// Priority fee buffer
    priority_fee_buffer: Arc<std::sync::Mutex<GasPriceBuffer>>,
    
    /// Active address tracker
    address_tracker: ActiveAddressTracker,
    
    /// Current block info
    current_block: AtomicU64,
    current_tvl: AtomicU64,
    
    /// Block utilization tracking
    gas_used_buffer: Arc<std::sync::Mutex<GasPriceBuffer>>,
    target_gas: AtomicU64,
    
    /// Last update timestamp
    last_update_ns: AtomicU64,
}

impl NetworkMetricsCollector {
    /// Create a new metrics collector for a specific network
    pub fn new() -> Self {
        Self::for_network(NetworkType::Ethereum)
    }
    
    /// Create a collector for a specific network type
    pub fn for_network(network: NetworkType) -> Self {
        let target_gas = match network {
            NetworkType::Ethereum => 30_000_000,
            NetworkType::Arbitrum => 32_000_000,
            NetworkType::Optimism => 30_000_000,
            NetworkType::Polygon => 30_000_000,
            NetworkType::Solana => 0, // Solana uses different model
        };
        
        Self {
            network,
            gas_buffer: Arc::new(std::sync::Mutex::new(GasPriceBuffer::new())),
            priority_fee_buffer: Arc::new(std::sync::Mutex::new(GasPriceBuffer::new())),
            address_tracker: ActiveAddressTracker::new(),
            current_block: AtomicU64::new(0),
            current_tvl: AtomicU64::new(50_000_000_000), // Default $50B TVL
            gas_used_buffer: Arc::new(std::sync::Mutex::new(GasPriceBuffer::new())),
            target_gas: AtomicU64::new(target_gas),
            last_update_ns: AtomicU64::new(get_timestamp_ns()),
        }
    }
    
    /// Record a new block
    pub fn record_block(&self, block: &crate::onchain::provider::EvmBlockData) {
        self.current_block.store(block.block_number, Ordering::Relaxed);
        
        // Record gas used for utilization calculation
        if let Ok(mut buffer) = self.gas_used_buffer.lock() {
            buffer.push(block.gas_used);
        }
        
        // Calculate base fee from block
        if let Some(base_fee) = block.base_fee_per_gas {
            self.record_gas_price(base_fee);
        }
        
        self.last_update_ns.store(get_timestamp_ns(), Ordering::Relaxed);
    }
    
    /// Record a Solana block
    pub fn record_solana_block(&self, block: &crate::onchain::provider::SolanaBlockData) {
        self.current_block.store(block.slot, Ordering::Relaxed);
        self.last_update_ns.store(get_timestamp_ns(), Ordering::Relaxed);
    }
    
    /// Record a gas price observation
    pub fn record_gas_price(&self, gas_price_gwei: u64) {
        if let Ok(mut buffer) = self.gas_buffer.lock() {
            buffer.push(gas_price_gwei);
        }
    }
    
    /// Record a priority fee observation
    pub fn record_priority_fee(&self, fee_gwei: u64) {
        if let Ok(mut buffer) = self.priority_fee_buffer.lock() {
            buffer.push(fee_gwei);
        }
    }
    
    /// Record an active address
    pub fn record_active_address(&self, address: &[u8]) {
        self.address_tracker.record_address(address);
        self.address_tracker.reset_if_needed();
    }
    
    /// Update TVL estimate
    pub fn update_tvl(&self, tvl_usd: u64) {
        self.current_tvl.store(tvl_usd, Ordering::Relaxed);
    }
    
    /// Get current network metrics
    pub fn get_current_metrics(&self) -> NetworkMetrics {
        let gas_avg = self.gas_buffer.lock().unwrap().average();
        let gas_p50 = self.gas_buffer.lock().unwrap().get_percentile(50);
        let gas_p90 = self.gas_buffer.lock().unwrap().get_percentile(90);
        
        let priority_fee = self.priority_fee_buffer.lock().unwrap().average();
        
        // Calculate block utilization
        let utilization = self.calculate_utilization();
        let congestion = CongestionLevel::from_utilization(utilization);
        
        NetworkMetrics {
            network: self.network,
            block_number: self.current_block.load(Ordering::Relaxed),
            gas_price_gwei: gas_avg,
            gas_price_p50_gwei: gas_p50,
            gas_price_p90_gwei: gas_p90,
            priority_fee_gwei: priority_fee,
            base_fee_gwei: gas_avg.saturating_sub(priority_fee),
            tvl_usd: self.current_tvl.load(Ordering::Relaxed),
            active_addresses_24h: self.address_tracker.get_24h_count(),
            transactions_per_block: 0, // Would track from blocks
            block_utilization_pct: utilization,
            congestion_level: congestion,
            recommended_slippage_bps: congestion.recommended_slippage(),
            timestamp_ns: get_timestamp_ns(),
        }
    }
    
    /// Get gas price oracle recommendation
    pub fn get_gas_oracle(&self) -> GasOracle {
        let mut buffer = self.gas_buffer.lock().unwrap();
        
        GasOracle {
            slow: buffer.get_percentile(25),
            standard: buffer.get_percentile(50),
            fast: buffer.get_percentile(75),
            instant: buffer.get_percentile(90),
        }
    }
    
    /// Calculate current block utilization percentage
    fn calculate_utilization(&self) -> f32 {
        let target = self.target_gas.load(Ordering::Relaxed);
        if target == 0 {
            return 50.0; // Default for Solana
        }
        
        let buffer = self.gas_used_buffer.lock().unwrap();
        let avg_used = buffer.average();
        
        (avg_used as f32 / target as f32) * 100.0
    }
}

impl Default for NetworkMetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// Gas price oracle recommendations
#[derive(Debug, Clone)]
pub struct GasOracle {
    pub slow: u64,      // ~25th percentile
    pub standard: u64,  // ~50th percentile
    pub fast: u64,      // ~75th percentile
    pub instant: u64,   // ~90th percentile
}

impl CongestionLevel {
    fn from_utilization(utilization: f32) -> Self {
        if utilization < 30.0 {
            CongestionLevel::Low
        } else if utilization < 70.0 {
            CongestionLevel::Medium
        } else if utilization < 90.0 {
            CongestionLevel::High
        } else {
            CongestionLevel::Extreme
        }
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
    fn test_collector_creation() {
        let collector = NetworkMetricsCollector::new();
        let metrics = collector.get_current_metrics();
        assert_eq!(metrics.network, NetworkType::Ethereum);
    }
    
    #[test]
    fn test_gas_buffer() {
        let mut buffer = GasPriceBuffer::new();
        
        for i in 0..100 {
            buffer.push(i * 10);
        }
        
        let p50 = buffer.get_percentile(50);
        assert!(p50 > 0);
        
        let avg = buffer.average();
        assert!(avg > 0);
    }
    
    #[test]
    fn test_congestion_levels() {
        assert_eq!(CongestionLevel::from_utilization(20.0), CongestionLevel::Low);
        assert_eq!(CongestionLevel::from_utilization(50.0), CongestionLevel::Medium);
        assert_eq!(CongestionLevel::from_utilization(80.0), CongestionLevel::High);
        assert_eq!(CongestionLevel::from_utilization(95.0), CongestionLevel::Extreme);
    }
    
    #[test]
    fn test_slippage_recommendation() {
        assert_eq!(CongestionLevel::Low.recommended_slippage(), 10);
        assert_eq!(CongestionLevel::Extreme.recommended_slippage(), 100);
    }
}
