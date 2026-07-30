//! Lock-Free Whale Clustering Engine
//! 
//! Identifies large wallet movements and exchange inflows/outflows in real-time.
//! Uses Bloom filters for efficient exchange hot wallet tracking.
//! Implements lock-free data structures to prevent thread contention.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use crossbeam_channel::{bounded, Sender, Receiver};
use dashmap::DashMap;

/// Threshold for whale transaction detection (in USD equivalent)
const WHALE_THRESHOLD_USD: u64 = 100_000;

/// Maximum number of tracked wallets
const MAX_TRacked_WALLETS: usize = 100_000;

/// Channel capacity for whale alerts
const ALERT_CHANNEL_CAPACITY: usize = 1000;

/// Bloom filter configuration
const BLOOM_FILTER_SIZE: usize = 1 << 20; // 1MB
const BLOOM_HASH_COUNT: usize = 7;

/// Whale alert types
#[derive(Debug, Clone)]
pub enum WhaleAlertType {
    LargeTransfer,
    ExchangeInflow,
    ExchangeOutflow,
    ClusterMovement,
    StablecoinMint,
    StablecoinBurn,
}

/// Whale alert event
#[derive(Debug, Clone)]
pub struct WhaleAlert {
    pub alert_type: WhaleAlertType,
    pub from_address: [u8; 20],
    pub to_address: [u8; 20],
    pub value_usd: u64,
    pub token_symbol: &'static str,
    pub timestamp_ns: u64,
    pub cluster_id: Option<u64>,
    pub confidence_score: f32,
}

/// Wallet cluster information
#[derive(Debug, Clone)]
pub struct WalletCluster {
    pub cluster_id: u64,
    pub wallet_count: usize,
    pub total_value_usd: u64,
    pub last_activity_ns: u64,
    pub is_exchange: bool,
    pub exchange_name: Option<&'static str>,
}

/// Lock-free Bloom filter for exchange wallet tracking
pub struct BloomFilter {
    bits: Vec<AtomicUsize>,
    size: usize,
    hash_count: usize,
}

impl BloomFilter {
    /// Create a new Bloom filter
    pub fn new(size: usize, hash_count: usize) -> Self {
        let num_words = size / mem::size_of::<usize>();
        let bits = (0..num_words).map(|_| AtomicUsize::new(0)).collect();
        
        Self {
            bits,
            size,
            hash_count,
        }
    }
    
    /// Insert an address into the filter
    pub fn insert(&self, address: &[u8]) {
        let hashes = self.compute_hashes(address);
        for hash in hashes {
            let word_idx = hash / mem::size_of::<usize>();
            let bit_idx = hash % mem::size_of::<usize>();
            
            if word_idx < self.bits.len() {
                self.bits[word_idx].fetch_or(1 << bit_idx, Ordering::Relaxed);
            }
        }
    }
    
    /// Check if an address might be in the filter
    pub fn contains(&self, address: &[u8]) -> bool {
        let hashes = self.compute_hashes(address);
        for hash in hashes {
            let word_idx = hash / mem::size_of::<usize>();
            let bit_idx = hash % mem::size_of::<usize>();
            
            if word_idx >= self.bits.len() {
                return false;
            }
            
            let word = self.bits[word_idx].load(Ordering::Relaxed);
            if word & (1 << bit_idx) == 0 {
                return false;
            }
        }
        true
    }
    
    /// Compute multiple hash values for an address
    fn compute_hashes(&self, address: &[u8]) -> Vec<usize> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hashes = Vec::with_capacity(self.hash_count);
        
        // Use double hashing technique: h(i) = h1 + i * h2
        let mut h1_hasher = DefaultHasher::new();
        address.hash(&mut h1_hasher);
        let h1 = h1_hasher.finish() as usize;
        
        let mut h2_hasher = DefaultHasher::new();
        address.iter().rev().copied().collect::<Vec<_>>().hash(&mut h2_hasher);
        let h2 = h2_hasher.finish() as usize;
        
        for i in 0..self.hash_count {
            let hash = (h1.wrapping_add(i.wrapping_mul(h2))) % self.size;
            hashes.push(hash);
        }
        
        hashes
    }
}

use std::mem;

/// Whale clustering engine with lock-free data structures
pub struct WhaleClusterEngine {
    /// Bloom filter for known exchange wallets
    exchange_filter: Arc<BloomFilter>,
    
    /// Map of wallet addresses to cluster IDs (lock-free)
    wallet_clusters: DashMap<[u8; 20], u64>,
    
    /// Cluster metadata
    cluster_metadata: DashMap<u64, WalletCluster>,
    
    /// Next cluster ID counter
    next_cluster_id: AtomicU64,
    
    /// Alert channel for whale movements
    alert_tx: Sender<WhaleAlert>,
    alert_rx: Receiver<WhaleAlert>,
    
    /// Statistics counters
    total_alerts: AtomicUsize,
    large_transfers_detected: AtomicUsize,
    exchange_flows_detected: AtomicUsize,
}

impl WhaleClusterEngine {
    /// Create a new whale cluster engine
    pub fn new(known_exchange_wallets: Vec<[u8; 20]>) -> Self {
        let bloom_filter = Arc::new(BloomFilter::new(BLOOM_FILTER_SIZE, BLOOM_HASH_COUNT));
        
        // Initialize Bloom filter with known exchange wallets
        for wallet in &known_exchange_wallets {
            bloom_filter.insert(wallet.as_slice());
        }
        
        let (alert_tx, alert_rx) = bounded(ALERT_CHANNEL_CAPACITY);
        
        Self {
            exchange_filter: bloom_filter,
            wallet_clusters: DashMap::with_capacity(MAX_TRacked_WALLETS),
            cluster_metadata: DashMap::new(),
            next_cluster_id: AtomicU64::new(1),
            alert_tx,
            alert_rx,
            total_alerts: AtomicUsize::new(0),
            large_transfers_detected: AtomicUsize::new(0),
            exchange_flows_detected: AtomicUsize::new(0),
        }
    }
    
    /// Process a transfer event for whale detection
    pub fn process_transfer(&self, transfer: &crate::onchain::decoder::TransferEvent) {
        let value_usd = self.estimate_usd_value(transfer.value, &transfer.token_address);
        
        // Check if this is a whale-sized transfer
        if value_usd >= WHALE_THRESHOLD_USD {
            self.large_transfers_detected.fetch_add(1, Ordering::Relaxed);
            
            let from_is_exchange = self.exchange_filter.contains(&transfer.from);
            let to_is_exchange = self.exchange_filter.contains(&transfer.to);
            
            let alert_type = match (from_is_exchange, to_is_exchange) {
                (true, false) => {
                    self.exchange_flows_detected.fetch_add(1, Ordering::Relaxed);
                    WhaleAlertType::ExchangeOutflow
                }
                (false, true) => {
                    self.exchange_flows_detected.fetch_add(1, Ordering::Relaxed);
                    WhaleAlertType::ExchangeInflow
                }
                _ => WhaleAlertType::LargeTransfer,
            };
            
            let alert = WhaleAlert {
                alert_type,
                from_address: transfer.from,
                to_address: transfer.to,
                value_usd,
                token_symbol: self.get_token_symbol(&transfer.token_address),
                timestamp_ns: get_timestamp_ns(),
                cluster_id: self.get_or_create_cluster(&transfer.from),
                confidence_score: self.calculate_confidence(value_usd, from_is_exchange, to_is_exchange),
            };
            
            // Try to send alert (non-blocking)
            let _ = self.alert_tx.try_send(alert);
            self.total_alerts.fetch_add(1, Ordering::Relaxed);
        }
        
        // Update cluster tracking
        self.update_cluster_activity(&transfer.from);
        self.update_cluster_activity(&transfer.to);
    }
    
    /// Process a swap event
    pub fn process_swap(&self, swap: &crate::onchain::decoder::SwapEvent) {
        let total_value_usd = self.estimate_usd_value(swap.amount0_in, &swap.pool_address)
            + self.estimate_usd_value(swap.amount1_in, &swap.pool_address);
        
        if total_value_usd >= WHALE_THRESHOLD_USD {
            let alert = WhaleAlert {
                alert_type: WhaleAlertType::LargeTransfer,
                from_address: swap.sender,
                to_address: swap.to,
                value_usd: total_value_usd,
                token_symbol: "SWAP",
                timestamp_ns: get_timestamp_ns(),
                cluster_id: self.get_or_create_cluster(&swap.sender),
                confidence_score: 0.8,
            };
            
            let _ = self.alert_tx.try_send(alert);
            self.total_alerts.fetch_add(1, Ordering::Relaxed);
        }
    }
    
    /// Get receiver for whale alerts
    pub fn alert_receiver(&self) -> Receiver<WhaleAlert> {
        self.alert_rx.clone()
    }
    
    /// Check if an address is a known exchange wallet
    pub fn is_exchange_wallet(&self, address: &[u8; 20]) -> bool {
        self.exchange_filter.contains(address.as_slice())
    }
    
    /// Add a new exchange wallet to the filter
    pub fn add_exchange_wallet(&self, address: [u8; 20]) {
        self.exchange_filter.insert(&address);
    }
    
    /// Get statistics
    pub fn get_stats(&self) -> WhaleStats {
        WhaleStats {
            total_alerts: self.total_alerts.load(Ordering::Relaxed),
            large_transfers: self.large_transfers_detected.load(Ordering::Relaxed),
            exchange_flows: self.exchange_flows_detected.load(Ordering::Relaxed),
            tracked_wallets: self.wallet_clusters.len(),
            cluster_count: self.cluster_metadata.len(),
        }
    }
    
    /// Estimate USD value of a token amount
    fn estimate_usd_value(&self, amount: u128, token_address: &[u8; 20]) -> u64 {
        // Simplified estimation - in production would use price oracle
        // Assuming 18 decimals and $1 per token as baseline
        let decimals = self.get_token_decimals(token_address);
        let adjusted_amount = amount / 10u128.pow(decimals as u32);
        
        // Rough USD estimate (would use real price feed in production)
        (adjusted_amount as u64).saturating_mul(2000) // Assume $2000 per ETH-like token
    }
    
    /// Get token symbol from address
    fn get_token_symbol(&self, address: &[u8; 20]) -> &'static str {
        // Simplified - would use token registry in production
        if address[0] == 0xd && address[1] == 0xA && address[2] == 0x0b {
            "USDT"
        } else if address[0] == 0xA && address[1] == 0x0b {
            "USDC"
        } else {
            "UNK"
        }
    }
    
    /// Get token decimals
    fn get_token_decimals(&self, _address: &[u8; 20]) -> u8 {
        18 // Default assumption
    }
    
    /// Calculate confidence score for alert
    fn calculate_confidence(&self, value_usd: u64, from_exchange: bool, to_exchange: bool) -> f32 {
        let mut score = 0.5;
        
        // Higher value = higher confidence
        if value_usd > 1_000_000 {
            score += 0.3;
        } else if value_usd > 100_000 {
            score += 0.1;
        }
        
        // Exchange involvement increases confidence
        if from_exchange || to_exchange {
            score += 0.2;
        }
        
        score.min(1.0)
    }
    
    /// Get or create cluster for a wallet
    fn get_or_create_cluster(&self, address: &[u8; 20]) -> Option<u64> {
        if let Some(entry) = self.wallet_clusters.get(address) {
            Some(*entry)
        } else {
            // Create new cluster
            let cluster_id = self.next_cluster_id.fetch_add(1, Ordering::Relaxed);
            self.wallet_clusters.insert(*address, cluster_id);
            
            self.cluster_metadata.insert(cluster_id, WalletCluster {
                cluster_id,
                wallet_count: 1,
                total_value_usd: 0,
                last_activity_ns: get_timestamp_ns(),
                is_exchange: self.exchange_filter.contains(address),
                exchange_name: None,
            });
            
            Some(cluster_id)
        }
    }
    
    /// Update cluster activity timestamp
    fn update_cluster_activity(&self, address: &[u8; 20]) {
        if let Some((_, cluster_id)) = self.wallet_clusters.get_key_value(address) {
            if let Some(mut cluster) = self.cluster_metadata.get_mut(cluster_id) {
                cluster.last_activity_ns = get_timestamp_ns();
            }
        }
    }
}

/// Whale statistics snapshot
#[derive(Debug, Clone)]
pub struct WhaleStats {
    pub total_alerts: usize,
    pub large_transfers: usize,
    pub exchange_flows: usize,
    pub tracked_wallets: usize,
    pub cluster_count: usize,
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
    fn test_bloom_filter() {
        let filter = BloomFilter::new(1024, 3);
        let address = [0x11u8; 20];
        
        assert!(!filter.contains(&address));
        
        filter.insert(&address);
        assert!(filter.contains(&address));
    }
    
    #[test]
    fn test_whale_engine_creation() {
        let exchange_wallets = vec![[0x22u8; 20]];
        let engine = WhaleClusterEngine::new(exchange_wallets);
        
        assert!(engine.is_exchange_wallet(&[0x22u8; 20]));
        assert!(!engine.is_exchange_wallet(&[0x33u8; 20]));
    }
    
    #[test]
    fn test_process_transfer() {
        let engine = WhaleClusterEngine::new(vec![]);
        
        let transfer = crate::onchain::decoder::TransferEvent {
            from: [0x11u8; 20],
            to: [0x22u8; 20],
            value: 1_000_000_000_000_000_000_000u128, // Large value
            token_address: [0x33u8; 20],
        };
        
        engine.process_transfer(&transfer);
        
        let stats = engine.get_stats();
        assert!(stats.total_alerts > 0 || stats.large_transfers > 0);
    }
}
