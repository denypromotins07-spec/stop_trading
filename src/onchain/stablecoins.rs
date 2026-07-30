//! Stablecoin Flow Monitor
//! 
//! Tracks USDT/USDC minting, burning, and cross-chain bridge flows in real-time.
//! Correlates stablecoin supply expansion with market regimes.
//! Optimized for minimal memory footprint using ring buffers.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use crossbeam_channel::{bounded, Sender, Receiver};

/// Ring buffer size for flow history (keeps memory bounded)
const FLOW_HISTORY_SIZE: usize = 1024;

/// Channel capacity for stablecoin events
const EVENT_CHANNEL_CAPACITY: usize = 500;

/// Known stablecoin addresses (EVM format)
const USDT_ADDRESS: [u8; 20] = [
    0xd, 0xA, 0x0b, 0x87, 0x9D, 0x3c, 0x4F, 0x4D, 0x8d, 0x6C, 0x8A, 0x5e, 0x1F, 0x8e, 0x5e, 0x1F, 0x8e, 0x5e, 0x1F, 0x8e,
];

const USDC_ADDRESS: [u8; 20] = [
    0xA, 0x0b, 0x87, 0x9D, 0x3c, 0x4F, 0x4D, 0x8d, 0x6C, 0x8A, 0x5e, 0x1F, 0x8e, 0x5e, 0x1F, 0x8e, 0x5e, 0x1F, 0x8e, 0x00,
];

/// Stablecoin event types
#[derive(Debug, Clone)]
pub enum StablecoinEvent {
    Mint {
        token: StablecoinType,
        amount: u128,
        to: [u8; 20],
        timestamp_ns: u64,
    },
    Burn {
        token: StablecoinType,
        amount: u128,
        from: [u8; 20],
        timestamp_ns: u64,
    },
    BridgeInflow {
        token: StablecoinType,
        amount: u128,
        source_chain: ChainId,
        destination_chain: ChainId,
        timestamp_ns: u64,
    },
    BridgeOutflow {
        token: StablecoinType,
        amount: u128,
        source_chain: ChainId,
        destination_chain: ChainId,
        timestamp_ns: u64,
    },
}

/// Stablecoin types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StablecoinType {
    USDT,
    USDC,
    DAI,
    Other,
}

/// Chain identifiers
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ChainId {
    Ethereum,
    Solana,
    Arbitrum,
    Optimism,
    Polygon,
    Avalanche,
    BSC,
    Unknown,
}

/// Flow metrics snapshot
#[derive(Debug, Clone)]
pub struct FlowMetrics {
    pub net_flow_1h: i128,
    pub net_flow_24h: i128,
    pub mint_volume_24h: u128,
    pub burn_volume_24h: u128,
    pub bridge_inflow_24h: u128,
    pub bridge_outflow_24h: u128,
    pub supply_change_pct: f64,
}

/// Market regime signal based on stablecoin flows
#[derive(Debug, Clone, PartialEq)]
pub enum MarketRegime {
    BullishExpansion,   // Net positive stablecoin minting
    BearishContraction, // Net negative stablecoin burning
    Neutral,
    HighVolatility,     // Large bridge movements
}

/// Ring buffer for flow history (fixed size, no allocations)
struct FlowRingBuffer {
    data: Vec<i128>,
    head: usize,
    count: usize,
    capacity: usize,
}

impl FlowRingBuffer {
    fn new(capacity: usize) -> Self {
        Self {
            data: vec![0; capacity],
            head: 0,
            count: 0,
            capacity,
        }
    }
    
    fn push(&mut self, value: i128) {
        self.data[self.head] = value;
        self.head = (self.head + 1) % self.capacity;
        if self.count < self.capacity {
            self.count += 1;
        }
    }
    
    fn sum(&self) -> i128 {
        self.data.iter().take(self.count).sum()
    }
    
    fn clear(&mut self) {
        self.head = 0;
        self.count = 0;
        for v in &mut self.data {
            *v = 0;
        }
    }
}

/// Stablecoin flow monitor
pub struct StablecoinFlowMonitor {
    /// Event channel
    event_tx: Sender<StablecoinEvent>,
    event_rx: Receiver<StablecoinEvent>,
    
    /// Rolling flow buffers (net flows per time bucket)
    flow_buffer_1h: Arc<std::sync::Mutex<FlowRingBuffer>>,
    flow_buffer_24h: Arc<std::sync::Mutex<FlowRingBuffer>>,
    
    /// Cumulative counters
    total_mints: AtomicU128,
    total_burns: AtomicU128,
    total_bridge_in: AtomicU128,
    total_bridge_out: AtomicU128,
    
    /// Current supply estimates
    usdt_supply: AtomicU128,
    usdc_supply: AtomicU128,
    
    /// Event counter
    event_count: AtomicUsize,
}

impl StablecoinFlowMonitor {
    /// Create a new stablecoin monitor
    pub fn new() -> Self {
        let (event_tx, event_rx) = bounded(EVENT_CHANNEL_CAPACITY);
        
        Self {
            event_tx,
            event_rx,
            flow_buffer_1h: Arc::new(std::sync::Mutex::new(FlowRingBuffer::new(FLOW_HISTORY_SIZE))),
            flow_buffer_24h: Arc::new(std::sync::Mutex::new(FlowRingBuffer::new(FLOW_HISTORY_SIZE * 24))),
            total_mints: AtomicU128::new(0),
            total_burns: AtomicU128::new(0),
            total_bridge_in: AtomicU128::new(0),
            total_bridge_out: AtomicU128::new(0),
            usdt_supply: AtomicU128::new(40_000_000_000_000_000), // ~40B USDT initial
            usdc_supply: AtomicU128::new(30_000_000_000_000_000), // ~30B USDC initial
            event_count: AtomicUsize::new(0),
        }
    }
    
    /// Process a transfer event (check for mint/burn patterns)
    pub fn process_transfer(&self, transfer: &crate::onchain::decoder::TransferEvent) {
        let token_type = self.identify_token(&transfer.token_address);
        
        // Check for mint pattern (from zero address)
        if transfer.from == [0u8; 20] {
            self.total_mints.fetch_add(transfer.value, Ordering::Relaxed);
            
            // Update supply estimate
            match token_type {
                StablecoinType::USDT => {
                    self.usdt_supply.fetch_add(transfer.value, Ordering::Relaxed);
                }
                StablecoinType::USDC => {
                    self.usdc_supply.fetch_add(transfer.value, Ordering::Relaxed);
                }
                _ => {}
            }
            
            let _ = self.event_tx.try_send(StablecoinEvent::Mint {
                token: token_type,
                amount: transfer.value,
                to: transfer.to,
                timestamp_ns: get_timestamp_ns(),
            });
            
            self.record_flow(transfer.value as i128);
        }
        // Check for burn pattern (to zero address)
        else if transfer.to == [0u8; 20] {
            self.total_burns.fetch_add(transfer.value, Ordering::Relaxed);
            
            // Update supply estimate
            match token_type {
                StablecoinType::USDT => {
                    self.usdt_supply.fetch_sub(transfer.value, Ordering::Relaxed);
                }
                StablecoinType::USDC => {
                    self.usdc_supply.fetch_sub(transfer.value, Ordering::Relaxed);
                }
                _ => {}
            }
            
            let _ = self.event_tx.try_send(StablecoinEvent::Burn {
                token: token_type,
                amount: transfer.value,
                from: transfer.from,
                timestamp_ns: get_timestamp_ns(),
            });
            
            self.record_flow(-(transfer.value as i128));
        }
        
        self.event_count.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Process a mint event directly
    pub fn process_mint(&self, mint: &crate::onchain::decoder::MintEvent) {
        let token_type = self.identify_token(&mint.token_address);
        
        self.total_mints.fetch_add(mint.value, Ordering::Relaxed);
        
        match token_type {
            StablecoinType::USDT => {
                self.usdt_supply.fetch_add(mint.value, Ordering::Relaxed);
            }
            StablecoinType::USDC => {
                self.usdc_supply.fetch_add(mint.value, Ordering::Relaxed);
            }
            _ => {}
        }
        
        let _ = self.event_tx.try_send(StablecoinEvent::Mint {
            token: token_type,
            amount: mint.value,
            to: mint.to,
            timestamp_ns: get_timestamp_ns(),
        });
        
        self.record_flow(mint.value as i128);
        self.event_count.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Record a bridge inflow
    pub fn record_bridge_inflow(
        &self,
        token: StablecoinType,
        amount: u128,
        source: ChainId,
        dest: ChainId,
    ) {
        self.total_bridge_in.fetch_add(amount, Ordering::Relaxed);
        
        let _ = self.event_tx.try_send(StablecoinEvent::BridgeInflow {
            token,
            amount,
            source_chain: source,
            destination_chain: dest,
            timestamp_ns: get_timestamp_ns(),
        });
        
        self.record_flow(amount as i128);
    }
    
    /// Record a bridge outflow
    pub fn record_bridge_outflow(
        &self,
        token: StablecoinType,
        amount: u128,
        source: ChainId,
        dest: ChainId,
    ) {
        self.total_bridge_out.fetch_add(amount, Ordering::Relaxed);
        
        let _ = self.event_tx.try_send(StablecoinEvent::BridgeOutflow {
            token,
            amount,
            source_chain: source,
            destination_chain: dest,
            timestamp_ns: get_timestamp_ns(),
        });
        
        self.record_flow(-(amount as i128));
    }
    
    /// Get current flow metrics
    pub fn get_metrics(&self) -> FlowMetrics {
        let buffer_1h = self.flow_buffer_1h.lock().unwrap();
        let buffer_24h = self.flow_buffer_24h.lock().unwrap();
        
        let net_flow_1h = buffer_1h.sum();
        let net_flow_24h = buffer_24h.sum();
        
        let total_supply = self.usdt_supply.load(Ordering::Relaxed) + self.usdc_supply.load(Ordering::Relaxed);
        let supply_change_pct = if total_supply > 0 {
            (net_flow_24h as f64) / (total_supply as f64) * 100.0
        } else {
            0.0
        };
        
        FlowMetrics {
            net_flow_1h,
            net_flow_24h,
            mint_volume_24h: self.total_mints.load(Ordering::Relaxed),
            burn_volume_24h: self.total_burns.load(Ordering::Relaxed),
            bridge_inflow_24h: self.total_bridge_in.load(Ordering::Relaxed),
            bridge_outflow_24h: self.total_bridge_out.load(Ordering::Relaxed),
            supply_change_pct,
        }
    }
    
    /// Determine market regime from flow patterns
    pub fn get_market_regime(&self) -> MarketRegime {
        let metrics = self.get_metrics();
        
        // Check for high volatility (large bridge movements)
        let bridge_total = metrics.bridge_inflow_24h + metrics.bridge_outflow_24h;
        if bridge_total > 1_000_000_000_000_000u128 { // > 1B
            return MarketRegime::HighVolatility;
        }
        
        // Check net flow direction
        if metrics.net_flow_24h > 500_000_000_000_000i128 { // > 500M net inflow
            MarketRegime::BullishExpansion
        } else if metrics.net_flow_24h < -500_000_000_000_000i128 { // > 500M net outflow
            MarketRegime::BearishContraction
        } else {
            MarketRegime::Neutral
        }
    }
    
    /// Get receiver for stablecoin events
    pub fn event_receiver(&self) -> Receiver<StablecoinEvent> {
        self.event_rx.clone()
    }
    
    /// Get current USDT supply estimate
    pub fn get_usdt_supply(&self) -> u128 {
        self.usdt_supply.load(Ordering::Relaxed)
    }
    
    /// Get current USDC supply estimate
    pub fn get_usdc_supply(&self) -> u128 {
        self.usdc_supply.load(Ordering::Relaxed)
    }
    
    /// Identify token type from address
    fn identify_token(&self, address: &[u8; 20]) -> StablecoinType {
        if address[..6] == USDT_ADDRESS[..6] {
            StablecoinType::USDT
        } else if address[..4] == USDC_ADDRESS[..4] {
            StablecoinType::USDC
        } else {
            StablecoinType::Other
        }
    }
    
    /// Record flow in rolling buffers
    fn record_flow(&self, value: i128) {
        if let Ok(mut buffer) = self.flow_buffer_1h.lock() {
            buffer.push(value);
        }
        if let Ok(mut buffer) = self.flow_buffer_24h.lock() {
            buffer.push(value);
        }
    }
}

impl Default for StablecoinFlowMonitor {
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
    fn test_monitor_creation() {
        let monitor = StablecoinFlowMonitor::new();
        assert!(monitor.get_usdt_supply() > 0);
        assert!(monitor.get_usdc_supply() > 0);
    }
    
    #[test]
    fn test_process_mint() {
        let monitor = StablecoinFlowMonitor::new();
        
        let mint = crate::onchain::decoder::MintEvent {
            to: [0x11u8; 20],
            value: 1_000_000_000_000u128,
            token_address: USDT_ADDRESS,
        };
        
        monitor.process_mint(&mint);
        
        let metrics = monitor.get_metrics();
        assert!(metrics.mint_volume_24h > 0);
    }
    
    #[test]
    fn test_ring_buffer() {
        let mut buffer = FlowRingBuffer::new(10);
        
        for i in 0..15 {
            buffer.push(i as i128);
        }
        
        assert_eq!(buffer.count, 10);
        assert!(buffer.sum() > 0);
    }
    
    #[test]
    fn test_market_regime() {
        let monitor = StablecoinFlowMonitor::new();
        let regime = monitor.get_market_regime();
        assert_eq!(regime, MarketRegime::Neutral);
    }
}
