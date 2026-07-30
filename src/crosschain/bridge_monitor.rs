//! Cross-Chain Bridge Monitor
//! 
//! Monitors bridge finality, liquidity depth, and transaction queues for wrapped assets.
//! Detects bridge congestion or liquidity evaporation that could cause massive slippage.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::collections::HashMap;

/// Bridge status enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BridgeStatus {
    Healthy,
    Congested,
    Degraded,
    Halted,
    Unknown,
}

/// Wrapped asset information
#[derive(Debug, Clone)]
pub struct WrappedAsset {
    pub symbol: String,
    pub native_chain: String,
    pub wrapped_chain: String,
    pub wrapped_symbol: String,
    pub bridge_address: String,
}

/// Bridge liquidity snapshot
#[derive(Debug, Clone)]
pub struct BridgeLiquidity {
    pub asset: String,
    pub chain: String,
    pub available_liquidity: f64,
    pub pending_withdrawals: f64,
    pub pending_deposits: f64,
    pub utilization_rate: f64,
    pub last_update_ns: u64,
}

/// Transaction queue metrics
#[derive(Debug, Clone)]
pub struct TxQueueMetrics {
    pub pending_count: usize,
    pub avg_wait_time_ms: u64,
    pub max_wait_time_ms: u64,
    pub gas_price_gwei: f64,
    pub finality_blocks_remaining: u32,
}

/// Bridge health report
#[derive(Debug, Clone)]
pub struct BridgeHealthReport {
    pub bridge_id: String,
    pub status: BridgeStatus,
    pub liquidity: BridgeLiquidity,
    pub queue_metrics: Option<TxQueueMetrics>,
    pub health_score: f64,
    pub risk_level: RiskLevel,
    pub recommendations: Vec<String>,
    pub timestamp_ns: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

/// Circular buffer for historical data
struct CircularBuffer<T: Clone> {
    data: Vec<Option<T>>,
    head: usize,
    count: usize,
    capacity: usize,
}

impl<T: Clone> CircularBuffer<T> {
    fn new(capacity: usize) -> Self {
        Self {
            data: (0..capacity).map(|_| None).collect(),
            head: 0,
            count: 0,
            capacity,
        }
    }

    fn push(&mut self, item: T) {
        self.data[self.head] = Some(item);
        self.head = (self.head + 1) % self.capacity;
        if self.count < self.capacity {
            self.count += 1;
        }
    }

    fn iter(&self) -> impl Iterator<Item = &T> {
        self.data.iter().filter_map(|x| x.as_ref())
    }

    fn average(&self) -> Option<f64> 
    where
        T: Into<f64> + Copy,
    {
        if self.count == 0 {
            return None;
        }
        let sum: f64 = self.iter().map(|&x| x.into()).sum();
        Some(sum / self.count as f64)
    }
}

/// Bridge monitor for tracking cross-chain bridge health
pub struct BridgeMonitor {
    /// Registered bridges
    bridges: dashmap::DashMap<String, WrappedAsset>,
    /// Liquidity snapshots per bridge
    liquidity_snapshots: dashmap::DashMap<String, CircularBuffer<BridgeLiquidity>>,
    /// Queue metrics per bridge
    queue_metrics: dashmap::DashMap<String, CircularBuffer<TxQueueMetrics>>,
    /// Current status per bridge
    current_status: dashmap::DashMap<String, BridgeStatus>,
    /// Health scores
    health_scores: dashmap::DashMap<String, f64>,
    /// Alert threshold for utilization rate
    utilization_alert_threshold: f64,
    /// Alert threshold for wait time
    wait_time_alert_threshold_ms: u64,
    /// Total bridges monitored
    bridge_count: AtomicUsize,
    /// Last update timestamp
    last_update_ns: AtomicU64,
    /// Global halt flag
    global_halt: AtomicBool,
}

impl BridgeMonitor {
    pub fn new(utilization_threshold: f64, wait_time_threshold_ms: u64) -> Self {
        Self {
            bridges: dashmap::DashMap::new(),
            liquidity_snapshots: dashmap::DashMap::new(),
            queue_metrics: dashmap::DashMap::new(),
            current_status: dashmap::DashMap::new(),
            health_scores: dashmap::DashMap::new(),
            utilization_alert_threshold: utilization_threshold,
            wait_time_alert_threshold_ms: wait_time_threshold_ms,
            bridge_count: AtomicUsize::new(0),
            last_update_ns: AtomicU64::new(0),
            global_halt: AtomicBool::new(false),
        }
    }

    /// Register a new bridge to monitor
    pub fn register_bridge(&self, asset: WrappedAsset) {
        let bridge_id = format!("{}:{}->{}", asset.wrapped_symbol, asset.native_chain, asset.wrapped_chain);
        self.bridges.insert(bridge_id.clone(), asset);
        self.liquidity_snapshots.insert(bridge_id.clone(), CircularBuffer::new(60));
        self.queue_metrics.insert(bridge_id.clone(), CircularBuffer::new(60));
        self.current_status.insert(bridge_id.clone(), BridgeStatus::Unknown);
        self.health_scores.insert(bridge_id.clone(), 0.5);
        self.bridge_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Update liquidity snapshot for a bridge
    pub fn update_liquidity(&self, bridge_id: &str, liquidity: BridgeLiquidity) {
        if let Some(mut buffer) = self.liquidity_snapshots.get_mut(bridge_id) {
            buffer.value().push(liquidity.clone());
            
            // Update utilization-based status
            if liquidity.utilization_rate > self.utilization_alert_threshold {
                self.current_status.insert(bridge_id.to_string(), BridgeStatus::Degraded);
            }
        }
        
        self.last_update_ns.store(timestamp_ns(), Ordering::Relaxed);
    }

    /// Update queue metrics for a bridge
    pub fn update_queue_metrics(&self, bridge_id: &str, metrics: TxQueueMetrics) {
        if let Some(mut buffer) = self.queue_metrics.get_mut(bridge_id) {
            buffer.value().push(metrics.clone());
            
            // Check for congestion
            if metrics.avg_wait_time_ms > self.wait_time_alert_threshold_ms 
                || metrics.finality_blocks_remaining > 100 
            {
                self.current_status.insert(bridge_id.to_string(), BridgeStatus::Congested);
            }
        }
        
        self.last_update_ns.store(timestamp_ns(), Ordering::Relaxed);
    }

    /// Calculate comprehensive health score for a bridge
    pub fn calculate_health_score(&self, bridge_id: &str) -> Option<BridgeHealthReport> {
        let asset = self.bridges.get(bridge_id)?;
        let liq_buffer = self.liquidity_snapshots.get(bridge_id)?;
        let queue_buffer = self.queue_metrics.get(bridge_id);
        
        // Get latest liquidity
        let latest_liquidity = liq_buffer.value().iter().last()?.clone();
        
        // Calculate liquidity score (0-1)
        let liq_score = if latest_liquidity.available_liquidity > 1000000.0 {
            1.0
        } else if latest_liquidity.available_liquidity > 100000.0 {
            0.8
        } else if latest_liquidity.available_liquidity > 10000.0 {
            0.5
        } else {
            0.2
        };
        
        // Calculate utilization penalty
        let util_penalty = if latest_liquidity.utilization_rate > 0.9 {
            0.5
        } else if latest_liquidity.utilization_rate > 0.7 {
            0.2
        } else {
            0.0
        };
        
        // Calculate queue score
        let queue_score = if let Some(qb) = queue_buffer {
            if let Some(metrics) = qb.value().iter().last() {
                if metrics.avg_wait_time_ms > 60000 {
                    0.2
                } else if metrics.avg_wait_time_ms > 10000 {
                    0.5
                } else if metrics.avg_wait_time_ms > 1000 {
                    0.8
                } else {
                    1.0
                }
            } else {
                0.5
            }
        } else {
            0.5
        };
        
        // Combined health score
        let health_score = ((liq_score + queue_score) / 2.0 - util_penalty).max(0.0).min(1.0);
        
        // Determine risk level
        let risk_level = if health_score > 0.8 {
            RiskLevel::Low
        } else if health_score > 0.6 {
            RiskLevel::Medium
        } else if health_score > 0.3 {
            RiskLevel::High
        } else {
            RiskLevel::Critical
        };
        
        // Determine status
        let status = if self.global_halt.load(Ordering::Relaxed) {
            BridgeStatus::Halted
        } else if health_score < 0.3 {
            BridgeStatus::Halted
        } else if health_score < 0.5 {
            BridgeStatus::Degraded
        } else if queue_buffer.map_or(false, |qb| {
            qb.value().iter().last().map_or(false, |m| m.avg_wait_time_ms > self.wait_time_alert_threshold_ms)
        }) {
            BridgeStatus::Congested
        } else {
            BridgeStatus::Healthy
        };
        
        // Generate recommendations
        let mut recommendations = Vec::new();
        
        if latest_liquidity.utilization_rate > 0.8 {
            recommendations.push("High utilization - consider reducing trade size".to_string());
        }
        if let Some(qb) = queue_buffer {
            if let Some(metrics) = qb.value().iter().last() {
                if metrics.avg_wait_time_ms > 30000 {
                    recommendations.push("High queue wait times - expect delayed settlements".to_string());
                }
                if metrics.finality_blocks_remaining > 50 {
                    recommendations.push("Long finality time - increased settlement risk".to_string());
                }
            }
        }
        if latest_liquidity.available_liquidity < 50000.0 {
            recommendations.push("Low liquidity - high slippage risk".to_string());
        }
        
        let report = BridgeHealthReport {
            bridge_id: bridge_id.to_string(),
            status,
            liquidity: latest_liquidity,
            queue_metrics: queue_buffer.and_then(|qb| qb.value().iter().last().cloned()),
            health_score,
            risk_level,
            recommendations,
            timestamp_ns: timestamp_ns(),
        };
        
        self.health_scores.insert(bridge_id.to_string(), health_score);
        self.current_status.insert(bridge_id.to_string(), status);
        
        Some(report)
    }

    /// Get all bridge health reports
    pub fn get_all_reports(&self) -> Vec<BridgeHealthReport> {
        self.bridges
            .iter()
            .filter_map(|entry| self.calculate_health_score(entry.key()))
            .collect()
    }

    /// Check if any bridge is in critical state
    pub fn has_critical_bridges(&self) -> bool {
        self.current_status.iter().any(|entry| {
            *entry.value() == BridgeStatus::Halted || *entry.value() == BridgeStatus::Degraded
        })
    }

    /// Get bridges for a specific asset
    pub fn get_bridges_for_asset(&self, symbol: &str) -> Vec<String> {
        self.bridges
            .iter()
            .filter(|entry| entry.value().wrapped_symbol == symbol || entry.value().symbol == symbol)
            .map(|entry| entry.key().clone())
            .collect()
    }

    /// Trigger global halt
    pub fn trigger_halt(&self) {
        self.global_halt.store(true, Ordering::SeqCst);
    }

    /// Clear global halt
    pub fn clear_halt(&self) {
        self.global_halt.store(false, Ordering::SeqCst);
    }

    /// Check if halted
    pub fn is_halted(&self) -> bool {
        self.global_halt.load(Ordering::Relaxed)
    }

    /// Get number of monitored bridges
    pub fn bridge_count(&self) -> usize {
        self.bridge_count.load(Ordering::Relaxed)
    }

    /// Get last update timestamp
    pub fn last_update_ns(&self) -> u64 {
        self.last_update_ns.load(Ordering::Relaxed)
    }
}

/// Get current timestamp in nanoseconds
#[inline]
fn timestamp_ns() -> u64 {
    Instant::now()
        .duration_since(Instant::now() - Duration::from_secs(1))
        .as_nanos() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bridge_monitor_basic() {
        let monitor = BridgeMonitor::new(0.8, 30000);
        
        let asset = WrappedAsset {
            symbol: "BTC".to_string(),
            native_chain: "Bitcoin".to_string(),
            wrapped_chain: "Ethereum".to_string(),
            wrapped_symbol: "wBTC".to_string(),
            bridge_address: "0x1234...".to_string(),
        };
        
        monitor.register_bridge(asset);
        assert_eq!(monitor.bridge_count(), 1);
        
        let liquidity = BridgeLiquidity {
            asset: "wBTC".to_string(),
            chain: "Ethereum".to_string(),
            available_liquidity: 500000.0,
            pending_withdrawals: 10000.0,
            pending_deposits: 5000.0,
            utilization_rate: 0.3,
            last_update_ns: timestamp_ns(),
        };
        
        monitor.update_liquidity("wBTC:Bitcoin->Ethereum", liquidity);
        
        let queue = TxQueueMetrics {
            pending_count: 50,
            avg_wait_time_ms: 5000,
            max_wait_time_ms: 30000,
            gas_price_gwei: 25.0,
            finality_blocks_remaining: 12,
        };
        
        monitor.update_queue_metrics("wBTC:Bitcoin->Ethereum", queue);
        
        let report = monitor.calculate_health_score("wBTC:Bitcoin->Ethereum");
        assert!(report.is_some());
        
        let report = report.unwrap();
        assert!(report.health_score > 0.5);
        assert_eq!(report.status, BridgeStatus::Healthy);
    }

    #[test]
    fn test_critical_bridge_detection() {
        let monitor = BridgeMonitor::new(0.8, 30000);
        
        let asset = WrappedAsset {
            symbol: "USDT".to_string(),
            native_chain: "Tron".to_string(),
            wrapped_chain: "Ethereum".to_string(),
            wrapped_symbol: "wUSDT".to_string(),
            bridge_address: "0x5678...".to_string(),
        };
        
        monitor.register_bridge(asset);
        
        // Simulate critical conditions
        let liquidity = BridgeLiquidity {
            asset: "wUSDT".to_string(),
            chain: "Ethereum".to_string(),
            available_liquidity: 1000.0,
            pending_withdrawals: 50000.0,
            pending_deposits: 0.0,
            utilization_rate: 0.95,
            last_update_ns: timestamp_ns(),
        };
        
        monitor.update_liquidity("wUSDT:Tron->Ethereum", liquidity);
        
        let queue = TxQueueMetrics {
            pending_count: 1000,
            avg_wait_time_ms: 120000,
            max_wait_time_ms: 600000,
            gas_price_gwei: 200.0,
            finality_blocks_remaining: 200,
        };
        
        monitor.update_queue_metrics("wUSDT:Tron->Ethereum", queue);
        
        let report = monitor.calculate_health_score("wUSDT:Tron->Ethereum");
        assert!(report.is_some());
        assert!(report.unwrap().risk_level == RiskLevel::Critical || report.unwrap().risk_level == RiskLevel::High);
    }
}
