//! Cross-Chain Module Root
//! 
//! Adjusts asset correlation weights dynamically based on real-time bridge health.

pub mod bridge_monitor;
pub mod depeg_guard;

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use crate::crosschain::bridge_monitor::{BridgeMonitor, BridgeHealthReport, RiskLevel, BridgeStatus};
use crate::crosschain::depeg_guard::{DepegGuard, DepegAlert, DepegSeverity};

/// Cross-chain module configuration
#[derive(Debug, Clone)]
pub struct CrossChainConfig {
    /// Z-score threshold for depeg detection
    pub z_score_threshold: f64,
    /// Emergency deviation percentage
    pub emergency_deviation_pct: f64,
    /// Bridge utilization alert threshold
    pub bridge_utilization_threshold: f64,
    /// Bridge wait time alert threshold (ms)
    pub bridge_wait_time_threshold_ms: u64,
    /// Correlation update interval (ms)
    pub correlation_update_interval_ms: u64,
}

impl Default for CrossChainConfig {
    fn default() -> Self {
        Self {
            z_score_threshold: 3.0,
            emergency_deviation_pct: 5.0,
            bridge_utilization_threshold: 0.8,
            bridge_wait_time_threshold_ms: 30000,
            correlation_update_interval_ms: 1000,
        }
    }
}

/// Asset correlation with risk adjustment
#[derive(Debug, Clone)]
pub struct AssetCorrelation {
    pub asset_a: String,
    pub asset_b: String,
    pub base_correlation: f64,
    pub adjusted_correlation: f64,
    pub risk_multiplier: f64,
    pub bridge_health_factor: f64,
    pub last_update_ns: u64,
}

/// Cross-chain risk assessment
#[derive(Debug, Clone)]
pub struct CrossChainRisk {
    pub overall_risk_score: f64,
    pub bridge_risks: Vec<BridgeHealthReport>,
    pub depeg_alerts: Vec<DepegAlert>,
    pub correlated_asset_risks: Vec<AssetCorrelation>,
    pub recommended_leverage: f64,
    pub halt_recommended: bool,
    pub timestamp_ns: u64,
}

/// Cross-chain module handle
pub struct CrossChainModule {
    bridge_monitor: Arc<BridgeMonitor>,
    depeg_guard: Arc<DepegGuard>,
    config: CrossChainConfig,
    /// Asset correlations
    correlations: dashmap::DashMap<(String, String), AssetCorrelation>,
    /// Overall risk score
    risk_score: AtomicU64, // Stored as fixed point (score * 1000)
    /// Last correlation update
    last_correlation_update: AtomicU64,
    /// Global halt flag
    global_halt: AtomicBool,
}

impl CrossChainModule {
    pub fn new(config: CrossChainConfig) -> Self {
        let bridge_monitor = Arc::new(BridgeMonitor::new(
            config.bridge_utilization_threshold,
            config.bridge_wait_time_threshold_ms,
        ));
        
        let depeg_guard = Arc::new(DepegGuard::new(
            config.z_score_threshold,
            config.emergency_deviation_pct,
        ));

        Self {
            bridge_monitor,
            depeg_guard,
            config,
            correlations: dashmap::DashMap::new(),
            risk_score: AtomicU64::new(0),
            last_correlation_update: AtomicU64::new(0),
            global_halt: AtomicBool::new(false),
        }
    }

    /// Get reference to bridge monitor
    pub fn bridge_monitor(&self) -> &Arc<BridgeMonitor> {
        &self.bridge_monitor
    }

    /// Get reference to depeg guard
    pub fn depeg_guard(&self) -> &Arc<DepegGuard> {
        &self.depeg_guard
    }

    /// Register or update asset correlation
    pub fn update_correlation(&self, correlation: AssetCorrelation) {
        let key = (correlation.asset_a.clone(), correlation.asset_b.clone());
        self.correlations.insert(key, correlation);
        self.last_correlation_update.store(timestamp_ns(), Ordering::Relaxed);
    }

    /// Calculate adjusted correlation based on bridge health
    pub fn calculate_adjusted_correlation(&self, asset_a: &str, asset_b: &str, base_corr: f64) -> f64 {
        let mut risk_multiplier = 1.0;
        
        // Check bridge health for both assets
        let bridges_a = self.bridge_monitor.get_bridges_for_asset(asset_a);
        let bridges_b = self.bridge_monitor.get_bridges_for_asset(asset_b);
        
        for bridge_id in &bridges_a {
            if let Some(report) = self.bridge_monitor.calculate_health_score(bridge_id) {
                match report.risk_level {
                    RiskLevel::Low => {}
                    RiskLevel::Medium => risk_multiplier *= 0.9,
                    RiskLevel::High => risk_multiplier *= 0.7,
                    RiskLevel::Critical => risk_multiplier *= 0.5,
                }
            }
        }
        
        for bridge_id in &bridges_b {
            if let Some(report) = self.bridge_monitor.calculate_health_score(bridge_id) {
                match report.risk_level {
                    RiskLevel::Low => {}
                    RiskLevel::Medium => risk_multiplier *= 0.9,
                    RiskLevel::High => risk_multiplier *= 0.7,
                    RiskLevel::Critical => risk_multiplier *= 0.5,
                }
            }
        }
        
        // Apply depeg status
        if self.depeg_guard.is_halted() {
            risk_multiplier *= 0.5;
        }
        
        (base_corr * risk_multiplier).max(-1.0).min(1.0)
    }

    /// Get comprehensive cross-chain risk assessment
    pub fn get_risk_assessment(&self) -> CrossChainRisk {
        let bridge_risks = self.bridge_monitor.get_all_reports();
        let depeg_alerts: Vec<DepegAlert> = Vec::new(); // Would need method to get all alerts
        
        // Calculate overall risk score from bridge risks
        let mut bridge_risk_sum = 0.0;
        let mut critical_count = 0;
        
        for report in &bridge_risks {
            match report.risk_level {
                RiskLevel::Low => bridge_risk_sum += 0.1,
                RiskLevel::Medium => bridge_risk_sum += 0.3,
                RiskLevel::High => {
                    bridge_risk_sum += 0.6;
                    critical_count += 1;
                }
                RiskLevel::Critical => {
                    bridge_risk_sum += 1.0;
                    critical_count += 1;
                }
            }
        }
        
        let avg_bridge_risk = if bridge_risks.is_empty() {
            0.0
        } else {
            bridge_risk_sum / bridge_risks.len() as f64
        };
        
        // Add depeg risk
        let depeg_risk = if self.depeg_guard.is_halted() { 1.0 } else { 0.0 };
        
        let overall_risk = ((avg_bridge_risk + depeg_risk) / 2.0).min(1.0);
        
        // Calculate recommended leverage
        let recommended_leverage = if overall_risk > 0.8 {
            0.0 // No leverage
        } else if overall_risk > 0.6 {
            1.0 // 1x only
        } else if overall_risk > 0.4 {
            2.0
        } else if overall_risk > 0.2 {
            5.0
        } else {
            10.0
        };
        
        let halt_recommended = overall_risk > 0.7 || critical_count >= 2 || self.depeg_guard.is_halted();
        
        if halt_recommended && !self.global_halt.load(Ordering::Relaxed) {
            self.trigger_halt();
        }
        
        self.risk_score.store((overall_risk * 1000.0) as u64, Ordering::Relaxed);
        
        CrossChainRisk {
            overall_risk_score: overall_risk,
            bridge_risks,
            depeg_alerts,
            correlated_asset_risks: self.correlations.iter().map(|e| e.value().clone()).collect(),
            recommended_leverage,
            halt_recommended,
            timestamp_ns: timestamp_ns(),
        }
    }

    /// Trigger global halt
    pub fn trigger_halt(&self) {
        self.global_halt.store(true, Ordering::SeqCst);
        self.bridge_monitor.trigger_halt();
        self.depeg_guard.trigger_halt();
    }

    /// Clear global halt
    pub fn clear_halt(&self) {
        self.global_halt.store(false, Ordering::SeqCst);
        self.bridge_monitor.clear_halt();
        self.depeg_guard.clear_halt();
    }

    /// Check if halted
    pub fn is_halted(&self) -> bool {
        self.global_halt.load(Ordering::Relaxed)
            || self.bridge_monitor.is_halted()
            || self.depeg_guard.is_halted()
    }

    /// Get current risk score (0-1000)
    pub fn risk_score(&self) -> u64 {
        self.risk_score.load(Ordering::Relaxed)
    }

    /// Periodic maintenance
    pub fn maintenance(&self) {
        let now = timestamp_ns();
        let last_update = self.last_correlation_update.load(Ordering::Relaxed);
        
        if now - last_update > (self.config.correlation_update_interval_ms as u64) * 1_000_000 {
            // Update all correlations based on current bridge health
            let keys: Vec<_> = self.correlations.iter()
                .map(|e| (e.key().0.clone(), e.key().1.clone(), e.value().base_correlation))
                .collect();
            
            for (asset_a, asset_b, base_corr) in keys {
                let adjusted = self.calculate_adjusted_correlation(&asset_a, &asset_b, base_corr);
                
                if let Some(mut entry) = self.correlations.get_mut(&(asset_a.clone(), asset_b.clone())) {
                    entry.value().adjusted_correlation = adjusted;
                    entry.value().last_update_ns = now;
                }
            }
            
            self.last_correlation_update.store(now, Ordering::Relaxed);
        }
    }

    /// Clear all data
    pub fn clear(&self) {
        self.correlations.clear();
        self.risk_score.store(0, Ordering::Relaxed);
        self.global_halt.store(false, Ordering::SeqCst);
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
    use crate::crosschain::bridge_monitor::{WrappedAsset, BridgeLiquidity, TxQueueMetrics};

    #[test]
    fn test_cross_chain_module_basic() {
        let config = CrossChainConfig::default();
        let module = CrossChainModule::new(config);

        assert!(!module.is_halted());
        assert_eq!(module.risk_score(), 0);

        // Register a bridge
        let asset = WrappedAsset {
            symbol: "USDC".to_string(),
            native_chain: "Ethereum".to_string(),
            wrapped_chain: "Arbitrum".to_string(),
            wrapped_symbol: "USDC.e".to_string(),
            bridge_address: "0xabcd...".to_string(),
        };

        module.bridge_monitor().register_bridge(asset);

        // Update liquidity
        let liquidity = BridgeLiquidity {
            asset: "USDC.e".to_string(),
            chain: "Arbitrum".to_string(),
            available_liquidity: 1000000.0,
            pending_withdrawals: 10000.0,
            pending_deposits: 5000.0,
            utilization_rate: 0.3,
            last_update_ns: timestamp_ns(),
        };

        module.bridge_monitor().update_liquidity("USDC.e:Ethereum->Arbitrum", liquidity);

        // Get risk assessment
        let risk = module.get_risk_assessment();
        assert!(risk.overall_risk_score < 0.5); // Should be low risk
        assert!(!risk.halt_recommended);
    }

    #[test]
    fn test_correlation_adjustment() {
        let config = CrossChainConfig::default();
        let module = CrossChainModule::new(config);

        let base_corr = 0.85;
        let adjusted = module.calculate_adjusted_correlation("BTC", "ETH", base_corr);
        
        // With no bridge issues, should be close to base
        assert!((adjusted - base_corr).abs() < 0.1);
    }
}
