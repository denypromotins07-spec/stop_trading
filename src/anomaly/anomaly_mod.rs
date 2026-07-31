//! Anomaly Detection Module Root
//! 
//! Feeds Rust-native anomaly scores into the pre-trade risk bus to block toxic trades.

pub mod isolation_forest;
pub mod knn;

pub use isolation_forest::{
    AnomalyResult, AnomalyType, IsolationForest, MarketAnomalyDetector, PaddedNode,
    SeverityLevel,
};
pub use knn::{
    CircularBuffer, KnnClassifier, KnnResult, OrderBookAnalyzer, OrderBookFeatures,
    RegimeLabel, RegimeShiftResult, simd_math,
};

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::sync::Arc;

/// Pre-trade risk score from anomaly detection
#[derive(Debug, Clone)]
pub struct AnomalyRiskScore {
    pub overall_score: f64,           // 0.0 - 1.0 (higher = more risky)
    pub isolation_forest_score: f64,
    pub knn_regime_score: f64,
    pub orderbook_anomaly_score: f64,
    pub should_block: bool,
    pub block_reason: Option<BlockReason>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockReason {
    FlashCrashDetected,
    SpoofingPattern,
    LiquidityShock,
    RegimeShift,
    HighVolatility,
    CompositeThreshold,
}

impl AnomalyRiskScore {
    pub fn new() -> Self {
        AnomalyRiskScore {
            overall_score: 0.0,
            isolation_forest_score: 0.0,
            knn_regime_score: 0.0,
            orderbook_anomaly_score: 0.0,
            should_block: false,
            block_reason: None,
        }
    }

    /// Calculate composite score and determine if trade should be blocked
    pub fn calculate(&mut self, if_score: f64, knn_score: f64, ob_score: f64, threshold: f64) {
        self.isolation_forest_score = if_score.clamp(0.0, 1.0);
        self.knn_regime_score = knn_score.clamp(0.0, 1.0);
        self.orderbook_anomaly_score = ob_score.clamp(0.0, 1.0);

        // Weighted composite score
        self.overall_score = 
            self.isolation_forest_score * 0.4 +
            self.knn_regime_score * 0.3 +
            self.orderbook_anomaly_score * 0.3;

        self.should_block = self.overall_score > threshold;

        if self.should_block {
            self.block_reason = Some(self.determine_block_reason());
        }
    }

    fn determine_block_reason(&self) -> BlockReason {
        // Determine primary reason for blocking
        let max_score = self.isolation_forest_score
            .max(self.knn_regime_score)
            .max(self.orderbook_anomaly_score);

        if max_score == self.isolation_forest_score && self.isolation_forest_score > 0.8 {
            BlockReason::FlashCrashDetected
        } else if max_score == self.knn_regime_score && self.knn_regime_score > 0.7 {
            BlockReason::RegimeShift
        } else if max_score == self.orderbook_anomaly_score && self.orderbook_anomaly_score > 0.7 {
            BlockReason::SpoofingPattern
        } else {
            BlockReason::CompositeThreshold
        }
    }
}

impl Default for AnomalyRiskScore {
    fn default() -> Self {
        Self::new()
    }
}

/// Unified Anomaly Detection Engine combining all detectors
pub struct AnomalyEngine<const IF_TREES: usize, const IF_NODES: usize, const KNN_SAMPLES: usize> {
    pub market_detector: MarketAnomalyDetector<IF_TREES, IF_NODES>,
    pub regime_classifier: KnnClassifier<KNN_SAMPLES, 32>,
    pub orderbook_analyzer: OrderBookAnalyzer<KNN_SAMPLES>,
    pub block_threshold: f64,
    pub enabled: AtomicBool,
    pub blocks_counter: AtomicU64,
    pub total_checks: AtomicU64,
}

impl<const IF_TREES: usize, const IF_NODES: usize, const KNN_SAMPLES: usize> AnomalyEngine<IF_TREES, IF_NODES, KNN_SAMPLES> {
    pub fn new(block_threshold: f64) -> Self {
        AnomalyEngine {
            market_detector: MarketAnomalyDetector::new(),
            regime_classifier: KnnClassifier::new(5),
            orderbook_analyzer: OrderBookAnalyzer::new(5),
            block_threshold,
            enabled: AtomicBool::new(true),
            blocks_counter: AtomicU64::new(0),
            total_checks: AtomicU64::new(0),
        }
    }

    /// Run pre-trade anomaly check
    pub fn pre_trade_check(&self, features: &[f64]) -> AnomalyRiskScore {
        self.total_checks.fetch_add(1, Ordering::Relaxed);

        if !self.enabled.load(Ordering::Relaxed) {
            return AnomalyRiskScore::new();
        }

        let mut score = AnomalyRiskScore::new();

        // Get scores from each detector
        let if_score = self.market_detector.forest.anomaly_score(features);
        
        // KNN classification confidence as inverse risk
        let knn_result = self.regime_classifier.classify(
            &features.try_into().unwrap_or([0.0; 32])
        );
        let knn_score = 1.0 - knn_result.confidence; // Low confidence = high risk

        // Order book anomaly (simplified)
        let ob_score = if features.len() >= 2 {
            (features[0] - features[1]).abs() / features[0].max(1.0)
        } else {
            0.0
        };

        score.calculate(if_score, knn_score, ob_score, self.block_threshold);

        if score.should_block {
            self.blocks_counter.fetch_add(1, Ordering::Relaxed);
        }

        score
    }

    /// Check specifically for flash crash conditions
    pub fn check_flash_crash(&self, price_change_pct: f64, volume_spike: f64) -> bool {
        // Flash crash: large price move with volume spike
        price_change_pct.abs() > 5.0 && volume_spike > 3.0
    }

    /// Check for spoofing patterns
    pub fn check_spoofing(&self, bid_ask_imbalance: f64, order_cancellation_rate: f64) -> bool {
        // Spoofing: high imbalance with high cancellation
        bid_ask_imbalance.abs() > 0.8 && order_cancellation_rate > 0.7
    }

    /// Update training data online
    pub fn update_training(&mut self, features: [f64; 32], label: RegimeLabel) {
        self.regime_classifier.add_sample(features, label);
        self.orderbook_analyzer.knn.add_sample(features, label);
    }

    /// Enable/disable anomaly detection
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Relaxed);
    }

    /// Get statistics
    pub fn get_stats(&self) -> AnomalyStats {
        AnomalyStats {
            total_checks: self.total_checks.load(Ordering::Relaxed),
            blocks: self.blocks_counter.load(Ordering::Relaxed),
            block_rate: self.calculate_block_rate(),
            enabled: self.enabled.load(Ordering::Relaxed),
        }
    }

    fn calculate_block_rate(&self) -> f64 {
        let total = self.total_checks.load(Ordering::Relaxed);
        let blocks = self.blocks_counter.load(Ordering::Relaxed);
        if total == 0 {
            return 0.0;
        }
        blocks as f64 / total as f64
    }
}

impl<const IF_TREES: usize, const IF_NODES: usize, const KNN_SAMPLES: usize> Default for AnomalyEngine<IF_TREES, IF_NODES, KNN_SAMPLES> {
    fn default() -> Self {
        Self::new(0.6)
    }
}

/// Anomaly detection statistics
#[derive(Debug, Clone)]
pub struct AnomalyStats {
    pub total_checks: u64,
    pub blocks: u64,
    pub block_rate: f64,
    pub enabled: bool,
}

/// Integration with pre-trade risk bus
pub struct PreTradeRiskBus {
    pub anomaly_engine: Arc<AnomalyEngine<10, 1000, 500>>,
    pub hard_block_enabled: AtomicBool,
    pub soft_block_enabled: AtomicBool,
    pub alert_threshold: f64,
}

impl PreTradeRiskBus {
    pub fn new(anomaly_engine: Arc<AnomalyEngine<10, 1000, 500>>) -> Self {
        PreTradeRiskBus {
            anomaly_engine,
            hard_block_enabled: AtomicBool::new(true),
            soft_block_enabled: AtomicBool::new(true),
            alert_threshold: 0.4,
        }
    }

    /// Evaluate trade through risk bus
    pub fn evaluate_trade(&self, features: &[f64]) -> RiskBusDecision {
        let score = self.anomaly_engine.pre_trade_check(features);

        if !score.should_block {
            return RiskBusDecision::Allow;
        }

        if self.hard_block_enabled.load(Ordering::Relaxed) {
            RiskBusDecision::Block(score.block_reason.unwrap_or(BlockReason::CompositeThreshold))
        } else if self.soft_block_enabled.load(Ordering::Relaxed) {
            RiskBusDecision::Warn(score)
        } else {
            RiskBusDecision::Allow
        }
    }

    /// Set alert threshold for warnings
    pub fn set_alert_threshold(&mut self, threshold: f64) {
        self.alert_threshold = threshold.clamp(0.0, 1.0);
    }

    /// Enable/disable hard blocks
    pub fn set_hard_blocks(&self, enabled: bool) {
        self.hard_block_enabled.store(enabled, Ordering::Relaxed);
    }

    /// Enable/disable soft blocks (warnings)
    pub fn set_soft_blocks(&self, enabled: bool) {
        self.soft_block_enabled.store(enabled, Ordering::Relaxed);
    }
}

/// Decision from risk bus
#[derive(Debug, Clone)]
pub enum RiskBusDecision {
    Allow,
    Block(BlockReason),
    Warn(AnomalyRiskScore),
}

impl RiskBusDecision {
    pub fn is_allowed(&self) -> bool {
        matches!(self, RiskBusDecision::Allow)
    }

    pub fn is_blocked(&self) -> bool {
        matches!(self, RiskBusDecision::Block(_))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_anomaly_risk_score_calculation() {
        let mut score = AnomalyRiskScore::new();
        score.calculate(0.8, 0.6, 0.7, 0.5);

        assert!(score.overall_score > 0.6);
        assert!(score.should_block);
        assert!(score.block_reason.is_some());
    }

    #[test]
    fn test_anomaly_engine_creation() {
        let engine: AnomalyEngine<5, 100, 50> = AnomalyEngine::new(0.6);
        assert!(engine.enabled.load(Ordering::Relaxed));
        assert_eq!(engine.block_threshold, 0.6);
    }

    #[test]
    fn test_pre_trade_check() {
        let engine: AnomalyEngine<5, 100, 50> = AnomalyEngine::new(0.5);
        
        let features = vec![0.5, 0.5, 0.5, 0.5];
        let score = engine.pre_trade_check(&features);

        assert!(score.overall_score >= 0.0);
        assert!(score.overall_score <= 1.0);
    }

    #[test]
    fn test_risk_bus_decision() {
        let allow = RiskBusDecision::Allow;
        let block = RiskBusDecision::Block(BlockReason::FlashCrashDetected);

        assert!(allow.is_allowed());
        assert!(!block.is_allowed());
        assert!(block.is_blocked());
    }

    #[test]
    fn test_flash_crash_detection() {
        let engine: AnomalyEngine<5, 100, 50> = AnomalyEngine::new(0.5);
        
        assert!(engine.check_flash_crash(10.0, 5.0));
        assert!(!engine.check_flash_crash(1.0, 1.0));
    }
}
