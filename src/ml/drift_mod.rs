//! Drift Module Root
//! 
//! Manages automated Python retraining pipeline triggers via IPC when drift thresholds are breached.
//! Coordinates drift detection with model lifecycle management.

use alloc::vec::Vec;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::ml::drift_detector::{
    MultiFeatureDriftMonitor, 
    PageHinkleyParams, 
    AdwinParams, 
    DriftResult,
    DriftNotificationIPC,
};
use crate::ml::online_stats::OnlineFeatureStats;

/// Drift configuration for triggering retraining
#[derive(Debug, Clone)]
pub struct DriftConfig {
    /// Number of features to monitor
    pub n_features: usize,
    /// Page-Hinkley threshold
    pub ph_threshold: f64,
    /// ADWIN delta parameter
    pub adwin_delta: f64,
    /// Minimum alerts before triggering retrain
    pub alert_threshold: u64,
    /// Cooldown period between retrain requests (ns)
    pub cooldown_ns: u64,
    /// Enable automatic retraining
    pub auto_retrain: bool,
}

impl Default for DriftConfig {
    fn default() -> Self {
        DriftConfig {
            n_features: 64,
            ph_threshold: 50.0,
            adwin_delta: 0.002,
            alert_threshold: 5,
            cooldown_ns: 300_000_000_000, // 5 minutes
            auto_retrain: false,
        }
    }
}

/// Drift alert message
#[derive(Debug, Clone)]
pub struct DriftAlert {
    pub timestamp_ns: u64,
    pub feature_id: u64,
    pub severity: f64,
    pub drift_type: DriftType,
    pub retrain_requested: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DriftType {
    PageHinkley,
    ADWIN,
    Both,
    Statistical,
}

/// Drift manager coordinating detection and response
pub struct DriftManager {
    config: DriftConfig,
    monitor: MultiFeatureDriftMonitor,
    stats_tracker: OnlineFeatureStats,
    last_retrain_ts: AtomicU64,
    total_alerts: AtomicU64,
    retrain_pending: AtomicBool,
    is_active: AtomicBool,
}

impl DriftManager {
    pub fn new(config: DriftConfig) -> Self {
        let ph_params = PageHinkleyParams {
            threshold: config.ph_threshold,
            ..Default::default()
        };
        
        let adwin_params = AdwinParams {
            delta: config.adwin_delta,
            ..Default::default()
        };

        DriftManager {
            config,
            monitor: MultiFeatureDriftMonitor::new(
                config.n_features,
                ph_params,
                adwin_params,
            ),
            stats_tracker: OnlineFeatureStats::new(config.n_features),
            last_retrain_ts: AtomicU64::new(0),
            total_alerts: AtomicU64::new(0),
            retrain_pending: AtomicBool::new(false),
            is_active: AtomicBool::new(true),
        }
    }

    /// Process new feature observation
    pub fn observe(&self, features: &[f64], timestamp_ns: u64) -> Option<DriftAlert> {
        if !self.is_active.load(Ordering::Acquire) {
            return None;
        }

        assert_eq!(features.len(), self.config.n_features);

        // Update online statistics
        self.stats_tracker.update(features);

        // Check for drift
        let drift_result = self.monitor.update(features, timestamp_ns);

        // Create alert if drift detected
        if drift_result.page_hinkley_drift || drift_result.adwin_drift {
            self.total_alerts.fetch_add(1, Ordering::Relaxed);

            let drift_type = match (drift_result.page_hinkley_drift, drift_result.adwin_drift) {
                (true, true) => DriftType::Both,
                (true, false) => DriftType::PageHinkley,
                (false, true) => DriftType::ADWIN,
                _ => DriftType::Statistical,
            };

            let should_retrain = self.check_retrain_trigger(timestamp_ns);

            let alert = DriftAlert {
                timestamp_ns,
                feature_id: drift_result.feature_id,
                severity: drift_result.ph_statistic,
                drift_type,
                retrain_requested: should_retrain,
            };

            if should_retrain {
                self.retrain_pending.store(true, Ordering::Release);
            }

            return Some(alert);
        }

        None
    }

    /// Check if retraining should be triggered
    fn check_retrain_trigger(&self, timestamp_ns: u64) -> bool {
        if !self.config.auto_retrain {
            return false;
        }

        // Check cooldown
        let last_ts = self.last_retrain_ts.load(Ordering::Acquire);
        if timestamp_ns.saturating_sub(last_ts) < self.config.cooldown_ns {
            return false;
        }

        // Check alert threshold
        let alerts = self.total_alerts.load(Ordering::Acquire);
        alerts >= self.config.alert_threshold
    }

    /// Get pending retrain request
    pub fn get_retrain_request(&self) -> bool {
        self.retrain_pending.load(Ordering::Acquire)
    }

    /// Acknowledge retrain request (clears pending flag)
    pub fn acknowledge_retrain(&self) {
        self.retrain_pending.store(false, Ordering::Release);
        self.last_retrain_ts.store(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64,
            Ordering::Release,
        );
    }

    /// Convert alert to IPC message
    pub fn alert_to_ipc(alert: &DriftAlert) -> DriftNotificationIPC {
        let drift_type = match alert.drift_type {
            DriftType::PageHinkley => 0,
            DriftType::ADWIN => 1,
            DriftType::Both => 2,
            DriftType::Statistical => 3,
        };

        DriftNotificationIPC {
            feature_id: alert.feature_id,
            drift_type,
            severity: alert.severity,
            timestamp_ns: alert.timestamp_ns,
            retrain_requested: alert.retrain_requested,
        }
    }

    /// Get current statistics for a feature
    pub fn get_feature_stats(&self, feature_idx: usize) -> Option<FeatureStatsSnapshot> {
        if feature_idx >= self.config.n_features {
            return None;
        }

        Some(FeatureStatsSnapshot {
            mean: self.stats_tracker.get_mean(feature_idx)?,
            std_dev: self.stats_tracker.get_std_dev(feature_idx)?,
            skewness: self.stats_tracker.get_skewness(feature_idx)?,
            excess_kurtosis: self.stats_tracker.get_excess_kurtosis(feature_idx)?,
        })
    }

    /// Get total alerts count
    pub fn total_alerts(&self) -> u64 {
        self.total_alerts.load(Ordering::Acquire)
    }

    /// Reset all drift detectors
    pub fn reset(&self) {
        self.monitor.reset_retrain_flag();
        self.total_alerts.store(0, Ordering::Relaxed);
        self.retrain_pending.store(false, Ordering::Release);
        self.stats_tracker.reset_all();
    }

    /// Activate/deactivate drift monitoring
    pub fn set_active(&self, active: bool) {
        self.is_active.store(active, Ordering::Release);
    }

    pub fn is_active(&self) -> bool {
        self.is_active.load(Ordering::Acquire)
    }
}

/// Snapshot of feature statistics
#[derive(Debug, Clone)]
pub struct FeatureStatsSnapshot {
    pub mean: f64,
    pub std_dev: f64,
    pub skewness: f64,
    pub excess_kurtosis: f64,
}

/// Retraining pipeline coordinator
pub struct RetrainingPipeline {
    drift_manager: DriftManager,
    pipeline_enabled: AtomicBool,
    pending_retrains: AtomicU64,
    completed_retrains: AtomicU64,
    failed_retrains: AtomicU64,
}

impl RetrainingPipeline {
    pub fn new(config: DriftConfig) -> Self {
        RetrainingPipeline {
            drift_manager: DriftManager::new(config),
            pipeline_enabled: AtomicBool::new(true),
            pending_retrains: AtomicU64::new(0),
            completed_retrains: AtomicU64::new(0),
            failed_retrains: AtomicU64::new(0),
        }
    }

    /// Process observation and potentially trigger retraining
    pub fn process(&self, features: &[f64], timestamp_ns: u64) -> PipelineResult {
        if !self.pipeline_enabled.load(Ordering::Acquire) {
            return PipelineResult::Disabled;
        }

        if let Some(alert) = self.drift_manager.observe(features, timestamp_ns) {
            if alert.retrain_requested {
                self.pending_retrains.fetch_add(1, Ordering::Relaxed);
                return PipelineResult::RetrainRequested(alert);
            }
            return PipelineResult::DriftDetected(alert);
        }

        PipelineResult::NoDrift
    }

    /// Mark retraining as started
    pub fn start_retrain(&self) {
        self.pending_retrains.fetch_sub(1, Ordering::Relaxed);
    }

    /// Mark retraining as completed
    pub fn complete_retrain(&self) {
        self.completed_retrains.fetch_add(1, Ordering::Relaxed);
    }

    /// Mark retraining as failed
    pub fn fail_retrain(&self) {
        self.failed_retrains.fetch_add(1, Ordering::Relaxed);
    }

    /// Get pipeline statistics
    pub fn get_stats(&self) -> PipelineStats {
        PipelineStats {
            pending: self.pending_retrains.load(Ordering::Relaxed),
            completed: self.completed_retrains.load(Ordering::Relaxed),
            failed: self.failed_retrains.load(Ordering::Relaxed),
            total_alerts: self.drift_manager.total_alerts(),
        }
    }

    /// Get reference to drift manager
    pub fn drift_manager(&self) -> &DriftManager {
        &self.drift_manager
    }

    /// Enable/disable pipeline
    pub fn set_enabled(&self, enabled: bool) {
        self.pipeline_enabled.store(enabled, Ordering::Release);
    }
}

/// Result from pipeline processing
#[derive(Debug)]
pub enum PipelineResult {
    NoDrift,
    DriftDetected(DriftAlert),
    RetrainRequested(DriftAlert),
    Disabled,
}

/// Pipeline statistics
#[derive(Debug, Clone)]
pub struct PipelineStats {
    pub pending: u64,
    pub completed: u64,
    pub failed: u64,
    pub total_alerts: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_drift_manager_basic() {
        let config = DriftConfig {
            n_features: 3,
            auto_retrain: false,
            ..Default::default()
        };

        let manager = DriftManager::new(config);
        let features = vec![0.5, 0.3, 0.7];

        // Initial observations should not trigger alerts
        for i in 0..10 {
            let result = manager.observe(&features, i * 1_000_000);
            assert!(result.is_none());
        }

        assert!(!manager.get_retrain_request());
    }

    #[test]
    fn test_retraining_pipeline() {
        let config = DriftConfig {
            n_features: 2,
            auto_retrain: false,
            ..Default::default()
        };

        let pipeline = RetrainingPipeline::new(config);
        let features = vec![1.0, 2.0];

        let result = pipeline.process(&features, 1_000_000_000);
        assert!(matches!(result, PipelineResult::NoDrift));

        let stats = pipeline.get_stats();
        assert_eq!(stats.pending, 0);
    }

    #[test]
    fn test_pipeline_enable_disable() {
        let config = DriftConfig::default();
        let pipeline = RetrainingPipeline::new(config);

        pipeline.set_enabled(false);
        let result = pipeline.process(&vec![0.0], 0);
        assert!(matches!(result, PipelineResult::Disabled));

        pipeline.set_enabled(true);
    }
}
