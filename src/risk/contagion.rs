//! Real-Time Contagion Detector
//! 
//! Monitors for sudden correlation spikes across uncorrelated assets.
//! Identifies systemic panic selling where traditional stat arb pairs break down.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::collections::HashMap;

/// Maximum number of assets to track
pub const MAX_ASSETS: usize = 100;

/// Contagion alert level
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContagionLevel {
    None = 0,
    Watch = 1,
    Warning = 2,
    Critical = 3,
}

/// Contagion detection result
#[derive(Debug, Clone, Copy)]
pub struct ContagionSignal {
    /// Alert level
    pub level: ContagionLevel,
    /// Average correlation spike (0-1)
    pub avg_correlation_spike: f64,
    /// Number of assets showing contagion
    pub affected_assets: u8,
    /// Recommended deleveraging percentage (0-100)
    pub deleverage_pct: u8,
    /// Confidence score (0-1)
    pub confidence: f64,
    /// Timestamp in nanoseconds
    pub timestamp_ns: u64,
}

impl Default for ContagionSignal {
    fn default() -> Self {
        Self {
            level: ContagionLevel::None,
            avg_correlation_spike: 0.0,
            affected_assets: 0,
            deleverage_pct: 0,
            confidence: 0.0,
            timestamp_ns: 0,
        }
    }
}

/// Asset correlation state
#[derive(Debug, Clone)]
struct AssetCorrelation {
    /// Normal/base correlation
    base_correlation: f64,
    /// Current rolling correlation
    current_correlation: f64,
    /// Correlation spike magnitude
    spike: f64,
    /// Last update timestamp
    last_update_ns: u64,
}

impl AssetCorrelation {
    fn new(base_corr: f64) -> Self {
        Self {
            base_correlation: base_corr,
            current_correlation: base_corr,
            spike: 0.0,
            last_update_ns: 0,
        }
    }

    fn update(&mut self, new_corr: f64, timestamp_ns: u64) {
        self.current_correlation = new_corr;
        self.spike = (new_corr - self.base_correlation).max(0.0);
        self.last_update_ns = timestamp_ns;
    }
}

/// Cache-line aligned contagion detector state
#[repr(align(64))]
pub struct ContagionDetector {
    /// Asset correlations (simplified - would use HashMap in production)
    correlations: [Option<AssetCorrelation>; MAX_ASSETS],
    /// Asset ID hashes
    asset_ids: [u64; MAX_ASSETS],
    /// Number of tracked assets
    asset_count: usize,
    /// Correlation lookback window (samples)
    lookback_window: usize,
    /// Spike threshold for warning
    warning_threshold: f64,
    /// Spike threshold for critical
    critical_threshold: f64,
    /// Systemic event flag
    systemic_event: AtomicBool,
    /// Alerts triggered count
    alerts_triggered: AtomicU64,
    _pad: [u8; 32],
}

unsafe impl Send for ContagionDetector {}
unsafe impl Sync for ContagionDetector {}

impl ContagionDetector {
    /// Create new contagion detector
    pub fn new(warning_threshold: f64, critical_threshold: f64) -> Self {
        Self {
            correlations: std::array::from_fn(|_| None),
            asset_ids: [0; MAX_ASSETS],
            asset_count: 0,
            lookback_window: 60, // 60 samples
            warning_threshold,
            critical_threshold,
            systemic_event: AtomicBool::new(false),
            alerts_triggered: AtomicU64::new(0),
            _pad: [0; 32],
        }
    }

    /// Register asset pair for correlation tracking
    pub fn register_asset(&mut self, asset_id: u64, base_correlation: f64) -> bool {
        if self.asset_count >= MAX_ASSETS {
            return false;
        }

        self.asset_ids[self.asset_count] = asset_id;
        self.correlations[self.asset_count] = Some(AssetCorrelation::new(base_correlation));
        self.asset_count += 1;

        true
    }

    /// Update correlation for an asset
    pub fn update_correlation(&mut self, asset_id: u64, correlation: f64, timestamp_ns: u64) {
        for i in 0..self.asset_count {
            if self.asset_ids[i] == asset_id {
                if let Some(ref mut corr) = self.correlations[i] {
                    corr.update(correlation, timestamp_ns);
                }
                return;
            }
        }
    }

    /// Detect contagion across all tracked assets
    pub fn detect(&self, timestamp_ns: u64) -> ContagionSignal {
        let mut signal = ContagionSignal::default();
        signal.timestamp_ns = timestamp_ns;

        if self.asset_count == 0 {
            return signal;
        }

        let mut total_spike = 0.0f64;
        let mut affected = 0u8;
        let mut max_spike = 0.0f64;

        for i in 0..self.asset_count {
            if let Some(ref corr) = self.correlations[i] {
                if corr.spike > 0.1 {
                    total_spike += corr.spike;
                    affected += 1;
                    max_spike = max_spike.max(corr.spike);
                }
            }
        }

        if affected == 0 {
            return signal;
        }

        let avg_spike = total_spike / affected as f64;
        signal.avg_correlation_spike = avg_spike;
        signal.affected_assets = affected;

        // Determine alert level
        let pct_affected = affected as f64 / self.asset_count as f64;

        signal.level = if avg_spike >= self.critical_threshold || pct_affected > 0.7 {
            ContagionLevel::Critical
        } else if avg_spike >= self.warning_threshold || pct_affected > 0.4 {
            ContagionLevel::Warning
        } else if pct_affected > 0.2 {
            ContagionLevel::Watch
        } else {
            ContagionLevel::None
        };

        // Calculate recommended deleveraging
        signal.deleverage_pct = match signal.level {
            ContagionLevel::None => 0,
            ContagionLevel::Watch => 10,
            ContagionLevel::Warning => 30,
            ContagionLevel::Critical => 50,
        };

        // Adjust based on severity
        signal.deleverage_pct = ((signal.deleverage_pct as f64 * (1.0 + avg_spike)) as u8).min(80);

        // Confidence based on number of affected assets and spike magnitude
        signal.confidence = ((affected as f64 / self.asset_count as f64) * 0.5 + 
                            (avg_spike.min(1.0) * 0.5)).min(1.0);

        // Set systemic event flag for critical
        if signal.level == ContagionLevel::Critical {
            self.systemic_event.store(true, Ordering::Release);
            self.alerts_triggered.fetch_add(1, Ordering::Relaxed);
        }

        signal
    }

    /// Check if systemic event is active
    #[inline]
    pub fn is_systemic_event(&self) -> bool {
        self.systemic_event.load(Ordering::Acquire)
    }

    /// Clear systemic event flag
    #[inline]
    pub fn clear_systemic_event(&self) {
        self.systemic_event.store(false, Ordering::Release);
    }

    /// Get alerts triggered count
    #[inline]
    pub fn alerts_triggered(&self) -> u64 {
        self.alerts_triggered.load(Ordering::Relaxed)
    }

    /// Set lookback window
    #[inline]
    pub fn set_lookback_window(&mut self, window: usize) {
        self.lookback_window = window;
    }

    /// Reset detector state
    pub fn reset(&mut self) {
        for i in 0..self.asset_count {
            if let Some(ref mut corr) = self.correlations[i] {
                corr.current_correlation = corr.base_correlation;
                corr.spike = 0.0;
            }
        }
        self.clear_systemic_event();
    }
}

/// Builder for contagion detector
pub struct ContagionDetectorBuilder {
    warning_threshold: f64,
    critical_threshold: f64,
}

impl ContagionDetectorBuilder {
    pub fn new() -> Self {
        Self {
            warning_threshold: 0.3,
            critical_threshold: 0.5,
        }
    }

    pub fn warning_threshold(mut self, threshold: f64) -> Self {
        self.warning_threshold = threshold;
        self
    }

    pub fn critical_threshold(mut self, threshold: f64) -> Self {
        self.critical_threshold = threshold;
        self
    }

    pub fn build(self) -> ContagionDetector {
        ContagionDetector::new(self.warning_threshold, self.critical_threshold)
    }
}

impl Default for ContagionDetectorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contagion_detection() {
        let mut detector = ContagionDetectorBuilder::new()
            .warning_threshold(0.2)
            .critical_threshold(0.4)
            .build();

        // Register some assets with low base correlation
        detector.register_asset(1, 0.1);
        detector.register_asset(2, 0.15);
        detector.register_asset(3, 0.05);
        detector.register_asset(4, 0.2);

        // Update with high correlations (contagion scenario)
        detector.update_correlation(1, 0.8, 1000000);
        detector.update_correlation(2, 0.85, 1000000);
        detector.update_correlation(3, 0.75, 1000000);
        detector.update_correlation(4, 0.9, 1000000);

        let signal = detector.detect(2000000);

        assert!(signal.level == ContagionLevel::Critical);
        assert!(signal.affected_assets >= 4);
        assert!(signal.deleverage_pct >= 50);
        assert!(detector.is_systemic_event());
    }

    #[test]
    fn test_no_contagion() {
        let mut detector = ContagionDetectorBuilder::new().build();

        detector.register_asset(1, 0.5);
        detector.register_asset(2, 0.4);

        // Keep correlations near normal
        detector.update_correlation(1, 0.55, 1000000);
        detector.update_correlation(2, 0.45, 1000000);

        let signal = detector.detect(2000000);

        assert_eq!(signal.level, ContagionLevel::None);
        assert_eq!(signal.affected_assets, 0);
    }

    #[test]
    fn test_reset() {
        let mut detector = ContagionDetectorBuilder::new().build();
        detector.register_asset(1, 0.1);
        detector.update_correlation(1, 0.9, 1000000);

        detector.detect(2000000);
        assert!(detector.is_systemic_event());

        detector.clear_systemic_event();
        assert!(!detector.is_systemic_event());
    }
}
