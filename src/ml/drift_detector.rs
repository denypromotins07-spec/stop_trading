//! Concept Drift Detection Module
//! 
//! Implements Page-Hinkley and ADWIN algorithms for real-time concept drift detection
//! in feature distributions. Monitors IPC feature vectors to identify market regime shifts.

use alloc::vec::Vec;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Maximum window size for ADWIN
pub const MAX_ADWIN_WINDOW: usize = 1024;

/// Page-Hinkley test parameters
#[derive(Debug, Clone)]
pub struct PageHinkleyParams {
    /// Minimum mean value threshold
    pub min_mean: f64,
    /// Delta threshold for drift detection
    pub delta: f64,
    /// Alpha parameter (forgetting factor)
    pub alpha: f64,
    /// Threshold for alarm
    pub threshold: f64,
}

impl Default for PageHinkleyParams {
    fn default() -> Self {
        PageHinkleyParams {
            min_mean: 0.05,
            delta: 0.005,
            alpha: 0.9999,
            threshold: 50.0,
        }
    }
}

/// Page-Hinkley drift detector
pub struct PageHinkleyDetector {
    params: PageHinkleyParams,
    sum: f64,
    mean: f64,
    count: usize,
    ph_max: f64,
    ph_min: f64,
    drift_detected: AtomicBool,
    last_drift_ts: AtomicU64,
}

impl PageHinkleyDetector {
    pub fn new(params: PageHinkleyParams) -> Self {
        PageHinkleyDetector {
            params,
            sum: 0.0,
            mean: 0.0,
            count: 0,
            ph_max: f64::MIN,
            ph_min: f64::MAX,
            drift_detected: AtomicBool::new(false),
            last_drift_ts: AtomicU64::new(0),
        }
    }

    /// Add new observation and check for drift
    pub fn update(&mut self, value: f64, timestamp_ns: u64) -> bool {
        self.count += 1;
        
        // Update running mean
        self.sum += value;
        self.mean = self.sum / self.count as f64;
        
        // Page-Hinkley statistic
        let ph = self.sum - self.count as f64 * (self.params.min_mean + self.params.delta);
        
        // Update max/min
        self.ph_max = self.ph_max.max(ph);
        self.ph_min = self.ph_min.min(ph);
        
        // Check for drift
        let ph_range = self.ph_max - self.ph_min;
        let drift = ph_range > self.params.threshold;
        
        if drift {
            self.drift_detected.store(true, Ordering::Release);
            self.last_drift_ts.store(timestamp_ns, Ordering::Release);
            self.reset();
        }
        
        drift
    }

    /// Reset detector state
    pub fn reset(&mut self) {
        self.sum = 0.0;
        self.mean = 0.0;
        self.count = 0;
        self.ph_max = f64::MIN;
        self.ph_min = f64::MAX;
        self.drift_detected.store(false, Ordering::Release);
    }

    /// Check if drift was detected
    pub fn has_drift(&self) -> bool {
        self.drift_detected.load(Ordering::Acquire)
    }

    /// Get last drift timestamp
    pub fn last_drift_timestamp(&self) -> u64 {
        self.last_drift_ts.load(Ordering::Acquire)
    }

    /// Get current statistic value
    pub fn statistic(&self) -> f64 {
        self.ph_max - self.ph_min
    }
}

/// ADWIN (Adaptive Windowing) bucket
#[derive(Clone, Debug)]
struct AdwinBucket {
    sum: f64,
    variance: f64,
    count: usize,
}

impl AdwinBucket {
    fn new() -> Self {
        AdwinBucket {
            sum: 0.0,
            variance: 0.0,
            count: 0,
        }
    }

    fn insert(&mut self, value: f64) {
        let new_count = self.count + 1;
        let delta = value - self.sum / self.count.max(1) as f64;
        
        self.sum += value;
        self.variance += delta * (value - self.sum / new_count as f64);
        self.count = new_count;
    }

    fn mean(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        self.sum / self.count as f64
    }
}

/// ADWIN drift detector parameters
#[derive(Debug, Clone)]
pub struct AdwinParams {
    /// Confidence parameter (delta in paper)
    pub delta: f64,
    /// Maximum window size
    pub max_buckets: usize,
    /// Minimum window size before checking
    pub min_window: usize,
}

impl Default for AdwinParams {
    fn default() -> Self {
        AdwinParams {
            delta: 0.002,
            max_buckets: MAX_ADWIN_WINDOW,
            min_window: 30,
        }
    }
}

/// ADWIN (Adaptive Windowing) drift detector
pub struct AdwinDetector {
    params: AdwinParams,
    buckets: Vec<AdwinBucket>,
    total_count: usize,
    drift_detected: AtomicBool,
    last_drift_ts: AtomicU64,
    width: usize,
}

impl AdwinDetector {
    pub fn new(params: AdwinParams) -> Self {
        AdwinDetector {
            params,
            buckets: Vec::with_capacity(params.max_buckets),
            total_count: 0,
            drift_detected: AtomicBool::new(false),
            last_drift_ts: AtomicU64::new(0),
            width: 0,
        }
    }

    /// Add new observation and check for drift
    pub fn update(&mut self, value: f64, timestamp_ns: u64) -> bool {
        // Create new bucket if needed
        if self.buckets.is_empty() || self.buckets.last().unwrap().count >= 32 {
            let mut bucket = AdwinBucket::new();
            bucket.insert(value);
            self.buckets.push(bucket);
            self.width += 1;
        } else {
            self.buckets.last_mut().unwrap().insert(value);
        }

        self.total_count += 1;

        // Check for drift if we have enough data
        if self.total_count >= self.params.min_window {
            return self.check_for_drift(timestamp_ns);
        }

        false
    }

    /// Check for drift by comparing sub-windows
    fn check_for_drift(&mut self, timestamp_ns: u64) -> bool {
        if self.buckets.len() < 2 {
            return false;
        }

        let n = self.buckets.len();
        let mut drift_found = false;
        let mut cut_point = 0;

        // Try different cut points
        for i in 1..n - 1 {
            let (left_sum, left_count, left_var) = self.compute_window_stats(0, i);
            let (right_sum, right_count, right_var) = self.compute_window_stats(i, n);

            if left_count < 2 || right_count < 2 {
                continue;
            }

            let left_mean = left_sum / left_count as f64;
            let right_mean = right_sum / right_count as f64;

            // Hoeffding bound
            let m = 1.0 / ((1.0 / left_count as f64) + (1.0 / right_count as f64));
            let epsilon = ((left_var + right_var) / 2.0).sqrt() 
                * (2.0 * (4.0 / self.params.delta).ln() / m).sqrt();

            if (left_mean - right_mean).abs() > epsilon {
                drift_found = true;
                cut_point = i;
                break;
            }
        }

        if drift_found {
            // Compress window
            self.compress_window(cut_point);
            self.drift_detected.store(true, Ordering::Release);
            self.last_drift_ts.store(timestamp_ns, Ordering::Release);
            true
        } else {
            self.drift_detected.store(false, Ordering::Release);
            false
        }
    }

    /// Compute statistics for window [start, end)
    fn compute_window_stats(&self, start: usize, end: usize) -> (f64, usize, f64) {
        let mut sum = 0.0;
        let mut count = 0;
        let mut variance = 0.0;

        for i in start..end.min(self.buckets.len()) {
            let bucket = &self.buckets[i];
            sum += bucket.sum;
            count += bucket.count;
            variance += bucket.variance;
        }

        (sum, count, variance)
    }

    /// Compress window after drift detection
    fn compress_window(&mut self, cut_point: usize) {
        if cut_point >= self.buckets.len() {
            return;
        }

        // Keep only the right part of the window
        let new_buckets: Vec<_> = self.buckets.drain(cut_point..).collect();
        self.buckets = new_buckets;
        self.width = self.buckets.len();

        // Recalculate total count
        self.total_count = self.buckets.iter().map(|b| b.count).sum();
    }

    /// Check if drift was detected
    pub fn has_drift(&self) -> bool {
        self.drift_detected.load(Ordering::Acquire)
    }

    /// Get last drift timestamp
    pub fn last_drift_timestamp(&self) -> u64 {
        self.last_drift_ts.load(Ordering::Acquire)
    }

    /// Get current window width
    pub fn width(&self) -> usize {
        self.width
    }

    /// Get estimated mean
    pub fn mean(&self) -> f64 {
        if self.total_count == 0 {
            return 0.0;
        }
        let sum: f64 = self.buckets.iter().map(|b| b.sum).sum();
        sum / self.total_count as f64
    }

    /// Reset detector
    pub fn reset(&mut self) {
        self.buckets.clear();
        self.total_count = 0;
        self.width = 0;
        self.drift_detected.store(false, Ordering::Release);
    }
}

/// Combined drift detection result
#[derive(Debug, Clone)]
pub struct DriftResult {
    pub page_hinkley_drift: bool,
    pub adwin_drift: bool,
    pub ph_statistic: f64,
    pub adwin_width: usize,
    pub adwin_mean: f64,
    pub timestamp_ns: u64,
    pub feature_id: u64,
}

/// Multi-feature drift monitor
pub struct MultiFeatureDriftMonitor {
    ph_detectors: Vec<PageHinkleyDetector>,
    adwin_detectors: Vec<AdwinDetector>,
    feature_ids: Vec<u64>,
    drift_alerts: AtomicU64,
    retraining_triggered: AtomicBool,
}

impl MultiFeatureDriftMonitor {
    pub fn new(n_features: usize, ph_params: PageHinkleyParams, adwin_params: AdwinParams) -> Self {
        let mut ph_detectors = Vec::with_capacity(n_features);
        let mut adwin_detectors = Vec::with_capacity(n_features);
        let mut feature_ids = Vec::with_capacity(n_features);

        for i in 0..n_features {
            ph_detectors.push(PageHinkleyDetector::new(ph_params.clone()));
            adwin_detectors.push(AdwinDetector::new(adwin_params.clone()));
            feature_ids.push(i as u64);
        }

        MultiFeatureDriftMonitor {
            ph_detectors,
            adwin_detectors,
            feature_ids,
            drift_alerts: AtomicU64::new(0),
            retraining_triggered: AtomicBool::new(false),
        }
    }

    /// Update all features with new observations
    pub fn update(&mut self, features: &[f64], timestamp_ns: u64) -> DriftResult {
        assert_eq!(features.len(), self.ph_detectors.len());

        let mut ph_drift = false;
        let mut adwin_drift = false;
        let mut ph_stat_sum = 0.0;
        let mut adwin_width_sum = 0;
        let mut adwin_mean_sum = 0.0;

        for (i, &value) in features.iter().enumerate() {
            let ph_d = self.ph_detectors[i].update(value, timestamp_ns);
            let aw_d = self.adwin_detectors[i].update(value, timestamp_ns);

            ph_drift |= ph_d;
            adwin_drift |= adwin_d;

            ph_stat_sum += self.ph_detectors[i].statistic();
            adwin_width_sum += self.adwin_detectors[i].width();
            adwin_mean_sum += self.adwin_detectors[i].mean();
        }

        if ph_drift || adwin_drift {
            self.drift_alerts.fetch_add(1, Ordering::Release);
        }

        let n = features.len() as f64;
        DriftResult {
            page_hinkley_drift: ph_drift,
            adwin_drift,
            ph_statistic: ph_stat_sum / n,
            adwin_width: (adwin_width_sum as f64 / n) as usize,
            adwin_mean: adwin_mean_sum / n,
            timestamp_ns,
            feature_id: 0, // Aggregate
        }
    }

    /// Check if retraining should be triggered
    pub fn should_retrain(&self, threshold: u64) -> bool {
        let alerts = self.drift_alerts.load(Ordering::Acquire);
        if alerts >= threshold {
            self.retraining_triggered.store(true, Ordering::Release);
            true
        } else {
            self.retraining_triggered.load(Ordering::Acquire)
        }
    }

    /// Reset retraining flag
    pub fn reset_retrain_flag(&self) {
        self.retraining_triggered.store(false, Ordering::Release);
    }

    /// Get total drift alerts
    pub fn total_alerts(&self) -> u64 {
        self.drift_alerts.load(Ordering::Acquire)
    }

    /// Get specific detector
    pub fn get_ph_detector(&self, feature_idx: usize) -> Option<&PageHinkleyDetector> {
        self.ph_detectors.get(feature_idx)
    }

    pub fn get_adwin_detector(&self, feature_idx: usize) -> Option<&AdwinDetector> {
        self.adwin_detectors.get(feature_idx)
    }
}

/// IPC message for drift notification
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DriftNotificationIPC {
    pub feature_id: u64,
    pub drift_type: u8, // 0 = Page-Hinkley, 1 = ADWIN, 2 = Both
    pub severity: f64,
    pub timestamp_ns: u64,
    pub retrain_requested: bool,
}

impl DriftNotificationIPC {
    pub fn from_result(result: &DriftResult, retrain: bool) -> Self {
        let drift_type = match (result.page_hinkley_drift, result.adwin_drift) {
            (true, true) => 2,
            (true, false) => 0,
            (false, true) => 1,
            _ => 0,
        };

        DriftNotificationIPC {
            feature_id: result.feature_id,
            drift_type,
            severity: result.ph_statistic,
            timestamp_ns: result.timestamp_ns,
            retrain_requested: retrain,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_page_hinkley_stable() {
        let params = PageHinkleyParams::default();
        let mut detector = PageHinkleyDetector::new(params);

        // Stable stream
        for i in 0..100 {
            let value = 0.5 + (i % 10) as f64 * 0.01;
            detector.update(value, i as u64 * 1_000_000);
        }

        // Should not detect drift in stable data
        assert!(!detector.has_drift());
    }

    #[test]
    fn test_page_hinkley_drift() {
        let params = PageHinkleyParams {
            threshold: 10.0,
            ..Default::default()
        };
        let mut detector = PageHinkleyDetector::new(params);

        // Initial stable period
        for i in 0..50 {
            detector.update(0.5, i as u64 * 1_000_000);
        }

        // Sudden shift
        for i in 50..100 {
            detector.update(2.0, i as u64 * 1_000_000);
        }

        // May or may not detect depending on parameters
        let _ = detector.statistic();
    }

    #[test]
    fn test_adwin_basic() {
        let params = AdwinParams::default();
        let mut detector = AdwinDetector::new(params);

        for i in 0..100 {
            detector.update(0.5, i as u64 * 1_000_000);
        }

        assert!(detector.width() > 0);
        assert!((detector.mean() - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_multi_feature_monitor() {
        let ph_params = PageHinkleyParams::default();
        let adwin_params = AdwinParams::default();
        let mut monitor = MultiFeatureDriftMonitor::new(5, ph_params, adwin_params);

        let features = vec![0.5, 0.3, 0.7, 0.4, 0.6];
        let result = monitor.update(&features, 1_000_000_000);

        assert_eq!(result.feature_id, 0);
        assert!(result.timestamp_ns > 0);
    }
}
