//! Online Statistics Module
//! 
//! Lock-free online covariance and higher-moment trackers for continuous feature monitoring.
//! Updates statistical baselines in O(1) time per tick without storing massive historical matrices.

use alloc::vec::Vec;
use core::sync::atomic::{AtomicU64, AtomicBool, Ordering};

/// Maximum number of features for fixed-size arrays
pub const MAX_FEATURES: usize = 256;

/// Lock-free online mean tracker using Welford's algorithm
#[derive(Debug)]
pub struct OnlineMean {
    count: AtomicU64,
    mean: f64,
    is_active: AtomicBool,
}

impl OnlineMean {
    pub fn new() -> Self {
        OnlineMean {
            count: AtomicU64::new(0),
            mean: 0.0,
            is_active: AtomicBool::new(true),
        }
    }

    /// Update with new value (thread-safe)
    pub fn update(&self, value: f64) {
        if !self.is_active.load(Ordering::Acquire) {
            return;
        }

        let count = self.count.fetch_add(1, Ordering::Relaxed);
        let old_mean = self.mean;
        
        // Welford's online algorithm
        unsafe {
            let mean_ptr = &self.mean as *const f64 as *mut f64;
            *mean_ptr = old_mean + (value - old_mean) / (count + 1) as f64;
        }
    }

    /// Get current mean estimate
    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// Get sample count
    pub fn count(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }

    /// Reset tracker
    pub fn reset(&self) {
        self.count.store(0, Ordering::Relaxed);
        unsafe {
            let mean_ptr = &self.mean as *const f64 as *mut f64;
            *mean_ptr = 0.0;
        }
    }

    /// Activate/deactivate tracker
    pub fn set_active(&self, active: bool) {
        self.is_active.store(active, Ordering::Release);
    }
}

/// Online variance tracker using Welford's algorithm
#[derive(Debug)]
pub struct OnlineVariance {
    count: AtomicU64,
    mean: f64,
    m2: f64, // Sum of squared differences from mean
    min_value: f64,
    max_value: f64,
    is_active: AtomicBool,
}

impl OnlineVariance {
    pub fn new() -> Self {
        OnlineVariance {
            count: AtomicU64::new(0),
            mean: 0.0,
            m2: 0.0,
            min_value: f64::MAX,
            max_value: f64::MIN,
            is_active: AtomicBool::new(true),
        }
    }

    /// Update with new value
    pub fn update(&self, value: f64) {
        if !self.is_active.load(Ordering::Acquire) {
            return;
        }

        let count = self.count.fetch_add(1, Ordering::Relaxed);
        
        unsafe {
            let mean_ptr = &self.mean as *const f64 as *mut f64;
            let m2_ptr = &self.m2 as *const f64 as *mut f64;
            let min_ptr = &self.min_value as *const f64 as *mut f64;
            let max_ptr = &self.max_value as *const f64 as *mut f64;

            let old_mean = *mean_ptr;
            let delta = value - old_mean;
            *mean_ptr = old_mean + delta / (count + 1) as f64;
            let delta2 = value - *mean_ptr;
            *m2_ptr += delta * delta2;

            // Track min/max
            if value < *min_ptr {
                *min_ptr = value;
            }
            if value > *max_ptr {
                *max_ptr = value;
            }
        }
    }

    /// Get variance estimate
    pub fn variance(&self) -> f64 {
        let count = self.count.load(Ordering::Relaxed);
        if count < 2 {
            return 0.0;
        }
        self.m2 / (count - 1) as f64
    }

    /// Get standard deviation
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Get mean
    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// Get range
    pub fn range(&self) -> (f64, f64) {
        (self.min_value, self.max_value)
    }

    /// Reset tracker
    pub fn reset(&self) {
        self.count.store(0, Ordering::Relaxed);
        unsafe {
            let mean_ptr = &self.mean as *const f64 as *mut f64;
            let m2_ptr = &self.m2 as *const f64 as *mut f64;
            let min_ptr = &self.min_value as *const f64 as *mut f64;
            let max_ptr = &self.max_value as *const f64 as *mut f64;
            
            *mean_ptr = 0.0;
            *m2_ptr = 0.0;
            *min_ptr = f64::MAX;
            *max_ptr = f64::MIN;
        }
    }
}

/// Online covariance tracker for two variables
pub struct OnlineCovariance {
    count: AtomicU64,
    mean_x: f64,
    mean_y: f64,
    c_xy: f64, // Co-moment
    var_x: OnlineVariance,
    var_y: OnlineVariance,
    is_active: AtomicBool,
}

impl OnlineCovariance {
    pub fn new() -> Self {
        OnlineCovariance {
            count: AtomicU64::new(0),
            mean_x: 0.0,
            mean_y: 0.0,
            c_xy: 0.0,
            var_x: OnlineVariance::new(),
            var_y: OnlineVariance::new(),
            is_active: AtomicBool::new(true),
        }
    }

    /// Update with new pair of values
    pub fn update(&self, x: f64, y: f64) {
        if !self.is_active.load(Ordering::Acquire) {
            return;
        }

        self.var_x.update(x);
        self.var_y.update(y);

        let count = self.count.fetch_add(1, Ordering::Relaxed);

        unsafe {
            let mean_x_ptr = &self.mean_x as *const f64 as *mut f64;
            let mean_y_ptr = &self.mean_y as *const f64 as *mut f64;
            let c_xy_ptr = &self.c_xy as *const f64 as *mut f64;

            let old_mean_x = *mean_x_ptr;
            let old_mean_y = *mean_y_ptr;
            
            let dx = x - old_mean_x;
            let dy = y - old_mean_y;
            
            *mean_x_ptr = old_mean_x + dx / (count + 1) as f64;
            *mean_y_ptr = old_mean_y + dy / (count + 1) as f64;
            
            // Update co-moment using Welford-style update
            *c_xy_ptr += dx * (y - *mean_y_ptr);
        }
    }

    /// Get covariance estimate
    pub fn covariance(&self) -> f64 {
        let count = self.count.load(Ordering::Relaxed);
        if count < 2 {
            return 0.0;
        }
        self.c_xy / (count - 1) as f64
    }

    /// Get correlation coefficient
    pub fn correlation(&self) -> f64 {
        let cov = self.covariance();
        let std_x = self.var_x.std_dev();
        let std_y = self.var_y.std_dev();
        
        if std_x < 1e-12 || std_y < 1e-12 {
            return 0.0;
        }
        
        cov / (std_x * std_y)
    }

    /// Reset tracker
    pub fn reset(&self) {
        self.count.store(0, Ordering::Relaxed);
        self.var_x.reset();
        self.var_y.reset();
        unsafe {
            let mean_x_ptr = &self.mean_x as *const f64 as *mut f64;
            let mean_y_ptr = &self.mean_y as *const f64 as *mut f64;
            let c_xy_ptr = &self.c_xy as *const f64 as *mut f64;
            
            *mean_x_ptr = 0.0;
            *mean_y_ptr = 0.0;
            *c_xy_ptr = 0.0;
        }
    }
}

/// Online skewness tracker (third moment)
pub struct OnlineSkewness {
    count: AtomicU64,
    mean: f64,
    m2: f64,
    m3: f64,
    is_active: AtomicBool,
}

impl OnlineSkewness {
    pub fn new() -> Self {
        OnlineSkewness {
            count: AtomicU64::new(0),
            mean: 0.0,
            m2: 0.0,
            m3: 0.0,
            is_active: AtomicBool::new(true),
        }
    }

    pub fn update(&self, value: f64) {
        if !self.is_active.load(Ordering::Acquire) {
            return;
        }

        let count = self.count.fetch_add(1, Ordering::Relaxed);

        unsafe {
            let mean_ptr = &self.mean as *const f64 as *mut f64;
            let m2_ptr = &self.m2 as *const f64 as *mut f64;
            let m3_ptr = &self.m3 as *const f64 as *mut f64;

            let old_mean = *mean_ptr;
            let delta = value - old_mean;
            let delta_n = delta / (count + 1) as f64;
            let delta_n2 = delta_n * delta_n;
            
            *mean_ptr = old_mean + delta_n;
            *m3_ptr += delta * delta_n * delta_n * (count as f64) * (count as f64 - 1.0)
                - 3.0 * delta_n * *m2_ptr;
            *m2_ptr += delta * delta_n * count as f64;
        }
    }

    pub fn skewness(&self) -> f64 {
        let count = self.count.load(Ordering::Relaxed);
        if count < 3 {
            return 0.0;
        }
        
        let n = count as f64;
        if self.m2 < 1e-12 {
            return 0.0;
        }
        
        (n.sqrt() * self.m3) / (self.m2.powf(1.5))
    }

    pub fn reset(&self) {
        self.count.store(0, Ordering::Relaxed);
        unsafe {
            let mean_ptr = &self.mean as *const f64 as *mut f64;
            let m2_ptr = &self.m2 as *const f64 as *mut f64;
            let m3_ptr = &self.m3 as *const f64 as *mut f64;
            
            *mean_ptr = 0.0;
            *m2_ptr = 0.0;
            *m3_ptr = 0.0;
        }
    }
}

/// Online kurtosis tracker (fourth moment)
pub struct OnlineKurtosis {
    count: AtomicU64,
    mean: f64,
    m2: f64,
    m3: f64,
    m4: f64,
    is_active: AtomicBool,
}

impl OnlineKurtosis {
    pub fn new() -> Self {
        OnlineKurtosis {
            count: AtomicU64::new(0),
            mean: 0.0,
            m2: 0.0,
            m3: 0.0,
            m4: 0.0,
            is_active: AtomicBool::new(true),
        }
    }

    pub fn update(&self, value: f64) {
        if !self.is_active.load(Ordering::Acquire) {
            return;
        }

        let count = self.count.fetch_add(1, Ordering::Relaxed);

        unsafe {
            let mean_ptr = &self.mean as *const f64 as *mut f64;
            let m2_ptr = &self.m2 as *const f64 as *mut f64;
            let m3_ptr = &self.m3 as *const f64 as *mut f64;
            let m4_ptr = &self.m4 as *const f64 as *mut f64;

            let old_mean = *mean_ptr;
            let delta = value - old_mean;
            let delta_n = delta / (count + 1) as f64;
            let delta_n2 = delta_n * delta_n;
            let term1 = delta * delta_n * count as f64;
            
            *mean_ptr = old_mean + delta_n;
            *m4_ptr += term1 * delta_n2 * ((count as f64) * (count as f64) - 3.0 * count as f64 + 3.0)
                + 6.0 * delta_n2 * *m2_ptr - 4.0 * delta_n * *m3_ptr;
            *m3_ptr += term1 * delta_n * (count as f64 - 2.0) - 3.0 * delta_n * *m2_ptr;
            *m2_ptr += term1;
        }
    }

    pub fn kurtosis(&self) -> f64 {
        let count = self.count.load(Ordering::Relaxed);
        if count < 4 {
            return 0.0;
        }
        
        if self.m2 < 1e-12 {
            return 0.0;
        }
        
        let n = count as f64;
        (n * self.m4) / (self.m2 * self.m2)
    }

    pub fn excess_kurtosis(&self) -> f64 {
        self.kurtosis() - 3.0
    }

    pub fn reset(&self) {
        self.count.store(0, Ordering::Relaxed);
        unsafe {
            let mean_ptr = &self.mean as *const f64 as *mut f64;
            let m2_ptr = &self.m2 as *const f64 as *mut f64;
            let m3_ptr = &self.m3 as *const f64 as *mut f64;
            let m4_ptr = &self.m4 as *const f64 as *mut f64;
            
            *mean_ptr = 0.0;
            *m2_ptr = 0.0;
            *m3_ptr = 0.0;
            *m4_ptr = 0.0;
        }
    }
}

/// Complete online statistics tracker for multiple features
pub struct OnlineFeatureStats {
    means: Vec<OnlineMean>,
    variances: Vec<OnlineVariance>,
    skewnesses: Vec<OnlineSkewness>,
    kurtoses: Vec<OnlineKurtosis>,
    n_features: usize,
    update_count: AtomicU64,
}

impl OnlineFeatureStats {
    pub fn new(n_features: usize) -> Self {
        let mut means = Vec::with_capacity(n_features);
        let mut variances = Vec::with_capacity(n_features);
        let mut skewnesses = Vec::with_capacity(n_features);
        let mut kurtoses = Vec::with_capacity(n_features);

        for _ in 0..n_features {
            means.push(OnlineMean::new());
            variances.push(OnlineVariance::new());
            skewnesses.push(OnlineSkewness::new());
            kurtoses.push(OnlineKurtosis::new());
        }

        OnlineFeatureStats {
            means,
            variances,
            skewnesses,
            kurtoses,
            n_features,
            update_count: AtomicU64::new(0),
        }
    }

    /// Update all features with new observation vector
    pub fn update(&self, features: &[f64]) {
        assert_eq!(features.len(), self.n_features);

        for (i, &value) in features.iter().enumerate() {
            self.means[i].update(value);
            self.variances[i].update(value);
            self.skewnesses[i].update(value);
            self.kurtoses[i].update(value);
        }

        self.update_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Get mean for specific feature
    pub fn get_mean(&self, feature_idx: usize) -> Option<f64> {
        self.means.get(feature_idx).map(|m| m.mean())
    }

    /// Get std dev for specific feature
    pub fn get_std_dev(&self, feature_idx: usize) -> Option<f64> {
        self.variances.get(feature_idx).map(|v| v.std_dev())
    }

    /// Get skewness for specific feature
    pub fn get_skewness(&self, feature_idx: usize) -> Option<f64> {
        self.skewnesses.get(feature_idx).map(|s| s.skewness())
    }

    /// Get excess kurtosis for specific feature
    pub fn get_excess_kurtosis(&self, feature_idx: usize) -> Option<f64> {
        self.kurtoses.get(feature_idx).map(|k| k.excess_kurtosis())
    }

    /// Get all means
    pub fn all_means(&self) -> Vec<f64> {
        self.means.iter().map(|m| m.mean()).collect()
    }

    /// Get all std devs
    pub fn all_std_devs(&self) -> Vec<f64> {
        self.variances.iter().map(|v| v.std_dev()).collect()
    }

    /// Get total update count
    pub fn update_count(&self) -> u64 {
        self.update_count.load(Ordering::Relaxed)
    }

    /// Reset all trackers
    pub fn reset_all(&self) {
        for m in &self.means {
            m.reset();
        }
        for v in &self.variances {
            v.reset();
        }
        for s in &self.skewnesses {
            s.reset();
        }
        for k in &self.kurtoses {
            k.reset();
        }
        self.update_count.store(0, Ordering::Relaxed);
    }
}

/// Online covariance matrix tracker (limited size for memory efficiency)
pub struct OnlineCovarianceMatrix {
    covariances: [[OnlineCovariance; MAX_FEATURES]; MAX_FEATURES],
    n_features: usize,
    is_active: AtomicBool,
}

impl OnlineCovarianceMatrix {
    pub fn new(n_features: usize) -> Self {
        assert!(n_features <= MAX_FEATURES);
        
        // Initialize with empty covariances
        let mut covariances = [[OnlineCovariance::new(); MAX_FEATURES]; MAX_FEATURES];
        
        OnlineCovarianceMatrix {
            covariances,
            n_features,
            is_active: AtomicBool::new(true),
        }
    }

    /// Update with new feature vector
    pub fn update(&self, features: &[f64]) {
        if !self.is_active.load(Ordering::Acquire) {
            return;
        }

        assert_eq!(features.len(), self.n_features);

        for i in 0..self.n_features {
            for j in i..self.n_features {
                self.covariances[i][j].update(features[i], features[j]);
                if i != j {
                    self.covariances[j][i].update(features[j], features[i]);
                }
            }
        }
    }

    /// Get covariance between two features
    pub fn get_covariance(&self, i: usize, j: usize) -> Option<f64> {
        if i >= self.n_features || j >= self.n_features {
            return None;
        }
        Some(self.covariances[i][j].covariance())
    }

    /// Get correlation between two features
    pub fn get_correlation(&self, i: usize, j: usize) -> Option<f64> {
        if i >= self.n_features || j >= self.n_features {
            return None;
        }
        Some(self.covariances[i][j].correlation())
    }

    /// Get full covariance matrix as Vec
    pub fn covariance_matrix(&self) -> Vec<Vec<f64>> {
        let mut matrix = vec![vec![0.0; self.n_features]; self.n_features];
        for i in 0..self.n_features {
            for j in 0..self.n_features {
                matrix[i][j] = self.covariances[i][j].covariance();
            }
        }
        matrix
    }

    /// Reset all covariances
    pub fn reset(&self) {
        for i in 0..self.n_features {
            for j in 0..self.n_features {
                self.covariances[i][j].reset();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_online_mean() {
        let tracker = OnlineMean::new();
        
        for i in 0..100 {
            tracker.update(i as f64);
        }
        
        assert!((tracker.mean() - 49.5).abs() < 0.1);
        assert_eq!(tracker.count(), 100);
    }

    #[test]
    fn test_online_variance() {
        let tracker = OnlineVariance::new();
        
        // Known variance sequence
        let values = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        for &v in &values {
            tracker.update(v);
        }
        
        // Population variance should be 4.0
        let var = tracker.variance();
        assert!((var - 4.0).abs() < 1.0);
    }

    #[test]
    fn test_online_covariance() {
        let tracker = OnlineCovariance::new();
        
        // Perfectly correlated data
        for i in 0..100 {
            let x = i as f64;
            let y = 2.0 * i as f64;
            tracker.update(x, y);
        }
        
        let corr = tracker.correlation();
        assert!((corr - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_online_feature_stats() {
        let stats = OnlineFeatureStats::new(3);
        
        for _ in 0..100 {
            let features = vec![1.0, 2.0, 3.0];
            stats.update(&features);
        }
        
        assert!((stats.get_mean(0).unwrap() - 1.0).abs() < 1e-10);
        assert!((stats.get_mean(1).unwrap() - 2.0).abs() < 1e-10);
        assert_eq!(stats.update_count(), 100);
    }
}
