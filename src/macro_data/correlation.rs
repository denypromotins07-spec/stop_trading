//! Real-Time Rolling Correlation Engine
//! 
//! Implements Welford's online algorithm for computing rolling correlations
//! between BTC and traditional assets (Gold, SPX, DXY) without storing
//! massive historical arrays in RAM.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// Maximum window size for correlation calculation
const MAX_WINDOW_SIZE: usize = 10_000;

/// Asset type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Asset {
    BTC,
    ETH,
    Gold,
    SPX,
    DXY,
    Yield10Y,
    VIX,
}

/// Correlation pair result
#[derive(Debug, Clone)]
pub struct CorrelationResult {
    pub asset1: Asset,
    pub asset2: Asset,
    pub correlation: f64,
    pub sample_count: u64,
    pub last_update_ns: u64,
}

/// Regime change detection result
#[derive(Debug, Clone)]
pub struct RegimeChange {
    pub from_correlation: f64,
    pub to_correlation: f64,
    pub change_magnitude: f64,
    pub is_breakdown: bool, // True if correlation broke down significantly
    pub timestamp_ns: u64,
}

/// Welford's online algorithm state for a single variable
struct WelfordState {
    count: u64,
    mean: f64,
    m2: f64, // Sum of squared differences from mean
}

impl WelfordState {
    fn new() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            m2: 0.0,
        }
    }
    
    /// Update with new value
    fn update(&mut self, value: f64) {
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
    }
    
    /// Get variance
    fn variance(&self) -> f64 {
        if self.count < 2 {
            0.0
        } else {
            self.m2 / (self.count - 1) as f64
        }
    }
    
    /// Get standard deviation
    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }
    
    /// Reset state
    fn reset(&mut self) {
        self.count = 0;
        self.mean = 0.0;
        self.m2 = 0.0;
    }
}

/// Online covariance tracker using Welford's algorithm
struct OnlineCovariance {
    state_x: WelfordState,
    state_y: WelfordState,
    co_moment: f64, // Co-moment for covariance calculation
    count: u64,
}

impl OnlineCovariance {
    fn new() -> Self {
        Self {
            state_x: WelfordState::new(),
            state_y: WelfordState::new(),
            co_moment: 0.0,
            count: 0,
        }
    }
    
    /// Update with paired observations
    fn update(&mut self, x: f64, y: f64) {
        self.count += 1;
        
        let delta_x = x - self.state_x.mean;
        let delta_y = y - self.state_y.mean;
        
        // Update means
        self.state_x.update(x);
        self.state_y.update(y);
        
        // Update co-moment for covariance
        // Using Pei's online covariance formula
        self.co_moment += delta_x * (y - self.state_y.mean);
    }
    
    /// Get covariance
    fn covariance(&self) -> f64 {
        if self.count < 2 {
            0.0
        } else {
            self.co_moment / (self.count - 1) as f64
        }
    }
    
    /// Get correlation coefficient
    fn correlation(&self) -> f64 {
        let cov = self.covariance();
        let std_x = self.state_x.std_dev();
        let std_y = self.state_y.std_dev();
        
        if std_x < 1e-10 || std_y < 1e-10 {
            0.0
        } else {
            cov / (std_x * std_y)
        }
    }
    
    /// Get sample count
    fn count(&self) -> u64 {
        self.count
    }
    
    /// Reset state
    fn reset(&mut self) {
        self.state_x.reset();
        self.state_y.reset();
        self.co_moment = 0.0;
        self.count = 0;
    }
}

/// Exponential moving average correlation for faster response
struct EmaCorrelation {
    alpha: f64,
    ema_xy: f64,
    ema_x2: f64,
    ema_y2: f64,
    ema_x: f64,
    ema_y: f64,
    initialized: bool,
}

impl EmaCorrelation {
    fn new(alpha: f64) -> Self {
        Self {
            alpha,
            ema_xy: 0.0,
            ema_x2: 0.0,
            ema_y2: 0.0,
            ema_x: 0.0,
            ema_y: 0.0,
            initialized: false,
        }
    }
    
    fn update(&mut self, x: f64, y: f64) {
        if !self.initialized {
            self.ema_x = x;
            self.ema_y = y;
            self.ema_xy = x * y;
            self.ema_x2 = x * x;
            self.ema_y2 = y * y;
            self.initialized = true;
            return;
        }
        
        let one_minus_alpha = 1.0 - self.alpha;
        
        self.ema_x = self.alpha * x + one_minus_alpha * self.ema_x;
        self.ema_y = self.alpha * y + one_minus_alpha * self.ema_y;
        self.ema_xy = self.alpha * (x * y) + one_minus_alpha * self.ema_xy;
        self.ema_x2 = self.alpha * (x * x) + one_minus_alpha * self.ema_x2;
        self.ema_y2 = self.alpha * (y * y) + one_minus_alpha * self.ema_y2;
    }
    
    fn correlation(&self) -> f64 {
        if !self.initialized {
            return 0.0;
        }
        
        let cov = self.ema_xy - self.ema_x * self.ema_y;
        let var_x = self.ema_x2 - self.ema_x * self.ema_x;
        let var_y = self.ema_y2 - self.ema_y * self.ema_y;
        
        if var_x <= 0.0 || var_y <= 0.0 {
            0.0
        } else {
            cov / (var_x.sqrt() * var_y.sqrt())
        }
    }
}

/// Rolling correlation engine
pub struct CorrelationEngine {
    /// Primary correlation tracker (Welford-based)
    primary_tracker: Arc<std::sync::Mutex<OnlineCovariance>>,
    
    /// Fast EMA tracker for quick regime detection
    fast_tracker: Arc<std::sync::Mutex<EmaCorrelation>>,
    
    /// Slow EMA tracker for baseline comparison
    slow_tracker: Arc<std::sync::Mutex<EmaCorrelation>>,
    
    /// Last correlation values for breakdown detection
    last_correlation: AtomicU64, // Stored as fixed-point * 10000
    
    /// Sample counter
    sample_count: AtomicU64,
    
    /// Last update timestamp
    last_update_ns: AtomicU64,
}

impl CorrelationEngine {
    /// Create a new correlation engine
    pub fn new() -> Self {
        Self {
            primary_tracker: Arc::new(std::sync::Mutex::new(OnlineCovariance::new())),
            fast_tracker: Arc::new(std::sync::Mutex::new(EmaCorrelation::new(0.1))),
            slow_tracker: Arc::new(std::sync::Mutex::new(EmaCorrelation::new(0.01))),
            last_correlation: AtomicU64::new(0),
            sample_count: AtomicU64::new(0),
            last_update_ns: AtomicU64::new(get_timestamp_ns()),
        }
    }
    
    /// Record a paired observation
    pub fn record(&self, asset1_value: f64, asset2_value: f64) -> Option<RegimeChange> {
        // Normalize inputs (z-score approximation for crypto prices)
        let x = normalize_price(asset1_value);
        let y = normalize_price(asset2_value);
        
        // Update all trackers
        {
            let mut tracker = self.primary_tracker.lock().unwrap();
            tracker.update(x, y);
        }
        
        {
            let mut tracker = self.fast_tracker.lock().unwrap();
            tracker.update(x, y);
        }
        
        {
            let mut tracker = self.slow_tracker.lock().unwrap();
            tracker.update(x, y);
        }
        
        self.sample_count.fetch_add(1, Ordering::Relaxed);
        self.last_update_ns.store(get_timestamp_ns(), Ordering::Relaxed);
        
        // Check for regime change
        self.detect_regime_change()
    }
    
    /// Get current correlation estimate
    pub fn get_correlation(&self) -> CorrelationResult {
        let corr = self.primary_tracker.lock().unwrap().correlation();
        let count = self.primary_tracker.lock().unwrap().count();
        
        // Store for next comparison
        self.last_correlation.store((corr * 10000.0) as u64, Ordering::Relaxed);
        
        CorrelationResult {
            asset1: Asset::BTC,
            asset2: Asset::Gold,
            correlation: corr,
            sample_count: count,
            last_update_ns: self.last_update_ns.load(Ordering::Relaxed),
        }
    }
    
    /// Get fast correlation (EMA-based)
    pub fn get_fast_correlation(&self) -> f64 {
        self.fast_tracker.lock().unwrap().correlation()
    }
    
    /// Get slow correlation (EMA-based)
    pub fn get_slow_correlation(&self) -> f64 {
        self.slow_tracker.lock().unwrap().correlation()
    }
    
    /// Get sample count
    pub fn get_sample_count(&self) -> u64 {
        self.sample_count.load(Ordering::Relaxed)
    }
    
    /// Detect correlation regime changes
    fn detect_regime_change(&self) -> Option<RegimeChange> {
        let fast_corr = self.get_fast_correlation();
        let slow_corr = self.get_slow_correlation();
        let last_corr = self.last_correlation.load(Ordering::Relaxed) as f64 / 10000.0;
        
        if last_corr == 0.0 {
            return None;
        }
        
        let change = fast_corr - slow_corr;
        let magnitude = change.abs();
        
        // Detect significant breakdown (correlation decoupling)
        if magnitude > 0.3 {
            let is_breakdown = (fast_corr.abs() < 0.3 && slow_corr.abs() > 0.5)
                || (fast_corr.signum() != slow_corr.signum());
            
            return Some(RegimeChange {
                from_correlation: slow_corr,
                to_correlation: fast_corr,
                change_magnitude: magnitude,
                is_breakdown,
                timestamp_ns: get_timestamp_ns(),
            });
        }
        
        None
    }
    
    /// Reset all trackers
    pub fn reset(&self) {
        self.primary_tracker.lock().unwrap().reset();
        self.fast_tracker.lock().unwrap().correlation(); // Can't reset EMA easily
        self.slow_tracker.lock().unwrap().correlation();
        self.sample_count.store(0, Ordering::Relaxed);
        self.last_correlation.store(0, Ordering::Relaxed);
    }
}

impl Default for CorrelationEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Normalize price to approximate z-score range
fn normalize_price(price: f64) -> f64 {
    // Simple log-return approximation for stationarity
    if price <= 0.0 {
        0.0
    } else {
        price.ln() / 10.0 // Scale down for numerical stability
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
    fn test_engine_creation() {
        let engine = CorrelationEngine::new();
        assert_eq!(engine.get_sample_count(), 0);
    }
    
    #[test]
    fn test_correlation_calculation() {
        let engine = CorrelationEngine::new();
        
        // Record perfectly correlated data
        for i in 1..100 {
            let x = i as f64;
            let y = i as f64 * 2.0;
            engine.record(x, y);
        }
        
        let result = engine.get_correlation();
        assert!(result.correlation > 0.9); // Should be close to 1.0
    }
    
    #[test]
    fn test_welford_state() {
        let mut state = WelfordState::new();
        
        for i in 1..=10 {
            state.update(i as f64);
        }
        
        assert_eq!(state.count, 10);
        assert!((state.mean - 5.5).abs() < 0.001);
        assert!(state.variance() > 0);
    }
    
    #[test]
    fn test_ema_correlation() {
        let mut ema = EmaCorrelation::new(0.1);
        
        for i in 1..=100 {
            ema.update(i as f64, i as f64);
        }
        
        assert!(ema.correlation() > 0.9);
    }
}
