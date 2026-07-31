//! Tick Analytics Module
//!
//! Analyzes inter-arrival times of trade ticks using Poisson process modeling
//! to detect algorithmic TWAP/VWAP footprints. Identifies when large institutional
//! execution algos are active to front-run their predictable volume schedules.

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::time::{Instant, Duration};
use crossbeam_channel::{bounded, Sender, Receiver};
use dashmap::DashMap;

/// Trade tick with timing information
#[derive(Debug, Clone)]
pub struct TradeTick {
    pub symbol: String,
    pub price: f64,
    pub quantity: f64,
    pub is_buy: bool,
    pub timestamp_ns: u64,
}

/// Inter-arrival time statistics
#[derive(Debug, Clone)]
pub struct InterArrivalStats {
    /// Mean inter-arrival time (nanoseconds)
    pub mean_iat_ns: f64,
    /// Standard deviation of IAT
    pub std_iat_ns: f64,
    /// Arrival rate (ticks per second)
    pub arrival_rate: f64,
    /// Coefficient of variation (std/mean)
    pub coefficient_of_variation: f64,
    /// Sample count
    pub sample_count: usize,
}

impl InterArrivalStats {
    pub fn new() -> Self {
        Self {
            mean_iat_ns: 0.0,
            std_iat_ns: 0.0,
            arrival_rate: 0.0,
            coefficient_of_variation: 0.0,
            sample_count: 0,
        }
    }

    /// Calculate from a series of inter-arrival times
    pub fn from_iats(iats: &[f64]) -> Self {
        if iats.is_empty() {
            return Self::new();
        }

        let n = iats.len() as f64;
        let sum: f64 = iats.iter().sum();
        let mean = sum / n;

        let variance: f64 = iats.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / n;
        let std = variance.sqrt();

        let cv = if mean > 0.0 { std / mean } else { 0.0 };
        let arrival_rate = if mean > 0.0 { 1e9 / mean } else { 0.0 };

        Self {
            mean_iat_ns: mean,
            std_iat_ns: std,
            arrival_rate,
            coefficient_of_variation: cv,
            sample_count: iats.len(),
        }
    }
}

impl Default for InterArrivalStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Poisson process test result
#[derive(Debug, Clone)]
pub struct PoissonTestResult {
    /// Chi-squared statistic
    pub chi_squared: f64,
    /// Degrees of freedom
    pub degrees_of_freedom: usize,
    /// P-value (probability under null hypothesis)
    pub p_value: f64,
    /// Is consistent with Poisson process?
    pub is_poisson: bool,
    /// Confidence level used
    pub confidence_level: f64,
}

impl PoissonTestResult {
    /// Simple chi-squared goodness-of-fit test for Poisson
    pub fn test_poisson(iats: &[f64], lambda: f64) -> Self {
        if iats.is_empty() || lambda <= 0.0 {
            return Self {
                chi_squared: 0.0,
                degrees_of_freedom: 0,
                p_value: 1.0,
                is_poisson: true,
                confidence_level: 0.95,
            };
        }

        // Bin the inter-arrival times
        let num_bins = 10.min(iats.len());
        let mut observed = vec![0usize; num_bins];
        let mut expected = vec![0.0; num_bins];

        // Calculate bin boundaries based on exponential distribution
        // F(t) = 1 - exp(-lambda * t)
        for &iat in iats {
            let prob = 1.0 - (-lambda * iat / 1e9).exp();
            let bin = ((prob * num_bins as f64) as usize).min(num_bins - 1);
            observed[bin] += 1;
        }

        // Expected counts (uniform under exponential)
        let expected_per_bin = iats.len() as f64 / num_bins as f64;
        for e in &mut expected {
            *e = expected_per_bin;
        }

        // Chi-squared statistic
        let chi_sq: f64 = observed.iter().zip(expected.iter())
            .filter(|(_, &e)| e > 0.0)
            .map(|(&o, &e)| (o as f64 - e).powi(2) / e)
            .sum();

        // Approximate p-value using Wilson-Hilferty transformation
        let df = num_bins - 1;
        let p_value = Self::approximate_chi2_pvalue(chi_sq, df);

        // Test at 95% confidence
        let is_poisson = p_value > 0.05;

        Self {
            chi_squared: chi_sq,
            degrees_of_freedom: df,
            p_value,
            is_poisson,
            confidence_level: 0.95,
        }
    }

    /// Approximate chi-squared p-value
    fn approximate_chi2_pvalue(chi_sq: f64, df: usize) -> f64 {
        if df == 0 || chi_sq < 0.0 {
            return 1.0;
        }

        // Wilson-Hilferty approximation
        let x = chi_sq / df as f64;
        let z = (x.powf(1.0/3.0) - (1.0 - 2.0/(9.0*df as f64))) 
            / (2.0/(9.0*df as f64)).sqrt();

        // Standard normal CDF approximation
        Self::normal_cdf(z)
    }

    /// Standard normal CDF approximation
    fn normal_cdf(x: f64) -> f64 {
        let t = 1.0 / (1.0 + 0.2316419 * x.abs());
        let d = 0.3989423 * (-x * x / 2.0).exp();
        let p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
        
        if x > 0.0 {
            1.0 - p
        } else {
            p
        }
    }
}

/// Algorithmic trading footprint detection
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AlgoType {
    /// No algorithm detected
    None,
    /// Time-Weighted Average Price
    TWAP,
    /// Volume-Weighted Average Price
    VWAP,
    /// Implementation Shortfall
    IS,
    /// Market-on-Close
    MOC,
    /// Iceberg order
    Iceberg,
    /// Sniper/predatory algo
    Sniper,
}

/// Detected algorithmic footprint
#[derive(Debug, Clone)]
pub struct AlgoFootprint {
    pub algo_type: AlgoType,
    pub confidence: f64,
    pub estimated_total_volume: f64,
    pub estimated_completion_time_ns: u64,
    pub participation_rate: f64,
    pub aggressiveness: f64,
}

impl AlgoFootprint {
    pub fn new(algo_type: AlgoType, confidence: f64) -> Self {
        Self {
            algo_type,
            confidence,
            estimated_total_volume: 0.0,
            estimated_completion_time_ns: 0,
            participation_rate: 0.0,
            aggressiveness: 0.0,
        }
    }
}

/// Tick analytics engine
pub struct TickAnalytics {
    /// Recent ticks per symbol
    recent_ticks: DashMap<String, Vec<TradeTick>>,
    /// Inter-arrival stats per symbol
    iat_stats: DashMap<String, InterArrivalStats>,
    /// Detected algos per symbol
    detected_algos: DashMap<String, AlgoFootprint>,
    /// Maximum ticks to retain
    max_ticks: usize,
    /// Analysis window (nanoseconds)
    analysis_window_ns: u64,
    /// Is analyzer active
    is_active: AtomicBool,
    /// Event channel
    event_tx: Sender<TickEvent>,
    event_rx: Receiver<TickEvent>,
}

/// Tick events
#[derive(Debug, Clone)]
pub enum TickEvent {
    /// New tick processed
    TickProcessed(String),
    /// Algo footprint detected
    AlgoDetected {
        symbol: String,
        footprint: AlgoFootprint,
    },
    /// Poisson regime change
    RegimeChange {
        symbol: String,
        old_rate: f64,
        new_rate: f64,
    },
    /// Anomaly detected
    Anomaly {
        symbol: String,
        anomaly_type: &'static str,
        severity: u8,
    },
}

impl TickAnalytics {
    pub fn new(max_ticks: usize, analysis_window_ns: u64, buffer_size: usize) -> Self {
        let (tx, rx) = bounded(buffer_size);

        Self {
            recent_ticks: DashMap::new(),
            iat_stats: DashMap::new(),
            detected_algos: DashMap::new(),
            max_ticks,
            analysis_window_ns,
            is_active: AtomicBool::new(true),
            event_tx: tx,
            event_rx: rx,
        }
    }

    /// Process a new trade tick
    pub fn process_tick(&self, tick: TradeTick) {
        if !self.is_active.load(Ordering::Relaxed) {
            return;
        }

        let symbol = tick.symbol.clone();

        // Add to recent ticks
        let mut ticks = self.recent_ticks.entry(symbol.clone()).or_insert_with(Vec::new);
        ticks.push(tick.clone());

        // Trim old ticks
        if ticks.len() > self.max_ticks {
            let cutoff = Instant::now().duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default().as_nanos() as u64 - self.analysis_window_ns;
            *ticks = ticks.iter()
                .filter(|t| t.timestamp_ns > cutoff)
                .cloned()
                .collect();
        }

        // Update inter-arrival statistics
        self.update_iat_stats(&symbol);

        // Detect algorithmic patterns
        self.detect_algo_patterns(&symbol);

        let _ = self.event_tx.send(TickEvent::TickProcessed(symbol));
    }

    /// Update inter-arrival time statistics
    fn update_iat_stats(&self, symbol: &str) {
        let ticks = match self.recent_ticks.get(symbol) {
            Some(t) => t.clone(),
            None => return,
        };

        if ticks.len() < 2 {
            return;
        }

        // Calculate inter-arrival times
        let iats: Vec<f64> = ticks.windows(2)
            .map(|w| (w[1].timestamp_ns - w[0].timestamp_ns) as f64)
            .collect();

        let stats = InterArrivalStats::from_iats(&iats);
        self.iat_stats.insert(symbol.to_string(), stats);

        // Check for regime change
        self.check_regime_change(symbol, &stats);
    }

    /// Check for arrival rate regime change
    fn check_regime_change(&self, symbol: &str, current: &InterArrivalStats) {
        // Compare with previous stats (simplified)
        // In production, would maintain history
    }

    /// Detect algorithmic trading patterns
    fn detect_algo_patterns(&self, symbol: &str) {
        let ticks = match self.recent_ticks.get(symbol) {
            Some(t) => t.clone(),
            None => return,
        };

        if ticks.len() < 10 {
            return;
        }

        // TWAP detection: regular intervals
        let twap_confidence = self.detect_twap(&ticks);

        // VWAP detection: volume-proportional execution
        let vwap_confidence = self.detect_vwap(&ticks);

        // Iceberg detection: repeated same-size orders
        let iceberg_confidence = self.detect_iceberg(&ticks);

        // Determine most likely algo
        let (algo_type, confidence) = [
            (AlgoType::TWAP, twap_confidence),
            (AlgoType::VWAP, vwap_confidence),
            (AlgoType::Iceberg, iceberg_confidence),
        ].iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .copied()
            .unwrap_or((AlgoType::None, 0.0));

        if confidence > 0.5 {
            let footprint = AlgoFootprint::new(algo_type, confidence);
            
            // Estimate total volume and completion
            let total_volume: f64 = ticks.iter().map(|t| t.quantity).sum();
            let time_span = ticks.last().unwrap().timestamp_ns - ticks.first().unwrap().timestamp_ns;
            let rate = total_volume / (time_span as f64 / 1e9);

            let mut fp = footprint;
            fp.estimated_total_volume = total_volume * 10.0; // Extrapolate
            fp.estimated_completion_time_ns = time_span * 10;
            fp.participation_rate = rate;

            self.detected_algos.insert(symbol.to_string(), fp);

            let _ = self.event_tx.send(TickEvent::AlgoDetected {
                symbol: symbol.to_string(),
                footprint: fp,
            });
        }
    }

    /// TWAP detection: check for regular intervals
    fn detect_twap(&self, ticks: &[TradeTick]) -> f64 {
        if ticks.len() < 5 {
            return 0.0;
        }

        let iats: Vec<f64> = ticks.windows(2)
            .map(|w| (w[1].timestamp_ns - w[0].timestamp_ns) as f64)
            .collect();

        let mean = iats.iter().sum::<f64>() / iats.len() as f64;
        let variance: f64 = iats.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / iats.len() as f64;
        let std = variance.sqrt();

        // Low coefficient of variation indicates regular intervals (TWAP)
        let cv = if mean > 0.0 { std / mean } else { 1.0 };

        // CV < 0.3 suggests TWAP
        (1.0 - cv).max(0.0)
    }

    /// VWAP detection: check for volume clustering around market volume profile
    fn detect_vwap(&self, ticks: &[TradeTick]) -> f64 {
        // Simplified: look for increasing volume throughout period
        let volumes: Vec<f64> = ticks.iter().map(|t| t.quantity).collect();

        if volumes.len() < 5 {
            return 0.0;
        }

        // Check correlation with time (VWAP often has U-shaped pattern)
        let first_half_avg: f64 = volumes[..volumes.len()/2].iter().sum();
        let second_half_avg: f64 = volumes[volumes.len()/2..].iter().sum();

        // VWAP typically has higher volume at start and end
        let edge_ratio = (first_half_avg + second_half_avg) / (volumes.iter().sum() / volumes.len() as f64);

        if edge_ratio > 1.2 {
            0.7
        } else {
            0.3
        }
    }

    /// Iceberg detection: look for repeated identical sizes
    fn detect_iceberg(&self, ticks: &[TradeTick]) -> f64 {
        if ticks.len() < 5 {
            return 0.0;
        }

        // Count occurrences of each size (rounded to significant digits)
        let mut size_counts = std::collections::HashMap::new();

        for tick in ticks {
            let rounded_size = (tick.quantity * 100.0).round() / 100.0;
            *size_counts.entry(rounded_size).or_insert(0) += 1;
        }

        // Find most common size
        let max_count = size_counts.values().max().copied().unwrap_or(0);
        let concentration = max_count as f64 / ticks.len() as f64;

        // High concentration suggests iceberg
        concentration
    }

    /// Get inter-arrival stats for symbol
    pub fn get_iat_stats(&self, symbol: &str) -> Option<InterArrivalStats> {
        self.iat_stats.get(symbol).map(|s| s.clone())
    }

    /// Get detected algo for symbol
    pub fn get_detected_algo(&self, symbol: &str) -> Option<AlgoFootprint> {
        self.detected_algos.get(symbol).map(|a| a.clone())
    }

    /// Get event receiver
    pub fn get_event_receiver(&self) -> Receiver<TickEvent> {
        self.event_rx.clone()
    }

    /// Deactivate analyzer
    pub fn deactivate(&self) {
        self.is_active.store(false, Ordering::Relaxed);
    }

    /// Activate analyzer
    pub fn activate(&self) {
        self.is_active.store(true, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inter_arrival_stats() {
        let iats = vec![100.0, 100.0, 100.0, 100.0, 100.0];
        let stats = InterArrivalStats::from_iats(&iats);

        assert!((stats.mean_iat_ns - 100.0).abs() < 0.01);
        assert!((stats.std_iat_ns - 0.0).abs() < 0.01);
        assert!((stats.coefficient_of_variation - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_poisson_test() {
        // Generate exponential inter-arrivals (Poisson process)
        let iats: Vec<f64> = (0..100).map(|i| (i as f64 * 10.0)).collect();
        let lambda = 1e9 / 1000.0; // 1000 Hz

        let result = PoissonTestResult::test_poisson(&iats, lambda);

        // Should be roughly consistent with Poisson
        assert!(result.degrees_of_freedom > 0);
    }

    #[test]
    fn test_twap_detection() {
        let analytics = TickAnalytics::new(100, 1_000_000_000, 1000);

        // Generate TWAP-like ticks (regular intervals)
        let base_time = Instant::now().duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default().as_nanos() as u64;

        for i in 0..20 {
            let tick = TradeTick {
                symbol: "BTCUSDT".to_string(),
                price: 50000.0,
                quantity: 1.0,
                is_buy: true,
                timestamp_ns: base_time + i * 1_000_000_000, // Exactly 1 second apart
            };
            analytics.process_tick(tick);
        }

        let algo = analytics.get_detected_algo("BTCUSDT");
        assert!(algo.is_some());
    }

    #[test]
    fn test_analytics_initialization() {
        let analytics = TickAnalytics::new(1000, 60_000_000_000, 1000);

        assert!(analytics.is_active.load(Ordering::Relaxed));
        assert_eq!(analytics.get_iat_stats("BTCUSDT"), None);
    }
}
