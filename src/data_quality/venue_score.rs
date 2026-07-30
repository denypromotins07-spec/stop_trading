//! Dynamic Venue Scoring Engine
//! 
//! Evaluates latency, depth, and fee structures.
//! Automatically re-weights Smart Order Router (SOR) to favor reliable venues.

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::time::Instant;
use dashmap::DashMap;

/// Maximum venues tracked
const MAX_VENUES: usize = 32;

/// Venue score components
#[derive(Debug, Clone)]
pub struct VenueScore {
    /// Overall score (0-100)
    pub overall: f64,
    /// Latency score (0-100)
    pub latency_score: f64,
    /// Depth score (0-100)
    pub depth_score: f64,
    /// Fee score (0-100)
    pub fee_score: f64,
    /// Reliability score (0-100)
    pub reliability_score: f64,
    /// Last update timestamp
    pub updated_ns: u64,
}

/// Venue metrics
#[derive(Debug, Clone)]
pub struct VenueMetrics {
    /// Average latency in microseconds
    pub avg_latency_us: f64,
    /// Latency standard deviation
    pub latency_stddev_us: f64,
    /// Best bid depth (in quote currency)
    pub best_bid_depth: f64,
    /// Best ask depth (in quote currency)
    pub best_ask_depth: f64,
    /// Total depth within 1%
    pub depth_1pct: f64,
    /// Maker fee in bps
    pub maker_fee_bps: f64,
    /// Taker fee in bps
    pub taker_fee_bps: f64,
    /// Success rate (0-1)
    pub success_rate: f64,
    /// Error count last minute
    pub errors_last_minute: u32,
    /// Quote rate per second
    pub quote_rate: f64,
}

/// SOR weight for a venue
#[derive(Debug, Clone)]
pub struct SORWeight {
    /// Venue identifier
    pub venue: String,
    /// Weight for routing (0-1)
    pub weight: f64,
    /// Rank among venues
    pub rank: usize,
}

/// Lock-free venue scoring engine
pub struct VenueScorer {
    /// Scores per venue per symbol
    scores: DashMap<String, VenueScore>,
    /// Metrics per venue per symbol
    metrics: DashMap<String, VenueMetrics>,
    /// Latency samples (ring buffer simulation)
    latency_samples: DashMap<String, Vec<f64>>,
    /// Weights for SOR
    sor_weights: DashMap<String, Vec<SORWeight>>,
    /// Score weights configuration
    latency_weight: f64,
    depth_weight: f64,
    fee_weight: f64,
    reliability_weight: f64,
    /// Is scorer active
    is_active: AtomicBool,
}

impl VenueScorer {
    pub fn new(
        latency_weight: f64,
        depth_weight: f64,
        fee_weight: f64,
        reliability_weight: f64,
    ) -> Self {
        let total = latency_weight + depth_weight + fee_weight + reliability_weight;
        
        Self {
            scores: DashMap::new(),
            metrics: DashMap::new(),
            latency_samples: DashMap::new(),
            sor_weights: DashMap::new(),
            latency_weight: latency_weight / total,
            depth_weight: depth_weight / total,
            fee_weight: fee_weight / total,
            reliability_weight: reliability_weight / total,
            is_active: AtomicBool::new(true),
        }
    }

    /// Update metrics for a venue/symbol
    pub fn update_metrics(&self, venue: &str, symbol: &str, metrics: VenueMetrics) {
        if !self.is_active.load(Ordering::Relaxed) {
            return;
        }

        let key = format!("{}:{}", venue, symbol);
        
        // Store metrics
        self.metrics.insert(key.clone(), metrics);

        // Update latency samples
        let mut samples = self.latency_samples.entry(key.clone()).or_insert_with(Vec::new);
        samples.push(metrics.avg_latency_us);
        if samples.len() > 100 {
            samples.remove(0);
        }

        // Calculate score
        let score = self.calculate_score(&key, &metrics);
        self.scores.insert(key, score);

        // Update SOR weights
        self.update_sor_weights(symbol);
    }

    fn calculate_score(&self, key: &str, metrics: &VenueMetrics) -> VenueScore {
        let now_ns = Instant::now().duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default().as_nanos() as u64;

        // Latency score: lower is better, normalize to 0-100
        // Assume 1ms = 50 score, 10ms = 0 score
        let latency_score = (100.0 - (metrics.avg_latency_us / 100.0).min(100.0)).max(0.0);

        // Depth score: higher is better
        // Normalize based on expected depth (e.g., $1M = 100 score)
        let total_depth = metrics.depth_1pct;
        let depth_score = (total_depth / 1_000_000.0 * 100.0).min(100.0).max(0.0);

        // Fee score: lower is better
        let avg_fee = (metrics.maker_fee_bps + metrics.taker_fee_bps) / 2.0;
        // Assume 1bps = 100 score, 50bps = 0 score
        let fee_score = (100.0 - avg_fee * 2.0).max(0.0);

        // Reliability score: based on success rate and error count
        let reliability_score = (metrics.success_rate * 100.0) 
            - (metrics.errors_last_minute as f64 * 2.0);
        let reliability_score = reliability_score.max(0.0).min(100.0);

        // Weighted overall
        let overall = latency_score * self.latency_weight
            + depth_score * self.depth_weight
            + fee_score * self.fee_weight
            + reliability_score * self.reliability_weight;

        VenueScore {
            overall,
            latency_score,
            depth_score,
            fee_score,
            reliability_score,
            updated_ns: now_ns,
        }
    }

    fn update_sor_weights(&self, symbol: &str) {
        let mut venue_scores: Vec<(String, f64)> = Vec::new();

        // Collect all scores for this symbol
        for entry in self.scores.iter() {
            let key = entry.key();
            if key.ends_with(&format!(":{}", symbol)) {
                let parts: Vec<&str> = key.split(':').collect();
                if parts.len() == 2 {
                    venue_scores.push((parts[0].to_string(), entry.value().overall));
                }
            }
        }

        if venue_scores.is_empty() {
            return;
        }

        // Sort by score descending
        venue_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Calculate total score for normalization
        let total_score: f64 = venue_scores.iter().map(|(_, s)| s).sum();

        // Create weights
        let mut weights: Vec<SORWeight> = Vec::new();
        for (rank, (venue, score)) in venue_scores.into_iter().enumerate() {
            let weight = if total_score > 0.0 { score / total_score } else { 0.0 };
            weights.push(SORWeight {
                venue,
                weight,
                rank: rank + 1,
            });
        }

        self.sor_weights.insert(symbol.to_string(), weights);
    }

    /// Get SOR weights for a symbol
    pub fn get_sor_weights(&self, symbol: &str) -> Vec<SORWeight> {
        self.sor_weights.get(symbol).map(|w| w.clone()).unwrap_or_default()
    }

    /// Get best venue for a symbol
    pub fn get_best_venue(&self, symbol: &str) -> Option<String> {
        self.sor_weights.get(symbol)
            .and_then(|weights| weights.first().map(|w| w.venue.clone()))
    }

    /// Get venue score
    pub fn get_venue_score(&self, venue: &str, symbol: &str) -> Option<VenueScore> {
        let key = format!("{}:{}", venue, symbol);
        self.scores.get(&key).map(|s| s.clone())
    }

    /// Get venue metrics
    pub fn get_venue_metrics(&self, venue: &str, symbol: &str) -> Option<VenueMetrics> {
        let key = format!("{}:{}", venue, symbol);
        self.metrics.get(&key).map(|m| m.clone())
    }

    /// Record a successful execution
    pub fn record_success(&self, venue: &str, symbol: &str) {
        if let Some(mut metrics) = self.metrics.get_mut(&format!("{}:{}", venue, symbol)) {
            // Update success rate with exponential moving average
            metrics.success_rate = metrics.success_rate * 0.99 + 1.0 * 0.01;
            
            // Recalculate score
            let score = self.calculate_score(&format!("{}:{}", venue, symbol), &metrics);
            self.scores.insert(format!("{}:{}", venue, symbol), score);
        }
    }

    /// Record a failed execution
    pub fn record_failure(&self, venue: &str, symbol: &str) {
        if let Some(mut metrics) = self.metrics.get_mut(&format!("{}:{}", venue, symbol)) {
            metrics.success_rate = metrics.success_rate * 0.99;
            metrics.errors_last_minute += 1;
            
            let score = self.calculate_score(&format!("{}:{}", venue, symbol), &metrics);
            self.scores.insert(format!("{}:{}", venue, symbol), score);
        }
    }

    /// Deactivate scorer
    pub fn deactivate(&self) {
        self.is_active.store(false, Ordering::Relaxed);
    }

    /// Activate scorer
    pub fn activate(&self) {
        self.is_active.store(true, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_venue_scoring() {
        let scorer = VenueScorer::new(0.4, 0.3, 0.2, 0.1);

        let metrics_good = VenueMetrics {
            avg_latency_us: 100.0,
            latency_stddev_us: 20.0,
            best_bid_depth: 100_000.0,
            best_ask_depth: 100_000.0,
            depth_1pct: 1_000_000.0,
            maker_fee_bps: 10.0,
            taker_fee_bps: 10.0,
            success_rate: 0.99,
            errors_last_minute: 0,
            quote_rate: 1000.0,
        };

        let metrics_bad = VenueMetrics {
            avg_latency_us: 5000.0,
            latency_stddev_us: 500.0,
            best_bid_depth: 10_000.0,
            best_ask_depth: 10_000.0,
            depth_1pct: 100_000.0,
            maker_fee_bps: 50.0,
            taker_fee_bps: 50.0,
            success_rate: 0.80,
            errors_last_minute: 5,
            quote_rate: 100.0,
        };

        scorer.update_metrics("binance", "BTCUSDT", metrics_good);
        scorer.update_metrics("bybit", "BTCUSDT", metrics_bad);

        let binance_score = scorer.get_venue_score("binance", "BTCUSDT");
        let bybit_score = scorer.get_venue_score("bybit", "BTCUSDT");

        assert!(binance_score.is_some());
        assert!(bybit_score.is_some());

        let b_score = binance_score.unwrap();
        let y_score = bybit_score.unwrap();

        assert!(b_score.overall > y_score.overall, "Binance should score higher");
    }

    #[test]
    fn test_sor_weights() {
        let scorer = VenueScorer::new(0.4, 0.3, 0.2, 0.1);

        let metrics1 = VenueMetrics {
            avg_latency_us: 100.0,
            latency_stddev_us: 20.0,
            best_bid_depth: 100_000.0,
            best_ask_depth: 100_000.0,
            depth_1pct: 1_000_000.0,
            maker_fee_bps: 10.0,
            taker_fee_bps: 10.0,
            success_rate: 0.99,
            errors_last_minute: 0,
            quote_rate: 1000.0,
        };

        let metrics2 = VenueMetrics {
            avg_latency_us: 200.0,
            latency_stddev_us: 30.0,
            best_bid_depth: 80_000.0,
            best_ask_depth: 80_000.0,
            depth_1pct: 800_000.0,
            maker_fee_bps: 15.0,
            taker_fee_bps: 15.0,
            success_rate: 0.98,
            errors_last_minute: 0,
            quote_rate: 800.0,
        };

        scorer.update_metrics("venue1", "ETHUSDT", metrics1);
        scorer.update_metrics("venue2", "ETHUSDT", metrics2);

        let weights = scorer.get_sor_weights("ETHUSDT");
        assert_eq!(weights.len(), 2);
        assert!(weights[0].weight > weights[1].weight);
        assert_eq!(weights[0].rank, 1);
    }
}
