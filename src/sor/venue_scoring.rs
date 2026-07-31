//! Dynamic Venue Scoring Engine for Smart Order Routing
//! 
//! Evaluates latency, depth, and fee structures in real-time.
//! Continuously re-weights routing probabilities to favor venues
//! with the highest current reliability and lowest adverse selection.

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// Maximum number of venues supported
const MAX_VENUES: usize = 32;

/// Weight factors for venue scoring
#[derive(Debug, Clone, Copy)]
pub struct VenueWeights {
    pub latency_weight: f64,
    pub depth_weight: f64,
    pub fee_weight: f64,
    pub reliability_weight: f64,
    pub fill_rate_weight: f64,
}

impl Default for VenueWeights {
    fn default() -> Self {
        Self {
            latency_weight: 0.25,
            depth_weight: 0.25,
            fee_weight: 0.20,
            reliability_weight: 0.15,
            fill_rate_weight: 0.15,
        }
    }
}

/// Venue metrics snapshot
#[derive(Debug, Clone)]
pub struct VenueMetrics {
    pub venue_id: u32,
    pub name: String,
    /// Average round-trip latency in microseconds
    pub avg_latency_us: u64,
    /// Best bid depth in base currency
    pub bid_depth: f64,
    /// Best ask depth in base currency
    pub ask_depth: f64,
    /// Maker fee in basis points (negative = rebate)
    pub maker_fee_bps: f64,
    /// Taker fee in basis points
    pub taker_fee_bps: f64,
    /// Fill rate (0.0 to 1.0)
    pub fill_rate: f64,
    /// Reliability score (0.0 to 1.0)
    pub reliability: f64,
    /// Adverse selection estimate (0.0 to 1.0, lower is better)
    pub adverse_selection: f64,
    /// Last update timestamp
    pub last_update: Instant,
}

impl VenueMetrics {
    pub fn new(venue_id: u32, name: &str) -> Self {
        Self {
            venue_id,
            name: name.to_string(),
            avg_latency_us: 0,
            bid_depth: 0.0,
            ask_depth: 0.0,
            maker_fee_bps: 0.0,
            taker_fee_bps: 0.0,
            fill_rate: 1.0,
            reliability: 1.0,
            adverse_selection: 0.0,
            last_update: Instant::now(),
        }
    }
}

/// Computed venue score
#[derive(Debug, Clone)]
pub struct VenueScore {
    pub venue_id: u32,
    pub total_score: f64,
    pub normalized_score: f64, // 0.0 to 1.0
    pub routing_probability: f64, // 0.0 to 1.0
    pub latency_component: f64,
    pub depth_component: f64,
    pub fee_component: f64,
    pub reliability_component: f64,
    pub fill_rate_component: f64,
}

/// Venue scoring engine with atomic state
pub struct VenueScoringEngine {
    venues: [Option<VenueMetrics>; MAX_VENUES],
    scores: [Option<VenueScore>; MAX_VENUES],
    weights: VenueWeights,
    active_count: AtomicU64,
    last_recalc: AtomicU64,
    recalc_interval_ms: u64,
}

impl VenueScoringEngine {
    pub fn new(weights: VenueWeights) -> Self {
        Self {
            venues: Default::default(),
            scores: Default::default(),
            weights,
            active_count: AtomicU64::new(0),
            last_recalc: AtomicU64::new(0),
            recalc_interval_ms: 100, // Recalculate every 100ms
        }
    }
    
    /// Register or update a venue
    pub fn update_venue(&mut self, metrics: VenueMetrics) {
        let idx = metrics.venue_id as usize;
        if idx < MAX_VENUES {
            let is_new = self.venues[idx].is_none();
            self.venues[idx] = Some(metrics);
            
            if is_new {
                self.active_count.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
    
    /// Update latency measurement for a venue
    pub fn update_latency(&mut self, venue_id: u32, latency_us: u64) {
        let idx = venue_id as usize;
        if idx < MAX_VENUES {
            if let Some(ref mut venue) = self.venues[idx] {
                // Exponential moving average for latency
                let alpha = 0.3;
                venue.avg_latency_us = ((1.0 - alpha) * venue.avg_latency_us as f64 
                    + alpha * latency_us as f64) as u64;
                venue.last_update = Instant::now();
            }
        }
    }
    
    /// Update depth information for a venue
    pub fn update_depth(&mut self, venue_id: u32, bid_depth: f64, ask_depth: f64) {
        let idx = venue_id as usize;
        if idx < MAX_VENUES {
            if let Some(ref mut venue) = self.venues[idx] {
                venue.bid_depth = bid_depth;
                venue.ask_depth = ask_depth;
                venue.last_update = Instant::now();
            }
        }
    }
    
    /// Update fill rate for a venue
    pub fn update_fill_rate(&mut self, venue_id: u32, fill_rate: f64) {
        let idx = venue_id as usize;
        if idx < MAX_VENUES {
            if let Some(ref mut venue) = self.venues[idx] {
                let alpha = 0.2;
                venue.fill_rate = (1.0 - alpha) * venue.fill_rate + alpha * fill_rate.min(1.0);
                venue.last_update = Instant::now();
            }
        }
    }
    
    /// Compute latency component score (lower latency = higher score)
    fn compute_latency_score(&self, latency_us: u64) -> f64 {
        const MIN_LATENCY: f64 = 50.0; // Best case ~50us
        const MAX_LATENCY: f64 = 5000.0; // Worst case ~5ms
        
        let latency = latency_us as f64;
        if latency <= MIN_LATENCY {
            1.0
        } else if latency >= MAX_LATENCY {
            0.0
        } else {
            1.0 - (latency - MIN_LATENCY) / (MAX_LATENCY - MIN_LATENCY)
        }
    }
    
    /// Compute depth component score (higher depth = higher score)
    fn compute_depth_score(&self, bid_depth: f64, ask_depth: f64) -> f64 {
        let total_depth = bid_depth + ask_depth;
        const MIN_DEPTH: f64 = 1000.0; // 1k units
        const MAX_DEPTH: f64 = 1_000_000.0; // 1M units
        
        if total_depth <= MIN_DEPTH {
            0.0
        } else if total_depth >= MAX_DEPTH {
            1.0
        } else {
            (total_depth - MIN_DEPTH) / (MAX_DEPTH - MIN_DEPTH)
        }
    }
    
    /// Compute fee component score (lower fees = higher score)
    fn compute_fee_score(&self, maker_fee_bps: f64, taker_fee_bps: f64, is_maker: bool) -> f64 {
        let fee = if is_maker { maker_fee_bps } else { taker_fee_bps };
        
        // Negative fees (rebates) get best score
        if fee <= -10.0 {
            1.0
        } else if fee >= 50.0 {
            0.0
        } else {
            1.0 - (fee + 10.0) / 60.0
        }
    }
    
    /// Calculate scores for all venues
    pub fn recalculate_scores(&mut self, is_maker: bool) {
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        
        // Check if recalculation is needed
        let last = self.last_recalc.load(Ordering::Relaxed);
        if now_ms - last < self.recalc_interval_ms {
            return;
        }
        
        self.last_recalc.store(now_ms, Ordering::Relaxed);
        
        let mut max_score = 0.0;
        let mut scores_sum = 0.0;
        
        // First pass: compute raw scores
        for i in 0..MAX_VENUES {
            if let Some(ref venue) = self.venues[i] {
                let latency_score = self.compute_latency_score(venue.avg_latency_us);
                let depth_score = self.compute_depth_score(venue.bid_depth, venue.ask_depth);
                let fee_score = self.compute_fee_score(venue.maker_fee_bps, venue.taker_fee_bps, is_maker);
                let reliability_score = venue.reliability;
                let fill_rate_score = venue.fill_rate;
                
                let total_score = 
                    latency_score * self.weights.latency_weight +
                    depth_score * self.weights.depth_weight +
                    fee_score * self.weights.fee_weight +
                    reliability_score * self.weights.reliability_weight +
                    fill_rate_score * self.weights.fill_rate_weight;
                
                // Apply adverse selection penalty
                let adjusted_score = total_score * (1.0 - venue.adverse_selection * 0.5);
                
                self.scores[i] = Some(VenueScore {
                    venue_id: venue.venue_id,
                    total_score: adjusted_score,
                    normalized_score: 0.0, // Will be set in second pass
                    routing_probability: 0.0, // Will be set in second pass
                    latency_component: latency_score,
                    depth_component: depth_score,
                    fee_component: fee_score,
                    reliability_component: reliability_score,
                    fill_rate_component: fill_rate_score,
                });
                
                max_score = max_score.max(adjusted_score);
                scores_sum += adjusted_score;
            } else {
                self.scores[i] = None;
            }
        }
        
        // Second pass: normalize scores and compute routing probabilities
        for i in 0..MAX_VENUES {
            if let Some(ref mut score) = self.scores[i] {
                if max_score > 0.0 {
                    score.normalized_score = score.total_score / max_score;
                }
                
                if scores_sum > 0.0 {
                    score.routing_probability = score.total_score / scores_sum;
                }
            }
        }
    }
    
    /// Get top N venues by score
    pub fn get_top_venues(&self, n: usize) -> Vec<VenueScore> {
        let mut valid_scores: Vec<VenueScore> = self.scores
            .iter()
            .filter_map(|s| s.clone())
            .collect();
        
        valid_scores.sort_by(|a, b| b.total_score.partial_cmp(&a.total_score).unwrap_or(std::cmp::Ordering::Equal));
        valid_scores.truncate(n);
        valid_scores
    }
    
    /// Select venue based on routing probabilities (weighted random)
    pub fn select_venue(&self, seed: u64) -> Option<u32> {
        let r = (seed % 10000) as f64 / 10000.0;
        let mut cumulative = 0.0;
        
        for i in 0..MAX_VENUES {
            if let Some(ref score) = self.scores[i] {
                cumulative += score.routing_probability;
                if r <= cumulative {
                    return Some(score.venue_id);
                }
            }
        }
        
        // Fallback to best venue
        self.get_top_venues(1).first().map(|s| s.venue_id)
    }
    
    /// Get venue by ID
    pub fn get_venue(&self, venue_id: u32) -> Option<&VenueMetrics> {
        let idx = venue_id as usize;
        if idx < MAX_VENUES {
            self.venues[idx].as_ref()
        } else {
            None
        }
    }
    
    /// Get active venue count
    pub fn active_count(&self) -> usize {
        self.active_count.load(Ordering::Relaxed) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_venue_scoring() {
        let mut engine = VenueScoringEngine::new(VenueWeights::default());
        
        // Add venues with different characteristics
        let mut venue1 = VenueMetrics::new(0, "Binance");
        venue1.avg_latency_us = 100;
        venue1.bid_depth = 100000.0;
        venue1.ask_depth = 100000.0;
        venue1.maker_fee_bps = -5.0; // Rebate
        venue1.fill_rate = 0.95;
        engine.update_venue(venue1);
        
        let mut venue2 = VenueMetrics::new(1, "Coinbase");
        venue2.avg_latency_us = 500;
        venue2.bid_depth = 50000.0;
        venue2.ask_depth = 50000.0;
        venue2.maker_fee_bps = 10.0;
        venue2.fill_rate = 0.85;
        engine.update_venue(venue2);
        
        engine.recalculate_scores(true);
        
        let top = engine.get_top_venues(2);
        assert_eq!(top.len(), 2);
        assert!(top[0].total_score > top[1].total_score);
    }
    
    #[test]
    fn test_venue_selection() {
        let mut engine = VenueScoringEngine::new(VenueWeights::default());
        
        for i in 0..5 {
            let mut venue = VenueMetrics::new(i, &format!("Venue{}", i));
            venue.avg_latency_us = 100 + i * 100;
            venue.bid_depth = 100000.0 - i * 10000.0;
            venue.ask_depth = venue.bid_depth;
            engine.update_venue(venue);
        }
        
        engine.recalculate_scores(true);
        
        // Test deterministic selection with same seed
        let selected1 = engine.select_venue(42);
        let selected2 = engine.select_venue(42);
        assert_eq!(selected1, selected2);
    }
}
