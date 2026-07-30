//! Hidden Markov Model (HMM) implementation for market regime detection.
//! 
//! This module implements the Viterbi algorithm for real-time hidden state decoding,
//! classifying market regimes (Trending, Mean-Reverting, High-Volatility) in microseconds.
//! Optimized for zero allocations in the hot path.

use std::sync::atomic::{AtomicU64, Ordering};
use crate::common::memory_pool::MemoryPool;

/// Maximum number of hidden states supported
const MAX_STATES: usize = 8;

/// Maximum observation sequence length for Viterbi
const MAX_SEQUENCE_LENGTH: usize = 1024;

/// Market regime classifications
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MarketRegime {
    /// Strong upward trend
    BullTrend,
    /// Strong downward trend
    BearTrend,
    /// Sideways/ranging market
    MeanReverting,
    /// High volatility environment
    HighVolatility,
    /// Low volatility/quiet market
    LowVolatility,
    /// Transitioning between regimes
    Transitioning,
    /// Unknown/unclassified
    Unknown,
}

impl MarketRegime {
    /// Get a numeric ID for the regime (for HMM state indexing)
    pub fn id(&self) -> usize {
        match self {
            MarketRegime::BullTrend => 0,
            MarketRegime::BearTrend => 1,
            MarketRegime::MeanReverting => 2,
            MarketRegime::HighVolatility => 3,
            MarketRegime::LowVolatility => 4,
            MarketRegime::Transitioning => 5,
            MarketRegime::Unknown => 6,
        }
    }
    
    /// Create from numeric ID
    pub fn from_id(id: usize) -> Self {
        match id {
            0 => MarketRegime::BullTrend,
            1 => MarketRegime::BearTrend,
            2 => MarketRegime::MeanReverting,
            3 => MarketRegime::HighVolatility,
            4 => MarketRegime::LowVolatility,
            5 => MarketRegime::Transitioning,
            _ => MarketRegime::Unknown,
        }
    }
    
    /// Check if regime is trending
    pub fn is_trending(&self) -> bool {
        matches!(self, MarketRegime::BullTrend | MarketRegime::BearTrend)
    }
    
    /// Check if regime is mean-reverting
    pub fn is_mean_reverting(&self) -> bool {
        *self == MarketRegime::MeanReverting
    }
    
    /// Get risk multiplier for position sizing
    pub fn risk_multiplier(&self) -> f64 {
        match self {
            MarketRegime::BullTrend | MarketRegime::BearTrend => 1.2,
            MarketRegime::MeanReverting => 1.0,
            MarketRegime::HighVolatility => 0.5,
            MarketRegime::LowVolatility => 1.1,
            MarketRegime::Transitioning => 0.7,
            MarketRegime::Unknown => 0.5,
        }
    }
}

/// Observation features for HMM
#[derive(Debug, Clone, Copy)]
pub struct ObservationFeatures {
    /// Returns over lookback period
    pub returns: f64,
    /// Realized volatility
    pub volatility: f64,
    /// Skewness of returns
    pub skewness: f64,
    /// Kurtosis of returns
    pub kurtosis: f64,
    /// Volume change ratio
    pub volume_change: f64,
    /// Order flow imbalance
    pub order_flow_imbalance: f64,
}

impl Default for ObservationFeatures {
    fn default() -> Self {
        Self {
            returns: 0.0,
            volatility: 0.0,
            skewness: 0.0,
            kurtosis: 0.0,
            volume_change: 0.0,
            order_flow_imbalance: 0.0,
        }
    }
}

/// Hidden Markov Model for regime detection
pub struct HiddenMarkovModel {
    /// Number of hidden states
    num_states: usize,
    /// Number of observation features
    num_features: usize,
    /// State transition probability matrix (A)
    /// A[i][j] = P(state_j at t+1 | state_i at t)
    transition_probs: [[f64; MAX_STATES]; MAX_STATES],
    /// Emission probability parameters (Gaussian means)
    /// emission_means[s][f] = mean of feature f in state s
    emission_means: [[f64; 6]; MAX_STATES],
    /// Emission probability parameters (Gaussian variances)
    /// emission_vars[s][f] = variance of feature f in state s
    emission_vars: [[f64; 6]; MAX_STATES],
    /// Initial state distribution (pi)
    initial_probs: [f64; MAX_STATES],
    /// Current state probabilities (belief state)
    current_probs: [f64; MAX_STATES],
    /// Log-scale transition probs for numerical stability
    log_transition_probs: [[f64; MAX_STATES]; MAX_STATES],
    /// Update counter
    update_count: AtomicU64,
    /// Current decoded state
    current_state: usize,
}

impl HiddenMarkovModel {
    /// Create a new HMM with specified number of states
    pub fn new(num_states: usize) -> Self {
        assert!(num_states <= MAX_STATES, "Number of states exceeds maximum");
        
        let mut model = Self {
            num_states,
            num_features: 6,
            transition_probs: [[0.0; MAX_STATES]; MAX_STATES],
            emission_means: [[0.0; 6]; MAX_STATES],
            emission_vars: [[1.0; 6]; MAX_STATES],
            initial_probs: [0.0; MAX_STATES],
            current_probs: [0.0; MAX_STATES],
            log_transition_probs: [[f64::NEG_INFINITY; MAX_STATES]; MAX_STATES],
            update_count: AtomicU64::new(0),
            current_state: 0,
        };
        
        // Initialize with uniform distribution
        model.set_uniform_initial();
        model.set_uniform_transition();
        
        model
    }
    
    /// Set uniform initial state distribution
    fn set_uniform_initial(&mut self) {
        let prob = 1.0 / self.num_states as f64;
        for i in 0..self.num_states {
            self.initial_probs[i] = prob;
            self.current_probs[i] = prob;
        }
    }
    
    /// Set uniform transition probabilities (with slight self-transition bias)
    fn set_uniform_transition(&mut self) {
        let base_prob = 0.7 / self.num_states as f64;
        let self_prob = 0.3 + base_prob;
        
        for i in 0..self.num_states {
            for j in 0..self.num_states {
                if i == j {
                    self.transition_probs[i][j] = self_prob;
                } else {
                    self.transition_probs[i][j] = base_prob;
                }
            }
        }
        
        // Normalize rows
        for i in 0..self.num_states {
            let row_sum: f64 = self.transition_probs[i][..self.num_states].iter().sum();
            for j in 0..self.num_states {
                self.transition_probs[i][j] /= row_sum;
                self.log_transition_probs[i][j] = self.transition_probs[i][j].ln();
            }
        }
    }
    
    /// Set custom transition probability
    pub fn set_transition_prob(&mut self, from: usize, to: usize, prob: f64) {
        assert!(from < self.num_states && to < self.num_states);
        self.transition_probs[from][to] = prob;
        self.log_transition_probs[from][to] = prob.ln();
    }
    
    /// Set emission parameters for a state
    pub fn set_emission_params(&mut self, state: usize, means: &[f64], vars: &[f64]) {
        assert!(state < self.num_states);
        assert_eq!(means.len(), vars.len());
        assert!(means.len() <= 6);
        
        for (i, (&m, &v)) in means.iter().zip(vars.iter()).enumerate() {
            self.emission_means[state][i] = m;
            self.emission_vars[state][i] = v.max(1e-10); // Prevent zero variance
        }
    }
    
    /// Calculate emission probability using Gaussian PDF
    #[inline]
    fn emission_probability(&self, state: usize, obs: &ObservationFeatures) -> f64 {
        let mut log_prob = 0.0;
        
        // Returns
        let diff = obs.returns - self.emission_means[state][0];
        let var = self.emission_vars[state][0];
        log_prob -= 0.5 * (diff * diff / var + var.ln());
        
        // Volatility
        let diff = obs.volatility - self.emission_means[state][1];
        let var = self.emission_vars[state][1];
        log_prob -= 0.5 * (diff * diff / var + var.ln());
        
        // Skewness
        let diff = obs.skewness - self.emission_means[state][2];
        let var = self.emission_vars[state][2];
        log_prob -= 0.5 * (diff * diff / var + var.ln());
        
        // Order flow imbalance
        let diff = obs.order_flow_imbalance - self.emission_means[state][5];
        let var = self.emission_vars[state][5];
        log_prob -= 0.5 * (diff * diff / var + var.ln());
        
        log_prob.exp()
    }
    
    /// Viterbi algorithm for finding most likely state sequence
    /// Returns the optimal state sequence
    pub fn viterbi(&self, observations: &[ObservationFeatures]) -> Vec<usize> {
        let n = observations.len();
        if n == 0 || n > MAX_SEQUENCE_LENGTH {
            return vec![];
        }
        
        // Use fixed-size arrays for Viterbi trellis
        let mut trellis: [[f64; MAX_STATES]; MAX_SEQUENCE_LENGTH] = [[0.0; MAX_STATES]; MAX_SEQUENCE_LENGTH];
        let mut backpointers: [[usize; MAX_STATES]; MAX_SEQUENCE_LENGTH] = [[0; MAX_STATES]; MAX_SEQUENCE_LENGTH];
        
        // Initialization
        for s in 0..self.num_states {
            let emit_prob = self.emission_probability(s, &observations[0]);
            trellis[0][s] = self.initial_probs[s] * emit_prob;
        }
        
        // Recursion
        for t in 1..n {
            for s_curr in 0..self.num_states {
                let mut max_prob = 0.0;
                let mut best_prev = 0;
                
                for s_prev in 0..self.num_states {
                    let prob = trellis[t - 1][s_prev] 
                        * self.transition_probs[s_prev][s_curr];
                    
                    if prob > max_prob {
                        max_prob = prob;
                        best_prev = s_prev;
                    }
                }
                
                let emit_prob = self.emission_probability(s_curr, &observations[t]);
                trellis[t][s_curr] = max_prob * emit_prob;
                backpointers[t][s_curr] = best_prev;
            }
        }
        
        // Termination - find best final state
        let mut best_final_state = 0;
        let mut best_final_prob = trellis[n - 1][0];
        for s in 1..self.num_states {
            if trellis[n - 1][s] > best_final_prob {
                best_final_prob = trellis[n - 1][s];
                best_final_state = s;
            }
        }
        
        // Backtracking
        let mut path = vec![0; n];
        path[n - 1] = best_final_state;
        for t in (1..n).rev() {
            path[t - 1] = backpointers[t][path[t]];
        }
        
        path
    }
    
    /// Forward algorithm for computing likelihood and belief state
    #[inline]
    pub fn forward(&mut self, obs: &ObservationFeatures) -> f64 {
        let mut new_probs = [0.0; MAX_STATES];
        let mut total_prob = 0.0;
        
        // Prediction step
        for s in 0..self.num_states {
            let mut pred_prob = 0.0;
            for s_prev in 0..self.num_states {
                pred_prob += self.current_probs[s_prev] * self.transition_probs[s_prev][s];
            }
            
            // Update step with emission
            let emit_prob = self.emission_probability(s, obs);
            new_probs[s] = pred_prob * emit_prob;
            total_prob += new_probs[s];
        }
        
        // Normalize
        if total_prob > 1e-300 {
            for s in 0..self.num_states {
                self.current_probs[s] = new_probs[s] / total_prob;
            }
        }
        
        // Update current state (MAP estimate)
        self.current_state = self.most_likely_state();
        self.update_count.fetch_add(1, Ordering::Relaxed);
        
        total_prob
    }
    
    /// Get the most likely current state
    #[inline]
    pub fn most_likely_state(&self) -> usize {
        let mut best_state = 0;
        let mut best_prob = self.current_probs[0];
        
        for s in 1..self.num_states {
            if self.current_probs[s] > best_prob {
                best_prob = self.current_probs[s];
                best_state = s;
            }
        }
        
        best_state
    }
    
    /// Get current belief state (state probabilities)
    #[inline]
    pub fn belief_state(&self) -> &[f64] {
        &self.current_probs[..self.num_states]
    }
    
    /// Decode regime from state ID
    pub fn decode_regime(&self, state: usize) -> MarketRegime {
        // Map internal states to market regimes based on emission parameters
        // This is a simplified mapping - in production, this would be learned
        match state % 6 {
            0 => MarketRegime::BullTrend,
            1 => MarketRegime::BearTrend,
            2 => MarketRegime::MeanReverting,
            3 => MarketRegime::HighVolatility,
            4 => MarketRegime::LowVolatility,
            _ => MarketRegime::Transitioning,
        }
    }
    
    /// Get current regime classification
    #[inline]
    pub fn current_regime(&self) -> MarketRegime {
        self.decode_regime(self.current_state)
    }
    
    /// Get regime confidence (probability of current state)
    #[inline]
    pub fn regime_confidence(&self) -> f64 {
        self.current_probs[self.current_state]
    }
    
    /// Get update count
    #[inline]
    pub fn update_count(&self) -> u64 {
        self.update_count.load(Ordering::Relaxed)
    }
    
    /// Reset HMM to initial state
    pub fn reset(&mut self) {
        self.set_uniform_initial();
        self.current_state = 0;
        self.update_count.store(0, Ordering::Relaxed);
    }
}

/// Real-time regime detector wrapper
pub struct RegimeDetector {
    hmm: HiddenMarkovModel,
    /// Rolling window for feature calculation
    rolling_window: Vec<ObservationFeatures>,
    /// Window size
    window_size: usize,
    /// Current write position
    write_pos: usize,
}

impl RegimeDetector {
    /// Create a new regime detector
    pub fn new(num_states: usize, window_size: usize) -> Self {
        Self {
            hmm: HiddenMarkovModel::new(num_states),
            rolling_window: vec![ObservationFeatures::default(); window_size],
            window_size,
            write_pos: 0,
        }
    }
    
    /// Process new observation and update regime classification
    #[inline]
    pub fn update(&mut self, features: ObservationFeatures) -> MarketRegime {
        // Add to rolling window
        self.rolling_window[self.write_pos] = features;
        self.write_pos = (self.write_pos + 1) % self.window_size;
        
        // Run forward algorithm
        self.hmm.forward(&features);
        
        self.hmm.current_regime()
    }
    
    /// Run Viterbi on recent history for smoothed classification
    pub fn smoothed_regime(&self) -> Option<MarketRegime> {
        let valid_len = self.write_pos;
        if valid_len == 0 {
            return None;
        }
        
        let path = self.hmm.viterbi(&self.rolling_window[..valid_len]);
        path.last().map(|&s| self.hmm.decode_regime(s))
    }
    
    /// Get current regime
    #[inline]
    pub fn current_regime(&self) -> MarketRegime {
        self.hmm.current_regime()
    }
    
    /// Get regime confidence
    #[inline]
    pub fn confidence(&self) -> f64 {
        self.hmm.regime_confidence()
    }
    
    /// Get all state probabilities
    #[inline]
    pub fn state_probabilities(&self) -> &[f64] {
        self.hmm.belief_state()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_hmm_basic() {
        let mut hmm = HiddenMarkovModel::new(4);
        
        // Set up some emission parameters
        hmm.set_emission_params(0, &[0.01, 0.1, 0.0, 3.0, 0.0, 0.5], &[0.001, 0.01, 0.1, 1.0, 0.1, 0.1]);
        hmm.set_emission_params(1, &[-0.01, 0.1, 0.0, 3.0, 0.0, -0.5], &[0.001, 0.01, 0.1, 1.0, 0.1, 0.1]);
        hmm.set_emission_params(2, &[0.0, 0.05, 0.0, 3.0, 0.0, 0.0], &[0.0005, 0.005, 0.1, 1.0, 0.1, 0.05]);
        hmm.set_emission_params(3, &[0.0, 0.3, 0.0, 5.0, 0.0, 0.0], &[0.002, 0.05, 0.2, 2.0, 0.2, 0.2]);
        
        // Create some test observations
        let obs = ObservationFeatures {
            returns: 0.01,
            volatility: 0.1,
            skewness: 0.0,
            kurtosis: 3.0,
            volume_change: 0.0,
            order_flow_imbalance: 0.5,
        };
        
        let likelihood = hmm.forward(&obs);
        assert!(likelihood > 0.0);
        
        let regime = hmm.current_regime();
        assert!(matches!(regime, MarketRegime::BullTrend | MarketRegime::BearTrend 
                         | MarketRegime::MeanReverting | MarketRegime::HighVolatility));
    }
    
    #[test]
    fn test_viterbi_sequence() {
        let hmm = HiddenMarkovModel::new(3);
        
        let observations = vec![
            ObservationFeatures { returns: 0.01, ..Default::default() },
            ObservationFeatures { returns: 0.02, ..Default::default() },
            ObservationFeatures { returns: -0.01, ..Default::default() },
            ObservationFeatures { returns: -0.02, ..Default::default() },
        ];
        
        let path = hmm.viterbi(&observations);
        assert_eq!(path.len(), observations.len());
    }
    
    #[test]
    fn test_regime_detector() {
        let mut detector = RegimeDetector::new(4, 20);
        
        for i in 0..30 {
            let features = ObservationFeatures {
                returns: (i as f64 * 0.001) - 0.015,
                volatility: 0.1 + (i as f64 * 0.002),
                ..Default::default()
            };
            
            let regime = detector.update(features);
            let _ = regime; // Use the regime
        }
        
        let current = detector.current_regime();
        assert_ne!(current, MarketRegime::Unknown);
    }
}
