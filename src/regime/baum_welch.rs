//! Online Baum-Welch algorithm for continuous HMM parameter updates.
//! 
//! This module implements an incremental, online version of the Baum-Welch algorithm
//! that continuously updates HMM transition and emission probabilities without requiring
//! batch retraining. Allows the Rust core to adapt to shifting market dynamics in real-time.

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use crate::regime::hmm::{HiddenMarkovModel, ObservationFeatures, MAX_STATES};

/// Configuration for online Baum-Welch learning
#[derive(Debug, Clone)]
pub struct BaumWelchConfig {
    /// Learning rate for parameter updates (0.0 to 1.0)
    pub learning_rate: f64,
    /// Minimum number of observations before starting updates
    pub warmup_samples: usize,
    /// Update frequency (update every N observations)
    pub update_frequency: usize,
    /// Decay factor for older observations
    pub decay_factor: f64,
    /// Minimum probability threshold (prevent zero probabilities)
    pub min_probability: f64,
}

impl Default for BaumWelchConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            warmup_samples: 100,
            update_frequency: 50,
            decay_factor: 0.99,
            min_probability: 1e-10,
        }
    }
}

/// Accumulator for sufficient statistics
#[derive(Debug, Clone)]
struct SufficientStatistics {
    /// Expected count of transitions from state i to state j
    transition_counts: [[f64; MAX_STATES]; MAX_STATES],
    /// Expected count of being in each state
    state_counts: [f64; MAX_STATES],
    /// Weighted sum of observations for each state/feature
    observation_sums: [[f64; 6]; MAX_STATES],
    /// Weighted sum of squared observations for each state/feature
    observation_sq_sums: [[f64; 6]; MAX_STATES],
    /// Total weight accumulated
    total_weight: f64,
}

impl SufficientStatistics {
    fn new() -> Self {
        Self {
            transition_counts: [[0.0; MAX_STATES]; MAX_STATES],
            state_counts: [0.0; MAX_STATES],
            observation_sums: [[0.0; 6]; MAX_STATES],
            observation_sq_sums: [[0.0; 6]; MAX_STATES],
            total_weight: 0.0,
        }
    }
    
    /// Reset all accumulators
    fn reset(&mut self) {
        self.transition_counts = [[0.0; MAX_STATES]; MAX_STATES];
        self.state_counts = [0.0; MAX_STATES];
        self.observation_sums = [[0.0; 6]; MAX_STATES];
        self.observation_sq_sums = [[0.0; 6]; MAX_STATES];
        self.total_weight = 0.0;
    }
}

/// Online Baum-Welch learner for adaptive HMM parameter estimation
pub struct OnlineBaumWelch {
    /// Reference to the HMM being trained
    hmm: HiddenMarkovModel,
    /// Learning configuration
    config: BaumWelchConfig,
    /// Sufficient statistics accumulator
    stats: SufficientStatistics,
    /// Number of observations processed
    observation_count: AtomicU64,
    /// Number of parameter updates performed
    update_count: AtomicU64,
    /// Whether learning is enabled
    learning_enabled: AtomicBool,
    /// Log-likelihood history for convergence monitoring
    log_likelihood_window: Vec<f64>,
    /// Window size for convergence check
    ll_window_size: usize,
    /// Previous belief state (for transition counting)
    prev_belief: [f64; MAX_STATES],
}

impl OnlineBaumWelch {
    /// Create a new online Baum-Welch learner
    pub fn new(hmm: HiddenMarkovModel, config: BaumWelchConfig) -> Self {
        Self {
            hmm,
            config,
            stats: SufficientStatistics::new(),
            observation_count: AtomicU64::new(0),
            update_count: AtomicU64::new(0),
            learning_enabled: AtomicBool::new(true),
            log_likelihood_window: Vec::with_capacity(100),
            ll_window_size: 50,
            prev_belief: [0.0; MAX_STATES],
        }
    }
    
    /// Process a new observation and optionally update parameters
    #[inline]
    pub fn observe(&mut self, obs: &ObservationFeatures) -> f64 {
        let count = self.observation_count.fetch_add(1, Ordering::Relaxed);
        
        // Run forward algorithm to get likelihood and update belief
        let likelihood = self.hmm.forward(obs);
        let log_likelihood = likelihood.ln().max(-1000.0); // Clamp for numerical stability
        
        // Get current belief state
        let belief = self.hmm.belief_state();
        let num_states = belief.len();
        
        // During warmup, just collect statistics
        if count < self.config.warmup_samples as u64 {
            self.accumulate_statistics(obs, belief);
            return log_likelihood;
        }
        
        // Accumulate statistics
        self.accumulate_statistics(obs, belief);
        
        // Periodically update parameters
        if count % self.config.update_frequency as u64 == 0 && self.learning_enabled.load(Ordering::Relaxed) {
            self.update_parameters();
        }
        
        // Store log-likelihood for convergence monitoring
        self.log_likelihood_window.push(log_likelihood);
        if self.log_likelihood_window.len() > self.ll_window_size {
            self.log_likelihood_window.remove(0);
        }
        
        log_likelihood
    }
    
    /// Accumulate sufficient statistics from current observation
    fn accumulate_statistics(&mut self, obs: &ObservationFeatures, belief: &[f64]) {
        let weight = self.config.decay_factor.powi(
            (self.observation_count.load(Ordering::Relaxed) / self.config.update_frequency as u64) as i32
        );
        
        // Update state counts
        for s in 0..belief.len() {
            self.stats.state_counts[s] += belief[s] * weight;
            
            // Update observation statistics for each feature
            let features = [obs.returns, obs.volatility, obs.skewness, obs.kurtosis, obs.volume_change, obs.order_flow_imbalance];
            for f in 0..6.min(features.len()) {
                self.stats.observation_sums[s][f] += features[f] * belief[s] * weight;
                self.stats.observation_sq_sums[s][f] += features[f].powi(2) * belief[s] * weight;
            }
        }
        
        // Update transition counts using previous and current belief
        for i in 0..belief.len() {
            for j in 0..belief.len() {
                let trans_prob = self.hmm.transition_probs[i][j];
                self.stats.transition_counts[i][j] += self.prev_belief[i] * trans_prob * belief[j] * weight;
            }
        }
        
        self.stats.total_weight += weight;
        
        // Store current belief for next iteration
        self.prev_belief.copy_from_slice(belief);
    }
    
    /// Update HMM parameters based on accumulated statistics
    pub fn update_parameters(&mut self) {
        if self.stats.total_weight < 1.0 {
            return;
        }
        
        let lr = self.config.learning_rate;
        let min_prob = self.config.min_probability;
        
        // Update initial probabilities
        let mut new_initial = [0.0; MAX_STATES];
        for s in 0..self.hmm.num_states {
            new_initial[s] = (self.stats.state_counts[s] / self.stats.total_weight).max(min_prob);
        }
        // Normalize
        let sum: f64 = new_initial.iter().sum();
        for s in 0..self.hmm.num_states {
            let old = self.hmm.initial_probs[s];
            self.hmm.initial_probs[s] = old + lr * (new_initial[s] / sum - old);
        }
        
        // Update transition probabilities
        for i in 0..self.hmm.num_states {
            let row_sum: f64 = self.stats.transition_counts[i][..self.hmm.num_states].iter().sum();
            if row_sum < min_prob {
                continue;
            }
            
            for j in 0..self.hmm.num_states {
                let new_prob = (self.stats.transition_counts[i][j] / row_sum).max(min_prob);
                let old = self.hmm.transition_probs[i][j];
                self.hmm.transition_probs[i][j] = old + lr * (new_prob - old);
                self.hmm.log_transition_probs[i][j] = self.hmm.transition_probs[i][j].ln();
            }
            
            // Re-normalize row
            let row_sum: f64 = self.hmm.transition_probs[i][..self.hmm.num_states].iter().sum();
            for j in 0..self.hmm.num_states {
                self.hmm.transition_probs[i][j] /= row_sum;
                self.hmm.log_transition_probs[i][j] = self.hmm.transition_probs[i][j].ln();
            }
        }
        
        // Update emission parameters (Gaussian means and variances)
        for s in 0..self.hmm.num_states {
            let state_count = self.stats.state_counts[s].max(min_prob);
            
            for f in 0..6 {
                // Update mean
                let new_mean = self.stats.observation_sums[s][f] / state_count;
                let old_mean = self.hmm.emission_means[s][f];
                self.hmm.emission_means[s][f] = old_mean + lr * (new_mean - old_mean);
                
                // Update variance using E[X^2] - E[X]^2
                let e_x2 = self.stats.observation_sq_sums[s][f] / state_count;
                let e_x = self.hmm.emission_means[s][f];
                let new_var = (e_x2 - e_x * e_x).max(min_prob);
                let old_var = self.hmm.emission_vars[s][f];
                self.hmm.emission_vars[s][f] = old_var + lr * (new_var - old_var);
            }
        }
        
        self.update_count.fetch_add(1, Ordering::Relaxed);
        
        // Optionally decay the learning rate over time
        // self.config.learning_rate *= 0.999;
    }
    
    /// Check if the model has converged based on log-likelihood stability
    pub fn check_convergence(&self) -> bool {
        if self.log_likelihood_window.len() < self.ll_window_size {
            return false;
        }
        
        let recent: Vec<f64> = self.log_likelihood_window.iter()
            .rev()
            .take(self.ll_window_size / 2)
            .copied()
            .collect();
        let older: Vec<f64> = self.log_likelihood_window.iter()
            .rev()
            .skip(self.ll_window_size / 2)
            .take(self.ll_window_size / 2)
            .copied()
            .collect();
        
        let recent_avg: f64 = recent.iter().sum::<f64>() / recent.len() as f64;
        let older_avg: f64 = older.iter().sum::<f64>() / older.len() as f64;
        
        // Check if change is small relative to magnitude
        let rel_change = (recent_avg - older_avg).abs() / (older_avg.abs() + 1.0);
        rel_change < 0.001 // 0.1% relative change threshold
    }
    
    /// Enable/disable online learning
    pub fn set_learning_enabled(&self, enabled: bool) {
        self.learning_enabled.store(enabled, Ordering::Relaxed);
    }
    
    /// Check if learning is enabled
    pub fn is_learning_enabled(&self) -> bool {
        self.learning_enabled.load(Ordering::Relaxed)
    }
    
    /// Get total observation count
    pub fn observation_count(&self) -> u64 {
        self.observation_count.load(Ordering::Relaxed)
    }
    
    /// Get total update count
    pub fn update_count(&self) -> u64 {
        self.update_count.load(Ordering::Relaxed)
    }
    
    /// Get current log-likelihood
    pub fn current_log_likelihood(&self) -> Option<f64> {
        self.log_likelihood_window.last().copied()
    }
    
    /// Get average log-likelihood over window
    pub fn average_log_likelihood(&self) -> Option<f64> {
        if self.log_likelihood_window.is_empty() {
            return None;
        }
        let sum: f64 = self.log_likelihood_window.iter().sum();
        Some(sum / self.log_likelihood_window.len() as f64)
    }
    
    /// Reset learner state (keep HMM parameters)
    pub fn reset(&mut self) {
        self.stats.reset();
        self.observation_count.store(0, Ordering::Relaxed);
        self.log_likelihood_window.clear();
        self.prev_belief = [0.0; MAX_STATES];
    }
    
    /// Get reference to underlying HMM
    pub fn hmm(&self) -> &HiddenMarkovModel {
        &self.hmm
    }
    
    /// Get mutable reference to underlying HMM
    pub fn hmm_mut(&mut self) -> &mut HiddenMarkovModel {
        &mut self.hmm
    }
}

/// Adaptive regime detector with online learning
pub struct AdaptiveRegimeDetector {
    learner: OnlineBaumWelch,
    /// Current regime
    current_regime: crate::regime::hmm::MarketRegime,
}

impl AdaptiveRegimeDetector {
    /// Create a new adaptive regime detector
    pub fn new(num_states: usize, config: BaumWelchConfig) -> Self {
        let hmm = HiddenMarkovModel::new(num_states);
        let learner = OnlineBaumWelch::new(hmm, config);
        
        Self {
            learner,
            current_regime: crate::regime::hmm::MarketRegime::Unknown,
        }
    }
    
    /// Process observation and update regime classification
    #[inline]
    pub fn update(&mut self, obs: &ObservationFeatures) -> crate::regime::hmm::MarketRegime {
        let _log_ll = self.learner.observe(obs);
        self.current_regime = self.learner.hmm().current_regime();
        self.current_regime
    }
    
    /// Get current regime
    pub fn current_regime(&self) -> crate::regime::hmm::MarketRegime {
        self.current_regime
    }
    
    /// Check if model has converged
    pub fn has_converged(&self) -> bool {
        self.learner.check_convergence()
    }
    
    /// Get convergence status and statistics
    pub fn training_status(&self) -> TrainingStatus {
        TrainingStatus {
            observations: self.learner.observation_count(),
            updates: self.learner.update_count(),
            converged: self.learner.check_convergence(),
            avg_log_likelihood: self.learner.average_log_likelihood(),
            learning_enabled: self.learner.is_learning_enabled(),
        }
    }
}

/// Training status information
#[derive(Debug, Clone)]
pub struct TrainingStatus {
    pub observations: u64,
    pub updates: u64,
    pub converged: bool,
    pub avg_log_likelihood: Option<f64>,
    pub learning_enabled: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_online_learning() {
        let hmm = HiddenMarkovModel::new(3);
        let config = BaumWelchConfig {
            warmup_samples: 10,
            update_frequency: 5,
            ..Default::default()
        };
        
        let mut learner = OnlineBaumWelch::new(hmm, config);
        
        // Generate some synthetic observations
        for i in 0..50 {
            let obs = ObservationFeatures {
                returns: (i as f64 * 0.001) - 0.025,
                volatility: 0.1 + (i as f64 * 0.001),
                ..Default::default()
            };
            
            let _ll = learner.observe(&obs);
        }
        
        // Should have performed at least one update
        assert!(learner.update_count() > 0);
        
        // Log-likelihood should be finite
        let avg_ll = learner.average_log_likelihood();
        assert!(avg_ll.is_some());
        assert!(avg_ll.unwrap().is_finite());
    }
    
    #[test]
    fn test_adaptive_detector() {
        let config = BaumWelchConfig {
            warmup_samples: 5,
            update_frequency: 3,
            ..Default::default()
        };
        
        let mut detector = AdaptiveRegimeDetector::new(4, config);
        
        for i in 0..30 {
            let obs = ObservationFeatures {
                returns: (i as f64 * 0.002) - 0.03,
                volatility: 0.15,
                ..Default::default()
            };
            
            let _regime = detector.update(&obs);
        }
        
        let status = detector.training_status();
        assert!(status.observations > 0);
        assert_ne!(detector.current_regime(), crate::regime::hmm::MarketRegime::Unknown);
    }
}
