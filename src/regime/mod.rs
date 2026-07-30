//! Regime detection module root.
//! 
//! Provides Hidden Markov Model (HMM) based market regime classification:
//! - Viterbi algorithm for real-time state decoding
//! - Online Baum-Welch for adaptive parameter learning
//! - Integration with strategy routing system

pub mod hmm;
pub mod baum_welch;

pub use hmm::{
    HiddenMarkovModel, MarketRegime, ObservationFeatures, 
    RegimeDetector, MAX_STATES,
};
pub use baum_welch::{
    OnlineBaumWelch, BaumWelchConfig, AdaptiveRegimeDetector, TrainingStatus,
};

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use crate::strategy::core::StrategyId;

/// Regime-aware strategy router configuration
#[derive(Debug, Clone)]
pub struct RegimeRouterConfig {
    /// Minimum confidence threshold for regime-based routing
    pub min_confidence: f64,
    /// Cooldown period between regime switches (in updates)
    pub switch_cooldown: usize,
    /// Enable automatic strategy activation/deactivation
    pub auto_activate: bool,
}

impl Default for RegimeRouterConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.5,
            switch_cooldown: 10,
            auto_activate: true,
        }
    }
}

/// Strategy assignment for a specific regime
#[derive(Debug, Clone)]
pub struct RegimeStrategyMapping {
    /// The market regime
    pub regime: MarketRegime,
    /// List of strategy IDs to activate in this regime
    pub strategy_ids: Vec<StrategyId>,
    /// Position size scaling factor (0.0 to 1.0)
    pub size_scale: f64,
    /// Maximum concurrent positions
    pub max_positions: usize,
}

/// Regime-aware strategy router
/// Routes incoming signals to appropriate strategies based on detected market regime
pub struct RegimeRouter {
    /// Underlying regime detector
    detector: AdaptiveRegimeDetector,
    /// Router configuration
    config: RegimeRouterConfig,
    /// Strategy mappings for each regime
    mappings: [Option<RegimeStrategyMapping>; 7],
    /// Current active regime
    current_regime: MarketRegime,
    /// Previous regime (for transition detection)
    previous_regime: MarketRegime,
    /// Updates since last regime switch
    updates_since_switch: usize,
    /// Total regime switches
    switch_count: AtomicU64,
    /// Whether routing is enabled
    enabled: AtomicBool,
}

impl RegimeRouter {
    /// Create a new regime router
    pub fn new(num_states: usize, config: RegimeRouterConfig) -> Self {
        let bw_config = BaumWelchConfig::default();
        let detector = AdaptiveRegimeDetector::new(num_states, bw_config);
        
        Self {
            detector,
            config,
            mappings: Default::default(),
            current_regime: MarketRegime::Unknown,
            previous_regime: MarketRegime::Unknown,
            updates_since_switch: 0,
            switch_count: AtomicU64::new(0),
            enabled: AtomicBool::new(true),
        }
    }
    
    /// Register strategy mapping for a regime
    pub fn register_mapping(&mut self, mapping: RegimeStrategyMapping) {
        let idx = mapping.regime.id().min(6);
        self.mappings[idx] = Some(mapping);
    }
    
    /// Process observation and get active strategies
    #[inline]
    pub fn update(&mut self, obs: &ObservationFeatures) -> ActiveStrategies {
        if !self.enabled.load(Ordering::Relaxed) {
            return ActiveStrategies::empty();
        }
        
        // Update regime detection
        let new_regime = self.detector.update(obs);
        let confidence = self.detector.confidence();
        
        self.updates_since_switch += 1;
        
        // Check for regime switch
        if new_regime != self.current_regime 
            && confidence >= self.config.min_confidence
            && self.updates_since_switch >= self.config.switch_cooldown 
        {
            self.previous_regime = self.current_regime;
            self.current_regime = new_regime;
            self.updates_since_switch = 0;
            self.switch_count.fetch_add(1, Ordering::Relaxed);
        }
        
        // Get active strategies for current regime
        self.get_active_strategies()
    }
    
    /// Get currently active strategies based on regime
    #[inline]
    pub fn get_active_strategies(&self) -> ActiveStrategies {
        let mut strategies = ActiveStrategies::empty();
        
        if let Some(idx) = self.get_current_mapping_index() {
            if let Some(ref mapping) = self.mappings[idx] {
                strategies.strategy_ids = mapping.strategy_ids.clone();
                strategies.size_scale = mapping.size_scale;
                strategies.max_positions = mapping.max_positions;
            }
        }
        
        strategies.current_regime = self.current_regime;
        strategies.confidence = self.detector.confidence();
        strategies
    }
    
    /// Get mapping index for current regime
    fn get_current_mapping_index(&self) -> Option<usize> {
        let idx = self.current_regime.id().min(6);
        if self.mappings[idx].is_some() {
            Some(idx)
        } else {
            None
        }
    }
    
    /// Get the current regime
    #[inline]
    pub fn current_regime(&self) -> MarketRegime {
        self.current_regime
    }
    
    /// Get the previous regime
    #[inline]
    pub fn previous_regime(&self) -> MarketRegime {
        self.previous_regime
    }
    
    /// Check if regime has recently switched
    #[inline]
    pub fn recently_switched(&self) -> bool {
        self.updates_since_switch < self.config.switch_cooldown
    }
    
    /// Get regime confidence
    #[inline]
    pub fn confidence(&self) -> f64 {
        self.detector.confidence()
    }
    
    /// Get regime switch count
    #[inline]
    pub fn switch_count(&self) -> u64 {
        self.switch_count.load(Ordering::Relaxed)
    }
    
    /// Enable/disable routing
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Relaxed);
    }
    
    /// Check if routing is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }
    
    /// Get training status
    pub fn training_status(&self) -> TrainingStatus {
        self.detector.training_status()
    }
    
    /// Get all state probabilities
    pub fn state_probabilities(&self) -> Vec<f64> {
        self.detector.state_probabilities().to_vec()
    }
}

/// Active strategies result from regime routing
#[derive(Debug, Clone)]
pub struct ActiveStrategies {
    /// Currently detected regime
    pub current_regime: MarketRegime,
    /// List of active strategy IDs
    pub strategy_ids: Vec<StrategyId>,
    /// Position size scaling factor
    pub size_scale: f64,
    /// Maximum concurrent positions
    pub max_positions: usize,
    /// Confidence in current regime classification
    pub confidence: f64,
}

impl ActiveStrategies {
    /// Create empty active strategies
    pub fn empty() -> Self {
        Self {
            current_regime: MarketRegime::Unknown,
            strategy_ids: Vec::new(),
            size_scale: 0.0,
            max_positions: 0,
            confidence: 0.0,
        }
    }
    
    /// Check if any strategies are active
    pub fn is_empty(&self) -> bool {
        self.strategy_ids.is_empty()
    }
    
    /// Get number of active strategies
    pub fn len(&self) -> usize {
        self.strategy_ids.len()
    }
}

/// Regime transition event
#[derive(Debug, Clone)]
pub struct RegimeTransition {
    /// Previous regime
    pub from: MarketRegime,
    /// New regime
    pub to: MarketRegime,
    /// Timestamp of transition
    pub timestamp_ns: u64,
    /// Confidence in new regime
    pub confidence: f64,
    /// Duration spent in previous regime (in updates)
    pub previous_duration: usize,
}

/// Regime statistics tracker
#[derive(Debug, Default)]
pub struct RegimeStatistics {
    /// Time spent in each regime (in updates)
    pub time_in_regime: [u64; 7],
    /// Number of transitions to each regime
    pub transitions_to: [u64; 7],
    /// Average confidence per regime
    pub avg_confidence: [f64; 7],
    /// Confidence sample counts
    pub confidence_counts: [u64; 7],
    /// Total transitions
    pub total_transitions: u64,
}

impl RegimeStatistics {
    /// Record a regime observation
    pub fn record(&mut self, regime: MarketRegime, confidence: f64) {
        let idx = regime.id().min(6);
        self.time_in_regime[idx] += 1;
        
        // Update running average confidence
        let count = &mut self.confidence_counts[idx];
        let avg = &mut self.avg_confidence[idx];
        *avg = (*avg * *count as f64 + confidence) / (*count as f64 + 1.0);
        *count += 1;
    }
    
    /// Record a regime transition
    pub fn record_transition(&mut self, to_regime: MarketRegime) {
        let idx = to_regime.id().min(6);
        self.transitions_to[idx] += 1;
        self.total_transitions += 1;
    }
    
    /// Get most frequent regime
    pub fn most_frequent_regime(&self) -> MarketRegime {
        let mut best_idx = 0;
        let mut best_time = self.time_in_regime[0];
        
        for i in 1..7 {
            if self.time_in_regime[i] > best_time {
                best_time = self.time_in_regime[i];
                best_idx = i;
            }
        }
        
        MarketRegime::from_id(best_idx)
    }
    
    /// Reset statistics
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_regime_router() {
        let config = RegimeRouterConfig::default();
        let mut router = RegimeRouter::new(4, config);
        
        // Register some strategy mappings
        router.register_mapping(RegimeStrategyMapping {
            regime: MarketRegime::BullTrend,
            strategy_ids: vec![StrategyId(1), StrategyId(2)],
            size_scale: 1.0,
            max_positions: 5,
        });
        
        router.register_mapping(RegimeStrategyMapping {
            regime: MarketRegime::HighVolatility,
            strategy_ids: vec![StrategyId(3)],
            size_scale: 0.5,
            max_positions: 2,
        });
        
        // Process some observations
        for i in 0..50 {
            let obs = ObservationFeatures {
                returns: (i as f64 * 0.002) - 0.05,
                volatility: 0.1 + (i as f64 * 0.003),
                ..Default::default()
            };
            
            let active = router.update(&obs);
            let _ = active;
        }
        
        let current = router.current_regime();
        assert_ne!(current, MarketRegime::Unknown);
        
        let active = router.get_active_strategies();
        let _ = active;
    }
    
    #[test]
    fn test_regime_statistics() {
        let mut stats = RegimeStatistics::default();
        
        stats.record(MarketRegime::BullTrend, 0.8);
        stats.record(MarketRegime::BullTrend, 0.9);
        stats.record(MarketRegime::BearTrend, 0.7);
        
        assert_eq!(stats.time_in_regime[MarketRegime::BullTrend.id()], 2);
        assert_eq!(stats.time_in_regime[MarketRegime::BearTrend.id()], 1);
        
        let most_freq = stats.most_frequent_regime();
        assert_eq!(most_freq, MarketRegime::BullTrend);
    }
}
