//! Order Flow Module Root
//! 
//! Aggregates CVD and imbalance metrics for the alpha engine.
//! Exports public traits and types for order flow analysis.

pub mod cvd;
pub mod imbalance;

pub use cvd::{CvdTracker, CvdSnapshot, TradeTick, CvdError, DivergenceSignal, CvdRateOfChange};
pub use imbalance::{
    OrderFlowImbalance, ImbalanceMetrics, ImbalanceSignal, ImbalanceStats,
    PriceLevel, TopOfBook, ImbalanceError, RollingImbalanceCalculator,
};

/// Trait for order flow analysis components
pub trait OrderFlowAnalyzer {
    /// Process incoming tick data
    fn process_tick(&self, tick: &TradeTick) -> Result<(), CvdError>;
    
    /// Get current CVD snapshot
    fn cvd_snapshot(&self) -> CvdSnapshot;
    
    /// Get net delta
    fn net_delta(&self) -> f64;
}

/// Trait for imbalance analysis
pub trait ImbalanceAnalyzer {
    /// Process top of book update
    fn process_tob(&self, tob: &TopOfBook) -> Result<ImbalanceMetrics, ImbalanceError>;
    
    /// Detect absorption patterns
    fn detect_absorption(&self, metrics: &ImbalanceMetrics, price_change: f64) -> bool;
    
    /// Generate trading signal
    fn generate_signal(
        &self,
        metrics: &ImbalanceMetrics,
        price_change: f64,
        avg_volume: f64,
    ) -> ImbalanceSignal;
}

/// Combined order flow state for the alpha engine
#[derive(Debug, Clone)]
pub struct OrderFlowState {
    pub cvd: CvdSnapshot,
    pub imbalance: ImbalanceMetrics,
    pub timestamp_ns: u64,
}

impl OrderFlowState {
    pub fn new(cvd: CvdSnapshot, imbalance: ImbalanceMetrics) -> Self {
        let timestamp_ns = cvd.timestamp_ns.max(imbalance.timestamp_ns);
        Self {
            cvd,
            imbalance,
            timestamp_ns,
        }
    }

    /// Check for confluence between CVD and imbalance signals
    pub fn has_confluence(&self) -> ConfluenceSignal {
        let cvd_positive = self.cvd.net_delta > 0.0;
        let imbalance_positive = self.imbalance.obi > 0.0;

        match (cvd_positive, imbalance_positive) {
            (true, true) => ConfluenceSignal::Bullish,
            (false, false) => ConfluenceSignal::Bearish,
            _ => ConfluenceSignal::Neutral,
        }
    }

    /// Calculate combined strength score (-1.0 to 1.0)
    pub fn combined_strength(&self) -> f64 {
        // Normalize CVD contribution
        let cvd_strength = (self.cvd.net_delta / (self.cvd.cumulative_buy_volume + self.cvd.cumulative_sell_volume).max(1.0))
            .clamp(-1.0, 1.0);
        
        // OBI is already normalized
        let obi_strength = self.imbalance.obi.clamp(-1.0, 1.0);
        
        // Weighted average (CVD more reliable for trend, OBI for short-term)
        cvd_strength * 0.6 + obi_strength * 0.4
    }
}

/// Signal indicating confluence between different order flow metrics
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfluenceSignal {
    Bullish,
    Bearish,
    Neutral,
}

/// Builder for creating order flow analysis pipeline
pub struct OrderFlowPipelineBuilder {
    cvd_scale: i64,
    imbalance_scale: i64,
    lookback_window_ms: u64,
}

impl OrderFlowPipelineBuilder {
    pub fn new() -> Self {
        Self {
            cvd_scale: 1_000_000_000,
            imbalance_scale: 1_000_000_000,
            lookback_window_ms: 60_000,
        }
    }

    pub fn with_cvd_scale(mut self, scale: i64) -> Self {
        self.cvd_scale = scale;
        self
    }

    pub fn with_imbalance_scale(mut self, scale: i64) -> Self {
        self.imbalance_scale = scale;
        self
    }

    pub fn with_lookback_window_ms(mut self, window_ms: u64) -> Self {
        self.lookback_window_ms = window_ms;
        self
    }

    pub fn build(self) -> OrderFlowPipeline {
        let cvd_tracker = CvdTracker::with_scale(self.cvd_scale);
        let imbalance_calc = OrderFlowImbalance::with_scale(self.imbalance_scale);
        imbalance_calc.set_lookback_window_ms(self.lookback_window_ms);

        OrderFlowPipeline {
            cvd_tracker,
            imbalance_calc,
        }
    }
}

impl Default for OrderFlowPipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Complete order flow analysis pipeline
pub struct OrderFlowPipeline {
    pub cvd_tracker: CvdTracker,
    pub imbalance_calc: OrderFlowImbalance,
}

impl OrderFlowPipeline {
    /// Process a trade tick through the pipeline
    pub fn process_tick(&self, tick: &TradeTick) -> Result<(), CvdError> {
        self.cvd_tracker.process_tick(tick)
    }

    /// Process top of book update through the pipeline
    pub fn process_tob(&self, tob: &TopOfBook) -> Result<ImbalanceMetrics, ImbalanceError> {
        self.imbalance_calc.process_update(tob)
    }

    /// Get combined order flow state
    pub fn get_state(&self) -> OrderFlowState {
        OrderFlowState::new(
            self.cvd_tracker.snapshot(),
            ImbalanceMetrics::new(
                self.imbalance_calc.last_timestamp_ns.load(std::sync::atomic::Ordering::Relaxed),
                0.5, // Placeholder - would need to track separately
                self.imbalance_calc.current_obi(),
                0.0, // Placeholder
                self.imbalance_calc.current_obi(),
                0.0, // Placeholder
            ),
        )
    }

    /// Reset all trackers
    pub fn reset(&self) {
        self.cvd_tracker.reset();
        self.imbalance_calc.reset();
    }
}

impl OrderFlowAnalyzer for OrderFlowPipeline {
    fn process_tick(&self, tick: &TradeTick) -> Result<(), CvdError> {
        self.cvd_tracker.process_tick(tick)
    }

    fn cvd_snapshot(&self) -> CvdSnapshot {
        self.cvd_tracker.snapshot()
    }

    fn net_delta(&self) -> f64 {
        self.cvd_tracker.net_delta()
    }
}

impl ImbalanceAnalyzer for OrderFlowPipeline {
    fn process_tob(&self, tob: &TopOfBook) -> Result<ImbalanceMetrics, ImbalanceError> {
        self.imbalance_calc.process_update(tob)
    }

    fn detect_absorption(&self, metrics: &ImbalanceMetrics, price_change: f64) -> bool {
        self.imbalance_calc.detect_absorption(metrics, price_change)
    }

    fn generate_signal(
        &self,
        metrics: &ImbalanceMetrics,
        price_change: f64,
        avg_volume: f64,
    ) -> ImbalanceSignal {
        self.imbalance_calc.generate_signal(metrics, price_change, avg_volume)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_builder() {
        let pipeline = OrderFlowPipelineBuilder::new()
            .with_cvd_scale(1_000_000)
            .with_lookback_window_ms(30_000)
            .build();

        let tick = TradeTick::new(1000, 50000.0, 1.0, false);
        pipeline.process_tick(&tick).unwrap();

        let snapshot = pipeline.cvd_snapshot();
        assert!((snapshot.cumulative_buy_volume - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_confluence_detection() {
        let cvd_snap = CvdSnapshot::new(1000, 100.0, 50.0, 10, 5);
        let imb_metrics = ImbalanceMetrics::new(1000, 0.7, 0.4, 0.0, 0.4, 15.0);
        
        let state = OrderFlowState::new(cvd_snap, imb_metrics);
        
        // Both CVD and OBI positive = bullish confluence
        assert_eq!(state.has_confluence(), ConfluenceSignal::Bullish);
    }

    #[test]
    fn test_combined_strength() {
        let cvd_snap = CvdSnapshot::new(1000, 80.0, 20.0, 8, 2);
        let imb_metrics = ImbalanceMetrics::new(1000, 0.8, 0.6, 0.0, 0.6, 10.0);
        
        let state = OrderFlowState::new(cvd_snap, imb_metrics);
        let strength = state.combined_strength();
        
        // Should be strongly positive
        assert!(strength > 0.5);
    }
}
