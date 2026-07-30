//! Multi-Asset Strategy Router
//! 
//! Parallel strategy router capable of handling simultaneous signals for BTC, ETH, SOL, and USDT pairs.
//! Routes alpha signals from either native Rust SMC engine or Python ML bridge based on latency requirements.

use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, RwLock,
    },
    time::Duration,
};

use crossbeam_channel::{bounded, Receiver, Sender};

/// Cache-line padding constant
const CACHE_LINE_SIZE: usize = 64;

/// Maximum concurrent strategies per symbol
pub const MAX_STRATEGIES_PER_SYMBOL: usize = 8;

/// Signal direction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalDirection {
    Long,
    Short,
    Flat,
}

/// Signal source
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalSource {
    /// Native Rust SMC (Smart Money Concepts) engine
    RustSMC,
    /// Python ML ensemble
    PythonML,
    /// Hybrid (blended)
    Hybrid,
}

/// Trading signal with metadata
#[derive(Debug, Clone)]
pub struct TradingSignal {
    /// Symbol (e.g., "BTCUSDT")
    pub symbol: String,
    /// Signal direction
    pub direction: SignalDirection,
    /// Signal strength (-1.0 to 1.0)
    pub strength: f32,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
    /// Source of the signal
    pub source: SignalSource,
    /// Strategy ID that generated this signal
    pub strategy_id: String,
    /// Timestamp in nanoseconds
    pub timestamp_ns: u64,
    /// Recommended position size (0.0 to 1.0 of max)
    pub position_size: f32,
    /// Stop loss price (0 if not set)
    pub stop_loss: f64,
    /// Take profit price (0 if not set)
    pub take_profit: f64,
    /// Time-to-live in milliseconds
    pub ttl_ms: u64,
}

impl TradingSignal {
    /// Create a new trading signal
    pub fn new(
        symbol: &str,
        direction: SignalDirection,
        strength: f32,
        confidence: f32,
        source: SignalSource,
        strategy_id: &str,
    ) -> Self {
        Self {
            symbol: symbol.to_string(),
            direction,
            strength,
            confidence,
            source,
            strategy_id: strategy_id.to_string(),
            timestamp_ns: get_timestamp_ns(),
            position_size: 0.0,
            stop_loss: 0.0,
            take_profit: 0.0,
            ttl_ms: 5000, // Default 5 second TTL
        }
    }

    /// Set position size
    pub fn with_position_size(mut self, size: f32) -> Self {
        self.position_size = size.clamp(0.0, 1.0);
        self
    }

    /// Set stop loss and take profit
    pub fn with_levels(mut self, stop_loss: f64, take_profit: f64) -> Self {
        self.stop_loss = stop_loss;
        self.take_profit = take_profit;
        self
    }

    /// Check if signal is still valid
    pub fn is_valid(&self) -> bool {
        let now = get_timestamp_ns();
        let age_ms = (now - self.timestamp_ns) / 1_000_000;
        age_ms < self.ttl_ms
    }

    /// Get effective score (strength * confidence)
    pub fn effective_score(&self) -> f32 {
        self.strength * self.confidence
    }
}

/// Strategy actor for parallel signal generation
pub struct StrategyActor {
    /// Strategy ID
    pub id: String,
    /// Strategy type
    pub strategy_type: StrategyType,
    /// Assigned symbols
    pub symbols: Vec<String>,
    /// Signal channel
    signal_sender: Sender<TradingSignal>,
    /// Running flag
    running: Arc<AtomicBool>,
}

/// Strategy types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StrategyType {
    /// Smart Money Concepts (native Rust)
    SMC,
    /// Mean Reversion
    MeanReversion,
    /// Momentum
    Momentum,
    /// Order Flow
    OrderFlow,
    /// ML Ensemble (Python-backed)
    MLEnsemble,
}

/// Strategy router managing parallel strategy execution
pub struct StrategyRouter {
    /// Registered strategies per symbol
    strategies: Arc<RwLock<HashMap<String, Vec<Arc<StrategyActor>>>>>,
    /// Signal receiver
    signal_receiver: Receiver<TradingSignal>,
    /// Signal sender (shared)
    signal_sender: Arc<Sender<TradingSignal>>,
    /// Running flag
    running: Arc<AtomicBool>,
    /// Total signals processed
    signals_processed: Arc<AtomicU64>,
    /// Signals per source
    smc_signals: Arc<AtomicU64>,
    ml_signals: Arc<AtomicU64>,
}

unsafe impl Send for StrategyRouter {}
unsafe impl Sync for StrategyRouter {}

impl StrategyRouter {
    /// Create a new strategy router
    pub fn new() -> Self {
        let (sender, receiver) = bounded(10000); // 10K signal buffer
        
        Self {
            strategies: Arc::new(RwLock::new(HashMap::new())),
            signal_receiver: receiver,
            signal_sender: Arc::new(sender),
            running: Arc::new(AtomicBool::new(false)),
            signals_processed: Arc::new(AtomicU64::new(0)),
            smc_signals: Arc::new(AtomicU64::new(0)),
            ml_signals: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Register a strategy for a symbol
    pub fn register_strategy(
        &self,
        symbol: &str,
        strategy_id: &str,
        strategy_type: StrategyType,
    ) {
        let (sender, _) = bounded(1000);
        
        let actor = Arc::new(StrategyActor {
            id: strategy_id.to_string(),
            strategy_type,
            symbols: vec![symbol.to_string()],
            signal_sender: sender,
            running: self.running.clone(),
        });

        if let Ok(mut strategies) = self.strategies.write() {
            strategies
                .entry(symbol.to_string())
                .or_insert_with(Vec::new)
                .push(actor);
        }
    }

    /// Route a signal from a strategy
    pub fn route_signal(&self, signal: TradingSignal) -> Result<(), crossbeam_channel::TrySendError<TradingSignal>> {
        self.signal_sender.try_send(signal)?;
        
        // Update counters
        self.signals_processed.fetch_add(1, Ordering::Relaxed);
        match signal.source {
            SignalSource::RustSMC | SignalSource::Hybrid => {
                self.smc_signals.fetch_add(1, Ordering::Relaxed);
            }
            SignalSource::PythonML => {
                self.ml_signals.fetch_add(1, Ordering::Relaxed);
            }
        }
        
        Ok(())
    }

    /// Receive routed signals (non-blocking)
    pub fn try_recv_signal(&self) -> Option<TradingSignal> {
        self.signal_receiver.try_recv().ok()
    }

    /// Receive routed signals (blocking with timeout)
    pub fn recv_signal_timeout(&self, timeout: Duration) -> Option<TradingSignal> {
        self.signal_receiver.recv_timeout(timeout).ok()
    }

    /// Start the router
    pub fn start(&self) {
        self.running.store(true, Ordering::Release);
    }

    /// Stop the router
    pub fn stop(&self) {
        self.running.store(false, Ordering::Release);
    }

    /// Get signal statistics
    pub fn get_stats(&self) -> (u64, u64, u64) {
        (
            self.signals_processed.load(Ordering::Relaxed),
            self.smc_signals.load(Ordering::Relaxed),
            self.ml_signals.load(Ordering::Relaxed),
        )
    }

    /// Get registered symbols
    pub fn get_symbols(&self) -> Vec<String> {
        self.strategies.read().ok().map(|s| s.keys().cloned().collect()).unwrap_or_default()
    }

    /// Check if running
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Acquire)
    }
}

impl Default for StrategyRouter {
    fn default() -> Self {
        Self::new()
    }
}

/// Get current timestamp in nanoseconds
fn get_timestamp_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64
}

/// Signal aggregator for combining multiple signals
pub struct SignalAggregator {
    /// Recent signals per symbol
    recent_signals: Arc<RwLock<HashMap<String, Vec<TradingSignal>>>>,
    /// Maximum signals to keep per symbol
    max_signals: usize,
}

unsafe impl Send for SignalAggregator {}
unsafe impl Sync for SignalAggregator {}

impl SignalAggregator {
    pub fn new(max_signals: usize) -> Self {
        Self {
            recent_signals: Arc::new(RwLock::new(HashMap::new())),
            max_signals,
        }
    }

    /// Add a signal to the aggregator
    pub fn add_signal(&self, signal: TradingSignal) {
        if let Ok(mut signals) = self.recent_signals.write() {
            let entry = signals.entry(signal.symbol.clone()).or_insert_with(Vec::new);
            
            // Remove expired signals
            entry.retain(|s| s.is_valid());
            
            // Add new signal
            entry.push(signal);
            
            // Trim to max size
            while entry.len() > self.max_signals {
                entry.remove(0);
            }
        }
    }

    /// Get aggregated signal for a symbol
    pub fn get_aggregated(&self, symbol: &str) -> Option<TradingSignal> {
        let signals = self.recent_signals.read().ok()?;
        let symbol_signals = signals.get(symbol)?;

        if symbol_signals.is_empty() {
            return None;
        }

        // Weighted average of signals
        let mut total_weight = 0.0;
        let mut weighted_direction = 0.0;
        let mut avg_confidence = 0.0;
        let mut best_source = SignalSource::RustSMC;

        for signal in symbol_signals {
            let weight = signal.effective_score().abs();
            total_weight += weight;
            
            let dir_value = match signal.direction {
                SignalDirection::Long => 1.0,
                SignalDirection::Short => -1.0,
                SignalDirection::Flat => 0.0,
            };
            
            weighted_direction += dir_value * weight;
            avg_confidence += signal.confidence;
            
            // Prefer lower latency sources
            if signal.source == SignalSource::RustSMC {
                best_source = SignalSource::RustSMC;
            } else if best_source != SignalSource::RustSMC {
                best_source = signal.source;
            }
        }

        if total_weight == 0.0 {
            return None;
        }

        let direction = if weighted_direction > 0.3 {
            SignalDirection::Long
        } else if weighted_direction < -0.3 {
            SignalDirection::Short
        } else {
            SignalDirection::Flat
        };

        Some(TradingSignal {
            symbol: symbol.to_string(),
            direction,
            strength: (weighted_direction / total_weight).abs() as f32,
            confidence: (avg_confidence / symbol_signals.len() as f32),
            source: best_source,
            strategy_id: "aggregated".to_string(),
            timestamp_ns: get_timestamp_ns(),
            position_size: 0.0,
            stop_loss: 0.0,
            take_profit: 0.0,
            ttl_ms: 1000,
        })
    }

    /// Clear all signals
    pub fn clear(&self) {
        if let Ok(mut signals) = self.recent_signals.write() {
            signals.clear();
        }
    }
}

impl Default for SignalAggregator {
    fn default() -> Self {
        Self::new(100)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trading_signal_creation() {
        let signal = TradingSignal::new(
            "BTCUSDT",
            SignalDirection::Long,
            0.8,
            0.9,
            SignalSource::RustSMC,
            "smc_v1",
        );

        assert_eq!(signal.symbol, "BTCUSDT");
        assert_eq!(signal.direction, SignalDirection::Long);
        assert!((signal.effective_score() - 0.72).abs() < 0.01);
        assert!(signal.is_valid());
    }

    #[test]
    fn test_signal_ttl() {
        let mut signal = TradingSignal::new(
            "ETHUSDT",
            SignalDirection::Short,
            0.5,
            0.7,
            SignalSource::PythonML,
            "ml_v2",
        );
        
        assert!(signal.is_valid());
        
        // Set very short TTL
        signal.ttl_ms = 0;
        assert!(!signal.is_valid());
    }

    #[test]
    fn test_strategy_router() {
        let router = StrategyRouter::new();
        
        router.register_strategy("BTCUSDT", "smc_1", StrategyType::SMC);
        router.register_strategy("BTCUSDT", "ml_1", StrategyType::MLEnsemble);
        
        router.start();
        assert!(router.is_running());
        
        let signal = TradingSignal::new(
            "BTCUSDT",
            SignalDirection::Long,
            0.7,
            0.85,
            SignalSource::Hybrid,
            "smc_1",
        );
        
        let result = router.route_signal(signal);
        assert!(result.is_ok());
        
        let (total, smc, ml) = router.get_stats();
        assert_eq!(total, 1);
        assert_eq!(smc, 1);
        assert_eq!(ml, 0);
        
        router.stop();
    }

    #[test]
    fn test_signal_aggregator() {
        let aggregator = SignalAggregator::new(10);
        
        // Add conflicting signals
        aggregator.add_signal(TradingSignal::new(
            "SOLUSDT",
            SignalDirection::Long,
            0.8,
            0.9,
            SignalSource::RustSMC,
            "smc_1",
        ));
        
        aggregator.add_signal(TradingSignal::new(
            "SOLUSDT",
            SignalDirection::Short,
            0.3,
            0.5,
            SignalSource::PythonML,
            "ml_1",
        ));
        
        let aggregated = aggregator.get_aggregated("SOLUSDT");
        assert!(aggregated.is_some());
        
        let agg = aggregated.unwrap();
        assert_eq!(agg.direction, SignalDirection::Long); // Stronger long signal
    }
}
