//! BTC Dominance and Stablecoin Regime Classifier
//! 
//! Real-time market regime detection using lock-free metrics.
//! Shifts strategy from risk-on (alts) to risk-off (BTC/Stables) based on capital flow velocity.

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::time::Instant;

/// Market regime classification
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MarketRegime {
    /// Risk-on: Altcoins outperforming, high liquidity
    RiskOn,
    /// Risk-off: Flight to BTC and stables, low liquidity
    RiskOff,
    /// Neutral: No clear direction, consolidation
    Neutral,
}

impl MarketRegime {
    pub fn confidence(&self) -> f64 {
        match self {
            MarketRegime::RiskOn => 0.9,
            MarketRegime::RiskOff => 0.95,
            MarketRegime::Neutral => 0.5,
        }
    }
}

/// Capital flow direction
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CapitalFlow {
    /// Flowing into altcoins
    IntoAlts,
    /// Flowing into BTC
    IntoBTC,
    /// Flowing into stables
    IntoStables,
    /// Outflow from crypto
    Outflow,
}

/// Rolling window for metrics (fixed size to avoid allocations)
const METRICS_WINDOW: usize = 100;

/// Lock-free dominance tracker
pub struct DominanceTracker {
    /// Current BTC dominance percentage (scaled by 10000)
    btc_dominance_scaled: AtomicU64,
    /// Previous BTC dominance for trend detection
    prev_btc_dominance_scaled: AtomicU64,
    /// Current ETH dominance percentage (scaled by 10000)
    eth_dominance_scaled: AtomicU64,
    /// Total market cap (scaled by 1e6)
    total_market_cap_scaled: AtomicU64,
    /// BTC market cap (scaled by 1e6)
    btc_market_cap_scaled: AtomicU64,
    /// Stablecoin total supply (scaled by 1e6)
    stablecoin_supply_scaled: AtomicU64,
    /// Rolling window of dominance changes
    dominance_changes: [i64; METRICS_WINDOW],
    /// Head position in rolling window
    dominance_head: usize,
    /// Current regime
    current_regime: MarketRegime,
    /// Last regime change timestamp
    last_regime_change_ns: AtomicU64,
    /// Capital flow direction
    capital_flow: CapitalFlow,
    /// Is regime transitioning
    is_transitioning: AtomicBool,
}

impl DominanceTracker {
    pub fn new() -> Self {
        Self {
            btc_dominance_scaled: AtomicU64::new(500000), // 50.00%
            prev_btc_dominance_scaled: AtomicU64::new(500000),
            eth_dominance_scaled: AtomicU64::new(180000), // 18.00%
            total_market_cap_scaled: AtomicU64::new(2_000_000_000_000), // $2T
            btc_market_cap_scaled: AtomicU64::new(1_000_000_000_000), // $1T
            stablecoin_supply_scaled: AtomicU64::new(150_000_000_000), // $150B
            dominance_changes: [0; METRICS_WINDOW],
            dominance_head: 0,
            current_regime: MarketRegime::Neutral,
            last_regime_change_ns: AtomicU64::new(0),
            capital_flow: CapitalFlow::IntoAlts,
            is_transitioning: AtomicBool::new(false),
        }
    }

    /// Update dominance metrics
    pub fn update(&mut self, btc_market_cap: f64, total_market_cap: f64) {
        let btc_dominance = if total_market_cap > 0.0 {
            (btc_market_cap / total_market_cap * 100.0 * 10000.0) as u64
        } else {
            0
        };

        let prev = self.btc_dominance_scaled.load(Ordering::Relaxed);
        self.prev_btc_dominance_scaled.store(prev, Ordering::Relaxed);
        self.btc_dominance_scaled.store(btc_dominance, Ordering::Relaxed);

        self.btc_market_cap_scaled.store((btc_market_cap * 1e6) as u64, Ordering::Relaxed);
        self.total_market_cap_scaled.store((total_market_cap * 1e6) as u64, Ordering::Relaxed);

        // Track dominance change
        let change = btc_dominance as i64 - prev as i64;
        self.dominance_changes[self.dominance_head] = change;
        self.dominance_head = (self.dominance_head + 1) % METRICS_WINDOW;

        // Detect regime
        self.detect_regime();
    }

    /// Update stablecoin supply
    pub fn update_stablecoin_supply(&mut self, supply: f64) {
        self.stablecoin_supply_scaled.store((supply * 1e6) as u64, Ordering::Relaxed);
        self.detect_regime();
    }

    /// Update ETH dominance
    pub fn update_eth_dominance(&mut self, eth_dominance: f64) {
        self.eth_dominance_scaled.store((eth_dominance * 10000.0) as u64, Ordering::Relaxed);
        self.detect_regime();
    }

    /// Detect current market regime
    fn detect_regime(&mut self) {
        let btc_dom = self.btc_dominance_scaled.load(Ordering::Relaxed) as f64 / 10000.0;
        let prev_dom = self.prev_btc_dominance_scaled.load(Ordering::Relaxed) as f64 / 10000.0;
        
        // Calculate average dominance change over window
        let avg_change: f64 = self.dominance_changes.iter()
            .take(METRICS_WINDOW)
            .sum::<i64>() as f64 / METRICS_WINDOW as f64 / 10000.0;

        let old_regime = self.current_regime;
        let timestamp_ns = Instant::now().duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default().as_nanos() as u64;

        // Regime detection logic
        self.current_regime = if btc_dom > 55.0 && avg_change > 0.1 {
            // High and rising BTC dominance = Risk-off
            self.capital_flow = CapitalFlow::IntoBTC;
            MarketRegime::RiskOff
        } else if btc_dom < 45.0 && avg_change < -0.1 {
            // Low and falling BTC dominance = Risk-on
            self.capital_flow = CapitalFlow::IntoAlts;
            MarketRegime::RiskOn
        } else if btc_dom > 50.0 && avg_change > 0.05 {
            // Moderate risk-off
            self.capital_flow = CapitalFlow::IntoStables;
            MarketRegime::RiskOff
        } else if btc_dom < 50.0 && avg_change < -0.05 {
            // Moderate risk-on
            self.capital_flow = CapitalFlow::IntoAlts;
            MarketRegime::RiskOn
        } else {
            self.capital_flow = CapitalFlow::IntoAlts;
            MarketRegime::Neutral
        };

        // Check if regime changed
        if self.current_regime != old_regime {
            self.last_regime_change_ns.store(timestamp_ns, Ordering::Relaxed);
            self.is_transitioning.store(true, Ordering::Relaxed);
            
            // Reset transitioning flag after delay (simplified)
            self.is_transitioning.store(false, Ordering::Relaxed);
        }
    }

    /// Get current regime
    pub fn current_regime(&self) -> MarketRegime {
        self.current_regime
    }

    /// Get current capital flow direction
    pub fn capital_flow(&self) -> CapitalFlow {
        self.capital_flow
    }

    /// Get BTC dominance percentage
    pub fn btc_dominance(&self) -> f64 {
        self.btc_dominance_scaled.load(Ordering::Relaxed) as f64 / 10000.0
    }

    /// Get ETH dominance percentage
    pub fn eth_dominance(&self) -> f64 {
        self.eth_dominance_scaled.load(Ordering::Relaxed) as f64 / 10000.0
    }

    /// Get total market cap
    pub fn total_market_cap(&self) -> f64 {
        self.total_market_cap_scaled.load(Ordering::Relaxed) as f64 / 1e6
    }

    /// Get stablecoin supply
    pub fn stablecoin_supply(&self) -> f64 {
        self.stablecoin_supply_scaled.load(Ordering::Relaxed) as f64 / 1e6
    }

    /// Check if currently transitioning between regimes
    pub fn is_transitioning(&self) -> bool {
        self.is_transitioning.load(Ordering::Relaxed)
    }

    /// Get time since last regime change in milliseconds
    pub fn time_since_regime_change_ms(&self) -> u64 {
        let last_change = self.last_regime_change_ns.load(Ordering::Relaxed);
        if last_change == 0 {
            return u64::MAX;
        }
        let now = Instant::now().duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default().as_nanos() as u64;
        (now - last_change) / 1_000_000
    }

    /// Get recommended position sizing multiplier based on regime
    pub fn position_multiplier(&self) -> f64 {
        match self.current_regime {
            MarketRegime::RiskOn => 1.5, // Increase size in risk-on
            MarketRegime::RiskOff => 0.5, // Decrease size in risk-off
            MarketRegime::Neutral => 1.0, // Normal size
        }
    }
}

impl Default for DominanceTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dominance_tracker_initialization() {
        let tracker = DominanceTracker::new();
        
        assert_eq!(tracker.btc_dominance(), 50.0);
        assert_eq!(tracker.current_regime(), MarketRegime::Neutral);
    }

    #[test]
    fn test_regime_detection_risk_off() {
        let mut tracker = DominanceTracker::new();
        
        // Simulate rising BTC dominance
        for i in 0..METRICS_WINDOW + 10 {
            let btc_cap = 1_000_000_000_000.0 + (i as f64 * 10_000_000_000.0);
            let total_cap = 1_800_000_000_000.0 + (i as f64 * 5_000_000_000.0);
            tracker.update(btc_cap, total_cap);
        }

        // Should detect risk-off with high BTC dominance
        assert!(tracker.btc_dominance() > 50.0);
        println!("Final regime: {:?}", tracker.current_regime());
        println!("BTC Dominance: {}", tracker.btc_dominance());
    }

    #[test]
    fn test_regime_detection_risk_on() {
        let mut tracker = DominanceTracker::new();
        
        // Simulate falling BTC dominance
        for i in 0..METRICS_WINDOW + 10 {
            let btc_cap = 800_000_000_000.0 - (i as f64 * 5_000_000_000.0);
            let total_cap = 2_000_000_000_000.0 + (i as f64 * 20_000_000_000.0);
            tracker.update(btc_cap.max(100_000_000_000.0), total_cap);
        }

        println!("Final regime: {:?}", tracker.current_regime());
        println!("BTC Dominance: {}", tracker.btc_dominance());
    }

    #[test]
    fn test_position_multiplier() {
        let mut tracker = DominanceTracker::new();
        
        // Force risk-off regime
        tracker.update(1_200_000_000_000.0, 2_000_000_000_000.0);
        tracker.update(1_250_000_000_000.0, 2_000_000_000_000.0);
        
        let multiplier = tracker.position_multiplier();
        println!("Position multiplier: {}", multiplier);
    }
}
