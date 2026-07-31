//! Toxicity Module Root
//! 
//! Routes VPIN spikes directly to the market making quoter to widen spreads defensively.

pub mod vpin;
pub mod imbalance;

use vpin::{VpinCalculator, VpinConfig};
use imbalance::{OrderImbalanceCalculator, OibConfig, PriceLevel};

/// Aggregated toxicity signal combining VPIN and Order Imbalance
#[derive(Debug, Clone, Copy)]
pub struct ToxicitySignal {
    /// VPIN value (0 to 1)
    pub vpin: f64,
    /// Weighted order imbalance (-1 to 1)
    pub oib: f64,
    /// Combined toxicity score (0 to 1)
    pub combined_score: f64,
    /// Recommended spread multiplier
    pub spread_multiplier: f64,
    /// Whether to pause aggressive quoting
    pub should_pause_aggressive: bool,
}

impl Default for ToxicitySignal {
    fn default() -> Self {
        Self {
            vpin: 0.0,
            oib: 0.0,
            combined_score: 0.0,
            spread_multiplier: 1.0,
            should_pause_aggressive: false,
        }
    }
}

/// Configuration for the toxicity module
#[derive(Debug, Clone)]
pub struct ToxicityConfig {
    pub vpin_config: VpinConfig,
    pub oib_config: OibConfig,
    /// VPIN threshold for defensive mode
    pub vpin_defensive_threshold: f64,
    /// OIB threshold for directional adjustment
    pub oib_directional_threshold: f64,
    /// Maximum spread multiplier
    pub max_spread_multiplier: f64,
}

impl Default for ToxicityConfig {
    fn default() -> Self {
        Self {
            vpin_config: VpinConfig::default(),
            oib_config: OibConfig::default(),
            vpin_defensive_threshold: 0.75,
            oib_directional_threshold: 0.4,
            max_spread_multiplier: 5.0,
        }
    }
}

/// Main toxicity analytics engine
pub struct ToxicityEngine {
    vpin_calc: VpinCalculator,
    oib_calc: OrderImbalanceCalculator,
    config: ToxicityConfig,
    current_signal: ToxicitySignal,
}

impl ToxicityEngine {
    pub fn new(config: ToxicityConfig) -> Self {
        Self {
            vpin_calc: VpinCalculator::new(config.vpin_config.clone()),
            oib_calc: OrderImbalanceCalculator::new(config.oib_config.clone()),
            config,
            current_signal: ToxicitySignal::default(),
        }
    }

    /// Process a trade tick and update toxicity metrics
    pub fn process_trade(&mut self, volume: u64, price: f64, is_buyer_maker: bool) {
        let _ = self.vpin_calc.process_trade(volume, price, is_buyer_maker);
        self.update_signal();
    }

    /// Update order book levels and recalculate OIB
    pub fn update_order_book(&mut self, bids: &[PriceLevel], asks: &[PriceLevel]) {
        self.oib_calc.update(bids, asks);
        self.update_signal();
    }

    fn update_signal(&mut self) {
        let vpin = self.vpin_calc.vpin();
        let oib = self.oib_calc.weighted_oib();
        
        // Combined score: weighted average of VPIN and |OIB|
        // VPIN indicates overall toxicity, OIB indicates direction
        let oib_abs = oib.abs();
        let combined = (vpin * 0.6 + oib_abs * 0.4).clamp(0.0, 1.0);
        
        // Calculate spread multiplier based on toxicity
        let base_multiplier = if vpin >= self.config.vpin_defensive_threshold {
            1.0 + ((vpin - self.config.vpin_defensive_threshold) * 4.0).exp().min(self.config.max_spread_multiplier - 1.0)
        } else {
            1.0
        };
        
        // Additional adjustment for extreme OIB
        let oib_adjustment = if oib_abs > self.config.oib_directional_threshold {
            1.0 + (oib_abs - self.config.oib_directional_threshold) * 2.0
        } else {
            1.0
        };
        
        let spread_multiplier = (base_multiplier * oib_adjustment)
            .clamp(1.0, self.config.max_spread_multiplier);
        
        // Determine if we should pause aggressive quoting
        let should_pause = vpin >= self.config.vpin_defensive_threshold * 1.2;
        
        self.current_signal = ToxicitySignal {
            vpin,
            oib,
            combined_score: combined,
            spread_multiplier,
            should_pause_aggressive: should_pause,
        };
    }

    /// Get the current toxicity signal
    pub fn get_signal(&self) -> ToxicitySignal {
        self.current_signal
    }

    /// Check if market is toxic
    pub fn is_toxic(&self) -> bool {
        self.vpin_calc.is_toxic()
    }

    /// Get recommended bid/ask spread given a base spread
    pub fn adjusted_spread(&self, base_bid: f64, base_ask: f64) -> (f64, f64) {
        let mid = (base_bid + base_ask) / 2.0;
        let half_spread = (base_ask - base_bid) / 2.0 * self.current_signal.spread_multiplier;
        
        // Adjust spread direction based on OIB
        let oib_adjustment = self.current_signal.oib * half_spread * 0.2;
        
        (mid - half_spread - oib_adjustment, mid + half_spread - oib_adjustment)
    }

    /// Get the VPIN calculator for direct access
    pub fn vpin_calculator(&self) -> &VpinCalculator {
        &self.vpin_calc
    }

    /// Get the OIB calculator for direct access
    pub fn oib_calculator(&self) -> &OrderImbalanceCalculator {
        &self.oib_calc
    }

    /// Reset all calculators
    pub fn reset(&mut self) {
        self.vpin_calc.reset();
        self.current_signal = ToxicitySignal::default();
    }
}

/// Event types emitted by the toxicity engine
#[derive(Debug, Clone)]
pub enum ToxicityEvent {
    /// VPIN crossed above threshold
    VpinSpike(f64),
    /// OIB shifted significantly
    OibShift(f64),
    /// Entered defensive mode
    DefensiveModeEntered,
    /// Exited defensive mode
    DefensiveModeExited,
}

/// Stream processor for toxicity events
pub struct ToxicityEventStream {
    prev_vpin: f64,
    prev_oib: f64,
    was_defensive: bool,
    vpin_spike_threshold: f64,
    oib_shift_threshold: f64,
}

impl ToxicityEventStream {
    pub fn new(vpin_spike_threshold: f64, oib_shift_threshold: f64) -> Self {
        Self {
            prev_vpin: 0.0,
            prev_oib: 0.0,
            was_defensive: false,
            vpin_spike_threshold,
            oib_shift_threshold,
        }
    }

    /// Check for events and return any that occurred
    pub fn check_events(&mut self, signal: ToxicitySignal) -> Vec<ToxicityEvent> {
        let mut events = Vec::new();

        // Check for VPIN spike
        if signal.vpin >= self.vpin_spike_threshold && self.prev_vpin < self.vpin_spike_threshold {
            events.push(ToxicityEvent::VpinSpike(signal.vpin));
        }

        // Check for OIB shift
        if (signal.oib - self.prev_oib).abs() > self.oib_shift_threshold {
            events.push(ToxicityEvent::OibShift(signal.oib));
        }

        // Check for defensive mode transitions
        if signal.should_pause_aggressive && !self.was_defensive {
            events.push(ToxicityEvent::DefensiveModeEntered);
        } else if !signal.should_pause_aggressive && self.was_defensive {
            events.push(ToxicityEvent::DefensiveModeExited);
        }

        self.prev_vpin = signal.vpin;
        self.prev_oib = signal.oib;
        self.was_defensive = signal.should_pause_aggressive;

        events
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_toxicity_engine_basic() {
        let config = ToxicityConfig::default();
        let mut engine = ToxicityEngine::new(config);

        // Initial state should be non-toxic
        assert!(!engine.is_toxic());
        
        let signal = engine.get_signal();
        assert_eq!(signal.spread_multiplier, 1.0);
    }

    #[test]
    fn test_event_stream() {
        let mut stream = ToxicityEventStream::new(0.8, 0.3);
        
        let normal_signal = ToxicitySignal {
            vpin: 0.5,
            oib: 0.1,
            ..Default::default()
        };
        
        let toxic_signal = ToxicitySignal {
            vpin: 0.9,
            oib: 0.5,
            combined_score: 0.8,
            spread_multiplier: 2.0,
            should_pause_aggressive: true,
        };
        
        let events1 = stream.check_events(normal_signal);
        assert!(events1.is_empty());
        
        let events2 = stream.check_events(toxic_signal);
        assert!(!events2.is_empty());
    }
}
