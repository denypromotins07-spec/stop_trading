//! Alpha Module Root
//! 
//! Routes multi-asset signals directly to concurrent per-symbol actors.

pub mod triangular;
pub mod statistical;
pub mod lead_lag;
pub mod dominance;
pub mod relative_value;

use std::sync::Arc;
use crossbeam_channel::{bounded, Sender, Receiver};
use dashmap::DashMap;

/// Maximum number of concurrent symbol actors
const MAX_SYMBOL_ACTORS: usize = 512;

/// Message types for symbol actors
#[derive(Debug, Clone)]
pub enum AlphaMessage {
    /// New price tick
    Tick {
        symbol: String,
        price: f64,
        timestamp_ns: u64,
    },
    /// Triangular arb opportunity detected
    TriangularOpportunity(triangular::ArbOpportunity),
    /// Statistical arb signal
    StatSignal {
        pair_idx: usize,
        signal: statistical::StatArbSignal,
    },
    /// Lead-lag relationship update
    LeadLagUpdate {
        leader: String,
        lagger: String,
        correlation: f64,
        lag_ms: u64,
    },
    /// Dominance regime shift
    DominanceShift(dominance::MarketRegime),
    /// Relative value signal
    RelativeValue(relative_value::PairMispricing),
}

/// Alpha signal output
#[derive(Debug, Clone)]
pub struct AlphaSignal {
    /// Signal type identifier
    pub signal_type: &'static str,
    /// Primary asset
    pub asset: String,
    /// Secondary asset (if applicable)
    pub secondary_asset: Option<String>,
    /// Signal strength (-1.0 to 1.0)
    pub strength: f64,
    /// Recommended action
    pub action: AlphaAction,
    /// Timestamp in nanoseconds
    pub timestamp_ns: u64,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
}

/// Action to take based on signal
#[derive(Debug, Clone, PartialEq)]
pub enum AlphaAction {
    /// Enter long position
    Long,
    /// Enter short position
    Short,
    /// Close existing position
    Close,
    /// Increase position size
    Increase,
    /// Decrease position size
    Decrease,
    /// Hold / No action
    Hold,
}

/// Symbol actor handling alpha signals for a specific asset
pub struct SymbolActor {
    /// Symbol name
    pub symbol: String,
    /// Triangular arb graph reference
    pub tri_arb: Option<triangular::TriangularArbGraph>,
    /// Statistical arb pairs involving this symbol
    pub stat_pairs: Vec<usize>,
    /// Lead-lag correlations
    pub lead_lag_stats: DashMap<String, f64>,
}

impl SymbolActor {
    pub fn new(symbol: &str) -> Self {
        Self {
            symbol: symbol.to_string(),
            tri_arb: None,
            stat_pairs: Vec::new(),
            lead_lag_stats: DashMap::new(),
        }
    }

    /// Process incoming message
    pub fn process(&self, msg: AlphaMessage, tx: &Sender<AlphaSignal>) {
        match msg {
            AlphaMessage::Tick { symbol, price, timestamp_ns } => {
                if symbol == self.symbol {
                    // Check for any signals triggered by this tick
                    self.check_signals(price, timestamp_ns, tx);
                }
            }
            AlphaMessage::TriangularOpportunity(opp) => {
                // Forward triangular arb opportunity
                let signal = AlphaSignal {
                    signal_type: "triangular_arb",
                    asset: format!("{:?}", opp.path),
                    secondary_asset: None,
                    strength: opp.profit_bps as f64 / 100.0,
                    action: AlphaAction::Long,
                    timestamp_ns: opp.timestamp_ns,
                    confidence: 0.95,
                };
                let _ = tx.send(signal);
            }
            AlphaMessage::StatSignal { pair_idx, signal } => {
                // Convert stat arb signal to alpha signal
                let action = match signal {
                    statistical::StatArbSignal::LongA_ShortB => AlphaAction::Long,
                    statistical::StatArbSignal::ShortA_LongB => AlphaAction::Short,
                    statistical::StatArbSignal::Close => AlphaAction::Close,
                    statistical::StatArbSignal::Hold => AlphaAction::Hold,
                };

                let alpha_signal = AlphaSignal {
                    signal_type: "statistical_arb",
                    asset: self.symbol.clone(),
                    secondary_asset: None,
                    strength: 1.0,
                    action,
                    timestamp_ns: timestamp_ns(),
                    confidence: 0.85,
                };
                let _ = tx.send(alpha_signal);
            }
            _ => {}
        }
    }

    fn check_signals(&self, _price: f64, _timestamp_ns: u64, _tx: &Sender<AlphaSignal>) {
        // Placeholder for additional signal checks
    }
}

/// Alpha Engine managing all symbol actors
pub struct AlphaEngine {
    /// Symbol actors
    actors: DashMap<String, Arc<SymbolActor>>,
    /// Triangular arb graph
    tri_arb_graph: triangular::TriangularArbGraph,
    /// Statistical arb engine
    stat_arb_engine: statistical::StatArbEngine,
    /// Lead-lag engine
    lead_lag_engine: lead_lag::LeadLagEngine,
    /// Dominance tracker
    dominance_tracker: dominance::DominanceTracker,
    /// Relative value matrix
    rel_value_matrix: relative_value::RelativeValueMatrix,
    /// Output channel for signals
    signal_tx: Sender<AlphaSignal>,
    signal_rx: Receiver<AlphaSignal>,
}

fn timestamp_ns() -> u64 {
    std::time::Instant::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

impl AlphaEngine {
    pub fn new(buffer_size: usize) -> Self {
        let (tx, rx) = bounded(buffer_size);
        
        Self {
            actors: DashMap::new(),
            tri_arb_graph: triangular::TriangularArbGraph::new(5),
            stat_arb_engine: statistical::StatArbEngine::new(2.0, 0.5),
            lead_lag_engine: lead_lag::LeadLagEngine::new(100),
            dominance_tracker: dominance::DominanceTracker::new(),
            rel_value_matrix: relative_value::RelativeValueMatrix::new(32),
            signal_tx: tx,
            signal_rx: rx,
        }
    }

    /// Register a new symbol
    pub fn register_symbol(&self, symbol: &str) {
        let actor = Arc::new(SymbolActor::new(symbol));
        self.actors.insert(symbol.to_string(), actor);
        
        // Also register in triangular arb graph
        let _ = self.tri_arb_graph.register_asset(symbol);
    }

    /// Process a price tick
    pub fn process_tick(&self, symbol: &str, price: f64) {
        let timestamp_ns = timestamp_ns();

        // Update triangular arb edges
        self.update_triangular_edges(symbol, price, timestamp_ns);

        // Check for triangular arb opportunities
        if let Some(asset_idx) = self.get_asset_index(symbol) {
            let opportunities = self.tri_arb_graph.check_cycles_for_asset(asset_idx);
            for opp in opportunities {
                let _ = self.signal_tx.send(AlphaSignal {
                    signal_type: "triangular_arb",
                    asset: format!("{}->{}->{}", 
                        self.tri_arb_graph.get_asset_name(opp.path[0]).unwrap_or("Unknown"),
                        self.tri_arb_graph.get_asset_name(opp.path[1]).unwrap_or("Unknown"),
                        self.tri_arb_graph.get_asset_name(opp.path[2]).unwrap_or("Unknown")
                    ),
                    secondary_asset: None,
                    strength: opp.profit_bps as f64 / 100.0,
                    action: AlphaAction::Long,
                    timestamp_ns: opp.timestamp_ns,
                    confidence: 0.95,
                });
            }
        }

        // Update lead-lag correlations
        self.lead_lag_engine.update_tick(symbol, price, timestamp_ns);

        // Update relative value matrix
        self.rel_value_matrix.update_price(symbol, price, timestamp_ns);

        // Notify symbol actor if exists
        if let Some(actor) = self.actors.get(symbol) {
            let msg = AlphaMessage::Tick {
                symbol: symbol.to_string(),
                price,
                timestamp_ns,
            };
            actor.process(msg, &self.signal_tx);
        }
    }

    fn update_triangular_edges(&self, symbol: &str, price: f64, timestamp_ns: u64) {
        // Update edges involving this symbol in the triangular arb graph
        // This is simplified; in production would update all relevant pairs
        let _ = (symbol, price, timestamp_ns);
    }

    fn get_asset_index(&self, symbol: &str) -> Option<u8> {
        // Find asset index in triangular arb graph
        for i in 0..32u8 {
            if let Some(name) = self.tri_arb_graph.get_asset_name(i) {
                if name == symbol {
                    return Some(i);
                }
            }
        }
        None
    }

    /// Get signal receiver
    pub fn get_signal_receiver(&self) -> Receiver<AlphaSignal> {
        self.signal_rx.clone()
    }

    /// Add statistical arb pair
    pub fn add_stat_pair(&mut self, symbol_a: &str, symbol_b: &str, hedge_ratio: f64) {
        self.stat_arb_engine.add_pair(symbol_a, symbol_b, hedge_ratio);
    }

    /// Update stat pair prices
    pub fn update_stat_pair(&mut self, symbol_a: &str, symbol_b: &str, price_a: f64, price_b: f64) {
        self.stat_arb_engine.update_pair_prices(symbol_a, symbol_b, price_a, price_b);
        
        // Check for signals
        let signals = self.stat_arb_engine.get_all_signals();
        for (pair_idx, signal) in signals {
            let _ = self.signal_tx.send(AlphaSignal {
                signal_type: "statistical_arb",
                asset: symbol_a.to_string(),
                secondary_asset: Some(symbol_b.to_string()),
                strength: 1.0,
                action: match signal {
                    statistical::StatArbSignal::LongA_ShortB => AlphaAction::Long,
                    statistical::StatArbSignal::ShortA_LongB => AlphaAction::Short,
                    statistical::StatArbSignal::Close => AlphaAction::Close,
                    _ => AlphaAction::Hold,
                },
                timestamp_ns: timestamp_ns(),
                confidence: 0.85,
            });
        }
    }

    /// Get current market regime
    pub fn get_market_regime(&self) -> dominance::MarketRegime {
        self.dominance_tracker.current_regime()
    }

    /// Update dominance metrics
    pub fn update_dominance(&mut self, btc_market_cap: f64, total_market_cap: f64) {
        self.dominance_tracker.update(btc_market_cap, total_market_cap);
        
        let regime = self.dominance_tracker.current_regime();
        let _ = self.signal_tx.send(AlphaSignal {
            signal_type: "dominance_regime",
            asset: "BTC".to_string(),
            secondary_asset: None,
            strength: regime.confidence(),
            action: match regime {
                dominance::MarketRegime::RiskOn => AlphaAction::Hold,
                dominance::MarketRegime::RiskOff => AlphaAction::Decrease,
                dominance::MarketRegime::Neutral => AlphaAction::Hold,
            },
            timestamp_ns: timestamp_ns(),
            confidence: regime.confidence(),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alpha_engine_initialization() {
        let engine = AlphaEngine::new(1000);
        
        engine.register_symbol("BTC");
        engine.register_symbol("ETH");
        engine.register_symbol("USDT");

        assert!(engine.actors.contains_key("BTC"));
        assert!(engine.actors.contains_key("ETH"));
        assert!(engine.actors.contains_key("USDT"));
    }

    #[test]
    fn test_tick_processing() {
        let engine = AlphaEngine::new(1000);
        
        engine.register_symbol("BTC");
        engine.register_symbol("ETH");
        
        engine.process_tick("BTC", 50000.0);
        engine.process_tick("ETH", 3000.0);

        // Should not panic
        println!("Tick processing completed");
    }
}
