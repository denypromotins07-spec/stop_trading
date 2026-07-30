//! Data Quality Module Root
//! 
//! Wires anomaly alerts directly to the global kill switch.

pub mod anomaly;
pub mod venue_score;

use std::sync::atomic::{AtomicU64, AtomicBool, AtomicU8, Ordering};
use std::time::Instant;
use crossbeam_channel::{bounded, Sender, Receiver};
use dashmap::DashMap;

/// Kill switch states
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KillSwitchState {
    /// All systems operational
    Green,
    /// Minor issues, reduced activity
    Yellow,
    /// Major issues, halt new trades
    Red,
    /// Emergency shutdown
    Emergency,
}

/// Alert severity levels
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub enum AlertSeverity {
    Info = 1,
    Warning = 2,
    Error = 3,
    Critical = 4,
    Fatal = 5,
}

/// Global alert
#[derive(Debug, Clone)]
pub struct GlobalAlert {
    /// Alert ID
    pub id: u64,
    /// Source module
    pub source: &'static str,
    /// Severity
    pub severity: AlertSeverity,
    /// Message
    pub message: String,
    /// Timestamp
    pub timestamp_ns: u64,
    /// Triggers kill switch
    pub triggers_kill_switch: bool,
}

/// Data quality metrics
#[derive(Debug, Clone)]
pub struct DataQualityMetrics {
    /// Total ticks processed
    pub total_ticks: u64,
    /// Anomalies detected
    pub anomalies_detected: u64,
    /// Quarantined feeds
    pub quarantined_feeds: usize,
    /// Average venue score
    pub avg_venue_score: f64,
    /// Latency p99 in microseconds
    pub latency_p99_us: u64,
}

/// Data Quality Engine with kill switch integration
pub struct DataQualityEngine {
    /// Anomaly detector
    anomaly_detector: anomaly::AnomalyDetector,
    /// Venue scorer
    venue_scorer: venue_score::VenueScorer,
    /// Kill switch state
    kill_switch_state: AtomicU8,
    /// Alert counter
    alert_counter: AtomicU64,
    /// Alerts generated
    alerts: DashMap<u64, GlobalAlert>,
    /// Event channel
    event_tx: Sender<DataQualityEvent>,
    event_rx: Receiver<DataQualityEvent>,
    /// Is engine active
    is_active: AtomicBool,
    /// Consecutive errors before yellow
    yellow_threshold: u32,
    /// Consecutive errors before red
    red_threshold: u32,
    /// Consecutive critical errors
    consecutive_critical: AtomicU32,
}

/// Data quality events
#[derive(Debug, Clone)]
pub enum DataQualityEvent {
    /// New alert
    Alert(GlobalAlert),
    /// Kill switch state change
    KillSwitchChanged(KillSwitchState),
    /// Feed quarantined
    FeedQuarantined { symbol: String, venue: String },
    /// Metrics update
    MetricsUpdate(DataQualityMetrics),
}

impl DataQualityEngine {
    pub fn new(buffer_size: usize) -> Self {
        let (tx, rx) = bounded(buffer_size);

        Self {
            anomaly_detector: anomaly::AnomalyDetector::new(5000, 1000, 10000),
            venue_scorer: venue_score::VenueScorer::new(0.4, 0.3, 0.2, 0.1),
            kill_switch_state: AtomicU8::new(KillSwitchState::Green as u8),
            alert_counter: AtomicU64::new(0),
            alerts: DashMap::new(),
            event_tx: tx,
            event_rx: rx,
            is_active: AtomicBool::new(true),
            yellow_threshold: 10,
            red_threshold: 50,
            consecutive_critical: AtomicU32::new(0),
        }
    }

    /// Check a quote through the data quality pipeline
    pub fn check_quote(&self, symbol: &str, venue: &str, quote: &anomaly::Quote) -> bool {
        if !self.is_active.load(Ordering::Relaxed) {
            return false;
        }

        // Check current kill switch state
        let state = self.get_kill_switch_state();
        if state == KillSwitchState::Red || state == KillSwitchState::Emergency {
            return false;
        }

        // Run anomaly detection
        let alerts = self.anomaly_detector.check_quote(symbol, venue, quote);

        // Process each alert
        for alert in alerts {
            let severity = match alert.anomaly_type {
                anomaly::AnomalyType::InvalidPrice => AlertSeverity::Fatal,
                anomaly::AnomalyType::CrossedBook => AlertSeverity::Critical,
                anomaly::AnomalyType::PriceSpike => AlertSeverity::Error,
                anomaly::AnomalyType::StaleQuote => AlertSeverity::Warning,
                anomaly::AnomalyType::ZeroSize => AlertSeverity::Info,
                anomaly::AnomalyType::OutOfSequence => AlertSeverity::Warning,
                anomaly::AnomalyType::LatencySpike => AlertSeverity::Warning,
            };

            let triggers_kill = severity >= AlertSeverity::Critical;
            
            let global_alert = GlobalAlert {
                id: self.alert_counter.fetch_add(1, Ordering::Relaxed),
                source: "anomaly_detector",
                severity,
                message: alert.description,
                timestamp_ns: alert.timestamp_ns,
                triggers_kill_switch: triggers_kill,
            };

            self.process_alert(global_alert);

            // Handle quarantine
            if alert.quarantine_recommended {
                let _ = self.event_tx.send(DataQualityEvent::FeedQuarantined {
                    symbol: symbol.to_string(),
                    venue: venue.to_string(),
                });
            }
        }

        // Update venue metrics
        let now_ns = Instant::now().duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default().as_nanos() as u64;
        
        let metrics = venue_score::VenueMetrics {
            avg_latency_us: ((now_ns - quote.received_ns) / 1000) as f64,
            latency_stddev_us: 0.0, // Would track over time
            best_bid_depth: quote.bid_size * quote.bid,
            best_ask_depth: quote.ask_size * quote.ask,
            depth_1pct: (quote.bid_size * quote.bid + quote.ask_size * quote.ask),
            maker_fee_bps: 10.0, // Would get from venue config
            taker_fee_bps: 10.0,
            success_rate: 1.0,
            errors_last_minute: 0,
            quote_rate: 1000.0,
        };

        self.venue_scorer.update_metrics(venue, symbol, metrics);

        // Return whether quote is safe to use
        state == KillSwitchState::Green && alerts.is_empty()
    }

    fn process_alert(&self, alert: GlobalAlert) {
        let id = alert.id;
        self.alerts.insert(id, alert.clone());

        let _ = self.event_tx.send(DataQualityEvent::Alert(alert.clone()));

        // Update kill switch if needed
        if alert.triggers_kill_switch {
            self.consecutive_critical.fetch_add(1, Ordering::Relaxed);
            let consecutive = self.consecutive_critical.load(Ordering::Relaxed);

            let current_state = self.get_kill_switch_state();
            let new_state = if consecutive >= 5 {
                KillSwitchState::Emergency
            } else if consecutive >= 2 || alert.severity == AlertSeverity::Fatal {
                KillSwitchState::Red
            } else if current_state == KillSwitchState::Green {
                KillSwitchState::Yellow
            } else {
                current_state
            };

            if new_state != current_state {
                self.set_kill_switch_state(new_state);
            }
        } else {
            // Reset consecutive counter on non-critical
            self.consecutive_critical.store(0, Ordering::Relaxed);
        }
    }

    /// Get current kill switch state
    pub fn get_kill_switch_state(&self) -> KillSwitchState {
        match self.kill_switch_state.load(Ordering::Relaxed) {
            0 => KillSwitchState::Green,
            1 => KillSwitchState::Yellow,
            2 => KillSwitchState::Red,
            3 => KillSwitchState::Emergency,
            _ => KillSwitchState::Green,
        }
    }

    fn set_kill_switch_state(&self, state: KillSwitchState) {
        self.kill_switch_state.store(state as u8, Ordering::Relaxed);
        let _ = self.event_tx.send(DataQualityEvent::KillSwitchChanged(state));
    }

    /// Manually trigger kill switch
    pub fn trigger_kill_switch(&self, severity: KillSwitchState) {
        self.set_kill_switch_state(severity);
    }

    /// Reset kill switch to green
    pub fn reset_kill_switch(&self) {
        self.consecutive_critical.store(0, Ordering::Relaxed);
        self.set_kill_switch_state(KillSwitchState::Green);
    }

    /// Get data quality metrics
    pub fn get_metrics(&self) -> DataQualityMetrics {
        DataQualityMetrics {
            total_ticks: self.alert_counter.load(Ordering::Relaxed),
            anomalies_detected: self.anomaly_detector.get_alert_count(),
            quarantined_feeds: self.anomaly_detector.get_quarantine_count(),
            avg_venue_score: 75.0, // Would calculate from venue scores
            latency_p99_us: 1000,  // Would calculate from samples
        }
    }

    /// Get event receiver
    pub fn get_event_receiver(&self) -> Receiver<DataQualityEvent> {
        self.event_rx.clone()
    }

    /// Check if trading is allowed
    pub fn is_trading_allowed(&self) -> bool {
        let state = self.get_kill_switch_state();
        state == KillSwitchState::Green || state == KillSwitchState::Yellow
    }

    /// Deactivate engine
    pub fn deactivate(&self) {
        self.is_active.store(false, Ordering::Relaxed);
    }

    /// Activate engine
    pub fn activate(&self) {
        self.is_active.store(true, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kill_switch_states() {
        let engine = DataQualityEngine::new(1000);
        
        assert_eq!(engine.get_kill_switch_state(), KillSwitchState::Green);
        assert!(engine.is_trading_allowed());

        // Trigger yellow
        engine.trigger_kill_switch(KillSwitchState::Yellow);
        assert_eq!(engine.get_kill_switch_state(), KillSwitchState::Yellow);
        assert!(engine.is_trading_allowed());

        // Trigger red
        engine.trigger_kill_switch(KillSwitchState::Red);
        assert_eq!(engine.get_kill_switch_state(), KillSwitchState::Red);
        assert!(!engine.is_trading_allowed());

        // Reset
        engine.reset_kill_switch();
        assert_eq!(engine.get_kill_switch_state(), KillSwitchState::Green);
    }

    #[test]
    fn test_valid_quote_processing() {
        let engine = DataQualityEngine::new(1000);

        let quote = anomaly::Quote {
            bid: 50000.0,
            ask: 50001.0,
            bid_size: 1.0,
            ask_size: 1.0,
            timestamp_ns: 1000000000,
            received_ns: 1000000000,
        };

        let result = engine.check_quote("BTCUSDT", "binance", &quote);
        assert!(result, "Valid quote should be accepted");
    }

    #[test]
    fn test_crossed_book_triggers_alert() {
        let engine = DataQualityEngine::new(1000);

        let quote = anomaly::Quote {
            bid: 50100.0,
            ask: 50000.0, // Crossed!
            bid_size: 1.0,
            ask_size: 1.0,
            timestamp_ns: 1000000000,
            received_ns: 1000000000,
        };

        let result = engine.check_quote("BTCUSDT", "binance", &quote);
        assert!(!result, "Crossed book should be rejected");
        
        // Should have triggered an alert
        let metrics = engine.get_metrics();
        assert!(metrics.anomalies_detected > 0);
    }
}
