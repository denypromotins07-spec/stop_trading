//! Bot Behavior Anomaly Detection Module
//! 
//! Tracks order-to-trade ratios, cancellation rates, and quote stuffing patterns.
//! Prevents the bot from entering toxic, high-frequency cancellation loops that
//! could trigger exchange API bans.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};
use crate::gateway::venue::VenueId;

/// Time window for behavior analysis (1 minute default)
const BEHAVIOR_WINDOW_MS: u64 = 60_000;

/// Alert threshold constants
const ORDER_TO_TRADE_RATIO_WARNING: f64 = 50.0;
const ORDER_TO_TRADE_RATIO_CRITICAL: f64 = 100.0;
const CANCELLATION_RATE_WARNING: f64 = 0.8;
const CANCELLATION_RATE_CRITICAL: f64 = 0.95;
const QUOTE_STUFFING_THRESHOLD_PER_SEC: u64 = 100;

/// Behavior event types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum BehaviorEvent {
    OrderSubmitted = 0,
    OrderCancelled = 1,
    OrderFilled = 2,
    OrderRejected = 3,
    OrderModified = 4,
}

/// Anomaly type detected
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnomalyType {
    HighOrderToTradeRatio,
    ExcessiveCancellations,
    QuoteStuffing,
    RapidModifications,
    OrderFlood,
    UnusualPattern,
}

/// Severity level for anomalies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum AnomalySeverity {
    Info = 0,
    Warning = 1,
    Critical = 2,
    Emergency = 3,
}

/// Behavior anomaly alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehaviorAlert {
    pub alert_id: u64,
    pub venue_id: VenueId,
    pub symbol: Option<[u8; 12]>,
    pub anomaly_type: AnomalyType,
    pub severity: AnomalySeverity,
    pub current_value: f64,
    pub threshold: f64,
    pub message: String,
    pub timestamp_ns: u64,
    pub recommended_action: &'static str,
}

/// Per-symbol behavior metrics
#[derive(Debug, Clone, Default)]
struct SymbolMetrics {
    /// Orders submitted in window
    orders_submitted: u64,
    /// Orders cancelled in window
    orders_cancelled: u64,
    /// Orders filled in window
    orders_filled: u64,
    /// Orders rejected in window
    orders_rejected: u64,
    /// Orders modified in window
    orders_modified: u64,
    /// Timestamps of recent events for rate detection
    recent_events: VecDeque<u64>,
    /// Total message count for quote stuffing detection
    message_count: u64,
}

impl SymbolMetrics {
    fn new() -> Self {
        Self {
            recent_events: VecDeque::with_capacity(1000),
            ..Default::default()
        }
    }

    fn record_event(&mut self, event_type: BehaviorEvent, timestamp_ns: u64) {
        match event_type {
            BehaviorEvent::OrderSubmitted => self.orders_submitted += 1,
            BehaviorEvent::OrderCancelled => self.orders_cancelled += 1,
            BehaviorEvent::OrderFilled => self.orders_filled += 1,
            BehaviorEvent::OrderRejected => self.orders_rejected += 1,
            BehaviorEvent::OrderModified => self.orders_modified += 1,
        }

        self.message_count += 1;
        self.recent_events.push_back(timestamp_ns);

        // Limit history size
        if self.recent_events.len() > 1000 {
            self.recent_events.drain(0..500);
        }
    }

    fn get_order_to_trade_ratio(&self) -> f64 {
        if self.orders_filled == 0 {
            if self.orders_submitted == 0 {
                return 0.0;
            }
            f64::INFINITY
        } else {
            self.orders_submitted as f64 / self.orders_filled as f64
        }
    }

    fn get_cancellation_rate(&self) -> f64 {
        if self.orders_submitted == 0 {
            return 0.0;
        }
        self.orders_cancelled as f64 / self.orders_submitted as f64
    }

    fn get_messages_per_second(&self, window_ns: u64) -> f64 {
        if window_ns == 0 {
            return 0.0;
        }
        self.message_count as f64 * 1_000_000_000.0 / window_ns as f64
    }

    fn reset(&mut self) {
        *self = SymbolMetrics::new();
    }
}

/// Per-venue behavior tracker
struct VenueBehaviorTracker {
    venue_id: VenueId,
    /// Per-symbol metrics
    symbol_metrics: HashMap<[u8; 12], SymbolMetrics>,
    /// Aggregate venue metrics
    venue_metrics: SymbolMetrics,
    /// Window start timestamp
    window_start_ns: u64,
    /// Window duration in nanoseconds
    window_duration_ns: u64,
}

impl VenueBehaviorTracker {
    fn new(venue_id: VenueId) -> Self {
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        Self {
            venue_id,
            symbol_metrics: HashMap::new(),
            venue_metrics: SymbolMetrics::new(),
            window_start_ns: now_ns,
            window_duration_ns: BEHAVIOR_WINDOW_MS * 1_000_000,
        }
    }

    fn record_event(&mut self, symbol: [u8; 12], event_type: BehaviorEvent, timestamp_ns: u64) {
        // Check if window has expired
        if timestamp_ns - self.window_start_ns > self.window_duration_ns {
            self.reset_window(timestamp_ns);
        }

        // Record to symbol metrics
        let symbol_metric = self.symbol_metrics.entry(symbol).or_insert_with(SymbolMetrics::new);
        symbol_metric.record_event(event_type, timestamp_ns);

        // Record to venue aggregate
        self.venue_metrics.record_event(event_type, timestamp_ns);
    }

    fn reset_window(&mut self, new_start_ns: u64) {
        self.window_start_ns = new_start_ns;
        self.venue_metrics.reset();
        for metric in self.symbol_metrics.values_mut() {
            metric.reset();
        }
    }

    fn check_anomalies(&self, current_time_ns: u64) -> Vec<BehaviorAlert> {
        let mut alerts = Vec::new();
        let window_sec = self.window_duration_ns as f64 / 1_000_000_000.0;

        // Check venue-level anomalies
        if let Some(alert) = self.check_venue_anomalies(window_sec, current_time_ns) {
            alerts.push(alert);
        }

        // Check per-symbol anomalies
        for (symbol, metrics) in &self.symbol_metrics {
            if let Some(mut alert) = self.check_symbol_anomalies(*symbol, metrics, window_sec, current_time_ns) {
                alert.symbol = Some(*symbol);
                alerts.push(alert);
            }
        }

        alerts
    }

    fn check_venue_anomalies(&self, window_sec: f64, current_time_ns: u64) -> Option<BehaviorAlert> {
        static ALERT_COUNTER: AtomicU64 = AtomicU64::new(0);

        // Check order-to-trade ratio
        let otr = self.venue_metrics.get_order_to_trade_ratio();
        if otr.is_finite() && otr > ORDER_TO_TRADE_RATIO_CRITICAL {
            return Some(BehaviorAlert {
                alert_id: ALERT_COUNTER.fetch_add(1, Ordering::Relaxed),
                venue_id: self.venue_id,
                symbol: None,
                anomaly_type: AnomalyType::HighOrderToTradeRatio,
                severity: AnomalySeverity::Critical,
                current_value: otr,
                threshold: ORDER_TO_TRADE_RATIO_CRITICAL,
                message: format!("Critical order-to-trade ratio: {:.1}", otr),
                timestamp_ns: current_time_ns,
                recommended_action: "Reduce order submission frequency",
            });
        }

        // Check cancellation rate
        let cancel_rate = self.venue_metrics.get_cancellation_rate();
        if cancel_rate > CANCELLATION_RATE_CRITICAL {
            return Some(BehaviorAlert {
                alert_id: ALERT_COUNTER.fetch_add(1, Ordering::Relaxed),
                venue_id: self.venue_id,
                symbol: None,
                anomaly_type: AnomalyType::ExcessiveCancellations,
                severity: AnomalySeverity::Critical,
                current_value: cancel_rate,
                threshold: CANCELLATION_RATE_CRITICAL,
                message: format!("Critical cancellation rate: {:.1}%", cancel_rate * 100.0),
                timestamp_ns: current_time_ns,
                recommended_action: "Review quoting strategy parameters",
            });
        }

        // Check quote stuffing
        let msg_per_sec = self.venue_metrics.get_messages_per_second(self.window_duration_ns);
        if msg_per_sec > QUOTE_STUFFING_THRESHOLD_PER_SEC as f64 {
            return Some(BehaviorAlert {
                alert_id: ALERT_COUNTER.fetch_add(1, Ordering::Relaxed),
                venue_id: self.venue_id,
                symbol: None,
                anomaly_type: AnomalyType::QuoteStuffing,
                severity: AnomalySeverity::Emergency,
                current_value: msg_per_sec,
                threshold: QUOTE_STUFFING_THRESHOLD_PER_SEC as f64,
                message: format!("Quote stuffing detected: {:.0} msg/sec", msg_per_sec),
                timestamp_ns: current_time_ns,
                recommended_action: "IMMEDIATE: Reduce message rate or pause trading",
            });
        }

        None
    }

    fn check_symbol_anomalies(
        &self,
        symbol: [u8; 12],
        metrics: &SymbolMetrics,
        window_sec: f64,
        current_time_ns: u64,
    ) -> Option<BehaviorAlert> {
        static ALERT_COUNTER: AtomicU64 = AtomicU64::new(10000);

        // Check excessive modifications
        if metrics.orders_modified > metrics.orders_submitted / 2 && metrics.orders_submitted > 10 {
            return Some(BehaviorAlert {
                alert_id: ALERT_COUNTER.fetch_add(1, Ordering::Relaxed),
                venue_id: self.venue_id,
                symbol: Some(symbol),
                anomaly_type: AnomalyType::RapidModifications,
                severity: AnomalySeverity::Warning,
                current_value: metrics.orders_modified as f64,
                threshold: (metrics.orders_submitted / 2) as f64,
                message: format!("Excessive order modifications on symbol"),
                timestamp_ns: current_time_ns,
                recommended_action: "Review price update logic",
            });
        }

        None
    }
}

/// Main Behavior Anomaly Detector
pub struct BehaviorAnomalyDetector {
    /// Per-venue trackers
    venue_trackers: parking_lot::RwLock<HashMap<VenueId, VenueBehaviorTracker>>,
    /// Detector enabled flag
    enabled: AtomicBool,
    /// Total alerts generated
    alerts_generated: AtomicU64,
    /// Events processed
    events_processed: AtomicU64,
    /// Alert callback
    alert_callback: Option<Arc<dyn Fn(BehaviorAlert) + Send + Sync>>,
}

impl BehaviorAnomalyDetector {
    pub fn new(venues: &[VenueId]) -> Self {
        let mut venue_trackers = HashMap::with_capacity(venues.len());
        for &venue_id in venues {
            venue_trackers.insert(venue_id, VenueBehaviorTracker::new(venue_id));
        }

        Self {
            venue_trackers: parking_lot::RwLock::new(venue_trackers),
            enabled: AtomicBool::new(true),
            alerts_generated: AtomicU64::new(0),
            events_processed: AtomicU64::new(0),
            alert_callback: None,
        }
    }

    /// Set alert callback
    pub fn set_alert_callback<F>(&mut self, callback: F)
    where
        F: Fn(BehaviorAlert) + Send + Sync + 'static,
    {
        self.alert_callback = Some(Arc::new(callback));
    }

    /// Record a behavior event
    pub fn record_event(&self, venue_id: VenueId, symbol: [u8; 12], event_type: BehaviorEvent) {
        if !self.enabled.load(Ordering::Acquire) {
            return;
        }

        self.events_processed.fetch_add(1, Ordering::Relaxed);

        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        let mut trackers = self.venue_trackers.write();
        if let Some(tracker) = trackers.get_mut(&venue_id) {
            tracker.record_event(symbol, event_type, now_ns);
        }
    }

    /// Check for anomalies across all venues
    pub fn check_all_anomalies(&self) -> Vec<BehaviorAlert> {
        if !self.enabled.load(Ordering::Acquire) {
            return Vec::new();
        }

        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        let trackers = self.venue_trackers.read();
        let mut all_alerts = Vec::new();

        for tracker in trackers.values() {
            let alerts = tracker.check_anomalies(now_ns);
            for alert in alerts {
                self.alerts_generated.fetch_add(1, Ordering::Relaxed);

                if let Some(ref callback) = self.alert_callback {
                    callback(alert.clone());
                }

                all_alerts.push(alert);
            }
        }

        all_alerts
    }

    /// Get metrics for specific venue/symbol
    pub fn get_metrics(&self, venue_id: VenueId, symbol: [u8; 12]) -> Option<BehaviorMetrics> {
        let trackers = self.venue_trackers.read();
        trackers.get(&venue_id).and_then(|tracker| {
            tracker.symbol_metrics.get(&symbol).map(|m| BehaviorMetrics {
                orders_submitted: m.orders_submitted,
                orders_cancelled: m.orders_cancelled,
                orders_filled: m.orders_filled,
                orders_rejected: m.orders_rejected,
                orders_modified: m.orders_modified,
                order_to_trade_ratio: m.get_order_to_trade_ratio(),
                cancellation_rate: m.get_cancellation_rate(),
            })
        })
    }

    /// Enable/disable detector
    #[inline]
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Release);
    }

    /// Get statistics
    pub fn get_stats(&self) -> BehaviorStats {
        BehaviorStats {
            venues_tracked: self.venue_trackers.read().len(),
            events_processed: self.events_processed.load(Ordering::Relaxed),
            alerts_generated: self.alerts_generated.load(Ordering::Relaxed),
            enabled: self.enabled.load(Ordering::Acquire),
        }
    }
}

/// Behavior metrics snapshot
#[derive(Debug, Clone, Default)]
pub struct BehaviorMetrics {
    pub orders_submitted: u64,
    pub orders_cancelled: u64,
    pub orders_filled: u64,
    pub orders_rejected: u64,
    pub orders_modified: u64,
    pub order_to_trade_ratio: f64,
    pub cancellation_rate: f64,
}

/// Behavior statistics
#[derive(Debug, Clone, Default)]
pub struct BehaviorStats {
    pub venues_tracked: usize,
    pub events_processed: u64,
    pub alerts_generated: u64,
    pub enabled: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detector_creation() {
        let venues = vec![VenueId::Nasdaq];
        let detector = BehaviorAnomalyDetector::new(&venues);
        
        assert!(detector.enabled.load(Ordering::Acquire));
        assert_eq!(detector.get_stats().venues_tracked, 1);
    }

    #[test]
    fn test_event_recording() {
        let venues = vec![VenueId::Nasdaq];
        let detector = BehaviorAnomalyDetector::new(&venues);
        
        let symbol = *b"AAPL        ";
        detector.record_event(VenueId::Nasdaq, symbol, BehaviorEvent::OrderSubmitted);
        detector.record_event(VenueId::Nasdaq, symbol, BehaviorEvent::OrderFilled);
        
        let metrics = detector.get_metrics(VenueId::Nasdaq, symbol);
        assert!(metrics.is_some());
        let m = metrics.unwrap();
        assert_eq!(m.orders_submitted, 1);
        assert_eq!(m.orders_filled, 1);
    }

    #[test]
    fn test_order_to_trade_ratio_calculation() {
        let mut metrics = SymbolMetrics::new();
        metrics.orders_submitted = 100;
        metrics.orders_filled = 10;
        
        assert_eq!(metrics.get_order_to_trade_ratio(), 10.0);
    }
}
