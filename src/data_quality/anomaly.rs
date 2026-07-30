//! Real-time Tick Anomaly Detector
//! 
//! Identifies stale quotes, crossed books, and API glitches.
//! Quarantines toxic data feeds instantly to prevent trading on phantom liquidity.

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::time::Instant;
use dashmap::DashMap;

/// Maximum ticks stored per symbol for anomaly detection
const MAX_TICKS_PER_SYMBOL: usize = 1000;

/// Anomaly types detected
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AnomalyType {
    /// Stale quote (no update for too long)
    StaleQuote,
    /// Crossed book (bid >= ask)
    CrossedBook,
    /// Price spike (> threshold move)
    PriceSpike,
    /// Zero size quote
    ZeroSize,
    /// Out of sequence timestamp
    OutOfSequence,
    /// Latency spike
    LatencySpike,
    /// Invalid price (negative or NaN)
    InvalidPrice,
}

/// Anomaly alert
#[derive(Debug, Clone)]
pub struct AnomalyAlert {
    /// Symbol
    pub symbol: String,
    /// Venue/source
    pub venue: String,
    /// Anomaly type
    pub anomaly_type: AnomalyType,
    /// Severity (1-5)
    pub severity: u8,
    /// Description
    pub description: String,
    /// Timestamp
    pub timestamp_ns: u64,
    /// Should quarantine feed
    pub quarantine_recommended: bool,
}

/// Quote data
#[derive(Debug, Clone)]
pub struct Quote {
    /// Bid price
    pub bid: f64,
    /// Ask price
    pub ask: f64,
    /// Bid size
    pub bid_size: f64,
    /// Ask size
    pub ask_size: f64,
    /// Timestamp
    pub timestamp_ns: u64,
    /// Received timestamp
    pub received_ns: u64,
}

/// Per-symbol anomaly tracker
struct SymbolAnomalyTracker {
    /// Recent ticks (ring buffer)
    ticks: [Option<Quote>; MAX_TICKS_PER_SYMBOL],
    /// Head position
    head: usize,
    /// Count
    count: usize,
    /// Last valid price
    last_valid_price: f64,
    /// Last update timestamp
    last_update_ns: u64,
    /// Consecutive anomalies
    consecutive_anomalies: u32,
    /// Is quarantined
    is_quarantined: bool,
}

impl SymbolAnomalyTracker {
    fn new() -> Self {
        Self {
            ticks: [None; MAX_TICKS_PER_SYMBOL],
            head: 0,
            count: 0,
            last_valid_price: 0.0,
            last_update_ns: 0,
            consecutive_anomalies: 0,
            is_quarantined: false,
        }
    }

    fn add_tick(&mut self, quote: &Quote) {
        self.ticks[self.head] = Some(quote.clone());
        self.head = (self.head + 1) % MAX_TICKS_PER_SYMBOL;
        if self.count < MAX_TICKS_PER_SYMBOL {
            self.count += 1;
        }
        self.last_update_ns = quote.timestamp_ns;
    }

    fn get_recent_ticks(&self, n: usize) -> Vec<&Quote> {
        let mut result = Vec::new();
        let start = if self.head >= n {
            self.head - n
        } else {
            MAX_TICKS_PER_SYMBOL - (n - self.head)
        };

        for i in 0..n.min(self.count) {
            let idx = (start + i) % MAX_TICKS_PER_SYMBOL;
            if let Some(ref tick) = self.ticks[idx] {
                result.push(tick);
            }
        }

        result
    }
}

impl Default for SymbolAnomalyTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Lock-free anomaly detection engine
pub struct AnomalyDetector {
    /// Per-symbol trackers
    trackers: DashMap<String, SymbolAnomalyTracker>,
    /// Stale threshold in milliseconds
    stale_threshold_ms: u64,
    /// Price spike threshold in basis points
    spike_threshold_bps: u16,
    /// Latency threshold in microseconds
    latency_threshold_us: u64,
    /// Alerts generated
    alerts_generated: AtomicU64,
    /// Quarantined feeds
    quarantined_feeds: DashMap<String, u64>,
    /// Is detector active
    is_active: AtomicBool,
}

impl AnomalyDetector {
    pub fn new(
        stale_threshold_ms: u64,
        spike_threshold_bps: u16,
        latency_threshold_us: u64,
    ) -> Self {
        Self {
            trackers: DashMap::new(),
            stale_threshold_ms,
            spike_threshold_bps,
            latency_threshold_us,
            alerts_generated: AtomicU64::new(0),
            quarantined_feeds: DashMap::new(),
            is_active: AtomicBool::new(true),
        }
    }

    /// Check a quote for anomalies
    pub fn check_quote(&self, symbol: &str, venue: &str, quote: &Quote) -> Vec<AnomalyAlert> {
        if !self.is_active.load(Ordering::Relaxed) {
            return Vec::new();
        }

        // Check if feed is quarantined
        if self.quarantined_feeds.contains_key(&format!("{}:{}", symbol, venue)) {
            return vec![AnomalyAlert {
                symbol: symbol.to_string(),
                venue: venue.to_string(),
                anomaly_type: AnomalyType::StaleQuote,
                severity: 5,
                description: "Feed is quarantined".to_string(),
                timestamp_ns: Instant::now().duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default().as_nanos() as u64,
                quarantine_recommended: true,
            }];
        }

        let mut alerts = Vec::new();
        let now_ns = Instant::now().duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default().as_nanos() as u64;

        // Get or create tracker
        let mut tracker = self.trackers.entry(symbol.to_string()).or_insert_with(SymbolAnomalyTracker::new);

        // Check for invalid prices
        if quote.bid <= 0.0 || quote.ask <= 0.0 || quote.bid.is_nan() || quote.ask.is_nan() {
            alerts.push(AnomalyAlert {
                symbol: symbol.to_string(),
                venue: venue.to_string(),
                anomaly_type: AnomalyType::InvalidPrice,
                severity: 5,
                description: format!("Invalid price: bid={}, ask={}", quote.bid, quote.ask),
                timestamp_ns: now_ns,
                quarantine_recommended: true,
            });
            tracker.consecutive_anomalies += 1;
        }

        // Check for crossed book
        if quote.bid >= quote.ask && quote.bid > 0.0 && quote.ask > 0.0 {
            alerts.push(AnomalyAlert {
                symbol: symbol.to_string(),
                venue: venue.to_string(),
                anomaly_type: AnomalyType::CrossedBook,
                severity: 4,
                description: format!("Crossed book: bid={} >= ask={}", quote.bid, quote.ask),
                timestamp_ns: now_ns,
                quarantine_recommended: true,
            });
            tracker.consecutive_anomalies += 1;
        }

        // Check for zero size
        if quote.bid_size <= 0.0 || quote.ask_size <= 0.0 {
            alerts.push(AnomalyAlert {
                symbol: symbol.to_string(),
                venue: venue.to_string(),
                anomaly_type: AnomalyType::ZeroSize,
                severity: 2,
                description: "Zero size quote".to_string(),
                timestamp_ns: now_ns,
                quarantine_recommended: false,
            });
        }

        // Check for stale quote
        let age_ms = (now_ns.saturating_sub(quote.timestamp_ns)) / 1_000_000;
        if age_ms > self.stale_threshold_ms {
            alerts.push(AnomalyAlert {
                symbol: symbol.to_string(),
                venue: venue.to_string(),
                anomaly_type: AnomalyType::StaleQuote,
                severity: 3,
                description: format!("Stale quote: {}ms old", age_ms),
                timestamp_ns: now_ns,
                quarantine_recommended: age_ms > self.stale_threshold_ms * 10,
            });
            tracker.consecutive_anomalies += 1;
        }

        // Check for latency spike
        let latency_us = (now_ns.saturating_sub(quote.received_ns)) / 1000;
        if latency_us > self.latency_threshold_us {
            alerts.push(AnomalyAlert {
                symbol: symbol.to_string(),
                venue: venue.to_string(),
                anomaly_type: AnomalyType::LatencySpike,
                severity: 2,
                description: format!("High latency: {}us", latency_us),
                timestamp_ns: now_ns,
                quarantine_recommended: false,
            });
        }

        // Check for price spike
        let mid_price = (quote.bid + quote.ask) / 2.0;
        if tracker.last_valid_price > 0.0 && mid_price > 0.0 {
            let change_bps = ((mid_price - tracker.last_valid_price) / tracker.last_valid_price * 10000.0).abs() as u16;
            if change_bps > self.spike_threshold_bps {
                alerts.push(AnomalyAlert {
                    symbol: symbol.to_string(),
                    venue: venue.to_string(),
                    anomaly_type: AnomalyType::PriceSpike,
                    severity: 4,
                    description: format!("Price spike: {} bps", change_bps),
                    timestamp_ns: now_ns,
                    quarantine_recommended: change_bps > self.spike_threshold_bps * 5,
                });
                tracker.consecutive_anomalies += 1;
            }
        }

        // Update tracker
        if alerts.is_empty() || alerts.iter().all(|a| a.severity < 4) {
            tracker.add_tick(quote);
            if mid_price > 0.0 {
                tracker.last_valid_price = mid_price;
            }
            tracker.consecutive_anomalies = 0;
        }

        // Check for quarantine threshold
        if tracker.consecutive_anomalies >= 5 {
            let key = format!("{}:{}", symbol, venue);
            self.quarantined_feeds.insert(key, now_ns);
            tracker.is_quarantined = true;

            alerts.push(AnomalyAlert {
                symbol: symbol.to_string(),
                venue: venue.to_string(),
                anomaly_type: AnomalyType::StaleQuote,
                severity: 5,
                description: "Feed quarantined due to consecutive anomalies".to_string(),
                timestamp_ns: now_ns,
                quarantine_recommended: true,
            });
        }

        // Update alert count
        for _ in 0..alerts.len() {
            self.alerts_generated.fetch_add(1, Ordering::Relaxed);
        }

        alerts
    }

    /// Check if a feed is quarantined
    pub fn is_quarantined(&self, symbol: &str, venue: &str) -> bool {
        let key = format!("{}:{}", symbol, venue);
        self.quarantined_feeds.contains_key(&key)
    }

    /// Release a quarantined feed
    pub fn release_feed(&self, symbol: &str, venue: &str) {
        let key = format!("{}:{}", symbol, venue);
        self.quarantined_feeds.remove(&key);
        
        if let Some(mut tracker) = self.trackers.get_mut(&symbol.to_string()) {
            tracker.is_quarantined = false;
            tracker.consecutive_anomalies = 0;
        }
    }

    /// Get alerts count
    pub fn get_alert_count(&self) -> u64 {
        self.alerts_generated.load(Ordering::Relaxed)
    }

    /// Get quarantined feeds count
    pub fn get_quarantine_count(&self) -> usize {
        self.quarantined_feeds.len()
    }

    /// Deactivate detector
    pub fn deactivate(&self) {
        self.is_active.store(false, Ordering::Relaxed);
    }

    /// Activate detector
    pub fn activate(&self) {
        self.is_active.store(true, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crossed_book_detection() {
        let detector = AnomalyDetector::new(5000, 1000, 10000);

        let quote = Quote {
            bid: 100.0,
            ask: 99.0, // Crossed!
            bid_size: 10.0,
            ask_size: 10.0,
            timestamp_ns: 1000000000,
            received_ns: 1000000000,
        };

        let alerts = detector.check_quote("BTCUSDT", "binance", &quote);
        
        assert!(!alerts.is_empty(), "Should detect crossed book");
        assert!(alerts.iter().any(|a| a.anomaly_type == AnomalyType::CrossedBook));
    }

    #[test]
    fn test_stale_quote_detection() {
        let detector = AnomalyDetector::new(100, 1000, 10000); // 100ms stale threshold

        let now = Instant::now().duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default().as_nanos() as u64;

        let quote = Quote {
            bid: 50000.0,
            ask: 50001.0,
            bid_size: 1.0,
            ask_size: 1.0,
            timestamp_ns: now - 200_000_000, // 200ms ago
            received_ns: now - 200_000_000,
        };

        let alerts = detector.check_quote("BTCUSDT", "binance", &quote);
        
        assert!(alerts.iter().any(|a| a.anomaly_type == AnomalyType::StaleQuote));
    }

    #[test]
    fn test_valid_quote() {
        let detector = AnomalyDetector::new(5000, 1000, 10000);

        let quote = Quote {
            bid: 50000.0,
            ask: 50001.0,
            bid_size: 1.0,
            ask_size: 1.0,
            timestamp_ns: 1000000000,
            received_ns: 1000000000,
        };

        let alerts = detector.check_quote("BTCUSDT", "binance", &quote);
        
        assert!(alerts.is_empty(), "Valid quote should not trigger alerts");
    }
}
