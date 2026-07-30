//! Wash Trade Detection Module
//! 
//! Internal wash-trade detector analyzing portfolio fills to prevent artificial volume inflation.
//! Logs and alerts on internal crossing events to refine routing logic and maintain strict regulatory compliance.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};
use crate::gateway::venue::VenueId;

/// Maximum fill history to retain per symbol (memory-conscious)
const MAX_FILL_HISTORY_PER_SYMBOL: usize = 1000;

/// Time window for wash trade detection (5 minutes default)
const WASH_TRADE_WINDOW_MS: u64 = 300_000;

/// Fill record for analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FillRecord {
    pub fill_id: u64,
    pub order_id: u64,
    pub symbol: [u8; 12],
    pub venue_id: VenueId,
    pub side: FillSide,
    pub price_ticks: i64,
    pub quantity: u64,
    pub timestamp_ns: u64,
    pub liquidity_type: LiquidityType,
    pub stp_group_id: Option<u32>,
    pub strategy_id: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum FillSide {
    Buy = 0,
    Sell = 1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum LiquidityType {
    Maker = 0,
    Taker = 1,
}

/// Wash trade detection result
#[derive(Debug, Clone)]
pub struct WashTradeAlert {
    pub alert_id: u64,
    pub symbol: [u8; 12],
    pub venue_id: VenueId,
    /// Pair of fills that constitute potential wash trade
    pub fill_pair: (u64, u64),  // (fill_id_1, fill_id_2)
    pub time_diff_ms: u64,
    pub matched_quantity: u64,
    pub severity: WashTradeSeverity,
    pub reason: &'static str,
    pub timestamp_ns: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum WashTradeSeverity {
    Low = 0,    // Potential, needs review
    Medium = 1, // Likely wash trade
    High = 2,   // Definite wash trade pattern
    Critical = 3, // Repeated violation pattern
}

/// Pattern types detected by wash trade analyzer
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WashPattern {
    /// Buy and sell at same price within short window
    SamePriceCrossing,
    /// Circular trading pattern (A->B->A)
    CircularTrading,
    /// Rapid back-and-forth trading
    FlipFlopping,
    /// Self-matching with different accounts
    AccountMatching,
}

/// Per-symbol fill tracker with circular buffer for memory efficiency
struct SymbolFillTracker {
    symbol: [u8; 12],
    fills: VecDeque<FillRecord>,
    max_fills: usize,
}

impl SymbolFillTracker {
    fn new(symbol: [u8; 12], max_fills: usize) -> Self {
        Self {
            symbol,
            fills: VecDeque::with_capacity(max_fills.min(500)),
            max_fills,
        }
    }

    fn add_fill(&mut self, fill: FillRecord) {
        if self.fills.len() >= self.max_fills {
            self.fills.pop_front();
        }
        self.fills.push_back(fill);
    }

    fn get_recent_fills(&self, window_ms: u64, current_time_ns: u64) -> Vec<&FillRecord> {
        let window_ns = window_ms * 1_000_000;
        let cutoff = current_time_ns.saturating_sub(window_ns);
        
        self.fills.iter().filter(|f| f.timestamp_ns >= cutoff).collect()
    }

    fn clear_old(&mut self, max_age_ms: u64, current_time_ns: u64) {
        let cutoff = current_time_ns.saturating_sub(max_age_ms * 1_000_000);
        while let Some(front) = self.fills.front() {
            if front.timestamp_ns < cutoff {
                self.fills.pop_front();
            } else {
                break;
            }
        }
    }
}

/// Main Wash Trade Detector
pub struct WashTradeDetector {
    /// Per-symbol fill trackers
    symbol_trackers: parking_lot::RwLock<HashMap<[u8; 12], SymbolFillTracker>>,
    /// Alert counter
    alert_counter: AtomicU64,
    /// Total fills processed
    fills_processed: AtomicU64,
    /// Wash trades detected
    wash_trades_detected: AtomicU64,
    /// Detector enabled flag
    enabled: AtomicBool,
    /// Maximum age of fills to consider (ms)
    max_fill_age_ms: u64,
    /// Minimum quantity ratio to flag (0.8 = 80% match)
    min_quantity_ratio: f64,
    /// Alert callback (could be connected to notification system)
    alert_callback: Option<Arc<dyn Fn(WashTradeAlert) + Send + Sync>>,
}

impl WashTradeDetector {
    pub fn new() -> Self {
        Self {
            symbol_trackers: parking_lot::RwLock::new(HashMap::new()),
            alert_counter: AtomicU64::new(0),
            fills_processed: AtomicU64::new(0),
            wash_trades_detected: AtomicU64::new(0),
            enabled: AtomicBool::new(true),
            max_fill_age_ms: WASH_TRADE_WINDOW_MS,
            min_quantity_ratio: 0.8,
            alert_callback: None,
        }
    }

    /// Set alert callback function
    pub fn set_alert_callback<F>(&mut self, callback: F)
    where
        F: Fn(WashTradeAlert) + Send + Sync + 'static,
    {
        self.alert_callback = Some(Arc::new(callback));
    }

    /// Record a fill for analysis
    pub fn record_fill(&self, fill: FillRecord) -> Option<WashTradeAlert> {
        if !self.enabled.load(Ordering::Acquire) {
            return None;
        }

        self.fills_processed.fetch_add(1, Ordering::Relaxed);

        let current_time_ns = fill.timestamp_ns;

        // Add to tracker
        {
            let mut trackers = self.symbol_trackers.write();
            let tracker = trackers
                .entry(fill.symbol)
                .or_insert_with(|| SymbolFillTracker::new(fill.symbol, MAX_FILL_HISTORY_PER_SYMBOL));
            tracker.add_fill(fill.clone());
        }

        // Check for wash trade patterns
        if let Some(alert) = self.check_wash_trade(&fill, current_time_ns) {
            self.wash_trades_detected.fetch_add(1, Ordering::Relaxed);
            
            // Trigger callback if set
            if let Some(ref callback) = self.alert_callback {
                callback(alert.clone());
            }
            
            return Some(alert);
        }

        None
    }

    /// Check if new fill creates wash trade pattern
    fn check_wash_trade(&self, new_fill: &FillRecord, current_time_ns: u64) -> Option<WashTradeAlert> {
        let trackers = self.symbol_trackers.read();
        let tracker = trackers.get(&new_fill.symbol)?;

        let recent_fills = tracker.get_recent_fills(self.max_fill_age_ms, current_time_ns);

        for existing_fill in recent_fills {
            if existing_fill.fill_id == new_fill.fill_id {
                continue;
            }

            // Check for opposite sides
            if existing_fill.side == new_fill.side {
                continue;
            }

            // Check for same venue
            if existing_fill.venue_id != new_fill.venue_id {
                continue;
            }

            // Check for same STP group (definite self-trade)
            let same_stp_group = match (existing_fill.stp_group_id, new_fill.stp_group_id) {
                (Some(a), Some(b)) => a == b,
                _ => false,
            };

            // Check quantity match
            let qty_ratio = (existing_fill.quantity as f64 / new_fill.quantity as f64).min(
                new_fill.quantity as f64 / existing_fill.quantity as f64
            );

            if qty_ratio < self.min_quantity_ratio {
                continue;
            }

            // Calculate time difference
            let time_diff_ns = current_time_ns.saturating_sub(existing_fill.timestamp_ns);
            let time_diff_ms = time_diff_ns / 1_000_000;

            // Determine severity
            let severity = if same_stp_group {
                WashTradeSeverity::Critical
            } else if time_diff_ms < 100 {
                WashTradeSeverity::High
            } else if time_diff_ms < 1000 {
                WashTradeSeverity::Medium
            } else {
                WashTradeSeverity::Low
            };

            // Determine reason
            let reason = if same_stp_group {
                "Same STP group self-match"
            } else if existing_fill.price_ticks == new_fill.price_ticks {
                "Same price crossing"
            } else if time_diff_ms < 100 {
                "Rapid flip-flop pattern"
            } else {
                "Potential wash trade pattern"
            };

            let alert_id = self.alert_counter.fetch_add(1, Ordering::Relaxed);

            return Some(WashTradeAlert {
                alert_id,
                symbol: new_fill.symbol,
                venue_id: new_fill.venue_id,
                fill_pair: (existing_fill.fill_id, new_fill.fill_id),
                time_diff_ms,
                matched_quantity: existing_fill.quantity.min(new_fill.quantity),
                severity,
                reason,
                timestamp_ns: current_time_ns,
            });
        }

        None
    }

    /// Analyze fills for complex patterns (circular trading, etc.)
    pub fn analyze_patterns(&self, symbol: &[u8; 12]) -> Vec<WashPattern> {
        let trackers = self.symbol_trackers.read();
        let tracker = match trackers.get(symbol) {
            Some(t) => t,
            None => return Vec::new(),
        };

        let mut patterns = Vec::new();
        let fills: Vec<&FillRecord> = tracker.fills.iter().collect();

        if fills.len() < 3 {
            return patterns;
        }

        // Check for flip-flopping (rapid alternation)
        let mut alternations = 0;
        for i in 1..fills.len().min(10) {
            if fills[i].side != fills[i-1].side {
                let time_diff = fills[i].timestamp_ns.saturating_sub(fills[i-1].timestamp_ns) / 1_000_000;
                if time_diff < 500 {
                    alternations += 1;
                }
            }
        }
        if alternations >= 4 {
            patterns.push(WashPattern::FlipFlopping);
        }

        // Check for same-price crossings
        let mut same_price_count = 0;
        for i in 0..fills.len() {
            for j in (i+1)..fills.len() {
                if fills[i].side != fills[j].side && fills[i].price_ticks == fills[j].price_ticks {
                    let time_diff = fills[j].timestamp_ns.saturating_sub(fills[i].timestamp_ns) / 1_000_000;
                    if time_diff < self.max_fill_age_ms {
                        same_price_count += 1;
                    }
                }
            }
        }
        if same_price_count >= 2 {
            patterns.push(WashPattern::SamePriceCrossing);
        }

        patterns
    }

    /// Get statistics
    pub fn get_stats(&self) -> WashTradeStats {
        WashTradeStats {
            fills_processed: self.fills_processed.load(Ordering::Relaxed),
            wash_trades_detected: self.wash_trades_detected.load(Ordering::Relaxed),
            symbols_tracked: self.symbol_trackers.read().len() as u64,
        }
    }

    /// Enable/disable detection
    #[inline]
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Release);
    }

    /// Clean up old fills periodically
    pub fn cleanup_old_fills(&self) {
        let current_time_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        let mut trackers = self.symbol_trackers.write();
        for tracker in trackers.values_mut() {
            tracker.clear_old(self.max_fill_age_ms * 2, current_time_ns);
        }
    }

    /// Get recent alerts for a symbol (for UI display)
    pub fn get_recent_alerts(&self, symbol: &[u8; 12], limit: usize) -> Vec<WashTradeAlert> {
        // This would typically query an alert store
        // For now, return empty - alerts should be stored separately
        Vec::new()
    }
}

/// Wash trade statistics
#[derive(Debug, Clone, Default)]
pub struct WashTradeStats {
    pub fills_processed: u64,
    pub wash_trades_detected: u64,
    pub symbols_tracked: u64,
}

impl Default for WashTradeDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_fill(
        fill_id: u64,
        symbol: [u8; 12],
        side: FillSide,
        price: i64,
        qty: u64,
        ts_ns: u64,
    ) -> FillRecord {
        FillRecord {
            fill_id,
            order_id: fill_id,
            symbol,
            venue_id: VenueId::Nasdaq,
            side,
            price_ticks: price,
            quantity: qty,
            timestamp_ns: ts_ns,
            liquidity_type: LiquidityType::Taker,
            stp_group_id: Some(1),
            strategy_id: 1,
        }
    }

    #[test]
    fn test_wash_trade_detection_same_stp() {
        let detector = WashTradeDetector::new();
        let symbol = *b"AAPL        ";
        let base_time = 1000000000000u64;

        // Record buy fill
        let buy_fill = create_test_fill(1, symbol, FillSide::Buy, 15000, 100, base_time);
        detector.record_fill(buy_fill);

        // Record matching sell fill from same STP group
        let sell_fill = create_test_fill(2, symbol, FillSide::Sell, 15000, 100, base_time + 50_000_000);
        let alert = detector.record_fill(sell_fill);

        assert!(alert.is_some());
        let alert = alert.unwrap();
        assert_eq!(alert.severity, WashTradeSeverity::Critical);
        assert_eq!(alert.matched_quantity, 100);
    }

    #[test]
    fn test_no_wash_trade_different_sides() {
        let detector = WashTradeDetector::new();
        let symbol = *b"AAPL        ";
        let base_time = 1000000000000u64;

        // Record two buys - no wash trade
        let buy1 = create_test_fill(1, symbol, FillSide::Buy, 15000, 100, base_time);
        detector.record_fill(buy1);

        let buy2 = create_test_fill(2, symbol, FillSide::Buy, 15000, 100, base_time + 50_000_000);
        let alert = detector.record_fill(buy2);

        assert!(alert.is_none());
    }

    #[test]
    fn test_detector_stats() {
        let detector = WashTradeDetector::new();
        assert_eq!(detector.get_stats().fills_processed, 0);
        assert_eq!(detector.get_stats().wash_trades_detected, 0);
    }
}
