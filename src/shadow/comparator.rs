//! Shadow Comparator Module
//! 
//! Real-time comparator measuring Shadow PnL vs Live PnL to detect execution degradation
//! or model drift. Automatically flags when live execution deviates significantly from
//! theoretical shadow performance.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};
use crate::gateway::venue::VenueId;
use super::engine::{ShadowPnL, ShadowEngine};

/// Comparison result between shadow and live performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonResult {
    pub symbol: [u8; 12],
    pub venue_id: VenueId,
    pub shadow_pnl: f64,
    pub live_pnl: f64,
    pub pnl_divergence: f64,
    pub divergence_bps: f64,
    pub shadow_trades: u64,
    pub live_trades: u64,
    pub fill_rate_shadow: f64,
    pub fill_rate_live: f64,
    pub avg_slippage_shadow: f64,
    pub avg_slippage_live: f64,
    pub timestamp_ns: u64,
    pub alert_level: AlertLevel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum AlertLevel {
    Normal = 0,
    Warning = 1,
    Critical = 2,
    Severe = 3,
}

impl AlertLevel {
    pub fn from_divergence_bps(divergence_bps: f64) -> Self {
        if divergence_bps.abs() < 5.0 {
            AlertLevel::Normal
        } else if divergence_bps.abs() < 15.0 {
            AlertLevel::Warning
        } else if divergence_bps.abs() < 30.0 {
            AlertLevel::Critical
        } else {
            AlertLevel::Severe
        }
    }
}

/// Historical comparison record for trend analysis
#[derive(Debug, Clone)]
struct HistoricalComparison {
    timestamp_ns: u64,
    pnl_divergence: f64,
    fill_rate_diff: f64,
}

/// Per-symbol comparator state
struct SymbolComparatorState {
    symbol: [u8; 12],
    /// Historical comparisons (circular buffer)
    history: VecDeque<HistoricalComparison>,
    max_history: usize,
    /// Cumulative shadow PnL
    cumulative_shadow_pnl: f64,
    /// Cumulative live PnL
    cumulative_live_pnl: f64,
    /// Last comparison timestamp
    last_comparison_ns: u64,
    /// Alert cooldown tracking
    last_alert_ns: u64,
}

impl SymbolComparatorState {
    fn new(symbol: [u8; 12]) -> Self {
        Self {
            symbol,
            history: VecDeque::with_capacity(100),
            max_history: 100,
            cumulative_shadow_pnl: 0.0,
            cumulative_live_pnl: 0.0,
            last_comparison_ns: 0,
            last_alert_ns: 0,
        }
    }

    fn add_comparison(&mut self, comparison: &HistoricalComparison) {
        if self.history.len() >= self.max_history {
            self.history.pop_front();
        }
        self.history.push_back(comparison.clone());
    }

    fn get_trend(&self) -> TrendAnalysis {
        if self.history.len() < 5 {
            return TrendAnalysis::InsufficientData;
        }

        let recent: Vec<f64> = self.history.iter().map(|h| h.pnl_divergence).collect();
        
        // Calculate simple linear trend
        let n = recent.len() as f64;
        let sum_x: f64 = (0..recent.len()).map(|i| i as f64).sum();
        let sum_y: f64 = recent.iter().sum();
        let sum_xy: f64 = recent.iter().enumerate().map(|(i, v)| i as f64 * v).sum();
        let sum_xx: f64 = (0..recent.len()).map(|i| (i as f64).powi(2)).sum();

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x.powi(2));

        if slope.abs() < 0.1 {
            TrendAnalysis::Stable
        } else if slope > 0.0 {
            TrendAnalysis::Diverging(slope)
        } else {
            TrendAnalysis::Converging(slope)
        }
    }
}

enum TrendAnalysis {
    InsufficientData,
    Stable,
    Diverging(f64),
    Converging(f64),
}

/// Live PnL tracker for comparison
pub struct LivePnLTracker {
    symbol: [u8; 12],
    fills: VecDeque<LiveFill>,
    position: i64,
    avg_entry: i64,
    realized_pnl: f64,
}

#[derive(Debug, Clone)]
struct LiveFill {
    timestamp_ns: u64,
    side: u8,  // 0 = buy, 1 = sell
    price: i64,
    quantity: u64,
}

impl LivePnLTracker {
    fn new(symbol: [u8; 12]) -> Self {
        Self {
            symbol,
            fills: VecDeque::with_capacity(1000),
            position: 0,
            avg_entry: 0,
            realized_pnl: 0.0,
        }
    }

    fn record_fill(&mut self, side: u8, price: i64, quantity: u64, timestamp_ns: u64) {
        self.fills.push_back(LiveFill {
            timestamp_ns,
            side,
            price,
            quantity,
        });

        // Update position and PnL
        match side {
            0 => {  // Buy
                if self.position >= 0 {
                    let total = (self.avg_entry as u64 * self.position as u64) as f64 
                        + (price as f64 * quantity as f64);
                    self.position += quantity as i64;
                    if self.position > 0 {
                        self.avg_entry = (total / self.position as f64) as i64;
                    }
                } else {
                    // Covering short - realize PnL
                    let pnl = (self.avg_entry - price) as f64 * quantity.min(-self.position as u64) as f64;
                    self.realized_pnl += pnl;
                    self.position += quantity as i64;
                }
            }
            1 => {  // Sell
                if self.position <= 0 {
                    let total = (self.avg_entry as u64 * (-self.position) as u64) as f64 
                        + (price as f64 * quantity as f64);
                    self.position -= quantity as i64;
                    if self.position < 0 {
                        self.avg_entry = (total / (-self.position) as f64) as i64;
                    }
                } else {
                    // Closing long - realize PnL
                    let pnl = (price - self.avg_entry) as f64 * quantity.min(self.position as u64) as f64;
                    self.realized_pnl += pnl;
                    self.position -= quantity as i64;
                }
            }
            _ => {}
        }

        // Limit history
        if self.fills.len() >= 10000 {
            self.fills.drain(0..5000);
        }
    }

    fn get_total_pnl(&self, current_price: i64) -> f64 {
        let unrealized = if self.position != 0 {
            (current_price - self.avg_entry) as f64 * self.position as f64
        } else {
            0.0
        };
        self.realized_pnl + unrealized
    }
}

/// Main Shadow Comparator
pub struct ShadowComparator {
    venue_id: VenueId,
    /// Per-symbol comparator states
    symbol_states: parking_lot::RwLock<HashMap<[u8; 12], SymbolComparatorState>>,
    /// Live PnL trackers
    live_trackers: parking_lot::RwLock<HashMap<[u8; 12], LivePnLTracker>>,
    /// Reference to shadow engine
    shadow_engine: Option<Arc<ShadowEngine>>,
    /// Comparator enabled
    enabled: AtomicBool,
    /// Total comparisons performed
    comparisons_count: AtomicU64,
    /// Alerts generated
    alerts_generated: AtomicU64,
    /// Divergence threshold in basis points
    warning_threshold_bps: f64,
    critical_threshold_bps: f64,
    /// Alert callback
    alert_callback: Option<Arc<dyn Fn(ComparisonResult) + Send + Sync>>,
}

impl ShadowComparator {
    pub fn new(venue_id: VenueId) -> Self {
        Self {
            venue_id,
            symbol_states: parking_lot::RwLock::new(HashMap::new()),
            live_trackers: parking_lot::RwLock::new(HashMap::new()),
            shadow_engine: None,
            enabled: AtomicBool::new(true),
            comparisons_count: AtomicU64::new(0),
            alerts_generated: AtomicU64::new(0),
            warning_threshold_bps: 10.0,
            critical_threshold_bps: 25.0,
            alert_callback: None,
        }
    }

    /// Set shadow engine reference
    pub fn set_shadow_engine(&mut self, engine: Arc<ShadowEngine>) {
        self.shadow_engine = Some(engine);
    }

    /// Set alert callback
    pub fn set_alert_callback<F>(&mut self, callback: F)
    where
        F: Fn(ComparisonResult) + Send + Sync + 'static,
    {
        self.alert_callback = Some(Arc::new(callback));
    }

    /// Record live fill for comparison
    pub fn record_live_fill(&self, symbol: &[u8; 12], side: u8, price: i64, quantity: u64) {
        let mut trackers = self.live_trackers.write();
        let tracker = trackers.entry(*symbol).or_insert_with(|| LivePnLTracker::new(*symbol));
        tracker.record_fill(side, price, quantity, 0);
    }

    /// Compare shadow vs live performance for a symbol
    pub fn compare(&self, symbol: &[u8; 12], current_price: i64) -> Option<ComparisonResult> {
        if !self.enabled.load(Ordering::Acquire) {
            return None;
        }

        self.comparisons_count.fetch_add(1, Ordering::Relaxed);

        // Get shadow PnL
        let shadow_pnl = if let Some(ref engine) = self.shadow_engine {
            engine.get_shadow_pnl(symbol)
        } else {
            ShadowPnL::default()
        };

        // Get live PnL
        let live_pnl = {
            let trackers = self.live_trackers.read();
            if let Some(tracker) = trackers.get(symbol) {
                tracker.get_total_pnl(current_price)
            } else {
                0.0
            }
        };

        let pnl_divergence = shadow_pnl.total_pnl - live_pnl;
        let notional = (current_price as f64 * (shadow_pnl.trades_count.max(1)) as f64).max(1.0);
        let divergence_bps = (pnl_divergence / notional) * 10000.0;

        let alert_level = AlertLevel::from_divergence_bps(divergence_bps);

        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        // Update state
        {
            let mut states = self.symbol_states.write();
            let state = states.entry(*symbol).or_insert_with(|| SymbolComparatorState::new(*symbol));
            
            state.cumulative_shadow_pnl += shadow_pnl.total_pnl;
            state.cumulative_live_pnl += live_pnl;
            state.last_comparison_ns = now_ns;

            state.add_comparison(&HistoricalComparison {
                timestamp_ns: now_ns,
                pnl_divergence,
                fill_rate_diff: 0.0,  // Could be calculated
            });

            // Check if alert should be triggered
            if alert_level != AlertLevel::Normal {
                let cooldown_ns = Duration::from_secs(60).as_nanos() as u64;
                if now_ns - state.last_alert_ns > cooldown_ns {
                    state.last_alert_ns = now_ns;
                    self.alerts_generated.fetch_add(1, Ordering::Relaxed);

                    let result = ComparisonResult {
                        symbol: *symbol,
                        venue_id: self.venue_id,
                        shadow_pnl: shadow_pnl.total_pnl,
                        live_pnl,
                        pnl_divergence,
                        divergence_bps,
                        shadow_trades: shadow_pnl.trades_count,
                        live_trades: 0,
                        fill_rate_shadow: shadow_pnl.win_rate,
                        fill_rate_live: 0.0,
                        avg_slippage_shadow: 0.0,
                        avg_slippage_live: 0.0,
                        timestamp_ns: now_ns,
                        alert_level,
                    };

                    if let Some(ref callback) = self.alert_callback {
                        callback(result.clone());
                    }

                    return Some(result);
                }
            }
        }

        Some(ComparisonResult {
            symbol: *symbol,
            venue_id: self.venue_id,
            shadow_pnl: shadow_pnl.total_pnl,
            live_pnl,
            pnl_divergence,
            divergence_bps,
            shadow_trades: shadow_pnl.trades_count,
            live_trades: 0,
            fill_rate_shadow: shadow_pnl.win_rate,
            fill_rate_live: 0.0,
            avg_slippage_shadow: 0.0,
            avg_slippage_live: 0.0,
            timestamp_ns: now_ns,
            alert_level,
        })
    }

    /// Get trend analysis for symbol
    pub fn get_trend(&self, symbol: &[u8; 12]) -> &'static str {
        let states = self.symbol_states.read();
        if let Some(state) = states.get(symbol) {
            match state.get_trend() {
                TrendAnalysis::InsufficientData => "Insufficient data",
                TrendAnalysis::Stable => "Stable",
                TrendAnalysis::Diverging(slope) => "Diverging",
                TrendAnalysis::Converging(slope) => "Converging",
            }
        } else {
            "No data"
        }
    }

    /// Enable/disable comparator
    #[inline]
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Release);
    }

    /// Get statistics
    pub fn get_stats(&self) -> ComparatorStats {
        let states = self.symbol_states.read();
        ComparatorStats {
            symbols_compared: states.len(),
            total_comparisons: self.comparisons_count.load(Ordering::Relaxed),
            alerts_generated: self.alerts_generated.load(Ordering::Relaxed),
            enabled: self.enabled.load(Ordering::Acquire),
        }
    }

    /// Configure thresholds
    pub fn configure_thresholds(&mut self, warning_bps: f64, critical_bps: f64) {
        self.warning_threshold_bps = warning_bps;
        self.critical_threshold_bps = critical_bps;
    }
}

/// Comparator statistics
#[derive(Debug, Clone, Default)]
pub struct ComparatorStats {
    pub symbols_compared: usize,
    pub total_comparisons: u64,
    pub alerts_generated: u64,
    pub enabled: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alert_level_thresholds() {
        assert_eq!(AlertLevel::from_divergence_bps(3.0), AlertLevel::Normal);
        assert_eq!(AlertLevel::from_divergence_bps(10.0), AlertLevel::Warning);
        assert_eq!(AlertLevel::from_divergence_bps(20.0), AlertLevel::Critical);
        assert_eq!(AlertLevel::from_divergence_bps(50.0), AlertLevel::Severe);
    }

    #[test]
    fn test_comparator_creation() {
        let comparator = ShadowComparator::new(VenueId::Nasdaq);
        assert!(comparator.enabled.load(Ordering::Acquire));
        assert_eq!(comparator.get_stats().symbols_compared, 0);
    }

    #[test]
    fn test_live_pnl_tracker() {
        let symbol = *b"AAPL        ";
        let mut tracker = LivePnLTracker::new(symbol);
        
        tracker.record_fill(0, 15000, 100, 0);  // Buy 100 @ 150
        tracker.record_fill(1, 15100, 100, 0);  // Sell 100 @ 151
        
        let pnl = tracker.get_total_pnl(15100);
        assert!(pnl > 0.0);  // Should have profit
    }
}
