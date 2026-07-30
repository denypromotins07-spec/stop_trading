//! Manipulation Detection Module Root
//! Feeds alerts into the global kill switch and alpha engines.

pub mod stop_hunt;
pub mod spoofing;

pub use stop_hunt::{
    StopHuntDetector,
    StopHuntSignal,
    StopHuntPattern,
    PriceBar,
    StopHuntStats,
};

pub use spoofing::{
    SpoofingDetector,
    SpoofingSignal,
    SpoofingPattern,
    OrderLifecycle,
    SpoofingStats,
};

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};

const CACHE_LINE_SIZE: usize = 64;

#[repr(C, align(64))]
struct CachePadded<T> {
    data: T,
    _pad: [u8; CACHE_LINE_SIZE],
}

impl<T: Default> Default for CachePadded<T> {
    fn default() -> Self {
        Self {
            data: T::default(),
            _pad: [0u8; CACHE_LINE_SIZE],
        }
    }
}

/// Alert severity level
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AlertSeverity {
    Low = 1,
    Medium = 2,
    High = 3,
    Critical = 4,
}

/// Unified manipulation alert
#[derive(Debug, Clone, Copy)]
pub struct ManipulationAlert {
    pub alert_type: ManipulationType,
    pub severity: AlertSeverity,
    pub timestamp_ns: u64,
    pub price_level: i64,
    pub confidence: f64,
    pub description: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ManipulationType {
    StopHunt,
    Spoofing,
    Layering,
    WashTrading,
    QuoteStuffing,
    MomentumIgnition,
}

/// Combined manipulation detection engine
pub struct ManipulationEngine {
    /// Stop-hunt detector
    stop_hunt_detector: StopHuntDetector,
    /// Spoofing detector
    spoofing_detector: SpoofingDetector,
    /// Total alerts generated
    total_alerts: CachePadded<AtomicU64>,
    /// Critical alerts count
    critical_alerts: CachePadded<AtomicU64>,
    /// Kill switch triggered flag
    kill_switch_triggered: CachePadded<AtomicBool>,
    /// Active flag
    is_active: CachePadded<AtomicBool>,
}

impl ManipulationEngine {
    pub fn new(
        min_touches: u32,
        sweep_threshold_ticks: i64,
        lifetime_threshold_us: u64,
        size_threshold: u64,
    ) -> Self {
        Self {
            stop_hunt_detector: StopHuntDetector::new(min_touches, sweep_threshold_ticks),
            spoofing_detector: SpoofingDetector::new(lifetime_threshold_us, size_threshold),
            total_alerts: CachePadded::default(),
            critical_alerts: CachePadded::default(),
            kill_switch_triggered: CachePadded::default(),
            is_active: CachePadded::new(AtomicBool::new(true)),
        }
    }

    /// Analyze price bars for stop-hunt patterns
    pub fn analyze_price_action(&self, bars: &[PriceBar]) -> Option<ManipulationAlert> {
        if !self.is_active.data.load(Ordering::Acquire) {
            return None;
        }

        self.stop_hunt_detector.analyze(bars).map(|signal| {
            self.total_alerts.data.fetch_add(1, Ordering::AcqRel);
            
            let severity = if signal.confidence > 0.8 {
                self.critical_alerts.data.fetch_add(1, Ordering::AcqRel);
                AlertSeverity::High
            } else if signal.confidence > 0.6 {
                AlertSeverity::Medium
            } else {
                AlertSeverity::Low
            };

            ManipulationAlert {
                alert_type: ManipulationType::StopHunt,
                severity,
                timestamp_ns: signal.timestamp_ns,
                price_level: signal.hunt_price,
                confidence: signal.confidence,
                description: match signal.pattern_type {
                    StopHuntPattern::EqualHighsSweep => "Equal highs swept - long stops targeted",
                    StopHuntPattern::EqualLowsSweep => "Equal lows swept - short stops targeted",
                    StopHuntPattern::ResistanceGrab => "Liquidity grab above resistance",
                    StopHuntPattern::SupportGrab => "Liquidity grab below support",
                    StopHuntPattern::WickReversal => "Wick reversal after liquidity sweep",
                },
            }
        })
    }

    /// Analyze order lifecycle for spoofing
    pub fn analyze_order(&self, lifecycle: OrderLifecycle) -> Option<ManipulationAlert> {
        if !self.is_active.data.load(Ordering::Acquire) {
            return None;
        }

        self.spoofing_detector.analyze_order(lifecycle).map(|signal| {
            self.total_alerts.data.fetch_add(1, Ordering::AcqRel);

            let severity = if signal.confidence > 0.8 {
                self.critical_alerts.data.fetch_add(1, Ordering::AcqRel);
                AlertSeverity::High
            } else if signal.confidence > 0.6 {
                AlertSeverity::Medium
            } else {
                AlertSeverity::Low
            };

            ManipulationAlert {
                alert_type: match signal.pattern_type {
                    SpoofingPattern::FlashOrder => ManipulationType::Spoofing,
                    SpoofingPattern::Layering => ManipulationType::Layering,
                    _ => ManipulationType::Spoofing,
                },
                severity,
                timestamp_ns: signal.timestamp_ns,
                price_level: signal.price_level,
                confidence: signal.confidence,
                description: match signal.pattern_type {
                    SpoofingPattern::FlashOrder => "Flash order detected - rapid placement/cancellation",
                    SpoofingPattern::Layering => "Layering pattern detected - fake depth on multiple levels",
                    SpoofingPattern::WalkingBook => "Walking book pattern - orders moved away from touch",
                    SpoofingPattern::MirrorOrders => "Mirror orders detected - potential wash trading",
                },
            }
        })
    }

    /// Batch analyze orders for layering
    pub fn analyze_layering(&self, lifecycles: &[OrderLifecycle]) -> Option<ManipulationAlert> {
        if !self.is_active.data.load(Ordering::Acquire) {
            return None;
        }

        self.spoofing_detector.analyze_layering(lifecycles).map(|signal| {
            self.total_alerts.data.fetch_add(1, Ordering::AcqRel);
            self.critical_alerts.data.fetch_add(1, Ordering::AcqRel);

            ManipulationAlert {
                alert_type: ManipulationType::Layering,
                severity: AlertSeverity::High,
                timestamp_ns: signal.timestamp_ns,
                price_level: signal.price_level,
                confidence: signal.confidence,
                description: "Significant layering detected - coordinated fake liquidity",
            }
        })
    }

    /// Check if kill switch should be triggered
    pub fn check_kill_switch(&self, alert: &ManipulationAlert) -> bool {
        // Trigger kill switch on critical alerts
        if alert.severity == AlertSeverity::Critical {
            self.kill_switch_triggered.data.store(true, Ordering::Release);
            return true;
        }

        // Also trigger if too many high-severity alerts in quick succession
        let critical_count = self.critical_alerts.data.load(Ordering::Acquire);
        if critical_count >= 5 {
            self.kill_switch_triggered.data.store(true, Ordering::Release);
            return true;
        }

        false
    }

    /// Get filtered book volume (excluding detected spoofed orders)
    pub fn get_filtered_volume(&self, raw_volume: u64, side_is_bid: bool) -> u64 {
        // Get spoofing stats to estimate suspicious ratio
        let stats = self.spoofing_detector.get_stats();
        
        // Use detection rate as proxy for suspicious activity level
        let suspicious_ratio = stats.detection_rate;
        
        self.spoofing_detector.get_real_volume_estimate(raw_volume, suspicious_ratio)
    }

    /// Get combined statistics
    pub fn get_stats(&self) -> ManipulationStats {
        let stop_hunt_stats = self.stop_hunt_detector.get_stats();
        let spoofing_stats = self.spoofing_detector.get_stats();

        ManipulationStats {
            total_alerts: self.total_alerts.data.load(Ordering::Acquire),
            critical_alerts: self.critical_alerts.data.load(Ordering::Acquire),
            kill_switch_triggered: self.kill_switch_triggered.data.load(Ordering::Acquire),
            stop_hunt_signals: stop_hunt_stats.signals_detected,
            spoofing_signals: spoofing_stats.signals_detected,
            spoofing_detection_rate: spoofing_stats.detection_rate,
            is_active: self.is_active.data.load(Ordering::Acquire),
        }
    }

    /// Reset kill switch
    #[inline]
    pub fn reset_kill_switch(&self) {
        self.kill_switch_triggered.data.store(false, Ordering::Release);
        self.critical_alerts.data.store(0, Ordering::Release);
    }

    #[inline]
    pub fn set_active(&self, active: bool) {
        self.is_active.data.store(active, Ordering::Release);
        self.stop_hunt_detector.set_active(active);
        self.spoofing_detector.set_active(active);
    }

    #[inline]
    pub fn is_active(&self) -> bool {
        self.is_active.data.load(Ordering::Acquire)
    }

    pub fn reset(&self) {
        self.total_alerts.data.store(0, Ordering::Release);
        self.critical_alerts.data.store(0, Ordering::Release);
        self.reset_kill_switch();
        self.stop_hunt_detector.reset();
        self.spoofing_detector.reset();
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ManipulationStats {
    pub total_alerts: u64,
    pub critical_alerts: u64,
    pub kill_switch_triggered: bool,
    pub stop_hunt_signals: u64,
    pub spoofing_signals: u64,
    pub spoofing_detection_rate: f64,
    pub is_active: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manipulation_engine_stop_hunt() {
        let engine = ManipulationEngine::new(2, 10, 1000, 1000);

        let bars = vec![
            PriceBar { timestamp_ns: 1000, open: 10000, high: 10050, low: 9990, close: 10020, volume: 100 },
            PriceBar { timestamp_ns: 2000, open: 10020, high: 10050, low: 10000, close: 10030, volume: 100 },
            PriceBar { timestamp_ns: 3000, open: 10030, high: 10055, low: 10020, close: 10025, volume: 150 },
        ];

        let alert = engine.analyze_price_action(&bars);
        assert!(alert.is_some());
        
        let alert = alert.unwrap();
        assert_eq!(alert.alert_type, ManipulationType::StopHunt);
    }

    #[test]
    fn test_manipulation_engine_spoofing() {
        let engine = ManipulationEngine::new(2, 10, 1000, 1000);

        let lifecycle = OrderLifecycle {
            order_id: 1,
            price: 10000,
            quantity: 5000,
            is_bid: true,
            placement_time_ns: 1000000,
            cancellation_time_ns: 1000500,
            was_modified: false,
        };

        let alert = engine.analyze_order(lifecycle);
        assert!(alert.is_some());
        assert_eq!(alert.unwrap().alert_type, ManipulationType::Spoofing);
    }

    #[test]
    fn test_kill_switch_trigger() {
        let engine = ManipulationEngine::new(2, 10, 1000, 1000);

        let alert = ManipulationAlert {
            alert_type: ManipulationType::StopHunt,
            severity: AlertSeverity::Critical,
            timestamp_ns: 1000,
            price_level: 10000,
            confidence: 0.95,
            description: "Test critical alert",
        };

        let triggered = engine.check_kill_switch(&alert);
        assert!(triggered);

        let stats = engine.get_stats();
        assert!(stats.kill_switch_triggered);
    }
}
