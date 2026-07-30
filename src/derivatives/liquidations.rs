//! Liquidation Cascade Detection Engine
//! 
//! Real-time liquidation cascade detection from exchange WebSocket streams.
//! Identifies forced selling/buying events to trigger momentum or mean-reversion algos.

use std::sync::atomic::{AtomicU64, AtomicBool, AtomicI64, Ordering};
use thiserror::Error;

/// Errors that can occur in liquidation detection
#[derive(Debug, Error)]
pub enum LiquidationError {
    #[error("Invalid liquidation data: {0}")]
    InvalidData(String),
    #[error("Overflow detected")]
    Overflow,
    #[error("Stream closed unexpectedly")]
    StreamClosed,
}

/// Single liquidation event
#[derive(Debug, Clone, Copy)]
pub struct LiquidationEvent {
    pub timestamp_ns: u64,
    pub symbol: [u8; 16],
    pub side: LiquidationSide,
    pub price: f64,
    pub quantity: f64,
    pub notional_value: f64,
}

impl LiquidationEvent {
    pub fn new(
        timestamp_ns: u64,
        symbol: &str,
        side: LiquidationSide,
        price: f64,
        quantity: f64,
    ) -> Result<Self, LiquidationError> {
        if price <= 0.0 || quantity <= 0.0 {
            return Err(LiquidationError::InvalidData(
                "Price and quantity must be positive".to_string(),
            ));
        }

        let mut bytes = [0u8; 16];
        let slice = symbol.as_bytes();
        let copy_len = slice.len().min(16);
        bytes[..copy_len].copy_from_slice(&slice[..copy_len]);

        Ok(Self {
            timestamp_ns,
            symbol: bytes,
            side,
            price,
            quantity,
            notional_value: price * quantity,
        })
    }

    pub fn symbol_str(&self) -> String {
        String::from_utf8_lossy(&self.symbol)
            .trim_end_matches('\0')
            .to_string()
    }
}

/// Side of liquidation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LiquidationSide {
    Long, // Long position liquidated (forced sell)
    Short, // Short position liquidated (forced buy)
}

/// Liquidation cascade signal
#[derive(Debug, Clone, Copy)]
pub struct CascadeSignal {
    pub direction: CascadeDirection,
    pub intensity: f64,      // 0.0 to 1.0
    pub total_notional: f64,
    pub event_count: u32,
    pub duration_ms: u64,
    pub recommended_action: CascadeAction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CascadeDirection {
    Downward, // Long liquidations causing price drop
    Upward,   // Short liquidations causing price rise
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CascadeAction {
    FollowMomentum,   // Trade with the cascade
    MeanReversion,    // Fade the cascade
    Wait,             // Too uncertain
}

/// Lock-free Liquidation Detector
pub struct LiquidationDetector {
    /// Running count of long liquidations
    long_count: AtomicU64,
    /// Running count of short liquidations
    short_count: AtomicU64,
    /// Total long liquidation notional (scaled by 1e6)
    long_notional: AtomicU64,
    /// Total short liquidation notional (scaled by 1e6)
    short_notional: AtomicU64,
    /// Last event timestamp
    last_event_ns: AtomicU64,
    /// Cascade threshold (number of events in window)
    cascade_threshold: AtomicU64,
    /// Window size in milliseconds
    window_size_ms: AtomicU64,
    /// Active flag
    active: AtomicBool,
    /// Price scale factor
    notional_scale: i64,
}

unsafe impl Send for LiquidationDetector {}
unsafe impl Sync for LiquidationDetector {}

impl LiquidationDetector {
    /// Create a new liquidation detector
    pub fn new(cascade_threshold: u64, window_size_ms: u64) -> Self {
        Self {
            long_count: AtomicU64::new(0),
            short_count: AtomicU64::new(0),
            long_notional: AtomicU64::new(0),
            short_notional: AtomicU64::new(0),
            last_event_ns: AtomicU64::new(0),
            cascade_threshold: AtomicU64::new(cascade_threshold),
            window_size_ms: AtomicU64::new(window_size_ms),
            active: AtomicBool::new(true),
            notional_scale: 1_000_000,
        }
    }

    /// Process a liquidation event (lock-free)
    pub fn process_event(&self, event: &LiquidationEvent) -> Result<Option<CascadeSignal>, LiquidationError> {
        if !self.active.load(Ordering::Relaxed) {
            return Ok(None);
        }

        // Update counters atomically
        match event.side {
            LiquidationSide::Long => {
                self.long_count.fetch_add(1, Ordering::Relaxed);
                let scaled_notional = (event.notional_value * self.notional_scale as f64) as u64;
                self.long_notional.fetch_add(scaled_notional, Ordering::Relaxed);
            }
            LiquidationSide::Short => {
                self.short_count.fetch_add(1, Ordering::Relaxed);
                let scaled_notional = (event.notional_value * self.notional_scale as f64) as u64;
                self.short_notional.fetch_add(scaled_notional, Ordering::Relaxed);
            }
        }

        self.last_event_ns.store(event.timestamp_ns, Ordering::Relaxed);

        // Check for cascade
        if let Some(signal) = self.check_cascade() {
            return Ok(Some(signal));
        }

        Ok(None)
    }

    /// Check if current liquidation activity constitutes a cascade
    fn check_cascade(&self) -> Option<CascadeSignal> {
        let threshold = self.cascade_threshold.load(Ordering::Relaxed);
        let long_cnt = self.long_count.load(Ordering::Relaxed);
        let short_cnt = self.short_count.load(Ordering::Relaxed);
        let total_cnt = long_cnt + short_cnt;

        if total_cnt < threshold {
            return None;
        }

        let long_not = self.long_notional.load(Ordering::Relaxed) as f64 / self.notional_scale as f64;
        let short_not = self.short_notional.load(Ordering::Relaxed) as f64 / self.notional_scale as f64;
        let total_notional = long_not + short_not;

        // Determine dominant direction
        let (direction, intensity) = if long_not > short_not * 2.0 {
            (CascadeDirection::Downward, (long_not / total_notional.max(0.001)).min(1.0))
        } else if short_not > long_not * 2.0 {
            (CascadeDirection::Upward, (short_not / total_notional.max(0.001)).min(1.0))
        } else {
            return None; // Mixed signals
        };

        // Calculate duration
        let window_ms = self.window_size_ms.load(Ordering::Relaxed);
        
        // Determine recommended action based on intensity
        let action = if intensity > 0.8 {
            CascadeAction::FollowMomentum
        } else if intensity > 0.5 {
            CascadeAction::MeanReversion
        } else {
            CascadeAction::Wait
        };

        Some(CascadeSignal {
            direction,
            intensity,
            total_notional,
            event_count: total_cnt as u32,
            duration_ms: window_ms,
            recommended_action: action,
        })
    }

    /// Get imbalance ratio (long / short)
    pub fn imbalance_ratio(&self) -> f64 {
        let long_cnt = self.long_count.load(Ordering::Relaxed);
        let short_cnt = self.short_count.load(Ordering::Relaxed);

        if short_cnt == 0 {
            if long_cnt == 0 {
                1.0
            } else {
                f64::MAX
            }
        } else {
            long_cnt as f64 / short_cnt as f64
        }
    }

    /// Get net liquidation pressure (-1.0 to 1.0)
    pub fn net_pressure(&self) -> f64 {
        let long_not = self.long_notional.load(Ordering::Relaxed) as f64;
        let short_not = self.short_notional.load(Ordering::Relaxed) as f64;
        let total = long_not + short_not;

        if total == 0.0 {
            0.0
        } else {
            (short_not - long_not) / total // Positive = upward pressure
        }
    }

    /// Reset counters (for new window)
    pub fn reset_window(&self) {
        self.long_count.store(0, Ordering::Relaxed);
        self.short_count.store(0, Ordering::Relaxed);
        self.long_notional.store(0, Ordering::Relaxed);
        self.short_notional.store(0, Ordering::Relaxed);
    }

    /// Update cascade threshold
    pub fn set_cascade_threshold(&self, threshold: u64) {
        self.cascade_threshold.store(threshold, Ordering::Relaxed);
    }

    /// Activate/deactivate detector
    pub fn set_active(&self, active: bool) {
        self.active.store(active, Ordering::Relaxed);
    }

    pub fn is_active(&self) -> bool {
        self.active.load(Ordering::Relaxed)
    }

    /// Get statistics
    pub fn stats(&self) -> LiquidationStats {
        LiquidationStats {
            long_count: self.long_count.load(Ordering::Relaxed),
            short_count: self.short_count.load(Ordering::Relaxed),
            long_notional: self.long_notional.load(Ordering::Relaxed) as f64 / self.notional_scale as f64,
            short_notional: self.short_notional.load(Ordering::Relaxed) as f64 / self.notional_scale as f64,
            net_pressure: self.net_pressure(),
        }
    }
}

impl Default for LiquidationDetector {
    fn default() -> Self {
        Self::new(10, 60000) // 10 events in 60 seconds
    }
}

/// Liquidation statistics
#[derive(Debug, Clone, Copy)]
pub struct LiquidationStats {
    pub long_count: u64,
    pub short_count: u64,
    pub long_notional: f64,
    pub short_notional: f64,
    pub net_pressure: f64,
}

/// Rolling window for liquidation analysis
pub struct LiquidationWindow {
    window_size_ms: u64,
    events: crossbeam::queue::SegQueue<LiquidationEvent>,
    max_events: usize,
}

impl LiquidationWindow {
    pub fn new(window_size_ms: u64, max_events: usize) -> Self {
        Self {
            window_size_ms,
            events: crossbeam::queue::SegQueue::new(),
            max_events,
        }
    }

    /// Add event to window
    pub fn add_event(&self, event: LiquidationEvent) {
        self.events.push(event);

        // Prune old events
        while self.events.len() > self.max_events {
            let _ = self.events.pop();
        }
    }

    /// Get events within time range
    pub fn get_events_in_range(&self, start_ns: u64, end_ns: u64) -> Vec<LiquidationEvent> {
        self.events.iter()
            .filter(|e| e.timestamp_ns >= start_ns && e.timestamp_ns <= end_ns)
            .cloned()
            .collect()
    }

    /// Calculate rolling notional sum
    pub fn rolling_notional_sum(&self, current_time_ns: u64) -> (f64, f64) {
        let window_start = current_time_ns.saturating_sub(self.window_size_ms * 1_000_000);
        
        let mut long_sum = 0.0;
        let mut short_sum = 0.0;

        for event in self.events.iter() {
            if event.timestamp_ns >= window_start {
                match event.side {
                    LiquidationSide::Long => long_sum += event.notional_value,
                    LiquidationSide::Short => short_sum += event.notional_value,
                }
            }
        }

        (long_sum, short_sum)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_liquidation_event() {
        let event = LiquidationEvent::new(1000, "BTCUSDT", LiquidationSide::Long, 50000.0, 1.0).unwrap();
        
        assert!((event.price - 50000.0).abs() < 0.001);
        assert!((event.quantity - 1.0).abs() < 0.001);
        assert!((event.notional_value - 50000.0).abs() < 0.001);
        assert_eq!(event.symbol_str(), "BTCUSDT");
    }

    #[test]
    fn test_detector_basic() {
        let detector = LiquidationDetector::new(5, 60000);
        
        for i in 0..6 {
            let event = LiquidationEvent::new(
                1000 + i * 1000,
                "BTCUSDT",
                LiquidationSide::Long,
                50000.0,
                1.0,
            ).unwrap();
            detector.process_event(&event).unwrap();
        }

        let stats = detector.stats();
        assert_eq!(stats.long_count, 6);
        assert!(detector.net_pressure() < 0.0); // Long liquidations = negative pressure
    }

    #[test]
    fn test_cascade_detection() {
        let detector = LiquidationDetector::new(3, 60000);
        
        // Generate long liquidation cascade
        for i in 0..5 {
            let event = LiquidationEvent::new(
                1000 + i * 1000,
                "BTCUSDT",
                LiquidationSide::Long,
                50000.0,
                10.0,
            ).unwrap();
            let signal = detector.process_event(&event).unwrap();
            
            if let Some(sig) = signal {
                assert_eq!(sig.direction, CascadeDirection::Downward);
                assert_eq!(sig.recommended_action, CascadeAction::FollowMomentum);
            }
        }
    }
}
