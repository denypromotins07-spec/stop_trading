//! Cumulative Volume Delta (CVD) Tracker
//! 
//! Implements a lock-free CVD tracker to classify aggressive buy vs. sell volume
//! using atomic counters for real-time aggregation without blocking the main event loop.

use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};
use thiserror::Error;

/// Errors that can occur in CVD tracking
#[derive(Debug, Error)]
pub enum CvdError {
    #[error("Overflow detected in volume counter")]
    Overflow,
    #[error("Invalid tick data: {0}")]
    InvalidTickData(String),
}

/// Represents a single trade tick
#[derive(Debug, Clone, Copy)]
pub struct TradeTick {
    pub timestamp_ns: u64,
    pub price: f64,
    pub volume: f64,
    pub is_buyer_maker: bool, // true = aggressive sell, false = aggressive buy
}

impl TradeTick {
    pub fn new(timestamp_ns: u64, price: f64, volume: f64, is_buyer_maker: bool) -> Self {
        Self {
            timestamp_ns,
            price,
            volume,
            is_buyer_maker,
        }
    }

    pub fn from_exchange_data(
        timestamp_ms: u64,
        price: f64,
        volume: f64,
        is_buyer_maker: bool,
    ) -> Result<Self, CvdError> {
        if price <= 0.0 || volume <= 0.0 {
            return Err(CvdError::InvalidTickData(
                "Price and volume must be positive".to_string(),
            ));
        }

        let timestamp_ns = timestamp_ms
            .checked_mul(1_000_000)
            .ok_or(CvdError::Overflow)?;

        Ok(Self::new(timestamp_ns, price, volume, is_buyer_maker))
    }
}

/// Snapshot of CVD state at a point in time
#[derive(Debug, Clone, Copy)]
pub struct CvdSnapshot {
    pub timestamp_ns: u64,
    pub cumulative_buy_volume: f64,
    pub cumulative_sell_volume: f64,
    pub net_delta: f64,
    pub buy_count: u64,
    pub sell_count: u64,
}

impl CvdSnapshot {
    pub fn new(
        timestamp_ns: u64,
        cumulative_buy_volume: f64,
        cumulative_sell_volume: f64,
        buy_count: u64,
        sell_count: u64,
    ) -> Self {
        Self {
            timestamp_ns,
            cumulative_buy_volume,
            cumulative_sell_volume,
            net_delta: cumulative_buy_volume - cumulative_sell_volume,
            buy_count,
            sell_count,
        }
    }
}

/// Lock-free Cumulative Volume Delta tracker
/// 
/// Uses atomic counters to aggregate trade ticks in real-time without blocking.
/// The design prioritizes write performance for the hot path while allowing
/// occasional snapshots for analysis.
pub struct CvdTracker {
    /// Cumulative aggressive buy volume (scaled by 1e9 for integer storage)
    buy_volume_atomic: AtomicI64,
    /// Cumulative aggressive sell volume (scaled by 1e9 for integer storage)
    sell_volume_atomic: AtomicI64,
    /// Count of buy trades
    buy_count_atomic: AtomicU64,
    /// Count of sell trades
    sell_count_atomic: AtomicU64,
    /// Last update timestamp
    last_update_ns: AtomicU64,
    /// Volume scale factor (to convert f64 to i64)
    volume_scale: i64,
}

impl CvdTracker {
    /// Create a new CVD tracker with default scaling
    pub fn new() -> Self {
        Self {
            buy_volume_atomic: AtomicI64::new(0),
            sell_volume_atomic: AtomicI64::new(0),
            buy_count_atomic: AtomicU64::new(0),
            sell_count_atomic: AtomicU64::new(0),
            last_update_ns: AtomicU64::new(0),
            volume_scale: 1_000_000_000, // 1e9 scaling for nanounits
        }
    }

    /// Create a new CVD tracker with custom volume scaling
    pub fn with_scale(volume_scale: i64) -> Self {
        Self {
            buy_volume_atomic: AtomicI64::new(0),
            sell_volume_atomic: AtomicI64::new(0),
            buy_count_atomic: AtomicU64::new(0),
            sell_count_atomic: AtomicU64::new(0),
            last_update_ns: AtomicU64::new(0),
            volume_scale,
        }
    }

    /// Process a single trade tick (lock-free, atomic update)
    /// 
    /// Returns Ok(()) on success, or an error if the tick is invalid
    #[inline]
    pub fn process_tick(&self, tick: &TradeTick) -> Result<(), CvdError> {
        if tick.volume <= 0.0 {
            return Err(CvdError::InvalidTickData(
                "Volume must be positive".to_string(),
            ));
        }

        // Convert volume to scaled integer
        let scaled_volume = (tick.volume * self.volume_scale as f64) as i64;

        // Use fetch_add for lock-free atomic update
        if tick.is_buyer_maker {
            // Aggressive sell (taker hit the bid)
            self.sell_volume_atomic
                .fetch_add(scaled_volume, Ordering::Relaxed);
            self.sell_count_atomic.fetch_add(1, Ordering::Relaxed);
        } else {
            // Aggressive buy (taker lifted the ask)
            self.buy_volume_atomic
                .fetch_add(scaled_volume, Ordering::Relaxed);
            self.buy_count_atomic.fetch_add(1, Ordering::Relaxed);
        }

        // Update timestamp
        self.last_update_ns.store(tick.timestamp_ns, Ordering::Relaxed);

        Ok(())
    }

    /// Process multiple ticks in batch (still lock-free but more efficient)
    pub fn process_batch(&self, ticks: &[TradeTick]) -> Result<(), CvdError> {
        let mut buy_vol = 0i64;
        let mut sell_vol = 0i64;
        let mut buy_cnt = 0u64;
        let mut sell_cnt = 0u64;
        let mut last_ts = 0u64;

        for tick in ticks {
            if tick.volume <= 0.0 {
                return Err(CvdError::InvalidTickData(
                    "Volume must be positive".to_string(),
                ));
            }

            let scaled_volume = (tick.volume * self.volume_scale as f64) as i64;
            last_ts = tick.timestamp_ns;

            if tick.is_buyer_maker {
                sell_vol += scaled_volume;
                sell_cnt += 1;
            } else {
                buy_vol += scaled_volume;
                buy_cnt += 1;
            }
        }

        // Single atomic update per counter
        self.buy_volume_atomic.fetch_add(buy_vol, Ordering::Relaxed);
        self.sell_volume_atomic.fetch_add(sell_vol, Ordering::Relaxed);
        self.buy_count_atomic.fetch_add(buy_cnt, Ordering::Relaxed);
        self.sell_count_atomic.fetch_add(sell_cnt, Ordering::Relaxed);
        
        if last_ts > 0 {
            self.last_update_ns.store(last_ts, Ordering::Relaxed);
        }

        Ok(())
    }

    /// Get a snapshot of current CVD state (eventually consistent)
    pub fn snapshot(&self) -> CvdSnapshot {
        let buy_vol = self.buy_volume_atomic.load(Ordering::Relaxed);
        let sell_vol = self.sell_volume_atomic.load(Ordering::Relaxed);
        let buy_cnt = self.buy_count_atomic.load(Ordering::Relaxed);
        let sell_cnt = self.sell_count_atomic.load(Ordering::Relaxed);
        let ts = self.last_update_ns.load(Ordering::Relaxed);

        CvdSnapshot::new(
            ts,
            buy_vol as f64 / self.volume_scale as f64,
            sell_vol as f64 / self.volume_scale as f64,
            buy_cnt,
            sell_cnt,
        )
    }

    /// Get the net delta (buy volume - sell volume)
    #[inline]
    pub fn net_delta(&self) -> f64 {
        let buy_vol = self.buy_volume_atomic.load(Ordering::Relaxed);
        let sell_vol = self.sell_volume_atomic.load(Ordering::Relaxed);
        (buy_vol - sell_vol) as f64 / self.volume_scale as f64
    }

    /// Get the total volume processed
    #[inline]
    pub fn total_volume(&self) -> f64 {
        let buy_vol = self.buy_volume_atomic.load(Ordering::Relaxed);
        let sell_vol = self.sell_volume_atomic.load(Ordering::Relaxed);
        (buy_vol + sell_vol).abs() as f64 / self.volume_scale as f64
    }

    /// Reset all counters atomically
    pub fn reset(&self) {
        self.buy_volume_atomic.store(0, Ordering::Relaxed);
        self.sell_volume_atomic.store(0, Ordering::Relaxed);
        self.buy_count_atomic.store(0, Ordering::Relaxed);
        self.sell_count_atomic.store(0, Ordering::Relaxed);
        self.last_update_ns.store(0, Ordering::Relaxed);
    }

    /// Calculate CVD divergence from price action
    /// 
    /// Positive divergence: Price making lower lows but CVD making higher lows
    /// Negative divergence: Price making higher highs but CVD making lower highs
    pub fn calculate_divergence(
        &self,
        current_price: f64,
        previous_price: f64,
        previous_cvd: f64,
    ) -> DivergenceSignal {
        let current_cvd = self.net_delta();

        let price_up = current_price > previous_price;
        let cvd_up = current_cvd > previous_cvd;

        match (price_up, cvd_up) {
            (true, true) => DivergenceSignal::None,      // Confirmed uptrend
            (false, false) => DivergenceSignal::None,    // Confirmed downtrend
            (true, false) => DivergenceSignal::Bearish,  // Price up, CVD down (weakness)
            (false, true) => DivergenceSignal::Bullish,  // Price down, CVD up (strength)
        }
    }
}

impl Default for CvdTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Signal indicating potential divergence between price and volume flow
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DivergenceSignal {
    None,
    Bullish,
    Bearish,
}

/// CVD rate of change calculator for detecting acceleration/deceleration
pub struct CvdRateOfChange {
    prev_snapshot: std::cell::RefCell<Option<CvdSnapshot>>,
    lookback_window_ns: u64,
}

impl CvdRateOfChange {
    pub fn new(lookback_window_ms: u64) -> Self {
        Self {
            prev_snapshot: std::cell::RefCell::new(None),
            lookback_window_ns: lookback_window_ms * 1_000_000,
        }
    }

    /// Calculate the rate of change of CVD (delta per second)
    pub fn calculate_roc(&self, current: &CvdSnapshot) -> Option<f64> {
        let mut prev = self.prev_snapshot.borrow_mut();
        
        let roc = if let Some(prev_snap) = prev.as_ref() {
            let time_diff_s = (current.timestamp_ns - prev_snap.timestamp_ns) as f64 / 1_000_000_000.0;
            if time_diff_s > 0.0 {
                let cvd_diff = current.net_delta - prev_snap.net_delta;
                Some(cvd_diff / time_diff_s)
            } else {
                None
            }
        } else {
            None
        };

        // Store current snapshot if within lookback window or first snapshot
        if prev.is_none() {
            *prev = Some(*current);
        } else if let Some(prev_snap) = prev.as_ref() {
            if current.timestamp_ns - prev_snap.timestamp_ns >= self.lookback_window_ns {
                *prev = Some(*current);
            }
        }

        roc
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cvd_tracker_basic() {
        let tracker = CvdTracker::new();
        
        // Process aggressive buy
        let buy_tick = TradeTick::new(1000, 50000.0, 1.5, false);
        tracker.process_tick(&buy_tick).unwrap();
        
        // Process aggressive sell
        let sell_tick = TradeTick::new(2000, 50001.0, 2.0, true);
        tracker.process_tick(&sell_tick).unwrap();
        
        let snapshot = tracker.snapshot();
        assert!((snapshot.cumulative_buy_volume - 1.5).abs() < 1e-9);
        assert!((snapshot.cumulative_sell_volume - 2.0).abs() < 1e-9);
        assert!((snapshot.net_delta - (-0.5)).abs() < 1e-9);
    }

    #[test]
    fn test_cvd_batch_processing() {
        let tracker = CvdTracker::new();
        
        let ticks = vec![
            TradeTick::new(1000, 50000.0, 1.0, false),
            TradeTick::new(2000, 50001.0, 1.0, false),
            TradeTick::new(3000, 50002.0, 1.0, true),
        ];
        
        tracker.process_batch(&ticks).unwrap();
        
        let snapshot = tracker.snapshot();
        assert!((snapshot.cumulative_buy_volume - 2.0).abs() < 1e-9);
        assert!((snapshot.cumulative_sell_volume - 1.0).abs() < 1e-9);
        assert_eq!(snapshot.buy_count, 2);
        assert_eq!(snapshot.sell_count, 1);
    }

    #[test]
    fn test_divergence_detection() {
        let tracker = CvdTracker::new();
        
        // Simulate CVD accumulation
        tracker.process_tick(&TradeTick::new(1000, 50000.0, 5.0, false)).unwrap();
        
        // Price down but CVD up (bullish divergence)
        let signal = tracker.calculate_divergence(49900.0, 50000.0, 0.0);
        assert_eq!(signal, DivergenceSignal::Bullish);
    }
}
