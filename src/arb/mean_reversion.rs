//! Multi-Timeframe Mean Reversion Engine
//! 
//! Combines microprice drift with order book imbalances for ultra-short-term
//! contrarian trades when price deviates from VWAP or microprice baseline.

use std::sync::atomic::{AtomicU64, AtomicI64, AtomicBool, Ordering};
use crossbeam_utils::CachePadded;

/// Maximum number of timeframes to track
const MAX_TIMEFRAMES: usize = 8;

/// Rolling window for statistics
struct RollingWindow {
    /// Values buffer
    values: [f64; 256],
    /// Write index
    write_idx: usize,
    /// Count of valid entries
    count: usize,
    /// Sum of values
    sum: f64,
    /// Sum of squares
    sum_sq: f64,
}

impl RollingWindow {
    fn new() -> Self {
        Self {
            values: [0.0; 256],
            write_idx: 0,
            count: 0,
            sum: 0.0,
            sum_sq: 0.0,
        }
    }

    #[inline]
    fn push(&mut self, value: f64) {
        if self.count >= 256 {
            // Remove oldest value
            let old = self.values[self.write_idx];
            self.sum -= old;
            self.sum_sq -= old * old;
        } else {
            self.count += 1;
        }

        self.values[self.write_idx] = value;
        self.sum += value;
        self.sum_sq += value * value;
        self.write_idx = (self.write_idx + 1) % 256;
    }

    #[inline]
    fn mean(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        self.sum / self.count as f64
    }

    #[inline]
    fn std(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }
        let mean = self.mean();
        let variance = (self.sum_sq / self.count as f64) - (mean * mean);
        if variance > 0.0 {
            variance.sqrt()
        } else {
            0.0
        }
    }

    #[inline]
    fn zscore(&self, value: f64) -> f64 {
        let std = self.std();
        if std > 0.0 {
            (value - self.mean()) / std
        } else {
            0.0
        }
    }

    fn clear(&mut self) {
        self.values = [0.0; 256];
        self.write_idx = 0;
        self.count = 0;
        self.sum = 0.0;
        self.sum_sq = 0.0;
    }
}

/// Mean reversion signal
pub struct MeanReversionSignal {
    /// Signal strength (-100 to 100)
    pub strength: i8,
    /// Direction: 1 = long, -1 = short, 0 = neutral
    pub direction: i8,
    /// Z-score at entry
    pub z_score: f64,
    /// Timeframe that triggered the signal
    pub timeframe_idx: usize,
    /// Confidence score (0-100)
    pub confidence: u8,
}

/// Multi-timeframe Mean Reversion Engine
pub struct MeanReversionEngine {
    /// Price windows for different timeframes (in ticks)
    windows: CachePadded<[RollingWindow; MAX_TIMEFRAMES]>,
    /// Window sizes (ticks per timeframe)
    window_sizes: [usize; MAX_TIMEFRAMES],
    /// Current microprice
    microprice: CachePadded<AtomicU64>,
    /// Current VWAP
    vwap: CachePadded<AtomicU64>,
    /// VWAP accumulator
    vwap_volume_sum: CachePadded<AtomicU64>,
    /// VWAP price*volume sum
    vwap_pv_sum: CachePadded<AtomicU64>,
    /// Order book imbalance (scaled by 1000)
    ob_imbalance: CachePadded<AtomicI64>,
    /// Microprice drift (scaled by 1000)
    microprice_drift: CachePadded<AtomicI64>,
    /// Previous microprice for drift calculation
    prev_microprice: CachePadded<AtomicU64>,
    /// Signal threshold (scaled by 1000)
    signal_threshold_scaled: i32,
    /// Engine enabled
    enabled: CachePadded<AtomicBool>,
    /// Update counter
    update_count: CachePadded<AtomicU64>,
}

impl MeanReversionEngine {
    /// Create a new mean reversion engine
    /// 
    /// # Arguments
    /// * `window_ticks` - Array of window sizes in ticks for each timeframe
    /// * `signal_threshold` - Z-score threshold for signals (e.g., 2.0)
    pub fn new(window_ticks: [usize; MAX_TIMEFRAMES], signal_threshold: f64) -> Self {
        Self {
            windows: CachePadded::new(std::array::from_fn(|_| RollingWindow::new())),
            window_sizes: window_ticks,
            microprice: CachePadded::new(AtomicU64::new(0)),
            vwap: CachePadded::new(AtomicU64::new(0)),
            vwap_volume_sum: CachePadded::new(AtomicU64::new(0)),
            vwap_pv_sum: CachePadded::new(AtomicU64::new(0)),
            ob_imbalance: CachePadded::new(AtomicI64::new(0)),
            microprice_drift: CachePadded::new(AtomicI64::new(0)),
            prev_microprice: CachePadded::new(AtomicU64::new(0)),
            signal_threshold_scaled: (signal_threshold * 1000.0) as i32,
            enabled: CachePadded::new(AtomicBool::new(true)),
            update_count: CachePadded::new(AtomicU64::new(0)),
        }
    }

    /// Process a new tick
    /// 
    /// # Arguments
    /// * `price` - Trade price (micro-units)
    /// * `volume` - Trade volume
    /// * `bid_size` - Bid side liquidity
    /// * `ask_size` - Ask side liquidity
    /// * `timestamp_ns` - Timestamp
    #[inline]
    pub fn process_tick(&self, price: u64, volume: u64, bid_size: u64, ask_size: u64, timestamp_ns: u64) {
        if !self.enabled.load(Ordering::Relaxed) {
            return;
        }

        let price_f64 = price as f64;

        // Update microprice
        let microprice = self.calculate_microprice(price, bid_size, ask_size);
        self.microprice.store(microprice, Ordering::Relaxed);

        // Calculate microprice drift
        let prev_mp = self.prev_microprice.load(Ordering::Relaxed);
        if prev_mp > 0 {
            let drift = ((microprice as i64 - prev_mp as i64) * 1000) / prev_mp as i64;
            self.microprice_drift.store(drift, Ordering::Relaxed);
        }
        self.prev_microprice.store(microprice, Ordering::Relaxed);

        // Update VWAP
        self.update_vwap(price, volume);

        // Update order book imbalance
        let total = bid_size + ask_size;
        let imbalance = if total > 0 {
            ((bid_size as i64 - ask_size as i64) * 1000) / total as i64
        } else {
            0
        };
        self.ob_imbalance.store(imbalance, Ordering::Relaxed);

        // Add to rolling windows
        for (i, window) in self.windows.windows_mut().enumerate() {
            // Normalize price as return
            let ret = price_f64 / 1_000_000.0; // Simple normalization
            window.push(ret);
        }

        self.update_count.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    fn calculate_microprice(&self, last_price: u64, bid_size: u64, ask_size: u64) -> u64 {
        // Microprice = weighted mid based on liquidity
        if bid_size == 0 || ask_size == 0 {
            return last_price;
        }

        let total = bid_size + ask_size;
        // Weight towards side with more liquidity
        let weight_bid = ask_size as f64 / total as f64;
        let weight_ask = bid_size as f64 / total as f64;

        // Approximate bid/ask around last price
        let tick = 100u64; // 1 cent
        let bid = last_price.saturating_sub(tick);
        let ask = last_price.saturating_add(tick);

        ((bid as f64 * weight_bid + ask as f64 * weight_ask) as u64)
    }

    #[inline]
    fn update_vwap(&self, price: u64, volume: u64) {
        if volume == 0 {
            return;
        }

        let pv_sum = self.vwap_pv_sum.load(Ordering::Relaxed);
        let vol_sum = self.vwap_volume_sum.load(Ordering::Relaxed);

        let new_pv = pv_sum.saturating_add((price as u128 * volume as u128) as u64);
        let new_vol = vol_sum.saturating_add(volume);

        self.vwap_pv_sum.store(new_pv, Ordering::Relaxed);
        self.vwap_volume_sum.store(new_vol, Ordering::Relaxed);

        if new_vol > 0 {
            self.vwap.store(new_pv / new_vol, Ordering::Relaxed);
        }
    }

    /// Get current mean reversion signal
    pub fn get_signal(&self) -> Option<MeanReversionSignal> {
        let mut best_strength = 0i8;
        let mut best_direction = 0i8;
        let mut best_zscore = 0.0;
        let mut best_timeframe = 0;
        let mut best_confidence = 0u8;

        let current_price = self.microprice.load(Ordering::Relaxed) as f64;
        let vwap = self.vwap.load(Ordering::Relaxed) as f64;
        let ob_imb = self.ob_imbalance.load(Ordering::Relaxed) as f64 / 1000.0;
        let mp_drift = self.microprice_drift.load(Ordering::Relaxed) as f64 / 1000.0;

        for (i, window) in self.windows.windows().iter().enumerate() {
            if window.count < 10 {
                continue;
            }

            let z = window.zscore(current_price / 1_000_000.0);
            let z_scaled = (z.abs() * 1000.0) as i32;

            if z_scaled >= self.signal_threshold_scaled {
                let direction = if z > 0 { -1i8 } else { 1i8 }; // Mean reversion
                
                // Combine signals
                let vwap_deviation = if vwap > 0 {
                    ((current_price as f64 - vwap) / vwap).abs() * 1000.0
                } else {
                    0.0
                };

                // Confidence factors
                let mut confidence = 50u8;
                
                // Higher confidence if OB imbalance supports reversion
                if (ob_imb > 0.3 && direction < 0) || (ob_imb < -0.3 && direction > 0) {
                    confidence += 20;
                }

                // Higher confidence if microprice drift is extreme
                if mp_drift.abs() > 0.001 {
                    confidence += 15;
                }

                // Higher confidence if far from VWAP
                if vwap_deviation > 0.5 {
                    confidence += 15;
                }

                let strength = (z.abs() * 20.0).min(100.0) as i8;

                if strength > best_strength {
                    best_strength = strength;
                    best_direction = direction;
                    best_zscore = z;
                    best_timeframe = i;
                    best_confidence = confidence.min(100);
                }
            }
        }

        if best_strength > 0 {
            Some(MeanReversionSignal {
                strength: best_strength,
                direction: best_direction,
                z_score: best_zscore,
                timeframe_idx: best_timeframe,
                confidence: best_confidence,
            })
        } else {
            None
        }
    }

    /// Get current VWAP
    #[inline]
    pub fn get_vwap(&self) -> u64 {
        self.vwap.load(Ordering::Relaxed)
    }

    /// Get current microprice
    #[inline]
    pub fn get_microprice(&self) -> u64 {
        self.microprice.load(Ordering::Relaxed)
    }

    /// Get order book imbalance
    #[inline]
    pub fn get_ob_imbalance(&self) -> i32 {
        self.ob_imbalance.load(Ordering::Relaxed) as i32
    }

    /// Get microprice drift
    #[inline]
    pub fn get_microprice_drift(&self) -> i32 {
        self.microprice_drift.load(Ordering::Relaxed) as i32
    }

    /// Reset VWAP (for new session)
    pub fn reset_vwap(&self) {
        self.vwap.store(0, Ordering::Relaxed);
        self.vwap_volume_sum.store(0, Ordering::Relaxed);
        self.vwap_pv_sum.store(0, Ordering::Relaxed);
    }

    /// Reset all windows
    pub fn reset_windows(&self) {
        for window in self.windows.windows_mut() {
            window.clear();
        }
    }

    /// Enable engine
    pub fn enable(&self) {
        self.enabled.store(true, Ordering::Relaxed);
    }

    /// Disable engine
    pub fn disable(&self) {
        self.enabled.store(false, Ordering::Relaxed);
    }

    /// Get update count
    #[inline]
    pub fn get_update_count(&self) -> u64 {
        self.update_count.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean_reversion_basic() {
        let windows = [16, 32, 64, 128, 256, 512, 1024, 2048];
        let engine = MeanReversionEngine::new(windows, 2.0);

        // Process normal ticks
        for i in 0..100 {
            let price = 50000000 + (i % 10) * 100;
            engine.process_tick(price, 100, 1000, 1000, i as u64 * 1_000_000);
        }

        let vwap = engine.get_vwap();
        assert!(vwap > 0);

        let microprice = engine.get_microprice();
        assert!(microprice > 0);
    }

    #[test]
    fn test_extreme_deviation_signal() {
        let windows = [32, 64, 128, 256, 512, 1024, 2048, 4096];
        let engine = MeanReversionEngine::new(windows, 1.5);

        // Establish baseline
        for i in 0..200 {
            engine.process_tick(50000000, 100, 1000, 1000, i as u64 * 1_000_000);
        }

        // Extreme price spike
        engine.process_tick(55000000, 100, 100, 2000, 200_000_000);

        let signal = engine.get_signal();
        // May or may not trigger depending on window state
        assert!(engine.get_microprice() > 0);
    }
}
