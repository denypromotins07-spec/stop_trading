//! Market Profile Generator
//! 
//! Tracks Time Price Opportunity (TPO) and initial balance ranges.
//! Feeds structural auction market theory data into regime detection.

use std::sync::atomic::{AtomicU64, AtomicUsize, AtomicBool, Ordering};
use crossbeam_utils::CachePadded;

/// Maximum TPO levels
const MAX_TPO_LEVELS: usize = 2048;

/// Maximum letters for TPO (A-Z = 26 periods)
const MAX_TPO_PERIODS: usize = 26;

/// A single TPO level
#[repr(C)]
#[derive(Clone, Copy)]
pub struct TpoLevel {
    /// Price level (micro-units)
    pub price_micros: u64,
    /// TPO count per period (bitmask)
    pub tpo_mask: u32,
    /// Total time at price (in periods)
    pub total_periods: u8,
    /// Volume at price (scaled by 1000)
    pub volume_scaled: u64,
}

impl Default for TpoLevel {
    fn default() -> Self {
        Self {
            price_micros: 0,
            tpo_mask: 0,
            total_periods: 0,
            volume_scaled: 0,
        }
    }
}

/// Initial Balance range
pub struct InitialBalance {
    /// High of initial balance period
    pub high: u64,
    /// Low of initial balance period
    pub low: u64,
    /// Range size
    pub range: u64,
    /// Whether IB has been established
    pub established: bool,
}

/// Market Profile session data
pub struct MarketProfileSession {
    /// Session open price
    pub open_price: u64,
    /// Session high
    pub high: u64,
    /// Session low
    pub low: u64,
    /// Session close (last price)
    pub close: u64,
    /// Point of Control
    pub poc: u64,
    /// Value Area High
    pub vah: u64,
    /// Value Area Low
    pub val: u64,
    /// Initial Balance High
    pub ib_high: u64,
    /// Initial Balance Low
    pub ib_low: u64,
    /// TPO count
    pub tpo_count: usize,
}

/// Lock-free Market Profile engine
pub struct MarketProfileEngine {
    /// TPO levels
    tpo_levels: CachePadded<[TpoLevel; MAX_TPO_LEVELS]>,
    /// Active TPO count
    active_count: CachePadded<AtomicUsize>,
    /// Current period (0-25 for A-Z)
    current_period: CachePadded<AtomicUsize>,
    /// Period duration (milliseconds)
    period_duration_ms: u64,
    /// Session start timestamp
    session_start_ns: CachePadded<AtomicU64>,
    /// Last period update
    last_period_update_ns: CachePadded<AtomicU64>,
    /// Session open price
    session_open: CachePadded<AtomicU64>,
    /// Session high
    session_high: CachePadded<AtomicU64>,
    /// Session low
    session_low: CachePadded<AtomicU64>,
    /// Session close
    session_close: CachePadded<AtomicU64>,
    /// Initial Balance period (minutes)
    ib_period_minutes: u32,
    /// IB established flag
    ib_established: CachePadded<AtomicBool>,
    /// IB high
    ib_high: CachePadded<AtomicU64>,
    /// IB low
    ib_low: CachePadded<AtomicU64>,
    /// Profile enabled
    enabled: CachePadded<AtomicBool>,
    /// Version counter
    version: CachePadded<AtomicU64>,
}

impl MarketProfileEngine {
    /// Create a new market profile engine
    /// 
    /// # Arguments
    /// * `period_duration_ms` - Duration of each TPO period
    /// * `ib_period_minutes` - Initial balance period in minutes (typically 30 or 60)
    pub fn new(period_duration_ms: u64, ib_period_minutes: u32) -> Self {
        Self {
            tpo_levels: CachePadded::new(std::array::from_fn(|_| TpoLevel::default())),
            active_count: CachePadded::new(AtomicUsize::new(0)),
            current_period: CachePadded::new(AtomicUsize::new(0)),
            period_duration_ms,
            session_start_ns: CachePadded::new(AtomicU64::new(0)),
            last_period_update_ns: CachePadded::new(AtomicU64::new(0)),
            session_open: CachePadded::new(AtomicU64::new(0)),
            session_high: CachePadded::new(AtomicU64::new(0)),
            session_low: CachePadded::new(AtomicU64::new(u64::MAX)),
            session_close: CachePadded::new(AtomicU64::new(0)),
            ib_period_minutes,
            ib_established: CachePadded::new(AtomicBool::new(false)),
            ib_high: CachePadded::new(AtomicU64::new(0)),
            ib_low: CachePadded::new(AtomicU64::new(u64::MAX)),
            enabled: CachePadded::new(AtomicBool::new(true)),
            version: CachePadded::new(AtomicU64::new(0)),
        }
    }

    /// Record price at time for TPO construction
    /// 
    /// # Arguments
    /// * `price_micros` - Current price
    /// * `timestamp_ns` - Current timestamp
    /// * `volume_scaled` - Volume at this price (scaled by 1000)
    #[inline]
    pub fn record_price(&self, price_micros: u64, timestamp_ns: u64, volume_scaled: u64) {
        if !self.enabled.load(Ordering::Relaxed) {
            return;
        }

        // Initialize session on first tick
        let session_start = self.session_start_ns.load(Ordering::Relaxed);
        if session_start == 0 {
            self.session_start_ns.store(timestamp_ns, Ordering::Relaxed);
            self.session_open.store(price_micros, Ordering::Relaxed);
            self.last_period_update_ns.store(timestamp_ns, Ordering::Relaxed);
        }

        // Update session extremes
        self.update_session_extremes(price_micros);

        // Update Initial Balance
        self.update_initial_balance(price_micros, timestamp_ns);

        // Check for period rollover
        self.check_period_rollover(timestamp_ns);

        // Add TPO at current price/period
        let period = self.current_period.load(Ordering::Relaxed);
        self.add_tpo(price_micros, period, volume_scaled);

        self.session_close.store(price_micros, Ordering::Relaxed);
        self.version.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    fn update_session_extremes(&self, price: u64) {
        let high = self.session_high.load(Ordering::Relaxed);
        let low = self.session_low.load(Ordering::Relaxed);

        if price > high {
            self.session_high.store(price, Ordering::Relaxed);
        }
        if price < low {
            self.session_low.store(price, Ordering::Relaxed);
        }
    }

    #[inline]
    fn update_initial_balance(&self, price: u64, timestamp_ns: u64) {
        if self.ib_established.load(Ordering::Relaxed) {
            return;
        }

        let session_start = self.session_start_ns.load(Ordering::Relaxed);
        let elapsed_ms = timestamp_ns.saturating_sub(session_start) / 1_000_000;
        let ib_duration_ms = (self.ib_period_minutes as u64) * 60 * 1000;

        if elapsed_ms >= ib_duration_ms {
            // IB period complete
            let high = self.ib_high.load(Ordering::Relaxed);
            let low = self.ib_low.load(Ordering::Relaxed);
            
            if high > 0 && low < u64::MAX {
                self.ib_established.store(true, Ordering::Relaxed);
            }
            return;
        }

        // Update IB range
        let current_high = self.ib_high.load(Ordering::Relaxed);
        let current_low = self.ib_low.load(Ordering::Relaxed);

        if price > current_high || current_high == 0 {
            self.ib_high.store(price, Ordering::Relaxed);
        }
        if price < current_low || current_low == u64::MAX {
            self.ib_low.store(price, Ordering::Relaxed);
        }
    }

    #[inline]
    fn check_period_rollover(&self, timestamp_ns: u64) {
        let last_update = self.last_period_update_ns.load(Ordering::Relaxed);
        let elapsed_ms = timestamp_ns.saturating_sub(last_update);

        if elapsed_ms >= self.period_duration_ms {
            let current = self.current_period.load(Ordering::Relaxed);
            if current < MAX_TPO_PERIODS - 1 {
                self.current_period.store(current + 1, Ordering::Relaxed);
                self.last_period_update_ns.store(timestamp_ns, Ordering::Relaxed);
            }
        }
    }

    #[inline]
    fn add_tpo(&self, price_micros: u64, period: usize, volume_scaled: u64) {
        let bucket_size = 100u64; // 1 cent buckets
        let bucketed_price = (price_micros / bucket_size) * bucket_size;

        let hash = (bucketed_price / bucket_size) as u32;
        let mut idx = (hash as usize) % MAX_TPO_LEVELS;
        let mut first_empty = None;

        for _ in 0..MAX_TPO_LEVELS {
            let level_price = self.tpo_levels[idx].price_micros;

            if level_price == 0 {
                if first_empty.is_none() {
                    first_empty = Some(idx);
                }
                idx = (idx + 1) % MAX_TPO_LEVELS;
                continue;
            }

            if level_price == bucketed_price {
                unsafe {
                    let level_ptr = &self.tpo_levels[idx] as *const TpoLevel as *mut TpoLevel;
                    (*level_ptr).tpo_mask |= (1u32 << period);
                    
                    // Count set bits for total periods
                    (*level_ptr).total_periods = (*level_ptr).tpo_mask.count_ones() as u8;
                    
                    (*level_ptr).volume_scaled = (*level_ptr).volume_scaled.saturating_add(volume_scaled);
                }
                return;
            }

            idx = (idx + 1) % MAX_TPO_LEVELS;
        }

        // Insert new level
        if let Some(insert_idx) = first_empty {
            unsafe {
                let level_ptr = &self.tpo_levels[insert_idx] as *const TpoLevel as *mut TpoLevel;
                (*level_ptr) = TpoLevel {
                    price_micros: bucketed_price,
                    tpo_mask: 1u32 << period,
                    total_periods: 1,
                    volume_scaled,
                };
            }
            self.active_count.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Calculate Point of Control
    #[inline]
    pub fn calculate_poc(&self) -> u64 {
        let mut poc_price = 0u64;
        let mut max_periods = 0u8;

        for i in 0..MAX_TPO_LEVELS {
            let level = &self.tpo_levels[i];
            if level.price_micros > 0 && level.total_periods > max_periods {
                max_periods = level.total_periods;
                poc_price = level.price_micros;
            }
        }

        poc_price
    }

    /// Get Initial Balance
    #[inline]
    pub fn get_initial_balance(&self) -> InitialBalance {
        let high = self.ib_high.load(Ordering::Relaxed);
        let low = self.ib_low.load(Ordering::Relaxed);
        let established = self.ib_established.load(Ordering::Relaxed);

        InitialBalance {
            high,
            low: if low == u64::MAX { 0 } else { low },
            range: if high > 0 && low < u64::MAX { high - low } else { 0 },
            established,
        }
    }

    /// Get session statistics
    pub fn get_session_stats(&self) -> MarketProfileSession {
        let open = self.session_open.load(Ordering::Relaxed);
        let high = self.session_high.load(Ordering::Relaxed);
        let low = self.session_low.load(Ordering::Relaxed);
        let close = self.session_close.load(Ordering::Relaxed);
        let poc = self.calculate_poc();
        let (vah, val) = self.calculate_value_area();
        let ib = self.get_initial_balance();

        MarketProfileSession {
            open_price: open,
            high,
            low: if low == u64::MAX { 0 } else { low },
            close,
            poc,
            vah,
            val,
            ib_high: ib.high,
            ib_low: if ib.low == u64::MAX { 0 } else { ib.low },
            tpo_count: self.active_count.load(Ordering::Relaxed),
        }
    }

    /// Calculate Value Area (70% of TPOs around POC)
    pub fn calculate_value_area(&self) -> (u64, u64) {
        let poc = self.calculate_poc();
        if poc == 0 {
            return (0, 0);
        }

        // Collect all levels with their period counts
        let mut levels: [(u64, u8); MAX_TPO_LEVELS] = [(0, 0); MAX_TPO_LEVELS];
        let mut count = 0;

        for i in 0..MAX_TPO_LEVELS {
            let level = &self.tpo_levels[i];
            if level.price_micros > 0 {
                levels[count] = (level.price_micros, level.total_periods);
                count += 1;
            }
        }

        if count == 0 {
            return (0, 0);
        }

        // Sort by price
        for i in 0..count.saturating_sub(1) {
            for j in 0..count.saturating_sub(i + 1) {
                if levels[j].0 > levels[j + 1].0 {
                    levels.swap(j, j + 1);
                }
            }
        }

        // Find POC index
        let mut poc_idx = 0;
        for i in 0..count {
            if levels[i].0 == poc {
                poc_idx = i;
                break;
            }
        }

        // Calculate total TPOs
        let mut total_tpos = 0u32;
        for i in 0..count {
            total_tpos += levels[i].1 as u32;
        }

        let target = (total_tpos * 70) / 100;

        // Expand from POC
        let mut accumulated = levels[poc_idx].1 as u32;
        let mut left = poc_idx;
        let mut right = poc_idx;

        while accumulated < target {
            let left_periods = if left > 0 { levels[left - 1].1 as u32 } else { 0 };
            let right_periods = if right + 1 < count { levels[right + 1].1 as u32 } else { 0 };

            if left_periods >= right_periods {
                if left > 0 {
                    left -= 1;
                    accumulated += left_periods;
                } else if right + 1 < count {
                    right += 1;
                    accumulated += right_periods;
                } else {
                    break;
                }
            } else {
                if right + 1 < count {
                    right += 1;
                    accumulated += right_periods;
                } else if left > 0 {
                    left -= 1;
                    accumulated += left_periods;
                } else {
                    break;
                }
            }
        }

        (levels[right].0, levels[left].0)
    }

    /// Check if price is in value area
    #[inline]
    pub fn is_in_value_area(&self, price_micros: u64) -> bool {
        let (vah, val) = self.calculate_value_area();
        price_micros >= val && price_micros <= vah
    }

    /// Get profile type based on structure
    pub fn get_profile_type(&self) -> &'static str {
        let stats = self.get_session_stats();
        let ib_range = stats.ib_high.saturating_sub(stats.ib_low);
        let total_range = stats.high.saturating_sub(stats.low);

        if ib_range == 0 || total_range == 0 {
            return "Unknown";
        }

        let ib_ratio = (total_range as f64) / (ib_range as f64);

        if ib_ratio < 1.5 {
            "Normal"
        } else if ib_ratio < 2.5 {
            "Trend"
        } else {
            "Neutral"
        }
    }

    /// Reset for new session
    pub fn reset(&self) {
        for i in 0..MAX_TPO_LEVELS {
            self.tpo_levels.tpo_levels[i] = TpoLevel::default();
        }
        self.active_count.store(0, Ordering::Relaxed);
        self.current_period.store(0, Ordering::Relaxed);
        self.session_start_ns.store(0, Ordering::Relaxed);
        self.last_period_update_ns.store(0, Ordering::Relaxed);
        self.session_open.store(0, Ordering::Relaxed);
        self.session_high.store(0, Ordering::Relaxed);
        self.session_low.store(u64::MAX, Ordering::Relaxed);
        self.session_close.store(0, Ordering::Relaxed);
        self.ib_established.store(false, Ordering::Relaxed);
        self.ib_high.store(0, Ordering::Relaxed);
        self.ib_low.store(u64::MAX, Ordering::Relaxed);
        self.version.fetch_add(1, Ordering::Relaxed);
    }

    /// Get version
    #[inline]
    pub fn get_version(&self) -> u64 {
        self.version.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_market_profile_basic() {
        let profile = MarketProfileEngine::new(1800000, 30); // 30min periods, 30min IB
        
        // Record some prices
        profile.record_price(50000000, 1000000, 1000);
        profile.record_price(50000100, 2000000, 500);
        profile.record_price(50000000, 3000000, 1500);
        
        let stats = profile.get_session_stats();
        assert_eq!(stats.open_price, 50000000);
        assert!(stats.poc > 0);
    }
}
