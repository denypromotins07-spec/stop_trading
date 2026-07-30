//! Liquidity Heatmap Engine
//! 
//! Rolling liquidity heatmap analyzing resting limit order density over time.
//! Detects liquidity walls and spoofing clusters for SMC engine integration.

use std::sync::atomic::{AtomicU64, AtomicUsize, AtomicBool, Ordering};
use crossbeam_utils::CachePadded;

/// Maximum price levels in the heatmap
const MAX_LEVELS: usize = 4096;

/// Maximum time buckets for historical analysis
const MAX_TIME_BUCKETS: usize = 64;

/// A single liquidity level entry
#[repr(C)]
#[derive(Clone, Copy)]
pub struct LiquidityLevel {
    /// Price level (micro-units)
    pub price_micros: u64,
    /// Total bid liquidity (base units, scaled by 1000)
    pub bid_liquidity_scaled: u64,
    /// Total ask liquidity (base units, scaled by 1000)
    pub ask_liquidity_scaled: u64,
    /// Bid order count
    pub bid_count: u32,
    /// Ask order count
    pub ask_count: u32,
    /// Last update timestamp (nanoseconds)
    pub last_update_ns: u64,
    /// Hash for lookup
    pub hash: u32,
}

impl Default for LiquidityLevel {
    fn default() -> Self {
        Self {
            price_micros: 0,
            bid_liquidity_scaled: 0,
            ask_liquidity_scaled: 0,
            bid_count: 0,
            ask_count: 0,
            last_update_ns: 0,
            hash: 0,
        }
    }
}

/// Historical snapshot for rolling analysis
#[repr(C)]
#[derive(Clone, Copy)]
pub struct LiquiditySnapshot {
    /// Snapshot timestamp
    pub timestamp_ns: u64,
    /// Total bid liquidity at snapshot
    pub total_bid_scaled: u64,
    /// Total ask liquidity at snapshot
    pub total_ask_scaled: u64,
    /// Imbalance ratio (scaled by 1000)
    pub imbalance_scaled: i32,
    /// Weighted average bid price
    pub wap_bid: u64,
    /// Weighted average ask price
    pub wap_ask: u64,
}

impl Default for LiquiditySnapshot {
    fn default() -> Self {
        Self {
            timestamp_ns: 0,
            total_bid_scaled: 0,
            total_ask_scaled: 0,
            imbalance_scaled: 0,
            wap_bid: 0,
            wap_ask: 0,
        }
    }
}

/// Detected liquidity wall
pub struct LiquidityWall {
    /// Price level of the wall
    pub price_micros: u64,
    /// Side: true = bid wall, false = ask wall
    pub is_bid: bool,
    /// Wall size (scaled by 1000)
    pub size_scaled: u64,
    /// Confidence score (0-100)
    pub confidence: u8,
    /// Duration observed (milliseconds)
    pub duration_ms: u32,
    /// Whether it appears to be spoofing
    pub is_spoofing_suspect: bool,
}

/// Lock-free Liquidity Heatmap with rolling history
pub struct LiquidityHeatmap {
    /// Current liquidity levels
    levels: CachePadded<[LiquidityLevel; MAX_LEVELS]>,
    /// Active level count
    active_count: CachePadded<AtomicUsize>,
    /// Rolling snapshots for historical analysis
    snapshots: CachePadded<[LiquiditySnapshot; MAX_TIME_BUCKETS]>,
    /// Current snapshot index
    snapshot_idx: CachePadded<AtomicUsize>,
    /// Snapshot interval (nanoseconds)
    snapshot_interval_ns: u64,
    /// Last snapshot timestamp
    last_snapshot_ns: CachePadded<AtomicU64>,
    /// Price bucket size
    bucket_size_micros: u64,
    /// Heatmap enabled
    enabled: CachePadded<AtomicBool>,
    /// Version counter
    version: CachePadded<AtomicU64>,
}

impl LiquidityHeatmap {
    /// Create a new liquidity heatmap
    /// 
    /// # Arguments
    /// * `bucket_size_micros` - Price bucket size for grouping levels
    /// * `snapshot_interval_ms` - Interval between historical snapshots
    pub fn new(bucket_size_micros: u64, snapshot_interval_ms: u64) -> Self {
        Self {
            levels: CachePadded::new(std::array::from_fn(|_| LiquidityLevel::default())),
            active_count: CachePadded::new(AtomicUsize::new(0)),
            snapshots: CachePadded::new(std::array::from_fn(|_| LiquiditySnapshot::default())),
            snapshot_idx: CachePadded::new(AtomicUsize::new(0)),
            snapshot_interval_ns: snapshot_interval_ms * 1_000_000,
            last_snapshot_ns: CachePadded::new(AtomicU64::new(0)),
            bucket_size_micros,
            enabled: CachePadded::new(AtomicBool::new(true)),
            version: CachePadded::new(AtomicU64::new(0)),
        }
    }

    /// Update liquidity at a price level
    /// 
    /// # Arguments
    /// * `price_micros` - Price level
    /// * `volume_scaled` - Volume change (positive = add, negative = remove)
    /// * `is_bid` - True for bid side, false for ask
    /// * `timestamp_ns` - Current timestamp
    #[inline]
    pub fn update_liquidity(&self, price_micros: u64, volume_scaled: i64, is_bid: bool, timestamp_ns: u64) {
        if !self.enabled.load(Ordering::Relaxed) {
            return;
        }

        let bucketed_price = (price_micros / self.bucket_size_micros) * self.bucket_size_micros;
        
        // Find or create level
        let hash = (bucketed_price / self.bucket_size_micros) as u32;
        let mut idx = (hash as usize) % MAX_LEVELS;
        let mut first_empty = None;

        for _ in 0..MAX_LEVELS {
            let level_price = self.levels[idx].price_micros;
            
            if level_price == 0 {
                if first_empty.is_none() {
                    first_empty = Some(idx);
                }
                idx = (idx + 1) % MAX_LEVELS;
                continue;
            }

            if level_price == bucketed_price {
                unsafe {
                    let level_ptr = &self.levels[idx] as *const LiquidityLevel as *mut LiquidityLevel;
                    if is_bid {
                        if volume_scaled >= 0 {
                            (*level_ptr).bid_liquidity_scaled = 
                                (*level_ptr).bid_liquidity_scaled.saturating_add(volume_scaled as u64);
                            (*level_ptr).bid_count = (*level_ptr).bid_count.saturating_add(1);
                        } else {
                            (*level_ptr).bid_liquidity_scaled = 
                                (*level_ptr).bid_liquidity_scaled.saturating_sub((-volume_scaled) as u64);
                        }
                    } else {
                        if volume_scaled >= 0 {
                            (*level_ptr).ask_liquidity_scaled = 
                                (*level_ptr).ask_liquidity_scaled.saturating_add(volume_scaled as u64);
                            (*level_ptr).ask_count = (*level_ptr).ask_count.saturating_add(1);
                        } else {
                            (*level_ptr).ask_liquidity_scaled = 
                                (*level_ptr).ask_liquidity_scaled.saturating_sub((-volume_scaled) as u64);
                        }
                    }
                    (*level_ptr).last_update_ns = timestamp_ns;
                }
                self.version.fetch_add(1, Ordering::Relaxed);
                
                // Check if snapshot needed
                self.maybe_snapshot(timestamp_ns);
                return;
            }

            idx = (idx + 1) % MAX_LEVELS;
        }

        // Insert into empty slot
        if let Some(insert_idx) = first_empty {
            unsafe {
                let level_ptr = &self.levels[insert_idx] as *const LiquidityLevel as *mut LiquidityLevel;
                (*level_ptr) = LiquidityLevel::default();
                (*level_ptr).price_micros = bucketed_price;
                (*level_ptr).hash = hash;
                (*level_ptr).last_update_ns = timestamp_ns;
                
                if is_bid {
                    (*level_ptr).bid_liquidity_scaled = volume_scaled.unsigned_abs();
                    (*level_ptr).bid_count = 1;
                } else {
                    (*level_ptr).ask_liquidity_scaled = volume_scaled.unsigned_abs();
                    (*level_ptr).ask_count = 1;
                }
            }
            self.active_count.fetch_add(1, Ordering::Relaxed);
            self.version.fetch_add(1, Ordering::Relaxed);
            self.maybe_snapshot(timestamp_ns);
        }
    }

    /// Take a snapshot if interval has elapsed
    #[inline]
    fn maybe_snapshot(&self, timestamp_ns: u64) {
        let last = self.last_snapshot_ns.load(Ordering::Relaxed);
        if timestamp_ns.saturating_sub(last) < self.snapshot_interval_ns {
            return;
        }

        // Calculate totals for snapshot
        let mut total_bid = 0u64;
        let mut total_ask = 0u64;
        let mut weighted_bid = 0u128;
        let mut weighted_ask = 0u128;

        for i in 0..MAX_LEVELS {
            let level = &self.levels[i];
            if level.price_micros > 0 {
                total_bid = total_bid.saturating_add(level.bid_liquidity_scaled);
                total_ask = total_ask.saturating_add(level.ask_liquidity_scaled);
                weighted_bid += (level.price_micros as u128) * (level.bid_liquidity_scaled as u128);
                weighted_ask += (level.price_micros as u128) * (level.ask_liquidity_scaled as u128);
            }
        }

        let imbalance = if total_bid + total_ask > 0 {
            (((total_bid as i64 - total_ask as i64) * 1000) / (total_bid + total_ask) as i64) as i32
        } else {
            0
        };

        let wap_bid = if total_bid > 0 {
            (weighted_bid / total_bid as u128) as u64
        } else {
            0
        };

        let wap_ask = if total_ask > 0 {
            (weighted_ask / total_ask as u128) as u64
        } else {
            0
        };

        // Store snapshot in rolling buffer
        let idx = self.snapshot_idx.load(Ordering::Relaxed);
        let snapshot = LiquiditySnapshot {
            timestamp_ns,
            total_bid_scaled: total_bid,
            total_ask_scaled: total_ask,
            imbalance_scaled: imbalance,
            wap_bid,
            wap_ask,
        };

        self.snapshots.snapshots[idx] = snapshot;
        self.last_snapshot_ns.store(timestamp_ns, Ordering::Relaxed);
        self.snapshot_idx.store((idx + 1) % MAX_TIME_BUCKETS, Ordering::Relaxed);
    }

    /// Detect liquidity walls above threshold
    /// 
    /// # Arguments
    /// * `threshold_scaled` - Minimum wall size (scaled by 1000)
    /// * `current_price` - Current mid price for reference
    pub fn detect_walls(&self, threshold_scaled: u64, current_price: u64) -> Vec<LiquidityWall> {
        let mut walls = Vec::with_capacity(32);
        let now_ns = self.get_current_time_ns();

        for i in 0..MAX_LEVELS {
            let level = &self.levels[i];
            if level.price_micros == 0 {
                continue;
            }

            // Check bid walls
            if level.bid_liquidity_scaled >= threshold_scaled {
                let distance_pct = if current_price > 0 {
                    ((current_price - level.price_micros) * 100) / current_price
                } else {
                    0
                };

                let confidence = self.calculate_wall_confidence(level, now_ns, distance_pct);
                
                walls.push(LiquidityWall {
                    price_micros: level.price_micros,
                    is_bid: true,
                    size_scaled: level.bid_liquidity_scaled,
                    confidence,
                    duration_ms: ((now_ns - level.last_update_ns) / 1_000_000) as u32,
                    is_spoofing_suspect: self.is_spoofing_suspect(level, now_ns),
                });
            }

            // Check ask walls
            if level.ask_liquidity_scaled >= threshold_scaled {
                let distance_pct = if current_price > 0 {
                    ((level.price_micros - current_price) * 100) / current_price
                } else {
                    0
                };

                let confidence = self.calculate_wall_confidence(level, now_ns, distance_pct);
                
                walls.push(LiquidityWall {
                    price_micros: level.price_micros,
                    is_bid: false,
                    size_scaled: level.ask_liquidity_scaled,
                    confidence,
                    duration_ms: ((now_ns - level.last_update_ns) / 1_000_000) as u32,
                    is_spoofing_suspect: self.is_spoofing_suspect(level, now_ns),
                });
            }
        }

        // Sort by size descending
        walls.sort_by(|a, b| b.size_scaled.cmp(&a.size_scaled));
        walls
    }

    #[inline]
    fn calculate_wall_confidence(&self, level: &LiquidityLevel, now_ns: u64, distance_pct: u64) -> u8 {
        let mut confidence = 50u8;

        // Higher confidence for longer duration
        let duration_ms = ((now_ns - level.last_update_ns) / 1_000_000) as u32;
        if duration_ms > 60000 {
            confidence = confidence.saturating_add(20);
        } else if duration_ms > 10000 {
            confidence = confidence.saturating_add(10);
        }

        // Higher confidence for closer walls
        if distance_pct < 100 {
            confidence = confidence.saturating_add(20);
        } else if distance_pct < 500 {
            confidence = confidence.saturating_add(10);
        }

        // Higher confidence for larger walls
        let total_liq = level.bid_liquidity_scaled.saturating_add(level.ask_liquidity_scaled);
        if total_liq > 10_000_000 {
            confidence = confidence.saturating_add(10);
        }

        confidence.min(100)
    }

    #[inline]
    fn is_spoofing_suspect(&self, level: &LiquidityLevel, now_ns: u64) -> bool {
        // Spoofing indicators:
        // 1. Large size but very recent appearance
        // 2. Frequent updates (flickering)
        // 3. Far from current price
        
        let duration_ms = ((now_ns - level.last_update_ns) / 1_000_000) as u32;
        let total_liq = level.bid_liquidity_scaled.saturating_add(level.ask_liquidity_scaled);
        
        // Large wall that appeared very recently
        if total_liq > 5_000_000 && duration_ms < 1000 {
            return true;
        }

        false
    }

    /// Get current imbalance ratio
    #[inline]
    pub fn get_imbalance(&self) -> i32 {
        let mut total_bid = 0u64;
        let mut total_ask = 0u64;

        for i in 0..MAX_LEVELS {
            let level = &self.levels[i];
            if level.price_micros > 0 {
                total_bid = total_bid.saturating_add(level.bid_liquidity_scaled);
                total_ask = total_ask.saturating_add(level.ask_liquidity_scaled);
            }
        }

        if total_bid + total_ask == 0 {
            return 0;
        }

        (((total_bid as i64 - total_ask as i64) * 1000) / (total_bid + total_ask) as i64) as i32
    }

    /// Get liquidity at specific price
    #[inline]
    pub fn get_liquidity_at_price(&self, price_micros: u64) -> (u64, u64) {
        let bucketed_price = (price_micros / self.bucket_size_micros) * self.bucket_size_micros;

        for i in 0..MAX_LEVELS {
            let level = &self.levels[i];
            if level.price_micros == bucketed_price {
                return (level.bid_liquidity_scaled, level.ask_liquidity_scaled);
            }
        }
        (0, 0)
    }

    /// Get recent snapshots for trend analysis
    pub fn get_recent_snapshots(&self, count: usize) -> Vec<LiquiditySnapshot> {
        let mut result = Vec::with_capacity(count);
        let current_idx = self.snapshot_idx.load(Ordering::Relaxed);
        let valid_count = self.active_count.load(Ordering::Relaxed).min(MAX_TIME_BUCKETS);

        for i in 0..count.min(valid_count) {
            let idx = (current_idx + MAX_TIME_BUCKETS - 1 - i) % MAX_TIME_BUCKETS;
            if self.snapshots.snapshots[idx].timestamp_ns > 0 {
                result.push(self.snapshots.snapshots[idx]);
            }
        }

        result
    }

    /// Get heatmap version
    #[inline]
    pub fn get_version(&self) -> u64 {
        self.version.load(Ordering::Relaxed)
    }

    /// Enable heatmap updates
    pub fn enable(&self) {
        self.enabled.store(true, Ordering::Relaxed);
    }

    /// Disable heatmap updates
    pub fn disable(&self) {
        self.enabled.store(false, Ordering::Relaxed);
    }

    /// Clear all liquidity data
    pub fn clear(&self) {
        for i in 0..MAX_LEVELS {
            self.levels.levels[i] = LiquidityLevel::default();
        }
        self.active_count.store(0, Ordering::Relaxed);
        self.version.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    fn get_current_time_ns(&self) -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_liquidity_update() {
        let heatmap = LiquidityHeatmap::new(100, 1000);
        
        heatmap.update_liquidity(50000000, 1000000, true, 1000000);
        heatmap.update_liquidity(50000100, 500000, false, 1000000);
        
        let (bid, ask) = heatmap.get_liquidity_at_price(50000000);
        assert!(bid > 0);
        assert_eq!(ask, 0);
    }

    #[test]
    fn test_imbalance_calculation() {
        let heatmap = LiquidityHeatmap::new(100, 1000);
        
        // Add more bid liquidity
        heatmap.update_liquidity(50000000, 1000000, true, 1000000);
        heatmap.update_liquidity(49999900, 1000000, true, 1000000);
        heatmap.update_liquidity(50000100, 500000, false, 1000000);
        
        let imbalance = heatmap.get_imbalance();
        assert!(imbalance > 0); // More bids than asks
    }
}
