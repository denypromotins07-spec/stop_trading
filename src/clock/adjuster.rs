//! Latency Adjuster Implementation
//!
//! Builds a latency adjuster calculating the true event time
//! (Exchange Timestamp + Network Transit). Re-orders incoming
//! WebSocket messages based on true exchange time rather than
//! local arrival time to fix out-of-order ticks.

use std::sync::atomic::{AtomicU64, AtomicI64, Ordering};

/// Maximum venues supported for latency tracking
pub const MAX_VENUES: usize = 32;

/// Latency statistics per venue
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct VenueLatencyStats {
    /// Venue ID
    pub venue_id: u32,
    /// Average RTT in nanoseconds
    pub avg_rtt_ns: u64,
    /// Minimum RTT observed
    pub min_rtt_ns: u64,
    /// Maximum RTT observed
    pub max_rtt_ns: u64,
    /// One-way latency estimate (RTT / 2)
    pub one_way_latency_ns: u64,
    /// Sample count
    pub sample_count: u64,
    /// Last update timestamp
    pub last_update_ns: u64,
}

impl VenueLatencyStats {
    #[inline]
    pub fn new(venue_id: u32) -> Self {
        Self {
            venue_id,
            avg_rtt_ns: 0,
            min_rtt_ns: u64::MAX,
            max_rtt_ns: 0,
            one_way_latency_ns: 0,
            sample_count: 0,
            last_update_ns: 0,
        }
    }

    #[inline]
    pub fn update(&mut self, rtt_ns: u64, timestamp_ns: u64) {
        self.sample_count += 1;
        
        // Update min/max
        if rtt_ns < self.min_rtt_ns {
            self.min_rtt_ns = rtt_ns;
        }
        if rtt_ns > self.max_rtt_ns {
            self.max_rtt_ns = rtt_ns;
        }

        // Exponential moving average for RTT
        let alpha = 0.1f64; // Smoothing factor
        let avg = self.avg_rtt_ns as f64;
        let rtt = rtt_ns as f64;
        self.avg_rtt_ns = ((1.0 - alpha) * avg + alpha * rtt) as u64;

        // One-way latency is approximately half of RTT
        self.one_way_latency_ns = self.avg_rtt_ns / 2;
        self.last_update_ns = timestamp_ns;
    }
}

/// Latency statistics across all venues
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct LatencyStats {
    /// Total RTT measurements
    pub total_measurements: u64,
    /// Overall average RTT
    pub overall_avg_rtt_ns: u64,
    /// Best (minimum) RTT observed
    pub best_rtt_ns: u64,
    /// Worst (maximum) RTT observed
    pub worst_rtt_ns: u64,
    /// Out-of-order events detected
    pub out_of_order_events: u64,
    /// Events reordered
    pub events_reordered: u64,
}

impl LatencyStats {
    #[inline]
    pub fn new() -> Self {
        Self {
            total_measurements: 0,
            overall_avg_rtt_ns: 0,
            best_rtt_ns: u64::MAX,
            worst_rtt_ns: 0,
            out_of_order_events: 0,
            events_reordered: 0,
        }
    }
}

impl Default for LatencyStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Adjusted timestamp with metadata
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AdjustedTimestamp {
    /// Original timestamp (local arrival)
    pub original_ns: u64,
    /// Exchange timestamp
    pub exchange_ts_ns: u64,
    /// Adjusted true time
    pub adjusted_ns: u64,
    /// Network transit time added
    pub transit_time_ns: u64,
    /// Venue ID
    pub venue_id: u32,
    /// Was this timestamp reordered
    pub was_reordered: bool,
    /// Confidence level (0-100)
    pub confidence: u8,
}

impl AdjustedTimestamp {
    #[inline]
    pub fn new(original_ns: u64, exchange_ts_ns: u64, venue_id: u32) -> Self {
        Self {
            original_ns,
            exchange_ts_ns,
            adjusted_ns: exchange_ts_ns,
            transit_time_ns: 0,
            venue_id,
            was_reordered: false,
            confidence: 50,
        }
    }

    #[inline]
    pub fn with_transit(mut self, transit_ns: u64) -> Self {
        self.transit_time_ns = transit_ns;
        self.adjusted_ns = self.exchange_ts_ns.saturating_add(transit_ns);
        self.confidence = if transit_ns < 1_000_000 {
            100u8
        } else if transit_ns < 5_000_000 {
            80u8
        } else {
            60u8
        };
        self
    }

    #[inline]
    pub fn mark_reordered(&mut self) {
        self.was_reordered = true;
    }
}

/// Latency adjuster for calculating true event times
#[repr(C)]
pub struct LatencyAdjuster {
    /// Per-venue latency stats
    venue_stats: [VenueLatencyStats; MAX_VENUES],
    /// Total measurements
    total_measurements: AtomicU64,
    /// RTT sum for averaging
    rtt_sum: AtomicU64,
    /// Out-of-order counter
    out_of_order_count: AtomicU64,
    /// Reordered events counter
    reordered_count: AtomicU64,
    /// Last sequence number per venue (for reorder detection)
    last_sequences: [AtomicU64; MAX_VENUES],
}

impl LatencyAdjuster {
    /// Create a new latency adjuster
    pub fn new() -> Self {
        Self {
            venue_stats: std::array::from_fn(|i| VenueLatencyStats::new(i as u32)),
            total_measurements: AtomicU64::new(0),
            rtt_sum: AtomicU64::new(0),
            out_of_order_count: AtomicU64::new(0),
            reordered_count: AtomicU64::new(0),
            last_sequences: std::array::from_fn(|_| AtomicU64::new(0)),
        }
    }

    /// Record RTT measurement for a venue
    #[inline]
    pub fn record_rtt(&self, rtt_ns: u64, venue_id: u32) {
        let idx = (venue_id as usize).min(MAX_VENUES - 1);
        let now = self.get_timestamp_ns();

        self.venue_stats[idx].update(rtt_ns, now);
        
        self.total_measurements.fetch_add(1, Ordering::Relaxed);
        self.rtt_sum.fetch_add(rtt_ns, Ordering::Relaxed);
    }

    /// Calculate true event time from exchange timestamp
    #[inline]
    pub fn calculate_true_event_time(&self, exchange_ts_ns: u64, venue_id: u32) -> u64 {
        let idx = (venue_id as usize).min(MAX_VENUES - 1);
        let transit = self.venue_stats[idx].one_way_latency_ns;
        
        // True time = exchange timestamp + network transit time
        exchange_ts_ns.saturating_add(transit)
    }

    /// Adjust a timestamp with full metadata
    #[inline]
    pub fn adjust_timestamp(&self, local_arrival_ns: u64, exchange_ts_ns: u64, venue_id: u32) -> AdjustedTimestamp {
        let mut adjusted = AdjustedTimestamp::new(local_arrival_ns, exchange_ts_ns, venue_id);
        
        let idx = (venue_id as usize).min(MAX_VENUES - 1);
        let transit = self.venue_stats[idx].one_way_latency_ns;
        
        adjusted = adjusted.with_transit(transit);

        // Check for reordering
        if self.detect_reorder(exchange_ts_ns, venue_id) {
            adjusted.mark_reordered();
            self.reordered_count.fetch_add(1, Ordering::Relaxed);
        }

        adjusted
    }

    /// Detect if message arrived out of order
    #[inline]
    pub fn detect_reorder(&self, exchange_ts_ns: u64, venue_id: u32) -> bool {
        let idx = (venue_id as usize).min(MAX_VENUES - 1);
        let last_ts = self.last_sequences[idx].load(Ordering::Acquire);

        if exchange_ts_ns < last_ts && last_ts > 0 {
            // This message has an earlier timestamp than the previous one
            self.out_of_order_count.fetch_add(1, Ordering::Relaxed);
            return true;
        }

        // Update last seen timestamp
        self.last_sequences[idx].store(exchange_ts_ns, Ordering::Release);
        false
    }

    /// Get latency stats for a venue
    #[inline]
    pub fn get_venue_stats(&self, venue_id: u32) -> VenueLatencyStats {
        let idx = (venue_id as usize).min(MAX_VENUES - 1);
        self.venue_stats[idx]
    }

    /// Get overall statistics
    #[inline]
    pub fn get_stats(&self) -> LatencyStats {
        let total = self.total_measurements.load(Ordering::Relaxed);
        let sum = self.rtt_sum.load(Ordering::Relaxed);

        let mut best = u64::MAX;
        let mut worst = 0u64;

        for stats in &self.venue_stats {
            if stats.sample_count > 0 {
                if stats.min_rtt_ns < best {
                    best = stats.min_rtt_ns;
                }
                if stats.max_rtt_ns > worst {
                    worst = stats.max_rtt_ns;
                }
            }
        }

        LatencyStats {
            total_measurements: total,
            overall_avg_rtt_ns: if total > 0 { sum / total } else { 0 },
            best_rtt_ns: if best == u64::MAX { 0 } else { best },
            worst_rtt_ns: worst,
            out_of_order_events: self.out_of_order_count.load(Ordering::Relaxed),
            events_reordered: self.reordered_count.load(Ordering::Relaxed),
        }
    }

    /// Reset statistics for a venue
    #[inline]
    pub fn reset_venue(&self, venue_id: u32) {
        let idx = (venue_id as usize).min(MAX_VENUES - 1);
        self.venue_stats[idx] = VenueLatencyStats::new(venue_id);
        self.last_sequences[idx].store(0, Ordering::Release);
    }

    /// Get current timestamp in nanoseconds
    #[inline]
    fn get_timestamp_ns(&self) -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64
    }
}

impl Default for LatencyAdjuster {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_venue_latency_stats() {
        let mut stats = VenueLatencyStats::new(1);
        
        assert_eq!(stats.venue_id, 1);
        assert_eq!(stats.sample_count, 0);
        assert_eq!(stats.min_rtt_ns, u64::MAX);

        stats.update(1_000_000, 1234567890);
        assert_eq!(stats.sample_count, 1);
        assert_eq!(stats.min_rtt_ns, 1_000_000);
        assert_eq!(stats.max_rtt_ns, 1_000_000);
        assert_eq!(stats.one_way_latency_ns, 500_000);
    }

    #[test]
    fn test_latency_adjuster_creation() {
        let adjuster = LatencyAdjuster::new();
        let stats = adjuster.get_stats();

        assert_eq!(stats.total_measurements, 0);
        assert_eq!(stats.out_of_order_events, 0);
    }

    #[test]
    fn test_record_rtt() {
        let adjuster = LatencyAdjuster::new();

        adjuster.record_rtt(1_000_000, 1);
        adjuster.record_rtt(2_000_000, 1);
        adjuster.record_rtt(1_500_000, 1);

        let stats = adjuster.get_venue_stats(1);
        assert_eq!(stats.sample_count, 3);
        assert!(stats.avg_rtt_ns > 0);
    }

    #[test]
    fn test_true_event_time() {
        let adjuster = LatencyAdjuster::new();

        // Record some RTT first
        adjuster.record_rtt(2_000_000, 1);

        let exchange_ts = 1_000_000_000_000u64;
        let true_time = adjuster.calculate_true_event_time(exchange_ts, 1);

        // Should be exchange_ts + one_way_latency (approximately 1_000_000)
        assert!(true_time >= exchange_ts);
        assert!(true_time <= exchange_ts + 1_000_000);
    }

    #[test]
    fn test_adjusted_timestamp() {
        let ts = AdjustedTimestamp::new(1000, 900, 1)
            .with_transit(50);

        assert_eq!(ts.original_ns, 1000);
        assert_eq!(ts.exchange_ts_ns, 900);
        assert_eq!(ts.adjusted_ns, 950);
        assert_eq!(ts.transit_time_ns, 50);
        assert!(!ts.was_reordered);
    }

    #[test]
    fn test_reorder_detection() {
        let adjuster = LatencyAdjuster::new();

        // First message - normal
        assert!(!adjuster.detect_reorder(1000, 1));

        // Second message - later timestamp, normal
        assert!(!adjuster.detect_reorder(2000, 1));

        // Third message - earlier timestamp, out of order!
        assert!(adjuster.detect_reorder(1500, 1));

        let stats = adjuster.get_stats();
        assert_eq!(stats.out_of_order_events, 1);
    }

    #[test]
    fn test_reset_venue() {
        let adjuster = LatencyAdjuster::new();

        adjuster.record_rtt(1_000_000, 1);
        adjuster.detect_reorder(1000, 1);

        let before = adjuster.get_venue_stats(1);
        assert!(before.sample_count > 0);

        adjuster.reset_venue(1);

        let after = adjuster.get_venue_stats(1);
        assert_eq!(after.sample_count, 0);
    }
}
