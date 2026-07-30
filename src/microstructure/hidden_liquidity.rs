//! Hidden Liquidity Detection Module
//! Detects iceberg orders and hidden liquidity pools using trade tick size anomalies.
//! Correlates aggressive trade prints with L2 order book delta mismatches to uncover institutional dark pools.

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Cache line padding for AMD Ryzen cores
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

/// Represents a detected iceberg order
#[derive(Debug, Clone, Copy)]
pub struct IcebergOrder {
    /// Price level where iceberg is detected
    pub price: i64,
    /// Side: true for bid, false for ask
    pub is_bid: bool,
    /// Visible size (what shows in the order book)
    pub visible_size: u64,
    /// Estimated total size (including hidden portion)
    pub estimated_total_size: u64,
    /// Number of refreshes detected
    pub refresh_count: u32,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
    /// First detection timestamp (nanoseconds)
    pub first_seen_ns: u64,
    /// Last update timestamp
    pub last_update_ns: u64,
}

/// Trade tick record for anomaly detection
#[derive(Debug, Clone, Copy)]
pub struct TradeTick {
    /// Trade price
    pub price: i64,
    /// Trade size
    pub size: u64,
    /// Trade timestamp (nanoseconds)
    pub timestamp_ns: u64,
    /// Is this a buyer-initiated trade?
    pub is_buyer_maker: bool,
    /// Order book volume at this price before trade
    pub book_volume_before: u64,
}

/// Anomaly detection result
#[derive(Debug, Clone, Copy)]
pub struct LiquidityAnomaly {
    /// Type of anomaly detected
    pub anomaly_type: AnomalyType,
    /// Price level
    pub price: i64,
    /// Side
    pub is_bid: bool,
    /// Severity score (0.0 to 1.0)
    pub severity: f64,
    /// Timestamp of detection
    pub timestamp_ns: u64,
    /// Additional metadata
    pub metadata: AnomalyMetadata,
}

#[derive(Debug, Clone, Copy)]
pub enum AnomalyType {
    /// Iceberg order detected
    Iceberg,
    /// Hidden liquidity pool
    HiddenPool,
    /// Book delta mismatch
    DeltaMismatch,
    /// Unusual refresh pattern
    RefreshPattern,
}

#[derive(Debug, Clone, Copy)]
pub struct AnomalyMetadata {
    /// Expected volume vs actual volume
    pub expected_volume: u64,
    pub actual_volume: u64,
    /// Number of suspicious trades
    pub suspicious_trade_count: u32,
    /// Average trade size
    pub avg_trade_size: f64,
}

/// Lock-free hidden liquidity detector
pub struct HiddenLiquidityDetector {
    /// Recent trade ticks (circular buffer via indices)
    trade_buffer: Vec<CachePadded<AtomicU64>>,
    /// Buffer head index
    head_index: CachePadded<AtomicU64>,
    /// Buffer capacity
    capacity: u64,
    
    /// Detected icebergs (stored externally, this holds counters)
    iceberg_count: CachePadded<AtomicU64>,
    /// Active detection flag
    is_active: CachePadded<AtomicBool>,
    
    /// Price level tracking (simplified as atomic counters per level)
    /// In production, this would be a proper concurrent map
    refresh_events: CachePadded<AtomicU64>,
    mismatch_events: CachePadded<AtomicU64>,
    
    /// Configuration
    min_iceberg_confidence: f64,
    tick_window_ns: u64,
}

impl HiddenLiquidityDetector {
    /// Create new detector with specified buffer size (must be power of 2)
    pub fn new(buffer_size: usize, min_confidence: f64) -> Self {
        // Ensure power of 2
        let capacity = buffer_size.next_power_of_two() as u64;
        
        let mut trade_buffer = Vec::with_capacity(capacity as usize);
        for _ in 0..capacity {
            trade_buffer.push(CachePadded::default());
        }
        
        Self {
            trade_buffer,
            head_index: CachePadded::default(),
            capacity,
            iceberg_count: CachePadded::default(),
            is_active: CachePadded::new(AtomicBool::new(true)),
            refresh_events: CachePadded::default(),
            mismatch_events: CachePadded::default(),
            min_iceberg_confidence: min_confidence,
            tick_window_ns: Duration::from_secs(1).as_nanos() as u64,
        }
    }

    /// Record a trade tick for analysis
    #[inline]
    pub fn record_trade(&self, trade: TradeTick) -> Option<LiquidityAnomaly> {
        if !self.is_active.data.load(Ordering::Acquire) {
            return None;
        }

        // Store trade in circular buffer (using atomic for lock-free access)
        let head = self.head_index.data.fetch_add(1, Ordering::AcqRel);
        let index = (head & (self.capacity - 1)) as usize;
        
        // Encode trade as u64 tuple (simplified - in production use proper serialization)
        // This is a placeholder for the actual trade storage mechanism
        let _ = &self.trade_buffer[index];
        
        // Analyze for anomalies
        self.analyze_trade(trade)
    }

    /// Analyze a single trade for hidden liquidity signals
    fn analyze_trade(&self, trade: TradeTick) -> Option<LiquidityAnomaly> {
        // Check for iceberg signature:
        // 1. Trade size exceeds visible book volume
        // 2. Multiple trades at same price without book depletion
        // 3. Regular refresh patterns
        
        if trade.book_volume_before > 0 && trade.size > trade.book_volume_before {
            // Trade executed more than visible liquidity - hidden liquidity detected
            let excess = trade.size - trade.book_volume_before;
            let confidence = (excess as f64 / trade.size as f64).min(1.0);
            
            if confidence >= self.min_iceberg_confidence {
                self.iceberg_count.data.fetch_add(1, Ordering::AcqRel);
                
                return Some(LiquidityAnomaly {
                    anomaly_type: AnomalyType::Iceberg,
                    price: trade.price,
                    is_bid: !trade.is_buyer_maker, // If buyer maker, it's on bid side
                    severity: confidence,
                    timestamp_ns: trade.timestamp_ns,
                    metadata: AnomalyMetadata {
                        expected_volume: trade.book_volume_before,
                        actual_volume: trade.size,
                        suspicious_trade_count: 1,
                        avg_trade_size: trade.size as f64,
                    },
                });
            }
        }
        
        None
    }

    /// Process order book delta to detect mismatches
    pub fn analyze_book_delta(
        &self,
        price: i64,
        is_bid: bool,
        expected_volume: u64,
        actual_volume: u64,
        timestamp_ns: u64,
    ) -> Option<LiquidityAnomaly> {
        if !self.is_active.data.load(Ordering::Acquire) {
            return None;
        }

        // Significant mismatch indicates hidden liquidity or iceberg
        if actual_volume > expected_volume {
            let ratio = actual_volume as f64 / expected_volume.max(1) as f64;
            
            if ratio > 2.0 {
                // More than 2x expected volume - likely hidden liquidity
                self.mismatch_events.data.fetch_add(1, Ordering::AcqRel);
                
                let severity = ((ratio - 1.0) / ratio).min(1.0);
                
                return Some(LiquidityAnomaly {
                    anomaly_type: AnomalyType::DeltaMismatch,
                    price,
                    is_bid,
                    severity,
                    timestamp_ns,
                    metadata: AnomalyMetadata {
                        expected_volume,
                        actual_volume,
                        suspicious_trade_count: 0,
                        avg_trade_size: 0.0,
                    },
                });
            }
        } else if expected_volume > actual_volume && expected_volume > 100 {
            // Volume disappeared without trades - possible iceberg refresh
            let depletion_ratio = (expected_volume - actual_volume) as f64 / expected_volume as f64;
            
            if depletion_ratio > 0.5 {
                self.refresh_events.data.fetch_add(1, Ordering::AcqRel);
                
                return Some(LiquidityAnomaly {
                    anomaly_type: AnomalyType::RefreshPattern,
                    price,
                    is_bid,
                    severity: depletion_ratio,
                    timestamp_ns,
                    metadata: AnomalyMetadata {
                        expected_volume,
                        actual_volume,
                        suspicious_trade_count: 0,
                        avg_trade_size: 0.0,
                    },
                });
            }
        }
        
        None
    }

    /// Track multiple trades at same price level for iceberg detection
    pub fn detect_iceberg_pattern(
        &self,
        price: i64,
        is_bid: bool,
        trades: &[TradeTick],
        visible_size: u64,
    ) -> Option<IcebergOrder> {
        if trades.is_empty() || !self.is_active.data.load(Ordering::Acquire) {
            return None;
        }

        let total_volume: u64 = trades.iter().map(|t| t.size).sum();
        let trade_count = trades.len() as u32;
        
        // Iceberg indicators:
        // 1. Total volume >> visible size
        // 2. Multiple trades at exact same price
        // 3. Consistent trade sizes (institutional algorithm)
        
        if total_volume <= visible_size {
            return None;
        }

        let volume_ratio = total_volume as f64 / visible_size as f64;
        
        // Calculate size consistency (low variance = algorithmic)
        let avg_size = total_volume as f64 / trade_count as f64;
        let variance: f64 = trades.iter()
            .map(|t| {
                let diff = t.size as f64 - avg_size;
                diff * diff
            })
            .sum::<f64>() / trade_count as f64;
        
        let std_dev = variance.sqrt();
        let coefficient_of_variation = if avg_size > 0.0 { std_dev / avg_size } else { f64::INFINITY };
        
        // Confidence calculation
        let volume_confidence = (volume_ratio - 1.0).min(1.0);
        let pattern_confidence = (1.0 - coefficient_of_variation.min(1.0));
        let confidence = (volume_confidence * 0.6 + pattern_confidence * 0.4).min(1.0);
        
        if confidence < self.min_iceberg_confidence {
            return None;
        }

        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_nanos() as u64;

        Some(IcebergOrder {
            price,
            is_bid,
            visible_size,
            estimated_total_size: total_volume,
            refresh_count: 1,
            confidence,
            first_seen_ns: trades.first().map(|t| t.timestamp_ns).unwrap_or(now_ns),
            last_update_ns: now_ns,
        })
    }

    /// Estimate hidden liquidity at a price level
    pub fn estimate_hidden_liquidity(
        &self,
        price: i64,
        is_bid: bool,
        visible_size: u64,
        recent_trades: &[TradeTick],
    ) -> u64 {
        if recent_trades.is_empty() {
            return 0;
        }

        let total_executed: u64 = recent_trades.iter()
            .filter(|t| t.price == price)
            .map(|t| t.size)
            .sum();

        if total_executed <= visible_size {
            return 0;
        }

        // Hidden = executed - visible (minimum estimate)
        total_executed - visible_size
    }

    /// Get current statistics
    pub fn get_stats(&self) -> HiddenLiquidityStats {
        HiddenLiquidityStats {
            iceberg_count: self.iceberg_count.data.load(Ordering::Acquire),
            refresh_events: self.refresh_events.data.load(Ordering::Acquire),
            mismatch_events: self.mismatch_events.data.load(Ordering::Acquire),
            is_active: self.is_active.data.load(Ordering::Acquire),
        }
    }

    /// Enable/disable detection
    #[inline]
    pub fn set_active(&self, active: bool) {
        self.is_active.data.store(active, Ordering::Release);
    }

    /// Clear all counters
    #[inline]
    pub fn reset_counters(&self) {
        self.iceberg_count.data.store(0, Ordering::Release);
        self.refresh_events.data.store(0, Ordering::Release);
        self.mismatch_events.data.store(0, Ordering::Release);
    }
}

/// Statistics from the hidden liquidity detector
#[derive(Debug, Clone, Copy)]
pub struct HiddenLiquidityStats {
    pub iceberg_count: u64,
    pub refresh_events: u64,
    pub mismatch_events: u64,
    pub is_active: bool,
}

/// Dark pool correlation analyzer
pub struct DarkPoolCorrelator {
    /// Correlation window in nanoseconds
    window_ns: u64,
    /// Detected correlations
    correlation_count: CachePadded<AtomicU64>,
    /// Is active
    is_active: CachePadded<AtomicBool>,
}

impl DarkPoolCorrelator {
    pub fn new(window_ms: u64) -> Self {
        Self {
            window_ns: Duration::from_millis(window_ms).as_nanos() as u64,
            correlation_count: CachePadded::default(),
            is_active: CachePadded::new(AtomicBool::new(true)),
        }
    }

    /// Correlate trades across venues to detect dark pool activity
    pub fn correlate_trades(
        &self,
        venue_a_trades: &[TradeTick],
        venue_b_trades: &[TradeTick],
    ) -> Vec<DarkPoolSignal> {
        if !self.is_active.data.load(Ordering::Acquire) || venue_a_trades.is_empty() || venue_b_trades.is_empty() {
            return Vec::new();
        }

        let mut signals = Vec::new();

        // Look for near-simultaneous trades at similar prices
        for trade_a in venue_a_trades {
            for trade_b in venue_b_trades {
                let time_diff = trade_a.timestamp_ns.abs_diff(trade_b.timestamp_ns);
                
                if time_diff <= self.window_ns {
                    let price_diff = (trade_a.price as i64 - trade_b.price as i64).abs();
                    
                    // Same direction trades within tight time/price window
                    if trade_a.is_buyer_maker == trade_b.is_buyer_maker && price_diff < 10 {
                        self.correlation_count.data.fetch_add(1, Ordering::AcqRel);
                        
                        signals.push(DarkPoolSignal {
                            timestamp_ns: trade_a.timestamp_ns.min(trade_b.timestamp_ns),
                            venue_a_size: trade_a.size,
                            venue_b_size: trade_b.size,
                            total_size: trade_a.size + trade_b.size,
                            is_buy: !trade_a.is_buyer_maker,
                            confidence: 1.0 - (time_diff as f64 / self.window_ns as f64),
                        });
                    }
                }
            }
        }

        signals
    }

    /// Get correlation statistics
    pub fn correlation_count(&self) -> u64 {
        self.correlation_count.data.load(Ordering::Acquire)
    }
}

/// Signal indicating potential dark pool activity
#[derive(Debug, Clone, Copy)]
pub struct DarkPoolSignal {
    pub timestamp_ns: u64,
    pub venue_a_size: u64,
    pub venue_b_size: u64,
    pub total_size: u64,
    pub is_buy: bool,
    pub confidence: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iceberg_detection() {
        let detector = HiddenLiquidityDetector::new(1024, 0.5);
        
        let trade = TradeTick {
            price: 10000,
            size: 500,
            timestamp_ns: 1000000,
            is_buyer_maker: false,
            book_volume_before: 100,
        };

        let anomaly = detector.record_trade(trade);
        assert!(anomaly.is_some());
        
        let anomaly = anomaly.unwrap();
        assert_eq!(anomaly.anomaly_type, AnomalyType::Iceberg);
        assert!(anomaly.severity > 0.5);
    }

    #[test]
    fn test_book_delta_mismatch() {
        let detector = HiddenLiquidityDetector::new(1024, 0.5);
        
        let anomaly = detector.analyze_book_delta(
            10000,
            true,
            100,  // expected
            500,  // actual
            2000000,
        );

        assert!(anomaly.is_some());
        assert_eq!(anomaly.unwrap().anomaly_type, AnomalyType::DeltaMismatch);
    }

    #[test]
    fn test_iceberg_pattern() {
        let detector = HiddenLiquidityDetector::new(1024, 0.5);
        
        let trades = vec![
            TradeTick { price: 10000, size: 100, timestamp_ns: 1000, is_buyer_maker: false, book_volume_before: 50 },
            TradeTick { price: 10000, size: 100, timestamp_ns: 2000, is_buyer_maker: false, book_volume_before: 50 },
            TradeTick { price: 10000, size: 100, timestamp_ns: 3000, is_buyer_maker: false, book_volume_before: 50 },
            TradeTick { price: 10000, size: 100, timestamp_ns: 4000, is_buyer_maker: false, book_volume_before: 50 },
        ];

        let iceberg = detector.detect_iceberg_pattern(10000, true, &trades, 100);
        assert!(iceberg.is_some());
        
        let iceberg = iceberg.unwrap();
        assert!(iceberg.confidence > 0.5);
        assert_eq!(iceberg.estimated_total_size, 400);
    }

    #[test]
    fn test_dark_pool_correlation() {
        let correlator = DarkPoolCorrelator::new(100); // 100ms window
        
        let trades_a = vec![
            TradeTick { price: 10000, size: 100, timestamp_ns: 1000, is_buyer_maker: false, book_volume_before: 50 },
        ];
        let trades_b = vec![
            TradeTick { price: 10000, size: 150, timestamp_ns: 1050, is_buyer_maker: false, book_volume_before: 75 },
        ];

        let signals = correlator.correlate_trades(&trades_a, &trades_b);
        assert!(!signals.is_empty());
        assert_eq!(signals[0].total_size, 250);
    }
}
