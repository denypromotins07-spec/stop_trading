//! Stablecoin Premium Arbitrage Module
//!
//! Tracks USDT/USDC fiat premiums across regional exchanges (Kimchi premium, Indian premium, etc.)
//! Detects structural mispricings caused by localized fiat on/off ramp friction and capital controls.
//! Uses normalized L2 spreads to identify genuine arbitrage opportunities vs phantom premiums.

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Instant, Duration};
use crossbeam_channel::{bounded, Sender, Receiver};
use dashmap::DashMap;

/// Regional exchange identifiers for stablecoin tracking
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RegionalExchange {
    /// South Korean exchanges (Kimchi premium)
    Upbit,
    Bithumb,
    Korbit,
    /// Indian exchanges (Indian premium)
    WazirX,
    CoinDCX,
    ZebPay,
    /// Nigerian exchanges (Naira premium)
    Bundle,
    Quidax,
    /// Turkish exchanges (Lira premium)
    BtcTurk,
    Paribu,
    /// Argentine exchanges (Peso premium)
    SatoshiTango,
    BuenBit,
    /// Vietnamese exchanges (Dong premium)
    Remitano,
    Vndex,
    /// Generic international
    Binance,
    Coinbase,
    Kraken,
}

impl RegionalExchange {
    pub fn region(&self) -> &'static str {
        match self {
            Self::Upbit | Self::Bithumb | Self::Korbit => "South Korea",
            Self::WazirX | Self::CoinDCX | Self::ZebPay => "India",
            Self::Bundle | Self::Quidax => "Nigeria",
            Self::BtcTurk | Self::Paribu => "Turkey",
            Self::SatoshiTango | Self::BuenBit => "Argentina",
            Self::Remitano | Self::Vndex => "Vietnam",
            Self::Binance | Self::Coinbase | Self::Kraken => "International",
        }
    }

    pub fn withdrawal_limits_usd(&self) -> f64 {
        // Approximate daily withdrawal limits in USD
        match self {
            Self::Upbit => 50_000.0,
            Self::Bithumb => 30_000.0,
            Self::Korbit => 25_000.0,
            Self::WazirX => 10_000.0,
            Self::CoinDCX => 15_000.0,
            Self::ZebPay => 8_000.0,
            Self::Bundle => 5_000.0,
            Self::Quidax => 5_000.0,
            Self::BtcTurk => 20_000.0,
            Self::Paribu => 15_000.0,
            Self::SatoshiTango => 3_000.0,
            Self::BuenBit => 2_000.0,
            Self::Remitano => 10_000.0,
            Self::Vndex => 8_000.0,
            Self::Binance => 100_000.0,
            Self::Coinbase => 250_000.0,
            Self::Kraken => 500_000.0,
        }
    }

    pub fn typical_transfer_delay_minutes(&self) -> u32 {
        match self {
            Self::Upbit | Self::Bithumb | Self::Korbit => 60,  // KYC delays
            Self::WazirX | Self::CoinDCX | Self::ZebPay => 120, // Banking integration issues
            Self::Bundle | Self::Quidax => 180,                 // Regulatory scrutiny
            Self::BtcTurk | Self::Paribu => 90,
            Self::SatoshiTango | Self::BuenBit => 240,          // Capital controls
            Self::Remitano | Self::Vndex => 150,
            Self::Binance | Self::Coinbase | Self::Kraken => 30,
        }
    }
}

/// Stablecoin pair types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StablecoinPair {
    USDTUSD,
    USDCUSD,
    USDTUSDC,
    DAIUSD,
    BUSDUSD,
}

impl StablecoinPair {
    pub fn base(&self) -> &'static str {
        match self {
            Self::USDTUSD => "USDT",
            Self::USDCUSD => "USDC",
            Self::USDTUSDC => "USDT",
            Self::DAIUSD => "DAI",
            Self::BUSDUSD => "BUSD",
        }
    }

    pub fn quote(&self) -> &'static str {
        match self {
            Self::USDTUSD => "USD",
            Self::USDCUSD => "USD",
            Self::USDTUSDC => "USDC",
            Self::DAIUSD => "USD",
            Self::BUSDUSD => "USD",
        }
    }
}

/// L2 Order Book snapshot for spread calculation
#[derive(Debug, Clone)]
pub struct L2Snapshot {
    pub exchange: RegionalExchange,
    pub pair: StablecoinPair,
    pub best_bid: f64,
    pub best_ask: f64,
    pub bid_depth: f64,
    pub ask_depth: f64,
    pub timestamp_ns: u64,
}

impl L2Snapshot {
    pub fn mid_price(&self) -> f64 {
        (self.best_bid + self.best_ask) / 2.0
    }

    pub fn spread_bps(&self) -> f64 {
        if self.best_bid <= 0.0 || self.best_ask <= 0.0 {
            return f64::MAX;
        }
        ((self.best_ask - self.best_bid) / self.mid_price()) * 10000.0
    }

    pub fn is_valid(&self) -> bool {
        self.best_bid > 0.0 
            && self.best_ask > 0.0 
            && self.best_ask > self.best_bid
            && self.spread_bps() < 1000.0 // Sanity check: < 10% spread
    }
}

/// Normalized spread accounting for depth and volatility
#[derive(Debug, Clone)]
pub struct NormalizedSpread {
    pub raw_spread_bps: f64,
    pub depth_adjusted_spread_bps: f64,
    pub volume_weighted_spread_bps: f64,
    pub confidence_score: f64, // 0.0 to 1.0
}

impl NormalizedSpread {
    pub fn from_snapshot(snapshot: &L2Snapshot, target_volume: f64) -> Self {
        let raw_spread = snapshot.spread_bps();
        
        // Depth adjustment: penalize wide spreads with shallow depth
        let depth_factor = if snapshot.bid_depth > 0.0 && snapshot.ask_depth > 0.0 {
            let min_depth = snapshot.bid_depth.min(snapshot.ask_depth);
            (target_volume / min_depth).min(10.0) // Cap at 10x
        } else {
            10.0
        };
        
        let depth_adjusted = raw_spread * depth_factor.sqrt();
        
        // Volume-weighted spread (estimate slippage)
        let vwap_spread = Self::estimate_slippage_bps(snapshot, target_volume);
        
        // Confidence based on depth and recency
        let now_ns = Instant::now().duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default().as_nanos() as u64;
        let age_ms = (now_ns - snapshot.timestamp_ns) as f64 / 1e6;
        let freshness_factor = (-age_ms / 5000.0).exp(); // Decay over 5 seconds
        
        let depth_confidence = (snapshot.bid_depth.min(snapshot.ask_depth) / target_volume).min(1.0);
        let confidence = freshness_factor * depth_confidence;
        
        Self {
            raw_spread_bps: raw_spread,
            depth_adjusted_spread_bps: depth_adjusted,
            volume_weighted_spread_bps: vwap_spread,
            confidence_score: confidence.clamp(0.0, 1.0),
        }
    }

    fn estimate_slippage_bps(snapshot: &L2Snapshot, volume: f64) -> f64 {
        // Simple linear slippage model
        // In production, this would use actual L2 depth data
        let avg_depth = (snapshot.bid_depth + snapshot.ask_depth) / 2.0;
        if avg_depth <= 0.0 {
            return f64::MAX;
        }
        let slippage_ratio = volume / avg_depth;
        slippage_ratio * snapshot.spread_bps() * 0.5
    }
}

/// Premium signal detected between regional exchanges
#[derive(Debug, Clone)]
pub struct PremiumSignal {
    pub pair: StablecoinPair,
    pub source_exchange: RegionalExchange,
    pub target_exchange: RegionalExchange,
    pub premium_bps: f64,
    pub risk_adjusted_premium_bps: f64,
    pub max_executable_volume: f64,
    pub estimated_profit_usd: f64,
    pub transfer_delay_minutes: u32,
    pub withdrawal_limit_usd: f64,
    pub confidence_score: f64,
    pub is_phantom: bool,
    pub phantom_reason: Option<&'static str>,
    pub timestamp_ns: u64,
}

impl PremiumSignal {
    pub fn new(
        source: L2Snapshot,
        target: L2Snapshot,
        target_volume: f64,
    ) -> Option<Self> {
        if !source.is_valid() || !target.is_valid() {
            return None;
        }

        let source_mid = source.mid_price();
        let target_mid = target.mid_price();
        
        // Calculate premium (positive = target is more expensive)
        let premium_bps = ((target_mid - source_mid) / source_mid) * 10000.0;
        
        // Get normalized spreads
        let source_norm = NormalizedSpread::from_snapshot(&source, target_volume);
        let target_norm = NormalizedSpread::from_snapshot(&target, target_volume);
        
        // Risk-adjusted premium accounts for execution costs
        let execution_cost_bps = source_norm.volume_weighted_spread_bps 
            + target_norm.volume_weighted_spread_bps;
        let risk_adjusted_premium = premium_bps - execution_cost_bps;
        
        // Determine max executable volume
        let source_limit = source.withdrawal_limit_usd();
        let target_limit = target.exchange.withdrawal_limits_usd();
        let depth_limit = source.bid_depth.min(target.ask_depth);
        let max_volume = source_limit.min(target_limit).min(depth_limit);
        
        // Estimate profit
        let estimated_profit = (risk_adjusted_premium / 10000.0) * max_volume * source_mid;
        
        // Transfer delay
        let transfer_delay = source.exchange.typical_transfer_delay_minutes()
            + target.exchange.typical_transfer_delay_minutes();
        
        // Check for phantom premium conditions
        let (is_phantom, phantom_reason) = Self::detect_phantom_premium(
            &source, &target, premium_bps, max_volume, transfer_delay,
        );
        
        // Confidence score
        let confidence = (source_norm.confidence_score * target_norm.confidence_score).sqrt();
        
        Some(Self {
            pair: source.pair,
            source_exchange: source.exchange,
            target_exchange: target.exchange,
            premium_bps,
            risk_adjusted_premium_bps: risk_adjusted_premium,
            max_executable_volume: max_volume,
            estimated_profit_usd: estimated_profit.max(0.0),
            transfer_delay_minutes: transfer_delay,
            withdrawal_limit_usd: max_volume,
            confidence_score: confidence,
            is_phantom,
            phantom_reason,
            timestamp_ns: Instant::now().duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default().as_nanos() as u64,
        })
    }

    fn detect_phantom_premium(
        source: &L2Snapshot,
        target: &L2Snapshot,
        premium_bps: f64,
        max_volume: f64,
        transfer_delay: u32,
    ) -> (bool, Option<&'static str>) {
        // Phantom detection heuristics
        
        // 1. Extreme premium (> 5%) likely indicates data error or frozen market
        if premium_bps.abs() > 500.0 {
            return (true, Some("EXTREME_PREMIUM"));
        }
        
        // 2. Insufficient depth for meaningful execution
        if max_volume < 100.0 {
            return (true, Some("INSUFFICIENT_LIQUIDITY"));
        }
        
        // 3. Excessive transfer delay makes arb infeasible
        if transfer_delay > 480 {
            return (true, Some("EXCESSIVE_TRANSFER_DELAY"));
        }
        
        // 4. Wide spreads indicate illiquid/stressed market
        if source.spread_bps() > 100.0 || target.spread_bps() > 100.0 {
            return (true, Some("EXCESSIVE_SPREAD"));
        }
        
        // 5. Stale data
        let now_ns = Instant::now().duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default().as_nanos() as u64;
        if now_ns - source.timestamp_ns > 10_000_000_000 
            || now_ns - target.timestamp_ns > 10_000_000_000 
        {
            return (true, Some("STALE_DATA"));
        }
        
        (false, None)
    }
}

/// Premium tracker state
#[derive(Debug, Clone)]
pub struct PremiumTrackerState {
    pub signals_detected: u64,
    pub phantom_signals_filtered: u64,
    pub profitable_signals: u64,
    pub total_premium_captured_bps: f64,
    pub last_signal_timestamp_ns: u64,
}

/// Main premium tracker engine
pub struct StablecoinPremiumTracker {
    /// Latest L2 snapshots per exchange/pair
    snapshots: DashMap<(RegionalExchange, StablecoinPair), L2Snapshot>,
    /// Detected premium signals
    signals: DashMap<u64, PremiumSignal>,
    /// Signal counter
    signal_counter: AtomicU64,
    /// Tracker state
    state: Arc<PremiumTrackerState>,
    /// Is tracker active
    is_active: AtomicBool,
    /// Minimum confidence threshold
    min_confidence: f64,
    /// Minimum risk-adjusted premium threshold (bps)
    min_premium_bps: f64,
    /// Event channel
    event_tx: Sender<PremiumEvent>,
    event_rx: Receiver<PremiumEvent>,
}

/// Premium events
#[derive(Debug, Clone)]
pub enum PremiumEvent {
    /// New L2 snapshot received
    SnapshotUpdated(RegionalExchange, StablecoinPair),
    /// Premium signal detected
    SignalDetected(PremiumSignal),
    /// Phantom premium filtered out
    PhantomFiltered {
        source: RegionalExchange,
        target: RegionalExchange,
        reason: &'static str,
    },
    /// Confidence threshold breach
    LowConfidence {
        exchange: RegionalExchange,
        confidence: f64,
    },
}

impl StablecoinPremiumTracker {
    pub fn new(
        min_confidence: f64,
        min_premium_bps: f64,
        buffer_size: usize,
    ) -> Self {
        let (tx, rx) = bounded(buffer_size);
        
        Self {
            snapshots: DashMap::new(),
            signals: DashMap::new(),
            signal_counter: AtomicU64::new(0),
            state: Arc::new(PremiumTrackerState {
                signals_detected: 0,
                phantom_signals_filtered: 0,
                profitable_signals: 0,
                total_premium_captured_bps: 0.0,
                last_signal_timestamp_ns: 0,
            }),
            is_active: AtomicBool::new(true),
            min_confidence,
            min_premium_bps,
            event_tx: tx,
            event_rx: rx,
        }
    }

    /// Update L2 snapshot for an exchange/pair
    pub fn update_snapshot(&self, snapshot: L2Snapshot) {
        if !snapshot.is_valid() {
            return;
        }

        let key = (snapshot.exchange, snapshot.pair);
        self.snapshots.insert(key, snapshot.clone());
        
        let _ = self.event_tx.send(PremiumEvent::SnapshotUpdated(
            snapshot.exchange, snapshot.pair
        ));
        
        // Trigger premium scan against all other exchanges
        self.scan_for_premiums(&snapshot);
    }

    /// Scan for premiums against a given snapshot
    fn scan_for_premiums(&self, reference: &L2Snapshot) {
        if !self.is_active.load(Ordering::Relaxed) {
            return;
        }

        let target_volume = 10_000.0; // Default scan volume: $10k
        
        for entry in self.snapshots.iter() {
            let (exchange, pair) = entry.key();
            
            // Only compare same pair across different exchanges
            if pair != &reference.pair || exchange == &reference.exchange {
                continue;
            }

            let other = entry.value();
            
            // Check both directions
            if let Some(signal) = PremiumSignal::new(reference.clone(), other.clone(), target_volume) {
                self.process_signal(signal);
            }
            
            if let Some(signal) = PremiumSignal::new(other.clone(), reference.clone(), target_volume) {
                self.process_signal(signal);
            }
        }
    }

    /// Process a detected premium signal
    fn process_signal(&self, mut signal: PremiumSignal) {
        // Filter by confidence
        if signal.confidence_score < self.min_confidence {
            let _ = self.event_tx.send(PremiumEvent::LowConfidence {
                exchange: signal.target_exchange,
                confidence: signal.confidence_score,
            });
            return;
        }

        // Handle phantom premiums
        if signal.is_phantom {
            if let Some(reason) = signal.phantom_reason {
                let _ = self.event_tx.send(PremiumEvent::PhantomFiltered {
                    source: signal.source_exchange,
                    target: signal.target_exchange,
                    reason,
                });
            }
            return;
        }

        // Filter by minimum premium threshold
        if signal.risk_adjusted_premium_bps < self.min_premium_bps {
            return;
        }

        // Valid signal - store and emit
        let signal_id = self.signal_counter.fetch_add(1, Ordering::Relaxed);
        signal.timestamp_ns = Instant::now().duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default().as_nanos() as u64;
        
        self.signals.insert(signal_id, signal.clone());
        
        let _ = self.event_tx.send(PremiumEvent::SignalDetected(signal));
    }

    /// Get latest snapshot for exchange/pair
    pub fn get_snapshot(&self, exchange: RegionalExchange, pair: StablecoinPair) -> Option<L2Snapshot> {
        self.snapshots.get(&(exchange, pair)).map(|s| s.clone())
    }

    /// Get all current premium signals
    pub fn get_active_signals(&self) -> Vec<PremiumSignal> {
        self.signals.iter()
            .filter(|e| !e.value().is_phantom)
            .map(|e| e.value().clone())
            .collect()
    }

    /// Get best signal by risk-adjusted premium
    pub fn get_best_signal(&self) -> Option<PremiumSignal> {
        self.signals.iter()
            .filter(|e| !e.value().is_phantom)
            .max_by(|a, b| {
                a.value().risk_adjusted_premium_bps
                    .partial_cmp(&b.value().risk_adjusted_premium_bps)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|e| e.value().clone())
    }

    /// Get event receiver
    pub fn get_event_receiver(&self) -> Receiver<PremiumEvent> {
        self.event_rx.clone()
    }

    /// Deactivate tracker
    pub fn deactivate(&self) {
        self.is_active.store(false, Ordering::Relaxed);
    }

    /// Activate tracker
    pub fn activate(&self) {
        self.is_active.store(true, Ordering::Relaxed);
    }

    /// Clear old signals (older than specified duration)
    pub fn clear_old_signals(&self, max_age: Duration) {
        let now_ns = Instant::now().duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default().as_nanos() as u64;
        let max_age_ns = max_age.as_nanos() as u64;
        
        self.signals.retain(|_, signal| {
            now_ns - signal.timestamp_ns < max_age_ns
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l2_snapshot_validation() {
        let valid_snapshot = L2Snapshot {
            exchange: RegionalExchange::Upbit,
            pair: StablecoinPair::USDTUSD,
            best_bid: 0.9998,
            best_ask: 1.0002,
            bid_depth: 100_000.0,
            ask_depth: 100_000.0,
            timestamp_ns: Instant::now().duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default().as_nanos() as u64,
        };

        assert!(valid_snapshot.is_valid());
        assert!((valid_snapshot.spread_bps() - 4.0).abs() < 0.1);

        let invalid_snapshot = L2Snapshot {
            best_bid: 0.0,
            ..valid_snapshot.clone()
        };
        assert!(!invalid_snapshot.is_valid());
    }

    #[test]
    fn test_premium_signal_detection() {
        let now_ns = Instant::now().duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default().as_nanos() as u64;

        // Simulate Kimchi premium: USDT more expensive in Korea
        let international = L2Snapshot {
            exchange: RegionalExchange::Binance,
            pair: StablecoinPair::USDTUSD,
            best_bid: 0.9999,
            best_ask: 1.0001,
            bid_depth: 1_000_000.0,
            ask_depth: 1_000_000.0,
            timestamp_ns: now_ns,
        };

        let korea = L2Snapshot {
            exchange: RegionalExchange::Upbit,
            pair: StablecoinPair::USDTUSD,
            best_bid: 1.0050, // ~50 bps premium
            best_ask: 1.0055,
            bid_depth: 500_000.0,
            ask_depth: 500_000.0,
            timestamp_ns: now_ns,
        };

        let signal = PremiumSignal::new(international.clone(), korea.clone(), 10_000.0)
            .expect("Should detect premium signal");

        assert!(signal.premium_bps > 40.0); // Should detect ~50 bps premium
        assert!(!signal.is_phantom);
        assert_eq!(signal.source_exchange, RegionalExchange::Binance);
        assert_eq!(signal.target_exchange, RegionalExchange::Upbit);
    }

    #[test]
    fn test_phantom_premium_detection() {
        let now_ns = Instant::now().duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default().as_nanos() as u64;

        // Create extreme premium scenario (likely data error)
        let normal = L2Snapshot {
            exchange: RegionalExchange::Binance,
            pair: StablecoinPair::USDTUSD,
            best_bid: 0.9999,
            best_ask: 1.0001,
            bid_depth: 1_000_000.0,
            ask_depth: 1_000_000.0,
            timestamp_ns: now_ns,
        };

        let extreme = L2Snapshot {
            exchange: RegionalExchange::Upbit,
            pair: StablecoinPair::USDTUSD,
            best_bid: 1.10, // 10% premium - unrealistic
            best_ask: 1.15,
            bid_depth: 1_000.0,
            ask_depth: 1_000.0,
            timestamp_ns: now_ns,
        };

        let signal = PremiumSignal::new(normal, extreme, 10_000.0)
            .expect("Should create signal");

        assert!(signal.is_phantom);
        assert_eq!(signal.phantom_reason, Some("EXTREME_PREMIUM"));
    }

    #[test]
    fn test_tracker_initialization() {
        let tracker = StablecoinPremiumTracker::new(0.5, 5.0, 1000);

        assert!(tracker.is_active.load(Ordering::Relaxed));
        assert_eq!(tracker.get_active_signals().len(), 0);
    }
}
