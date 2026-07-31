//! Iceberg Sniping Execution Engine
//! 
//! Detects hidden liquidity via L3 trade tick anomalies and executes aggressive sweeps.
//! Front-runs dark pool replenishment to capture institutional size before market reprices.

use std::sync::atomic::{AtomicU64, AtomicI64, Ordering};
use std::time::{Duration, Instant};

/// Maximum number of tracked iceberg candidates
pub const MAX_ICEBERG_CANDIDATES: usize = 256;

/// Minimum trades to confirm iceberg pattern
pub const MIN_ICEBERG_TRADES: usize = 5;

/// Iceberg detection states
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum IcebergState {
    Unknown = 0,
    Suspected = 1,
    Confirmed = 2,
    Exhausted = 3,
}

/// Iceberg candidate structure
#[derive(Debug, Clone)]
pub struct IcebergCandidate {
    pub price_level: i64,          // Price in fixed-point Q16.48
    pub side: u8,                  // 0 = bid, 1 = ask
    pub initial_size: i64,         // Initial detected size
    pub executed_size: i64,        // Size already executed
    pub refresh_count: u32,        // Number of times refreshed
    pub last_refresh_time: u64,    // Timestamp of last refresh (nanos)
    pub trade_count: u32,          // Number of trades at this level
    pub state: IcebergState,
    pub confidence: f64,           // Confidence score [0, 1]
}

/// L3 trade tick for anomaly detection
#[derive(Debug, Clone, Copy)]
pub struct L3TradeTick {
    pub price: i64,
    pub size: i64,
    pub aggressor_side: u8,  // 0 = buyer, 1 = seller
    pub timestamp_ns: u64,
    pub trade_id: u64,
}

/// Iceberg sniping engine
pub struct IcebergSniper {
    /// Tracked iceberg candidates
    candidates: [Option<IcebergCandidate>; MAX_ICEBERG_CANDIDATES],
    /// Candidate count
    candidate_count: AtomicU64,
    /// Recent trade history (ring buffer)
    trade_history: [L3TradeTick; 1024],
    /// Trade history head
    trade_head: AtomicU64,
    /// Trade history count
    trade_count: AtomicU64,
    /// Sniper enabled flag
    enabled: AtomicU64,
    /// Minimum confidence threshold
    min_confidence: AtomicU64,  // Q16.48 fixed-point
    /// Aggressive sweep size multiplier
    sweep_multiplier: AtomicU64, // Q16.48 fixed-point
}

impl IcebergSniper {
    pub const fn new() -> Self {
        Self {
            candidates: [None; MAX_ICEBERG_CANDIDATES],
            candidate_count: AtomicU64::new(0),
            trade_history: [L3TradeTick {
                price: 0,
                size: 0,
                aggressor_side: 0,
                timestamp_ns: 0,
                trade_id: 0,
            }; 1024],
            trade_head: AtomicU64::new(0),
            trade_count: AtomicU64::new(0),
            enabled: AtomicU64::new(1),
            min_confidence: AtomicU64::new((0.7 * (1u64 << 48) as f64) as i64 as u64),
            sweep_multiplier: AtomicU64::new((1.5 * (1u64 << 48) as f64) as i64 as u64),
        }
    }
    
    /// Record a new L3 trade tick
    pub fn record_trade(&mut self, tick: L3TradeTick) {
        let head = self.trade_head.load(Ordering::Acquire);
        let idx = (head % 1024) as usize;
        
        self.trade_history[idx] = tick;
        self.trade_head.store(head + 1, Ordering::Release);
        
        if self.trade_count.load(Ordering::Acquire) < 1024 {
            self.trade_count.fetch_add(1, Ordering::Release);
        }
        
        // Check for iceberg patterns
        self.detect_iceberg(tick);
    }
    
    /// Detect iceberg patterns from trade flow
    fn detect_iceberg(&mut self, current_tick: L3TradeTick) {
        let trade_count = self.trade_count.load(Ordering::Acquire);
        if trade_count < MIN_ICEBERG_TRADES as u64 {
            return;
        }
        
        // Look for repeated executions at same price level
        let mut same_price_trades = 0u32;
        let mut total_size = 0i64;
        let head = self.trade_head.load(Ordering::Acquire);
        
        for i in 1..=trade_count.min(100) {
            let idx = ((head - i) % 1024) as usize;
            let trade = self.trade_history[idx];
            
            if trade.price == current_tick.price && 
               trade.aggressor_side == current_tick.aggressor_side {
                same_price_trades += 1;
                total_size += trade.size;
            }
        }
        
        // Iceberg detection logic
        if same_price_trades >= MIN_ICEBERG_TRADES as u32 {
            let confidence = (same_price_trades as f64 / 20.0).min(1.0);
            
            // Find or create candidate
            let existing_idx = self.find_candidate(current_tick.price, current_tick.aggressor_side);
            
            match existing_idx {
                Some(idx) => {
                    // Update existing candidate
                    if let Some(ref mut cand) = self.candidates[idx] {
                        cand.executed_size += current_tick.size;
                        cand.trade_count += 1;
                        cand.last_refresh_time = current_tick.timestamp_ns;
                        cand.confidence = confidence;
                        
                        // Check for refresh (size reset after execution)
                        if cand.executed_size > cand.initial_size {
                            cand.refresh_count += 1;
                            cand.state = IcebergState::Confirmed;
                        }
                    }
                }
                None => {
                    // Create new candidate
                    let count = self.candidate_count.load(Ordering::Acquire);
                    if count < MAX_ICEBERG_CANDIDATES as u64 {
                        let idx = count as usize;
                        self.candidates[idx] = Some(IcebergCandidate {
                            price_level: current_tick.price,
                            side: current_tick.aggressor_side,
                            initial_size: total_size,
                            executed_size: current_tick.size,
                            refresh_count: 0,
                            last_refresh_time: current_tick.timestamp_ns,
                            trade_count: same_price_trades,
                            state: if confidence > 0.8 { 
                                IcebergState::Confirmed 
                            } else { 
                                IcebergState::Suspected 
                            },
                            confidence,
                        });
                        self.candidate_count.store(count + 1, Ordering::Release);
                    }
                }
            }
        }
    }
    
    /// Find existing candidate by price and side
    fn find_candidate(&self, price: i64, side: u8) -> Option<usize> {
        let count = self.candidate_count.load(Ordering::Acquire);
        for i in 0..count as usize {
            if let Some(ref cand) = self.candidates[i] {
                if cand.price_level == price && cand.side == side {
                    return Some(i);
                }
            }
        }
        None
    }
    
    /// Generate sniper order for confirmed icebergs
    pub fn generate_snipe_order(&self) -> Option<SnipeOrder> {
        if self.enabled.load(Ordering::Acquire) == 0 {
            return None;
        }
        
        let min_conf = self.min_confidence.load(Ordering::Acquire) as i64 as f64 / (1u64 << 48) as f64;
        let sweep_mult = self.sweep_multiplier.load(Ordering::Acquire) as i64 as f64 / (1u64 << 48) as f64;
        
        let count = self.candidate_count.load(Ordering::Acquire);
        for i in 0..count as usize {
            if let Some(ref cand) = self.candidates[i] {
                if cand.state == IcebergState::Confirmed && cand.confidence >= min_conf {
                    // Calculate sweep size
                    let remaining = cand.initial_size - cand.executed_size;
                    let sweep_size = (remaining as f64 * sweep_mult) as i64;
                    
                    if sweep_size > 0 {
                        return Some(SnipeOrder {
                            price: cand.price_level,
                            side: cand.side,
                            size: sweep_size.max(remaining),
                            urgency: Urgency::High,
                            candidate_idx: i as u32,
                            confidence: cand.confidence,
                        });
                    }
                }
            }
        }
        None
    }
    
    /// Mark iceberg as exhausted after successful snipe
    pub fn mark_exhausted(&mut self, candidate_idx: usize) {
        if candidate_idx < MAX_ICEBERG_CANDIDATES {
            if let Some(ref mut cand) = self.candidates[candidate_idx] {
                cand.state = IcebergState::Exhausted;
            }
        }
    }
    
    /// Get best snipe opportunity
    pub fn get_best_opportunity(&self) -> Option<&IcebergCandidate> {
        let min_conf = self.min_confidence.load(Ordering::Acquire) as i64 as f64 / (1u64 << 48) as f64;
        let mut best: Option<&IcebergCandidate> = None;
        let mut best_score = 0.0f64;
        
        let count = self.candidate_count.load(Ordering::Acquire);
        for i in 0..count as usize {
            if let Some(ref cand) = self.candidates[i] {
                if cand.state == IcebergState::Confirmed && cand.confidence >= min_conf {
                    let score = cand.confidence * (cand.refresh_count + 1) as f64;
                    if score > best_score {
                        best_score = score;
                        best = Some(cand);
                    }
                }
            }
        }
        best
    }
    
    /// Enable/disable sniping
    #[inline]
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(if enabled { 1 } else { 0 }, Ordering::Release);
    }
    
    /// Set minimum confidence threshold
    #[inline]
    pub fn set_min_confidence(&self, threshold: f64) {
        let fixed = ((threshold.clamp(0.0, 1.0) * (1u64 << 48) as f64) as i64) as u64;
        self.min_confidence.store(fixed, Ordering::Release);
    }
    
    /// Set sweep size multiplier
    #[inline]
    pub fn set_sweep_multiplier(&self, multiplier: f64) {
        let fixed = ((multiplier.max(1.0) * (1u64 << 48) as f64) as i64) as u64;
        self.sweep_multiplier.store(fixed, Ordering::Release);
    }
    
    /// Get candidate count
    #[inline]
    pub fn candidate_count(&self) -> usize {
        self.candidate_count.load(Ordering::Acquire) as usize
    }
    
    /// Clean up exhausted candidates
    pub fn cleanup_exhausted(&mut self) {
        let count = self.candidate_count.load(Ordering::Acquire);
        let mut write_idx = 0;
        
        for read_idx in 0..count as usize {
            if let Some(ref cand) = self.candidates[read_idx] {
                if cand.state != IcebergState::Exhausted {
                    if write_idx != read_idx {
                        self.candidates[write_idx] = self.candidates[read_idx].clone();
                    }
                    write_idx += 1;
                }
            }
        }
        
        // Clear remaining slots
        for i in write_idx..count as usize {
            self.candidates[i] = None;
        }
        
        self.candidate_count.store(write_idx as u64, Ordering::Release);
    }
}

/// Snipe order structure
#[derive(Debug, Clone)]
pub struct SnipeOrder {
    pub price: i64,
    pub side: u8,
    pub size: i64,
    pub urgency: Urgency,
    pub candidate_idx: u32,
    pub confidence: f64,
}

/// Order urgency levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Urgency {
    Low = 0,
    Medium = 1,
    High = 2,
    Critical = 3,
}

/// Dark pool detection metrics
#[derive(Debug, Clone)]
pub struct DarkPoolMetrics {
    pub estimated_dark_volume: i64,
    pub dark_print_frequency: f64,
    pub replenishment_rate: f64,
    pub hidden_liquidity_ratio: f64,
}

impl DarkPoolMetrics {
    pub const fn new() -> Self {
        Self {
            estimated_dark_volume: 0,
            dark_print_frequency: 0.0,
            replenishment_rate: 0.0,
            hidden_liquidity_ratio: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_iceberg_detection() {
        let mut sniper = IcebergSniper::new();
        
        // Simulate repeated trades at same price (iceberg pattern)
        let base_time = 1000000000u64;
        for i in 0..10 {
            sniper.record_trade(L3TradeTick {
                price: 50000 << 48,  // $50,000
                size: 100,
                aggressor_side: 1,   // Seller
                timestamp_ns: base_time + i * 1000000,
                trade_id: i as u64,
            });
        }
        
        // Should detect iceberg
        assert!(sniper.candidate_count() > 0);
        
        // Check for snipe opportunity
        let order = sniper.generate_snipe_order();
        assert!(order.is_some());
        
        let ord = order.unwrap();
        assert_eq!(ord.price, 50000 << 48);
        assert_eq!(ord.side, 1);
        assert!(ord.confidence > 0.5);
    }
    
    #[test]
    fn test_iceberg_exhaustion() {
        let mut sniper = IcebergSniper::new();
        
        // Create iceberg
        for i in 0..10 {
            sniper.record_trade(L3TradeTick {
                price: 50000 << 48,
                size: 100,
                aggressor_side: 0,
                timestamp_ns: 1000000000 + i * 1000000,
                trade_id: i as u64,
            });
        }
        
        // Mark as exhausted
        sniper.mark_exhausted(0);
        
        // Should not generate order for exhausted iceberg
        sniper.cleanup_exhausted();
        assert_eq!(sniper.candidate_count(), 0);
    }
}
