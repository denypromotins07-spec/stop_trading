//! Smart Money Concepts Module Root
//! 
//! Aggregates Break of Structure (BOS) and Change of Character (CHoCH) logic.
//! Exports order block and FVG detection for the alpha engine.

pub mod order_blocks;
pub mod fvg;

pub use order_blocks::{
    Candle, OrderBlock, OrderBlockType, OrderBlockDetector, OrderBlockSignal,
    OrderBlockAction, ZoneType, OrderBlockError,
};
pub use fvg::{
    FvgCandle, FairValueGap, FvgType, FvgDetector, FvgSignal, FvgAction,
    LiquidityVoid, NormalizedTick, FvgError,
};

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use thiserror::Error;

/// SMC-specific errors
#[derive(Debug, Error)]
pub enum SmcError {
    #[error("Invalid swing point data: {0}")]
    InvalidSwingData(String),
    #[error("Insufficient price history")]
    InsufficientHistory,
    #[error("Structure analysis error: {0}")]
    StructureError(String),
}

/// Market structure direction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarketStructure {
    Bullish, // Higher highs, higher lows
    Bearish, // Lower highs, lower lows
    Ranging, // No clear direction
}

/// Break of Structure (BOS) event
#[derive(Debug, Clone, Copy)]
pub struct BreakOfStructure {
    pub direction: MarketStructure,
    pub break_price: f64,
    pub previous_level: f64,
    pub timestamp_ns: u64,
    pub confirmed: bool,
    pub strength: f64, // 0.0 to 1.0
}

impl BreakOfStructure {
    pub fn new(
        direction: MarketStructure,
        break_price: f64,
        previous_level: f64,
        timestamp_ns: u64,
    ) -> Self {
        Self {
            direction,
            break_price,
            previous_level,
            timestamp_ns,
            confirmed: false,
            strength: 0.5,
        }
    }

    pub fn mark_confirmed(&mut self, strength: f64) {
        self.confirmed = true;
        self.strength = strength.clamp(0.0, 1.0);
    }
}

/// Change of Character (CHoCH) event - potential trend reversal
#[derive(Debug, Clone, Copy)]
pub struct ChangeOfCharacter {
    pub from_structure: MarketStructure,
    pub to_structure: MarketStructure,
    pub change_price: f64,
    pub timestamp_ns: u64,
    pub confirmed: bool,
}

impl ChangeOfCharacter {
    pub fn new(
        from_structure: MarketStructure,
        to_structure: MarketStructure,
        change_price: f64,
        timestamp_ns: u64,
    ) -> Self {
        Self {
            from_structure,
            to_structure,
            change_price,
            timestamp_ns,
            confirmed: false,
        }
    }

    pub fn mark_confirmed(&mut self) {
        self.confirmed = true;
    }
}

/// Swing point in price action
#[derive(Debug, Clone, Copy)]
pub struct SwingPoint {
    pub price: f64,
    pub timestamp_ns: u64,
    pub is_high: bool,
    pub strength: u8, // Number of candles confirming the swing
}

impl SwingPoint {
    pub fn new(price: f64, timestamp_ns: u64, is_high: bool, strength: u8) -> Self {
        Self {
            price,
            timestamp_ns,
            is_high,
            strength,
        }
    }
}

/// Structure analysis engine for BOS and CHoCH detection
pub struct StructureAnalyzer {
    /// Current market structure
    current_structure: AtomicU64, // Encoded as u8: 0=Ranging, 1=Bullish, 2=Bearish
    /// Last significant high
    last_high: AtomicU64,
    /// Last significant low
    last_low: AtomicU64,
    /// Price scale factor
    price_scale: i64,
    /// Minimum swing strength
    min_swing_strength: u8,
    /// Active flag
    active: AtomicBool,
}

impl StructureAnalyzer {
    /// Create a new structure analyzer
    pub fn new(min_swing_strength: u8) -> Self {
        Self {
            current_structure: AtomicU64::new(MarketStructure::Ranging as u64),
            last_high: AtomicU64::new(0),
            last_low: AtomicU64::new(u64::MAX),
            price_scale: 1_000_000_000,
            min_swing_strength,
            active: AtomicBool::new(true),
        }
    }

    /// Set price scale factor
    pub fn set_price_scale(&self, scale: i64) {
        self.price_scale = scale;
    }

    fn encode_price(&self, price: f64) -> u64 {
        (price * self.price_scale as f64) as u64
    }

    fn decode_price(&self, encoded: u64) -> f64 {
        encoded as f64 / self.price_scale as f64
    }

    /// Update with new swing high
    pub fn update_swing_high(&self, price: f64, timestamp_ns: u64, strength: u8) {
        if strength >= self.min_swing_strength {
            self.last_high.store(self.encode_price(price), Ordering::Relaxed);
        }
    }

    /// Update with new swing low
    pub fn update_swing_low(&self, price: f64, timestamp_ns: u64, strength: u8) {
        if strength >= self.min_swing_strength {
            self.last_low.store(self.encode_price(price), Ordering::Relaxed);
        }
    }

    /// Check for Break of Structure
    pub fn check_bos(
        &self,
        current_price: f64,
        timestamp_ns: u64,
        swing_points: &[SwingPoint],
    ) -> Option<BreakOfStructure> {
        if swing_points.is_empty() {
            return None;
        }

        let current_structure = self.get_current_structure();

        // Find relevant swing levels
        let last_significant_high = swing_points
            .iter()
            .filter(|sp| sp.is_high)
            .max_by_key(|sp| sp.timestamp_ns);

        let last_significant_low = swing_points
            .iter()
            .filter(|sp| !sp.is_high)
            .max_by_key(|sp| sp.timestamp_ns);

        // Check for bullish BOS (breaking above previous high)
        if let Some(prev_high) = last_significant_high {
            if current_price > prev_high.price && current_structure != MarketStructure::Bullish {
                let mut bos = BreakOfStructure::new(
                    MarketStructure::Bullish,
                    current_price,
                    prev_high.price,
                    timestamp_ns,
                );
                
                // Calculate strength based on how far price broke through
                let break_distance = (current_price - prev_high.price) / prev_high.price;
                bos.mark_confirmed((break_distance * 100.0).min(1.0));
                
                return Some(bos);
            }
        }

        // Check for bearish BOS (breaking below previous low)
        if let Some(prev_low) = last_significant_low {
            if current_price < prev_low.price && current_structure != MarketStructure::Bearish {
                let mut bos = BreakOfStructure::new(
                    MarketStructure::Bearish,
                    current_price,
                    prev_low.price,
                    timestamp_ns,
                );
                
                let break_distance = (prev_low.price - current_price) / prev_low.price;
                bos.mark_confirmed((break_distance * 100.0).min(1.0));
                
                return Some(bos);
            }
        }

        None
    }

    /// Check for Change of Character
    pub fn check_choch(
        &self,
        current_price: f64,
        timestamp_ns: u64,
        swing_points: &[SwingPoint],
    ) -> Option<ChangeOfCharacter> {
        if swing_points.len() < 4 {
            return None; // Need enough history for CHoCH
        }

        let current_structure = self.get_current_structure();

        match current_structure {
            MarketStructure::Bullish => {
                // Check if price breaks below last significant low (potential reversal)
                let lows: Vec<&SwingPoint> = swing_points.iter().filter(|sp| !sp.is_high).collect();
                if lows.len() >= 2 {
                    let last_low = lows[lows.len() - 1];
                    let prev_low = lows[lows.len() - 2];

                    if current_price < prev_low.price {
                        let mut choch = ChangeOfCharacter::new(
                            MarketStructure::Bullish,
                            MarketStructure::Bearish,
                            current_price,
                            timestamp_ns,
                        );
                        choch.mark_confirmed();
                        return Some(choch);
                    }
                }
            }
            MarketStructure::Bearish => {
                // Check if price breaks above last significant high (potential reversal)
                let highs: Vec<&SwingPoint> = swing_points.iter().filter(|sp| sp.is_high).collect();
                if highs.len() >= 2 {
                    let last_high = highs[highs.len() - 1];
                    let prev_high = highs[highs.len() - 2];

                    if current_price > prev_high.price {
                        let mut choch = ChangeOfCharacter::new(
                            MarketStructure::Bearish,
                            MarketStructure::Bullish,
                            current_price,
                            timestamp_ns,
                        );
                        choch.mark_confirmed();
                        return Some(choch);
                    }
                }
            }
            _ => {}
        }

        None
    }

    /// Get current market structure
    pub fn get_current_structure(&self) -> MarketStructure {
        match self.current_structure.load(Ordering::Relaxed) {
            1 => MarketStructure::Bullish,
            2 => MarketStructure::Bearish,
            _ => MarketStructure::Ranging,
        }
    }

    /// Update market structure
    pub fn set_structure(&self, structure: MarketStructure) {
        self.current_structure.store(structure as u64, Ordering::Relaxed);
    }

    /// Get last significant high
    pub fn get_last_high(&self) -> Option<f64> {
        let encoded = self.last_high.load(Ordering::Relaxed);
        if encoded == 0 {
            None
        } else {
            Some(self.decode_price(encoded))
        }
    }

    /// Get last significant low
    pub fn get_last_low(&self) -> Option<f64> {
        let encoded = self.last_low.load(Ordering::Relaxed);
        if encoded == u64::MAX {
            None
        } else {
            Some(self.decode_price(encoded))
        }
    }

    /// Activate/deactivate analyzer
    pub fn set_active(&self, active: bool) {
        self.active.store(active, Ordering::Relaxed);
    }

    pub fn is_active(&self) -> bool {
        self.active.load(Ordering::Relaxed)
    }
}

impl Default for StructureAnalyzer {
    fn default() -> Self {
        Self::new(2)
    }
}

/// Combined SMC state for the alpha engine
#[derive(Debug, Clone)]
pub struct SmcState {
    pub structure: MarketStructure,
    pub order_blocks: Vec<OrderBlock>,
    pub fvgs: Vec<FairValueGap>,
    pub last_bos: Option<BreakOfStructure>,
    pub last_choch: Option<ChangeOfCharacter>,
    pub timestamp_ns: u64,
}

impl SmcState {
    pub fn new(
        structure: MarketStructure,
        order_blocks: Vec<OrderBlock>,
        fvgs: Vec<FairValueGap>,
        timestamp_ns: u64,
    ) -> Self {
        Self {
            structure,
            order_blocks,
            fvgs,
            last_bos: None,
            last_choch: None,
            timestamp_ns,
        }
    }

    /// Check for confluence between SMC signals
    pub fn has_confluence(&self) -> SmcConfluence {
        let bullish_ob_count = self.order_blocks.iter()
            .filter(|ob| matches!(ob.block_type, OrderBlockType::Bullish | OrderBlockType::MitigationBullish))
            .filter(|ob| !ob.mitigated)
            .count();

        let bearish_ob_count = self.order_blocks.iter()
            .filter(|ob| matches!(ob.block_type, OrderBlockType::Bearish | OrderBlockType::MitigationBearish))
            .filter(|ob| !ob.mitigated)
            .count();

        let bullish_fvg_count = self.fvgs.iter()
            .filter(|fvg| fvg.fvg_type == FvgType::Bullish && !fvg.filled)
            .count();

        let bearish_fvg_count = self.fvgs.iter()
            .filter(|fvg| fvg.fvg_type == FvgType::Bearish && !fvg.filled)
            .count();

        let bullish_score = bullish_ob_count + bullish_fvg_count;
        let bearish_score = bearish_ob_count + bearish_fvg_count;

        if bullish_score > bearish_score + 1 {
            SmcConfluence::Bullish
        } else if bearish_score > bullish_score + 1 {
            SmcConfluence::Bearish
        } else {
            SmcConfluence::Neutral
        }
    }

    /// Get optimal entry zone based on SMC analysis
    pub fn get_optimal_entry(&self) -> Option<SmcEntryZone> {
        match self.has_confluence() {
            SmcConfluence::Bullish => {
                // Find best discount zone (order block or FVG)
                let best_ob = self.order_blocks.iter()
                    .filter(|ob| matches!(ob.block_type, OrderBlockType::Bullish))
                    .filter(|ob| !ob.mitigated)
                    .min_by(|a, b| a.high.partial_cmp(&b.high).unwrap_or(std::cmp::Ordering::Equal));

                let best_fvg = self.fvgs.iter()
                    .filter(|fvg| fvg.fvg_type == FvgType::Bullish && !fvg.filled)
                    .min_by(|a, b| a.high.partial_cmp(&b.high).unwrap_or(std::cmp::Ordering::Equal));

                if let Some(ob) = best_ob {
                    Some(SmcEntryZone::new(ob.low, ob.consequent_encroachment, ob.high))
                } else if let Some(fvg) = best_fvg {
                    Some(SmcEntryZone::new(fvg.low, fvg.midpoint, fvg.high))
                } else {
                    None
                }
            }
            SmcConfluence::Bearish => {
                // Find best premium zone
                let best_ob = self.order_blocks.iter()
                    .filter(|ob| matches!(ob.block_type, OrderBlockType::Bearish))
                    .filter(|ob| !ob.mitigated)
                    .max_by(|a, b| a.low.partial_cmp(&b.low).unwrap_or(std::cmp::Ordering::Equal));

                let best_fvg = self.fvgs.iter()
                    .filter(|fvg| fvg.fvg_type == FvgType::Bearish && !fvg.filled)
                    .max_by(|a, b| a.low.partial_cmp(&b.low).unwrap_or(std::cmp::Ordering::Equal));

                if let Some(ob) = best_ob {
                    Some(SmcEntryZone::new(ob.low, ob.consequent_encroachment, ob.high))
                } else if let Some(fvg) = best_fvg {
                    Some(SmcEntryZone::new(fvg.low, fvg.midpoint, fvg.high))
                } else {
                    None
                }
            }
            SmcConfluence::Neutral => None,
        }
    }
}

/// SMC confluence signal
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SmcConfluence {
    Bullish,
    Bearish,
    Neutral,
}

/// Optimal entry zone based on SMC
#[derive(Debug, Clone, Copy)]
pub struct SmcEntryZone {
    pub low: f64,
    pub midpoint: f64,
    pub high: f64,
}

impl SmcEntryZone {
    pub fn new(low: f64, midpoint: f64, high: f64) -> Self {
        Self { low, midpoint, high }
    }

    pub fn contains_price(&self, price: f64) -> bool {
        price >= self.low && price <= self.high
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_structure_analyzer_basic() {
        let analyzer = StructureAnalyzer::new(2);
        
        analyzer.update_swing_high(100.0, 1000, 3);
        analyzer.update_swing_low(90.0, 2000, 3);
        
        assert_eq!(analyzer.get_last_high(), Some(100.0));
        assert_eq!(analyzer.get_last_low(), Some(90.0));
    }

    #[test]
    fn test_bos_detection() {
        let analyzer = StructureAnalyzer::new(2);
        analyzer.set_structure(MarketStructure::Ranging);
        
        let swing_points = vec![
            SwingPoint::new(100.0, 1000, true, 3),
            SwingPoint::new(90.0, 2000, false, 3),
        ];
        
        // Price breaking above previous high
        let bos = analyzer.check_bos(105.0, 3000, &swing_points);
        
        assert!(bos.is_some());
        let bos = bos.unwrap();
        assert_eq!(bos.direction, MarketStructure::Bullish);
        assert_eq!(bos.previous_level, 100.0);
    }

    #[test]
    fn test_smc_confluence() {
        let bullish_ob = OrderBlock::new(
            OrderBlockType::Bullish,
            100.0, 95.0, 98.0, 99.0, 1000, 0.8,
        );
        
        let bullish_fvg = FairValueGap::new(FvgType::Bullish, 98.0, 96.0, 2000);
        
        let state = SmcState::new(
            MarketStructure::Bullish,
            vec![bullish_ob],
            vec![bullish_fvg],
            3000,
        );
        
        assert_eq!(state.has_confluence(), SmcConfluence::Bullish);
    }
}
