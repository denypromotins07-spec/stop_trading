//! Breaker Blocks Detection
//! Identifies failed order blocks that flip polarity (support becomes resistance).
//! Uses strict fixed-point geometric validation.

use super::liquidity_pools::{LiquidityPool, PoolType, SwingPoint, SwingType};

/// Maximum number of breaker blocks to track
const MAX_BREAKER_BLOCKS: usize = 64;

/// Order block type
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u8)]
pub enum OrderBlockType {
    Bullish = 0,
    Bearish = 1,
}

/// Order block structure with precise geometric boundaries
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct OrderBlock {
    pub high: i64,            // Fixed-point high of the block
    pub low: i64,             // Fixed-point low of the block
    pub open: i64,            // Fixed-point open
    pub close: i64,           // Fixed-point close
    pub timestamp: u64,       // Unix timestamp in microseconds
    pub block_type: OrderBlockType,
    pub tested_count: u8,     // Number of times price returned to this block
    pub broken: bool,         // Whether the block has been broken
    _padding: [u8; 2],        // Cache-line alignment
}

/// Breaker block - a failed order block that flipped polarity
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BreakerBlock {
    pub price_level: i64,          // The key price level (high or low of broken block)
    pub original_type: OrderBlockType,
    pub new_polarity: Polarity,    // What it flipped to
    pub break_timestamp: u64,
    pub break_price: i64,          // Price at which the block was broken
    pub mitigation_target: i64,    // Expected target for mitigation
    pub strength: u8,              // Strength based on how decisively it broke
    _padding: [u8; 7],             // Cache-line alignment to 64 bytes
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u8)]
pub enum Polarity {
    Support = 0,
    Resistance = 1,
}

/// State machine for detecting breaker blocks
pub struct BreakerBlockDetector {
    order_blocks: [OrderBlock; MAX_BREAKER_BLOCKS],
    block_count: usize,
    breaker_blocks: [BreakerBlock; MAX_BREAKER_BLOCKS],
    breaker_count: usize,
    validation_threshold_bps: i64,
}

impl Default for BreakerBlockDetector {
    fn default() -> Self {
        Self::new(50)
    }
}

impl BreakerBlockDetector {
    /// Create a new detector with specified validation threshold (in basis points)
    pub const fn new(validation_threshold_bps: i64) -> Self {
        Self {
            order_blocks: unsafe { core::mem::zeroed() },
            block_count: 0,
            breaker_blocks: unsafe { core::mem::zeroed() },
            breaker_count: 0,
            validation_threshold_bps,
        }
    }

    /// Process a new candle and detect breaker blocks
    /// Returns newly detected breaker blocks
    pub fn process_candle(
        &mut self,
        open: i64,
        high: i64,
        low: i64,
        close: i64,
        timestamp: u64,
    ) -> Option<[BreakerBlock; 2]> {
        let mut detected = [
            BreakerBlock {
                price_level: 0,
                original_type: OrderBlockType::Bullish,
                new_polarity: Polarity::Support,
                break_timestamp: 0,
                break_price: 0,
                mitigation_target: 0,
                strength: 0,
                _padding: [0; 7],
            },
            BreakerBlock {
                price_level: 0,
                original_type: OrderBlockType::Bullish,
                new_polarity: Polarity::Support,
                break_timestamp: 0,
                break_price: 0,
                mitigation_target: 0,
                strength: 0,
                _padding: [0; 7],
            },
        ];
        let mut breaker_idx = 0;

        // Check existing order blocks for breaks
        for i in 0..self.block_count {
            let block = &mut self.order_blocks[i];
            
            if block.broken {
                continue;
            }

            match block.block_type {
                OrderBlockType::Bullish => {
                    // Bullish block should act as support
                    // If price breaks below the low, it's a failed bullish block
                    if self.is_valid_break(low, block.low, false) {
                        let breaker = self.create_breaker_block(block, low, timestamp, Polarity::Resistance);
                        block.broken = true;
                        
                        if breaker_idx < 2 {
                            detected[breaker_idx] = breaker;
                            breaker_idx += 1;
                        }
                    } else {
                        // Check if price tapped into the block (mitigation)
                        if self.price_in_block(low, high, block) {
                            block.tested_count = block.tested_count.saturating_add(1);
                        }
                    }
                }
                OrderBlockType::Bearish => {
                    // Bearish block should act as resistance
                    // If price breaks above the high, it's a failed bearish block
                    if self.is_valid_break(high, block.high, true) {
                        let breaker = self.create_breaker_block(block, high, timestamp, Polarity::Support);
                        block.broken = true;
                        
                        if breaker_idx < 2 {
                            detected[breaker_idx] = breaker;
                            breaker_idx += 1;
                        }
                    } else {
                        // Check if price tapped into the block (mitigation)
                        if self.price_in_block(low, high, block) {
                            block.tested_count = block.tested_count.saturating_add(1);
                        }
                    }
                }
            }
        }

        // Add new order block based on current candle
        self.add_order_block(open, high, low, close, timestamp);

        if breaker_idx > 0 {
            Some(detected)
        } else {
            None
        }
    }

    /// Validate that a break is genuine (not just a wick)
    #[inline]
    fn is_valid_break(&self, break_price: i64, block_level: i64, is_upside: bool) -> bool {
        let threshold = (block_level.abs() * self.validation_threshold_bps) / 10000;
        let threshold = threshold.max(1);

        if is_upside {
            break_price > block_level + threshold
        } else {
            break_price < block_level - threshold
        }
    }

    /// Check if current price range intersects with the order block
    #[inline]
    fn price_in_block(&self, low: i64, high: i64, block: &OrderBlock) -> bool {
        low <= block.high && high >= block.low
    }

    /// Create a breaker block from a broken order block
    fn create_breaker_block(
        &self,
        block: &OrderBlock,
        break_price: i64,
        timestamp: u64,
        new_polarity: Polarity,
    ) -> BreakerBlock {
        let price_level = match block.block_type {
            OrderBlockType::Bullish => block.low,
            OrderBlockType::Bearish => block.high,
        };

        // Calculate mitigation target (50% retracement of the break move)
        let break_distance = (break_price - price_level).abs();
        let mitigation_target = match new_polarity {
            Polarity::Support => price_level + (break_distance / 2),
            Polarity::Resistance => price_level - (break_distance / 2),
        };

        // Strength based on how many times it was tested before breaking
        let strength = (block.tested_count + 1).min(10);

        BreakerBlock {
            price_level,
            original_type: block.block_type,
            new_polarity,
            break_timestamp: timestamp,
            break_price,
            mitigation_target,
            strength,
            _padding: [0; 7],
        }
    }

    /// Add a new order block based on candle characteristics
    fn add_order_block(
        &mut self,
        open: i64,
        high: i64,
        low: i64,
        close: i64,
        timestamp: u64,
    ) {
        if self.block_count >= MAX_BREAKER_BLOCKS {
            // Shift left to make room
            unsafe {
                core::ptr::copy(
                    self.order_blocks.as_ptr().add(1),
                    self.order_blocks.as_mut_ptr(),
                    MAX_BREAKER_BLOCKS - 1,
                );
            }
            self.block_count = MAX_BREAKER_BLOCKS - 1;
        }

        // Determine block type based on candle direction
        let block_type = if close > open {
            OrderBlockType::Bullish
        } else {
            OrderBlockType::Bearish
        };

        let block = OrderBlock {
            high,
            low,
            open,
            close,
            timestamp,
            block_type,
            tested_count: 0,
            broken: false,
            _padding: [0; 2],
        };

        self.order_blocks[self.block_count] = block;
        self.block_count += 1;
    }

    /// Get all active (unbroken) order blocks
    pub fn get_active_order_blocks(&self) -> &[OrderBlock] {
        unsafe {
            core::slice::from_raw_parts(
                self.order_blocks.as_ptr(),
                self.block_count,
            )
        }
    }

    /// Get all detected breaker blocks
    pub fn get_breaker_blocks(&self) -> &[BreakerBlock] {
        unsafe {
            core::slice::from_raw_parts(
                self.breaker_blocks.as_ptr(),
                self.breaker_count,
            )
        }
    }

    /// Check if a price level is near a breaker block (potential reversal zone)
    pub fn is_near_breaker(&self, price: i64, tolerance_bps: i64) -> Option<&BreakerBlock> {
        for i in 0..self.breaker_count {
            let breaker = unsafe { self.breaker_blocks.get_unchecked(i) };
            let diff = (price - breaker.price_level).abs();
            let threshold = (breaker.price_level.abs() * tolerance_bps) / 10000;
            
            if diff <= threshold.max(1) {
                return Some(breaker);
            }
        }
        None
    }

    /// Record a breaker block for external use
    pub fn record_breaker_block(&mut self, breaker: BreakerBlock) {
        if self.breaker_count < MAX_BREAKER_BLOCKS {
            self.breaker_blocks[self.breaker_count] = breaker;
            self.breaker_count += 1;
        }
    }

    /// Geometric validation: check if breaker block aligns with liquidity pool
    pub fn validate_with_liquidity(&self, breaker: &BreakerBlock, pool: &LiquidityPool) -> bool {
        let price_diff = (breaker.price_level - pool.price).abs();
        let threshold = (breaker.price_level.abs() * 50) / 10000; // 0.5% tolerance
        price_diff <= threshold.max(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_breaker_detection() {
        let mut detector = BreakerBlockDetector::new(10);
        
        // Create a bullish order block
        let open = 100_0000_0000i64;
        let high = 101_0000_0000i64;
        let low = 99_0000_0000i64;
        let close = 100_5000_0000i64;
        
        detector.process_candle(open, high, low, close, 1000);
        
        assert_eq!(detector.block_count, 1);
        assert_eq!(detector.order_blocks[0].block_type, OrderBlockType::Bullish);
    }

    #[test]
    fn test_valid_break() {
        let detector = BreakerBlockDetector::new(10);
        let block_level = 100_0000_0000i64;
        
        // Break below by more than threshold
        let break_price = 99_8000_0000i64; // 0.2% below
        assert!(detector.is_valid_break(break_price, block_level, false));
        
        // Small dip that shouldn't count
        let break_price_small = 99_9500_0000i64; // 0.05% below
        assert!(!detector.is_valid_break(break_price_small, block_level, false));
    }
}
