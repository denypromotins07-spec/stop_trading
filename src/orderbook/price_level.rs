//! Order Book Price Level
//! 
//! Defines a compact `PriceLevel` struct holding price, quantity, and order count
//! using fixed-point arithmetic (i64) to avoid floating-point inaccuracies.

use crate::market_data::{Price, Quantity};

/// A single price level in the order book
/// 
/// Memory layout is optimized for cache efficiency with explicit padding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C, align(32))]
pub struct PriceLevel {
    /// Price in fixed-point format (price * 10^8)
    pub price: Price,
    /// Total quantity at this price level in fixed-point format
    pub quantity: Quantity,
    /// Number of orders at this price level
    pub order_count: u32,
    /// Padding for cache alignment
    _padding: u32,
    /// Cached hash for fast lookups (optional optimization)
    cached_hash: u64,
}

impl PriceLevel {
    /// Create a new price level
    #[inline]
    pub const fn new(price: Price, quantity: Quantity, order_count: u32) -> Self {
        PriceLevel {
            price,
            quantity,
            order_count,
            _padding: 0,
            cached_hash: 0,
        }
    }

    /// Create an empty price level at the given price
    #[inline]
    pub const fn empty(price: Price) -> Self {
        PriceLevel::new(price, Quantity::new(0), 0)
    }

    /// Check if this level has zero quantity
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.quantity.raw() == 0
    }

    /// Check if this level has non-zero quantity
    #[inline]
    pub const fn has_volume(&self) -> bool {
        self.quantity.raw() > 0
    }

    /// Update the quantity at this level
    #[inline]
    pub fn update_quantity(&mut self, new_quantity: Quantity, new_order_count: u32) {
        self.quantity = new_quantity;
        self.order_count = new_order_count;
        self.cached_hash = 0; // Invalidate cached hash
    }

    /// Add quantity to this level
    #[inline]
    pub fn add_quantity(&mut self, delta: Quantity) -> Result<(), &'static str> {
        let current = self.quantity.raw();
        let delta_raw = delta.raw();
        
        // Check for overflow
        let new_qty = current.checked_add(delta_raw)
            .ok_or("Quantity overflow")?;
        
        self.quantity = Quantity::new(new_qty);
        self.order_count += 1;
        Ok(())
    }

    /// Remove quantity from this level
    #[inline]
    pub fn remove_quantity(&mut self, delta: Quantity) -> Result<(), &'static str> {
        let current = self.quantity.raw();
        let delta_raw = delta.raw();
        
        if delta_raw > current {
            return Err("Cannot remove more quantity than exists");
        }
        
        self.quantity = Quantity::new(current - delta_raw);
        if self.is_empty() {
            self.order_count = 0;
        } else {
            self.order_count = self.order_count.saturating_sub(1);
        }
        Ok(())
    }

    /// Set the level to completely empty
    #[inline]
    pub fn clear(&mut self) {
        self.quantity = Quantity::new(0);
        self.order_count = 0;
        self.cached_hash = 0;
    }

    /// Get or compute the hash for this level
    #[inline]
    pub fn hash(&self) -> u64 {
        if self.cached_hash == 0 && !self.is_empty() {
            // Simple hash combining price and quantity
            use std::hash::{Hash, Hasher};
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            self.price.raw().hash(&mut hasher);
            self.quantity.raw().hash(&mut hasher);
            self.cached_hash = hasher.finish();
        }
        self.cached_hash
    }

    /// Compare prices (for sorting)
    #[inline]
    pub fn price_cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.price.cmp(&other.price)
    }
}

impl Default for PriceLevel {
    #[inline]
    fn default() -> Self {
        PriceLevel::empty(Price::new(0))
    }
}

/// Side-specific price level operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LevelSide {
    Bid,
    Ask,
}

impl LevelSide {
    #[inline]
    pub const fn is_bid(self) -> bool {
        matches!(self, LevelSide::Bid)
    }

    #[inline]
    pub const fn is_ask(self) -> bool {
        matches!(self, LevelSide::Ask)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_price_level_creation() {
        let level = PriceLevel::new(
            Price::from_f64(50000.0),
            Quantity::from_f64(1.5),
            3,
        );
        
        assert!(!level.is_empty());
        assert_eq!(level.order_count, 3);
        assert_eq!(level.price.to_f64(), 50000.0);
        assert_eq!(level.quantity.to_f64(), 1.5);
    }

    #[test]
    fn test_price_level_update() {
        let mut level = PriceLevel::new(
            Price::from_f64(50000.0),
            Quantity::from_f64(1.0),
            1,
        );
        
        level.add_quantity(Quantity::from_f64(0.5)).unwrap();
        assert_eq!(level.quantity.to_f64(), 1.5);
        assert_eq!(level.order_count, 2);
        
        level.remove_quantity(Quantity::from_f64(0.5)).unwrap();
        assert_eq!(level.quantity.to_f64(), 1.0);
    }

    #[test]
    fn test_price_level_clear() {
        let mut level = PriceLevel::new(
            Price::from_f64(50000.0),
            Quantity::from_f64(1.0),
            2,
        );
        
        level.clear();
        assert!(level.is_empty());
        assert_eq!(level.order_count, 0);
    }

    #[test]
    fn test_overflow_protection() {
        let mut level = PriceLevel::new(
            Price::from_f64(50000.0),
            Quantity::new(i64::MAX),
            1,
        );
        
        let result = level.add_quantity(Quantity::new(1));
        assert!(result.is_err());
    }
}
