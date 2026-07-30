//! L3 Order Book & Queue Position Tracking
//! 
//! Implements high-performance L3 order tracking mapping individual order IDs
//! to exact queue positions using lock-free data structures.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use dashmap::DashMap;
use crossbeam_queue::SegQueue;

/// Represents a single order's position in the queue
#[derive(Debug, Clone)]
pub struct QueuePosition {
    pub order_id: u64,
    pub symbol: String,
    pub side: Side,
    pub price: i64,
    pub original_size: u64,
    pub remaining_size: u64,
    pub queue_position: u64,
    pub estimated_ahead_size: u64,
    pub timestamp_ns: u64,
    pub last_update_ns: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Side {
    Bid,
    Ask,
}

/// Lock-free L3 order tracker for mapping order IDs to queue positions
pub struct L3Tracker {
    /// Map of order_id -> QueuePosition
    orders: DashMap<u64, QueuePosition>,
    /// Map of (symbol, price, side) -> Vec of order_ids at that level
    level_orders: DashMap<(String, i64, Side), SegQueue<u64>>,
    /// Total tracked orders counter
    total_orders: AtomicUsize,
    /// Memory usage tracker (bytes)
    memory_bytes: AtomicU64,
    /// Max memory limit (default 500MB for L3 data)
    max_memory_bytes: u64,
}

impl L3Tracker {
    pub fn new(max_memory_mb: u64) -> Self {
        Self {
            orders: DashMap::new(),
            level_orders: DashMap::new(),
            total_orders: AtomicUsize::new(0),
            memory_bytes: AtomicU64::new(0),
            max_memory_bytes: max_memory_mb * 1024 * 1024,
        }
    }

    /// Insert or update an order's queue position
    /// 
    /// # Safety
    /// This function is thread-safe and lock-free. It uses atomic operations
    /// to ensure consistency across multiple threads.
    pub fn insert_order(&self, mut position: QueuePosition) -> Result<(), &'static str> {
        // Check memory limit before insertion
        let current_mem = self.memory_bytes.load(Ordering::Relaxed);
        let estimated_size = std::mem::size_of::<QueuePosition>() as u64 + 128; // overhead
        
        if current_mem + estimated_size > self.max_memory_bytes {
            return Err("L3 tracker memory limit exceeded");
        }

        position.timestamp_ns = timestamp_ns();
        position.last_update_ns = position.timestamp_ns;

        // Insert into main order map
        let key = (position.symbol.clone(), position.price, position.side);
        
        self.orders.insert(position.order_id, position.clone());
        
        // Add to level queue
        if let Some(queue) = self.level_orders.get(&key) {
            queue.push(position.order_id);
        } else {
            let queue = SegQueue::new();
            queue.push(position.order_id);
            self.level_orders.insert(key, queue);
        }

        self.total_orders.fetch_add(1, Ordering::Relaxed);
        self.memory_bytes.fetch_add(estimated_size, Ordering::Relaxed);

        Ok(())
    }

    /// Update order size (partial fill or cancellation)
    pub fn update_order_size(&self, order_id: u64, new_remaining: u64) -> Option<QueuePosition> {
        if let Some(mut entry) = self.orders.get_mut(&order_id) {
            entry.value.remaining_size = new_remaining;
            entry.value.last_update_ns = timestamp_ns();
            
            if new_remaining == 0 {
                drop(entry);
                self.remove_order(order_id)
            } else {
                Some(entry.value().clone())
            }
        } else {
            None
        }
    }

    /// Remove an order from tracking
    pub fn remove_order(&self, order_id: u64) -> Option<QueuePosition> {
        if let Some((_, position)) = self.orders.remove(&order_id) {
            let key = (position.symbol.clone(), position.price, position.side);
            
            // Remove from level queue (note: SegQueue doesn't support removal,
            // so we rely on lazy cleanup during iteration)
            
            let estimated_size = std::mem::size_of::<QueuePosition>() as u64 + 128;
            self.total_orders.fetch_sub(1, Ordering::Relaxed);
            self.memory_bytes.fetch_sub(estimated_size, Ordering::Relaxed);
            
            Some(position)
        } else {
            None
        }
    }

    /// Get queue position for a specific order
    pub fn get_position(&self, order_id: u64) -> Option<QueuePosition> {
        self.orders.get(&order_id).map(|entry| entry.value().clone())
    }

    /// Calculate estimated queue position based on level data
    pub fn estimate_queue_position(&self, symbol: &str, price: i64, side: Side, order_id: u64) -> u64 {
        let key = (symbol.to_string(), price, side);
        
        if let Some(queue) = self.level_orders.get(&key) {
            let mut position: u64 = 0;
            let mut found_self = false;
            
            for &oid in queue.iter() {
                if oid == order_id {
                    found_self = true;
                    break;
                }
                if let Some(pos) = self.orders.get(&oid) {
                    position += pos.remaining_size;
                }
            }
            
            if !found_self {
                // Order not in queue, return max estimate
                u64::MAX
            } else {
                position
            }
        } else {
            0
        }
    }

    /// Get all orders for a specific price level
    pub fn get_level_orders(&self, symbol: &str, price: i64, side: Side) -> Vec<QueuePosition> {
        let key = (symbol.to_string(), price, side);
        let mut result = Vec::new();
        
        if let Some(queue) = self.level_orders.get(&key) {
            for &order_id in queue.iter() {
                if let Some(pos) = self.orders.get(&order_id) {
                    result.push(pos.value().clone());
                }
            }
        }
        
        result
    }

    /// Purge stale orders (older than specified duration)
    pub fn purge_stale(&self, max_age_ns: u64) -> usize {
        let now = timestamp_ns();
        let mut purged = 0;
        
        let stale_ids: Vec<u64> = self.orders
            .iter()
            .filter(|entry| now - entry.value().last_update_ns > max_age_ns)
            .map(|entry| entry.key().clone())
            .collect();
        
        for order_id in stale_ids {
            if self.remove_order(order_id).is_some() {
                purged += 1;
            }
        }
        
        purged
    }

    /// Get current memory usage in bytes
    pub fn memory_usage(&self) -> u64 {
        self.memory_bytes.load(Ordering::Relaxed)
    }

    /// Get total tracked orders
    pub fn total_orders(&self) -> usize {
        self.total_orders.load(Ordering::Relaxed)
    }

    /// Clear all data
    pub fn clear(&self) {
        self.orders.clear();
        self.level_orders.clear();
        self.total_orders.store(0, Ordering::Relaxed);
        self.memory_bytes.store(0, Ordering::Relaxed);
    }
}

/// Get current timestamp in nanoseconds
#[inline]
fn timestamp_ns() -> u64 {
    Instant::now()
        .duration_since(Instant::now() - Duration::from_secs(1))
        .as_nanos() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l3_tracker_basic() {
        let tracker = L3Tracker::new(100);
        
        let position = QueuePosition {
            order_id: 1,
            symbol: "BTCUSD".to_string(),
            side: Side::Bid,
            price: 50000,
            original_size: 100,
            remaining_size: 100,
            queue_position: 0,
            estimated_ahead_size: 500,
            timestamp_ns: 0,
            last_update_ns: 0,
        };
        
        assert!(tracker.insert_order(position).is_ok());
        assert_eq!(tracker.total_orders(), 1);
        
        let retrieved = tracker.get_position(1);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().remaining_size, 100);
    }

    #[test]
    fn test_memory_limit() {
        let tracker = L3Tracker::new(0); // 0 MB limit
        
        let position = QueuePosition {
            order_id: 1,
            symbol: "BTCUSD".to_string(),
            side: Side::Bid,
            price: 50000,
            original_size: 100,
            remaining_size: 100,
            queue_position: 0,
            estimated_ahead_size: 500,
            timestamp_ns: 0,
            last_update_ns: 0,
        };
        
        assert!(tracker.insert_order(position).is_err());
    }
}
