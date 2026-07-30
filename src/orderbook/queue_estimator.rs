//! Queue Depletion Rate Calculator
//! 
//! Builds a queue depletion rate calculator using L2/L3 deltas and trade tick
//! aggressor data to predict exact microsecond an order will reach top of book.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use crate::orderbook::l3_tracker::{L3Tracker, Side, QueuePosition};

/// Trade tick with aggressor information
#[derive(Debug, Clone)]
pub struct TradeTick {
    pub symbol: String,
    pub price: i64,
    pub size: u64,
    pub aggressor_side: AggressorSide,
    pub timestamp_ns: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggressorSide {
    Buy,   // Taker bought from maker (hit ask)
    Sell,  // Taker sold to maker (hit bid)
}

/// Queue depletion statistics for a price level
#[derive(Debug, Clone)]
pub struct QueueDepletionStats {
    pub symbol: String,
    pub price: i64,
    pub side: Side,
    pub initial_size: u64,
    pub current_size: u64,
    pub depletion_rate_per_sec: f64,
    pub estimated_time_to_top_us: u64,
    pub confidence: f64,
    pub last_trade_ts: u64,
    pub trade_count: usize,
}

/// Circular buffer for recent trades at a price level
struct TradeBuffer {
    trades: Vec<TradeTick>,
    head: usize,
    tail: usize,
    count: usize,
    capacity: usize,
}

impl TradeBuffer {
    fn new(capacity: usize) -> Self {
        Self {
            trades: Vec::with_capacity(capacity),
            head: 0,
            tail: 0,
            count: 0,
            capacity,
        }
    }

    fn push(&mut self, trade: TradeTick) {
        if self.count < self.capacity {
            if self.trades.len() < self.capacity {
                self.trades.push(trade);
            } else {
                self.trades[self.tail] = trade;
            }
            self.count += 1;
        } else {
            // Overwrite oldest
            self.trades[self.tail] = trade;
            self.head = (self.head + 1) % self.capacity;
        }
        self.tail = (self.tail + 1) % self.capacity;
    }

    fn iter(&self) -> impl Iterator<Item = &TradeTick> {
        if self.count == 0 {
            return TradeIter { buffer: self, index: 0, end: 0 };
        }
        
        let start = if self.count < self.capacity { 0 } else { self.head };
        let end = if self.count < self.capacity { self.count } else { self.capacity };
        
        TradeIter { buffer: self, index: start, end }
    }

    fn time_range_ns(&self) -> Option<(u64, u64)> {
        if self.count == 0 {
            return None;
        }
        
        let mut min_ts = u64::MAX;
        let mut max_ts = 0u64;
        
        for trade in self.iter() {
            min_ts = min_ts.min(trade.timestamp_ns);
            max_ts = max_ts.max(trade.timestamp_ns);
        }
        
        Some((min_ts, max_ts))
    }

    fn total_volume(&self, side: AggressorSide) -> u64 {
        self.iter()
            .filter(|t| t.aggressor_side == side)
            .map(|t| t.size)
            .sum()
    }
}

struct TradeIter<'a> {
    buffer: &'a TradeBuffer,
    index: usize,
    end: usize,
}

impl<'a> Iterator for TradeIter<'a> {
    type Item = &'a TradeTick;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.end {
            return None;
        }
        let idx = self.index % self.buffer.capacity;
        self.index += 1;
        self.buffer.trades.get(idx)
    }
}

/// Queue depletion rate calculator
pub struct QueueEstimator {
    /// Map of (symbol, price, side) -> TradeBuffer
    trade_buffers: dashmap::DashMap<(String, i64, Side), TradeBuffer>,
    /// Reference to L3 tracker for queue position data
    l3_tracker: Arc<L3Tracker>,
    /// Buffer capacity per level (trades)
    buffer_capacity: usize,
    /// Total buffers created
    buffer_count: AtomicUsize,
    /// Memory usage tracker
    memory_bytes: AtomicU64,
    /// Max memory limit
    max_memory_bytes: u64,
}

impl QueueEstimator {
    pub fn new(l3_tracker: Arc<L3Tracker>, max_memory_mb: u64, buffer_capacity: usize) -> Self {
        Self {
            trade_buffers: dashmap::DashMap::new(),
            l3_tracker,
            buffer_capacity,
            buffer_count: AtomicUsize::new(0),
            memory_bytes: AtomicU64::new(0),
            max_memory_bytes: max_memory_mb * 1024 * 1024,
        }
    }

    /// Add a trade tick to the estimator
    pub fn add_trade(&self, trade: TradeTick) -> Result<(), &'static str> {
        // Check memory limit
        let current_mem = self.memory_bytes.load(Ordering::Relaxed);
        let estimated_size = (std::mem::size_of::<TradeTick>() * self.buffer_capacity) as u64 + 256;
        
        if current_mem + estimated_size > self.max_memory_bytes {
            // Purge old buffers if over limit
            self.purge_oldest_buffers();
            
            if self.memory_bytes.load(Ordering::Relaxed) + estimated_size > self.max_memory_bytes {
                return Err("Queue estimator memory limit exceeded");
            }
        }

        let key = (trade.symbol.clone(), trade.price, match trade.aggressor_side {
            AggressorSide::Buy => Side::Ask,
            AggressorSide::Sell => Side::Bid,
        });

        if let Some(mut entry) = self.trade_buffers.get_mut(&key) {
            entry.value().push(trade);
        } else {
            let mut buffer = TradeBuffer::new(self.buffer_capacity);
            buffer.push(trade);
            self.trade_buffers.insert(key, buffer);
            self.buffer_count.fetch_add(1, Ordering::Relaxed);
            self.memory_bytes.fetch_add(estimated_size, Ordering::Relaxed);
        }

        Ok(())
    }

    /// Calculate queue depletion stats for a specific order
    pub fn estimate_time_to_top(&self, order_id: u64) -> Option<QueueDepletionStats> {
        let position = self.l3_tracker.get_position(order_id)?;
        
        let key = (position.symbol.clone(), position.price, position.side);
        let buffer = self.trade_buffers.get(&key)?;
        
        // Calculate depletion rate based on trades hitting this level
        let relevant_side = match position.side {
            Side::Bid => AggressorSide::Sell, // Sells hit bids
            Side::Ask => AggressorSide::Buy,  // Buys hit asks
        };

        let (min_ts, max_ts) = buffer.time_range_ns()?;
        let time_range_sec = if max_ts > min_ts {
            (max_ts - min_ts) as f64 / 1e9
        } else {
            0.001 // Avoid division by zero
        };

        let volume_depleted = buffer.total_volume(relevant_side);
        let depletion_rate = if time_range_sec > 0.0 {
            volume_depleted as f64 / time_range_sec
        } else {
            0.0
        };

        // Estimate time to deplete ahead size
        let ahead_size = position.estimated_ahead_size;
        let estimated_time_sec = if depletion_rate > 0.0 {
            ahead_size as f64 / depletion_rate
        } else {
            f64::MAX
        };

        let estimated_time_us = (estimated_time_sec * 1e6) as u64;

        // Calculate confidence based on trade count and time range
        let trade_count = buffer.iter().count();
        let confidence = if trade_count >= 10 && time_range_sec >= 0.1 {
            0.9
        } else if trade_count >= 5 && time_range_sec >= 0.05 {
            0.7
        } else if trade_count >= 1 {
            0.4
        } else {
            0.1
        };

        Some(QueueDepletionStats {
            symbol: position.symbol,
            price: position.price,
            side: position.side,
            initial_size: position.original_size,
            current_size: position.remaining_size,
            depletion_rate_per_sec: depletion_rate,
            estimated_time_to_top_us: estimated_time_us.min(u64::MAX),
            confidence,
            last_trade_ts: max_ts,
            trade_count,
        })
    }

    /// Get all queue depletion stats for a symbol
    pub fn get_symbol_stats(&self, symbol: &str) -> Vec<QueueDepletionStats> {
        let mut results = Vec::new();

        for entry in self.trade_buffers.iter() {
            let (sym, price, side) = entry.key();
            if sym != symbol {
                continue;
            }

            let buffer = entry.value();
            let (min_ts, max_ts) = match buffer.time_range_ns() {
                Some(ts) => ts,
                None => continue,
            };

            let time_range_sec = if max_ts > min_ts {
                (max_ts - min_ts) as f64 / 1e9
            } else {
                continue;
            };

            let relevant_side = match side {
                Side::Bid => AggressorSide::Sell,
                Side::Ask => AggressorSide::Buy,
            };

            let volume_depleted = buffer.total_volume(relevant_side);
            let depletion_rate = if time_range_sec > 0.0 {
                volume_depleted as f64 / time_range_sec
            } else {
                0.0
            };

            let trade_count = buffer.iter().count();
            let confidence = if trade_count >= 10 { 0.9 } else { 0.5 };

            results.push(QueueDepletionStats {
                symbol: symbol.to_string(),
                price: *price,
                side: *side,
                initial_size: 0,
                current_size: 0,
                depletion_rate_per_sec: depletion_rate,
                estimated_time_to_top_us: 0,
                confidence,
                last_trade_ts: max_ts,
                trade_count,
            });
        }

        results
    }

    /// Purge oldest buffers when memory is tight
    fn purge_oldest_buffers(&self) {
        // Simple strategy: remove half the buffers
        let keys_to_remove: Vec<_> = self.trade_buffers
            .iter()
            .take(self.buffer_count.load(Ordering::Relaxed) / 2)
            .map(|entry| entry.key().clone())
            .collect();

        let mut freed = 0u64;
        for key in keys_to_remove {
            if let Some((_, _)) = self.trade_buffers.remove(&key) {
                freed += (std::mem::size_of::<TradeTick>() * self.buffer_capacity) as u64 + 256;
            }
        }

        self.memory_bytes.fetch_sub(freed, Ordering::Relaxed);
        self.buffer_count.fetch_sub(keys_to_remove.len(), Ordering::Relaxed);
    }

    /// Predict optimal amendment time for an order
    pub fn predict_amendment_time(&self, order_id: u64, lead_time_us: u64) -> Option<u64> {
        let stats = self.estimate_time_to_top(order_id)?;
        
        if stats.estimated_time_to_top_us == u64::MAX {
            return None;
        }

        // Return timestamp when we should amend (lead_time before reaching top)
        let now_ns = Instant::now().duration_since(Instant::now() - Duration::from_secs(1)).as_nanos() as u64;
        let trigger_time_ns = now_ns + ((stats.estimated_time_to_top_us.saturating_sub(lead_time_us)) * 1000);
        
        Some(trigger_time_ns)
    }

    /// Get memory usage
    pub fn memory_usage(&self) -> u64 {
        self.memory_bytes.load(Ordering::Relaxed)
    }

    /// Clear all data
    pub fn clear(&self) {
        self.trade_buffers.clear();
        self.buffer_count.store(0, Ordering::Relaxed);
        self.memory_bytes.store(0, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_queue_estimator_basic() {
        let l3 = Arc::new(L3Tracker::new(100));
        let estimator = QueueEstimator::new(l3.clone(), 50, 100);

        // Add a trade
        let trade = TradeTick {
            symbol: "BTCUSD".to_string(),
            price: 50000,
            size: 10,
            aggressor_side: AggressorSide::Sell,
            timestamp_ns: 1000000000,
        };

        assert!(estimator.add_trade(trade).is_ok());

        // Add order to L3 tracker
        let position = QueuePosition {
            order_id: 1,
            symbol: "BTCUSD".to_string(),
            side: Side::Bid,
            price: 50000,
            original_size: 100,
            remaining_size: 100,
            queue_position: 0,
            estimated_ahead_size: 50,
            timestamp_ns: 0,
            last_update_ns: 0,
        };

        assert!(l3.insert_order(position).is_ok());

        // Estimate time to top
        let stats = estimator.estimate_time_to_top(1);
        assert!(stats.is_some());
    }
}
