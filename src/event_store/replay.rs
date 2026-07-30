//! Replay Engine Module
//! Deterministic replay engine to reconstruct exact historical order book states.
//! Enables microsecond-accurate walk-forward testing and post-mortem bug hunting.

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::collections::VecDeque;
use std::time::Duration;

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

/// Event types for replay
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ReplayEventType {
    /// Order book snapshot
    Snapshot,
    /// New order level
    LevelAdd,
    /// Order level update
    LevelUpdate,
    /// Order level delete
    LevelDelete,
    /// Trade execution
    Trade,
    /// Sequence gap detected
    SequenceGap,
}

/// Replay event record
#[derive(Debug, Clone, Copy)]
pub struct ReplayEvent {
    pub event_type: ReplayEventType,
    pub timestamp_ns: u64,
    pub sequence: u64,
    pub price: i64,
    pub quantity: u64,
    pub side: Side,
    pub order_id: u64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Side {
    Bid,
    Ask,
}

/// Order book level for reconstruction
#[derive(Debug, Clone, Copy)]
pub struct BookLevel {
    pub price: i64,
    pub quantity: u64,
    pub order_count: u32,
}

/// Reconstructed order book state
#[derive(Debug, Clone)]
pub struct ReconstructedBook {
    pub bids: Vec<BookLevel>,
    pub asks: Vec<BookLevel>,
    pub last_update_ns: u64,
    pub sequence: u64,
}

impl ReconstructedBook {
    pub fn new() -> Self {
        Self {
            bids: Vec::with_capacity(100),
            asks: Vec::with_capacity(100),
            last_update_ns: 0,
            sequence: 0,
        }
    }

    /// Apply an event to reconstruct book state
    pub fn apply_event(&mut self, event: &ReplayEvent) {
        self.last_update_ns = event.timestamp_ns;
        self.sequence = event.sequence;

        let levels = match event.side {
            Side::Bid => &mut self.bids,
            Side::Ask => &mut self.asks,
        };

        match event.event_type {
            ReplayEventType::Snapshot => {
                // Clear and rebuild from snapshot
                levels.clear();
                if event.quantity > 0 {
                    levels.push(BookLevel {
                        price: event.price,
                        quantity: event.quantity,
                        order_count: 1,
                    });
                }
            }
            ReplayEventType::LevelAdd => {
                // Add new level
                let pos = levels.iter().position(|l| {
                    if event.side == Side::Bid {
                        l.price < event.price
                    } else {
                        l.price > event.price
                    }
                });

                match pos {
                    Some(i) => levels.insert(i, BookLevel {
                        price: event.price,
                        quantity: event.quantity,
                        order_count: 1,
                    }),
                    None => levels.push(BookLevel {
                        price: event.price,
                        quantity: event.quantity,
                        order_count: 1,
                    }),
                }
            }
            ReplayEventType::LevelUpdate => {
                // Update existing level
                if let Some(level) = levels.iter_mut().find(|l| l.price == event.price) {
                    level.quantity = event.quantity;
                }
            }
            ReplayEventType::LevelDelete => {
                // Remove level
                levels.retain(|l| l.price != event.price);
            }
            ReplayEventType::Trade => {
                // Adjust quantity at price level
                if let Some(level) = levels.iter_mut().find(|l| l.price == event.price) {
                    level.quantity = level.quantity.saturating_sub(event.quantity);
                }
            }
            ReplayEventType::SequenceGap => {
                // Mark gap - may need snapshot recovery
            }
        }
    }

    /// Get best bid price
    pub fn best_bid(&self) -> Option<i64> {
        self.bids.first().map(|l| l.price)
    }

    /// Get best ask price
    pub fn best_ask(&self) -> Option<i64> {
        self.asks.first().map(|l| l.price)
    }

    /// Get mid price
    pub fn mid_price(&self) -> Option<i64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some((bid + ask) / 2),
            (Some(bid), None) => Some(bid),
            (None, Some(ask)) => Some(ask),
            (None, None) => None,
        }
    }

    /// Get spread in ticks
    pub fn spread(&self) -> Option<i64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some(ask - bid),
            _ => None,
        }
    }
}

impl Default for ReconstructedBook {
    fn default() -> Self {
        Self::new()
    }
}

/// Lock-free replay engine
pub struct ReplayEngine {
    /// Event buffer (circular via indices)
    events: Vec<CachePadded<AtomicU64>>,
    /// Head index
    head: CachePadded<AtomicU64>,
    /// Tail index
    tail: CachePadded<AtomicU64>,
    /// Capacity (power of 2)
    capacity: u64,
    /// Events processed count
    events_processed: CachePadded<AtomicU64>,
    /// Gaps detected count
    gaps_detected: CachePadded<AtomicU64>,
    /// Active flag
    is_active: CachePadded<AtomicBool>,
    /// Current reconstructed state
    current_book: ReconstructedBook,
}

impl ReplayEngine {
    /// Create new replay engine with specified buffer size
    pub fn new(buffer_size: usize) -> Self {
        let capacity = buffer_size.next_power_of_two() as u64;
        
        let mut events = Vec::with_capacity(capacity as usize);
        for _ in 0..capacity {
            events.push(CachePadded::default());
        }

        Self {
            events,
            head: CachePadded::default(),
            tail: CachePadded::default(),
            capacity,
            events_processed: CachePadded::default(),
            gaps_detected: CachePadded::default(),
            is_active: CachePadded::new(AtomicBool::new(true)),
            current_book: ReconstructedBook::new(),
        }
    }

    /// Push an event for replay
    #[inline]
    pub fn push_event(&self, event: ReplayEvent) -> bool {
        if !self.is_active.data.load(Ordering::Acquire) {
            return false;
        }

        // Check for sequence gap
        let expected_seq = self.current_book.sequence + 1;
        if event.sequence != expected_seq && self.current_book.sequence > 0 {
            self.gaps_detected.data.fetch_add(1, Ordering::AcqRel);
            
            // Push gap event first
            let gap_event = ReplayEvent {
                event_type: ReplayEventType::SequenceGap,
                timestamp_ns: event.timestamp_ns,
                sequence: expected_seq,
                price: 0,
                quantity: 0,
                side: Side::Bid,
                order_id: 0,
            };
            self.apply_event_internal(gap_event);
        }

        self.apply_event_internal(event);
        true
    }

    /// Apply event internally
    fn apply_event_internal(&self, event: ReplayEvent) {
        self.current_book.apply_event(&event);
        self.events_processed.data.fetch_add(1, Ordering::AcqRel);
    }

    /// Replay from a list of events
    pub fn replay_events(&mut self, events: &[ReplayEvent]) -> ReplayResult {
        if !self.is_active.data.load(Ordering::Acquire) {
            return ReplayResult::inactive();
        }

        let start_ns = std::time::Instant::now();
        let mut event_count = 0;
        let mut gap_count = 0;

        for &event in events {
            if event.event_type == ReplayEventType::SequenceGap {
                gap_count += 1;
            }
            self.current_book.apply_event(&event);
            event_count += 1;
        }

        let elapsed = start_ns.elapsed();

        ReplayResult {
            events_replayed: event_count,
            gaps_found: gap_count,
            elapsed_ns: elapsed.as_nanos() as u64,
            final_sequence: self.current_book.sequence,
            success: true,
        }
    }

    /// Get current reconstructed book state
    #[inline]
    pub fn get_current_book(&self) -> &ReconstructedBook {
        &self.current_book
    }

    /// Reset to initial state
    #[inline]
    pub fn reset(&mut self) {
        self.current_book = ReconstructedBook::new();
        self.head.data.store(0, Ordering::Release);
        self.tail.data.store(0, Ordering::Release);
        self.events_processed.data.store(0, Ordering::Release);
        self.gaps_detected.data.store(0, Ordering::Release);
    }

    /// Get statistics
    pub fn get_stats(&self) -> ReplayStats {
        ReplayStats {
            events_processed: self.events_processed.data.load(Ordering::Acquire),
            gaps_detected: self.gaps_detected.data.load(Ordering::Acquire),
            current_sequence: self.current_book.sequence,
            is_active: self.is_active.data.load(Ordering::Acquire),
        }
    }

    #[inline]
    pub fn set_active(&self, active: bool) {
        self.is_active.data.store(active, Ordering::Release);
    }
}

/// Replay execution result
#[derive(Debug, Clone, Copy)]
pub struct ReplayResult {
    pub events_replayed: usize,
    pub gaps_found: usize,
    pub elapsed_ns: u64,
    pub final_sequence: u64,
    pub success: bool,
}

impl ReplayResult {
    fn inactive() -> Self {
        Self {
            events_replayed: 0,
            gaps_found: 0,
            elapsed_ns: 0,
            final_sequence: 0,
            success: false,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ReplayStats {
    pub events_processed: u64,
    pub gaps_detected: u64,
    pub current_sequence: u64,
    pub is_active: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_book_reconstruction() {
        let mut book = ReconstructedBook::new();

        // Add bid levels
        let events = vec![
            ReplayEvent {
                event_type: ReplayEventType::LevelAdd,
                timestamp_ns: 1000,
                sequence: 1,
                price: 10000,
                quantity: 100,
                side: Side::Bid,
                order_id: 1,
            },
            ReplayEvent {
                event_type: ReplayEventType::LevelAdd,
                timestamp_ns: 2000,
                sequence: 2,
                price: 9999,
                quantity: 200,
                side: Side::Bid,
                order_id: 2,
            },
            ReplayEvent {
                event_type: ReplayEventType::LevelAdd,
                timestamp_ns: 3000,
                sequence: 3,
                price: 10001,
                quantity: 150,
                side: Side::Ask,
                order_id: 3,
            },
        ];

        for event in events {
            book.apply_event(&event);
        }

        assert_eq!(book.best_bid(), Some(10000));
        assert_eq!(book.best_ask(), Some(10001));
        assert_eq!(book.spread(), Some(1));
    }

    #[test]
    fn test_replay_engine_basic() {
        let mut engine = ReplayEngine::new(1024);

        let events = vec![
            ReplayEvent {
                event_type: ReplayEventType::LevelAdd,
                timestamp_ns: 1000,
                sequence: 1,
                price: 10000,
                quantity: 100,
                side: Side::Bid,
                order_id: 1,
            },
            ReplayEvent {
                event_type: ReplayEventType::LevelAdd,
                timestamp_ns: 2000,
                sequence: 2,
                price: 10001,
                quantity: 100,
                side: Side::Ask,
                order_id: 2,
            },
        ];

        let result = engine.replay_events(&events);
        assert!(result.success);
        assert_eq!(result.events_replayed, 2);

        let book = engine.get_current_book();
        assert_eq!(book.mid_price(), Some(10000));
    }
}
