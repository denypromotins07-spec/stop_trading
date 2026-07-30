//! Event Store Module Root
//! Trigger automated REST snapshots on WS sequence gaps.

pub mod wal;
pub mod replay;

pub use wal::{
    WALWriter,
    WALReader,
    WALEntry,
    WALEntryHeader,
    WALEntryType,
};

pub use replay::{
    ReplayEngine,
    ReplayEvent,
    ReplayEventType,
    ReconstructedBook,
    BookLevel,
    ReplayResult,
    ReplayStats,
};

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
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

/// Sequence gap detection result
#[derive(Debug, Clone, Copy)]
pub struct SequenceGapInfo {
    pub expected_sequence: u64,
    pub received_sequence: u64,
    pub gap_size: u64,
    pub timestamp_ns: u64,
    pub snapshot_requested: bool,
}

/// Event store combining WAL and replay with gap handling
pub struct EventStore {
    /// WAL writer
    wal_writer: Option<WALWriter>,
    /// Replay engine
    replay_engine: ReplayEngine,
    /// Expected next sequence
    expected_sequence: CachePadded<AtomicU64>,
    /// Total gaps detected
    total_gaps: CachePadded<AtomicU64>,
    /// Snapshots triggered
    snapshots_triggered: CachePadded<AtomicU64>,
    /// Active flag
    is_active: CachePadded<AtomicBool>,
    /// Auto-snapshot on gap flag
    auto_snapshot_on_gap: bool,
}

impl EventStore {
    /// Create new event store
    pub fn new<P: AsRef<std::path::Path>>(
        wal_path: Option<P>,
        buffer_size: usize,
        auto_snapshot: bool,
    ) -> std::io::Result<Self> {
        let wal_writer = if let Some(path) = wal_path {
            Some(WALWriter::new(path, buffer_size, 4096)?)
        } else {
            None
        };

        Ok(Self {
            wal_writer,
            replay_engine: ReplayEngine::new(buffer_size),
            expected_sequence: CachePadded::default(),
            total_gaps: CachePadded::default(),
            snapshots_triggered: CachePadded::default(),
            is_active: CachePadded::new(AtomicBool::new(true)),
            auto_snapshot_on_gap: auto_snapshot,
        })
    }

    /// Process an incoming event with sequence checking
    pub fn process_event(&self, event: ReplayEvent) -> Option<SequenceGapInfo> {
        if !self.is_active.data.load(Ordering::Acquire) {
            return None;
        }

        let expected = self.expected_sequence.data.load(Ordering::Acquire);
        
        // Check for gap
        if event.sequence != expected && expected > 0 {
            let gap_size = if event.sequence > expected {
                event.sequence - expected
            } else {
                0
            };

            let gap_info = SequenceGapInfo {
                expected_sequence: expected,
                received_sequence: event.sequence,
                gap_size,
                timestamp_ns: event.timestamp_ns,
                snapshot_requested: false,
            };

            // Record gap
            self.total_gaps.data.fetch_add(1, Ordering::AcqRel);

            // Trigger snapshot if configured
            let mut gap_info = gap_info;
            if self.auto_snapshot_on_gap && gap_size > 0 {
                gap_info.snapshot_requested = true;
                self.snapshots_triggered.data.fetch_add(1, Ordering::AcqRel);
            }

            // Update expected sequence
            self.expected_sequence
                .data
                .store(event.sequence + 1, Ordering::Release);

            // Write to WAL
            if let Some(ref wal) = self.wal_writer {
                let _ = wal.append(WALEntryType::SequenceGap, &[]);
            }

            return Some(gap_info);
        }

        // No gap - normal processing
        self.expected_sequence
            .data
            .store(event.sequence + 1, Ordering::Release);

        // Push to replay engine
        self.replay_engine.push_event(event);

        // Write to WAL
        if let Some(ref wal) = self.wal_writer {
            let entry_type = match event.event_type {
                ReplayEventType::Snapshot => WALEntryType::StateSnapshot,
                ReplayEventType::LevelAdd | ReplayEventType::LevelUpdate | ReplayEventType::LevelDelete => {
                    WALEntryType::OrderModified
                }
                ReplayEventType::Trade => WALEntryType::OrderFilled,
                ReplayEventType::SequenceGap => WALEntryType::Heartbeat,
            };

            // Serialize event data (simplified)
            let data = unsafe {
                std::slice::from_raw_parts(
                    &event as *const ReplayEvent as *const u8,
                    std::mem::size_of::<ReplayEvent>(),
                )
            };
            let _ = wal.append(entry_type, data);
        }

        None
    }

    /// Get current book state
    #[inline]
    pub fn get_current_book(&self) -> &ReconstructedBook {
        self.replay_engine.get_current_book()
    }

    /// Request a REST snapshot (called externally when gap detected)
    pub fn request_snapshot(&self) {
        self.snapshots_triggered.data.fetch_add(1, Ordering::AcqRel);
    }

    /// Get statistics
    pub fn get_stats(&self) -> EventStoreStats {
        let replay_stats = self.replay_engine.get_stats();

        EventStoreStats {
            expected_sequence: self.expected_sequence.data.load(Ordering::Acquire),
            total_gaps: self.total_gaps.data.load(Ordering::Acquire),
            snapshots_triggered: self.snapshots_triggered.data.load(Ordering::Acquire),
            events_processed: replay_stats.events_processed,
            is_active: self.is_active.data.load(Ordering::Acquire),
        }
    }

    /// Sync WAL to disk
    pub fn sync(&self) -> std::io::Result<()> {
        if let Some(ref wal) = self.wal_writer {
            wal.sync()?;
        }
        Ok(())
    }

    #[inline]
    pub fn set_active(&self, active: bool) {
        self.is_active.data.store(active, Ordering::Release);
        self.replay_engine.set_active(active);
        if let Some(ref wal) = self.wal_writer {
            wal.set_active(active);
        }
    }

    #[inline]
    pub fn is_active(&self) -> bool {
        self.is_active.data.load(Ordering::Acquire)
    }

    pub fn reset(&mut self) {
        self.expected_sequence.data.store(0, Ordering::Release);
        self.total_gaps.data.store(0, Ordering::Release);
        self.snapshots_triggered.data.store(0, Ordering::Release);
        self.replay_engine.reset();
    }
}

#[derive(Debug, Clone, Copy)]
pub struct EventStoreStats {
    pub expected_sequence: u64,
    pub total_gaps: u64,
    pub snapshots_triggered: u64,
    pub events_processed: u64,
    pub is_active: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_store_basic() {
        let store = EventStore::<&str>::new(None, 1024, true).unwrap();

        let event = ReplayEvent {
            event_type: ReplayEventType::LevelAdd,
            timestamp_ns: 1000,
            sequence: 1,
            price: 10000,
            quantity: 100,
            side: replay::Side::Bid,
            order_id: 1,
        };

        let gap = store.process_event(event);
        assert!(gap.is_none()); // No gap for first event

        let stats = store.get_stats();
        assert_eq!(stats.expected_sequence, 2);
        assert_eq!(stats.events_processed, 1);
    }

    #[test]
    fn test_gap_detection() {
        let store = EventStore::<&str>::new(None, 1024, true).unwrap();

        // First event
        let event1 = ReplayEvent {
            event_type: ReplayEventType::LevelAdd,
            timestamp_ns: 1000,
            sequence: 1,
            price: 10000,
            quantity: 100,
            side: replay::Side::Bid,
            order_id: 1,
        };
        store.process_event(event1);

        // Gap event (sequence jumps from 2 to 5)
        let event2 = ReplayEvent {
            event_type: ReplayEventType::LevelAdd,
            timestamp_ns: 2000,
            sequence: 5,
            price: 10001,
            quantity: 100,
            side: replay::Side::Ask,
            order_id: 2,
        };

        let gap = store.process_event(event2);
        assert!(gap.is_some());
        assert_eq!(gap.unwrap().gap_size, 3);

        let stats = store.get_stats();
        assert_eq!(stats.total_gaps, 1);
        assert!(stats.snapshots_triggered > 0);
    }
}
