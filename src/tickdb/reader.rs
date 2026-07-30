//! Trade Tick Database Reader
//! 
//! Zero-copy reader for historical data retrieval used in walk-forward testing.
//! Streams historical ticks directly into the backtesting engine using Rust's Iterator trait.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use memmap2::{Mmap, MmapOptions};
use thiserror::Error;
use serde::{Serialize, Deserialize};

use super::writer::{StoredTick, FileHeader, TickDbError};

/// Zero-copy tick iterator for backtesting
pub struct TickIterator<'a> {
    mmap: &'a Mmap,
    current_pos: usize,
    end_pos: usize,
    header_size: usize,
}

impl<'a> TickIterator<'a> {
    pub fn new(mmap: &'a Mmap, start_pos: usize, end_pos: usize) -> Self {
        Self {
            mmap,
            current_pos: start_pos,
            end_pos,
            header_size: FileHeader::SIZE,
        }
    }
}

impl<'a> Iterator for TickIterator<'a> {
    type Item = Result<StoredTick, TickDbError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_pos >= self.end_pos {
            return None;
        }

        // Try to deserialize a tick from current position
        let available = self.end_pos - self.current_pos;
        if available < StoredTick::SERIALIZE_SIZE * 2 {
            return None;
        }

        match bincode::deserialize::<StoredTick>(&self.mmap[self.current_pos..]) {
            Ok(tick) => {
                let serialized_size = bincode::serialized_size(&tick).ok()? as usize;
                self.current_pos += serialized_size;
                Some(Ok(tick))
            }
            Err(_) => {
                // Corrupted data or end of valid data
                None
            }
        }
    }
}

/// Time range filter for tick queries
#[derive(Debug, Clone, Copy)]
pub struct TimeRange {
    pub start_ns: u64,
    pub end_ns: u64,
}

impl TimeRange {
    pub fn new(start_ns: u64, end_ns: u64) -> Self {
        Self { start_ns, end_ns }
    }

    pub fn contains(&self, timestamp_ns: u64) -> bool {
        timestamp_ns >= self.start_ns && timestamp_ns <= self.end_ns
    }
}

/// Price range filter for tick queries
#[derive(Debug, Clone, Copy)]
pub struct PriceRange {
    pub min_price: f64,
    pub max_price: f64,
}

impl PriceRange {
    pub fn new(min_price: f64, max_price: f64) -> Self {
        Self { min_price, max_price }
    }

    pub fn contains(&self, price: f64) -> bool {
        price >= self.min_price && price <= self.max_price
    }
}

/// Query builder for tick database
pub struct TickQuery {
    time_range: Option<TimeRange>,
    price_range: Option<PriceRange>,
    limit: Option<usize>,
    reverse: bool,
}

impl TickQuery {
    pub fn new() -> Self {
        Self {
            time_range: None,
            price_range: None,
            limit: None,
            reverse: false,
        }
    }

    pub fn with_time_range(mut self, start_ns: u64, end_ns: u64) -> Self {
        self.time_range = Some(TimeRange::new(start_ns, end_ns));
        self
    }

    pub fn with_price_range(mut self, min_price: f64, max_price: f64) -> Self {
        self.price_range = Some(PriceRange::new(min_price, max_price));
        self
    }

    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    pub fn reverse(mut self) -> Self {
        self.reverse = true;
        self
    }

    pub fn matches(&self, tick: &StoredTick) -> bool {
        if let Some(ref tr) = self.time_range {
            if !tr.contains(tick.timestamp_ns) {
                return false;
            }
        }
        if let Some(ref pr) = self.price_range {
            if !pr.contains(tick.price) {
                return false;
            }
        }
        true
    }
}

impl Default for TickQuery {
    fn default() -> Self {
        Self::new()
    }
}

/// High-performance tick database reader
pub struct TickDbReader {
    path: PathBuf,
    mmap: Option<Mmap>,
    file: Option<File>,
    header: Option<FileHeader>,
    data_start: usize,
    data_end: usize,
}

unsafe impl Send for TickDbReader {}
unsafe impl Sync for TickDbReader {}

impl TickDbReader {
    /// Open a tick database for reading
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, TickDbError> {
        let path = path.as_ref().to_path_buf();
        
        let file = File::open(&path)?;
        let metadata = file.metadata()?;
        let file_len = metadata.len();

        if file_len < FileHeader::SIZE as u64 {
            return Err(TickDbError::Corruption("File too small".to_string()));
        }

        let mmap = unsafe {
            MmapOptions::new()
                .map(&file)?
        };

        // Read and validate header
        let header = unsafe {
            &*(mmap[0..FileHeader::SIZE].as_ptr() as *const FileHeader)
        };
        header.validate()?;

        let data_start = FileHeader::SIZE;
        let data_end = file_len as usize;

        Ok(Self {
            path,
            mmap: Some(mmap),
            file: Some(file),
            header: Some(*header),
            data_start,
            data_end,
        })
    }

    /// Get total tick count
    pub fn tick_count(&self) -> u64 {
        self.header.map(|h| h.tick_count).unwrap_or(0)
    }

    /// Get file size
    pub fn file_size(&self) -> u64 {
        self.header.map(|h| h.file_size).unwrap_or(0)
    }

    /// Get creation timestamp
    pub fn created_timestamp(&self) -> Option<u64> {
        self.header.map(|h| h.created_timestamp)
    }

    /// Iterate over all ticks (zero-copy)
    pub fn iter(&self) -> Option<TickIterator> {
        self.mmap.as_ref().map(|mmap| {
            TickIterator::new(mmap, self.data_start, self.data_end)
        })
    }

    /// Query ticks with filters
    pub fn query(&self, query: &TickQuery) -> TickQueryResult {
        TickQueryResult {
            reader: self,
            query,
            count: 0,
            limit_reached: false,
        }
    }

    /// Get first N ticks
    pub fn first_n(&self, n: usize) -> FirstNTicks {
        FirstNTicks {
            inner: self.iter(),
            remaining: n,
        }
    }

    /// Get last N ticks (requires scanning from end)
    pub fn last_n(&self, n: usize) -> Result<Vec<StoredTick>, TickDbError> {
        let mut ticks = Vec::with_capacity(n);
        let mut count = 0;

        if let Some(iter) = self.iter() {
            for result in iter {
                match result {
                    Ok(tick) => {
                        ticks.push(tick);
                        count += 1;
                    }
                    Err(e) => return Err(e),
                }
            }
        }

        // Return last n elements
        if ticks.len() > n {
            ticks.drain(0..ticks.len() - n);
        }

        Ok(ticks)
    }

    /// Stream ticks to a callback function
    pub fn stream<F>(&self, mut callback: F) -> Result<usize, TickDbError>
    where
        F: FnMut(&StoredTick) -> bool,
    {
        let mut count = 0;

        if let Some(iter) = self.iter() {
            for result in iter {
                match result {
                    Ok(tick) => {
                        count += 1;
                        if !callback(&tick) {
                            break;
                        }
                    }
                    Err(e) => return Err(e),
                }
            }
        }

        Ok(count)
    }

    /// Calculate statistics over tick data
    pub fn calculate_stats(&self) -> Result<TickStats, TickDbError> {
        let mut stats = TickStats::default();
        let mut first = true;

        self.stream(|tick| {
            if first {
                stats.first_timestamp = tick.timestamp_ns;
                stats.first_price = tick.price;
                first = false;
            }

            stats.last_timestamp = tick.timestamp_ns;
            stats.last_price = tick.price;
            stats.min_price = stats.min_price.min(tick.price);
            stats.max_price = stats.max_price.max(tick.price);
            stats.total_volume += tick.volume;
            stats.tick_count += 1;

            if tick.is_buyer_maker {
                stats.sell_volume += tick.volume;
            } else {
                stats.buy_volume += tick.volume;
            }

            true
        })?;

        Ok(stats)
    }

    /// Close the reader
    pub fn close(&mut self) {
        self.mmap = None;
        self.file = None;
        self.header = None;
    }
}

impl Drop for TickDbReader {
    fn drop(&mut self) {
        self.close();
    }
}

/// Query result iterator with filtering
pub struct TickQueryResult<'a> {
    reader: &'a TickDbReader,
    query: &'a TickQuery,
    count: usize,
    limit_reached: bool,
}

impl<'a> Iterator for TickQueryResult<'a> {
    type Item = Result<StoredTick, TickDbError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.limit_reached {
            return None;
        }

        if let Some(iter) = self.reader.iter() {
            for result in iter.skip(self.count) {
                self.count += 1;

                match result {
                    Ok(tick) => {
                        if self.query.matches(&tick) {
                            if let Some(limit) = self.query.limit {
                                if self.count >= limit {
                                    self.limit_reached = true;
                                }
                            }
                            return Some(Ok(tick));
                        }
                    }
                    Err(e) => return Some(Err(e)),
                }
            }
        }

        None
    }
}

/// Iterator for first N ticks
pub struct FirstNTicks<'a> {
    inner: Option<TickIterator<'a>>,
    remaining: usize,
}

impl<'a> Iterator for FirstNTicks<'a> {
    type Item = Result<StoredTick, TickDbError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }

        if let Some(ref mut iter) = self.inner {
            self.remaining -= 1;
            iter.next()
        } else {
            None
        }
    }
}

/// Statistics calculated from tick data
#[derive(Debug, Clone, Default)]
pub struct TickStats {
    pub tick_count: u64,
    pub first_timestamp: u64,
    pub last_timestamp: u64,
    pub first_price: f64,
    pub last_price: f64,
    pub min_price: f64,
    pub max_price: f64,
    pub total_volume: f64,
    pub buy_volume: f64,
    pub sell_volume: f64,
}

impl TickStats {
    /// Calculate VWAP (Volume Weighted Average Price)
    /// Note: This requires tick-by-tick calculation which we simplify here
    pub fn vwap_approx(&self) -> f64 {
        (self.min_price + self.max_price + self.last_price) / 3.0
    }

    /// Calculate price change
    pub fn price_change(&self) -> f64 {
        self.last_price - self.first_price
    }

    /// Calculate price change percentage
    pub fn price_change_pct(&self) -> f64 {
        if self.first_price == 0.0 {
            return 0.0;
        }
        (self.price_change() / self.first_price) * 100.0
    }

    /// Calculate buy/sell ratio
    pub fn buy_sell_ratio(&self) -> f64 {
        if self.sell_volume == 0.0 {
            if self.buy_volume == 0.0 {
                1.0
            } else {
                f64::MAX
            }
        } else {
            self.buy_volume / self.sell_volume
        }
    }

    /// Get duration in milliseconds
    pub fn duration_ms(&self) -> u64 {
        (self.last_timestamp - self.first_timestamp) / 1_000_000
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::writer::{TickDbWriter, TickDbConfig};
    use std::fs;

    #[test]
    fn test_reader_basic() {
        let temp_path = "/tmp/test_tickdb_read.db";
        let _ = fs::remove_file(temp_path);

        // Write some ticks first
        {
            let config = TickDbConfig::default();
            let writer = TickDbWriter::new(temp_path, config).unwrap();
            
            let ticks = vec![
                StoredTick::new(1000, 50000.0, 1.0, false, 0),
                StoredTick::new(2000, 50001.0, 2.0, true, 1),
                StoredTick::new(3000, 50002.0, 1.5, false, 2),
            ];
            writer.append_batch(&ticks).unwrap();
        }

        // Read them back
        let reader = TickDbReader::open(temp_path).unwrap();
        assert_eq!(reader.tick_count(), 3);

        let mut count = 0;
        if let Some(iter) = reader.iter() {
            for result in iter {
                assert!(result.is_ok());
                count += 1;
            }
        }
        assert_eq!(count, 3);

        let _ = fs::remove_file(temp_path);
    }

    #[test]
    fn test_query_filtering() {
        let temp_path = "/tmp/test_tickdb_query.db";
        let _ = fs::remove_file(temp_path);

        {
            let config = TickDbConfig::default();
            let writer = TickDbWriter::new(temp_path, config).unwrap();
            
            let ticks = vec![
                StoredTick::new(1000, 50000.0, 1.0, false, 0),
                StoredTick::new(2000, 50001.0, 2.0, true, 1),
                StoredTick::new(3000, 50002.0, 1.5, false, 2),
            ];
            writer.append_batch(&ticks).unwrap();
        }

        let reader = TickDbReader::open(temp_path).unwrap();
        
        // Query with time range
        let query = TickQuery::new()
            .with_time_range(1500, 2500);
        
        let results: Vec<_> = reader.query(&query).filter_map(|r| r.ok()).collect();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].timestamp_ns, 2000);

        let _ = fs::remove_file(temp_path);
    }

    #[test]
    fn test_stats_calculation() {
        let temp_path = "/tmp/test_tickdb_stats.db";
        let _ = fs::remove_file(temp_path);

        {
            let config = TickDbConfig::default();
            let writer = TickDbWriter::new(temp_path, config).unwrap();
            
            let ticks = vec![
                StoredTick::new(1000, 50000.0, 1.0, false, 0),
                StoredTick::new(2000, 50001.0, 2.0, true, 1),
                StoredTick::new(3000, 50002.0, 1.5, false, 2),
            ];
            writer.append_batch(&ticks).unwrap();
        }

        let reader = TickDbReader::open(temp_path).unwrap();
        let stats = reader.calculate_stats().unwrap();

        assert_eq!(stats.tick_count, 3);
        assert_eq!(stats.first_price, 50000.0);
        assert_eq!(stats.last_price, 50002.0);
        assert!((stats.total_volume - 4.5).abs() < 0.001);

        let _ = fs::remove_file(temp_path);
    }
}
