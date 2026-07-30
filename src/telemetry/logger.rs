//! Zero-Cost Asynchronous Structured Logger
//!
//! This module implements a zero-cost, asynchronous structured logger using a background thread.
//! Batch-write logs to disk so that logging operations never block the main trading logic
//! or event loop.
//!
//! Features:
//! - Lock-free log entry queue
//! - Background writer thread
//! - Batch writes for I/O efficiency
//! - Structured JSON output for parsing

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};
use crossbeam_channel::{bounded, Sender, Receiver};
use serde::{Serialize, Deserialize};
use anyhow::Context;

/// Log level enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum LogLevel {
    Trace = 0,
    Debug = 1,
    Info = 2,
    Warn = 3,
    Error = 4,
}

impl LogLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            LogLevel::Trace => "TRACE",
            LogLevel::Debug => "DEBUG",
            LogLevel::Info => "INFO",
            LogLevel::Warn => "WARN",
            LogLevel::Error => "ERROR",
        }
    }
    
    pub fn from_str(s: &str) -> Option<LogLevel> {
        match s.to_uppercase().as_str() {
            "TRACE" => Some(LogLevel::Trace),
            "DEBUG" => Some(LogLevel::Debug),
            "INFO" => Some(LogLevel::Info),
            "WARN" => Some(LogLevel::Warn),
            "ERROR" => Some(LogLevel::Error),
            _ => None,
        }
    }
}

/// A structured log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    /// Timestamp in nanoseconds since epoch
    pub timestamp_ns: u64,
    /// Log level
    pub level: LogLevel,
    /// Target/module name
    pub target: String,
    /// Log message
    pub message: String,
    /// Optional file location
    pub file: Option<String>,
    /// Optional line number
    pub line: Option<u32>,
    /// Additional key-value pairs (serialized as JSON string)
    pub fields: Option<String>,
    /// Thread ID
    pub thread_id: u64,
    /// Thread name
    pub thread_name: Option<String>,
}

impl LogEntry {
    /// Create a new log entry
    pub fn new(
        level: LogLevel,
        target: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            timestamp_ns: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
            level,
            target: target.into(),
            message: message.into(),
            file: None,
            line: None,
            fields: None,
            thread_id: thread::current().id().as_u64(),
            thread_name: thread::current().name().map(|s| s.to_string()),
        }
    }
    
    /// Set file and line information
    pub fn with_location(mut self, file: impl Into<String>, line: u32) -> Self {
        self.file = Some(file.into());
        self.line = Some(line);
        self
    }
    
    /// Add additional fields
    pub fn with_fields(mut self, fields: impl Into<String>) -> Self {
        self.fields = Some(fields.into());
        self
    }
    
    /// Format as JSON string
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| "{\"error\": \"failed to serialize\"}".to_string())
    }
    
    /// Format for human-readable output
    pub fn format_human(&self) -> String {
        let time_str = chrono::DateTime::from_timestamp(
            (self.timestamp_ns / 1_000_000_000) as i64,
            ((self.timestamp_ns % 1_000_000_000) as u32) * 1_000_000_000,
        )
        .map(|dt| dt.format("%Y-%m-%d %H:%M:%S%.9f").to_string())
        .unwrap_or_else(|| "UNKNOWN".to_string());
        
        format!(
            "{} {:5} [{}] {}: {}",
            time_str,
            self.level.as_str(),
            self.thread_name.as_deref().unwrap_or("unknown"),
            self.target,
            self.message
        )
    }
}

/// Configuration for the async logger
#[derive(Debug, Clone)]
pub struct LoggerConfig {
    /// Maximum queue size for pending log entries
    pub queue_size: usize,
    /// Batch size for writing to disk
    pub batch_size: usize,
    /// Flush interval in milliseconds
    pub flush_interval_ms: u64,
    /// Output file path (None for stdout only)
    pub output_file: Option<String>,
    /// Minimum log level to record
    pub min_level: LogLevel,
    /// Include file and line information
    pub include_location: bool,
}

impl Default for LoggerConfig {
    fn default() -> Self {
        Self {
            queue_size: 10000,
            batch_size: 100,
            flush_interval_ms: 100,
            output_file: None,
            min_level: LogLevel::Info,
            include_location: false,
        }
    }
}

/// Asynchronous logger with background writer
pub struct AsyncLogger {
    /// Sender for log entries
    sender: Sender<LogEntry>,
    /// Receiver for log entries (owned by writer thread)
    receiver: Option<Receiver<LogEntry>>,
    /// Shutdown flag
    shutdown: Arc<AtomicBool>,
    /// Writer thread handle
    writer: Option<JoinHandle<()>>,
    /// Entries logged counter
    entries_logged: AtomicUsize,
    /// Entries dropped counter
    entries_dropped: AtomicUsize,
    /// Configuration
    config: LoggerConfig,
}

unsafe impl Send for AsyncLogger {}
unsafe impl Sync for AsyncLogger {}

impl AsyncLogger {
    /// Create a new async logger with the given configuration
    pub fn new(config: LoggerConfig) -> Result<Arc<Self>, anyhow::Error> {
        let (tx, rx) = bounded::<LogEntry>(config.queue_size);
        
        let logger = Arc::new(AsyncLogger {
            sender: tx,
            receiver: Some(rx),
            shutdown: Arc::new(AtomicBool::new(false)),
            writer: None,
            entries_logged: AtomicUsize::new(0),
            entries_dropped: AtomicUsize::new(0),
            config,
        });
        
        Ok(logger)
    }
    
    /// Start the background writer thread
    pub fn start(self: &Arc<Self>) -> Result<(), anyhow::Error> {
        let receiver = self.receiver
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Logger already started"))?
            .clone();
        
        let shutdown = Arc::clone(&self.shutdown);
        let config = self.config.clone();
        let logger_clone = Arc::clone(self);
        
        let handle = thread::Builder::new()
            .name("log-writer".to_string())
            .spawn(move || {
                Self::writer_loop(receiver, shutdown, config, logger_clone);
            })
            .context("Failed to spawn log writer thread")?;
        
        // Safety: We're storing the handle after successful spawn
        let this_mut = unsafe {
            &mut *(Arc::as_ptr(self) as *mut AsyncLogger)
        };
        this_mut.writer = Some(handle);
        
        Ok(())
    }
    
    /// Writer loop that batches and writes log entries
    fn writer_loop(
        receiver: Receiver<LogEntry>,
        shutdown: Arc<AtomicBool>,
        config: LoggerConfig,
        logger: Arc<AsyncLogger>,
    ) {
        let mut buffer: Vec<LogEntry> = Vec::with_capacity(config.batch_size);
        let mut last_flush = Instant::now();
        let mut file_handle: Option<std::fs::File> = None;
        
        // Open output file if configured
        if let Some(ref path) = config.output_file {
            match std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
            {
                Ok(f) => {
                    file_handle = Some(f);
                    tracing::info!("Log file opened: {}", path);
                }
                Err(e) => {
                    eprintln!("Failed to open log file {}: {}", path, e);
                }
            }
        }
        
        while !shutdown.load(Ordering::Relaxed) {
            // Try to receive log entries
            match receiver.recv_timeout(Duration::from_millis(10)) {
                Ok(entry) => {
                    buffer.push(entry);
                    
                    // Flush if buffer is full
                    if buffer.len() >= config.batch_size {
                        Self::flush_buffer(&mut buffer, file_handle.as_mut(), &logger);
                        last_flush = Instant::now();
                    }
                }
                Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                    // Check if we should flush due to time interval
                    if !buffer.is_empty() && last_flush.elapsed() >= Duration::from_millis(config.flush_interval_ms) {
                        Self::flush_buffer(&mut buffer, file_handle.as_mut(), &logger);
                        last_flush = Instant::now();
                    }
                }
                Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                    break;
                }
            }
        }
        
        // Flush remaining entries
        if !buffer.is_empty() {
            Self::flush_buffer(&mut buffer, file_handle.as_mut(), &logger);
        }
        
        tracing::info!("Log writer shutting down");
    }
    
    /// Flush the buffer to output
    fn flush_buffer(
        buffer: &mut Vec<LogEntry>,
        file: Option<&mut std::fs::File>,
        logger: &AsyncLogger,
    ) {
        use std::io::Write;
        
        for entry in buffer.drain(..) {
            // Write to stdout/stderr based on level
            let output = entry.format_human();
            
            if entry.level >= LogLevel::Error {
                eprintln!("{}", output);
            } else {
                println!("{}", output);
            }
            
            // Write to file if configured
            if let Some(f) = file {
                let json = entry.to_json();
                let _ = writeln!(f, "{}", json);
            }
            
            logger.entries_logged.fetch_add(1, Ordering::Relaxed);
        }
        
        // Flush file handle
        if let Some(f) = file {
            let _ = f.flush();
        }
    }
    
    /// Log an entry (non-blocking, drops if queue is full)
    pub fn log(&self, entry: LogEntry) {
        if entry.level < self.config.min_level {
            return;
        }
        
        if self.sender.try_send(entry).is_err() {
            self.entries_dropped.fetch_add(1, Ordering::Relaxed);
        }
    }
    
    /// Convenience methods for different log levels
    pub fn trace(&self, target: impl Into<String>, message: impl Into<String>) {
        self.log(LogEntry::new(LogLevel::Trace, target, message));
    }
    
    pub fn debug(&self, target: impl Into<String>, message: impl Into<String>) {
        self.log(LogEntry::new(LogLevel::Debug, target, message));
    }
    
    pub fn info(&self, target: impl Into<String>, message: impl Into<String>) {
        self.log(LogEntry::new(LogLevel::Info, target, message));
    }
    
    pub fn warn(&self, target: impl Into<String>, message: impl Into<String>) {
        self.log(LogEntry::new(LogLevel::Warn, target, message));
    }
    
    pub fn error(&self, target: impl Into<String>, message: impl Into<String>) {
        self.log(LogEntry::new(LogLevel::Error, target, message));
    }
    
    /// Stop the logger gracefully
    pub fn stop(&self) {
        self.shutdown.store(true, Ordering::Release);
        
        if let Some(writer) = &self.writer {
            let _ = writer.thread().unpark();
        }
    }
    
    /// Get statistics
    pub fn get_stats(&self) -> LoggerStats {
        LoggerStats {
            entries_logged: self.entries_logged.load(Ordering::Relaxed),
            entries_dropped: self.entries_dropped.load(Ordering::Relaxed),
            queue_len: self.sender.len(),
        }
    }
}

impl Drop for AsyncLogger {
    fn drop(&mut self) {
        self.stop();
        
        if let Some(writer) = self.writer.take() {
            let _ = writer.join();
        }
    }
}

/// Logger statistics
#[derive(Debug, Clone, Default)]
pub struct LoggerStats {
    pub entries_logged: usize,
    pub entries_dropped: usize,
    pub queue_len: usize,
}

impl LoggerStats {
    pub fn format(&self) -> String {
        format!(
            "Logger | Logged: {} | Dropped: {} | Queue: {}",
            self.entries_logged,
            self.entries_dropped,
            self.queue_len
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_log_entry_creation() {
        let entry = LogEntry::new(LogLevel::Info, "test", "Hello, World!");
        
        assert_eq!(entry.level, LogLevel::Info);
        assert_eq!(entry.target, "test");
        assert_eq!(entry.message, "Hello, World!");
        assert!(entry.timestamp_ns > 0);
    }
    
    #[test]
    fn test_log_entry_format() {
        let entry = LogEntry::new(LogLevel::Warn, "module", "Warning message");
        let formatted = entry.format_human();
        
        assert!(formatted.contains("WARN"));
        assert!(formatted.contains("Warning message"));
    }
    
    #[test]
    fn test_async_logger_basic() {
        let config = LoggerConfig {
            queue_size: 100,
            batch_size: 10,
            flush_interval_ms: 50,
            output_file: None,
            min_level: LogLevel::Debug,
            include_location: false,
        };
        
        let logger = AsyncLogger::new(config).unwrap();
        logger.start().unwrap();
        
        logger.info("test", "Test message 1");
        logger.debug("test", "Test message 2");
        logger.warn("test", "Test message 3");
        
        std::thread::sleep(Duration::from_millis(200));
        
        let stats = logger.get_stats();
        assert!(stats.entries_logged >= 3);
        
        logger.stop();
    }
}
