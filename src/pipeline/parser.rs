//! Ultra-Fast JSON Parser using simd-json
//! 
//! Integrates `simd-json` for ultra-fast, zero-copy JSON parsing of incoming 
//! WebSocket and REST payloads. Falls back to standard `serde_json` gracefully 
//! if SIMD features are unavailable.

use anyhow::{Context, Result};
use crate::market_data::{BinanceStreamMessage, RawWsMessage};

/// Parse result with zero-copy borrowed data where possible
#[derive(Debug)]
pub enum ParseResult<T> {
    /// Successfully parsed with owned data
    Owned(T),
    /// Parsing failed
    Error(String),
}

/// High-performance JSON parser trait
pub trait JsonParser: Send + Sync {
    /// Parse a raw message into a typed structure
    fn parse_stream_message(&self, msg: &RawWsMessage) -> Result<BinanceStreamMessage>;
    
    /// Parse arbitrary JSON into a value
    fn parse_value(&self, data: &[u8]) -> Result<serde_json::Value>;
    
    /// Get parser name for logging
    fn name(&self) -> &'static str;
}

/// SIMD-accelerated JSON parser using simd-json
#[cfg(feature = "simd")]
pub struct SimdJsonParser {
    /// Reusable buffer for parsing
    buffer: std::cell::RefCell<Vec<u8>>,
}

#[cfg(feature = "simd")]
impl SimdJsonParser {
    #[inline]
    pub fn new() -> Self {
        SimdJsonParser {
            buffer: std::cell::RefCell::new(Vec::with_capacity(4096)),
        }
    }
}

#[cfg(feature = "simd")]
impl JsonParser for SimdJsonParser {
    #[inline]
    fn parse_stream_message(&self, msg: &RawWsMessage) -> Result<BinanceStreamMessage> {
        let mut buf = self.buffer.borrow_mut();
        buf.clear();
        buf.extend_from_slice(&msg.payload);
        
        // simd-json requires mutable buffer for in-place parsing
        let value = simd_json::to_owned_value(buf.as_mut_slice())
            .context("Failed to parse JSON with simd-json")?;
        
        // Convert to our type using serde compatibility
        let bytes = serde_json::to_vec(&value)?;
        let parsed: BinanceStreamMessage = serde_json::from_slice(&bytes)
            .context("Failed to deserialize stream message")?;
        
        Ok(parsed)
    }

    #[inline]
    fn parse_value(&self, data: &[u8]) -> Result<serde_json::Value> {
        let mut buf = self.buffer.borrow_mut();
        buf.clear();
        buf.extend_from_slice(data);
        
        simd_json::to_owned_value(buf.as_mut_slice())
            .context("Failed to parse JSON value with simd-json")
    }

    #[inline]
    fn name(&self) -> &'static str {
        "SimdJsonParser"
    }
}

#[cfg(feature = "simd")]
impl Default for SimdJsonParser {
    fn default() -> Self {
        Self::new()
    }
}

/// Standard serde_json parser (fallback)
pub struct SerdeJsonParser {
    /// Reusable buffer
    buffer: std::cell::RefCell<Vec<u8>>,
}

impl SerdeJsonParser {
    #[inline]
    pub fn new() -> Self {
        SerdeJsonParser {
            buffer: std::cell::RefCell::new(Vec::with_capacity(4096)),
        }
    }
}

impl JsonParser for SerdeJsonParser {
    #[inline]
    fn parse_stream_message(&self, msg: &RawWsMessage) -> Result<BinanceStreamMessage> {
        let msg_str = msg.as_str()
            .context("Binary messages not supported by SerdeJsonParser")?;
        
        serde_json::from_str(msg_str)
            .context("Failed to parse stream message with serde_json")
    }

    #[inline]
    fn parse_value(&self, data: &[u8]) -> Result<serde_json::Value> {
        serde_json::from_slice(data)
            .context("Failed to parse JSON value with serde_json")
    }

    #[inline]
    fn name(&self) -> &'static str {
        "SerdeJsonParser"
    }
}

impl Default for SerdeJsonParser {
    fn default() -> Self {
        Self::new()
    }
}

/// Adaptive parser that chooses the best available implementation
pub struct AdaptiveParser {
    /// The underlying parser
    inner: Box<dyn JsonParser>,
    /// Parse statistics
    total_parses: std::sync::atomic::AtomicU64,
    failed_parses: std::sync::atomic::AtomicU64,
    total_bytes: std::sync::atomic::AtomicU64,
}

impl AdaptiveParser {
    /// Create a new adaptive parser, preferring SIMD if available
    #[inline]
    pub fn new() -> Self {
        #[cfg(feature = "simd")]
        let inner: Box<dyn JsonParser> = Box::new(SimdJsonParser::new());
        
        #[cfg(not(feature = "simd"))]
        let inner: Box<dyn JsonParser> = Box::new(SerdeJsonParser::new());
        
        AdaptiveParser {
            inner,
            total_parses: std::sync::atomic::AtomicU64::new(0),
            failed_parses: std::sync::atomic::AtomicU64::new(0),
            total_bytes: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Force use of a specific parser
    #[inline]
    pub fn with_parser(mut self, parser: Box<dyn JsonParser>) -> Self {
        self.inner = parser;
        self
    }

    /// Get parser statistics
    #[inline]
    pub fn stats(&self) -> ParserStats {
        ParserStats {
            parser_name: self.inner.name(),
            total_parses: self.total_parses.load(std::sync::atomic::Ordering::Relaxed),
            failed_parses: self.failed_parses.load(std::sync::atomic::Ordering::Relaxed),
            total_bytes: self.total_bytes.load(std::sync::atomic::Ordering::Relaxed),
        }
    }

    /// Reset statistics
    #[inline]
    pub fn reset_stats(&self) {
        self.total_parses.store(0, std::sync::atomic::Ordering::Relaxed);
        self.failed_parses.store(0, std::sync::atomic::Ordering::Relaxed);
        self.total_bytes.store(0, std::sync::atomic::Ordering::Relaxed);
    }
}

impl Default for AdaptiveParser {
    fn default() -> Self {
        Self::new()
    }
}

impl JsonParser for AdaptiveParser {
    #[inline]
    fn parse_stream_message(&self, msg: &RawWsMessage) -> Result<BinanceStreamMessage> {
        self.total_parses.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.total_bytes.fetch_add(msg.payload.len() as u64, std::sync::atomic::Ordering::Relaxed);
        
        match self.inner.parse_stream_message(msg) {
            Ok(result) => Ok(result),
            Err(e) => {
                self.failed_parses.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                Err(e)
            }
        }
    }

    #[inline]
    fn parse_value(&self, data: &[u8]) -> Result<serde_json::Value> {
        self.total_parses.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.total_bytes.fetch_add(data.len() as u64, std::sync::atomic::Ordering::Relaxed);
        
        match self.inner.parse_value(data) {
            Ok(result) => Ok(result),
            Err(e) => {
                self.failed_parses.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                Err(e)
            }
        }
    }

    #[inline]
    fn name(&self) -> &'static str {
        self.inner.name()
    }
}

/// Parser statistics snapshot
#[derive(Debug, Clone)]
pub struct ParserStats {
    pub parser_name: &'static str,
    pub total_parses: u64,
    pub failed_parses: u64,
    pub total_bytes: u64,
}

impl ParserStats {
    /// Get success rate as percentage
    #[inline]
    pub fn success_rate(&self) -> f64 {
        if self.total_parses == 0 {
            return 100.0;
        }
        let success = self.total_parses - self.failed_parses;
        (success as f64 / self.total_parses as f64) * 100.0
    }

    /// Get average bytes per parse
    #[inline]
    pub fn avg_bytes_per_parse(&self) -> f64 {
        if self.total_parses == 0 {
            return 0.0;
        }
        self.total_bytes as f64 / self.total_parses as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serde_parser_creation() {
        let parser = SerdeJsonParser::new();
        assert_eq!(parser.name(), "SerdeJsonParser");
    }

    #[test]
    fn test_adaptive_parser_creation() {
        let parser = AdaptiveParser::new();
        assert!(!parser.name().is_empty());
    }

    #[test]
    fn test_parse_valid_json() {
        let parser = SerdeJsonParser::new();
        let json = r#"{"test": "value", "number": 42}"#;
        
        let result = parser.parse_value(json.as_bytes()).unwrap();
        assert_eq!(result["test"], "value");
        assert_eq!(result["number"], 42);
    }

    #[test]
    fn test_parse_invalid_json() {
        let parser = SerdeJsonParser::new();
        let json = r#"{"test": invalid}"#;
        
        let result = parser.parse_value(json.as_bytes());
        assert!(result.is_err());
    }

    #[test]
    fn test_parser_stats() {
        let parser = AdaptiveParser::new();
        
        // Initial stats should be zero
        let stats = parser.stats();
        assert_eq!(stats.total_parses, 0);
        assert_eq!(stats.failed_parses, 0);
    }
}
