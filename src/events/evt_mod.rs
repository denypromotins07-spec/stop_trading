//! Events Module Root
//! 
//! Wires parsed news signals directly to the global kill switch and alpha engines.

pub mod news_stream;
pub mod keyword_matcher;

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use self::news_stream::{NewsStream, NewsMessage, NewsPriority, NewsStreamState};
use self::keyword_matcher::{AhoCorasickMatcher, KeywordMatch, CRITICAL_KEYWORDS, KeywordMatcherBuilder};

/// Event signal for trading systems
#[derive(Debug, Clone, Copy)]
pub struct EventSignal {
    /// Signal type: 0=none, 1=volatility, 2=halt, 3=alpha adjustment
    pub signal_type: u8,
    /// Priority level (0-3)
    pub priority: u8,
    /// Sentiment impact (-1.0 to 1.0)
    pub sentiment_impact: f64,
    /// Volatility multiplier (1.0 = normal, >1.0 = elevated)
    pub volatility_multiplier: f64,
    /// Whether to trigger halt
    pub should_halt: bool,
    /// Affected symbol hash
    pub symbol_hash: u64,
    /// Timestamp in nanoseconds
    pub timestamp_ns: u64,
}

impl Default for EventSignal {
    fn default() -> Self {
        Self {
            signal_type: 0,
            priority: 0,
            sentiment_impact: 0.0,
            volatility_multiplier: 1.0,
            should_halt: false,
            symbol_hash: 0,
            timestamp_ns: 0,
        }
    }
}

/// Cache-line aligned event processor state
#[repr(align(64))]
pub struct EventProcessorState {
    /// Events processed count
    events_processed: AtomicU64,
    /// Critical events count
    critical_events: AtomicU64,
    /// Halts triggered count
    halts_triggered: AtomicU64,
    /// Last event timestamp
    last_event_ns: AtomicU64,
    /// System halted flag
    system_halted: AtomicBool,
    _pad: [u8; 32],
}

impl EventProcessorState {
    pub const fn new() -> Self {
        Self {
            events_processed: AtomicU64::new(0),
            critical_events: AtomicU64::new(0),
            halts_triggered: AtomicU64::new(0),
            last_event_ns: AtomicU64::new(0),
            system_halted: AtomicBool::new(false),
            _pad: [0; 32],
        }
    }

    #[inline]
    pub fn record_event(&self, is_critical: bool) {
        self.events_processed.fetch_add(1, Ordering::Relaxed);
        if is_critical {
            self.critical_events.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[inline]
    pub fn record_halt(&self) {
        self.halts_triggered.fetch_add(1, Ordering::Relaxed);
        self.system_halted.store(true, Ordering::Release);
    }

    #[inline]
    pub fn clear_halt(&self) {
        self.system_halted.store(false, Ordering::Release);
    }

    #[inline]
    pub fn update_timestamp(&self, ts_ns: u64) {
        self.last_event_ns.store(ts_ns, Ordering::Relaxed);
    }

    #[inline]
    pub fn is_halted(&self) -> bool {
        self.system_halted.load(Ordering::Acquire)
    }

    #[inline]
    pub fn events_processed(&self) -> u64 {
        self.events_processed.load(Ordering::Relaxed)
    }

    #[inline]
    pub fn critical_events(&self) -> u64 {
        self.critical_events.load(Ordering::Relaxed)
    }

    #[inline]
    pub fn halts_triggered(&self) -> u64 {
        self.halts_triggered.load(Ordering::Relaxed)
    }
}

/// Main event processor combining news stream and keyword matching
#[repr(align(64))]
pub struct EventProcessor {
    news_stream: NewsStream,
    keyword_matcher: AhoCorasickMatcher,
    state: Arc<EventProcessorState>,
    /// Volatility multiplier for elevated conditions
    elevated_vol_mult: f64,
    /// Halt on critical flag
    halt_on_critical: AtomicBool,
    _pad: [u8; 32],
}

impl EventProcessor {
    /// Create new event processor with default critical keywords
    pub fn new() -> Self {
        let mut matcher = AhoCorasickMatcher::new();
        let patterns: Vec<(&str, u8)> = CRITICAL_KEYWORDS.to_vec();
        matcher.build(&patterns);

        Self {
            news_stream: NewsStream::new(),
            keyword_matcher: matcher,
            state: Arc::new(EventProcessorState::new()),
            elevated_vol_mult: 2.0,
            halt_on_critical: AtomicBool::new(true),
            _pad: [0; 32],
        }
    }

    /// Create with custom keyword matcher
    pub fn with_matcher(matcher: AhoCorasickMatcher) -> Self {
        Self {
            news_stream: NewsStream::new(),
            keyword_matcher: matcher,
            state: Arc::new(EventProcessorState::new()),
            elevated_vol_mult: 2.0,
            halt_on_critical: AtomicBool::new(true),
            _pad: [0; 32],
        }
    }

    /// Enable/disable halt on critical events
    #[inline]
    pub fn set_halt_on_critical(&self, enabled: bool) {
        self.halt_on_critical.store(enabled, Ordering::Relaxed);
    }

    /// Set elevated volatility multiplier
    #[inline]
    pub fn set_volatility_multiplier(&mut self, mult: f64) {
        self.elevated_vol_mult = mult.max(1.0);
    }

    /// Process incoming news message
    pub fn process_message(&self, message: &NewsMessage) -> EventSignal {
        let timestamp_ns = Instant::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos() as u64;

        // Check for critical keywords in headline
        let has_critical = self.keyword_matcher.has_critical(&message.headline);
        
        // Also scan body if present
        let body_critical = message.body.as_ref()
            .map(|body| self.keyword_matcher.has_critical(body))
            .unwrap_or(false);

        let is_critical = has_critical || body_critical || message.priority == NewsPriority::Critical;

        // Record event
        self.state.record_event(is_critical);
        self.state.update_timestamp(timestamp_ns);

        // Build signal
        let mut signal = EventSignal::default();
        signal.timestamp_ns = timestamp_ns;
        signal.priority = message.priority as u8;

        if is_critical {
            signal.signal_type = 2; // Volatility/Alert
            signal.volatility_multiplier = self.elevated_vol_mult;
            signal.sentiment_impact = -0.5; // Negative sentiment for critical events

            if self.halt_on_critical.load(Ordering::Relaxed) || message.halt_trigger {
                signal.should_halt = true;
                signal.signal_type = 3; // Halt signal
                self.state.record_halt();
            }
        } else if message.volatility_trigger {
            signal.signal_type = 1; // Volatility adjustment
            signal.volatility_multiplier = 1.5;
        }

        // Calculate sentiment from message
        signal.sentiment_impact = self.calculate_sentiment(message);

        // Symbol hash (simple hash of first symbol)
        if let Some(first_symbol) = message.symbols.first() {
            signal.symbol_hash = self.hash_symbol(first_symbol);
        }

        signal
    }

    /// Process raw text directly
    pub fn process_raw(&self, source: &str, text: &[u8]) -> Option<EventSignal> {
        if !self.news_stream.is_enabled() {
            return None;
        }

        // Process through news stream
        self.news_stream.process_raw(source, text);

        // Quick critical check
        if let Ok(text_str) = std::str::from_utf8(text) {
            if self.keyword_matcher.has_critical(text_str) {
                let mut msg = NewsMessage::new(
                    0,
                    source,
                    text_str,
                    NewsPriority::Critical,
                );
                msg.set_now();
                msg.halt_trigger = true;
                return Some(self.process_message(&msg));
            }
        }

        None
    }

    /// Calculate sentiment score from message
    fn calculate_sentiment(&self, message: &NewsMessage) -> f64 {
        let text = format!("{} {}", message.headline, message.body.as_deref().unwrap_or("")).to_lowercase();

        let positive_words = ["upgrade", "beat", "surge", "gain", "profit", "growth", "bullish"];
        let negative_words = ["downgrade", "miss", "crash", "loss", "hack", "exploit", "bearish"];

        let mut score = 0.0f64;

        for word in positive_words.iter() {
            if text.contains(word) {
                score += 0.15;
            }
        }

        for word in negative_words.iter() {
            if text.contains(word) {
                score -= 0.15;
            }
        }

        score.clamp(-1.0, 1.0)
    }

    /// Simple symbol hash
    fn hash_symbol(&self, symbol: &str) -> u64 {
        let mut hash: u64 = 0;
        for byte in symbol.bytes() {
            hash = hash.wrapping_mul(31).wrapping_add(byte as u64);
        }
        hash
    }

    /// Check if system is halted
    #[inline]
    pub fn is_halted(&self) -> bool {
        self.state.is_halted()
    }

    /// Clear halt status (manual override)
    #[inline]
    pub fn clear_halt(&self) {
        self.state.clear_halt();
    }

    /// Get processor statistics
    pub fn stats(&self) -> EventProcessorStats {
        EventProcessorStats {
            events_processed: self.state.events_processed(),
            critical_events: self.state.critical_events(),
            halts_triggered: self.state.halts_triggered(),
            is_halted: self.state.is_halted(),
            news_stats: self.news_stream.stats(),
        }
    }

    /// Get reference to news stream
    #[inline]
    pub fn news_stream(&self) -> &NewsStream {
        &self.news_stream
    }

    /// Get reference to keyword matcher
    #[inline]
    pub fn keyword_matcher(&self) -> &AhoCorasickMatcher {
        &self.keyword_matcher
    }

    /// Reset processor state
    pub fn reset(&self) {
        self.news_stream.reset();
        self.state.clear_halt();
    }
}

/// Event processor statistics
#[derive(Debug, Clone, Copy)]
pub struct EventProcessorStats {
    pub events_processed: u64,
    pub critical_events: u64,
    pub halts_triggered: u64,
    pub is_halted: bool,
    pub news_stats: news_stream::NewsStreamStats,
}

impl Default for EventProcessor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_critical_event_processing() {
        let processor = EventProcessor::new();

        let mut msg = NewsMessage::new(1, "test", "Major hack detected", NewsPriority::Critical);
        msg.set_now();
        msg.halt_trigger = true;

        let signal = processor.process_message(&msg);

        assert!(signal.should_halt);
        assert_eq!(signal.signal_type, 3);
        assert!(signal.volatility_multiplier > 1.0);
        assert!(processor.is_halted());
    }

    #[test]
    fn test_normal_event_processing() {
        let processor = EventProcessor::new();

        let mut msg = NewsMessage::new(1, "test", "Company reports earnings", NewsPriority::Medium);
        msg.set_now();

        let signal = processor.process_message(&msg);

        assert!(!signal.should_halt);
        assert!(signal.signal_type <= 1);
    }

    #[test]
    fn test_raw_processing() {
        let processor = EventProcessor::new();
        processor.news_stream.set_enabled(true);

        let result = processor.process_raw("test", b"Exchange hack reported");
        
        assert!(result.is_some());
        let signal = result.unwrap();
        assert!(signal.should_halt);
    }

    #[test]
    fn test_halt_override() {
        let processor = EventProcessor::new();
        processor.set_halt_on_critical(false);

        let mut msg = NewsMessage::new(1, "test", "Hack alert", NewsPriority::Critical);
        msg.set_now();

        let signal = processor.process_message(&msg);
        
        // Should still be critical but may not halt if halt_on_critical is false
        // unless message itself has halt_trigger
        assert_eq!(signal.priority, 3);
    }

    #[test]
    fn test_statistics() {
        let processor = EventProcessor::new();

        let mut msg1 = NewsMessage::new(1, "test", "Normal news", NewsPriority::Low);
        msg1.set_now();
        processor.process_message(&msg1);

        let mut msg2 = NewsMessage::new(2, "test", "Critical hack", NewsPriority::Critical);
        msg2.set_now();
        msg2.halt_trigger = true;
        processor.process_message(&msg2);

        let stats = processor.stats();
        assert_eq!(stats.events_processed, 2);
        assert_eq!(stats.critical_events, 1);
        assert_eq!(stats.halts_triggered, 1);
        assert!(stats.is_halted);
    }
}
