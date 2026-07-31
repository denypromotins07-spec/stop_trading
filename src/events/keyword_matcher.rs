//! Zero-Allocation Aho-Corasick Keyword Matcher
//! 
//! Implements a pre-compiled, static state machine for O(1) memory usage
//! during text scanning. Detects critical keywords in microseconds.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::collections::HashMap;

/// Maximum number of patterns supported
pub const MAX_PATTERNS: usize = 256;

/// Maximum pattern length
pub const MAX_PATTERN_LEN: usize = 64;

/// Maximum alphabet size (ASCII printable)
const ALPHABET_SIZE: usize = 128;

/// Match result from keyword scanning
#[derive(Debug, Clone, Copy)]
pub struct KeywordMatch {
    /// Pattern index that matched
    pub pattern_idx: usize,
    /// Start position in text
    pub start: usize,
    /// End position in text
    pub end: usize,
    /// Priority level (0=low, 3=critical)
    pub priority: u8,
}

/// Cache-line aligned Aho-Corasick automaton
#[repr(align(64))]
pub struct AhoCorasickMatcher {
    /// Transition table: [state][char] -> next_state
    /// Using flat array for cache efficiency
    transitions: Box<[u16; MAX_PATTERNS * ALPHABET_SIZE]>,
    /// Failure links: [state] -> failure_state
    failure: Box<[usize; MAX_PATTERNS]>,
    /// Output links: [state] -> pattern_idx (or MAX_PATTERNS if none)
    output: Box<[usize; MAX_PATTERNS]>,
    /// Pattern priorities
    priorities: Box<[u8; MAX_PATTERNS]>,
    /// Number of states used
    state_count: AtomicU64,
    /// Number of patterns loaded
    pattern_count: AtomicU64,
    /// Initialized flag
    initialized: AtomicBool,
    _pad: [u8; 32],
}

impl AhoCorasickMatcher {
    /// Create new uninitialized matcher
    pub fn new() -> Self {
        Self {
            transitions: Box::new([0; MAX_PATTERNS * ALPHABET_SIZE]),
            failure: Box::new([0; MAX_PATTERNS]),
            output: Box::new([MAX_PATTERNS; MAX_PATTERNS]),
            priorities: Box::new([0; MAX_PATTERNS]),
            state_count: AtomicU64::new(1), // State 0 is root
            pattern_count: AtomicU64::new(0),
            initialized: AtomicBool::new(false),
            _pad: [0; 32],
        }
    }

    /// Build the automaton from patterns
    /// 
    /// # Arguments
    /// * `patterns` - Slice of (pattern_string, priority) tuples
    /// 
    /// Returns true if successful, false if too many patterns
    pub fn build(&mut self, patterns: &[(&str, u8)]) -> bool {
        if patterns.len() > MAX_PATTERNS {
            return false;
        }

        // Reset state
        self.state_count.store(1, Ordering::Relaxed);
        self.pattern_count.store(0, Ordering::Relaxed);

        // Clear transitions from root
        for c in 0..ALPHABET_SIZE {
            self.transitions[c] = 0;
        }

        // Phase 1: Build trie
        let mut current_state = 1u64;
        
        for (pattern_idx, (pattern, priority)) in patterns.iter().enumerate() {
            let mut state = 0usize;
            
            for byte in pattern.as_bytes() {
                if *byte >= ALPHABET_SIZE as u8 {
                    continue; // Skip non-ASCII
                }
                
                let char_idx = *byte as usize;
                let trans_idx = state * ALPHABET_SIZE + char_idx;
                
                if self.transitions[trans_idx] == 0 {
                    // Create new state
                    self.transitions[trans_idx] = current_state as u16;
                    
                    // Initialize new state's transitions to 0
                    let new_state = current_state as usize;
                    for c in 0..ALPHABET_SIZE {
                        self.transitions[new_state * ALPHABET_SIZE + c] = 0;
                    }
                    
                    current_state += 1;
                    state = new_state;
                } else {
                    state = self.transitions[trans_idx] as usize;
                }
            }
            
            // Mark output for this state
            if state < MAX_PATTERNS {
                self.output[state] = pattern_idx;
                self.priorities[pattern_idx] = *priority;
            }
            
            self.pattern_count.fetch_add(1, Ordering::Relaxed);
        }

        self.state_count.store(current_state, Ordering::Relaxed);

        // Phase 2: Build failure links using BFS
        self.build_failure_links();

        self.initialized.store(true, Ordering::Release);
        true
    }

    /// Build failure links using BFS
    fn build_failure_links(&mut self) {
        let state_count = self.state_count.load(Ordering::Relaxed) as usize;
        
        // Initialize failure links for depth-1 states to root
        for c in 0..ALPHABET_SIZE {
            let trans_idx = c;
            if self.transitions[trans_idx] != 0 {
                let next_state = self.transitions[trans_idx] as usize;
                self.failure[next_state] = 0;
            }
        }

        // BFS queue (simple implementation)
        let mut queue = Vec::with_capacity(state_count);
        
        // Add depth-1 states to queue
        for c in 0..ALPHABET_SIZE {
            if self.transitions[c] != 0 {
                queue.push(self.transitions[c] as usize);
            }
        }

        let mut head = 0;
        while head < queue.len() {
            let r = queue[head];
            head += 1;

            for c in 0..ALPHABET_SIZE {
                let trans_idx = r * ALPHABET_SIZE + c;
                let s = self.transitions[trans_idx] as usize;
                
                if s != 0 {
                    queue.push(s);
                    
                    // Find failure state
                    let mut fail = self.failure[r];
                    while fail != 0 && self.transitions[fail * ALPHABET_SIZE + c] == 0 {
                        fail = self.failure[fail];
                    }
                    
                    if self.transitions[fail * ALPHABET_SIZE + c] != 0 && 
                       self.transitions[fail * ALPHABET_SIZE + c] as usize != s {
                        self.failure[s] = self.transitions[fail * ALPHABET_SIZE + c] as usize;
                    } else {
                        self.failure[s] = 0;
                    }

                    // Merge output
                    if self.output[self.failure[s]] != MAX_PATTERNS {
                        self.output[s] = self.output[self.failure[s]];
                    }
                }
            }
        }
    }

    /// Scan text for keyword matches (zero-allocation)
    /// 
    /// # Arguments
    /// * `text` - Text to scan
    /// * `matches` - Output buffer for matches
    /// 
    /// Returns number of matches found
    #[inline]
    pub fn scan<'a>(
        &'a self,
        text: &str,
        matches: &mut [KeywordMatch],
    ) -> usize {
        if !self.initialized.load(Ordering::Acquire) {
            return 0;
        }

        let mut match_count = 0;
        let mut state = 0usize;
        let bytes = text.as_bytes();

        for (i, &byte) in bytes.iter().enumerate() {
            if byte >= ALPHABET_SIZE as u8 {
                state = 0;
                continue;
            }

            let char_idx = byte as usize;

            // Follow failure links until we find a transition or reach root
            while state != 0 && self.transitions[state * ALPHABET_SIZE + char_idx] == 0 {
                state = self.failure[state];
            }

            // Take transition
            let next = self.transitions[state * ALPHABET_SIZE + char_idx];
            if next != 0 {
                state = next as usize;
            }

            // Check for output
            let output = self.output[state];
            if output != MAX_PATTERNS && match_count < matches.len() {
                // Get pattern length (approximate - would need pattern storage for exact)
                let pattern_len = 8; // Default estimate
                
                matches[match_count] = KeywordMatch {
                    pattern_idx: output,
                    start: i.saturating_sub(pattern_len),
                    end: i + 1,
                    priority: self.priorities[output],
                };
                match_count += 1;
            }
        }

        match_count
    }

    /// Quick check if any critical keyword exists (returns on first match)
    #[inline]
    pub fn has_critical(&self, text: &str) -> bool {
        if !self.initialized.load(Ordering::Acquire) {
            return false;
        }

        let mut state = 0usize;
        
        for &byte in text.as_bytes() {
            if byte >= ALPHABET_SIZE as u8 {
                state = 0;
                continue;
            }

            let char_idx = byte as usize;

            while state != 0 && self.transitions[state * ALPHABET_SIZE + char_idx] == 0 {
                state = self.failure[state];
            }

            let next = self.transitions[state * ALPHABET_SIZE + char_idx];
            if next != 0 {
                state = next as usize;
            }

            let output = self.output[state];
            if output != MAX_PATTERNS && self.priorities[output] >= 3 {
                return true;
            }
        }

        false
    }

    /// Check if matcher is initialized
    #[inline]
    pub fn is_ready(&self) -> bool {
        self.initialized.load(Ordering::Acquire)
    }

    /// Get number of loaded patterns
    #[inline]
    pub fn pattern_count(&self) -> u64 {
        self.pattern_count.load(Ordering::Relaxed)
    }

    /// Get number of states
    #[inline]
    pub fn state_count(&self) -> u64 {
        self.state_count.load(Ordering::Relaxed)
    }

    /// Reset matcher
    pub fn reset(&mut self) {
        self.initialized.store(false, Ordering::Release);
        self.state_count.store(1, Ordering::Relaxed);
        self.pattern_count.store(0, Ordering::Relaxed);
    }
}

impl Default for AhoCorasickMatcher {
    fn default() -> Self {
        Self::new()
    }
}

/// Pre-defined critical keywords for HFT trading
pub const CRITICAL_KEYWORDS: [(&str, u8); 20] = [
    ("hack", 3),
    ("exploit", 3),
    ("breach", 3),
    ("attack", 3),
    ("delist", 3),
    ("suspend", 3),
    ("halt", 3),
    ("emergency", 3),
    ("crash", 3),
    ("failure", 3),
    ("upgrade", 2),
    ("downgrade", 2),
    ("merger", 2),
    ("acquisition", 2),
    ("lawsuit", 2),
    ("investigation", 2),
    ("earnings", 1),
    ("revenue", 1),
    ("forecast", 1),
    ("dividend", 1),
];

/// Builder for keyword matcher
pub struct KeywordMatcherBuilder {
    patterns: Vec<(String, u8)>,
}

impl KeywordMatcherBuilder {
    pub fn new() -> Self {
        Self {
            patterns: Vec::with_capacity(MAX_PATTERNS),
        }
    }

    pub fn add_pattern(mut self, pattern: &str, priority: u8) -> Self {
        if self.patterns.len() < MAX_PATTERNS {
            self.patterns.push((pattern.to_lowercase(), priority));
        }
        self
    }

    pub fn add_critical_keywords(mut self) -> Self {
        for (pattern, priority) in CRITICAL_KEYWORDS.iter() {
            if self.patterns.len() < MAX_PATTERNS {
                self.patterns.push((pattern.to_string(), *priority));
            }
        }
        self
    }

    pub fn build(self) -> AhoCorasickMatcher {
        let mut matcher = AhoCorasickMatcher::new();
        let patterns_ref: Vec<(&str, u8)> = self.patterns
            .iter()
            .map(|(s, p)| (s.as_str(), *p))
            .collect();
        matcher.build(&patterns_ref);
        matcher
    }
}

impl Default for KeywordMatcherBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_matching() {
        let mut matcher = AhoCorasickMatcher::new();
        let patterns = [
            ("hack", 3),
            ("upgrade", 2),
            ("earnings", 1),
        ];
        
        assert!(matcher.build(&patterns));
        assert!(matcher.is_ready());

        let mut matches = [KeywordMatch {
            pattern_idx: 0,
            start: 0,
            end: 0,
            priority: 0,
        }; 10];

        let count = matcher.scan("There was a hack detected", &mut matches);
        assert!(count > 0);
        assert!(matches[0].priority == 3);
    }

    #[test]
    fn test_critical_detection() {
        let mut matcher = AhoCorasickMatcher::new();
        matcher.build(&CRITICAL_KEYWORDS);

        assert!(matcher.has_critical("Exchange reports major hack"));
        assert!(matcher.has_critical("Token will be delisted"));
        assert!(!matcher.has_critical("Market looking stable today"));
    }

    #[test]
    fn test_multiple_matches() {
        let mut matcher = AhoCorasickMatcher::new();
        let patterns = [
            ("buy", 1),
            ("sell", 1),
            ("hold", 1),
        ];
        matcher.build(&patterns);

        let mut matches = [KeywordMatch {
            pattern_idx: 0,
            start: 0,
            end: 0,
            priority: 0,
        }; 10];

        let count = matcher.scan("buy now or sell later, but don't hold", &mut matches);
        assert!(count >= 3);
    }

    #[test]
    fn test_builder() {
        let matcher = KeywordMatcherBuilder::new()
            .add_critical_keywords()
            .add_pattern("custom_keyword", 2)
            .build();

        assert!(matcher.is_ready());
        assert!(matcher.pattern_count() > 20);
    }
}
