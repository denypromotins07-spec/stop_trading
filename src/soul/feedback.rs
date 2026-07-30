//! SOUL Feedback Loop Module
//! 
//! Asynchronous file watcher detecting real-time modifications to SOUL.md.
//! Parses updated adaptive weights and confidence scores injected by Python training loop.

use std::{
    fs,
    io,
    path::Path,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, RwLock,
    },
    time::{Duration, Instant},
};

use crossbeam_channel::{bounded, Receiver, Sender};
use notify::{Config, RecommendedWatcher, RecursiveMode, Watcher, EventKind};

/// Cache-line padding constant
const CACHE_LINE_SIZE: usize = 64;

/// Default watch interval in milliseconds
const DEFAULT_WATCH_INTERVAL_MS: u64 = 100;

/// Adaptive weight update from Python ML backend
#[derive(Debug, Clone)]
pub struct AdaptiveWeightUpdate {
    /// Strategy ID
    pub strategy_id: String,
    /// Updated weights
    pub weights: Vec<f32>,
    /// Confidence score for this update
    pub confidence: f32,
    /// Validation score (e.g., Sharpe ratio)
    pub validation_score: f32,
    /// Timestamp of update
    pub timestamp_ns: u64,
    /// Model version
    pub model_version: u32,
}

/// Parsed SOUL.md entry for feedback
#[derive(Debug, Clone)]
pub struct SoulEntry {
    pub trade_id: u64,
    pub symbol: String,
    pub pnl: f64,
    pub outcome: String,
    pub confidence: f32,
    pub strategy: String,
}

/// SOUL.md feedback parser and watcher
pub struct SoulFeedbackWatcher {
    /// Path to SOUL.md
    soul_path: String,
    /// Last known file size
    last_size: Arc<RwLock<u64>>,
    /// Weight updates channel
    weight_sender: Sender<AdaptiveWeightUpdate>,
    weight_receiver: Receiver<AdaptiveWeightUpdate>,
    /// Running flag
    running: Arc<AtomicBool>,
    /// Last modification time
    last_modified: Arc<AtomicU64>,
    /// Total updates received
    total_updates: Arc<AtomicU64>,
    /// Current adaptive weights per strategy
    adaptive_weights: Arc<RwLock<std::collections::HashMap<String, Vec<f32>>>>,
}

unsafe impl Send for SoulFeedbackWatcher {}
unsafe impl Sync for SoulFeedbackWatcher {}

impl SoulFeedbackWatcher {
    /// Create a new SOUL feedback watcher
    pub fn new(soul_path: &str) -> io::Result<Self> {
        let (weight_sender, weight_receiver) = bounded(1000);
        
        // Get initial file size
        let last_size = Arc::new(RwLock::new(
            fs::metadata(soul_path)
                .map(|m| m.len())
                .unwrap_or(0),
        ));

        Ok(Self {
            soul_path: soul_path.to_string(),
            last_size,
            weight_sender,
            weight_receiver,
            running: Arc::new(AtomicBool::new(false)),
            last_modified: Arc::new(AtomicU64::new(0)),
            total_updates: Arc::new(AtomicU64::new(0)),
            adaptive_weights: Arc::new(RwLock::new(std::collections::HashMap::new())),
        })
    }

    /// Start watching for SOUL.md changes
    pub fn start(&self) -> io::Result<()> {
        if self.running.load(Ordering::Acquire) {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                "Watcher already running",
            ));
        }

        self.running.store(true, Ordering::Release);

        // Set up file watcher
        let soul_path = self.soul_path.clone();
        let running = self.running.clone();
        let last_size = self.last_size.clone();
        let last_modified = self.last_modified.clone();
        let total_updates = self.total_updates.clone();
        let weight_sender = self.weight_sender.clone();
        let adaptive_weights = self.adaptive_weights.clone();

        // Use polling watcher for cross-platform compatibility
        std::thread::spawn(move || {
            let mut last_check = Instant::now();
            let check_interval = Duration::from_millis(DEFAULT_WATCH_INTERVAL_MS);

            while running.load(Ordering::Acquire) {
                if last_check.elapsed() >= check_interval {
                    if let Ok(metadata) = fs::metadata(&soul_path) {
                        let current_size = metadata.len();
                        let modified = metadata.modified()
                            .map(|t| t.duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64)
                            .unwrap_or(0);

                        if let Ok(mut size_guard) = last_size.write() {
                            if current_size > *size_guard {
                                // File has grown - parse new entries
                                if let Ok(entries) = parse_new_entries(&soul_path, *size_guard) {
                                    for entry in entries {
                                        // Check for weight update markers
                                        if let Some(update) = extract_weight_update(&entry) {
                                            let _ = weight_sender.try_send(update);
                                            total_updates.fetch_add(1, Ordering::Relaxed);
                                            
                                            // Update adaptive weights cache
                                            if let Ok(mut weights_guard) = adaptive_weights.write() {
                                                weights_guard.insert(
                                                    update.strategy_id.clone(),
                                                    update.weights.clone(),
                                                );
                                            }
                                        }
                                    }
                                }
                                *size_guard = current_size;
                            }
                        }

                        last_modified.store(modified, Ordering::Release);
                    }
                    
                    last_check = Instant::now();
                }

                std::thread::sleep(Duration::from_millis(10));
            }
        });

        Ok(())
    }

    /// Stop watching
    pub fn stop(&self) {
        self.running.store(false, Ordering::Release);
    }

    /// Receive weight updates (non-blocking)
    pub fn try_recv_weight_update(&self) -> Option<AdaptiveWeightUpdate> {
        self.weight_receiver.try_recv().ok()
    }

    /// Receive weight updates (blocking with timeout)
    pub fn recv_weight_update_timeout(&self, timeout: Duration) -> Option<AdaptiveWeightUpdate> {
        self.weight_receiver.recv_timeout(timeout).ok()
    }

    /// Get adaptive weights for a strategy
    pub fn get_adaptive_weights(&self, strategy_id: &str) -> Option<Vec<f32>> {
        self.adaptive_weights.read().ok()?.get(strategy_id).cloned()
    }

    /// Get total updates received
    pub fn get_total_updates(&self) -> u64 {
        self.total_updates.load(Ordering::Relaxed)
    }

    /// Get last modification timestamp
    pub fn get_last_modified(&self) -> u64 {
        self.last_modified.load(Ordering::Relaxed)
    }

    /// Check if watcher is running
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Acquire)
    }
}

/// Parse new entries from SOUL.md since last_position
fn parse_new_entries(path: &str, last_position: u64) -> io::Result<Vec<SoulEntry>> {
    let file = fs::File::open(path)?;
    let reader = BufReader::new(file);
    let mut entries = Vec::new();

    for (index, line) in reader.lines().enumerate() {
        if index as u64 < last_position / 80 {
            // Approximate line skip based on average line length
            continue;
        }

        if let Ok(line) = line {
            if let Some(entry) = parse_soul_entry(&line) {
                entries.push(entry);
            }
        }
    }

    Ok(entries)
}

/// Parse a single SOUL.md JSON line
fn parse_soul_entry(line: &str) -> Option<SoulEntry> {
    // Simple JSON parsing (in production, use serde_json)
    if !line.starts_with('{') {
        return None;
    }

    // Extract key fields using string manipulation
    let trade_id = extract_json_field(line, "trade_id")?.parse::<u64>().ok()?;
    let symbol = extract_json_field(line, "symbol")?.trim_matches('"').to_string();
    let pnl = extract_json_field(line, "pnl")?.parse::<f64>().ok()?;
    let outcome = extract_json_field(line, "outcome")?.trim_matches('"').to_string();
    let confidence = extract_json_field(line, "confidence")?.parse::<f32>().ok()?;
    let strategy = extract_json_field(line, "strategy")?.trim_matches('"').to_string();

    Some(SoulEntry {
        trade_id,
        symbol,
        pnl,
        outcome,
        confidence,
        strategy,
    })
}

/// Extract weight update from soul entry
fn extract_weight_update(entry: &SoulEntry) -> Option<AdaptiveWeightUpdate> {
    // In a real implementation, Python would write special weight update markers
    // For now, we simulate based on trade outcomes
    
    if entry.outcome == "Win" || entry.outcome == "Loss" {
        // Generate simulated weight adjustment based on outcome
        let adjustment_factor = if entry.outcome == "Win" { 1.0 + entry.confidence * 0.1 } else { 1.0 - entry.confidence * 0.1 };
        
        // Simulated weights (in reality, these would come from Python)
        let weights = vec![
            entry.confidence * adjustment_factor,
            (entry.pnl.abs() / 1000.0).min(1.0) as f32,
            if entry.pnl > 0.0 { 1.0 } else { -1.0 },
        ];

        Some(AdaptiveWeightUpdate {
            strategy_id: entry.strategy.clone(),
            weights,
            confidence: entry.confidence,
            validation_score: entry.confidence,
            timestamp_ns: get_timestamp_ns(),
            model_version: 1,
        })
    } else {
        None
    }
}

/// Extract a JSON field value
fn extract_json_field(json: &str, field: &str) -> Option<String> {
    let pattern = format!("\"{}\":", field);
    if let Some(start) = json.find(&pattern) {
        let value_start = start + pattern.len();
        let rest = &json[value_start..];
        
        // Find the value (handle strings and numbers)
        let rest = rest.trim_start();
        if rest.starts_with('"') {
            // String value
            if let Some(end) = rest[1..].find('"') {
                return Some(rest[1..end+1].to_string());
            }
        } else {
            // Numeric or other value
            let end = rest.find(|c| c == ',' || c == '}').unwrap_or(rest.len());
            return Some(rest[..end].trim().to_string());
        }
    }
    None
}

/// Get current timestamp in nanoseconds
fn get_timestamp_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64
}

use std::io::BufReader;
use std::time::UNIX_EPOCH;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_json_field_extraction() {
        let json = r#"{"trade_id":42,"symbol":"BTCUSDT","pnl":150.5,"outcome":"Win"}"#;
        
        assert_eq!(extract_json_field(json, "trade_id"), Some("42".to_string()));
        assert_eq!(extract_json_field(json, "symbol"), Some("BTCUSDT".to_string()));
        assert_eq!(extract_json_field(json, "pnl"), Some("150.5".to_string()));
        assert_eq!(extract_json_field(json, "outcome"), Some("Win".to_string()));
    }

    #[test]
    fn test_soul_entry_parsing() {
        let json = r#"{"trade_id":100,"symbol":"ETHUSDT","pnl":-50.0,"outcome":"Loss","confidence":0.65,"strategy":"mean_revert"}"#;
        
        let entry = parse_soul_entry(json).unwrap();
        assert_eq!(entry.trade_id, 100);
        assert_eq!(entry.symbol, "ETHUSDT");
        assert!((entry.pnl + 50.0).abs() < 0.01);
        assert_eq!(entry.outcome, "Loss");
    }

    #[test]
    fn test_weight_update_extraction() {
        let entry = SoulEntry {
            trade_id: 1,
            symbol: "BTCUSDT".to_string(),
            pnl: 200.0,
            outcome: "Win".to_string(),
            confidence: 0.85,
            strategy: "momentum_v1".to_string(),
        };

        let update = extract_weight_update(&entry).unwrap();
        assert_eq!(update.strategy_id, "momentum_v1");
        assert!(!update.weights.is_empty());
        assert!(update.confidence > 0.8);
    }

    #[test]
    fn test_feedback_watcher_creation() {
        // Create a temporary SOUL.md for testing
        let temp_path = "/tmp/test_soul.md";
        let _ = fs::write(temp_path, "# Test SOUL.md\n");
        
        let watcher = SoulFeedbackWatcher::new(temp_path).unwrap();
        assert!(!watcher.is_running());
        assert_eq!(watcher.get_total_updates(), 0);
        
        // Cleanup
        let _ = fs::remove_file(temp_path);
    }
}
