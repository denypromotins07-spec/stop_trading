//! SOUL Trade Journaling Engine
//! 
//! Asynchronous trade journaling that logs entry/exit prices, slippage, PnL, and market regimes.
//! Writes structured trade outcomes directly to SOUL.md for Python ML backend analysis.

use std::{
    fs::{File, OpenOptions},
    io::{self, BufRead, BufReader, Write},
    path::Path,
    sync::{
        atomic::{AtomicU64, AtomicUsize, Ordering},
        Arc,
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use crossbeam_channel::{bounded, Receiver, Sender};

/// Cache-line padding constant
const CACHE_LINE_SIZE: usize = 64;

/// Maximum journal entries in memory buffer
const MAX_BUFFER_SIZE: usize = 1000;

/// Trade outcome enum
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TradeOutcome {
    Win,
    Loss,
    BreakEven,
    Pending,
}

/// Market regime at time of trade
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarketRegime {
    TrendingUp,
    TrendingDown,
    Ranging,
    Volatile,
    Unknown,
}

/// Trade journal entry
#[derive(Debug, Clone)]
pub struct TradeJournalEntry {
    /// Unique trade ID
    pub trade_id: u64,
    /// Symbol traded
    pub symbol: String,
    /// Entry price
    pub entry_price: f64,
    /// Exit price (0 if pending)
    pub exit_price: f64,
    /// Position size
    pub quantity: f64,
    /// Direction (true = long, false = short)
    pub is_long: bool,
    /// PnL in quote currency
    pub pnl: f64,
    /// Slippage in basis points
    pub slippage_bps: i32,
    /// Entry timestamp (ns)
    pub entry_timestamp_ns: u64,
    /// Exit timestamp (ns)
    pub exit_timestamp_ns: u64,
    /// Trade outcome
    pub outcome: TradeOutcome,
    /// Market regime
    pub market_regime: MarketRegime,
    /// Strategy ID that generated the signal
    pub strategy_id: String,
    /// Confidence score at entry
    pub confidence: f32,
    /// AI model version used
    pub model_version: u32,
    /// Notes/comments
    pub notes: String,
}

impl TradeJournalEntry {
    /// Create a new pending trade entry
    pub fn new(
        trade_id: u64,
        symbol: &str,
        entry_price: f64,
        quantity: f64,
        is_long: bool,
        strategy_id: &str,
        confidence: f32,
        model_version: u32,
    ) -> Self {
        let now = get_timestamp_ns();
        
        Self {
            trade_id,
            symbol: symbol.to_string(),
            entry_price,
            exit_price: 0.0,
            quantity,
            is_long,
            pnl: 0.0,
            slippage_bps: 0,
            entry_timestamp_ns: now,
            exit_timestamp_ns: 0,
            outcome: TradeOutcome::Pending,
            market_regime: MarketRegime::Unknown,
            strategy_id: strategy_id.to_string(),
            confidence,
            model_version,
            notes: String::new(),
        }
    }

    /// Close the trade with exit price
    pub fn close(&mut self, exit_price: f64, slippage_bps: i32) {
        self.exit_price = exit_price;
        self.exit_timestamp_ns = get_timestamp_ns();
        self.slippage_bps = slippage_bps;
        
        // Calculate PnL
        let price_diff = if self.is_long {
            exit_price - self.entry_price
        } else {
            self.entry_price - exit_price
        };
        
        self.pnl = price_diff * self.quantity;
        
        // Determine outcome
        if self.pnl > 0.0 {
            self.outcome = TradeOutcome::Win;
        } else if self.pnl < 0.0 {
            self.outcome = TradeOutcome::Loss;
        } else {
            self.outcome = TradeOutcome::BreakEven;
        }
    }

    /// Set market regime
    pub fn set_market_regime(&mut self, regime: MarketRegime) {
        self.market_regime = regime;
    }

    /// Add notes
    pub fn add_notes(&mut self, notes: &str) {
        self.notes.push_str(notes);
    }

    /// Serialize to JSON-like format for SOUL.md
    pub fn to_json(&self) -> String {
        format!(
            r#"{{"trade_id":{},"symbol":"{}","entry_price":{},"exit_price":{},"quantity":{},"is_long":{},"pnl":{},"slippage_bps":{},"entry_ts":{},"exit_ts":{},"outcome":"{:?}","regime":"{:?}","strategy":"{}","confidence":{:.4},"model_version":{},"notes":"{}"}}"#,
            self.trade_id,
            self.symbol,
            self.entry_price,
            self.exit_price,
            self.quantity,
            self.is_long,
            self.pnl,
            self.slippage_bps,
            self.entry_timestamp_ns,
            self.exit_timestamp_ns,
            self.outcome,
            self.market_regime,
            self.strategy_id,
            self.confidence,
            self.model_version,
            self.notes.replace('"', "\\\"")
        )
    }
}

/// Async trade journal writer
pub struct TradeJournal {
    /// Path to SOUL.md file
    soul_path: String,
    /// Entry sender
    sender: Sender<TradeJournalEntry>,
    /// Entry receiver
    receiver: Receiver<TradeJournalEntry>,
    /// Running flag
    running: Arc<AtomicU64>,
    /// Total trades logged
    total_trades: Arc<AtomicUsize>,
    /// Total PnL
    total_pnl: Arc<AtomicU64>, // Stored as fixed-point (multiply by 1e6)
}

unsafe impl Send for TradeJournal {}
unsafe impl Sync for TradeJournal {}

impl TradeJournal {
    /// Create a new trade journal
    pub fn new(soul_path: &str) -> io::Result<Self> {
        let (sender, receiver) = bounded(MAX_BUFFER_SIZE);
        
        // Ensure SOUL.md exists
        if !Path::new(soul_path).exists() {
            let mut file = File::create(soul_path)?;
            writeln!(file, "# SOUL.md - Self-Learning Trade Journal")?;
            writeln!(file, "# Format: JSON lines for ML analysis")?;
            writeln!(file, "# Generated by HFT Trading Engine")?;
            writeln!(file, "---")?;
        }

        Ok(Self {
            soul_path: soul_path.to_string(),
            sender,
            receiver,
            running: Arc::new(AtomicU64::new(1)),
            total_trades: Arc::new(AtomicUsize::new(0)),
            total_pnl: Arc::new(AtomicU64::new(0)),
        })
    }

    /// Log a new trade entry
    pub fn log_entry(&self, entry: TradeJournalEntry) -> io::Result<()> {
        self.sender.try_send(entry).map_err(|e| {
            io::Error::new(io::ErrorKind::WouldBlock, "Journal buffer full")
        })
    }

    /// Log a trade exit/update
    pub fn log_exit(&self, trade_id: u64, exit_price: f64, slippage_bps: i32) -> io::Result<()> {
        // This would typically look up the trade and update it
        // For simplicity, we assume the entry was already updated
        Ok(())
    }

    /// Start the background writer thread
    pub fn start(&self) {
        let running = self.running.clone();
        let receiver = self.receiver.clone();
        let soul_path = self.soul_path.clone();
        let total_trades = self.total_trades.clone();
        let total_pnl = self.total_pnl.clone();

        std::thread::spawn(move || {
            while running.load(Ordering::Acquire) != 0 {
                if let Ok(entry) = receiver.recv_timeout(Duration::from_millis(100)) {
                    // Write to SOUL.md
                    if let Ok(mut file) = OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(&soul_path)
                    {
                        let _ = writeln!(file, "{}", entry.to_json());
                        
                        total_trades.fetch_add(1, Ordering::Relaxed);
                        
                        // Update total PnL (fixed-point)
                        let pnl_fixed = (entry.pnl * 1_000_000.0) as i64;
                        total_pnl.fetch_update(
                            Ordering::Relaxed,
                            Ordering::Relaxed,
                            |current| {
                                let current_i64 = current as i64;
                                Some((current_i64 + pnl_fixed) as u64)
                            },
                        ).ok();
                    }
                }
            }
        });
    }

    /// Stop the journal writer
    pub fn stop(&self) {
        self.running.store(0, Ordering::Release);
    }

    /// Get total trades logged
    pub fn get_total_trades(&self) -> usize {
        self.total_trades.load(Ordering::Relaxed)
    }

    /// Get total PnL
    pub fn get_total_pnl(&self) -> f64 {
        self.total_pnl.load(Ordering::Relaxed) as f64 / 1_000_000.0
    }

    /// Flush remaining entries
    pub fn flush(&self) {
        while let Ok(entry) = self.receiver.try_recv() {
            if let Ok(mut file) = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&self.soul_path)
            {
                let _ = writeln!(file, "{}", entry.to_json());
            }
        }
    }
}

impl Drop for TradeJournal {
    fn drop(&mut self) {
        self.stop();
        self.flush();
    }
}

/// Get current timestamp in nanoseconds
fn get_timestamp_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64
}

/// Trade statistics calculator
pub struct TradeStatistics {
    wins: AtomicUsize,
    losses: AtomicUsize,
    break_evens: AtomicUsize,
    total_pnl: AtomicU64,
    max_drawdown: AtomicU64,
    peak_pnl: AtomicU64,
}

unsafe impl Send for TradeStatistics {}
unsafe impl Sync for TradeStatistics {}

impl TradeStatistics {
    pub fn new() -> Self {
        Self {
            wins: AtomicUsize::new(0),
            losses: AtomicUsize::new(0),
            break_evens: AtomicUsize::new(0),
            total_pnl: AtomicU64::new(0),
            max_drawdown: AtomicU64::new(0),
            peak_pnl: AtomicU64::new(0),
        }
    }

    pub fn record_trade(&self, pnl: f64) {
        let pnl_fixed = (pnl * 1_000_000.0) as i64;
        
        if pnl > 0.0 {
            self.wins.fetch_add(1, Ordering::Relaxed);
        } else if pnl < 0.0 {
            self.losses.fetch_add(1, Ordering::Relaxed);
        } else {
            self.break_evens.fetch_add(1, Ordering::Relaxed);
        }

        // Update total PnL
        self.total_pnl.fetch_update(
            Ordering::Relaxed,
            Ordering::Relaxed,
            |current| {
                let current_i64 = current as i64;
                Some((current_i64 + pnl_fixed) as u64)
            },
        ).ok();

        // Update peak and drawdown
        let current_pnl = self.total_pnl.load(Ordering::Relaxed) as i64;
        let peak = self.peak_pnl.load(Ordering::Relaxed) as i64;
        
        if current_pnl > peak {
            self.peak_pnl.store(current_pnl as u64, Ordering::Relaxed);
        } else {
            let drawdown = (peak - current_pnl) as u64;
            self.max_drawdown.fetch_max(drawdown, Ordering::Relaxed);
        }
    }

    pub fn get_win_rate(&self) -> f64 {
        let wins = self.wins.load(Ordering::Relaxed);
        let losses = self.losses.load(Ordering::Relaxed);
        let total = wins + losses;
        
        if total == 0 {
            return 0.0;
        }
        
        wins as f64 / total as f64
    }

    pub fn get_total_pnl(&self) -> f64 {
        self.total_pnl.load(Ordering::Relaxed) as f64 / 1_000_000.0
    }

    pub fn get_max_drawdown(&self) -> f64 {
        self.max_drawdown.load(Ordering::Relaxed) as f64 / 1_000_000.0
    }

    pub fn get_profit_factor(&self) -> f64 {
        // Simplified - would need gross profit/loss tracking
        let wins = self.wins.load(Ordering::Relaxed);
        let losses = self.losses.load(Ordering::Relaxed);
        
        if losses == 0 {
            if wins == 0 { return 0.0; }
            return f64::INFINITY;
        }
        
        wins as f64 / losses as f64
    }
}

impl Default for TradeStatistics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trade_journal_entry() {
        let mut entry = TradeJournalEntry::new(
            1,
            "BTCUSDT",
            45000.0,
            0.1,
            true,
            "momentum_v1",
            0.85,
            1,
        );
        
        assert_eq!(entry.trade_id, 1);
        assert_eq!(entry.symbol, "BTCUSDT");
        assert_eq!(entry.outcome, TradeOutcome::Pending);
        
        // Close the trade
        entry.close(45500.0, 5);
        
        assert!(entry.pnl > 0.0);
        assert_eq!(entry.outcome, TradeOutcome::Win);
        assert_eq!(entry.slippage_bps, 5);
    }

    #[test]
    fn test_trade_statistics() {
        let stats = TradeStatistics::new();
        
        stats.record_trade(100.0);
        stats.record_trade(-50.0);
        stats.record_trade(75.0);
        
        assert_eq!(stats.wins.load(Ordering::Relaxed), 2);
        assert_eq!(stats.losses.load(Ordering::Relaxed), 1);
        assert!((stats.get_total_pnl() - 125.0).abs() < 0.01);
        assert!((stats.get_win_rate() - 0.667).abs() < 0.01);
    }

    #[test]
    fn test_json_serialization() {
        let mut entry = TradeJournalEntry::new(
            42,
            "ETHUSDT",
            3000.0,
            1.0,
            false,
            "mean_revert",
            0.72,
            2,
        );
        entry.close(2950.0, 3);
        
        let json = entry.to_json();
        assert!(json.contains("\"trade_id\":42"));
        assert!(json.contains("\"symbol\":\"ETHUSDT\""));
        assert!(json.contains("\"outcome\":\"Win\""));
    }
}
