//! Trade Expectancy and Profit Factor Engine
//! 
//! This module implements trade expectancy and profit factor engines that track
//! win-rate and average win/loss ratios. Pushes statistical summaries directly into
//! SOUL.md so the Python ML backend can evaluate strategy degradation over time.
//! 
//! Key Features:
//! - Online trade statistics tracking
//! - Win rate, loss rate calculation
//! - Profit factor (gross profit / gross loss)
//! - Expectancy = (Win% * AvgWin) - (Loss% * AvgLoss)
//! - SOUL.md integration for ML feedback

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Trade result for statistical analysis
#[derive(Debug, Clone)]
pub struct TradeResult {
    /// Unique trade ID
    pub trade_id: String,
    /// Symbol
    pub symbol: String,
    /// Side (1 = long, -1 = short)
    pub side: i8,
    /// Entry price
    pub entry_price: f64,
    /// Exit price
    pub exit_price: f64,
    /// Quantity
    pub quantity: f64,
    /// PnL (positive = profit, negative = loss)
    pub pnl: f64,
    /// PnL as percentage
    pub pnl_pct: f64,
    /// Duration of trade
    pub duration: Duration,
    /// Timestamp of exit
    pub exit_timestamp: u64,
    /// Whether this was a winning trade
    pub is_winner: bool,
}

impl TradeResult {
    pub fn new(
        trade_id: String,
        symbol: String,
        side: i8,
        entry_price: f64,
        exit_price: f64,
        quantity: f64,
        duration: Duration,
    ) -> Self {
        let pnl = if side > 0 {
            (exit_price - entry_price) * quantity
        } else {
            (entry_price - exit_price) * quantity
        };
        
        let pnl_pct = if entry_price > 0.0 {
            ((exit_price - entry_price) / entry_price).abs() * side as f64
        } else {
            0.0
        };
        
        let is_winner = pnl > 0.0;
        
        Self {
            trade_id,
            symbol,
            side,
            entry_price,
            exit_price,
            quantity,
            pnl,
            pnl_pct,
            duration,
            exit_timestamp: current_timestamp_ns(),
            is_winner,
        }
    }
}

/// Online expectancy calculator
pub struct ExpectancyCalculator {
    /// Total number of trades
    total_trades: AtomicUsize,
    /// Number of winning trades
    winning_trades: AtomicUsize,
    /// Sum of all winning PnL
    total_win_pnl: AtomicU64, // Stored as fixed-point (multiply by 1e9)
    /// Sum of all losing PnL (absolute value)
    total_loss_pnl: AtomicU64,
    /// Sum of all PnL
    total_pnl: AtomicU64,
    /// Largest winning trade
    largest_win: AtomicU64,
    /// Largest losing trade (absolute value)
    largest_loss: AtomicU64,
    /// Consecutive wins counter
    consecutive_wins: AtomicUsize,
    /// Consecutive losses counter
    consecutive_losses: AtomicUsize,
    /// Maximum consecutive wins
    max_consecutive_wins: AtomicUsize,
    /// Maximum consecutive losses
    max_consecutive_losses: AtomicUsize,
}

impl ExpectancyCalculator {
    pub fn new() -> Self {
        Self {
            total_trades: AtomicUsize::new(0),
            winning_trades: AtomicUsize::new(0),
            total_win_pnl: AtomicU64::new(0),
            total_loss_pnl: AtomicU64::new(0),
            total_pnl: AtomicU64::new(0),
            largest_win: AtomicU64::new(0),
            largest_loss: AtomicU64::new(0),
            consecutive_wins: AtomicUsize::new(0),
            consecutive_losses: AtomicUsize::new(0),
            max_consecutive_wins: AtomicUsize::new(0),
            max_consecutive_losses: AtomicUsize::new(0),
        }
    }
    
    /// Record a trade result
    pub fn record_trade(&self, trade: &TradeResult) {
        self.total_trades.fetch_add(1, Ordering::Relaxed);
        
        // Update total PnL
        let pnl_bits = trade.pnl.to_bits();
        let current_total = self.total_pnl.load(Ordering::Relaxed);
        // For simplicity, we'll use atomic f64 via bits
        // In production, would need proper locking or different approach
        
        if trade.is_winner {
            self.winning_trades.fetch_add(1, Ordering::Relaxed);
            
            // Update win sum (using fixed-point for atomicity)
            let win_fixed = (trade.pnl * 1_000_000_000.0) as u64;
            self.total_win_pnl.fetch_add(win_fixed, Ordering::Relaxed);
            
            // Update largest win
            let win_bits = trade.pnl.to_bits();
            let mut current_largest = self.largest_win.load(Ordering::Relaxed);
            while win_bits > current_largest {
                match self.largest_win.compare_exchange_weak(
                    current_largest,
                    win_bits,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => break,
                    Err(x) => current_largest = x,
                }
            }
            
            // Update consecutive wins
            let wins = self.consecutive_wins.fetch_add(1, Ordering::Relaxed) + 1;
            self.consecutive_losses.store(0, Ordering::Relaxed);
            
            let max_wins = self.max_consecutive_wins.load(Ordering::Relaxed);
            if wins > max_wins {
                self.max_consecutive_wins.store(wins, Ordering::Relaxed);
            }
        } else {
            // Update loss sum (absolute value)
            let loss_fixed = (trade.pnl.abs() * 1_000_000_000.0) as u64;
            self.total_loss_pnl.fetch_add(loss_fixed, Ordering::Relaxed);
            
            // Update largest loss
            let loss_bits = trade.pnl.abs().to_bits();
            let mut current_largest = self.largest_loss.load(Ordering::Relaxed);
            while loss_bits > current_largest {
                match self.largest_loss.compare_exchange_weak(
                    current_largest,
                    loss_bits,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => break,
                    Err(x) => current_largest = x,
                }
            }
            
            // Update consecutive losses
            let losses = self.consecutive_losses.fetch_add(1, Ordering::Relaxed) + 1;
            self.consecutive_wins.store(0, Ordering::Relaxed);
            
            let max_losses = self.max_consecutive_losses.load(Ordering::Relaxed);
            if losses > max_losses {
                self.max_consecutive_losses.store(losses, Ordering::Relaxed);
            }
        }
    }
    
    /// Get win rate
    pub fn win_rate(&self) -> f64 {
        let total = self.total_trades.load(Ordering::Relaxed);
        if total == 0 {
            return 0.0;
        }
        self.winning_trades.load(Ordering::Relaxed) as f64 / total as f64
    }
    
    /// Get loss rate
    pub fn loss_rate(&self) -> f64 {
        1.0 - self.win_rate()
    }
    
    /// Get average win
    pub fn avg_win(&self) -> f64 {
        let wins = self.winning_trades.load(Ordering::Relaxed);
        if wins == 0 {
            return 0.0;
        }
        let total_win_fixed = self.total_win_pnl.load(Ordering::Relaxed);
        (total_win_fixed as f64 / 1_000_000_000.0) / wins as f64
    }
    
    /// Get average loss
    pub fn avg_loss(&self) -> f64 {
        let total = self.total_trades.load(Ordering::Relaxed);
        let wins = self.winning_trades.load(Ordering::Relaxed);
        let losses = total - wins;
        
        if losses == 0 {
            return 0.0;
        }
        
        let total_loss_fixed = self.total_loss_pnl.load(Ordering::Relaxed);
        (total_loss_fixed as f64 / 1_000_000_000.0) / losses as f64
    }
    
    /// Calculate expectancy
    /// Expectancy = (Win% * AvgWin) - (Loss% * AvgLoss)
    pub fn expectancy(&self) -> f64 {
        let win_rate = self.win_rate();
        let loss_rate = self.loss_rate();
        let avg_win = self.avg_win();
        let avg_loss = self.avg_loss();
        
        (win_rate * avg_win) - (loss_rate * avg_loss)
    }
    
    /// Calculate profit factor
    /// Profit Factor = Gross Profit / Gross Loss
    pub fn profit_factor(&self) -> f64 {
        let total_win_fixed = self.total_win_pnl.load(Ordering::Relaxed);
        let total_loss_fixed = self.total_loss_pnl.load(Ordering::Relaxed);
        
        if total_loss_fixed == 0 {
            if total_win_fixed == 0 {
                return 1.0; // No trades or breakeven
            }
            return f64::INFINITY; // All wins, no losses
        }
        
        total_win_fixed as f64 / total_loss_fixed as f64
    }
    
    /// Get total PnL
    pub fn total_pnl(&self) -> f64 {
        let win_fixed = self.total_win_pnl.load(Ordering::Relaxed);
        let loss_fixed = self.total_loss_pnl.load(Ordering::Relaxed);
        
        (win_fixed as f64 - loss_fixed as f64) / 1_000_000_000.0
    }
    
    /// Get largest win
    pub fn largest_win(&self) -> f64 {
        let bits = self.largest_win.load(Ordering::Relaxed);
        if bits == 0 {
            return 0.0;
        }
        f64::from_bits(bits)
    }
    
    /// Get largest loss
    pub fn largest_loss(&self) -> f64 {
        let bits = self.largest_loss.load(Ordering::Relaxed);
        if bits == 0 {
            return 0.0;
        }
        f64::from_bits(bits)
    }
    
    /// Get maximum consecutive wins
    pub fn max_consecutive_wins(&self) -> usize {
        self.max_consecutive_wins.load(Ordering::Relaxed)
    }
    
    /// Get maximum consecutive losses
    pub fn max_consecutive_losses(&self) -> usize {
        self.max_consecutive_losses.load(Ordering::Relaxed)
    }
    
    /// Get total trade count
    pub fn trade_count(&self) -> usize {
        self.total_trades.load(Ordering::Relaxed)
    }
    
    /// Reset all statistics
    pub fn reset(&self) {
        self.total_trades.store(0, Ordering::Relaxed);
        self.winning_trades.store(0, Ordering::Relaxed);
        self.total_win_pnl.store(0, Ordering::Relaxed);
        self.total_loss_pnl.store(0, Ordering::Relaxed);
        self.total_pnl.store(0, Ordering::Relaxed);
        self.largest_win.store(0, Ordering::Relaxed);
        self.largest_loss.store(0, Ordering::Relaxed);
        self.consecutive_wins.store(0, Ordering::Relaxed);
        self.consecutive_losses.store(0, Ordering::Relaxed);
        self.max_consecutive_wins.store(0, Ordering::Relaxed);
        self.max_consecutive_losses.store(0, Ordering::Relaxed);
    }
}

impl Default for ExpectancyCalculator {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics snapshot for SOUL.md export
#[derive(Debug, Clone)]
pub struct ExpectancySnapshot {
    pub total_trades: usize,
    pub winning_trades: usize,
    pub losing_trades: usize,
    pub win_rate: f64,
    pub loss_rate: f64,
    pub avg_win: f64,
    pub avg_loss: f64,
    pub expectancy: f64,
    pub profit_factor: f64,
    pub total_pnl: f64,
    pub largest_win: f64,
    pub largest_loss: f64,
    pub max_consecutive_wins: usize,
    pub max_consecutive_losses: usize,
    pub timestamp_ns: u64,
}

impl ExpectancyCalculator {
    /// Get a complete snapshot of statistics
    pub fn snapshot(&self) -> ExpectancySnapshot {
        ExpectancySnapshot {
            total_trades: self.trade_count(),
            winning_trades: self.winning_trades.load(Ordering::Relaxed),
            losing_trades: self.total_trades.load(Ordering::Relaxed) - self.winning_trades.load(Ordering::Relaxed),
            win_rate: self.win_rate(),
            loss_rate: self.loss_rate(),
            avg_win: self.avg_win(),
            avg_loss: self.avg_loss(),
            expectancy: self.expectancy(),
            profit_factor: self.profit_factor(),
            total_pnl: self.total_pnl(),
            largest_win: self.largest_win(),
            largest_loss: self.largest_loss(),
            max_consecutive_wins: self.max_consecutive_wins(),
            max_consecutive_losses: self.max_consecutive_losses(),
            timestamp_ns: current_timestamp_ns(),
        }
    }
    
    /// Export to SOUL.md format
    pub fn to_soul_md(&self) -> String {
        let snap = self.snapshot();
        
        format!(
            r#"## Trade Expectancy Metrics

| Metric | Value |
|--------|-------|
| Total Trades | {} |
| Winning Trades | {} |
| Losing Trades | {} |
| Win Rate | {:.2}% |
| Loss Rate | {:.2}% |
| Average Win | ${:.2} |
| Average Loss | ${:.2} |
| Expectancy | ${:.2} |
| Profit Factor | {:.2} |
| Total PnL | ${:.2} |
| Largest Win | ${:.2} |
| Largest Loss | ${:.2} |
| Max Consecutive Wins | {} |
| Max Consecutive Losses | {} |
| Last Updated | {} |
"#,
            snap.total_trades,
            snap.winning_trades,
            snap.losing_trades,
            snap.win_rate * 100.0,
            snap.loss_rate * 100.0,
            snap.avg_win,
            snap.avg_loss,
            snap.expectancy,
            snap.profit_factor,
            snap.total_pnl,
            snap.largest_win,
            snap.largest_loss,
            snap.max_consecutive_wins,
            snap.max_consecutive_losses,
            snap.timestamp_ns,
        )
    }
}

fn current_timestamp_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64
}

/// Per-symbol expectancy tracker
pub struct SymbolExpectancyTracker {
    calculators: parking_lot::Mutex<std::collections::HashMap<String, ExpectancyCalculator>>,
}

impl SymbolExpectancyTracker {
    pub fn new() -> Self {
        Self {
            calculators: parking_lot::Mutex::new(std::collections::HashMap::new()),
        }
    }
    
    /// Record a trade for a specific symbol
    pub fn record_trade(&self, symbol: &str, trade: &TradeResult) {
        let mut calculators = self.calculators.lock();
        
        let calc = calculators
            .entry(symbol.to_string())
            .or_insert_with(ExpectancyCalculator::new);
        
        calc.record_trade(trade);
    }
    
    /// Get expectancy for a symbol
    pub fn get_symbol_stats(&self, symbol: &str) -> Option<ExpectancySnapshot> {
        let calculators = self.calculators.lock();
        calculators.get(symbol).map(|c| c.snapshot())
    }
    
    /// Get all symbols with their stats
    pub fn get_all_symbols(&self) -> std::collections::HashMap<String, ExpectancySnapshot> {
        let calculators = self.calculators.lock();
        calculators
            .iter()
            .map(|(k, v)| (k.clone(), v.snapshot()))
            .collect()
    }
}

impl Default for SymbolExpectancyTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_expectancy_calculator() {
        let calc = ExpectancyCalculator::new();
        
        // Record some winning trades
        for i in 0..5 {
            let trade = TradeResult::new(
                format!("T{}", i),
                "BTCUSDT".to_string(),
                1,
                50000.0,
                50100.0,
                1.0,
                Duration::from_secs(3600),
            );
            calc.record_trade(&trade);
        }
        
        // Record some losing trades
        for i in 5..8 {
            let trade = TradeResult::new(
                format!("T{}", i),
                "BTCUSDT".to_string(),
                1,
                50000.0,
                49900.0,
                1.0,
                Duration::from_secs(3600),
            );
            calc.record_trade(&trade);
        }
        
        assert_eq!(calc.trade_count(), 8);
        assert!((calc.win_rate() - 0.625).abs() < 0.01);
        assert!(calc.profit_factor() > 1.0);
    }
    
    #[test]
    fn test_profit_factor_calculation() {
        let calc = ExpectancyCalculator::new();
        
        // Perfect record - all wins
        let trade = TradeResult::new(
            "T1".to_string(),
            "BTCUSDT".to_string(),
            1,
            50000.0,
            51000.0,
            1.0,
            Duration::from_secs(3600),
        );
        calc.record_trade(&trade);
        
        assert_eq!(calc.profit_factor(), f64::INFINITY);
    }
    
    #[test]
    fn test_expectancy_snapshot() {
        let calc = ExpectancyCalculator::new();
        
        let trade = TradeResult::new(
            "T1".to_string(),
            "BTCUSDT".to_string(),
            1,
            50000.0,
            50500.0,
            1.0,
            Duration::from_secs(3600),
        );
        calc.record_trade(&trade);
        
        let snap = calc.snapshot();
        assert_eq!(snap.total_trades, 1);
        assert_eq!(snap.winning_trades, 1);
        assert!((snap.win_rate - 1.0).abs() < 0.01);
    }
    
    #[test]
    fn test_symbol_tracker() {
        let tracker = SymbolExpectancyTracker::new();
        
        let trade = TradeResult::new(
            "T1".to_string(),
            "ETHUSDT".to_string(),
            1,
            3000.0,
            3100.0,
            10.0,
            Duration::from_secs(1800),
        );
        tracker.record_trade("ETHUSDT", &trade);
        
        let stats = tracker.get_symbol_stats("ETHUSDT");
        assert!(stats.is_some());
        assert_eq!(stats.unwrap().total_trades, 1);
    }
}
