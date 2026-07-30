//! SOUL Module Root
//! 
//! Orchestrates the continuous feedback loop between Rust execution and Python learning.
//! Re-exports journal and feedback components.

pub mod journal;
pub mod feedback;

use std::{
    io,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, RwLock,
    },
    time::Duration,
};

use crate::soul::journal::{TradeJournal, TradeJournalEntry, TradeStatistics};
use crate::soul::feedback::{SoulFeedbackWatcher, AdaptiveWeightUpdate};

/// SOUL system manager coordinating the self-learning loop
pub struct SoulSystem {
    /// Trade journal for logging
    journal: Arc<TradeJournal>,
    /// Feedback watcher for ML updates
    feedback_watcher: Arc<SoulFeedbackWatcher>,
    /// Trade statistics
    statistics: Arc<TradeStatistics>,
    /// Running flag
    running: Arc<AtomicBool>,
    /// Total cycles completed
    cycle_count: Arc<AtomicU64>,
    /// Current adaptive weights per strategy
    adaptive_weights: Arc<RwLock<std::collections::HashMap<String, Vec<f32>>>>,
}

unsafe impl Send for SoulSystem {}
unsafe impl Sync for SoulSystem {}

impl SoulSystem {
    /// Create a new SOUL system
    pub fn new(soul_path: &str) -> io::Result<Self> {
        let journal = Arc::new(TradeJournal::new(soul_path)?);
        let feedback_watcher = Arc::new(SoulFeedbackWatcher::new(soul_path)?);
        
        Ok(Self {
            journal,
            feedback_watcher,
            statistics: Arc::new(TradeStatistics::new()),
            running: Arc::new(AtomicBool::new(false)),
            cycle_count: Arc::new(AtomicU64::new(0)),
            adaptive_weights: Arc::new(RwLock::new(std::collections::HashMap::new())),
        })
    }

    /// Start the SOUL system
    pub fn start(&self) -> io::Result<()> {
        if self.running.load(Ordering::Acquire) {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                "SOUL system already running",
            ));
        }

        // Start journal writer
        self.journal.start();

        // Start feedback watcher
        self.feedback_watcher.start()?;

        // Start feedback processing loop
        let running = self.running.clone();
        let feedback = self.feedback_watcher.clone();
        let weights = self.adaptive_weights.clone();
        let stats = self.statistics.clone();
        let cycle_count = self.cycle_count.clone();

        std::thread::spawn(move || {
            while running.load(Ordering::Acquire) {
                // Check for weight updates
                if let Some(update) = feedback.try_recv_weight_update() {
                    // Update adaptive weights
                    if let Ok(mut weights_guard) = weights.write() {
                        weights_guard.insert(update.strategy_id.clone(), update.weights.clone());
                    }
                    
                    // Record in statistics
                    // (In reality, would use the actual PnL from the trade)
                    cycle_count.fetch_add(1, Ordering::Relaxed);
                }

                std::thread::sleep(Duration::from_millis(10));
            }
        });

        self.running.store(true, Ordering::Release);
        Ok(())
    }

    /// Stop the SOUL system
    pub fn stop(&self) {
        self.running.store(false, Ordering::Release);
        self.journal.stop();
        self.feedback_watcher.stop();
    }

    /// Log a trade entry
    pub fn log_trade(&self, entry: TradeJournalEntry) -> io::Result<()> {
        // Record statistics
        if entry.outcome != journal::TradeOutcome::Pending {
            self.statistics.record_trade(entry.pnl);
        }
        
        self.journal.log_entry(entry)
    }

    /// Get adaptive weights for a strategy
    pub fn get_adaptive_weights(&self, strategy_id: &str) -> Option<Vec<f32>> {
        // First check local cache
        if let Some(weights) = self.adaptive_weights.read().ok()?.get(strategy_id).cloned() {
            return Some(weights);
        }
        
        // Then check feedback watcher
        self.feedback_watcher.get_adaptive_weights(strategy_id)
    }

    /// Get trade statistics
    pub fn get_statistics(&self) -> &TradeStatistics {
        &self.statistics
    }

    /// Get total trades logged
    pub fn get_total_trades(&self) -> usize {
        self.journal.get_total_trades()
    }

    /// Get total PnL
    pub fn get_total_pnl(&self) -> f64 {
        self.journal.get_total_pnl()
    }

    /// Get cycle count
    pub fn get_cycle_count(&self) -> u64 {
        self.cycle_count.load(Ordering::Relaxed)
    }

    /// Check if system is running
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Acquire)
    }

    /// Flush journal entries
    pub fn flush_journal(&self) {
        self.journal.flush();
    }
}

/// Builder for creating SoulSystem with custom configuration
pub struct SoulSystemBuilder {
    soul_path: String,
}

impl SoulSystemBuilder {
    pub fn new() -> Self {
        Self {
            soul_path: "SOUL.md".to_string(),
        }
    }

    pub fn with_soul_path(mut self, path: &str) -> Self {
        self.soul_path = path.to_string();
        self
    }

    pub fn build(self) -> io::Result<SoulSystem> {
        SoulSystem::new(&self.soul_path)
    }
}

impl Default for SoulSystemBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::soul::journal::{TradeJournalEntry, MarketRegime};

    #[test]
    fn test_soul_system_creation() {
        let temp_path = "/tmp/test_soul_system.md";
        let _ = std::fs::remove_file(temp_path); // Clean up if exists
        
        let system = SoulSystem::new(temp_path).unwrap();
        assert!(!system.is_running());
        assert_eq!(system.get_total_trades(), 0);
        assert_eq!(system.get_cycle_count(), 0);
        
        // Cleanup
        let _ = std::fs::remove_file(temp_path);
    }

    #[test]
    fn test_soul_system_lifecycle() {
        let temp_path = "/tmp/test_soul_lifecycle.md";
        let _ = std::fs::remove_file(temp_path);
        
        let system = SoulSystem::new(temp_path).unwrap();
        
        system.start().unwrap();
        assert!(system.is_running());
        
        // Log a test trade
        let entry = TradeJournalEntry::new(
            1,
            "BTCUSDT",
            45000.0,
            0.1,
            true,
            "test_strategy",
            0.75,
            1,
        );
        
        let result = system.log_trade(entry);
        assert!(result.is_ok());
        
        system.stop();
        assert!(!system.is_running());
        
        // Cleanup
        let _ = std::fs::remove_file(temp_path);
    }

    #[test]
    fn test_builder_pattern() {
        let temp_path = "/tmp/test_builder.md";
        let _ = std::fs::remove_file(temp_path);
        
        let system = SoulSystemBuilder::new()
            .with_soul_path(temp_path)
            .build()
            .unwrap();
        
        assert!(system.is_running() == false);
        
        // Cleanup
        let _ = std::fs::remove_file(temp_path);
    }

    #[test]
    fn test_statistics_integration() {
        let temp_path = "/tmp/test_stats.md";
        let _ = std::fs::remove_file(temp_path);
        
        let system = SoulSystem::new(temp_path).unwrap();
        
        // Record some trades through the system
        let mut win_entry = TradeJournalEntry::new(
            1,
            "ETHUSDT",
            3000.0,
            1.0,
            true,
            "momentum",
            0.8,
            1,
        );
        win_entry.close(3100.0, 2);
        
        let mut loss_entry = TradeJournalEntry::new(
            2,
            "ETHUSDT",
            3000.0,
            1.0,
            false,
            "mean_revert",
            0.6,
            1,
        );
        loss_entry.close(3050.0, 3);
        
        let _ = system.log_trade(win_entry);
        let _ = system.log_trade(loss_entry);
        
        let stats = system.get_statistics();
        assert_eq!(stats.wins.load(Ordering::Relaxed), 1);
        assert_eq!(stats.losses.load(Ordering::Relaxed), 1);
        
        // Cleanup
        let _ = std::fs::remove_file(temp_path);
    }
}
