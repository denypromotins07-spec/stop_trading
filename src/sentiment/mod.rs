//! Sentiment Module Root
//! 
//! Coordinates sentiment scraping, scoring, and aggregation.
//! Pushes aggregated scores to shared memory IPC feature store.

pub mod scraper;
pub mod scorer;

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use crossbeam_channel::{bounded, Sender, Receiver};
use tokio::sync::RwLock;
use tracing::{info, debug, warn};

/// Aggregation window size (number of items)
const AGGREGATION_WINDOW: usize = 100;

/// Channel capacity for aggregated scores
const SCORE_CHANNEL_CAPACITY: usize = 500;

/// Aggregated sentiment score
#[derive(Debug, Clone)]
pub struct AggregatedScore {
    pub overall_score: f32,
    pub twitter_score: Option<f32>,
    pub reddit_score: Option<f32>,
    pub news_score: Option<f32>,
    pub volume_weighted_score: f32,
    pub trend: SentimentTrend,
    pub timestamp_ns: u64,
    pub sample_count: usize,
}

/// Sentiment trend direction
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SentimentTrend {
    Improving,
    Deteriorating,
    Stable,
    Volatile,
}

/// Sentiment manager coordinating all sentiment components
pub struct SentimentManager {
    scraper: Arc<scraper::SentimentScraper>,
    scorer: Arc<scorer::SentimentScorer>,
    
    /// Aggregated score channel
    score_tx: Sender<AggregatedScore>,
    score_rx: Receiver<AggregatedScore>,
    
    /// Rolling score buffer for trend detection
    score_history: Arc<std::sync::Mutex<Vec<f32>>>,
    
    /// Statistics
    total_processed: AtomicU64,
    average_score: AtomicU64, // Fixed-point * 1000
}

impl SentimentManager {
    /// Create a new sentiment manager
    pub fn new() -> Self {
        let (score_tx, score_rx) = bounded(SCORE_CHANNEL_CAPACITY);
        
        Self {
            scraper: Arc::new(scraper::SentimentScraper::new()),
            scorer: Arc::new(scorer::SentimentScorer::new()),
            score_tx,
            score_rx,
            score_history: Arc::new(std::sync::Mutex::new(Vec::with_capacity(AGGREGATION_WINDOW))),
            total_processed: AtomicU64::new(0),
            average_score: AtomicU64::new(0),
        }
    }
    
    /// Process a scraped item through the scoring pipeline
    pub fn process_item(&self, item: &scraper::ScrapedItem) -> Option<AggregatedScore> {
        // Score the text
        let result = self.scorer.score(&item.text);
        
        debug!(
            "Scored item from {:?}: score={:.3}, confidence={:.3}",
            item.source, result.overall_score, result.confidence
        );
        
        self.total_processed.fetch_add(1, Ordering::Relaxed);
        
        // Update rolling average (fixed-point)
        let current_avg = self.average_score.load(Ordering::Relaxed) as f32 / 1000.0;
        let new_avg = current_avg + (result.overall_score - current_avg) / 
            (self.total_processed.load(Ordering::Relaxed) as f32).min(10000.0);
        self.average_score.store((new_avg * 1000.0) as u64, Ordering::Relaxed);
        
        // Add to history for aggregation
        self.add_to_history(result.overall_score);
        
        // Check if we should emit aggregated score
        if self.should_emit_aggregate() {
            Some(self.create_aggregated_score())
        } else {
            None
        }
    }
    
    /// Get receiver for scraped items (from external scrapers)
    pub fn item_receiver(&self) -> Receiver<scraper::ScrapedItem> {
        self.scraper.item_receiver()
    }
    
    /// Get receiver for aggregated scores
    pub fn score_receiver(&self) -> Receiver<AggregatedScore> {
        self.score_rx.clone()
    }
    
    /// Submit item directly (for external scrapers)
    pub fn submit_item(&self, item: scraper::ScrapedItem) -> bool {
        self.scraper.submit_item(item)
    }
    
    /// Get current average sentiment
    pub fn get_average_sentiment(&self) -> f32 {
        self.average_score.load(Ordering::Relaxed) as f32 / 1000.0
    }
    
    /// Get total processed count
    pub fn get_total_processed(&self) -> u64 {
        self.total_processed.load(Ordering::Relaxed)
    }
    
    /// Get scrape statistics
    pub fn get_scrape_stats(&self) -> scraper::ScrapeStats {
        self.scraper.get_stats()
    }
    
    /// Start background processing loop
    pub async fn start(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        info!("Starting sentiment manager");
        
        let scraper_rx = self.item_receiver();
        let score_tx = self.score_tx.clone();
        let scorer = self.scorer.clone();
        let history = self.score_history.clone();
        let total = self.total_processed.clone();
        let avg = self.average_score.clone();
        
        tokio::spawn(async move {
            while let Ok(item) = scraper_rx.recv() {
                let result = scorer.score(&item.text);
                
                total.fetch_add(1, Ordering::Relaxed);
                
                // Update average
                let current_avg = avg.load(Ordering::Relaxed) as f32 / 1000.0;
                let count = total.load(Ordering::Relaxed) as f32;
                let new_avg = current_avg + (result.overall_score - current_avg) / count.min(10000.0);
                avg.store((new_avg * 1000.0) as u64, Ordering::Relaxed);
                
                // Add to history
                if let Ok(mut hist) = history.lock() {
                    hist.push(result.overall_score);
                    if hist.len() > AGGREGATION_WINDOW {
                        hist.remove(0);
                    }
                }
                
                debug!("Processed sentiment item: {:.3}", result.overall_score);
            }
        });
        
        Ok(())
    }
    
    /// Add score to history
    fn add_to_history(&self, score: f32) {
        if let Ok(mut history) = self.score_history.lock() {
            history.push(score);
            if history.len() > AGGREGATION_WINDOW {
                history.remove(0);
            }
        }
    }
    
    /// Check if we should emit aggregated score
    fn should_emit_aggregate(&self) -> bool {
        if let Ok(history) = self.score_history.lock() {
            history.len() >= AGGREGATION_WINDOW
        } else {
            false
        }
    }
    
    /// Create aggregated score from history
    fn create_aggregated_score(&self) -> AggregatedScore {
        let history = self.score_history.lock().unwrap();
        
        if history.is_empty() {
            return AggregatedScore {
                overall_score: 0.0,
                twitter_score: None,
                reddit_score: None,
                news_score: None,
                volume_weighted_score: 0.0,
                trend: SentimentTrend::Stable,
                timestamp_ns: get_timestamp_ns(),
                sample_count: 0,
            };
        }
        
        let overall = history.iter().sum::<f32>() / history.len() as f32;
        
        // Calculate trend from recent vs older scores
        let mid = history.len() / 2;
        let recent_avg = if mid < history.len() {
            history[mid..].iter().sum::<f32>() / (history.len() - mid) as f32
        } else {
            overall
        };
        
        let older_avg = if mid > 0 {
            history[..mid].iter().sum::<f32>() / mid as f32
        } else {
            overall
        };
        
        let trend = if (recent_avg - older_avg).abs() < 0.05 {
            SentimentTrend::Stable
        } else if recent_avg > older_avg {
            SentimentTrend::Improving
        } else {
            SentimentTrend::Deteriorating
        };
        
        AggregatedScore {
            overall_score: overall,
            twitter_score: None, // Would calculate per-source in production
            reddit_score: None,
            news_score: None,
            volume_weighted_score: overall, // Would weight by engagement in production
            trend,
            timestamp_ns: get_timestamp_ns(),
            sample_count: history.len(),
        }
    }
}

impl Default for SentimentManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert sentiment to trading signal multiplier
pub fn sentiment_to_multiplier(score: f32) -> f64 {
    // Map [-1, 1] to [0.5, 1.5]
    1.0 + (score * 0.5).clamp(-0.5, 0.5)
}

/// Get current timestamp in nanoseconds
fn get_timestamp_ns() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_manager_creation() {
        let manager = SentimentManager::new();
        assert_eq!(manager.get_total_processed(), 0);
        assert_eq!(manager.get_average_sentiment(), 0.0);
    }
    
    #[test]
    fn test_process_item() {
        let manager = SentimentManager::new();
        
        let item = manager.scraper.parse_twitter_post(
            "crypto_trader",
            "Bitcoin is going to moon! Bullish rally!",
            true,
            50000,
            100,
            50,
        );
        
        let _result = manager.process_item(&item);
        assert!(manager.get_total_processed() > 0);
    }
    
    #[test]
    fn test_sentiment_multiplier() {
        assert_eq!(sentiment_to_multiplier(0.0), 1.0);
        assert!(sentiment_to_multiplier(1.0) > 1.0);
        assert!(sentiment_to_multiplier(-1.0) < 1.0);
    }
}
