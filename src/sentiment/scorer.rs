//! Fast Lexicon-Based Sentiment Scorer
//! 
//! Pure Rust implementation of heuristic sentiment scoring.
//! Provides immediate baseline scores before heavier ML models refine data.
//! Optimized for low-latency with pre-computed word lookups.

use std::collections::HashMap;
use std::sync::Arc;

/// Maximum words to process per text (prevents DoS)
const MAX_WORDS: usize = 500;

/// Sentiment score range
pub const MIN_SENTIMENT: f32 = -1.0;
pub const MAX_SENTIMENT: f32 = 1.0;

/// Sentiment classification
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SentimentClass {
    VeryNegative,
    Negative,
    Neutral,
    Positive,
    VeryPositive,
}

impl SentimentClass {
    pub fn from_score(score: f32) -> Self {
        if score <= -0.6 {
            SentimentClass::VeryNegative
        } else if score <= -0.2 {
            SentimentClass::Negative
        } else if score <= 0.2 {
            SentimentClass::Neutral
        } else if score <= 0.6 {
            SentimentClass::Positive
        } else {
            SentimentClass::VeryPositive
        }
    }
}

/// Sentiment analysis result
#[derive(Debug, Clone)]
pub struct SentimentResult {
    pub overall_score: f32,
    pub classification: SentimentClass,
    pub confidence: f32,
    pub positive_words: usize,
    pub negative_words: usize,
    pub neutral_words: usize,
    pub intensity: f32,
}

/// Pre-built sentiment lexicon
pub struct SentimentLexicon {
    positive_words: HashMap<&'static str, f32>,
    negative_words: HashMap<&'static str, f32>,
    intensifiers: HashMap<&'static str, f32>,
    negators: Vec<&'static str>,
}

impl SentimentLexicon {
    /// Create a new sentiment lexicon with common crypto/trading terms
    pub fn new() -> Self {
        let mut positive_words = HashMap::new();
        let mut negative_words = HashMap::new();
        let mut intensifiers = HashMap::new();
        let negators = vec!["not", "no", "never", "neither", "nobody", "nothing", "nowhere"];
        
        // Positive words with scores
        let pos_terms = [
            ("bullish", 0.8), ("moon", 0.9), ("rocket", 0.8), ("gain", 0.6),
            ("profit", 0.7), ("rally", 0.7), ("breakout", 0.6), ("surge", 0.7),
            ("pump", 0.6), ("buy", 0.5), ("long", 0.4), ("hodl", 0.6),
            ("diamond", 0.5), ("hands", 0.3), ("green", 0.4), ("up", 0.3),
            ("higher", 0.4), ("growth", 0.5), ("adoption", 0.6), ("institutional", 0.4),
            ("etf", 0.5), ("approval", 0.6), ("positive", 0.5), ("good", 0.4),
            ("great", 0.6), ("excellent", 0.7), ("amazing", 0.7), ("awesome", 0.6),
            ("win", 0.6), ("success", 0.6), ("profitable", 0.7), ("bull", 0.7),
            ("halving", 0.5), ("scarce", 0.4), ("limited", 0.3), ("demand", 0.4),
        ];
        
        for (word, score) in pos_terms.iter() {
            positive_words.insert(*word, *score);
        }
        
        // Negative words with scores
        let neg_terms = [
            ("bearish", -0.8), ("crash", -0.9), ("dump", -0.7), ("loss", -0.7),
            ("sell", -0.5), ("short", -0.4), ("red", -0.4), ("down", -0.3),
            ("lower", -0.4), ("drop", -0.5), ("plunge", -0.7), ("collapse", -0.8),
            ("panic", -0.7), ("fud", -0.6), ("scam", -0.9), ("rug", -0.9),
            ("hack", -0.8), ("exploit", -0.7), ("negative", -0.5), ("bad", -0.4),
            ("terrible", -0.7), ("horrible", -0.8), ("disaster", -0.8), ("catastrophe", -0.9),
            ("bear", -0.7), ("recession", -0.6), ("inflation", -0.4), ("rate", -0.2),
            ("hike", -0.3), ("fed", -0.2), ("sec", -0.3), ("lawsuit", -0.6),
            ("ban", -0.7), ("regulation", -0.4), ("restrict", -0.5), ("forbidden", -0.6),
        ];
        
        for (word, score) in neg_terms.iter() {
            negative_words.insert(*word, *score);
        }
        
        // Intensifiers (multiply following word score)
        let intensifier_terms = [
            ("very", 1.5), ("extremely", 2.0), ("highly", 1.4), ("really", 1.3),
            ("super", 1.5), ("incredibly", 1.8), ("massively", 1.7), ("huge", 1.4),
            ("massive", 1.4), ("big", 1.2), ("major", 1.3), ("significant", 1.3),
        ];
        
        for (word, factor) in intensifier_terms.iter() {
            intensifiers.insert(*word, *factor);
        }
        
        Self {
            positive_words,
            negative_words,
            intensifiers,
            negators,
        }
    }
    
    /// Get score for a word (positive)
    fn get_positive_score(&self, word: &str) -> Option<f32> {
        self.positive_words.get(word).copied()
    }
    
    /// Get score for a word (negative)
    fn get_negative_score(&self, word: &str) -> Option<f32> {
        self.negative_words.get(word).copied()
    }
    
    /// Get intensifier factor
    fn get_intensifier(&self, word: &str) -> Option<f32> {
        self.intensifiers.get(word).copied()
    }
    
    /// Check if word is a negator
    fn is_negator(&self, word: &str) -> bool {
        self.negators.contains(&word)
    }
}

/// Fast sentiment scorer
pub struct SentimentScorer {
    lexicon: Arc<SentimentLexicon>,
}

impl SentimentScorer {
    /// Create a new sentiment scorer
    pub fn new() -> Self {
        Self {
            lexicon: Arc::new(SentimentLexicon::new()),
        }
    }
    
    /// Score a text string
    pub fn score(&self, text: &str) -> SentimentResult {
        let words: Vec<&str> = text
            .to_lowercase()
            .split_whitespace()
            .take(MAX_WORDS)
            .collect();
        
        let mut total_score: f32 = 0.0;
        let mut positive_count = 0;
        let mut negative_count = 0;
        let mut neutral_count = 0;
        let mut intensity_sum: f32 = 0.0;
        let mut word_count = 0;
        
        let mut prev_intensifier: Option<f32> = None;
        let mut prev_negator = false;
        
        for word in words {
            // Clean word
            let clean_word = clean_word(word);
            if clean_word.is_empty() {
                continue;
            }
            
            word_count += 1;
            let mut word_score = 0.0;
            
            // Check for intensifier
            if let Some(factor) = self.lexicon.get_intensifier(&clean_word) {
                prev_intensifier = Some(factor);
                neutral_count += 1;
                continue;
            }
            
            // Check for negator
            if self.lexicon.is_negator(&clean_word) {
                prev_negator = true;
                neutral_count += 1;
                continue;
            }
            
            // Check positive words
            if let Some(score) = self.lexicon.get_positive_score(&clean_word) {
                let mut adjusted_score = score;
                
                // Apply intensifier
                if let Some(factor) = prev_intensifier.take() {
                    adjusted_score *= factor;
                    intensity_sum += factor;
                }
                
                // Apply negation
                if prev_negator {
                    adjusted_score = -adjusted_score * 0.5; // Negation reduces but doesn't fully invert
                    prev_negator = false;
                }
                
                word_score = adjusted_score;
                positive_count += 1;
            }
            // Check negative words
            else if let Some(score) = self.lexicon.get_negative_score(&clean_word) {
                let mut adjusted_score = score;
                
                // Apply intensifier
                if let Some(factor) = prev_intensifier.take() {
                    adjusted_score *= factor;
                    intensity_sum += factor;
                }
                
                // Apply negation
                if prev_negator {
                    adjusted_score = -adjusted_score * 0.5;
                    prev_negator = false;
                }
                
                word_score = adjusted_score;
                negative_count += 1;
            } else {
                neutral_count += 1;
            }
            
            total_score += word_score;
        }
        
        // Normalize score
        let normalized_score = if word_count > 0 {
            (total_score / word_count as f32).clamp(MIN_SENTIMENT, MAX_SENTIMENT)
        } else {
            0.0
        };
        
        // Calculate confidence based on word coverage
        let sentiment_words = positive_count + negative_count;
        let confidence = if word_count > 0 {
            (sentiment_words as f32 / word_count as f32).min(1.0)
        } else {
            0.0
        };
        
        // Calculate average intensity
        let avg_intensity = if sentiment_words > 0 && intensity_sum > 0.0 {
            intensity_sum / sentiment_words as f32
        } else {
            1.0
        };
        
        SentimentResult {
            overall_score: normalized_score,
            classification: SentimentClass::from_score(normalized_score),
            confidence,
            positive_words: positive_count,
            negative_words: negative_count,
            neutral_words: neutral_count,
            intensity: avg_intensity,
        }
    }
    
    /// Score multiple texts and aggregate
    pub fn score_batch(&self, texts: &[&str]) -> SentimentResult {
        let mut total_score = 0.0;
        let mut total_positive = 0;
        let mut total_negative = 0;
        let mut total_neutral = 0;
        let mut max_intensity = 0.0;
        
        for text in texts {
            let result = self.score(text);
            total_score += result.overall_score;
            total_positive += result.positive_words;
            total_negative += result.negative_words;
            total_neutral += result.neutral_words;
            max_intensity = max_intensity.max(result.intensity);
        }
        
        let count = texts.len() as f32;
        let avg_score = if count > 0.0 {
            total_score / count
        } else {
            0.0
        };
        
        SentimentResult {
            overall_score: avg_score.clamp(MIN_SENTIMENT, MAX_SENTIMENT),
            classification: SentimentClass::from_score(avg_score),
            confidence: (total_positive + total_negative) as f32 
                / (total_positive + total_negative + total_neutral) as f32,
            positive_words: total_positive,
            negative_words: total_negative,
            neutral_words: total_neutral,
            intensity: max_intensity,
        }
    }
}

impl Default for SentimentScorer {
    fn default() -> Self {
        Self::new()
    }
}

/// Clean a word (remove punctuation, lowercase)
fn clean_word(word: &str) -> String {
    word.chars()
        .filter(|c| c.is_alphabetic())
        .collect::<String>()
        .to_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_scorer_creation() {
        let scorer = SentimentScorer::new();
        let result = scorer.score("bitcoin is going to the moon");
        assert!(result.overall_score > 0.0);
    }
    
    #[test]
    fn test_positive_sentiment() {
        let scorer = SentimentScorer::new();
        let result = scorer.score("bullish rally breakout profit gain");
        assert_eq!(result.classification, SentimentClass::Positive);
        assert!(result.overall_score > 0.2);
    }
    
    #[test]
    fn test_negative_sentiment() {
        let scorer = SentimentScorer::new();
        let result = scorer.score("bearish crash dump loss sell");
        assert_eq!(result.classification, SentimentClass::Negative);
        assert!(result.overall_score < -0.2);
    }
    
    #[test]
    fn test_intensifier() {
        let scorer = SentimentScorer::new();
        let result1 = scorer.score("good");
        let result2 = scorer.score("very good");
        assert!(result2.overall_score > result1.overall_score);
    }
    
    #[test]
    fn test_negation() {
        let scorer = SentimentScorer::new();
        let result = scorer.score("not good");
        assert!(result.overall_score < 0.0);
    }
    
    #[test]
    fn test_sentiment_classification() {
        assert_eq!(SentimentClass::from_score(-0.8), SentimentClass::VeryNegative);
        assert_eq!(SentimentClass::from_score(-0.4), SentimentClass::Negative);
        assert_eq!(SentimentClass::from_score(0.0), SentimentClass::Neutral);
        assert_eq!(SentimentClass::from_score(0.4), SentimentClass::Positive);
        assert_eq!(SentimentClass::from_score(0.8), SentimentClass::VeryPositive);
    }
}
