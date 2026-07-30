//! REST API Rate Limiter using Token Bucket Algorithm
//! 
//! Implements a strict, thread-safe Token Bucket algorithm to track and enforce 
//! REST API rate limits (e.g., 1200 req/min for Binance).

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::time::{Duration, Instant};
use anyhow::{Context, Result};

/// Token bucket configuration
#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    /// Maximum tokens (burst capacity)
    pub max_tokens: u64,
    /// Tokens added per second
    pub refill_rate: f64,
    /// Minimum interval between requests in milliseconds
    pub min_interval_ms: u64,
}

impl RateLimitConfig {
    /// Create config for Binance standard rate limit (1200 req/min = 20 req/sec)
    #[inline]
    pub fn binance_standard() -> Self {
        RateLimitConfig {
            max_tokens: 1200,           // Burst capacity
            refill_rate: 20.0,          // 20 requests per second
            min_interval_ms: 50,        // Minimum 50ms between requests
        }
    }

    /// Create config for Binance order rate limit (varies by account)
    #[inline]
    pub fn binance_orders(max_orders_per_second: u64) -> Self {
        RateLimitConfig {
            max_tokens: max_orders_per_second * 2,
            refill_rate: max_orders_per_second as f64,
            min_interval_ms: 1000 / max_orders_per_second,
        }
    }

    /// Create custom config
    #[inline]
    pub fn new(max_tokens: u64, refill_rate: f64, min_interval_ms: u64) -> Self {
        RateLimitConfig {
            max_tokens,
            refill_rate,
            min_interval_ms,
        }
    }
}

/// Thread-safe token bucket rate limiter
pub struct TokenBucket {
    /// Current token count (scaled by 1000 for precision)
    tokens: AtomicU64,
    /// Last refill timestamp
    last_refill: AtomicU64,
    /// Configuration
    config: RateLimitConfig,
    /// Whether limiter is enabled
    enabled: AtomicBool,
    /// Total requests made
    total_requests: AtomicU64,
    /// Total rejections
    total_rejections: AtomicU64,
}

impl TokenBucket {
    /// Create a new token bucket
    #[inline]
    pub fn new(config: RateLimitConfig) -> Self {
        let now = Instant::now().elapsed().as_millis() as u64;
        
        TokenBucket {
            // Start with full bucket (scaled by 1000)
            tokens: AtomicU64::new(config.max_tokens * 1000),
            last_refill: AtomicU64::new(now),
            config,
            enabled: AtomicBool::new(true),
            total_requests: AtomicU64::new(0),
            total_rejections: AtomicU64::new(0),
        }
    }

    /// Try to acquire a token
    /// 
    /// Returns true if token was acquired, false if rate limited
    #[inline]
    pub fn try_acquire(&self) -> bool {
        if !self.enabled.load(Ordering::Relaxed) {
            return true; // Disabled, allow all
        }

        self.total_requests.fetch_add(1, Ordering::Relaxed);

        let now = Instant::now().elapsed().as_millis() as u64;
        
        // Refill tokens
        self.refill(now);

        // Check minimum interval
        let last_refill = self.last_refill.load(Ordering::Relaxed);
        if now - last_refill < self.config.min_interval_ms {
            self.total_rejections.fetch_add(1, Ordering::Relaxed);
            return false;
        }

        // Try to consume a token (tokens are scaled by 1000)
        let token_cost = 1000u64;
        
        loop {
            let current = self.tokens.load(Ordering::Acquire);
            
            if current < token_cost {
                self.total_rejections.fetch_add(1, Ordering::Relaxed);
                return false;
            }

            let new_value = current - token_cost;
            if self.tokens.compare_exchange_weak(
                current,
                new_value,
                Ordering::AcqRel,
                Ordering::Acquire,
            ).is_ok() {
                return true;
            }
        }
    }

    /// Acquire a token, waiting if necessary
    /// 
    /// Returns the wait duration if had to wait
    #[inline]
    pub async fn acquire(&self) -> Duration {
        let mut wait_count = 0;
        const MAX_WAITS: u32 = 100;

        while !self.try_acquire() {
            wait_count += 1;
            
            if wait_count >= MAX_WAITS {
                log::warn!("Rate limit wait exceeded {} attempts", MAX_WAITS);
                break;
            }

            // Wait a small amount before retrying
            tokio::time::sleep(Duration::from_millis(self.config.min_interval_ms)).await;
        }

        Duration::from_millis(wait_count * self.config.min_interval_ms)
    }

    /// Refill tokens based on elapsed time
    #[inline]
    fn refill(&self, now: u64) {
        let last = self.last_refill.load(Ordering::Relaxed);
        let elapsed_ms = now.saturating_sub(last);

        if elapsed_ms == 0 {
            return;
        }

        // Calculate tokens to add (scaled by 1000)
        let tokens_to_add = (elapsed_ms as f64 * self.config.refill_rate / 1000.0 * 1000.0) as u64;

        if tokens_to_add == 0 {
            return;
        }

        loop {
            let current = self.tokens.load(Ordering::Acquire);
            let max_scaled = self.config.max_tokens * 1000;
            let new_value = (current + tokens_to_add).min(max_scaled);

            if self.last_refill.compare_exchange_weak(
                last,
                now,
                Ordering::AcqRel,
                Ordering::Acquire,
            ).is_ok() {
                self.tokens.store(new_value, Ordering::Release);
                break;
            }
        }
    }

    /// Get current token count (unscaled)
    #[inline]
    pub fn available_tokens(&self) -> f64 {
        self.tokens.load(Ordering::Relaxed) as f64 / 1000.0
    }

    /// Enable or disable the limiter
    #[inline]
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Relaxed);
    }

    /// Check if enabled
    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }

    /// Get statistics
    #[inline]
    pub fn stats(&self) -> RateLimitStats {
        RateLimitStats {
            available_tokens: self.available_tokens(),
            max_tokens: self.config.max_tokens,
            refill_rate: self.config.refill_rate,
            total_requests: self.total_requests.load(Ordering::Relaxed),
            total_rejections: self.total_rejections.load(Ordering::Relaxed),
        }
    }

    /// Reset the bucket to full
    #[inline]
    pub fn reset(&self) {
        let now = Instant::now().elapsed().as_millis() as u64;
        self.tokens.store(self.config.max_tokens * 1000, Ordering::Release);
        self.last_refill.store(now, Ordering::Release);
    }
}

/// Rate limit statistics
#[derive(Debug, Clone)]
pub struct RateLimitStats {
    pub available_tokens: f64,
    pub max_tokens: u64,
    pub refill_rate: f64,
    pub total_requests: u64,
    pub total_rejections: u64,
}

impl RateLimitStats {
    /// Get rejection rate
    #[inline]
    pub fn rejection_rate(&self) -> f64 {
        if self.total_requests == 0 {
            return 0.0;
        }
        self.total_rejections as f64 / self.total_requests as f64 * 100.0
    }
}

/// Multi-bucket rate limiter for different endpoint categories
pub struct RateLimiter {
    /// Request weight limiter (general API calls)
    request_limiter: TokenBucket,
    /// Order limiter (order placement/modification)
    order_limiter: Option<TokenBucket>,
    /// Raw order limiter (separate from weighted)
    raw_order_limiter: Option<TokenBucket>,
}

impl RateLimiter {
    #[inline]
    pub fn new(request_config: RateLimitConfig) -> Self {
        RateLimiter {
            request_limiter: TokenBucket::new(request_config),
            order_limiter: None,
            raw_order_limiter: None,
        }
    }

    /// Add order-specific rate limiting
    #[inline]
    pub fn with_order_limit(mut self, config: RateLimitConfig) -> Self {
        self.order_limiter = Some(TokenBucket::new(config));
        self
    }

    /// Add raw order rate limiting
    #[inline]
    pub fn with_raw_order_limit(mut self, config: RateLimitConfig) -> Self {
        self.raw_order_limiter = Some(TokenBucket::new(config));
        self
    }

    /// Try to acquire a request token
    #[inline]
    pub fn try_acquire_request(&self) -> bool {
        self.request_limiter.try_acquire()
    }

    /// Try to acquire an order token
    #[inline]
    pub fn try_acquire_order(&self) -> bool {
        if let Some(ref limiter) = self.order_limiter {
            limiter.try_acquire()
        } else {
            true // No order limiter configured
        }
    }

    /// Acquire both request and order tokens
    #[inline]
    pub async fn acquire_for_order(&self) -> Result<()> {
        // First acquire request token
        if !self.request_limiter.try_acquire() {
            self.request_limiter.acquire().await;
        }

        // Then acquire order token if configured
        if let Some(ref limiter) = self.order_limiter {
            if !limiter.try_acquire() {
                limiter.acquire().await;
            }
        }

        Ok(())
    }

    /// Get request limiter stats
    #[inline]
    pub fn request_stats(&self) -> RateLimitStats {
        self.request_limiter.stats()
    }

    /// Get order limiter stats
    #[inline]
    pub fn order_stats(&self) -> Option<RateLimitStats> {
        self.order_limiter.as_ref().map(|l| l.stats())
    }

    /// Disable all limiters (for testing/emergency)
    #[inline]
    pub fn disable_all(&self) {
        self.request_limiter.set_enabled(false);
        if let Some(ref limiter) = self.order_limiter {
            limiter.set_enabled(false);
        }
        if let Some(ref limiter) = self.raw_order_limiter {
            limiter.set_enabled(false);
        }
    }

    /// Enable all limiters
    #[inline]
    pub fn enable_all(&self) {
        self.request_limiter.set_enabled(true);
        if let Some(ref limiter) = self.order_limiter {
            limiter.set_enabled(true);
        }
        if let Some(ref limiter) = self.raw_order_limiter {
            limiter.set_enabled(true);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_bucket_creation() {
        let config = RateLimitConfig::binance_standard();
        let bucket = TokenBucket::new(config.clone());
        
        assert_eq!(bucket.available_tokens(), config.max_tokens as f64);
        assert!(bucket.is_enabled());
    }

    #[test]
    fn test_token_acquisition() {
        let config = RateLimitConfig::new(10, 10.0, 10);
        let bucket = TokenBucket::new(config);
        
        // Should be able to acquire tokens up to limit
        for _ in 0..10 {
            assert!(bucket.try_acquire());
        }
        
        // Next should fail (rate limited)
        assert!(!bucket.try_acquire());
    }

    #[test]
    fn test_rate_limit_stats() {
        let config = RateLimitConfig::new(5, 10.0, 10);
        let bucket = TokenBucket::new(config);
        
        // Acquire some tokens
        for _ in 0..3 {
            bucket.try_acquire();
        }
        
        let stats = bucket.stats();
        assert_eq!(stats.total_requests, 3);
        assert!(stats.available_tokens() > 0.0);
    }

    #[tokio::test]
    async fn test_async_acquire() {
        let config = RateLimitConfig::new(2, 100.0, 10);
        let bucket = TokenBucket::new(config);
        
        // Exhaust tokens
        bucket.try_acquire();
        bucket.try_acquire();
        
        // Async acquire should wait and then succeed
        let wait_time = bucket.acquire().await;
        assert!(wait_time.as_millis() >= 10);
    }
}
