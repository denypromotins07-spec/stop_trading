//! Binance API Authentication using HMAC-SHA256
//! 
//! Implements Binance API authentication using HMAC-SHA256 signatures via the `ring` crate.
//! Ensures timestamps are synchronized with the exchange server time to prevent 
//! "Timestamp outside recvwindow" rejections.

use anyhow::{Context, Result};
use hmac::{Hmac, Mac};
use sha2::Sha256;
use std::time::{SystemTime, UNIX_EPOCH};
use std::sync::atomic::{AtomicI64, Ordering};

type HmacSha256 = Hmac<Sha256>;

/// Binance API authenticator
pub struct BinanceAuth {
    /// API key (public)
    api_key: String,
    /// API secret (private, stored securely)
    api_secret: Vec<u8>,
    /// Server time offset in milliseconds (server_time - local_time)
    server_time_offset_ms: AtomicI64,
    /// Receive window in milliseconds
    recv_window_ms: u64,
}

impl BinanceAuth {
    /// Create a new authenticator with API credentials
    #[inline]
    pub fn new(api_key: &str, api_secret: &str) -> Self {
        BinanceAuth {
            api_key: api_key.to_string(),
            api_secret: api_secret.as_bytes().to_vec(),
            server_time_offset_ms: AtomicI64::new(0),
            recv_window_ms: 5000, // Default 5 seconds
        }
    }

    /// Create from environment variables
    #[inline]
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("BINANCE_API_KEY")
            .context("BINANCE_API_KEY not set")?;
        let api_secret = std::env::var("BINANCE_API_SECRET")
            .context("BINANCE_API_SECRET not set")?;
        
        Ok(Self::new(&api_key, &api_secret))
    }

    /// Get the API key
    #[inline]
    pub fn api_key(&self) -> &str {
        &self.api_key
    }

    /// Sign a message with HMAC-SHA256
    #[inline]
    pub fn sign(&self, message: &str) -> String {
        let mut mac = HmacSha256::new_from_slice(&self.api_secret)
            .expect("HMAC can take key of any size");
        mac.update(message.as_bytes());
        let result = mac.finalize();
        
        // Convert to hex string
        hex::encode(result.into_bytes())
    }

    /// Get current local timestamp in milliseconds
    #[inline]
    pub fn local_timestamp_ms(&self) -> i64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0)
    }

    /// Get current server-adjusted timestamp in milliseconds
    #[inline]
    pub fn current_timestamp_ms(&self) -> i64 {
        let local = self.local_timestamp_ms();
        let offset = self.server_time_offset_ms.load(Ordering::Relaxed);
        local + offset
    }

    /// Update the server time offset
    /// 
    /// Call this after fetching server time: offset = server_time - local_time
    #[inline]
    pub fn update_server_time_offset(&self, server_time_ms: i64) {
        let local = self.local_timestamp_ms();
        let offset = server_time_ms - local;
        self.server_time_offset_ms.store(offset, Ordering::Relaxed);
        log::debug!("Server time offset updated to {}ms", offset);
    }

    /// Synchronize time with server
    #[inline]
    pub async fn sync_time(&mut self, server_time_ms: i64) {
        self.update_server_time_offset(server_time_ms);
    }

    /// Check if timestamp is within receive window
    #[inline]
    pub fn is_timestamp_valid(&self, timestamp_ms: i64) -> bool {
        let now = self.current_timestamp_ms();
        let diff = (timestamp_ms - now).abs();
        diff <= self.recv_window_ms as i64
    }

    /// Set the receive window
    #[inline]
    pub fn set_recv_window(&mut self, window_ms: u64) {
        self.recv_window_ms = window_ms;
    }

    /// Get the receive window
    #[inline]
    pub fn recv_window(&self) -> u64 {
        self.recv_window_ms
    }

    /// Generate signed headers for a request
    #[inline]
    pub fn generate_headers(&self, endpoint: &str, params: &str) -> std::collections::HashMap<String, String> {
        let mut headers = std::collections::HashMap::new();
        headers.insert("X-MBX-APIKEY".to_string(), self.api_key.clone());
        
        // The signature is typically added as a query parameter, not header
        // But we include it here for completeness
        let signature = self.sign(params);
        headers.insert("X-SIGNATURE".to_string(), signature);
        
        headers
    }

    /// Validate that credentials are not empty
    #[inline]
    pub fn validate(&self) -> Result<()> {
        if self.api_key.is_empty() {
            return Err(anyhow::anyhow!("API key is empty"));
        }
        if self.api_secret.is_empty() {
            return Err(anyhow::anyhow!("API secret is empty"));
        }
        Ok(())
    }

    /// Get the time offset (for debugging)
    #[inline]
    pub fn time_offset_ms(&self) -> i64 {
        self.server_time_offset_ms.load(Ordering::Relaxed)
    }
}

/// Time synchronizer for maintaining accurate server time
pub struct TimeSynchronizer {
    auth: BinanceAuth,
    /// Last sync timestamp
    last_sync_ms: AtomicI64,
    /// Sync interval in milliseconds
    sync_interval_ms: u64,
}

impl TimeSynchronizer {
    #[inline]
    pub fn new(auth: BinanceAuth, sync_interval_ms: u64) -> Self {
        TimeSynchronizer {
            auth,
            last_sync_ms: AtomicI64::new(0),
            sync_interval_ms,
        }
    }

    /// Check if time sync is needed
    #[inline]
    pub fn needs_sync(&self) -> bool {
        let now = self.auth.local_timestamp_ms();
        let last = self.last_sync_ms.load(Ordering::Relaxed);
        
        if last == 0 {
            return true; // Never synced
        }
        
        (now - last) >= self.sync_interval_ms as i64
    }

    /// Mark sync as complete
    #[inline]
    pub fn mark_synced(&self) {
        let now = self.auth.local_timestamp_ms();
        self.last_sync_ms.store(now, Ordering::Relaxed);
    }

    /// Get the underlying auth reference
    #[inline]
    pub fn auth(&self) -> &BinanceAuth {
        &self.auth
    }

    /// Get mutable auth reference
    #[inline]
    pub fn auth_mut(&mut self) -> &mut BinanceAuth {
        &mut self.auth
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auth_creation() {
        let auth = BinanceAuth::new("test_key", "test_secret");
        assert_eq!(auth.api_key(), "test_key");
        assert!(auth.validate().is_ok());
    }

    #[test]
    fn test_sign_message() {
        let auth = BinanceAuth::new("test_key", "test_secret");
        let signature = auth.sign("symbol=BTCUSDT&quantity=0.001");
        
        // Signature should be 64 hex characters (32 bytes)
        assert_eq!(signature.len(), 64);
        
        // Same input should produce same output
        let signature2 = auth.sign("symbol=BTCUSDT&quantity=0.001");
        assert_eq!(signature, signature2);
    }

    #[test]
    fn test_timestamp_generation() {
        let auth = BinanceAuth::new("test_key", "test_secret");
        let ts1 = auth.current_timestamp_ms();
        std::thread::sleep(std::time::Duration::from_millis(10));
        let ts2 = auth.current_timestamp_ms();
        
        assert!(ts2 >= ts1);
    }

    #[test]
    fn test_time_offset() {
        let auth = BinanceAuth::new("test_key", "test_secret");
        
        // Simulate server being 1000ms ahead
        auth.update_server_time_offset(1000);
        assert_eq!(auth.time_offset_ms(), 1000);
        
        // Current timestamp should include offset
        let local = auth.local_timestamp_ms();
        let adjusted = auth.current_timestamp_ms();
        assert_eq!(adjusted, local + 1000);
    }

    #[test]
    fn test_empty_credentials_validation() {
        let auth = BinanceAuth::new("", "test_secret");
        assert!(auth.validate().is_err());
        
        let auth = BinanceAuth::new("test_key", "");
        assert!(auth.validate().is_err());
    }
}
