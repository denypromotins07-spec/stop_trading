//! Network Module Root
//! 
//! Initializes connection pools and exports WebSocket manager interfaces.

pub mod ws_client;
pub mod reconnection;
pub mod rest_client;
pub mod auth;
pub mod rate_limiter;

pub use ws_client::*;
pub use reconnection::*;
pub use rest_client::*;
pub use auth::*;
pub use rate_limiter::*;

use std::sync::OnceLock;
use anyhow::Result;

/// Global network manager singleton
static NETWORK_MANAGER: OnceLock<NetworkManager> = OnceLock::new();

/// Network configuration aggregating all network-related settings
#[derive(Debug, Clone)]
pub struct NetworkConfig {
    /// WebSocket configuration
    pub ws_config: WsConfig,
    /// REST configuration
    pub rest_config: RestConfig,
    /// Rate limit configuration
    pub rate_limit_config: RateLimitConfig,
    /// Enable authentication
    pub enable_auth: bool,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        NetworkConfig {
            ws_config: WsConfig::default(),
            rest_config: RestConfig::default(),
            rate_limit_config: RateLimitConfig::binance_standard(),
            enable_auth: false,
        }
    }
}

/// Centralized network manager coordinating all network components
pub struct NetworkManager {
    /// REST client
    rest_client: RestClient,
    /// Rate limiter
    rate_limiter: RateLimiter,
    /// Reconnection manager
    reconnection_manager: ReconnectionManager,
    /// Authentication (optional)
    auth: Option<BinanceAuth>,
    /// Time synchronizer
    time_synchronizer: Option<TimeSynchronizer>,
}

impl NetworkManager {
    /// Create a new network manager with the given configuration
    #[inline]
    pub fn new(config: NetworkConfig) -> Result<Self> {
        let auth = if config.enable_auth {
            Some(BinanceAuth::from_env()?)
        } else {
            None
        };

        let rest_client = if let Some(ref a) = auth {
            RestClient::with_auth(config.rest_config.clone(), a.clone())?
        } else {
            RestClient::new(config.rest_config.clone())?
        };

        let rate_limiter = RateLimiter::new(config.rate_limit_config.clone());

        Ok(NetworkManager {
            rest_client,
            rate_limiter,
            reconnection_manager: ReconnectionManager::new(),
            auth,
            time_synchronizer: None,
        })
    }

    /// Initialize with time synchronization
    #[inline]
    pub async fn with_time_sync(mut self, sync_interval_ms: u64) -> Result<Self> {
        if let Some(auth) = self.auth.take() {
            // Initial time sync
            match self.rest_client.get_server_time().await {
                Ok(server_time) => {
                    let mut synced_auth = auth;
                    synced_auth.sync_time(server_time).await;
                    
                    self.time_synchronizer = Some(TimeSynchronizer::new(
                        synced_auth,
                        sync_interval_ms,
                    ));
                }
                Err(e) => {
                    log::warn!("Failed to sync server time: {}", e);
                    self.auth = Some(auth);
                }
            }
        }
        
        Ok(self)
    }

    /// Get the REST client reference
    #[inline]
    pub fn rest_client(&self) -> &RestClient {
        &self.rest_client
    }

    /// Get the rate limiter reference
    #[inline]
    pub fn rate_limiter(&self) -> &RateLimiter {
        &self.rate_limiter
    }

    /// Get the reconnection manager reference
    #[inline]
    pub fn reconnection_manager(&self) -> &ReconnectionManager {
        &self.reconnection_manager
    }

    /// Get mutable reconnection manager
    #[inline]
    pub fn reconnection_manager_mut(&mut self) -> &mut ReconnectionManager {
        &mut self.reconnection_manager
    }

    /// Get auth reference if configured
    #[inline]
    pub fn auth(&self) -> Option<&BinanceAuth> {
        self.auth.as_ref()
    }

    /// Check if time sync is needed and perform it
    #[inline]
    pub async fn maybe_sync_time(&mut self) -> Result<()> {
        if let Some(ref mut sync) = self.time_synchronizer {
            if sync.needs_sync() {
                match self.rest_client.get_server_time().await {
                    Ok(server_time) => {
                        sync.auth_mut().sync_time(server_time).await;
                        sync.mark_synced();
                        log::debug("Time synchronized with server");
                    }
                    Err(e) => {
                        log::warn!("Time sync failed: {}", e);
                    }
                }
            }
        }
        Ok(())
    }

    /// Make a rate-limited REST request
    #[inline]
    pub async fn rate_limited_request<F, T>(&self, f: F) -> Result<T>
    where
        F: FnOnce() -> futures_util::future::LocalBoxFuture<'static, Result<T>>,
    {
        // Wait for rate limit
        self.rate_limiter.request_limiter.acquire().await;
        
        // Execute request
        f()().await
    }

    /// Get network statistics
    #[inline]
    pub fn stats(&self) -> NetworkStats {
        NetworkStats {
            rate_limit_stats: self.rate_limiter.request_stats(),
            order_rate_limit_stats: self.rate_limiter.order_stats(),
            reconnect_state: self.reconnection_manager.state(),
            gap_stats: self.reconnection_manager.gap_stats(),
            has_auth: self.auth.is_some(),
            has_time_sync: self.time_synchronizer.is_some(),
        }
    }

    /// Reset all network state
    #[inline]
    pub fn reset(&mut self) {
        self.reconnection_manager.reset();
        self.rate_limiter.request_limiter.reset();
    }
}

/// Network statistics snapshot
#[derive(Debug, Clone)]
pub struct NetworkStats {
    pub rate_limit_stats: RateLimitStats,
    pub order_rate_limit_stats: Option<RateLimitStats>,
    pub reconnect_state: ReconnectState,
    pub gap_stats: (u32, u64),
    pub has_auth: bool,
    pub has_time_sync: bool,
}

/// Initialize the global network manager
#[inline]
pub fn init_network_manager(config: NetworkConfig) -> Result<&'static NetworkManager> {
    let manager = NetworkManager::new(config)?;
    Ok(NETWORK_MANAGER.get_or_init(|| manager))
}

/// Get the global network manager
#[inline]
pub fn get_network_manager() -> Option<&'static NetworkManager> {
    NETWORK_MANAGER.get()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_config_default() {
        let config = NetworkConfig::default();
        assert!(!config.enable_auth);
        assert_eq!(config.rate_limit_config.max_tokens, 1200);
    }

    #[test]
    fn test_network_manager_creation() {
        let config = NetworkConfig {
            enable_auth: false,
            ..Default::default()
        };
        
        let manager = NetworkManager::new(config).unwrap();
        assert!(manager.auth().is_none());
        assert!(!manager.stats().has_time_sync);
    }

    #[test]
    fn test_network_stats() {
        let config = NetworkConfig::default();
        let manager = NetworkManager::new(config).unwrap();
        let stats = manager.stats();
        
        assert_eq!(stats.rate_limit_stats.max_tokens, 1200);
        assert!(!stats.has_auth);
    }
}
