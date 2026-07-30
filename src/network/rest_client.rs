//! REST API Client for Binance
//! 
//! Builds a highly optimized HTTP client using `reqwest` with aggressive connection pooling
//! and HTTP/2 multiplexing. Used for fetching initial order book snapshots, account balances,
//! and placing medium-frequency execution orders.

use anyhow::{Context, Result};
use reqwest::{Client, Response, StatusCode};
use std::time::Duration;
use crate::network::auth::BinanceAuth;
use crate::market_data::{SymbolId, OrderBookSnapshot, Level, Price, Quantity};

/// Binance REST API base URL
const BINANCE_API_URL: &str = "https://api.binance.com";

/// Default timeout for requests
const DEFAULT_TIMEOUT_MS: u64 = 5000;

/// Connection pool configuration
#[derive(Debug, Clone)]
pub struct RestConfig {
    /// Base API URL
    pub base_url: String,
    /// Request timeout in milliseconds
    pub timeout_ms: u64,
    /// Maximum connections per host
    pub max_connections_per_host: usize,
    /// Keep-alive interval
    pub keep_alive_ms: u64,
    /// Enable HTTP/2
    pub http2: bool,
}

impl Default for RestConfig {
    fn default() -> Self {
        RestConfig {
            base_url: BINANCE_API_URL.to_string(),
            timeout_ms: DEFAULT_TIMEOUT_MS,
            max_connections_per_host: 100,
            keep_alive_ms: 90_000, // 90 seconds
            http2: true,
        }
    }
}

/// High-performance REST client
pub struct RestClient {
    config: RestConfig,
    client: Client,
    auth: Option<BinanceAuth>,
}

impl RestClient {
    /// Create a new REST client without authentication (public endpoints only)
    #[inline]
    pub fn new(config: RestConfig) -> Result<Self> {
        let builder = Client::builder()
            .timeout(Duration::from_millis(config.timeout_ms))
            .tcp_keepalive(Duration::from_millis(config.keep_alive_ms))
            .pool_max_idle_per_host(config.max_connections_per_host);
        
        let builder = if config.http2 {
            builder.http2_adaptive_window(true)
                .http2_keep_alive_interval(Duration::from_millis(30_000))
                .http2_keep_alive_timeout(Duration::from_millis(10_000))
        } else {
            builder
        };

        let client = builder
            .build()
            .context("Failed to create HTTP client")?;

        Ok(RestClient {
            config,
            client,
            auth: None,
        })
    }

    /// Create a new REST client with authentication
    #[inline]
    pub fn with_auth(config: RestConfig, auth: BinanceAuth) -> Result<Self> {
        let mut client = Self::new(config)?;
        client.auth = Some(auth);
        Ok(client)
    }

    /// Get the base URL
    #[inline]
    pub fn base_url(&self) -> &str {
        &self.config.base_url
    }

    /// Make a GET request
    #[inline]
    pub async fn get(&self, endpoint: &str, params: &[(&str, &str)]) -> Result<Response> {
        let url = format!("{}/api/v3{}", self.config.base_url, endpoint);
        
        let request = self.client.get(&url).query(params);
        
        let response = request
            .send()
            .await
            .context("GET request failed")?;
        
        Ok(response)
    }

    /// Make an authenticated GET request with signature
    #[inline]
    pub async fn get_signed(&self, endpoint: &str, mut params: Vec<(String, String)>) -> Result<Response> {
        let auth = self.auth.as_ref().context("No authentication configured")?;
        
        // Add timestamp
        let timestamp = auth.current_timestamp_ms();
        params.push(("timestamp".to_string(), timestamp.to_string()));
        
        // Create signature
        let query_string = params
            .iter()
            .map(|(k, v)| format!("{}={}", k, v))
            .collect::<Vec<_>>()
            .join("&");
        
        let signature = auth.sign(&query_string);
        params.push(("signature".to_string(), signature));
        
        let url = format!("{}/api/v3{}", self.config.base_url, endpoint);
        
        let request = self.client.get(&url).query(&params);
        
        let response = request
            .send()
            .await
            .context("Signed GET request failed")?;
        
        Ok(response)
    }

    /// Make a POST request
    #[inline]
    pub async fn post(&self, endpoint: &str, body: &serde_json::Value) -> Result<Response> {
        let url = format!("{}/api/v3{}", self.config.base_url, endpoint);
        
        let response = self.client
            .post(&url)
            .json(body)
            .send()
            .await
            .context("POST request failed")?;
        
        Ok(response)
    }

    /// Make an authenticated POST request with signature
    #[inline]
    pub async fn post_signed(&self, endpoint: &str, mut params: Vec<(String, String)>) -> Result<Response> {
        let auth = self.auth.as_ref().context("No authentication configured")?;
        
        // Add timestamp
        let timestamp = auth.current_timestamp_ms();
        params.push(("timestamp".to_string(), timestamp.to_string()));
        
        // Create signature from sorted params
        params.sort_by(|a, b| a.0.cmp(&b.0));
        let query_string = params
            .iter()
            .map(|(k, v)| format!("{}={}", k, v))
            .collect::<Vec<_>>()
            .join("&");
        
        let signature = auth.sign(&query_string);
        params.push(("signature".to_string(), signature));
        
        let url = format!("{}/api/v3{}", self.config.base_url, endpoint);
        
        let response = self.client
            .post(&url)
            .form(&params)
            .send()
            .await
            .context("Signed POST request failed")?;
        
        Ok(response)
    }

    /// Make a DELETE request
    #[inline]
    pub async fn delete_signed(&self, endpoint: &str, mut params: Vec<(String, String)>) -> Result<Response> {
        let auth = self.auth.as_ref().context("No authentication configured")?;
        
        let timestamp = auth.current_timestamp_ms();
        params.push(("timestamp".to_string(), timestamp.to_string()));
        
        params.sort_by(|a, b| a.0.cmp(&b.0));
        let query_string = params
            .iter()
            .map(|(k, v)| format!("{}={}", k, v))
            .collect::<Vec<_>>()
            .join("&");
        
        let signature = auth.sign(&query_string);
        params.push(("signature".to_string(), signature));
        
        let url = format!("{}/api/v3{}", self.config.base_url, endpoint);
        
        let response = self.client
            .delete(&url)
            .query(&params)
            .send()
            .await
            .context("Signed DELETE request failed")?;
        
        Ok(response)
    }

    /// Fetch order book depth from REST API
    #[inline]
    pub async fn get_order_book(&self, symbol: &str, limit: u16) -> Result<OrderBookSnapshot> {
        let params = [
            ("symbol", symbol),
            ("limit", &limit.to_string()),
        ];
        
        let response = self.get("/depth", &params).await?;
        
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!("API error {}: {}", status, error_text));
        }

        let json: serde_json::Value = response.json().await
            .context("Failed to parse order book response")?;

        let last_update_id = json["lastUpdateId"].as_u64().unwrap_or(0);
        let symbol_id = SymbolId::from_str(symbol);
        
        let mut snapshot = OrderBookSnapshot::new(symbol_id, last_update_id);
        
        // Parse bids
        if let Some(bids) = json["bids"].as_array() {
            for bid in bids {
                if let Some(arr) = bid.as_array() {
                    if arr.len() >= 2 {
                        let price = arr[0].as_str().and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.0);
                        let qty = arr[1].as_str().and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.0);
                        snapshot.bids.push(Level::new(
                            Price::from_f64(price),
                            Quantity::from_f64(qty),
                            1,
                        ));
                    }
                }
            }
        }

        // Parse asks
        if let Some(asks) = json["asks"].as_array() {
            for ask in asks {
                if let Some(arr) = ask.as_array() {
                    if arr.len() >= 2 {
                        let price = arr[0].as_str().and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.0);
                        let qty = arr[1].as_str().and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.0);
                        snapshot.asks.push(Level::new(
                            Price::from_f64(price),
                            Quantity::from_f64(qty),
                            1,
                        ));
                    }
                }
            }
        }

        Ok(snapshot)
    }

    /// Check server time
    #[inline]
    pub async fn get_server_time(&self) -> Result<i64> {
        let response = self.get("/time", &[]).await?;
        let json: serde_json::Value = response.json().await
            .context("Failed to parse server time response")?;
        
        Ok(json["serverTime"].as_i64().unwrap_or(0))
    }

    /// Get exchange info
    #[inline]
    pub async fn get_exchange_info(&self) -> Result<serde_json::Value> {
        let response = self.get("/exchangeInfo", &[]).await?;
        response.json().await.context("Failed to parse exchange info")
    }

    /// Get account info (authenticated)
    #[inline]
    pub async fn get_account_info(&self) -> Result<serde_json::Value> {
        let response = self.get_signed("/account", vec![]).await?;
        response.json().await.context("Failed to parse account info")
    }

    /// Get current balance for an asset
    #[inline]
    pub async fn get_balance(&self, asset: &str) -> Result<f64> {
        let account = self.get_account_info().await?;
        
        if let Some(balances) = account["balances"].as_array() {
            for balance in balances {
                if balance["asset"].as_str() == Some(asset) {
                    return Ok(balance["free"].as_str()
                        .and_then(|s| s.parse::<f64>().ok())
                        .unwrap_or(0.0));
                }
            }
        }
        
        Ok(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rest_config_default() {
        let config = RestConfig::default();
        assert_eq!(config.timeout_ms, 5000);
        assert!(config.http2);
    }

    #[tokio::test]
    async fn test_get_server_time() {
        let config = RestConfig::default();
        let client = RestClient::new(config).unwrap();
        
        // This will fail in offline tests but validates the code path
        let _ = client.get_server_time().await;
    }
}
