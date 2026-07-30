//! WebSocket Client for High-Performance Market Data Ingestion
//! 
//! Implements an asynchronous WebSocket client using `tokio-tungstenite` to subscribe
//! to Binance market data streams. Handles raw message ingestion, ping/pong keep-alives,
//! and binary/text frame decoding with minimal overhead.

use anyhow::{Context, Result};
use futures_util::{stream::SplitSink, SinkExt, StreamExt};
use tokio::net::TcpStream;
use tokio_tungstenite::{connect_async, tungstenite::Message, WebSocketStream};
use crate::market_data::{SymbolId, OrderBookDelta, Trade, Ticker};
use std::sync::Arc;
use std::time::Duration;

/// Binance WebSocket base URL
const BINANCE_WS_URL: &str = "wss://stream.binance.com:9443/ws";

/// Combined stream URL for multiple symbols
const BINANCE_COMBINED_URL: &str = "wss://stream.binance.com:9443/stream?streams=";

/// WebSocket connection state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WsState {
    Disconnected,
    Connecting,
    Connected,
    Reconnecting,
    Error,
}

/// Configuration for WebSocket connection
#[derive(Debug, Clone)]
pub struct WsConfig {
    /// Base URL for the WebSocket endpoint
    pub base_url: String,
    /// Whether to use combined stream
    pub use_combined: bool,
    /// Ping interval in milliseconds
    pub ping_interval_ms: u64,
    /// Reconnect delay in milliseconds
    pub reconnect_delay_ms: u64,
    /// Maximum reconnect attempts (0 = infinite)
    pub max_reconnect_attempts: u32,
}

impl Default for WsConfig {
    fn default() -> Self {
        WsConfig {
            base_url: BINANCE_WS_URL.to_string(),
            use_combined: true,
            ping_interval_ms: 30_000, // 30 seconds
            reconnect_delay_ms: 1_000, // 1 second
            max_reconnect_attempts: 0, // Infinite
        }
    }
}

/// Raw WebSocket message received from exchange
#[derive(Debug, Clone)]
pub struct RawWsMessage {
    pub payload: Vec<u8>,
    pub timestamp_ns: i64,
    pub is_binary: bool,
}

impl RawWsMessage {
    #[inline]
    pub fn new(payload: Vec<u8>, is_binary: bool) -> Self {
        RawWsMessage {
            payload,
            timestamp_ns: chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0),
            is_binary,
        }
    }

    #[inline]
    pub fn as_str(&self) -> Option<&str> {
        if self.is_binary {
            None
        } else {
            std::str::from_utf8(&self.payload).ok()
        }
    }
}

/// WebSocket client for market data ingestion
pub struct WsClient {
    config: WsConfig,
    state: WsState,
    reconnect_count: u32,
    last_message_time_ns: i64,
}

impl WsClient {
    #[inline]
    pub fn new(config: WsConfig) -> Self {
        WsClient {
            config,
            state: WsState::Disconnected,
            reconnect_count: 0,
            last_message_time_ns: 0,
        }
    }

    /// Build a combined stream URL for multiple symbols
    #[inline]
    pub fn build_combined_url(symbols: &[&str], channels: &[&str]) -> String {
        let mut streams = Vec::new();
        
        for symbol in symbols {
            let sym_lower = symbol.to_lowercase();
            for channel in channels {
                streams.push(format!("{}@{}", sym_lower, channel));
            }
        }
        
        format!("{}{}", BINANCE_COMBINED_URL, streams.join("/"))
    }

    /// Connect to the WebSocket endpoint
    #[inline]
    pub async fn connect(&mut self, url: &str) -> Result<WebSocketStream<TcpStream>> {
        self.state = WsState::Connecting;
        
        let (ws_stream, response) = connect_async(url)
            .await
            .context("Failed to connect to WebSocket")?;
        
        log::info!("Connected to WebSocket: {}", url);
        self.state = WsState::Connected;
        self.reconnect_count = 0;
        
        Ok(ws_stream)
    }

    /// Send a subscription message
    #[inline]
    pub async fn subscribe<S: SinkExt<Message> + Unpin>(
        &self,
        sink: &mut S,
        streams: &[String],
    ) -> Result<()>
    where
        S::Error: std::error::Error + Send + Sync + 'static,
    {
        let subscribe_msg = serde_json::json!({
            "method": "SUBSCRIBE",
            "params": streams,
            "id": 1
        });
        
        sink.send(Message::Text(subscribe_msg.to_string()))
            .await
            .context("Failed to send subscribe message")?;
        
        Ok(())
    }

    /// Receive and decode messages
    #[inline]
    pub async fn receive_message<S: StreamExt + Unpin>(
        &mut self,
        stream: &mut S,
    ) -> Option<RawWsMessage>
    where
        S::Item: Into<Result<Message, tokio_tungstenite::tungstenite::Error>>,
    {
        match stream.next().await {
            Some(item) => {
                match item.into() {
                    Ok(Message::Text(text)) => {
                        self.last_message_time_ns = chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0);
                        Some(RawWsMessage::new(text.into_bytes(), false))
                    }
                    Ok(Message::Binary(data)) => {
                        self.last_message_time_ns = chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0);
                        Some(RawWsMessage::new(data, true))
                    }
                    Ok(Message::Ping(data)) => {
                        // Auto-pong handled by tungstenite
                        Some(RawWsMessage::new(data, true))
                    }
                    Ok(Message::Pong(_)) => {
                        None // Ignore pong
                    }
                    Ok(Message::Close(_)) => {
                        self.state = WsState::Disconnected;
                        None
                    }
                    Ok(Message::Frame(_)) => None,
                    Err(e) => {
                        log::error!("WebSocket error: {}", e);
                        self.state = WsState::Error;
                        None
                    }
                }
            }
            None => {
                self.state = WsState::Disconnected;
                None
            }
        }
    }

    /// Check if reconnection is needed based on last message time
    #[inline]
    pub fn needs_reconnect(&self, timeout_ms: u64) -> bool {
        if self.state != WsState::Connected {
            return false;
        }
        
        let now_ns = chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0);
        let elapsed_ms = (now_ns - self.last_message_time_ns) / 1_000_000;
        
        elapsed_ms > timeout_ms as i64
    }

    /// Get current connection state
    #[inline]
    pub fn state(&self) -> WsState {
        self.state
    }

    /// Increment reconnect counter
    #[inline]
    pub fn increment_reconnect(&mut self) {
        self.reconnect_count += 1;
    }

    /// Check if max reconnect attempts exceeded
    #[inline]
    pub fn should_stop_reconnecting(&self) -> bool {
        self.config.max_reconnect_attempts > 0 
            && self.reconnect_count >= self.config.max_reconnect_attempts
    }

    /// Get reconnect count
    #[inline]
    pub fn reconnect_count(&self) -> u32 {
        self.reconnect_count
    }
}

/// Stream-specific WebSocket manager
pub struct WsStreamManager {
    client: WsClient,
    subscribed_symbols: Vec<SymbolId>,
    subscribed_channels: Vec<String>,
}

impl WsStreamManager {
    #[inline]
    pub fn new() -> Self {
        WsStreamManager {
            client: WsClient::new(WsConfig::default()),
            subscribed_symbols: Vec::new(),
            subscribed_channels: Vec::new(),
        }
    }

    /// Subscribe to trade streams for multiple symbols
    #[inline]
    pub fn subscribe_trades(&mut self, symbols: &[&str]) {
        for sym in symbols {
            self.subscribed_symbols.push(SymbolId::from_str(sym));
        }
        self.subscribed_channels.push("trade".to_string());
    }

    /// Subscribe to depth streams for multiple symbols
    #[inline]
    pub fn subscribe_depth(&mut self, symbols: &[&str], update_speed: &str) {
        for sym in symbols {
            self.subscribed_symbols.push(SymbolId::from_str(sym));
        }
        self.subscribed_channels.push(format!("depth{}", update_speed));
    }

    /// Subscribe to book ticker streams
    #[inline]
    pub fn subscribe_book_ticker(&mut self, symbols: &[&str]) {
        for sym in symbols {
            self.subscribed_symbols.push(SymbolId::from_str(sym));
        }
        self.subscribed_channels.push("bookTicker".to_string());
    }

    /// Build the combined stream URL
    #[inline]
    pub fn build_url(&self) -> String {
        let symbols: Vec<&str> = self.subscribed_symbols
            .iter()
            .map(|s| s.as_str())
            .collect();
        
        WsClient::build_combined_url(&symbols, &self.subscribed_channels.iter().map(|s| s.as_str()).collect::<Vec<_>>())
    }

    /// Run the WebSocket event loop
    #[inline]
    pub async fn run<F>(
        &mut self,
        mut handler: F,
    ) -> Result<()>
    where
        F: FnMut(RawWsMessage) -> futures_util::future::Ready<Result<()>> + Send,
    {
        let url = self.build_url();
        let mut ws_stream = self.client.connect(&url).await?;
        let (mut sink, mut stream) = ws_stream.split();

        // Build subscription list
        let streams: Vec<String> = self.subscribed_symbols
            .iter()
            .flat_map(|sym| {
                self.subscribed_channels.iter().map(move |ch| {
                    format!("{}@{}", sym.as_str().to_lowercase(), ch)
                })
            })
            .collect();

        // Subscribe to streams
        self.client.subscribe(&mut sink, &streams).await?;

        // Main message loop
        loop {
            if let Some(msg) = self.client.receive_message(&mut stream).await {
                handler(msg).await?;
            }

            // Check for reconnection need
            if self.client.needs_reconnect(60_000) {
                log::warn!("No messages received for 60s, reconnecting...");
                break; // Exit to trigger reconnection
            }
        }

        Ok(())
    }
}

impl Default for WsStreamManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_combined_url() {
        let url = WsClient::build_combined_url(
            &["BTCUSDT", "ETHUSDT"],
            &["trade", "depth20"]
        );
        
        assert!(url.contains("btcusdt@trade"));
        assert!(url.contains("ethusdt@depth20"));
    }

    #[test]
    fn test_ws_client_creation() {
        let client = WsClient::new(WsConfig::default());
        assert_eq!(client.state(), WsState::Disconnected);
        assert_eq!(client.reconnect_count(), 0);
    }
}
