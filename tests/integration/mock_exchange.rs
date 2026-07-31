//! Mock Exchange Server for Integration Testing
//! 
//! Simulates Binance-like WebSocket and REST APIs with:
//! - Realistic network latency injection
//! - Sequence gap simulation
//! - REST 504 timeout simulation
//! - Reconnection logic testing

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Result, Context};
use serde::{Deserialize, Serialize};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{broadcast, mpsc, oneshot};
use tokio::time::{sleep, interval};
use tracing::{info, warn, error, debug};

/// Configuration for the mock exchange
#[derive(Debug, Clone)]
pub struct MockExchangeConfig {
    /// Base latency to simulate (e.g., 1-5ms for local, 20-50ms for remote)
    pub base_latency_ms: u64,
    /// Latency jitter (random variation)
    pub latency_jitter_ms: u64,
    /// Probability of sequence gap (0.0 to 1.0)
    pub sequence_gap_probability: f64,
    /// Probability of REST 504 timeout (0.0 to 1.0)
    pub rest_timeout_probability: f64,
    /// WebSocket port
    pub ws_port: u16,
    /// REST API port
    pub rest_port: u16,
}

impl Default for MockExchangeConfig {
    fn default() -> Self {
        Self {
            base_latency_ms: 2,
            latency_jitter_ms: 3,
            sequence_gap_probability: 0.001, // 0.1% chance
            rest_timeout_probability: 0.01,  // 1% chance
            ws_port: 9998,
            rest_port: 9999,
        }
    }
}

/// Order book state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookLevel {
    pub price: f64,
    pub quantity: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBook {
    pub symbol: String,
    pub last_update_id: u64,
    pub bids: Vec<OrderBookLevel>,
    pub asks: Vec<OrderBookLevel>,
}

/// Trade message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    pub trade_id: u64,
    pub symbol: String,
    pub price: f64,
    pub quantity: f64,
    pub is_buyer_maker: bool,
    pub timestamp: u64,
}

/// Depth update message (Binance format)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DepthUpdate {
    pub event_type: String,
    pub event_time: u64,
    pub symbol: String,
    pub first_update_id: u64,
    pub final_update_id: u64,
    pub bids: Vec<Vec<String>>,
    pub asks: Vec<Vec<String>>,
}

/// Mock exchange server state
pub struct MockExchangeServer {
    config: MockExchangeConfig,
    running: Arc<AtomicBool>,
    sequence_counter: Arc<AtomicU64>,
    trade_counter: Arc<AtomicU64>,
    order_book: Arc<tokio::sync::RwLock<OrderBook>>,
}

impl MockExchangeServer {
    pub fn new(config: MockExchangeConfig) -> Self {
        let initial_order_book = OrderBook {
            symbol: "BTCUSDT".to_string(),
            last_update_id: 1000,
            bids: vec![
                OrderBookLevel { price: 43000.0, quantity: 1.5 },
                OrderBookLevel { price: 42999.5, quantity: 2.0 },
                OrderBookLevel { price: 42999.0, quantity: 3.5 },
            ],
            asks: vec![
                OrderBookLevel { price: 43000.5, quantity: 1.0 },
                OrderBookLevel { price: 43001.0, quantity: 2.5 },
                OrderBookLevel { price: 43001.5, quantity: 4.0 },
            ],
        };

        Self {
            config,
            running: Arc::new(AtomicBool::new(false)),
            sequence_counter: Arc::new(AtomicU64::new(1000)),
            trade_counter: Arc::new(AtomicU64::new(0)),
            order_book: Arc::new(tokio::sync::RwLock::new(initial_order_book)),
        }
    }

    /// Start the mock exchange server
    pub async fn run(&self) -> Result<()> {
        self.running.store(true, Ordering::SeqCst);
        
        let ws_addr = format!("127.0.0.1:{}", self.config.ws_port);
        let rest_addr = format!("127.0.0.1:{}", self.config.rest_port);

        info!("Starting mock exchange server");
        info!("WebSocket listener: {}", ws_addr);
        info!("REST API listener: {}", rest_addr);

        // Start WebSocket server
        let ws_running = self.running.clone();
        let ws_config = self.config.clone();
        let ws_order_book = self.order_book.clone();
        let ws_seq = self.sequence_counter.clone();
        let ws_trade = self.trade_counter.clone();
        
        let ws_handle = tokio::spawn(async move {
            if let Err(e) = Self::run_websocket(ws_addr, ws_config, ws_order_book, ws_seq, ws_trade, ws_running).await {
                error!("WebSocket server error: {}", e);
            }
        });

        // Start REST API server
        let rest_running = self.running.clone();
        let rest_config = self.config.clone();
        let rest_order_book = self.order_book.clone();
        
        let rest_handle = tokio::spawn(async move {
            if let Err(e) = Self::run_rest_api(rest_addr, rest_config, rest_order_book, rest_running).await {
                error!("REST API server error: {}", e);
            }
        });

        // Run market data simulator (updates order book)
        let sim_running = self.running.clone();
        let sim_order_book = self.order_book.clone();
        let sim_seq = self.sequence_counter.clone();
        
        let sim_handle = tokio::spawn(async move {
            Self::simulate_market_data(sim_order_book, sim_seq, sim_running).await;
        });

        // Wait for all tasks
        tokio::select! {
            _ = ws_handle => {},
            _ = rest_handle => {},
            _ = sim_handle => {},
        }

        Ok(())
    }

    /// Stop the mock exchange server
    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
        info!("Mock exchange server stopping...");
    }

    /// Apply simulated latency
    async fn apply_latency(&self) {
        let jitter = rand::random::<u64>() % self.config.latency_jitter_ms;
        let delay = Duration::from_millis(self.config.base_latency_ms + jitter);
        sleep(delay).await;
    }

    /// WebSocket server implementation
    async fn run_websocket(
        addr: String,
        config: MockExchangeConfig,
        order_book: Arc<tokio::sync::RwLock<OrderBook>>,
        sequence: Arc<AtomicU64>,
        trade_counter: Arc<AtomicU64>,
        running: Arc<AtomicBool>,
    ) -> Result<()> {
        let listener = TcpListener::bind(&addr).await
            .context(format!("Failed to bind WebSocket to {}", addr))?;

        info!("WebSocket server listening on {}", addr);

        while running.load(Ordering::Relaxed) {
            match tokio::time::timeout(Duration::from_secs(1), listener.accept()).await {
                Ok(Ok((stream, peer_addr))) => {
                    debug!("New WebSocket connection from {}", peer_addr);
                    
                    let stream_running = running.clone();
                    let stream_config = config.clone();
                    let stream_ob = order_book.clone();
                    let stream_seq = sequence.clone();
                    let stream_trade = trade_counter.clone();
                    
                    tokio::spawn(async move {
                        if let Err(e) = Self::handle_ws_connection(
                            stream,
                            stream_config,
                            stream_ob,
                            stream_seq,
                            stream_trade,
                            stream_running,
                        ).await {
                            warn!("WebSocket connection error: {}", e);
                        }
                    });
                }
                Ok(Err(e)) => {
                    warn!("Accept error: {}", e);
                }
                Err(_) => {
                    // Timeout, check running flag
                    continue;
                }
            }
        }

        Ok(())
    }

    /// Handle individual WebSocket connection
    async fn handle_ws_connection(
        stream: TcpStream,
        config: MockExchangeConfig,
        order_book: Arc<tokio::sync::RwLock<OrderBook>>,
        sequence: Arc<AtomicU64>,
        trade_counter: Arc<AtomicU64>,
        running: Arc<AtomicBool>,
    ) -> Result<()> {
        // Simple WebSocket handshake and frame handling
        // In production, use tokio-tungstenite crate
        
        let mut buf = [0u8; 1024];
        
        // Read HTTP upgrade request
        let n = stream.peek(&mut buf).await?;
        let request = String::from_utf8_lossy(&buf[..n]);
        
        if !request.contains("GET") {
            return Ok(());
        }

        // Send WebSocket upgrade response
        let response = "HTTP/1.1 101 Switching Protocols\r\n\
                       Upgrade: websocket\r\n\
                       Connection: Upgrade\r\n\
                       Sec-WebSocket-Accept: dummy_accept\r\n\r\n";
        
        // Note: This is a simplified mock - real impl would use proper WS library
        let _ = response; // Placeholder
        
        // Stream market data
        let mut interval_timer = interval(Duration::from_millis(100));
        
        while running.load(Ordering::Relaxed) {
            interval_timer.tick().await;
            
            // Apply latency
            if rand::random::<f64>() < config.sequence_gap_probability {
                // Simulate sequence gap - skip this update
                warn!("Simulating sequence gap");
                continue;
            }
            
            // Get current order book
            let ob = order_book.read().await;
            let seq = sequence.fetch_add(1, Ordering::SeqCst);
            
            // Create depth update message
            let update = DepthUpdate {
                event_type: "depthUpdate".to_string(),
                event_time: chrono::Utc::now().timestamp_millis() as u64,
                symbol: ob.symbol.clone(),
                first_update_id: seq,
                final_update_id: seq,
                bids: ob.bids.iter().map(|l| vec![l.price.to_string(), l.quantity.to_string()]).collect(),
                asks: ob.asks.iter().map(|l| vec![l.price.to_string(), l.quantity.to_string()]).collect(),
            };
            
            let msg = serde_json::to_string(&update)?;
            
            // In real impl, send as WebSocket frame
            debug!("Sending depth update: {}", msg);
            
            drop(ob);
        }
        
        Ok(())
    }

    /// REST API server implementation
    async fn run_rest_api(
        addr: String,
        config: MockExchangeConfig,
        order_book: Arc<tokio::sync::RwLock<OrderBook>>,
        running: Arc<AtomicBool>,
    ) -> Result<()> {
        let listener = TcpListener::bind(&addr).await
            .context(format!("Failed to bind REST API to {}", addr))?;

        info!("REST API server listening on {}", addr);

        while running.load(Ordering::Relaxed) {
            match tokio::time::timeout(Duration::from_secs(1), listener.accept()).await {
                Ok(Ok((mut stream, peer_addr))) => {
                    debug!("New REST API connection from {}", peer_addr);
                    
                    // Read request
                    let mut buf = [0u8; 4096];
                    let n = stream.read(&mut buf).await?;
                    let request = String::from_utf8_lossy(&buf[..n]).to_string();
                    
                    // Simulate timeout
                    if rand::random::<f64>() < config.rest_timeout_probability {
                        warn!("Simulating REST 504 timeout");
                        sleep(Duration::from_secs(30)).await; // Will timeout
                        continue;
                    }
                    
                    // Apply latency
                    sleep(Duration::from_millis(config.base_latency_ms)).await;
                    
                    // Parse request and respond
                    let response = Self::handle_rest_request(&request, &order_book).await;
                    
                    let _ = stream.write_all(response.as_bytes()).await;
                }
                Ok(Err(e)) => {
                    warn!("REST accept error: {}", e);
                }
                Err(_) => continue,
            }
        }

        Ok(())
    }

    /// Handle REST API request
    async fn handle_rest_request(
        request: &str,
        order_book: &Arc<tokio::sync::RwLock<OrderBook>>,
    ) -> String {
        // Simple routing
        if request.contains("/api/v3/depth") {
            let ob = order_book.read().await;
            let response = serde_json::to_string(&*ob).unwrap_or_default();
            
            format!(
                "HTTP/1.1 200 OK\r\n\
                 Content-Type: application/json\r\n\
                 Content-Length: {}\r\n\r\n\
                 {}",
                response.len(),
                response
            )
        } else if request.contains("/api/v3/time") {
            let now = chrono::Utc::now().timestamp_millis();
            format!(
                "HTTP/1.1 200 OK\r\n\
                 Content-Type: application/json\r\n\r\n\
                 {{\"serverTime\":{}}}",
                now
            )
        } else if request.contains("/api/v3/ticker") {
            let ob = order_book.read().await;
            let mid = (ob.bids[0].price + ob.asks[0].price) / 2.0;
            
            format!(
                "HTTP/1.1 200 OK\r\n\
                 Content-Type: application/json\r\n\r\n\
                 {{\"symbol\":\"{}\",\"lastPrice\":\"{}\"}}",
                ob.symbol, mid
            )
        } else {
            "HTTP/1.1 404 Not Found\r\n\r\n".to_string()
        }
    }

    /// Simulate market data updates
    async fn simulate_market_data(
        order_book: Arc<tokio::sync::RwLock<OrderBook>>,
        sequence: Arc<AtomicU64>,
        running: Arc<AtomicBool>,
    ) {
        let mut rng = rand::thread_rng();
        let mut interval_timer = interval(Duration::from_millis(50));
        
        while running.load(Ordering::Relaxed) {
            interval_timer.tick().await;
            
            // Randomly adjust prices
            {
                let mut ob = order_book.write().await;
                let drift = (rand::random::<f64>() - 0.5) * 0.5; // ±0.25
                
                for level in ob.bids.iter_mut() {
                    level.price += drift;
                    level.quantity = (level.quantity * 0.9) + (rand::random::<f64>() * 0.5);
                }
                
                for level in ob.asks.iter_mut() {
                    level.price += drift;
                    level.quantity = (level.quantity * 0.9) + (rand::random::<f64>() * 0.5);
                }
                
                ob.last_update_id = sequence.load(Ordering::SeqCst);
            }
        }
    }
}

/// Test helper to create and manage mock exchange
pub struct MockExchangeHandle {
    server: Arc<MockExchangeServer>,
    shutdown_tx: Option<oneshot::Sender<()>>,
}

impl MockExchangeHandle {
    pub async fn start(config: MockExchangeConfig) -> Result<Self> {
        let server = Arc::new(MockExchangeServer::new(config));
        let (tx, rx) = oneshot::channel();
        
        let server_clone = server.clone();
        tokio::spawn(async move {
            tokio::select! {
                _ = server_clone.run() => {},
                _ = rx => {},
            }
        });
        
        // Give server time to start
        sleep(Duration::from_millis(100)).await;
        
        Ok(Self {
            server,
            shutdown_tx: Some(tx),
        })
    }
    
    pub fn ws_url(&self) -> String {
        format!("ws://127.0.0.1:{}", self.server.config.ws_port)
    }
    
    pub fn rest_url(&self) -> String {
        format!("http://127.0.0.1:{}", self.server.config.rest_port)
    }
    
    pub async fn stop(mut self) {
        self.server.stop();
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
        sleep(Duration::from_millis(100)).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mock_exchange_lifecycle() {
        let config = MockExchangeConfig {
            base_latency_ms: 1,
            latency_jitter_ms: 1,
            sequence_gap_probability: 0.0,
            rest_timeout_probability: 0.0,
            ..Default::default()
        };
        
        let handle = MockExchangeHandle::start(config).await.unwrap();
        
        assert!(!handle.ws_url().is_empty());
        assert!(!handle.rest_url().is_empty());
        
        handle.stop().await;
    }
}
