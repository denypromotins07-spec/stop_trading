//! Market Data Normalizer
//! 
//! Implements translation logic converting exchange-specific JSON payloads (e.g., Binance streams)
//! into unified internal types. Handles decimal precision normalization and unified symbol mapping.

use crate::market_data::types::*;
use anyhow::{Context, Result};

/// Symbol mapping registry for normalizing exchange symbols to internal format
pub struct SymbolRegistry {
    /// Mapping from exchange symbol (e.g., "BTCUSDT") to internal SymbolId
    exchange_to_internal: std::collections::HashMap<String, SymbolId>,
    /// Reverse mapping
    internal_to_exchange: std::collections::HashMap<SymbolId, String>,
}

impl SymbolRegistry {
    #[inline]
    pub fn new() -> Self {
        SymbolRegistry {
            exchange_to_internal: std::collections::HashMap::with_capacity(256),
            internal_to_exchange: std::collections::HashMap::with_capacity(256),
        }
    }

    #[inline]
    pub fn register(&mut self, exchange_symbol: &str, internal_symbol: &str) {
        let sym_id = SymbolId::from_str(internal_symbol);
        self.exchange_to_internal
            .insert(exchange_symbol.to_uppercase(), sym_id);
        self.internal_to_exchange.insert(sym_id, internal_symbol.to_string());
    }

    #[inline]
    pub fn get_internal(&self, exchange_symbol: &str) -> Option<SymbolId> {
        self.exchange_to_internal.get(&exchange_symbol.to_uppercase()).copied()
    }

    #[inline]
    pub fn get_exchange(&self, symbol: SymbolId) -> Option<&str> {
        self.internal_to_exchange.get(&symbol).map(|s| s.as_str())
    }
}

impl Default for SymbolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Decimal precision configuration per symbol
#[derive(Debug, Clone, Copy)]
pub struct PrecisionConfig {
    pub price_precision: u8,
    pub quantity_precision: u8,
    pub quote_precision: u8,
    pub base_asset_precision: u8,
}

impl PrecisionConfig {
    #[inline]
    pub const fn default_config() -> Self {
        PrecisionConfig {
            price_precision: 8,
            quantity_precision: 8,
            quote_precision: 8,
            base_asset_precision: 8,
        }
    }

    #[inline]
    pub fn price_multiplier(self) -> f64 {
        10_f64.powi(self.price_precision as i32)
    }

    #[inline]
    pub fn quantity_multiplier(self) -> f64 {
        10_f64.powi(self.quantity_precision as i32)
    }
}

/// Binance-specific stream message types
#[derive(Debug, serde::Deserialize)]
#[serde(tag = "e")]
pub enum BinanceStreamMessage {
    #[serde(rename = "trade")]
    Trade(BinanceTrade),
    #[serde(rename = "depthUpdate")]
    DepthUpdate(BinanceDepthUpdate),
    #[serde(rename = "kline")]
    Kline(BinanceKline),
    #[serde(rename = "ticker")]
    Ticker24h(BinanceTicker24h),
    #[serde(rename = "bookTicker")]
    BookTicker(BinanceBookTicker),
}

#[derive(Debug, serde::Deserialize)]
pub struct BinanceTrade {
    #[serde(rename = "s")]
    pub symbol: String,
    #[serde(rename = "t", deserialize_with = "deserialize_u64_from_str_or_num")]
    pub trade_id: u64,
    #[serde(rename = "p", deserialize_with = "deserialize_f64_from_str")]
    pub price: f64,
    #[serde(rename = "q", deserialize_with = "deserialize_f64_from_str")]
    pub quantity: f64,
    #[serde(rename = "b")]
    pub buyer_order_id: u64,
    #[serde(rename = "a")]
    pub seller_order_id: u64,
    #[serde(rename = "T")]
    pub timestamp: i64,
    #[serde(rename = "m")]
    pub is_buyer_maker: bool,
}

#[derive(Debug, serde::Deserialize)]
pub struct BinanceDepthUpdate {
    #[serde(rename = "s")]
    pub symbol: String,
    #[serde(rename = "U")]
    pub first_update_id: u64,
    #[serde(rename = "u")]
    pub last_update_id: u64,
    #[serde(rename = "b")]
    pub bids: Vec<(String, String)>,
    #[serde(rename = "a")]
    pub asks: Vec<(String, String)>,
}

#[derive(Debug, serde::Deserialize)]
pub struct BinanceKline {
    #[serde(rename = "s")]
    pub symbol: String,
    #[serde(rename = "k")]
    pub kline_data: BinanceKlineData,
}

#[derive(Debug, serde::Deserialize)]
pub struct BinanceKlineData {
    #[serde(rename = "t")]
    pub start_time: i64,
    #[serde(rename = "c", deserialize_with = "deserialize_f64_from_str")]
    pub close: f64,
    #[serde(rename = "o", deserialize_with = "deserialize_f64_from_str")]
    pub open: f64,
    #[serde(rename = "h", deserialize_with = "deserialize_f64_from_str")]
    pub high: f64,
    #[serde(rename = "l", deserialize_with = "deserialize_f64_from_str")]
    pub low: f64,
    #[serde(rename = "v", deserialize_with = "deserialize_f64_from_str")]
    pub volume: f64,
    #[serde(rename = "x")]
    pub is_final: bool,
}

#[derive(Debug, serde::Deserialize)]
pub struct BinanceTicker24h {
    #[serde(rename = "s")]
    pub symbol: String,
    #[serde(rename = "c", deserialize_with = "deserialize_f64_from_str")]
    pub last_price: f64,
    #[serde(rename = "b", deserialize_with = "deserialize_f64_from_str")]
    pub bid_price: f64,
    #[serde(rename = "a", deserialize_with = "deserialize_f64_from_str")]
    pub ask_price: f64,
    #[serde(rename = "v", deserialize_with = "deserialize_f64_from_str")]
    pub volume_24h: f64,
    #[serde(rename = "q", deserialize_with = "deserialize_f64_from_str")]
    pub quote_volume_24h: f64,
}

#[derive(Debug, serde::Deserialize)]
pub struct BinanceBookTicker {
    #[serde(rename = "s")]
    pub symbol: String,
    #[serde(rename = "b", deserialize_with = "deserialize_f64_from_str")]
    pub bid_price: f64,
    #[serde(rename = "B", deserialize_with = "deserialize_f64_from_str")]
    pub bid_quantity: f64,
    #[serde(rename = "a", deserialize_with = "deserialize_f64_from_str")]
    pub ask_price: f64,
    #[serde(rename = "A", deserialize_with = "deserialize_f64_from_str")]
    pub ask_quantity: f64,
}

/// Custom deserializer for f64 that handles both string and number formats
fn deserialize_f64_from_str<'de, D>(deserializer: D) -> Result<f64, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::Deserialize;
    
    #[derive(serde::Deserialize)]
    #[serde(untagged)]
    enum FloatOrString {
        Float(f64),
        String(String),
    }

    match FloatOrString::deserialize(deserializer)? {
        FloatOrString::Float(v) => Ok(v),
        FloatOrString::String(s) => s.parse::<f64>().map_err(serde::de::Error::custom),
    }
}

/// Custom deserializer for u64 that handles both string and number formats
fn deserialize_u64_from_str_or_num<'de, D>(deserializer: D) -> Result<u64, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::Deserialize;
    
    #[derive(serde::Deserialize)]
    #[serde(untagged)]
    enum IntOrString {
        Int(u64),
        String(String),
    }

    match IntOrString::deserialize(deserializer)? {
        IntOrString::Int(v) => Ok(v),
        IntOrString::String(s) => s.parse::<u64>().map_err(serde::de::Error::custom),
    }
}

/// Main normalizer struct - converts raw exchange data to unified types
pub struct Normalizer {
    registry: SymbolRegistry,
    precision_configs: std::collections::HashMap<SymbolId, PrecisionConfig>,
}

impl Normalizer {
    #[inline]
    pub fn new(registry: SymbolRegistry) -> Self {
        Normalizer {
            registry,
            precision_configs: std::collections::HashMap::new(),
        }
    }

    #[inline]
    pub fn with_precision(mut self, symbol: SymbolId, config: PrecisionConfig) -> Self {
        self.precision_configs.insert(symbol, config);
        self
    }

    /// Normalize a Binance trade message
    #[inline]
    pub fn normalize_trade(&self, msg: &BinanceTrade) -> Result<Trade> {
        let symbol = self.registry
            .get_internal(&msg.symbol)
            .context(format!("Unknown symbol: {}", msg.symbol))?;
        
        let config = self.precision_configs
            .get(&symbol)
            .unwrap_or(&PrecisionConfig::default_config());

        let side = if msg.is_buyer_maker { Side::Sell } else { Side::Buy };

        Ok(Trade {
            symbol,
            trade_id: msg.trade_id,
            price: Price::from_f64(msg.price),
            quantity: Quantity::from_f64(msg.quantity),
            side,
            timestamp_ns: msg.timestamp * 1_000_000, // ms to ns
            buyer_order_id: msg.buyer_order_id,
            seller_order_id: msg.seller_order_id,
        })
    }

    /// Normalize a Binance depth update message
    #[inline]
    pub fn normalize_depth_update(&self, msg: &BinanceDepthUpdate) -> Result<OrderBookDelta> {
        let symbol = self.registry
            .get_internal(&msg.symbol)
            .context(format!("Unknown symbol: {}", msg.symbol))?;

        let mut delta = OrderBookDelta::with_capacity(
            symbol,
            msg.last_update_id,
            msg.bids.len(),
            msg.asks.len(),
        );

        let config = self.precision_configs
            .get(&symbol)
            .unwrap_or(&PrecisionConfig::default_config());

        // Parse bids
        for (price_str, qty_str) in &msg.bids {
            let price = price_str.parse::<f64>()
                .context("Failed to parse bid price")?;
            let quantity = qty_str.parse::<f64>()
                .context("Failed to parse bid quantity")?;
            
            delta.bids.push(Level::new(
                Price::from_f64(price),
                Quantity::from_f64(quantity),
                1, // Order count unknown in delta
            ));
        }

        // Parse asks
        for (price_str, qty_str) in &msg.asks {
            let price = price_str.parse::<f64>()
                .context("Failed to parse ask price")?;
            let quantity = qty_str.parse::<f64>()
                .context("Failed to parse ask quantity")?;
            
            delta.asks.push(Level::new(
                Price::from_f64(price),
                Quantity::from_f64(quantity),
                1,
            ));
        }

        Ok(delta)
    }

    /// Normalize a Binance ticker 24h message
    #[inline]
    pub fn normalize_ticker_24h(&self, msg: &BinanceTicker24h) -> Result<Ticker> {
        let symbol = self.registry
            .get_internal(&msg.symbol)
            .context(format!("Unknown symbol: {}", msg.symbol))?;

        Ok(Ticker {
            symbol,
            last_price: Price::from_f64(msg.last_price),
            bid_price: Price::from_f64(msg.bid_price),
            ask_price: Price::from_f64(msg.ask_price),
            volume_24h: Quantity::from_f64(msg.volume_24h),
            quote_volume_24h: Quantity::from_f64(msg.quote_volume_24h),
            timestamp_ns: chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0),
            sequence: 0,
        })
    }

    /// Normalize a Binance book ticker message
    #[inline]
    pub fn normalize_book_ticker(&self, msg: &BinanceBookTicker) -> Result<Ticker> {
        let symbol = self.registry
            .get_internal(&msg.symbol)
            .context(format!("Unknown symbol: {}", msg.symbol))?;

        Ok(Ticker {
            symbol,
            last_price: Price::from_f64((msg.bid_price + msg.ask_price) / 2.0),
            bid_price: Price::from_f64(msg.bid_price),
            ask_price: Price::from_f64(msg.ask_price),
            volume_24h: Quantity::new(0),
            quote_volume_24h: Quantity::new(0),
            timestamp_ns: chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0),
            sequence: 0,
        })
    }

    /// Get the symbol registry reference
    #[inline]
    pub fn registry(&self) -> &SymbolRegistry {
        &self.registry
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symbol_registry() {
        let mut registry = SymbolRegistry::new();
        registry.register("BTCUSDT", "BTC-USD");
        
        let sym = registry.get_internal("btcusdt").unwrap();
        assert_eq!(sym.as_str(), "BTC-USD");
    }

    #[test]
    fn test_normalizer_trade() {
        let mut registry = SymbolRegistry::new();
        registry.register("BTCUSDT", "BTC-USD");
        
        let normalizer = Normalizer::new(registry);
        
        let trade = BinanceTrade {
            symbol: "BTCUSDT".to_string(),
            trade_id: 12345,
            price: 50000.50,
            quantity: 0.001,
            buyer_order_id: 100,
            seller_order_id: 200,
            timestamp: 1700000000000,
            is_buyer_maker: false,
        };

        let normalized = normalizer.normalize_trade(&trade).unwrap();
        assert_eq!(normalized.symbol.as_str(), "BTC-USD");
        assert_eq!(normalized.side, Side::Buy);
    }
}
