//! Market Data Module Root
//! 
//! Exports public interfaces and defines the unified symbol registry.

pub mod types;
pub mod normalizer;

pub use types::*;
pub use normalizer::*;

use std::sync::OnceLock;

/// Global symbol registry singleton for consistent symbol mapping across the application
static GLOBAL_SYMBOL_REGISTRY: OnceLock<std::sync::RwLock<SymbolRegistry>> = OnceLock::new();

/// Initialize the global symbol registry with default mappings
#[inline]
pub fn init_global_registry() -> &'static std::sync::RwLock<SymbolRegistry> {
    GLOBAL_SYMBOL_REGISTRY.get_or_init(|| {
        let mut registry = SymbolRegistry::new();
        
        // Register common Binance symbols
        registry.register("BTCUSDT", "BTC-USD");
        registry.register("ETHUSDT", "ETH-USD");
        registry.register("BNBUSDT", "BNB-USD");
        registry.register("SOLUSDT", "SOL-USD");
        registry.register("XRPUSDT", "XRP-USD");
        registry.register("ADAUSDT", "ADA-USD");
        registry.register("DOGEUSDT", "DOGE-USD");
        registry.register("AVAXUSDT", "AVAX-USD");
        registry.register("LINKUSDT", "LINK-USD");
        registry.register("DOTUSDT", "DOT-USD");
        
        std::sync::RwLock::new(registry)
    })
}

/// Get a reference to the global symbol registry
#[inline]
pub fn get_global_registry() -> Option<&'static std::sync::RwLock<SymbolRegistry>> {
    GLOBAL_SYMBOL_REGISTRY.get()
}

/// Create a new normalizer with the global registry
#[inline]
pub fn create_normalizer() -> Normalizer {
    let registry = GLOBAL_SYMBOL_REGISTRY
        .get()
        .map(|lock| lock.read().unwrap().clone())
        .unwrap_or_else(SymbolRegistry::new);
    
    Normalizer::new(registry)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_global_registry_init() {
        let registry = init_global_registry();
        let read_guard = registry.read().unwrap();
        assert!(read_guard.get_internal("BTCUSDT").is_some());
    }
}
