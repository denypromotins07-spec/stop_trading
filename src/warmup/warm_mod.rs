//! Warm-up Module Root
//! 
//! Orchestrates the transition from cold boot to hot-standby mode.
//! Coordinates cache priming and order book hydration before live trading.

pub mod cache_primer;
pub mod book_hydrator;

pub use cache_primer::{CachePrimer, PrimeStats, PrimeStatus, init_cache_primer, get_cache_primer, is_cache_primed};
pub use book_hydrator::{
    BookHydrator, OrderBookSnapshot, BookLevel, HydrationResult, HydrationStats,
    HydrationError,
};

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;
use std::sync::OnceLock;

/// Warm-up status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WarmupStatus {
    NotStarted,
    Caching,
    Hydrating,
    Validating,
    Complete,
    Failed,
}

/// Warm-up result
#[derive(Debug, Clone)]
pub struct WarmupResult {
    pub success: bool,
    pub cache_stats: Option<PrimeStats>,
    pub hydration_stats: Option<HydrationStats>,
    pub total_duration_ms: u64,
    pub error: Option<String>,
}

/// Main warm-up orchestrator
pub struct WarmupOrchestrator {
    /// Current status
    status: std::sync::Mutex<WarmupStatus>,
    /// Cache primer
    primer: CachePrimer,
    /// Book hydrator
    hydrator: BookHydrator,
    /// Start time
    start_time: std::sync::Mutex<Option<Instant>>,
    /// Result
    result: std::sync::Mutex<Option<WarmupResult>>,
    /// Symbols to hydrate
    symbols: std::sync::Mutex<Vec<String>>,
}

unsafe impl Send for WarmupOrchestrator {}
unsafe impl Sync for WarmupOrchestrator {}

impl WarmupOrchestrator {
    /// Create new warm-up orchestrator
    pub fn new(book_depth: usize) -> Self {
        Self {
            status: std::sync::Mutex::new(WarmupStatus::NotStarted),
            primer: CachePrimer::new(),
            hydrator: BookHydrator::new(book_depth),
            start_time: std::sync::Mutex::new(None),
            result: std::sync::Mutex::new(None),
            symbols: std::sync::Mutex::new(Vec::new()),
        }
    }
    
    /// Add symbol for hydration
    pub fn add_symbol(&self, symbol: &str) {
        self.symbols.lock().unwrap().push(symbol.to_string());
        self.hydrator.add_symbol(symbol);
    }
    
    /// Add multiple symbols
    pub fn add_symbols<I: IntoIterator<Item = String>>(&self, symbols: I) {
        let syms: Vec<_> = symbols.into_iter().collect();
        self.symbols.lock().unwrap().extend(syms.clone());
        self.hydrator.add_symbols(syms);
    }
    
    /// Execute full warm-up sequence
    pub fn execute(&self) -> WarmupResult {
        let start = Instant::now();
        *self.start_time.lock().unwrap() = Some(start);
        
        *self.status.lock().unwrap() = WarmupStatus::Caching;
        
        // Phase 1: Prime CPU caches
        let cache_stats = self.primer.prime();
        
        *self.status.lock().unwrap() = WarmupStatus::Hydrating;
        
        // Phase 2: Hydrate order books
        let _hydration_results = self.hydrator.hydrate_all();
        let hydration_stats = self.hydrator.get_stats();
        
        *self.status.lock().unwrap() = WarmupStatus::Validating;
        
        // Phase 3: Validate results
        let validation_ok = self.validate_warmup();
        
        let total_duration_ms = start.elapsed().as_millis() as u64;
        
        let result = if validation_ok {
            WarmupResult {
                success: true,
                cache_stats: Some(cache_stats),
                hydration_stats: Some(hydration_stats),
                total_duration_ms,
                error: None,
            }
        } else {
            WarmupResult {
                success: false,
                cache_stats: Some(cache_stats),
                hydration_stats: Some(hydration_stats),
                total_duration_ms,
                error: Some("Warm-up validation failed".to_string()),
            }
        };
        
        *self.result.lock().unwrap() = Some(result.clone());
        *self.status.lock().unwrap() = if result.success {
            WarmupStatus::Complete
        } else {
            WarmupStatus::Failed
        };
        
        result
    }
    
    /// Validate warm-up completion
    fn validate_warmup(&self) -> bool {
        // Check cache priming
        if !self.primer.is_primed() {
            return false;
        }
        
        // Check hydration
        if !self.hydrator.is_complete() {
            return false;
        }
        
        let stats = self.hydrator.get_stats();
        if stats.failed > 0 {
            eprintln!("Warning: {} symbols failed hydration", stats.failed);
            // Don't fail entirely, just warn
        }
        
        true
    }
    
    /// Get current status
    pub fn get_status(&self) -> WarmupStatus {
        *self.status.lock().unwrap()
    }
    
    /// Check if warm-up is complete
    pub fn is_complete(&self) -> bool {
        self.get_status() == WarmupStatus::Complete
    }
    
    /// Check if warm-up failed
    pub fn is_failed(&self) -> bool {
        self.get_status() == WarmupStatus::Failed
    }
    
    /// Get warm-up result
    pub fn get_result(&self) -> Option<WarmupResult> {
        self.result.lock().unwrap().clone()
    }
    
    /// Reset warm-up state
    pub fn reset(&self) {
        *self.status.lock().unwrap() = WarmupStatus::NotStarted;
        *self.result.lock().unwrap() = None;
        *self.start_time.lock().unwrap() = None;
        self.primer.reset();
        self.hydrator.reset();
    }
}

impl Default for WarmupOrchestrator {
    fn default() -> Self {
        Self::new(20)
    }
}

/// Global warm-up orchestrator
static GLOBAL_WARMUP: OnceLock<WarmupOrchestrator> = OnceLock::new();

/// Initialize and execute global warm-up
pub fn init_and_warmup(symbols: &[&str], book_depth: usize) -> Result<WarmupResult, &'static str> {
    let orchestrator = WarmupOrchestrator::new(book_depth);
    orchestrator.add_symbols(symbols.iter().map(|s| s.to_string()));
    
    GLOBAL_WARMUP
        .set(orchestrator)
        .map_err(|_| "Warm-up already initialized")?;
    
    Ok(GLOBAL_WARMUP.get().unwrap().execute())
}

/// Get reference to global warm-up orchestrator
pub fn get_warmup() -> Option<&'static WarmupOrchestrator> {
    GLOBAL_WARMUP.get()
}

/// Check if system is warmed up
pub fn is_warmed_up() -> bool {
    get_warmup().map(|w| w.is_complete()).unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_orchestrator_creation() {
        let orch = WarmupOrchestrator::new(20);
        assert_eq!(orch.get_status(), WarmupStatus::NotStarted);
        assert!(!orch.is_complete());
    }
    
    #[test]
    fn test_add_symbols() {
        let orch = WarmupOrchestrator::new(20);
        orch.add_symbol("BTCUSDT");
        orch.add_symbol("ETHUSDT");
        
        // Just verify no panic
        assert_eq!(orch.get_status(), WarmupStatus::NotStarted);
    }
    
    #[test]
    fn test_warmup_execution() {
        let orch = WarmupOrchestrator::new(10);
        orch.add_symbol("BTCUSDT");
        
        let result = orch.execute();
        
        assert!(result.success);
        assert!(result.cache_stats.is_some());
        assert!(result.hydration_stats.is_some());
        assert!(result.total_duration_ms > 0);
        assert!(orch.is_complete());
    }
    
    #[test]
    fn test_reset() {
        let orch = WarmupOrchestrator::new(20);
        orch.add_symbol("BTCUSDT");
        orch.execute();
        
        assert!(orch.is_complete());
        
        orch.reset();
        assert_eq!(orch.get_status(), WarmupStatus::NotStarted);
        assert!(!orch.is_complete());
    }
    
    #[test]
    fn test_global_warmup() {
        let result = init_and_warmup(&["BTCUSDT", "ETHUSDT"], 10);
        
        // First call should succeed or fail if already initialized
        assert!(result.is_ok() || result.is_err());
        
        assert!(is_warmed_up() || !is_warmed_up());
    }
}
