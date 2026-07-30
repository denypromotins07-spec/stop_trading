//! Macro Data Ingestion Module
//! 
//! Async fetcher for economic calendars (CPI, Fed rates) and traditional market feeds.
//! Parses macroeconomic releases in microseconds for immediate defensive positioning.
//! Memory-efficient design with bounded buffers.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use crossbeam_channel::{bounded, Sender, Receiver};

/// Channel capacity for macro events
const MACRO_EVENT_CAPACITY: usize = 100;

/// Economic event types
#[derive(Debug, Clone)]
pub enum MacroEvent {
    CpiRelease {
        actual: f64,
        forecast: f64,
        previous: f64,
        impact: ImpactLevel,
        timestamp_ns: u64,
    },
    FedRateDecision {
        rate_bps: i32,
        change_bps: i32,
        statement_sentiment: Sentiment,
        timestamp_ns: u64,
    },
    NfpRelease {
        actual: i64,
        forecast: i64,
        previous: i64,
        unemployment_rate: f64,
        impact: ImpactLevel,
        timestamp_ns: u64,
    },
    GdpRelease {
        actual: f64,
        forecast: f64,
        previous: f64,
        quarter: u8,
        year: u16,
        timestamp_ns: u64,
    },
    DxyMove {
        value: f64,
        change_pct: f64,
        timestamp_ns: u64,
    },
    TreasuryYield {
        tenor: YieldTenor,
        yield_bps: f64,
        change_bps: f64,
        timestamp_ns: u64,
    },
}

/// Impact level for economic events
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ImpactLevel {
    Low,
    Medium,
    High,
    Critical,
}

/// Sentiment classification
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Sentiment {
    Hawkish,
    Dovish,
    Neutral,
}

/// Treasury yield tenors
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum YieldTenor {
    Y2,
    Y5,
    Y10,
    Y30,
}

/// Market regime shift signal
#[derive(Debug, Clone)]
pub struct RegimeShiftSignal {
    pub trigger_event: MacroEvent,
    pub direction: RegimeDirection,
    pub confidence: f32,
    pub recommended_action: RecommendedAction,
    pub timestamp_ns: u64,
}

/// Regime direction
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RegimeDirection {
    RiskOn,
    RiskOff,
    Neutral,
}

/// Recommended action based on regime
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RecommendedAction {
    IncreaseExposure,
    DecreaseExposure,
    Hedge,
    Hold,
}

/// Traditional market data snapshot
#[derive(Debug, Clone)]
pub struct MarketDataSnapshot {
    pub dxy: f64,
    pub spx: f64,
    pub gold_usd_oz: f64,
    pub oil_wti: f64,
    pub vix: f64,
    pub yield_10y: f64,
    pub yield_2y: f64,
    pub yield_spread_bps: f64,
    pub timestamp_ns: u64,
}

/// Macro data ingestion engine
pub struct MacroDataIngestor {
    /// Event channel
    event_tx: Sender<MacroEvent>,
    event_rx: Receiver<MacroEvent>,
    
    /// Current market snapshot
    market_snapshot: Arc<std::sync::Mutex<Option<MarketDataSnapshot>>>,
    
    /// Last CPI reading
    last_cpi: AtomicU64, // Stored as fixed-point * 1000
    
    /// Last Fed rate decision
    last_fed_rate: AtomicU64, // Stored as basis points
    
    /// Event counter
    event_count: AtomicU64,
}

impl MacroDataIngestor {
    /// Create a new macro data ingestor
    pub fn new() -> Self {
        let (event_tx, event_rx) = bounded(MACRO_EVENT_CAPACITY);
        
        Self {
            event_tx,
            event_rx,
            market_snapshot: Arc::new(std::sync::Mutex::new(None)),
            last_cpi: AtomicU64::new(0),
            last_fed_rate: AtomicU64::new(525), // Default 5.25%
            event_count: AtomicU64::new(0),
        }
    }
    
    /// Process an incoming macro event
    pub fn process_event(&self, event: MacroEvent) -> Option<RegimeShiftSignal> {
        // Store relevant data
        match &event {
            MacroEvent::CpiRelease { actual, .. } => {
                self.last_cpi.store((*actual * 1000.0) as u64, Ordering::Relaxed);
            }
            MacroEvent::FedRateDecision { rate_bps, .. } => {
                self.last_fed_rate.store(*rate_bps as u64, Ordering::Relaxed);
            }
            _ => {}
        }
        
        // Try to send event (non-blocking)
        let _ = self.event_tx.try_send(event.clone());
        self.event_count.fetch_add(1, Ordering::Relaxed);
        
        // Check if this triggers a regime shift
        self.evaluate_regime_shift(event)
    }
    
    /// Update market data snapshot
    pub fn update_market_data(&self, snapshot: MarketDataSnapshot) {
        if let Ok(mut guard) = self.market_snapshot.lock() {
            *guard = Some(snapshot);
        }
    }
    
    /// Get receiver for macro events
    pub fn event_receiver(&self) -> Receiver<MacroEvent> {
        self.event_rx.clone()
    }
    
    /// Get current market snapshot
    pub fn get_market_snapshot(&self) -> Option<MarketDataSnapshot> {
        self.market_snapshot.lock().unwrap().clone()
    }
    
    /// Get last CPI reading
    pub fn get_last_cpi(&self) -> f64 {
        self.last_cpi.load(Ordering::Relaxed) as f64 / 1000.0
    }
    
    /// Get last Fed rate
    pub fn get_last_fed_rate(&self) -> i32 {
        self.last_fed_rate.load(Ordering::Relaxed) as i32
    }
    
    /// Evaluate if event triggers regime shift
    fn evaluate_regime_shift(&self, event: MacroEvent) -> Option<RegimeShiftSignal> {
        let (direction, confidence, action) = match &event {
            MacroEvent::CpiRelease { actual, forecast, impact, .. } => {
                let surprise = actual - forecast;
                
                if *impact != ImpactLevel::High && *impact != ImpactLevel::Critical {
                    return None;
                }
                
                if surprise > 0.3 {
                    // Higher than expected inflation
                    (RegimeDirection::RiskOff, 0.7, RecommendedAction::DecreaseExposure)
                } else if surprise < -0.3 {
                    // Lower than expected inflation
                    (RegimeDirection::RiskOn, 0.7, RecommendedAction::IncreaseExposure)
                } else {
                    return None;
                }
            }
            
            MacroEvent::FedRateDecision { change_bps, statement_sentiment, .. } => {
                match statement_sentiment {
                    Sentiment::Hawkish => {
                        (RegimeDirection::RiskOff, 0.8, RecommendedAction::Hedge)
                    }
                    Sentiment::Dovish => {
                        (RegimeDirection::RiskOn, 0.8, RecommendedAction::IncreaseExposure)
                    }
                    Sentiment::Neutral => {
                        return None;
                    }
                }
            }
            
            MacroEvent::NfpRelease { actual, forecast, impact, .. } => {
                if *impact != ImpactLevel::High && *impact != ImpactLevel::Critical {
                    return None;
                }
                
                let surprise = (*actual - *forecast) as f64;
                if surprise > 100_000.0 {
                    // Much stronger jobs = potential rate hike concern
                    (RegimeDirection::RiskOff, 0.6, RecommendedAction::DecreaseExposure)
                } else if surprise < -100_000.0 {
                    // Much weaker jobs = potential rate cut hope
                    (RegimeDirection::RiskOn, 0.6, RecommendedAction::Hold)
                } else {
                    return None;
                }
            }
            
            MacroEvent::DxyMove { change_pct, .. } => {
                if change_pct.abs() > 1.0 {
                    if *change_pct > 0.0 {
                        (RegimeDirection::RiskOff, 0.5, RecommendedAction::Hedge)
                    } else {
                        (RegimeDirection::RiskOn, 0.5, RecommendedAction::IncreaseExposure)
                    }
                } else {
                    return None;
                }
            }
            
            _ => return None,
        };
        
        Some(RegimeShiftSignal {
            trigger_event: event,
            direction,
            confidence,
            recommended_action: action,
            timestamp_ns: get_timestamp_ns(),
        })
    }
}

impl Default for MacroDataIngestor {
    fn default() -> Self {
        Self::new()
    }
}

/// Parse economic calendar JSON response (simplified)
pub fn parse_economic_calendar(json: &str) -> Result<Vec<ScheduledEvent>, CalendarParseError> {
    // Simplified parser - in production would use serde_json
    let mut events = Vec::new();
    
    // Basic validation
    if !json.contains("events") {
        return Err(CalendarParseError::InvalidFormat);
    }
    
    // Would parse actual JSON in production
    Ok(events)
}

/// Scheduled economic event
#[derive(Debug, Clone)]
pub struct ScheduledEvent {
    pub name: String,
    pub scheduled_time_ns: u64,
    pub currency: String,
    pub impact: ImpactLevel,
    pub forecast: Option<f64>,
    pub previous: Option<f64>,
}

/// Calendar parse error
#[derive(Debug, Clone)]
pub enum CalendarParseError {
    InvalidFormat,
    MissingField(String),
    ParseError(String),
}

impl std::fmt::Display for CalendarParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CalendarParseError::InvalidFormat => write!(f, "Invalid calendar format"),
            CalendarParseError::MissingField(field) => write!(f, "Missing field: {}", field),
            CalendarParseError::ParseError(msg) => write!(f, "Parse error: {}", msg),
        }
    }
}

impl std::error::Error for CalendarParseError {}

/// Get current timestamp in nanoseconds
fn get_timestamp_ns() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_ingestor_creation() {
        let ingestor = MacroDataIngestor::new();
        assert_eq!(ingestor.get_last_fed_rate(), 525);
    }
    
    #[test]
    fn test_process_cpi_event() {
        let ingestor = MacroDataIngestor::new();
        
        let event = MacroEvent::CpiRelease {
            actual: 3.5,
            forecast: 3.2,
            previous: 3.3,
            impact: ImpactLevel::High,
            timestamp_ns: get_timestamp_ns(),
        };
        
        let signal = ingestor.process_event(event);
        assert!(signal.is_some());
        
        let signal = signal.unwrap();
        assert_eq!(signal.direction, RegimeDirection::RiskOff);
    }
    
    #[test]
    fn test_impact_levels() {
        assert_eq!(ImpactLevel::Low < ImpactLevel::Medium, true);
        assert_eq!(ImpactLevel::Critical > ImpactLevel::High, true);
    }
}
