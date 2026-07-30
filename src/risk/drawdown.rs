//! Dynamic peak-to-trough drawdown tracker with automated circuit breakers.
//! 
//! Scales down position sizing linearly as drawdown approaches maximum acceptable threshold.

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering, AtomicF64};

/// Drawdown state
#[derive(Debug, Clone)]
pub struct DrawdownState {
    /// Current drawdown (as fraction, negative value)
    pub current_drawdown: f64,
    /// Maximum drawdown seen
    pub max_drawdown: f64,
    /// Peak portfolio value
    pub peak_value: f64,
    /// Current portfolio value
    pub current_value: f64,
    /// Time since peak (in updates)
    pub time_since_peak: u64,
    /// Whether in drawdown
    pub is_drawdown: bool,
}

/// Circuit breaker tier
#[derive(Debug, Clone)]
pub struct CircuitBreakerTier {
    /// Drawdown threshold that triggers this tier
    pub threshold: f64,
    /// Position size reduction (0.0 to 1.0)
    pub size_reduction: f64,
    /// Trading halt duration (in milliseconds)
    pub halt_duration_ms: u64,
    /// Requires manual reset
    pub requires_manual_reset: bool,
}

/// Default circuit breaker tiers
pub mod default_tiers {
    use super::CircuitBreakerTier;
    
    /// Tier 1: Mild drawdown - reduce size by 25%
    pub const TIER_1: CircuitBreakerTier = CircuitBreakerTier {
        threshold: -0.05, // 5% drawdown
        size_reduction: 0.25,
        halt_duration_ms: 0, // No halt
        requires_manual_reset: false,
    };
    
    /// Tier 2: Moderate drawdown - reduce size by 50%
    pub const TIER_2: CircuitBreakerTier = CircuitBreakerTier {
        threshold: -0.10, // 10% drawdown
        size_reduction: 0.50,
        halt_duration_ms: 60000, // 1 minute halt
        requires_manual_reset: false,
    };
    
    /// Tier 3: Severe drawdown - reduce size by 75%
    pub const TIER_3: CircuitBreakerTier = CircuitBreakerTier {
        threshold: -0.15, // 15% drawdown
        size_reduction: 0.75,
        halt_duration_ms: 300000, // 5 minute halt
        requires_manual_reset: false,
    };
    
    /// Tier 4: Critical drawdown - halt trading
    pub const TIER_4: CircuitBreakerTier = CircuitBreakerTier {
        threshold: -0.20, // 20% drawdown
        size_reduction: 1.0,
        halt_duration_ms: 3600000, // 1 hour halt
        requires_manual_reset: true,
    };
    
    /// All default tiers
    pub const ALL: &[&CircuitBreakerTier] = &[&TIER_1, &TIER_2, &TIER_3, &TIER_4];
}

/// Circuit breaker status
#[derive(Debug, Clone)]
pub struct CircuitBreakerStatus {
    /// Whether any circuit breaker is active
    pub is_active: bool,
    /// Active tier level (0 if none)
    pub active_tier: usize,
    /// Current position size multiplier
    pub size_multiplier: f64,
    /// Halt end timestamp (nanoseconds)
    pub halt_end_ns: Option<u64>,
    /// Requires manual reset
    pub requires_manual_reset: bool,
}

impl CircuitBreakerStatus {
    /// Create inactive status
    pub fn inactive() -> Self {
        Self {
            is_active: false,
            active_tier: 0,
            size_multiplier: 1.0,
            halt_end_ns: None,
            requires_manual_reset: false,
        }
    }
}

/// Dynamic drawdown tracker with circuit breakers
pub struct DrawdownTracker {
    /// Peak portfolio value
    peak_value: f64,
    /// Current portfolio value
    current_value: f64,
    /// Maximum drawdown observed
    max_drawdown: f64,
    /// Update counter
    update_count: AtomicU64,
    /// Time since peak
    time_since_peak: AtomicU64,
    /// Circuit breaker tiers
    tiers: Vec<CircuitBreakerTier>,
    /// Current circuit breaker status
    breaker_status: CircuitBreakerStatus,
    /// Halt end timestamp
    halt_end_ns: AtomicU64,
    /// Manually overridden
    manually_overridden: AtomicBool,
    /// Maximum allowed drawdown from config
    max_allowed_drawdown: f64,
}

impl DrawdownTracker {
    /// Create a new drawdown tracker
    pub fn new(initial_value: f64, max_allowed_drawdown: f64) -> Self {
        Self {
            peak_value: initial_value,
            current_value: initial_value,
            max_drawdown: 0.0,
            update_count: AtomicU64::new(0),
            time_since_peak: AtomicU64::new(0),
            tiers: default_tiers::ALL.iter().map(|&t| (*t).clone()).collect(),
            breaker_status: CircuitBreakerStatus::inactive(),
            halt_end_ns: AtomicU64::new(0),
            manually_overridden: AtomicBool::new(false),
            max_allowed_drawdown,
        }
    }
    
    /// Update with new portfolio value
    #[inline]
    pub fn update(&mut self, current_value: f64) -> DrawdownState {
        self.current_value = current_value;
        self.update_count.fetch_add(1, Ordering::Relaxed);
        
        // Check for new peak
        if current_value > self.peak_value {
            self.peak_value = current_value;
            self.time_since_peak.store(0, Ordering::Relaxed);
        } else {
            self.time_since_peak.fetch_add(1, Ordering::Relaxed);
        }
        
        // Calculate current drawdown
        let drawdown = (current_value - self.peak_value) / self.peak_value;
        
        // Update max drawdown
        if drawdown < self.max_drawdown {
            self.max_drawdown = drawdown;
        }
        
        // Update circuit breaker status
        self.update_circuit_breaker(drawdown);
        
        DrawdownState {
            current_drawdown: drawdown,
            max_drawdown: self.max_drawdown,
            peak_value: self.peak_value,
            current_value: self.current_value,
            time_since_peak: self.time_since_peak.load(Ordering::Relaxed),
            is_drawdown: drawdown < 0.0,
        }
    }
    
    /// Update circuit breaker status based on drawdown
    fn update_circuit_breaker(&mut self, drawdown: f64) {
        // Check if halt has expired
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        
        if self.halt_end_ns.load(Ordering::Relaxed) > 0 && now >= self.halt_end_ns.load(Ordering::Relaxed) {
            // Halt expired, check if manual reset required
            if self.breaker_status.requires_manual_reset && !self.manually_overridden.load(Ordering::Relaxed) {
                return; // Still halted pending manual reset
            }
            self.halt_end_ns.store(0, Ordering::Relaxed);
        }
        
        // Find active tier
        let mut active_tier = 0;
        let mut size_multiplier = 1.0;
        let mut requires_manual = false;
        
        for (i, tier) in self.tiers.iter().enumerate() {
            if drawdown <= tier.threshold {
                active_tier = i + 1;
                size_multiplier = 1.0 - tier.size_reduction;
                requires_manual = tier.requires_manual_reset;
                
                // Set halt if applicable
                if tier.halt_duration_ms > 0 && self.halt_end_ns.load(Ordering::Relaxed) == 0 {
                    let halt_end = now + (tier.halt_duration_ms * 1_000_000);
                    self.halt_end_ns.store(halt_end, Ordering::Relaxed);
                }
                
                break;
            }
        }
        
        self.breaker_status = CircuitBreakerStatus {
            is_active: active_tier > 0,
            active_tier,
            size_multiplier,
            halt_end_ns: if self.halt_end_ns.load(Ordering::Relaxed) > now {
                Some(self.halt_end_ns.load(Ordering::Relaxed))
            } else {
                None
            },
            requires_manual_reset: requires_manual,
        };
    }
    
    /// Get current circuit breaker status
    #[inline]
    pub fn circuit_breaker_status(&self) -> &CircuitBreakerStatus {
        &self.breaker_status
    }
    
    /// Check if trading is allowed
    #[inline]
    pub fn can_trade(&self) -> bool {
        if self.manually_overridden.load(Ordering::Relaxed) {
            return true;
        }
        
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        
        // Check if in halt period
        if self.halt_end_ns.load(Ordering::Relaxed) > now {
            return false;
        }
        
        // Check if max drawdown exceeded
        let drawdown = (self.current_value - self.peak_value) / self.peak_value;
        if drawdown <= self.max_allowed_drawdown {
            return false;
        }
        
        !self.breaker_status.is_active || self.breaker_status.size_multiplier > 0.0
    }
    
    /// Get position size multiplier
    #[inline]
    pub fn size_multiplier(&self) -> f64 {
        if self.manually_overridden.load(Ordering::Relaxed) {
            return 1.0;
        }
        self.breaker_status.size_multiplier
    }
    
    /// Get current drawdown
    #[inline]
    pub fn current_drawdown(&self) -> f64 {
        (self.current_value - self.peak_value) / self.peak_value
    }
    
    /// Get max drawdown
    #[inline]
    pub fn max_drawdown(&self) -> f64 {
        self.max_drawdown
    }
    
    /// Manual override of circuit breaker (for emergency situations)
    pub fn manual_override(&self, override_active: bool) {
        self.manually_overridden.store(override_active, Ordering::Relaxed);
    }
    
    /// Reset circuit breaker (after manual review)
    pub fn reset_breaker(&mut self) {
        self.halt_end_ns.store(0, Ordering::Relaxed);
        self.manually_overridden.store(false, Ordering::Relaxed);
        self.breaker_status = CircuitBreakerStatus::inactive();
    }
    
    /// Reset tracker to initial state
    pub fn reset(&mut self, new_value: f64) {
        self.peak_value = new_value;
        self.current_value = new_value;
        self.max_drawdown = 0.0;
        self.time_since_peak.store(0, Ordering::Relaxed);
        self.reset_breaker();
    }
    
    /// Get drawdown state
    pub fn state(&self) -> DrawdownState {
        let drawdown = self.current_drawdown();
        DrawdownState {
            current_drawdown: drawdown,
            max_drawdown: self.max_drawdown,
            peak_value: self.peak_value,
            current_value: self.current_value,
            time_since_peak: self.time_since_peak.load(Ordering::Relaxed),
            is_drawdown: drawdown < 0.0,
        }
    }
}

/// Drawdown-based position scaler
pub struct PositionScaler {
    tracker: DrawdownTracker,
    /// Base position size
    base_size: f64,
}

impl PositionScaler {
    /// Create a new position scaler
    pub fn new(initial_value: f64, base_size: f64, max_drawdown: f64) -> Self {
        Self {
            tracker: DrawdownTracker::new(initial_value, max_drawdown),
            base_size,
        }
    }
    
    /// Calculate scaled position size
    pub fn calculate_size(&mut self, current_value: f64) -> f64 {
        let _state = self.tracker.update(current_value);
        
        if !self.tracker.can_trade() {
            return 0.0;
        }
        
        self.base_size * self.tracker.size_multiplier()
    }
    
    /// Get the underlying tracker
    pub fn tracker(&self) -> &DrawdownTracker {
        &self.tracker
    }
    
    /// Get mutable tracker
    pub fn tracker_mut(&mut self) -> &mut DrawdownTracker {
        &mut self.tracker
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_drawdown_tracking() {
        let mut tracker = DrawdownTracker::new(1_000_000.0, -0.25);
        
        // Simulate drawdown
        let state = tracker.update(950_000.0);
        assert!(state.is_drawdown);
        assert!((state.current_drawdown + 0.05).abs() < 0.001);
        
        // New peak
        let state = tracker.update(1_100_000.0);
        assert!(!state.is_drawdown);
        assert_eq!(state.peak_value, 1_100_000.0);
    }
    
    #[test]
    fn test_circuit_breaker() {
        let mut tracker = DrawdownTracker::new(1_000_000.0, -0.25);
        
        // Trigger tier 1 (5% drawdown)
        tracker.update(950_000.0);
        let status = tracker.circuit_breaker_status();
        assert!(status.is_active);
        assert!(status.size_multiplier < 1.0);
        
        // Check trading still allowed
        assert!(tracker.can_trade());
    }
    
    #[test]
    fn test_position_scaler() {
        let mut scaler = PositionScaler::new(1_000_000.0, 100.0, -0.25);
        
        // Normal conditions
        let size = scaler.calculate_size(1_000_000.0);
        assert_eq!(size, 100.0);
        
        // Drawdown - should reduce size
        let size = scaler.calculate_size(900_000.0);
        assert!(size < 100.0);
    }
}
