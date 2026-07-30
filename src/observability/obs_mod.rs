//! Observability Module Root
//! 
//! Pushes behavioral metrics to the Terminal UI and triggers safety alerts.
//! Central coordination for behavior anomaly detection and self-healing.

pub mod behavior;
pub mod healing;

pub use behavior::{
    BehaviorAnomalyDetector, BehaviorAlert, BehaviorEvent, AnomalyType,
    AnomalySeverity, BehaviorMetrics, BehaviorStats,
};
pub use healing::{
    SelfHealingManager, HealthCheckResult, HealthStatus, HealingAction,
    HealingEvent, ComponentType, HealingStats,
};

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use crate::gateway::venue::VenueId;

/// Unified observability alert combining behavior and health alerts
#[derive(Debug, Clone)]
pub enum ObservabilityAlert {
    Behavior(BehaviorAlert),
    Healing(HealingEvent),
    System(SystemAlert),
}

/// System-level alert
#[derive(Debug, Clone)]
pub struct SystemAlert {
    pub alert_id: u64,
    pub severity: AnomalySeverity,
    pub message: String,
    pub component: String,
    pub timestamp_ns: u64,
    pub recommended_action: &'static str,
}

/// Main Observability Manager
/// Coordinates behavior detection and self-healing across all venues
pub struct ObservabilityManager {
    /// Behavior anomaly detector
    behavior_detector: Arc<BehaviorAnomalyDetector>,
    /// Self-healing manager
    healing_manager: Arc<SelfHealingManager>,
    /// Manager enabled flag
    enabled: AtomicBool,
    /// Total alerts generated
    total_alerts: AtomicU64,
    /// Alert callback for UI integration
    alert_callback: Option<Arc<dyn Fn(ObservabilityAlert) + Send + Sync>>,
}

impl ObservabilityManager {
    pub fn new(venues: &[VenueId]) -> Self {
        let behavior_detector = Arc::new(BehaviorAnomalyDetector::new(venues));
        let healing_manager = Arc::new(SelfHealingManager::new());

        Self {
            behavior_detector,
            healing_manager,
            enabled: AtomicBool::new(true),
            total_alerts: AtomicU64::new(0),
            alert_callback: None,
        }
    }

    /// Get behavior detector reference
    #[inline]
    pub fn behavior_detector(&self) -> &Arc<BehaviorAnomalyDetector> {
        &self.behavior_detector
    }

    /// Get healing manager reference
    #[inline]
    pub fn healing_manager(&self) -> &Arc<SelfHealingManager> {
        &self.healing_manager
    }

    /// Set unified alert callback
    pub fn set_alert_callback<F>(&mut self, callback: F)
    where
        F: Fn(ObservabilityAlert) + Send + Sync + 'static,
    {
        self.alert_callback = Some(Arc::new(callback));

        // Also set up individual callbacks
        let alert_tx = self.alert_callback.clone();
        self.behavior_detector.set_alert_callback(move |alert| {
            if let Some(ref cb) = alert_tx {
                cb(ObservabilityAlert::Behavior(alert));
            }
        });

        let alert_tx = self.alert_callback.clone();
        self.healing_manager.set_healing_callback(move |event| {
            if let Some(ref cb) = alert_tx {
                cb(ObservabilityAlert::Healing(event));
            }
        });
    }

    /// Record behavior event
    pub fn record_behavior_event(&self, venue_id: VenueId, symbol: [u8; 12], event: BehaviorEvent) {
        self.behavior_detector.record_event(venue_id, symbol, event);
    }

    /// Record heartbeat from component
    pub fn record_heartbeat(&self, component_id: &str) {
        self.healing_manager.record_heartbeat(component_id);
    }

    /// Register component for health monitoring
    pub fn register_component(
        &self,
        component_id: String,
        component_type: ComponentType,
        venue_id: Option<VenueId>,
        symbol: Option<[u8; 12]>,
    ) {
        self.healing_manager.register_component(
            component_id,
            component_type,
            venue_id,
            symbol,
        );
    }

    /// Run full observability check
    /// Returns all detected issues requiring attention
    pub fn run_full_check(&self) -> Vec<ObservabilityAlert> {
        if !self.enabled.load(Ordering::Acquire) {
            return Vec::new();
        }

        let mut alerts = Vec::new();

        // Check behavior anomalies
        let behavior_alerts = self.behavior_detector.check_all_anomalies();
        for alert in behavior_alerts {
            self.total_alerts.fetch_add(1, Ordering::Relaxed);
            
            if let Some(ref callback) = self.alert_callback {
                callback(ObservabilityAlert::Behavior(alert.clone()));
            }
            
            alerts.push(ObservabilityAlert::Behavior(alert));
        }

        // Attempt healing for unhealthy components
        let healing_events = self.healing_manager.attempt_healing();
        for event in healing_events {
            self.total_alerts.fetch_add(1, Ordering::Relaxed);
            
            if let Some(ref callback) = self.alert_callback {
                callback(ObservabilityAlert::Healing(event.clone()));
            }
            
            alerts.push(ObservabilityAlert::Healing(event));
        }

        alerts
    }

    /// Get combined metrics for UI display
    pub fn get_dashboard_metrics(&self) -> DashboardMetrics {
        let behavior_stats = self.behavior_detector.get_stats();
        let healing_stats = self.healing_manager.get_stats();

        DashboardMetrics {
            behavior_stats,
            healing_stats,
            total_alerts: self.total_alerts.load(Ordering::Relaxed),
            enabled: self.enabled.load(Ordering::Acquire),
        }
    }

    /// Enable/disable all observability features
    #[inline]
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Release);
        self.behavior_detector.set_enabled(enabled);
        self.healing_manager.set_enabled(enabled);
    }

    /// Get aggregate statistics
    pub fn get_stats(&self) -> ObservabilityStats {
        ObservabilityStats {
            behavior_stats: self.behavior_detector.get_stats(),
            healing_stats: self.healing_manager.get_stats(),
            total_alerts: self.total_alerts.load(Ordering::Relaxed),
            enabled: self.enabled.load(Ordering::Acquire),
        }
    }

    /// Emit system alert
    pub fn emit_system_alert(&self, alert: SystemAlert) {
        self.total_alerts.fetch_add(1, Ordering::Relaxed);
        
        if let Some(ref callback) = self.alert_callback {
            callback(ObservabilityAlert::System(alert.clone()));
        }
    }
}

/// Combined dashboard metrics for UI
#[derive(Debug, Clone)]
pub struct DashboardMetrics {
    pub behavior_stats: BehaviorStats,
    pub healing_stats: HealingStats,
    pub total_alerts: u64,
    pub enabled: bool,
}

/// Aggregate observability statistics
#[derive(Debug, Clone)]
pub struct ObservabilityStats {
    pub behavior_stats: BehaviorStats,
    pub healing_stats: HealingStats,
    pub total_alerts: u64,
    pub enabled: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manager_creation() {
        let venues = vec![VenueId::Nasdaq];
        let manager = ObservabilityManager::new(&venues);

        assert!(manager.enabled.load(Ordering::Acquire));
        assert_eq!(manager.get_stats().total_alerts, 0);
    }

    #[test]
    fn test_component_registration() {
        let venues = vec![VenueId::Nasdaq];
        let manager = ObservabilityManager::new(&venues);

        manager.register_component(
            "test_actor".to_string(),
            ComponentType::SymbolActor,
            Some(VenueId::Nasdaq),
            Some(*b"AAPL        "),
        );

        let stats = manager.get_stats();
        assert_eq!(stats.healing_stats.components_tracked, 1);
    }

    #[test]
    fn test_heartbeat_recording() {
        let venues = vec![VenueId::Nasdaq];
        let manager = ObservabilityManager::new(&venues);

        manager.register_component(
            "test".to_string(),
            ComponentType::Custom,
            None,
            None,
        );

        manager.record_heartbeat("test");

        let status = manager.healing_manager().get_component_status("test");
        assert_eq!(status, Some(HealthStatus::Healthy));
    }

    #[test]
    fn test_full_check_empty() {
        let venues = vec![VenueId::Nasdaq];
        let manager = ObservabilityManager::new(&venues);

        let alerts = manager.run_full_check();
        assert!(alerts.is_empty());  // No issues with fresh state
    }
}
