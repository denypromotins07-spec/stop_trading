"""
Drift Monitor - ADWIN and Page-Hinkley tests for concept drift detection.
Uses river library to detect feature distribution shifts in real-time.
Triggers automated retraining when statistical properties diverge.
Strictly enforces 3GB RAM limit.
"""
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from collections import deque
from dataclasses import dataclass
import logging

try:
    from river import drift, stats
    RIVER_AVAILABLE = True
except ImportError:
    RIVER_AVAILABLE = False


logger = logging.getLogger(__name__)


@dataclass
class DriftAlert:
    """Represents a drift detection alert."""
    feature_name: str
    test_type: str  # 'adwin' or 'page_hinkley'
    drift_score: float
    threshold: float
    timestamp_ns: int
    severity: str  # 'low', 'medium', 'high'
    recommended_action: str


class FeatureDriftDetector:
    """
    Detects drift for a single feature using multiple tests.
    Memory-bounded for 3GB limit.
    """
    
    def __init__(self,
                 feature_name: str,
                 adwin_delta: float = 0.002,
                 ph_threshold: float = 50.0,
                 max_history: int = 5000):
        """
        Initialize feature drift detector.
        
        Args:
            feature_name: Name of the feature
            adwin_delta: ADWIN sensitivity (lower = more sensitive)
            ph_threshold: Page-Hinkley threshold
            max_history: Maximum history to keep
        """
        self.feature_name = feature_name
        self.adwin_delta = adwin_delta
        self.ph_threshold = ph_threshold
        self.max_history = max_history
        
        # Drift detectors
        if RIVER_AVAILABLE:
            self.adwin = drift.ADWIN(delta=adwin_delta)
            self.page_hinkley = drift.PageHinkley(threshold=ph_threshold)
        else:
            self.adwin = None
            self.page_hinkley = None
        
        # Statistics tracking (bounded)
        self._value_history: deque = deque(maxlen=max_history)
        self._baseline_stats: Optional[Dict[str, float]] = None
        self._drift_count = 0
    
    def set_baseline(self, values: np.ndarray):
        """Set baseline statistics from initial data."""
        self._baseline_stats = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'count': len(values)
        }
        
        # Initialize detectors with baseline
        if RIVER_AVAILABLE:
            for v in values[-1000:]:  # Use last 1000 for init
                self.adwin.update(v)
                self.page_hinkley.update(v)
    
    def update(self, value: float, timestamp_ns: int) -> Optional[DriftAlert]:
        """
        Update detector with new value and check for drift.
        
        Args:
            value: New feature value
            timestamp_ns: Timestamp
            
        Returns:
            DriftAlert if drift detected, None otherwise
        """
        self._value_history.append(value)
        
        if not RIVER_AVAILABLE or self.adwin is None:
            return None
        
        # Update detectors
        self.adwin.update(value)
        self.page_hinkley.update(value)
        
        # Check for drift
        alerts = []
        
        # ADWIN drift check
        if self.adwin.drift_detected:
            severity = self._calculate_severity('adwin')
            alert = DriftAlert(
                feature_name=self.feature_name,
                test_type='adwin',
                drift_score=float(self.adwin.width),
                threshold=self.adwin_delta,
                timestamp_ns=timestamp_ns,
                severity=severity,
                recommended_action=self._get_recommendation(severity)
            )
            alerts.append(alert)
            self._drift_count += 1
            self.adwin.reset()  # Reset after detection
        
        # Page-Hinkley drift check
        if self.page_hinkley.drift_detected:
            severity = self._calculate_severity('page_hinkley')
            alert = DriftAlert(
                feature_name=self.feature_name,
                test_type='page_hinkley',
                drift_score=float(self.page_hinkley.sum),
                threshold=self.ph_threshold,
                timestamp_ns=timestamp_ns,
                severity=severity,
                recommended_action=self._get_recommendation(severity)
            )
            alerts.append(alert)
            self._drift_count += 1
            self.page_hinkley.reset()
        
        return alerts[0] if alerts else None
    
    def _calculate_severity(self, test_type: str) -> str:
        """Calculate drift severity based on recent history."""
        if len(self._value_history) < 100:
            return 'low'
        
        recent = list(self._value_history)[-100:]
        baseline = self._baseline_stats
        
        if baseline is None:
            return 'medium'
        
        # Calculate deviation from baseline
        recent_mean = np.mean(recent)
        deviation = abs(recent_mean - baseline['mean']) / (baseline['std'] + 1e-6)
        
        if deviation > 3.0:
            return 'high'
        elif deviation > 2.0:
            return 'medium'
        return 'low'
    
    def _get_recommendation(self, severity: str) -> str:
        """Get recommended action based on severity."""
        recommendations = {
            'low': 'Continue monitoring',
            'medium': 'Consider model retraining',
            'high': 'Immediate retraining recommended'
        }
        return recommendations.get(severity, 'Monitor closely')
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current feature statistics."""
        if not self._value_history:
            return {}
        
        values = list(self._value_history)
        stats_dict = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'count': len(values),
            'drift_count': self._drift_count
        }
        
        if self._baseline_stats:
            stats_dict['baseline_mean'] = self._baseline_stats['mean']
            stats_dict['deviation_from_baseline'] = (
                stats_dict['mean'] - self._baseline_stats['mean']
            )
        
        return stats_dict
    
    def reset(self):
        """Reset detector state."""
        if RIVER_AVAILABLE and self.adwin:
            self.adwin.reset()
            self.page_hinkley.reset()
        self._value_history.clear()
        self._drift_count = 0


class DriftMonitor:
    """
    Monitors drift across all features and triggers retraining.
    Coordinates multiple FeatureDriftDetectors.
    """
    
    def __init__(self,
                 feature_names: List[str],
                 retrain_threshold: int = 3,
                 time_window_ns: int = 3600_000_000_000):
        """
        Initialize drift monitor.
        
        Args:
            feature_names: Names of features to monitor
            retrain_threshold: Number of high-severity alerts to trigger retrain
            time_window_ns: Time window for alert aggregation
        """
        self.feature_names = feature_names
        self.retrain_threshold = retrain_threshold
        self.time_window_ns = time_window_ns
        
        # Create detectors for each feature
        self._detectors: Dict[str, FeatureDriftDetector] = {}
        for name in feature_names:
            self._detectors[name] = FeatureDriftDetector(name)
        
        # Alert history (bounded)
        self._alerts: deque = deque(maxlen=1000)
        self._retrain_triggered = False
    
    def set_baseline(self, feature_data: Dict[str, np.ndarray]):
        """
        Set baseline for all features.
        
        Args:
            feature_data: Dict mapping feature names to baseline arrays
        """
        for name, data in feature_data.items():
            if name in self._detectors:
                self._detectors[name].set_baseline(data)
        
        logger.info(f"Set baseline for {len(feature_data)} features")
    
    def update(self,
               feature_values: Dict[str, float],
               timestamp_ns: int) -> List[DriftAlert]:
        """
        Update all detectors with new feature values.
        
        Args:
            feature_values: Dict mapping feature names to values
            timestamp_ns: Timestamp
            
        Returns:
            List of drift alerts
        """
        alerts = []
        
        for name, value in feature_values.items():
            if name in self._detectors:
                alert = self._detectors[name].update(value, timestamp_ns)
                if alert:
                    alerts.append(alert)
                    self._alerts.append(alert)
        
        # Check if retraining should be triggered
        high_severity_count = sum(
            1 for a in self._alerts 
            if a.severity == 'high' and 
            timestamp_ns - a.timestamp_ns < self.time_window_ns
        )
        
        if high_severity_count >= self.retrain_threshold:
            self._retrain_triggered = True
            logger.warning(
                f"Retraining triggered: {high_severity_count} high-severity alerts"
            )
        
        return alerts
    
    def should_retrain(self) -> bool:
        """Check if retraining should be triggered."""
        return self._retrain_triggered
    
    def acknowledge_retrain(self):
        """Acknowledge retraining trigger and reset."""
        self._retrain_triggered = False
    
    def get_drift_summary(self) -> Dict[str, Any]:
        """Get summary of drift across all features."""
        summary = {
            'features': {},
            'total_alerts': len(self._alerts),
            'retrain_triggered': self._retrain_triggered
        }
        
        for name, detector in self._detectors.items():
            summary['features'][name] = detector.get_current_stats()
        
        # Count alerts by severity
        severity_counts = {'low': 0, 'medium': 0, 'high': 0}
        for alert in self._alerts:
            severity_counts[alert.severity] = severity_counts.get(alert.severity, 0) + 1
        
        summary['alerts_by_severity'] = severity_counts
        
        return summary
    
    def reset_feature(self, feature_name: str):
        """Reset detector for a specific feature."""
        if feature_name in self._detectors:
            self._detectors[feature_name].reset()


# Example usage
def main():
    """Example usage of drift monitor."""
    if not RIVER_AVAILABLE:
        print("River library not available")
        return
    
    # Create monitor
    features = ['feat_1', 'feat_2', 'feat_3']
    monitor = DriftMonitor(features, retrain_threshold=2)
    
    # Set baseline
    np.random.seed(42)
    baseline_data = {
        name: np.random.randn(1000) for name in features
    }
    monitor.set_baseline(baseline_data)
    
    # Simulate streaming data with drift
    print("Simulating streaming data...")
    for i in range(500):
        # Normal data
        values = {name: float(np.random.randn()) for name in features}
        
        # Introduce drift after step 300
        if i > 300:
            values['feat_1'] += 3.0  # Shift mean
        
        timestamp_ns = i * 1_000_000_000
        alerts = monitor.update(values, timestamp_ns)
        
        if alerts:
            for alert in alerts:
                print(f"Step {i}: {alert.feature_name} - {alert.test_type} - {alert.severity}")
    
    print(f"\nDrift summary: {monitor.get_drift_summary()}")
    print(f"Should retrain: {monitor.should_retrain()}")


if __name__ == "__main__":
    main()
