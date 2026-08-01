"""
Drift Attribution Engine - Links prediction drift to feature distribution shifts.
Implements statistical tests for detecting and attributing model prediction drift
to specific feature-level changes in real-time trading environments.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque
import numpy as np
from scipy import stats
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FeatureStats:
    """Statistics for a single feature over time."""
    mean: float = 0.0
    std: float = 1.0
    min_val: float = 0.0
    max_val: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 3.0
    sample_count: int = 0
    
    @classmethod
    def from_array(cls, data: np.ndarray) -> 'FeatureStats':
        """Compute statistics from array."""
        return cls(
            mean=np.mean(data),
            std=np.std(data) if len(data) > 1 else 1.0,
            min_val=np.min(data),
            max_val=np.max(data),
            skewness=stats.skew(data) if len(data) > 2 else 0.0,
            kurtosis=stats.kurtosis(data) if len(data) > 3 else 3.0,
            sample_count=len(data)
        )


@dataclass
class DriftEvent:
    """Represents a detected drift event with attribution."""
    timestamp: float
    feature_name: str
    drift_type: str  # 'mean', 'variance', 'distribution', 'correlation'
    test_statistic: float
    p_value: float
    baseline_stats: FeatureStats
    current_stats: FeatureStats
    contribution_to_prediction_drift: float
    severity: str  # 'low', 'medium', 'high', 'critical'
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "feature": self.feature_name,
            "drift_type": self.drift_type,
            "test_statistic": self.test_statistic,
            "p_value": self.p_value,
            "baseline_mean": self.baseline_stats.mean,
            "current_mean": self.current_stats.mean,
            "baseline_std": self.baseline_stats.std,
            "current_std": self.current_stats.std,
            "contribution": self.contribution_to_prediction_drift,
            "severity": self.severity
        }


@dataclass
class PredictionDrift:
    """Overall prediction drift metrics."""
    timestamp: float
    baseline_mean_pred: float
    current_mean_pred: float
    mean_shift: float
    ks_statistic: float
    ks_p_value: float
    psi_score: float  # Population Stability Index
    attributed_features: List[DriftEvent] = field(default_factory=list)


class DriftAttributionEngine:
    """
    Engine for detecting prediction drift and attributing it to feature shifts.
    Uses multiple statistical tests for robust detection.
    """
    
    def __init__(self, 
                 feature_names: List[str],
                 window_size: int = 1000,
                 baseline_window: int = 5000,
                 detection_threshold: float = 0.01,
                 psi_bins: int = 10):
        """
        Initialize drift attribution engine.
        
        Args:
            feature_names: Names of features to monitor
            window_size: Size of rolling window for current statistics
            baseline_window: Size of baseline window for comparison
            detection_threshold: P-value threshold for drift detection
            psi_bins: Number of bins for PSI calculation
        """
        self.feature_names = feature_names
        self.window_size = window_size
        self.baseline_window = baseline_window
        self.detection_threshold = detection_threshold
        self.psi_bins = psi_bins
        
        # Rolling buffers for features and predictions
        self._feature_buffers: Dict[str, deque] = {
            name: deque(maxlen=window_size + baseline_window) 
            for name in feature_names
        }
        self._prediction_buffer: deque = deque(maxlen=window_size + baseline_window)
        
        # Baseline statistics (computed once or periodically)
        self._baseline_feature_stats: Dict[str, FeatureStats] = {}
        self._baseline_prediction_stats: Optional[FeatureStats] = None
        self._baseline_correlations: Optional[np.ndarray] = None
        
        # Detected events
        self._drift_events: deque = deque(maxlen=1000)
        self._prediction_drifts: deque = deque(maxlen=100)
        
        self._is_calibrated = False
        self._last_check_time: float = 0.0
    
    def calibrate_baseline(self, features: np.ndarray, predictions: np.ndarray):
        """
        Calibrate baseline statistics from initial data.
        
        Args:
            features: Initial feature matrix (n_samples, n_features)
            predictions: Initial predictions
        """
        if len(features) < self.baseline_window // 2:
            logger.warning("Insufficient data for baseline calibration")
            return
        
        # Use first baseline_window samples
        n_samples = min(len(features), self.baseline_window)
        baseline_features = features[:n_samples]
        baseline_preds = predictions[:n_samples]
        
        # Compute baseline feature statistics
        for i, name in enumerate(self.feature_names):
            self._baseline_feature_stats[name] = FeatureStats.from_array(
                baseline_features[:, i]
            )
        
        # Baseline prediction statistics
        self._baseline_prediction_stats = FeatureStats.from_array(baseline_preds)
        
        # Baseline correlation matrix
        self._baseline_correlations = np.corrcoef(baseline_features.T)
        
        # Fill buffers with baseline data
        for i, name in enumerate(self.feature_names):
            self._feature_buffers[name].extend(baseline_features[:, i])
        self._prediction_buffer.extend(baseline_preds)
        
        self._is_calibrated = True
        logger.info(f"Baseline calibrated with {n_samples} samples")
    
    def update(self, features: np.ndarray, predictions: np.ndarray, 
               timestamp: Optional[float] = None) -> Optional[PredictionDrift]:
        """
        Update engine with new data and check for drift.
        
        Args:
            features: New feature matrix (can be single sample or batch)
            predictions: Corresponding predictions
            timestamp: Optional timestamp
            
        Returns:
            PredictionDrift object if drift detected, None otherwise
        """
        if not self._is_calibrated:
            logger.warning("Engine not calibrated, skipping update")
            return None
        
        timestamp = timestamp or asyncio.get_event_loop().time()
        
        # Handle single sample vs batch
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
            predictions = predictions.reshape(-1) if len(predictions.shape) == 0 else predictions
        
        # Update buffers
        for i, name in enumerate(self.feature_names):
            self._feature_buffers[name].extend(features[:, i])
        self._prediction_buffer.extend(predictions)
        
        # Check drift periodically (every 100 new samples)
        total_samples = len(self._prediction_buffer)
        if total_samples < self.window_size + self.baseline_window:
            return None
        
        if total_samples % 100 != 0:
            return None
        
        return self._check_drift(timestamp)
    
    def _check_drift(self, timestamp: float) -> PredictionDrift:
        """Perform comprehensive drift check."""
        buffer = np.array(self._prediction_buffer)
        baseline_preds = buffer[:self.baseline_window]
        current_preds = buffer[-self.window_size:]
        
        # Overall prediction drift metrics
        baseline_mean = np.mean(baseline_preds)
        current_mean = np.mean(current_preds)
        mean_shift = current_mean - baseline_mean
        
        # KS test for distribution shift
        ks_stat, ks_pval = stats.ks_2samp(baseline_preds, current_preds)
        
        # PSI calculation
        psi_score = self._calculate_psi(baseline_preds, current_preds)
        
        drift = PredictionDrift(
            timestamp=timestamp,
            baseline_mean_pred=baseline_mean,
            current_mean_pred=current_mean,
            mean_shift=mean_shift,
            ks_statistic=ks_stat,
            ks_p_value=ks_pval,
            psi_score=psi_score
        )
        
        # Attribute drift to specific features
        attributed_features = []
        for name in self.feature_names:
            feature_drift = self._check_feature_drift(
                name, timestamp, baseline_preds, current_preds
            )
            if feature_drift:
                attributed_features.append(feature_drift)
                self._drift_events.append(feature_drift)
        
        drift.attributed_features = attributed_features
        self._prediction_drifts.append(drift)
        
        # Log significant drift
        if ks_pval < self.detection_threshold or psi_score > 0.2:
            severity = "critical" if psi_score > 0.25 else "high" if psi_score > 0.1 else "medium"
            logger.warning(
                f"Prediction drift detected: KS p={ks_pval:.4f}, PSI={psi_score:.4f}, "
                f"severity={severity}, attributed to {len(attributed_features)} features"
            )
        
        return drift
    
    def _check_feature_drift(self, feature_name: str, timestamp: float,
                             baseline_preds: np.ndarray, 
                             current_preds: np.ndarray) -> Optional[DriftEvent]:
        """Check individual feature for drift and compute contribution."""
        buffer = np.array(self._feature_buffers[feature_name])
        baseline_data = buffer[:self.baseline_window]
        current_data = buffer[-self.window_size:]
        
        baseline_stats = self._baseline_feature_stats.get(
            feature_name, FeatureStats.from_array(baseline_data)
        )
        current_stats = FeatureStats.from_array(current_data)
        
        # Multiple tests for different drift types
        drift_events = []
        
        # Mean shift (t-test)
        t_stat, t_pval = stats.ttest_ind(baseline_data, current_data, equal_var=False)
        if t_pval < self.detection_threshold:
            drift_events.append(('mean', t_stat, t_pval))
        
        # Variance shift (Levene's test)
        _, var_pval = stats.levene(baseline_data, current_data)
        if var_pval < self.detection_threshold:
            drift_events.append(('variance', np.std(current_data)/np.std(baseline_data), var_pval))
        
        # Distribution shift (KS test)
        ks_stat, ks_pval = stats.ks_2samp(baseline_data, current_data)
        if ks_pval < self.detection_threshold:
            drift_events.append(('distribution', ks_stat, ks_pval))
        
        if not drift_events:
            return None
        
        # Pick most significant drift
        best_drift = min(drift_events, key=lambda x: x[2])
        drift_type, test_stat, p_value = best_drift
        
        # Estimate contribution to prediction drift using simple sensitivity
        contribution = self._estimate_contribution(
            feature_name, baseline_data, current_data, baseline_preds, current_preds
        )
        
        # Determine severity
        if p_value < 0.001 or contribution > 0.3:
            severity = "critical"
        elif p_value < 0.01 or contribution > 0.15:
            severity = "high"
        elif p_value < 0.05 or contribution > 0.05:
            severity = "medium"
        else:
            severity = "low"
        
        return DriftEvent(
            timestamp=timestamp,
            feature_name=feature_name,
            drift_type=drift_type,
            test_statistic=test_stat,
            p_value=p_value,
            baseline_stats=baseline_stats,
            current_stats=current_stats,
            contribution_to_prediction_drift=contribution,
            severity=severity
        )
    
    def _calculate_psi(self, baseline: np.ndarray, current: np.ndarray) -> float:
        """
        Calculate Population Stability Index.
        
        PSI = Σ (actual% - expected%) * ln(actual% / expected%)
        """
        # Create bins based on baseline distribution
        percentiles = np.linspace(0, 100, self.psi_bins + 1)
        bin_edges = np.percentile(baseline, percentiles)
        
        # Remove duplicate edges
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 3:
            return 0.0
        
        # Histogram counts
        baseline_counts, _ = np.histogram(baseline, bins=bin_edges)
        current_counts, _ = np.histogram(current, bins=bin_edges)
        
        # Convert to percentages (add small epsilon to avoid log(0))
        epsilon = 1e-6
        baseline_pct = (baseline_counts + epsilon) / (len(baseline) + epsilon * len(bin_edges))
        current_pct = (current_counts + epsilon) / (len(current) + epsilon * len(bin_edges))
        
        # PSI calculation
        psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))
        
        return psi
    
    def _estimate_contribution(self, feature_name: str, 
                               baseline_feat: np.ndarray, current_feat: np.ndarray,
                               baseline_pred: np.ndarray, current_pred: np.ndarray) -> float:
        """
        Estimate feature's contribution to overall prediction drift.
        Uses simple regression-based attribution.
        """
        # Simple approach: correlation between feature shift and prediction shift
        feat_shift = current_feat - baseline_feat[:len(current_feat)] if len(baseline_feat) >= len(current_feat) else np.zeros(len(current_feat))
        
        if len(feat_shift) != len(current_pred):
            min_len = min(len(feat_shift), len(current_pred))
            feat_shift = feat_shift[:min_len]
            pred_shift = current_pred[:min_len] - baseline_pred[:min_len]
        else:
            pred_shift = current_pred[:len(feat_shift)] - baseline_pred[:len(feat_shift)]
        
        if len(feat_shift) < 10 or np.std(feat_shift) < 1e-10:
            return 0.0
        
        # Correlation-based contribution
        corr, _ = stats.pearsonr(feat_shift, pred_shift)
        
        return abs(corr) if not np.isnan(corr) else 0.0
    
    def get_recent_drift_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent drift events."""
        return [event.to_dict() for event in list(self._drift_events)[-limit:]]
    
    def get_drift_summary(self) -> Dict[str, Any]:
        """Get comprehensive drift summary."""
        if not self._prediction_drifts:
            return {"status": "no_data", "calibrated": self._is_calibrated}
        
        latest = self._prediction_drifts[-1]
        
        # Count events by severity
        severity_counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        for event in self._drift_events:
            severity_counts[event.severity] = severity_counts.get(event.severity, 0) + 1
        
        # Top drifting features
        feature_drift_counts = {}
        for event in self._drift_events:
            feature_drift_counts[event.feature_name] = feature_drift_counts.get(event.feature_name, 0) + 1
        
        top_drifting = sorted(feature_drift_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "status": "monitoring",
            "calibrated": self._is_calibrated,
            "latest_drift": {
                "timestamp": latest.timestamp,
                "mean_shift": latest.mean_shift,
                "ks_statistic": latest.ks_statistic,
                "ks_p_value": latest.ks_p_value,
                "psi_score": latest.psi_score
            },
            "severity_counts": severity_counts,
            "top_drifting_features": top_drifting,
            "total_events": len(self._drift_events),
            "samples_processed": len(self._prediction_buffer)
        }
    
    def reset_baseline(self, features: np.ndarray, predictions: np.ndarray):
        """Recalibrate baseline with new data."""
        self._drift_events.clear()
        self._prediction_drifts.clear()
        self.calibrate_baseline(features, predictions)
        logger.info("Baseline recalibrated")


# Module singleton instance
_engine: Optional[DriftAttributionEngine] = None


def get_drift_engine(feature_names: List[str], **kwargs) -> DriftAttributionEngine:
    """Get or create the global drift attribution engine."""
    global _engine
    
    if _engine is None or set(_engine.feature_names) != set(feature_names):
        _engine = DriftAttributionEngine(feature_names, **kwargs)
        logger.info(f"Created drift attribution engine for {len(feature_names)} features")
    
    return _engine


if __name__ == "__main__":
    # Test the engine
    np.random.seed(42)
    
    feature_names = ["feature_1", "feature_2", "feature_3", "feature_4", "feature_5"]
    n_features = len(feature_names)
    
    engine = DriftAttributionEngine(feature_names, window_size=500, baseline_window=2000)
    
    # Generate baseline data
    baseline_features = np.random.randn(2000, n_features)
    baseline_preds = np.sum(baseline_features, axis=1) + np.random.randn(2000) * 0.5
    
    engine.calibrate_baseline(baseline_features, baseline_preds)
    
    # Simulate streaming data with drift introduced at sample 1000
    print("Simulating streaming data...")
    drift_detected = False
    
    for i in range(1500):
        if i < 500:
            # Normal regime
            features = np.random.randn(10, n_features)
        else:
            # Introduce drift in feature_1
            features = np.random.randn(10, n_features)
            features[:, 0] += 2.0  # Mean shift
        
        predictions = np.sum(features, axis=1) + np.random.randn(10) * 0.5
        
        result = engine.update(features, predictions)
        
        if result and not drift_detected:
            print(f"\nDrift detected at iteration {i}!")
            print(f"  PSI Score: {result.psi_score:.4f}")
            print(f"  KS P-value: {result.ks_p_value:.6f}")
            print(f"  Mean Shift: {result.mean_shift:.4f}")
            print(f"  Attributed Features: {len(result.attributed_features)}")
            
            for feat in result.attributed_features[:3]:
                print(f"    - {feat.feature_name}: {feat.drift_type} (p={feat.p_value:.6f}, contrib={feat.contribution:.3f})")
            
            drift_detected = True
    
    print(f"\nFinal Summary: {engine.get_drift_summary()}")
