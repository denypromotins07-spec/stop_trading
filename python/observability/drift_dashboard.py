"""
Population Stability Index (PSI) and Jensen-Shannon divergence for data drift detection.
Detects subtle feature distribution changes that trigger automated retraining pipelines.
"""

import numpy as np
from numba import njit, prange
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass, field
import threading
import time


@njit(cache=True)
def compute_histogram(
    data: np.ndarray,
    n_bins: int = 20,
    min_val: float = 0.0,
    max_val: float = 0.0
) -> np.ndarray:
    """Compute normalized histogram."""
    if max_val <= min_val:
        min_val = np.min(data)
        max_val = np.max(data)
    
    bin_width = (max_val - min_val) / n_bins
    hist = np.zeros(n_bins, dtype=np.float64)
    
    for val in data:
        bin_idx = int((val - min_val) / bin_width)
        bin_idx = max(0, min(n_bins - 1, bin_idx))
        hist[bin_idx] += 1
    
    # Normalize to probabilities
    total = np.sum(hist)
    if total > 0:
        hist = hist / total
    
    return hist


@njit(cache=True)
def compute_psi(expected: np.ndarray, actual: np.ndarray) -> float:
    """
    Compute Population Stability Index (PSI).
    PSI < 0.1: No significant change
    PSI 0.1-0.25: Moderate change
    PSI > 0.25: Significant change
    """
    n_bins = len(expected)
    psi = 0.0
    
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    
    for i in range(n_bins):
        exp_pct = expected[i] + eps
        act_pct = actual[i] + eps
        
        psi += (act_pct - exp_pct) * np.log(act_pct / exp_pct)
    
    return psi


@njit(cache=True)
def compute_js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Compute Jensen-Shannon divergence between two distributions.
    Returns value in [0, 1] where 0 means identical distributions.
    """
    n = len(p)
    eps = 1e-10
    
    # Compute midpoint distribution
    m = np.zeros(n, dtype=np.float64)
    for i in range(n):
        m[i] = (p[i] + q[i]) / 2
    
    # Compute KL divergences
    kl_pm = 0.0
    kl_qm = 0.0
    
    for i in range(n):
        if p[i] > eps:
            kl_pm += p[i] * np.log(p[i] / (m[i] + eps))
        if q[i] > eps:
            kl_qm += q[i] * np.log(q[i] / (m[i] + eps))
    
    # JS divergence is average of KL divergences
    js = (kl_pm + kl_qm) / 2
    
    return js


@njit(cache=True)
def compute_feature_drift(
    baseline_data: np.ndarray,
    current_data: np.ndarray,
    n_bins: int = 20
) -> Tuple[float, float]:
    """Compute both PSI and JS divergence for a feature."""
    # Get range from combined data
    all_data = np.concatenate([baseline_data, current_data])
    min_val = np.min(all_data)
    max_val = np.max(all_data)
    
    # Compute histograms
    baseline_hist = compute_histogram(baseline_data, n_bins, min_val, max_val)
    current_hist = compute_histogram(current_data, n_bins, min_val, max_val)
    
    # Compute metrics
    psi = compute_psi(baseline_hist, current_hist)
    js = compute_js_divergence(baseline_hist, current_hist)
    
    return psi, js


@njit(parallel=True, cache=True)
def compute_multivariate_drift(
    baseline: np.ndarray,
    current: np.ndarray,
    n_bins: int = 20
) -> np.ndarray:
    """Compute drift metrics for each feature in multivariate data."""
    n_features = baseline.shape[1]
    drift_scores = np.zeros(n_features, dtype=np.float64)
    
    for feat in prange(n_features):
        baseline_feat = baseline[:, feat]
        current_feat = current[:, feat]
        
        psi, _ = compute_feature_drift(baseline_feat, current_feat, n_bins)
        drift_scores[feat] = psi
    
    return drift_scores


@dataclass
class DriftResult:
    """Drift detection result for a single feature."""
    
    feature_name: str
    psi: float = 0.0
    js_divergence: float = 0.0
    
    # Status
    status: str = "STABLE"  # STABLE, WARNING, CRITICAL
    severity: float = 0.0  # 0-1 scale
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "psi": self.psi,
            "js_divergence": self.js_divergence,
            "status": self.status,
            "severity": self.severity
        }


@dataclass
class DriftReport:
    """Complete drift analysis report."""
    
    # Per-feature results
    feature_results: List[DriftResult] = field(default_factory=list)
    
    # Aggregate metrics
    avg_psi: float = 0.0
    max_psi: float = 0.0
    avg_js: float = 0.0
    n_drifting_features: int = 0
    
    # Overall status
    overall_status: str = "STABLE"
    retrain_recommended: bool = False
    
    # Metadata
    timestamp: float = field(default_factory=time.time)
    baseline_size: int = 0
    current_size: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_results": [r.to_dict() for r in self.feature_results],
            "avg_psi": self.avg_psi,
            "max_psi": self.max_psi,
            "avg_js": self.avg_js,
            "n_drifting_features": self.n_drifting_features,
            "overall_status": self.overall_status,
            "retrain_recommended": self.retrain_recommended,
            "timestamp": self.timestamp,
            "baseline_size": self.baseline_size,
            "current_size": self.current_size
        }


class DriftDetector:
    """
    Detects data drift using PSI and Jensen-Shannon divergence.
    Triggers retraining when drift exceeds thresholds.
    """
    
    def __init__(
        self,
        psi_warning_threshold: float = 0.1,
        psi_critical_threshold: float = 0.25,
        js_warning_threshold: float = 0.1,
        n_bins: int = 20,
        min_samples: int = 100
    ):
        self.psi_warning = psi_warning_threshold
        self.psi_critical = psi_critical_threshold
        self.js_warning = js_warning_threshold
        self.n_bins = n_bins
        self.min_samples = min_samples
        
        # Baseline data storage
        self._baseline_data: Optional[np.ndarray] = None
        self._baseline_timestamp: float = 0.0
        self._feature_names: List[str] = []
        
        # Drift history
        self._drift_history: List[DriftReport] = []
        self._history_max = 100
        
        # Thread safety
        self._lock = threading.RLock()
    
    def set_baseline(
        self,
        data: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> None:
        """Set baseline data for drift comparison."""
        with self._lock:
            if len(data) < self.min_samples:
                raise ValueError(f"Baseline requires at least {self.min_samples} samples")
            
            self._baseline_data = data.copy()
            self._baseline_timestamp = time.time()
            
            if feature_names:
                self._feature_names = feature_names
            else:
                self._feature_names = [f"feature_{i}" for i in range(data.shape[1])]
    
    def detect_drift(
        self,
        current_data: np.ndarray,
        window_size: Optional[int] = None
    ) -> DriftReport:
        """Detect drift between baseline and current data."""
        with self._lock:
            if self._baseline_data is None:
                raise ValueError("No baseline data set")
            
            if len(current_data) < self.min_samples:
                return self._create_empty_report(len(current_data))
            
            # Use window if specified
            if window_size and len(current_data) > window_size:
                current_data = current_data[-window_size:]
            
            n_features = current_data.shape[1]
            
            # Ensure feature names match
            while len(self._feature_names) < n_features:
                self._feature_names.append(f"feature_{len(self._feature_names)}")
            
            # Compute drift for each feature
            feature_results = []
            psi_values = []
            js_values = []
            drifting_count = 0
            
            for feat_idx in range(n_features):
                baseline_feat = self._baseline_data[:, feat_idx]
                current_feat = current_data[:, feat_idx]
                
                psi, js = compute_feature_drift(
                    baseline_feat, current_feat, self.n_bins
                )
                
                # Determine status
                if psi >= self.psi_critical:
                    status = "CRITICAL"
                    severity = min(psi / self.psi_critical, 1.0)
                    drifting_count += 1
                elif psi >= self.psi_warning:
                    status = "WARNING"
                    severity = psi / self.psi_warning
                    drifting_count += 1
                else:
                    status = "STABLE"
                    severity = 0.0
                
                result = DriftResult(
                    feature_name=self._feature_names[feat_idx],
                    psi=psi,
                    js_divergence=js,
                    status=status,
                    severity=severity
                )
                
                feature_results.append(result)
                psi_values.append(psi)
                js_values.append(js)
            
            # Aggregate metrics
            avg_psi = np.mean(psi_values)
            max_psi = np.max(psi_values)
            avg_js = np.mean(js_values)
            
            # Overall status
            if max_psi >= self.psi_critical or drifting_count > n_features * 0.3:
                overall_status = "CRITICAL"
                retrain_recommended = True
            elif max_psi >= self.psi_warning or drifting_count > 0:
                overall_status = "WARNING"
                retrain_recommended = drifting_count > n_features * 0.1
            else:
                overall_status = "STABLE"
                retrain_recommended = False
            
            report = DriftReport(
                feature_results=feature_results,
                avg_psi=avg_psi,
                max_psi=max_psi,
                avg_js=avg_js,
                n_drifting_features=drifting_count,
                overall_status=overall_status,
                retrain_recommended=retrain_recommended,
                baseline_size=len(self._baseline_data),
                current_size=len(current_data)
            )
            
            # Update history
            self._drift_history.append(report)
            if len(self._drift_history) > self._history_max:
                self._drift_history.pop(0)
            
            return report
    
    def _create_empty_report(self, current_size: int) -> DriftReport:
        """Create empty report when insufficient data."""
        return DriftReport(
            overall_status="INSUFFICIENT_DATA",
            current_size=current_size,
            baseline_size=len(self._baseline_data) if self._baseline_data is not None else 0
        )
    
    def get_drift_trend(self) -> str:
        """Get trend of drift over recent reports."""
        with self._lock:
            if len(self._drift_history) < 3:
                return "UNKNOWN"
            
            recent = self._drift_history[-3:]
            psi_trend = [r.avg_psi for r in recent]
            
            if psi_trend[-1] > psi_trend[0] * 1.2:
                return "INCREASING"
            elif psi_trend[-1] < psi_trend[0] * 0.8:
                return "DECREASING"
            else:
                return "STABLE"
    
    def should_retrain(self) -> bool:
        """Check if retraining is recommended."""
        with self._lock:
            if not self._drift_history:
                return False
            
            latest = self._drift_history[-1]
            return latest.retrain_recommended
    
    def get_critical_features(self) -> List[str]:
        """Get list of features with critical drift."""
        with self._lock:
            if not self._drift_history:
                return []
            
            latest = self._drift_history[-1]
            return [
                r.feature_name for r in latest.feature_results
                if r.status == "CRITICAL"
            ]
    
    def reset_baseline(self) -> None:
        """Reset baseline data."""
        with self._lock:
            self._baseline_data = None
            self._baseline_timestamp = 0.0
            self._feature_names = []
            self._drift_history.clear()
    
    def update_baseline_rolling(
        self,
        new_data: np.ndarray,
        decay_factor: float = 0.1
    ) -> None:
        """Update baseline with exponential decay."""
        with self._lock:
            if self._baseline_data is None:
                return
            
            # Sample new data to match baseline size
            if len(new_data) > len(self._baseline_data):
                indices = np.random.choice(
                    len(new_data), len(self._baseline_data), replace=False
                )
                new_data = new_data[indices]
            
            # Exponential moving average update
            self._baseline_data = (
                (1 - decay_factor) * self._baseline_data +
                decay_factor * new_data
            )
            
            self._baseline_timestamp = time.time()


# Global singleton instance
_drift_instance: Optional[DriftDetector] = None
_instance_lock = threading.Lock()


def get_drift_detector() -> DriftDetector:
    """Get or create the global drift detector."""
    global _drift_instance
    if _drift_instance is None:
        with _instance_lock:
            if _drift_instance is None:
                _drift_instance = DriftDetector()
    return _drift_instance


if __name__ == "__main__":
    # Test drift detector
    print("Testing DriftDetector:")
    
    detector = DriftDetector()
    
    np.random.seed(42)
    
    # Generate baseline data (stable distribution)
    baseline = np.random.normal(0, 1, (1000, 5))
    detector.set_baseline(baseline, ["feat_A", "feat_B", "feat_C", "feat_D", "feat_E"])
    
    # Test with similar data (should be stable)
    print("\n--- Test 1: Similar Distribution ---")
    current_stable = np.random.normal(0, 1.1, (500, 5))
    report = detector.detect_drift(current_stable)
    
    print(f"Overall Status: {report.overall_status}")
    print(f"Avg PSI: {report.avg_psi:.4f}")
    print(f"Max PSI: {report.max_psi:.4f}")
    print(f"Retrain Recommended: {report.retrain_recommended}")
    
    # Test with drifted data
    print("\n--- Test 2: Drifted Distribution ---")
    current_drifted = np.random.normal(2, 1.5, (500, 5))  # Shifted mean
    report = detector.detect_drift(current_drifted)
    
    print(f"Overall Status: {report.overall_status}")
    print(f"Avg PSI: {report.avg_psi:.4f}")
    print(f"Max PSI: {report.max_psi:.4f}")
    print(f"N Drifting Features: {report.n_drifting_features}")
    print(f"Retrain Recommended: {report.retrain_recommended}")
    
    # Print per-feature results
    print("\nPer-Feature Results:")
    for result in report.feature_results:
        print(f"  {result.feature_name}: PSI={result.psi:.4f}, Status={result.status}")
    
    # Check drift trend
    print(f"\nDrift Trend: {detector.get_drift_trend()}")
    print(f"Should Retrain: {detector.should_retrain()}")
    print(f"Critical Features: {detector.get_critical_features()}")
