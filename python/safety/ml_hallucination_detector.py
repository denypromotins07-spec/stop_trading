"""
Python-Side Circuit Breakers & Safety Interlocks
Stage 49: ML Hallucination Detector using rolling Kolmogorov-Smirnov tests.
Triggers Python-side halt if models output out-of-distribution probabilities.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
from scipy import stats
import logging
import zmq

logger = logging.getLogger(__name__)


@dataclass
class DistributionStats:
    """Statistics for a probability distribution."""
    mean: float
    std: float
    skewness: float
    kurtosis: float
    min_val: float
    max_val: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class HallucinationAlert:
    """Alert triggered by hallucination detection."""
    alert_type: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    ks_statistic: float
    p_value: float
    description: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


class MLHallucinationDetector:
    """
    Monitors statistical distribution of ML outputs using rolling KS tests.
    Instantly triggers Python-side halt on extreme out-of-distribution outputs.
    """
    
    def __init__(self,
                 window_size: int = 500,
                 ks_threshold: float = 0.15,
                 p_value_threshold: float = 0.01,
                 num_classes: int = 5):
        
        self.window_size = window_size
        self.ks_threshold = ks_threshold
        self.p_value_threshold = p_value_threshold
        self.num_classes = num_classes
        
        # Reference distribution (learned during calibration)
        self._reference_distribution: Optional[np.ndarray] = None
        self._reference_stats: Optional[DistributionStats] = None
        
        # Rolling windows for each class probability
        self._probability_windows: List[deque] = [
            deque(maxlen=window_size) for _ in range(num_classes)
        ]
        
        # Alert state
        self._halt_triggered = False
        self._alert_history: deque = deque(maxlen=100)
        self._consecutive_violations = 0
        self._max_consecutive_violations = 3
        
        # Pre-allocated arrays for performance
        self._current_probs = np.zeros(num_classes, dtype=np.float64)
        self._reference_cdf = np.zeros(100, dtype=np.float64)
        
        # ZMQ socket for Rust IPC
        self._zmq_context = zmq.Context()
        self._zmq_socket = self._zmq_context.socket(zmq.PUSH)
        self._zmq_socket.connect("tcp://localhost:5564")  # Global Kill Switch
    
    def calibrate(self, sample_probabilities: np.ndarray) -> bool:
        """
        Calibrate reference distribution from historical valid outputs.
        
        Args:
            sample_probabilities: Array of shape (N, num_classes) with valid probabilities
        """
        try:
            if len(sample_probabilities) < 50:
                logger.warning("Insufficient samples for calibration (need >= 50)")
                return False
            
            # Store reference distribution statistics
            self._reference_distribution = sample_probabilities.flatten()
            
            # Calculate reference statistics per class
            means = np.mean(sample_probabilities, axis=0)
            stds = np.std(sample_probabilities, axis=0)
            skewnesses = stats.skew(sample_probabilities, axis=0)
            kurtoses = stats.kurtosis(sample_probabilities, axis=0)
            
            self._reference_stats = DistributionStats(
                mean=float(np.mean(means)),
                std=float(np.mean(stds)),
                skewness=float(np.mean(skewnesses)),
                kurtosis=float(np.mean(kurtoses)),
                min_val=float(np.min(sample_probabilities)),
                max_val=float(np.max(sample_probabilities)),
            )
            
            logger.info(f"Calibrated with {len(sample_probabilities)} samples")
            return True
            
        except Exception as e:
            logger.error(f"Calibration failed: {e}")
            return False
    
    def check_distribution(self, probabilities: np.ndarray) -> Tuple[bool, Optional[HallucinationAlert]]:
        """
        Check if new probability distribution is within expected bounds.
        
        Args:
            probabilities: Array of shape (num_classes,) or (batch, num_classes)
        
        Returns:
            Tuple of (is_valid, alert_if_invalid)
        """
        if self._halt_triggered:
            return False, HallucinationAlert(
                alert_type="HALLUCINATION_HALT_ACTIVE",
                severity="CRITICAL",
                ks_statistic=0.0,
                p_value=0.0,
                description="System halted due to previous hallucination detection",
            )
        
        if self._reference_distribution is None:
            # No calibration yet, just record
            self._record_probabilities(probabilities)
            return True, None
        
        # Ensure 2D array
        if probabilities.ndim == 1:
            probabilities = probabilities.reshape(1, -1)
        
        # Record for rolling window
        self._record_probabilities(probabilities)
        
        # Perform KS test against reference
        ks_stat, p_value = self._perform_ks_test(probabilities)
        
        # Check for extreme values
        extreme_check = self._check_extreme_values(probabilities)
        
        # Check statistical divergence
        divergence_check = self._check_divergence(probabilities)
        
        # Determine if hallucination detected
        is_hallucination = (
            ks_stat > self.ks_threshold or 
            p_value < self.p_value_threshold or
            extreme_check or
            divergence_check
        )
        
        if is_hallucination:
            self._consecutive_violations += 1
            
            if self._consecutive_violations >= self._max_consecutive_violations:
                self._halt_triggered = True
                severity = "CRITICAL"
            else:
                severity = "HIGH" if self._consecutive_violations >= 2 else "MEDIUM"
            
            alert = HallucinationAlert(
                alert_type="ML_HALLUCINATION_DETECTED",
                severity=severity,
                ks_statistic=float(ks_stat),
                p_value=float(p_value),
                description=f"KS={ks_stat:.4f}, p={p_value:.6f}, consecutive={self._consecutive_violations}",
            )
            
            self._alert_history.append(alert)
            self._notify_rust(alert)
            
            return False, alert
        else:
            self._consecutive_violations = 0
            return True, None
    
    def _record_probabilities(self, probabilities: np.ndarray):
        """Record probabilities in rolling windows."""
        for prob_array in probabilities:
            for i, prob in enumerate(prob_array[:self.num_classes]):
                self._probability_windows[i].append(prob)
    
    def _perform_ks_test(self, probabilities: np.ndarray) -> Tuple[float, float]:
        """Perform Kolmogorov-Smirnov test against reference distribution."""
        try:
            # Flatten current and reference
            current_flat = probabilities.flatten()
            
            # KS test
            ks_stat, p_value = stats.ks_2samp(current_flat, self._reference_distribution)
            
            return ks_stat, p_value
            
        except Exception as e:
            logger.error(f"KS test error: {e}")
            return 0.0, 1.0
    
    def _check_extreme_values(self, probabilities: np.ndarray) -> bool:
        """Check for extreme probability values indicating model breakdown."""
        # Check for overconfident predictions (all mass on one class)
        max_probs = np.max(probabilities, axis=1)
        if np.any(max_probs > 0.99):
            logger.warning("Overconfident predictions detected")
            return True
        
        # Check for uniform distribution (model confused)
        entropies = -np.sum(probabilities * np.log(probabilities + 1e-10), axis=1)
        max_entropy = np.log(self.num_classes)
        if np.any(entropies > 0.95 * max_entropy):
            logger.warning("Near-uniform distribution detected")
            return True
        
        return False
    
    def _check_divergence(self, probabilities: np.ndarray) -> bool:
        """Check KL divergence from reference distribution."""
        if self._reference_stats is None:
            return False
        
        current_mean = np.mean(probabilities)
        
        # Simple z-score check on mean
        if self._reference_stats.std > 0:
            z_score = abs(current_mean - self._reference_stats.mean) / self._reference_stats.std
            if z_score > 4.0:  # More than 4 standard deviations
                logger.warning(f"Extreme divergence detected (z={z_score:.2f})")
                return True
        
        return False
    
    def _notify_rust(self, alert: HallucinationAlert):
        """Send alert to Rust Global Kill Switch via ZMQ."""
        try:
            self._zmq_socket.send_json({
                'type': 'HALLUCINATION_ALERT',
                'severity': alert.severity,
                'ks_statistic': alert.ks_statistic,
                'p_value': alert.p_value,
                'halt_triggered': self._halt_triggered,
                'timestamp': alert.timestamp.isoformat(),
            }, flags=zmq.NOBLOCK)
        except Exception as e:
            logger.error(f"Failed to send alert to Rust: {e}")
    
    def reset_halt(self) -> bool:
        """Reset halt state (requires manual intervention)."""
        if not self._halt_triggered:
            return False
        
        logger.warning("Manual halt reset requested")
        self._halt_triggered = False
        self._consecutive_violations = 0
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get detector status."""
        return {
            'halt_triggered': self._halt_triggered,
            'calibrated': self._reference_distribution is not None,
            'consecutive_violations': self._consecutive_violations,
            'alert_count': len(self._alert_history),
            'reference_stats': {
                'mean': self._reference_stats.mean,
                'std': self._reference_stats.std,
            } if self._reference_stats else None,
        }
    
    def shutdown(self):
        """Cleanup resources."""
        self._zmq_socket.close()
        self._zmq_context.term()
        logger.info("MLHallucinationDetector shut down")


# Global instance
_detector: Optional[MLHallucinationDetector] = None


def get_detector() -> MLHallucinationDetector:
    """Get or create the global MLHallucinationDetector instance."""
    global _detector
    if _detector is None:
        _detector = MLHallucinationDetector()
    return _detector


def create_detector(window_size: int = 500,
                   ks_threshold: float = 0.15,
                   p_value_threshold: float = 0.01) -> MLHallucinationDetector:
    """Create a new MLHallucinationDetector with custom configuration."""
    global _detector
    _detector = MLHallucinationDetector(
        window_size=window_size,
        ks_threshold=ks_threshold,
        p_value_threshold=p_value_threshold,
    )
    return _detector
