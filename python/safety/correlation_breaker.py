"""
Correlation Breaker
Stage 49: Real-time eigenvalue spectrum analysis of portfolio correlation matrix.
Detects systemic panic when leading eigenvalue approaches N (all correlations -> 1.0).
Uses scipy.linalg.eigh optimized for symmetric matrices (O(N^2) time).
"""

import numpy as np
from scipy import linalg
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
import logging
import zmq

logger = logging.getLogger(__name__)


@dataclass
class CorrelationAlert:
    """Alert triggered by correlation breakdown detection."""
    alert_type: str
    severity: str
    leading_eigenvalue: float
    eigenvalue_ratio: float
    num_assets: int
    description: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


class CorrelationBreaker:
    """
    Calculates real-time eigenvalue spectrum of portfolio correlation matrix.
    Halts all new risk-taking if leading eigenvalue approaches N (systemic panic).
    
    Uses scipy.linalg.eigh optimized for symmetric matrices to achieve O(N^2) time.
    """
    
    # Threshold ratios for alerts
    WARNING_RATIO = 0.5      # Leading eigenvalue / N
    CRITICAL_RATIO = 0.7     # Approaching full correlation
    PANIC_RATIO = 0.85       # Systemic panic imminent
    
    def __init__(self,
                 num_assets: int,
                 window_size: int = 252,
                 min_samples: int = 60):
        
        self.num_assets = num_assets
        self.window_size = window_size
        self.min_samples = min_samples
        
        # Rolling return windows for each asset
        self._return_windows: List[deque] = [
            deque(maxlen=window_size) for _ in range(num_assets)
        ]
        
        # Pre-allocated arrays
        self._returns_matrix = np.zeros((window_size, num_assets), dtype=np.float64)
        self._correlation_matrix = np.zeros((num_assets, num_assets), dtype=np.float64)
        self._eigenvalues = np.zeros(num_assets, dtype=np.float64)
        
        # Alert state
        self._halt_triggered = False
        self._alert_history: deque = deque(maxlen=100)
        self._last_leading_eigenvalue = 0.0
        self._consecutive_high_correlation = 0
        
        # ZMQ socket for Rust IPC
        self._zmq_context = zmq.Context()
        self._zmq_socket = self._zmq_context.socket(zmq.PUSH)
        self._zmq_socket.connect("tcp://localhost:5565")  # Global Kill Switch
    
    def add_returns(self, asset_id: int, returns: np.ndarray) -> None:
        """Add returns for an asset to the rolling window."""
        if asset_id < 0 or asset_id >= self.num_assets:
            logger.warning(f"Invalid asset_id: {asset_id}")
            return
        
        for ret in returns:
            self._return_windows[asset_id].append(ret)
    
    def check_correlation(self) -> Tuple[bool, Optional[CorrelationAlert]]:
        """
        Check current correlation matrix for systemic risk.
        
        Returns:
            Tuple of (is_safe, alert_if_unsafe)
        """
        # Check if we have enough data
        min_window_len = min(len(w) for w in self._return_windows)
        if min_window_len < self.min_samples:
            return True, None
        
        # Build returns matrix
        for i, window in enumerate(self._return_windows):
            self._returns_matrix[:len(window), i] = list(window)
        
        # Calculate correlation matrix
        self._calculate_correlation_matrix()
        
        # Compute eigenvalues using eigh (optimized for symmetric matrices)
        try:
            # eigh is O(N^2) for tridiagonal reduction + QR iteration
            # Much faster than general eig for symmetric positive semi-definite
            eigenvalues = linalg.eigh(
                self._correlation_matrix,
                subset_by_index=[self.num_assets - 1, self.num_assets - 1],  # Only largest
                driver='evr'  # Fastest for subset
            )[0]
            
            leading_eigenvalue = eigenvalues[0] if len(eigenvalues) > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Eigenvalue computation failed: {e}")
            # Fallback to full eigendecomposition
            self._eigenvalues = linalg.eigvalsh(self._correlation_matrix)
            leading_eigenvalue = np.max(self._eigenvalues)
        
        self._last_leading_eigenvalue = leading_eigenvalue
        
        # Calculate ratio (leading eigenvalue / N)
        # In fully correlated market, leading eigenvalue approaches N
        eigenvalue_ratio = leading_eigenvalue / self.num_assets
        
        # Check thresholds
        if eigenvalue_ratio >= self.PANIC_RATIO:
            self._halt_triggered = True
            self._consecutive_high_correlation += 1
            
            alert = CorrelationAlert(
                alert_type="CORRELATION_PANIC",
                severity="CRITICAL",
                leading_eigenvalue=float(leading_eigenvalue),
                eigenvalue_ratio=float(eigenvalue_ratio),
                num_assets=self.num_assets,
                description=f"Systemic panic: λ_max={leading_eigenvalue:.2f}/{self.num_assets} ({eigenvalue_ratio*100:.1f}%)",
            )
            
            self._alert_history.append(alert)
            self._notify_rust(alert)
            
            return False, alert
            
        elif eigenvalue_ratio >= self.CRITICAL_RATIO:
            self._consecutive_high_correlation += 1
            
            if self._consecutive_high_correlation >= 3:
                self._halt_triggered = True
                severity = "CRITICAL"
            else:
                severity = "HIGH"
            
            alert = CorrelationAlert(
                alert_type="CORRELATION_CRITICAL",
                severity=severity,
                leading_eigenvalue=float(leading_eigenvalue),
                eigenvalue_ratio=float(eigenvalue_ratio),
                num_assets=self.num_assets,
                description=f"Critical correlation: λ_max={leading_eigenvalue:.2f}/{self.num_assets}",
            )
            
            self._alert_history.append(alert)
            self._notify_rust(alert)
            
            return False, alert
            
        elif eigenvalue_ratio >= self.WARNING_RATIO:
            self._consecutive_high_correlation = max(0, self._consecutive_high_correlation - 1)
            
            alert = CorrelationAlert(
                alert_type="CORRELATION_WARNING",
                severity="MEDIUM",
                leading_eigenvalue=float(leading_eigenvalue),
                eigenvalue_ratio=float(eigenvalue_ratio),
                num_assets=self.num_assets,
                description=f"Elevated correlation: λ_max={leading_eigenvalue:.2f}/{self.num_assets}",
            )
            
            # Don't store warnings in history to avoid clutter
            self._notify_rust(alert)
            
            return True, alert  # Still safe but warning
        
        else:
            self._consecutive_high_correlation = 0
            return True, None
    
    def _calculate_correlation_matrix(self):
        """Calculate correlation matrix from returns using numpy (vectorized)."""
        # Get valid data length
        valid_len = min(len(w) for w in self._return_windows)
        
        if valid_len < 2:
            return
        
        # Extract valid data
        data = np.zeros((valid_len, self.num_assets), dtype=np.float64)
        for i, window in enumerate(self._return_windows):
            data[:, i] = list(window)[-valid_len:]
        
        # Calculate correlation matrix (numpy corrcoef is optimized)
        self._correlation_matrix = np.corrcoef(data.T)
        
        # Handle NaN values (can occur with constant returns)
        np.nan_to_num(self._correlation_matrix, nan=0.0, copy=False)
    
    def get_eigenvalue_spectrum(self) -> Optional[np.ndarray]:
        """Get full eigenvalue spectrum for analysis."""
        min_window_len = min(len(w) for w in self._return_windows)
        if min_window_len < self.min_samples:
            return None
        
        self._calculate_correlation_matrix()
        
        try:
            self._eigenvalues = linalg.eigvalsh(self._correlation_matrix)
            return np.sort(self._eigenvalues)[::-1]  # Descending order
        except Exception as e:
            logger.error(f"Failed to compute eigenvalue spectrum: {e}")
            return None
    
    def get_diversification_ratio(self) -> float:
        """
        Calculate portfolio diversification ratio.
        Higher ratio = more diversified, lower = more correlated.
        """
        spectrum = self.get_eigenvalue_spectrum()
        if spectrum is None or len(spectrum) == 0:
            return 1.0
        
        # Diversification ratio based on eigenvalue distribution
        # More uniform eigenvalues = higher diversification
        entropy = -np.sum((spectrum / np.sum(spectrum)) * np.log(spectrum / np.sum(spectrum) + 1e-10))
        max_entropy = np.log(self.num_assets)
        
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def _notify_rust(self, alert: CorrelationAlert):
        """Send alert to Rust Global Kill Switch via ZMQ."""
        try:
            self._zmq_socket.send_json({
                'type': 'CORRELATION_ALERT',
                'severity': alert.severity,
                'leading_eigenvalue': alert.leading_eigenvalue,
                'eigenvalue_ratio': alert.eigenvalue_ratio,
                'halt_triggered': self._halt_triggered,
                'diversification_ratio': self.get_diversification_ratio(),
                'timestamp': alert.timestamp.isoformat(),
            }, flags=zmq.NOBLOCK)
        except Exception as e:
            logger.error(f"Failed to send alert to Rust: {e}")
    
    def reset_halt(self) -> bool:
        """Reset halt state (requires manual intervention)."""
        if not self._halt_triggered:
            return False
        
        logger.warning("Manual correlation halt reset requested")
        self._halt_triggered = False
        self._consecutive_high_correlation = 0
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get breaker status."""
        return {
            'halt_triggered': self._halt_triggered,
            'leading_eigenvalue': float(self._last_leading_eigenvalue),
            'eigenvalue_ratio': float(self._last_leading_eigenvalue / self.num_assets) if self.num_assets > 0 else 0.0,
            'consecutive_high_correlation': self._consecutive_high_correlation,
            'diversification_ratio': self.get_diversification_ratio(),
            'min_window_length': min(len(w) for w in self._return_windows),
        }
    
    def shutdown(self):
        """Cleanup resources."""
        self._zmq_socket.close()
        self._zmq_context.term()
        logger.info("CorrelationBreaker shut down")


# Global instance
_breaker: Optional[CorrelationBreaker] = None


def get_breaker(num_assets: int = 50) -> CorrelationBreaker:
    """Get or create the global CorrelationBreaker instance."""
    global _breaker
    if _breaker is None:
        _breaker = CorrelationBreaker(num_assets=num_assets)
    return _breaker


def create_breaker(num_assets: int = 50,
                  window_size: int = 252,
                  min_samples: int = 60) -> CorrelationBreaker:
    """Create a new CorrelationBreaker with custom configuration."""
    global _breaker
    _breaker = CorrelationBreaker(
        num_assets=num_assets,
        window_size=window_size,
        min_samples=min_samples,
    )
    return _breaker
