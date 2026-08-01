"""
Shadow PnL Variance Analyzer
Compares shadow PnL against live production PnL to detect execution degradation, 
model hallucination, or adverse market impact.

Automatically flags when the live execution deviates significantly from the 
theoretical shadow performance, triggering model quarantine.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import threading
import time
import statistics


class VarianceStatus(Enum):
    """Variance analysis status."""
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    QUARANTINE = "quarantine"


@dataclass
class PnLSnapshot:
    """Snapshot of PnL at a point in time."""
    timestamp: float
    shadow_pnl: float
    live_pnl: float
    shadow_position: float
    live_position: float
    shadow_fees: float
    live_fees: float
    shadow_slippage: float
    live_slippage: float


@dataclass
class VarianceReport:
    """Report on PnL variance analysis."""
    timestamp: float
    status: VarianceStatus
    pnl_difference: float
    pnl_difference_pct: float
    position_difference: float
    fee_difference: float
    slippage_difference: float
    
    # Statistical measures
    z_score: float
    rolling_correlation: float
    max_drawdown_diff: float
    
    # Flags
    execution_degradation: bool
    model_hallucination: bool
    adverse_impact: bool
    
    recommendation: str


class ShadowVarianceAnalyzer:
    """
    Analyzes variance between shadow and live PnL.
    Detects execution issues and model problems.
    """
    
    def __init__(self,
                 warning_threshold_pct: float = 1.0,
                 critical_threshold_pct: float = 3.0,
                 quarantine_threshold_pct: float = 5.0,
                 rolling_window: int = 100,
                 min_samples_for_analysis: int = 20):
        """
        Initialize variance analyzer.
        
        Args:
            warning_threshold_pct: Warning threshold for PnL difference
            critical_threshold_pct: Critical threshold for PnL difference
            quarantine_threshold_pct: Threshold for model quarantine
            rolling_window: Rolling window for statistics
            min_samples_for_analysis: Minimum samples before analysis
        """
        self.warning_threshold = warning_threshold_pct
        self.critical_threshold = critical_threshold_pct
        self.quarantine_threshold = quarantine_threshold_pct
        self.rolling_window = rolling_window
        self.min_samples = min_samples_for_analysis
        
        # Snapshot history
        self._snapshots: deque = deque(maxlen=rolling_window)
        
        # PnL difference history
        self._pnl_diffs: deque = deque(maxlen=rolling_window)
        self._position_diffs: deque = deque(maxlen=rolling_window)
        
        # Statistics
        self._mean_diff = 0.0
        self._std_diff = 0.0
        self._correlation = 1.0
        
        # Quarantine state
        self._quarantine_active = False
        self._quarantine_reason: Optional[str] = None
        
        # Thread safety
        self._lock = threading.Lock()
    
    def record_snapshot(self, snapshot: PnLSnapshot):
        """Record a PnL snapshot."""
        with self._lock:
            self._snapshots.append(snapshot)
            
            # Calculate differences
            pnl_diff = snapshot.shadow_pnl - snapshot.live_pnl
            position_diff = snapshot.shadow_position - snapshot.live_position
            
            self._pnl_diffs.append(pnl_diff)
            self._position_diffs.append(position_diff)
            
            # Update statistics
            self._update_statistics()
    
    def _update_statistics(self):
        """Update rolling statistics."""
        if len(self._pnl_diffs) < 2:
            return
        
        diffs = list(self._pnl_diffs)
        
        # Mean and std
        self._mean_diff = statistics.mean(diffs)
        self._std_diff = statistics.stdev(diffs) if len(diffs) > 1 else 0.0
        
        # Correlation between shadow and live PnL
        if len(self._snapshots) >= 2:
            shadow_pnls = [s.shadow_pnl for s in self._snapshots]
            live_pnls = [s.live_pnl for s in self._snapshots]
            
            if np.std(shadow_pnls) > 0 and np.std(live_pnls) > 0:
                self._correlation = np.corrcoef(shadow_pnls, live_pnls)[0, 1]
            else:
                self._correlation = 1.0
    
    def analyze(self) -> VarianceReport:
        """Perform variance analysis."""
        with self._lock:
            if len(self._snapshots) < self.min_samples:
                return self._get_insufficient_data_report()
            
            latest = self._snapshots[-1]
            
            # Calculate absolute PnL difference
            pnl_diff = latest.shadow_pnl - latest.live_pnl
            
            # Calculate percentage difference relative to notional
            avg_notional = (abs(latest.shadow_pnl) + abs(latest.live_pnl)) / 2 + 1e-6
            pnl_diff_pct = abs(pnl_diff) / avg_notional * 100
            
            # Position difference
            position_diff = abs(latest.shadow_position - latest.live_position)
            
            # Fee and slippage differences
            fee_diff = latest.shadow_fees - latest.live_fees
            slippage_diff = latest.shadow_slippage - latest.live_slippage
            
            # Z-score of current difference
            z_score = abs(pnl_diff - self._mean_diff) / (self._std_diff + 1e-6) if self._std_diff > 0 else 0
            
            # Determine status
            status, recommendation = self._determine_status(
                pnl_diff_pct, z_score, position_diff
            )
            
            # Detect specific issues
            execution_degradation = self._detect_execution_degradation(
                fee_diff, slippage_diff, z_score
            )
            model_hallucination = self._detect_model_hallucination(
                pnl_diff_pct, self._correlation, z_score
            )
            adverse_impact = self._detect_adverse_impact(
                slippage_diff, pnl_diff, latest.live_position
            )
            
            # Calculate max drawdown difference
            max_dd_diff = self._calculate_max_drawdown_diff()
            
            # Auto-quarantine if threshold exceeded
            if pnl_diff_pct >= self.quarantine_threshold:
                self._quarantine_active = True
                self._quarantine_reason = f"PnL variance {pnl_diff_pct:.2f}% exceeds quarantine threshold"
            
            return VarianceReport(
                timestamp=time.time(),
                status=status,
                pnl_difference=pnl_diff,
                pnl_difference_pct=pnl_diff_pct,
                position_difference=position_diff,
                fee_difference=fee_diff,
                slippage_difference=slippage_diff,
                z_score=z_score,
                rolling_correlation=self._correlation,
                max_drawdown_diff=max_dd_diff,
                execution_degradation=execution_degradation,
                model_hallucination=model_hallucination,
                adverse_impact=adverse_impact,
                recommendation=recommendation
            )
    
    def _determine_status(self, 
                          pnl_diff_pct: float,
                          z_score: float,
                          position_diff: float) -> Tuple[VarianceStatus, str]:
        """Determine overall status."""
        if self._quarantine_active:
            return VarianceStatus.QUARANTINE, "MODEL_QUARANTINE_ACTIVE"
        
        if pnl_diff_pct >= self.quarantine_threshold or z_score > 4.0:
            return VarianceStatus.CRITICAL, "IMMEDIATE_INVESTIGATION_REQUIRED"
        
        if pnl_diff_pct >= self.critical_threshold or z_score > 3.0:
            return VarianceStatus.CRITICAL, "CRITICAL_VARIANCE_DETECTED"
        
        if pnl_diff_pct >= self.warning_threshold or z_score > 2.0:
            return VarianceStatus.WARNING, "MONITOR_CLOSELY"
        
        return VarianceStatus.NORMAL, "WITHIN_ACCEPTABLE_BOUNDS"
    
    def _detect_execution_degradation(self,
                                       fee_diff: float,
                                       slippage_diff: float,
                                       z_score: float) -> bool:
        """Detect if live execution is degrading vs shadow."""
        # Live fees significantly higher than shadow
        if fee_diff < -0.0001:  # More than 1 bps difference
            return True
        
        # Live slippage significantly worse
        if slippage_diff < -0.0005:  # More than 5 bps difference
            return True
        
        # High z-score with negative PnL diff (live underperforming)
        if z_score > 2.5 and self._mean_diff < 0:
            return True
        
        return False
    
    def _detect_model_hallucination(self,
                                     pnl_diff_pct: float,
                                     correlation: float,
                                     z_score: float) -> bool:
        """Detect potential model hallucination."""
        # Low correlation suggests shadow and live are diverging fundamentally
        if correlation < 0.5 and len(self._snapshots) >= 20:
            return True
        
        # Extreme z-score suggests anomalous behavior
        if z_score > 5.0:
            return True
        
        return False
    
    def _detect_adverse_impact(self,
                                slippage_diff: float,
                                pnl_diff: float,
                                live_position: float) -> bool:
        """Detect adverse market impact."""
        # Consistently worse slippage on live
        if slippage_diff < -0.001:  # More than 10 bps
            return True
        
        # Large positions with large PnL differences
        if abs(live_position) > 1.0 and abs(pnl_diff) > 0.01:
            return True
        
        return False
    
    def _calculate_max_drawdown_diff(self) -> float:
        """Calculate maximum drawdown difference between shadow and live."""
        if len(self._snapshots) < 2:
            return 0.0
        
        shadow_pnls = [s.shadow_pnl for s in self._snapshots]
        live_pnls = [s.live_pnl for s in self._snapshots]
        
        # Calculate running max and drawdowns
        shadow_max = np.maximum.accumulate(shadow_pnls)
        live_max = np.maximum.accumulate(live_pnls)
        
        shadow_dd = np.max(shadow_max - np.array(shadow_pnls))
        live_dd = np.max(live_max - np.array(live_pnls))
        
        return live_dd - shadow_dd
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get analyzer statistics."""
        with self._lock:
            return {
                'sample_count': len(self._snapshots),
                'mean_pnl_diff': self._mean_diff,
                'std_pnl_diff': self._std_diff,
                'correlation': self._correlation,
                'quarantine_active': self._quarantine_active,
                'quarantine_reason': self._quarantine_reason
            }
    
    def reset_quarantine(self):
        """Manually reset quarantine state."""
        with self._lock:
            self._quarantine_active = False
            self._quarantine_reason = None
    
    def clear_history(self):
        """Clear all historical data."""
        with self._lock:
            self._snapshots.clear()
            self._pnl_diffs.clear()
            self._position_diffs.clear()
            self._mean_diff = 0.0
            self._std_diff = 0.0
            self._correlation = 1.0
            self._quarantine_active = False
            self._quarantine_reason = None


class ModelQuarantineManager:
    """
    Manages model quarantine decisions based on variance analysis.
    """
    
    def __init__(self, analyzer: ShadowVarianceAnalyzer):
        """Initialize quarantine manager."""
        self.analyzer = analyzer
        self._quarantine_start_time: Optional[float] = None
        self._quarantine_duration_sec: float = 0.0
        self._total_quarantines = 0
    
    def check_and_quarantine(self) -> Optional[str]:
        """
        Check if quarantine is needed and apply it.
        
        Returns:
            Quarantine reason if quarantined, None otherwise
        """
        report = self.analyzer.analyze()
        
        if report.status == VarianceStatus.QUARANTINE:
            if not self._quarantine_start_time:
                self._quarantine_start_time = time.time()
                self._total_quarantines += 1
            
            return report.recommendation
        
        return None
    
    def release_quarantine(self) -> bool:
        """
        Attempt to release quarantine.
        
        Returns:
            True if released, False if still quarantined
        """
        if not self._quarantine_start_time:
            return True  # Not quarantined
        
        report = self.analyzer.analyze()
        
        if report.status != VarianceStatus.QUARANTINE:
            self.analyzer.reset_quarantine()
            self._quarantine_duration_sec += time.time() - self._quarantine_start_time
            self._quarantine_start_time = None
            return True
        
        return False
    
    def get_quarantine_stats(self) -> Dict[str, Any]:
        """Get quarantine statistics."""
        active_duration = 0.0
        if self._quarantine_start_time:
            active_duration = time.time() - self._quarantine_start_time
        
        return {
            'is_quarantined': self._quarantine_start_time is not None,
            'active_duration_sec': active_duration,
            'total_duration_sec': self._quarantine_duration_sec,
            'total_quarantines': self._total_quarantines
        }


# Module exports
__all__ = [
    'VarianceStatus',
    'PnLSnapshot',
    'VarianceReport',
    'ShadowVarianceAnalyzer',
    'ModelQuarantineManager'
]
