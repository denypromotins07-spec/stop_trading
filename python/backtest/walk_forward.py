"""
Anchored Walk-Forward analysis pipeline for ML model robustness validation.
Continuously validates models out-of-sample and quarantines degrading models.
"""

import numpy as np
from numba import njit, prange
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass, field
import threading
import time
from enum import IntEnum


class ModelStatus(IntEnum):
    """Model validation status."""
    ACTIVE = 0
    WARNING = 1      # Degrading but usable
    QUARANTINED = 2  # Must be retrained
    ARCHIVED = 3     # No longer used


@njit(cache=True)
def anchored_split(
    n_samples: int,
    initial_train_size: int,
    test_size: int,
    step_size: int
) -> np.ndarray:
    """
    Generate anchored walk-forward split indices.
    Training set grows (anchors), test set slides forward.
    Returns array of (train_start, train_end, test_start, test_end) tuples.
    """
    max_splits = (n_samples - initial_train_size) // step_size
    
    splits = np.zeros((max_splits, 4), dtype=np.int32)
    
    for i in range(max_splits):
        train_start = 0  # Anchored at start
        train_end = initial_train_size + i * step_size
        test_start = train_end
        test_end = min(test_start + test_size, n_samples)
        
        splits[i, 0] = train_start
        splits[i, 1] = train_end
        splits[i, 2] = test_start
        splits[i, 3] = test_end
        
        if test_end >= n_samples:
            break
    
    return splits[:i+1]


@njit(cache=True)
def compute_rolling_sharpe(returns: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling Sharpe ratio."""
    n = len(returns)
    sharpe = np.zeros(n, dtype=np.float64)
    
    for i in range(window, n):
        window_returns = returns[i-window:i]
        mean_ret = np.mean(window_returns)
        std_ret = np.std(window_returns) + 1e-10
        sharpe[i] = np.sqrt(252) * mean_ret / std_ret
    
    return sharpe


@njit(cache=True)
def detect_sharpe_degradation(
    sharpe_series: np.ndarray,
    threshold: float,
    lookback: int
) -> np.ndarray:
    """Detect significant Sharpe ratio degradation."""
    n = len(sharpe_series)
    signals = np.zeros(n, dtype=np.int32)  # 0: normal, 1: warning, 2: critical
    
    for i in range(lookback, n):
        recent = sharpe_series[i-lookback:i]
        previous = sharpe_series[i-2*lookback:i-lookback]
        
        recent_avg = np.mean(recent)
        previous_avg = np.mean(previous)
        
        degradation = previous_avg - recent_avg
        
        if degradation > threshold * 2:
            signals[i] = 2  # Critical
        elif degradation > threshold:
            signals[i] = 1  # Warning
    
    return signals


@dataclass
class WalkForwardResult:
    """Results from a single walk-forward iteration."""
    
    split_index: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    
    # In-sample metrics
    is_sharpe: float = 0.0
    is_return: float = 0.0
    is_max_dd: float = 0.0
    
    # Out-of-sample metrics
    oos_sharpe: float = 0.0
    oos_return: float = 0.0
    oos_max_dd: float = 0.0
    
    # Degradation metrics
    sharpe_decay: float = 0.0  # (IS - OOS) / IS
    overfitting_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "split_index": self.split_index,
            "train_periods": self.train_end - self.train_start,
            "test_periods": self.test_end - self.test_start,
            "is_sharpe": self.is_sharpe,
            "oos_sharpe": self.oos_sharpe,
            "sharpe_decay": self.sharpe_decay,
            "overfitting_score": self.overfitting_score,
            "oos_return": self.oos_return,
            "oos_max_dd": self.oos_max_dd
        }


@dataclass
class ModelValidation:
    """Complete model validation state."""
    
    model_id: str
    status: ModelStatus = ModelStatus.ACTIVE
    
    # Walk-forward results
    wf_results: List[WalkForwardResult] = field(default_factory=list)
    
    # Aggregate metrics
    avg_oos_sharpe: float = 0.0
    avg_is_sharpe: float = 0.0
    avg_sharpe_decay: float = 0.0
    oos_is_ratio: float = 0.0
    
    # Degradation tracking
    recent_sharpe_trend: float = 0.0
    degradation_warnings: int = 0
    
    # Timestamps
    last_validated: float = 0.0
    created_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "status": self.status.name,
            "avg_oos_sharpe": self.avg_oos_sharpe,
            "avg_is_sharpe": self.avg_is_sharpe,
            "avg_sharpe_decay": self.avg_sharpe_decay,
            "oos_is_ratio": self.oos_is_ratio,
            "recent_sharpe_trend": self.recent_sharpe_trend,
            "degradation_warnings": self.degradation_warnings,
            "last_validated": self.last_validated,
            "n_validations": len(self.wf_results)
        }


class WalkForwardAnalyzer:
    """
    Anchored Walk-Forward analysis for continuous model validation.
    Automatically detects and quarantines degrading models.
    """
    
    def __init__(
        self,
        initial_train_size: int = 500,
        test_size: int = 100,
        step_size: int = 50,
        sharpe_degradation_threshold: float = 0.5,
        quarantine_sharpe: float = 0.5,
        min_oos_is_ratio: float = 0.7
    ):
        self.initial_train_size = initial_train_size
        self.test_size = test_size
        self.step_size = step_size
        self.sharpe_degradation_threshold = sharpe_degradation_threshold
        self.quarantine_sharpe = quarantine_sharpe
        self.min_oos_is_ratio = min_oos_is_ratio
        
        # Model validations
        self._validations: Dict[str, ModelValidation] = {}
        
        # Thread safety
        self._lock = threading.RLock()
    
    def register_model(self, model_id: str) -> None:
        """Register a new model for validation."""
        with self._lock:
            if model_id not in self._validations:
                self._validations[model_id] = ModelValidation(model_id=model_id)
    
    def run_walk_forward(
        self,
        model_id: str,
        returns: np.ndarray,
        predictions: np.ndarray,
        actual_signals: np.ndarray
    ) -> WalkForwardResult:
        """
        Run a single walk-forward iteration for a model.
        
        Args:
            model_id: Model identifier
            returns: Asset returns
            predictions: Model predicted signals
            actual_signals: Actual target signals
        """
        with self._lock:
            if model_id not in self._validations:
                self.register_model(model_id)
            
            validation = self._validations[model_id]
            
            n_samples = len(returns)
            
            # Generate splits
            splits = anchored_split(
                n_samples,
                self.initial_train_size,
                self.test_size,
                self.step_size
            )
            
            # Use the next split based on current validation count
            split_idx = len(validation.wf_results)
            
            if split_idx >= len(splits):
                # No more splits available
                return self._create_empty_result(split_idx)
            
            train_start, train_end, test_start, test_end = splits[split_idx]
            
            # Compute in-sample metrics
            is_returns = self._apply_signals(
                returns[:train_end], actual_signals[:train_end], predictions[:train_end]
            )
            is_sharpe = compute_sharpe(is_returns)
            is_max_dd = compute_max_drawdown(is_returns)
            
            # Compute out-of-sample metrics
            oos_returns = self._apply_signals(
                returns[test_start:test_end],
                actual_signals[test_start:test_end],
                predictions[test_start:test_end]
            )
            oos_sharpe = compute_sharpe(oos_returns)
            oos_max_dd = compute_max_drawdown(oos_returns)
            
            # Compute degradation
            sharpe_decay = (is_sharpe - oos_sharpe) / max(abs(is_sharpe), 0.01)
            overfitting = max(0, sharpe_decay)
            
            result = WalkForwardResult(
                split_index=split_idx,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                is_sharpe=is_sharpe,
                is_return=np.sum(is_returns),
                is_max_dd=is_max_dd,
                oos_sharpe=oos_sharpe,
                oos_return=np.sum(oos_returns),
                oos_max_dd=oos_max_dd,
                sharpe_decay=sharpe_decay,
                overfitting_score=overfitting
            )
            
            # Update validation state
            validation.wf_results.append(result)
            self._update_aggregates(validation)
            self._check_status(validation)
            validation.last_validated = time.time()
            
            return result
    
    def _apply_signals(
        self,
        returns: np.ndarray,
        actual: np.ndarray,
        predicted: np.ndarray
    ) -> np.ndarray:
        """Apply predicted signals to returns."""
        n = len(returns)
        strat_returns = np.zeros(n, dtype=np.float64)
        
        for i in range(1, n):
            signal = predicted[i-1]
            if signal != 0:
                strat_returns[i] = signal * returns[i] - 0.0005  # Transaction cost
        
        return strat_returns
    
    def _update_aggregates(self, validation: ModelValidation) -> None:
        """Update aggregate metrics from walk-forward results."""
        if not validation.wf_results:
            return
        
        results = validation.wf_results
        
        validation.avg_is_sharpe = np.mean([r.is_sharpe for r in results])
        validation.avg_oos_sharpe = np.mean([r.oos_sharpe for r in results])
        validation.avg_sharpe_decay = np.mean([r.sharpe_decay for r in results])
        
        if validation.avg_is_sharpe != 0:
            validation.oos_is_ratio = validation.avg_oos_sharpe / validation.avg_is_sharpe
        else:
            validation.oos_is_ratio = 1.0
        
        # Recent trend (last 5 results)
        recent = results[-5:] if len(results) >= 5 else results
        if len(recent) >= 2:
            recent_sharpes = [r.oos_sharpe for r in recent]
            validation.recent_sharpe_trend = (recent_sharpes[-1] - recent_sharpes[0]) / len(recent_sharpes)
    
    def _check_status(self, validation: ModelValidation) -> None:
        """Check and update model status based on validation results."""
        if len(validation.wf_results) < 3:
            return  # Need minimum samples
        
        recent_results = validation.wf_results[-3:]
        avg_oos_sharpe = np.mean([r.oos_sharpe for r in recent_results])
        avg_decay = np.mean([r.sharpe_decay for r in recent_results])
        
        # Check for quarantine conditions
        should_quarantine = False
        
        if avg_oos_sharpe < self.quarantine_sharpe:
            should_quarantine = True
        
        if avg_decay > 0.5:  # More than 50% decay
            should_quarantine = True
        
        if validation.oos_is_ratio < self.min_oos_is_ratio:
            should_quarantine = True
        
        if should_quarantine:
            validation.status = ModelStatus.QUARANTINED
            validation.degradation_warnings += 1
        elif avg_decay > 0.3 or validation.recent_sharpe_trend < -0.1:
            validation.status = ModelStatus.WARNING
        else:
            validation.status = ModelStatus.ACTIVE
    
    def _create_empty_result(self, split_index: int) -> WalkForwardResult:
        """Create empty result when no data available."""
        return WalkForwardResult(
            split_index=split_index,
            train_start=0,
            train_end=0,
            test_start=0,
            test_end=0
        )
    
    def get_model_status(self, model_id: str) -> Optional[ModelStatus]:
        """Get current model status."""
        with self._lock:
            if model_id in self._validations:
                return self._validations[model_id].status
            return None
    
    def should_use_model(self, model_id: str) -> bool:
        """Check if model should be used for trading."""
        status = self.get_model_status(model_id)
        return status == ModelStatus.ACTIVE
    
    def get_quarantined_models(self) -> List[str]:
        """Get list of quarantined model IDs."""
        with self._lock:
            return [
                mid for mid, val in self._validations.items()
                if val.status == ModelStatus.QUARANTINED
            ]
    
    def get_validation_summary(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get validation summary for a model."""
        with self._lock:
            if model_id in self._validations:
                return self._validations[model_id].to_dict()
            return None
    
    def reset_model(self, model_id: str) -> None:
        """Reset validation history for a model."""
        with self._lock:
            if model_id in self._validations:
                val = self._validations[model_id]
                val.wf_results.clear()
                val.status = ModelStatus.ACTIVE
                val.avg_oos_sharpe = 0.0
                val.avg_is_sharpe = 0.0
                val.avg_sharpe_decay = 0.0
                val.oos_is_ratio = 0.0
                val.recent_sharpe_trend = 0.0
                val.degradation_warnings = 0


# Helper functions (need to be defined for numba compatibility)
@njit(cache=True)
def compute_sharpe(returns: np.ndarray) -> float:
    """Compute Sharpe ratio."""
    n = len(returns)
    if n < 2:
        return 0.0
    
    mean_ret = np.mean(returns)
    std_ret = np.std(returns) + 1e-10
    
    return np.sqrt(252) * mean_ret / std_ret


@njit(cache=True)
def compute_max_drawdown(equity: np.ndarray) -> float:
    """Compute maximum drawdown."""
    n = len(equity)
    if n == 0:
        return 0.0
    
    # Convert returns to equity
    cum_equity = np.zeros(n + 1, dtype=np.float64)
    cum_equity[0] = 1.0
    
    for i in range(n):
        cum_equity[i + 1] = cum_equity[i] * (1 + equity[i])
    
    running_max = cum_equity[0]
    max_dd = 0.0
    
    for i in range(len(cum_equity)):
        if cum_equity[i] > running_max:
            running_max = cum_equity[i]
        
        if running_max > 0:
            dd = (running_max - cum_equity[i]) / running_max
            if dd > max_dd:
                max_dd = dd
    
    return max_dd


# Global singleton instance
_wf_instance: Optional[WalkForwardAnalyzer] = None
_instance_lock = threading.Lock()


def get_wf_analyzer() -> WalkForwardAnalyzer:
    """Get or create the global walk-forward analyzer."""
    global _wf_instance
    if _wf_instance is None:
        with _instance_lock:
            if _wf_instance is None:
                _wf_instance = WalkForwardAnalyzer()
    return _wf_instance


if __name__ == "__main__":
    # Test walk-forward analyzer
    print("Testing WalkForwardAnalyzer:")
    
    analyzer = WalkForwardAnalyzer(
        initial_train_size=200,
        test_size=50,
        step_size=25
    )
    
    np.random.seed(42)
    n_samples = 500
    
    # Generate synthetic data
    returns = 0.0001 + 0.02 * np.random.randn(n_samples)
    
    # Simulate model predictions (with some degradation over time)
    predictions = np.zeros(n_samples)
    actual = np.sign(returns)
    
    for i in range(n_samples):
        # Good predictions early, degrading later
        if i < 300:
            predictions[i] = actual[i] if np.random.random() > 0.3 else -actual[i]
        else:
            predictions[i] = actual[i] if np.random.random() > 0.5 else -actual[i]
    
    # Register model
    analyzer.register_model("test_model")
    
    # Run multiple walk-forward iterations
    print("\n--- Walk-Forward Results ---")
    for i in range(5):
        result = analyzer.run_walk_forward("test_model", returns, predictions, actual)
        
        if result.test_end > 0:
            print(f"\nSplit {result.split_index}:")
            print(f"  IS Sharpe: {result.is_sharpe:.4f}")
            print(f"  OOS Sharpe: {result.oos_sharpe:.4f}")
            print(f"  Sharpe Decay: {result.sharpe_decay:.4f}")
    
    # Get summary
    summary = analyzer.get_validation_summary("test_model")
    print(f"\nValidation Summary: {summary}")
    
    # Check status
    status = analyzer.get_model_status("test_model")
    print(f"Model Status: {status.name if status else 'Unknown'}")
    
    # Check if should use
    print(f"Should Use Model: {analyzer.should_use_model('test_model')}")
