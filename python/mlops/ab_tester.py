"""
A/B Tester - Sequential Probability Ratio Test (SPRT) for model comparison.
Determines statistical significance between production and canary models.
Memory-efficient implementation for continuous monitoring.
"""
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import time
from collections import deque

logger = logging.getLogger(__name__)


class SequentialProbabilityRatioTest:
    """
    Wald's Sequential Probability Ratio Test (SPRT) for A/B testing.
    Provides early stopping when statistical significance is reached.
    """
    
    def __init__(self, alpha: float = 0.05, beta: float = 0.10,
                 effect_size: float = 0.01):
        """
        Initialize SPRT.
        
        Args:
            alpha: Type I error rate (false positive)
            beta: Type II error rate (false negative)
            effect_size: Minimum detectable effect size
        """
        self.alpha = alpha
        self.beta = beta
        self.effect_size = effect_size
        
        # Calculate decision boundaries
        # A = log((1 - beta) / alpha)
        # B = log(beta / (1 - alpha))
        self.A = np.log((1 - beta) / alpha)
        self.B = np.log(beta / (1 - alpha))
        
        # Likelihood ratio (log scale)
        self.log_lambda = 0.0
        
        # Sample counts
        self._n_a = 0  # Production samples
        self._n_b = 0  # Canary samples
        
        # Running statistics
        self._sum_a = 0.0
        self._sum_sq_a = 0.0
        self._sum_b = 0.0
        self._sum_sq_b = 0.0
        
        # Decision history
        self._decisions: deque = deque(maxlen=1000)
        
        # State
        self._test_started = False
        self._test_concluded = False
        self._conclusion = None
        
        logger.info(f"SPRT initialized: alpha={alpha}, beta={beta}, effect={effect_size}")
    
    def update(self, metric_a: float, metric_b: float) -> Optional[str]:
        """
        Update test with new paired observations.
        
        Args:
            metric_a: Metric from production (control)
            metric_b: Metric from canary (treatment)
            
        Returns:
            Decision if concluded: 'accept', 'reject', or None if continuing
        """
        self._test_started = True
        
        # Update sample counts
        self._n_a += 1
        self._n_b += 1
        
        # Update running sums
        self._sum_a += metric_a
        self._sum_sq_a += metric_a ** 2
        self._sum_b += metric_b
        self._sum_sq_b += metric_b ** 2
        
        # Calculate likelihood ratio increment
        # Assuming normal distribution with known variance
        # Simplified: use difference in means
        diff = metric_b - metric_a
        
        # Log-likelihood ratio update (simplified for normal means)
        # LLR += (diff * effect_size) / sigma^2
        # Using unit variance assumption
        self.log_lambda += diff * self.effect_size
        
        # Check decision boundaries
        if self.log_lambda >= self.A:
            self._test_concluded = True
            self._conclusion = 'accept'  # Accept alternative (canary is better)
            self._decisions.append(('accept', time.time(), self.log_lambda))
            return 'accept'
        
        elif self.log_lambda <= self.B:
            self._test_concluded = True
            self._conclusion = 'reject'  # Reject alternative (canary is worse)
            self._decisions.append(('reject', time.time(), self.log_lambda))
            return 'reject'
        
        # Continue test
        self._decisions.append(('continue', time.time(), self.log_lambda))
        return None
    
    def get_status(self) -> Dict[str, Any]:
        """Get current test status."""
        status = 'not_started'
        if self._test_concluded:
            status = 'concluded'
        elif self._test_started:
            status = 'running'
        
        return {
            'status': status,
            'conclusion': self._conclusion,
            'n_production': self._n_a,
            'n_canary': self._n_b,
            'log_likelihood_ratio': self.log_lambda,
            'upper_bound': self.A,
            'lower_bound': self.B,
            'mean_production': self._sum_a / max(1, self._n_a),
            'mean_canary': self._sum_b / max(1, self._n_b),
            'variance_production': (
                self._sum_sq_a / max(1, self._n_a) - 
                (self._sum_a / max(1, self._n_a)) ** 2
            ),
            'variance_canary': (
                self._sum_sq_b / max(1, self._n_b) - 
                (self._sum_b / max(1, self._n_b)) ** 2
            )
        }
    
    def reset(self) -> None:
        """Reset test state."""
        self.log_lambda = 0.0
        self._n_a = 0
        self._n_b = 0
        self._sum_a = 0.0
        self._sum_sq_a = 0.0
        self._sum_b = 0.0
        self._sum_sq_b = 0.0
        self._test_started = False
        self._test_concluded = False
        self._conclusion = None
        self._decisions.clear()
        logger.info("SPRT reset")
    
    def is_significant(self, confidence: float = 0.95) -> bool:
        """Check if result is statistically significant."""
        if not self._test_concluded:
            return False
        
        return self._conclusion in ('accept', 'reject')
    
    def get_confidence(self) -> float:
        """Estimate confidence level based on position between bounds."""
        if not self._test_started:
            return 0.0
        
        range_size = self.A - self.B
        position = (self.log_lambda - self.B) / range_size
        
        # Map to confidence (0.5 to 1.0)
        confidence = 0.5 + abs(position - 0.5)
        return min(1.0, confidence)


class ABTester:
    """
    A/B testing framework for production vs canary model comparison.
    Uses SPRT for sequential testing with early stopping.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize SPRT
        self.sprt = SequentialProbabilityRatioTest(
            alpha=self.config.get('alpha', 0.05),
            beta=self.config.get('beta', 0.10),
            effect_size=self.config.get('effect_size', 0.01)
        )
        
        # Metrics storage (bounded)
        self._max_samples = self.config.get('max_samples', 10_000)
        self._production_metrics: deque = deque(maxlen=self._max_samples)
        self._canary_metrics: deque = deque(maxlen=self._max_samples)
        
        # PnL tracking
        self._pnl_production = 0.0
        self._pnl_canary = 0.0
        
        # Test state
        self._test_active = False
        self._test_start_time = None
        
        # Divergence tracking
        self._max_pnl_divergence = self.config.get('max_pnl_divergence', 0.05)
        self._divergence_alerts = 0
        
        logger.info("ABTester initialized")
    
    def start_test(self) -> None:
        """Start a new A/B test."""
        self.sprt.reset()
        self._production_metrics.clear()
        self._canary_metrics.clear()
        self._pnl_production = 0.0
        self._pnl_canary = 0.0
        self._test_active = True
        self._test_start_time = time.time()
        logger.info("A/B test started")
    
    def stop_test(self) -> Dict[str, Any]:
        """Stop current A/B test and return results."""
        self._test_active = False
        
        return {
            'status': 'stopped',
            'duration_seconds': time.time() - self._test_start_time if self._test_start_time else 0,
            'final_results': self.get_results()
        }
    
    def record_metrics(self, production_metric: float, canary_metric: float,
                       pnl_production: float = 0.0, pnl_canary: float = 0.0) -> Optional[str]:
        """
        Record metrics from both models.
        
        Args:
            production_metric: Metric from production model (e.g., Sharpe ratio)
            canary_metric: Metric from canary model
            pnl_production: PnL from production trades
            pnl_canary: PnL from canary trades
            
        Returns:
            SPRT decision if concluded
        """
        if not self._test_active:
            return None
        
        # Store metrics
        self._production_metrics.append({
            'value': production_metric,
            'timestamp': time.time()
        })
        self._canary_metrics.append({
            'value': canary_metric,
            'timestamp': time.time()
        })
        
        # Update PnL
        self._pnl_production += pnl_production
        self._pnl_canary += pnl_canary
        
        # Check PnL divergence
        if self._check_pnl_divergence():
            logger.warning("PnL divergence threshold exceeded!")
        
        # Update SPRT
        decision = self.sprt.update(production_metric, canary_metric)
        
        if decision:
            logger.info(f"SPRT concluded: {decision}")
            self._test_active = False
        
        return decision
    
    def _check_pnl_divergence(self) -> bool:
        """Check if PnL divergence exceeds threshold."""
        if abs(self._pnl_production) < 1e-6:
            return False
        
        divergence = abs(self._pnl_canary - self._pnl_production) / abs(self._pnl_production)
        
        if divergence > self._max_pnl_divergence:
            self._divergence_alerts += 1
            return True
        
        return False
    
    def get_results(self) -> Dict[str, Any]:
        """Get comprehensive test results."""
        sprt_status = self.sprt.get_status()
        
        # Calculate additional statistics
        n_samples = len(self._production_metrics)
        
        if n_samples > 0:
            prod_values = [m['value'] for m in self._production_metrics]
            canary_values = [m['value'] for m in self._canary_metrics]
            
            mean_diff = np.mean(canary_values) - np.mean(prod_values)
            std_diff = np.std([c - p for p, c in zip(prod_values, canary_values)])
        else:
            mean_diff = 0.0
            std_diff = 0.0
        
        return {
            'sprt_status': sprt_status,
            'n_samples': n_samples,
            'mean_difference': mean_diff,
            'std_difference': std_diff,
            'pnl_production': self._pnl_production,
            'pnl_canary': self._pnl_canary,
            'pnl_divergence': abs(self._pnl_canary - self._pnl_production),
            'divergence_alerts': self._divergence_alerts,
            'test_active': self._test_active,
            'test_duration': time.time() - self._test_start_time if self._test_start_time else 0
        }
    
    def should_promote(self) -> bool:
        """Determine if canary should be promoted."""
        status = self.sprt.get_status()
        
        # Must be concluded with acceptance
        if status['status'] != 'concluded':
            return False
        
        if status['conclusion'] != 'accept':
            return False
        
        # Check PnL divergence
        if self._divergence_alerts > 0:
            logger.warning("Cannot promote: PnL divergence detected")
            return False
        
        return True
    
    def should_rollback(self) -> bool:
        """Determine if canary should be rolled back."""
        status = self.sprt.get_status()
        
        # Concluded with rejection
        if status['status'] == 'concluded' and status['conclusion'] == 'reject':
            return True
        
        # Excessive divergence
        if self._divergence_alerts >= 3:
            return True
        
        return False
    
    def get_recommendation(self) -> str:
        """Get action recommendation."""
        if not self._test_active and self.sprt._test_concluded:
            if self.should_promote():
                return 'PROMOTE'
            elif self.should_rollback():
                return 'ROLLBACK'
            else:
                return 'INCONCLUSIVE'
        
        if self._test_active:
            return 'TESTING'
        
        return 'NO_TEST'
    
    def reset(self) -> None:
        """Reset tester state."""
        self.sprt.reset()
        self._production_metrics.clear()
        self._canary_metrics.clear()
        self._pnl_production = 0.0
        self._pnl_canary = 0.0
        self._test_active = False
        self._test_start_time = None
        self._divergence_alerts = 0
        logger.info("ABTester reset")


# Singleton instance
_ab_tester: Optional[ABTester] = None


def get_ab_tester(config: Optional[Dict[str, Any]] = None) -> ABTester:
    """Get or create singleton ABTester instance."""
    global _ab_tester
    if _ab_tester is None:
        _ab_tester = ABTester(config)
    return _ab_tester


def reset_ab_tester() -> None:
    """Reset singleton instance."""
    global _ab_tester
    if _ab_tester is not None:
        _ab_tester.reset()
    _ab_tester = None


__all__ = [
    'SequentialProbabilityRatioTest',
    'ABTester',
    'get_ab_tester',
    'reset_ab_tester'
]
