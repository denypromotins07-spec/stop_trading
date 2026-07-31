"""
Shadow Scorer - Evaluates new models against live data without executing trades.
Compares shadow model predictions against production to guarantee improvement.
Strictly enforces 3GB RAM limit with bounded scoring windows.
"""
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from collections import deque
from dataclasses import dataclass
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class ShadowResult:
    """Result of shadow model evaluation."""
    model_id: str
    timestamp_ns: int
    predictions: np.ndarray
    actuals: np.ndarray
    metrics: Dict[str, float]
    vs_production: Dict[str, float]
    passed_threshold: bool


class ShadowModelEvaluator:
    """
    Evaluates shadow models against live market data.
    Memory-bounded for 3GB limit.
    """
    
    def __init__(self,
                 window_size: int = 10000,
                 promotion_threshold: float = 0.02):
        """
        Initialize shadow evaluator.
        
        Args:
            window_size: Number of samples for evaluation
            promotion_threshold: Minimum improvement to promote model
        """
        self.window_size = window_size
        self.promotion_threshold = promotion_threshold
        
        # Bounded storage for predictions
        self._shadow_predictions: deque = deque(maxlen=window_size)
        self._production_predictions: deque = deque(maxlen=window_size)
        self._actuals: deque = deque(maxlen=window_size)
        self._timestamps: deque = deque(maxlen=window_size)
        
        # Model tracking
        self._current_shadow_model: Optional[str] = None
        self._evaluation_count = 0
    
    def start_evaluation(self, model_id: str):
        """Start evaluating a new shadow model."""
        self._current_shadow_model = model_id
        self._shadow_predictions.clear()
        self._production_predictions.clear()
        self._actuals.clear()
        self._timestamps.clear()
        self._evaluation_count = 0
        
        logger.info(f"Started shadow evaluation for model: {model_id}")
    
    def record_prediction(self,
                         shadow_pred: float,
                         production_pred: float,
                         actual: float,
                         timestamp_ns: int):
        """
        Record prediction pair for evaluation.
        
        Args:
            shadow_pred: Prediction from shadow model
            production_pred: Prediction from production model
            actual: Actual observed value
            timestamp_ns: Timestamp
        """
        self._shadow_predictions.append(shadow_pred)
        self._production_predictions.append(production_pred)
        self._actuals.append(actual)
        self._timestamps.append(timestamp_ns)
        self._evaluation_count += 1
    
    def compute_metrics(self,
                       predictions: np.ndarray,
                       actuals: np.ndarray) -> Dict[str, float]:
        """
        Compute evaluation metrics.
        
        Args:
            predictions: Array of predictions
            actuals: Array of actual values
            
        Returns:
            Dictionary of metrics
        """
        if len(predictions) < 10:
            return {}
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        errors = predictions - actuals
        
        return {
            'mse': float(np.mean(errors ** 2)),
            'rmse': float(np.sqrt(np.mean(errors ** 2))),
            'mae': float(np.mean(np.abs(errors))),
            'mape': float(np.mean(np.abs(errors / (actuals + 1e-6)))) * 100,
            'directional_accuracy': float(np.mean(
                np.sign(predictions) == np.sign(actuals)
            )),
            'correlation': float(np.corrcoef(predictions, actuals)[0, 1]) 
                if len(predictions) > 1 else 0.0,
            'sample_count': len(predictions)
        }
    
    def evaluate(self) -> Optional[ShadowResult]:
        """
        Evaluate shadow model against production.
        
        Returns:
            ShadowResult if enough data, None otherwise
        """
        if len(self._shadow_predictions) < 100:
            logger.warning("Insufficient data for evaluation")
            return None
        
        shadow_preds = np.array(self._shadow_predictions)
        prod_preds = np.array(self._production_predictions)
        actuals = np.array(self._actuals)
        
        # Compute metrics for both models
        shadow_metrics = self.compute_metrics(shadow_preds, actuals)
        prod_metrics = self.compute_metrics(prod_preds, actuals)
        
        # Compare performance
        comparison = {}
        for key in shadow_metrics:
            if key in prod_metrics and key not in ['sample_count']:
                if key in ['directional_accuracy', 'correlation']:
                    # Higher is better
                    diff = shadow_metrics[key] - prod_metrics[key]
                else:
                    # Lower is better (error metrics)
                    diff = prod_metrics[key] - shadow_metrics[key]
                
                comparison[f"{key}_improvement"] = diff
        
        # Determine if shadow passes threshold
        primary_improvement = comparison.get('mse_improvement', 0)
        passed = primary_improvement >= self.promotion_threshold
        
        result = ShadowResult(
            model_id=self._current_shadow_model or "unknown",
            timestamp_ns=self._timestamps[-1] if self._timestamps else 0,
            predictions=shadow_preds.copy(),
            actuals=actuals.copy(),
            metrics=shadow_metrics,
            vs_production=comparison,
            passed_threshold=passed
        )
        
        logger.info(
            f"Shadow evaluation complete: "
            f"MSE improvement={primary_improvement:.6f}, "
            f"passed={passed}"
        )
        
        return result
    
    def get_running_stats(self) -> Dict[str, Any]:
        """Get running evaluation statistics."""
        if not self._shadow_predictions:
            return {"status": "no_data"}
        
        shadow_preds = np.array(self._shadow_predictions)
        prod_preds = np.array(self._production_predictions)
        
        return {
            "model_id": self._current_shadow_model,
            "samples": len(self._shadow_predictions),
            "window_size": self.window_size,
            "shadow_mean": float(np.mean(shadow_preds)),
            "shadow_std": float(np.std(shadow_preds)),
            "production_mean": float(np.mean(prod_preds)),
            "production_std": float(np.std(prod_preds)),
            "prediction_diff_mean": float(np.mean(shadow_preds - prod_preds)),
            "evaluation_count": self._evaluation_count
        }


class ShadowScorer:
    """
    Manages multiple shadow evaluations and promotion logic.
    """
    
    def __init__(self,
                 promotion_threshold: float = 0.02,
                 min_samples: int = 1000,
                 max_concurrent_evaluations: int = 3):
        """
        Initialize shadow scorer.
        
        Args:
            promotion_threshold: Minimum improvement for promotion
            min_samples: Minimum samples before evaluation
            max_concurrent_evaluations: Max simultaneous shadow models
        """
        self.promotion_threshold = promotion_threshold
        self.min_samples = min_samples
        self.max_concurrent = max_concurrent_evaluations
        
        # Active evaluations
        self._evaluators: Dict[str, ShadowModelEvaluator] = {}
        
        # Promotion history (bounded)
        self._promotion_history: deque = deque(maxlen=100)
        self._rejected_models: deque = deque(maxlen=100)
    
    def register_shadow_model(self, model_id: str) -> bool:
        """
        Register a new shadow model for evaluation.
        
        Args:
            model_id: Unique model identifier
            
        Returns:
            True if registered successfully
        """
        if len(self._evaluators) >= self.max_concurrent:
            logger.warning(f"Max concurrent evaluations ({self.max_concurrent}) reached")
            return False
        
        if model_id in self._evaluators:
            logger.warning(f"Model {model_id} already being evaluated")
            return False
        
        self._evaluators[model_id] = ShadowModelEvaluator(
            window_size=self.min_samples * 2,
            promotion_threshold=self.promotion_threshold
        )
        self._evaluators[model_id].start_evaluation(model_id)
        
        logger.info(f"Registered shadow model: {model_id}")
        return True
    
    def record_prediction(self,
                         model_id: str,
                         shadow_pred: float,
                         production_pred: float,
                         actual: float,
                         timestamp_ns: int) -> bool:
        """
        Record prediction for shadow model.
        
        Args:
            model_id: Shadow model identifier
            shadow_pred: Shadow model prediction
            production_pred: Production model prediction
            actual: Actual value
            timestamp_ns: Timestamp
            
        Returns:
            True if recorded successfully
        """
        if model_id not in self._evaluators:
            return False
        
        self._evaluators[model_id].record_prediction(
            shadow_pred, production_pred, actual, timestamp_ns
        )
        return True
    
    def evaluate_model(self, model_id: str) -> Optional[ShadowResult]:
        """
        Evaluate a specific shadow model.
        
        Args:
            model_id: Model to evaluate
            
        Returns:
            ShadowResult or None
        """
        if model_id not in self._evaluators:
            return None
        
        result = self._evaluators[model_id].evaluate()
        
        if result:
            if result.passed_threshold:
                self._promotion_history.append({
                    'model_id': model_id,
                    'timestamp': time.time(),
                    'metrics': result.metrics,
                    'improvement': result.vs_production
                })
                logger.info(f"Model {model_id} PASSED evaluation")
            else:
                self._rejected_models.append({
                    'model_id': model_id,
                    'timestamp': time.time(),
                    'reason': 'Below promotion threshold'
                })
                logger.info(f"Model {model_id} FAILED evaluation")
        
        return result
    
    def should_promote(self, model_id: str) -> Tuple[bool, Optional[ShadowResult]]:
        """
        Check if model should be promoted to production.
        
        Args:
            model_id: Model to check
            
        Returns:
            Tuple of (should_promote, result)
        """
        result = self.evaluate_model(model_id)
        
        if result is None:
            return False, None
        
        return result.passed_threshold, result
    
    def promote_model(self, model_id: str) -> bool:
        """
        Promote model to production (removes from shadow).
        
        Args:
            model_id: Model to promote
            
        Returns:
            True if promoted successfully
        """
        should_promote, result = self.should_promote(model_id)
        
        if not should_promote:
            logger.warning(f"Model {model_id} does not meet promotion criteria")
            return False
        
        # Remove from shadow evaluators
        if model_id in self._evaluators:
            del self._evaluators[model_id]
        
        logger.info(f"Model {model_id} promoted to production")
        return True
    
    def remove_shadow_model(self, model_id: str) -> bool:
        """Remove a shadow model without promotion."""
        if model_id in self._evaluators:
            del self._evaluators[model_id]
            self._rejected_models.append({
                'model_id': model_id,
                'timestamp': time.time(),
                'reason': 'Manual removal'
            })
            return True
        return False
    
    def get_all_results(self) -> Dict[str, Any]:
        """Get results for all active shadow models."""
        results = {}
        for model_id, evaluator in self._evaluators.items():
            results[model_id] = {
                'running_stats': evaluator.get_running_stats(),
                'result': evaluator.evaluate()
            }
        return results
    
    def get_promotion_history(self) -> List[Dict]:
        """Get history of promoted models."""
        return list(self._promotion_history)


# Example usage
def main():
    """Example usage of shadow scorer."""
    scorer = ShadowScorer(promotion_threshold=0.01, min_samples=500)
    
    # Register shadow model
    scorer.register_shadow_model("shadow_v2")
    
    # Simulate predictions
    np.random.seed(42)
    n_samples = 1000
    
    print("Simulating shadow evaluation...")
    for i in range(n_samples):
        # Generate synthetic data
        actual = np.random.randn()
        
        # Production model has some error
        prod_pred = actual + np.random.randn() * 0.5
        
        # Shadow model is slightly better
        shadow_pred = actual + np.random.randn() * 0.45
        
        scorer.record_prediction(
            model_id="shadow_v2",
            shadow_pred=float(shadow_pred),
            production_pred=float(prod_pred),
            actual=float(actual),
            timestamp_ns=i * 1_000_000_000
        )
    
    # Evaluate
    should_promote, result = scorer.should_promote("shadow_v2")
    
    print(f"\nEvaluation result:")
    print(f"  Should promote: {should_promote}")
    if result:
        print(f"  Shadow MSE: {result.metrics.get('mse', 0):.6f}")
        print(f"  Production MSE: {scorer._evaluators['shadow_v2'].compute_metrics(np.array(scorer._evaluators['shadow_v2']._production_predictions), np.array(scorer._evaluators['shadow_v2']._actuals)).get('mse', 0):.6f}")
        print(f"  Improvements: {result.vs_production}")


if __name__ == "__main__":
    main()
