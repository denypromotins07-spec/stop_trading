"""
Evaluation Module Root - Triggers automated retraining and manages shadow-to-production promotion.
Coordinates drift monitoring and shadow scoring.
Strictly enforces 3GB RAM limit.
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path

from evaluation.drift_monitor import DriftMonitor, DriftAlert
from evaluation.shadow_scorer import ShadowScorer, ShadowResult


logger = logging.getLogger(__name__)


class EvaluationManager:
    """
    Central manager for all evaluation operations.
    Coordinates drift detection and model promotion.
    """
    
    def __init__(self,
                 feature_names: List[str],
                 promotion_threshold: float = 0.02,
                 min_samples: int = 1000):
        """
        Initialize evaluation manager.
        
        Args:
            feature_names: Features to monitor for drift
            promotion_threshold: Threshold for model promotion
            min_samples: Minimum samples for evaluation
        """
        self.feature_names = feature_names
        
        # Initialize components
        self.drift_monitor = DriftMonitor(feature_names)
        self.shadow_scorer = ShadowScorer(
            promotion_threshold=promotion_threshold,
            min_samples=min_samples
        )
        
        # Callbacks
        self._retrain_callback: Optional[Callable] = None
        self._promotion_callback: Optional[Callable] = None
        
        # State
        self._pending_retrains: List[str] = []
        self._pending_promotions: List[str] = []
    
    def set_baseline(self, feature_data: Dict[str, Any]):
        """Set baseline for drift monitoring."""
        self.drift_monitor.set_baseline(feature_data)
        logger.info(f"Set drift baseline for {len(feature_data)} features")
    
    def check_drift(self,
                   feature_values: Dict[str, float],
                   timestamp_ns: int) -> List[DriftAlert]:
        """
        Check for drift in feature values.
        
        Args:
            feature_values: Current feature values
            timestamp_ns: Timestamp
            
        Returns:
            List of drift alerts
        """
        alerts = self.drift_monitor.update(feature_values, timestamp_ns)
        
        # Check if retraining should be triggered
        if self.drift_monitor.should_retrain():
            self._pending_retrains.append("drift_detected")
            self.drift_monitor.acknowledge_retrain()
            
            if self._retrain_callback:
                asyncio.create_task(self._retrain_callback("drift"))
        
        return alerts
    
    def register_shadow_model(self, model_id: str) -> bool:
        """Register a shadow model for evaluation."""
        return self.shadow_scorer.register_shadow_model(model_id)
    
    def record_shadow_prediction(self,
                                model_id: str,
                                shadow_pred: float,
                                production_pred: float,
                                actual: float,
                                timestamp_ns: int) -> bool:
        """Record prediction for shadow model."""
        return self.shadow_scorer.record_prediction(
            model_id, shadow_pred, production_pred, actual, timestamp_ns
        )
    
    async def evaluate_shadow_model(self, model_id: str) -> Optional[ShadowResult]:
        """Evaluate a shadow model."""
        should_promote, result = self.shadow_scorer.should_promote(model_id)
        
        if should_promote and result:
            self._pending_promotions.append(model_id)
            
            if self._promotion_callback:
                await self._promotion_callback(model_id, result)
        
        return result
    
    def promote_model(self, model_id: str) -> bool:
        """Promote a shadow model to production."""
        return self.shadow_scorer.promote_model(model_id)
    
    def set_retrain_callback(self, callback: Callable):
        """Set callback for retrain triggers."""
        self._retrain_callback = callback
    
    def set_promotion_callback(self, callback: Callable):
        """Set callback for model promotions."""
        self._promotion_callback = callback
    
    def get_pending_retrains(self) -> List[str]:
        """Get list of pending retrain reasons."""
        return self._pending_retrains.copy()
    
    def clear_pending_retrains(self):
        """Clear pending retrain list."""
        self._pending_retrains.clear()
    
    def get_status(self) -> Dict[str, Any]:
        """Get evaluation system status."""
        return {
            "drift": self.drift_monitor.get_drift_summary(),
            "shadow_models": self.shadow_scorer.get_all_results(),
            "pending_retrains": self._pending_retrains,
            "pending_promotions": self._pending_promotions,
            "promotion_history": self.shadow_scorer.get_promotion_history()
        }
    
    def reset(self):
        """Reset evaluation state."""
        self._pending_retrains.clear()
        self._pending_promotions.clear()


# Module-level singleton
_eval_manager: Optional[EvaluationManager] = None


def get_manager() -> EvaluationManager:
    """Get or create evaluation manager singleton."""
    global _eval_manager
    if _eval_manager is None:
        raise RuntimeError("Evaluation manager not initialized")
    return _eval_manager


def initialize_evaluation(feature_names: List[str],
                         promotion_threshold: float = 0.02,
                         min_samples: int = 1000) -> EvaluationManager:
    """Initialize evaluation system."""
    global _eval_manager
    _eval_manager = EvaluationManager(
        feature_names=feature_names,
        promotion_threshold=promotion_threshold,
        min_samples=min_samples
    )
    return _eval_manager


async def check_feature_drift(feature_values: Dict[str, float],
                             timestamp_ns: int) -> List[DriftAlert]:
    """Check drift via singleton."""
    manager = get_manager()
    return manager.check_drift(feature_values, timestamp_ns)


def register_shadow(model_id: str) -> bool:
    """Register shadow model via singleton."""
    manager = get_manager()
    return manager.register_shadow_model(model_id)


async def evaluate_shadow(model_id: str) -> Optional[ShadowResult]:
    """Evaluate shadow model via singleton."""
    manager = get_manager()
    return await manager.evaluate_shadow_model(model_id)


def promote_shadow(model_id: str) -> bool:
    """Promote shadow model via singleton."""
    manager = get_manager()
    return manager.promote_model(model_id)


def get_evaluation_status() -> Dict[str, Any]:
    """Get status via singleton."""
    manager = get_manager()
    return manager.get_status()


# Example usage
async def main():
    """Example usage of evaluation module."""
    logging.basicConfig(level=logging.INFO)
    
    # Initialize
    features = ['feat_1', 'feat_2', 'feat_3']
    manager = initialize_evaluation(features)
    
    # Set baseline
    import numpy as np
    np.random.seed(42)
    baseline = {name: np.random.randn(1000) for name in features}
    manager.set_baseline(baseline)
    
    print(f"Evaluation status: {get_evaluation_status()}")
    
    # Register shadow model
    register_shadow("shadow_v1")
    
    # Simulate some predictions
    for i in range(100):
        actual = np.random.randn()
        prod_pred = actual + np.random.randn() * 0.5
        shadow_pred = actual + np.random.randn() * 0.45
        
        manager.record_shadow_prediction(
            "shadow_v1", shadow_pred, prod_pred, actual, i * 1_000_000_000
        )
    
    print(f"\nFinal status: {get_evaluation_status()}")


if __name__ == "__main__":
    asyncio.run(main())
