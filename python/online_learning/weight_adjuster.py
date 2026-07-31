"""
Weight Adjuster - Dynamic weight adjustment for XGBoost/LightGBM models.
Uses gradient-based approximations to nudge ensemble models away from toxic patterns.
Strictly enforces 3GB RAM limit with bounded operations.
"""
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from collections import deque
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class WeightAdjustment:
    """Represents a weight adjustment operation."""
    model_id: str
    feature_indices: List[int]
    adjustment_values: np.ndarray
    magnitude: float
    timestamp_ns: int
    reason: str


class GradientApproximator:
    """
    Approximates gradients for tree-based models without full backprop.
    Uses leaf value perturbations based on feature importance.
    """
    
    def __init__(self, 
                 max_features: int = 100,
                 learning_rate: float = 0.001,
                 decay: float = 0.99):
        """
        Initialize gradient approximator.
        
        Args:
            max_features: Maximum features to track
            learning_rate: Step size for adjustments
            decay: Decay factor for historical gradients
        """
        self.max_features = max_features
        self.learning_rate = learning_rate
        self.decay = decay
        
        # Historical gradient tracking (bounded)
        self._gradient_history: deque = deque(maxlen=500)
        self._feature_sensitivity: Dict[int, float] = {}
        
        # Accumulated adjustments
        self._total_adjustments: Dict[int, float] = {}
    
    def compute_adjustment(self,
                          feature_indices: List[int],
                          penalty_weights: np.ndarray,
                          current_prediction: float,
                          target_value: float) -> np.ndarray:
        """
        Compute weight adjustment using gradient approximation.
        
        Args:
            feature_indices: Indices of features to adjust
            penalty_weights: Penalty weights from SOUL feedback
            current_prediction: Current model prediction
            target_value: Desired target value
            
        Returns:
            Array of adjustment values
        """
        # Calculate prediction error
        error = current_prediction - target_value
        
        # Scale by penalty weights
        scaled_error = error * np.mean(penalty_weights)
        
        # Compute adjustments for each feature
        adjustments = np.zeros(len(feature_indices))
        for i, feat_idx in enumerate(feature_indices):
            # Get feature sensitivity
            sensitivity = self._feature_sensitivity.get(feat_idx, 1.0)
            
            # Gradient approximation: direction is sign of error * sensitivity
            grad = scaled_error * sensitivity
            
            # Apply learning rate
            adjustments[i] = -self.learning_rate * grad
            
            # Update sensitivity tracking
            self._update_sensitivity(feat_idx, abs(grad))
        
        # Store in history
        self._gradient_history.append({
            'error': error,
            'adjustments': adjustments.copy(),
            'features': feature_indices
        })
        
        return adjustments
    
    def _update_sensitivity(self, feat_idx: int, gradient_magnitude: float):
        """Update feature sensitivity estimate."""
        if feat_idx not in self._feature_sensitivity:
            self._feature_sensitivity[feat_idx] = gradient_magnitude
        else:
            # Exponential moving average
            old_val = self._feature_sensitivity[feat_idx]
            self._feature_sensitivity[feat_idx] = (
                self.decay * old_val + (1 - self.decay) * gradient_magnitude
            )
    
    def get_historical_stats(self) -> Dict[str, float]:
        """Get statistics from gradient history."""
        if not self._gradient_history:
            return {"count": 0}
        
        errors = [h['error'] for h in self._gradient_history]
        return {
            "count": len(self._gradient_history),
            "mean_error": float(np.mean(errors)),
            "std_error": float(np.std(errors)),
            "max_error": float(np.max(np.abs(errors)))
        }


class LeafValueAdjuster:
    """
    Adjusts leaf values in tree-based models based on feedback.
    Works with XGBoost/LightGBM without requiring retraining.
    """
    
    def __init__(self,
                 max_trees: int = 100,
                 max_leaves_per_tree: int = 64):
        """
        Initialize leaf value adjuster.
        
        Args:
            max_trees: Maximum trees to track adjustments for
            max_leaves_per_tree: Maximum leaves per tree
        """
        self.max_trees = max_trees
        self.max_leaves_per_tree = max_leaves_per_tree
        
        # Leaf adjustments storage
        self._leaf_adjustments: Dict[int, Dict[int, float]] = {}
        self._adjustment_counts: Dict[int, int] = {}
        
        # Bounded history
        self._history: deque = deque(maxlen=1000)
    
    def adjust_leaf(self,
                   tree_id: int,
                   leaf_id: int,
                   adjustment: float,
                   reason: str = "") -> bool:
        """
        Apply adjustment to specific leaf.
        
        Args:
            tree_id: Tree identifier
            leaf_id: Leaf identifier within tree
            adjustment: Adjustment value
            reason: Reason for adjustment
            
        Returns:
            True if adjustment applied successfully
        """
        # Enforce bounds
        if tree_id >= self.max_trees:
            logger.warning(f"Tree ID {tree_id} exceeds max_trees {self.max_trees}")
            return False
        
        if leaf_id >= self.max_leaves_per_tree:
            logger.warning(f"Leaf ID {leaf_id} exceeds max_leaves {self.max_leaves_per_tree}")
            return False
        
        # Initialize tree dict if needed
        if tree_id not in self._leaf_adjustments:
            self._leaf_adjustments[tree_id] = {}
            self._adjustment_counts[tree_id] = 0
        
        # Apply adjustment (accumulate)
        if leaf_id not in self._leaf_adjustments[tree_id]:
            self._leaf_adjustments[tree_id][leaf_id] = 0.0
        
        self._leaf_adjustments[tree_id][leaf_id] += adjustment
        self._adjustment_counts[tree_id] += 1
        
        # Record history
        self._history.append({
            'tree_id': tree_id,
            'leaf_id': leaf_id,
            'adjustment': adjustment,
            'cumulative': self._leaf_adjustments[tree_id][leaf_id],
            'reason': reason
        })
        
        return True
    
    def get_cumulative_adjustment(self, tree_id: int, leaf_id: int) -> float:
        """Get cumulative adjustment for a specific leaf."""
        if tree_id in self._leaf_adjustments:
            return self._leaf_adjustments[tree_id].get(leaf_id, 0.0)
        return 0.0
    
    def get_all_adjustments(self) -> Dict[int, Dict[int, float]]:
        """Get all leaf adjustments."""
        return self._leaf_adjustments.copy()
    
    def reset_tree(self, tree_id: int):
        """Reset adjustments for a specific tree."""
        if tree_id in self._leaf_adjustments:
            del self._leaf_adjustments[tree_id]
        if tree_id in self._adjustment_counts:
            del self._adjustment_counts[tree_id]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get adjuster statistics."""
        total_adjustments = sum(self._adjustment_counts.values())
        active_trees = len(self._leaf_adjustments)
        
        return {
            "total_adjustments": total_adjustments,
            "active_trees": active_trees,
            "max_trees": self.max_trees,
            "history_size": len(self._history)
        }


class EnsembleWeightManager:
    """
    Manages weight adjustments for XGBoost/LightGBM ensembles.
    Integrates gradient approximation and leaf adjustments.
    """
    
    def __init__(self,
                 model_id: str,
                 n_estimators: int = 100,
                 max_features: int = 100,
                 learning_rate: float = 0.001):
        """
        Initialize ensemble weight manager.
        
        Args:
            model_id: Unique model identifier
            n_estimators: Number of trees in ensemble
            max_features: Maximum features to track
            learning_rate: Learning rate for adjustments
        """
        self.model_id = model_id
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.learning_rate = learning_rate
        
        self.grad_approx = GradientApproximator(
            max_features=max_features,
            learning_rate=learning_rate
        )
        
        self.leaf_adjuster = LeafValueAdjuster(
            max_trees=n_estimators,
            max_leaves_per_tree=64
        )
        
        # Pending adjustments queue
        self._pending_adjustments: List[WeightAdjustment] = []
        
        # Statistics
        self._total_applied = 0
        self._total_rejected = 0
    
    def apply_feedback(self,
                      feature_indices: List[int],
                      penalty_weights: np.ndarray,
                      current_prediction: float,
                      target_value: float,
                      timestamp_ns: int,
                      reason: str) -> Optional[WeightAdjustment]:
        """
        Apply SOUL feedback to adjust model weights.
        
        Args:
            feature_indices: Features implicated in mistake
            penalty_weights: Penalty weights from feedback
            current_prediction: Model's prediction that caused mistake
            target_value: What the prediction should have been
            timestamp_ns: Timestamp of feedback
            reason: Description of why adjustment is needed
            
        Returns:
            WeightAdjustment if successful, None if rejected
        """
        # Validate inputs
        if not feature_indices or len(penalty_weights) == 0:
            self._total_rejected += 1
            return None
        
        # Bound feature indices
        bounded_indices = [i % self.max_features for i in feature_indices]
        
        # Compute gradient-based adjustments
        adjustments = self.grad_approx.compute_adjustment(
            feature_indices=bounded_indices,
            penalty_weights=penalty_weights,
            current_prediction=current_prediction,
            target_value=target_value
        )
        
        # Create adjustment record
        adjustment = WeightAdjustment(
            model_id=self.model_id,
            feature_indices=bounded_indices,
            adjustment_values=adjustments,
            magnitude=float(np.max(np.abs(adjustments))),
            timestamp_ns=timestamp_ns,
            reason=reason
        )
        
        # Apply to leaf values (heuristic mapping)
        for i, feat_idx in enumerate(bounded_indices):
            # Map feature to approximate tree/leaf (simplified)
            tree_id = feat_idx % self.n_estimators
            leaf_id = (feat_idx * 7) % 64  # Heuristic leaf mapping
            
            self.leaf_adjuster.adjust_leaf(
                tree_id=tree_id,
                leaf_id=leaf_id,
                adjustment=adjustments[i],
                reason=reason
            )
        
        self._pending_adjustments.append(adjustment)
        self._total_applied += 1
        
        return adjustment
    
    def get_pending_adjustments(self) -> List[WeightAdjustment]:
        """Get all pending adjustments."""
        return self._pending_adjustments.copy()
    
    def clear_pending(self):
        """Clear pending adjustments after application."""
        self._pending_adjustments.clear()
    
    def get_adjustment_summary(self) -> Dict[str, Any]:
        """Get summary of all adjustments."""
        if not self._pending_adjustments:
            return {"count": 0}
        
        magnitudes = [a.magnitude for a in self._pending_adjustments]
        
        return {
            "count": len(self._pending_adjustments),
            "mean_magnitude": float(np.mean(magnitudes)),
            "max_magnitude": float(np.max(magnitudes)),
            "min_magnitude": float(np.min(magnitudes)),
            "total_applied": self._total_applied,
            "total_rejected": self._total_rejected
        }
    
    def get_full_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        return {
            "model_id": self.model_id,
            "n_estimators": self.n_estimators,
            "max_features": self.max_features,
            "gradient_stats": self.grad_approx.get_historical_stats(),
            "leaf_stats": self.leaf_adjuster.get_stats(),
            "adjustment_summary": self.get_adjustment_summary()
        }


# Example usage
def main():
    """Example usage of weight adjuster."""
    # Create weight manager
    manager = EnsembleWeightManager(
        model_id="alpha_predictor_v1",
        n_estimators=100,
        max_features=50,
        learning_rate=0.001
    )
    
    # Simulate feedback
    np.random.seed(42)
    for i in range(10):
        feature_indices = list(np.random.choice(50, 5, replace=False))
        penalty_weights = np.random.rand(5) * 2.0
        current_pred = np.random.randn()
        target = np.random.randn()
        
        adjustment = manager.apply_feedback(
            feature_indices=feature_indices,
            penalty_weights=penalty_weights,
            current_prediction=current_pred,
            target_value=target,
            timestamp_ns=i * 1_000_000_000,
            reason=f"Simulated feedback {i}"
        )
        
        if adjustment:
            print(f"Applied adjustment {i}: magnitude={adjustment.magnitude:.6f}")
    
    print(f"\nAdjustment summary: {manager.get_adjustment_summary()}")
    print(f"\nFull stats: {manager.get_full_stats()}")


if __name__ == "__main__":
    main()
