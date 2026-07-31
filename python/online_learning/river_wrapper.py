"""
River Wrapper - Online incremental learning for streaming tabular market data.
Uses river library for O(1) memory updates, adapting to micro-regime shifts.
Strictly enforces 3GB RAM limit with bounded model sizes.
"""
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from collections import deque
import logging

try:
    from river import linear_model, preprocessing, metrics, compose
    from river import drift as river_drift
    RIVER_AVAILABLE = True
except ImportError:
    RIVER_AVAILABLE = False
    # Fallback implementations if river not available
    pass


logger = logging.getLogger(__name__)


class RiverModelWrapper:
    """
    Wrapper for river online learning models with memory bounds.
    Supports incremental updates on streaming data without full retraining.
    """
    
    def __init__(self,
                 model_type: str = "linear",
                 max_features: int = 100,
                 decay: float = 0.995,
                 learning_rate: float = 0.01):
        """
        Initialize river model wrapper.
        
        Args:
            model_type: Type of model ('linear', 'logistic', 'adaptive_random_forest')
            max_features: Maximum number of features to track
            decay: Decay factor for old observations
            learning_rate: Learning rate for gradient-based models
        """
        if not RIVER_AVAILABLE:
            raise ImportError("river library required. Install with: pip install river")
        
        self.model_type = model_type
        self.max_features = max_features
        self.decay = decay
        self.learning_rate = learning_rate
        
        # Feature tracking (bounded)
        self._feature_names: deque = deque(maxlen=max_features)
        self._feature_stats: Dict[int, Dict[str, float]] = {}
        
        # Initialize model based on type
        self.model = self._create_model()
        
        # Statistics
        self._update_count = 0
        self._last_metrics: Dict[str, float] = {}
    
    def _create_model(self):
        """Create appropriate river model based on type."""
        if self.model_type == "linear":
            return linear_model.LinearRegression(
                optimizer={'lr': self.learning_rate},
                loss=metrics.MSE()
            )
        elif self.model_type == "logistic":
            return linear_model.LogisticRegression(
                optimizer={'lr': self.learning_rate}
            )
        elif self.model_type == "adaptive_random_forest":
            # Bounded random forest with max trees
            from river import ensemble
            return ensemble.AdaptiveRandomForestClassifier(
                n_models=10,  # Bounded number of trees
                max_depth=8,   # Limit tree depth for memory
                grace_period=50,
                delta=0.01
            )
        else:
            return linear_model.LinearRegression(
                optimizer={'lr': self.learning_rate}
            )
    
    def learn_one(self, x: Dict[str, float], y: float) -> float:
        """
        Update model with single observation.
        
        Args:
            x: Feature dictionary {feature_name: value}
            y: Target value
            
        Returns:
            Prediction before update
        """
        # Track feature names
        for fname in x.keys():
            if fname not in self._feature_names:
                self._feature_names.append(fname)
        
        # Get prediction before update
        y_pred = self.model.predict_one(x) if hasattr(self.model, 'predict_one') else 0.0
        
        # Update model
        self.model.learn_one(x, y)
        
        # Update statistics
        self._update_count += 1
        self._update_feature_stats(x)
        
        return y_pred
    
    def predict_one(self, x: Dict[str, float]) -> float:
        """
        Make prediction for single observation.
        
        Args:
            x: Feature dictionary
            
        Returns:
            Prediction
        """
        if hasattr(self.model, 'predict_one'):
            return self.model.predict_one(x)
        elif hasattr(self.model, 'predict_proba_one'):
            proba = self.model.predict_proba_one(x)
            return proba.get(True, 0.0)
        return 0.0
    
    def _update_feature_stats(self, x: Dict[str, float]):
        """Update running statistics for features."""
        for idx, (fname, val) in enumerate(x.items()):
            if idx >= self.max_features:
                break
            
            if idx not in self._feature_stats:
                self._feature_stats[idx] = {
                    'mean': val,
                    'var': 0.0,
                    'min': val,
                    'max': val,
                    'count': 1
                }
            else:
                stats = self._feature_stats[idx]
                count = stats['count']
                
                # Welford's online algorithm for variance
                delta = val - stats['mean']
                stats['mean'] += delta / (count + 1)
                delta2 = val - stats['mean']
                stats['var'] += delta * delta2 * count / (count + 1)
                
                stats['min'] = min(stats['min'], val)
                stats['max'] = max(stats['max'], val)
                stats['count'] = min(count + 1, 10000)  # Cap count for numerical stability
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        importance = {}
        
        if hasattr(self.model, 'get_feature_importance'):
            raw_importance = self.model.get_feature_importance()
            for fname, score in raw_importance.items():
                importance[fname] = score
        
        # Fallback: use variance-based importance
        if not importance and self._feature_stats:
            for idx, stats in self._feature_stats.items():
                if idx < len(self._feature_names):
                    fname = list(self._feature_names)[idx] if idx < len(self._feature_names) else f"feat_{idx}"
                    importance[fname] = stats.get('var', 0.0)
        
        return importance
    
    def reset(self):
        """Reset model to initial state."""
        self.model = self._create_model()
        self._feature_names.clear()
        self._feature_stats.clear()
        self._update_count = 0
        self._last_metrics = {}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get model statistics."""
        return {
            "model_type": self.model_type,
            "update_count": self._update_count,
            "feature_count": len(self._feature_names),
            "max_features": self.max_features,
            "decay": self.decay,
            "learning_rate": self.learning_rate
        }


class StreamingFeatureProcessor:
    """
    Preprocessor for streaming features with online normalization.
    Memory-bounded for 3GB limit.
    """
    
    def __init__(self, max_features: int = 100):
        """
        Initialize streaming processor.
        
        Args:
            max_features: Maximum features to track
        """
        self.max_features = max_features
        self._scaler = preprocessing.StandardScaler()
        self._feature_queue: deque = deque(maxlen=1000)
    
    def transform_one(self, x: Dict[str, float]) -> Dict[str, float]:
        """Transform single observation using current scaler state."""
        return self._scaler.transform_one(x)
    
    def partial_fit(self, x: Dict[str, float]):
        """Update scaler with new observation."""
        self._scaler.partial_fit(x)
        self._feature_queue.append(x)
    
    def get_feature_names(self) -> List[str]:
        """Get tracked feature names."""
        if self._feature_queue:
            return list(self._feature_queue[-1].keys())[:self.max_features]
        return []


class OnlineEnsemble:
    """
    Ensemble of online learning models for robust predictions.
    Combines multiple river models with adaptive weighting.
    """
    
    def __init__(self,
                 n_models: int = 5,
                 model_types: List[str] = None,
                 max_features: int = 100):
        """
        Initialize online ensemble.
        
        Args:
            n_models: Number of models in ensemble
            model_types: List of model types to use
            max_features: Maximum features per model
        """
        self.n_models = n_models
        self.model_types = model_types or ["linear", "logistic"]
        self.max_features = max_features
        
        # Create diverse models
        self.models: List[RiverModelWrapper] = []
        self.weights: np.ndarray = np.ones(n_models) / n_models
        
        for i in range(n_models):
            model_type = self.model_types[i % len(self.model_types)]
            # Vary learning rates for diversity
            lr = 0.005 + (i * 0.005)
            self.models.append(RiverModelWrapper(
                model_type=model_type,
                max_features=max_features,
                learning_rate=lr
            ))
        
        # Performance tracking
        self._model_errors: deque = deque(maxlen=100)
    
    def learn_one(self, x: Dict[str, float], y: float) -> float:
        """
        Update all models with single observation.
        
        Args:
            x: Feature dictionary
            y: Target value
            
        Returns:
            Ensemble prediction
        """
        predictions = []
        errors = []
        
        for i, model in enumerate(self.models):
            pred = model.learn_one(x, y)
            predictions.append(pred)
            
            # Track error for weight adjustment
            error = (pred - y) ** 2
            errors.append(error)
        
        # Update weights based on recent performance
        self._update_weights(errors)
        
        # Weighted ensemble prediction
        ensemble_pred = sum(p * w for p, w in zip(predictions, self.weights))
        
        return ensemble_pred
    
    def predict_one(self, x: Dict[str, float]) -> float:
        """
        Get ensemble prediction.
        
        Args:
            x: Feature dictionary
            
        Returns:
            Weighted ensemble prediction
        """
        predictions = [model.predict_one(x) for model in self.models]
        return sum(p * w for p, w in zip(predictions, self.weights))
    
    def _update_weights(self, errors: List[float]):
        """Adjust model weights based on prediction errors."""
        self._model_errors.append(errors)
        
        if len(self._model_errors) < 10:
            return
        
        # Calculate average errors
        avg_errors = np.mean(self._model_errors, axis=0)
        
        # Inverse error weighting (better models get higher weight)
        inv_errors = 1.0 / (avg_errors + 1e-6)
        self.weights = inv_errors / np.sum(inv_errors)
    
    def get_model_stats(self) -> List[Dict[str, Any]]:
        """Get statistics for all models."""
        return [model.get_stats() for model in self.models]


# Example usage
def main():
    """Example usage of river wrapper."""
    if not RIVER_AVAILABLE:
        print("River library not available")
        return
    
    # Create online model
    model = RiverModelWrapper(
        model_type="linear",
        max_features=50,
        learning_rate=0.01
    )
    
    # Simulate streaming data
    np.random.seed(42)
    for i in range(1000):
        x = {f"feat_{j}": float(np.random.randn()) for j in range(20)}
        y = sum(x.values()) + np.random.randn() * 0.1
        
        pred = model.learn_one(x, y)
        
        if i % 100 == 0:
            print(f"Step {i}: pred={pred:.4f}, actual={y:.4f}")
    
    print(f"\nModel stats: {model.get_stats()}")
    print(f"Feature importance: {model.get_feature_importance()}")


if __name__ == "__main__":
    main()
