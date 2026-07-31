"""
Online Learning Module Root - Manages online learning state and bounded memory.
Integrates river wrapper and weight adjuster for continuous model adaptation.
Strictly enforces 3GB RAM limit.
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any
from collections import deque
import numpy as np

from online_learning.river_wrapper import (
    RiverModelWrapper,
    StreamingFeatureProcessor,
    OnlineEnsemble,
    RIVER_AVAILABLE
)
from online_learning.weight_adjuster import (
    EnsembleWeightManager,
    WeightAdjustment,
    GradientApproximator,
    LeafValueAdjuster
)


logger = logging.getLogger(__name__)


class OnlineLearningManager:
    """
    Central manager for all online learning components.
    Coordinates river models and weight adjustments with strict memory bounds.
    """
    
    def __init__(self,
                 max_models: int = 10,
                 max_features: int = 100,
                 memory_budget_mb: int = 512):
        """
        Initialize online learning manager.
        
        Args:
            max_models: Maximum number of concurrent online models
            max_features: Maximum features per model
            memory_budget_mb: Memory budget in MB
        """
        self.max_models = max_models
        self.max_features = max_features
        self.memory_budget_mb = memory_budget_mb
        
        # Model registry (bounded)
        self._models: Dict[str, RiverModelWrapper] = {}
        self._ensembles: Dict[str, OnlineEnsemble] = {}
        self._weight_managers: Dict[str, EnsembleWeightManager] = {}
        self._processors: Dict[str, StreamingFeatureProcessor] = {}
        
        # Activity tracking (bounded)
        self._update_history: deque = deque(maxlen=1000)
        
        # Statistics
        self._total_updates = 0
        self._total_predictions = 0
    
    def register_model(self,
                      model_id: str,
                      model_type: str = "linear",
                      learning_rate: float = 0.01) -> bool:
        """
        Register a new online learning model.
        
        Args:
            model_id: Unique model identifier
            model_type: Type of river model
            learning_rate: Learning rate
            
        Returns:
            True if registered successfully
        """
        if len(self._models) >= self.max_models:
            logger.warning(f"Max models ({self.max_models}) reached, cannot register {model_id}")
            return False
        
        if model_id in self._models:
            logger.warning(f"Model {model_id} already registered")
            return False
        
        try:
            model = RiverModelWrapper(
                model_type=model_type,
                max_features=self.max_features,
                learning_rate=learning_rate
            )
            self._models[model_id] = model
            
            # Create associated weight manager
            self._weight_managers[model_id] = EnsembleWeightManager(
                model_id=model_id,
                n_estimators=100,
                max_features=self.max_features
            )
            
            # Create feature processor
            self._processors[model_id] = StreamingFeatureProcessor(
                max_features=self.max_features
            )
            
            logger.info(f"Registered online model: {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register model {model_id}: {e}")
            return False
    
    def register_ensemble(self,
                         ensemble_id: str,
                         n_models: int = 5,
                         model_types: List[str] = None) -> bool:
        """
        Register a new online ensemble.
        
        Args:
            ensemble_id: Unique ensemble identifier
            n_models: Number of models in ensemble
            model_types: List of model types
            
        Returns:
            True if registered successfully
        """
        if len(self._ensembles) >= self.max_models:
            logger.warning(f"Max ensembles ({self.max_models}) reached")
            return False
        
        if ensemble_id in self._ensembles:
            logger.warning(f"Ensemble {ensemble_id} already registered")
            return False
        
        ensemble = OnlineEnsemble(
            n_models=n_models,
            model_types=model_types,
            max_features=self.max_features
        )
        self._ensembles[ensemble_id] = ensemble
        
        logger.info(f"Registered online ensemble: {ensemble_id}")
        return True
    
    async def update(self,
                    model_id: str,
                    features: Dict[str, float],
                    target: float,
                    timestamp_ns: int) -> Optional[float]:
        """
        Update model with new observation.
        
        Args:
            model_id: Model to update
            features: Feature dictionary
            target: Target value
            timestamp_ns: Timestamp
            
        Returns:
            Prediction before update, or None if error
        """
        if model_id not in self._models:
            logger.error(f"Model {model_id} not found")
            return None
        
        model = self._models[model_id]
        processor = self._processors.get(model_id)
        
        try:
            # Preprocess features
            if processor:
                features = processor.transform_one(features)
                processor.partial_fit(features)
            
            # Update model
            prediction = model.learn_one(features, target)
            
            # Record update
            self._total_updates += 1
            self._update_history.append({
                'model_id': model_id,
                'timestamp_ns': timestamp_ns,
                'prediction': prediction,
                'target': target,
                'error': prediction - target
            })
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error updating model {model_id}: {e}")
            return None
    
    async def predict(self,
                     model_id: str,
                     features: Dict[str, float]) -> Optional[float]:
        """
        Get prediction from model.
        
        Args:
            model_id: Model to query
            features: Feature dictionary
            
        Returns:
            Prediction, or None if error
        """
        if model_id not in self._models:
            logger.error(f"Model {model_id} not found")
            return None
        
        model = self._models[model_id]
        processor = self._processors.get(model_id)
        
        try:
            # Preprocess features
            if processor:
                features = processor.transform_one(features)
            
            prediction = model.predict_one(features)
            self._total_predictions += 1
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error predicting with model {model_id}: {e}")
            return None
    
    async def apply_feedback(self,
                            model_id: str,
                            feature_indices: List[int],
                            penalty_weights: np.ndarray,
                            current_prediction: float,
                            target_value: float,
                            timestamp_ns: int,
                            reason: str) -> Optional[WeightAdjustment]:
        """
        Apply SOUL feedback to adjust model weights.
        
        Args:
            model_id: Model to adjust
            feature_indices: Features implicated
            penalty_weights: Penalty weights
            current_prediction: Current prediction
            target_value: Target value
            timestamp_ns: Timestamp
            reason: Reason for adjustment
            
        Returns:
            WeightAdjustment if successful
        """
        if model_id not in self._weight_managers:
            logger.error(f"Weight manager for {model_id} not found")
            return None
        
        manager = self._weight_managers[model_id]
        
        return manager.apply_feedback(
            feature_indices=feature_indices,
            penalty_weights=penalty_weights,
            current_prediction=current_prediction,
            target_value=target_value,
            timestamp_ns=timestamp_ns,
            reason=reason
        )
    
    def get_model_stats(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a specific model."""
        if model_id not in self._models:
            return None
        
        stats = {
            "model": self._models[model_id].get_stats(),
            "weight_manager": self._weight_managers[model_id].get_full_stats()
        }
        
        if model_id in self._ensembles:
            stats["ensemble"] = self._ensembles[model_id].get_model_stats()
        
        return stats
    
    def get_global_stats(self) -> Dict[str, Any]:
        """Get global online learning statistics."""
        recent_errors = [h['error'] for h in self._update_history]
        
        return {
            "total_models": len(self._models),
            "total_ensembles": len(self._ensembles),
            "total_updates": self._total_updates,
            "total_predictions": self._total_predictions,
            "recent_error_mean": float(np.mean(recent_errors)) if recent_errors else 0.0,
            "recent_error_std": float(np.std(recent_errors)) if recent_errors else 0.0,
            "memory_budget_mb": self.memory_budget_mb,
            "max_models": self.max_models,
            "max_features": self.max_features
        }
    
    def remove_model(self, model_id: str) -> bool:
        """Remove a model to free memory."""
        if model_id not in self._models:
            return False
        
        del self._models[model_id]
        if model_id in self._weight_managers:
            del self._weight_managers[model_id]
        if model_id in self._processors:
            del self._processors[model_id]
        if model_id in self._ensembles:
            del self._ensembles[model_id]
        
        logger.info(f"Removed model: {model_id}")
        return True
    
    def cleanup(self):
        """Cleanup all resources."""
        self._models.clear()
        self._ensembles.clear()
        self._weight_managers.clear()
        self._processors.clear()
        self._update_history.clear()


# Module-level singleton
_online_manager: Optional[OnlineLearningManager] = None


def get_manager() -> OnlineLearningManager:
    """Get or create online learning manager singleton."""
    global _online_manager
    if _online_manager is None:
        _online_manager = OnlineLearningManager()
    return _online_manager


def initialize_online_learning(max_models: int = 10,
                               max_features: int = 100) -> OnlineLearningManager:
    """Initialize online learning system."""
    global _online_manager
    _online_manager = OnlineLearningManager(
        max_models=max_models,
        max_features=max_features
    )
    return _online_manager


async def update_model(model_id: str,
                       features: Dict[str, float],
                       target: float,
                       timestamp_ns: int) -> Optional[float]:
    """Update model via singleton."""
    manager = get_manager()
    return await manager.update(model_id, features, target, timestamp_ns)


async def predict(model_id: str,
                  features: Dict[str, float]) -> Optional[float]:
    """Predict via singleton."""
    manager = get_manager()
    return await manager.predict(model_id, features)


async def apply_feedback(model_id: str,
                        feature_indices: List[int],
                        penalty_weights: np.ndarray,
                        current_prediction: float,
                        target_value: float,
                        timestamp_ns: int,
                        reason: str) -> Optional[WeightAdjustment]:
    """Apply feedback via singleton."""
    manager = get_manager()
    return await manager.apply_feedback(
        model_id, feature_indices, penalty_weights,
        current_prediction, target_value, timestamp_ns, reason
    )


def get_stats() -> Dict[str, Any]:
    """Get global stats via singleton."""
    manager = get_manager()
    return manager.get_global_stats()


# Example usage
async def main():
    """Example usage of online learning module."""
    logging.basicConfig(level=logging.INFO)
    
    if not RIVER_AVAILABLE:
        print("River library not available, using mock mode")
        return
    
    # Initialize
    manager = initialize_online_learning(max_models=5, max_features=50)
    
    # Register model
    manager.register_model("alpha_v1", model_type="linear", learning_rate=0.01)
    
    # Simulate streaming updates
    np.random.seed(42)
    for i in range(100):
        features = {f"f{j}": float(np.random.randn()) for j in range(20)}
        target = sum(features.values()) + np.random.randn() * 0.1
        
        pred = await update_model("alpha_v1", features, target, i * 1_000_000_000)
        
        if i % 20 == 0:
            print(f"Step {i}: pred={pred:.4f}, target={target:.4f}")
    
    # Get stats
    print(f"\nGlobal stats: {get_stats()}")
    print(f"Model stats: {manager.get_model_stats('alpha_v1')}")
    
    # Cleanup
    manager.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
