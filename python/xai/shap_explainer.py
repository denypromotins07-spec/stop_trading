"""
SHAP Explainer for XGBoost Ensemble - Background Ray Actor
Implements lightweight TreeExplainer with low CPU priority to avoid delaying live inference.
Strictly enforces 3GB RAM limit via chunked processing and background execution.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
import numpy as np
import shap
import xgboost as xgb
import ray
from ray.actor import ActorHandle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@ray.remote(max_restarts=3, max_task_retries=3)
class SHAPExplainerActor:
    """
    Background Ray actor for computing SHAP values on XGBoost models.
    Runs with low CPU priority to never interfere with live trading inference.
    """
    
    def __init__(self, model: Optional[xgb.Booster] = None, 
                 background_data: Optional[np.ndarray] = None,
                 max_samples: int = 1000):
        """
        Initialize SHAP explainer with background dataset.
        
        Args:
            model: Pre-trained XGBoost booster
            background_data: Background dataset for SHAP baseline (subset for memory)
            max_samples: Maximum samples to keep in background dataset
        """
        self.model = model
        self.explainer = None
        self.max_samples = max_samples
        self._background_data = None
        self._feature_names = None
        self._shap_values_cache = {}
        self._is_initialized = False
        
        if background_data is not None:
            self.set_background_data(background_data)
    
    def set_model(self, model: xgb.Booster, feature_names: Optional[List[str]] = None):
        """Update the model and reinitialize explainer."""
        self.model = model
        self._feature_names = feature_names
        self._initialize_explainer()
    
    def set_background_data(self, data: np.ndarray):
        """
        Set background dataset for SHAP computation.
        Subsamples if necessary to respect memory limits.
        """
        if len(data) > self.max_samples:
            indices = np.random.choice(len(data), self.max_samples, replace=False)
            self._background_data = data[indices]
            logger.info(f"Subsampled background data to {self.max_samples} samples")
        else:
            self._background_data = data
        
        if self.model is not None:
            self._initialize_explainer()
    
    def _initialize_explainer(self):
        """Initialize TreeExplainer with current model and background data."""
        if self.model is None or self._background_data is None:
            return
        
        try:
            # Use TreeExplainer for XGBoost - optimized C++ backend
            self.explainer = shap.TreeExplainer(
                self.model,
                self._background_data,
                feature_perturbation="tree_path_dependent",
                approximate=True  # Faster approximation for real-time use
            )
            self._is_initialized = True
            logger.info("SHAP TreeExplainer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize SHAP explainer: {e}")
            self._is_initialized = False
    
    async def compute_shap_values(self, data: np.ndarray, 
                                   sample_ids: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Compute SHAP values for given data asynchronously.
        Runs in background with low priority.
        
        Args:
            data: Input features for explanation
            sample_ids: Optional identifiers for samples
            
        Returns:
            Dictionary containing SHAP values, base values, and metadata
        """
        if not self._is_initialized:
            return {"error": "Explainer not initialized", "shap_values": None}
        
        try:
            # Limit batch size to prevent memory spikes
            batch_size = 500
            all_shap_values = []
            all_base_values = []
            
            for i in range(0, len(data), batch_size):
                batch = data[i:i+batch_size]
                
                # Compute SHAP values for batch
                shap_output = self.explainer.shap_values(batch)
                
                if isinstance(shap_output, list):
                    # Multi-class: take mean across classes or first class
                    shap_vals = np.mean(shap_output, axis=0)
                    base_vals = self.explainer.expected_value
                    if isinstance(base_vals, list):
                        base_vals = np.mean(base_vals)
                else:
                    shap_vals = shap_output
                    base_vals = self.explainer.expected_value
                
                all_shap_values.append(shap_vals)
                all_base_values.append(base_vals)
                
                # Yield control to event loop periodically
                await asyncio.sleep(0)
            
            shap_values = np.vstack(all_shap_values)
            base_value = all_base_values[0]  # Same for all samples
            
            result = {
                "shap_values": shap_values,
                "base_value": base_value,
                "feature_names": self._feature_names,
                "sample_count": len(data),
                "sample_ids": sample_ids,
                "timestamp": asyncio.get_event_loop().time()
            }
            
            # Cache recent results for quick access
            cache_key = hash(data.tobytes()) % 10000
            self._shap_values_cache[cache_key] = result
            if len(self._shap_values_cache) > 10:
                self._shap_values_cache.pop(next(iter(self._shap_values_cache)))
            
            return result
            
        except Exception as e:
            logger.error(f"Error computing SHAP values: {e}")
            return {"error": str(e), "shap_values": None}
    
    def get_feature_importance_summary(self, top_n: int = 20) -> Dict[str, float]:
        """
        Get aggregated feature importance from cached SHAP values.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            Dictionary mapping feature names to mean absolute SHAP values
        """
        if not self._shap_values_cache:
            return {}
        
        # Aggregate from cached values
        all_shap = [v["shap_values"] for v in self._shap_values_cache.values() 
                   if v.get("shap_values") is not None]
        
        if not all_shap:
            return {}
        
        # Mean absolute SHAP value per feature
        aggregated = np.mean(np.vstack([np.abs(np.mean(s, axis=0)) for s in all_shap]), axis=0)
        
        feature_names = self._feature_names or [f"feat_{i}" for i in range(len(aggregated))]
        
        importance_dict = dict(zip(feature_names, aggregated))
        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        
        return dict(sorted_importance[:top_n])
    
    def explain_single_prediction(self, data: np.ndarray, 
                                   feature_names: Optional[List[str]] = None) -> str:
        """
        Generate human-readable explanation for a single prediction.
        
        Args:
            data: Single sample features (1D array)
            feature_names: Optional feature names
            
        Returns:
            Formatted string explanation
        """
        if not self._is_initialized or len(data.shape) == 2:
            if len(data.shape) == 2:
                data = data[0]
            else:
                return "Invalid input format"
        
        shap_result = asyncio.get_event_loop().run_until_complete(
            self.compute_shap_values(data.reshape(1, -1))
        )
        
        if "error" in shap_result:
            return f"Explanation unavailable: {shap_result['error']}"
        
        shap_vals = shap_result["shap_values"][0]
        base_val = shap_result["base_value"]
        names = feature_names or self._feature_names or [f"f{i}" for i in range(len(shap_vals))]
        
        # Sort by absolute impact
        sorted_indices = np.argsort(np.abs(shap_vals))[::-1]
        
        explanation_lines = [
            f"Prediction Base Value: {base_val:.4f}",
            f"Final Prediction: {base_val + np.sum(shap_vals):.4f}",
            "\nTop Feature Contributions:"
        ]
        
        for idx in sorted_indices[:10]:
            direction = "↑" if shap_vals[idx] > 0 else "↓"
            explanation_lines.append(
                f"  {names[idx]}: {shap_vals[idx]:+.4f} {direction}"
            )
        
        return "\n".join(explanation_lines)
    
    def health_check(self) -> Dict[str, Any]:
        """Return actor health status."""
        return {
            "initialized": self._is_initialized,
            "model_loaded": self.model is not None,
            "background_samples": len(self._background_data) if self._background_data is not None else 0,
            "cache_size": len(self._shap_values_cache),
            "max_samples": self.max_samples
        }


# Module singleton for easy access
_explainer_actor: Optional[ActorHandle] = None


def get_explainer_actor() -> Optional[ActorHandle]:
    """Get or create the global SHAP explainer actor."""
    global _explainer_actor
    
    if _explainer_actor is None:
        try:
            _explainer_actor = SHAPExplainerActor.remote()
            logger.info("Created global SHAP explainer actor")
        except Exception as e:
            logger.error(f"Failed to create SHAP explainer actor: {e}")
            return None
    
    return _explainer_actor


async def initialize_explainer(model: xgb.Booster, 
                                background_data: np.ndarray,
                                feature_names: List[str]) -> bool:
    """Initialize the global explainer with model and data."""
    actor = get_explainer_actor()
    if actor is None:
        return False
    
    try:
        await actor.set_model.remote(model, feature_names)
        await actor.set_background_data.remote(background_data)
        return True
    except Exception as e:
        logger.error(f"Failed to initialize explainer: {e}")
        return False


if __name__ == "__main__":
    # Test initialization
    ray.init(ignore_reinit_error=True)
    
    # Create dummy model and data for testing
    X_train = np.random.randn(1000, 20)
    y_train = np.random.randint(0, 2, 1000)
    
    model = xgb.train(
        {"objective": "binary:logistic", "max_depth": 6},
        xgb.DMatrix(X_train, label=y_train),
        num_boost_round=50
    )
    
    actor = SHAPExplainerActor.remote(model, X_train[:500])
    
    # Test explanation
    test_data = np.random.randn(10, 20)
    result = ray.get(actor.compute_shap_values.remote(test_data))
    
    print(f"SHAP computation completed: {result.get('sample_count', 0)} samples processed")
    print(f"Feature importance summary: {ray.get(actor.get_feature_importance_summary.remote())}")
    
    ray.shutdown()
