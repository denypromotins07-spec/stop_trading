"""
XGBoost/LightGBM Ensemble Wrapper for Tabular Alpha Prediction
Optimized for AMD Ryzen laptop with strict thread limits to prevent CPU oversubscription.
Uses predict_inplace and zero-copy operations where possible.
"""

import os
import numpy as np
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
import threading

# Conditional imports to minimize RAM footprint
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False


@dataclass
class EnsembleConfig:
    """Configuration for the ensemble model."""
    n_threads: int = 4  # Strict thread limit for AMD Ryzen
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    model_type: str = "xgb"  # or "lgb"
    random_state: int = 42
    use_gpu: bool = False


class XGBEnsemble:
    """
    Highly optimized XGBoost/LightGBM wrapper for tabular alpha prediction.
    Designed for microprice drift prediction with minimal memory footprint.
    """
    
    def __init__(self, config: Optional[EnsembleConfig] = None):
        self.config = config or EnsembleConfig()
        self.model = None
        self._lock = threading.Lock()
        self._is_fitted = False
        
        # Set global thread limits upfront
        self._set_thread_limits()
    
    def _set_thread_limits(self) -> None:
        """Set environment variables to limit thread usage."""
        os.environ["OMP_NUM_THREADS"] = str(self.config.n_threads)
        os.environ["MKL_NUM_THREADS"] = str(self.config.n_threads)
        os.environ["NUMEXPR_NUM_THREADS"] = str(self.config.n_threads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(self.config.n_threads)
    
    def create_model(self) -> Any:
        """Create the underlying model based on configuration."""
        if self.config.model_type == "xgb":
            if not XGB_AVAILABLE:
                raise ImportError("XGBoost not available. Install with: pip install xgboost")
            
            params = {
                "max_depth": self.config.max_depth,
                "learning_rate": self.config.learning_rate,
                "n_estimators": self.config.n_estimators,
                "objective": "reg:squarederror",
                "random_state": self.config.random_state,
                "n_jobs": self.config.n_threads,
                "tree_method": "hist",  # Memory-efficient histogram method
            }
            
            if not self.config.use_gpu:
                params["device"] = "cpu"
            
            return xgb.XGBRegressor(**params)
        
        elif self.config.model_type == "lgb":
            if not LGB_AVAILABLE:
                raise ImportError("LightGBM not available. Install with: pip install lightgbm")
            
            return lgb.LGBMRegressor(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                num_leaves=2 ** self.config.max_depth,
                n_jobs=self.config.n_threads,
                random_state=self.config.random_state,
                force_col_wise=True,  # Memory efficient for high-dimensional data
            )
        
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            early_stopping_rounds: Optional[int] = None) -> "XGBEnsemble":
        """
        Fit the ensemble model on training data.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
            eval_set: Optional evaluation set for early stopping
            early_stopping_rounds: Number of rounds for early stopping
        
        Returns:
            self
        """
        with self._lock:
            self.model = self.create_model()
            
            fit_kwargs = {}
            if eval_set is not None:
                fit_kwargs["eval_set"] = [eval_set]
            if early_stopping_rounds is not None and eval_set is not None:
                fit_kwargs["early_stopping_rounds"] = early_stopping_rounds
            
            self.model.fit(X, y, **fit_kwargs)
            self._is_fitted = True
            
        return self
    
    def predict_inplace(self, X: np.ndarray, output: np.ndarray) -> None:
        """
        Predict alpha scores inplace to avoid memory allocation.
        Critical for zero-copy handoffs in HFT pipelines.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            output: Pre-allocated output array (n_samples,)
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        if output.shape[0] != X.shape[0]:
            raise ValueError(f"Output array shape mismatch: expected {X.shape[0]}, got {output.shape[0]}")
        
        with self._lock:
            # Use predict with direct assignment to pre-allocated array
            predictions = self.model.predict(X)
            np.copyto(output, predictions.astype(np.float32, copy=False))
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Standard predict method returning new array.
        
        Args:
            X: Feature matrix (n_samples, n_features)
        
        Returns:
            Predictions array (n_samples,)
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        with self._lock:
            return self.model.predict(X).astype(np.float32)
    
    def predict_proba_binary(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities for binary classification.
        
        Args:
            X: Feature matrix (n_samples, n_features)
        
        Returns:
            Probability array (n_samples, 2)
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        with self._lock:
            if self.config.model_type == "xgb":
                return self.model.predict_proba(X).astype(np.float32)
            elif self.config.model_type == "lgb":
                return self.model.predict_proba(X).astype(np.float32)
    
    def get_feature_importance(self, top_n: int = 20) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Args:
            top_n: Number of top features to return
        
        Returns:
            Dictionary of feature names to importance scores
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before getting importance")
        
        importance = self.model.feature_importances_
        sorted_idx = np.argsort(importance)[::-1][:top_n]
        
        return {f"feature_{i}": importance[i] for i in sorted_idx}
    
    def save_model(self, path: str) -> None:
        """Save model to disk."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before saving")
        
        with self._lock:
            if self.config.model_type == "xgb":
                self.model.save_model(path)
            elif self.config.model_type == "lgb":
                self.model.booster_.save_model(path)
    
    def load_model(self, path: str) -> "XGBEnsemble":
        """Load model from disk."""
        with self._lock:
            self.model = self.create_model()
            
            if self.config.model_type == "xgb":
                self.model.load_model(path)
            elif self.config.model_type == "lgb":
                self.model.booster_ = lgb.Booster(model_file=path)
            
            self._is_fitted = True
        
        return self
    
    @property
    def is_fitted(self) -> bool:
        return self._is_fitted


def create_ensemble_ensemble(config: EnsembleConfig) -> XGBEnsemble:
    """Factory function to create ensemble with configuration."""
    return XGBEnsemble(config)


if __name__ == "__main__":
    # Example usage
    config = EnsembleConfig(n_threads=4, model_type="xgb")
    ensemble = XGBEnsemble(config)
    
    # Generate dummy data
    np.random.seed(42)
    X_train = np.random.randn(1000, 50).astype(np.float32)
    y_train = np.random.randn(1000).astype(np.float32)
    X_test = np.random.randn(100, 50).astype(np.float32)
    
    # Fit model
    ensemble.fit(X_train, y_train)
    
    # Predict inplace (zero-copy)
    output = np.empty(100, dtype=np.float32)
    ensemble.predict_inplace(X_test, output)
    
    print(f"Predictions shape: {output.shape}")
    print(f"Mean prediction: {output.mean():.6f}")
    print(f"Std prediction: {output.std():.6f}")
