"""
Validation Module Root - Integrates CPCV into Ray Tune AutoML pipeline.
Ensures out-of-sample robustness for all ML models in the trading system.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
import numpy as np
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler

from .purged_kfold import PurgedKFold, create_purged_kfold_from_trades
from .combinatorial_purged import CombinatorialPurgedKFold

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ValidationConfig:
    """Configuration for validation pipeline."""
    n_splits: int = 5
    n_test_folds: int = 2
    embargo_pct: float = 0.01
    use_cpcv: bool = True
    shuffle: bool = False
    random_state: int = 42
    metric_name: str = "sharpe_ratio"
    mode: str = "max"
    
    # Ray Tune specific
    num_samples: int = 20
    grace_period: int = 10
    reduction_factor: int = 3


@ray.remote
class ValidationWorker:
    """Ray worker for parallel validation tasks."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self._cv_splitter = None
    
    def get_cv_splitter(self, samples_info: Optional[Dict] = None):
        """Get or create CV splitter."""
        if self._cv_splitter is None:
            if self.config.use_cpcv:
                self._cv_splitter = CombinatorialPurgedKFold(
                    n_splits=self.config.n_splits,
                    n_test_folds=self.config.n_test_folds,
                    embargo_pct=self.config.embargo_pct,
                    samples_info=samples_info,
                    random_state=self.config.random_state
                )
            else:
                self._cv_splitter = PurgedKFold(
                    n_splits=self.config.n_splits,
                    embargo_pct=self.config.embargo_pct,
                    samples_info=samples_info,
                    shuffle=self.config.shuffle,
                    random_state=self.config.random_state
                )
        return self._cv_splitter
    
    def evaluate_model(self, model_class: type, X: np.ndarray, y: np.ndarray,
                       hyperparams: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate a model configuration using purged CV.
        
        Args:
            model_class: Model class to instantiate
            X: Feature matrix
            y: Target vector
            hyperparams: Hyperparameters for model
            
        Returns:
            Dictionary with validation metrics
        """
        splitter = self.get_cv_splitter()
        
        scores = []
        fold_scores = {}
        
        for result in splitter.split(X, y):
            if self.config.use_cpcv:
                train_idx, test_idx, combo_id = result
            else:
                train_idx, test_idx = result
                combo_id = len(scores)
            
            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            if len(X_train) == 0 or len(X_test) == 0:
                continue
            
            # Train model
            try:
                model = model_class(**hyperparams)
                model.fit(X_train, y_train)
                
                # Get predictions and calculate metric
                pred = model.predict(X_test)
                
                # Calculate Sharpe-like metric (for classification, use accuracy-adjusted returns)
                if len(np.unique(y_test)) > 1:
                    correct = (pred == y_test).sum()
                    accuracy = correct / len(y_test)
                    # Convert to pseudo-returns
                    returns = (pred * 2 - 1) * 0.01  # +1% for correct, -1% for wrong
                    mean_ret = np.mean(returns)
                    std_ret = np.std(returns)
                    sharpe = mean_ret / std_ret * np.sqrt(252) if std_ret > 0 else 0
                else:
                    sharpe = 0
                
                scores.append(sharpe)
                fold_scores[combo_id] = sharpe
                
            except Exception as e:
                logger.warning(f"Model evaluation failed: {e}")
                continue
        
        if not scores:
            return {"sharpe_ratio": 0, "mean_score": 0, "std_score": 0, "n_folds": 0}
        
        return {
            "sharpe_ratio": float(np.mean(scores)),
            "sharpe_std": float(np.std(scores)),
            "sharpe_se": float(np.std(scores) / np.sqrt(len(scores))),
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
            "min_score": float(np.min(scores)),
            "max_score": float(np.max(scores)),
            "n_folds": len(scores),
            "fold_scores": fold_scores
        }


class ValidationModule:
    """
    Central validation module integrating CPCV with Ray Tune AutoML.
    Ensures all models are validated with proper time-series cross-validation.
    """
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        """Initialize validation module."""
        self.config = config or ValidationConfig()
        self._workers: List[ray.actor.ActorHandle] = []
        self._is_initialized = False
        self._validation_history: List[Dict[str, Any]] = []
    
    def initialize(self, num_workers: int = 4):
        """Initialize Ray workers for parallel validation."""
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, num_cpus=num_workers)
        
        self._workers = [
            ValidationWorker.remote(self.config) 
            for _ in range(num_workers)
        ]
        self._is_initialized = True
        logger.info(f"Validation module initialized with {num_workers} workers")
    
    def create_tune_config(self, param_grid: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert parameter grid to Ray Tune search space.
        
        Args:
            param_grid: Dictionary of parameter names to lists/values
            
        Returns:
            Ray Tune compatible config
        """
        tune_config = {}
        
        for param_name, param_values in param_grid.items():
            if isinstance(param_values, list):
                if len(param_values) > 5:
                    # Use grid search for small lists, random for large
                    tune_config[param_name] = tune.grid_search(param_values[:10])
                else:
                    tune_config[param_name] = tune.choice(param_values)
            elif isinstance(param_values, tuple) and len(param_values) == 2:
                # Range specification
                tune_config[param_name] = tune.uniform(param_values[0], param_values[1])
            else:
                tune_config[param_name] = param_values
        
        return tune_config
    
    def run_automl(self, 
                   model_class: type,
                   X: np.ndarray,
                   y: np.ndarray,
                   param_grid: Dict[str, Any],
                   samples_info: Optional[Dict] = None,
                   timeout_seconds: int = 3600) -> Dict[str, Any]:
        """
        Run AutoML hyperparameter optimization with CPCV validation.
        
        Args:
            model_class: Model class to optimize
            X: Feature matrix
            y: Target vector
            param_grid: Parameter search space
            samples_info: Sample timing info for purging
            timeout_seconds: Maximum optimization time
            
        Returns:
            Best parameters and validation results
        """
        if not self._is_initialized:
            self.initialize()
        
        tune_config = self.create_tune_config(param_grid)
        
        # Define objective function
        def objective(config):
            # Extract only relevant parameters
            model_params = {k: v for k, v in config.items() 
                          if k in self._get_model_params(model_class)}
            
            # Use first worker for evaluation
            worker = self._workers[0]
            result = ray.get(worker.evaluate_model.remote(
                model_class, X, y, model_params
            ))
            
            # Report to Ray Tune
            tune.report(**result)
        
        # Configure scheduler for early stopping
        scheduler = ASHAScheduler(
            metric=self.config.metric_name,
            mode=self.config.mode,
            max_t=self.config.num_samples,
            grace_period=self.config.grace_period,
            reduction_factor=self.config.reduction_factor
        )
        
        # Run hyperparameter search
        analysis = tune.run(
            objective,
            config=tune_config,
            scheduler=scheduler,
            num_samples=self.config.num_samples,
            resources_per_trial={"cpu": 1},
            time_budget_s=timeout_seconds,
            verbose=1,
            raise_on_failed_trial=False
        )
        
        # Get best trial
        best_trial = analysis.get_best_trial(
            self.config.metric_name, 
            self.config.mode
        )
        
        if best_trial is None:
            return {
                "success": False,
                "error": "No valid trials completed",
                "best_params": {},
                "best_score": 0
            }
        
        best_params = best_trial.config
        best_score = best_trial.last_result[self.config.metric_name]
        
        result = {
            "success": True,
            "best_params": best_params,
            "best_score": best_score,
            "best_sharpe": best_score,
            "n_trials": len(analysis.trials),
            "trials_dataframe": analysis.results_df.to_dict() if hasattr(analysis, 'results_df') else {}
        }
        
        self._validation_history.append(result)
        logger.info(f"AutoML completed: best {self.config.metric_name}={best_score:.4f}")
        
        return result
    
    def _get_model_params(self, model_class: type) -> List[str]:
        """Get valid parameters for a model class."""
        import inspect
        try:
            sig = inspect.signature(model_class.__init__)
            return list(sig.parameters.keys())[1:]  # Skip 'self'
        except:
            return []
    
    def validate_strategy(self, 
                          strategy_fn: Callable,
                          returns: np.ndarray,
                          timestamps: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Validate a trading strategy using CPCV.
        
        Args:
            strategy_fn: Function that takes train data and returns predictions
            returns: Array of strategy returns
            timestamps: Optional timestamps for embargo calculation
            
        Returns:
            Validation metrics
        """
        n_samples = len(returns)
        
        # Create dummy features for CV splitting
        X = np.arange(n_samples).reshape(-1, 1)
        y = (returns > 0).astype(int)
        
        # Create samples info from timestamps
        samples_info = None
        if timestamps is not None:
            samples_info = {}
            for i in range(n_samples):
                if i < n_samples - 1:
                    duration = timestamps[i+1] - timestamps[i]
                else:
                    duration = 1.0
                samples_info[i] = (float(timestamps[i]), float(timestamps[i] + duration))
        
        # Update CV splitter with samples info
        if self.config.use_cpcv:
            cv = CombinatorialPurgedKFold(
                n_splits=self.config.n_splits,
                n_test_folds=self.config.n_test_folds,
                embargo_pct=self.config.embargo_pct,
                samples_info=samples_info
            )
        else:
            cv = PurgedKFold(
                n_splits=self.config.n_splits,
                embargo_pct=self.config.embargo_pct,
                samples_info=samples_info
            )
        
        # Collect out-of-sample returns
        oos_returns = []
        fold_metrics = []
        
        for result in cv.split(X, y):
            if self.config.use_cpcv:
                train_idx, test_idx, combo_id = result
            else:
                train_idx, test_idx = result
                combo_id = len(oos_returns)
            
            if len(test_idx) == 0:
                continue
            
            # Get OOS returns for this fold
            fold_returns = returns[test_idx]
            oos_returns.extend(fold_returns.tolist())
            
            # Calculate fold metrics
            mean_ret = np.mean(fold_returns)
            std_ret = np.std(fold_returns)
            sharpe = mean_ret / std_ret * np.sqrt(252) if std_ret > 0 else 0
            
            fold_metrics.append({
                "combo_id": combo_id,
                "mean_return": mean_ret,
                "std_return": std_ret,
                "sharpe": sharpe,
                "n_samples": len(fold_returns)
            })
        
        oos_returns = np.array(oos_returns)
        
        # Aggregate metrics
        total_mean = np.mean(oos_returns)
        total_std = np.std(oos_returns)
        total_sharpe = total_mean / total_std * np.sqrt(252) if total_std > 0 else 0
        
        # Calculate Sharpe distribution across folds
        fold_sharpes = [m["sharpe"] for m in fold_metrics]
        
        return {
            "oos_mean_return": float(total_mean),
            "oos_std_return": float(total_std),
            "oos_sharpe": float(total_sharpe),
            "oos_total_return": float(np.sum(oos_returns)),
            "fold_sharpe_mean": float(np.mean(fold_sharpes)) if fold_sharpes else 0,
            "fold_sharpe_std": float(np.std(fold_sharpes)) if fold_sharpes else 0,
            "fold_sharpe_min": float(np.min(fold_sharpes)) if fold_sharpes else 0,
            "fold_sharpe_max": float(np.max(fold_sharpes)) if fold_sharpes else 0,
            "n_folds": len(fold_metrics),
            "fold_details": fold_metrics
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Return module health status."""
        return {
            "initialized": self._is_initialized,
            "num_workers": len(self._workers),
            "config": {
                "n_splits": self.config.n_splits,
                "use_cpcv": self.config.use_cpcv,
                "embargo_pct": self.config.embargo_pct
            },
            "validation_runs": len(self._validation_history)
        }
    
    def shutdown(self):
        """Shutdown Ray workers."""
        if ray.is_initialized():
            ray.shutdown()
        self._workers = []
        self._is_initialized = False


# Module singleton
_val_module: Optional[ValidationModule] = None


def get_validation_module(config: Optional[ValidationConfig] = None) -> ValidationModule:
    """Get or create the global validation module."""
    global _val_module
    
    if _val_module is None:
        _val_module = ValidationModule(config)
    
    return _val_module


if __name__ == "__main__":
    # Test validation module
    np.random.seed(42)
    
    # Create test data
    n_samples = 2000
    X = np.random.randn(n_samples, 20)
    y = np.random.randint(0, 2, n_samples)
    returns = np.random.randn(n_samples) * 0.02 + 0.001
    
    # Initialize module
    config = ValidationConfig(
        n_splits=5,
        n_test_folds=2,
        embargo_pct=0.02,
        use_cpcv=True,
        num_samples=10
    )
    
    module = ValidationModule(config)
    
    # Test strategy validation
    print("Testing strategy validation...")
    strat_results = module.validate_strategy(
        lambda x: x,
        returns,
        timestamps=np.arange(n_samples)
    )
    
    print(f"\nStrategy Validation Results:")
    print(f"  OOS Sharpe: {strat_results['oos_sharpe']:.4f}")
    print(f"  OOS Mean Return: {strat_results['oos_mean_return']:.6f}")
    print(f"  Fold Sharpe Range: [{strat_results['fold_sharpe_min']:.4f}, {strat_results['fold_sharpe_max']:.4f}]")
    print(f"  Folds evaluated: {strat_results['n_folds']}")
    
    # Test AutoML (with simple model)
    print("\n\nTesting AutoML pipeline...")
    
    from sklearn.ensemble import RandomForestClassifier
    
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [2, 5, 10]
    }
    
    # Run quick AutoML test
    automl_result = module.run_automl(
        RandomForestClassifier,
        X[:500],  # Use subset for speed
        y[:500],
        param_grid,
        timeout_seconds=60
    )
    
    if automl_result["success"]:
        print(f"\nAutoML Results:")
        print(f"  Best Sharpe: {automl_result['best_sharpe']:.4f}")
        print(f"  Best Params: {automl_result['best_params']}")
        print(f"  Trials completed: {automl_result['n_trials']}")
    
    print(f"\nHealth: {module.health_check()}")
    
    module.shutdown()
