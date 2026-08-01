"""
Bayesian Hyperparameter Optimization using Ray Tune and Optuna.
Implements asynchronous successive halving for early pruning of bad trials.
Strictly limits concurrent trials to prevent exceeding 3GB Python RAM allocation.

Optimized for finding optimal strategy parameters during offline training windows.
"""

import numpy as np
from typing import Dict, Any, Optional, Callable, List
import threading
import logging
import time
from dataclasses import dataclass
from pathlib import Path

# Ray and Optuna imports (lazy loading to minimize memory footprint)
try:
    import ray
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.search.optuna import OptunaSearch
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for Bayesian optimization run."""
    # Search space bounds
    param_bounds: Dict[str, tuple]  # {param_name: (min, max)}
    
    # Resource constraints
    max_concurrent_trials: int = 4  # Strict limit for 3GB RAM
    max_total_trials: int = 50
    trial_timeout_seconds: float = 300.0
    
    # Early stopping
    grace_period_trials: int = 5
    reduction_factor: float = 2.0
    
    # Objective weights
    sharpe_weight: float = 1.0
    drawdown_penalty: float = 2.0
    turnover_penalty: float = 0.5
    
    # Output
    results_dir: str = "./optimization_results"
    checkpoint_freq: int = 10


@dataclass
class TrialResult:
    """Results from a single optimization trial."""
    trial_id: str
    parameters: Dict[str, Any]
    sharpe_ratio: float
    max_drawdown: float
    total_return: float
    turnover: float
    objective_score: float
    duration_seconds: float
    status: str  # 'completed', 'pruned', 'failed'


class BayesianOptimizer:
    """
    Ray Tune-driven Bayesian Optimization with Optuna sampler.
    Uses Asynchronous Successive Halving (ASHA) for early pruning.
    """
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self._lock = threading.Lock()
        self._ray_initialized = False
        self._current_analysis = None
        self._trial_history: List[TrialResult] = []
        
        # Objective function (set by user)
        self._objective_fn: Optional[Callable] = None
    
    def initialize_ray(self) -> bool:
        """Initialize Ray cluster with strict memory limits."""
        if not RAY_AVAILABLE:
            logger.warning("Ray not available, falling back to sequential optimization")
            return False
        
        with self._lock:
            if self._ray_initialized:
                return True
            
            try:
                # Initialize Ray with constrained resources
                ray.init(
                    num_cpus=min(self.config.max_concurrent_trials * 2, 8),
                    _system_memory_limit_gb=2.0,  # Strict limit within 3GB total
                    object_store_memory=512 * 1024 * 1024,  # 512MB object store
                    log_to_driver=False,
                    dashboard_host=None,  # Disable dashboard to save memory
                    include_dashboard=False
                )
                self._ray_initialized = True
                logger.info(f"Ray initialized with {self.config.max_concurrent_trials} max concurrent trials")
                return True
                
            except Exception as e:
                logger.error(f"Failed to initialize Ray: {e}")
                return False
    
    def set_objective_function(self, objective_fn: Callable[[Dict[str, Any]], Dict[str, float]]) -> None:
        """
        Set the objective function for optimization.
        
        Args:
            objective_fn: Function that takes parameters dict and returns metrics dict
                         Must return: {'sharpe': float, 'drawdown': float, 'return': float, 'turnover': float}
        """
        self._objective_fn = objective_fn
    
    def _build_search_space(self) -> Dict[str, Any]:
        """Build Ray Tune search space from config bounds."""
        search_space = {}
        
        for param_name, (min_val, max_val) in self.config.param_bounds.items():
            if isinstance(min_val, int) and isinstance(max_val, int):
                # Integer parameter
                search_space[param_name] = tune.randint(min_val, max_val + 1)
            elif isinstance(min_val, float) and isinstance(max_val, float):
                # Float parameter - use log scale if spanning orders of magnitude
                if max_val > 0 and min_val > 0 and max_val / min_val > 10:
                    search_space[param_name] = tune.loguniform(min_val, max_val)
                else:
                    search_space[param_name] = tune.uniform(min_val, max_val)
            else:
                # Mixed types - default to uniform
                search_space[param_name] = tune.uniform(float(min_val), float(max_val))
        
        return search_space
    
    def _create_scheduler(self) -> ASHAScheduler:
        """Create ASHA scheduler for early stopping."""
        return ASHAScheduler(
            metric='objective_score',
            mode='max',
            max_t=self.config.max_total_trials,
            grace_period=self.config.grace_period_trials,
            reduction_factor=self.config.reduction_factor,
            brackets=3
        )
    
    def _create_search_algorithm(self) -> Optional[OptunaSearch]:
        """Create Optuna-based search algorithm for Bayesian optimization."""
        if not RAY_AVAILABLE:
            return None
        
        return OptunaSearch(
            metric='objective_score',
            mode='max',
            n_initial_points=max(5, self.config.max_concurrent_trials),
            gamma=1.0,  # Exploration-exploitation balance
            eta=1.0  # Acquisition function parameter
        )
    
    def _wrap_objective(self, config: Dict[str, Any]) -> Dict[str, float]:
        """
        Wrap user objective function with penalty calculations.
        Reports intermediate results for ASHA pruning.
        """
        if self._objective_fn is None:
            raise ValueError("Objective function not set")
        
        start_time = time.time()
        
        try:
            # Run user's objective function
            metrics = self._objective_fn(config)
            
            # Extract base metrics
            sharpe = metrics.get('sharpe', 0.0)
            drawdown = abs(metrics.get('drawdown', 0.0))
            total_return = metrics.get('return', 0.0)
            turnover = metrics.get('turnover', 0.0)
            
            # Calculate penalized objective score
            objective_score = (
                self.config.sharpe_weight * sharpe
                - self.config.drawdown_penalty * drawdown
                - self.config.turnover_penalty * turnover
            )
            
            duration = time.time() - start_time
            
            # Store trial result
            with self._lock:
                trial_result = TrialResult(
                    trial_id=f"trial_{len(self._trial_history)}",
                    parameters=config.copy(),
                    sharpe_ratio=sharpe,
                    max_drawdown=drawdown,
                    total_return=total_return,
                    turnover=turnover,
                    objective_score=objective_score,
                    duration_seconds=duration,
                    status='completed'
                )
                self._trial_history.append(trial_result)
            
            # Report to Ray Tune for pruning
            tune.report(
                sharpe=sharpe,
                drawdown=drawdown,
                objective_score=objective_score,
                duration=duration
            )
            
            return {'objective_score': objective_score}
            
        except Exception as e:
            logger.warning(f"Trial failed: {e}")
            
            # Record failed trial
            with self._lock:
                trial_result = TrialResult(
                    trial_id=f"trial_{len(self._trial_history)}",
                    parameters=config.copy(),
                    sharpe_ratio=0.0,
                    max_drawdown=0.0,
                    total_return=0.0,
                    turnover=0.0,
                    objective_score=-np.inf,
                    duration_seconds=time.time() - start_time,
                    status='failed'
                )
                self._trial_history.append(trial_result)
            
            tune.report(objective_score=-np.inf)
            return {'objective_score': -np.inf}
    
    def run_optimization(
        self,
        objective_fn: Callable[[Dict[str, Any]], Dict[str, float]],
        name: str = "strategy_optimization"
    ) -> Optional[Any]:
        """
        Run Bayesian optimization loop.
        
        Args:
            objective_fn: User's objective function
            name: Name for this optimization run
            
        Returns:
            Best trial results or None if optimization failed
        """
        self.set_objective_function(objective_fn)
        
        # Initialize Ray if needed
        if RAY_AVAILABLE and not self.initialize_ray():
            logger.warning("Ray initialization failed, running sequential fallback")
        
        if not RAY_AVAILABLE or not self._ray_initialized:
            return self._run_sequential_optimization()
        
        try:
            search_space = self._build_search_space()
            scheduler = self._create_scheduler()
            search_alg = self._create_search_algorithm()
            
            # Configure resource usage per trial
            resources_per_trial = {
                "cpu": 1,
                "memory": 512  # MB per trial
            }
            
            # Run optimization
            analysis = tune.run(
                self._wrap_objective,
                config=search_space,
                scheduler=scheduler,
                search_alg=search_alg,
                num_samples=self.config.max_total_trials,
                max_concurrent=self.config.max_concurrent_trials,
                resources_per_trial=resources_per_trial,
                local_dir=self.config.results_dir,
                name=name,
                verbose=1,
                raise_on_failed_trial=False,
                trial_executor=None,  # Use default
                storage_path=self.config.results_dir,
                checkpoint_freq=self.config.checkpoint_freq,
                checkpoint_at_end=True
            )
            
            self._current_analysis = analysis
            
            # Get best trial
            best_trial = analysis.get_best_trial('objective_score', 'max', 'last')
            if best_trial is None:
                logger.warning("No successful trials completed")
                return None
            
            logger.info(f"Best trial: {best_trial.config}")
            logger.info(f"Best objective score: {best_trial.last_result.get('objective_score', 0):.4f}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return None
    
    def _run_sequential_optimization(self) -> Dict[str, Any]:
        """Fallback sequential optimization when Ray is unavailable."""
        logger.info("Running sequential Bayesian optimization...")
        
        if self._objective_fn is None:
            raise ValueError("Objective function not set")
        
        # Simple random search with Bayesian-like selection
        best_params = None
        best_score = -np.inf
        evaluated_configs = []
        
        for i in range(self.config.max_total_trials):
            # Sample configuration
            config = {}
            for param_name, (min_val, max_val) in self.config.param_bounds.items():
                if isinstance(min_val, int) and isinstance(max_val, int):
                    config[param_name] = np.random.randint(min_val, max_val + 1)
                else:
                    config[param_name] = np.random.uniform(min_val, max_val)
            
            # Evaluate
            try:
                metrics = self._objective_fn(config)
                sharpe = metrics.get('sharpe', 0.0)
                drawdown = abs(metrics.get('drawdown', 0.0))
                turnover = metrics.get('turnover', 0.0)
                
                score = (
                    self.config.sharpe_weight * sharpe
                    - self.config.drawdown_penalty * drawdown
                    - self.config.turnover_penalty * turnover
                )
                
                evaluated_configs.append((config, score, metrics))
                
                if score > best_score:
                    best_score = score
                    best_params = config.copy()
                    
                if (i + 1) % 10 == 0:
                    logger.info(f"Trial {i+1}: score={score:.4f}, best={best_score:.4f}")
                    
            except Exception as e:
                logger.warning(f"Trial {i+1} failed: {e}")
        
        return {
            'best_config': best_params,
            'best_score': best_score,
            'all_results': evaluated_configs
        }
    
    def get_best_parameters(self) -> Optional[Dict[str, Any]]:
        """Get the best parameters found so far."""
        if self._current_analysis is not None:
            best_trial = self._current_analysis.get_best_trial('objective_score', 'max', 'last')
            if best_trial:
                return best_trial.config
        elif self._trial_history:
            # From sequential optimization
            best_trial = max(self._trial_history, key=lambda t: t.objective_score)
            return best_trial.parameters
        return None
    
    def get_trial_history(self) -> List[TrialResult]:
        """Get complete history of all trials."""
        with self._lock:
            return self._trial_history.copy()
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the optimization run."""
        with self._lock:
            if not self._trial_history:
                return {'status': 'no_trials'}
            
            completed = [t for t in self._trial_history if t.status == 'completed']
            pruned = [t for t in self._trial_history if t.status == 'pruned']
            failed = [t for t in self._trial_history if t.status == 'failed']
            
            if not completed:
                return {'status': 'no_completed_trials'}
            
            best = max(completed, key=lambda t: t.objective_score)
            
            return {
                'status': 'completed',
                'total_trials': len(self._trial_history),
                'completed': len(completed),
                'pruned': len(pruned),
                'failed': len(failed),
                'best_sharpe': best.sharpe_ratio,
                'best_drawdown': best.max_drawdown,
                'best_return': best.total_return,
                'best_params': best.parameters,
                'avg_duration': np.mean([t.duration_seconds for t in completed])
            }
    
    def shutdown(self) -> None:
        """Shutdown Ray cluster and release resources."""
        with self._lock:
            if self._ray_initialized and RAY_AVAILABLE:
                try:
                    ray.shutdown()
                    logger.info("Ray cluster shut down")
                except Exception as e:
                    logger.warning(f"Error shutting down Ray: {e}")
                finally:
                    self._ray_initialized = False


# Global optimizer instance
_optimizer_instance: Optional[BayesianOptimizer] = None
_optimizer_lock = threading.Lock()


def get_optimizer(config: Optional[OptimizationConfig] = None) -> BayesianOptimizer:
    """Thread-safe singleton access to Bayesian optimizer."""
    global _optimizer_instance
    
    with _optimizer_lock:
        if _optimizer_instance is None:
            if config is None:
                config = OptimizationConfig(param_bounds={})
            _optimizer_instance = BayesianOptimizer(config)
        
        return _optimizer_instance


if __name__ == "__main__":
    # Demo usage with mock objective function
    np.random.seed(42)
    
    def mock_objective(params: Dict[str, Any]) -> Dict[str, float]:
        """Mock objective function simulating backtest results."""
        # Simulate correlation between params and performance
        base_sharpe = 1.5
        sharpe_noise = np.random.randn() * 0.3
        
        # Parameters affect performance
        if 'lookback' in params:
            sharpe_noise += 0.1 * np.sin(params['lookback'] / 10)
        if 'threshold' in params:
            sharpe_noise -= 0.05 * abs(params['threshold'] - 0.5)
        
        sharpe = base_sharpe + sharpe_noise
        drawdown = 0.1 + np.random.exponential(0.05)
        total_return = sharpe * np.sqrt(252) * 0.1
        turnover = 50 + np.random.exponential(20)
        
        return {
            'sharpe': sharpe,
            'drawdown': drawdown,
            'return': total_return,
            'turnover': turnover
        }
    
    # Configure optimization
    config = OptimizationConfig(
        param_bounds={
            'lookback': (5, 100),
            'threshold': (0.1, 0.9),
            'stop_loss': (0.01, 0.10),
            'take_profit': (0.02, 0.20)
        },
        max_concurrent_trials=2,
        max_total_trials=20,
        sharpe_weight=1.0,
        drawdown_penalty=3.0,
        turnover_penalty=0.3
    )
    
    # Run optimization
    optimizer = get_optimizer(config)
    
    if RAY_AVAILABLE:
        print("Running with Ray Tune...")
        analysis = optimizer.run_optimization(mock_objective, name="demo_opt")
        
        if analysis:
            best_params = optimizer.get_best_parameters()
            print(f"\nBest parameters: {best_params}")
    else:
        print("Running sequential fallback...")
        results = optimizer.run_optimization(mock_objective)
        print(f"\nBest parameters: {results['best_config']}")
        print(f"Best score: {results['best_score']:.4f}")
    
    # Show summary
    summary = optimizer.get_optimization_summary()
    print(f"\nOptimization Summary: {summary}")
    
    optimizer.shutdown()
