"""
Module root wrapping cvxpy solvers in Ray actors for parallel multi-asset optimization.
Distributes portfolio optimization across different regime scenarios using Ray cluster.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging
import time

logger = logging.getLogger(__name__)

try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False


@dataclass
class RegimeScenario:
    """Market regime scenario for optimization."""
    name: str
    expected_returns: np.ndarray
    covariance_matrix: np.ndarray
    probability: float = 1.0
    
    # Regime-specific constraints
    max_leverage: float = 1.0
    sector_limits: Optional[Dict[str, float]] = None


@dataclass
class MultiScenarioResult:
    """Results from multi-scenario optimization."""
    scenario_results: Dict[str, Any]
    blended_weights: np.ndarray
    robustness_score: float
    solve_time_ms: float


class PortfolioOptimizerActor:
    """Ray actor for parallel portfolio optimization."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._optimizer = None
    
    def _get_optimizer(self):
        if self._optimizer is None:
            from .cvxpy_optimizer import ConvexPortfolioOptimizer, OptimizationConfig
            opt_config = OptimizationConfig(**self.config) if self.config else OptimizationConfig()
            self._optimizer = ConvexPortfolioOptimizer(opt_config)
        return self._optimizer
    
    def optimize(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        current_weights: Optional[np.ndarray] = None,
        asset_ids: Optional[List[str]] = None,
        optimization_type: str = "mean_variance"
    ) -> Dict[str, Any]:
        """Run optimization and return result."""
        optimizer = self._get_optimizer()
        
        if optimization_type == "mean_variance":
            result = optimizer.optimize_mean_variance(
                expected_returns, covariance_matrix,
                current_weights, asset_ids
            )
        elif optimization_type == "risk_parity":
            result = optimizer.optimize_risk_parity(
                covariance_matrix, current_weights, asset_ids
            )
        elif optimization_type == "min_volatility":
            result = optimizer.optimize_minimum_volatility(
                covariance_matrix, expected_returns,
                current_weights, asset_ids
            )
        else:
            raise ValueError(f"Unknown optimization type: {optimization_type}")
        
        return result.to_dict()
    
    def shutdown(self):
        """Cleanup actor resources."""
        self._optimizer = None


class ConvexOptimizationModule:
    """
    Main module for distributed convex optimization.
    Manages Ray actors for parallel scenario optimization.
    """
    
    def __init__(
        self,
        num_actors: int = 4,
        actor_config: Optional[Dict] = None
    ):
        self.num_actors = num_actors
        self.actor_config = actor_config or {}
        
        self._actors: List[Any] = []
        self._initialized = False
        
        logger.info("ConvexOptimizationModule initialized")
    
    def initialize_ray(self):
        """Initialize Ray cluster if not already running."""
        if not RAY_AVAILABLE:
            raise RuntimeError("Ray is not available")
        
        if not ray.is_initialized():
            ray.init(
                num_cpus=self.num_actors,
                include_dashboard=False,
                log_to_driver=False
            )
        
        # Create optimizer actors
        ActorClass = ray.remote(PortfolioOptimizerActor)
        self._actors = [ActorClass.remote(self.actor_config) for _ in range(self.num_actors)]
        self._initialized = True
        
        logger.info(f"Created {self.num_actors} optimizer actors")
    
    def optimize_scenarios_parallel(
        self,
        scenarios: List[RegimeScenario],
        current_weights: Optional[np.ndarray] = None,
        asset_ids: Optional[List[str]] = None
    ) -> MultiScenarioResult:
        """
        Run optimizations for multiple scenarios in parallel.
        
        Args:
            scenarios: List of regime scenarios
            current_weights: Current portfolio weights
            asset_ids: Asset identifiers
            
        Returns:
            MultiScenarioResult with all results and blended weights
        """
        if not self._initialized:
            self.initialize_ray()
        
        start_time = time.perf_counter()
        
        # Dispatch optimizations to actors
        futures = []
        for i, scenario in enumerate(scenarios):
            actor_idx = i % len(self._actors)
            future = self._actors[actor_idx].optimize.remote(
                scenario.expected_returns,
                scenario.covariance_matrix,
                current_weights,
                asset_ids,
                "mean_variance"
            )
            futures.append((scenario.name, scenario.probability, future))
        
        # Collect results
        scenario_results = {}
        weighted_weights = np.zeros_like(current_weights) if current_weights is not None else None
        
        for name, prob, future in futures:
            try:
                result_dict = ray.get(future)
                scenario_results[name] = result_dict
                
                if weighted_weights is not None:
                    weights = np.array(result_dict['optimal_weights'])
                    weighted_weights += weights * prob
                    
            except Exception as e:
                logger.error(f"Scenario {name} optimization failed: {e}")
                scenario_results[name] = {'status': 'failed', 'error': str(e)}
        
        solve_time_ms = (time.perf_counter() - start_time) * 1000
        
        # Compute robustness score (variance across scenarios)
        all_weights = [
            np.array(r['optimal_weights']) 
            for r in scenario_results.values() 
            if r.get('status') == 'OPTIMAL'
        ]
        
        if len(all_weights) > 1:
            weight_std = np.std(all_weights, axis=0).mean()
            robustness_score = 1.0 / (1.0 + weight_std)
        else:
            robustness_score = 1.0
        
        return MultiScenarioResult(
            scenario_results=scenario_results,
            blended_weights=weighted_weights,
            robustness_score=robustness_score,
            solve_time_ms=solve_time_ms
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Get module status."""
        return {
            'initialized': self._initialized,
            'num_actors': len(self._actors),
            'ray_running': ray.is_initialized() if RAY_AVAILABLE else False
        }
    
    def shutdown(self):
        """Shutdown all actors and Ray."""
        for actor in self._actors:
            try:
                ray.get(actor.shutdown.remote())
            except:
                pass
        
        if ray.is_initialized():
            ray.shutdown()
        
        self._actors = []
        self._initialized = False


# Module singleton
_module_instance: Optional[ConvexOptimizationModule] = None


def get_module() -> ConvexOptimizationModule:
    """Get or create module singleton."""
    global _module_instance
    if _module_instance is None:
        _module_instance = ConvexOptimizationModule()
    return _module_instance


def initialize_module(num_actors: int = 4, **kwargs) -> ConvexOptimizationModule:
    """Initialize the module."""
    global _module_instance
    _module_instance = ConvexOptimizationModule(num_actors=num_actors, actor_config=kwargs)
    return _module_instance
