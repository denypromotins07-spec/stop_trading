"""
Module root managing distribution of backtest configurations to Ray cluster.
Enables massive parallel parameter sweeps for strategy optimization.
"""

from __future__ import annotations

import asyncio
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
import logging
import time
import uuid

logger = logging.getLogger(__name__)

try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False


@dataclass
class ParameterSweepConfig:
    """Configuration for parameter sweep."""
    base_config: Dict[str, Any]
    parameters_to_sweep: Dict[str, List[Any]]
    
    def generate_configs(self) -> List[Dict[str, Any]]:
        """Generate all combinations of parameters."""
        from itertools import product
        
        param_names = list(self.parameters_to_sweep.keys())
        param_values = list(self.parameters_to_sweep.values())
        
        configs = []
        for values in product(*param_values):
            config = self.base_config.copy()
            for name, value in zip(param_names, values):
                config[name] = value
            configs.append(config)
        
        return configs


@dataclass
class SweepResult:
    """Results from a parameter sweep."""
    sweep_id: str
    total_runs: int
    successful_runs: int
    failed_runs: int
    best_config: Dict[str, Any]
    best_metrics: Dict[str, float]
    all_results: List[Dict[str, Any]]
    duration_ms: float


class BacktestWorkerActor:
    """Ray actor for running individual backtests."""
    
    def __init__(self, worker_id: str):
        self.worker_id = worker_id
        self._node = None
    
    def run_backtest(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single backtest with given config."""
        try:
            from .nautilus_backtest_node import NautilusBacktestNode, BacktestConfig
            
            bt_config = BacktestConfig(
                start_time=config.get('start_time', '2023-01-01'),
                end_time=config.get('end_time', '2023-12-31'),
                instruments=config.get('instruments', []),
                data_catalog_path=config.get('data_catalog_path', './data'),
                initial_cash=config.get('initial_cash', 1_000_000),
                commission_rate=config.get('commission_rate', 0.0001),
            )
            
            node = NautilusBacktestNode(bt_config)
            result = asyncio.run(node.run([]))
            
            return {
                'config': config,
                'result': result.to_dict(),
                'worker_id': self.worker_id
            }
        except Exception as e:
            return {
                'config': config,
                'error': str(e),
                'worker_id': self.worker_id,
                'status': 'FAILED'
            }


class BacktestOrchestrationModule:
    """
    Main module for orchestrating distributed backtests.
    Manages Ray workers and collects results.
    """
    
    def __init__(self, num_workers: int = 8):
        self.num_workers = num_workers
        self._workers: List[Any] = []
        self._initialized = False
    
    def initialize_ray(self):
        """Initialize Ray cluster."""
        if not RAY_AVAILABLE:
            raise RuntimeError("Ray is not available")
        
        if not ray.is_initialized():
            ray.init(
                num_cpus=self.num_workers,
                include_dashboard=False,
                log_to_driver=False
            )
        
        # Create worker actors
        WorkerClass = ray.remote(BacktestWorkerActor)
        self._workers = [WorkerClass.remote(f"worker_{i}") for i in range(self.num_workers)]
        self._initialized = True
        
        logger.info(f"Created {self.num_workers} backtest workers")
    
    async def run_parameter_sweep(
        self,
        sweep_config: ParameterSweepConfig,
        metric_to_optimize: str = "sharpe_ratio",
        maximize: bool = True
    ) -> SweepResult:
        """
        Run parameter sweep across all configurations.
        
        Args:
            sweep_config: Configuration defining parameter ranges
            metric_to_optimize: Metric name to optimize
            maximize: Whether to maximize or minimize the metric
            
        Returns:
            SweepResult with all results and best configuration
        """
        if not self._initialized:
            self.initialize_ray()
        
        sweep_id = str(uuid.uuid4())[:8]
        start_time = time.perf_counter()
        
        # Generate all configs
        configs = sweep_config.generate_configs()
        logger.info(f"Running sweep with {len(configs)} configurations")
        
        # Dispatch to workers
        futures = []
        for i, config in enumerate(configs):
            worker_idx = i % len(self._workers)
            future = self._workers[worker_idx].run_backtest.remote(config)
            futures.append(future)
        
        # Collect results
        all_results = []
        successful = 0
        failed = 0
        best_metric = float('-inf') if maximize else float('inf')
        best_config = None
        best_metrics = {}
        
        for future in futures:
            try:
                result = await asyncio.wrap_future(ray.get(future).as_future() if hasattr(ray.get(future), 'as_future') else ray.get(future))
                all_results.append(result)
                
                if 'error' not in result:
                    successful += 1
                    metrics = result.get('result', {})
                    
                    metric_value = metrics.get(metric_to_optimize, 0)
                    if maximize and metric_value > best_metric:
                        best_metric = metric_value
                        best_config = result.get('config')
                        best_metrics = metrics
                    elif not maximize and metric_value < best_metric:
                        best_metric = metric_value
                        best_config = result.get('config')
                        best_metrics = metrics
                else:
                    failed += 1
                    
            except Exception as e:
                logger.error(f"Backtest failed: {e}")
                failed += 1
        
        duration_ms = (time.perf_counter() - start_time) * 1000
        
        return SweepResult(
            sweep_id=sweep_id,
            total_runs=len(configs),
            successful_runs=successful,
            failed_runs=failed,
            best_config=best_config or {},
            best_metrics=best_metrics,
            all_results=all_results,
            duration_ms=duration_ms
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Get module status."""
        return {
            'initialized': self._initialized,
            'num_workers': len(self._workers),
            'ray_running': ray.is_initialized() if RAY_AVAILABLE else False
        }
    
    def shutdown(self):
        """Shutdown workers and Ray."""
        if ray.is_initialized():
            ray.shutdown()
        self._workers = []
        self._initialized = False


# Module singleton
_module_instance: Optional[BacktestOrchestrationModule] = None


def get_module() -> BacktestOrchestrationModule:
    """Get or create module singleton."""
    global _module_instance
    if _module_instance is None:
        _module_instance = BacktestOrchestrationModule()
    return _module_instance


def initialize_module(num_workers: int = 8) -> BacktestOrchestrationModule:
    """Initialize the module."""
    global _module_instance
    _module_instance = BacktestOrchestrationModule(num_workers=num_workers)
    return _module_instance
