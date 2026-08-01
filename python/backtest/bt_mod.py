"""
Backtest Module Root - Manages distributed Ray backtesting cluster.
Enforces strict memory limits on parameter grids and coordinates validation.
"""

from typing import Dict, List, Optional, Any, Tuple, Callable
import numpy as np
from dataclasses import dataclass, field
import threading
import time
import json

from .vectorbt_engine import VectorBTEngine, BacktestResult, get_bt_engine
from .walk_forward import WalkForwardAnalyzer, ModelStatus, WalkForwardResult, get_wf_analyzer


@dataclass
class BacktestJob:
    """A single backtest job in the queue."""
    
    job_id: str
    model_id: str
    signals: np.ndarray = None
    returns: np.ndarray = None
    
    # Parameters
    param_grid: Dict[str, List[float]] = field(default_factory=dict)
    transaction_costs: List[float] = field(default_factory=lambda: [0.0005])
    
    # Status
    status: str = "PENDING"  # PENDING, RUNNING, COMPLETED, FAILED
    results: Optional[Dict[str, Any]] = None
    error_message: str = ""
    
    # Timing
    created_at: float = field(default_factory=time.time)
    started_at: float = 0.0
    completed_at: float = 0.0
    
    # Memory tracking
    estimated_memory_mb: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "model_id": self.model_id,
            "status": self.status,
            "param_grid": self.param_grid,
            "estimated_memory_mb": self.estimated_memory_mb,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error_message": self.error_message
        }


@dataclass
class ClusterStats:
    """Ray cluster statistics."""
    
    total_jobs: int = 0
    pending_jobs: int = 0
    running_jobs: int = 0
    completed_jobs: int = 0
    failed_jobs: int = 0
    
    total_memory_used_mb: float = 0.0
    memory_limit_mb: float = 2500.0
    
    avg_job_duration_s: float = 0.0
    
    quarantined_models: int = 0
    active_models: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_jobs": self.total_jobs,
            "pending_jobs": self.pending_jobs,
            "running_jobs": self.running_jobs,
            "completed_jobs": self.completed_jobs,
            "failed_jobs": self.failed_jobs,
            "total_memory_used_mb": round(self.total_memory_used_mb, 2),
            "memory_limit_mb": self.memory_limit_mb,
            "memory_utilization": round(self.total_memory_used_mb / max(self.memory_limit_mb, 1), 3),
            "avg_job_duration_s": round(self.avg_job_duration_s, 3),
            "quarantined_models": self.quarantined_models,
            "active_models": self.active_models
        }


class BacktestClusterManager:
    """
    Manages distributed Ray backtesting cluster.
    Enforces memory limits and coordinates walk-forward validation.
    """
    
    def __init__(
        self,
        max_memory_mb: float = 2500,  # Stay under 3GB Python limit
        max_concurrent_jobs: int = 4,
        ray_address: Optional[str] = None
    ):
        self.max_memory_mb = max_memory_mb
        self.max_concurrent_jobs = max_concurrent_jobs
        self.ray_address = ray_address
        
        # Sub-modules
        self.bt_engine = VectorBTEngine(max_memory_mb=max_memory_mb)
        self.wf_analyzer = WalkForwardAnalyzer()
        
        # Job queue
        self._jobs: Dict[str, BacktestJob] = {}
        self._job_queue: List[str] = []  # Job IDs in priority order
        
        # Memory tracking
        self._current_memory_usage_mb = 0.0
        self._job_memory: Dict[str, float] = {}
        
        # Results storage
        self._results_store: Dict[str, Dict[str, Any]] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Performance tracking
        self._job_durations: List[float] = []
        self._total_jobs_processed = 0
    
    def submit_job(
        self,
        job_id: str,
        model_id: str,
        signals: np.ndarray,
        returns: np.ndarray,
        param_grid: Optional[Dict[str, List[float]]] = None
    ) -> str:
        """Submit a new backtest job."""
        with self._lock:
            # Estimate memory usage
            estimated_mb = (
                signals.nbytes + returns.nbytes + 
                len(param_grid or {}) * 100 * 8  # Parameter combinations
            ) / (1024 * 1024)
            
            # Check memory limit
            if self._current_memory_usage_mb + estimated_mb > self.max_memory_mb:
                raise MemoryError(
                    f"Job would exceed memory limit: "
                    f"{self._current_memory_usage_mb + estimated_mb:.1f}MB > {self.max_memory_mb:.1f}MB"
                )
            
            job = BacktestJob(
                job_id=job_id,
                model_id=model_id,
                signals=signals,
                returns=returns,
                param_grid=param_grid or {},
                estimated_memory_mb=estimated_mb
            )
            
            self._jobs[job_id] = job
            self._job_queue.append(job_id)
            self._job_memory[job_id] = estimated_mb
            
            # Register model for validation
            self.wf_analyzer.register_model(model_id)
            
            return job_id
    
    def run_job(self, job_id: str) -> Dict[str, Any]:
        """Execute a backtest job."""
        with self._lock:
            if job_id not in self._jobs:
                raise ValueError(f"Job {job_id} not found")
            
            job = self._jobs[job_id]
            
            if job.status != "PENDING":
                return job.results or {}
            
            # Update status
            job.status = "RUNNING"
            job.started_at = time.time()
            
            # Update memory tracking
            self._current_memory_usage_mb += job.estimated_memory_mb
        
        try:
            # Load data into engine
            self.bt_engine.load_prices(job.returns)
            
            if job.param_grid:
                # Run parameter sweep
                results = self._run_parameter_sweep(job)
            else:
                # Run single backtest
                signals = job.signals
                result = self.bt_engine.run_backtest(signals)
                results = {"backtest": result.to_dict()}
                
                # Run walk-forward validation
                wf_result = self.wf_analyzer.run_walk_forward(
                    job.model_id,
                    job.returns,
                    job.signals,
                    np.sign(job.returns)  # Actual signals
                )
                results["walk_forward"] = wf_result.to_dict()
            
            # Store results
            with self._lock:
                job.status = "COMPLETED"
                job.completed_at = time.time()
                job.results = results
                
                self._results_store[job_id] = results
                self._total_jobs_processed += 1
                
                # Track duration
                duration = job.completed_at - job.started_at
                self._job_durations.append(duration)
                if len(self._job_durations) > 100:
                    self._job_durations.pop(0)
                
                # Release memory
                self._current_memory_usage_mb -= job.estimated_memory_mb
                del self._job_memory[job_id]
            
            return results
            
        except Exception as e:
            with self._lock:
                job.status = "FAILED"
                job.completed_at = time.time()
                job.error_message = str(e)
                
                # Release memory
                self._current_memory_usage_mb -= job.estimated_memory_mb
                del self._job_memory[job_id]
            
            raise
    
    def _run_parameter_sweep(self, job: BacktestJob) -> Dict[str, Any]:
        """Run parameter sweep optimization."""
        # Flatten parameter grid
        param_names = list(job.param_grid.keys())
        param_values = list(job.param_grid.values())
        
        # Generate all combinations
        from itertools import product
        combinations = list(product(*param_values))
        
        results = {
            "parameters": param_names,
            "combinations": len(combinations),
            "best_params": None,
            "best_sharpe": 0.0,
            "all_results": []
        }
        
        best_sharpe = -np.inf
        best_params = None
        
        for i, combo in enumerate(combinations):
            # Adjust signals based on parameters
            signals = job.signals.copy()
            
            for j, (name, value) in enumerate(zip(param_names, combo)):
                if name == "threshold":
                    signals = np.where(np.abs(signals) > value, np.sign(signals), 0)
                elif name == "scale":
                    signals = signals * value
            
            # Run backtest
            result = self.bt_engine.run_backtest(signals)
            
            if result.sharpe_ratio > best_sharpe:
                best_sharpe = result.sharpe_ratio
                best_params = dict(zip(param_names, combo))
            
            results["all_results"].append({
                "params": dict(zip(param_names, combo)),
                "sharpe": result.sharpe_ratio,
                "return": result.total_return,
                "max_dd": result.max_drawdown
            })
        
        results["best_params"] = best_params
        results["best_sharpe"] = best_sharpe
        
        return results
    
    def get_job_status(self, job_id: str) -> str:
        """Get job status."""
        with self._lock:
            if job_id in self._jobs:
                return self._jobs[job_id].status
            return "UNKNOWN"
    
    def get_job_result(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job results."""
        with self._lock:
            if job_id in self._results_store:
                return self._results_store[job_id]
            return None
    
    def get_cluster_stats(self) -> ClusterStats:
        """Get cluster statistics."""
        with self._lock:
            stats = ClusterStats()
            
            for job in self._jobs.values():
                stats.total_jobs += 1
                if job.status == "PENDING":
                    stats.pending_jobs += 1
                elif job.status == "RUNNING":
                    stats.running_jobs += 1
                elif job.status == "COMPLETED":
                    stats.completed_jobs += 1
                elif job.status == "FAILED":
                    stats.failed_jobs += 1
            
            stats.total_memory_used_mb = self._current_memory_usage_mb
            stats.memory_limit_mb = self.max_memory_mb
            
            if self._job_durations:
                stats.avg_job_duration_s = np.mean(self._job_durations)
            
            # Count model statuses
            quarantined = self.wf_analyzer.get_quarantined_models()
            stats.quarantined_models = len(quarantined)
            stats.active_models = len(self.wf_analyzer._validations) - stats.quarantined_models
            
            return stats
    
    def should_accept_job(self, estimated_memory_mb: float) -> bool:
        """Check if cluster can accept a new job."""
        with self._lock:
            running_count = sum(
                1 for j in self._jobs.values() if j.status == "RUNNING"
            )
            
            if running_count >= self.max_concurrent_jobs:
                return False
            
            if self._current_memory_usage_mb + estimated_memory_mb > self.max_memory_mb:
                return False
            
            return True
    
    def get_next_pending_job(self) -> Optional[str]:
        """Get next pending job ID."""
        with self._lock:
            for job_id in self._job_queue:
                if job_id in self._jobs and self._jobs[job_id].status == "PENDING":
                    return job_id
            return None
    
    def quarantine_model(self, model_id: str) -> bool:
        """Manually quarantine a model."""
        with self._lock:
            if model_id in self.wf_analyzer._validations:
                self.wf_analyzer._validations[model_id].status = ModelStatus.QUARANTINED
                return True
            return False
    
    def get_model_status(self, model_id: str) -> Optional[str]:
        """Get model validation status."""
        status = self.wf_analyzer.get_model_status(model_id)
        return status.name if status else None
    
    def reset(self) -> None:
        """Reset all state."""
        with self._lock:
            self._jobs.clear()
            self._job_queue.clear()
            self._results_store.clear()
            self._job_memory.clear()
            self._current_memory_usage_mb = 0.0
            self._job_durations.clear()
            self._total_jobs_processed = 0
            
            self.bt_engine.reset()
            self.wf_analyzer = WalkForwardAnalyzer()


# Global singleton instance
_bt_cluster_instance: Optional[BacktestClusterManager] = None
_instance_lock = threading.Lock()


def get_bt_cluster() -> BacktestClusterManager:
    """Get or create the global backtest cluster manager."""
    global _bt_cluster_instance
    if _bt_cluster_instance is None:
        with _instance_lock:
            if _bt_cluster_instance is None:
                _bt_cluster_instance = BacktestClusterManager()
    return _bt_cluster_instance


if __name__ == "__main__":
    # Test backtest cluster manager
    print("Testing BacktestClusterManager:")
    
    cluster = BacktestClusterManager(max_memory_mb=500)
    
    np.random.seed(42)
    n_periods = 1000
    
    # Generate synthetic data
    returns = 0.0001 + 0.02 * np.random.randn(n_periods)
    prices = np.cumsum(returns) + 100
    
    # Generate signals
    signals = np.zeros(n_periods)
    for i in range(20, n_periods):
        momentum = prices[i] / prices[i-20] - 1
        signals[i] = np.sign(momentum)
    
    # Submit job
    job_id = cluster.submit_job(
        job_id="test_job_1",
        model_id="momentum_v1",
        signals=signals,
        returns=prices,
        param_grid={"threshold": [0.1, 0.2, 0.3]}
    )
    
    print(f"Submitted job: {job_id}")
    
    # Run job
    results = cluster.run_job(job_id)
    
    print(f"\nJob Status: {cluster.get_job_status(job_id)}")
    print(f"Results keys: {list(results.keys())}")
    
    if "backtest" in results:
        bt = results["backtest"]
        print(f"\nBacktest Results:")
        print(f"  Sharpe: {bt.get('sharpe_ratio', 0):.4f}")
        print(f"  Total Return: {bt.get('total_return', 0):.4f}")
    
    if "walk_forward" in results:
        wf = results["walk_forward"]
        print(f"\nWalk-Forward Results:")
        print(f"  OOS Sharpe: {wf.get('oos_sharpe', 0):.4f}")
        print(f"  Sharpe Decay: {wf.get('sharpe_decay', 0):.4f}")
    
    # Get cluster stats
    stats = cluster.get_cluster_stats()
    print(f"\nCluster Stats: {stats.to_dict()}")
    
    # Check model status
    model_status = cluster.get_model_status("momentum_v1")
    print(f"Model Status: {model_status}")
