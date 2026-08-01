"""
AutoML Module Root for Bayesian Hyperparameter Optimization.
Schedules optimization jobs strictly during the bot's 20-hour offline window
to prevent resource contention with live trading.

Provides:
- Scheduled job management
- Resource allocation within 3GB RAM limit
- Integration with Stage 37 backtesting engine
"""

import numpy as np
from typing import Dict, Any, Optional, Callable, List
import threading
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
import json

from .bayesian_optimizer import (
    BayesianOptimizer, 
    OptimizationConfig, 
    get_optimizer,
    TrialResult
)
from .objective_function import ObjectiveFunction, create_objective_function

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ScheduledJob:
    """Represents a scheduled optimization job."""
    job_id: str
    strategy_name: str
    param_bounds: Dict[str, tuple]
    start_time: datetime
    max_duration_hours: float
    priority: int  # 1=highest, 5=lowest
    status: str  # 'pending', 'running', 'completed', 'failed', 'cancelled'
    created_at: datetime
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None


class AutoMLScheduler:
    """
    Scheduler for Bayesian optimization jobs.
    Ensures jobs run only during offline windows (20 hours/day).
    """
    
    # Offline window: 4 hours for live trading, 20 hours for training
    OFFLINE_START_HOUR = 4  # 04:00 UTC
    OFFLINE_END_HOUR = 0    # 00:00 UTC (next day)
    
    MAX_CONCURRENT_JOBS = 2
    MAX_RAM_PER_JOB_MB = 400
    
    def __init__(self, results_dir: str = "./automl_results"):
        self._lock = threading.RLock()
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self._scheduled_jobs: List[ScheduledJob] = []
        self._current_jobs: List[ScheduledJob] = []
        self._job_history: List[ScheduledJob] = []
        
        self._optimizer_instances: Dict[str, BayesianOptimizer] = {}
        self._running = False
        self._scheduler_thread: Optional[threading.Thread] = None
        
        # Load existing schedule from disk
        self._load_schedule()
    
    def _load_schedule(self) -> None:
        """Load scheduled jobs from disk."""
        schedule_file = self.results_dir / "schedule.json"
        if schedule_file.exists():
            try:
                with open(schedule_file, 'r') as f:
                    data = json.load(f)
                    for job_data in data.get('pending_jobs', []):
                        job = ScheduledJob(
                            job_id=job_data['job_id'],
                            strategy_name=job_data['strategy_name'],
                            param_bounds=job_data['param_bounds'],
                            start_time=datetime.fromisoformat(job_data['start_time']),
                            max_duration_hours=job_data['max_duration_hours'],
                            priority=job_data['priority'],
                            status=job_data['status'],
                            created_at=datetime.fromisoformat(job_data['created_at'])
                        )
                        self._scheduled_jobs.append(job)
                logger.info(f"Loaded {len(self._scheduled_jobs)} pending jobs")
            except Exception as e:
                logger.warning(f"Failed to load schedule: {e}")
    
    def _save_schedule(self) -> None:
        """Save schedule to disk."""
        schedule_file = self.results_dir / "schedule.json"
        try:
            data = {
                'pending_jobs': [
                    {
                        'job_id': job.job_id,
                        'strategy_name': job.strategy_name,
                        'param_bounds': job.param_bounds,
                        'start_time': job.start_time.isoformat(),
                        'max_duration_hours': job.max_duration_hours,
                        'priority': job.priority,
                        'status': job.status,
                        'created_at': job.created_at.isoformat()
                    }
                    for job in self._scheduled_jobs
                ],
                'last_updated': datetime.now().isoformat()
            }
            with open(schedule_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save schedule: {e}")
    
    def is_offline_window(self) -> bool:
        """Check if current time is within offline training window."""
        now = datetime.utcnow()
        current_hour = now.hour
        
        # Offline window: 04:00 to 00:00 (next day)
        if self.OFFLINE_START_HOUR <= current_hour < 24:
            return True
        elif 0 <= current_hour < self.OFFLINE_END_HOUR:
            return True
        return False
    
    def schedule_job(
        self,
        strategy_name: str,
        param_bounds: Dict[str, tuple],
        start_time: Optional[datetime] = None,
        max_duration_hours: float = 4.0,
        priority: int = 3
    ) -> str:
        """
        Schedule an optimization job.
        
        Args:
            strategy_name: Name of the strategy to optimize
            param_bounds: Parameter search space
            start_time: When to start (defaults to next offline window)
            max_duration_hours: Maximum runtime
            priority: Job priority (1=highest)
            
        Returns:
            Job ID
        """
        import uuid
        
        if start_time is None:
            # Schedule for next offline window
            start_time = self._get_next_offline_start()
        
        job_id = str(uuid.uuid4())[:8]
        
        job = ScheduledJob(
            job_id=job_id,
            strategy_name=strategy_name,
            param_bounds=param_bounds,
            start_time=start_time,
            max_duration_hours=max_duration_hours,
            priority=priority,
            status='pending',
            created_at=datetime.utcnow()
        )
        
        with self._lock:
            self._scheduled_jobs.append(job)
            self._scheduled_jobs.sort(key=lambda j: (j.start_time, j.priority))
            self._save_schedule()
        
        logger.info(f"Scheduled job {job_id} for {strategy_name} at {start_time}")
        return job_id
    
    def _get_next_offline_start(self) -> datetime:
        """Get the start time of the next offline window."""
        now = datetime.utcnow()
        
        # If currently in offline window and before 20:00, start now
        if self.is_offline_window() and now.hour < 20:
            return now
        
        # Otherwise, start at 04:00 tomorrow (or today if before 04:00)
        if now.hour < self.OFFLINE_START_HOUR:
            return now.replace(hour=self.OFFLINE_START_HOUR, minute=0, second=0, microsecond=0)
        else:
            tomorrow = now + timedelta(days=1)
            return tomorrow.replace(hour=self.OFFLINE_START_HOUR, minute=0, second=0, microsecond=0)
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a scheduled job."""
        with self._lock:
            for i, job in enumerate(self._scheduled_jobs):
                if job.job_id == job_id:
                    if job.status == 'pending':
                        job.status = 'cancelled'
                        self._scheduled_jobs.pop(i)
                        self._save_schedule()
                        logger.info(f"Cancelled job {job_id}")
                        return True
                    elif job.status == 'running':
                        # Try to stop running optimizer
                        if job_id in self._optimizer_instances:
                            self._optimizer_instances[job_id].shutdown()
                        job.status = 'cancelled'
                        job.completed_at = datetime.utcnow()
                        self._save_schedule()
                        logger.info(f"Cancelled running job {job_id}")
                        return True
            return False
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a job."""
        with self._lock:
            # Check scheduled jobs
            for job in self._scheduled_jobs:
                if job.job_id == job_id:
                    return {
                        'job_id': job.job_id,
                        'strategy_name': job.strategy_name,
                        'status': job.status,
                        'start_time': job.start_time.isoformat(),
                        'created_at': job.created_at.isoformat()
                    }
            
            # Check current jobs
            for job in self._current_jobs:
                if job.job_id == job_id:
                    return {
                        'job_id': job.job_id,
                        'strategy_name': job.strategy_name,
                        'status': job.status,
                        'progress': self._get_job_progress(job_id)
                    }
            
            # Check history
            for job in self._job_history:
                if job.job_id == job_id:
                    return {
                        'job_id': job.job_id,
                        'strategy_name': job.strategy_name,
                        'status': job.status,
                        'result': job.result,
                        'completed_at': job.completed_at.isoformat() if job.completed_at else None
                    }
        
        return None
    
    def _get_job_progress(self, job_id: str) -> Dict[str, Any]:
        """Get progress of a running job."""
        if job_id not in self._optimizer_instances:
            return {'trials_completed': 0, 'trials_total': 0}
        
        optimizer = self._optimizer_instances[job_id]
        summary = optimizer.get_optimization_summary()
        
        return {
            'trials_completed': summary.get('total_trials', 0),
            'trials_total': optimizer.config.max_total_trials if optimizer.config else 0,
            'best_score': summary.get('best_sharpe', 0)
        }
    
    def run_pending_jobs(self, objective_fn_factory: Callable[[str], Callable]) -> None:
        """
        Run pending jobs that are due.
        
        Args:
            objective_fn_factory: Factory function that creates objective function given strategy name
        """
        now = datetime.utcnow()
        
        with self._lock:
            # Check if we're in offline window
            if not self.is_offline_window():
                logger.debug("Not in offline window, skipping job execution")
                return
            
            # Check concurrent job limit
            available_slots = self.MAX_CONCURRENT_JOBS - len(self._current_jobs)
            if available_slots <= 0:
                logger.debug("No available slots for new jobs")
                return
            
            # Find jobs ready to run
            jobs_to_start = []
            for job in self._scheduled_jobs:
                if job.status == 'pending' and job.start_time <= now:
                    jobs_to_start.append(job)
                    if len(jobs_to_start) >= available_slots:
                        break
            
            # Start jobs
            for job in jobs_to_start:
                self._start_job(job, objective_fn_factory)
    
    def _start_job(
        self, 
        job: ScheduledJob, 
        objective_fn_factory: Callable[[str], Callable]
    ) -> None:
        """Start a single optimization job."""
        try:
            job.status = 'running'
            self._scheduled_jobs.remove(job)
            self._current_jobs.append(job)
            
            # Create optimizer config
            config = OptimizationConfig(
                param_bounds=job.param_bounds,
                max_concurrent_trials=2,  # Conservative within job
                max_total_trials=30,
                results_dir=str(self.results_dir / job.job_id)
            )
            
            # Create optimizer
            optimizer = BayesianOptimizer(config)
            self._optimizer_instances[job.job_id] = optimizer
            
            # Create objective function
            objective_fn = objective_fn_factory(job.strategy_name)
            
            # Run optimization in separate thread
            def run_optimization():
                try:
                    result = optimizer.run_optimization(objective_fn, name=job.strategy_name)
                    
                    with self._lock:
                        job.status = 'completed'
                        job.completed_at = datetime.utcnow()
                        
                        if result is not None:
                            best_params = optimizer.get_best_parameters()
                            summary = optimizer.get_optimization_summary()
                            job.result = {
                                'best_params': best_params,
                                'summary': summary
                            }
                            
                            # Save results
                            self._save_job_results(job)
                        else:
                            job.result = {'error': 'Optimization returned no results'}
                        
                        self._current_jobs.remove(job)
                        self._job_history.append(job)
                    
                    logger.info(f"Job {job.job_id} completed")
                    
                except Exception as e:
                    logger.error(f"Job {job.job_id} failed: {e}")
                    with self._lock:
                        job.status = 'failed'
                        job.completed_at = datetime.utcnow()
                        job.result = {'error': str(e)}
                        self._current_jobs.remove(job)
                        self._job_history.append(job)
                finally:
                    optimizer.shutdown()
                    with self._lock:
                        if job.job_id in self._optimizer_instances:
                            del self._optimizer_instances[job.job_id]
            
            thread = threading.Thread(target=run_optimization, daemon=True)
            thread.start()
            
        except Exception as e:
            logger.error(f"Failed to start job {job.job_id}: {e}")
            job.status = 'failed'
            job.completed_at = datetime.utcnow()
            job.result = {'error': str(e)}
            with self._lock:
                if job in self._current_jobs:
                    self._current_jobs.remove(job)
                self._job_history.append(job)
    
    def _save_job_results(self, job: ScheduledJob) -> None:
        """Save job results to disk."""
        results_file = self.results_dir / job.job_id / "results.json"
        try:
            with open(results_file, 'w') as f:
                json.dump({
                    'job_id': job.job_id,
                    'strategy_name': job.strategy_name,
                    'completed_at': job.completed_at.isoformat() if job.completed_at else None,
                    'result': job.result
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save results for job {job.job_id}: {e}")
    
    def start_scheduler(self, objective_fn_factory: Callable[[str], Callable]) -> None:
        """Start the background scheduler thread."""
        if self._running:
            return
        
        self._running = True
        
        def scheduler_loop():
            while self._running:
                try:
                    self.run_pending_jobs(objective_fn_factory)
                except Exception as e:
                    logger.error(f"Scheduler error: {e}")
                
                # Check every 5 minutes
                time.sleep(300)
        
        self._scheduler_thread = threading.Thread(target=scheduler_loop, daemon=True)
        self._scheduler_thread.start()
        logger.info("AutoML scheduler started")
    
    def stop_scheduler(self) -> None:
        """Stop the scheduler thread."""
        self._running = False
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=10)
        logger.info("AutoML scheduler stopped")
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status."""
        with self._lock:
            return {
                'pending_jobs': len(self._scheduled_jobs),
                'running_jobs': len(self._current_jobs),
                'completed_jobs': len([j for j in self._job_history if j.status == 'completed']),
                'failed_jobs': len([j for j in self._job_history if j.status == 'failed']),
                'in_offline_window': self.is_offline_window(),
                'next_job_time': self._scheduled_jobs[0].start_time.isoformat() if self._scheduled_jobs else None
            }
    
    def shutdown(self) -> None:
        """Shutdown scheduler and all running optimizers."""
        self.stop_scheduler()
        
        with self._lock:
            # Cancel all pending jobs
            for job in self._scheduled_jobs:
                job.status = 'cancelled'
            self._scheduled_jobs.clear()
            
            # Stop all running jobs
            for job in self._current_jobs:
                if job.job_id in self._optimizer_instances:
                    self._optimizer_instances[job.job_id].shutdown()
            self._current_jobs.clear()
        
        logger.info("AutoML scheduler shut down complete")


# Global singleton instance
_automl_instance: Optional[AutoMLScheduler] = None
_automl_lock = threading.Lock()


def get_automl_scheduler(results_dir: str = "./automl_results") -> AutoMLScheduler:
    """Thread-safe singleton access to AutoML scheduler."""
    global _automl_instance
    
    with _automl_lock:
        if _automl_instance is None:
            _automl_instance = AutoMLScheduler(results_dir)
        
        return _automl_instance


if __name__ == "__main__":
    # Demo usage
    import sys
    
    scheduler = get_automl_scheduler()
    
    print(f"Offline window: {scheduler.is_offline_window()}")
    print(f"Queue status: {scheduler.get_queue_status()}")
    
    # Schedule a demo job
    job_id = scheduler.schedule_job(
        strategy_name="momentum_strategy",
        param_bounds={
            'lookback': (10, 60),
            'entry_threshold': (0.3, 0.8),
            'stop_loss': (0.01, 0.05)
        },
        max_duration_hours=2.0,
        priority=2
    )
    
    print(f"Scheduled job: {job_id}")
    print(f"Updated queue: {scheduler.get_queue_status()}")
    
    # Note: In production, you would call scheduler.start_scheduler() with a proper objective factory
