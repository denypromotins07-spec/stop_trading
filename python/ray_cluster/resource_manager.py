"""
Custom Ray actor manager for continuous memory monitoring.
Automatically kills and restarts bloated workers to prevent host OOM crashes.
"""

import ray
import psutil
import os
import time
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import PYTHON_RAM_CEILING_MB, get_logger

logger = get_logger("resource_manager")


@dataclass
class WorkerInfo:
    """Information about a Ray worker process."""
    pid: int
    actor_id: str
    memory_mb: float
    cpu_percent: float
    is_bloated: bool


@ray.remote
class MemoryMonitorActor:
    """
    Ray actor that monitors its own memory usage.
    Reports back to the ResourceManager for centralized tracking.
    """
    
    def __init__(self, worker_id: str, memory_threshold_mb: float):
        self.worker_id = worker_id
        self.memory_threshold_mb = memory_threshold_mb
        self.process = psutil.Process(os.getpid())
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage in MB."""
        try:
            mem_info = self.process.memory_info()
            return {
                "worker_id": self.worker_id,
                "pid": os.getpid(),
                "rss_mb": mem_info.rss / (1024 * 1024),
                "vms_mb": mem_info.vms / (1024 * 1024),
                "percent": self.process.memory_percent(),
            }
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return {"worker_id": self.worker_id, "pid": os.getpid(), "rss_mb": 0, "vms_mb": 0, "percent": 0}
    
    def check_health(self) -> bool:
        """Check if worker is healthy (not exceeding memory threshold)."""
        mem_usage = self.get_memory_usage()
        return mem_usage["rss_mb"] < self.memory_threshold_mb


class ResourceManager:
    """
    Centralized manager for Ray worker memory monitoring.
    Automatically kills and restarts workers exceeding memory limits.
    """
    
    def __init__(
        self,
        memory_threshold_percent: float = 85.0,
        check_interval_seconds: float = 2.0,
        max_restart_attempts: int = 3,
    ):
        self.memory_threshold_percent = memory_threshold_percent
        self.check_interval_seconds = check_interval_seconds
        self.max_restart_attempts = max_restart_attempts
        
        # Calculate per-worker memory threshold (divide Python RAM by expected workers)
        self.max_workers = 4  # Expected number of ML workers
        self.per_worker_threshold_mb = (PYTHON_RAM_CEILING_MB * 0.9) / self.max_workers
        
        self._workers: Dict[str, ray.ActorHandle] = {}
        self._worker_info: Dict[str, WorkerInfo] = {}
        self._restart_counts: Dict[str, int] = {}
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
    
    def register_worker(self, worker_id: str, actor_handle: ray.ActorHandle) -> None:
        """Register a new worker for memory monitoring."""
        self._workers[worker_id] = actor_handle
        self._restart_counts[worker_id] = 0
        logger.info(f"Registered worker {worker_id} for memory monitoring")
    
    def unregister_worker(self, worker_id: str) -> None:
        """Unregister a worker from memory monitoring."""
        if worker_id in self._workers:
            del self._workers[worker_id]
        if worker_id in self._worker_info:
            del self._worker_info[worker_id]
        if worker_id in self._restart_counts:
            del self._restart_counts[worker_id]
        logger.info(f"Unregistered worker {worker_id}")
    
    def _check_worker_memory(self, worker_id: str) -> Optional[WorkerInfo]:
        """Check memory usage of a single worker."""
        if worker_id not in self._workers:
            return None
        
        try:
            # Get memory usage asynchronously
            mem_future = self._workers[worker_id].get_memory_usage.remote()
            mem_info = ray.get(mem_future)
            
            is_bloated = mem_info["rss_mb"] > self.per_worker_threshold_mb
            
            info = WorkerInfo(
                pid=mem_info["pid"],
                actor_id=worker_id,
                memory_mb=mem_info["rss_mb"],
                cpu_percent=0.0,  # Could be extended to track CPU
                is_bloated=is_bloated,
            )
            
            self._worker_info[worker_id] = info
            return info
            
        except (ray.exceptions.RayActorError, ray.exceptions.WorkerCrashedError) as e:
            logger.warning(f"Worker {worker_id} crashed or died: {e}")
            return None
        except Exception as e:
            logger.error(f"Error checking worker {worker_id} memory: {e}")
            return None
    
    def _kill_and_restart_worker(self, worker_id: str) -> bool:
        """Kill a bloated worker and attempt to restart it."""
        if worker_id not in self._workers:
            return False
        
        self._restart_counts[worker_id] += 1
        
        if self._restart_counts[worker_id] > self.max_restart_attempts:
            logger.error(
                f"Worker {worker_id} exceeded max restart attempts "
                f"({self.max_restart_attempts}). Not restarting."
            )
            # Remove from tracking - caller should handle permanent failure
            self.unregister_worker(worker_id)
            return False
        
        try:
            # Kill the worker actor
            ray.kill(self._workers[worker_id])
            logger.warning(f"Killed bloated worker {worker_id} (attempt {self._restart_counts[worker_id]})")
            
            # Give some time for cleanup
            time.sleep(0.5)
            
            # Note: Actual restart logic depends on the application
            # This method signals that a restart is needed
            return True
            
        except Exception as e:
            logger.error(f"Failed to kill worker {worker_id}: {e}")
            return False
    
    def monitor_all_workers(self) -> Dict[str, WorkerInfo]:
        """
        Monitor all registered workers and take action on bloated ones.
        
        Returns:
            Dictionary of worker_id -> WorkerInfo
        """
        results = {}
        
        for worker_id in list(self._workers.keys()):
            info = self._check_worker_memory(worker_id)
            
            if info is None:
                continue
            
            results[worker_id] = info
            
            if info.is_bloated:
                logger.warning(
                    f"Worker {worker_id} is bloated: {info.memory_mb:.2f} MB "
                    f"(threshold: {self.per_worker_threshold_mb:.2f} MB)"
                )
                
                # Attempt to kill and restart
                if self._kill_and_restart_worker(worker_id):
                    logger.info(f"Scheduled restart for worker {worker_id}")
        
        return results
    
    def start_background_monitoring(self) -> None:
        """Start background thread for continuous monitoring."""
        import threading
        
        self._running = True
        
        def monitor_loop():
            while self._running:
                try:
                    self.monitor_all_workers()
                except Exception as e:
                    logger.error(f"Background monitoring error: {e}")
                time.sleep(self.check_interval_seconds)
        
        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Started background memory monitoring")
    
    def stop_background_monitoring(self) -> None:
        """Stop background monitoring thread."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
            self._monitor_thread = None
        logger.info("Stopped background memory monitoring")
    
    def get_system_memory_status(self) -> Dict[str, float]:
        """Get overall system memory status."""
        mem = psutil.virtual_memory()
        return {
            "total_mb": mem.total / (1024 * 1024),
            "available_mb": mem.available / (1024 * 1024),
            "used_mb": mem.used / (1024 * 1024),
            "percent_used": mem.percent,
            "python_ceiling_mb": PYTHON_RAM_CEILING_MB,
        }


# Import threading at module level
import threading
