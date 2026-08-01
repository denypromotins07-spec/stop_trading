"""
Health Monitor.
Background thread monitoring Python process health, GIL contention, and Ray cluster status.
Triggers automated worker recycling on memory fragmentation or thread starvation.
"""

import threading
import time
import os
import sys
import gc
import logging
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass
from enum import Enum

try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    ray = None

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNHEALTHY = "unhealthy"


@dataclass
class HealthMetrics:
    """Current health metrics."""
    status: HealthStatus
    memory_usage_mb: float
    memory_limit_mb: float
    memory_fragmentation_ratio: float
    gc_count: int
    active_threads: int
    gil_contention_pct: float
    ray_cluster_healthy: bool
    ray_workers_alive: int
    uptime_seconds: float
    timestamp_ns: int


class HealthMonitor:
    """
    Background health monitor for Python HFT processes.
    Monitors memory, threads, GIL, and Ray cluster health.
    """

    # Memory limit (3GB as per requirements)
    MEMORY_LIMIT_MB = 3072

    # Warning thresholds
    MEMORY_WARNING_PCT = 0.8
    FRAGMENTATION_WARNING = 0.3
    GIL_CONTENTION_WARNING = 0.5
    THREAD_STARVATION_THRESHOLD = 2

    def __init__(
        self,
        check_interval_seconds: float = 5.0,
        auto_restart_enabled: bool = True,
    ):
        self.check_interval = check_interval_seconds
        self.auto_restart_enabled = auto_restart_enabled

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._callbacks: List[Callable[[HealthMetrics], None]] = []
        self._worker_pids: List[int] = []
        self._start_time: float = 0
        self._last_gc_count: int = 0
        self._gil_check_count: int = 0
        self._gil_wait_count: int = 0

        self._latest_metrics: Optional[HealthMetrics] = None
        self._consecutive_critical = 0

    def start(self):
        """Start the health monitor background thread."""
        if self._running:
            return

        self._running = True
        self._start_time = time.time()
        self._thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="HealthMonitor",
        )
        self._thread.start()
        logger.info("Health monitor started")

    def stop(self):
        """Stop the health monitor."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.info("Health monitor stopped")

    def register_callback(self, callback: Callable[[HealthMetrics], None]):
        """Register a callback for health status changes."""
        self._callbacks.append(callback)

    def register_worker_pid(self, pid: int):
        """Register a worker PID to monitor."""
        self._worker_pids.append(pid)

    def _monitor_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                metrics = self._collect_metrics()
                self._latest_metrics = metrics

                # Check for issues
                self._check_and_alert(metrics)

                # Notify callbacks
                for callback in self._callbacks:
                    try:
                        callback(metrics)
                    except Exception as e:
                        logger.error(f"Health callback error: {e}")

            except Exception as e:
                logger.error(f"Health monitoring error: {e}")

            time.sleep(self.check_interval)

    def _collect_metrics(self) -> HealthMetrics:
        """Collect current health metrics."""
        import resource

        # Memory usage
        mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KB to MB
        mem_limit = self.MEMORY_LIMIT_MB
        mem_pct = mem_usage / mem_limit

        # Memory fragmentation (estimate via gc stats)
        gc_stats = gc.get_stats()
        total_collected = sum(s.get('collections', 0) for s in gc_stats)
        fragmentation = min(1.0, (total_collected - self._last_gc_count) / 100)
        self._last_gc_count = total_collected

        # Active threads
        active_threads = threading.active_count()

        # GIL contention (simplified estimate)
        gil_contention = self._estimate_gil_contention()

        # Ray cluster health
        ray_healthy = False
        ray_workers = 0
        if RAY_AVAILABLE and ray.is_initialized():
            try:
                ray_healthy = True
                ray_workers = len(ray.nodes())
            except Exception:
                ray_healthy = False

        # Determine overall status
        status = HealthStatus.HEALTHY

        if mem_pct > 0.95 or fragmentation > 0.5:
            status = HealthStatus.CRITICAL
        elif mem_pct > self.MEMORY_WARNING_PCT or fragmentation > self.FRAGMENTATION_WARNING:
            status = HealthStatus.WARNING

        if gil_contention > 0.7:
            status = HealthStatus.CRITICAL
        elif gil_contention > self.GIL_CONTENTION_WARNING:
            if status == HealthStatus.HEALTHY:
                status = HealthStatus.WARNING

        if active_threads < self.THREAD_STARVATION_THRESHOLD:
            if status == HealthStatus.HEALTHY:
                status = HealthStatus.WARNING

        return HealthMetrics(
            status=status,
            memory_usage_mb=mem_usage,
            memory_limit_mb=mem_limit,
            memory_fragmentation_ratio=fragmentation,
            gc_count=total_collected,
            active_threads=active_threads,
            gil_contention_pct=gil_contention,
            ray_cluster_healthy=ray_healthy,
            ray_workers_alive=ray_workers,
            uptime_seconds=time.time() - self._start_time,
            timestamp_ns=time.time_ns(),
        )

    def _estimate_gil_contention(self) -> float:
        """Estimate GIL contention ratio."""
        # Simplified estimation based on thread activity
        self._gil_check_count += 1

        # Count threads waiting (heuristic)
        waiting_threads = 0
        for thread in threading.enumerate():
            if not thread.is_alive():
                waiting_threads += 1

        total_threads = max(1, threading.active_count())
        ratio = waiting_threads / total_threads

        return min(1.0, ratio)

    def _check_and_alert(self, metrics: HealthMetrics):
        """Check metrics and trigger alerts/actions."""
        if metrics.status == HealthStatus.CRITICAL:
            self._consecutive_critical += 1

            if self._consecutive_critical >= 3:
                logger.critical(
                    f"Critical health status for {self._consecutive_critical} consecutive checks! "
                    f"Memory: {metrics.memory_usage_mb:.0f}/{metrics.memory_limit_mb:.0f} MB, "
                    f"Fragmentation: {metrics.memory_fragmentation_ratio:.2%}"
                )

                if self.auto_restart_enabled:
                    self._trigger_worker_recycle()
        else:
            self._consecutive_critical = 0

        # Log warnings
        if metrics.status == HealthStatus.WARNING:
            logger.warning(
                f"Health warning: Memory={metrics.memory_usage_mb:.0f}MB, "
                f"GIL contention={metrics.gil_contention_pct:.2%}"
            )

    def _trigger_worker_recycle(self):
        """Trigger automated worker recycling."""
        logger.info("Triggering worker recycling...")

        # Force garbage collection
        gc.collect()

        # If Ray is available, recycle workers
        if RAY_AVAILABLE and ray.is_initialized():
            try:
                # Shutdown and restart Ray
                ray.shutdown()
                time.sleep(1.0)
                ray.init(num_cpus=4, include_dashboard=False)
                logger.info("Ray cluster recycled")
            except Exception as e:
                logger.error(f"Failed to recycle Ray: {e}")

    def get_current_metrics(self) -> Optional[HealthMetrics]:
        """Get the latest health metrics."""
        return self._latest_metrics

    def is_healthy(self) -> bool:
        """Check if system is currently healthy."""
        if self._latest_metrics:
            return self._latest_metrics.status == HealthStatus.HEALTHY
        return True

    def force_gc(self):
        """Force garbage collection."""
        collected = gc.collect()
        logger.info(f"Forced GC collected {collected} objects")


# Module singleton
_monitor: Optional[HealthMonitor] = None


def get_health_monitor(
    check_interval: float = 5.0,
    auto_restart: bool = True,
) -> HealthMonitor:
    """Get or create the health monitor singleton."""
    global _monitor
    if _monitor is None:
        _monitor = HealthMonitor(
            check_interval_seconds=check_interval,
            auto_restart_enabled=auto_restart,
        )
    return _monitor


async def initialize_health_monitor(
    check_interval: float = 5.0,
) -> HealthMonitor:
    """Initialize and start the health monitor."""
    monitor = get_health_monitor(check_interval=check_interval)
    monitor.start()
    return monitor


async def shutdown_health_monitor():
    """Shutdown the health monitor."""
    global _monitor
    if _monitor:
        _monitor.stop()
        _monitor = None
