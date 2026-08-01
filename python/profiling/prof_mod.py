"""
Profiling Module Root - Exposes profiling metrics to Prometheus and triggers automated worker restarts.
Coordinates GIL monitoring and memory tracing for comprehensive Python-side observability.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
import time
import os
import sys

from .gil_monitor import GILMonitor, get_gil_monitor, GILContentionEvent
from .memory_tracer import MemoryTracer, get_memory_tracer, MemoryLeakAlert

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class WorkerHealth:
    """Health status of a worker process."""
    worker_id: str
    timestamp: float
    is_healthy: bool
    memory_usage_pct: float
    gil_contention_rate: float
    uptime_seconds: float
    issues: List[str]
    action_recommended: str  # 'none', 'restart_soon', 'restart_immediate'
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "timestamp": self.timestamp,
            "is_healthy": self.is_healthy,
            "memory_usage_pct": self.memory_usage_pct,
            "gil_contention_rate": self.gil_contention_rate,
            "uptime_seconds": self.uptime_seconds,
            "issues": self.issues,
            "action_recommended": self.action_recommended
        }


@dataclass
class PrometheusMetric:
    """Metric formatted for Prometheus export."""
    name: str
    value: float
    labels: Dict[str, str]
    metric_type: str  # 'gauge', 'counter', 'histogram'
    timestamp: float
    
    def format_prometheus(self) -> str:
        """Format as Prometheus text exposition."""
        label_str = ",".join(f'{k}="{v}"' for k, v in self.labels.items())
        if label_str:
            return f"{self.name}{{{label_str}}} {self.value} {int(self.timestamp)}"
        return f"{self.name} {self.value} {int(self.timestamp)}"


class ProfilingModule:
    """
    Central module for Python profiling and observability.
    Coordinates GIL monitoring, memory tracing, and Prometheus metrics export.
    """
    
    def __init__(self,
                 max_memory_mb: float = 3000.0,
                 memory_warning_pct: float = 80.0,
                 memory_critical_pct: float = 95.0,
                 gil_contention_threshold: float = 0.1,
                 auto_restart_enabled: bool = True,
                 prometheus_port: int = 9090):
        """
        Initialize profiling module.
        
        Args:
            max_memory_mb: Maximum allowed memory (3GB limit)
            memory_warning_pct: Percentage threshold for warning
            memory_critical_pct: Percentage threshold for critical alert
            gil_contention_threshold: Contention rate threshold
            auto_restart_enabled: Whether to trigger automatic restarts
            prometheus_port: Port for Prometheus metrics endpoint
        """
        self.max_memory_mb = max_memory_mb
        self.memory_warning_pct = memory_warning_pct
        self.memory_critical_pct = memory_critical_pct
        self.gil_contention_threshold = gil_contention_threshold
        self.auto_restart_enabled = auto_restart_enabled
        self.prometheus_port = prometheus_port
        
        # Initialize components
        self.gil_monitor = get_gil_monitor()
        self.memory_tracer = get_memory_tracer(max_memory_mb=max_memory_mb)
        
        # Health tracking
        self._worker_start_time: float = time.time()
        self._worker_id: str = f"worker_{os.getpid()}"
        self._health_history: deque = __import__('collections').deque(maxlen=100)
        
        # Callbacks
        self._restart_callbacks: List[Callable] = []
        self._alert_callbacks: List[Callable] = []
        
        # Background tasks
        self._monitor_task: Optional[asyncio.Task] = None
        self._prometheus_task: Optional[asyncio.Task] = None
        self._is_running = False
        
        # Metrics cache
        self._metrics_cache: List[PrometheusMetric] = []
    
    async def start(self):
        """Start the profiling module."""
        self._is_running = True
        self._worker_start_time = time.time()
        
        logger.info(f"Starting profiling module for worker {self._worker_id}")
        
        # Start background tasks
        stop_event = asyncio.Event()
        self._stop_event = stop_event
        
        self._monitor_task = asyncio.create_task(
            self._run_health_checks(stop_event)
        )
        
        logger.info("Profiling module started")
    
    async def stop(self):
        """Stop the profiling module."""
        self._is_running = False
        
        if hasattr(self, '_stop_event'):
            self._stop_event.set()
        
        if self._monitor_task:
            try:
                await asyncio.wait_for(self._monitor_task, timeout=5.0)
            except asyncio.TimeoutError:
                self._monitor_task.cancel()
        
        logger.info("Profiling module stopped")
    
    async def _run_health_checks(self, stop_event: asyncio.Event):
        """Run periodic health checks."""
        while not stop_event.is_set():
            try:
                health = self._check_worker_health()
                self._health_history.append(health)
                
                # Check if restart needed
                if health.action_recommended != 'none':
                    await self._handle_degradation(health)
                
                # Update metrics
                self._update_metrics(health)
                
                await asyncio.sleep(5.0)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Health check error: {e}")
                await asyncio.sleep(5.0)
    
    def _check_worker_health(self) -> WorkerHealth:
        """Perform comprehensive health check."""
        issues = []
        action = 'none'
        
        # Check memory
        mem_health = self.memory_tracer.health_check()
        memory_pct = mem_health.get('usage_percentage', 0)
        
        if memory_pct > self.memory_critical_pct:
            issues.append(f"CRITICAL: Memory at {memory_pct:.1f}%")
            action = 'restart_immediate'
        elif memory_pct > self.memory_warning_pct:
            issues.append(f"WARNING: Memory at {memory_pct:.1f}%")
            if action == 'none':
                action = 'restart_soon'
        
        # Check GIL contention
        gil_health = self.gil_monitor.health_check()
        contention_rate = gil_health.get('contention_rate', 0)
        
        if contention_rate > self.gil_contention_threshold * 2:
            issues.append(f"HIGH: GIL contention rate {contention_rate:.2%}")
            if action == 'none':
                action = 'restart_soon'
        elif contention_rate > self.gil_contention_threshold:
            issues.append(f"MEDIUM: GIL contention rate {contention_rate:.2%}")
        
        # Check uptime (suggest restart after 24 hours)
        uptime = time.time() - self._worker_start_time
        if uptime > 86400:  # 24 hours
            issues.append(f"INFO: Uptime {uptime/3600:.1f} hours, consider restart")
            if action == 'none':
                action = 'restart_soon'
        
        is_healthy = len(issues) == 0 or action == 'none'
        
        return WorkerHealth(
            worker_id=self._worker_id,
            timestamp=time.time(),
            is_healthy=is_healthy,
            memory_usage_pct=memory_pct,
            gil_contention_rate=contention_rate,
            uptime_seconds=uptime,
            issues=issues,
            action_recommended=action
        )
    
    async def _handle_degradation(self, health: WorkerHealth):
        """Handle worker degradation."""
        logger.warning(
            f"Worker degradation detected: {health.action_recommended} - "
            f"issues: {health.issues}"
        )
        
        if health.action_recommended == 'restart_immediate':
            # Send immediate alerts
            for callback in self._alert_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(health)
                    else:
                        callback(health)
                except Exception as e:
                    logger.error(f"Alert callback error: {e}")
            
            # Trigger restart if enabled
            if self.auto_restart_enabled:
                await self._trigger_restart("immediate", health.issues)
        
        elif health.action_recommended == 'restart_soon':
            # Schedule graceful restart
            if self.auto_restart_enabled:
                await self._trigger_restart("graceful", health.issues)
    
    async def _trigger_restart(self, restart_type: str, reasons: List[str]):
        """Trigger worker restart."""
        logger.critical(
            f"Triggering {restart_type} restart for worker {self._worker_id}: "
            f"{reasons}"
        )
        
        for callback in self._restart_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(restart_type, reasons)
                else:
                    callback(restart_type, reasons)
            except Exception as e:
                logger.error(f"Restart callback error: {e}")
        
        # In production, would signal to orchestration system
        # For now, just log
        logger.info(f"Restart signal sent for {restart_type} restart")
    
    def _update_metrics(self, health: WorkerHealth):
        """Update Prometheus metrics."""
        now = time.time()
        
        self._metrics_cache = [
            PrometheusMetric(
                name="python_worker_memory_usage_percent",
                value=health.memory_usage_pct,
                labels={"worker_id": self._worker_id},
                metric_type="gauge",
                timestamp=now
            ),
            PrometheusMetric(
                name="python_worker_gil_contention_rate",
                value=health.gil_contention_rate,
                labels={"worker_id": self._worker_id},
                metric_type="gauge",
                timestamp=now
            ),
            PrometheusMetric(
                name="python_worker_uptime_seconds",
                value=health.uptime_seconds,
                labels={"worker_id": self._worker_id},
                metric_type="counter",
                timestamp=now
            ),
            PrometheusMetric(
                name="python_worker_health_status",
                value=1.0 if health.is_healthy else 0.0,
                labels={"worker_id": self._worker_id},
                metric_type="gauge",
                timestamp=now
            ),
        ]
    
    def get_prometheus_metrics(self) -> str:
        """Get metrics in Prometheus text format."""
        if not self._metrics_cache:
            return "# No metrics available\n"
        
        lines = [
            "# HELP python_worker_memory_usage_percent Worker memory usage percentage",
            "# TYPE python_worker_memory_usage_percent gauge",
            "# HELP python_worker_gil_contention_rate GIL contention rate",
            "# TYPE python_worker_gil_contention_rate gauge",
            "# HELP python_worker_uptime_seconds Worker uptime in seconds",
            "# TYPE python_worker_uptime_seconds counter",
            "# HELP python_worker_health_status Worker health status (1=healthy, 0=unhealthy)",
            "# TYPE python_worker_health_status gauge",
        ]
        
        for metric in self._metrics_cache:
            lines.append(metric.format_prometheus())
        
        return "\n".join(lines) + "\n"
    
    def register_restart_callback(self, callback: Callable):
        """Register a callback for restart events."""
        self._restart_callbacks.append(callback)
        logger.info(f"Registered restart callback: {callback.__name__}")
    
    def register_alert_callback(self, callback: Callable):
        """Register a callback for alerts."""
        self._alert_callbacks.append(callback)
        logger.info(f"Registered alert callback: {callback.__name__}")
    
    def get_module_stats(self) -> Dict[str, Any]:
        """Get module statistics."""
        return {
            "worker_id": self._worker_id,
            "is_running": self._is_running,
            "uptime_seconds": time.time() - self._worker_start_time,
            "memory_tracer": self.memory_tracer.get_tracer_stats(),
            "gil_monitor": self.gil_monitor.get_monitor_stats(),
            "health_checks_performed": len(self._health_history),
            "last_health": self._health_history[-1].to_dict() if self._health_history else None
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Return module health status."""
        latest = self._health_history[-1] if self._health_history else None
        
        return {
            "running": self._is_running,
            "worker_id": self._worker_id,
            "healthy": latest.is_healthy if latest else True,
            "memory_tracer": self.memory_tracer.health_check(),
            "gil_monitor": self.gil_monitor.health_check(),
            "auto_restart_enabled": self.auto_restart_enabled
        }
    
    def force_gc_and_report(self) -> Dict[str, Any]:
        """Force garbage collection and report results."""
        gc_result = self.memory_tracer.force_gc()
        
        # Take new snapshot after GC
        time.sleep(0.1)
        usage = self.memory_tracer.get_current_usage()
        
        return {
            "gc_result": gc_result,
            "memory_after_gc": usage,
            "freed_estimate_mb": gc_result.get('collected_objects', 0) / 1000  # Rough estimate
        }


# Module singleton
_prof_module: Optional[ProfilingModule] = None


def get_profiling_module(**kwargs) -> ProfilingModule:
    """Get or create the global profiling module."""
    global _prof_module
    
    if _prof_module is None:
        _prof_module = ProfilingModule(**kwargs)
        logger.info("Created profiling module")
    
    return _prof_module


async def initialize_profiling(auto_restart: bool = True) -> ProfilingModule:
    """Initialize and start the profiling module."""
    module = get_profiling_module(auto_restart_enabled=auto_restart)
    await module.start()
    return module


if __name__ == "__main__":
    # Test the profiling module
    print("Testing Profiling Module...")
    
    module = ProfilingModule(
        max_memory_mb=500.0,  # Lower for testing
        auto_restart_enabled=False
    )
    
    async def run_test():
        await module.start()
        
        # Simulate some work
        print("\nSimulating work...")
        for i in range(5):
            health = module._check_worker_health()
            print(f"Health check {i+1}: healthy={health.is_healthy}, memory={health.memory_usage_pct:.1f}%")
            
            # Allocate some memory
            data = bytearray(1024 * 1024 * 10)  # 10 MB
            time.sleep(0.5)
        
        # Get metrics
        print("\nPrometheus Metrics:")
        print(module.get_prometheus_metrics())
        
        # Force GC
        print(f"\nGC Result: {module.force_gc_and_report()}")
        
        print(f"\nStats: {module.get_module_stats()}")
        print(f"Health: {module.health_check()}")
        
        await module.stop()
    
    asyncio.run(run_test())
