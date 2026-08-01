"""
UI Module Root for Python Backend.
Manages non-blocking UI update loop ensuring rendering never blocks GIL or delays inference.

Provides:
- Unified interface for Rich dashboard and log aggregator
- Async update coordination
- ZMQ communication with Rust TUI
"""

import asyncio
import threading
import logging
import time
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass

from .rich_dashboard import RichDashboard, get_dashboard, ClusterHealth, NautilusState, MLInferenceQueue
from .log_aggregator import LogAggregator, get_log_aggregator, CriticalAlert

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class UIConfig:
    """Configuration for UI module."""
    dashboard_endpoint: str = "tcp://localhost:5555"
    kill_switch_endpoint: str = "tcp://localhost:5556"
    update_interval_seconds: float = 1.0
    enable_dashboard: bool = True
    enable_log_aggregation: bool = True


class UIManager:
    """
    Central manager for Python-side UI components.
    Coordinates dashboard updates and log aggregation without blocking inference.
    """
    
    def __init__(self, config: Optional[UIConfig] = None):
        self.config = config or UIConfig()
        
        self._lock = threading.RLock()
        
        # Initialize components
        self._dashboard: Optional[RichDashboard] = None
        self._log_aggregator: Optional[LogAggregator] = None
        
        if self.config.enable_dashboard:
            self._dashboard = get_dashboard(self.config.dashboard_endpoint)
        
        if self.config.enable_log_aggregation:
            self._log_aggregator = get_log_aggregator(self.config.kill_switch_endpoint)
        
        # Update loop control
        self._running = False
        self._update_thread: Optional[threading.Thread] = None
        
        # State providers (callbacks that provide current state)
        self._state_providers: Dict[str, Callable[[], Any]] = {}
        
        # Alert callbacks
        self._alert_callbacks: list = []
    
    def register_state_provider(
        self,
        name: str,
        provider: Callable[[], Any]
    ) -> None:
        """
        Register a state provider callback.
        
        Args:
            name: State name ('cluster', 'nautilus', 'ml_queue')
            provider: Callback that returns current state
        """
        with self._lock:
            self._state_providers[name] = provider
    
    def register_alert_callback(
        self,
        callback: Callable[[CriticalAlert], None]
    ) -> None:
        """Register callback for critical alerts."""
        self._alert_callbacks.append(callback)
        if self._log_aggregator:
            self._log_aggregator.register_alert_callback(callback)
    
    def start(self) -> None:
        """Start UI update loop."""
        if self._running:
            return
        
        with self._lock:
            self._running = True
            
            # Start dashboard
            if self._dashboard:
                self._dashboard.start()
            
            # Start log aggregator async loop
            if self._log_aggregator:
                self._log_aggregator.start_async_loop()
            
            # Start update thread
            self._update_thread = threading.Thread(
                target=self._update_loop,
                daemon=True
            )
            self._update_thread.start()
            
            logger.info("UI Manager started")
    
    def stop(self) -> None:
        """Stop UI update loop."""
        self._running = False
        
        if self._update_thread:
            self._update_thread.join(timeout=5)
        
        with self._lock:
            if self._dashboard:
                self._dashboard.stop()
            
            if self._log_aggregator:
                self._log_aggregator.stop_async_loop()
        
        logger.info("UI Manager stopped")
    
    def _update_loop(self) -> None:
        """Main update loop - collects state and pushes to dashboard."""
        while self._running:
            try:
                self._collect_and_update()
            except Exception as e:
                logger.error(f"UI update error: {e}")
            
            time.sleep(self.config.update_interval_seconds)
    
    def _collect_and_update(self) -> None:
        """Collect state from providers and update dashboard."""
        if not self._dashboard:
            return
        
        with self._lock:
            # Collect cluster health
            if 'cluster' in self._state_providers:
                try:
                    cluster_data = self._state_providers['cluster']()
                    if isinstance(cluster_data, ClusterHealth):
                        self._dashboard.update_cluster_health(cluster_data)
                    elif isinstance(cluster_data, dict):
                        self._dashboard.update_cluster_health(ClusterHealth(**cluster_data))
                except Exception as e:
                    logger.debug(f"Failed to collect cluster state: {e}")
            
            # Collect Nautilus state
            if 'nautilus' in self._state_providers:
                try:
                    nautilus_data = self._state_providers['nautilus']()
                    if isinstance(nautilus_data, NautilusState):
                        self._dashboard.update_nautilus_state(nautilus_data)
                    elif isinstance(nautilus_data, dict):
                        self._dashboard.update_nautilus_state(NautilusState(**nautilus_data))
                except Exception as e:
                    logger.debug(f"Failed to collect Nautilus state: {e}")
            
            # Collect ML queue state
            if 'ml_queue' in self._state_providers:
                try:
                    ml_data = self._state_providers['ml_queue']()
                    if isinstance(ml_data, MLInferenceQueue):
                        self._dashboard.update_ml_queue(ml_data)
                    elif isinstance(ml_data, dict):
                        self._dashboard.update_ml_queue(MLInferenceQueue(**ml_data))
                except Exception as e:
                    logger.debug(f"Failed to collect ML queue state: {e}")
            
            # Collect custom metrics
            if 'custom_metrics' in self._state_providers:
                try:
                    metrics = self._state_providers['custom_metrics']()
                    if isinstance(metrics, dict):
                        for name, value in metrics.items():
                            self._dashboard.update_custom_metric(name, value)
                except Exception as e:
                    logger.debug(f"Failed to collect custom metrics: {e}")
    
    def report_exception(
        self,
        exc_type: type,
        exc_value: Exception,
        exc_tb: Optional[Any] = None
    ) -> Optional[CriticalAlert]:
        """Report an exception to the log aggregator."""
        if self._log_aggregator:
            return self._log_aggregator.report_exception(exc_type, exc_value, exc_tb)
        return None
    
    def report_oom(
        self,
        memory_used_mb: float,
        memory_limit_mb: float
    ) -> Optional[CriticalAlert]:
        """Report OOM condition."""
        if self._log_aggregator:
            return self._log_aggregator.report_oom(memory_used_mb, memory_limit_mb)
        return None
    
    def report_worker_death(
        self,
        worker_id: str,
        exit_code: int
    ) -> Optional[CriticalAlert]:
        """Report worker death."""
        if self._log_aggregator:
            return self._log_aggregator.report_worker_death(worker_id, exit_code)
        return None
    
    def get_status(self) -> Dict[str, Any]:
        """Get UI manager status."""
        with self._lock:
            status = {
                'running': self._running,
                'dashboard_enabled': self._dashboard is not None,
                'log_aggregation_enabled': self._log_aggregator is not None,
                'state_providers': list(self._state_providers.keys()),
                'alert_callbacks': len(self._alert_callbacks)
            }
            
            if self._log_aggregator:
                status['log_aggregator_stats'] = self._log_aggregator.get_statistics()
            
            return status
    
    def shutdown(self) -> None:
        """Shutdown UI manager completely."""
        self.stop()
        
        with self._lock:
            if self._log_aggregator:
                self._log_aggregator.shutdown()
        
        logger.info("UI Manager shut down complete")


# Global singleton instance
_ui_manager_instance: Optional[UIManager] = None
_ui_manager_lock = threading.Lock()


def get_ui_manager(config: Optional[UIConfig] = None) -> UIManager:
    """Thread-safe singleton access to UI manager."""
    global _ui_manager_instance
    
    with _ui_manager_lock:
        if _ui_manager_instance is None:
            _ui_manager_instance = UIManager(config)
        
        return _ui_manager_instance


if __name__ == "__main__":
    # Demo usage
    import numpy as np
    
    config = UIConfig(
        update_interval_seconds=0.5
    )
    
    manager = get_ui_manager(config)
    
    print("=== UI Manager Demo ===\n")
    
    # Register state providers
    def get_cluster_state():
        return ClusterHealth(
            active_workers=4,
            total_workers=4,
            cpu_utilization=45 + np.random.randn() * 5,
            memory_utilization=60 + np.random.randn() * 3,
            object_store_usage=30,
            tasks_pending=np.random.randint(0, 10),
            tasks_running=np.random.randint(5, 20),
            status='healthy'
        )
    
    def get_nautilus_state():
        return NautilusState(
            position=np.random.randn() * 10,
            unrealized_pnl=np.random.randn() * 1000,
            realized_pnl=5000 + np.random.randn() * 500,
            orders_pending=np.random.randint(0, 5),
            orders_filled_today=np.random.randint(50, 200),
            risk_limit_remaining=50000 - np.random.randint(0, 5000),
            status='running'
        )
    
    def get_ml_queue_state():
        return MLInferenceQueue(
            queue_depth=np.random.randint(0, 20),
            avg_latency_us=50 + np.random.exponential(10),
            p99_latency_us=200 + np.random.exponential(50),
            requests_per_second=100 + np.random.randn() * 20,
            model_version="v1.2.3",
            gpu_utilization=70 + np.random.randn() * 5
        )
    
    manager.register_state_provider('cluster', get_cluster_state)
    manager.register_state_provider('nautilus', get_nautilus_state)
    manager.register_state_provider('ml_queue', get_ml_queue_state)
    
    # Register alert callback
    def on_alert(alert: CriticalAlert):
        print(f"ALERT: [{alert.severity}] {alert.alert_type} - {alert.message[:80]}")
    
    manager.register_alert_callback(on_alert)
    
    # Start manager
    manager.start()
    
    # Run for a few seconds
    print("Running UI manager for 5 seconds...")
    time.sleep(5)
    
    # Simulate an alert
    print("\nSimulating alert...")
    manager.report_oom(memory_used_mb=2800, memory_limit_mb=3000)
    
    # Show status
    status = manager.get_status()
    print(f"\nStatus: {status}")
    
    # Shutdown
    manager.shutdown()
    print("\nDemo complete")
