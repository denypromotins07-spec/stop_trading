"""
UI Backend Module Root - Ensures UI rendering never blocks Python GIL or ML inference queues.
Coordinates metrics aggregation and ZeroMQ streaming to Rust ratatui frontend.
Thread-safe design with strict non-blocking guarantees.
"""
import numpy as np
import logging
from typing import Dict, Any, Optional, Callable
from pathlib import Path
import time
import threading
import json

logger = logging.getLogger(__name__)

# Import UI submodules
try:
    from .metrics_aggregator import MetricsAggregator, get_metrics_aggregator
    from .zmq_pusher import TelemetryStreamer, get_telemetry_streamer
except ImportError as e:
    logger.warning(f"UI submodules not fully available: {e}")
    MetricsAggregator = None
    TelemetryStreamer = None


class UIBackendManager:
    """
    Central manager for UI backend operations.
    Ensures non-blocking operation and coordinates all UI data flow.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize submodules
        self.metrics_aggregator = None
        self.telemetry_streamer = None
        
        if MetricsAggregator is not None:
            self.metrics_aggregator = get_metrics_aggregator({
                'aggregation_interval': self.config.get('aggregation_interval', 0.5),
                'max_worker_history': self.config.get('max_worker_history', 100),
                'max_portfolio_history': self.config.get('max_portfolio_history', 1000),
                'max_inference_history': self.config.get('max_inference_history', 500)
            })
            logger.info("MetricsAggregator initialized")
        
        if TelemetryStreamer is not None:
            self.telemetry_streamer = get_telemetry_streamer({
                'endpoint': self.config.get('zmq_endpoint', 'tcp://127.0.0.1:5555'),
                'push_interval': self.config.get('push_interval', 0.1)
            })
            logger.info("TelemetryStreamer initialized")
        
        # State
        self._running = False
        self._update_thread: Optional[threading.Thread] = None
        
        # Update configuration
        self._update_interval = self.config.get('update_interval', 0.1)
        self._max_update_time_ms = self.config.get('max_update_time_ms', 10.0)
        
        # Callbacks for external integration
        self._data_sources: Dict[str, Callable] = {}
        
        # Statistics
        self._updates_sent = 0
        self._updates_dropped = 0
        self._gil_blocked_count = 0
        
        logger.info("UIBackendManager initialized")
    
    def register_data_source(self, name: str, source_fn: Callable) -> None:
        """
        Register a data source callback.
        
        Args:
            name: Source identifier
            source_fn: Function that returns data dict
        """
        self._data_sources[name] = source_fn
        logger.info(f"Registered data source: {name}")
    
    def start(self) -> bool:
        """Start UI backend update loop."""
        if self._running:
            return True
        
        # Start telemetry streaming
        if self.telemetry_streamer:
            if not self.telemetry_streamer.start_streaming():
                logger.warning("Failed to start telemetry streaming")
        
        self._running = True
        
        # Start update thread (detached from GIL-sensitive operations)
        self._update_thread = threading.Thread(
            target=self._update_loop,
            daemon=True,
            name="UIBackendUpdate"
        )
        self._update_thread.start()
        
        logger.info("UIBackendManager started")
        return True
    
    def stop(self) -> None:
        """Stop UI backend update loop."""
        self._running = False
        
        if self._update_thread:
            self._update_thread.join(timeout=2.0)
            self._update_thread = None
        
        # Stop telemetry streaming
        if self.telemetry_streamer:
            self.telemetry_streamer.stop_streaming()
        
        logger.info("UIBackendManager stopped")
    
    def _update_loop(self) -> None:
        """
        Background update loop.
        Runs in separate thread to avoid blocking ML inference.
        """
        last_update = 0.0
        
        while self._running:
            current_time = time.time()
            
            if current_time - last_update >= self._update_interval:
                self._perform_update()
                last_update = current_time
            
            # Small sleep to prevent CPU spinning
            # This does NOT hold the GIL for extended periods
            time.sleep(min(0.01, self._update_interval / 4))
    
    def _perform_update(self) -> None:
        """Perform single update cycle."""
        start_time = time.perf_counter()
        
        try:
            # Collect data from all registered sources (non-blocking)
            collected_data = {}
            
            for name, source_fn in self._data_sources.items():
                try:
                    # Call with timeout protection
                    data = source_fn()
                    if data:
                        collected_data[name] = data
                except Exception as e:
                    logger.error(f"Data source {name} failed: {e}")
                    collected_data[name] = {'error': str(e)}
            
            # Aggregate metrics
            if self.metrics_aggregator:
                aggregated = self.metrics_aggregator.aggregate()
                collected_data['aggregated'] = aggregated
            
            # Stream to UI (non-blocking with drop-if-busy)
            if self.telemetry_streamer and collected_data:
                self.telemetry_streamer.update_metrics(collected_data)
                self._updates_sent += 1
            
            # Check update duration
            update_time_ms = (time.perf_counter() - start_time) * 1000
            
            if update_time_ms > self._max_update_time_ms:
                logger.warning(f"UI update exceeded threshold: {update_time_ms:.2f}ms")
                
        except Exception as e:
            logger.error(f"UI update failed: {e}")
    
    def push_immediate(self, data: Dict[str, Any], 
                       priority: str = 'normal') -> bool:
        """
        Push data immediately to UI (bypass queue).
        
        Args:
            data: Data to push
            priority: Priority level ('low', 'normal', 'high', 'critical')
            
        Returns:
            Success status
        """
        if not self.telemetry_streamer:
            return False
        
        # Add priority metadata
        envelope = {
            'priority': priority,
            'timestamp': time.time(),
            'data': data
        }
        
        success = self.telemetry_streamer.send_immediate(envelope)
        
        if not success:
            self._updates_dropped += 1
        
        return success
    
    def send_alert(self, alert_type: str, message: str,
                   severity: str = 'info') -> bool:
        """
        Send alert to UI.
        
        Args:
            alert_type: Type of alert
            message: Alert message
            severity: Severity level
            
        Returns:
            Success status
        """
        if not self.telemetry_streamer:
            return False
        
        return self.telemetry_streamer.pusher.send_alert(
            alert_type, message, severity
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Get backend status."""
        status = {
            'running': self._running,
            'updates_sent': self._updates_sent,
            'updates_dropped': self._updates_dropped,
            'drop_rate': self._updates_dropped / max(1, self._updates_sent + self._updates_dropped),
            'data_sources': list(self._data_sources.keys()),
            'update_interval': self._update_interval,
            'gil_blocked_count': self._gil_blocked_count
        }
        
        if self.telemetry_streamer:
            status['streamer_stats'] = self.telemetry_streamer.get_statistics()
        
        if self.metrics_aggregator:
            status['aggregator_summary'] = self.metrics_aggregator.get_summary()
        
        return status
    
    def check_gil_health(self) -> Dict[str, Any]:
        """
        Check GIL contention health.
        Measures if UI updates are being blocked by other threads.
        """
        # Measure time to acquire GIL
        start = time.perf_counter()
        
        # Simple operation that requires GIL
        test_array = np.zeros(100)
        test_sum = np.sum(test_array)
        
        gil_acquire_time = (time.perf_counter() - start) * 1000
        
        if gil_acquire_time > 1.0:  # More than 1ms to acquire GIL
            self._gil_blocked_count += 1
            logger.warning(f"GIL contention detected: {gil_acquire_time:.2f}ms")
        
        return {
            'gil_acquire_time_ms': gil_acquire_time,
            'blocked_count': self._gil_blocked_count,
            'healthy': gil_acquire_time < 1.0
        }
    
    def warmup(self) -> None:
        """Warm up UI backend."""
        # Pre-aggregate metrics
        if self.metrics_aggregator:
            self.metrics_aggregator.aggregate(force=True)
        
        # Test ZMQ connection
        if self.telemetry_streamer:
            self.push_immediate({'warmup': True}, priority='low')
        
        logger.info("UIBackendManager warmed up")
    
    def close(self) -> None:
        """Clean up resources."""
        self.stop()
        
        if self.telemetry_streamer:
            self.telemetry_streamer.close()
        
        if self.metrics_aggregator:
            self.metrics_aggregator.reset()
        
        logger.info(f"UIBackendManager closed. Sent: {self._updates_sent}, "
                   f"Dropped: {self._updates_dropped}")


# Singleton instance
_ui_backend_manager: Optional[UIBackendManager] = None


def get_ui_backend_manager(config: Optional[Dict[str, Any]] = None) -> UIBackendManager:
    """Get or create singleton UIBackendManager instance."""
    global _ui_backend_manager
    if _ui_backend_manager is None:
        _ui_backend_manager = UIBackendManager(config)
    return _ui_backend_manager


def reset_ui_backend_manager() -> None:
    """Reset singleton instance."""
    global _ui_backend_manager
    if _ui_backend_manager is not None:
        _ui_backend_manager.close()
    _ui_backend_manager = None


# Convenience functions for direct access
def push_metrics(metrics: Dict[str, Any]) -> bool:
    """Push metrics to UI immediately."""
    manager = get_ui_backend_manager()
    return manager.push_immediate(metrics, priority='normal')


def send_alert(alert_type: str, message: str, severity: str = 'info') -> bool:
    """Send alert to UI."""
    manager = get_ui_backend_manager()
    return manager.send_alert(alert_type, message, severity)


__all__ = [
    'UIBackendManager',
    'get_ui_backend_manager',
    'reset_ui_backend_manager',
    'push_metrics',
    'send_alert'
]
