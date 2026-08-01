"""
Log Aggregator for Python Backend.
Async log aggregator catching Python exceptions, OOM warnings, and Ray worker deaths.
Formats critical alerts transmitted instantly to Rust Global Kill Switch.

Prevents catastrophic failures through early detection and rapid alerting.
"""

import asyncio
import logging
import threading
import time
import json
import re
import sys
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from collections import deque
import traceback

try:
    import zmq
    ZMQ_AVAILABLE = True
except ImportError:
    ZMQ_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CriticalAlert:
    """Critical alert for kill switch."""
    alert_id: str
    timestamp: float
    alert_type: str  # 'exception', 'oom', 'worker_death', 'threshold_breach'
    severity: str  # 'critical', 'warning', 'info'
    message: str
    source: str
    stack_trace: Optional[str]
    metadata: Dict[str, Any]
    action_required: bool


class LogAggregator:
    """
    Async log aggregator for monitoring system health.
    Detects critical conditions and alerts Rust kill switch.
    """
    
    def __init__(
        self,
        kill_switch_endpoint: str = "tcp://localhost:5556",
        max_alerts_buffer: int = 1000,
        alert_cooldown_seconds: float = 5.0
    ):
        self.kill_switch_endpoint = kill_switch_endpoint
        self.max_alerts_buffer = max_alerts_buffer
        self.alert_cooldown = alert_cooldown_seconds
        
        self._lock = threading.RLock()
        
        # Alert buffer
        self._alerts: deque = deque(maxlen=max_alerts_buffer)
        self._alert_counts: Dict[str, int] = {}
        self._last_alert_time: Dict[str, float] = {}
        
        # ZMQ socket for kill switch
        self._zmq_context: Optional[Any] = None
        self._zmq_socket: Optional[Any] = None
        
        # Custom log handler
        self._log_handler: Optional[logging.Handler] = None
        
        # Control
        self._running = False
        self._async_loop: Optional[asyncio.AbstractEventLoop] = None
        self._async_thread: Optional[threading.Thread] = None
        
        # Callbacks
        self._on_critical_alert_callbacks: List[Callable[[CriticalAlert], None]] = []
        
        # Pattern matchers for critical conditions
        self._critical_patterns = [
            (r'OutOfMemoryError', 'oom'),
            (r'MemoryError', 'oom'),
            (r'ray\.worker\.DiedException', 'worker_death'),
            (r'Worker crashed|died unexpectedly', 'worker_death'),
            (r'Killed|SIGKILL|SIGSEGV', 'worker_death'),
            (r'deadline exceeded|timeout', 'timeout'),
            (r'Connection refused|broken pipe', 'connection_error'),
        ]
        
        if ZMQ_AVAILABLE:
            self._setup_zmq()
        
        self._setup_log_handler()
    
    def _setup_zmq(self) -> None:
        """Setup ZMQ publisher for kill switch alerts."""
        try:
            self._zmq_context = zmq.Context()
            self._zmq_socket = self._zmq_context.socket(zmq.PUB)
            self._zmq_socket.bind(self.kill_switch_endpoint)
            logger.info(f"Kill switch ZMQ bound to {self.kill_switch_endpoint}")
        except Exception as e:
            logger.warning(f"Failed to setup ZMQ for kill switch: {e}")
            self._zmq_socket = None
    
    def _setup_log_handler(self) -> None:
        """Setup custom log handler to intercept logs."""
        class InterceptHandler(logging.Handler):
            def __init__(self, aggregator):
                super().__init__()
                self.aggregator = aggregator
            
            def emit(self, record):
                try:
                    msg = self.format(record)
                    self.aggregator._process_log_message(
                        level=record.levelno,
                        message=msg,
                        source=record.name,
                        exc_info=record.exc_info
                    )
                except Exception:
                    self.handleError(record)
        
        self._log_handler = InterceptHandler(self)
        self._log_handler.setLevel(logging.WARNING)
        
        # Add to root logger
        logging.getLogger().addHandler(self._log_handler)
    
    def register_alert_callback(self, callback: Callable[[CriticalAlert], None]) -> None:
        """Register callback for critical alerts."""
        self._on_critical_alert_callbacks.append(callback)
    
    def _process_log_message(
        self,
        level: int,
        message: str,
        source: str,
        exc_info: Optional[tuple] = None
    ) -> None:
        """Process incoming log message and check for critical conditions."""
        # Check for critical patterns
        for pattern, alert_type in self._critical_patterns:
            if re.search(pattern, message, re.IGNORECASE):
                self._create_alert(
                    alert_type=alert_type,
                    severity='critical' if alert_type in ['oom', 'worker_death'] else 'warning',
                    message=message,
                    source=source,
                    stack_trace=traceback.format_exception(*exc_info) if exc_info else None,
                    metadata={'pattern': pattern, 'log_level': level}
                )
                break
        
        # Check for OOM-like memory warnings
        if 'memory' in message.lower() and ('exhausted' in message.lower() or 'limit' in message.lower()):
            self._create_alert(
                alert_type='oom_warning',
                severity='warning',
                message=message,
                source=source,
                metadata={'log_level': level}
            )
    
    def _create_alert(
        self,
        alert_type: str,
        severity: str,
        message: str,
        source: str,
        stack_trace: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[CriticalAlert]:
        """Create and dispatch a critical alert."""
        import uuid
        
        # Check cooldown
        with self._lock:
            cooldown_key = f"{alert_type}:{source}"
            now = time.time()
            
            if cooldown_key in self._last_alert_time:
                if now - self._last_alert_time[cooldown_key] < self.alert_cooldown:
                    return None  # Suppress due to cooldown
            
            self._last_alert_time[cooldown_key] = now
            
            # Update counts
            self._alert_counts[alert_type] = self._alert_counts.get(alert_type, 0) + 1
        
        alert = CriticalAlert(
            alert_id=f"alert_{uuid.uuid4().hex[:8]}",
            timestamp=now,
            alert_type=alert_type,
            severity=severity,
            message=message[:500],  # Truncate long messages
            source=source,
            stack_trace=stack_trace,
            metadata=metadata or {},
            action_required=severity == 'critical'
        )
        
        with self._lock:
            self._alerts.append(alert)
        
        # Dispatch alert
        self._dispatch_alert(alert)
        
        return alert
    
    def _dispatch_alert(self, alert: CriticalAlert) -> None:
        """Dispatch alert to kill switch and callbacks."""
        # Send to kill switch via ZMQ
        if self._zmq_socket:
            try:
                message = json.dumps(asdict(alert))
                self._zmq_socket.send_string(message)
                logger.info(f"Alert sent to kill switch: {alert.alert_type} ({alert.severity})")
            except Exception as e:
                logger.error(f"Failed to send alert to kill switch: {e}")
        
        # Notify callbacks
        for callback in self._on_critical_alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
        
        # Log critical alerts
        if alert.severity == 'critical':
            logger.critical(f"CRITICAL ALERT [{alert.alert_type}]: {alert.message}")
    
    def report_exception(
        self,
        exc_type: type,
        exc_value: Exception,
        exc_tb: Optional[Any] = None
    ) -> Optional[CriticalAlert]:
        """Report an uncaught exception."""
        stack_trace = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
        
        return self._create_alert(
            alert_type='exception',
            severity='critical' if issubclass(exc_type, (MemoryError, OSError)) else 'warning',
            message=f"{exc_type.__name__}: {str(exc_value)}",
            source='exception_handler',
            stack_trace=stack_trace,
            metadata={'exception_type': exc_type.__name__}
        )
    
    def report_oom(self, memory_used_mb: float, memory_limit_mb: float) -> Optional[CriticalAlert]:
        """Report out-of-memory condition."""
        return self._create_alert(
            alert_type='oom',
            severity='critical',
            message=f"OOM: Using {memory_used_mb:.0f}MB of {memory_limit_mb:.0f}MB limit",
            source='memory_monitor',
            metadata={
                'memory_used_mb': memory_used_mb,
                'memory_limit_mb': memory_limit_mb,
                'usage_percent': (memory_used_mb / memory_limit_mb) * 100
            }
        )
    
    def report_worker_death(self, worker_id: str, exit_code: int) -> Optional[CriticalAlert]:
        """Report Ray worker death."""
        return self._create_alert(
            alert_type='worker_death',
            severity='critical',
            message=f"Worker {worker_id} died with exit code {exit_code}",
            source='ray_cluster',
            metadata={
                'worker_id': worker_id,
                'exit_code': exit_code
            }
        )
    
    def report_threshold_breach(
        self,
        metric_name: str,
        current_value: float,
        threshold: float
    ) -> Optional[CriticalAlert]:
        """Report metric threshold breach."""
        return self._create_alert(
            alert_type='threshold_breach',
            severity='critical' if abs(current_value - threshold) / threshold > 0.5 else 'warning',
            message=f"{metric_name} breached threshold: {current_value:.4f} > {threshold:.4f}",
            source='threshold_monitor',
            metadata={
                'metric_name': metric_name,
                'current_value': current_value,
                'threshold': threshold
            }
        )
    
    def start_async_loop(self) -> None:
        """Start async processing loop."""
        if self._running:
            return
        
        self._running = True
        
        def async_thread_func():
            self._async_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._async_loop)
            
            async def process_loop():
                while self._running:
                    await asyncio.sleep(0.1)  # Keep loop alive
                    # Additional async processing can be added here
            
            try:
                self._async_loop.run_until_complete(process_loop())
            except Exception as e:
                logger.error(f"Async loop error: {e}")
            finally:
                self._async_loop.close()
        
        self._async_thread = threading.Thread(target=async_thread_func, daemon=True)
        self._async_thread.start()
        logger.info("Log Aggregator async loop started")
    
    def stop_async_loop(self) -> None:
        """Stop async processing loop."""
        self._running = False
        if self._async_thread:
            self._async_thread.join(timeout=5)
        logger.info("Log Aggregator async loop stopped")
    
    def get_recent_alerts(
        self,
        count: int = 10,
        alert_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get recent alerts."""
        with self._lock:
            alerts = list(self._alerts)[-count:]
            
            if alert_type:
                alerts = [a for a in alerts if a.alert_type == alert_type]
            
            return [asdict(a) for a in alerts]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get aggregator statistics."""
        with self._lock:
            return {
                'total_alerts': sum(self._alert_counts.values()),
                'alerts_by_type': self._alert_counts.copy(),
                'buffer_size': len(self._alerts),
                'max_buffer_size': self.max_alerts_buffer,
                'kill_switch_connected': self._zmq_socket is not None
            }
    
    def shutdown(self) -> None:
        """Shutdown aggregator."""
        self.stop_async_loop()
        
        # Remove log handler
        if self._log_handler:
            logging.getLogger().removeHandler(self._log_handler)
        
        # Close ZMQ
        if self._zmq_socket:
            self._zmq_socket.close()
        if self._zmq_context:
            self._zmq_context.term()
        
        logger.info("Log Aggregator shut down")


# Global singleton instance
_aggregator_instance: Optional[LogAggregator] = None
_aggregator_lock = threading.Lock()


def get_log_aggregator(
    kill_switch_endpoint: str = "tcp://localhost:5556"
) -> LogAggregator:
    """Thread-safe singleton access to log aggregator."""
    global _aggregator_instance
    
    with _aggregator_lock:
        if _aggregator_instance is None:
            _aggregator_instance = LogAggregator(kill_switch_endpoint)
        
        return _aggregator_instance


# Set up global exception handler
def _global_exception_handler(exc_type, exc_value, exc_tb):
    """Global exception handler that reports to aggregator."""
    if issubclass(exc_type, KeyboardInterrupt):
        return
    
    aggregator = get_log_aggregator()
    aggregator.report_exception(exc_type, exc_value, exc_tb)


sys.excepthook = _global_exception_handler


if __name__ == "__main__":
    # Demo usage
    aggregator = get_log_aggregator()
    aggregator.start_async_loop()
    
    print("=== Log Aggregator Demo ===\n")
    
    # Register callback
    def on_alert(alert: CriticalAlert):
        print(f"ALERT: [{alert.severity}] {alert.alert_type} - {alert.message[:100]}")
    
    aggregator.register_alert_callback(on_alert)
    
    # Simulate various alerts
    print("Simulating alerts...")
    
    # OOM alert
    aggregator.report_oom(memory_used_mb=2800, memory_limit_mb=3000)
    
    # Worker death
    aggregator.report_worker_death(worker_id="worker_123", exit_code=-9)
    
    # Threshold breach
    aggregator.report_threshold_breach(
        metric_name="drawdown",
        current_value=0.18,
        threshold=0.15
    )
    
    # Log message with critical pattern
    logging.error("Ray worker DiedException detected in cluster")
    
    # Show statistics
    stats = aggregator.get_statistics()
    print(f"\nStatistics: {stats}")
    
    # Get recent alerts
    alerts = aggregator.get_recent_alerts(count=5)
    print(f"\nRecent alerts: {len(alerts)}")
    
    aggregator.shutdown()
    print("\nDemo complete")
