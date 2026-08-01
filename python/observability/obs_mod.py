"""
Observability Module Root - Aggregates Python-side telemetry.
Syncs with Rust eBPF observability stack for unified monitoring.
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from dataclasses import dataclass, field
import threading
import time
import json
import socket

from .prometheus_exporter import PrometheusExporter, MetricsSnapshot, get_prometheus_exporter
from .drift_dashboard import DriftDetector, DriftReport, get_drift_detector


@dataclass
class TelemetryEvent:
    """Single telemetry event."""
    
    event_type: str
    component: str
    timestamp: float
    value: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "component": self.component,
            "timestamp": self.timestamp,
            "value": self.value,
            "metadata": self.metadata
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())


@dataclass
class SystemHealth:
    """Overall system health status."""
    
    # Component health (0-1 scale)
    nlp_health: float = 1.0
    macro_health: float = 1.0
    onchain_health: float = 1.0
    backtest_health: float = 1.0
    observability_health: float = 1.0
    
    # Resource usage
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    disk_usage_percent: float = 0.0
    
    # Latency metrics
    avg_inference_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    
    # Error rates
    error_rate_per_minute: float = 0.0
    
    # Overall status
    overall_health: float = 1.0
    status: str = "HEALTHY"  # HEALTHY, DEGRADED, CRITICAL
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "nlp_health": self.nlp_health,
            "macro_health": self.macro_health,
            "onchain_health": self.onchain_health,
            "backtest_health": self.backtest_health,
            "observability_health": self.observability_health,
            "cpu_usage_percent": self.cpu_usage_percent,
            "memory_usage_mb": self.memory_usage_mb,
            "disk_usage_percent": self.disk_usage_percent,
            "avg_inference_latency_ms": self.avg_inference_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
            "error_rate_per_minute": self.error_rate_per_minute,
            "overall_health": self.overall_health,
            "status": self.status
        }


class ObservabilityManager:
    """
    Central observability management system.
    Aggregates telemetry and syncs with Rust eBPF stack.
    """
    
    def __init__(
        self,
        prometheus_port: int = 8000,
        rust_ipc_address: Optional[str] = None,
        health_check_interval: float = 5.0
    ):
        # Sub-modules
        self.prometheus_exporter = PrometheusExporter()
        self.drift_detector = DriftDetector()
        
        # Configuration
        self.prometheus_port = prometheus_port
        self.rust_ipc_address = rust_ipc_address
        self.health_check_interval = health_check_interval
        
        # Event storage
        self._events: List[TelemetryEvent] = []
        self._events_max = 10000
        
        # Health tracking
        self._component_health: Dict[str, float] = {
            "nlp": 1.0,
            "macro": 1.0,
            "onchain": 1.0,
            "backtest": 1.0,
            "observability": 1.0
        }
        self._error_counts: Dict[str, int] = {}
        self._last_health_check = 0.0
        
        # Rust IPC socket
        self._rust_socket: Optional[socket.socket] = None
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Background thread
        self._running = False
        self._health_thread: Optional[threading.Thread] = None
    
    def start(self) -> None:
        """Start observability manager and background tasks."""
        self._running = True
        self._health_thread = threading.Thread(
            target=self._health_check_loop,
            daemon=True
        )
        self._health_thread.start()
    
    def stop(self) -> None:
        """Stop observability manager."""
        self._running = False
        if self._health_thread:
            self._health_thread.join(timeout=5.0)
        if self._rust_socket:
            self._rust_socket.close()
    
    def record_event(
        self,
        event_type: str,
        component: str,
        value: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record a telemetry event."""
        with self._lock:
            event = TelemetryEvent(
                event_type=event_type,
                component=component,
                timestamp=time.time(),
                value=value,
                metadata=metadata or {}
            )
            
            self._events.append(event)
            
            # Trim events
            while len(self._events) > self._events_max:
                self._events.pop(0)
            
            # Update Prometheus
            if event_type == "inference":
                self.prometheus_exporter.record_inference(
                    model_id=metadata.get("model_id", "unknown"),
                    model_type=metadata.get("model_type", "unknown"),
                    latency_seconds=value / 1000,  # Convert ms to seconds
                    success=metadata.get("success", True)
                )
            elif event_type == "error":
                self._record_error(component, metadata)
    
    def _record_error(
        self,
        component: str,
        metadata: Optional[Dict[str, Any]]
    ) -> None:
        """Record an error event."""
        key = f"{component}_{metadata.get('error_type', 'unknown')}"
        self._error_counts[key] = self._error_counts.get(key, 0) + 1
        
        # Update component health
        self._component_health[component] = max(
            self._component_health.get(component, 1.0) - 0.1, 0.0
        )
    
    def update_component_health(
        self,
        component: str,
        health: float
    ) -> None:
        """Update health score for a component."""
        with self._lock:
            self._component_health[component] = np.clip(health, 0.0, 1.0)
    
    def get_system_health(self) -> SystemHealth:
        """Get current system health status."""
        with self._lock:
            health = SystemHealth()
            
            # Component health
            health.nlp_health = self._component_health.get("nlp", 1.0)
            health.macro_health = self._component_health.get("macro", 1.0)
            health.onchain_health = self._component_health.get("onchain", 1.0)
            health.backtest_health = self._component_health.get("backtest", 1.0)
            health.observability_health = self._component_health.get("observability", 1.0)
            
            # Get Prometheus snapshot
            snapshot = self.prometheus_exporter._get_snapshot()
            health.avg_inference_latency_ms = snapshot.avg_inference_latency_ms
            health.p99_latency_ms = snapshot.p99_inference_latency_ms
            health.memory_usage_mb = snapshot.python_memory_mb
            
            # Calculate error rate
            total_errors = sum(self._error_counts.values())
            elapsed_minutes = max((time.time() - self._last_health_check) / 60, 1)
            health.error_rate_per_minute = total_errors / elapsed_minutes
            
            # Calculate overall health
            component_avg = np.mean([
                health.nlp_health,
                health.macro_health,
                health.onchain_health,
                health.backtest_health,
                health.observability_health
            ])
            
            # Penalize for high error rate
            error_penalty = min(health.error_rate_per_minute * 0.05, 0.5)
            health.overall_health = max(component_avg - error_penalty, 0.0)
            
            # Determine status
            if health.overall_health >= 0.8:
                health.status = "HEALTHY"
            elif health.overall_health >= 0.5:
                health.status = "DEGRADED"
            else:
                health.status = "CRITICAL"
            
            self._last_health_check = time.time()
            
            return health
    
    def check_data_drift(
        self,
        current_features: np.ndarray,
        window_size: int = 500
    ) -> Optional[DriftReport]:
        """Check for data drift in feature distributions."""
        try:
            report = self.drift_detector.detect_drift(
                current_features, window_size=window_size
            )
            
            # Update health based on drift
            if report.retrain_recommended:
                self.update_component_health("ml_models", 0.7)
            
            return report
        except Exception:
            return None
    
    def sync_to_rust(self, data: Dict[str, Any]) -> bool:
        """Sync telemetry data to Rust eBPF stack via IPC."""
        if not self.rust_ipc_address:
            return False
        
        try:
            if self._rust_socket is None:
                self._rust_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                self._rust_socket.connect(self.rust_ipc_address)
            
            message = json.dumps(data).encode('utf-8')
            self._rust_socket.sendall(message)
            return True
        except Exception:
            self._rust_socket = None
            return False
    
    def _health_check_loop(self) -> None:
        """Background health check loop."""
        while self._running:
            try:
                # Get health status
                health = self.get_system_health()
                
                # Update Prometheus
                self.prometheus_exporter.update_memory_usage(
                    "python", health.memory_usage_mb * 1024 * 1024
                )
                
                # Sync to Rust
                self.sync_to_rust({
                    "type": "health_update",
                    "data": health.to_dict()
                })
                
            except Exception:
                pass
            
            time.sleep(self.health_check_interval)
    
    def get_events(
        self,
        component: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get recent telemetry events."""
        with self._lock:
            events = self._events.copy()
            
            if component:
                events = [e for e in events if e.component == component]
            
            if event_type:
                events = [e for e in events if e.event_type == event_type]
            
            return [e.to_dict() for e in events[-limit:]]
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        with self._lock:
            snapshot = self.prometheus_exporter._get_snapshot()
            health = self.get_system_health()
            
            return {
                "health": health.to_dict(),
                "metrics": snapshot.to_dict(),
                "error_counts": dict(self._error_counts),
                "component_health": dict(self._component_health),
                "drift_status": {
                    "should_retrain": self.drift_detector.should_retrain(),
                    "critical_features": self.drift_detector.get_critical_features()
                }
            }
    
    def reset(self) -> None:
        """Reset all observability state."""
        with self._lock:
            self._events.clear()
            self._error_counts.clear()
            self._component_health = {k: 1.0 for k in self._component_health}
            self.drift_detector.reset_baseline()


# Global singleton instance
_obs_instance: Optional[ObservabilityManager] = None
_instance_lock = threading.Lock()


def get_observability_manager() -> ObservabilityManager:
    """Get or create the global observability manager."""
    global _obs_instance
    if _obs_instance is None:
        with _instance_lock:
            if _obs_instance is None:
                _obs_instance = ObservabilityManager()
    return _obs_instance


if __name__ == "__main__":
    # Test observability manager
    print("Testing ObservabilityManager:")
    
    manager = ObservabilityManager()
    manager.start()
    
    # Simulate some events
    for i in range(50):
        manager.record_event(
            event_type="inference",
            component="nlp",
            value=1 + np.random.exponential(5),  # Latency in ms
            metadata={
                "model_id": f"model_{i % 3}",
                "model_type": "alpha",
                "success": np.random.random() > 0.05
            }
        )
    
    # Record some errors
    manager.record_event(
        event_type="error",
        component="macro",
        metadata={"error_type": "regime_classification_failed"}
    )
    
    # Update component health
    manager.update_component_health("onchain", 0.85)
    
    # Get health status
    health = manager.get_system_health()
    print(f"\nSystem Health: {health.status}")
    print(f"Overall Health Score: {health.overall_health:.2f}")
    print(f"NLP Health: {health.nlp_health:.2f}")
    print(f"Macro Health: {health.macro_health:.2f}")
    
    # Get metrics summary
    summary = manager.get_metrics_summary()
    print(f"\nMetrics Summary:")
    print(f"  Avg Latency: {summary['metrics']['avg_inference_latency_ms']:.2f}ms")
    print(f"  Error Counts: {summary['error_counts']}")
    
    # Get recent events
    events = manager.get_events(limit=5)
    print(f"\nRecent Events: {len(events)}")
    
    manager.stop()
