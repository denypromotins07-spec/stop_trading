"""
Diagnostics Module Root
Triggers self-healing routines and transmits fatal degradation alerts 
to the Rust Global Kill Switch.

Provides unified interface for all system diagnostics including GIL contention, 
memory leaks, and health monitoring.
"""

import threading
import time
import signal
import os
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json

from .gil_contention import (
    get_gil_monitor, get_starvation_detector, 
    start_diagnostics as start_gil_diagnostics,
    stop_diagnostics as stop_gil_diagnostics,
    get_diagnostics_report as get_gil_report
)
from .leak_detector import (
    get_health_monitor, start_monitoring as start_memory_monitoring,
    stop_monitoring as stop_memory_monitoring,
    check_restart_needed, get_memory_diagnostics
)


class HealthStatus(Enum):
    """System health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    FAILING = "failing"


@dataclass
class DiagnosticAlert:
    """Represents a diagnostic alert."""
    timestamp: float
    severity: str  # "info", "warning", "error", "critical"
    component: str
    message: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    action_required: bool = False


@dataclass
class SelfHealAction:
    """Represents a self-healing action."""
    action_type: str  # "restart_worker", "reduce_load", "disable_feature", "kill_switch"
    target: str
    reason: str
    priority: int  # 1-5, 5 being highest
    timeout_sec: float = 30.0


class DiagnosticsOrchestrator:
    """
    Central orchestrator for all diagnostic systems.
    Coordinates health checks, alerts, and self-healing actions.
    """
    
    def __init__(self,
                 health_check_interval_sec: float = 1.0,
                 alert_buffer_size: int = 100,
                 rust_ipc_channel: Optional[Any] = None):
        """
        Initialize diagnostics orchestrator.
        
        Args:
            health_check_interval_sec: Interval between health checks
            alert_buffer_size: Maximum alerts to buffer
            rust_ipc_channel: Optional IPC channel to Rust kill switch
        """
        self.health_check_interval = health_check_interval_sec
        self.alert_buffer_size = alert_buffer_size
        self.rust_ipc_channel = rust_ipc_channel
        
        # Alert buffer
        self._alerts: List[DiagnosticAlert] = []
        
        # Self-heal action queue
        self._heal_queue: List[SelfHealAction] = []
        
        # Monitoring state
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Callbacks
        self._alert_callbacks: List[Callable[[DiagnosticAlert], None]] = []
        self._heal_callbacks: List[Callable[[SelfHealAction], None]] = []
        
        # Statistics
        self._stats = {
            'health_checks': 0,
            'alerts_generated': 0,
            'heal_actions_triggered': 0,
            'kill_switch_activations': 0
        }
        
        # Component health tracking
        self._component_health: Dict[str, HealthStatus] = {
            'gil': HealthStatus.HEALTHY,
            'memory': HealthStatus.HEALTHY,
            'inference': HealthStatus.HEALTHY,
            'execution': HealthStatus.HEALTHY
        }
    
    def start(self):
        """Start diagnostics orchestration."""
        if self._running:
            return
        
        # Start underlying monitors
        start_gil_diagnostics()
        start_memory_monitoring()
        
        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="DiagnosticsOrchestrator"
        )
        self._monitor_thread.start()
    
    def stop(self):
        """Stop diagnostics orchestration."""
        self._running = False
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=5.0)
            self._monitor_thread = None
        
        stop_gil_diagnostics()
        stop_memory_monitoring()
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self._running:
            try:
                self._perform_health_check()
                self._process_heal_queue()
            except Exception as e:
                self._generate_alert(
                    severity="error",
                    component="orchestrator",
                    message=f"Health check failed: {str(e)}"
                )
            
            time.sleep(self.health_check_interval)
    
    def _perform_health_check(self):
        """Perform comprehensive health check."""
        with self._lock:
            self._stats['health_checks'] += 1
            
            # Check GIL contention
            gil_report = get_gil_monitor().get_diagnostics()
            if not gil_report.get('is_healthy', True):
                self._component_health['gil'] = HealthStatus.DEGRADED
                self._generate_alert(
                    severity="warning",
                    component="gil",
                    message="GIL contention detected",
                    metrics=gil_report,
                    action_required=True
                )
            else:
                self._component_health['gil'] = HealthStatus.HEALTHY
            
            # Check memory health
            mem_diag = get_memory_diagnostics()
            mem_state = mem_diag.get('state', 'healthy')
            
            if mem_state == 'critical':
                self._component_health['memory'] = HealthStatus.CRITICAL
                self._queue_heal_action(SelfHealAction(
                    action_type="restart_worker",
                    target="inference_worker",
                    reason="Critical memory usage",
                    priority=5
                ))
            elif mem_state == 'warning':
                self._component_health['memory'] = HealthStatus.DEGRADED
            else:
                self._component_health['memory'] = HealthStatus.HEALTHY
            
            # Check if restart needed
            restart_needed, restart_reason = check_restart_needed()
            if restart_needed:
                self._queue_heal_action(SelfHealAction(
                    action_type="restart_worker",
                    target="bloated_worker",
                    reason=restart_reason,
                    priority=4,
                    timeout_sec=60.0
                ))
    
    def _process_heal_queue(self):
        """Process pending self-heal actions."""
        with self._lock:
            if not self._heal_queue:
                return
            
            # Sort by priority
            self._heal_queue.sort(key=lambda x: x.priority, reverse=True)
            
            # Process highest priority action
            action = self._heal_queue.pop(0)
            self._execute_heal_action(action)
    
    def _execute_heal_action(self, action: SelfHealAction):
        """Execute a self-heal action."""
        self._stats['heal_actions_triggered'] += 1
        
        # Notify callbacks
        for callback in self._heal_callbacks:
            try:
                callback(action)
            except Exception:
                pass
        
        # Execute based on action type
        if action.action_type == "kill_switch":
            self._activate_kill_switch(action.reason)
        elif action.action_type == "restart_worker":
            self._trigger_worker_restart(action.target)
        elif action.action_type == "reduce_load":
            self._reduce_system_load()
        elif action.action_type == "disable_feature":
            self._disable_feature(action.target)
    
    def _activate_kill_switch(self, reason: str):
        """Activate the Rust global kill switch."""
        self._stats['kill_switch_activations'] += 1
        
        # Generate critical alert
        self._generate_alert(
            severity="critical",
            component="kill_switch",
            message=f"Global kill switch activated: {reason}",
            action_required=False
        )
        
        # Signal Rust via IPC if available
        if self.rust_ipc_channel is not None:
            try:
                message = {
                    'type': 'KILL_SWITCH',
                    'reason': reason,
                    'timestamp': time.time()
                }
                self.rust_ipc_channel.send(json.dumps(message))
            except Exception:
                pass
        
        # Also send SIGTERM to self for graceful shutdown
        os.kill(os.getpid(), signal.SIGTERM)
    
    def _trigger_worker_restart(self, target: str):
        """Trigger worker restart."""
        # Generate alert
        self._generate_alert(
            severity="warning",
            component="worker_restart",
            message=f"Restarting worker: {target}",
            action_required=False
        )
        
        # In production, this would signal Ray to restart the worker
        # For now, just log the action
    
    def _reduce_system_load(self):
        """Reduce system load by disabling non-critical features."""
        self._generate_alert(
            severity="info",
            component="load_reduction",
            message="Reducing system load",
            action_required=False
        )
    
    def _disable_feature(self, feature: str):
        """Disable a specific feature."""
        self._generate_alert(
            severity="info",
            component="feature_disable",
            message=f"Disabling feature: {feature}",
            action_required=False
        )
    
    def _generate_alert(self,
                        severity: str,
                        component: str,
                        message: str,
                        metrics: Optional[Dict] = None,
                        action_required: bool = False):
        """Generate a diagnostic alert."""
        with self._lock:
            alert = DiagnosticAlert(
                timestamp=time.time(),
                severity=severity,
                component=component,
                message=message,
                metrics=metrics or {},
                action_required=action_required
            )
            
            self._alerts.append(alert)
            self._stats['alerts_generated'] += 1
            
            # Trim buffer
            if len(self._alerts) > self.alert_buffer_size:
                self._alerts = self._alerts[-self.alert_buffer_size:]
            
            # Notify callbacks
            for callback in self._alert_callbacks:
                try:
                    callback(alert)
                except Exception:
                    pass
    
    def _queue_heal_action(self, action: SelfHealAction):
        """Queue a self-heal action."""
        with self._lock:
            self._heal_queue.append(action)
    
    def register_alert_callback(self, callback: Callable[[DiagnosticAlert], None]):
        """Register callback for alerts."""
        self._alert_callbacks.append(callback)
    
    def register_heal_callback(self, callback: Callable[[SelfHealAction], None]):
        """Register callback for heal actions."""
        self._heal_callbacks.append(callback)
    
    def get_overall_health(self) -> HealthStatus:
        """Get overall system health status."""
        with self._lock:
            statuses = list(self._component_health.values())
            
            if HealthStatus.FAILING in statuses:
                return HealthStatus.FAILING
            if HealthStatus.CRITICAL in statuses:
                return HealthStatus.CRITICAL
            if HealthStatus.DEGRADED in statuses:
                return HealthStatus.DEGRADED
            return HealthStatus.HEALTHY
    
    def get_component_health(self) -> Dict[str, str]:
        """Get per-component health status."""
        with self._lock:
            return {k: v.value for k, v in self._component_health.items()}
    
    def get_recent_alerts(self, count: int = 10) -> List[DiagnosticAlert]:
        """Get recent alerts."""
        with self._lock:
            return self._alerts[-count:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get diagnostics statistics."""
        with self._lock:
            stats = self._stats.copy()
            stats['overall_health'] = self.get_overall_health().value
            stats['component_health'] = self.get_component_health()
            stats['pending_heal_actions'] = len(self._heal_queue)
            return stats
    
    def force_kill_switch(self, reason: str):
        """Force activate kill switch."""
        self._activate_kill_switch(reason)


# Module-level singleton
_orchestrator: Optional[DiagnosticsOrchestrator] = None
_lock = threading.Lock()


def get_orchestrator(rust_ipc_channel: Optional[Any] = None) -> DiagnosticsOrchestrator:
    """Get or create global diagnostics orchestrator."""
    global _orchestrator
    
    with _lock:
        if _orchestrator is None:
            _orchestrator = DiagnosticsOrchestrator(rust_ipc_channel=rust_ipc_channel)
        return _orchestrator


def start_diagnostics(rust_ipc_channel: Optional[Any] = None):
    """Start all diagnostics."""
    get_orchestrator(rust_ipc_channel).start()


def stop_diagnostics():
    """Stop all diagnostics."""
    global _orchestrator
    if _orchestrator is not None:
        _orchestrator.stop()


def get_health_status() -> Dict[str, Any]:
    """Get current health status."""
    if _orchestrator is None:
        return {'status': 'not_started'}
    return {
        'overall': _orchestrator.get_overall_health().value,
        'components': _orchestrator.get_component_health(),
        'statistics': _orchestrator.get_statistics()
    }


def trigger_kill_switch(reason: str):
    """Trigger global kill switch."""
    if _orchestrator is not None:
        _orchestrator.force_kill_switch(reason)


# Module exports
__all__ = [
    'HealthStatus',
    'DiagnosticAlert',
    'SelfHealAction',
    'DiagnosticsOrchestrator',
    'get_orchestrator',
    'start_diagnostics',
    'stop_diagnostics',
    'get_health_status',
    'trigger_kill_switch'
]
