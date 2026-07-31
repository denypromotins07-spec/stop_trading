"""
Safety Module Root.
Manages health checks of all Ray workers and Nautilus kernels, ensuring 24/7 stability.
Coordinates state sync, kill switch, and system monitoring for production reliability.
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import time
import json


from .state_sync import StateSyncMonitor, StateSyncEngine
from .python_kill_switch import PythonKillSwitch, SafetyThresholds, AnomalyDetector


@dataclass
class SafetyConfig:
    """Configuration for safety system."""
    check_interval_ms: int = 100
    max_consecutive_failures: int = 5
    ray_worker_timeout_seconds: float = 30.0
    nautilus_kernel_timeout_seconds: float = 10.0
    zmq_endpoint: str = "tcp://localhost:5555"
    enable_state_sync: bool = True
    enable_kill_switch: bool = True
    health_check_endpoints: List[str] = field(default_factory=list)


@dataclass
class HealthStatus:
    """Health status of a component."""
    component_id: str
    component_type: str
    is_healthy: bool
    last_check_time: float
    response_time_ms: float
    consecutive_failures: int
    error_message: str = ""


class HealthChecker:
    """
    Manages health checks for Ray workers and Nautilus kernels.
    Implements bounded retry logic and failure tracking.
    """
    
    def __init__(self, config: SafetyConfig):
        self.config = config
        
        # Component tracking
        self._components: Dict[str, HealthStatus] = {}
        self._component_metadata: Dict[str, Dict] = {}
        
        # Health history (bounded)
        self._health_history: List[Dict] = []
        self._max_history = 1000
        
        # Alert tracking
        self._alerts: List[Dict] = []
        self._max_alerts = 100
    
    def register_component(self, component_id: str, component_type: str,
                           metadata: Dict = None):
        """Register a component for health monitoring."""
        self._components[component_id] = HealthStatus(
            component_id=component_id,
            component_type=component_type,
            is_healthy=True,
            last_check_time=time.time(),
            response_time_ms=0,
            consecutive_failures=0
        )
        
        self._component_metadata[component_id] = metadata or {}
    
    def unregister_component(self, component_id: str):
        """Unregister a component from health monitoring."""
        if component_id in self._components:
            del self._components[component_id]
        if component_id in self._component_metadata:
            del self._component_metadata[component_id]
    
    def record_health_check(self, component_id: str, is_healthy: bool,
                            response_time_ms: float = 0,
                            error_message: str = "") -> HealthStatus:
        """Record result of a health check."""
        if component_id not in self._components:
            return None
        
        status = self._components[component_id]
        status.is_healthy = is_healthy
        status.last_check_time = time.time()
        status.response_time_ms = response_time_ms
        status.error_message = error_message
        
        if not is_healthy:
            status.consecutive_failures += 1
        else:
            status.consecutive_failures = 0
        
        # Record in history
        self._health_history.append({
            "component_id": component_id,
            "timestamp": time.time(),
            "is_healthy": is_healthy,
            "response_time_ms": response_time_ms,
            "error_message": error_message
        })
        
        # Keep history bounded
        if len(self._health_history) > self._max_history:
            self._health_history = self._health_history[-self._max_history:]
        
        # Generate alert if needed
        if status.consecutive_failures >= self.config.max_consecutive_failures:
            self._generate_alert(component_id, status)
        
        return status
    
    def _generate_alert(self, component_id: str, status: HealthStatus):
        """Generate alert for unhealthy component."""
        alert = {
            "type": "component_unhealthy",
            "severity": "critical" if status.consecutive_failures >= self.config.max_consecutive_failures * 2 else "warning",
            "component_id": component_id,
            "component_type": status.component_type,
            "consecutive_failures": status.consecutive_failures,
            "error_message": status.error_message,
            "timestamp": int(time.time() * 1e9)
        }
        
        self._alerts.append(alert)
        
        # Keep bounded
        if len(self._alerts) > self._max_alerts:
            self._alerts = self._alerts[-self._max_alerts:]
    
    def get_component_status(self, component_id: str) -> Optional[HealthStatus]:
        """Get current status of a component."""
        return self._components.get(component_id)
    
    def get_all_statuses(self) -> Dict[str, HealthStatus]:
        """Get status of all components."""
        return self._components.copy()
    
    def get_pending_alerts(self) -> List[Dict]:
        """Get and clear pending alerts."""
        alerts = self._alerts.copy()
        self._alerts.clear()
        return alerts
    
    def get_system_health_summary(self) -> Dict:
        """Get summary of overall system health."""
        if not self._components:
            return {"status": "no_components"}
        
        healthy_count = sum(1 for c in self._components.values() if c.is_healthy)
        total_count = len(self._components)
        health_pct = healthy_count / total_count
        
        # Determine overall status
        if health_pct == 1.0:
            overall_status = "healthy"
        elif health_pct >= 0.8:
            overall_status = "degraded"
        else:
            overall_status = "critical"
        
        return {
            "overall_status": overall_status,
            "healthy_components": healthy_count,
            "total_components": total_count,
            "health_percentage": float(health_pct),
            "unhealthy_components": [
                c.component_id for c in self._components.values() if not c.is_healthy
            ],
            "timestamp": int(time.time() * 1e9)
        }
    
    def check_ray_workers(self, worker_ids: List[str]) -> Dict[str, bool]:
        """Check health of Ray workers."""
        results = {}
        
        for worker_id in worker_ids:
            start_time = time.time()
            
            try:
                # In production, would use ray.get() with timeout
                # For now, simulate check
                is_healthy = True
                response_time = (time.time() - start_time) * 1000
                
                self.record_health_check(worker_id, is_healthy, response_time)
                results[worker_id] = is_healthy
                
            except Exception as e:
                response_time = (time.time() - start_time) * 1000
                self.record_health_check(worker_id, False, response_time, str(e))
                results[worker_id] = False
        
        return results
    
    def check_nautilus_kernels(self, kernel_ids: List[str]) -> Dict[str, bool]:
        """Check health of Nautilus kernels."""
        results = {}
        
        for kernel_id in kernel_ids:
            start_time = time.time()
            
            try:
                # In production, would ping the kernel via IPC
                is_healthy = True
                response_time = (time.time() - start_time) * 1000
                
                self.record_health_check(kernel_id, is_healthy, response_time)
                results[kernel_id] = is_healthy
                
            except Exception as e:
                response_time = (time.time() - start_time) * 1000
                self.record_health_check(kernel_id, False, response_time, str(e))
                results[kernel_id] = False
        
        return results


class SafetySystem:
    """
    Main safety system coordinating all safety components.
    Provides unified interface for monitoring and intervention.
    """
    
    def __init__(self, config: SafetyConfig = None):
        self.config = config or SafetyConfig()
        
        # Initialize components
        self.health_checker = HealthChecker(self.config)
        
        self.state_sync_monitor = None
        if self.config.enable_state_sync:
            self.state_sync_monitor = StateSyncMonitor()
        
        self.kill_switch = None
        if self.config.enable_kill_switch:
            self.kill_switch = PythonKillSwitch(
                zmq_endpoint=self.config.zmq_endpoint
            )
        
        # System state
        self._running = False
        self._system_start_time: float = 0
        
        # Register default components
        self._register_default_components()
    
    def _register_default_components(self):
        """Register default system components."""
        self.health_checker.register_component(
            "portfolio_optimizer", "ray_worker",
            {"module": "portfolio.port_mod"}
        )
        self.health_checker.register_component(
            "risk_predictor", "ray_worker",
            {"module": "risk.risk_mod"}
        )
        self.health_checker.register_component(
            "execution_engine", "nautilus_kernel",
            {"module": "execution.exec_mod"}
        )
        self.health_checker.register_component(
            "sor_router", "nautilus_kernel",
            {"module": "sor.sor_mod"}
        )
    
    def start(self):
        """Start the safety system."""
        self._running = True
        self._system_start_time = time.time()
        
        if self.kill_switch:
            self.kill_switch.connect()
            self.kill_switch.arm()
        
        print("[SafetySystem] Started")
    
    def stop(self):
        """Stop the safety system."""
        self._running = False
        
        if self.kill_switch:
            self.kill_switch.disarm()
            self.kill_switch.disconnect()
        
        print("[SafetySystem] Stopped")
    
    def run_health_checks(self) -> Dict:
        """Run all health checks and return summary."""
        # Check Ray workers
        ray_workers = ["portfolio_optimizer", "risk_predictor"]
        ray_results = self.health_checker.check_ray_workers(ray_workers)
        
        # Check Nautilus kernels
        nautilus_kernels = ["execution_engine", "sor_router"]
        nautilus_results = self.health_checker.check_nautilus_kernels(nautilus_kernels)
        
        # Get summary
        summary = self.health_checker.get_system_health_summary()
        summary["ray_workers"] = ray_results
        summary["nautilus_kernels"] = nautilus_results
        
        return summary
    
    def check_model_safety(self, model_output: Dict[str, Any],
                           additional_checks: Dict = None) -> Dict:
        """Check ML model output for safety violations."""
        if not self.kill_switch:
            return {"status": "disabled"}
        
        return self.kill_switch.check_and_maybe_trigger(
            model_output, additional_checks
        )
    
    def check_state_sync(self, nautilus_state: Dict, rust_state: Dict) -> Dict:
        """Check synchronization between Nautilus and Rust states."""
        if not self.state_sync_monitor:
            return {"status": "disabled"}
        
        return self.state_sync_monitor.update_and_check(nautilus_state, rust_state)
    
    def get_safety_dashboard(self) -> Dict:
        """Get comprehensive safety dashboard data."""
        dashboard = {
            "system_running": self._running,
            "uptime_seconds": time.time() - self._system_start_time if self._running else 0,
            "health": self.health_checker.get_system_health_summary(),
            "alerts": self.health_checker.get_pending_alerts(),
            "kill_switch": self.kill_switch.get_status() if self.kill_switch else {"status": "disabled"},
            "state_sync": None
        }
        
        if self.state_sync_monitor:
            dashboard["state_sync"] = self.state_sync_monitor.engine.get_sync_health()
        
        return dashboard
    
    def emergency_stop(self, reason: str = "Manual emergency stop"):
        """Trigger emergency stop across all systems."""
        print(f"[SafetySystem] EMERGENCY STOP: {reason}")
        
        # Trigger kill switch if available
        if self.kill_switch and self.kill_switch._armed:
            self.kill_switch._trigger(reason)
        
        # Stop the system
        self.stop()
        
        return {
            "status": "emergency_stop_triggered",
            "reason": reason,
            "timestamp": int(time.time() * 1e9)
        }


def create_safety_system(config: SafetyConfig = None) -> SafetySystem:
    """Factory function to create configured safety system."""
    return SafetySystem(config)


if __name__ == "__main__":
    # Example usage
    config = SafetyConfig(
        max_consecutive_failures=3,
        enable_state_sync=True,
        enable_kill_switch=True
    )
    
    safety = create_safety_system(config)
    safety.start()
    
    print("Safety System Dashboard:\n")
    
    # Run health checks
    health = safety.run_health_checks()
    print(f"System Health: {health['overall_status']}")
    print(f"Healthy: {health['healthy_components']}/{health['total_components']}")
    
    # Simulate model safety check
    print("\nModel Safety Check:")
    model_output = {
        "action": np.array([0.5, 0.3, 0.2]),
        "value": 0.85,
        "reward": 0.1
    }
    safety_result = safety.check_model_safety(model_output)
    print(f"  Status: {safety_result.get('status', 'checked')}")
    
    # Simulate state sync check
    print("\nState Sync Check:")
    nautilus_state = {
        "positions": {"BTC": 1.5, "ETH": 10.0},
        "cash_balances": {"USD": 50000},
        "pending_orders": {}
    }
    rust_state = {
        "positions": {"BTC": 1.5, "ETH": 10.0},
        "cash_balances": {"USD": 50000},
        "pending_orders": {}
    }
    sync_result = safety.check_state_sync(nautilus_state, rust_state)
    print(f"  Status: {sync_result.get('status', 'unknown')}")
    
    # Get full dashboard
    print("\nFull Dashboard:")
    dashboard = safety.get_safety_dashboard()
    print(f"  Uptime: {dashboard['uptime_seconds']:.1f}s")
    print(f"  Kill Switch Armed: {dashboard['kill_switch'].get('armed', False)}")
    if dashboard['state_sync']:
        print(f"  State Sync Success Rate: {dashboard['state_sync'].get('success_rate', 0):.1%}")
    
    safety.stop()
