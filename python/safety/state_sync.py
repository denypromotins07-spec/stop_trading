"""
Continuous background sync comparing Nautilus internal portfolio state against Rust IPC state.
Detects fatal divergence or missed execution reports that could lead to unhedged toxic exposure.
Memory-bounded implementation for 3GB RAM limit - no Pandas in hot-path.
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import time
import hashlib


@dataclass
class StateSnapshot:
    """Point-in-time state snapshot."""
    timestamp: int
    positions: Dict[str, float]
    cash_balances: Dict[str, float]
    pending_orders: Dict[str, Dict]
    checksum: str


class StateSyncEngine:
    """
    Continuous state synchronization engine comparing Python and Rust states.
    Detects divergences and triggers reconciliation when needed.
    """
    
    def __init__(self, check_interval_ms: int = 100,
                 max_history_length: int = 1000):
        self.check_interval_ms = check_interval_ms
        self.max_history_length = max_history_length
        
        # State storage (bounded)
        self._nautilus_state_history: List[StateSnapshot] = []
        self._rust_state_history: List[StateSnapshot] = []
        
        # Last known synchronized state
        self._last_sync_state: Optional[StateSnapshot] = None
        self._last_sync_time: float = 0
        
        # Divergence tracking
        self._divergence_events: List[Dict] = []
        self._consecutive_divergences: int = 0
        
        # Sync statistics
        self._total_checks: int = 0
        self._successful_syncs: int = 0
        self._failed_syncs: int = 0
    
    def _compute_checksum(self, positions: Dict[str, float],
                          cash_balances: Dict[str, float]) -> str:
        """Compute deterministic checksum of state."""
        # Sort keys for determinism
        pos_str = ",".join(f"{k}:{v:.8f}" for k, v in sorted(positions.items()))
        cash_str = ",".join(f"{k}:{v:.2f}" for k, v in sorted(cash_balances.items()))
        
        combined = f"{pos_str}|{cash_str}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]
    
    def record_nautilus_state(self, positions: Dict[str, float],
                               cash_balances: Dict[str, float],
                               pending_orders: Dict[str, Dict]) -> StateSnapshot:
        """Record state from Nautilus system."""
        snapshot = StateSnapshot(
            timestamp=int(time.time() * 1e9),
            positions=positions.copy(),
            cash_balances=cash_balances.copy(),
            pending_orders=pending_orders.copy(),
            checksum=self._compute_checksum(positions, cash_balances)
        )
        
        self._nautilus_state_history.append(snapshot)
        
        # Keep history bounded
        if len(self._nautilus_state_history) > self.max_history_length:
            self._nautilus_state_history = self._nautilus_state_history[-self.max_history_length:]
        
        return snapshot
    
    def record_rust_state(self, positions: Dict[str, float],
                          cash_balances: Dict[str, float],
                          pending_orders: Dict[str, Dict]) -> StateSnapshot:
        """Record state from Rust IPC."""
        snapshot = StateSnapshot(
            timestamp=int(time.time() * 1e9),
            positions=positions.copy(),
            cash_balances=cash_balances.copy(),
            pending_orders=pending_orders.copy(),
            checksum=self._compute_checksum(positions, cash_balances)
        )
        
        self._rust_state_history.append(snapshot)
        
        # Keep history bounded
        if len(self._rust_state_history) > self.max_history_length:
            self._rust_state_history = self._rust_state_history[-self.max_history_length:]
        
        return snapshot
    
    def check_synchronization(self) -> Dict:
        """
        Check synchronization between Nautilus and Rust states.
        
        Returns:
            Synchronization status dictionary
        """
        self._total_checks += 1
        
        if not self._nautilus_state_history or not self._rust_state_history:
            return {
                "status": "incomplete",
                "reason": "Missing state history",
                "timestamp": int(time.time() * 1e9)
            }
        
        # Get latest states
        nautilus_state = self._nautilus_state_history[-1]
        rust_state = self._rust_state_history[-1]
        
        result = {
            "status": "synced",
            "timestamp": int(time.time() * 1e9),
            "checks_performed": self._total_checks,
            "divergences_detected": []
        }
        
        # Compare checksums first (fast path)
        if nautilus_state.checksum == rust_state.checksum:
            self._successful_syncs += 1
            self._consecutive_divergences = 0
            self._last_sync_state = nautilus_state
            self._last_sync_time = time.time()
            return result
        
        # Detailed comparison on checksum mismatch
        divergences = []
        
        # Position comparison
        all_instruments = set(nautilus_state.positions.keys()) | set(rust_state.positions.keys())
        for inst in all_instruments:
            nat_pos = nautilus_state.positions.get(inst, 0.0)
            rust_pos = rust_state.positions.get(inst, 0.0)
            
            diff = abs(nat_pos - rust_pos)
            tolerance = max(abs(nat_pos) * 0.001, 1e-6)  # 0.1% tolerance
            
            if diff > tolerance:
                divergences.append({
                    "type": "position_mismatch",
                    "instrument": inst,
                    "nautilus_value": nat_pos,
                    "rust_value": rust_pos,
                    "difference": diff,
                    "severity": "critical" if diff > abs(nat_pos) * 0.01 else "warning"
                })
        
        # Cash balance comparison
        all_currencies = set(nautilus_state.cash_balances.keys()) | set(rust_state.cash_balances.keys())
        for curr in all_currencies:
            nat_cash = nautilus_state.cash_balances.get(curr, 0.0)
            rust_cash = rust_state.cash_balances.get(curr, 0.0)
            
            diff = abs(nat_cash - rust_cash)
            tolerance = max(abs(nat_cash) * 0.001, 0.01)  # 0.1% or $0.01
            
            if diff > tolerance:
                divergences.append({
                    "type": "cash_mismatch",
                    "currency": curr,
                    "nautilus_value": nat_cash,
                    "rust_value": rust_cash,
                    "difference": diff,
                    "severity": "critical" if diff > abs(nat_cash) * 0.01 else "warning"
                })
        
        # Pending orders comparison
        nat_orders = set(nautilus_state.pending_orders.keys())
        rust_orders = set(rust_state.pending_orders.keys())
        
        missing_in_rust = nat_orders - rust_orders
        missing_in_nat = rust_orders - nat_orders
        
        for order_id in missing_in_rust:
            divergences.append({
                "type": "missing_execution_report",
                "order_id": order_id,
                "location": "rust_missing",
                "severity": "critical"
            })
        
        for order_id in missing_in_nat:
            divergences.append({
                "type": "phantom_execution_report",
                "order_id": order_id,
                "location": "nautilus_missing",
                "severity": "warning"
            })
        
        # Update result
        if divergences:
            result["status"] = "diverged"
            result["divergences_detected"] = divergences
            self._failed_syncs += 1
            self._consecutive_divergences += 1
            
            # Record divergence event
            self._divergence_events.append({
                "timestamp": int(time.time() * 1e9),
                "divergences": divergences,
                "nautilus_checksum": nautilus_state.checksum,
                "rust_checksum": rust_state.checksum
            })
            
            # Keep bounded
            if len(self._divergence_events) > 100:
                self._divergence_events = self._divergence_events[-100:]
        else:
            self._successful_syncs += 1
            self._consecutive_divergences = 0
            self._last_sync_state = nautilus_state
            self._last_sync_time = time.time()
        
        return result
    
    def get_reconciliation_commands(self) -> List[Dict]:
        """Generate commands to reconcile state divergence."""
        if self._last_sync_state is None:
            return []
        
        commands = []
        
        # Compare current Nautilus state with last known good sync
        if self._nautilus_state_history:
            current = self._nautilus_state_history[-1]
            
            # Check for position drift since last sync
            for inst in current.positions:
                last_pos = self._last_sync_state.positions.get(inst, 0.0)
                current_pos = current.positions.get(inst, 0.0)
                
                drift = abs(current_pos - last_pos)
                if drift > abs(last_pos) * 0.05:  # 5% drift threshold
                    commands.append({
                        "type": "reconciliation_check",
                        "instrument": inst,
                        "last_known_position": last_pos,
                        "current_position": current_pos,
                        "drift_pct": float(drift / (abs(last_pos) + 1e-10) * 100),
                        "action": "verify_executions"
                    })
        
        return commands
    
    def get_sync_health(self) -> Dict:
        """Get synchronization health metrics."""
        success_rate = self._successful_syncs / (self._total_checks + 1e-10)
        
        return {
            "total_checks": self._total_checks,
            "successful_syncs": self._successful_syncs,
            "failed_syncs": self._failed_syncs,
            "success_rate": float(success_rate),
            "consecutive_divergences": self._consecutive_divergences,
            "last_sync_time": self._last_sync_time,
            "divergence_events_count": len(self._divergence_events),
            "health_status": "healthy" if success_rate > 0.99 else "degraded" if success_rate > 0.95 else "critical"
        }
    
    def clear_divergence_history(self):
        """Clear divergence history after manual reconciliation."""
        self._divergence_events.clear()
        self._consecutive_divergences = 0


class StateSyncMonitor:
    """
    Background monitor running continuous state synchronization checks.
    Triggers alerts and kill switches on critical divergences.
    """
    
    def __init__(self, alert_threshold: int = 3):
        self.engine = StateSyncEngine()
        self.alert_threshold = alert_threshold
        self._alerts: List[Dict] = []
        self._running: bool = False
    
    def update_and_check(self, nautilus_state: Dict, rust_state: Dict) -> Dict:
        """Update states and perform synchronization check."""
        # Record states
        self.engine.record_nautilus_state(
            positions=nautilus_state.get("positions", {}),
            cash_balances=nautilus_state.get("cash_balances", {}),
            pending_orders=nautilus_state.get("pending_orders", {})
        )
        
        self.engine.record_rust_state(
            positions=rust_state.get("positions", {}),
            cash_balances=rust_state.get("cash_balances", {}),
            pending_orders=rust_state.get("pending_orders", {})
        )
        
        # Check sync
        result = self.engine.check_synchronization()
        
        # Generate alerts if needed
        if result["status"] == "diverged":
            critical_count = sum(
                1 for d in result.get("divergences_detected", [])
                if d.get("severity") == "critical"
            )
            
            if critical_count > 0:
                alert = {
                    "type": "state_divergence",
                    "severity": "critical",
                    "timestamp": int(time.time() * 1e9),
                    "divergence_count": critical_count,
                    "details": [d for d in result["divergences_detected"] if d.get("severity") == "critical"]
                }
                self._alerts.append(alert)
                
                # Keep bounded
                if len(self._alerts) > 50:
                    self._alerts = self._alerts[-50:]
        
        return result
    
    def get_pending_alerts(self) -> List[Dict]:
        """Get and clear pending alerts."""
        alerts = self._alerts.copy()
        self._alerts.clear()
        return alerts
    
    def should_trigger_kill_switch(self) -> Tuple[bool, str]:
        """Determine if kill switch should be triggered."""
        health = self.engine.get_sync_health()
        
        if health["consecutive_divergences"] >= self.alert_threshold:
            return True, f"Consecutive divergences exceeded threshold ({health['consecutive_divergences']})"
        
        if health["success_rate"] < 0.90 and health["total_checks"] > 100:
            return True, f"Sync success rate critically low ({health['success_rate']:.2%})"
        
        return False, ""


if __name__ == "__main__":
    # Example usage
    monitor = StateSyncMonitor(alert_threshold=3)
    
    # Simulate states
    np.random.seed(42)
    
    print("Simulating State Synchronization:\n")
    
    for i in range(20):
        # Nautilus state
        nautilus_state = {
            "positions": {
                "BTC": 1.5 + np.random.normal(0, 0.01),
                "ETH": 10.0 + np.random.normal(0, 0.1),
                "SOL": 100.0 + np.random.normal(0, 1)
            },
            "cash_balances": {
                "USD": 50000 + np.random.normal(0, 10),
                "EUR": 10000
            },
            "pending_orders": {
                f"order_{i}": {"status": "pending"}
            }
        }
        
        # Rust state (with occasional divergence)
        rust_state = {
            "positions": {
                "BTC": 1.5 + np.random.normal(0, 0.01),
                "ETH": 10.0 + np.random.normal(0, 0.1),
                "SOL": 100.0 + np.random.normal(0, 1)
            },
            "cash_balances": {
                "USD": 50000 + np.random.normal(0, 10),
                "EUR": 10000
            },
            "pending_orders": {
                f"order_{i}": {"status": "pending"}
            }
        }
        
        # Introduce divergence at step 10
        if i == 10:
            rust_state["positions"]["BTC"] = 1.3  # Mismatch!
            del rust_state["pending_orders"][f"order_{i}"]  # Missing order
        
        result = monitor.update_and_check(nautilus_state, rust_state)
        
        if result["status"] == "diverged":
            print(f"Iteration {i}: DIVERGENCE DETECTED!")
            for div in result.get("divergences_detected", []):
                print(f"  - {div['type']}: {div.get('instrument', div.get('order_id', 'N/A'))}")
        elif i % 5 == 0:
            print(f"Iteration {i}: Synced ✓")
    
    # Health report
    health = monitor.engine.get_sync_health()
    print(f"\nSync Health:")
    print(f"  Success Rate: {health['success_rate']:.1%}")
    print(f"  Total Checks: {health['total_checks']}")
    print(f"  Status: {health['health_status']}")
    
    # Check kill switch
    should_kill, reason = monitor.should_trigger_kill_switch()
    if should_kill:
        print(f"\n⚠️  KILL SWITCH TRIGGERED: {reason}")
    else:
        print("\n✓ System operating normally")
