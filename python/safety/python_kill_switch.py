"""
Python-side circuit breaker monitoring ML model hallucination, NaN outputs, or extreme reward anomalies.
Instantly transmits high-priority interrupt via ZeroMQ to Rust Global Kill Switch.
Uses non-blocking, ultra-fast ZMQ push sockets for microsecond halt signal delivery.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import time
import zmq


@dataclass
class SafetyThresholds:
    """Safety threshold configuration."""
    max_nan_count: int = 3
    max_inf_count: int = 1
    max_reward_deviation_std: float = 5.0
    max_position_change_pct: float = 0.5
    max_slippage_bps: float = 100
    max_drawdown_pct: float = 0.15
    cooldown_seconds: float = 60.0


class AnomalyDetector:
    """
    Detects ML model anomalies including hallucinations, NaN outputs, and extreme values.
    Uses statistical methods for real-time anomaly detection.
    """
    
    def __init__(self, thresholds: SafetyThresholds = None):
        self.thresholds = thresholds or SafetyThresholds()
        
        # Reward tracking (bounded window)
        self._reward_history: List[float] = []
        self._max_reward_history = 1000
        
        # Anomaly counters
        self._nan_count: int = 0
        self._inf_count: int = 0
        self._anomaly_events: List[Dict] = []
        
        # Last anomaly time for cooldown
        self._last_anomaly_time: float = 0
    
    def check_array_validity(self, array: np.ndarray, name: str = "array") -> Dict:
        """Check array for NaN, Inf, and extreme values."""
        result = {
            "valid": True,
            "name": name,
            "issues": []
        }
        
        # Check for NaN
        nan_mask = np.isnan(array)
        nan_count = np.sum(nan_mask)
        
        if nan_count > 0:
            result["valid"] = False
            result["issues"].append({
                "type": "nan_detected",
                "count": int(nan_count),
                "pct": float(nan_count / array.size * 100)
            })
            self._nan_count += nan_count
        
        # Check for Inf
        inf_mask = np.isinf(array)
        inf_count = np.sum(inf_mask)
        
        if inf_count > 0:
            result["valid"] = False
            result["issues"].append({
                "type": "inf_detected",
                "count": int(inf_count),
                "pct": float(inf_count / array.size * 100)
            })
            self._inf_count += inf_count
        
        # Check for extreme values (> 10 sigma)
        if array.size > 10 and result["valid"]:
            mean = np.mean(array)
            std = np.std(array)
            
            if std > 1e-10:
                z_scores = np.abs((array - mean) / std)
                extreme_mask = z_scores > 10
                extreme_count = np.sum(extreme_mask)
                
                if extreme_count > 0:
                    result["issues"].append({
                        "type": "extreme_values",
                        "count": int(extreme_count),
                        "threshold": "10_sigma"
                    })
        
        return result
    
    def check_reward_anomaly(self, current_reward: float) -> Dict:
        """Check for anomalous reward values."""
        result = {
            "valid": True,
            "reward": current_reward,
            "anomaly_detected": False,
            "reason": ""
        }
        
        # Add to history
        self._reward_history.append(current_reward)
        if len(self._reward_history) > self._max_reward_history:
            self._reward_history = self._reward_history[-self._max_reward_history:]
        
        # Need sufficient history for statistical check
        if len(self._reward_history) < 30:
            return result
        
        rewards = np.array(self._reward_history[:-1])  # Exclude current
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        
        if std_reward < 1e-10:
            return result
        
        # Z-score check
        z_score = abs(current_reward - mean_reward) / std_reward
        
        if z_score > self.thresholds.max_reward_deviation_std:
            result["valid"] = False
            result["anomaly_detected"] = True
            result["reason"] = f"Reward deviation {z_score:.2f}σ exceeds threshold {self.thresholds.max_reward_deviation_std}σ"
            result["z_score"] = float(z_score)
            result["mean_reward"] = float(mean_reward)
            result["std_reward"] = float(std_reward)
        
        return result
    
    def check_model_output(self, output: Dict[str, Any]) -> Dict:
        """Comprehensive check of ML model output."""
        issues = []
        
        # Check all numpy arrays in output
        for key, value in output.items():
            if isinstance(value, np.ndarray):
                validity = self.check_array_validity(value, name=key)
                if not validity["valid"]:
                    issues.extend(validity["issues"])
            elif isinstance(value, (float, np.floating)):
                if np.isnan(value):
                    issues.append({"type": "nan_scalar", "key": key})
                    self._nan_count += 1
                elif np.isinf(value):
                    issues.append({"type": "inf_scalar", "key": key})
                    self._inf_count += 1
        
        # Check reward if present
        if "reward" in output:
            reward_check = self.check_reward_anomaly(float(output["reward"]))
            if not reward_check["valid"]:
                issues.append({
                    "type": "reward_anomaly",
                    "details": reward_check
                })
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "total_nan_count": self._nan_count,
            "total_inf_count": self._inf_count
        }
    
    def should_trigger(self) -> Tuple[bool, str]:
        """Determine if kill switch should be triggered."""
        reasons = []
        
        # Check NaN threshold
        if self._nan_count >= self.thresholds.max_nan_count:
            reasons.append(f"NaN count ({self._nan_count}) exceeds threshold ({self.thresholds.max_nan_count})")
        
        # Check Inf threshold
        if self._inf_count >= self.thresholds.max_inf_count:
            reasons.append(f"Inf count ({self._inf_count}) exceeds threshold ({self.thresholds.max_inf_count})")
        
        # Check cooldown
        if reasons:
            elapsed = time.time() - self._last_anomaly_time
            if elapsed < self.thresholds.cooldown_seconds:
                return False, f"In cooldown ({elapsed:.1f}s elapsed)"
            
            self._last_anomaly_time = time.time()
            return True, "; ".join(reasons)
        
        return False, ""
    
    def reset_counters(self):
        """Reset anomaly counters after manual intervention."""
        self._nan_count = 0
        self._inf_count = 0
        self._reward_history.clear()


class PythonKillSwitch:
    """
    Python-side kill switch with ZeroMQ integration for instant Rust communication.
    Monitors ML models and triggers circuit breaker on anomalies.
    """
    
    def __init__(self, zmq_endpoint: str = "tcp://localhost:5555",
                 thresholds: SafetyThresholds = None):
        self.thresholds = thresholds or SafetyThresholds()
        self.anomaly_detector = AnomalyDetector(self.thresholds)
        
        # ZeroMQ socket setup (non-blocking PUSH)
        self._context = zmq.Context.instance()
        self._socket: Optional[zmq.Socket] = None
        self._zmq_endpoint = zmq_endpoint
        
        # State
        self._armed = True
        self._triggered = False
        self._trigger_time: Optional[float] = None
        self._trigger_reason: str = ""
        
        # Metrics
        self._checks_performed: int = 0
        self._false_positives: int = 0
    
    def connect(self):
        """Connect to Rust kill switch via ZeroMQ."""
        try:
            self._socket = self._context.socket(zmq.PUSH)
            self._socket.setsockopt(zmq.LINGER, 0)  # No blocking on close
            self._socket.setsockopt(zmq.SNDHWM, 1)  # Low water mark
            self._socket.connect(self._zmq_endpoint)
            print(f"[KillSwitch] Connected to {self._zmq_endpoint}")
        except Exception as e:
            print(f"[KillSwitch] Connection failed: {e}")
            self._socket = None
    
    def disconnect(self):
        """Disconnect ZeroMQ socket."""
        if self._socket:
            self._socket.close()
            self._socket = None
    
    def check_and_maybe_trigger(self, model_output: Dict[str, Any],
                                 additional_checks: Dict = None) -> Dict:
        """
        Check model output and trigger kill switch if needed.
        
        Args:
            model_output: ML model output dictionary
            additional_checks: Additional safety checks
            
        Returns:
            Check result dictionary
        """
        self._checks_performed += 1
        
        result = {
            "triggered": False,
            "reason": "",
            "timestamp": int(time.time() * 1e9),
            "checks_performed": self._checks_performed
        }
        
        if not self._armed or self._triggered:
            result["status"] = "disabled" if not self._armed else "already_triggered"
            return result
        
        # Check model output for anomalies
        anomaly_result = self.anomaly_detector.check_model_output(model_output)
        
        if not anomaly_result["valid"]:
            # Record anomaly event
            self.anomaly_detector._anomaly_events.append({
                "timestamp": time.time(),
                "issues": anomaly_result["issues"]
            })
            
            # Keep bounded
            if len(self.anomaly_detector._anomaly_events) > 100:
                self.anomaly_detector._anomaly_events = self.anomaly_detector._anomaly_events[-100:]
        
        # Check if trigger conditions met
        should_trigger, reason = self.anomaly_detector.should_trigger()
        
        if should_trigger:
            self._trigger(reason)
            result["triggered"] = True
            result["reason"] = reason
            result["anomaly_details"] = anomaly_result["issues"]
        
        # Additional checks
        if additional_checks:
            for check_name, check_value in additional_checks.items():
                if self._check_additional(check_name, check_value):
                    self._trigger(f"Additional check failed: {check_name}")
                    result["triggered"] = True
                    result["reason"] = f"Additional check failed: {check_name}"
                    break
        
        return result
    
    def _check_additional(self, check_name: str, value: Any) -> bool:
        """Perform additional safety checks."""
        if check_name == "position_change":
            # Check position change percentage
            if abs(value) > self.thresholds.max_position_change_pct:
                return True
        
        elif check_name == "slippage_bps":
            # Check slippage
            if value > self.thresholds.max_slippage_bps:
                return True
        
        elif check_name == "drawdown":
            # Check drawdown
            if value > self.thresholds.max_drawdown_pct:
                return True
        
        return False
    
    def _trigger(self, reason: str):
        """Trigger the kill switch."""
        self._triggered = True
        self._trigger_time = time.time()
        self._trigger_reason = reason
        
        # Send halt signal to Rust via ZeroMQ
        self._send_halt_signal(reason)
        
        print(f"[KillSwitch] TRIGGERED: {reason}")
    
    def _send_halt_signal(self, reason: str):
        """Send halt signal to Rust core via ZeroMQ."""
        if self._socket is None:
            print("[KillSwitch] Cannot send halt: Not connected")
            return
        
        halt_message = {
            "type": "kill_switch_triggered",
            "source": "python_safety",
            "reason": reason,
            "timestamp": int(time.time() * 1e9),
            "priority": "critical"
        }
        
        try:
            # Non-blocking send
            self._socket.send_json(halt_message, flags=zmq.NOBLOCK)
            print("[KillSwitch] Halt signal sent to Rust")
        except zmq.Again:
            print("[KillSwitch] Failed to send halt: Socket buffer full")
        except Exception as e:
            print(f"[KillSwitch] Error sending halt: {e}")
    
    def arm(self):
        """Arm the kill switch."""
        self._armed = True
        print("[KillSwitch] Armed")
    
    def disarm(self):
        """Disarm the kill switch."""
        self._armed = False
        print("[KillSwitch] Disarmed")
    
    def reset(self):
        """Reset the kill switch after manual intervention."""
        self._triggered = False
        self._trigger_time = None
        self._trigger_reason = ""
        self.anomaly_detector.reset_counters()
        print("[KillSwitch] Reset")
    
    def get_status(self) -> Dict:
        """Get current kill switch status."""
        return {
            "armed": self._armed,
            "triggered": self._triggered,
            "trigger_time": self._trigger_time,
            "trigger_reason": self._trigger_reason,
            "checks_performed": self._checks_performed,
            "nan_count": self.anomaly_detector._nan_count,
            "inf_count": self.anomaly_detector._inf_count,
            "connected": self._socket is not None
        }


if __name__ == "__main__":
    # Example usage
    kill_switch = PythonKillSwitch(zmq_endpoint="tcp://localhost:5555")
    
    # Connect to Rust (would fail gracefully if not running)
    kill_switch.connect()
    kill_switch.arm()
    
    print("Testing Python Kill Switch:\n")
    
    # Normal operation
    print("1. Normal model output:")
    normal_output = {
        "action": np.array([0.5, 0.3, 0.2]),
        "value": 0.85,
        "reward": 0.1
    }
    result = kill_switch.check_and_maybe_trigger(normal_output)
    print(f"   Triggered: {result['triggered']}")
    
    # Simulate NaN anomaly
    print("\n2. Model output with NaN:")
    for i in range(4):
        nan_output = {
            "action": np.array([np.nan, 0.3, 0.2]),
            "value": 0.85,
            "reward": 0.1
        }
        result = kill_switch.check_and_maybe_trigger(nan_output)
        print(f"   Check {i+1}: NaN count = {kill_switch.anomaly_detector._nan_count}")
    
    # Should trigger now
    print("\n3. Checking trigger condition:")
    result = kill_switch.check_and_maybe_trigger(normal_output)
    print(f"   Triggered: {result['triggered']}")
    if result["triggered"]:
        print(f"   Reason: {result['reason']}")
    
    # Status
    print(f"\n4. Kill Switch Status:")
    status = kill_switch.get_status()
    for key, value in status.items():
        print(f"   {key}: {value}")
    
    # Reset
    print("\n5. Resetting kill switch...")
    kill_switch.reset()
    print(f"   New status: triggered={kill_switch.get_status()['triggered']}")
    
    kill_switch.disconnect()
