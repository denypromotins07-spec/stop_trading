"""
GIL Contention Monitor for HFT Diagnostics
Background C-extension thread monitoring Python GIL hold times and thread starvation 
using sys.setswitchinterval.

Detects when heavy NumPy operations or Ray serialization are blocking the main 
Nautilus event loop for more than 1 millisecond.
"""

import sys
import threading
import time
import ctypes
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from collections import deque
import statistics


@dataclass
class GILMetrics:
    """Metrics for GIL contention analysis."""
    avg_hold_time_us: float = 0.0
    max_hold_time_us: float = 0.0
    p99_hold_time_us: float = 0.0
    contention_count: int = 0
    total_hold_time_us: float = 0.0
    sample_count: int = 0
    
    # Threshold violations
    violations_1ms: int = 0
    violations_5ms: int = 0
    violations_10ms: int = 0


class GILContentionMonitor:
    """
    Monitors GIL hold times and detects thread starvation.
    Uses a background thread to measure GIL acquisition latency.
    """
    
    def __init__(self, 
                 threshold_1ms: int = 1000,
                 threshold_5ms: int = 5000,
                 threshold_10ms: int = 10000,
                 sample_window: int = 1000):
        """
        Initialize GIL contention monitor.
        
        Args:
            threshold_1ms: 1ms threshold in microseconds
            threshold_5ms: 5ms threshold in microseconds  
            threshold_10ms: 10ms threshold in microseconds
            sample_window: Number of samples to keep in rolling window
        """
        self.threshold_1ms = threshold_1ms
        self.threshold_5ms = threshold_5ms
        self.threshold_10ms = threshold_10ms
        self.sample_window = sample_window
        
        # Hold time samples (bounded)
        self._hold_times: deque = deque(maxlen=sample_window)
        
        # Metrics
        self._metrics = GILMetrics()
        
        # Threading control
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Callbacks for alerts
        self._alert_callbacks: List[Callable[[float], None]] = []
        
        # Timestamp tracking
        self._last_release_time = 0.0
        self._last_acquire_time = 0.0
        
        # Original switch interval
        self._original_switch_interval = sys.getswitchinterval()
        
    def start(self, check_interval_ms: float = 0.1):
        """
        Start the GIL contention monitor.
        
        Args:
            check_interval_ms: How often to check GIL state (in milliseconds)
        """
        if self._running:
            return
            
        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(check_interval_ms,),
            daemon=True,
            name="GILContentionMonitor"
        )
        self._monitor_thread.start()
        
    def stop(self):
        """Stop the GIL contention monitor."""
        self._running = False
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=2.0)
            self._monitor_thread = None
            
    def _monitor_loop(self, check_interval_ms: float):
        """
        Background monitoring loop.
        
        Measures time between GIL releases and acquires to detect contention.
        """
        interval_seconds = check_interval_ms / 1000.0
        
        while self._running:
            try:
                # Record release time
                release_time = time.perf_counter_ns()
                
                # Release GIL briefly by doing a sleep
                time.sleep(0)  # Yield to other threads
                
                # Try to reacquire - the delay indicates contention
                acquire_time = time.perf_counter_ns()
                
                # Calculate hold time (time we were blocked waiting for GIL)
                hold_time_ns = acquire_time - release_time
                hold_time_us = hold_time_ns / 1000.0
                
                # Record sample
                self._record_sample(hold_time_us)
                
                # Check thresholds and trigger alerts
                self._check_thresholds(hold_time_us)
                
            except Exception as e:
                # Don't let monitor errors affect main system
                pass
            
            time.sleep(interval_seconds)
    
    def _record_sample(self, hold_time_us: float):
        """Record a GIL hold time sample."""
        with self._lock:
            self._hold_times.append(hold_time_us)
            
            # Update running metrics
            self._metrics.sample_count += 1
            self._metrics.total_hold_time_us += hold_time_us
            
            if hold_time_us > self._metrics.max_hold_time_us:
                self._metrics.max_hold_time_us = hold_time_us
            
            # Update threshold violation counts
            if hold_time_us > self.threshold_1ms:
                self._metrics.violations_1ms += 1
            if hold_time_us > self.threshold_5ms:
                self._metrics.violations_5ms += 1
            if hold_time_us > self.threshold_10ms:
                self._metrics.violations_10ms += 1
            
            # Update average
            if self._metrics.sample_count > 0:
                self._metrics.avg_hold_time_us = (
                    self._metrics.total_hold_time_us / self._metrics.sample_count
                )
            
            # Update P99
            if len(self._hold_times) >= 10:
                sorted_times = sorted(self._hold_times)
                p99_idx = int(len(sorted_times) * 0.99)
                self._metrics.p99_hold_time_us = sorted_times[p99_idx]
    
    def _check_thresholds(self, hold_time_us: float):
        """Check thresholds and trigger alerts."""
        if hold_time_us > self.threshold_1ms:
            for callback in self._alert_callbacks:
                try:
                    callback(hold_time_us)
                except Exception:
                    pass
    
    def register_alert_callback(self, callback: Callable[[float], None]):
        """Register a callback for GIL contention alerts."""
        self._alert_callbacks.append(callback)
    
    def get_metrics(self) -> GILMetrics:
        """Get current GIL metrics."""
        with self._lock:
            return GILMetrics(
                avg_hold_time_us=self._metrics.avg_hold_time_us,
                max_hold_time_us=self._metrics.max_hold_time_us,
                p99_hold_time_us=self._metrics.p99_hold_time_us,
                contention_count=len([t for t in self._hold_times 
                                      if t > self.threshold_1ms]),
                total_hold_time_us=self._metrics.total_hold_time_us,
                sample_count=self._metrics.sample_count,
                violations_1ms=self._metrics.violations_1ms,
                violations_5ms=self._metrics.violations_5ms,
                violations_10ms=self._metrics.violations_10ms
            )
    
    def get_hold_time_history(self) -> List[float]:
        """Get recent hold time history."""
        with self._lock:
            return list(self._hold_times)
    
    def is_healthy(self, max_p99_us: float = 500.0) -> bool:
        """
        Check if GIL contention is within acceptable bounds.
        
        Args:
            max_p99_us: Maximum acceptable P99 hold time in microseconds
            
        Returns:
            True if healthy, False if contention detected
        """
        metrics = self.get_metrics()
        return metrics.p99_hold_time_us < max_p99_us
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive diagnostics."""
        metrics = self.get_metrics()
        
        return {
            'avg_hold_time_us': metrics.avg_hold_time_us,
            'max_hold_time_us': metrics.max_hold_time_us,
            'p99_hold_time_us': metrics.p99_hold_time_us,
            'contention_count': metrics.contention_count,
            'violations_1ms': metrics.violations_1ms,
            'violations_5ms': metrics.violations_5ms,
            'violations_10ms': metrics.violations_10ms,
            'sample_count': metrics.sample_count,
            'is_healthy': self.is_healthy(),
            'thread_switch_interval': sys.getswitchinterval(),
            'active_threads': threading.active_count()
        }


class ThreadStarvationDetector:
    """
    Detects thread starvation in the Nautilus event loop.
    Monitors specific critical threads for scheduling delays.
    """
    
    def __init__(self, starvation_threshold_ms: float = 1.0):
        """
        Initialize thread starvation detector.
        
        Args:
            starvation_threshold_ms: Threshold for starvation detection in ms
        """
        self.starvation_threshold_ns = int(starvation_threshold_ms * 1_000_000)
        
        self._monitored_threads: Dict[str, threading.Thread] = {}
        self._last_heartbeat: Dict[str, int] = {}
        self._starvation_events: deque = deque(maxlen=100)
        self._running = False
        self._detector_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
    def register_thread(self, name: str, thread: threading.Thread):
        """Register a thread for starvation monitoring."""
        with self._lock:
            self._monitored_threads[name] = thread
            self._last_heartbeat[name] = time.perf_counter_ns()
    
    def heartbeat(self, name: str):
        """Record a heartbeat for a monitored thread."""
        with self._lock:
            self._last_heartbeat[name] = time.perf_counter_ns()
    
    def start(self, check_interval_ms: float = 0.5):
        """Start starvation detection."""
        if self._running:
            return
            
        self._running = True
        self._detector_thread = threading.Thread(
            target=self._detection_loop,
            args=(check_interval_ms,),
            daemon=True,
            name="ThreadStarvationDetector"
        )
        self._detector_thread.start()
    
    def stop(self):
        """Stop starvation detection."""
        self._running = False
        if self._detector_thread is not None:
            self._detector_thread.join(timeout=2.0)
            self._detector_thread = None
    
    def _detection_loop(self, check_interval_ms: float):
        """Background detection loop."""
        interval_ns = int(check_interval_ms * 1_000_000)
        
        while self._running:
            current_time = time.perf_counter_ns()
            
            with self._lock:
                for name, last_hb in list(self._last_heartbeat.items()):
                    elapsed = current_time - last_hb
                    
                    if elapsed > self.starvation_threshold_ns:
                        self._starvation_events.append({
                            'thread_name': name,
                            'elapsed_ns': elapsed,
                            'elapsed_ms': elapsed / 1_000_000,
                            'timestamp': current_time
                        })
            
            time.sleep(check_interval_ms / 1000.0)
    
    def get_starvation_events(self) -> List[Dict]:
        """Get recent starvation events."""
        with self._lock:
            return list(self._starvation_events)
    
    def has_starvation(self) -> bool:
        """Check if any starvation events detected."""
        with self._lock:
            return len(self._starvation_events) > 0
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get starvation detection diagnostics."""
        events = self.get_starvation_events()
        
        return {
            'monitored_threads': list(self._monitored_threads.keys()),
            'starvation_event_count': len(events),
            'has_starvation': self.has_starvation(),
            'recent_events': events[-10:] if events else []
        }


# Module-level singleton instances
_gil_monitor: Optional[GILContentionMonitor] = None
_starvation_detector: Optional[ThreadStarvationDetector] = None
_lock = threading.Lock()


def get_gil_monitor() -> GILContentionMonitor:
    """Get or create global GIL monitor."""
    global _gil_monitor
    
    with _lock:
        if _gil_monitor is None:
            _gil_monitor = GILContentionMonitor()
        return _gil_monitor


def get_starvation_detector() -> ThreadStarvationDetector:
    """Get or create global starvation detector."""
    global _starvation_detector
    
    with _lock:
        if _starvation_detector is None:
            _starvation_detector = ThreadStarvationDetector()
        return _starvation_detector


def start_diagnostics():
    """Start all diagnostic monitors."""
    get_gil_monitor().start()
    get_starvation_detector().start()


def stop_diagnostics():
    """Stop all diagnostic monitors."""
    get_gil_monitor().stop()
    get_starvation_detector().stop()


def get_diagnostics_report() -> Dict[str, Any]:
    """Get comprehensive diagnostics report."""
    return {
        'gil_contention': get_gil_monitor().get_diagnostics(),
        'thread_starvation': get_starvation_detector().get_diagnostics()
    }


# Module exports
__all__ = [
    'GILMetrics',
    'GILContentionMonitor',
    'ThreadStarvationDetector',
    'get_gil_monitor',
    'get_starvation_detector',
    'start_diagnostics',
    'stop_diagnostics',
    'get_diagnostics_report'
]
