"""
Memory Leak Detector for HFT Diagnostics
Automated tracemalloc snapshot comparator to identify memory fragmentation 
in long-running Ray workers.

Automatically gracefully restarts bloated inference workers during low-volatility 
periods before they trigger an OS-level OOM kill.
"""

import tracemalloc
import gc
import threading
import time
import os
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import deque
from enum import Enum


class MemoryState(Enum):
    """Memory state classification."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    LEAK_DETECTED = "leak_detected"


@dataclass
class MemorySnapshot:
    """Represents a memory snapshot."""
    timestamp: float
    total_bytes: int
    peak_bytes: int
    top_allocations: List[Tuple[str, int]]  # (traceback, size)
    gc_counts: Tuple[int, int, int]
    
    def diff_from(self, other: 'MemorySnapshot') -> 'MemoryDiff':
        """Compute difference from another snapshot."""
        return MemoryDiff(
            timestamp=self.timestamp - other.timestamp,
            bytes_delta=self.total_bytes - other.total_bytes,
            peak_delta=self.peak_bytes - other.peak_bytes,
            new_allocations=self._find_new_allocations(other),
            gc_delta=(
                self.gc_counts[0] - other.gc_counts[0],
                self.gc_counts[1] - other.gc_counts[1],
                self.gc_counts[2] - other.gc_counts[2]
            )
        )
    
    def _find_new_allocations(self, other: 'MemorySnapshot') -> List[Tuple[str, int]]:
        """Find allocations present in this snapshot but not in other."""
        other_traces = set(t[0] for t in other.top_allocations[:50])
        new = []
        for trace, size in self.top_allocations[:50]:
            if trace not in other_traces:
                new.append((trace, size))
        return new


@dataclass
class MemoryDiff:
    """Difference between two memory snapshots."""
    timestamp: float
    bytes_delta: int
    peak_delta: int
    new_allocations: List[Tuple[str, int]]
    gc_delta: Tuple[int, int, int]


@dataclass
class LeakReport:
    """Memory leak detection report."""
    is_leak: bool
    leak_rate_bytes_per_sec: float
    confidence: float  # 0.0 to 1.0
    suspected_sources: List[str]
    recommendation: str


class MemoryLeakDetector:
    """
    Detects memory leaks using tracemalloc snapshot comparison.
    Monitors memory growth patterns over time.
    """
    
    def __init__(self,
                 baseline_window: int = 10,
                 leak_threshold_mb: float = 50.0,
                 check_interval_sec: float = 5.0,
                 max_snapshots: int = 100):
        """
        Initialize memory leak detector.
        
        Args:
            baseline_window: Number of snapshots for baseline calculation
            leak_threshold_mb: Threshold for leak detection in MB
            check_interval_sec: Interval between checks in seconds
            max_snapshots: Maximum snapshots to retain
        """
        self.baseline_window = baseline_window
        self.leak_threshold_bytes = int(leak_threshold_mb * 1024 * 1024)
        self.check_interval_sec = check_interval_sec
        self.max_snapshots = max_snapshots
        
        # Snapshot history
        self._snapshots: deque = deque(maxlen=max_snapshots)
        self._baseline_snapshot: Optional[tracemalloc.Snapshot] = None
        
        # Monitoring state
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Callbacks
        self._leak_callbacks: List[Callable[[LeakReport], None]] = []
        self._restart_callbacks: List[Callable[[], None]] = []
        
        # Statistics
        self._total_growth_bytes = 0
        self._leak_detections = 0
        
        # Start tracemalloc if not already started
        if not tracemalloc.is_tracing():
            tracemalloc.start(nframes=10, limit=100)
    
    def start(self):
        """Start memory monitoring."""
        if self._running:
            return
            
        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="MemoryLeakDetector"
        )
        self._monitor_thread.start()
    
    def stop(self):
        """Stop memory monitoring."""
        self._running = False
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=5.0)
            self._monitor_thread = None
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self._running:
            try:
                self.take_snapshot()
                self.analyze_growth()
            except Exception as e:
                pass  # Don't crash monitoring on errors
            
            time.sleep(self.check_interval_sec)
    
    def take_snapshot(self) -> MemorySnapshot:
        """Take a memory snapshot."""
        with self._lock:
            current, peak = tracemalloc.get_traced_memory()
            snapshot = tracemalloc.take_snapshot()
            
            # Get top allocations
            top_stats = snapshot.statistics('traceback')[:20]
            top_allocations = [
                (str(stat.traceback), stat.size)
                for stat in top_stats
            ]
            
            # Get GC counts
            gc_counts = gc.get_count()
            
            memory_snapshot = MemorySnapshot(
                timestamp=time.time(),
                total_bytes=current,
                peak_bytes=peak,
                top_allocations=top_allocations,
                gc_counts=gc_counts
            )
            
            self._snapshots.append(memory_snapshot)
            
            # Set baseline if first snapshot
            if len(self._snapshots) == 1:
                self._baseline_snapshot = snapshot
            
            return memory_snapshot
    
    def analyze_growth(self) -> Optional[LeakReport]:
        """Analyze memory growth for leak detection."""
        with self._lock:
            if len(self._snapshots) < self.baseline_window:
                return None
            
            # Get recent snapshots
            recent = list(self._snapshots)[-self.baseline_window:]
            
            # Calculate growth rate
            first = recent[0]
            last = recent[-1]
            
            time_delta = last.timestamp - first.timestamp
            if time_delta <= 0:
                return None
            
            bytes_delta = last.total_bytes - first.total_bytes
            growth_rate = bytes_delta / time_delta
            
            # Check for leak
            leak_report = self._classify_leak(growth_rate, bytes_delta, recent)
            
            if leak_report.is_leak:
                self._leak_detections += 1
                self._total_growth_bytes += bytes_delta
                
                # Trigger callbacks
                for callback in self._leak_callbacks:
                    try:
                        callback(leak_report)
                    except Exception:
                        pass
            
            return leak_report
    
    def _classify_leak(self, 
                       growth_rate: float,
                       total_delta: int,
                       snapshots: List[MemorySnapshot]) -> LeakReport:
        """Classify whether growth constitutes a leak."""
        # Convert threshold to bytes/sec
        threshold_rate = self.leak_threshold_bytes / 60.0  # Per minute
        
        is_leak = growth_rate > threshold_rate and total_delta > 0
        
        if not is_leak:
            return LeakReport(
                is_leak=False,
                leak_rate_bytes_per_sec=growth_rate,
                confidence=0.0,
                suspected_sources=[],
                recommendation="No leak detected"
            )
        
        # Calculate confidence based on consistency of growth
        deltas = []
        for i in range(1, len(snapshots)):
            delta = snapshots[i].total_bytes - snapshots[i-1].total_bytes
            deltas.append(delta)
        
        if len(deltas) >= 2:
            mean_delta = sum(deltas) / len(deltas)
            variance = sum((d - mean_delta) ** 2 for d in deltas) / len(deltas)
            std_delta = variance ** 0.5
            
            # Higher confidence if growth is consistent (low variance)
            if mean_delta > 0:
                cv = std_delta / mean_delta if mean_delta > 0 else float('inf')
                confidence = max(0.0, min(1.0, 1.0 - cv))
            else:
                confidence = 0.0
        else:
            confidence = 0.5
        
        # Identify suspected sources
        suspected_sources = []
        if snapshots:
            last = snapshots[-1]
            for trace, size in last.new_allocations[:5]:
                if size > 1024 * 1024:  # > 1MB allocations
                    suspected_sources.append(trace[:200])
        
        # Generate recommendation
        if confidence > 0.8:
            recommendation = "IMMEDIATE_RESTART_REQUIRED"
        elif confidence > 0.5:
            recommendation = "SCHEDULE_RESTART_LOW_VOLATILITY"
        else:
            recommendation = "MONITOR_CLOSELY"
        
        return LeakReport(
            is_leak=True,
            leak_rate_bytes_per_sec=growth_rate,
            confidence=confidence,
            suspected_sources=suspected_sources,
            recommendation=recommendation
        )
    
    def register_leak_callback(self, callback: Callable[[LeakReport], None]):
        """Register callback for leak detection."""
        self._leak_callbacks.append(callback)
    
    def register_restart_callback(self, callback: Callable[[], None]):
        """Register callback for automatic restart."""
        self._restart_callbacks.append(callback)
    
    def get_current_usage(self) -> Dict[str, int]:
        """Get current memory usage."""
        current, peak = tracemalloc.get_traced_memory()
        return {
            'current_bytes': current,
            'peak_bytes': peak,
            'current_mb': current / (1024 * 1024),
            'peak_mb': peak / (1024 * 1024)
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get leak detection statistics."""
        return {
            'snapshot_count': len(self._snapshots),
            'total_growth_bytes': self._total_growth_bytes,
            'leak_detections': self._leak_detections,
            'current_usage': self.get_current_usage()
        }
    
    def force_gc(self) -> int:
        """Force garbage collection and return freed bytes."""
        before = tracemalloc.get_traced_memory()[0]
        gc.collect()
        after = tracemalloc.get_traced_memory()[0]
        return before - after


class WorkerHealthMonitor:
    """
    Monitors Ray worker health and triggers graceful restarts.
    Integrates with memory leak detection.
    """
    
    def __init__(self,
                 max_memory_mb: float = 2500.0,
                 warning_memory_mb: float = 2000.0,
                 restart_cooldown_sec: float = 300.0):
        """
        Initialize worker health monitor.
        
        Args:
            max_memory_mb: Maximum allowed memory in MB
            warning_memory_mb: Warning threshold in MB
            restart_cooldown_sec: Cooldown period between restarts
        """
        self.max_memory_bytes = int(max_memory_mb * 1024 * 1024)
        self.warning_memory_bytes = int(warning_memory_mb * 1024 * 1024)
        self.restart_cooldown_sec = restart_cooldown_sec
        
        self._leak_detector = MemoryLeakDetector()
        self._last_restart_time = 0.0
        self._restart_requested = False
        self._lock = threading.Lock()
        
        # Register leak callback
        self._leak_detector.register_leak_callback(self._on_leak_detected)
    
    def start(self):
        """Start health monitoring."""
        self._leak_detector.start()
    
    def stop(self):
        """Stop health monitoring."""
        self._leak_detector.stop()
    
    def _on_leak_detected(self, report: LeakReport):
        """Handle leak detection."""
        if report.confidence > 0.7:
            self.request_restart("High-confidence leak detected")
    
    def request_restart(self, reason: str = ""):
        """Request a graceful worker restart."""
        with self._lock:
            current_time = time.time()
            
            # Check cooldown
            if current_time - self._last_restart_time < self.restart_cooldown_sec:
                return False
            
            self._restart_requested = True
            return True
    
    def should_restart(self) -> Tuple[bool, str]:
        """Check if restart is needed."""
        with self._lock:
            # Check memory usage
            usage = self._leak_detector.get_current_usage()
            
            if usage['current_bytes'] > self.max_memory_bytes:
                return True, "Memory exceeded maximum threshold"
            
            if self._restart_requested:
                self._restart_requested = False
                self._last_restart_time = time.time()
                return True, "Restart requested due to leak detection"
            
            return False, ""
    
    def get_state(self) -> MemoryState:
        """Get current memory state."""
        usage = self._leak_detector.get_current_usage()
        current = usage['current_bytes']
        
        if current > self.max_memory_bytes:
            return MemoryState.CRITICAL
        
        if current > self.warning_memory_bytes:
            return MemoryState.WARNING
        
        stats = self._leak_detector.get_statistics()
        if stats['leak_detections'] > 0:
            return MemoryState.LEAK_DETECTED
        
        return MemoryState.HEALTHY
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive diagnostics."""
        return {
            'state': self.get_state().value,
            'memory_usage': self._leak_detector.get_current_usage(),
            'statistics': self._leak_detector.get_statistics(),
            'restart_pending': self._restart_requested,
            'time_since_last_restart': time.time() - self._last_restart_time
        }


# Module-level singleton
_health_monitor: Optional[WorkerHealthMonitor] = None
_lock = threading.Lock()


def get_health_monitor() -> WorkerHealthMonitor:
    """Get or create global health monitor."""
    global _health_monitor
    
    with _lock:
        if _health_monitor is None:
            _health_monitor = WorkerHealthMonitor()
        return _health_monitor


def start_monitoring():
    """Start memory monitoring."""
    get_health_monitor().start()


def stop_monitoring():
    """Stop memory monitoring."""
    get_health_monitor().stop()


def check_restart_needed() -> Tuple[bool, str]:
    """Check if worker restart is needed."""
    return get_health_monitor().should_restart()


def get_memory_diagnostics() -> Dict[str, Any]:
    """Get memory diagnostics."""
    return get_health_monitor().get_diagnostics()


# Module exports
__all__ = [
    'MemoryState',
    'MemorySnapshot',
    'MemoryDiff',
    'LeakReport',
    'MemoryLeakDetector',
    'WorkerHealthMonitor',
    'get_health_monitor',
    'start_monitoring',
    'stop_monitoring',
    'check_restart_needed',
    'get_memory_diagnostics'
]
