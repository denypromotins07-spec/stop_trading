"""
GIL Contention Tracker - Monitors Python GIL starvation using sys.setswitchinterval and thread dumps.
Non-blocking asynchronous checks to avoid stalling the Nautilus event loop during high-throughput trading.
"""

import asyncio
import logging
import sys
import threading
import time
import traceback
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import deque
import weakref

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GILContentionEvent:
    """Record of a GIL contention event."""
    timestamp: float
    thread_id: int
    thread_name: str
    wait_time_ms: float
    stack_trace: str
    is_critical: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "thread_id": self.thread_id,
            "thread_name": self.thread_name,
            "wait_time_ms": self.wait_time_ms,
            "stack_trace": self.stack_trace[:500] if len(self.stack_trace) > 500 else self.stack_trace,
            "is_critical": self.is_critical
        }


@dataclass
class ThreadSnapshot:
    """Snapshot of thread state."""
    thread_id: int
    thread_name: str
    is_alive: bool
    is_daemon: bool
    stack_frames: List[str]
    gil_holding: bool = False


class GILMonitor:
    """
    Non-blocking GIL contention monitor.
    
    Uses async checks and periodic sampling to detect thread starvation
    without impacting the main trading event loop.
    """
    
    def __init__(self,
                 check_interval: float = 0.1,
                 contention_threshold_ms: float = 10.0,
                 critical_threshold_ms: float = 50.0,
                 max_history: int = 1000):
        """
        Initialize GIL monitor.
        
        Args:
            check_interval: Seconds between GIL checks
            contention_threshold_ms: Wait time threshold for contention event
            critical_threshold_ms: Wait time for critical contention
            max_history: Maximum events to keep in history
        """
        self.check_interval = check_interval
        self.contention_threshold_ms = contention_threshold_ms
        self.critical_threshold_ms = critical_threshold_ms
        self.max_history = max_history
        
        # Track thread wait times
        self._thread_last_active: Dict[int, float] = {}
        self._contention_events: deque = deque(maxlen=max_history)
        
        # Statistics
        self._total_checks = 0
        self._contention_count = 0
        self._critical_count = 0
        self._max_wait_time_ms = 0.0
        self._avg_wait_time_ms = 0.0
        
        # Monitoring state
        self._is_running = False
        self._monitor_task: Optional[asyncio.Task] = None
        
        # Original switch interval
        self._original_switch_interval = sys.getswitchinterval()
    
    def start_monitoring(self):
        """Start GIL monitoring with adjusted switch interval."""
        # Reduce switch interval for more responsive GIL handling
        # But not too low to cause excessive thrashing
        try:
            sys.setswitchinterval(0.005)  # 5ms default is often too high
            logger.info(f"Adjusted GIL switch interval to {sys.getswitchinterval()}s")
        except Exception as e:
            logger.warning(f"Could not adjust switch interval: {e}")
        
        self._is_running = True
    
    def stop_monitoring(self):
        """Stop monitoring and restore original settings."""
        self._is_running = False
        
        # Restore original switch interval
        try:
            sys.setswitchinterval(self._original_switch_interval)
            logger.info(f"Restored GIL switch interval to {self._original_switch_interval}s")
        except Exception as e:
            logger.warning(f"Could not restore switch interval: {e}")
    
    def record_thread_activity(self, thread: Optional[threading.Thread] = None):
        """Record that a thread was active (call from each thread periodically)."""
        if thread is None:
            thread = threading.current_thread()
        
        now = time.time()
        thread_id = thread.ident
        
        if thread_id in self._thread_last_active:
            # Calculate wait time since last activity
            last_active = self._thread_last_active[thread_id]
            wait_time_s = now - last_active
            
            # Check for contention
            wait_time_ms = wait_time_s * 1000
            
            if wait_time_ms > self.contention_threshold_ms:
                is_critical = wait_time_ms > self.critical_threshold_ms
                
                # Capture stack trace non-blockingly
                try:
                    stack = self._get_thread_stack(thread_id)
                except:
                    stack = "Stack capture failed"
                
                event = GILContentionEvent(
                    timestamp=now,
                    thread_id=thread_id,
                    thread_name=thread.name,
                    wait_time_ms=wait_time_ms,
                    stack_trace=stack,
                    is_critical=is_critical
                )
                
                self._contention_events.append(event)
                self._contention_count += 1
                
                if is_critical:
                    self._critical_count += 1
                    logger.warning(
                        f"CRITICAL GIL contention: {thread.name} waited {wait_time_ms:.1f}ms"
                    )
                elif self._contention_count % 10 == 0:
                    logger.info(f"GIL contention detected: {thread.name} waited {wait_time_ms:.1f}ms")
            
            # Update statistics
            self._max_wait_time_ms = max(self._max_wait_time_ms, wait_time_ms)
            n = self._contention_count
            self._avg_wait_time_ms = ((n-1) * self._avg_wait_time_ms + wait_time_ms) / n
        
        self._thread_last_active[thread_id] = now
        self._total_checks += 1
    
    def _get_thread_stack(self, thread_id: int) -> str:
        """Get stack trace for a thread (non-blocking)."""
        frames = sys._current_frames()
        
        if thread_id in frames:
            return ''.join(traceback.format_stack(frames[thread_id]))
        
        return "No frames available"
    
    def get_thread_snapshots(self) -> List[ThreadSnapshot]:
        """Get snapshots of all threads (call sparingly)."""
        snapshots = []
        frames = sys._current_frames()
        
        for thread in threading.enumerate():
            stack_frames = []
            
            if thread.ident in frames:
                try:
                    stack_frames = [
                        line.strip() 
                        for line in traceback.format_stack(frames[thread.ident])
                    ]
                except:
                    pass
            
            snapshots.append(ThreadSnapshot(
                thread_id=thread.ident,
                thread_name=thread.name,
                is_alive=thread.is_alive(),
                is_daemon=thread.daemon,
                stack_frames=stack_frames[-10:],  # Last 10 frames
                gil_holding=False  # Would need more sophisticated detection
            ))
        
        return snapshots
    
    async def run_monitor_loop(self, stop_event: asyncio.Event):
        """Run the async monitoring loop."""
        self.start_monitoring()
        logger.info("GIL monitor started")
        
        while not stop_event.is_set():
            try:
                # Record activity for current thread
                self.record_thread_activity()
                
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"GIL monitor error: {e}")
                await asyncio.sleep(self.check_interval)
        
        self.stop_monitoring()
        logger.info("GIL monitor stopped")
    
    def get_contention_events(self, limit: int = 50, 
                               critical_only: bool = False) -> List[Dict[str, Any]]:
        """Get recent contention events."""
        events = list(self._contention_events)[-limit:]
        
        if critical_only:
            events = [e for e in events if e.is_critical]
        
        return [e.to_dict() for e in events]
    
    def get_monitor_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        return {
            "total_checks": self._total_checks,
            "contention_events": self._contention_count,
            "critical_events": self._critical_count,
            "max_wait_ms": self._max_wait_time_ms,
            "avg_wait_ms": self._avg_wait_time_ms,
            "current_switch_interval": sys.getswitchinterval(),
            "threads_active": len(self._thread_last_active),
            "is_running": self._is_running
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Return monitor health status."""
        return {
            "running": self._is_running,
            "contention_rate": self._contention_count / max(self._total_checks, 1),
            "events_cached": len(self._contention_events),
            "critical_pending": sum(1 for e in self._contention_events if e.is_critical)
        }


# Module singleton
_gil_monitor: Optional[GILMonitor] = None


def get_gil_monitor(**kwargs) -> GILMonitor:
    """Get or create the global GIL monitor."""
    global _gil_monitor
    
    if _gil_monitor is None:
        _gil_monitor = GILMonitor(**kwargs)
        logger.info("Created GIL monitor")
    
    return _gil_monitor


def record_activity():
    """Convenience function for threads to record activity."""
    if _gil_monitor is not None:
        _gil_monitor.record_thread_activity()


if __name__ == "__main__":
    # Test the GIL monitor
    print("Testing GIL Monitor...")
    
    monitor = GILMonitor(
        check_interval=0.1,
        contention_threshold_ms=5.0,
        critical_threshold_ms=20.0
    )
    
    # Simulate some thread activity
    print("\nSimulating thread activity...")
    
    for i in range(20):
        monitor.record_thread_activity()
        time.sleep(0.05)  # Normal activity
        
        # Simulate occasional contention
        if i == 10:
            time.sleep(0.03)  # Slight delay
    
    # Simulate critical contention
    print("\nSimulating critical contention...")
    monitor.record_thread_activity()
    time.sleep(0.1)  # Long delay = critical contention
    
    print(f"\nStatistics: {monitor.get_monitor_stats()}")
    print(f"Health: {monitor.health_check()}")
    
    # Get events
    events = monitor.get_contention_events(critical_only=True)
    print(f"\nCritical events: {len(events)}")
    for e in events[:3]:
        print(f"  Thread {e['thread_name']}: {e['wait_time_ms']:.1f}ms")
    
    # Get thread snapshots
    snapshots = monitor.get_thread_snapshots()
    print(f"\nActive threads: {len(snapshots)}")
    for s in snapshots[:3]:
        print(f"  {s.thread_name} (id={s.thread_id})")
