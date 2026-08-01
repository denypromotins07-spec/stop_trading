"""
Memory Tracer - Lightweight tracemalloc wrapper for identifying memory leaks.
Designed for 24/7 Ray worker and Nautilus adapter monitoring with minimal overhead.
"""

import asyncio
import logging
import tracemalloc
import gc
import time
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from collections import deque
from pathlib import Path
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    """Point-in-time memory snapshot."""
    timestamp: float
    current_mb: float
    peak_mb: float
    top_allocations: List[Tuple[str, int]]
    gc_counts: Tuple[int, int, int]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "current_mb": self.current_mb,
            "peak_mb": self.peak_mb,
            "top_allocations": [
                {"location": loc, "size_kb": size/1024} 
                for loc, size in self.top_allocations[:10]
            ],
            "gc_counts": list(self.gc_counts)
        }


@dataclass
class MemoryLeakAlert:
    """Alert for detected memory leak."""
    timestamp: float
    growth_rate_mb_per_hour: float
    current_usage_mb: float
    projected_exhaustion_hours: Optional[float]
    suspect_locations: List[str]
    severity: str  # 'low', 'medium', 'high', 'critical'
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "growth_rate_mb_per_hour": self.growth_rate_mb_per_hour,
            "current_usage_mb": self.current_usage_mb,
            "projected_exhaustion_hours": self.projected_exhaustion_hours,
            "suspect_locations": self.suspect_locations,
            "severity": self.severity
        }


class MemoryTracer:
    """
    Lightweight memory tracer using tracemalloc.
    Monitors memory usage and detects leaks with minimal overhead.
    """
    
    def __init__(self,
                 sample_interval: float = 60.0,
                 leak_threshold_mb_per_hour: float = 50.0,
                 critical_threshold_mb_per_hour: float = 200.0,
                 max_memory_mb: float = 3000.0,  # 3GB limit
                 snapshot_history: int = 100):
        """
        Initialize memory tracer.
        
        Args:
            sample_interval: Seconds between memory samples
            leak_threshold_mb_per_hour: Growth rate threshold for leak alert
            critical_threshold_mb_per_hour: Critical leak threshold
            max_memory_mb: Maximum allowed memory (enforce 3GB limit)
            snapshot_history: Number of snapshots to keep
        """
        self.sample_interval = sample_interval
        self.leak_threshold_mb_per_hour = leak_threshold_mb_per_hour
        self.critical_threshold_mb_per_hour = critical_threshold_mb_per_hour
        self.max_memory_mb = max_memory_mb
        self.snapshot_history = snapshot_history
        
        # Storage
        self._snapshots: deque = deque(maxlen=snapshot_history)
        self._alerts: deque = deque(maxlen=100)
        
        # Leak detection
        self._baseline_memory: Optional[float] = None
        self._leak_detected = False
        self._last_alert_time: float = 0.0
        
        # Monitoring state
        self._is_tracing = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._start_time: float = 0.0
    
    def start_tracing(self, nframes: int = 10):
        """Start memory tracing."""
        if not tracemalloc.is_tracing():
            tracemalloc.start(nframes)
            logger.info(f"tracemalloc started with {nframes} frames")
        
        self._is_tracing = True
        self._start_time = time.time()
        
        # Take baseline snapshot
        self._take_snapshot()
    
    def stop_tracing(self):
        """Stop memory tracing."""
        if tracemalloc.is_tracing():
            tracemalloc.stop()
            logger.info("tracemalloc stopped")
        
        self._is_tracing = False
    
    def _take_snapshot(self) -> Optional[MemorySnapshot]:
        """Take a memory snapshot."""
        if not tracemalloc.is_tracing():
            return None
        
        current, peak = tracemalloc.get_traced_memory()
        current_mb = current / (1024 * 1024)
        peak_mb = peak / (1024 * 1024)
        
        # Get top allocations
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')[:20]
        
        top_allocations = [
            (str(stat.traceback), stat.size)
            for stat in top_stats[:10]
        ]
        
        # GC counts
        gc_counts = gc.get_count()
        
        mem_snapshot = MemorySnapshot(
            timestamp=time.time(),
            current_mb=current_mb,
            peak_mb=peak_mb,
            top_allocations=top_allocations,
            gc_counts=gc_counts
        )
        
        self._snapshots.append(mem_snapshot)
        
        # Set baseline on first snapshot
        if self._baseline_memory is None:
            self._baseline_memory = current_mb
            logger.info(f"Memory baseline set: {current_mb:.1f} MB")
        
        return mem_snapshot
    
    def _detect_leak(self) -> Optional[MemoryLeakAlert]:
        """Analyze snapshots for memory leaks."""
        if len(self._snapshots) < 10:
            return None
        
        recent = list(self._snapshots)[-10:]
        
        # Calculate growth rate
        first = recent[0]
        last = recent[-1]
        
        time_diff_hours = (last.timestamp - first.timestamp) / 3600
        
        if time_diff_hours < 0.01:  # Less than ~36 seconds
            return None
        
        memory_diff = last.current_mb - first.current_mb
        growth_rate = memory_diff / time_diff_hours
        
        if growth_rate < self.leak_threshold_mb_per_hour:
            return None
        
        # Determine severity
        if growth_rate > self.critical_threshold_mb_per_hour:
            severity = "critical"
        elif growth_rate > self.leak_threshold_mb_per_hour * 2:
            severity = "high"
        else:
            severity = "medium"
        
        # Find suspect locations
        suspect_locations = []
        if last.top_allocations:
            for loc, size in last.top_allocations[:5]:
                if size > 1024 * 1024:  # > 1MB
                    suspect_locations.append(loc.split('\\n')[-1][:100])
        
        # Project exhaustion
        remaining_mb = self.max_memory_mb - last.current_mb
        projected_hours = remaining_mb / growth_rate if growth_rate > 0 else None
        
        alert = MemoryLeakAlert(
            timestamp=time.time(),
            growth_rate_mb_per_hour=growth_rate,
            current_usage_mb=last.current_mb,
            projected_exhaustion_hours=projected_hours,
            suspect_locations=suspect_locations,
            severity=severity
        )
        
        self._alerts.append(alert)
        self._leak_detected = True
        
        log_level = logging.CRITICAL if severity == "critical" else logging.WARNING
        logger.log(
            log_level,
            f"Memory leak detected: {growth_rate:.1f} MB/hour, "
            f"current: {last.current_mb:.1f} MB, "
            f"severity: {severity}"
        )
        
        return alert
    
    async def run_monitor_loop(self, stop_event: asyncio.Event):
        """Run the async monitoring loop."""
        self.start_tracing()
        logger.info("Memory tracer started")
        
        while not stop_event.is_set():
            try:
                # Take snapshot
                snapshot = self._take_snapshot()
                
                # Check for leaks periodically
                if len(self._snapshots) % 5 == 0:
                    self._detect_leak()
                
                # Force GC if memory is high
                if snapshot and snapshot.current_mb > self.max_memory_mb * 0.9:
                    logger.warning("Memory near limit, forcing GC")
                    gc.collect()
                
                await asyncio.sleep(self.sample_interval)
                
            except Exception as e:
                logger.error(f"Memory tracer error: {e}")
                await asyncio.sleep(self.sample_interval)
        
        self.stop_tracing()
        logger.info("Memory tracer stopped")
    
    def get_current_usage(self) -> Dict[str, float]:
        """Get current memory usage."""
        if not tracemalloc.is_tracing():
            return {"current_mb": 0, "peak_mb": 0}
        
        current, peak = tracemalloc.get_traced_memory()
        return {
            "current_mb": current / (1024 * 1024),
            "peak_mb": peak / (1024 * 1024),
            "limit_mb": self.max_memory_mb,
            "usage_pct": (current / (1024 * 1024)) / self.max_memory_mb * 100
        }
    
    def get_top_allocations(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get top memory allocations."""
        if not tracemalloc.is_tracing():
            return []
        
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('traceback')[:limit]
        
        return [
            {
                "traceback": str(stat.traceback)[:200],
                "size_kb": stat.size / 1024,
                "count": stat.count
            }
            for stat in top_stats
        ]
    
    def get_recent_alerts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent leak alerts."""
        return [a.to_dict() for a in list(self._alerts)[-limit:]]
    
    def get_tracer_stats(self) -> Dict[str, Any]:
        """Get tracer statistics."""
        usage = self.get_current_usage()
        
        return {
            "is_tracing": self._is_tracing,
            "uptime_seconds": time.time() - self._start_time if self._start_time else 0,
            "snapshots_taken": len(self._snapshots),
            "alerts_generated": len(self._alerts),
            "leak_detected": self._leak_detected,
            **usage
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Return tracer health status."""
        usage = self.get_current_usage()
        
        return {
            "running": self._is_tracing,
            "memory_usage_mb": usage.get("current_mb", 0),
            "memory_limit_mb": self.max_memory_mb,
            "usage_percentage": usage.get("usage_pct", 0),
            "alerts_pending": len(self._alerts),
            "near_limit": usage.get("current_mb", 0) > self.max_memory_mb * 0.8
        }
    
    def force_gc(self) -> Dict[str, int]:
        """Force garbage collection and return counts."""
        before = gc.get_count()
        collected = gc.collect()
        after = gc.get_count()
        
        return {
            "collected_objects": collected,
            "gen0_before": before[0],
            "gen1_before": before[1],
            "gen2_before": before[2],
            "gen0_after": after[0],
            "gen1_after": after[1],
            "gen2_after": after[2]
        }


# Module singleton
_memory_tracer: Optional[MemoryTracer] = None


def get_memory_tracer(**kwargs) -> MemoryTracer:
    """Get or create the global memory tracer."""
    global _memory_tracer
    
    if _memory_tracer is None:
        _memory_tracer = MemoryTracer(**kwargs)
        logger.info("Created memory tracer")
    
    return _memory_tracer


if __name__ == "__main__":
    # Test the memory tracer
    print("Testing Memory Tracer...")
    
    tracer = MemoryTracer(
        sample_interval=1.0,
        leak_threshold_mb_per_hour=100.0,
        max_memory_mb=500.0  # Lower limit for testing
    )
    
    # Start tracing
    tracer.start_tracing()
    
    # Simulate some allocations
    print("\nSimulating memory allocations...")
    
    data = []
    for i in range(10):
        # Allocate some memory
        chunk = bytearray(1024 * 1024 * 5)  # 5 MB each
        data.append(chunk)
        
        usage = tracer.get_current_usage()
        print(f"Iteration {i}: {usage['current_mb']:.1f} MB ({usage['usage_pct']:.1f}%)")
        
        time.sleep(0.5)
    
    # Get top allocations
    print("\nTop allocations:")
    for alloc in tracer.get_top_allocations(5):
        print(f"  {alloc['size_kb']:.1f} KB - {alloc['traceback'][:50]}...")
    
    # Force GC
    print(f"\nForcing GC: {tracer.force_gc()}")
    
    print(f"\nStats: {tracer.get_tracer_stats()}")
    print(f"Health: {tracer.health_check()}")
    
    tracer.stop_tracing()
