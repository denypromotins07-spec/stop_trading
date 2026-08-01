#!/usr/bin/env python3
"""
Telemetry Module Root - Stage 50
Manages final rendering loop on dedicated thread without exceeding 100MB RAM.
"""

import os
import sys
import logging
from datetime import datetime
from typing import Dict, Optional, Any
from pathlib import Path
import threading
import queue
import gc
import psutil

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
)
logger = logging.getLogger('TelMod')

# Constants
MAX_TELEMETRY_RAM_MB = 100
RENDER_THREAD_NAME = "telemetry_render"
MEMORY_CHECK_INTERVAL_SEC = 30


class MemoryMonitor:
    """Monitors telemetry memory usage."""
    
    def __init__(self, max_memory_mb: float = MAX_TELEMETRY_RAM_MB):
        self.max_memory_mb = max_memory_mb
        self.process = psutil.Process(os.getpid())
        self.baseline_memory_mb = 0
        self.peak_memory_mb = 0
    
    def set_baseline(self):
        """Set baseline memory after initialization."""
        self.baseline_memory_mb = self._get_current_memory()
        logger.info(f"Telemetry baseline memory: {self.baseline_memory_mb:.1f}MB")
    
    def _get_current_memory(self) -> float:
        """Get current process memory in MB."""
        try:
            return self.process.memory_info().rss / (1024 * 1024)
        except:
            return 0
    
    def check_memory(self) -> bool:
        """Check if memory usage is within limits."""
        current = self._get_current_memory()
        self.peak_memory_mb = max(self.peak_memory_mb, current)
        
        delta = current - self.baseline_memory_mb
        
        if delta > self.max_memory_mb:
            logger.warning(
                f"Telemetry memory exceeded: {delta:.1f}MB delta "
                f"(limit: {self.max_memory_mb}MB)"
            )
            return False
        
        return True
    
    def get_stats(self) -> Dict:
        """Get memory statistics."""
        current = self._get_current_memory()
        return {
            'baseline_mb': self.baseline_memory_mb,
            'current_mb': current,
            'peak_mb': self.peak_memory_mb,
            'delta_mb': current - self.baseline_memory_mb,
            'within_limit': self.check_memory()
        }


class RenderLoopManager:
    """Manages the telemetry rendering loop on dedicated thread."""
    
    def __init__(self):
        self.render_thread: Optional[threading.Thread] = None
        self.running = False
        self.frame_queue = queue.Queue(maxsize=60)  # 60 frames buffer
        self.current_frame: Dict = {}
        self.fps_target = 4
        self.actual_fps = 0
        self.frame_count = 0
        self.last_fps_check = datetime.now()
        self.memory_monitor = MemoryMonitor()
    
    def start(self, render_callback):
        """Start render loop on dedicated thread."""
        self.running = True
        self.memory_monitor.set_baseline()
        
        self.render_thread = threading.Thread(
            target=self._render_loop,
            args=(render_callback,),
            name=RENDER_THREAD_NAME,
            daemon=True
        )
        self.render_thread.start()
        
        logger.info(f"Render loop started on thread: {RENDER_THREAD_NAME}")
    
    def _render_loop(self, render_callback):
        """Main render loop."""
        import time
        
        frame_interval = 1.0 / self.fps_target
        last_frame_time = time.time()
        
        while self.running:
            try:
                current_time = time.time()
                
                # Calculate FPS
                elapsed = current_time - last_frame_time
                if elapsed >= 1.0:
                    self.actual_fps = self.frame_count / elapsed
                    self.frame_count = 0
                    last_frame_time = current_time
                
                # Get latest frame data
                try:
                    frame_data = self.frame_queue.get_nowait()
                    self.current_frame = frame_data
                except queue.Empty:
                    pass
                
                # Render frame
                if render_callback and self.current_frame:
                    render_callback(self.current_frame)
                
                self.frame_count += 1
                
                # Sleep to maintain target FPS
                sleep_time = frame_interval - (time.time() - current_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                # Periodic memory check
                if self.frame_count % (self.fps_target * MEMORY_CHECK_INTERVAL_SEC) == 0:
                    self._check_and_gc()
            
            except Exception as e:
                logger.error(f"Render loop error: {e}")
                time.sleep(0.1)
    
    def update_frame(self, frame_data: Dict):
        """Update frame data for rendering."""
        try:
            # Non-blocking put, drop old frames if queue full
            if not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except:
                    pass
            
            self.frame_queue.put_nowait(frame_data)
        except queue.Full:
            pass  # Drop frame if still full
    
    def _check_and_gc(self):
        """Check memory and trigger GC if needed."""
        stats = self.memory_monitor.get_stats()
        
        if not stats['within_limit']:
            logger.warning("Triggering garbage collection...")
            gc.collect()
            
            # Check again after GC
            stats = self.memory_monitor.get_stats()
            if not stats['within_limit']:
                logger.error("Memory still exceeded after GC!")
    
    def get_status(self) -> Dict:
        """Get render loop status."""
        return {
            'running': self.running,
            'fps_target': self.fps_target,
            'actual_fps': round(self.actual_fps, 1),
            'frame_queue_size': self.frame_queue.qsize(),
            'memory': self.memory_monitor.get_stats()
        }
    
    def stop(self):
        """Stop render loop."""
        self.running = False
        
        if self.render_thread:
            self.render_thread.join(timeout=2.0)
        
        logger.info("Render loop stopped")


class TelemetryModuleCoordinator:
    """Coordinates all telemetry operations."""
    
    def __init__(self):
        self.render_manager = RenderLoopManager()
        self.data_aggregator = {}
        self.subscribers = []
    
    def register_data_source(self, name: str, source_callback):
        """Register a data source for aggregation."""
        self.data_aggregator[name] = source_callback
        logger.info(f"Registered telemetry source: {name}")
    
    def subscribe(self, callback):
        """Subscribe to telemetry updates."""
        self.subscribers.append(callback)
    
    def _aggregate_and_render(self, frame_data: Dict):
        """Aggregate data and notify subscribers."""
        # Aggregate from all sources
        aggregated = {
            'timestamp': datetime.now().isoformat(),
            'sources': {}
        }
        
        for name, callback in self.data_aggregator.items():
            try:
                aggregated['sources'][name] = callback()
            except Exception as e:
                logger.error(f"Error getting data from {name}: {e}")
        
        # Merge with frame data
        aggregated.update(frame_data)
        
        # Notify subscribers
        for callback in self.subscribers:
            try:
                callback(aggregated)
            except Exception as e:
                logger.error(f"Subscriber error: {e}")
    
    def start(self):
        """Start telemetry module."""
        logger.info("Starting telemetry module...")
        self.render_manager.start(self._aggregate_and_render)
    
    def update(self, **kwargs):
        """Update telemetry data."""
        self.render_manager.update_frame(kwargs)
    
    def get_status(self) -> Dict:
        """Get module status."""
        return self.render_manager.get_status()
    
    def stop(self):
        """Stop telemetry module."""
        self.render_manager.stop()


def create_telemetry_module() -> TelemetryModuleCoordinator:
    """Factory function to create telemetry coordinator."""
    return TelemetryModuleCoordinator()


def main():
    """Entry point for telemetry module testing."""
    import time
    
    coordinator = create_telemetry_module()
    
    # Register dummy data source
    coordinator.register_data_source('system', lambda: {
        'cpu_percent': psutil.cpu_percent(),
        'memory_percent': psutil.virtual_memory().percent
    })
    
    # Subscribe with printer
    def printer(data):
        print(f"\rTelemetry: CPU={data['sources'].get('system', {}).get('cpu_percent', 0):.1f}% "
              f"FPS={coordinator.render_manager.actual_fps:.1f}", end="", flush=True)
    
    coordinator.subscribe(printer)
    coordinator.start()
    
    try:
        # Simulate updates
        for i in range(100):
            coordinator.update(
                test_value=i,
                timestamp=datetime.now().isoformat()
            )
            time.sleep(0.25)
        
        print(f"\n\nFinal Status: {coordinator.get_status()}")
    
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        coordinator.stop()


if __name__ == '__main__':
    main()
