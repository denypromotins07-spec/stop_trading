"""
Chaos monkey that randomly terminates Ray inference workers.
Validates automatic respawning and state recovery logic.
Ensures ML ensemble maintains uptime even when processes are killed.
"""

from __future__ import annotations

import os
import signal
import random
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class ChaosConfig:
    """Configuration for chaos monkey."""
    # Probability of killing a worker per check interval
    kill_probability: float = 0.01
    
    # Minimum time between kills (seconds)
    min_kill_interval: float = 5.0
    
    # Maximum percentage of workers to kill at once
    max_kill_percentage: float = 0.25
    
    # Whether to kill in production (should be False)
    allow_production_kills: bool = False
    
    # Target worker types
    target_worker_types: List[str] = field(default_factory=lambda: ["inference", "optimization"])


@dataclass
class KillEvent:
    """Record of a worker kill event."""
    worker_id: str
    worker_type: str
    kill_method: str
    timestamp_ns: int
    respawned: bool = False
    recovery_time_ms: float = 0.0


class RayWorkerKiller:
    """
    Chaos monkey for Ray workers.
    Randomly terminates workers to test resilience.
    """
    
    def __init__(self, config: Optional[ChaosConfig] = None):
        self.config = config or ChaosConfig()
        self._kill_events: List[KillEvent] = []
        self._last_kill_time: float = 0.0
        self._total_kills = 0
        
        # Track worker PIDs
        self._tracked_workers: Dict[str, int] = {}
        
        logger.info("RayWorkerKiller initialized")
    
    def track_worker(self, worker_id: str, pid: int, worker_type: str = "inference"):
        """Track a worker for potential termination."""
        if worker_type in self.config.target_worker_types:
            self._tracked_workers[worker_id] = pid
            logger.debug(f"Tracking worker {worker_id} (PID: {pid})")
    
    def untrack_worker(self, worker_id: str):
        """Stop tracking a worker."""
        self._tracked_workers.pop(worker_id, None)
    
    def should_kill(self) -> bool:
        """Determine if a kill should occur based on probability and timing."""
        current_time = time.time()
        
        # Check minimum interval
        if current_time - self._last_kill_time < self.config.min_kill_interval:
            return False
        
        # Check probability
        return random.random() < self.kill_probability
    
    def kill_random_worker(self) -> Optional[KillEvent]:
        """Kill a random tracked worker."""
        if not self._tracked_workers:
            return None
        
        # Select random worker
        worker_id = random.choice(list(self._tracked_workers.keys()))
        pid = self._tracked_workers[worker_id]
        
        return self.kill_worker(worker_id, pid)
    
    def kill_worker(
        self,
        worker_id: str,
        pid: int,
        method: str = "sigkill"
    ) -> Optional[KillEvent]:
        """
        Kill a specific worker by PID.
        
        Args:
            worker_id: Worker identifier
            pid: Process ID
            method: Kill method (sigkill, sigterm, sigint)
            
        Returns:
            KillEvent if successful, None otherwise
        """
        try:
            # Choose signal
            if method == "sigkill":
                sig = signal.SIGKILL
            elif method == "sigterm":
                sig = signal.SIGTERM
            elif method == "sigint":
                sig = signal.SIGINT
            else:
                sig = signal.SIGKILL
            
            # Send signal
            os.kill(pid, sig)
            
            event = KillEvent(
                worker_id=worker_id,
                worker_type="unknown",
                kill_method=method,
                timestamp_ns=time.time_ns()
            )
            
            self._kill_events.append(event)
            self._total_kills += 1
            self._last_kill_time = time.time()
            
            logger.warning(f"Killed worker {worker_id} (PID: {pid}) with {method}")
            
            # Remove from tracking
            self._tracked_workers.pop(worker_id, None)
            
            return event
            
        except ProcessLookupError:
            logger.debug(f"Worker {worker_id} already dead (PID: {pid})")
            return None
        except PermissionError:
            logger.warning(f"No permission to kill worker {worker_id} (PID: {pid})")
            return None
        except Exception as e:
            logger.error(f"Failed to kill worker {worker_id}: {e}")
            return None
    
    def kill_percentage_of_workers(self, percentage: float) -> List[KillEvent]:
        """Kill a percentage of tracked workers."""
        n_to_kill = max(1, int(len(self._tracked_workers) * percentage))
        n_to_kill = min(n_to_kill, int(len(self._tracked_workers) * self.config.max_kill_percentage))
        
        events = []
        workers_to_kill = random.sample(
            list(self._tracked_workers.items()),
            min(n_to_kill, len(self._tracked_workers))
        )
        
        for worker_id, pid in workers_to_kill:
            event = self.kill_worker(worker_id, pid)
            if event:
                events.append(event)
        
        return events
    
    def run_chaos_loop(self, check_interval: float = 1.0):
        """
        Run continuous chaos loop.
        Should be run in a separate thread/process.
        """
        logger.info("Starting chaos loop")
        
        while True:
            try:
                if self.should_kill():
                    self.kill_random_worker()
                
                time.sleep(check_interval)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Chaos loop error: {e}")
                time.sleep(check_interval)
    
    def mark_respawned(self, worker_id: str, recovery_time_ms: float):
        """Mark a worker as having been respawned."""
        for event in reversed(self._kill_events):
            if event.worker_id == worker_id and not event.respawned:
                event.respawned = True
                event.recovery_time_ms = recovery_time_ms
                break
    
    def get_stats(self) -> Dict[str, Any]:
        """Get chaos statistics."""
        total_tracked = len(self._tracked_workers)
        total_kills = len(self._kill_events)
        respawned = sum(1 for e in self._kill_events if e.respawned)
        
        avg_recovery = (
            sum(e.recovery_time_ms for e in self._kill_events if e.respawned) / respawned
            if respawned > 0 else 0.0
        )
        
        return {
            'tracked_workers': total_tracked,
            'total_kills': total_kills,
            'respawned_count': respawned,
            'avg_recovery_time_ms': avg_recovery,
            'config': {
                'kill_probability': self.config.kill_probability,
                'min_interval': self.config.min_kill_interval
            }
        }
    
    def reset(self):
        """Reset all tracking and statistics."""
        self._tracked_workers.clear()
        self._kill_events.clear()
        self._total_kills = 0
        self._last_kill_time = 0.0


def create_chaos_killer(config: Optional[ChaosConfig] = None) -> RayWorkerKiller:
    """Factory function to create killer."""
    return RayWorkerKiller(config)
