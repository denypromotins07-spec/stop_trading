"""
Reinforcement Learning Module Root
Connects RL inference server to Nautilus execution router via lock-free queues.
Manages PPO agent lifecycle and action routing.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
import threading
import queue
import time
import logging

from .ppo_agent import PPOInferenceServer, PPOConfig, PolicyAction
from .execution_env import ExecutionEnv, ExecutionConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RLActionRequest:
    """Request for RL action."""
    request_id: str
    observation: np.ndarray
    instrument_id: str
    timestamp: float = field(default_factory=time.time)
    priority: int = 0


@dataclass
class RLActionResponse:
    """Response containing RL action."""
    request_id: str
    action: Optional[PolicyAction]
    instrument_id: str
    latency_ms: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class RLConfig:
    """Configuration for RL module."""
    ppo_checkpoint_path: str = ""
    n_inference_workers: int = 2
    max_queue_size: int = 1000
    default_participation_rate: float = 0.3
    default_aggression_level: float = 0.5
    enable_ray_serve: bool = False


class ReinforcementLearningModule:
    """
    Central module for reinforcement learning inference.
    Connects PPO server to Nautilus execution router.
    """
    
    def __init__(self, config: Optional[RLConfig] = None):
        self.config = config or RLConfig()
        
        # PPO inference server
        self._ppo_server: Optional[PPOInferenceServer] = None
        
        # Action request queue (lock-free using atomic operations where possible)
        self._request_queue: queue.PriorityQueue = queue.PriorityQueue(
            maxsize=self.config.max_queue_size
        )
        
        # Response cache
        self._responses: Dict[str, RLActionResponse] = {}
        self._responses_lock = threading.Lock()
        
        # Execution router callback
        self._execution_callback: Optional[Callable[[PolicyAction, str], None]] = None
        
        # Worker threads
        self._running = False
        self._workers: List[threading.Thread] = []
        
        # Statistics
        self._total_requests = 0
        self._total_actions = 0
        self._fallback_count = 0
    
    def initialize(self) -> bool:
        """Initialize the RL module and PPO server."""
        try:
            # Create PPO server
            ppo_config = PPOConfig(
                checkpoint_path=self.config.ppo_checkpoint_path,
                n_workers=self.config.n_inference_workers
            )
            
            self._ppo_server = PPOInferenceServer(ppo_config)
            
            if not self._ppo_server.initialize():
                logger.warning("PPO server initialization failed, using fallback mode")
                self._ppo_server = None
            
            self._running = True
            logger.info("RL Module initialized")
            return True
        
        except Exception as e:
            logger.error(f"Failed to initialize RL module: {e}")
            return False
    
    def set_execution_callback(self, 
                               callback: Callable[[PolicyAction, str], None]) -> None:
        """
        Set callback for routing actions to execution engine.
        
        Args:
            callback: Function(action, instrument_id)
        """
        self._execution_callback = callback
    
    def request_action(self,
                       observation: np.ndarray,
                       instrument_id: str,
                       priority: int = 0,
                       request_id: Optional[str] = None) -> str:
        """
        Request an action from the RL policy.
        
        Args:
            observation: Environment observation
            instrument_id: Target instrument
            priority: Request priority
            request_id: Optional custom request ID
        
        Returns:
            Request ID
        """
        if request_id is None:
            request_id = f"rl_{instrument_id}_{time.time_ns()}"
        
        request = RLActionRequest(
            request_id=request_id,
            observation=observation,
            instrument_id=instrument_id,
            priority=-priority  # Negate for min-heap
        )
        
        try:
            self._request_queue.put_nowait((-priority, time.time(), request))
            self._total_requests += 1
            return request_id
        except queue.Full:
            logger.warning("RL request queue full")
            return ""
    
    def get_action(self,
                   observation: np.ndarray,
                   instrument_id: str,
                   timeout_ms: float = 10.0) -> Optional[PolicyAction]:
        """
        Synchronously get an action (blocking).
        
        Args:
            observation: Environment observation
            instrument_id: Target instrument
            timeout_ms: Maximum wait time
        
        Returns:
            PolicyAction or None
        """
        request_id = self.request_action(observation, instrument_id)
        
        if not request_id:
            return self._get_fallback_action(observation, instrument_id)
        
        start_time = time.time()
        
        while True:
            with self._responses_lock:
                if request_id in self._responses:
                    response = self._responses.pop(request_id)
                    if response.action is not None:
                        return response.action
            
            if (time.time() - start_time) * 1000 > timeout_ms:
                return self._get_fallback_action(observation, instrument_id)
            
            time.sleep(0.001)
    
    def _get_fallback_action(self, 
                             observation: np.ndarray,
                             instrument_id: str) -> PolicyAction:
        """Get fallback action when RL is unavailable."""
        self._fallback_count += 1
        
        # Simple heuristic based on volatility
        volatility = observation[3] if len(observation) > 3 else 0.0005
        
        return PolicyAction(
            participation_rate=self.config.default_participation_rate,
            aggression_level=self.config.default_aggression_level,
            position_size_multiplier=min(1.0 / (volatility * 1000 + 0.1), 2.0),
            confidence=0.3,
            metadata={
                "source": "fallback",
                "instrument_id": instrument_id,
                "fallback_reason": "rl_unavailable"
            }
        )
    
    def _worker_loop(self, worker_id: int) -> None:
        """Worker thread for processing action requests."""
        logger.info(f"RL worker {worker_id} started")
        
        while self._running:
            try:
                _, _, request = self._request_queue.get(timeout=0.1)
                
                start_time = time.perf_counter()
                
                # Get action from PPO server
                if self._ppo_server is not None and self._ppo_server.is_initialized:
                    action = self._ppo_server.get_action(request.observation)
                else:
                    action = self._get_fallback_action(
                        request.observation, 
                        request.instrument_id
                    )
                
                latency_ms = (time.perf_counter() - start_time) * 1000
                
                # Create response
                response = RLActionResponse(
                    request_id=request.request_id,
                    action=action,
                    instrument_id=request.instrument_id,
                    latency_ms=latency_ms
                )
                
                # Cache response
                with self._responses_lock:
                    self._responses[request.request_id] = response
                
                # Route to execution if callback set
                if action is not None and self._execution_callback is not None:
                    try:
                        self._execution_callback(action, request.instrument_id)
                        self._total_actions += 1
                    except Exception as e:
                        logger.error(f"Execution callback error: {e}")
                
                self._request_queue.task_done()
            
            except queue.Empty:
                pass
            except Exception as e:
                logger.error(f"RL worker {worker_id} error: {e}")
        
        logger.info(f"RL worker {worker_id} stopped")
    
    def start_workers(self, n_workers: Optional[int] = None) -> None:
        """Start worker threads."""
        if not self._running:
            return
        
        n_workers = n_workers or max(1, self.config.n_inference_workers)
        
        for i in range(n_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                args=(i,),
                daemon=True,
                name=f"RL_Worker_{i}"
            )
            worker.start()
            self._workers.append(worker)
        
        logger.info(f"Started {n_workers} RL workers")
    
    def stop_workers(self) -> None:
        """Stop all workers."""
        self._running = False
        
        for worker in self._workers:
            worker.join(timeout=2.0)
        
        self._workers.clear()
        logger.info("Stopped all RL workers")
    
    def shutdown(self) -> None:
        """Shutdown the module."""
        self.stop_workers()
        
        if self._ppo_server is not None:
            self._ppo_server.shutdown()
        
        logger.info("RL Module shutdown")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get module statistics."""
        stats = {
            "total_requests": self._total_requests,
            "total_actions": self._total_actions,
            "fallback_count": self._fallback_count,
            "queue_size": self._request_queue.qsize(),
            "active_workers": len(self._workers),
            "ppo_server_initialized": (
                self._ppo_server.is_initialized 
                if self._ppo_server else False
            ),
        }
        
        if self._ppo_server is not None:
            stats["ppo_statistics"] = self._ppo_server.get_statistics()
        
        return stats
    
    @property
    def is_running(self) -> bool:
        return self._running


# Global module instance (singleton pattern)
_rl_module: Optional[ReinforcementLearningModule] = None
_module_lock = threading.Lock()


def get_rl_module(config: Optional[RLConfig] = None) -> ReinforcementLearningModule:
    """
    Get or create the global RL module instance.
    
    Args:
        config: Module configuration
    
    Returns:
        ReinforcementLearningModule instance
    """
    global _rl_module
    
    with _module_lock:
        if _rl_module is None:
            _rl_module = ReinforcementLearningModule(config)
        
        return _rl_module


def reset_rl_module() -> None:
    """Reset the global module instance."""
    global _rl_module
    
    with _module_lock:
        if _rl_module is not None:
            _rl_module.shutdown()
            _rl_module = None


if __name__ == "__main__":
    print("RL Module Demo")
    print("=" * 40)
    
    # Initialize module
    config = RLConfig(
        ppo_checkpoint_path="",
        n_inference_workers=2
    )
    
    module = get_rl_module(config)
    
    if not module.initialize():
        print("Failed to initialize RL module")
        exit(1)
    
    # Set execution callback
    def mock_execution_callback(action: PolicyAction, instrument_id: str):
        print(f"Executing: {instrument_id}")
        print(f"  Participation: {action.participation_rate:.4f}")
        print(f"  Aggression: {action.aggression_level:.4f}")
    
    module.set_execution_callback(mock_execution_callback)
    
    # Start workers
    module.start_workers()
    
    # Test action request
    test_observation = np.array([
        0.5, 0.3, 1.0, 0.0005, 0.01, 0.0, 0.1, 0.5
    ], dtype=np.float32)
    
    action = module.get_action(test_observation, "BTC/USDT")
    
    if action:
        print(f"\nReceived action:")
        print(f"  Source: {action.metadata.get('source', 'unknown')}")
        print(f"  Confidence: {action.confidence:.4f}")
    
    # Get statistics
    stats = module.get_statistics()
    print(f"\nStatistics: {stats}")
    
    # Cleanup
    module.shutdown()
    reset_rl_module()
    print("\nRL Module demo complete")
