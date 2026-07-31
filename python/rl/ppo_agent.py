"""
Ray RLlib PPO Inference Server
Serves trained PPO policy via Ray Serve for dynamic position sizing and execution aggression.
Configured for inference-only operation to minimize CPU usage.
"""

import numpy as np
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import threading
import time
import logging

# Conditional imports for Ray
try:
    import ray
    from ray import serve
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

try:
    from ray.rllib.algorithms.ppo import PPO
    from ray.rllib.policy.sample_batch import SampleBatch
    RLLIB_AVAILABLE = True
except ImportError:
    RLLIB_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PPOConfig:
    """Configuration for PPO inference server."""
    checkpoint_path: str = ""
    n_workers: int = 2
    inference_batch_size: int = 32
    max_concurrent_queries: int = 100
    gpu_fraction: float = 0.0  # CPU-only by default
    memory_limit_mb: int = 512


@dataclass
class PolicyAction:
    """Action output from PPO policy."""
    participation_rate: float
    aggression_level: float
    position_size_multiplier: float
    confidence: float
    metadata: Dict[str, Any]


class PPOInferenceServer:
    """
    Ray Serve-based PPO inference server.
    Designed for inference-only operation with minimal overhead.
    """
    
    def __init__(self, config: Optional[PPOConfig] = None):
        self.config = config or PPOConfig()
        
        self._policy = None
        self._is_initialized = False
        self._lock = threading.Lock()
        
        # Statistics
        self._total_queries = 0
        self._total_errors = 0
        self._latencies: List[float] = []
    
    def initialize(self) -> bool:
        """Initialize Ray and load PPO policy."""
        if not RAY_AVAILABLE:
            logger.error("Ray not available. Install with: pip install ray")
            return False
        
        if not RLLIB_AVAILABLE:
            logger.error("RLlib not available. Install with: pip install ray[rllib]")
            return False
        
        try:
            # Initialize Ray (if not already)
            if not ray.is_initialized():
                ray.init(
                    num_cpus=self.config.n_workers,
                    include_dashboard=False,
                    log_to_driver=False,
                    _temp_dir="/tmp/ray_hft"
                )
            
            # Load policy from checkpoint
            if self.config.checkpoint_path:
                self._policy = PPO.from_checkpoint(self.config.checkpoint_path)
            else:
                # Create dummy policy for demonstration
                logger.warning("No checkpoint provided, using dummy policy")
                self._policy = None
            
            self._is_initialized = True
            logger.info("PPO Inference Server initialized")
            return True
        
        except Exception as e:
            logger.error(f"Failed to initialize PPO server: {e}")
            return False
    
    def get_action(self, observation: np.ndarray) -> Optional[PolicyAction]:
        """
        Get action from PPO policy for a single observation.
        
        Args:
            observation: Environment observation vector
        
        Returns:
            PolicyAction or None
        """
        if not self._is_initialized:
            logger.error("Server not initialized")
            return None
        
        start_time = time.perf_counter()
        
        try:
            if self._policy is None:
                # Return default action for demo
                action = self._get_default_action(observation)
            else:
                # Compute action using policy
                action_dict = self._policy.compute_single_action(observation)
                action = self._parse_action(action_dict, observation)
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            self._latencies.append(latency_ms)
            self._total_queries += 1
            
            return action
        
        except Exception as e:
            logger.error(f"Inference error: {e}")
            self._total_errors += 1
            return None
    
    def get_batch_actions(self, 
                          observations: np.ndarray) -> Optional[List[PolicyAction]]:
        """
        Get actions for a batch of observations.
        
        Args:
            observations: Batch of observations (batch_size, obs_dim)
        
        Returns:
            List of PolicyActions
        """
        if not self._is_initialized:
            return None
        
        actions = []
        for i in range(observations.shape[0]):
            action = self.get_action(observations[i])
            if action is not None:
                actions.append(action)
        
        return actions
    
    def _get_default_action(self, observation: np.ndarray) -> PolicyAction:
        """Generate default action when no policy is loaded."""
        # Simple heuristic based on observation
        # Observation: [norm_remaining_qty, norm_time, spread, volatility, ...]
        
        remaining_qty = observation[0] if len(observation) > 0 else 0.5
        volatility = observation[3] if len(observation) > 3 else 0.0005
        
        # Higher participation when more quantity remains
        participation_rate = min(0.5 + remaining_qty * 0.3, 1.0)
        
        # Lower aggression in high volatility
        aggression_level = max(0.3 - volatility * 100, 0.1)
        
        # Position sizing based on volatility inverse
        position_multiplier = min(1.0 / (volatility * 1000 + 0.1), 2.0)
        
        return PolicyAction(
            participation_rate=float(participation_rate),
            aggression_level=float(aggression_level),
            position_size_multiplier=float(position_multiplier),
            confidence=0.5,
            metadata={"source": "default_heuristic"}
        )
    
    def _parse_action(self, 
                      action_dict: Any,
                      observation: np.ndarray) -> PolicyAction:
        """Parse raw policy output into PolicyAction."""
        # Extract action components
        # Assuming action is [participation_rate, aggression_level]
        
        if isinstance(action_dict, np.ndarray):
            raw_action = action_dict
        elif hasattr(action_dict, 'action'):
            raw_action = action_dict.action
        else:
            raw_action = np.array([0.5, 0.5])
        
        participation_rate = float(np.clip(raw_action[0], 0.0, 1.0))
        aggression_level = float(np.clip(raw_action[1], 0.0, 1.0))
        
        # Calculate position size multiplier based on state
        volatility = observation[3] if len(observation) > 3 else 0.0005
        position_multiplier = float(min(1.0 / (volatility * 1000 + 0.1), 2.0))
        
        # Confidence from policy value function (if available)
        confidence = 0.5  # Default
        
        return PolicyAction(
            participation_rate=participation_rate,
            aggression_level=aggression_level,
            position_size_multiplier=position_multiplier,
            confidence=confidence,
            metadata={"source": "ppo_policy"}
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get server statistics."""
        latencies = np.array(self._latencies[-1000:])
        
        stats = {
            "total_queries": self._total_queries,
            "total_errors": self._total_errors,
            "is_initialized": self._is_initialized,
            "checkpoint_path": self.config.checkpoint_path,
        }
        
        if len(latencies) > 0:
            stats.update({
                "mean_latency_ms": float(np.mean(latencies)),
                "median_latency_ms": float(np.median(latencies)),
                "p95_latency_ms": float(np.percentile(latencies, 95)),
                "p99_latency_ms": float(np.percentile(latencies, 99)),
            })
        
        return stats
    
    def shutdown(self) -> None:
        """Shutdown the server."""
        if ray.is_initialized():
            ray.shutdown()
        logger.info("PPO Inference Server shutdown")
    
    @property
    def is_initialized(self) -> bool:
        return self._is_initialized


class RayServeDeployment:
    """
    Ray Serve deployment wrapper for PPO inference.
    Provides HTTP endpoint for remote inference requests.
    """
    
    def __init__(self, config: Optional[PPOConfig] = None):
        self.config = config or PPOConfig()
        self._deployment = None
        self._handle = None
    
    async def deploy(self) -> bool:
        """Deploy the PPO inference server via Ray Serve."""
        if not RAY_AVAILABLE:
            return False
        
        try:
            # Define Serve deployment
            @serve.deployment(
                num_replicas=1,
                ray_actor_options={
                    "num_cpus": 1,
                    "memory": self.config.memory_limit_mb * 1024 * 1024
                }
            )
            class PPORouter:
                def __init__(self, config_dict: Dict):
                    self.server = PPOInferenceServer(PPOConfig(**config_dict))
                    self.server.initialize()
                
                async def infer(self, observation: List[float]) -> Dict[str, Any]:
                    obs = np.array(observation, dtype=np.float32)
                    action = self.server.get_action(obs)
                    
                    if action is None:
                        return {"error": "Inference failed"}
                    
                    return {
                        "participation_rate": action.participation_rate,
                        "aggression_level": action.aggression_level,
                        "position_size_multiplier": action.position_size_multiplier,
                        "confidence": action.confidence,
                    }
                
                async def get_stats(self) -> Dict[str, Any]:
                    return self.server.get_statistics()
            
            # Deploy
            config_dict = {
                "checkpoint_path": self.config.checkpoint_path,
                "n_workers": self.config.n_workers,
                "inference_batch_size": self.config.inference_batch_size,
            }
            
            self._deployment = PPORouter.bind(config_dict)
            self._handle = serve.run(self._deployment)
            
            logger.info("Ray Serve deployment successful")
            return True
        
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            return False
    
    def undeploy(self) -> None:
        """Undeploy the service."""
        if serve.context._global_state is not None:
            serve.shutdown()
        logger.info("Ray Serve undeployed")


def create_ppo_server(config: Optional[PPOConfig] = None) -> PPOInferenceServer:
    """
    Factory function to create PPO inference server.
    
    Args:
        config: Server configuration
    
    Returns:
        PPOInferenceServer instance
    """
    return PPOInferenceServer(config)


if __name__ == "__main__":
    print("PPO Inference Server Demo")
    print("=" * 40)
    
    # Create server
    config = PPOConfig(
        checkpoint_path="",  # No checkpoint for demo
        n_workers=2
    )
    
    server = create_ppo_server(config)
    
    # Initialize
    if not server.initialize():
        print("Failed to initialize server")
        exit(1)
    
    # Test inference
    test_observation = np.array([
        0.5,   # normalized remaining qty
        0.3,   # normalized time
        1.0,   # spread (bps)
        0.0005,  # volatility
        0.01,  # momentum
        0.0,   # order imbalance
        0.1,   # fill rate
        0.5,   # avg slippage
    ], dtype=np.float32)
    
    action = server.get_action(test_observation)
    
    if action:
        print(f"\nAction received:")
        print(f"  Participation Rate: {action.participation_rate:.4f}")
        print(f"  Aggression Level: {action.aggression_level:.4f}")
        print(f"  Position Multiplier: {action.position_size_multiplier:.4f}")
        print(f"  Confidence: {action.confidence:.4f}")
    
    # Get statistics
    stats = server.get_statistics()
    print(f"\nStatistics: {stats}")
    
    # Cleanup
    server.shutdown()
    print("\nServer shutdown complete")
