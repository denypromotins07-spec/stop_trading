"""
Chapter 1: Multi-Agent Reinforcement Learning (MARL) for Cross-Asset Coordination
File: python/marl/marl_mod.py

Module root for MARL infrastructure.
Integrates Ray RLlib's MAPPO environment with strict rollout batch size limits
to prevent OOM errors during offline training.
"""

import os
import sys
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.registry import register_env
import numpy as np

# Import local modules
from .centralized_critic import (
    CriticConfig,
    MAPPOCentralizedCritic,
    register_custom_critic,
    get_mappo_config
)
from .decentralized_actors import (
    ActorConfig,
    AssetClass,
    DecentralizedActorAgent,
    register_custom_actor,
    create_multi_agent_env_config
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class MARLConfig:
    """Master configuration for MARL system."""
    # Memory constraints
    max_python_memory_gb: float = 3.0
    max_rollout_batch_size: int = 4096
    max_mini_batch_size: int = 512
    num_rollout_workers: int = 2
    num_envs_per_worker: int = 4
    
    # Training parameters
    training_iterations: int = 1000
    checkpoint_frequency: int = 50
    evaluation_interval: int = 100
    
    # Multi-agent setup
    agent_classes: List[AssetClass] = None
    
    def __post_init__(self):
        if self.agent_classes is None:
            self.agent_classes = [AssetClass.BTC, AssetClass.ETH, AssetClass.SOL]


class MARLSystem:
    """
    Main MARL system orchestrator.
    Manages centralized critic training and decentralized actor deployment.
    """
    
    def __init__(self, config: Optional[MARLConfig] = None):
        self.config = config or MARLConfig()
        self.is_initialized = False
        self.critic = None
        self.agents: Dict[str, ray.actor.ActorHandle] = {}
        self.trainer = None
        
        # Register custom models
        register_custom_critic()
        register_custom_actor()
    
    def initialize_ray(self):
        """Initialize Ray cluster with memory constraints."""
        if not ray.is_initialized():
            # Calculate memory limits based on 3GB ceiling
            object_store_memory = int(self.config.max_python_memory_gb * 1e9 * 0.3)
            
            ray.init(
                num_cpus=self.config.num_rollout_workers + 2,
                _system_config={
                    "object_store_memory": object_store_memory
                },
                log_to_driver=False,
                ignore_reinit_error=True
            )
            logger.info(f"Ray initialized with {object_store_memory / 1e9:.2f}GB object store")
        
        self.is_initialized = True
    
    def create_centralized_critic(self) -> MAPPOCentralizedCritic:
        """Create and initialize the centralized critic."""
        critic_config = CriticConfig(
            input_dim=256,
            hidden_dims=[256, 128, 64],
            mini_batch_size=self.config.max_mini_batch_size,
            train_batch_size=self.config.max_rollout_batch_size
        )
        self.critic = MAPPOCentralizedCritic(critic_config)
        logger.info("Centralized critic created")
        return self.critic
    
    def create_decentralized_agents(self) -> Dict[str, ray.actor.ActorHandle]:
        """Create decentralized actor agents for each asset class."""
        self.agents = {}
        
        for asset_class in self.config.agent_classes:
            agent_id = f"{asset_class.value.lower()}_agent"
            actor_config = ActorConfig(asset_class=asset_class)
            
            # Create Ray actor
            agent = DecentralizedActorAgent.remote(agent_id, actor_config)
            self.agents[agent_id] = agent
            
            logger.info(f"Created decentralized agent: {agent_id}")
        
        return self.agents
    
    def get_mappo_trainer_config(self) -> Dict:
        """Generate complete MAPPO trainer configuration."""
        critic_config = CriticConfig(
            mini_batch_size=self.config.max_mini_batch_size,
            train_batch_size=self.config.max_rollout_batch_size
        )
        
        return {
            **get_mappo_config(critic_config),
            "num_workers": self.config.num_rollout_workers,
            "num_envs_per_worker": self.config.num_envs_per_worker,
            "checkpoint_freq": self.config.checkpoint_frequency,
            "evaluation_interval": self.config.evaluation_interval
        }
    
    async def train_iteration(self) -> Dict[str, Any]:
        """Execute one training iteration."""
        if not self.is_initialized:
            raise RuntimeError("MARL system not initialized. Call initialize_ray() first.")
        
        # Collect experiences from all agents
        experiences = await self._collect_experiences()
        
        # Train centralized critic
        if experiences and self.critic:
            critic_loss = self._train_critic(experiences)
            
            # Update decentralized agents with new policy
            await self._update_agents()
            
            return {
                "critic_loss": critic_loss,
                "n_experiences": len(experiences)
            }
        
        return {"status": "no_data"}
    
    async def _collect_experiences(self) -> List[Dict]:
        """Collect experiences from all decentralized agents."""
        tasks = [
            agent.get_stats.remote() 
            for agent in self.agents.values()
        ]
        results = await ray.get(tasks)
        return results
    
    def _train_critic(self, experiences: List[Dict]) -> float:
        """Train centralized critic on collected experiences."""
        # Placeholder for actual training logic
        # In production, this would process real trajectory data
        return 0.0
    
    async def _update_agents(self):
        """Update decentralized agents with latest policy weights."""
        if not self.critic:
            return
        
        # Get latest policy weights from critic
        # In production, this would extract and distribute actor weights
        pass
    
    async def shutdown(self):
        """Gracefully shutdown MARL system."""
        logger.info("Shutting down MARL system...")
        
        # Shutdown agents
        for agent_id, agent in self.agents.items():
            try:
                ray.kill(agent)
                logger.info(f"Terminated agent: {agent_id}")
            except Exception as e:
                logger.error(f"Error terminating agent {agent_id}: {e}")
        
        self.agents.clear()
        
        # Shutdown Ray
        if ray.is_initialized():
            ray.shutdown()
            logger.info("Ray cluster shutdown complete")
        
        self.is_initialized = False


def create_marl_environment(
    env_name: str = "marl_trading_env"
) -> Any:
    """
    Create and register the MARL trading environment.
    Returns the environment class for Ray RLlib registration.
    """
    from gymnasium import spaces
    import gymnasium as gym
    
    class MARLTradingEnv(gym.Env):
        """
        Multi-agent trading environment for MARL training.
        Supports BTC, ETH, SOL coordinated strategies.
        """
        
        metadata = {"render_modes": ["human", "rgb_array"]}
        
        def __init__(self, config: Optional[MARLConfig] = None):
            super().__init__()
            self.config = config or MARLConfig()
            
            # Observation space: local OB + global regime
            self.observation_space = spaces.Dict({
                "local_obs": spaces.Box(
                    low=-np.inf, 
                    high=np.inf, 
                    shape=(64,), 
                    dtype=np.float32
                ),
                "global_obs": spaces.Box(
                    low=-np.inf, 
                    high=np.inf, 
                    shape=(32,), 
                    dtype=np.float32
                )
            })
            
            # Action space: Buy, Sell, Hold
            self.action_space = spaces.Discrete(3)
            
            # State tracking
            self.current_step = 0
            self.max_steps = 10000
        
        def reset(self, seed=None, options=None):
            """Reset environment state."""
            super().reset(seed=seed)
            self.current_step = 0
            
            # Return dummy observation
            return {
                "local_obs": np.zeros(64, dtype=np.float32),
                "global_obs": np.zeros(32, dtype=np.float32)
            }, {}
        
        def step(self, action):
            """Execute action and return next state, reward, done, truncated, info."""
            self.current_step += 1
            
            # Placeholder reward (implement actual P&L logic)
            reward = 0.0
            
            done = self.current_step >= self.max_steps
            truncated = False
            
            info = {
                "step": self.current_step,
                "action": action
            }
            
            obs = {
                "local_obs": np.zeros(64, dtype=np.float32),
                "global_obs": np.zeros(32, dtype=np.float32)
            }
            
            return obs, reward, done, truncated, info
        
        def render(self, mode="human"):
            """Render environment state."""
            pass
    
    # Register environment
    register_env(env_name, lambda config: MARLTradingEnv(config))
    
    return MARLTradingEnv


def get_training_config(
    marl_config: Optional[MARLConfig] = None
) -> Dict:
    """
    Get complete training configuration for Ray Tune.
    Optimized for 3GB RAM constraint.
    """
    config = marl_config or MARLConfig()
    
    return {
        "env": "marl_trading_env",
        "framework": "torch",
        "num_workers": config.num_rollout_workers,
        "num_envs_per_worker": config.num_envs_per_worker,
        "rollout_fragment_length": 200,
        "train_batch_size": config.max_rollout_batch_size,
        "sgd_minibatch_size": config.max_mini_batch_size,
        "num_sgd_iter": 10,
        "lr": 3e-4,
        "gamma": 0.99,
        "lambda": 0.95,
        "clip_param": 0.2,
        "grad_clip": 0.5,
        "model": {
            "custom_model": "ray_centralized_critic",
            "custom_model_config": {
                "input_dim": 256,
                "hidden_dims": [256, 128, 64]
            }
        },
        "batch_mode": "truncate_episodes",
        "observation_filter": "NoFilter",
        "_fake_gpus": True,  # Force CPU for memory control
        "checkpoint_freq": config.checkpoint_frequency,
        "checkpoint_at_end": True,
        "keep_checkpoints_num": 3,  # Limit disk usage
        "stop": {
            "training_iteration": config.training_iterations
        }
    }


def run_hyperparameter_tuning(
    base_config: Optional[Dict] = None,
    num_trials: int = 10
) -> Any:
    """
    Run hyperparameter tuning with Ray Tune.
    Strictly bounded to prevent resource exhaustion.
    """
    config = base_config or get_training_config()
    
    # Define search space (limited to prevent OOM)
    search_space = {
        "lr": tune.choice([1e-4, 3e-4, 1e-3]),
        "gamma": tune.choice([0.99, 0.995]),
        "entropy_coeff": tune.choice([0.01, 0.02, 0.05])
    }
    
    # Run tuning with resource limits
    analysis = tune.run(
        PPOTrainer,
        config={**config, **search_space},
        num_samples=num_trials,
        resources_per_trial={"cpu": 2, "memory": 1024},  # 1GB per trial
        checkpoint_freq=50,
        verbose=1,
        stop={"training_iteration": 100}
    )
    
    return analysis


# Export for module use
__all__ = [
    "MARLConfig",
    "MARLSystem",
    "create_marl_environment",
    "get_training_config",
    "run_hyperparameter_tuning"
]
