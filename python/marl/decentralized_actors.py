"""
Chapter 1: Multi-Agent Reinforcement Learning (MARL) for Cross-Asset Coordination
File: python/marl/decentralized_actors.py

Implements decentralized Nautilus strategy actors that execute based on MARL policy.
Uses strictly bounded observation spaces (local order book + global regime).
Memory bounded to <50MB per agent.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import ray
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModelConfigDict, TensorType

# Memory constraints
MAX_AGENT_MEMORY_MB = 50
TORCH_MEMORY_FRACTION = 0.15  # Limit per-agent GPU memory


class AssetClass(Enum):
    """Supported asset classes for decentralized actors."""
    BTC = "BTC"
    ETH = "ETH"
    SOL = "SOL"
    ALT = "ALT"


@dataclass
class ActorConfig:
    """Configuration for decentralized actor network."""
    asset_class: AssetClass = AssetClass.BTC
    obs_dim_local: int = 64  # Local order book features
    obs_dim_global: int = 32  # Global regime features
    total_obs_dim: int = 96  # Combined observation space
    action_dim: int = 3  # Buy, Sell, Hold
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64])
    learning_rate: float = 3e-4
    max_grad_norm: float = 0.5
    entropy_coeff: float = 0.02
    gamma: float = 0.99
    clip_param: float = 0.2
    
    def __post_init__(self):
        self.total_obs_dim = self.obs_dim_local + self.obs_dim_global
        # Verify memory bounds
        estimated_params = sum(
            d1 * d2 for d1, d2 in zip(
                [self.total_obs_dim] + self.hidden_dims,
                self.hidden_dims + [self.action_dim]
            )
        )
        estimated_memory_mb = (estimated_params * 4) / (1024 * 1024)  # float32
        if estimated_memory_mb > MAX_AGENT_MEMORY_MB:
            raise ValueError(
                f"Actor network exceeds {MAX_AGENT_MEMORY_MB}MB limit: {estimated_memory_mb:.2f}MB"
            )


class DecentralizedActorNetwork(nn.Module):
    """
    Decentralized Actor Network for individual asset strategies.
    Processes local order book + global regime to output actions.
    """
    
    def __init__(self, config: ActorConfig):
        super().__init__()
        self.config = config
        
        # Local observation encoder (order book features)
        self.local_encoder = nn.Sequential(
            nn.Linear(config.obs_dim_local, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(p=0.1)
        )
        
        # Global observation encoder (regime features)
        self.global_encoder = nn.Sequential(
            nn.Linear(config.obs_dim_global, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(p=0.1)
        )
        
        # Combined policy network
        combined_dim = 64 + 32
        policy_layers = []
        prev_dim = combined_dim
        
        for hidden_dim in config.hidden_dims:
            policy_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        self.policy_network = nn.Sequential(*policy_layers)
        
        # Action heads
        self.action_head = nn.Linear(prev_dim, config.action_dim)
        self.value_head = nn.Linear(prev_dim, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Orthogonal initialization for stable PPO training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        nn.init.orthogonal_(self.action_head.weight, gain=0.1)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
    
    def forward(
        self, 
        local_obs: torch.Tensor, 
        global_obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning action logits and value estimates."""
        # Encode observations
        local_features = self.local_encoder(local_obs)
        global_features = self.global_encoder(global_obs)
        
        # Combine features
        combined = torch.cat([local_features, global_features], dim=-1)
        
        # Policy and value
        features = self.policy_network(combined)
        action_logits = self.action_head(features)
        values = self.value_head(features)
        
        return action_logits, values.squeeze(-1)
    
    def get_action_distribution(
        self, 
        local_obs: np.ndarray, 
        global_obs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get action probabilities and value estimate for inference."""
        self.eval()
        with torch.no_grad():
            local_tensor = torch.FloatTensor(local_obs).unsqueeze(0)
            global_tensor = torch.FloatTensor(global_obs).unsqueeze(0)
            
            logits, value = self.forward(local_tensor, global_tensor)
            probs = F.softmax(logits, dim=-1).squeeze(0).numpy()
            value = value.squeeze(0).numpy()
        
        return probs, value
    
    def sample_action(
        self, 
        local_obs: np.ndarray, 
        global_obs: np.ndarray,
        temperature: float = 1.0
    ) -> Tuple[int, float]:
        """Sample action from policy distribution with temperature scaling."""
        self.eval()
        with torch.no_grad():
            local_tensor = torch.FloatTensor(local_obs).unsqueeze(0)
            global_tensor = torch.FloatTensor(global_obs).unsqueeze(0)
            
            logits, value = self.forward(local_tensor, global_tensor)
            
            # Apply temperature scaling
            scaled_logits = logits / temperature
            dist = torch.distributions.Categorical(logits=scaled_logits)
            action = dist.sample().item()
            log_prob = dist.log_prob(torch.tensor(action)).item()
            value = value.squeeze(0).numpy()
        
        return action, log_prob, value


class RayDecentralizedActor(TorchModelV2, nn.Module):
    """
    Ray RLlib compatible decentralized actor wrapper.
    Integrates with MAPPO for multi-agent coordination.
    """
    
    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
        **kwargs
    ):
        nn.Module.__init__(self)
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        
        # Extract config
        custom_config = model_config.get("custom_model_config", {})
        self.actor_config = ActorConfig(**custom_config)
        
        # Build network
        self.network = DecentralizedActorNetwork(self.actor_config)
        self._last_value = None
    
    @override(TorchModelV2)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> Tuple[TensorType, List[TensorType]]:
        """Forward pass returning action logits."""
        obs = input_dict["obs"]
        
        # Split observations into local and global components
        if isinstance(obs, dict):
            local_obs = obs["local_obs"]
            global_obs = obs["global_obs"]
        else:
            # Assume concatenated observation
            local_obs = obs[..., :self.actor_config.obs_dim_local]
            global_obs = obs[..., self.actor_config.obs_dim_local:]
        
        # Convert to tensors if needed
        if not isinstance(local_obs, torch.Tensor):
            local_obs = torch.FloatTensor(local_obs)
        if not isinstance(global_obs, torch.Tensor):
            global_obs = torch.FloatTensor(global_obs)
        
        # Get action logits and values
        logits, values = self.network(local_obs, global_obs)
        
        # Store value for critic access
        self._last_value = values
        
        return logits, state
    
    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        """Return the most recent value function output."""
        assert self._last_value is not None, "Must call forward() first"
        return self._last_value


@ray.remote(num_cpus=1, max_actor_restarts=3)
class DecentralizedActorAgent:
    """
    Ray Actor for decentralized strategy execution.
    Each agent manages one asset class strategy with strict memory bounds.
    """
    
    def __init__(
        self, 
        agent_id: str, 
        config: ActorConfig,
        checkpoint_path: Optional[str] = None
    ):
        self.agent_id = agent_id
        self.config = config
        self.device = torch.device("cpu")  # Default to CPU for memory control
        
        # Set memory limits
        torch.set_num_threads(2)
        
        # Initialize network
        self.network = DecentralizedActorNetwork(config).to(self.device)
        
        # Load checkpoint if provided
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        
        # Execution statistics
        self.stats = {
            "total_actions": 0,
            "buy_count": 0,
            "sell_count": 0,
            "hold_count": 0,
            "avg_confidence": 0.0
        }
    
    def execute(
        self, 
        local_obs: np.ndarray, 
        global_obs: np.ndarray,
        deterministic: bool = True,
        min_confidence: float = 0.3
    ) -> Dict[str, Any]:
        """
        Execute trading decision based on current observations.
        Returns action, confidence, and metadata.
        """
        if deterministic:
            probs, value = self.network.get_action_distribution(
                local_obs, global_obs
            )
            action = int(np.argmax(probs))
            confidence = float(np.max(probs))
        else:
            action, log_prob, value = self.network.sample_action(
                local_obs, global_obs
            )
            probs, _ = self.network.get_action_distribution(
                local_obs, global_obs
            )
            confidence = float(probs[action])
        
        # Filter low-confidence actions
        if confidence < min_confidence:
            action = 2  # Force hold
            confidence = 1.0 - confidence
        
        # Update stats
        self.stats["total_actions"] += 1
        if action == 0:
            self.stats["buy_count"] += 1
        elif action == 1:
            self.stats["sell_count"] += 1
        else:
            self.stats["hold_count"] += 1
        
        # Running average confidence
        n = self.stats["total_actions"]
        self.stats["avg_confidence"] = (
            self.stats["avg_confidence"] * (n - 1) + confidence
        ) / n
        
        return {
            "agent_id": self.agent_id,
            "asset_class": self.config.asset_class.value,
            "action": action,
            "action_name": ["BUY", "SELL", "HOLD"][action],
            "confidence": confidence,
            "value_estimate": float(value),
            "probabilities": probs.tolist()
        }
    
    def update_policy(
        self, 
        state_dict: Dict[str, np.ndarray]
    ):
        """Update policy weights from centralized trainer."""
        torch_state = {
            k: torch.FloatTensor(v) for k, v in state_dict.items()
        }
        self.network.load_state_dict(torch_state)
        self.network.eval()
    
    def load_checkpoint(self, path: str):
        """Load policy checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.network.eval()
    
    def get_stats(self) -> Dict[str, Any]:
        """Return agent execution statistics."""
        return {
            "agent_id": self.agent_id,
            "asset_class": self.config.asset_class.value,
            **self.stats
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check and return memory usage."""
        import gc
        gc.collect()
        
        return {
            "agent_id": self.agent_id,
            "status": "healthy",
            "memory_allocated_mb": torch.cuda.memory_allocated(0) / 1e6 if torch.cuda.is_available() else 0,
            "memory_cached_mb": torch.cuda.memory_reserved(0) / 1e6 if torch.cuda.is_available() else 0
        }


def register_custom_actor():
    """Register the custom actor with Ray RLlib."""
    ModelCatalog.register_custom_model(
        "ray_decentralized_actor",
        RayDecentralizedActor
    )


def get_actor_config_for_asset(asset: AssetClass) -> ActorConfig:
    """Get optimized actor configuration for specific asset class."""
    configs = {
        AssetClass.BTC: ActorConfig(
            asset_class=AssetClass.BTC,
            obs_dim_local=64,
            obs_dim_global=32,
            hidden_dims=[128, 64]
        ),
        AssetClass.ETH: ActorConfig(
            asset_class=AssetClass.ETH,
            obs_dim_local=64,
            obs_dim_global=32,
            hidden_dims=[128, 64]
        ),
        AssetClass.SOL: ActorConfig(
            asset_class=AssetClass.SOL,
            obs_dim_local=48,
            obs_dim_global=32,
            hidden_dims=[96, 48]
        ),
        AssetClass.ALT: ActorConfig(
            asset_class=AssetClass.ALT,
            obs_dim_local=48,
            obs_dim_global=24,
            hidden_dims=[64, 32]
        )
    }
    return configs.get(asset, configs[AssetClass.ALT])


def create_multi_agent_env_config() -> Dict:
    """
    Create multi-agent environment configuration for Ray RLlib.
    Sets up separate agents for each asset class.
    """
    return {
        "agents": {
            "btc_agent": {
                "config": get_actor_config_for_asset(AssetClass.BTC)
            },
            "eth_agent": {
                "config": get_actor_config_for_asset(AssetClass.ETH)
            },
            "sol_agent": {
                "config": get_actor_config_for_asset(AssetClass.SOL)
            }
        },
        "shared_policy_map": {
            "btc_policy": ["btc_agent"],
            "eth_policy": ["eth_agent"],
            "sol_policy": ["sol_agent"]
        }
    }


# Export for module use
__all__ = [
    "AssetClass",
    "ActorConfig",
    "DecentralizedActorNetwork",
    "RayDecentralizedActor",
    "DecentralizedActorAgent",
    "register_custom_actor",
    "get_actor_config_for_asset",
    "create_multi_agent_env_config"
]
