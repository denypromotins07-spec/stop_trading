"""
Chapter 1: Multi-Agent Reinforcement Learning (MARL) for Cross-Asset Coordination
File: python/marl/centralized_critic.py

Implements a MAPPO centralized critic network evaluating global portfolio delta
and cross-margin states. Trained offline to guide decentralized agents.
Strictly bounded to prevent OOM during training.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModelConfigDict, TensorType

# Enforce strict memory limits
torch.set_num_threads(4)
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.3)  # Limit GPU usage


@dataclass
class CriticConfig:
    """Configuration for the centralized critic."""
    input_dim: int = 256  # Global state dimension (portfolio + cross-margin)
    hidden_dims: List[int] = None
    output_dim: int = 1  # Value estimate
    learning_rate: float = 3e-4
    max_grad_norm: float = 0.5
    entropy_coeff: float = 0.01
    vf_loss_coeff: float = 0.5
    clip_param: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    mini_batch_size: int = 512  # Strictly bounded to prevent OOM
    train_batch_size: int = 4096  # Total batch size for gradient accumulation
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 128, 64]


class CentralizedCriticNetwork(nn.Module):
    """
    Centralized Critic Network for MAPPO.
    Evaluates global portfolio state to guide decentralized actors.
    """
    
    def __init__(self, config: CriticConfig):
        super().__init__()
        self.config = config
        
        layers = []
        prev_dim = config.input_dim
        
        for hidden_dim in config.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=0.1)  # Prevent overfitting
            ])
            prev_dim = hidden_dim
        
        self.shared_layers = nn.Sequential(*layers)
        self.value_head = nn.Linear(prev_dim, config.output_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Orthogonal initialization for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Forward pass returning value estimates."""
        features = self.shared_layers(observations)
        values = self.value_head(features)
        return values.squeeze(-1)
    
    def get_value(self, observations: np.ndarray) -> np.ndarray:
        """Inference method for getting value estimates."""
        self.eval()
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observations)
            values = self.forward(obs_tensor)
        return values.numpy()


class RayCentralizedCritic(TorchModelV2, nn.Module):
    """
    Ray RLlib compatible centralized critic wrapper.
    Integrates with MAPPO trainer for multi-agent coordination.
    """
    
    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
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
        self.critic_config = CriticConfig(**custom_config)
        
        # Build network
        self.network = CentralizedCriticNetwork(self.critic_config)
        self._value_function = None
    
    @override(TorchModelV2)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> Tuple[TensorType, List[TensorType]]:
        """Forward pass for policy and value function."""
        obs = input_dict["obs"]
        
        # Handle different observation formats
        if isinstance(obs, dict):
            obs = obs["global_state"]
        
        # Convert to tensor if needed
        if not isinstance(obs, torch.Tensor):
            obs = torch.FloatTensor(obs)
        
        # Get value estimates
        values = self.network(obs)
        
        # Store for value function access
        self._value_function = values
        
        # Return dummy action logits (critic doesn't output actions)
        # In MAPPO, actors have separate networks
        return torch.zeros_like(values), state
    
    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        """Return the most recent value function output."""
        assert self._value_function is not None, "Must call forward() first"
        return self._value_function


class MAPPOCentralizedCritic:
    """
    Main class for managing the MAPPO centralized critic.
    Handles training, evaluation, and integration with Ray RLlib.
    """
    
    def __init__(self, config: Optional[CriticConfig] = None):
        self.config = config or CriticConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = CentralizedCriticNetwork(self.config).to(self.device)
        self.optimizer = optim.Adam(
            self.network.parameters(), 
            lr=self.config.learning_rate
        )
        self.training_stats = {
            "total_updates": 0,
            "avg_loss": 0.0,
            "avg_entropy": 0.0
        }
    
    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        next_values: np.ndarray,
        dones: np.ndarray,
        gamma: Optional[float] = None,
        lam: Optional[float] = None
    ) -> np.ndarray:
        """
        Compute Generalized Advantage Estimation (GAE).
        Efficiently calculates advantages with low memory footprint.
        """
        gamma = gamma or self.config.gamma
        lam = lam or self.config.gae_lambda
        
        advantages = np.zeros_like(rewards)
        last_advantage = 0.0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values[t] if next_values is not None else 0.0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_advantage = delta + gamma * lam * (1 - dones[t]) * last_advantage
        
        return advantages
    
    def update(
        self,
        observations: np.ndarray,
        returns: np.ndarray,
        old_values: np.ndarray,
        mini_batch_size: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Perform one update step using mini-batch gradient descent.
        Implements gradient accumulation for large batches.
        """
        mini_batch_size = mini_batch_size or self.config.mini_batch_size
        n_samples = len(observations)
        n_batches = max(1, n_samples // mini_batch_size)
        
        total_loss = 0.0
        n_updates = 0
        
        # Shuffle indices
        indices = np.random.permutation(n_samples)
        
        for i in range(n_batches):
            start_idx = i * mini_batch_size
            end_idx = min((i + 1) * mini_batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            
            obs_batch = torch.FloatTensor(observations[batch_indices]).to(self.device)
            returns_batch = torch.FloatTensor(returns[batch_indices]).to(self.device)
            old_values_batch = torch.FloatTensor(old_values[batch_indices]).to(self.device)
            
            # Forward pass
            self.network.train()
            self.optimizer.zero_grad()
            
            pred_values = self.network(obs_batch)
            
            # Value loss (clipped to prevent large updates)
            value_pred_clipped = old_values_batch + (pred_values - old_values_batch).clamp(
                -self.config.clip_param, self.config.clip_param
            )
            value_losses = (pred_values - returns_batch).pow(2)
            value_losses_clipped = (value_pred_clipped - returns_batch).pow(2)
            value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
            
            # Backward pass
            value_loss.backward()
            nn.utils.clip_grad_norm_(
                self.network.parameters(), 
                self.config.max_grad_norm
            )
            self.optimizer.step()
            
            total_loss += value_loss.item()
            n_updates += 1
        
        # Update stats
        self.training_stats["total_updates"] += n_updates
        self.training_stats["avg_loss"] = (
            self.training_stats["avg_loss"] * (n_updates - 1) + total_loss
        ) / n_updates
        
        return {
            "value_loss": total_loss / n_updates,
            "n_updates": n_updates
        }
    
    def evaluate(
        self,
        observations: np.ndarray,
        batch_size: int = 1024
    ) -> np.ndarray:
        """
        Evaluate value function for a batch of observations.
        Memory-efficient batched inference.
        """
        self.network.eval()
        all_values = []
        
        with torch.no_grad():
            for i in range(0, len(observations), batch_size):
                batch = observations[i:i + batch_size]
                obs_tensor = torch.FloatTensor(batch).to(self.device)
                values = self.network(obs_tensor).cpu().numpy()
                all_values.append(values)
        
        return np.concatenate(all_values, axis=0)
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            "network_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "training_stats": self.training_stats,
            "config": self.config
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.training_stats = checkpoint["training_stats"]


def register_custom_critic():
    """Register the custom critic with Ray RLlib."""
    ModelCatalog.register_custom_model(
        "ray_centralized_critic", 
        RayCentralizedCritic
    )


def get_mappo_config(
    critic_config: Optional[CriticConfig] = None
) -> Dict:
    """
    Generate Ray RLlib MAPPO configuration.
    Strictly bounds batch sizes to prevent OOM.
    """
    config = critic_config or CriticConfig()
    
    return {
        "framework": "torch",
        "num_workers": 2,  # Limited workers to control memory
        "num_envs_per_worker": 4,
        "rollout_fragment_length": 200,
        "train_batch_size": config.train_batch_size,
        "sgd_minibatch_size": config.mini_batch_size,
        "num_sgd_iter": 10,
        "lr": config.learning_rate,
        "gamma": config.gamma,
        "lambda": config.gae_lambda,
        "clip_param": config.clip_param,
        "entropy_coeff": config.entropy_coeff,
        "vf_loss_coeff": config.vf_loss_coeff,
        "grad_clip": config.max_grad_norm,
        "model": {
            "custom_model": "ray_centralized_critic",
            "custom_model_config": {
                "input_dim": config.input_dim,
                "hidden_dims": config.hidden_dims,
                "output_dim": config.output_dim
            }
        },
        "batch_mode": "truncate_episodes",
        "observation_filter": "NoFilter",  # Disable to save memory
        "_fake_gpus": not torch.cuda.is_available()
    }


# Export for module use
__all__ = [
    "CriticConfig",
    "CentralizedCriticNetwork",
    "RayCentralizedCritic",
    "MAPPOCentralizedCritic",
    "register_custom_critic",
    "get_mappo_config"
]
