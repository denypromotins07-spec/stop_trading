"""
Chapter 3: Meta-Learning & Few-Shot Adaptation (MAML)
File: python/meta_learning/maml_trainer.py

Implements Model-Agnostic Meta-Learning (MAML) for rapid few-shot adaptation
to new altcoins or sudden regime shifts. Trains base model weights that can
fine-tune to new market microstructure in just 5-10 gradient steps.
Strictly bounded mini-batch limits to stay under 3GB RAM ceiling.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from copy import deepcopy
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MAMLConfig:
    """Configuration for MAML training."""
    # Model architecture
    input_dim: int = 64
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64])
    output_dim: int = 3  # Buy, Sell, Hold
    
    # MAML hyperparameters
    inner_lr: float = 0.01  # Task-specific learning rate
    outer_lr: float = 0.001  # Meta-learning rate
    inner_steps: int = 5  # Gradient steps per task
    meta_batch_size: int = 4  # Tasks per meta-update (strictly bounded)
    
    # Memory constraints
    max_gradient_norm: float = 1.0
    use_second_order: bool = False  # First-order MAML for memory efficiency
    
    # Training
    num_iterations: int = 1000
    eval_interval: int = 50


class MAMLModel(nn.Module):
    """
    Base model for MAML meta-learning.
    Simple feedforward network for fast adaptation.
    """
    
    def __init__(self, config: MAMLConfig):
        super().__init__()
        self.config = config
        
        layers = []
        prev_dim = config.input_dim
        
        for hidden_dim in config.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        self.output_head = nn.Linear(prev_dim, config.output_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Orthogonal initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        features = self.feature_extractor(x)
        return self.output_head(features)
    
    def clone_params(self) -> Dict[str, torch.Tensor]:
        """Clone current parameters for inner loop optimization."""
        return {k: v.clone() for k, v in self.named_parameters()}
    
    def load_params(self, params: Dict[str, torch.Tensor]):
        """Load parameters from dict."""
        with torch.no_grad():
            for name, param in self.named_parameters():
                if name in params:
                    param.copy_(params[name])


class MAMLTrainer:
    """
    MAML trainer for few-shot adaptation.
    Implements first-order MAML (FOMAML) for memory efficiency.
    """
    
    def __init__(self, config: Optional[MAMLConfig] = None):
        self.config = config or MAMLConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model
        self.model = MAMLModel(self.config).to(self.device)
        self.meta_optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.config.outer_lr
        )
        
        # Training state
        self.iteration = 0
        self.meta_losses: List[float] = []
        self.adaptation_losses: List[float] = []
        
        # Memory control
        torch.set_num_threads(4)
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.3)
    
    def create_adapted_model(
        self, 
        support_data: Tuple[torch.Tensor, torch.Tensor],
        n_steps: Optional[int] = None
    ) -> MAMLModel:
        """
        Create task-adapted model using inner loop optimization.
        
        Args:
            support_data: (x_support, y_support) for the task
            n_steps: Number of inner gradient steps
        
        Returns:
            Adapted model with task-specific weights
        """
        n_steps = n_steps or self.config.inner_steps
        x_support, y_support = support_data
        
        # Clone model parameters for adaptation
        adapted_params = self.model.clone_params()
        
        # Inner loop optimization
        for step in range(n_steps):
            # Forward pass with adapted parameters
            features = self._forward_with_params(
                x_support, 
                adapted_params, 
                self.model.feature_extractor
            )
            logits = self._linear_with_params(
                features, 
                adapted_params, 
                "output_head"
            )
            
            # Compute loss
            loss = nn.functional.cross_entropy(logits, y_support)
            
            # Compute gradients
            grads = torch.autograd.grad(
                loss, 
                adapted_params.values(),
                create_graph=not self.config.use_second_order
            )
            
            # Update adapted parameters
            with torch.no_grad():
                for (name, param), grad in zip(adapted_params.items(), grads):
                    param.sub_(self.config.inner_lr * grad)
        
        # Create adapted model
        adapted_model = deepcopy(self.model)
        adapted_model.load_params(adapted_params)
        
        return adapted_model
    
    def _forward_with_params(
        self, 
        x: torch.Tensor, 
        params: Dict[str, torch.Tensor],
        module: nn.Module,
        prefix: str = ""
    ) -> torch.Tensor:
        """Forward pass using provided parameters."""
        # Simplified forward for feature extractor
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                param_name = f"{prefix}feature_extractor.{name}.weight" if prefix else f"feature_extractor.{name}.weight"
                bias_name = param_name.replace("weight", "bias")
                
                weight = params.get(param_name)
                bias = params.get(bias_name)
                
                if weight is not None:
                    x = nn.functional.linear(x, weight, bias)
            elif isinstance(child, nn.ReLU):
                x = nn.functional.relu(x)
            elif isinstance(child, nn.LayerNorm):
                ln_name = param_name.replace("weight", "weight")
                ln_bias = bias_name.replace("weight", "bias")
                ln_weight = params.get(ln_name)
                ln_bias = params.get(ln_bias)
                if ln_weight is not None:
                    x = nn.functional.layer_norm(
                        x, 
                        child.normalized_shape, 
                        ln_weight, 
                        ln_bias,
                        child.eps
                    )
        return x
    
    def _linear_with_params(
        self,
        x: torch.Tensor,
        params: Dict[str, torch.Tensor],
        head_name: str
    ) -> torch.Tensor:
        """Linear layer forward with provided parameters."""
        weight_name = f"{head_name}.weight"
        bias_name = f"{head_name}.bias"
        
        weight = params.get(weight_name)
        bias = params.get(bias_name)
        
        if weight is not None:
            return nn.functional.linear(x, weight, bias)
        return x
    
    def meta_update(
        self, 
        task_batch: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> float:
        """
        Perform one meta-learning update step.
        
        Args:
            task_batch: List of (support_x, support_y) tuples for each task
        
        Returns:
            Meta-loss value
        """
        if len(task_batch) > self.config.meta_batch_size:
            # Subsample to respect memory bounds
            indices = np.random.choice(
                len(task_batch), 
                self.config.meta_batch_size, 
                replace=False
            )
            task_batch = [task_batch[i] for i in indices]
        
        total_meta_loss = 0.0
        
        # Accumulate gradients across tasks
        self.meta_optimizer.zero_grad()
        
        for task_data in task_batch:
            # Create adapted model for this task
            adapted_model = self.create_adapted_model(task_data)
            
            # Evaluate on query set (or same support for simplicity)
            x_query, y_query = task_data
            
            # Forward pass with adapted model
            query_logits = adapted_model(x_query)
            query_loss = nn.functional.cross_entropy(query_logits, y_query)
            
            total_meta_loss += query_loss.item()
            
            # Backpropagate through adaptation (first-order approximation)
            if self.config.use_second_order:
                query_loss.backward()
            else:
                # FOMAML: use gradients from adapted model directly
                query_loss.backward(retain_graph=True)
        
        # Average meta-loss
        meta_loss = total_meta_loss / len(task_batch)
        
        # Clip gradients
        nn.utils.clip_grad_norm_(
            self.model.parameters(), 
            self.config.max_gradient_norm
        )
        
        # Meta-update
        self.meta_optimizer.step()
        
        self.iteration += 1
        self.meta_losses.append(meta_loss)
        
        logger.debug(
            f"MAML iteration {self.iteration}: meta_loss={meta_loss:.4f}"
        )
        
        return meta_loss
    
    def adapt_to_task(
        self, 
        support_data: Tuple[np.ndarray, np.ndarray],
        n_steps: int = 5
    ) -> MAMLModel:
        """
        Rapidly adapt model to a new task with few gradient steps.
        
        Args:
            support_data: (x, y) numpy arrays for few-shot learning
            n_steps: Number of adaptation steps
        
        Returns:
            Adapted model ready for inference
        """
        x_support = torch.FloatTensor(support_data[0]).to(self.device)
        y_support = torch.LongTensor(support_data[1]).to(self.device)
        
        self.model.eval()
        adapted_model = self.create_adapted_model(
            (x_support, y_support), 
            n_steps=n_steps
        )
        
        return adapted_model
    
    def predict(
        self, 
        adapted_model: MAMLModel, 
        x: np.ndarray
    ) -> np.ndarray:
        """Make predictions with adapted model."""
        adapted_model.eval()
        x_tensor = torch.FloatTensor(x).to(self.device)
        
        with torch.no_grad():
            logits = adapted_model(x_tensor)
            probs = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(probs, dim=-1)
        
        return predictions.cpu().numpy(), probs.cpu().numpy()
    
    def save_checkpoint(self, path: str):
        """Save meta-learning checkpoint."""
        torch.save({
            "iteration": self.iteration,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.meta_optimizer.state_dict(),
            "meta_losses": self.meta_losses,
            "adaptation_losses": self.adaptation_losses,
            "config": self.config
        }, path)
        logger.info(f"MAML checkpoint saved: {path}")
    
    def load_checkpoint(self, path: str):
        """Load meta-learning checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.meta_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.iteration = checkpoint["iteration"]
        self.meta_losses = checkpoint.get("meta_losses", [])
        self.adaptation_losses = checkpoint.get("adaptation_losses", [])
        logger.info(f"MAML checkpoint loaded: {path}")
    
    def get_base_weights(self) -> Dict[str, np.ndarray]:
        """Extract base model weights for distribution."""
        return {
            k: v.cpu().numpy() 
            for k, v in self.model.named_parameters()
        }


# Export for module use
__all__ = [
    "MAMLConfig",
    "MAMLModel",
    "MAMLTrainer"
]
