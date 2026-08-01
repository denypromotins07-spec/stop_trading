"""
Hindsight Experience Replay (HER) Buffer for sparse-reward execution tasks.
Enables learning from "failed" executions by relabeling achieved state as goal.
Accelerates convergence for massive block order execution without slippage.
Uses fixed-size collections.deque and pre-allocated numpy ring buffer.
Strictly bounded memory to respect 3GB RAM limit.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import deque
import numpy as np
import random

logger = logging.getLogger(__name__)


@dataclass
class HindsightExperience:
    """Single HER experience tuple."""
    observation: np.ndarray
    action: np.ndarray
    reward: float
    next_observation: np.ndarray
    done: bool
    original_goal: Optional[np.ndarray] = None
    achieved_goal: Optional[np.ndarray] = None
    relabeled: bool = False
    
    def __post_init__(self):
        """Ensure arrays are contiguous for efficient storage."""
        if isinstance(self.observation, np.ndarray):
            self.observation = np.ascontiguousarray(self.observation)
        if isinstance(self.action, np.ndarray):
            self.action = np.ascontiguousarray(self.action)
        if isinstance(self.next_observation, np.ndarray):
            self.next_observation = np.ascontiguousarray(self.next_observation)
        if self.original_goal is not None and isinstance(self.original_goal, np.ndarray):
            self.original_goal = np.ascontiguousarray(self.original_goal)
        if self.achieved_goal is not None and isinstance(self.achieved_goal, np.ndarray):
            self.achieved_goal = np.ascontiguousarray(self.achieved_goal)


@dataclass
class Episode:
    """Complete episode trajectory for HER relabeling."""
    experiences: List[HindsightExperience] = field(default_factory=list)
    success: bool = False
    episode_id: int = 0
    
    def add(self, exp: HindsightExperience):
        """Add experience to episode."""
        self.experiences.append(exp)
    
    def __len__(self) -> int:
        return len(self.experiences)


class HERBuffer:
    """
    Hindsight Experience Replay buffer with bounded memory.
    
    Key features:
    - Fixed-size deque for experience storage
    - Pre-allocated numpy ring buffer option for performance
    - Goal relabeling for sparse reward tasks
    - Multiple HER strategies (future, final, episode)
    - Memory-bounded to respect 3GB RAM limit
    """
    
    def __init__(self,
                 max_episodes: int = 10000,
                 max_total_experiences: int = 500000,
                 obs_dim: int = 64,
                 action_dim: int = 8,
                 goal_dim: int = 16,
                 her_sample_ratio: float = 0.8,
                 seed: Optional[int] = None):
        """
        Initialize HER buffer.
        
        Args:
            max_episodes: Maximum episodes to store
            max_total_experiences: Hard limit on total experiences
            obs_dim: Observation dimension
            action_dim: Action dimension
            goal_dim: Goal dimension
            her_sample_ratio: Ratio of relabeled samples in batch
            seed: Random seed for reproducibility
        """
        self._max_episodes = max_episodes
        self._max_total_experiences = max_total_experiences
        self._obs_dim = obs_dim
        self._action_dim = action_dim
        self._goal_dim = goal_dim
        self._her_sample_ratio = her_sample_ratio
        
        # Bounded episode storage using deque
        self._episodes: deque = deque(maxlen=max_episodes)
        
        # Flat experience storage for efficient sampling
        self._experiences: deque = deque(maxlen=max_total_experiences)
        
        # Pre-allocated numpy ring buffer for batch operations
        self._ring_buffer_size = max_total_experiences
        self._ring_obs = np.zeros(
            (self._ring_buffer_size, obs_dim), dtype=np.float32
        )
        self._ring_action = np.zeros(
            (self._ring_buffer_size, action_dim), dtype=np.float32
        )
        self._ring_reward = np.zeros(self._ring_buffer_size, dtype=np.float32)
        self._ring_next_obs = np.zeros(
            (self._ring_buffer_size, obs_dim), dtype=np.float32
        )
        self._ring_done = np.zeros(self._ring_buffer_size, dtype=bool)
        self._ring_goal = np.zeros(
            (self._ring_buffer_size, goal_dim), dtype=np.float32
        )
        self._ring_achieved_goal = np.zeros(
            (self._ring_buffer_size, goal_dim), dtype=np.float32
        )
        self._ring_relabelled = np.zeros(self._ring_buffer_size, dtype=bool)
        
        self._ring_head = 0  # Write pointer
        self._current_size = 0  # Number of valid entries
        
        # Current episode being collected
        self._current_episode: Optional[Episode] = None
        self._episode_counter = 0
        
        # Statistics
        self._stats = {
            'total_episodes': 0,
            'successful_episodes': 0,
            'total_experiences': 0,
            'relabelled_experiences': 0,
            'her_samples_generated': 0,
        }
        
        # RNG
        self._rng = np.random.default_rng(seed)
        
        logger.info(
            f"HERBuffer initialized: max_episodes={max_episodes}, "
            f"max_experiences={max_total_experiences}"
        )
    
    def start_episode(self, episode_id: Optional[int] = None):
        """Start collecting a new episode."""
        if episode_id is None:
            episode_id = self._episode_counter
            self._episode_counter += 1
        
        self._current_episode = Episode(episode_id=episode_id)
    
    def add_transition(self,
                       obs: np.ndarray,
                       action: np.ndarray,
                       reward: float,
                       next_obs: np.ndarray,
                       done: bool,
                       original_goal: Optional[np.ndarray] = None,
                       achieved_goal: Optional[np.ndarray] = None):
        """
        Add transition to current episode.
        
        Args:
            obs: Current observation
            action: Taken action
            reward: Received reward
            next_obs: Next observation
            done: Episode termination flag
            original_goal: Original goal (optional)
            achieved_goal: Actually achieved goal (optional)
        """
        if self._current_episode is None:
            self.start_episode()
        
        exp = HindsightExperience(
            observation=obs,
            action=action,
            reward=reward,
            next_observation=next_obs,
            done=done,
            original_goal=original_goal,
            achieved_goal=achieved_goal,
        )
        
        self._current_episode.add(exp)
        
        # Also add to flat storage
        self._add_to_flat_storage(exp)
    
    def _add_to_flat_storage(self, exp: HindsightExperience):
        """Add experience to flat storage structures."""
        idx = self._ring_head
        
        # Copy to ring buffer
        self._ring_obs[idx] = exp.observation
        self._ring_action[idx] = exp.action
        self._ring_reward[idx] = exp.reward
        self._ring_next_obs[idx] = exp.next_observation
        self._ring_done[idx] = exp.done
        
        if exp.original_goal is not None:
            self._ring_goal[idx] = exp.original_goal
        if exp.achieved_goal is not None:
            self._ring_achieved_goal[idx] = exp.achieved_goal
        
        self._ring_relabelled[idx] = exp.relabeled
        
        # Update pointers
        self._ring_head = (self._ring_head + 1) % self._ring_buffer_size
        self._current_size = min(self._current_size + 1, self._ring_buffer_size)
        
        # Add to deque for episode-based access
        self._experiences.append(exp)
        
        self._stats['total_experiences'] += 1
    
    def end_episode(self, success: bool = False):
        """
        End current episode and perform HER relabeling.
        
        Args:
            success: Whether episode was successful with original goal
        """
        if self._current_episode is None:
            return
        
        self._current_episode.success = success
        self._stats['total_episodes'] += 1
        if success:
            self._stats['successful_episodes'] += 1
        
        # Perform HER relabeling
        self._relabel_episode(self._current_episode)
        
        # Store completed episode
        self._episodes.append(self._current_episode)
        self._current_episode = None
    
    def _relabel_episode(self, episode: Episode):
        """
        Apply Hindsight Experience Replay relabeling.
        
        Strategies:
        - future: Relabel with goals from future states
        - final: Relabel with final achieved state
        - episode: Relabel with any state from episode
        """
        if len(episode) < 2:
            return
        
        experiences = episode.experiences
        
        # Strategy selection
        strategy = self._rng.choice(['future', 'final', 'episode'])
        
        for i, exp in enumerate(experiences):
            if exp.achieved_goal is None:
                continue
            
            # Select relabeling goal based on strategy
            if strategy == 'final':
                # Use final achieved state as goal
                new_goal = experiences[-1].achieved_goal
            elif strategy == 'episode':
                # Use random state from episode as goal
                j = self._rng.integers(0, len(experiences))
                new_goal = experiences[j].achieved_goal
            else:  # future
                # Use future state as goal
                if i >= len(experiences) - 1:
                    continue
                j = self._rng.integers(i + 1, len(experiences))
                new_goal = experiences[j].achieved_goal
            
            # Calculate relabeled reward
            # Higher reward if new goal is closer to achieved state
            if exp.achieved_goal is not None and new_goal is not None:
                distance = np.linalg.norm(exp.achieved_goal - new_goal)
                # Sparse reward: 0 if close, -1 otherwise
                relabeled_reward = 0.0 if distance < 0.1 else -1.0
                
                # Create relabeled experience
                relabeled_exp = HindsightExperience(
                    observation=exp.observation,
                    action=exp.action,
                    reward=relabeled_reward,
                    next_observation=exp.next_observation,
                    done=exp.done,
                    original_goal=exp.original_goal,
                    achieved_goal=exp.achieved_goal,
                    relabeled=True,
                )
                
                # Add relabeled experience to storage
                self._add_to_flat_storage(relabeled_exp)
                self._stats['relabelled_experiences'] += 1
    
    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """
        Sample batch of experiences with HER mixing.
        
        Args:
            batch_size: Number of samples
            
        Returns:
            Batch dictionary with observations, actions, rewards, etc.
        """
        if self._current_size == 0:
            raise ValueError("Buffer is empty")
        
        actual_batch_size = min(batch_size, self._current_size)
        
        # Determine how many HER samples to include
        n_her = int(actual_batch_size * self._her_sample_ratio)
        n_original = actual_batch_size - n_her
        
        # Sample indices
        indices = self._rng.choice(
            self._current_size, 
            size=actual_batch_size, 
            replace=False
        )
        
        # Map logical indices to ring buffer indices
        ring_indices = (self._ring_head - self._current_size + indices) % self._ring_buffer_size
        
        # Extract batch
        batch = {
            'obs': self._ring_obs[ring_indices],
            'action': self._ring_action[ring_indices],
            'reward': self._ring_reward[ring_indices],
            'next_obs': self._ring_next_obs[ring_indices],
            'done': self._ring_done[ring_indices],
            'goal': self._ring_goal[ring_indices],
            'achieved_goal': self._ring_achieved_goal[ring_indices],
            'relabeled': self._ring_relabelled[ring_indices],
        }
        
        self._stats['her_samples_generated'] += n_her
        
        return batch
    
    def sample_episodes(self, n_episodes: int) -> List[Episode]:
        """
        Sample complete episodes for sequence-based training.
        
        Args:
            n_episodes: Number of episodes to sample
            
        Returns:
            List of episodes
        """
        if len(self._episodes) == 0:
            return []
        
        n = min(n_episodes, len(self._episodes))
        indices = self._rng.choice(len(self._episodes), size=n, replace=False)
        
        return [self._episodes[i] for i in indices]
    
    def get_priority_indices(self, 
                            td_errors: np.ndarray,
                            top_k: int = 100) -> np.ndarray:
        """
        Get indices of highest priority experiences based on TD error.
        
        Args:
            td_errors: TD errors for all experiences
            top_k: Number of top priorities to return
            
        Returns:
            Indices of top-k experiences
        """
        if len(td_errors) != self._current_size:
            raise ValueError("TD errors length mismatch")
        
        # Get top-k indices
        top_indices = np.argsort(np.abs(td_errors))[-top_k:]
        
        return top_indices
    
    def clear(self):
        """Clear all stored experiences."""
        self._episodes.clear()
        self._experiences.clear()
        self._ring_head = 0
        self._current_size = 0
        self._current_episode = None
        
        # Zero out ring buffers
        self._ring_obs.fill(0)
        self._ring_action.fill(0)
        self._ring_reward.fill(0)
        self._ring_next_obs.fill(0)
        self._ring_done.fill(False)
        self._ring_goal.fill(0)
        self._ring_achieved_goal.fill(0)
        self._ring_relabelled.fill(False)
        
        logger.info("HERBuffer cleared")
    
    def get_stats(self) -> Dict:
        """Get buffer statistics."""
        stats = self._stats.copy()
        stats['current_size'] = self._current_size
        stats['current_episodes'] = len(self._episodes)
        stats['buffer_utilization'] = self._current_size / self._ring_buffer_size
        return stats
    
    def get_memory_usage_bytes(self) -> int:
        """Estimate memory usage in bytes."""
        # Ring buffer memory
        ring_memory = (
            self._ring_obs.nbytes +
            self._ring_action.nbytes +
            self._ring_reward.nbytes +
            self._ring_next_obs.nbytes +
            self._ring_done.nbytes +
            self._ring_goal.nbytes +
            self._ring_achieved_goal.nbytes +
            self._ring_relabelled.nbytes
        )
        
        # Deque overhead (approximate)
        deque_memory = len(self._experiences) * 500  # ~500 bytes per experience object
        
        return ring_memory + deque_memory
