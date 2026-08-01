"""
Queue-Aware RL Agent for Market Making.
Implements reinforcement learning for optimal queue-jumping and cancellation policies
based on L3 microstructure features.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from collections import deque
import numpy as np
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class QueueState:
    """State representation for queue position RL."""
    queue_position: int  # Position in order queue
    queue_size_ahead: int  # Volume ahead in queue
    queue_size_behind: int  # Volume behind in queue
    bid_ask_imbalance: float
    recent_cancellations: int
    recent_insertions: int
    depletion_rate: float  # Rate at which queue is being consumed
    insertion_rate: float  # Rate of new orders joining
    price_momentum: float
    volatility: float
    spread: float
    time_in_queue: float  # Seconds since order placed
    
    def to_array(self) -> np.ndarray:
        """Convert state to numpy array."""
        return np.array([
            self.queue_position / 100.0,  # Normalize
            self.queue_size_ahead / 10000.0,
            self.queue_size_behind / 10000.0,
            self.bid_ask_imbalance,
            min(self.recent_cancellations / 50.0, 1.0),
            min(self.recent_insertions / 50.0, 1.0),
            self.depletion_rate,
            self.insertion_rate,
            self.price_momentum,
            self.volatility,
            self.spread,
            min(self.time_in_queue / 60.0, 1.0)
        ])


@dataclass 
class RLAction:
    """Action taken by RL agent."""
    action_type: str  # 'hold', 'cancel', 'reprice_better', 'reprice_worse', 'jump_queue'
    confidence: float
    expected_value: float
    reasoning: str = ""


class QueueAwareRLAgent:
    """
    Reinforcement Learning agent for queue management.
    Uses Q-learning with function approximation for real-time decisions.
    """
    
    def __init__(self,
                 learning_rate: float = 0.01,
                 discount_factor: float = 0.95,
                 epsilon: float = 0.1,
                 state_dim: int = 12,
                 action_dim: int = 5):
        """
        Initialize RL agent.
        
        Args:
            learning_rate: Alpha for Q-learning updates
            discount_factor: Gamma for future rewards
            epsilon: Exploration rate
            state_dim: Dimension of state space
            action_dim: Number of possible actions
        """
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actions = ['hold', 'cancel', 'reprice_better', 'reprice_worse', 'jump_queue']
        
        # Initialize Q-network weights (simple linear approximation)
        self.Q_weights = np.random.randn(state_dim, action_dim) * 0.1
        self.Q_bias = np.zeros(action_dim)
        
        # Experience replay buffer
        self.replay_buffer: deque = deque(maxlen=10000)
        
        # Training statistics
        self._total_updates = 0
        self._avg_reward: float = 0.0
        self._reward_history: deque = deque(maxlen=1000)
    
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for all actions given state."""
        if len(state.shape) == 1:
            state = state.reshape(1, -1)
        return np.dot(state, self.Q_weights) + self.Q_bias
    
    def select_action(self, state: np.ndarray, training: bool = True) -> Tuple[int, float]:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state array
            training: Whether in training mode (use exploration)
            
        Returns:
            Tuple of (action_index, confidence)
        """
        q_values = self.get_q_values(state)[0]
        
        if training and np.random.random() < self.epsilon:
            # Explore: random action
            action = np.random.randint(self.action_dim)
            confidence = 1.0 / self.action_dim
        else:
            # Exploit: best action
            action = np.argmax(q_values)
            # Confidence from softmax
            exp_q = np.exp(q_values - np.max(q_values))
            confidence = exp_q[action] / np.sum(exp_q)
        
        return action, confidence
    
    def update(self, state: np.ndarray, action: int, reward: float, 
               next_state: np.ndarray, done: bool):
        """
        Update Q-values using TD learning.
        
        Args:
            state: Previous state
            action: Action taken
            reward: Reward received
            next_state: Resulting state
            done: Whether episode ended
        """
        # Store experience
        self.replay_buffer.append((state, action, reward, next_state, done))
        
        # Sample mini-batch
        batch_size = min(32, len(self.replay_buffer))
        if batch_size < 4:
            return
        
        indices = np.random.choice(len(self.replay_buffer), batch_size, replace=False)
        
        for idx in indices:
            s, a, r, s_next, d = self.replay_buffer[idx]
            
            # Calculate TD target
            if d:
                target = r
            else:
                target = r + self.gamma * np.max(self.get_q_values(s_next))
            
            # Update weights
            current_q = self.get_q_values(s.reshape(1, -1))[0, a]
            td_error = target - current_q
            
            # Gradient update
            self.Q_weights[:, a] += self.lr * td_error * s.flatten()
            self.Q_bias[a] += self.lr * td_error
        
        self._total_updates += 1
        
        # Update running average reward
        self._reward_history.append(reward)
        self._avg_reward = np.mean(self._reward_history)
        
        # Decay epsilon
        if self._total_updates % 100 == 0:
            self.epsilon = max(0.01, self.epsilon * 0.995)
    
    def train_offline(self, states: np.ndarray, actions: np.ndarray,
                      rewards: np.ndarray, next_states: np.ndarray,
                      dones: np.ndarray, epochs: int = 10):
        """
        Train offline from historical data.
        
        Args:
            states: Array of states
            actions: Array of actions taken
            rewards: Array of rewards received
            next_states: Array of resulting states
            dones: Array of episode termination flags
            epochs: Number of training epochs
        """
        n_samples = len(states)
        
        for epoch in range(epochs):
            total_loss = 0.0
            
            # Shuffle data
            indices = np.random.permutation(n_samples)
            
            for i in indices:
                s = states[i]
                a = actions[i]
                r = rewards[i]
                s_next = next_states[i]
                d = dones[i]
                
                # Calculate TD target
                if d:
                    target = r
                else:
                    target = r + self.gamma * np.max(self.get_q_values(s_next.reshape(1, -1)))
                
                current_q = self.get_q_values(s.reshape(1, -1))[0, a]
                td_error = target - current_q
                
                # Update
                self.Q_weights[:, a] += self.lr * td_error * s.flatten()
                self.Q_bias[a] += self.lr * td_error
                
                total_loss += td_error ** 2
            
            avg_loss = total_loss / n_samples
            if epoch % 2 == 0:
                logger.debug(f"Epoch {epoch}: Avg TD Loss = {avg_loss:.6f}")
    
    def get_action_recommendation(self, state: QueueState) -> RLAction:
        """
        Get action recommendation for current queue state.
        
        Args:
            state: QueueState object
            
        Returns:
            RLAction with recommendation
        """
        state_array = state.to_array()
        action_idx, confidence = self.select_action(state_array, training=False)
        
        action_type = self.actions[action_idx]
        q_values = self.get_q_values(state_array)[0]
        expected_value = float(q_values[action_idx])
        
        # Generate reasoning
        reasoning = self._generate_reasoning(state, action_type, q_values)
        
        return RLAction(
            action_type=action_type,
            confidence=float(confidence),
            expected_value=expected_value,
            reasoning=reasoning
        )
    
    def _generate_reasoning(self, state: QueueState, action: str, 
                           q_values: np.ndarray) -> str:
        """Generate human-readable reasoning for action."""
        reasons = []
        
        if state.queue_position > 50:
            reasons.append(f"deep in queue (pos={state.queue_position})")
        
        if state.depletion_rate > 0.5:
            reasons.append("high depletion rate")
        elif state.depletion_rate < 0.1:
            reasons.append("low depletion rate")
        
        if state.recent_cancellations > 20:
            reasons.append("many cancellations ahead")
        
        if abs(state.bid_ask_imbalance) > 0.7:
            side = "bid" if state.bid_ask_imbalance > 0 else "ask"
            reasons.append(f"strong {side} imbalance")
        
        if state.time_in_queue > 30:
            reasons.append("long time in queue")
        
        action_reasons = {
            'cancel': "cutting losses due to unfavorable queue dynamics",
            'hold': "queue position acceptable, waiting for fill",
            'reprice_better': "improving priority to increase fill probability",
            'reprice_worse': "widening spread for better execution price",
            'jump_queue': "repricing to jump ahead in queue"
        }
        
        base_reason = action_reasons.get(action, "maintaining current strategy")
        
        if reasons:
            return f"{base_reason.capitalize()} ({'; '.join(reasons)})"
        return base_reason.capitalize()
    
    def save_policy(self, path: str):
        """Save policy weights to file."""
        np.savez(path, weights=self.Q_weights, bias=self.Q_bias)
        logger.info(f"Policy saved to {path}")
    
    def load_policy(self, path: str):
        """Load policy weights from file."""
        try:
            data = np.load(path)
            self.Q_weights = data['weights']
            self.Q_bias = data['bias']
            logger.info(f"Policy loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load policy: {e}")
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        return {
            "total_updates": self._total_updates,
            "avg_reward": self._avg_reward,
            "epsilon": self.epsilon,
            "buffer_size": len(self.replay_buffer),
            "weight_norm": float(np.linalg.norm(self.Q_weights))
        }


class QueuePositionSimulator:
    """
    Simulator for generating training data through self-play.
    Models queue dynamics for realistic RL training.
    """
    
    def __init__(self, 
                 initial_queue_size: int = 1000,
                 arrival_rate: float = 10.0,
                 depletion_rate: float = 5.0):
        """Initialize simulator."""
        self.initial_queue_size = initial_queue_size
        self.arrival_rate = arrival_rate
        self.depletion_rate = depletion_rate
        
        self.reset()
    
    def reset(self):
        """Reset simulation state."""
        self.queue_position = np.random.randint(10, 100)
        self.queue_size_ahead = np.random.randint(100, 1000)
        self.queue_size_behind = np.random.randint(100, 500)
        self.imbalance = np.random.uniform(-0.5, 0.5)
        self.cancellations = 0
        self.insertions = 0
        self.time_in_queue = 0.0
        self.price = 100.0
        self.step_count = 0
    
    def step(self, action: int) -> Tuple[QueueState, float, bool]:
        """
        Execute one simulation step.
        
        Args:
            action: Action index (0=hold, 1=cancel, 2=reprice_better, etc.)
            
        Returns:
            Tuple of (new_state, reward, done)
        """
        self.step_count += 1
        self.time_in_queue += 0.1  # 100ms steps
        
        # Simulate queue dynamics
        arrivals = np.random.poisson(self.arrival_rate * 0.1)
        depletions = np.random.poisson(self.depletion_rate * 0.1)
        
        # Update queue based on action
        action_names = ['hold', 'cancel', 'reprice_better', 'reprice_worse', 'jump_queue']
        action_name = action_names[action]
        
        if action_name == 'cancel':
            # Episode ends on cancel
            reward = -0.1  # Small penalty for canceling
            done = True
            return self._get_state(), reward, done
        
        elif action_name == 'hold':
            # Natural queue progression
            self.queue_size_ahead = max(0, self.queue_size_ahead - depletions)
            self.queue_size_behind += arrivals
            
            if self.queue_size_ahead <= 0:
                # Order gets filled!
                reward = 1.0 + np.random.uniform(-0.2, 0.2)
                done = True
                return self._get_state(), reward, done
        
        elif action_name in ['reprice_better', 'jump_queue']:
            # Move to back of queue but with better price
            self.queue_position = self.queue_size_ahead + self.queue_size_behind
            self.queue_size_ahead = 0
            self.queue_size_behind = 0
            # Pay spread cost
            reward = -0.05
        
        elif action_name == 'reprice_worse':
            # Improve price but worse queue position
            self.queue_size_ahead += np.random.randint(50, 200)
            reward = -0.02
        
        # Random market movements
        self.imbalance += np.random.uniform(-0.1, 0.1)
        self.imbalance = np.clip(self.imbalance, -1, 1)
        
        self.cancellations = np.random.randint(0, 30)
        self.insertions = np.random.randint(0, 30)
        
        # Calculate reward
        fill_probability = 1.0 / (1 + self.queue_size_ahead / 100)
        time_penalty = -0.001 * self.time_in_queue
        reward = fill_probability * 0.1 + time_penalty
        
        # Add noise
        reward += np.random.uniform(-0.05, 0.05)
        
        done = self.step_count >= 100 or self.time_in_queue >= 60
        
        return self._get_state(), reward, done
    
    def _get_state(self) -> QueueState:
        """Get current state."""
        return QueueState(
            queue_position=self.queue_position,
            queue_size_ahead=max(0, self.queue_size_ahead),
            queue_size_behind=max(0, self.queue_size_behind),
            bid_ask_imbalance=self.imbalance,
            recent_cancellations=self.cancellations,
            recent_insertions=self.insertions,
            depletion_rate=self.depletion_rate / 10.0,
            insertion_rate=self.arrival_rate / 10.0,
            price_momentum=np.random.uniform(-0.1, 0.1),
            volatility=np.random.uniform(0.01, 0.1),
            spread=np.random.uniform(0.01, 0.05),
            time_in_queue=self.time_in_queue
        )


def generate_training_data(agent: QueueAwareRLAgent, 
                           n_episodes: int = 1000) -> Tuple[np.ndarray, ...]:
    """
    Generate training data through simulation.
    
    Returns:
        Tuple of (states, actions, rewards, next_states, dones)
    """
    simulator = QueuePositionSimulator()
    
    all_states = []
    all_actions = []
    all_rewards = []
    all_next_states = []
    all_dones = []
    
    for ep in range(n_episodes):
        simulator.reset()
        state = simulator._get_state()
        done = False
        
        while not done:
            state_array = state.to_array()
            action, _ = agent.select_action(state_array, training=True)
            
            next_state, reward, done = simulator.step(action)
            next_state_array = next_state.to_array()
            
            all_states.append(state_array)
            all_actions.append(action)
            all_rewards.append(reward)
            all_next_states.append(next_state_array)
            all_dones.append(done)
            
            state = next_state
        
        if (ep + 1) % 100 == 0:
            logger.info(f"Generated {ep + 1}/{n_episodes} episodes")
    
    return (
        np.array(all_states),
        np.array(all_actions),
        np.array(all_rewards),
        np.array(all_next_states),
        np.array(all_dones)
    )


# Module singleton
_rl_agent: Optional[QueueAwareRLAgent] = None


def get_queue_rl_agent() -> QueueAwareRLAgent:
    """Get or create the global RL agent."""
    global _rl_agent
    
    if _rl_agent is None:
        _rl_agent = QueueAwareRLAgent()
        logger.info("Created queue-aware RL agent")
    
    return _rl_agent


if __name__ == "__main__":
    # Test the RL agent
    print("Training Queue-Aware RL Agent...")
    
    agent = QueueAwareRLAgent(learning_rate=0.05, epsilon=0.3)
    
    # Generate training data
    print("Generating training data...")
    states, actions, rewards, next_states, dones = generate_training_data(
        agent, n_episodes=500
    )
    
    print(f"Generated {len(states)} transitions")
    
    # Train offline
    print("Training offline...")
    agent.train_offline(states, actions, rewards, next_states, dones, epochs=5)
    
    # Test trained agent
    print("\nTesting trained agent...")
    simulator = QueuePositionSimulator()
    
    total_reward = 0
    n_test_episodes = 50
    
    for ep in range(n_test_episodes):
        simulator.reset()
        state = simulator._get_state()
        done = False
        ep_reward = 0
        
        while not done:
            action_rec = agent.get_action_recommendation(state)
            action_idx = agent.actions.index(action_rec.action_type)
            
            next_state, reward, done = simulator.step(action_idx)
            ep_reward += reward
            state = next_state
        
        total_reward += ep_reward
    
    avg_reward = total_reward / n_test_episodes
    print(f"\nTest Results:")
    print(f"  Average Episode Reward: {avg_reward:.4f}")
    print(f"  Training Stats: {agent.get_training_stats()}")
    
    # Example recommendation
    test_state = QueueState(
        queue_position=75,
        queue_size_ahead=500,
        queue_size_behind=200,
        bid_ask_imbalance=-0.3,
        recent_cancellations=25,
        recent_insertions=10,
        depletion_rate=0.2,
        insertion_rate=0.5,
        price_momentum=-0.02,
        volatility=0.05,
        spread=0.02,
        time_in_queue=45.0
    )
    
    rec = agent.get_action_recommendation(test_state)
    print(f"\nExample Recommendation:")
    print(f"  Action: {rec.action_type}")
    print(f"  Confidence: {rec.confidence:.4f}")
    print(f"  Expected Value: {rec.expected_value:.4f}")
    print(f"  Reasoning: {rec.reasoning}")
