"""
Reinforcement Learning for Avellaneda-Stoikov Market Making Calibration.
Builds a custom RL environment for calibrating MM parameters (gamma, kappa, sigma).
Trains an agent to dynamically adjust risk-aversion based on order book toxicity (VPIN).

Uses lightweight RL algorithms compatible with ONNX export for minimal memory footprint.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass
import threading
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MarketMakingState:
    """Current state of the market making environment."""
    mid_price: float
    inventory: float
    inventory_value: float
    cash: float
    total_value: float
    vpin: float  # VPIN toxicity measure
    spread_bps: float
    volatility: float
    time_remaining: float  # Fraction of trading day remaining
    
    # Derived features
    inventory_risk: float
    skew_factor: float


@dataclass
class MarketMakingAction:
    """Action taken by the market making agent."""
    gamma: float      # Risk aversion parameter
    kappa: float      # Order arrival intensity
    spread_half: float  # Half-spread in basis points
    skew: float       # Quote skew based on inventory
    
    def get_quotes(self, mid_price: float) -> Tuple[float, float]:
        """Calculate bid and ask quotes from action."""
        half_spread_abs = mid_price * self.spread_half / 10000
        bid = mid_price - half_spread_abs + self.skew
        ask = mid_price + half_spread_abs + self.skew
        return bid, ask


@dataclass
class MarketMakingStepResult:
    """Result of a single step in the environment."""
    reward: float
    done: bool
    pnl: float
    trades_executed: int
    adverse_selection_cost: float


class AvellanedaStoikovEnvironment:
    """
    Custom RL environment for Avellaneda-Stoikov market making.
    
    State space:
    - Inventory level and value
    - VPIN toxicity
    - Volatility estimate
    - Time remaining
    
    Action space:
    - Gamma (risk aversion)
    - Kappa (order arrival intensity)  
    - Spread width
    - Quote skew
    """
    
    def __init__(
        self,
        initial_cash: float = 1_000_000,
        max_inventory: float = 100.0,
        trading_day_minutes: int = 390,
        fee_rate_bps: float = 2.5
    ):
        self.initial_cash = initial_cash
        self.max_inventory = max_inventory
        self.trading_day_minutes = trading_day_minutes
        self.fee_rate_bps = fee_rate_bps
        
        self._lock = threading.Lock()
        self.reset()
    
    def reset(
        self,
        initial_mid_price: float = 50000.0,
        initial_volatility: float = 0.0002
    ) -> MarketMakingState:
        """Reset environment to initial state."""
        with self._lock:
            self.mid_price = initial_mid_price
            self.volatility = initial_volatility
            self.inventory = 0.0
            self.cash = self.initial_cash
            self.time_step = 0
            self.total_trades = 0
            self.adverse_selection_total = 0.0
            
            return self._get_state()
    
    def _get_state(self) -> MarketMakingState:
        """Get current environment state."""
        inventory_value = self.inventory * self.mid_price
        total_value = self.cash + inventory_value
        
        # Inventory risk (normalized)
        inventory_risk = self.inventory / self.max_inventory
        
        # Skew factor based on inventory position
        skew_factor = -inventory_risk * 0.5  # Negative inventory -> positive skew
        
        return MarketMakingState(
            mid_price=self.mid_price,
            inventory=self.inventory,
            inventory_value=inventory_value,
            cash=self.cash,
            total_value=total_value,
            vpin=getattr(self, 'vpin', 0.5),
            spread_bps=getattr(self, 'spread_bps', 10.0),
            volatility=self.volatility,
            time_remaining=1.0 - (self.time_step / self.trading_day_minutes),
            inventory_risk=inventory_risk,
            skew_factor=skew_factor
        )
    
    def step(
        self,
        action: MarketMakingAction,
        price_change: float,
        order_flow: float,
        vpin: float
    ) -> MarketMakingStepResult:
        """
        Execute one step in the environment.
        
        Args:
            action: Action from RL agent
            price_change: Price change in this step (absolute)
            order_flow: Net order flow (positive = more buys)
            vpin: Current VPIN toxicity
            
        Returns:
            Step result with reward and metrics
        """
        with self._lock:
            self.vpin = vpin
            old_mid_price = self.mid_price
            self.mid_price += price_change
            self.time_step += 1
            
            # Get quotes from action
            bid, ask = action.get_quotes(old_mid_price)
            
            # Simulate order executions based on order flow and quotes
            trades_executed = 0
            execution_pnl = 0.0
            adverse_cost = 0.0
            
            # Probability of execution based on spread and order flow
            buy_prob = self._execution_probability(action.kappa, bid, old_mid_price, order_flow, side='buy')
            sell_prob = self._execution_probability(action.kappa, ask, old_mid_price, order_flow, side='sell')
            
            # Execute buy orders (we provide liquidity at bid)
            if np.random.random() < buy_prob and abs(self.inventory) < self.max_inventory:
                trade_size = min(np.random.exponential(10), self.max_inventory - self.inventory)
                if trade_size > 0:
                    self.inventory += trade_size
                    self.cash -= trade_size * bid
                    trades_executed += 1
                    
                    # Check for adverse selection
                    if price_change < 0:
                        adverse_cost += abs(price_change) * trade_size
            
            # Execute sell orders (we provide liquidity at ask)
            if np.random.random() < sell_prob and abs(self.inventory) < self.max_inventory:
                trade_size = min(np.random.exponential(10), self.inventory + self.max_inventory)
                if trade_size > 0:
                    self.inventory -= trade_size
                    self.cash += trade_size * ask
                    trades_executed += 1
                    
                    # Check for adverse selection
                    if price_change > 0:
                        adverse_cost += abs(price_change) * trade_size
            
            # Mark-to-market PnL
            mtm_pnl = self.inventory * price_change
            total_pnl = execution_pnl + mtm_pnl
            
            # Calculate fees
            if trades_executed > 0:
                notional = trades_executed * old_mid_price * 10  # Approximate
                fees = notional * self.fee_rate_bps / 10000
                self.cash -= fees
            
            self.total_trades += trades_executed
            self.adverse_selection_total += adverse_cost
            
            # Calculate reward
            reward = self._calculate_reward(
                action=action,
                pnl=total_pnl,
                trades=trades_executed,
                adverse_cost=adverse_cost,
                inventory_risk=self._get_state().inventory_risk
            )
            
            # Check if episode is done
            done = self.time_step >= self.trading_day_minutes
            
            return MarketMakingStepResult(
                reward=reward,
                done=done,
                pnl=total_pnl,
                trades_executed=trades_executed,
                adverse_selection_cost=adverse_cost
            )
    
    def _execution_probability(
        self,
        kappa: float,
        quote_price: float,
        mid_price: float,
        order_flow: float,
        side: str
    ) -> float:
        """
        Calculate probability of order execution.
        Based on Avellaneda-Stoikov model with order flow adjustment.
        """
        distance_from_mid = abs(quote_price - mid_price) / mid_price
        
        # Base probability from AS model
        base_prob = np.exp(-kappa * distance_from_mid * 10000)  # Scale by bps
        
        # Adjust for order flow direction
        if side == 'buy':
            flow_adjustment = 1.0 + 0.5 * np.tanh(order_flow)
        else:
            flow_adjustment = 1.0 - 0.5 * np.tanh(order_flow)
        
        return np.clip(base_prob * flow_adjustment, 0.01, 0.9)
    
    def _calculate_reward(
        self,
        action: MarketMakingAction,
        pnl: float,
        trades: int,
        adverse_cost: float,
        inventory_risk: float
    ) -> float:
        """
        Calculate reward with advanced shaping.
        
        Rewards:
        - Spread capture
        - Inventory management
        
        Penalties:
        - Adverse selection
        - Inventory risk
        - Fee drag
        """
        # Spread capture reward (per trade)
        spread_reward = trades * action.spread_half * 0.0001 * self.mid_price * 10
        
        # PnL component
        pnl_reward = pnl * 0.001  # Scale down
        
        # Adverse selection penalty (heavy)
        adverse_penalty = -2.0 * adverse_cost
        
        # Inventory risk penalty
        inventory_penalty = -0.5 * (inventory_risk ** 2)
        
        # Gamma-based risk penalty (encourage appropriate risk aversion)
        risk_penalty = -0.1 * action.gamma * abs(inventory_risk)
        
        total_reward = spread_reward + pnl_reward + adverse_penalty + inventory_penalty + risk_penalty
        
        return total_reward
    
    def get_action_space_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Get bounds for action space."""
        return {
            'gamma': (0.1, 5.0),        # Risk aversion
            'kappa': (0.5, 5.0),         # Order arrival intensity
            'spread_half': (2.0, 50.0),  # Half-spread in bps
            'skew': (-0.5, 0.5)          # Quote skew in price units
        }
    
    def get_state_features(self) -> np.ndarray:
        """Get state as feature vector for RL agent."""
        state = self._get_state()
        return np.array([
            state.inventory_risk,
            state.vpin,
            state.volatility * 10000,  # Scale up
            state.time_remaining,
            state.spread_bps / 10,     # Normalize
            state.mid_price / 50000,   # Normalize to typical BTC price
            np.log1p(abs(state.inventory)),
            state.skew_factor
        ], dtype=np.float32)


class RLMarketMakingAgent:
    """
    Lightweight RL agent for market making calibration.
    Uses simple policy gradient or DQN that can be exported to ONNX.
    """
    
    def __init__(
        self,
        env: AvellanedaStoikovEnvironment,
        learning_rate: float = 0.001,
        gamma: float = 0.99
    ):
        self.env = env
        self.lr = learning_rate
        self.discount = gamma
        
        self.action_bounds = env.get_action_space_bounds()
        self.state_dim = 8  # From get_state_features
        
        # Simple linear policy parameters (for demonstration)
        # In production, this would be a neural network
        self._policy_weights = np.zeros((self.state_dim, 4))  # 4 action dimensions
        self._policy_bias = np.zeros(4)
        
        self._episode_rewards: List[float] = []
        self._training_mode = False
    
    def select_action(self, state_features: np.ndarray, explore: bool = True) -> MarketMakingAction:
        """Select action based on current policy."""
        # Linear policy
        action_logits = state_features @ self._policy_weights + self._policy_bias
        
        # Convert to action parameters
        action_params = np.tanh(action_logits)  # Bound to [-1, 1]
        
        # Add exploration noise
        if explore and self._training_mode:
            action_params += np.random.randn(4) * 0.1
        
        # Map to actual action bounds
        gamma_val = self._map_to_range(action_params[0], self.action_bounds['gamma'])
        kappa_val = self._map_to_range(action_params[1], self.action_bounds['kappa'])
        spread_val = self._map_to_range(action_params[2], self.action_bounds['spread_half'])
        skew_val = self._map_to_range(action_params[3], self.action_bounds['skew'])
        
        return MarketMakingAction(
            gamma=gamma_val,
            kappa=kappa_val,
            spread_half=spread_val,
            skew=skew_val
        )
    
    def _map_to_range(self, x: float, bounds: Tuple[float, float]) -> float:
        """Map [-1, 1] to actual parameter range."""
        min_val, max_val = bounds
        return min_val + (x + 1) / 2 * (max_val - min_val)
    
    def update_policy(
        self,
        states: List[np.ndarray],
        actions: List[MarketMakingAction],
        rewards: List[float],
        next_states: List[np.ndarray],
        dones: List[bool]
    ) -> float:
        """
        Update policy using REINFORCE-style update.
        Returns the mean reward for the batch.
        """
        if len(states) == 0:
            return 0.0
        
        # Calculate discounted returns
        returns = []
        G = 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            G = r + self.discount * G * (not d)
            returns.insert(0, G)
        
        returns = np.array(returns)
        
        # Normalize returns
        if len(returns) > 1:
            returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
        
        # Simple policy gradient update
        for i, (state, action, ret) in enumerate(zip(states, actions, returns)):
            # Compute action gradient (simplified)
            action_vector = np.array([
                action.gamma,
                action.kappa,
                action.spread_half,
                action.skew
            ])
            
            # Gradient update
            grad = np.outer(state, action_vector) * ret
            self._policy_weights += self.lr * grad
            self._policy_bias += self.lr * action_vector * ret
        
        mean_reward = np.mean(rewards)
        self._episode_rewards.append(mean_reward)
        
        return mean_reward
    
    def set_training_mode(self, training: bool) -> None:
        """Set training mode."""
        self._training_mode = training
    
    def get_policy_stats(self) -> Dict[str, Any]:
        """Get statistics about current policy."""
        return {
            'mean_gamma': np.mean(self._policy_weights[:, 0]),
            'mean_kappa': np.mean(self._policy_weights[:, 1]),
            'mean_spread': np.mean(self._policy_weights[:, 2]),
            'mean_skew': np.mean(self._policy_weights[:, 3]),
            'episodes_trained': len(self._episode_rewards),
            'avg_reward': np.mean(self._episode_rewards[-100:]) if len(self._episode_rewards) >= 100 else np.mean(self._episode_rewards) if self._episode_rewards else 0.0
        }


# Global singleton
_rl_agent_instance: Optional[RLMarketMakingAgent] = None
_rl_lock = threading.Lock()


def get_rl_agent(env: Optional[AvellanedaStoikovEnvironment] = None) -> RLMarketMakingAgent:
    """Thread-safe singleton access to RL agent."""
    global _rl_agent_instance
    
    with _rl_lock:
        if _rl_agent_instance is None:
            if env is None:
                env = AvellanedaStoikovEnvironment()
            _rl_agent_instance = RLMarketMakingAgent(env)
        
        return _rl_agent_instance


if __name__ == "__main__":
    # Demo usage
    np.random.seed(42)
    
    # Create environment
    env = AvellanedaStoikovEnvironment(
        initial_cash=1_000_000,
        max_inventory=50.0
    )
    
    # Create agent
    agent = RLMarketMakingAgent(env, learning_rate=0.001)
    agent.set_training_mode(True)
    
    # Run training episode
    state = env.reset(initial_mid_price=50000.0)
    episode_rewards = []
    
    for step in range(100):
        # Get state features
        features = env.get_state_features()
        
        # Select action
        action = agent.select_action(features, explore=True)
        
        # Simulate market dynamics
        price_change = np.random.randn() * env.volatility * env.mid_price
        order_flow = np.random.randn() * 0.5
        vpin = 0.3 + np.random.beta(2, 5) * 0.4
        
        # Execute step
        result = env.step(action, price_change, order_flow, vpin)
        episode_rewards.append(result.reward)
        
        if result.done:
            break
    
    print(f"Episode completed: {len(episode_rewards)} steps")
    print(f"Mean reward: {np.mean(episode_rewards):.4f}")
    print(f"Total trades: {env.total_trades}")
    print(f"Adverse selection cost: {env.adverse_selection_total:.2f}")
    
    # Show policy stats
    stats = agent.get_policy_stats()
    print(f"\nPolicy stats: {stats}")
