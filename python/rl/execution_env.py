"""
Custom Gymnasium Environment for RL Training
Wraps Nautilus execution engine for training agents on optimal TWAP/VWAP pacing.
Simulates market impact, slippage, and queue dynamics.
"""

import numpy as np
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass, field
import gymnasium as gym
from gymnasium import spaces
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OrderState:
    """State of an order in the execution environment."""
    remaining_quantity: float
    filled_quantity: float
    average_price: float
    start_time: float
    current_time: float
    side: int  # 1 = buy, -1 = sell
    instrument_id: str


@dataclass
class MarketState:
    """Current market state for simulation."""
    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float
    mid_price: float
    spread: float
    volatility: float
    volume_profile: np.ndarray = field(default_factory=lambda: np.zeros(10))


@dataclass
class ExecutionConfig:
    """Configuration for execution environment."""
    max_steps: int = 100
    time_limit_seconds: float = 3600.0  # 1 hour
    initial_quantity: float = 1.0
    commission_rate: float = 0.0005
    market_impact_coefficient: float = 0.01
    slippage_model: str = "linear"  # linear, quadratic, sqrt
    reward_scale: float = 1.0
    penalty_for_unfilled: float = 10.0


class ExecutionEnv(gym.Env):
    """
    Custom Gymnasium environment for execution optimization.
    Trains agents on optimal TWAP/VWAP pacing with market impact simulation.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"]}
    
    def __init__(self, 
                 config: Optional[ExecutionConfig] = None,
                 render_mode: Optional[str] = None):
        super().__init__()
        
        self.config = config or ExecutionConfig()
        self.render_mode = render_mode
        
        # Action space: [participation_rate, aggression_level]
        # participation_rate: 0.0 to 1.0 (fraction of available liquidity)
        # aggression_level: 0.0 to 1.0 (0 = passive limit, 1 = aggressive market)
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Observation space
        # [normalized_remaining_qty, normalized_time, spread, volatility, 
        #  price_momentum, order_imbalance, recent_fill_rate, avg_slippage]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(8,),
            dtype=np.float32
        )
        
        # State variables
        self._order_state: Optional[OrderState] = None
        self._market_state: Optional[MarketState] = None
        self._step_count = 0
        self._total_cost = 0.0
        self._benchmark_cost = 0.0
        self._fills: List[Dict[str, Any]] = []
        
        # Price trajectory simulation
        self._price_path: List[float] = []
        self._true_price: float = 100.0
    
    def reset(self, 
              seed: Optional[int] = None,
              options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment."""
        super().reset(seed=seed)
        
        # Reset order state
        initial_qty = options.get("initial_quantity", self.config.initial_quantity) if options else self.config.initial_quantity
        side = options.get("side", 1) if options else 1
        
        self._order_state = OrderState(
            remaining_quantity=initial_qty,
            filled_quantity=0.0,
            average_price=0.0,
            start_time=0.0,
            current_time=0.0,
            side=side,
            instrument_id=options.get("instrument_id", "BTC/USDT") if options else "BTC/USDT"
        )
        
        # Initialize market state
        self._true_price = options.get("initial_price", 100.0) if options else 100.0
        self._market_state = self._generate_market_state()
        
        # Reset counters
        self._step_count = 0
        self._total_cost = 0.0
        self._benchmark_cost = self._true_price * initial_qty  # Arrival cost benchmark
        self._fills = []
        self._price_path = [self._true_price]
        
        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: [participation_rate, aggression_level]
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        self._step_count += 1
        
        # Parse action
        participation_rate = np.clip(action[0], 0.0, 1.0)
        aggression_level = np.clip(action[1], 0.0, 1.0)
        
        # Calculate order quantity based on participation
        available_liquidity = self._market_state.ask_size if self._order_state.side == 1 else self._market_state.bid_size
        order_qty = min(
            self._order_state.remaining_quantity,
            available_liquidity * participation_rate
        )
        
        if order_qty <= 0:
            # No order placed
            reward = -0.01  # Small penalty for inaction
            self._advance_time()
            return self._get_observation(), reward, False, False, self._get_info()
        
        # Execute order with market impact and slippage
        fill_price, slippage = self._execute_order(order_qty, aggression_level)
        
        # Update order state
        prev_avg = self._order_state.average_price
        prev_filled = self._order_state.filled_quantity
        
        self._order_state.filled_quantity += order_qty
        self._order_state.remaining_quantity -= order_qty
        self._order_state.average_price = (
            (prev_avg * prev_filled + fill_price * order_qty) / 
            self._order_state.filled_quantity
        )
        
        # Track fills
        self._fills.append({
            "quantity": order_qty,
            "price": fill_price,
            "slippage": slippage,
            "aggression": aggression_level,
        })
        
        # Update total cost
        self._total_cost += fill_price * order_qty
        
        # Advance time and market
        self._advance_time()
        self._update_market_state()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check termination
        terminated = self._order_state.remaining_quantity <= 1e-6
        truncated = self._step_count >= self.config.max_steps
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()
    
    def _execute_order(self, quantity: float, aggression: float) -> Tuple[float, float]:
        """
        Execute an order with market impact and slippage.
        
        Args:
            quantity: Order quantity
            aggression: Aggression level (0=passive, 1=aggressive)
        
        Returns:
            Tuple of (fill_price, slippage_bps)
        """
        base_price = self._market_state.ask_price if self._order_state.side == 1 else self._market_state.bid_price
        
        # Market impact model
        if self.config.slippage_model == "linear":
            impact = self.config.market_impact_coefficient * quantity
        elif self.config.slippage_model == "quadratic":
            impact = self.config.market_impact_coefficient * quantity ** 2
        else:  # sqrt
            impact = self.config.market_impact_coefficient * np.sqrt(quantity)
        
        # Aggression reduces spread cost but increases impact
        spread_fraction = self._market_state.spread * (1 - aggression)
        
        # Total slippage
        slippage_bps = (impact + spread_fraction) / base_price * 10000
        
        # Final fill price
        if self._order_state.side == 1:  # Buy
            fill_price = base_price + impact + spread_fraction * 0.5
        else:  # Sell
            fill_price = base_price - impact - spread_fraction * 0.5
        
        # Add commission
        fill_price *= (1 + self.config.commission_rate * self._order_state.side)
        
        return fill_price, slippage_bps
    
    def _calculate_reward(self) -> float:
        """Calculate reward based on execution quality."""
        if self._order_state.filled_quantity <= 0:
            return 0.0
        
        # Implementation shortfall
        actual_cost = self._total_cost
        benchmark = self._benchmark_cost * (self._order_state.filled_quantity / self.config.initial_quantity)
        
        if self._order_state.side == 1:  # Buy
            shortfall = actual_cost - benchmark
        else:  # Sell
            shortfall = benchmark - actual_cost
        
        # Normalize by benchmark
        normalized_shortfall = shortfall / benchmark
        
        # Reward is negative shortfall (we want to minimize cost)
        reward = -normalized_shortfall * self.config.reward_scale
        
        # Penalty for unfilled quantity at end
        if self._order_state.remaining_quantity > 0.01 * self.config.initial_quantity:
            unfilled_penalty = (
                self.config.penalty_for_unfilled * 
                self._order_state.remaining_quantity / self.config.initial_quantity
            )
            reward -= unfilled_penalty
        
        return reward
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation vector."""
        if self._order_state is None or self._market_state is None:
            return np.zeros(8, dtype=np.float32)
        
        # Normalized remaining quantity
        norm_remaining = self._order_state.remaining_quantity / self.config.initial_quantity
        
        # Normalized time elapsed
        time_elapsed = self._order_state.current_time - self._order_state.start_time
        norm_time = min(time_elapsed / self.config.time_limit_seconds, 1.0)
        
        # Spread (normalized)
        norm_spread = self._market_state.spread / self._market_state.mid_price * 10000  # bps
        
        # Volatility
        volatility = self._market_state.volatility
        
        # Price momentum (recent returns)
        if len(self._price_path) >= 5:
            momentum = (self._price_path[-1] - self._price_path[-5]) / self._price_path[-5]
        else:
            momentum = 0.0
        
        # Order imbalance (simplified)
        order_imbalance = (self._market_state.ask_size - self._market_state.bid_size) / \
                         (self._market_state.ask_size + self._market_state.bid_size + 1e-6)
        
        # Recent fill rate
        if len(self._fills) > 0:
            recent_fills = self._fills[-5:]
            fill_rate = sum(f["quantity"] for f in recent_fills) / len(recent_fills)
        else:
            fill_rate = 0.0
        
        # Average slippage
        if len(self._fills) > 0:
            avg_slippage = np.mean([f["slippage"] for f in self._fills])
        else:
            avg_slippage = 0.0
        
        obs = np.array([
            norm_remaining,
            norm_time,
            norm_spread,
            volatility,
            momentum,
            order_imbalance,
            fill_rate,
            avg_slippage,
        ], dtype=np.float32)
        
        return obs
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional info."""
        return {
            "step": self._step_count,
            "remaining_quantity": self._order_state.remaining_quantity if self._order_state else 0,
            "filled_quantity": self._order_state.filled_quantity if self._order_state else 0,
            "average_price": self._order_state.average_price if self._order_state else 0,
            "total_cost": self._total_cost,
            "benchmark_cost": self._benchmark_cost,
            "implementation_shortfall": self._total_cost - self._benchmark_cost,
            "num_fills": len(self._fills),
            "current_price": self._true_price,
        }
    
    def _generate_market_state(self) -> MarketState:
        """Generate a new market state."""
        spread = self._true_price * 0.0001  # 1 bps spread
        bid = self._true_price - spread / 2
        ask = self._true_price + spread / 2
        
        return MarketState(
            bid_price=bid,
            ask_price=ask,
            bid_size=np.random.uniform(0.5, 2.0),
            ask_size=np.random.uniform(0.5, 2.0),
            mid_price=self._true_price,
            spread=spread,
            volatility=np.random.uniform(0.0001, 0.001),
            volume_profile=np.random.uniform(0.1, 1.0, 10)
        )
    
    def _update_market_state(self) -> None:
        """Update market state with price dynamics."""
        # Random walk with mean reversion
        drift = 0.0
        mean_reversion = -0.001 * (self._true_price - 100.0)
        
        shock = np.random.normal(0, self._market_state.volatility)
        self._true_price *= (1 + drift + mean_reversion + shock)
        self._true_price = max(self._true_price, 1.0)  # Floor
        
        self._price_path.append(self._true_price)
        self._market_state = self._generate_market_state()
        self._market_state.mid_price = self._true_price
    
    def _advance_time(self) -> None:
        """Advance simulation time."""
        if self._order_state:
            self._order_state.current_time += self.config.time_limit_seconds / self.config.max_steps
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            print(f"Step: {self._step_count}")
            print(f"Remaining: {self._order_state.remaining_quantity:.4f}")
            print(f"Filled: {self._order_state.filled_quantity:.4f}")
            print(f"Avg Price: {self._order_state.average_price:.2f}")
            print(f"Current Price: {self._true_price:.2f}")
            print("---")
    
    def close(self):
        """Clean up resources."""
        pass


def create_execution_env(config: Optional[ExecutionConfig] = None,
                         render_mode: Optional[str] = None) -> ExecutionEnv:
    """
    Factory function to create execution environment.
    
    Args:
        config: Environment configuration
        render_mode: Render mode
    
    Returns:
        ExecutionEnv instance
    """
    return ExecutionEnv(config, render_mode)


if __name__ == "__main__":
    # Example usage
    print("Execution Environment Demo")
    print("=" * 40)
    
    config = ExecutionConfig(
        max_steps=50,
        initial_quantity=1.0,
        market_impact_coefficient=0.005
    )
    
    env = create_execution_env(config, render_mode="human")
    
    # Test environment
    obs, info = env.reset(seed=42)
    print(f"Initial observation: {obs}")
    print(f"Initial info: {info}")
    
    # Run random actions
    total_reward = 0.0
    for i in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            break
    
    print(f"\nTotal reward: {total_reward:.4f}")
    print(f"Final info: {info}")
    
    env.close()
