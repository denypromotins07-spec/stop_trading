"""
Portfolio Gymnasium Environment.
Multi-asset portfolio environment with realistic fee, slippage, and latency models.
Trains RL agents for global delta, gamma, and cross-margin management.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PortfolioState:
    """Current portfolio state."""
    cash: float
    positions: Dict[str, float]
    unrealized_pnl: float
    realized_pnl: float
    delta: float
    gamma: float
    margin_used: float
    margin_available: float


class PortfolioEnv(gym.Env):
    """
    Multi-asset portfolio Gymnasium environment.
    Realistic simulation of fees, slippage, and latency.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    # Action space: [asset_idx, side, size, order_type]
    # side: 0=hold, 1=buy, 2=sell
    # order_type: 0=market, 1=limit, 2=post_only

    def __init__(
        self,
        assets: List[str] = None,
        initial_cash: float = 1_000_000.0,
        max_positions: int = 10,
        fee_bps: float = 5.0,
        slippage_model: str = "linear",
        latency_ms: float = 10.0,
        margin_requirement: float = 0.1,
    ):
        super().__init__()

        self.assets = assets or ["BTC", "ETH", "SOL"]
        self.n_assets = len(self.assets)
        self.initial_cash = initial_cash
        self.max_positions = max_positions
        self.fee_bps = fee_bps
        self.slippage_model = slippage_model
        self.latency_ms = latency_ms
        self.margin_requirement = margin_requirement

        # Price simulation
        self._prices: Dict[str, float] = {}
        self._price_history: Dict[str, List[float]] = {}
        self._volatility: Dict[str, float] = {}

        # Portfolio state
        self._cash = initial_cash
        self._positions: Dict[str, float] = {a: 0.0 for a in self.assets}
        self._avg_entry: Dict[str, float] = {a: 0.0 for a in self.assets}
        self._realized_pnl = 0.0

        # Risk metrics
        self._delta = 0.0
        self._gamma = 0.0
        self._margin_used = 0.0

        # Episode tracking
        self._step_count = 0
        self._max_steps = 10000

        # Action space
        self.action_space = spaces.Box(
            low=np.array([0, 0, 0, 0], dtype=np.float32),
            high=np.array([self.n_assets - 1, 2, 1.0, 2], dtype=np.float32),
            shape=(4,),
            dtype=np.float32,
        )

        # Observation space
        obs_dim = 5 + (self.n_assets * 4)  # cash, pnl, delta, gamma, margin + per asset
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """Reset the environment."""
        super().reset(seed=seed)

        # Reset portfolio
        self._cash = self.initial_cash
        self._positions = {a: 0.0 for a in self.assets}
        self._avg_entry = {a: 0.0 for a in self.assets}
        self._realized_pnl = 0.0
        self._delta = 0.0
        self._gamma = 0.0
        self._margin_used = 0.0

        # Initialize prices
        base_prices = options.get("base_prices", {}) if options else {}
        for asset in self.assets:
            price = base_prices.get(asset, 100.0 * (1 + np.random.uniform(-0.1, 0.1)))
            self._prices[asset] = price
            self._price_history[asset] = [price]
            self._volatility[asset] = options.get("volatility", {}).get(asset, 0.02)

        self._step_count = 0

        return self._get_observation(), self._get_info()

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        self._step_count += 1

        # Parse action
        asset_idx = int(np.clip(action[0], 0, self.n_assets - 1))
        side = int(np.clip(action[1], 0, 2))
        size_pct = np.clip(action[2], 0, 1)
        order_type = int(np.clip(action[3], 0, 2))

        asset = self.assets[asset_idx]

        # Simulate market movement first
        self._simulate_price_movement()

        # Execute trade if not holding
        reward = 0.0
        if side > 0:  # Buy or Sell
            reward = self._execute_trade(asset, side, size_pct, order_type)

        # Calculate risk metrics
        self._update_risk_metrics()

        # Check termination conditions
        terminated = self._check_termination()
        truncated = self._step_count >= self._max_steps

        # Get new observation
        obs = self._get_observation()

        # Calculate reward (PnL change + risk penalty)
        info = self._get_info()
        reward += self._calculate_reward()

        return obs, reward, terminated, truncated, info

    def _simulate_price_movement(self):
        """Simulate price movements with realistic dynamics."""
        for asset in self.assets:
            vol = self._volatility[asset]

            # Geometric Brownian Motion
            drift = 0.0
            diffusion = vol * np.random.randn()

            # Add some mean reversion
            mean_price = self._price_history[asset][0]
            mean_rev = -0.0001 * (self._prices[asset] - mean_price) / mean_price

            new_price = self._prices[asset] * (1 + drift + diffusion + mean_rev)
            new_price = max(new_price, 0.01)  # Floor

            self._prices[asset] = new_price
            self._price_history[asset].append(new_price)

            # Keep history bounded
            if len(self._price_history[asset]) > 1000:
                self._price_history[asset] = self._price_history[asset][-1000:]

    def _execute_trade(
        self,
        asset: str,
        side: int,
        size_pct: float,
        order_type: int,
    ) -> float:
        """Execute a trade with realistic costs."""
        price = self._prices[asset]
        current_pos = self._positions[asset]

        # Calculate trade size
        max_size = self._cash / price if side == 1 else abs(current_pos)
        trade_size = max_size * size_pct

        if trade_size <= 0:
            return 0.0

        # Calculate slippage
        slippage = self._calculate_slippage(asset, trade_size, side)

        # Calculate fees
        fee_rate = self.fee_bps / 10000
        if order_type == 2:  # Post-only (maker)
            fee_rate *= 0.5  # Reduced maker fee

        trade_value = trade_size * price
        total_cost = trade_value * (slippage + fee_rate)

        if side == 1:  # Buy
            # Check margin
            required_margin = trade_value * self.margin_requirement
            if required_margin > self._margin_used + self._cash:
                return -1.0  # Penalty for invalid trade

            self._cash -= trade_value + total_cost

            # Update average entry
            if current_pos >= 0:
                total_value = self._avg_entry[asset] * current_pos + trade_value
                total_qty = current_pos + trade_size
                self._avg_entry[asset] = total_value / max(total_qty, 0.001)

            self._positions[asset] += trade_size

        else:  # Sell
            self._cash += trade_value - total_cost

            # Realize PnL
            if current_pos > 0:
                pnl = (price - self._avg_entry[asset]) * trade_size
                self._realized_pnl += pnl

            self._positions[asset] -= trade_size

        return -total_cost / trade_value  # Cost as negative reward

    def _calculate_slippage(
        self,
        asset: str,
        size: float,
        side: int,
    ) -> float:
        """Calculate slippage based on trade size and model."""
        if self.slippage_model == "linear":
            # Linear slippage: 1 bps per 1% of daily volume
            base_slippage = 0.0001
            size_impact = size * 0.00001
            return base_slippage + size_impact

        elif self.slippage_model == "square_root":
            # Square root model (more realistic)
            base = 0.0005
            return base * np.sqrt(size)

        return 0.001  # Default

    def _update_risk_metrics(self):
        """Update portfolio risk metrics."""
        # Calculate delta (sum of position values)
        self._delta = sum(
            self._positions[a] * self._prices[a]
            for a in self.assets
        )

        # Simplified gamma (using position concentration)
        position_values = [abs(self._positions[a] * self._prices[a]) for a in self.assets]
        self._gamma = np.std(position_values) if position_values else 0.0

        # Calculate margin used
        self._margin_used = sum(
            abs(self._positions[a] * self._prices[a]) * self.margin_requirement
            for a in self.assets
        )

    def _check_termination(self) -> bool:
        """Check if episode should terminate."""
        # Margin call
        if self._margin_used > self._cash * 5:
            logger.warning("Margin call triggered")
            return True

        # Bankruptcy
        if self._cash < 0:
            logger.warning("Bankruptcy")
            return True

        return False

    def _calculate_reward(self) -> float:
        """Calculate step reward."""
        # PnL-based reward
        total_pnl = self._realized_pnl + self._get_unrealized_pnl()
        pnl_reward = total_pnl / self.initial_cash

        # Risk penalty
        risk_penalty = 0.0
        if abs(self._delta) > self.initial_cash * 0.5:
            risk_penalty -= 0.1
        if self._gamma > self.initial_cash * 0.1:
            risk_penalty -= 0.05

        return pnl_reward + risk_penalty

    def _get_unrealized_pnl(self) -> float:
        """Calculate unrealized PnL."""
        unrealized = 0.0
        for asset in self.assets:
            pos = self._positions[asset]
            if pos != 0:
                unrealized += (self._prices[asset] - self._avg_entry[asset]) * pos
        return unrealized

    def _get_observation(self) -> np.ndarray:
        """Get current observation vector."""
        obs = [
            self._cash / self.initial_cash,
            self._realized_pnl / self.initial_cash,
            self._delta / self.initial_cash,
            self._gamma / self.initial_cash,
            self._margin_used / self.initial_cash,
        ]

        for asset in self.assets:
            pos = self._positions[asset]
            price = self._prices[asset]
            entry = self._avg_entry[asset]
            pct = pos * price / self.initial_cash

            obs.extend([pos, price / 100, (price - entry) / max(entry, 0.01), pct])

        return np.array(obs, dtype=np.float32)

    def _get_info(self) -> Dict[str, Any]:
        """Get additional info."""
        return {
            "cash": self._cash,
            "positions": dict(self._positions),
            "prices": dict(self._prices),
            "realized_pnl": self._realized_pnl,
            "unrealized_pnl": self._get_unrealized_pnl(),
            "delta": self._delta,
            "gamma": self._gamma,
            "margin_used": self._margin_used,
            "step": self._step_count,
        }

    def render(self, mode="human"):
        """Render the environment."""
        if mode == "human":
            print(f"Step: {self._step_count}")
            print(f"Cash: ${self._cash:,.2f}")
            print(f"Positions: {self._positions}")
            print(f"PnL: ${self._realized_pnl + self._get_unrealized_pnl():,.2f}")
            print(f"Delta: ${self._delta:,.2f}, Gamma: ${self._gamma:,.2f}")
            print("---")


# Register environment
def register_portfolio_env():
    """Register the portfolio environment with Gymnasium."""
    from gymnasium.envs.registration import register

    register(
        id="PortfolioEnv-v0",
        entry_point="python.simulation.portfolio_env:PortfolioEnv",
        max_episode_steps=10000,
    )
