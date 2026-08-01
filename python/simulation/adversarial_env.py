"""
Adversarial Simulation Environment.
Generates toxic market conditions for stress-testing circuit breakers and risk models.
Models spoofing, liquidity evaporation, and synthetic black swan events.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class AdversarialEvent:
    """Represents an adversarial market event."""
    event_type: str  # "spoofing", "liquidity_evaporation", "flash_crash", "latency_spike"
    severity: float  # 0-1
    duration_steps: int
    start_step: int
    affected_assets: List[str]


class AdversarialEnv(gym.Env):
    """
    Adversarial simulation environment for stress-testing.
    Generates toxic market conditions including:
    - Spoofing (fake order book depth)
    - Liquidity evaporation
    - Flash crashes
    - Latency spikes
    - Exchange API glitches
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        assets: List[str] = None,
        initial_cash: float = 1_000_000.0,
        base_volatility: float = 0.02,
        adversarial_intensity: float = 0.5,
    ):
        super().__init__()

        self.assets = assets or ["BTC", "ETH", "SOL"]
        self.n_assets = len(self.assets)
        self.initial_cash = initial_cash
        self.base_volatility = base_volatility
        self.adversarial_intensity = adversarial_intensity

        # State
        self._prices: Dict[str, float] = {}
        self._order_book: Dict[str, Dict] = {}
        self._cash = initial_cash
        self._positions: Dict[str, float] = {a: 0.0 for a in self.assets}
        self._step_count = 0

        # Adversarial state
        self._active_events: List[AdversarialEvent] = []
        self._spoofed_depth: Dict[str, float] = {}
        self._liquidity_factor: Dict[str, float] = {}
        self._latency_multiplier = 1.0
        self._black_swan_active = False

        # Circuit breaker state
        self._circuit_breaker_triggered = False
        self._max_drawdown = 0.0
        self._current_drawdown = 0.0

        # Action space: [action_type, asset_idx, size]
        self.action_space = spaces.Box(
            low=np.array([0, 0, 0], dtype=np.float32),
            high=np.array([3, self.n_assets - 1, 1.0], dtype=np.float32),
            shape=(3,),
            dtype=np.float32,
        )

        # Observation space includes adversarial indicators
        obs_dim = 10 + (self.n_assets * 5)
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
        """Reset with optional adversarial scenario."""
        super().reset(seed=seed)

        # Reset basic state
        self._cash = self.initial_cash
        self._positions = {a: 0.0 for a in self.assets}
        self._step_count = 0
        self._circuit_breaker_triggered = False
        self._max_drawdown = 0.0
        self._current_drawdown = 0.0
        self._active_events = []
        self._black_swan_active = False
        self._latency_multiplier = 1.0

        # Initialize prices and order books
        for asset in self.assets:
            self._prices[asset] = 100.0 * (1 + np.random.uniform(-0.1, 0.1))
            self._order_book[asset] = self._generate_normal_orderbook()
            self._spoofed_depth[asset] = 0.0
            self._liquidity_factor[asset] = 1.0

        # Schedule adversarial events if specified
        if options and options.get("scenario"):
            self._schedule_scenario(options["scenario"])

        return self._get_observation(), self._get_info()

    def _generate_normal_orderbook(self) -> Dict:
        """Generate a normal order book."""
        return {
            "bids": [(100 - i * 0.01, 100 + np.random.uniform(0, 50)) for i in range(10)],
            "asks": [(100 + i * 0.01, 100 + np.random.uniform(0, 50)) for i in range(10)],
            "spread_bps": 10,
            "total_bid_depth": 1000,
            "total_ask_depth": 1000,
        }

    def _schedule_scenario(self, scenario: str):
        """Schedule a specific adversarial scenario."""
        if scenario == "spoofing_attack":
            self._active_events.append(AdversarialEvent(
                event_type="spoofing",
                severity=0.8,
                duration_steps=50,
                start_step=10,
                affected_assets=self.assets[:1],
            ))

        elif scenario == "liquidity_crisis":
            self._active_events.append(AdversarialEvent(
                event_type="liquidity_evaporation",
                severity=0.9,
                duration_steps=100,
                start_step=20,
                affected_assets=self.assets,
            ))

        elif scenario == "flash_crash":
            self._active_events.append(AdversarialEvent(
                event_type="flash_crash",
                severity=1.0,
                duration_steps=20,
                start_step=30,
                affected_assets=self.assets,
            ))

        elif scenario == "black_swan":
            self._black_swan_active = True

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute step with adversarial conditions."""
        self._step_count += 1

        # Process active adversarial events
        self._process_adversarial_events()

        # Simulate price movement with adversarial effects
        self._simulate_adversarial_prices()

        # Update order book with adversarial conditions
        self._update_adversarial_orderbook()

        # Execute agent action
        reward = self._execute_action(action)

        # Check circuit breakers
        terminated = self._check_circuit_breakers()

        # Get observation
        obs = self._get_observation()
        info = self._get_info()

        # Reward includes penalty for adverse conditions
        if self._circuit_breaker_triggered:
            reward -= 1.0

        return obs, reward, terminated, False, info

    def _process_adversarial_events(self):
        """Process and update active adversarial events."""
        remaining_events = []

        for event in self._active_events:
            if self._step_count < event.start_step:
                remaining_events.append(event)
                continue

            steps_active = self._step_count - event.start_step

            if steps_active >= event.duration_steps:
                # Event expired, clean up
                self._cleanup_event(event)
            else:
                remaining_events.append(event)
                self._apply_event_effects(event, steps_active / event.duration_steps)

        self._active_events = remaining_events

    def _apply_event_effects(self, event: AdversarialEvent, progress: float):
        """Apply effects of an active adversarial event."""
        severity = event.severity * progress

        if event.event_type == "spoofing":
            for asset in event.affected_assets:
                self._spoofed_depth[asset] = severity * 10000  # Fake depth

        elif event.event_type == "liquidity_evaporation":
            for asset in event.affected_assets:
                self._liquidity_factor[asset] = max(0.1, 1.0 - severity * 0.9)

        elif event.event_type == "flash_crash":
            self._latency_multiplier = 1.0 + severity * 10

        elif event.event_type == "latency_spike":
            self._latency_multiplier = 1.0 + severity * 5

    def _cleanup_event(self, event: AdversarialEvent):
        """Clean up after an event expires."""
        for asset in event.affected_assets:
            self._spoofed_depth[asset] = 0.0
            self._liquidity_factor[asset] = 1.0
        self._latency_multiplier = 1.0

    def _simulate_adversarial_prices(self):
        """Simulate price movements with adversarial effects."""
        for asset in self.assets:
            vol = self.base_volatility

            # Increase volatility during adversarial events
            if self._liquidity_factor[asset] < 1.0:
                vol *= (2.0 - self._liquidity_factor[asset])

            # Black swan: extreme moves
            if self._black_swan_active and np.random.random() < 0.05:
                # 5% chance of extreme move
                shock = np.random.choice([-0.2, -0.15, -0.1, 0.1, 0.15])
                new_price = self._prices[asset] * (1 + shock)
            else:
                # Normal GBM
                drift = 0.0
                diffusion = vol * np.random.randn()
                new_price = self._prices[asset] * (1 + drift + diffusion)

            self._prices[asset] = max(new_price, 0.01)

    def _update_adversarial_orderbook(self):
        """Update order book reflecting adversarial conditions."""
        for asset in self.assets:
            liq_factor = self._liquidity_factor[asset]
            spoof_depth = self._spoofed_depth[asset]

            # Reduce real depth
            base_depth = 1000 * liq_factor

            ob = {
                "bids": [],
                "asks": [],
                "spread_bps": int(10 / liq_factor),  # Wider spread
                "total_bid_depth": base_depth + spoof_depth,
                "total_ask_depth": base_depth + spoof_depth,
                "is_spoofed": spoof_depth > 0,
            }

            # Generate bid/ask levels
            price = self._prices[asset]
            for i in range(10):
                bid_price = price * (1 - (i + 1) * 0.001 * (2 - liq_factor))
                ask_price = price * (1 + (i + 1) * 0.001 * (2 - liq_factor))

                # First level has spoofed depth
                bid_depth = base_depth / 10
                ask_depth = base_depth / 10

                if i == 0 and spoof_depth > 0:
                    bid_depth += spoof_depth  # Fake bid wall

                ob["bids"].append((bid_price, bid_depth))
                ob["asks"].append((ask_price, ask_depth))

            self._order_book[asset] = ob

    def _execute_action(self, action: np.ndarray) -> float:
        """Execute agent action with adversarial execution quality."""
        action_type = int(np.clip(action[0], 0, 3))
        asset_idx = int(np.clip(action[1], 0, self.n_assets - 1))
        size_pct = np.clip(action[2], 0, 1)

        asset = self.assets[asset_idx]
        price = self._prices[asset]

        if action_type == 0:  # Hold
            return 0.0

        elif action_type == 1:  # Buy
            max_buy = self._cash / price
            size = max_buy * size_pct

            # Slippage is worse during adversarial conditions
            liq_factor = self._liquidity_factor[asset]
            slippage = 0.001 * (2 - liq_factor) * size_pct

            self._cash -= size * price * (1 + slippage)
            self._positions[asset] += size

        elif action_type == 2:  # Sell
            size = abs(self._positions[asset]) * size_pct

            liq_factor = self._liquidity_factor[asset]
            slippage = 0.001 * (2 - liq_factor) * size_pct

            self._cash += size * price * (1 - slippage)
            self._positions[asset] -= size

        elif action_type == 3:  # Hedge
            # Automatic delta hedge
            self._hedge_position(asset)

        return -0.001  # Transaction cost penalty

    def _hedge_position(self, asset: str):
        """Execute automatic hedging."""
        pos = self._positions[asset]
        if pos > 0:
            hedge_size = min(pos, self._cash / self._prices[asset])
            self._positions[asset] -= hedge_size
            self._cash += hedge_size * self._prices[asset] * 0.999
        elif pos < 0:
            hedge_size = min(abs(pos), self._cash / self._prices[asset])
            self._positions[asset] += hedge_size
            self._cash -= hedge_size * self._prices[asset] * 1.001

    def _check_circuit_breakers(self) -> bool:
        """Check if circuit breakers should trigger."""
        # Calculate current PnL
        total_value = self._cash
        for asset in self.assets:
            total_value += self._positions[asset] * self._prices[asset]

        pnl_pct = (total_value - self.initial_cash) / self.initial_cash
        self._current_drawdown = min(0, pnl_pct)
        self._max_drawdown = min(self._max_drawdown, self._current_drawdown)

        # Circuit breaker triggers
        if self._max_drawdown < -0.1:  # 10% drawdown
            logger.warning(f"Circuit breaker triggered! Drawdown: {self._max_drawdown:.2%}")
            self._circuit_breaker_triggered = True
            return True

        # Check for exchange glitch detection
        for asset in self.assets:
            ob = self._order_book[asset]
            if ob.get("is_spoofed") and ob["spread_bps"] > 100:
                logger.warning(f"Exchange glitch detected for {asset}")
                self._circuit_breaker_triggered = True
                return True

        return False

    def _get_observation(self) -> np.ndarray:
        """Get observation including adversarial indicators."""
        obs = [
            self._cash / self.initial_cash,
            self._current_drawdown,
            self._max_drawdown,
            1.0 if self._circuit_breaker_triggered else 0.0,
            self._latency_multiplier,
            self._step_count / 1000,
            float(self._black_swan_active),
            sum(1 for e in self._active_events),
            np.mean([self._liquidity_factor[a] for a in self.assets]),
            np.mean([self._spoofed_depth[a] for a in self.assets]) / 10000,
        ]

        for asset in self.assets:
            obs.extend([
                self._positions[asset],
                self._prices[asset] / 100,
                self._liquidity_factor[asset],
                self._spoofed_depth[asset] / 10000,
                self._order_book[asset]["spread_bps"] / 100,
            ])

        return np.array(obs, dtype=np.float32)

    def _get_info(self) -> Dict[str, Any]:
        """Get additional info."""
        return {
            "cash": self._cash,
            "positions": dict(self._positions),
            "prices": dict(self._prices),
            "active_events": [e.event_type for e in self._active_events],
            "circuit_breaker_triggered": self._circuit_breaker_triggered,
            "max_drawdown": self._max_drawdown,
            "liquidity_factors": dict(self._liquidity_factor),
            "latency_multiplier": self._latency_multiplier,
            "black_swan_active": self._black_swan_active,
        }

    def render(self, mode="human"):
        """Render the environment."""
        if mode == "human":
            print(f"Step: {self._step_count}")
            print(f"Active Events: {[e.event_type for e in self._active_events]}")
            print(f"Circuit Breaker: {self._circuit_breaker_triggered}")
            print(f"Max Drawdown: {self._max_drawdown:.2%}")
            print(f"Latency Multiplier: {self._latency_multiplier:.2f}x")
            print("---")


def register_adversarial_env():
    """Register the adversarial environment."""
    from gymnasium.envs.registration import register

    register(
        id="AdversarialEnv-v0",
        entry_point="python.simulation.adversarial_env:AdversarialEnv",
        max_episode_steps=1000,
    )
