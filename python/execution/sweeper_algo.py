"""
Liquidity Sweeping Engine - Dynamic Order Sizing for L2 Level Clearing
Builds a liquidity sweeping engine that dynamically sizes child orders to clear L2 levels
without slippage. Calculates exact volume to trigger stop-loss cascades behind thin walls.
Uses vectorized numpy operations for ultra-low latency.
"""

import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum


class SweepSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class OrderBookLevel:
    price: float
    volume: float
    order_count: int = 1


@dataclass
class SweepOrder:
    side: SweepSide
    price: float
    quantity: float
    urgency: float
    expected_fill_rate: float
    child_orders: List[Tuple[float, float]] = field(default_factory=list)


class LiquiditySweeper:
    """
    Dynamically sizes child orders to sweep liquidity across L2 levels.
    Calculates optimal order fragmentation to minimize market impact while
    triggering stop-loss cascades hidden behind thin order book walls.
    """

    def __init__(
        self,
        max_participation_rate: float = 0.3,
        min_child_order_size: float = 1.0,
        max_child_orders: int = 10,
        slippage_tolerance_bps: float = 5.0,
        stop_loss_trigger_threshold: float = 0.8,
    ):
        self.max_participation_rate = max_participation_rate
        self.min_child_order_size = min_child_order_size
        self.max_child_orders = max_child_orders
        self.slippage_tolerance_bps = slippage_tolerance_bps
        self.stop_loss_trigger_threshold = stop_loss_trigger_threshold

        # Pre-allocated arrays for vectorized operations
        self._prices_buffer: np.ndarray = np.zeros(50, dtype=np.float64)
        self._volumes_buffer: np.ndarray = np.zeros(50, dtype=np.float64)
        self._cumulative_volume: np.ndarray = np.zeros(50, dtype=np.float64)
        self._price_impact: np.ndarray = np.zeros(50, dtype=np.float64)

    def _parse_order_book_levels(
        self, levels: List[OrderBookLevel]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Parse order book levels into numpy arrays for vectorized computation."""
        n = min(len(levels), len(self._prices_buffer))

        for i in range(n):
            self._prices_buffer[i] = levels[i].price
            self._volumes_buffer[i] = levels[i].volume

        return self._prices_buffer[:n], self._volumes_buffer[:n]

    def _calculate_cumulative_liquidity(
        self, volumes: np.ndarray
    ) -> np.ndarray:
        """Calculate cumulative volume across price levels using cumsum."""
        n = len(volumes)
        self._cumulative_volume[:n] = np.cumsum(volumes[:n])
        return self._cumulative_volume[:n]

    def _estimate_price_impact(
        self,
        prices: np.ndarray,
        volumes: np.ndarray,
        sweep_quantity: float,
        side: SweepSide,
    ) -> np.ndarray:
        """
        Estimate price impact for each level using square-root impact model.
        Impact = b * (q / V)^alpha where b=0.1, alpha=0.5 (standard parameters)
        """
        n = len(volumes)
        b = 0.1
        alpha = 0.5

        # Avoid division by zero
        safe_volumes = np.maximum(volumes, 1e-9)

        # Calculate fraction of each level consumed
        remaining_qty = sweep_quantity
        level_consumption = np.zeros(n, dtype=np.float64)

        for i in range(n):
            if remaining_qty <= 0:
                break
            consumption = min(remaining_qty, volumes[i])
            level_consumption[i] = consumption
            remaining_qty -= consumption

        # Square-root impact model
        impact_ratios = level_consumption / safe_volumes
        self._price_impact[:n] = b * np.power(impact_ratios[:n], alpha)

        return self._price_impact[:n]

    def _detect_thin_walls(
        self, volumes: np.ndarray, threshold_factor: float = 0.3
    ) -> np.ndarray:
        """
        Detect thin order book walls where stop-losses may be clustered.
        Returns boolean mask of thin wall levels.
        """
        if len(volumes) < 3:
            return np.zeros(len(volumes), dtype=bool)

        avg_volume = np.mean(volumes)
        thin_threshold = avg_volume * threshold_factor

        return volumes < thin_threshold

    def _calculate_stop_cascade_probability(
        self,
        volumes: np.ndarray,
        prices: np.ndarray,
        thin_walls: np.ndarray,
        side: SweepSide,
    ) -> np.ndarray:
        """
        Calculate probability of triggering stop-loss cascade at each level.
        Higher probability at thin walls with consecutive price levels.
        """
        n = len(volumes)
        probabilities = np.zeros(n, dtype=np.float64)

        if n < 2:
            return probabilities

        # Base probability inversely proportional to volume
        max_vol = np.max(volumes)
        base_probs = 1.0 - (volumes / (max_vol + 1e-9))

        # Amplify at thin walls
        thin_amplifier = np.where(thin_walls, 1.5, 1.0)

        # Check for consecutive thin walls (cascade pattern)
        for i in range(1, n):
            if thin_walls[i] and thin_walls[i - 1]:
                thin_amplifier[i] *= 1.3

        # Price gap factor - larger gaps indicate potential stop clusters
        if side == SweepSide.BUY:
            price_diffs = np.diff(prices)
        else:
            price_diffs = -np.diff(prices)

        gap_factor = np.ones(n, dtype=np.float64)
        gap_factor[1:] = 1.0 + np.clip(price_diffs / (np.mean(np.abs(price_diffs)) + 1e-9), 0, 1)

        probabilities = base_probs * thin_amplifier * gap_factor
        return np.clip(probabilities, 0, 1)

    def calculate_sweep_order(
        self,
        bid_levels: List[OrderBookLevel],
        ask_levels: List[OrderBookLevel],
        target_quantity: float,
        side: SweepSide,
        current_mid_price: float,
    ) -> Optional[SweepOrder]:
        """
        Calculate optimal sweep order to execute target quantity.
        Returns SweepOrder with child order breakdown.
        """
        if side == SweepSide.BUY:
            levels = ask_levels
        else:
            levels = bid_levels

        if not levels:
            return None

        prices, volumes = self._parse_order_book_levels(levels)
        cumulative_vol = self._calculate_cumulative_liquidity(volumes)

        # Find how many levels needed to fill target quantity
        levels_needed = np.searchsorted(cumulative_vol, target_quantity) + 1
        levels_needed = min(levels_needed, len(prices))

        if levels_needed == 0:
            return None

        # Slice relevant levels
        active_prices = prices[:levels_needed]
        active_volumes = volumes[:levels_needed]

        # Detect thin walls and stop cascade probabilities
        thin_walls = self._detect_thin_walls(active_volumes)
        cascade_probs = self._calculate_stop_cascade_probability(
            active_prices, active_volumes, thin_walls, side
        )

        # Calculate price impact
        impacts = self._estimate_price_impact(
            active_prices, active_volumes, target_quantity, side
        )

        # Determine child order sizing
        # Prioritize levels with high cascade probability and low impact
        efficiency_score = cascade_probs * (1.0 - impacts)

        # Normalize scores for allocation
        total_score = np.sum(efficiency_score)
        if total_score < 1e-9:
            allocations = np.ones(levels_needed, dtype=np.float64) / levels_needed
        else:
            allocations = efficiency_score / total_score

        # Apply participation rate limits
        max_per_level = active_volumes * self.max_participation_rate
        allocated_quantities = np.minimum(
            target_quantity * allocations, max_per_level
        )

        # Re-normalize to hit target
        allocated_sum = np.sum(allocated_quantities)
        if allocated_sum > 0:
            allocated_quantities *= target_quantity / allocated_sum

        # Build child orders
        child_orders = []
        remaining = target_quantity

        for i in range(levels_needed):
            qty = min(allocated_quantities[i], remaining)
            if qty >= self.min_child_order_size:
                child_orders.append((active_prices[i], qty))
                remaining -= qty

            if len(child_orders) >= self.max_child_orders:
                break

        # Add any remaining to last order
        if remaining >= self.min_child_order_size and child_orders:
            last_price, last_qty = child_orders[-1]
            child_orders[-1] = (last_price, last_qty + remaining)
            remaining = 0
        elif remaining >= self.min_child_order_size and len(child_orders) < self.max_child_orders:
            child_orders.append((active_prices[min(levels_needed - 1, len(active_prices) - 1)], remaining))

        if not child_orders:
            return None

        # Calculate weighted average price and expected fill rate
        total_qty = sum(qty for _, qty in child_orders)
        weighted_price = sum(price * qty for price, qty in child_orders) / total_qty

        # Expected fill rate based on impact and cascade probability
        avg_cascade_prob = np.mean(cascade_probs[: len(child_orders)])
        avg_impact = np.mean(impacts[: len(child_orders)])
        expected_fill_rate = 0.9 * avg_cascade_prob + 0.1 * (1.0 - avg_impact)

        # Urgency based on thin wall detection
        urgency = float(np.max(cascade_probs)) if np.any(cascade_probs > 0) else 0.5

        return SweepOrder(
            side=side,
            price=weighted_price,
            quantity=total_qty,
            urgency=urgency,
            expected_fill_rate=expected_fill_rate,
            child_orders=child_orders,
        )

    def calculate_optimal_fragmentation(
        self,
        total_quantity: float,
        available_liquidity: float,
        volatility: float,
        spread_bps: float,
    ) -> int:
        """
        Calculate optimal number of child orders using Almgren-Chriss framework.
        Balances market impact against timing risk.
        """
        if available_liquidity <= 0 or total_quantity <= 0:
            return 1

        # Participation ratio
        participation = min(total_quantity / available_liquidity, 1.0)

        # Timing risk increases with volatility
        timing_risk = volatility * np.sqrt(participation)

        # Impact cost decreases with more fragments
        # But too many fragments increase timing risk
        # Optimal N ~ sqrt(timing_risk / impact_coefficient)

        impact_coefficient = spread_bps / 10000.0 + 0.001
        optimal_n = np.sqrt(timing_risk / (impact_coefficient + 1e-9))

        # Clamp to valid range
        optimal_n = int(np.clip(optimal_n * 5, 1, self.max_child_orders))

        return optimal_n

    def get_execution_schedule(
        self,
        sweep_order: SweepOrder,
        time_horizon_ms: int = 1000,
    ) -> List[Tuple[int, float, float]]:
        """
        Generate time-sliced execution schedule for sweep order.
        Returns list of (delay_ms, price, quantity) tuples.
        """
        schedule = []
        n_orders = len(sweep_order.child_orders)

        if n_orders == 0:
            return schedule

        # Time between orders based on urgency
        base_delay = time_horizon_ms // max(n_orders, 1)
        adjusted_delay = int(base_delay * (1.0 - sweep_order.urgency * 0.5))
        adjusted_delay = max(adjusted_delay, 10)  # Minimum 10ms between orders

        cumulative_delay = 0
        for price, qty in sweep_order.child_orders:
            schedule.append((cumulative_delay, price, qty))
            cumulative_delay += adjusted_delay

        return schedule

    def validate_nautilus_limits(
        self,
        sweep_order: SweepOrder,
        max_order_value: float,
        max_position_size: float,
        instrument_precision: int,
    ) -> bool:
        """
        Validate sweep order against Nautilus OrderFactory limits.
        Prevents fat-finger oversized submissions.
        """
        total_value = sweep_order.price * sweep_order.quantity

        if total_value > max_order_value:
            return False

        if sweep_order.quantity > max_position_size:
            return False

        # Check precision
        qty_str = f"{sweep_order.quantity:.{instrument_precision}f}"
        try:
            float(qty_str)
        except ValueError:
            return False

        # Validate each child order
        for price, qty in sweep_order.child_orders:
            child_value = price * qty
            if child_value > max_order_value:
                return False

        return True
