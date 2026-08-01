"""
Reward Shaping for RL Market Making.
Implements advanced reward shaping that heavily penalizes adverse selection
and toxic inventory accumulation while rewarding spread capture.

Designed to train agents that maintain flat delta and minimize market impact.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RewardComponents:
    """Decomposed reward components for interpretability."""
    spread_capture: float
    pnl_component: float
    adverse_selection_penalty: float
    inventory_penalty: float
    fee_penalty: float
    skew_reward: float
    stability_bonus: float
    total_reward: float


class RewardShaper:
    """
    Advanced reward shaping for market making RL.
    
    Key principles:
    1. Heavy penalty for adverse selection (toxic flow)
    2. Penalty for inventory accumulation in wrong direction
    3. Reward for spread capture with flat inventory
    4. Bonus for maintaining stable quoting behavior
    5. Penalty for excessive market impact
    """
    
    def __init__(
        self,
        # Penalty weights
        adverse_selection_weight: float = 5.0,
        inventory_risk_weight: float = 2.0,
        fee_weight: float = 1.0,
        
        # Reward weights  
        spread_capture_weight: float = 1.0,
        pnl_weight: float = 0.5,
        skew_efficiency_weight: float = 0.3,
        stability_weight: float = 0.2,
        
        # Thresholds
        max_inventory_ratio: float = 0.8,
        adverse_selection_threshold_bps: float = 10.0
    ):
        self.adverse_selection_weight = adverse_selection_weight
        self.inventory_risk_weight = inventory_risk_weight
        self.fee_weight = fee_weight
        self.spread_capture_weight = spread_capture_weight
        self.pnl_weight = pnl_weight
        self.skew_efficiency_weight = skew_efficiency_weight
        self.stability_weight = stability_weight
        
        self.max_inventory_ratio = max_inventory_ratio
        self.adverse_selection_threshold_bps = adverse_selection_threshold_bps
        
        # State tracking for stability calculations
        self._prev_spread: Optional[float] = None
        self._prev_skew: Optional[float] = None
        self._action_history: List[Dict[str, float]] = []
    
    def calculate_reward(
        self,
        # Current state
        mid_price: float,
        inventory: float,
        max_inventory: float,
        
        # Action taken
        spread_half_bps: float,
        skew: float,
        
        # Execution results
        trades_executed: int,
        buy_volume: float,
        sell_volume: float,
        avg_execution_price: float,
        
        # Market outcomes
        price_change_after_trade: float,
        vpin: float,
        
        # Costs
        fees_paid: float,
        slippage_cost: float
    ) -> RewardComponents:
        """
        Calculate shaped reward from trading results.
        
        Args:
            mid_price: Current mid price
            inventory: Current inventory position
            max_inventory: Maximum allowed inventory
            spread_half_bps: Half-spread quoted (in bps)
            skew: Quote skew applied
            trades_executed: Number of trades executed
            buy_volume: Volume bought
            sell_volume: Volume sold
            avg_execution_price: Average execution price
            price_change_after_trade: Price movement after our trades
            vpin: VPIN toxicity measure
            fees_paid: Total fees paid
            slippage_cost: Slippage costs
            
        Returns:
            Decomposed reward components
        """
        # 1. Spread capture reward
        # Reward for capturing spread on round-trip trades
        min_volume = min(buy_volume, sell_volume)
        spread_capture = (
            self.spread_capture_weight * 
            min_volume * mid_price * (spread_half_bps * 2 / 10000)
        )
        
        # 2. PnL component (scaled down to not dominate)
        # Realized + unrealized PnL
        realized_pnl = (sell_volume - buy_volume) * (avg_execution_price - mid_price)
        unrealized_pnl = inventory * price_change_after_trade
        pnl_component = self.pnl_weight * (realized_pnl + unrealized_pnl) * 0.001
        
        # 3. Adverse selection penalty (HEAVY)
        # Penalize when price moves against us after we trade
        adverse_cost = 0.0
        
        if buy_volume > 0 and price_change_after_trade < 0:
            # We bought, price went down
            adverse_cost += abs(price_change_after_trade) * buy_volume
        
        if sell_volume > 0 and price_change_after_trade > 0:
            # We sold, price went up
            adverse_cost += abs(price_change_after_trade) * sell_volume
        
        # Scale by VPIN (higher toxicity = higher penalty multiplier)
        toxicity_multiplier = 1.0 + 2.0 * vpin
        adverse_selection_penalty = (
            -self.adverse_selection_weight * adverse_cost * toxicity_multiplier
        )
        
        # Additional penalty if adverse selection exceeds threshold
        adverse_bps = abs(price_change_after_trade) / mid_price * 10000
        if adverse_bps > self.adverse_selection_threshold_bps:
            adverse_selection_penalty *= 1.5  # Extra penalty for large adverse moves
        
        # 4. Inventory risk penalty
        inventory_ratio = abs(inventory) / max(max_inventory, 1e-8)
        
        # Base inventory penalty (quadratic in inventory ratio)
        inventory_penalty = -self.inventory_risk_weight * (inventory_ratio ** 2)
        
        # Extra penalty if approaching inventory limit
        if inventory_ratio > self.max_inventory_ratio:
            excess_ratio = (inventory_ratio - self.max_inventory_ratio) / (1 - self.max_inventory_ratio)
            inventory_penalty -= 3.0 * (excess_ratio ** 2)
        
        # Directional penalty: penalize inventory that conflicts with recent price movement
        if inventory > 0 and price_change_after_trade < 0:
            inventory_penalty -= 0.5 * inventory_ratio
        elif inventory < 0 and price_change_after_trade > 0:
            inventory_penalty -= 0.5 * inventory_ratio
        
        # 5. Fee penalty
        fee_penalty = -self.fee_weight * fees_paid
        
        # 6. Skew efficiency reward
        # Reward for using skew effectively to manage inventory
        skew_efficiency = 0.0
        if inventory > 0 and skew < 0:
            # Positive inventory, negative skew (encourages sells) - good
            skew_efficiency = abs(skew) / mid_price * sell_volume
        elif inventory < 0 and skew > 0:
            # Negative inventory, positive skew (encourages buys) - good
            skew_efficiency = abs(skew) / mid_price * buy_volume
        
        skew_reward = self.skew_efficiency_weight * skew_efficiency
        
        # 7. Stability bonus
        # Reward for consistent quoting behavior (reduce market impact)
        stability_bonus = self._calculate_stability_bonus(spread_half_bps, skew)
        
        # Calculate total
        total_reward = (
            spread_capture +
            pnl_component +
            adverse_selection_penalty +
            inventory_penalty +
            fee_penalty +
            skew_reward +
            stability_bonus
        )
        
        return RewardComponents(
            spread_capture=float(spread_capture),
            pnl_component=float(pnl_component),
            adverse_selection_penalty=float(adverse_selection_penalty),
            inventory_penalty=float(inventory_penalty),
            fee_penalty=float(fee_penalty),
            skew_reward=float(skew_reward),
            stability_bonus=float(stability_bonus),
            total_reward=float(total_reward)
        )
    
    def _calculate_stability_bonus(
        self,
        current_spread: float,
        current_skew: float
    ) -> float:
        """
        Calculate bonus for stable quoting behavior.
        Penalizes erratic spread/skew changes that increase market impact.
        """
        bonus = 0.0
        
        if self._prev_spread is not None:
            # Penalize large spread changes
            spread_change = abs(current_spread - self._prev_spread)
            spread_penalty = spread_change * 0.1
            bonus -= spread_penalty
        
        if self._prev_skew is not None:
            # Penalize large skew reversals
            if (self._prev_skew > 0.1 and current_skew < -0.1) or \
               (self._prev_skew < -0.1 and current_skew > 0.1):
                bonus -= 0.5  # Penalty for skew reversal
        
        # Update history
        self._prev_spread = current_spread
        self._prev_skew = current_skew
        
        # Track action history for longer-term stability
        self._action_history.append({
            'spread': current_spread,
            'skew': current_skew
        })
        
        # Keep only last 100 actions
        if len(self._action_history) > 100:
            self._action_history.pop(0)
        
        # Bonus for low variance in recent actions
        if len(self._action_history) >= 10:
            recent_spreads = [a['spread'] for a in self._action_history[-10:]]
            spread_std = np.std(recent_spreads)
            if spread_std < 1.0:  # Low spread variance
                bonus += 0.2
        
        return float(bonus)
    
    def reset(self) -> None:
        """Reset state tracking."""
        self._prev_spread = None
        self._prev_skew = None
        self._action_history.clear()
    
    def get_reward_statistics(self) -> Dict[str, float]:
        """Get statistics about recent rewards."""
        if not self._action_history:
            return {}
        
        return {
            'actions_tracked': len(self._action_history),
            'avg_spread': np.mean([a['spread'] for a in self._action_history]),
            'spread_std': np.std([a['spread'] for a in self._action_history]),
            'avg_skew': np.mean([a['skew'] for a in self._action_history]),
            'skew_std': np.std([a['skew'] for a in self._action_history])
        }


class ToxicInventoryDetector:
    """
    Detects toxic inventory accumulation patterns.
    Used to provide additional penalty signals to the RL agent.
    """
    
    def __init__(
        self,
        lookback_window: int = 20,
        toxicity_threshold: float = 0.7
    ):
        self.lookback_window = lookback_window
        self.toxicity_threshold = toxicity_threshold
        
        self._price_history: List[float] = []
        self._inventory_history: List[float] = []
        self._trade_directions: List[int] = []  # +1 for buy, -1 for sell
    
    def update(
        self,
        price: float,
        inventory: float,
        trade_direction: int
    ) -> bool:
        """
        Update detector with new observation.
        
        Returns:
            True if toxic pattern detected
        """
        self._price_history.append(price)
        self._inventory_history.append(inventory)
        self._trade_directions.append(trade_direction)
        
        # Trim history
        if len(self._price_history) > self.lookback_window:
            self._price_history.pop(0)
            self._inventory_history.pop(0)
            self._trade_directions.pop(0)
        
        return self._detect_toxicity()
    
    def _detect_toxicity(self) -> bool:
        """Detect toxic inventory patterns."""
        if len(self._price_history) < 10:
            return False
        
        # Check if we're accumulating inventory while price moves against us
        recent_prices = self._price_history[-10:]
        recent_inventory = self._inventory_history[-10:]
        recent_trades = self._trade_directions[-10:]
        
        # Price trend
        price_trend = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
        
        # Inventory trend
        inventory_trend = recent_inventory[-1] - recent_inventory[0]
        
        # Trade direction bias
        trade_bias = np.mean(recent_trades)
        
        # Toxic patterns:
        # 1. Buying (positive trade bias) while price declining AND inventory increasing
        if trade_bias > 0.3 and price_trend < -0.001 and inventory_trend > 0:
            return True
        
        # 2. Selling (negative trade bias) while price rising AND inventory decreasing
        if trade_bias < -0.3 and price_trend > 0.001 and inventory_trend < 0:
            return True
        
        # 3. High VPIN-like pattern: one-sided trades with adverse price movement
        if abs(trade_bias) > 0.7:
            expected_direction = 1 if trade_bias > 0 else -1
            actual_price_move = 1 if price_trend > 0 else -1
            
            if expected_direction != actual_price_move:
                return True
        
        return False
    
    def get_toxicity_score(self) -> float:
        """
        Get continuous toxicity score [0, 1].
        Higher = more toxic pattern.
        """
        if len(self._price_history) < 10:
            return 0.0
        
        recent_prices = self._price_history[-10:]
        recent_inventory = self._inventory_history[-10:]
        recent_trades = self._trade_directions[-10:]
        
        price_trend = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
        inventory_trend = recent_inventory[-1] - recent_inventory[0]
        trade_bias = np.mean(recent_trades)
        
        # Calculate toxicity indicators
        indicators = []
        
        # Indicator 1: Adverse selection
        if trade_bias * price_trend < 0:
            indicators.append(abs(trade_bias) * abs(price_trend) * 1000)
        
        # Indicator 2: Inventory building against trend
        if inventory_trend * price_trend < 0:
            indicators.append(abs(inventory_trend) / max(abs(recent_inventory[0]), 1) * abs(price_trend) * 100)
        
        if not indicators:
            return 0.0
        
        # Normalize to [0, 1]
        raw_score = np.mean(indicators)
        return float(np.clip(raw_score, 0.0, 1.0))
    
    def reset(self) -> None:
        """Reset detector state."""
        self._price_history.clear()
        self._inventory_history.clear()
        self._trade_directions.clear()


if __name__ == "__main__":
    # Demo usage
    np.random.seed(42)
    
    shaper = RewardShaper(
        adverse_selection_weight=5.0,
        inventory_risk_weight=2.0
    )
    
    detector = ToxicInventoryDetector()
    
    # Simulate trading scenario
    mid_price = 50000.0
    inventory = 0.0
    max_inventory = 100.0
    
    print("=== Reward Shaping Demo ===\n")
    
    for step in range(20):
        # Simulate trade
        spread_bps = 10.0 + np.random.randn() * 2
        skew = np.random.randn() * 0.2
        
        buy_vol = np.random.exponential(5) if np.random.random() > 0.5 else 0
        sell_vol = np.random.exponential(5) if np.random.random() > 0.5 else 0
        
        # Simulate adverse selection
        if buy_vol > sell_vol:
            price_change = -abs(np.random.randn()) * 5  # Price drops after we buy
        elif sell_vol > buy_vol:
            price_change = abs(np.random.randn()) * 5  # Price rises after we sell
        else:
            price_change = np.random.randn() * 3
        
        vpin = 0.3 + np.random.beta(2, 5) * 0.5
        fees = (buy_vol + sell_vol) * mid_price * 0.00025
        
        # Calculate reward
        reward = shaper.calculate_reward(
            mid_price=mid_price,
            inventory=inventory,
            max_inventory=max_inventory,
            spread_half_bps=spread_bps,
            skew=skew,
            trades_executed=(1 if buy_vol > 0 else 0) + (1 if sell_vol > 0 else 0),
            buy_volume=buy_vol,
            sell_volume=sell_vol,
            avg_execution_price=mid_price,
            price_change_after_trade=price_change,
            vpin=vpin,
            fees_paid=fees,
            slippage_cost=0.0
        )
        
        # Update detector
        trade_dir = 1 if buy_vol > sell_vol else (-1 if sell_vol > buy_vol else 0)
        is_toxic = detector.update(mid_price, inventory, trade_dir)
        toxicity_score = detector.get_toxicity_score()
        
        print(f"Step {step+1}:")
        print(f"  Total Reward: {reward.total_reward:.4f}")
        print(f"  Components:")
        print(f"    Spread Capture: {reward.spread_capture:.4f}")
        print(f"    Adverse Penalty: {reward.adverse_selection_penalty:.4f}")
        print(f"    Inventory Penalty: {reward.inventory_penalty:.4f}")
        print(f"  Toxic Pattern: {is_toxic}, Score: {toxicity_score:.4f}")
        print()
        
        # Update state
        inventory += buy_vol - sell_vol
        mid_price += price_change
    
    # Show statistics
    stats = shaper.get_reward_statistics()
    print(f"Reward Statistics: {stats}")
