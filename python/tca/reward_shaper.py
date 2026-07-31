"""
Reward Shaper - Translates TCA metrics into dense RL reward signals.
Penalizes PPO execution agents for slippage and adverse selection.
Ensures timestamp alignment to prevent look-ahead bias.
Strictly enforces 3GB RAM limit.
"""
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from collections import deque
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RewardSignal:
    """Dense reward signal for PPO execution agent."""
    agent_id: str
    timestamp_ns: int
    base_reward: float
    slippage_penalty: float
    impact_penalty: float
    rebate_bonus: float
    adverse_selection_penalty: float
    total_reward: float
    metadata: Dict[str, Any]


class TCARewardShaper:
    """
    Shapes rewards for PPO execution agents based on TCA metrics.
    Memory-bounded for 3GB limit.
    """
    
    def __init__(self,
                 slippage_scale: float = 1.0,
                 impact_scale: float = 0.5,
                 rebate_scale: float = 0.3,
                 adverse_selection_scale: float = 2.0,
                 max_history: int = 5000):
        """
        Initialize reward shaper.
        
        Args:
            slippage_scale: Scaling factor for slippage penalties
            impact_scale: Scaling factor for market impact penalties
            rebate_scale: Scaling factor for maker rebate bonuses
            adverse_selection_scale: Scaling for adverse selection penalties
            max_history: Maximum reward history to keep
        """
        self.slippage_scale = slippage_scale
        self.impact_scale = impact_scale
        self.rebate_scale = rebate_scale
        self.adverse_selection_scale = adverse_selection_scale
        
        # Bounded reward history
        self._reward_history: deque = deque(maxlen=max_history)
        
        # Agent-specific statistics
        self._agent_stats: Dict[str, Dict] = {}
        
        # Timestamp tracking for look-ahead prevention
        self._last_processed_timestamp: int = 0
    
    def shape_reward(self,
                    agent_id: str,
                    slippage_bps: float,
                    market_impact_bps: float,
                    maker_rebate: float,
                    adverse_selection_cost: float,
                    timestamp_ns: int,
                    fill_quantity: float,
                    expected_alpha: float) -> RewardSignal:
        """
        Shape dense reward from TCA metrics.
        
        Args:
            agent_id: Execution agent identifier
            slippage_bps: Slippage in basis points
            market_impact_bps: Market impact in bps
            maker_rebate: Maker rebate captured (positive for maker)
            adverse_selection_cost: Cost from adverse selection
            timestamp_ns: Execution timestamp (must be <= current time)
            fill_quantity: Quantity filled
            expected_alpha: Expected alpha from the trade
            
        Returns:
            RewardSignal with shaped reward
        """
        # Prevent look-ahead bias
        if timestamp_ns < self._last_processed_timestamp:
            logger.warning(f"Out-of-order timestamp detected: {timestamp_ns}")
        
        # Base reward from alpha capture
        base_reward = expected_alpha * (fill_quantity / 100.0)
        
        # Slippage penalty (negative reward)
        slippage_penalty = abs(slippage_bps) * self.slippage_scale * 0.001
        
        # Market impact penalty
        impact_penalty = abs(market_impact_bps) * self.impact_scale * 0.001
        
        # Maker rebate bonus (positive for providing liquidity)
        rebate_bonus = maker_rebate * self.rebate_scale
        
        # Adverse selection penalty (heavy penalty for toxic flow)
        adverse_penalty = abs(adverse_selection_cost) * self.adverse_selection_scale * 0.001
        
        # Calculate total reward
        total_reward = (
            base_reward 
            - slippage_penalty 
            - impact_penalty 
            + rebate_bonus 
            - adverse_penalty
        )
        
        # Create reward signal
        signal = RewardSignal(
            agent_id=agent_id,
            timestamp_ns=timestamp_ns,
            base_reward=float(base_reward),
            slippage_penalty=float(-slippage_penalty),
            impact_penalty=float(-impact_penalty),
            rebate_bonus=float(rebate_bonus),
            adverse_selection_penalty=float(-adverse_penalty),
            total_reward=float(total_reward),
            metadata={
                "slippage_bps": slippage_bps,
                "market_impact_bps": market_impact_bps,
                "maker_rebate": maker_rebate,
                "adverse_selection_cost": adverse_selection_cost,
                "fill_quantity": fill_quantity,
                "expected_alpha": expected_alpha
            }
        )
        
        # Store in bounded history
        self._reward_history.append(signal)
        
        # Update agent statistics
        self._update_agent_stats(agent_id, signal)
        
        # Update last processed timestamp
        self._last_processed_timestamp = timestamp_ns
        
        return signal
    
    def _update_agent_stats(self, agent_id: str, signal: RewardSignal):
        """Update running statistics for an agent."""
        if agent_id not in self._agent_stats:
            self._agent_stats[agent_id] = {
                'count': 0,
                'total_reward': 0.0,
                'total_slippage_penalty': 0.0,
                'total_rebate_bonus': 0.0,
                'rewards': deque(maxlen=1000)
            }
        
        stats = self._agent_stats[agent_id]
        stats['count'] += 1
        stats['total_reward'] += signal.total_reward
        stats['total_slippage_penalty'] += signal.slippage_penalty
        stats['total_rebate_bonus'] += signal.rebate_bonus
        stats['rewards'].append(signal.total_reward)
    
    def get_agent_performance(self, agent_id: str) -> Dict[str, Any]:
        """Get performance statistics for an agent."""
        if agent_id not in self._agent_stats:
            return {}
        
        stats = self._agent_stats[agent_id]
        rewards = list(stats['rewards'])
        
        return {
            'agent_id': agent_id,
            'total_executions': stats['count'],
            'avg_reward': stats['total_reward'] / max(stats['count'], 1),
            'avg_slippage_penalty': stats['total_slippage_penalty'] / max(stats['count'], 1),
            'avg_rebate_bonus': stats['total_rebate_bonus'] / max(stats['count'], 1),
            'reward_std': float(np.std(rewards)) if len(rewards) > 1 else 0.0,
            'reward_mean': float(np.mean(rewards)) if rewards else 0.0
        }
    
    def get_recent_rewards(self, agent_id: str = None, n: int = 100) -> List[float]:
        """Get recent rewards for analysis."""
        if agent_id and agent_id in self._agent_stats:
            rewards = list(self._agent_stats[agent_id]['rewards'])
            return rewards[-n:]
        
        # All agents
        return [s.total_reward for s in list(self._reward_history)[-n:]]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get overall reward shaping summary."""
        if not self._reward_history:
            return {"status": "no_data"}
        
        all_rewards = [s.total_reward for s in self._reward_history]
        
        return {
            "total_signals": len(self._reward_history),
            "agents_tracked": len(self._agent_stats),
            "avg_total_reward": float(np.mean(all_rewards)),
            "std_total_reward": float(np.std(all_rewards)),
            "min_reward": float(np.min(all_rewards)),
            "max_reward": float(np.max(all_rewards)),
            "scales": {
                "slippage": self.slippage_scale,
                "impact": self.impact_scale,
                "rebate": self.rebate_scale,
                "adverse_selection": self.adverse_selection_scale
            }
        }
    
    def reset_agent(self, agent_id: str):
        """Reset statistics for a specific agent."""
        if agent_id in self._agent_stats:
            del self._agent_stats[agent_id]


# Example usage
def main():
    """Example usage of reward shaper."""
    shaper = TCARewardShaper()
    
    # Simulate some executions
    np.random.seed(42)
    
    for i in range(50):
        signal = shaper.shape_reward(
            agent_id="execution_agent_1",
            slippage_bps=np.random.randn() * 2.0,
            market_impact_bps=np.random.randn() * 1.0,
            maker_rebate=max(0, np.random.randn() * 0.5),
            adverse_selection_cost=abs(np.random.randn()) * 0.5,
            timestamp_ns=i * 1_000_000_000,
            fill_quantity=100.0,
            expected_alpha=np.random.randn() * 0.001
        )
        
        if i < 5:
            print(f"Step {i}: Total reward={signal.total_reward:.6f}")
            print(f"  Components: base={signal.base_reward:.6f}, "
                  f"slippage={signal.slippage_penalty:.6f}, "
                  f"rebate={signal.rebate_bonus:.6f}")
    
    print(f"\nAgent performance: {shaper.get_agent_performance('execution_agent_1')}")
    print(f"\nSummary: {shaper.get_summary()}")


if __name__ == "__main__":
    main()
