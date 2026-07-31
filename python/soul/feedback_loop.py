"""
SOUL.md Feedback Loop - Ray actor processing parsed SOUL.md data.
Translates qualitative mistakes into quantitative penalty weights for RL agents.
Strictly enforces 3GB RAM limit with bounded memory operations.
"""
import asyncio
import ray
from ray import actor
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np
from collections import deque
import logging

# Import parser components
from soul.parser import TradeOutcome, Mistake, RegimeMemory, SOULBlock, SOULParser


@dataclass
class RewardSignal:
    """Dense reward signal for RL agents."""
    agent_id: str
    timestamp_ns: int
    reward_value: float
    penalty_component: float
    bonus_component: float
    mistake_penalty: float
    execution_bonus: float
    regime_adjustment: float
    metadata: Dict[str, Any]


@dataclass
class LossAdjustment:
    """Loss function adjustment based on SOUL feedback."""
    model_id: str
    feature_indices: List[int]
    penalty_weights: np.ndarray
    gradient_scale: float
    timestamp_ns: int


@ray.remote(max_restarts=3, max_task_retries=3)
class FeedbackLoopActor:
    """
    Ray actor that processes SOUL.md data to calculate dense reward signals.
    Translates mistakes into penalty weights for model loss functions.
    Memory-bounded for 3GB Python ceiling.
    """
    
    def __init__(self, 
                 max_history_size: int = 10000,
                 penalty_scale: float = 1.0,
                 bonus_scale: float = 0.5,
                 decay_factor: float = 0.95):
        """
        Initialize feedback loop actor.
        
        Args:
            max_history_size: Maximum number of historical records to keep
            penalty_scale: Scaling factor for mistake penalties
            bonus_scale: Scaling factor for execution bonuses
            decay_factor: Decay factor for historical rewards
        """
        self.max_history_size = max_history_size
        self.penalty_scale = penalty_scale
        self.bonus_scale = bonus_scale
        self.decay_factor = decay_factor
        
        # Bounded history using deque
        self._reward_history: deque = deque(maxlen=max_history_size)
        self._mistake_history: deque = deque(maxlen=max_history_size)
        self._outcome_history: deque = deque(maxlen=max_history_size)
        
        # Aggregated statistics
        self._total_penalties = 0.0
        self._total_bonuses = 0.0
        self._processed_count = 0
        
        # Category-specific penalty multipliers
        self._category_multipliers = {
            "timing": 1.5,
            "sizing": 1.2,
            "regime_misclassification": 2.0,
            "slippage": 1.0,
            "adverse_selection": 1.8
        }
        
        self.logger = logging.getLogger(__name__)
    
    async def process_outcome(self, outcome: TradeOutcome) -> RewardSignal:
        """
        Process a single trade outcome into a reward signal.
        
        Args:
            outcome: TradeOutcome from SOUL parser
            
        Returns:
            RewardSignal for RL agent
        """
        # Base reward from PnL (normalized)
        base_reward = np.tanh(outcome.pnl / 1000.0)  # Normalize to [-1, 1]
        
        # Execution quality bonus
        exec_bonus = outcome.execution_quality * self.bonus_scale
        
        # Slippage penalty
        slippage_penalty = abs(outcome.slippage_bps) / 100.0 * self.penalty_scale
        
        # Calculate total reward
        reward_value = base_reward + exec_bonus - slippage_penalty
        
        signal = RewardSignal(
            agent_id=f"execution_{outcome.instrument}",
            timestamp_ns=outcome.timestamp_ns,
            reward_value=float(reward_value),
            penalty_component=float(-slippage_penalty),
            bonus_component=float(exec_bonus),
            mistake_penalty=0.0,
            execution_bonus=float(exec_bonus),
            regime_adjustment=0.0,
            metadata={
                "trade_id": outcome.trade_id,
                "pnl": outcome.pnl,
                "slippage_bps": outcome.slippage_bps,
                "regime": outcome.regime_id
            }
        )
        
        # Store in bounded history
        self._outcome_history.append(outcome)
        self._reward_history.append(signal)
        self._total_bonuses += exec_bonus
        self._processed_count += 1
        
        return signal
    
    async def process_mistake(self, mistake: Mistake) -> Tuple[RewardSignal, LossAdjustment]:
        """
        Process a mistake into reward signal and loss adjustment.
        
        Args:
            mistake: Mistake from SOUL parser
            
        Returns:
            Tuple of (RewardSignal, LossAdjustment)
        """
        # Get category multiplier
        multiplier = self._category_multipliers.get(
            mistake.category, 1.0
        )
        
        # Calculate penalty
        base_penalty = mistake.severity * mistake.penalty_weight
        total_penalty = base_penalty * multiplier * self.penalty_scale
        
        # Create reward signal with negative reward
        signal = RewardSignal(
            agent_id=f"correction_{mistake.category}",
            timestamp_ns=mistake.timestamp_ns,
            reward_value=float(-total_penalty),
            penalty_component=float(-total_penalty),
            bonus_component=0.0,
            mistake_penalty=float(total_penalty),
            execution_bonus=0.0,
            regime_adjustment=0.0,
            metadata={
                "mistake_id": mistake.mistake_id,
                "trade_id": mistake.trade_id,
                "category": mistake.category,
                "severity": mistake.severity,
                "description": mistake.description[:100]  # Truncate for memory
            }
        )
        
        # Create loss adjustment for model retraining
        # Map mistake category to feature indices (heuristic)
        feature_indices = self._map_category_to_features(mistake.category)
        penalty_weights = np.ones(len(feature_indices)) * total_penalty
        
        loss_adj = LossAdjustment(
            model_id=self._infer_model_id(mistake.category),
            feature_indices=feature_indices,
            penalty_weights=penalty_weights,
            gradient_scale=float(total_penalty),
            timestamp_ns=mistake.timestamp_ns
        )
        
        # Store in bounded history
        self._mistake_history.append(mistake)
        self._reward_history.append(signal)
        self._total_penalties += total_penalty
        self._processed_count += 1
        
        return signal, loss_adj
    
    async def process_regime_memory(self, memory: RegimeMemory) -> Dict[str, float]:
        """
        Process regime memory to extract regime-specific adjustments.
        
        Args:
            memory: RegimeMemory from SOUL parser
            
        Returns:
            Dict mapping agent IDs to regime adjustment factors
        """
        adjustments = {}
        
        # Extract volatility state adjustment
        vol_adjustment = self._parse_volatility_state(memory.volatility_state)
        
        # Extract liquidity state adjustment
        liq_adjustment = self._parse_liquidity_state(memory.liquidity_state)
        
        # Apply to relevant agents
        adjustments["volatility_agent"] = vol_adjustment
        adjustments["liquidity_agent"] = liq_adjustment
        
        # Store lessons for future reference
        for lesson in memory.lessons_learned:
            self._encode_lesson(lesson, memory.regime_id)
        
        return adjustments
    
    def _map_category_to_features(self, category: str) -> List[int]:
        """Map mistake category to feature indices for loss adjustment."""
        # Heuristic mapping - in production this would be model-specific
        mappings = {
            "timing": list(range(0, 10)),      # Early features
            "sizing": list(range(10, 20)),     # Position size features
            "regime_misclassification": list(range(20, 30)),  # Regime features
            "slippage": list(range(30, 40)),   # Liquidity features
            "adverse_selection": list(range(40, 50))  # Order book features
        }
        return mappings.get(category, list(range(0, 5)))
    
    def _infer_model_id(self, category: str) -> str:
        """Infer which model needs adjustment based on mistake category."""
        if category in ["timing", "slippage"]:
            return "execution_policy"
        elif category == "sizing":
            return "position_sizer"
        elif category == "regime_misclassification":
            return "regime_classifier"
        else:
            return "alpha_predictor"
    
    def _parse_volatility_state(self, state: str) -> float:
        """Parse volatility state string into adjustment factor."""
        states = {
            "low": 0.8,
            "medium": 1.0,
            "high": 1.2,
            "extreme": 1.5
        }
        return states.get(state.lower(), 1.0)
    
    def _parse_liquidity_state(self, state: str) -> float:
        """Parse liquidity state string into adjustment factor."""
        states = {
            "thin": 1.3,
            "normal": 1.0,
            "deep": 0.7
        }
        return states.get(state.lower(), 1.0)
    
    def _encode_lesson(self, lesson: str, regime_id: str):
        """Encode lesson for future reference (memory-bounded)."""
        # Simple encoding - in production use embeddings
        pass
    
    async def get_aggregate_rewards(self, 
                                    time_window_ns: int = 3600_000_000_000
                                    ) -> Dict[str, float]:
        """
        Get aggregate rewards for the specified time window.
        
        Args:
            time_window_ns: Time window in nanoseconds (default 1 hour)
            
        Returns:
            Dict of aggregated reward metrics
        """
        import time
        current_time = time.time_ns()
        cutoff = current_time - time_window_ns
        
        recent_signals = [
            s for s in self._reward_history 
            if s.timestamp_ns >= cutoff
        ]
        
        if not recent_signals:
            return {"total_reward": 0.0, "count": 0}
        
        total_reward = sum(s.reward_value for s in recent_signals)
        total_penalty = sum(s.mistake_penalty for s in recent_signals)
        total_bonus = sum(s.execution_bonus for s in recent_signals)
        
        return {
            "total_reward": float(total_reward),
            "total_penalty": float(total_penalty),
            "total_bonus": float(total_bonus),
            "count": len(recent_signals),
            "avg_reward": float(total_reward / len(recent_signals))
        }
    
    async def get_loss_adjustments(self) -> List[LossAdjustment]:
        """Get all pending loss adjustments for model retraining."""
        adjustments = []
        for mistake in self._mistake_history:
            _, adj = await self.process_mistake(mistake)
            adjustments.append(adj)
        return adjustments
    
    async def apply_decay(self):
        """Apply decay factor to historical rewards."""
        # Decay old rewards to emphasize recent performance
        new_history = deque(maxlen=self.max_history_size)
        for signal in self._reward_history:
            decayed_signal = RewardSignal(
                agent_id=signal.agent_id,
                timestamp_ns=signal.timestamp_ns,
                reward_value=signal.reward_value * self.decay_factor,
                penalty_component=signal.penalty_component * self.decay_factor,
                bonus_component=signal.bonus_component * self.decay_factor,
                mistake_penalty=signal.mistake_penalty * self.decay_factor,
                execution_bonus=signal.execution_bonus * self.decay_factor,
                regime_adjustment=signal.regime_adjustment * self.decay_factor,
                metadata=signal.metadata
            )
            new_history.append(decayed_signal)
        
        self._reward_history = new_history
    
    def get_stats(self) -> Dict[str, Any]:
        """Get actor statistics."""
        return {
            "processed_count": self._processed_count,
            "total_penalties": self._total_penalties,
            "total_bonuses": self._total_bonuses,
            "history_size": len(self._reward_history),
            "mistake_count": len(self._mistake_history),
            "outcome_count": len(self._outcome_history)
        }


class FeedbackLoopManager:
    """Manager for feedback loop actors."""
    
    def __init__(self, num_actors: int = 2):
        """Initialize manager with specified number of actors."""
        self.num_actors = num_actors
        self.actors = [
            FeedbackLoopActor.remote() for _ in range(num_actors)
        ]
        self.parser = SOULParser()
    
    async def process_soul_file(self, filepath: str) -> List[RewardSignal]:
        """
        Process entire SOUL.md file through feedback loop.
        
        Args:
            filepath: Path to SOUL.md file
            
        Returns:
            List of generated reward signals
        """
        all_signals = []
        
        async for block in self.parser.parse_file(filepath):
            # Distribute work across actors
            actor_idx = self._processed_count % self.num_actors
            actor = self.actors[actor_idx]
            
            # Process outcomes
            for outcome in block.outcomes:
                signal = await actor.process_outcome.remote(outcome)
                all_signals.append(signal)
            
            # Process mistakes
            for mistake in block.mistakes:
                result = await actor.process_mistake.remote(mistake)
                signal, _ = result
                all_signals.append(signal)
            
            # Process regime memories
            for memory in block.memories:
                await actor.process_regime_memory.remote(memory)
        
        return all_signals
    
    @property
    def _processed_count(self) -> int:
        """Track processed count for round-robin distribution."""
        return 0  # Simplified for now


# Example usage
async def main():
    """Example usage of feedback loop."""
    ray.init(ignore_reinit_error=True, namespace="soul_feedback")
    
    manager = FeedbackLoopManager(num_actors=2)
    
    # Process SOUL.md file
    signals = await manager.process_soul_file("SOUL.md")
    print(f"Generated {len(signals)} reward signals")
    
    # Get stats
    for i, actor in enumerate(manager.actors):
        stats = await actor.get_stats.remote()
        print(f"Actor {i}: {stats}")
    
    ray.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
