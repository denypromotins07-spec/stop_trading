"""
Chapter 3: Meta-Learning & Few-Shot Adaptation (MAML)
File: python/meta_learning/task_distribution.py

Task sampling engine that feeds diverse historical market crashes, pumps,
and sideways chops into the MAML loop. Ensures meta-learned models are robust
to extreme distribution shifts and black swan events.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime types for task distribution."""
    CRASH = "crash"           # Rapid downward movement
    PUMP = "pump"             # Rapid upward movement
    SIDEWAYS = "sideways"     # Low volatility range-bound
    RECOVERY = "recovery"     # Post-crash recovery
    BULL_RUN = "bull_run"     # Sustained uptrend
    BEAR_MARKET = "bear_market"  # Sustained downtrend
    FLASH_CRASH = "flash_crash"   # Extreme short-term drop
    LIQUIDITY_CRISIS = "liquidity_crisis"  # Wide spreads, low volume


@dataclass
class TaskDefinition:
    """Defines a single few-shot learning task."""
    task_id: str
    regime: MarketRegime
    symbol: str
    start_time: datetime
    end_time: datetime
    
    # Few-shot data
    support_x: np.ndarray = None  # Input features
    support_y: np.ndarray = None  # Labels
    query_x: np.ndarray = None
    query_y: np.ndarray = None
    
    # Metadata
    difficulty: float = 0.5  # 0=easy, 1=hard
    sample_weight: float = 1.0


@dataclass
class TaskDistributionConfig:
    """Configuration for task distribution sampling."""
    # Regime balance
    regime_weights: Dict[MarketRegime, float] = field(default_factory=lambda: {
        MarketRegime.CRASH: 0.15,
        MarketRegime.PUMP: 0.15,
        MarketRegime.SIDEWAYS: 0.20,
        MarketRegime.RECOVERY: 0.10,
        MarketRegime.BULL_RUN: 0.10,
        MarketRegime.BEAR_MARKET: 0.10,
        MarketRegime.FLASH_CRASH: 0.10,
        MarketRegime.LIQUIDITY_CRISIS: 0.10
    })
    
    # Task parameters
    n_support_samples: int = 10  # Few-shot support set size
    n_query_samples: int = 5     # Query set size
    feature_dim: int = 64
    
    # Difficulty curriculum
    use_curriculum: bool = True
    initial_difficulty: float = 0.3
    max_difficulty: float = 1.0
    difficulty_increase_rate: float = 0.01
    
    # Diversity constraints
    min_regime_diversity: int = 3  # Min different regimes per batch
    max_symbol_concentration: float = 0.5  # Max fraction from one symbol


class TaskSampler:
    """
    Samples diverse tasks from historical market data.
    Implements curriculum learning and regime balancing.
    """
    
    def __init__(self, config: Optional[TaskDistributionConfig] = None):
        self.config = config or TaskDistributionConfig()
        
        # Historical data registry (would be populated from database)
        self.historical_episodes: Dict[str, List[Dict]] = {}
        
        # Curriculum state
        self.current_difficulty = self.config.initial_difficulty
        self.total_tasks_sampled = 0
        
        # Sampling statistics
        self.regime_counts: Dict[MarketRegime, int] = {r: 0 for r in MarketRegime}
        self.symbol_counts: Dict[str, int] = {}
    
    def register_historical_episode(
        self,
        episode_id: str,
        regime: MarketRegime,
        symbol: str,
        features: np.ndarray,
        labels: np.ndarray,
        timestamps: List[datetime]
    ):
        """Register a historical market episode for task sampling."""
        if episode_id not in self.historical_episodes:
            self.historical_episodes[episode_id] = []
        
        self.historical_episodes[episode_id].append({
            "regime": regime,
            "symbol": symbol,
            "features": features,
            "labels": labels,
            "timestamps": timestamps,
            "n_samples": len(features)
        })
        
        logger.debug(
            f"Registered episode {episode_id}: {regime.value} "
            f"({len(features)} samples)"
        )
    
    def _select_regime(self) -> MarketRegime:
        """Select regime based on configured weights."""
        regimes = list(self.config.regime_weights.keys())
        weights = list(self.config.regime_weights.values())
        return np.random.choice(regimes, p=weights)
    
    def _find_episodes_for_regime(
        self, 
        regime: MarketRegime
    ) -> List[Tuple[str, Dict]]:
        """Find all episodes matching a regime."""
        matches = []
        for ep_id, episodes in self.historical_episodes.items():
            for ep in episodes:
                if ep["regime"] == regime:
                    matches.append((ep_id, ep))
        return matches
    
    def _extract_few_shot_task(
        self,
        episode: Dict,
        n_support: int,
        n_query: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extract few-shot task from episode data."""
        features = episode["features"]
        labels = episode["labels"]
        n_total = len(features)
        
        if n_total < n_support + n_query:
            # Not enough samples, use what we have
            n_support = max(1, n_total // 2)
            n_query = n_total - n_support
        
        # Random split
        indices = np.random.permutation(n_total)
        support_idx = indices[:n_support]
        query_idx = indices[n_support:n_support + n_query]
        
        return (
            features[support_idx],
            labels[support_idx],
            features[query_idx],
            labels[query_idx]
        )
    
    def sample_task(
        self,
        force_regime: Optional[MarketRegime] = None
    ) -> Optional[TaskDefinition]:
        """
        Sample a single few-shot task.
        
        Args:
            force_regime: Optionally force specific regime
        
        Returns:
            TaskDefinition or None if no data available
        """
        # Select regime
        regime = force_regime or self._select_regime()
        
        # Find matching episodes
        episodes = self._find_episodes_for_regime(regime)
        if not episodes:
            logger.warning(f"No episodes found for regime: {regime.value}")
            return None
        
        # Select random episode
        ep_id, episode = episodes[np.random.randint(len(episodes))]
        
        # Extract few-shot data
        try:
            support_x, support_y, query_x, query_y = self._extract_few_shot_task(
                episode,
                self.config.n_support_samples,
                self.config.n_query_samples
            )
        except Exception as e:
            logger.error(f"Failed to extract task: {e}")
            return None
        
        # Calculate difficulty based on regime
        difficulty_map = {
            MarketRegime.SIDEWAYS: 0.2,
            MarketRegime.BULL_RUN: 0.3,
            MarketRegime.BEAR_MARKET: 0.4,
            MarketRegime.RECOVERY: 0.5,
            MarketRegime.PUMP: 0.6,
            MarketRegime.CRASH: 0.7,
            MarketRegime.LIQUIDITY_CRISIS: 0.8,
            MarketRegime.FLASH_CRASH: 0.9
        }
        base_difficulty = difficulty_map.get(regime, 0.5)
        
        # Apply curriculum scaling
        if self.config.use_curriculum:
            difficulty = min(
                base_difficulty * (0.5 + self.current_difficulty),
                self.config.max_difficulty
            )
        else:
            difficulty = base_difficulty
        
        # Create task
        task = TaskDefinition(
            task_id=f"{regime.value}_{ep_id}_{self.total_tasks_sampled}",
            regime=regime,
            symbol=episode["symbol"],
            start_time=episode["timestamps"][0] if len(episode["timestamps"]) > 0 else datetime.utcnow(),
            end_time=episode["timestamps"][-1] if len(episode["timestamps"]) > 0 else datetime.utcnow(),
            support_x=support_x,
            support_y=support_y,
            query_x=query_x,
            query_y=query_y,
            difficulty=difficulty,
            sample_weight=1.0 / (1.0 + difficulty)  # Harder tasks get lower weight initially
        )
        
        # Update statistics
        self.total_tasks_sampled += 1
        self.regime_counts[regime] += 1
        symbol = episode["symbol"]
        self.symbol_counts[symbol] = self.symbol_counts.get(symbol, 0) + 1
        
        # Update curriculum
        if self.config.use_curriculum:
            self.current_difficulty = min(
                self.current_difficulty + self.config.difficulty_increase_rate,
                self.config.max_difficulty
            )
        
        return task
    
    def sample_task_batch(
        self,
        batch_size: int = 4
    ) -> List[TaskDefinition]:
        """
        Sample a batch of diverse tasks.
        Ensures regime diversity and symbol balance.
        """
        tasks = []
        regimes_used = set()
        symbols_used: Dict[str, int] = {}
        
        attempts = 0
        max_attempts = batch_size * 3
        
        while len(tasks) < batch_size and attempts < max_attempts:
            attempts += 1
            
            # Force regime diversity early in batch
            if len(regimes_used) < self.config.min_regime_diversity:
                available_regimes = [
                    r for r in MarketRegime 
                    if r not in regimes_used and 
                    self.config.regime_weights.get(r, 0) > 0
                ]
                if available_regimes:
                    force_regime = np.random.choice(available_regimes)
                else:
                    force_regime = None
            else:
                force_regime = None
            
            # Sample task
            task = self.sample_task(force_regime=force_regime)
            if task is None:
                continue
            
            # Check symbol concentration
            symbol_count = symbols_used.get(task.symbol, 0)
            max_allowed = int(batch_size * self.config.max_symbol_concentration)
            if symbol_count >= max_allowed:
                continue  # Skip to maintain symbol diversity
            
            # Accept task
            tasks.append(task)
            regimes_used.add(task.regime)
            symbols_used[task.symbol] = symbol_count + 1
        
        if len(tasks) < batch_size:
            logger.warning(
                f"Could only sample {len(tasks)}/{batch_size} diverse tasks"
            )
        
        return tasks
    
    def prepare_maml_batch(
        self,
        tasks: List[TaskDefinition]
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Prepare tasks for MAML training.
        Returns list of (support_x, support_y) tuples.
        """
        batch = []
        for task in tasks:
            # Combine support and query for training (simplified)
            x = np.concatenate([task.support_x, task.query_x], axis=0)
            y = np.concatenate([task.support_y, task.query_y], axis=0)
            
            # Apply sample weighting via duplication (simple approach)
            if task.sample_weight > 0.8:
                batch.append((x, y))
            elif task.sample_weight > 0.5 and len(x) > 1:
                # Include with probability
                if np.random.random() < task.sample_weight:
                    batch.append((x, y))
            # Lower weight tasks may be skipped
        
        return batch
    
    def get_sampling_statistics(self) -> Dict[str, Any]:
        """Get current sampling statistics."""
        total = sum(self.regime_counts.values())
        
        return {
            "total_tasks": self.total_tasks_sampled,
            "regime_distribution": {
                r.value: count / total if total > 0 else 0
                for r, count in self.regime_counts.items()
            },
            "symbol_distribution": dict(self.symbol_counts),
            "current_difficulty": self.current_difficulty,
            "n_registered_episodes": len(self.historical_episodes)
        }
    
    def reset_curriculum(self):
        """Reset curriculum learning state."""
        self.current_difficulty = self.config.initial_difficulty
        logger.info("Curriculum reset")


# Export for module use
__all__ = [
    "MarketRegime",
    "TaskDefinition",
    "TaskDistributionConfig",
    "TaskSampler"
]
