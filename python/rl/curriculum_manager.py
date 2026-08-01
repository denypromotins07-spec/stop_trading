"""
Curriculum Learning Scheduler for PPO execution agents.
Starts training in low-volatility, high-liquidity regimes.
Gradually introduces toxic order books and flash crashes as policy matures.
Memory-efficient design respecting 3GB RAM constraint.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Tuple
from enum import Enum
from datetime import datetime
import numpy as np
from collections import deque

logger = logging.getLogger(__name__)


class CurriculumStage(Enum):
    """Training curriculum stages."""
    WARMUP = 0  # Low volatility, simple environments
    BASIC = 1   # Normal market conditions
    INTERMEDIATE = 2  # Moderate volatility
    ADVANCED = 3  # High volatility, thin liquidity
    EXPERT = 4  # Flash crashes, toxic flows
    ADAPTIVE = 5  # Dynamic difficulty adjustment


@dataclass
class TrainingRegime:
    """Defines parameters for a training regime."""
    stage: CurriculumStage
    volatility_target: float  # Target annualized volatility
    liquidity_threshold: float  # Minimum spread in bps
    toxicity_level: float  # 0-1, probability of adverse selection
    max_slippage_bps: float  # Maximum acceptable slippage
    episode_length_steps: int
    reward_shaping_strength: float  # 0-1, how much to shape rewards
    
    # Progression criteria
    min_success_rate: float  # Must achieve this rate to advance
    min_episodes: int  # Minimum episodes before considering advancement
    target_sharpe_ratio: float  # Minimum risk-adjusted return


@dataclass
class PerformanceMetrics:
    """Tracks agent performance for curriculum progression."""
    stage: CurriculumStage
    total_episodes: int = 0
    successful_episodes: int = 0
    avg_reward: float = 0.0
    avg_sharpe: float = 0.0
    avg_slippage_bps: float = 0.0
    recent_rewards: deque = field(default_factory=lambda: deque(maxlen=100))
    recent_sharpes: deque = field(default_factory=lambda: deque(maxlen=100))
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_episodes == 0:
            return 0.0
        return self.successful_episodes / self.total_episodes
    
    @property
    def recent_avg_reward(self) -> float:
        """Average of recent rewards."""
        if not self.recent_rewards:
            return 0.0
        return np.mean(self.recent_rewards)
    
    @property
    def recent_avg_sharpe(self) -> float:
        """Average of recent Sharpe ratios."""
        if not self.recent_sharpes:
            return 0.0
        return np.mean(self.recent_sharpes)


class CurriculumManager:
    """
    Manages curriculum learning progression for RL agents.
    
    Features:
    - Progressive difficulty increase
    - Performance-based advancement
    - Regressive steps on poor performance
    - Adaptive difficulty adjustment
    - Memory-bounded metrics tracking
    """
    
    DEFAULT_REGIMES = {
        CurriculumStage.WARMUP: TrainingRegime(
            stage=CurriculumStage.WARMUP,
            volatility_target=0.2,
            liquidity_threshold=5.0,
            toxicity_level=0.0,
            max_slippage_bps=10.0,
            episode_length_steps=100,
            reward_shaping_strength=0.8,
            min_success_rate=0.9,
            min_episodes=50,
            target_sharpe_ratio=1.0,
        ),
        CurriculumStage.BASIC: TrainingRegime(
            stage=CurriculumStage.BASIC,
            volatility_target=0.4,
            liquidity_threshold=3.0,
            toxicity_level=0.1,
            max_slippage_bps=15.0,
            episode_length_steps=200,
            reward_shaping_strength=0.6,
            min_success_rate=0.8,
            min_episodes=100,
            target_sharpe_ratio=1.2,
        ),
        CurriculumStage.INTERMEDIATE: TrainingRegime(
            stage=CurriculumStage.INTERMEDIATE,
            volatility_target=0.6,
            liquidity_threshold=2.0,
            toxicity_level=0.2,
            max_slippage_bps=20.0,
            episode_length_steps=300,
            reward_shaping_strength=0.4,
            min_success_rate=0.75,
            min_episodes=150,
            target_sharpe_ratio=1.5,
        ),
        CurriculumStage.ADVANCED: TrainingRegime(
            stage=CurriculumStage.ADVANCED,
            volatility_target=0.8,
            liquidity_threshold=1.5,
            toxicity_level=0.3,
            max_slippage_bps=25.0,
            episode_length_steps=400,
            reward_shaping_strength=0.2,
            min_success_rate=0.7,
            min_episodes=200,
            target_sharpe_ratio=1.8,
        ),
        CurriculumStage.EXPERT: TrainingRegime(
            stage=CurriculumStage.EXPERT,
            volatility_target=1.2,
            liquidity_threshold=1.0,
            toxicity_level=0.5,
            max_slippage_bps=30.0,
            episode_length_steps=500,
            reward_shaping_strength=0.1,
            min_success_rate=0.65,
            min_episodes=300,
            target_sharpe_ratio=2.0,
        ),
        CurriculumStage.ADAPTIVE: TrainingRegime(
            stage=CurriculumStage.ADAPTIVE,
            volatility_target=1.0,  # Dynamically adjusted
            liquidity_threshold=1.0,
            toxicity_level=0.4,
            max_slippage_bps=25.0,
            episode_length_steps=500,
            reward_shaping_strength=0.15,
            min_success_rate=0.7,
            min_episodes=100,
            target_sharpe_ratio=2.0,
        ),
    }
    
    def __init__(self, 
                 custom_regimes: Optional[Dict[CurriculumStage, TrainingRegime]] = None,
                 start_stage: CurriculumStage = CurriculumStage.WARMUP,
                 allow_regression: bool = True,
                 adaptive_window: int = 50):
        """
        Initialize curriculum manager.
        
        Args:
            custom_regimes: Optional custom training regimes
            start_stage: Starting curriculum stage
            allow_regression: Allow demotion on poor performance
            adaptive_window: Window size for adaptive difficulty
        """
        self._regimes = custom_regimes or self.DEFAULT_REGIMES.copy()
        self._current_stage = start_stage
        self._allow_regression = allow_regression
        self._adaptive_window = adaptive_window
        
        # Performance tracking per stage
        self._metrics: Dict[CurriculumStage, PerformanceMetrics] = {
            stage: PerformanceMetrics(stage=stage)
            for stage in CurriculumStage
        }
        
        # Callbacks for stage changes
        self._stage_change_callbacks: List[Callable[[CurriculumStage, CurriculumStage], Any]] = []
        
        # Statistics
        self._stats = {
            'total_episodes': 0,
            'stage_advancements': 0,
            'stage_regressions': 0,
            'current_stage': start_stage.value,
        }
        
        logger.info(f"CurriculumManager initialized at stage {start_stage.name}")
    
    def get_current_regime(self) -> TrainingRegime:
        """Get current training regime."""
        return self._regimes[self._current_stage]
    
    def record_episode(self, 
                       reward: float,
                       sharpe_ratio: float,
                       slippage_bps: float,
                       success: bool,
                       episode_length: Optional[int] = None):
        """
        Record episode results for curriculum progression.
        
        Args:
            reward: Episode reward
            sharpe_ratio: Risk-adjusted return
            slippage_bps: Execution slippage in basis points
            success: Whether episode met objectives
            episode_length: Optional actual episode length
        """
        metrics = self._metrics[self._current_stage]
        
        metrics.total_episodes += 1
        self._stats['total_episodes'] += 1
        
        if success:
            metrics.successful_episodes += 1
        
        metrics.avg_reward = (
            (metrics.avg_reward * (metrics.total_episodes - 1) + reward) 
            / metrics.total_episodes
        )
        metrics.avg_sharpe = (
            (metrics.avg_sharpe * (metrics.total_episodes - 1) + sharpe_ratio)
            / metrics.total_episodes
        )
        metrics.avg_slippage_bps = (
            (metrics.avg_slippage_bps * (metrics.total_episodes - 1) + slippage_bps)
            / metrics.total_episodes
        )
        
        # Track recent performance
        metrics.recent_rewards.append(reward)
        metrics.recent_sharpes.append(sharpe_ratio)
        
        # Check for progression after minimum episodes
        if metrics.total_episodes >= self._regimes[self._current_stage].min_episodes:
            self._check_progression()
    
    def _check_progression(self):
        """Check if agent should advance or regress."""
        metrics = self._metrics[self._current_stage]
        regime = self._regimes[self._current_stage]
        
        # Check for advancement
        if self._should_advance(metrics, regime):
            self._advance_stage()
        
        # Check for regression
        elif self._allow_regression and self._should_regress(metrics, regime):
            self._regress_stage()
    
    def _should_advance(self, metrics: PerformanceMetrics,
                        regime: TrainingRegime) -> bool:
        """Determine if agent should advance to next stage."""
        if self._current_stage == CurriculumStage.EXPERT:
            return False  # Already at max
        
        # Check success rate
        if metrics.success_rate < regime.min_success_rate:
            return False
        
        # Check Sharpe ratio
        if metrics.recent_avg_sharpe < regime.target_sharpe_ratio:
            return False
        
        # Check slippage
        if metrics.avg_slippage_bps > regime.max_slippage_bps * 0.8:
            return False  # Keep practicing if slippage too high
        
        return True
    
    def _should_regress(self, metrics: PerformanceMetrics,
                        regime: TrainingRegime) -> bool:
        """Determine if agent should regress to previous stage."""
        if self._current_stage == CurriculumStage.WARMUP:
            return False  # Already at minimum
        
        # Regress if performing poorly
        if metrics.success_rate < regime.min_success_rate * 0.5:
            return True
        
        # Regress if Sharpe is very poor
        if metrics.recent_avg_sharpe < regime.target_sharpe_ratio * 0.3:
            return True
        
        return False
    
    def _advance_stage(self):
        """Advance to next curriculum stage."""
        old_stage = self._current_stage
        stage_index = list(CurriculumStage).index(self._current_stage)
        
        if stage_index < len(CurriculumStage) - 1:
            self._current_stage = list(CurriculumStage)[stage_index + 1]
            self._stats['stage_advancements'] += 1
            self._stats['current_stage'] = self._current_stage.value
            
            logger.info(
                f"Curriculum advanced: {old_stage.name} -> {self._current_stage.name}"
            )
            
            # Notify callbacks
            for callback in self._stage_change_callbacks:
                try:
                    callback(old_stage, self._current_stage)
                except Exception as e:
                    logger.error(f"Stage change callback error: {e}")
    
    def _regress_stage(self):
        """Regress to previous curriculum stage."""
        old_stage = self._current_stage
        stage_index = list(CurriculumStage).index(self._current_stage)
        
        if stage_index > 0:
            self._current_stage = list(CurriculumStage)[stage_index - 1]
            self._stats['stage_regressions'] += 1
            self._stats['current_stage'] = self._current_stage.value
            
            logger.warning(
                f"Curriculum regressed: {old_stage.name} -> {self._current_stage.name}"
            )
            
            # Notify callbacks
            for callback in self._stage_change_callbacks:
                try:
                    callback(old_stage, self._current_stage)
                except Exception as e:
                    logger.error(f"Stage change callback error: {e}")
    
    def register_stage_callback(self, 
                                callback: Callable[[CurriculumStage, CurriculumStage], Any]):
        """Register callback for stage changes."""
        self._stage_change_callbacks.append(callback)
    
    def get_environment_config(self) -> Dict:
        """
        Get environment configuration for current regime.
        
        Returns:
            Dict with environment parameters
        """
        regime = self._regimes[self._current_stage]
        
        return {
            'volatility_target': regime.volatility_target,
            'liquidity_threshold': regime.liquidity_threshold,
            'toxicity_level': regime.toxicity_level,
            'max_slippage_bps': regime.max_slippage_bps,
            'episode_length_steps': regime.episode_length_steps,
            'reward_shaping_strength': regime.reward_shaping_strength,
            'curriculum_stage': self._current_stage.value,
            'curriculum_stage_name': self._current_stage.name,
        }
    
    def generate_market_scenario(self, rng: np.random.Generator) -> Dict:
        """
        Generate market scenario based on current regime.
        
        Args:
            rng: NumPy random generator
            
        Returns:
            Market scenario parameters
        """
        regime = self._regimes[self._current_stage]
        
        # Base volatility with regime scaling
        base_vol = 0.0001  # Per-step base volatility
        volatility = base_vol * regime.volatility_target * rng.lognormal(0, 0.2)
        
        # Liquidity (spread) inversely related to liquidity threshold
        base_spread = 1.0  # bps
        spread = base_spread / regime.liquidity_threshold * rng.lognormal(0, 0.3)
        
        # Toxicity probability
        is_toxic = rng.random() < regime.toxicity_level
        
        # Flash crash simulation (expert stage only)
        flash_crash = False
        if self._current_stage == CurriculumStage.EXPERT:
            flash_crash = rng.random() < 0.02  # 2% chance
        
        return {
            'volatility_per_step': volatility,
            'initial_spread_bps': spread,
            'is_toxic_flow': is_toxic,
            'flash_crash': flash_crash,
            'adverse_selection_prob': regime.toxicity_level if is_toxic else 0.0,
        }
    
    def get_curriculum_state(self) -> Dict:
        """Get full curriculum state for checkpointing."""
        return {
            'current_stage': self._current_stage.value,
            'current_stage_name': self._current_stage.name,
            'metrics': {
                stage.name: {
                    'total_episodes': m.total_episodes,
                    'success_rate': m.success_rate,
                    'avg_reward': m.avg_reward,
                    'avg_sharpe': m.avg_sharpe,
                    'avg_slippage_bps': m.avg_slippage_bps,
                }
                for stage, m in self._metrics.items()
            },
            'stats': self._stats.copy(),
        }
    
    def load_curriculum_state(self, state: Dict):
        """Load curriculum state from checkpoint."""
        stage_value = state.get('current_stage', 0)
        self._current_stage = list(CurriculumStage)[stage_value]
        
        # Restore metrics
        for stage_name, metrics_data in state.get('metrics', {}).items():
            stage = CurriculumStage[stage_name]
            metrics = self._metrics[stage]
            metrics.total_episodes = metrics_data.get('total_episodes', 0)
            metrics.successful_episodes = int(
                metrics.total_episodes * metrics_data.get('success_rate', 0)
            )
            metrics.avg_reward = metrics_data.get('avg_reward', 0.0)
            metrics.avg_sharpe = metrics_data.get('avg_sharpe', 0.0)
            metrics.avg_slippage_bps = metrics_data.get('avg_slippage_bps', 0.0)
    
    def get_stats(self) -> Dict:
        """Get curriculum statistics."""
        return self._stats.copy()
    
    def force_stage(self, stage: CurriculumStage):
        """Force curriculum to specific stage (for debugging/testing)."""
        old_stage = self._current_stage
        self._current_stage = stage
        self._stats['current_stage'] = stage.value
        
        logger.info(f"Curriculum forced: {old_stage.name} -> {stage.name}")
