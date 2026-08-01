"""
Reward Module Root
Combines execution, risk, and soul penalties into a unified, normalized dense 
reward signal for Ray RLlib.

Provides single interface for all reward computation in the RL pipeline.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import threading
import time

from .profit_factor_shaper import ProfitFactorShaper, RewardConfig, AdaptiveRewardScaler
from .soul_penalty import SOULPenaltyEngine, SOULConfig


@dataclass
class UnifiedRewardConfig:
    """Unified configuration for reward system."""
    profit_factor_config: RewardConfig = None
    soul_config: SOULConfig = None
    
    # Component weights
    base_reward_weight: float = 1.0
    soul_penalty_weight: float = 2.0
    risk_adjustment_weight: float = 1.5
    
    # Normalization parameters
    reward_clip_min: float = -10.0
    reward_clip_max: float = 10.0
    ema_alpha: float = 0.01  # For reward normalization
    
    # Volatility scaling
    enable_volatility_scaling: bool = True
    target_volatility: float = 0.01


class UnifiedRewardEngine:
    """
    Unified reward engine combining all reward components.
    Produces normalized, dense rewards for RL training.
    """
    
    def __init__(self, config: Optional[UnifiedRewardConfig] = None):
        self.config = config or UnifiedRewardConfig()
        
        if self.config.profit_factor_config is None:
            self.config.profit_factor_config = RewardConfig()
        if self.config.soul_config is None:
            self.config.soul_config = SOULConfig()
        
        # Initialize sub-components
        self.profit_shaper = ProfitFactorShaper(self.config.profit_factor_config)
        self.soul_engine = SOULPenaltyEngine(self.config.soul_config)
        
        # Volatility scaler (optional)
        if self.config.enable_volatility_scaling:
            self.vol_scaler = AdaptiveRewardScaler(self.profit_shaper)
        else:
            self.vol_scaler = None
        
        # EMA for reward normalization
        self._reward_ema = 0.0
        self._reward_ema_var = 0.0
        self._step_count = 0
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Statistics
        self._stats = {
            'total_rewards': 0.0,
            'total_penalties': 0.0,
            'normalized_rewards': 0.0,
            'step_count': 0
        }
        
    def compute_reward(self,
                       state: Dict[str, np.ndarray],
                       action: int,
                       prev_action: int,
                       metrics: Dict[str, float]) -> Tuple[float, Dict[str, Any]]:
        """
        Compute unified reward for current step.
        
        Args:
            state: Current state with 'regime_features', 'orderbook_features'
            action: Current action
            prev_action: Previous action
            metrics: Dictionary with pnl, fees, turnover, inventory, etc.
            
        Returns:
            Normalized reward and detailed breakdown
        """
        with self._lock:
            start_time = time.perf_counter_ns()
            
            # Extract metrics
            current_pnl = metrics.get('pnl', 0.0)
            fees = metrics.get('fees', 0.0)
            turnover = metrics.get('turnover', 0.0)
            inventory = metrics.get('inventory', 0.0)
            avg_volume = metrics.get('avg_volume', 1.0)
            volatility = metrics.get('volatility', 0.01)
            
            # Step 1: Compute base reward from profit factor shaper
            base_reward, pf_breakdown = self.profit_shaper.compute_reward(
                current_pnl=current_pnl,
                fees=fees,
                turnover=turnover,
                inventory=inventory,
                avg_volume=avg_volume,
                action=action,
                prev_action=prev_action
            )
            
            # Apply volatility scaling if enabled
            if self.vol_scaler is not None:
                base_reward = self.vol_scaler.scale_reward(base_reward, volatility)
            
            # Step 2: Compute SOUL penalty
            regime_features = state.get('regime_features', np.zeros(6))
            orderbook_features = state.get('orderbook_features', np.zeros(6))
            
            soul_penalty, soul_meta = self.soul_engine.compute_penalty(
                regime_features=regime_features,
                orderbook_features=orderbook_features,
                action=action
            )
            
            # Step 3: Combine components with weights
            raw_reward = (
                base_reward * self.config.base_reward_weight +
                soul_penalty * self.config.soul_penalty_weight
            )
            
            # Step 4: Normalize reward using EMA
            normalized_reward = self._normalize_reward(raw_reward)
            
            # Step 5: Clip to prevent explosion
            clipped_reward = np.clip(
                normalized_reward,
                self.config.reward_clip_min,
                self.config.reward_clip_max
            )
            
            # Update statistics
            self._step_count += 1
            self._stats['total_rewards'] += base_reward
            self._stats['total_penalties'] += abs(soul_penalty)
            self._stats['normalized_rewards'] += clipped_reward
            self._stats['step_count'] = self._step_count
            
            # Advance soul engine step counter
            self.soul_engine.step()
            
            latency_us = (time.perf_counter_ns() - start_time) / 1000
            
            # Build comprehensive breakdown
            breakdown = {
                'base_reward': base_reward,
                'pf_breakdown': pf_breakdown,
                'soul_penalty': soul_penalty,
                'soul_matches': soul_meta.get('matches_found', 0),
                'raw_reward': raw_reward,
                'normalized_reward': normalized_reward,
                'final_reward': clipped_reward,
                'latency_us': latency_us,
                'reward_ema': self._reward_ema,
                'reward_std': np.sqrt(self._reward_ema_var + 1e-8)
            }
            
            return clipped_reward, breakdown
    
    def _normalize_reward(self, reward: float) -> float:
        """Normalize reward using exponential moving average."""
        if self._step_count == 0:
            self._reward_ema = reward
            self._reward_ema_var = 0.0
            return reward
        
        # Update EMA
        diff = reward - self._reward_ema
        self._reward_ema += self.config.ema_alpha * diff
        
        # Update variance EMA
        self._reward_ema_var = (
            (1 - self.config.ema_alpha) * 
            (self._reward_ema_var + self.config.ema_alpha * diff ** 2)
        )
        
        # Normalize
        std = np.sqrt(self._reward_ema_var + 1e-8)
        normalized = diff / std
        
        return normalized
    
    def record_mistake(self,
                       regime_features: np.ndarray,
                       orderbook_features: np.ndarray,
                       action: int,
                       consequence: str,
                       severity: float,
                       pnl_impact: float):
        """Record a mistake for future SOUL penalty."""
        self.soul_engine.record_mistake(
            regime_features=regime_features,
            orderbook_features=orderbook_features,
            action=action,
            consequence=consequence,
            severity=severity,
            pnl_impact=pnl_impact
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive reward statistics."""
        stats = self._stats.copy()
        
        # Add component statistics
        stats['profit_factor_stats'] = self.profit_shaper.get_statistics()
        stats['soul_stats'] = self.soul_engine.database.get_statistics()
        
        if self.vol_scaler is not None:
            stats['vol_scaling'] = self.vol_scaler.get_scaling_stats()
        
        stats['reward_ema'] = self._reward_ema
        stats['reward_std'] = np.sqrt(self._reward_ema_var + 1e-8)
        
        return stats
    
    def reset(self):
        """Reset all reward components."""
        with self._lock:
            self.profit_shaper.reset()
            self.soul_engine.reset()
            self._reward_ema = 0.0
            self._reward_ema_var = 0.0
            self._step_count = 0
            self._stats = {
                'total_rewards': 0.0,
                'total_penalties': 0.0,
                'normalized_rewards': 0.0,
                'step_count': 0
            }


class RayRLlibRewardWrapper:
    """
    Wrapper for Ray RLlib compatibility.
    Provides the standard RLlib reward function signature.
    """
    
    def __init__(self, reward_engine: UnifiedRewardEngine):
        self.engine = reward_engine
        self._prev_action = 0
        self._episode_reward = 0.0
        
    def __call__(self, 
                 obs: Dict[str, Any],
                 action: int,
                 info: Dict[str, Any]) -> float:
        """
        RLlib-compatible reward function.
        
        Args:
            obs: Observation dictionary
            action: Action taken
            info: Info dict with metrics
            
        Returns:
            Scalar reward
        """
        # Extract state from observation
        state = {
            'regime_features': obs.get('regime_features', np.zeros(6)),
            'orderbook_features': obs.get('orderbook_features', np.zeros(6))
        }
        
        # Extract metrics from info
        metrics = {
            'pnl': info.get('pnl', 0.0),
            'fees': info.get('fees', 0.0),
            'turnover': info.get('turnover', 0.0),
            'inventory': info.get('inventory', 0.0),
            'avg_volume': info.get('avg_volume', 1.0),
            'volatility': info.get('volatility', 0.01)
        }
        
        # Compute reward
        reward, breakdown = self.engine.compute_reward(
            state=state,
            action=action,
            prev_action=self._prev_action,
            metrics=metrics
        )
        
        # Track episode reward
        self._episode_reward += reward
        self._prev_action = action
        
        return reward
    
    def on_episode_end(self, info: Dict[str, Any]):
        """Callback for episode end."""
        info['episode_reward'] = self._episode_reward
        self._episode_reward = 0.0
        
    def reset_prev_action(self):
        """Reset previous action tracker."""
        self._prev_action = 0


# Module-level singleton
_reward_engine: Optional[UnifiedRewardEngine] = None
_rllib_wrapper: Optional[RayRLlibRewardWrapper] = None
_lock = threading.Lock()


def get_reward_engine(config: Optional[UnifiedRewardConfig] = None) -> UnifiedRewardEngine:
    """Get or create global reward engine."""
    global _reward_engine
    
    with _lock:
        if _reward_engine is None:
            _reward_engine = UnifiedRewardEngine(config)
        return _reward_engine


def get_rllib_wrapper(config: Optional[UnifiedRewardConfig] = None) -> RayRLlibRewardWrapper:
    """Get RLlib-compatible wrapper."""
    global _rllib_wrapper, _reward_engine
    
    with _lock:
        if _reward_engine is None:
            _reward_engine = UnifiedRewardEngine(config)
        if _rllib_wrapper is None:
            _rllib_wrapper = RayRLlibRewardWrapper(_reward_engine)
        return _rllib_wrapper


def reset_reward_system():
    """Reset the global reward system."""
    global _reward_engine, _rllib_wrapper
    
    with _lock:
        if _reward_engine is not None:
            _reward_engine.reset()
        if _rllib_wrapper is not None:
            _rllib_wrapper.reset_prev_action()


# Convenience functions
def compute_reward(state: Dict[str, np.ndarray],
                   action: int,
                   prev_action: int,
                   metrics: Dict[str, float]) -> Tuple[float, Dict]:
    """Compute reward using global engine."""
    engine = get_reward_engine()
    return engine.compute_reward(state, action, prev_action, metrics)


def get_reward_stats() -> Dict[str, Any]:
    """Get global reward statistics."""
    engine = get_reward_engine()
    return engine.get_statistics()


# Module exports
__all__ = [
    'UnifiedRewardConfig',
    'UnifiedRewardEngine',
    'RayRLlibRewardWrapper',
    'get_reward_engine',
    'get_rllib_wrapper',
    'reset_reward_system',
    'compute_reward',
    'get_reward_stats',
    'ProfitFactorShaper',
    'RewardConfig',
    'SOULPenaltyEngine',
    'SOULConfig'
]
