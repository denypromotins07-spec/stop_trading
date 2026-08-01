"""
Profit Factor Reward Shaper for RL
Implements a dense reward shaper that heavily penalizes high turnover, fee drag, 
and toxic inventory accumulation.

Strictly maximizes rolling profit factor and Sharpe ratio rather than just raw PnL, 
ensuring sustainable capital growth.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque


@dataclass
class RewardConfig:
    """Configuration for reward shaping."""
    # Base reward weights
    pnl_weight: float = 1.0
    sharpe_weight: float = 2.0
    profit_factor_weight: float = 3.0
    
    # Penalty weights
    turnover_penalty: float = 0.5
    fee_penalty: float = 1.0
    inventory_penalty: float = 0.3
    drawdown_penalty: float = 2.0
    
    # Rolling window sizes
    sharpe_window: int = 60
    profit_factor_window: int = 100
    
    # Thresholds
    max_turnover_rate: float = 0.1  # 10% per step
    max_inventory_ratio: float = 5.0  # Max inventory / avg_volume


class ProfitFactorShaper:
    """
    Dense reward shaper focusing on profit factor and risk-adjusted returns.
    """
    
    def __init__(self, config: Optional[RewardConfig] = None):
        self.config = config or RewardConfig()
        
        # Rolling buffers for statistics
        self._returns_buffer: deque = deque(maxlen=self.config.sharpe_window)
        self._pnl_buffer: deque = deque(maxlen=self.config.profit_factor_window)
        self._gross_profit_buffer: deque = deque(maxlen=self.config.profit_factor_window)
        self._gross_loss_buffer: deque = deque(maxlen=self.config.profit_factor_window)
        
        # State tracking
        self._cumulative_pnl = 0.0
        self._cumulative_fees = 0.0
        self._cumulative_turnover = 0.0
        self._peak_pnl = 0.0
        self._max_drawdown = 0.0
        
        # Inventory tracking
        self._inventory_history: deque = deque(maxlen=60)
        
    def compute_reward(self,
                       current_pnl: float,
                       fees: float,
                       turnover: float,
                       inventory: float,
                       avg_volume: float,
                       action: int,
                       prev_action: int) -> Tuple[float, Dict[str, float]]:
        """
        Compute dense reward for current step.
        
        Args:
            current_pnl: Current step PnL
            fees: Transaction fees paid
            turnover: Notional turnover this step
            inventory: Current position inventory
            avg_volume: Average market volume
            action: Current action taken
            prev_action: Previous action
            
        Returns:
            Total reward and breakdown dict
        """
        # Update cumulative tracking
        self._cumulative_pnl += current_pnl
        self._cumulative_fees += fees
        self._cumulative_turnover += turnover
        
        # Track peak and drawdown
        if self._cumulative_pnl > self._peak_pnl:
            self._peak_pnl = self._cumulative_pnl
        drawdown = self._peak_pnl - self._cumulative_pnl
        if drawdown > self._max_drawdown:
            self._max_drawdown = drawdown
        
        # Update rolling buffers
        self._returns_buffer.append(current_pnl)
        self._pnl_buffer.append(current_pnl)
        
        if current_pnl > 0:
            self._gross_profit_buffer.append(current_pnl)
            self._gross_loss_buffer.append(0.0)
        else:
            self._gross_profit_buffer.append(0.0)
            self._gross_loss_buffer.append(abs(current_pnl))
        
        self._inventory_history.append(inventory)
        
        # Compute individual reward components
        base_reward = current_pnl * self.config.pnl_weight
        
        # Sharpe ratio component (risk-adjusted)
        sharpe_component = self._compute_sharpe_component()
        
        # Profit factor component
        pf_component = self._compute_profit_factor_component()
        
        # Turnover penalty
        turnover_pen = self._compute_turnover_penalty(turnover, avg_volume)
        
        # Fee penalty
        fee_pen = fees * self.config.fee_penalty
        
        # Inventory penalty
        inv_pen = self._compute_inventory_penalty(inventory, avg_volume)
        
        # Drawdown penalty
        dd_pen = self._compute_drawdown_penalty(drawdown)
        
        # Action consistency penalty (discourage excessive flipping)
        flip_pen = self._compute_flip_penalty(action, prev_action)
        
        # Combine all components
        total_reward = (
            base_reward +
            sharpe_component * self.config.sharpe_weight +
            pf_component * self.config.profit_factor_weight -
            turnover_pen * self.config.turnover_penalty -
            fee_pen -
            inv_pen * self.config.inventory_penalty -
            dd_pen * self.config.drawdown_penalty -
            flip_pen
        )
        
        breakdown = {
            'base_pnl': base_reward,
            'sharpe_component': sharpe_component,
            'profit_factor_component': pf_component,
            'turnover_penalty': turnover_pen,
            'fee_penalty': fee_pen,
            'inventory_penalty': inv_pen,
            'drawdown_penalty': dd_pen,
            'flip_penalty': flip_pen,
            'total_reward': total_reward
        }
        
        return total_reward, breakdown
    
    def _compute_sharpe_component(self) -> float:
        """Compute Sharpe ratio based component."""
        if len(self._returns_buffer) < 10:
            return 0.0
        
        returns = np.array(self._returns_buffer)
        mean_ret = np.mean(returns)
        std_ret = np.std(returns) + 1e-8
        
        sharpe = mean_ret / std_ret
        # Clip to prevent extreme values
        return np.clip(sharpe, -5.0, 5.0)
    
    def _compute_profit_factor_component(self) -> float:
        """Compute profit factor based component."""
        if len(self._gross_profit_buffer) < 20:
            return 0.0
        
        gross_profits = sum(self._gross_profit_buffer)
        gross_losses = sum(self._gross_loss_buffer) + 1e-8
        
        profit_factor = gross_profits / gross_losses
        
        # Transform to reward: PF > 1 is good, PF < 1 is bad
        # Use log transform for symmetry
        if profit_factor > 1:
            return np.log(profit_factor)
        else:
            return -np.log(1 / profit_factor + 1e-8)
    
    def _compute_turnover_penalty(self, turnover: float, avg_volume: float) -> float:
        """Compute turnover-based penalty."""
        if avg_volume <= 0:
            return 0.0
        
        turnover_rate = turnover / avg_volume
        
        if turnover_rate > self.config.max_turnover_rate:
            # Exponential penalty for excessive turnover
            excess = turnover_rate - self.config.max_turnover_rate
            return excess ** 2
        
        return 0.0
    
    def _compute_inventory_penalty(self, inventory: float, avg_volume: float) -> float:
        """Compute inventory accumulation penalty."""
        if avg_volume <= 0:
            return 0.0
        
        inventory_ratio = abs(inventory) / avg_volume
        
        if inventory_ratio > self.config.max_inventory_ratio:
            # Penalize toxic inventory buildup
            excess = inventory_ratio - self.config.max_inventory_ratio
            return excess ** 2
        
        return 0.0
    
    def _compute_drawdown_penalty(self, drawdown: float) -> float:
        """Compute drawdown-based penalty."""
        if drawdown <= 0:
            return 0.0
        
        # Quadratic penalty for drawdowns
        return drawdown ** 0.5  # Square root for smoother gradient
    
    def _compute_flip_penalty(self, action: int, prev_action: int) -> float:
        """Penalize excessive action flipping."""
        if action != prev_action and action != 0 and prev_action != 0:
            # Flipped between long and short
            return 0.1
        return 0.0
    
    def get_statistics(self) -> Dict[str, float]:
        """Get current reward statistics."""
        stats = {
            'cumulative_pnl': self._cumulative_pnl,
            'cumulative_fees': self._cumulative_fees,
            'cumulative_turnover': self._cumulative_turnover,
            'peak_pnl': self._peak_pnl,
            'max_drawdown': self._max_drawdown,
            'current_drawdown': self._peak_pnl - self._cumulative_pnl
        }
        
        # Add rolling statistics
        if len(self._returns_buffer) >= 10:
            returns = np.array(self._returns_buffer)
            stats['rolling_sharpe'] = np.mean(returns) / (np.std(returns) + 1e-8)
        
        if len(self._gross_profit_buffer) >= 20:
            gp = sum(self._gross_profit_buffer)
            gl = sum(self._gross_loss_buffer) + 1e-8
            stats['rolling_profit_factor'] = gp / gl
        
        return stats
    
    def reset(self):
        """Reset all state."""
        self._returns_buffer.clear()
        self._pnl_buffer.clear()
        self._gross_profit_buffer.clear()
        self._gross_loss_buffer.clear()
        self._inventory_history.clear()
        
        self._cumulative_pnl = 0.0
        self._cumulative_fees = 0.0
        self._cumulative_turnover = 0.0
        self._peak_pnl = 0.0
        self._max_drawdown = 0.0


class AdaptiveRewardScaler:
    """
    Adaptively scales rewards based on market conditions.
    Prevents reward explosion during high volatility periods.
    """
    
    def __init__(self, base_shaper: ProfitFactorShaper,
                 adaptation_window: int = 100):
        self.shaper = base_shaper
        self.adaptation_window = adaptation_window
        
        self._reward_history: deque = deque(maxlen=adaptation_window)
        self._volatility_history: deque = deque(maxlen=adaptation_window)
        
    def scale_reward(self, 
                     raw_reward: float,
                     current_volatility: float) -> float:
        """
        Scale reward based on current volatility regime.
        
        Args:
            raw_reward: Raw computed reward
            current_volatility: Current market volatility
            
        Returns:
            Scaled reward
        """
        self._reward_history.append(raw_reward)
        self._volatility_history.append(current_volatility)
        
        if len(self._volatility_history) < 20:
            return raw_reward
        
        # Compute volatility percentile
        vol_array = np.array(self._volatility_history)
        current_percentile = np.searchsorted(np.sort(vol_array), current_volatility) / len(vol_array)
        
        # Scale down rewards during extreme volatility
        if current_percentile > 0.9:
            # High volatility: reduce reward magnitude
            scale_factor = 0.5
        elif current_percentile < 0.1:
            # Low volatility: normal rewards
            scale_factor = 1.0
        else:
            # Normal: slight scaling
            scale_factor = 0.8 + 0.2 * (1 - current_percentile)
        
        return raw_reward * scale_factor
    
    def get_scaling_stats(self) -> Dict[str, float]:
        """Get scaling statistics."""
        if len(self._volatility_history) < 10:
            return {'avg_scale_factor': 1.0}
        
        vol_array = np.array(self._volatility_history)
        return {
            'avg_volatility': np.mean(vol_array),
            'volatility_std': np.std(vol_array),
            'current_vol_percentile': (
                np.searchsorted(np.sort(vol_array), vol_array[-1]) / len(vol_array)
            )
        }


# Module exports
__all__ = ['RewardConfig', 'ProfitFactorShaper', 'AdaptiveRewardScaler']
