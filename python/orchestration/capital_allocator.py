"""
Real-time Capital Allocation Engine
Stage 49: Fractional Kelly and rolling Sortino ratios for dynamic capital scaling.
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import logging
import zmq

logger = logging.getLogger(__name__)


@dataclass
class StrategyAllocation:
    """Capital allocation state for a single strategy."""
    strategy_id: str
    allocated_capital: float
    max_capital: float
    kelly_fraction: float
    sortino_ratio: float
    win_rate: float
    avg_win: float
    avg_loss: float
    last_updated: datetime = field(default_factory=datetime.utcnow)


class RollingStatistics:
    """Efficient rolling statistics calculator with bounded memory."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.returns = deque(maxlen=window_size)
        self._sum = 0.0
        self._sum_sq = 0.0
        
    def add_return(self, ret: float):
        """Add a new return to the rolling window."""
        if len(self.returns) == self.window_size:
            old_ret = self.returns[0]
            self._sum -= old_ret
            self._sum_sq -= old_ret ** 2
        
        self.returns.append(ret)
        self._sum += ret
        self._sum_sq += ret ** 2
    
    @property
    def mean(self) -> float:
        """Calculate rolling mean."""
        n = len(self.returns)
        if n == 0:
            return 0.0
        return self._sum / n
    
    @property
    def std(self) -> float:
        """Calculate rolling standard deviation."""
        n = len(self.returns)
        if n < 2:
            return 0.0
        variance = (self._sum_sq / n) - (self.mean ** 2)
        return np.sqrt(max(0, variance))
    
    @property
    def downside_std(self) -> float:
        """Calculate rolling downside deviation (for Sortino ratio)."""
        n = len(self.returns)
        if n < 2:
            return 0.0
        
        downside_returns = [r for r in self.returns if r < 0]
        if len(downside_returns) == 0:
            return 0.0
        
        downside_var = sum(r ** 2 for r in downside_returns) / len(downside_returns)
        return np.sqrt(downside_var)
    
    @property
    def count(self) -> int:
        """Return number of samples in window."""
        return len(self.returns)


class CapitalAllocator:
    """
    Real-time capital allocation engine using Fractional Kelly and Sortino ratios.
    Dynamically scales strategy sizing without kernel restart.
    """
    
    def __init__(self, 
                 total_capital: float = 1_000_000.0,
                 max_kelly_fraction: float = 0.25,
                 min_sortino_threshold: float = 0.5,
                 allocation_window: int = 100):
        
        self.total_capital = total_capital
        self.max_kelly_fraction = max_kelly_fraction
        self.min_sortino_threshold = min_sortino_threshold
        self.allocation_window = allocation_window
        
        # Per-strategy tracking
        self.allocations: Dict[str, StrategyAllocation] = {}
        self.statistics: Dict[str, RollingStatistics] = {}
        self.pnl_history: Dict[str, deque] = {}
        
        # ZMQ socket for Rust IPC
        self._zmq_context = zmq.Context()
        self._zmq_socket = self._zmq_context.socket(zmq.PUSH)
        self._zmq_socket.connect("tcp://localhost:5558")
        
        # Pre-allocated numpy arrays for performance
        self._returns_buffer = np.zeros(allocation_window, dtype=np.float64)
        self._weights_buffer = np.zeros(allocation_window, dtype=np.float64)
    
    def register_strategy(self, strategy_id: str, initial_capital: float, 
                         max_capital: float) -> bool:
        """Register a strategy for capital allocation tracking."""
        if strategy_id in self.allocations:
            logger.warning(f"Strategy {strategy_id} already registered")
            return False
        
        self.allocations[strategy_id] = StrategyAllocation(
            strategy_id=strategy_id,
            allocated_capital=initial_capital,
            max_capital=max_capital,
            kelly_fraction=0.0,
            sortino_ratio=0.0,
            win_rate=0.0,
            avg_win=0.0,
            avg_loss=0.0,
        )
        
        self.statistics[strategy_id] = RollingStatistics(self.allocation_window)
        self.pnl_history[strategy_id] = deque(maxlen=self.allocation_window)
        
        logger.info(f"Strategy {strategy_id} registered with initial capital ${initial_capital:,.2f}")
        return True
    
    def record_pnl(self, strategy_id: str, pnl: float) -> None:
        """Record PnL for a strategy and update statistics."""
        if strategy_id not in self.statistics:
            logger.warning(f"Unknown strategy {strategy_id}")
            return
        
        # Calculate return based on allocated capital
        allocated = self.allocations[strategy_id].allocated_capital
        if allocated <= 0:
            return
        
        ret = pnl / allocated
        self.statistics[strategy_id].add_return(ret)
        self.pnl_history[strategy_id].append(pnl)
        
        # Update allocation periodically (every 10 records)
        if len(self.pnl_history[strategy_id]) % 10 == 0:
            self._update_allocation(strategy_id)
    
    def _calculate_kelly_fraction(self, strategy_id: str) -> float:
        """
        Calculate Fractional Kelly criterion.
        Kelly = W/L - (1-W)/L where W=win_rate, L=loss_ratio
        """
        stats = self.statistics[strategy_id]
        
        if stats.count < 10:
            return 0.0  # Not enough data
        
        # Calculate win rate and win/loss ratio
        returns = list(stats.returns)
        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r < 0]
        
        if len(wins) == 0 or len(losses) == 0:
            return 0.0
        
        win_rate = len(wins) / len(returns)
        avg_win = np.mean(wins)
        avg_loss = abs(np.mean(losses))
        
        if avg_loss == 0:
            return 0.0
        
        loss_ratio = avg_win / avg_loss
        
        # Full Kelly fraction
        kelly_full = win_rate - (1 - win_rate) / loss_ratio
        
        # Apply fractional Kelly (conservative scaling)
        kelly_fraction = kelly_full * self.max_kelly_fraction
        
        # Clamp to valid range
        return max(0.0, min(self.max_kelly_fraction, kelly_fraction))
    
    def _calculate_sortino_ratio(self, strategy_id: str) -> float:
        """Calculate rolling Sortino ratio."""
        stats = self.statistics[strategy_id]
        
        if stats.count < 10 or stats.downside_std == 0:
            return 0.0
        
        # Annualized Sortino (assuming daily returns)
        annualization_factor = np.sqrt(252)
        sortino = (stats.mean * annualization_factor) / (stats.downside_std * annualization_factor)
        
        return sortino
    
    def _update_allocation(self, strategy_id: str) -> None:
        """Update capital allocation for a strategy based on performance metrics."""
        if strategy_id not in self.allocations:
            return
        
        alloc = self.allocations[strategy_id]
        
        # Calculate metrics
        kelly_fraction = self._calculate_kelly_fraction(strategy_id)
        sortino_ratio = self._calculate_sortino_ratio(strategy_id)
        
        # Update statistics
        stats = self.statistics[strategy_id]
        returns = list(stats.returns)
        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r < 0]
        
        win_rate = len(wins) / len(returns) if returns else 0.0
        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = abs(np.mean(losses)) if losses else 0.0
        
        # Calculate new allocation
        # Scale by Sortino ratio (higher Sortino = more capital)
        sortino_multiplier = min(2.0, max(0.5, sortino_ratio / self.min_sortino_threshold))
        
        new_allocation = alloc.max_capital * kelly_fraction * sortino_multiplier
        new_allocation = min(new_allocation, alloc.max_capital)
        new_allocation = max(new_allocation, alloc.max_capital * 0.01)  # Minimum 1%
        
        # Check if significant change (>5%)
        if abs(new_allocation - alloc.allocated_capital) / alloc.allocated_capital > 0.05:
            old_allocation = alloc.allocated_capital
            alloc.allocated_capital = new_allocation
            alloc.kelly_fraction = kelly_fraction
            alloc.sortino_ratio = sortino_ratio
            alloc.win_rate = win_rate
            alloc.avg_win = avg_win
            alloc.avg_loss = avg_loss
            alloc.last_updated = datetime.utcnow()
            
            # Notify Rust side of allocation change
            self._notify_allocation_change(strategy_id, old_allocation, new_allocation)
            
            logger.info(
                f"Strategy {strategy_id} allocation updated: "
                f"${old_allocation:,.2f} -> ${new_allocation:,.2f} "
                f"(Kelly={kelly_fraction:.3f}, Sortino={sortino_ratio:.2f})"
            )
    
    def _notify_allocation_change(self, strategy_id: str, 
                                  old_alloc: float, new_alloc: float) -> None:
        """Send allocation update to Rust via ZMQ."""
        try:
            self._zmq_socket.send_json({
                'type': 'CAPITAL_ALLOCATION_UPDATE',
                'strategy_id': strategy_id,
                'old_allocation': old_alloc,
                'new_allocation': new_alloc,
                'change_pct': (new_alloc - old_alloc) / old_alloc if old_alloc > 0 else 0.0,
                'timestamp': datetime.utcnow().isoformat(),
            }, flags=zmq.NOBLOCK)
        except Exception as e:
            logger.error(f"Failed to send allocation update: {e}")
    
    def get_allocation(self, strategy_id: str) -> Optional[StrategyAllocation]:
        """Get current allocation for a strategy."""
        return self.allocations.get(strategy_id)
    
    def get_all_allocations(self) -> Dict[str, StrategyAllocation]:
        """Get all current allocations."""
        return self.allocations.copy()
    
    def get_total_allocated(self) -> float:
        """Get total capital currently allocated."""
        return sum(a.allocated_capital for a in self.allocations.values())
    
    def get_available_capital(self) -> float:
        """Get unallocated capital."""
        return self.total_capital - self.get_total_allocated()
    
    def rebalance_all(self) -> Dict[str, float]:
        """Rebalance all strategies based on current metrics."""
        updates = {}
        for strategy_id in self.allocations.keys():
            old_alloc = self.allocations[strategy_id].allocated_capital
            self._update_allocation(strategy_id)
            new_alloc = self.allocations[strategy_id].allocated_capital
            
            if old_alloc != new_alloc:
                updates[strategy_id] = new_alloc - old_alloc
        
        return updates
    
    def shutdown(self):
        """Cleanup resources."""
        self._zmq_socket.close()
        self._zmq_context.term()
        logger.info("CapitalAllocator shut down")


# Global instance
_allocator: Optional[CapitalAllocator] = None


def get_allocator() -> CapitalAllocator:
    """Get or create the global CapitalAllocator instance."""
    global _allocator
    if _allocator is None:
        _allocator = CapitalAllocator()
    return _allocator
