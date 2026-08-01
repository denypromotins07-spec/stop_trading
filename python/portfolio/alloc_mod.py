"""
Chapter 3: Dynamic Capital Allocation & Risk Budgeting
alloc_mod.py - Module root dynamically scaling Nautilus strategy instance sizes based on risk budgets and Kelly fractions
"""

import numpy as np
from typing import Dict, Optional, Tuple, List, Any
from dataclasses import dataclass, field
import threading
from collections import deque

# Import local modules
from .risk_budgeting import (
    RiskBudgetingEngine, 
    RiskBudgetingResult,
    create_risk_budgeting_engine,
    calculate_portfolio_volatility
)
from .fat_tail_kelly import (
    FatTailKellyCalculator,
    KellyResult,
    create_kelly_calculator
)


@dataclass
class StrategyAllocation:
    """Allocation for a single strategy instance"""
    strategy_id: str
    symbol: str
    
    # Position sizing
    raw_size: float         # Raw position size from signal
    scaled_size: float      # Size after risk/Kelly scaling
    max_size: float         # Maximum allowed size
    
    # Risk metrics
    risk_budget_pct: float  # Allocated risk budget percentage
    kelly_fraction: float   # Applied Kelly fraction
    volatility_adjustment: float
    
    # Limits
    var_limit: float        # VaR limit for this strategy
    drawdown_limit: float   # Max drawdown before reduction
    
    # State
    active: bool = True
    reduced_due_to_risk: bool = False


@dataclass
class PortfolioAllocation:
    """Complete portfolio allocation state"""
    timestamp: int
    total_capital: float
    allocated_capital: float
    available_capital: float
    
    # Per-strategy allocations
    strategy_allocations: Dict[str, StrategyAllocation] = field(default_factory=dict)
    
    # Portfolio metrics
    portfolio_var: float
    portfolio_volatility: float
    total_risk_usage: float
    risk_budget_remaining: float
    
    # Kelly metrics
    aggregate_kelly: float
    recommended_leverage: float


class NautilusStrategyScaler:
    """
    Dynamically scale Nautilus strategy instance sizes based on:
    1. Real-time risk budgets from RiskBudgetingEngine
    2. Kelly fractions from FatTailKellyCalculator
    3. Current market volatility regime
    """
    
    def __init__(
        self,
        total_capital: float = 1_000_000.0,
        max_portfolio_var: float = 0.02,
        max_strategy_allocation: float = 0.25,
        default_kelly_fraction: float = 0.5
    ):
        self.total_capital = total_capital
        self.max_portfolio_var = max_portfolio_var
        self.max_strategy_allocation = max_strategy_allocation
        self.default_kelly_fraction = default_kelly_fraction
        
        # Initialize engines
        self.risk_engine = create_risk_budgeting_engine()
        self.kelly_calc = create_kelly_calculator(default_kelly_fraction)
        
        # State tracking
        self._allocations: Dict[str, StrategyAllocation] = {}
        self._returns_history: Dict[str, deque] = {}
        self._lock = threading.Lock()
        
        # Volatility regime detection
        self._volatility_regime = 'normal'  # normal, elevated, extreme
        self._volatility_threshold_high = 0.02
        self._volatility_threshold_extreme = 0.05
    
    def update_strategy_allocation(
        self,
        strategy_id: str,
        symbol: str,
        raw_signal_size: float,
        returns_history: np.ndarray,
        covariance_row: np.ndarray,
        all_covariance_rows: np.ndarray
    ) -> StrategyAllocation:
        """
        Update allocation for a single strategy.
        
        Args:
            strategy_id: Unique strategy identifier
            symbol: Trading pair symbol
            raw_signal_size: Raw position size from alpha signal
            returns_history: Historical returns for this strategy
            covariance_row: Covariance of this strategy with others
            all_covariance_rows: Full covariance matrix rows
        
        Returns:
            Updated StrategyAllocation
        """
        with self._lock:
            # Initialize returns history if needed
            if strategy_id not in self._returns_history:
                self._returns_history[strategy_id] = deque(maxlen=1000)
            
            # Add new returns
            for r in returns_history[-10:]:  # Batch update
                self._returns_history[strategy_id].append(r)
            
            # Calculate Kelly fraction
            hist_returns = np.array(list(self._returns_history[strategy_id]))
            kelly_result = self.kelly_calc.calculate(hist_returns)
            
            # Get historical returns for volatility estimate
            if len(hist_returns) > 0:
                strategy_vol = np.std(hist_returns)
            else:
                strategy_vol = 0.01  # Default
            
            # Determine volatility regime
            self._update_volatility_regime(strategy_vol)
            
            # Calculate volatility adjustment
            vol_adjustment = self._calculate_volatility_adjustment(strategy_vol)
            
            # Calculate risk budget allocation
            n_strategies = len(self._allocations) + 1
            base_risk_budget = 1.0 / n_strategies
            
            # Apply Kelly scaling
            kelly_scaled_budget = base_risk_budget * kelly_result.recommended_fraction
            
            # Apply volatility adjustment
            final_risk_budget = kelly_scaled_budget * vol_adjustment
            
            # Ensure within limits
            final_risk_budget = min(final_risk_budget, self.max_strategy_allocation)
            
            # Calculate maximum position size
            max_size = self.total_capital * self.max_strategy_allocation
            
            # Scale raw signal by risk budget
            scaled_size = raw_signal_size * final_risk_budget * self.total_capital
            
            # Apply VaR limit
            var_limit = self._calculate_var_limit(strategy_id, final_risk_budget)
            
            # Create/update allocation
            allocation = StrategyAllocation(
                strategy_id=strategy_id,
                symbol=symbol,
                raw_size=raw_signal_size,
                scaled_size=scaled_size,
                max_size=max_size,
                risk_budget_pct=final_risk_budget,
                kelly_fraction=kelly_result.recommended_fraction,
                volatility_adjustment=vol_adjustment,
                var_limit=var_limit,
                drawdown_limit=self._calculate_drawdown_limit(strategy_id),
                active=True,
                reduced_due_to_risk=(scaled_size < raw_signal_size)
            )
            
            self._allocations[strategy_id] = allocation
            
            return allocation
    
    def get_portfolio_allocation(self) -> PortfolioAllocation:
        """Get current complete portfolio allocation state."""
        with self._lock:
            allocated = 0.0
            total_var = 0.0
            total_kelly = 0.0
            
            for alloc in self._allocations.values():
                if alloc.active:
                    allocated += abs(alloc.scaled_size)
                    total_var += alloc.var_limit
                    total_kelly += alloc.kelly_fraction
            
            n_active = len([a for a in self._allocations.values() if a.active])
            avg_kelly = total_kelly / n_active if n_active > 0 else 0.0
            
            # Estimate portfolio volatility (simplified)
            port_vol = np.sqrt(sum(a.var_limit ** 2 for a in self._allocations.values() if a.active))
            
            # Total risk usage
            risk_usage = total_var / self.max_portfolio_var if self.max_portfolio_var > 0 else 0.0
            
            return PortfolioAllocation(
                timestamp=0,  # Would be set by caller
                total_capital=self.total_capital,
                allocated_capital=allocated,
                available_capital=self.total_capital - allocated,
                strategy_allocations=dict(self._allocations),
                portfolio_var=total_var,
                portfolio_volatility=port_vol,
                total_risk_usage=risk_usage,
                risk_budget_remaining=max(0, 1.0 - risk_usage),
                aggregate_kelly=avg_kelly,
                recommended_leverage=1.0 / (1.0 - avg_kelly) if avg_kelly < 1 else 10.0
            )
    
    def _update_volatility_regime(self, current_vol: float):
        """Update volatility regime based on current conditions."""
        if current_vol > self._volatility_threshold_extreme:
            self._volatility_regime = 'extreme'
        elif current_vol > self._volatility_threshold_high:
            self._volatility_regime = 'elevated'
        else:
            self._volatility_regime = 'normal'
    
    def _calculate_volatility_adjustment(self, strategy_vol: float) -> float:
        """Calculate position size adjustment based on volatility regime."""
        if self._volatility_regime == 'extreme':
            return 0.25  # Reduce to 25% in extreme volatility
        elif self._volatility_regime == 'elevated':
            return 0.5   # Reduce to 50% in elevated volatility
        else:
            return 1.0   # Normal sizing
    
    def _calculate_var_limit(self, strategy_id: str, risk_budget: float) -> float:
        """Calculate VaR limit for a strategy."""
        # Simple parametric VaR at 99% confidence
        if strategy_id in self._returns_history:
            returns = np.array(list(self._returns_history[strategy_id]))
            if len(returns) > 0:
                vol = np.std(returns)
                # 99% VaR = 2.33 * sigma
                return 2.33 * vol * risk_budget * self.total_capital
        
        return 0.01 * risk_budget * self.total_capital  # Default 1%
    
    def _calculate_drawdown_limit(self, strategy_id: str) -> float:
        """Calculate maximum drawdown limit before position reduction."""
        # Base limit: 10% of allocated capital
        if strategy_id in self._allocations:
            alloc = self._allocations[strategy_id]
            return abs(alloc.scaled_size) * 0.10
        
        return self.total_capital * 0.001  # Default 0.1% of total
    
    def reduce_strategy_risk(
        self,
        strategy_id: str,
        reduction_factor: float = 0.5
    ) -> Optional[StrategyAllocation]:
        """Manually reduce a strategy's risk allocation."""
        with self._lock:
            if strategy_id not in self._allocations:
                return None
            
            alloc = self._allocations[strategy_id]
            alloc.scaled_size *= reduction_factor
            alloc.risk_budget_pct *= reduction_factor
            alloc.reduced_due_to_risk = True
            
            return alloc
    
    def deactivate_strategy(self, strategy_id: str) -> bool:
        """Deactivate a strategy (set allocation to zero)."""
        with self._lock:
            if strategy_id not in self._allocations:
                return False
            
            self._allocations[strategy_id].active = False
            self._allocations[strategy_id].scaled_size = 0.0
            
            return True
    
    def reactivate_strategy(self, strategy_id: str) -> bool:
        """Reactivate a previously deactivated strategy."""
        with self._lock:
            if strategy_id not in self._allocations:
                return False
            
            self._allocations[strategy_id].active = True
            return True


# Module singleton instance
_alloc_module: Optional[NautilusStrategyScaler] = None


def get_allocation_module(
    total_capital: float = 1_000_000.0,
    max_var: float = 0.02
) -> NautilusStrategyScaler:
    """Get or create the global allocation module instance."""
    global _alloc_module
    if _alloc_module is None:
        _alloc_module = NautilusStrategyScaler(total_capital, max_var)
    return _alloc_module


def reset_allocation_module():
    """Reset the global allocation module (for testing)."""
    global _alloc_module
    _alloc_module = None


# Convenience functions
def quick_strategy_scaling(
    raw_size: float,
    strategy_returns: np.ndarray,
    total_capital: float = 1_000_000.0,
    kelly_fraction: float = 0.5
) -> float:
    """
    Quick strategy size calculation with Kelly scaling.
    
    Returns:
        Scaled position size
    """
    calc = create_kelly_calculator(kelly_fraction)
    result = calc.calculate(strategy_returns)
    
    # Scale by Kelly fraction and capital
    return raw_size * result.recommended_bet_size * total_capital


def calculate_nautilus_position(
    strategy_id: str,
    symbol: str,
    raw_signal: float,
    returns: np.ndarray,
    current_volatility: float,
    total_capital: float = 1_000_000.0
) -> Dict[str, Any]:
    """
    Calculate complete Nautilus position sizing.
    
    Returns:
        Dictionary with all sizing parameters
    """
    scaler = get_allocation_module(total_capital)
    
    # Mock covariance (would come from real portfolio)
    cov_row = np.array([0.0001])
    cov_matrix = np.array([[0.0001]])
    
    # Update allocation
    alloc = scaler.update_strategy_allocation(
        strategy_id, symbol, raw_signal, returns, cov_row, cov_matrix
    )
    
    return {
        'strategy_id': strategy_id,
        'symbol': symbol,
        'raw_size': alloc.raw_size,
        'scaled_size': alloc.scaled_size,
        'max_size': alloc.max_size,
        'risk_budget_pct': alloc.risk_budget_pct,
        'kelly_fraction': alloc.kelly_fraction,
        'volatility_adjustment': alloc.volatility_adjustment,
        'var_limit': alloc.var_limit,
        'active': alloc.active,
        'reduced': alloc.reduced_due_to_risk
    }
