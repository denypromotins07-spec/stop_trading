"""
Non-linear transaction cost and market impact penalty function for convex optimization.
Constrains the optimizer from over-trading by accounting for bid-ask spread and predicted slippage.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class MarketImpactParams:
    """Parameters for market impact model."""
    # Almgren-Chriss style parameters
    temporary_impact_coefficient: float = 0.1  # Linear temporary impact
    permanent_impact_coefficient: float = 0.01  # Linear permanent impact
    nonlinearity_exponent: float = 0.5  # Square-root impact (typical)
    
    # Asset-specific liquidity
    avg_daily_volume: float = 1e6  # USD
    participation_rate_limit: float = 0.1  # Max 10% of ADV
    
    # Spread parameters
    base_spread_bps: float = 10.0  # Base bid-ask spread in bps
    spread_sensitivity: float = 0.5  # How spread widens with size


@dataclass
class TransactionCostEstimate:
    """Estimated transaction costs for a trade."""
    # Cost components (in basis points)
    spread_cost_bps: float = 0.0
    temporary_impact_bps: float = 0.0
    permanent_impact_bps: float = 0.0
    fixed_cost_bps: float = 0.0
    
    # Total cost
    total_cost_bps: float = 0.0
    total_cost_usd: float = 0.0
    
    # Trade details
    trade_value_usd: float = 0.0
    participation_rate: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'spread_cost_bps': self.spread_cost_bps,
            'temporary_impact_bps': self.temporary_impact_bps,
            'permanent_impact_bps': self.permanent_impact_bps,
            'fixed_cost_bps': self.fixed_cost_bps,
            'total_cost_bps': self.total_cost_bps,
            'total_cost_usd': self.total_cost_usd,
            'trade_value_usd': self.trade_value_usd,
            'participation_rate': self.participation_rate
        }


class NonLinearTransactionCostModel:
    """
    Non-linear transaction cost model for portfolio optimization.
    
    Implements:
    - Bid-ask spread costs (linear)
    - Temporary market impact (non-linear, typically square-root)
    - Permanent market impact (linear)
    - Fixed costs (commissions, fees)
    
    The model can be used to:
    1. Estimate costs for given trades
    2. Generate cvxpy-compatible cost functions for optimization
    3. Constrain trades based on cost budgets
    """
    
    def __init__(
        self,
        params: Optional[MarketImpactParams] = None,
        asset_params: Optional[Dict[str, MarketImpactParams]] = None
    ):
        self.default_params = params or MarketImpactParams()
        self.asset_params = asset_params or {}
        
        # Pre-computed coefficients for fast evaluation
        self._cost_cache: Dict[str, Any] = {}
    
    def get_asset_params(self, asset_id: str) -> MarketImpactParams:
        """Get parameters for specific asset."""
        return self.asset_params.get(asset_id, self.default_params)
    
    def estimate_cost(
        self,
        asset_id: str,
        trade_size: float,  # Signed: positive = buy, negative = sell
        asset_price: float,
        adv_usd: Optional[float] = None
    ) -> TransactionCostEstimate:
        """
        Estimate transaction cost for a single asset trade.
        
        Args:
            asset_id: Asset identifier
            trade_size: Number of shares (signed)
            asset_price: Current price per share
            adv_usd: Average daily volume in USD (optional override)
            
        Returns:
            TransactionCostEstimate with cost breakdown
        """
        params = self.get_asset_params(asset_id)
        
        trade_value = abs(trade_size) * asset_price
        
        # Participation rate
        effective_adv = adv_usd or params.avg_daily_volume
        participation_rate = trade_value / (effective_adv + 1e-8)
        
        # 1. Spread cost (linear in trade size)
        spread_bps = params.base_spread_bps * (1 + params.spread_sensitivity * participation_rate)
        spread_cost_bps = spread_bps
        
        # 2. Temporary impact (non-linear)
        # Standard model: impact = coefficient * (size/ADV)^exponent
        temp_impact_bps = (params.temporary_impact_coefficient * 100 * 
                          (participation_rate ** params.nonlinearity_exponent))
        
        # 3. Permanent impact (linear in trade size relative to ADV)
        perm_impact_bps = params.permanent_impact_coefficient * 100 * participation_rate
        
        # 4. Fixed costs (commissions, exchange fees)
        fixed_cost_bps = 1.0  # Base commission
        
        # Total cost
        total_bps = spread_cost_bps + temp_impact_bps + perm_impact_bps + fixed_cost_bps
        total_usd = trade_value * total_bps / 10000
        
        return TransactionCostEstimate(
            spread_cost_bps=spread_cost_bps,
            temporary_impact_bps=temp_impact_bps,
            permanent_impact_bps=perm_impact_bps,
            fixed_cost_bps=fixed_cost_bps,
            total_cost_bps=total_bps,
            total_cost_usd=total_usd,
            trade_value_usd=trade_value,
            participation_rate=participation_rate
        )
    
    def estimate_portfolio_costs(
        self,
        asset_ids: List[str],
        current_weights: np.ndarray,
        target_weights: np.ndarray,
        prices: np.ndarray,
        portfolio_value: float,
        adv_usd: Optional[Dict[str, float]] = None
    ) -> Dict[str, TransactionCostEstimate]:
        """
        Estimate costs for portfolio rebalancing.
        
        Args:
            asset_ids: List of asset identifiers
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            prices: Current prices per share
            portfolio_value: Total portfolio value in USD
            adv_usd: ADV overrides by asset
            
        Returns:
            Dict mapping asset_id to cost estimate
        """
        estimates = {}
        
        for i, asset_id in enumerate(asset_ids):
            weight_change = target_weights[i] - current_weights[i]
            trade_value = weight_change * portfolio_value
            trade_size = trade_value / (prices[i] + 1e-8)
            
            adv = adv_usd.get(asset_id) if adv_usd else None
            
            estimate = self.estimate_cost(
                asset_id, trade_size, prices[i], adv
            )
            estimates[asset_id] = estimate
        
        return estimates
    
    def get_total_rebalancing_cost(
        self,
        asset_ids: List[str],
        current_weights: np.ndarray,
        target_weights: np.ndarray,
        portfolio_value: float
    ) -> float:
        """
        Get total estimated cost for rebalancing (in USD).
        """
        estimates = self.estimate_portfolio_costs(
            asset_ids, current_weights, target_weights,
            np.ones(len(asset_ids)),  # Dummy prices (cancel out)
            portfolio_value
        )
        
        return sum(e.total_cost_usd for e in estimates.values())
    
    def create_cvxpy_cost_function(
        self,
        n_assets: int,
        linear_tc_rate: Optional[float] = None,
        quadratic_tc_rate: Optional[float] = None
    ) -> Callable:
        """
        Create a cvxpy-compatible cost function for optimization.
        
        This returns a function that can be used in cvxpy objectives.
        For non-linear costs, we use a piecewise linear approximation
        or convex surrogate.
        
        Args:
            n_assets: Number of assets
            linear_tc_rate: Override for linear cost coefficient
            quadratic_tc_rate: Override for quadratic cost coefficient
            
        Returns:
            Function that takes weight change vector and returns cost expression
        """
        import cvxpy as cp
        
        # Use average parameters for simplified model
        avg_temp = self.default_params.temporary_impact_coefficient
        avg_perm = self.default_params.permanent_impact_coefficient
        avg_spread = self.default_params.base_spread_bps / 10000
        
        lin_rate = linear_tc_rate or (avg_spread + avg_perm * 0.1)
        quad_rate = quadratic_tc_rate or (avg_temp * 0.01)
        
        def cost_func(weight_changes: cp.Variable) -> cp.Expression:
            """
            Compute transaction cost as function of weight changes.
            
            Uses convex approximation:
            TC(w) = linear_rate * |w| + quadratic_rate * w^2
            """
            abs_changes = cp.abs(weight_changes)
            squared_changes = cp.square(weight_changes)
            
            return lin_rate * cp.sum(abs_changes) + quad_rate * cp.sum(squared_changes)
        
        return cost_func
    
    def get_marginal_cost_derivative(
        self,
        asset_id: str,
        trade_size: float,
        asset_price: float
    ) -> float:
        """
        Compute marginal cost (derivative) for small trade size changes.
        
        Useful for gradient-based optimization methods.
        """
        params = self.get_asset_params(asset_id)
        
        trade_value = abs(trade_size) * asset_price
        participation_rate = trade_value / (params.avg_daily_volume + 1e-8)
        
        # Derivative of cost w.r.t. trade size
        # d/dx [a*x + b*x^n] = a + b*n*x^(n-1)
        
        spread_deriv = params.base_spread_bps / 10000
        
        exp = params.nonlinearity_exponent
        temp_coeff = params.temporary_impact_coefficient
        
        if participation_rate > 1e-8:
            temp_deriv = temp_coeff * exp * (participation_rate ** (exp - 1)) / params.avg_daily_volume
        else:
            temp_deriv = 0.0
        
        perm_deriv = params.permanent_impact_coefficient / params.avg_daily_volume
        
        return (spread_deriv + temp_deriv + perm_deriv) * asset_price


class CostAwareRebalancer:
    """
    Utility class for cost-aware portfolio rebalancing decisions.
    Determines whether rebalancing is worthwhile given transaction costs.
    """
    
    def __init__(
        self,
        cost_model: NonLinearTransactionCostModel,
        min_benefit_threshold_bps: float = 5.0
    ):
        self.cost_model = cost_model
        self.min_benefit_threshold = min_benefit_threshold_bps
    
    def should_rebalance(
        self,
        asset_ids: List[str],
        current_weights: np.ndarray,
        target_weights: np.ndarray,
        expected_alpha_bps: float,
        portfolio_value: float
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Determine if rebalancing is economically justified.
        
        Args:
            asset_ids: Asset identifiers
            current_weights: Current weights
            target_weights: Target weights
            expected_alpha_bps: Expected alpha from rebalancing (in bps)
            portfolio_value: Portfolio value in USD
            
        Returns:
            Tuple of (should_rebalance, analysis_dict)
        """
        # Estimate transaction costs
        total_cost = self.cost_model.get_total_rebalancing_cost(
            asset_ids, current_weights, target_weights, portfolio_value
        )
        
        cost_bps = (total_cost / portfolio_value) * 10000
        
        # Net benefit
        net_benefit_bps = expected_alpha_bps - cost_bps
        
        should = net_benefit_bps > self.min_benefit_threshold
        
        analysis = {
            'expected_alpha_bps': expected_alpha_bps,
            'estimated_cost_bps': cost_bps,
            'net_benefit_bps': net_benefit_bps,
            'threshold_bps': self.min_benefit_threshold,
            'should_rebalance': should,
            'cost_usd': total_cost
        }
        
        return should, analysis
    
    def get_optimal_partial_rebalance(
        self,
        asset_ids: List[str],
        current_weights: np.ndarray,
        target_weights: np.ndarray,
        max_cost_bps: float = 10.0,
        portfolio_value: float = 1e6
    ) -> np.ndarray:
        """
        Compute optimal partial rebalance given cost budget.
        
        Scales rebalancing to stay within cost budget while maximizing
        movement toward target.
        """
        weight_diff = target_weights - current_weights
        
        # Binary search for optimal scaling factor
        low, high = 0.0, 1.0
        optimal_scale = 0.0
        
        for _ in range(20):
            mid = (low + high) / 2
            test_weights = current_weights + mid * weight_diff
            
            cost = self.cost_model.get_total_rebalancing_cost(
                asset_ids, current_weights, test_weights, portfolio_value
            )
            cost_bps = (cost / portfolio_value) * 10000
            
            if cost_bps <= max_cost_bps:
                optimal_scale = mid
                low = mid
            else:
                high = mid
        
        return current_weights + optimal_scale * weight_diff


# Factory function
def create_cost_model(
    temporary_impact: float = 0.1,
    permanent_impact: float = 0.01,
    base_spread_bps: float = 10.0
) -> NonLinearTransactionCostModel:
    """Factory function to create cost model."""
    params = MarketImpactParams(
        temporary_impact_coefficient=temporary_impact,
        permanent_impact_coefficient=permanent_impact,
        base_spread_bps=base_spread_bps
    )
    return NonLinearTransactionCostModel(params)
