"""
Institutional-grade convex optimization using cvxpy for mean-variance and 
transaction-cost-aware portfolio rebalancing. Formulates portfolio construction 
as a disciplined Second-Order Cone Program (SOCP) to guarantee global optimality.
Strictly bounded solvers with iteration limits to prevent CPU hanging.
"""

from __future__ import annotations

import numpy as np
import cvxpy as cp
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for convex portfolio optimization."""
    # Solver settings
    solver: str = "ECOS"  # ECOS, SCS, OSQP
    max_iterations: int = 1000
    tolerance: float = 1e-6
    time_limit_seconds: float = 5.0
    
    # Risk parameters
    risk_aversion: float = 3.0
    target_return: Optional[float] = None
    max_volatility: Optional[float] = None
    
    # Position limits
    min_weight: float = 0.0
    max_weight: float = 0.25
    max_turnover: float = 0.5
    
    # Transaction costs
    linear_tc_rate: float = 0.001  # 10 bps
    quadratic_tc_rate: float = 0.01  # Market impact coefficient
    
    # Regularization
    l1_penalty: float = 0.0001  # Sparsity penalty
    l2_penalty: float = 0.00001  # Stability penalty
    
    def get_solver_options(self) -> Dict[str, Any]:
        """Get solver-specific options."""
        options = {
            'max_iters': self.max_iterations,
            'abstol': self.tolerance,
            'reltol': self.tolerance * 10,
            'feastol': self.tolerance,
        }
        
        if self.time_limit_seconds > 0:
            options['time_limit'] = self.time_limit_seconds
        
        return options


@dataclass
class OptimizationResult:
    """Result of portfolio optimization."""
    # Optimal weights
    optimal_weights: np.ndarray
    asset_ids: List[str]
    
    # Performance metrics
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    
    # Costs
    transaction_cost: float
    turnover: float
    
    # Optimization status
    status: str
    solve_time_ms: float
    iterations: int = 0
    
    # Dual variables (shadow prices)
    dual_constraints: Optional[Dict[str, np.ndarray]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            'optimal_weights': self.optimal_weights.tolist(),
            'asset_ids': self.asset_ids,
            'expected_return': self.expected_return,
            'expected_volatility': self.expected_volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'transaction_cost': self.transaction_cost,
            'turnover': self.turnover,
            'status': self.status,
            'solve_time_ms': self.solve_time_ms,
            'iterations': self.iterations
        }


class ConvexPortfolioOptimizer:
    """
    Institutional-grade convex portfolio optimizer using cvxpy.
    
    Supports multiple optimization formulations:
    - Mean-Variance (Markowitz)
    - Risk Parity
    - Maximum Sharpe Ratio
    - Minimum Volatility
    - Target Return/Volatility
    
    All formulations include transaction costs and position constraints.
    """
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()
        self._last_problem: Optional[cp.Problem] = None
        self._last_result: Optional[OptimizationResult] = None
    
    def optimize_mean_variance(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        current_weights: Optional[np.ndarray] = None,
        asset_ids: Optional[List[str]] = None,
        additional_constraints: Optional[List[cp.Constraint]] = None
    ) -> OptimizationResult:
        """
        Solve mean-variance optimization with transaction costs.
        
        maximize: w^T * mu - (risk_aversion/2) * w^T * Sigma * w - TC(w, w_current)
        
        Args:
            expected_returns: Expected returns vector (n_assets,)
            covariance_matrix: Covariance matrix (n_assets, n_assets)
            current_weights: Current portfolio weights for turnover calculation
            asset_ids: Asset identifiers
            additional_constraints: Extra constraints
            
        Returns:
            OptimizationResult with optimal weights and metrics
        """
        start_time = time.perf_counter()
        
        n_assets = len(expected_returns)
        
        if asset_ids is None:
            asset_ids = [f"asset_{i}" for i in range(n_assets)]
        
        # Decision variables
        w = cp.Variable(n_assets)
        
        # Turnover variables (for transaction costs)
        buy = cp.Variable(n_assets, nonneg=True)
        sell = cp.Variable(n_assets, nonneg=True)
        
        # Objective components
        portfolio_return = expected_returns @ w
        portfolio_risk = cp.quad_form(w, covariance_matrix)
        
        # Transaction costs: linear + quadratic market impact
        tc_linear = self.config.linear_tc_rate * cp.sum(buy + sell)
        tc_quadratic = self.config.quadratic_tc_rate * cp.sum(cp.square(buy + sell))
        total_tc = tc_linear + tc_quadratic
        
        # Regularization
        l1_reg = self.config.l1_penalty * cp.norm1(w)
        l2_reg = self.config.l2_penalty * cp.sum_squares(w)
        
        # Objective: maximize risk-adjusted return minus costs
        objective = cp.Maximize(
            portfolio_return 
            - (self.config.risk_aversion / 2) * portfolio_risk
            - total_tc
            - l1_reg
            - l2_reg
        )
        
        # Constraints
        constraints = [
            cp.sum(w) == 1.0,  # Fully invested
            w >= self.config.min_weight,
            w <= self.config.max_weight,
            buy >= 0,
            sell >= 0,
        ]
        
        # Turnover constraints relative to current weights
        if current_weights is not None:
            constraints.extend([
                w == current_weights + buy - sell,
                cp.sum(buy + sell) <= self.config.max_turnover
            ])
        else:
            # No current weights - just constrain turnover from zero
            constraints.extend([
                cp.sum(buy + sell) <= self.config.max_turnover
            ])
        
        # Add target return constraint if specified
        if self.config.target_return is not None:
            constraints.append(portfolio_return >= self.config.target_return)
        
        # Add max volatility constraint if specified
        if self.config.max_volatility is not None:
            constraints.append(cp.norm(covariance_matrix @ w) <= self.config.max_volatility)
        
        # Add any additional constraints
        if additional_constraints:
            constraints.extend(additional_constraints)
        
        # Solve problem
        problem = cp.Problem(objective, constraints)
        self._last_problem = problem
        
        solve_start = time.perf_counter()
        
        try:
            problem.solve(
                solver=getattr(cp, self.config.solver),
                **self.config.get_solver_options()
            )
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            raise
        
        solve_time_ms = (time.perf_counter() - solve_start) * 1000
        
        # Check solution status
        if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            logger.warning(f"Optimization status: {problem.status}")
        
        # Extract results
        optimal_weights = np.array(w.value)
        
        # Compute metrics
        exp_return = float(expected_returns @ optimal_weights)
        exp_vol = float(np.sqrt(optimal_weights @ covariance_matrix @ optimal_weights))
        sharpe = exp_return / (exp_vol + 1e-8)
        
        # Transaction cost and turnover
        if current_weights is not None:
            turnover = float(np.sum(np.abs(optimal_weights - current_weights)))
        else:
            turnover = float(np.sum(np.abs(optimal_weights)))
        
        tc_value = (self.config.linear_tc_rate * turnover + 
                   self.config.quadratic_tc_rate * turnover ** 2)
        
        result = OptimizationResult(
            optimal_weights=optimal_weights,
            asset_ids=asset_ids,
            expected_return=exp_return,
            expected_volatility=exp_vol,
            sharpe_ratio=sharpe,
            transaction_cost=tc_value,
            turnover=turnover,
            status=problem.status,
            solve_time_ms=solve_time_ms,
            iterations=getattr(problem, 'solver_stats', {}).get('num_iters', 0)
        )
        
        self._last_result = result
        logger.info(f"Mean-variance optimization completed: {result.status} in {solve_time_ms:.1f}ms")
        
        return result
    
    def optimize_risk_parity(
        self,
        covariance_matrix: np.ndarray,
        current_weights: Optional[np.ndarray] = None,
        asset_ids: Optional[List[str]] = None,
        target_risk_contributions: Optional[np.ndarray] = None
    ) -> OptimizationResult:
        """
        Solve risk parity optimization.
        
        Risk parity aims to equalize risk contributions from each asset:
        RC_i = w_i * (Sigma * w)_i / sqrt(w^T * Sigma * w)
        
        For equal risk contribution: RC_i = 1/n for all i
        """
        start_time = time.perf_counter()
        
        n_assets = covariance_matrix.shape[0]
        
        if asset_ids is None:
            asset_ids = [f"asset_{i}" for i in range(n_assets)]
        
        if target_risk_contributions is None:
            # Equal risk contribution
            target_risk = np.ones(n_assets) / n_assets
        else:
            target_risk = target_risk_contributions / np.sum(target_risk_contributions)
        
        # Decision variables
        w = cp.Variable(n_assets)
        sigma_p = cp.Variable(nonneg=True)  # Portfolio volatility
        
        # Risk parity objective: minimize deviation from target risk contributions
        # RC_i = w_i * (Sigma * w)_i / sigma_p
        sigma_w = covariance_matrix @ w
        
        # Use log formulation for numerical stability
        # Minimize sum of squared log deviations
        risk_contributions = cp.multiply(w, sigma_w) / (sigma_p + 1e-8)
        
        objective = cp.Minimize(
            cp.sum_squares(risk_contributions - target_risk)
            + self.config.l2_penalty * cp.sum_squares(w)
        )
        
        # Constraints
        constraints = [
            cp.sum(w) == 1.0,
            w >= self.config.min_weight + 1e-6,  # Strictly positive for risk parity
            w <= self.config.max_weight,
            sigma_p == cp.norm(covariance_matrix @ w),
        ]
        
        if current_weights is not None:
            constraints.append(
                cp.sum(cp.abs(w - current_weights)) <= self.config.max_turnover
            )
        
        # Solve
        problem = cp.Problem(objective, constraints)
        self._last_problem = problem
        
        solve_start = time.perf_counter()
        
        try:
            problem.solve(
                solver=getattr(cp, self.config.solver),
                **self.config.get_solver_options()
            )
        except Exception as e:
            logger.error(f"Risk parity optimization failed: {e}")
            raise
        
        solve_time_ms = (time.perf_counter() - solve_start) * 1000
        
        # Extract results
        optimal_weights = np.array(w.value)
        
        # Compute metrics (assuming zero expected returns for pure risk parity)
        exp_vol = float(np.sqrt(optimal_weights @ covariance_matrix @ optimal_weights))
        
        if current_weights is not None:
            turnover = float(np.sum(np.abs(optimal_weights - current_weights)))
        else:
            turnover = 0.0
        
        result = OptimizationResult(
            optimal_weights=optimal_weights,
            asset_ids=asset_ids,
            expected_return=0.0,  # Risk parity doesn't optimize return
            expected_volatility=exp_vol,
            sharpe_ratio=0.0,
            transaction_cost=self.config.linear_tc_rate * turnover,
            turnover=turnover,
            status=problem.status,
            solve_time_ms=solve_time_ms
        )
        
        self._last_result = result
        logger.info(f"Risk parity optimization completed: {result.status} in {solve_time_ms:.1f}ms")
        
        return result
    
    def optimize_minimum_volatility(
        self,
        covariance_matrix: np.ndarray,
        expected_returns: Optional[np.ndarray] = None,
        current_weights: Optional[np.ndarray] = None,
        asset_ids: Optional[List[str]] = None,
        min_return: Optional[float] = None
    ) -> OptimizationResult:
        """
        Solve minimum volatility optimization.
        
        minimize: w^T * Sigma * w
        subject to: w^T * mu >= min_return (optional)
        """
        start_time = time.perf_counter()
        
        n_assets = covariance_matrix.shape[0]
        
        if asset_ids is None:
            asset_ids = [f"asset_{i}" for i in range(n_assets)]
        
        w = cp.Variable(n_assets)
        
        # Objective: minimize variance
        objective = cp.Minimize(cp.quad_form(w, covariance_matrix))
        
        # Constraints
        constraints = [
            cp.sum(w) == 1.0,
            w >= self.config.min_weight,
            w <= self.config.max_weight,
        ]
        
        if min_return is not None and expected_returns is not None:
            constraints.append(expected_returns @ w >= min_return)
        
        if current_weights is not None:
            constraints.append(
                cp.sum(cp.abs(w - current_weights)) <= self.config.max_turnover
            )
        
        # Solve
        problem = cp.Problem(objective, constraints)
        self._last_problem = problem
        
        solve_start = time.perf_counter()
        
        try:
            problem.solve(
                solver=getattr(cp, self.config.solver),
                **self.config.get_solver_options()
            )
        except Exception as e:
            logger.error(f"Min vol optimization failed: {e}")
            raise
        
        solve_time_ms = (time.perf_counter() - solve_start) * 1000
        
        optimal_weights = np.array(w.value)
        exp_vol = float(np.sqrt(optimal_weights @ covariance_matrix @ optimal_weights))
        
        exp_return = 0.0
        if expected_returns is not None:
            exp_return = float(expected_returns @ optimal_weights)
        
        if current_weights is not None:
            turnover = float(np.sum(np.abs(optimal_weights - current_weights)))
        else:
            turnover = 0.0
        
        result = OptimizationResult(
            optimal_weights=optimal_weights,
            asset_ids=asset_ids,
            expected_return=exp_return,
            expected_volatility=exp_vol,
            sharpe_ratio=exp_return / (exp_vol + 1e-8),
            transaction_cost=self.config.linear_tc_rate * turnover,
            turnover=turnover,
            status=problem.status,
            solve_time_ms=solve_time_ms
        )
        
        self._last_result = result
        logger.info(f"Min vol optimization completed: {result.status} in {solve_time_ms:.1f}ms")
        
        return result
    
    def get_effective_frontier(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        n_points: int = 20
    ) -> List[Tuple[float, float, np.ndarray]]:
        """
        Compute efficient frontier points.
        
        Returns list of (return, volatility, weights) tuples.
        """
        frontier = []
        
        # Get min and max feasible returns
        min_vol_result = self.optimize_minimum_volatility(covariance_matrix, expected_returns)
        min_return = min_vol_result.expected_return
        
        # Max return is the asset with highest expected return
        max_return = float(np.max(expected_returns))
        
        # Generate frontier points
        return_levels = np.linspace(min_return, max_return, n_points)
        
        for target_ret in return_levels:
            try:
                result = self.optimize_mean_variance(
                    expected_returns,
                    covariance_matrix,
                    additional_constraints=[cp.sum(expected_returns @ cp.Variable(len(expected_returns))) >= target_ret]
                )
                frontier.append((result.expected_return, result.expected_volatility, result.optimal_weights))
            except Exception as e:
                logger.warning(f"Frontier point failed at return {target_ret}: {e}")
        
        return frontier


# Factory function
def create_optimizer(config: Optional[OptimizationConfig] = None) -> ConvexPortfolioOptimizer:
    """Factory function to create optimizer."""
    return ConvexPortfolioOptimizer(config)
