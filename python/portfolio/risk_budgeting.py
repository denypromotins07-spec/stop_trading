"""
Chapter 3: Dynamic Capital Allocation & Risk Budgeting
risk_budgeting.py - Risk Budgeting optimization using cyclical coordinate descent
"""

import numpy as np
from numba import njit, prange
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass


@njit(cache=True, nogil=True)
def calculate_marginal_risk_contribution(
    weights: np.ndarray,
    cov_matrix: np.ndarray,
    asset_idx: int
) -> float:
    """
    Calculate marginal risk contribution of a single asset.
    MRC_i = (Σ * w)_i / σ_p
    
    Args:
        weights: Portfolio weights
        cov_matrix: Covariance matrix
        asset_idx: Index of asset to calculate
    
    Returns:
        Marginal risk contribution
    """
    n_assets = len(weights)
    
    # Calculate portfolio variance: w' * Σ * w
    port_var = 0.0
    for i in range(n_assets):
        for j in range(n_assets):
            port_var += weights[i] * cov_matrix[i, j] * weights[j]
    
    port_vol = np.sqrt(port_var) if port_var > 0 else 1e-10
    
    # Calculate (Σ * w)_i
    sigma_w_i = 0.0
    for j in range(n_assets):
        sigma_w_i += cov_matrix[asset_idx, j] * weights[j]
    
    # MRC = (Σ * w)_i / σ_p
    return sigma_w_i / port_vol


@njit(cache=True, nogil=True)
def calculate_risk_contribution(
    weights: np.ndarray,
    cov_matrix: np.ndarray
) -> np.ndarray:
    """
    Calculate risk contribution of each asset.
    RC_i = w_i * MRC_i
    
    Args:
        weights: Portfolio weights
        cov_matrix: Covariance matrix
    
    Returns:
        Array of risk contributions
    """
    n_assets = len(weights)
    risk_contribs = np.empty(n_assets, dtype=np.float64)
    
    # Calculate portfolio volatility
    port_var = 0.0
    for i in range(n_assets):
        for j in range(n_assets):
            port_var += weights[i] * cov_matrix[i, j] * weights[j]
    
    port_vol = np.sqrt(port_var) if port_var > 0 else 1e-10
    
    # Calculate risk contributions
    for i in range(n_assets):
        # MRC_i
        sigma_w_i = 0.0
        for j in range(n_assets):
            sigma_w_i += cov_matrix[i, j] * weights[j]
        
        mrc_i = sigma_w_i / port_vol
        
        # RC_i = w_i * MRC_i
        risk_contribs[i] = weights[i] * mrc_i
    
    return risk_contribs


@njit(cache=True, nogil=True)
def coordinate_descent_step(
    weights: np.ndarray,
    cov_matrix: np.ndarray,
    risk_budgets: np.ndarray,
    asset_idx: int,
    learning_rate: float = 0.1
) -> float:
    """
    Perform one coordinate descent step for a single asset.
    
    Args:
        weights: Current portfolio weights
        cov_matrix: Covariance matrix
        risk_budgets: Target risk budgets (sum to 1)
        asset_idx: Asset to update
        learning_rate: Step size
    
    Returns:
        Weight change magnitude
    """
    n_assets = len(weights)
    
    # Calculate current risk contribution
    risk_contribs = calculate_risk_contribution(weights, cov_matrix)
    
    # Total risk
    total_risk = 0.0
    for rc in risk_contribs:
        total_risk += rc
    if total_risk == 0:
        total_risk = 1e-10
    
    # Current risk budget ratio
    current_ratio = risk_contribs[asset_idx] / total_risk
    target_ratio = risk_budgets[asset_idx]
    
    # Gradient: direction to reduce difference
    gradient = current_ratio - target_ratio
    
    # Update weight
    old_weight = weights[asset_idx]
    weights[asset_idx] -= learning_rate * gradient * old_weight
    
    # Ensure non-negative
    if weights[asset_idx] < 0:
        weights[asset_idx] = 0.0
    
    return abs(weights[asset_idx] - old_weight)


@njit(cache=True, nogil=True)
def normalize_weights(weights: np.ndarray) -> np.ndarray:
    """Normalize weights to sum to 1."""
    total = 0.0
    for w in weights:
        total += w
    
    if total == 0:
        n = len(weights)
        return np.ones(n, dtype=np.float64) / n
    
    result = np.empty(len(weights), dtype=np.float64)
    for i in range(len(weights)):
        result[i] = weights[i] / total
    
    return result


@njit(cache=True, nogil=True)
def risk_budgeting_optimize(
    cov_matrix: np.ndarray,
    risk_budgets: np.ndarray,
    initial_weights: Optional[np.ndarray] = None,
    max_iterations: int = 1000,
    tolerance: float = 1e-8,
    learning_rate: float = 0.1
) -> Tuple[np.ndarray, int, float]:
    """
    Optimize portfolio weights using cyclical coordinate descent.
    
    Args:
        cov_matrix: Asset covariance matrix (n x n)
        risk_budgets: Target risk budgets (must sum to 1)
        initial_weights: Starting weights (optional)
        max_iterations: Maximum iterations
        tolerance: Convergence tolerance
        learning_rate: Learning rate for updates
    
    Returns:
        Tuple of (optimal_weights, iterations, final_objective)
    """
    n_assets = cov_matrix.shape[0]
    
    # Initialize weights
    if initial_weights is not None:
        weights = initial_weights.copy()
    else:
        weights = np.ones(n_assets, dtype=np.float64) / n_assets
    
    weights = normalize_weights(weights)
    
    # Normalize risk budgets
    rb_sum = 0.0
    for rb in risk_budgets:
        rb_sum += rb
    
    if rb_sum > 0:
        target_rb = np.empty(n_assets, dtype=np.float64)
        for i in range(n_assets):
            target_rb[i] = risk_budgets[i] / rb_sum
    else:
        target_rb = np.ones(n_assets, dtype=np.float64) / n_assets
    
    # Cyclical coordinate descent
    for iteration in range(max_iterations):
        max_change = 0.0
        
        # Cycle through all assets
        for i in range(n_assets):
            change = coordinate_descent_step(
                weights, cov_matrix, target_rb, i, learning_rate
            )
            max_change = max(max_change, change)
        
        # Re-normalize
        weights = normalize_weights(weights)
        
        # Check convergence
        if max_change < tolerance:
            break
    
    # Calculate final objective (sum of squared differences from target)
    risk_contribs = calculate_risk_contribution(weights, cov_matrix)
    total_risk = 0.0
    for rc in risk_contribs:
        total_risk += rc
    
    objective = 0.0
    for i in range(n_assets):
        actual_ratio = risk_contribs[i] / total_risk if total_risk > 0 else 0
        diff = actual_ratio - target_rb[i]
        objective += diff * diff
    
    return weights, iteration + 1, objective


@njit(cache=True, nogil=True)
def calculate_portfolio_volatility(
    weights: np.ndarray,
    cov_matrix: np.ndarray
) -> float:
    """Calculate portfolio volatility given weights and covariance."""
    n = len(weights)
    var = 0.0
    
    for i in range(n):
        for j in range(n):
            var += weights[i] * cov_matrix[i, j] * weights[j]
    
    return np.sqrt(var) if var > 0 else 0.0


@njit(cache=True, nogil=True)
def equal_risk_contribution(
    cov_matrix: np.ndarray,
    initial_weights: Optional[np.ndarray] = None,
    max_iterations: int = 1000,
    tolerance: float = 1e-8
) -> Tuple[np.ndarray, int]:
    """
    Special case: Equal Risk Contribution (ERC) portfolio.
    Each asset contributes equally to total risk.
    
    Returns:
        Tuple of (weights, iterations)
    """
    n_assets = cov_matrix.shape[0]
    equal_budgets = np.ones(n_assets, dtype=np.float64) / n_assets
    
    weights, iters, _ = risk_budgeting_optimize(
        cov_matrix, equal_budgets, initial_weights, max_iterations, tolerance
    )
    
    return weights, iters


@dataclass
class RiskBudgetingResult:
    """Container for risk budgeting optimization results"""
    weights: np.ndarray
    risk_contributions: np.ndarray
    portfolio_volatility: float
    iterations: int
    convergence_achieved: bool
    objective_value: float


class RiskBudgetingEngine:
    """
    Engine for dynamic risk budgeting and capital allocation.
    Ensures no single strategy or asset dominates portfolio variance.
    """
    
    def __init__(
        self,
        max_iterations: int = 1000,
        tolerance: float = 1e-8,
        min_weight: float = 0.0,
        max_weight: float = 1.0
    ):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.min_weight = min_weight
        self.max_weight = max_weight
        
        # Cached results
        self._last_weights = None
        self._last_cov = None
        self._last_budgets = None
    
    def optimize(
        self,
        cov_matrix: np.ndarray,
        risk_budgets: np.ndarray,
        initial_weights: Optional[np.ndarray] = None
    ) -> RiskBudgetingResult:
        """
        Perform risk budgeting optimization.
        
        Args:
            cov_matrix: Covariance matrix
            risk_budgets: Target risk budgets
            initial_weights: Optional starting weights
        
        Returns:
            RiskBudgetingResult
        """
        n_assets = cov_matrix.shape[0]
        
        # Validate inputs
        assert cov_matrix.shape == (n_assets, n_assets), "Covariance matrix must be square"
        assert len(risk_budgets) == n_assets, "Risk budgets must match number of assets"
        
        # Run optimization
        weights, iterations, objective = risk_budgeting_optimize(
            cov_matrix,
            risk_budgets,
            initial_weights,
            self.max_iterations,
            self.tolerance
        )
        
        # Apply weight constraints
        weights = np.clip(weights, self.min_weight, self.max_weight)
        weights = normalize_weights(weights)
        
        # Calculate risk contributions
        risk_contribs = calculate_risk_contribution(weights, cov_matrix)
        
        # Calculate portfolio volatility
        port_vol = calculate_portfolio_volatility(weights, cov_matrix)
        
        # Check convergence
        converged = objective < self.tolerance * 10
        
        # Cache results
        self._last_weights = weights
        self._last_cov = cov_matrix
        self._last_budgets = risk_budgets
        
        return RiskBudgetingResult(
            weights=weights,
            risk_contributions=risk_contribs,
            portfolio_volatility=port_vol,
            iterations=iterations,
            convergence_achieved=converged,
            objective_value=objective
        )
    
    def equal_risk_contribution(
        self,
        cov_matrix: np.ndarray,
        initial_weights: Optional[np.ndarray] = None
    ) -> RiskBudgetingResult:
        """Calculate ERC (Equal Risk Contribution) portfolio."""
        n_assets = cov_matrix.shape[0]
        equal_budgets = np.ones(n_assets, dtype=np.float64) / n_assets
        
        return self.optimize(cov_matrix, equal_budgets, initial_weights)
    
    def get_implied_risk_budgets(
        self,
        weights: np.ndarray,
        cov_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Given weights, calculate implied risk budgets.
        Useful for analyzing existing portfolios.
        """
        risk_contribs = calculate_risk_contribution(weights, cov_matrix)
        
        total_risk = 0.0
        for rc in risk_contribs:
            total_risk += rc
        
        if total_risk > 0:
            return risk_contribs / total_risk
        
        return np.zeros(len(weights), dtype=np.float64)
    
    def scale_to_volatility_target(
        self,
        weights: np.ndarray,
        cov_matrix: np.ndarray,
        vol_target: float
    ) -> np.ndarray:
        """
        Scale portfolio weights to achieve target volatility.
        
        Args:
            weights: Current weights
            cov_matrix: Covariance matrix
            vol_target: Target volatility
        
        Returns:
            Scaled weights
        """
        current_vol = calculate_portfolio_volatility(weights, cov_matrix)
        
        if current_vol == 0:
            return weights
        
        scale_factor = vol_target / current_vol
        
        return weights * scale_factor


# Module convenience functions
def create_risk_budgeting_engine(
    max_iterations: int = 1000,
    tolerance: float = 1e-8
) -> RiskBudgetingEngine:
    """Factory function to create risk budgeting engine."""
    return RiskBudgetingEngine(max_iterations, tolerance)


def quick_erc(cov_matrix: np.ndarray) -> np.ndarray:
    """Quick ERC calculation with default parameters."""
    weights, _ = equal_risk_contribution(cov_matrix)
    return weights
