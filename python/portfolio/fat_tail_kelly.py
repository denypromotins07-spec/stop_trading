"""
Chapter 3: Dynamic Capital Allocation & Risk Budgeting
fat_tail_kelly.py - Fractional Kelly Criterion adjusted for fat-tailed crypto distributions
"""

import numpy as np
from numba import njit, prange
from typing import Tuple, Optional, List
from dataclasses import dataclass


@njit(cache=True, nogil=True)
def estimate_tail_index_hill(
    returns: np.ndarray,
    n_top: int = 100
) -> float:
    """
    Estimate tail index using Hill estimator.
    Lower values indicate fatter tails.
    
    Args:
        returns: Return series
        n_top: Number of top order statistics to use
    
    Returns:
        Tail index (alpha). Values < 4 indicate significant fat tails.
    """
    n = len(returns)
    if n < n_top:
        n_top = n // 2
    
    # Sort by absolute value (descending)
    abs_returns = np.abs(returns)
    sorted_indices = np.argsort(abs_returns)[::-1]
    
    # Get top n_top returns
    top_returns = abs_returns[sorted_indices[:n_top]]
    
    # Hill estimator
    tail_sum = 0.0
    for i in range(n_top - 1):
        if top_returns[i + 1] > 0:
            tail_sum += np.log(top_returns[i] / top_returns[i + 1])
    
    if tail_sum == 0:
        return 3.0  # Default to moderate fat tail
    
    alpha = (n_top - 1) / tail_sum
    
    return alpha


@njit(cache=True, nogil=True)
def calculate_skewness(returns: np.ndarray) -> float:
    """Calculate skewness of returns."""
    n = len(returns)
    if n < 3:
        return 0.0
    
    mean = 0.0
    for r in returns:
        mean += r
    mean /= n
    
    m2 = 0.0
    m3 = 0.0
    
    for r in returns:
        diff = r - mean
        m2 += diff * diff
        m3 += diff * diff * diff
    
    m2 /= n
    m3 /= n
    
    std = np.sqrt(m2)
    if std == 0:
        return 0.0
    
    return m3 / (std * std * std)


@njit(cache=True, nogil=True)
def calculate_excess_kurtosis(returns: np.ndarray) -> float:
    """Calculate excess kurtosis of returns."""
    n = len(returns)
    if n < 4:
        return 0.0
    
    mean = 0.0
    for r in returns:
        mean += r
    mean /= n
    
    m2 = 0.0
    m4 = 0.0
    
    for r in returns:
        diff = r - mean
        m2 += diff * diff
        m4 += diff * diff * diff * diff
    
    m2 /= n
    m4 /= n
    
    std_sq = m2
    if std_sq == 0:
        return 0.0
    
    # Kurtosis - 3 (excess)
    kurtosis = m4 / (std_sq * std_sq)
    return kurtosis - 3.0


@njit(cache=True, nogil=True)
def standard_kelly_fraction(
    win_rate: float,
    avg_win: float,
    avg_loss: float
) -> float:
    """
    Calculate standard Kelly fraction.
    
    Args:
        win_rate: Probability of winning (0 to 1)
        avg_win: Average win amount (positive)
        avg_loss: Average loss amount (positive)
    
    Returns:
        Kelly fraction (bet size as fraction of capital)
    """
    if avg_loss == 0 or win_rate <= 0 or win_rate >= 1:
        return 0.0
    
    b = avg_win / avg_loss  # Odds received
    p = win_rate
    q = 1.0 - p
    
    # Kelly formula: f* = (bp - q) / b
    kelly = (b * p - q) / b
    
    return max(0.0, kelly)


@njit(cache=True, nogil=True)
def fat_tail_kelly_adjustment(
    base_kelly: float,
    tail_index: float,
    excess_kurtosis: float,
    skewness: float,
    confidence_level: float = 0.95
) -> float:
    """
    Adjust Kelly fraction for fat-tailed distributions.
    
    The adjustment reduces position size based on:
    1. Tail index (lower = fatter tails = more reduction)
    2. Excess kurtosis (higher = fatter tails = more reduction)
    3. Skewness (negative skew = more reduction)
    
    Args:
        base_kelly: Standard Kelly fraction
        tail_index: Estimated tail index (Hill estimator)
        excess_kurtosis: Excess kurtosis of returns
        skewness: Skewness of returns
        confidence_level: Confidence level for adjustment
    
    Returns:
        Adjusted Kelly fraction
    """
    if base_kelly <= 0:
        return 0.0
    
    # Tail risk adjustment
    # If tail_index < 4, variance may not exist; if < 2, mean may not exist
    if tail_index >= 4:
        tail_factor = 1.0
    elif tail_index >= 2:
        # Quadratic decay between 4 and 2
        tail_factor = (tail_index - 1.5) / 2.5
    else:
        # Severe reduction for very fat tails
        tail_factor = tail_index / 3.0
    
    tail_factor = max(0.1, min(1.0, tail_factor))
    
    # Kurtosis adjustment
    # Higher kurtosis = more extreme events = reduce size
    if excess_kurtosis <= 0:
        kurtosis_factor = 1.0
    else:
        # Reduce by up to 50% for high kurtosis
        kurtosis_factor = 1.0 / (1.0 + excess_kurtosis * 0.1)
    
    kurtosis_factor = max(0.5, min(1.0, kurtosis_factor))
    
    # Skewness adjustment
    # Negative skew = more downside risk = reduce size
    if skewness >= 0:
        skew_factor = 1.0
    else:
        # Reduce by up to 30% for negative skew
        skew_factor = 1.0 + skewness * 0.1
    
    skew_factor = max(0.7, min(1.0, skew_factor))
    
    # Combined adjustment
    combined_factor = tail_factor * kurtosis_factor * skew_factor
    
    # Apply confidence level scaling
    # Higher confidence = more conservative
    confidence_multiplier = 1.0 - (1.0 - confidence_level) * 0.5
    
    adjusted_kelly = base_kelly * combined_factor * confidence_multiplier
    
    return max(0.0, adjusted_kelly)


@njit(cache=True, nogil=True)
def fractional_kelly(
    base_kelly: float,
    fraction: float = 0.5
) -> float:
    """
    Apply fractional Kelly betting for reduced volatility.
    
    Common fractions:
    - 0.5 (Half Kelly): ~75% of optimal growth with half the volatility
    - 0.25 (Quarter Kelly): More conservative
    
    Args:
        base_kelly: Base Kelly fraction
        fraction: Fraction to apply (0 to 1)
    
    Returns:
        Fractional Kelly bet size
    """
    return base_kelly * fraction


@dataclass
class KellyResult:
    """Container for Kelly calculation results"""
    standard_kelly: float
    fat_tail_adjusted: float
    fractional_kelly_half: float
    fractional_kelly_quarter: float
    tail_index: float
    excess_kurtosis: float
    skewness: float
    recommended_fraction: float
    recommended_bet_size: float


class FatTailKellyCalculator:
    """
    Kelly Criterion calculator adjusted for fat-tailed crypto distributions.
    Incorporates SOUL.md historical win-rates and tail risk metrics.
    """
    
    def __init__(
        self,
        default_fraction: float = 0.5,
        min_tail_index: float = 2.0,
        confidence_level: float = 0.95
    ):
        self.default_fraction = default_fraction
        self.min_tail_index = min_tail_index
        self.confidence_level = confidence_level
        
        # Historical metrics cache
        self._last_returns = None
        self._last_metrics = None
    
    def calculate(
        self,
        returns: np.ndarray,
        win_rate: Optional[float] = None,
        avg_win: Optional[float] = None,
        avg_loss: Optional[float] = None
    ) -> KellyResult:
        """
        Calculate fat-tail adjusted Kelly fraction.
        
        Args:
            returns: Historical return series
            win_rate: Optional pre-calculated win rate
            avg_win: Optional pre-calculated average win
            avg_loss: Optional pre-calculated average loss
        
        Returns:
            KellyResult with all calculations
        """
        n = len(returns)
        
        # Calculate tail metrics
        tail_index = estimate_tail_index_hill(returns, n_top=min(100, n // 5))
        tail_index = max(self.min_tail_index, tail_index)
        
        skewness = calculate_skewness(returns)
        excess_kurtosis = calculate_excess_kurtosis(returns)
        
        # Calculate win/loss statistics if not provided
        if win_rate is None or avg_win is None or avg_loss is None:
            win_rate, avg_win, avg_loss = self._calculate_win_loss_stats(returns)
        
        # Standard Kelly
        standard_kelly = standard_kelly_fraction(win_rate, avg_win, avg_loss)
        
        # Fat tail adjustment
        fat_tail_kelly = fat_tail_kelly_adjustment(
            standard_kelly,
            tail_index,
            excess_kurtosis,
            skewness,
            self.confidence_level
        )
        
        # Fractional Kelly options
        half_kelly = fractional_kelly(fat_tail_kelly, 0.5)
        quarter_kelly = fractional_kelly(fat_tail_kelly, 0.25)
        
        # Determine recommended fraction based on tail risk
        if tail_index < 2.5:
            recommended_fraction = 0.25  # Very fat tails
        elif tail_index < 3.5:
            recommended_fraction = 0.5   # Moderate fat tails
        else:
            recommended_fraction = self.default_fraction
        
        recommended_bet = fractional_kelly(fat_tail_kelly, recommended_fraction)
        
        # Cache results
        self._last_returns = returns
        self._last_metrics = {
            'tail_index': tail_index,
            'skewness': skewness,
            'excess_kurtosis': excess_kurtosis
        }
        
        return KellyResult(
            standard_kelly=standard_kelly,
            fat_tail_adjusted=fat_tail_kelly,
            fractional_kelly_half=half_kelly,
            fractional_kelly_quarter=quarter_kelly,
            tail_index=tail_index,
            excess_kurtosis=excess_kurtosis,
            skewness=skewness,
            recommended_fraction=recommended_fraction,
            recommended_bet_size=recommended_bet
        )
    
    @staticmethod
    @njit(cache=True, nogil=True)
    def _calculate_win_loss_stats(
        returns: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Calculate win rate, average win, and average loss from returns.
        
        Returns:
            Tuple of (win_rate, avg_win, avg_loss)
        """
        n = len(returns)
        if n == 0:
            return 0.0, 0.0, 0.0
        
        wins = 0
        losses = 0
        total_win = 0.0
        total_loss = 0.0
        
        for r in returns:
            if r > 0:
                wins += 1
                total_win += r
            elif r < 0:
                losses += 1
                total_loss -= r  # Make positive
        
        win_rate = wins / n if n > 0 else 0.0
        avg_win = total_win / wins if wins > 0 else 0.0
        avg_loss = total_loss / losses if losses > 0 else 0.0
        
        return win_rate, avg_win, avg_loss
    
    def get_tail_risk_assessment(self) -> str:
        """Get qualitative assessment of tail risk based on last calculation."""
        if self._last_metrics is None:
            return "No data available"
        
        tail_index = self._last_metrics['tail_index']
        kurtosis = self._last_metrics['excess_kurtosis']
        
        if tail_index < 2.0:
            risk_level = "EXTREME"
        elif tail_index < 3.0:
            risk_level = "HIGH"
        elif tail_index < 4.0:
            risk_level = "MODERATE"
        else:
            risk_level = "LOW"
        
        return f"Tail Risk: {risk_level} (α={tail_index:.2f}, κ={kurtosis:.2f})"


# Module convenience functions
def create_kelly_calculator(
    default_fraction: float = 0.5,
    confidence_level: float = 0.95
) -> FatTailKellyCalculator:
    """Factory function to create Kelly calculator."""
    return FatTailKellyCalculator(default_fraction, confidence_level)


def quick_fat_tail_kelly(
    returns: np.ndarray,
    fraction: float = 0.5
) -> float:
    """
    Quick fat-tail adjusted Kelly calculation.
    
    Returns:
        Recommended bet size as fraction of capital
    """
    calc = create_kelly_calculator(fraction)
    result = calc.calculate(returns)
    return result.recommended_bet_size


def soul_md_kelly(
    historical_wins: int,
    historical_losses: int,
    total_profit: float,
    total_loss: float,
    tail_adjustment: float = 0.7
) -> float:
    """
    Kelly calculation specifically for SOUL.md historical performance.
    
    Args:
        historical_wins: Number of winning trades
        historical_losses: Number of losing trades
        total_profit: Sum of all profits
        total_loss: Sum of all losses (absolute value)
        tail_adjustment: Reduction factor for fat tails (default 0.7)
    
    Returns:
        Adjusted Kelly fraction
    """
    total_trades = historical_wins + historical_losses
    
    if total_trades == 0:
        return 0.0
    
    win_rate = historical_wins / total_trades
    avg_win = total_profit / historical_wins if historical_wins > 0 else 0.0
    avg_loss = total_loss / historical_losses if historical_losses > 0 else 0.0
    
    standard = standard_kelly_fraction(win_rate, avg_win, avg_loss)
    
    return standard * tail_adjustment
