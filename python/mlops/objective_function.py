"""
Objective Function for Bayesian Optimization.
Defines the evaluation function that runs rapid vectorized backtests
to evaluate parameter sets, with penalties for turnover, fees, and drawdown.

Integrates with Stage 37's vectorized backtesting engine for fast evaluation.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BacktestMetrics:
    """Complete metrics from a backtest run."""
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    total_return: float
    annualized_return: float
    win_rate: float
    profit_factor: float
    avg_trade_duration_ms: float
    turnover: float
    fee_drag: float
    calmar_ratio: float
    tail_ratio: float


class ObjectiveFunction:
    """
    Objective function evaluator for Bayesian optimization.
    Runs vectorized backtests and applies penalty calculations.
    """
    
    def __init__(
        self,
        max_drawdown_limit: float = 0.15,
        min_sharpe_threshold: float = 0.5,
        fee_rate_bps: float = 5.0,
        slippage_model: str = "linear"
    ):
        self.max_drawdown_limit = max_drawdown_limit
        self.min_sharpe_threshold = min_sharpe_threshold
        self.fee_rate_bps = fee_rate_bps
        self.slippage_model = slippage_model
        
        # Penalty weights
        self.drawdown_penalty_weight = 3.0
        self.turnover_penalty_weight = 0.5
        self.fee_penalty_weight = 1.0
        self.sharpe_bonus_weight = 2.0
    
    def evaluate(
        self,
        params: Dict[str, Any],
        returns: np.ndarray,
        prices: np.ndarray,
        signals: np.ndarray,
        volumes: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Evaluate a parameter set using vectorized backtest results.
        
        Args:
            params: Parameter dictionary from optimizer
            returns: Array of strategy returns
            prices: Price series
            signals: Trading signals (-1, 0, 1)
            volumes: Optional volume series for turnover calculation
            
        Returns:
            Dictionary with metrics for optimization
        """
        # Calculate base metrics
        metrics = self._calculate_metrics(returns, prices, signals, volumes)
        
        # Apply hard constraints (disqualify if breached)
        if metrics.max_drawdown > self.max_drawdown_limit:
            logger.debug(f"Drawdown limit breached: {metrics.max_drawdown:.4f}")
            return {
                'sharpe': -np.inf,
                'drawdown': metrics.max_drawdown,
                'return': metrics.total_return,
                'turnover': metrics.turnover,
                'objective_score': -np.inf
            }
        
        if metrics.sharpe_ratio < self.min_sharpe_threshold:
            logger.debug(f"Sharpe below threshold: {metrics.sharpe_ratio:.4f}")
            return {
                'sharpe': metrics.sharpe_ratio,
                'drawdown': metrics.max_drawdown,
                'return': metrics.total_return,
                'turnover': metrics.turnover,
                'objective_score': -10.0  # Heavy penalty but not disqualification
            }
        
        # Calculate penalized objective score
        objective_score = self._calculate_objective_score(metrics)
        
        return {
            'sharpe': metrics.sharpe_ratio,
            'drawdown': metrics.max_drawdown,
            'return': metrics.total_return,
            'turnover': metrics.turnover,
            'fee_drag': metrics.fee_drag,
            'objective_score': objective_score
        }
    
    def _calculate_metrics(
        self,
        returns: np.ndarray,
        prices: np.ndarray,
        signals: np.ndarray,
        volumes: Optional[np.ndarray]
    ) -> BacktestMetrics:
        """Calculate comprehensive backtest metrics."""
        n = len(returns)
        if n == 0:
            return self._empty_metrics()
        
        # Basic statistics
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Sharpe ratio (annualized, assuming daily returns)
        if std_return > 1e-10:
            sharpe_ratio = mean_return / std_return * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_std = np.std(downside_returns)
            if downside_std > 1e-10:
                sortino_ratio = mean_return / downside_std * np.sqrt(252)
            else:
                sortino_ratio = sharpe_ratio
        else:
            sortino_ratio = sharpe_ratio
        
        # Maximum drawdown
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        max_drawdown = abs(np.min(drawdowns))
        
        # Total and annualized return
        total_return = cumulative[-1] - 1
        years = n / 252
        if years > 0:
            annualized_return = (1 + total_return) ** (1 / years) - 1
        else:
            annualized_return = 0.0
        
        # Win rate
        winning_trades = returns[returns > 0]
        losing_trades = returns[returns < 0]
        win_rate = len(winning_trades) / max(len(returns), 1)
        
        # Profit factor
        gross_profit = np.sum(winning_trades) if len(winning_trades) > 0 else 0
        gross_loss = abs(np.sum(losing_trades)) if len(losing_trades) > 0 else 1e-10
        profit_factor = gross_profit / max(gross_loss, 1e-10)
        
        # Turnover estimation
        if volumes is not None and len(volumes) == len(signals):
            position_changes = np.abs(np.diff(np.concatenate([[0], signals])))
            turnover = np.mean(position_changes * volumes[:-1]) / np.mean(volumes)
        else:
            turnover = np.mean(np.abs(np.diff(np.concatenate([[0], signals]))))
        
        # Fee drag estimation
        trades = np.sum(np.abs(np.diff(np.concatenate([[0], signals]))))
        fee_drag = trades * self.fee_rate_bps / 10000
        
        # Calmar ratio
        if max_drawdown > 1e-10:
            calmar_ratio = annualized_return / max_drawdown
        else:
            calmar_ratio = 0.0
        
        # Tail ratio (95th percentile / 5th percentile of returns)
        if len(returns) > 20:
            p95 = np.percentile(returns, 95)
            p05 = abs(np.percentile(returns, 5))
            tail_ratio = p95 / max(p05, 1e-10)
        else:
            tail_ratio = 1.0
        
        # Average trade duration (placeholder - would need timestamps)
        avg_trade_duration_ms = 0.0
        
        return BacktestMetrics(
            sharpe_ratio=float(sharpe_ratio),
            sortino_ratio=float(sortino_ratio),
            max_drawdown=float(max_drawdown),
            total_return=float(total_return),
            annualized_return=float(annualized_return),
            win_rate=float(win_rate),
            profit_factor=float(profit_factor),
            avg_trade_duration_ms=avg_trade_duration_ms,
            turnover=float(turnover),
            fee_drag=float(fee_drag),
            calmar_ratio=float(calmar_ratio),
            tail_ratio=float(tail_ratio)
        )
    
    def _calculate_objective_score(self, metrics: BacktestMetrics) -> float:
        """
        Calculate penalized objective score for optimization.
        
        Formula:
        score = sharpe_bonus - drawdown_penalty - turnover_penalty - fee_penalty
        """
        # Sharpe bonus (encourages high risk-adjusted returns)
        sharpe_bonus = self.sharpe_bonus_weight * max(0, metrics.sharpe_ratio)
        
        # Drawdown penalty (exponential penalty for large drawdowns)
        drawdown_ratio = metrics.max_drawdown / self.max_drawdown_limit
        drawdown_penalty = self.drawdown_penalty_weight * (drawdown_ratio ** 2)
        
        # Turnover penalty (linear penalty for excessive trading)
        turnover_penalty = self.turnover_penalty_weight * metrics.turnover
        
        # Fee penalty
        fee_penalty = self.fee_penalty_weight * metrics.fee_drag
        
        # Combined score
        objective_score = sharpe_bonus - drawdown_penalty - turnover_penalty - fee_penalty
        
        return float(objective_score)
    
    def _empty_metrics(self) -> BacktestMetrics:
        """Return empty metrics for edge cases."""
        return BacktestMetrics(
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=0.0,
            total_return=0.0,
            annualized_return=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            avg_trade_duration_ms=0.0,
            turnover=0.0,
            fee_drag=0.0,
            calmar_ratio=0.0,
            tail_ratio=0.0
        )
    
    def run_vectorized_backtest(
        self,
        params: Dict[str, Any],
        prices: np.ndarray,
        features: Optional[Dict[str, np.ndarray]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run a simplified vectorized backtest for quick evaluation.
        This is a placeholder that should integrate with Stage 37's vectorbt_engine.
        
        Args:
            params: Strategy parameters
            prices: Price series
            features: Optional feature dictionary
            
        Returns:
            (returns, signals) tuple
        """
        n = len(prices)
        
        # Extract common parameters
        lookback = params.get('lookback', 20)
        entry_threshold = params.get('entry_threshold', 0.5)
        exit_threshold = params.get('exit_threshold', 0.3)
        stop_loss = params.get('stop_loss', 0.02)
        take_profit = params.get('take_profit', 0.04)
        
        # Generate signals based on momentum (simplified)
        if features is not None and 'momentum' in features:
            signal_raw = features['momentum']
        else:
            # Calculate momentum from prices
            returns = np.diff(prices) / prices[:-1]
            returns = np.concatenate([[0], returns])
            signal_raw = np.convolve(returns, np.ones(lookback)/lookback, mode='same')
        
        # Normalize signal
        signal_std = np.std(signal_raw)
        if signal_std > 1e-10:
            signal_normalized = signal_raw / signal_std
        else:
            signal_normalized = signal_raw
        
        # Generate positions
        signals = np.zeros(n)
        position = 0
        
        for i in range(1, n):
            if position == 0:
                if signal_normalized[i] > entry_threshold:
                    position = 1
                elif signal_normalized[i] < -entry_threshold:
                    position = -1
            else:
                # Check exit conditions
                if abs(signal_normalized[i]) < exit_threshold:
                    position = 0
                
                # Check stop loss / take profit (simplified)
                if position != 0:
                    entry_idx = max(0, i - lookback)
                    price_change = (prices[i] - prices[entry_idx]) / prices[entry_idx]
                    
                    if position == 1:
                        if price_change <= -stop_loss or price_change >= take_profit:
                            position = 0
                    else:
                        if price_change >= stop_loss or price_change <= -take_profit:
                            position = 0
            
            signals[i] = position
        
        # Calculate returns
        price_returns = np.diff(prices) / prices[:-1]
        price_returns = np.concatenate([[0], price_returns])
        
        # Strategy returns (shifted signals to avoid lookahead)
        strategy_returns = np.roll(signals, 1) * price_returns
        strategy_returns[0] = 0
        
        return strategy_returns, signals


# Convenience function for optimizer integration
def create_objective_function(
    prices: np.ndarray,
    features: Optional[Dict[str, np.ndarray]] = None,
    **kwargs
):
    """
    Create a closure-based objective function for Bayesian optimization.
    
    Args:
        prices: Historical price series
        features: Optional feature dictionary
        **kwargs: Additional arguments for ObjectiveFunction
        
    Returns:
        Callable objective function
    """
    evaluator = ObjectiveFunction(**kwargs)
    
    def objective(params: Dict[str, Any]) -> Dict[str, float]:
        returns, signals = evaluator.run_vectorized_backtest(params, prices, features)
        return evaluator.evaluate(params, returns, prices, signals)
    
    return objective


if __name__ == "__main__":
    # Demo usage
    np.random.seed(42)
    
    # Generate synthetic price data
    n_days = 500
    returns = np.random.randn(n_days) * 0.02
    prices = 100 * np.cumprod(1 + returns)
    
    # Create objective function
    obj_fn = create_objective_function(
        prices=prices,
        max_drawdown_limit=0.15,
        min_sharpe_threshold=0.5
    )
    
    # Test with sample parameters
    test_params = {
        'lookback': 20,
        'entry_threshold': 0.5,
        'exit_threshold': 0.3,
        'stop_loss': 0.02,
        'take_profit': 0.04
    }
    
    result = obj_fn(test_params)
    print(f"Objective function result: {result}")
    
    # Test with different parameters
    test_params2 = {
        'lookback': 50,
        'entry_threshold': 0.8,
        'exit_threshold': 0.4,
        'stop_loss': 0.03,
        'take_profit': 0.06
    }
    
    result2 = obj_fn(test_params2)
    print(f"Objective function result 2: {result2}")
