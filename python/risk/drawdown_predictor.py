"""
Lightweight ONNX-compiled sequence model for drawdown probability prediction.
Triggers automated deleveraging when predicted drawdown probability exceeds safety limits.
Memory-bounded implementation for 3GB RAM limit - no Pandas in hot-path.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import time


class DrawdownPredictor:
    """
    Sequence model for predicting maximum drawdown breach probability.
    Uses lightweight numpy-based computations suitable for real-time inference.
    In production, this would load a pre-trained ONNX model.
    """
    
    def __init__(self, lookback_window: int = 60, 
                 threshold_levels: List[float] = None):
        """
        Initialize the drawdown predictor.
        
        Args:
            lookback_window: Number of periods to look back for features
            threshold_levels: Drawdown thresholds to monitor (e.g., [0.05, 0.10, 0.15])
        """
        self.lookback_window = lookback_window
        self.threshold_levels = threshold_levels or [0.05, 0.10, 0.15, 0.20]
        
        # Model parameters (would be loaded from ONNX in production)
        self._weights: Optional[np.ndarray] = None
        self._bias: float = 0.0
        self._feature_dim = 20  # Number of input features
        
        # State tracking
        self._running_max: Dict[str, float] = {}
        self._current_drawdown: Dict[str, float] = {}
        self._drawdown_history: Dict[str, List[float]] = {}
        
        # Initialize with default parameters
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize model with default parameters (placeholder for ONNX loading)."""
        # In production, load from ONNX:
        # import onnxruntime as ort
        # session = ort.InferenceSession("drawdown_model.onnx")
        # self._session = session
        
        # Placeholder weights for demonstration
        np.random.seed(42)
        self._weights = np.random.randn(self._feature_dim) * 0.1
        self._bias = -2.0  # Negative bias to reduce false positives
    
    def _extract_sequence_features(self, returns: np.ndarray, 
                                    cumulative_pnl: np.ndarray,
                                    volatility: np.ndarray,
                                    volumes: np.ndarray) -> np.ndarray:
        """
        Extract features from market data sequence for model input.
        
        Args:
            returns: Recent returns sequence
            cumulative_pnl: Cumulative PnL sequence
            volatility: Rolling volatility
            volumes: Trading volumes
            
        Returns:
            Feature vector (1 x feature_dim)
        """
        n = len(returns)
        if n < self.lookback_window:
            return np.zeros(self._feature_dim)
        
        features = []
        
        # Drawdown-related features
        current_dd = self._compute_current_drawdown(cumulative_pnl)
        features.append(current_dd)
        
        max_dd_recent = np.min(np.minimum.accumulate(cumulative_pnl[-self.lookback_window:]) - cumulative_pnl[-self.lookback_window:])
        features.append(abs(max_dd_recent))
        
        dd_duration = self._compute_drawdown_duration(cumulative_pnl)
        features.append(dd_duration / self.lookback_window)
        
        # Return statistics
        recent_returns = returns[-self.lookback_window:]
        features.append(np.mean(recent_returns))
        features.append(np.std(recent_returns))
        features.append(self._compute_skewness(recent_returns))
        features.append(self._compute_kurtosis(recent_returns))
        
        # Volatility features
        features.append(np.mean(volatility[-self.lookback_window:]))
        features.append(np.std(volatility[-self.lookback_window:]))
        features.append(volatility[-1] / (np.mean(volatility[-20:]) + 1e-10))
        
        # Volume features
        vol_mean = np.mean(volumes[-self.lookback_window:])
        features.append(volumes[-1] / (vol_mean + 1e-10))
        features.append(np.std(volumes[-self.lookback_window:]) / (vol_mean + 1e-10))
        
        # Momentum features
        features.append(cumulative_pnl[-1] - cumulative_pnl[-5])
        features.append(cumulative_pnl[-1] - cumulative_pnl[-20])
        
        # Recovery indicators
        peak_idx = np.argmax(cumulative_pnl[-self.lookback_window:])
        features.append((self.lookback_window - peak_idx) / self.lookback_window)
        
        # Tail risk indicators
        sorted_returns = np.sort(recent_returns)
        features.append(sorted_returns[5])  # 5th percentile
        features.append(sorted_returns[-5])  # 95th percentile
        
        # Correlation with market stress (placeholder)
        features.append(0.0)
        
        # Time since last large loss
        large_losses = np.where(recent_returns < -0.02)[0]
        if len(large_losses) > 0:
            time_since_loss = self.lookback_window - large_losses[-1]
        else:
            time_since_loss = self.lookback_window
        features.append(time_since_loss / self.lookback_window)
        
        # VaR estimate
        features.append(-np.percentile(recent_returns, 5))
        
        # Fill remaining dimensions if needed
        while len(features) < self._feature_dim:
            features.append(0.0)
        
        return np.array(features[:self._feature_dim])
    
    def _compute_current_drawdown(self, cumulative_pnl: np.ndarray) -> float:
        """Compute current drawdown from cumulative PnL."""
        if len(cumulative_pnl) == 0:
            return 0.0
        
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdown = (running_max - cumulative_pnl) / (np.abs(running_max) + 1e-10)
        return float(drawdown[-1])
    
    def _compute_drawdown_duration(self, cumulative_pnl: np.ndarray) -> int:
        """Compute number of periods since last peak."""
        running_max = np.maximum.accumulate(cumulative_pnl)
        in_drawdown = cumulative_pnl < running_max
        
        if not np.any(in_drawdown):
            return 0
        
        # Find start of current drawdown period
        idx = len(cumulative_pnl) - 1
        duration = 0
        while idx >= 0 and in_drawdown[idx]:
            duration += 1
            idx -= 1
        
        return duration
    
    def _compute_skewness(self, x: np.ndarray) -> float:
        """Compute skewness of array."""
        if len(x) < 3:
            return 0.0
        mean = np.mean(x)
        std = np.std(x) + 1e-10
        return float(np.mean(((x - mean) / std) ** 3))
    
    def _compute_kurtosis(self, x: np.ndarray) -> float:
        """Compute excess kurtosis of array."""
        if len(x) < 4:
            return 0.0
        mean = np.mean(x)
        std = np.std(x) + 1e-10
        return float(np.mean(((x - mean) / std) ** 4) - 3)
    
    def predict_probability(self, strategy_id: str,
                            returns: np.ndarray,
                            cumulative_pnl: np.ndarray,
                            volatility: np.ndarray,
                            volumes: np.ndarray) -> Dict[float, float]:
        """
        Predict probability of breaching each drawdown threshold.
        
        Args:
            strategy_id: Unique strategy identifier
            returns: Recent returns
            cumulative_pnl: Cumulative PnL series
            volatility: Rolling volatility
            volumes: Trading volumes
            
        Returns:
            Dict mapping threshold levels to breach probabilities
        """
        # Extract features
        features = self._extract_sequence_features(
            returns, cumulative_pnl, volatility, volumes
        )
        
        # Compute logits using linear model (placeholder for ONNX inference)
        logit = np.dot(features, self._weights) + self._bias
        
        # Update state
        current_dd = self._compute_current_drawdown(cumulative_pnl)
        self._current_drawdown[strategy_id] = current_dd
        
        if strategy_id not in self._drawdown_history:
            self._drawdown_history[strategy_id] = []
        self._drawdown_history[strategy_id].append(current_dd)
        
        # Keep history bounded
        if len(self._drawdown_history[strategy_id]) > 500:
            self._drawdown_history[strategy_id] = self._drawdown_history[strategy_id][-500:]
        
        # Compute probabilities for each threshold
        probabilities = {}
        for threshold in self.threshold_levels:
            # Adjust logit based on distance to threshold
            distance = (threshold - current_dd) / (threshold + 1e-10)
            adjusted_logit = logit - distance * 2.0
            
            # Sigmoid activation
            prob = 1.0 / (1.0 + np.exp(-adjusted_logit))
            probabilities[threshold] = float(prob)
        
        return probabilities
    
    def should_delever(self, strategy_id: str,
                       probabilities: Dict[float, float],
                       max_acceptable_prob: float = 0.7) -> Tuple[bool, float]:
        """
        Determine if deleveraging should be triggered.
        
        Args:
            strategy_id: Strategy identifier
            probabilities: Breach probabilities by threshold
            max_acceptable_prob: Maximum acceptable breach probability
            
        Returns:
            Tuple of (should_delever, recommended_reduction_pct)
        """
        current_dd = self._current_drawdown.get(strategy_id, 0.0)
        
        # Check highest threshold breach probability
        max_prob = max(probabilities.values())
        
        if max_prob > max_acceptable_prob:
            # Calculate recommended reduction based on probability level
            excess_prob = max_prob - max_acceptable_prob
            reduction_pct = min(0.8, excess_prob * 2.0)
            return True, reduction_pct
        
        # Also check absolute drawdown level
        if current_dd > 0.15:  # Hard stop at 15% drawdown
            return True, 0.5
        
        return False, 0.0
    
    def get_delever_command(self, strategy_id: str, 
                            reduction_pct: float) -> Dict:
        """
        Generate Nautilus-compatible deleveraging command.
        
        Args:
            strategy_id: Strategy identifier
            reduction_pct: Percentage to reduce positions
            
        Returns:
            Command dictionary for Nautilus execution
        """
        return {
            "type": "delever",
            "strategy_id": strategy_id,
            "reduction_pct": float(reduction_pct),
            "reason": "drawdown_risk",
            "timestamp": int(time.time() * 1e9),
            "priority": "high"
        }
    
    def reset_strategy(self, strategy_id: str):
        """Reset tracking for a strategy after deleveraging."""
        if strategy_id in self._current_drawdown:
            del self._current_drawdown[strategy_id]
        if strategy_id in self._drawdown_history:
            self._drawdown_history[strategy_id] = []


class DrawdownMonitor:
    """
    Real-time drawdown monitoring system coordinating predictions and actions.
    Integrates with Nautilus strategies for automated risk management.
    """
    
    def __init__(self, strategies: List[str], 
                 max_drawdown_threshold: float = 0.10):
        self.strategies = strategies
        self.max_drawdown_threshold = max_drawdown_threshold
        
        self.predictor = DrawdownPredictor(
            lookback_window=60,
            threshold_levels=[0.05, 0.08, 0.10, 0.15, 0.20]
        )
        
        self._pending_commands: List[Dict] = []
        self._last_check_time: Dict[str, float] = {}
    
    def check_strategies(self, market_data: Dict[str, Dict]) -> List[Dict]:
        """
        Check all strategies for drawdown risk.
        
        Args:
            market_data: Dict mapping strategy_id to market data
            
        Returns:
            List of commands for Nautilus execution
        """
        commands = []
        
        for strategy_id in self.strategies:
            data = market_data.get(strategy_id, {})
            
            returns = data.get("returns", np.zeros(60))
            cumulative_pnl = data.get("cumulative_pnl", np.zeros(60))
            volatility = data.get("volatility", np.zeros(60))
            volumes = data.get("volumes", np.ones(60))
            
            # Predict breach probabilities
            probs = self.predictor.predict_probability(
                strategy_id, returns, cumulative_pnl, volatility, volumes
            )
            
            # Check if deleveraging needed
            should_dd, reduction = self.predictor.should_delever(
                strategy_id, probs, max_acceptable_prob=0.6
            )
            
            if should_dd:
                cmd = self.predictor.get_delever_command(strategy_id, reduction)
                commands.append(cmd)
                self._pending_commands.append(cmd)
            
            self._last_check_time[strategy_id] = time.time()
        
        return commands
    
    def get_pending_commands(self) -> List[Dict]:
        """Get and clear pending commands."""
        cmds = self._pending_commands.copy()
        self._pending_commands.clear()
        return cmds
    
    def get_risk_status(self, strategy_id: str) -> Dict:
        """Get current risk status for a strategy."""
        current_dd = self.predictor._current_drawdown.get(strategy_id, 0.0)
        history = self.predictor._drawdown_history.get(strategy_id, [])
        
        return {
            "strategy_id": strategy_id,
            "current_drawdown": current_dd,
            "max_drawdown_threshold": self.max_drawdown_threshold,
            "breach_risk": "high" if current_dd > self.max_drawdown_threshold * 0.8 else "normal",
            "history_length": len(history),
            "last_check": self._last_check_time.get(strategy_id, 0)
        }


if __name__ == "__main__":
    # Example usage
    strategies = ["stat_arb_01", "momentum_01", "mean_rev_01"]
    
    monitor = DrawdownMonitor(strategies, max_drawdown_threshold=0.10)
    
    # Simulate market data
    np.random.seed(42)
    market_data = {}
    
    for strategy in strategies:
        # Generate synthetic PnL with some drawdown
        returns = np.random.randn(60) * 0.01
        returns[:20] *= 1.5  # Higher volatility at start
        cumulative_pnl = np.cumsum(returns)
        
        # Create a drawdown scenario
        cumulative_pnl[30:40] -= 0.08
        
        market_data[strategy] = {
            "returns": returns,
            "cumulative_pnl": cumulative_pnl,
            "volatility": np.abs(returns) * 2,
            "volumes": np.random.lognormal(10, 0.5, 60)
        }
    
    # Check strategies
    commands = monitor.check_strategies(market_data)
    
    print("Drawdown Monitoring Results:")
    for strategy in strategies:
        status = monitor.get_risk_status(strategy)
        print(f"\n{strategy}:")
        print(f"  Current Drawdown: {status['current_drawdown']:.4f}")
        print(f"  Breach Risk: {status['breach_risk']}")
    
    if commands:
        print(f"\nPending Commands: {len(commands)}")
        for cmd in commands:
            print(f"  {cmd}")
    else:
        print("\nNo deleveraging commands generated.")
