"""
Exchange Net-Flow Z-scores and Stablecoin Velocity metrics.
Predicts structural supply/demand shocks using numba-optimized calculations.
"""

import numpy as np
from numba import njit, prange
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass
import threading
import time


@njit(cache=True)
def compute_ewma(
    data: np.ndarray,
    alpha: float,
    initial_value: float
) -> np.ndarray:
    """Compute exponentially weighted moving average."""
    n = len(data)
    ewma = np.zeros(n, dtype=np.float64)
    ewma[0] = initial_value
    
    for i in range(1, n):
        ewma[i] = alpha * data[i] + (1 - alpha) * ewma[i - 1]
    
    return ewma


@njit(cache=True)
def compute_ewma_std(
    data: np.ndarray,
    ewma_mean: np.ndarray,
    alpha: float
) -> np.ndarray:
    """Compute EWMA standard deviation."""
    n = len(data)
    ewma_var = np.zeros(n, dtype=np.float64)
    ewma_std = np.zeros(n, dtype=np.float64)
    
    # Initial variance estimate
    ewma_var[0] = np.var(data[:min(20, n)]) if n > 1 else 1.0
    
    for i in range(1, n):
        diff_sq = (data[i] - ewma_mean[i]) ** 2
        ewma_var[i] = alpha * diff_sq + (1 - alpha) * ewma_var[i - 1]
        ewma_std[i] = np.sqrt(ewma_var[i])
    
    return ewma_std


@njit(cache=True)
def compute_zscore(
    value: float,
    mean: float,
    std: float
) -> float:
    """Compute z-score for a single value."""
    if std < 1e-10:
        return 0.0
    return (value - mean) / std


@njit(cache=True)
def compute_velocity(
    amounts: np.ndarray,
    balances: np.ndarray,
    time_deltas: np.ndarray
) -> np.ndarray:
    """
    Compute velocity metric.
    Velocity = Total Transacted Volume / Average Balance
    """
    n = len(amounts)
    velocity = np.zeros(n, dtype=np.float64)
    
    cumulative_volume = 0.0
    cumulative_balance = 0.0
    
    for i in range(n):
        cumulative_volume += abs(amounts[i])
        cumulative_balance += balances[i]
        
        if time_deltas[i] > 0:
            avg_balance = cumulative_balance / (i + 1)
            if avg_balance > 1e-10:
                velocity[i] = (cumulative_volume / avg_balance) / time_deltas[i]
            else:
                velocity[i] = 0.0
        else:
            velocity[i] = 0.0
    
    return velocity


@njit(cache=True)
def detect_flow_imbalance(
    inflows: np.ndarray,
    outflows: np.ndarray,
    threshold_zscore: float
) -> np.ndarray:
    """Detect significant flow imbalances."""
    n = len(inflows)
    signals = np.zeros(n, dtype=np.int32)  # -1: outflow shock, 0: neutral, 1: inflow shock
    
    net_flows = inflows - outflows
    
    # Calculate rolling statistics
    window = min(50, n)
    
    for i in range(window, n):
        window_net = net_flows[i - window:i]
        mean = np.mean(window_net)
        std = np.std(window_net) + 1e-10
        
        z = (net_flows[i] - mean) / std
        
        if z > threshold_zscore:
            signals[i] = 1  # Inflow shock (buying pressure)
        elif z < -threshold_zscore:
            signals[i] = -1  # Outflow shock (selling pressure)
    
    return signals


@dataclass
class FlowMetrics:
    """Exchange flow metrics snapshot."""
    
    # Net flow metrics
    net_flow_24h: float = 0.0
    net_flow_zscore: float = 0.0
    inflow_usd: float = 0.0
    outflow_usd: float = 0.0
    
    # Velocity metrics
    stablecoin_velocity: float = 0.0
    btc_velocity: float = 0.0
    velocity_zscore: float = 0.0
    
    # Imbalance signals
    flow_signal: int = 0  # -1, 0, 1
    imbalance_strength: float = 0.0
    
    # Exchange reserves
    exchange_btc_reserves: float = 0.0
    exchange_usd_reserves: float = 0.0
    reserve_change_24h: float = 0.0
    
    # Metadata
    timestamp: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "net_flow_24h": self.net_flow_24h,
            "net_flow_zscore": self.net_flow_zscore,
            "inflow_usd": self.inflow_usd,
            "outflow_usd": self.outflow_usd,
            "stablecoin_velocity": self.stablecoin_velocity,
            "btc_velocity": self.btc_velocity,
            "velocity_zscore": self.velocity_zscore,
            "flow_signal": self.flow_signal,
            "imbalance_strength": self.imbalance_strength,
            "exchange_btc_reserves": self.exchange_btc_reserves,
            "exchange_usd_reserves": self.exchange_usd_reserves,
            "reserve_change_24h": self.reserve_change_24h,
            "timestamp": self.timestamp
        }


class FlowImbalanceAnalyzer:
    """
    Analyzes exchange flows and stablecoin velocity for supply/demand shocks.
    Uses Numba for high-performance z-score and velocity calculations.
    """
    
    def __init__(
        self,
        ewma_halflife: int = 24,  # hours
        zscore_threshold: float = 2.5,
        history_length: int = 500
    ):
        self.ewma_halflife = ewma_halflife
        self.zscore_threshold = zscore_threshold
        self.history_length = history_length
        
        # Calculate EWMA alpha
        self.alpha = 1 - np.exp(-np.log(2) / ewma_halflife)
        
        # Data buffers
        self._inflows: List[float] = []
        self._outflows: List[float] = []
        self._timestamps: List[float] = []
        self._stablecoin_volumes: List[float] = []
        self._btc_volumes: List[float] = []
        self._exchange_balances: List[float] = []
        
        # Computed metrics
        self._net_flow_mean = 0.0
        self._net_flow_std = 1.0
        self._velocity_mean = 0.0
        self._velocity_std = 1.0
        
        # Current state
        self._current_metrics: Optional[FlowMetrics] = None
        
        # Thread safety
        self._lock = threading.RLock()
    
    def add_flow_data(
        self,
        inflow_usd: float,
        outflow_usd: float,
        stablecoin_volume: float,
        btc_volume: float,
        exchange_balance: float,
        timestamp: Optional[float] = None
    ) -> FlowMetrics:
        """Add new flow data point and compute metrics."""
        if timestamp is None:
            timestamp = time.time()
        
        with self._lock:
            # Add to buffers
            self._inflows.append(inflow_usd)
            self._outflows.append(outflow_usd)
            self._timestamps.append(timestamp)
            self._stablecoin_volumes.append(stablecoin_volume)
            self._btc_volumes.append(btc_volume)
            self._exchange_balances.append(exchange_balance)
            
            # Trim buffers
            while len(self._inflows) > self.history_length:
                self._inflows.pop(0)
                self._outflows.pop(0)
                self._timestamps.pop(0)
                self._stablecoin_volumes.pop(0)
                self._btc_volumes.pop(0)
                self._exchange_balances.pop(0)
            
            # Compute metrics
            metrics = self._compute_metrics()
            self._current_metrics = metrics
            
            return metrics
    
    def _compute_metrics(self) -> FlowMetrics:
        """Compute all flow metrics from buffered data."""
        n = len(self._inflows)
        
        if n < 2:
            return FlowMetrics(timestamp=self._timestamps[-1] if self._timestamps else time.time())
        
        # Convert to arrays
        inflows = np.array(self._inflows, dtype=np.float64)
        outflows = np.array(self._outflows, dtype=np.float64)
        net_flows = inflows - outflows
        
        # Compute EWMA statistics
        ewma_mean = compute_ewma(net_flows, self.alpha, np.mean(net_flows[:min(20, n)]))
        ewma_std = compute_ewma_std(net_flows, ewma_mean, self.alpha)
        
        # Current z-score
        current_zscore = compute_zscore(
            net_flows[-1], ewma_mean[-1], ewma_std[-1]
        )
        
        # Update running statistics
        self._net_flow_mean = ewma_mean[-1]
        self._net_flow_std = ewma_std[-1]
        
        # Compute velocity
        stablecoin_vol = np.array(self._stablecoin_volumes, dtype=np.float64)
        balances = np.array(self._exchange_balances, dtype=np.float64)
        
        # Time deltas in hours
        time_deltas = np.ones(n, dtype=np.float64)
        if n > 1:
            ts_arr = np.array(self._timestamps, dtype=np.float64)
            for i in range(1, n):
                time_deltas[i] = max((ts_arr[i] - ts_arr[i-1]) / 3600, 0.01)
        
        velocity = compute_velocity(stablecoin_vol, balances, time_deltas)
        
        # Velocity statistics
        vel_window = min(100, n)
        vel_mean = np.mean(velocity[-vel_window:])
        vel_std = np.std(velocity[-vel_window:]) + 1e-10
        self._velocity_mean = vel_mean
        self._velocity_std = vel_std
        
        current_velocity = velocity[-1]
        velocity_zscore = compute_zscore(current_velocity, vel_mean, vel_std)
        
        # Detect flow imbalance
        signals = detect_flow_imbalance(inflows, outflows, self.zscore_threshold)
        current_signal = signals[-1] if n > 50 else 0
        
        # Calculate imbalance strength
        imbalance_strength = abs(current_zscore) / self.zscore_threshold
        imbalance_strength = min(imbalance_strength, 1.0)
        
        # 24h aggregates (assuming hourly data)
        hours_24 = min(24, n)
        net_flow_24h = np.sum(net_flows[-hours_24:])
        inflow_24h = np.sum(inflows[-hours_24:])
        outflow_24h = np.sum(outflows[-hours_24:])
        
        # Reserve change
        reserve_change = 0.0
        if n >= 24:
            reserve_change = self._exchange_balances[-1] - self._exchange_balances[-24]
        
        return FlowMetrics(
            net_flow_24h=net_flow_24h,
            net_flow_zscore=float(current_zscore),
            inflow_usd=inflow_24h,
            outflow_usd=outflow_24h,
            stablecoin_velocity=float(current_velocity),
            btc_velocity=float(np.mean(self._btc_volumes[-hours_24:])),
            velocity_zscore=float(velocity_zscore),
            flow_signal=int(current_signal),
            imbalance_strength=float(imbalance_strength),
            exchange_btc_reserves=self._exchange_balances[-1] / 50000,  # Approx BTC
            exchange_usd_reserves=self._exchange_balances[-1],
            reserve_change_24h=reserve_change,
            timestamp=self._timestamps[-1]
        )
    
    def get_shock_probability(self) -> float:
        """
        Calculate probability of supply/demand shock.
        Based on combined z-scores and signal strength.
        """
        with self._lock:
            if self._current_metrics is None:
                return 0.0
            
            metrics = self._current_metrics
            
            # Combine signals
            flow_component = min(abs(metrics.net_flow_zscore) / 3.0, 1.0)
            velocity_component = min(abs(metrics.velocity_zscore) / 2.0, 1.0)
            signal_component = abs(metrics.flow_signal) * metrics.imbalance_strength
            
            # Weighted combination
            probability = (
                0.5 * flow_component +
                0.3 * velocity_component +
                0.2 * signal_component
            )
            
            return float(min(probability, 1.0))
    
    def get_directional_bias(self) -> float:
        """
        Get directional bias from flow analysis.
        Positive = bullish (accumulation), Negative = bearish (distribution)
        """
        with self._lock:
            if self._current_metrics is None:
                return 0.0
            
            metrics = self._current_metrics
            
            # Negative net flow (outflows) = bullish (moving to cold storage)
            # Positive net flow (inflows) = bearish (moving to exchanges)
            flow_signal = -metrics.net_flow_zscore / 3.0
            
            # High velocity = more trading = potential top/bottom
            velocity_adjustment = 0.0
            if metrics.velocity_zscore > 2:
                velocity_adjustment = -0.1  # High velocity at top
            
            return float(np.clip(flow_signal + velocity_adjustment, -1.0, 1.0))
    
    def get_current_metrics(self) -> Optional[FlowMetrics]:
        """Get current flow metrics."""
        with self._lock:
            return self._current_metrics
    
    def reset(self) -> None:
        """Reset all state."""
        with self._lock:
            self._inflows.clear()
            self._outflows.clear()
            self._timestamps.clear()
            self._stablecoin_volumes.clear()
            self._btc_volumes.clear()
            self._exchange_balances.clear()
            self._net_flow_mean = 0.0
            self._net_flow_std = 1.0
            self._velocity_mean = 0.0
            self._velocity_std = 1.0
            self._current_metrics = None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get analyzer statistics."""
        with self._lock:
            return {
                "data_points": len(self._inflows),
                "history_length": self.history_length,
                "ewma_halflife": self.ewma_halflife,
                "zscore_threshold": self.zscore_threshold,
                "net_flow_mean": self._net_flow_mean,
                "net_flow_std": self._net_flow_std,
                "shock_probability": self.get_shock_probability(),
                "directional_bias": self.get_directional_bias()
            }


# Global singleton instance
_flow_instance: Optional[FlowImbalanceAnalyzer] = None
_instance_lock = threading.Lock()


def get_flow_analyzer() -> FlowImbalanceAnalyzer:
    """Get or create the global flow analyzer."""
    global _flow_instance
    if _flow_instance is None:
        with _instance_lock:
            if _flow_instance is None:
                _flow_instance = FlowImbalanceAnalyzer()
    return _flow_instance


if __name__ == "__main__":
    # Test flow imbalance analyzer
    print("Testing FlowImbalanceAnalyzer:")
    
    analyzer = FlowImbalanceAnalyzer(zscore_threshold=2.0)
    
    np.random.seed(42)
    
    # Simulate normal flow period
    print("\n--- Normal Period ---")
    for i in range(100):
        inflow = 1e7 + np.random.randn() * 1e6
        outflow = 1e7 + np.random.randn() * 1e6
        stablecoin_vol = 5e7 + np.random.randn() * 5e6
        btc_vol = 1000 + np.random.randn() * 100
        balance = 2e9 + np.random.randn() * 1e8
        
        metrics = analyzer.add_flow_data(
            inflow, outflow, stablecoin_vol, btc_vol, balance
        )
    
    print(f"Net Flow Z-Score: {metrics.net_flow_zscore:.2f}")
    print(f"Velocity Z-Score: {metrics.velocity_zscore:.2f}")
    print(f"Shock Probability: {analyzer.get_shock_probability():.2f}")
    print(f"Directional Bias: {analyzer.get_directional_bias():.2f}")
    
    # Simulate shock event
    print("\n--- Shock Event ---")
    for i in range(10):
        # Large inflows (bearish - moving to exchanges)
        inflow = 5e7 + np.random.randn() * 5e6
        outflow = 5e6 + np.random.randn() * 1e6
        
        metrics = analyzer.add_flow_data(
            inflow, outflow, stablecoin_vol, btc_vol, balance
        )
    
    print(f"Net Flow Z-Score: {metrics.net_flow_zscore:.2f}")
    print(f"Flow Signal: {metrics.flow_signal}")
    print(f"Shock Probability: {analyzer.get_shock_probability():.2f}")
    print(f"Directional Bias: {analyzer.get_directional_bias():.2f}")
    
    print(f"\nStats: {analyzer.get_stats()}")
