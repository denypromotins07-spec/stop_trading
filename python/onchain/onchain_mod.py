"""
On-Chain Module Root - Integrates on-chain structural imbalances as alpha filters.
Combines whale clustering and flow analysis for HFT strategy signals.
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from dataclasses import dataclass, field
import threading
import time

from .whale_clustering import WhaleClustering, WalletProfile, WalletType, ClusterResult, get_whale_clusterer
from .flow_imbalance import FlowImbalanceAnalyzer, FlowMetrics, get_flow_analyzer


@dataclass
class OnChainSignal:
    """Combined on-chain signal for alpha generation."""
    
    # Whale/syndicate signals
    syndicate_activity: float = 0.0  # 0-1 scale
    institutional_flow: float = 0.0  # USD value
    whale_accumulation: bool = False
    
    # Flow imbalance signals
    net_flow_zscore: float = 0.0
    velocity_zscore: float = 0.0
    shock_probability: float = 0.0
    directional_bias: float = 0.0  # -1 to 1
    
    # Combined alpha signal
    alpha_signal: float = 0.0  # -1 (bearish) to 1 (bullish)
    signal_strength: float = 0.0  # 0-1 confidence
    signal_type: str = "NEUTRAL"  # ACCUMULATION, DISTRIBUTION, NEUTRAL
    
    # Metadata
    timestamp: float = 0.0
    active_syndicates: int = 0
    exchange_reserves_change: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "syndicate_activity": self.syndicate_activity,
            "institutional_flow": self.institutional_flow,
            "whale_accumulation": self.whale_accumulation,
            "net_flow_zscore": self.net_flow_zscore,
            "velocity_zscore": self.velocity_zscore,
            "shock_probability": self.shock_probability,
            "directional_bias": self.directional_bias,
            "alpha_signal": self.alpha_signal,
            "signal_strength": self.signal_strength,
            "signal_type": self.signal_type,
            "timestamp": self.timestamp,
            "active_syndicates": self.active_syndicates,
            "exchange_reserves_change": self.exchange_reserves_change
        }


class OnChainAlphaGenerator:
    """
    Generates alpha signals from on-chain data.
    Combines whale clustering, flow analysis, and structural imbalances.
    """
    
    def __init__(
        self,
        syndicate_weight: float = 0.35,
        flow_weight: float = 0.40,
        velocity_weight: float = 0.25
    ):
        # Sub-modules
        self.whale_clusterer = WhaleClustering(min_samples=3, min_cluster_size=2)
        self.flow_analyzer = FlowImbalanceAnalyzer()
        
        # Signal weights
        self.syndicate_weight = syndicate_weight
        self.flow_weight = flow_weight
        self.velocity_weight = velocity_weight
        
        # State
        self._current_signal: Optional[OnChainSignal] = None
        self._signal_history: List[OnChainSignal] = []
        self._history_max = 200
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Performance tracking
        self._total_updates = 0
        self._last_update_time = 0.0
    
    def add_wallet_data(self, profiles: List[WalletProfile]) -> None:
        """Add wallet profiles for clustering."""
        with self._lock:
            self.whale_clusterer.add_wallets_batch(profiles)
    
    def add_flow_data(
        self,
        inflow_usd: float,
        outflow_usd: float,
        stablecoin_volume: float,
        btc_volume: float,
        exchange_balance: float
    ) -> None:
        """Add flow data for analysis."""
        with self._lock:
            self.flow_analyzer.add_flow_data(
                inflow_usd, outflow_usd, stablecoin_volume,
                btc_volume, exchange_balance
            )
    
    def generate_signal(self) -> OnChainSignal:
        """Generate combined on-chain alpha signal."""
        with self._lock:
            start_time = time.perf_counter()
            
            # Run whale clustering
            cluster_result = self.whale_clusterer.cluster()
            
            # Get flow metrics
            flow_metrics = self.flow_analyzer.get_current_metrics()
            
            # Calculate component signals
            syndicate_signal = self._compute_syndicate_signal(cluster_result)
            flow_signal = self._compute_flow_signal(flow_metrics)
            velocity_signal = self._compute_velocity_signal(flow_metrics)
            
            # Combine signals
            alpha_signal = (
                self.syndicate_weight * syndicate_signal +
                self.flow_weight * flow_signal +
                self.velocity_weight * velocity_signal
            )
            
            # Clip to [-1, 1]
            alpha_signal = float(np.clip(alpha_signal, -1.0, 1.0))
            
            # Calculate signal strength
            signal_strength = self._compute_signal_strength(
                cluster_result, flow_metrics
            )
            
            # Determine signal type
            signal_type = self._classify_signal(alpha_signal, signal_strength)
            
            # Build signal object
            institutional_flow = self.whale_clusterer.get_institutional_flow()
            directional_bias = self.flow_analyzer.get_directional_bias()
            
            signal = OnChainSignal(
                syndicate_activity=syndicate_signal,
                institutional_flow=institutional_flow,
                whale_accumulation=directional_bias > 0.3,
                net_flow_zscore=flow_metrics.net_flow_zscore if flow_metrics else 0.0,
                velocity_zscore=flow_metrics.velocity_zscore if flow_metrics else 0.0,
                shock_probability=self.flow_analyzer.get_shock_probability(),
                directional_bias=directional_bias,
                alpha_signal=alpha_signal,
                signal_strength=signal_strength,
                signal_type=signal_type,
                timestamp=time.time(),
                active_syndicates=len(self.whale_clusterer.get_syndicate_wallets()),
                exchange_reserves_change=flow_metrics.reserve_change_24h if flow_metrics else 0.0
            )
            
            # Update history
            self._current_signal = signal
            self._signal_history.append(signal)
            if len(self._signal_history) > self._history_max:
                self._signal_history.pop(0)
            
            self._total_updates += 1
            self._last_update_time = time.time()
            
            return signal
    
    def _compute_syndicate_signal(self, cluster_result: ClusterResult) -> float:
        """Compute signal from syndicate activity."""
        n_syndicates = len(self.whale_clusterer._syndicate_clusters)
        n_total = cluster_result.n_clusters
        
        if n_total == 0:
            return 0.0
        
        # Ratio of syndicate clusters
        syndicate_ratio = n_syndicates / max(n_total, 1)
        
        # Check if syndicates are accumulating
        inst_flow = self.whale_clusterer.get_institutional_flow()
        flow_direction = np.tanh(inst_flow / 1e7)  # Normalize
        
        return float(syndicate_ratio * 0.5 + 0.5 * flow_direction)
    
    def _compute_flow_signal(self, metrics: Optional[FlowMetrics]) -> float:
        """Compute signal from flow imbalances."""
        if metrics is None:
            return 0.0
        
        # Negative z-score = outflows = bullish
        zscore_component = -metrics.net_flow_zscore / 3.0
        
        # Shock probability adjustment
        shock_adj = 0.0
        if metrics.shock_probability > 0.5:
            # High shock probability = stronger signal
            shock_adj = np.sign(zscore_component) * 0.3
        
        return float(np.clip(zscore_component + shock_adj, -1.0, 1.0))
    
    def _compute_velocity_signal(self, metrics: Optional[FlowMetrics]) -> float:
        """Compute signal from velocity metrics."""
        if metrics is None:
            return 0.0
        
        # High velocity can indicate tops or bottoms
        vel_zscore = metrics.velocity_zscore
        
        # Extreme velocity often marks turning points
        if abs(vel_zscore) > 2:
            # Velocity spike - potential reversal
            # If price going up with high velocity = distribution (bearish)
            # If price going down with high velocity = accumulation (bullish)
            # We use flow direction as proxy
            flow_dir = -metrics.net_flow_zscore / 3.0
            return float(-np.sign(vel_zscore) * flow_dir * 0.5)
        
        return 0.0
    
    def _compute_signal_strength(
        self,
        cluster_result: ClusterResult,
        metrics: Optional[FlowMetrics]
    ) -> float:
        """Compute overall signal confidence/strength."""
        components = []
        
        # Syndicate clarity
        if cluster_result.n_clusters > 0:
            syndicate_clarity = len(self.whale_clusterer._syndicate_clusters) / cluster_result.n_clusters
            components.append(min(syndicate_clarity, 1.0))
        
        # Z-score magnitude
        if metrics:
            zscore_confidence = min(abs(metrics.net_flow_zscore) / 3.0, 1.0)
            components.append(zscore_confidence)
            
            shock_confidence = metrics.shock_probability
            components.append(shock_confidence)
        
        if not components:
            return 0.5
        
        return float(np.mean(components))
    
    def _classify_signal(self, alpha: float, strength: float) -> str:
        """Classify signal type based on alpha and strength."""
        if strength < 0.3:
            return "NEUTRAL"
        
        if alpha > 0.3:
            return "ACCUMULATION"
        elif alpha < -0.3:
            return "DISTRIBUTION"
        else:
            return "NEUTRAL"
    
    def get_current_signal(self) -> Optional[OnChainSignal]:
        """Get current on-chain signal."""
        with self._lock:
            return self._current_signal
    
    def get_alpha_for_strategy(self) -> Tuple[float, float]:
        """
        Get alpha signal and strength for HFT strategy.
        Returns (alpha, strength) tuple.
        """
        with self._lock:
            if self._current_signal is None:
                return 0.0, 0.0
            
            return self._current_signal.alpha_signal, self._current_signal.signal_strength
    
    def should_filter_trade(self, side: str) -> bool:
        """
        Determine if a trade should be filtered based on on-chain signals.
        
        Args:
            side: 'LONG' or 'SHORT'
        
        Returns:
            True if trade should be filtered (avoided)
        """
        with self._lock:
            if self._current_signal is None:
                return False
            
            signal = self._current_signal
            
            # Strong contrary signal = filter
            if side == 'LONG' and signal.alpha_signal < -0.5 and signal.signal_strength > 0.6:
                return True  # Filter long when strong bearish on-chain
            
            if side == 'SHORT' and signal.alpha_signal > 0.5 and signal.signal_strength > 0.6:
                return True  # Filter short when strong bullish on-chain
            
            # High shock probability = reduce trading
            if signal.shock_probability > 0.7:
                return True
            
            return False
    
    def reset(self) -> None:
        """Reset all state."""
        with self._lock:
            self.whale_clusterer.reset()
            self.flow_analyzer.reset()
            self._current_signal = None
            self._signal_history.clear()
            self._total_updates = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get generator statistics."""
        with self._lock:
            return {
                "total_updates": self._total_updates,
                "last_update_time": self._last_update_time,
                "signal_history_size": len(self._signal_history),
                "whale_stats": self.whale_clusterer.get_stats(),
                "flow_stats": self.flow_analyzer.get_stats(),
                "current_alpha": self._current_signal.alpha_signal if self._current_signal else 0.0
            }


# Global singleton instance
_onchain_instance: Optional[OnChainAlphaGenerator] = None
_instance_lock = threading.Lock()


def get_onchain_generator() -> OnChainAlphaGenerator:
    """Get or create the global on-chain alpha generator."""
    global _onchain_instance
    if _onchain_instance is None:
        with _instance_lock:
            if _onchain_instance is None:
                _onchain_instance = OnChainAlphaGenerator()
    return _onchain_instance


if __name__ == "__main__":
    # Test on-chain alpha generator
    print("Testing OnChainAlphaGenerator:")
    
    generator = OnChainAlphaGenerator()
    
    np.random.seed(42)
    
    # Add some wallet data
    wallets = []
    for i in range(10):
        wallets.append(WalletProfile(
            address=f"wallet_{i}",
            total_volume=np.random.uniform(1e6, 1e8),
            transaction_count=np.random.randint(50, 200),
            avg_tx_size=np.random.uniform(1e4, 1e6),
            tx_frequency=np.random.uniform(2, 10),
            unique_counterparties=np.random.randint(10, 50),
            net_flow_24h=np.random.uniform(-1e6, 5e6),
            balance_usd=np.random.uniform(1e6, 1e8)
        ))
    
    generator.add_wallet_data(wallets)
    
    # Add flow data
    print("\n--- Adding Flow Data ---")
    for i in range(60):
        inflow = 1e7 + np.random.randn() * 2e6
        outflow = 1e7 + np.random.randn() * 2e6
        
        # Simulate accumulation pattern
        if i > 40:
            outflow = 2e7 + np.random.randn() * 2e6  # More outflows
        
        generator.add_flow_data(
            inflow, outflow,
            5e7 + np.random.randn() * 1e7,
            1000 + np.random.randn() * 200,
            2e9 + np.random.randn() * 1e8
        )
    
    # Generate signal
    signal = generator.generate_signal()
    
    print(f"\nAlpha Signal: {signal.alpha_signal:.4f}")
    print(f"Signal Strength: {signal.signal_strength:.4f}")
    print(f"Signal Type: {signal.signal_type}")
    print(f"Directional Bias: {signal.directional_bias:.4f}")
    print(f"Shock Probability: {signal.shock_probability:.4f}")
    print(f"Active Syndicates: {signal.active_syndicates}")
    
    # Test trade filtering
    print(f"\nFilter LONG: {generator.should_filter_trade('LONG')}")
    print(f"Filter SHORT: {generator.should_filter_trade('SHORT')}")
    
    print(f"\nStats: {generator.get_stats()}")
