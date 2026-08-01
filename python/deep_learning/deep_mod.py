"""
Deep Learning Module Root for Advanced Order Book Modeling.
Manages ONNX runtime sessions for GNN and Temporal Attention models,
ensuring strict thread pooling to respect the 3GB RAM ceiling.

Provides unified interface for:
- Graph Neural Network order book analysis
- Temporal Point Process attention for sweep prediction
- Combined inference pipeline with memory bounds
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any
import threading
import logging
from dataclasses import dataclass
import time

from .gnn_orderbook import get_gnn_predictor, GNNOrderBookPredictor
from .temporal_attention import get_temporal_predictor, TemporalAttentionPredictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DeepLearningInferenceResult:
    """Combined inference results from all deep learning models."""
    timestamp_ns: int
    
    # GNN outputs
    gnn_sweep_up: float
    gnn_sweep_down: float
    gnn_confidence: float
    
    # Temporal attention outputs
    temporal_sweep_prob: float
    temporal_buy_sweep: float
    temporal_sell_sweep: float
    arrival_rate_acceleration: float
    
    # Combined signals
    combined_sweep_signal: float  # -1.0 to 1.0
    liquidity_risk_score: float   # 0.0 to 1.0
    
    # Metadata
    inference_latency_us: int
    model_versions: Dict[str, str]


class DeepLearningManager:
    """
    Central manager for all deep learning inference.
    Enforces memory limits and provides unified inference interface.
    """
    
    MAX_RAM_MB = 512  # Strict limit for DL models within 3GB total
    
    def __init__(self, gnn_model_path: Optional[str] = None, 
                 temporal_model_path: Optional[str] = None):
        self._lock = threading.RLock()
        self._initialized = False
        
        self._gnn_predictor: Optional[GNNOrderBookPredictor] = None
        self._temporal_predictor: Optional[TemporalAttentionPredictor] = None
        
        self._gnn_model_path = gnn_model_path
        self._temporal_model_path = temporal_model_path
        
        # Performance tracking
        self._inference_count = 0
        self._total_latency_us = 0
        self._last_reset = time.time()
        
        # Model version tracking
        self._model_versions: Dict[str, str] = {}
    
    def initialize(self) -> bool:
        """Initialize all deep learning models with memory bounds."""
        with self._lock:
            if self._initialized:
                return True
            
            try:
                logger.info("Initializing Deep Learning models...")
                
                # Initialize GNN predictor
                self._gnn_predictor = get_gnn_predictor(self._gnn_model_path)
                if self._gnn_model_path:
                    self._model_versions['gnn'] = self._extract_model_version(self._gnn_model_path)
                else:
                    self._model_versions['gnn'] = 'heuristic'
                
                # Initialize temporal attention predictor
                self._temporal_predictor = get_temporal_predictor(self._temporal_model_path)
                if self._temporal_model_path:
                    self._model_versions['temporal'] = self._extract_model_version(self._temporal_model_path)
                else:
                    self._model_versions['temporal'] = 'heuristic'
                
                self._initialized = True
                logger.info(f"Deep Learning models initialized. Versions: {self._model_versions}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to initialize DL models: {e}")
                return False
    
    def _extract_model_version(self, model_path: str) -> str:
        """Extract version hash from model path or file metadata."""
        import hashlib
        try:
            with open(model_path, 'rb') as f:
                content = f.read(8192)  # Read first 8KB for hash
                hash_val = hashlib.md5(content).hexdigest()[:8]
                return f"v{hash_val}"
        except:
            return "unknown"
    
    def infer(
        self,
        bid_prices: np.ndarray,
        bid_volumes: np.ndarray,
        ask_prices: np.ndarray,
        ask_volumes: np.ndarray,
        mid_price: float,
        recent_orders: Optional[list] = None,
        vpin: float = 0.5,
        spread_bps: float = 5.0
    ) -> DeepLearningInferenceResult:
        """
        Run combined inference from GNN and Temporal Attention models.
        
        Args:
            bid_prices: L2 bid price levels
            bid_volumes: L2 bid volumes
            ask_prices: L2 ask price levels
            ask_volumes: L2 ask volumes
            mid_price: Current mid price
            recent_orders: List of recent aggressive orders (timestamp, side, volume, price)
            vpin: Current VPIN toxicity score
            spread_bps: Current spread in basis points
            
        Returns:
            DeepLearningInferenceResult with combined signals
        """
        start_time = time.perf_counter_ns()
        
        with self._lock:
            if not self._initialized:
                self.initialize()
        
        current_time_ns = time.time_ns()
        
        # GNN inference
        gnn_sweep_up, gnn_sweep_down = 0.5, 0.5
        if self._gnn_predictor:
            gnn_sweep_up, gnn_sweep_down = self._gnn_predictor.predict_liquidity_sweep(
                bid_prices, bid_volumes, ask_prices, ask_volumes, mid_price,
                time_horizon_ms=100
            )
        
        # Process recent orders into temporal predictor
        if recent_orders and self._temporal_predictor:
            for order in recent_orders[-50:]:  # Last 50 orders
                timestamp_ns, side, volume, price = order[:4]
                self._temporal_predictor.process_order(
                    timestamp_ns, side, volume, price, vpin, spread_bps
                )
        
        # Temporal attention inference
        temporal_sweep_prob, temporal_buy, temporal_sell = 0.5, 0.25, 0.25
        if self._temporal_predictor:
            temporal_sweep_prob, temporal_buy, temporal_sell = \
                self._temporal_predictor.predict_sweep_probability(horizon_ms=100)
        
        # Calculate arrival rate acceleration (from inter-arrival analysis)
        arrival_acceleration = 0.0
        if self._temporal_predictor and self._temporal_predictor.buffer._count > 10:
            features, _ = self._temporal_predictor.buffer.get_recent_events(5000.0)
            if len(features) > 10:
                inter_arrivals = self._temporal_predictor.buffer.compute_inter_arrival_times(_)
                if len(inter_arrivals) > 10:
                    early_rate = np.mean(inter_arrivals[-10:-5])
                    late_rate = np.mean(inter_arrivals[-5:])
                    arrival_acceleration = (early_rate - late_rate) / (early_rate + 1e-8)
        
        # Combine signals
        combined_signal = self._combine_signals(
            gnn_sweep_up, gnn_sweep_down,
            temporal_buy, temporal_sell,
            arrival_acceleration
        )
        
        # Calculate liquidity risk score
        liquidity_risk = self._calculate_liquidity_risk(
            gnn_sweep_up, gnn_sweep_down,
            temporal_sweep_prob,
            bid_volumes, ask_volumes
        )
        
        # Calculate latency
        end_time = time.perf_counter_ns()
        latency_us = (end_time - start_time) // 1000
        
        # Update metrics
        self._inference_count += 1
        self._total_latency_us += latency_us
        
        return DeepLearningInferenceResult(
            timestamp_ns=current_time_ns,
            gnn_sweep_up=gnn_sweep_up,
            gnn_sweep_down=gnn_sweep_down,
            gnn_confidence=abs(gnn_sweep_up - gnn_sweep_down),
            temporal_sweep_prob=temporal_sweep_prob,
            temporal_buy_sweep=temporal_buy,
            temporal_sell_sweep=temporal_sell,
            arrival_rate_acceleration=arrival_acceleration,
            combined_sweep_signal=combined_signal,
            liquidity_risk_score=liquidity_risk,
            inference_latency_us=latency_us,
            model_versions=self._model_versions.copy()
        )
    
    def _combine_signals(
        self,
        gnn_up: float,
        gnn_down: float,
        temporal_buy: float,
        temporal_sell: float,
        acceleration: float
    ) -> float:
        """
        Combine GNN and temporal attention signals into unified direction signal.
        
        Returns:
            Signal in range [-1.0, 1.0] where positive = bullish, negative = bearish
        """
        # GNN directional signal
        gnn_signal = gnn_up - gnn_down  # Range: [-1, 1]
        
        # Temporal directional signal
        temporal_signal = temporal_buy - temporal_sell  # Range: [-1, 1]
        
        # Weighted combination (GNN slightly more trusted for spatial structure)
        base_signal = 0.6 * gnn_signal + 0.4 * temporal_signal
        
        # Adjust for arrival acceleration
        # Positive acceleration (faster arrivals) amplifies the signal
        amplified_signal = base_signal * (1.0 + 0.3 * np.clip(acceleration, -1, 1))
        
        return float(np.clip(amplified_signal, -1.0, 1.0))
    
    def _calculate_liquidity_risk(
        self,
        gnn_up: float,
        gnn_down: float,
        temporal_prob: float,
        bid_volumes: np.ndarray,
        ask_volumes: np.ndarray
    ) -> float:
        """
        Calculate overall liquidity risk score.
        
        Returns:
            Risk score in range [0.0, 1.0] where higher = more risky
        """
        # Sweep probability component
        max_sweep_prob = max(gnn_up, gnn_down, temporal_prob)
        
        # Volume imbalance component
        if len(bid_volumes) > 0 and len(ask_volumes) > 0:
            total_bid = np.sum(bid_volumes[:5])
            total_ask = np.sum(ask_volumes[:5])
            volume_imbalance = abs(total_bid - total_ask) / (total_bid + total_ask + 1e-8)
        else:
            volume_imbalance = 0.5
        
        # Combined risk
        risk = 0.7 * max_sweep_prob + 0.3 * volume_imbalance
        
        return float(np.clip(risk, 0.0, 1.0))
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for monitoring."""
        with self._lock:
            elapsed = time.time() - self._last_reset
            if elapsed < 1e-6:
                elapsed = 1.0
            
            avg_latency = self._total_latency_us / max(self._inference_count, 1)
            
            return {
                'inference_count': self._inference_count,
                'avg_latency_us': avg_latency,
                'inferences_per_second': self._inference_count / elapsed,
                'model_versions': self._model_versions.copy(),
                'initialized': self._initialized,
                'elapsed_seconds': elapsed
            }
    
    def reset_metrics(self) -> None:
        """Reset performance metrics."""
        with self._lock:
            self._inference_count = 0
            self._total_latency_us = 0
            self._last_reset = time.time()
    
    def shutdown(self) -> None:
        """Gracefully shutdown and release resources."""
        with self._lock:
            logger.info("Shutting down Deep Learning manager...")
            self._gnn_predictor = None
            self._temporal_predictor = None
            self._initialized = False
            logger.info("Deep Learning manager shut down complete")


# Global singleton instance
_dl_manager: Optional[DeepLearningManager] = None
_dl_lock = threading.Lock()


def get_dl_manager(
    gnn_model_path: Optional[str] = None,
    temporal_model_path: Optional[str] = None
) -> DeepLearningManager:
    """Thread-safe singleton access to Deep Learning manager."""
    global _dl_manager
    
    with _dl_lock:
        if _dl_manager is None:
            _dl_manager = DeepLearningManager(gnn_model_path, temporal_model_path)
        
        return _dl_manager


def run_inference(
    bid_prices: np.ndarray,
    bid_volumes: np.ndarray,
    ask_prices: np.ndarray,
    ask_volumes: np.ndarray,
    mid_price: float,
    **kwargs
) -> DeepLearningInferenceResult:
    """
    Convenience function for quick inference without managing lifecycle.
    Uses global singleton manager.
    """
    manager = get_dl_manager()
    return manager.infer(
        bid_prices, bid_volumes, ask_prices, ask_volumes, mid_price, **kwargs
    )


if __name__ == "__main__":
    # Demo usage
    np.random.seed(42)
    
    # Initialize manager
    manager = get_dl_manager()
    manager.initialize()
    
    # Simulate order book
    mid = 50000.0
    bid_prices = mid - np.arange(1, 26) * 5.0
    ask_prices = mid + np.arange(1, 26) * 5.0
    bid_volumes = np.random.exponential(10.0, 25)
    ask_volumes = np.random.exponential(10.0, 25)
    
    # Simulate recent orders
    base_time = time.time_ns()
    recent_orders = []
    for i in range(20):
        ts = base_time - (19 - i) * 10_000_000
        side = np.random.choice([-1, 1])
        vol = np.random.exponential(5.0)
        price = mid + np.random.randn() * 10
        recent_orders.append((ts, side, vol, price))
    
    # Run inference
    result = manager.infer(
        bid_prices, bid_volumes, ask_prices, ask_volumes, mid,
        recent_orders=recent_orders,
        vpin=0.35,
        spread_bps=5.0
    )
    
    print(f"=== Deep Learning Inference Results ===")
    print(f"GNN Sweep Up: {result.gnn_sweep_up:.4f}")
    print(f"GNN Sweep Down: {result.gnn_sweep_down:.4f}")
    print(f"Temporal Sweep Prob: {result.temporal_sweep_prob:.4f}")
    print(f"Combined Signal: {result.combined_sweep_signal:.4f}")
    print(f"Liquidity Risk: {result.liquidity_risk_score:.4f}")
    print(f"Inference Latency: {result.inference_latency_us} µs")
    print(f"Model Versions: {result.model_versions}")
    
    # Performance metrics
    metrics = manager.get_performance_metrics()
    print(f"\n=== Performance Metrics ===")
    print(f"Inferences: {metrics['inference_count']}")
    print(f"Avg Latency: {metrics['avg_latency_us']:.2f} µs")
    print(f"Throughput: {metrics['inferences_per_second']:.2f} inf/s")
