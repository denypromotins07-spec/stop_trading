"""
Temporal Point Process Attention for Market Order Arrival Prediction.
Models exact arrival times of aggressive market orders to predict
liquidity sweeps in the next N milliseconds.

Uses bounded tensor operations and ONNX export for minimal memory footprint.
Strictly avoids heavy PyTorch eager-mode during inference.
"""

import numpy as np
from typing import Tuple, Optional, List
import onnxruntime as ort
from dataclasses import dataclass
import threading
from collections import deque
import time

# Configure ONNX Runtime for minimal footprint
SESSION_OPTIONS = ort.SessionOptions()
SESSION_OPTIONS.intra_op_num_threads = 1
SESSION_OPTIONS.inter_op_num_threads = 1
SESSION_OPTIONS.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
SESSION_OPTIONS.enable_mem_pattern = True


@dataclass
class OrderEvent:
    """Represents a single aggressive market order event."""
    timestamp_ns: int
    side: int  # 1=buy, -1=sell
    volume: float
    price: float
    aggressor_initiated: bool
    order_id_hash: int


class TemporalAttentionBuffer:
    """
    Circular buffer for temporal attention computation.
    Pre-allocates memory to avoid heap allocations during hot paths.
    """
    
    MAX_EVENTS = 1000
    FEATURE_DIM = 8
    
    def __init__(self):
        self._lock = threading.Lock()
        # Pre-allocate feature buffer [MAX_EVENTS, FEATURE_DIM]
        self._features = np.zeros((self.MAX_EVENTS, self.FEATURE_DIM), dtype=np.float32)
        # Pre-allocate timestamp buffer
        self._timestamps = np.zeros(self.MAX_EVENTS, dtype=np.int64)
        # Circular buffer pointers
        self._head = 0
        self._count = 0
        self._last_timestamp = 0
    
    def add_event(
        self,
        timestamp_ns: int,
        side: int,
        volume: float,
        price: float,
        inter_arrival_ms: float,
        cumulative_volume: float,
        vpin: float,
        spread_bps: float
    ) -> None:
        """Add a new order event to the buffer."""
        with self._lock:
            # Encode features
            self._features[self._head] = [
                float(side),  # Direction
                np.log1p(volume),  # Log volume
                inter_arrival_ms,  # Time since last event
                cumulative_volume,  # Running volume
                vpin,  # VPIN toxicity
                spread_bps,  # Current spread
                (timestamp_ns % 1_000_000_000) / 1e9,  # Sub-second timing
                np.sin(2 * np.pi * (timestamp_ns % 86400_000_000_000) / 86400_000_000_000)  # Daily cycle
            ]
            self._timestamps[self._head] = timestamp_ns
            
            # Update circular buffer
            self._head = (self._head + 1) % self.MAX_EVENTS
            if self._count < self.MAX_EVENTS:
                self._count += 1
            
            self._last_timestamp = timestamp_ns
    
    def get_recent_events(self, window_ms: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get events within the specified time window.
        
        Returns:
            features: [N, FEATURE_DIM] array
            timestamps: [N,] array
        """
        with self._lock:
            if self._count == 0:
                return np.zeros((0, self.FEATURE_DIM), dtype=np.float32), np.zeros(0, dtype=np.int64)
            
            current_time = self._last_timestamp
            window_ns = int(window_ms * 1_000_000)
            cutoff = current_time - window_ns
            
            # Collect recent events
            features_list = []
            timestamps_list = []
            
            idx = self._head - 1
            if idx < 0:
                idx = self.MAX_EVENTS - 1
            
            while True:
                if self._timestamps[idx] >= cutoff:
                    features_list.append(self._features[idx].copy())
                    timestamps_list.append(self._timestamps[idx])
                    
                    idx -= 1
                    if idx < 0:
                        idx = self.MAX_EVENTS - 1
                    
                    # Check if we've wrapped around completely
                    if idx == self._head:
                        break
                else:
                    break
                
                # Safety check to prevent infinite loop
                if len(features_list) >= self._count:
                    break
            
            if len(features_list) == 0:
                return np.zeros((0, self.FEATURE_DIM), dtype=np.float32), np.zeros(0, dtype=np.int64)
            
            return np.vstack(features_list), np.array(timestamps_list, dtype=np.int64)
    
    def compute_inter_arrival_times(self, timestamps: np.ndarray) -> np.ndarray:
        """Compute inter-arrival times in milliseconds."""
        if len(timestamps) < 2:
            return np.zeros(len(timestamps), dtype=np.float32)
        
        sorted_ts = np.sort(timestamps)
        diffs = np.diff(sorted_ts).astype(np.float32) / 1_000_000.0  # ns to ms
        return np.concatenate([[0.0], diffs])


class TemporalAttentionPredictor:
    """
    Temporal Point Process attention mechanism for order arrival prediction.
    Predicts probability of liquidity sweep in next N milliseconds.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.buffer = TemporalAttentionBuffer()
        self.session: Optional[ort.InferenceSession] = None
        self._cumulative_volume = 0.0
        self._last_volume_reset = time.time()
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> None:
        """Load pre-trained temporal attention model exported to ONNX."""
        self.session = ort.InferenceSession(
            model_path,
            sess_options=SESSION_OPTIONS,
            providers=['CPUExecutionProvider']
        )
    
    def process_order(
        self,
        timestamp_ns: int,
        side: int,
        volume: float,
        price: float,
        vpin: float,
        spread_bps: float
    ) -> None:
        """Process an incoming aggressive order."""
        # Reset cumulative volume every minute
        current_time = time.time()
        if current_time - self._last_volume_reset > 60.0:
            self._cumulative_volume = 0.0
            self._last_volume_reset = current_time
        
        self._cumulative_volume += volume
        
        # Calculate inter-arrival time
        if self.buffer._count > 0:
            inter_arrival_ms = (timestamp_ns - self.buffer._last_timestamp) / 1_000_000.0
        else:
            inter_arrival_ms = 0.0
        
        # Add to buffer
        self.buffer.add_event(
            timestamp_ns, side, volume, price,
            inter_arrival_ms, self._cumulative_volume, vpin, spread_bps
        )
    
    def predict_sweep_probability(
        self,
        horizon_ms: float = 100.0,
        lookback_window_ms: float = 5000.0
    ) -> Tuple[float, float, float]:
        """
        Predict probability of liquidity sweep in next N milliseconds.
        
        Args:
            horizon_ms: Prediction horizon
            lookback_window_ms: Historical window for attention
            
        Returns:
            sweep_prob: Probability of any sweep
            buy_sweep_prob: Probability of upward sweep (asks hit)
            sell_sweep_prob: Probability of downward sweep (bids hit)
        """
        # Get recent events
        features, timestamps = self.buffer.get_recent_events(lookback_window_ms)
        
        if len(features) < 5:
            # Not enough data, return neutral probabilities
            return 0.5, 0.25, 0.25
        
        # Compute inter-arrival times
        inter_arrivals = self.buffer.compute_inter_arrival_times(timestamps)
        
        if self.session is None:
            return self._heuristic_prediction(features, inter_arrivals, horizon_ms)
        
        # Prepare inputs for ONNX model
        # Expected: features, inter_arrivals, horizon, sequence_length
        seq_len = min(len(features), 100)  # Cap sequence length
        
        inputs = {
            'features': features[:seq_len].astype(np.float32),
            'inter_arrival_times': inter_arrivals[:seq_len].astype(np.float32),
            'horizon': np.array([horizon_ms], dtype=np.float32),
            'sequence_length': np.array([seq_len], dtype=np.int64)
        }
        
        # Pad if necessary (model expects fixed size)
        max_seq = 100
        if seq_len < max_seq:
            pad_features = np.zeros((max_seq - seq_len, features.shape[1]), dtype=np.float32)
            pad_inter = np.zeros(max_seq - seq_len, dtype=np.float32)
            inputs['features'] = np.vstack([inputs['features'], pad_features])
            inputs['inter_arrival_times'] = np.concatenate([inputs['inter_arrival_times'], pad_inter])
        
        # Run inference
        outputs = self.session.run(None, inputs)
        
        # Output: [sweep_prob, buy_sweep_prob, sell_sweep_prob, expected_volume, confidence]
        sweep_prob = float(outputs[0][0])
        buy_sweep_prob = float(outputs[0][1])
        sell_sweep_prob = float(outputs[0][2])
        
        return sweep_prob, buy_sweep_prob, sell_sweep_prob
    
    def _heuristic_prediction(
        self,
        features: np.ndarray,
        inter_arrivals: np.ndarray,
        horizon_ms: float
    ) -> Tuple[float, float, float]:
        """Fallback heuristic based on temporal patterns."""
        if len(features) < 2:
            return 0.5, 0.25, 0.25
        
        # Analyze recent order flow
        recent_sides = features[-20:, 0]  # Last 20 sides
        recent_volumes = np.expm1(features[-20:, 1])  # Inverse log
        recent_inter = inter_arrivals[-20:]
        
        # Calculate directional pressure
        buy_pressure = np.sum(recent_volumes[recent_sides > 0])
        sell_pressure = np.sum(recent_volumes[recent_sides < 0])
        total_pressure = buy_pressure + sell_pressure + 1e-8
        
        # Calculate arrival rate acceleration
        if len(recent_inter) > 10:
            early_rate = np.mean(recent_inter[-10:-5])
            late_rate = np.mean(recent_inter[-5:])
            acceleration = (early_rate - late_rate) / (early_rate + 1e-8)
        else:
            acceleration = 0.0
        
        # Base sweep probability from pressure imbalance
        pressure_imbalance = abs(buy_pressure - sell_pressure) / total_pressure
        base_prob = 0.3 + 0.4 * pressure_imbalance
        
        # Adjust for acceleration (faster arrivals = higher sweep prob)
        sweep_prob = np.clip(base_prob + 0.2 * acceleration, 0.0, 1.0)
        
        # Directional probabilities
        if buy_pressure > sell_pressure:
            buy_sweep_prob = sweep_prob * (0.5 + 0.3 * pressure_imbalance)
            sell_sweep_prob = sweep_prob * (0.5 - 0.3 * pressure_imbalance)
        else:
            buy_sweep_prob = sweep_prob * (0.5 - 0.3 * pressure_imbalance)
            sell_sweep_prob = sweep_prob * (0.5 + 0.3 * pressure_imbalance)
        
        return sweep_prob, np.clip(buy_sweep_prob, 0.0, 1.0), np.clip(sell_sweep_prob, 0.0, 1.0)
    
    def get_attention_weights(
        self,
        horizon_ms: float = 100.0
    ) -> Optional[np.ndarray]:
        """
        Extract attention weights for interpretability.
        Shows which past events are most predictive.
        """
        if self.session is None:
            return None
        
        features, timestamps = self.buffer.get_recent_events(5000.0)
        if len(features) < 2:
            return None
        
        seq_len = min(len(features), 100)
        inter_arrivals = self.buffer.compute_inter_arrival_times(timestamps)
        
        inputs = {
            'features': features[:seq_len].astype(np.float32),
            'inter_arrival_times': inter_arrivals[:seq_len].astype(np.float32),
            'horizon': np.array([horizon_ms], dtype=np.float32),
            'sequence_length': np.array([seq_len], dtype=np.int64),
            'return_attention': np.array([1], dtype=np.int64)
        }
        
        # Pad if necessary
        max_seq = 100
        if seq_len < max_seq:
            pad_features = np.zeros((max_seq - seq_len, features.shape[1]), dtype=np.float32)
            pad_inter = np.zeros(max_seq - seq_len, dtype=np.float32)
            inputs['features'] = np.vstack([inputs['features'], pad_features])
            inputs['inter_arrival_times'] = np.concatenate([inputs['inter_arrival_times'], pad_inter])
        
        outputs = self.session.run(None, inputs)
        return outputs[1]  # Attention weights


# Global singleton instance
_temporal_instance: Optional[TemporalAttentionPredictor] = None
_temporal_lock = threading.Lock()


def get_temporal_predictor(model_path: Optional[str] = None) -> TemporalAttentionPredictor:
    """Thread-safe singleton access to temporal attention predictor."""
    global _temporal_instance
    
    with _temporal_lock:
        if _temporal_instance is None:
            _temporal_instance = TemporalAttentionPredictor(model_path)
        elif model_path and _temporal_instance.session is None:
            _temporal_instance.load_model(model_path)
        
        return _temporal_instance


if __name__ == "__main__":
    # Demo usage
    np.random.seed(42)
    
    predictor = TemporalAttentionPredictor()
    
    # Simulate order flow
    base_time = int(time.time() * 1_000_000_000)
    cumulative_vol = 0.0
    
    for i in range(50):
        timestamp = base_time + i * 10_000_000  # 10ms apart
        side = np.random.choice([-1, 1])
        volume = np.random.exponential(5.0)
        price = 50000.0 + np.random.randn() * 10
        vpin = np.random.beta(2, 5)
        spread = 5.0 + np.random.exponential(2.0)
        
        predictor.process_order(timestamp, side, volume, price, vpin, spread)
    
    # Make prediction
    sweep_prob, buy_prob, sell_prob = predictor.predict_sweep_probability(horizon_ms=100)
    print(f"Sweep probability: {sweep_prob:.4f}")
    print(f"Buy sweep: {buy_prob:.4f}, Sell sweep: {sell_prob:.4f}")
    
    # Get attention weights if available
    attention = predictor.get_attention_weights()
    if attention is not None:
        print(f"Attention weights shape: {attention.shape}")
