# Order Book Imbalance Features Calculator
# Generates multi-level imbalance matrices for CNN/Transformer models

from __future__ import annotations
import logging
import numpy as np
from typing import Optional, Tuple

log = logging.getLogger(__name__)


class OrderBookImbalanceCalculator:
    """
    Calculate multi-level order book imbalance features.
    Generates 2D tensors formatted for Time-Series Transformer or CNN models.
    
    Optimized for HFT with pre-allocated buffers and zero-copy operations.
    """

    def __init__(
        self,
        n_levels: int = 10,  # Top N levels of order book
        history_length: int = 100,  # Time steps of history
        dtype: np.dtype = np.float64,
    ) -> None:
        self.n_levels = n_levels
        self.history_length = history_length
        self.dtype = dtype
        
        # Pre-allocated buffers for order book state
        self._bid_prices = np.zeros(n_levels, dtype=dtype)
        self._ask_prices = np.zeros(n_levels, dtype=dtype)
        self._bid_volumes = np.zeros(n_levels, dtype=dtype)
        self._ask_volumes = np.zeros(n_levels, dtype=dtype)
        
        # History buffer for tensor generation (circular)
        self._imbalance_history = np.zeros(
            (history_length, n_levels * 4),  # 4 features per level
            dtype=dtype,
            order='C'
        )
        self._history_head = 0
        self._history_count = 0
        
        # Cached tensors
        self._current_imbalance_tensor: Optional[np.ndarray] = None
        self._prev_mid_price: Optional[float] = None

    def update_order_book(
        self,
        bid_prices: np.ndarray,
        bid_volumes: np.ndarray,
        ask_prices: np.ndarray,
        ask_volumes: np.ndarray,
    ) -> None:
        """
        Update order book state with new L2 data.
        Expects arrays of length n_levels.
        """
        n = min(len(bid_prices), len(ask_prices), self.n_levels)
        
        self._bid_prices[:n] = bid_prices[:n]
        self._bid_volumes[:n] = bid_volumes[:n]
        self._ask_prices[:n] = ask_prices[:n]
        self._ask_volumes[:n] = ask_volumes[:n]
        
        # Compute and store imbalance features
        self._compute_and_store_imbalance()

    def _compute_and_store_imbalance(self) -> None:
        """Compute imbalance features and store in history buffer."""
        # Level-wise volume imbalance: (bid_vol - ask_vol) / (bid_vol + ask_vol)
        total_vol = self._bid_volumes + self._ask_volumes + 1e-9
        volume_imbalance = (self._bid_volumes - self._ask_volumes) / total_vol
        
        # Price imbalance relative to mid
        mid_price = (self._bid_prices[0] + self._ask_prices[0]) / 2
        if self._prev_mid_price is not None:
            mid_price_return = (mid_price - self._prev_mid_price) / (self._prev_mid_price + 1e-9)
        else:
            mid_price_return = 0.0
        self._prev_mid_price = mid_price
        
        # Bid-ask spread
        spread = self._ask_prices - self._bid_prices
        spread_pct = spread / (mid_price + 1e-9)
        
        # Weighted imbalance (weight by inverse distance from mid)
        distances = np.arange(1, self.n_levels + 1, dtype=self.dtype)
        weights = 1.0 / distances
        weighted_bid_vol = np.sum(self._bid_volumes * weights)
        weighted_ask_vol = np.sum(self._ask_volumes * weights)
        weighted_imbalance = (weighted_bid_vol - weighted_ask_vol) / (weighted_bid_vol + weighted_ask_vol + 1e-9)
        
        # Construct feature vector: [vol_imbalance, spread_pct, mid_return, weighted_imbalance] per level
        features = np.zeros(self.n_levels * 4, dtype=self.dtype)
        
        for i in range(self.n_levels):
            base_idx = i * 4
            features[base_idx] = volume_imbalance[i]
            features[base_idx + 1] = spread_pct[i] if i == 0 else spread_pct[0] * (1.0 / (i + 1))
            features[base_idx + 2] = mid_price_return
            features[base_idx + 3] = weighted_imbalance
        
        # Store in circular buffer
        self._imbalance_history[self._history_head] = features
        self._history_head = (self._history_head + 1) % self.history_length
        self._history_count = min(self._history_count + 1, self.history_length)
        
        # Update cached tensor
        self._current_imbalance_tensor = features.copy()

    def get_imbalance_tensor(self, lookback: Optional[int] = None) -> np.ndarray:
        """
        Get imbalance tensor for ML model input.
        
        Args:
            lookback: Number of time steps to include. If None, returns single timestep.
            
        Returns:
            Tensor of shape (lookback, n_levels, 4) or (n_levels, 4)
        """
        if lookback is None:
            if self._current_imbalance_tensor is None:
                return np.zeros((self.n_levels, 4), dtype=self.dtype)
            return self._current_imbalance_tensor.reshape(self.n_levels, 4)
        
        lookback = min(lookback, self._history_count)
        
        if lookback == 0:
            return np.zeros((0, self.n_levels, 4), dtype=self.dtype)
        
        # Extract from circular buffer
        start_idx = (self._history_head - lookback) % self.history_length
        
        if start_idx < self._history_head:
            data = self._imbalance_history[start_idx:self._history_head]
        else:
            # Wrap around
            data = np.vstack([
                self._imbalance_history[start_idx:],
                self._imbalance_history[:self._history_head]
            ])
        
        # Reshape to (lookback, n_levels, 4)
        return data.reshape(lookback, self.n_levels, 4)

    def get_imbalance_features(self) -> dict[str, float]:
        """
        Generate scalar imbalance features for traditional ML models.
        """
        if self._current_imbalance_tensor is None:
            return {}
        
        features_flat = self._current_imbalance_tensor
        n = self.n_levels
        
        # Aggregate features
        vol_imb = features_flat[0::4]  # Volume imbalance per level
        spread_pcts = features_flat[1::4]
        weighted_imb = features_flat[3::4]
        
        return {
            "imbalance_level_0": float(vol_imb[0]),
            "imbalance_mean": float(np.mean(vol_imb)),
            "imbalance_std": float(np.std(vol_imb)),
            "imbalance_trend": float(np.mean(vol_imb[:3]) - np.mean(vol_imb[-3:])),
            "spread_pct": float(spread_pcts[0]),
            "weighted_imbalance": float(weighted_imb[0]),
            "total_bid_vol": float(np.sum(self._bid_volumes)),
            "total_ask_vol": float(np.sum(self._ask_volumes)),
            "volume_ratio": float(np.sum(self._bid_volumes) / (np.sum(self._ask_volumes) + 1e-9)),
        }

    def get_cnn_ready_tensor(self, channels: int = 4) -> np.ndarray:
        """
        Get tensor formatted specifically for CNN input.
        Shape: (channels, history_length, n_levels)
        """
        lookback = min(50, self._history_count)  # Default 50 timesteps for CNN
        tensor = self.get_imbalance_tensor(lookback)  # (lookback, n_levels, 4)
        
        # Rearrange to (channels, lookback, n_levels)
        if tensor.shape[0] == 0:
            return np.zeros((channels, 1, self.n_levels), dtype=self.dtype)
        
        # Transpose: (lookback, n_levels, 4) -> (4, lookback, n_levels)
        return tensor.T.reshape(channels, lookback, self.n_levels)

    def get_transformer_ready_tensor(self) -> np.ndarray:
        """
        Get tensor formatted for Transformer input.
        Shape: (sequence_length, feature_dim) where feature_dim = n_levels * 4
        """
        lookback = min(100, self._history_count)
        tensor = self.get_imbalance_tensor(lookback)  # (lookback, n_levels, 4)
        
        # Flatten last two dimensions: (lookback, n_levels * 4)
        return tensor.reshape(lookback, self.n_levels * 4)

    def reset(self) -> None:
        """Reset all state."""
        self._bid_prices.fill(0.0)
        self._ask_prices.fill(0.0)
        self._bid_volumes.fill(0.0)
        self._ask_volumes.fill(0.0)
        self._imbalance_history.fill(0.0)
        self._history_head = 0
        self._history_count = 0
        self._current_imbalance_tensor = None
        self._prev_mid_price = None
        log.info("OrderBookImbalanceCalculator reset")


def create_imbalance_batch(
    calculator: OrderBookImbalanceCalculator,
    batch_size: int = 32,
    sequence_length: int = 50,
) -> np.ndarray:
    """
    Create a batch of imbalance tensors for model inference.
    Returns shape: (batch_size, sequence_length, n_levels * 4)
    """
    batch = []
    for _ in range(batch_size):
        tensor = calculator.get_transformer_ready_tensor()
        if len(tensor) < sequence_length:
            # Pad with zeros
            padding = np.zeros(
                (sequence_length - len(tensor), tensor.shape[1]),
                dtype=tensor.dtype
            )
            tensor = np.vstack([padding, tensor])
        elif len(tensor) > sequence_length:
            tensor = tensor[-sequence_length:]
        batch.append(tensor)
    
    return np.stack(batch, axis=0)
