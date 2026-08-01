"""
Chapter 4: Latency Arbitrage & Queue Position ML Prediction
queue_predictor.py - XGBoost model to predict time-to-fill for passive limit orders based on L2 depletion rates
"""

import numpy as np
from numba import njit, prange
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass
from collections import deque

# Try to import xgboost, provide fallback if not available
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


@dataclass
class QueuePositionFeatures:
    """Features for queue position prediction"""
    # Order book features
    bid_depth_1: float
    bid_depth_2: float
    bid_depth_3: float
    ask_depth_1: float
    ask_depth_2: float
    ask_depth_3: float
    
    # Depletion rates
    bid_depletion_rate: float
    ask_depletion_rate: float
    net_depletion_rate: float
    
    # Queue metrics
    queue_position: int
    order_size: float
    relative_size: float  # Size vs depth at price level
    
    # Price dynamics
    mid_price: float
    spread: float
    price_momentum: float
    
    # Time features
    time_of_day: float
    day_of_week: float
    
    # Volatility
    recent_volatility: float
    
    def to_array(self) -> np.ndarray:
        """Convert features to numpy array for model input."""
        return np.array([
            self.bid_depth_1, self.bid_depth_2, self.bid_depth_3,
            self.ask_depth_1, self.ask_depth_2, self.ask_depth_3,
            self.bid_depletion_rate, self.ask_depletion_rate, self.net_depletion_rate,
            self.queue_position, self.order_size, self.relative_size,
            self.mid_price, self.spread, self.price_momentum,
            self.time_of_day, self.day_of_week, self.recent_volatility
        ], dtype=np.float64)


@njit(cache=True, nogil=True)
def calculate_depletion_rate(
    depth_history: np.ndarray,
    window: int = 10
) -> float:
    """
    Calculate rate of order book depth depletion.
    
    Args:
        depth_history: Historical depth values (most recent last)
        window: Lookback window
    
    Returns:
        Depletion rate (negative = decreasing depth, positive = increasing)
    """
    n = len(depth_history)
    if n < 2:
        return 0.0
    
    effective_window = min(window, n)
    
    # Linear regression slope
    sum_x = 0.0
    sum_y = 0.0
    sum_xy = 0.0
    sum_xx = 0.0
    
    for i in range(effective_window):
        idx = n - effective_window + i
        x = float(i)
        y = depth_history[idx]
        
        sum_x += x
        sum_y += y
        sum_xy += x * y
        sum_xx += x * x
    
    denom = effective_window * sum_xx - sum_x * sum_x
    
    if abs(denom) < 1e-10:
        return 0.0
    
    slope = (effective_window * sum_xy - sum_x * sum_y) / denom
    
    return slope


@njit(cache=True, nogil=True)
def estimate_queue_position(
    order_size: float,
    depth_at_price: float,
    recent_trade_volume: float
) -> int:
    """
    Estimate queue position for a limit order.
    
    Args:
        order_size: Size of our order
        depth_at_price: Total depth at our price level
        recent_trade_volume: Recent aggressive trade volume
    
    Returns:
        Estimated queue position (0 = front of queue)
    """
    if depth_at_price <= 0:
        return 0
    
    # Simple estimation: assume pro-rata or FIFO
    # Position is roughly proportional to our size vs total depth
    ratio = order_size / depth_at_price
    
    # Adjust for recent trades that may have cleared queue ahead
    adjusted_position = int((1.0 - ratio) * depth_at_price * 0.5)
    
    return max(0, adjusted_position - int(recent_trade_volume))


@njit(cache=True, nogil=True)
def calculate_fill_probability(
    queue_position: int,
    depletion_rate: float,
    spread: float,
    volatility: float,
    time_horizon: float = 60.0
) -> float:
    """
    Calculate probability of order fill within time horizon.
    
    Args:
        queue_position: Position in queue
        depletion_rate: Rate at which queue is being depleted
        spread: Current bid-ask spread
        volatility: Recent price volatility
        time_horizon: Time horizon in seconds
    
    Returns:
        Fill probability [0, 1]
    """
    if queue_position <= 0:
        return 1.0
    
    # Base probability from queue position and depletion
    if depletion_rate < 0:
        # Queue is building, lower fill probability
        base_prob = 0.3
    else:
        # Queue is depleting
        time_to_fill = queue_position / max(depletion_rate, 1e-10)
        
        if time_to_fill <= time_horizon:
            base_prob = 0.9
        elif time_to_fill <= time_horizon * 2:
            base_prob = 0.6
        else:
            base_prob = 0.3
    
    # Adjust for spread (wider spread = less likely to fill)
    spread_factor = 1.0 / (1.0 + spread * 100)
    
    # Adjust for volatility (higher vol = more likely to fill)
    vol_factor = 1.0 + volatility * 10
    
    probability = base_prob * spread_factor * vol_factor
    
    return np.clip(probability, 0.0, 1.0)


class QueuePositionPredictor:
    """
    ML-based predictor for limit order fill times.
    Uses gradient boosting to predict time-to-fill based on L2 features.
    """
    
    def __init__(
        self,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        use_gpu: bool = False
    ):
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.use_gpu = use_gpu
        
        # Model state
        self._model = None
        self._is_trained = False
        
        # Feature history for real-time calculation
        self._bid_depth_history: deque = deque(maxlen=100)
        self._ask_depth_history: deque = deque(maxlen=100)
        self._trade_history: deque = deque(maxlen=1000)
        
        # Calibration data
        self._fill_times: List[float] = []
        self._feature_samples: List[np.ndarray] = []
    
    def extract_features(
        self,
        bid_depths: np.ndarray,
        ask_depths: np.ndarray,
        mid_price: float,
        spread: float,
        order_size: float,
        side: int  # 1 = bid, -1 = ask
    ) -> QueuePositionFeatures:
        """
        Extract features from current order book state.
        
        Args:
            bid_depths: Depth at first 3 bid levels
            ask_depths: Depth at first 3 ask levels
            mid_price: Current mid price
            spread: Bid-ask spread
            order_size: Our order size
            side: Order side (1=bid, -1=ask)
        
        Returns:
            QueuePositionFeatures object
        """
        # Update history
        self._bid_depth_history.append(bid_depths[0])
        self._ask_depth_history.append(ask_depths[0])
        
        # Calculate depletion rates
        bid_hist = np.array(list(self._bid_depth_history))
        ask_hist = np.array(list(self._ask_depth_history))
        
        bid_depletion = calculate_depletion_rate(bid_hist)
        ask_depletion = calculate_depletion_rate(ask_hist)
        net_depletion = bid_depletion - ask_depletion
        
        # Estimate queue position
        depth_at_price = bid_depths[0] if side == 1 else ask_depths[0]
        recent_volume = sum(t[1] for t in list(self._trade_history)[-100:])
        
        queue_pos = estimate_queue_position(order_size, depth_at_price, recent_volume)
        
        # Calculate relative size
        relative_size = order_size / max(depth_at_price, 1e-10)
        
        # Price momentum (simplified)
        price_momentum = 0.0
        if len(self._trade_history) > 10:
            recent_prices = [t[0] for t in list(self._trade_history)[-10:]]
            price_momentum = (recent_prices[-1] - recent_prices[0]) / max(recent_prices[0], 1e-10)
        
        # Recent volatility
        recent_volatility = 0.0
        if len(self._trade_history) > 100:
            recent_returns = np.diff([t[0] for t in list(self._trade_history)[-100:]])
            if len(recent_returns) > 0:
                recent_volatility = np.std(recent_returns)
        
        # Time features (would be set from actual timestamp)
        time_of_day = 12.0  # Default noon
        day_of_week = 0.0   # Default Monday
        
        return QueuePositionFeatures(
            bid_depth_1=bid_depths[0],
            bid_depth_2=bid_depths[1] if len(bid_depths) > 1 else 0.0,
            bid_depth_3=bid_depths[2] if len(bid_depths) > 2 else 0.0,
            ask_depth_1=ask_depths[0],
            ask_depth_2=ask_depths[1] if len(ask_depths) > 1 else 0.0,
            ask_depth_3=ask_depths[2] if len(ask_depths) > 2 else 0.0,
            bid_depletion_rate=bid_depletion,
            ask_depletion_rate=ask_depletion,
            net_depletion_rate=net_depletion,
            queue_position=queue_pos,
            order_size=order_size,
            relative_size=relative_size,
            mid_price=mid_price,
            spread=spread,
            price_momentum=price_momentum,
            time_of_day=time_of_day,
            day_of_week=day_of_week,
            recent_volatility=recent_volatility
        )
    
    def record_fill(
        self,
        features: QueuePositionFeatures,
        time_to_fill: float
    ):
        """Record a fill event for training/calibration."""
        self._feature_samples.append(features.to_array())
        self._fill_times.append(time_to_fill)
    
    def record_trade(
        self,
        price: float,
        volume: float,
        side: int
    ):
        """Record a trade for feature calculation."""
        self._trade_history.append((price, volume, side))
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_split: float = 0.2
    ) -> Dict[str, float]:
        """
        Train the prediction model.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Time-to-fill targets (seconds)
            validation_split: Fraction for validation
        
        Returns:
            Training metrics
        """
        if not XGB_AVAILABLE:
            return {'error': 'XGBoost not available'}
        
        n_samples = len(X)
        n_train = int(n_samples * (1 - validation_split))
        
        X_train, X_val = X[:n_train], X[n_train:]
        y_train, y_val = y[:n_train], y[n_train:]
        
        # Create DMatrix for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # Model parameters
        params = {
            'objective': 'reg:squarederror',
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'tree_method': 'hist' if not self.use_gpu else 'gpu_hist',
            'device': 'cuda' if self.use_gpu else 'cpu'
        }
        
        # Train model
        self._model = xgb.train(
            params,
            dtrain,
            num_boost_round=self.n_estimators,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=10,
            verbose_eval=False
        )
        
        # Calculate metrics
        train_pred = self._model.predict(dtrain)
        val_pred = self._model.predict(dval)
        
        train_mse = np.mean((train_pred - y_train) ** 2)
        val_mse = np.mean((val_pred - y_val) ** 2)
        
        self._is_trained = True
        
        return {
            'train_mse': float(train_mse),
            'val_mse': float(val_mse),
            'n_samples': n_samples,
            'n_features': X.shape[1]
        }
    
    def predict_time_to_fill(
        self,
        features: QueuePositionFeatures
    ) -> Tuple[float, float]:
        """
        Predict time to fill for an order.
        
        Args:
            features: Current order book features
        
        Returns:
            Tuple of (predicted_time_seconds, confidence)
        """
        X = features.to_array().reshape(1, -1)
        
        if self._model is not None and self._is_trained:
            if XGB_AVAILABLE:
                dmatrix = xgb.DMatrix(X)
                prediction = self._model.predict(dmatrix)[0]
                
                # Estimate confidence based on feature similarity to training data
                confidence = self._estimate_confidence(X)
                
                return max(0.0, prediction), confidence
        
        # Fallback: heuristic-based prediction
        fill_prob = calculate_fill_probability(
            features.queue_position,
            features.net_depletion_rate,
            features.spread,
            features.recent_volatility
        )
        
        # Rough time estimate
        if fill_prob > 0.7:
            predicted_time = 30.0  # ~30 seconds
            confidence = 0.5
        elif fill_prob > 0.4:
            predicted_time = 120.0  # ~2 minutes
            confidence = 0.4
        else:
            predicted_time = 600.0  # ~10 minutes
            confidence = 0.3
        
        return predicted_time, confidence
    
    def _estimate_confidence(self, X: np.ndarray) -> float:
        """Estimate prediction confidence based on feature space density."""
        if len(self._feature_samples) == 0:
            return 0.5
        
        # Calculate distance to nearest training samples
        samples = np.array(self._feature_samples)
        
        # Simplified: just check if features are in reasonable range
        feature_ranges = np.ptp(samples, axis=0)
        feature_means = np.mean(samples, axis=0)
        
        normalized_dist = np.abs(X[0] - feature_means) / (feature_ranges + 1e-10)
        avg_normalized_dist = np.mean(normalized_dist)
        
        # Convert distance to confidence
        confidence = 1.0 / (1.0 + avg_normalized_dist)
        
        return float(confidence)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model."""
        if self._model is None or not XGB_AVAILABLE:
            return {}
        
        importance = self._model.get_score(importance_type='gain')
        
        feature_names = [
            'bid_depth_1', 'bid_depth_2', 'bid_depth_3',
            'ask_depth_1', 'ask_depth_2', 'ask_depth_3',
            'bid_depletion', 'ask_depletion', 'net_depletion',
            'queue_position', 'order_size', 'relative_size',
            'mid_price', 'spread', 'momentum',
            'time_of_day', 'day_of_week', 'volatility'
        ]
        
        result = {}
        for i, name in enumerate(feature_names):
            key = f'f{i}'
            result[name] = importance.get(key, 0.0)
        
        return result


# Module convenience functions
def create_queue_predictor(
    max_depth: int = 6,
    learning_rate: float = 0.1
) -> QueuePositionPredictor:
    """Factory function to create queue position predictor."""
    return QueuePositionPredictor(max_depth, learning_rate)


def quick_fill_prediction(
    queue_position: int,
    depletion_rate: float,
    spread: float,
    volatility: float
) -> float:
    """Quick heuristic fill time prediction without full model."""
    features = QueuePositionFeatures(
        bid_depth_1=100.0, bid_depth_2=200.0, bid_depth_3=300.0,
        ask_depth_1=100.0, ask_depth_2=200.0, ask_depth_3=300.0,
        bid_depletion_rate=depletion_rate,
        ask_depletion_rate=0.0,
        net_depletion_rate=depletion_rate,
        queue_position=queue_position,
        order_size=1.0,
        relative_size=0.01,
        mid_price=50000.0,
        spread=spread,
        price_momentum=0.0,
        time_of_day=12.0,
        day_of_week=0.0,
        recent_volatility=volatility
    )
    
    predictor = QueuePositionPredictor()
    time_pred, _ = predictor.predict_time_to_fill(features)
    
    return time_pred
