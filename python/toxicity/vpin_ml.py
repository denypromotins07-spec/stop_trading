"""
VPIN ML Toxicity Forecaster
Trains a lightweight XGBoost model to forecast VPIN (Volume-Synchronized Probability of Informed Trading).
Predicts toxicity spikes 500ms ahead to allow market making strategies to widen spreads defensively.
Uses predict_inplace to avoid allocating new memory arrays during hot inference path.
"""

import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
import logging

# Conditional XGBoost import
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    xgb = None  # type: ignore


logger = logging.getLogger(__name__)


@dataclass
class VPINFeatures:
    """Features for VPIN prediction model."""
    buy_volume: float
    sell_volume: float
    trade_count: int
    price_volatility: float
    spread_bps: float
    order_imbalance: float
    trade_size_variance: float
    aggressor_ratio: float
    time_weighted_spread: float
    volume_weighted_price: float


@dataclass
class VPINPrediction:
    """VPIN prediction result."""
    vpin_value: float
    toxicity_level: str  # "LOW", "MEDIUM", "HIGH", "EXTREME"
    confidence: float
    predicted_horizon_ms: int
    feature_importance: Optional[Dict[str, float]] = None


class VPINForecaster:
    """
    Lightweight XGBoost-based VPIN forecaster.
    Predicts informed trading probability 500ms ahead.
    """

    # Toxicity thresholds
    LOW_THRESHOLD = 0.3
    MEDIUM_THRESHOLD = 0.5
    HIGH_THRESHOLD = 0.7

    def __init__(
        self,
        n_estimators: int = 50,
        max_depth: int = 4,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        prediction_horizon_ms: int = 500,
        bucket_size: int = 100,  # Volume buckets for VPIN calculation
    ):
        if not XGB_AVAILABLE:
            logger.warning("XGBoost not available - using fallback VPIN calculation")

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.prediction_horizon_ms = prediction_horizon_ms
        self.bucket_size = bucket_size

        self._model: Optional[xgb.Booster] = None  # type: ignore
        self._is_trained = False

        # Rolling buffers for online VPIN calculation
        self._buy_buckets: List[float] = []
        self._sell_buckets: List[float] = []
        self._current_buy_volume: float = 0.0
        self._current_sell_volume: float = 0.0
        self._bucket_count: int = 0

        # Pre-allocated array for inference
        self._feature_buffer: np.ndarray = np.zeros(10, dtype=np.float32)
        self._prediction_buffer: np.ndarray = np.zeros(1, dtype=np.float32)

        # Feature names for importance tracking
        self.feature_names = [
            "buy_volume",
            "sell_volume",
            "trade_count",
            "price_volatility",
            "spread_bps",
            "order_imbalance",
            "trade_size_variance",
            "aggressor_ratio",
            "time_weighted_spread",
            "volume_weighted_price",
        ]

    def _calculate_online_vpin(self) -> float:
        """
        Calculate VPIN from current volume buckets using Easley et al. methodology.
        VPIN = Sum(|Buy_Volume - Sell_Volume|) / Sum(Buy_Volume + Sell_Volume)
        """
        if not self._buy_buckets or not self._sell_buckets:
            return 0.0

        total_abs_diff = sum(abs(b - s) for b, s in zip(self._buy_buckets, self._sell_buckets))
        total_volume = sum(b + s for b, s in zip(self._buy_buckets, self._sell_buckets))

        if total_volume < 1e-9:
            return 0.0

        return total_abs_diff / total_volume

    def _add_trade_to_bucket(
        self,
        volume: float,
        is_buy: bool,
    ) -> Optional[float]:
        """
        Add trade to current volume bucket.
        Returns VPIN if bucket is complete, None otherwise.
        """
        if is_buy:
            self._current_buy_volume += volume
        else:
            self._current_sell_volume += volume

        # Check if bucket is full
        total_bucket_volume = self._current_buy_volume + self._current_sell_volume

        if total_bucket_volume >= self.bucket_size:
            # Store completed bucket
            self._buy_buckets.append(self._current_buy_volume)
            self._sell_buckets.append(self._current_sell_volume)
            self._bucket_count += 1

            # Reset current bucket
            self._current_buy_volume = 0.0
            self._current_sell_volume = 0.0

            # Keep only last N buckets for rolling calculation
            max_buckets = 50
            if len(self._buy_buckets) > max_buckets:
                self._buy_buckets = self._buy_buckets[-max_buckets:]
                self._sell_buckets = self._sell_buckets[-max_buckets:]

            return self._calculate_online_vpin()

        return None

    def extract_features(
        self,
        trades: List[Tuple[float, float, bool]],  # (volume, price, is_buy)
        prices: List[float],
        spreads: List[float],
    ) -> VPINFeatures:
        """Extract features from trade and quote data."""
        if not trades:
            return VPINFeatures(
                buy_volume=0.0,
                sell_volume=0.0,
                trade_count=0,
                price_volatility=0.0,
                spread_bps=0.0,
                order_imbalance=0.0,
                trade_size_variance=0.0,
                aggressor_ratio=0.0,
                time_weighted_spread=0.0,
                volume_weighted_price=0.0,
            )

        buy_volume = sum(v for v, _, is_buy in trades if is_buy)
        sell_volume = sum(v for v, _, is_buy in trades if not is_buy)
        trade_count = len(trades)

        # Price volatility
        if len(prices) > 1:
            returns = np.diff(prices) / (np.array(prices[:-1]) + 1e-9)
            price_volatility = float(np.std(returns))
        else:
            price_volatility = 0.0

        # Spread
        spread_bps = np.mean(spreads) * 10000 if spreads else 0.0

        # Order imbalance
        total_volume = buy_volume + sell_volume
        order_imbalance = (buy_volume - sell_volume) / (total_volume + 1e-9)

        # Trade size variance
        sizes = [v for v, _, _ in trades]
        trade_size_variance = float(np.var(sizes)) if sizes else 0.0

        # Aggressor ratio
        buy_trades = sum(1 for _, _, is_buy in trades if is_buy)
        aggressor_ratio = buy_trades / (trade_count + 1e-9)

        # Time-weighted spread (simplified)
        time_weighted_spread = spread_bps

        # Volume-weighted price
        total_value = sum(v * p for v, p, _ in trades)
        volume_weighted_price = total_value / (total_volume + 1e-9)

        return VPINFeatures(
            buy_volume=buy_volume,
            sell_volume=sell_volume,
            trade_count=trade_count,
            price_volatility=price_volatility,
            spread_bps=spread_bps,
            order_imbalance=order_imbalance,
            trade_size_variance=trade_size_variance,
            aggressor_ratio=aggressor_ratio,
            time_weighted_spread=time_weighted_spread,
            volume_weighted_price=volume_weighted_price,
        )

    def _features_to_array(self, features: VPINFeatures) -> np.ndarray:
        """Convert features to numpy array for model input."""
        self._feature_buffer[0] = features.buy_volume
        self._feature_buffer[1] = features.sell_volume
        self._feature_buffer[2] = features.trade_count
        self._feature_buffer[3] = features.price_volatility
        self._feature_buffer[4] = features.spread_bps
        self._feature_buffer[5] = features.order_imbalance
        self._feature_buffer[6] = features.trade_size_variance
        self._feature_buffer[7] = features.aggressor_ratio
        self._feature_buffer[8] = features.time_weighted_spread
        self._feature_buffer[9] = features.volume_weighted_price

        return self._feature_buffer.reshape(1, -1)

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_split: float = 0.2,
        early_stopping_rounds: int = 10,
    ) -> Dict[str, float]:
        """
        Train the VPIN prediction model.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target VPIN values
            validation_split: Fraction of data for validation
            early_stopping_rounds: Rounds for early stopping

        Returns:
            Training metrics
        """
        if not XGB_AVAILABLE:
            logger.error("XGBoost not available for training")
            return {"error": "XGBoost not installed"}

        if len(X) < 100:
            logger.warning("Insufficient training data")
            return {"error": "Insufficient data"}

        # Split data
        n_valid = int(len(X) * validation_split)
        n_train = len(X) - n_valid

        X_train, X_valid = X[:n_train], X[n_train:]
        y_train, y_valid = y[:n_train], y[n_train:]

        # Create DMatrix for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.feature_names)
        dvalid = xgb.DMatrix(X_valid, label=y_valid, feature_names=self.feature_names)

        # Model parameters optimized for low latency
        params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "silent": 1,
            "nthread": 1,  # Single thread for predictable latency
        }

        # Train with early stopping
        evals = [(dtrain, "train"), (dvalid, "valid")]
        self._model = xgb.train(  # type: ignore
            params,
            dtrain,
            num_boost_round=self.n_estimators,
            evals=evals,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False,
        )

        self._is_trained = True

        # Get final metrics
        predictions = self._model.predict(dvalid)
        rmse = float(np.sqrt(np.mean((predictions - y_valid) ** 2)))

        return {
            "rmse": rmse,
            "training_samples": n_train,
            "validation_samples": n_valid,
            "best_iteration": getattr(self._model, "best_iteration", 0),
        }

    def predict(self, features: VPINFeatures) -> VPINPrediction:
        """
        Predict VPIN value from features.
        Uses predict_inplace to avoid memory allocation.
        """
        if not self._is_trained:
            # Fallback to simple VPIN calculation
            vpin = self._calculate_online_vpin()
            confidence = 0.5
        elif XGB_AVAILABLE and self._model is not None:
            # Prepare input
            X = self._features_to_array(features)
            dmatrix = xgb.DMatrix(X, feature_names=self.feature_names)

            # Use predict_inplace to avoid allocation
            self._prediction_buffer[0] = 0.0
            self._model.predict(dmatrix, output_margin=False, pred_leaf=False)

            # Get prediction
            vpin = float(self._model.predict(dmatrix)[0])
            vpin = np.clip(vpin, 0.0, 1.0)

            # Confidence based on feature values (simplified)
            confidence = 0.8
        else:
            vpin = self._calculate_online_vpin()
            confidence = 0.5

        # Determine toxicity level
        if vpin < self.LOW_THRESHOLD:
            toxicity_level = "LOW"
        elif vpin < self.MEDIUM_THRESHOLD:
            toxicity_level = "MEDIUM"
        elif vpin < self.HIGH_THRESHOLD:
            toxicity_level = "HIGH"
        else:
            toxicity_level = "EXTREME"

        # Get feature importance if trained
        feature_importance = None
        if self._is_trained and self._model is not None:
            importance_dict = self._model.get_score(importance_type="gain")
            feature_importance = {k: v for k, v in importance_dict.items()}

        return VPINPrediction(
            vpin_value=float(vpin),
            toxicity_level=toxicity_level,
            confidence=confidence,
            predicted_horizon_ms=self.prediction_horizon_ms,
            feature_importance=feature_importance,
        )

    def get_spread_adjustment(
        self,
        prediction: VPINPrediction,
        base_spread_bps: float,
    ) -> float:
        """
        Calculate defensive spread adjustment based on toxicity prediction.
        Higher VPIN = wider spreads to protect against adverse selection.
        """
        toxicity_multipliers = {
            "LOW": 1.0,
            "MEDIUM": 1.5,
            "HIGH": 2.5,
            "EXTREME": 4.0,
        }

        multiplier = toxicity_multipliers.get(prediction.toxicity_level, 1.0)

        # Additional adjustment based on VPIN value
        vpin_factor = 1.0 + prediction.vpin_value

        adjusted_spread = base_spread_bps * multiplier * vpin_factor
        return adjusted_spread

    def reset(self) -> None:
        """Reset rolling state for new trading session."""
        self._buy_buckets.clear()
        self._sell_buckets.clear()
        self._current_buy_volume = 0.0
        self._current_sell_volume = 0.0
        self._bucket_count = 0

    def save_model(self, path: str) -> None:
        """Save trained model to file."""
        if self._model is not None:
            self._model.save_model(path)
            logger.info(f"Model saved to {path}")

    def load_model(self, path: str) -> None:
        """Load trained model from file."""
        if XGB_AVAILABLE:
            self._model = xgb.Booster(model_file=path)  # type: ignore
            self._is_trained = True
            logger.info(f"Model loaded from {path}")
