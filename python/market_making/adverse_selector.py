"""
Adverse Selection Predictor - Binary classifier for predicting toxic fills.
Uses LightGBM to predict probability of passive limit orders being run over by informed flow.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np
import asyncio
from collections import deque

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    lgb = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OrderBookFeatures:
    """Features extracted from order book for adverse selection prediction."""
    bid_ask_spread: float = 0.0
    mid_price: float = 0.0
    order_imbalance: float = 0.0
    bid_depth: float = 0.0
    ask_depth: float = 0.0
    depth_imbalance: float = 0.0
    recent_trade_flow: float = 0.0
    trade_sign_imbalance: float = 0.0
    price_momentum_1s: float = 0.0
    price_momentum_5s: float = 0.0
    volatility_1s: float = 0.0
    volatility_5s: float = 0.0
    queue_position: int = 0
    cancellation_rate: float = 0.0
    large_order_ratio: float = 0.0
    
    def to_array(self) -> np.ndarray:
        """Convert features to numpy array."""
        return np.array([
            self.bid_ask_spread,
            self.mid_price,
            self.order_imbalance,
            self.bid_depth,
            self.ask_depth,
            self.depth_imbalance,
            self.recent_trade_flow,
            self.trade_sign_imbalance,
            self.price_momentum_1s,
            self.price_momentum_5s,
            self.volatility_1s,
            self.volatility_5s,
            self.queue_position,
            self.cancellation_rate,
            self.large_order_ratio
        ])
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OrderBookFeatures':
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


@dataclass
class AdverseSelectionPrediction:
    """Prediction result for adverse selection risk."""
    timestamp: float
    symbol: str
    side: str  # 'buy' or 'sell'
    probability_adverse: float
    expected_shortfall: float
    confidence_interval: Tuple[float, float]
    feature_contributions: Optional[Dict[str, float]] = None
    recommendation: str = "hold"  # 'quote', 'widen', 'cancel', 'hold'
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "symbol": self.symbol,
            "side": self.side,
            "probability_adverse": self.probability_adverse,
            "expected_shortfall": self.expected_shortfall,
            "confidence_low": self.confidence_interval[0],
            "confidence_high": self.confidence_interval[1],
            "feature_contributions": self.feature_contributions,
            "recommendation": self.recommendation
        }


class AdverseSelectorModel:
    """
    LightGBM-based model for predicting adverse selection.
    Trained on historical fill outcomes labeled as toxic/non-toxic.
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 n_estimators: int = 200,
                 learning_rate: float = 0.05,
                 max_depth: int = 8,
                 num_leaves: int = 31,
                 min_child_samples: int = 20):
        """Initialize the adverse selection model."""
        if not HAS_LIGHTGBM:
            logger.warning("LightGBM not available, using fallback predictions")
        
        self.model = None
        self.feature_names = [
            'bid_ask_spread', 'mid_price', 'order_imbalance',
            'bid_depth', 'ask_depth', 'depth_imbalance',
            'recent_trade_flow', 'trade_sign_imbalance',
            'price_momentum_1s', 'price_momentum_5s',
            'volatility_1s', 'volatility_5s',
            'queue_position', 'cancellation_rate', 'large_order_ratio'
        ]
        
        self.params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': num_leaves,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'min_child_samples': min_child_samples,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'n_jobs': -1,
            'random_state': 42
        }
        
        self.n_estimators = n_estimators
        self._is_trained = False
        
        if model_path is not None:
            self.load_model(model_path)
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> 'AdverseSelectorModel':
        """
        Train the model on historical data.
        
        Args:
            X: Feature matrix
            y: Labels (1 = toxic fill, 0 = normal fill)
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Self for chaining
        """
        if not HAS_LIGHTGBM:
            logger.warning("Cannot train without LightGBM")
            return self
        
        try:
            train_data = lgb.Dataset(X, label=y, feature_name=self.feature_names)
            
            valid_sets = [train_data]
            valid_names = ['train']
            
            if X_val is not None and y_val is not None:
                valid_sets.append(lgb.Dataset(X_val, label=y_val, feature_name=self.feature_names))
                valid_names.append('valid')
            
            self.model = lgb.train(
                self.params,
                train_data,
                num_boost_round=self.n_estimators,
                valid_sets=valid_sets,
                valid_names=valid_names,
                early_stopping_rounds=20,
                verbose_eval=False
            )
            
            self._is_trained = True
            logger.info(f"Adverse selector trained: {len(y)} samples")
            
            if X_val is not None:
                train_auc = self._calculate_auc(X, y)
                val_auc = self._calculate_auc(X_val, y_val)
                logger.info(f"Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
        
        return self
    
    def _calculate_auc(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate AUC score."""
        from sklearn.metrics import roc_auc_score
        try:
            preds = self.predict_proba(X)
            return roc_auc_score(y, preds)
        except:
            return 0.5
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability of adverse selection."""
        if not self._is_trained or self.model is None:
            # Fallback: use simple heuristic based on order imbalance
            if len(X.shape) == 1:
                X = X.reshape(1, -1)
            
            # Simple heuristic: high imbalance = higher adverse selection risk
            imbalance_idx = 2  # order_imbalance column
            probs = 0.5 + 0.3 * np.tanh(X[:, imbalance_idx] if X.shape[1] > imbalance_idx else 0)
            return np.clip(probs, 0.01, 0.99)
        
        return self.model.predict(X)
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary adverse selection labels."""
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)
    
    def get_feature_importance(self, importance_type: str = 'gain') -> Dict[str, float]:
        """Get feature importance scores."""
        if self.model is None:
            return {name: 0.0 for name in self.feature_names}
        
        importance = self.model.feature_importance(importance_type=importance_type)
        return dict(zip(self.feature_names, importance.tolist()))
    
    def save_model(self, path: str):
        """Save model to file."""
        if self.model is not None:
            self.model.save_model(path)
            logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model from file."""
        if HAS_LIGHTGBM:
            try:
                self.model = lgb.Booster(model_file=path)
                self._is_trained = True
                logger.info(f"Model loaded from {path}")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
    
    def explain_prediction(self, X: np.ndarray) -> Dict[str, float]:
        """Get approximate feature contributions for a prediction."""
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        base_pred = 0.5
        actual_pred = self.predict_proba(X)[0]
        total_contribution = actual_pred - base_pred
        
        # Distribute contribution proportionally to feature importance
        importance = self.get_feature_importance()
        total_imp = sum(abs(v) for v in importance.values())
        
        if total_imp == 0:
            return {name: 0.0 for name in self.feature_names}
        
        contributions = {}
        for name, imp in importance.items():
            idx = self.feature_names.index(name)
            sign = 1 if X[0, idx] > 0 else -1
            contributions[name] = total_contribution * (abs(imp) / total_imp) * sign
        
        return contributions


class AdverseSelectorEngine:
    """
    Real-time engine for adverse selection prediction.
    Maintains rolling windows of microstructure features.
    """
    
    def __init__(self, 
                 model: Optional[AdverseSelectorModel] = None,
                 window_size: int = 1000,
                 update_interval: float = 0.1):
        """
        Initialize the adverse selection engine.
        
        Args:
            model: Pre-trained AdverseSelectorModel
            window_size: Size of rolling feature window
            update_interval: Minimum time between updates (seconds)
        """
        self.model = model or AdverseSelectorModel()
        self.window_size = window_size
        self.update_interval = update_interval
        
        # Rolling buffers for feature calculation
        self._trade_buffer: deque = deque(maxlen=window_size)
        self._price_buffer: deque = deque(maxlen=window_size)
        self._order_book_snapshots: deque = deque(maxlen=100)
        
        self._last_update_time: float = 0.0
        self._current_features: Optional[OrderBookFeatures] = None
        self._predictions_cache: Dict[str, AdverseSelectionPrediction] = {}
    
    def add_trade(self, timestamp: float, price: float, size: float, 
                  side: str, aggressor: str):
        """Add a trade to the rolling buffer."""
        self._trade_buffer.append({
            'timestamp': timestamp,
            'price': price,
            'size': size,
            'side': side,
            'aggressor': aggressor  # 'buyer' or 'seller'
        })
        self._price_buffer.append(price)
    
    def add_order_book_snapshot(self, timestamp: float, 
                                 bids: List[Tuple[float, float]],
                                 asks: List[Tuple[float, float]]):
        """Add an order book snapshot."""
        self._order_book_snapshots.append({
            'timestamp': timestamp,
            'bids': bids,
            'asks': asks
        })
    
    def calculate_features(self, current_price: float,
                           bid_depth_total: float,
                           ask_depth_total: float,
                           queue_position: int = 0) -> OrderBookFeatures:
        """Calculate current microstructure features."""
        trades = list(self._trade_buffer)
        prices = list(self._price_buffer)
        
        if len(trades) < 10:
            return OrderBookFeatures(mid_price=current_price)
        
        # Calculate features
        now = trades[-1]['timestamp']
        
        # Recent trades (last 1 second worth)
        recent_trades = [t for t in trades if now - t['timestamp'] < 1.0]
        
        # Trade flow
        buy_volume = sum(t['size'] for t in recent_trades if t['aggressor'] == 'buyer')
        sell_volume = sum(t['size'] for t in recent_trades if t['aggressor'] == 'seller')
        total_volume = buy_volume + sell_volume
        
        recent_trade_flow = (buy_volume - sell_volume) / (total_volume + 1e-6)
        trade_sign_imbalance = (len([t for t in recent_trades if t['aggressor'] == 'buyer']) - 
                               len([t for t in recent_trades if t['aggressor'] == 'seller'])) / (len(recent_trades) + 1e-6)
        
        # Price momentum
        if len(prices) > 10:
            price_1s_ago = prices[-min(len(prices), int(len(prices) * 0.1))] if len(prices) > 10 else prices[0]
            price_5s_ago = prices[-min(len(prices), int(len(prices) * 0.5))] if len(prices) > 50 else prices[0]
            price_momentum_1s = (current_price - price_1s_ago) / (price_1s_ago + 1e-6)
            price_momentum_5s = (current_price - price_5s_ago) / (price_5s_ago + 1e-6)
        else:
            price_momentum_1s = 0.0
            price_momentum_5s = 0.0
        
        # Volatility
        if len(prices) > 20:
            returns = np.diff(np.log(prices[-20:]))
            volatility_1s = float(np.std(returns)) * np.sqrt(252 * 60 * 60) if len(returns) > 1 else 0.0
            volatility_5s = float(np.std(returns)) * np.sqrt(252 * 60 * 12) if len(returns) > 1 else 0.0
        else:
            volatility_1s = 0.0
            volatility_5s = 0.0
        
        # Order book imbalance
        order_imbalance = (bid_depth_total - ask_depth_total) / (bid_depth_total + ask_depth_total + 1e-6)
        depth_imbalance = order_imbalance
        
        # Spread
        spread = 0.0
        if self._order_book_snapshots:
            last_ob = self._order_book_snapshots[-1]
            if last_ob['bids'] and last_ob['asks']:
                spread = last_ob['asks'][0][0] - last_ob['bids'][0][0]
        
        # Cancellation rate (simplified)
        cancellation_rate = 0.1  # Would need more sophisticated tracking
        
        # Large order ratio
        if trades:
            median_size = np.median([t['size'] for t in trades])
            large_trades = sum(1 for t in trades if t['size'] > median_size * 3)
            large_order_ratio = large_trades / len(trades)
        else:
            large_order_ratio = 0.0
        
        features = OrderBookFeatures(
            bid_ask_spread=spread,
            mid_price=current_price,
            order_imbalance=order_imbalance,
            bid_depth=bid_depth_total,
            ask_depth=ask_depth_total,
            depth_imbalance=depth_imbalance,
            recent_trade_flow=recent_trade_flow,
            trade_sign_imbalance=trade_sign_imbalance,
            price_momentum_1s=price_momentum_1s,
            price_momentum_5s=price_momentum_5s,
            volatility_1s=volatility_1s,
            volatility_5s=volatility_5s,
            queue_position=queue_position,
            cancellation_rate=cancellation_rate,
            large_order_ratio=large_order_ratio
        )
        
        self._current_features = features
        return features
    
    async def predict_adverse_selection(self, symbol: str, side: str,
                                        features: Optional[OrderBookFeatures] = None) -> AdverseSelectionPrediction:
        """
        Predict adverse selection risk for a potential order.
        
        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            features: Pre-calculated features (or uses current)
            
        Returns:
            AdverseSelectionPrediction with risk assessment
        """
        import time
        timestamp = time.time()
        
        if features is None:
            features = self._current_features
        
        if features is None:
            features = OrderBookFeatures()
        
        feat_array = features.to_array().reshape(1, -1)
        
        # Get probability
        prob = self.model.predict_proba(feat_array)[0]
        
        # Calculate expected shortfall (simplified)
        expected_shortfall = prob * 0.02  # Assume 2% adverse move if selected
        
        # Confidence interval (using bootstrap approximation)
        ci_width = 0.1 * (1 - prob)  # Higher uncertainty for extreme probabilities
        ci_low = max(0, prob - ci_width)
        ci_high = min(1, prob + ci_width)
        
        # Feature contributions
        contributions = self.model.explain_prediction(feat_array)
        
        # Generate recommendation
        if prob > 0.7:
            recommendation = "cancel"
        elif prob > 0.5:
            recommendation = "widen"
        elif prob > 0.3:
            recommendation = "hold"
        else:
            recommendation = "quote"
        
        prediction = AdverseSelectionPrediction(
            timestamp=timestamp,
            symbol=symbol,
            side=side,
            probability_adverse=float(prob),
            expected_shortfall=float(expected_shortfall),
            confidence_interval=(float(ci_low), float(ci_high)),
            feature_contributions=contributions,
            recommendation=recommendation
        )
        
        # Cache prediction
        cache_key = f"{symbol}_{side}"
        self._predictions_cache[cache_key] = prediction
        
        # Limit cache size
        if len(self._predictions_cache) > 100:
            self._predictions_cache.pop(next(iter(self._predictions_cache)))
        
        return prediction
    
    def get_current_risk_summary(self) -> Dict[str, Any]:
        """Get summary of current adverse selection risk."""
        if not self._predictions_cache:
            return {"status": "no_predictions"}
        
        probs = [p.probability_adverse for p in self._predictions_cache.values()]
        
        return {
            "mean_adverse_prob": float(np.mean(probs)),
            "max_adverse_prob": float(np.max(probs)),
            "min_adverse_prob": float(np.min(probs)),
            "high_risk_count": sum(1 for p in probs if p > 0.5),
            "predictions_cached": len(self._predictions_cache)
        }


# Module singleton
_selector_engine: Optional[AdverseSelectorEngine] = None


def get_adverse_selector(model_path: Optional[str] = None) -> AdverseSelectorEngine:
    """Get or create the global adverse selector engine."""
    global _selector_engine
    
    if _selector_engine is None:
        model = AdverseSelectorModel(model_path=model_path) if model_path else AdverseSelectorModel()
        _selector_engine = AdverseSelectorEngine(model=model)
        logger.info("Created adverse selector engine")
    
    return _selector_engine


if __name__ == "__main__":
    # Test the adverse selector
    np.random.seed(42)
    
    # Create synthetic training data
    n_samples = 5000
    X_train = np.random.randn(n_samples, 15)
    # Toxic fills more likely when imbalance is high and momentum against position
    y_train = ((X_train[:, 2] > 0.5) & (X_train[:, 8] < -0.3)).astype(float)
    y_train += np.random.random(n_samples) * 0.2  # Add noise
    
    # Train model
    model = AdverseSelectorModel(n_estimators=50)
    model.fit(X_train[:4000], y_train[:4000], X_train[4000:], y_train[4000:])
    
    print(f"Feature Importance:")
    for feat, imp in sorted(model.get_feature_importance().items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {feat}: {imp:.2f}")
    
    # Test engine
    engine = AdverseSelectorEngine(model=model)
    
    # Simulate some trades
    base_price = 100.0
    for i in range(100):
        timestamp = time.time() - (100 - i) * 0.1
        price = base_price + np.random.randn() * 0.01
        size = np.random.exponential(10)
        side = np.random.choice(['buy', 'sell'])
        aggressor = np.random.choice(['buyer', 'seller'])
        engine.add_trade(timestamp, price, size, side, aggressor)
    
    # Calculate features and predict
    features = engine.calculate_features(
        current_price=base_price,
        bid_depth_total=1000,
        ask_depth_total=1200,
        queue_position=5
    )
    
    print(f"\nCurrent Features:")
    print(f"  Order Imbalance: {features.order_imbalance:.4f}")
    print(f"  Trade Flow: {features.recent_trade_flow:.4f}")
    print(f"  Momentum 1s: {features.price_momentum_1s:.6f}")
    
    # Predict adverse selection
    prediction = asyncio.run(engine.predict_adverse_selection("BTC/USD", "buy"))
    
    print(f"\nAdverse Selection Prediction:")
    print(f"  Probability: {prediction.probability_adverse:.4f}")
    print(f"  Expected Shortfall: {prediction.expected_shortfall:.6f}")
    print(f"  Recommendation: {prediction.recommendation}")
    print(f"  Risk Summary: {engine.get_current_risk_summary()}")
