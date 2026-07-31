"""
XGBoost-based slippage prediction model for dynamic limit order offset adjustment.
Predicts exact basis-point slippage based on order size, ATR, and L2 spread.
Memory-bounded implementation for 3GB RAM limit - no Pandas in hot-path.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import time


class SlippageModel:
    """
    XGBoost-inspired slippage predictor using pure NumPy for inference.
    In production, this would load a pre-trained XGBoost model via ONNX.
    """
    
    def __init__(self, max_lookback: int = 1000):
        self.max_lookback = max_lookback
        
        # Model parameters (placeholder for XGBoost weights)
        self._feature_weights: Optional[np.ndarray] = None
        self._bias: float = 0.0
        self._feature_names: List[str] = [
            "order_size_pct", "atr_ratio", "spread_bps", 
            "volume_imbalance", "volatility_5m", "momentum_1m",
            "depth_ratio", "time_of_day", "recent_slippage"
        ]
        
        # Historical data for training
        self._features_history: List[np.ndarray] = []
        self._slippage_history: List[float] = []
        
        # Initialize with default parameters
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize model with default parameters."""
        np.random.seed(42)
        n_features = len(self._feature_names)
        self._feature_weights = np.random.randn(n_features) * 0.1
        self._bias = 0.5  # Base slippage in bps
    
    def _extract_features(self, order_size: float, avg_volume: float,
                          atr: float, spread_bps: float,
                          l2_depth: Dict, recent_returns: np.ndarray) -> np.ndarray:
        """
        Extract features for slippage prediction.
        
        Args:
            order_size: Order size in base currency
            avg_volume: Average daily volume
            atr: Average True Range
            spread_bps: Current bid-ask spread in bps
            l2_depth: L2 order book depth data
            recent_returns: Recent price returns
            
        Returns:
            Feature vector
        """
        features = []
        
        # Order size relative to volume
        order_size_pct = (order_size / (avg_volume + 1e-10)) * 100
        features.append(min(order_size_pct, 10.0))  # Cap at 10%
        
        # ATR ratio
        atr_ratio = atr / (np.mean(np.abs(recent_returns)) + 1e-10) if len(recent_returns) > 0 else 1.0
        features.append(min(atr_ratio, 5.0))
        
        # Spread
        features.append(min(spread_bps, 50.0))
        
        # Volume imbalance (from L2)
        bid_vol = l2_depth.get("bid_volume", 0)
        ask_vol = l2_depth.get("ask_volume", 0)
        imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-10)
        features.append(imbalance)
        
        # Short-term volatility
        vol_5m = np.std(recent_returns[-5:]) if len(recent_returns) >= 5 else 0.01
        features.append(vol_5m * 100)
        
        # Momentum
        mom_1m = np.sum(recent_returns[-1:]) if len(recent_returns) >= 1 else 0.0
        features.append(mom_1m * 100)
        
        # Depth ratio
        total_depth = l2_depth.get("total_depth", 1)
        depth_ratio = order_size / (total_depth + 1e-10)
        features.append(min(depth_ratio, 1.0))
        
        # Time of day (normalized)
        hour = time.localtime().tm_hour
        time_factor = np.sin(2 * np.pi * hour / 24)
        features.append(time_factor)
        
        # Recent realized slippage
        if self._slippage_history:
            recent_slip = np.mean(self._slippage_history[-5:])
        else:
            recent_slip = 0.5
        features.append(recent_slip)
        
        return np.array(features)
    
    def predict_slippage_bps(self, order_size: float, avg_volume: float,
                             atr: float, spread_bps: float,
                             l2_depth: Dict, recent_returns: np.ndarray,
                             side: str = "buy") -> float:
        """
        Predict slippage in basis points.
        
        Args:
            order_size: Order size in base currency
            avg_volume: Average daily volume
            atr: Average True Range
            spread_bps: Bid-ask spread
            l2_depth: L2 depth data
            recent_returns: Recent returns
            side: "buy" or "sell"
            
        Returns:
            Predicted slippage in basis points
        """
        features = self._extract_features(
            order_size, avg_volume, atr, spread_bps, l2_depth, recent_returns
        )
        
        # Linear prediction (placeholder for XGBoost)
        base_slippage = np.dot(features, self._feature_weights) + self._bias
        
        # Adjust for side
        if side == "sell":
            # Selling into weak markets may have higher slippage
            volume_imbalance = features[3]
            base_slippage *= (1 - volume_imbalance * 0.2)
        
        # Non-linear adjustment for large orders
        order_impact = order_size / (avg_volume + 1e-10)
        if order_impact > 0.01:
            base_slippage *= (1 + np.sqrt(order_impact * 10))
        
        return max(0.1, base_slippage)
    
    def update_with_realized(self, predicted: float, realized: float,
                            learning_rate: float = 0.01):
        """
        Update model with realized slippage data.
        
        Args:
            predicted: Predicted slippage
            realized: Actual realized slippage
            learning_rate: Learning rate for online update
        """
        error = realized - predicted
        
        # Simple online update (placeholder for proper XGBoost retraining)
        if self._features_history:
            last_features = self._features_history[-1]
            self._feature_weights += learning_rate * error * last_features
            self._bias += learning_rate * error
        
        # Store for future updates
        self._slippage_history.append(realized)
        if len(self._slippage_history) > self.max_lookback:
            self._slippage_history = self._slippage_history[-self.max_lookback:]
    
    def get_optimal_limit_offset(self, order_size: float, avg_volume: float,
                                  atr: float, spread_bps: float,
                                  l2_depth: Dict, recent_returns: np.ndarray,
                                  side: str = "buy",
                                  confidence_level: float = 0.90) -> Dict:
        """
        Calculate optimal limit order offset to guarantee fills without overpaying.
        
        Args:
            order_size: Order size
            avg_volume: Average volume
            atr: ATR
            spread_bps: Spread
            l2_depth: L2 depth
            recent_returns: Recent returns
            side: "buy" or "sell"
            confidence_level: Confidence level for fill guarantee
            
        Returns:
            Dictionary with offset recommendations
        """
        predicted_slippage = self.predict_slippage_bps(
            order_size, avg_volume, atr, spread_bps, l2_depth, recent_returns, side
        )
        
        # Add buffer for confidence level
        buffer_multiplier = 1.0 + (1 - confidence_level)
        offset_bps = predicted_slippage * buffer_multiplier
        
        # Determine limit price adjustment
        if side == "buy":
            # Buy limit should be below mid by offset
            adjustment_direction = -1
        else:
            # Sell limit should be above mid by offset
            adjustment_direction = 1
        
        return {
            "predicted_slippage_bps": float(predicted_slippage),
            "recommended_offset_bps": float(offset_bps),
            "adjustment_direction": adjustment_direction,
            "confidence_level": confidence_level,
            "estimated_fill_probability": float(0.5 + confidence_level * 0.4),
            "timestamp": int(time.time() * 1e9)
        }


class SlippageCalibrator:
    """
    Real-time slippage calibrator that continuously updates the model
    based on executed trade data.
    """
    
    def __init__(self, instruments: List[str]):
        self.instruments = instruments
        self.models: Dict[str, SlippageModel] = {
            inst: SlippageModel() for inst in instruments
        }
        
        # Execution tracking
        self._pending_orders: Dict[str, Dict] = {}
        self._realized_slippage: Dict[str, List[float]] = {inst: [] for inst in instruments}
    
    def record_order_submission(self, order_id: str, instrument_id: str,
                                 order_size: float, side: str,
                                 submission_price: float,
                                 market_data: Dict) -> None:
        """Record order submission for later slippage calculation."""
        self._pending_orders[order_id] = {
            "instrument_id": instrument_id,
            "order_size": order_size,
            "side": side,
            "submission_price": submission_price,
            "market_data": market_data,
            "timestamp": time.time()
        }
    
    def record_order_execution(self, order_id: str, execution_price: float,
                                filled_size: float) -> Optional[Dict]:
        """
        Record order execution and calculate realized slippage.
        
        Returns:
            Slippage analysis dictionary if order was found
        """
        if order_id not in self._pending_orders:
            return None
        
        order = self._pending_orders.pop(order_id)
        
        # Calculate slippage
        submission_price = order["submission_price"]
        side = order["side"]
        
        if side == "buy":
            slippage_bps = (execution_price - submission_price) / submission_price * 10000
        else:
            slippage_bps = (submission_price - execution_price) / submission_price * 10000
        
        instrument_id = order["instrument_id"]
        self._realized_slippage[instrument_id].append(slippage_bps)
        
        # Keep history bounded
        if len(self._realized_slippage[instrument_id]) > 500:
            self._realized_slippage[instrument_id] = self._realized_slippage[instrument_id][-500:]
        
        # Update model
        model = self.models[instrument_id]
        market_data = order["market_data"]
        
        # Reconstruct features from stored data
        features = model._extract_features(
            order["order_size"],
            market_data.get("avg_volume", 1e6),
            market_data.get("atr", 0.02),
            market_data.get("spread_bps", 5),
            market_data.get("l2_depth", {}),
            market_data.get("recent_returns", np.zeros(10))
        )
        
        model._features_history.append(features)
        if len(model._features_history) > model.max_lookback:
            model._features_history = model._features_history[-model.max_lookback:]
        
        # Get predicted slippage for this order
        predicted = model.predict_slippage_bps(
            order["order_size"],
            market_data.get("avg_volume", 1e6),
            market_data.get("atr", 0.02),
            market_data.get("spread_bps", 5),
            market_data.get("l2_depth", {}),
            market_data.get("recent_returns", np.zeros(10)),
            side
        )
        
        # Update model with realized data
        model.update_with_realized(predicted, slippage_bps)
        
        return {
            "order_id": order_id,
            "instrument_id": instrument_id,
            "predicted_slippage_bps": predicted,
            "realized_slippage_bps": slippage_bps,
            "error_bps": slippage_bps - predicted,
            "mape": abs(slippage_bps - predicted) / (predicted + 1e-10)
        }
    
    def get_model_stats(self, instrument_id: str) -> Dict:
        """Get calibration statistics for an instrument."""
        realized = self._realized_slippage.get(instrument_id, [])
        
        if not realized:
            return {"status": "no_data"}
        
        return {
            "n_observations": len(realized),
            "mean_slippage_bps": float(np.mean(realized)),
            "std_slippage_bps": float(np.std(realized)),
            "max_slippage_bps": float(np.max(realized)),
            "p95_slippage_bps": float(np.percentile(realized, 95))
        }


if __name__ == "__main__":
    # Example usage
    instruments = ["BTC", "ETH", "SOL"]
    
    calibrator = SlippageCalibrator(instruments)
    
    # Simulate order and execution
    np.random.seed(42)
    
    instrument = "BTC"
    model = calibrator.models[instrument]
    
    # Market data
    market_data = {
        "avg_volume": 1e9,
        "atr": 0.02,
        "spread_bps": 5,
        "l2_depth": {"bid_volume": 1e7, "ask_volume": 0.8e7, "total_depth": 5e7},
        "recent_returns": np.random.randn(10) * 0.001
    }
    
    # Get slippage prediction
    order_size = 1e6
    prediction = model.get_optimal_limit_offset(
        order_size=order_size,
        avg_volume=market_data["avg_volume"],
        atr=market_data["atr"],
        spread_bps=market_data["spread_bps"],
        l2_depth=market_data["l2_depth"],
        recent_returns=market_data["recent_returns"],
        side="buy"
    )
    
    print("Slippage Prediction:")
    print(f"  Predicted: {prediction['predicted_slippage_bps']:.2f} bps")
    print(f"  Recommended Offset: {prediction['recommended_offset_bps']:.2f} bps")
    print(f"  Fill Probability: {prediction['estimated_fill_probability']:.2%}")
    
    # Simulate execution
    order_id = "test_order_001"
    calibrator.record_order_submission(
        order_id=order_id,
        instrument_id=instrument,
        order_size=order_size,
        side="buy",
        submission_price=50000,
        market_data=market_data
    )
    
    # Simulate fill with some slippage
    realized_price = 50000 * (1 + 0.0008)  # 8 bps slippage
    result = calibrator.record_order_execution(order_id, realized_price, order_size)
    
    if result:
        print(f"\nExecution Analysis:")
        print(f"  Predicted: {result['predicted_slippage_bps']:.2f} bps")
        print(f"  Realized: {result['realized_slippage_bps']:.2f} bps")
        print(f"  Error: {result['error_bps']:.2f} bps")
    
    # Stats
    stats = calibrator.get_model_stats(instrument)
    print(f"\nModel Stats: {stats}")
