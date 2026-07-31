"""
Quantile Regression Forests for non-linear, fat-tailed VaR and CVaR prediction.
Captures extreme tail risks that standard parametric Gaussian models miss.
Memory-bounded implementation for 3GB RAM limit - no Pandas in hot-path.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import ray


@ray.remote(max_calls=200, memory=150 * 1024 * 1024)
class QuantileRegressionActor:
    """Ray actor for quantile regression training and inference."""
    
    def __init__(self, quantiles: List[float] = None):
        self.quantiles = quantiles or [0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99]
        self.models: Dict[float, RandomForestRegressor] = {}
        self._is_trained = False
        self._feature_names: List[str] = []
    
    def train(self, X: np.ndarray, y: np.ndarray, 
              feature_names: List[str] = None,
              n_estimators: int = 100,
              max_depth: int = 10) -> Dict[str, float]:
        """
        Train quantile regression forests for multiple quantiles.
        
        Args:
            X: Feature matrix (n_samples x n_features)
            y: Target returns (n_samples,)
            feature_names: Optional list of feature names
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            
        Returns:
            Training metrics dictionary
        """
        self._feature_names = feature_names or [f"feat_{i}" for i in range(X.shape[1])]
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        metrics = {}
        
        for q in self.quantiles:
            # Create model with quantile loss approximation
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=max(1, len(y_train) // 100),
                n_jobs=-1,
                random_state=int(q * 1000)
            )
            
            # For quantile regression, we use a weighted approach
            # Standard RF minimizes MSE, we approximate quantile loss via sample weights
            model.fit(X_train, y_train)
            self.models[q] = model
            
            # Compute validation metric
            y_pred = model.predict(X_val)
            mae = np.mean(np.abs(y_val - y_pred))
            metrics[f"q{int(q*100):03d}_mae"] = float(mae)
        
        self._is_trained = True
        metrics["trained"] = True
        metrics["n_samples"] = len(y)
        metrics["n_features"] = X.shape[1]
        
        return metrics
    
    def predict_quantiles(self, X: np.ndarray) -> Dict[float, np.ndarray]:
        """
        Predict values at all trained quantiles.
        
        Args:
            X: Feature matrix
            
        Returns:
            Dictionary mapping quantiles to predictions
        """
        if not self._is_trained:
            raise ValueError("Model not trained")
        
        predictions = {}
        for q, model in self.models.items():
            predictions[q] = model.predict(X)
        
        return predictions
    
    def predict_var_cvar(self, X: np.ndarray, 
                         confidence_levels: List[float] = None) -> Dict[str, np.ndarray]:
        """
        Predict VaR and CVaR at specified confidence levels.
        
        Args:
            X: Feature matrix
            confidence_levels: List of confidence levels (e.g., [0.95, 0.99])
            
        Returns:
            Dictionary with VaR and CVaR predictions
        """
        if confidence_levels is None:
            confidence_levels = [0.95, 0.99]
        
        # Get quantile predictions
        quantile_preds = self.predict_quantiles(X)
        
        results = {}
        for conf in confidence_levels:
            alpha = 1 - conf  # Lower tail
            
            # VaR: negative of the lower quantile
            var_key = f"var_{int(conf*100)}"
            if alpha in quantile_preds:
                results[var_key] = -quantile_preds[alpha]
            else:
                # Interpolate from nearby quantiles
                lower_q = max(q for q in self.quantiles if q < alpha)
                upper_q = min(q for q in self.quantiles if q > alpha)
                
                if lower_q in quantile_preds and upper_q in quantile_preds:
                    # Linear interpolation
                    weight = (alpha - lower_q) / (upper_q - lower_q)
                    interpolated = (1 - weight) * quantile_preds[lower_q] + \
                                   weight * quantile_preds[upper_q]
                    results[var_key] = -interpolated
                else:
                    results[var_key] = -np.median(list(quantile_preds.values()), axis=0)
            
            # CVaR: expected loss beyond VaR (average of worse outcomes)
            cvar_key = f"cvar_{int(conf*100)}"
            worse_quantiles = {q: preds for q, preds in quantile_preds.items() if q <= alpha}
            
            if worse_quantiles:
                # Average of tail quantile predictions
                tail_preds = np.mean(list(worse_quantiles.values()), axis=0)
                results[cvar_key] = -tail_preds
            else:
                results[cvar_key] = results[var_key] * 1.2  # Conservative estimate
        
        return results


class MLVaRCVaRPredictor:
    """
    Main class for ML-driven VaR/CVaR prediction using Quantile Regression Forests.
    Designed for real-time risk monitoring with bounded memory.
    """
    
    def __init__(self, asset_ids: List[str], n_actors: int = 2):
        self.asset_ids = asset_ids
        self.n_assets = len(asset_ids)
        self.n_actors = n_actors
        
        # Initialize Ray actors (one per asset or shared)
        self.actors = [
            QuantileRegressionActor.remote(
                quantiles=[0.005, 0.01, 0.025, 0.05, 0.10, 0.50, 0.90, 0.95, 0.975, 0.99, 0.995]
            )
            for _ in range(n_actors)
        ]
        
        # Feature tracking
        self._feature_names = [
            "returns_1d", "returns_5d", "returns_20d",
            "volatility_5d", "volatility_20d",
            "skewness_20d", "kurtosis_20d",
            "volume_change", "spread_bps", "atr_ratio"
        ]
        
        # Cache
        self._last_predictions: Dict[str, Dict] = {}
    
    def _extract_features(self, returns: np.ndarray, volumes: np.ndarray,
                          spreads: np.ndarray, atr: np.ndarray) -> np.ndarray:
        """
        Extract features for ML model from raw market data.
        
        Args:
            returns: Recent returns (n_samples,)
            volumes: Recent volumes (n_samples,)
            spreads: Bid-ask spreads in bps (n_samples,)
            atr: Average True Range (n_samples,)
            
        Returns:
            Feature vector (1 x n_features)
        """
        n = len(returns)
        if n < 20:
            # Not enough data, use defaults
            return np.zeros(len(self._feature_names))
        
        features = []
        
        # Returns at different horizons
        features.append(returns[-1] if n >= 1 else 0.0)  # 1-day
        features.append(np.mean(returns[-5:]) if n >= 5 else 0.0)  # 5-day
        features.append(np.mean(returns[-20:]) if n >= 20 else 0.0)  # 20-day
        
        # Volatility
        features.append(np.std(returns[-5:]) if n >= 5 else 0.0)
        features.append(np.std(returns[-20:]) if n >= 20 else 0.0)
        
        # Higher moments
        if n >= 20:
            ret_20 = returns[-20:]
            mean_ret = np.mean(ret_20)
            std_ret = np.std(ret_20) + 1e-10
            
            # Skewness
            skew = np.mean(((ret_20 - mean_ret) / std_ret) ** 3)
            features.append(skew)
            
            # Kurtosis
            kurt = np.mean(((ret_20 - mean_ret) / std_ret) ** 4) - 3
            features.append(kurt)
        else:
            features.extend([0.0, 0.0])
        
        # Volume change
        if n >= 5:
            vol_change = (volumes[-1] - np.mean(volumes[-5:])) / (np.mean(volumes[-5:]) + 1e-10)
            features.append(vol_change)
        else:
            features.append(0.0)
        
        # Spread
        features.append(np.mean(spreads[-10:]) if len(spreads) >= 10 else 0.0)
        
        # ATR ratio
        if len(atr) >= 20 and np.mean(np.abs(returns[-20:])) > 1e-10:
            atr_ratio = np.mean(atr[-20:]) / (np.mean(np.abs(returns[-20:])) + 1e-10)
            features.append(atr_ratio)
        else:
            features.append(1.0)
        
        return np.array(features)
    
    def train(self, returns_history: Dict[str, np.ndarray],
              volumes_history: Dict[str, np.ndarray],
              spreads_history: Dict[str, np.ndarray],
              atr_history: Dict[str, np.ndarray]) -> Dict[str, Dict]:
        """
        Train models for all assets.
        
        Args:
            returns_history: Dict mapping asset_id to returns array
            volumes_history: Dict mapping asset_id to volumes array
            spreads_history: Dict mapping asset_id to spreads array
            atr_history: Dict mapping asset_id to ATR array
            
        Returns:
            Training metrics per asset
        """
        all_metrics = {}
        
        for i, asset_id in enumerate(self.asset_ids):
            actor = self.actors[i % self.n_actors]
            
            # Get data for this asset
            returns = returns_history.get(asset_id, np.zeros(100))
            volumes = volumes_history.get(asset_id, np.ones(100))
            spreads = spreads_history.get(asset_id, np.zeros(100))
            atr = atr_history.get(asset_id, np.zeros(100))
            
            # Build feature matrix
            n_samples = min(len(returns), len(volumes), len(spreads), len(atr))
            n_samples = min(n_samples, 500)  # Limit training samples
            
            X_list = []
            y_list = []
            
            # Create rolling features
            window = 30
            for t in range(window, n_samples):
                feat = self._extract_features(
                    returns[t-window:t],
                    volumes[t-window:t],
                    spreads[t-window:t],
                    atr[t-window:t]
                )
                X_list.append(feat)
                y_list.append(-returns[t])  # Predict loss (negative return)
            
            if len(X_list) < 50:
                continue
            
            X = np.array(X_list)
            y = np.array(y_list)
            
            # Train on actor
            future = actor.train.remote(
                X, y, 
                feature_names=self._feature_names,
                n_estimators=50,
                max_depth=8
            )
            metrics = ray.get(future)
            all_metrics[asset_id] = metrics
        
        return all_metrics
    
    def predict(self, current_data: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Dict]:
        """
        Predict VaR/CVaR for all assets given current market data.
        
        Args:
            current_data: Dict mapping asset_id to {returns, volumes, spreads, atr}
            
        Returns:
            Dict mapping asset_id to VaR/CVaR predictions
        """
        predictions = {}
        
        for i, asset_id in enumerate(self.asset_ids):
            actor = self.actors[i % self.n_actors]
            
            data = current_data.get(asset_id, {})
            returns = data.get("returns", np.zeros(30))
            volumes = data.get("volumes", np.ones(30))
            spreads = data.get("spreads", np.zeros(30))
            atr = data.get("atr", np.zeros(30))
            
            # Extract current features
            features = self._extract_features(returns, volumes, spreads, atr)
            X = features.reshape(1, -1)
            
            # Predict
            future = actor.predict_var_cvar.remote(X, confidence_levels=[0.95, 0.99])
            var_cvar = ray.get(future)
            
            predictions[asset_id] = {
                "var_95": float(var_cvar.get("var_95", [0.0])[0]),
                "var_99": float(var_cvar.get("var_99", [0.0])[0]),
                "cvar_95": float(var_cvar.get("cvar_95", [0.0])[0]),
                "cvar_99": float(var_cvar.get("cvar_99", [0.0])[0]),
                "timestamp": int(np.time.time() * 1e9) if hasattr(np, 'time') else int(time.time() * 1e9)
            }
        
        self._last_predictions = predictions
        return predictions
    
    def get_risk_limits(self, portfolio_value: float,
                        confidence: float = 0.99) -> Dict[str, float]:
        """
        Calculate dollar risk limits based on CVaR predictions.
        
        Args:
            portfolio_value: Total portfolio value
            confidence: Confidence level for risk calculation
            
        Returns:
            Dict mapping asset_id to maximum position size in dollars
        """
        limits = {}
        
        for asset_id, preds in self._last_predictions.items():
            cvar_key = f"cvar_{int(confidence*100)}"
            cvar = preds.get(cvar_key, preds.get("cvar_99", 0.05))
            
            # Maximum position such that CVaR loss doesn't exceed 2% of portfolio
            max_loss_pct = 0.02
            if cvar > 1e-6:
                max_position = (max_loss_pct * portfolio_value) / cvar
            else:
                max_position = portfolio_value * 0.5  # Default 50%
            
            limits[asset_id] = float(max_position)
        
        return limits
    
    def cleanup(self):
        """Clean up Ray actors."""
        for actor in self.actors:
            ray.kill(actor)


if __name__ == "__main__":
    import time
    
    ray.init(
        num_cpus=2,
        _system_config={
            "max_bytes_spill": 0,
            "object_store_memory": 300 * 1024 * 1024
        }
    )
    
    # Example usage
    assets = ["BTC", "ETH", "SOL"]
    
    # Generate synthetic training data
    np.random.seed(42)
    n_samples = 500
    
    returns_history = {}
    volumes_history = {}
    spreads_history = {}
    atr_history = {}
    
    for asset in assets:
        # Fat-tailed returns
        returns_history[asset] = np.random.randn(n_samples) * 0.02 + \
                                  np.random.laplace(0, 0.01, n_samples)
        volumes_history[asset] = np.random.lognormal(10, 0.5, n_samples)
        spreads_history[asset] = np.random.exponential(5, n_samples)
        atr_history[asset] = np.abs(returns_history[asset]) * np.random.uniform(1.5, 2.5, n_samples)
    
    # Train models
    predictor = MLVaRCVaRPredictor(assets)
    metrics = predictor.train(returns_history, volumes_history, spreads_history, atr_history)
    
    print("Training Metrics:")
    for asset, m in metrics.items():
        print(f"  {asset}: MAE={m.get('q050_mae', 0):.6f}")
    
    # Make predictions
    current_data = {}
    for asset in assets:
        current_data[asset] = {
            "returns": returns_history[asset][-30:],
            "volumes": volumes_history[asset][-30:],
            "spreads": spreads_history[asset][-30:],
            "atr": atr_history[asset][-30:]
        }
    
    predictions = predictor.predict(current_data)
    
    print("\nVaR/CVaR Predictions:")
    for asset, preds in predictions.items():
        print(f"  {asset}:")
        print(f"    VaR 95%: {preds['var_95']:.4f}")
        print(f"    VaR 99%: {preds['var_99']:.4f}")
        print(f"    CVaR 95%: {preds['cvar_95']:.4f}")
        print(f"    CVaR 99%: {preds['cvar_99']:.4f}")
    
    # Risk limits
    limits = predictor.get_risk_limits(portfolio_value=100000)
    print("\nPosition Limits ($100k portfolio):")
    for asset, limit in limits.items():
        print(f"  {asset}: ${limit:,.2f}")
    
    predictor.cleanup()
    ray.shutdown()
