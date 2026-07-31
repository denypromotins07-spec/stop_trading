"""
Time-series model forecasting L2 liquidity evaporation.
Optimally times execution of large block orders.
Memory-bounded implementation for 3GB RAM limit - no Pandas in hot-path.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import time


class LiquidityForecaster:
    """
    ARIMA-inspired liquidity forecasting using pure NumPy.
    Predicts L2 liquidity evaporation to optimize block order timing.
    """
    
    def __init__(self, lookback_window: int = 100, forecast_horizon: int = 10):
        self.lookback_window = lookback_window
        self.forecast_horizon = forecast_horizon
        
        # Historical data storage (bounded)
        self._liquidity_history: deque = deque(maxlen=lookback_window)
        self._spread_history: deque = deque(maxlen=lookback_window)
        self._volume_history: deque = deque(maxlen=lookback_window)
        
        # Model parameters (AR(1) + trend)
        self._ar_coef: float = 0.7
        self._trend_coef: float = 0.0
        self._mean_liquidity: float = 1e7
        self._volatility: float = 0.1
        
        # Update counters
        self._n_updates: int = 0
    
    def update(self, bid_depth: float, ask_depth: float, spread_bps: float,
               volume: float):
        """Update model with new liquidity data."""
        total_liquidity = bid_depth + ask_depth
        
        self._liquidity_history.append(total_liquidity)
        self._spread_history.append(spread_bps)
        self._volume_history.append(volume)
        
        self._n_updates += 1
        
        # Periodic parameter re-estimation
        if self._n_updates % 50 == 0 and len(self._liquidity_history) > 20:
            self._reestimate_parameters()
    
    def _reestimate_parameters(self):
        """Re-estimate AR model parameters from history."""
        liq_array = np.array(self._liquidity_history)
        
        if len(liq_array) < 10:
            return
        
        # Simple AR(1) estimation
        y_t = liq_array[1:]
        y_t1 = liq_array[:-1]
        
        mean_liq = np.mean(liq_array)
        self._mean_liquidity = mean_liq
        
        # Centered series
        y_t_c = y_t - mean_liq
        y_t1_c = y_t1 - mean_liq
        
        # AR coefficient via OLS
        numerator = np.dot(y_t1_c, y_t_c)
        denominator = np.dot(y_t1_c, y_t1_c)
        
        if abs(denominator) > 1e-10:
            self._ar_coef = np.clip(numerator / denominator, 0.1, 0.95)
        
        # Estimate volatility
        residuals = y_t - mean_liq - self._ar_coef * y_t1_c
        self._volatility = np.std(residuals) / (mean_liq + 1e-10)
        
        # Trend estimation
        if len(liq_array) > 20:
            recent_mean = np.mean(liq_array[-10:])
            older_mean = np.mean(liq_array[-20:-10])
            self._trend_coef = (recent_mean - older_mean) / (mean_liq + 1e-10) * 0.1
    
    def forecast_liquidity(self, n_steps: int = None) -> Dict:
        """
        Forecast future liquidity levels.
        
        Returns:
            Dictionary with liquidity forecasts and confidence intervals
        """
        if n_steps is None:
            n_steps = self.forecast_horizon
        
        if not self._liquidity_history:
            return {
                "forecasts": [self._mean_liquidity] * n_steps,
                "confidence_lower": [self._mean_liquidity * 0.5] * n_steps,
                "confidence_upper": [self._mean_liquidity * 1.5] * n_steps,
                "evaporation_risk": "unknown"
            }
        
        current_liq = self._liquidity_history[-1]
        forecasts = []
        conf_lower = []
        conf_upper = []
        
        prev_forecast = current_liq
        cumulative_var = 0
        
        for step in range(n_steps):
            # AR(1) forecast
            forecast = self._mean_liquidity + self._ar_coef * (prev_forecast - self._mean_liquidity)
            
            # Add trend
            forecast *= (1 + self._trend_coef * (step + 1))
            
            forecasts.append(forecast)
            
            # Confidence interval widens with horizon
            cumulative_var += self._volatility ** 2 * (self._ar_coef ** (2 * step))
            std_err = np.sqrt(cumulative_var) * self._mean_liquidity
            
            conf_lower.append(max(0, forecast - 1.96 * std_err))
            conf_upper.append(forecast + 1.96 * std_err)
            
            prev_forecast = forecast
        
        # Calculate evaporation risk
        avg_forecast = np.mean(forecasts[:min(5, len(forecasts))])
        current_avg = np.mean(list(self._liquidity_history)[-5:])
        
        if avg_forecast < current_avg * 0.7:
            evaporation_risk = "high"
        elif avg_forecast < current_avg * 0.85:
            evaporation_risk = "medium"
        else:
            evaporation_risk = "low"
        
        return {
            "forecasts": [float(f) for f in forecasts],
            "confidence_lower": [float(c) for c in conf_lower],
            "confidence_upper": [float(c) for c in conf_upper],
            "evaporation_risk": evaporation_risk,
            "current_liquidity": float(current_liq),
            "predicted_change_pct": float((forecasts[0] - current_liq) / (current_liq + 1e-10) * 100)
        }
    
    def should_delay_execution(self, order_size: float, 
                                urgency: float = 0.5) -> Dict:
        """
        Determine if execution should be delayed based on liquidity forecast.
        
        Args:
            order_size: Size of order to execute
            urgency: Order urgency (0-1)
            
        Returns:
            Execution timing recommendation
        """
        forecast = self.forecast_liquidity(n_steps=5)
        
        current_liq = forecast["current_liquidity"]
        avg_forecast = np.mean(forecast["forecasts"][:3])
        
        # Liquidity ratio
        liq_ratio = order_size / (current_liq + 1e-10)
        
        # Decision logic
        should_delay = False
        recommended_delay_seconds = 0
        reason = ""
        
        if forecast["evaporation_risk"] == "high" and urgency < 0.7:
            should_delay = True
            recommended_delay_seconds = 60  # Wait 1 minute
            reason = "High liquidity evaporation risk detected"
        elif liq_ratio > 0.1 and urgency < 0.5:
            should_delay = True
            recommended_delay_seconds = 30
            reason = f"Order size ({liq_ratio:.1%}) too large relative to current liquidity"
        elif forecast["predicted_change_pct"] > 5 and urgency < 0.3:
            should_delay = True
            recommended_delay_seconds = 45
            reason = "Liquidity expected to improve shortly"
        
        return {
            "should_delay": should_delay,
            "recommended_delay_seconds": recommended_delay_seconds,
            "reason": reason,
            "current_liquidity_ratio": float(liq_ratio),
            "evaporation_risk": forecast["evaporation_risk"],
            "optimal_execution_window": self._find_optimal_window(forecast, order_size)
        }
    
    def _find_optimal_window(self, forecast: Dict, order_size: float) -> Dict:
        """Find optimal execution window within forecast horizon."""
        forecasts = forecast["forecasts"]
        
        if not forecasts:
            return {"best_step": 0, "expected_slippage_reduction": 0}
        
        # Find step with best liquidity
        best_step = int(np.argmax(forecasts))
        best_liq = forecasts[best_step]
        
        # Estimate slippage reduction
        current_liq = forecast["current_liquidity"]
        slippage_current = order_size / (current_liq + 1e-10)
        slippage_best = order_size / (best_liq + 1e-10)
        
        reduction = (slippage_current - slippage_best) / slippage_current
        
        return {
            "best_step": best_step,
            "expected_liquidity": float(best_liq),
            "expected_slippage_reduction": float(max(0, reduction)),
            "wait_time_seconds": best_step * 10  # Assuming 10-second steps
        }


class LiquidityMonitor:
    """
    Real-time liquidity monitoring across instruments.
    Coordinates forecasting and execution timing recommendations.
    """
    
    def __init__(self, instruments: List[str]):
        self.instruments = instruments
        self.forecasters: Dict[str, LiquidityForecaster] = {
            inst: LiquidityForecaster() for inst in instruments
        }
        
        # Alert thresholds
        self._alert_thresholds: Dict[str, float] = {
            inst: 0.15 for inst in instruments  # 15% liquidity drop
        }
        
        # Alert history
        self._alerts: deque = deque(maxlen=100)
    
    def update_liquidity(self, instrument_id: str, bid_depth: float,
                         ask_depth: float, spread_bps: float, volume: float):
        """Update liquidity data for an instrument."""
        if instrument_id not in self.forecasters:
            return
        
        forecaster = self.forecasters[instrument_id]
        prev_liq = forecaster._liquidity_history[-1] if forecaster._liquidity_history else None
        
        forecaster.update(bid_depth, ask_depth, spread_bps, volume)
        
        # Check for alerts
        if prev_liq is not None:
            current_liq = bid_depth + ask_depth
            change_pct = (current_liq - prev_liq) / (prev_liq + 1e-10)
            
            if change_pct < -self._alert_thresholds[instrument_id]:
                alert = {
                    "type": "liquidity_drop",
                    "instrument_id": instrument_id,
                    "change_pct": float(change_pct * 100),
                    "timestamp": int(time.time() * 1e9)
                }
                self._alerts.append(alert)
    
    def get_execution_timing(self, instrument_id: str, order_size: float,
                             urgency: float = 0.5) -> Dict:
        """Get execution timing recommendation for an instrument."""
        if instrument_id not in self.forecasters:
            return {"error": "Instrument not found"}
        
        forecaster = self.forecasters[instrument_id]
        timing = forecaster.should_delay_execution(order_size, urgency)
        
        forecast = forecaster.forecast_liquidity()
        
        return {
            "instrument_id": instrument_id,
            "order_size": order_size,
            "timing_recommendation": timing,
            "liquidity_forecast": {
                "next_5_steps": forecast["forecasts"][:5],
                "evaporation_risk": forecast["evaporation_risk"]
            },
            "timestamp": int(time.time() * 1e9)
        }
    
    def get_all_alerts(self) -> List[Dict]:
        """Get and clear all pending alerts."""
        alerts = list(self._alerts)
        self._alerts.clear()
        return alerts


if __name__ == "__main__":
    # Example usage
    instruments = ["BTC", "ETH", "SOL"]
    
    monitor = LiquidityMonitor(instruments)
    
    # Simulate liquidity updates
    np.random.seed(42)
    
    print("Simulating Liquidity Updates:\n")
    
    for t in range(50):
        for inst in instruments:
            # Simulate varying liquidity
            base_liq = {"BTC": 1e7, "ETH": 5e6, "SOL": 2e6}[inst]
            liq_noise = np.random.lognormal(0, 0.1)
            
            # Occasional liquidity drops
            if t == 30 and inst == "ETH":
                liq_noise *= 0.6  # 40% drop
            
            bid_depth = base_liq * liq_noise * 0.5
            ask_depth = base_liq * liq_noise * 0.5
            spread_bps = np.random.exponential(5)
            volume = np.random.lognormal(14, 0.5)
            
            monitor.update_liquidity(inst, bid_depth, ask_depth, spread_bps, volume)
    
    # Get alerts
    alerts = monitor.get_all_alerts()
    if alerts:
        print("Liquidity Alerts:")
        for alert in alerts:
            print(f"  {alert['instrument_id']}: {alert['change_pct']:.1f}% drop")
    
    # Get timing recommendations
    print("\nExecution Timing Recommendations:")
    
    for inst in instruments:
        timing = monitor.get_execution_timing(inst, order_size=1e5, urgency=0.5)
        rec = timing["timing_recommendation"]
        
        print(f"\n{inst}:")
        print(f"  Should Delay: {rec['should_delay']}")
        if rec["should_delay"]:
            print(f"  Reason: {rec['reason']}")
            print(f"  Recommended Delay: {rec['recommended_delay_seconds']}s")
        print(f"  Evaporation Risk: {rec['evaporation_risk']}")
        print(f"  Optimal Window: {rec['optimal_execution_window']['wait_time_seconds']}s wait")
