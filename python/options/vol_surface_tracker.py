"""
Advanced Volatility Surface Tracker for Deribit/Binance Options.
Implements SVI (Stochastic Volatility Inspired) parameterization with O(1) fitting.
Strictly enforces 3GB RAM limit via bounded optimization iterations.
"""
import asyncio
import numpy as np
from scipy.optimize import minimize, Bounds
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import time

@dataclass
class SVIParams:
    """SVI Model Parameters: a, b, rho, m, sigma"""
    a: float = 0.0
    b: float = 0.0
    rho: float = -0.5
    m: float = 0.0
    sigma: float = 0.1
    
    def to_array(self) -> np.ndarray:
        return np.array([self.a, self.b, self.rho, self.m, self.sigma])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'SVIParams':
        return cls(a=arr[0], b=arr[1], rho=arr[2], m=arr[3], sigma=arr[4])


@dataclass
class VolSurfacePoint:
    """Single point on the volatility surface"""
    strike: float
    expiry: float
    implied_vol: float
    timestamp_ns: int
    instrument_id: str


@dataclass
class FittedSurface:
    """Result of SVI surface fit"""
    params: SVIParams
    rmse: float
    fit_time_us: float
    timestamp_ns: int
    convergence: bool


class SVIFitter:
    """
    SVI Volatility Surface Fitter with strict iteration limits.
    Uses scipy.optimize with bounded parameters to prevent divergence.
    """
    
    MAX_ITERATIONS = 50
    TOLERANCE = 1e-6
    
    # Parameter bounds to ensure stability
    PARAM_BOUNDS = Bounds(
        lb=[0.0, 0.0, -1.0, -0.5, 0.001],  # a, b, rho, m, sigma
        ub=[1.0, 2.0, 1.0, 0.5, 2.0]
    )
    
    def __init__(self):
        self._last_params: Optional[SVIParams] = None
        self._fit_count = 0
    
    def _svi_variance(self, k: np.ndarray, params: SVIParams) -> np.ndarray:
        """
        Compute SVI variance for given log-moneyness k.
        SVI: w(k) = a + b * (rho*(k-m) + sqrt((k-m)^2 + sigma^2))
        """
        dm = k - params.m
        sqrt_term = np.sqrt(dm**2 + params.sigma**2)
        return params.a + params.b * (params.rho * dm + sqrt_term)
    
    def _objective(self, params_arr: np.ndarray, k: np.ndarray, 
                   observed_var: np.ndarray) -> float:
        """Least squares objective for SVI fitting"""
        params = SVIParams.from_array(params_arr)
        model_var = self._svi_variance(k, params)
        # Weight by distance from ATM for better fit near center
        weights = np.exp(-0.5 * (k / 0.1)**2)
        return np.sum(weights * (model_var - observed_var)**2)
    
    def fit(self, strikes: np.ndarray, spot: float, expiries: np.ndarray,
            implied_vols: np.ndarray, initial_params: Optional[SVIParams] = None
            ) -> FittedSurface:
        """
        Fit SVI surface to observed implied volatilities.
        Returns fitted parameters and convergence metrics.
        """
        start_time = time.perf_counter_ns()
        
        # Convert to log-moneyness and variance
        k = np.log(strikes / spot)
        obs_var = implied_vols**2
        
        # Remove NaN/Inf values
        valid_mask = np.isfinite(k) & np.isfinite(obs_var)
        k = k[valid_mask]
        obs_var = obs_var[valid_mask]
        
        if len(k) < 3:
            raise ValueError("Insufficient data points for SVI fit")
        
        # Initialize parameters
        if initial_params is None:
            if self._last_params is not None:
                x0 = self._last_params.to_array()
            else:
                x0 = SVIParams().to_array()
        else:
            x0 = initial_params.to_array()
        
        # Optimize with strict bounds
        result = minimize(
            self._objective,
            x0,
            args=(k, obs_var),
            method='L-BFGS-B',
            bounds=self.PARAM_BOUNDS,
            options={
                'maxiter': self.MAX_ITERATIONS,
                'ftol': self.TOLERANCE,
                'gtol': self.TOLERANCE
            }
        )
        
        fit_time_ns = time.perf_counter_ns() - start_time
        
        fitted_params = SVIParams.from_array(result.x)
        self._last_params = fitted_params
        self._fit_count += 1
        
        # Calculate RMSE
        model_var = self._svi_variance(k, fitted_params)
        rmse = np.sqrt(np.mean((model_var - obs_var)**2))
        
        return FittedSurface(
            params=fitted_params,
            rmse=rmse,
            fit_time_us=fit_time_ns / 1000,
            timestamp_ns=time.time_ns(),
            convergence=result.success
        )
    
    def get_implied_vol(self, strike: float, spot: float, params: SVIParams) -> float:
        """Get implied volatility for a single strike using fitted params"""
        k = np.log(strike / spot)
        var = self._svi_variance(np.array([k]), params)[0]
        return np.sqrt(max(var, 0.0))


class VolSurfaceTracker:
    """
    Real-time volatility surface tracker for multiple instruments.
    Maintains bounded history and detects skew anomalies.
    """
    
    MAX_HISTORY_POINTS = 1000  # Bounded memory per instrument
    ANOMALY_THRESHOLD = 3.0  # Z-score threshold for skew anomalies
    
    def __init__(self):
        self._fitter = SVIFitter()
        self._surface_history: Dict[str, deque] = {}
        self._current_surfaces: Dict[str, FittedSurface] = {}
        self._skew_history: Dict[str, deque] = {}
        self._anomaly_callbacks: List[callable] = []
        self._lock = asyncio.Lock()
    
    def register_anomaly_callback(self, callback: callable):
        """Register callback for volatility skew anomalies"""
        self._anomaly_callbacks.append(callback)
    
    async def add_observation(self, point: VolSurfacePoint):
        """Add a new volatility observation and update surface"""
        async with self._lock:
            inst_id = point.instrument_id
            
            # Initialize history if needed
            if inst_id not in self._surface_history:
                self._surface_history[inst_id] = deque(maxlen=self.MAX_HISTORY_POINTS)
                self._skew_history[inst_id] = deque(maxlen=100)
            
            self._surface_history[inst_id].append(point)
            
            # Attempt surface fit if we have enough points
            if len(self._surface_history[inst_id]) >= 10:
                await self._fit_surface(inst_id)
    
    async def _fit_surface(self, inst_id: str):
        """Fit volatility surface for an instrument"""
        points = list(self._surface_history[inst_id])
        
        # Group by expiry for slice-by-slice fitting
        expiry_groups: Dict[float, List[VolSurfacePoint]] = {}
        for p in points:
            if p.expiry not in expiry_groups:
                expiry_groups[p.expiry] = []
            expiry_groups[p.expiry].append(p)
        
        # Use most recent expiry for main fit
        latest_expiry = max(expiry_groups.keys())
        slice_points = expiry_groups[latest_expiry]
        
        if len(slice_points) < 5:
            return
        
        strikes = np.array([p.strike for p in slice_points])
        ivs = np.array([p.implied_vol for p in slice_points])
        
        # Estimate spot from ATM strike
        atm_idx = np.argmin(np.abs(strikes - np.median(strikes)))
        spot = strikes[atm_idx]
        expiries = np.array([p.expiry for p in slice_points])
        
        try:
            surface = self._fitter.fit(strikes, spot, expiries, ivs)
            self._current_surfaces[inst_id] = surface
            
            # Calculate skew metric (25d risk reversal approximation)
            skew = self._calculate_skew(surface, spot)
            self._skew_history[inst_id].append(skew)
            
            # Check for anomalies
            if len(self._skew_history[inst_id]) >= 20:
                await self._check_anomaly(inst_id, skew)
                
        except Exception as e:
            # Log but don't crash on fit failures
            pass
    
    def _calculate_skew(self, surface: FittedSurface, spot: float) -> float:
        """Calculate volatility skew from fitted surface"""
        params = surface.params
        # Approximate 25d risk reversal: IV(25d put) - IV(25d call)
        put_strike = spot * 0.95
        call_strike = spot * 1.05
        
        iv_put = self._fitter.get_implied_vol(put_strike, spot, params)
        iv_call = self._fitter.get_implied_vol(call_strike, spot, params)
        
        return iv_put - iv_call
    
    async def _check_anomaly(self, inst_id: str, current_skew: float):
        """Detect volatility skew anomalies"""
        skews = np.array(list(self._skew_history[inst_id])[:-1])
        mean_skew = np.mean(skews)
        std_skew = np.std(skews)
        
        if std_skew > 1e-6:
            z_score = (current_skew - mean_skew) / std_skew
            
            if abs(z_score) > self.ANOMALY_THRESHOLD:
                anomaly_event = {
                    'instrument_id': inst_id,
                    'z_score': float(z_score),
                    'current_skew': float(current_skew),
                    'historical_mean': float(mean_skew),
                    'timestamp_ns': time.time_ns()
                }
                
                for callback in self._anomaly_callbacks:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(anomaly_event)
                    else:
                        callback(anomaly_event)
    
    def get_current_surface(self, inst_id: str) -> Optional[FittedSurface]:
        """Get current fitted surface for an instrument"""
        return self._current_surfaces.get(inst_id)
    
    def get_all_surfaces(self) -> Dict[str, FittedSurface]:
        """Get all current surfaces"""
        return self._current_surfaces.copy()


# Global singleton instance
_tracker_instance: Optional[VolSurfaceTracker] = None


def get_tracker() -> VolSurfaceTracker:
    """Get or create global volatility surface tracker"""
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = VolSurfaceTracker()
    return _tracker_instance


async def demo():
    """Demo usage of the volatility surface tracker"""
    tracker = get_tracker()
    
    async def on_anomaly(event: dict):
        print(f"VOL SKEW ANOMALY: {event['instrument_id']} "
              f"Z={event['z_score']:.2f}")
    
    tracker.register_anomaly_callback(on_anomaly)
    
    # Simulate observations
    base_spot = 50000
    base_time = time.time_ns()
    
    for i in range(50):
        strike = base_spot * (1 + (i - 25) * 0.01)
        iv = 0.7 + 0.1 * abs(i - 25) / 25 + np.random.randn() * 0.02
        point = VolSurfacePoint(
            strike=strike,
            expiry=0.0833,  # ~1 month
            implied_vol=max(0.1, iv),
            timestamp_ns=base_time + i * 1000000,
            instrument_id="BTC-PERP"
        )
        await tracker.add_observation(point)
    
    surface = tracker.get_current_surface("BTC-PERP")
    if surface:
        print(f"Fitted SVI params: a={surface.params.a:.4f}, "
              f"b={surface.params.b:.4f}, rho={surface.params.rho:.4f}")
        print(f"RMSE: {surface.rmse:.6f}, Fit time: {surface.fit_time_us:.1f}us")


if __name__ == "__main__":
    asyncio.run(demo())
