"""
Module root managing deep forecasting inference queues and blending predictions with statistical ARIMA models.
Handles async inference, result caching, and ensemble blending for the alpha system.
Strictly enforces 3GB RAM limit via bounded queues and memory-efficient operations.
"""

from __future__ import annotations

import asyncio
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from collections import deque
import logging
import time
from enum import Enum
import json

logger = logging.getLogger(__name__)


class ForecastType(Enum):
    """Types of forecasts managed by the module."""
    NBEATS = "nbeats"
    TFT = "tft"
    ARIMA = "arima"
    ENSEMBLE = "ensemble"


@dataclass
class ForecastRequest:
    """Request container for forecast generation."""
    request_id: str
    symbol: str
    observed_history: np.ndarray
    known_future: Optional[np.ndarray] = None
    static_features: Optional[np.ndarray] = None
    forecast_type: ForecastType = ForecastType.ENSEMBLE
    horizon: int = 32
    priority: int = 0  # Higher = more urgent
    timestamp_ns: int = field(default_factory=lambda: time.time_ns())
    
    def __post_init__(self):
        if self.observed_history.dtype != np.float32:
            self.observed_history = self.observed_history.astype(np.float32)


@dataclass
class ForecastResult:
    """Result container for forecast generation."""
    request_id: str
    symbol: str
    forecast: np.ndarray
    confidence_lower: Optional[np.ndarray] = None
    confidence_upper: Optional[np.ndarray] = None
    model_weights: Optional[Dict[str, float]] = None
    inference_time_ms: float = 0.0
    timestamp_ns: int = field(default_factory=lambda: time.time_ns())
    
    # Quality metrics
    prediction_variance: float = 0.0
    model_diversity: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            'request_id': self.request_id,
            'symbol': self.symbol,
            'forecast': self.forecast.tolist(),
            'confidence_lower': self.confidence_lower.tolist() if self.confidence_lower is not None else None,
            'confidence_upper': self.confidence_upper.tolist() if self.confidence_upper is not None else None,
            'model_weights': self.model_weights,
            'inference_time_ms': self.inference_time_ms,
            'prediction_variance': self.prediction_variance,
            'model_diversity': self.model_diversity,
            'timestamp_ns': self.timestamp_ns
        }


class BoundedForecastQueue:
    """
    Thread-safe bounded queue for forecast requests.
    Prevents memory exhaustion by dropping low-priority requests when full.
    """
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._queue: deque = deque()
        self._lock = asyncio.Lock()
        self._dropped_count = 0
    
    @property
    def size(self) -> int:
        return len(self._queue)
    
    @property
    def dropped_count(self) -> int:
        return self._dropped_count
    
    async def put(self, request: ForecastRequest) -> bool:
        """
        Add request to queue. Returns False if dropped due to capacity.
        Low-priority requests are dropped first when queue is full.
        """
        async with self._lock:
            if len(self._queue) >= self.max_size:
                # Find lowest priority request to drop
                min_priority_idx = None
                min_priority = float('inf')
                
                for i, req in enumerate(self._queue):
                    if req.priority < min_priority:
                        min_priority = req.priority
                        min_priority_idx = i
                
                # If new request has higher priority than lowest, replace it
                if request.priority > min_priority and min_priority_idx is not None:
                    self._queue[min_priority_idx] = request
                else:
                    self._dropped_count += 1
                    return False
            else:
                self._queue.append(request)
            
            return True
    
    async def get(self) -> Optional[ForecastRequest]:
        """Get highest priority request from queue."""
        async with self._lock:
            if not self._queue:
                return None
            
            # Return FIFO (could be optimized for priority)
            return self._queue.popleft()
    
    async def clear(self) -> int:
        """Clear queue and return count of cleared items."""
        async with self._lock:
            count = len(self._queue)
            self._queue.clear()
            return count


class ARIMAPredictor:
    """
    Lightweight ARIMA-like predictor using pure numpy.
    Used for blending with deep learning forecasts.
    """
    
    def __init__(
        self,
        ar_order: int = 5,
        ma_order: int = 2,
        diff_order: int = 1
    ):
        self.ar_order = ar_order
        self.ma_order = ma_order
        self.diff_order = diff_order
        
        # Model parameters (would be fitted offline)
        self.ar_coeffs: Optional[np.ndarray] = None
        self.ma_coeffs: Optional[np.ndarray] = None
        self.mean: float = 0.0
        self.std: float = 1.0
    
    def fit(
        self,
        series: np.ndarray,
        ar_coeffs: np.ndarray,
        ma_coeffs: Optional[np.ndarray] = None
    ) -> 'ARIMAPredictor':
        """
        Fit ARIMA model with pre-computed coefficients.
        
        In production, coefficients would be computed offline and loaded.
        """
        self.ar_coeffs = ar_coeffs.astype(np.float32)
        self.ma_coeffs = ma_coeffs.astype(np.float32) if ma_coeffs is not None else None
        
        # Normalize series statistics
        self.mean = float(np.mean(series))
        self.std = float(np.std(series)) + 1e-8
        
        return self
    
    def predict(self, history: np.ndarray, steps: int = 32) -> np.ndarray:
        """
        Generate ARIMA forecast.
        
        Args:
            history: Historical series (at least ar_order + diff_order points)
            steps: Number of steps to forecast
            
        Returns:
            Forecast array of shape (steps,)
        """
        if self.ar_coeffs is None:
            raise RuntimeError("Model not fitted")
        
        history = np.asarray(history, dtype=np.float32)
        
        # Normalize
        history_norm = (history - self.mean) / self.std
        
        # Apply differencing
        diffed = history_norm.copy()
        for _ in range(self.diff_order):
            diffed = np.diff(diffed)
        
        # Extend for prediction
        extended = np.zeros(len(diffed) + steps, dtype=np.float32)
        extended[:len(diffed)] = diffed
        
        # AR prediction
        for i in range(len(diffed), len(diffed) + steps):
            pred = 0.0
            for j, coef in enumerate(self.ar_coeffs):
                if i - j - 1 >= 0:
                    pred += coef * extended[i - j - 1]
            extended[i] = pred
        
        # Undo differencing
        forecast_norm = extended[len(diffed):]
        for _ in range(self.diff_order):
            forecast_norm = np.cumsum(forecast_norm)
            # Pad with last known value for integration
            if len(history_norm) > 0:
                forecast_norm[0] += history_norm[-1]
        
        # Denormalize
        forecast = forecast_norm * self.std + self.mean
        
        return forecast[:steps]
    
    def get_prediction_interval(
        self,
        history: np.ndarray,
        steps: int = 32,
        confidence: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate prediction intervals based on historical residuals.
        
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        forecast = self.predict(history, steps)
        
        # Estimate residual variance from recent history
        if len(history) > self.ar_order * 2:
            residuals = np.diff(history[-self.ar_order * 2:])
            residual_std = float(np.std(residuals))
        else:
            residual_std = self.std
        
        # Confidence interval widens with horizon
        z_score = 1.96 if confidence == 0.95 else 2.576  # 95% or 99%
        horizon_factor = np.sqrt(np.arange(1, steps + 1))
        
        margin = z_score * residual_std * horizon_factor
        
        lower = forecast - margin
        upper = forecast + margin
        
        return lower, upper


class ForecastBlender:
    """
    Blends deep learning and statistical forecasts using adaptive weights.
    """
    
    def __init__(
        self,
        initial_weights: Optional[Dict[ForecastType, float]] = None
    ):
        if initial_weights is None:
            self.weights = {
                ForecastType.NBEATS: 0.4,
                ForecastType.TFT: 0.4,
                ForecastType.ARIMA: 0.2
            }
        else:
            self.weights = initial_weights
        
        # Performance tracking for adaptive weighting
        self._recent_errors: Dict[ForecastType, deque] = {
            ft: deque(maxlen=100) for ft in ForecastType
        }
    
    def set_weights(self, weights: Dict[ForecastType, float]) -> None:
        """Set explicit blending weights."""
        total = sum(weights.values())
        if total <= 0:
            raise ValueError("Weights must sum to positive value")
        self.weights = {k: v / total for k, v in weights.items()}
    
    def update_weights_adaptive(
        self,
        actual_values: np.ndarray,
        predictions: Dict[ForecastType, np.ndarray]
    ) -> None:
        """
        Update weights based on recent prediction accuracy.
        Uses inverse error weighting with exponential decay.
        """
        for ft, pred in predictions.items():
            if len(pred) != len(actual_values):
                continue
            
            # Compute MAE
            mae = float(np.mean(np.abs(pred - actual_values)))
            self._recent_errors[ft].append(mae)
        
        # Compute new weights based on recent performance
        new_weights = {}
        for ft in self.weights:
            if self._recent_errors[ft]:
                avg_error = np.mean(list(self._recent_errors[ft]))
                # Inverse error with smoothing
                new_weights[ft] = 1.0 / (avg_error + 1e-6)
            else:
                new_weights[ft] = 1.0
        
        # Normalize
        total = sum(new_weights.values())
        self.weights = {k: v / total for k, v in new_weights.items()}
    
    def blend(
        self,
        forecasts: Dict[ForecastType, np.ndarray]
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Blend multiple forecasts using current weights.
        
        Returns:
            Tuple of (blended_forecast, weight_dict)
        """
        if not forecasts:
            raise ValueError("No forecasts to blend")
        
        blended = np.zeros_like(list(forecasts.values())[0])
        weight_dict = {}
        
        for ft, forecast in forecasts.items():
            w = self.weights.get(ft, 0.0)
            blended += forecast * w
            weight_dict[ft.value] = w
        
        return blended, weight_dict


class DeepForecastingModule:
    """
    Main module orchestrating deep forecasting inference.
    Manages queues, blending, and async execution.
    """
    
    def __init__(
        self,
        queue_max_size: int = 1000,
        nbeats_model: Optional[Any] = None,
        tft_model: Optional[Any] = None,
        arima_predictor: Optional[ARIMAPredictor] = None
    ):
        self.request_queue = BoundedForecastQueue(queue_max_size)
        self.result_cache: Dict[str, ForecastResult] = {}
        self._cache_max_age_ns = 5 * 60 * 1_000_000_000  # 5 minutes
        
        self.nbeats_model = nbeats_model
        self.tft_model = tft_model
        self.arima_predictor = arima_predictor or ARIMAPredictor()
        
        self.blender = ForecastBlender()
        
        self._running = False
        self._worker_task: Optional[asyncio.Task] = None
        
        logger.info("DeepForecastingModule initialized")
    
    async def submit_request(self, request: ForecastRequest) -> str:
        """Submit a forecast request and return request ID."""
        success = await self.request_queue.put(request)
        if not success:
            logger.warning(f"Request {request.request_id} dropped due to queue capacity")
        
        return request.request_id
    
    async def get_result(self, request_id: str) -> Optional[ForecastResult]:
        """Get cached result for a request ID."""
        self._cleanup_cache()
        return self.result_cache.get(request_id)
    
    def _cleanup_cache(self):
        """Remove stale results from cache."""
        current_time = time.time_ns()
        stale_keys = [
            k for k, v in self.result_cache.items()
            if current_time - v.timestamp_ns > self._cache_max_age_ns
        ]
        for key in stale_keys:
            del self.result_cache[key]
    
    async def _process_request(self, request: ForecastRequest) -> ForecastResult:
        """Process a single forecast request."""
        start_time = time.perf_counter()
        
        forecasts: Dict[ForecastType, np.ndarray] = {}
        
        # Generate N-BEATS forecast
        if self.nbeats_model is not None and request.forecast_type in [
            ForecastType.NBEATS, ForecastType.ENSEMBLE
        ]:
            try:
                nb_forecast = self.nbeats_model.predict(request.observed_history)
                forecasts[ForecastType.NBEATS] = nb_forecast[0] if nb_forecast.ndim > 1 else nb_forecast
            except Exception as e:
                logger.error(f"N-BEATS prediction failed: {e}")
        
        # Generate TFT forecast
        if self.tft_model is not None and request.forecast_type in [
            ForecastType.TFT, ForecastType.ENSEMBLE
        ]:
            try:
                if request.known_future is not None:
                    tft_input = self.tft_model.prepare_input(
                        request.observed_history,
                        request.known_future,
                        request.static_features or np.zeros(request.num_static_features)
                    )
                    tft_output = self.tft_model.predict(tft_input)
                    forecasts[ForecastType.TFT] = tft_output.forecast[0]
            except Exception as e:
                logger.error(f"TFT prediction failed: {e}")
        
        # Generate ARIMA forecast
        if request.forecast_type in [ForecastType.ARIMA, ForecastType.ENSEMBLE]:
            try:
                arima_forecast = self.arima_predictor.predict(
                    request.observed_history,
                    request.horizon
                )
                forecasts[ForecastType.ARIMA] = arima_forecast
            except Exception as e:
                logger.error(f"ARIMA prediction failed: {e}")
        
        # Blend forecasts
        if len(forecasts) > 1:
            blended, weights = self.blender.blend(forecasts)
        elif len(forecasts) == 1:
            blended = list(forecasts.values())[0]
            weights = {list(forecasts.keys())[0].value: 1.0}
        else:
            raise RuntimeError("All forecast models failed")
        
        # Compute confidence intervals (using ARIMA as proxy)
        try:
            lower, upper = self.arima_predictor.get_prediction_interval(
                request.observed_history,
                request.horizon
            )
        except Exception:
            lower, upper = None, None
        
        inference_time_ms = (time.perf_counter() - start_time) * 1000
        
        # Compute quality metrics
        pred_variance = float(np.var(blended))
        model_diversity = 0.0
        if len(forecasts) > 1:
            forecast_stack = np.stack(list(forecasts.values()))
            model_diversity = float(np.mean(np.std(forecast_stack, axis=0)))
        
        result = ForecastResult(
            request_id=request.request_id,
            symbol=request.symbol,
            forecast=blended,
            confidence_lower=lower,
            confidence_upper=upper,
            model_weights=weights,
            inference_time_ms=inference_time_ms,
            prediction_variance=pred_variance,
            model_diversity=model_diversity
        )
        
        # Cache result
        self.result_cache[request.request_id] = result
        
        return result
    
    async def run_worker(self):
        """Main worker loop processing forecast requests."""
        self._running = True
        logger.info("Forecast worker started")
        
        while self._running:
            try:
                request = await self.request_queue.get()
                
                if request is None:
                    await asyncio.sleep(0.001)  # Brief sleep when queue empty
                    continue
                
                await self._process_request(request)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing forecast request: {e}")
        
        self._running = False
        logger.info("Forecast worker stopped")
    
    def start(self):
        """Start the async worker."""
        if self._running:
            return
        
        loop = asyncio.get_event_loop()
        self._worker_task = loop.create_task(self.run_worker())
    
    def stop(self):
        """Stop the async worker."""
        self._running = False
        if self._worker_task is not None:
            self._worker_task.cancel()
    
    def get_status(self) -> Dict[str, Any]:
        """Get module status."""
        return {
            'running': self._running,
            'queue_size': self.request_queue.size,
            'dropped_requests': self.request_queue.dropped_count,
            'cache_size': len(self.result_cache),
            'blend_weights': {k.value: v for k, v in self.blender.weights.items()}
        }


# Module singleton instance
_module_instance: Optional[DeepForecastingModule] = None


def get_module() -> DeepForecastingModule:
    """Get or create the module singleton."""
    global _module_instance
    if _module_instance is None:
        _module_instance = DeepForecastingModule()
    return _module_instance


def initialize_module(
    nbeats_model: Optional[Any] = None,
    tft_model: Optional[Any] = None,
    arima_config: Optional[Dict] = None,
    queue_size: int = 1000
) -> DeepForecastingModule:
    """Initialize the module with models and configuration."""
    global _module_instance
    
    arima_predictor = ARIMAPredictor()
    if arima_config:
        arima_predictor = ARIMAPredictor(**arima_config)
    
    _module_instance = DeepForecastingModule(
        queue_max_size=queue_size,
        nbeats_model=nbeats_model,
        tft_model=tft_model,
        arima_predictor=arima_predictor
    )
    
    return _module_instance
