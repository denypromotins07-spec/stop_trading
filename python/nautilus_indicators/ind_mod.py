"""
Nautilus Indicators Module Root
Registers ultra-fast, zero-allocation indicators with the Nautilus DataEngine 
for tick-by-tick updates.

Provides unified interface for Numba and Cython-accelerated indicators.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
import threading
import time

from .numba_rsi import (
    compute_rsi, compute_macd, compute_bollinger_bands,
    compute_atr, NumbaIndicatorEngine
)
from .cython_vwap import (
    CythonVWAPCalculator, VolumeProfileCalculator, CythonIndicatorEngine
)


@dataclass
class IndicatorConfig:
    """Configuration for indicator engine."""
    # RSI settings
    rsi_period: int = 14
    
    # MACD settings
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    # Bollinger Bands settings
    bb_period: int = 20
    bb_std: float = 2.0
    
    # ATR settings
    atr_period: int = 14
    
    # VWAP settings
    vwap_max_periods: int = 10000
    
    # Volume Profile settings
    vp_num_bins: int = 100
    vp_price_range_pct: float = 2.0
    
    # Performance settings
    max_history: int = 1000
    update_interval_ms: float = 0.0  # 0 = every tick


class NautilusIndicatorRegistry:
    """
    Registry for all available indicators.
    Manages indicator lifecycle and registration.
    """
    
    def __init__(self):
        self._indicators: Dict[str, Callable] = {}
        self._engines: Dict[str, Any] = {}
        self._lock = threading.Lock()
        
        # Register built-in indicators
        self._register_builtin_indicators()
    
    def _register_builtin_indicators(self):
        """Register built-in indicators."""
        # Numba indicators
        self.register('rsi', lambda config: NumbaIndicatorEngine(
            rsi_period=config.rsi_period
        ))
        self.register('macd', lambda config: NumbaIndicatorEngine(
            macd_fast=config.macd_fast,
            macd_slow=config.macd_slow,
            macd_signal=config.macd_signal
        ))
        self.register('bollinger', lambda config: NumbaIndicatorEngine(
            bb_period=config.bb_period,
            bb_std=config.bb_std
        ))
        self.register('atr', lambda config: NumbaIndicatorEngine(
            atr_period=config.atr_period
        ))
        
        # Cython-style indicators
        self.register('vwap', lambda config: CythonVWAPCalculator(
            max_periods=config.vwap_max_periods
        ))
        self.register('volume_profile', lambda config: VolumeProfileCalculator(
            num_bins=config.vp_num_bins,
            price_range_pct=config.vp_price_range_pct
        ))
    
    def register(self, name: str, factory: Callable):
        """Register an indicator."""
        with self._lock:
            self._indicators[name] = factory
    
    def get_indicator(self, name: str, config: IndicatorConfig) -> Any:
        """Get or create an indicator instance."""
        with self._lock:
            if name not in self._indicators:
                raise ValueError(f"Unknown indicator: {name}")
            
            if name not in self._engines:
                self._engines[name] = self._indicators[name](config)
            
            return self._engines[name]
    
    def reset(self):
        """Reset all indicators."""
        with self._lock:
            for name, engine in self._engines.items():
                if hasattr(engine, 'reset'):
                    engine.reset()


class UnifiedIndicatorEngine:
    """
    Unified engine combining all indicator types.
    Provides single interface for Nautilus DataEngine integration.
    """
    
    def __init__(self, config: Optional[IndicatorConfig] = None):
        self.config = config or IndicatorConfig()
        
        # Initialize sub-engines
        self.numba_engine = NumbaIndicatorEngine(
            rsi_period=self.config.rsi_period,
            macd_fast=self.config.macd_fast,
            macd_slow=self.config.macd_slow,
            macd_signal=self.config.macd_signal,
            bb_period=self.config.bb_period,
            bb_std=self.config.bb_std,
            atr_period=self.config.atr_period,
            max_history=self.config.max_history
        )
        
        self.cython_engine = CythonIndicatorEngine(
            vwap_max_periods=self.config.vwap_max_periods,
            vp_num_bins=self.config.vp_num_bins,
            vp_price_range_pct=self.config.vp_price_range_pct
        )
        
        # Statistics
        self._tick_count = 0
        self._last_update_ns = 0
        self._avg_latency_ns = 0
        
        # Session tracking
        self._session_active = False
    
    def on_tick(self, 
                timestamp: float,
                price: float,
                high: float,
                low: float,
                volume: float) -> Dict[str, Any]:
        """
        Process a tick through all indicators.
        
        Args:
            timestamp: Tick timestamp
            price: Current price
            high: Period high
            low: Period low
            volume: Trade volume
            
        Returns:
            Dictionary with all indicator values
        """
        start_ns = time.perf_counter_ns()
        
        # Update Numba indicators
        numba_result = self.numba_engine.update(price, high, low, price)
        
        # Update Cython-style indicators
        cython_result = self.cython_engine.on_tick(price, volume)
        
        # Combine results
        result = {
            'timestamp': timestamp,
            **numba_result,
            **cython_result
        }
        
        # Update statistics
        self._tick_count += 1
        latency_ns = time.perf_counter_ns() - start_ns
        self._last_update_ns = latency_ns
        
        # Running average latency
        alpha = 0.01
        self._avg_latency_ns = (
            (1 - alpha) * self._avg_latency_ns + alpha * latency_ns
        )
        
        return result
    
    def on_bar(self,
               timestamp: float,
               open_: float,
               high: float,
               low: float,
               close: float,
               volume: float) -> Dict[str, Any]:
        """
        Process a bar through all indicators.
        
        Args:
            timestamp: Bar timestamp
            open_: Open price
            high: High price
            low: Low price
            close: Close price
            volume: Bar volume
            
        Returns:
            Dictionary with all indicator values
        """
        return self.on_tick(timestamp, close, high, low, volume)
    
    def on_session_start(self):
        """Handle new session start."""
        self._session_active = True
        self.cython_engine.on_session_start()
    
    def on_session_end(self, close_price: float):
        """Handle session end."""
        self.cython_engine.on_session_end(close_price)
        self._session_active = False
    
    def get_snapshot(self) -> Dict[str, Any]:
        """Get current indicator snapshot."""
        return {
            'numba': self.numba_engine.get_latest(),
            'cython': self.cython_engine.get_all_indicators(),
            'statistics': {
                'tick_count': self._tick_count,
                'last_latency_ns': self._last_update_ns,
                'avg_latency_ns': self._avg_latency_ns
            }
        }
    
    def reset(self):
        """Reset all indicators."""
        self.numba_engine.reset()
        self.cython_engine = CythonIndicatorEngine(
            vwap_max_periods=self.config.vwap_max_periods,
            vp_num_bins=self.config.vp_num_bins,
            vp_price_range_pct=self.config.vp_price_range_pct
        )
        self._tick_count = 0
        self._last_update_ns = 0
        self._avg_latency_ns = 0


class NautilusDataEngineAdapter:
    """
    Adapter for integrating with Nautilus DataEngine.
    Handles subscription and callback routing.
    """
    
    def __init__(self, 
                 indicator_engine: UnifiedIndicatorEngine,
                 instrument_id: str = "BTC/USDT"):
        """
        Initialize adapter.
        
        Args:
            indicator_engine: Indicator engine to use
            instrument_id: Instrument identifier
        """
        self.indicator_engine = indicator_engine
        self.instrument_id = instrument_id
        
        self._callbacks: List[Callable[[Dict], None]] = []
        self._subscribed = False
    
    def subscribe(self):
        """Subscribe to market data."""
        self._subscribed = True
    
    def unsubscribe(self):
        """Unsubscribe from market data."""
        self._subscribed = False
    
    def register_callback(self, callback: Callable[[Dict], None]):
        """Register callback for indicator updates."""
        self._callbacks.append(callback)
    
    def handle_quote(self, quote: Dict[str, Any]):
        """
        Handle incoming quote.
        
        Args:
            quote: Quote dictionary with bid, ask, etc.
        """
        if not self._subscribed:
            return
        
        mid_price = (quote.get('bid', 0) + quote.get('ask', 0)) / 2
        
        result = self.indicator_engine.on_tick(
            timestamp=quote.get('timestamp', time.time()),
            price=mid_price,
            high=quote.get('high', mid_price),
            low=quote.get('low', mid_price),
            volume=quote.get('volume', 0)
        )
        
        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback(result)
            except Exception:
                pass
    
    def handle_trade(self, trade: Dict[str, Any]):
        """
        Handle incoming trade.
        
        Args:
            trade: Trade dictionary with price, volume
        """
        if not self._subscribed:
            return
        
        result = self.indicator_engine.on_tick(
            timestamp=trade.get('timestamp', time.time()),
            price=trade.get('price', 0),
            high=trade.get('price', 0),
            low=trade.get('price', 0),
            volume=trade.get('volume', 0)
        )
        
        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback(result)
            except Exception:
                pass
    
    def handle_bar(self, bar: Dict[str, Any]):
        """
        Handle incoming bar.
        
        Args:
            bar: Bar dictionary with OHLCV
        """
        if not self._subscribed:
            return
        
        result = self.indicator_engine.on_bar(
            timestamp=bar.get('timestamp', time.time()),
            open_=bar.get('open', 0),
            high=bar.get('high', 0),
            low=bar.get('low', 0),
            close=bar.get('close', 0),
            volume=bar.get('volume', 0)
        )
        
        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback(result)
            except Exception:
                pass


# Module-level singleton
_registry: Optional[NautilusIndicatorRegistry] = None
_adapter: Optional[NautilusDataEngineAdapter] = None
_lock = threading.Lock()


def get_registry() -> NautilusIndicatorRegistry:
    """Get or create global registry."""
    global _registry
    
    with _lock:
        if _registry is None:
            _registry = NautilusIndicatorRegistry()
        return _registry


def get_adapter(instrument_id: str = "BTC/USDT") -> NautilusDataEngineAdapter:
    """Get or create global adapter."""
    global _adapter, _registry
    
    with _lock:
        if _registry is None:
            _registry = NautilusIndicatorRegistry()
        
        if _adapter is None:
            engine = UnifiedIndicatorEngine()
            _adapter = NautilusDataEngineAdapter(engine, instrument_id)
        
        return _adapter


def create_indicator_engine(config: Optional[IndicatorConfig] = None) -> UnifiedIndicatorEngine:
    """Create a new indicator engine."""
    return UnifiedIndicatorEngine(config)


# Module exports
__all__ = [
    'IndicatorConfig',
    'NautilusIndicatorRegistry',
    'UnifiedIndicatorEngine',
    'NautilusDataEngineAdapter',
    'get_registry',
    'get_adapter',
    'create_indicator_engine'
]
