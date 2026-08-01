"""
Chapter 1: Advanced Order Flow & Footprint Analytics
flow_mod.py - Module root pushing footprint and divergence signals to alpha ensemble
"""

import numpy as np
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
import threading
from collections import deque

# Import local modules
from .footprint_cluster import FootprintClusterEngine, FootprintNode, create_footprint_engine
from .delta_divergence import DeltaDivergenceEngine, DivergenceSignal, create_divergence_engine


@dataclass
class MicrostructureSignal:
    """Unified microstructure signal for alpha ensemble consumption"""
    timestamp: int
    symbol: str
    
    # Footprint metrics
    poc_price: float = 0.0
    vah_price: float = 0.0
    val_price: float = 0.0
    poc_imbalance: float = 0.0
    stacked_imbalance_count: int = 0
    
    # Divergence metrics
    cvd_value: float = 0.0
    cvd_trend: float = 0.0
    divergence_type: int = 0  # 0=none, 1=bullish, 2=bearish
    divergence_strength: float = 0.0
    trapped_trader_index: float = 0.0
    
    # Composite score
    alpha_score: float = 0.0  # -1.0 to 1.0
    confidence: float = 0.0
    
    # Zero-copy data views (memoryviews for efficient transfer)
    footprint_data: Optional[np.ndarray] = None
    cvd_series: Optional[np.ndarray] = None


class ZeroCopyBuffer:
    """
    Thread-safe zero-copy buffer for sharing numpy arrays between threads.
    Uses memoryviews to avoid copying data.
    """
    
    def __init__(self, max_size: int = 10000, dtype: np.dtype = np.float64):
        self._buffer = np.empty(max_size, dtype=dtype)
        self._size = 0
        self._lock = threading.Lock()
        self._version = 0
    
    def write(self, data: np.ndarray) -> int:
        """
        Write data to buffer without copying if shapes match.
        Returns version number for change detection.
        """
        with self._lock:
            n = min(len(data), len(self._buffer))
            # Zero-copy assignment using memoryview
            mv_buffer = memoryview(self._buffer)
            mv_data = memoryview(data)
            
            # Direct memory copy (fastest for contiguous arrays)
            self._buffer[:n] = data[:n]
            self._size = n
            self._version += 1
            
            return self._version
    
    def read(self) -> memoryview:
        """Get memoryview of current data without copying."""
        with self._lock:
            return memoryview(self._buffer[:self._size])
    
    @property
    def version(self) -> int:
        return self._version
    
    @property
    def size(self) -> int:
        return self._size


class MicrostructureFlowModule:
    """
    Central module for processing and distributing microstructure signals.
    Aggregates footprint and divergence analytics into unified alpha signals.
    """
    
    def __init__(
        self,
        tick_size: float = 0.01,
        lookback_levels: int = 50,
        divergence_lookback: int = 20,
        buffer_size: int = 10000
    ):
        # Initialize engines
        self.footprint_engine = create_footprint_engine(tick_size, lookback_levels)
        self.divergence_engine = create_divergence_engine(divergence_lookback)
        
        # Zero-copy buffers for inter-thread communication
        self._footprint_buffer = ZeroCopyBuffer(buffer_size)
        self._cvd_buffer = ZeroCopyBuffer(buffer_size)
        self._price_buffer = ZeroCopyBuffer(buffer_size)
        
        # Signal queue for alpha ensemble
        self._signal_queue: deque = deque(maxlen=1000)
        self._lock = threading.Lock()
        
        # State tracking
        self._last_poc = None
        self._last_vah = None
        self._last_val = None
        self._signal_counter = 0
        
        # Alpha scoring weights
        self.weights = {
            'imbalance': 0.3,
            'divergence': 0.3,
            'tti': 0.2,
            'poc_migration': 0.2
        }
    
    def process_tick_batch(
        self,
        prices: np.ndarray,
        volumes: np.ndarray,
        sides: np.ndarray,
        timestamps: np.ndarray,
        symbol: str
    ) -> Optional[MicrostructureSignal]:
        """
        Process a batch of ticks and generate microstructure signal.
        
        Args:
            prices: Trade prices
            volumes: Trade volumes
            sides: Trade sides (1=ask, -1=bid)
            timestamps: Trade timestamps
            symbol: Trading pair symbol
        
        Returns:
            MicrostructureSignal or None if insufficient data
        """
        if len(prices) < 10:
            return None
        
        # Calculate deltas from sides and volumes
        deltas = sides.astype(np.float64) * volumes
        
        # Process footprint
        footprint_data = self.footprint_engine.process_ticks(prices, volumes, sides)
        
        # Process divergence
        # Use bar-level aggregation for divergence (simplified)
        bar_prices = self._resample_to_bars(prices, timestamps, interval_ms=1000)
        bar_volumes = np.array([volumes.sum()])  # Simplified
        bar_deltas = np.array([deltas.sum()])
        
        # Extend historical context for divergence
        price_history = self._get_price_history(bar_prices[-1])
        volume_history = self._get_volume_history()
        delta_history = self._get_delta_history()
        
        full_prices = np.concatenate([price_history, bar_prices])
        full_volumes = np.concatenate([volume_history, bar_volumes])
        full_deltas = np.concatenate([delta_history, bar_deltas])
        
        divergence_data = self.divergence_engine.analyze(
            full_prices, full_volumes, full_deltas
        )
        
        # Update zero-copy buffers
        self._footprint_buffer.write(footprint_data['total_volumes'])
        self._cvd_buffer.write(divergence_data['cvd'])
        self._price_buffer.write(full_prices)
        
        # Calculate alpha score
        alpha_score, confidence = self._calculate_alpha_score(
            footprint_data, divergence_data
        )
        
        # Extract key levels
        poc_idx = footprint_data['poc_idx']
        poc_price = footprint_data['poc_price']
        vah_idx = footprint_data['vah_idx']
        val_idx = footprint_data['val_idx']
        
        vah_price = footprint_data['price_levels'][vah_idx] if vah_idx < len(footprint_data['price_levels']) else poc_price
        val_price = footprint_data['price_levels'][val_idx] if val_idx < len(footprint_data['price_levels']) else poc_price
        
        # Get POC imbalance
        poc_imbalance = footprint_data['imbalances'][poc_idx] if poc_idx < len(footprint_data['imbalances']) else 0.0
        
        # Determine divergence type
        div_signals = divergence_data['divergence_signals']
        div_type = 0
        div_strength = 0.0
        
        if len(div_signals) > 0:
            last_signal = div_signals[-1]
            div_type = int(last_signal[1])
            div_strength = last_signal[2]
        
        # Create signal
        self._signal_counter += 1
        signal = MicrostructureSignal(
            timestamp=int(timestamps[-1]),
            symbol=symbol,
            poc_price=poc_price,
            vah_price=vah_price,
            val_price=val_price,
            poc_imbalance=poc_imbalance,
            stacked_imbalance_count=len(footprint_data['stacked_imbalances']),
            cvd_value=divergence_data['current_cvd'],
            cvd_trend=divergence_data['cvd_trend'],
            divergence_type=div_type,
            divergence_strength=div_strength,
            trapped_trader_index=divergence_data['current_tti'],
            alpha_score=alpha_score,
            confidence=confidence,
            footprint_data=self._footprint_buffer.read().copy(),  # Copy for signal snapshot
            cvd_series=self._cvd_buffer.read().copy()
        )
        
        # Store in queue
        with self._lock:
            self._signal_queue.append(signal)
        
        # Update state tracking
        self._last_poc = poc_price
        self._last_vah = vah_price
        self._last_val = val_price
        
        return signal
    
    def _resample_to_bars(
        self,
        prices: np.ndarray,
        timestamps: np.ndarray,
        interval_ms: int = 1000
    ) -> np.ndarray:
        """Resample ticks to bars for divergence analysis."""
        if len(prices) == 0:
            return np.array([])
        
        # Simple OHLC resampling (using close price)
        n_bars = max(1, len(prices) // 100)  # Approximate
        bar_prices = np.empty(n_bars, dtype=np.float64)
        
        step = max(1, len(prices) // n_bars)
        for i in range(n_bars):
            start = i * step
            end = min(start + step, len(prices))
            if start < len(prices):
                bar_prices[i] = prices[end - 1]  # Close price
        
        return bar_prices
    
    def _get_price_history(self, last_price: float) -> np.ndarray:
        """Get historical price context."""
        # Return cached history or initialize with last price
        if self._price_buffer.size > 0:
            return self._price_buffer.read().copy()
        return np.array([last_price])
    
    def _get_volume_history(self) -> np.ndarray:
        """Get historical volume context."""
        return np.zeros(20, dtype=np.float64)
    
    def _get_delta_history(self) -> np.ndarray:
        """Get historical delta context."""
        return np.zeros(20, dtype=np.float64)
    
    def _calculate_alpha_score(
        self,
        footprint_data: Dict,
        divergence_data: Dict
    ) -> Tuple[float, float]:
        """
        Calculate composite alpha score from microstructure signals.
        
        Returns:
            Tuple of (score, confidence) where score is in [-1, 1]
        """
        score_components = []
        weight_sum = 0.0
        
        # 1. Imbalance component
        poc_idx = footprint_data['poc_idx']
        imbalances = footprint_data['imbalances']
        
        if poc_idx < len(imbalances):
            poc_imb = imbalances[poc_idx]
            imb_score = poc_imb  # Already normalized [-1, 1]
            score_components.append(imb_score * self.weights['imbalance'])
            weight_sum += self.weights['imbalance']
        
        # 2. Divergence component
        div_signals = divergence_data['divergence_signals']
        if len(div_signals) > 0:
            last_signal = div_signals[-1]
            div_type = int(last_signal[1])
            div_strength = last_signal[2]
            
            if div_type == 1:  # Bullish
                div_score = div_strength
            elif div_type == 2:  # Bearish
                div_score = -div_strength
            else:
                div_score = 0.0
            
            score_components.append(div_score * self.weights['divergence'])
            weight_sum += self.weights['divergence']
        
        # 3. Trapped Trader Index component
        tti = divergence_data['current_tti']
        cvd_trend = divergence_data['cvd_trend']
        
        # High TTI + positive CVD trend = bullish (shorts trapped)
        # High TTI + negative CVD trend = bearish (longs trapped)
        if tti > 0.3:
            tti_score = np.sign(cvd_trend) * tti
            score_components.append(tti_score * self.weights['tti'])
            weight_sum += self.weights['tti']
        
        # 4. POC migration component
        if self._last_poc is not None:
            poc_price = footprint_data['poc_price']
            poc_change = (poc_price - self._last_poc) / max(self._last_poc, 1e-10)
            
            # POC moving up = bullish, POC moving down = bearish
            poc_score = np.clip(poc_change * 1000, -1, 1)  # Scale appropriately
            score_components.append(poc_score * self.weights['poc_migration'])
            weight_sum += self.weights['poc_migration']
        
        # Calculate final score
        if weight_sum > 0 and len(score_components) > 0:
            raw_score = sum(score_components) / weight_sum
            alpha_score = np.clip(raw_score, -1.0, 1.0)
        else:
            alpha_score = 0.0
        
        # Calculate confidence based on signal convergence
        if len(score_components) >= 3:
            # Higher confidence when signals agree
            signs = [np.sign(s) for s in score_components if s != 0]
            if len(signs) > 0:
                agreement = abs(sum(signs)) / len(signs)
                confidence = 0.5 + (agreement * 0.5)
            else:
                confidence = 0.5
        else:
            confidence = 0.3
        
        return alpha_score, confidence
    
    def get_latest_signal(self) -> Optional[MicrostructureSignal]:
        """Get most recent signal from queue."""
        with self._lock:
            if len(self._signal_queue) > 0:
                return self._signal_queue[-1]
        return None
    
    def get_signals(self, count: int = 10) -> list:
        """Get last N signals from queue."""
        with self._lock:
            signals = list(self._signal_queue)
            return signals[-count:]
    
    def get_footprint_memoryview(self) -> memoryview:
        """Get zero-copy view of latest footprint data."""
        return self._footprint_buffer.read()
    
    def get_cvd_memoryview(self) -> memoryview:
        """Get zero-copy view of CVD series."""
        return self._cvd_buffer.read()


# Module singleton instance
_flow_module: Optional[MicrostructureFlowModule] = None


def get_flow_module(
    tick_size: float = 0.01,
    lookback_levels: int = 50
) -> MicrostructureFlowModule:
    """Get or create the global flow module instance."""
    global _flow_module
    if _flow_module is None:
        _flow_module = MicrostructureFlowModule(tick_size, lookback_levels)
    return _flow_module


def reset_flow_module():
    """Reset the global flow module (for testing)."""
    global _flow_module
    _flow_module = None
