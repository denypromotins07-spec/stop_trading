"""
Cross-Exchange & Statistical Arbitrage Spread Monitor.
Consumes normalized cross-venue and triangular spreads from Rust IPC bridge.
Uses zero-copy numpy arrays for memory efficiency.
Triggers Python-side stat-arb execution on volatility-adjusted Z-score breaches.
Optimized for 3GB RAM constraint.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Any
from collections import deque
import numpy as np
from datetime import datetime, timezone
import time

logger = logging.getLogger(__name__)


@dataclass
class SpreadSignal:
    """Represents a statistical arbitrage signal."""
    pair_id: str
    venue_a: str
    venue_b: str
    spread_bps: float
    z_score: float
    threshold: float
    timestamp_ns: int
    direction: int  # 1 = long A/short B, -1 = short A/long B
    confidence: float
    volume_imbalance: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'pair_id': self.pair_id,
            'venue_a': self.venue_a,
            'venue_b': self.venue_b,
            'spread_bps': self.spread_bps,
            'z_score': self.z_score,
            'threshold': self.threshold,
            'timestamp_ns': self.timestamp_ns,
            'direction': self.direction,
            'confidence': self.confidence,
            'volume_imbalance': self.volume_imbalance,
        }


@dataclass
class SpreadWindow:
    """Memory-efficient rolling window for spread calculations."""
    max_size: int = 1000
    values: deque = field(default_factory=lambda: deque(maxlen=1000))
    sum_x: float = 0.0
    sum_x2: float = 0.0
    
    def add(self, value: float):
        """Add value to rolling window with Welford's algorithm."""
        if len(self.values) == self.max_size:
            old_value = self.values[0]
            self.sum_x -= old_value
            self.sum_x2 -= old_value ** 2
        
        self.values.append(value)
        self.sum_x += value
        self.sum_x2 += value ** 2
    
    @property
    def mean(self) -> float:
        """Calculate rolling mean."""
        n = len(self.values)
        if n == 0:
            return 0.0
        return self.sum_x / n
    
    @property
    def std(self) -> float:
        """Calculate rolling standard deviation."""
        n = len(self.values)
        if n < 2:
            return 1.0  # Default to prevent division by zero
        
        variance = (self.sum_x2 / n) - (self.mean ** 2)
        # Bessel's correction for sample std
        variance = variance * n / (n - 1)
        return max(np.sqrt(variance), 1e-6)  # Floor to prevent division by zero
    
    def z_score(self, value: float) -> float:
        """Calculate Z-score for a value."""
        return (value - self.mean) / self.std


class SpreadMonitor:
    """
    Monitors cross-exchange and triangular spreads for arbitrage opportunities.
    Uses zero-copy numpy arrays from Rust IPC bridge.
    Implements dynamic, volatility-adjusted Z-score thresholds.
    """
    
    def __init__(self, 
                 default_threshold: float = 2.5,
                 min_samples: int = 50,
                 volatility_lookback: int = 200):
        """
        Initialize spread monitor.
        
        Args:
            default_threshold: Default Z-score threshold for signals
            min_samples: Minimum samples before generating signals
            volatility_lookback: Lookback window for volatility estimation
        """
        self._spreads: Dict[str, SpreadWindow] = {}
        self._volatility: Dict[str, float] = {}
        self._thresholds: Dict[str, float] = {}
        self._default_threshold = default_threshold
        self._min_samples = min_samples
        self._volatility_lookback = volatility_lookback
        
        # Signal callbacks
        self._signal_callbacks: List[Callable[[SpreadSignal], Any]] = []
        
        # Triangular arbitrage tracking
        self._triangular_spreads: Dict[str, Tuple[str, str, str]] = {}
        
        # Statistics
        self._stats = {
            'signals_generated': 0,
            'threshold_breaches': 0,
            'pairs_monitored': 0,
        }
        
        # Memory tracking
        self._max_pairs = 500  # Limit concurrent pairs to respect RAM
        
    def register_pair(self, pair_id: str, venue_a: str, venue_b: str,
                     initial_spread_bps: float = 0.0):
        """
        Register a new pair to monitor.
        
        Args:
            pair_id: Unique identifier for the pair
            venue_a: First exchange/venue
            venue_b: Second exchange/venue
            initial_spread_bps: Initial spread in basis points
        """
        if len(self._spreads) >= self._max_pairs:
            logger.warning(f"Max pairs ({self._max_pairs}) reached, dropping oldest")
            # Remove oldest pair (FIFO)
            oldest_key = next(iter(self._spreads))
            del self._spreads[oldest_key]
            if oldest_key in self._volatility:
                del self._volatility[oldest_key]
            if oldest_key in self._thresholds:
                del self._thresholds[oldest_key]
        
        self._spreads[pair_id] = SpreadWindow(max_size=self._volatility_lookback)
        self._volatility[pair_id] = 1.0
        self._thresholds[pair_id] = self._default_threshold
        
        # Add initial value if provided
        if initial_spread_bps != 0.0:
            self._spreads[pair_id].add(initial_spread_bps)
            
        self._stats['pairs_monitored'] = len(self._spreads)
        logger.debug(f"Registered pair {pair_id} ({venue_a} vs {venue_b})")
    
    def register_triangular(self, triangle_id: str, 
                           leg1: str, leg2: str, leg3: str):
        """
        Register a triangular arbitrage opportunity.
        
        Args:
            triangle_id: Unique identifier for the triangle
            leg1: First pair (e.g., BTC/USD)
            leg2: Second pair (e.g., ETH/BTC)
            leg3: Third pair (e.g., ETH/USD)
        """
        self._triangular_spreads[triangle_id] = (leg1, leg2, leg3)
    
    def update_spread(self, pair_id: str, spread_bps: float, 
                     timestamp_ns: Optional[int] = None,
                     volume_imbalance: float = 0.0) -> Optional[SpreadSignal]:
        """
        Update spread for a pair and check for arbitrage signals.
        
        Args:
            pair_id: Pair identifier
            spread_bps: Current spread in basis points
            timestamp_ns: Nanosecond timestamp
            volume_imbalance: Volume imbalance between venues
            
        Returns:
            SpreadSignal if threshold breached, None otherwise
        """
        if timestamp_ns is None:
            timestamp_ns = time.time_ns()
        
        # Ensure pair is registered
        if pair_id not in self._spreads:
            self.register_pair(pair_id, "unknown", "unknown", spread_bps)
            return None
        
        window = self._spreads[pair_id]
        
        # Calculate current Z-score before adding new value
        if len(window.values) >= self._min_samples:
            z_score = window.z_score(spread_bps)
            
            # Update rolling volatility estimate
            self._volatility[pair_id] = window.std
            
            # Dynamic threshold adjustment based on volatility regime
            threshold = self._calculate_dynamic_threshold(pair_id, spread_bps)
            
            # Check for threshold breach
            if abs(z_score) > threshold:
                self._stats['threshold_breaches'] += 1
                
                # Determine direction
                direction = 1 if z_score > 0 else -1
                
                # Calculate confidence based on Z-score magnitude
                confidence = min(1.0, abs(z_score) / (threshold * 2))
                
                signal = SpreadSignal(
                    pair_id=pair_id,
                    venue_a="venue_a",  # Would come from registration
                    venue_b="venue_b",
                    spread_bps=spread_bps,
                    z_score=z_score,
                    threshold=threshold,
                    timestamp_ns=timestamp_ns,
                    direction=direction,
                    confidence=confidence,
                    volume_imbalance=volume_imbalance,
                )
                
                self._stats['signals_generated'] += 1
                
                # Notify callbacks
                for callback in self._signal_callbacks:
                    try:
                        callback(signal)
                    except Exception as e:
                        logger.error(f"Signal callback error: {e}")
                
                # Add to window after signal generation
                window.add(spread_bps)
                
                return signal
        
        # Add to window
        window.add(spread_bps)
        return None
    
    def _calculate_dynamic_threshold(self, pair_id: str, 
                                    spread_bps: float) -> float:
        """
        Calculate dynamic threshold based on volatility regime.
        Widens thresholds in high volatility to reduce false signals.
        
        Args:
            pair_id: Pair identifier
            spread_bps: Current spread
            
        Returns:
            Dynamic Z-score threshold
        """
        base_threshold = self._default_threshold
        volatility = self._volatility.get(pair_id, 1.0)
        
        # Volatility scaling factor
        # Higher volatility = wider thresholds
        vol_scale = 1.0 + (volatility / 10.0)  # Adjust divisor based on typical vol
        
        # Mean reversion adjustment
        # If spread is extremely wide, tighten threshold slightly
        window = self._spreads[pair_id]
        if len(window.values) > 0:
            deviation = abs(spread_bps - window.mean) / max(window.mean, 1.0)
            if deviation > 3.0:
                vol_scale *= 0.9  # Slightly tighter on extreme moves
        
        dynamic_threshold = base_threshold * vol_scale
        
        # Cache the threshold
        self._thresholds[pair_id] = dynamic_threshold
        
        return dynamic_threshold
    
    def process_zero_copy_update(self, pair_ids: np.ndarray,
                                 spreads: np.ndarray,
                                 timestamps: np.ndarray):
        """
        Process batch update from Rust IPC bridge using zero-copy numpy arrays.
        
        Args:
            pair_ids: Array of pair IDs (as integers or encoded strings)
            spreads: Array of spread values in bps
            timestamps: Array of nanosecond timestamps
        """
        # Validate array lengths match
        if not (len(pair_ids) == len(spreads) == len(timestamps)):
            logger.error("Array length mismatch in zero-copy update")
            return
        
        # Process each update efficiently
        for i in range(len(pair_ids)):
            pair_id = str(pair_ids[i])
            spread = float(spreads[i])
            ts = int(timestamps[i])
            
            self.update_spread(pair_id, spread, ts)
    
    def register_signal_callback(self, callback: Callable[[SpreadSignal], Any]):
        """Register callback for arbitrage signals."""
        self._signal_callbacks.append(callback)
    
    def get_current_z_scores(self, pair_id: Optional[str] = None) -> Dict[str, float]:
        """
        Get current Z-scores for monitored pairs.
        
        Args:
            pair_id: Optional specific pair (None for all)
            
        Returns:
            Dict mapping pair IDs to Z-scores
        """
        result = {}
        
        pairs = [pair_id] if pair_id else list(self._spreads.keys())
        
        for pid in pairs:
            if pid not in self._spreads:
                continue
            window = self._spreads[pid]
            if len(window.values) == 0:
                continue
            
            # Use latest value for Z-score
            latest = window.values[-1]
            result[pid] = window.z_score(latest)
        
        return result
    
    def get_volatility_surface(self) -> Dict[str, float]:
        """Get current volatility estimates for all pairs."""
        return self._volatility.copy()
    
    def get_stats(self) -> Dict[str, int]:
        """Get monitoring statistics."""
        return self._stats.copy()
    
    def calculate_triangular_spread(self, triangle_id: str,
                                   prices: Dict[str, float]) -> Optional[float]:
        """
        Calculate implied spread for triangular arbitrage.
        
        Args:
            triangle_id: Triangle identifier
            prices: Dict mapping pair names to current prices
            
        Returns:
            Implied spread in bps, or None if calculation not possible
        """
        if triangle_id not in self._triangular_spreads:
            return None
        
        leg1, leg2, leg3 = self._triangular_spreads[triangle_id]
        
        # Get prices for all legs
        p1 = prices.get(leg1)
        p2 = prices.get(leg2)
        p3 = prices.get(leg3)
        
        if None in (p1, p2, p3):
            return None
        
        # Calculate theoretical price of leg3 via legs 1 and 2
        # Example: BTC/USD * ETH/BTC should equal ETH/USD
        theoretical = p1 * p2
        
        # Calculate spread
        spread_bps = abs(theoretical - p3) / p3 * 10000
        
        return spread_bps
