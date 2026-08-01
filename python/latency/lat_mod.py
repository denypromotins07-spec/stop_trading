"""
Chapter 4: Latency Arbitrage & Queue Position ML Prediction
lat_mod.py - Module root routing latency arb signals to execution engine
"""

import numpy as np
from typing import Dict, Optional, Tuple, List, Any
from dataclasses import dataclass, field
import threading
from collections import deque
from enum import IntEnum

# Import local modules
from .queue_predictor import (
    QueuePositionPredictor,
    QueuePositionFeatures,
    create_queue_predictor,
    calculate_fill_probability
)
from .cross_venue_lag import (
    CrossVenueLagTracker,
    VenuePairLag,
    create_lag_tracker,
    estimate_latency_ms
)


class SignalType(IntEnum):
    """Types of latency arbitrage signals"""
    NONE = 0
    CROSS_VENUE_LAG = 1      # Price discrepancy between venues
    QUEUE_POSITION = 2       # Favorable queue position detected
    STATISTICAL_ARBITRAGE = 3  # Statistical front-running opportunity
    LIQUIDITY_TAKING = 4     # Consume liquidity before price moves


@dataclass
class LatencyArbitrageSignal:
    """Unified latency arbitrage signal for execution engine"""
    timestamp_ns: int
    signal_type: SignalType
    
    # Venue information
    fast_venue: str
    slow_venue: str
    symbol: str
    
    # Price/size info
    fast_price: float
    slow_price: float
    order_size: float
    expected_slippage: float
    
    # Timing
    estimated_latency_ms: float
    time_to_fill_estimate_ms: float
    signal_half_life_ms: float
    
    # Confidence metrics
    confidence: float
    fill_probability: float
    edge_bps: float  # Expected edge in basis points
    
    # Execution parameters
    urgency: int  # 1-5 scale
    max_participation_rate: float  # % of volume
    aggressive: bool  # Whether to use market orders
    
    # State
    active: bool = True
    executed: bool = False


class LatencyArbitrageModule:
    """
    Central module for detecting and routing latency arbitrage opportunities.
    Combines queue prediction and cross-venue lag tracking.
    """
    
    def __init__(
        self,
        venue_names: List[str],
        default_venues: Tuple[str, str] = None,
        min_edge_bps: float = 2.0,
        max_latency_ms: float = 50.0
    ):
        self.venue_names = venue_names
        self.default_venues = default_venues or (venue_names[0], venue_names[1]) if len(venue_names) >= 2 else ('A', 'B')
        self.min_edge_bps = min_edge_bps
        self.max_latency_ms = max_latency_ms
        
        # Initialize sub-modules
        self._lag_tracker = create_lag_tracker(venue_names)
        self._queue_predictors: Dict[str, QueuePositionPredictor] = {}
        
        for venue in venue_names:
            self._queue_predictors[venue] = create_queue_predictor()
        
        # Signal queue
        self._signal_queue: deque = deque(maxlen=1000)
        self._lock = threading.Lock()
        
        # State tracking
        self._last_prices: Dict[str, Dict[str, float]] = {v: {} for v in venue_names}
        self._active_signals: Dict[str, LatencyArbitrageSignal] = {}
        
        # Statistics
        self._signals_generated = 0
        self._signals_executed = 0
        self._total_edge_captured = 0.0
    
    def update_price(
        self,
        venue: str,
        symbol: str,
        timestamp_ns: int,
        price: float,
        bid_depths: Optional[np.ndarray] = None,
        ask_depths: Optional[np.ndarray] = None
    ) -> Optional[LatencyArbitrageSignal]:
        """
        Update price from a venue and check for arbitrage opportunities.
        
        Args:
            venue: Venue identifier
            symbol: Trading pair symbol
            timestamp_ns: Timestamp in nanoseconds
            price: Current mid price
            bid_depths: Bid side depth at top levels
            ask_depths: Ask side depth at top levels
        
        Returns:
            LatencyArbitrageSignal if opportunity detected, None otherwise
        """
        # Update lag tracker
        self._lag_tracker.add_observation(venue, timestamp_ns, price)
        
        # Update last prices
        self._last_prices[venue][symbol] = price
        
        # Update queue predictor if depths provided
        if bid_depths is not None and ask_depths is not None and venue in self._queue_predictors:
            self._queue_predictors[venue].record_trade(price, 1.0, 1)
        
        # Check for cross-venue arbitrage
        signal = self._check_cross_venue_arb(
            venue, symbol, timestamp_ns, price
        )
        
        if signal is not None:
            with self._lock:
                self._signal_queue.append(signal)
                self._active_signals[f"{symbol}_{venue}"] = signal
                self._signals_generated += 1
        
        return signal
    
    def _check_cross_venue_arb(
        self,
        fast_venue: str,
        symbol: str,
        timestamp_ns: int,
        fast_price: float
    ) -> Optional[LatencyArbitrageSignal]:
        """Check for cross-venue arbitrage opportunity."""
        slow_venue = self.default_venues[1] if fast_venue == self.default_venues[0] else self.default_venues[0]
        
        # Get lag estimate
        lag_info = self._lag_tracker.get_lag_estimate(fast_venue, slow_venue)
        
        if lag_info is None:
            return None
        
        # Check if lag is significant
        if abs(lag_info.estimated_lag_ms) < 1.0:
            return None
        
        if lag_info.confidence < 0.3:
            return None
        
        # Get slow venue price
        if slow_venue not in self._last_prices or symbol not in self._last_prices[slow_venue]:
            return None
        
        slow_price = self._last_prices[slow_venue][symbol]
        
        # Calculate price discrepancy
        price_diff = fast_price - slow_price
        avg_price = (fast_price + slow_price) / 2
        
        if avg_price == 0:
            return None
        
        edge_bps = abs(price_diff) / avg_price * 10000
        
        # Check minimum edge threshold
        if edge_bps < self.min_edge_bps:
            return None
        
        # Determine direction
        if lag_info.lead_lag_direction > 0:
            # Fast venue leads - expect slow venue to follow
            if price_diff > 0:
                # Fast moved up, slow should follow - buy on slow
                direction = 'buy_slow'
            else:
                # Fast moved down, slow should follow - sell on slow
                direction = 'sell_slow'
        else:
            return None
        
        # Estimate fill probability
        fill_prob = calculate_fill_probability(
            queue_position=10,  # Assumed position
            depletion_rate=100.0,  # Assumed rate
            spread=abs(price_diff),
            volatility=0.001
        )
        
        # Calculate expected slippage
        expected_slippage = avg_price * 0.0001  # 1 bps assumption
        
        # Signal half-life based on lag
        signal_half_life = abs(lag_info.estimated_lag_ms) * 2
        
        # Urgency based on edge and latency
        urgency = min(5, max(1, int(edge_bps / self.min_edge_bps)))
        
        signal = LatencyArbitrageSignal(
            timestamp_ns=timestamp_ns,
            signal_type=SignalType.CROSS_VENUE_LAG,
            fast_venue=fast_venue,
            slow_venue=slow_venue,
            symbol=symbol,
            fast_price=fast_price,
            slow_price=slow_price,
            order_size=1.0,  # Would be determined by risk module
            expected_slippage=expected_slippage,
            estimated_latency_ms=lag_info.estimated_lag_ms,
            time_to_fill_estimate_ms=signal_half_life / 2,
            signal_half_life_ms=signal_half_life,
            confidence=lag_info.confidence,
            fill_probability=fill_prob,
            edge_bps=edge_bps,
            urgency=urgency,
            max_participation_rate=0.1,  # 10% of volume
            aggressive=(urgency >= 4)
        )
        
        return signal
    
    def get_queue_prediction(
        self,
        venue: str,
        bid_depths: np.ndarray,
        ask_depths: np.ndarray,
        mid_price: float,
        spread: float,
        order_size: float,
        side: int
    ) -> Tuple[float, float]:
        """
        Get time-to-fill prediction for a limit order.
        
        Returns:
            Tuple of (predicted_time_ms, confidence)
        """
        if venue not in self._queue_predictors:
            return 60000.0, 0.3  # Default 1 minute, low confidence
        
        predictor = self._queue_predictors[venue]
        
        features = predictor.extract_features(
            bid_depths, ask_depths, mid_price, spread, order_size, side
        )
        
        time_pred, confidence = predictor.predict_time_to_fill(features)
        
        return time_pred * 1000, confidence  # Convert to ms
    
    def mark_signal_executed(self, signal_id: str) -> bool:
        """Mark a signal as executed."""
        with self._lock:
            if signal_id in self._active_signals:
                signal = self._active_signals[signal_id]
                signal.executed = True
                signal.active = False
                self._signals_executed += 1
                self._total_edge_captured += signal.edge_bps
                return True
        return False
    
    def cancel_signal(self, signal_id: str) -> bool:
        """Cancel an active signal."""
        with self._lock:
            if signal_id in self._active_signals:
                self._active_signals[signal_id].active = False
                del self._active_signals[signal_id]
                return True
        return False
    
    def get_active_signals(self) -> List[LatencyArbitrageSignal]:
        """Get all active signals."""
        with self._lock:
            return [s for s in self._active_signals.values() if s.active]
    
    def get_latest_signals(self, count: int = 10) -> List[LatencyArbitrageSignal]:
        """Get most recent signals."""
        with self._lock:
            signals = list(self._signal_queue)
            return signals[-count:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get module statistics."""
        with self._lock:
            return {
                'signals_generated': self._signals_generated,
                'signals_executed': self._signals_executed,
                'execution_rate': self._signals_executed / max(1, self._signals_generated),
                'total_edge_captured_bps': self._total_edge_captured,
                'avg_edge_per_signal': self._total_edge_captured / max(1, self._signals_executed),
                'active_signals': len(self._active_signals),
                'venues_tracked': len(self.venue_names)
            }
    
    def set_tick_sizes(self, tick_sizes: Dict[str, float]):
        """Set tick sizes for all venues."""
        for venue, tick in tick_sizes.items():
            self._lag_tracker.set_tick_size(venue, tick)


# Module singleton instance
_lat_module: Optional[LatencyArbitrageModule] = None


def get_latency_module(
    venues: List[str],
    min_edge_bps: float = 2.0
) -> LatencyArbitrageModule:
    """Get or create the global latency arbitrage module instance."""
    global _lat_module
    if _lat_module is None:
        _lat_module = LatencyArbitrageModule(venues, min_edge_bps=min_edge_bps)
    return _lat_module


def reset_latency_module():
    """Reset the global latency module (for testing)."""
    global _lat_module
    _lat_module = None


# Convenience functions
def quick_arb_check(
    venue_a_price: float,
    venue_b_price: float,
    venue_a_time_ns: int,
    venue_b_time_ns: int,
    tick_size: float = 0.01
) -> Dict[str, Any]:
    """
    Quick cross-venue arbitrage check.
    
    Returns:
        Dictionary with arb analysis
    """
    price_diff = venue_a_price - venue_b_price
    avg_price = (venue_a_price + venue_b_price) / 2
    
    time_diff_ms = (venue_a_time_ns - venue_b_time_ns) / 1_000_000
    
    edge_bps = abs(price_diff) / avg_price * 10000 if avg_price > 0 else 0
    
    # Estimate latency using simple method
    latency_ms = estimate_latency_ms(
        np.array([venue_a_time_ns]),
        np.array([venue_a_price]),
        np.array([venue_b_time_ns]),
        np.array([venue_b_price]),
        tick_size
    )
    
    return {
        'price_diff': price_diff,
        'edge_bps': edge_bps,
        'time_diff_ms': time_diff_ms,
        'estimated_latency_ms': latency_ms,
        'is_arb_opportunity': edge_bps > 2.0 and abs(time_diff_ms) > 1.0,
        'direction': 'buy_b_sell_a' if price_diff > 0 else 'buy_a_sell_b'
    }
