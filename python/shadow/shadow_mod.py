"""
Shadow Trading Module Root
Manages the shadow engine, strictly blocking REST order submissions while 
consuming live market data ring buffers.

Provides unified interface for paper trading and variance analysis.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
import threading
import time
import json

from .paper_executor import (
    PaperExecutor, OrderSide, OrderType, OrderStatus,
    PaperOrder, FillReport, OrderBookSimulator
)
from .variance_analyzer import (
    ShadowVarianceAnalyzer, ModelQuarantineManager,
    PnLSnapshot, VarianceStatus, VarianceReport
)


@dataclass
class ShadowConfig:
    """Configuration for shadow trading engine."""
    # Executor settings
    commission_rate: float = 0.0005
    slippage_model: str = "volume_weighted"
    
    # Variance analyzer settings
    warning_threshold_pct: float = 1.0
    critical_threshold_pct: float = 3.0
    quarantine_threshold_pct: float = 5.0
    
    # Ring buffer settings
    max_book_depth: int = 100
    max_tick_history: int = 10000
    
    # Reporting settings
    snapshot_interval_sec: float = 1.0
    report_interval_sec: float = 10.0
    
    # Safety settings
    block_live_orders: bool = True
    auto_quarantine: bool = True


class LiveMarketDataRingBuffer:
    """
    Lock-free ring buffer for live market data.
    Shares read-only buffers to avoid memory duplication.
    """
    
    def __init__(self, capacity: int = 10000):
        """Initialize ring buffer."""
        self.capacity = capacity
        
        # Pre-allocate arrays
        self._timestamps = np.zeros(capacity, dtype=np.float64)
        self._bids_price = np.zeros((capacity, 10), dtype=np.float64)
        self._bids_volume = np.zeros((capacity, 10), dtype=np.float64)
        self._asks_price = np.zeros((capacity, 10), dtype=np.float64)
        self._asks_volume = np.zeros((capacity, 10), dtype=np.float64)
        self._last_prices = np.zeros(capacity, dtype=np.float64)
        
        # Write index (atomic-ish for single writer)
        self._write_idx = 0
        
        # Read index for consumers
        self._read_idx = 0
        
        # Lock for thread safety
        self._lock = threading.Lock()
    
    def write_tick(self,
                   timestamp: float,
                   bids: List[Tuple[float, float]],
                   asks: List[Tuple[float, float]],
                   last_price: float):
        """Write a tick to the ring buffer."""
        with self._lock:
            idx = self._write_idx % self.capacity
            
            self._timestamps[idx] = timestamp
            self._last_prices[idx] = last_price
            
            # Write book levels
            for i, (price, vol) in enumerate(bids[:10]):
                self._bids_price[idx, i] = price
                self._bids_volume[idx, i] = vol
            
            for i, (price, vol) in enumerate(asks[:10]):
                self._asks_price[idx, i] = price
                self._asks_volume[idx, i] = vol
            
            self._write_idx += 1
    
    def read_latest(self) -> Dict[str, Any]:
        """Read latest tick data."""
        with self._lock:
            if self._write_idx == 0:
                return {}
            
            idx = (self._write_idx - 1) % self.capacity
            
            bids = [
                (self._bids_price[idx, i], self._bids_volume[idx, i])
                for i in range(10) if self._bids_price[idx, i] > 0
            ]
            asks = [
                (self._asks_price[idx, i], self._asks_volume[idx, i])
                for i in range(10) if self._asks_price[idx, i] > 0
            ]
            
            return {
                'timestamp': self._timestamps[idx],
                'bids': bids,
                'asks': asks,
                'last_price': self._last_prices[idx]
            }
    
    def get_write_index(self) -> int:
        """Get current write index."""
        return self._write_idx


class ShadowTradingEngine:
    """
    Main shadow trading engine.
    Coordinates paper execution and variance analysis.
    """
    
    def __init__(self, config: Optional[ShadowConfig] = None):
        self.config = config or ShadowConfig()
        
        # Initialize components
        self.executor = PaperExecutor(
            commission_rate=self.config.commission_rate,
            slippage_model=self.config.slippage_model
        )
        
        self.analyzer = ShadowVarianceAnalyzer(
            warning_threshold_pct=self.config.warning_threshold_pct,
            critical_threshold_pct=self.config.critical_threshold_pct,
            quarantine_threshold_pct=self.config.quarantine_threshold_pct
        )
        
        self.quarantine_manager = ModelQuarantineManager(self.analyzer)
        
        # Ring buffer for market data
        self.market_buffer = LiveMarketDataRingBuffer(
            capacity=self.config.max_tick_history
        )
        
        # State tracking
        self._running = False
        self._shadow_pnl = 0.0
        self._live_pnl = 0.0  # Would be fed from live system
        
        # Snapshot scheduling
        self._last_snapshot_time = 0.0
        self._snapshot_count = 0
        
        # Callbacks
        self._variance_callbacks: List[Callable[[VarianceReport], None]] = []
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Statistics
        self._stats = {
            'ticks_processed': 0,
            'orders_submitted': 0,
            'snapshots_taken': 0,
            'quarantines_triggered': 0
        }
    
    def start(self):
        """Start shadow engine."""
        self._running = True
    
    def stop(self):
        """Stop shadow engine."""
        self._running = False
    
    def on_market_data(self,
                       timestamp: float,
                       bids: List[Tuple[float, float]],
                       asks: List[Tuple[float, float]],
                       last_price: float):
        """
        Handle incoming market data.
        
        Args:
            timestamp: Data timestamp
            bids: Bid levels
            asks: Ask levels
            last_price: Last traded price
        """
        if not self._running:
            return
        
        # Write to ring buffer (shared with other consumers)
        self.market_buffer.write_tick(timestamp, bids, asks, last_price)
        
        # Update executor's order book
        self.executor.order_book.update_book(bids, asks)
        self.executor.order_book.update_last_price(last_price)
        
        self._stats['ticks_processed'] += 1
        
        # Take periodic snapshots
        if timestamp - self._last_snapshot_time >= self.config.snapshot_interval_sec:
            self._take_snapshot(last_price)
            self._last_snapshot_time = timestamp
    
    def _take_snapshot(self, current_price: float):
        """Take PnL snapshot for variance analysis."""
        with self._lock:
            shadow_pos, shadow_avg = self.executor.get_position()
            
            # Calculate shadow PnL
            if shadow_pos != 0:
                self._shadow_pnl = (current_price - shadow_avg) * shadow_pos
            
            # Get live PnL (would come from production system)
            # For now, simulate with small random difference
            self._live_pnl = self._shadow_pnl * (1 + np.random.normal(0, 0.001))
            
            # Get fee and slippage stats
            exec_stats = self.executor.get_statistics()
            
            snapshot = PnLSnapshot(
                timestamp=time.time(),
                shadow_pnl=self._shadow_pnl,
                live_pnl=self._live_pnl,
                shadow_position=shadow_pos,
                live_position=shadow_pos * (1 + np.random.normal(0, 0.0001)),
                shadow_fees=exec_stats['total_commission'],
                live_fees=exec_stats['total_commission'] * (1 + np.random.normal(0, 0.01)),
                shadow_slippage=exec_stats['avg_slippage_bps'],
                live_slippage=exec_stats['avg_slippage_bps'] * (1 + np.random.normal(0, 0.05))
            )
            
            self.analyzer.record_snapshot(snapshot)
            self._snapshot_count += 1
            self._stats['snapshots_taken'] = self._snapshot_count
    
    def submit_order(self,
                     side: str,
                     order_type: str,
                     price: float,
                     quantity: float) -> Optional[PaperOrder]:
        """
        Submit a shadow order.
        
        Blocks if quarantine is active.
        
        Args:
            side: 'buy' or 'sell'
            order_type: 'market' or 'limit'
            price: Order price
            quantity: Order quantity
            
        Returns:
            PaperOrder if submitted, None if blocked
        """
        # Check quarantine
        if self.config.auto_quarantine:
            quarantine_reason = self.quarantine_manager.check_and_quarantine()
            if quarantine_reason:
                self._stats['quarantines_triggered'] += 1
                return None
        
        # Convert enums
        side_enum = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL
        type_enum = OrderType.MARKET if order_type.lower() == 'market' else OrderType.LIMIT
        
        # Submit to paper executor
        order = self.executor.submit_order(side_enum, type_enum, price, quantity)
        self._stats['orders_submitted'] += 1
        
        return order
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a shadow order."""
        return self.executor.cancel_order(order_id)
    
    def get_variance_report(self) -> VarianceReport:
        """Get current variance analysis report."""
        return self.analyzer.analyze()
    
    def register_variance_callback(self, callback: Callable[[VarianceReport], None]):
        """Register callback for variance alerts."""
        self._variance_callbacks.append(callback)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics."""
        with self._lock:
            return {
                **self._stats,
                'shadow_pnl': self._shadow_pnl,
                'live_pnl': self._live_pnl,
                'pnl_difference': self._shadow_pnl - self._live_pnl,
                'executor_stats': self.executor.get_statistics(),
                'analyzer_stats': self.analyzer.get_statistics(),
                'quarantine_stats': self.quarantine_manager.get_quarantine_stats()
            }
    
    def get_position(self) -> Tuple[float, float]:
        """Get current shadow position."""
        return self.executor.get_position()
    
    def is_quarantined(self) -> bool:
        """Check if model is quarantined."""
        return self.quarantine_manager._quarantine_start_time is not None
    
    def release_quarantine(self) -> bool:
        """Manually release quarantine."""
        return self.quarantine_manager.release_quarantine()


class ShadowRESTBlocker:
    """
    Intercepts and blocks REST order submissions.
    Ensures shadow mode cannot accidentally send live orders.
    """
    
    def __init__(self, shadow_engine: ShadowTradingEngine):
        """Initialize REST blocker."""
        self.shadow_engine = shadow_engine
        self._blocked_count = 0
        self._allowed_readonly = True
    
    def intercept_request(self, method: str, path: str, body: Dict) -> Tuple[bool, Any]:
        """
        Intercept REST request.
        
        Args:
            method: HTTP method
            path: Request path
            body: Request body
            
        Returns:
            Tuple of (allowed, response_or_error)
        """
        # Block order submissions
        order_paths = ['/orders', '/order', '/execute', '/trade']
        
        if method.upper() in ['POST', 'PUT', 'DELETE']:
            if any(p in path.lower() for p in order_paths):
                if self.shadow_engine.is_quarantined():
                    self._blocked_count += 1
                    return False, {
                        'error': 'QUARANTINE_ACTIVE',
                        'message': 'Orders blocked due to model quarantine'
                    }
                
                if self.shadow_engine.config.block_live_orders:
                    self._blocked_count += 1
                    return False, {
                        'error': 'SHADOW_MODE',
                        'message': 'Live orders blocked in shadow mode'
                    }
        
        # Allow read-only requests
        if method.upper() == 'GET' and self._allowed_readonly:
            return True, None
        
        return True, None
    
    def get_blocked_count(self) -> int:
        """Get count of blocked requests."""
        return self._blocked_count


# Module-level singleton
_engine: Optional[ShadowTradingEngine] = None
_rest_blocker: Optional[ShadowRESTBlocker] = None
_lock = threading.Lock()


def get_shadow_engine(config: Optional[ShadowConfig] = None) -> ShadowTradingEngine:
    """Get or create global shadow engine."""
    global _engine
    
    with _lock:
        if _engine is None:
            _engine = ShadowTradingEngine(config)
        return _engine


def get_rest_blocker() -> ShadowRESTBlocker:
    """Get or create REST blocker."""
    global _rest_blocker, _engine
    
    with _lock:
        if _engine is None:
            _engine = ShadowTradingEngine()
        if _rest_blocker is None:
            _rest_blocker = ShadowRESTBlocker(_engine)
        return _rest_blocker


def start_shadow_trading(config: Optional[ShadowConfig] = None):
    """Start shadow trading."""
    engine = get_shadow_engine(config)
    engine.start()


def stop_shadow_trading():
    """Stop shadow trading."""
    global _engine
    if _engine is not None:
        _engine.stop()


def submit_shadow_order(side: str, order_type: str, price: float, quantity: float) -> Optional[PaperOrder]:
    """Submit order through shadow engine."""
    engine = get_shadow_engine()
    return engine.submit_order(side, order_type, price, quantity)


def get_shadow_status() -> Dict[str, Any]:
    """Get shadow trading status."""
    engine = get_shadow_engine()
    return {
        'running': engine._running,
        'quarantined': engine.is_quarantined(),
        'statistics': engine.get_statistics(),
        'latest_variance': engine.get_variance_report().__dict__ if engine._snapshot_count > 0 else None
    }


# Module exports
__all__ = [
    'ShadowConfig',
    'LiveMarketDataRingBuffer',
    'ShadowTradingEngine',
    'ShadowRESTBlocker',
    'get_shadow_engine',
    'get_rest_blocker',
    'start_shadow_trading',
    'stop_shadow_trading',
    'submit_shadow_order',
    'get_shadow_status'
]
