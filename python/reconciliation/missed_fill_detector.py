"""
Missed Fill Detector - Heuristic engine to detect and reconstruct missed WebSocket execution reports.
Uses order state logic and sequence gap analysis to identify fills that weren't reported.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import numpy as np
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    """Order status states."""
    NEW = "new"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class OrderState:
    """Current state of an order."""
    order_id: str
    client_order_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    order_type: str
    quantity: float
    price: Optional[float]
    
    # Tracking fields
    filled_qty: float = 0.0
    remaining_qty: float = 0.0
    avg_fill_price: float = 0.0
    status: OrderStatus = OrderStatus.NEW
    
    # Sequence tracking
    last_sequence: int = 0
    last_update_time: float = 0.0
    
    # Fill history
    fills: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        if self.remaining_qty == 0:
            self.remaining_qty = self.quantity
    
    def apply_fill(self, fill_qty: float, fill_price: float, 
                   sequence: int, timestamp: float):
        """Apply a fill to the order."""
        old_filled = self.filled_qty
        
        self.filled_qty += fill_qty
        self.remaining_qty -= fill_qty
        
        # Update average fill price
        total_value = self.avg_fill_price * old_filled + fill_price * fill_qty
        self.avg_fill_price = total_value / self.filled_qty if self.filled_qty > 0 else 0
        
        # Update status
        if self.remaining_qty <= 0:
            self.status = OrderStatus.FILLED
        elif self.filled_qty > 0:
            self.status = OrderStatus.PARTIALLY_FILLED
        
        # Record fill
        self.fills.append({
            'quantity': fill_qty,
            'price': fill_price,
            'sequence': sequence,
            'timestamp': timestamp
        })
        
        self.last_sequence = sequence
        self.last_update_time = timestamp
    
    def is_terminal(self) -> bool:
        """Check if order is in terminal state."""
        return self.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, 
                               OrderStatus.REJECTED, OrderStatus.EXPIRED]


@dataclass
class MissedFill:
    """Detected missed fill event."""
    detection_time: float
    order_id: str
    symbol: str
    side: str
    estimated_qty: float
    estimated_price: float
    estimated_time: float
    confidence: float
    detection_method: str
    reconstruction_source: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "detection_time": self.detection_time,
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side,
            "estimated_qty": self.estimated_qty,
            "estimated_price": self.estimated_price,
            "estimated_time": self.estimated_time,
            "confidence": self.confidence,
            "detection_method": self.detection_method,
            "reconstruction_source": self.reconstruction_source
        }


class MissedFillDetector:
    """
    Heuristic engine for detecting missed WebSocket execution reports.
    Uses multiple detection methods:
    1. Sequence gap analysis
    2. Order state inconsistency detection
    3. Exchange REST reconciliation
    4. Trade flow correlation
    """
    
    def __init__(self,
                 sequence_timeout: float = 5.0,
                 max_pending_time: float = 60.0,
                 price_tolerance_pct: float = 0.001):
        """
        Initialize detector.
        
        Args:
            sequence_timeout: Seconds before sequence gap triggers alert
            max_pending_time: Maximum time order can be pending without update
            price_tolerance_pct: Tolerance for price matching
        """
        self.sequence_timeout = sequence_timeout
        self.max_pending_time = max_pending_time
        self.price_tolerance_pct = price_tolerance_pct
        
        # Active orders being tracked
        self._orders: Dict[str, OrderState] = {}
        
        # Sequence tracking per symbol
        self._expected_sequences: Dict[str, int] = {}
        self._sequence_history: Dict[str, deque] = {}
        
        # Detected missed fills
        self._missed_fills: deque = deque(maxlen=1000)
        
        # Detection statistics
        self._total_detected = 0
        self._false_positives = 0
    
    def track_order(self, order: OrderState):
        """Start tracking an order."""
        self._orders[order.order_id] = order
        logger.debug(f"Tracking order {order.order_id}")
    
    def process_execution_report(self, order_id: str, fill_qty: float, 
                                  fill_price: float, sequence: int,
                                  timestamp: float, symbol: str):
        """
        Process an execution report from WebSocket.
        
        Args:
            order_id: Order ID
            fill_qty: Fill quantity
            fill_price: Fill price
            sequence: Message sequence number
            timestamp: Report timestamp
            symbol: Trading symbol
        """
        # Check for sequence gaps
        self._check_sequence(symbol, sequence, timestamp)
        
        # Update order state
        if order_id in self._orders:
            order = self._orders[order_id]
            order.apply_fill(fill_qty, fill_price, sequence, timestamp)
            
            if order.is_terminal():
                logger.info(f"Order {order_id} completed: {order.filled_qty}/{order.quantity}")
        else:
            logger.warning(f"Execution report for unknown order {order_id}")
    
    def _check_sequence(self, symbol: str, sequence: int, timestamp: float):
        """Check for sequence gaps."""
        if symbol not in self._expected_sequences:
            self._expected_sequences[symbol] = sequence
            self._sequence_history[symbol] = deque(maxlen=100)
            return
        
        expected = self._expected_sequences[symbol]
        
        if sequence > expected:
            # Gap detected!
            gap_size = sequence - expected
            logger.warning(
                f"Sequence gap on {symbol}: expected {expected}, got {sequence} "
                f"(gap={gap_size})"
            )
            
            # Record potential missed messages
            self._record_sequence_gap(symbol, expected, sequence, timestamp)
        
        self._expected_sequences[symbol] = sequence + 1
        self._sequence_history[symbol].append((sequence, timestamp))
    
    def _record_sequence_gap(self, symbol: str, start_seq: int, 
                             end_seq: int, timestamp: float):
        """Record a sequence gap for analysis."""
        # Would attempt to correlate with order state changes
        pass
    
    def detect_stale_orders(self, current_time: float) -> List[MissedFill]:
        """
        Detect orders that appear stale (no updates for too long).
        
        Args:
            current_time: Current timestamp
            
        Returns:
            List of potential missed fills
        """
        missed = []
        
        for order_id, order in self._orders.items():
            if order.is_terminal():
                continue
            
            time_since_update = current_time - order.last_update_time
            
            # Check if order should have been filled by now
            if order.status == OrderStatus.NEW and time_since_update > self.sequence_timeout:
                # Might have been filled without report
                missed_fill = self._reconstruct_missed_fill(order, current_time, "stale_detection")
                if missed_fill:
                    missed.append(missed_fill)
            
            # Partially filled but no recent updates
            elif order.status == OrderStatus.PARTIALLY_FILLED:
                if time_since_update > self.max_pending_time:
                    # Check if remaining qty might have been filled
                    pass
        
        return missed
    
    def reconcile_with_rest(self, rest_snapshots: Dict[str, Dict[str, Any]]) -> List[MissedFill]:
        """
        Reconcile internal state with exchange REST API snapshots.
        
        Args:
            rest_snapshots: Dict mapping order_id -> REST API state
            
        Returns:
            List of detected missed fills
        """
        missed = []
        
        for order_id, rest_state in rest_snapshots.items():
            if order_id not in self._orders:
                logger.warning(f"Order {order_id} exists on exchange but not tracked locally")
                continue
            
            local_order = self._orders[order_id]
            
            # Compare filled quantities
            rest_filled = rest_state.get('filled_qty', 0)
            
            if rest_filled > local_order.filled_qty:
                # Exchange shows more filled than we know about
                missing_qty = rest_filled - local_order.filled_qty
                
                # Estimate fill price from REST data
                rest_avg_price = rest_state.get('avg_fill_price', local_order.avg_fill_price)
                
                missed_fill = MissedFill(
                    detection_time=time.time(),
                    order_id=order_id,
                    symbol=local_order.symbol,
                    side=local_order.side,
                    estimated_qty=missing_qty,
                    estimated_price=rest_avg_price,
                    estimated_time=rest_state.get('last_fill_time', time.time()),
                    confidence=0.9,  # High confidence from REST confirmation
                    detection_method="rest_reconciliation",
                    reconstruction_source="exchange_rest"
                )
                
                missed.append(missed_fill)
                self._missed_fills.append(missed_fill)
                self._total_detected += 1
                
                # Update local state
                local_order.apply_fill(
                    missing_qty, rest_avg_price,
                    rest_state.get('sequence', 0),
                    rest_state.get('last_fill_time', time.time())
                )
                
                logger.info(
                    f"Detected missed fill via REST: {order_id} "
                    f"qty={missing_qty}, price={rest_avg_price}"
                )
        
        return missed
    
    def _reconstruct_missed_fill(self, order: OrderState, 
                                  current_time: float, method: str) -> Optional[MissedFill]:
        """Attempt to reconstruct a missed fill."""
        # Use heuristics based on order book state at the time
        # This is simplified - would need actual market data
        
        if order.status == OrderStatus.NEW:
            # Order was new, might have been immediately filled
            confidence = 0.5  # Low confidence without confirmation
            
            return MissedFill(
                detection_time=current_time,
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                estimated_qty=order.quantity,  # Assume full fill
                estimated_price=order.price if order.price else 0,
                estimated_time=order.last_update_time + self.sequence_timeout / 2,
                confidence=confidence,
                detection_method=method,
                reconstruction_source="heuristic"
            )
        
        return None
    
    def correlate_with_trades(self, trade_flow: List[Dict[str, Any]]) -> List[MissedFill]:
        """
        Correlate with observed trade flow to detect unreported fills.
        
        Args:
            trade_flow: List of trades observed on the tape
            
        Returns:
            Potential missed fills correlated with tape trades
        """
        missed = []
        
        for order_id, order in self._orders.items():
            if order.is_terminal() or order.price is None:
                continue
            
            # Look for trades at our price level
            for trade in trade_flow:
                if trade['symbol'] != order.symbol:
                    continue
                
                # Check if trade could be ours
                price_match = abs(trade['price'] - order.price) / order.price < self.price_tolerance_pct
                
                if price_match:
                    # Could be our order - check timing
                    time_diff = abs(trade['timestamp'] - order.last_update_time)
                    
                    if time_diff < self.sequence_timeout:
                        # Suspicious - might be our unreported fill
                        pass
        
        return missed
    
    def get_missed_fills(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent missed fills."""
        return [mf.to_dict() for mf in list(self._missed_fills)[-limit:]]
    
    def get_detector_stats(self) -> Dict[str, Any]:
        """Get detection statistics."""
        active_orders = sum(1 for o in self._orders.values() if not o.is_terminal())
        
        return {
            "active_orders_tracked": active_orders,
            "total_orders_seen": len(self._orders),
            "missed_fills_detected": self._total_detected,
            "false_positives": self._false_positives,
            "symbols_tracked": len(self._expected_sequences)
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Return detector health status."""
        return {
            "tracking_orders": len(self._orders),
            "missed_fills_cached": len(self._missed_fills),
            "sequences_tracked": len(self._expected_sequences)
        }


# Module singleton
_detector: Optional[MissedFillDetector] = None


def get_missed_fill_detector(**kwargs) -> MissedFillDetector:
    """Get or create the global missed fill detector."""
    global _detector
    
    if _detector is None:
        _detector = MissedFillDetector(**kwargs)
        logger.info("Created missed fill detector")
    
    return _detector


if __name__ == "__main__":
    # Test the missed fill detector
    print("Testing Missed Fill Detector...")
    
    detector = MissedFillDetector(sequence_timeout=2.0)
    
    # Create and track an order
    order = OrderState(
        order_id="ord_123",
        client_order_id="client_123",
        symbol="BTC/USD",
        side="buy",
        order_type="limit",
        quantity=10.0,
        price=50000.0
    )
    
    detector.track_order(order)
    
    # Simulate sequence progression
    base_time = time.time()
    
    # Normal execution report
    detector.process_execution_report(
        order_id="ord_123",
        fill_qty=5.0,
        fill_price=50000.0,
        sequence=100,
        timestamp=base_time,
        symbol="BTC/USD"
    )
    
    print(f"After normal fill: {detector.get_detector_stats()}")
    
    # Simulate sequence gap (missed message)
    detector.process_execution_report(
        order_id="ord_123",
        fill_qty=5.0,
        fill_price=50000.0,
        sequence=105,  # Jumped from 100 to 105!
        timestamp=base_time + 0.1,
        symbol="BTC/USD"
    )
    
    # Simulate REST reconciliation finding discrepancy
    rest_snapshot = {
        "ord_123": {
            "filled_qty": 12.0,  # More than we know about
            "avg_fill_price": 50000.0,
            "sequence": 105,
            "last_fill_time": base_time + 0.05
        }
    }
    
    missed = detector.reconcile_with_rest(rest_snapshot)
    print(f"\nREST reconciliation found {len(missed)} missed fills:")
    for mf in missed:
        print(f"  Order {mf.order_id}: qty={mf.estimated_qty}, price={mf.estimated_price}")
        print(f"    Confidence: {mf.confidence}, Method: {mf.detection_method}")
    
    print(f"\nStats: {detector.get_detector_stats()}")
    print(f"Health: {detector.health_check()}")
