"""
Paper Trading Executor for Shadow Engine
Simulates order matching against the live L2 book, tracking theoretical fills, 
slippage, and queue position without broadcasting to the exchange.

Allows the bot to test new ML weights and RL policies in real-time market 
conditions without risking actual capital.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import time
import threading


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class PaperOrder:
    """Represents a paper trading order."""
    order_id: str
    side: OrderSide
    order_type: OrderType
    price: float
    quantity: float
    timestamp: float
    
    # Execution tracking
    filled_quantity: float = 0.0
    average_fill_price: float = 0.0
    status: OrderStatus = OrderStatus.PENDING
    
    # Queue simulation
    queue_position: int = 0
    estimated_slippage: float = 0.0
    
    # Fill history
    fills: List[Dict] = field(default_factory=list)


@dataclass
class FillReport:
    """Represents a fill execution report."""
    order_id: str
    fill_id: str
    side: OrderSide
    price: float
    quantity: float
    timestamp: float
    commission: float
    slippage: float


class OrderBookSimulator:
    """
    Simulates L2 order book for paper trading.
    Tracks queue positions and estimates fills.
    """
    
    def __init__(self, max_depth: int = 100):
        """
        Initialize order book simulator.
        
        Args:
            max_depth: Maximum depth levels to track
        """
        self.max_depth = max_depth
        
        # Order book levels
        self._bids: List[Tuple[float, float]] = []  # (price, volume)
        self._asks: List[Tuple[float, float]] = []
        
        # Last trade price
        self._last_price = 0.0
        
        # Spread tracking
        self._spread_bps = 0.0
        
        # Lock for thread safety
        self._lock = threading.Lock()
    
    def update_book(self, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]]):
        """
        Update order book from live data.
        
        Args:
            bids: Bid levels (price, volume)
            asks: Ask levels (price, volume)
        """
        with self._lock:
            # Sort and limit depth
            self._bids = sorted(bids, key=lambda x: -x[0])[:self.max_depth]
            self._asks = sorted(asks, key=lambda x: x[0])[:self.max_depth]
            
            # Update spread
            if self._bids and self._asks:
                mid = (self._bids[0][0] + self._asks[0][0]) / 2
                self._spread_bps = (self._asks[0][0] - self._bids[0][0]) / mid * 10000
    
    def update_last_price(self, price: float):
        """Update last traded price."""
        with self._lock:
            self._last_price = price
    
    def get_best_bid(self) -> Optional[float]:
        """Get best bid price."""
        with self._lock:
            return self._bids[0][0] if self._bids else None
    
    def get_best_ask(self) -> Optional[float]:
        """Get best ask price."""
        with self._lock:
            return self._asks[0][0] if self._asks else None
    
    def get_mid_price(self) -> float:
        """Get mid price."""
        with self._lock:
            if not self._bids or not self._asks:
                return self._last_price
            return (self._bids[0][0] + self._asks[0][0]) / 2
    
    def estimate_fill(self, 
                      side: OrderSide,
                      order_type: OrderType,
                      limit_price: float,
                      quantity: float) -> Tuple[float, float, int]:
        """
        Estimate fill for an order.
        
        Args:
            side: Order side
            order_type: Order type
            limit_price: Limit price (or None for market)
            quantity: Order quantity
            
        Returns:
            Tuple of (expected_fill_price, expected_slippage_bps, queue_position)
        """
        with self._lock:
            if order_type == OrderType.MARKET:
                return self._estimate_market_fill(side, quantity)
            else:
                return self._estimate_limit_fill(side, limit_price, quantity)
    
    def _estimate_market_fill(self, 
                               side: OrderSide,
                               quantity: float) -> Tuple[float, float, int]:
        """Estimate market order fill."""
        if side == OrderSide.BUY:
            # Buying hits asks
            book = self._asks
            sign = 1
        else:
            # Selling hits bids
            book = self._bids
            sign = -1
        
        if not book:
            return self._last_price, 0.0, 0
        
        remaining = quantity
        total_value = 0.0
        filled = 0.0
        
        for price, volume in book:
            fill_qty = min(remaining, volume)
            total_value += fill_qty * price
            filled += fill_qty
            remaining -= fill_qty
            
            if remaining <= 0:
                break
        
        if filled == 0:
            return self._last_price, 0.0, 0
        
        avg_price = total_value / filled
        
        # Calculate slippage vs mid
        mid = self.get_mid_price()
        if mid > 0:
            slippage_bps = abs(avg_price - mid) / mid * 10000 * sign
        else:
            slippage_bps = 0.0
        
        return avg_price, slippage_bps, 0
    
    def _estimate_limit_fill(self,
                              side: OrderSide,
                              limit_price: float,
                              quantity: float) -> Tuple[float, float, int]:
        """Estimate limit order fill probability and queue position."""
        best_bid = self._bids[0][0] if self._bids else 0
        best_ask = self._asks[0][0] if self._asks else float('inf')
        
        if side == OrderSide.BUY:
            if limit_price >= best_ask:
                # Would execute immediately
                return self._estimate_market_fill(side, quantity)
            elif limit_price < best_bid:
                # Behind current best bid
                # Estimate queue position based on price distance
                price_distance = (best_bid - limit_price) / best_bid * 10000
                queue_position = int(price_distance * 10)  # Rough estimate
                return limit_price, 0.0, queue_position
            else:
                # At or near best bid
                queue_position = int(quantity * 0.5)  # Partial priority
                return limit_price, 0.0, queue_position
        else:  # SELL
            if limit_price <= best_bid:
                # Would execute immediately
                return self._estimate_market_fill(side, quantity)
            elif limit_price > best_ask:
                # Behind current best ask
                price_distance = (limit_price - best_ask) / best_ask * 10000
                queue_position = int(price_distance * 10)
                return limit_price, 0.0, queue_position
            else:
                queue_position = int(quantity * 0.5)
                return limit_price, 0.0, queue_position
    
    def get_spread_bps(self) -> float:
        """Get current spread in basis points."""
        with self._lock:
            return self._spread_bps


class PaperExecutor:
    """
    Main paper trading executor.
    Simulates order lifecycle without real execution.
    """
    
    def __init__(self, 
                 commission_rate: float = 0.0005,  # 5 bps
                 slippage_model: str = "volume_weighted"):
        """
        Initialize paper executor.
        
        Args:
            commission_rate: Commission rate per fill
            slippage_model: Slippage estimation model
        """
        self.commission_rate = commission_rate
        self.slippage_model = slippage_model
        
        self.order_book = OrderBookSimulator()
        
        # Active orders
        self._orders: Dict[str, PaperOrder] = {}
        self._order_counter = 0
        
        # Fill history
        self._fills: deque = deque(maxlen=10000)
        
        # Statistics
        self._total_filled = 0.0
        self._total_commission = 0.0
        self._total_slippage = 0.0
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Position tracking
        self._position = 0.0
        self._avg_entry = 0.0
    
    def submit_order(self,
                     side: OrderSide,
                     order_type: OrderType,
                     price: float,
                     quantity: float) -> PaperOrder:
        """
        Submit a paper order.
        
        Args:
            side: Buy or sell
            order_type: Market or limit
            price: Price (limit price for limit orders)
            quantity: Order quantity
            
        Returns:
            PaperOrder object
        """
        with self._lock:
            self._order_counter += 1
            order_id = f"PAPER_{self._order_counter}_{int(time.time())}"
            
            order = PaperOrder(
                order_id=order_id,
                side=side,
                order_type=order_type,
                price=price,
                quantity=quantity,
                timestamp=time.time()
            )
            
            # Try immediate fill for market orders
            if order_type == OrderType.MARKET:
                self._try_market_fill(order)
            else:
                # Check if limit order would cross
                self._check_limit_cross(order)
            
            self._orders[order_id] = order
            return order
    
    def _try_market_fill(self, order: PaperOrder):
        """Try to fill a market order immediately."""
        est_price, slippage_bps, queue_pos = self.order_book.estimate_fill(
            order.side, order.order_type, order.price, order.quantity
        )
        
        # Apply slippage model
        if self.slippage_model == "conservative":
            slippage_bps *= 1.5
        
        # Calculate fill
        fill_price = est_price * (1 + slippage_bps / 10000 * (1 if order.side == OrderSide.BUY else -1))
        fill_qty = order.quantity
        commission = fill_price * fill_qty * self.commission_rate
        
        # Create fill report
        fill = FillReport(
            order_id=order.order_id,
            fill_id=f"FILL_{order.order_id}",
            side=order.side,
            price=fill_price,
            quantity=fill_qty,
            timestamp=time.time(),
            commission=commission,
            slippage=abs(slippage_bps)
        )
        
        # Update order
        order.filled_quantity = fill_qty
        order.average_fill_price = fill_price
        order.status = OrderStatus.FILLED
        order.estimated_slippage = slippage_bps
        order.fills.append({
            'price': fill_price,
            'quantity': fill_qty,
            'commission': commission,
            'timestamp': fill.timestamp
        })
        
        # Update statistics
        self._record_fill(fill)
        self._update_position(fill)
    
    def _check_limit_cross(self, order: PaperOrder):
        """Check if limit order crosses the spread."""
        best_bid = self.order_book.get_best_bid()
        best_ask = self.order_book.get_best_ask()
        
        crossed = False
        if order.side == OrderSide.BUY and best_ask and order.price >= best_ask:
            crossed = True
        elif order.side == OrderSide.SELL and best_bid and order.price <= best_bid:
            crossed = True
        
        if crossed:
            # Execute at better of limit or market price
            self._try_market_fill(order)
        else:
            # Set queue position
            _, _, queue_pos = self.order_book.estimate_fill(
                order.side, order.order_type, order.price, order.quantity
            )
            order.queue_position = queue_pos
            order.status = OrderStatus.PENDING
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        with self._lock:
            if order_id not in self._orders:
                return False
            
            order = self._orders[order_id]
            if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
                return False
            
            order.status = OrderStatus.CANCELLED
            return True
    
    def _record_fill(self, fill: FillReport):
        """Record a fill."""
        self._fills.append(fill)
        self._total_filled += fill.price * fill.quantity
        self._total_commission += fill.commission
        self._total_slippage += fill.slippage
    
    def _update_position(self, fill: FillReport):
        """Update position from fill."""
        if fill.side == OrderSide.BUY:
            if self._position >= 0:
                # Adding to long or opening long
                total_value = self._position * self._avg_entry + fill.quantity * fill.price
                self._position += fill.quantity
                if self._position > 0:
                    self._avg_entry = total_value / self._position
            else:
                # Covering short
                cover_qty = min(abs(self._position), fill.quantity)
                remaining = fill.quantity - cover_qty
                self._position += fill.quantity
                
                if self._position > 0:
                    # Flipped to long
                    self._avg_entry = fill.price
        else:  # SELL
            if self._position <= 0:
                # Adding to short or opening short
                total_value = abs(self._position) * self._avg_entry + fill.quantity * fill.price
                self._position -= fill.quantity
                if self._position < 0:
                    self._avg_entry = total_value / abs(self._position)
            else:
                # Reducing long
                reduce_qty = min(self._position, fill.quantity)
                remaining = fill.quantity - reduce_qty
                self._position -= fill.quantity
                
                if self._position < 0:
                    # Flipped to short
                    self._avg_entry = fill.price
    
    def get_position(self) -> Tuple[float, float]:
        """Get current position and average entry."""
        with self._lock:
            return self._position, self._avg_entry
    
    def get_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized PnL."""
        with self._lock:
            if self._position == 0:
                return 0.0
            return (current_price - self._avg_entry) * self._position
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get execution statistics."""
        with self._lock:
            return {
                'total_orders': len(self._orders),
                'active_orders': sum(1 for o in self._orders.values() 
                                    if o.status == OrderStatus.PENDING),
                'filled_orders': sum(1 for o in self._orders.values() 
                                    if o.status == OrderStatus.FILLED),
                'total_filled_notional': self._total_filled,
                'total_commission': self._total_commission,
                'avg_slippage_bps': self._total_slippage / len(self._fills) if self._fills else 0,
                'current_position': self._position,
                'avg_entry': self._avg_entry
            }
    
    def get_recent_fills(self, count: int = 10) -> List[FillReport]:
        """Get recent fills."""
        with self._lock:
            fills = list(self._fills)
            return fills[-count:]


# Module exports
__all__ = [
    'OrderSide',
    'OrderType',
    'OrderStatus',
    'PaperOrder',
    'FillReport',
    'OrderBookSimulator',
    'PaperExecutor'
]
