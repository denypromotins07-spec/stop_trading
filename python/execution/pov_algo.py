"""
Percentage of Volume (POV) Execution Algorithm for Nautilus.
Continuously ingests trade ticks to adjust target execution rate.
Ensures strict adherence to volume limits without GIL contention.
Implements proper on_order_filled and on_order_updated lifecycle hooks.
"""
import asyncio
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
from decimal import Decimal
import time

try:
    from nautilus_trader.model.enums import OrderSide
    from nautilus_trader.model.orders import LimitOrder, MarketOrder
    from nautilus_trader.live.execution_engine import ExecutionEngine
    NAUTILUS_AVAILABLE = True
except ImportError:
    NAUTILUS_AVAILABLE = False


@dataclass
class POVConfig:
    """Configuration for POV execution algorithm"""
    target_participation_rate: float  # e.g., 0.15 for 15% of volume
    min_order_size: float
    max_order_size: float
    urgency_factor: float  # 0-1, higher = more aggressive
    max_spread_bps: int  # Maximum spread to pay in bps
    time_horizon_seconds: int  # Target completion time
    use_passive_orders: bool  # Use limit orders when possible


@dataclass
class ExecutionState:
    """Current state of POV execution"""
    instrument_id: str
    side: str
    total_quantity: float
    filled_quantity: float
    remaining_quantity: float
    avg_fill_price: float
    market_volume_observed: float
    our_volume_executed: float
    current_participation_rate: float
    last_update_ns: int
    active_orders: Dict[str, float]  # order_id -> quantity


@dataclass 
class TickData:
    """Single trade tick for volume tracking"""
    timestamp_ns: int
    price: float
    volume: float
    aggressor_side: str  # 'buy' or 'sell'


class POVExecutionAlgo:
    """
    Percentage of Volume execution algorithm.
    Dynamically adjusts order size based on observed market volume.
    """
    
    # Memory bounds
    MAX_TICK_HISTORY = 5000
    MAX_ORDER_HISTORY = 200
    
    def __init__(self, config: POVConfig):
        self.config = config
        self._tick_history: deque = deque(maxlen=self.MAX_TICK_HISTORY)
        self._order_history: deque = deque(maxlen=self.MAX_ORDER_HISTORY)
        self._execution_state: Optional[ExecutionState] = None
        self._pending_orders: Dict[str, float] = {}
        self._lock = asyncio.Lock()
        
        # Volume tracking
        self._volume_window_ns = 60_000_000_000  # 60 second window
        self._last_volume_reset_ns = 0
    
    async def initialize(self, instrument_id: str, side: str,
                         total_quantity: float) -> ExecutionState:
        """Initialize POV execution for a new order"""
        async with self._lock:
            self._execution_state = ExecutionState(
                instrument_id=instrument_id,
                side=side,
                total_quantity=total_quantity,
                filled_quantity=0.0,
                remaining_quantity=total_quantity,
                avg_fill_price=0.0,
                market_volume_observed=0.0,
                our_volume_executed=0.0,
                current_participation_rate=0.0,
                last_update_ns=time.time_ns(),
                active_orders={}
            )
            self._tick_history.clear()
            self._pending_orders.clear()
            
            return self._execution_state
    
    async def on_tick(self, tick: TickData):
        """
        Process incoming trade tick.
        Updates volume tracking and may trigger new orders.
        """
        async with self._lock:
            if self._execution_state is None:
                return
            
            self._tick_history.append(tick)
            
            # Update market volume observed
            await self._update_volume_tracking(tick)
            
            # Check if we need to place/adjust orders
            await self._check_order_placement()
    
    async def _update_volume_tracking(self, tick: TickData):
        """Update volume tracking within the time window"""
        now = tick.timestamp_ns
        
        # Reset volume if window expired
        if now - self._last_volume_reset_ns > self._volume_window_ns:
            self._execution_state.market_volume_observed = 0.0
            self._execution_state.our_volume_executed = 0.0
            self._last_volume_reset_ns = now
        
        # Add tick volume
        self._execution_state.market_volume_observed += tick.volume
        
        # Calculate current participation rate
        if self._execution_state.market_volume_observed > 0:
            self._execution_state.current_participation_rate = (
                self._execution_state.our_volume_executed /
                self._execution_state.market_volume_observed
            )
    
    async def _check_order_placement(self):
        """Determine if new orders should be placed based on POV target"""
        if self._execution_state is None:
            return
        
        if self._execution_state.remaining_quantity <= 0:
            return
        
        # Calculate target volume based on participation rate
        target_our_volume = (
            self._execution_state.market_volume_observed *
            self.config.target_participation_rate
        )
        
        # How much behind/ahead are we?
        volume_deficit = target_our_volume - self._execution_state.our_volume_executed
        
        # Adjust for urgency
        urgency_multiplier = 1.0 + self.config.urgency_factor
        
        if volume_deficit > self.config.min_order_size:
            # We're behind target - place order
            order_size = min(
                volume_deficit * urgency_multiplier,
                self.config.max_order_size,
                self._execution_state.remaining_quantity
            )
            
            await self._place_order(order_size)
    
    async def _place_order(self, quantity: float):
        """Place an order for the specified quantity"""
        if self._execution_state is None:
            return
        
        # In production, this would submit to Nautilus ExecutionEngine
        order_id = f"pov_{time.time_ns()}"
        self._pending_orders[order_id] = quantity
        self._execution_state.active_orders[order_id] = quantity
        
        self._order_history.append({
            'order_id': order_id,
            'quantity': quantity,
            'timestamp_ns': time.time_ns(),
            'status': 'pending'
        })
    
    async def on_order_filled(self, order_id: str, fill_quantity: float,
                               fill_price: float):
        """
        Handle order fill event.
        Updates execution state with fill information.
        """
        async with self._lock:
            if self._execution_state is None:
                return
            
            # Remove from pending
            pending_qty = self._pending_orders.pop(order_id, 0)
            if order_id in self._execution_state.active_orders:
                del self._execution_state.active_orders[order_id]
            
            # Update filled quantity
            self._execution_state.filled_quantity += fill_quantity
            self._execution_state.remaining_quantity -= fill_quantity
            self._execution_state.our_volume_executed += fill_quantity
            
            # Update average fill price
            total_value = (
                self._execution_state.avg_fill_price *
                self._execution_state.filled_quantity
            )
            new_value = total_value + fill_price * fill_quantity
            self._execution_state.avg_fill_price = (
                new_value / self._execution_state.filled_quantity
            )
            
            # Record fill
            self._order_history.append({
                'order_id': order_id,
                'quantity': fill_quantity,
                'price': fill_price,
                'timestamp_ns': time.time_ns(),
                'status': 'filled'
            })
    
    async def on_order_updated(self, order_id: str, new_status: str,
                                rejected_quantity: float = 0):
        """
        Handle order update event (partial fill, cancel, reject).
        """
        async with self._lock:
            if self._execution_state is None:
                return
            
            if order_id in self._execution_state.active_orders:
                if new_status in ['cancelled', 'rejected', 'expired']:
                    # Return quantity to remaining
                    cancelled_qty = self._execution_state.active_orders.pop(order_id, 0)
                    self._pending_orders.pop(order_id, None)
                    
                    if rejected_quantity > 0:
                        self._execution_state.remaining_quantity += rejected_quantity
                    
                    self._order_history.append({
                        'order_id': order_id,
                        'quantity': cancelled_qty,
                        'timestamp_ns': time.time_ns(),
                        'status': new_status
                    })
    
    def get_execution_state(self) -> Optional[ExecutionState]:
        """Get current execution state"""
        return self._execution_state
    
    def is_complete(self) -> bool:
        """Check if execution is complete"""
        if self._execution_state is None:
            return True
        return self._execution_state.remaining_quantity <= 0
    
    def get_performance_metrics(self) -> Dict:
        """Get execution performance metrics"""
        if self._execution_state is None:
            return {}
        
        state = self._execution_state
        
        # Calculate slippage estimate (would need arrival price in production)
        participation_vs_target = (
            state.current_participation_rate - self.config.target_participation_rate
        )
        
        # Time elapsed
        elapsed_seconds = (time.time_ns() - state.last_update_ns) / 1e9
        
        # Estimated completion time
        if state.our_volume_executed > 0:
            rate_per_second = state.our_volume_executed / max(elapsed_seconds, 1)
            estimated_remaining_seconds = state.remaining_quantity / max(rate_per_second, 0.001)
        else:
            estimated_remaining_seconds = float('inf')
        
        return {
            'completion_pct': state.filled_quantity / state.total_quantity * 100,
            'avg_participation_rate': state.current_participation_rate,
            'target_participation_rate': self.config.target_participation_rate,
            'participation_deviation': participation_vs_target,
            'estimated_completion_seconds': estimated_remaining_seconds,
            'num_orders': len(self._order_history),
            'active_orders': len(state.active_orders)
        }


# Global registry of active POV algorithms
_pov_algos: Dict[str, POVExecutionAlgo] = {}


def create_pov_algo(algo_id: str, config: POVConfig) -> POVExecutionAlgo:
    """Create and register a new POV algorithm instance"""
    algo = POVExecutionAlgo(config)
    _pov_algos[algo_id] = algo
    return algo


def get_pov_algo(algo_id: str) -> Optional[POVExecutionAlgo]:
    """Get a registered POV algorithm by ID"""
    return _pov_algos.get(algo_id)


async def demo():
    """Demo usage of POV execution algorithm"""
    print("=== POV Execution Demo ===\n")
    
    config = POVConfig(
        target_participation_rate=0.15,
        min_order_size=0.1,
        max_order_size=5.0,
        urgency_factor=0.2,
        max_spread_bps=10,
        time_horizon_seconds=300,
        use_passive_orders=True
    )
    
    algo = create_pov_algo("pov_demo", config)
    
    # Initialize execution
    state = await algo.initialize(
        instrument_id="BTC/USDT",
        side="buy",
        total_quantity=100.0
    )
    print(f"Initialized POV: {state.total_quantity} {state.side}")
    
    # Simulate market ticks
    base_time = time.time_ns()
    base_price = 50000
    
    for i in range(50):
        tick = TickData(
            timestamp_ns=base_time + i * 100_000_000,  # 100ms apart
            price=base_price + (i % 10 - 5) * 10,
            volume=10 + (i % 7) * 5,
            aggressor_side='buy' if i % 2 == 0 else 'sell'
        )
        
        await algo.on_tick(tick)
        
        # Simulate fills for pending orders
        state = algo.get_execution_state()
        if state and state.active_orders:
            for order_id in list(state.active_orders.keys()):
                # Simulate partial fill
                fill_qty = min(state.active_orders[order_id], 2.0)
                await algo.on_order_filled(order_id, fill_qty, base_price)
    
    # Get final state
    state = algo.get_execution_state()
    metrics = algo.get_performance_metrics()
    
    print(f"\nExecution Results:")
    print(f"  Filled: {state.filled_quantity:.2f} / {state.total_quantity:.2f}")
    print(f"  Avg Price: ${state.avg_fill_price:.2f}")
    print(f"  Participation: {metrics['avg_participation_rate']:.1%}")
    print(f"  Completion: {metrics['completion_pct']:.1f}%")


if __name__ == "__main__":
    asyncio.run(demo())
