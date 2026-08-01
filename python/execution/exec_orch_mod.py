"""
Chapter 2: Advanced Execution Orchestration & Smart Scheduling
File: python/execution/exec_orch_mod.py

Module root for execution orchestration.
Wires advanced schedulers directly into Nautilus ExecutionEngine via MessageBus.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime
import json

# Import local modules
from .twap_scheduler import (
    TWAPConfig,
    MarketCondition,
    RLDrivenTWAPScheduler,
    ExecutionState
)
from .iceberg_manager import (
    IcebergConfig,
    OrderSide,
    L3QueueState,
    IcebergOrderManager,
    IcebergState
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ExecutionOrder:
    """Represents an execution order for the orchestrator."""
    order_id: str
    strategy: str  # "twap" or "iceberg"
    symbol: str
    side: str
    total_quantity: float
    limit_price: float
    duration_minutes: int = 60
    config: Optional[Dict] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


@dataclass
class ExecutionResult:
    """Result of an executed order."""
    order_id: str
    status: str
    filled_quantity: float
    avg_price: float
    slippage_bps: float
    start_time: datetime
    end_time: Optional[datetime] = None
    metadata: Dict = None


class NautilusMessageBusAdapter:
    """
    Adapter for Nautilus Trader MessageBus integration.
    Handles message routing between Python execution and Rust engine.
    """
    
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
        self.publishers: Dict[str, Callable] = {}
    
    def subscribe(self, topic: str, callback: Callable):
        """Subscribe to a message topic."""
        if topic not in self.subscribers:
            self.subscribers[topic] = []
        self.subscribers[topic].append(callback)
        logger.info(f"Subscribed to topic: {topic}")
    
    def unsubscribe(self, topic: str, callback: Callable):
        """Unsubscribe from a topic."""
        if topic in self.subscribers:
            self.subscribers[topic].remove(callback)
    
    def publish(self, topic: str, message: Any):
        """Publish a message to all subscribers."""
        if topic in self.subscribers:
            for callback in self.subscribers[topic]:
                try:
                    callback(message)
                except Exception as e:
                    logger.error(f"Error publishing to {topic}: {e}")
    
    def register_publisher(self, topic: str, publisher: Callable):
        """Register a publisher for a topic."""
        self.publishers[topic] = publisher
    
    async def send_order(self, order: Dict) -> bool:
        """Send order to Nautilus ExecutionEngine."""
        if "nautilus_order" in self.publishers:
            return await self.publishers["nautilus_order"](order)
        logger.warning("No Nautilus order publisher registered")
        return False
    
    async def receive_fill(self, fill_data: Dict):
        """Receive fill update from Nautilus."""
        self.publish("nautilus_fill", fill_data)
    
    async def receive_l3_update(self, l3_data: Dict):
        """Receive L3 queue update from Rust."""
        self.publish("l3_update", l3_data)


class ExecutionOrchestrator:
    """
    Main execution orchestrator managing TWAP and Iceberg strategies.
    Integrates with Nautilus ExecutionEngine via MessageBus.
    """
    
    def __init__(self):
        self.message_bus = NautilusMessageBusAdapter()
        
        # Active executions
        self.active_twap: Dict[str, RLDrivenTWAPScheduler] = {}
        self.active_iceberg: Dict[str, IcebergOrderManager] = {}
        
        # Completed executions
        self.completed_orders: Dict[str, ExecutionResult] = {}
        
        # Market data cache
        self.market_conditions: Dict[str, MarketCondition] = {}
        self.l3_states: Dict[str, L3QueueState] = {}
        
        # Setup message handlers
        self._setup_message_handlers()
    
    def _setup_message_handlers(self):
        """Setup message bus handlers for Nautilus integration."""
        # Handle fill updates
        self.message_bus.subscribe("nautilus_fill", self._handle_fill_update)
        
        # Handle L3 updates
        self.message_bus.subscribe("l3_update", self._handle_l3_update)
        
        # Register order submission handler
        self.message_bus.register_publisher(
            "nautilus_order", 
            self._submit_to_nautilus
        )
    
    async def _submit_to_nautilus(self, order: Dict) -> bool:
        """Submit order to Nautilus ExecutionEngine."""
        # This would integrate with actual Nautilus message bus
        logger.info(f"Submitting order to Nautilus: {order.get('order_id')}")
        # In production: send via ZMQ or shared memory to Rust
        return True
    
    def _handle_fill_update(self, fill_data: Dict):
        """Handle fill update from Nautilus."""
        order_id = fill_data.get("order_id")
        
        # Update TWAP if active
        if order_id in self.active_twap:
            logger.debug(f"TWAP fill update: {order_id}")
        
        # Update Iceberg if active
        if order_id in self.active_iceberg:
            iceberg = self.active_iceberg[order_id]
            clip_id = fill_data.get("clip_id", 0)
            filled_qty = fill_data.get("quantity", 0)
            fill_price = fill_data.get("price", 0)
            iceberg.update_fill(clip_id, filled_qty, fill_price)
    
    def _handle_l3_update(self, l3_data: Dict):
        """Handle L3 queue update from Rust."""
        symbol = l3_data.get("symbol", "default")
        
        l3_state = L3QueueState(
            best_bid=l3_data.get("best_bid", 0),
            best_ask=l3_data.get("best_ask", 0),
            bid_queue_size=l3_data.get("bid_queue_size", 0),
            ask_queue_size=l3_data.get("ask_queue_size", 0),
            recent_trade_volume=l3_data.get("recent_trade_volume", 0),
            trade_flow_rate=l3_data.get("trade_flow_rate", 0)
        )
        
        self.l3_states[symbol] = l3_state
        
        # Update TWAP market conditions if active
        if symbol in self.active_twap:
            twap = self.active_twap[symbol]
            # Convert L3 to market condition
            spread_bps = l3_state.get_spread_bps()
            market_cond = MarketCondition(
                spread_bps=spread_bps,
                volume_profile=l3_state.recent_trade_volume,
                timestamp=datetime.utcnow()
            )
            self.market_conditions[symbol] = market_cond
    
    async def start_twap(
        self,
        order: ExecutionOrder
    ) -> bool:
        """Start a TWAP execution."""
        if order.order_id in self.active_twap:
            logger.warning(f"TWAP already active for {order.order_id}")
            return False
        
        # Create TWAP scheduler
        twap_config = TWAPConfig(
            total_quantity=order.total_quantity,
            duration_minutes=order.duration_minutes,
            **(order.config or {})
        )
        
        twap = RLDrivenTWAPScheduler(twap_config)
        
        # Initialize
        twap.initialize(
            total_quantity=order.total_quantity,
            duration_minutes=order.duration_minutes
        )
        
        # Setup callbacks
        async def execute_callback(quantity: float, market_condition: MarketCondition):
            order_msg = {
                "order_id": order.order_id,
                "symbol": order.symbol,
                "side": order.side,
                "quantity": quantity,
                "limit_price": order.limit_price,
                "time_in_force": "IOC"
            }
            success = await self.message_bus.send_order(order_msg)
            return {
                "executed_quantity": quantity if success else 0,
                "avg_price": order.limit_price,
                "slippage_bps": 0
            }
        
        twap.execution_callback = execute_callback
        
        self.active_twap[order.order_id] = twap
        
        # Start async execution
        asyncio.create_task(
            self._run_twap_execution(order.order_id, twap)
        )
        
        logger.info(f"Started TWAP: {order.order_id}")
        return True
    
    async def _run_twap_execution(
        self, 
        order_id: str, 
        twap: RLDrivenTWAPScheduler
    ):
        """Run TWAP execution loop."""
        async def get_market_condition():
            # Get latest market condition or create default
            return self.market_conditions.get(
                "default", 
                MarketCondition()
            )
        
        try:
            await twap.run(get_market_condition)
            
            # Mark as completed
            progress = twap.get_progress()
            result = ExecutionResult(
                order_id=order_id,
                status=progress["state"],
                filled_quantity=progress["total_executed"],
                avg_price=0,  # Would be tracked in real implementation
                slippage_bps=0,
                start_time=datetime.utcnow()
            )
            self.completed_orders[order_id] = result
            
        except Exception as e:
            logger.error(f"TWAP execution failed: {e}")
        finally:
            if order_id in self.active_twap:
                del self.active_twap[order_id]
    
    async def start_iceberg(
        self,
        order: ExecutionOrder
    ) -> bool:
        """Start an iceberg execution."""
        if order.order_id in self.active_iceberg:
            logger.warning(f"Iceberg already active for {order.order_id}")
            return False
        
        # Create iceberg manager
        iceberg_config = IcebergConfig(
            total_quantity=order.total_quantity,
            **(order.config or {})
        )
        
        iceberg = IcebergOrderManager(iceberg_config)
        
        # Setup callbacks
        async def submit_callback(side: str, quantity: float, limit_price: float, **kwargs):
            order_msg = {
                "order_id": order.order_id,
                "symbol": order.symbol,
                "side": side,
                "quantity": quantity,
                "limit_price": limit_price,
                "is_iceberg": True
            }
            return {"success": await self.message_bus.send_order(order_msg)}
        
        iceberg.submit_callback = submit_callback
        
        # Sync with L3 state
        if order.symbol in self.l3_states:
            iceberg.sync_l3_state(self.l3_states[order.symbol])
        
        self.active_iceberg[order.order_id] = iceberg
        
        # Start async execution
        side = OrderSide.BUY if order.side.lower() == "buy" else OrderSide.SELL
        asyncio.create_task(
            iceberg.run(side, order.limit_price)
        )
        
        logger.info(f"Started Iceberg: {order.order_id}")
        return True
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an active order."""
        cancelled = False
        
        if order_id in self.active_twap:
            self.active_twap[order_id].abort("User cancelled")
            del self.active_twap[order_id]
            cancelled = True
        
        if order_id in self.active_iceberg:
            self.active_iceberg[order_id].cancel()
            del self.active_iceberg[order_id]
            cancelled = True
        
        if cancelled:
            logger.info(f"Cancelled order: {order_id}")
        
        return cancelled
    
    def get_order_status(self, order_id: str) -> Optional[Dict]:
        """Get status of an order."""
        if order_id in self.active_twap:
            return {
                "type": "twap",
                "status": "active",
                "progress": self.active_twap[order_id].get_progress()
            }
        
        if order_id in self.active_iceberg:
            return {
                "type": "iceberg",
                "status": "active",
                "progress": self.active_iceberg[order_id].get_progress()
            }
        
        if order_id in self.completed_orders:
            result = self.completed_orders[order_id]
            return {
                "type": "completed",
                "status": result.status,
                "filled_quantity": result.filled_quantity,
                "avg_price": result.avg_price
            }
        
        return None
    
    def get_all_active_orders(self) -> List[Dict]:
        """Get all active orders."""
        active = []
        
        for order_id, twap in self.active_twap.items():
            active.append({
                "order_id": order_id,
                "type": "twap",
                "progress": twap.get_progress()
            })
        
        for order_id, iceberg in self.active_iceberg.items():
            active.append({
                "order_id": order_id,
                "type": "iceberg",
                "progress": iceberg.get_progress()
            })
        
        return active
    
    async def shutdown(self):
        """Gracefully shutdown all active executions."""
        logger.info("Shutting down execution orchestrator...")
        
        # Cancel all TWAPs
        for order_id in list(self.active_twap.keys()):
            self.cancel_order(order_id)
        
        # Cancel all icebergs
        for order_id in list(self.active_iceberg.keys()):
            self.cancel_order(order_id)
        
        logger.info("Execution orchestrator shutdown complete")


# Export for module use
__all__ = [
    "ExecutionOrder",
    "ExecutionResult",
    "NautilusMessageBusAdapter",
    "ExecutionOrchestrator"
]
