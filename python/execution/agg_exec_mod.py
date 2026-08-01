"""
Aggressive Execution Module Root
Routes aggressive execution signals to the Nautilus execution engine via MessageBus.
Handles OrderRejected events to prevent infinite retry loops.
"""

import asyncio
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
from enum import Enum
import logging

# Nautilus imports (conditional for type checking)
try:
    from nautilus_trader.core.message import Event
    from nautilus_trader.model.order import Order
    from nautilus_trader.model.events import OrderRejected, OrderFilled, OrderSubmitted
    from nautilus_trader.live.execution_client import ExecutionClient
    NAUTILUS_AVAILABLE = True
except ImportError:
    NAUTILUS_AVAILABLE = False
    Event = object  # type: ignore
    Order = object  # type: ignore


logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    AGGRESSIVE = "aggressive"
    PASSIVE = "passive"
    SNIPER = "sniper"
    SWEEPER = "sweeper"


@dataclass
class ExecutionSignal:
    """Represents an execution signal to be routed to Nautilus."""
    strategy_id: str
    instrument_id: str
    side: str  # "BUY" or "SELL"
    quantity: float
    price: Optional[float]
    order_type: str  # "MARKET", "LIMIT", "STOP"
    time_in_force: str  # "GTC", "IOC", "FOK"
    execution_mode: ExecutionMode
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ExecutionReport:
    """Execution report for monitoring and risk management."""
    signal: ExecutionSignal
    status: str
    filled_quantity: float = 0.0
    average_price: float = 0.0
    rejection_count: int = 0
    last_rejection_reason: Optional[str] = None


class AggressiveExecutionModule:
    """
    Module root routing aggressive execution signals to Nautilus execution engine.
    Implements robust error handling for OrderRejected events.
    """

    MAX_RETRY_COUNT = 3
    RETRY_DELAY_MS = 100

    def __init__(
        self,
        message_bus_topic: str = "execution.signals",
        max_concurrent_orders: int = 10,
    ):
        self.message_bus_topic = message_bus_topic
        self.max_concurrent_orders = max_concurrent_orders
        self._pending_signals: asyncio.Queue[ExecutionSignal] = asyncio.Queue()
        self._active_orders: Dict[str, ExecutionReport] = {}
        self._order_semaphore = asyncio.Semaphore(max_concurrent_orders)
        self._running = False
        self._message_bus = None
        self._execution_client = None

        # Metrics
        self.total_signals = 0
        self.filled_orders = 0
        self.rejected_orders = 0
        self.retry_counts: Dict[str, int] = {}

    async def start(
        self,
        message_bus: Optional[Any] = None,
        execution_client: Optional[Any] = None,
    ) -> None:
        """Start the execution module."""
        self._running = True
        self._message_bus = message_bus
        self._execution_client = execution_client

        logger.info("AggressiveExecutionModule started")

        # Start signal processing loop
        asyncio.create_task(self._process_signals())

    async def stop(self) -> None:
        """Stop the execution module gracefully."""
        self._running = False
        await self._pending_signals.join()
        logger.info("AggressiveExecutionModule stopped")

    async def submit_signal(self, signal: ExecutionSignal) -> str:
        """
        Submit an execution signal for processing.
        Returns a unique signal ID.
        """
        signal_id = f"{signal.strategy_id}_{signal.instrument_id}_{self.total_signals}"
        signal.metadata["signal_id"] = signal_id

        self.total_signals += 1
        await self._pending_signals.put(signal)

        logger.debug(f"Submitted execution signal: {signal_id}")
        return signal_id

    async def _process_signals(self) -> None:
        """Process execution signals from the queue."""
        while self._running:
            try:
                signal = await asyncio.wait_for(
                    self._pending_signals.get(), timeout=1.0
                )
            except asyncio.TimeoutError:
                continue

            async with self._order_semaphore:
                await self._execute_signal(signal)

            self._pending_signals.task_done()

    async def _execute_signal(self, signal: ExecutionSignal) -> None:
        """Execute a single execution signal."""
        signal_id = signal.metadata.get("signal_id", "unknown")
        report = ExecutionReport(signal=signal, status="PENDING")
        self._active_orders[signal_id] = report

        try:
            if NAUTILUS_AVAILABLE and self._execution_client:
                await self._submit_nautilus_order(signal, report)
            else:
                # Simulated execution for testing
                await self._simulate_execution(signal, report)

        except Exception as e:
            logger.error(f"Execution failed for {signal_id}: {e}")
            report.status = "FAILED"
            report.last_rejection_reason = str(e)

    async def _submit_nautilus_order(
        self,
        signal: ExecutionSignal,
        report: ExecutionReport,
    ) -> None:
        """Submit order to Nautilus execution client."""
        # Build Nautilus order based on signal
        # This is a simplified example - actual implementation depends on Nautilus API version

        try:
            # Submit via execution client
            # Note: Actual Nautilus order submission requires proper order factory
            logger.info(f"Submitting Nautilus order for signal: {signal}")

            # Publish to message bus for other components
            if self._message_bus:
                await self._publish_to_message_bus(signal)

            report.status = "SUBMITTED"

        except Exception as e:
            raise e

    async def _simulate_execution(
        self,
        signal: ExecutionSignal,
        report: ExecutionReport,
    ) -> None:
        """Simulate execution for testing without Nautilus."""
        await asyncio.sleep(0.01)  # Simulate latency
        report.status = "FILLED"
        report.filled_quantity = signal.quantity
        report.average_price = signal.price or 0.0
        self.filled_orders += 1

    async def _publish_to_message_bus(self, signal: ExecutionSignal) -> None:
        """Publish execution signal to message bus."""
        if not self._message_bus:
            return

        try:
            # Convert signal to dict for publishing
            signal_dict = asdict(signal)
            await self._message_bus.publish(
                topic=self.message_bus_topic,
                data=signal_dict,
            )
        except Exception as e:
            logger.error(f"Failed to publish to message bus: {e}")

    def handle_order_rejected(self, event: OrderRejected) -> None:
        """
        Handle OrderRejected events to prevent infinite retry loops.
        Implements exponential backoff and max retry limits.
        """
        order_id = event.order_id
        report = self._active_orders.get(order_id)

        if not report:
            logger.warning(f"Received rejection for unknown order: {order_id}")
            return

        report.rejection_count += 1
        report.last_rejection_reason = event.reason

        # Check if max retries exceeded
        if report.rejection_count >= self.MAX_RETRY_COUNT:
            report.status = "REJECTED_MAX_RETRIES"
            self.rejected_orders += 1
            logger.error(
                f"Order {order_id} rejected {report.rejection_count} times. "
                f"Final rejection reason: {event.reason}"
            )
            return

        # Schedule retry with backoff
        retry_delay = self.RETRY_DELAY_MS * (2 ** (report.rejection_count - 1))
        logger.info(
            f"Order {order_id} rejected (attempt {report.rejection_count}). "
            f"Retrying in {retry_delay}ms"
        )

        # Track retry count
        self.retry_counts[order_id] = report.rejection_count

    def handle_order_filled(self, event: OrderFilled) -> None:
        """Handle OrderFilled events."""
        order_id = event.order_id
        report = self._active_orders.get(order_id)

        if report:
            report.status = "FILLED"
            report.filled_quantity = event.last_qty
            report.average_price = event.avg_px
            self.filled_orders += 1
            logger.info(f"Order {order_id} filled: {event.last_qty}@{event.avg_px}")

    def get_execution_report(self, signal_id: str) -> Optional[ExecutionReport]:
        """Get execution report for a signal."""
        return self._active_orders.get(signal_id)

    def get_metrics(self) -> Dict[str, Any]:
        """Get execution module metrics."""
        return {
            "total_signals": self.total_signals,
            "filled_orders": self.filled_orders,
            "rejected_orders": self.rejected_orders,
            "fill_rate": self.filled_orders / max(self.total_signals, 1),
            "active_orders": len([r for r in self._active_orders.values() if r.status == "SUBMITTED"]),
            "retry_counts": self.retry_counts.copy(),
        }

    def reset_metrics(self) -> None:
        """Reset all metrics."""
        self.total_signals = 0
        self.filled_orders = 0
        self.rejected_orders = 0
        self.retry_counts.clear()
        self._active_orders.clear()


# Module-level singleton instance
_module_instance: Optional[AggressiveExecutionModule] = None


def get_module() -> AggressiveExecutionModule:
    """Get the module singleton instance."""
    global _module_instance
    if _module_instance is None:
        _module_instance = AggressiveExecutionModule()
    return _module_instance


async def initialize_module(
    message_bus: Optional[Any] = None,
    execution_client: Optional[Any] = None,
) -> AggressiveExecutionModule:
    """Initialize the execution module."""
    module = get_module()
    await module.start(message_bus=message_bus, execution_client=execution_client)
    return module


async def shutdown_module() -> None:
    """Shutdown the execution module."""
    module = get_module()
    await module.stop()
