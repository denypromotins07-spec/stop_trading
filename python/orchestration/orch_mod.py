"""
Orchestration Module Root
Stage 49: Wiring strategy manager and allocator to Nautilus Portfolio and RiskEngine via MessageBus.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Callable
from datetime import datetime
import zmq
import json

from .strategy_manager import StrategyManager, get_manager, initialize_ray, StrategyState
from .capital_allocator import CapitalAllocator, get_allocator

logger = logging.getLogger(__name__)


class OrchestratorModule:
    """
    Central orchestration module wiring together strategy management,
    capital allocation, and Nautilus infrastructure via MessageBus.
    Handles /START and /KILL commands from Rust side.
    """
    
    def __init__(self, config: Dict[str, Any], ipc_endpoint: str = "tcp://localhost:5556"):
        self.config = config
        self.ipc_endpoint = ipc_endpoint
        self.strategy_manager: Optional[StrategyManager] = None
        self.capital_allocator: Optional[CapitalAllocator] = None
        self._running = False
        self._started = False
        self._tasks = []
        
        # ZMQ sockets for IPC
        self._zmq_context: Optional[zmq.Context] = None
        self._pub_socket: Optional[zmq.Socket] = None
        self._sub_socket: Optional[zmq.Socket] = None
        self._rep_socket: Optional[zmq.Socket] = None
        
        # MessageBus integration points
        self._message_handlers = {}
        self._start_callbacks = []
        self._kill_callbacks = []
        
    async def initialize(self) -> bool:
        """Initialize the orchestration module."""
        try:
            logger.info("Initializing OrchestratorModule...")
            
            # Initialize Ray for distributed strategy execution
            initialize_ray()
            
            # Create strategy manager
            self.strategy_manager = StrategyManager(
                max_strategies=self.config.get('max_strategies', 50),
                quarantine_threshold=self.config.get('quarantine_threshold', 3),
            )
            
            # Create capital allocator
            self.capital_allocator = CapitalAllocator(
                total_capital=self.config.get('total_capital', 1_000_000.0),
                max_kelly_fraction=self.config.get('max_kelly_fraction', 0.25),
                min_sortino_threshold=self.config.get('min_sortino_threshold', 0.5),
            )
            
            # Setup ZMQ for IPC with Rust
            self._zmq_context = zmq.Context()
            self._rep_socket = self._zmq_context.socket(zmq.REP)
            self._rep_socket.bind(self.ipc_endpoint)
            
            self._pub_socket = self._zmq_context.socket(zmq.PUB)
            self._pub_socket.bind("tcp://*:5559")
            
            # Setup MessageBus subscriptions
            await self._setup_messagebus()
            
            self._running = True
            logger.info("OrchestratorModule initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize OrchestratorModule: {e}")
            return False
    
    async def _setup_messagebus(self):
        """Setup Nautilus MessageBus integration."""
        # Subscribe to relevant Nautilus events
        self._message_handlers = {
            'OrderFilled': self._on_order_filled,
            'PositionClosed': self._on_position_closed,
            'PortfolioUpdated': self._on_portfolio_updated,
            'RiskEngineAlert': self._on_risk_alert,
        }
        
        logger.info("MessageBus handlers registered")
    
    async def _on_order_filled(self, event: Dict[str, Any]):
        """Handle OrderFilled events from Nautilus."""
        strategy_id = event.get('strategy_id')
        pnl = event.get('pnl', 0.0)
        
        if strategy_id and self.capital_allocator:
            self.capital_allocator.record_pnl(strategy_id, pnl)
        
        # Publish to internal bus
        self._publish_event('ORDER_FILLED', event)
    
    async def _on_position_closed(self, event: Dict[str, Any]):
        """Handle PositionClosed events from Nautilus."""
        strategy_id = event.get('strategy_id')
        pnl = event.get('realized_pnl', 0.0)
        
        if strategy_id and self.capital_allocator:
            self.capital_allocator.record_pnl(strategy_id, pnl)
        
        self._publish_event('POSITION_CLOSED', event)
    
    async def _on_portfolio_updated(self, event: Dict[str, Any]):
        """Handle Portfolio update events."""
        self._publish_event('PORTFOLIO_UPDATED', event)
    
    async def _on_risk_alert(self, event: Dict[str, Any]):
        """Handle RiskEngine alerts."""
        alert_type = event.get('alert_type')
        severity = event.get('severity')
        
        logger.warning(f"Risk alert: {alert_type} (severity: {severity})")
        
        # Potentially trigger strategy quarantine based on risk alerts
        if severity == 'CRITICAL':
            strategy_id = event.get('strategy_id')
            if strategy_id and self.strategy_manager:
                logger.warning(f"Quarantining strategy {strategy_id} due to critical risk alert")
    
    def _publish_event(self, event_type: str, data: Dict[str, Any]):
        """Publish event to internal message bus."""
        try:
            message = {
                'type': event_type,
                'data': data,
                'timestamp': datetime.utcnow().isoformat(),
            }
            self._pub_socket.send_json(message, flags=zmq.NOBLOCK)
        except Exception as e:
            logger.error(f"Failed to publish event: {e}")
    
    async def start_listening(self):
        """Start listening for IPC commands from Rust."""
        if self._running:
            return
        
        logger.info("Starting IPC command listener...")
        
        while self._running:
            try:
                message = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._rep_socket.poll(timeout=1000),
                )
                
                if message:
                    command = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self._rep_socket.recv_string(),
                    )
                    await self._handle_command(command)
                    
            except Exception as e:
                logger.error(f"IPC listener error: {e}")
                await asyncio.sleep(1.0)
    
    async def _handle_command(self, command: str):
        """Handle incoming IPC command from Rust."""
        try:
            cmd_data = json.loads(command) if command.startswith("{") else {"cmd": command}
            cmd = cmd_data.get("cmd", command).upper()
            
            logger.info(f"Received command: {cmd}")
            
            response = {"status": "ok", "cmd": cmd}
            
            if cmd == "START":
                if not self._started:
                    await self._execute_start()
                else:
                    response["status"] = "already_started"
            
            elif cmd == "KILL":
                await self._execute_kill(cmd_data.get("reason", "unknown"))
            
            elif cmd == "STATUS":
                response["data"] = self.get_status()
            
            elif cmd == "REGISTER_STRATEGY":
                strategy_id = cmd_data.get('strategy_id')
                strategy_type = cmd_data.get('strategy_type')
                instance_id = cmd_data.get('instance_id')
                initial_capital = cmd_data.get('initial_capital', 10000.0)
                max_capital = cmd_data.get('max_capital', 50000.0)
                
                success = await self.register_strategy(
                    strategy_id, strategy_type, instance_id, {},
                    initial_capital, max_capital
                )
                response["status"] = "success" if success else "failed"
            
            elif cmd == "GET_ALLOCATIONS":
                response["data"] = self.get_strategy_allocations()
            
            else:
                response["status"] = "unknown_command"
            
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._rep_socket.send_string(json.dumps(response)),
            )
            
        except Exception as e:
            logger.error(f"Command handling error: {e}")
            try:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._rep_socket.send_string(json.dumps({
                        "status": "error",
                        "error": str(e),
                    })),
                )
            except Exception:
                pass
    
    async def _execute_start(self):
        """Execute START command - launch all subsystems."""
        logger.info("Executing START command...")
        
        self._started = True
        
        for callback in self._start_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback()
                else:
                    callback()
            except Exception as e:
                logger.error(f"Start callback error: {e}")
        
        logger.info("START command completed")
    
    async def _execute_kill(self, reason: str = "unknown"):
        """Execute KILL command - graceful shutdown."""
        logger.critical(f"Executing KILL command (reason: {reason})")
        
        for callback in reversed(self._kill_callbacks):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback()
                else:
                    callback()
            except Exception as e:
                logger.error(f"Kill callback error: {e}")
        
        self._running = False
        self._started = False
        
        logger.info("KILL command completed")
    
    async def register_strategy(self, strategy_id: str, strategy_type: str,
                               instance_id: str, config: Dict[str, Any],
                               initial_capital: float, max_capital: float) -> bool:
        """Register a new strategy with capital allocation."""
        if not self.strategy_manager or not self.capital_allocator:
            logger.error("OrchestratorModule not initialized")
            return False
        
        # Register with strategy manager
        success = self.strategy_manager.register_strategy(
            strategy_id, strategy_type, instance_id, config
        )
        
        if not success:
            return False
        
        # Register with capital allocator
        self.capital_allocator.register_strategy(
            strategy_id, initial_capital, max_capital
        )
        
        logger.info(f"Strategy {strategy_id} fully registered")
        return True
    
    async def start_health_monitor(self, check_interval: float = 1.0):
        """Start the strategy health monitoring loop."""
        if not self.strategy_manager:
            raise RuntimeError("StrategyManager not initialized")
        
        async def monitor_loop():
            await self.strategy_manager.monitor_health(check_interval)
        
        task = asyncio.create_task(monitor_loop())
        self._tasks.append(task)
        logger.info("Health monitor started")
    
    async def start_rebalancer(self, rebalance_interval: float = 60.0):
        """Start periodic capital rebalancing."""
        if not self.capital_allocator:
            raise RuntimeError("CapitalAllocator not initialized")
        
        async def rebalance_loop():
            while self._running:
                try:
                    updates = self.capital_allocator.rebalance_all()
                    
                    if updates:
                        for strategy_id, delta in updates.items():
                            strategy_meta = self.strategy_manager.metadata.get(strategy_id)
                            if strategy_meta:
                                actor = self.strategy_manager.strategies.get(strategy_id)
                                if actor:
                                    await actor.update_parameters.remote({
                                        'capital_allocation': strategy_meta.capital_allocated,
                                    })
                    
                    await asyncio.sleep(rebalance_interval)
                    
                except Exception as e:
                    logger.error(f"Rebalance error: {e}")
                    await asyncio.sleep(rebalance_interval)
        
        task = asyncio.create_task(rebalance_loop())
        self._tasks.append(task)
        logger.info("Capital rebalancer started")
    
    def get_active_strategies(self) -> list:
        """Get list of active strategy IDs."""
        if not self.strategy_manager:
            return []
        return self.strategy_manager.get_active_strategies()
    
    def get_strategy_allocations(self) -> Dict[str, Any]:
        """Get current capital allocations for all strategies."""
        if not self.capital_allocator:
            return {}
        return {
            sid: {
                'allocated': alloc.allocated_capital,
                'kelly_fraction': alloc.kelly_fraction,
                'sortino_ratio': alloc.sortino_ratio,
            }
            for sid, alloc in self.capital_allocator.get_all_allocations().items()
        }
    
    def register_start_callback(self, callback: Callable):
        """Register a callback for START command."""
        self._start_callbacks.append(callback)
    
    def register_kill_callback(self, callback: Callable):
        """Register a callback for KILL command."""
        self._kill_callbacks.append(callback)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current orchestration status."""
        return {
            "running": self._running,
            "started": self._started,
            "strategy_count": len(self.strategy_manager.strategies) if self.strategy_manager else 0,
            "active_strategies": len(self.get_active_strategies()),
            "total_allocated": self.capital_allocator.get_total_allocated() if self.capital_allocator else 0.0,
            "ipc_endpoint": self.ipc_endpoint,
        }
    
    async def shutdown(self):
        """Gracefully shutdown the orchestration module."""
        logger.info("Shutting down OrchestratorModule...")
        self._running = False
        
        # Cancel all tasks
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        # Shutdown components
        if self.strategy_manager:
            self.strategy_manager.shutdown_all()
        
        if self.capital_allocator:
            self.capital_allocator.shutdown()
        
        # Close ZMQ sockets
        if self._rep_socket:
            self._rep_socket.close()
        if self._pub_socket:
            self._pub_socket.close()
        if self._zmq_context:
            self._zmq_context.term()
        
        logger.info("OrchestratorModule shut down complete")


# Global module instance
_module: Optional[OrchestratorModule] = None


def get_module() -> OrchestratorModule:
    """Get or create the global OrchestratorModule instance."""
    global _module
    if _module is None:
        _module = OrchestratorModule({})
    return _module


def create_module(config: Dict[str, Any]) -> OrchestratorModule:
    """Create a new OrchestratorModule with custom configuration."""
    global _module
    _module = OrchestratorModule(config)
    return _module
