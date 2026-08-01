"""
Multi-Strategy Orchestration & Capital Allocation
Stage 49: Dynamic registry and lifecycle manager for concurrent Nautilus strategies.
Uses Ray actors for health monitoring and automatic quarantine.
"""

import ray
import asyncio
import logging
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import traceback
import zmq
import numpy as np

logger = logging.getLogger(__name__)


class StrategyState(Enum):
    """Strategy lifecycle states."""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    PAUSED = "paused"
    QUARANTINED = "quarantined"
    TERMINATED = "terminated"


@dataclass
class StrategyMetadata:
    """Metadata for a registered strategy."""
    strategy_id: str
    strategy_type: str  # StatArb, Trend, MM
    instance_id: str
    created_at: datetime
    state: StrategyState = StrategyState.INITIALIZING
    exception_count: int = 0
    last_heartbeat: datetime = field(default_factory=datetime.utcnow)
    capital_allocated: float = 0.0
    sortino_ratio: float = 0.0
    pnl_cumulative: float = 0.0


@ray.remote
class StrategyActor:
    """Ray actor wrapping a Nautilus strategy instance with health monitoring."""
    
    def __init__(self, strategy_id: str, strategy_type: str, instance_id: str, config: Dict[str, Any]):
        self.strategy_id = strategy_id
        self.strategy_type = strategy_type
        self.instance_id = instance_id
        self.config = config
        self.state = StrategyState.INITIALIZING
        self.exception_count = 0
        self.last_heartbeat = datetime.utcnow()
        self._strategy_instance = None
        self._initialized = False
        
    def initialize(self) -> bool:
        """Initialize the underlying Nautilus strategy."""
        try:
            # Lazy import to avoid circular dependencies
            from nautilus_trader.core.message import Message
            from nautilus_trader.model.identifiers import TraderId, StrategyId
            
            # Placeholder for actual strategy instantiation
            # In production, this would load the specific strategy class
            self._strategy_instance = {
                'type': self.strategy_type,
                'config': self.config,
                'trader_id': TraderId(f"TRADER-{self.instance_id}"),
                'strategy_id': StrategyId(self.strategy_id),
            }
            self.state = StrategyState.ACTIVE
            self._initialized = True
            logger.info(f"Strategy {self.strategy_id} initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize strategy {self.strategy_id}: {e}")
            self.state = StrategyState.QUARANTINED
            self.exception_count += 1
            return False
    
    def execute_tick(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Execute a single tick of the strategy logic."""
        if self.state != StrategyState.ACTIVE:
            return None
            
        try:
            self.last_heartbeat = datetime.utcnow()
            
            # Placeholder for actual strategy execution
            # Returns order signals or None
            if self._strategy_instance is None:
                return None
                
            # Simulate strategy logic based on type
            if self.strategy_type == "StatArb":
                return self._execute_statarb(market_data)
            elif self.strategy_type == "Trend":
                return self._execute_trend(market_data)
            elif self.strategy_type == "MM":
                return self._execute_mm(market_data)
            else:
                return None
                
        except Exception as e:
            logger.error(f"Strategy {self.strategy_id} exception: {e}")
            self.exception_count += 1
            if self.exception_count >= 3:
                self.state = StrategyState.QUARANTINED
                logger.warning(f"Strategy {self.strategy_id} quarantined after {self.exception_count} exceptions")
            return None
    
    def _execute_statarb(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Statistical arbitrage strategy execution."""
        # Placeholder implementation
        return {'signal': 'mean_revert', 'strength': 0.5, 'strategy_id': self.strategy_id}
    
    def _execute_trend(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Trend following strategy execution."""
        # Placeholder implementation
        return {'signal': 'momentum', 'strength': 0.7, 'strategy_id': self.strategy_id}
    
    def _execute_mm(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Market making strategy execution."""
        # Placeholder implementation
        return {'signal': 'spread', 'bid_size': 1.0, 'ask_size': 1.0, 'strategy_id': self.strategy_id}
    
    def update_parameters(self, params: Dict[str, Any]) -> bool:
        """Dynamically update strategy parameters without restart."""
        try:
            if self.state == StrategyState.QUARANTINED:
                return False
            
            # Update instance_id sizing parameters
            if 'instance_id' in params:
                self.config['instance_id'] = params['instance_id']
            
            if 'capital_allocation' in params:
                self.config['capital_allocation'] = params['capital_allocation']
            
            logger.info(f"Strategy {self.strategy_id} parameters updated")
            return True
        except Exception as e:
            logger.error(f"Failed to update parameters for {self.strategy_id}: {e}")
            return False
    
    def get_health_status(self) -> Dict[str, Any]:
        """Return current health status of the strategy."""
        return {
            'strategy_id': self.strategy_id,
            'state': self.state.value,
            'exception_count': self.exception_count,
            'last_heartbeat': self.last_heartbeat.isoformat(),
            'is_healthy': self.state == StrategyState.ACTIVE and self.exception_count < 3,
        }
    
    def shutdown(self) -> bool:
        """Gracefully shutdown the strategy."""
        try:
            self.state = StrategyState.TERMINATED
            logger.info(f"Strategy {self.strategy_id} shut down")
            return True
        except Exception as e:
            logger.error(f"Error shutting down strategy {self.strategy_id}: {e}")
            return False


class StrategyManager:
    """
    Central manager for registering, monitoring, and orchestrating multiple strategies.
    Uses Ray for distributed execution and automatic fault tolerance.
    """
    
    def __init__(self, max_strategies: int = 50, quarantine_threshold: int = 3):
        self.max_strategies = max_strategies
        self.quarantine_threshold = quarantine_threshold
        self.strategies: Dict[str, StrategyActor] = {}
        self.metadata: Dict[str, StrategyMetadata] = {}
        self.quarantined: Set[str] = set()
        self._zmq_context = zmq.Context()
        self._zmq_socket = self._zmq_context.socket(zmq.PUSH)
        self._zmq_socket.connect("tcp://localhost:5557")  # Rust IPC socket
        
    def register_strategy(self, strategy_id: str, strategy_type: str, 
                         instance_id: str, config: Dict[str, Any]) -> bool:
        """Register a new strategy with the manager."""
        if len(self.strategies) >= self.max_strategies:
            logger.error(f"Maximum strategy limit reached ({self.max_strategies})")
            return False
        
        if strategy_id in self.strategies:
            logger.warning(f"Strategy {strategy_id} already registered")
            return False
        
        try:
            # Create Ray actor
            actor = StrategyActor.remote(strategy_id, strategy_type, instance_id, config)
            
            # Initialize the strategy
            init_success = ray.get(actor.initialize.remote())
            
            if not init_success:
                logger.error(f"Strategy {strategy_id} failed initialization")
                return False
            
            self.strategies[strategy_id] = actor
            self.metadata[strategy_id] = StrategyMetadata(
                strategy_id=strategy_id,
                strategy_type=strategy_type,
                instance_id=instance_id,
                created_at=datetime.utcnow(),
                state=StrategyState.ACTIVE,
            )
            
            logger.info(f"Strategy {strategy_id} ({strategy_type}) registered successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register strategy {strategy_id}: {e}")
            return False
    
    def unregister_strategy(self, strategy_id: str) -> bool:
        """Unregister and shutdown a strategy."""
        if strategy_id not in self.strategies:
            return False
        
        try:
            actor = self.strategies[strategy_id]
            ray.get(actor.shutdown.remote())
            
            del self.strategies[strategy_id]
            del self.metadata[strategy_id]
            self.quarantined.discard(strategy_id)
            
            logger.info(f"Strategy {strategy_id} unregistered")
            return True
        except Exception as e:
            logger.error(f"Failed to unregister strategy {strategy_id}: {e}")
            return False
    
    async def monitor_health(self, check_interval: float = 1.0):
        """Continuously monitor strategy health and quarantine failing instances."""
        while True:
            try:
                for strategy_id, actor in list(self.strategies.items()):
                    health_status = await actor.get_health_status.remote()
                    
                    if not health_status['is_healthy']:
                        self.metadata[strategy_id].state = StrategyState.QUARANTINED
                        self.quarantined.add(strategy_id)
                        
                        # Notify Rust side via ZMQ
                        self._zmq_socket.send_json({
                            'type': 'STRATEGY_QUARANTINE',
                            'strategy_id': strategy_id,
                            'reason': f"Exception count: {health_status['exception_count']}",
                            'timestamp': datetime.utcnow().isoformat(),
                        })
                        
                        logger.warning(f"Strategy {strategy_id} quarantined")
                    
                    # Update heartbeat
                    self.metadata[strategy_id].last_heartbeat = datetime.utcnow()
                
                await asyncio.sleep(check_interval)
                
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(check_interval)
    
    def get_active_strategies(self) -> List[str]:
        """Return list of active (non-quarantined) strategy IDs."""
        return [
            sid for sid, meta in self.metadata.items()
            if meta.state == StrategyState.ACTIVE
        ]
    
    def get_strategy_by_type(self, strategy_type: str) -> List[str]:
        """Return list of strategy IDs matching a specific type."""
        return [
            sid for sid, meta in self.metadata.items()
            if meta.strategy_type == strategy_type and meta.state == StrategyState.ACTIVE
        ]
    
    def get_all_metadata(self) -> Dict[str, StrategyMetadata]:
        """Return metadata for all registered strategies."""
        return self.metadata.copy()
    
    def shutdown_all(self):
        """Shutdown all strategies and cleanup resources."""
        for strategy_id in list(self.strategies.keys()):
            self.unregister_strategy(strategy_id)
        
        self._zmq_socket.close()
        self._zmq_context.term()
        logger.info("All strategies shut down")


# Global instance for module-level access
_manager: Optional[StrategyManager] = None


def get_manager() -> StrategyManager:
    """Get or create the global StrategyManager instance."""
    global _manager
    if _manager is None:
        _manager = StrategyManager()
    return _manager


def initialize_ray():
    """Initialize Ray with memory constraints for 3GB Python RAM limit."""
    if not ray.is_initialized():
        ray.init(
            num_cpus=4,
            object_store_memory=512 * 1024 * 1024,  # 512MB object store
            _memory=2 * 1024 * 1024 * 1024,  # 2GB heap memory
            log_to_driver=False,
            ignore_reinit_error=True,
        )
