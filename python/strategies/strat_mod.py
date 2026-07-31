"""
Strategy Module Root
Registers all strategies with Nautilus Trader and configures instrument IDs.
Central hub for strategy lifecycle management.
"""

from typing import Dict, List, Optional, Any, Type
from dataclasses import dataclass, field
import threading
import logging
import time

# Import strategies
from .alpha_strategy import AlphaStrategy, PositionConfig, SignalData
from .portfolio_manager import PortfolioManager, VaRLimits, PortfolioState

# Conditional Nautilus imports
try:
    from nautilus_trader.trader.trader import Trader
    from nautilus_trader.model.identifiers import TraderId, StrategyId
    NAUTILUS_AVAILABLE = True
except ImportError:
    NAUTILUS_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StrategyConfig:
    """Configuration for a registered strategy."""
    strategy_id: str
    strategy_type: str  # "alpha", "portfolio"
    instruments: List[str]
    position_config: Optional[PositionConfig] = None
    var_limits: Optional[VaRLimits] = None
    enabled: bool = True


@dataclass
class RegisteredStrategy:
    """Wrapper for a registered strategy instance."""
    config: StrategyConfig
    instance: Any
    created_at: float = field(default_factory=time.time)
    is_active: bool = False


class StrategyModule:
    """
    Central module for strategy registration and lifecycle management.
    Integrates with Nautilus Trader.
    """
    
    def __init__(self):
        self._strategies: Dict[str, RegisteredStrategy] = {}
        self._trader: Optional[Any] = None
        self._lock = threading.RLock()
        
        # Statistics
        self._total_signals_processed = 0
        self._total_orders_submitted = 0
    
    def register_strategy(self, config: StrategyConfig) -> bool:
        """
        Register a new strategy.
        
        Args:
            config: Strategy configuration
        
        Returns:
            True if successful
        """
        with self._lock:
            if config.strategy_id in self._strategies:
                logger.warning(f"Strategy {config.strategy_id} already registered")
                return False
            
            # Create strategy instance based on type
            if config.strategy_type == "alpha":
                position_config = config.position_config or PositionConfig()
                instance = AlphaStrategy(position_config)
            
            elif config.strategy_type == "portfolio":
                var_limits = config.var_limits or VaRLimits()
                instance = PortfolioManager(var_limits)
            
            else:
                logger.error(f"Unknown strategy type: {config.strategy_type}")
                return False
            
            # Register with Nautilus if available
            if NAUTILUS_AVAILABLE and self._trader is not None:
                try:
                    strategy_id = StrategyId(config.strategy_id)
                    self._trader.add_strategy(instance)
                    
                    # Subscribe to instruments
                    for inst in config.instruments:
                        # instance.subscribe_quote_ticks(InstrumentId.from_str(inst))
                        pass
                    
                    logger.info(f"Registered {config.strategy_id} with Nautilus")
                except Exception as e:
                    logger.error(f"Failed to register with Nautilus: {e}")
            
            # Store registration
            self._strategies[config.strategy_id] = RegisteredStrategy(
                config=config,
                instance=instance
            )
            
            logger.info(f"Registered strategy: {config.strategy_id}")
            return True
    
    def unregister_strategy(self, strategy_id: str) -> bool:
        """Unregister a strategy."""
        with self._lock:
            if strategy_id not in self._strategies:
                return False
            
            registered = self._strategies.pop(strategy_id)
            
            # Remove from Nautilus
            if NAUTILUS_AVAILABLE and self._trader is not None:
                try:
                    self._trader.remove_strategy(registered.instance)
                except Exception:
                    pass
            
            logger.info(f"Unregistered strategy: {strategy_id}")
            return True
    
    def get_strategy(self, strategy_id: str) -> Optional[Any]:
        """Get strategy instance by ID."""
        with self._lock:
            registered = self._strategies.get(strategy_id)
            return registered.instance if registered else None
    
    def get_alpha_strategy(self, strategy_id: str) -> Optional[AlphaStrategy]:
        """Get AlphaStrategy instance by ID."""
        instance = self.get_strategy(strategy_id)
        if isinstance(instance, AlphaStrategy):
            return instance
        return None
    
    def get_portfolio_manager(self, strategy_id: str) -> Optional[PortfolioManager]:
        """Get PortfolioManager instance by ID."""
        instance = self.get_strategy(strategy_id)
        if isinstance(instance, PortfolioManager):
            return instance
        return None
    
    def set_trader(self, trader: Any) -> None:
        """
        Set the Nautilus Trader instance.
        
        Args:
            trader: Nautilus Trader instance
        """
        with self._lock:
            self._trader = trader
            logger.info("Trader instance set")
    
    def start_all_strategies(self) -> None:
        """Start all registered strategies."""
        with self._lock:
            for strategy_id, registered in self._strategies.items():
                if registered.config.enabled:
                    try:
                        if hasattr(registered.instance, 'on_start'):
                            registered.instance.on_start()
                        registered.is_active = True
                        logger.info(f"Started strategy: {strategy_id}")
                    except Exception as e:
                        logger.error(f"Failed to start {strategy_id}: {e}")
    
    def stop_all_strategies(self) -> None:
        """Stop all registered strategies."""
        with self._lock:
            for strategy_id, registered in self._strategies.items():
                try:
                    if hasattr(registered.instance, 'on_stop'):
                        registered.instance.on_stop()
                    registered.is_active = False
                    logger.info(f"Stopped strategy: {strategy_id}")
                except Exception as e:
                    logger.error(f"Failed to stop {strategy_id}: {e}")
    
    def route_signal(self, signal: SignalData) -> None:
        """
        Route ML signal to appropriate strategy.
        
        Args:
            signal: SignalData from ML module
        """
        with self._lock:
            # Find alpha strategy for this instrument
            for strategy_id, registered in self._strategies.items():
                if registered.config.strategy_type != "alpha":
                    continue
                
                if signal.instrument_id in registered.config.instruments:
                    try:
                        if hasattr(registered.instance, 'on_signal'):
                            registered.instance.on_signal(signal)
                            self._total_signals_processed += 1
                    except Exception as e:
                        logger.error(f"Signal routing error: {e}")
    
    def check_risk_limits(self, 
                          instrument_id: str,
                          quantity: float,
                          price: float) -> bool:
        """
        Check risk limits across all portfolio managers.
        
        Args:
            instrument_id: Target instrument
            quantity: Proposed quantity
            price: Current price
        
        Returns:
            True if within limits
        """
        with self._lock:
            for registered in self._strategies.values():
                if registered.config.strategy_type != "portfolio":
                    continue
                
                pm = registered.instance
                if hasattr(pm, 'check_position_limit'):
                    if not pm.check_position_limit(instrument_id, quantity, price):
                        return False
            
            return True
    
    def get_all_statistics(self) -> Dict[str, Any]:
        """Get statistics from all strategies."""
        stats = {
            "total_strategies": len(self._strategies),
            "active_strategies": sum(1 for s in self._strategies.values() if s.is_active),
            "total_signals_processed": self._total_signals_processed,
            "total_orders_submitted": self._total_orders_submitted,
            "strategies": {}
        }
        
        with self._lock:
            for strategy_id, registered in self._strategies.items():
                if hasattr(registered.instance, 'get_statistics'):
                    stats["strategies"][strategy_id] = registered.instance.get_statistics()
        
        return stats
    
    def list_strategies(self) -> List[Dict[str, Any]]:
        """List all registered strategies."""
        with self._lock:
            return [
                {
                    "strategy_id": reg.config.strategy_id,
                    "type": reg.config.strategy_type,
                    "instruments": reg.config.instruments,
                    "enabled": reg.config.enabled,
                    "active": reg.is_active,
                    "created_at": reg.created_at,
                }
                for reg in self._strategies.values()
            ]


# Global module instance
_strategy_module: Optional[StrategyModule] = None
_module_lock = threading.Lock()


def get_strategy_module() -> StrategyModule:
    """Get or create global StrategyModule instance."""
    global _strategy_module
    
    with _module_lock:
        if _strategy_module is None:
            _strategy_module = StrategyModule()
        
        return _strategy_module


def reset_strategy_module() -> None:
    """Reset the global module."""
    global _strategy_module
    
    with _module_lock:
        if _strategy_module is not None:
            _strategy_module.stop_all_strategies()
            _strategy_module = None


def setup_default_strategies(trader: Optional[Any] = None) -> StrategyModule:
    """
    Setup default strategy configuration.
    
    Args:
        trader: Optional Nautilus Trader instance
    
    Returns:
        Configured StrategyModule
    """
    module = get_strategy_module()
    
    if trader is not None:
        module.set_trader(trader)
    
    # Register Alpha Strategy
    alpha_config = StrategyConfig(
        strategy_id="alpha_btc_eth",
        strategy_type="alpha",
        instruments=["BTC/USDT", "ETH/USDT"],
        position_config=PositionConfig(
            max_position_size=2.0,
            kelly_fraction=0.25,
            use_kelly=True
        ),
        enabled=True
    )
    module.register_strategy(alpha_config)
    
    # Register Portfolio Manager
    portfolio_config = StrategyConfig(
        strategy_id="portfolio_risk",
        strategy_type="portfolio",
        instruments=["BTC/USDT", "ETH/USDT", "SOL/USDT"],
        var_limits=VaRLimits(
            daily_var_limit=10000.0,
            concentration_limit=0.4
        ),
        enabled=True
    )
    module.register_strategy(portfolio_config)
    
    logger.info("Default strategies configured")
    return module


if __name__ == "__main__":
    print("Strategy Module Demo")
    print("=" * 40)
    
    # Setup default strategies
    module = setup_default_strategies()
    
    # List strategies
    strategies = module.list_strategies()
    print(f"\nRegistered Strategies:")
    for s in strategies:
        print(f"  - {s['strategy_id']} ({s['type']})")
        print(f"    Instruments: {s['instruments']}")
        print(f"    Enabled: {s['enabled']}, Active: {s['active']}")
    
    # Start strategies
    module.start_all_strategies()
    
    # Simulate signal
    signal = SignalData(
        signal_id="test_1",
        instrument_id="BTC/USDT",
        alpha_score=0.65,
        probability=0.72,
        confidence=0.85,
        timestamp=time.time(),
        model_ids=["model_v1"]
    )
    
    module.route_signal(signal)
    print(f"\nRouted signal to BTC/USDT strategy")
    
    # Get statistics
    stats = module.get_all_statistics()
    print(f"\nStatistics: {stats}")
    
    # Stop strategies
    module.stop_all_strategies()
    
    # Cleanup
    reset_strategy_module()
    print("\nStrategy Module demo complete")
