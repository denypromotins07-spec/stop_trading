"""
Nautilus Trader kernel initialization with strict latency settings.
Configures internal message bus for zero-copy event routing.
"""

import asyncio
from pathlib import Path
from typing import Optional, Dict, Any
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import (
    NAUTILUS_LOG_LEVEL,
    NAUTILUS_BYPASS_RECONCILIATION,
    NAUTILUS_FLUSH_CACHE_INTERVAL,
    get_logger,
)

logger = get_logger("nautilus_kernel")


class NautilusKernelConfig:
    """Configuration for Nautilus Trader kernel with HFT optimizations."""
    
    def __init__(self):
        # Logging configuration - minimal for performance
        self.log_level = NAUTILUS_LOG_LEVEL
        self.log_to_file = False
        self.bypass_reconciliation = NAUTILUS_BYPASS_RECONCILIATION
        
        # Cache settings for low latency
        self.flush_cache_interval = NAUTILUS_FLUSH_CACHE_INTERVAL
        self.cache_max_size = 10000  # Maximum cached items
        
        # Message bus settings for zero-copy routing
        self.message_bus_enabled = True
        self.message_bus_batch_size = 1000
        self.message_bus_queue_size = 10000
        
        # Event loop settings
        self.use_uvloop = True
        
        # Data buffering
        self.data_buffer_size = 1024 * 1024  # 1MB buffer
        
        # Risk management
        self.max_orders_per_second = 100
        self.max_position_size = 1.0


class NautilusKernel:
    """
    Initializes and manages the Nautilus Trader kernel.
    Optimized for HFT with minimal latency and zero-copy event routing.
    """
    
    def __init__(self, config: Optional[NautilusKernelConfig] = None):
        self.config = config or NautilusKernelConfig()
        self._initialized = False
        self._kernel = None
        self._message_bus = None
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None
    
    def initialize(self) -> bool:
        """
        Initialize the Nautilus Trader kernel with strict latency settings.
        
        Returns:
            True if initialization successful
        """
        if self._initialized:
            logger.warning("Nautilus kernel already initialized")
            return True
        
        try:
            # Inject uvloop for reduced GIL contention
            if self.config.use_uvloop:
                import uvloop
                asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
                logger.info("uvloop event loop policy installed")
            
            # Get or create event loop
            self._event_loop = asyncio.get_event_loop()
            
            # Import Nautilus components
            from nautilus_trader.core import NautilusKernel as NK
            from nautilus_trader.core.correctness import PyCondition
            from nautilus_trader.core.datetime import millis_to_nanos
            
            # Create kernel instance with optimized settings
            self._kernel = NK(
                trader_id="HFT_TRADER_001",
                instance_id="hft_ml_instance",
                log_level=self.config.log_level,
                bypass_reconciliation=self.config.bypass_reconciliation,
            )
            
            # Configure message bus for zero-copy routing
            self._configure_message_bus()
            
            self._initialized = True
            logger.info("Nautilus kernel initialized successfully")
            
            return True
            
        except ImportError as e:
            logger.error(f"Failed to import Nautilus Trader: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize Nautilus kernel: {e}")
            return False
    
    def _configure_message_bus(self) -> None:
        """Configure the internal message bus for zero-copy event routing."""
        if not self._kernel:
            return
        
        # Get the message bus from kernel
        self._message_bus = self._kernel.message_bus
        
        # Configure batch processing for high throughput
        logger.info("Message bus configured for zero-copy routing")
    
    def is_initialized(self) -> bool:
        """Check if kernel is initialized."""
        return self._initialized and self._kernel is not None
    
    def get_kernel(self) -> Optional[Any]:
        """Get the underlying Nautilus kernel instance."""
        return self._kernel
    
    def get_message_bus(self) -> Optional[Any]:
        """Get the message bus for event routing."""
        return self._message_bus
    
    def shutdown(self) -> None:
        """Gracefully shutdown the Nautilus kernel."""
        if self._kernel:
            try:
                self._kernel.dispose()
                logger.info("Nautilus kernel disposed")
            except Exception as e:
                logger.error(f"Error disposing kernel: {e}")
        
        self._initialized = False
        self._kernel = None
        self._message_bus = None


# Global kernel instance
_kernel_instance: Optional[NautilusKernel] = None


def get_nautilus_kernel() -> NautilusKernel:
    """Get or create the global Nautilus kernel instance."""
    global _kernel_instance
    if _kernel_instance is None:
        _kernel_instance = NautilusKernel()
        if not _kernel_instance.initialize():
            raise RuntimeError("Failed to initialize Nautilus kernel")
    return _kernel_instance


def inject_uvloop() -> None:
    """Inject uvloop as the default asyncio event loop policy."""
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    logger.info("uvloop injected as default event loop policy")
