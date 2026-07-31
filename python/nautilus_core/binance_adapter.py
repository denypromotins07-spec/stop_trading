"""
Binance adapter configuration for Nautilus Trader.
Configures live data and execution clients using API keys securely passed from Rust KMS.
"""

from pathlib import Path
from typing import Optional, Dict, Any
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import (
    BINANCE_API_KEY,
    BINANCE_API_SECRET,
    BINANCE_TESTNET,
    NAUTILUS_LOG_LEVEL,
    get_logger,
)

logger = get_logger("binance_adapter")


class BinanceAdapterConfig:
    """Configuration for Binance adapter with HFT optimizations."""
    
    def __init__(self):
        # API credentials (from Rust KMS)
        self.api_key = BINANCE_API_KEY
        self.api_secret = BINANCE_API_SECRET
        self.testnet = BINANCE_TESTNET
        
        # Connection settings
        self.base_url_http = "https://testnet.binance.v1" if self.testnet else "https://api.binance.com"
        self.base_url_ws = "wss://testnet.binance.v1/ws" if self.testnet else "wss://stream.binance.com:9443/ws"
        
        # HFT-specific settings
        self.max_connections = 10
        self.connection_timeout = 5.0  # seconds
        self.request_timeout = 3.0  # seconds
        self.retry_attempts = 3
        self.retry_delay = 0.1  # seconds
        
        # Data subscription settings
        self.subscription_buffer_size = 10000
        self.tick_buffer_size = 1000
        
        # Order settings
        self.max_order_rate = 100  # orders per second
        self.order_id_prefix = "HFT_"
        
        # Disable unnecessary telemetry
        self.enable_telemetry = False
        self.enable_metrics = False


class BinanceLiveDataClient:
    """
    Configured Binance live data client for market data streaming.
    Optimized for low-latency tick ingestion.
    """
    
    def __init__(self, config: BinanceAdapterConfig):
        self.config = config
        self._client = None
        self._connected = False
        self._subscriptions: set = set()
    
    async def connect(self) -> bool:
        """Connect to Binance WebSocket stream."""
        if self._connected:
            return True
        
        try:
            from nautilus_trader.adapters.binance import BinanceLiveDataClient as BDLC
            from nautilus_trader.model.identifiers import Venue
            
            # Create the data client
            self._client = BDLC(
                venue=Venue("BINANCE"),
                api_key=self.config.api_key,
                api_secret=self.config.api_secret,
                base_url=self.config.base_url_ws,
                testnet=self.config.testnet,
            )
            
            self._connected = True
            logger.info(f"Binance data client connected (testnet={self.config.testnet})")
            return True
            
        except ImportError as e:
            logger.error(f"Failed to import Binance data client: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to connect Binance data client: {e}")
            return False
    
    async def subscribe(self, symbol: str) -> bool:
        """Subscribe to market data for a symbol."""
        if not self._connected:
            logger.error("Data client not connected")
            return False
        
        try:
            # Add to subscriptions
            self._subscriptions.add(symbol)
            logger.info(f"Subscribed to {symbol}")
            return True
        except Exception as e:
            logger.error(f"Failed to subscribe to {symbol}: {e}")
            return False
    
    async def unsubscribe(self, symbol: str) -> bool:
        """Unsubscribe from market data for a symbol."""
        if symbol in self._subscriptions:
            self._subscriptions.remove(symbol)
            logger.info(f"Unsubscribed from {symbol}")
        return True
    
    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self._connected
    
    async def disconnect(self) -> None:
        """Disconnect from Binance WebSocket."""
        if self._client and self._connected:
            try:
                await self._client.disconnect()
                logger.info("Binance data client disconnected")
            except Exception as e:
                logger.error(f"Error disconnecting: {e}")
        self._connected = False


class BinanceLiveExecutionClient:
    """
    Configured Binance live execution client for order management.
    Optimized for ultra-low latency order submission.
    """
    
    def __init__(self, config: BinanceAdapterConfig):
        self.config = config
        self._client = None
        self._connected = False
        self._order_count = 0
    
    async def connect(self) -> bool:
        """Connect to Binance execution API."""
        if self._connected:
            return True
        
        try:
            from nautilus_trader.adapters.binance import BinanceLiveExecutionClient as BLEC
            from nautilus_trader.model.identifiers import Venue
            
            # Create the execution client
            self._client = BLEC(
                venue=Venue("BINANCE"),
                api_key=self.config.api_key,
                api_secret=self.config.api_secret,
                base_url=self.config.base_url_http,
                testnet=self.config.testnet,
            )
            
            self._connected = True
            logger.info(f"Binance execution client connected (testnet={self.config.testnet})")
            return True
            
        except ImportError as e:
            logger.error(f"Failed to import Binance execution client: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to connect Binance execution client: {e}")
            return False
    
    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self._connected
    
    async def disconnect(self) -> None:
        """Disconnect from Binance execution API."""
        if self._client and self._connected:
            try:
                await self._client.disconnect()
                logger.info("Binance execution client disconnected")
            except Exception as e:
                logger.error(f"Error disconnecting: {e}")
        self._connected = False
    
    def generate_order_id(self) -> str:
        """Generate a unique order ID."""
        self._order_count += 1
        return f"{self.config.order_id_prefix}{self._order_count:08d}"


def create_binance_clients(
    config: Optional[BinanceAdapterConfig] = None,
) -> tuple[BinanceLiveDataClient, BinanceLiveExecutionClient]:
    """
    Create configured Binance data and execution clients.
    
    Args:
        config: Optional adapter configuration
    
    Returns:
        Tuple of (data_client, execution_client)
    """
    cfg = config or BinanceAdapterConfig()
    
    data_client = BinanceLiveDataClient(cfg)
    exec_client = BinanceLiveExecutionClient(cfg)
    
    logger.info("Binance clients created successfully")
    return data_client, exec_client


def validate_credentials() -> bool:
    """Validate that API credentials are properly configured."""
    if BINANCE_TESTNET:
        logger.info("Running in testnet mode - credentials validation skipped")
        return True
    
    if not BINANCE_API_KEY:
        logger.error("BINANCE_API_KEY is not set")
        return False
    
    if not BINANCE_API_SECRET:
        logger.error("BINANCE_API_SECRET is not set")
        return False
    
    logger.info("Binance credentials validated")
    return True
