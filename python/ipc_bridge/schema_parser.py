"""
Schema parser for Rust #[repr(C)] memory layouts using ctypes.
Ensures byte-level alignment without serialization overhead.
"""

import ctypes
from pathlib import Path
from typing import Optional, Dict, Any, List
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import get_logger

logger = get_logger("schema_parser")


# ============================================================================
# Rust Memory Layout Definitions (matching Stage 30 Rust structs)
# These must exactly match the #[repr(C)] layouts from Rust
# ============================================================================

class RustFeatureVector(ctypes.Structure):
    """
    Matches Rust FeatureVector struct from Stage 30.
    Layout: #[repr(C)] struct with fixed-size arrays.
    """
    _pack_ = 1  # Ensure no padding
    _fields_ = [
        ("timestamp", ctypes.c_int64),      # i64 - Unix timestamp in nanoseconds
        ("symbol_id", ctypes.c_uint32),     # u32 - Symbol identifier
        ("feature_count", ctypes.c_uint32), # u32 - Number of features
        ("features", ctypes.c_double * 64), # [f64; 64] - Fixed feature array
        ("confidence", ctypes.c_float),     # f32 - Confidence score
        ("padding", ctypes.c_uint8 * 4),    # Alignment padding to 8 bytes
    ]
    
    @classmethod
    def get_dtype(cls) -> np.dtype:
        """Get numpy dtype matching this structure."""
        return np.dtype([
            ('timestamp', np.int64),
            ('symbol_id', np.uint32),
            ('feature_count', np.uint32),
            ('features', np.float64, 64),
            ('confidence', np.float32),
            ('padding', np.uint8, 4),
        ], align=True)
    
    @classmethod
    def size_bytes(cls) -> int:
        """Get size of structure in bytes."""
        return ctypes.sizeof(cls)


class RustSignalMessage(ctypes.Structure):
    """
    Matches Rust SignalMessage struct for execution signals.
    Layout: #[repr(C)] struct for signal transmission.
    """
    _pack_ = 1
    _fields_ = [
        ("signal_id", ctypes.c_uint64),     # u64 - Unique signal identifier
        ("timestamp", ctypes.c_int64),      # i64 - Timestamp in nanoseconds
        ("signal_type", ctypes.c_uint8),    # u8 - Signal type enum (0=BUY, 1=SELL, 2=HOLD)
        ("strength", ctypes.c_float),       # f32 - Signal strength (0.0-1.0)
        ("symbol_id", ctypes.c_uint32),     # u32 - Target symbol
        ("quantity", ctypes.c_double),      # f64 - Order quantity
        ("stop_loss", ctypes.c_double),     # f64 - Stop loss price
        ("take_profit", ctypes.c_double),   # f64 - Take profit price
        ("flags", ctypes.c_uint32),         # u32 - Execution flags
        ("reserved", ctypes.c_uint8 * 16),  # Reserved for future use
    ]
    
    @classmethod
    def get_dtype(cls) -> np.dtype:
        """Get numpy dtype matching this structure."""
        return np.dtype([
            ('signal_id', np.uint64),
            ('timestamp', np.int64),
            ('signal_type', np.uint8),
            ('strength', np.float32),
            ('symbol_id', np.uint32),
            ('quantity', np.float64),
            ('stop_loss', np.float64),
            ('take_profit', np.float64),
            ('flags', np.uint32),
            ('reserved', np.uint8, 16),
        ], align=True)
    
    @classmethod
    def size_bytes(cls) -> int:
        """Get size of structure in bytes."""
        return ctypes.sizeof(cls)


class RustMarketTick(ctypes.Structure):
    """
    Matches Rust MarketTick struct for market data.
    Layout: #[repr(C)] struct for tick data.
    """
    _pack_ = 1
    _fields_ = [
        ("timestamp", ctypes.c_int64),      # i64 - Tick timestamp
        ("symbol_id", ctypes.c_uint32),     # u32 - Symbol identifier
        ("bid_price", ctypes.c_double),     # f64 - Best bid price
        ("ask_price", ctypes.c_double),     # f64 - Best ask price
        ("bid_size", ctypes.c_double),      # f64 - Bid quantity
        ("ask_size", ctypes.c_double),      # f64 - Ask quantity
        ("last_price", ctypes.c_double),    # f64 - Last traded price
        ("volume", ctypes.c_uint64),        # u64 - Cumulative volume
        ("sequence", ctypes.c_uint64),      # u64 - Sequence number
    ]
    
    @classmethod
    def get_dtype(cls) -> np.dtype:
        """Get numpy dtype matching this structure."""
        return np.dtype([
            ('timestamp', np.int64),
            ('symbol_id', np.uint32),
            ('bid_price', np.float64),
            ('ask_price', np.float64),
            ('bid_size', np.float64),
            ('ask_size', np.float64),
            ('last_price', np.float64),
            ('volume', np.uint64),
            ('sequence', np.uint64),
        ], align=True)
    
    @classmethod
    def size_bytes(cls) -> int:
        """Get size of structure in bytes."""
        return ctypes.sizeof(cls)


class RustStateSync(ctypes.Structure):
    """
    Matches Rust StateSync struct for state synchronization.
    Layout: #[repr(C)] struct for ML model state.
    """
    _pack_ = 1
    _fields_ = [
        ("state_id", ctypes.c_uint64),      # u64 - State identifier
        ("timestamp", ctypes.c_int64),      # i64 - Sync timestamp
        ("regime", ctypes.c_uint8),         # u8 - Market regime (0-7)
        ("volatility", ctypes.c_float),     # f32 - Current volatility
        ("trend", ctypes.c_float),          # f32 - Trend indicator (-1 to 1)
        ("momentum", ctypes.c_float),       # f32 - Momentum indicator
        ("model_version", ctypes.c_uint32), # u32 - Model version
        ("weights_hash", ctypes.c_uint64),  # u64 - Hash of current weights
        ("active_features", ctypes.c_uint32), # u32 - Number of active features
        ("reserved", ctypes.c_uint8 * 28),  # Reserved
    ]
    
    @classmethod
    def get_dtype(cls) -> np.dtype:
        """Get numpy dtype matching this structure."""
        return np.dtype([
            ('state_id', np.uint64),
            ('timestamp', np.int64),
            ('regime', np.uint8),
            ('volatility', np.float32),
            ('trend', np.float32),
            ('momentum', np.float32),
            ('model_version', np.uint32),
            ('weights_hash', np.uint64),
            ('active_features', np.uint32),
            ('reserved', np.uint8, 28),
        ], align=True)
    
    @classmethod
    def size_bytes(cls) -> int:
        """Get size of structure in bytes."""
        return ctypes.sizeof(cls)


# ============================================================================
# Schema Registry and Parser
# ============================================================================

class SchemaRegistry:
    """
    Registry for all Rust memory layouts.
    Provides validation and conversion utilities.
    """
    
    SCHEMAS: Dict[str, type] = {
        "FeatureVector": RustFeatureVector,
        "SignalMessage": RustSignalMessage,
        "MarketTick": RustMarketTick,
        "StateSync": RustStateSync,
    }
    
    @classmethod
    def get_schema(cls, name: str) -> Optional[type]:
        """Get a schema by name."""
        return cls.SCHEMAS.get(name)
    
    @classmethod
    def get_all_dtypes(cls) -> Dict[str, np.dtype]:
        """Get all schemas as numpy dtypes."""
        return {name: schema.get_dtype() for name, schema in cls.SCHEMAS.items()}
    
    @classmethod
    def validate_alignment(cls) -> bool:
        """Validate that all structures have correct alignment."""
        all_valid = True
        for name, schema in cls.SCHEMAS.items():
            expected_size = schema.size_bytes()
            dtype_size = schema.get_dtype().itemsize
            
            if expected_size != dtype_size:
                logger.error(
                    f"Alignment mismatch for {name}: "
                    f"ctypes={expected_size}, numpy={dtype_size}"
                )
                all_valid = False
            else:
                logger.debug(f"{name} alignment validated ({expected_size} bytes)")
        
        return all_valid
    
    @classmethod
    def parse_buffer(
        cls,
        buffer: bytes,
        schema_name: str,
        count: int = 1,
    ) -> Optional[np.ndarray]:
        """
        Parse a byte buffer using the specified schema.
        
        Args:
            buffer: Raw byte buffer
            schema_name: Name of schema to use
            count: Number of records
        
        Returns:
            NumPy structured array or None
        """
        schema = cls.get_schema(schema_name)
        if not schema:
            logger.error(f"Unknown schema: {schema_name}")
            return None
        
        dtype = schema.get_dtype()
        expected_size = dtype.itemsize * count
        
        if len(buffer) < expected_size:
            logger.error(
                f"Buffer too small: got {len(buffer)} bytes, "
                f"need {expected_size} bytes for {count} records"
            )
            return None
        
        # Zero-copy parsing using numpy.frombuffer
        arr = np.frombuffer(buffer[:expected_size], dtype=dtype, count=count)
        return arr


def validate_rust_schemas() -> bool:
    """
    Validate all Rust schema alignments.
    Should be called at startup to ensure compatibility.
    
    Returns:
        True if all schemas are valid
    """
    registry = SchemaRegistry()
    return registry.validate_alignment()


def get_schema_info() -> Dict[str, Dict[str, Any]]:
    """
    Get detailed information about all registered schemas.
    
    Returns:
        Dictionary of schema information
    """
    info = {}
    for name, schema in SchemaRegistry.SCHEMAS.items():
        info[name] = {
            "size_bytes": schema.size_bytes(),
            "dtype": schema.get_dtype(),
            "fields": [(f[0], f[1]) for f in schema._fields_],
        }
    return info
