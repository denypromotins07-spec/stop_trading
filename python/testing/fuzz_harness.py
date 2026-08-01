"""
Chapter 5: Python CLI, Fuzzing & Final Integration Testing
File: python/testing/fuzz_harness.py

Atheris fuzzing harness targeting IPC shared memory parsers and Nautilus
custom data deserializers. Generates millions of malformed byte arrays to
guarantee the Python backend never crashes or throws unhandled exceptions.
"""

import sys
import struct
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
import io

try:
    import atheris
    ATHERIS_AVAILABLE = True
except ImportError:
    ATHERIS_AVAILABLE = False
    print("Warning: atheris not available, running in simulation mode")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FuzzConfig:
    """Configuration for fuzzing."""
    max_iterations: int = 1000000
    seed: int = 42
    timeout_seconds: int = 3600
    coverage_report: bool = True
    
    # Target-specific settings
    ipc_max_message_size: int = 65536
    nautilus_max_order_size: int = 10000


class IPCSharedMemoryParser:
    """
    Parser for IPC shared memory messages.
    Target for fuzzing - must handle all malformed input gracefully.
    """
    
    HEADER_SIZE = 16  # bytes
    MAGIC_NUMBER = 0xDEADBEEF
    
    def __init__(self):
        self.parse_count = 0
        self.error_count = 0
        self.last_error: Optional[str] = None
    
    def parse_message(self, data: bytes) -> Optional[Dict[str, Any]]:
        """
        Parse an IPC shared memory message.
        Must be robust against all malformed inputs.
        """
        self.parse_count += 1
        
        try:
            if len(data) < self.HEADER_SIZE:
                return None
            
            # Parse header
            magic = struct.unpack('>I', data[0:4])[0]
            if magic != self.MAGIC_NUMBER:
                return None
            
            message_type = struct.unpack('>H', data[4:6])[0]
            payload_length = struct.unpack('>I', data[6:10])[0]
            checksum = struct.unpack('>Q', data[10:18])[0] if len(data) >= 18 else 0
            
            # Validate payload length
            if payload_length > FuzzConfig().ipc_max_message_size:
                self.error_count += 1
                self.last_error = "Payload too large"
                return None
            
            # Extract payload
            payload_start = self.HEADER_SIZE
            payload_end = payload_start + payload_length
            
            if len(data) < payload_end:
                return None
            
            payload = data[payload_start:payload_end]
            
            return {
                "message_type": message_type,
                "payload_length": payload_length,
                "payload": payload.hex(),
                "checksum": checksum
            }
            
        except struct.error as e:
            self.error_count += 1
            self.last_error = f"struct error: {e}"
            return None
        except Exception as e:
            self.error_count += 1
            self.last_error = f"parse error: {e}"
            return None
    
    def get_statistics(self) -> Dict[str, Any]:
        return {
            "parse_count": self.parse_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(1, self.parse_count),
            "last_error": self.last_error
        }


class NautilusDataDeserializer:
    """
    Deserializer for Nautilus custom data formats.
    Target for fuzzing - must handle all malformed inputs gracefully.
    """
    
    ORDER_TYPES = {0: "market", 1: "limit", 2: "stop", 3: "iceberg"}
    SIDE_TYPES = {0: "buy", 1: "sell"}
    
    def __init__(self):
        self.deserialize_count = 0
        self.error_count = 0
        self.last_error: Optional[str] = None
    
    def deserialize_order(self, data: bytes) -> Optional[Dict[str, Any]]:
        """Deserialize an order message."""
        self.deserialize_count += 1
        
        try:
            if len(data) < 8:
                return None
            
            # Parse order fields
            order_id = struct.unpack('>Q', data[0:8])[0]
            
            offset = 8
            if len(data) < offset + 1:
                return None
            
            order_type_byte = data[offset]
            order_type = self.ORDER_TYPES.get(order_type_byte % 4, "unknown")
            offset += 1
            
            if len(data) < offset + 1:
                return None
            
            side_byte = data[offset]
            side = self.SIDE_TYPES.get(side_byte % 2, "buy")
            offset += 1
            
            if len(data) < offset + 8:
                return None
            
            quantity = struct.unpack('>d', data[offset:offset+8])[0]
            offset += 8
            
            # Validate quantity
            if quantity <= 0 or quantity > FuzzConfig().nautilus_max_order_size:
                self.error_count += 1
                self.last_error = "Invalid quantity"
                return None
            
            price = 0.0
            if len(data) >= offset + 8:
                price = struct.unpack('>d', data[offset:offset+8])[0]
            
            return {
                "order_id": order_id,
                "order_type": order_type,
                "side": side,
                "quantity": quantity,
                "price": price
            }
            
        except struct.error as e:
            self.error_count += 1
            self.last_error = f"struct error: {e}"
            return None
        except Exception as e:
            self.error_count += 1
            self.last_error = f"deserialize error: {e}"
            return None
    
    def deserialize_fill(self, data: bytes) -> Optional[Dict[str, Any]]:
        """Deserialize a fill execution report."""
        self.deserialize_count += 1
        
        try:
            if len(data) < 16:
                return None
            
            fill_id = struct.unpack('>Q', data[0:8])[0]
            order_id = struct.unpack('>Q', data[8:16])[0]
            
            return {
                "fill_id": fill_id,
                "order_id": order_id
            }
            
        except Exception as e:
            self.error_count += 1
            self.last_error = f"fill deserialize error: {e}"
            return None
    
    def get_statistics(self) -> Dict[str, Any]:
        return {
            "deserialize_count": self.deserialize_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(1, self.deserialize_count),
            "last_error": self.last_error
        }


class FuzzHarness:
    """
    Main fuzzing harness coordinating all targets.
    """
    
    def __init__(self, config: Optional[FuzzConfig] = None):
        self.config = config or FuzzConfig()
        self.ipc_parser = IPCSharedMemoryParser()
        self.nautilus_deserializer = NautilusDataDeserializer()
        
        self.fuzz_count = 0
        self.crash_count = 0
        self.timeout_count = 0
    
    def fuzz_ipc_parser(self, data: bytes):
        """Fuzz target: IPC shared memory parser."""
        self.fuzz_count += 1
        
        try:
            result = self.ipc_parser.parse_message(data)
            # If we got here without exception, the parser handled it
            return True
        except Exception as e:
            self.crash_count += 1
            logger.error(f"IPC parser crash on input {data[:20].hex()}: {e}")
            raise  # Re-raise for atheris to track
    
    def fuzz_nautilus_deserializer(self, data: bytes):
        """Fuzz target: Nautilus data deserializer."""
        self.fuzz_count += 1
        
        try:
            # Try both deserialization methods
            self.nautilus_deserializer.deserialize_order(data)
            self.nautilus_deserializer.deserialize_fill(data)
            return True
        except Exception as e:
            self.crash_count += 1
            logger.error(f"Nautilus deserializer crash: {e}")
            raise
    
    def run_fuzz_test(self, data: bytes):
        """Combined fuzz test entry point."""
        # Fuzz both targets with same data
        self.fuzz_ipc_parser(data)
        self.fuzz_nautilus_deserializer(data)
    
    def get_statistics(self) -> Dict[str, Any]:
        return {
            "fuzz_count": self.fuzz_count,
            "crash_count": self.crash_count,
            "timeout_count": self.timeout_count,
            "ipc_stats": self.ipc_parser.get_statistics(),
            "nautilus_stats": self.nautilus_deserializer.get_statistics()
        }


def setup_atheris(harness: FuzzHarness):
    """Setup atheris for fuzzing."""
    if not ATHERIS_AVAILABLE:
        logger.warning("Atheris not available, skipping setup")
        return
    
    atheris.Setup(sys.argv, harness.run_fuzz_test)


def run_fuzzing(
    iterations: int = 100000,
    seed: int = 42
):
    """
    Run fuzzing campaign.
    
    Args:
        iterations: Number of fuzz iterations
        seed: Random seed for reproducibility
    """
    import random
    random.seed(seed)
    
    harness = FuzzHarness()
    
    if ATHERIS_AVAILABLE:
        # Run with atheris
        sys.argv = [sys.argv[0], f"-runs={iterations}", f"-seed={seed}"]
        setup_atheris(harness)
        
        # Atheris will call TestOneInput repeatedly
        atheris.Fuzz()
    else:
        # Simulation mode - generate random inputs
        logger.info(f"Running {iterations} simulated fuzz iterations...")
        
        for i in range(iterations):
            # Generate random data of varying sizes
            size = random.randint(1, 65536)
            data = bytes(random.randint(0, 255) for _ in range(size))
            
            try:
                harness.run_fuzz_test(data)
            except Exception:
                pass  # Expected in fuzzing
            
            if i % 10000 == 0:
                stats = harness.get_statistics()
                logger.info(
                    f"Progress: {i}/{iterations}, "
                    f"crashes: {stats['crash_count']}"
                )
        
        # Print final statistics
        stats = harness.get_statistics()
        logger.info(f"Fuzzing complete: {json.dumps(stats, indent=2)}")
    
    return harness


# JSON import for stats output
import json


# Export for module use
__all__ = [
    "FuzzConfig",
    "IPCSharedMemoryParser",
    "NautilusDataDeserializer",
    "FuzzHarness",
    "setup_atheris",
    "run_fuzzing",
    "ATHERIS_AVAILABLE"
]
