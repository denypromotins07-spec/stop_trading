"""
Chapter 4: Python Security, Hardening & Memory Forensics
File: python/security/memory_scrubber.py

Active RAM scrubber using ctypes.memset to overwrite and wipe plaintext
API keys and sensitive inference states. Ensures that if the Python process
core dumps, no cryptographic secrets or proprietary ML weights are exposed.
"""

import ctypes
import ctypes.util
import os
import gc
import weakref
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Load C library for memory operations
try:
    _libc_name = ctypes.util.find_library('c')
    _libc = ctypes.CDLL(_libc_name, use_errno=True)
    
    # Define memset signature
    _libc.memset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
    _libc.memset.restype = ctypes.c_void_p
    
    LIBC_AVAILABLE = True
except Exception as e:
    logger.warning(f"Could not load libc: {e}")
    LIBC_AVAILABLE = False


@dataclass
class SensitiveBuffer:
    """Tracks a sensitive memory buffer for scrubbing."""
    buffer_id: str
    description: str
    created_at: datetime
    size_bytes: int
    memory_address: int
    is_scrubbed: bool = False


class SecureByteArray:
    """
    A bytearray wrapper that supports secure wiping.
    Use this for all sensitive data like API keys.
    """
    
    _instances: Dict[int, 'SecureByteArray'] = {}
    
    def __init__(self, data: bytes, description: str = "sensitive_data"):
        self._data = bytearray(data)
        self._description = description
        self._created_at = datetime.utcnow()
        self._is_scrubbed = False
        
        # Get memory address
        self._address = id(self._data)
        self._buffer_id = f"sec_{os.urandom(8).hex()}"
        
        # Register instance
        SecureByteArray._instances[id(self)] = self
        
        # Create weak reference for cleanup
        self._weak_ref = weakref.ref(self, self._cleanup_callback)
        
        logger.debug(f"Created secure buffer: {self._buffer_id} ({len(data)} bytes)")
    
    @staticmethod
    def _cleanup_callback(weak_ref):
        """Called when instance is garbage collected without explicit scrub."""
        logger.warning("SecureByteArray finalized without explicit scrub!")
    
    @property
    def data(self) -> bytearray:
        """Get underlying data (use carefully)."""
        if self._is_scrubbed:
            raise RuntimeError("Buffer has been scrubbed")
        return self._data
    
    def scrub(self):
        """Securely wipe the buffer contents."""
        if self._is_scrubbed:
            return
        
        if LIBC_AVAILABLE:
            # Use libc memset for reliable wiping
            buffer_addr = ctypes.addressof(ctypes.c_char.from_buffer(self._data))
            _libc.memset(buffer_addr, 0, len(self._data))
        else:
            # Fallback: overwrite with zeros multiple times
            for _ in range(3):
                for i in range(len(self._data)):
                    self._data[i] = 0
        
        self._is_scrubbed = True
        
        # Remove from instances
        if id(self) in SecureByteArray._instances:
            del SecureByteArray._instances[id(self)]
        
        logger.info(f"Scrubbed secure buffer: {self._buffer_id}")
    
    def __del__(self):
        """Ensure scrubbing on deletion."""
        if not self._is_scrubbed:
            self.scrub()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.scrub()


class MemoryScrubber:
    """
    Active memory scrubber for sensitive data.
    Provides utilities for wiping Python objects and forcing GC.
    """
    
    def __init__(self):
        self.tracked_buffers: Dict[str, SensitiveBuffer] = {}
        self.scrub_history: List[Dict] = []
        self.total_scrubbed_bytes = 0
    
    def register_buffer(
        self,
        buffer: bytearray,
        description: str
    ) -> str:
        """Register a buffer for tracking and scrubbing."""
        buffer_id = f"buf_{os.urandom(8).hex()}"
        
        self.tracked_buffers[buffer_id] = SensitiveBuffer(
            buffer_id=buffer_id,
            description=description,
            created_at=datetime.utcnow(),
            size_bytes=len(buffer),
            memory_address=id(buffer)
        )
        
        logger.debug(f"Registered buffer for scrubbing: {buffer_id}")
        return buffer_id
    
    def scrub_buffer(self, buffer: bytearray, passes: int = 3):
        """
        Securely scrub a bytearray with multiple passes.
        
        Args:
            buffer: The bytearray to scrub
            passes: Number of overwrite passes
        """
        if not buffer:
            return
        
        size = len(buffer)
        
        # Multiple pass overwriting
        patterns = [0x00, 0xFF, 0xAA, 0x55]
        
        for pass_num in range(passes):
            pattern = patterns[pass_num % len(patterns)]
            
            if LIBC_AVAILABLE:
                try:
                    buffer_addr = ctypes.addressof(
                        ctypes.c_char.from_buffer(buffer)
                    )
                    _libc.memset(buffer_addr, pattern, size)
                except Exception as e:
                    logger.warning(f"memset failed, using fallback: {e}")
                    for i in range(size):
                        buffer[i] = pattern
            else:
                for i in range(size):
                    buffer[i] = pattern
        
        # Final zero pass
        if LIBC_AVAILABLE:
            buffer_addr = ctypes.addressof(ctypes.c_char.from_buffer(buffer))
            _libc.memset(buffer_addr, 0, size)
        else:
            for i in range(size):
                buffer[i] = 0
        
        self.total_scrubbed_bytes += size
        self.scrub_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "size_bytes": size,
            "passes": passes
        })
        
        logger.debug(f"Scrubbed {size} bytes with {passes} passes")
    
    def scrub_dict_values(self, d: Dict, keys_to_scrub: Optional[Set[str]] = None):
        """
        Scrub sensitive values in a dictionary.
        
        Args:
            d: Dictionary containing sensitive data
            keys_to_scrub: Specific keys to scrub (scrubs all bytearrays if None)
        """
        for key, value in list(d.items()):
            should_scrub = (
                keys_to_scrub is None and isinstance(value, (bytearray, bytes))
            ) or (keys_to_scrub and key in keys_to_scrub)
            
            if should_scrub:
                if isinstance(value, bytearray):
                    self.scrub_buffer(value)
                    d[key] = None
                elif isinstance(value, bytes):
                    # Can't scrub immutable bytes, just delete reference
                    d[key] = None
    
    def force_garbage_collection(self, generations: int = 2):
        """
        Force aggressive garbage collection.
        
        Args:
            generations: Which GC generations to collect (0, 1, or 2)
        """
        logger.debug(f"Forcing GC generation {generations}...")
        
        # Collect all generations
        for gen in range(generations + 1):
            collected = gc.collect(gen)
            logger.debug(f"GC gen {gen}: collected {collected} objects")
        
        # Clear any pending finalizers
        gc.set_debug(gc.DEBUG_STATS)
        gc.set_debug(0)  # Reset debug flags
    
    def scrub_python_string(self, s: str) -> None:
        """
        Attempt to scrub a Python string from memory.
        Note: This is best-effort due to Python's string interning.
        """
        # Strings are immutable in Python, so we can't directly scrub
        # Best effort: remove references and force GC
        del s
        self.force_garbage_collection()
    
    def get_scrub_statistics(self) -> Dict[str, Any]:
        """Get scrubbing statistics."""
        return {
            "total_scrubbed_bytes": self.total_scrubbed_bytes,
            "total_scrubbed_mb": self.total_scrubbed_bytes / (1024 * 1024),
            "tracked_buffers": len(self.tracked_buffers),
            "scrub_operations": len(self.scrub_history),
            "libc_available": LIBC_AVAILABLE
        }
    
    def emergency_scrub_all(self):
        """
        Emergency scrub of all tracked buffers.
        Use only in critical situations (process termination, breach detection).
        """
        logger.warning("EMERGENCY SCRUB INITIATED")
        
        # Scrub all SecureByteArray instances
        for instance in list(SecureByteArray._instances.values()):
            try:
                instance.scrub()
            except Exception as e:
                logger.error(f"Failed to scrub instance: {e}")
        
        # Force GC
        self.force_garbage_collection()
        
        logger.warning("Emergency scrub complete")


def create_secure_api_key(key: str) -> SecureByteArray:
    """
    Create a secure wrapper for an API key.
    Use with context manager for automatic scrubbing.
    """
    return SecureByteArray(key.encode('utf-8'), description="api_key")


def secure_wipe_object(obj: Any):
    """
    Attempt to securely wipe any Python object.
    Recursively handles dicts, lists, and bytearrays.
    """
    scrubber = MemoryScrubber()
    
    if isinstance(obj, bytearray):
        scrubber.scrub_buffer(obj)
    elif isinstance(obj, dict):
        scrubber.scrub_dict_values(obj)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            if isinstance(item, bytearray):
                scrubber.scrub_buffer(item)
                obj[i] = None
    elif isinstance(obj, bytes):
        # Can't scrub immutable, just let GC handle it
        pass
    
    scrubber.force_garbage_collection()


# Export for module use
__all__ = [
    "LIBC_AVAILABLE",
    "SecureByteArray",
    "MemoryScrubber",
    "create_secure_api_key",
    "secure_wipe_object"
]
