"""
Rust Handshake Validator - Validates IPC shared memory handshake with Rust orchestrator.
Ensures Python correctly reads Rust "READY" flag before unlocking strategies.
Thread-safe implementation with timeout handling.
"""
import numpy as np
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import time
import threading
import struct
import mmap

logger = logging.getLogger(__name__)


class SharedMemoryIPC:
    """
    Shared memory IPC for Rust-Python communication.
    Uses memory-mapped files for zero-copy data exchange.
    """
    
    # Protocol constants
    MAGIC_NUMBER = 0x52555348  # "RUSH" in ASCII
    READY_FLAG_OFFSET = 0
    STATUS_OFFSET = 4
    DATA_OFFSET = 8
    
    def __init__(self, name: str = 'hft_ipc_shm', size: int = 4096):
        """
        Initialize shared memory IPC.
        
        Args:
            name: Shared memory segment name
            size: Size in bytes
        """
        self.name = name
        self.size = size
        
        self._mmap: Optional[mmap.mmap] = None
        self._connected = False
        self._lock = threading.Lock()
        
        # State
        self._rust_ready = False
        self._last_heartbeat = 0.0
        self._sequence_number = 0
        
        logger.info(f"SharedMemoryIPC initialized: {name} ({size} bytes)")
    
    def create(self) -> bool:
        """Create and initialize shared memory segment."""
        try:
            # Create file-backed mmap (for cross-process sharing)
            shm_path = Path(f'/tmp/{self.name}')
            
            # Create/truncate file
            with open(shm_path, 'wb') as f:
                f.write(b'\x00' * self.size)
            
            # Open for read/write
            self._fd = open(shm_path, 'r+b')
            self._mmap = mmap.mmap(self._fd.fileno(), self.size, 
                                   access=mmap.ACCESS_WRITE)
            
            # Write magic number
            self._mmap[:4] = struct.pack('I', self.MAGIC_NUMBER)
            self._mmap.flush()
            
            self._connected = True
            logger.info(f"Created shared memory: {self.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create shared memory: {e}")
            return False
    
    def attach(self) -> bool:
        """Attach to existing shared memory segment."""
        try:
            shm_path = Path(f'/tmp/{self.name}')
            
            if not shm_path.exists():
                logger.warning(f"Shared memory not found: {shm_path}")
                return False
            
            self._fd = open(shm_path, 'r+b')
            self._mmap = mmap.mmap(self._fd.fileno(), self.size,
                                   access=mmap.ACCESS_WRITE)
            
            # Verify magic number
            magic = struct.unpack('I', self._mmap[:4])[0]
            if magic != self.MAGIC_NUMBER:
                logger.error(f"Invalid magic number: {magic:#x}")
                self._mmap.close()
                return False
            
            self._connected = True
            logger.info(f"Attached to shared memory: {self.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to attach shared memory: {e}")
            return False
    
    def write_ready_flag(self, ready: bool) -> bool:
        """
        Write READY flag to shared memory.
        
        Args:
            ready: Ready status
            
        Returns:
            Success status
        """
        if not self._connected or not self._mmap:
            return False
        
        with self._lock:
            try:
                # Write ready flag (1 byte)
                self._mmap[self.READY_FLAG_OFFSET] = 1 if ready else 0
                
                # Update sequence number
                self._sequence_number = (self._sequence_number + 1) & 0xFFFFFFFF
                self._mmap[self.STATUS_OFFSET:self.STATUS_OFFSET + 4] = \
                    struct.pack('I', self._sequence_number)
                
                self._mmap.flush()
                return True
                
            except Exception as e:
                logger.error(f"Failed to write ready flag: {e}")
                return False
    
    def read_ready_flag(self) -> bool:
        """
        Read READY flag from shared memory.
        
        Returns:
            Ready status
        """
        if not self._connected or not self._mmap:
            return False
        
        try:
            flag = self._mmap[self.READY_FLAG_OFFSET]
            self._rust_ready = (flag == 1)
            
            if self._rust_ready:
                self._last_heartbeat = time.time()
            
            return self._rust_ready
            
        except Exception as e:
            logger.error(f"Failed to read ready flag: {e}")
            return False
    
    def write_data(self, data: bytes, offset: int = DATA_OFFSET) -> bool:
        """Write raw data to shared memory."""
        if not self._connected or not self._mmap:
            return False
        
        if offset + len(data) > self.size:
            logger.error(f"Data too large for shared memory")
            return False
        
        with self._lock:
            try:
                self._mmap[offset:offset + len(data)] = data
                self._mmap.flush()
                return True
            except Exception as e:
                logger.error(f"Failed to write data: {e}")
                return False
    
    def read_data(self, length: int, offset: int = DATA_OFFSET) -> Optional[bytes]:
        """Read raw data from shared memory."""
        if not self._connected or not self._mmap:
            return None
        
        if offset + length > self.size:
            logger.error(f"Read exceeds shared memory size")
            return None
        
        try:
            return bytes(self._mmap[offset:offset + length])
        except Exception as e:
            logger.error(f"Failed to read data: {e}")
            return None
    
    def get_heartbeat_age(self) -> float:
        """Get age of last heartbeat in seconds."""
        if self._last_heartbeat == 0:
            return float('inf')
        return time.time() - self._last_heartbeat
    
    def is_healthy(self, max_heartbeat_age: float = 5.0) -> bool:
        """Check if connection is healthy."""
        if not self._connected:
            return False
        
        return self.get_heartbeat_age() < max_heartbeat_age
    
    def close(self) -> None:
        """Close shared memory connection."""
        if self._mmap:
            self._mmap.flush()
            self._mmap.close()
            self._mmap = None
        
        if hasattr(self, '_fd') and self._fd:
            self._fd.close()
        
        self._connected = False
        logger.info("SharedMemoryIPC closed")


class RustHandshakeValidator:
    """
    Validates handshake with Rust orchestrator.
    Ensures proper synchronization before strategy activation.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize shared memory
        self.ipc = SharedMemoryIPC(
            name=self.config.get('shm_name', 'hft_ipc_shm'),
            size=self.config.get('shm_size', 4096)
        )
        
        # Handshake state
        self._handshake_complete = False
        self._python_ready = False
        self._rust_ready = False
        
        # Timeout configuration
        self._handshake_timeout = self.config.get('handshake_timeout', 30.0)
        self._heartbeat_interval = self.config.get('heartbeat_interval', 1.0)
        
        # Strategy lock
        self._strategies_unlocked = False
        
        logger.info("RustHandshakeValidator initialized")
    
    def initiate_handshake(self) -> bool:
        """
        Initiate handshake with Rust side.
        
        Returns:
            Success status
        """
        # Create/attach to shared memory
        if not self.ipc.create():
            if not self.ipc.attach():
                logger.error("Failed to initialize shared memory")
                return False
        
        # Signal Python readiness
        self._python_ready = True
        self.ipc.write_ready_flag(True)
        
        logger.info("Python signaled READY, waiting for Rust...")
        
        # Wait for Rust readiness
        start_time = time.time()
        
        while time.time() - start_time < self._handshake_timeout:
            if self.ipc.read_ready_flag():
                self._rust_ready = True
                self._handshake_complete = True
                logger.info("Rust handshake completed successfully")
                return True
            
            time.sleep(0.1)
        
        logger.error(f"Handshake timeout after {self._handshake_timeout}s")
        return False
    
    def validate_handshake(self) -> bool:
        """Validate that handshake is complete and healthy."""
        if not self._handshake_complete:
            return False
        
        # Check Rust is still ready
        if not self.ipc.read_ready_flag():
            logger.warning("Rust no longer signals READY")
            self._rust_ready = False
            return False
        
        # Check heartbeat
        if self.ipc.get_heartbeat_age() > self._handshake_timeout:
            logger.warning("Rust heartbeat stale")
            return False
        
        return True
    
    def unlock_strategies(self) -> bool:
        """
        Unlock trading strategies after successful handshake.
        
        Returns:
            Success status
        """
        if not self.validate_handshake():
            logger.error("Cannot unlock strategies: handshake not valid")
            return False
        
        self._strategies_unlocked = True
        logger.info("Strategies UNLOCKED - trading enabled")
        return True
    
    def lock_strategies(self) -> None:
        """Lock trading strategies (emergency stop)."""
        self._strategies_unlocked = False
        logger.warning("Strategies LOCKED - trading disabled")
    
    def are_strategies_unlocked(self) -> bool:
        """Check if strategies are unlocked for trading."""
        return self._strategies_unlocked and self.validate_handshake()
    
    def send_heartbeat(self) -> bool:
        """Send heartbeat to Rust side."""
        if not self._connected:
            return False
        
        # Toggle ready flag as heartbeat
        return self.ipc.write_ready_flag(self._python_ready)
    
    def get_status(self) -> Dict[str, Any]:
        """Get handshake status."""
        return {
            'handshake_complete': self._handshake_complete,
            'python_ready': self._python_ready,
            'rust_ready': self._rust_ready,
            'strategies_unlocked': self._strategies_unlocked,
            'ipc_connected': self.ipc._connected,
            'heartbeat_age_seconds': self.ipc.get_heartbeat_age(),
            'healthy': self.ipc.is_healthy()
        }
    
    def close(self) -> None:
        """Clean up handshake resources."""
        self.lock_strategies()
        self._python_ready = False
        self.ipc.write_ready_flag(False)
        self.ipc.close()
        logger.info("RustHandshakeValidator closed")


# Singleton instance
_rust_handshake: Optional[RustHandshakeValidator] = None


def get_rust_handshake(config: Optional[Dict[str, Any]] = None) -> RustHandshakeValidator:
    """Get or create singleton RustHandshakeValidator instance."""
    global _rust_handshake
    if _rust_handshake is None:
        _rust_handshake = RustHandshakeValidator(config)
    return _rust_handshake


def reset_rust_handshake() -> None:
    """Reset singleton instance."""
    global _rust_handshake
    if _rust_handshake is not None:
        _rust_handshake.close()
    _rust_handshake = None


__all__ = [
    'SharedMemoryIPC',
    'RustHandshakeValidator',
    'get_rust_handshake',
    'reset_rust_handshake'
]
