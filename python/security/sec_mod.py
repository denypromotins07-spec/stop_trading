"""
Chapter 4: Python Security, Hardening & Memory Forensics
File: python/security/sec_mod.py

Module root for security infrastructure.
Manages secure enclaves, IPC encryption keys, and automated memory wiping routines.
"""

import os
import hashlib
import secrets
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import threading
import json

# Import local modules
from .memory_scrubber import (
    LIBC_AVAILABLE,
    SecureByteArray,
    MemoryScrubber,
    create_secure_api_key,
    secure_wipe_object
)
from .sandbox_enforcer import (
    SandboxConfig,
    SandboxViolation,
    ExecutionSandbox,
    create_ray_worker_sandbox,
    sandboxed_execution
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EncryptionKey:
    """Represents an encryption key with lifecycle management."""
    key_id: str
    key_bytes: bytes
    created_at: datetime
    expires_at: Optional[datetime]
    usage_count: int = 0
    max_usage: int = 1000
    is_revoked: bool = False


@dataclass
class SecureEnclaveConfig:
    """Configuration for secure enclave."""
    key_rotation_interval_hours: int = 24
    max_keys_in_memory: int = 10
    enable_auto_scrub: bool = True
    scrub_interval_seconds: int = 60


class KeyManager:
    """
    Manages encryption keys for IPC communication.
    Implements automatic rotation and secure deletion.
    """
    
    def __init__(self, config: Optional[SecureEnclaveConfig] = None):
        self.config = config or SecureEnclaveConfig()
        self.keys: Dict[str, EncryptionKey] = {}
        self.active_key_id: Optional[str] = None
        self.scrubber = MemoryScrubber()
        self._lock = threading.RLock()
        
        # Start background key rotation
        self._rotation_thread: Optional[threading.Thread] = None
        self._stop_rotation = threading.Event()
    
    def generate_key(self, key_size: int = 32) -> EncryptionKey:
        """Generate a new encryption key."""
        key_id = f"key_{secrets.token_hex(8)}"
        key_bytes = secrets.token_bytes(key_size)
        
        now = datetime.utcnow()
        expires_at = now + timedelta(hours=self.config.key_rotation_interval_hours)
        
        key = EncryptionKey(
            key_id=key_id,
            key_bytes=key_bytes,
            created_at=now,
            expires_at=expires_at
        )
        
        with self._lock:
            self.keys[key_id] = key
            
            # Enforce max keys limit
            if len(self.keys) > self.config.max_keys_in_memory:
                self._evict_oldest_key()
            
            # Set as active if first key
            if self.active_key_id is None:
                self.active_key_id = key_id
        
        logger.info(f"Generated new encryption key: {key_id}")
        return key
    
    def _evict_oldest_key(self):
        """Evict the oldest non-active key."""
        if not self.keys:
            return
        
        # Find oldest non-active key
        oldest_id = None
        oldest_time = datetime.utcnow()
        
        for key_id, key in self.keys.items():
            if key_id != self.active_key_id and key.created_at < oldest_time:
                oldest_id = key_id
                oldest_time = key.created_at
        
        if oldest_id:
            # Securely wipe before deletion
            key = self.keys[oldest_id]
            self.scrubber.scrub_buffer(bytearray(key.key_bytes))
            del self.keys[oldest_id]
            logger.debug(f"Evicted key: {oldest_id}")
    
    def get_active_key(self) -> Optional[EncryptionKey]:
        """Get the currently active encryption key."""
        with self._lock:
            if self.active_key_id and self.active_key_id in self.keys:
                key = self.keys[self.active_key_id]
                
                # Check expiration
                if key.expires_at and datetime.utcnow() > key.expires_at:
                    self.rotate_key()
                    return self.get_active_key()
                
                # Check usage limit
                if key.usage_count >= key.max_usage:
                    self.rotate_key()
                    return self.get_active_key()
                
                key.usage_count += 1
                return key
            
            return None
    
    def rotate_key(self):
        """Rotate to a new encryption key."""
        with self._lock:
            old_key_id = self.active_key_id
            
            # Generate new key
            new_key = self.generate_key()
            self.active_key_id = new_key.key_id
            
            # Revoke old key
            if old_key_id and old_key_id in self.keys:
                self.keys[old_key_id].is_revoked = True
            
            logger.info(f"Rotated encryption key: {old_key_id} -> {new_key.key_id}")
    
    def revoke_key(self, key_id: str):
        """Revoke a specific key."""
        with self._lock:
            if key_id in self.keys:
                self.keys[key_id].is_revoked = True
                
                # Rotate if revoking active key
                if key_id == self.active_key_id:
                    self.active_key_id = None
                    self.rotate_key()
                
                logger.info(f"Revoked key: {key_id}")
    
    def encrypt_data(self, data: bytes) -> Optional[bytes]:
        """Encrypt data using active key (XOR cipher for demonstration)."""
        key = self.get_active_key()
        if not key:
            return None
        
        # Simple XOR encryption (use proper crypto in production)
        key_bytes = key.key_bytes
        encrypted = bytes(a ^ b for a, b in zip(data, (key_bytes * ((len(data) // len(key_bytes)) + 1))[:len(data)]))
        
        return encrypted
    
    def decrypt_data(self, encrypted: bytes, key_id: str) -> Optional[bytes]:
        """Decrypt data using specified key."""
        with self._lock:
            if key_id not in self.keys:
                return None
            
            key = self.keys[key_id]
            if key.is_revoked:
                logger.warning(f"Attempted decrypt with revoked key: {key_id}")
                return None
        
        # Simple XOR decryption
        key_bytes = key.key_bytes
        decrypted = bytes(a ^ b for a, b in zip(encrypted, (key_bytes * ((len(encrypted) // len(key_bytes)) + 1))[:len(encrypted)]))
        
        return decrypted
    
    def start_background_rotation(self):
        """Start background key rotation thread."""
        def rotation_loop():
            while not self._stop_rotation.is_set():
                # Wait for rotation interval
                self._stop_rotation.wait(self.config.key_rotation_interval_hours * 3600)
                
                if not self._stop_rotation.is_set():
                    self.rotate_key()
                    
                    # Scrub memory if enabled
                    if self.config.enable_auto_scrub:
                        self.scrubber.force_garbage_collection()
        
        self._rotation_thread = threading.Thread(target=rotation_loop, daemon=True)
        self._rotation_thread.start()
        logger.info("Background key rotation started")
    
    def stop_background_rotation(self):
        """Stop background key rotation."""
        self._stop_rotation.set()
        if self._rotation_thread:
            self._rotation_thread.join(timeout=5)
        logger.info("Background key rotation stopped")
    
    def shutdown(self):
        """Securely shutdown and wipe all keys."""
        self.stop_background_rotation()
        
        with self._lock:
            for key in self.keys.values():
                self.scrubber.scrub_buffer(bytearray(key.key_bytes))
            self.keys.clear()
            self.active_key_id = None
        
        logger.info("Key manager shutdown complete, all keys wiped")


class SecureEnclave:
    """
    Main secure enclave manager.
    Combines key management, memory scrubbing, and sandbox enforcement.
    """
    
    def __init__(self, config: Optional[SecureEnclaveConfig] = None):
        self.config = config or SecureEnclaveConfig()
        self.key_manager = KeyManager(self.config)
        self.scrubber = MemoryScrubber()
        self.worker_sandbox = create_ray_worker_sandbox()
        
        self.is_initialized = False
        self.secure_contexts: Dict[str, Dict] = {}
    
    def initialize(self):
        """Initialize the secure enclave."""
        self.key_manager.generate_key()  # Create initial key
        self.key_manager.start_background_rotation()
        self.is_initialized = True
        logger.info("Secure enclave initialized")
    
    def create_secure_context(
        self,
        context_id: str,
        sensitive_data: Optional[Dict] = None
    ) -> str:
        """Create a secure execution context."""
        context = {
            "context_id": context_id,
            "created_at": datetime.utcnow(),
            "sensitive_data": {},
            "sandbox_active": False
        }
        
        if sensitive_data:
            for key, value in sensitive_data.items():
                if isinstance(value, str):
                    # Store as secure bytearray
                    context["sensitive_data"][key] = create_secure_api_key(value)
                else:
                    context["sensitive_data"][key] = value
        
        self.secure_contexts[context_id] = context
        logger.info(f"Created secure context: {context_id}")
        return context_id
    
    def execute_in_sandbox(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute a function within the sandbox."""
        if not self.is_initialized:
            raise RuntimeError("Secure enclave not initialized")
        
        with self.worker_sandbox:
            try:
                return func(*args, **kwargs)
            except SandboxViolation as e:
                logger.error(f"Sandbox violation: {e}")
                raise
    
    def wipe_context(self, context_id: str):
        """Securely wipe a context and all its data."""
        if context_id not in self.secure_contexts:
            return
        
        context = self.secure_contexts[context_id]
        
        # Wipe all sensitive data
        for key, value in context.get("sensitive_data", {}).items():
            if isinstance(value, SecureByteArray):
                value.scrub()
            else:
                secure_wipe_object(value)
        
        del self.secure_contexts[context_id]
        self.scrubber.force_garbage_collection()
        
        logger.info(f"Wiped secure context: {context_id}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get enclave statistics."""
        return {
            "is_initialized": self.is_initialized,
            "active_keys": len(self.key_manager.keys),
            "active_contexts": len(self.secure_contexts),
            "libc_available": LIBC_AVAILABLE,
            "scrub_stats": self.scrubber.get_scrub_statistics(),
            "sandbox_stats": self.worker_sandbox.get_statistics()
        }
    
    def emergency_lockdown(self):
        """Emergency lockdown - wipe everything immediately."""
        logger.critical("EMERGENCY LOCKDOWN INITIATED")
        
        # Wipe all contexts
        for context_id in list(self.secure_contexts.keys()):
            self.wipe_context(context_id)
        
        # Shutdown key manager (wipes all keys)
        self.key_manager.shutdown()
        
        # Emergency memory scrub
        self.scrubber.emergency_scrub_all()
        
        logger.critical("Emergency lockdown complete")
    
    def shutdown(self):
        """Graceful shutdown with secure cleanup."""
        logger.info("Initiating secure shutdown...")
        
        # Wipe all contexts
        for context_id in list(self.secure_contexts.keys()):
            self.wipe_context(context_id)
        
        # Shutdown key manager
        self.key_manager.shutdown()
        
        self.is_initialized = False
        logger.info("Secure enclave shutdown complete")


# Module-level singleton
_enclave: Optional[SecureEnclave] = None


def get_enclave() -> SecureEnclave:
    """Get or create the module-level enclave singleton."""
    global _enclave
    if _enclave is None:
        _enclave = SecureEnclave()
        _enclave.initialize()
    return _enclave


def initialize_security():
    """Initialize the security module."""
    get_enclave()
    logger.info("Security module initialized")


def shutdown_security():
    """Shutdown the security module."""
    global _enclave
    if _enclave:
        _enclave.shutdown()
        _enclave = None


# Export for module use
__all__ = [
    "EncryptionKey",
    "SecureEnclaveConfig",
    "KeyManager",
    "SecureEnclave",
    "get_enclave",
    "initialize_security",
    "shutdown_security"
]
