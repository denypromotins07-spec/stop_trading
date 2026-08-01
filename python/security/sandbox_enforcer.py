"""
Chapter 4: Python Security, Hardening & Memory Forensics
File: python/security/sandbox_enforcer.py

Python-level execution sandbox restricting file system access and network calls
for untrusted Ray worker payloads. Prevents rogue ML training scripts from
accidentally modifying SOUL.md or accessing host OS network interfaces.
"""

import os
import sys
import socket
import builtins
import functools
import threading
from typing import Dict, List, Optional, Set, Callable, Any
from dataclasses import dataclass
from datetime import datetime
import logging
import contextlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SandboxConfig:
    """Configuration for sandbox enforcement."""
    # File system restrictions
    allowed_read_paths: List[str] = None
    allowed_write_paths: List[str] = None
    forbidden_paths: List[str] = None
    
    # Network restrictions
    allow_network: bool = False
    allowed_hosts: List[str] = None
    blocked_ports: List[int] = None
    
    # Module restrictions
    blocked_modules: List[str] = None
    
    # Process restrictions
    allow_subprocess: bool = False
    
    def __post_init__(self):
        if self.allowed_read_paths is None:
            self.allowed_read_paths = ['/tmp', '/workspace/data']
        if self.allowed_write_paths is None:
            self.allowed_write_paths = ['/tmp', '/workspace/output']
        if self.forbidden_paths is None:
            self.forbidden_paths = [
                '/etc', '/root', '/home', '/var/log',
                'SOUL.md', '.git', '__pycache__'
            ]
        if self.allowed_hosts is None:
            self.allowed_hosts = ['localhost', '127.0.0.1']
        if self.blocked_ports is None:
            self.blocked_ports = [22, 23, 3389]  # SSH, Telnet, RDP
        if self.blocked_modules is None:
            self.blocked_modules = [
                'subprocess', 'multiprocessing', 'os.system',
                'pty', 'commands', 'fabric'
            ]


class SandboxViolation(Exception):
    """Raised when sandbox rules are violated."""
    pass


class FilesystemSandbox:
    """Enforces file system access restrictions."""
    
    def __init__(self, config: SandboxConfig):
        self.config = config
        self.violations: List[Dict] = []
        self._original_open = builtins.open
        self._original_os_access = os.access
        self._is_active = False
    
    def _is_path_allowed(self, path: str, mode: str = 'r') -> bool:
        """Check if path access is allowed."""
        # Normalize path
        abs_path = os.path.abspath(path)
        
        # Check forbidden paths first
        for forbidden in self.config.forbidden_paths:
            if forbidden in abs_path:
                return False
        
        if 'w' in mode or 'a' in mode or '+' in mode:
            # Write mode - check allowed write paths
            allowed = False
            for allowed_path in self.config.allowed_write_paths:
                if abs_path.startswith(allowed_path):
                    allowed = True
                    break
            if not allowed:
                return False
        else:
            # Read mode - check allowed read paths
            allowed = False
            for allowed_path in self.config.allowed_read_paths:
                if abs_path.startswith(allowed_path):
                    allowed = True
                    break
            if not allowed:
                return False
        
        return True
    
    def _sandboxed_open(self, *args, **kwargs):
        """Sandboxed version of open()."""
        if not self._is_active:
            return self._original_open(*args, **kwargs)
        
        path = args[0] if args else kwargs.get('file')
        mode = args[1] if len(args) > 1 else kwargs.get('mode', 'r')
        
        if not self._is_path_allowed(path, mode):
            self.violations.append({
                "type": "filesystem",
                "operation": "open",
                "path": path,
                "mode": mode,
                "timestamp": datetime.utcnow().isoformat()
            })
            raise SandboxViolation(f"Access denied to {path} with mode {mode}")
        
        return self._original_open(*args, **kwargs)
    
    def activate(self):
        """Activate filesystem sandboxing."""
        builtins.open = self._sandboxed_open
        self._is_active = True
        logger.info("Filesystem sandbox activated")
    
    def deactivate(self):
        """Deactivate filesystem sandboxing."""
        builtins.open = self._original_open
        self._is_active = False
        logger.info("Filesystem sandbox deactivated")


class NetworkSandbox:
    """Enforces network access restrictions."""
    
    def __init__(self, config: SandboxConfig):
        self.config = config
        self.violations: List[Dict] = []
        self._original_socket_connect = socket.socket.connect
        self._original_socket_sendto = socket.socket.sendto
        self._is_active = False
    
    def _is_host_allowed(self, host: str, port: int) -> bool:
        """Check if host:port connection is allowed."""
        if not self.config.allow_network:
            return False
        
        # Check blocked ports
        if port in self.config.blocked_ports:
            return False
        
        # Check allowed hosts
        for allowed in self.config.allowed_hosts:
            if host == allowed or host.endswith(f".{allowed}"):
                return True
        
        return False
    
    def _sandboxed_connect(self, sock, address, *args, **kwargs):
        """Sandboxed version of socket.connect()."""
        if not self._is_active:
            return self._original_socket_connect(sock, address, *args, **kwargs)
        
        host, port = address[0], address[1]
        
        if not self._is_host_allowed(host, port):
            self.violations.append({
                "type": "network",
                "operation": "connect",
                "host": host,
                "port": port,
                "timestamp": datetime.utcnow().isoformat()
            })
            raise SandboxViolation(f"Connection denied to {host}:{port}")
        
        return self._original_socket_connect(sock, address, *args, **kwargs)
    
    def activate(self):
        """Activate network sandboxing."""
        # Monkey-patch socket methods
        original_connect = socket.socket.connect
        
        def patched_connect(self_sock, address, *args, **kwargs):
            return self._sandboxed_connect(self_sock, address, *args, **kwargs)
        
        socket.socket.connect = patched_connect
        self._is_active = True
        logger.info("Network sandbox activated")
    
    def deactivate(self):
        """Deactivate network sandboxing."""
        self._is_active = False
        logger.info("Network sandbox deactivated")


class ModuleSandbox:
    """Restricts dangerous module imports and functions."""
    
    def __init__(self, config: SandboxConfig):
        self.config = config
        self.violations: List[Dict] = []
        self._blocked_imports: Set[str] = set(config.blocked_modules)
        self._is_active = False
    
    def _check_import(self, name: str) -> bool:
        """Check if module import is allowed."""
        for blocked in self._blocked_imports:
            if name == blocked or name.startswith(blocked + '.'):
                return False
        return True
    
    def activate(self):
        """Activate module sandboxing."""
        self._is_active = True
        logger.info(f"Module sandbox activated, blocking: {self._blocked_imports}")
    
    def deactivate(self):
        """Deactivate module sandboxing."""
        self._is_active = False
        logger.info("Module sandbox deactivated")
    
    def validate_import(self, name: str):
        """Validate an import attempt."""
        if self._is_active and not self._check_import(name):
            self.violations.append({
                "type": "module",
                "operation": "import",
                "module": name,
                "timestamp": datetime.utcnow().isoformat()
            })
            raise SandboxViolation(f"Import denied for module: {name}")


class ExecutionSandbox:
    """
    Main sandbox enforcer combining all restrictions.
    Use as context manager for scoped sandboxing.
    """
    
    def __init__(self, config: Optional[SandboxConfig] = None):
        self.config = config or SandboxConfig()
        self.fs_sandbox = FilesystemSandbox(self.config)
        self.net_sandbox = NetworkSandbox(self.config)
        self.module_sandbox = ModuleSandbox(self.config)
        
        self.all_violations: List[Dict] = []
        self.is_active = False
    
    def activate(self):
        """Activate all sandbox restrictions."""
        self.fs_sandbox.activate()
        self.net_sandbox.activate()
        self.module_sandbox.activate()
        self.is_active = True
        logger.info("Execution sandbox fully activated")
    
    def deactivate(self):
        """Deactivate all sandbox restrictions."""
        self.fs_sandbox.deactivate()
        self.net_sandbox.deactivate()
        self.module_sandbox.deactivate()
        self.is_active = False
        logger.info("Execution sandbox deactivated")
    
    def collect_violations(self) -> List[Dict]:
        """Collect all violations from sub-sandboxes."""
        violations = []
        violations.extend(self.fs_sandbox.violations)
        violations.extend(self.net_sandbox.violations)
        violations.extend(self.module_sandbox.violations)
        return violations
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get sandbox statistics."""
        violations = self.collect_violations()
        return {
            "is_active": self.is_active,
            "total_violations": len(violations),
            "fs_violations": len(self.fs_sandbox.violations),
            "network_violations": len(self.net_sandbox.violations),
            "module_violations": len(self.module_sandbox.violations),
            "config": {
                "allow_network": self.config.allow_network,
                "allow_subprocess": self.config.allow_subprocess,
                "blocked_modules_count": len(self.config.blocked_modules)
            }
        }
    
    def __enter__(self):
        self.activate()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.deactivate()
        return False


def sandboxed_execution(func: Callable) -> Callable:
    """
    Decorator for running functions in a sandboxed environment.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        config = kwargs.pop('sandbox_config', None)
        with ExecutionSandbox(config) as sandbox:
            try:
                result = func(*args, **kwargs)
                return result
            except SandboxViolation as e:
                logger.error(f"Sandbox violation in {func.__name__}: {e}")
                violations = sandbox.collect_violations()
                raise RuntimeError(
                    f"Sandbox violation: {e}. Violations: {violations}"
                )
    return wrapper


def create_ray_worker_sandbox() -> ExecutionSandbox:
    """
    Create a sandbox specifically configured for Ray workers.
    More restrictive than default to protect SOUL.md and system files.
    """
    config = SandboxConfig(
        allowed_read_paths=['/tmp', '/workspace/data', '/workspace/python'],
        allowed_write_paths=['/tmp', '/workspace/output', '/workspace/logs'],
        forbidden_paths=[
            'SOUL.md', '.git', '__pycache__', 
            '/etc/passwd', '/etc/shadow', '/root'
        ],
        allow_network=False,  # No network for workers
        blocked_modules=['subprocess', 'multiprocessing', 'socket'],
        allow_subprocess=False
    )
    
    return ExecutionSandbox(config)


# Export for module use
__all__ = [
    "SandboxConfig",
    "SandboxViolation",
    "FilesystemSandbox",
    "NetworkSandbox",
    "ModuleSandbox",
    "ExecutionSandbox",
    "sandboxed_execution",
    "create_ray_worker_sandbox"
]
