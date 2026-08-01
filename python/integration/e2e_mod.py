"""
End-to-End Integration Module Root - Ties Python lifecycle to Rust orchestrator.
Provides flawless 24/7 stability through coordinated startup, monitoring, and shutdown.
"""
import numpy as np
import logging
from typing import Dict, Any, Optional, Callable
from pathlib import Path
import time
import threading

logger = logging.getLogger(__name__)

# Import integration submodules
try:
    from .rust_handshake import RustHandshakeValidator, get_rust_handshake
    from .graceful_teardown import (
        GracefulTeardown, 
        setup_graceful_shutdown,
        get_graceful_teardown
    )
except ImportError as e:
    logger.warning(f"Integration submodules not fully available: {e}")
    RustHandshakeValidator = None
    GracefulTeardown = None


class E2EIntegrationManager:
    """
    Central manager for end-to-end system integration.
    Coordinates Python lifecycle with Rust orchestrator for 24/7 stability.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize submodules
        self.handshake_validator = None
        self.teardown_handler = None
        
        if RustHandshakeValidator is not None:
            self.handshake_validator = get_rust_handshake({
                'shm_name': self.config.get('shm_name', 'hft_ipc_shm'),
                'shm_size': self.config.get('shm_size', 4096),
                'handshake_timeout': self.config.get('handshake_timeout', 30.0)
            })
            logger.info("RustHandshakeValidator initialized")
        
        if GracefulTeardown is not None:
            self.teardown_handler = setup_graceful_shutdown({
                'nautilus_state_path': self.config.get('nautilus_state_path', 'data/nautilus_state.pkl')
            })
            logger.info("GracefulTeardown configured")
        
        # State
        self._initialized = False
        self._running = False
        self._healthy = False
        
        # Health monitoring
        self._health_thread: Optional[threading.Thread] = None
        self._health_check_interval = self.config.get('health_check_interval', 5.0)
        self._consecutive_failures = 0
        self._max_failures_before_restart = self.config.get('max_failures', 3)
        
        # Callbacks
        self._on_system_ready: Optional[Callable] = None
        self._on_system_unhealthy: Optional[Callable] = None
        self._on_restart_requested: Optional[Callable] = None
        
        # Statistics
        self._uptime_start = 0.0
        self._restart_count = 0
        
        logger.info("E2EIntegrationManager initialized")
    
    def initialize(self) -> bool:
        """
        Initialize all integration components.
        
        Returns:
            Success status
        """
        if self._initialized:
            logger.info("Already initialized")
            return True
        
        try:
            # Step 1: Perform handshake with Rust
            if self.handshake_validator:
                if not self.handshake_validator.initiate_handshake():
                    logger.error("Failed to complete Rust handshake")
                    return False
                
                # Unlock strategies after successful handshake
                if not self.handshake_validator.unlock_strategies():
                    logger.error("Failed to unlock strategies")
                    return False
            
            # Step 2: Register cleanup handlers
            self._register_cleanup_handlers()
            
            self._initialized = True
            self._uptime_start = time.time()
            
            logger.info("E2EIntegrationManager initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False
    
    def _register_cleanup_handlers(self) -> None:
        """Register additional cleanup handlers."""
        if self.teardown_handler:
            # Register module cleanup
            self.teardown_handler.register_cleanup(
                self._cleanup_modules,
                "module_cleanup"
            )
    
    def _cleanup_modules(self) -> None:
        """Clean up all Python modules before shutdown."""
        logger.info("Cleaning up Python modules...")
        
        # Reset singleton instances in reverse dependency order
        try:
            # UI components
            from ..ui.ui_backend_mod import reset_ui_backend_manager
            reset_ui_backend_manager()
        except Exception as e:
            logger.debug(f"UI cleanup: {e}")
        
        try:
            # MLOps components
            from ..mlops.adv_mlops_mod import reset_mlops_manager
            reset_mlops_manager()
        except Exception as e:
            logger.debug(f"MLOps cleanup: {e}")
        
        try:
            # Compliance components
            from ..compliance.comp_mod import reset_compliance_router
            reset_compliance_router()
        except Exception as e:
            logger.debug(f"Compliance cleanup: {e}")
        
        try:
            # SMC components
            from ..smc.smc_mod import reset_smc_manager
            reset_smc_manager()
        except Exception as e:
            logger.debug(f"SMC cleanup: {e}")
        
        logger.info("Module cleanup completed")
    
    def start_health_monitoring(self) -> bool:
        """Start background health monitoring."""
        if not self._initialized:
            logger.error("Cannot start monitoring: not initialized")
            return False
        
        if self._health_thread and self._health_thread.is_alive():
            logger.info("Health monitoring already running")
            return True
        
        self._running = True
        self._healthy = True
        self._consecutive_failures = 0
        
        self._health_thread = threading.Thread(
            target=self._health_monitoring_loop,
            daemon=True,
            name="E2EHealthMonitor"
        )
        self._health_thread.start()
        
        logger.info("Health monitoring started")
        return True
    
    def _health_monitoring_loop(self) -> None:
        """Background health monitoring loop."""
        while self._running:
            try:
                healthy = self._perform_health_check()
                
                if healthy:
                    self._consecutive_failures = 0
                    self._healthy = True
                else:
                    self._consecutive_failures += 1
                    
                    if self._consecutive_failures >= self._max_failures_before_restart:
                        logger.error(f"System unhealthy: {self._consecutive_failures} consecutive failures")
                        self._healthy = False
                        
                        if self._on_system_unhealthy:
                            self._on_system_unhealthy()
                        
                        # Request restart if configured
                        if self.config.get('auto_restart_on_failure', False):
                            self._request_restart()
                
            except Exception as e:
                logger.error(f"Health check error: {e}")
                self._consecutive_failures += 1
            
            time.sleep(self._health_check_interval)
    
    def _perform_health_check(self) -> bool:
        """Perform comprehensive health check."""
        checks_passed = 0
        checks_total = 0
        
        # Check 1: Handshake validity
        checks_total += 1
        if self.handshake_validator:
            if self.handshake_validator.validate_handshake():
                checks_passed += 1
            else:
                logger.warning("Handshake validation failed")
        
        # Check 2: Strategies unlocked
        checks_total += 1
        if self.handshake_validator:
            if self.handshake_validator.are_strategies_unlocked():
                checks_passed += 1
            else:
                logger.warning("Strategies not unlocked")
        
        # Check 3: IPC connection healthy
        checks_total += 1
        if self.handshake_validator:
            if self.handshake_validator.ipc.is_healthy():
                checks_passed += 1
            else:
                logger.warning("IPC connection unhealthy")
        
        # Overall health (all checks must pass)
        return checks_passed == checks_total
    
    def _request_restart(self) -> None:
        """Request system restart."""
        logger.warning("Restart requested")
        
        if self._on_restart_requested:
            self._on_restart_requested()
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        uptime = time.time() - self._uptime_start if self._uptime_start else 0
        
        status = {
            'initialized': self._initialized,
            'running': self._running,
            'healthy': self._healthy,
            'uptime_seconds': uptime,
            'restart_count': self._restart_count,
            'consecutive_failures': self._consecutive_failures,
            'strategies_unlocked': (
                self.handshake_validator.are_strategies_unlocked() 
                if self.handshake_validator else False
            )
        }
        
        # Add subsystem statuses
        if self.handshake_validator:
            status['handshake'] = self.handshake_validator.get_status()
        
        if self.teardown_handler:
            status['teardown'] = self.teardown_handler.get_status()
        
        return status
    
    def set_callbacks(self, on_ready: Optional[Callable] = None,
                      on_unhealthy: Optional[Callable] = None,
                      on_restart: Optional[Callable] = None) -> None:
        """Set system event callbacks."""
        self._on_system_ready = on_ready
        self._on_system_unhealthy = on_unhealthy
        self._on_restart_requested = on_restart
    
    def notify_system_ready(self) -> None:
        """Notify that system is ready for trading."""
        if self._on_system_ready:
            self._on_system_ready()
        logger.info("System READY notification sent")
    
    def request_restart(self) -> bool:
        """Manually request system restart."""
        self._request_restart()
        return True
    
    def close(self) -> None:
        """Clean up integration manager."""
        logger.info("Shutting down E2EIntegrationManager...")
        
        self._running = False
        
        if self._health_thread:
            self._health_thread.join(timeout=5.0)
        
        if self.handshake_validator:
            self.handshake_validator.close()
        
        if self.teardown_handler:
            self.teardown_handler.restore_signal_handlers()
        
        logger.info("E2EIntegrationManager closed")


# Singleton instance
_e2e_integration: Optional[E2EIntegrationManager] = None


def get_e2e_integration(config: Optional[Dict[str, Any]] = None) -> E2EIntegrationManager:
    """Get or create singleton E2EIntegrationManager instance."""
    global _e2e_integration
    if _e2e_integration is None:
        _e2e_integration = E2EIntegrationManager(config)
    return _e2e_integration


def reset_e2e_integration() -> None:
    """Reset singleton instance."""
    global _e2e_integration
    if _e2e_integration is not None:
        _e2e_integration.close()
    _e2e_integration = None


def initialize_system(config: Optional[Dict[str, Any]] = None) -> bool:
    """
    Initialize the complete Python system.
    Convenience function for Rust orchestrator.
    """
    manager = get_e2e_integration(config)
    
    if not manager.initialize():
        return False
    
    if not manager.start_health_monitoring():
        return False
    
    # Notify system ready
    manager.notify_system_ready()
    
    return True


def get_system_status() -> Dict[str, Any]:
    """Get current system status."""
    manager = get_e2e_integration()
    return manager.get_status()


def request_system_restart() -> bool:
    """Request system restart."""
    manager = get_e2e_integration()
    return manager.request_restart()


__all__ = [
    'E2EIntegrationManager',
    'get_e2e_integration',
    'reset_e2e_integration',
    'initialize_system',
    'get_system_status',
    'request_system_restart'
]
