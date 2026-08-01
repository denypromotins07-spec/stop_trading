"""
Graceful Teardown - Python-side /KILL handler for clean shutdown.
Flushes Ray object stores, saves Nautilus state, and exits cleanly.
Forces hard exit if flush operations exceed 5 seconds to prevent zombie processes.
"""
import numpy as np
import logging
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
import time
import signal
import threading
import sys
import os

logger = logging.getLogger(__name__)


class GracefulTeardown:
    """
    Handles graceful shutdown of Python components.
    Intercepts SIGTERM and ensures clean resource cleanup.
    """
    
    FLUSH_TIMEOUT_SECONDS = 5.0
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # State
        self._shutting_down = False
        self._shutdown_complete = False
        self._start_time = 0.0
        
        # Registered cleanup handlers
        self._cleanup_handlers: List[Callable] = []
        
        # State savers
        self._state_savers: List[Callable] = []
        
        # Thread for async shutdown
        self._shutdown_thread: Optional[threading.Thread] = None
        
        # Original signal handlers
        self._original_sigterm = None
        self._original_sigint = None
        
        logger.info("GracefulTeardown initialized")
    
    def register_cleanup(self, handler: Callable, name: str = None) -> None:
        """
        Register a cleanup handler.
        
        Args:
            handler: Function to call during shutdown
            name: Optional handler name for logging
        """
        self._cleanup_handlers.append((name or str(handler), handler))
        logger.debug(f"Registered cleanup handler: {name}")
    
    def register_state_saver(self, saver: Callable, name: str = None) -> None:
        """
        Register a state saver function.
        
        Args:
            saver: Function to save state before shutdown
            name: Optional saver name
        """
        self._state_savers.append((name or str(saver), saver))
        logger.debug(f"Registered state saver: {name}")
    
    def install_signal_handlers(self) -> None:
        """Install SIGTERM and SIGINT handlers."""
        def sigterm_handler(signum, frame):
            logger.warning(f"Received SIGTERM (signal {signum})")
            self.initiate_shutdown()
        
        def sigint_handler(signum, frame):
            logger.warning(f"Received SIGINT (signal {signum})")
            self.initiate_shutdown()
        
        # Store original handlers
        self._original_sigterm = signal.getsignal(signal.SIGTERM)
        self._original_sigint = signal.getsignal(signal.SIGINT)
        
        # Install new handlers
        signal.signal(signal.SIGTERM, sigterm_handler)
        signal.signal(signal.SIGINT, sigint_handler)
        
        logger.info("Signal handlers installed")
    
    def restore_signal_handlers(self) -> None:
        """Restore original signal handlers."""
        if self._original_sigterm:
            signal.signal(signal.SIGTERM, self._original_sigterm)
        if self._original_sigint:
            signal.signal(signal.SIGINT, self._original_sigint)
        
        logger.info("Signal handlers restored")
    
    def initiate_shutdown(self) -> None:
        """
        Initiate graceful shutdown process.
        Runs in background thread with timeout enforcement.
        """
        if self._shutting_down:
            logger.warning("Shutdown already in progress")
            return
        
        self._shutting_down = True
        self._start_time = time.time()
        
        logger.info("Initiating graceful shutdown...")
        
        # Start shutdown in background thread
        self._shutdown_thread = threading.Thread(
            target=self._perform_shutdown,
            daemon=True,
            name="GracefulShutdown"
        )
        self._shutdown_thread.start()
        
        # Wait for completion or timeout
        elapsed = time.time() - self._start_time
        remaining = self.FLUSH_TIMEOUT_SECONDS - elapsed
        
        if remaining > 0:
            self._shutdown_thread.join(timeout=remaining)
        
        # Force exit if still running
        if self._shutdown_thread.is_alive():
            logger.error(f"Shutdown timeout exceeded ({self.FLUSH_TIMEOUT_SECONDS}s)")
            logger.error("Forcing hard exit...")
            self._force_exit()
        else:
            logger.info("Graceful shutdown completed")
            self._shutdown_complete = True
    
    def _perform_shutdown(self) -> None:
        """Perform shutdown sequence."""
        try:
            # Step 1: Save state
            self._save_all_state()
            
            # Step 2: Run cleanup handlers
            self._run_all_cleanup()
            
            # Step 3: Flush logs
            self._flush_logs()
            
        except Exception as e:
            logger.error(f"Shutdown error: {e}")
    
    def _save_all_state(self) -> None:
        """Save state from all registered savers."""
        logger.info("Saving state...")
        
        for name, saver in self._state_savers:
            try:
                start = time.perf_counter()
                saver()
                elapsed = (time.perf_counter() - start) * 1000
                logger.debug(f"State saved: {name} ({elapsed:.2f}ms)")
            except Exception as e:
                logger.error(f"Failed to save state ({name}): {e}")
    
    def _run_all_cleanup(self) -> None:
        """Run all cleanup handlers."""
        logger.info("Running cleanup handlers...")
        
        for name, handler in reversed(self._cleanup_handlers):
            try:
                start = time.perf_counter()
                handler()
                elapsed = (time.perf_counter() - start) * 1000
                logger.debug(f"Cleanup completed: {name} ({elapsed:.2f}ms)")
            except Exception as e:
                logger.error(f"Cleanup failed ({name}): {e}")
    
    def _flush_logs(self) -> None:
        """Flush all log handlers."""
        logging.shutdown()
    
    def _force_exit(self) -> None:
        """Force immediate exit to prevent zombie processes."""
        logger.critical("FORCED EXIT - cleaning up was not completed")
        
        # Try to flush any pending output
        sys.stdout.flush()
        sys.stderr.flush()
        
        # Exit with error code
        os._exit(1)
    
    def is_shutting_down(self) -> bool:
        """Check if shutdown is in progress."""
        return self._shutting_down
    
    def is_shutdown_complete(self) -> bool:
        """Check if shutdown completed successfully."""
        return self._shutdown_complete
    
    def get_status(self) -> Dict[str, Any]:
        """Get teardown status."""
        return {
            'shutting_down': self._shutting_down,
            'shutdown_complete': self._shutdown_complete,
            'elapsed_seconds': time.time() - self._start_time if self._start_time else 0,
            'cleanup_handlers_count': len(self._cleanup_handlers),
            'state_savers_count': len(self._state_savers)
        }


def flush_ray_object_store() -> None:
    """Flush Ray object store to disk."""
    try:
        import ray
        
        if ray.is_initialized():
            # Get object store memory usage
            mem_stats = ray.memory_stats()
            logger.info(f"Ray memory stats before flush: {mem_stats}")
            
            # Force garbage collection
            ray.internal.free()
            
            # Wait for objects to be cleaned up
            time.sleep(0.5)
            
            logger.info("Ray object store flushed")
    except ImportError:
        logger.debug("Ray not available, skipping object store flush")
    except Exception as e:
        logger.error(f"Failed to flush Ray object store: {e}")


def save_nautilus_state(state_path: str = 'data/nautilus_state.pkl') -> None:
    """Save Nautilus portfolio state to disk."""
    try:
        import pickle
        
        # Placeholder for actual Nautilus state
        # In production, this would query the actual Nautilus trader/portfolio
        state = {
            'timestamp': time.time(),
            'saved_by': 'graceful_teardown',
            'portfolio_snapshot': {}  # Would contain actual portfolio data
        }
        
        # Ensure directory exists
        Path(state_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(state_path, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"Nautilus state saved to {state_path}")
        
    except Exception as e:
        logger.error(f"Failed to save Nautilus state: {e}")


# Singleton instance
_graceful_teardown: Optional[GracefulTeardown] = None


def get_graceful_teardown(config: Optional[Dict[str, Any]] = None) -> GracefulTeardown:
    """Get or create singleton GracefulTeardown instance."""
    global _graceful_teardown
    if _graceful_teardown is None:
        _graceful_teardown = GracefulTeardown(config)
    return _graceful_teardown


def setup_graceful_shutdown(config: Optional[Dict[str, Any]] = None) -> GracefulTeardown:
    """
    Set up complete graceful shutdown handling.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured GracefulTeardown instance
    """
    teardown = get_graceful_teardown(config)
    
    # Register standard cleanup handlers
    teardown.register_cleanup(flush_ray_object_store, "ray_object_store")
    teardown.register_state_saver(
        lambda: save_nautilus_state(config.get('nautilus_state_path', 'data/nautilus_state.pkl')),
        "nautilus_state"
    )
    
    # Install signal handlers
    teardown.install_signal_handlers()
    
    logger.info("Graceful shutdown configured")
    return teardown


def reset_graceful_teardown() -> None:
    """Reset singleton instance."""
    global _graceful_teardown
    if _graceful_teardown is not None:
        _graceful_teardown.restore_signal_handlers()
    _graceful_teardown = None


__all__ = [
    'GracefulTeardown',
    'flush_ray_object_store',
    'save_nautilus_state',
    'get_graceful_teardown',
    'setup_graceful_shutdown',
    'reset_graceful_teardown'
]
