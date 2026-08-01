#!/usr/bin/env python3
"""
Daemon Module Root - Stage 50
Finalizes deployment, locks memory, and hands control to OS service manager.
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict
import threading
import signal
import ctypes
import resource

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
)
logger = logging.getLogger('DaemonMod')

# Constants
WORKSPACE_ROOT = Path('/workspace')
PID_FILE = WORKSPACE_ROOT / 'daemon.pid'
LOCK_FILE = WORKSPACE_ROOT / '.daemon_lock'


class MemoryLocker:
    """Locks memory pages to prevent swapping."""
    
    def __init__(self, max_lock_mb: int = 512):
        self.max_lock_bytes = max_lock_mb * 1024 * 1024
        self.locked = False
    
    def lock_memory(self) -> bool:
        """Attempt to lock memory pages."""
        if sys.platform.startswith('linux'):
            return self._lock_linux()
        elif sys.platform == 'darwin':
            return self._lock_macos()
        else:
            logger.warning(f"Memory locking not supported on {sys.platform}")
            return True
    
    def _lock_linux(self) -> bool:
        """Lock memory on Linux using mlockall."""
        try:
            # Try to use ctypes to call mlockall
            libc = ctypes.CDLL('libc.so.6', use_errno=True)
            
            MCL_CURRENT = 1      # Lock all currently mapped pages
            MCL_FUTURE = 2       # Lock all pages that will become mapped
            
            result = libc.mlockall(MCL_CURRENT | MCL_FUTURE)
            
            if result == 0:
                self.locked = True
                logger.info("Memory locked successfully (mlockall)")
                return True
            else:
                errno = ctypes.get_errno()
                logger.warning(f"mlockall failed with errno {errno}: {os.strerror(errno)}")
                return False
        
        except Exception as e:
            logger.warning(f"Could not lock memory: {e}")
            return False
    
    def _lock_macos(self) -> bool:
        """Lock memory on macOS."""
        try:
            libc = ctypes.CDLL('libSystem.dylib', use_errno=True)
            
            MCL_CURRENT = 1
            MCL_FUTURE = 2
            
            result = libc.mlockall(MCL_CURRENT | MCL_FUTURE)
            
            if result == 0:
                self.locked = True
                logger.info("Memory locked successfully (macOS)")
                return True
            else:
                logger.warning("mlockall failed on macOS")
                return False
        
        except Exception as e:
            logger.warning(f"Could not lock memory on macOS: {e}")
            return False
    
    def set_rlimit(self, memlock_mb: int = 512):
        """Set resource limit for memory locking."""
        try:
            soft_limit = memlock_mb * 1024 * 1024
            hard_limit = resource.RLIM_INFINITY
            
            resource.setrlimit(resource.RLIMIT_MEMLOCK, (soft_limit, hard_limit))
            logger.info(f"Set RLIMIT_MEMLOCK to {memlock_mb}MB")
        
        except Exception as e:
            logger.warning(f"Could not set memory lock limit: {e}")
    
    def unlock_memory(self):
        """Unlock memory pages."""
        if not self.locked:
            return
        
        try:
            if sys.platform.startswith('linux'):
                libc = ctypes.CDLL('libc.so.6')
                libc.munlockall()
            elif sys.platform == 'darwin':
                libc = ctypes.CDLL('libSystem.dylib')
                libc.munlockall()
            
            self.locked = False
            logger.info("Memory unlocked")
        
        except Exception as e:
            logger.error(f"Error unlocking memory: {e}")


class PIDManager:
    """Manages PID file for daemon process."""
    
    def __init__(self, pid_file: Path = PID_FILE):
        self.pid_file = pid_file
        self.pid = os.getpid()
    
    def write_pid(self) -> bool:
        """Write PID to file."""
        try:
            self.pid_file.write_text(str(self.pid))
            logger.info(f"PID {self.pid} written to {self.pid_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to write PID file: {e}")
            return False
    
    def read_pid(self) -> Optional[int]:
        """Read PID from file."""
        try:
            if self.pid_file.exists():
                return int(self.pid_file.read_text().strip())
        except:
            pass
        return None
    
    def check_running(self) -> bool:
        """Check if another instance is running."""
        existing_pid = self.read_pid()
        
        if existing_pid is None:
            return False
        
        # Check if process exists
        try:
            os.kill(existing_pid, 0)
            return True
        except OSError:
            # Process doesn't exist, stale PID file
            return False
    
    def remove_pid(self):
        """Remove PID file."""
        try:
            if self.pid_file.exists():
                self.pid_file.unlink()
                logger.info(f"PID file removed: {self.pid_file}")
        except Exception as e:
            logger.error(f"Failed to remove PID file: {e}")


class DaemonCoordinator:
    """Coordinates daemon operations and handoff to OS service manager."""
    
    def __init__(self):
        self.memory_locker = MemoryLocker()
        self.pid_manager = PIDManager()
        self.running = False
        self.shutdown_event = threading.Event()
        self.service_type: Optional[str] = None
    
    def detect_service_manager(self) -> str:
        """Detect which service manager is available."""
        if Path('/run/systemd/system').exists():
            self.service_type = 'systemd'
            return 'systemd'
        elif sys.platform.startswith('win'):
            self.service_type = 'windows'
            return 'windows'
        elif Path('/Library/LaunchDaemons').exists():
            self.service_type = 'launchd'
            return 'launchd'
        else:
            self.service_type = 'unknown'
            return 'unknown'
    
    def initialize(self) -> bool:
        """Initialize daemon environment."""
        logger.info("Initializing daemon environment...")
        
        # Set memory limits
        self.memory_locker.set_rlimit()
        
        # Lock memory
        if not self.memory_locker.lock_memory():
            logger.warning("Memory locking failed, continuing anyway")
        
        # Write PID file
        if self.pid_manager.check_running():
            existing_pid = self.pid_manager.read_pid()
            logger.error(f"Another instance may be running (PID {existing_pid})")
            return False
        
        if not self.pid_manager.write_pid():
            return False
        
        # Create lock file
        try:
            LOCK_FILE.write_text(datetime.now().isoformat())
        except:
            pass
        
        self.running = True
        logger.info("Daemon environment initialized")
        return True
    
    def handoff_to_service_manager(self) -> bool:
        """Hand off control to the OS service manager."""
        service_type = self.detect_service_manager()
        
        logger.info(f"Detected service manager: {service_type}")
        
        if service_type == 'systemd':
            logger.info("System will manage service via systemd")
            logger.info("Use: sudo systemctl [start|stop|status] crypto_bot")
        
        elif service_type == 'windows':
            logger.info("Service configured for Windows Task Scheduler or NSSM")
            logger.info("Run the generated PowerShell script as Administrator")
        
        elif service_type == 'launchd':
            logger.info("Service configured for macOS launchd")
        
        else:
            logger.warning("No service manager detected, running in standalone mode")
        
        return True
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def handler(signum, frame):
            sig_name = signal.Signals(signum).name
            logger.info(f"Received {sig_name}, initiating shutdown...")
            self.shutdown()
        
        signal.signal(signal.SIGTERM, handler)
        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGHUP, lambda s, f: logger.info("Received SIGHUP"))
    
    def wait_for_shutdown(self, timeout: Optional[float] = None):
        """Wait for shutdown signal."""
        logger.info("Daemon running, waiting for shutdown signal...")
        self.shutdown_event.wait(timeout=timeout)
    
    def shutdown(self):
        """Graceful shutdown."""
        if not self.running:
            return
        
        logger.info("Shutting down daemon...")
        self.running = False
        self.shutdown_event.set()
        
        # Cleanup
        self.memory_locker.unlock_memory()
        self.pid_manager.remove_pid()
        
        try:
            if LOCK_FILE.exists():
                LOCK_FILE.unlink()
        except:
            pass
        
        logger.info("Daemon shutdown complete")
    
    def run(self, main_callback=None, timeout: Optional[float] = None):
        """Main daemon run loop."""
        if not self.initialize():
            return 1
        
        self.setup_signal_handlers()
        self.handoff_to_service_manager()
        
        if main_callback:
            # Run main callback in separate thread
            main_thread = threading.Thread(target=main_callback, daemon=True)
            main_thread.start()
        
        try:
            self.wait_for_shutdown(timeout=timeout)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        finally:
            self.shutdown()
        
        return 0


def create_daemon() -> DaemonCoordinator:
    """Factory function to create daemon coordinator."""
    return DaemonCoordinator()


def finalize_deployment():
    """Finalize deployment and prepare for handoff."""
    logger.info("=" * 60)
    logger.info("FINALIZING DEPLOYMENT - STAGE 50")
    logger.info("=" * 60)
    
    coordinator = create_daemon()
    
    # Initialize
    if not coordinator.initialize():
        logger.error("Deployment initialization failed")
        return False
    
    # Handoff
    coordinator.handoff_to_service_manager()
    
    # Cleanup
    coordinator.shutdown()
    
    logger.info("=" * 60)
    logger.info("DEPLOYMENT FINALIZED")
    logger.info("=" * 60)
    
    return True


def main():
    """Entry point for daemon module."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Daemon Module')
    parser.add_argument('--finalize', action='store_true', 
                       help='Finalize deployment')
    parser.add_argument('--timeout', type=float, default=None,
                       help='Shutdown timeout in seconds')
    args = parser.parse_args()
    
    if args.finalize:
        success = finalize_deployment()
        sys.exit(0 if success else 1)
    
    coordinator = create_daemon()
    sys.exit(coordinator.run(timeout=args.timeout))


if __name__ == '__main__':
    main()
