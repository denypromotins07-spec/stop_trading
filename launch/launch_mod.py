#!/usr/bin/env python3
"""
Launch Module Root - Stage 50
Manages lifecycle, signal trapping (SIGINT, SIGTERM), and interactive shutdown prompts.
"""

import os
import sys
import signal
import time
import logging
from datetime import datetime
from typing import Optional, Callable, List
from pathlib import Path
import threading
import queue

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
)
logger = logging.getLogger('LaunchMod')

# Import sibling modules
sys.path.insert(0, str(Path(__file__).parent.parent))


class SignalHandler:
    """Centralized signal handling with graceful shutdown support."""
    
    def __init__(self):
        self.handlers: List[Callable] = []
        self.shutdown_requested = False
        self.shutdown_reason: Optional[str] = None
        self._lock = threading.Lock()
        
        # Register default handlers
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGHUP, self._handle_sighup)
    
    def register_handler(self, callback: Callable):
        """Register a callback to be invoked on shutdown."""
        self.handlers.append(callback)
    
    def _handle_signal(self, signum, frame):
        """Handle interrupt signals."""
        sig_name = signal.Signals(signum).name
        logger.info(f"Received signal: {sig_name}")
        
        with self._lock:
            if self.shutdown_requested:
                logger.warning("Shutdown already in progress, ignoring duplicate signal")
                return
            
            self.shutdown_requested = True
            self.shutdown_reason = f"Signal {sig_name}"
        
        # Invoke all registered handlers
        for handler in self.handlers:
            try:
                handler(sig_name)
            except Exception as e:
                logger.error(f"Error in shutdown handler: {e}")
    
    def _handle_sighup(self, signum, frame):
        """Handle SIGHUP for configuration reload (optional)."""
        logger.info("Received SIGHUP - configuration reload not implemented")
    
    def request_shutdown(self, reason: str = "User requested"):
        """Programmatically request shutdown."""
        with self._lock:
            if self.shutdown_requested:
                return
            self.shutdown_requested = True
            self.shutdown_reason = reason
        
        logger.info(f"Shutdown requested: {reason}")
        for handler in self.handlers:
            try:
                handler(reason)
            except Exception as e:
                logger.error(f"Error in shutdown handler: {e}")
    
    def is_shutdown_requested(self) -> bool:
        """Check if shutdown has been requested."""
        return self.shutdown_requested
    
    def get_shutdown_reason(self) -> Optional[str]:
        """Get the reason for shutdown."""
        return self.shutdown_reason


class InteractiveShutdownPrompt:
    """Handles interactive Yes/No shutdown confirmation."""
    
    def __init__(self, signal_handler: SignalHandler):
        self.signal_handler = signal_handler
        self.prompt_queue = queue.Queue()
        self.input_thread: Optional[threading.Thread] = None
        self.enabled = False
    
    def enable(self):
        """Enable interactive prompt mode."""
        self.enabled = True
    
    def disable(self):
        """Disable interactive prompt mode."""
        self.enabled = False
    
    def start_input_listener(self):
        """Start background thread to listen for user input."""
        self.input_thread = threading.Thread(target=self._input_loop, daemon=True)
        self.input_thread.start()
        logger.info("Interactive shutdown prompt enabled")
    
    def _input_loop(self):
        """Background loop waiting for user input."""
        while self.enabled and not self.signal_handler.is_shutdown_requested():
            try:
                # Non-blocking check for input availability
                import select
                if select.select([sys.stdin], [], [], 1.0)[0]:
                    user_input = sys.stdin.readline().strip().lower()
                    if user_input in ['yes', 'y']:
                        logger.info("User confirmed shutdown via interactive prompt")
                        self.signal_handler.request_shutdown("User confirmed via prompt")
                    elif user_input in ['no', 'n']:
                        logger.info("User declined shutdown, continuing operation")
                    elif user_input:
                        logger.info(f"Unrecognized input: {user_input}")
            except Exception as e:
                # Ignore input errors, continue running
                pass
    
    def show_prompt(self, message: str = "Do you want to stop the bot? (yes/no): "):
        """Display shutdown prompt to user."""
        if not self.enabled:
            return
        
        try:
            print(f"\n⚠️  {message}", end="", flush=True)
        except:
            pass


class LifecycleManager:
    """Manages the complete lifecycle of the trading system."""
    
    def __init__(self):
        self.signal_handler = SignalHandler()
        self.interactive_prompt = InteractiveShutdownPrompt(self.signal_handler)
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.state = "INITIALIZING"
        self._state_lock = threading.Lock()
        
        # Register self as shutdown handler
        self.signal_handler.register_handler(self._on_shutdown)
    
    def set_state(self, new_state: str):
        """Update system state."""
        with self._state_lock:
            old_state = self.state
            self.state = new_state
            logger.info(f"State transition: {old_state} → {new_state}")
    
    def get_state(self) -> str:
        """Get current system state."""
        return self.state
    
    def start(self, duration_hours: int = 4):
        """Start the trading system lifecycle."""
        self.start_time = datetime.now()
        self.end_time = self.start_time + timedelta(hours=duration_hours)
        self.set_state("RUNNING")
        
        logger.info("=" * 60)
        logger.info(f"Trading session started at {self.start_time.isoformat()}")
        logger.info(f"Scheduled end at {self.end_time.isoformat()}")
        logger.info(f"Duration: {duration_hours} hours")
        logger.info("=" * 60)
        
        # Enable interactive prompt
        self.interactive_prompt.enable()
        self.interactive_prompt.start_input_listener()
    
    def stop(self, reason: str = "Normal shutdown"):
        """Stop the trading system."""
        self.signal_handler.request_shutdown(reason)
    
    def _on_shutdown(self, reason: str):
        """Callback invoked when shutdown is requested."""
        self.set_state("SHUTTING_DOWN")
        
        elapsed = datetime.now() - self.start_time if self.start_time else timedelta(0)
        logger.info("=" * 60)
        logger.info("SHUTDOWN SEQUENCE INITIATED")
        logger.info(f"Reason: {reason}")
        logger.info(f"Session duration: {elapsed}")
        logger.info("=" * 60)
    
    def get_uptime(self) -> timedelta:
        """Get current uptime."""
        if not self.start_time:
            return timedelta(0)
        return datetime.now() - self.start_time
    
    def get_remaining_time(self) -> timedelta:
        """Get remaining time in trading window."""
        if not self.end_time:
            return timedelta(0)
        remaining = self.end_time - datetime.now()
        return max(timedelta(0), remaining)
    
    def is_within_window(self) -> bool:
        """Check if currently within trading window."""
        return self.get_remaining_time().total_seconds() > 0


class LaunchCoordinator:
    """Coordinates the launch sequence across all subsystems."""
    
    def __init__(self):
        self.lifecycle = LifecycleManager()
        self.components: List[str] = []
        self.component_status: dict = {}
    
    def register_component(self, name: str, init_func: Callable, shutdown_func: Optional[Callable] = None):
        """Register a component for coordinated startup/shutdown."""
        self.components.append(name)
        self.component_status[name] = {
            'init': init_func,
            'shutdown': shutdown_func,
            'started': False,
            'healthy': True
        }
    
    def launch_sequence(self):
        """Execute the full launch sequence."""
        logger.info("Beginning launch sequence...")
        
        # Start lifecycle
        self.lifecycle.start()
        
        # Initialize components in order
        for name, info in self.component_status.items():
            try:
                logger.info(f"Initializing {name}...")
                result = info['init']()
                info['started'] = True
                info['healthy'] = result is not False
                logger.info(f"{name} initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize {name}: {e}")
                info['healthy'] = False
                
                # Critical components fail fast
                if name in ['rust_core', 'message_bus']:
                    logger.critical(f"Critical component {name} failed, aborting launch")
                    self.abort_launch()
                    return False
        
        logger.info("Launch sequence complete - system operational")
        return True
    
    def shutdown_sequence(self):
        """Execute graceful shutdown sequence."""
        logger.info("Beginning shutdown sequence...")
        
        # Shutdown components in reverse order
        for name in reversed(self.components):
            info = self.component_status[name]
            if info['started'] and info['shutdown']:
                try:
                    logger.info(f"Shutting down {name}...")
                    info['shutdown']()
                    logger.info(f"{name} shut down complete")
                except Exception as e:
                    logger.error(f"Error shutting down {name}: {e}")
        
        self.lifecycle.set_state("STOPPED")
        logger.info("Shutdown sequence complete")
    
    def abort_launch(self):
        """Abort the launch sequence due to critical failure."""
        logger.critical("ABORTING LAUNCH - Critical failure detected")
        self.shutdown_sequence()
        sys.exit(1)
    
    def run(self):
        """Main run loop."""
        if not self.launch_sequence():
            return 1
        
        # Main loop - wait for shutdown signal
        try:
            while not self.lifecycle.signal_handler.is_shutdown_requested():
                # Check trading window
                if not self.lifecycle.is_within_window():
                    logger.info("Trading window expired")
                    self.lifecycle.stop("Trading window expired")
                    break
                
                # Check component health
                unhealthy = [
                    name for name, info in self.component_status.items()
                    if info['started'] and not info['healthy']
                ]
                
                if unhealthy:
                    logger.warning(f"Unhealthy components: {unhealthy}")
                
                time.sleep(5)
        
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
            self.lifecycle.stop("Keyboard interrupt")
        
        finally:
            self.shutdown_sequence()
        
        return 0


# Import timedelta for LifecycleManager
from datetime import timedelta


def create_launch_coordinator() -> LaunchCoordinator:
    """Factory function to create a configured launch coordinator."""
    return LaunchCoordinator()


def main():
    """Entry point for launch module testing."""
    logger.info("Launch Module - Standalone Test Mode")
    
    coordinator = create_launch_coordinator()
    
    # Example component registration (would be populated by actual system)
    def fake_init():
        return True
    
    def fake_shutdown():
        pass
    
    coordinator.register_component("test_component", fake_init, fake_shutdown)
    
    sys.exit(coordinator.run())


if __name__ == '__main__':
    main()
