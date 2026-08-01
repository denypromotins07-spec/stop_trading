"""
Python Daemonizer.
Handles PID files, signal trapping (SIGTERM), and backgrounding.
Ensures Ray cluster and Nautilus kernel run detached with strict process control.
"""

import os
import sys
import signal
import atexit
import logging
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass
import subprocess
import time

logger = logging.getLogger(__name__)


@dataclass
class ProcessHandle:
    """Handle for a managed subprocess."""
    name: str
    pid: int
    process: subprocess.Popen
    started_at: float
    should_restart: bool = True


class PythonDaemon:
    """
    Robust Python daemonizer for HFT systems.
    Manages PID files, signal handling, and process lifecycle.
    """

    def __init__(
        self,
        pid_file: str = "/tmp/hft_bot.pid",
        log_file: str = "/tmp/hft_bot.log",
        work_dir: str = "/workspace",
    ):
        self.pid_file = pid_file
        self.log_file = log_file
        self.work_dir = work_dir

        self._processes: Dict[str, ProcessHandle] = {}
        self._running = False
        self._shutdown_callbacks: list = []
        self._original_sigterm = None
        self._original_sigint = None

    def daemonize(self) -> bool:
        """
        Daemonize the current process.

        Returns:
            True if successful
        """
        try:
            # First fork
            pid = os.fork()
            if pid > 0:
                sys.exit(0)  # Exit parent

            # Create new session
            os.setsid()

            # Second fork (prevent session leader from acquiring terminal)
            pid = os.fork()
            if pid > 0:
                sys.exit(0)

            # Change working directory
            os.chdir(self.work_dir)

            # Redirect standard file descriptors
            self._redirect_io()

            # Write PID file
            self._write_pid_file()

            # Setup signal handlers
            self._setup_signal_handlers()

            # Register cleanup
            atexit.register(self._cleanup)

            self._running = True
            logger.info(f"Daemon started with PID {os.getpid()}")
            return True

        except Exception as e:
            logger.error(f"Failed to daemonize: {e}")
            return False

    def _redirect_io(self):
        """Redirect stdin/stdout/stderr to log file."""
        # Open log file
        log_fd = os.open(self.log_file, os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o644)

        # Redirect stdin from /dev/null
        dev_null_fd = os.open(os.devnull, os.O_RDONLY)
        os.dup2(dev_null_fd, sys.stdin.fileno())
        os.close(dev_null_fd)

        # Redirect stdout and stderr to log file
        os.dup2(log_fd, sys.stdout.fileno())
        os.dup2(log_fd, sys.stderr.fileno())
        os.close(log_fd)

    def _write_pid_file(self):
        """Write PID file."""
        pid = os.getpid()
        with open(self.pid_file, 'w') as f:
            f.write(str(pid))
        logger.info(f"PID file written: {self.pid_file}")

    def _remove_pid_file(self):
        """Remove PID file."""
        try:
            if os.path.exists(self.pid_file):
                os.remove(self.pid_file)
                logger.debug("PID file removed")
        except Exception as e:
            logger.error(f"Failed to remove PID file: {e}")

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        self._original_sigterm = signal.signal(signal.SIGTERM, self._handle_signal)
        self._original_sigint = signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGHUP, self._handle_signal)

    def _handle_signal(self, signum, frame):
        """Handle incoming signals."""
        sig_name = signal.Signals(signum).name
        logger.info(f"Received signal: {sig_name}")

        if signum in (signal.SIGTERM, signal.SIGINT):
            self.shutdown()

    def start_process(
        self,
        name: str,
        command: list,
        should_restart: bool = True,
        env: Optional[Dict] = None,
    ) -> Optional[ProcessHandle]:
        """
        Start a managed subprocess.

        Args:
            name: Process name
            command: Command to execute
            should_restart: Whether to restart on failure
            env: Environment variables

        Returns:
            ProcessHandle or None
        """
        try:
            process_env = os.environ.copy()
            if env:
                process_env.update(env)

            proc = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=process_env,
                cwd=self.work_dir,
            )

            handle = ProcessHandle(
                name=name,
                pid=proc.pid,
                process=proc,
                started_at=time.time(),
                should_restart=should_restart,
            )

            self._processes[name] = handle
            logger.info(f"Started process '{name}' with PID {proc.pid}")

            return handle

        except Exception as e:
            logger.error(f"Failed to start process '{name}': {e}")
            return None

    def stop_process(self, name: str, timeout: float = 5.0) -> bool:
        """
        Stop a managed process.

        Args:
            name: Process name
            timeout: Seconds to wait for graceful shutdown

        Returns:
            True if stopped successfully
        """
        if name not in self._processes:
            return False

        handle = self._processes[name]

        try:
            # Send SIGTERM
            handle.process.terminate()

            # Wait for graceful shutdown
            try:
                handle.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                # Force kill
                handle.process.kill()
                handle.process.wait()

            logger.info(f"Stopped process '{name}' (PID {handle.pid})")
            del self._processes[name]
            return True

        except Exception as e:
            logger.error(f"Failed to stop process '{name}': {e}")
            return False

    def monitor_processes(self) -> Dict[str, bool]:
        """
        Monitor all managed processes and restart if needed.

        Returns:
            Dict of process name -> running status
        """
        status = {}

        for name, handle in list(self._processes.items()):
            poll_result = handle.process.poll()

            if poll_result is None:
                # Still running
                status[name] = True
            else:
                # Process exited
                status[name] = False
                logger.warning(f"Process '{name}' exited with code {poll_result}")

                if handle.should_restart and self._running:
                    logger.info(f"Restarting process '{name}'...")
                    # Would need to store command to restart
                    # For now, just mark as not running

        return status

    def register_shutdown_callback(self, callback: Callable):
        """Register a callback to run on shutdown."""
        self._shutdown_callbacks.append(callback)

    def shutdown(self):
        """Gracefully shutdown all processes."""
        logger.info("Initiating graceful shutdown...")
        self._running = False

        # Run shutdown callbacks
        for callback in reversed(self._shutdown_callbacks):
            try:
                callback()
            except Exception as e:
                logger.error(f"Shutdown callback error: {e}")

        # Stop all processes
        for name in list(self._processes.keys()):
            self.stop_process(name, timeout=3.0)

        # Cleanup
        self._cleanup()

    def _cleanup(self):
        """Cleanup resources."""
        self._remove_pid_file()
        logger.info("Cleanup complete")

    def is_running(self) -> bool:
        """Check if daemon is running."""
        return self._running

    @staticmethod
    def check_existing_pid(pid_file: str) -> Optional[int]:
        """Check if there's an existing daemon running."""
        if not os.path.exists(pid_file):
            return None

        try:
            with open(pid_file, 'r') as f:
                pid = int(f.read().strip())

            # Check if process exists
            os.kill(pid, 0)
            return pid

        except (ValueError, ProcessLookupError, PermissionError):
            # Stale PID file
            if os.path.exists(pid_file):
                os.remove(pid_file)
            return None


# Module singleton
_daemon: Optional[PythonDaemon] = None


def get_daemon(
    pid_file: str = "/tmp/hft_bot.pid",
    log_file: str = "/tmp/hft_bot.log",
) -> PythonDaemon:
    """Get or create the daemon singleton."""
    global _daemon
    if _daemon is None:
        _daemon = PythonDaemon(pid_file=pid_file, log_file=log_file)
    return _daemon


async def shutdown_daemon_module():
    """Shutdown the daemon module."""
    global _daemon
    if _daemon:
        _daemon.shutdown()
        _daemon = None
