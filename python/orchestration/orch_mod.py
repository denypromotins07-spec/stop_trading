"""
Orchestration Module Root.
Ties Python lifecycle directly to Rust /START and /KILL IPC commands.
"""

import asyncio
from typing import Optional, Dict, Any, Callable
import logging
import signal
import zmq
import json

from .daemon import (
    PythonDaemon,
    get_daemon,
    shutdown_daemon_module,
)
from .health_monitor import (
    HealthMonitor,
    HealthMetrics,
    HealthStatus,
    get_health_monitor,
    initialize_health_monitor,
    shutdown_health_monitor,
)

logger = logging.getLogger(__name__)


class OrchestrationModule:
    """
    Central orchestrator tying Python lifecycle to Rust IPC commands.
    Handles /START and /KILL commands from Rust side.
    """

    def __init__(
        self,
        ipc_endpoint: str = "tcp://localhost:5556",
        pid_file: str = "/tmp/hft_bot.pid",
        log_file: str = "/tmp/hft_bot.log",
    ):
        self.ipc_endpoint = ipc_endpoint
        self.pid_file = pid_file
        self.log_file = log_file

        self._daemon: Optional[PythonDaemon] = None
        self._health_monitor: Optional[HealthMonitor] = None
        self._zmq_context: Optional[zmq.Context] = None
        self._zmq_socket: Optional[zmq.Socket] = None
        self._running = False
        self._started = False

        self._start_callbacks: list = []
        self._kill_callbacks: list = []

    async def initialize(self) -> bool:
        """Initialize the orchestration module."""
        try:
            # Initialize daemon
            self._daemon = get_daemon(
                pid_file=self.pid_file,
                log_file=self.log_file,
            )

            # Check for existing daemon
            existing_pid = PythonDaemon.check_existing_pid(self.pid_file)
            if existing_pid:
                logger.warning(f"Existing daemon found with PID {existing_pid}")

            # Initialize health monitor
            self._health_monitor = await initialize_health_monitor()

            # Setup ZMQ for IPC
            self._zmq_context = zmq.Context()
            self._zmq_socket = self._zmq_context.socket(zmq.REP)
            self._zmq_socket.bind(self.ipc_endpoint)

            # Register health callback
            self._health_monitor.register_callback(self._on_health_change)

            logger.info(f"Orchestration module initialized, listening on {self.ipc_endpoint}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize orchestration: {e}")
            return False

    async def start_listening(self):
        """Start listening for IPC commands."""
        if self._running:
            return

        self._running = True
        logger.info("Starting IPC command listener...")

        while self._running:
            try:
                # Wait for command (with timeout)
                message = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._zmq_socket.poll(timeout=1000),
                )

                if message:
                    command = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self._zmq_socket.recv_string(),
                    )
                    await self._handle_command(command)

            except Exception as e:
                logger.error(f"IPC listener error: {e}")
                await asyncio.sleep(1.0)

    async def _handle_command(self, command: str):
        """Handle incoming IPC command."""
        try:
            cmd_data = json.loads(command) if command.startswith("{") else {"cmd": command}
            cmd = cmd_data.get("cmd", command).upper()

            logger.info(f"Received command: {cmd}")

            response = {"status": "ok", "cmd": cmd}

            if cmd == "START":
                if not self._started:
                    await self._execute_start()
                else:
                    response["status"] = "already_started"

            elif cmd == "KILL":
                await self._execute_kill(cmd_data.get("reason", "unknown"))

            elif cmd == "STATUS":
                response["data"] = self.get_status()

            elif cmd == "HEALTH":
                metrics = self._health_monitor.get_current_metrics()
                response["data"] = {
                    "status": metrics.status.value if metrics else "unknown",
                } if metrics else {"status": "unknown"}

            else:
                response["status"] = "unknown_command"

            # Send response
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._zmq_socket.send_string(json.dumps(response)),
            )

        except Exception as e:
            logger.error(f"Command handling error: {e}")
            try:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._zmq_socket.send_string(json.dumps({
                        "status": "error",
                        "error": str(e),
                    })),
                )
            except Exception:
                pass

    async def _execute_start(self):
        """Execute START command - launch all subsystems."""
        logger.info("Executing START command...")

        self._started = True

        # Call registered callbacks
        for callback in self._start_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback()
                else:
                    callback()
            except Exception as e:
                logger.error(f"Start callback error: {e}")

        logger.info("START command completed")

    async def _execute_kill(self, reason: str = "unknown"):
        """Execute KILL command - graceful shutdown."""
        logger.critical(f"Executing KILL command (reason: {reason})")

        # Call kill callbacks first
        for callback in reversed(self._kill_callbacks):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback()
                else:
                    callback()
            except Exception as e:
                logger.error(f"Kill callback error: {e}")

        # Stop health monitor
        if self._health_monitor:
            self._health_monitor.stop()

        # Shutdown daemon
        if self._daemon:
            self._daemon.shutdown()

        self._running = False
        self._started = False

        logger.info("KILL command completed - shutdown complete")

    def _on_health_change(self, metrics: HealthMetrics):
        """Handle health status changes."""
        if metrics.status == HealthStatus.CRITICAL:
            logger.critical(f"Critical health status detected!")
            # Could auto-trigger kill here if configured

    def register_start_callback(self, callback: Callable):
        """Register a callback for START command."""
        self._start_callbacks.append(callback)

    def register_kill_callback(self, callback: Callable):
        """Register a callback for KILL command."""
        self._kill_callbacks.append(callback)

    def get_status(self) -> Dict[str, Any]:
        """Get current orchestration status."""
        return {
            "running": self._running,
            "started": self._started,
            "daemon_active": self._daemon.is_running() if self._daemon else False,
            "health_healthy": self._health_monitor.is_healthy() if self._health_monitor else False,
            "ipc_endpoint": self.ipc_endpoint,
        }

    async def shutdown(self):
        """Gracefully shutdown the orchestration module."""
        logger.info("Shutting down orchestration module...")

        self._running = False

        if self._health_monitor:
            await shutdown_health_monitor()

        if self._daemon:
            await shutdown_daemon_module()

        if self._zmq_socket:
            self._zmq_socket.close()
        if self._zmq_context:
            self._zmq_context.term()

        logger.info("Orchestration module shutdown complete")


# Module singleton
_module: Optional[OrchestrationModule] = None


def get_orchestration_module(
    ipc_endpoint: str = "tcp://localhost:5556",
) -> OrchestrationModule:
    """Get or create the orchestration module singleton."""
    global _module
    if _module is None:
        _module = OrchestrationModule(ipc_endpoint=ipc_endpoint)
    return _module


async def initialize_orchestration(
    ipc_endpoint: str = "tcp://localhost:5556",
) -> OrchestrationModule:
    """Initialize the orchestration module."""
    module = get_orchestration_module(ipc_endpoint=ipc_endpoint)
    await module.initialize()
    return module


async def start_orchestration_listener():
    """Start the orchestration command listener."""
    module = get_orchestration_module()
    await module.start_listening()


async def shutdown_orchestration_module():
    """Gracefully shutdown the orchestration module."""
    global _module
    if _module:
        await _module.shutdown()
        _module = None
