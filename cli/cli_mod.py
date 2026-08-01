#!/usr/bin/env python3
"""
CLI Module Root - Stage 50
Wires CLI inputs to master orchestrator via local sockets and shared memory flags.
"""

import os
import sys
import signal
import logging
from datetime import datetime
from typing import Optional, Dict, Any, Callable
from pathlib import Path
import threading
import queue
import zmq
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
)
logger = logging.getLogger('CLIMod')

# Constants
CLI_SOCKET_DIR = Path('/tmp/crypto_bot_cli')
CTRL_SOCKET_PATH = CLI_SOCKET_DIR / 'control.ipc'
STATUS_SOCKET_PATH = CLI_SOCKET_DIR / 'status.ipc'
SHM_FLAG_PATH = CLI_SOCKET_DIR / 'cli_flag.shm'

ZMQ_CTRL_PORT = 5558
ZMQ_STATUS_PORT = 5559
ZMQ_KILL_PORT = 5557


class CLISocketManager:
    """Manages IPC sockets for CLI communication."""
    
    def __init__(self):
        self.context = zmq.Context()
        self.ctrl_socket: Optional[zmq.Socket] = None
        self.status_socket: Optional[zmq.Socket] = None
        self.running = False
    
    def setup_server(self):
        """Setup server-side sockets for receiving CLI commands."""
        # Ensure socket directory exists
        CLI_SOCKET_DIR.mkdir(parents=True, exist_ok=True)
        
        # Control socket (REP pattern)
        self.ctrl_socket = self.context.socket(zmq.REP)
        self.ctrl_socket.bind(f"ipc://{CTRL_SOCKET_PATH}")
        logger.info(f"Control socket bound: ipc://{CTRL_SOCKET_PATH}")
        
        # Status socket (PUB pattern)
        self.status_socket = self.context.socket(zmq.PUB)
        self.status_socket.bind(f"ipc://{STATUS_SOCKET_PATH}")
        logger.info(f"Status socket bound: ipc://{STATUS_SOCKET_PATH}")
        
        # Also bind TCP for remote CLI access
        self.ctrl_socket.bind(f"tcp://*:{ZMQ_CTRL_PORT}")
        self.status_socket.bind(f"tcp://*:{ZMQ_STATUS_PORT}")
        
        self.running = True
    
    def setup_client(self) -> bool:
        """Setup client-side sockets for sending CLI commands."""
        try:
            self.ctrl_socket = self.context.socket(zmq.REQ)
            self.ctrl_socket.setsockopt(zmq.LINGER, 2000)
            self.ctrl_socket.connect(f"ipc://{CTRL_SOCKET_PATH}")
            
            self.status_socket = self.context.socket(zmq.SUB)
            self.status_socket.setsockopt(zmq.SUBSCRIBE, b"")
            self.status_socket.connect(f"ipc://{STATUS_SOCKET_PATH}")
            
            # Fallback to TCP if IPC fails
            if not self._test_connection():
                logger.warning("IPC connection failed, trying TCP fallback")
                self.ctrl_socket.disconnect(f"ipc://{CTRL_SOCKET_PATH}")
                self.status_socket.disconnect(f"ipc://{STATUS_SOCKET_PATH}")
                self.ctrl_socket.connect(f"tcp://localhost:{ZMQ_CTRL_PORT}")
                self.status_socket.connect(f"tcp://localhost:{ZMQ_STATUS_PORT}")
            
            return self._test_connection()
        except Exception as e:
            logger.error(f"Failed to setup client sockets: {e}")
            return False
    
    def _test_connection(self) -> bool:
        """Test if sockets are connected."""
        try:
            poller = zmq.Poller()
            poller.register(self.ctrl_socket, zmq.POLLOUT)
            socks = dict(poller.poll(timeout=1000))
            return self.ctrl_socket in socks
        except:
            return False
    
    def receive_command(self, timeout_ms: int = 1000) -> Optional[Dict]:
        """Receive a command from CLI (server side)."""
        if not self.ctrl_socket:
            return None
        
        poller = zmq.Poller()
        poller.register(self.ctrl_socket, zmq.POLLIN)
        socks = dict(poller.poll(timeout=timeout_ms))
        
        if self.ctrl_socket in socks:
            try:
                message = self.ctrl_socket.recv_json(flags=zmq.NOBLOCK)
                return message
            except Exception as e:
                logger.error(f"Error receiving command: {e}")
                return None
        return None
    
    def send_response(self, response: Dict):
        """Send response to CLI command (server side)."""
        if self.ctrl_socket:
            try:
                self.ctrl_socket.send_json(response)
            except Exception as e:
                logger.error(f"Error sending response: {e}")
    
    def publish_status(self, status: Dict):
        """Publish status update to subscribers (server side)."""
        if self.status_socket:
            try:
                self.status_socket.send_json(status, flags=zmq.NOBLOCK)
            except Exception as e:
                pass  # Silent fail for status updates
    
    def send_command(self, command: str, payload: Dict = None, timeout_ms: int = 5000) -> Optional[Dict]:
        """Send command and wait for response (client side)."""
        if not self.ctrl_socket:
            return None
        
        message = {
            'command': command,
            'timestamp': datetime.now().isoformat(),
            'payload': payload or {}
        }
        
        try:
            self.ctrl_socket.send_json(message)
            
            poller = zmq.Poller()
            poller.register(self.ctrl_socket, zmq.POLLIN)
            socks = dict(poller.poll(timeout=timeout_ms))
            
            if self.ctrl_socket in socks:
                return self.ctrl_socket.recv_json()
            else:
                logger.warning("Command timed out")
                return {'status': 'error', 'message': 'Timeout'}
        except Exception as e:
            logger.error(f"Error sending command: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def close(self):
        """Close all sockets."""
        self.running = False
        if self.ctrl_socket:
            self.ctrl_socket.close()
        if self.status_socket:
            self.status_socket.close()
        self.context.term()
        
        # Cleanup IPC files
        for path in [CTRL_SOCKET_PATH, STATUS_SOCKET_PATH]:
            try:
                if path.exists():
                    path.unlink()
            except:
                pass


class SharedMemoryFlag:
    """Simple file-based shared memory flag for CLI signaling."""
    
    def __init__(self):
        self.flag_path = SHM_FLAG_PATH
        self.lock_path = CLI_SOCKET_DIR / 'cli_flag.lock'
    
    def set_flag(self, flag_name: str, value: Any):
        """Set a flag value."""
        CLI_SOCKET_DIR.mkdir(parents=True, exist_ok=True)
        
        flag_data = {
            'name': flag_name,
            'value': value,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            self.flag_path.write_text(json.dumps(flag_data))
        except Exception as e:
            logger.error(f"Failed to write flag: {e}")
    
    def get_flag(self, flag_name: str) -> Optional[Any]:
        """Get a flag value."""
        try:
            if not self.flag_path.exists():
                return None
            
            data = json.loads(self.flag_path.read_text())
            if data.get('name') == flag_name:
                return data.get('value')
        except Exception as e:
            logger.warning(f"Failed to read flag: {e}")
        
        return None
    
    def clear_flag(self):
        """Clear the flag."""
        try:
            if self.flag_path.exists():
                self.flag_path.unlink()
        except:
            pass


class CommandRouter:
    """Routes CLI commands to appropriate handlers."""
    
    def __init__(self):
        self.handlers: Dict[str, Callable] = {}
        self.socket_manager = CLISocketManager()
        self.shared_flag = SharedMemoryFlag()
        self.command_queue = queue.Queue()
        self.response_cache: Dict[str, Dict] = {}
        self.cache_ttl_seconds = 5
    
    def register_handler(self, command: str, handler: Callable):
        """Register a command handler."""
        self.handlers[command.upper()] = handler
        logger.info(f"Registered handler for command: {command}")
    
    def route_command(self, command: str, payload: Dict = None) -> Dict:
        """Route a command to its handler."""
        cmd = command.upper()
        
        if cmd in self.handlers:
            try:
                result = self.handlers[cmd](payload or {})
                return {
                    'status': 'ok',
                    'command': cmd,
                    'result': result
                }
            except Exception as e:
                logger.error(f"Handler error for {cmd}: {e}")
                return {
                    'status': 'error',
                    'command': cmd,
                    'message': str(e)
                }
        else:
            return {
                'status': 'error',
                'command': cmd,
                'message': f'Unknown command: {cmd}'
            }
    
    def run_server(self):
        """Run command router as server."""
        self.socket_manager.setup_server()
        logger.info("CLI command server started")
        
        while self.socket_manager.running:
            message = self.socket_manager.receive_command(timeout_ms=1000)
            
            if message:
                command = message.get('command', message.get('type', ''))
                payload = message.get('payload', {})
                
                logger.info(f"Received command: {command}")
                response = self.route_command(command, payload)
                
                self.socket_manager.send_response(response)
        
        self.socket_manager.close()
    
    def close(self):
        """Shutdown router."""
        self.socket_manager.close()


class CLIModuleCoordinator:
    """Coordinates CLI module operations."""
    
    def __init__(self):
        self.router = CommandRouter()
        self.shared_flag = SharedMemoryFlag()
        self._setup_default_handlers()
    
    def _setup_default_handlers(self):
        """Setup default command handlers."""
        self.router.register_handler('START', self._handle_start)
        self.router.register_handler('KILL', self._handle_kill)
        self.router.register_handler('STATUS', self._handle_status)
        self.router.register_handler('PING', self._handle_ping)
    
    def _handle_start(self, payload: Dict) -> Dict:
        """Handle START command."""
        # Set shared memory flag for orchestrator
        self.shared_flag.set_flag('cli_request', {
            'type': 'START',
            'requested_at': datetime.now().isoformat()
        })
        
        return {
            'message': 'Start request queued',
            'flag_set': True
        }
    
    def _handle_kill(self, payload: Dict) -> Dict:
        """Handle KILL command - routes to Rust kill switch."""
        reason = payload.get('reason', 'CLI requested')
        
        # Send to kill switch via ZMQ
        context = zmq.Context()
        try:
            kill_socket = context.socket(zmq.PUSH)
            kill_socket.setsockopt(zmq.LINGER, 100)
            kill_socket.connect(f"tcp://localhost:{ZMQ_KILL_PORT}")
            
            kill_message = {
                'type': 'CLI_KILL',
                'reason': reason,
                'timestamp': datetime.now().isoformat()
            }
            kill_socket.send_json(kill_message, flags=zmq.NOBLOCK)
            kill_socket.close()
            
            return {
                'message': 'Kill signal sent to Rust core',
                'reason': reason
            }
        except Exception as e:
            context.term()
            return {
                'status': 'error',
                'message': f'Failed to send kill signal: {e}'
            }
        finally:
            context.term()
    
    def _handle_status(self, payload: Dict) -> Dict:
        """Handle STATUS command."""
        return {
            'state': 'RUNNING',
            'uptime': 'N/A',
            'cli_connected': True,
            'timestamp': datetime.now().isoformat()
        }
    
    def _handle_ping(self, payload: Dict) -> Dict:
        """Handle PING command."""
        return {
            'pong': True,
            'latency_ms': 0,
            'timestamp': datetime.now().isoformat()
        }
    
    def run(self, mode: str = 'server'):
        """Run CLI module."""
        if mode == 'server':
            try:
                self.router.run_server()
            except KeyboardInterrupt:
                logger.info("Server interrupted")
            finally:
                self.router.close()
        else:
            logger.info("CLI module ready for client operations")


def create_cli_coordinator() -> CLIModuleCoordinator:
    """Factory function to create CLI coordinator."""
    return CLIModuleCoordinator()


def main():
    """Entry point for CLI module."""
    import argparse
    
    parser = argparse.ArgumentParser(description='CLI Module')
    parser.add_argument('--mode', choices=['server', 'client'], default='server',
                       help='Run mode: server or client')
    args = parser.parse_args()
    
    coordinator = create_cli_coordinator()
    coordinator.run(mode=args.mode)


if __name__ == '__main__':
    main()
