"""
High-throughput ZeroMQ SUB socket listener for execution signals.
Receives ultra-low latency signals and state syncs from Rust.
"""

import zmq
import zmq.asyncio
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, Callable, List
import sys
import json

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import (
    ZMQ_SIGNAL_PORT,
    ZMQ_STATE_PORT,
    ZMQ_HOST,
    get_logger,
)

logger = get_logger("signal_receiver")


class SignalReceiver:
    """
    High-throughput ZeroMQ subscriber for receiving signals from Rust.
    Uses async I/O for minimal latency and maximum throughput.
    """
    
    def __init__(
        self,
        signal_port: Optional[int] = None,
        state_port: Optional[int] = None,
        host: Optional[str] = None,
    ):
        self.host = host or ZMQ_HOST
        self.signal_port = signal_port or ZMQ_SIGNAL_PORT
        self.state_port = state_port or ZMQ_STATE_PORT
        
        self._context: Optional[zmq.asyncio.Context] = None
        self._signal_socket: Optional[zmq.asyncio.Socket] = None
        self._state_socket: Optional[zmq.asyncio.Socket] = None
        
        self._running = False
        self._signal_handlers: List[Callable] = []
        self._state_handlers: List[Callable] = []
        
        # Statistics
        self._messages_received = 0
        self._last_message_time: Optional[float] = None
    
    async def connect(self) -> bool:
        """
        Connect to Rust ZeroMQ publishers.
        
        Returns:
            True if successfully connected
        """
        try:
            # Create async context
            self._context = zmq.asyncio.Context()
            
            # Create signal subscription socket
            self._signal_socket = self._context.socket(zmq.SUB)
            self._signal_socket.setsockopt(zmq.SUBSCRIBE, b"")  # Subscribe to all
            self._signal_socket.setsockopt(zmq.RCVHWM, 10000)  # High water mark
            self._signal_socket.setsockopt(zmq.RCVBUF, 1024 * 1024)  # 1MB receive buffer
            self._signal_socket.connect(f"tcp://{self.host}:{self.signal_port}")
            
            # Create state subscription socket
            self._state_socket = self._context.socket(zmq.SUB)
            self._state_socket.setsockopt(zmq.SUBSCRIBE, b"")  # Subscribe to all
            self._state_socket.setsockopt(zmq.RCVHWM, 10000)
            self._state_socket.setsockopt(zmq.RCVBUF, 1024 * 1024)
            self._state_socket.connect(f"tcp://{self.host}:{self.state_port}")
            
            logger.info(
                f"Connected to ZMQ publishers: "
                f"signals=tcp://{self.host}:{self.signal_port}, "
                f"state=tcp://{self.host}:{self.state_port}"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to ZMQ publishers: {e}")
            await self.disconnect()
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from ZeroMQ publishers."""
        self._running = False
        
        if self._signal_socket:
            try:
                self._signal_socket.close(linger=0)
            except Exception:
                pass
            self._signal_socket = None
        
        if self._state_socket:
            try:
                self._state_socket.close(linger=0)
            except Exception:
                pass
            self._state_socket = None
        
        if self._context:
            try:
                self._context.term()
            except Exception:
                pass
            self._context = None
        
        logger.info("Disconnected from ZMQ publishers")
    
    def register_signal_handler(self, handler: Callable) -> None:
        """Register a handler for execution signals."""
        self._signal_handlers.append(handler)
        logger.debug(f"Registered signal handler: {handler.__name__}")
    
    def register_state_handler(self, handler: Callable) -> None:
        """Register a handler for state sync messages."""
        self._state_handlers.append(handler)
        logger.debug(f"Registered state handler: {handler.__name__}")
    
    async def receive_signals(self) -> None:
        """
        Continuously receive and process execution signals.
        Runs until stopped.
        """
        if not self._signal_socket:
            logger.error("Signal socket not connected")
            return
        
        self._running = True
        logger.info("Starting signal receiver loop")
        
        while self._running:
            try:
                # Receive signal message with timeout
                message = await asyncio.wait_for(
                    self._signal_socket.recv_multipart(),
                    timeout=1.0,
                )
                
                self._messages_received += 1
                self._last_message_time = asyncio.get_event_loop().time()
                
                # Parse and dispatch to handlers
                signal_data = self._parse_signal(message)
                
                for handler in self._signal_handlers:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(signal_data)
                        else:
                            handler(signal_data)
                    except Exception as e:
                        logger.error(f"Signal handler error: {e}")
                
            except asyncio.TimeoutError:
                continue
            except zmq.ZMQError as e:
                if self._running:
                    logger.error(f"ZMQ error: {e}")
                break
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Unexpected error in signal receiver: {e}")
        
        logger.info("Signal receiver loop stopped")
    
    async def receive_state_syncs(self) -> None:
        """
        Continuously receive and process state synchronization messages.
        Runs until stopped.
        """
        if not self._state_socket:
            logger.error("State socket not connected")
            return
        
        logger.info("Starting state sync receiver loop")
        
        while self._running:
            try:
                # Receive state message with timeout
                message = await asyncio.wait_for(
                    self._state_socket.recv_multipart(),
                    timeout=1.0,
                )
                
                # Parse and dispatch to handlers
                state_data = self._parse_state(message)
                
                for handler in self._state_handlers:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(state_data)
                        else:
                            handler(state_data)
                    except Exception as e:
                        logger.error(f"State handler error: {e}")
                
            except asyncio.TimeoutError:
                continue
            except zmq.ZMQError as e:
                if self._running:
                    logger.error(f"ZMQ error: {e}")
                break
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Unexpected error in state receiver: {e}")
        
        logger.info("State sync receiver loop stopped")
    
    def _parse_signal(self, message: list) -> Dict[str, Any]:
        """Parse a raw ZMQ message into a signal dictionary."""
        try:
            if len(message) >= 2:
                topic = message[0].decode('utf-8')
                data = message[1]
                
                # Try JSON parsing first
                try:
                    payload = json.loads(data.decode('utf-8'))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    # Fall back to binary interpretation
                    payload = {"binary": data.hex()}
                
                return {
                    "topic": topic,
                    "payload": payload,
                    "timestamp": self._last_message_time,
                }
            else:
                return {"raw": [m.hex() if isinstance(m, bytes) else m for m in message]}
        except Exception as e:
            logger.error(f"Error parsing signal: {e}")
            return {"error": str(e)}
    
    def _parse_state(self, message: list) -> Dict[str, Any]:
        """Parse a raw ZMQ message into a state dictionary."""
        try:
            if len(message) >= 2:
                topic = message[0].decode('utf-8')
                data = message[1]
                
                try:
                    payload = json.loads(data.decode('utf-8'))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    payload = {"binary": data.hex()}
                
                return {
                    "topic": topic,
                    "payload": payload,
                    "timestamp": self._last_message_time,
                }
            else:
                return {"raw": [m.hex() if isinstance(m, bytes) else m for m in message]}
        except Exception as e:
            logger.error(f"Error parsing state: {e}")
            return {"error": str(e)}
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get receiver statistics."""
        return {
            "messages_received": self._messages_received,
            "last_message_time": self._last_message_time,
            "is_running": self._running,
            "signal_handlers": len(self._signal_handlers),
            "state_handlers": len(self._state_handlers),
        }


async def run_signal_receiver() -> SignalReceiver:
    """
    Convenience function to create and run a signal receiver.
    
    Returns:
        Running SignalReceiver instance
    """
    receiver = SignalReceiver()
    
    if not await receiver.connect():
        raise RuntimeError("Failed to connect signal receiver")
    
    # Start receivers as background tasks
    loop = asyncio.get_event_loop()
    loop.create_task(receiver.receive_signals())
    loop.create_task(receiver.receive_state_syncs())
    
    logger.info("Signal receiver started")
    return receiver
