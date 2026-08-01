"""
ZeroMQ Pusher - High-throughput, non-blocking ZMQ PUSH socket for UI telemetry.
Streams metrics to Rust ratatui frontend with frame dropping if TUI falls behind.
Prevents Python queue bloat using zmq.NOBLOCK.
"""
import numpy as np
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import time
import threading
import json

logger = logging.getLogger(__name__)

# Try to import zmq
try:
    import zmq
    ZMQ_AVAILABLE = True
except ImportError:
    ZMQ_AVAILABLE = False
    logger.warning("ZeroMQ not available, UI streaming disabled")


class ZmqPusher:
    """
    High-throughput ZeroMQ PUSH socket for streaming UI telemetry.
    Uses non-blocking sends with frame dropping to prevent backpressure.
    """
    
    def __init__(self, endpoint: str = 'tcp://127.0.0.1:5555',
                 high_water_mark: int = 1000,
                 max_queue_size: int = 100):
        """
        Initialize ZMQ pusher.
        
        Args:
            endpoint: ZMQ endpoint (tcp:// or ipc://)
            high_water_mark: ZMQ SNDHWM for backpressure
            max_queue_size: Max frames to queue before dropping
        """
        self.endpoint = endpoint
        self.high_water_mark = high_water_mark
        self.max_queue_size = max_queue_size
        
        self._context = None
        self._socket = None
        self._connected = False
        
        # Statistics
        self._frames_sent = 0
        self._frames_dropped = 0
        self._last_send_time = 0.0
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Rate limiting
        self._min_interval = 0.0  # Minimum seconds between sends
        self._last_send = 0.0
        
        if ZMQ_AVAILABLE:
            self._init_zmq()
        
        logger.info(f"ZmqPusher initialized: {endpoint}")
    
    def _init_zmq(self) -> None:
        """Initialize ZeroMQ context and socket."""
        try:
            self._context = zmq.Context.instance()
            self._socket = self._context.socket(zmq.PUSH)
            
            # Configure socket for non-blocking operation
            self._socket.setsockopt(zmq.SNDHWM, self.high_water_mark)
            self._socket.setsockopt(zmq.LINGER, 0)  # Don't block on close
            
            # Connect to endpoint
            self._socket.connect(self.endpoint)
            self._connected = True
            
            logger.info(f"ZMQ PUSH socket connected to {self.endpoint}")
        except Exception as e:
            logger.error(f"Failed to initialize ZMQ: {e}")
            self._connected = False
    
    def send(self, data: Dict[str, Any], drop_if_busy: bool = True) -> bool:
        """
        Send data frame to UI.
        
        Args:
            data: Data dictionary to send
            drop_if_busy: Drop frame if socket is busy (non-blocking)
            
        Returns:
            Success status
        """
        if not ZMQ_AVAILABLE or not self._connected:
            return False
        
        # Serialize to JSON
        try:
            payload = json.dumps(data, separators=(',', ':')).encode('utf-8')
        except Exception as e:
            logger.error(f"Failed to serialize data: {e}")
            return False
        
        # Check rate limit
        current_time = time.time()
        if current_time - self._last_send < self._min_interval:
            if drop_if_busy:
                self._frames_dropped += 1
                return True  # Silently drop
            else:
                time.sleep(self._min_interval - (current_time - self._last_send))
        
        # Check queue size (approximate via HWM)
        with self._lock:
            try:
                # Non-blocking send with DONTWAIT
                flags = zmq.NOBLOCK if drop_if_busy else 0
                
                self._socket.send(payload, flags=flags)
                
                self._frames_sent += 1
                self._last_send_time = time.perf_counter()
                self._last_send = current_time
                
                return True
                
            except zmq.Again:
                # Socket buffer full, drop frame
                if drop_if_busy:
                    self._frames_dropped += 1
                    logger.debug("Frame dropped (TUI falling behind)")
                    return True  # Intentional drop, not an error
                else:
                    logger.warning("ZMQ send blocked")
                    return False
                    
            except Exception as e:
                logger.error(f"ZMQ send failed: {e}")
                self._connected = False
                return False
    
    def send_metrics(self, metrics: Dict[str, Any]) -> bool:
        """
        Send metrics payload with standard envelope.
        
        Args:
            metrics: Metrics dictionary
            
        Returns:
            Success status
        """
        envelope = {
            'type': 'metrics',
            'timestamp': time.time(),
            'data': metrics
        }
        
        return self.send(envelope)
    
    def send_alert(self, alert_type: str, message: str, 
                   severity: str = 'info') -> bool:
        """
        Send alert to UI.
        
        Args:
            alert_type: Type of alert
            message: Alert message
            severity: Severity level (info/warning/error/critical)
            
        Returns:
            Success status
        """
        envelope = {
            'type': 'alert',
            'timestamp': time.time(),
            'severity': severity,
            'data': {
                'alert_type': alert_type,
                'message': message
            }
        }
        
        return self.send(envelope)
    
    def send_state_update(self, state: Dict[str, Any]) -> bool:
        """
        Send system state update.
        
        Args:
            state: State dictionary
            
        Returns:
            Success status
        """
        envelope = {
            'type': 'state',
            'timestamp': time.time(),
            'data': state
        }
        
        return self.send(envelope)
    
    def set_rate_limit(self, fps: float) -> None:
        """
        Set maximum frame rate.
        
        Args:
            fps: Maximum frames per second
        """
        self._min_interval = 1.0 / max(1, fps)
        logger.info(f"ZMQ rate limit set to {fps} FPS")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pusher statistics."""
        total_frames = self._frames_sent + self._frames_dropped
        drop_rate = self._frames_dropped / max(1, total_frames)
        
        return {
            'connected': self._connected,
            'endpoint': self.endpoint,
            'frames_sent': self._frames_sent,
            'frames_dropped': self._frames_dropped,
            'drop_rate': drop_rate,
            'last_send_latency_ms': (
                (time.perf_counter() - self._last_send_time) * 1000 
                if self._last_send_time > 0 else 0
            ),
            'rate_limit_fps': 1.0 / self._min_interval if self._min_interval > 0 else float('inf')
        }
    
    def reconnect(self) -> bool:
        """Attempt to reconnect to endpoint."""
        if self._socket:
            try:
                self._socket.close(linger=0)
            except Exception:
                pass
        
        self._socket = None
        self._connected = False
        
        self._init_zmq()
        return self._connected
    
    def flush(self) -> None:
        """Flush any pending messages (best effort)."""
        if self._socket and self._connected:
            try:
                self._socket.flush()
            except Exception:
                pass
    
    def close(self) -> None:
        """Close ZMQ socket and context."""
        self.flush()
        
        if self._socket:
            try:
                self._socket.close(linger=0)
            except Exception:
                pass
            self._socket = None
        
        if self._context:
            # Don't destroy shared context
            # self._context.term()
            pass
        
        self._connected = False
        logger.info(f"ZmqPusher closed. Sent: {self._frames_sent}, "
                   f"Dropped: {self._frames_dropped}")


class TelemetryStreamer:
    """
    High-level telemetry streamer that coordinates metrics aggregation and ZMQ pushing.
    """
    
    def __init__(self, endpoint: str = 'tcp://127.0.0.1:5555',
                 push_interval: float = 0.1):
        """
        Initialize telemetry streamer.
        
        Args:
            endpoint: ZMQ endpoint
            push_interval: Seconds between automatic pushes
        """
        self.push_interval = push_interval
        
        # Initialize ZMQ pusher
        self.pusher = ZmqPusher(endpoint=endpoint)
        
        # Streaming state
        self._streaming = False
        self._stream_thread: Optional[threading.Thread] = None
        
        # Pending data
        self._pending_metrics: Optional[Dict] = None
        self._lock = threading.Lock()
        
        logger.info(f"TelemetryStreamer initialized: {endpoint}")
    
    def start_streaming(self) -> bool:
        """Start background streaming thread."""
        if self._streaming:
            return True
        
        if not self.pusher._connected:
            logger.warning("Cannot start streaming: ZMQ not connected")
            return False
        
        self._streaming = True
        self._stream_thread = threading.Thread(
            target=self._stream_loop,
            daemon=True
        )
        self._stream_thread.start()
        
        logger.info("Telemetry streaming started")
        return True
    
    def stop_streaming(self) -> None:
        """Stop background streaming."""
        self._streaming = False
        
        if self._stream_thread:
            self._stream_thread.join(timeout=2.0)
            self._stream_thread = None
        
        logger.info("Telemetry streaming stopped")
    
    def _stream_loop(self) -> None:
        """Background streaming loop."""
        last_push = 0.0
        
        while self._streaming:
            current_time = time.time()
            
            if current_time - last_push >= self.push_interval:
                self._push_pending()
                last_push = current_time
            
            time.sleep(0.01)  # Small sleep to prevent CPU spinning
    
    def _push_pending(self) -> None:
        """Push pending metrics."""
        with self._lock:
            if self._pending_metrics:
                self.pusher.send_metrics(self._pending_metrics)
                self._pending_metrics = None
    
    def update_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Update metrics to be streamed.
        
        Args:
            metrics: New metrics dictionary
        """
        with self._lock:
            self._pending_metrics = metrics
    
    def send_immediate(self, data: Dict[str, Any]) -> bool:
        """
        Send data immediately (bypass queue).
        
        Args:
            data: Data to send
            
        Returns:
            Success status
        """
        return self.pusher.send(data)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get streamer statistics."""
        return {
            'streaming': self._streaming,
            'push_interval': self.push_interval,
            'has_pending': self._pending_metrics is not None,
            'pusher_stats': self.pusher.get_statistics()
        }
    
    def close(self) -> None:
        """Clean up resources."""
        self.stop_streaming()
        self.pusher.close()
        logger.info("TelemetryStreamer closed")


# Singleton instance
_zmq_pusher: Optional[ZmqPusher] = None
_telemetry_streamer: Optional[TelemetryStreamer] = None


def get_zmq_pusher(config: Optional[Dict[str, Any]] = None) -> ZmqPusher:
    """Get or create singleton ZmqPusher instance."""
    global _zmq_pusher
    if _zmq_pusher is None:
        config = config or {}
        _zmq_pusher = ZmqPusher(
            endpoint=config.get('endpoint', 'tcp://127.0.0.1:5555'),
            high_water_mark=config.get('high_water_mark', 1000),
            max_queue_size=config.get('max_queue_size', 100)
        )
    return _zmq_pusher


def get_telemetry_streamer(config: Optional[Dict[str, Any]] = None) -> TelemetryStreamer:
    """Get or create singleton TelemetryStreamer instance."""
    global _telemetry_streamer
    if _telemetry_streamer is None:
        config = config or {}
        _telemetry_streamer = TelemetryStreamer(
            endpoint=config.get('endpoint', 'tcp://127.0.0.1:5555'),
            push_interval=config.get('push_interval', 0.1)
        )
    return _telemetry_streamer


def reset_ui_components() -> None:
    """Reset all UI components."""
    global _zmq_pusher, _telemetry_streamer
    
    if _telemetry_streamer:
        _telemetry_streamer.close()
    if _zmq_pusher:
        _zmq_pusher.close()
    
    _zmq_pusher = None
    _telemetry_streamer = None


__all__ = [
    'ZmqPusher',
    'TelemetryStreamer',
    'get_zmq_pusher',
    'get_telemetry_streamer',
    'reset_ui_components'
]
