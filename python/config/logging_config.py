"""
High-performance asynchronous logging configuration.
Routes logs to a background thread via bounded queue to prevent GIL blocking.
"""

import logging
import logging.handlers
import threading
import queue
import sys
from typing import Optional


class AsyncLogHandler(logging.Handler):
    """
    Asynchronous log handler that writes logs in a background thread.
    Uses a bounded queue to prevent memory buildup and GIL contention.
    """
    
    def __init__(self, max_queue_size: int = 1000, flush_interval: float = 1.0):
        super().__init__()
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._shutdown_event = threading.Event()
        self._worker_thread = threading.Thread(target=self._process_logs, daemon=True)
        self._worker_thread.start()
        self._flush_interval = flush_interval
    
    def _process_logs(self) -> None:
        """Background thread that processes log records from the queue."""
        while not self._shutdown_event.is_set():
            try:
                # Use timeout to allow periodic shutdown checks
                record = self._queue.get(timeout=self._flush_interval)
                if record is None:
                    # Sentinel value to signal shutdown
                    break
                
                # Format and emit the log record
                msg = self.format(record)
                sys.stderr.write(msg + "\n")
                sys.stderr.flush()
                self._queue.task_done()
            except queue.Empty:
                continue
            except Exception:
                # Prevent worker thread from dying on exception
                self.handleError(None)
    
    def emit(self, record: logging.LogRecord) -> None:
        """Queue the log record for async processing."""
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            # Drop log if queue is full to prevent blocking
            pass
    
    def close(self) -> None:
        """Shutdown the background worker thread."""
        self._shutdown_event.set()
        # Send sentinel to wake up worker
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        self._worker_thread.join(timeout=2.0)
        super().close()


def setup_logging(log_level: str = "WARNING", log_file: Optional[str] = None) -> logging.Logger:
    """
    Configure high-performance logging with async handler.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for log output
    
    Returns:
        Configured logger instance
    """
    # Create root logger
    logger = logging.getLogger("hft_nautilus_ml")
    logger.setLevel(getattr(logging, log_level.upper(), logging.WARNING))
    
    # Remove any existing handlers
    logger.handlers.clear()
    
    # Create formatter with minimal overhead
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Create async handler for console output
    async_handler = AsyncLogHandler(max_queue_size=500, flush_interval=0.5)
    async_handler.setFormatter(formatter)
    logger.addHandler(async_handler)
    
    # Optional file handler (synchronous for reliability)
    if log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=3,
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a child logger with the specified name."""
    return logging.getLogger(f"hft_nautilus_ml.{name}")
