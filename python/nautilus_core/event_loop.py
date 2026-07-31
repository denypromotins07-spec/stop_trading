"""
Event loop configuration with uvloop injection.
Drastically reduces Python network I/O latency and GIL contention.
"""

import asyncio
from pathlib import Path
from typing import Optional, Callable, Any, Coroutine
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import get_logger

logger = get_logger("event_loop")


class UVLoopEventLoop:
    """
    Wrapper for uvloop-based event loop with HFT optimizations.
    Provides ultra-low latency async execution for Nautilus Trader.
    """
    
    def __init__(self):
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._uvloop_installed = False
        self._task_count = 0
        self._start_time: Optional[float] = None
    
    def install_uvloop(self) -> bool:
        """
        Install uvloop as the default event loop policy.
        
        Returns:
            True if successfully installed
        """
        if self._uvloop_installed:
            return True
        
        try:
            import uvloop
            asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
            self._uvloop_installed = True
            logger.info("uvloop event loop policy installed successfully")
            return True
        except ImportError as e:
            logger.error(f"Failed to import uvloop: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to install uvloop: {e}")
            return False
    
    def get_loop(self) -> asyncio.AbstractEventLoop:
        """
        Get or create the current event loop.
        
        Returns:
            The current asyncio event loop
        """
        if not self._uvloop_installed:
            self.install_uvloop()
        
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop, create a new one
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
        
        return self._loop
    
    def run_until_complete(self, coro: Coroutine) -> Any:
        """
        Run a coroutine until completion.
        
        Args:
            coro: Coroutine to run
        
        Returns:
            Result of the coroutine
        """
        loop = self.get_loop()
        self._start_time = time.perf_counter()
        
        try:
            result = loop.run_until_complete(coro)
            elapsed = time.perf_counter() - self._start_time
            logger.debug(f"Coroutine completed in {elapsed * 1000:.3f} ms")
            return result
        except Exception as e:
            logger.error(f"Coroutine failed: {e}")
            raise
        finally:
            self._start_time = None
    
    def create_task(self, coro: Coroutine, name: Optional[str] = None) -> asyncio.Task:
        """
        Create a task on the event loop.
        
        Args:
            coro: Coroutine to run as a task
            name: Optional task name
        
        Returns:
            Created asyncio Task
        """
        loop = self.get_loop()
        task = loop.create_task(coro, name=name)
        self._task_count += 1
        logger.debug(f"Created task {name or self._task_count}")
        return task
    
    async def schedule_callback(self, callback: Callable, delay: float = 0) -> None:
        """
        Schedule a callback to be executed after a delay.
        
        Args:
            callback: Callback function to execute
            delay: Delay in seconds (default 0 for immediate)
        """
        if delay > 0:
            await asyncio.sleep(delay)
        
        if asyncio.iscoroutinefunction(callback):
            await callback()
        else:
            callback()
    
    def run_forever(self) -> None:
        """Run the event loop forever."""
        loop = self.get_loop()
        logger.info("Event loop running forever")
        try:
            loop.run_forever()
        except KeyboardInterrupt:
            logger.info("Event loop interrupted")
        finally:
            self.shutdown()
    
    def shutdown(self) -> None:
        """Gracefully shutdown the event loop."""
        if self._loop:
            # Cancel all pending tasks
            pending = asyncio.all_tasks(self._loop)
            for task in pending:
                task.cancel()
            
            # Run until all tasks are cancelled
            if pending:
                self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            
            # Close the loop
            self._loop.close()
            logger.info("Event loop shutdown complete")
        
        self._loop = None


class AsyncEventCoordinator:
    """
    Coordinates async events between Rust IPC, Nautilus, and Ray workers.
    Ensures minimal latency for cross-component communication.
    """
    
    def __init__(self, event_loop: UVLoopEventLoop):
        self.event_loop = event_loop
        self._event_handlers: dict = {}
        self._event_queue: Optional[asyncio.Queue] = None
    
    def setup_event_queue(self, max_size: int = 10000) -> None:
        """Setup the internal event queue."""
        self._event_queue = asyncio.Queue(maxsize=max_size)
        logger.info(f"Event queue setup with max size {max_size}")
    
    def register_handler(self, event_type: str, handler: Callable) -> None:
        """Register an event handler for a specific event type."""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)
        logger.debug(f"Registered handler for event type: {event_type}")
    
    async def dispatch_event(self, event_type: str, data: Any) -> None:
        """Dispatch an event to all registered handlers."""
        if event_type not in self._event_handlers:
            return
        
        handlers = self._event_handlers[event_type]
        tasks = []
        
        for handler in handlers:
            if asyncio.iscoroutinefunction(handler):
                tasks.append(handler(data))
            else:
                # Run sync handler in executor to avoid blocking
                loop = self.event_loop.get_loop()
                tasks.append(loop.run_in_executor(None, handler, data))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def process_event_queue(self) -> None:
        """Continuously process events from the queue."""
        if not self._event_queue:
            logger.error("Event queue not initialized")
            return
        
        while True:
            try:
                event = await self._event_queue.get()
                event_type = event.get("type", "unknown")
                data = event.get("data")
                
                await self.dispatch_event(event_type, data)
                self._event_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing event: {e}")


# Global event loop instance
_event_loop_instance: Optional[UVLoopEventLoop] = None


def get_event_loop() -> UVLoopEventLoop:
    """Get or create the global event loop instance."""
    global _event_loop_instance
    if _event_loop_instance is None:
        _event_loop_instance = UVLoopEventLoop()
        _event_loop_instance.install_uvloop()
    return _event_loop_instance


def setup_async_environment() -> UVLoopEventLoop:
    """
    Setup the complete async environment with uvloop.
    
    Returns:
        Configured UVLoopEventLoop instance
    """
    loop = get_event_loop()
    logger.info("Async environment setup complete with uvloop")
    return loop
