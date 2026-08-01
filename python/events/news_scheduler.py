"""
Asyncio-based event scheduler aligning macroeconomic releases with Nautilus global clock.
Pre-positions volatility breakout strategies and widens execution spreads before high-impact news.
Optimized for sub-microsecond precision and 3GB RAM constraint.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Callable, Any
import logging
import time

logger = logging.getLogger(__name__)


class ImpactLevel(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class MacroEvent:
    """Represents a scheduled macroeconomic event."""
    ticker: str
    event_name: str
    scheduled_time: datetime  # UTC nanosecond precision
    impact: ImpactLevel
    actual: Optional[float] = None
    forecast: Optional[float] = None
    previous: Optional[float] = None
    processed: bool = False
    
    def time_to_event_ns(self) -> int:
        """Return nanoseconds until event."""
        now = datetime.now(timezone.utc)
        delta = self.scheduled_time - now
        return int(delta.total_seconds() * 1e9)


@dataclass
class VolatilityState:
    """Tracks volatility state adjustments around news events."""
    base_spread_bps: float = 5.0
    multiplier: float = 1.0
    pre_event_window_ns: int = 60_000_000_000  # 60 seconds
    post_event_window_ns: int = 300_000_000_000  # 5 minutes
    last_update_ns: int = 0


class NewsScheduler:
    """
    Asyncio-based scheduler for macroeconomic events.
    Aligns with Nautilus global clock and manages volatility pre-positioning.
    """
    
    def __init__(self, nautilus_clock_sync: Optional[Callable[[], int]] = None):
        """
        Initialize scheduler.
        
        Args:
            nautilus_clock_sync: Optional callback to get Nautilus global clock in ns.
        """
        self._events: Dict[str, List[MacroEvent]] = {}
        self._volatility_states: Dict[str, VolatilityState] = {}
        self._callbacks: Dict[ImpactLevel, List[Callable[[MacroEvent], Any]]] = {
            level: [] for level in ImpactLevel
        }
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._nautilus_clock = nautilus_clock_sync
        self._lock = asyncio.Lock()
        
        # Memory-efficient event queue (bounded)
        self._max_pending_events = 1000
        
    def get_current_time_ns(self) -> int:
        """Get current time in nanoseconds, synced with Nautilus if available."""
        if self._nautilus_clock:
            return self._nautilus_clock()
        return time.time_ns()
    
    def add_event(self, event: MacroEvent) -> bool:
        """
        Add a macroeconomic event to the schedule.
        
        Returns:
            True if added successfully, False if queue is full.
        """
        if len(self._events.get(event.ticker, [])) >= self._max_pending_events:
            logger.warning(f"Event queue full for {event.ticker}")
            return False
            
        async def _add():
            async with self._lock:
                if event.ticker not in self._events:
                    self._events[event.ticker] = []
                    self._volatility_states[event.ticker] = VolatilityState()
                self._events[event.ticker].append(event)
                # Keep sorted by time
                self._events[event.ticker].sort(key=lambda e: e.scheduled_time)
        
        # Run in event loop if available, else sync
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_add())
        except RuntimeError:
            asyncio.run(_add())
            
        return True
    
    def register_callback(self, impact: ImpactLevel, callback: Callable[[MacroEvent], Any]):
        """Register callback for events of specific impact level."""
        self._callbacks[impact].append(callback)
        
    def get_volatility_multiplier(self, ticker: str) -> float:
        """Get current volatility multiplier for spread widening."""
        if ticker not in self._volatility_states:
            return 1.0
        state = self._volatility_states[ticker]
        now_ns = self.get_current_time_ns()
        
        # Check if within pre/post event windows
        for event in self._events.get(ticker, []):
            if event.processed:
                continue
            time_to_event = event.time_to_event_ns()
            
            # Pre-event: ramp up volatility
            if 0 < time_to_event <= state.pre_event_window_ns:
                ratio = 1.0 - (time_to_event / state.pre_event_window_ns)
                return state.base_spread_bps * (1.0 + ratio * 4.0)  # Up to 5x spread
            
            # Post-event: decay
            if -state.post_event_window_ns <= time_to_event < 0:
                ratio = abs(time_to_event) / state.post_event_window_ns
                return state.base_spread_bps * (1.0 + ratio * 3.0)
                
        return state.base_spread_bps
    
    def _decay_impact(self, event: MacroEvent, elapsed_ns: int) -> float:
        """
        Calculate decaying impact score for news event.
        Uses exponential decay model optimized for low memory.
        """
        half_life_ns = 300_000_000_000  # 5 minutes half-life
        decay_constant = 0.693 / half_life_ns  # ln(2) / half_life
        impact_score = event.impact.value * math.exp(-decay_constant * elapsed_ns)
        return max(0.0, impact_score)
    
    async def _process_event(self, event: MacroEvent):
        """Process a triggered event and notify callbacks."""
        event.processed = True
        logger.info(f"Processing {event.event_name} for {event.ticker}")
        
        # Notify registered callbacks
        for callback in self._callbacks[event.impact]:
            try:
                result = callback(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Callback error for {event.event_name}: {e}")
        
        # Schedule cleanup
        await asyncio.sleep(300)  # Keep event for 5 min for decay calculations
        async with self._lock:
            if event.ticker in self._events:
                self._events[event.ticker] = [
                    e for e in self._events[event.ticker] 
                    if e != event or not e.processed
                ]
    
    async def _scheduler_loop(self):
        """Main scheduler loop checking for upcoming events."""
        check_interval_s = 0.001  # 1ms check interval
        
        while self._running:
            now_ns = self.get_current_time_ns()
            
            async with self._lock:
                for ticker, events in list(self._events.items()):
                    for event in events:
                        if event.processed:
                            continue
                        
                        time_to_event = event.time_to_event_ns()
                        
                        # Trigger 100ms before event for pre-positioning
                        if 0 < time_to_event <= 100_000_000:
                            logger.debug(
                                f"Pre-positioning for {event.event_name} "
                                f"in {time_to_event/1e6:.2f}ms"
                            )
                            # Create task for processing
                            asyncio.create_task(self._process_event(event))
                            
            await asyncio.sleep(check_interval_s)
    
    async def start(self):
        """Start the scheduler."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._scheduler_loop())
        logger.info("NewsScheduler started")
    
    async def stop(self):
        """Stop the scheduler gracefully."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("NewsScheduler stopped")
    
    def get_upcoming_events(self, ticker: Optional[str] = None, 
                           window_ns: int = 3600_000_000_000) -> List[MacroEvent]:
        """
        Get upcoming events within time window.
        
        Args:
            ticker: Filter by ticker (None for all)
            window_ns: Time window in nanoseconds (default 1 hour)
            
        Returns:
            List of upcoming events
        """
        result = []
        tickers = [ticker] if ticker else list(self._events.keys())
        
        for t in tickers:
            for event in self._events.get(t, []):
                if not event.processed and 0 < event.time_to_event_ns() <= window_ns:
                    result.append(event)
                    
        return sorted(result, key=lambda e: e.scheduled_time)


# Import math for decay calculation
import math
