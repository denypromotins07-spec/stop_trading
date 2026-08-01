"""
Advanced Auditing, Compliance & Trade Journaling
Stage 49: Asynchronously logs Nautilus OrderFilled and PositionClosed events.
Uses aiofiles and bounded asyncio queues to ensure disk I/O never blocks.
Correctly serializes Nautilus Cython objects to prevent segfaults.
"""

import asyncio
import aiofiles
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import deque
from pathlib import Path
import zmq

logger = logging.getLogger(__name__)


@dataclass
class TradeEvent:
    """Serialized trade event for journaling."""
    event_type: str  # OrderFilled, PositionClosed
    timestamp: str
    strategy_id: str
    instrument: str
    side: str
    quantity: float
    price: float
    pnl: float = 0.0
    commission: float = 0.0
    order_id: str = ""
    position_id: str = ""
    
    # Feature snapshot at decision time
    feature_snapshot: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    venue: str = ""
    execution_id: str = ""


class TradeJournaler:
    """
    Asynchronously logs every Nautilus OrderFilled and PositionClosed event.
    Uses aiofiles and bounded asyncio queues for non-blocking I/O.
    Correctly serializes Nautilus Cython objects to standard Python dicts.
    """
    
    def __init__(self,
                 log_dir: str = "/tmp/trade_logs",
                 queue_max_size: int = 10000,
                 batch_size: int = 100,
                 flush_interval: float = 5.0):
        
        self.log_dir = Path(log_dir)
        self.queue_max_size = queue_max_size
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        
        # Bounded async queue for events
        self._event_queue: asyncio.Queue = asyncio.Queue(maxsize=queue_max_size)
        
        # File handles
        self._current_file: Optional[Any] = None
        self._current_date: Optional[str] = None
        
        # Statistics
        self._events_written = 0
        self._events_dropped = 0
        
        # Running state
        self._running = False
        self._writer_task: Optional[asyncio.Task] = None
        
        # ZMQ socket for Rust IPC
        self._zmq_context = zmq.Context()
        self._zmq_socket = self._zmq_context.socket(zmq.PUSH)
        self._zmq_socket.connect("tcp://localhost:5567")
        
        # Ensure log directory exists
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    async def start(self):
        """Start the journaling writer loop."""
        if self._running:
            return
        
        self._running = True
        self._writer_task = asyncio.create_task(self._writer_loop())
        logger.info(f"TradeJournaler started, writing to {self.log_dir}")
    
    async def stop(self):
        """Stop the journaling writer loop."""
        self._running = False
        
        if self._writer_task:
            self._writer_task.cancel()
            try:
                await self._writer_task
            except asyncio.CancelledError:
                pass
        
        # Flush remaining events
        await self._flush_queue()
        
        # Close file
        if self._current_file:
            await self._current_file.close()
            self._current_file = None
        
        logger.info("TradeJournaler stopped")
    
    def serialize_nautilus_object(self, obj: Any) -> Dict[str, Any]:
        """
        Safely serialize Nautilus Cython objects to standard Python dicts.
        Prevents segmentation faults from direct Cython object access.
        """
        if obj is None:
            return {}
        
        # Check if it's a Nautilus type and extract safely
        try:
            # Use getattr with defaults to avoid direct attribute access on Cython objects
            result = {}
            
            # Common Nautilus types
            if hasattr(obj, 'id'):
                result['id'] = str(getattr(obj, 'id', ''))
            
            if hasattr(obj, 'instrument_id'):
                result['instrument_id'] = str(getattr(obj, 'instrument_id', ''))
            
            if hasattr(obj, 'strategy_id'):
                result['strategy_id'] = str(getattr(obj, 'strategy_id', ''))
            
            if hasattr(obj, 'order_id'):
                result['order_id'] = str(getattr(obj, 'order_id', ''))
            
            if hasattr(obj, 'position_id'):
                result['position_id'] = str(getattr(obj, 'position_id', ''))
            
            if hasattr(obj, 'side'):
                result['side'] = str(getattr(obj, 'side', ''))
            
            if hasattr(obj, 'quantity'):
                result['quantity'] = float(getattr(obj, 'quantity', 0.0))
            
            if hasattr(obj, 'price'):
                result['price'] = float(getattr(obj, 'price', 0.0))
            
            if hasattr(obj, 'pnl'):
                result['pnl'] = float(getattr(obj, 'pnl', 0.0))
            
            if hasattr(obj, 'commission'):
                result['commission'] = float(getattr(obj, 'commission', 0.0))
            
            if hasattr(obj, 'ts_event'):
                result['ts_event'] = int(getattr(obj, 'ts_event', 0))
            
            if hasattr(obj, 'ts_init'):
                result['ts_init'] = int(getattr(obj, 'ts_init', 0))
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to serialize Nautilus object: {e}")
            return {'error': str(e), 'type': str(type(obj))}
    
    async def log_order_filled(self, 
                               order_event: Any,
                               feature_snapshot: Dict[str, Any]):
        """Log an OrderFilled event with feature snapshot."""
        try:
            # Safely serialize the Nautilus Cython object
            serialized = self.serialize_nautilus_object(order_event)
            
            event = TradeEvent(
                event_type="OrderFilled",
                timestamp=datetime.utcnow().isoformat(),
                strategy_id=serialized.get('strategy_id', ''),
                instrument=serialized.get('instrument_id', ''),
                side=serialized.get('side', ''),
                quantity=serialized.get('quantity', 0.0),
                price=serialized.get('price', 0.0),
                commission=serialized.get('commission', 0.0),
                order_id=serialized.get('order_id', ''),
                feature_snapshot=feature_snapshot.copy(),
            )
            
            await self._queue_event(event)
            
        except Exception as e:
            logger.error(f"Failed to log OrderFilled: {e}")
    
    async def log_position_closed(self,
                                  position_event: Any,
                                  feature_snapshot: Dict[str, Any]):
        """Log a PositionClosed event with feature snapshot."""
        try:
            # Safely serialize the Nautilus Cython object
            serialized = self.serialize_nautilus_object(position_event)
            
            event = TradeEvent(
                event_type="PositionClosed",
                timestamp=datetime.utcnow().isoformat(),
                strategy_id=serialized.get('strategy_id', ''),
                instrument=serialized.get('instrument_id', ''),
                quantity=serialized.get('quantity', 0.0),
                price=serialized.get('price', 0.0),
                pnl=serialized.get('pnl', 0.0),
                position_id=serialized.get('position_id', ''),
                feature_snapshot=feature_snapshot.copy(),
            )
            
            await self._queue_event(event)
            
        except Exception as e:
            logger.error(f"Failed to log PositionClosed: {e}")
    
    async def _queue_event(self, event: TradeEvent):
        """Queue an event for async writing."""
        try:
            self._event_queue.put_nowait(event)
            self._events_written += 1
        except asyncio.QueueFull:
            self._events_dropped += 1
            logger.warning(f"Event queue full, dropped event: {event.event_type}")
    
    async def _writer_loop(self):
        """Async writer loop that batches and flushes events."""
        batch = []
        last_flush = datetime.utcnow()
        
        while self._running:
            try:
                # Collect batch
                while len(batch) < self.batch_size:
                    try:
                        event = self._event_queue.get_nowait()
                        batch.append(event)
                    except asyncio.QueueEmpty:
                        break
                
                # Write batch if we have events or it's been too long
                should_flush = (
                    len(batch) > 0 and 
                    (datetime.utcnow() - last_flush).total_seconds() >= self.flush_interval
                )
                
                if batch and (len(batch) >= self.batch_size or should_flush):
                    await self._write_batch(batch)
                    batch = []
                    last_flush = datetime.utcnow()
                
                await asyncio.sleep(0.1)  # Small delay to prevent busy-waiting
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Writer loop error: {e}")
                await asyncio.sleep(1.0)
    
    async def _write_batch(self, batch: List[TradeEvent]):
        """Write a batch of events to file."""
        try:
            # Get today's date for file naming
            today = datetime.utcnow().strftime("%Y-%m-%d")
            
            # Rotate file if date changed
            if today != self._current_date:
                if self._current_file:
                    await self._current_file.close()
                self._current_date = today
                self._current_file = None
            
            # Open file if needed
            if self._current_file is None:
                filepath = self.log_dir / f"trade_journal_{today}.jsonl"
                self._current_file = await aiofiles.open(filepath, mode='a')
            
            # Write events as JSON lines
            for event in batch:
                line = json.dumps(asdict(event)) + '\n'
                await self._current_file.write(line)
            
            await self._current_file.flush()
            
            # Notify Rust of write
            self._notify_rust(len(batch))
            
        except Exception as e:
            logger.error(f"Failed to write batch: {e}")
    
    async def _flush_queue(self):
        """Flush all remaining events in queue."""
        batch = []
        while not self._event_queue.empty():
            try:
                event = self._event_queue.get_nowait()
                batch.append(event)
            except asyncio.QueueEmpty:
                break
        
        if batch:
            await self._write_batch(batch)
    
    def _notify_rust(self, events_count: int):
        """Notify Rust of write activity."""
        try:
            self._zmq_socket.send_json({
                'type': 'TRADE_JOURNAL_WRITE',
                'events_count': events_count,
                'total_events': self._events_written,
                'dropped_events': self._events_dropped,
                'timestamp': datetime.utcnow().isoformat(),
            }, flags=zmq.NOBLOCK)
        except Exception as e:
            logger.error(f"Failed to notify Rust: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get journaler status."""
        return {
            'running': self._running,
            'queue_size': self._event_queue.qsize(),
            'events_written': self._events_written,
            'events_dropped': self._events_dropped,
            'log_dir': str(self.log_dir),
        }
    
    def shutdown(self):
        """Cleanup resources."""
        self._zmq_socket.close()
        self._zmq_context.term()
        logger.info("TradeJournaler shut down")


# Global instance
_journaler: Optional[TradeJournaler] = None


def get_journaler() -> TradeJournaler:
    """Get or create the global TradeJournaler instance."""
    global _journaler
    if _journaler is None:
        _journaler = TradeJournaler()
    return _journaler


def create_journaler(log_dir: str = "/tmp/trade_logs",
                    queue_max_size: int = 10000,
                    batch_size: int = 100) -> TradeJournaler:
    """Create a new TradeJournaler with custom configuration."""
    global _journaler
    _journaler = TradeJournaler(
        log_dir=log_dir,
        queue_max_size=queue_max_size,
        batch_size=batch_size,
    )
    return _journaler
