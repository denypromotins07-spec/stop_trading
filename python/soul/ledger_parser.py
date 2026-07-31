"""
Asynchronous watchdog file observer for SOUL.md monitoring.
Monitors for new trade outcomes and mistakes written by Rust journal.
"""

import asyncio
from pathlib import Path
from typing import Optional, Callable, List, Dict, Any
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent
import time
import re

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import SOUL_LEDGER_PATH, SOUL_WATCHDOG_INTERVAL, get_logger

logger = get_logger("ledger_parser")


class SOULChangeHandler(FileSystemEventHandler):
    """
    File system event handler for SOUL.md changes.
    Triggers callbacks when the ledger is updated.
    """
    
    def __init__(self, callback: Callable[[str], None]):
        self.callback = callback
        self._last_modified: float = 0
        self._debounce_seconds: float = 0.5
    
    def on_modified(self, event) -> None:
        """Handle file modification events."""
        if isinstance(event, FileModifiedEvent):
            # Check if it's the SOUL.md file
            if Path(event.src_path).name == "SOUL.md":
                current_time = time.time()
                
                # Debounce rapid modifications
                if current_time - self._last_modified > self._debounce_seconds:
                    self._last_modified = current_time
                    logger.debug(f"SOUL.md modified at {current_time}")
                    
                    try:
                        with open(event.src_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        self.callback(content)
                    except Exception as e:
                        logger.error(f"Error reading SOUL.md: {e}")


class LedgerParser:
    """
    Asynchronous parser for SOUL.md ledger entries.
    Extracts trade outcomes, mistakes, and adaptive weights.
    """
    
    # Regex patterns for parsing SOUL.md sections
    PATTERNS = {
        'trade_outcome': re.compile(
            r'###\s*Trade\s*#\d+\s*\n.*?'
            r'Result:\s*(WIN|LOSS|BREAKEVEN)\s*\n'
            r'PnL:\s*([+-]?\d+\.?\d*)\s*\n'
            r'Timestamp:\s*(\d+)',
            re.MULTILINE | re.DOTALL
        ),
        'mistake': re.compile(
            r'##\s*Mistakes?\s*\n(.*?)(?=##|\Z)',
            re.MULTILINE | re.DOTALL
        ),
        'adaptive_weights': re.compile(
            r'##\s*Adaptive\s*Weights?\s*\n(.*?)(?=##|\Z)',
            re.MULTILINE | re.DOTALL
        ),
        'regime_memory': re.compile(
            r'##\s*Regime\s*Memories?\s*\n(.*?)(?=##|\Z)',
            re.MULTILINE | re.DOTALL
        ),
        'weight_entry': re.compile(
            r'(\w+):\s*([+-]?\d+\.?\d*)',
            re.MULTILINE
        ),
    }
    
    def __init__(self):
        self._content_cache: str = ""
        self._parsed_entries: List[Dict[str, Any]] = []
        self._change_callbacks: List[Callable] = []
    
    def parse_content(self, content: str) -> Dict[str, Any]:
        """
        Parse SOUL.md content and extract structured data.
        
        Args:
            content: Raw content of SOUL.md
        
        Returns:
            Dictionary with parsed sections
        """
        result = {
            'trade_outcomes': [],
            'mistakes': [],
            'adaptive_weights': {},
            'regime_memories': [],
            'raw_content_length': len(content),
            'parse_timestamp': time.time(),
        }
        
        # Parse trade outcomes
        for match in self.PATTERNS['trade_outcome'].finditer(content):
            result['trade_outcomes'].append({
                'result': match.group(1),
                'pnl': float(match.group(2)),
                'timestamp': int(match.group(3)),
            })
        
        # Parse mistakes section
        mistake_match = self.PATTERNS['mistake'].search(content)
        if mistake_match:
            mistakes_text = mistake_match.group(1).strip()
            result['mistakes'] = [
                line.strip() for line in mistakes_text.split('\n')
                if line.strip() and not line.strip().startswith('#')
            ]
        
        # Parse adaptive weights
        weights_match = self.PATTERNS['adaptive_weights'].search(content)
        if weights_match:
            weights_text = weights_match.group(1)
            for weight_match in self.PATTERNS['weight_entry'].finditer(weights_text):
                key = weight_match.group(1)
                value = float(weight_match.group(2))
                result['adaptive_weights'][key] = value
        
        # Parse regime memories
        regime_match = self.PATTERNS['regime_memory'].search(content)
        if regime_match:
            regime_text = regime_match.group(1).strip()
            result['regime_memories'] = [
                line.strip() for line in regime_text.split('\n')
                if line.strip() and not line.strip().startswith('#')
            ]
        
        return result
    
    def register_change_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Register a callback for when new entries are parsed."""
        self._change_callbacks.append(callback)
        logger.debug(f"Registered change callback: {callback.__name__}")
    
    def _notify_callbacks(self, parsed_data: Dict[str, Any]) -> None:
        """Notify all registered callbacks of new parsed data."""
        for callback in self._change_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(parsed_data))
                else:
                    callback(parsed_data)
            except Exception as e:
                logger.error(f"Callback error: {e}")


class SOULWatchdog:
    """
    Async watchdog observer for SOUL.md file changes.
    Monitors the ledger and triggers parsing on updates.
    """
    
    def __init__(self, ledger_path: Optional[Path] = None):
        self.ledger_path = ledger_path or SOUL_LEDGER_PATH
        self._observer: Optional[Observer] = None
        self._parser = LedgerParser()
        self._running = False
        self._latest_data: Optional[Dict[str, Any]] = None
    
    def start(self) -> bool:
        """
        Start the watchdog observer.
        
        Returns:
            True if started successfully
        """
        if self._running:
            return True
        
        try:
            # Ensure directory exists
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create file if it doesn't exist
            if not self.ledger_path.exists():
                self.ledger_path.touch()
                logger.info(f"Created SOUL.md at {self.ledger_path}")
            
            # Setup change handler
            def on_change(content: str) -> None:
                parsed = self._parser.parse_content(content)
                self._latest_data = parsed
                self._parser._notify_callbacks(parsed)
                
                # Log summary
                if parsed['trade_outcomes']:
                    logger.info(
                        f"SOUL.md update: {len(parsed['trade_outcomes'])} trades, "
                        f"{len(parsed['mistakes'])} mistakes, "
                        f"{len(parsed['adaptive_weights'])} weights"
                    )
            
            handler = SOULChangeHandler(on_change)
            
            # Start observer
            self._observer = Observer()
            self._observer.schedule(handler, str(self.ledger_path.parent), recursive=False)
            self._observer.start()
            
            self._running = True
            logger.info(f"SOUL watchdog started for {self.ledger_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start SOUL watchdog: {e}")
            return False
    
    def stop(self) -> None:
        """Stop the watchdog observer."""
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=2.0)
            self._observer = None
        
        self._running = False
        logger.info("SOUL watchdog stopped")
    
    def register_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Register a callback for parsed ledger updates."""
        self._parser.register_change_callback(callback)
    
    def get_latest_data(self) -> Optional[Dict[str, Any]]:
        """Get the latest parsed data from SOUL.md."""
        return self._latest_data
    
    def is_running(self) -> bool:
        """Check if watchdog is running."""
        return self._running


# Global watchdog instance
_watchdog_instance: Optional[SOULWatchdog] = None


def get_soul_watchdog() -> SOULWatchdog:
    """Get or create the global SOUL watchdog instance."""
    global _watchdog_instance
    if _watchdog_instance is None:
        _watchdog_instance = SOULWatchdog()
    return _watchdog_instance


def start_ledger_monitoring() -> SOULWatchdog:
    """
    Start monitoring SOUL.md for changes.
    
    Returns:
        Running SOULWatchdog instance
    """
    watchdog = get_soul_watchdog()
    if not watchdog.start():
        raise RuntimeError("Failed to start SOUL watchdog")
    return watchdog
