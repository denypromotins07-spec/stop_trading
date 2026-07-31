"""
Background file watcher for atomic hot-swapping of ML models.
Detects new .onnx or .xgb models and swaps live inference pointers without dropping ticks.
"""

import os
import threading
import time
import hashlib
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from pathlib import Path
import logging

# Conditional imports
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileCreatedEvent, FileModifiedEvent
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    class FileSystemEventHandler:
        pass
    class Observer:
        pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelFileInfo:
    """Information about a model file."""
    file_path: str
    file_hash: str
    file_size: int
    modified_time: float
    model_type: str  # "onnx", "xgb", "lgb"
    model_id: str = ""


@dataclass
class HotSwapConfig:
    """Configuration for hot swap watcher."""
    watch_directories: List[str] = field(default_factory=lambda: ["/tmp/models"])
    poll_interval_seconds: float = 1.0
    debounce_seconds: float = 2.0
    supported_extensions: Set[str] = field(default_factory=lambda: {".onnx", ".xgb", ".joblib", ".pkl"})
    max_file_size_mb: int = 500
    enable_watchdog: bool = True


class ModelFileHandler(FileSystemEventHandler):
    """Handler for file system events related to model files."""
    
    def __init__(self, callback: Callable[[ModelFileInfo], None], 
                 config: HotSwapConfig):
        super().__init__()
        self._callback = callback
        self._config = config
        self._pending_files: Dict[str, float] = {}
        self._lock = threading.Lock()
    
    def _should_process(self, path: str) -> bool:
        """Check if file should be processed."""
        ext = Path(path).suffix.lower()
        return ext in self._config.supported_extensions
    
    def _get_file_hash(self, path: str) -> str:
        """Calculate file hash."""
        try:
            with open(path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()[:16]
        except Exception:
            return ""
    
    def _debounce(self, path: str) -> None:
        """Debounce file events."""
        with self._lock:
            self._pending_files[path] = time.time() + self._config.debounce_seconds
    
    def _process_pending(self) -> None:
        """Process debounced files."""
        now = time.time()
        
        with self._lock:
            ready = []
            
            for path, ready_time in list(self._pending_files.items()):
                if now >= ready_time:
                    ready.append(path)
                    del self._pending_files[path]
        
        for path in ready:
            try:
                if not os.path.exists(path):
                    continue
                
                stat = os.stat(path)
                
                # Check size limit
                max_size = self._config.max_file_size_mb * 1024 * 1024
                if stat.st_size > max_size:
                    logger.warning(f"File too large: {path}")
                    continue
                
                info = ModelFileInfo(
                    file_path=path,
                    file_hash=self._get_file_hash(path),
                    file_size=stat.st_size,
                    modified_time=stat.st_mtime,
                    model_type=Path(path).suffix.lower().replace(".", ""),
                    model_id=Path(path).stem
                )
                
                self._callback(info)
            
            except Exception as e:
                logger.error(f"Error processing {path}: {e}")
    
    def on_created(self, event):
        """Handle file creation."""
        if event.is_directory:
            return
        
        if self._should_process(event.src_path):
            self._debounce(event.src_path)
    
    def on_modified(self, event):
        """Handle file modification."""
        if event.is_directory:
            return
        
        if self._should_process(event.src_path):
            self._debounce(event.src_path)
    
    def run_debouncer(self, stop_event: threading.Event) -> None:
        """Run debouncer loop."""
        while not stop_event.is_set():
            self._process_pending()
            time.sleep(0.1)


class ModelHotSwapper:
    """
    Background file watcher for atomic model hot-swapping.
    Detects new models and triggers atomic pointer swaps.
    """
    
    def __init__(self, 
                 config: Optional[HotSwapConfig] = None,
                 swap_callback: Optional[Callable[[ModelFileInfo], bool]] = None):
        self.config = config or HotSwapConfig()
        self._swap_callback = swap_callback
        
        self._observer: Optional[Observer] = None
        self._handler: Optional[ModelFileHandler] = None
        self._running = False
        self._stop_event = threading.Event()
        self._debouncer_thread: Optional[threading.Thread] = None
        
        # Track swapped files
        self._swapped_files: Dict[str, ModelFileInfo] = {}
        self._total_swaps = 0
        self._failed_swaps = 0
        
        self._lock = threading.RLock()
    
    def start(self) -> bool:
        """Start the file watcher."""
        if self._running:
            return True
        
        # Validate watch directories
        valid_dirs = []
        for d in self.config.watch_directories:
            if os.path.isdir(d):
                valid_dirs.append(d)
            else:
                try:
                    os.makedirs(d, exist_ok=True)
                    valid_dirs.append(d)
                    logger.info(f"Created watch directory: {d}")
                except Exception as e:
                    logger.warning(f"Cannot create directory {d}: {e}")
        
        if not valid_dirs:
            logger.error("No valid watch directories")
            return False
        
        self.config.watch_directories = valid_dirs
        self._running = True
        self._stop_event.clear()
        
        if WATCHDOG_AVAILABLE and self.config.enable_watchdog:
            # Use watchdog for efficient file watching
            self._handler = ModelFileHandler(self._on_model_detected, self.config)
            self._observer = Observer()
            
            for d in self.config.watch_directories:
                self._observer.schedule(self._handler, d, recursive=False)
            
            self._observer.start()
            
            # Start debouncer thread
            self._debouncer_thread = threading.Thread(
                target=self._handler.run_debouncer,
                args=(self._stop_event,),
                daemon=True
            )
            self._debouncer_thread.start()
            
            logger.info(f"Started watchdog observer on {valid_dirs}")
        
        else:
            # Fallback to polling
            self._start_polling()
        
        return True
    
    def _start_polling(self) -> None:
        """Start polling-based file watching (fallback)."""
        def poll_loop():
            known_files = set()
            
            while self._running and not self._stop_event.is_set():
                try:
                    current_files = set()
                    
                    for d in self.config.watch_directories:
                        if not os.path.isdir(d):
                            continue
                        
                        for f in os.listdir(d):
                            path = os.path.join(d, f)
                            ext = Path(f).suffix.lower()
                            
                            if ext not in self.config.supported_extensions:
                                continue
                            
                            current_files.add(path)
                            
                            if path not in known_files:
                                # New file detected
                                stat = os.stat(path)
                                info = ModelFileInfo(
                                    file_path=path,
                                    file_hash="",
                                    file_size=stat.st_size,
                                    modified_time=stat.st_mtime,
                                    model_type=ext.replace(".", ""),
                                    model_id=Path(f).stem
                                )
                                self._on_model_detected(info)
                    
                    known_files = current_files
                    
                except Exception as e:
                    logger.error(f"Polling error: {e}")
                
                self._stop_event.wait(self.config.poll_interval_seconds)
        
        self._debouncer_thread = threading.Thread(
            target=poll_loop,
            daemon=True,
            name="ModelWatcher_Poller"
        )
        self._debouncer_thread.start()
        
        logger.info("Started polling-based file watcher")
    
    def _on_model_detected(self, info: ModelFileInfo) -> None:
        """Handle detected model file."""
        logger.info(f"Detected model file: {info.file_path} (hash: {info.file_hash})")
        
        # Check if already swapped
        with self._lock:
            if info.file_path in self._swapped_files:
                existing = self._swapped_files[info.file_path]
                if existing.file_hash == info.file_hash:
                    logger.debug(f"Model unchanged: {info.file_path}")
                    return
        
        # Invoke swap callback
        success = False
        
        if self._swap_callback is not None:
            try:
                success = self._swap_callback(info)
            except Exception as e:
                logger.error(f"Swap callback error: {e}")
                self._failed_swaps += 1
        else:
            # Default behavior: just track
            success = True
        
        if success:
            with self._lock:
                self._swapped_files[info.file_path] = info
                self._total_swaps += 1
            
            logger.info(f"Successfully swapped model: {info.model_id}")
        else:
            self._failed_swaps += 1
            logger.error(f"Failed to swap model: {info.file_path}")
    
    def stop(self) -> None:
        """Stop the file watcher."""
        self._running = False
        self._stop_event.set()
        
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5.0)
        
        if self._debouncer_thread is not None:
            self._debouncer_thread.join(timeout=2.0)
        
        logger.info("Stopped model hot swapper")
    
    def trigger_swap(self, file_path: str) -> bool:
        """Manually trigger a swap for a specific file."""
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return False
        
        stat = os.stat(file_path)
        info = ModelFileInfo(
            file_path=file_path,
            file_hash="",
            file_size=stat.st_size,
            modified_time=stat.st_mtime,
            model_type=Path(file_path).suffix.lower().replace(".", ""),
            model_id=Path(file_path).stem
        )
        
        self._on_model_detected(info)
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get swapper statistics."""
        with self._lock:
            return {
                "total_swaps": self._total_swaps,
                "failed_swaps": self._failed_swaps,
                "watch_directories": self.config.watch_directories,
                "swapped_files": list(self._swapped_files.keys()),
                "is_running": self._running,
                "using_watchdog": WATCHDOG_AVAILABLE and self.config.enable_watchdog,
            }
    
    @property
    def is_running(self) -> bool:
        return self._running


# Global swapper instance
_swapper: Optional[ModelHotSwapper] = None
_lock = threading.Lock()


def get_model_hot_swapper(
    callback: Optional[Callable[[ModelFileInfo], bool]] = None,
    config: Optional[HotSwapConfig] = None
) -> ModelHotSwapper:
    """Get global ModelHotSwapper instance."""
    global _swapper
    
    with _lock:
        if _swapper is None:
            _swapper = ModelHotSwapper(config, callback)
        
        return _swapper


def reset_model_hot_swapper() -> None:
    """Reset the global swapper."""
    global _swapper
    
    with _lock:
        if _swapper is not None:
            _swapper.stop()
            _swapper = None


if __name__ == "__main__":
    print("Model Hot Swapper Demo")
    print("=" * 40)
    
    # Create test directory
    test_dir = "/tmp/test_models"
    os.makedirs(test_dir, exist_ok=True)
    
    # Define swap callback
    def mock_swap_callback(info: ModelFileInfo) -> bool:
        print(f"\n=== SWAP TRIGGERED ===")
        print(f"Model ID: {info.model_id}")
        print(f"File: {info.file_path}")
        print(f"Type: {info.model_type}")
        print(f"Hash: {info.file_hash}")
        print(f"Size: {info.file_size} bytes")
        return True
    
    # Create swapper
    config = HotSwapConfig(
        watch_directories=[test_dir],
        enable_watchdog=WATCHDOG_AVAILABLE
    )
    
    swapper = get_model_hot_swapper(mock_swap_callback, config)
    
    if not swapper.start():
        print("Failed to start swapper")
        exit(1)
    
    print(f"Watching directory: {test_dir}")
    print(f"Using watchdog: {WATCHDOG_AVAILABLE and config.enable_watchdog}")
    
    # Create a test model file
    test_file = os.path.join(test_dir, "test_model.onnx")
    print(f"\nCreating test file: {test_file}")
    
    with open(test_file, 'wb') as f:
        f.write(b"dummy onnx content")
    
    # Wait for detection
    time.sleep(3)
    
    # Get statistics
    stats = swapper.get_statistics()
    print(f"\nStatistics: {stats}")
    
    # Cleanup
    swapper.stop()
    reset_model_hot_swapper()
    
    # Remove test file
    if os.path.exists(test_file):
        os.remove(test_file)
    
    print("\nHot swapper demo complete")
