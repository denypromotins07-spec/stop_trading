#!/usr/bin/env python3
"""
Master Orchestrator - Stage 50
The absolute root Python script that spawns the Rust binary and Python Ray/Nautilus cluster.
Manages PIDs, IPC handshakes, and enforces the 4-hour trading window.
"""

import os
import sys
import time
import signal
import subprocess
import threading
import multiprocessing as mp
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any
import logging
import zmq
import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/workspace/logs/master_orchestrator.log')
    ]
)
logger = logging.getLogger('MasterOrchestrator')

# Constants
TRADING_WINDOW_HOURS = 4
RUST_BINARY_PATH = Path('/workspace/target/release/crypto_bot')
PYTHON_DAEMON_PATH = Path('/workspace/python/daemon/nautilus_daemon.py')
SHM_NAME = '/crypto_bot_shm'
SHM_SIZE = 4096
READY_FLAG_OFFSET = 0
HALT_FLAG_OFFSET = 1

class SharedMemoryIPC:
    """Handles shared memory communication between Rust and Python."""
    
    def __init__(self, name: str, size: int):
        self.name = name
        self.size = size
        self.shm = None
        self._setup()
    
    def _setup(self):
        """Initialize shared memory segment."""
        try:
            from multiprocessing import shared_memory
            try:
                self.shm = shared_memory.SharedMemory(name=self.name)
            except FileNotFoundError:
                self.shm = shared_memory.SharedMemory(name=self.name, create=True, size=self.size)
        except ImportError:
            # Fallback for older Python versions
            import mmap
            self.shm = mmap.mmap(-1, self.size, tagname=self.name)
    
    def write_ready_flag(self, ready: bool):
        """Write READY flag to shared memory."""
        if hasattr(self.shm, 'buf'):
            self.shm.buf[READY_FLAG_OFFSET] = 1 if ready else 0
        else:
            self.shm.seek(READY_FLAG_OFFSET)
            self.shm.write(bytes([1 if ready else 0]))
    
    def read_halt_flag(self) -> bool:
        """Read HALT flag from shared memory."""
        if hasattr(self.shm, 'buf'):
            return bool(self.shm.buf[HALT_FLAG_OFFSET])
        else:
            self.shm.seek(HALT_FLAG_OFFSET)
            return bool(self.shm.read(1)[0])
    
    def close(self):
        """Close shared memory segment."""
        if self.shm:
            try:
                self.shm.close()
                self.shm.unlink()
            except:
                pass


class ProcessManager:
    """Manages Rust and Python child processes."""
    
    def __init__(self):
        self.rust_process: Optional[subprocess.Popen] = None
        self.python_process: Optional[subprocess.Popen] = None
        self.start_time: Optional[datetime] = None
        self.shutdown_event = threading.Event()
        self.shm_ipc: Optional[SharedMemoryIPC] = None
    
    def start_rust_binary(self) -> bool:
        """Spawn the Rust ultra-low-latency core."""
        if not RUST_BINARY_PATH.exists():
            logger.error(f"Rust binary not found at {RUST_BINARY_PATH}")
            return False
        
        try:
            env = os.environ.copy()
            env['RUST_LOG'] = 'info'
            env['TRADING_MODE'] = 'live'
            
            self.rust_process = subprocess.Popen(
                [str(RUST_BINARY_PATH)],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid
            )
            logger.info(f"Rust binary spawned with PID: {self.rust_process.pid}")
            return True
        except Exception as e:
            logger.error(f"Failed to start Rust binary: {e}")
            return False
    
    def start_python_daemon(self) -> bool:
        """Spawn the Python Ray/Nautilus cluster."""
        if not PYTHON_DAEMON_PATH.exists():
            logger.warning(f"Python daemon not found at {PYTHON_DAEMON_PATH}, skipping...")
            return True  # Non-fatal for now
        
        try:
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'
            env['RAY_DISABLE_DOCKER_CPU_WARNING'] = '1'
            
            self.python_process = subprocess.Popen(
                [sys.executable, str(PYTHON_DAEMON_PATH)],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid
            )
            logger.info(f"Python daemon spawned with PID: {self.python_process.pid}")
            return True
        except Exception as e:
            logger.error(f"Failed to start Python daemon: {e}")
            return False
    
    def wait_for_rust_ready(self, timeout: int = 30) -> bool:
        """Wait for Rust 'READY' flag in shared memory."""
        self.shm_ipc = SharedMemoryIPC(SHM_NAME, SHM_SIZE)
        start_wait = time.time()
        
        while time.time() - start_wait < timeout:
            if self.shm_ipc.read_halt_flag():  # Reusing offset for ready check initially
                # Check actual ready flag
                if hasattr(self.shm_ipc.shm, 'buf'):
                    ready = bool(self.shm_ipc.shm.buf[READY_FLAG_OFFSET])
                else:
                    self.shm_ipc.shm.seek(READY_FLAG_OFFSET)
                    ready = bool(self.shm_ipc.shm.read(1)[0])
                
                if ready:
                    logger.info("Rust core signaled READY via shared memory")
                    return True
            
            # Also check process health
            if self.rust_process and self.rust_process.poll() is not None:
                logger.error("Rust process crashed before signaling READY")
                return False
            
            time.sleep(0.1)
        
        logger.warning("Timeout waiting for Rust READY flag, proceeding anyway...")
        return True
    
    def check_process_health(self) -> Dict[str, Any]:
        """Check health of both managed processes."""
        status = {
            'rust_alive': False,
            'python_alive': False,
            'rust_pid': None,
            'python_pid': None,
            'rust_memory_mb': 0,
            'python_memory_mb': 0
        }
        
        if self.rust_process:
            status['rust_alive'] = self.rust_process.poll() is None
            status['rust_pid'] = self.rust_process.pid
            try:
                proc = psutil.Process(self.rust_process.pid)
                status['rust_memory_mb'] = proc.memory_info().rss / (1024 * 1024)
            except:
                pass
        
        if self.python_process:
            status['python_alive'] = self.python_process.poll() is None
            status['python_pid'] = self.python_process.pid
            try:
                proc = psutil.Process(self.python_process.pid)
                status['python_memory_mb'] = proc.memory_info().rss / (1024 * 1024)
            except:
                pass
        
        return status
    
    def graceful_shutdown(self, reason: str = "User requested"):
        """Initiate graceful shutdown of all processes."""
        logger.info(f"Initiating graceful shutdown: {reason}")
        
        # Signal Rust to halt via shared memory
        if self.shm_ipc:
            self.shm_ipc.write_ready_flag(False)  # Reuse as halt signal
        
        # Send SIGTERM to process groups
        for proc, name in [(self.rust_process, "Rust"), (self.python_process, "Python")]:
            if proc and proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    logger.info(f"Sent SIGTERM to {name} process group")
                except:
                    try:
                        proc.terminate()
                    except:
                        pass
        
        # Wait for graceful exit
        for proc, name in [(self.rust_process, "Rust"), (self.python_process, "Python")]:
            if proc:
                try:
                    proc.wait(timeout=10)
                    logger.info(f"{name} process exited gracefully")
                except subprocess.TimeoutExpired:
                    logger.warning(f"{name} process did not exit gracefully, forcing kill")
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except:
                        try:
                            proc.kill()
                        except:
                            pass
        
        # Cleanup shared memory
        if self.shm_ipc:
            self.shm_ipc.close()
        
        logger.info("Graceful shutdown complete")


class TradingWindowManager:
    """Manages the 4-hour trading window constraint."""
    
    def __init__(self, duration_hours: int = TRADING_WINDOW_HOURS):
        self.duration = timedelta(hours=duration_hours)
        self.start_time: Optional[datetime] = None
        self.timer_thread: Optional[threading.Thread] = None
    
    def start_window(self):
        """Start the trading window timer."""
        self.start_time = datetime.now()
        logger.info(f"Trading window started: will end at {self.start_time + self.duration}")
        
        self.timer_thread = threading.Thread(target=self._countdown, daemon=True)
        self.timer_thread.start()
    
    def _countdown(self):
        """Countdown thread that triggers shutdown when window expires."""
        while True:
            elapsed = datetime.now() - self.start_time
            remaining = self.duration - elapsed
            
            if remaining.total_seconds() <= 0:
                logger.critical("TRADING WINDOW EXPIRED - Initiating automatic shutdown")
                # Signal master orchestrator to shutdown
                if hasattr(self, 'shutdown_callback'):
                    self.shutdown_callback("Trading window expired")
                break
            
            # Log progress every 15 minutes
            if int(elapsed.total_seconds()) % 900 == 0:
                logger.info(f"Trading progress: {elapsed.total_seconds()/3600:.2f}h / {self.duration.total_seconds()/3600:.2f}h")
            
            time.sleep(1)
    
    def get_remaining_seconds(self) -> float:
        """Get remaining seconds in trading window."""
        if not self.start_time:
            return 0
        remaining = self.duration - (datetime.now() - self.start_time)
        return max(0, remaining.total_seconds())


class MasterOrchestrator:
    """Main orchestrator class tying everything together."""
    
    def __init__(self):
        self.process_manager = ProcessManager()
        self.trading_window = TradingWindowManager()
        self.running = False
        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.PUB)
        self.zmq_socket.bind("tcp://*:5555")
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Setup shutdown callback
        self.trading_window.shutdown_callback = self._trigger_shutdown
    
    def _signal_handler(self, signum, frame):
        """Handle interrupt signals."""
        logger.info(f"Received signal {signum}")
        self._trigger_shutdown("Signal received")
    
    def _trigger_shutdown(self, reason: str):
        """Trigger shutdown sequence."""
        if not self.running:
            return
        
        self.running = False
        self.process_manager.graceful_shutdown(reason)
        self.zmq_socket.send_string(f"SHUTDOWN:{reason}")
    
    def run(self):
        """Main execution loop."""
        logger.info("=" * 60)
        logger.info("CRYPTO MEDIUM FREQUENCY TRADING BOT - STAGE 50")
        logger.info("Master Orchestrator Starting...")
        logger.info("=" * 60)
        
        # Validate environment
        if not self._validate_environment():
            logger.error("Environment validation failed")
            return 1
        
        # Start Rust binary
        if not self.process_manager.start_rust_binary():
            logger.error("Failed to start Rust binary")
            return 1
        
        # Wait for Rust READY flag
        if not self.process_manager.wait_for_rust_ready():
            logger.error("Rust binary failed to signal READY")
            return 1
        
        # Start Python daemon
        if not self.process_manager.start_python_daemon():
            logger.warning("Python daemon failed to start, continuing with Rust only")
        
        # Start trading window
        self.trading_window.start_window()
        self.running = True
        
        # Publish startup event
        self.zmq_socket.send_string("STATUS:RUNNING")
        logger.info("System fully operational - 4 hour trading window active")
        
        # Main monitoring loop
        try:
            while self.running:
                # Check process health
                health = self.process_manager.check_process_health()
                
                if not health['rust_alive']:
                    logger.critical("Rust process crashed!")
                    self._trigger_shutdown("Rust process crash")
                    break
                
                if health['python_alive'] and health['python_memory_mb'] > 3000:
                    logger.warning(f"Python daemon memory usage high: {health['python_memory_mb']:.0f}MB")
                
                # Log telemetry
                telemetry = {
                    'timestamp': datetime.now().isoformat(),
                    'rust_pid': health['rust_pid'],
                    'rust_memory_mb': health['rust_memory_mb'],
                    'trading_remaining_sec': self.trading_window.get_remaining_seconds()
                }
                self.zmq_socket.send_json(telemetry)
                
                time.sleep(5)  # Health check interval
        
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
            self._trigger_shutdown("User interrupt")
        
        finally:
            self.zmq_socket.close()
            self.zmq_context.term()
            logger.info("Master Orchestrator shutdown complete")
        
        return 0
    
    def _validate_environment(self) -> bool:
        """Validate pre-launch requirements."""
        checks_passed = True
        
        # Check Rust binary
        if not RUST_BINARY_PATH.exists():
            logger.error(f"Rust binary not found: {RUST_BINARY_PATH}")
            checks_passed = False
        
        # Check RAM constraint
        total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        if total_ram_gb < 6.0:
            logger.warning(f"Total system RAM ({total_ram_gb:.1f}GB) below recommended 6.5GB")
        
        # Check shared memory
        try:
            shm_path = Path('/dev/shm')
            if shm_path.exists():
                shm_stat = os.statvfs(shm_path)
                shm_available_gb = (shm_stat.f_bavail * shm_stat.f_frsize) / (1024 ** 3)
                logger.info(f"Available shared memory: {shm_available_gb:.2f}GB")
        except Exception as e:
            logger.warning(f"Could not check shared memory: {e}")
        
        return checks_passed


def main():
    """Entry point for master orchestrator."""
    orchestrator = MasterOrchestrator()
    sys.exit(orchestrator.run())


if __name__ == '__main__':
    main()
