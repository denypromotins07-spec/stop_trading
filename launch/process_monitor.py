#!/usr/bin/env python3
"""
Process Monitor - Stage 50
Watchdog daemon monitoring Rust and Python PIDs, automatically triggering
global kill switch if either process crashes or OOMs.
"""

import os
import sys
import time
import signal
import logging
from datetime import datetime
from typing import Optional, Dict, List
import psutil
import zmq

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
)
logger = logging.getLogger('ProcessMonitor')

# Constants
MAX_RUST_MEMORY_MB = 2048
MAX_PYTHON_MEMORY_MB = 3072
OOM_THRESHOLD_PERCENT = 95
HEALTH_CHECK_INTERVAL_SEC = 2
ZMQ_PUB_URL = "tcp://localhost:5556"


class ProcessHealthChecker:
    """Monitors individual process health metrics."""
    
    def __init__(self, pid: int, name: str, max_memory_mb: float):
        self.pid = pid
        self.name = name
        self.max_memory_mb = max_memory_mb
        self.crash_count = 0
        self.last_check_time: Optional[datetime] = None
        self.memory_history: List[float] = []
        self.max_history_length = 100
    
    def check_health(self) -> Dict:
        """Perform comprehensive health check on the process."""
        result = {
            'alive': False,
            'memory_mb': 0.0,
            'cpu_percent': 0.0,
            'threads': 0,
            'status': 'unknown',
            'oom_risk': False,
            'memory_trend': 'stable'
        }
        
        try:
            proc = psutil.Process(self.pid)
            result['alive'] = proc.is_running()
            
            if result['alive']:
                # Memory metrics
                mem_info = proc.memory_info()
                result['memory_mb'] = mem_info.rss / (1024 * 1024)
                result['memory_percent'] = proc.memory_percent()
                
                # CPU metrics
                result['cpu_percent'] = proc.cpu_percent(interval=0.1)
                
                # Thread count
                result['threads'] = proc.num_threads()
                
                # Process status
                result['status'] = proc.status()
                
                # Track memory history for trend analysis
                self.memory_history.append(result['memory_mb'])
                if len(self.memory_history) > self.max_history_length:
                    self.memory_history.pop(0)
                
                # Analyze memory trend
                if len(self.memory_history) >= 10:
                    recent_avg = sum(self.memory_history[-10:]) / 10
                    older_avg = sum(self.memory_history[:10]) / 10
                    if recent_avg > older_avg * 1.5:
                        result['memory_trend'] = 'increasing'
                    elif recent_avg < older_avg * 0.9:
                        result['memory_trend'] = 'decreasing'
                
                # Check OOM risk
                if result['memory_mb'] > self.max_memory_mb * 0.9:
                    result['oom_risk'] = True
                
                # Check system-wide memory pressure
                system_mem = psutil.virtual_memory()
                if system_mem.percent > OOM_THRESHOLD_PERCENT:
                    result['oom_risk'] = True
                
                self.last_check_time = datetime.now()
        
        except psutil.NoSuchProcess:
            result['status'] = 'terminated'
            self.crash_count += 1
        except psutil.AccessDenied:
            result['status'] = 'access_denied'
        except Exception as e:
            logger.error(f"Error checking {self.name} (PID {self.pid}): {e}")
            result['status'] = 'error'
        
        return result


class GlobalKillSwitch:
    """Triggers system-wide emergency shutdown."""
    
    def __init__(self):
        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.PUSH)
        self.zmq_socket.setsockopt(zmq.LINGER, 0)
        self.zmq_socket.connect("tcp://localhost:5557")
        self.triggered = False
    
    def trigger(self, reason: str, culprit: Optional[str] = None):
        """Activate global kill switch."""
        if self.triggered:
            return
        
        self.triggered = True
        message = {
            'type': 'KILL_SWITCH',
            'reason': reason,
            'culprit': culprit,
            'timestamp': datetime.now().isoformat(),
            'hostname': os.uname().nodename
        }
        
        try:
            self.zmq_socket.send_json(message, flags=zmq.NOBLOCK)
            logger.critical(f"GLOBAL KILL SWITCH TRIGGERED: {reason}")
        except Exception as e:
            logger.critical(f"Failed to send kill signal via ZMQ: {e}")
            # Fallback: write to file
            try:
                with open('/tmp/kill_switch_triggered.txt', 'w') as f:
                    f.write(f"{reason}\n{culprit}\n{message['timestamp']}")
            except:
                pass
    
    def close(self):
        """Close ZMQ socket."""
        self.zmq_socket.close()
        self.zmq_context.term()


class ProcessMonitorDaemon:
    """Main watchdog daemon."""
    
    def __init__(self, rust_pid: Optional[int] = None, python_pid: Optional[int] = None):
        self.rust_pid = rust_pid
        self.python_pid = python_pid
        self.running = False
        self.checkers: Dict[str, ProcessHealthChecker] = {}
        self.kill_switch = GlobalKillSwitch()
        self.alert_cooldown: Dict[str, datetime] = {}
        self.cooldown_seconds = 60
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down monitor...")
        self.running = False
    
    def register_process(self, pid: int, name: str, max_memory_mb: float):
        """Register a process for monitoring."""
        self.checkers[name] = ProcessHealthChecker(pid, name, max_memory_mb)
        logger.info(f"Registered {name} (PID {pid}) for monitoring")
    
    def _should_alert(self, alert_key: str) -> bool:
        """Check if we should send an alert (respecting cooldown)."""
        now = datetime.now()
        if alert_key in self.alert_cooldown:
            if (now - self.alert_cooldown[alert_key]).total_seconds() < self.cooldown_seconds:
                return False
        self.alert_cooldown[alert_key] = now
        return True
    
    def run(self):
        """Main monitoring loop."""
        self.running = True
        logger.info("Process Monitor Daemon started")
        logger.info(f"Monitoring: Rust PID={self.rust_pid}, Python PID={self.python_pid}")
        
        # Register processes if PIDs provided
        if self.rust_pid:
            self.register_process(self.rust_pid, 'rust_core', MAX_RUST_MEMORY_MB)
        if self.python_pid:
            self.register_process(self.python_pid, 'python_daemon', MAX_PYTHON_MEMORY_MB)
        
        consecutive_failures = 0
        max_consecutive_failures = 5
        
        while self.running:
            try:
                all_healthy = True
                
                for name, checker in self.checkers.items():
                    health = checker.check_health()
                    
                    # Log health status
                    if health['alive']:
                        log_msg = (
                            f"{name}: MEM={health['memory_mb']:.0f}MB "
                            f"CPU={health['cpu_percent']:.1f}% "
                            f"THR={health['threads']} "
                            f"TREND={health['memory_trend']}"
                        )
                        if health['oom_risk'] and self._should_alert(f"{name}_oom"):
                            logger.warning(f"OOM RISK detected for {name}: {health['memory_mb']:.0f}MB")
                        
                        if checker.max_memory_mb and health['memory_mb'] > checker.max_memory_mb:
                            logger.critical(f"{name} exceeded memory limit: {health['memory_mb']:.0f}MB > {checker.max_memory_mb:.0f}MB")
                            self.kill_switch.trigger(
                                f"{name} memory limit exceeded",
                                culprit=name
                            )
                            all_healthy = False
                            break
                    else:
                        logger.error(f"{name} is not running! Status: {health['status']}")
                        all_healthy = False
                        
                        # Check if this is a crash vs intentional stop
                        if checker.crash_count > 0 and self._should_alert(f"{name}_crash"):
                            logger.critical(f"{name} has crashed {checker.crash_count} times")
                            
                            if checker.crash_count >= 3:
                                self.kill_switch.trigger(
                                    f"{name} repeated crashes",
                                    culprit=name
                                )
                                all_healthy = False
                                break
                
                if not all_healthy:
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        logger.critical("Too many consecutive health check failures, triggering kill switch")
                        self.kill_switch.trigger("Consecutive health check failures")
                        break
                else:
                    consecutive_failures = 0
                
                time.sleep(HEALTH_CHECK_INTERVAL_SEC)
            
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                consecutive_failures += 1
                time.sleep(1)
        
        self.shutdown()
    
    def shutdown(self):
        """Graceful shutdown of monitor."""
        logger.info("Process Monitor Daemon shutting down...")
        self.kill_switch.close()
        logger.info("Process Monitor Daemon stopped")


def discover_pids() -> tuple:
    """Attempt to discover Rust and Python PIDs from various sources."""
    rust_pid = None
    python_pid = None
    
    # Try to read from PID files
    pid_files = {
        'rust': '/var/run/crypto_bot_rust.pid',
        'python': '/var/run/crypto_bot_python.pid'
    }
    
    for name, path in pid_files.items():
        try:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    pid = int(f.read().strip())
                    if name == 'rust':
                        rust_pid = pid
                    else:
                        python_pid = pid
                logger.info(f"Discovered {name} PID {pid} from {path}")
        except Exception as e:
            logger.warning(f"Could not read {path}: {e}")
    
    # Fallback: search by process name
    if not rust_pid or not python_pid:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = ' '.join(proc.info.get('cmdline', []) or [])
                name = proc.info.get('name', '')
                
                if not rust_pid and ('crypto_bot' in name or 'crypto_bot' in cmdline):
                    rust_pid = proc.info['pid']
                
                if not python_pid and 'nautilus_daemon' in cmdline:
                    python_pid = proc.info['pid']
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    
    return rust_pid, python_pid


def main():
    """Entry point for process monitor."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process Monitor Daemon')
    parser.add_argument('--rust-pid', type=int, help='Rust process PID')
    parser.add_argument('--python-pid', type=int, help='Python process PID')
    parser.add_argument('--discover', action='store_true', help='Auto-discover PIDs')
    args = parser.parse_args()
    
    rust_pid = args.rust_pid
    python_pid = args.python_pid
    
    if args.discover or (not rust_pid and not python_pid):
        rust_pid, python_pid = discover_pids()
    
    if not rust_pid and not python_pid:
        logger.error("No PIDs provided and auto-discovery failed")
        sys.exit(1)
    
    monitor = ProcessMonitorDaemon(rust_pid=rust_pid, python_pid=python_pid)
    
    try:
        monitor.run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        monitor.shutdown()


if __name__ == '__main__':
    main()
