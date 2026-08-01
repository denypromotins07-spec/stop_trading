"""
Safety Interlock Module Root
Stage 49: Wires Python-side circuit breakers to Rust Global Kill Switch.
Uses non-blocking ZMQ PUSH sockets for instant halt propagation.
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import zmq

from .ml_hallucination_detector import MLHallucinationDetector, get_detector
from .correlation_breaker import CorrelationBreaker, get_breaker

logger = logging.getLogger(__name__)


class SafetyInterlockModule:
    """
    Central module wiring Python-side safety breakers to Rust Global Kill Switch.
    Monitors all safety systems and triggers coordinated shutdown on critical alerts.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Safety components
        self.hallucination_detector: Optional[MLHallucinationDetector] = None
        self.correlation_breaker: Optional[CorrelationBreaker] = None
        
        # State
        self._running = False
        self._halt_state = False
        self._halt_reason: Optional[str] = None
        
        # ZMQ socket for Rust Global Kill Switch
        self._zmq_context: Optional[zmq.Context] = None
        self._kill_socket: Optional[zmq.Socket] = None
        
        # Monitoring task
        self._monitor_task: Optional[asyncio.Task] = None
        
        # Alert callbacks
        self._alert_callbacks = []
    
    async def initialize(self) -> bool:
        """Initialize the safety interlock module."""
        try:
            logger.info("Initializing SafetyInterlockModule...")
            
            # Create safety detectors
            self.hallucination_detector = MLHallucinationDetector(
                window_size=self.config.get('hallucination_window', 500),
                ks_threshold=self.config.get('ks_threshold', 0.15),
                p_value_threshold=self.config.get('p_value_threshold', 0.01),
            )
            
            self.correlation_breaker = CorrelationBreaker(
                num_assets=self.config.get('num_assets', 50),
                window_size=self.config.get('correlation_window', 252),
                min_samples=self.config.get('min_correlation_samples', 60),
            )
            
            # Setup ZMQ connection to Rust Global Kill Switch
            self._zmq_context = zmq.Context()
            self._kill_socket = self._zmq_context.socket(zmq.PUSH)
            self._kill_socket.connect("tcp://localhost:5566")  # Rust kill switch endpoint
            
            self._running = True
            logger.info("SafetyInterlockModule initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize SafetyInterlockModule: {e}")
            return False
    
    async def start_monitoring(self, check_interval: float = 0.5):
        """Start continuous safety monitoring loop."""
        if not self._running:
            raise RuntimeError("Module not initialized")
        
        async def monitor_loop():
            while self._running:
                try:
                    await self._check_all_safety_systems()
                    await asyncio.sleep(check_interval)
                except Exception as e:
                    logger.error(f"Monitoring error: {e}")
                    await asyncio.sleep(check_interval)
        
        self._monitor_task = asyncio.create_task(monitor_loop())
        logger.info("Safety monitoring started")
    
    async def _check_all_safety_systems(self):
        """Check all safety systems and trigger halt if needed."""
        # Check ML hallucination detector
        if self.hallucination_detector:
            status = self.hallucination_detector.get_status()
            if status['halt_triggered']:
                await self._trigger_halt("ML_HALLUCINATION")
                return
        
        # Check correlation breaker
        if self.correlation_breaker:
            status = self.correlation_breaker.get_status()
            if status['halt_triggered']:
                await self._trigger_halt("CORRELATION_PANIC")
                return
    
    async def _trigger_halt(self, reason: str):
        """Trigger global halt and notify Rust."""
        if self._halt_state:
            return  # Already halted
        
        self._halt_state = True
        self._halt_reason = reason
        
        logger.critical(f"SAFETY HALT TRIGGERED: {reason}")
        
        # Notify Rust Global Kill Switch
        await self._notify_rust_kill(reason)
        
        # Execute alert callbacks
        for callback in self._alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(reason)
                else:
                    callback(reason)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
    
    async def _notify_rust_kill(self, reason: str):
        """Send kill signal to Rust via ZMQ."""
        try:
            self._kill_socket.send_json({
                'type': 'PYTHON_SAFETY_KILL',
                'reason': reason,
                'timestamp': datetime.utcnow().isoformat(),
                'requires_manual_reset': True,
            }, flags=zmq.NOBLOCK)
        except Exception as e:
            logger.error(f"Failed to send kill signal to Rust: {e}")
    
    def register_alert_callback(self, callback):
        """Register callback for safety alerts."""
        self._alert_callbacks.append(callback)
    
    def check_ml_distribution(self, probabilities) -> tuple:
        """Check ML output distribution for hallucination."""
        if not self.hallucination_detector:
            return True, None
        return self.hallucination_detector.check_distribution(probabilities)
    
    def add_asset_returns(self, asset_id: int, returns):
        """Add asset returns for correlation monitoring."""
        if self.correlation_breaker:
            self.correlation_breaker.add_returns(asset_id, returns)
    
    def check_correlation(self) -> tuple:
        """Check portfolio correlation for systemic risk."""
        if not self.correlation_breaker:
            return True, None
        return self.correlation_breaker.check_correlation()
    
    def reset_halt(self, force: bool = False) -> bool:
        """Reset halt state (requires explicit confirmation)."""
        if not self._halt_state:
            return True
        
        if not force:
            logger.warning("Halt reset requires force=True confirmation")
            return False
        
        logger.warning("Manual safety halt reset confirmed")
        
        if self.hallucination_detector:
            self.hallucination_detector.reset_halt()
        
        if self.correlation_breaker:
            self.correlation_breaker.reset_halt()
        
        self._halt_state = False
        self._halt_reason = None
        
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive safety status."""
        return {
            'running': self._running,
            'halt_state': self._halt_state,
            'halt_reason': self._halt_reason,
            'hallucination_detector': self.hallucination_detector.get_status() if self.hallucination_detector else None,
            'correlation_breaker': self.correlation_breaker.get_status() if self.correlation_breaker else None,
        }
    
    async def shutdown(self):
        """Gracefully shutdown the safety module."""
        logger.info("Shutting down SafetyInterlockModule...")
        self._running = False
        
        # Cancel monitoring task
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        # Shutdown components
        if self.hallucination_detector:
            self.hallucination_detector.shutdown()
        
        if self.correlation_breaker:
            self.correlation_breaker.shutdown()
        
        # Close ZMQ
        if self._kill_socket:
            self._kill_socket.close()
        if self._zmq_context:
            self._zmq_context.term()
        
        logger.info("SafetyInterlockModule shut down complete")


# Global module instance
_module: Optional[SafetyInterlockModule] = None


def get_module() -> SafetyInterlockModule:
    """Get or create the global SafetyInterlockModule instance."""
    global _module
    if _module is None:
        _module = SafetyInterlockModule({})
    return _module


def create_module(config: Dict[str, Any]) -> SafetyInterlockModule:
    """Create a new SafetyInterlockModule with custom configuration."""
    global _module
    _module = SafetyInterlockModule(config)
    return _module
