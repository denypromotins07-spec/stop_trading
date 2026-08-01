"""
Pre-Flight Module Root
Stage 49: Gates Python daemon startup, ensuring all hardware, network, and Rust IPC checks pass.
Requires 100% success rate on all pre-flight checks before allowing daemon start.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import zmq

from .hardware_validator import HardwareValidator, get_validator, validate_hardware
from .network_latency_test import NetworkLatencyTester, get_tester, test_network_latency

logger = logging.getLogger(__name__)


class PreflightModule:
    """
    Central module gating Python daemon startup.
    Ensures all hardware, network, and Rust IPC handshake checks pass with 100% success.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Components
        self.hardware_validator: Optional[HardwareValidator] = None
        self.network_tester: Optional[NetworkLatencyTester] = None
        
        # State
        self._running = False
        self._checks_passed = False
        
        # ZMQ socket for Rust IPC
        self._zmq_context: Optional[zmq.Context] = None
        self._handshake_socket: Optional[zmq.Socket] = None
        
        # Check results
        self._hardware_results: List = []
        self._network_results: Dict = {}
        self._ipc_handshake_ok = False
    
    async def initialize(self) -> bool:
        """Initialize the pre-flight module."""
        try:
            logger.info("Initializing PreflightModule...")
            
            # Create components
            self.hardware_validator = HardwareValidator()
            self.network_tester = NetworkLatencyTester(
                num_samples=self.config.get('network_samples', 5),
            )
            
            # Setup ZMQ for Rust IPC handshake
            self._zmq_context = zmq.Context()
            self._handshake_socket = self._zmq_context.socket(zmq.REQ)
            self._handshake_socket.connect("tcp://localhost:5572")  # Rust handshake endpoint
            
            self._running = True
            logger.info("PreflightModule initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize PreflightModule: {e}")
            return False
    
    async def run_all_checks(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Run all pre-flight checks.
        
        Returns:
            Tuple of (all_passed, detailed_results)
        """
        logger.info("=" * 60)
        logger.info("STARTING PRE-FLIGHT CHECKS")
        logger.info("=" * 60)
        
        results = {
            'hardware': {'passed': False, 'details': []},
            'network': {'passed': False, 'details': {}},
            'ipc_handshake': {'passed': False, 'details': ''},
            'overall': False,
        }
        
        # Check 1: Hardware validation
        logger.info("\n[HARDWARE VALIDATION]")
        hw_passed, hw_results = self.hardware_validator.validate_all()
        results['hardware']['passed'] = hw_passed
        results['hardware']['details'] = [
            {'check': r.check_name, 'passed': r.passed, 'message': r.message}
            for r in hw_results
        ]
        self._hardware_results = hw_results
        
        if not hw_passed:
            logger.critical("HARDWARE VALIDATION FAILED - Cannot proceed")
            results['overall'] = False
            self._checks_passed = False
            return False, results
        
        # Check 2: Network latency tests
        logger.info("\n[NETWORK LATENCY TESTS]")
        try:
            network_results = await test_network_latency()
            results['network']['passed'] = True
            results['network']['details'] = {
                'baselines_us': network_results.get('baselines', {}),
                'slippage_tolerances': network_results.get('slippage_tolerances', {}),
            }
            self._network_results = network_results
            
            # Check if any critical endpoints failed
            critical_failures = self._check_critical_endpoints(network_results)
            if critical_failures:
                logger.warning(f"Critical endpoint failures: {critical_failures}")
                results['network']['passed'] = False
                
        except Exception as e:
            logger.error(f"Network tests failed: {e}")
            results['network']['passed'] = False
            results['network']['details'] = {'error': str(e)}
        
        # Check 3: Rust IPC handshake
        logger.info("\n[RUST IPC HANDSHAKE]")
        ipc_ok = await self._test_ipc_handshake()
        results['ipc_handshake']['passed'] = ipc_ok
        results['ipc_handshake']['details'] = "Handshake successful" if ipc_ok else "Handshake failed"
        self._ipc_handshake_ok = ipc_ok
        
        # Determine overall result
        all_passed = (
            results['hardware']['passed'] and
            results['network']['passed'] and
            results['ipc_handshake']['passed']
        )
        
        results['overall'] = all_passed
        self._checks_passed = all_passed
        
        # Log final result
        logger.info("\n" + "=" * 60)
        if all_passed:
            logger.info("ALL PRE-FLIGHT CHECKS PASSED - Ready for daemon start")
        else:
            logger.critical("PRE-FLIGHT CHECKS FAILED - Daemon startup blocked")
        logger.info("=" * 60)
        
        return all_passed, results
    
    def _check_critical_endpoints(self, network_results: Dict) -> List[str]:
        """Check if any critical endpoints failed."""
        critical = []
        baselines = network_results.get('baselines', {})
        
        # Define critical endpoints and max acceptable latency
        critical_thresholds = {
            'binance_spot': 100000,  # 100ms max
            'binance_futures': 100000,
        }
        
        for name, threshold in critical_thresholds.items():
            baseline = baselines.get(name, float('inf'))
            if baseline > threshold:
                critical.append(f"{name}: {baseline:.0f}μs > {threshold}μs")
        
        return critical
    
    async def _test_ipc_handshake(self) -> bool:
        """Test IPC handshake with Rust side."""
        try:
            # Send handshake request
            self._handshake_socket.send_json({
                'type': 'HANDSHAKE_REQUEST',
                'python_version': '3.x',
                'timestamp': datetime.utcnow().isoformat(),
            })
            
            # Wait for response with timeout
            self._handshake_socket.setsockopt(zmq.RCVTIMEO, 5000)
            
            try:
                response = self._handshake_socket.recv_json()
                
                if response.get('type') == 'HANDSHAKE_ACK':
                    logger.info("Rust IPC handshake successful")
                    return True
                else:
                    logger.warning(f"Unexpected handshake response: {response}")
                    return False
                    
            except zmq.Again:
                logger.error("Rust IPC handshake timeout")
                return False
                
        except Exception as e:
            logger.error(f"IPC handshake failed: {e}")
            return False
    
    def can_start_daemon(self) -> bool:
        """Check if daemon is allowed to start."""
        return self._checks_passed
    
    def get_status(self) -> Dict[str, Any]:
        """Get pre-flight status."""
        return {
            'running': self._running,
            'checks_passed': self._checks_passed,
            'hardware_validated': len(self._hardware_results) > 0,
            'network_tested': bool(self._network_results),
            'ipc_handshake_ok': self._ipc_handshake_ok,
        }
    
    async def shutdown(self):
        """Gracefully shutdown the pre-flight module."""
        logger.info("Shutting down PreflightModule...")
        self._running = False
        
        if self.hardware_validator:
            self.hardware_validator.shutdown()
        
        if self.network_tester:
            self.network_tester.shutdown()
        
        if self._handshake_socket:
            self._handshake_socket.close()
        if self._zmq_context:
            self._zmq_context.term()
        
        logger.info("PreflightModule shut down complete")


# Global module instance
_module: Optional[PreflightModule] = None


def get_module() -> PreflightModule:
    """Get or create the global PreflightModule instance."""
    global _module
    if _module is None:
        _module = PreflightModule({})
    return _module


def create_module(config: Dict[str, Any]) -> PreflightModule:
    """Create a new PreflightModule with custom configuration."""
    global _module
    _module = PreflightModule(config)
    return _module


async def run_preflight_checks(config: Optional[Dict[str, Any]] = None) -> Tuple[bool, Dict]:
    """
    Convenience function to run all pre-flight checks.
    
    Returns:
        Tuple of (success, results_dict)
    """
    module = get_module()
    
    if not module._running:
        await module.initialize()
    
    return await module.run_all_checks()
