"""
Compliance Module Root - Wires compliance checks into Nautilus DEX execution router.
Integrates OFAC and mixer detection with Rust IPC bridge for real-time blocking.
"""
import numpy as np
import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import time

logger = logging.getLogger(__name__)

# Import compliance submodules
try:
    from .ofac_checker import OFACChecker, get_ofac_checker
    from .mixer_detector import MixerDetector, get_mixer_detector
except ImportError as e:
    logger.warning(f"Compliance submodules not fully available: {e}")
    OFACChecker = None
    MixerDetector = None


class ComplianceRouter:
    """
    Central compliance router that integrates all checks.
    Routes decisions to Nautilus DEX execution engine via IPC.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize checkers
        self.ofac_checker = None
        self.mixer_detector = None
        
        if OFACChecker is not None:
            self.ofac_checker = get_ofac_checker({
                'db_path': self.config.get('ofac_db_path', 'data/ofac_bloom.bin'),
                'auto_populate': self.config.get('ofac_auto_populate', True)
            })
            logger.info("OFAC checker initialized")
        
        if MixerDetector is not None:
            self.mixer_detector = get_mixer_detector({
                'graph_max_nodes': self.config.get('mixer_graph_max', 100_000),
                'risk_threshold': self.config.get('mixer_risk_threshold', 0.5),
                'auto_populate': self.config.get('mixer_auto_populate', True)
            })
            logger.info("Mixer detector initialized")
        
        # Statistics
        self._total_checks = 0
        self._total_blocks = 0
        self._ofac_blocks = 0
        self._mixer_blocks = 0
        
        # IPC bridge state
        self._ipc_connected = False
        self._rust_ready = False
        
        # Pre-allocated result buffer for zero-copy
        self._result_buffer = np.zeros(4, dtype=np.int32)
        # [action_code, ofac_score, mixer_score, combined_risk]
        
        logger.info("ComplianceRouter initialized")
    
    def check_transaction(self, from_addr: str, to_addr: str,
                          amount: float = 0.0, chain: str = 'evm') -> Dict[str, Any]:
        """
        Perform comprehensive compliance check on a transaction.
        
        Args:
            from_addr: Sender address
            to_addr: Receiver address
            amount: Transaction amount
            chain: Blockchain (evm/solana)
            
        Returns:
            Comprehensive compliance decision
        """
        self._total_checks += 1
        start_time = time.perf_counter()
        
        results = {
            'from_address': from_addr,
            'to_address': to_addr,
            'amount': amount,
            'chain': chain,
            'timestamp': time.time(),
            'checks': {},
            'decision': 'ALLOW',
            'block_reasons': [],
            'latency_ms': 0.0
        }
        
        # OFAC check on both addresses
        if self.ofac_checker is not None:
            from_ofac = self.ofac_checker.check(from_addr, chain)
            to_ofac = self.ofac_checker.check(to_addr, chain)
            
            results['checks']['ofac'] = {
                'from': from_ofac,
                'to': to_ofac
            }
            
            if from_ofac.get('is_sanctioned'):
                results['decision'] = 'BLOCK'
                results['block_reasons'].append(f"From address sanctioned: {from_addr}")
                self._ofac_blocks += 1
            
            if to_ofac.get('is_sanctioned'):
                results['decision'] = 'BLOCK'
                results['block_reasons'].append(f"To address sanctioned: {to_addr}")
                self._ofac_blocks += 1
        
        # Mixer detection
        if self.mixer_detector is not None:
            from_mixer = self.mixer_detector.analyze(from_addr)
            to_mixer = self.mixer_detector.analyze(to_addr)
            
            results['checks']['mixer'] = {
                'from': from_mixer,
                'to': to_mixer
            }
            
            if from_mixer.get('is_high_risk'):
                results['decision'] = 'BLOCK'
                results['block_reasons'].append(f"From address high mixer risk: {from_addr}")
                self._mixer_blocks += 1
            
            if to_mixer.get('is_high_risk'):
                results['decision'] = 'BLOCK'
                results['block_reasons'].append(f"To address high mixer risk: {to_addr}")
                self._mixer_blocks += 1
        
        # Calculate combined risk score
        ofac_risk = max(
            results['checks'].get('ofac', {}).get('from', {}).get('is_sanctioned', False),
            results['checks'].get('ofac', {}).get('to', {}).get('is_sanctioned', False)
        )
        
        mixer_risk = max(
            results['checks'].get('mixer', {}).get('from', {}).get('risk_score', 0.0),
            results['checks'].get('mixer', {}).get('to', {}).get('risk_score', 0.0)
        )
        
        results['risk_scores'] = {
            'ofac': 1.0 if ofac_risk else 0.0,
            'mixer': mixer_risk,
            'combined': max(ofac_risk, mixer_risk)
        }
        
        # Update block count
        if results['decision'] == 'BLOCK':
            self._total_blocks += 1
        
        # Calculate latency
        results['latency_ms'] = (time.perf_counter() - start_time) * 1000
        
        # Update zero-copy result buffer
        action_code = 0 if results['decision'] == 'ALLOW' else 1
        self._result_buffer[0] = action_code
        self._result_buffer[1] = int(results['risk_scores']['ofac'] * 1000)
        self._result_buffer[2] = int(results['risk_scores']['mixer'] * 1000)
        self._result_buffer[3] = int(results['risk_scores']['combined'] * 1000)
        
        return results
    
    def check_settlement_batch(self, transactions: List[Dict]) -> List[Dict]:
        """
        Check multiple transactions efficiently.
        
        Args:
            transactions: List of transaction dicts with from/to/amount/chain
            
        Returns:
            List of compliance decisions
        """
        return [
            self.check_transaction(
                tx.get('from', ''),
                tx.get('to', ''),
                tx.get('amount', 0.0),
                tx.get('chain', 'evm')
            )
            for tx in transactions
        ]
    
    def get_result_buffer(self) -> np.ndarray:
        """Return zero-copy view of result buffer."""
        return self._result_buffer
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get compliance statistics."""
        ofac_stats = self.ofac_checker.get_statistics() if self.ofac_checker else {}
        mixer_stats = self.mixer_detector.get_statistics() if self.mixer_detector else {}
        
        return {
            'total_checks': self._total_checks,
            'total_blocks': self._total_blocks,
            'ofac_blocks': self._ofac_blocks,
            'mixer_blocks': self._mixer_blocks,
            'block_rate': self._total_blocks / max(1, self._total_checks),
            'ofac_stats': ofac_stats,
            'mixer_stats': mixer_stats,
            'ipc_connected': self._ipc_connected,
            'rust_ready': self._rust_ready
        }
    
    def connect_ipc(self, ipc_path: str) -> bool:
        """
        Connect to Rust IPC bridge for Nautilus integration.
        
        Args:
            ipc_path: Path to Unix domain socket or named pipe
            
        Returns:
            Success status
        """
        try:
            # IPC connection logic would go here
            # This is a placeholder for the actual IPC implementation
            self._ipc_connected = True
            self._rust_ready = True
            logger.info(f"Connected to Rust IPC at {ipc_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect IPC: {e}")
            return False
    
    def send_to_rust(self, decision: Dict) -> bool:
        """
        Send compliance decision to Rust orchestrator.
        
        Args:
            decision: Compliance decision dictionary
            
        Returns:
            Success status
        """
        if not self._ipc_connected:
            logger.warning("IPC not connected, cannot send decision")
            return False
        
        try:
            # Serialize and send via IPC
            # Placeholder for actual IPC send logic
            logger.debug(f"Sent decision to Rust: {decision['decision']}")
            return True
        except Exception as e:
            logger.error(f"Failed to send to Rust: {e}")
            return False
    
    def receive_rust_ready_flag(self) -> bool:
        """Check if Rust side is ready for trading."""
        # Placeholder - would read from shared memory
        return self._rust_ready
    
    def warmup(self) -> None:
        """Warm up all compliance checkers."""
        if self.ofac_checker:
            # Warm up Bloom filter
            _ = self.ofac_checker.check('0x0000000000000000000000000000000000000000')
        
        if self.mixer_detector:
            # Warm up graph
            _ = self.mixer_detector.analyze('0x0000000000000000000000000000000000000000')
        
        logger.info("ComplianceRouter warmed up")
    
    def close(self) -> None:
        """Clean up resources."""
        if self.ofac_checker:
            self.ofac_checker.close()
        
        logger.info(f"ComplianceRouter closed. Total checks: {self._total_checks}, "
                   f"blocks: {self._total_blocks}")


# Singleton instance
_compliance_router: Optional[ComplianceRouter] = None


def get_compliance_router(config: Optional[Dict[str, Any]] = None) -> ComplianceRouter:
    """Get or create singleton ComplianceRouter instance."""
    global _compliance_router
    if _compliance_router is None:
        _compliance_router = ComplianceRouter(config)
    return _compliance_router


def reset_compliance_router() -> None:
    """Reset singleton instance."""
    global _compliance_router
    if _compliance_router is not None:
        _compliance_router.close()
    _compliance_router = None


# Integration hooks for Nautilus DEX
def nautilus_pre_trade_check(from_addr: str, to_addr: str, 
                             amount: float, chain: str = 'evm') -> Tuple[bool, str]:
    """
    Nautilus DEX pre-trade compliance hook.
    Called before every trade execution.
    
    Returns:
        Tuple of (allowed, reason)
    """
    router = get_compliance_router()
    result = router.check_transaction(from_addr, to_addr, amount, chain)
    
    allowed = result['decision'] == 'ALLOW'
    reason = result['block_reasons'][0] if result['block_reasons'] else 'OK'
    
    return allowed, reason


__all__ = [
    'ComplianceRouter',
    'get_compliance_router',
    'reset_compliance_router',
    'nautilus_pre_trade_check'
]
