"""
SMC Module Root - Smart Money Concepts Integration
Exports SMC structural probabilities to alpha ensemble via zero-copy numpy arrays.
"""
import numpy as np
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Import SMC detectors
try:
    from .order_block_detector import OrderBlockDetector
    from .liquidity_sweep import LiquiditySweepHMM
except ImportError as e:
    logger.warning(f"SMC submodules not fully available: {e}")
    OrderBlockDetector = None
    LiquiditySweepHMM = None


class SMCManager:
    """
    Central manager for Smart Money Concepts analysis.
    Aggregates signals from Order Block and Liquidity Sweep detectors.
    Uses zero-copy numpy arrays for efficient data transfer to alpha ensemble.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.ob_detector = None
        self.ls_detector = None
        self._structural_probs = None
        self._prob_buffer = None
        
        # Initialize detectors if available
        if OrderBlockDetector is not None:
            self.ob_detector = OrderBlockDetector(
                model_path=self.config.get('ob_model_path', 'models/order_block.onnx')
            )
            logger.info("OrderBlockDetector initialized")
            
        if LiquiditySweepHMM is not None:
            self.ls_detector = LiquiditySweepHMM(
                n_states=self.config.get('hmm_states', 3),
                model_path=self.config.get('hmm_model_path', 'models/liquidity_hmm.pkl')
            )
            logger.info("LiquiditySweepHMM initialized")
        
        # Pre-allocate shared memory buffer for zero-copy transfers
        # Format: [ob_prob, fvg_prob, sweep_prob, stop_hunt_prob, confidence]
        self._prob_buffer = np.zeros(5, dtype=np.float32)
        logger.info("SMCManager initialized with zero-copy buffers")
    
    def analyze_tick(self, footprint_data: np.ndarray) -> np.ndarray:
        """
        Analyze a single tick/bar of footprint data.
        Returns zero-copy view of structural probabilities array.
        
        Args:
            footprint_data: Raw footprint chart data (price, volume, delta)
            
        Returns:
            Zero-copy numpy array view of [ob_prob, fvg_prob, sweep_prob, stop_hunt_prob, confidence]
        """
        ob_prob = 0.0
        fvg_prob = 0.0
        sweep_prob = 0.0
        stop_hunt_prob = 0.0
        
        # Run Order Block detection
        if self.ob_detector is not None:
            ob_result = self.ob_detector.detect(footprint_data)
            ob_prob = ob_result.get('order_block_prob', 0.0)
            fvg_prob = ob_result.get('fvg_prob', 0.0)
        
        # Run Liquidity Sweep classification
        if self.ls_detector is not None:
            ls_result = self.ls_detector.classify(footprint_data)
            sweep_prob = ls_result.get('sweep_prob', 0.0)
            stop_hunt_prob = ls_result.get('stop_hunt_prob', 0.0)
        
        # Calculate overall confidence
        max_signal = max(ob_prob, fvg_prob, sweep_prob, stop_hunt_prob)
        confidence = max_signal * 0.9 if max_signal > 0.5 else max_signal * 0.5
        
        # Update buffer in-place (zero-copy)
        self._prob_buffer[0] = ob_prob
        self._prob_buffer[1] = fvg_prob
        self._prob_buffer[2] = sweep_prob
        self._prob_buffer[3] = stop_hunt_prob
        self._prob_buffer[4] = confidence
        
        return self._prob_buffer
    
    def get_structural_probs(self) -> np.ndarray:
        """Return zero-copy view of current structural probabilities."""
        return self._prob_buffer
    
    def get_signals_summary(self) -> Dict[str, float]:
        """Get human-readable summary of SMC signals."""
        probs = self._prob_buffer
        return {
            'order_block_probability': float(probs[0]),
            'fair_value_gap_probability': float(probs[1]),
            'liquidity_sweep_probability': float(probs[2]),
            'stop_hunt_probability': float(probs[3]),
            'overall_confidence': float(probs[4])
        }
    
    def warmup(self, historical_data: np.ndarray) -> None:
        """Warm up detectors with historical data."""
        if self.ob_detector is not None:
            self.ob_detector.warmup(historical_data)
        if self.ls_detector is not None:
            self.ls_detector.warmup(historical_data)
        logger.info("SMC detectors warmed up")
    
    def save_state(self, path: str) -> None:
        """Save detector states to disk."""
        state = {}
        if self.ob_detector is not None:
            state['ob_detector'] = self.ob_detector.save()
        if self.ls_detector is not None:
            state['ls_detector'] = self.ls_detector.save()
        
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        logger.info(f"SMC state saved to {path}")
    
    def load_state(self, path: str) -> None:
        """Load detector states from disk."""
        import pickle
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        if 'ob_detector' in state and self.ob_detector is not None:
            self.ob_detector.load(state['ob_detector'])
        if 'ls_detector' in state and self.ls_detector is not None:
            self.ls_detector.load(state['ls_detector'])
        logger.info(f"SMC state loaded from {path}")


# Singleton instance
_smc_manager: Optional[SMCManager] = None


def get_smc_manager(config: Optional[Dict[str, Any]] = None) -> SMCManager:
    """Get or create singleton SMCManager instance."""
    global _smc_manager
    if _smc_manager is None:
        _smc_manager = SMCManager(config)
    return _smc_manager


def reset_smc_manager() -> None:
    """Reset singleton instance (for testing/reconfiguration)."""
    global _smc_manager
    _smc_manager = None


__all__ = [
    'SMCManager',
    'get_smc_manager',
    'reset_smc_manager'
]
