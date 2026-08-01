"""
RL Market Making Module Root.
Serves the trained RL calibration policy via Ray Serve to feed optimal
parameters to Nautilus portfolio manager.

Provides:
- Policy serving endpoint
- Real-time parameter calibration
- Integration with market making execution engine
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import threading
import logging
import time
import json
from pathlib import Path
from dataclasses import dataclass, asdict

from .avellaneda_rl import (
    AvellanedaStoikovEnvironment,
    RLMarketMakingAgent,
    MarketMakingAction,
    get_rl_agent
)
from .reward_shaping import RewardShaper, ToxicInventoryDetector, RewardComponents

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CalibratedParameters:
    """Calibrated market making parameters."""
    timestamp_ns: int
    gamma: float           # Risk aversion
    kappa: float           # Order arrival intensity
    spread_half_bps: float # Half-spread in basis points
    skew: float            # Quote skew
    confidence: float      # Policy confidence score
    
    # Derived values
    bid_quote: Optional[float] = None
    ask_quote: Optional[float] = None
    
    # Metadata
    vpin_context: float = 0.0
    inventory_context: float = 0.0
    regime_context: str = "normal"


@dataclass
class CalibrationRequest:
    """Request for parameter calibration."""
    mid_price: float
    inventory: float
    max_inventory: float
    vpin: float
    volatility: float
    spread_bps: float
    time_remaining: float = 1.0
    recent_pnl: float = 0.0


class RLCalibrationService:
    """
    Service for serving RL-calibrated market making parameters.
    Can be deployed via Ray Serve for distributed access.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        enable_reward_shaping: bool = True
    ):
        self._lock = threading.RLock()
        self.model_path = model_path
        
        # Initialize environment and agent
        self.env = AvellanedaStoikovEnvironment()
        self.agent = RLMarketMakingAgent(self.env)
        
        # Load model if path provided
        if model_path:
            self.load_policy(model_path)
        
        # Reward shaping components
        self.enable_reward_shaping = enable_reward_shaping
        self.reward_shaper = RewardShaper() if enable_reward_shaping else None
        self.toxicity_detector = ToxicInventoryDetector()
        
        # State tracking
        self._last_calibration: Optional[CalibratedParameters] = None
        self._calibration_count = 0
        self._price_history: List[float] = []
        self._inventory_history: List[float] = []
        
        # Performance metrics
        self._total_latency_us = 0
        self._last_reset = time.time()
    
    def load_policy(self, model_path: str) -> bool:
        """Load trained policy from file."""
        try:
            # In production, this would load ONNX weights or numpy arrays
            # For now, we simulate loading by resetting agent state
            policy_data = np.load(model_path)
            
            with self._lock:
                if 'weights' in policy_data:
                    self.agent._policy_weights = policy_data['weights']
                if 'bias' in policy_data:
                    self.agent._policy_bias = policy_data['bias']
            
            logger.info(f"Loaded policy from {model_path}")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load policy from {model_path}: {e}")
            return False
    
    def save_policy(self, model_path: str) -> bool:
        """Save current policy to file."""
        try:
            with self._lock:
                np.savez(
                    model_path,
                    weights=self.agent._policy_weights,
                    bias=self.agent._policy_bias
                )
            
            logger.info(f"Saved policy to {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save policy: {e}")
            return False
    
    def calibrate(
        self,
        request: CalibrationRequest
    ) -> CalibratedParameters:
        """
        Calibrate market making parameters based on current state.
        
        Args:
            request: Current market and position state
            
        Returns:
            Calibrated parameters with quotes
        """
        start_time = time.perf_counter_ns()
        
        with self._lock:
            # Update environment state
            self.env.mid_price = request.mid_price
            self.env.inventory = request.inventory
            self.env.vpin = request.vpin
            self.env.volatility = request.volatility
            self.env.spread_bps = request.spread_bps
            
            # Get state features
            state_features = self.env.get_state_features()
            
            # Select action (no exploration during inference)
            action = self.agent.select_action(state_features, explore=False)
            
            # Calculate quotes
            half_spread_abs = request.mid_price * action.spread_half / 10000
            bid_quote = request.mid_price - half_spread_abs + action.skew
            ask_quote = request.mid_price + half_spread_abs + action.skew
            
            # Determine regime context
            regime = self._determine_regime(request.vpin, request.volatility)
            
            # Calculate confidence based on state familiarity
            confidence = self._calculate_confidence(state_features)
            
            # Update tracking
            self._price_history.append(request.mid_price)
            self._inventory_history.append(request.inventory)
            
            if len(self._price_history) > 100:
                self._price_history.pop(0)
                self._inventory_history.pop(0)
            
            # Update toxicity detector
            if len(self._inventory_history) >= 2:
                trade_dir = np.sign(request.inventory - self._inventory_history[-2]) if len(self._inventory_history) >= 2 else 0
                self.toxicity_detector.update(
                    request.mid_price,
                    request.inventory,
                    int(trade_dir)
                )
            
            result = CalibratedParameters(
                timestamp_ns=time.time_ns(),
                gamma=action.gamma,
                kappa=action.kappa,
                spread_half_bps=action.spread_half,
                skew=action.skew,
                confidence=confidence,
                bid_quote=float(bid_quote),
                ask_quote=float(ask_quote),
                vpin_context=request.vpin,
                inventory_context=request.inventory / max(request.max_inventory, 1e-8),
                regime_context=regime
            )
            
            self._last_calibration = result
            self._calibration_count += 1
            
            # Update latency metrics
            end_time = time.perf_counter_ns()
            self._total_latency_us += (end_time - start_time) // 1000
            
            return result
    
    def _determine_regime(
        self,
        vpin: float,
        volatility: float
    ) -> str:
        """Determine market regime based on conditions."""
        if vpin > 0.7:
            return "toxic"
        elif volatility > 0.001:  # High volatility
            return "volatile"
        elif vpin < 0.3 and volatility < 0.0003:
            return "calm"
        else:
            return "normal"
    
    def _calculate_confidence(self, state_features: np.ndarray) -> float:
        """
        Calculate confidence score for current calibration.
        Based on how familiar the state is compared to training distribution.
        """
        # Simple heuristic: confidence decreases with extreme feature values
        feature_norms = np.abs(state_features)
        
        # Penalize extreme values
        penalties = []
        
        # High inventory ratio
        if feature_norms[0] > 0.8:  # inventory_risk
            penalties.append(0.3)
        
        # High VPIN
        if feature_norms[1] > 0.7:  # vpin
            penalties.append(0.2)
        
        # Very high volatility
        if feature_norms[2] > 50:  # volatility (scaled)
            penalties.append(0.2)
        
        # Very little time remaining (end of day uncertainty)
        if feature_norms[3] < 0.1:  # time_remaining
            penalties.append(0.1)
        
        base_confidence = 0.9
        return float(np.clip(base_confidence - sum(penalties), 0.3, 0.95))
    
    def update_policy_from_feedback(
        self,
        request: CalibrationRequest,
        action_taken: MarketMakingAction,
        reward_received: float
    ) -> None:
        """
        Update policy based on real-world feedback.
        Enables online learning from actual trading results.
        """
        with self._lock:
            # Get state features
            self.env.mid_price = request.mid_price
            self.env.inventory = request.inventory
            state_features = self.env.get_state_features()
            
            # Single-step policy update
            self.agent._policy_weights += (
                self.agent.lr * 
                np.outer(state_features, np.array([
                    action_taken.gamma,
                    action_taken.kappa,
                    action_taken.spread_half,
                    action_taken.skew
                ])) * reward_received * 0.01  # Scale down for online learning
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get service statistics."""
        with self._lock:
            elapsed = time.time() - self._last_reset
            avg_latency = self._total_latency_us / max(self._calibration_count, 1)
            
            stats = {
                'calibrations_performed': self._calibration_count,
                'avg_latency_us': avg_latency,
                'calibrations_per_second': self._calibration_count / max(elapsed, 1e-6),
                'policy_stats': self.agent.get_policy_stats()
            }
            
            if self._last_calibration:
                stats['last_calibration'] = asdict(self._last_calibration)
            
            if self.enable_reward_shaping:
                stats['reward_stats'] = self.reward_shaper.get_reward_statistics()
            
            stats['toxicity_score'] = self.toxicity_detector.get_toxicity_score()
            
            return stats
    
    def reset(self) -> None:
        """Reset service state."""
        with self._lock:
            self.env.reset()
            self.reward_shaper.reset() if self.reward_shaper else None
            self.toxicity_detector.reset()
            self._price_history.clear()
            self._inventory_history.clear()
            self._last_calibration = None
    
    def shutdown(self) -> None:
        """Shutdown service gracefully."""
        logger.info("RL Calibration Service shutting down")
        self.reset()


# Ray Serve deployment class (optional, for distributed serving)
try:
    from ray import serve
    
    @serve.deployment
    class RLCalibrationDeployment:
        """Ray Serve deployment for RL calibration service."""
        
        def __init__(self, model_path: Optional[str] = None):
            self.service = RLCalibrationService(model_path)
        
        async def calibrate(self, request_dict: Dict[str, Any]) -> Dict[str, Any]:
            """Async calibration endpoint."""
            request = CalibrationRequest(**request_dict)
            result = self.service.calibrate(request)
            return asdict(result)
        
        async def get_stats(self) -> Dict[str, Any]:
            """Get service statistics."""
            return self.service.get_statistics()
    
    RAY_SERVE_AVAILABLE = True
    
except ImportError:
    RAY_SERVE_AVAILABLE = False
    logger.debug("Ray Serve not available, using local service only")


# Global singleton instance
_rl_service_instance: Optional[RLCalibrationService] = None
_rl_service_lock = threading.Lock()


def get_rl_calibration_service(
    model_path: Optional[str] = None
) -> RLCalibrationService:
    """Thread-safe singleton access to RL calibration service."""
    global _rl_service_instance
    
    with _rl_service_lock:
        if _rl_service_instance is None:
            _rl_service_instance = RLCalibrationService(model_path)
        elif model_path and _rl_service_instance.model_path != model_path:
            _rl_service_instance.load_policy(model_path)
        
        return _rl_service_instance


def calibrate_parameters(
    mid_price: float,
    inventory: float,
    max_inventory: float,
    vpin: float,
    volatility: float,
    spread_bps: float,
    **kwargs
) -> CalibratedParameters:
    """
    Convenience function for quick calibration.
    Uses global singleton service.
    """
    service = get_rl_calibration_service()
    request = CalibrationRequest(
        mid_price=mid_price,
        inventory=inventory,
        max_inventory=max_inventory,
        vpin=vpin,
        volatility=volatility,
        spread_bps=spread_bps,
        **kwargs
    )
    return service.calibrate(request)


if __name__ == "__main__":
    # Demo usage
    np.random.seed(42)
    
    service = get_rl_calibration_service()
    
    print("=== RL Market Making Calibration Demo ===\n")
    
    # Simulate calibration requests
    mid_price = 50000.0
    inventory = 0.0
    
    for step in range(10):
        # Simulate market changes
        mid_price += np.random.randn() * 10
        inventory += np.random.randn() * 5
        vpin = 0.3 + np.random.beta(2, 5) * 0.5
        volatility = 0.0002 + abs(np.random.randn()) * 0.0001
        
        request = CalibrationRequest(
            mid_price=mid_price,
            inventory=inventory,
            max_inventory=100.0,
            vpin=vpin,
            volatility=volatility,
            spread_bps=10.0
        )
        
        result = service.calibrate(request)
        
        print(f"Step {step + 1}:")
        print(f"  Mid Price: ${result.bid_quote:.2f} - ${result.ask_quote:.2f}")
        print(f"  Gamma: {result.gamma:.4f}, Kappa: {result.kappa:.4f}")
        print(f"  Spread: {result.spread_half_bps:.2f} bps, Skew: {result.skew:.4f}")
        print(f"  Regime: {result.regime_context}, Confidence: {result.confidence:.4f}")
        print()
    
    # Show statistics
    stats = service.get_statistics()
    print(f"Service Statistics:")
    print(f"  Calibrations: {stats['calibrations_performed']}")
    print(f"  Avg Latency: {stats['avg_latency_us']:.2f} µs")
    print(f"  Throughput: {stats['calibrations_per_second']:.2f} cal/s")
