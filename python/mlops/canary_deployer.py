"""
Canary Deployer - Routes 5% of live execution volume to shadow models.
Implements safe canary deployments with automatic rollback capabilities.
Memory-efficient design for HFT environment.
"""
import numpy as np
import logging
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
import time
import threading
from collections import deque

logger = logging.getLogger(__name__)


class CanaryRouter:
    """
    Routes traffic between production and canary models.
    Exactly 5% of volume goes to canary by default.
    """
    
    def __init__(self, canary_percentage: float = 0.05,
                 max_canary_samples: int = 10_000):
        self.canary_percentage = canary_percentage
        self.max_canary_samples = max_canary_samples
        
        # Traffic counters
        self._total_requests = 0
        self._canary_requests = 0
        self._production_requests = 0
        
        # Canary sample storage (bounded)
        self._canary_results: deque = deque(maxlen=max_canary_samples)
        self._production_results: deque = deque(maxlen=max_canary_samples)
        
        # Model references
        self._production_model = None
        self._canary_model = None
        
        # State
        self._canary_active = False
        self._lock = threading.Lock()
        
        # Random state for deterministic routing
        self._rng = np.random.RandomState(42)
        
        logger.info(f"CanaryRouter initialized with {canary_percentage*100:.1f}% canary traffic")
    
    def set_production_model(self, model: Any) -> None:
        """Set the production model."""
        self._production_model = model
        logger.info("Production model set")
    
    def set_canary_model(self, model: Any) -> None:
        """Set the canary (shadow) model."""
        self._canary_model = model
        logger.info("Canary model set")
    
    def activate_canary(self) -> bool:
        """Activate canary routing."""
        if self._canary_model is None:
            logger.warning("Cannot activate canary: no canary model set")
            return False
        
        with self._lock:
            self._canary_active = True
            logger.info(f"Canary activated: {self.canary_percentage*100:.1f}% traffic")
        
        return True
    
    def deactivate_canary(self) -> None:
        """Deactivate canary routing."""
        with self._lock:
            self._canary_active = False
            logger.info("Canary deactivated")
    
    def route_request(self, input_data: np.ndarray) -> tuple:
        """
        Route a request to production or canary model.
        
        Args:
            input_data: Model input
            
        Returns:
            Tuple of (result, is_canary, metadata)
        """
        with self._lock:
            self._total_requests += 1
            
            # Determine routing
            if not self._canary_active:
                # All traffic to production
                self._production_requests += 1
                is_canary = False
            elif self._rng.random() < self.canary_percentage:
                # Route to canary
                self._canary_requests += 1
                is_canary = True
            else:
                # Route to production
                self._production_requests += 1
                is_canary = False
        
        # Execute model inference
        start_time = time.perf_counter()
        
        if is_canary and self._canary_model is not None:
            result = self._infer_model(self._canary_model, input_data)
            model_type = 'canary'
        elif self._production_model is not None:
            result = self._infer_model(self._production_model, input_data)
            model_type = 'production'
        else:
            result = None
            model_type = 'none'
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Store results for A/B testing
        metadata = {
            'timestamp': time.time(),
            'model_type': model_type,
            'latency_ms': latency_ms,
            'request_id': self._total_requests
        }
        
        self._store_result(result, is_canary, metadata)
        
        return result, is_canary, metadata
    
    def _infer_model(self, model: Any, input_data: np.ndarray) -> Any:
        """Execute model inference with error handling."""
        try:
            if hasattr(model, 'predict'):
                return model.predict(input_data)
            elif hasattr(model, 'inference'):
                return model.inference(input_data)
            elif callable(model):
                return model(input_data)
            else:
                logger.warning("Model has no callable interface")
                return None
        except Exception as e:
            logger.error(f"Model inference failed: {e}")
            return None
    
    def _store_result(self, result: Any, is_canary: bool, metadata: Dict) -> None:
        """Store result for later analysis."""
        entry = {
            'result': result,
            'metadata': metadata,
            'is_canary': is_canary
        }
        
        if is_canary:
            self._canary_results.append(entry)
        else:
            self._production_results.append(entry)
    
    def get_canary_results(self) -> List[Dict]:
        """Get all canary results."""
        return list(self._canary_results)
    
    def get_production_results(self) -> List[Dict]:
        """Get all production results."""
        return list(self._production_results)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get routing statistics."""
        with self._lock:
            return {
                'total_requests': self._total_requests,
                'canary_requests': self._canary_requests,
                'production_requests': self._production_requests,
                'canary_percentage_actual': (
                    self._canary_requests / max(1, self._total_requests)
                ),
                'canary_active': self._canary_active,
                'canary_samples_stored': len(self._canary_results),
                'production_samples_stored': len(self._production_results)
            }
    
    def reset_counters(self) -> None:
        """Reset all counters and stored results."""
        with self._lock:
            self._total_requests = 0
            self._canary_requests = 0
            self._production_requests = 0
            self._canary_results.clear()
            self._production_results.clear()
            logger.info("CanaryRouter counters reset")
    
    def promote_canary(self) -> bool:
        """Promote canary model to production."""
        with self._lock:
            if self._canary_model is None:
                logger.warning("Cannot promote: no canary model")
                return False
            
            # Swap models
            self._production_model = self._canary_model
            self._canary_model = None
            self._canary_active = False
            
            logger.info("Canary promoted to production")
            return True
    
    def rollback(self, previous_model: Any) -> bool:
        """Rollback to previous production model."""
        with self._lock:
            if previous_model is None:
                logger.warning("Cannot rollback: no previous model")
                return False
            
            self._production_model = previous_model
            self._canary_active = False
            
            logger.info("Rolled back to previous production model")
            return True


class CanaryDeployer:
    """
    Manages canary deployment lifecycle.
    Handles deployment, monitoring, and promotion/rollback decisions.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize router
        self.router = CanaryRouter(
            canary_percentage=self.config.get('canary_percentage', 0.05),
            max_canary_samples=self.config.get('max_canary_samples', 10_000)
        )
        
        # Deployment state
        self._deployment_id = None
        self._start_time = None
        self._previous_model = None
        self._current_model = None
        self._canary_model = None
        
        # Monitoring
        self._health_checks_passed = 0
        self._health_checks_failed = 0
        
        # Callbacks
        self._on_promote_callback: Optional[Callable] = None
        self._on_rollback_callback: Optional[Callable] = None
        
        logger.info("CanaryDeployer initialized")
    
    def deploy(self, new_model: Any, deployment_id: str = None) -> bool:
        """
        Deploy a new model as canary.
        
        Args:
            new_model: New model to deploy
            deployment_id: Optional deployment identifier
            
        Returns:
            Success status
        """
        if self.router._production_model is None:
            logger.warning("No production model set, cannot deploy canary")
            return False
        
        self._deployment_id = deployment_id or f"deploy_{int(time.time())}"
        self._start_time = time.time()
        self._previous_model = self.router._production_model
        self._canary_model = new_model
        
        # Set canary model
        self.router.set_canary_model(new_model)
        
        # Activate canary routing
        if not self.router.activate_canary():
            logger.error("Failed to activate canary")
            return False
        
        logger.info(f"Deployed canary: {self._deployment_id}")
        return True
    
    def check_health(self, metrics: Dict[str, float]) -> bool:
        """
        Perform health check on canary deployment.
        
        Args:
            metrics: Current deployment metrics
            
        Returns:
            Health check passed status
        """
        # Check error rate
        error_rate = metrics.get('error_rate', 0.0)
        if error_rate > self.config.get('max_error_rate', 0.01):
            self._health_checks_failed += 1
            logger.warning(f"Health check failed: high error rate ({error_rate:.4f})")
            return False
        
        # Check latency
        p99_latency = metrics.get('p99_latency_ms', 0.0)
        if p99_latency > self.config.get('max_latency_ms', 100.0):
            self._health_checks_failed += 1
            logger.warning(f"Health check failed: high latency ({p99_latency:.2f}ms)")
            return False
        
        # Check divergence from production
        divergence = metrics.get('divergence', 0.0)
        if divergence > self.config.get('max_divergence', 0.1):
            self._health_checks_failed += 1
            logger.warning(f"Health check failed: high divergence ({divergence:.4f})")
            return False
        
        self._health_checks_passed += 1
        return True
    
    def promote(self) -> bool:
        """Promote canary to production."""
        if self._on_promote_callback:
            self._on_promote_callback(self._deployment_id)
        
        success = self.router.promote_canary()
        
        if success:
            self._current_model = self._canary_model
            self._canary_model = None
            logger.info(f"Promoted canary: {self._deployment_id}")
        
        return success
    
    def rollback(self) -> bool:
        """Rollback canary deployment."""
        if self._on_rollback_callback:
            self._on_rollback_callback(self._deployment_id)
        
        success = self.router.rollback(self._previous_model)
        
        if success:
            logger.info(f"Rolled back deployment: {self._deployment_id}")
            self._canary_model = None
        
        return success
    
    def get_deployment_status(self) -> Dict[str, Any]:
        """Get current deployment status."""
        router_stats = self.router.get_statistics()
        
        return {
            'deployment_id': self._deployment_id,
            'status': 'active' if self.router._canary_active else 'inactive',
            'start_time': self._start_time,
            'duration_seconds': time.time() - self._start_time if self._start_time else 0,
            'health_checks_passed': self._health_checks_passed,
            'health_checks_failed': self._health_checks_failed,
            'router_stats': router_stats
        }
    
    def set_promote_callback(self, callback: Callable) -> None:
        """Set callback for promotion events."""
        self._on_promote_callback = callback
    
    def set_rollback_callback(self, callback: Callable) -> None:
        """Set callback for rollback events."""
        self._on_rollback_callback = callback
    
    def close(self) -> None:
        """Clean up deployment."""
        self.router.deactivate_canary()
        logger.info("CanaryDeployer closed")


# Singleton instance
_canary_deployer: Optional[CanaryDeployer] = None


def get_canary_deployer(config: Optional[Dict[str, Any]] = None) -> CanaryDeployer:
    """Get or create singleton CanaryDeployer instance."""
    global _canary_deployer
    if _canary_deployer is None:
        _canary_deployer = CanaryDeployer(config)
    return _canary_deployer


def reset_canary_deployer() -> None:
    """Reset singleton instance."""
    global _canary_deployer
    if _canary_deployer is not None:
        _canary_deployer.close()
    _canary_deployer = None


__all__ = [
    'CanaryRouter',
    'CanaryDeployer',
    'get_canary_deployer',
    'reset_canary_deployer'
]
