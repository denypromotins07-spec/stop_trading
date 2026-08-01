"""
Advanced MLOps Module Root - Manages automated promotion/rollback of canary models.
Monitors live PnL divergence and triggers actions based on A/B test results.
Integrates with Rust orchestrator for seamless model updates.
"""
import numpy as np
import logging
from typing import Dict, Any, Optional, Callable, List
from pathlib import Path
import time
import threading

logger = logging.getLogger(__name__)

# Import MLOps submodules
try:
    from .canary_deployer import CanaryDeployer, get_canary_deployer
    from .ab_tester import ABTester, get_ab_tester
except ImportError as e:
    logger.warning(f"MLOps submodules not fully available: {e}")
    CanaryDeployer = None
    ABTester = None


class MLOpsManager:
    """
    Central manager for advanced MLOps operations.
    Coordinates canary deployments, A/B testing, and automated decisions.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize submodules
        self.canary_deployer = None
        self.ab_tester = None
        
        if CanaryDeployer is not None:
            self.canary_deployer = get_canary_deployer({
                'canary_percentage': self.config.get('canary_percentage', 0.05),
                'max_canary_samples': self.config.get('max_canary_samples', 10_000),
                'max_error_rate': self.config.get('max_error_rate', 0.01),
                'max_latency_ms': self.config.get('max_latency_ms', 100.0)
            })
            logger.info("CanaryDeployer initialized")
        
        if ABTester is not None:
            self.ab_tester = get_ab_tester({
                'alpha': self.config.get('alpha', 0.05),
                'beta': self.config.get('beta', 0.10),
                'effect_size': self.config.get('effect_size', 0.01),
                'max_pnl_divergence': self.config.get('max_pnl_divergence', 0.05)
            })
            logger.info("ABTester initialized")
        
        # State
        self._deployment_active = False
        self._auto_promote_enabled = self.config.get('auto_promote', True)
        self._auto_rollback_enabled = self.config.get('auto_rollback', True)
        
        # Monitoring thread
        self._monitor_thread: Optional[threading.Thread] = None
        self._monitor_running = False
        
        # Callbacks
        self._on_promote: Optional[Callable] = None
        self._on_rollback: Optional[Callable] = None
        self._on_alert: Optional[Callable] = None
        
        # Statistics
        self._total_deployments = 0
        self._successful_promotions = 0
        self._rollbacks = 0
        
        # PnL tracking
        self._pnl_baseline = 0.0
        self._pnl_canary = 0.0
        self._pnl_divergence_threshold = self.config.get('pnl_divergence_threshold', 0.1)
        
        logger.info("MLOpsManager initialized")
    
    def deploy_model(self, new_model: Any, model_id: str = None) -> bool:
        """
        Deploy a new model as canary.
        
        Args:
            new_model: Model to deploy
            model_id: Optional model identifier
            
        Returns:
            Success status
        """
        if self.canary_deployer is None:
            logger.error("CanaryDeployer not available")
            return False
        
        deployment_id = model_id or f"model_{int(time.time())}"
        
        success = self.canary_deployer.deploy(new_model, deployment_id)
        
        if success:
            self._deployment_active = True
            self._total_deployments += 1
            
            # Start A/B test
            if self.ab_tester is not None:
                self.ab_tester.start_test()
            
            # Set up callbacks
            self.canary_deployer.set_promote_callback(self._handle_promotion)
            self.canary_deployer.set_rollback_callback(self._handle_rollback)
            
            logger.info(f"Deployed model: {deployment_id}")
        
        return success
    
    def record_inference(self, production_result: Any, canary_result: Any,
                         pnl_production: float = 0.0, pnl_canary: float = 0.0) -> None:
        """
        Record inference results for A/B testing.
        
        Args:
            production_result: Result from production model
            canary_result: Result from canary model
            pnl_production: PnL from production trade
            pnl_canary: PnL from canary trade
        """
        if self.ab_tester is None or not self._deployment_active:
            return
        
        # Calculate metrics (e.g., prediction accuracy, Sharpe ratio contribution)
        prod_metric = self._calculate_metric(production_result, pnl_production)
        canary_metric = self._calculate_metric(canary_result, pnl_canary)
        
        # Record in A/B tester
        decision = self.ab_tester.record_metrics(
            prod_metric, canary_metric,
            pnl_production, pnl_canary
        )
        
        # Update PnL tracking
        self._pnl_baseline += pnl_production
        self._pnl_canary += pnl_canary
        
        # Check for automatic actions
        if decision and self._auto_promote_enabled:
            self._evaluate_auto_action()
    
    def _calculate_metric(self, result: Any, pnl: float) -> float:
        """Calculate metric for A/B comparison."""
        # Default: use PnL as metric
        if pnl != 0.0:
            return pnl
        
        # Fallback: use result magnitude
        if isinstance(result, (int, float)):
            return abs(float(result))
        elif isinstance(result, np.ndarray):
            return float(np.mean(np.abs(result)))
        
        return 0.0
    
    def _evaluate_auto_action(self) -> None:
        """Evaluate if automatic promotion or rollback is needed."""
        if self.ab_tester is None:
            return
        
        recommendation = self.ab_tester.get_recommendation()
        
        if recommendation == 'PROMOTE' and self._auto_promote_enabled:
            logger.info("Auto-promotion triggered by A/B test")
            self.promote()
        
        elif recommendation == 'ROLLBACK' and self._auto_rollback_enabled:
            logger.warning("Auto-rollback triggered by A/B test")
            self.rollback()
    
    def check_health(self) -> Dict[str, Any]:
        """Check overall MLOps health."""
        health = {
            'status': 'healthy',
            'deployment_active': self._deployment_active,
            'issues': []
        }
        
        # Check PnL divergence
        if abs(self._pnl_baseline) > 1e-6:
            divergence = abs(self._pnl_canary - self._pnl_baseline) / abs(self._pnl_baseline)
            if divergence > self._pnl_divergence_threshold:
                health['status'] = 'warning'
                health['issues'].append(f'High PnL divergence: {divergence:.2%}')
                
                if self._on_alert:
                    self._on_alert('pnl_divergence', divergence)
        
        # Check A/B test status
        if self.ab_tester and self._deployment_active:
            ab_status = self.ab_tester.get_results()
            if ab_status.get('divergence_alerts', 0) > 0:
                health['status'] = 'warning'
                health['issues'].append(f"AB test divergence alerts: {ab_status['divergence_alerts']}")
        
        # Check canary deployment status
        if self.canary_deployer:
            deploy_status = self.canary_deployer.get_deployment_status()
            if deploy_status.get('health_checks_failed', 0) > 0:
                health['status'] = 'warning'
                health['issues'].append(f"Health checks failed: {deploy_status['health_checks_failed']}")
        
        return health
    
    def promote(self) -> bool:
        """Manually promote canary to production."""
        if self.canary_deployer is None:
            return False
        
        success = self.canary_deployer.promote()
        
        if success:
            self._successful_promotions += 1
            self._deployment_active = False
            self._pnl_baseline = self._pnl_canary
            self._pnl_canary = 0.0
            
            if self._on_promote:
                self._on_promote()
            
            logger.info("Model promoted successfully")
        
        return success
    
    def rollback(self) -> bool:
        """Manually rollback canary deployment."""
        if self.canary_deployer is None:
            return False
        
        success = self.canary_deployer.rollback()
        
        if success:
            self._rollbacks += 1
            self._deployment_active = False
            self._pnl_canary = 0.0
            
            if self._on_rollback:
                self._on_rollback()
            
            logger.info("Model rolled back successfully")
        
        return success
    
    def _handle_promotion(self, deployment_id: str) -> None:
        """Handle promotion event."""
        self._successful_promotions += 1
        self._deployment_active = False
        
        logger.info(f"Promotion completed: {deployment_id}")
        
        if self._on_promote:
            self._on_promote()
    
    def _handle_rollback(self, deployment_id: str) -> None:
        """Handle rollback event."""
        self._rollbacks += 1
        self._deployment_active = False
        
        logger.warning(f"Rollback completed: {deployment_id}")
        
        if self._on_rollback:
            self._on_rollback()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get MLOps statistics."""
        stats = {
            'total_deployments': self._total_deployments,
            'successful_promotions': self._successful_promotions,
            'rollbacks': self._rollbacks,
            'promotion_rate': self._successful_promotions / max(1, self._total_deployments),
            'deployment_active': self._deployment_active,
            'pnl_baseline': self._pnl_baseline,
            'pnl_canary': self._pnl_canary,
            'auto_promote_enabled': self._auto_promote_enabled,
            'auto_rollback_enabled': self._auto_rollback_enabled
        }
        
        if self.canary_deployer:
            stats['canary_stats'] = self.canary_deployer.get_deployment_status()
        
        if self.ab_tester:
            stats['ab_test_stats'] = self.ab_tester.get_results()
        
        return stats
    
    def start_monitoring(self, interval_seconds: float = 60.0) -> None:
        """Start background monitoring thread."""
        if self._monitor_running:
            return
        
        self._monitor_running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self._monitor_thread.start()
        logger.info(f"MLOps monitoring started (interval: {interval_seconds}s)")
    
    def _monitor_loop(self, interval: float) -> None:
        """Background monitoring loop."""
        while self._monitor_running:
            try:
                health = self.check_health()
                if health['status'] != 'healthy':
                    logger.warning(f"MLOps health issue: {health['issues']}")
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
            
            time.sleep(interval)
    
    def stop_monitoring(self) -> None:
        """Stop background monitoring."""
        self._monitor_running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        logger.info("MLOps monitoring stopped")
    
    def set_callbacks(self, on_promote: Optional[Callable] = None,
                      on_rollback: Optional[Callable] = None,
                      on_alert: Optional[Callable] = None) -> None:
        """Set event callbacks."""
        self._on_promote = on_promote
        self._on_rollback = on_rollback
        self._on_alert = on_alert
    
    def close(self) -> None:
        """Clean up MLOps resources."""
        self.stop_monitoring()
        
        if self.canary_deployer:
            self.canary_deployer.close()
        
        if self.ab_tester:
            self.ab_tester.reset()
        
        logger.info("MLOpsManager closed")


# Singleton instance
_mlops_manager: Optional[MLOpsManager] = None


def get_mlops_manager(config: Optional[Dict[str, Any]] = None) -> MLOpsManager:
    """Get or create singleton MLOpsManager instance."""
    global _mlops_manager
    if _mlops_manager is None:
        _mlops_manager = MLOpsManager(config)
    return _mlops_manager


def reset_mlops_manager() -> None:
    """Reset singleton instance."""
    global _mlops_manager
    if _mlops_manager is not None:
        _mlops_manager.close()
    _mlops_manager = None


__all__ = [
    'MLOpsManager',
    'get_mlops_manager',
    'reset_mlops_manager'
]
