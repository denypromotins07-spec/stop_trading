"""
MLOps Pipeline Module Root.
Manages the end-to-end ML model lifecycle ensuring zero-downtime hot-swapping
of ML weights.

Provides:
- Unified interface for CI/CD pipeline and promotion gate
- Automated drift detection integration
- Model registry management
- Health monitoring and alerting
"""

import numpy as np
from typing import Dict, Any, Optional, List, Callable
import threading
import logging
import time
import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict

from ..ci_cd.ci_cd_pipeline import CICDPipeline, get_cicd_pipeline, PipelineRun
from ..ci_cd.model_promotion import ModelPromotionGate, get_promotion_gate, PromotionResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MLOpsStatus:
    """Current status of the MLOps pipeline."""
    production_model_id: Optional[str]
    shadow_model_id: Optional[str]
    pipeline_active: bool
    current_pipeline_stage: Optional[str]
    drift_metrics: Dict[str, float]
    last_promotion_at: Optional[float]
    system_health: str  # 'healthy', 'degraded', 'critical'


@dataclass
class DriftAlert:
    """Alert triggered by drift detection."""
    alert_id: str
    timestamp: float
    metric_name: str
    current_value: float
    threshold: float
    severity: str  # 'warning', 'critical'
    action_taken: str


class MLOpsPipelineManager:
    """
    Central manager for MLOps pipeline operations.
    Coordinates CI/CD pipeline, model promotion, and drift detection.
    """
    
    def __init__(
        self,
        workspace_dir: str = "./mlops_workspace",
        registry_dir: str = "./model_registry",
        drift_check_interval_seconds: float = 60.0
    ):
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        
        self.drift_check_interval = drift_check_interval_seconds
        
        # Initialize subsystems
        self.cicd_pipeline = get_cicd_pipeline(str(self.workspace_dir / "pipeline"))
        self.promotion_gate = get_promotion_gate(str(self.registry_dir))
        
        self._lock = threading.RLock()
        
        # Drift tracking
        self._drift_thresholds: Dict[str, float] = {
            'psi': 0.2,
            'js_divergence': 0.15,
            'feature_drift': 0.1,
            'prediction_drift': 0.15
        }
        
        self._current_drift_metrics: Dict[str, float] = {}
        self._drift_history: List[Dict[str, Any]] = []
        self._drift_alerts: List[DriftAlert] = []
        
        # Monitoring state
        self._monitoring_active = False
        self._monitor_thread: Optional[threading.Thread] = None
        
        # Callbacks
        self._on_drift_alert_callbacks: List[Callable[[DriftAlert], None]] = []
        self._on_promotion_callbacks: List[Callable[[PromotionResult], None]] = []
        
        # Performance metrics
        self._last_promotion_at: Optional[float] = None
        self._total_promotions = 0
        self._failed_promotions = 0
    
    def register_drift_callback(self, callback: Callable[[DriftAlert], None]) -> None:
        """Register callback for drift alerts."""
        self._on_drift_alert_callbacks.append(callback)
    
    def register_promotion_callback(self, callback: Callable[[PromotionResult], None]) -> None:
        """Register callback for promotion events."""
        self._on_promotion_callbacks.append(callback)
    
    def update_drift_metrics(self, metrics: Dict[str, float]) -> Optional[DriftAlert]:
        """
        Update drift metrics and check for threshold violations.
        
        Args:
            metrics: Current drift metric values
            
        Returns:
            DriftAlert if threshold exceeded, None otherwise
        """
        with self._lock:
            self._current_drift_metrics.update(metrics)
            self._drift_history.append({
                'timestamp': time.time(),
                'metrics': metrics.copy()
            })
            
            # Keep history bounded
            if len(self._drift_history) > 1000:
                self._drift_history.pop(0)
            
            # Check thresholds
            alert = None
            for metric_name, value in metrics.items():
                threshold_key = metric_name.lower().replace(' ', '_')
                if threshold_key in self._drift_thresholds:
                    threshold = self._drift_thresholds[threshold_key]
                    
                    if value > threshold:
                        severity = 'critical' if value > threshold * 1.5 else 'warning'
                        
                        import uuid
                        alert = DriftAlert(
                            alert_id=f"alert_{uuid.uuid4().hex[:8]}",
                            timestamp=time.time(),
                            metric_name=metric_name,
                            current_value=value,
                            threshold=threshold,
                            severity=severity,
                            action_taken='pipeline_triggered' if severity == 'critical' else 'monitoring'
                        )
                        
                        self._drift_alerts.append(alert)
                        
                        # Notify callbacks
                        for callback in self._on_drift_alert_callbacks:
                            try:
                                callback(alert)
                            except Exception as e:
                                logger.error(f"Drift callback error: {e}")
                        
                        # Trigger pipeline for critical alerts
                        if severity == 'critical':
                            self.trigger_retraining(f"drift_{metric_name}", metrics)
                        
                        logger.warning(f"Drift alert: {metric_name}={value:.4f} > {threshold}")
                        break  # Only one alert per update
            
            return alert
    
    def trigger_retraining(
        self,
        reason: str,
        drift_metrics: Optional[Dict[str, float]] = None
    ) -> Optional[str]:
        """
        Trigger model retraining pipeline.
        
        Args:
            reason: Reason for retraining
            drift_metrics: Current drift metrics
            
        Returns:
            Pipeline run ID or None if pipeline unavailable
        """
        metrics = drift_metrics or self._current_drift_metrics.copy()
        
        try:
            run_id = self.cicd_pipeline.trigger_pipeline(
                trigger_reason=reason,
                drift_metrics=metrics
            )
            logger.info(f"Triggered retraining pipeline: {run_id} ({reason})")
            return run_id
        except Exception as e:
            logger.error(f"Failed to trigger retraining: {e}")
            return None
    
    def register_candidate_model(
        self,
        model_path: str,
        training_metrics: Dict[str, float]
    ) -> str:
        """
        Register a trained model candidate for promotion evaluation.
        
        Args:
            model_path: Path to model file
            training_metrics: Metrics from training
            
        Returns:
            Model ID
        """
        model_id = self.promotion_gate.register_candidate(model_path, training_metrics)
        logger.info(f"Registered candidate model: {model_id}")
        return model_id
    
    def evaluate_and_promote(self) -> PromotionResult:
        """
        Evaluate shadow model and promote if it passes gates.
        
        Returns:
            Promotion result
        """
        result = self.promotion_gate.promote()
        
        if result.success:
            self._last_promotion_at = time.time()
            self._total_promotions += 1
            
            # Notify callbacks
            for callback in self._on_promotion_callbacks:
                try:
                    callback(result)
                except Exception as e:
                    logger.error(f"Promotion callback error: {e}")
            
            logger.info(f"Model promoted: {result.new_model_id} (Sharpe +{result.sharpe_improvement:.4f})")
        else:
            self._failed_promotions += 1
            logger.warning(f"Model promotion failed: {result.reason}")
        
        return result
    
    def start_monitoring(self) -> None:
        """Start continuous drift monitoring."""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        
        def monitor_loop():
            while self._monitoring_active:
                try:
                    # In production, this would collect real drift metrics
                    # For now, just check existing metrics
                    self._check_system_health()
                except Exception as e:
                    logger.error(f"Monitoring error: {e}")
                
                time.sleep(self.drift_check_interval)
        
        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("MLOps monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop drift monitoring."""
        self._monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("MLOps monitoring stopped")
    
    def _check_system_health(self) -> None:
        """Check overall system health."""
        # Placeholder for health checks
        pass
    
    def get_status(self) -> MLOpsStatus:
        """Get current MLOps pipeline status."""
        with self._lock:
            prod_info = self.promotion_gate.get_production_info()
            shadow_info = self.promotion_gate.get_shadow_info()
            
            # Get pipeline status
            pipeline_active = False
            current_stage = None
            
            # Determine system health
            system_health = 'healthy'
            if len(self._drift_alerts) > 0:
                recent_alerts = [a for a in self._drift_alerts if time.time() - a.timestamp < 3600]
                if any(a.severity == 'critical' for a in recent_alerts):
                    system_health = 'critical'
                elif len(recent_alerts) > 3:
                    system_health = 'degraded'
            
            return MLOpsStatus(
                production_model_id=prod_info['model_id'] if prod_info else None,
                shadow_model_id=shadow_info['model_id'] if shadow_info else None,
                pipeline_active=pipeline_active,
                current_stage=current_stage,
                drift_metrics=self._current_drift_metrics.copy(),
                last_promotion_at=self._last_promotion_at,
                system_health=system_health
            )
    
    def get_drift_history(
        self,
        hours: int = 24
    ) -> List[Dict[str, Any]]:
        """Get drift metric history."""
        with self._lock:
            cutoff = time.time() - (hours * 3600)
            return [h for h in self._drift_history if h['timestamp'] >= cutoff]
    
    def get_alerts(
        self,
        hours: int = 24,
        severity: Optional[str] = None
    ) -> List[DriftAlert]:
        """Get drift alerts."""
        with self._lock:
            cutoff = time.time() - (hours * 3600)
            alerts = [a for a in self._drift_alerts if a.timestamp >= cutoff]
            
            if severity:
                alerts = [a for a in alerts if a.severity == severity]
            
            return alerts
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get MLOps pipeline statistics."""
        with self._lock:
            return {
                'total_promotions': self._total_promotions,
                'failed_promotions': self._failed_promotions,
                'promotion_success_rate': self._total_promotions / max(self._total_promotions + self._failed_promotions, 1),
                'last_promotion_at': datetime.fromtimestamp(self._last_promotion_at).isoformat() if self._last_promotion_at else None,
                'alerts_last_24h': len(self.get_alerts(hours=24)),
                'critical_alerts_last_24h': len(self.get_alerts(hours=24, severity='critical')),
                'drift_metrics_current': self._current_drift_metrics.copy(),
                'monitoring_active': self._monitoring_active
            }
    
    def shutdown(self) -> None:
        """Shutdown MLOps pipeline manager."""
        self.stop_monitoring()
        logger.info("MLOps Pipeline Manager shut down")


# Global singleton instance
_mlops_instance: Optional[MLOpsPipelineManager] = None
_mlops_lock = threading.Lock()


def get_mlops_manager(
    workspace_dir: str = "./mlops_workspace",
    registry_dir: str = "./model_registry"
) -> MLOpsPipelineManager:
    """Thread-safe singleton access to MLOps pipeline manager."""
    global _mlops_instance
    
    with _mlops_lock:
        if _mlops_instance is None:
            _mlops_instance = MLOpsPipelineManager(workspace_dir, registry_dir)
        
        return _mlops_instance


if __name__ == "__main__":
    # Demo usage
    manager = get_mlops_manager()
    
    print("=== MLOps Pipeline Manager Demo ===\n")
    
    # Register initial production model
    manager.register_candidate_model(
        model_path="./models/prod_v1.onnx",
        training_metrics={
            'sharpe': 1.5,
            'max_drawdown': 0.08
        }
    )
    manager.evaluate_and_promote()
    
    # Simulate drift metrics updates
    print("Simulating drift metrics...")
    for i in range(5):
        drift = {
            'PSI': 0.1 + i * 0.05,
            'JS_Divergence': 0.08 + i * 0.02
        }
        alert = manager.update_drift_metrics(drift)
        if alert:
            print(f"  Alert: {alert.metric_name} = {alert.current_value:.4f} ({alert.severity})")
    
    # Show status
    status = manager.get_status()
    print(f"\nSystem Status:")
    print(f"  Production Model: {status.production_model_id}")
    print(f"  System Health: {status.system_health}")
    print(f"  Drift Metrics: {status.drift_metrics}")
    
    # Show statistics
    stats = manager.get_statistics()
    print(f"\nStatistics:")
    print(f"  Total Promotions: {stats['total_promotions']}")
    print(f"  Alerts (24h): {stats['alerts_last_24h']}")
    
    manager.shutdown()
