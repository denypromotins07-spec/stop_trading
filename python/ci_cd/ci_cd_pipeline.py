"""
CI/CD Pipeline for ML Model Lifecycle.
Automated pipeline triggered when SOUL.md drift metrics cross critical threshold.
Spins up Ray training cluster, retrains models, evaluates against shadow scoring.

Provides zero-downtime model updates with automated rollback capability.
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
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PipelineStage:
    """Represents a stage in the CI/CD pipeline."""
    name: str
    status: str  # 'pending', 'running', 'completed', 'failed', 'skipped'
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class PipelineRun:
    """Complete record of a pipeline execution."""
    run_id: str
    trigger_reason: str
    drift_threshold_exceeded: bool
    drift_metrics: Dict[str, float]
    stages: List[PipelineStage]
    status: str  # 'running', 'completed', 'failed', 'cancelled'
    created_at: float
    completed_at: Optional[float] = None
    model_artifact_path: Optional[str] = None


class CICDPipeline:
    """
    CI/CD Pipeline for automated ML model retraining and deployment.
    
    Stages:
    1. Drift Detection Validation
    2. Data Preparation
    3. Ray Cluster Spin-up
    4. Model Training
    5. Shadow Evaluation
    6. Performance Gate
    7. Model Promotion
    8. Deployment & Rollback Setup
    """
    
    def __init__(
        self,
        workspace_dir: str = "./pipeline_workspace",
        max_parallel_runs: int = 2
    ):
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_parallel_runs = max_parallel_runs
        self._lock = threading.RLock()
        
        self._active_runs: Dict[str, PipelineRun] = {}
        self._completed_runs: List[PipelineRun] = []
        self._run_counter = 0
        
        # Pipeline configuration
        self.stages_config = [
            "drift_validation",
            "data_preparation",
            "cluster_spinup",
            "model_training",
            "shadow_evaluation",
            "performance_gate",
            "model_promotion",
            "deployment"
        ]
        
        # Callbacks for each stage (set by user)
        self._stage_callbacks: Dict[str, Callable] = {}
        
        # Drift threshold configuration
        self.drift_thresholds = {
            'psi': 0.2,
            'js_divergence': 0.15,
            'feature_drift': 0.1
        }
    
    def register_stage_callback(self, stage_name: str, callback: Callable) -> None:
        """Register a callback function for a pipeline stage."""
        self._stage_callbacks[stage_name] = callback
    
    def trigger_pipeline(
        self,
        trigger_reason: str,
        drift_metrics: Dict[str, float],
        priority: int = 3
    ) -> str:
        """
        Trigger a new pipeline run.
        
        Args:
            trigger_reason: Reason for triggering (e.g., "drift_threshold_exceeded")
            drift_metrics: Current drift metric values
            priority: Run priority (1=highest)
            
        Returns:
            Run ID
        """
        import uuid
        
        with self._lock:
            # Check parallel run limit
            active_count = len([r for r in self._active_runs.values() if r.status == 'running'])
            if active_count >= self.max_parallel_runs:
                logger.warning(f"Pipeline at capacity ({active_count}/{self.max_parallel_runs} runs)")
                raise RuntimeError("Pipeline at maximum capacity")
            
            run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
            
            # Check if drift threshold exceeded
            drift_exceeded = self._check_drift_thresholds(drift_metrics)
            
            # Create pipeline run
            stages = [
                PipelineStage(name=name, status='pending')
                for name in self.stages_config
            ]
            
            run = PipelineRun(
                run_id=run_id,
                trigger_reason=trigger_reason,
                drift_threshold_exceeded=drift_exceeded,
                drift_metrics=drift_metrics.copy(),
                stages=stages,
                status='pending',
                created_at=time.time()
            )
            
            self._active_runs[run_id] = run
            self._run_counter += 1
            
            logger.info(f"Triggered pipeline run {run_id}: {trigger_reason}")
            
            # Start pipeline execution in background
            thread = threading.Thread(
                target=self._execute_pipeline,
                args=(run_id,),
                daemon=True
            )
            thread.start()
            
            return run_id
    
    def _check_drift_thresholds(self, drift_metrics: Dict[str, float]) -> bool:
        """Check if any drift metric exceeds threshold."""
        for metric_name, value in drift_metrics.items():
            threshold_key = metric_name.lower().replace(' ', '_')
            if threshold_key in self.drift_thresholds:
                if value > self.drift_thresholds[threshold_key]:
                    logger.warning(f"Drift metric {metric_name}={value:.4f} exceeds threshold {self.drift_thresholds[threshold_key]}")
                    return True
        return False
    
    def _execute_pipeline(self, run_id: str) -> None:
        """Execute the full pipeline."""
        run = self._active_runs.get(run_id)
        if not run:
            return
        
        try:
            run.status = 'running'
            logger.info(f"Starting pipeline run {run_id}")
            
            for i, stage in enumerate(run.stages):
                if run.status != 'running':
                    # Pipeline was cancelled
                    stage.status = 'skipped'
                    continue
                
                logger.info(f"[{run_id}] Starting stage: {stage.name}")
                stage.status = 'running'
                stage.start_time = time.time()
                
                try:
                    # Execute stage
                    result = self._execute_stage(run_id, stage.name, run.drift_metrics)
                    
                    stage.status = 'completed'
                    stage.result = result
                    stage.end_time = time.time()
                    
                    logger.info(f"[{run_id}] Completed stage: {stage.name} ({stage.end_time - stage.start_time:.2f}s)")
                    
                    # Check for early termination conditions
                    if stage.name == 'performance_gate' and result.get('passed') is False:
                        logger.warning(f"[{run_id}] Performance gate failed, aborting pipeline")
                        run.status = 'failed'
                        break
                    
                except Exception as e:
                    stage.status = 'failed'
                    stage.error = str(e)
                    stage.end_time = time.time()
                    logger.error(f"[{run_id}] Stage {stage.name} failed: {e}")
                    run.status = 'failed'
                    break
            
            # Finalize
            if run.status == 'running':
                run.status = 'completed'
                logger.info(f"[{run_id}] Pipeline completed successfully")
            
            run.completed_at = time.time()
            
            # Move to completed runs
            with self._lock:
                self._completed_runs.append(run)
                del self._active_runs[run_id]
            
            # Save run results
            self._save_run_results(run)
            
        except Exception as e:
            logger.error(f"[{run_id}] Pipeline execution failed: {e}")
            run.status = 'failed'
            run.completed_at = time.time()
    
    def _execute_stage(
        self,
        run_id: str,
        stage_name: str,
        drift_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Execute a single pipeline stage."""
        # Check for registered callback
        if stage_name in self._stage_callbacks:
            return self._stage_callbacks[stage_name](run_id, drift_metrics)
        
        # Default stage implementations
        if stage_name == 'drift_validation':
            return self._stage_drift_validation(drift_metrics)
        elif stage_name == 'data_preparation':
            return self._stage_data_preparation()
        elif stage_name == 'cluster_spinup':
            return self._stage_cluster_spinup()
        elif stage_name == 'model_training':
            return self._stage_model_training(run_id)
        elif stage_name == 'shadow_evaluation':
            return self._stage_shadow_evaluation(run_id)
        elif stage_name == 'performance_gate':
            return self._stage_performance_gate(run_id)
        elif stage_name == 'model_promotion':
            return self._stage_model_promotion(run_id)
        elif stage_name == 'deployment':
            return self._stage_deployment(run_id)
        
        return {'status': 'unknown_stage'}
    
    def _stage_drift_validation(self, drift_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Validate drift metrics and determine retraining necessity."""
        validation_result = {
            'metrics_validated': list(drift_metrics.keys()),
            'thresholds_used': self.drift_thresholds.copy(),
            'exceeded_metrics': []
        }
        
        for metric_name, value in drift_metrics.items():
            threshold_key = metric_name.lower().replace(' ', '_')
            if threshold_key in self.drift_thresholds:
                if value > self.drift_thresholds[threshold_key]:
                    validation_result['exceeded_metrics'].append({
                        'name': metric_name,
                        'value': value,
                        'threshold': self.drift_thresholds[threshold_key]
                    })
        
        validation_result['retraining_required'] = len(validation_result['exceeded_metrics']) > 0
        
        return validation_result
    
    def _stage_data_preparation(self) -> Dict[str, Any]:
        """Prepare training data from feature store."""
        # Placeholder - would integrate with feature store
        return {
            'data_source': 'feature_store',
            'samples_collected': 0,
            'features_selected': [],
            'train_val_split': [0.8, 0.2]
        }
    
    def _stage_cluster_spinup(self) -> Dict[str, Any]:
        """Spin up Ray training cluster."""
        # Placeholder - would initialize Ray cluster
        return {
            'cluster_size': 4,
            'cpu_per_worker': 2,
            'memory_per_worker_gb': 4,
            'startup_time_seconds': 30
        }
    
    def _stage_model_training(self, run_id: str) -> Dict[str, Any]:
        """Execute model training."""
        # Placeholder - would run distributed training
        return {
            'model_type': 'ensemble',
            'training_samples': 0,
            'epochs_completed': 10,
            'final_loss': 0.0
        }
    
    def _stage_shadow_evaluation(self, run_id: str) -> Dict[str, Any]:
        """Evaluate model in shadow mode."""
        # Placeholder - would run shadow evaluation
        return {
            'shadow_sharpe': 0.0,
            'shadow_drawdown': 0.0,
            'prediction_accuracy': 0.0,
            'inference_latency_us': 0
        }
    
    def _stage_performance_gate(self, run_id: str) -> Dict[str, Any]:
        """Check if model passes performance gates."""
        # Placeholder - would compare against production model
        return {
            'passed': True,
            'metrics_comparison': {},
            'statistical_significance': 0.95
        }
    
    def _stage_model_promotion(self, run_id: str) -> Dict[str, Any]:
        """Promote model to production registry."""
        # Placeholder - would update model registry
        artifact_path = str(self.workspace_dir / f"models/{run_id}.onnx")
        return {
            'artifact_path': artifact_path,
            'registry_updated': True,
            'version_tag': run_id
        }
    
    def _stage_deployment(self, run_id: str) -> Dict[str, Any]:
        """Deploy model with rollback capability."""
        # Placeholder - would deploy to production
        return {
            'deployed': True,
            'rollback_configured': True,
            'health_check_passed': True
        }
    
    def _save_run_results(self, run: PipelineRun) -> None:
        """Save pipeline run results to disk."""
        results_file = self.workspace_dir / f"runs/{run.run_id}.json"
        results_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(results_file, 'w') as f:
                json.dump({
                    'run_id': run.run_id,
                    'trigger_reason': run.trigger_reason,
                    'drift_threshold_exceeded': run.drift_threshold_exceeded,
                    'drift_metrics': run.drift_metrics,
                    'status': run.status,
                    'created_at': datetime.fromtimestamp(run.created_at).isoformat(),
                    'completed_at': datetime.fromtimestamp(run.completed_at).isoformat() if run.completed_at else None,
                    'stages': [asdict(s) for s in run.stages],
                    'model_artifact_path': run.model_artifact_path
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save run results: {e}")
    
    def get_run_status(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a pipeline run."""
        with self._lock:
            if run_id in self._active_runs:
                run = self._active_runs[run_id]
                return {
                    'run_id': run.run_id,
                    'status': run.status,
                    'current_stage': next((s.name for s in run.stages if s.status == 'running'), None),
                    'progress': len([s for s in run.stages if s.status == 'completed']) / len(run.stages)
                }
            
            for run in self._completed_runs:
                if run.run_id == run_id:
                    return {
                        'run_id': run.run_id,
                        'status': run.status,
                        'completed_at': run.completed_at,
                        'result': run.model_artifact_path
                    }
        
        return None
    
    def cancel_run(self, run_id: str) -> bool:
        """Cancel a running pipeline."""
        with self._lock:
            if run_id in self._active_runs:
                self._active_runs[run_id].status = 'cancelled'
                logger.info(f"Cancelled pipeline run {run_id}")
                return True
        return False
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get overall pipeline statistics."""
        with self._lock:
            completed = len(self._completed_runs)
            successful = len([r for r in self._completed_runs if r.status == 'completed'])
            failed = len([r for r in self._completed_runs if r.status == 'failed'])
            
            avg_duration = 0
            if completed > 0:
                durations = [
                    r.completed_at - r.created_at 
                    for r in self._completed_runs 
                    if r.completed_at
                ]
                avg_duration = np.mean(durations) if durations else 0
            
            return {
                'total_runs': self._run_counter,
                'active_runs': len(self._active_runs),
                'completed_runs': completed,
                'successful_runs': successful,
                'failed_runs': failed,
                'success_rate': successful / max(completed, 1),
                'avg_duration_seconds': avg_duration
            }


# Global singleton instance
_pipeline_instance: Optional[CICDPipeline] = None
_pipeline_lock = threading.Lock()


def get_cicd_pipeline(workspace_dir: str = "./pipeline_workspace") -> CICDPipeline:
    """Thread-safe singleton access to CI/CD pipeline."""
    global _pipeline_instance
    
    with _pipeline_lock:
        if _pipeline_instance is None:
            _pipeline_instance = CICDPipeline(workspace_dir)
        
        return _pipeline_instance


if __name__ == "__main__":
    # Demo usage
    pipeline = get_cicd_pipeline()
    
    print("=== CI/CD Pipeline Demo ===\n")
    
    # Simulate drift detection triggering pipeline
    drift_metrics = {
        'PSI': 0.25,  # Exceeds 0.2 threshold
        'JS_Divergence': 0.12,
        'Feature_Drift': 0.08
    }
    
    run_id = pipeline.trigger_pipeline(
        trigger_reason="drift_threshold_exceeded",
        drift_metrics=drift_metrics
    )
    
    print(f"Triggered pipeline run: {run_id}")
    
    # Wait for pipeline to complete (demo purposes)
    time.sleep(2)
    
    # Check status
    status = pipeline.get_run_status(run_id)
    print(f"Pipeline status: {status}")
    
    # Show stats
    stats = pipeline.get_pipeline_stats()
    print(f"\nPipeline Statistics: {stats}")
