"""
MLOps Module Root
Integrates hot-swapper with SOUL.md feedback loop to trigger retraining on concept drift.
Central hub for model lifecycle management.
"""

import threading
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
import logging
import json
from pathlib import Path

# Import submodules
from .registry import (
    ModelRegistry, 
    ModelType, 
    ModelStatus,
    ModelMetadata,
    get_model_registry,
    reset_model_registry
)
from .hot_swap import (
    ModelHotSwapper,
    HotSwapConfig,
    ModelFileInfo,
    get_model_hot_swapper,
    reset_model_hot_swapper
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DriftMetrics:
    """Metrics for detecting concept drift."""
    model_id: str
    timestamp: float = field(default_factory=time.time)
    prediction_drift: float = 0.0
    feature_drift: float = 0.0
    accuracy_decay: float = 0.0
    sharpe_decay: float = 0.0
    trigger_retrain: bool = False
    drift_threshold: float = 0.15


@dataclass
class SOULFeedback:
    """SOUL.md feedback loop state."""
    iteration: int = 0
    last_retrain_time: float = 0.0
    models_trained: int = 0
    models_deployed: int = 0
    drift_events: List[DriftMetrics] = field(default_factory=list)
    performance_history: Dict[str, List[float]] = field(default_factory=dict)


@dataclass
class MLOpsConfig:
    """Configuration for MLOps module."""
    model_watch_dirs: List[str] = field(default_factory=lambda: ["/tmp/models"])
    drift_check_interval_seconds: float = 60.0
    drift_threshold: float = 0.15
    auto_retrain: bool = True
    retrain_callback: Optional[Callable[[str], bool]] = None
    enable_hot_swap: bool = True


class MLOpsModule:
    """
    Central MLOps module integrating registry, hot-swap, and SOUL feedback.
    Manages complete model lifecycle from training to deployment.
    """
    
    def __init__(self, config: Optional[MLOpsConfig] = None):
        self.config = config or MLOpsConfig()
        
        # Submodules
        self._registry = get_model_registry()
        self._hot_swapper: Optional[ModelHotSwapper] = None
        
        # SOUL feedback state
        self._soul_feedback = SOULFeedback()
        
        # Drift tracking
        self._drift_metrics: Dict[str, DriftMetrics] = {}
        self._last_drift_check = 0.0
        
        # Retraining state
        self._retraining_queue: List[str] = []
        self._is_retraining = False
        
        # Thread management
        self._running = False
        self._drift_checker_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        
        # Statistics
        self._total_drift_checks = 0
        self._total_retrains_triggered = 0
    
    def initialize(self) -> bool:
        """Initialize the MLOps module."""
        try:
            # Initialize hot swapper
            if self.config.enable_hot_swap:
                self._hot_swapper = get_model_hot_swapper(
                    callback=self._on_model_detected,
                    config=HotSwapConfig(
                        watch_directories=self.config.model_watch_dirs
                    )
                )
                
                if not self._hot_swapper.start():
                    logger.warning("Failed to start hot swapper")
            
            self._running = True
            
            # Start drift checker
            self._drift_checker_thread = threading.Thread(
                target=self._drift_check_loop,
                daemon=True,
                name="MLOps_DriftChecker"
            )
            self._drift_checker_thread.start()
            
            logger.info("MLOps Module initialized")
            return True
        
        except Exception as e:
            logger.error(f"Failed to initialize MLOps: {e}")
            return False
    
    def _on_model_detected(self, info: ModelFileInfo) -> bool:
        """Handle detected model file from hot swapper."""
        logger.info(f"New model detected: {info.model_id}")
        
        try:
            # Load model based on type
            model_instance = self._load_model(info)
            
            if model_instance is None:
                return False
            
            # Determine model type
            model_type = self._infer_model_type(info.model_type)
            
            # Register in registry
            success = self._registry.register_model(
                model_id=info.model_id,
                model_instance=model_instance,
                model_type=model_type,
                version="auto",
                file_path=info.file_path
            )
            
            if success:
                # Set as active
                self._registry.set_active(info.model_id, model_type)
                
                # Update SOUL feedback
                self._soul_feedback.models_deployed += 1
                
                logger.info(f"Deployed model: {info.model_id}")
                return True
            
            return False
        
        except Exception as e:
            logger.error(f"Error deploying model {info.model_id}: {e}")
            return False
    
    def _load_model(self, info: ModelFileInfo) -> Optional[Any]:
        """Load model from file."""
        try:
            ext = info.model_type.lower()
            
            if ext == "onnx":
                from .onnx_inference import ONNXInferenceSession, ONNXConfig
                session = ONNXInferenceSession(
                    info.file_path,
                    ONNXConfig(n_threads=4)
                )
                return session
            
            elif ext in ["xgb", "joblib", "pkl"]:
                import joblib
                model = joblib.load(info.file_path)
                return model
            
            else:
                logger.warning(f"Unknown model type: {ext}")
                return None
        
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None
    
    def _infer_model_type(self, file_ext: str) -> ModelType:
        """Infer ModelType from file extension."""
        ext_map = {
            "xgb": ModelType.XGB,
            "lgb": ModelType.LGB,
            "onnx": ModelType.ONNX_TRANSFORMER,
            "joblib": ModelType.CUSTOM,
            "pkl": ModelType.CUSTOM,
        }
        return ext_map.get(file_ext, ModelType.CUSTOM)
    
    def _drift_check_loop(self) -> None:
        """Background loop for drift detection."""
        while self._running:
            try:
                self._check_drift()
            except Exception as e:
                logger.error(f"Drift check error: {e}")
            
            self._lock.acquire()
            wait_time = self.config.drift_check_interval_seconds
            self._lock.release()
            
            # Use event wait for clean shutdown
            stop_event = threading.Event()
            stop_event.wait(wait_time)
            if not self._running:
                break
    
    def _check_drift(self) -> None:
        """Check for concept drift across all active models."""
        now = time.time()
        
        with self._lock:
            if now - self._last_drift_check < self.config.drift_check_interval_seconds:
                return
            
            self._last_drift_check = now
            self._total_drift_checks += 1
        
        # Get all active models
        models = self._registry.list_models(status=ModelStatus.ACTIVE)
        
        for model_info in models:
            model_id = model_info["model_id"]
            
            # Simulate drift calculation (in practice, would use real metrics)
            drift = self._calculate_drift(model_id)
            
            if drift.trigger_retrain:
                logger.warning(f"Drift detected for {model_id}: {drift.prediction_drift:.4f}")
                
                self._soul_feedback.drift_events.append(drift)
                
                if self.config.auto_retrain:
                    self._queue_retrain(model_id)
    
    def _calculate_drift(self, model_id: str) -> DriftMetrics:
        """Calculate drift metrics for a model."""
        # In production, this would compare recent predictions vs actuals
        # For now, return placeholder metrics
        
        metrics = DriftMetrics(
            model_id=model_id,
            prediction_drift=0.0,
            feature_drift=0.0,
            accuracy_decay=0.0,
            sharpe_decay=0.0,
            drift_threshold=self.config.drift_threshold
        )
        
        # Check stored metrics
        if model_id in self._drift_metrics:
            stored = self._drift_metrics[model_id]
            metrics.prediction_drift = stored.prediction_drift
            metrics.feature_drift = stored.feature_drift
        
        # Determine if retrain needed
        metrics.trigger_retrain = (
            metrics.prediction_drift > self.config.drift_threshold or
            metrics.feature_drift > self.config.drift_threshold
        )
        
        return metrics
    
    def _queue_retrain(self, model_id: str) -> None:
        """Queue a model for retraining."""
        with self._lock:
            if model_id not in self._retraining_queue:
                self._retraining_queue.append(model_id)
                self._total_retrains_triggered += 1
        
        logger.info(f"Queued retrain for: {model_id}")
        
        # Trigger retrain callback if set
        if self.config.retrain_callback:
            try:
                self.config.retrain_callback(model_id)
            except Exception as e:
                logger.error(f"Retrain callback error: {e}")
    
    def update_drift_metrics(self, 
                             model_id: str,
                             prediction_drift: float,
                             feature_drift: float,
                             accuracy_decay: float = 0.0,
                             sharpe_decay: float = 0.0) -> None:
        """Update drift metrics for a model."""
        with self._lock:
            self._drift_metrics[model_id] = DriftMetrics(
                model_id=model_id,
                prediction_drift=prediction_drift,
                feature_drift=feature_drift,
                accuracy_decay=accuracy_decay,
                sharpe_decay=sharpe_decay,
                drift_threshold=self.config.drift_threshold,
                trigger_retrain=(
                    prediction_drift > self.config.drift_threshold or
                    feature_drift > self.config.drift_threshold
                )
            )
    
    def register_model(self,
                       model_id: str,
                       model_instance: Any,
                       model_type: ModelType,
                       version: str,
                       **kwargs) -> bool:
        """Register a model directly."""
        return self._registry.register_model(
            model_id=model_id,
            model_instance=model_instance,
            model_type=model_type,
            version=version,
            **kwargs
        )
    
    def get_active_model(self, model_type: ModelType) -> Optional[Any]:
        """Get active model for a type."""
        return self._registry.get_active_model(model_type)
    
    def swap_model_for_regime(self, 
                               model_type: ModelType,
                               regime: str) -> Optional[str]:
        """Swap model based on market regime."""
        return self._registry.swap_for_regime(model_type, regime)
    
    def get_soul_feedback(self) -> SOULFeedback:
        """Get current SOUL feedback state."""
        return self._soul_feedback
    
    def shutdown(self) -> None:
        """Shutdown the module."""
        self._running = False
        
        if self._hot_swapper is not None:
            self._hot_swapper.stop()
        
        if self._drift_checker_thread is not None:
            self._drift_checker_thread.join(timeout=2.0)
        
        logger.info("MLOps Module shutdown")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get module statistics."""
        with self._lock:
            return {
                "registry_stats": self._registry.get_statistics(),
                "hot_swapper_stats": (
                    self._hot_swapper.get_statistics() 
                    if self._hot_swapper else {}
                ),
                "soul_feedback": {
                    "iteration": self._soul_feedback.iteration,
                    "models_trained": self._soul_feedback.models_trained,
                    "models_deployed": self._soul_feedback.models_deployed,
                    "drift_events_count": len(self._soul_feedback.drift_events),
                },
                "total_drift_checks": self._total_drift_checks,
                "total_retrains_triggered": self._total_retrains_triggered,
                "retraining_queue": list(self._retraining_queue),
                "is_running": self._running,
            }


# Global module instance
_mlops_module: Optional[MLOpsModule] = None
_module_lock = threading.Lock()


def get_mlops_module(config: Optional[MLOpsConfig] = None) -> MLOpsModule:
    """Get global MLOpsModule instance."""
    global _mlops_module
    
    with _module_lock:
        if _mlops_module is None:
            _mlops_module = MLOpsModule(config)
        
        return _mlops_module


def reset_mlops_module() -> None:
    """Reset the global module."""
    global _mlops_module
    
    with _module_lock:
        if _mlops_module is not None:
            _mlops_module.shutdown()
            _mlops_module = None
        
        reset_model_registry()
        reset_model_hot_swapper()


if __name__ == "__main__":
    print("MLOps Module Demo")
    print("=" * 40)
    
    # Create test directory
    test_dir = "/tmp/mlops_models"
    Path(test_dir).mkdir(exist_ok=True)
    
    # Initialize module
    config = MLOpsConfig(
        model_watch_dirs=[test_dir],
        drift_check_interval_seconds=30.0,
        drift_threshold=0.15,
        auto_retrain=False
    )
    
    module = get_mlops_module(config)
    
    if not module.initialize():
        print("Failed to initialize MLOps")
        exit(1)
    
    # Register a mock model
    class MockModel:
        pass
    
    module.register_model(
        model_id="demo_xgb_v1",
        model_instance=MockModel(),
        model_type=ModelType.XGB,
        version="1.0.0",
        tags=["demo"]
    )
    
    # Simulate drift
    module.update_drift_metrics(
        model_id="demo_xgb_v1",
        prediction_drift=0.05,
        feature_drift=0.08
    )
    
    # Get statistics
    stats = module.get_statistics()
    print(f"\nStatistics:")
    print(f"  Registry: {stats['registry_stats']['total_models']} models")
    print(f"  Drift checks: {stats['total_drift_checks']}")
    print(f"  Retrains triggered: {stats['total_retrains_triggered']}")
    
    # Get SOUL feedback
    soul = module.get_soul_feedback()
    print(f"\nSOUL Feedback:")
    print(f"  Iteration: {soul.iteration}")
    print(f"  Models deployed: {soul.models_deployed}")
    print(f"  Drift events: {len(soul.drift_events)}")
    
    # Shutdown
    module.shutdown()
    reset_mlops_module()
    
    print("\nMLOps Module demo complete")
