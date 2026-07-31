"""
Thread-safe Model Registry managing versioning and active model pointers.
Allows multiple models to reside in memory, swapped instantly based on HMM regime state.
"""

import threading
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Supported model types."""
    XGB = "xgb"
    LGB = "lgb"
    ONNX_TRANSFORMER = "onnx_transformer"
    ONNX_RNN = "onnx_rnn"
    CUSTOM = "custom"


class ModelStatus(Enum):
    """Model lifecycle status."""
    LOADING = "loading"
    ACTIVE = "active"
    STANDBY = "standby"
    DEPRECATED = "deprecated"
    ERROR = "error"


@dataclass
class ModelMetadata:
    """Metadata for a registered model."""
    model_id: str
    model_type: ModelType
    version: str
    created_at: float = field(default_factory=time.time)
    file_path: str = ""
    file_hash: str = ""
    input_dim: int = 0
    output_dim: int = 0
    tags: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    regime_compatibility: List[str] = field(default_factory=list)  # e.g., ["trending", "mean_reverting"]


@dataclass
class RegisteredModel:
    """A registered model with its instance and metadata."""
    metadata: ModelMetadata
    instance: Any
    status: ModelStatus = ModelStatus.LOADING
    load_time_ms: float = 0.0
    last_used: float = 0.0
    use_count: int = 0


class ModelRegistry:
    """
    Thread-safe registry for ML models.
    Supports multiple models in memory with instant swapping.
    """
    
    def __init__(self):
        self._models: Dict[str, RegisteredModel] = {}
        self._active_model_ids: Dict[str, str] = {}  # model_type -> active_model_id
        self._lock = threading.RLock()
        
        # Statistics
        self._total_loads = 0
        self._total_swaps = 0
        self._total_errors = 0
    
    def register_model(self,
                       model_id: str,
                       model_instance: Any,
                       model_type: ModelType,
                       version: str,
                       file_path: str = "",
                       input_dim: int = 0,
                       output_dim: int = 0,
                       tags: Optional[List[str]] = None,
                       regime_compatibility: Optional[List[str]] = None) -> bool:
        """
        Register a new model.
        
        Args:
            model_id: Unique model identifier
            model_instance: Loaded model instance
            model_type: Type of model
            version: Model version string
            file_path: Path to model file
            input_dim: Input dimension
            output_dim: Output dimension
            tags: Optional tags
            regime_compatibility: Compatible market regimes
        
        Returns:
            True if successful
        """
        with self._lock:
            start_time = time.perf_counter()
            
            # Check for duplicate
            if model_id in self._models:
                logger.warning(f"Model {model_id} already registered, updating")
            
            # Calculate file hash
            file_hash = ""
            if file_path:
                try:
                    with open(file_path, 'rb') as f:
                        file_hash = hashlib.sha256(f.read()).hexdigest()[:16]
                except Exception:
                    pass
            
            # Create metadata
            metadata = ModelMetadata(
                model_id=model_id,
                model_type=model_type,
                version=version,
                file_path=file_path,
                file_hash=file_hash,
                input_dim=input_dim,
                output_dim=output_dim,
                tags=tags or [],
                regime_compatibility=regime_compatibility or []
            )
            
            # Create registered model
            load_time_ms = (time.perf_counter() - start_time) * 1000
            
            registered = RegisteredModel(
                metadata=metadata,
                instance=model_instance,
                status=ModelStatus.STANDBY,
                load_time_ms=load_time_ms
            )
            
            self._models[model_id] = registered
            self._total_loads += 1
            
            logger.info(f"Registered model: {model_id} ({model_type.value}) v{version}")
            return True
    
    def unregister_model(self, model_id: str) -> bool:
        """Unregister a model."""
        with self._lock:
            if model_id not in self._models:
                return False
            
            # Check if it's active
            for model_type, active_id in list(self._active_model_ids.items()):
                if active_id == model_id:
                    del self._active_model_ids[model_type]
            
            del self._models[model_id]
            logger.info(f"Unregistered model: {model_id}")
            return True
    
    def set_active(self, model_id: str, model_type: Optional[ModelType] = None) -> bool:
        """
        Set a model as active for its type.
        
        Args:
            model_id: Model to activate
            model_type: Optional specific model type
        
        Returns:
            True if successful
        """
        with self._lock:
            if model_id not in self._models:
                logger.error(f"Model {model_id} not found")
                return False
            
            registered = self._models[model_id]
            
            # Deactivate current active model of same type
            mt = model_type or registered.metadata.model_type
            
            old_active = self._active_model_ids.get(mt.value)
            if old_active and old_active in self._models:
                self._models[old_active].status = ModelStatus.STANDBY
            
            # Activate new model
            registered.status = ModelStatus.ACTIVE
            self._active_model_ids[mt.value] = model_id
            self._total_swaps += 1
            
            logger.info(f"Set active model: {model_id} (type: {mt.value})")
            return True
    
    def get_active_model(self, model_type: ModelType) -> Optional[Any]:
        """Get the active model instance for a type."""
        with self._lock:
            model_id = self._active_model_ids.get(model_type.value)
            
            if model_id is None:
                return None
            
            registered = self._models.get(model_id)
            if registered is None:
                return None
            
            registered.last_used = time.time()
            registered.use_count += 1
            
            return registered.instance
    
    def get_model(self, model_id: str) -> Optional[Any]:
        """Get a specific model by ID."""
        with self._lock:
            registered = self._models.get(model_id)
            return registered.instance if registered else None
    
    def get_model_metadata(self, model_id: str) -> Optional[ModelMetadata]:
        """Get metadata for a model."""
        with self._lock:
            registered = self._models.get(model_id)
            return registered.metadata if registered else None
    
    def select_model_for_regime(self, 
                                 model_type: ModelType,
                                 regime: str) -> Optional[str]:
        """
        Select best model for current market regime.
        
        Args:
            model_type: Type of model needed
            regime: Current market regime
        
        Returns:
            Model ID or None
        """
        with self._lock:
            candidates = []
            
            for model_id, registered in self._models.items():
                if registered.metadata.model_type != model_type:
                    continue
                
                if registered.status == ModelStatus.DEPRECATED:
                    continue
                
                # Check regime compatibility
                compat = registered.metadata.regime_compatibility
                if not compat or regime in compat:
                    candidates.append((model_id, registered))
            
            if not candidates:
                return None
            
            # Select by performance metric (e.g., sharpe_ratio)
            best_model = max(
                candidates,
                key=lambda x: x[1].metadata.performance_metrics.get("sharpe_ratio", 0)
            )
            
            return best_model[0]
    
    def swap_for_regime(self, model_type: ModelType, regime: str) -> Optional[str]:
        """
        Swap active model based on regime.
        
        Args:
            model_type: Type of model
            regime: Current regime
        
        Returns:
            New active model ID or None
        """
        model_id = self.select_model_for_regime(model_type, regime)
        
        if model_id:
            self.set_active(model_id, model_type)
            return model_id
        
        return None
    
    def list_models(self, 
                    model_type: Optional[ModelType] = None,
                    status: Optional[ModelStatus] = None) -> List[Dict[str, Any]]:
        """List registered models with optional filters."""
        with self._lock:
            results = []
            
            for model_id, registered in self._models.items():
                if model_type and registered.metadata.model_type != model_type:
                    continue
                
                if status and registered.status != status:
                    continue
                
                results.append({
                    "model_id": registered.metadata.model_id,
                    "type": registered.metadata.model_type.value,
                    "version": registered.metadata.version,
                    "status": registered.status.value,
                    "file_path": registered.metadata.file_path,
                    "input_dim": registered.metadata.input_dim,
                    "output_dim": registered.metadata.output_dim,
                    "tags": registered.metadata.tags,
                    "regime_compatibility": registered.metadata.regime_compatibility,
                    "performance_metrics": registered.metadata.performance_metrics,
                    "use_count": registered.use_count,
                    "last_used": registered.last_used,
                })
            
            return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics."""
        with self._lock:
            active_count = sum(
                1 for r in self._models.values() 
                if r.status == ModelStatus.ACTIVE
            )
            
            return {
                "total_models": len(self._models),
                "active_models": active_count,
                "standby_models": sum(
                    1 for r in self._models.values() 
                    if r.status == ModelStatus.STANDBY
                ),
                "total_loads": self._total_loads,
                "total_swaps": self._total_swaps,
                "total_errors": self._total_errors,
                "active_by_type": dict(self._active_model_ids),
            }
    
    def update_performance_metrics(self, 
                                    model_id: str,
                                    metrics: Dict[str, float]) -> bool:
        """Update performance metrics for a model."""
        with self._lock:
            if model_id not in self._models:
                return False
            
            self._models[model_id].metadata.performance_metrics.update(metrics)
            return True


# Global registry instance
_registry: Optional[ModelRegistry] = None
_lock = threading.Lock()


def get_model_registry() -> ModelRegistry:
    """Get global ModelRegistry instance."""
    global _registry
    
    with _lock:
        if _registry is None:
            _registry = ModelRegistry()
        
        return _registry


def reset_model_registry() -> None:
    """Reset the global registry."""
    global _registry
    
    with _lock:
        if _registry is not None:
            _registry = None


if __name__ == "__main__":
    print("Model Registry Demo")
    print("=" * 40)
    
    registry = get_model_registry()
    
    # Register mock models
    class MockModel:
        pass
    
    registry.register_model(
        model_id="trend_xgb_v1",
        model_instance=MockModel(),
        model_type=ModelType.XGB,
        version="1.0.0",
        tags=["trend", "production"],
        regime_compatibility=["trending"]
    )
    
    registry.register_model(
        model_id="meanrev_xgb_v1",
        model_instance=MockModel(),
        model_type=ModelType.XGB,
        version="1.0.0",
        tags=["mean_reversion", "production"],
        regime_compatibility=["mean_reverting"]
    )
    
    # Set active
    registry.set_active("trend_xgb_v1", ModelType.XGB)
    
    # List models
    models = registry.list_models()
    print(f"\nRegistered Models:")
    for m in models:
        print(f"  - {m['model_id']} ({m['type']}) v{m['version']}")
        print(f"    Status: {m['status']}, Regimes: {m['regime_compatibility']}")
    
    # Swap for regime
    new_active = registry.swap_for_regime(ModelType.XGB, "mean_reverting")
    print(f"\nSwapped to: {new_active}")
    
    # Get statistics
    stats = registry.get_statistics()
    print(f"\nStatistics: {stats}")
    
    # Cleanup
    reset_model_registry()
    print("\nRegistry demo complete")
