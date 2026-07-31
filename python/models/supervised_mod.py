"""
Supervised Learning Module Root
Manages ensemble weights and routes predictions to Nautilus MessageBus.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
import threading
import queue
import time
import logging

from .xgb_ensemble import XGBEnsemble, EnsembleConfig
from .tabular_predictor import TabularPredictor, BatchConfig, PredictionResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelWeight:
    """Weight configuration for ensemble model."""
    model_id: str
    weight: float
    model_type: str  # "trend", "mean_reversion", "momentum"
    active: bool = True


@dataclass
class EnsembleWeights:
    """Container for managing multiple model weights."""
    weights: Dict[str, ModelWeight] = field(default_factory=dict)
    
    def add_weight(self, model_id: str, weight: float, model_type: str) -> None:
        self.weights[model_id] = ModelWeight(
            model_id=model_id,
            weight=weight,
            model_type=model_type,
            active=True
        )
    
    def remove_weight(self, model_id: str) -> None:
        if model_id in self.weights:
            del self.weights[model_id]
    
    def set_active(self, model_id: str, active: bool) -> None:
        if model_id in self.weights:
            self.weights[model_id].active = active
    
    def get_active_weights(self) -> Dict[str, float]:
        return {
            mid: mw.weight 
            for mid, mw in self.weights.items() 
            if mw.active
        }
    
    def normalize_weights(self) -> Dict[str, float]:
        """Normalize weights to sum to 1.0."""
        active = self.get_active_weights()
        total = sum(active.values())
        if total == 0:
            return active
        return {mid: w / total for mid, w in active.items()}


@dataclass
class SignalMessage:
    """Message structure for Nautilus MessageBus."""
    signal_id: str
    timestamp: float
    instrument_id: str
    alpha_score: float
    probability: float
    model_ids: List[str]
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class SupervisedLearningModule:
    """
    Central module for supervised learning inference.
    Manages ensemble weights and routes predictions to Nautilus MessageBus.
    """
    
    def __init__(self, 
                 message_bus_callback: Optional[Callable[[SignalMessage], None]] = None):
        self._models: Dict[str, TabularPredictor] = {}
        self._ensemble_weights = EnsembleWeights()
        self._message_bus_callback = message_bus_callback
        
        # Thread-safe message queue
        self._message_queue: queue.Queue = queue.Queue(maxsize=10000)
        
        # Routing thread
        self._running = False
        self._router_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Statistics
        self._signals_sent = 0
        self._last_signal_time = 0.0
    
    def register_model(self, 
                       model_id: str,
                       ensemble: XGBEnsemble,
                       n_features: int,
                       batch_config: Optional[BatchConfig] = None,
                       weight: float = 1.0,
                       model_type: str = "default") -> None:
        """
        Register a new ensemble model.
        
        Args:
            model_id: Unique identifier for the model
            ensemble: Trained XGBEnsemble instance
            n_features: Number of input features
            batch_config: Batch prediction configuration
            weight: Initial weight for this model
            model_type: Type of model (trend, mean_reversion, momentum)
        """
        with self._lock:
            predictor = TabularPredictor(ensemble, batch_config)
            predictor.initialize(n_features)
            
            self._models[model_id] = predictor
            self._ensemble_weights.add_weight(model_id, weight, model_type)
            
            logger.info(f"Registered model {model_id} with {n_features} features")
    
    def unregister_model(self, model_id: str) -> None:
        """Unregister a model."""
        with self._lock:
            if model_id in self._models:
                predictor = self._models.pop(model_id)
                predictor.stop_background_worker()
                self._ensemble_weights.remove_weight(model_id)
                logger.info(f"Unregistered model {model_id}")
    
    def submit_features(self, 
                        features: np.ndarray,
                        instrument_id: str,
                        timestamp: Optional[float] = None) -> None:
        """
        Submit features to all registered models.
        
        Args:
            features: Feature vector (n_features,)
            instrument_id: Target instrument ID
            timestamp: Optional timestamp
        """
        with self._lock:
            for model_id, predictor in self._models.items():
                if predictor.is_initialized:
                    try:
                        predictor.submit_features(features, timestamp)
                    except Exception as e:
                        logger.error(f"Error submitting to {model_id}: {e}")
    
    def get_prediction(self, model_id: str, 
                       timeout_ms: float = 10.0) -> Optional[PredictionResult]:
        """
        Get prediction from a specific model.
        
        Args:
            model_id: Model identifier
            timeout_ms: Timeout in milliseconds
        
        Returns:
            PredictionResult or None
        """
        with self._lock:
            if model_id not in self._models:
                logger.warning(f"Model {model_id} not found")
                return None
            
            predictor = self._models[model_id]
            return predictor.predict_batch(timeout_ms)
    
    def get_ensemble_prediction(self, 
                                features: np.ndarray,
                                timeout_ms: float = 50.0) -> Optional[Dict[str, Any]]:
        """
        Get weighted ensemble prediction from all active models.
        
        Args:
            features: Feature vector (n_features,)
            timeout_ms: Total timeout for all models
        
        Returns:
            Dictionary with ensemble prediction details
        """
        active_weights = self._ensemble_weights.get_active_weights()
        
        if not active_weights:
            logger.warning("No active models for ensemble prediction")
            return None
        
        predictions = []
        probabilities = []
        model_ids = []
        
        start_time = time.time()
        per_model_timeout = timeout_ms / len(active_weights)
        
        with self._lock:
            for model_id in active_weights.keys():
                if time.time() - start_time > timeout_ms / 1000:
                    break
                
                if model_id not in self._models:
                    continue
                
                predictor = self._models[model_id]
                
                # Use single prediction for low latency
                result = predictor.predict_single(features)
                
                if result is not None:
                    predictions.append(result.predictions[0])
                    if result.probabilities is not None:
                        probabilities.append(result.probabilities[0, 1])  # Positive class
                    else:
                        probabilities.append(0.5)
                    model_ids.append(model_id)
        
        if not predictions:
            return None
        
        # Calculate weighted ensemble prediction
        weights = np.array([active_weights[mid] for mid in model_ids])
        weights = weights / weights.sum()  # Normalize
        
        ensemble_alpha = np.dot(predictions, weights)
        ensemble_prob = np.dot(probabilities, weights)
        
        # Calculate confidence as inverse of prediction variance
        if len(predictions) > 1:
            confidence = 1.0 / (np.std(predictions) + 1e-6)
        else:
            confidence = 1.0
        
        return {
            "alpha": float(ensemble_alpha),
            "probability": float(ensemble_prob),
            "confidence": float(confidence),
            "model_ids": model_ids,
            "weights": {mid: float(weights[i]) for i, mid in enumerate(model_ids)},
            "raw_predictions": predictions,
        }
    
    def create_signal_message(self, 
                              instrument_id: str,
                              ensemble_result: Dict[str, Any]) -> SignalMessage:
        """
        Create a SignalMessage for Nautilus MessageBus.
        
        Args:
            instrument_id: Target instrument
            ensemble_result: Result from get_ensemble_prediction
        
        Returns:
            SignalMessage instance
        """
        return SignalMessage(
            signal_id=f"sl_{instrument_id}_{time.time_ns()}",
            timestamp=time.time(),
            instrument_id=instrument_id,
            alpha_score=ensemble_result["alpha"],
            probability=ensemble_result["probability"],
            model_ids=ensemble_result["model_ids"],
            confidence=ensemble_result["confidence"],
            metadata={
                "weights": ensemble_result["weights"],
                "raw_predictions": ensemble_result["raw_predictions"],
            }
        )
    
    def send_to_message_bus(self, message: SignalMessage) -> None:
        """
        Send signal message to Nautilus MessageBus.
        
        Args:
            message: SignalMessage to send
        """
        try:
            self._message_queue.put_nowait(message)
            
            if self._message_bus_callback is not None:
                self._message_bus_callback(message)
            
            with self._lock:
                self._signals_sent += 1
                self._last_signal_time = time.time()
        
        except queue.Full:
            logger.warning("Message queue full, dropping signal")
    
    def process_and_send(self, 
                         features: np.ndarray,
                         instrument_id: str,
                         timestamp: Optional[float] = None) -> Optional[SignalMessage]:
        """
        Full pipeline: submit features, get ensemble prediction, send to message bus.
        
        Args:
            features: Feature vector
            instrument_id: Target instrument
            timestamp: Optional timestamp
        
        Returns:
            Sent SignalMessage or None
        """
        ensemble_result = self.get_ensemble_prediction(features)
        
        if ensemble_result is None:
            return None
        
        message = self.create_signal_message(instrument_id, ensemble_result)
        self.send_to_message_bus(message)
        
        return message
    
    def start_message_router(self) -> None:
        """Start background message router thread."""
        if self._running:
            return
        
        self._running = True
        self._router_thread = threading.Thread(
            target=self._router_loop, 
            daemon=True,
            name="SL_MessageRouter"
        )
        self._router_thread.start()
        logger.info("Started message router thread")
    
    def stop_message_router(self) -> None:
        """Stop background message router thread."""
        self._running = False
        if self._router_thread is not None:
            self._router_thread.join(timeout=2.0)
            self._router_thread = None
        logger.info("Stopped message router thread")
    
    def _router_loop(self) -> None:
        """Background loop to process message queue."""
        while self._running:
            try:
                message = self._message_queue.get(timeout=0.1)
                
                if self._message_bus_callback is not None:
                    self._message_bus_callback(message)
                
                self._message_queue.task_done()
            
            except queue.Empty:
                pass
            except Exception as e:
                logger.error(f"Router error: {e}")
    
    def update_model_weight(self, model_id: str, new_weight: float) -> None:
        """Update weight for a specific model."""
        self._ensemble_weights.set_active(model_id, new_weight > 0)
        if model_id in self._ensemble_weights.weights:
            self._ensemble_weights.weights[model_id].weight = new_weight
            logger.info(f"Updated weight for {model_id}: {new_weight}")
    
    def activate_model(self, model_id: str) -> None:
        """Activate a model for ensemble predictions."""
        self._ensemble_weights.set_active(model_id, True)
        logger.info(f"Activated model {model_id}")
    
    def deactivate_model(self, model_id: str) -> None:
        """Deactivate a model from ensemble predictions."""
        self._ensemble_weights.set_active(model_id, False)
        logger.info(f"Deactivated model {model_id}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get module statistics."""
        stats = {
            "registered_models": list(self._models.keys()),
            "active_models": list(self._ensemble_weights.get_active_weights().keys()),
            "signals_sent": self._signals_sent,
            "last_signal_time": self._last_signal_time,
            "queue_size": self._message_queue.qsize(),
        }
        
        # Add per-model statistics
        with self._lock:
            for model_id, predictor in self._models.items():
                stats[f"{model_id}_stats"] = predictor.get_statistics()
        
        return stats
    
    @property
    def is_running(self) -> bool:
        return self._running


# Global module instance (singleton pattern)
_supervised_module: Optional[SupervisedLearningModule] = None
_module_lock = threading.Lock()


def get_supervised_module(
    message_bus_callback: Optional[Callable[[SignalMessage], None]] = None
) -> SupervisedLearningModule:
    """
    Get or create the global SupervisedLearningModule instance.
    
    Args:
        message_bus_callback: Callback for MessageBus integration
    
    Returns:
        SupervisedLearningModule instance
    """
    global _supervised_module
    
    with _module_lock:
        if _supervised_module is None:
            _supervised_module = SupervisedLearningModule(message_bus_callback)
        
        return _supervised_module


def reset_supervised_module() -> None:
    """Reset the global module instance."""
    global _supervised_module
    
    with _module_lock:
        if _supervised_module is not None:
            _supervised_module.stop_message_router()
            _supervised_module = None


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Create callback mock
    def mock_callback(msg: SignalMessage):
        print(f"Signal sent: {msg.signal_id}, alpha={msg.alpha_score:.4f}")
    
    # Initialize module
    module = get_supervised_module(mock_callback)
    
    # Create and register a model
    config = EnsembleConfig(n_threads=4, n_estimators=50)
    ensemble = XGBEnsemble(config)
    
    # Train on dummy data
    X_train = np.random.randn(500, 30).astype(np.float32)
    y_train = np.random.randint(0, 2, 500).astype(np.float32)
    ensemble.fit(X_train, y_train)
    
    # Register model
    module.register_model(
        model_id="trend_model_v1",
        ensemble=ensemble,
        n_features=30,
        weight=1.0,
        model_type="trend"
    )
    
    # Start message router
    module.start_message_router()
    
    # Process features
    features = np.random.randn(30).astype(np.float32)
    message = module.process_and_send(features, "BTC/USDT")
    
    if message:
        print(f"Generated signal: {message.signal_id}")
        print(f"Alpha score: {message.alpha_score:.4f}")
        print(f"Confidence: {message.confidence:.4f}")
    
    # Get statistics
    stats = module.get_statistics()
    print(f"\nStatistics: {stats}")
    
    # Cleanup
    module.stop_message_router()
    reset_supervised_module()
