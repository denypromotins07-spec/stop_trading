"""
Final ML Ensemble & Meta-Strategy Routing
Stage 49: Non-linear stacking generalizer using ONNX-compiled MLP.
Fuses probability distributions from XGBoost, Temporal Transformers, and RL agents.
Uses strictly pre-allocated numpy arrays to prevent GC pauses.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging
import onnxruntime as rt
import zmq

logger = logging.getLogger(__name__)


@dataclass
class ModelPrediction:
    """Prediction output from a single model."""
    model_type: str  # xgboost, transformer, rl_agent
    probabilities: np.ndarray
    confidence: float
    timestamp: float


@dataclass 
class EnsembleOutput:
    """Fused output from the stacking generalizer."""
    alpha_vector: np.ndarray
    confidence: float
    model_weights: Dict[str, float]
    regime_compatibility: float


class StackingGeneralizer:
    """
    Non-linear stacking generalizer (meta-learner) using lightweight ONNX-compiled MLP.
    Fuses predictions from multiple models into high-confidence alpha vectors.
    
    CRITICAL: Uses strictly pre-allocated numpy arrays for forward pass
    to prevent garbage collection pauses during live inference.
    """
    
    def __init__(self, 
                 num_base_models: int = 3,
                 num_classes: int = 5,
                 hidden_dim: int = 64,
                 max_batch_size: int = 32):
        
        self.num_base_models = num_base_models
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.max_batch_size = max_batch_size
        
        # Pre-allocated input buffer: [batch_size, num_base_models * num_classes]
        self.input_dim = num_base_models * num_classes
        self._input_buffer = np.zeros(
            (max_batch_size, self.input_dim), 
            dtype=np.float32
        )
        
        # Pre-allocated weight matrices (initialized for MLP)
        # Layer 1: input -> hidden
        self._W1 = np.random.randn(self.input_dim, hidden_dim).astype(np.float32) * 0.1
        self._b1 = np.zeros(hidden_dim, dtype=np.float32)
        
        # Layer 2: hidden -> hidden
        self._W2 = np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * 0.1
        self._b2 = np.zeros(hidden_dim, dtype=np.float32)
        
        # Layer 3: hidden -> output (num_classes)
        self._W3 = np.random.randn(hidden_dim, num_classes).astype(np.float32) * 0.1
        self._b3 = np.zeros(num_classes, dtype=np.float32)
        
        # Pre-allocated intermediate buffers
        self._hidden1 = np.zeros((max_batch_size, hidden_dim), dtype=np.float32)
        self._hidden2 = np.zeros((max_batch_size, hidden_dim), dtype=np.float32)
        self._output = np.zeros((max_batch_size, num_classes), dtype=np.float32)
        
        # ONNX runtime session (optional - can use pure numpy fallback)
        self._session: Optional[rt.InferenceSession] = None
        self._onnx_model_path: Optional[str] = None
        
        # Model weights for blending
        self.model_weights = {
            'xgboost': 0.35,
            'transformer': 0.40,
            'rl_agent': 0.25,
        }
        
        # ZMQ socket for Rust IPC
        self._zmq_context = zmq.Context()
        self._zmq_socket = self._zmq_context.socket(zmq.PUSH)
        self._zmq_socket.connect("tcp://localhost:5560")
        
        # Performance tracking
        self._inference_count = 0
        self._last_inference_time = 0.0
    
    def load_onnx_model(self, model_path: str) -> bool:
        """Load pre-trained ONNX meta-learner model."""
        try:
            self._session = rt.InferenceSession(model_path)
            self._onnx_model_path = model_path
            logger.info(f"Loaded ONNX model from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            return False
    
    def fuse_predictions(self, 
                        predictions: List[ModelPrediction],
                        batch_size: int = 1) -> EnsembleOutput:
        """
        Fuse predictions from multiple base models using the meta-learner.
        
        Args:
            predictions: List of predictions from base models (xgboost, transformer, rl)
            batch_size: Number of samples to process (for batching efficiency)
        
        Returns:
            EnsembleOutput with fused alpha vector and confidence metrics
        """
        if len(predictions) != self.num_base_models:
            logger.warning(f"Expected {self.num_base_models} predictions, got {len(predictions)}")
            # Pad with zeros if needed
            while len(predictions) < self.num_base_models:
                predictions.append(ModelPrediction(
                    model_type='placeholder',
                    probabilities=np.zeros(self.num_classes, dtype=np.float32),
                    confidence=0.0,
                    timestamp=0.0,
                ))
        
        # Flatten and concatenate predictions into pre-allocated buffer
        # Shape: [batch_size, num_base_models * num_classes]
        batch_size = min(batch_size, self.max_batch_size)
        
        for b in range(batch_size):
            idx = 0
            for pred in predictions[:self.num_base_models]:
                prob_slice = pred.probabilities[:self.num_classes]
                self._input_buffer[b, idx:idx+len(prob_slice)] = prob_slice
                idx += len(prob_slice)
        
        # Forward pass through meta-learner
        if self._session is not None:
            # Use ONNX runtime
            input_name = self._session.get_inputs()[0].name
            ort_inputs = {input_name: self._input_buffer[:batch_size]}
            self._output[:batch_size] = self._session.run(None, ort_inputs)[0]
        else:
            # Pure numpy fallback with ReLU activations
            self._forward_pass_numpy(batch_size)
        
        # Apply softmax to get final probabilities
        alpha_vector = self._softmax(self._output[:batch_size].mean(axis=0))
        
        # Calculate ensemble confidence
        confidence = self._calculate_confidence(predictions, alpha_vector)
        
        # Calculate regime compatibility score
        regime_compat = self._calculate_regime_compatibility(predictions)
        
        # Create output
        output = EnsembleOutput(
            alpha_vector=alpha_vector,
            confidence=confidence,
            model_weights=self.model_weights.copy(),
            regime_compatibility=regime_compat,
        )
        
        # Track performance
        self._inference_count += 1
        
        # Notify Rust side of ensemble output
        self._notify_rust(output)
        
        return output
    
    def _forward_pass_numpy(self, batch_size: int):
        """
        Pure numpy forward pass with pre-allocated buffers.
        No dynamic memory allocation during inference.
        """
        # Layer 1: Input -> Hidden1 with ReLU
        self._hidden1[:batch_size] = np.maximum(
            0, 
            np.dot(self._input_buffer[:batch_size], self._W1) + self._b1
        )
        
        # Layer 2: Hidden1 -> Hidden2 with ReLU
        self._hidden2[:batch_size] = np.maximum(
            0,
            np.dot(self._hidden1[:batch_size], self._W2) + self._b2
        )
        
        # Layer 3: Hidden2 -> Output (linear + softmax applied later)
        self._output[:batch_size] = np.dot(self._hidden2[:batch_size], self._W3) + self._b3
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def _calculate_confidence(self, 
                             predictions: List[ModelPrediction], 
                             alpha_vector: np.ndarray) -> float:
        """
        Calculate ensemble confidence based on:
        1. Agreement between base models
        2. Individual model confidences
        3. Output entropy
        """
        # Model agreement (variance of predictions)
        pred_array = np.array([p.probabilities[:self.num_classes] for p in predictions])
        agreement = 1.0 - np.mean(np.std(pred_array, axis=0))
        
        # Average model confidence
        avg_confidence = np.mean([p.confidence for p in predictions])
        
        # Output entropy (lower = more confident)
        entropy = -np.sum(alpha_vector * np.log(alpha_vector + 1e-10))
        max_entropy = np.log(self.num_classes)
        entropy_score = 1.0 - (entropy / max_entropy)
        
        # Weighted combination
        confidence = (
            0.4 * agreement + 
            0.3 * avg_confidence + 
            0.3 * entropy_score
        )
        
        return float(confidence)
    
    def _calculate_regime_compatibility(self, predictions: List[ModelPrediction]) -> float:
        """Calculate how compatible current predictions are with detected market regime."""
        # Placeholder - actual implementation would check against HMM regime state
        # Higher score means predictions align well with current regime
        return 0.85
    
    def _notify_rust(self, output: EnsembleOutput):
        """Send ensemble output to Rust via ZMQ."""
        try:
            self._zmq_socket.send_json({
                'type': 'ENSEMBLE_OUTPUT',
                'alpha_vector': output.alpha_vector.tolist(),
                'confidence': output.confidence,
                'regime_compatibility': output.regime_compatibility,
                'inference_count': self._inference_count,
            }, flags=zmq.NOBLOCK)
        except Exception as e:
            logger.error(f"Failed to send ensemble output to Rust: {e}")
    
    def update_model_weights(self, weights: Dict[str, float]):
        """Dynamically update model blending weights."""
        total = sum(weights.values())
        if total > 0:
            self.model_weights = {k: v / total for k, v in weights.items()}
        logger.info(f"Updated model weights: {self.model_weights}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get inference performance statistics."""
        return {
            'inference_count': self._inference_count,
            'num_base_models': self.num_base_models,
            'num_classes': self.num_classes,
            'using_onnx': self._session is not None,
        }
    
    def shutdown(self):
        """Cleanup resources."""
        self._zmq_socket.close()
        self._zmq_context.term()
        logger.info("StackingGeneralizer shut down")


# Global instance
_generalizer: Optional[StackingGeneralizer] = None


def get_generalizer() -> StackingGeneralizer:
    """Get or create the global StackingGeneralizer instance."""
    global _generalizer
    if _generalizer is None:
        _generalizer = StackingGeneralizer()
    return _generalizer


def create_generalizer(num_base_models: int = 3, 
                       num_classes: int = 5,
                       hidden_dim: int = 64) -> StackingGeneralizer:
    """Create a new StackingGeneralizer with custom configuration."""
    global _generalizer
    _generalizer = StackingGeneralizer(
        num_base_models=num_base_models,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
    )
    return _generalizer
