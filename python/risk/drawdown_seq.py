"""
Dynamic Drawdown Predictor using ONNX-compiled LSTM
Implements an ONNX-compiled LSTM to predict the probability of breaching max drawdown thresholds.
Triggers automated deleveraging in Nautilus strategies if predicted drawdown probability exceeds safety limits.
Uses strictly bounded sequence lengths and ONNX CPU providers to respect the 3GB RAM ceiling.
Normalizes inputs using Welford online statistics matching Rust feature store calculations.
"""

import numpy as np
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
import logging

# Conditional ONNX runtime import
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    ort = None  # type: ignore


logger = logging.getLogger(__name__)


@dataclass
class DrawdownPrediction:
    """Drawdown prediction result."""
    predicted_drawdown: float
    breach_probability: float
    confidence: float
    recommended_leverage: float
    time_horizon_steps: int
    risk_level: str  # "LOW", "MEDIUM", "HIGH", "CRITICAL"


class WelfordStatistics:
    """
    Welford online algorithm for computing running mean and variance.
    Matches Rust feature store normalization exactly.
    """

    def __init__(self):
        self.count: int = 0
        self.mean: float = 0.0
        self.m2: float = 0.0

    def update(self, value: float) -> None:
        """Update statistics with new value."""
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

    @property
    def variance(self) -> float:
        """Return population variance."""
        if self.count < 1:
            return 0.0
        return self.m2 / self.count

    @property
    def std(self) -> float:
        """Return standard deviation."""
        return np.sqrt(self.variance)

    def normalize(self, value: float) -> float:
        """Normalize value using current statistics."""
        if self.std < 1e-9:
            return 0.0
        return (value - self.mean) / self.std

    def reset(self) -> None:
        """Reset all statistics."""
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0


class DrawdownPredictorLSTM:
    """
    ONNX-compiled LSTM for drawdown prediction.
    Uses bounded sequence lengths and CPU-only execution for memory efficiency.
    """

    # Risk level thresholds
    LOW_THRESHOLD = 0.2
    MEDIUM_THRESHOLD = 0.4
    HIGH_THRESHOLD = 0.6

    def __init__(
        self,
        model_path: Optional[str] = None,
        sequence_length: int = 50,
        feature_dim: int = 8,
        max_drawdown_threshold: float = 0.05,
        cpu_threads: int = 1,
    ):
        """
        Initialize the drawdown predictor.

        Args:
            model_path: Path to ONNX model file
            sequence_length: Fixed sequence length for LSTM input
            feature_dim: Number of input features
            max_drawdown_threshold: Maximum acceptable drawdown
            cpu_threads: Number of CPU threads for inference
        """
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.max_drawdown_threshold = max_drawdown_threshold

        # Pre-allocated buffers to avoid allocation during inference
        self._input_buffer: np.ndarray = np.zeros(
            (1, sequence_length, feature_dim), dtype=np.float32
        )
        self._sequence_buffer: List[np.ndarray] = []
        self._seq_idx: int = 0

        # Welford statistics for each feature (matching Rust)
        self._feature_stats: List[WelfordStatistics] = [
            WelfordStatistics() for _ in range(feature_dim)
        ]

        # Model state
        self._session: Optional[ort.InferenceSession] = None  # type: ignore
        self._is_loaded = False

        # Configure ONNX runtime for low-latency CPU inference
        if ONNX_AVAILABLE and model_path:
            self._load_model(model_path, cpu_threads)

    def _load_model(self, model_path: str, cpu_threads: int) -> None:
        """Load ONNX model with optimized session options."""
        try:
            # Configure session for minimal memory footprint
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = cpu_threads
            sess_options.inter_op_num_threads = 1
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.enable_cpu_mem_arena = True

            # Use only CPU providers to stay within memory limits
            providers = ['CPUExecutionProvider']

            self._session = ort.InferenceSession(
                model_path,
                sess_options=sess_options,
                providers=providers,
            )
            self._is_loaded = True
            logger.info(f"Loaded ONNX model from {model_path}")

        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            self._is_loaded = False

    def create_default_model(self) -> bytes:
        """
        Create a minimal default LSTM model for testing.
        In production, this would be trained and exported from PyTorch/TensorFlow.
        """
        if not ONNX_AVAILABLE:
            raise RuntimeError("ONNX runtime not available")

        try:
            from onnx import helper, TensorProto, numpy_helper
            import onnx

            # Create a simple LSTM-based model structure
            # This is a placeholder - real model would be trained separately

            # Input definition
            input_tensor = helper.make_tensor_value_info(
                'input', TensorProto.FLOAT, [1, self.sequence_length, self.feature_dim]
            )

            # Output definition
            output_tensor = helper.make_tensor_value_info(
                'output', TensorProto.FLOAT, [1, 2]  # [drawdown, breach_prob]
            )

            # Simple identity-like graph for placeholder
            # Real implementation would have proper LSTM nodes
            node = helper.make_node(
                'ReduceMean',
                inputs=['input'],
                outputs=['output'],
                axes=[1, 2],
                keepdims=0,
            )

            graph_def = helper.make_graph(
                [node],
                'drawdown_lstm',
                [input_tensor],
                [output_tensor],
            )

            model_def = helper.make_model(graph_def, opset_imports=[helper.make_opsetid("", 13)])
            model_def.ir_version = 7

            return model_def.SerializeToString()

        except ImportError:
            # Fallback: create minimal numpy-based prediction
            logger.warning("Could not create ONNX model, using fallback")
            return b""

    def add_feature_vector(self, features: np.ndarray) -> None:
        """
        Add a new feature vector to the sequence buffer.
        Features are normalized using Welford statistics.

        Args:
            features: Array of shape (feature_dim,)
        """
        if len(features) != self.feature_dim:
            raise ValueError(f"Expected {self.feature_dim} features, got {len(features)}")

        # Update Welford statistics and normalize
        normalized = np.zeros(self.feature_dim, dtype=np.float32)
        for i in range(self.feature_dim):
            self._feature_stats[i].update(float(features[i]))
            normalized[i] = self._feature_stats[i].normalize(float(features[i]))

        # Add to sequence buffer
        if len(self._sequence_buffer) < self.sequence_length:
            self._sequence_buffer.append(normalized.copy())
        else:
            # Circular buffer behavior
            self._sequence_buffer.pop(0)
            self._sequence_buffer.append(normalized.copy())

    def _prepare_input(self) -> Optional[np.ndarray]:
        """Prepare input tensor for model inference."""
        if len(self._sequence_buffer) < self.sequence_length:
            return None

        # Copy to pre-allocated buffer
        for i, seq in enumerate(self._sequence_buffer):
            self._input_buffer[0, i, :] = seq

        return self._input_buffer

    def predict(self) -> Optional[DrawdownPrediction]:
        """
        Predict drawdown and breach probability.
        Returns None if insufficient sequence data.

        Uses ONNX runtime with CPU provider for memory-efficient inference.
        """
        if not self._is_loaded or self._session is None:
            # Fallback prediction without model
            return self._fallback_prediction()

        input_tensor = self._prepare_input()
        if input_tensor is None:
            return None

        try:
            # Run inference with bounded output
            outputs = self._session.run(None, {'input': input_tensor})

            # Parse outputs
            raw_output = outputs[0][0]
            predicted_drawdown = float(raw_output[0])
            breach_probability = float(raw_output[1]) if len(raw_output) > 1 else 0.0

            # Clamp predictions to valid ranges
            predicted_drawdown = np.clip(predicted_drawdown, 0.0, 1.0)
            breach_probability = np.clip(breach_probability, 0.0, 1.0)

            # Calculate recommended leverage based on breach probability
            recommended_leverage = self._calculate_recommended_leverage(breach_probability)

            # Determine risk level
            risk_level = self._get_risk_level(breach_probability)

            return DrawdownPrediction(
                predicted_drawdown=predicted_drawdown,
                breach_probability=breach_probability,
                confidence=0.8,  # Would come from model in production
                recommended_leverage=recommended_leverage,
                time_horizon_steps=self.sequence_length,
                risk_level=risk_level,
            )

        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return self._fallback_prediction()

    def _fallback_prediction(self) -> Optional[DrawdownPrediction]:
        """Fallback prediction when model is not available."""
        if len(self._sequence_buffer) < self.sequence_length:
            return None

        # Simple heuristic based on recent volatility
        recent_features = np.array(self._sequence_buffer[-10:])
        volatility = np.std(recent_features[:, 0]) if len(recent_features) > 0 else 0.0

        # Map volatility to breach probability (simplified)
        breach_probability = min(volatility * 10, 1.0)
        predicted_drawdown = breach_probability * 0.5

        return DrawdownPrediction(
            predicted_drawdown=predicted_drawdown,
            breach_probability=breach_probability,
            confidence=0.3,
            recommended_leverage=self._calculate_recommended_leverage(breach_probability),
            time_horizon_steps=self.sequence_length,
            risk_level=self._get_risk_level(breach_probability),
        )

    def _calculate_recommended_leverage(self, breach_probability: float) -> float:
        """
        Calculate recommended leverage based on breach probability.
        Reduces leverage linearly as breach probability increases.
        """
        # Base leverage
        base_leverage = 1.0

        # Reduction factor based on breach probability
        if breach_probability > self.HIGH_THRESHOLD:
            # Critical: reduce to minimal leverage
            reduction = 0.8
        elif breach_probability > self.MEDIUM_THRESHOLD:
            # High risk: significant reduction
            reduction = 0.5
        elif breach_probability > self.LOW_THRESHOLD:
            # Medium risk: moderate reduction
            reduction = 0.25
        else:
            # Low risk: full leverage allowed
            reduction = 0.0

        return base_leverage * (1.0 - reduction)

    def _get_risk_level(self, breach_probability: float) -> str:
        """Determine risk level from breach probability."""
        if breach_probability > self.HIGH_THRESHOLD:
            return "CRITICAL"
        elif breach_probability > self.MEDIUM_THRESHOLD:
            return "HIGH"
        elif breach_probability > self.LOW_THRESHOLD:
            return "MEDIUM"
        else:
            return "LOW"

    def should_delever(self, prediction: DrawdownPrediction, threshold: float = 0.5) -> bool:
        """
        Check if deleveraging should be triggered.
        """
        return prediction.breach_probability > threshold

    def get_normalization_stats(self) -> List[Dict[str, float]]:
        """Get current Welford statistics for all features."""
        return [
            {
                "mean": stats.mean,
                "std": stats.std,
                "count": stats.count,
            }
            for stats in self._feature_stats
        ]

    def set_normalization_stats(self, means: List[float], stds: List[float]) -> None:
        """
        Set normalization statistics to match Rust feature store.
        Used for consistent normalization across Python/Rust boundary.
        """
        if len(means) != self.feature_dim or len(stds) != self.feature_dim:
            raise ValueError("Stats dimension mismatch")

        for i in range(self.feature_dim):
            # Set Welford stats to match provided parameters
            self._feature_stats[i].mean = means[i]
            self._feature_stats[i].m2 = (stds[i] ** 2) * 100  # Approximate from sample
            self._feature_stats[i].count = 100

    def reset(self) -> None:
        """Reset all state including sequence buffer and statistics."""
        self._sequence_buffer.clear()
        self._seq_idx = 0
        self._input_buffer.fill(0.0)
        for stats in self._feature_stats:
            stats.reset()

    def save_model(self, path: str) -> None:
        """Save current model to file."""
        if self._session is not None and ONNX_AVAILABLE:
            try:
                import onnx
                # Get model proto and save
                model_proto = onnx.load_model_from_string(
                    self._session._sess.model_data  # type: ignore
                )
                onnx.save(model_proto, path)
                logger.info(f"Model saved to {path}")
            except Exception as e:
                logger.error(f"Failed to save model: {e}")

    def load_model(self, path: str) -> None:
        """Load model from file."""
        self._load_model(path, cpu_threads=1)
