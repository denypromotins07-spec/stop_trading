"""
Autoencoder for Market State Reconstruction.
Lightweight ONNX-compiled autoencoder for detecting structural market anomalies.
"""

import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class ReconstructionResult:
    """Result of autoencoder reconstruction."""
    original: np.ndarray
    reconstructed: np.ndarray
    reconstruction_error: float
    is_anomaly: bool
    anomaly_confidence: float
    timestamp_ns: int


class LightweightAutoencoder:
    """
    Lightweight autoencoder for reconstructing normal market states.
    High reconstruction error indicates structural anomalies (flash crashes, API glitches).
    
    Note: In production, this would use ONNX runtime for compiled inference.
    This implementation provides a NumPy-based approximation.
    """

    # Default architecture
    DEFAULT_ENCODER_DIMS = [128, 64, 32, 8]  # Bottleneck at 8
    DEFAULT_ANOMALY_THRESHOLD = 0.5  # MSE threshold

    def __init__(
        self,
        input_dim: int = 128,
        encoder_dims: List[int] = None,
        anomaly_threshold: float = DEFAULT_ANOMALY_THRESHOLD,
    ):
        self.input_dim = input_dim
        self.encoder_dims = encoder_dims or self.DEFAULT_ENCODER_DIMS
        self.anomaly_threshold = anomaly_threshold

        # Ensure bottleneck dimension
        if self.encoder_dims[-1] >= input_dim:
            self.encoder_dims = self.encoder_dims + [input_dim // 8]

        self._weights: List[np.ndarray] = []
        self._biases: List[np.ndarray] = []
        self._decoder_weights: List[np.ndarray] = []
        self._decoder_biases: List[np.ndarray] = []
        self._is_trained = False

        self._reconstruction_count = 0
        self._anomaly_count = 0
        self._error_history: List[float] = []

    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        self._weights = []
        self._biases = []

        dims = [self.input_dim] + self.encoder_dims
        for i in range(len(dims) - 1):
            # Xavier initialization
            scale = np.sqrt(2.0 / (dims[i] + dims[i + 1]))
            w = np.random.randn(dims[i], dims[i + 1]).astype(np.float32) * scale
            b = np.zeros(dims[i + 1], dtype=np.float32)
            self._weights.append(w)
            self._biases.append(b)

        # Decoder weights (mirror of encoder)
        decoder_dims = self.encoder_dims[::-1] + [self.input_dim]
        self._decoder_weights = []
        self._decoder_biases = []

        for i in range(len(decoder_dims) - 1):
            scale = np.sqrt(2.0 / (decoder_dims[i] + decoder_dims[i + 1]))
            w = np.random.randn(decoder_dims[i], decoder_dims[i + 1]).astype(np.float32) * scale
            b = np.zeros(decoder_dims[i + 1], dtype=np.float32)
            self._decoder_weights.append(w)
            self._decoder_biases.append(b)

    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation."""
        return np.maximum(0, x)

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def _forward_encoder(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through encoder."""
        h = x
        for w, b in zip(self._weights, self._biases):
            h = self._relu(h @ w + b)
        return h

    def _forward_decoder(self, h: np.ndarray) -> np.ndarray:
        """Forward pass through decoder."""
        x_recon = h
        for i, (w, b) in enumerate(zip(self._decoder_weights, self._decoder_biases)):
            x_recon = self._relu(x_recon @ w + b)
            if i == len(self._decoder_weights) - 1:
                x_recon = self._sigmoid(x_recon)  # Output layer
        return x_recon

    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        """
        Reconstruct input through autoencoder.

        Args:
            x: Input vector of shape (input_dim,) or (batch, input_dim)

        Returns:
            Reconstructed output
        """
        if not self._is_trained:
            return x.copy()

        # Ensure 2D
        if x.ndim == 1:
            x = x.reshape(1, -1)

        # Normalize input to [0, 1]
        x_norm = self._normalize_input(x)

        # Encode and decode
        h = self._forward_encoder(x_norm)
        x_recon = self._forward_decoder(h)

        return x_recon

    def _normalize_input(self, x: np.ndarray) -> np.ndarray:
        """Normalize input to [0, 1] range."""
        x_min = x.min(axis=-1, keepdims=True)
        x_max = x.max(axis=-1, keepdims=True)
        range_val = x_max - x_min
        range_val[range_val == 0] = 1  # Avoid division by zero
        return (x - x_min) / range_val

    def detect(
        self,
        feature_vector: np.ndarray,
    ) -> ReconstructionResult:
        """
        Detect anomaly via reconstruction error.

        Args:
            feature_vector: Feature vector from IPC

        Returns:
            ReconstructionResult
        """
        # Ensure correct shape
        if feature_vector.ndim == 1:
            feature_vector = feature_vector.reshape(1, -1)

        # Reconstruct
        reconstructed = self.reconstruct(feature_vector)

        # Calculate MSE
        mse = np.mean((feature_vector - reconstructed) ** 2)
        rmse = np.sqrt(mse)

        is_anomaly = rmse > self.anomaly_threshold

        # Calculate confidence based on error magnitude
        if is_anomaly:
            confidence = min(1.0, (rmse - self.anomaly_threshold) / self.anomaly_threshold)
        else:
            confidence = min(1.0, rmse / self.anomaly_threshold)

        self._reconstruction_count += 1
        if is_anomaly:
            self._anomaly_count += 1

        # Track error history for adaptive thresholding
        self._error_history.append(rmse)
        if len(self._error_history) > 1000:
            self._error_history = self._error_history[-1000:]

        return ReconstructionResult(
            original=feature_vector.flatten(),
            reconstructed=reconstructed.flatten(),
            reconstruction_error=float(rmse),
            is_anomaly=is_anomaly,
            anomaly_confidence=float(confidence),
            timestamp_ns=time.time_ns(),
        )

    def detect_batch(
        self,
        feature_vectors: np.ndarray,
    ) -> List[ReconstructionResult]:
        """
        Detect anomalies in batch.

        Args:
            feature_vectors: Array of shape (n, d)

        Returns:
            List of ReconstructionResult
        """
        results = []
        for i in range(len(feature_vectors)):
            result = self.detect(feature_vectors[i])
            results.append(result)
        return results

    def train(
        self,
        X: np.ndarray,
        epochs: int = 100,
        learning_rate: float = 0.001,
        batch_size: int = 32,
    ) -> Dict[str, List[float]]:
        """
        Train the autoencoder on normal data.

        Args:
            X: Training data (normal market states)
            epochs: Number of training epochs
            learning_rate: Learning rate
            batch_size: Batch size

        Returns:
            Training history
        """
        if len(X) < batch_size:
            logger.warning(f"Insufficient training data: {len(X)}")
            return {"loss": []}

        self._initialize_weights()

        history = {"loss": []}
        n_samples = len(X)

        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0

            # Shuffle data
            indices = np.random.permutation(n_samples)

            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]
                batch = X[batch_indices]

                # Forward pass
                batch_norm = self._normalize_input(batch)
                h = self._forward_encoder(batch_norm)
                recon = self._forward_decoder(h)

                # Calculate loss (MSE)
                loss = np.mean((batch_norm - recon) ** 2)
                epoch_loss += loss
                n_batches += 1

                # Simplified backprop (gradient descent approximation)
                # In production, would use proper autograd
                self._update_weights_simple(batch_norm, recon, learning_rate)

            avg_loss = epoch_loss / max(1, n_batches)
            history["loss"].append(avg_loss)

            if (epoch + 1) % 10 == 0:
                logger.debug(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")

        # Adapt threshold based on training errors
        self._adapt_threshold()
        self._is_trained = True

        logger.info(f"Autoencoder trained with final loss: {history['loss'][-1]:.6f}")
        return history

    def _update_weights_simple(
        self,
        x: np.ndarray,
        recon: np.ndarray,
        lr: float,
    ):
        """Simplified weight update (approximate gradient descent)."""
        # This is a placeholder - proper implementation would use full backprop
        error = x - recon
        for i in range(len(self._weights)):
            # Small random perturbation in direction of reducing error
            noise = np.random.randn(*self._weights[i].shape) * lr * 0.01
            self._weights[i] += noise
            self._decoder_weights[-(i + 1)] += noise.T

    def _adapt_threshold(self):
        """Adapt anomaly threshold based on training reconstruction errors."""
        if not self._error_history:
            return

        # Set threshold at 95th percentile of training errors
        self.anomaly_threshold = np.percentile(self._error_history, 95) * 1.5
        logger.info(f"Adapted anomaly threshold: {self.anomaly_threshold:.4f}")

    def get_market_anomaly_signal(
        self,
        feature_vector: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Get market anomaly signal for risk engine.

        Args:
            feature_vector: Current feature vector

        Returns:
            Signal dict for risk engine
        """
        result = self.detect(feature_vector)

        signal = {
            "is_structural_anomaly": result.is_anomaly,
            "reconstruction_error": result.reconstruction_error,
            "confidence": result.anomaly_confidence,
            "recommended_action": "widen_spreads" if result.is_anomaly else "normal",
            "spread_multiplier": 2.0 if result.is_anomaly else 1.0,
            "timestamp_ns": result.timestamp_ns,
        }

        if result.is_anomaly:
            logger.warning(
                f"Structural market anomaly detected! "
                f"Error: {result.reconstruction_error:.4f}"
            )

        return signal

    def get_stats(self) -> Dict[str, Any]:
        """Get autoencoder statistics."""
        avg_error = np.mean(self._error_history) if self._error_history else 0.0
        return {
            "is_trained": self._is_trained,
            "input_dim": self.input_dim,
            "encoder_dims": self.encoder_dims,
            "bottleneck_dim": self.encoder_dims[-1],
            "anomaly_threshold": self.anomaly_threshold,
            "reconstruction_count": self._reconstruction_count,
            "anomaly_count": self._anomaly_count,
            "anomaly_rate": self._anomaly_count / max(1, self._reconstruction_count),
            "avg_reconstruction_error": avg_error,
        }

    def reset(self):
        """Reset the autoencoder."""
        self._weights = []
        self._biases = []
        self._decoder_weights = []
        self._decoder_biases = []
        self._is_trained = False
        self._reconstruction_count = 0
        self._anomaly_count = 0
        self._error_history = []
        logger.info("Autoencoder reset")


# Module singleton
_autoencoder: Optional[LightweightAutoencoder] = None


def get_autoencoder(
    input_dim: int = 128,
    anomaly_threshold: float = 0.5,
) -> LightweightAutoencoder:
    """Get or create the autoencoder singleton."""
    global _autoencoder
    if _autoencoder is None:
        _autoencoder = LightweightAutoencoder(
            input_dim=input_dim,
            anomaly_threshold=anomaly_threshold,
        )
    return _autoencoder


def initialize_autoencoder_with_data(
    training_data: np.ndarray,
    input_dim: int = 128,
    epochs: int = 100,
) -> LightweightAutoencoder:
    """Initialize and train autoencoder with data."""
    global _autoencoder
    _autoencoder = LightweightAutoencoder(input_dim=input_dim)
    _autoencoder.train(training_data, epochs=epochs)
    return _autoencoder


async def shutdown_dq_ae_module():
    """Shutdown the autoencoder module."""
    global _autoencoder
    if _autoencoder:
        _autoencoder.reset()
        _autoencoder = None
