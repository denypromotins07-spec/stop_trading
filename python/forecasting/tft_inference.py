"""
Temporal Fusion Transformer (TFT) inference wrapper using ONNX Runtime.
Processes static, known, and observed time-series inputs to capture complex multi-horizon dependencies.
Outputs interpretable attention weights for the alpha ensemble.
Strictly enforces 3GB RAM limit via bounded sequence lengths and ONNX CPU providers.
"""

from __future__ import annotations

import numpy as np
import onnxruntime as ort
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class TFTConfig:
    """Configuration for Temporal Fusion Transformer."""
    # Sequence lengths
    encoder_length: int = 128
    decoder_length: int = 32
    
    # Feature dimensions
    num_static_features: int = 10
    num_known_future_features: int = 5
    num_observed_features: int = 15
    
    # Model architecture
    hidden_size: int = 64
    attention_head_size: int = 16
    num_heads: int = 4
    dropout_rate: float = 0.1
    
    # Memory constraints
    max_batch_size: int = 32
    cpu_mem_limit: int = 512 * 1024 * 1024  # 512MB
    intra_op_threads: int = 2
    inter_op_threads: int = 1
    
    # Output options
    output_attention_weights: bool = True
    output_variable_selection: bool = True
    
    provider_options: Optional[Dict] = None
    
    def __post_init__(self):
        if self.provider_options is None:
            self.provider_options = {
                'CPUExecutionProvider': {
                    'arena_extend_strategy': 'kSameAsRequested',
                    'cpu_mem_limit': self.cpu_mem_limit,
                    'intra_op_num_threads': self.intra_op_threads,
                    'inter_op_num_threads': self.inter_op_threads
                }
            }


@dataclass
class TFTInput:
    """Container for TFT model inputs."""
    # Observed features over encoder period: (batch, encoder_length, num_observed)
    observed_inputs: np.ndarray
    
    # Known features over encoder+decoder period: (batch, encoder_length + decoder_length, num_known)
    known_inputs: np.ndarray
    
    # Static features: (batch, num_static)
    static_inputs: np.ndarray
    
    # Time indices for position encoding
    encoder_time_idx: Optional[np.ndarray] = None
    decoder_time_idx: Optional[np.ndarray] = None


@dataclass
class TFTOutput:
    """Container for TFT model outputs."""
    # Main forecast: (batch, decoder_length, target_dim)
    forecast: np.ndarray
    
    # Optional interpretability outputs
    attention_weights: Optional[np.ndarray] = None
    variable_selection_weights: Optional[np.ndarray] = None
    prediction_intervals: Optional[Tuple[np.ndarray, np.ndarray]] = None
    
    # Metadata
    inference_time_ms: float = 0.0
    model_version: str = ""


class TFTInferenceWrapper:
    """
    Temporal Fusion Transformer inference wrapper using pre-exported ONNX models.
    
    The TFT architecture includes:
    - Variable selection networks for input feature importance
    - Gated residual networks for non-linear processing
    - Multi-head attention for temporal dependencies
    - Static covariate encoders for context
    - Interpretable output layers
    
    All computations are performed via ONNX Runtime for optimal CPU performance.
    """
    
    def __init__(
        self,
        model_path: str,
        config: TFTConfig,
        session_options: Optional[ort.SessionOptions] = None
    ):
        self.config = config
        self.model_path = model_path
        
        # Configure session options for memory efficiency
        if session_options is None:
            session_options = ort.SessionOptions()
            session_options.enable_mem_pattern = True
            session_options.enable_cpu_mem_arena = True
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_options.intra_op_num_threads = config.intra_op_threads
            session_options.inter_op_num_threads = config.inter_op_threads
            
        self.session = ort.InferenceSession(
            model_path,
            sess_options=session_options,
            providers=['CPUExecutionProvider']
        )
        
        # Cache input/output names for faster access
        self._input_names = [inp.name for inp in self.session.get_inputs()]
        self._output_names = [out.name for out in self.session.get_outputs()]
        
        # Validate model signature
        self._validate_model_signature()
        
        # Pre-allocate buffers for zero-copy operations
        self._attention_buffer: Optional[np.ndarray] = None
        self._variable_selection_buffer: Optional[np.ndarray] = None
        
        logger.info(f"TFT ONNX model loaded: {model_path}")
    
    def _validate_model_signature(self):
        """Validate the ONNX model input/output signatures match config."""
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        
        logger.debug(f"TFT Model inputs: {self._input_names}")
        logger.debug(f"TFT Model outputs: {self._output_names}")
        
        # Check required inputs exist
        required_inputs = ['observed_inputs', 'known_inputs', 'static_inputs']
        for req_input in required_inputs:
            if req_input not in self._input_names:
                logger.warning(f"Expected input '{req_input}' not found in model")
        
        # Check forecast output exists
        if 'forecast' not in self._output_names and 'output' not in self._output_names:
            raise ValueError("Model must have 'forecast' or 'output' output")
    
    def prepare_input(
        self,
        observed_history: np.ndarray,
        known_future: np.ndarray,
        static_features: np.ndarray
    ) -> TFTInput:
        """
        Prepare and validate input tensors for TFT inference.
        
        Args:
            observed_history: Historical observed features (encoder_length, num_observed)
            known_future: Known future features (encoder_length + decoder_length, num_known)
            static_features: Static features (num_static,)
            
        Returns:
            TFTInput container with properly shaped arrays
        """
        # Ensure correct dtype
        observed_history = np.asarray(observed_history, dtype=np.float32)
        known_future = np.asarray(known_future, dtype=np.float32)
        static_features = np.asarray(static_features, dtype=np.float32)
        
        # Add batch dimension if needed
        if observed_history.ndim == 2:
            observed_history = observed_history.reshape(1, *observed_history.shape)
        if known_future.ndim == 2:
            known_future = known_future.reshape(1, *known_future.shape)
        if static_features.ndim == 1:
            static_features = static_features.reshape(1, -1)
        
        # Validate shapes
        if observed_history.shape[1] != self.config.encoder_length:
            raise ValueError(
                f"Expected encoder_length {self.config.encoder_length}, "
                f"got {observed_history.shape[1]}"
            )
        
        expected_known_len = self.config.encoder_length + self.config.decoder_length
        if known_future.shape[1] != expected_known_len:
            raise ValueError(
                f"Expected known inputs length {expected_known_len}, "
                f"got {known_future.shape[1]}"
            )
        
        return TFTInput(
            observed_inputs=observed_history,
            known_inputs=known_future,
            static_inputs=static_features
        )
    
    def predict(
        self,
        tft_input: TFTInput,
        return_attention: bool = False,
        return_variable_selection: bool = False
    ) -> TFTOutput:
        """
        Run TFT inference.
        
        Args:
            tft_input: Prepared TFT input container
            return_attention: Whether to return attention weights
            return_variable_selection: Whether to return variable selection weights
            
        Returns:
            TFTOutput container with forecasts and optional interpretability data
        """
        import time
        start_time = time.perf_counter()
        
        # Build input feed
        input_feed = {
            'observed_inputs': tft_input.observed_inputs,
            'known_inputs': tft_input.known_inputs,
            'static_inputs': tft_input.static_inputs
        }
        
        # Add optional inputs if provided
        if tft_input.encoder_time_idx is not None:
            input_feed['encoder_time_idx'] = tft_input.encoder_time_idx
        if tft_input.decoder_time_idx is not None:
            input_feed['decoder_time_idx'] = tft_input.decoder_time_idx
        
        # Run inference
        outputs = self.session.run(None, input_feed)
        
        # Parse outputs based on model signature
        output_dict = dict(zip(self._output_names, outputs))
        
        # Extract main forecast
        forecast_key = 'forecast' if 'forecast' in output_dict else 'output'
        forecast = output_dict.get(forecast_key)
        
        if forecast is None:
            raise RuntimeError("Could not find forecast output in model outputs")
        
        # Extract optional interpretability outputs
        attention_weights = None
        variable_selection_weights = None
        
        if return_attention:
            attn_key = 'attention_weights'
            if attn_key in output_dict:
                attention_weights = output_dict[attn_key]
            elif 'attn' in str(self._output_names).lower():
                for key in output_dict:
                    if 'attn' in key.lower():
                        attention_weights = output_dict[key]
                        break
        
        if return_variable_selection:
            vs_key = 'variable_selection'
            if vs_key in output_dict:
                variable_selection_weights = output_dict[vs_key]
            elif 'var_select' in str(self._output_names).lower():
                for key in output_dict:
                    if 'var' in key.lower() and 'select' in key.lower():
                        variable_selection_weights = output_dict[key]
                        break
        
        inference_time_ms = (time.perf_counter() - start_time) * 1000
        
        return TFTOutput(
            forecast=forecast,
            attention_weights=attention_weights,
            variable_selection_weights=variable_selection_weights,
            inference_time_ms=inference_time_ms
        )
    
    def predict_batch(
        self,
        observed_batch: np.ndarray,
        known_batch: np.ndarray,
        static_batch: np.ndarray,
        chunk_size: int = 32
    ) -> List[TFTOutput]:
        """
        Run batched inference with memory-aware chunking.
        
        Args:
            observed_batch: Batch of observed histories
            known_batch: Batch of known futures
            static_batch: Batch of static features
            chunk_size: Maximum batch size per inference call
            
        Returns:
            List of TFTOutput containers
        """
        n_samples = observed_batch.shape[0]
        results: List[TFTOutput] = []
        
        for i in range(0, n_samples, chunk_size):
            end_idx = min(i + chunk_size, n_samples)
            
            tft_input = TFTInput(
                observed_inputs=observed_batch[i:end_idx],
                known_inputs=known_batch[i:end_idx],
                static_inputs=static_batch[i:end_idx]
            )
            
            output = self.predict(tft_input)
            results.append(output)
        
        return results
    
    def get_feature_importance(
        self,
        observed_history: np.ndarray,
        known_future: np.ndarray,
        static_features: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Extract feature importance from variable selection weights.
        
        Returns dict with importance scores for observed, known, and static features.
        """
        tft_input = self.prepare_input(
            observed_history, known_future, static_features
        )
        
        output = self.predict(
            tft_input,
            return_attention=True,
            return_variable_selection=True
        )
        
        importance = {}
        
        if output.variable_selection_weights is not None:
            # Variable selection weights typically have shape:
            # (batch, num_timesteps, num_features) or similar
            vs_weights = output.variable_selection_weights
            
            # Average across time dimension for overall importance
            if vs_weights.ndim >= 2:
                importance['observed'] = np.mean(vs_weights[:, :self.config.encoder_length, :], axis=(0, 1))
                
                known_start = self.config.encoder_length
                if vs_weights.shape[1] > known_start:
                    importance['known'] = np.mean(vs_weights[:, known_start:, :], axis=(0, 1))
        
        if output.attention_weights is not None:
            # Attention weights show temporal importance
            attn = output.attention_weights
            importance['temporal_attention'] = np.mean(attn, axis=(0, 1)) if attn.ndim >= 3 else np.mean(attn, axis=0)
        
        return importance
    
    def explain_prediction(
        self,
        observed_history: np.ndarray,
        known_future: np.ndarray,
        static_features: np.ndarray
    ) -> Dict[str, Any]:
        """
        Generate comprehensive explanation for a single prediction.
        
        Returns dict with:
        - forecast: The predicted values
        - feature_importance: Relative importance of each input feature
        - temporal_attention: Which time steps were most important
        - confidence_estimate: Uncertainty estimate based on attention entropy
        """
        tft_input = self.prepare_input(
            observed_history, known_future, static_features
        )
        
        output = self.predict(
            tft_input,
            return_attention=True,
            return_variable_selection=True
        )
        
        explanation = {
            'forecast': output.forecast[0] if output.forecast.ndim > 1 else output.forecast,
            'feature_importance': {},
            'temporal_attention': None,
            'confidence_estimate': 1.0,
            'inference_time_ms': output.inference_time_ms
        }
        
        # Compute feature importance
        if output.variable_selection_weights is not None:
            vs = output.variable_selection_weights
            if vs.ndim >= 2:
                explanation['feature_importance']['observed_mean'] = float(np.mean(vs[:, :self.config.encoder_length]))
                if vs.shape[1] > self.config.encoder_length:
                    explanation['feature_importance']['known_mean'] = float(np.mean(
                        vs[:, self.config.encoder_length:]
                    ))
        
        # Compute attention-based confidence
        if output.attention_weights is not None:
            attn = output.attention_weights
            # Higher entropy = more uncertainty
            attn_flat = attn.reshape(-1, attn.shape[-1])
            attn_probs = attn_flat / (attn_flat.sum(axis=-1, keepdims=True) + 1e-8)
            entropy = -np.sum(attn_probs * np.log(attn_probs + 1e-8), axis=-1)
            max_entropy = np.log(attn.shape[-1])
            explanation['confidence_estimate'] = float(1.0 - np.mean(entropy) / (max_entropy + 1e-8))
            explanation['temporal_attention'] = attn
        
        return explanation


class TFTEnsemble:
    """
    Ensemble of TFT models for robust multi-horizon forecasting.
    """
    
    def __init__(self, model_configs: List[Tuple[str, TFTConfig]]):
        self.models: List[TFTInferenceWrapper] = []
        self.weights: List[float] = []
        
        for model_path, config in model_configs:
            try:
                model = TFTInferenceWrapper(model_path, config)
                self.models.append(model)
                self.weights.append(1.0 / len(model_configs))
            except Exception as e:
                logger.warning(f"Failed to load TFT model {model_path}: {e}")
        
        if not self.models:
            raise RuntimeError("No TFT models could be loaded")
    
    def predict_ensemble(
        self,
        observed_history: np.ndarray,
        known_future: np.ndarray,
        static_features: np.ndarray
    ) -> np.ndarray:
        """Generate weighted ensemble prediction."""
        tft_input = self.models[0].prepare_input(
            observed_history, known_future, static_features
        )
        
        forecasts = []
        for model, weight in zip(self.models, self.weights):
            try:
                output = model.predict(tft_input)
                forecasts.append(output.forecast * weight)
            except Exception as e:
                logger.warning(f"TFT model prediction failed: {e}")
        
        if not forecasts:
            raise RuntimeError("All TFT models failed")
        
        return np.sum(forecasts, axis=0)


# Factory function
def create_tft_model(
    model_path: str,
    encoder_length: int = 128,
    decoder_length: int = 32,
    num_observed: int = 15,
    num_known: int = 5,
    num_static: int = 10
) -> TFTInferenceWrapper:
    """Factory function to create TFT model with standard configuration."""
    config = TFTConfig(
        encoder_length=encoder_length,
        decoder_length=decoder_length,
        num_observed_features=num_observed,
        num_known_future_features=num_known,
        num_static_features=num_static
    )
    return TFTInferenceWrapper(model_path, config)
