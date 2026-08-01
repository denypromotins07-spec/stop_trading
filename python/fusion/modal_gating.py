"""
Dynamic Modal Gating Network for HFT
Builds a dynamic gating network that weights the importance of technical, on-chain, 
and sentiment features based on the active HMM regime.

Suppresses noisy modalities during high-volatility flash crashes to prevent the 
ML ensemble from hallucinating false breakout signals.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from enum import Enum


class MarketRegime(Enum):
    """Market regime classification."""
    LOW_VOL = 0
    NORMAL = 1
    HIGH_VOL = 2
    FLASH_CRASH = 3
    TRENDING_UP = 4
    TRENDING_DOWN = 5


class ModalGatingNetwork:
    """
    Dynamic gating network for multi-modal feature weighting.
    Adjusts feature importance based on detected market regime.
    """
    
    # Pre-defined gating weights for each regime
    # Format: [technical_weight, onchain_weight, sentiment_weight]
    REGIME_WEIGHTS = {
        MarketRegime.LOW_VOL: np.array([0.5, 0.3, 0.2], dtype=np.float32),
        MarketRegime.NORMAL: np.array([0.4, 0.35, 0.25], dtype=np.float32),
        MarketRegime.HIGH_VOL: np.array([0.6, 0.3, 0.1], dtype=np.float32),
        MarketRegime.FLASH_CRASH: np.array([0.8, 0.15, 0.05], dtype=np.float32),  # Suppress sentiment
        MarketRegime.TRENDING_UP: np.array([0.45, 0.35, 0.2], dtype=np.float32),
        MarketRegime.TRENDING_DOWN: np.array([0.45, 0.35, 0.2], dtype=np.float32),
    }
    
    def __init__(self, 
                 num_modalities: int = 3,
                 hidden_dim: int = 64,
                 temperature: float = 1.0):
        """
        Initialize modal gating network.
        
        Args:
            num_modalities: Number of input modalities (technical, onchain, sentiment)
            hidden_dim: Hidden layer dimension for gating MLP
            temperature: Softmax temperature for weight smoothing
        """
        self.num_modalities = num_modalities
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        
        # Initialize gating MLP weights (lightweight)
        rng = np.random.default_rng(seed=42)
        self.W1 = rng.normal(0, 0.1, (num_modalities * 16, hidden_dim)).astype(np.float32)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = rng.normal(0, 0.1, (hidden_dim, num_modalities)).astype(np.float32)
        self.b2 = np.zeros(num_modalities, dtype=np.float32)
        
        # Regime detection thresholds
        self.vol_thresholds = {
            'low': 0.001,
            'normal': 0.005,
            'high': 0.02,
            'flash': 0.05
        }
        
        # Momentum thresholds for trend detection
        self.momentum_threshold = 0.01
        
        # Cache for last regime
        self._last_regime = MarketRegime.NORMAL
        self._regime_confidence = 0.0
        
    def detect_regime(self, 
                      volatility: float,
                      momentum: float,
                      volume_spike: float) -> MarketRegime:
        """
        Detect current market regime from market metrics.
        
        Args:
            volatility: Realized volatility (e.g., 5-min std dev)
            momentum: Price momentum indicator
            volume_spike: Volume relative to rolling average
            
        Returns:
            Detected MarketRegime
        """
        # Flash crash detection: extreme volatility + negative momentum
        if volatility > self.vol_thresholds['flash'] and momentum < -self.momentum_threshold:
            self._last_regime = MarketRegime.FLASH_CRASH
            self._regime_confidence = min(1.0, volatility / 0.1)
            return MarketRegime.FLASH_CRASH
        
        # High volatility
        if volatility > self.vol_thresholds['high']:
            if momentum > self.momentum_threshold:
                self._last_regime = MarketRegime.TRENDING_UP
            elif momentum < -self.momentum_threshold:
                self._last_regime = MarketRegime.TRENDING_DOWN
            else:
                self._last_regime = MarketRegime.HIGH_VOL
            self._regime_confidence = min(1.0, volatility / 0.05)
            return self._last_regime
        
        # Low volatility
        if volatility < self.vol_thresholds['low']:
            self._last_regime = MarketRegime.LOW_VOL
            self._regime_confidence = 1.0 - (volatility / self.vol_thresholds['low'])
            return MarketRegime.LOW_VOL
        
        # Normal regime
        self._last_regime = MarketRegime.NORMAL
        self._regime_confidence = 0.8
        return MarketRegime.NORMAL
    
    def compute_gating_weights(self, 
                               regime: Optional[MarketRegime] = None,
                               regime_features: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute gating weights for each modality.
        
        Args:
            regime: Explicit regime (if None, uses detected regime)
            regime_features: Features for neural gating [volatility, momentum, volume_spike, ...]
            
        Returns:
            Normalized gating weights [num_modalities]
        """
        if regime is None:
            regime = self._last_regime
        
        # Get base weights for regime
        base_weights = self.REGIME_WEIGHTS[regime].copy()
        
        # Apply neural refinement if features provided
        if regime_features is not None:
            refined_weights = self._neural_gate(regime_features)
            # Blend base and refined weights
            alpha = self._regime_confidence
            base_weights = (1 - alpha) * base_weights + alpha * refined_weights
        
        # Apply temperature scaling
        logits = np.log(base_weights + 1e-8) / self.temperature
        weights = self._softmax(logits)
        
        return weights
    
    def _neural_gate(self, features: np.ndarray) -> np.ndarray:
        """
        Lightweight neural network for fine-grained gating.
        
        Args:
            features: Input features [batch, num_modalities * 16]
            
        Returns:
            Refined gating weights
        """
        # Ensure 2D
        if features.ndim == 1:
            features = features[np.newaxis, :]
        
        # Forward pass through gating MLP
        hidden = np.maximum(0, features @ self.W1 + self.b1)  # ReLU
        logits = hidden @ self.W2 + self.b2
        
        # Softmax to get weights
        weights = self._softmax(logits / self.temperature)
        
        return weights[0] if features.shape[0] == 1 else weights
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / (exp_x.sum(axis=-1, keepdims=True) + 1e-8)
    
    def gate_features(self,
                      technical_features: np.ndarray,
                      onchain_features: np.ndarray,
                      sentiment_features: np.ndarray,
                      volatility: float,
                      momentum: float,
                      volume_spike: float) -> np.ndarray:
        """
        Apply gating to multi-modal features.
        
        Args:
            technical_features: Technical indicators [seq_len, tech_dim]
            onchain_features: On-chain flow features [seq_len, chain_dim]
            sentiment_features: Sentiment scores [seq_len, sent_dim]
            volatility: Current volatility
            momentum: Current momentum
            volume_spike: Volume spike ratio
            
        Returns:
            Weighted combined features [seq_len, combined_dim]
        """
        # Detect regime and compute gating weights
        regime = self.detect_regime(volatility, momentum, volume_spike)
        weights = self.compute_gating_weights(regime)
        
        # Ensure features are aligned in sequence length
        seq_len = technical_features.shape[0]
        
        # Normalize each modality to same dimension for combination
        tech_norm = self._normalize_features(technical_features)
        chain_norm = self._normalize_features(onchain_features)
        sent_norm = self._normalize_features(sentiment_features)
        
        # Apply gating weights
        gated = (
            weights[0] * tech_norm +
            weights[1] * chain_norm +
            weights[2] * sent_norm
        )
        
        return gated, weights, regime
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features to common dimension."""
        # Simple L2 normalization per timestep
        norm = np.linalg.norm(features, axis=-1, keepdims=True) + 1e-8
        return features / norm
    
    def suppress_noisy_modality(self, 
                                modality_idx: int,
                                suppression_factor: float = 0.1) -> np.ndarray:
        """
        Dynamically suppress a specific modality.
        
        Args:
            modality_idx: Index of modality to suppress (0=tech, 1=chain, 2=sent)
            suppression_factor: Factor to reduce weight by
            
        Returns:
            Updated gating weights
        """
        weights = self.compute_gating_weights()
        weights[modality_idx] *= suppression_factor
        weights = weights / weights.sum()  # Re-normalize
        return weights


class AdaptiveFusionController:
    """
    High-level controller for adaptive multi-modal fusion.
    Manages regime transitions and gating adjustments.
    """
    
    def __init__(self, 
                 num_modalities: int = 3,
                 adaptation_window: int = 100):
        self.gating_network = ModalGatingNetwork(num_modalities=num_modalities)
        self.adaptation_window = adaptation_window
        
        # History for regime stability checking
        self._regime_history = []
        self._weight_history = []
        
    def process_tick(self,
                     technical_features: np.ndarray,
                     onchain_features: np.ndarray,
                     sentiment_features: np.ndarray,
                     market_metrics: Dict[str, float]) -> Tuple[np.ndarray, Dict]:
        """
        Process a single tick through adaptive fusion.
        
        Args:
            technical_features: Technical indicators
            onchain_features: On-chain features
            sentiment_features: Sentiment features
            market_metrics: Dictionary with volatility, momentum, volume_spike
            
        Returns:
            Fused features and metadata dict
        """
        volatility = market_metrics.get('volatility', 0.005)
        momentum = market_metrics.get('momentum', 0.0)
        volume_spike = market_metrics.get('volume_spike', 1.0)
        
        # Apply gating
        fused, weights, regime = self.gating_network.gate_features(
            technical_features,
            onchain_features,
            sentiment_features,
            volatility,
            momentum,
            volume_spike
        )
        
        # Track history
        self._regime_history.append(regime)
        self._weight_history.append(weights.copy())
        
        # Maintain bounded history
        if len(self._regime_history) > self.adaptation_window:
            self._regime_history.pop(0)
            self._weight_history.pop(0)
        
        # Check for regime instability
        instability = self._check_regime_stability()
        
        metadata = {
            'regime': regime.value,
            'gating_weights': weights.tolist(),
            'instability_score': instability,
            'suppressed_modalities': self._get_suppressed_modalities(weights)
        }
        
        return fused, metadata
    
    def _check_regime_stability(self) -> float:
        """Check if regime has been stable recently."""
        if len(self._regime_history) < 10:
            return 0.0
        
        recent_regimes = self._regime_history[-10:]
        unique_regimes = len(set(recent_regimes))
        
        # Higher value = more instability
        return (unique_regimes - 1) / 5.0
    
    def _get_suppressed_modalities(self, weights: np.ndarray) -> List[str]:
        """Identify which modalities are suppressed."""
        modalities = ['technical', 'onchain', 'sentiment']
        suppressed = []
        threshold = 0.15
        
        for i, w in enumerate(weights):
            if w < threshold:
                suppressed.append(modalities[i])
        
        return suppressed
    
    def force_regime(self, regime: MarketRegime) -> np.ndarray:
        """Force a specific regime (for testing or manual override)."""
        self._last_regime = regime
        return self.gating_network.compute_gating_weights(regime)


# Module exports
__all__ = ['ModalGatingNetwork', 'AdaptiveFusionController', 'MarketRegime']
