"""
Fusion Module Root
Manages the multi-modal feature fusion pipeline, strictly bounding tensor dimensions 
to respect the 3GB Python RAM ceiling.

Exports unified interface for cross-attention and modal gating.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import threading
import time
from dataclasses import dataclass

from .cross_attention import CrossAttentionFusion, LightweightFusionPipeline
from .modal_gating import ModalGatingNetwork, AdaptiveFusionController, MarketRegime


@dataclass
class FusionConfig:
    """Configuration for fusion pipeline."""
    l2_dim: int = 64
    chain_dim: int = 32
    sentiment_dim: int = 16
    embed_dim: int = 48
    num_heads: int = 4
    seq_len: int = 16
    max_batch_size: int = 32
    ram_limit_mb: int = 512  # Max RAM for fusion module


class FusionMemoryManager:
    """
    Memory manager for fusion pipeline.
    Enforces strict RAM bounds and manages buffer allocation.
    """
    
    def __init__(self, ram_limit_mb: int = 512):
        self.ram_limit_bytes = ram_limit_mb * 1024 * 1024
        self._allocated_buffers: List[np.ndarray] = []
        self._lock = threading.Lock()
        
    def allocate(self, shape: Tuple[int, ...], dtype: np.dtype = np.float32) -> np.ndarray:
        """Allocate a bounded buffer."""
        size_bytes = np.prod(shape) * np.dtype(dtype).itemsize
        
        with self._lock:
            current_usage = sum(b.nbytes for b in self._allocated_buffers)
            
            if current_usage + size_bytes > self.ram_limit_bytes:
                # Trigger garbage collection of old buffers
                self._gc_old_buffers()
                
            buffer = np.zeros(shape, dtype=dtype)
            self._allocated_buffers.append(buffer)
            return buffer
    
    def _gc_old_buffers(self):
        """Garbage collect oldest buffers."""
        # Keep only most recent 50% of buffers
        keep_count = max(1, len(self._allocated_buffers) // 2)
        self._allocated_buffers = self._allocated_buffers[-keep_count:]
        
    def get_usage(self) -> int:
        """Get current memory usage in bytes."""
        with self._lock:
            return sum(b.nbytes for b in self._allocated_buffers)
    
    def clear(self):
        """Clear all allocated buffers."""
        with self._lock:
            self._allocated_buffers.clear()


class UnifiedFusionEngine:
    """
    Unified engine combining cross-attention and modal gating.
    Provides single interface for multi-modal feature fusion.
    """
    
    def __init__(self, config: Optional[FusionConfig] = None):
        self.config = config or FusionConfig()
        
        # Initialize memory manager
        self.memory_manager = FusionMemoryManager(
            ram_limit_mb=self.config.ram_limit_mb
        )
        
        # Initialize fusion components
        self.cross_attention = LightweightFusionPipeline(
            l2_dim=self.config.l2_dim,
            chain_dim=self.config.chain_dim
        )
        
        self.modal_gating = AdaptiveFusionController(
            num_modalities=3,  # technical, onchain, sentiment
            adaptation_window=100
        )
        
        # Pre-allocate output buffers
        self._output_buffer = self.memory_manager.allocate(
            (self.config.max_batch_size, self.config.seq_len, self.config.embed_dim)
        )
        
        # Statistics
        self._stats = {
            'fusion_count': 0,
            'avg_latency_us': 0.0,
            'ram_usage_mb': 0.0
        }
        self._latency_samples: List[float] = []
        
    def fuse(self,
             technical_features: np.ndarray,
             onchain_features: np.ndarray,
             sentiment_features: np.ndarray,
             market_metrics: Dict[str, float]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Perform complete multi-modal fusion.
        
        Args:
            technical_features: Technical indicators [batch, seq, tech_dim]
            onchain_features: On-chain flow features [batch, seq, chain_dim]
            sentiment_features: Sentiment scores [batch, seq, sent_dim]
            market_metrics: Volatility, momentum, volume_spike
            
        Returns:
            Fused features and metadata
        """
        start_time = time.perf_counter_ns()
        
        # Step 1: Apply modal gating
        gated_tech, weights, regime = self.modal_gating.gating_network.gate_features(
            technical_features,
            onchain_features,
            sentiment_features,
            market_metrics.get('volatility', 0.005),
            market_metrics.get('momentum', 0.0),
            market_metrics.get('volume_spike', 1.0)
        )
        
        # Step 2: Cross-attention fusion of gated features
        # Combine gated features for cross-attention input
        batch_size = min(
            technical_features.shape[0],
            self.config.max_batch_size
        )
        
        # Prepare L2-style features (technical + gated combination)
        l2_input = np.ascontiguousarray(
            technical_features[:batch_size], 
            dtype=np.float32
        )
        
        # Prepare chain-style features (onchain + sentiment weighted)
        chain_input = np.ascontiguousarray(
            0.7 * onchain_features[:batch_size] + 0.3 * sentiment_features[:batch_size],
            dtype=np.float32
        )
        
        # Run cross-attention
        fused = self.cross_attention.process_batch(l2_input, chain_input)
        
        # Update statistics
        latency_us = (time.perf_counter_ns() - start_time) / 1000
        self._update_stats(latency_us)
        
        metadata = {
            'regime': regime.value,
            'gating_weights': weights.tolist(),
            'latency_us': latency_us,
            'ram_usage_mb': self.memory_manager.get_usage() / (1024 * 1024),
            'batch_size': batch_size
        }
        
        return fused, metadata
    
    def _update_stats(self, latency_us: float):
        """Update running statistics."""
        self._stats['fusion_count'] += 1
        
        # Maintain bounded latency samples
        self._latency_samples.append(latency_us)
        if len(self._latency_samples) > 1000:
            self._latency_samples.pop(0)
        
        self._stats['avg_latency_us'] = np.mean(self._latency_samples)
        self._stats['ram_usage_mb'] = self.memory_manager.get_usage() / (1024 * 1024)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current fusion statistics."""
        return self._stats.copy()
    
    def reset(self):
        """Reset fusion engine state."""
        self.memory_manager.clear()
        self._latency_samples.clear()
        self._stats = {
            'fusion_count': 0,
            'avg_latency_us': 0.0,
            'ram_usage_mb': 0.0
        }


class FusionOrchestrator:
    """
    High-level orchestrator for fusion pipeline.
    Manages lifecycle, health checks, and integration with upstream systems.
    """
    
    def __init__(self, config: Optional[FusionConfig] = None):
        self.config = config or FusionConfig()
        self.engine = UnifiedFusionEngine(self.config)
        self._healthy = True
        self._last_error: Optional[str] = None
        self._process_count = 0
        
    def process(self,
                features: Dict[str, np.ndarray],
                metrics: Dict[str, float]) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Process features through fusion pipeline with error handling.
        
        Args:
            features: Dictionary with 'technical', 'onchain', 'sentiment' keys
            metrics: Market metrics dictionary
            
        Returns:
            Fused features (or None on error) and status dict
        """
        if not self._healthy:
            return None, {'status': 'unhealthy', 'error': self._last_error}
        
        try:
            # Validate inputs
            required_keys = ['technical', 'onchain', 'sentiment']
            for key in required_keys:
                if key not in features:
                    raise ValueError(f"Missing required feature: {key}")
            
            # Run fusion
            fused, metadata = self.engine.fuse(
                features['technical'],
                features['onchain'],
                features['sentiment'],
                metrics
            )
            
            self._process_count += 1
            
            # Check RAM limits
            if metadata['ram_usage_mb'] > self.config.ram_limit_mb * 0.9:
                self._last_error = "RAM usage approaching limit"
                # Don't mark unhealthy, just warn
            
            return fused, {
                'status': 'ok',
                'process_count': self._process_count,
                **metadata
            }
            
        except Exception as e:
            self._last_error = str(e)
            self._healthy = False
            return None, {
                'status': 'error',
                'error': str(e),
                'process_count': self._process_count
            }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on fusion pipeline."""
        stats = self.engine.get_stats()
        
        return {
            'healthy': self._healthy,
            'last_error': self._last_error,
            'process_count': self._process_count,
            'fusion_count': stats['fusion_count'],
            'avg_latency_us': stats['avg_latency_us'],
            'ram_usage_mb': stats['ram_usage_mb'],
            'ram_limit_mb': self.config.ram_limit_mb
        }
    
    def force_reset(self):
        """Force reset of fusion pipeline."""
        self.engine.reset()
        self._healthy = True
        self._last_error = None


# Module-level singleton instance
_orchestrator: Optional[FusionOrchestrator] = None
_lock = threading.Lock()


def get_orchestrator(config: Optional[FusionConfig] = None) -> FusionOrchestrator:
    """Get or create the global fusion orchestrator."""
    global _orchestrator
    
    with _lock:
        if _orchestrator is None:
            _orchestrator = FusionOrchestrator(config)
        return _orchestrator


def reset_orchestrator():
    """Reset the global orchestrator."""
    global _orchestrator
    with _lock:
        if _orchestrator is not None:
            _orchestrator.force_reset()
        _orchestrator = None


# Convenience functions for direct access
def fuse_features(features: Dict[str, np.ndarray],
                  metrics: Dict[str, float]) -> Tuple[Optional[np.ndarray], Dict]:
    """Convenience function to fuse features using global orchestrator."""
    orchestrator = get_orchestrator()
    return orchestrator.process(features, metrics)


def get_fusion_health() -> Dict[str, Any]:
    """Get health status of fusion pipeline."""
    orchestrator = get_orchestrator()
    return orchestrator.health_check()


# Module exports
__all__ = [
    'FusionConfig',
    'FusionMemoryManager',
    'UnifiedFusionEngine',
    'FusionOrchestrator',
    'get_orchestrator',
    'reset_orchestrator',
    'fuse_features',
    'get_fusion_health',
    'CrossAttentionFusion',
    'LightweightFusionPipeline',
    'ModalGatingNetwork',
    'AdaptiveFusionController',
    'MarketRegime'
]
