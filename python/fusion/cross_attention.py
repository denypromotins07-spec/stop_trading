"""
Cross-Attention Feature Fusion for HFT
Lightweight ONNX-compiled cross-attention mechanism fusing L2 order book tensors 
with on-chain flow vectors. Captures non-linear interactions between high-frequency 
microstructure and low-frequency structural capital rotations.

Strictly bounded tensor dimensions to respect 3GB Python RAM ceiling.
"""

import numpy as np
from typing import Tuple, Optional
import onnxruntime as ort
import onnx
from onnx import helper, TensorProto, numpy_helper


class CrossAttentionFusion:
    """
    Lightweight cross-attention mechanism for multi-modal feature fusion.
    Fuses L2 order book tensors (high-freq) with on-chain flow vectors (low-freq).
    """
    
    def __init__(self, 
                 l2_dim: int = 64,
                 chain_dim: int = 32,
                 embed_dim: int = 48,
                 num_heads: int = 4,
                 seq_len: int = 16):
        """
        Initialize cross-attention fusion module.
        
        Args:
            l2_dim: Dimension of L2 order book features
            chain_dim: Dimension of on-chain flow features
            embed_dim: Embedding dimension for attention
            num_heads: Number of attention heads
            seq_len: Sequence length for temporal context
        """
        self.l2_dim = l2_dim
        self.chain_dim = chain_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.head_dim = embed_dim // num_heads
        
        # Pre-compute weight matrices (fixed after initialization)
        self._init_weights()
        
        # Build and compile ONNX model
        self._build_onnx_model()
        self.session = ort.InferenceSession(
            self.model_path,
            providers=['CPUExecutionProvider']
        )
        
    def _init_weights(self):
        """Initialize fixed weight matrices for projection."""
        rng = np.random.default_rng(seed=42)
        
        # Query projection (L2 -> embed)
        self.W_q = rng.normal(0, 0.1, (self.l2_dim, self.embed_dim)).astype(np.float32)
        
        # Key projection (Chain -> embed)
        self.W_k = rng.normal(0, 0.1, (self.chain_dim, self.embed_dim)).astype(np.float32)
        
        # Value projection (Chain -> embed)
        self.W_v = rng.normal(0, 0.1, (self.chain_dim, self.embed_dim)).astype(np.float32)
        
        # Output projection
        self.W_out = rng.normal(0, 0.1, (self.embed_dim, self.embed_dim)).astype(np.float32)
        
    def _build_onnx_model(self):
        """Build ONNX model for cross-attention computation."""
        import tempfile
        import os
        
        # Input definitions
        l2_input = helper.make_tensor_value_info('l2_features', TensorProto.FLOAT, 
                                                  [1, self.seq_len, self.l2_dim])
        chain_input = helper.make_tensor_value_info('chain_features', TensorProto.FLOAT,
                                                     [1, self.seq_len, self.chain_dim])
        
        # Output definition
        output = helper.make_tensor_value_info('fused_output', TensorProto.FLOAT,
                                                [1, self.seq_len, self.embed_dim])
        
        # Store weights as initializers
        initializers = [
            numpy_helper.from_array(self.W_q, 'W_q'),
            numpy_helper.from_array(self.W_k, 'W_k'),
            numpy_helper.from_array(self.W_v, 'W_v'),
            numpy_helper.from_array(self.W_out, 'W_out'),
        ]
        
        # Create nodes for cross-attention computation
        nodes = []
        
        # Project L2 to Query: Q = L2 @ W_q
        q_proj = helper.make_node(
            'MatMul', ['l2_features', 'W_q'], ['Q'], name='Q_proj'
        )
        nodes.append(q_proj)
        
        # Project Chain to Key: K = Chain @ W_k
        k_proj = helper.make_node(
            'MatMul', ['chain_features', 'W_k'], ['K'], name='K_proj'
        )
        nodes.append(k_proj)
        
        # Project Chain to Value: V = Chain @ W_v
        v_proj = helper.make_node(
            'MatMul', ['chain_features', 'W_v'], ['V'], name='V_proj'
        )
        nodes.append(v_proj)
        
        # Scaled Dot-Product Attention: Attention(Q, K, V) = softmax(QK^T/sqrt(d))V
        # Transpose K for multiplication
        k_transpose = helper.make_node(
            'Transpose', ['K'], ['K_T'], perm=[0, 2, 1], name='K_transpose'
        )
        nodes.append(k_transpose)
        
        # Q @ K^T
        qk_matmul = helper.make_node(
            'MatMul', ['Q', 'K_T'], ['QK'], name='QK_matmul'
        )
        nodes.append(qk_matmul)
        
        # Scale by sqrt(head_dim)
        scale_factor = 1.0 / np.sqrt(self.head_dim)
        scale_const = helper.make_node(
            'Constant', [], ['scale'], 
            value=helper.make_tensor('scale_val', TensorProto.FLOAT, [], [scale_factor])
        )
        nodes.append(scale_const)
        
        qk_scaled = helper.make_node(
            'Mul', ['QK', 'scale'], ['QK_scaled'], name='QK_scale'
        )
        nodes.append(qk_scaled)
        
        # Softmax
        softmax = helper.make_node(
            'Softmax', ['QK_scaled'], ['attn_weights'], axis=-1, name='softmax'
        )
        nodes.append(softmax)
        
        # Attention @ V
        attn_v = helper.make_node(
            'MatMul', ['attn_weights', 'V'], ['attn_out'], name='attn_V'
        )
        nodes.append(attn_v)
        
        # Output projection
        output_proj = helper.make_node(
            'MatMul', ['attn_out', 'W_out'], ['fused_output'], name='out_proj'
        )
        nodes.append(output_proj)
        
        # Create graph
        graph = helper.make_graph(
            nodes,
            'cross_attention_graph',
            [l2_input, chain_input],
            [output],
            initializers
        )
        
        # Create model
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
        model.ir_version = 7
        
        # Save model
        temp_dir = tempfile.gettempdir()
        self.model_path = os.path.join(temp_dir, 'cross_attention_fusion.onnx')
        onnx.save(model, self.model_path)
        
    def fuse(self, 
             l2_features: np.ndarray, 
             chain_features: np.ndarray) -> np.ndarray:
        """
        Perform cross-attention fusion of L2 and on-chain features.
        
        Args:
            l2_features: L2 order book features [batch, seq_len, l2_dim]
            chain_features: On-chain flow features [batch, seq_len, chain_dim]
            
        Returns:
            Fused features [batch, seq_len, embed_dim]
        """
        # Ensure correct shapes and types
        if l2_features.ndim == 2:
            l2_features = l2_features[np.newaxis, :, :]
        if chain_features.ndim == 2:
            chain_features = chain_features[np.newaxis, :, :]
            
        l2_features = np.ascontiguousarray(l2_features, dtype=np.float32)
        chain_features = np.ascontiguousarray(chain_features, dtype=np.float32)
        
        # Run inference
        fused = self.session.run(
            None,
            {
                'l2_features': l2_features,
                'chain_features': chain_features
            }
        )[0]
        
        return fused
    
    def get_fused_dimension(self) -> int:
        """Return the dimension of fused output."""
        return self.embed_dim


class LightweightFusionPipeline:
    """
    Production-ready fusion pipeline with memory bounds checking.
    Manages batch processing and RAM constraints.
    """
    
    MAX_BATCH_SIZE = 32
    MAX_SEQ_LEN = 32
    
    def __init__(self, l2_dim: int = 64, chain_dim: int = 32):
        self.attention = CrossAttentionFusion(
            l2_dim=l2_dim,
            chain_dim=chain_dim,
            embed_dim=48,
            num_heads=4,
            seq_len=self.MAX_SEQ_LEN
        )
        self._buffer_pool = []
        
    def process_batch(self, 
                      l2_batch: np.ndarray, 
                      chain_batch: np.ndarray) -> np.ndarray:
        """
        Process a batch of features through fusion pipeline.
        
        Args:
            l2_batch: Batch of L2 features
            chain_batch: Batch of chain features
            
        Returns:
            Fused features batch
        """
        # Enforce batch size limits
        if l2_batch.shape[0] > self.MAX_BATCH_SIZE:
            # Process in chunks
            results = []
            for i in range(0, l2_batch.shape[0], self.MAX_BATCH_SIZE):
                end_idx = min(i + self.MAX_BATCH_SIZE, l2_batch.shape[0])
                chunk_l2 = l2_batch[i:end_idx]
                chunk_chain = chain_batch[i:end_idx]
                result = self.attention.fuse(chunk_l2, chunk_chain)
                results.append(result)
            return np.concatenate(results, axis=0)
        
        return self.attention.fuse(l2_batch, chain_batch)
    
    def get_memory_footprint(self) -> int:
        """Estimate memory footprint in bytes."""
        # Approximate: weights + buffers
        weight_size = (
            self.attention.W_q.nbytes +
            self.attention.W_k.nbytes +
            self.attention.W_v.nbytes +
            self.attention.W_out.nbytes
        )
        buffer_size = (
            self.MAX_BATCH_SIZE * self.MAX_SEQ_LEN * 
            self.attention.embed_dim * 4  # float32
        )
        return weight_size + buffer_size


# Module exports
__all__ = ['CrossAttentionFusion', 'LightweightFusionPipeline']
