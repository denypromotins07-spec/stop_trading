"""
Lightweight Time-Series Transformer for Sequence Modeling
Designed specifically for ONNX export to minimize memory footprint.
Avoids heavy PyTorch eager-mode imports in production.
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
import math

# Conditional PyTorch import - only for model definition/export
# Production inference uses ONNX runtime only
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class TransformerConfig:
    """Configuration for lightweight time-series transformer."""
    input_dim: int = 64
    d_model: int = 128  # Model dimension (kept small for low RAM)
    n_heads: int = 4
    n_layers: int = 2
    dim_feedforward: int = 256
    dropout: float = 0.1
    max_seq_len: int = 128
    output_dim: int = 1
    use_positional_encoding: bool = True
    layer_norm_eps: float = 1e-5


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for sequence positions.
    Pre-computed for efficiency.
    """
    
    def __init__(self, d_model: int, max_len: int = 128, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (seq_len, batch_size, d_model)
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class LightweightTransformerEncoderLayer(nn.Module):
    """
    Memory-efficient transformer encoder layer.
    Uses pre-normalization for better training stability.
    """
    
    def __init__(self, 
                 d_model: int,
                 n_heads: int,
                 dim_feedforward: int = 256,
                 dropout: float = 0.1,
                 layer_norm_eps: float = 1e-5):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=False
        )
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = nn.GELU()  # More efficient than ReLU for transformers
    
    def forward(self, 
                src: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with pre-normalization.
        
        Args:
            src: Input tensor (seq_len, batch_size, d_model)
            src_mask: Attention mask
            src_key_padding_mask: Padding mask
        
        Returns:
            Output tensor (seq_len, batch_size, d_model)
        """
        # Pre-norm attention
        src_norm = self.norm1(src)
        attn_output, _ = self.self_attn(
            src_norm, src_norm, src_norm,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )
        src = src + self.dropout1(attn_output)
        
        # Pre-norm feedforward
        src_norm = self.norm2(src)
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(src_norm))))
        src = src + self.dropout2(ff_output)
        
        return src


class TimeSeriesTransformer(nn.Module):
    """
    Lightweight time-series transformer for sequence modeling.
    Optimized for ONNX export and low-memory inference.
    """
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for model definition")
        
        # Input projection
        self.input_projection = nn.Linear(config.input_dim, config.d_model)
        
        # Positional encoding
        if config.use_positional_encoding:
            self.pos_encoder = PositionalEncoding(
                config.d_model, 
                config.max_seq_len,
                config.dropout
            )
        else:
            self.pos_encoder = None
        
        # Transformer encoder layers
        encoder_layers = [
            LightweightTransformerEncoderLayer(
                d_model=config.d_model,
                n_heads=config.n_heads,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout,
                layer_norm_eps=config.layer_norm_eps
            )
            for _ in range(config.n_layers)
        ]
        
        self.encoder_layers = nn.ModuleList(encoder_layers)
        self.norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        
        # Output head
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # Global average pooling
        self.output_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, config.output_dim)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize model weights with Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate causal mask for autoregressive prediction."""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self, 
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through transformer.
        
        Args:
            x: Input tensor (batch_size, seq_len, input_dim)
            mask: Optional attention mask
        
        Returns:
            Output predictions (batch_size, output_dim)
        """
        # Transpose for transformer: (batch, seq, dim) -> (seq, batch, dim)
        x = x.transpose(0, 1)
        
        # Project input
        x = self.input_projection(x)
        
        # Add positional encoding
        if self.pos_encoder is not None:
            x = self.pos_encoder(x)
        
        # Apply transformer layers
        for layer in self.encoder_layers:
            x = layer(x, src_mask=mask)
        
        # Final normalization
        x = self.norm(x)
        
        # Transpose back: (seq, batch, dim) -> (batch, dim, seq)
        x = x.transpose(0, 1).transpose(1, 2)
        
        # Global pooling: (batch, dim, 1)
        x = self.global_pool(x).squeeze(-1)
        
        # Output head
        return self.output_head(x)
    
    def predict_next(self, 
                     x: torch.Tensor,
                     steps: int = 1) -> torch.Tensor:
        """
        Autoregressive prediction for next timesteps.
        
        Args:
            x: Input sequence (batch_size, seq_len, input_dim)
            steps: Number of steps to predict
        
        Returns:
            Predictions (batch_size, steps, output_dim)
        """
        self.eval()
        predictions = []
        
        with torch.no_grad():
            current_input = x.clone()
            
            for _ in range(steps):
                # Get prediction for last timestep
                pred = self.forward(current_input)
                predictions.append(pred.unsqueeze(1))
                
                # Update input (simple approach - in practice would need proper feature handling)
                # This is a placeholder for actual autoregressive logic
                break  # For now, just single step
        
        return torch.cat(predictions, dim=1) if predictions else torch.tensor([])
    
    def export_to_onnx(self, 
                       output_path: str,
                       batch_size: int = 1,
                       seq_len: int = 64,
                       opset_version: int = 14) -> None:
        """
        Export model to ONNX format for efficient inference.
        
        Args:
            output_path: Path to save ONNX model
            batch_size: Batch size for export
            seq_len: Sequence length for export
            opset_version: ONNX opset version
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for ONNX export")
        
        self.eval()
        
        # Create dummy input
        dummy_input = torch.randn(batch_size, seq_len, self.config.input_dim)
        
        # Export
        torch.onnx.export(
            self,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size', 1: 'sequence_length'},
                'output': {0: 'batch_size'}
            }
        )
        
        print(f"Model exported to {output_path}")
    
    def get_model_size(self) -> Dict[str, Any]:
        """Get model size statistics."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "d_model": self.config.d_model,
            "n_layers": self.config.n_layers,
            "n_heads": self.config.n_heads,
        }


def create_transformer(config: Optional[TransformerConfig] = None) -> TimeSeriesTransformer:
    """
    Factory function to create a TimeSeriesTransformer.
    
    Args:
        config: Transformer configuration
    
    Returns:
        TimeSeriesTransformer instance
    """
    if config is None:
        config = TransformerConfig()
    
    return TimeSeriesTransformer(config)


def load_transformer_from_onnx(onnx_path: str) -> Dict[str, Any]:
    """
    Load transformer info from ONNX model (without PyTorch).
    Returns metadata about the model.
    
    Args:
        onnx_path: Path to ONNX model
    
    Returns:
        Dictionary with model metadata
    """
    try:
        import onnx
        model = onnx.load(onnx_path)
        
        # Extract input/output shapes
        inputs = []
        outputs = []
        
        for inp in model.graph.input:
            shape = [d.dim_value if d.dim_value > 0 else -1 for d in inp.type.tensor_type.shape.dim]
            inputs.append({"name": inp.name, "shape": shape})
        
        for out in model.graph.output:
            shape = [d.dim_value if d.dim_value > 0 else -1 for d in out.type.tensor_type.shape.dim]
            outputs.append({"name": out.name, "shape": shape})
        
        return {
            "onnx_version": model.opset_import[0].version,
            "inputs": inputs,
            "outputs": outputs,
            "num_nodes": len(model.graph.node),
        }
    
    except ImportError:
        return {"error": "onnx package not available"}


if __name__ == "__main__":
    if not TORCH_AVAILABLE:
        print("PyTorch not available. Skipping transformer creation.")
        exit(0)
    
    # Example usage
    config = TransformerConfig(
        input_dim=32,
        d_model=64,
        n_heads=4,
        n_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        max_seq_len=64,
        output_dim=1
    )
    
    model = create_transformer(config)
    
    # Print model info
    print(f"Model created: {model.get_model_size()}")
    
    # Test forward pass
    batch_size = 4
    seq_len = 32
    x = torch.randn(batch_size, seq_len, config.input_dim)
    
    model.eval()
    with torch.no_grad():
        output = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
    
    # Export to ONNX (optional)
    # model.export_to_onnx("transformer.onnx", batch_size=1, seq_len=32)
