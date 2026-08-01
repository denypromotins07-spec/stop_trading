"""
Order Block Detector - Lightweight 1D-CNN for detecting institutional Order Blocks and Fair Value Gaps.
Exports to ONNX for efficient inference across language boundaries.
Memory-efficient design targeting <100MB RAM footprint.
"""
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import ONNX runtime, fall back to numpy implementation
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logger.info("ONNX runtime not available, using numpy fallback")

# Try to import torch for model export
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.info("PyTorch not available, ONNX export disabled")


class OrderBlockCNN(nn.Module):
    """
    Lightweight 1D-CNN architecture for Order Block and FVG detection.
    Designed for sub-millisecond inference on CPU.
    """
    
    def __init__(self, input_channels: int = 5, num_classes: int = 2):
        super().__init__()
        
        # Convolutional blocks
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU()
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        # Classification heads
        self.fc_ob = nn.Linear(128, 1)  # Order Block probability
        self.fc_fvg = nn.Linear(128, 1)  # Fair Value Gap probability
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x shape: (batch, channels, sequence_length)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.gap(x).squeeze(-1)
        
        ob_prob = self.sigmoid(self.fc_ob(x))
        fvg_prob = self.sigmoid(self.fc_fvg(x))
        
        return ob_prob, fvg_prob


class OrderBlockDetector:
    """
    Detects institutional Order Blocks and Fair Value Gaps from footprint charts.
    Uses ONNX-exported model for efficient inference.
    """
    
    def __init__(self, model_path: str = 'models/order_block.onnx', 
                 input_window: int = 50,
                 threshold_ob: float = 0.7,
                 threshold_fvg: float = 0.6):
        self.model_path = Path(model_path)
        self.input_window = input_window
        self.threshold_ob = threshold_ob
        self.threshold_fvg = threshold_fvg
        
        self.session = None
        self._input_buffer = None
        self._feature_buffer = None
        
        # Initialize buffers for zero-copy operations
        # Input: [open, high, low, close, volume] x window_size
        self._input_buffer = np.zeros((1, 5, input_window), dtype=np.float32)
        self._feature_buffer = np.zeros(input_window, dtype=np.float32)
        
        # Load or initialize model
        self._load_model()
        
        logger.info(f"OrderBlockDetector initialized with window={input_window}")
    
    def _load_model(self) -> None:
        """Load ONNX model or initialize fallback."""
        if ONNX_AVAILABLE and self.model_path.exists():
            try:
                self.session = ort.InferenceSession(
                    str(self.model_path),
                    providers=['CPUExecutionProvider']
                )
                logger.info(f"Loaded ONNX model from {self.model_path}")
            except Exception as e:
                logger.warning(f"Failed to load ONNX model: {e}, using fallback")
                self.session = None
        else:
            logger.info("Using numpy fallback for order block detection")
            self.session = None
    
    def export_to_onnx(self, output_path: str) -> bool:
        """Export PyTorch model to ONNX format."""
        if not TORCH_AVAILABLE:
            logger.error("PyTorch not available for ONNX export")
            return False
        
        model = OrderBlockCNN()
        model.eval()
        
        dummy_input = torch.randn(1, 5, self.input_window)
        
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['footprint'],
            output_names=['ob_prob', 'fvg_prob'],
            dynamic_axes=None
        )
        
        logger.info(f"Exported OrderBlockCNN to {output_path}")
        return True
    
    def _extract_features(self, footprint_data: np.ndarray) -> np.ndarray:
        """
        Extract features from raw footprint data.
        Footprint data expected format: [price, volume, bid_volume, ask_volume, delta]
        
        Returns normalized feature tensor for CNN input.
        """
        # Ensure we have the right shape
        if footprint_data.ndim == 1:
            # Single tick, append to buffer
            self._feature_buffer[:-1] = self._feature_buffer[1:]
            self._feature_buffer[-1] = footprint_data[0] if len(footprint_data) > 0 else 0
            recent_data = self._feature_buffer.copy()
        else:
            # Use most recent window
            recent_data = footprint_data[-self.input_window:]
            if len(recent_data) < self.input_window:
                # Pad with zeros
                padded = np.zeros((self.input_window, footprint_data.shape[1]))
                padded[-len(recent_data):] = recent_data
                recent_data = padded
        
        # Transpose to (channels, sequence) format
        features = recent_data.T.astype(np.float32)
        
        # Normalize features
        if features.shape[0] >= 4:
            # Price normalization (z-score over window)
            prices = features[:4]
            mean_price = np.mean(prices, axis=1, keepdims=True)
            std_price = np.std(prices, axis=1, keepdims=True) + 1e-8
            features[:4] = (prices - mean_price) / std_price
            
            # Volume normalization (log scale)
            if features.shape[0] > 4:
                volumes = features[4:]
                features[4:] = np.log1p(volumes) / 10.0
        
        return features
    
    def detect(self, footprint_data: np.ndarray) -> Dict[str, float]:
        """
        Detect Order Blocks and Fair Value Gaps in footprint data.
        
        Args:
            footprint_data: Array of footprint data [tick, features]
                           Features: [open, high, low, close, volume]
        
        Returns:
            Dictionary with probabilities and detected levels
        """
        # Extract and normalize features
        features = self._extract_features(footprint_data)
        
        # Update input buffer (zero-copy)
        self._input_buffer[0] = features
        
        ob_prob = 0.0
        fvg_prob = 0.0
        
        if self.session is not None and ONNX_AVAILABLE:
            # Run ONNX inference
            try:
                outputs = self.session.run(
                    None,
                    {'footprint': self._input_buffer}
                )
                ob_prob = float(outputs[0][0][0])
                fvg_prob = float(outputs[1][0][0])
            except Exception as e:
                logger.error(f"ONNX inference failed: {e}")
                ob_prob, fvg_prob = self._numpy_fallback(features)
        else:
            # Use numpy fallback
            ob_prob, fvg_prob = self._numpy_fallback(features)
        
        # Detect specific levels
        ob_levels = self._find_order_block_levels(footprint_data, ob_prob)
        fvg_levels = self._find_fvg_levels(footprint_data, fvg_prob)
        
        return {
            'order_block_prob': ob_prob,
            'fvg_prob': fvg_prob,
            'is_order_block': ob_prob > self.threshold_ob,
            'is_fvg': fvg_prob > self.threshold_fvg,
            'ob_levels': ob_levels,
            'fvg_levels': fvg_levels,
            'confidence': max(ob_prob, fvg_prob)
        }
    
    def _numpy_fallback(self, features: np.ndarray) -> Tuple[float, float]:
        """
        Simple numpy-based heuristic fallback when ONNX model unavailable.
        Uses volatility and volume patterns to estimate OB/FVG probability.
        """
        if features.shape[0] < 4:
            return 0.0, 0.0
        
        prices = features[:4]  # OHLC
        
        # Calculate price range and position
        highs = prices[1]
        lows = prices[2]
        closes = prices[3]
        
        # Volatility measure
        ranges = highs - lows
        avg_range = np.mean(ranges[-10:]) + 1e-8
        
        # Recent volatility spike detection (potential OB)
        recent_ranges = ranges[-5:]
        vol_spike = np.max(recent_ranges) / avg_range
        
        # Close position within range (institutional absorption pattern)
        range_positions = (closes - lows) / (ranges + 1e-8)
        extreme_closes = np.sum((range_positions < 0.2) | (range_positions > 0.8))
        
        # Order Block probability based on volatility and absorption
        ob_prob = min(1.0, (vol_spike * 0.4 + extreme_closes / 5 * 0.6))
        
        # FVG detection (large gaps between candles)
        if len(closes) > 1:
            gaps = np.abs(closes[1:] - closes[:-1])
            avg_gap = np.mean(gaps) + 1e-8
            max_gap_ratio = np.max(gaps[-5:]) / avg_gap
            fvg_prob = min(1.0, max_gap_ratio * 0.3)
        else:
            fvg_prob = 0.0
        
        return float(ob_prob), float(fvg_prob)
    
    def _find_order_block_levels(self, footprint_data: np.ndarray, 
                                  prob: float) -> list:
        """Find specific price levels for detected order blocks."""
        levels = []
        if prob < self.threshold_ob or len(footprint_data) < 10:
            return levels
        
        # Look for consolidation zones with high volume
        if footprint_data.ndim > 1 and footprint_data.shape[1] >= 4:
            closes = footprint_data[:, 3]
            volumes = footprint_data[:, 4] if footprint_data.shape[1] > 4 else np.ones(len(closes))
            
            # Find tight range with elevated volume
            for i in range(len(closes) - 10, 0, -1):
                window_close = closes[i:i+10]
                window_vol = volumes[i:i+10]
                
                price_range = np.max(window_close) - np.min(window_close)
                avg_vol = np.mean(window_vol)
                
                if price_range < np.std(closes[-50:]) * 0.5 and avg_vol > np.mean(volumes) * 1.2:
                    levels.append({
                        'price': float(np.mean(window_close)),
                        'start_idx': i,
                        'strength': float(prob)
                    })
                    break  # Return most recent
        
        return levels
    
    def _find_fvg_levels(self, footprint_data: np.ndarray, 
                         prob: float) -> list:
        """Find specific price levels for detected Fair Value Gaps."""
        levels = []
        if prob < self.threshold_fvg or len(footprint_data) < 3:
            return levels
        
        if footprint_data.ndim > 1 and footprint_data.shape[1] >= 2:
            highs = footprint_data[:, 1]
            lows = footprint_data[:, 2]
            
            # Detect gaps between consecutive candles
            for i in range(len(highs) - 1, 0, -1):
                prev_high = highs[i-1]
                curr_low = lows[i]
                
                if prev_high > curr_low * 1.001:  # Gap detected
                    gap_size = (prev_high - curr_low) / curr_low
                    if gap_size > 0.0005:  # Minimum gap threshold
                        levels.append({
                            'top': float(prev_high),
                            'bottom': float(curr_low),
                            'mid': float((prev_high + curr_low) / 2),
                            'gap_size': float(gap_size),
                            'idx': i
                        })
                        if len(levels) >= 3:  # Return up to 3 most recent
                            break
        
        return levels
    
    def warmup(self, historical_data: np.ndarray) -> None:
        """Warm up detector with historical data."""
        # Process historical data to prime any internal state
        if len(historical_data) > self.input_window:
            _ = self.detect(historical_data[-self.input_window:])
        logger.debug("OrderBlockDetector warmed up")
    
    def save(self) -> Dict[str, Any]:
        """Save detector state (configuration only, model is static)."""
        return {
            'threshold_ob': self.threshold_ob,
            'threshold_fvg': self.threshold_fvg,
            'input_window': self.input_window
        }
    
    def load(self, state: Dict[str, Any]) -> None:
        """Load detector state."""
        self.threshold_ob = state.get('threshold_ob', self.threshold_ob)
        self.threshold_fvg = state.get('threshold_fvg', self.threshold_fvg)
        self.input_window = state.get('input_window', self.input_window)
        
        # Reinitialize buffers with new window size
        self._input_buffer = np.zeros((1, 5, self.input_window), dtype=np.float32)
        self._feature_buffer = np.zeros(self.input_window, dtype=np.float32)


__all__ = ['OrderBlockDetector', 'OrderBlockCNN']
