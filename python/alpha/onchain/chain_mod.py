"""
On-Chain Module Root.
Blends on-chain structural alpha with CEX high-frequency order flow imbalances.
Integrates flow analytics, TVL momentum, and exchange data.
Memory-efficient design with bounded buffers.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import time


class AlphaType(Enum):
    """Types of alpha signals."""
    ONCHAIN_FLOW = "onchain_flow"
    TVL_MOMENTUM = "tvl_momentum"
    STABLECOIN = "stablecoin"
    WHALE_ALERT = "whale_alert"
    CEX_ORDERFLOW = "cex_orderflow"
    BLEND = "blend"


@dataclass
class BlendedAlphaSignal:
    """Combined on-chain + CEX alpha signal."""
    timestamp_ns: int
    asset: str
    alpha_type: AlphaType
    direction: int  # 1=long, -1=short
    strength: float  # 0-1
    confidence: float
    onchain_component: float
    cex_component: float
    metadata: Dict


class OnChainCEXBlender:
    """
    Blends on-chain structural signals with CEX order flow.
    Uses adaptive weighting based on signal quality and regime.
    """
    
    def __init__(self, 
                 assets: List[str],
                 onchain_weight: float = 0.4,
                 cex_weight: float = 0.6):
        """
        Args:
            assets: Assets to monitor
            onchain_weight: Base weight for on-chain signals
            cex_weight: Base weight for CEX signals
        """
        self.assets = assets
        self.onchain_weight = onchain_weight
        self.cex_weight = cex_weight
        
        # Initialize sub-components
        from .flow_analytics import OnChainAlphaGenerator
        from .tvl_momentum import CapitalRotationDetector
        
        self.onchain_generator = OnChainAlphaGenerator(assets)
        self.rotation_detector = CapitalRotationDetector(
            protocols=assets,
            stablecoins=['USDT', 'USDC', 'DAI']
        )
        
        # CEX order flow tracking (simplified)
        self.orderflow_history = {asset: np.zeros(100) for asset in assets}
        self.orderflow_idx = {asset: 0 for asset in assets}
        
        # Signal history
        self.signal_buffer_size = 500
        self.signal_history = []
    
    def update_cex_orderflow(self, asset: str, imbalance: float):
        """
        Update CEX order flow imbalance.
        
        Args:
            asset: Asset symbol
            imbalance: Order flow imbalance (-1 to 1, positive = buy pressure)
        """
        if asset not in self.orderflow_history:
            self.orderflow_history[asset] = np.zeros(100)
        
        idx = self.orderflow_idx[asset] % 100
        self.orderflow_history[asset][idx] = imbalance
        self.orderflow_idx[asset] += 1
    
    def get_cex_signal(self, asset: str) -> Tuple[float, float]:
        """
        Get CEX order flow signal.
        
        Returns:
            Tuple of (signal_direction, confidence)
        """
        if asset not in self.orderflow_history:
            return 0.0, 0.0
        
        recent = self.orderflow_history[asset][-20:]  # Last 20 samples
        avg_imbalance = np.mean(recent)
        std_imbalance = np.std(recent) + 1e-10
        
        # Z-score of imbalance
        z_score = avg_imbalance / std_imbalance
        
        # Signal direction
        direction = np.sign(avg_imbalance)
        confidence = min(abs(z_score) / 3.0, 1.0)
        
        return direction, confidence
    
    def blend_signals(self, 
                      asset: str,
                      onchain_signal: Dict,
                      cex_direction: float,
                      cex_confidence: float) -> BlendedAlphaSignal:
        """
        Blend on-chain and CEX signals.
        
        Args:
            asset: Asset symbol
            onchain_signal: On-chain signal dictionary
            cex_direction: CEX signal direction
            cex_confidence: CEX signal confidence
            
        Returns:
            BlendedAlphaSignal
        """
        timestamp_ns = time.time_ns()
        
        # Extract on-chain components
        onchain_direction = onchain_signal.get('direction', 0)
        onchain_confidence = onchain_signal.get('confidence', 0.0)
        
        # Adaptive weighting based on confidence
        total_confidence = onchain_confidence + cex_confidence
        if total_confidence > 0:
            dynamic_onchain_weight = onchain_confidence / total_confidence
            dynamic_cex_weight = cex_confidence / total_confidence
        else:
            dynamic_onchain_weight = self.onchain_weight
            dynamic_cex_weight = self.cex_weight
        
        # Calculate blended direction
        if onchain_direction == cex_direction:
            # Signals agree - boost confidence
            blended_direction = onchain_direction
            combined_confidence = max(onchain_confidence, cex_confidence) * 1.2
        elif onchain_direction == 0:
            # Only CEX signal
            blended_direction = cex_direction
            combined_confidence = cex_confidence * self.cex_weight
        elif cex_direction == 0:
            # Only on-chain signal
            blended_direction = onchain_direction
            combined_confidence = onchain_confidence * self.onchain_weight
        else:
            # Signals conflict - reduce or flat
            if abs(onchain_confidence - cex_confidence) < 0.2:
                blended_direction = 0  # Conflicting signals cancel
                combined_confidence = abs(onchain_confidence - cex_confidence) * 0.5
            else:
                # Follow stronger signal
                if onchain_confidence > cex_confidence:
                    blended_direction = onchain_direction
                    combined_confidence = (onchain_confidence - cex_confidence) * 0.7
                else:
                    blended_direction = cex_direction
                    combined_confidence = (cex_confidence - onchain_confidence) * 0.7
        
        # Calculate strength
        strength = combined_confidence * abs(blended_direction)
        
        signal = BlendedAlphaSignal(
            timestamp_ns=timestamp_ns,
            asset=asset,
            alpha_type=AlphaType.BLEND,
            direction=int(np.sign(blended_direction)),
            strength=min(strength, 1.0),
            confidence=min(combined_confidence, 1.0),
            onchain_component=onchain_direction * onchain_confidence,
            cex_component=cex_direction * cex_confidence,
            metadata={
                'onchain_weight': dynamic_onchain_weight,
                'cex_weight': dynamic_cex_weight,
                'onchain_raw': onchain_signal,
                'cex_direction': cex_direction,
                'cex_confidence': cex_confidence
            }
        )
        
        # Store in history
        self.signal_history.append(signal)
        if len(self.signal_history) > self.signal_buffer_size:
            self.signal_history.pop(0)
        
        return signal
    
    def process_all_assets(self, 
                          prices: Dict[str, float],
                          price_changes: Dict[str, float]) -> List[BlendedAlphaSignal]:
        """
        Process all assets and generate blended signals.
        
        Args:
            prices: Current prices
            price_changes: Recent price changes (%)
            
        Returns:
            List of BlendedAlphaSignal
        """
        signals = []
        
        for asset in self.assets:
            # Get on-chain signals
            onchain_signals = self.onchain_generator.get_alpha_signals()
            asset_onchain = next((s for s in onchain_signals if s.get('asset') == asset), None)
            
            if asset_onchain is None:
                # Create default on-chain signal
                asset_onchain = {'direction': 0, 'confidence': 0.0}
            
            # Get CEX signal
            cex_direction, cex_confidence = self.get_cex_signal(asset)
            
            # Blend signals
            blended = self.blend_signals(
                asset,
                asset_onchain,
                cex_direction,
                cex_confidence
            )
            
            if blended.direction != 0 and blended.confidence > 0.3:
                signals.append(blended)
        
        return signals
    
    def get_nautilus_commands(self) -> List[Dict]:
        """Generate Nautilus Trader commands from blended signals."""
        commands = []
        
        for signal in self.signal_history[-100:]:  # Last 100 signals
            if signal.direction == 0:
                continue
            
            command = {
                'type': 'onchain_cex_blend',
                'instrument_id': f"{signal.asset}/USD",
                'side': 'BUY' if signal.direction > 0 else 'SELL',
                'strength': signal.strength,
                'confidence': signal.confidence,
                'alpha_type': signal.alpha_type.value,
                'timestamp_ns': signal.timestamp_ns,
                'metadata': {
                    'onchain_component': signal.onchain_component,
                    'cex_component': signal.cex_component,
                    **signal.metadata
                }
            }
            commands.append(command)
        
        return commands


class OnChainMessageBus:
    """
    Message bus for distributing on-chain signals to strategies.
    Similar pattern to volatility module for consistency.
    """
    
    def __init__(self):
        self.subscribers = {}
        self.signal_queues = {}
        self.stats = {'published': 0, 'dropped': 0}
    
    def subscribe(self, strategy: str, callback: callable):
        """Subscribe strategy to receive signals."""
        self.subscribers[strategy] = callback
    
    def publish(self, signal: BlendedAlphaSignal):
        """Publish signal to subscribers."""
        if signal.asset not in self.signal_queues:
            self.signal_queues[signal.asset] = []
        
        self.signal_queues[signal.asset].append(signal)
        
        # Trim queue
        if len(self.signal_queues[signal.asset]) > 1000:
            self.signal_queues[signal.asset].pop(0)
            self.stats['dropped'] += 1
        
        # Notify subscribers
        for callback in self.subscribers.values():
            try:
                callback(signal)
            except Exception:
                pass
        
        self.stats['published'] += 1


# Global blender instance factory
def create_onchain_system(assets: List[str]) -> Tuple[OnChainCEXBlender, OnChainMessageBus]:
    """
    Factory function to create on-chain alpha system.
    
    Args:
        assets: List of assets to monitor
        
    Returns:
        Tuple of (blender, message_bus)
    """
    blender = OnChainCEXBlender(assets)
    message_bus = OnChainMessageBus()
    
    return blender, message_bus


__all__ = [
    'OnChainCEXBlender',
    'OnChainMessageBus',
    'BlendedAlphaSignal',
    'AlphaType',
    'create_onchain_system'
]
