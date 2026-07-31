"""
On-Chain Flow Analytics.
Consumes exchange inflow/outflow Z-scores and whale alerts from Rust IPC bridge.
Translates raw on-chain byte streams into normalized alpha features.
Uses bounded NumPy arrays to prevent GIL contention.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import struct


class FlowDirection(Enum):
    """Direction of on-chain flow."""
    INFLOW = 1      # To exchange (bearish)
    OUTFLOW = -1    # From exchange (bullish)
    NEUTRAL = 0


@dataclass
class WhaleAlert:
    """Whale transaction alert."""
    timestamp_ns: int
    asset: str
    amount_usd: float
    direction: FlowDirection
    from_address: str
    to_address: str
    tx_hash: str
    z_score: float


@dataclass
class FlowSignal:
    """Normalized flow-based trading signal."""
    timestamp_ns: int
    asset: str
    net_flow_zscore: float
    whale_count: int
    total_volume_usd: float
    signal_direction: int  # 1=long, -1=short
    confidence: float
    cumulative_flow_1h: float


class OnChainFlowParser:
    """
    Parses raw byte streams from Rust IPC bridge into structured flow data.
    Optimized for zero-copy parsing where possible.
    """
    
    def __init__(self, buffer_size: int = 10000):
        """
        Args:
            buffer_size: Maximum events to keep in memory
        """
        self.buffer_size = buffer_size
        
        # Circular buffers for flow data
        self.flow_timestamps = np.zeros(buffer_size, dtype=np.int64)
        self.flow_amounts = np.zeros(buffer_size, dtype=np.float64)
        self.flow_directions = np.zeros(buffer_size, dtype=np.int8)
        self.flow_assets = []  # String list (unavoidable)
        
        self.buf_idx = 0
        self.samples_count = 0
        
        # Asset-specific tracking
        self.asset_flows = {}  # asset -> net flow accumulator
        
    def parse_ipc_message(self, data: bytes) -> Optional[Dict]:
        """
        Parse IPC message from Rust bridge.
        
        Expected format (little-endian):
        - 8 bytes: timestamp_ns (int64)
        - 8 bytes: amount_usd (float64)
        - 1 byte: direction (int8)
        - 1 byte: asset_len
        - N bytes: asset symbol
        
        Args:
            data: Raw bytes from IPC
            
        Returns:
            Parsed message dictionary or None
        """
        if len(data) < 18:  # Minimum size
            return None
        
        try:
            # Unpack fixed-size fields
            timestamp_ns, amount_usd, direction = struct.unpack('<qdB', data[:17])
            asset_len = data[17]
            
            if len(data) < 18 + asset_len:
                return None
            
            # Extract asset symbol
            asset = data[18:18+asset_len].decode('utf-8')
            
            return {
                'timestamp_ns': timestamp_ns,
                'amount_usd': amount_usd,
                'direction': direction,
                'asset': asset
            }
        except (struct.error, UnicodeDecodeError):
            return None
    
    def add_flow_event(self, 
                       timestamp_ns: int,
                       amount_usd: float,
                       direction: int,
                       asset: str):
        """
        Add parsed flow event to buffers.
        
        Args:
            timestamp_ns: Event timestamp
            amount_usd: USD value
            direction: 1=inflow, -1=outflow
            asset: Asset symbol
        """
        idx = self.buf_idx % self.buffer_size
        
        self.flow_timestamps[idx] = timestamp_ns
        self.flow_amounts[idx] = amount_usd
        self.flow_directions[idx] = direction
        
        # Track asset flows
        signed_amount = amount_usd * direction
        if asset not in self.asset_flows:
            self.asset_flows[asset] = 0.0
        self.asset_flows[asset] += signed_amount
        
        self.buf_idx += 1
        self.samples_count += 1
    
    def get_net_flow(self, asset: str, window_ns: int = 3600_000_000_000) -> float:
        """
        Get net flow for asset over time window.
        
        Args:
            asset: Asset symbol
            window_ns: Time window in nanoseconds (default 1 hour)
            
        Returns:
            Net flow in USD (positive = inflow, negative = outflow)
        """
        if self.samples_count == 0:
            return 0.0
        
        current_ns = self.flow_timestamps[(self.buf_idx - 1) % self.buffer_size]
        cutoff_ns = current_ns - window_ns
        
        # Find relevant events
        valid_count = min(self.buf_idx, self.buffer_size)
        start_idx = max(0, self.buf_idx - valid_count)
        
        net_flow = 0.0
        for i in range(start_idx, self.buf_idx):
            idx = i % self.buffer_size
            if self.flow_timestamps[idx] >= cutoff_ns:
                # Would need asset tracking per event for accurate filtering
                # Simplified: return total recent flow
                net_flow += self.flow_amounts[idx] * self.flow_directions[idx]
        
        return net_flow
    
    def reset_asset_flow(self, asset: str):
        """Reset flow accumulator for an asset."""
        if asset in self.asset_flows:
            self.asset_flows[asset] = 0.0


class ExchangeFlowAnalyzer:
    """
    Analyzes exchange flow data for alpha signals.
    Calculates Z-scores and detects anomalous flow patterns.
    """
    
    def __init__(self, 
                 assets: List[str],
                 lookback_hours: int = 24,
                 zscore_threshold: float = 2.0):
        """
        Args:
            assets: Assets to monitor
            lookback_hours: Historical lookback for statistics
            zscore_threshold: Threshold for anomaly detection
        """
        self.assets = assets
        self.lookback_hours = lookback_hours
        self.zscore_threshold = zscore_threshold
        
        # Flow history per asset (hourly buckets)
        self.hourly_flows = {asset: np.zeros(lookback_hours) for asset in assets}
        self.current_hour_idx = {asset: 0 for asset in assets}
        
        # Statistics
        self.flow_means = {asset: 0.0 for asset in assets}
        self.flow_stds = {asset: 1.0 for asset in assets}
        
        # Whale tracking
        self.whale_threshold_usd = 100_000  # $100k minimum for whale
        self.whale_alerts = []
        
    def update_hourly_flow(self, asset: str, flow_usd: float):
        """Update hourly flow for an asset."""
        if asset not in self.hourly_flows:
            self.hourly_flows[asset] = np.zeros(self.lookback_hours)
            self.current_hour_idx[asset] = 0
        
        idx = self.current_hour_idx[asset] % self.lookback_hours
        self.hourly_flows[asset][idx] = flow_usd
        self.current_hour_idx[asset] += 1
        
        # Update statistics
        self._update_statistics(asset)
    
    def _update_statistics(self, asset: str):
        """Update flow statistics for an asset."""
        flows = self.hourly_flows[asset]
        valid_flows = flows[flows != 0] if np.any(flows != 0) else flows
        
        if len(valid_flows) > 5:
            self.flow_means[asset] = np.mean(valid_flows)
            self.flow_stds[asset] = np.std(valid_flows) + 1e-10
    
    def calculate_flow_zscore(self, asset: str, current_flow: float) -> float:
        """Calculate Z-score of current flow."""
        if asset not in self.flow_stds or self.flow_stds[asset] < 1e-10:
            return 0.0
        
        z = (current_flow - self.flow_means[asset]) / self.flow_stds[asset]
        return z
    
    def detect_whale_transaction(self, 
                                  timestamp_ns: int,
                                  asset: str,
                                  amount_usd: float,
                                  direction: int,
                                  from_addr: str,
                                  to_addr: str,
                                  tx_hash: str) -> Optional[WhaleAlert]:
        """
        Detect and record whale transactions.
        
        Args:
            timestamp_ns: Transaction timestamp
            asset: Asset symbol
            amount_usd: USD value
            direction: Flow direction
            from_addr: Source address
            to_addr: Destination address
            tx_hash: Transaction hash
            
        Returns:
            WhaleAlert if whale detected, None otherwise
        """
        if amount_usd < self.whale_threshold_usd:
            return None
        
        # Calculate flow Z-score
        z_score = self.calculate_flow_zscore(asset, amount_usd * direction)
        
        alert = WhaleAlert(
            timestamp_ns=timestamp_ns,
            asset=asset,
            amount_usd=amount_usd,
            direction=FlowDirection(direction),
            from_address=from_addr,
            to_address=to_addr,
            tx_hash=tx_hash,
            z_score=z_score
        )
        
        self.whale_alerts.append(alert)
        
        # Trim alerts list
        if len(self.whale_alerts) > 1000:
            self.whale_alerts.pop(0)
        
        return alert
    
    def generate_flow_signal(self, 
                             asset: str,
                             current_flow: float,
                             whale_count: int,
                             total_volume: float) -> FlowSignal:
        """
        Generate trading signal from flow analysis.
        
        Args:
            asset: Asset symbol
            current_flow: Current period net flow
            whale_count: Number of recent whale transactions
            total_volume: Total volume in period
            
        Returns:
            FlowSignal
        """
        import time
        timestamp_ns = time.time_ns()
        
        # Calculate Z-score
        z_score = self.calculate_flow_zscore(asset, current_flow)
        
        # Determine signal direction
        # Negative flow (outflow) is bullish, positive flow (inflow) is bearish
        if z_score < -self.zscore_threshold:
            # Significant outflow = bullish
            signal_direction = 1
            confidence = min(abs(z_score) / 3.0, 1.0)
        elif z_score > self.zscore_threshold:
            # Significant inflow = bearish
            signal_direction = -1
            confidence = min(abs(z_score) / 3.0, 1.0)
        else:
            signal_direction = 0
            confidence = abs(z_score) / self.zscore_threshold
        
        # Boost confidence with whale activity
        if whale_count > 3:
            confidence = min(confidence + 0.2, 1.0)
        
        return FlowSignal(
            timestamp_ns=timestamp_ns,
            asset=asset,
            net_flow_zscore=z_score,
            whale_count=whale_count,
            total_volume_usd=total_volume,
            signal_direction=signal_direction,
            confidence=confidence,
            cumulative_flow_1h=current_flow
        )


class OnChainAlphaGenerator:
    """
    Main generator combining flow analytics with price data.
    Produces normalized alpha features for downstream models.
    """
    
    def __init__(self, assets: List[str]):
        """
        Args:
            assets: Assets to monitor
        """
        self.assets = assets
        self.flow_parser = OnChainFlowParser()
        self.analyzers = {asset: ExchangeFlowAnalyzer([asset]) for asset in assets}
        
        # Feature storage (bounded)
        self.feature_buffer_size = 1000
        self.features = {asset: np.zeros((self.feature_buffer_size, 10)) 
                        for asset in assets}
        self.feature_idx = {asset: 0 for asset in assets}
        
    def process_ipc_stream(self, ipc_data: List[bytes]) -> List[FlowSignal]:
        """
        Process stream of IPC messages from Rust bridge.
        
        Args:
            ipc_data: List of raw byte messages
            
        Returns:
            List of generated FlowSignals
        """
        signals = []
        
        for data in ipc_data:
            msg = self.flow_parser.parse_ipc_message(data)
            if msg is None:
                continue
            
            asset = msg['asset']
            if asset not in self.analyzers:
                continue
            
            # Add to flow parser
            self.flow_parser.add_flow_event(
                msg['timestamp_ns'],
                msg['amount_usd'],
                msg['direction'],
                asset
            )
            
            # Check for whale
            analyzer = self.analyzers[asset]
            whale_alert = analyzer.detect_whale_transaction(
                msg['timestamp_ns'],
                asset,
                msg['amount_usd'],
                msg['direction'],
                "unknown", "unknown", "unknown"
            )
        
        return signals
    
    def generate_features(self, asset: str, 
                         current_flow: float,
                         price_change_pct: float) -> np.ndarray:
        """
        Generate feature vector for ML models.
        
        Args:
            asset: Asset symbol
            current_flow: Current period flow
            price_change_pct: Recent price change percentage
            
        Returns:
            Feature vector (10 dimensions)
        """
        analyzer = self.analyzers.get(asset)
        if analyzer is None:
            return np.zeros(10)
        
        # Calculate components
        flow_zscore = analyzer.calculate_flow_zscore(asset, current_flow)
        whale_count = len([a for a in analyzer.whale_alerts[-100:] 
                          if a.asset == asset])
        
        # Feature vector:
        # 0: Flow Z-score
        # 1: Flow direction sign
        # 2: Flow magnitude (normalized)
        # 3: Whale count
        # 4: Price-flow divergence
        # 5-9: Lagged flow values
        
        features = np.zeros(10)
        features[0] = flow_zscore
        features[1] = np.sign(current_flow)
        features[2] = min(abs(current_flow) / 1e7, 1.0)  # Normalize to $10M
        features[3] = min(whale_count / 10.0, 1.0)
        features[4] = flow_zscore * np.sign(price_change_pct)  # Divergence
        
        # Lagged flows (would need history in production)
        features[5:10] = flow_zscore * np.array([1.0, 0.8, 0.6, 0.4, 0.2])
        
        # Store in buffer
        idx = self.feature_idx[asset] % self.feature_buffer_size
        self.features[asset][idx] = features
        self.feature_idx[asset] += 1
        
        return features
    
    def get_alpha_signals(self) -> List[Dict]:
        """Get current alpha signals for all assets."""
        signals = []
        
        for asset in self.assets:
            analyzer = self.analyzers[asset]
            
            # Get recent flow (simplified)
            recent_flow = analyzer.flow_means[asset]
            whale_count = len(analyzer.whale_alerts[-50:])
            total_volume = sum(abs(a.amount_usd) for a in analyzer.whale_alerts[-50:])
            
            signal = analyzer.generate_flow_signal(asset, recent_flow, 
                                                   whale_count, total_volume)
            
            if signal.signal_direction != 0:
                signals.append({
                    'asset': signal.asset,
                    'direction': 'LONG' if signal.signal_direction > 0 else 'SHORT',
                    'flow_zscore': signal.net_flow_zscore,
                    'whale_count': signal.whale_count,
                    'confidence': signal.confidence,
                    'cumulative_flow': signal.cumulative_flow_1h
                })
        
        return signals


__all__ = [
    'OnChainFlowParser',
    'ExchangeFlowAnalyzer',
    'OnChainAlphaGenerator',
    'FlowSignal',
    'WhaleAlert',
    'FlowDirection'
]
