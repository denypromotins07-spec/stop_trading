"""
DeFi TVL and Stablecoin Minting Momentum Tracker.
Identifies structural capital rotations from on-chain data.
Memory-efficient implementation with bounded arrays.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class RotationType(Enum):
    """Types of capital rotation detected."""
    STABLE_TO_RISK = "stable_to_risk"     # Stables -> Crypto (bullish)
    RISK_TO_STABLE = "risk_to_stable"     # Crypto -> Stables (bearish)
    L2_MIGRATION = "l2_migration"         # L1 -> L2 rotation
    DEFI_ROTATION = "defi_rotation"       # Between DeFi protocols
    CROSS_CHAIN = "cross_chain"           # Cross-chain bridge flow


@dataclass
class TVLMetrics:
    """TVL metrics for a protocol or chain."""
    timestamp_ns: int
    protocol: str
    tvl_usd: float
    tvl_change_24h: float
    tvl_change_7d: float
    volume_24h: float
    unique_users_24h: int


@dataclass
class StablecoinSignal:
    """Stablecoin minting/burning signal."""
    timestamp_ns: int
    stablecoin: str
    net_mint_usd: float
    mint_rate_7d: float
    z_score: float
    signal_type: RotationType
    confidence: float


class TVLTracker:
    """
    Tracks Total Value Locked across DeFi protocols and chains.
    Uses circular buffers for memory efficiency.
    """
    
    def __init__(self, 
                 protocols: List[str],
                 lookback_days: int = 30,
                 buffer_size: int = 1000):
        """
        Args:
            protocols: List of protocol/chain identifiers
            lookback_days: Historical lookback period
            buffer_size: Maximum data points per protocol
        """
        self.protocols = protocols
        self.lookback_days = lookback_days
        self.buffer_size = buffer_size
        
        # TVL storage per protocol
        self.tvl_history = {}
        self.timestamp_history = {}
        self.current_idx = {}
        
        for protocol in protocols:
            self.tvl_history[protocol] = np.zeros(buffer_size)
            self.timestamp_history[protocol] = np.zeros(buffer_size, dtype=np.int64)
            self.current_idx[protocol] = 0
    
    def update_tvl(self, protocol: str, timestamp_ns: int, tvl_usd: float):
        """
        Update TVL for a protocol.
        
        Args:
            protocol: Protocol identifier
            timestamp_ns: Timestamp in nanoseconds
            tvl_usd: TVL in USD
        """
        if protocol not in self.tvl_history:
            self.tvl_history[protocol] = np.zeros(self.buffer_size)
            self.timestamp_history[protocol] = np.zeros(self.buffer_size, dtype=np.int64)
            self.current_idx[protocol] = 0
        
        idx = self.current_idx[protocol] % self.buffer_size
        self.tvl_history[protocol][idx] = tvl_usd
        self.timestamp_history[protocol][idx] = timestamp_ns
        self.current_idx[protocol] += 1
    
    def get_tvl_change(self, protocol: str, hours: int) -> float:
        """
        Calculate TVL change over specified hours.
        
        Args:
            protocol: Protocol identifier
            hours: Time period in hours
            
        Returns:
            Percentage change
        """
        if protocol not in self.tvl_history:
            return 0.0
        
        current_idx = (self.current_idx[protocol] - 1) % self.buffer_size
        current_ts = self.timestamp_history[protocol][current_idx]
        current_tvl = self.tvl_history[protocol][current_idx]
        
        # Find historical point
        target_ns = int(hours * 3600 * 1e9)
        cutoff_ts = current_ts - target_ns
        
        # Search backwards for matching timestamp
        n_points = min(self.current_idx[protocol], self.buffer_size)
        historical_tvl = current_tvl
        
        for i in range(n_points):
            idx = (current_idx - i) % self.buffer_size
            if self.timestamp_history[protocol][idx] <= cutoff_ts:
                historical_tvl = self.tvl_history[protocol][idx]
                break
        
        if historical_tvl <= 0:
            return 0.0
        
        return (current_tvl - historical_tvl) / historical_tvl * 100
    
    def get_metrics(self, protocol: str) -> Optional[TVLMetrics]:
        """Get current TVL metrics for a protocol."""
        import time
        
        if protocol not in self.tvl_history or self.current_idx[protocol] == 0:
            return None
        
        idx = (self.current_idx[protocol] - 1) % self.buffer_size
        
        return TVLMetrics(
            timestamp_ns=time.time_ns(),
            protocol=protocol,
            tvl_usd=self.tvl_history[protocol][idx],
            tvl_change_24h=self.get_tvl_change(protocol, 24),
            tvl_change_7d=self.get_tvl_change(protocol, 168),
            volume_24h=0.0,  # Would need separate tracking
            unique_users_24h=0
        )


class StablecoinMomentumTracker:
    """
    Tracks stablecoin minting/burning for capital flow signals.
    Detects acceleration in stablecoin supply growth.
    """
    
    def __init__(self, 
                 stablecoins: List[str],
                 lookback_days: int = 30,
                 zscore_threshold: float = 2.0):
        """
        Args:
            stablecoins: List of stablecoin symbols
            lookback_days: Historical lookback
            zscore_threshold: Threshold for signal generation
        """
        self.stablecoins = stablecoins
        self.lookback_days = lookback_days
        self.zscore_threshold = zscore_threshold
        
        # Daily supply tracking
        self.daily_supply = {sc: np.zeros(lookback_days) for sc in stablecoins}
        self.day_idx = {sc: 0 for sc in stablecoins}
        
        # Statistics
        self.supply_means = {sc: 0.0 for sc in stablecoins}
        self.supply_stds = {sc: 1.0 for sc in stablecoins}
        self.supply_trends = {sc: 0.0 for sc in stablecoins}
    
    def update_supply(self, stablecoin: str, supply_usd: float, day_offset: int = 0):
        """
        Update stablecoin supply.
        
        Args:
            stablecoin: Stablecoin symbol
            supply_usd: Total supply in USD
            day_offset: Days since epoch (for alignment)
        """
        if stablecoin not in self.daily_supply:
            self.daily_supply[stablecoin] = np.zeros(self.lookback_days)
        
        idx = day_offset % self.lookback_days
        self.daily_supply[stablecoin][idx] = supply_usd
        self.day_idx[stablecoin] = max(self.day_idx[stablecoin], day_offset + 1)
        
        # Update statistics
        self._update_statistics(stablecoin)
    
    def _update_statistics(self, stablecoin: str):
        """Update supply statistics."""
        supply = self.daily_supply[stablecoin]
        valid_supply = supply[supply > 0] if np.any(supply > 0) else supply
        
        if len(valid_supply) > 5:
            self.supply_means[stablecoin] = np.mean(valid_supply)
            self.supply_stds[stablecoin] = np.std(valid_supply) + 1e-10
            
            # Calculate trend (simple linear regression slope)
            n = len(valid_supply)
            x = np.arange(n)
            if n > 2:
                slope = np.polyfit(x, valid_supply, 1)[0]
                self.supply_trends[stablecoin] = slope / self.supply_means[stablecoin]
    
    def calculate_mint_zscore(self, stablecoin: str, 
                              daily_mint_usd: float) -> float:
        """Calculate Z-score of daily mint amount."""
        if stablecoin not in self.supply_stds:
            return 0.0
        
        mean_mint = self.supply_means[stablecoin] * 0.01  # Assume ~1% daily mint
        std_mint = self.supply_stds[stablecoin] * 0.005
        
        if std_mint < 1e-10:
            return 0.0
        
        return (daily_mint_usd - mean_mint) / std_mint
    
    def generate_signal(self, stablecoin: str,
                       daily_mint_usd: float) -> Optional[StablecoinSignal]:
        """
        Generate signal from stablecoin minting data.
        
        Args:
            stablecoin: Stablecoin symbol
            daily_mint_usd: Net mint amount (positive = mint, negative = burn)
            
        Returns:
            StablecoinSignal or None
        """
        import time
        timestamp_ns = time.time_ns()
        
        z_score = self.calculate_mint_zscore(stablecoin, daily_mint_usd)
        
        # Calculate 7-day mint rate
        if stablecoin in self.supply_trends:
            mint_rate = self.supply_trends[stablecoin] * 7  # Weekly rate
        else:
            mint_rate = 0.0
        
        # Determine signal type
        if z_score > self.zscore_threshold:
            # Significant minting = new capital entering (bullish)
            signal_type = RotationType.STABLE_TO_RISK
            confidence = min(abs(z_score) / 3.0, 1.0)
        elif z_score < -self.zscore_threshold:
            # Significant burning = capital leaving (bearish)
            signal_type = RotationType.RISK_TO_STABLE
            confidence = min(abs(z_score) / 3.0, 1.0)
        else:
            signal_type = RotationType.CROSS_CHAIN
            confidence = abs(z_score) / self.zscore_threshold
        
        return StablecoinSignal(
            timestamp_ns=timestamp_ns,
            stablecoin=stablecoin,
            net_mint_usd=daily_mint_usd,
            mint_rate_7d=mint_rate,
            z_score=z_score,
            signal_type=signal_type,
            confidence=confidence
        )


class CapitalRotationDetector:
    """
    Detects capital rotations between asset classes and chains.
    Combines TVL and stablecoin data for structural alpha.
    """
    
    def __init__(self, 
                 protocols: List[str],
                 stablecoins: List[str]):
        """
        Args:
            protocols: Protocols/chains to track
            stablecoins: Stablecoins to monitor
        """
        self.tvl_tracker = TVLTracker(protocols)
        self.stablecoin_tracker = StablecoinMomentumTracker(stablecoins)
        
        # Rotation history
        self.rotation_buffer_size = 100
        self.rotations = []
    
    def detect_rotation(self) -> List[Dict]:
        """
        Detect active capital rotations.
        
        Returns:
            List of rotation signals
        """
        rotations = []
        
        # Check TVL movements
        for protocol in self.tvl_tracker.protocols:
            metrics = self.tvl_tracker.get_metrics(protocol)
            if metrics is None:
                continue
            
            # Large TVL increase
            if metrics.tvl_change_24h > 10:
                rotations.append({
                    'type': 'tvl_inflow',
                    'protocol': protocol,
                    'change_24h': metrics.tvl_change_24h,
                    'change_7d': metrics.tvl_change_7d,
                    'signal': 'BULLISH' if metrics.tvl_change_7d > 0 else 'NEUTRAL'
                })
            elif metrics.tvl_change_24h < -10:
                rotations.append({
                    'type': 'tvl_outflow',
                    'protocol': protocol,
                    'change_24h': metrics.tvl_change_24h,
                    'change_7d': metrics.tvl_change_7d,
                    'signal': 'BEARISH'
                })
        
        # Store rotations
        self.rotations.extend(rotations)
        if len(self.rotations) > self.rotation_buffer_size:
            self.rotations = self.rotations[-self.rotation_buffer_size:]
        
        return rotations
    
    def get_structural_alpha_signals(self) -> List[Dict]:
        """
        Get combined structural alpha signals.
        
        Returns:
            List of actionable signals
        """
        signals = []
        
        # Combine TVL and stablecoin signals
        rotations = self.detect_rotation()
        
        for rotation in rotations:
            if rotation['signal'] in ['BULLISH', 'BEARISH']:
                signals.append({
                    'type': 'capital_rotation',
                    'subtype': rotation['type'],
                    'target': rotation['protocol'],
                    'direction': rotation['signal'],
                    'strength': min(abs(rotation['change_24h']) / 20.0, 1.0),
                    'metadata': {
                        'change_24h': rotation['change_24h'],
                        'change_7d': rotation['change_7d']
                    }
                })
        
        return signals


__all__ = [
    'TVLTracker',
    'StablecoinMomentumTracker',
    'CapitalRotationDetector',
    'TVLMetrics',
    'StablecoinSignal',
    'RotationType'
]
