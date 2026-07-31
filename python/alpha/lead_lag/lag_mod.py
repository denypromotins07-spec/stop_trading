"""
Lead-Lag Module Root.
Routes detected "leader" signals to "lagger" Nautilus strategy actors for front-running convergence.
Integrates Hayashi-Yoshida and Thermal Optimal detectors with execution routing.
Memory-efficient design with strict Ray actor configuration.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import time


class SignalPriority(Enum):
    """Priority levels for lead-lag signals."""
    CRITICAL = 1   # Immediate execution
    HIGH = 2       # Execute within milliseconds
    MEDIUM = 3     # Standard execution
    LOW = 4        # Opportunistic


@dataclass
class LeadLagSignal:
    """Container for lead-lag trading signal."""
    timestamp_ns: int
    leader_asset: str
    follower_asset: str
    lag_ms: float
    correlation: float
    confidence: float
    direction: int  # 1=positive, -1=negative
    priority: SignalPriority
    expected_move_bps: float
    signal_id: str


class LeadLagRouter:
    """
    Routes lead-lag signals to appropriate Nautilus strategy actors.
    Implements signal filtering and prioritization.
    """
    
    def __init__(self, 
                 min_confidence: float = 0.3,
                 min_lag_ms: float = 5.0,
                 max_pending_signals: int = 100):
        """
        Args:
            min_confidence: Minimum confidence threshold for signals
            min_lag_ms: Minimum detectable lag in milliseconds
            max_pending_signals: Maximum pending signals in queue
        """
        self.min_confidence = min_confidence
        self.min_lag_ms = min_lag_ms
        self.max_pending_signals = max_pending_signals
        
        # Signal queues by priority
        self.signal_queues = {
            SignalPriority.CRITICAL: [],
            SignalPriority.HIGH: [],
            SignalPriority.MEDIUM: [],
            SignalPriority.LOW: []
        }
        
        # Processed signal tracking (for deduplication)
        self.processed_signals = {}
        self.signal_ttl_ns = int(5e9)  # 5 second TTL
        
        # Strategy actor references
        self.strategy_actors = {}
        
        # Statistics
        self.stats = {
            'signals_received': 0,
            'signals_routed': 0,
            'signals_filtered': 0,
            'avg_latency_ns': 0
        }
        
    def receive_signal(self, 
                       leader: str, 
                       follower: str,
                       lag_ms: float,
                       correlation: float,
                       confidence: float,
                       direction: int,
                       expected_move_bps: float = 10.0) -> Optional[LeadLagSignal]:
        """
        Receive a new lead-lag signal and route it.
        
        Args:
            leader: Leading asset symbol
            follower: Following asset symbol
            lag_ms: Detected lag in milliseconds
            correlation: Cross-correlation strength
            confidence: Signal confidence [0, 1]
            direction: Direction of relationship (1 or -1)
            expected_move_bps: Expected move in basis points
            
        Returns:
            LeadLagSignal if routed, None if filtered
        """
        timestamp_ns = time.time_ns()
        
        # Filter by confidence
        if confidence < self.min_confidence:
            self.stats['signals_filtered'] += 1
            return None
        
        # Filter by lag (too small = noise, too large = stale)
        if lag_ms < self.min_lag_ms or lag_ms > 5000:
            self.stats['signals_filtered'] += 1
            return None
        
        # Check for duplicate/recent signal
        signal_key = f"{leader}_{follower}"
        if signal_key in self.processed_signals:
            last_time = self.processed_signals[signal_key]
            if timestamp_ns - last_time < self.signal_ttl_ns:
                self.stats['signals_filtered'] += 1
                return None
        
        # Determine priority based on confidence and lag
        priority = self._determine_priority(confidence, lag_ms, expected_move_bps)
        
        # Create signal
        signal = LeadLagSignal(
            timestamp_ns=timestamp_ns,
            leader_asset=leader,
            follower_asset=follower,
            lag_ms=lag_ms,
            correlation=correlation,
            confidence=confidence,
            direction=direction,
            priority=priority,
            expected_move_bps=expected_move_bps,
            signal_id=f"{leader}_{follower}_{timestamp_ns}"
        )
        
        # Add to queue
        self.signal_queues[priority].append(signal)
        
        # Trim queue if needed
        if len(self.signal_queues[priority]) > self.max_pending_signals:
            self.signal_queues[priority].pop(0)
        
        # Update processed tracking
        self.processed_signals[signal_key] = timestamp_ns
        
        # Clean old processed signals
        self._cleanup_processed_signals(timestamp_ns)
        
        self.stats['signals_received'] += 1
        self.stats['signals_routed'] += 1
        
        return signal
    
    def _determine_priority(self, confidence: float, lag_ms: float, 
                           expected_move_bps: float) -> SignalPriority:
        """Determine signal priority based on metrics."""
        score = confidence * 2 + (expected_move_bps / 50.0)
        
        if score > 2.5 and lag_ms < 50:
            return SignalPriority.CRITICAL
        elif score > 1.5 and lag_ms < 100:
            return SignalPriority.HIGH
        elif score > 1.0:
            return SignalPriority.MEDIUM
        else:
            return SignalPriority.LOW
    
    def _cleanup_processed_signals(self, current_ns: int):
        """Remove expired processed signal entries."""
        expired_keys = [
            k for k, ts in self.processed_signals.items()
            if current_ns - ts > self.signal_ttl_ns
        ]
        for key in expired_keys:
            del self.processed_signals[key]
    
    def get_next_signals(self, n: int = 10) -> List[LeadLagSignal]:
        """
        Get next N signals to process, ordered by priority.
        
        Args:
            n: Maximum number of signals to return
            
        Returns:
            List of signals in priority order
        """
        result = []
        
        for priority in [SignalPriority.CRITICAL, SignalPriority.HIGH, 
                         SignalPriority.MEDIUM, SignalPriority.LOW]:
            queue = self.signal_queues[priority]
            take = min(n - len(result), len(queue))
            result.extend(queue[:take])
            
            if len(result) >= n:
                break
        
        return result
    
    def acknowledge_signal(self, signal_id: str, latency_ns: int):
        """Acknowledge that a signal was processed."""
        for priority, queue in self.signal_queues.items():
            self.signal_queues[priority] = [
                s for s in queue if s.signal_id != signal_id
            ]
        
        # Update latency statistics
        total_count = self.stats.get('total_latency_samples', 0)
        current_avg = self.stats['avg_latency_ns']
        new_avg = (current_avg * total_count + latency_ns) / (total_count + 1)
        
        self.stats['avg_latency_ns'] = new_avg
        self.stats['total_latency_samples'] = total_count + 1
    
    def register_strategy_actor(self, strategy_name: str, actor_ref: Any):
        """Register a Nautilus strategy actor for signal routing."""
        self.strategy_actors[strategy_name] = actor_ref
    
    async def route_to_strategies_async(self):
        """Route pending signals to registered strategy actors."""
        if not self.strategy_actors:
            return
        
        signals = self.get_next_signals(n=20)
        
        for signal in signals:
            # Find relevant strategy actors
            for strategy_name, actor in self.strategy_actors.items():
                if signal.follower_asset in strategy_name or signal.leader_asset in strategy_name:
                    try:
                        # Send signal to actor
                        await actor.receive_lead_lag_signal.remote(signal)
                        self.acknowledge_signal(signal.signal_id, 
                                                time.time_ns() - signal.timestamp_ns)
                    except Exception as e:
                        pass  # Log error but continue
    
    def get_statistics(self) -> Dict:
        """Get router statistics."""
        queue_lengths = {p.name: len(q) for p, q in self.signal_queues.items()}
        
        return {
            **self.stats,
            'queue_lengths': queue_lengths,
            'total_pending': sum(len(q) for q in self.signal_queues.values()),
            'processed_cache_size': len(self.processed_signals)
        }


class NautilusIntegrationLayer:
    """
    Integration layer between lead-lag detection and Nautilus Trader.
    Converts signals to Nautilus-compatible order commands.
    """
    
    def __init__(self, router: LeadLagRouter):
        """
        Args:
            router: LeadLagRouter instance
        """
        self.router = router
        self.position_limits = {}  # Per-asset position limits
        self.risk_parameters = {
            'max_position_pct': 0.05,
            'stop_loss_bps': 50,
            'take_profit_bps': 100
        }
    
    def generate_nautilus_command(self, signal: LeadLagSignal,
                                   current_price: float) -> Optional[Dict]:
        """
        Generate Nautilus order command from lead-lag signal.
        
        Args:
            signal: Lead-lag signal
            current_price: Current price of follower asset
            
        Returns:
            Nautilus command dictionary or None
        """
        # Calculate position size based on confidence and expected move
        confidence_factor = signal.confidence
        move_factor = signal.expected_move_bps / 100.0
        
        # Risk-adjusted position size
        position_size_pct = min(
            self.risk_parameters['max_position_pct'] * confidence_factor * move_factor,
            self.risk_parameters['max_position_pct']
        )
        
        # Determine order side based on leader movement and direction
        # This would need real-time leader price data
        side = "BUY" if signal.direction > 0 else "SELL"
        
        # Calculate stop loss and take profit levels
        stop_loss_bps = self.risk_parameters['stop_loss_bps']
        take_profit_bps = self.risk_parameters['take_profit_bps']
        
        if side == "BUY":
            stop_loss_price = current_price * (1 - stop_loss_bps / 10000)
            take_profit_price = current_price * (1 + take_profit_bps / 10000)
        else:
            stop_loss_price = current_price * (1 + stop_loss_bps / 10000)
            take_profit_price = current_price * (1 - take_profit_bps / 10000)
        
        command = {
            'instrument_id': f"{signal.follower_asset}/USD",
            'order_type': 'MARKET',
            'side': side,
            'quantity_pct': position_size_pct,
            'time_in_force': 'IOC',
            'tags': {
                'strategy': 'lead_lag',
                'signal_id': signal.signal_id,
                'leader': signal.leader_asset,
                'lag_ms': signal.lag_ms,
                'confidence': signal.confidence
            },
            'risk_params': {
                'stop_loss_price': stop_loss_price,
                'take_profit_price': take_profit_price
            }
        }
        
        return command
    
    def set_position_limit(self, asset: str, limit_pct: float):
        """Set position limit for an asset."""
        self.position_limits[asset] = limit_pct
    
    def check_risk_limits(self, asset: str, proposed_size_pct: float) -> bool:
        """Check if proposed trade is within risk limits."""
        limit = self.position_limits.get(asset, self.risk_parameters['max_position_pct'])
        return proposed_size_pct <= limit


# Factory function for creating configured systems
def create_lead_lag_system(pairs: List[Tuple[str, str]], 
                           mode: str = 'integrated',
                           **kwargs) -> Tuple[Any, LeadLagRouter]:
    """
    Factory function to create complete lead-lag system.
    
    Args:
        pairs: List of (asset_x, asset_y) tuples to monitor
        mode: 'integrated' for full system, 'router_only' for just routing
        **kwargs: Additional configuration
        
    Returns:
        Tuple of (detector/monitor, router)
    """
    from .hayashi_yoshida import LeadLagMonitor
    from .thermal_optimal import MultiAssetLeadLagTracker
    
    router = LeadLagRouter(**kwargs.get('router_kwargs', {}))
    
    if mode == 'router_only':
        return router, router
    
    # Create comprehensive monitoring system
    assets = set()
    for pair in pairs:
        assets.add(pair[0])
        assets.add(pair[1])
    
    assets = list(assets)
    
    # Use thermal optimal detector for more robust signals
    tracker = MultiAssetLeadLagTracker(assets, **kwargs.get('detector_kwargs', {}))
    
    return tracker, router


__all__ = [
    'LeadLagRouter',
    'NautilusIntegrationLayer',
    'LeadLagSignal',
    'SignalPriority',
    'create_lead_lag_system'
]
