"""
Ensemble Alpha Module Root.
Publishes final meta-labeled, regime-adjusted alpha signals to Nautilus Trader strategies.
Integrates Meta-Labeling filtering with HMM-based signal routing.
Memory-efficient design with strict look-ahead bias prevention.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import time


class SignalStatus(Enum):
    """Status of processed alpha signals."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTED = "executed"
    EXPIRED = "expired"


@dataclass
class FinalAlphaSignal:
    """Final processed signal ready for execution."""
    signal_id: str
    timestamp_ns: int
    asset: str
    instrument_id: str
    side: str  # BUY or SELL
    strength: float  # 0-1
    confidence: float  # 0-1
    meta_label_prob: float
    regime: str
    category: str
    status: SignalStatus
    nautilus_command: Optional[Dict]


class EnsembleAlphaEngine:
    """
    Main ensemble engine combining meta-labeling and regime routing.
    Produces final alpha signals for Nautilus execution.
    """
    
    def __init__(self, 
                 assets: List[str],
                 min_meta_prob: float = 0.5,
                 max_pending_signals: int = 500):
        """
        Args:
            assets: Assets to process
            min_meta_prob: Minimum meta-label probability for execution
            max_pending_signals: Maximum pending signals in memory
        """
        self.assets = assets
        self.min_meta_prob = min_meta_prob
        self.max_pending_signals = max_pending_signals
        
        # Initialize components
        from .meta_labeler import MetaLabeler
        from .signal_router import SignalRouter, SignalCategory
        
        self.meta_labeler = MetaLabeler()
        self.signal_router = SignalRouter()
        
        # Signal tracking
        self.pending_signals = {}
        self.executed_signals = []
        self.rejected_signals = []
        
        # Category mapping (would be set by signal source)
        self.signal_categories = {}
        
        # Statistics
        self.stats = {
            'signals_received': 0,
            'signals_approved': 0,
            'signals_rejected': 0,
            'by_meta_filter': 0,
            'by_regime_filter': 0,
            'total_executed_value': 0.0
        }
    
    def receive_primary_signal(self,
                               signal_id: str,
                               signal: Dict,
                               category: SignalCategory,
                               market_context: Dict) -> Optional[FinalAlphaSignal]:
        """
        Receive and process a primary alpha signal through the full pipeline.
        
        Pipeline:
        1. Validate signal
        2. Apply meta-labeling filter
        3. Apply regime-based weighting
        4. Generate Nautilus command if approved
        
        Args:
            signal_id: Unique signal identifier
            signal: Primary signal dictionary
            category: Signal category
            market_context: Current market context
            
        Returns:
            FinalAlphaSignal or None
        """
        timestamp_ns = signal.get('timestamp_ns', time.time_ns())
        asset = signal.get('asset', 'UNKNOWN')
        
        self.stats['signals_received'] += 1
        
        # Step 1: Meta-labeling
        meta_result = self.meta_labeler.process_primary_signal(
            signal_id=signal_id,
            primary_signal=signal,
            market_context=market_context,
            timestamp_ns=timestamp_ns
        )
        
        if meta_result is None:
            self.stats['signals_rejected'] += 1
            return None
        
        # Check meta-label threshold
        if meta_result.success_probability < self.min_meta_prob:
            self.stats['by_meta_filter'] += 1
            self.stats['signals_rejected'] += 1
            
            # Record rejection
            self.rejected_signals.append({
                'signal_id': signal_id,
                'reason': 'meta_label_low',
                'meta_prob': meta_result.success_probability,
                'timestamp_ns': timestamp_ns
            })
            
            return None
        
        # Step 2: Regime routing
        routed_signal = self.signal_router.route_signal(
            signal=signal,
            category=category,
            timestamp_ns=timestamp_ns
        )
        
        if routed_signal is None or not routed_signal.should_execute:
            self.stats['by_regime_filter'] += 1
            self.stats['signals_rejected'] += 1
            return None
        
        # Step 3: Generate final signal
        direction = signal.get('direction', 0)
        side = 'BUY' if direction > 0 else 'SELL'
        
        final_signal = FinalAlphaSignal(
            signal_id=signal_id,
            timestamp_ns=timestamp_ns,
            asset=asset,
            instrument_id=f"{asset}/USD",
            side=side,
            strength=routed_signal.final_score,
            confidence=routed_signal.adjusted_confidence,
            meta_label_prob=meta_result.success_probability,
            regime=routed_signal.regime.value,
            category=category.value,
            status=SignalStatus.APPROVED,
            nautilus_command=self._generate_nautilus_command(
                final_signal, routed_signal
            )
        )
        
        # Track pending signal
        self.pending_signals[signal_id] = final_signal
        
        # Trim if needed
        if len(self.pending_signals) > self.max_pending_signals:
            oldest_id = next(iter(self.pending_signals))
            self.pending_signals[oldest_id].status = SignalStatus.EXPIRED
            del self.pending_signals[oldest_id]
        
        self.stats['signals_approved'] += 1
        
        return final_signal
    
    def _generate_nautilus_command(self, 
                                    final_signal: FinalAlphaSignal,
                                    routed_signal: Any) -> Dict:
        """Generate Nautilus Trader command from final signal."""
        # Calculate position size based on strength and confidence
        base_size_pct = 0.02  # 2% base position
        size_multiplier = final_signal.strength * final_signal.confidence
        
        quantity_pct = min(base_size_pct * size_multiplier * 2, 0.10)  # Max 10%
        
        command = {
            'type': 'alpha_signal',
            'instrument_id': final_signal.instrument_id,
            'order_type': 'MARKET',
            'side': final_signal.side,
            'quantity_pct': quantity_pct,
            'time_in_force': 'IOC',
            'priority': routed_signal.execution_priority,
            'tags': {
                'strategy': 'ensemble_alpha',
                'signal_id': final_signal.signal_id,
                'category': final_signal.category,
                'regime': final_signal.regime,
                'meta_prob': final_signal.meta_label_prob
            },
            'metadata': {
                'strength': final_signal.strength,
                'confidence': final_signal.confidence,
                'timestamp_ns': final_signal.timestamp_ns
            }
        }
        
        return command
    
    def acknowledge_execution(self, 
                              signal_id: str, 
                              executed_value: float,
                              success: bool):
        """
        Acknowledge that a signal was executed.
        
        Args:
            signal_id: Signal identifier
            executed_value: USD value executed
            success: Whether execution was successful
        """
        if signal_id not in self.pending_signals:
            return
        
        signal = self.pending_signals[signal_id]
        signal.status = SignalStatus.EXECUTED if success else SignalStatus.REJECTED
        
        # Move to executed list
        self.executed_signals.append({
            'signal_id': signal_id,
            'asset': signal.asset,
            'side': signal.side,
            'value': executed_value,
            'success': success,
            'timestamp_ns': signal.timestamp_ns,
            'meta_prob': signal.meta_label_prob
        })
        
        # Update statistics
        if success:
            self.stats['total_executed_value'] += executed_value
        
        # Record outcome for meta-labeler training
        outcome = 1 if success else 0
        self.meta_labeler.record_outcome(signal_id, outcome)
        
        # Clean up
        del self.pending_signals[signal_id]
        
        # Trim executed history
        if len(self.executed_signals) > 10000:
            self.executed_signals = self.executed_signals[-10000:]
    
    def update_market_regime(self,
                             market_return: float,
                             volatility: float,
                             correlation: float):
        """Update HMM regime with latest market data."""
        self.signal_router.update_regime(
            market_return=market_return,
            volatility=volatility,
            correlation=correlation
        )
    
    def get_execution_queue(self) -> List[FinalAlphaSignal]:
        """Get ordered queue of signals ready for execution."""
        # Get routed execution queue
        routed_queue = self.signal_router.get_execution_queue()
        
        # Map back to final signals
        execution_list = []
        for routed in routed_queue:
            # Find corresponding final signal
            for signal in self.pending_signals.values():
                if signal.signal_id == routed.original_signal.get('signal_id'):
                    execution_list.append(signal)
                    break
        
        return execution_list
    
    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary."""
        meta_metrics = self.meta_labeler.get_performance_metrics()
        router_stats = self.signal_router.get_statistics()
        
        return {
            'engine_stats': self.stats,
            'meta_labeler': meta_metrics,
            'signal_router': router_stats,
            'pending_count': len(self.pending_signals),
            'executed_count': len(self.executed_signals),
            'rejected_count': len(self.rejected_signals)
        }
    
    def get_nautilus_commands(self) -> List[Dict]:
        """Get all pending Nautilus commands for execution."""
        commands = []
        
        for signal in self.get_execution_queue():
            if signal.nautilus_command is not None:
                commands.append(signal.nautilus_command)
        
        return commands


class NautilusMessagePublisher:
    """
    Publisher for sending alpha signals to Nautilus MessageBus.
    Compatible with Nautilus Trader's messaging system.
    """
    
    def __init__(self):
        self.subscribers = {}
        self.message_queue = []
        self.max_queue_size = 1000
        self.stats = {'published': 0, 'dropped': 0}
    
    def subscribe(self, strategy_id: str, callback: callable):
        """Subscribe a strategy to receive signals."""
        self.subscribers[strategy_id] = callback
    
    def publish_signal(self, signal: FinalAlphaSignal):
        """Publish signal to all subscribers."""
        message = {
            'type': 'alpha_signal',
            'signal_id': signal.signal_id,
            'instrument_id': signal.instrument_id,
            'side': signal.side,
            'strength': signal.strength,
            'confidence': signal.confidence,
            'metadata': {
                'category': signal.category,
                'regime': signal.regime,
                'meta_prob': signal.meta_label_prob,
                'timestamp_ns': signal.timestamp_ns
            }
        }
        
        self.message_queue.append(message)
        
        if len(self.message_queue) > self.max_queue_size:
            self.message_queue.pop(0)
            self.stats['dropped'] += 1
        
        # Notify subscribers
        for callback in self.subscribers.values():
            try:
                callback(message)
            except Exception:
                pass
        
        self.stats['published'] += 1
    
    def get_statistics(self) -> Dict:
        """Get publisher statistics."""
        return {
            **self.stats,
            'queue_size': len(self.message_queue),
            'subscriber_count': len(self.subscribers)
        }


# Factory function
def create_ensemble_system(assets: List[str], 
                           **kwargs) -> Tuple[EnsembleAlphaEngine, NautilusMessagePublisher]:
    """
    Factory function to create complete ensemble alpha system.
    
    Args:
        assets: List of assets to monitor
        **kwargs: Additional configuration
        
    Returns:
        Tuple of (engine, publisher)
    """
    engine = EnsembleAlphaEngine(assets, **kwargs)
    publisher = NautilusMessagePublisher()
    
    return engine, publisher


__all__ = [
    'EnsembleAlphaEngine',
    'NautilusMessagePublisher',
    'FinalAlphaSignal',
    'SignalStatus',
    'create_ensemble_system'
]
