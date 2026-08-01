"""
Market Making Module Root - Feeds ML market making signals to Nautilus quoting engine.
Integrates adverse selection prediction and queue-aware RL for optimal quote management.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

from .adverse_selector import (
    AdverseSelectorEngine, 
    AdverseSelectorModel,
    AdverseSelectionPrediction,
    get_adverse_selector
)
from .queue_aware_rl import (
    QueueAwareRLAgent,
    QueueState,
    RLAction,
    get_queue_rl_agent
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QuoteAction(Enum):
    """Possible quoting actions."""
    PLACE = "place"
    UPDATE = "update"
    CANCEL = "cancel"
    WIDEN = "widen"
    HOLD = "hold"


@dataclass
class QuoteSignal:
    """Signal for Nautilus quoting engine."""
    timestamp: float
    symbol: str
    side: str  # 'bid' or 'ask'
    action: QuoteAction
    price: float
    size: float
    priority: int  # 0-100, higher = more urgent
    confidence: float
    adverse_prob: float
    queue_recommendation: Optional[str] = None
    reasoning: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "symbol": self.symbol,
            "side": self.side,
            "action": self.action.value,
            "price": self.price,
            "size": self.size,
            "priority": self.priority,
            "confidence": self.confidence,
            "adverse_prob": self.adverse_prob,
            "queue_recommendation": self.queue_recommendation,
            "reasoning": self.reasoning
        }


@dataclass
class MarketMakingState:
    """Current state of market making system."""
    symbol: str
    mid_price: float
    bid_price: float
    ask_price: float
    spread: float
    bid_size: float
    ask_size: float
    inventory: float
    inventory_risk: float
    volatility: float
    queue_position_bid: int
    queue_position_ask: int
    recent_fill_rate: float
    
    def to_queue_state(self, side: str) -> QueueState:
        """Convert to QueueState for RL agent."""
        is_bid = side == 'bid'
        return QueueState(
            queue_position=self.queue_position_bid if is_bid else self.queue_position_ask,
            queue_size_ahead=int(self.bid_size if is_bid else self.ask_size),
            queue_size_behind=int(self.ask_size if is_bid else self.bid_size),
            bid_ask_imbalance=(self.bid_size - self.ask_size) / (self.bid_size + self.ask_size + 1e-6),
            recent_cancellations=0,  # Would track separately
            recent_insertions=0,
            depletion_rate=1.0 - self.recent_fill_rate,
            insertion_rate=self.recent_fill_rate,
            price_momentum=0.0,  # Would calculate from price history
            volatility=self.volatility,
            spread=self.spread,
            time_in_queue=0.0  # Would track per order
        )


class MarketMakingModule:
    """
    Central module for ML-driven market making.
    Coordinates adverse selection prediction and queue RL for optimal quoting.
    """
    
    def __init__(self,
                 adverse_model_path: Optional[str] = None,
                 rl_policy_path: Optional[str] = None,
                 max_inventory: float = 100.0,
                 risk_aversion: float = 0.5):
        """
        Initialize market making module.
        
        Args:
            adverse_model_path: Path to trained adverse selection model
            rl_policy_path: Path to trained RL policy
            max_inventory: Maximum allowed inventory position
            risk_aversion: Risk aversion parameter for pricing
        """
        self.max_inventory = max_inventory
        self.risk_aversion = risk_aversion
        
        # Initialize components
        self.adverse_engine = get_adverse_selector(adverse_model_path)
        self.rl_agent = get_queue_rl_agent()
        
        if rl_policy_path:
            self.rl_agent.load_policy(rl_policy_path)
        
        # State tracking
        self._states: Dict[str, MarketMakingState] = {}
        self._active_signals: Dict[str, QuoteSignal] = {}
        self._signal_history: List[QuoteSignal] = []
        
        self._is_running = False
    
    def update_state(self, state: MarketMakingState):
        """Update internal state for a symbol."""
        self._states[state.symbol] = state
    
    async def generate_quote_signal(self, symbol: str) -> Optional[QuoteSignal]:
        """
        Generate quoting signal based on ML models.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            QuoteSignal or None if no action needed
        """
        if symbol not in self._states:
            logger.warning(f"No state for symbol {symbol}")
            return None
        
        state = self._states[symbol]
        timestamp = asyncio.get_event_loop().time()
        
        signals = []
        
        # Process both sides
        for side in ['bid', 'ask']:
            signal = await self._generate_side_signal(state, side, timestamp)
            if signal:
                signals.append(signal)
        
        # Return highest priority signal
        if signals:
            best_signal = max(signals, key=lambda s: s.priority)
            self._active_signals[f"{symbol}_{best_signal.side}"] = best_signal
            self._signal_history.append(best_signal)
            
            # Limit history
            if len(self._signal_history) > 1000:
                self._signal_history.pop(0)
            
            return best_signal
        
        return None
    
    async def _generate_side_signal(self, state: MarketMakingState, 
                                     side: str, timestamp: float) -> Optional[QuoteSignal]:
        """Generate signal for one side of the book."""
        is_bid = side == 'bid'
        
        # Get adverse selection prediction
        adverse_pred = await self.adverse_engine.predict_adverse_selection(
            symbol=state.symbol,
            side='buy' if is_bid else 'sell'
        )
        
        # Get queue recommendation
        queue_state = state.to_queue_state(side)
        queue_rec = self.rl_agent.get_action_recommendation(queue_state)
        
        # Determine action based on both signals
        action, priority, reasoning = self._determine_action(
            adverse_pred, queue_rec, state, side
        )
        
        if action == QuoteAction.HOLD:
            return None
        
        # Calculate optimal price and size
        price, size = self._calculate_quote_params(state, side, adverse_pred.probability_adverse)
        
        return QuoteSignal(
            timestamp=timestamp,
            symbol=state.symbol,
            side=side,
            action=action,
            price=price,
            size=size,
            priority=priority,
            confidence=queue_rec.confidence,
            adverse_prob=adverse_pred.probability_adverse,
            queue_recommendation=queue_rec.action_type,
            reasoning=reasoning
        )
    
    def _determine_action(self, adverse_pred: AdverseSelectionPrediction,
                          queue_rec: RLAction, state: MarketMakingState,
                          side: str) -> Tuple[QuoteAction, int, str]:
        """Determine quoting action based on signals."""
        adverse_prob = adverse_pred.probability_adverse
        queue_action = queue_rec.action_type
        
        reasons = []
        priority = 50  # Default medium priority
        
        # High adverse selection risk
        if adverse_prob > 0.7:
            if queue_action == 'cancel':
                return QuoteAction.CANCEL, 90, "High adverse risk + cancel recommendation"
            elif queue_action in ['reprice_better', 'jump_queue']:
                return QuoteAction.WIDEN, 80, "Widening due to adverse selection risk"
            reasons.append(f"high adverse prob ({adverse_prob:.2f})")
            priority += 20
        
        # Queue dynamics
        if queue_action == 'cancel':
            return QuoteAction.CANCEL, 85, f"Queue cancel: {queue_rec.reasoning}"
        elif queue_action == 'jump_queue':
            reasons.append("queue jump recommended")
            priority += 15
        elif queue_action == 'reprice_better':
            reasons.append("improving queue position")
            priority += 10
        elif queue_action == 'reprice_worse':
            reasons.append("accepting worse queue for better price")
        
        # Inventory management
        inv_ratio = abs(state.inventory) / self.max_inventory
        if inv_ratio > 0.8:
            # Reduce inventory
            if (isinstance(side, str) and 
                ((side == 'bid' and state.inventory > 0) or 
                 (side == 'ask' and state.inventory < 0))):
                return QuoteAction.CANCEL, 95, f"High inventory ({inv_ratio:.1%})"
            reasons.append(f"high inventory ({inv_ratio:.1%})")
            priority += 25
        
        # Normal operation
        if not reasons:
            reasons.append("normal quoting conditions")
        
        return QuoteAction.UPDATE, priority, "; ".join(reasons)
    
    def _calculate_quote_params(self, state: MarketMakingState, 
                                 side: str,
                                 adverse_prob: float) -> Tuple[float, float]:
        """Calculate optimal price and size."""
        is_bid = side == 'bid'
        
        # Base price from mid
        base_spread = state.spread / 2
        
        # Adjust spread for adverse selection
        adverse_adjustment = adverse_prob * state.spread * 0.5
        
        # Adjust for inventory
        inventory_adjustment = (state.inventory / self.max_inventory) * state.spread * self.risk_aversion
        
        if is_bid:
            price = state.mid_price - base_spread - adverse_adjustment + inventory_adjustment
        else:
            price = state.mid_price + base_spread + adverse_adjustment + inventory_adjustment
        
        # Size based on risk
        base_size = 10.0  # Would be configurable
        size_reduction = adverse_prob * 0.5  # Reduce size when adverse risk high
        size = base_size * (1 - size_reduction)
        
        # Further reduce if near inventory limits
        inv_ratio = abs(state.inventory) / self.max_inventory
        if inv_ratio > 0.5:
            size *= (1 - inv_ratio)
        
        return round(price, 2), round(max(0.1, size), 2)
    
    def get_active_signals(self) -> Dict[str, QuoteSignal]:
        """Get all active signals."""
        return self._active_signals.copy()
    
    def get_module_stats(self) -> Dict[str, Any]:
        """Get module statistics."""
        if not self._signal_history:
            return {"status": "no_signals"}
        
        recent_signals = self._signal_history[-100:]
        
        action_counts = {}
        for sig in recent_signals:
            action = sig.action.value
            action_counts[action] = action_counts.get(action, 0) + 1
        
        avg_adverse = np.mean([s.adverse_prob for s in recent_signals])
        avg_priority = np.mean([s.priority for s in recent_signals])
        
        return {
            "total_signals": len(self._signal_history),
            "active_signals": len(self._active_signals),
            "recent_actions": action_counts,
            "avg_adverse_prob": float(avg_adverse),
            "avg_priority": float(avg_priority),
            "adverse_engine_status": self.adverse_engine.get_current_risk_summary(),
            "rl_agent_status": self.rl_agent.get_training_stats()
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Return module health status."""
        return {
            "running": self._is_running,
            "symbols_tracked": len(self._states),
            "active_signals": len(self._active_signals),
            "adverse_engine": "initialized",
            "rl_agent": "initialized"
        }


# Module singleton
_mm_module: Optional[MarketMakingModule] = None


def get_market_making_module(**kwargs) -> MarketMakingModule:
    """Get or create the global market making module."""
    global _mm_module
    
    if _mm_module is None:
        _mm_module = MarketMakingModule(**kwargs)
        logger.info("Created market making module")
    
    return _mm_module


async def initialize_mm_module(adverse_model_path: Optional[str] = None,
                               rl_policy_path: Optional[str] = None) -> MarketMakingModule:
    """Initialize the market making module."""
    module = get_market_making_module(
        adverse_model_path=adverse_model_path,
        rl_policy_path=rl_policy_path
    )
    module._is_running = True
    return module


if __name__ == "__main__":
    # Test the market making module
    print("Testing Market Making Module...")
    
    module = MarketMakingModule()
    
    # Create test state
    test_state = MarketMakingState(
        symbol="BTC/USD",
        mid_price=50000.0,
        bid_price=49995.0,
        ask_price=50005.0,
        spread=10.0,
        bid_size=100.0,
        ask_size=120.0,
        inventory=25.0,
        inventory_risk=0.25,
        volatility=0.02,
        queue_position_bid=15,
        queue_position_ask=20,
        recent_fill_rate=0.3
    )
    
    module.update_state(test_state)
    
    # Add some trades to adverse selector
    import time
    for i in range(50):
        module.adverse_engine.add_trade(
            timestamp=time.time() - (50-i)*0.1,
            price=test_state.mid_price + np.random.randn() * 5,
            size=np.random.exponential(10),
            side=np.random.choice(['buy', 'sell']),
            aggressor=np.random.choice(['buyer', 'seller'])
        )
    
    # Calculate features
    module.adverse_engine.calculate_features(
        current_price=test_state.mid_price,
        bid_depth_total=test_state.bid_size,
        ask_depth_total=test_state.ask_size,
        queue_position=test_state.queue_position_bid
    )
    
    # Generate signals
    print("\nGenerating quote signals...")
    
    async def run_test():
        signal = await module.generate_quote_signal("BTC/USD")
        
        if signal:
            print(f"\nSignal Generated:")
            print(f"  Symbol: {signal.symbol}")
            print(f"  Side: {signal.side}")
            print(f"  Action: {signal.action.value}")
            print(f"  Price: {signal.price}")
            print(f"  Size: {signal.size}")
            print(f"  Priority: {signal.priority}")
            print(f"  Adverse Prob: {signal.adverse_prob:.4f}")
            print(f"  Queue Rec: {signal.queue_recommendation}")
            print(f"  Reasoning: {signal.reasoning}")
        else:
            print("No signal generated (HOLD)")
        
        print(f"\nModule Stats: {module.get_module_stats()}")
    
    asyncio.run(run_test())
