"""
Core Nautilus Strategy Class consuming ML signals from MessageBus.
Translates probability scores into Limit/Market orders with Kelly Criterion sizing.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import numpy as np
import logging

# Nautilus Trader imports (conditional for compatibility)
try:
    from nautilus_trader.strategy.strategy import Strategy
    from nautilus_trader.model.data import Bar, QuoteTick, TradeTick
    from nautilus_trader.model.events import OrderFilled, OrderSubmitted
    from nautilus_trader.model.identifiers import InstrumentId, StrategyId
    from nautilus_trader.model.orders import LimitOrder, MarketOrder
    NAUTILUS_AVAILABLE = True
except ImportError:
    NAUTILUS_AVAILABLE = False
    # Mock classes for standalone testing
    class Strategy:
        pass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SignalData:
    """ML signal data structure."""
    signal_id: str
    instrument_id: str
    alpha_score: float
    probability: float
    confidence: float
    timestamp: float
    model_ids: List[str]


@dataclass
class PositionConfig:
    """Position sizing configuration."""
    max_position_size: float = 10.0
    min_order_size: float = 0.001
    kelly_fraction: float = 0.25  # Fractional Kelly
    max_kelly_bet: float = 0.1   # Maximum Kelly bet size
    use_kelly: bool = True


class AlphaStrategy(Strategy):
    """
    Core Nautilus strategy consuming ML signals.
    Translates alpha probabilities into executable orders.
    """
    
    def __init__(self, config: Optional[PositionConfig] = None):
        self.config = config or PositionConfig()
        
        # State
        self._positions: Dict[str, float] = {}  # instrument_id -> net position
        self._pending_signals: Dict[str, SignalData] = {}
        self._order_counts: Dict[str, int] = {}
        
        # Performance tracking
        self._total_pnl = 0.0
        self._total_trades = 0
        self._winning_trades = 0
    
    def on_start(self) -> None:
        """Called when strategy starts."""
        logger.info("AlphaStrategy started")
        
        if NAUTILUS_AVAILABLE:
            # Subscribe to data feeds
            self.subscribe_quote_ticks()
            self.subscribe_bar_types()
    
    def on_stop(self) -> None:
        """Called when strategy stops."""
        logger.info("AlphaStrategy stopped")
        
        # Close all positions on stop
        self._close_all_positions()
    
    def on_signal(self, signal: SignalData) -> None:
        """
        Handle incoming ML signal from MessageBus.
        
        Args:
            signal: SignalData from ML module
        """
        logger.debug(f"Received signal: {signal.signal_id}, alpha={signal.alpha_score:.4f}")
        
        # Store pending signal
        self._pending_signals[signal.instrument_id] = signal
        
        # Generate order based on signal
        self._process_signal(signal)
    
    def on_data(self, data: Any) -> None:
        """
        Handle incoming market data.
        
        Args:
            data: QuoteTick, TradeTick, or Bar
        """
        # Update internal state based on market data
        if hasattr(data, 'instrument_id'):
            instrument_id = str(data.instrument_id)
            
            # Check if we have a pending signal for this instrument
            if instrument_id in self._pending_signals:
                signal = self._pending_signals[instrument_id]
                
                # Could trigger execution based on market conditions
                # e.g., wait for favorable spread before submitting
    
    def on_fill(self, fill: Any) -> None:
        """
        Handle order fill events.
        
        Args:
            fill: OrderFilled event
        """
        self._total_trades += 1
        
        if hasattr(fill, 'pnl'):
            pnl = fill.pnl
            self._total_pnl += float(pnl)
            
            if pnl > 0:
                self._winning_trades += 1
        
        logger.info(f"Order filled. Total PnL: {self._total_pnl:.2f}, Trades: {self._total_trades}")
    
    def _process_signal(self, signal: SignalData) -> None:
        """
        Process ML signal and generate orders.
        
        Args:
            signal: SignalData to process
        """
        instrument_id = signal.instrument_id
        
        # Calculate position size using Kelly Criterion
        position_size = self._calculate_kelly_size(
            probability=signal.probability,
            confidence=signal.confidence
        )
        
        # Determine order direction
        if abs(position_size) < self.config.min_order_size:
            logger.debug(f"Position size too small: {position_size}")
            return
        
        # Get current position
        current_position = self._positions.get(instrument_id, 0.0)
        
        # Calculate target position
        target_position = position_size
        
        # Determine if we need to adjust position
        position_delta = target_position - current_position
        
        if abs(position_delta) < self.config.min_order_size:
            return
        
        # Submit order
        self._submit_order(instrument_id, position_delta, signal)
    
    def _calculate_kelly_size(self, 
                              probability: float,
                              confidence: float) -> float:
        """
        Calculate position size using Kelly Criterion.
        
        Kelly formula: f* = (p * b - q) / b
        where p = win probability, q = loss probability, b = odds
        
        Args:
            probability: Win probability from ML model
            confidence: Model confidence score
        
        Returns:
            Position size (positive = long, negative = short)
        """
        if not self.config.use_kelly:
            # Fixed size if Kelly disabled
            return self.config.max_position_size * 0.1 if probability > 0.5 else -self.config.max_position_size * 0.1
        
        # Convert probability to win/loss odds
        p = probability  # Win probability
        q = 1 - p        # Loss probability
        
        # Assume even odds (b = 1) for simplicity
        # In practice, this could be derived from expected return
        b = 1.0
        
        # Kelly fraction
        kelly_fraction = (p * b - q) / b
        
        # Apply confidence weighting
        kelly_fraction *= confidence
        
        # Apply fractional Kelly (risk reduction)
        kelly_fraction *= self.config.kelly_fraction
        
        # Clamp to maximum
        kelly_fraction = np.clip(kelly_fraction, -self.config.max_kelly_bet, self.config.max_kelly_bet)
        
        # Convert to position size
        position_size = kelly_fraction * self.config.max_position_size
        
        return position_size
    
    def _submit_order(self,
                      instrument_id: str,
                      quantity: float,
                      signal: SignalData) -> None:
        """
        Submit order to exchange.
        
        Args:
            instrument_id: Target instrument
            quantity: Order quantity (positive = buy, negative = sell)
            signal: Triggering signal
        """
        if not NAUTILUS_AVAILABLE:
            logger.warning(f"Would submit order: {quantity} {instrument_id}")
            return
        
        # Determine order type based on urgency
        is_urgent = abs(signal.alpha_score) > 0.7
        
        if is_urgent:
            # Market order for urgent signals
            order = self.order_factory.market(
                instrument_id=self._instrument_id(instrument_id),
                side=self._get_side(quantity),
                quantity=self._quantity(abs(quantity)),
            )
        else:
            # Limit order for normal signals
            # Could use mid-price or better based on aggression
            order = self.order_factory.limit(
                instrument_id=self._instrument_id(instrument_id),
                side=self._get_side(quantity),
                quantity=self._quantity(abs(quantity)),
                price=self._get_limit_price(instrument_id, quantity),
            )
        
        self.submit_order(order)
        
        # Track order count
        self._order_counts[instrument_id] = self._order_counts.get(instrument_id, 0) + 1
        
        logger.info(f"Order submitted: {quantity} @ {instrument_id}")
    
    def _get_side(self, quantity: float) -> Any:
        """Get order side from quantity."""
        if not NAUTILUS_AVAILABLE:
            return "BUY" if quantity > 0 else "SELL"
        
        from nautilus_trader.core.enums import OrderSide
        return OrderSide.BUY if quantity > 0 else OrderSide.SELL
    
    def _quantity(self, size: float) -> Any:
        """Create quantity object."""
        if not NAUTILUS_AVAILABLE:
            return size
        
        from nautilus_trader.model.quantity import Quantity
        return Quantity.from_int(int(size * 1000)) if size < 1 else Quantity.from_float(size)
    
    def _instrument_id(self, instrument_id: str) -> Any:
        """Create instrument ID."""
        if not NAUTILUS_AVAILABLE:
            return instrument_id
        
        return InstrumentId.from_str(instrument_id)
    
    def _get_limit_price(self, instrument_id: str, quantity: float) -> Any:
        """Calculate limit price for order."""
        # Could use recent tick data or fair value estimate
        # For now, return placeholder
        if not NAUTILUS_AVAILABLE:
            return 100.0
        
        from nautilus_trader.model.price import Price
        return Price.from_str("100.00")
    
    def _close_all_positions(self) -> None:
        """Close all open positions."""
        for instrument_id, position in list(self._positions.items()):
            if position != 0:
                self._submit_order(
                    instrument_id,
                    -position,
                    SignalData(
                        signal_id="close_all",
                        instrument_id=instrument_id,
                        alpha_score=0,
                        probability=0.5,
                        confidence=1.0,
                        timestamp=0,
                        model_ids=[]
                    )
                )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get strategy statistics."""
        win_rate = self._winning_trades / max(self._total_trades, 1)
        
        return {
            "total_pnl": self._total_pnl,
            "total_trades": self._total_trades,
            "winning_trades": self._winning_trades,
            "win_rate": win_rate,
            "open_positions": len([p for p in self._positions.values() if p != 0]),
            "pending_signals": len(self._pending_signals),
        }


def create_alpha_strategy(config: Optional[PositionConfig] = None) -> AlphaStrategy:
    """
    Factory function to create AlphaStrategy.
    
    Args:
        config: Position configuration
    
    Returns:
        AlphaStrategy instance
    """
    return AlphaStrategy(config)


if __name__ == "__main__":
    print("Alpha Strategy Demo")
    print("=" * 40)
    
    # Create strategy
    config = PositionConfig(
        max_position_size=1.0,
        kelly_fraction=0.25,
        use_kelly=True
    )
    
    strategy = create_alpha_strategy(config)
    
    # Simulate signal
    signal = SignalData(
        signal_id="test_signal_1",
        instrument_id="BTC/USDT",
        alpha_score=0.65,
        probability=0.72,
        confidence=0.85,
        timestamp=time.time(),
        model_ids=["trend_model_v1", "momentum_model_v2"]
    )
    
    # Import time here to avoid issues
    import time
    signal.timestamp = time.time()
    
    print(f"\nProcessing signal:")
    print(f"  Instrument: {signal.instrument_id}")
    print(f"  Alpha: {signal.alpha_score:.4f}")
    print(f"  Probability: {signal.probability:.4f}")
    print(f"  Confidence: {signal.confidence:.4f}")
    
    # Calculate Kelly size
    kelly_size = strategy._calculate_kelly_size(signal.probability, signal.confidence)
    print(f"\nKelly position size: {kelly_size:.6f}")
    
    # Get statistics
    stats = strategy.get_statistics()
    print(f"\nStatistics: {stats}")
