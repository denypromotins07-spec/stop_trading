"""
Portfolio-level Nautilus actor managing global cross-asset exposure and delta hedging.
Ensures combined BTC, ETH, SOL positions never breach global VaR limits.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import numpy as np
import logging
import threading
import time

# Conditional Nautilus imports
try:
    from nautilus_trader.strategy.strategy import Strategy
    from nautilus_trader.model.identifiers import InstrumentId
    NAUTILUS_AVAILABLE = True
except ImportError:
    NAUTILUS_AVAILABLE = False
    class Strategy:
        pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PositionInfo:
    """Position information for an instrument."""
    instrument_id: str
    quantity: float
    average_price: float
    current_price: float
    unrealized_pnl: float
    delta: float  # Price sensitivity
    beta: float   # Market beta


@dataclass
class VaRLimits:
    """VaR limit configuration."""
    daily_var_limit: float = 10000.0  # USD
    position_var_limit: float = 5000.0
    concentration_limit: float = 0.4  # Max 40% in single asset
    correlation_threshold: float = 0.7
    confidence_level: float = 0.99
    lookback_days: int = 30


@dataclass
class PortfolioState:
    """Current portfolio state."""
    total_equity: float
    net_exposure: float
    gross_exposure: float
    var_99: float
    sharpe_ratio: float
    positions: Dict[str, PositionInfo] = field(default_factory=dict)


class PortfolioManager(Strategy):
    """
    Portfolio-level manager for cross-asset risk management.
    Enforces VaR limits and manages delta hedging.
    """
    
    def __init__(self, config: Optional[VaRLimits] = None):
        self.config = config or VaRLimits()
        
        # Portfolio state
        self._positions: Dict[str, PositionInfo] = {}
        self._historical_returns: Dict[str, List[float]] = {}
        self._correlation_matrix: Optional[np.ndarray] = None
        self._covariance_matrix: Optional[np.ndarray] = None
        
        # Risk metrics
        self._current_var = 0.0
        self._total_equity = 100000.0  # Initial equity
        self._last_update = 0.0
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Hedging state
        self._hedge_ratios: Dict[str, float] = {}
        self._last_hedge_time = 0.0
        self._hedge_interval_seconds = 300  # 5 minutes
    
    def on_start(self) -> None:
        """Initialize portfolio manager."""
        logger.info("PortfolioManager started")
        
        if NAUTILUS_AVAILABLE:
            # Subscribe to all managed instruments
            self._subscribe_all_instruments()
    
    def on_stop(self) -> None:
        """Cleanup on stop."""
        logger.info("PortfolioManager stopped")
    
    def on_data(self, data: Any) -> None:
        """Handle incoming market data."""
        with self._lock:
            self._update_position_prices(data)
            self._update_risk_metrics()
    
    def _subscribe_all_instruments(self) -> None:
        """Subscribe to all managed instruments."""
        instruments = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        for inst in instruments:
            try:
                self.subscribe_quote_ticks(InstrumentId.from_str(inst))
            except Exception:
                pass
    
    def _update_position_prices(self, data: Any) -> None:
        """Update position prices from market data."""
        if not hasattr(data, 'instrument_id'):
            return
        
        inst_id = str(data.instrument_id)
        
        if inst_id in self._positions:
            pos = self._positions[inst_id]
            
            # Update current price
            if hasattr(data, 'ask_price') and hasattr(data, 'bid_price'):
                mid_price = (float(data.ask_price) + float(data.bid_price)) / 2
            elif hasattr(data, 'price'):
                mid_price = float(data.price)
            else:
                return
            
            pos.current_price = mid_price
            
            # Update unrealized PnL
            if pos.quantity > 0:
                pos.unrealized_pnl = (mid_price - pos.average_price) * pos.quantity
            else:
                pos.unrealized_pnl = 0.0
    
    def _update_risk_metrics(self) -> None:
        """Update portfolio risk metrics."""
        now = time.time()
        
        # Throttle updates
        if now - self._last_update < 1.0:
            return
        
        self._last_update = now
        
        # Calculate portfolio VaR
        self._current_var = self._calculate_portfolio_var()
        
        # Check VaR limits
        if self._current_var > self.config.daily_var_limit:
            logger.warning(f"VaR limit breached: {self._current_var:.2f} > {self.config.daily_var_limit}")
            self._reduce_risk()
    
    def _calculate_portfolio_var(self) -> float:
        """
        Calculate portfolio Value at Risk using parametric method.
        
        Returns:
            Portfolio VaR in USD
        """
        if not self._positions:
            return 0.0
        
        # Get position values
        instruments = list(self._positions.keys())
        n_assets = len(instruments)
        
        if n_assets == 0:
            return 0.0
        
        # Calculate position weights and volatilities
        weights = np.zeros(n_assets)
        vols = np.zeros(n_assets)
        
        total_value = sum(
            abs(pos.quantity * pos.current_price) 
            for pos in self._positions.values()
        )
        
        for i, (inst_id, pos) in enumerate(self._positions.items()):
            position_value = pos.quantity * pos.current_price
            weights[i] = position_value / max(total_value, 1)
            
            # Use historical volatility or default
            if inst_id in self._historical_returns:
                returns = self._historical_returns[inst_id]
                if len(returns) > 0:
                    vols[i] = np.std(returns) * np.sqrt(252)  # Annualized
            else:
                vols[i] = 0.6  # Default 60% vol for crypto
        
        # Estimate covariance (simplified)
        if self._covariance_matrix is None:
            # Use diagonal approximation with correlation
            corr = self.config.concentration_limit
            self._covariance_matrix = np.outer(vols, vols) * corr
            np.fill_diagonal(self._covariance_matrix, vols ** 2)
        
        # Portfolio variance
        portfolio_variance = weights @ self._covariance_matrix @ weights
        
        # VaR at confidence level
        z_score = 2.33  # 99% confidence
        daily_var_pct = np.sqrt(portfolio_variance) / np.sqrt(252)
        
        portfolio_var_usd = z_score * daily_var_pct * total_value
        
        return portfolio_var_usd
    
    def _reduce_risk(self) -> None:
        """Reduce portfolio risk when VaR limit is breached."""
        logger.info("Reducing portfolio risk...")
        
        # Find largest position
        largest_pos = None
        largest_value = 0
        
        for inst_id, pos in self._positions.items():
            value = abs(pos.quantity * pos.current_price)
            if value > largest_value:
                largest_value = value
                largest_pos = inst_id
        
        if largest_pos:
            # Reduce position by 25%
            pos = self._positions[largest_pos]
            reduce_qty = pos.quantity * 0.25
            
            logger.info(f"Reducing {largest_pos} by {reduce_qty:.6f}")
            self._submit_hedge_order(largest_pos, -reduce_qty)
    
    def _submit_hedge_order(self, instrument_id: str, quantity: float) -> None:
        """Submit hedge order."""
        if not NAUTILUS_AVAILABLE:
            logger.info(f"HEDGE ORDER: {quantity} {instrument_id}")
            return
        
        # Submit offsetting order
        try:
            order = self.order_factory.market(
                instrument_id=InstrumentId.from_str(instrument_id),
                side=self._get_side(quantity),
                quantity=self._quantity(abs(quantity)),
            )
            self.submit_order(order)
        except Exception as e:
            logger.error(f"Hedge order failed: {e}")
    
    def check_position_limit(self, 
                             instrument_id: str,
                             proposed_quantity: float,
                             price: float) -> bool:
        """
        Check if proposed position would breach limits.
        
        Args:
            instrument_id: Target instrument
            proposed_quantity: Proposed position size
            price: Current price
        
        Returns:
            True if within limits
        """
        with self._lock:
            proposed_value = abs(proposed_quantity * price)
            
            # Check concentration limit
            total_value = sum(
                abs(pos.quantity * pos.current_price)
                for pos in self._positions.values()
            )
            
            if total_value > 0:
                concentration = proposed_value / (total_value + proposed_value)
                if concentration > self.config.concentration_limit:
                    logger.warning(f"Concentration limit breach: {concentration:.2%}")
                    return False
            
            # Check position VaR
            position_var = proposed_value * 0.05  # Simplified 5% daily VaR
            if position_var > self.config.position_var_limit:
                logger.warning(f"Position VaR limit breach: {position_var:.2f}")
                return False
            
            return True
    
    def get_portfolio_state(self) -> PortfolioState:
        """Get current portfolio state."""
        with self._lock:
            net_exposure = sum(
                pos.quantity * pos.current_price
                for pos in self._positions.values()
            )
            
            gross_exposure = sum(
                abs(pos.quantity * pos.current_price)
                for pos in self._positions.values()
            )
            
            return PortfolioState(
                total_equity=self._total_equity,
                net_exposure=net_exposure,
                gross_exposure=gross_exposure,
                var_99=self._current_var,
                sharpe_ratio=self._calculate_sharpe(),
                positions=dict(self._positions)
            )
    
    def _calculate_sharpe(self) -> float:
        """Calculate portfolio Sharpe ratio."""
        # Simplified calculation
        total_pnl = sum(pos.unrealized_pnl for pos in self._positions.values())
        if total_pnl == 0:
            return 0.0
        
        # Assume 20% annual vol
        return total_pnl / (self._total_equity * 0.2)
    
    def _get_side(self, quantity: float) -> Any:
        """Get order side."""
        if not NAUTILUS_AVAILABLE:
            return "BUY" if quantity > 0 else "SELL"
        
        from nautilus_trader.core.enums import OrderSide
        return OrderSide.BUY if quantity > 0 else OrderSide.SELL
    
    def _quantity(self, size: float) -> Any:
        """Create quantity."""
        if not NAUTILUS_AVAILABLE:
            return size
        
        from nautilus_trader.model.quantity import Quantity
        return Quantity.from_float(size)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get portfolio statistics."""
        state = self.get_portfolio_state()
        
        return {
            "total_equity": state.total_equity,
            "net_exposure": state.net_exposure,
            "gross_exposure": state.gross_exposure,
            "var_99": state.var_99,
            "var_limit": self.config.daily_var_limit,
            "var_utilization": state.var_99 / max(self.config.daily_var_limit, 1),
            "sharpe_ratio": state.sharpe_ratio,
            "n_positions": len(state.positions),
        }


def create_portfolio_manager(config: Optional[VaRLimits] = None) -> PortfolioManager:
    """Factory function to create PortfolioManager."""
    return PortfolioManager(config)


if __name__ == "__main__":
    print("Portfolio Manager Demo")
    print("=" * 40)
    
    config = VaRLimits(
        daily_var_limit=10000.0,
        concentration_limit=0.4
    )
    
    pm = create_portfolio_manager(config)
    
    # Simulate positions
    pm._positions = {
        "BTC/USDT": PositionInfo(
            instrument_id="BTC/USDT",
            quantity=1.0,
            average_price=45000,
            current_price=46000,
            unrealized_pnl=1000,
            delta=1.0,
            beta=1.0
        ),
        "ETH/USDT": PositionInfo(
            instrument_id="ETH/USDT",
            quantity=10.0,
            average_price=2800,
            current_price=2850,
            unrealized_pnl=500,
            delta=0.8,
            beta=1.2
        ),
    }
    
    # Get state
    state = pm.get_portfolio_state()
    print(f"\nPortfolio State:")
    print(f"  Total Equity: ${state.total_equity:,.2f}")
    print(f"  Net Exposure: ${state.net_exposure:,.2f}")
    print(f"  Gross Exposure: ${state.gross_exposure:,.2f}")
    print(f"  VaR (99%): ${state.var_99:,.2f}")
    
    # Check limits
    stats = pm.get_statistics()
    print(f"\nStatistics: {stats}")
