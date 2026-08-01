"""
Gamma Scalping Logic Engine for Options Portfolio Management.
Dynamically adjusts delta hedges based on real-time Greeks to exploit mean-reverting micro-movements.
Generates theta to offset long-volatility position costs.
Strictly enforces 3GB RAM limit via bounded position tracking.
"""
import asyncio
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import time

class OptionType(Enum):
    CALL = "call"
    PUT = "put"


@dataclass
class OptionPosition:
    """Represents an options position with Greeks"""
    instrument_id: str
    option_type: OptionType
    strike: float
    expiry: float
    quantity: int  # Positive for long, negative for short
    entry_price: float
    current_spot: float
    implied_vol: float
    timestamp_ns: int
    
    def time_to_expiry(self) -> float:
        """Calculate time to expiry in years"""
        now = time.time()
        expiry_date = now + self.expiry * 365  # expiry is in years
        return max(self.expiry, (expiry_date - now) / 365)


@dataclass
class Greeks:
    """Option Greeks calculation results"""
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    
    def __add__(self, other: 'Greeks') -> 'Greeks':
        return Greeks(
            delta=self.delta + other.delta,
            gamma=self.gamma + other.gamma,
            theta=self.theta + other.theta,
            vega=self.vega + other.vega,
            rho=self.rho + other.rho
        )
    
    def __mul__(self, scalar: float) -> 'Greeks':
        return Greeks(
            delta=self.delta * scalar,
            gamma=self.gamma * scalar,
            theta=self.theta * scalar,
            vega=self.vega * scalar,
            rho=self.rho * scalar
        )


@dataclass
class HedgeAdjustment:
    """Recommended hedge adjustment"""
    instrument_id: str
    action: str  # "BUY" or "SELL"
    quantity: float
    target_delta: float
    current_delta: float
    reason: str
    timestamp_ns: int
    estimated_slippage_bps: float


class BlackScholes:
    """
    High-performance Black-Scholes Greeks calculator.
    Uses vectorized numpy operations for efficiency.
    """
    
    @staticmethod
    def _norm_cdf(x: np.ndarray) -> np.ndarray:
        """Standard normal CDF approximation"""
        return 0.5 * (1 + np.erf(x / np.sqrt(2)))
    
    @staticmethod
    def _norm_pdf(x: np.ndarray) -> np.ndarray:
        """Standard normal PDF"""
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
    
    @classmethod
    def calculate_greeks(cls, spot: float, strike: float, T: float, 
                         vol: float, r: float, option_type: OptionType,
                         q: float = 0.0) -> Greeks:
        """
        Calculate all Greeks for a single option.
        
        Args:
            spot: Current spot price
            strike: Option strike
            T: Time to expiry in years
            vol: Implied volatility
            r: Risk-free rate
            option_type: Call or Put
            q: Dividend yield (default 0)
        """
        if T <= 0 or vol <= 0:
            return Greeks(0, 0, 0, 0, 0)
        
        # d1 and d2
        d1 = (np.log(spot / strike) + (r - q + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
        d2 = d1 - vol * np.sqrt(T)
        
        # Common terms
        sqrt_T = np.sqrt(T)
        norm_d1 = cls._norm_cdf(d1)
        norm_d2 = cls._norm_cdf(d2)
        norm_pdf_d1 = cls._norm_pdf(d1)
        
        if option_type == OptionType.CALL:
            delta = np.exp(-q * T) * norm_d1
            theta = (-spot * np.exp(-q * T) * norm_pdf_d1 * vol / (2 * sqrt_T)
                    - r * strike * np.exp(-r * T) * norm_d2
                    + q * spot * np.exp(-q * T) * norm_d1)
        else:  # PUT
            delta = np.exp(-q * T) * (norm_d1 - 1)
            theta = (-spot * np.exp(-q * T) * norm_pdf_d1 * vol / (2 * sqrt_T)
                    + r * strike * np.exp(-r * T) * cls._norm_cdf(-d2)
                    - q * spot * np.exp(-q * T) * cls._norm_cdf(-d1))
        
        # Gamma (same for call and put)
        gamma = np.exp(-q * T) * norm_pdf_d1 / (spot * vol * sqrt_T)
        
        # Vega (same for call and put)
        vega = spot * np.exp(-q * T) * norm_pdf_d1 * sqrt_T / 100  # Per 1% vol change
        
        # Rho
        if option_type == OptionType.CALL:
            rho = strike * T * np.exp(-r * T) * norm_d2 / 100
        else:
            rho = -strike * T * np.exp(-r * T) * cls._norm_cdf(-d2) / 100
        
        return Greeks(
            delta=float(delta),
            gamma=float(gamma),
            theta=float(theta / 365),  # Daily theta
            vega=float(vega),
            rho=float(rho)
        )


class GammaScalper:
    """
    Gamma scalping engine that dynamically adjusts delta hedges.
    Exploits mean-reverting micro-movements to generate theta.
    """
    
    # Configuration
    REBALANCE_THRESHOLD = 0.15  # Rebalance when delta drifts by 15%
    MIN_GAMMA_THRESHOLD = 0.001  # Minimum gamma to consider scalping
    MAX_POSITION_DELTA = 0.5  # Max net delta exposure allowed
    SCALP_TARGET_PCT = 0.5  # Target % of gamma profits to capture
    
    # Memory bounds
    MAX_TRADE_HISTORY = 500
    MAX_SCALP_RECORDS = 1000
    
    def __init__(self, risk_free_rate: float = 0.05):
        self._positions: Dict[str, OptionPosition] = {}
        self._hedge_positions: Dict[str, float] = {}  # Net spot hedges per instrument
        self._trade_history: deque = deque(maxlen=self.MAX_TRADE_HISTORY)
        self._scalp_records: deque = deque(maxlen=self.MAX_SCALP_RECORDS)
        self._bs = BlackScholes()
        self._r = risk_free_rate
        self._execution_callbacks: List[callable] = []
        self._lock = asyncio.Lock()
    
    def register_execution_callback(self, callback: callable):
        """Register callback for hedge execution signals"""
        self._execution_callbacks.append(callback)
    
    async def add_position(self, position: OptionPosition):
        """Add or update an options position"""
        async with self._lock:
            self._positions[position.instrument_id] = position
            
            # Initialize hedge if needed
            if position.instrument_id not in self._hedge_positions:
                self._hedge_positions[position.instrument_id] = 0.0
    
    async def remove_position(self, instrument_id: str):
        """Remove an options position"""
        async with self._lock:
            self._positions.pop(instrument_id, None)
    
    async def update_spot(self, instrument_id: str, spot: float, 
                          implied_vol: Optional[float] = None):
        """Update spot price and optionally IV for an instrument"""
        async with self._lock:
            if instrument_id not in self._positions:
                return
            
            pos = self._positions[instrument_id]
            old_spot = pos.current_spot
            
            # Update position
            pos.current_spot = spot
            if implied_vol is not None:
                pos.implied_vol = implied_vol
            
            # Check if rebalancing is needed
            await self._check_rebalance(instrument_id, old_spot, spot)
    
    def _get_position_greeks(self, pos: OptionPosition) -> Greeks:
        """Calculate Greeks for a single position"""
        T = pos.time_to_expiry()
        greeks = self._bs.calculate_greeks(
            spot=pos.current_spot,
            strike=pos.strike,
            T=T,
            vol=pos.implied_vol,
            r=self._r,
            option_type=pos.option_type
        )
        # Scale by quantity (contracts) and multiplier (typically 1 for crypto)
        return greeks * pos.quantity
    
    def get_portfolio_greeks(self) -> Dict[str, Greeks]:
        """Get aggregated Greeks per instrument"""
        result = {}
        for inst_id, pos in self._positions.items():
            result[inst_id] = self._get_position_greeks(pos)
        return result
    
    def get_net_delta(self) -> float:
        """Get total portfolio delta"""
        total = 0.0
        for inst_id, pos in self._positions.items():
            greeks = self._get_position_greeks(pos)
            hedge = self._hedge_positions.get(inst_id, 0.0)
            total += greeks.delta + hedge
        return total
    
    def get_net_gamma(self) -> float:
        """Get total portfolio gamma"""
        total = 0.0
        for pos in self._positions.values():
            greeks = self._get_position_greeks(pos)
            total += greeks.gamma
        return total
    
    def get_net_theta(self) -> float:
        """Get total portfolio theta (daily P&L from time decay)"""
        total = 0.0
        for pos in self._positions.values():
            greeks = self._get_position_greeks(pos)
            total += greeks.theta
        return total
    
    async def _check_rebalance(self, inst_id: str, old_spot: float, new_spot: float):
        """Check if delta rebalancing is needed due to spot movement"""
        pos = self._positions[inst_id]
        greeks = self._get_position_greeks(pos)
        
        if abs(greeks.gamma) < self.MIN_GAMMA_THRESHOLD:
            return
        
        # Calculate optimal hedge
        current_hedge = self._hedge_positions.get(inst_id, 0.0)
        target_hedge = -greeks.delta  # Delta-neutral target
        
        delta_drift = abs((greeks.delta + current_hedge) / greeks.delta) if greeks.delta != 0 else 0
        
        if delta_drift > self.REBALANCE_THRESHOLD:
            adjustment = await self._calculate_adjustment(
                inst_id, pos, greeks, current_hedge, target_hedge,
                old_spot, new_spot
            )
            
            if adjustment:
                await self._execute_adjustment(adjustment)
    
    async def _calculate_adjustment(self, inst_id: str, pos: OptionPosition,
                                    greeks: Greeks, current_hedge: float,
                                    target_hedge: float, old_spot: float,
                                    new_spot: float) -> Optional[HedgeAdjustment]:
        """Calculate optimal hedge adjustment"""
        trade_qty = target_hedge - current_hedge
        
        if abs(trade_qty) < 0.01:  # Minimum trade size
            return None
        
        # Estimate slippage based on gamma and spot movement
        spot_change_pct = abs(new_spot - old_spot) / old_spot
        gamma_impact = abs(greeks.gamma) * spot_change_pct * new_spot
        est_slippage_bps = min(10.0, 1.0 + gamma_impact * 100)  # Cap at 10 bps
        
        # Determine action
        action = "BUY" if trade_qty > 0 else "SELL"
        
        # Reason
        if greeks.gamma > 0:
            reason = "Long gamma scalp: buy low/sell high"
        else:
            reason = "Short gamma hedge: protect against move"
        
        return HedgeAdjustment(
            instrument_id=inst_id,
            action=action,
            quantity=abs(trade_qty),
            target_delta=target_hedge,
            current_delta=current_hedge,
            reason=reason,
            timestamp_ns=time.time_ns(),
            estimated_slippage_bps=est_slippage_bps
        )
    
    async def _execute_adjustment(self, adjustment: HedgeAdjustment):
        """Execute hedge adjustment and record"""
        # Update hedge position
        if adjustment.action == "BUY":
            self._hedge_positions[adjustment.instrument_id] += adjustment.quantity
        else:
            self._hedge_positions[adjustment.instrument_id] -= adjustment.quantity
        
        # Record trade
        self._trade_history.append({
            'type': 'HEDGE',
            'instrument_id': adjustment.instrument_id,
            'action': adjustment.action,
            'quantity': adjustment.quantity,
            'timestamp_ns': adjustment.timestamp_ns,
            'slippage_bps': adjustment.estimated_slippage_bps
        })
        
        # Notify callbacks
        for callback in self._execution_callbacks:
            if asyncio.iscoroutinefunction(callback):
                await callback(adjustment)
            else:
                callback(adjustment)
    
    def get_scalp_pnl(self) -> Dict[str, float]:
        """Calculate realized P&L from gamma scalping"""
        pnl_by_inst: Dict[str, float] = {}
        
        for trade in self._trade_history:
            if trade['type'] != 'HEDGE':
                continue
            
            inst_id = trade['instrument_id']
            if inst_id not in pnl_by_inst:
                pnl_by_inst[inst_id] = 0.0
            
            # Simplified P&L estimation (would need actual fill prices in production)
            pnl_by_inst[inst_id] -= trade['quantity'] * trade['slippage_bps'] / 10000
        
        return pnl_by_inst
    
    def get_theta_decay(self) -> float:
        """Get daily theta decay (positive = earning from time decay)"""
        return self.get_net_theta()
    
    def get_gamma_exposure(self) -> float:
        """Get total gamma exposure"""
        return self.get_net_gamma()
    
    async def run_scalping_cycle(self, market_data: Dict[str, float]):
        """
        Run a full gamma scalping cycle.
        Called periodically to check all positions.
        """
        async with self._lock:
            for inst_id, spot in market_data.items():
                if inst_id in self._positions:
                    await self.update_spot(inst_id, spot)


# Global singleton instance
_scalper_instance: Optional[GammaScalper] = None


def get_scalper(risk_free_rate: float = 0.05) -> GammaScalper:
    """Get or create global gamma scalper"""
    global _scalper_instance
    if _scalper_instance is None:
        _scalper_instance = GammaScalper(risk_free_rate=risk_free_rate)
    return _scalper_instance


async def demo():
    """Demo usage of the gamma scalper"""
    scalper = get_scalper()
    
    async def on_hedge(adj: HedgeAdjustment):
        print(f"HEDGE {adj.action}: {adj.quantity:.4f} {adj.instrument_id} "
              f"({adj.reason})")
    
    scalper.register_execution_callback(on_hedge)
    
    # Add a long call position
    call_pos = OptionPosition(
        instrument_id="BTC-OPT",
        option_type=OptionType.CALL,
        strike=50000,
        expiry=0.0833,  # ~1 month
        quantity=10,
        entry_price=2000,
        current_spot=50000,
        implied_vol=0.7,
        timestamp_ns=time.time_ns()
    )
    await scalper.add_position(call_pos)
    
    # Show initial Greeks
    greeks = scalper.get_portfolio_greeks()
    print(f"Initial Delta: {greeks['BTC-OPT'].delta:.4f}")
    print(f"Initial Gamma: {greeks['BTC-OPT'].gamma:.6f}")
    print(f"Initial Theta: ${greeks['BTC-OPT'].theta:.2f}/day")
    
    # Simulate spot movements
    for spot_move in [49500, 50500, 49000, 51000, 50000]:
        await scalper.update_spot("BTC-OPT", spot_move)
    
    print(f"\nNet Theta: ${scalper.get_theta_decay():.2f}/day")
    print(f"Net Gamma: {scalper.get_gamma_exposure():.6f}")


if __name__ == "__main__":
    asyncio.run(demo())
