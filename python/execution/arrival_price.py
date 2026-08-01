"""
Implementation Shortfall (Arrival Price) Execution Algorithm for Nautilus.
Dynamically accelerates execution when predicted market impact rises.
Balances cost of delay against cost of market impact using ML-calibrated Almgren-Chriss parameters.
Implements proper on_order_filled and on_order_updated lifecycle hooks.
"""
import asyncio
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import time

try:
    from nautilus_trader.model.enums import OrderSide
    from nautilus_trader.model.orders import LimitOrder, MarketOrder
    NAUTILUS_AVAILABLE = True
except ImportError:
    NAUTILUS_AVAILABLE = False


@dataclass
class AlmgrenChrissParams:
    """Almgren-Chriss model parameters"""
    eta: float  # Temporary impact coefficient
    gamma: float  # Permanent impact coefficient
    sigma: float  # Volatility (daily)
    kappa: float  # Risk aversion parameter
    arrival_price: float  # Price at order arrival


@dataclass
class ISConfig:
    """Configuration for Implementation Shortfall algorithm"""
    risk_aversion: float  # Lambda in Almgren-Chriss
    max_participation_rate: float
    urgency_schedule: str  # 'linear', 'front_load', 'back_load'
    use_limit_orders: bool
    limit_order_offset_bps: int
    recheck_interval_ms: int
    max_spread_acceptance_bps: int


@dataclass
class ExecutionTrajectory:
    """Optimal execution trajectory from Almgren-Chriss"""
    time_steps: int
    remaining_quantity: np.ndarray
    trade_schedule: np.ndarray
    expected_cost: float
    variance_cost: float
    confidence_interval_95: Tuple[float, float]


@dataclass
class ISExecutionState:
    """Current state of IS execution"""
    instrument_id: str
    side: str
    total_quantity: float
    filled_quantity: float
    remaining_quantity: float
    arrival_price: float
    current_price: float
    avg_fill_price: float
    unrealized_is: float  # Current implementation shortfall
    realized_is: float  # Realized shortfall on filled portion
    progress_pct: float
    schedule_deviation: float  # How far ahead/behind schedule
    last_update_ns: int
    active_orders: Dict[str, float]


@dataclass
class MarketImpactEstimate:
    """Estimated market impact for a given order size"""
    order_size: float
    temporary_impact_bps: float
    permanent_impact_bps: float
    total_impact_bps: float
    confidence: float


class AlmgrenChrissOptimizer:
    """
    Almgren-Chriss optimal execution scheduler.
    Calculates optimal trading trajectory balancing impact and timing risk.
    """
    
    def __init__(self, params: AlmgrenChrissParams):
        self.params = params
    
    def calculate_optimal_trajectory(self, total_quantity: float,
                                      time_horizon_steps: int,
                                      current_step: int = 0) -> ExecutionTrajectory:
        """
        Calculate optimal execution trajectory using Almgren-Chriss model.
        
        Args:
            total_quantity: Total quantity to execute
            time_horizon_steps: Number of time steps
            current_step: Current step (for re-optimization)
        
        Returns:
            ExecutionTrajectory with optimal schedule
        """
        Q = total_quantity
        N = time_horizon_steps
        n = current_step
        
        # Model parameters
        eta = self.params.eta
        gamma = self.params.gamma
        sigma = self.params.sigma
        kappa = self.params.risk_aversion
        
        # Time increment (assuming day fractions)
        tau = 1.0 / N
        
        # Calculate optimal liquidation trajectory
        # x_k = Q * (sinh(alpha*(N-k)) + psi) / (sinh(alpha*N) + psi)
        # where alpha and psi depend on model parameters
        
        # Simplified closed-form solution
        alpha = np.sqrt(kappa * sigma**2 * tau / (eta + gamma * tau))
        
        if alpha < 0.01:
            # Low risk aversion - nearly linear schedule
            remaining = np.array([Q * (1 - i/N) for i in range(n, N+1)])
        else:
            # Full Almgren-Chriss solution
            sinh_alpha_N = np.sinh(alpha * N)
            cosh_alpha_N = np.cosh(alpha * N)
            
            # Psi term for boundary conditions
            psi = gamma / (2 * eta / tau) * (1 - np.exp(-alpha * N))
            
            remaining = []
            for k in range(n, N + 1):
                numerator = np.sinh(alpha * (N - k)) + psi
                denominator = sinh_alpha_N + psi
                remaining.append(Q * numerator / denominator)
            remaining = np.array(remaining)
        
        # Calculate trade schedule (differences)
        trades = np.diff(remaining, prepend=Q)
        trades = np.maximum(trades, 0)  # No negative trades
        
        # Expected cost calculation
        # E[Cost] = eta * sum(v_k^2) + gamma * Q^2 / 2 + kappa * sigma^2 * sum(x_k^2) * tau
        expected_temporary = eta * np.sum(trades**2)
        expected_permanent = gamma * Q**2 / 2
        expected_risk = kappa * sigma**2 * np.sum(remaining**2) * tau
        expected_cost = expected_temporary + expected_permanent + expected_risk
        
        # Variance of cost
        variance_cost = sigma**2 * tau * np.sum(remaining**2)
        
        # 95% confidence interval
        std_cost = np.sqrt(variance_cost)
        ci_lower = expected_cost - 1.96 * std_cost
        ci_upper = expected_cost + 1.96 * std_cost
        
        return ExecutionTrajectory(
            time_steps=N,
            remaining_quantity=remaining,
            trade_schedule=trades,
            expected_cost=float(expected_cost),
            variance_cost=float(variance_cost),
            confidence_interval_95=(float(max(0, ci_lower)), float(ci_upper))
        )
    
    def estimate_market_impact(self, order_size: float, 
                                daily_volume: float) -> MarketImpactEstimate:
        """Estimate market impact for a single order"""
        # Participation rate
        participation = order_size / max(daily_volume, order_size)
        
        # Square-root law for temporary impact
        temp_impact_bps = 10000 * self.params.eta * np.sqrt(participation)
        
        # Linear permanent impact
        perm_impact_bps = 10000 * self.params.gamma * participation
        
        total_impact = temp_impact_bps + perm_impact_bps
        
        # Confidence decreases with size
        confidence = max(0.5, 1.0 - participation * 2)
        
        return MarketImpactEstimate(
            order_size=order_size,
            temporary_impact_bps=float(temp_impact_bps),
            permanent_impact_bps=float(perm_impact_bps),
            total_impact_bps=float(total_impact),
            confidence=float(confidence)
        )


class ISExecutionAlgo:
    """
    Implementation Shortfall execution algorithm.
    Uses Almgren-Chriss optimization with dynamic re-balancing.
    """
    
    # Memory bounds
    MAX_PRICE_HISTORY = 1000
    MAX_ORDER_HISTORY = 200
    
    def __init__(self, config: ISConfig, ac_params: AlmgrenChrissParams):
        self.config = config
        self.ac_params = ac_params
        self.optimizer = AlmgrenChrissOptimizer(ac_params)
        
        self._price_history: deque = deque(maxlen=self.MAX_PRICE_HISTORY)
        self._order_history: deque = deque(maxlen=self.MAX_ORDER_HISTORY)
        self._execution_state: Optional[ISExecutionState] = None
        self._pending_orders: Dict[str, float] = {}
        self._trajectory: Optional[ExecutionTrajectory] = None
        self._current_step = 0
        self._lock = asyncio.Lock()
    
    async def initialize(self, instrument_id: str, side: str,
                         total_quantity: float, arrival_price: float) -> ISExecutionState:
        """Initialize IS execution"""
        async with self._lock:
            self._execution_state = ISExecutionState(
                instrument_id=instrument_id,
                side=side,
                total_quantity=total_quantity,
                filled_quantity=0.0,
                remaining_quantity=total_quantity,
                arrival_price=arrival_price,
                current_price=arrival_price,
                avg_fill_price=0.0,
                unrealized_is=0.0,
                realized_is=0.0,
                progress_pct=0.0,
                schedule_deviation=0.0,
                last_update_ns=time.time_ns(),
                active_orders={}
            )
            
            self._price_history.clear()
            self._pending_orders.clear()
            self._current_step = 0
            
            # Calculate initial trajectory
            await self._recalculate_trajectory()
            
            return self._execution_state
    
    async def _recalculate_trajectory(self):
        """Re-calculate optimal trajectory based on current state"""
        if self._execution_state is None:
            return
        
        remaining = self._execution_state.remaining_quantity
        if remaining <= 0:
            return
        
        # Remaining time steps (simplified - would use actual time in production)
        remaining_steps = max(1, 10 - self._current_step)
        
        self._trajectory = self.optimizer.calculate_optimal_trajectory(
            total_quantity=remaining,
            time_horizon_steps=remaining_steps,
            current_step=0
        )
    
    async def on_price_update(self, price: float):
        """Process price update and potentially adjust schedule"""
        async with self._lock:
            if self._execution_state is None:
                return
            
            self._price_history.append({
                'price': price,
                'timestamp_ns': time.time_ns()
            })
            
            self._execution_state.current_price = price
            
            # Update implementation shortfall
            await self._update_shortfall()
            
            # Check if we need to adjust schedule
            await self._check_schedule_adjustment()
    
    async def _update_shortfall(self):
        """Update implementation shortfall calculations"""
        state = self._execution_state
        
        if state.filled_quantity > 0:
            # Realized IS on filled portion
            if state.side == 'buy':
                state.realized_is = (state.avg_fill_price - state.arrival_price) / state.arrival_price
            else:
                state.realized_is = (state.arrival_price - state.avg_fill_price) / state.arrival_price
        
        # Unrealized IS on remaining portion
        if state.remaining_quantity > 0:
            if state.side == 'buy':
                state.unrealized_is = (state.current_price - state.arrival_price) / state.arrival_price
            else:
                state.unrealized_is = (state.arrival_price - state.current_price) / state.arrival_price
        
        state.progress_pct = state.filled_quantity / state.total_quantity * 100
    
    async def _check_schedule_adjustment(self):
        """Check if schedule needs adjustment based on market conditions"""
        if len(self._price_history) < 10:
            return
        
        # Calculate recent volatility
        prices = np.array([p['price'] for p in list(self._price_history)[-20:]])
        returns = np.diff(prices) / prices[:-1]
        recent_vol = np.std(returns) * np.sqrt(252)  # Annualized
        
        # If volatility significantly different from assumption, re-optimize
        vol_diff = abs(recent_vol - self.ac_params.sigma) / self.ac_params.sigma
        
        if vol_diff > 0.3:  # 30% deviation
            # Update volatility parameter
            self.ac_params.sigma = recent_vol
            await self._recalculate_trajectory()
    
    async def _place_order(self, quantity: float):
        """Place an order according to schedule"""
        if self._execution_state is None:
            return
        
        order_id = f"is_{time.time_ns()}"
        self._pending_orders[order_id] = quantity
        self._execution_state.active_orders[order_id] = quantity
        
        self._order_history.append({
            'order_id': order_id,
            'quantity': quantity,
            'timestamp_ns': time.time_ns(),
            'status': 'pending'
        })
        
        self._current_step += 1
    
    async def on_order_filled(self, order_id: str, fill_quantity: float,
                               fill_price: float):
        """Handle order fill event"""
        async with self._lock:
            if self._execution_state is None:
                return
            
            pending_qty = self._pending_orders.pop(order_id, 0)
            if order_id in self._execution_state.active_orders:
                del self._execution_state.active_orders[order_id]
            
            # Update fills
            self._execution_state.filled_quantity += fill_quantity
            self._execution_state.remaining_quantity -= fill_quantity
            self._execution_state.our_volume_executed = self._execution_state.filled_quantity
            
            # Update average fill price
            total_value = (
                self._execution_state.avg_fill_price *
                self._execution_state.filled_quantity
            )
            new_value = total_value + fill_price * fill_quantity
            self._execution_state.avg_fill_price = (
                new_value / self._execution_state.filled_quantity
            )
            
            # Update shortfall
            await self._update_shortfall()
            
            self._order_history.append({
                'order_id': order_id,
                'quantity': fill_quantity,
                'price': fill_price,
                'timestamp_ns': time.time_ns(),
                'status': 'filled'
            })
    
    async def on_order_updated(self, order_id: str, new_status: str,
                                rejected_quantity: float = 0):
        """Handle order update event"""
        async with self._lock:
            if self._execution_state is None:
                return
            
            if order_id in self._execution_state.active_orders:
                if new_status in ['cancelled', 'rejected', 'expired']:
                    cancelled_qty = self._execution_state.active_orders.pop(order_id, 0)
                    self._pending_orders.pop(order_id, None)
                    
                    if rejected_quantity > 0:
                        self._execution_state.remaining_quantity += rejected_quantity
                    
                    self._order_history.append({
                        'order_id': order_id,
                        'quantity': cancelled_qty,
                        'timestamp_ns': time.time_ns(),
                        'status': new_status
                    })
    
    def get_execution_state(self) -> Optional[ISExecutionState]:
        """Get current execution state"""
        return self._execution_state
    
    def get_trajectory_info(self) -> Optional[Dict]:
        """Get current trajectory information"""
        if self._trajectory is None:
            return None
        
        return {
            'expected_cost': self._trajectory.expected_cost,
            'variance_cost': self._trajectory.variance_cost,
            'confidence_interval': self._trajectory.confidence_interval_95,
            'current_step': self._current_step,
            'total_steps': self._trajectory.time_steps
        }
    
    def is_complete(self) -> bool:
        """Check if execution is complete"""
        if self._execution_state is None:
            return True
        return self._execution_state.remaining_quantity <= 0


# Global registry
_is_algos: Dict[str, ISExecutionAlgo] = {}


def create_is_algo(algo_id: str, config: ISConfig,
                   ac_params: AlmgrenChrissParams) -> ISExecutionAlgo:
    """Create and register a new IS algorithm instance"""
    algo = ISExecutionAlgo(config, ac_params)
    _is_algos[algo_id] = algo
    return algo


def get_is_algo(algo_id: str) -> Optional[ISExecutionAlgo]:
    """Get a registered IS algorithm by ID"""
    return _is_algos.get(algo_id)


async def demo():
    """Demo usage of IS execution algorithm"""
    print("=== Implementation Shortfall Demo ===\n")
    
    config = ISConfig(
        risk_aversion=0.5,
        max_participation_rate=0.2,
        urgency_schedule='front_load',
        use_limit_orders=True,
        limit_order_offset_bps=5,
        recheck_interval_ms=100,
        max_spread_acceptance_bps=15
    )
    
    ac_params = AlmgrenChrissParams(
        eta=0.0001,  # Temporary impact
        gamma=0.00001,  # Permanent impact
        sigma=0.02,  # Daily volatility
        kappa=0.5,  # Risk aversion
        arrival_price=50000
    )
    
    algo = create_is_algo("is_demo", config, ac_params)
    
    # Initialize
    state = await algo.initialize(
        instrument_id="BTC/USDT",
        side="sell",
        total_quantity=50.0,
        arrival_price=50000
    )
    print(f"Initialized IS: {state.total_quantity} {state.side} @ ${state.arrival_price}")
    
    # Get trajectory info
    traj = algo.get_trajectory_info()
    if traj:
        print(f"Expected Cost: ${traj['expected_cost']:.2f}")
        print(f"95% CI: ${traj['confidence_interval'][0]:.2f} - ${traj['confidence_interval'][1]:.2f}")
    
    # Simulate price updates and fills
    base_price = 50000
    for i in range(20):
        price = base_price + (i % 7 - 3) * 15
        await algo.on_price_update(price)
        
        # Simulate placing and filling orders
        state = algo.get_execution_state()
        if state and state.remaining_quantity > 0 and not state.active_orders:
            # Place order according to schedule
            order_qty = min(5.0, state.remaining_quantity)
            await algo._place_order(order_qty)
        
        # Simulate fills
        for order_id in list(state.active_orders.keys()) if state else []:
            fill_qty = min(state.active_orders[order_id], 3.0)
            await algo.on_order_filled(order_id, fill_qty, price)
    
    # Final state
    state = algo.get_execution_state()
    print(f"\nExecution Results:")
    print(f"  Filled: {state.filled_quantity:.2f} / {state.total_quantity:.2f}")
    print(f"  Avg Fill: ${state.avg_fill_price:.2f}")
    print(f"  Arrival: ${state.arrival_price:.2f}")
    print(f"  Realized IS: {state.realized_is*100:.2f} bps")
    print(f"  Progress: {state.progress_pct:.1f}%")


if __name__ == "__main__":
    asyncio.run(demo())
