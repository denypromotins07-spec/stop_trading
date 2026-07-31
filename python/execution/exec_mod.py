"""
Execution Module Root.
Feeds slippage and impact predictions directly into RL execution agents and TWAP/VWAP schedulers.
Integrates with Nautilus execution client for optimal order routing.
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import json
import time


from .slippage_model import SlippageModel, SlippageCalibrator
from .market_impact import AlmgrenChrissModel, ImpactCalibrator, MarketImpactParams


@dataclass
class ExecutionConfig:
    """Configuration for execution system."""
    instruments: List[str] = field(default_factory=list)
    default_urgency: float = 0.5
    max_participation_rate: float = 0.1
    min_slice_size: float = 1000
    max_slices: int = 48
    enable_adaptive_scheduling: bool = True
    slippage_buffer_bps: float = 2.0


@dataclass
class ExecutionCommand:
    """Nautilus-compatible execution command."""
    instrument_id: str
    side: str
    quantity: float
    order_type: str  # "twap", "vwap", "aggressive", "passive"
    urgency: float
    limit_offset_bps: float
    n_slices: int
    slice_interval_seconds: int
    max_participation_rate: float
    metadata: Dict = field(default_factory=dict)


class ExecutionEngine:
    """
    Main execution engine coordinating slippage prediction and market impact models.
    Generates optimal execution schedules for Nautilus strategies.
    """
    
    def __init__(self, config: ExecutionConfig):
        self.config = config
        
        # Initialize models
        self.slippage_calibrator = SlippageCalibrator(config.instruments)
        self.impact_calibrator = ImpactCalibrator(config.instruments)
        
        # Active orders tracking
        self._active_orders: Dict[str, Dict] = {}
        self._order_counter: int = 0
        
        # Execution queue
        self._pending_commands: List[ExecutionCommand] = []
        
        # Market data cache
        self._market_data: Dict[str, Dict] = {inst: {} for inst in config.instruments}
    
    def update_market_data(self, instrument_id: str, data: Dict):
        """Update cached market data for an instrument."""
        if instrument_id in self._market_data:
            self._market_data[instrument_id].update(data)
    
    def create_execution_plan(self, instrument_id: str, side: str,
                               total_quantity: float,
                               urgency: float = None,
                               strategy: str = "adaptive") -> ExecutionCommand:
        """
        Create optimal execution plan for an order.
        
        Args:
            instrument_id: Asset identifier
            side: "buy" or "sell"
            total_quantity: Total order quantity
            urgency: Urgency factor (0-1), uses default if None
            strategy: Execution strategy type
            
        Returns:
            Execution command for Nautilus
        """
        urgency = urgency if urgency is not None else self.config.default_urgency
        
        # Get market data
        market_data = self._market_data.get(instrument_id, {})
        daily_volume = market_data.get("daily_volume", 1e9)
        volatility = market_data.get("volatility", 0.6)
        spread_bps = market_data.get("spread_bps", 5)
        atr = market_data.get("atr", 0.02)
        l2_depth = market_data.get("l2_depth", {})
        recent_returns = market_data.get("recent_returns", np.zeros(10))
        
        # Get slippage prediction
        slippage_model = self.slippage_calibrator.models.get(instrument_id)
        if slippage_model:
            slippage_pred = slippage_model.get_optimal_limit_offset(
                order_size=total_quantity,
                avg_volume=daily_volume,
                atr=atr,
                spread_bps=spread_bps,
                l2_depth=l2_depth,
                recent_returns=recent_returns,
                side=side
            )
            limit_offset = slippage_pred["recommended_offset_bps"] + self.config.slippage_buffer_bps
        else:
            limit_offset = spread_bps + self.config.slippage_buffer_bps
        
        # Get impact-based schedule
        impact_strategy = self.impact_calibrator.get_optimal_execution_strategy(
            instrument_id=instrument_id,
            order_size=total_quantity,
            daily_volume=daily_volume,
            volatility=volatility,
            urgency=urgency
        )
        
        # Determine number of slices
        if strategy == "twap":
            n_slices = max(4, int(impact_strategy.get("n_slices", 12)))
        elif strategy == "vwap":
            n_slices = max(8, int(impact_strategy.get("n_slices", 24)))
        else:  # adaptive
            base_slices = impact_strategy.get("n_slices", 12)
            if urgency > 0.7:
                n_slices = max(4, int(base_slices * 0.5))
            elif urgency < 0.3:
                n_slices = min(self.config.max_slices, int(base_slices * 1.5))
            else:
                n_slices = base_slices
        
        # Calculate slice size
        slice_size = total_quantity / n_slices
        slice_size = max(slice_size, self.config.min_slice_size)
        
        # Adjust n_slices based on minimum slice size
        n_slices = int(np.ceil(total_quantity / slice_size))
        
        # Calculate interval between slices
        duration_hours = impact_strategy.get("estimated_duration_hours", 1.0)
        slice_interval = int((duration_hours * 3600) / n_slices)
        slice_interval = max(slice_interval, 60)  # Minimum 1 minute
        
        # Participation rate limit
        participation_rate = min(
            impact_strategy.get("participation_rate", 0.05),
            self.config.max_participation_rate
        )
        
        # Create command
        self._order_counter += 1
        order_id = f"exec_{self._order_counter}"
        
        cmd = ExecutionCommand(
            instrument_id=instrument_id,
            side=side,
            quantity=total_quantity,
            order_type=strategy,
            urgency=urgency,
            limit_offset_bps=limit_offset,
            n_slices=n_slices,
            slice_interval_seconds=slice_interval,
            max_participation_rate=participation_rate,
            metadata={
                "order_id": order_id,
                "expected_impact_bps": impact_strategy.get("expected_total_impact_bps", 0),
                "expected_slippage_bps": slippage_pred.get("predicted_slippage_bps", 0) if slippage_model else 0,
                "created_at": int(time.time() * 1e9)
            }
        )
        
        # Track active order
        self._active_orders[order_id] = {
            "command": cmd,
            "remaining_quantity": total_quantity,
            "filled_quantity": 0,
            "slices_executed": 0,
            "status": "pending"
        }
        
        return cmd
    
    def submit_command(self, cmd: ExecutionCommand) -> str:
        """Submit execution command to queue."""
        self._pending_commands.append(cmd)
        return cmd.metadata.get("order_id", "unknown")
    
    def get_pending_commands(self) -> List[Dict]:
        """Get and clear pending commands as dictionaries."""
        commands = []
        for cmd in self._pending_commands:
            commands.append({
                "type": "execution_order",
                "instrument_id": cmd.instrument_id,
                "side": cmd.side,
                "quantity": cmd.quantity,
                "order_type": cmd.order_type,
                "urgency": cmd.urgency,
                "limit_offset_bps": cmd.limit_offset_bps,
                "n_slices": cmd.n_slices,
                "slice_interval_seconds": cmd.slice_interval_seconds,
                "max_participation_rate": cmd.max_participation_rate,
                "metadata": cmd.metadata
            })
        
        self._pending_commands.clear()
        return commands
    
    def record_fill(self, order_id: str, fill_price: float, 
                    fill_quantity: float, fill_cost: float = 0):
        """Record a fill for an active order."""
        if order_id not in self._active_orders:
            return
        
        order = self._active_orders[order_id]
        cmd = order["command"]
        
        # Update fill tracking
        order["filled_quantity"] += fill_quantity
        order["remaining_quantity"] -= fill_quantity
        order["slices_executed"] += 1
        
        # Calculate realized slippage
        submission_time = cmd.metadata.get("created_at", 0)
        # In production, get mid price at submission time
        estimated_mid = fill_price  # Placeholder
        
        if cmd.side == "buy":
            slippage_bps = (fill_price - estimated_mid) / (estimated_mid + 1e-10) * 10000
        else:
            slippage_bps = (estimated_mid - fill_price) / (estimated_mid + 1e-10) * 10000
        
        # Update slippage model
        market_data = self._market_data.get(cmd.instrument_id, {})
        self.slippage_calibrator.record_order_execution(
            order_id=f"{order_id}_{order['slices_executed']}",
            execution_price=fill_price,
            filled_size=fill_quantity
        )
        
        # Check if order complete
        if order["remaining_quantity"] <= 0 or order["slices_executed"] >= cmd.n_slices:
            order["status"] = "completed"
    
    def get_order_status(self, order_id: str) -> Dict:
        """Get status of an active order."""
        if order_id not in self._active_orders:
            return {"status": "not_found"}
        
        order = self._active_orders[order_id]
        cmd = order["command"]
        
        fill_pct = order["filled_quantity"] / cmd.quantity if cmd.quantity > 0 else 0
        
        return {
            "order_id": order_id,
            "status": order["status"],
            "total_quantity": cmd.quantity,
            "filled_quantity": order["filled_quantity"],
            "remaining_quantity": order["remaining_quantity"],
            "fill_percentage": fill_pct,
            "slices_executed": order["slices_executed"],
            "total_slices": cmd.n_slices
        }
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an active order."""
        if order_id not in self._active_orders:
            return False
        
        self._active_orders[order_id]["status"] = "cancelled"
        return True
    
    def get_execution_analytics(self, instrument_id: str) -> Dict:
        """Get execution analytics for an instrument."""
        slippage_stats = self.slippage_calibrator.get_model_stats(instrument_id)
        impact_stats = self.impact_calibrator.models.get(instrument_id, {}).get_calibration_stats()
        
        return {
            "instrument_id": instrument_id,
            "slippage_model": slippage_stats,
            "impact_model": impact_stats or {},
            "active_orders": len([o for o in self._active_orders.values() 
                                  if o["command"].instrument_id == instrument_id 
                                  and o["status"] == "pending"])
        }


def create_execution_engine(instruments: List[str],
                            default_urgency: float = 0.5) -> ExecutionEngine:
    """Factory function to create configured execution engine."""
    config = ExecutionConfig(
        instruments=instruments,
        default_urgency=default_urgency
    )
    return ExecutionEngine(config)


if __name__ == "__main__":
    # Example usage
    instruments = ["BTC", "ETH", "SOL"]
    
    engine = create_execution_engine(instruments)
    
    # Update market data
    np.random.seed(42)
    for inst in instruments:
        engine.update_market_data(inst, {
            "daily_volume": np.random.lognormal(18, 0.5),
            "volatility": np.random.uniform(0.4, 0.8),
            "spread_bps": np.random.exponential(5),
            "atr": np.random.uniform(0.015, 0.03),
            "l2_depth": {
                "bid_volume": np.random.lognormal(16, 0.5),
                "ask_volume": np.random.lognormal(16, 0.5),
                "total_depth": np.random.lognormal(17, 0.5)
            },
            "recent_returns": np.random.randn(10) * 0.001
        })
    
    # Create execution plans
    print("Creating Execution Plans:\n")
    
    for inst in instruments:
        cmd = engine.create_execution_plan(
            instrument_id=inst,
            side="buy",
            total_quantity=np.random.uniform(1e5, 1e6),
            urgency=np.random.uniform(0.3, 0.7),
            strategy="adaptive"
        )
        
        print(f"{inst}:")
        print(f"  Quantity: ${cmd.quantity:,.0f}")
        print(f"  Strategy: {cmd.order_type}")
        print(f"  Slices: {cmd.n_slices}")
        print(f"  Limit Offset: {cmd.limit_offset_bps:.2f} bps")
        print(f"  Interval: {cmd.slice_interval_seconds}s")
        print()
        
        # Submit command
        engine.submit_command(cmd)
    
    # Get pending commands
    commands = engine.get_pending_commands()
    print(f"Pending Commands: {len(commands)}")
    
    # Simulate fills
    for cmd in commands[:2]:
        order_id = cmd["metadata"]["order_id"]
        engine.record_fill(order_id, fill_price=50000, fill_quantity=cmd["quantity"] / cmd["n_slices"])
        status = engine.get_order_status(order_id)
        print(f"\nOrder {order_id}: {status['fill_percentage']:.1%} filled")
    
    # Analytics
    print("\nExecution Analytics:")
    for inst in instruments:
        analytics = engine.get_execution_analytics(inst)
        print(f"  {inst}: {analytics['slippage_model'].get('status', 'no_data')}")
