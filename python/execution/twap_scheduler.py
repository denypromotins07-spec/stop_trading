"""
Chapter 2: Advanced Execution Orchestration & Smart Scheduling
File: python/execution/twap_scheduler.py

Implements an RL-driven TWAP scheduler that dynamically adjusts execution pace
based on real-time VPIN and spread volatility. Pauses execution automatically
when toxicity predictor flags imminent adverse selection.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExecutionState(Enum):
    """TWAP execution states."""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ABORTED = "aborted"


@dataclass
class TWAPConfig:
    """Configuration for RL-driven TWAP scheduler."""
    # Basic TWAP parameters
    total_quantity: float = 1000.0
    duration_minutes: int = 60
    min_slice_size: float = 1.0
    max_slice_size: float = 100.0
    
    # RL adaptation parameters
    base_participation_rate: float = 0.1  # 10% of volume
    max_participation_rate: float = 0.3   # 30% cap
    adaptation_lr: float = 0.01
    
    # Toxicity thresholds
    vpin_threshold: float = 0.7  # Pause if VPIN > 0.7
    spread_threshold_bps: float = 50.0  # Pause if spread > 50bps
    volatility_threshold: float = 0.02  # Pause if vol > 2%
    
    # Safety limits
    max_slippage_bps: float = 20.0
    timeout_seconds: int = 7200  # 2 hour max runtime


@dataclass
class MarketCondition:
    """Current market conditions for RL decision making."""
    vpin: float = 0.0
    spread_bps: float = 0.0
    volatility: float = 0.0
    volume_profile: float = 0.0
    order_book_imbalance: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def is_toxic(self, config: TWAPConfig) -> bool:
        """Check if current conditions are toxic for execution."""
        return (
            self.vpin > config.vpin_threshold or
            self.spread_bps > config.spread_threshold_bps or
            self.volatility > config.volatility_threshold
        )


@dataclass
class ExecutionSlice:
    """Represents a single TWAP execution slice."""
    slice_id: int
    scheduled_time: datetime
    quantity: float
    executed_quantity: float = 0.0
    avg_price: float = 0.0
    status: str = "pending"
    slippage_bps: float = 0.0


class RLDrivenTWAPScheduler:
    """
    RL-driven TWAP scheduler with dynamic adaptation.
    Adjusts execution pace based on market conditions and toxicity signals.
    """
    
    def __init__(self, config: Optional[TWAPConfig] = None):
        self.config = config or TWAPConfig()
        self.state = ExecutionState.IDLE
        
        # Execution tracking
        self.slices: List[ExecutionSlice] = []
        self.current_slice_idx = 0
        self.total_executed = 0.0
        self.total_value = 0.0
        
        # RL policy weights (simple linear policy)
        self.policy_weights = np.zeros(5)  # One weight per feature
        
        # Timing
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.last_execution_time: Optional[datetime] = None
        
        # Callbacks
        self.toxicity_callback: Optional[callable] = None
        self.execution_callback: Optional[callable] = None
    
    def initialize(
        self,
        total_quantity: float,
        duration_minutes: int,
        start_time: Optional[datetime] = None
    ):
        """Initialize TWAP schedule."""
        self.config.total_quantity = total_quantity
        self.config.duration_minutes = duration_minutes
        self.start_time = start_time or datetime.utcnow()
        self.end_time = self.start_time + timedelta(minutes=duration_minutes)
        
        # Calculate number of slices (aim for ~1 minute intervals)
        n_slices = max(1, duration_minutes)
        slice_quantity = total_quantity / n_slices
        
        # Ensure slice sizes within bounds
        slice_quantity = np.clip(
            slice_quantity,
            self.config.min_slice_size,
            self.config.max_slice_size
        )
        
        # Generate schedule
        self.slices = []
        for i in range(n_slices):
            scheduled_time = self.start_time + timedelta(minutes=i)
            self.slices.append(ExecutionSlice(
                slice_id=i,
                scheduled_time=scheduled_time,
                quantity=slice_quantity
            ))
        
        self.current_slice_idx = 0
        self.total_executed = 0.0
        self.total_value = 0.0
        self.state = ExecutionState.IDLE
        
        logger.info(
            f"TWAP initialized: {total_quantity} over {duration_minutes}min "
            f"({n_slices} slices)"
        )
    
    def update_policy_weights(self, weights: np.ndarray):
        """Update RL policy weights from training."""
        if len(weights) == len(self.policy_weights):
            self.policy_weights = weights.copy()
            logger.info(f"Policy weights updated: {weights}")
    
    def compute_participation_rate(self, market_condition: MarketCondition) -> float:
        """
        Compute optimal participation rate using RL policy.
        Returns rate between 0 (pause) and max_participation_rate.
        """
        # Feature vector
        features = np.array([
            market_condition.vpin,
            market_condition.spread_bps / 100.0,  # Normalize
            market_condition.volatility,
            market_condition.volume_profile,
            market_condition.order_book_imbalance
        ])
        
        # Linear policy with softmax-like activation
        score = np.dot(features, self.policy_weights)
        
        # Convert score to participation rate
        # Negative score reduces participation, positive increases
        base_rate = self.config.base_participation_rate
        
        if market_condition.is_toxic(self.config):
            # Toxic conditions: reduce or pause
            participation = max(0.0, base_rate * (1 - score))
        else:
            # Normal conditions: adapt based on score
            participation = base_rate * (1 + np.tanh(score))
        
        # Clamp to valid range
        participation = np.clip(
            participation,
            0.0,
            self.config.max_participation_rate
        )
        
        return participation
    
    def should_pause(self, market_condition: MarketCondition) -> Tuple[bool, str]:
        """
        Determine if execution should pause based on market conditions.
        Returns (should_pause, reason).
        """
        reasons = []
        
        if market_condition.vpin > self.config.vpin_threshold:
            reasons.append(f"High VPIN: {market_condition.vpin:.3f}")
        
        if market_condition.spread_bps > self.config.spread_threshold_bps:
            reasons.append(f"Wide spread: {market_condition.spread_bps:.1f}bps")
        
        if market_condition.volatility > self.config.volatility_threshold:
            reasons.append(f"High volatility: {market_condition.volatility:.4f}")
        
        # Check timeout
        if datetime.utcnow() > self.end_time:
            reasons.append("Timeout exceeded")
        
        should_pause = len(reasons) > 0
        
        if should_pause:
            logger.warning(f"Pause triggered: {', '.join(reasons)}")
        
        return should_pause, "; ".join(reasons) if reasons else ""
    
    async def execute_slice(
        self,
        slice_: ExecutionSlice,
        market_condition: MarketCondition
    ) -> ExecutionSlice:
        """Execute a single TWAP slice."""
        # Calculate adaptive quantity based on participation rate
        participation = self.compute_participation_rate(market_condition)
        adaptive_quantity = slice_.quantity * participation
        
        if adaptive_quantity < self.config.min_slice_size:
            # Skip this slice if quantity too small
            slice_.status = "skipped"
            return slice_
        
        # Execute (placeholder - integrate with actual execution engine)
        if self.execution_callback:
            result = await self.execution_callback(
                quantity=adaptive_quantity,
                market_condition=market_condition
            )
            slice_.executed_quantity = result.get("executed_quantity", adaptive_quantity)
            slice_.avg_price = result.get("avg_price", 0.0)
            slice_.slippage_bps = result.get("slippage_bps", 0.0)
        else:
            # Simulation mode
            slice_.executed_quantity = adaptive_quantity
            slice_.avg_price = 0.0
            slice_.slippage_bps = 0.0
        
        slice_.status = "executed"
        self.total_executed += slice_.executed_quantity
        
        logger.info(
            f"Slice {slice_.slice_id}: executed {slice_.executed_quantity} "
            f"@ {slice_.avg_price}"
        )
        
        return slice_
    
    async def run(
        self,
        market_condition_provider: callable,
        check_interval_seconds: int = 5
    ):
        """
        Run the TWAP scheduler with continuous market monitoring.
        
        Args:
            market_condition_provider: Async callable returning MarketCondition
            check_interval_seconds: How often to check market conditions
        """
        if not self.slices:
            raise ValueError("TWAP not initialized. Call initialize() first.")
        
        self.state = ExecutionState.RUNNING
        logger.info("TWAP scheduler started")
        
        while self.current_slice_idx < len(self.slices):
            # Get current market condition
            market_condition = await market_condition_provider()
            
            # Check for toxicity
            should_pause, pause_reason = self.should_pause(market_condition)
            
            if should_pause:
                if self.state != ExecutionState.PAUSED:
                    self.state = ExecutionState.PAUSED
                    logger.warning(f"TWAP paused: {pause_reason}")
                
                # Notify toxicity if callback registered
                if self.toxicity_callback:
                    await self.toxicity_callback(pause_reason)
                
                # Wait before rechecking
                await asyncio.sleep(check_interval_seconds)
                continue
            
            # Resume if previously paused
            if self.state == ExecutionState.PAUSED:
                self.state = ExecutionState.RUNNING
                logger.info("TWAP resumed")
            
            # Check if current slice is due
            current_slice = self.slices[self.current_slice_idx]
            now = datetime.utcnow()
            
            if now >= current_slice.scheduled_time:
                # Execute slice
                await self.execute_slice(current_slice, market_condition)
                self.current_slice_idx += 1
                self.last_execution_time = now
            
            # Wait for next check
            await asyncio.sleep(check_interval_seconds)
        
        # Completion
        self.state = ExecutionState.COMPLETED
        logger.info(
            f"TWAP completed: {self.total_executed}/{self.config.total_quantity} executed"
        )
    
    def get_progress(self) -> Dict[str, Any]:
        """Get current execution progress."""
        completion_pct = (
            self.total_executed / self.config.total_quantity * 100
            if self.config.total_quantity > 0 else 0.0
        )
        
        time_elapsed = (
            (datetime.utcnow() - self.start_time).total_seconds()
            if self.start_time else 0
        )
        time_remaining = (
            (self.end_time - datetime.utcnow()).total_seconds()
            if self.end_time else 0
        )
        
        return {
            "state": self.state.value,
            "completion_pct": completion_pct,
            "total_executed": self.total_executed,
            "total_quantity": self.config.total_quantity,
            "slices_completed": self.current_slice_idx,
            "total_slices": len(self.slices),
            "time_elapsed_sec": time_elapsed,
            "time_remaining_sec": max(0, time_remaining),
            "last_execution": self.last_execution_time.isoformat() if self.last_execution_time else None
        }
    
    def abort(self, reason: str = ""):
        """Abort TWAP execution."""
        self.state = ExecutionState.ABORTED
        logger.warning(f"TWAP aborted: {reason}")
    
    def get_rl_features(self, market_condition: MarketCondition) -> np.ndarray:
        """Extract features for RL policy."""
        return np.array([
            market_condition.vpin,
            market_condition.spread_bps / 100.0,
            market_condition.volatility,
            market_condition.volume_profile,
            market_condition.order_book_imbalance
        ], dtype=np.float32)


# Export for module use
__all__ = [
    "ExecutionState",
    "TWAPConfig",
    "MarketCondition",
    "ExecutionSlice",
    "RLDrivenTWAPScheduler"
]
